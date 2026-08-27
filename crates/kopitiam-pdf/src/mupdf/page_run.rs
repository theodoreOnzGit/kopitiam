//! Ported from MuPDF `source/pdf/pdf-page.c` (`pdf_page_contents`,
//! `pdf_page_resources`, `pdf_page_obj_transform_box`), `source/pdf/pdf-run.c`
//! (`pdf_run_page_contents_with_usage_imp`), and the Form-XObject recursion of
//! `source/pdf/pdf-op-run.c` (`pdf_run_xobject`) + `source/pdf/pdf-xobject.c`
//! (`pdf_xobject_matrix`/`_resources`, the `pdf_cycle` guard) (commit 19f1284,
//! AGPL-3.0, © Artifex Software, Inc.), translated to Rust for KOPITIAM
//! (AGPL-3.0-only). Close adaptation: the algorithms and numeric behaviour follow
//! MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # Driving a page
//!
//! [`run_page`] is the headline WAVE-5 entry point: get a page's `/Contents`
//! (one stream or an array of streams, concatenated with a separating newline as
//! MuPDF's `fz_open_contents_stream` does), its `/Resources`, and the
//! MediaBox-derived base CTM (`pdf_page_obj_transform_box`: flip y, apply page
//! rotation, move the crop origin to `(0, 0)`), then run the interpreter over the
//! bytes.
//!
//! A `Do` operator naming a Form XObject ([`Processor::op_do`]) runs that
//! XObject's own content stream with its `/Matrix` pre-concatenated onto the CTM
//! and its `/Resources` pushed, inside a `q`/`Q` pair, guarded against reference
//! cycles by object number.
//!
//! ## Deferred
//!
//! Form XObjects recurse and `/Image` XObjects are decoded + painted (via
//! [`Processor::draw_image_xobject`], driving the draw device's fill-image
//! callback); PostScript / unknown XObjects are skipped. The XObject `/BBox` clip
//! and transparency-group setup are not modelled -- neither affects glyph
//! positions. `UserUnit` is treated as 1, and only the MediaBox is used for the
//! base CTM (CropBox/ArtBox/etc. box selection is deferred).

use super::geometry::{Matrix, Rect};
use super::interpret::Processor;
use super::object::Object;
use super::text_device::TextDevice;
use super::xref::PdfDocument;

// MuPDF: pdf_run_page_contents_with_usage_imp (pdf-run.c:105).
/// Run the content stream of page `page_index` (0-based) in `doc`, emitting
/// positioned glyphs to `dev`. The base CTM is the MediaBox-derived page
/// transform (device space: origin top-left, y descending, 72 dpi).
///
/// This is the seam the structured-text (`stext`) wave consumes: implement
/// [`TextDevice`] and call this.
pub fn run_page<D: TextDevice + ?Sized>(
    doc: &PdfDocument,
    page_index: usize,
    dev: &mut D,
) -> super::error::Result<()> {
    let page = doc.page(page_index)?.clone();
    run_page_dict(doc, &page, dev)
}

/// Run a page given its (inheritance-flattened) page dict directly.
pub fn run_page_dict<D: TextDevice + ?Sized>(
    doc: &PdfDocument,
    page: &Object,
    dev: &mut D,
) -> super::error::Result<()> {
    let resources = doc.resolve_get(page, "Resources").unwrap_or(Object::Null);
    let ctm = page_ctm(doc, page);
    let contents = gather_contents(doc, page)?;

    let mut proc = Processor::new(doc, dev, ctm, resources);
    proc.run_stream(&contents)
}

// MuPDF: pdf_page_contents (pdf-page.c:730) + fz_open_contents_stream
// (concatenation of multiple content streams with a separating newline).
/// Fetch and concatenate a page's `/Contents` (a single stream or an array of
/// streams). Streams are separated by a newline so an operator split across the
/// stream boundary still lexes.
fn gather_contents(doc: &PdfDocument, page: &Object) -> super::error::Result<Vec<u8>> {
    let contents = match page.dict_gets("Contents") {
        Some(c) => c,
        None => return Ok(Vec::new()),
    };

    let mut out = Vec::new();
    match contents {
        Object::Array(items) => {
            for item in items {
                if let Ok(bytes) = doc.open_stream(item) {
                    out.extend_from_slice(&bytes);
                    out.push(b'\n');
                }
            }
        }
        Object::Ref { .. } => {
            if let Ok(bytes) = doc.open_stream(contents) {
                out.extend_from_slice(&bytes);
            }
        }
        _ => {}
    }
    Ok(out)
}

// MuPDF: pdf_page_obj_transform_box (pdf-page.c:742), MediaBox path, UserUnit=1.
/// Public because the annotation pass needs *the same* base transform the
/// content stream was run under: `/Annots` are positioned in the same default
/// user space as page content, so drawing them under a different CTM would put
/// every annotation in the wrong place. See [`super::annot_run::run_page_annots`].
///
/// The MediaBox-derived base page transform: flip y, apply the (0/90/180/270)
/// page rotation, and translate so the box origin lands at `(0, 0)`.
pub fn page_ctm(doc: &PdfDocument, page: &Object) -> Matrix {
    let mut mediabox =
        rect_from(doc, page, "MediaBox").unwrap_or(Rect::new(0.0, 0.0, 612.0, 792.0));
    // Normalise (x0<=x1, y0<=y1); degenerate boxes fall back to US Letter.
    if mediabox.x1 - mediabox.x0 < 1.0 || mediabox.y1 - mediabox.y0 < 1.0 {
        mediabox = Rect::new(0.0, 0.0, 612.0, 792.0);
    }
    let mediabox = Rect::new(
        mediabox.x0.min(mediabox.x1),
        mediabox.y0.min(mediabox.y1),
        mediabox.x0.max(mediabox.x1),
        mediabox.y0.max(mediabox.y1),
    );

    // Snap /Rotate to a multiple of 90 in [0, 360).
    let mut rotate = doc
        .resolve_get(page, "Rotate")
        .map(|o| o.to_int())
        .unwrap_or(0);
    if rotate < 0 {
        rotate = 360 - ((-rotate) % 360);
    }
    rotate %= 360;
    rotate = 90 * ((rotate + 45) / 90);
    if rotate >= 360 {
        rotate = 0;
    }

    // fitz page space <- PDF user space: left-handed (flip y), then rotate.
    let mut ctm = Matrix::scale(1.0, -1.0);
    ctm = ctm.pre_rotate(-(rotate as f32));

    // Move the (transformed) box origin to (0, 0).
    let box_t = mediabox.transform(ctm);
    ctm = ctm.concat(Matrix::translate(-box_t.x0, -box_t.y0));
    ctm
}

/// Read a 4-element numeric array (`/MediaBox`, …) into a [`Rect`].
fn rect_from(doc: &PdfDocument, dict: &Object, key: &str) -> Option<Rect> {
    let arr = doc.resolve_get(dict, key).ok()?;
    if arr.array_len() < 4 {
        return None;
    }
    let v = |i: usize| -> f32 {
        arr.array_get(i)
            .and_then(|o| doc.resolve(o).ok())
            .map(|o| o.to_real() as f32)
            .unwrap_or(0.0)
    };
    Some(Rect::new(v(0), v(1), v(2), v(3)))
}

impl<D: TextDevice + ?Sized> Processor<'_, D> {
    // MuPDF: pdf_process_Do (pdf-interpret.c:1046) dispatch to op_Do_form ->
    // pdf_run_xobject (pdf-op-run.c:2405). Form XObjects only; images deferred.
    /// Handle `Do`: if `name` resolves to a Form XObject, run its content stream
    /// with its `/Matrix` pre-concatenated onto the CTM and its `/Resources`
    /// pushed, inside a `q`/`Q` pair and guarded against reference cycles.
    pub(crate) fn op_do(&mut self, name: Option<&[u8]>) -> super::error::Result<()> {
        let Some(name) = name else { return Ok(()) };

        let xobj_ref = self.lookup_resource_raw("XObject", name);
        let xobj = self.doc.resolve(&xobj_ref)?;
        if !xobj.is_dict() {
            return Ok(());
        }

        // Form XObjects recurse; Image XObjects are painted (Subtype2 overrides
        // Subtype for a form).
        let subtype = self
            .doc
            .resolve_get(&xobj, "Subtype")
            .unwrap_or(Object::Null);
        let subtype = if xobj.dict_gets("Subtype2").is_some() {
            self.doc.resolve_get(&xobj, "Subtype2").unwrap_or(subtype)
        } else {
            subtype
        };
        if subtype.to_name() == b"Image" {
            self.draw_image_xobject(&xobj, &xobj_ref);
            return Ok(());
        }
        if subtype.to_name() != b"Form" {
            return Ok(()); // /PS, unknown: skip
        }

        // Content bytes come from the stream body (always an indirect stream).
        let content = match &xobj_ref {
            Object::Ref { .. } => match self.doc.open_stream(&xobj_ref) {
                Ok(b) => b,
                Err(_) => return Ok(()),
            },
            _ => return Ok(()),
        };

        // Cycle / depth guard (pdf_cycle).
        let num = xobj_ref.to_num();
        if num != 0 && self.cycle.contains(&num) {
            return Ok(());
        }
        if self.cycle.len() >= 64 {
            return Ok(());
        }

        // /Matrix (default identity) and /Resources (fall back to the current
        // resources -- MuPDF uses the page resources when the form omits its own).
        let matrix = matrix_from(&xobj).unwrap_or(Matrix::IDENTITY);
        let resources = self
            .doc
            .resolve_get(&xobj, "Resources")
            .unwrap_or(Object::Null);

        // gsave; ctm = matrix . ctm; push resources; run; restore.
        self.op_q();
        {
            let g = self.gstate_mut();
            g.ctm = matrix.concat(g.ctm);
        }
        let pushed = resources.is_dict();
        if pushed {
            self.resources.push(resources);
        }
        if num != 0 {
            self.cycle.push(num);
        }

        let result = self.run_stream(&content);

        if num != 0 {
            self.cycle.pop();
        }
        if pushed {
            self.resources.pop();
        }
        self.op_q_restore();

        result
    }

    // MuPDF: pdf_show_image (pdf-op-run.c:1043) -- decode the image and paint it
    // through the fill-image device callback under the current CTM.
    /// Paint an Image XObject (`dict`/`stream_ref`) at the current CTM. The image's
    /// unit square is placed with `[1 0 0 -1 0 1]` folded into the CTM so its top
    /// row lands at the top of the user-space unit square (PDF image space). A
    /// decode failure (deferred codec / colorspace) is a safe skip.
    fn draw_image_xobject(&mut self, dict: &Object, stream_ref: &Object) {
        let img = match super::page_image::decode_image_xobject(self.doc, dict, stream_ref) {
            Ok(img) => img,
            Err(_) => return, // deferred codec / unsupported colorspace: skip
        };
        let (ctm, clip) = {
            let g = self.gstate();
            (g.ctm, g.clip)
        };
        // fitz image space (0,0 top-left, unit square) -> PDF user space unit square
        // (image top row at y=1): flip y, then apply the page/user CTM.
        let image_ctm = Matrix::new(1.0, 0.0, 0.0, -1.0, 0.0, 1.0).concat(ctm);
        self.dev.draw_image(&img, image_ctm, 1.0, clip);
    }

    // MuPDF: pdf_lookup_resource (pdf-interpret.c:33), returning the *raw*
    // (unresolved) value so the caller can open the XObject stream by reference.
    /// Look up a resource without resolving the final value (for XObjects, whose
    /// indirect reference is needed to open the stream body).
    fn lookup_resource_raw(&self, typ: &str, name: &[u8]) -> Object {
        for res in self.resources.iter().rev() {
            if let Ok(sub) = self.doc.resolve_get(res, typ)
                && sub.is_dict()
                && let Some(v) = sub.dict_get(name)
            {
                return v.clone();
            }
        }
        Object::Null
    }
}

// MuPDF: pdf_xobject_matrix (pdf-xobject.c) -- the /Matrix entry, or identity.
/// Read an XObject's `/Matrix` (6 numbers) into a [`Matrix`], or `None` if absent
/// or malformed.
fn matrix_from(xobj: &Object) -> Option<Matrix> {
    let arr = xobj.dict_gets("Matrix")?;
    if arr.array_len() < 6 {
        return None;
    }
    let v = |i: usize| -> f32 { arr.array_get(i).map(|o| o.to_real() as f32).unwrap_or(0.0) };
    Some(Matrix::new(v(0), v(1), v(2), v(3), v(4), v(5)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mupdf::font::Font;
    use crate::mupdf::interpret::Processor;
    use crate::mupdf::text_device::TextDevice;

    /// A recording device: keeps `(unicode, trm.e, trm.f, cid, wmode)` per glyph.
    #[derive(Default)]
    struct Recorder {
        glyphs: Vec<(char, f32, f32, u32, u8)>,
    }

    impl TextDevice for Recorder {
        fn show_glyph(
            &mut self,
            _font: &Font,
            trm: Matrix,
            _adv: f32,
            unicode: char,
            cid: u32,
            wmode: u8,
        ) {
            self.glyphs.push((unicode, trm.e, trm.f, cid, wmode));
        }
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    /// A WinAnsi Helvetica simple font with explicit widths for the codes the
    /// tests use (space=300, A=B=600, H=700, i=300; others 0).
    fn font_dict() -> Object {
        let mut d = Object::new_dict();
        d.dict_put("Type", Object::new_name("Font"));
        d.dict_put("Subtype", Object::new_name("Type1"));
        d.dict_put("BaseFont", Object::new_name("Helvetica"));
        d.dict_put("Encoding", Object::new_name("WinAnsiEncoding"));
        d.dict_put("FirstChar", Object::new_int(32));
        d.dict_put("LastChar", Object::new_int(105));
        let mut widths = Object::new_array();
        for code in 32..=105i64 {
            let w = match code {
                32 => 300,  // space
                65 => 600,  // A
                66 => 600,  // B
                72 => 700,  // H
                105 => 300, // i
                _ => 0,
            };
            widths.array_push(Object::new_int(w));
        }
        d.dict_put("Widths", widths);
        d
    }

    /// A resources dict with the test font under `/Font /F1` (inline direct dict).
    fn resources() -> Object {
        let mut font_sub = Object::new_dict();
        font_sub.dict_put("F1", font_dict());
        let mut res = Object::new_dict();
        res.dict_put("Font", font_sub);
        res
    }

    /// The smallest valid one-page PDF, just so we have a `PdfDocument` to hand
    /// the interpreter (resources are supplied directly, not via this doc).
    fn minimal_doc() -> PdfDocument {
        let bodies: [&[u8]; 3] = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>",
        ];
        let pdf = build_pdf(&bodies);
        PdfDocument::open(pdf).unwrap()
    }

    /// Build a PDF with objects 1.. from `bodies` (each wrapped `N 0 obj … endobj`)
    /// and a classic xref table.
    fn build_pdf(bodies: &[&[u8]]) -> Vec<u8> {
        let mut pdf: Vec<u8> = Vec::new();
        pdf.extend_from_slice(b"%PDF-1.5\n");
        let mut offsets = vec![0usize; bodies.len() + 1];
        for (idx, body) in bodies.iter().enumerate() {
            let num = idx + 1;
            offsets[num] = pdf.len();
            pdf.extend_from_slice(format!("{num} 0 obj\n").as_bytes());
            pdf.extend_from_slice(body);
            pdf.extend_from_slice(b"\nendobj\n");
        }
        let xref_ofs = pdf.len();
        let size = bodies.len() + 1;
        pdf.extend_from_slice(format!("xref\n0 {size}\n").as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for off in offsets.iter().skip(1) {
            pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_ofs}\n%%EOF\n")
                .as_bytes(),
        );
        pdf
    }

    /// Drive the interpreter over `content` with an identity base CTM and the test
    /// resources, returning the recorded glyphs.
    fn run(content: &[u8]) -> Vec<(char, f32, f32, u32, u8)> {
        let doc = minimal_doc();
        let mut dev = Recorder::default();
        {
            let mut proc = Processor::new(&doc, &mut dev, Matrix::IDENTITY, resources());
            proc.run_stream(content).unwrap();
        }
        dev.glyphs
    }

    #[test]
    fn tj_positions_two_glyphs() {
        // Identity CTM: trm.e/f are the glyph origins directly.
        let g = run(b"BT /F1 12 Tf 100 700 Td (Hi) Tj ET");
        assert_eq!(g.len(), 2);
        // 'H' at (100, 700).
        assert_eq!(g[0].0, 'H');
        assert!(approx(g[0].1, 100.0), "H.x = {}", g[0].1);
        assert!(approx(g[0].2, 700.0));
        // 'i' at x = 100 + H_advance*12/1000 = 100 + 700*0.012 = 108.4.
        assert_eq!(g[1].0, 'i');
        assert!(approx(g[1].1, 100.0 + 700.0 * 0.012), "i.x = {}", g[1].1);
        assert!(approx(g[1].2, 700.0));
    }

    #[test]
    fn tj_kerning_number_shifts() {
        // Baseline: no adjustment -> B at A_advance (600*0.012 = 7.2).
        let base = run(b"BT /F1 12 Tf 0 0 Td [(A)(B)] TJ ET");
        assert!(approx(base[1].1, 7.2), "B baseline = {}", base[1].1);

        // A kerning number n adjusts by -n/1000*size*Tz. For n = -50: +0.6 (right,
        // per the PDF sign convention: a positive number moves left). Magnitude
        // 50/1000*12 = 0.6 either way.
        let g = run(b"BT /F1 12 Tf 0 0 Td [(A) -50 (B)] TJ ET");
        assert_eq!(g.len(), 2);
        assert!(approx(g[1].1, 7.2 + 0.6), "B shifted = {}", g[1].1);

        // Positive number shifts left by the same magnitude.
        let g2 = run(b"BT /F1 12 Tf 0 0 Td [(A) 50 (B)] TJ ET");
        assert!(approx(g2[1].1, 7.2 - 0.6), "B(+50) = {}", g2[1].1);
    }

    #[test]
    fn char_spacing_shifts_subsequent_glyphs() {
        // Tc 5: after A, advance = 600*0.012 + 5 = 12.2.
        let g = run(b"BT /F1 12 Tf 0 0 Td 5 Tc (AB) Tj ET");
        assert_eq!(g.len(), 2);
        assert!(approx(g[1].1, 7.2 + 5.0), "B.x = {}", g[1].1);
    }

    #[test]
    fn word_spacing_applies_on_space() {
        // "A B": A at 0, space glyph at 7.2, B at 7.2 + (300*0.012) + Tw(10) = 20.8.
        let g = run(b"BT /F1 12 Tf 0 0 Td 10 Tw (A B) Tj ET");
        assert_eq!(g.len(), 3, "space is emitted as a glyph too");
        assert_eq!(g[1].0, ' ');
        assert!(approx(g[2].1, 7.2 + 3.6 + 10.0), "B.x = {}", g[2].1);
    }

    #[test]
    fn horizontal_scale_halves_advance() {
        // Tz 50 -> scale 0.5: advance halved, and trm.a = size*scale = 6.
        let g = run(b"BT /F1 12 Tf 0 0 Td 50 Tz (AB) Tj ET");
        assert!(approx(g[1].1, 3.6), "B.x = {}", g[1].1);
    }

    #[test]
    fn cm_scales_glyph_positions_and_restores() {
        // Inside q..Q, cm doubles the CTM; the glyph lands at (200, 1400).
        let g = run(b"q 2 0 0 2 0 0 cm BT /F1 12 Tf 100 700 Td (H) Tj ET Q \
                      BT /F1 12 Tf 100 700 Td (i) Tj ET");
        assert_eq!(g.len(), 2);
        assert!(approx(g[0].1, 200.0), "H.x = {}", g[0].1);
        assert!(approx(g[0].2, 1400.0), "H.y = {}", g[0].2);
        // After Q the CTM is restored: the second glyph is back at (100, 700).
        assert!(approx(g[1].1, 100.0), "i.x = {}", g[1].1);
        assert!(approx(g[1].2, 700.0), "i.y = {}", g[1].2);
    }

    #[test]
    fn render_mode_3_still_emits() {
        // Tr 3 (invisible) must STILL emit glyphs -- extraction needs them.
        let g = run(b"BT /F1 12 Tf 100 700 Td 3 Tr (H) Tj ET");
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].0, 'H');
        assert!(approx(g[0].1, 100.0));
    }

    #[test]
    fn run_page_applies_base_ctm() {
        // A full page: MediaBox [0 0 200 200] flips y (origin top-left).
        let font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
/Encoding /WinAnsiEncoding /FirstChar 72 /LastChar 72 /Widths [700] >>";
        let content = b"<< /Length 29 >>\nstream\nBT /F1 12 Tf 0 0 Td (H) Tj ET\nendstream";
        let page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>";
        let bodies: [&[u8]; 5] = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            page,
            content,
            font,
        ];
        let doc = PdfDocument::open(build_pdf(&bodies)).unwrap();
        let mut dev = Recorder::default();
        run_page(&doc, 0, &mut dev).unwrap();
        assert_eq!(dev.glyphs.len(), 1);
        // Base CTM [1 0 0 -1 0 200]: text (0,0) maps to device (0, 200).
        assert!(approx(dev.glyphs[0].1, 0.0), "H.x = {}", dev.glyphs[0].1);
        assert!(approx(dev.glyphs[0].2, 200.0), "H.y = {}", dev.glyphs[0].2);
    }

    #[test]
    fn form_xobject_do_runs_nested_with_matrix() {
        // Page content is just `/Fm0 Do`; the form (obj 5) has /Matrix
        // [1 0 0 1 10 20] and shows 'H' at Td(0,0). Its own /Resources map /F1.
        let font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
/Encoding /WinAnsiEncoding /FirstChar 72 /LastChar 72 /Widths [700] >>";
        let page_content = b"<< /Length 7 >>\nstream\n/Fm0 Do\nendstream";
        let form_stream = b"BT /F1 12 Tf 0 0 Td (H) Tj ET";
        let mut form: Vec<u8> = Vec::new();
        form.extend_from_slice(
            format!(
                "<< /Type /XObject /Subtype /Form /Matrix [1 0 0 1 10 20] /BBox [0 0 100 100] \
/Resources << /Font << /F1 6 0 R >> >> /Length {} >>\nstream\n",
                form_stream.len()
            )
            .as_bytes(),
        );
        form.extend_from_slice(form_stream);
        form.extend_from_slice(b"\nendstream");
        let page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
/Resources << /XObject << /Fm0 5 0 R >> >> /Contents 4 0 R >>";
        let bodies: [&[u8]; 6] = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            page,
            page_content,
            &form,
            font,
        ];
        let doc = PdfDocument::open(build_pdf(&bodies)).unwrap();
        let mut dev = Recorder::default();
        run_page(&doc, 0, &mut dev).unwrap();
        assert_eq!(dev.glyphs.len(), 1, "nested form glyph emitted");
        // ctm [1 0 0 -1 0 200], form matrix adds (10,20) -> device (10, 180).
        assert!(approx(dev.glyphs[0].1, 10.0), "H.x = {}", dev.glyphs[0].1);
        assert!(approx(dev.glyphs[0].2, 180.0), "H.y = {}", dev.glyphs[0].2);
    }
}
