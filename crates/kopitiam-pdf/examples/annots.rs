//! `annots` -- a headless annotation-visibility harness for `kopitiam-pdf`.
//!
//! Run it on a PDF, it will tell you, page by page, what annotations the file
//! carries and whether the rasterizer actually *paints* them. No GUI, no
//! terminal needed -- unlike the `kpdf` example, this one is verifiable in CI
//! and by an agent, which is exactly why it exists.
//!
//! # Why got this example
//!
//! Okular can show the ink annotations in `test-annotation.pdf`, but `kpdf`
//! shows blank. Eyeballing a GUI cannot tell you *which* of the two possible
//! causes it is, so this harness separates them:
//!
//! 1. **The annotation never reaches the renderer at all.** A page's
//!    `/Annots` is a sibling of `/Contents`, not part of it -- rendering only
//!    the content stream draws none of them, no matter how correct the
//!    rasterizer is.
//! 2. **The annotation has no `/AP` (appearance stream).** `/AP` is the
//!    *drawing*; the annot dict is only the *data*. A viewer that renders
//!    strictly from `/AP` draws nothing for an annot that has none, and must
//!    synthesise the appearance from the annot's own geometry
//!    (`/InkList`, `/C`, `/Border`) instead -- see PDF 32000-1:2008 §12.5.5
//!    and §12.5.6.13.
//!
//! The report prints the `/AP` column so you can tell those two apart at a
//! glance, then renders and measures actual painted coverage inside each
//! annotation's `/Rect` so nobody has to trust a screenshot.
//!
//! # Usage
//!
//! ```text
//! cargo run --release -p kopitiam-pdf --example annots -- <file.pdf> [--dpi N] [--dump-dir DIR]
//! ```
//!
//! `--dump-dir` writes one binary PPM (P6) per page with annotations, so you
//! can open the render and look, without this example ever needing an image
//! encoder dependency.

use std::collections::BTreeMap;
use std::path::PathBuf;

use kopitiam_pdf::mupdf::{Object, PdfDocument, Pixmap};

/// Render a page with `hayro` directly, bypassing kopitiam's engine entirely.
///
/// This exists to answer one question with data instead of source-reading:
/// *would simply handing annotated pages to hayro make them visible?* hayro
/// gates every annotation behind `/AP` -> `/N`
/// (`hayro-interpret-0.7.0/src/interpret/mod.rs:157`), so for a file whose
/// annots carry no appearance stream the honest answer is no -- and this
/// function lets the harness demonstrate that rather than assert it.
fn render_hayro(bytes: &[u8], page_index: usize, dpi: f32) -> Option<(u32, u32, Vec<u8>)> {
    use hayro::hayro_interpret::InterpreterSettings;
    use hayro::hayro_syntax::Pdf;
    use hayro::vello_cpu::color::palette::css::WHITE;
    use hayro::{RenderCache, RenderSettings, render};

    let pdf = Pdf::new(bytes.to_vec()).ok()?;
    let page = pdf.pages().get(page_index)?;
    let scale = (dpi / 72.0).max(0.01);
    let settings = RenderSettings {
        x_scale: scale,
        y_scale: scale,
        bg_color: WHITE,
        ..Default::default()
    };
    let vp = render(
        page,
        &RenderCache::new(),
        &InterpreterSettings::default(),
        &settings,
    );
    let (w, h) = (vp.width() as u32, vp.height() as u32);
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for px in vp.take_unpremultiplied() {
        rgb.extend_from_slice(&[px.r, px.g, px.b]);
    }
    Some((w, h, rgb))
}

/// Non-white pixels inside `rect` of a raw RGB8 buffer -- the same measurement
/// [`ink_coverage`] makes on a [`Pixmap`], so the two engines are compared on
/// identical terms.
fn ink_coverage_rgb(buf: &(u32, u32, Vec<u8>), rect: &[f32; 4], page_h: f32, scale: f32) -> usize {
    let (w, h, data) = (buf.0 as i32, buf.1 as i32, &buf.2);
    let pad = 2i32;
    let x0 = ((rect[0] * scale).floor() as i32 - pad).max(0);
    let x1 = ((rect[2] * scale).ceil() as i32 + pad).min(w);
    let y0 = (((page_h - rect[3]) * scale).floor() as i32 - pad).max(0);
    let y1 = (((page_h - rect[1]) * scale).ceil() as i32 + pad).min(h);
    let mut n = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = ((y * w + x) * 3) as usize;
            if let Some(p) = data.get(i..i + 3) {
                let luma = (p[0] as u32 * 30 + p[1] as u32 * 59 + p[2] as u32 * 11) / 100;
                if luma < 250 {
                    n += 1;
                }
            }
        }
    }
    n
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut path: Option<PathBuf> = None;
    let mut dpi = 150.0f32;
    let mut dump_dir: Option<PathBuf> = None;
    let mut scan_fallback = false;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--dpi" => dpi = args.next().and_then(|v| v.parse().ok()).unwrap_or(150.0),
            "--dump-dir" => dump_dir = args.next().map(PathBuf::from),
            "--scan-fallback" => scan_fallback = true,
            "-h" | "--help" => return usage(),
            other => path = Some(PathBuf::from(other)),
        }
    }

    let Some(path) = path else { return usage() };

    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read {}: {e}", path.display());
            std::process::exit(2);
        }
    };
    let doc = match PdfDocument::open(bytes) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("cannot open {} as PDF: {e}", path.display());
            std::process::exit(2);
        }
    };

    if scan_fallback {
        return scan_fallback_report(&doc, dpi);
    }

    println!("file : {}", path.display());
    println!("pages: {}", doc.page_count());
    println!("dpi  : {dpi}");

    let mut total_annots = 0usize;
    let mut total_no_ap = 0usize;
    let mut pages_with_invisible = Vec::new();

    for page_index in 0..doc.page_count() {
        let Ok(page) = doc.page(page_index) else {
            continue;
        };
        let page = page.clone();
        let annots = collect_annots(&doc, &page);
        if annots.is_empty() {
            continue;
        }
        total_annots += annots.len();
        total_no_ap += annots.iter().filter(|a| !a.has_ap).count();

        println!(
            "\n=== page {} -- {} annotation(s) ===",
            page_index + 1,
            annots.len()
        );
        let mut by_subtype: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for a in &annots {
            let e = by_subtype.entry(a.subtype.clone()).or_default();
            e.0 += 1;
            if a.has_ap {
                e.1 += 1;
            }
        }
        for (st, (n, with_ap)) in &by_subtype {
            println!("  /{st:<12} x{n:<4} with /AP: {with_ap}/{n}");
        }

        // Render and measure. `rasterize_page` is the crate's real entry point
        // (native engine + the mandatory hayro fallback), so this measures what
        // a viewer like `kpdf` genuinely gets -- not some private path.
        let pix = match kopitiam_pdf::mupdf::rasterize_page(&doc, page_index, dpi) {
            Ok(p) => p,
            Err(e) => {
                println!("  render failed: {e}");
                continue;
            }
        };
        // Which engine actually drew this page? A non-zero advance-box count
        // means the native engine hit its glyph ceiling and `rasterize_page`
        // handed the page to hayro instead -- worth knowing, because the two
        // engines do not agree on annotations.
        if let Ok((_, boxes)) = kopitiam_pdf::mupdf::rasterize_page_ex(&doc, page_index, dpi) {
            println!(
                "  engine: {}",
                if boxes == 0 {
                    "native (no glyph fallback)".to_string()
                } else {
                    format!("hayro (native fell back on {boxes} glyph(s))")
                }
            );
        }
        let page_h = mediabox_height(&doc, &page);
        let scale = dpi / 72.0;

        let hay = render_hayro(doc.raw_bytes(), page_index, dpi);

        let mut invisible = 0usize;
        let mut invisible_hayro = 0usize;
        for a in &annots {
            if ink_coverage(&pix, &a.rect, page_h, scale) == 0 {
                invisible += 1;
            }
            if let Some(hb) = &hay
                && ink_coverage_rgb(hb, &a.rect, page_h, scale) == 0
            {
                invisible_hayro += 1;
            }
        }
        if hay.is_some() {
            println!(
                "  hayro (direct)      : {}/{} annotation(s) painted{}",
                annots.len() - invisible_hayro,
                annots.len(),
                if invisible_hayro > 0 {
                    "   <-- hayro blank too"
                } else {
                    ""
                }
            );
        }
        println!(
            "  painted inside /Rect: {}/{} annotation(s){}",
            annots.len() - invisible,
            annots.len(),
            if invisible > 0 {
                "   <-- INVISIBLE"
            } else {
                ""
            }
        );
        if invisible > 0 {
            pages_with_invisible.push(page_index + 1);
        }

        if let Some(dir) = &dump_dir {
            let _ = std::fs::create_dir_all(dir);
            let out = dir.join(format!("page-{:03}.ppm", page_index + 1));
            match write_ppm(&out, &pix) {
                Ok(()) => println!("  wrote {}", out.display()),
                Err(e) => println!("  cannot write {}: {e}", out.display()),
            }
        }
    }

    println!("\n--- summary ---");
    println!("annotations       : {total_annots}");
    println!("without /AP       : {total_no_ap}");
    if pages_with_invisible.is_empty() {
        println!("all annotations painted something inside their /Rect");
    } else {
        println!("pages with invisible annots: {pages_with_invisible:?}");
        std::process::exit(1);
    }
}

fn usage() {
    eprintln!(
        "usage: annots <file.pdf> [--dpi N] [--dump-dir DIR]\n\
         \n\
         Reports every page's annotations, whether each carries an /AP\n\
         appearance stream, and whether the rasterizer actually paints\n\
         anything inside the annotation's /Rect. Exits 1 if any annotation\n\
         renders blank."
    );
}

/// One annotation, reduced to just what the visibility question needs.
struct AnnotInfo {
    subtype: String,
    /// `/Rect` in PDF user space, normalised so `x0<=x1`, `y0<=y1`.
    rect: [f32; 4],
    /// Whether `/AP` -> `/N` resolves to anything. No `/AP` means an
    /// `/AP`-only renderer draws nothing at all for this annot.
    has_ap: bool,
}

/// Gather a page's `/Annots`, skipping the ones a viewer must not draw:
/// `/Popup` subtypes and anything flagged Hidden (`/F` bit 2) or NoView
/// (`/F` bit 6) -- PDF 32000-1:2008 table 165.
fn collect_annots(doc: &PdfDocument, page: &Object) -> Vec<AnnotInfo> {
    let Ok(annots) = doc.resolve_get(page, "Annots") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for i in 0..annots.array_len() {
        let Some(entry) = annots.array_get(i) else {
            continue;
        };
        let Ok(annot) = doc.resolve(entry) else {
            continue;
        };
        if !annot.is_dict() {
            continue;
        }
        let subtype = doc
            .resolve_get(&annot, "Subtype")
            .map(|o| String::from_utf8_lossy(o.to_name()).into_owned())
            .unwrap_or_default();
        if subtype == "Popup" {
            continue;
        }
        let flags = doc
            .resolve_get(&annot, "F")
            .map(|o| o.to_int())
            .unwrap_or(0);
        if flags & 2 != 0 || flags & 32 != 0 {
            continue;
        }
        let has_ap = doc
            .resolve_get(&annot, "AP")
            .ok()
            .and_then(|ap| doc.resolve_get(&ap, "N").ok())
            .map(|n| !n.is_null())
            .unwrap_or(false);
        let Some(rect) = rect_of(doc, &annot) else {
            continue;
        };
        out.push(AnnotInfo {
            subtype,
            rect,
            has_ap,
        });
    }
    out
}

/// `/Rect` as normalised `[x0, y0, x1, y1]` in user space.
fn rect_of(doc: &PdfDocument, annot: &Object) -> Option<[f32; 4]> {
    let r = doc.resolve_get(annot, "Rect").ok()?;
    if r.array_len() < 4 {
        return None;
    }
    let v = |i: usize| -> f32 {
        r.array_get(i)
            .and_then(|o| doc.resolve(o).ok())
            .map(|o| o.to_real() as f32)
            .unwrap_or(0.0)
    };
    let (a, b, c, d) = (v(0), v(1), v(2), v(3));
    Some([a.min(c), b.min(d), a.max(c), b.max(d)])
}

/// MediaBox height in points, US-Letter fallback -- matches the rasterizer's
/// own guard, so the device-space mapping below lines up with the pixmap.
fn mediabox_height(doc: &PdfDocument, page: &Object) -> f32 {
    let mb = doc.resolve_get(page, "MediaBox").unwrap_or(Object::Null);
    if mb.array_len() >= 4 {
        let v = |i: usize| -> f32 {
            mb.array_get(i)
                .and_then(|o| doc.resolve(o).ok())
                .map(|o| o.to_real() as f32)
                .unwrap_or(0.0)
        };
        let h = (v(1) - v(3)).abs();
        if h >= 1.0 {
            return h;
        }
    }
    792.0
}

/// Count non-white pixels inside `rect`.
///
/// The `/Rect` is in user space (y up from the bottom-left); the pixmap is
/// device space (y down from the top-left), hence the flip. The box is padded
/// by 2px because a stroke centred on the boundary of a hairline-thin `/Rect`
/// -- which is exactly what a near-vertical ink stroke produces -- lies half
/// outside it, and we would rather over-count a little than declare a visible
/// annotation invisible.
fn ink_coverage(pix: &Pixmap, rect: &[f32; 4], page_h: f32, scale: f32) -> usize {
    let pad = 2i32;
    let x0 = ((rect[0] * scale).floor() as i32 - pad).max(0);
    let x1 = ((rect[2] * scale).ceil() as i32 + pad).min(pix.width() as i32);
    // y flip: device_y = (page_height - user_y) * scale
    let y0 = (((page_h - rect[3]) * scale).floor() as i32 - pad).max(0);
    let y1 = (((page_h - rect[1]) * scale).ceil() as i32 + pad).min(pix.height() as i32);

    let mut n = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            if pix.luma(x, y).map(|l| l < 250).unwrap_or(false) {
                n += 1;
            }
        }
    }
    n
}

/// Write `pix` as a binary PPM (P6). Chosen over PNG so this example needs no
/// image-encoder dependency -- every image viewer reads PPM.
fn write_ppm(path: &std::path::Path, pix: &Pixmap) -> std::io::Result<()> {
    use std::io::Write;
    let (w, h) = (pix.width(), pix.height());
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(f, "P6\n{w} {h}\n255\n")?;
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            match pix.pixel(x, y) {
                Some(p) if p.len() >= 3 => f.write_all(&p[..3])?,
                Some(p) => f.write_all(&[p[0], p[0], p[0]])?,
                None => f.write_all(&[255, 255, 255])?,
            }
        }
    }
    f.flush()
}

/// Report, per page, how many glyphs fell back to a solid advance box -- i.e.
/// how much of a document kopitiam's own engine cannot draw, and would hand to
/// `hayro` instead.
///
/// This is the measurement that decides whether an embedded-font dependency is
/// worth taking: "the fallback exists" says nothing about how often it fires.
/// A document that never triggers it needs no new dependency; one where most
/// pages trigger it is a document our own engine effectively cannot render.
fn scan_fallback_report(doc: &PdfDocument, dpi: f32) {
    let mut pages_with_fallback = 0usize;
    let mut total_boxes = 0usize;
    let mut worst = (0usize, 0usize);

    for page_index in 0..doc.page_count() {
        match kopitiam_pdf::mupdf::rasterize_page_ex(doc, page_index, dpi) {
            Ok((_, boxes)) => {
                if boxes > 0 {
                    pages_with_fallback += 1;
                    total_boxes += boxes;
                    if boxes > worst.1 {
                        worst = (page_index + 1, boxes);
                    }
                }
            }
            Err(e) => println!("page {}: render failed: {e}", page_index + 1),
        }
    }

    let pages = doc.page_count();
    println!("pages                    : {pages}");
    println!("pages hitting fallback   : {pages_with_fallback}");
    println!("total advance-box glyphs : {total_boxes}");
    if worst.1 > 0 {
        println!("worst page               : {} ({} boxes)", worst.0, worst.1);
    }
    if pages_with_fallback == 0 {
        println!("\nthis document renders entirely with kopitiam's own engine");
    } else {
        println!(
            "\n{:.1}% of pages would be handed to hayro",
            100.0 * pages_with_fallback as f32 / pages.max(1) as f32
        );
    }
}
