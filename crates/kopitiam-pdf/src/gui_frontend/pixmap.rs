//! Bridging [`crate::mupdf::Pixmap`] (the rasterizer's own colourspace-
//! agnostic pixel buffer) and [`crate::mupdf::PdfDocument`]'s annotation
//! data to what a GUI viewer's frame loop actually needs: RGBA8 bytes ready
//! for a texture upload, and a count of what should visibly be drawn.

use crate::mupdf::{PdfDocument, Pixmap};

/// [`Pixmap`]'s DeviceRGB samples (`rasterize_page`'s output: `n = 3`,
/// `alpha = false`) to the RGBA8 bytes a GUI toolkit's image type needs
/// (e.g. egui's `ColorImage`), appending an opaque alpha byte per pixel.
/// Also tolerates a grayscale or already-RGBA pixmap, in case a caller ever
/// hands this a different colourspace.
pub fn rgb_to_rgba(pix: &Pixmap) -> Vec<u8> {
    let n = pix.n as usize;
    let px_count = (pix.w as usize) * (pix.h as usize);
    let mut out = Vec::with_capacity(px_count * 4);
    for row in 0..pix.h as usize {
        let row_start = row * pix.stride;
        for col in 0..pix.w as usize {
            let i = row_start + col * n;
            let s = &pix.samples[i..i + n];
            match (n, pix.alpha) {
                (4, true) => out.extend_from_slice(s), // already RGBA
                (3, false) => {
                    out.extend_from_slice(s);
                    out.push(255);
                }
                (1, false) => out.extend_from_slice(&[s[0], s[0], s[0], 255]),
                _ => {
                    let r = s[0];
                    let g = s.get(1).copied().unwrap_or(r);
                    let b = s.get(2).copied().unwrap_or(r);
                    out.extend_from_slice(&[r, g, b, 255]);
                }
            }
        }
    }
    out
}

/// Count the annotations on `page_index` that a viewer is expected to draw.
///
/// Mirrors the skip rules the renderer itself applies (see
/// `crate::mupdf::annot_run`): `/Popup` subtypes are never drawn inline, and
/// `/F` bit 2 (Hidden) or bit 6 (NoView) means "do not display" per PDF
/// 32000-1:2008 table 165. Anything else counts, whether or not it carries
/// an `/AP` -- an annotation stored as pure data still has an appearance
/// synthesised for it, so it is genuinely expected on screen.
pub fn drawable_annot_count(doc: &PdfDocument, page_index: usize) -> usize {
    let Ok(page) = doc.page(page_index) else {
        return 0;
    };
    let Ok(annots) = doc.resolve_get(page, "Annots") else {
        return 0;
    };
    (0..annots.array_len())
        .filter_map(|i| annots.array_get(i))
        .filter_map(|entry| doc.resolve(entry).ok())
        .filter(|annot| annot.is_dict())
        .filter(|annot| {
            doc.resolve_get(annot, "Subtype")
                .map(|st| st.to_name() != b"Popup")
                .unwrap_or(false)
        })
        .filter(|annot| {
            let flags = doc.resolve_get(annot, "F").map(|o| o.to_int()).unwrap_or(0);
            flags & 2 == 0 && flags & 32 == 0
        })
        .count()
}
