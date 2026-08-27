//! Cross-reader proof for annotation authoring (gh-80).
//!
//! `annot_edit.rs` has its own round-trip tests, but they all end at *our* own
//! parser: we write a file, we read it back, we rasterize it. That proves we are
//! self-consistent, which is exactly the failure mode that would let a bug ship.
//! A PDF only KOPITIAM can read is silent data loss on somebody's annotated
//! document.
//!
//! So this suite asks a different question: **can a completely independent
//! implementation see what we wrote?** It shells out to poppler (`pdftoppm`,
//! which is Okular's backend) and checks the annotation is actually there in
//! poppler's raster — not merely that poppler tolerated the file without
//! erroring.
//!
//! If `pdftoppm` is not installed the cross-reader tests **fail** rather than
//! skip. A silently-skipping compatibility test is worse than none: it turns
//! into a green tick that proves nothing, on precisely the property most likely
//! to break.

use std::process::Command;

use kopitiam_pdf::mupdf::{
    InkAnnotSpec, InkStroke, PdfDocument, Pixmap, add_ink_annot, delete_annot, page_annot_refs,
    rasterize_page,
};

const FIXTURE: &[u8] = include_bytes!("fixtures/ink-annots-no-ap.pdf");
const DPI: f32 = 150.0;

/// A stroke in a part of the 200x200pt fixture page that starts out blank, in a
/// colour no existing annotation uses, so any hit is unambiguously ours.
fn green_spec() -> InkAnnotSpec {
    InkAnnotSpec {
        page_index: 0,
        strokes: vec![InkStroke {
            points: vec![(30.0, 100.0), (90.0, 120.0), (150.0, 100.0)],
        }],
        color: [0.0, 1.0, 0.0],
        width: 3.0,
        opacity: 1.0,
        author: Some("kopitiam-test".to_string()),
    }
}

/// Count strongly-green pixels — the marker colour written above.
fn green_pixels(pix: &Pixmap) -> usize {
    let mut n = 0;
    for y in 0..pix.height() as i32 {
        for x in 0..pix.width() as i32 {
            let Some(p) = pix.pixel(x, y) else { continue };
            if p.len() >= 3 {
                let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
                if g > 120 && g - r > 60 && g - b > 60 {
                    n += 1;
                }
            }
        }
    }
    n
}

/// Green pixels in a poppler-rendered PNG, decoded without an image crate.
fn green_pixels_in_png(path: &std::path::Path) -> usize {
    let data = std::fs::read(path).expect("read png");
    let (mut idat, mut pos) = (Vec::new(), 8usize);
    let (mut w, mut h, mut colour) = (0u32, 0u32, 0u8);
    while pos + 8 <= data.len() {
        let len = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let kind = &data[pos + 4..pos + 8];
        let body = &data[pos + 8..(pos + 8 + len).min(data.len())];
        if kind == b"IHDR" {
            w = u32::from_be_bytes(body[0..4].try_into().unwrap());
            h = u32::from_be_bytes(body[4..8].try_into().unwrap());
            colour = body[9];
        } else if kind == b"IDAT" {
            idat.extend_from_slice(body);
        }
        pos += 12 + len;
    }
    assert_eq!(colour, 2, "expected 8-bit truecolour PNG from pdftoppm");
    let raw = miniz_oxide::inflate::decompress_to_vec_zlib(&idat).expect("inflate png");

    // Undo PNG's per-scanline filters (RFC 2083 §6). Doing this by hand keeps the
    // test free of an image-decoder dependency.
    let (nch, stride) = (3usize, w as usize * 3);
    let (mut out, mut prev) = (Vec::with_capacity(stride * h as usize), vec![0u8; stride]);
    let mut p = 0usize;
    let paeth = |a: i32, b: i32, c: i32| {
        let (pa, pb, pc) = ((b - c).abs(), (a - c).abs(), (a + b - 2 * c).abs());
        if pa <= pb && pa <= pc {
            a
        } else if pb <= pc {
            b
        } else {
            c
        }
    };
    for _ in 0..h {
        let filter = raw[p];
        p += 1;
        let mut line = raw[p..p + stride].to_vec();
        p += stride;
        for i in 0..stride {
            let a = if i >= nch { line[i - nch] as i32 } else { 0 };
            let b = prev[i] as i32;
            let c = if i >= nch { prev[i - nch] as i32 } else { 0 };
            let cur = line[i] as i32;
            line[i] = match filter {
                1 => cur + a,
                2 => cur + b,
                3 => cur + (a + b) / 2,
                4 => cur + paeth(a, b, c),
                _ => cur,
            } as u8;
        }
        out.extend_from_slice(&line);
        prev = line;
    }
    out.chunks_exact(3)
        .filter(|px| {
            let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
            g > 120 && g - r > 60 && g - b > 60
        })
        .count()
}

/// Render `bytes` with poppler and return the produced PNG's path.
fn poppler_render(bytes: &[u8], tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("kopitiam-authoring-{}-{tag}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let pdf = dir.join("doc.pdf");
    std::fs::write(&pdf, bytes).expect("write pdf");
    let out = Command::new("pdftoppm")
        .args(["-r", "150", "-png", "-f", "1", "-l", "1"])
        .arg(&pdf)
        .arg(dir.join("page"))
        .output()
        .expect(
            "pdftoppm (poppler) must be installed: these are cross-reader compatibility \
             tests, and skipping them would turn the one property most likely to break \
             into a green tick that proves nothing",
        );
    assert!(
        out.status.success(),
        "poppler REJECTED a file we wrote -- stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::read_dir(&dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "png"))
        .expect("pdftoppm exited 0 but produced no PNG")
}

/// The headline: an ink annotation we authored is visible **to poppler**.
///
/// This is the test that would have caught a writer producing a file only our
/// own parser accepts.
#[test]
fn authored_ink_is_visible_to_poppler() {
    let doc = PdfDocument::open(FIXTURE.to_vec()).expect("fixture opens");
    let before = poppler_render(FIXTURE, "before");
    let green_before = green_pixels_in_png(&before);
    assert_eq!(
        green_before, 0,
        "fixture should have no green ink to begin with"
    );

    let edited = add_ink_annot(&doc, &green_spec()).expect("add_ink_annot");
    let after = poppler_render(&edited, "after");
    let green_after = green_pixels_in_png(&after);

    assert!(
        green_after > 200,
        "poppler rendered our file but did not draw the ink we authored \
         (green px {green_before} -> {green_after}). Either the annotation was not \
         written, or it was written without a usable /AP -- which is exactly the \
         defect gh-79 was about, reintroduced from the write side."
    );
}

/// The same annotation is also visible to our own renderer, and shows up in the
/// page's annotation list. Guards the case where we satisfy poppler but not
/// ourselves.
#[test]
fn authored_ink_round_trips_through_our_own_reader() {
    let doc = PdfDocument::open(FIXTURE.to_vec()).expect("fixture opens");
    let n_before = page_annot_refs(&doc, 0).len();

    let edited = add_ink_annot(&doc, &green_spec()).expect("add_ink_annot");
    let redoc = PdfDocument::open(edited).expect("edited file must reopen");
    assert_eq!(
        page_annot_refs(&redoc, 0).len(),
        n_before + 1,
        "annot count did not grow"
    );

    let pix = rasterize_page(&redoc, 0, DPI).expect("rasterize edited");
    assert!(
        green_pixels(&pix) > 200,
        "our own renderer does not show the authored ink"
    );
}

/// Deleting removes it again, and the result still opens everywhere.
#[test]
fn deleting_an_authored_annot_removes_it_for_poppler_too() {
    let doc = PdfDocument::open(FIXTURE.to_vec()).expect("fixture opens");
    let edited = add_ink_annot(&doc, &green_spec()).expect("add_ink_annot");
    let redoc = PdfDocument::open(edited).expect("reopen");

    let added = page_annot_refs(&redoc, 0)
        .into_iter()
        .find(|a| a.subtype == "Ink" && a.rect.y0 > 90.0 && a.rect.y1 < 135.0)
        .expect("must find the annot we just added");

    let deleted = delete_annot(&redoc, 0, added.num).expect("delete_annot");
    let png = poppler_render(&deleted, "deleted");
    assert_eq!(
        green_pixels_in_png(&png),
        0,
        "poppler still draws an annotation we deleted"
    );
}

/// Undo is truncation: because every edit appends, the pre-edit state is a
/// literal prefix of the post-edit bytes. This pins the invariant the whole
/// `EditHistory` design rests on — if an edit path ever starts rewriting in
/// place, this fails loudly instead of undo quietly corrupting a document.
#[test]
fn every_edit_is_append_only_so_undo_can_truncate() {
    let doc = PdfDocument::open(FIXTURE.to_vec()).expect("fixture opens");
    let edited = add_ink_annot(&doc, &green_spec()).expect("add_ink_annot");

    assert!(
        edited.len() > FIXTURE.len(),
        "an edit must grow the file, not rewrite it"
    );
    assert_eq!(
        &edited[..FIXTURE.len()],
        FIXTURE,
        "the original bytes were modified -- incremental update must be append-only, \
         and EditHistory's undo-by-truncation depends on it"
    );

    let truncated = edited[..FIXTURE.len()].to_vec();
    let reopened = PdfDocument::open(truncated).expect("truncated file must still open");
    assert_eq!(
        page_annot_refs(&reopened, 0).len(),
        page_annot_refs(&doc, 0).len()
    );
}
