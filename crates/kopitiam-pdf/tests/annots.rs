//! Regression gate for gh-79 / bd-01x: annotations must actually be painted.
//!
//! The bug: `kopitiam-pdf` rendered no annotations at all, because two halves
//! were missing. A page's `/Annots` is a **sibling** of `/Contents`, so running
//! only the content stream draws none of them; and an annotation written
//! *without* an `/AP` appearance stream has nothing to draw even once you do
//! look at `/Annots` -- the viewer has to synthesise the appearance from
//! `/InkList` + `/C` + `/Border` (PDF 32000-1:2008 §12.5.6.13).
//!
//! The fixture (`fixtures/ink-annots-no-ap.pdf`, 989 bytes, regenerate with
//! `fixtures/make-ink-annots-no-ap.py`) is synthetic on purpose: the bug was
//! found on a private document, and a hand-built file gives exact coordinates
//! to assert on. It carries **no `/AP` anywhere** and three ink annots:
//!
//! | annot | colour | why it is there |
//! |---|---|---|
//! | 1 | blue  | ordinary stroke, well inside the page |
//! | 2 | red   | near-**vertical**: its `/Rect` is 1pt wide, so a renderer that forgets to widen the box for the stroke width clips it away |
//! | 3 | green | `/F 2` (Hidden) -- **must stay invisible** |
//!
//! Cross-checked against poppler (`pdftoppm`, Okular's backend) at 150 dpi:
//! blue 1044 px, red 1695 px, green 0 px. We do not assert poppler's exact
//! counts -- different rasterizers antialias differently -- but the
//! *qualitative* result (blue present, red present, green absent) is engine
//! independent, and a hard lower bound catches "drew a single stray pixel".

use kopitiam_pdf::mupdf::{PdfDocument, Pixmap, rasterize_page};

const FIXTURE: &[u8] = include_bytes!("fixtures/ink-annots-no-ap.pdf");
const DPI: f32 = 150.0;

/// Count pixels where one channel clearly dominates the other two -- a
/// rasterizer-independent way to ask "did this coloured annotation get drawn?"
/// without depending on antialiasing details.
fn channel_counts(pix: &Pixmap) -> (usize, usize, usize) {
    let (mut red, mut green, mut blue) = (0, 0, 0);
    for y in 0..pix.height() as i32 {
        for x in 0..pix.width() as i32 {
            let Some(p) = pix.pixel(x, y) else { continue };
            if p.len() < 3 {
                continue;
            }
            let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
            if r > 120 && r - g > 60 && r - b > 60 {
                red += 1;
            }
            if g > 120 && g - r > 60 && g - b > 60 {
                green += 1;
            }
            if b > 120 && b - r > 60 && b - g > 60 {
                blue += 1;
            }
        }
    }
    (red, green, blue)
}

fn render_fixture() -> Pixmap {
    let doc = PdfDocument::open(FIXTURE.to_vec()).expect("fixture must parse");
    rasterize_page(&doc, 0, DPI).expect("fixture must rasterize")
}

/// The headline regression: an `/Ink` annotation with **no `/AP`** is visible.
///
/// Before the fix this was 0 -- and stayed 0 even when the page was handed to
/// `hayro`, which gates annotations behind `/AP`. So this asserts the
/// synthesis path specifically, not merely "some annotation code ran".
#[test]
fn ink_annots_without_ap_are_painted() {
    let (red, _green, blue) = channel_counts(&render_fixture());

    // Lower bound, not an exact count: 200 px is far above stray-pixel noise
    // and far below poppler's 1044/1695, so it survives antialiasing
    // differences while still failing loudly on "nothing drawn".
    assert!(
        blue > 200,
        "blue ink annotation not painted (blue px = {blue}, want > 200)"
    );
    assert!(
        red > 200,
        "near-vertical red ink annotation not painted (red px = {red}, want > 200). \
         Its /Rect is only 1pt wide -- a stroke centred on the polyline falls \
         outside it, so the appearance box must be widened by the line width \
         (MuPDF pdf-appearance.c expands by lw + 6)."
    );
}

/// A `/F 2` (Hidden) annotation must NOT be drawn. Guards the obvious wrong
/// fix -- painting everything in `/Annots` and ignoring the flags.
#[test]
fn hidden_annots_stay_hidden() {
    let (_red, green, _blue) = channel_counts(&render_fixture());
    assert_eq!(
        green, 0,
        "annotation flagged /F 2 (Hidden) was painted ({green} green px)"
    );
}

/// Annotations are drawn *on top of* page content, and must not erase it: the
/// fixture's black rule at the top of the page has to survive.
#[test]
fn page_content_still_renders_alongside_annots() {
    let pix = render_fixture();
    let dark = (0..pix.height() as i32)
        .flat_map(|y| (0..pix.width() as i32).map(move |x| (x, y)))
        .filter(|&(x, y)| pix.luma(x, y).map(|l| l < 100).unwrap_or(false))
        .count();
    assert!(
        dark > 100,
        "page content stroke vanished (dark px = {dark})"
    );
}

/// A file with no annotations at all must be unaffected -- the annotation pass
/// must not change, or fail, an ordinary page.
#[test]
fn page_without_annots_is_unaffected() {
    // Same fixture minus /Annots: cheapest way to build one is to rename the
    // key in place to something no reader looks for, keeping every byte offset
    // (and so the xref) valid.
    //
    // This patch is done on BYTES, not via `String::from_utf8_lossy`. The
    // fixture opens with the conventional binary comment `%\xe2\xe3\xcf\xd3`
    // that marks a PDF as non-text; those bytes are not valid UTF-8, so a
    // lossy conversion rewrites each one as U+FFFD (3 bytes) and silently
    // grows the file, invalidating every xref offset after it.
    const KEY: &[u8] = b"/Annots";
    const REPLACEMENT: &[u8] = b"/Xnnots";
    let at = FIXTURE
        .windows(KEY.len())
        .position(|w| w == KEY)
        .expect("fixture must contain /Annots");
    let mut patched = FIXTURE.to_vec();
    patched[at..at + KEY.len()].copy_from_slice(REPLACEMENT);
    assert_eq!(
        patched.len(),
        FIXTURE.len(),
        "patch must preserve xref offsets"
    );
    assert_ne!(patched, FIXTURE, "patch must actually change something");

    let doc = PdfDocument::open(patched).expect("patched fixture must parse");
    let pix = rasterize_page(&doc, 0, DPI).expect("patched fixture must rasterize");
    let (red, green, blue) = channel_counts(&pix);
    assert_eq!(
        (red, green, blue),
        (0, 0, 0),
        "annots drawn despite no /Annots key"
    );
}

// ---------------------------------------------------------------------------
// The other half: consuming a real /AP, and not drawing it twice
// ---------------------------------------------------------------------------

/// `ink-annots-mixed-ap.pdf`: two ink annots, one **with** a real `/AP` form
/// XObject (magenta) and one **without** (cyan).
///
/// The `/AP` form's `/BBox` is `[0 0 50 10]` while its annot `/Rect` is
/// `[20 150 120 170]`, so the PDF 32000-1 §12.5.5 Algorithm 8.1 mapping has to
/// apply a real 2× scale **and** a translation. A port that composed the
/// matrices in the wrong order would still draw *something*, just in the wrong
/// place or at the wrong size — so this fixture is specifically shaped to
/// catch that, rather than letting an identity transform hide the bug.
///
/// Poppler ground truth at 150 dpi: magenta 1804 px, cyan 1531 px.
const MIXED: &[u8] = include_bytes!("fixtures/ink-annots-mixed-ap.pdf");

/// Count magenta (`/AP`) and cyan (synthesised) pixels.
fn mixed_counts(pix: &Pixmap) -> (usize, usize) {
    let (mut magenta, mut cyan) = (0, 0);
    for y in 0..pix.height() as i32 {
        for x in 0..pix.width() as i32 {
            let Some(p) = pix.pixel(x, y) else { continue };
            if p.len() < 3 {
                continue;
            }
            let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
            if r > 120 && b > 120 && r - g > 60 && b - g > 60 {
                magenta += 1;
            }
            if g > 120 && b > 120 && g - r > 60 && b - r > 60 {
                cyan += 1;
            }
        }
    }
    (magenta, cyan)
}

/// A real `/AP` appearance stream renders, and so does a synthesised one, in
/// the same pass. Covers the consume half (gap 1) end-to-end, including the
/// non-identity Algorithm 8.1 transform.
#[test]
fn real_ap_and_synthesized_both_render() {
    let doc = PdfDocument::open(MIXED.to_vec()).expect("mixed fixture must parse");
    let pix = rasterize_page(&doc, 0, DPI).expect("mixed fixture must rasterize");
    let (magenta, cyan) = mixed_counts(&pix);
    assert!(
        magenta > 200,
        "real /AP annotation not painted (magenta px = {magenta})"
    );
    assert!(
        cyan > 200,
        "synthesised annotation not painted (cyan px = {cyan})"
    );
}

/// `AnnotPass::SynthesizedOnly` must paint the AP-less annot and **skip** the
/// one with a real `/AP`.
///
/// This is the guarantee the `hayro` fallback path depends on. When a page hits
/// the glyph fallback, hayro renders it and draws the real-`/AP` annots itself
/// (it gates on `/AP` → `/N`), then we overlay only what it skipped. If this
/// filter leaked, every real-`/AP` annot on a fallback page would be composited
/// twice — invisible for an opaque stroke, but visibly darker for any annot
/// with `/CA` < 1.
#[test]
fn synthesized_only_pass_skips_real_ap() {
    use kopitiam_pdf::mupdf::{AnnotPass, DrawDevice, Matrix, page_run};

    let doc = PdfDocument::open(MIXED.to_vec()).expect("mixed fixture must parse");
    let page = doc.page(0).expect("page 0").clone();
    let scale = DPI / 72.0;

    // A blank device, so whatever appears came from this pass alone.
    let mut dev = DrawDevice::new(
        (200.0 * scale) as u32,
        (200.0 * scale) as u32,
        Matrix::scale(scale, scale),
    );
    kopitiam_pdf::mupdf::run_page_annots_with(
        &doc,
        &page,
        page_run::page_ctm(&doc, &page),
        &mut dev,
        AnnotPass::SynthesizedOnly,
    )
    .expect("annot pass must not fail");

    let (magenta, cyan) = mixed_counts(dev.pixmap());
    assert!(
        cyan > 200,
        "SynthesizedOnly dropped the AP-less annot (cyan px = {cyan})"
    );
    assert_eq!(
        magenta, 0,
        "SynthesizedOnly painted a real-/AP annot ({magenta} magenta px); on the hayro \
         fallback path this double-draws what hayro already drew"
    );
}

// ---------------------------------------------------------------------------
// Widget appearance states (radio buttons)
// ---------------------------------------------------------------------------

/// `radio-form.pdf`: three `/Widget` annotations whose `/AP` `/N` is a
/// **dictionary of appearance states**, differing only in whether `/AS` is
/// present.
///
/// | widget | `/AS` |
/// |---|---|
/// | radio A, `/Rect [20 150 40 170]` | present (`/Off`) |
/// | radio B, `/Rect [20 110 40 130]` | **absent** |
/// | checkbox, `/Rect [20 60 40 80]` | present (`/Off`) |
///
/// Regenerate with `fixtures/make-radio-form.py`.
const RADIO: &[u8] = include_bytes!("fixtures/radio-form.pdf");

/// Every widget draws, including one with **no `/AS`**.
///
/// This was a real, user-visible bug: radio buttons were completely invisible
/// in kpdf while still toggling correctly — you could click one, see nothing,
/// and then find the change had taken effect when opening the file in Okular.
/// A radio's `/N` always holds at least two states (`/Off` plus an on-state),
/// so the old "only pick a state when the dict has exactly one entry" rule
/// skipped every `/AS`-less widget and drew nothing at all.
///
/// Poppler renders these: it logs *"Invalid or missing AS value in annotation
/// containing one or more appearance subdictionaries"* and carries on, drawing
/// the same 439 dark pixels for the `/AS`-less radio as for the one with `/AS`.
/// A malformed widget is a file to recover from, not a reason to show the user
/// nothing.
#[test]
fn widgets_render_even_without_an_appearance_state() {
    let doc = PdfDocument::open(RADIO.to_vec()).expect("radio fixture must parse");
    let pix = rasterize_page(&doc, 0, DPI).expect("radio fixture must rasterize");
    let page_h = 200.0_f32;
    let scale = DPI / 72.0;

    // Dark pixels inside a user-space rect, y-flipped into device space.
    let dark_in = |x0: f32, y0: f32, x1: f32, y1: f32| -> usize {
        let (dx0, dx1) = ((x0 * scale) as i32, (x1 * scale) as i32);
        let (dy0, dy1) = (
            ((page_h - y1) * scale) as i32,
            ((page_h - y0) * scale) as i32,
        );
        (dy0..dy1)
            .flat_map(|y| (dx0..dx1).map(move |x| (x, y)))
            .filter(|&(x, y)| pix.luma(x, y).map(|l| l < 128).unwrap_or(false))
            .count()
    };

    let radio_with_as = dark_in(20.0, 150.0, 40.0, 170.0);
    let radio_without_as = dark_in(20.0, 110.0, 40.0, 130.0);
    let checkbox = dark_in(20.0, 60.0, 40.0, 80.0);

    assert!(
        radio_with_as > 50,
        "radio WITH /AS not drawn ({radio_with_as} dark px)"
    );
    assert!(checkbox > 50, "checkbox not drawn ({checkbox} dark px)");
    assert!(
        radio_without_as > 50,
        "radio with NO /AS was not drawn ({radio_without_as} dark px). It must fall back \
         to the field's /V, then to /Off, rather than being skipped -- an invisible \
         widget that still toggles is worse than a wrongly-stated one."
    );
}
