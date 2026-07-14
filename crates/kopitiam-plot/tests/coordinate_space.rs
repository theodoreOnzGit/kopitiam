//! Pins down the load-bearing assumption of the whole crate: that the path
//! geometry we extract ourselves and the text spans `kopitiam-pdf` extracts
//! land in the *same* coordinate system.
//!
//! If this ever fails, tick labels stop matching tick marks and every
//! calibration silently becomes wrong -- so it is asserted directly, against a
//! page whose geometry we chose, rather than being left as a comment.

mod common;

use common::{PdfBuilder, Plot};
use kopitiam_plot::content::paths_on_page;

#[test]
fn paths_and_text_share_one_coordinate_space() {
    let plot = Plot::linear_unit_square();
    let mut pdf = PdfBuilder::new(612.0, 792.0);
    plot.draw_axes(&mut pdf);
    let bytes = pdf.build();

    // Our own lopdf content-stream walk.
    let doc = lopdf::Document::load_mem(&bytes).expect("test PDF must parse");
    let paths = paths_on_page(&doc, 1);
    assert!(!paths.is_empty(), "vector paths must be recoverable at all");

    // The x-axis spine was drawn from (100,150) to (500,150). Find a
    // horizontal segment matching that, in page coordinates.
    let spine = paths
        .iter()
        .flat_map(|p| p.subpaths.iter())
        .flat_map(|s| s.line_segments())
        .find(|(a, b)| (a.y - 150.0).abs() < 0.01 && (b.x - a.x).abs() > 350.0)
        .expect("x-axis spine must be found where it was drawn");
    assert!((spine.0.x - 100.0).abs() < 0.01);
    assert!((spine.1.x - 500.0).abs() < 0.01);

    // kopitiam-pdf's text extraction, on the same bytes.
    let pages = kopitiam_pdf::extract_from_bytes(&bytes).expect("text must extract");
    let spans = &pages[0].spans;
    assert!(!spans.is_empty(), "tick labels must extract as text");

    // The tick labels were drawn at y = 150 - 16 = 134, i.e. BELOW the spine at
    // y = 150 in a y-up space. If pdf-extract were handing us a y-DOWN space,
    // the labels would come back *above* the spine and this assertion would
    // fail -- which is exactly the confusion being ruled out.
    let label_y = spans[0].y;
    assert!(
        label_y < 150.0,
        "tick labels must sit below the axis in the shared y-up space, got y={label_y}"
    );
    assert!(
        (label_y - 134.0).abs() < 2.0,
        "tick label baseline should be ~134 (the y we drew it at), got {label_y}"
    );

    // And the labels must span the same x range as the axis they belong to.
    let min_x = spans.iter().map(|s| s.x).fold(f32::MAX, f32::min);
    let max_x = spans.iter().map(|s| s.x + s.width).fold(f32::MIN, f32::max);
    assert!(
        min_x > 60.0 && max_x < 520.0,
        "label x range {min_x}..{max_x} must overlap the axis x range 100..500"
    );
}
