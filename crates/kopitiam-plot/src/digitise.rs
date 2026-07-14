//! The pipeline: PDF page in, digitised plots out.
//!
//! ```text
//!   PDF page
//!     |
//!     +-- content stream --> painted paths (crate::content)
//!     |                        |
//!     |                        +--> spines --> plot regions (crate::axes)
//!     |                        +--> tick marks
//!     |                        +--> data curves + markers (crate::series)
//!     |
//!     +-- text spans (kopitiam-pdf) --> labels (crate::labels)
//!                                        |
//!                                        +--> tick values, axis titles, legend
//!
//!   plot region + ticks + labels --> axis calibration (linear or log)
//!   calibration + curves          --> DataPoints, each carrying its page origin
//! ```
//!
//! The two extractions -- geometry and text -- are independent walks over the
//! same file, and everything downstream depends on them landing in the same
//! coordinate space. They do; see [`crate::geometry`] for why, and
//! `tests/coordinate_space.rs` for the proof.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::axes::{
    self, Axis, AxisKind, TickSource,
};
use crate::content::{PaintedPath, paths_on_page};
use crate::geometry::Rect;
use crate::labels::{self, Label};
use crate::series::{self, Series};

#[derive(Debug, thiserror::Error)]
pub enum PlotError {
    #[error("failed to load PDF: {0}")]
    Load(#[from] lopdf::Error),
    #[error("failed to extract text from PDF: {0}")]
    Text(#[from] kopitiam_pdf::ExtractError),
    #[error("page {requested} is out of range (document has {available} page(s))")]
    NoSuchPage { requested: usize, available: usize },
}

/// The calibration of both axes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisCalibration {
    pub x: Axis,
    pub y: Axis,
}

impl AxisCalibration {
    /// Whether both axes are calibrated. When this is false, the plot's series
    /// carry no data values -- only their page geometry.
    pub fn is_complete(&self) -> bool {
        self.x.is_calibrated() && self.y.is_calibrated()
    }
}

/// One digitised plot.
///
/// Everything needed to audit a recovered number is here: the calibration it was
/// mapped through (including the tick observations the calibration was fitted
/// from, with their printed text), the page coordinate each point came from, and
/// an explicit list of everything we were unsure about.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DigitisedPlot {
    /// 1-based page number in the source document.
    pub page: usize,
    /// The plot area in PDF user space.
    pub region: Rect,
    pub axes: AxisCalibration,
    pub series: Vec<Series>,
    /// Everything we could not be sure about, in plain language.
    ///
    /// **Read this.** A digitised plot with warnings is not necessarily wrong,
    /// but it is not to be published against without a human looking at the
    /// figure. An empty list is a real claim: it means the axes calibrated
    /// cleanly, the scale was determined from evidence rather than assumed, and
    /// no series was ambiguous.
    pub warnings: Vec<String>,
}

impl DigitisedPlot {
    /// Whether this digitisation carries no caveats at all.
    pub fn is_clean(&self) -> bool {
        self.warnings.is_empty() && self.axes.is_complete()
    }

    /// Total recovered data points across all series.
    pub fn point_count(&self) -> usize {
        self.series.iter().map(|s| s.points.len()).sum()
    }
}

/// Digitise every plot on one page of a PDF.
///
/// `page` is 1-based. Returns one [`DigitisedPlot`] per plot found, which for a
/// multi-panel figure means one per panel.
pub fn digitise(pdf: &Path, page: usize) -> Result<Vec<DigitisedPlot>, PlotError> {
    let bytes = std::fs::read(pdf).map_err(lopdf::Error::from)?;
    digitise_bytes(&bytes, page)
}

/// Digitise every plot on one page of a PDF already in memory.
pub fn digitise_bytes(bytes: &[u8], page: usize) -> Result<Vec<DigitisedPlot>, PlotError> {
    let doc = lopdf::Document::load_mem(bytes)?;
    let pages = kopitiam_pdf::extract_from_bytes(bytes)?;

    let text_page = pages
        .iter()
        .find(|p| p.number == page)
        .ok_or(PlotError::NoSuchPage {
            requested: page,
            available: pages.len(),
        })?;

    let paths = paths_on_page(&doc, page as u32);
    let page_box = Rect::from_corners(0.0, 0.0, text_page.width, text_page.height);
    let labels = labels::assemble(&text_page.spans);

    Ok(digitise_page(&paths, &labels, page_box, page))
}

/// The pure core of the pipeline: geometry and text in, plots out.
///
/// Split out from I/O so it can be driven directly from synthesised inputs in
/// tests, and so that a future caller holding an already-parsed page (the
/// Document Engine, say) need not re-read the file.
pub fn digitise_page(
    paths: &[PaintedPath],
    labels: &[Label],
    page_box: Rect,
    page_number: usize,
) -> Vec<DigitisedPlot> {
    let spines = axes::spines(paths);
    axes::plot_regions(&spines, page_box)
        .into_iter()
        .filter_map(|region| digitise_region(paths, labels, region, page_number))
        .collect()
}

/// Digitise one plot region, or reject it as not a plot.
fn digitise_region(
    paths: &[PaintedPath],
    labels: &[Label],
    region: Rect,
    page_number: usize,
) -> Option<DigitisedPlot> {
    let mut warnings = Vec::new();

    let x_labels = axes::labels_in_band(labels, region, AxisKind::X);
    let y_labels = axes::labels_in_band(labels, region, AxisKind::Y);

    let x_ticks = axes::tick_positions(paths, region, AxisKind::X);
    let y_ticks = axes::tick_positions(paths, region, AxisKind::Y);

    let x_obs = axes::observe_ticks(&x_labels, &x_ticks, AxisKind::X);
    let y_obs = axes::observe_ticks(&y_labels, &y_ticks, AxisKind::Y);

    // A rectangle bounded by two long lines with no numbers anywhere near it is
    // not a plot -- it is a table, a framed box, a figure border. Rejecting it
    // outright (rather than returning an uncalibrated "plot") keeps the output
    // honest: we return plots, not every rectangle on the page.
    if x_obs.is_empty() && y_obs.is_empty() {
        return None;
    }

    // Titles live beyond the tick labels, so they get their own, wider band.
    let x_title_band = axes::title_band(labels, region, AxisKind::X);
    let y_title_band = axes::title_band(labels, region, AxisKind::Y);

    let (x_axis, x_warnings) = axes::fit_axis(
        x_obs,
        axes::axis_title(&x_title_band, region, AxisKind::X),
    );
    let (y_axis, y_warnings) = axes::fit_axis(
        y_obs,
        axes::axis_title(&y_title_band, region, AxisKind::Y),
    );
    warnings.extend(axes::rename_axis_warnings(x_warnings, AxisKind::X));
    warnings.extend(axes::rename_axis_warnings(y_warnings, AxisKind::Y));

    // A y-axis title is conventionally rotated, and text extraction does not
    // report rotation -- so it arrives as glyph confetti. Say so rather than
    // leaving the caller to wonder why a clearly-labelled axis has no title.
    for (band, name) in [(&x_title_band, "x-axis"), (&y_title_band, "y-axis")] {
        if axes::rotated_text_suspected(band) {
            warnings.push(format!(
                "{name}: the title text appears to be rotated. Text rotation is not recovered \
                 by the PDF text layer, so the title was not read. Tick VALUES are unaffected."
            ));
        }
    }

    warn_about_duplicate_ticks(&x_axis, "x-axis", &mut warnings);
    warn_about_duplicate_ticks(&y_axis, "y-axis", &mut warnings);

    let series = series::extract(paths, region, labels, &x_axis, &y_axis, &mut warnings);

    if series.is_empty() {
        warnings.push(
            "No data series were found inside the plot region. Everything drawn there looked \
             like axis furniture."
                .to_string(),
        );
    }

    // Type 3 / glyph-drawn markers arrive as text, never as paths, so a scatter
    // plot drawn that way digitises as an empty plot with no explanation unless
    // we name the symptom.
    let glyph_markers = series::marker_like_text(labels, region);
    if glyph_markers >= 4 {
        warnings.push(format!(
            "{glyph_markers} marker-like glyphs were found inside the plot region. Some \
             producers draw scatter markers as font glyphs rather than paths; those points \
             are NOT recoverable by this crate and are missing from the series above."
        ));
    }

    if !x_axis.is_calibrated() || !y_axis.is_calibrated() {
        warnings.push(
            "At least one axis could not be calibrated, so NO data values were produced. \
             Series carry their page-space geometry only."
                .to_string(),
        );
    }

    Some(DigitisedPlot {
        page: page_number,
        region,
        axes: AxisCalibration {
            x: x_axis,
            y: y_axis,
        },
        series,
        warnings,
    })
}

/// A tick value appearing twice on one axis almost always means a label was
/// misread (or two labels were wrongly merged), and it will drag the fit.
fn warn_about_duplicate_ticks(axis: &Axis, name: &str, warnings: &mut Vec<String>) {
    let dupes = axes::duplicate_values(&axis.ticks);
    if !dupes.is_empty() {
        warnings.push(format!(
            "{name}: the value(s) {dupes:?} appear on more than one tick. A tick label was \
             probably misread; the calibration may be distorted."
        ));
    }
}

/// Summarise how a plot's calibration was obtained, for a CSV provenance header
/// or a CLI report.
pub fn describe_calibration(axis: &Axis, name: &str) -> String {
    let scale = match axis.scale {
        axes::AxisScale::Linear => "linear",
        axes::AxisScale::Log10 => "log10",
    };
    match &axis.fit {
        None => format!("{name}: UNCALIBRATED ({} tick(s) found)", axis.ticks.len()),
        Some(fit) => {
            let from_marks = axis
                .ticks
                .iter()
                .filter(|t| t.source == TickSource::TickMark)
                .count();
            format!(
                "{name}: {scale}, fitted from {} tick(s) ({from_marks} from tick marks), \
                 normalised residual {:.2e}",
                axis.ticks.len(),
                fit.residual_normalised
            )
        }
    }
}
