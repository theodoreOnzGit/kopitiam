//! **Plot digitisation** -- recovering the DATA behind a graph printed in a PDF.
//!
//! ```no_run
//! use std::path::Path;
//!
//! for plot in kopitiam_plot::digitise(Path::new("paper.pdf"), 4)? {
//!     for w in &plot.warnings {
//!         eprintln!("! {w}");           // read these before trusting anything
//!     }
//!     print!("{}", kopitiam_plot::to_csv_with_provenance(&plot, "paper.pdf"));
//! }
//! # Ok::<(), kopitiam_plot::PlotError>(())
//! ```
//!
//! # Why this crate exists
//!
//! Across scientific and technical literature generally, a paper's *figure* is
//! very often the only published form of a result. The numbers behind it were
//! never released, the author has moved on, and the lab notebook is gone. But
//! you cannot validate an analysis against a picture. You can only validate it
//! against numbers.
//!
//! So the figure has to become numbers again. That is all this crate does.
//!
//! # It is not image processing
//!
//! The instinct is to reach for pixels: rasterise the page, segment by colour,
//! trace the curve. Almost every existing plot digitiser works that way, and it
//! is a poor bargain -- it throws away information the file is still holding.
//!
//! A PDF plot is **vector graphics**. The data curve is a path, the axes are
//! paths, the tick labels are text. The producer wrote the data points into the
//! file as coordinates and they are still sitting there, transformed by an
//! affine map we can recover exactly. So we read the content stream, take the
//! path anchors, work out the map, and invert it. No rasterisation, no
//! anti-aliasing heuristics, no colour segmentation, and recovery is limited
//! only by the precision the producer wrote its coordinates at -- in practice
//! five or six significant figures.
//!
//! [`content`] is the module that makes this possible, and its docs explain the
//! extraction in detail.
//!
//! # Provenance is not decoration
//!
//! Every [`DataPoint`] carries `page_xy`: the page coordinate it was recovered
//! from. Every [`DigitisedPlot`] carries the [`AxisCalibration`] the point was
//! mapped through, *including the tick observations the calibration was fitted
//! from and the text printed on them*. Together these mean any recovered number
//! can be traced back to a position on the page and the tick labels that gave it
//! meaning -- and therefore checked, by a person, against the printed figure.
//!
//! This is CLAUDE.md's Scientific Standards requirement ("scientific software
//! should always remain explainable") and it is not a nicety. A digitised
//! validation dataset is going to be used to judge whether a solver is correct.
//! A number in that dataset that cannot be audited is worthless, because nobody
//! can tell whether a disagreement between code and data is the code's fault.
//!
//! # On being wrong
//!
//! The failure mode that matters here is not "it didn't work". It is **silent
//! confident wrongness**: a plausible-looking dataset that is quietly garbage,
//! because a log axis was read as linear, or two overlapping curves were merged,
//! or a legend's sample line was appended to a series as two extra points.
//! Someone will publish against that. A fabricated validation dataset is worse
//! than no dataset at all.
//!
//! So this crate is built to complain. [`DigitisedPlot::warnings`] is a
//! first-class output, not a log. It says so when:
//!
//! * an axis has too few labelled ticks to calibrate (and then **no data values
//!   are produced at all** -- see [`Series::points`]);
//! * an axis has exactly two ticks, which fit linear and log equally well, so
//!   the scale is genuinely undecidable from the figure;
//! * the calibration residual is worse than a clean vector plot should give;
//! * several paths share one style and may be one series or several;
//! * a curve is made of Béziers, so its anchors may be spline knots rather than
//!   data;
//! * marker-like glyphs were found that this crate cannot recover (see below).
//!
//! An empty `warnings` list is a positive claim, and it is meant to be trusted.
//!
//! # What this crate cannot do yet
//!
//! Stated plainly, because a digitiser that hides its blind spots is more
//! dangerous than one that has none:
//!
//! * **Raster figures.** A plot that is a scanned or embedded *image* carries no
//!   path geometry, and nothing here will recover it. There is no raster
//!   fallback. You will get no plot and no false confidence.
//! * **Glyph-drawn scatter markers.** A producer may draw markers as characters
//!   of a font rather than as paths. Those arrive as text and are not recovered.
//!   The symptom is detected and warned about, but the points are lost.
//! * **Error bar *magnitudes*.** A measured point drawn with error bars is
//!   recovered -- its central value is read exactly, from the vertex the bars
//!   cross at (see [`series`]) -- but the *lengths* of the bars, i.e. the stated
//!   uncertainty, are not. For validation work that is a real gap: the
//!   uncertainty is half the point of the measurement. It is warned about.
//! * **Fills, contours, heatmaps and shading.** Not modelled at all.
//! * **Rotated axis titles.** `kopitiam-pdf` reports a span's position but not
//!   its rotation, so a rotated title (the y-axis convention) arrives as
//!   scattered glyph fragments. It is detected, refused, and warned about rather
//!   than guessed at. Tick *values* are unaffected.
//! * **Two series drawn in an identical style** that do not touch. They are
//!   reported as separate series but cannot be labelled or told apart -- which is
//!   exactly the position a human reader is in, faced with the same figure and no
//!   legend.
//! * **Decimal-comma locales.** `1,000` is read as one thousand. If a figure
//!   came from a locale where that means one, every value on the axis is 1000x
//!   too large. It is warned about whenever it could apply.

pub mod axes;
pub mod content;
pub mod digitise;
pub mod export;
pub mod geometry;
pub mod knowledge;
pub mod labels;
pub mod series;
pub mod style;

pub use axes::{Axis, AxisFit, AxisScale, TickObservation, TickSource};
pub use digitise::{
    AxisCalibration, DigitisedPlot, PlotError, describe_calibration, digitise, digitise_bytes,
    digitise_page,
};
pub use export::{to_csv, to_csv_with_provenance};
pub use geometry::{Point, Rect};
pub use knowledge::to_entities;
pub use series::{DataPoint, Series, SeriesKind};
pub use style::{Paint, Rgb, SeriesStyle};
