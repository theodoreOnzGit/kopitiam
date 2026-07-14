//! Separating the data curves from the furniture, and from each other.
//!
//! # What is a series
//!
//! Once the plot region and the axes are known, everything else drawn inside
//! the region is either **data** or **furniture** (grid lines, the frame, tick
//! marks, the legend). The data is what we want; the furniture would corrupt it.
//!
//! Series are separated from one another by graphics state -- colour, line
//! width, dash -- for the reason given in [`crate::style`]: that is how the
//! figure was authored and how a reader tells them apart. Grouping paths by
//! their style key is therefore not a heuristic but a reconstruction of the
//! author's own intent.
//!
//! # The legend is not an optional extra
//!
//! It is tempting to treat legend recognition as a nice-to-have that fills in
//! `Series::label`. It is not: **a legend key is drawn in exactly the style of
//! the series it describes**, which is the whole point of a legend. So a
//! digitiser that ignores legends does not merely miss the labels -- it silently
//! appends the legend's little sample line to the series' own point list, as
//! two spurious data points sitting wherever the legend box happens to be.
//!
//! So legend detection is *required for correctness*, and it earns the labels as
//! a by-product.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::axes::Axis;
use crate::content::{PaintedPath, Subpath};
use crate::geometry::{Point, Rect};
use crate::labels::Label;
use crate::style::{SeriesStyle, StyleKey};

/// How a series was drawn, which determines how its points were recovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeriesKind {
    /// A stroked polyline. Its vertices are the data points, exactly.
    Line,
    /// Repeated small marks, one per data point. Each marker's centre is the
    /// data point.
    Scatter,
}

/// One recovered data point, carrying its own provenance.
///
/// `page_xy` is not a debugging aid. It is the evidence for `x` and `y`: with
/// it, plus the plot's [`crate::DigitisedPlot::axes`], any consumer can
/// re-derive this point independently and check our arithmetic, or go back to
/// the printed figure and put a ruler on it. A digitised value that cannot be
/// traced back to a position on the page is a number with no provenance, and
/// CLAUDE.md's Scientific Standards do not permit those.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DataPoint {
    pub x: f64,
    pub y: f64,
    /// The page coordinate, in PDF user space, that `(x, y)` was mapped from.
    pub page_xy: (f32, f32),
}

/// A recovered data series.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Series {
    /// The data. **Empty when the plot's axes could not be calibrated** -- an
    /// uncalibrated axis means these numbers are unknowable, and inventing them
    /// would be worse than returning nothing.
    pub points: Vec<DataPoint>,
    /// The raw page-space geometry, always present even when `points` is not.
    /// This is the evidence the digitisation rests on, and it survives a failed
    /// calibration so a caller can still see *that* a curve was found.
    pub page_points: Vec<(f32, f32)>,
    pub style: SeriesStyle,
    pub kind: SeriesKind,
    /// The legend text for this series, when a legend key in this exact style
    /// was found next to it.
    pub label: Option<String>,
    /// Whether the source path contained Bézier segments, meaning the recovered
    /// anchors may be spline knots rather than the plotted data. See
    /// [`crate::content`].
    pub interpolated: bool,
}

/// A subpath together with the style it was painted in.
struct Candidate<'a> {
    subpath: &'a Subpath,
    style: &'a SeriesStyle,
}

/// Extract the data series from a plot region.
///
/// `region` is the plot area; `labels` are the page's text labels, used for
/// legend matching. Warnings are appended to `warnings`.
pub fn extract(
    paths: &[PaintedPath],
    region: Rect,
    labels: &[Label],
    x_axis: &Axis,
    y_axis: &Axis,
    warnings: &mut Vec<String>,
) -> Vec<Series> {
    // Everything inside the region that is not structural furniture.
    let candidates: Vec<Candidate> = paths
        .iter()
        .flat_map(|p| {
            p.subpaths
                .iter()
                .map(move |s| Candidate {
                    subpath: s,
                    style: &p.style,
                })
        })
        .filter(|c| inside(c.subpath, region))
        .filter(|c| !is_furniture(c.subpath, region))
        .collect();

    // Group by visual identity. A BTreeMap (not a HashMap) because series
    // indices appear in the CSV export, and a non-deterministic order would make
    // two digitisations of the same PDF diff against each other.
    let mut groups: BTreeMap<StyleKey, Vec<&Candidate>> = BTreeMap::new();
    for c in &candidates {
        groups.entry(c.style.key()).or_default().push(c);
    }

    // Find legend keys across all groups first, so the legend *region* is known
    // before any group tries to turn its subpaths into data.
    let legend = find_legend(&groups, labels, region);
    if let Some(area) = legend.area {
        warnings.push(format!(
            "A legend was detected at ({:.0}, {:.0})-({:.0}, {:.0}); its sample lines and \
             markers were excluded from the data.",
            area.x,
            area.y,
            area.right(),
            area.top()
        ));
    }

    let mut series = Vec::new();
    for (key, members) in &groups {
        // Drop anything sitting inside the legend: it is a sample of the
        // series, not a measurement from it.
        let data: Vec<&&Candidate> = members
            .iter()
            .filter(|c| {
                legend
                    .area
                    .is_none_or(|area| !area.contains(c.subpath.bbox().center()))
            })
            .collect();
        if data.is_empty() {
            continue;
        }

        let style = data[0].style.clone();
        let label = legend.labels.get(key).cloned();

        // Split the group by size before deciding what it is. One style can
        // legitimately carry both kinds of thing at once -- a figure that plots
        // measured points *and* the correlation fitted through them will often
        // draw both in the same black -- and demanding that a group be wholly
        // one or wholly the other would throw away whichever kind is outnumbered.
        let max_extent = 0.05 * region.width.min(region.height);
        let (marks, strokes): (Vec<_>, Vec<_>) = data.iter().partition(|c| {
            let b = c.subpath.bbox();
            b.width <= max_extent && b.height <= max_extent
        });

        if let Some(s) = as_scatter(
            &marks,
            &style,
            label.clone(),
            x_axis,
            y_axis,
            warnings,
        ) {
            series.push(s);
        }

        if strokes.is_empty() {
            continue;
        }

        // A line series -- but the subpaths sharing this style may be one curve
        // drawn in pieces, or several curves drawn alike. Chaining tells them
        // apart; see `chain_subpaths`.
        let chains = chain_subpaths(&strokes);

        if chains.len() > 1 {
            warnings.push(format!(
                "{} disconnected curves share the style '{}'. They have been reported as \
                 {} separate series, but nothing in the figure distinguishes them, so they \
                 cannot be labelled -- and if they are really one series interrupted by gaps \
                 (missing data, a break in the axis), they have been split in two.",
                chains.len(),
                style.describe(),
                chains.len()
            ));
        }

        for chain in chains {
            let mut page_points = Vec::new();
            let mut interpolated = false;
            for c in &chain {
                interpolated |= c.subpath.has_curves();
                page_points.extend(c.subpath.anchors().iter().map(|p| (p.x, p.y)));
            }
            if page_points.len() < 2 {
                continue;
            }
            if interpolated {
                warnings.push(format!(
                    "Series '{}' contains Bezier segments. Only the on-curve anchor points are \
                     reported -- if the producer drew a smoothed spline, these are the data; if \
                     it drew an analytic curve, they are only its knots.",
                    style.describe()
                ));
            }

            series.push(build(
                page_points,
                style.clone(),
                SeriesKind::Line,
                label.clone(),
                interpolated,
                x_axis,
                y_axis,
            ));
        }
    }

    series
}

/// Group a style's subpaths into connected chains.
///
/// # Why this is not optional
///
/// Plenty of real producers -- this was found in a published journal figure, and
/// it is characteristic of chart engines that emit one path per line segment --
/// draw a single curve as *hundreds* of separate two-point subpaths, each
/// starting exactly where the last one ended. Taken at face value that is "171
/// paths share one style", which is indistinguishable, by style alone, from a
/// figure carrying 171 distinct curves all drawn in the same black.
///
/// The geometry resolves it. Pieces of one polyline **join end-to-start**;
/// genuinely separate curves do not. So consecutive subpaths are chained while
/// each begins where its predecessor finished, and each chain is one series.
///
/// The remaining ambiguity is honest and is warned about: two curves in an
/// identical style that happen not to touch are reported as two series with no
/// way to label them -- which is exactly the position a human reader is in,
/// looking at the same figure without a legend.
fn chain_subpaths<'a>(data: &[&&'a Candidate<'a>]) -> Vec<Vec<&'a Candidate<'a>>> {
    /// Two ends are "the same point" within this many page units. Producers
    /// round coordinates, so an exact match is too strict.
    const JOIN_TOLERANCE: f32 = 0.05;

    let mut chains: Vec<Vec<&Candidate>> = Vec::new();
    let mut last_end: Option<Point> = None;

    for c in data {
        let anchors = c.subpath.anchors();
        let (Some(first), Some(end)) = (anchors.first().copied(), anchors.last().copied()) else {
            continue;
        };

        let joins = last_end.is_some_and(|prev| {
            (prev.x - first.x).abs() <= JOIN_TOLERANCE && (prev.y - first.y).abs() <= JOIN_TOLERANCE
        });

        match chains.last_mut() {
            Some(chain) if joins => chain.push(c),
            _ => chains.push(vec![c]),
        }
        last_end = Some(end);
    }

    chains
}

/// Try to read a style group as a scatter series.
///
/// Scatter markers are many, small, and roughly identical. The "many" is what
/// makes this safe: a single small closed path is far more likely to be an
/// arrowhead or a piece of decoration than a one-point dataset, so the
/// threshold is deliberately set where a coincidence is implausible.
fn as_scatter(
    marks: &[&&Candidate],
    style: &SeriesStyle,
    label: Option<String>,
    x_axis: &Axis,
    y_axis: &Axis,
    warnings: &mut Vec<String>,
) -> Option<Series> {
    // Enough repetitions that a coincidence is implausible. A lone small closed
    // path is far more likely to be an arrowhead or a bit of decoration than a
    // one-point dataset.
    const MIN_MARKERS: usize = 4;

    // One data point per *cluster* of small strokes, not per stroke. See
    // `cluster_marks`.
    let clusters = cluster_marks(marks);
    if clusters.len() < MIN_MARKERS {
        return None;
    }
    let composite = clusters.iter().any(|c| c.len() > 1);

    let page_points: Vec<(f32, f32)> = clusters
        .iter()
        .map(|cluster| mark_position(cluster))
        .collect();

    if composite {
        warnings.push(format!(
            "Series '{}': {} marks were grouped into {} data point(s) -- each point is drawn \
             from several strokes, which is what a marker with error bars looks like. The \
             point is taken from the vertex its strokes share (the centre of the bars), and \
             the ERROR BAR MAGNITUDES ARE NOT RECOVERED -- only the central value.",
            style.describe(),
            marks.len(),
            clusters.len()
        ));
    }

    Some(build(
        page_points,
        style.clone(),
        SeriesKind::Scatter,
        label,
        false,
        x_axis,
        y_axis,
    ))
}

/// Group small marks into clusters, one per data point.
///
/// # Why one stroke is not one point
///
/// A scatter point in an experimental figure is very often not a single mark. In
/// a real published figure, each measured point is drawn as **eight separate line
/// segments** -- a horizontal error bar with two end caps, a vertical error bar
/// with two end caps -- all crossing at the measured value. Treating each stroke
/// as a data point would report 168 "measurements" where there were 21, and
/// scatter them around the true values at the ends of the error bars. That is a
/// fabricated dataset, and it would look entirely plausible.
///
/// Strokes belonging to one point overlap each other; strokes belonging to
/// different points do not. So marks are clustered by bounding-box adjacency,
/// and each cluster is one data point.
///
/// A plain single-stroke marker forms a cluster of one, so this changes nothing
/// for the ordinary case.
fn cluster_marks<'a>(data: &[&&'a Candidate<'a>]) -> Vec<Vec<&'a Candidate<'a>>> {
    /// Strokes of one marker touch or overlap; a point of slack absorbs rounding.
    const ADJACENCY: f32 = 1.0;

    let boxes: Vec<Rect> = data.iter().map(|c| c.subpath.bbox().padded(ADJACENCY)).collect();
    let mut assigned: Vec<Option<usize>> = vec![None; data.len()];
    let mut clusters: Vec<Vec<&Candidate>> = Vec::new();

    for i in 0..data.len() {
        if assigned[i].is_some() {
            continue;
        }
        let id = clusters.len();
        clusters.push(Vec::new());

        // Flood-fill from this mark through everything it touches.
        let mut stack = vec![i];
        assigned[i] = Some(id);
        while let Some(j) = stack.pop() {
            clusters[id].push(data[j]);
            for k in 0..data.len() {
                if assigned[k].is_none() && boxes[j].intersection_area(&boxes[k]) > 0.0 {
                    assigned[k] = Some(id);
                    stack.push(k);
                }
            }
        }
    }

    clusters
}

/// The data point a cluster of marks stands for.
///
/// For a marker with error bars, the four bar arms all *start* at the measured
/// value, so that vertex appears as an endpoint several times over while every
/// other endpoint appears once. Taking the most-shared vertex therefore recovers
/// the measurement **exactly**, and -- importantly -- it stays exact when the
/// error bars are asymmetric, where the cluster's bounding-box centre would not
/// be.
///
/// With no shared vertex (a plain marker: a circle, a filled square) the
/// bounding-box centre is the right answer and is used instead.
fn mark_position(cluster: &[&Candidate]) -> (f32, f32) {
    /// Vertices within this distance are the same vertex; producers round.
    const SAME_VERTEX: f32 = 0.1;

    let endpoints: Vec<Point> = cluster
        .iter()
        .flat_map(|c| {
            let a = c.subpath.anchors();
            // Only genuine ends of a stroke, not every vertex along it.
            match (a.first().copied(), a.last().copied()) {
                (Some(f), Some(l)) => vec![f, l],
                _ => vec![],
            }
        })
        .collect();

    let best = endpoints
        .iter()
        .map(|p| {
            let shared = endpoints
                .iter()
                .filter(|q| (q.x - p.x).abs() <= SAME_VERTEX && (q.y - p.y).abs() <= SAME_VERTEX)
                .count();
            (shared, p)
        })
        .max_by_key(|(shared, _)| *shared);

    // Three strokes meeting at one vertex is well past coincidence; two is not
    // (any polyline drawn in pieces has that), so the bar has to be set above it.
    match best {
        Some((shared, p)) if shared >= 3 => (p.x, p.y),
        _ => {
            let bbox = cluster
                .iter()
                .map(|c| c.subpath.bbox())
                .reduce(|acc, b| acc.union(&b))
                .unwrap_or(Rect::from_corners(0.0, 0.0, 0.0, 0.0));
            let c = bbox.center();
            (c.x, c.y)
        }
    }
}

/// Assemble a series, mapping page coordinates through the calibration.
///
/// If either axis is uncalibrated, `points` is left empty. This is the crate's
/// central promise: **no calibration, no numbers.**
fn build(
    page_points: Vec<(f32, f32)>,
    style: SeriesStyle,
    kind: SeriesKind,
    label: Option<String>,
    interpolated: bool,
    x_axis: &Axis,
    y_axis: &Axis,
) -> Series {
    let points = page_points
        .iter()
        .filter_map(|&(px, py)| {
            Some(DataPoint {
                x: x_axis.to_data(px)?,
                y: y_axis.to_data(py)?,
                page_xy: (px, py),
            })
        })
        .collect();

    Series {
        points,
        page_points,
        style,
        kind,
        label,
        interpolated,
    }
}

/// A subpath counts as being in the region if its centre is, and it does not
/// sprawl far outside it.
///
/// Centre-based rather than fully-contained, because a data curve that runs off
/// the edge of the axes is clipped by the renderer but its *path* may still
/// extend beyond -- and discarding it would lose the series entirely.
fn inside(sub: &Subpath, region: Rect) -> bool {
    let b = sub.bbox();
    region.padded(2.0).contains(b.center())
}

/// Whether a subpath is structural furniture rather than data.
///
/// Identified by **geometry, not colour**. It is tempting to say "grid lines are
/// grey, data is coloured", but plenty of engineering figures plot data in plain
/// black on a black-ruled grid, and a colour rule would throw the data away.
/// Geometry is the reliable signal: furniture is axis-aligned and spans the
/// plot (a spine or grid line), or is a tiny stub attached to a spine (a tick).
fn is_furniture(sub: &Subpath, region: Rect) -> bool {
    if sub.has_curves() {
        return false;
    }
    if is_plot_frame(sub, region) {
        return true;
    }
    let segments = sub.line_segments();
    // Beyond the frame, furniture is one straight stroke. A polyline with
    // several vertices is data.
    if segments.len() != 1 {
        return false;
    }
    let (a, b) = segments[0];
    let dx = (b.x - a.x).abs();
    let dy = (b.y - a.y).abs();

    let horizontal = dy <= 0.2 && dx > 0.2;
    let vertical = dx <= 0.2 && dy > 0.2;
    if !horizontal && !vertical {
        return false;
    }

    // A grid line or spine spans essentially the whole plot.
    let spans_plot = (horizontal && dx >= 0.9 * region.width)
        || (vertical && dy >= 0.9 * region.height);

    // A tick mark is a short stub touching a spine.
    let touches_spine = if horizontal {
        (a.x - region.x).abs() <= 2.0
            || (a.x - region.right()).abs() <= 2.0
            || (b.x - region.x).abs() <= 2.0
            || (b.x - region.right()).abs() <= 2.0
    } else {
        (a.y - region.y).abs() <= 2.0
            || (a.y - region.top()).abs() <= 2.0
            || (b.y - region.y).abs() <= 2.0
            || (b.y - region.top()).abs() <= 2.0
    };
    let is_stub = (horizontal && dx <= 0.1 * region.width)
        || (vertical && dy <= 0.1 * region.height);

    spans_plot || (is_stub && touches_spine)
}

/// Whether a subpath is the plot's own frame: an axis-aligned box tracing the
/// region itself.
///
/// # Why this needs its own case
///
/// A boxed plot frame is usually drawn as *one* closed subpath (`m l l l h`, or
/// a `re`), not as four separate strokes. The single-segment furniture test
/// therefore sails straight past it -- it has four segments, so it "must be a
/// polyline", so it is data. Run against a real conference paper, that is
/// exactly what happened: every figure gained a phantom five-point series
/// tracing its own border, sitting in the output looking like a measurement.
///
/// The frame is identified by what it is: every segment axis-aligned, and a
/// bounding box that *is* the plot region.
fn is_plot_frame(sub: &Subpath, region: Rect) -> bool {
    let segments = sub.line_segments();
    if segments.len() < 3 {
        return false;
    }
    let axis_aligned = segments.iter().all(|(a, b)| {
        let dx = (b.x - a.x).abs();
        let dy = (b.y - a.y).abs();
        dx <= 0.2 || dy <= 0.2
    });
    if !axis_aligned {
        return false;
    }
    // Its box is the region's box, to within a couple of points.
    let b = sub.bbox();
    let tol = 0.02 * region.width.max(region.height);
    (b.x - region.x).abs() <= tol
        && (b.y - region.y).abs() <= tol
        && (b.right() - region.right()).abs() <= tol
        && (b.top() - region.top()).abs() <= tol
}

/// The legend: which styles it names, and the area it occupies.
#[derive(Default)]
struct Legend {
    labels: BTreeMap<StyleKey, String>,
    area: Option<Rect>,
}

/// Find legend keys: a short sample of a series' style with text to its right.
///
/// The "text immediately to the right, vertically centred on the sample" test is
/// what a legend *is*, typographically, and it is stable across producers
/// because it is a convention readers depend on.
fn find_legend(
    groups: &BTreeMap<StyleKey, Vec<&Candidate>>,
    labels: &[Label],
    region: Rect,
) -> Legend {
    let mut legend = Legend::default();

    for (key, members) in groups {
        for c in members {
            let b = c.subpath.bbox();
            // A legend sample is small: a short flat line, or a lone marker.
            let is_sample = b.width <= 0.25 * region.width
                && b.height <= 0.05 * region.height
                && c.subpath.anchors().len() <= 5;
            if !is_sample {
                continue;
            }

            let Some(text) = label_right_of(&b, labels, region) else {
                continue;
            };

            legend.labels.entry(key.clone()).or_insert(text.0);
            let entry = b.union(&text.1);
            legend.area = Some(match legend.area {
                Some(a) => a.union(&entry),
                None => entry,
            });
        }
    }

    // Pad so that a marker sitting just outside the tightest box still counts.
    legend.area = legend.area.map(|a| a.padded(3.0));
    legend
}

/// The text label immediately to the right of a legend sample, if any.
fn label_right_of(sample: &Rect, labels: &[Label], region: Rect) -> Option<(String, Rect)> {
    labels
        .iter()
        .filter(|l| {
            let gap = l.rect.x - sample.right();
            let vertically_aligned =
                (l.center().y - sample.center().y).abs() < 0.8 * l.font_size.max(1.0);
            // The text must start close to the sample's right edge...
            gap > -1.0 && gap < 0.12 * region.width && vertically_aligned
        })
        // ...and where several qualify, the nearest wins.
        .min_by(|a, b| {
            (a.rect.x - sample.right())
                .partial_cmp(&(b.rect.x - sample.right()))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|l| (l.text.clone(), l.rect))
}

/// Marker-like *text* inside a plot region.
///
/// Some producers draw scatter markers as glyphs of a Type 3 font rather than as
/// paths (matplotlib historically did exactly this). Those never reach
/// [`crate::content`] -- they are text, not geometry -- so a plot full of them
/// would digitise as "no series found", with no explanation. Detecting the
/// symptom lets us say what actually happened.
pub fn marker_like_text(labels: &[Label], region: Rect) -> usize {
    labels
        .iter()
        .filter(|l| {
            region.contains(l.center())
                && l.text.chars().count() <= 2
                && l.text.chars().all(|c| !c.is_alphanumeric())
                && !l.text.trim().is_empty()
        })
        .count()
}

/// Points in page space, for tests and diagnostics.
pub fn centres(subpaths: &[Subpath]) -> Vec<Point> {
    subpaths.iter().map(|s| s.bbox().center()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::Segment;
    use crate::style::{Paint, Rgb};

    fn region() -> Rect {
        Rect::from_corners(100.0, 150.0, 500.0, 600.0)
    }

    fn line(points: &[(f32, f32)]) -> Subpath {
        Subpath {
            start: Point::new(points[0].0, points[0].1),
            segments: points[1..]
                .iter()
                .map(|&(x, y)| Segment::Line(Point::new(x, y)))
                .collect(),
            closed: false,
        }
    }

    #[test]
    fn grid_lines_and_spines_are_furniture() {
        let r = region();
        // A full-width horizontal grid line.
        assert!(is_furniture(&line(&[(100.0, 300.0), (500.0, 300.0)]), r));
        // A full-height vertical grid line.
        assert!(is_furniture(&line(&[(200.0, 150.0), (200.0, 600.0)]), r));
    }

    #[test]
    fn tick_stubs_are_furniture() {
        let r = region();
        // A 4pt tick hanging below the bottom spine.
        assert!(is_furniture(&line(&[(200.0, 150.0), (200.0, 146.0)]), r));
    }

    #[test]
    fn a_data_polyline_is_not_furniture() {
        let r = region();
        // Multi-vertex: data, even though it starts on the axis.
        assert!(!is_furniture(
            &line(&[(100.0, 150.0), (200.0, 300.0), (300.0, 450.0)]),
            r
        ));
        // A short diagonal is not axis-aligned, so not furniture.
        assert!(!is_furniture(&line(&[(200.0, 200.0), (250.0, 260.0)]), r));
    }

    #[test]
    fn a_short_horizontal_segment_mid_plot_is_not_furniture() {
        // Floating in the middle of the plot, touching no spine: this is data
        // (a flat stretch of a curve), not a tick.
        let r = region();
        assert!(!is_furniture(&line(&[(250.0, 400.0), (270.0, 400.0)]), r));
    }

    #[test]
    fn the_plot_frame_is_furniture_not_a_five_point_series() {
        // A boxed frame drawn as one closed subpath. Found in a real paper: it
        // was being reported as a data series tracing the plot's own border.
        let r = region();
        let frame = Subpath {
            start: Point::new(100.0, 150.0),
            segments: vec![
                Segment::Line(Point::new(100.0, 600.0)),
                Segment::Line(Point::new(500.0, 600.0)),
                Segment::Line(Point::new(500.0, 150.0)),
                Segment::Line(Point::new(100.0, 150.0)),
            ],
            closed: false,
        };
        assert!(is_plot_frame(&frame, r));
        assert!(is_furniture(&frame, r));

        // ...but a four-vertex data polyline inside the plot is NOT a frame,
        // even though it also has several segments.
        let curve = line(&[
            (150.0, 200.0),
            (250.0, 300.0),
            (350.0, 250.0),
            (450.0, 400.0),
        ]);
        assert!(!is_plot_frame(&curve, r));
        assert!(!is_furniture(&curve, r));
    }

    #[test]
    fn an_error_bar_cross_collapses_to_its_measured_point() {
        // Eight strokes: a horizontal bar with caps, a vertical bar with caps,
        // all meeting at the measurement. This is exactly how many published
        // figures draw their experimental points, and treating each stroke
        // as a datum would invent eight measurements out of one.
        let (cx, cy) = (456.62_f32, 493.45_f32);
        let subpaths = [
            line(&[(cx, cy), (cx + 3.36, cy)]),                     // right arm
            line(&[(cx + 3.36, cy - 1.99), (cx + 3.36, cy + 1.99)]), // right cap
            line(&[(cx, cy), (cx - 3.36, cy)]),                     // left arm
            line(&[(cx - 3.36, cy - 1.99), (cx - 3.36, cy + 1.99)]), // left cap
            line(&[(cx, cy), (cx, cy + 1.45)]),                     // up arm
            line(&[(cx - 1.99, cy + 1.45), (cx + 1.99, cy + 1.45)]), // top cap
            line(&[(cx, cy), (cx, cy - 1.44)]),                     // down arm
            line(&[(cx - 1.99, cy - 1.44), (cx + 1.99, cy - 1.44)]), // bottom cap
        ];
        let style = SeriesStyle {
            paint: Paint::Stroke,
            color: Rgb::BLACK,
            line_width: 0.4,
            dash: vec![],
        };
        let candidates: Vec<Candidate> = subpaths
            .iter()
            .map(|s| Candidate {
                subpath: s,
                style: &style,
            })
            .collect();
        let refs: Vec<&Candidate> = candidates.iter().collect();
        let data: Vec<&&Candidate> = refs.iter().collect();

        // All eight strokes are one cluster...
        let clusters = cluster_marks(&data);
        assert_eq!(clusters.len(), 1, "one measurement, one cluster");

        // ...and the point recovered is the vertex the four arms share, exactly.
        let (px, py) = mark_position(&clusters[0]);
        assert!(
            (px - cx).abs() < 1e-4 && (py - cy).abs() < 1e-4,
            "expected the shared vertex ({cx}, {cy}), got ({px}, {py})"
        );
    }

    #[test]
    fn asymmetric_error_bars_still_give_the_measured_point() {
        // The bounding-box centre would be WRONG here -- the bars are lopsided,
        // so the box centre sits away from the measurement. The shared vertex is
        // still exactly right, which is why it is the rule.
        let (cx, cy) = (200.0_f32, 300.0_f32);
        let subpaths = [
            line(&[(cx, cy), (cx + 10.0, cy)]), // long right arm
            line(&[(cx, cy), (cx - 2.0, cy)]),  // short left arm
            line(&[(cx, cy), (cx, cy + 9.0)]),  // long up arm
            line(&[(cx, cy), (cx, cy - 1.0)]),  // short down arm
        ];
        let style = SeriesStyle {
            paint: Paint::Stroke,
            color: Rgb::BLACK,
            line_width: 0.4,
            dash: vec![],
        };
        let candidates: Vec<Candidate> = subpaths
            .iter()
            .map(|s| Candidate {
                subpath: s,
                style: &style,
            })
            .collect();
        let refs: Vec<&Candidate> = candidates.iter().collect();
        let data: Vec<&&Candidate> = refs.iter().collect();
        let clusters = cluster_marks(&data);
        let (px, py) = mark_position(&clusters[0]);
        assert!((px - cx).abs() < 1e-4 && (py - cy).abs() < 1e-4);

        // Confirm the naive box-centre really would have been wrong, so this
        // test is guarding something real.
        let bbox = clusters[0]
            .iter()
            .map(|c| c.subpath.bbox())
            .reduce(|a, b| a.union(&b))
            .unwrap();
        assert!((bbox.center().x - cx).abs() > 3.0);
    }

    #[test]
    fn plain_markers_are_one_cluster_each() {
        // A conventional scatter: separated single-stroke markers. Clustering
        // must not change the ordinary case.
        let subpaths: Vec<Subpath> = (0..5)
            .map(|i| {
                let x = 150.0 + 40.0 * i as f32;
                line(&[(x, 300.0), (x + 3.0, 303.0)])
            })
            .collect();
        let style = SeriesStyle {
            paint: Paint::Stroke,
            color: Rgb::BLACK,
            line_width: 0.5,
            dash: vec![],
        };
        let candidates: Vec<Candidate> = subpaths
            .iter()
            .map(|s| Candidate {
                subpath: s,
                style: &style,
            })
            .collect();
        let refs: Vec<&Candidate> = candidates.iter().collect();
        let data: Vec<&&Candidate> = refs.iter().collect();
        assert_eq!(cluster_marks(&data).len(), 5);
    }

    #[test]
    fn marker_like_text_is_counted() {
        let r = region();
        let labels: Vec<Label> = [(200.0, 300.0), (250.0, 350.0)]
            .iter()
            .map(|&(x, y)| Label {
                text: "\u{25cf}".to_string(),
                superscript: None,
                rect: Rect::from_corners(x, y, x + 4.0, y + 4.0),
                font_size: 5.0,
            })
            .collect();
        assert_eq!(marker_like_text(&labels, r), 2);
    }

    #[test]
    fn style_ordering_is_total_and_stable() {
        let style = |r: f32, b: f32| SeriesStyle {
            paint: Paint::Stroke,
            color: Rgb::new(r, 0.0, b),
            line_width: 1.0,
            dash: vec![],
        }
        .key();
        let red = style(1.0, 0.0);
        let blue = style(0.0, 1.0);
        // Blue sorts before red on the red channel. The point is not *what* the
        // order is, only that there is one and it does not vary between runs --
        // which is what keeps series indices stable in the CSV export.
        assert!(blue < red);
        assert_eq!(red.cmp(&red), std::cmp::Ordering::Equal);
    }
}
