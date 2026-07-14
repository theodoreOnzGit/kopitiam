//! Finding the axes and recovering the map from page coordinates to data
//! coordinates.
//!
//! This is where a digitiser earns or loses its trustworthiness. Every number
//! the crate eventually reports is the output of the calibration fitted here;
//! if the calibration is wrong, the series are wrong, and -- crucially -- they
//! are wrong *plausibly*, in a way that looks like data and will be believed.
//!
//! # The pipeline
//!
//! 1. Reduce every straight path segment to a [`Spine`] (a maximal horizontal
//!    or vertical run), merging collinear pieces so an axis drawn in fragments
//!    still reads as one line.
//! 2. Pair long horizontal and vertical spines that actually intersect: those
//!    are the corners of a plot frame. Each surviving pair is a candidate plot
//!    region. This generalises to subplots for free, which matters because
//!    multi-panel figures are the norm in the literature.
//! 3. Inside a region, find **tick marks** (short perpendicular segments
//!    touching a spine) and **tick labels** (numeric text in the margin band).
//! 4. Match each label to its tick, giving `(page coordinate, data value)`
//!    pairs.
//! 5. Fit the axis, choosing between linear and logarithmic.
//!
//! # Detecting a log axis, and why it is not optional
//!
//! Log axes are everywhere in technical figures -- frequency-response plots,
//! calibration curves, measured spectra, exponential trends. And a log axis
//! mistaken for a linear one does not fail loudly: it produces a smooth,
//! well-behaved, entirely fictitious dataset.
//!
//! The discrimination is straightforward once the tick pairs are in hand. Fit
//! `value = a*p + b` and also `log10(value) = a*p + b`, and compare residuals.
//! On a decade-spaced log axis (ticks at 0.001, 0.01, 0.1, 1, evenly spaced
//! down the page) the linear fit is badly curved and the log fit is exact; on a
//! linear axis the reverse. The winner is decided by *normalised* residual so
//! that the comparison is scale-free.
//!
//! The honest part is the edge case. **Two ticks fit any monotone model
//! perfectly**, linear and log alike, so with only two labelled ticks the
//! question is genuinely undecidable from the figure. We do not guess silently:
//! we assume linear (the more common case) and emit a warning saying exactly
//! that. Three ticks is the minimum for an evidence-based answer.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::content::PaintedPath;
use crate::geometry::{Point, Rect};
use crate::labels::Label;

/// Two points are "the same" within this many page units. A tick mark that
/// touches its spine is drawn exactly on it, but producers round coordinates.
const TOUCH_TOLERANCE: f32 = 1.5;

/// A straight run of ink, horizontal or vertical.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spine {
    pub horizontal: bool,
    /// The constant coordinate: `y` for a horizontal spine, `x` for a vertical.
    pub pos: f32,
    /// The extent along the spine's own direction.
    pub min: f32,
    pub max: f32,
}

impl Spine {
    pub fn length(&self) -> f32 {
        self.max - self.min
    }

    /// Whether the two spines cross (or touch), which is what makes them a
    /// plausible pair of axes rather than two unrelated rules on the page.
    fn intersects(&self, other: &Spine) -> bool {
        if self.horizontal == other.horizontal {
            return false;
        }
        let (h, v) = if self.horizontal {
            (self, other)
        } else {
            (other, self)
        };
        (h.min - TOUCH_TOLERANCE..=h.max + TOUCH_TOLERANCE).contains(&v.pos)
            && (v.min - TOUCH_TOLERANCE..=v.max + TOUCH_TOLERANCE).contains(&h.pos)
    }
}

/// Every axis-aligned straight segment on the page, *unmerged*.
///
/// Kept separate from [`spines`] because merging is right for finding axes and
/// wrong for finding ticks, and the difference is not obvious until it bites.
///
/// Consider the tick at the origin of a conventional plot. Its y-axis tick mark
/// points left, from x=96 to x=100, sitting at y=150. The x-axis spine runs from
/// x=100 to x=500, also at y=150. The two are **collinear and abutting** -- they
/// are, as ink on the page, one continuous horizontal line. So [`merge_collinear`]
/// quite correctly fuses them into a single 404pt run... and the origin tick,
/// now part of a line far too long to be a tick, vanishes.
///
/// The consequence is not a crash. It is that the axis calibrates from four
/// ticks instead of five, with the fifth silently downgraded to a label-centre
/// estimate a point or two off true -- which is exactly the sort of quiet,
/// plausible degradation this crate is supposed to refuse. Tick detection
/// therefore works from unmerged segments, where a 4pt stub is still a 4pt stub.
fn raw_segments(paths: &[PaintedPath]) -> Vec<Spine> {
    let mut raw: Vec<Spine> = Vec::new();
    for path in paths {
        // Only stroked paths lay down a line. A *filled* rectangle is a
        // background panel, not an axis, and treating its edges as spines is a
        // reliable way to mistake a shaded legend box for a plot frame.
        if !path.style.paint.strokes() {
            continue;
        }
        for sub in &path.subpaths {
            for (a, b) in sub.line_segments() {
                if let Some(s) = classify(a, b) {
                    raw.push(s);
                }
            }
        }
    }
    raw
}

/// Reduce a page's paths to their axis-aligned straight runs, merging collinear
/// fragments.
///
/// Merging matters here: plenty of producers draw an axis as several abutting
/// segments (one per tick interval, say), and an unmerged 40pt fragment would
/// never clear the "is this long enough to be an axis" bar. Use this for finding
/// axes and plot regions -- and [`raw_segments`] for finding ticks.
pub fn spines(paths: &[PaintedPath]) -> Vec<Spine> {
    merge_collinear(raw_segments(paths))
}

fn classify(a: Point, b: Point) -> Option<Spine> {
    let dx = (b.x - a.x).abs();
    let dy = (b.y - a.y).abs();
    // A "horizontal" segment is one whose vertical drift is negligible, not one
    // that is exactly level: PDF coordinates are rounded decimals.
    if dy <= 0.2 && dx > 0.2 {
        Some(Spine {
            horizontal: true,
            pos: (a.y + b.y) / 2.0,
            min: a.x.min(b.x),
            max: a.x.max(b.x),
        })
    } else if dx <= 0.2 && dy > 0.2 {
        Some(Spine {
            horizontal: false,
            pos: (a.x + b.x) / 2.0,
            min: a.y.min(b.y),
            max: a.y.max(b.y),
        })
    } else {
        None
    }
}

/// Join collinear runs that are fragments of the same line -- but *not* a tick
/// stub that merely abuts a spine.
///
/// # Why a length test is needed, and what goes wrong without it
///
/// The y-axis tick at the plot's origin points left, from x=96 to x=100, at
/// y=150. The x-axis spine runs from x=100 to x=500, also at y=150. They are
/// collinear and they abut exactly -- as ink on the page they *are* one
/// continuous horizontal line, and a naive collinear merge duly fuses them.
///
/// The damage is subtle and entirely silent. The merged "spine" now starts at
/// x=96, so the recovered plot region starts 4pt left of the true axis. Every
/// margin band is computed from that region, and a band shifted by 4pt is
/// exactly enough to let the *x-axis's* corner tick label fall inside the
/// *y-axis's* band -- where it is duly fitted as a y observation, at a page
/// coordinate that has nothing to do with its value. The calibration is then
/// wrong, the residual explodes, and the crate would have to fall back on
/// warning about a mess it made itself.
///
/// The distinction that resolves it: fragments of one line are of *comparable
/// length*; a tick stub is one percent of the spine it touches. So a much
/// shorter run that merely abuts is left alone, while genuinely fragmented axes
/// (producers that draw one segment per tick interval) still merge, because
/// their pieces are all of a size.
fn merge_collinear(mut spines: Vec<Spine>) -> Vec<Spine> {
    /// A run shorter than this fraction of its neighbour is a stub (a tick),
    /// not a fragment of the same line.
    const FRAGMENT_RATIO: f32 = 0.2;

    spines.sort_by(|a, b| {
        a.horizontal
            .cmp(&b.horizontal)
            .then(a.pos.partial_cmp(&b.pos).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.min.partial_cmp(&b.min).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut out: Vec<Spine> = Vec::new();
    for s in spines {
        match out.last_mut() {
            Some(prev)
                if prev.horizontal == s.horizontal
                    && (prev.pos - s.pos).abs() <= TOUCH_TOLERANCE
                    && s.min <= prev.max + TOUCH_TOLERANCE
                    && is_fragment_of(prev, &s, FRAGMENT_RATIO) =>
            {
                prev.max = prev.max.max(s.max);
            }
            _ => out.push(s),
        }
    }
    out
}

/// Whether two abutting collinear runs are plausibly pieces of one line.
///
/// True when they genuinely overlap (the same line drawn twice, or overlapping
/// fragments), or when the shorter is a decent fraction of the longer. False for
/// a tick stub touching the end of a spine.
fn is_fragment_of(a: &Spine, b: &Spine, ratio: f32) -> bool {
    let overlap = a.max.min(b.max) - b.min.max(a.min);
    if overlap > TOUCH_TOLERANCE {
        return true;
    }
    let (short, long) = {
        let (x, y) = (a.length(), b.length());
        (x.min(y), x.max(y))
    };
    long <= 0.0 || short >= ratio * long
}

/// Candidate plot regions: rectangles bounded by a long horizontal and a long
/// vertical spine that meet.
///
/// `page` is the page box, used only to set a scale-relative minimum size --
/// what counts as a "long" line on A4 is not what counts on a poster.
pub fn plot_regions(spines: &[Spine], page: Rect) -> Vec<Rect> {
    // A spine must be a decent fraction of the page to be an axis. This is the
    // main filter that keeps table rules, underlines and text decorations out.
    let min_h = 0.08 * page.width;
    let min_v = 0.08 * page.height;

    let long_h: Vec<&Spine> = spines
        .iter()
        .filter(|s| s.horizontal && s.length() >= min_h)
        .collect();
    let long_v: Vec<&Spine> = spines
        .iter()
        .filter(|s| !s.horizontal && s.length() >= min_v)
        .collect();

    let mut candidates: Vec<Rect> = Vec::new();
    for h in &long_h {
        for v in &long_v {
            if h.intersects(v) {
                // The region is the area the axes span: the horizontal spine
                // gives its width, the vertical its height. A boxed frame
                // yields the same rectangle from all four corner pairings,
                // which the de-duplication below collapses.
                candidates.push(Rect::from_corners(h.min, v.min, h.max, v.max));
            }
        }
    }

    // Largest first, then greedily drop anything substantially covered by a
    // region already kept. This collapses the duplicates a boxed frame
    // produces, and prefers the outer frame over an inset drawn within it.
    candidates.sort_by(|a, b| {
        b.area()
            .partial_cmp(&a.area())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let min_area = 0.01 * page.area();
    let mut kept: Vec<Rect> = Vec::new();
    for c in candidates {
        if c.area() < min_area {
            continue;
        }
        if kept.iter().any(|k| k.overlap_ratio(&c) > 0.5) {
            continue;
        }
        kept.push(c);
    }
    kept
}

/// Where a calibration point's page coordinate came from. Provenance, not
/// trivia: a calibration fitted from label centres is materially less accurate
/// than one fitted from tick marks, and a reader auditing a number is entitled
/// to know which they are looking at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TickSource {
    /// A tick mark was found on the spine, and the label was matched to it.
    /// The page coordinate is the tick's, which is exact.
    TickMark,
    /// No tick mark was found; the label's own centre was used. Plotting tools
    /// centre a tick label on its tick, so this is a good estimate -- but it is
    /// an estimate, subject to the label's glyph metrics and any rounding the
    /// producer applied.
    LabelCentre,
}

/// One `(page coordinate, data value)` pair: the raw evidence a calibration is
/// fitted from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickObservation {
    /// Position along the axis, in page units.
    pub page: f32,
    /// The value its label denotes.
    pub value: f64,
    /// The label text as printed, kept verbatim so a human can check our
    /// reading of it.
    pub text: String,
    pub source: TickSource,
}

/// Linear, or base-10 logarithmic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AxisScale {
    Linear,
    Log10,
}

/// The fitted map from page coordinate to data value, and how well it fitted.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AxisFit {
    /// `value = slope * page + intercept` (linear), or
    /// `log10(value) = slope * page + intercept` (log).
    pub slope: f64,
    pub intercept: f64,
    /// RMS residual in the fitted variable's own units.
    pub residual_rms: f64,
    /// `residual_rms` divided by the span of the fitted variable, making it
    /// dimensionless and comparable across axes. This is the number to look at
    /// when deciding whether to trust a digitisation: a clean vector plot fits
    /// to ~1e-6, and anything above ~1e-3 means something is wrong.
    pub residual_normalised: f64,
}

/// One axis of a plot: what we found, what we fitted, and what we could not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Axis {
    pub scale: AxisScale,
    pub ticks: Vec<TickObservation>,
    /// `None` when the axis could not be calibrated at all -- fewer than two
    /// labelled ticks, or a degenerate fit. When this is `None`, **no data
    /// values are produced for the plot**: an uncalibrated axis means the
    /// numbers are unknowable, and reporting them anyway is the failure this
    /// crate exists to prevent.
    pub fit: Option<AxisFit>,
    /// The axis title, if a non-numeric label was found in the margin band.
    /// Often carries the units, which is exactly the provenance a validation
    /// dataset needs.
    pub title: Option<String>,
}

impl Axis {
    pub fn is_calibrated(&self) -> bool {
        self.fit.is_some()
    }

    /// Map a page coordinate to a data value.
    pub fn to_data(&self, page: f32) -> Option<f64> {
        let fit = self.fit?;
        let v = fit.slope * page as f64 + fit.intercept;
        Some(match self.scale {
            AxisScale::Linear => v,
            AxisScale::Log10 => 10.0_f64.powf(v),
        })
    }
}

/// Which side of a plot region a margin band lies on.
#[derive(Clone, Copy, PartialEq)]
pub enum AxisKind {
    X,
    Y,
}

/// Find tick-mark positions along one axis of a region.
///
/// A tick is a *short* segment perpendicular to a spine, with an endpoint on
/// it. The length cap is what separates a tick from a grid line: both are
/// perpendicular to the axis and both cross it, but a grid line spans the plot
/// and a tick does not. Getting this wrong in the permissive direction turns
/// every grid line into a spurious tick at a position with no label, which is
/// harmless; getting it wrong in the other direction loses the calibration
/// entirely.
pub fn tick_positions(paths: &[PaintedPath], region: Rect, axis: AxisKind) -> Vec<f32> {
    // Unmerged: see `raw_segments` for why merging would eat the origin tick.
    let all = raw_segments(paths);
    // Ticks may hang off either the bottom or the top spine (or left/right); we
    // only care about the coordinate *along* the axis, which is the same either
    // way, so both are pooled.
    let (spine_positions, max_len) = match axis {
        AxisKind::X => (vec![region.y, region.top()], 0.25 * region.height),
        AxisKind::Y => (vec![region.x, region.right()], 0.25 * region.width),
    };

    let mut out: Vec<f32> = Vec::new();
    for s in &all {
        // A tick for the x-axis is a *vertical* segment, and vice versa.
        let wanted_horizontal = matches!(axis, AxisKind::Y);
        if s.horizontal != wanted_horizontal {
            continue;
        }
        let len = s.length();
        if len < 0.5 || len > max_len {
            continue;
        }
        // One end must touch a spine of this region...
        let touches = spine_positions.iter().any(|&sp| {
            (s.min - sp).abs() <= TOUCH_TOLERANCE || (s.max - sp).abs() <= TOUCH_TOLERANCE
        });
        if !touches {
            continue;
        }
        // ...and it must sit within the region's extent along the axis.
        let (lo, hi) = match axis {
            AxisKind::X => (region.x, region.right()),
            AxisKind::Y => (region.y, region.top()),
        };
        if s.pos < lo - TOUCH_TOLERANCE || s.pos > hi + TOUCH_TOLERANCE {
            continue;
        }
        out.push(s.pos);
    }

    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out.dedup_by(|a, b| (*a - *b).abs() < TOUCH_TOLERANCE);
    out
}

/// Labels lying in the margin band of one axis: below/above for x, left/right
/// for y.
///
/// # The corner-label trap
///
/// The obvious rule -- "an x tick label is anything below the plot" -- is wrong,
/// and wrong in a way that quietly poisons the calibration rather than failing.
///
/// Look at the bottom-left corner of any conventional plot. The x-axis label
/// `0.0` sits just below the corner; the y-axis label `0.0` sits just to its
/// left. They are a few points apart. A band defined only by "below the plot,
/// and roughly within its width" catches *both*, so the x-axis ends up fitting
/// six observations instead of five -- with the intruder at a page coordinate
/// that has nothing to do with the value it carries. The same happens at the
/// top-left corner, and again on the y-axis. The result is a distorted fit and a
/// pair of duplicated tick values.
///
/// The discriminator that actually works follows from how figures are drawn: an
/// x tick label is **vertically outside** the plot (below the bottom spine, or
/// above the top one) and **horizontally within** it. A y tick label is the
/// transpose. A corner label satisfies exactly one of these, never both, and the
/// ambiguity dissolves.
pub fn labels_in_band(labels: &[Label], region: Rect, axis: AxisKind) -> Vec<&Label> {
    // Tick labels hug their axis. Reaching further out than this starts sweeping
    // up the caption and the body text beneath the figure -- and a body-text
    // number ("Figure 5", a year, a citation) parses perfectly well and would be
    // fitted as a tick observation at a page coordinate that means nothing. A
    // tight band is the cheapest defence against that, and costs nothing real:
    // no producer places a tick label a fifth of the plot away from its axis.
    labels_in_band_within(labels, region, axis, 0.12)
}

/// The band-membership test, parameterised by how far out it reaches.
///
/// Tick labels use a tight reach ([`labels_in_band`]); the axis *title* sits
/// beyond them and needs a wider one ([`axis_title`]).
fn labels_in_band_within(
    labels: &[Label],
    region: Rect,
    axis: AxisKind,
    reach: f32,
) -> Vec<&Label> {
    labels
        .iter()
        .filter(|l| {
            let c = l.center();
            // "Outside" needs a clear margin rather than a hairline. A tick
            // label sits several points clear of the spine, so demanding a
            // point of daylight costs nothing -- and it avoids a label whose
            // centre lands *exactly* on the region boundary (the rightmost
            // x label centres precisely on the corner) flipping in or out of
            // the perpendicular axis's band on a floating-point tie.
            const MARGIN: f32 = 1.0;
            match axis {
                AxisKind::X => {
                    // Outside the plot vertically -- strictly, since a label
                    // that straddles the spine is not a tick label.
                    let outside = (c.y < region.y - MARGIN
                        && c.y > region.y - reach * region.height)
                        || (c.y > region.top() + MARGIN
                            && c.y < region.top() + reach * region.height);
                    // ...and within its horizontal span. This is what keeps the
                    // y-axis's corner labels, which live in the left margin, out
                    // of the x-axis's fit.
                    let within = c.x >= region.x - 0.02 * region.width
                        && c.x <= region.right() + 0.02 * region.width;
                    outside && within
                }
                AxisKind::Y => {
                    let outside = (c.x < region.x - MARGIN
                        && c.x > region.x - reach * region.width)
                        || (c.x > region.right() + MARGIN
                            && c.x < region.right() + reach * region.width);
                    let within = c.y >= region.y - 0.02 * region.height
                        && c.y <= region.top() + 0.02 * region.height;
                    outside && within
                }
            }
        })
        .collect()
}

/// Match numeric labels to tick marks, producing the calibration evidence.
///
/// Each label is matched to the nearest tick within a tolerance scaled to the
/// label's own size. A label with no tick nearby still contributes -- via its
/// own centre -- because plenty of figures label an axis without drawing tick
/// marks at all, and refusing to calibrate those would be needlessly strict.
/// The distinction is recorded in [`TickObservation::source`] rather than
/// hidden.
pub fn observe_ticks(
    labels: &[&Label],
    ticks: &[f32],
    axis: AxisKind,
) -> Vec<TickObservation> {
    let mut used: Vec<bool> = vec![false; ticks.len()];
    let mut out = Vec::new();

    for label in labels {
        let Some(value) = label.value() else {
            continue;
        };
        let centre = match axis {
            AxisKind::X => label.center().x,
            AxisKind::Y => label.center().y,
        };
        // Along the axis, a label sits within about its own width of its tick.
        // Across it, we have already filtered by band.
        let tolerance = match axis {
            AxisKind::X => (label.rect.width * 0.75).max(4.0),
            AxisKind::Y => (label.rect.height * 1.5).max(4.0),
        };

        let best = ticks
            .iter()
            .enumerate()
            .filter(|(i, _)| !used[*i])
            .map(|(i, &t)| (i, (t - centre).abs()))
            .filter(|(_, d)| *d <= tolerance)
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let (page, source) = match best {
            Some((i, _)) => {
                used[i] = true;
                (ticks[i], TickSource::TickMark)
            }
            None => (centre, TickSource::LabelCentre),
        };

        out.push(TickObservation {
            page,
            value,
            text: label.text.clone(),
            source,
        });
    }

    out.sort_by(|a, b| a.page.partial_cmp(&b.page).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Pick the axis title: the non-numeric, *centred* label sitting beyond the tick
/// labels. `None` when there isn't one, or when the text is rotated.
///
/// # Why centring is the test, not distance
///
/// The obvious rule -- "the non-numeric label furthest from the axis" -- picks up
/// the paragraph underneath the figure. Run against a real conference paper it
/// duly reported the x-axis title as `"will"`, a word from the body text below
/// the plot.
///
/// An axis title is *centred on its axis*; body text is not. That single
/// constraint separates them cleanly, and it holds across every producer because
/// it is a typographic convention readers rely on.
///
/// # Rotated titles
///
/// A y-axis title is conventionally rotated 90 degrees, and `kopitiam-pdf`
/// reports a span's position but not its rotation. Rotated text therefore arrives
/// as a stack of one- and two-glyph fragments at nearly the same x -- a real
/// paper's `"ΔP pool pressure (Pa)"` came through as `"∆"`, `"P"`, `"oo"`,
/// `"l"`, `"essur"` and a dozen more. Left alone, the "title" would be whichever
/// fragment happened to sit nearest the axis's midpoint (`"essur"`, as it
/// happened), and that junk would go straight into the knowledge graph as the
/// name of the entity.
///
/// So the fragmentation is detected and the title is refused. `None` is the
/// honest answer, and [`rotated_text_suspected`] lets the caller say why.
pub fn axis_title(labels: &[&Label], region: Rect, axis: AxisKind) -> Option<String> {
    if rotated_text_suspected(labels) {
        return None;
    }

    let (centre, extent) = match axis {
        AxisKind::X => (region.center().x, region.width),
        AxisKind::Y => (region.center().y, region.height),
    };

    labels
        .iter()
        .filter(|l| {
            // Not a number, and actually words.
            l.value().is_none() && l.text.chars().filter(|c| c.is_alphabetic()).count() >= 2
        })
        .filter(|l| {
            // Centred on the axis it titles.
            let along = match axis {
                AxisKind::X => l.center().x,
                AxisKind::Y => l.center().y,
            };
            (along - centre).abs() < 0.3 * extent
        })
        // Among the centred candidates, the title is the one printed furthest
        // out -- beyond the tick labels it describes.
        .max_by(|a, b| {
            let d = |l: &Label| match axis {
                AxisKind::X => (region.y - l.center().y).abs(),
                AxisKind::Y => (region.x - l.center().x).abs(),
            };
            d(a).partial_cmp(&d(b)).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|l| l.text.clone())
}

/// Whether a band's text looks like it was rotated.
///
/// Rotated text has an unmistakable fingerprint once it has been through a text
/// extractor that does not report rotation: many very short non-numeric labels,
/// each barely wider than one glyph, stacked at nearly the same position across
/// the axis. Ordinary horizontal text in a margin band does not look like that.
pub fn rotated_text_suspected(labels: &[&Label]) -> bool {
    const MIN_FRAGMENTS: usize = 4;
    labels
        .iter()
        .filter(|l| {
            l.value().is_none()
                && !l.text.trim().is_empty()
                // A fragment is barely wider than a single glyph.
                && l.rect.width < 1.5 * l.font_size.max(1.0)
        })
        .count()
        >= MIN_FRAGMENTS
}

/// The wider band an axis *title* may live in -- beyond the tick labels.
pub fn title_band(labels: &[Label], region: Rect, axis: AxisKind) -> Vec<&Label> {
    labels_in_band_within(labels, region, axis, 0.30)
}

/// Fit an axis to its tick observations, choosing linear or logarithmic.
///
/// Returns the axis plus any warnings the fit raised. Warnings are returned
/// rather than logged because they must reach the caller: a plot whose
/// calibration is shaky is still worth reporting, but only if the shakiness
/// travels with it.
pub fn fit_axis(ticks: Vec<TickObservation>, title: Option<String>) -> (Axis, Vec<String>) {
    let mut warnings = Vec::new();
    let name = "axis";

    // Distinct positions, not distinct labels: two labels matched to the same
    // page coordinate carry no more information than one, and would make a
    // degenerate fit look well-determined.
    let distinct: Vec<&TickObservation> = {
        let mut seen: Vec<f32> = Vec::new();
        ticks
            .iter()
            .filter(|t| {
                if seen.iter().any(|&p| (p - t.page).abs() < 0.5) {
                    false
                } else {
                    seen.push(t.page);
                    true
                }
            })
            .collect()
    };

    if distinct.len() < 2 {
        warnings.push(format!(
            "{name}: only {} labelled tick(s) found; cannot calibrate. No data values \
             were produced for this axis.",
            distinct.len()
        ));
        return (
            Axis {
                scale: AxisScale::Linear,
                ticks,
                fit: None,
                title,
            },
            warnings,
        );
    }

    let pages: Vec<f64> = distinct.iter().map(|t| t.page as f64).collect();
    let values: Vec<f64> = distinct.iter().map(|t| t.value).collect();

    let linear = least_squares(&pages, &values);
    // A log fit is only meaningful for strictly positive values. A single
    // non-positive tick (a `0` on an otherwise-log axis is impossible, but a
    // mis-parse could produce one) rules it out.
    let log_values: Option<Vec<f64>> = values
        .iter()
        .map(|v| (*v > 0.0).then(|| v.log10()))
        .collect();
    let log = log_values.as_ref().and_then(|lv| least_squares(&pages, lv));

    // Two observations fit any two-parameter model exactly, so a 2-tick axis has
    // a zero residual by construction and no redundancy whatsoever: a misread
    // label cannot be detected, because there is nothing to check it against. Say
    // so regardless of which scale we end up choosing.
    if distinct.len() < 3 {
        warnings.push(format!(
            "{name}: calibrated from only 2 labelled ticks. The fit is exact by construction \
             and its zero residual means nothing -- there is no redundancy, so a misread tick \
             label cannot be detected."
        ));
    }

    let (scale, fit) = match (linear, log) {
        (Some(lin), Some(lg)) => {
            if distinct.len() < 3 {
                // ...and when a log fit is also possible, the *scale* itself is
                // undecidable, which is far more dangerous than a little noise.
                warnings.push(format!(
                    "{name}: linear and logarithmic scales BOTH fit these 2 ticks exactly, so \
                     the scale could not be determined from the figure. ASSUMED LINEAR -- if \
                     this axis is logarithmic, every value is wrong."
                ));
                (AxisScale::Linear, lin)
            } else if lg.residual_normalised < lin.residual_normalised * 0.1 {
                (AxisScale::Log10, lg)
            } else if lin.residual_normalised < lg.residual_normalised * 0.1 {
                (AxisScale::Linear, lin)
            } else {
                // Neither model is a decisive winner. That is itself
                // information: something about these ticks is not a clean axis.
                warnings.push(format!(
                    "{name}: neither a linear nor a logarithmic fit is clearly better \
                     (normalised residuals {:.2e} vs {:.2e}). The tick labels may have been \
                     misread. Assuming the better of the two.",
                    lin.residual_normalised, lg.residual_normalised
                ));
                if lg.residual_normalised < lin.residual_normalised {
                    (AxisScale::Log10, lg)
                } else {
                    (AxisScale::Linear, lin)
                }
            }
        }
        (Some(lin), None) => (AxisScale::Linear, lin),
        (None, Some(lg)) => (AxisScale::Log10, lg),
        (None, None) => {
            warnings.push(format!(
                "{name}: tick marks are degenerate (all at the same page coordinate, or all \
                 the same value); cannot calibrate."
            ));
            return (
                Axis {
                    scale: AxisScale::Linear,
                    ticks,
                    fit: None,
                    title,
                },
                warnings,
            );
        }
    };

    // A clean vector plot fits to machine precision. Anything appreciably worse
    // means a tick was mismatched, a label misread, or the axis is not the
    // scale we think it is.
    if fit.residual_normalised > 1e-2 {
        warnings.push(format!(
            "{name}: calibration fit is POOR (normalised residual {:.2e}). The recovered \
             values are probably wrong. Check the tick labels.",
            fit.residual_normalised
        ));
    } else if fit.residual_normalised > 1e-3 {
        warnings.push(format!(
            "{name}: calibration residual is high ({:.2e}) for a vector plot; treat the \
             recovered values with caution.",
            fit.residual_normalised
        ));
    }

    // Comma digit grouping is read as thousands (see `labels::strip_digit_grouping`).
    // That is right for the English-language scientific literature, but if the
    // figure came from a decimal-comma locale every value is 1000x too large --
    // and that is a catastrophic, entirely silent error. It fires only on axes
    // that actually carry grouped labels, so it does not dilute the other
    // warnings.
    if distinct.iter().any(|t| t.text.contains(',')) {
        warnings.push(format!(
            "{name}: tick labels use ',' as a digit-group separator (e.g. \"1,000\" read as \
             1000). If this figure uses the European decimal comma instead, every value on \
             this axis is 1000x too large. The labels are preserved verbatim in the tick \
             observations -- check them."
        ));
    }

    let from_labels = distinct
        .iter()
        .filter(|t| t.source == TickSource::LabelCentre)
        .count();
    if from_labels > 0 {
        warnings.push(format!(
            "{name}: {from_labels} of {} calibration points came from tick-label centres \
             rather than tick marks (no tick mark was found nearby). Accuracy is limited by \
             how well the producer centred its labels.",
            distinct.len()
        ));
    }

    (
        Axis {
            scale,
            ticks,
            fit: Some(fit),
            title,
        },
        warnings,
    )
}

/// Ordinary least squares of `y` on `x`, with residuals.
///
/// Returns `None` if `x` is degenerate (no spread), which would make the slope
/// undefined.
fn least_squares(x: &[f64], y: &[f64]) -> Option<AxisFit> {
    let n = x.len() as f64;
    if x.len() < 2 {
        return None;
    }
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let sxx: f64 = x.iter().map(|v| (v - mean_x).powi(2)).sum();
    let sxy: f64 = x
        .iter()
        .zip(y)
        .map(|(a, b)| (a - mean_x) * (b - mean_y))
        .sum();

    if sxx.abs() < f64::EPSILON {
        return None;
    }
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    if slope.abs() < f64::EPSILON || !slope.is_finite() {
        return None;
    }

    let residual_rms = (x
        .iter()
        .zip(y)
        .map(|(a, b)| (b - (slope * a + intercept)).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    let span = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - y.iter().cloned().fold(f64::INFINITY, f64::min);
    let residual_normalised = if span.abs() > f64::EPSILON {
        residual_rms / span
    } else {
        // No spread in the values means every tick claims the same number.
        // That is not an axis.
        return None;
    };

    Some(AxisFit {
        slope,
        intercept,
        residual_rms,
        residual_normalised,
    })
}

/// Rename the generic "axis" in a warning to the concrete axis it belongs to,
/// so a caller reading `warnings` can tell x from y.
pub fn rename_axis_warnings(warnings: Vec<String>, axis: AxisKind) -> Vec<String> {
    let name = match axis {
        AxisKind::X => "x-axis",
        AxisKind::Y => "y-axis",
    };
    warnings
        .into_iter()
        .map(|w| w.replacen("axis:", &format!("{name}:"), 1))
        .collect()
}

/// Group ticks by value for diagnostics. Used by [`crate::digitise`] to notice
/// a duplicated tick label, which usually means a mis-parse.
pub fn duplicate_values(ticks: &[TickObservation]) -> Vec<f64> {
    let mut counts: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for t in ticks {
        let entry = counts.entry(format!("{:.12e}", t.value)).or_insert((t.value, 0));
        entry.1 += 1;
    }
    counts
        .values()
        .filter(|(_, n)| *n > 1)
        .map(|(v, _)| *v)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tick(page: f32, value: f64) -> TickObservation {
        TickObservation {
            page,
            value,
            text: value.to_string(),
            source: TickSource::TickMark,
        }
    }

    #[test]
    fn fits_a_linear_axis_exactly() {
        // Ticks at 0, 0.25, 0.5, 0.75, 1.0 evenly spaced over 400pt.
        let ticks: Vec<_> = (0..=4)
            .map(|i| tick(100.0 + 100.0 * i as f32, 0.25 * i as f64))
            .collect();
        let (axis, warnings) = fit_axis(ticks, None);
        assert_eq!(axis.scale, AxisScale::Linear);
        let fit = axis.fit.expect("must calibrate");
        assert!(fit.residual_normalised < 1e-6, "{fit:?}");
        assert!((axis.to_data(300.0).unwrap() - 0.5).abs() < 1e-9);
        assert!(warnings.is_empty(), "clean fit must not warn: {warnings:?}");
    }

    #[test]
    fn detects_a_log_axis_from_decade_ticks() {
        // 1e-3 .. 1e0, evenly spaced. A linear reading of this is the single
        // most dangerous silent failure this crate can have.
        let ticks: Vec<_> = (0..=3)
            .map(|i| tick(100.0 + 100.0 * i as f32, 10f64.powi(i - 3)))
            .collect();
        let (axis, _) = fit_axis(ticks, None);
        assert_eq!(axis.scale, AxisScale::Log10);
        let v = axis.to_data(250.0).unwrap();
        // Halfway between 1e-2 and 1e-1 on a log axis is 10^-1.5.
        assert!((v - 10f64.powf(-1.5)).abs() < 1e-9, "got {v}");
    }

    #[test]
    fn detects_a_log_axis_with_non_decade_ticks() {
        // 1, 2, 5, 10 -- log-spaced but not decades, which a naive
        // "are the labels powers of ten?" check would miss entirely.
        let vals: [f64; 4] = [1.0, 2.0, 5.0, 10.0];
        let ticks: Vec<_> = vals
            .iter()
            .map(|&v| {
                let frac = v.log10() / 10f64.log10();
                tick(100.0 + 300.0 * frac as f32, v)
            })
            .collect();
        let (axis, _) = fit_axis(ticks, None);
        assert_eq!(axis.scale, AxisScale::Log10);
    }

    #[test]
    fn two_ticks_cannot_determine_the_scale_and_says_so() {
        let (axis, warnings) = fit_axis(vec![tick(100.0, 1.0), tick(500.0, 100.0)], None);
        // Both models fit two points perfectly, so we must NOT claim to know.
        assert!(
            warnings.iter().any(|w| w.contains("ASSUMED LINEAR")),
            "must warn loudly: {warnings:?}"
        );
        assert_eq!(axis.scale, AxisScale::Linear);
    }

    #[test]
    fn one_tick_refuses_to_calibrate() {
        let (axis, warnings) = fit_axis(vec![tick(100.0, 1.0)], None);
        assert!(axis.fit.is_none(), "must not invent a calibration");
        assert!(!axis.is_calibrated());
        assert!(axis.to_data(200.0).is_none());
        assert!(warnings.iter().any(|w| w.contains("cannot calibrate")));
    }

    #[test]
    fn zero_ticks_refuses_to_calibrate() {
        let (axis, warnings) = fit_axis(vec![], None);
        assert!(axis.fit.is_none());
        assert!(!warnings.is_empty());
    }

    #[test]
    fn degenerate_ticks_refuse_to_calibrate() {
        // Every tick claiming the same value is not an axis.
        let (axis, _) = fit_axis(vec![tick(100.0, 5.0), tick(200.0, 5.0), tick(300.0, 5.0)], None);
        assert!(axis.fit.is_none());
    }

    #[test]
    fn label_centre_calibration_is_flagged() {
        let ticks: Vec<_> = (0..=3)
            .map(|i| TickObservation {
                page: 100.0 + 100.0 * i as f32,
                value: i as f64,
                text: i.to_string(),
                source: TickSource::LabelCentre,
            })
            .collect();
        let (_, warnings) = fit_axis(ticks, None);
        assert!(
            warnings.iter().any(|w| w.contains("tick-label centres")),
            "{warnings:?}"
        );
    }

    #[test]
    fn spines_merge_collinear_fragments() {
        // An axis drawn as four abutting 100pt pieces must read as one 400pt
        // spine, or it will never clear the "is this long enough" bar.
        let merged = merge_collinear(
            (0..4)
                .map(|i| Spine {
                    horizontal: true,
                    pos: 150.0,
                    min: 100.0 + 100.0 * i as f32,
                    max: 200.0 + 100.0 * i as f32,
                })
                .collect(),
        );
        assert_eq!(merged.len(), 1);
        assert!((merged[0].length() - 400.0).abs() < 1e-3);
    }

    #[test]
    fn crossing_spines_form_a_region_and_parallel_ones_do_not() {
        let page = Rect::from_corners(0.0, 0.0, 612.0, 792.0);
        let h = Spine {
            horizontal: true,
            pos: 150.0,
            min: 100.0,
            max: 500.0,
        };
        let v = Spine {
            horizontal: false,
            pos: 100.0,
            min: 150.0,
            max: 600.0,
        };
        let regions = plot_regions(&[h, v], page);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0], Rect::from_corners(100.0, 150.0, 500.0, 600.0));

        // Two long horizontal rules (a table) must not become a plot.
        let table = [
            h,
            Spine {
                horizontal: true,
                pos: 300.0,
                min: 100.0,
                max: 500.0,
            },
        ];
        assert!(plot_regions(&table, page).is_empty());
    }
}
