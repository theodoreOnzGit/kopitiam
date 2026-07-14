//! Turning `kopitiam-pdf` text spans back into the *numbers* a tick label
//! stands for.
//!
//! # Why a tick label is not just a string
//!
//! Two things happen between a plotting library writing "0.5" and this crate
//! reading it back, and both of them will silently destroy a calibration if
//! they are not handled.
//!
//! **A label can arrive in pieces.** `kopitiam-pdf` emits one [`TextSpan`] per
//! `Tj`/`TJ`-string operand. A producer that kerns its digits writes `0.5` as a
//! `TJ` array of three separate strings, which comes back as three spans:
//! `"0"`, `"."`, `"5"`. Parsing those individually yields the values 0 and 5
//! and a piece of punctuation, and a calibration built on that is not merely
//! inaccurate -- it is confidently, catastrophically wrong. So spans have to be
//! re-assembled into labels first, by proximity, before anything tries to read
//! a number out of them.
//!
//! **A label may not be a decimal at all.** On a log axis, matplotlib's default
//! tick label is not `0.001`; it is the glyph `10` with a raised, smaller `-3`
//! next to it. Read naively, that is the number *ten*, at every single tick --
//! which produces a perfectly uniform set of "values" that will fit a straight
//! line with zero residual and report high confidence. This is the single most
//! dangerous failure mode in the crate, and it is why [`Label`] carries a
//! superscript as structure rather than flattening it into the text.
//!
//! And a detail that looks like pedantry until it costs you an afternoon:
//! matplotlib writes its minus signs as U+2212 MINUS SIGN, not U+002D HYPHEN.
//! `"−3".parse::<f64>()` fails. Every negative tick on the axis silently
//! disappears, and the calibration is fitted on the positive half only.

use kopitiam_pdf::TextSpan;

use crate::geometry::{Point, Rect};

/// A tick label reassembled from one or more spans: the text on the baseline,
/// plus any raised exponent found next to it.
#[derive(Debug, Clone, PartialEq)]
pub struct Label {
    pub text: String,
    /// A smaller, raised run immediately after `text` -- the `-3` of `10⁻³`.
    pub superscript: Option<String>,
    pub rect: Rect,
    pub font_size: f32,
}

impl Label {
    pub fn center(&self) -> Point {
        self.rect.center()
    }

    /// The numeric value this label denotes, or `None` if it is not a number.
    ///
    /// Returning `None` is a feature, not a shortfall: axis *titles*
    /// ("temperature"), legend entries and stray annotations all live in the same
    /// bands as tick labels, and the only thing separating them is that they do
    /// not parse. A parser that tried harder here would pull junk into the
    /// calibration.
    pub fn value(&self) -> Option<f64> {
        let text = normalise(&self.text);

        if let Some(sup) = &self.superscript {
            let exponent: f64 = normalise(sup).parse().ok()?;
            // The base must literally be a power of ten for the superscript to
            // mean an exponent. Anything else with a raised run next to it (a
            // footnote marker, a unit like "m³") is not a tick value, and
            // guessing at one would inject a fabricated number into the fit.
            let mantissa_text = text.strip_suffix("10")?;
            // `a x 10^n`, where the multiplication sign may be any of these
            // depending on the producer.
            let mantissa_text =
                mantissa_text.trim_end_matches(['x', '*', '\u{00d7}', '\u{22c5}', '\u{00b7}']);
            let mantissa = if mantissa_text.is_empty() {
                1.0
            } else {
                mantissa_text.parse::<f64>().ok()?
            };
            return Some(mantissa * 10.0_f64.powf(exponent));
        }

        text.parse::<f64>().ok()
    }
}

/// Normalise a numeric string into something `f64::from_str` will accept.
///
/// Handles the Unicode minus (U+2212) that matplotlib uses by default, the
/// Unicode hyphen variants, the thin/narrow spaces some producers use as digit
/// separators, and comma digit grouping (see [`strip_digit_grouping`]).
fn normalise(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .filter_map(|c| match c {
            // U+2212 MINUS SIGN, U+2013 EN DASH, U+2010 HYPHEN, U+00AD SOFT HYPHEN.
            '\u{2212}' | '\u{2013}' | '\u{2010}' | '\u{00ad}' => Some('-'),
            // Thin, narrow-no-break and hair spaces used as digit separators.
            '\u{2009}' | '\u{202f}' | '\u{200a}' | '\u{00a0}' => None,
            c if c.is_whitespace() => None,
            c => Some(c),
        })
        .collect();
    strip_digit_grouping(&cleaned)
}

/// Remove `,` when -- and only when -- it is being used to group thousands.
///
/// # Why this is not as simple as deleting commas
///
/// This was found by running the digitiser against a real conference paper. Its
/// pressure axis is labelled `0`, `1,000`, `2,000` ... `11,000` Pa, and every
/// one of those labels failed to parse: `"11,000".parse::<f64>()` is an error.
/// The axis collapsed to a single usable tick (`0`) and refused to calibrate --
/// which was at least honest, but it meant the crate could not read one of the
/// most ordinary axes in engineering.
///
/// The catch is that `,` is a **decimal separator** in much of the world, so
/// `1,5` means one and a half. Blindly deleting commas would turn that into 15 --
/// a silent factor-of-ten error, precisely the sort of thing this crate exists to
/// avoid.
///
/// The discriminator is the grouping pattern itself: a thousands separator is
/// always preceded by a digit and followed by **exactly three** digits, and every
/// comma in the number obeys that rule. `1,5` does not (one digit follows), so it
/// is left alone and simply fails to parse -- an honest `None` rather than a
/// confident 15.
///
/// The residual ambiguity is real and worth naming: `1,000` is *technically*
/// readable as "one, to three decimal places". Nobody labels an axis that way,
/// and the English-language scientific literature this crate targets uses comma
/// grouping -- so we read it as one thousand. But because the consequence of
/// being wrong is a 1000x error, [`crate::axes::fit_axis`] raises a warning
/// whenever it calibrates an axis from comma-grouped labels, and the label's
/// printed text is preserved verbatim in the tick observation so a reader can
/// check it.
fn strip_digit_grouping(s: &str) -> String {
    if !s.contains(',') {
        return s.to_string();
    }

    let bytes: Vec<char> = s.chars().collect();
    let is_grouping = |i: usize| -> bool {
        // Preceded by a digit...
        if i == 0 || !bytes[i - 1].is_ascii_digit() {
            return false;
        }
        // ...and followed by exactly three digits, then either the end of the
        // number or another separator.
        let after: Vec<char> = bytes[i + 1..].to_vec();
        if after.len() < 3 || !after[..3].iter().all(char::is_ascii_digit) {
            return false;
        }
        matches!(after.get(3), None | Some(',')) || after.get(3) == Some(&'.')
    };

    // Every comma must look like a group separator, or we touch none of them.
    let all_grouping = bytes
        .iter()
        .enumerate()
        .filter(|(_, c)| **c == ',')
        .all(|(i, _)| is_grouping(i));

    if all_grouping {
        s.chars().filter(|c| *c != ',').collect()
    } else {
        s.to_string()
    }
}

/// Reassemble text spans into labels.
///
/// Two spans join when they sit on the same baseline and are close enough
/// horizontally that no word break could be between them. A raised, smaller
/// span joins as a superscript instead.
///
/// The thresholds are proportional to font size rather than absolute, because
/// the same figure may carry 6pt tick labels and a 14pt title, and a fixed
/// point-gap would either split the small text or glue the large text together.
pub fn assemble(spans: &[TextSpan]) -> Vec<Label> {
    group_into_lines(spans)
        .iter()
        .flat_map(|line| assemble_line(line))
        .collect()
}

/// Group spans into rows of text.
///
/// This has to happen *before* left-to-right ordering, and the reason is a trap
/// worth naming. A superscript sits at a **higher** y than its base, so a naive
/// "sort by y descending, then x" puts the `-3` of `10⁻³` *before* the `10` it
/// belongs to -- and the joiner, which only ever looks backwards at the label it
/// is currently building, would never pair them. Rows first, then x within the
/// row, puts the base ahead of its exponent where the joiner can see it.
///
/// The row tolerance keys off the larger of the two font sizes in play,
/// precisely so a small raised exponent is still counted as being on the same
/// row as the large base it modifies.
fn group_into_lines(spans: &[TextSpan]) -> Vec<Vec<&TextSpan>> {
    let mut ordered: Vec<&TextSpan> = spans.iter().filter(|s| !s.text.trim().is_empty()).collect();
    ordered.sort_by(|a, b| b.y.partial_cmp(&a.y).unwrap_or(std::cmp::Ordering::Equal));

    let mut lines: Vec<Vec<&TextSpan>> = Vec::new();
    let mut reference_y = f32::NAN;
    let mut max_font = 0.0_f32;

    for span in ordered {
        let tolerance = 0.7 * max_font.max(span.font_size);
        if lines.is_empty() || (span.y - reference_y).abs() > tolerance {
            lines.push(vec![span]);
            reference_y = span.y;
            max_font = span.font_size;
        } else {
            max_font = max_font.max(span.font_size);
            if let Some(line) = lines.last_mut() {
                line.push(span);
            }
        }
    }

    for line in &mut lines {
        line.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
    }
    lines
}

/// Walk one row left to right, joining spans into labels.
///
/// Scoped to a single row so that the label being built can never absorb a span
/// from the row above it, however close the two rows happen to sit.
fn assemble_line(line: &[&TextSpan]) -> Vec<Label> {
    let mut labels: Vec<Label> = Vec::new();
    for span in line {
        if let Some(last) = labels.last_mut()
            && join(last, span)
        {
            continue;
        }
        labels.push(Label {
            text: span.text.trim().to_string(),
            superscript: None,
            rect: Rect::from_corners(span.x, span.y, span.x + span.width, span.y + span.height),
            font_size: span.font_size,
        });
    }
    labels
}

/// Try to merge `span` into `label`, returning whether it was absorbed.
fn join(label: &mut Label, span: &TextSpan) -> bool {
    let size = label.font_size.max(1.0);
    let gap = span.x - label.rect.right();

    // A run that has already taken a superscript is finished: `10^3 10^4` must
    // not become one label.
    if label.superscript.is_some() {
        return false;
    }

    // Only a *positive* gap can separate two labels. A negative gap means the
    // spans overlap, which no word break ever does -- it means our width
    // estimate for the previous span ran long, which is routine: `kopitiam-pdf`
    // derives a span's width from glyph advances, and the advance of a narrow
    // glyph like `.` is easily over-counted. So the lower bound is generous and
    // the upper bound is the one doing the real work.
    let min_gap = -size;

    // Superscript: markedly smaller, raised above this baseline, and adjacent.
    let raised = span.y - label.rect.y;
    if span.font_size < 0.85 * size
        && raised > 0.15 * size
        && raised < 0.9 * size
        && (min_gap..0.5 * size).contains(&gap)
    {
        label.superscript = Some(span.text.trim().to_string());
        label.rect = label.rect.union(&Rect::from_corners(
            span.x,
            span.y,
            span.x + span.width,
            span.y + span.height,
        ));
        return true;
    }

    // Same-baseline continuation. The *upper* threshold is deliberately tight: a
    // space between words is ~0.28 em in most fonts, so 0.25 em keeps kerned
    // digit runs together ("0" "." "5") while refusing to glue two adjacent
    // tick labels into one -- those are separated by many em of whitespace.
    let same_baseline = (span.y - label.rect.y).abs() < 0.25 * size;
    let same_size = (span.font_size - size).abs() < 0.2 * size;
    if same_baseline && same_size && (min_gap..0.25 * size).contains(&gap) {
        label.text.push_str(span.text.trim());
        label.rect = label.rect.union(&Rect::from_corners(
            span.x,
            span.y,
            span.x + span.width,
            span.y + span.height,
        ));
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(text: &str, x: f32, y: f32, size: f32) -> TextSpan {
        TextSpan {
            text: text.to_string(),
            x,
            y,
            width: text.len() as f32 * size * 0.55,
            height: size,
            font_size: size,
            ..TextSpan::default()
        }
    }

    fn label(text: &str, superscript: Option<&str>) -> Label {
        Label {
            text: text.to_string(),
            superscript: superscript.map(str::to_string),
            rect: Rect::from_corners(0.0, 0.0, 10.0, 10.0),
            font_size: 9.0,
        }
    }

    #[test]
    fn parses_plain_decimals() {
        assert_eq!(label("0.25", None).value(), Some(0.25));
        assert_eq!(label("-1.5", None).value(), Some(-1.5));
        assert_eq!(label("1e-3", None).value(), Some(1e-3));
        assert_eq!(label("100", None).value(), Some(100.0));
    }

    #[test]
    fn parses_matplotlibs_unicode_minus() {
        // U+2212. `"\u{2212}3".parse::<f64>()` fails outright, which would drop
        // every negative tick and fit the calibration on half the axis.
        assert_eq!(label("\u{2212}3.5", None).value(), Some(-3.5));
    }

    #[test]
    fn parses_power_of_ten_labels() {
        assert_eq!(label("10", Some("-3")).value(), Some(1e-3));
        assert_eq!(label("10", Some("0")).value(), Some(1.0));
        assert_eq!(label("10", Some("\u{2212}2")).value(), Some(1e-2));
        // The `a x 10^n` form.
        let v = label("2\u{00d7}10", Some("3")).value().unwrap();
        assert!((v - 2000.0).abs() < 1e-9, "got {v}");
    }

    #[test]
    fn parses_comma_grouped_thousands() {
        // Found against a real conference paper: a pressure axis labelled
        // 0 .. 11,000 Pa parsed as a single usable tick before this worked.
        assert_eq!(label("11,000", None).value(), Some(11000.0));
        assert_eq!(label("1,000", None).value(), Some(1000.0));
        assert_eq!(label("1,234,567", None).value(), Some(1234567.0));
        // With a Unicode minus, as matplotlib and Word both emit.
        assert_eq!(label("\u{2212}1,000", None).value(), Some(-1000.0));
        assert_eq!(label("12,345.67", None).value(), Some(12345.67));
    }

    #[test]
    fn refuses_to_guess_at_a_decimal_comma() {
        // `1,5` is one and a half in much of the world. Deleting the comma would
        // silently make it 15. An honest `None` is the only safe answer, since
        // nothing in the figure can tell us which convention was meant.
        assert_eq!(label("1,5", None).value(), None);
        assert_eq!(label("3,14159", None).value(), None);
        // A mix of a real group separator and a non-conforming one: touch neither.
        assert_eq!(label("1,000,5", None).value(), None);
    }

    #[test]
    fn rejects_non_numeric_text() {
        // Axis titles and legend text share the band with tick labels; they must
        // not be pulled into the calibration.
        assert_eq!(label("temperature", None).value(), None);
        assert_eq!(label("dB", None).value(), None);
        assert_eq!(label("", None).value(), None);
    }

    #[test]
    fn reassembles_kerned_digits_into_one_label() {
        // "0", ".", "5" written as three TJ strings, as a kerning producer does.
        let spans = vec![
            span("0", 100.0, 50.0, 9.0),
            span(".", 105.0, 50.0, 9.0),
            span("5", 107.0, 50.0, 9.0),
        ];
        let labels = assemble(&spans);
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].text, "0.5");
        assert_eq!(labels[0].value(), Some(0.5));
    }

    #[test]
    fn keeps_separate_tick_labels_apart() {
        // Two tick labels 40pt apart must not merge into "0.00.5".
        let spans = vec![span("0.0", 100.0, 50.0, 9.0), span("0.5", 140.0, 50.0, 9.0)];
        let labels = assemble(&spans);
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].value(), Some(0.0));
        assert_eq!(labels[1].value(), Some(0.5));
    }

    #[test]
    fn reassembles_superscript_exponent() {
        let mut sup = span("-3", 111.0, 54.0, 5.4);
        sup.font_size = 5.4;
        let spans = vec![span("10", 100.0, 50.0, 9.0), sup];
        let labels = assemble(&spans);
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].text, "10");
        assert_eq!(labels[0].superscript.as_deref(), Some("-3"));
        assert_eq!(labels[0].value(), Some(1e-3));
    }
}
