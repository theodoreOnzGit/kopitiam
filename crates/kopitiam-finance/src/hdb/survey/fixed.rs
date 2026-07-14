//! Exact decimal parsing for published statistics.
//!
//! Every quantity in this module is a fixed-point integer (see the module docs
//! in [`super`] for why there is no `f64` here). This is the one place that
//! turns a published decimal string — `"1,500"`, `"87.3"`, `"195.5"`, `"$540,000"`
//! — into that integer, exactly.
//!
//! # Why parsing rejects rather than rounds
//!
//! If a publication prints `87.35%` and our percentage scale only holds two
//! decimal places, the honest response is to *fail* and force the scale to be
//! reconsidered — not to quietly store `87.3` and let a lost digit propagate
//! into a knowledge graph that claims provenance. Silent rounding is exactly the
//! class of error this module exists to prevent, so [`parse_fixed`] returns
//! [`FixedParseError::TooPrecise`] instead.

/// Why a published decimal string could not be turned into an exact fixed-point
/// integer.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FixedParseError {
    /// The string held no digits at all (empty, or only separators/symbols).
    #[error("`{input}` contains no numeric value")]
    Empty { input: String },

    /// The string contained characters that are not part of a number, after
    /// the permitted separators and currency/percent symbols were stripped.
    #[error("`{input}` is not a number (unexpected character `{found}`)")]
    NotANumber { input: String, found: char },

    /// The value carries more decimal places than the target scale can hold
    /// exactly. Deliberately an error and not a rounding: see the module docs.
    #[error(
        "`{input}` has {digits} decimal places but this quantity holds only \
         {scale}; refusing to round away published precision"
    )]
    TooPrecise {
        input: String,
        digits: u32,
        scale: u32,
    },

    /// The value does not fit in the fixed-point representation.
    #[error("`{input}` is out of range for this quantity")]
    OutOfRange { input: String },

    /// The cell is a suppression marker rather than a value. Statistics
    /// agencies print `-`, `n.a.`, `s` or similar when a figure is unavailable
    /// or has been suppressed for confidentiality (a cell with too few
    /// respondents to publish safely). This is *data about the data* and must
    /// never be read as zero.
    #[error("`{input}` marks a suppressed or unavailable figure, not a value")]
    Suppressed { input: String },
}

/// Markers that statistical publications use for "no value here".
///
/// These are matched case-insensitively against the whole (trimmed) cell. The
/// critical property is that none of them may ever parse to `0` — a suppressed
/// cell and a genuine zero are different claims about the world, and conflating
/// them fabricates data.
const SUPPRESSION_MARKERS: &[&str] = &[
    "-", "--", "–", "—", "n.a.", "na", "n/a", "nil", "s", "..", ".", "*", "n.e.", "not available",
];

/// Parses a published decimal string into a fixed-point integer with `scale`
/// decimal places.
///
/// `scale` is the number of decimal digits the target type keeps: `2` for money
/// in cents, `2` for percentages in hundredths of a percent, `3` for index
/// points in thousandths.
///
/// Tolerated, because real publications contain them:
///
/// * thousands separators — `1,500` and `1 500` (including a non-breaking space,
///   which PDF text extraction produces routinely)
/// * a leading currency symbol — `$540,000`
/// * a trailing percent sign — `87.3%`
/// * surrounding whitespace, and a footnote marker already stripped by the caller
///
/// Not tolerated: anything else. An unrecognised character is an error, not a
/// character to skip over.
///
/// ```
/// # use kopitiam_finance::hdb::survey::SgdAmount;
/// // "$1,500" at money scale (cents) is exactly 150_000 cents.
/// let amount = SgdAmount::parse("$1,500").unwrap();
/// assert_eq!(amount.cents(), 150_000);
/// ```
pub fn parse_fixed(input: &str, scale: u32) -> Result<i64, FixedParseError> {
    let trimmed = input.trim();

    if SUPPRESSION_MARKERS
        .iter()
        .any(|marker| trimmed.eq_ignore_ascii_case(marker))
    {
        return Err(FixedParseError::Suppressed {
            input: input.to_string(),
        });
    }

    // Strip the decorations a publication puts *around* a number. Note we strip
    // only from the ends: a `$` in the middle of a cell means we have
    // misidentified the cell, and should fail rather than salvage.
    let body = trimmed
        .trim_start_matches(['$', 'S'])
        .trim_start()
        .trim_end_matches('%')
        .trim_end();

    let mut negative = false;
    let mut integer_digits = String::new();
    let mut fraction_digits = String::new();
    let mut seen_point = false;

    for (position, ch) in body.chars().enumerate() {
        match ch {
            '-' | '\u{2212}' if position == 0 => negative = true,
            '+' if position == 0 => {}
            // Thousands separators, including the non-breaking and thin spaces
            // that PDF extraction leaves behind.
            ',' | ' ' | '\u{00a0}' | '\u{2009}' | '\'' => {}
            '.' if !seen_point => seen_point = true,
            '0'..='9' if seen_point => fraction_digits.push(ch),
            '0'..='9' => integer_digits.push(ch),
            other => {
                return Err(FixedParseError::NotANumber {
                    input: input.to_string(),
                    found: other,
                });
            }
        }
    }

    if integer_digits.is_empty() && fraction_digits.is_empty() {
        return Err(FixedParseError::Empty {
            input: input.to_string(),
        });
    }

    // Trailing zeros beyond the scale are not a loss of precision — `87.30` at
    // scale 1 is exactly `87.3`. Only *significant* excess digits are an error.
    let significant = fraction_digits.trim_end_matches('0').len() as u32;
    if significant > scale {
        return Err(FixedParseError::TooPrecise {
            input: input.to_string(),
            digits: significant,
            scale,
        });
    }

    let integer: i64 = if integer_digits.is_empty() {
        0
    } else {
        integer_digits
            .parse()
            .map_err(|_| FixedParseError::OutOfRange {
                input: input.to_string(),
            })?
    };

    // Pad or truncate the fraction to exactly `scale` digits. Truncation here is
    // safe: we proved above that everything past `scale` is a zero.
    let mut fraction = fraction_digits;
    fraction.truncate(scale as usize);
    while (fraction.len() as u32) < scale {
        fraction.push('0');
    }
    let fraction: i64 = if fraction.is_empty() {
        0
    } else {
        fraction.parse().map_err(|_| FixedParseError::OutOfRange {
            input: input.to_string(),
        })?
    };

    let multiplier = 10_i64
        .checked_pow(scale)
        .ok_or_else(|| FixedParseError::OutOfRange {
            input: input.to_string(),
        })?;

    let magnitude = integer
        .checked_mul(multiplier)
        .and_then(|scaled| scaled.checked_add(fraction))
        .ok_or_else(|| FixedParseError::OutOfRange {
            input: input.to_string(),
        })?;

    Ok(if negative { -magnitude } else { magnitude })
}

/// Renders a fixed-point integer back to a decimal string with exactly `scale`
/// decimal places, so a value round-trips to the precision it was published at.
pub fn format_fixed(value: i64, scale: u32) -> String {
    if scale == 0 {
        return value.to_string();
    }
    let multiplier = 10_i64.pow(scale);
    let sign = if value < 0 { "-" } else { "" };
    let magnitude = value.unsigned_abs();
    let multiplier = multiplier as u64;
    format!(
        "{sign}{}.{:0width$}",
        magnitude / multiplier,
        magnitude % multiplier,
        width = scale as usize
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_thousands_separated_money() {
        assert_eq!(parse_fixed("1,500", 2).unwrap(), 150_000);
        assert_eq!(parse_fixed("$540,000", 2).unwrap(), 54_000_000);
        // PDF extraction commonly yields a non-breaking space as the separator.
        assert_eq!(parse_fixed("1\u{00a0}500", 2).unwrap(), 150_000);
    }

    #[test]
    fn parses_percentages_exactly() {
        assert_eq!(parse_fixed("87.3%", 2).unwrap(), 8730);
        assert_eq!(parse_fixed("100", 2).unwrap(), 10_000);
        assert_eq!(parse_fixed("0.05", 2).unwrap(), 5);
    }

    #[test]
    fn excess_precision_is_refused_not_rounded() {
        // The whole point: 87.345 at two decimal places must NOT become 87.34.
        let err = parse_fixed("87.345", 2).unwrap_err();
        assert!(matches!(
            err,
            FixedParseError::TooPrecise {
                digits: 3,
                scale: 2,
                ..
            }
        ));
    }

    #[test]
    fn trailing_zeros_are_not_excess_precision() {
        // 87.300 at scale 2 loses nothing, so it must parse rather than error.
        assert_eq!(parse_fixed("87.300", 2).unwrap(), 8730);
    }

    #[test]
    fn suppression_markers_never_parse_as_zero() {
        // This is the single most important test in this file. A statistics
        // agency printing `-` means "we are not telling you this number",
        // which is not the same claim as "this number is zero".
        for marker in ["-", "n.a.", "N/A", "s", "..", "–"] {
            let err = parse_fixed(marker, 2).unwrap_err();
            assert!(
                matches!(err, FixedParseError::Suppressed { .. }),
                "`{marker}` must be Suppressed, got {err:?}"
            );
        }
    }

    #[test]
    fn junk_is_rejected_rather_than_salvaged() {
        assert!(matches!(
            parse_fixed("12abc", 2),
            Err(FixedParseError::NotANumber { .. })
        ));
        assert!(matches!(
            parse_fixed("", 2),
            Err(FixedParseError::Empty { .. })
        ));
    }

    #[test]
    fn formats_back_to_published_precision() {
        assert_eq!(format_fixed(8730, 2), "87.30");
        assert_eq!(format_fixed(195_500, 3), "195.500");
        assert_eq!(format_fixed(-250, 2), "-2.50");
        assert_eq!(format_fixed(42, 0), "42");
    }

    #[test]
    fn round_trips_through_parse_and_format() {
        let value = parse_fixed("195.5", 3).unwrap();
        assert_eq!(format_fixed(value, 3), "195.500");
    }
}
