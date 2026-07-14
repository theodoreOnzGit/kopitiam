//! When a figure was observed.
//!
//! HDB publishes at mixed granularity: the Resale Price Index is quarterly,
//! resale transactions are stamped by month, the Sample Household Survey covers
//! a fieldwork window spanning months. A period model that only understood years
//! would force every one of those into a lie.
//!
//! # Ordering across mixed granularity
//!
//! `Period` is deliberately **not** `Ord`. `2023` and `2Q2023` do not have a
//! total order — the year *contains* the quarter, and asking which is "greater"
//! is a category error. What callers actually need is [`Period::start`] /
//! [`Period::end`] (a half-open month range), plus [`Period::contains`] and
//! [`Period::overlaps`]. Time-series code sorts by [`Period::start`] explicitly,
//! which makes the choice visible instead of accidental.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A calendar quarter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quarter {
    Q1,
    Q2,
    Q3,
    Q4,
}

impl Quarter {
    /// The first month of the quarter, 1-based.
    fn first_month(self) -> u32 {
        match self {
            Quarter::Q1 => 1,
            Quarter::Q2 => 4,
            Quarter::Q3 => 7,
            Quarter::Q4 => 10,
        }
    }
}

impl fmt::Display for Quarter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = match self {
            Quarter::Q1 => 1,
            Quarter::Q2 => 2,
            Quarter::Q3 => 3,
            Quarter::Q4 => 4,
        };
        write!(f, "Q{n}")
    }
}

/// Why a period could not be constructed or parsed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PeriodError {
    #[error("month {month} is not a calendar month (expected 1..=12)")]
    BadMonth { month: u32 },

    #[error("`{input}` is not a recognisable period")]
    Unrecognised { input: String },

    #[error("period range runs backwards: {from} is after {to}")]
    BackwardsRange { from: Box<Period>, to: Box<Period> },
}

/// The stretch of time a figure describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Period {
    /// A whole calendar year.
    Year(u16),
    /// A calendar quarter — the granularity the Resale Price Index is published
    /// at.
    Quarter { year: u16, quarter: Quarter },
    /// A calendar month — the granularity resale transactions are stamped at.
    Month { year: u16, month: u8 },
    /// A fieldwork or observation window spanning months, e.g. a survey run from
    /// November through the following March. Stored as an inclusive month range
    /// so it cannot silently collapse into a single year.
    Window {
        from_year: u16,
        from_month: u8,
        to_year: u16,
        to_month: u8,
    },
}

impl Period {
    /// A calendar month, validated.
    pub fn month(year: u16, month: u32) -> Result<Self, PeriodError> {
        if !(1..=12).contains(&month) {
            return Err(PeriodError::BadMonth { month });
        }
        Ok(Period::Month {
            year,
            month: month as u8,
        })
    }

    /// An inclusive observation window between two months.
    pub fn window(
        from_year: u16,
        from_month: u32,
        to_year: u16,
        to_month: u32,
    ) -> Result<Self, PeriodError> {
        if !(1..=12).contains(&from_month) {
            return Err(PeriodError::BadMonth { month: from_month });
        }
        if !(1..=12).contains(&to_month) {
            return Err(PeriodError::BadMonth { month: to_month });
        }
        let period = Period::Window {
            from_year,
            from_month: from_month as u8,
            to_year,
            to_month: to_month as u8,
        };
        if period.start() > period.end() {
            return Err(PeriodError::BackwardsRange {
                from: Box::new(Period::month(from_year, from_month)?),
                to: Box::new(Period::month(to_year, to_month)?),
            });
        }
        Ok(period)
    }

    /// The first month of the period, as an absolute month ordinal
    /// (`year * 12 + month - 1`). This is the canonical scalar the whole module
    /// sorts and compares by.
    pub fn start(self) -> u32 {
        match self {
            Period::Year(y) => u32::from(y) * 12,
            Period::Quarter { year, quarter } => {
                u32::from(year) * 12 + quarter.first_month() - 1
            }
            Period::Month { year, month } => u32::from(year) * 12 + u32::from(month) - 1,
            Period::Window {
                from_year,
                from_month,
                ..
            } => u32::from(from_year) * 12 + u32::from(from_month) - 1,
        }
    }

    /// The last month of the period, inclusive, as an absolute month ordinal.
    pub fn end(self) -> u32 {
        match self {
            Period::Year(y) => u32::from(y) * 12 + 11,
            Period::Quarter { year, quarter } => {
                u32::from(year) * 12 + quarter.first_month() + 1
            }
            Period::Month { year, month } => u32::from(year) * 12 + u32::from(month) - 1,
            Period::Window {
                to_year, to_month, ..
            } => u32::from(to_year) * 12 + u32::from(to_month) - 1,
        }
    }

    /// Whether `self` wholly contains `other` — e.g. `2023` contains `2Q2023`.
    pub fn contains(self, other: Period) -> bool {
        self.start() <= other.start() && other.end() <= self.end()
    }

    /// Whether the two periods share any month at all.
    ///
    /// Overlap is the test that matters when deciding whether two figures could
    /// be describing the same stretch of market activity. Two *overlapping* but
    /// unequal periods are a reason for caution, not a reason to join.
    pub fn overlaps(self, other: Period) -> bool {
        self.start() <= other.end() && other.start() <= self.end()
    }

    /// Parses the period spellings that appear in HDB table headers.
    ///
    /// Accepts `2023`, `2Q2023`, `Q2 2023`, `2023Q2`, and `Jan 2023` /
    /// `2023-01`. Anything else is [`PeriodError::Unrecognised`] — this parser
    /// does not guess, because a misread period silently reassigns a number to
    /// the wrong slice of market history.
    pub fn parse(input: &str) -> Result<Self, PeriodError> {
        let text = input.trim();
        let unrecognised = || PeriodError::Unrecognised {
            input: input.to_string(),
        };

        // Bare year: `2023`.
        if let Ok(year) = text.parse::<u16>() {
            if (1960..=2200).contains(&year) {
                return Ok(Period::Year(year));
            }
            return Err(unrecognised());
        }

        // `2023-01` / `2023/01`.
        if let Some((year, month)) = text
            .split_once('-')
            .or_else(|| text.split_once('/'))
            .and_then(|(y, m)| Some((y.trim().parse::<u16>().ok()?, m.trim().parse::<u32>().ok()?)))
        {
            return Period::month(year, month);
        }

        let upper = text.to_ascii_uppercase();

        // Quarter spellings: `2Q2023`, `Q2 2023`, `2023Q2`.
        if let Some(quarter_index) = upper.find('Q') {
            let (before, after) = upper.split_at(quarter_index);
            let after = &after[1..];
            let before = before.trim();
            let after = after.trim();

            let (quarter_digit, year_text) = if !before.is_empty() && before.len() == 1 {
                // `2Q2023`
                (before, after)
            } else if !before.is_empty() {
                // `2023Q2`
                (after, before)
            } else {
                // `Q2 2023`
                match after.split_once(char::is_whitespace) {
                    Some((q, y)) => (q, y.trim()),
                    None if after.len() > 1 => after.split_at(1),
                    None => return Err(unrecognised()),
                }
            };

            let quarter = match quarter_digit.trim() {
                "1" => Quarter::Q1,
                "2" => Quarter::Q2,
                "3" => Quarter::Q3,
                "4" => Quarter::Q4,
                _ => return Err(unrecognised()),
            };
            let year: u16 = year_text.trim().parse().map_err(|_| unrecognised())?;
            return Ok(Period::Quarter { year, quarter });
        }

        // `Jan 2023`.
        const MONTHS: [&str; 12] = [
            "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
        ];
        if let Some((month_text, year_text)) = upper.split_once(char::is_whitespace) {
            let month_text = month_text.trim_end_matches('.');
            if let Some(index) = MONTHS
                .iter()
                .position(|m| month_text.starts_with(m) && month_text.len() <= 9)
            {
                let year: u16 = year_text.trim().parse().map_err(|_| unrecognised())?;
                return Period::month(year, index as u32 + 1);
            }
        }

        Err(unrecognised())
    }
}

impl fmt::Display for Period {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Period::Year(y) => write!(f, "{y}"),
            Period::Quarter { year, quarter } => write!(f, "{quarter} {year}"),
            Period::Month { year, month } => write!(f, "{year}-{month:02}"),
            Period::Window {
                from_year,
                from_month,
                to_year,
                to_month,
            } => write!(
                f,
                "{from_year}-{from_month:02}..{to_year}-{to_month:02}"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_spellings_hdb_tables_use() {
        assert_eq!(Period::parse("2023").unwrap(), Period::Year(2023));
        assert_eq!(
            Period::parse("2Q2023").unwrap(),
            Period::Quarter {
                year: 2023,
                quarter: Quarter::Q2
            }
        );
        assert_eq!(
            Period::parse("Q2 2023").unwrap(),
            Period::Quarter {
                year: 2023,
                quarter: Quarter::Q2
            }
        );
        assert_eq!(
            Period::parse("2023Q2").unwrap(),
            Period::Quarter {
                year: 2023,
                quarter: Quarter::Q2
            }
        );
        assert_eq!(
            Period::parse("Jan 2023").unwrap(),
            Period::Month {
                year: 2023,
                month: 1
            }
        );
        assert_eq!(
            Period::parse("2023-03").unwrap(),
            Period::Month {
                year: 2023,
                month: 3
            }
        );
    }

    #[test]
    fn refuses_to_guess_at_an_unrecognisable_period() {
        assert!(matches!(
            Period::parse("sometime in the nineties"),
            Err(PeriodError::Unrecognised { .. })
        ));
        assert!(matches!(Period::parse(""), Err(PeriodError::Unrecognised { .. })));
    }

    #[test]
    fn a_year_contains_its_quarters_and_months() {
        let year = Period::Year(2023);
        assert!(year.contains(Period::Quarter {
            year: 2023,
            quarter: Quarter::Q3
        }));
        assert!(year.contains(Period::month(2023, 7).unwrap()));
        assert!(!year.contains(Period::Year(2024)));
        // Containment is directional: a quarter does not contain its year.
        assert!(!Period::Quarter {
            year: 2023,
            quarter: Quarter::Q3
        }
        .contains(year));
    }

    #[test]
    fn quarter_boundaries_are_right() {
        let q4 = Period::Quarter {
            year: 2023,
            quarter: Quarter::Q4,
        };
        assert_eq!(q4.start(), Period::month(2023, 10).unwrap().start());
        assert_eq!(q4.end(), Period::month(2023, 12).unwrap().start());
    }

    #[test]
    fn overlap_is_symmetric_and_detects_shared_months() {
        let survey = Period::window(2023, 11, 2024, 3).unwrap();
        let year = Period::Year(2024);
        assert!(survey.overlaps(year));
        assert!(year.overlaps(survey));
        assert!(!survey.overlaps(Period::Year(2022)));
        // ...but neither contains the other, so they must not be joined blindly.
        assert!(!survey.contains(year));
        assert!(!year.contains(survey));
    }

    #[test]
    fn a_backwards_window_is_rejected() {
        assert!(matches!(
            Period::window(2024, 3, 2023, 11),
            Err(PeriodError::BackwardsRange { .. })
        ));
        assert!(matches!(
            Period::month(2023, 13),
            Err(PeriodError::BadMonth { month: 13 })
        ));
    }
}
