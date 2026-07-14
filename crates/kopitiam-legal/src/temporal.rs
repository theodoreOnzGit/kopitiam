//! Temporal validity, amendment, and as-at-date querying.
//!
//! # "What does section 12 say?" is not a well-formed question
//!
//! It has no answer. Section 12 said one thing in 2018, was amended in 2021,
//! and may have been repealed since. The only well-formed question is:
//!
//! > **What did section 12 say *as at* 3 March 2021?**
//!
//! This is not pedantry. It is the single most common way legal software is
//! wrong, and it is wrong *silently*: a tool that stores "the text of s 12"
//! returns whatever version it happened to ingest, with total confidence,
//! to a user asking about a dispute that arose three years earlier. Every
//! part of that answer looks right. All of it is useless, and worse than
//! useless if relied on.
//!
//! So this crate makes the as-at date **structurally unavoidable**:
//!
//! * A [`crate::Provision`] cannot be constructed without a [`Validity`].
//! * The only accessor that returns a provision's text takes an
//!   [`AsAtDate`]: [`ProvisionHistory::as_at`]. There is deliberately **no**
//!   `fn text(&self) -> &str`, because if it existed people would call it.
//! * The one un-dated escape hatch, [`ProvisionHistory::latest_known`],
//!   returns a `#[must_use]` [`TemporalWarning`] that the caller cannot
//!   ignore without an explicit `let _ =`. Making the unsafe path *louder*
//!   than the safe one is the point.
//!
//! # The four honest answers
//!
//! An as-at query has more than two outcomes, and flattening them is itself
//! a bug. [`AsAtResult`] distinguishes:
//!
//! * **in force** — here is the text that was law on that date;
//! * **not yet in force** — the provision existed on paper but had not
//!   commenced;
//! * **repealed** — it was law once, but not on your date (and here is what
//!   it last said, because that is usually exactly what the asker needs);
//! * **not recorded** — *we do not know*. Our source does not cover that
//!   date. This is a **correct answer**, and a tool that cannot give it will
//!   instead give a wrong one. A 2020 Revised Edition simply does not tell
//!   you what the section said in 2018 (see [`crate::DocumentVersion`]).

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{AsAtDate, Date, DocumentId, LegalError, Provenance, Provision, ProvisionId, VerbatimText};

/// The window during which a provision had legal effect.
///
/// The interval is **half-open**: `[in_force_from, in_force_until)`. That
/// is, `in_force_until` is the first date on which the provision is *no
/// longer* in force, not the last date on which it is. Commencement and
/// repeal conventions differ between jurisdictions and even between
/// instruments, so rather than encode a guess we state the convention
/// explicitly here and require ingestion to normalise into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "ValidityRepr", into = "ValidityRepr")]
pub struct Validity {
    in_force_from: Date,
    in_force_until: Option<Date>,
}

impl Validity {
    /// A provision in force from `from` until further notice.
    pub fn from(from: Date) -> Self {
        Self {
            in_force_from: from,
            in_force_until: None,
        }
    }

    /// A provision in force over the half-open window `[from, until)`.
    ///
    /// Rejects an inverted or empty window: a provision cannot cease to be
    /// in force before (or on) the day it commenced.
    pub fn between(from: Date, until: Date) -> Result<Self, LegalError> {
        if until <= from {
            return Err(LegalError::InvalidValidity {
                detail: format!("in-force window [{from}, {until}) is empty or inverted"),
            });
        }
        Ok(Self {
            in_force_from: from,
            in_force_until: Some(until),
        })
    }

    pub fn in_force_from(&self) -> Date {
        self.in_force_from
    }

    /// The first date on which this provision is *no longer* in force.
    /// `None` means "still in force, so far as our source records".
    pub fn in_force_until(&self) -> Option<Date> {
        self.in_force_until
    }

    /// Whether this provision was in force on `as_at`.
    pub fn covers(&self, as_at: AsAtDate) -> bool {
        let d = as_at.date();
        d >= self.in_force_from && self.in_force_until.is_none_or(|until| d < until)
    }

    /// Whether two windows overlap. An overlap between two versions of the
    /// *same* provision is a contradiction in the source (or in our
    /// ingestion of it) and is reported as
    /// [`crate::AnomalyKind::OverlappingValidity`] rather than silently
    /// resolved by picking one.
    pub fn overlaps(&self, other: &Validity) -> bool {
        let starts_before_other_ends =
            other.in_force_until.is_none_or(|until| self.in_force_from < until);
        let other_starts_before_self_ends =
            self.in_force_until.is_none_or(|until| other.in_force_from < until);
        starts_before_other_ends && other_starts_before_self_ends
    }

    /// Closes an open-ended window at `until`, as happens when a later
    /// amendment supersedes this version.
    pub(crate) fn close_at(&mut self, until: Date) -> Result<(), LegalError> {
        if until <= self.in_force_from {
            return Err(LegalError::InvalidValidity {
                detail: format!(
                    "cannot close a window starting {} at the earlier date {until}",
                    self.in_force_from
                ),
            });
        }
        self.in_force_until = Some(until);
        Ok(())
    }
}

impl fmt::Display for Validity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.in_force_until {
            Some(until) => write!(f, "in force {} to {until} (exclusive)", self.in_force_from),
            None => write!(f, "in force from {}", self.in_force_from),
        }
    }
}

/// Serde shadow enforcing [`Validity`]'s window invariant on the
/// deserialize path — same reasoning as [`crate::Provenance`]'s shadow.
#[derive(Serialize, Deserialize)]
struct ValidityRepr {
    in_force_from: Date,
    in_force_until: Option<Date>,
}

impl TryFrom<ValidityRepr> for Validity {
    type Error = LegalError;
    fn try_from(r: ValidityRepr) -> Result<Self, Self::Error> {
        match r.in_force_until {
            Some(until) => Validity::between(r.in_force_from, until),
            None => Ok(Validity::from(r.in_force_from)),
        }
    }
}

impl From<Validity> for ValidityRepr {
    fn from(v: Validity) -> Self {
        Self {
            in_force_from: v.in_force_from,
            in_force_until: v.in_force_until,
        }
    }
}

/// What an amending instrument did to a provision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmendmentOperation {
    /// The provision was newly inserted by the amending instrument.
    Inserted,
    /// The provision's text was replaced wholesale ("...is repealed and the
    /// following substituted therefor:").
    Substituted,
    /// The provision was repealed. No text is in force after commencement.
    Repealed,
    /// **A textual amendment instruction we did not mechanically apply.**
    ///
    /// Real amending Acts are mostly written as *edit scripts*: "in
    /// subsection (2), delete the word 'may' and substitute the word
    /// 'must'". Applying those mechanically is possible and it is also how
    /// you produce a confidently wrong consolidated text — the instructions
    /// are prose, they interact, they are order-dependent, and they
    /// occasionally do not match the text they claim to edit.
    ///
    /// So we do not apply them. We record the instruction verbatim, surface
    /// it, and let a human do the consolidation. "Here is the amendment
    /// instruction; the consolidated text is not derivable by this tool" is
    /// a correct answer. A plausible-looking consolidated text that nobody
    /// checked is not.
    TextualInstructionNotApplied { instruction: VerbatimText },
}

impl fmt::Display for AmendmentOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inserted => f.write_str("inserted"),
            Self::Substituted => f.write_str("substituted"),
            Self::Repealed => f.write_str("repealed"),
            Self::TextualInstructionNotApplied { .. } => {
                f.write_str("textual amendment (not applied)")
            }
        }
    }
}

/// A change made to a provision by an amending instrument, with the date it
/// took effect and provenance for the amending words themselves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Amendment {
    /// The instrument that made the change (e.g. an amending Act, or a
    /// contract endorsement).
    by: DocumentId,
    /// The date the change took effect. Note this is the *commencement* of
    /// the amendment, which is frequently **not** the date the amending Act
    /// was passed — an Act commonly passes in one year and commences in the
    /// next, by a separate commencement notification.
    commencement: Date,
    operation: AmendmentOperation,
    /// Where the amending words themselves are found. The amendment is an
    /// extracted item like any other and carries full provenance.
    provenance: Provenance,
}

impl Amendment {
    pub fn new(
        by: DocumentId,
        commencement: Date,
        operation: AmendmentOperation,
        provenance: Provenance,
    ) -> Self {
        Self {
            by,
            commencement,
            operation,
            provenance,
        }
    }

    pub fn by(&self) -> &DocumentId {
        &self.by
    }

    pub fn commencement(&self) -> Date {
        self.commencement
    }

    pub fn operation(&self) -> &AmendmentOperation {
        &self.operation
    }

    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}

/// The answer to an as-at-date query. See the module docs for why there are
/// four outcomes and not two.
#[derive(Debug, Clone, PartialEq)]
pub enum AsAtResult<'a> {
    /// The provision was in force on the queried date; here is its text.
    InForce(&'a Provision),
    /// The provision exists in our record but had not commenced by the
    /// queried date.
    NotYetInForce {
        /// The earliest commencement we know of.
        earliest_commencement: Date,
    },
    /// The provision had been repealed by the queried date. The text it last
    /// carried is included, because that is very often what the asker
    /// actually needs (e.g. conduct that occurred while it was still law).
    Repealed {
        repealed_on: Date,
        last_in_force_text: &'a Provision,
    },
    /// **We do not know.** Our source records no version covering that date.
    ///
    /// This is a correct and useful answer, and the reason this variant
    /// exists rather than falling back to the nearest version. A tool that
    /// cannot say "I don't know" will instead say something wrong.
    NotRecorded {
        /// The windows our source *does* cover, so the reader can see the
        /// shape of the gap.
        recorded_windows: Vec<Validity>,
    },
}

/// Returned alongside any provision text obtained **without** an as-at date.
///
/// `#[must_use]` so that ignoring it requires an explicit `let _ = ...`. The
/// design intent is that the un-dated path is *available* (sometimes you
/// genuinely want "the newest text we hold") but never *convenient*, and
/// never silent.
#[must_use = "this text was obtained WITHOUT an as-at date and may not be the law \
              on the date you care about; surface this warning to the reader or \
              use `as_at` instead"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalWarning {
    provision: ProvisionId,
    validity: Validity,
}

impl TemporalWarning {
    pub fn provision(&self) -> &ProvisionId {
        &self.provision
    }

    /// The validity window of the text that was actually returned.
    pub fn validity(&self) -> Validity {
        self.validity
    }
}

impl fmt::Display for TemporalWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WARNING: {} was returned without an as-at date. The text shown is the \
             latest version recorded ({}). It is NOT necessarily the law on the date \
             you care about. Query with an explicit as-at date.",
            self.provision, self.validity
        )
    }
}

/// Every recorded version of one provision, in force-order.
///
/// This is the *history*, not just the current text — CLAUDE.md's Scientific
/// Standards demand that provenance and derivation stay visible, and for
/// legislation the derivation *is* the amendment chain. A reader must be
/// able to see that s 12(3) was substituted in 2021, and by what.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProvisionHistory {
    id: ProvisionId,
    /// Sorted by `in_force_from`. Non-empty by construction.
    versions: Vec<Provision>,
}

impl ProvisionHistory {
    /// Starts a history from the provision's original enactment.
    pub fn new(original: Provision) -> Self {
        Self {
            id: original.id().clone(),
            versions: vec![original],
        }
    }

    pub fn id(&self) -> &ProvisionId {
        &self.id
    }

    /// Every recorded version, oldest first. This is what makes the
    /// amendment history *visible* rather than merely applied.
    pub fn versions(&self) -> &[Provision] {
        &self.versions
    }

    /// Records a superseding version of this provision.
    ///
    /// The currently-open version (if any) is closed at the new version's
    /// commencement date, so the windows tile without overlap. Rejects a
    /// replacement whose id differs from the history's, and one that
    /// commences before the version it supersedes.
    pub fn supersede(&mut self, replacement: Provision) -> Result<(), LegalError> {
        if replacement.id() != &self.id {
            return Err(LegalError::InvalidValidity {
                detail: format!(
                    "cannot supersede {} with a version of {}",
                    self.id,
                    replacement.id()
                ),
            });
        }
        let commencement = replacement.validity().in_force_from();
        if let Some(last) = self.versions.last_mut()
            && last.validity().in_force_until().is_none()
        {
            last.close_validity_at(commencement)?;
        }
        self.versions.push(replacement);
        self.versions
            .sort_by_key(|v| v.validity().in_force_from());
        Ok(())
    }

    /// **The primary interface of this crate.** What did this provision say
    /// on `as_at`?
    ///
    /// There is no un-dated equivalent that returns text (see
    /// [`Self::latest_known`] for the deliberately-awkward escape hatch).
    pub fn as_at(&self, as_at: AsAtDate) -> AsAtResult<'_> {
        if let Some(version) = self.versions.iter().find(|v| v.validity().covers(as_at)) {
            return AsAtResult::InForce(version);
        }

        let earliest = self
            .versions
            .iter()
            .map(|v| v.validity().in_force_from())
            .min();
        if let Some(earliest) = earliest
            && as_at.date() < earliest
        {
            return AsAtResult::NotYetInForce {
                earliest_commencement: earliest,
            };
        }

        // After every recorded window closed => repealed, and we can show
        // what it last said.
        let last_closed = self
            .versions
            .iter()
            .filter_map(|v| v.validity().in_force_until().map(|until| (until, v)))
            .max_by_key(|(until, _)| *until);
        if let Some((repealed_on, version)) = last_closed
            && as_at.date() >= repealed_on
            && self.versions.iter().all(|v| {
                v.validity()
                    .in_force_until()
                    .is_some_and(|until| until <= repealed_on)
            })
        {
            return AsAtResult::Repealed {
                repealed_on,
                last_in_force_text: version,
            };
        }

        // A hole between recorded windows: we genuinely do not know.
        AsAtResult::NotRecorded {
            recorded_windows: self.versions.iter().map(|v| v.validity()).collect(),
        }
    }

    /// The newest recorded version, **without** regard to any as-at date.
    ///
    /// Returns a `#[must_use]` [`TemporalWarning`] alongside it. This is the
    /// only way to get text out of a history without naming a date, and it
    /// is intentionally more awkward than [`Self::as_at`]: you must bind and
    /// deal with the warning. Use it only when you truly mean "whatever the
    /// latest text we hold is" — e.g. rendering a document for browsing —
    /// and propagate the warning to the reader.
    pub fn latest_known(&self) -> (&Provision, TemporalWarning) {
        let latest = self
            .versions
            .last()
            .expect("ProvisionHistory is non-empty by construction");
        let warning = TemporalWarning {
            provision: self.id.clone(),
            validity: latest.validity(),
        };
        (latest, warning)
    }

    /// Any pairs of versions whose in-force windows overlap. An overlap is a
    /// contradiction — the same provision cannot have said two different
    /// things on the same day — and is surfaced, never resolved by guessing.
    pub fn overlapping_versions(&self) -> Vec<(&Provision, &Provision)> {
        let mut out = Vec::new();
        for (i, a) in self.versions.iter().enumerate() {
            for b in &self.versions[i + 1..] {
                if a.validity().overlaps(&b.validity()) {
                    out.push((a, b));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u8, day: u8) -> Date {
        Date::new(y, m, day).unwrap()
    }

    fn at(y: i32, m: u8, day: u8) -> AsAtDate {
        AsAtDate::new(d(y, m, day))
    }

    #[test]
    fn validity_window_is_half_open() {
        let v = Validity::between(d(2020, 1, 1), d(2024, 1, 1)).unwrap();
        assert!(v.covers(at(2020, 1, 1)), "in force on its commencement day");
        assert!(v.covers(at(2023, 12, 31)));
        assert!(
            !v.covers(at(2024, 1, 1)),
            "in_force_until is the first day it is NOT in force"
        );
        assert!(!v.covers(at(2019, 12, 31)));
    }

    #[test]
    fn rejects_an_inverted_or_empty_window() {
        assert!(Validity::between(d(2024, 1, 1), d(2020, 1, 1)).is_err());
        assert!(Validity::between(d(2020, 1, 1), d(2020, 1, 1)).is_err());
    }

    #[test]
    fn overlap_detection() {
        let a = Validity::between(d(2020, 1, 1), d(2024, 1, 1)).unwrap();
        let b = Validity::from(d(2022, 1, 1));
        assert!(a.overlaps(&b));

        let c = Validity::from(d(2024, 1, 1));
        assert!(!a.overlaps(&c), "tiling windows do not overlap");
    }
}
