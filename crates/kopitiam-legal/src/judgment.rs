//! Case law: judgments, the authorities they cite, and the one thing this
//! crate will not do.
//!
//! # We do not classify ratio vs obiter. Ever.
//!
//! The *ratio decidendi* of a case — the proposition of law it actually
//! decides, which binds later courts — versus *obiter dicta* — everything
//! else the judge said, which does not bind — is **the** central skill of
//! common-law reasoning. It is contested. It is argued over by counsel, and
//! decided by later courts, sometimes decades later, sometimes differently by
//! different courts. Two judges of the same court routinely disagree about
//! what the ratio of a case was.
//!
//! There is no textual feature that marks it. It is not "the paragraph after
//! 'I therefore hold'". It is a judgment about which propositions were
//! *necessary* to the outcome, which requires understanding the outcome, the
//! pleaded issues, and the alternative routes the court could have taken.
//!
//! A tool that auto-labels paragraph [47] as "ratio" is not extracting; it is
//! **doing law**, badly, and someone will cite it. So:
//!
//! * [`Holding`] defaults to [`Holding::Unmarked`].
//! * The only way to mark a paragraph as ratio or obiter is
//!   [`Holding::marked_by`], which **requires the name of the human who made
//!   the call** and preserves their note.
//! * Nothing in this crate ever constructs anything but `Unmarked`.
//!
//! The same reasoning applies to "is this case still good law?". A case can be
//! overruled, distinguished, doubted, or simply quietly ignored. We record
//! [`Treatment`] where a document *states* it, and never infer it.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{Date, Provenance, ProvisionId};

/// A neutral citation, e.g. `[2020] SGCA 12`.
///
/// Kept as a validated string rather than parsed into (year, court, number),
/// because citation formats vary enormously by jurisdiction and era — neutral
/// citations, law-report citations, and parallel citations all coexist — and a
/// parser that assumes one format will silently mangle the others. The string
/// is what a human can look up, which is all this crate needs it to be.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Citation(String);

impl Citation {
    pub fn new(citation: impl Into<String>) -> Result<Self, crate::LegalError> {
        let citation = citation.into();
        if citation.trim().is_empty() {
            return Err(crate::LegalError::MissingProvenance {
                what: "case citation",
            });
        }
        Ok(Self(citation))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Citation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Whether a paragraph of a judgment states binding ratio or non-binding
/// obiter — **as marked by a named human**, never as inferred by this crate.
///
/// See the module docs. `Unmarked` is not a failure state; it is the honest
/// and overwhelmingly common one.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Holding {
    /// Nobody has classified this paragraph. **This is the default and the
    /// only value this crate ever produces on its own.**
    #[default]
    Unmarked,
    /// A human judged this to be part of the ratio.
    Ratio { marked_by: String, note: Option<String> },
    /// A human judged this to be obiter.
    Obiter { marked_by: String, note: Option<String> },
}

impl Holding {
    /// Records a *human's* classification. The `marked_by` argument is
    /// mandatory and there is no default for it: an unattributed legal
    /// judgment is exactly what this crate exists to prevent, and requiring
    /// the name makes the classification traceable to the person who is
    /// accountable for it.
    pub fn marked_by(
        classification: Classification,
        marked_by: impl Into<String>,
        note: Option<String>,
    ) -> Self {
        let marked_by = marked_by.into();
        match classification {
            Classification::Ratio => Holding::Ratio { marked_by, note },
            Classification::Obiter => Holding::Obiter { marked_by, note },
        }
    }

    pub fn is_marked(&self) -> bool {
        !matches!(self, Holding::Unmarked)
    }
}

/// What a human decided a paragraph was. Only ever supplied *to*
/// [`Holding::marked_by`], never produced by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Classification {
    Ratio,
    Obiter,
}

/// How a later court treated an earlier authority, **where a document states
/// it explicitly**. Never inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Treatment {
    Applied,
    Followed,
    Distinguished,
    Doubted,
    Overruled,
    /// Cited without any stated treatment. The common case.
    Cited,
}

/// An authority cited by a judgment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CitedAuthority {
    /// The case name or instrument title as written.
    name: String,
    citation: Option<Citation>,
    /// Only ever set where the judgment *says so*. Defaults to
    /// [`Treatment::Cited`].
    treatment: Treatment,
    /// Which paragraph of the citing judgment cited it.
    cited_at: ProvisionId,
    provenance: Provenance,
}

impl CitedAuthority {
    pub fn new(
        name: impl Into<String>,
        citation: Option<Citation>,
        cited_at: ProvisionId,
        provenance: Provenance,
    ) -> Self {
        Self {
            name: name.into(),
            citation,
            treatment: Treatment::Cited,
            cited_at,
            provenance,
        }
    }

    /// Records an *explicitly stated* treatment.
    pub fn with_stated_treatment(mut self, treatment: Treatment) -> Self {
        self.treatment = treatment;
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn citation(&self) -> Option<&Citation> {
        self.citation.as_ref()
    }

    pub fn treatment(&self) -> Treatment {
        self.treatment
    }

    pub fn cited_at(&self) -> &ProvisionId {
        &self.cited_at
    }

    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}

/// The header material of a judgment.
///
/// The *paragraphs* of the judgment live in the [`crate::Instrument`]'s
/// provision map like any other provision (numbered `[47]` — see
/// [`crate::NumberingScheme::JudgmentParagraph`]); this struct holds only what
/// is peculiar to a case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Judgment {
    case_name: String,
    citation: Citation,
    court: String,
    /// The date the judgment was handed down. Note this is *not* a
    /// commencement date — a judgment is not "in force from" a date the way a
    /// statute is; it states what the law always was (the declaratory theory),
    /// subject to prospective overruling. We record the date and draw no
    /// conclusions from it.
    decided_on: Date,
    /// The judge(s). "Coram" is the term of art.
    coram: Vec<String>,
    /// Marked-up holdings, keyed by paragraph. **Empty unless a human has
    /// marked them.**
    holdings: Vec<(ProvisionId, Holding)>,
    authorities: Vec<CitedAuthority>,
}

impl Judgment {
    pub fn new(
        case_name: impl Into<String>,
        citation: Citation,
        court: impl Into<String>,
        decided_on: Date,
    ) -> Self {
        Self {
            case_name: case_name.into(),
            citation,
            court: court.into(),
            decided_on,
            coram: Vec::new(),
            holdings: Vec::new(),
            authorities: Vec::new(),
        }
    }

    pub fn with_coram(mut self, coram: Vec<String>) -> Self {
        self.coram = coram;
        self
    }

    pub fn case_name(&self) -> &str {
        &self.case_name
    }

    pub fn citation(&self) -> &Citation {
        &self.citation
    }

    pub fn court(&self) -> &str {
        &self.court
    }

    pub fn decided_on(&self) -> Date {
        self.decided_on
    }

    pub fn coram(&self) -> &[String] {
        &self.coram
    }

    pub fn authorities(&self) -> &[CitedAuthority] {
        &self.authorities
    }

    pub fn add_authority(&mut self, authority: CitedAuthority) {
        self.authorities.push(authority);
    }

    /// Records a **human's** ratio/obiter marking for a paragraph.
    pub fn mark_holding(&mut self, paragraph: ProvisionId, holding: Holding) {
        self.holdings.retain(|(id, _)| id != &paragraph);
        self.holdings.push((paragraph, holding));
    }

    /// The holding marking for a paragraph. [`Holding::Unmarked`] unless a
    /// human has marked it — which is the answer for almost every paragraph
    /// of almost every judgment, and is correct.
    pub fn holding(&self, paragraph: &ProvisionId) -> Holding {
        self.holdings
            .iter()
            .find(|(id, _)| id == paragraph)
            .map(|(_, h)| h.clone())
            .unwrap_or_default()
    }

    pub fn holdings(&self) -> &[(ProvisionId, Holding)] {
        &self.holdings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numbering::parse_judgment_paragraph;

    fn judgment() -> Judgment {
        Judgment::new(
            "SYNTHETIC Alpha v Beta",
            Citation::new("[2099] SYNTH 1").unwrap(),
            "Synthetic Court of Appeal",
            Date::new(2099, 3, 1).unwrap(),
        )
    }

    #[test]
    fn holdings_are_unmarked_until_a_human_marks_them() {
        let j = judgment();
        let para = parse_judgment_paragraph("[47]").unwrap();
        assert_eq!(
            j.holding(&para),
            Holding::Unmarked,
            "this crate must never infer ratio or obiter"
        );
        assert!(!j.holding(&para).is_marked());
    }

    #[test]
    fn a_marked_holding_records_who_made_the_call() {
        let mut j = judgment();
        let para = parse_judgment_paragraph("[47]").unwrap();
        j.mark_holding(
            para.clone(),
            Holding::marked_by(
                Classification::Ratio,
                "T. Ong",
                Some("necessary to the disposal of the appeal".into()),
            ),
        );
        match j.holding(&para) {
            Holding::Ratio { marked_by, note } => {
                assert_eq!(marked_by, "T. Ong");
                assert!(note.is_some());
            }
            other => panic!("expected a marked ratio, got {other:?}"),
        }
    }

    #[test]
    fn a_cited_authority_defaults_to_cited_not_to_a_guessed_treatment() {
        let provenance = crate::synthetic::synthetic_provision("1", "x", 2020)
            .provenance()
            .clone();
        let a = CitedAuthority::new(
            "SYNTHETIC Gamma v Delta",
            Citation::new("[2098] SYNTH 4").ok(),
            parse_judgment_paragraph("[12]").unwrap(),
            provenance,
        );
        assert_eq!(
            a.treatment(),
            Treatment::Cited,
            "whether a case was followed or distinguished is not textually inferable"
        );
    }
}
