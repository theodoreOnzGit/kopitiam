//! Cross-references: legal instruments are a *graph*, not a list.
//!
//! # Why references must be resolved, and dangling ones must be reported
//!
//! You cannot read s 12 on its own. It says:
//!
//! > **Subject to section 7**, and **notwithstanding subsection (2)**, a
//! > person who operates a widget **as defined in Part II** must ...
//!
//! Three other provisions have just been pulled into the meaning of this one.
//! A reader handed s 12 alone, with those references unresolved, has been
//! handed something that *looks* complete and is not. So the extraction is
//! not finished until the edges are drawn.
//!
//! And when an edge points nowhere — "subject to section 7" in an instrument
//! with no section 7 — that is **not noise to be filtered out**. It is one of
//! the most informative things the tool can tell you, because it means one of:
//!
//! * the reader is holding an **incomplete document** (very common: someone
//!   sent you pages 1-14 of a 30-page lease);
//! * the reference is to *another instrument* and we misread it as internal;
//! * the drafter **made a mistake**, which happens, and which matters.
//!
//! All three are invisible if the parser quietly drops what it cannot
//! resolve. So a dangling reference is surfaced as an
//! [`crate::AnomalyKind::DanglingCrossReference`], never discarded.
//!
//! # Relative references
//!
//! "subsection (2)", used inside s 12, means **s 12(2)** — not some free-
//! floating subsection 2. Resolution is therefore *relative to where the
//! reference was made*, which is why [`resolve_target`] takes the citing
//! provision's id. Getting this wrong silently links s 12 to the wrong
//! provision, which is worse than not linking it at all.

use std::fmt;
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::{
    numbering::{parse_statutory, NumberingScheme},
    Provenance, Provision, ProvisionComponent, ProvisionId,
};

/// The words the drafter used to make the reference.
///
/// We record the connective because it is *the drafter's own signal* about
/// how the two provisions interact — and we do **not** act on it, because
/// working out the actual effect of "notwithstanding" is legal construction.
/// Preserving the connective lets the reader see at a glance that s 12 is
/// *subordinate to* s 7 rather than merely mentioning it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceConnective {
    /// "subject to section 7" — the cited provision prevails over this one.
    SubjectTo,
    /// "notwithstanding subsection (2)" — this provision prevails over the
    /// cited one.
    Notwithstanding,
    /// "as defined in Part II" — a definitional pointer.
    AsDefinedIn,
    /// "in accordance with clause 4.2".
    InAccordanceWith,
    /// "for the purposes of section 3".
    ForThePurposesOf,
    /// "under section 12(3)", "pursuant to section 9".
    Under,
    /// A bare mention with no signalling connective: "see section 4".
    Plain,
}

impl fmt::Display for ReferenceConnective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::SubjectTo => "subject to",
            Self::Notwithstanding => "notwithstanding",
            Self::AsDefinedIn => "as defined in",
            Self::InAccordanceWith => "in accordance with",
            Self::ForThePurposesOf => "for the purposes of",
            Self::Under => "under",
            Self::Plain => "refers to",
        })
    }
}

/// What a reference points at.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceTarget {
    /// A provision of the same instrument.
    Internal(ProvisionId),
    /// A provision of a *different* instrument ("section 12 of the
    /// Companies Act"). We do not attempt to follow it — we do not have that
    /// instrument — but we record it, because an unfollowed external
    /// reference is a known gap rather than a silent one.
    External {
        instrument: String,
        provision: Option<ProvisionId>,
    },
    /// We recognised that a reference was being made but could not parse its
    /// target. Preserved verbatim rather than dropped.
    Unparsed(String),
}

impl fmt::Display for ReferenceTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Internal(id) => write!(f, "{id}"),
            Self::External {
                instrument,
                provision,
            } => match provision {
                Some(p) => write!(f, "{p} of {instrument}"),
                None => write!(f, "{instrument}"),
            },
            Self::Unparsed(raw) => write!(f, "{raw} (unparsed)"),
        }
    }
}

/// One cross-reference found in a provision's text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossReference {
    /// The literal words in the source, e.g. `"subject to section 7"`.
    raw: String,
    connective: ReferenceConnective,
    target: ReferenceTarget,
    /// Where the reference was made (not where it points).
    provenance: Provenance,
}

impl CrossReference {
    pub fn new(
        raw: impl Into<String>,
        connective: ReferenceConnective,
        target: ReferenceTarget,
        provenance: Provenance,
    ) -> Self {
        Self {
            raw: raw.into(),
            connective,
            target,
            provenance,
        }
    }

    pub fn raw(&self) -> &str {
        &self.raw
    }

    pub fn connective(&self) -> ReferenceConnective {
        self.connective
    }

    pub fn target(&self) -> &ReferenceTarget {
        &self.target
    }

    /// The provision that *makes* the reference.
    pub fn from(&self) -> &ProvisionId {
        self.provenance.provision()
    }

    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}

impl fmt::Display for CrossReference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} --{}--> {} ({:?})",
            self.from(),
            self.connective,
            self.target,
            self.raw
        )
    }
}

/// Matches the reference forms Commonwealth drafters actually use.
///
/// Deliberately anchored on the *connective plus a unit keyword* rather than
/// on bare numbers: text is full of numbers, and a parser that treats every
/// "(2)" as a cross-reference will generate a blizzard of false edges. A
/// false edge is worse than a missing one, because it asserts a relationship
/// between two provisions that the drafter never made.
static REFERENCE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?ix)
        (?P<connective>
              subject\s+to
            | notwithstanding
            | as\s+defined\s+in
            | in\s+accordance\s+with
            | for\s+the\s+purposes\s+of
            | pursuant\s+to
            | under
            | see
        )
        \s+
        (?P<unit>section|subsection|paragraph|clause|part|regulation|s\.|§)
        \s*
        (?P<target>[0-9A-Za-z().\u{2010}\u{2011}-]+)
        (?:\s+of\s+the\s+(?P<instrument>[A-Z][A-Za-z\s]*?Act|[A-Z][A-Za-z\s]*?Agreement|[A-Z][A-Za-z\s]*?Regulations))?
        "#,
    )
    .expect("reference regex is a compile-time constant")
});

/// Extracts every cross-reference made by a provision.
pub fn extract_references(provision: &Provision) -> Vec<CrossReference> {
    REFERENCE
        .captures_iter(provision.text())
        .filter_map(|caps| {
            let raw = caps.get(0)?.as_str().trim().to_string();
            let connective = match caps
                .name("connective")?
                .as_str()
                .to_lowercase()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .as_str()
            {
                "subject to" => ReferenceConnective::SubjectTo,
                "notwithstanding" => ReferenceConnective::Notwithstanding,
                "as defined in" => ReferenceConnective::AsDefinedIn,
                "in accordance with" => ReferenceConnective::InAccordanceWith,
                "for the purposes of" => ReferenceConnective::ForThePurposesOf,
                "under" | "pursuant to" => ReferenceConnective::Under,
                _ => ReferenceConnective::Plain,
            };

            let unit = caps.name("unit")?.as_str().to_lowercase();
            let target_text = caps.name("target")?.as_str().trim_end_matches(['.', ',', ';']);
            let instrument = caps.name("instrument").map(|m| m.as_str().to_string());

            let target = resolve_target(&unit, target_text, provision.id(), instrument);

            Some(CrossReference::new(
                raw,
                connective,
                target,
                provision.provenance().clone(),
            ))
        })
        .collect()
}

/// Turns the *unit keyword* plus the *target text* into a [`ReferenceTarget`],
/// **relative to the provision making the reference**.
///
/// The relative case is the one that matters: inside s 12, "subsection (2)"
/// means `s 12(2)`. We therefore graft the bare subsection onto the citing
/// provision's own section rather than inventing a top-level `(2)`.
pub fn resolve_target(
    unit: &str,
    target_text: &str,
    citing: &ProvisionId,
    instrument: Option<String>,
) -> ReferenceTarget {
    // A reference to another instrument: record it, do not follow it.
    if let Some(instrument) = instrument {
        let provision = parse_statutory(target_text).ok();
        return ReferenceTarget::External {
            instrument,
            provision,
        };
    }

    match unit {
        // Absolute: "section 12(3)".
        "section" | "s." | "§" | "regulation" => parse_statutory(target_text)
            .map(ReferenceTarget::Internal)
            .unwrap_or_else(|_| ReferenceTarget::Unparsed(target_text.to_string())),

        // Relative: "subsection (2)" inside s 12 means s 12(2).
        "subsection" | "paragraph" => {
            let Some(section) = citing
                .components()
                .iter()
                .find(|c| matches!(c, ProvisionComponent::Section(_)))
                .cloned()
            else {
                return ReferenceTarget::Unparsed(target_text.to_string());
            };
            // Reparse "(2)" / "(a)" by hanging it off the citing section, so
            // that depth-based disambiguation (see `numbering`) applies
            // correctly.
            let label = format!("{}{}", section_label(&section), normalise_brackets(target_text));
            parse_statutory(&label)
                .map(ReferenceTarget::Internal)
                .unwrap_or_else(|_| ReferenceTarget::Unparsed(target_text.to_string()))
        }

        "clause" => crate::numbering::parse(target_text, NumberingScheme::DecimalClause)
            .map(ReferenceTarget::Internal)
            .unwrap_or_else(|_| ReferenceTarget::Unparsed(target_text.to_string())),

        "part" => crate::numbering::parse_part(target_text)
            .map(ReferenceTarget::Internal)
            .unwrap_or_else(|_| ReferenceTarget::Unparsed(target_text.to_string())),

        _ => ReferenceTarget::Unparsed(target_text.to_string()),
    }
}

/// Renders just the section number of a `Section` component, for regrafting.
fn section_label(component: &ProvisionComponent) -> String {
    match component {
        ProvisionComponent::Section(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Ensures a relative target is bracketed: `2` -> `(2)`, `(2)` -> `(2)`.
fn normalise_brackets(target: &str) -> String {
    let trimmed = target.trim();
    if trimmed.starts_with('(') {
        trimmed.to_string()
    } else {
        format!("({trimmed})")
    }
}

/// The outcome of resolving one instrument's references against its own
/// contents.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReferenceResolution {
    /// References whose internal target exists in the instrument.
    pub resolved: Vec<CrossReference>,
    /// References whose internal target does **not** exist. These become
    /// [`crate::AnomalyKind::DanglingCrossReference`] anomalies. They are
    /// reported, never dropped.
    pub dangling: Vec<CrossReference>,
    /// References to other instruments, which we deliberately do not follow.
    pub external: Vec<CrossReference>,
    /// References we recognised but whose target we could not parse.
    pub unparsed: Vec<CrossReference>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::synthetic_provision;

    #[test]
    fn extracts_the_signalling_connectives() {
        let p = synthetic_provision(
            "12",
            "Subject to section 7, and notwithstanding subsection (2), a person must register.",
            2020,
        );
        let refs = extract_references(&p);
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].connective(), ReferenceConnective::SubjectTo);
        assert_eq!(
            refs[0].target(),
            &ReferenceTarget::Internal(parse_statutory("7").unwrap())
        );
        assert_eq!(refs[1].connective(), ReferenceConnective::Notwithstanding);
    }

    #[test]
    fn a_relative_subsection_reference_binds_to_the_citing_section() {
        // "notwithstanding subsection (2)" inside s 12 means s 12(2) —
        // NOT a free-floating "(2)".
        let p = synthetic_provision("12", "Notwithstanding subsection (2), a fee is payable.", 2020);
        let refs = extract_references(&p);
        assert_eq!(
            refs[0].target(),
            &ReferenceTarget::Internal(parse_statutory("12(2)").unwrap()),
            "a relative reference must bind to the section that makes it"
        );
    }

    #[test]
    fn an_external_reference_is_recorded_but_not_followed() {
        let p = synthetic_provision(
            "3",
            "A company registered under section 19 of the Companies Act may apply.",
            2020,
        );
        let refs = extract_references(&p);
        assert_eq!(refs.len(), 1);
        match refs[0].target() {
            ReferenceTarget::External {
                instrument,
                provision,
            } => {
                assert_eq!(instrument, "Companies Act");
                assert_eq!(provision.as_ref().unwrap().to_string(), "s 19");
            }
            other => panic!("expected an external reference, got {other:?}"),
        }
    }

    #[test]
    fn bare_numbers_do_not_become_phantom_references() {
        // Text is full of numbers. A parser that treats every "(2)" as an
        // edge asserts relationships the drafter never made.
        let p = synthetic_provision(
            "5",
            "The fee is $2 and the period is 14 days, being (2) weeks.",
            2020,
        );
        assert!(
            extract_references(&p).is_empty(),
            "no connective + unit keyword means no reference"
        );
    }
}
