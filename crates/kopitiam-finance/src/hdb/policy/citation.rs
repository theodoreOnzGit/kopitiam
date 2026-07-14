//! Provenance: where a policy figure is written down, and whether anyone has
//! actually checked.
//!
//! CLAUDE.md's Scientific Standards require that "scientific software should
//! always remain explainable". For a housing-policy engine that is not a
//! stylistic preference, it is the whole product. The figure "$14,000" is
//! worthless — someone else's blog said $14,000 too, and they were quoting 2019
//! from memory. What has value is:
//!
//! > *§"Eligibility to Buy a New Flat", HDB InfoWEB, effective 11 September
//! > 2019, [hdb.gov.sg/…]*
//!
//! because that can be **disputed**. A citation is a falsifiable claim about the
//! world; a bare number is a rumour with a type.
//!
//! Every [`Citation`] in this module is mandatory and non-optional — there is no
//! `Option<Citation>` anywhere a figure is returned.

use serde::{Deserialize, Serialize};

use super::temporal::Date;

/// A document HDB (or Parliament, or the Ministry of National Development)
/// published, in which a rule is written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceDocument {
    /// Who published it — "Housing & Development Board", "Ministry of National
    /// Development", "Prime Minister's Office".
    pub publisher: String,
    /// The document's title, as published.
    pub title: String,
}

impl SourceDocument {
    /// A new source document reference.
    pub fn new(publisher: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            publisher: publisher.into(),
            title: title.into(),
        }
    }
}

/// Whether a citation has been checked against the source, or merely *written
/// down*.
///
/// This distinction exists because of how this crate was built, and refusing to
/// model it would be the first lie the crate told.
///
/// Every rule currently in [`rules`](super::rules) was transcribed **offline,
/// from recollection, with no network access**. The transcription is careful and
/// the effective dates are real policy dates — but not one figure in this crate
/// has been read off an HDB page by the process that wrote it. A `Citation`
/// therefore records not just *where the rule should be found* but *whether
/// anyone has looked*.
///
/// The URL in each citation is the instruction for how to fix that. Verifying a
/// provision is a mechanical task: fetch, read, and flip the variant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verification {
    /// The cited source was retrieved and the figure read off it on `retrieved`.
    Verified {
        /// When the source was checked.
        retrieved: Date,
    },
    /// The figure was entered from recollection and has **not** been checked
    /// against the cited source.
    ///
    /// This is not a placeholder to be quietly ignored. A caller building
    /// anything a person might act on should surface it. See
    /// [`super::HdbPolicy::unverified_provisions`].
    Unverified {
        /// What was relied on, and what should be done about it.
        note: String,
    },
}

/// Where a policy figure comes from. Mandatory on every figure this crate
/// returns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Citation {
    /// The document.
    pub document: SourceDocument,
    /// The part of it: a section heading, a table number, a paragraph.
    pub section: String,
    /// The date the source carries — a gazette date, a Rally date, a Budget
    /// date, or (for HDB's continuously-revised web pages, which are undated)
    /// the anchor date described on [`Citation::hdb_infoweb`].
    pub published: Date,
    /// Where to go and check. The crate cannot check it; you can.
    pub url: String,
    /// Whether anyone has checked. See [`Verification`].
    pub verification: Verification,
}

impl Citation {
    /// A citation whose source has been read.
    pub fn verified(
        document: SourceDocument,
        section: impl Into<String>,
        published: Date,
        url: impl Into<String>,
        retrieved: Date,
    ) -> Self {
        Self {
            document,
            section: section.into(),
            published,
            url: url.into(),
            verification: Verification::Verified { retrieved },
        }
    }

    /// A citation to a page on HDB's InfoWEB (`hdb.gov.sg`).
    ///
    /// # The anchor-date convention
    ///
    /// HDB's eligibility pages are **undated and continuously revised**. There
    /// is no edition, no revision number, and no "last updated" stamp that
    /// survives a policy change. A citation to such a page is therefore, on its
    /// own, a citation to *whatever it says today* — which is precisely the
    /// non-reproducibility this crate exists to eliminate.
    ///
    /// So `published` is used as an **anchor**: the earliest date on which we
    /// are confident the page said what this provision says. It is not a claim
    /// that the page was created that day. Paired with
    /// [`EffectiveRange::in_force_at_least_from`](super::temporal::EffectiveRange::in_force_at_least_from),
    /// it means: *"this rule was in force at least from here; earlier dates are
    /// outside what we model"* — and a query about an earlier date then fails
    /// loudly instead of guessing backwards.
    ///
    /// The resulting citation is [`Verification::Unverified`], because this
    /// crate was written with no network access and nothing in it has been read
    /// off the live page.
    pub fn hdb_infoweb(
        section: impl Into<String>,
        anchor_note: impl Into<String>,
        anchor: Date,
        url: impl Into<String>,
    ) -> Self {
        Self {
            document: SourceDocument::new("Housing & Development Board", "HDB InfoWEB"),
            section: format!("{} — {}", section.into(), anchor_note.into()),
            published: anchor,
            url: url.into(),
            verification: Verification::Unverified {
                note: "transcribed offline from recollection; not yet checked against the cited \
                       page. Re-verify before anyone relies on this figure."
                    .to_string(),
            },
        }
    }

    /// A citation to a dated announcement: a National Day Rally speech, a Budget
    /// statement, a press release, a parliamentary reply.
    ///
    /// Unlike an InfoWEB page these *are* dated, and the date is the policy's
    /// own commencement anchor, so the anchor-date convention does not apply.
    /// The [`Verification`] is still [`Verification::Unverified`] for the same
    /// reason: nothing here has been read off the source.
    pub fn announcement(
        publisher: impl Into<String>,
        title: impl Into<String>,
        section: impl Into<String>,
        published: Date,
        url: impl Into<String>,
    ) -> Self {
        Self {
            document: SourceDocument::new(publisher, title),
            section: section.into(),
            published,
            url: url.into(),
            verification: Verification::Unverified {
                note: "transcribed offline from recollection; not yet checked against the cited \
                       announcement. Re-verify before anyone relies on this figure."
                    .to_string(),
            },
        }
    }

    /// Whether this citation has actually been checked against its source.
    pub fn is_verified(&self) -> bool {
        matches!(self.verification, Verification::Verified { .. })
    }
}

impl std::fmt::Display for Citation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}, {} — {} ({}) <{}>{}",
            self.document.publisher,
            self.document.title,
            self.section,
            self.published,
            self.url,
            if self.is_verified() {
                ""
            } else {
                " [UNVERIFIED]"
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdb::policy::temporal::Date;

    #[test]
    fn offline_citations_admit_they_are_unverified() {
        let c = Citation::hdb_infoweb(
            "Eligibility to Buy a New Flat",
            "in force at least from the 11 Sep 2019 revision",
            Date::new(2019, 9, 11).unwrap(),
            "https://www.hdb.gov.sg/residential/buying-a-flat",
        );
        assert!(!c.is_verified());
        assert!(c.to_string().contains("[UNVERIFIED]"));
    }

    #[test]
    fn a_checked_citation_records_when_it_was_checked() {
        let c = Citation::verified(
            SourceDocument::new("HDB", "Annual Report"),
            "p. 12",
            Date::new(2024, 1, 1).unwrap(),
            "https://example.invalid",
            Date::new(2026, 7, 14).unwrap(),
        );
        assert!(c.is_verified());
        assert!(!c.to_string().contains("UNVERIFIED"));
    }
}
