//! The resolution seam — **where a network resolver would go, and does not**.
//!
//! # This crate has no network, and that is not a limitation to be worked around
//!
//! A great deal of what a bibliography tool would *like* to do requires asking
//! somebody: turning `Okafor 2015` into a DOI, completing an `et al.` into a
//! full author list, checking that a DOI resolves to the paper it claims to.
//! CrossRef, Semantic Scholar and arXiv all have APIs for exactly this.
//!
//! KOPITIAM's Offline First principle puts the web **below** existing knowledge,
//! native implementations and local AI, and `kopitiam-web`'s own crate docs put
//! it plainly: *"Reaching for the web is an admission that the runtime didn't
//! know something."*
//!
//! So this module contains **no resolver**. It contains the *shape* of one.
//!
//! # Why there is no stub resolver returning plausible metadata
//!
//! Because that is the single most dangerous thing this crate could ship.
//!
//! A `MockResolver` that returned a plausible-looking DOI for a query would be
//! indistinguishable, to every caller and to the knowledge graph, from a real
//! one. It would be *convenient*. It would compile. Its tests would pass. And it
//! would put fabricated identifiers — pointing at other people's papers — into
//! a scientist's bibliography.
//!
//! [`NullResolver`] therefore **errors**, exactly as `kopitiam-web`'s
//! `NullProvider` errors rather than returning an empty result set, and for
//! precisely the same reason: *"I did not look"* and *"there is nothing there"*
//! are different sentences, and a type system that lets them be confused will
//! eventually see them confused.
//!
//! # What a future resolver would be given
//!
//! [`ResolutionRequest`] — a reference we have, that lacks an identifier — and
//! [`ResolutionRequest::search_text`], the query string a search engine or a
//! CrossRef bibliographic-matching endpoint would be handed.
//!
//! Wiring `kopitiam-web` to this is a few lines, in a crate **above** this one
//! (`kopitiam-literature` is the obvious home). This crate does not depend on
//! `kopitiam-web` and must not: a bibliography *model* that could open a socket
//! is a bibliography model that will, one day, open a socket. See
//! `docs/ai-decisions/AID-0018.md`.
//!
//! # And what it would have to prove before we believed it
//!
//! Recorded here because it is the hard part, and because whoever writes that
//! resolver will need it:
//!
//! A search engine returning a paper with a similar title is **not** evidence
//! that it is the same paper. Titles collide; preprints and published versions
//! differ; a corrigendum shares a title with the thing it corrects. Before a
//! resolved DOI may be attached to a [`crate::Reference`], the candidate must
//! agree with what the document *actually printed* — at minimum the first
//! author's family name and the year, and preferably the container too. A
//! disagreement is a **refusal**, not a warning.
//!
//! [`ResolutionOutcome`] is shaped to force that: a resolver hands back a
//! candidate *and the evidence for it*, and the caller decides. There is no path
//! where a resolver simply writes a DOI into a reference.

use serde::{Deserialize, Serialize};

use crate::identifier::Identifiers;
use crate::reference::Reference;

/// A resolver was asked to identify a work and could not.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ResolveError {
    /// Resolution is switched off — which is the default, and should stay the
    /// default.
    ///
    /// **An error, not an empty result.** See the module docs.
    #[error(
        "reference resolution is disabled ({resolver}): nothing was looked up. \
         This is NOT the same as `no match was found`."
    )]
    Disabled {
        /// The resolver's name.
        resolver: String,
    },

    /// The network was unreachable, or the service was.
    #[error("could not reach the resolver ({resolver}): {reason}")]
    Unreachable {
        /// The resolver's name.
        resolver: String,
        /// What went wrong.
        reason: String,
    },

    /// The service answered, and its answer could not be understood.
    #[error("the resolver ({resolver}) returned something unreadable: {reason}")]
    Unreadable {
        /// The resolver's name.
        resolver: String,
        /// What went wrong.
        reason: String,
    },
}

impl ResolveError {
    /// Whether retrying could plausibly help.
    pub fn is_transient(&self) -> bool {
        matches!(self, Self::Unreachable { .. })
    }
}

/// A reference we have, that we cannot yet identify.
///
/// Built from a [`Reference`] whose [`Identifiers`] are empty — i.e. exactly the
/// references a resolver would be for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolutionRequest {
    /// The first author's family name as printed, if we have a trustworthy one.
    pub first_author: Option<String>,
    /// The title, if we have one.
    pub title: Option<String>,
    /// The container (journal, proceedings), if we have one.
    pub container: Option<String>,
    /// The year, if we have one.
    pub year: Option<i32>,
    /// **The verbatim source string.** Carried through so that a resolver's
    /// answer can be checked against what the document actually printed, rather
    /// than against our interpretation of it.
    pub verbatim: String,
}

impl ResolutionRequest {
    /// Builds a request from a reference that lacks any identifier.
    ///
    /// Returns `None` when the reference **already has** an identifier — there
    /// is nothing to resolve — or when it has nothing to search on.
    pub fn for_reference(reference: &Reference) -> Option<Self> {
        if reference.identifiers().any() {
            return None;
        }
        if reference.title().is_none() && reference.authors().is_empty() {
            return None;
        }

        Some(Self {
            first_author: reference
                .authors()
                .first()
                .and_then(|author| author.family())
                .map(str::to_string),
            title: reference.title().map(str::to_string),
            container: reference.container().map(str::to_string),
            year: reference.year().map(|y| y.get()),
            verbatim: reference.provenance().verbatim().as_str().to_string(),
        })
    }

    /// The query string a search engine or a bibliographic-matching endpoint
    /// would be handed.
    ///
    /// **This is the entire seam.** A caller in a crate above this one builds a
    /// `kopitiam_web::SearchQuery` from it in one line, without this crate ever
    /// having heard of HTTP:
    ///
    /// ```ignore
    /// let query = kopitiam_web::SearchQuery::new(request.search_text());
    /// let response = provider.search(&query)?;
    /// ```
    pub fn search_text(&self) -> String {
        let mut parts: Vec<&str> = Vec::new();
        if let Some(title) = &self.title {
            parts.push(title);
        }
        if let Some(author) = &self.first_author {
            parts.push(author);
        }
        if let Some(container) = &self.container {
            parts.push(container);
        }
        let mut text = parts.join(" ");
        if let Some(year) = self.year {
            text.push(' ');
            text.push_str(&year.to_string());
        }
        text.trim().to_string()
    }
}

/// What a resolver came back with.
///
/// Deliberately **not** a bare `Identifiers`. A resolver must hand back its
/// candidate *and the evidence for it*, so that the caller — not the resolver —
/// decides whether the evidence is good enough to attach a DOI to a scientist's
/// citation. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolutionOutcome {
    /// The identifiers the resolver believes belong to this work.
    pub identifiers: Identifiers,
    /// The title the resolver's candidate carries, **so it can be compared with
    /// the one the document printed**. A resolver that returned identifiers
    /// without them would be asking to be trusted blindly.
    pub candidate_title: Option<String>,
    /// The candidate's first author's family name, for the same reason.
    pub candidate_first_author: Option<String>,
    /// The candidate's year, for the same reason.
    pub candidate_year: Option<i32>,
    /// Which service said so, and when — a resolved identifier is a *dated
    /// claim by a third party*, not a fact, and the graph must record it as one.
    pub resolver: String,
}

/// A pluggable connection to a bibliographic-metadata service.
///
/// Mirrors `kopitiam_web::SearchProvider` and `kopitiam_ai`'s `ModelAdapter`
/// deliberately, and for the same reason: the platform must never be written
/// against one vendor, and must remain fully functional when that vendor is
/// absent, unaffordable, or gone.
///
/// # The contract
///
/// 1. **Never fabricate.** If you did not retrieve it, do not return it. A
///    resolver that cannot resolve returns `Err`. It does not return a
///    plausible-looking `ResolutionOutcome`, because a caller cannot tell a
///    fabricated one from a real one — and neither can the knowledge graph it
///    ends up in, nor the paper that is eventually printed from it.
///
/// 2. **Return the evidence, not just the answer.** Populate the `candidate_*`
///    fields so the caller can check your answer against what the document
///    actually says.
///
/// 3. **Fail honestly.** [`ResolveError::Disabled`] and
///    [`ResolveError::Unreachable`] mean different things, and a caller must be
///    able to tell "I did not look" from "I looked and the network was down".
pub trait ReferenceResolver {
    /// A stable identifier for this resolver (`"crossref"`, `"arxiv"`, `"null"`).
    fn name(&self) -> &str;

    /// Attempts to identify the work described by `request`.
    ///
    /// # Errors
    ///
    /// [`ResolveError`] — and `Err` means *nothing was established*. Nothing
    /// about the world may be inferred from it.
    fn resolve(&self, request: &ResolutionRequest) -> Result<ResolutionOutcome, ResolveError>;
}

/// The resolver for a KOPITIAM that does not phone anybody — which is the
/// default, and should stay the default.
///
/// Every call returns [`ResolveError::Disabled`].
///
/// # Why an error and not a "no match"
///
/// The lazy implementation of "resolution is off" is to return "no match found".
/// It compiles, it never fails, and every caller handles it without a special
/// case. It is also a lie: a caller told "no match" concludes *this paper has no
/// DOI*, and may well write that conclusion into a document.
///
/// So `NullResolver` refuses. The cost is that callers must handle an error they
/// might have preferred to ignore. **That cost is the feature.**
#[derive(Debug, Default, Clone, Copy)]
pub struct NullResolver;

impl ReferenceResolver for NullResolver {
    fn name(&self) -> &str {
        "null"
    }

    fn resolve(&self, _request: &ResolutionRequest) -> Result<ResolutionOutcome, ResolveError> {
        Err(ResolveError::Disabled {
            resolver: self.name().to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::author::parse_printed_name_list;
    use crate::identifier::Doi;
    use crate::provenance::{DocumentId, Provenance};
    use crate::reference::{EntryKind, Year};

    fn reference_without_a_doi() -> Reference {
        let doc = DocumentId::new("paper.pdf").unwrap();
        let provenance = Provenance::from_page(
            &doc,
            15,
            "R. Okafor, Experimental validation of statistical alignment models. \
             University of California, Berkeley, 2015.",
        )
        .unwrap();
        Reference::builder(provenance)
            .kind(EntryKind::Book)
            .authors(parse_printed_name_list("R. Okafor"))
            .title("Experimental validation of statistical alignment models")
            .year(Year::new(2015).unwrap())
            .build()
    }

    #[test]
    fn the_null_resolver_refuses_rather_than_reporting_no_match() {
        // The whole point. "I did not look" and "there is nothing there" are
        // different sentences, and this crate will not let them be confused.
        let request = ResolutionRequest::for_reference(&reference_without_a_doi()).unwrap();
        let error = NullResolver
            .resolve(&request)
            .expect_err("a disabled resolver must not return an outcome");

        assert!(matches!(error, ResolveError::Disabled { .. }));
        assert!(!error.is_transient(), "retrying a disabled resolver is pointless");
    }

    #[test]
    fn a_reference_that_already_has_an_identifier_needs_no_resolution() {
        let doc = DocumentId::new("paper.pdf").unwrap();
        let provenance = Provenance::from_page(&doc, 15, "a line").unwrap();
        let reference = Reference::builder(provenance)
            .identifiers(Identifiers {
                doi: Some(Doi::parse("10.1016/j.x").unwrap()),
                ..Default::default()
            })
            .title("A paper")
            .build();

        assert_eq!(
            ResolutionRequest::for_reference(&reference),
            None,
            "there is nothing to look up"
        );
    }

    #[test]
    fn the_request_carries_the_documents_own_words_so_an_answer_can_be_checked() {
        // A resolver's answer must be checkable against what the document
        // actually printed -- not against our interpretation of it.
        let request = ResolutionRequest::for_reference(&reference_without_a_doi()).unwrap();
        assert!(request.verbatim.contains("Okafor"));
        assert!(request.verbatim.contains("Berkeley"));
    }

    #[test]
    fn the_search_text_is_what_a_future_resolver_would_be_handed() {
        let request = ResolutionRequest::for_reference(&reference_without_a_doi()).unwrap();
        let query = request.search_text();
        assert!(query.contains("Experimental validation of statistical alignment models"));
        assert!(query.contains("Okafor"));
        assert!(query.contains("2015"));
    }

    #[test]
    fn the_search_text_is_deterministic() {
        let request = ResolutionRequest::for_reference(&reference_without_a_doi()).unwrap();
        let first = request.search_text();
        for _ in 0..100 {
            assert_eq!(request.search_text(), first);
        }
    }
}
