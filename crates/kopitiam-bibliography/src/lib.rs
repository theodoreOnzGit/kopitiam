//! # KOPITIAM Bibliography Engine
//!
//! References, citations, BibTeX/BibLaTeX/Hayagriva, and the citation graph.
//! The reference layer of KOPITIAM's **Literature Engine**.
//!
//! ---
//!
//! ## A citation is a claim about provenance
//!
//! **Read this before using anything below.**
//!
//! "This result is due to Okafor (2015)" is not a formatting decision. It is an
//! assertion about *who established what*, made in public, under the citing
//! author's name, and checked by people who care: reviewers, examiners, and the
//! researchers being cited.
//!
//! So the failure modes of this crate are not bugs. They are academic-integrity
//! problems:
//!
//! * Attributing a result to the **wrong paper** sends a reader to work that does
//!   not support the claim.
//! * **Inventing a DOI** produces an identifier that resolves — to somebody
//!   else's paper.
//! * Silently "correcting" **an author's name** into someone else's renames a
//!   real person in a published document.
//!
//! Every design decision here follows from that, and they all reduce to one rule:
//!
//! > ### NEVER FABRICATE A REFERENCE.
//! >
//! > If a citation cannot be resolved, **say so and keep the raw string.**
//!
//! There is no `Reference::guess()` in this API and there never will be.
//!
//! ---
//!
//! ## The five rules
//!
//! ### 1. Provenance is structural
//!
//! Every extracted reference carries its **document, its page, and the verbatim
//! source string** — and it is the *type system*, not a code review, that
//! enforces it. [`Reference::builder`] takes a [`Provenance`] by value and there
//! is no other constructor; [`Provenance`] has private fields, one constructor,
//! and no `Default`; [`SourceText`] cannot be empty; and `#[serde(try_from)]`
//! means deserialisation cannot smuggle in an un-sourced value either. An
//! un-sourced reference does not compile.
//!
//! Even the *failures* carry it: [`ParsedReference::Unparsed`] holds a
//! [`RawEntry`] with the page and the words on it.
//!
//! ### 2. Confidence is modelled, not assumed
//!
//! A reference-list line comes back as [`Parsed`], [`Partial`] — a reference
//! **plus the text we could not account for** — or [`Unparsed`]. The middle one
//! is the important one. A partially-understood reference is useful and cannot
//! mislead anyone. A *confidently wrong* one is a citation to the wrong paper.
//!
//! The parser physically cannot drop a remainder: it works by
//! **consume-and-account** (see [`entry`]), marking every byte it uses, so
//! anything left over is reported by construction rather than by diligence.
//!
//! ### 3. Names are never mangled
//!
//! This is the hard part, and getting it wrong is an insult. `van der Waals` is
//! not called *Waals*. `Kim Jong-un`'s family name comes **first**. `Martin
//! Luther King Jr.` is not called *Jr*. And `Kim Jong-un` and `John Smith` are
//! the same shape, so no rule can tell them apart from the string alone.
//!
//! [`Author`] therefore models given names, particles, family names and suffixes
//! properly; keeps [`Author::Literal`] for names it cannot split (CJK script,
//! brace-protected names); and — crucially — records [`NameConfidence`]. A name
//! whose split is only *assumed* is **never emitted in reordered form**:
//! `Mao Zedong` goes into a `.bib` file as `Mao Zedong`, never as `Zedong, Mao`.
//! We may be wrong internally; we are never wrong in public.
//!
//! See the [`author`] module docs — they are the most important in this crate.
//!
//! ### 4. Identifiers are validated, not accepted
//!
//! An [`Isbn`]'s **checksum is computed**, not its digits counted. A [`Doi`] must
//! carry a real `10.NNNN/` registrant prefix. An [`ArxivId`]'s encoded month must
//! be 01-12 (so `2013.00020` — a natural typo for `2103.00020` — is refused). A
//! [`ResourceUrl`] is parsed by a real URL parser.
//!
//! What fails validation is **reported as an [`Anomaly`]**, never accepted. A
//! wrong identifier in a bibliography is worse than no identifier at all.
//!
//! ### 5. There is no network, and no *pretend* network
//!
//! This crate never phones CrossRef, arXiv, or anybody else. What it does instead
//! is leave a **seam**: [`resolve::ResolutionRequest`] describes a reference that
//! *would* need a lookup, and [`resolve::NullResolver`] **errors** rather than
//! returning a plausible answer — because a stub resolver returning invented
//! metadata is the single most dangerous thing this crate could ship.
//!
//! See [`resolve`]. Wiring `kopitiam-web` to it is a few lines, in a crate above
//! this one.
//!
//! [`Parsed`]: ParsedReference::Parsed
//! [`Partial`]: ParsedReference::Partial
//! [`Unparsed`]: ParsedReference::Unparsed
//!
//! ---
//!
//! ## The gap this fills
//!
//! [`kopitiam_document`]'s citation model is, in its entirety:
//!
//! ```ignore
//! pub struct Citation { pub text: String }
//! ```
//!
//! The Document Engine faithfully records that a citation was **seen**, and does
//! nothing further with it — correctly, since its job is layout, not literature.
//! This crate turns that string into structured, resolvable knowledge, and emits
//! `paper A cites paper B` into [`kopitiam_ontology`]'s shared graph.
//!
//! That last part is the point of the whole exercise. **A bibliography is the
//! graph of what a field knows and who established it.** See [`knowledge`].
//!
//! ---
//!
//! ## Example
//!
//! ```no_run
//! use kopitiam_bibliography::{
//!     bibtex::{Dialect, emit_references},
//!     extract_pdf, to_graph,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let bibliography = extract_pdf("paper.pdf")?;
//!
//! println!("{} reference-list entries", bibliography.entries().len());
//!
//! // What we could not work out -- READ THIS, always.
//! for anomaly in bibliography.anomalies() {
//!     println!("UNRESOLVED: {}", anomaly.summary());
//! }
//! // ...and the assumptions specifically, which are the ones that could be
//! // confidently wrong rather than honestly absent.
//! for assumption in bibliography.assumptions() {
//!     println!("ASSUMED:    {}", assumption.summary());
//! }
//!
//! // Which citations point at nothing? Usually a finding about OUR extraction.
//! for citation in bibliography.unresolved_citations() {
//!     println!("citation matched no entry: {:?}", citation.citation());
//! }
//!
//! // Emit a .bib for the LaTeX workflow.
//! let references: Vec<_> = bibliography.references().cloned().collect();
//! print!("{}", emit_references(&references, Dialect::Biblatex));
//!
//! // And put the citation graph into the knowledge graph, so the paper never
//! // has to be read again.
//! let graph = to_graph(&bibliography);
//! println!("{} entities, {} edges", graph.entities.len(), graph.relationships.len());
//! # Ok(())
//! # }
//! ```
//!
//! ---
//!
//! ## Provenance of the test corpus
//!
//! The reference strings in the test suite are **neutral academic examples**:
//! plausible computational-linguistics and computer-science references, some
//! clearly synthetic and some reproducing the shapes of real, widely-cited
//! papers. They are chosen to reproduce the things a real typesetter does to a
//! reference list rather than to stand in for any particular publication.
//!
//! Everything a real typesetter does — hyphenating across line breaks, splitting
//! URLs at `/` with no hyphen, printing an article number `111144` as `111 144`,
//! dropping the "PhD thesis" designator so that a dissertation is
//! indistinguishable from a book — is exercised by those examples, and **not one
//! of those hazards would have appeared in a naive fixture written from scratch.**
//! That lesson was learned expensively in this same session by `kopitiam-plot`,
//! whose synthetic tests all passed while four real bugs survived.
//!
//! No real author's bibliography is reproduced wholesale anywhere in this crate.
//! The DOIs, ISBNs and ISSNs in the tests are used purely as validation fixtures
//! — a checksum test needs a real checksum — and are never attached to a
//! fabricated work in a way that would misattribute it.

#![forbid(unsafe_code)]

pub mod anomaly;
pub mod author;
pub mod bibliography;
pub mod bibtex;
pub mod citation;
pub mod entry;
pub mod error;
pub mod extract;
pub mod hayagriva;
pub mod identifier;
pub mod knowledge;
pub mod provenance;
pub mod reference;
pub mod resolve;
pub mod text;

pub use anomaly::Anomaly;
pub use author::{Author, AuthorList, GivenName, NameConfidence, PersonName};
pub use bibliography::{Bibliography, ResolvedCitation};
pub use citation::{CitationRef, SourcedCitation};
pub use entry::{ParsedReference, RawEntry};
pub use error::Error;
pub use extract::{extract_pages, extract_pdf};
pub use identifier::{ArxivId, Doi, IdentifierError, Identifiers, Isbn, Issn, ResourceUrl};
pub use knowledge::{KnowledgeGraph, to_graph};
pub use provenance::{
    DocumentId, LineNumber, Locator, PageNumber, Provenance, ProvenanceError, SourceText,
};
pub use reference::{
    CitationKey, EntryKind, FieldError, Page, PageRange, Reference, ReferenceBuilder, Year,
};
pub use resolve::{NullResolver, ReferenceResolver, ResolutionRequest, ResolveError};
pub use text::{HyphenJoin, fold_typography, normalise, squeeze};
