//! Ingestion: PDF pages -> [`PolicyDocument`].
//!
//! This module writes **no PDF parser and no table parser**. [`kopitiam_pdf`]
//! recovers text spans with geometry and font style; [`kopitiam_document`]
//! reconstructs headings, paragraphs, lists and tables from them, handles
//! two-column layouts, and merges paragraphs across page breaks. Both are
//! reused as-is. What this module adds is the *insurance* structure on top:
//! clauses, the definitions section, exclusions, the schedule, endorsements.
//!
//! # Why reconstruction runs one page at a time
//!
//! This is the one place where this crate deliberately uses
//! `kopitiam-document` against the grain, and it is worth explaining, because
//! it looks like a mistake.
//!
//! `kopitiam_document::reconstruct` takes *all* the pages and returns a
//! `Vec<Block>` — a flat stream of headings, paragraphs and tables with **no
//! page attribution on any block**. `Document::metadata.source_pages` is a
//! count, not a mapping. For a scientific paper that is fine: the reader wants
//! the prose, not the pagination.
//!
//! For a legal contract it is fatal. Provenance in this crate is mandatory and
//! includes the page (see [`crate::Provenance`]) — a citation a reader cannot
//! turn to in the paper document is not a citation. So ingestion calls
//! `reconstruct` on **one page at a time**, which makes every block's page
//! knowable by construction, and then does the cross-page joining itself at
//! the level that matters for a contract: the **clause**.
//!
//! That trade is not merely acceptable here, it is an improvement.
//! `reconstruct`'s own cross-page merge joins two paragraphs when the first
//! does not end a sentence and the second does not start one — a good heuristic
//! for prose. A policy clause routinely *does* end a sentence at a page break
//! and *does* continue with a fresh capitalised sentence in the same clause, so
//! that heuristic would split it. Clause numbering, by contrast, tells us
//! exactly where a clause ends: at the next clause number. Joining by clause
//! number is both more correct for this document type and page-attributed.
//!
//! The shortfall in `kopitiam-document` is real and worth fixing upstream
//! (blocks should carry their page); this module is the workaround, not the
//! endorsement of one.

use std::slice;

use kopitiam_document::{Block, Document, List, Paragraph, Table};
use kopitiam_pdf::Page;

use crate::anomaly::Anomaly;
use crate::classify::{self, Shape};
use crate::clause::{Clause, ClauseId, ClauseLine, ClauseRole};
use crate::definition::Definitions;
use crate::endorsement::{self, Endorsement, EndorsementId};
use crate::error::Error;
use crate::exclusion;
use crate::policy::{Parts, PolicyDocument};
use crate::provenance::{DocumentId, PageNumber, SectionPath};
use crate::schedule::{BenefitTable, Schedule, render_row};

/// The largest number a clause identifier's first component may be.
///
/// A paragraph opening `"2026 was a difficult year for..."` would otherwise
/// mint a clause numbered 2026, and a schedule row reading `"500 per claim"`
/// would mint clause 500. Real top-level clause numbers do not run to three
/// digits; years and money do. This bound is the whole guard, and it is
/// deliberately crude, because a clause identifier invented out of a stray
/// number would show up in citations as if the document had printed it.
const MAX_CLAUSE_COMPONENT: u32 = 99;

/// Words that may precede a clause number in a heading: `Section 4 —
/// Exclusions`, `Clause 7`, `Part II`.
const CLAUSE_NUMBER_PREFIXES: &[&str] = &["clause", "section", "part", "item", "article"];

/// Reads an insurance PDF into a [`PolicyDocument`].
///
/// # Errors
///
/// [`Error::Pdf`] if the file cannot be read or parsed; [`Error::Provenance`]
/// if a citation could not be built (see [`crate::ProvenanceError`]).
pub fn ingest_pdf(path: impl AsRef<std::path::Path>) -> Result<PolicyDocument, Error> {
    let path = path.as_ref();
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    let pages = kopitiam_pdf::extract(path)?;
    ingest_pages(DocumentId::new(name)?, &pages)
}

/// Reads already-extracted PDF pages into a [`PolicyDocument`].
///
/// The entry point for callers that already have [`kopitiam_pdf::Page`]s — and
/// the one the tests use, since a synthetic policy is built as pages rather
/// than as a PDF file.
///
/// # Errors
///
/// [`Error::Provenance`] if a citation could not be built back to its own
/// clause. That indicates the extractor produced text the document does not
/// contain, which is precisely the failure the provenance model exists to
/// catch, so it is an error rather than a warning.
pub fn ingest_pages(id: DocumentId, pages: &[Page]) -> Result<PolicyDocument, Error> {
    let mut builder = Builder::new(id);

    for page in pages {
        let number = PageNumber::new(page.number)?;
        // One page at a time — see the module docs for why.
        let document: Document = kopitiam_document::reconstruct(slice::from_ref(page));
        builder.take_page(number, document);
    }

    builder.finish(pages.len())
}

/// A clause under construction: text accumulates into it until the next clause
/// number begins.
struct Draft {
    id: ClauseId,
    heading: Option<String>,
    path: SectionPath,
    lines: Vec<ClauseLine>,
    /// Set for a clause built from a reconstructed table, so the table's cells
    /// can be turned into schedule entries once the clause (and therefore its
    /// citations) exists.
    table: Option<Table>,
    numbered: bool,
}

/// One heading on the open section stack.
struct OpenHeading {
    /// Depth in the *insurance* hierarchy — see [`Builder::effective_level`].
    /// Not [`kopitiam_document::Heading::level`], which is not comparable
    /// across numbered and unnumbered headings.
    level: usize,
    text: String,
    /// Whether the heading carried a printed clause number.
    numbered: bool,
}

struct Builder {
    id: DocumentId,
    /// The open section hierarchy, outermost first.
    headings: Vec<OpenHeading>,
    /// Every heading seen, for document classification.
    all_headings: Vec<String>,
    title: Option<String>,
    draft: Option<Draft>,
    clauses: Vec<Clause>,
    tables: Vec<(usize, Table)>,
    shape: Shape,
    unnumbered_on_page: usize,
    page: Option<PageNumber>,
}

impl Builder {
    fn new(id: DocumentId) -> Self {
        Self {
            id,
            headings: Vec::new(),
            all_headings: Vec::new(),
            title: None,
            draft: None,
            clauses: Vec::new(),
            tables: Vec::new(),
            shape: Shape::default(),
            unnumbered_on_page: 0,
            page: None,
        }
    }

    fn take_page(&mut self, page: PageNumber, document: Document) {
        if self.page != Some(page) {
            self.page = Some(page);
            self.unnumbered_on_page = 0;
        }
        if self.title.is_none() {
            self.title = document.title.clone();
        }
        self.shape.pages += 1;

        for block in document.blocks {
            match block {
                Block::Heading(heading) => self.take_heading(page, heading.level, &heading.text),
                Block::Paragraph(Paragraph { text }) => self.take_paragraph(page, &text),
                Block::List(list) => self.take_list(page, &list),
                Block::Table(table) => self.take_table(page, table),
                // A figure caption, a code listing or a block quote in an
                // insurance document is prose belonging to the open clause;
                // dropping it would silently remove text from a contract.
                Block::Figure(figure) => {
                    if let Some(caption) = figure.caption {
                        self.take_paragraph(page, &caption);
                    }
                }
                Block::CodeBlock(code) => self.take_paragraph(page, &code.text),
                Block::Quote(quote) => self.take_paragraph(page, &quote.text),
            }
        }
    }

    /// A heading both moves the section hierarchy **and**, when it carries a
    /// clause number (`4.2 War and Related Perils`), opens a clause.
    fn take_heading(&mut self, page: PageNumber, doc_level: usize, text: &str) {
        self.flush();
        self.all_headings.push(text.to_string());
        self.shape.headings += 1;

        let numbered = split_clause_number(text);
        let level = self.effective_level(doc_level, numbered.map(|(number, _)| number));

        // Pop the siblings and descendants this heading closes, *then* take the
        // path — so a clause's section path is its ancestors, not itself.
        self.headings.retain(|open| open.level < level);
        let path = self.current_path();
        self.headings.push(OpenHeading {
            level,
            text: text.to_string(),
            numbered: numbered.is_some(),
        });

        let Some((number, rest)) = numbered else {
            return;
        };
        let Ok(id) = ClauseId::printed(number) else {
            return;
        };
        let heading = (!rest.trim().is_empty()).then(|| rest.trim().to_string());
        // The heading line is seeded as the clause's first line, verbatim: a
        // clause's printed text *does* include its own heading, and dropping it
        // would mean citing a clause with words missing.
        let Ok(line) = ClauseLine::new(page, text) else {
            return;
        };

        self.draft = Some(Draft {
            id,
            heading,
            path,
            lines: vec![line],
            table: None,
            numbered: true,
        });
    }

    /// The heading's depth in the **insurance** hierarchy.
    ///
    /// [`kopitiam_document`] assigns a heading level from its font size
    /// relative to body text, falling back to the dotted depth of a numbered
    /// section prefix. Those two scales are **not comparable**, and mixing them
    /// silently wrecks the section hierarchy: a policy prints `Section 4 —
    /// Exclusions` in a large font (font-derived level 2) and its child clause
    /// `4.1` at body size (number-derived level 2), so the child would be
    /// treated as its parent's *sibling*, pop it off the stack, and lose the
    /// `Exclusions` section path — which is the only thing that makes clause
    /// 4.1 an exclusion (see [`crate::exclusion`]). That is the difference
    /// between reporting a war exclusion and reporting nothing.
    ///
    /// So the level is recomputed here on one consistent scale:
    ///
    /// * A **numbered** heading is as deep as its number: `4` is 1, `4.2` is 2,
    ///   `4.2.1` is 3. This is the hierarchy the document itself is asserting,
    ///   and it is unambiguous.
    /// * An **unnumbered banner** — a heading meaningfully larger than body text
    ///   (`kopitiam-document` level 1 or 2, i.e. at least ~1.3x body size:
    ///   `PART II`, `POLICY SCHEDULE`, `Endorsement No. 1`) — is level 0. It
    ///   closes everything and starts a new hierarchy, which is exactly what a
    ///   part/section banner does. Without this, a `POLICY SCHEDULE` banner
    ///   arriving after clause `5.1` would nest *inside* clause 5.1, and every
    ///   schedule row would inherit `Section 5 — Conditions` in its section
    ///   path — a section it has nothing to do with. Section path drives role
    ///   classification, so a polluted path is not cosmetic.
    /// * An **unnumbered subheading** (only slightly larger than body text,
    ///   level 3) nests *under* the deepest numbered heading currently open
    ///   rather than resetting it — so a bold `War Risks` subheading printed
    ///   inside `4. Exclusions` does not knock `Exclusions` out of the path,
    ///   which would stop its clauses being recognised as exclusions at all.
    ///   Two such subheadings at the same font level are siblings.
    fn effective_level(&self, doc_level: usize, number: Option<&str>) -> usize {
        if let Some(number) = number {
            return number.matches('.').count() + 1;
        }
        if doc_level <= 2 {
            return 0;
        }
        let deepest_numbered = self
            .headings
            .iter()
            .filter(|open| open.numbered)
            .map(|open| open.level)
            .max()
            .unwrap_or(0);
        deepest_numbered + doc_level
    }

    fn take_paragraph(&mut self, page: PageNumber, text: &str) {
        if text.trim().is_empty() {
            return;
        }

        // A paragraph opening with a clause number starts a new clause.
        if let Some((number, _)) = split_clause_number(text)
            && let Ok(id) = ClauseId::printed(number)
        {
            self.flush();
            if let Ok(line) = ClauseLine::new(page, text) {
                self.draft = Some(Draft {
                    id,
                    heading: None,
                    path: self.current_path(),
                    lines: vec![line],
                    table: None,
                    numbered: true,
                });
            }
            return;
        }

        self.append_line(page, text);
    }

    /// List items are appended as lines of the open clause.
    ///
    /// **Known loss, and it is `kopitiam-document`'s to fix**:
    /// [`kopitiam_document::List`] is `{ ordered: bool, items: Vec<String> }`
    /// — the item *marker* is stripped and discarded during reconstruction. So
    /// a policy's enumerated sub-clauses (`a) ...`, `1) ...`) arrive here with
    /// their identifiers already gone, and cannot be recovered. They are kept
    /// as text of the parent clause, which preserves the words (nothing is
    /// dropped) but loses the ability to cite `4.2(a)` as such.
    ///
    /// In practice most policy sub-clauses are printed as `(a)`, which
    /// `kopitiam-document`'s list detector does not match (its marker regex
    /// requires the label *before* the punctuation), so they arrive as
    /// paragraphs with their markers intact and are unaffected. The loss is
    /// real but narrower than it first appears.
    fn take_list(&mut self, page: PageNumber, list: &List) {
        for item in &list.items {
            self.append_line(page, item);
        }
    }

    /// A table becomes a clause of its own, whose verbatim text is the table
    /// rendered row by row (`cell | cell`). That rendering is what makes a
    /// schedule value citable: [`crate::Clause::cite`] can quote the whole row,
    /// which is the smallest unit of a schedule that means anything on its own.
    fn take_table(&mut self, page: PageNumber, table: Table) {
        self.flush();

        let mut lines = Vec::new();
        if !table.headers.is_empty()
            && let Ok(line) = ClauseLine::new(page, render_row(table.headers.iter().map(String::as_str)))
        {
            lines.push(line);
        }
        for row in &table.rows {
            if let Ok(line) = ClauseLine::new(page, render_row(row.iter().map(String::as_str))) {
                lines.push(line);
            }
        }
        if lines.is_empty() {
            return;
        }

        self.shape.table_rows += table.rows.len();
        self.shape.widest_table = self.shape.widest_table.max(table.headers.len());

        let id = ClauseId::Unnumbered {
            page,
            ordinal: self.unnumbered_on_page,
        };
        self.unnumbered_on_page += 1;

        self.draft = Some(Draft {
            id,
            heading: None,
            path: self.current_path(),
            lines,
            table: Some(table),
            numbered: false,
        });
        self.flush();
    }

    fn append_line(&mut self, page: PageNumber, text: &str) {
        let Ok(line) = ClauseLine::new(page, text) else {
            return;
        };

        match &mut self.draft {
            Some(draft) => draft.lines.push(line),
            None => {
                let id = ClauseId::Unnumbered {
                    page,
                    ordinal: self.unnumbered_on_page,
                };
                self.unnumbered_on_page += 1;
                self.draft = Some(Draft {
                    id,
                    heading: None,
                    path: self.current_path(),
                    lines: vec![line],
                    table: None,
                    numbered: false,
                });
            }
        }
    }

    fn current_path(&self) -> SectionPath {
        SectionPath::new(self.headings.iter().map(|open| open.text.clone()))
    }

    /// Seals the open draft into a [`Clause`], classifying its role.
    fn flush(&mut self) {
        let Some(draft) = self.draft.take() else {
            return;
        };

        let text = draft
            .lines
            .iter()
            .map(ClauseLine::text)
            .collect::<Vec<_>>()
            .join("\n");

        let role = role_for(&draft, &text);
        classify::count_clause(&mut self.shape, role, draft.numbered);

        let Ok(clause) = Clause::new(
            self.id.clone(),
            draft.id,
            draft.heading,
            draft.path,
            role,
            draft.lines,
        ) else {
            return;
        };

        if let Some(table) = draft.table {
            self.tables.push((self.clauses.len(), table));
        }
        self.clauses.push(clause);
    }

    fn finish(mut self, pages: usize) -> Result<PolicyDocument, Error> {
        self.flush();
        self.shape.pages = pages;

        let classification =
            classify::classify(self.title.as_deref(), &self.all_headings, self.shape);

        let definitions = Definitions::extract(&self.clauses)?;

        let mut schedule = Schedule::default();
        let mut benefit_tables = Vec::new();
        for (index, table) in &self.tables {
            let clause = &self.clauses[*index];
            if clause.role() == ClauseRole::Definition {
                // A glossary table; its rows are definitions, already picked up
                // by `Definitions::extract`.
                continue;
            }
            if table.headers.len() >= 3 {
                benefit_tables.push(BenefitTable::from_table(clause, &table.headers, &table.rows)?);
            } else {
                let rows: Vec<(String, String)> = table
                    .rows
                    .iter()
                    .filter(|row| row.len() >= 2)
                    .map(|row| (row[0].clone(), row[1].clone()))
                    .collect();
                for entry in Schedule::from_two_column_rows(clause, &rows)? {
                    schedule.push(entry);
                }
            }
        }

        let endorsements = self.build_endorsements()?;

        Ok(PolicyDocument::assemble(Parts {
            id: self.id,
            classification,
            clauses: self.clauses,
            definitions,
            schedule,
            benefit_tables,
            endorsements,
            pages,
        }))
    }

    /// Builds an [`Endorsement`] for every clause that sits under an
    /// endorsement heading.
    ///
    /// Structure decides, not language — the same principle as exclusion
    /// classification, and for the same reason. The phrase "shall not apply"
    /// appears in every write-back in every exclusions section ever drafted;
    /// treating it as endorsement language wherever it occurs would turn
    /// ordinary exclusions into contract amendments.
    fn build_endorsements(&self) -> Result<Vec<Endorsement>, Error> {
        let mut endorsements = Vec::new();

        for clause in &self.clauses {
            if clause.role() != ClauseRole::Endorsement {
                continue;
            }

            let effect = endorsement::parse_effect(clause.text());
            let label = clause
                .path()
                .headings()
                .iter()
                .rev()
                .find(|heading| endorsement::is_endorsement_heading(heading))
                .cloned()
                .or_else(|| clause.heading().map(str::to_string))
                .unwrap_or_else(|| clause.id().to_string());

            endorsements.push(Endorsement::new(
                EndorsementId::new(label)?,
                effective_date(clause.text()),
                clause.clone(),
                effect,
            ));
        }

        Ok(endorsements)
    }
}

/// Classifies a draft clause. Table clauses default to
/// [`ClauseRole::Schedule`]; everything else goes through the structural
/// classifier in [`crate::exclusion`].
///
/// **A definitions heading beats an endorsement banner.** An endorsement that
/// redefines a term ("Endorsement No. 2 ... 2.1 *Accident* means any sudden
/// event, whether or not violent") sits under both. If the endorsement banner
/// won, that clause would never reach the definitions machinery, the policy's
/// two contradictory meanings of *Accident* would never be compared, and
/// [`crate::Resolution::Conflicting`] — the whole mechanism for surfacing a
/// self-contradicting policy — could not fire. A redefinition is exactly the
/// case that mechanism exists for, so it must not be the case that escapes it.
fn role_for(draft: &Draft, text: &str) -> ClauseRole {
    let heading = draft.heading.as_deref();

    let structural = exclusion::classify(&draft.path, heading, text);

    if structural == ClauseRole::Definition {
        return ClauseRole::Definition;
    }

    if draft.path.any(endorsement::is_endorsement_heading)
        || heading.is_some_and(endorsement::is_endorsement_heading)
    {
        return ClauseRole::Endorsement;
    }

    // Every table in an insurance document that is not a glossary holds numbers.
    if draft.table.is_some() {
        return ClauseRole::Schedule;
    }

    structural
}

/// Pulls an endorsement's effective date out of its text, **exactly as
/// printed**. Never parsed into a calendar type — see [`Endorsement`]'s field
/// docs for why mis-parsing the day cover starts is not a risk worth taking.
fn effective_date(text: &str) -> Option<String> {
    const MARKERS: &[&str] = &[
        "with effect from ",
        "effective date: ",
        "effective date ",
        "effective from ",
    ];
    let lower = text.to_ascii_lowercase();

    MARKERS.iter().find_map(|marker| {
        let at = lower.find(marker)? + marker.len();
        let rest = &text[at..];
        let end = rest
            .find(['.', '\n', ','])
            .unwrap_or(rest.len())
            .min(40);
        let date = rest[..end].trim();
        (!date.is_empty()).then(|| date.to_string())
    })
}

/// Splits a leading clause number off a heading or paragraph.
///
/// Handles `4.2.1 Something`, `4. Something`, `Section 4 — Something` and
/// `Clause 7`. Returns `(number, rest)`.
///
/// Rejects anything whose components exceed [`MAX_CLAUSE_COMPONENT`], because
/// the alternative is minting clause "2026" from a paragraph that opens with a
/// year, and a citation to a clause number the document never printed is worse
/// than no citation at all.
pub(crate) fn split_clause_number(text: &str) -> Option<(&str, &str)> {
    let trimmed = text.trim_start();

    // An optional "Clause"/"Section"/"Part" prefix.
    let after_prefix = CLAUSE_NUMBER_PREFIXES
        .iter()
        .find_map(|prefix| {
            let rest = trimmed.get(..prefix.len())?;
            (rest.eq_ignore_ascii_case(prefix)
                && trimmed[prefix.len()..].starts_with(char::is_whitespace))
            .then(|| trimmed[prefix.len()..].trim_start())
        })
        .unwrap_or(trimmed);

    let bytes = after_prefix.as_bytes();
    if bytes.is_empty() || !bytes[0].is_ascii_digit() {
        return None;
    }

    let mut end = 0;
    while end < bytes.len() {
        let byte = bytes[end];
        // A digit, or an interior '.' with a digit after it ("4.2.1"). A
        // trailing '.' is sentence punctuation and stops the scan.
        let in_number = byte.is_ascii_digit()
            || (byte == b'.' && end + 1 < bytes.len() && bytes[end + 1].is_ascii_digit());
        if !in_number {
            break;
        }
        end += 1;
    }

    let number = &after_prefix[..end];
    let mut rest = &after_prefix[end..];

    // A trailing "." or ")" or a dash (with or without a space before it, as in
    // "Section 4 — Exclusions") is punctuation, not part of the heading text.
    rest = rest
        .trim_start()
        .trim_start_matches(['.', ')', ':', '\u{2013}', '\u{2014}', '-'])
        .trim_start();

    // The number must be followed by whitespace/punctuation, not by a letter
    // ("4th quarter" is not clause 4) — unless it is the entire line.
    let separated = after_prefix.len() == end
        || after_prefix[end..].starts_with(|c: char| !c.is_alphanumeric());
    if !separated {
        return None;
    }

    if number.is_empty() || !plausible_clause_number(number) {
        return None;
    }

    Some((number, rest))
}

fn plausible_clause_number(number: &str) -> bool {
    number
        .split('.')
        .all(|component| component.parse::<u32>().is_ok_and(|n| n <= MAX_CLAUSE_COMPONENT))
}

/// Reports what could not be determined about a built document. Called by
/// [`PolicyDocument::assemble`].
pub(crate) fn audit(policy: &PolicyDocument) -> Vec<Anomaly> {
    let mut anomalies = Vec::new();

    // Clause identifiers printed twice: a cross-reference to one is ambiguous.
    let mut seen: std::collections::BTreeMap<String, Vec<&Clause>> = Default::default();
    for clause in policy.clauses() {
        if let Some(label) = clause.id().printed_label() {
            seen.entry(label.to_string()).or_default().push(clause);
        }
    }
    for (label, clauses) in &seen {
        if clauses.len() > 1
            && let Ok(id) = ClauseId::printed(label)
        {
            anomalies.push(Anomaly::DuplicateClauseId {
                id,
                occurrences: clauses.iter().map(|c| c.provenance()).collect(),
            });
        }
    }

    // Cross-references pointing at clauses that are not here.
    for clause in policy.clauses() {
        for reference in clause.cross_references() {
            if policy.base_clause(reference.target()).is_none()
                && let Ok(provenance) = clause.cite(reference.raw())
            {
                anomalies.push(Anomaly::DanglingCrossReference {
                    from: clause.id().clone(),
                    raw: reference.raw().to_string(),
                    target: reference.target().clone(),
                    provenance,
                });
            }
        }
    }

    // Terms the policy defines twice, inconsistently.
    for (term, definitions) in policy.definitions().conflicts() {
        anomalies.push(Anomaly::ConflictingDefinition {
            term: term.to_string(),
            definitions: definitions
                .iter()
                .map(|definition| definition.provenance().clone())
                .collect(),
        });
    }

    // Schedule values we declined to type.
    for entry in policy.schedule().entries() {
        if let crate::ScheduleValue::Unparseable { raw, reason } = entry.value().value() {
            anomalies.push(Anomaly::UnparseableScheduleValue {
                label: entry.label().to_string(),
                raw: raw.clone(),
                reason: reason.clone(),
                provenance: entry.value().provenance().clone(),
            });
        }
    }

    // Endorsements pointing at clauses that are not here, and endorsements
    // whose effect we could not determine.
    for endorsement in policy.endorsements() {
        match endorsement.effect() {
            crate::EndorsementEffect::Unspecified => {
                anomalies.push(Anomaly::UnspecifiedEndorsement {
                    endorsement: endorsement.id().clone(),
                    provenance: endorsement.provenance(),
                });
            }
            effect => {
                if let Some(target) = effect.target()
                    && policy.base_clause(target).is_none()
                    && !matches!(effect, crate::EndorsementEffect::Adds { .. })
                {
                    anomalies.push(Anomaly::EndorsementTargetNotFound {
                        endorsement: endorsement.id().clone(),
                        target: target.clone(),
                        provenance: endorsement.provenance(),
                    });
                }
            }
        }
    }

    // Clauses whose structure and language disagree.
    for clause in policy.clauses() {
        if exclusion::signals_conflict(clause.role(), clause.text()) {
            anomalies.push(Anomaly::ConflictingClauseSignals {
                clause: clause.id().clone(),
                reason: format!(
                    "printed as {:?} by its section, but its wording reads as the opposite",
                    clause.role()
                ),
                provenance: clause.provenance(),
            });
        }
    }

    // A wording with no definitions section: every defined term will silently
    // fall back to plain English, which is how a policy gets read backwards.
    if policy.classification().class() == crate::DocumentClass::PolicyWording
        && policy.definitions().is_empty()
    {
        anomalies.push(Anomaly::NoDefinitionsSection {
            classified_as: policy.classification().class().to_string(),
        });
    }

    anomalies
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_a_dotted_clause_number() {
        assert_eq!(
            split_clause_number("4.2.1 The Company shall not be liable"),
            Some(("4.2.1", "The Company shall not be liable"))
        );
        assert_eq!(
            split_clause_number("4. Exclusions"),
            Some(("4", "Exclusions"))
        );
    }

    #[test]
    fn splits_a_prefixed_clause_number() {
        assert_eq!(
            split_clause_number("Section 4 — Exclusions"),
            Some(("4", "Exclusions"))
        );
        assert_eq!(split_clause_number("Clause 7"), Some(("7", "")));
    }

    #[test]
    fn a_year_is_not_a_clause_number() {
        // Would otherwise mint "clause 2026" and cite it as if the document
        // had printed it.
        assert_eq!(split_clause_number("2026 was a difficult year."), None);
        assert_eq!(split_clause_number("500 per claim"), None);
    }

    #[test]
    fn a_number_glued_to_a_word_is_not_a_clause_number() {
        assert_eq!(split_clause_number("4th quarter results"), None);
    }

    #[test]
    fn plain_prose_has_no_clause_number() {
        assert_eq!(split_clause_number("The Company shall pay."), None);
    }

    #[test]
    fn reads_an_effective_date_exactly_as_printed() {
        assert_eq!(
            effective_date("This endorsement takes effect with effect from 1 March 2026."),
            Some("1 March 2026".to_string())
        );
        assert_eq!(effective_date("No date here."), None);
    }
}
