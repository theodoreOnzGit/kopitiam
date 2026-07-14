//! Turning [`SourceLine`]s into an [`Instrument`].
//!
//! # What this parser will and will not do
//!
//! It recognises the *structural* markers of Commonwealth statutory and
//! contractual drafting — Part headings, section numbers, the `12.—(1)` opener,
//! bracketed subsection and paragraph markers, quoted definitions — and hangs
//! verbatim text off them with page provenance intact.
//!
//! It does **not** try to be clever. Where a line does not match a structural
//! marker, it is appended to the text of the provision currently open, because
//! that is what a continuation line is. Where it matches nothing and no
//! provision is open, it is reported as
//! [`crate::AnomalyKind::UnattributedText`] rather than dropped — the one thing
//! a legal extractor must never do is lose operative words silently.
//!
//! # Headings: font first, shape second
//!
//! A section heading ("Duty to register") sits on its own line above the
//! section. In a PDF it is *bold*, and [`kopitiam_pdf`] can tell us that, so we
//! use it. In plain text we have no such signal, and fall back to a shape
//! heuristic (short line, no terminal full stop, not itself a numbered
//! provision). The heuristic is fallible; when it is wrong the text still
//! reaches the reader, just attached as a heading rather than as body — a
//! visible, harmless error rather than an invisible, harmful one.

use std::sync::LazyLock;

use regex::Regex;

use crate::{
    definition::extract_definitions,
    numbering::{self, NumberingScheme},
    source::SourceLine,
    Anomaly, AnomalyKind, AsAtDate, Date, DefinitionScope, DocumentId, DocumentVersion, Instrument,
    InstrumentKind, LegalError, Numeral, Provenance, Provision, ProvisionComponent, ProvisionId,
    Validity, VerbatimText,
};

/// `PART II` / `Part 2`
static PART: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?i)part\s+([IVXLC]+|\d+)\s*$").expect("const regex"));

/// `12.—(1) text` or `12. (1) text` or `12.-(1) text` — the Singapore/UK
/// section opener, where the em-dash joins the section number to its first
/// subsection.
static SECTION_WITH_SUBSECTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(\d+[A-Z]*)\.\s*[\u{2014}\u{2013}\-]?\s*\(([^)]+)\)\s*(.*)$")
        .expect("const regex")
});

/// `12. text` — a section with no subsections.
static SECTION_PLAIN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\d+[A-Z]*)\.\s+(.*)$").expect("const regex"));

/// `(2) text` / `(a) text` / `(ii) text` — a bracketed level, continuing the
/// section currently open.
static BRACKETED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\(([^)]+)\)\s*(.*)$").expect("const regex"));

/// `1.2.3 text` — a decimal contract clause.
static DECIMAL_CLAUSE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$").expect("const regex"));

/// `[47] text` — a judgment paragraph.
static JUDGMENT_PARA: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[(\d+)\]\s*(.*)$").expect("const regex"));

/// Everything needed to ingest a document. All fields are mandatory because
/// every one of them ends up in the [`Provenance`] of every extracted item, and
/// none of them has a safe default.
pub struct IngestRequest<'a> {
    pub id: DocumentId,
    pub version: DocumentVersion,
    pub kind: InstrumentKind,
    /// When the provisions of this document came into force. Required: a
    /// provision without an in-force date is not a provision this crate can
    /// represent. For a contract this is the effective date; for an Act, the
    /// commencement.
    pub in_force_from: Date,
    pub lines: &'a [SourceLine],
}

/// Parses a document into an [`Instrument`], then audits it.
///
/// The returned instrument carries its anomalies; a caller that ignores
/// [`Instrument::anomalies`] is discarding the tool's honesty.
pub fn ingest(request: IngestRequest<'_>) -> Result<Instrument, LegalError> {
    let IngestRequest {
        id,
        version,
        kind,
        in_force_from,
        lines,
    } = request;

    let scheme = kind.numbering_scheme();
    let validity = Validity::from(in_force_from);
    let mut instrument = Instrument::new(id.clone(), version.clone(), kind);

    let mut parser = Parser {
        id,
        version,
        validity,
        scheme,
        current_part: None,
        current_section: None,
        current_id: None,
        last_paragraph: None,
        last_subparagraph: None,
        pending_heading: None,
        open: None,
        finished: Vec::new(),
        anomalies: Vec::new(),
    };

    for line in lines {
        parser.feed(line);
    }
    parser.flush(&mut instrument)?;

    for anomaly in parser.anomalies {
        instrument.add_anomaly(anomaly);
    }

    // Definitions are extracted from the provisions we just built, so they
    // inherit their provenance and their validity automatically.
    let definitions: Vec<_> = instrument
        .provisions()
        .flat_map(|history| {
            let (provision, _warning) = history.latest_known();
            let scope = definition_scope(provision.id(), provision.part(), provision.text());
            extract_definitions(provision, scope)
        })
        .collect();
    for definition in definitions {
        instrument.add_definition(definition);
    }

    // Audit as at the commencement date. The caller can re-audit at any other
    // date; this just guarantees the instrument never leaves ingestion with
    // its anomalies un-computed.
    instrument.audit(AsAtDate::new(in_force_from));
    Ok(instrument)
}

/// Works out how far a definition reaches, from the drafter's own words.
///
/// "In this Act" is instrument-wide. "In this section" / "In this Part" reach
/// only that far. This is read from the text rather than assumed, because
/// assuming instrument-wide scope for a section-scoped definition would apply
/// it where it has no force — a definition applied out of scope is a wrong
/// answer that looks like a right one.
fn definition_scope(
    id: &ProvisionId,
    part: Option<Numeral>,
    text: &str,
) -> DefinitionScope {
    let lower = text.to_lowercase();
    if lower.contains("in this section") || lower.contains("for the purposes of this section") {
        // Scope is the containing SECTION, not the subsection the words happen
        // to sit in: "In this section" in s 14(1) reaches all of s 14.
        let section: Vec<ProvisionComponent> = id
            .components()
            .iter()
            .take_while(|c| matches!(c, ProvisionComponent::Section(_)))
            .cloned()
            .collect();
        if !section.is_empty() {
            return DefinitionScope::Within(ProvisionId::new(section));
        }
    }
    if lower.contains("in this part")
        && let Some(part) = part
    {
        return DefinitionScope::Part(part);
    }
    // "In this Act" / "In this Agreement" / anything else in an interpretation
    // section: instrument-wide.
    DefinitionScope::Instrument
}

/// A provision under construction: its id, heading, page and accumulated text.
struct Open {
    id: ProvisionId,
    heading: Option<String>,
    /// The Part this provision sits in — context, not identity.
    part: Option<Numeral>,
    page: crate::PageNumber,
    text: String,
}

struct Parser {
    id: DocumentId,
    version: DocumentVersion,
    validity: Validity,
    scheme: NumberingScheme,
    current_part: Option<Numeral>,
    current_section: Option<ProvisionComponent>,
    /// The id of the last provision opened.
    current_id: Option<ProvisionId>,
    /// The ordinal of the last paragraph seen in the open subsection, and of
    /// the last sub-paragraph seen in the open paragraph. These drive the
    /// successor rule in [`Parser::classify_level`], which is what resolves the
    /// `(i)`-is-it-a-letter-or-a-Roman-one ambiguity.
    last_paragraph: Option<u32>,
    last_subparagraph: Option<u32>,
    pending_heading: Option<String>,
    open: Option<Open>,
    /// Provisions closed but not yet emitted. Buffered rather than emitted
    /// immediately so that a provision's text can keep accumulating across
    /// continuation lines until the next structural marker arrives.
    finished: Vec<Open>,
    anomalies: Vec<Anomaly>,
}

impl Parser {
    fn feed(&mut self, line: &SourceLine) {
        let text = line.text.trim();
        if text.is_empty() {
            return;
        }

        // A Part heading resets the section context.
        if let Some(caps) = PART.captures(text) {
            if let Ok(part) = numbering::parse_part(&caps[1])
                && let Some(ProvisionComponent::Part(numeral)) = part.components().first()
            {
                self.close();
                self.current_part = Some(*numeral);
                self.current_section = None;
                self.current_id = None;
                self.reset_enumeration();
            }
            return;
        }

        match self.scheme {
            NumberingScheme::Statutory => self.feed_statutory(line, text),
            NumberingScheme::DecimalClause => self.feed_decimal(line, text),
            NumberingScheme::JudgmentParagraph => self.feed_judgment(line, text),
        }
    }

    fn feed_statutory(&mut self, line: &SourceLine, text: &str) {
        // `12.—(1) ...`
        if let Some(caps) = SECTION_WITH_SUBSECTION.captures(text) {
            self.close();
            let section = section_component(&caps[1]);
            self.current_section = Some(section.clone());
            let id = self
                .base()
                .child(section)
                .child(bracketed_component(&caps[2], 0));
            self.begin(id, line, &caps[3]);
            return;
        }

        // `12. ...` (no subsections)
        if let Some(caps) = SECTION_PLAIN.captures(text) {
            self.close();
            let section = section_component(&caps[1]);
            self.current_section = Some(section.clone());
            let id = self.base().child(section);
            self.begin(id, line, &caps[2]);
            return;
        }

        // `(2) ...` / `(a) ...` / `(ii) ...` — continues the open section.
        if let Some(caps) = BRACKETED.captures(text)
            && self.current_section.is_some()
        {
            let token = caps[1].to_string();
            let body = caps[2].to_string();
            let level = self.classify_level(&token);
            self.close();
            let id = self.id_at_level(level).child(numbering::classify_bracketed_public(&token, level));
            self.begin(id, line, &body);
            return;
        }

        self.continue_or_report(line, text);
    }

    /// Decide which level of the hierarchy a bare bracketed token sits at,
    /// **by enumeration successorship**.
    ///
    /// # This is the `(i)` problem, and this is the rule that solves it
    ///
    /// `(i)` is either the ninth *letter* (a paragraph, following `(h)`) or
    /// Roman *one* (a sub-paragraph, preceding `(ii)`). The glyphs are
    /// identical. Depth alone does not settle it, because both levels can
    /// legitimately appear at the same point in a section.
    ///
    /// What settles it is what came **before**. Legal enumerations run in
    /// sequence: paragraphs go (a), (b), ... (h), (i), (j); sub-paragraphs go
    /// (i), (ii), (iii). So a token is read as the *successor of whatever
    /// sequence is currently open*:
    ///
    /// * `(i)` after `(h)` — the next letter — is a **paragraph**.
    /// * `(i)` after `(a)` — not the next letter (that would be `(b)`), but it
    ///   *is* Roman one, opening a fresh sub-paragraph sequence — is a
    ///   **sub-paragraph**.
    /// * `(ii)` after `(i)` — the next Roman — is a **sub-paragraph**.
    ///
    /// That is exactly the rule a human reader applies, and it is decidable
    /// from the document alone with no guessing. Where the token is the
    /// successor of *neither* open sequence, we fall back to "deeper than the
    /// currently open level", and the numbering module records the token
    /// verbatim if it cannot classify it at all.
    fn classify_level(&self, token: &str) -> usize {
        // A digit-led token is always a subsection: (3), (3A).
        if token.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            return 0;
        }

        let as_alpha = numbering::alpha_value(token);
        let as_roman = numbering::roman_value(token);

        // Continues the paragraph sequence? (a) -> (b); (h) -> (i).
        if let Some(value) = as_alpha
            && value == self.last_paragraph.unwrap_or(0) + 1
        {
            return 1;
        }
        // Continues (or opens) the sub-paragraph sequence? (i) -> (ii).
        if let Some(value) = as_roman
            && value == self.last_subparagraph.unwrap_or(0) + 1
            && self.last_paragraph.is_some()
        {
            return 2;
        }
        // Neither sequence continues. Sit one level below whatever is open,
        // rather than inventing a position in a sequence we do not understand.
        if self.last_paragraph.is_some() {
            2
        } else {
            1
        }
    }

    /// The id of the unit a component at `level` hangs off: level 0 hangs off
    /// the section, level 1 off the subsection, level 2 off the paragraph.
    fn id_at_level(&self, level: usize) -> ProvisionId {
        let section = self
            .current_section
            .clone()
            .expect("caller checked current_section is Some");
        let base = self.base().child(section);
        let Some(current) = &self.current_id else {
            return base;
        };
        let mut components: Vec<ProvisionComponent> = Vec::new();
        let mut bracketed_seen = 0usize;
        for component in current.components() {
            let is_bracketed = matches!(
                component,
                ProvisionComponent::Subsection(_)
                    | ProvisionComponent::Paragraph(_)
                    | ProvisionComponent::Subparagraph(_)
                    | ProvisionComponent::SubSubparagraph(_)
            );
            if is_bracketed {
                if bracketed_seen >= level {
                    break;
                }
                bracketed_seen += 1;
            }
            components.push(component.clone());
        }
        if components.is_empty() { base } else { ProvisionId::new(components) }
    }

    fn feed_decimal(&mut self, line: &SourceLine, text: &str) {
        if let Some(caps) = DECIMAL_CLAUSE.captures(text)
            && let Ok(id) = numbering::parse_decimal_clause(&caps[1])
        {
            self.close();
            self.begin(id, line, &caps[2]);
            return;
        }
        self.continue_or_report(line, text);
    }

    fn feed_judgment(&mut self, line: &SourceLine, text: &str) {
        if let Some(caps) = JUDGMENT_PARA.captures(text)
            && let Ok(id) = numbering::parse_judgment_paragraph(&caps[1])
        {
            self.close();
            self.begin(id, line, &caps[2]);
            return;
        }
        self.continue_or_report(line, text);
    }

    /// A line that matched no structural marker. Either it continues the open
    /// provision, or it is a heading for the next one, or it is text we could
    /// not attribute — and that last case is *reported*, never dropped.
    fn continue_or_report(&mut self, line: &SourceLine, text: &str) {
        if let Some(open) = &mut self.open {
            open.text.push(' ');
            open.text.push_str(text);
            return;
        }

        if looks_like_heading(text, line) {
            self.pending_heading = Some(text.to_string());
            return;
        }

        // No provision is open and this is not a heading. We will not guess
        // where it belongs.
        let id = self
            .current_id
            .clone()
            .unwrap_or_else(|| ProvisionId::new(vec![ProvisionComponent::Unrecognized("?".into())]));
        if let Ok(verbatim) = VerbatimText::new(text) {
            self.anomalies.push(Anomaly::new(
                AnomalyKind::UnattributedText,
                Provenance::new(
                    self.id.clone(),
                    self.version.clone(),
                    id,
                    line.page,
                    verbatim,
                ),
            ));
        }
    }

    /// The id prefix a section hangs off.
    ///
    /// **Empty.** A provision's id is *section-rooted*: the Part is context, not
    /// identity, because section numbers run uniquely across a whole Act and
    /// every cross-reference to `s 7` says "section 7", never "Part I, section
    /// 7". Baking the Part in makes every such reference dangle. See
    /// [`crate::Provision::part`].
    fn base(&self) -> ProvisionId {
        ProvisionId::new(Vec::new())
    }

    fn begin(&mut self, id: ProvisionId, line: &SourceLine, body: &str) {
        // Keep the enumeration counters in step with the id we just built, so
        // the successor rule has the sequence state it needs for the next token.
        match id.components().last() {
            Some(ProvisionComponent::Section(_)) | Some(ProvisionComponent::Subsection(_)) => {
                self.reset_enumeration();
            }
            Some(ProvisionComponent::Paragraph(n)) => {
                self.last_paragraph = Some(n.value());
                self.last_subparagraph = None;
            }
            Some(ProvisionComponent::Subparagraph(n)) => {
                self.last_subparagraph = Some(n.value());
            }
            _ => {}
        }
        self.current_id = Some(id.clone());
        self.open = Some(Open {
            id,
            heading: self.pending_heading.take(),
            part: self.current_part,
            page: line.page,
            text: body.trim().to_string(),
        });
    }

    fn reset_enumeration(&mut self) {
        self.last_paragraph = None;
        self.last_subparagraph = None;
    }

    /// Closes the open provision, if any, stashing it for `flush`.
    fn close(&mut self) {
        if let Some(open) = self.open.take() {
            self.finished.push(open);
        }
    }

    fn flush(&mut self, instrument: &mut Instrument) -> Result<(), LegalError> {
        self.close();
        let finished = std::mem::take(&mut self.finished);
        for open in finished {
            // A provision whose body is empty carries no operative words. That
            // is normal for a section heading line like `12. Interpretation`
            // whose content is entirely in its subsections, so it is not an
            // anomaly — but it also cannot become a Provision, because
            // VerbatimText is non-empty by construction. We skip it, and the
            // subsections carry the text.
            let Ok(verbatim) = VerbatimText::new(open.text.clone()) else {
                continue;
            };
            let provenance = Provenance::new(
                self.id.clone(),
                self.version.clone(),
                open.id.clone(),
                open.page,
                verbatim,
            );
            let mut provision = Provision::new(provenance, self.validity);
            if let Some(heading) = open.heading {
                provision = provision.with_heading(heading);
            }
            if let Some(part) = open.part {
                provision = provision.with_part(part);
            }
            instrument.add_provision(provision)?;
        }
        Ok(())
    }
}

/// Whether a line looks like a section heading rather than body text.
///
/// **Font first**: if the PDF told us the line is bold and it does not end in a
/// full stop, it is a heading. That is a real signal from the document itself.
///
/// **Shape second**: with no font information (plain text), fall back to
/// "short, no terminal full stop, not numbered". This is a heuristic and it is
/// wrong sometimes; when it is wrong the text still reaches the reader as a
/// heading rather than vanishing, which is the failure mode we choose.
fn looks_like_heading(text: &str, line: &SourceLine) -> bool {
    if text.ends_with('.') || text.ends_with(';') || text.ends_with(',') {
        return false;
    }
    if line.emphasis.is_bold() {
        return true;
    }
    text.len() <= 60 && !text.starts_with('(') && !text.chars().next().is_some_and(|c| c.is_ascii_digit())
}

fn section_component(token: &str) -> ProvisionComponent {
    numbering::SectionNumber::parse(token)
        .map(ProvisionComponent::Section)
        .unwrap_or_else(|| ProvisionComponent::Unrecognized(token.to_string()))
}

/// Interprets a bracketed token at a known depth. Delegates to the numbering
/// module so the `(i)` disambiguation rule lives in exactly one place.
fn bracketed_component(token: &str, depth: usize) -> ProvisionComponent {
    numbering::classify_bracketed_public(token, depth)
}
