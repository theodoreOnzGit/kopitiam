//! Hierarchical legal numbering: `Part II`, `s 12(3)(a)(ii)`, `cl 1.2.3`,
//! `[47]`.
//!
//! # Why this is harder than it looks
//!
//! Legal numbering is not a decimal outline with brackets. It is a set of
//! *irregular conventions*, and every one of the irregularities exists for a
//! reason that a naive parser will get wrong:
//!
//! * **Inserted sections.** When a legislature inserts a section between
//!   s 12 and s 13, it cannot renumber the rest of the Act — every existing
//!   cross-reference, in that Act and in every other instrument and judgment
//!   that cites it, would break. So it inserts **s 12A**. Then **s 12AA**
//!   between 12 and 12A. This means section numbers are not integers, and
//!   the correct sort order is `12 < 12A < 12AA < 12B < 13` — see
//!   [`SectionNumber`].
//!
//! * **`(i)` is ambiguous.** At paragraph level, `(i)` is the ninth letter,
//!   following `(h)`. At sub-paragraph level, `(i)` is Roman one, preceding
//!   `(ii)`. The *glyphs are identical*; only the depth in the hierarchy
//!   tells you which is meant. We resolve it by depth (see
//!   [`parse_statutory`]) and record the resolved style in [`NumeralStyle`],
//!   so a renderer can reproduce what the drafter actually wrote.
//!
//! * **Different document kinds number differently.** A statute uses
//!   `12(3)(a)`; a commercial contract uses decimal clauses `1.2.3`; a
//!   judgment uses square-bracketed paragraphs `[47]`. Conflating these is
//!   how you end up citing "clause 12(3)" of a judgment. Hence
//!   [`NumberingScheme`].
//!
//! # Jurisdiction
//!
//! **This module targets Singapore / Commonwealth statutory drafting**
//! (which UK, Singapore, Malaysia, Australia, NZ, HK and India largely
//! share): Parts in Roman numerals, arabic sections with alphabetic
//! insertion suffixes, bracketed subsections/paragraphs/sub-paragraphs.
//! It is *not* a US parser: the US `§ 1983`, `Title 42, Ch. 21,
//! Subchapter I` conventions, and US-style `(a)(1)(A)(i)` ordering (which
//! runs letter-then-number, the opposite way round) are **not** handled and
//! will parse into [`ProvisionComponent::Unrecognized`] rather than being
//! silently misread. Supporting a second jurisdiction means adding a
//! variant to [`NumberingScheme`], not bending this one.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::LegalError;

/// Which numbering convention a document uses. Determined by the kind of
/// instrument, not guessed per-line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumberingScheme {
    /// `Part II`, `s 12(3)(a)(ii)` — statutes and subsidiary legislation.
    Statutory,
    /// `1.2.3` — commercial contracts, leases, deeds.
    DecimalClause,
    /// `[47]` — judgments and other numbered-paragraph reasons.
    JudgmentParagraph,
}

/// How a numeral was *written*, independent of its value.
///
/// Kept alongside the value because reproducing a citation faithfully
/// requires knowing whether the drafter wrote `(2)`, `(b)`, `(ii)` or
/// `(II)` — all of which have value 2. Ordering ignores style (see
/// [`Numeral`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumeralStyle {
    Arabic,
    LowerRoman,
    UpperRoman,
    LowerAlpha,
    UpperAlpha,
}

/// A numeral with both its ordinal value and the style it was written in.
///
/// Field order is load-bearing: `value` precedes `style` so that the derived
/// `Ord` sorts by value first, making `(i)` (Roman 1) sort before `(ii)`
/// (Roman 2) regardless of how either was rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Numeral {
    value: u32,
    style: NumeralStyle,
}

impl Numeral {
    pub fn new(value: u32, style: NumeralStyle) -> Self {
        Self { value, style }
    }

    pub fn value(&self) -> u32 {
        self.value
    }

    pub fn style(&self) -> NumeralStyle {
        self.style
    }

    /// Parses a numeral token, interpreting it according to `style`.
    ///
    /// The caller decides the style, because — as the module docs explain —
    /// the token alone cannot tell you: `i` is Roman 1 or the letter i
    /// depending on where in the hierarchy it sits.
    fn parse(token: &str, style: NumeralStyle) -> Option<Self> {
        let value = match style {
            NumeralStyle::Arabic => token.parse().ok()?,
            NumeralStyle::LowerRoman | NumeralStyle::UpperRoman => parse_roman(token)?,
            NumeralStyle::LowerAlpha | NumeralStyle::UpperAlpha => parse_alpha(token)?,
        };
        Some(Self::new(value, style))
    }
}

impl fmt::Display for Numeral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.style {
            NumeralStyle::Arabic => write!(f, "{}", self.value),
            NumeralStyle::LowerRoman => write!(f, "{}", to_roman(self.value).to_lowercase()),
            NumeralStyle::UpperRoman => write!(f, "{}", to_roman(self.value)),
            NumeralStyle::LowerAlpha => write!(f, "{}", to_alpha(self.value).to_lowercase()),
            NumeralStyle::UpperAlpha => write!(f, "{}", to_alpha(self.value)),
        }
    }
}

/// The ordinal value of `token` read as a Roman numeral, or `None` if it is
/// not one. Exposed for [`crate::ingest`]'s successor rule, which must ask
/// "could this token be Roman?" and "could it be alphabetic?" *separately* in
/// order to disambiguate `(i)`.
pub fn roman_value(token: &str) -> Option<u32> {
    parse_roman(token)
}

/// The ordinal value of `token` read as a bijective base-26 alphabetic
/// ordinal (`a`=1 ... `z`=26, `aa`=27), or `None`. See [`roman_value`].
pub fn alpha_value(token: &str) -> Option<u32> {
    parse_alpha(token)
}

/// Parses a Roman numeral (either case). Deliberately strict-ish: it accepts
/// the standard subtractive forms (`iv`, `ix`, `xl`, ...) and additive
/// sequences, and returns `None` on anything it does not recognise rather
/// than producing a number it is not sure about.
fn parse_roman(token: &str) -> Option<u32> {
    if token.is_empty() {
        return None;
    }
    let digit = |c: char| -> Option<u32> {
        Some(match c.to_ascii_lowercase() {
            'i' => 1,
            'v' => 5,
            'x' => 10,
            'l' => 50,
            'c' => 100,
            'd' => 500,
            'm' => 1000,
            _ => return None,
        })
    };
    let values: Vec<u32> = token.chars().map(digit).collect::<Option<_>>()?;
    // Accumulate as i64, not u32: in subtractive notation the FIRST digit can
    // be the one subtracted ("xl" = -10 + 50), so a u32 running total
    // underflows on the very first step. That bug silently rejected every
    // subtractive form above "ix" — which is to say, every sub-paragraph past
    // (xxxix) in a long list — and it was caught by the round-trip test rather
    // than by reading the code.
    let mut total: i64 = 0;
    for i in 0..values.len() {
        let value = i64::from(values[i]);
        if i + 1 < values.len() && values[i] < values[i + 1] {
            total -= value;
        } else {
            total += value;
        }
    }
    u32::try_from(total).ok().filter(|t| *t > 0)
}

/// Renders 1 => "I", 4 => "IV", 9 => "IX", ...
fn to_roman(mut value: u32) -> String {
    const TABLE: &[(u32, &str)] = &[
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];
    let mut out = String::new();
    for &(v, s) in TABLE {
        while value >= v {
            out.push_str(s);
            value -= v;
        }
    }
    out
}

/// Parses an alphabetic ordinal in the bijective base-26 system legal
/// drafters actually use: `a`..`z`, then `aa`, `ab`, ... — so `(aa)` is the
/// paragraph *inserted after* `(z)`, with value 27.
///
/// Note that legal insertion practice sometimes uses `(aa)` to mean "between
/// (a) and (b)" instead, which collides with this reading. We cannot tell
/// the two apart from the token alone, so we take the sequential reading and
/// preserve the original text verbatim regardless — the reader sees what the
/// drafter wrote.
fn parse_alpha(token: &str) -> Option<u32> {
    if token.is_empty() {
        return None;
    }
    let mut value: u32 = 0;
    for c in token.chars() {
        if !c.is_ascii_alphabetic() {
            return None;
        }
        let d = (c.to_ascii_lowercase() as u32) - ('a' as u32) + 1;
        value = value.checked_mul(26)?.checked_add(d)?;
    }
    Some(value)
}

/// Inverse of [`parse_alpha`]: 1 => "A", 26 => "Z", 27 => "AA".
fn to_alpha(mut value: u32) -> String {
    let mut out = Vec::new();
    while value > 0 {
        let rem = (value - 1) % 26;
        out.push((b'A' + rem as u8) as char);
        value = (value - 1) / 26;
    }
    out.iter().rev().collect()
}

/// A section number, which is **not an integer**: `12`, `12A`, `12AA`.
///
/// See the module docs for why inserted sections carry alphabetic suffixes.
/// Field order is load-bearing for the derived `Ord`: `number` first, then
/// `suffix`, and because `None < Some(_)` in Rust's derived ordering, the
/// bare section sorts before its insertions — giving exactly the legal
/// ordering `12 < 12A < 12AA < 12B < 13`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SectionNumber {
    number: u32,
    /// Uppercase insertion suffix, e.g. `A` in `12A`. `None` for a plain
    /// section.
    suffix: Option<String>,
}

impl SectionNumber {
    pub fn new(number: u32, suffix: Option<&str>) -> Self {
        Self {
            number,
            suffix: suffix.map(|s| s.to_ascii_uppercase()),
        }
    }

    pub fn number(&self) -> u32 {
        self.number
    }

    pub fn suffix(&self) -> Option<&str> {
        self.suffix.as_deref()
    }

    /// Parses `12`, `12A`, `12AA`.
    pub fn parse(token: &str) -> Option<Self> {
        let split = token
            .find(|c: char| c.is_ascii_alphabetic())
            .unwrap_or(token.len());
        let (digits, suffix) = token.split_at(split);
        let number: u32 = digits.parse().ok()?;
        if !suffix.is_empty() && !suffix.chars().all(|c| c.is_ascii_alphabetic()) {
            return None;
        }
        Some(Self::new(
            number,
            (!suffix.is_empty()).then_some(suffix),
        ))
    }
}

impl fmt::Display for SectionNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.number)?;
        if let Some(suffix) = &self.suffix {
            write!(f, "{suffix}")?;
        }
        Ok(())
    }
}

/// One level of a legal hierarchy.
///
/// **Variant declaration order is load-bearing.** The derived `Ord` compares
/// variants by declaration order first, so these are declared outermost-to-
/// innermost, matching the containment hierarchy. `Unrecognized` is last so
/// that anything we could not parse sorts after everything we could, rather
/// than being interleaved unpredictably among real provisions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvisionComponent {
    /// `Part II`
    Part(Numeral),
    /// `Division 1`
    Division(Numeral),
    /// `s 12`, `s 12A`
    Section(SectionNumber),
    /// `(3)`, `(3A)` — subsections take insertion suffixes too.
    Subsection(SectionNumber),
    /// `(a)`, `(aa)`
    Paragraph(Numeral),
    /// `(ii)`
    Subparagraph(Numeral),
    /// `(A)` — the fourth statutory level, rare but real.
    SubSubparagraph(Numeral),
    /// A decimal contract clause level: the `2` in `1.2.3`.
    Clause(SectionNumber),
    /// `[47]` in a judgment.
    JudgmentParagraph(u32),
    /// The `First Schedule` / `Schedule 2`.
    Schedule(Numeral),
    /// **A label we could not parse.** Its text is preserved verbatim and it
    /// is never dropped — see [`crate::AnomalyKind::UnparseableNumbering`].
    /// This variant is the reason this crate can honestly claim it never
    /// silently discards content: an unrecognised heading still gets an
    /// identity and still carries its text to the reader.
    Unrecognized(String),
}

impl fmt::Display for ProvisionComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Part(n) => write!(f, "Part {n}"),
            Self::Division(n) => write!(f, "Division {n}"),
            Self::Section(n) => write!(f, "s {n}"),
            Self::Subsection(n) => write!(f, "({n})"),
            Self::Paragraph(n) => write!(f, "({n})"),
            Self::Subparagraph(n) => write!(f, "({n})"),
            Self::SubSubparagraph(n) => write!(f, "({n})"),
            Self::Clause(n) => write!(f, "{n}"),
            Self::JudgmentParagraph(n) => write!(f, "[{n}]"),
            Self::Schedule(n) => write!(f, "Schedule {n}"),
            Self::Unrecognized(raw) => write!(f, "{raw}"),
        }
    }
}

/// The identity of a provision: an ordered path from the outermost
/// containing unit down to the provision itself.
///
/// `Part II > s 12 > (3) > (a) > (ii)` is the path for `s 12(3)(a)(ii)`.
/// Representing it as a *path* rather than a flat string is what makes
/// containment ([`ProvisionId::contains`]) and ordering decidable, which in
/// turn is what lets [`crate::Dictionary`] answer "does this definition's
/// scope cover this provision?" without string-matching.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProvisionId {
    components: Vec<ProvisionComponent>,
}

impl ProvisionId {
    /// Builds an id from an explicit component path.
    pub fn new(components: Vec<ProvisionComponent>) -> Self {
        Self { components }
    }

    pub fn components(&self) -> &[ProvisionComponent] {
        &self.components
    }

    /// Whether `self` is an ancestor of (or equal to) `other` — i.e. whether
    /// `other` sits inside `self` in the document hierarchy.
    ///
    /// `s 12` contains `s 12(3)(a)`; `s 12(3)` does not contain `s 12(4)`.
    /// This is a pure prefix test on the component path, which is exactly
    /// what legal containment means.
    pub fn contains(&self, other: &ProvisionId) -> bool {
        other.components.len() >= self.components.len()
            && self
                .components
                .iter()
                .zip(&other.components)
                .all(|(a, b)| a == b)
    }

    /// The id of the unit immediately containing this one, or `None` for a
    /// top-level unit.
    pub fn parent(&self) -> Option<ProvisionId> {
        (self.components.len() > 1).then(|| {
            ProvisionId::new(self.components[..self.components.len() - 1].to_vec())
        })
    }

    /// Appends a component, returning the child id. Non-mutating so that ids
    /// can be built up while walking a document without cloning ceremony.
    pub fn child(&self, component: ProvisionComponent) -> ProvisionId {
        let mut components = self.components.clone();
        components.push(component);
        ProvisionId::new(components)
    }

    /// Whether any component of this path failed to parse.
    pub fn has_unrecognized(&self) -> bool {
        self.components
            .iter()
            .any(|c| matches!(c, ProvisionComponent::Unrecognized(_)))
    }
}

impl fmt::Display for ProvisionId {
    /// Renders the path in the citation form appropriate to its components:
    /// `Part II, s 12(3)(a)(ii)` for statutory paths, `cl 1.2.3` for
    /// contract clauses, `[47]` for judgment paragraphs.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        let mut prev_was_bracketed = false;
        for component in &self.components {
            let bracketed = matches!(
                component,
                ProvisionComponent::Subsection(_)
                    | ProvisionComponent::Paragraph(_)
                    | ProvisionComponent::Subparagraph(_)
                    | ProvisionComponent::SubSubparagraph(_)
            );
            let clause = matches!(component, ProvisionComponent::Clause(_));
            if first {
                if clause {
                    write!(f, "cl ")?;
                }
            } else if clause {
                // Decimal clauses join with dots: 1.2.3
                write!(f, ".")?;
            } else if !bracketed {
                write!(f, ", ")?;
            } else if !prev_was_bracketed {
                // Section then subsection joins tight: s 12(3)
            }
            write!(f, "{component}")?;
            first = false;
            prev_was_bracketed = bracketed;
        }
        Ok(())
    }
}

/// Parses a statutory provision label such as `12(3)(a)(ii)` or `12A(1)`
/// into its component path.
///
/// **Depth resolves the `(i)` ambiguity.** The bracketed levels below a
/// section are, by Commonwealth convention and in order:
///
/// | Depth | Level | Style |
/// |---|---|---|
/// | 1 | subsection | arabic, `(3)`, `(3A)` |
/// | 2 | paragraph | lower alpha, `(a)` |
/// | 3 | sub-paragraph | lower roman, `(ii)` |
/// | 4 | sub-sub-paragraph | upper alpha, `(A)` |
///
/// so a bare `(i)` at depth 2 is the letter i, and at depth 3 is Roman one.
/// This is the *only* information available to disambiguate them, and it is
/// the same rule a human reader applies.
///
/// A token that does not fit its expected level is **not** coerced: it
/// becomes a [`ProvisionComponent::Unrecognized`] carrying the original
/// text, so the caller can raise an [`crate::AnomalyKind::UnparseableNumbering`]
/// and show the reader what was actually written.
pub fn parse_statutory(label: &str) -> Result<ProvisionId, LegalError> {
    let label = label.trim();
    // Strip a leading "s"/"section"/"§" if the caller left one on.
    let body = label
        .strip_prefix('§')
        .or_else(|| label.strip_prefix("section "))
        .or_else(|| label.strip_prefix("s "))
        .or_else(|| label.strip_prefix("s."))
        .unwrap_or(label)
        .trim();

    let (section_token, rest) = match body.find('(') {
        Some(idx) => (&body[..idx], &body[idx..]),
        None => (body, ""),
    };
    let section_token = section_token.trim();
    if section_token.is_empty() {
        return Err(LegalError::UnparseableNumbering {
            label: label.to_string(),
        });
    }

    let section = SectionNumber::parse(section_token)
        .map(ProvisionComponent::Section)
        .unwrap_or_else(|| ProvisionComponent::Unrecognized(section_token.to_string()));

    let mut components = vec![section];
    for (depth, token) in bracketed_tokens(rest)?.into_iter().enumerate() {
        components.push(classify_bracketed(&token, depth));
    }
    Ok(ProvisionId::new(components))
}

/// Splits `(3)(a)(ii)` into `["3", "a", "ii"]`, rejecting unbalanced
/// brackets rather than guessing where a level ends.
fn bracketed_tokens(mut rest: &str) -> Result<Vec<String>, LegalError> {
    let mut tokens = Vec::new();
    rest = rest.trim();
    while !rest.is_empty() {
        let Some(stripped) = rest.strip_prefix('(') else {
            return Err(LegalError::UnparseableNumbering {
                label: rest.to_string(),
            });
        };
        let Some(close) = stripped.find(')') else {
            return Err(LegalError::UnparseableNumbering {
                label: rest.to_string(),
            });
        };
        tokens.push(stripped[..close].to_string());
        rest = stripped[close + 1..].trim();
    }
    Ok(tokens)
}

/// Interprets one bracketed token at a given depth below the section, per the
/// table in [`parse_statutory`].
///
/// Exposed so that [`crate::ingest`], which discovers depth by walking the
/// document rather than by parsing a whole label, resolves the `(i)` ambiguity
/// through **exactly the same rule**. Two implementations of that rule would
/// eventually disagree, and the disagreement would be silent.
pub fn classify_bracketed_public(token: &str, depth: usize) -> ProvisionComponent {
    classify_bracketed(token, depth)
}

/// Interprets one bracketed token at a given depth below the section, per
/// the table in [`parse_statutory`].
fn classify_bracketed(token: &str, depth: usize) -> ProvisionComponent {
    let unrecognized = || ProvisionComponent::Unrecognized(format!("({token})"));
    match depth {
        // Subsection: arabic, possibly with an insertion suffix — (3), (3A).
        0 => SectionNumber::parse(token)
            .map(ProvisionComponent::Subsection)
            .unwrap_or_else(unrecognized),
        // Paragraph: lower alpha — (a), (aa).
        1 => Numeral::parse(token, NumeralStyle::LowerAlpha)
            .map(ProvisionComponent::Paragraph)
            .unwrap_or_else(unrecognized),
        // Sub-paragraph: lower roman — (i), (ii).
        2 => Numeral::parse(token, NumeralStyle::LowerRoman)
            .map(ProvisionComponent::Subparagraph)
            .unwrap_or_else(unrecognized),
        // Sub-sub-paragraph: upper alpha — (A), (B).
        3 => Numeral::parse(token, NumeralStyle::UpperAlpha)
            .map(ProvisionComponent::SubSubparagraph)
            .unwrap_or_else(unrecognized),
        // Deeper than the convention goes. We do not invent a fifth level.
        _ => unrecognized(),
    }
}

/// Parses a decimal contract clause label such as `1.2.3`.
pub fn parse_decimal_clause(label: &str) -> Result<ProvisionId, LegalError> {
    let label = label.trim().trim_end_matches('.');
    if label.is_empty() {
        return Err(LegalError::UnparseableNumbering {
            label: label.to_string(),
        });
    }
    let components = label
        .split('.')
        .map(|token| {
            SectionNumber::parse(token)
                .map(ProvisionComponent::Clause)
                .unwrap_or_else(|| ProvisionComponent::Unrecognized(token.to_string()))
        })
        .collect();
    Ok(ProvisionId::new(components))
}

/// Parses a judgment paragraph label such as `[47]` or `47`.
pub fn parse_judgment_paragraph(label: &str) -> Result<ProvisionId, LegalError> {
    let token = label.trim().trim_start_matches('[').trim_end_matches(']');
    token
        .parse::<u32>()
        .map(|n| ProvisionId::new(vec![ProvisionComponent::JudgmentParagraph(n)]))
        .map_err(|_| LegalError::UnparseableNumbering {
            label: label.to_string(),
        })
}

/// Parses a Part label such as `II`, `Part II`, or `2`.
///
/// Parts are conventionally Roman in Commonwealth drafting, but arabic Parts
/// exist (modern Australian and Singaporean drafting increasingly uses
/// `Part 2`), so both are accepted and the style used is preserved on the
/// [`Numeral`] so the citation can be rendered back as written.
pub fn parse_part(label: &str) -> Result<ProvisionId, LegalError> {
    let token = label
        .trim()
        .strip_prefix("Part ")
        .or_else(|| label.trim().strip_prefix("part "))
        .unwrap_or(label.trim())
        .trim();
    if token.is_empty() {
        return Err(LegalError::UnparseableNumbering {
            label: label.to_string(),
        });
    }
    let numeral = if token.chars().all(|c| c.is_ascii_digit()) {
        Numeral::parse(token, NumeralStyle::Arabic)
    } else if token.chars().all(|c| c.is_ascii_uppercase()) {
        Numeral::parse(token, NumeralStyle::UpperRoman)
    } else {
        None
    };
    numeral
        .map(|n| ProvisionId::new(vec![ProvisionComponent::Part(n)]))
        .ok_or_else(|| LegalError::UnparseableNumbering {
            label: label.to_string(),
        })
}

/// Parses `label` according to `scheme`.
pub fn parse(label: &str, scheme: NumberingScheme) -> Result<ProvisionId, LegalError> {
    match scheme {
        NumberingScheme::Statutory => parse_statutory(label),
        NumberingScheme::DecimalClause => parse_decimal_clause(label),
        NumberingScheme::JudgmentParagraph => parse_judgment_paragraph(label),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(label: &str) -> ProvisionId {
        parse_statutory(label).unwrap()
    }

    #[test]
    fn parses_the_canonical_nested_citation() {
        let id = ids("12(3)(a)(ii)");
        assert_eq!(
            id.components(),
            &[
                ProvisionComponent::Section(SectionNumber::new(12, None)),
                ProvisionComponent::Subsection(SectionNumber::new(3, None)),
                ProvisionComponent::Paragraph(Numeral::new(1, NumeralStyle::LowerAlpha)),
                ProvisionComponent::Subparagraph(Numeral::new(2, NumeralStyle::LowerRoman)),
            ]
        );
        assert_eq!(id.to_string(), "s 12(3)(a)(ii)");
    }

    #[test]
    fn depth_disambiguates_the_paragraph_i_from_roman_one() {
        // (i) at paragraph depth is the ninth LETTER.
        let para = ids("5(1)(i)");
        assert_eq!(
            para.components()[2],
            ProvisionComponent::Paragraph(Numeral::new(9, NumeralStyle::LowerAlpha)),
            "(i) directly under a subsection is the letter i, following (h)"
        );

        // (i) at sub-paragraph depth is ROMAN ONE.
        let subpara = ids("5(1)(a)(i)");
        assert_eq!(
            subpara.components()[3],
            ProvisionComponent::Subparagraph(Numeral::new(1, NumeralStyle::LowerRoman)),
            "(i) under a paragraph is roman one, preceding (ii)"
        );
    }

    #[test]
    fn parses_inserted_sections_and_orders_them_legally() {
        // The whole point of 12A: it is inserted between 12 and 13 without
        // renumbering 13.
        let mut got = [
            ids("13"),
            ids("12A"),
            ids("12"),
            ids("12B"),
            ids("12AA"),
            ids("2"),
        ];
        got.sort();
        let rendered: Vec<String> = got.iter().map(|id| id.to_string()).collect();
        assert_eq!(
            rendered,
            vec!["s 2", "s 12", "s 12A", "s 12AA", "s 12B", "s 13"],
            "inserted sections must sort between their neighbours, not after them"
        );
    }

    #[test]
    fn parses_inserted_subsections() {
        let id = ids("12(3A)");
        assert_eq!(
            id.components()[1],
            ProvisionComponent::Subsection(SectionNumber::new(3, Some("A")))
        );
        assert_eq!(id.to_string(), "s 12(3A)");
    }

    #[test]
    fn orders_subsections_and_paragraphs_within_a_section() {
        let mut got = [
            ids("12(3)(b)"),
            ids("12(2)"),
            ids("12(3)(a)(ii)"),
            ids("12(3)(a)(i)"),
            ids("12(3)(a)"),
        ];
        got.sort();
        let rendered: Vec<String> = got.iter().map(|id| id.to_string()).collect();
        assert_eq!(
            rendered,
            vec![
                "s 12(2)",
                "s 12(3)(a)",
                "s 12(3)(a)(i)",
                "s 12(3)(a)(ii)",
                "s 12(3)(b)",
            ]
        );
    }

    #[test]
    fn containment_is_prefix_containment() {
        assert!(ids("12").contains(&ids("12(3)(a)")));
        assert!(ids("12(3)").contains(&ids("12(3)(a)(ii)")));
        assert!(ids("12").contains(&ids("12")), "reflexive");

        assert!(!ids("12(3)").contains(&ids("12(4)")));
        assert!(!ids("12(3)(a)").contains(&ids("12(3)")), "not upward");
        assert!(!ids("12").contains(&ids("12A")), "12A is not inside 12");
        assert!(!ids("13").contains(&ids("12")));
    }

    #[test]
    fn parent_walks_up_the_hierarchy() {
        let id = ids("12(3)(a)(ii)");
        let parent = id.parent().unwrap();
        assert_eq!(parent.to_string(), "s 12(3)(a)");
        assert_eq!(parent.parent().unwrap().to_string(), "s 12(3)");
        assert_eq!(ids("12").parent(), None);
    }

    #[test]
    fn part_and_division_prefix_a_section() {
        let id = ProvisionId::new(vec![ProvisionComponent::Part(Numeral::new(
            2,
            NumeralStyle::UpperRoman,
        ))])
        .child(ProvisionComponent::Section(SectionNumber::new(12, None)))
        .child(ProvisionComponent::Subsection(SectionNumber::new(3, None)));
        assert_eq!(id.to_string(), "Part II, s 12(3)");
    }

    #[test]
    fn unparseable_label_is_preserved_not_dropped_or_guessed() {
        // A garbage bracketed token must not be coerced into a plausible
        // number. It keeps its literal text and marks the path unrecognized.
        let id = ids("12(%%)");
        assert!(id.has_unrecognized());
        assert_eq!(
            id.components()[1],
            ProvisionComponent::Unrecognized("(%%)".to_string())
        );
        assert!(
            id.to_string().contains("%%"),
            "the reader must still see what the document actually said"
        );
    }

    #[test]
    fn unbalanced_brackets_are_an_error_not_a_guess() {
        assert!(parse_statutory("12(3").is_err());
        assert!(parse_statutory("12)3(").is_err());
        assert!(parse_statutory("").is_err());
    }

    #[test]
    fn does_not_invent_a_fifth_statutory_level() {
        let id = ids("12(1)(a)(i)(A)(z)");
        assert!(
            matches!(
                id.components()[5],
                ProvisionComponent::Unrecognized(_)
            ),
            "the Commonwealth convention has four bracketed levels; we do not fabricate a fifth"
        );
    }

    #[test]
    fn roman_numerals_round_trip_including_subtractive_forms() {
        for (value, text) in [(1, "i"), (4, "iv"), (9, "ix"), (14, "xiv"), (40, "xl")] {
            let n = Numeral::parse(text, NumeralStyle::LowerRoman).unwrap();
            assert_eq!(n.value(), value, "parsing {text}");
            assert_eq!(n.to_string(), text, "rendering {value}");
        }
    }

    #[test]
    fn alpha_numerals_use_bijective_base_26() {
        assert_eq!(
            Numeral::parse("z", NumeralStyle::LowerAlpha).unwrap().value(),
            26
        );
        let aa = Numeral::parse("aa", NumeralStyle::LowerAlpha).unwrap();
        assert_eq!(aa.value(), 27, "(aa) follows (z)");
        assert_eq!(aa.to_string(), "aa");
    }

    #[test]
    fn parses_decimal_contract_clauses() {
        let id = parse_decimal_clause("1.2.3").unwrap();
        assert_eq!(id.to_string(), "cl 1.2.3");
        assert!(parse_decimal_clause("1.2").unwrap().contains(&id));
        assert!(!parse_decimal_clause("1.3").unwrap().contains(&id));
    }

    #[test]
    fn parses_judgment_paragraphs() {
        let id = parse_judgment_paragraph("[47]").unwrap();
        assert_eq!(id.to_string(), "[47]");
        assert_eq!(parse_judgment_paragraph("47").unwrap(), id);
        assert!(parse_judgment_paragraph("[forty-seven]").is_err());
    }

    #[test]
    fn schemes_do_not_bleed_into_each_other() {
        // "1.2" is a clause under DecimalClause, but is NOT a valid
        // statutory section — conflating them is how you cite "clause 12(3)"
        // of a judgment.
        let clause = parse("1.2", NumberingScheme::DecimalClause).unwrap();
        assert_eq!(clause.to_string(), "cl 1.2");
        let statutory = parse("1.2", NumberingScheme::Statutory).unwrap();
        assert!(
            statutory.has_unrecognized(),
            "a decimal clause label is not a statutory section number"
        );
    }
}
