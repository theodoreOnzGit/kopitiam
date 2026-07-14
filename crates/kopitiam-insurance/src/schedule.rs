//! The Schedule — the policy-specific numbers.
//!
//! The wording is the same for every customer. The **Schedule** is the part
//! that is about *you*: the sum insured, the limits, the excess or deductible,
//! the premium, the period of insurance. It is almost always a table, which is
//! why [`kopitiam_document`]'s table reconstruction is the dependency that
//! matters most here.
//!
//! # No floating point, anywhere
//!
//! Money in this module is **exact integer cents** ([`Money`]) and percentages
//! are **exact basis points** ([`Percentage`]). Neither is an `f64`.
//!
//! This is not fastidiousness. `0.1 + 0.2 != 0.3` in binary floating point,
//! and a co-insurance calculation that lands a cent away from the contract is
//! not a rounding artefact in this domain — it is a wrong statement about a
//! legal obligation. Money is a decimal quantity in a base-10 contract, and it
//! is represented as one. `f64` is available via
//! [`Money::to_f64_lossy`] for callers doing statistics, and its name says
//! what it costs.
//!
//! # Nothing is normalised away
//!
//! * A bare `$` is **not** silently read as USD, or as SGD. At least a dozen
//!   countries print their currency as `$`, and picking one would be inventing
//!   a fact. It becomes [`Currency::Ambiguous`], and the reader is told.
//! * `Nil` is **not** normalised to zero. A nil *excess* and a nil *benefit*
//!   are not the same thing, and deciding which one a document means is a
//!   domain judgment — the job of `kopitiam-health` or `kopitiam-finance`, not
//!   of this crate. It stays [`ScheduleValue::Nil`], verbatim.
//! * A value this module cannot type does not become a default or a zero. It
//!   becomes [`ScheduleValue::Unparseable`], carrying the raw text and the
//!   reason — which is a *correct* answer ("I could not determine this; here
//!   are the words; read them").

use serde::{Deserialize, Serialize};

use crate::clause::Clause;
use crate::provenance::{ExtractedTerm, ProvenanceError};

/// Currency symbols that are unambiguous, mapped to their ISO 4217 code.
/// Longest first, so `"S$"` is matched before `"$"` would be.
const UNAMBIGUOUS_SYMBOLS: &[(&str, &str)] = &[
    ("US$", "USD"),
    ("AU$", "AUD"),
    ("HK$", "HKD"),
    ("NZ$", "NZD"),
    ("S$", "SGD"),
    ("A$", "AUD"),
    ("RM", "MYR"),
    ("£", "GBP"),
    ("€", "EUR"),
    ("₹", "INR"),
];

/// Symbols that are genuinely ambiguous. `$` is used by Singapore, the US,
/// Australia, Canada, Hong Kong, New Zealand and more; `¥` by both Japan and
/// China. Reading one as a particular currency would be inventing a fact about
/// somebody's insurance policy.
const AMBIGUOUS_SYMBOLS: &[&str] = &["$", "¥"];

/// ISO 4217 codes an insurance schedule in this region plausibly prints. Used
/// only to recognise a code that is *already written out*, never to guess one.
const ISO_CODES: &[&str] = &[
    "SGD", "USD", "MYR", "EUR", "GBP", "AUD", "HKD", "JPY", "CNY", "NZD", "IDR", "THB", "PHP",
    "INR", "CHF", "CAD",
];

/// Words a schedule prints for "no amount".
const NIL_WORDS: &[&str] = &["nil", "none", "n.a.", "n/a", "na", "not applicable", "-", "—"];

/// Words a schedule prints for "no ceiling".
const UNLIMITED_WORDS: &[&str] = &["unlimited", "no limit", "not limited", "as charged"];

/// The currency of a monetary amount, **as the document printed it**.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Currency {
    /// An unambiguous currency: either an ISO 4217 code printed in the
    /// document (`SGD 150,000`) or a symbol that maps to exactly one
    /// (`S$150,000`, `£500`).
    Iso(String),

    /// A symbol that does **not** identify a currency, kept as printed. `$`
    /// is the usual case. A consumer must either ask the user or decline to
    /// answer; it must not assume.
    Ambiguous(String),

    /// The document printed a number with no currency marker at all (common
    /// inside a table whose header already established the currency). The
    /// number is real; the currency is not stated *here*.
    Unstated,
}

impl Currency {
    /// The ISO code, when the document actually identified the currency.
    /// `None` for [`Currency::Ambiguous`] and [`Currency::Unstated`] — which
    /// is the point: a caller cannot get a currency out of this type unless
    /// the document supplied one.
    pub fn iso(&self) -> Option<&str> {
        match self {
            Self::Iso(code) => Some(code),
            Self::Ambiguous(_) | Self::Unstated => None,
        }
    }
}

/// An exact monetary amount: integer cents, never a float. See the module docs.
///
/// Deliberately currency-*less*. The amount and the currency are separated
/// because they have different epistemic status: the amount is always known
/// once the digits parse, whereas the currency very often is not (see
/// [`Currency::Ambiguous`]). Fusing them would make an exactly-known amount
/// hostage to an unknown currency, and would tempt the obvious "just default
/// it to SGD" fix. [`MonetaryAmount`] pairs the two, honestly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Money {
    cents: i64,
}

impl Money {
    /// An amount in exact cents (hundredths of the currency's major unit).
    pub fn from_cents(cents: i64) -> Self {
        Self { cents }
    }

    /// The amount in exact cents.
    pub fn cents(self) -> i64 {
        self.cents
    }

    /// The amount as an exact decimal string, e.g. `"150000.00"`.
    pub fn to_decimal_string(self) -> String {
        let sign = if self.cents < 0 { "-" } else { "" };
        let abs = self.cents.unsigned_abs();
        format!("{sign}{}.{:02}", abs / 100, abs % 100)
    }

    /// The amount as an `f64`. **Lossy** — the name says so. For statistics
    /// and plotting, never for restating a contractual amount to a human.
    pub fn to_f64_lossy(self) -> f64 {
        self.cents as f64 / 100.0
    }
}

/// A monetary amount together with the currency the document printed for it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MonetaryAmount {
    amount: Money,
    currency: Currency,
}

impl MonetaryAmount {
    /// Pairs an exact amount with the currency as printed.
    pub fn new(amount: Money, currency: Currency) -> Self {
        Self { amount, currency }
    }

    /// The exact amount.
    pub fn amount(&self) -> Money {
        self.amount
    }

    /// The currency, as printed — possibly [`Currency::Ambiguous`].
    pub fn currency(&self) -> &Currency {
        &self.currency
    }
}

/// An exact percentage in basis points (1% = 100 bp), never a float.
///
/// Basis points represent every percentage an insurance document actually
/// prints (whole percents, halves, quarters, two decimal places) exactly. A
/// percentage needing finer precision than 0.01% is refused rather than
/// silently rounded — see [`parse_value`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Percentage {
    basis_points: i64,
}

impl Percentage {
    /// A percentage in basis points: 10% is `1000`.
    pub fn from_basis_points(basis_points: i64) -> Self {
        Self { basis_points }
    }

    /// The percentage in basis points.
    pub fn basis_points(self) -> i64 {
        self.basis_points
    }

    /// An exact decimal string of the percentage, e.g. `"12.5"` for 12.5%.
    pub fn to_decimal_string(self) -> String {
        let sign = if self.basis_points < 0 { "-" } else { "" };
        let abs = self.basis_points.unsigned_abs();
        let whole = abs / 100;
        let frac = abs % 100;
        if frac == 0 {
            format!("{sign}{whole}")
        } else if frac.is_multiple_of(10) {
            format!("{sign}{whole}.{}", frac / 10)
        } else {
            format!("{sign}{whole}.{frac:02}")
        }
    }

    /// Applies the percentage to an exact amount, rounding half away from
    /// zero — the convention a policy's own arithmetic uses.
    ///
    /// Returns `None` on overflow rather than wrapping. An insurance figure
    /// that overflows an `i64` of cents is not a figure; it is a parse error
    /// that got this far, and reporting it as a silently wrapped number would
    /// be indefensible.
    pub fn of(self, amount: Money) -> Option<Money> {
        let numerator = i128::from(amount.cents()) * i128::from(self.basis_points);
        let denominator = 10_000_i128;
        let rounded = if numerator >= 0 {
            (numerator + denominator / 2) / denominator
        } else {
            (numerator - denominator / 2) / denominator
        };
        i64::try_from(rounded).ok().map(Money::from_cents)
    }
}

/// A value as it appears in a schedule or benefit table.
///
/// The variants that look like failures — [`ScheduleValue::Text`],
/// [`ScheduleValue::Unparseable`] — are not failures. They are this crate
/// declining to invent a number, which is the behaviour a legal document
/// demands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleValue {
    /// A monetary amount.
    Money(MonetaryAmount),

    /// A percentage (a co-insurance rate, a no-claim discount).
    Percentage(Percentage),

    /// A bare count: days, visits, claims, years.
    Count(i64),

    /// `Nil`, `None`, `N/A` — as printed. **Not** normalised to zero: whether
    /// a nil excess and a nil benefit mean the same thing is a domain
    /// judgment, and it is not this crate's to make.
    Nil(String),

    /// `Unlimited`, `No limit`, `As charged` — as printed.
    Unlimited(String),

    /// Prose, with no numeric content to type. Kept verbatim.
    Text(String),

    /// It looked like a number and we could **not** type it. Surfaced, with
    /// the raw text and the reason, so a reader can go and read it. Never
    /// dropped, never defaulted.
    Unparseable {
        /// The value exactly as printed.
        raw: String,
        /// Why we declined to type it.
        reason: String,
    },
}

impl ScheduleValue {
    /// The monetary amount, if this value is one.
    pub fn as_money(&self) -> Option<&MonetaryAmount> {
        match self {
            Self::Money(money) => Some(money),
            _ => None,
        }
    }

    /// The percentage, if this value is one.
    pub fn as_percentage(&self) -> Option<Percentage> {
        match self {
            Self::Percentage(percentage) => Some(*percentage),
            _ => None,
        }
    }

    /// Whether this crate declined to type the value. A consumer that ignores
    /// this is a consumer that will one day report a limit of zero where the
    /// document said something it did not understand.
    pub fn is_unparseable(&self) -> bool {
        matches!(self, Self::Unparseable { .. })
    }
}

/// One labelled row of a schedule: `Sum Insured | S$150,000`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduleEntry {
    label: String,
    value: ExtractedTerm<ScheduleValue>,
}

impl ScheduleEntry {
    /// The row's label, as printed (`"Sum Insured"`, `"Excess"`).
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The typed value, inseparable from its citation.
    pub fn value(&self) -> &ExtractedTerm<ScheduleValue> {
        &self.value
    }
}

/// The policy-specific numbers, extracted from the schedule's tables.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schedule {
    entries: Vec<ScheduleEntry>,
}

impl Schedule {
    /// Every schedule entry, in document order.
    pub fn entries(&self) -> &[ScheduleEntry] {
        &self.entries
    }

    /// Looks up an entry by label, case- and whitespace-insensitively.
    ///
    /// Returns the **first** match. If a schedule labels two different rows
    /// identically, that is a defect in the document, and it is reported as
    /// [`crate::Anomaly::DuplicateScheduleLabel`] rather than resolved here.
    pub fn get(&self, label: &str) -> Option<&ScheduleEntry> {
        let wanted = normalise_label(label);
        self.entries
            .iter()
            .find(|entry| normalise_label(&entry.label) == wanted)
    }

    /// Whether the schedule has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn push(&mut self, entry: ScheduleEntry) {
        self.entries.push(entry);
    }

    /// Builds schedule entries from a two-column `label | value` table.
    ///
    /// The citation attached to each value is the **whole row**, not the cell.
    /// A cell cited alone (`"500"`) is meaningless; the row (`"Excess | S$500
    /// each and every claim"`) is what a reader needs to see to check us.
    ///
    /// # Errors
    ///
    /// [`ProvenanceError`] if a row cannot be cited back to its own clause.
    pub(crate) fn from_two_column_rows(
        clause: &Clause,
        rows: &[(String, String)],
    ) -> Result<Vec<ScheduleEntry>, ProvenanceError> {
        rows.iter()
            .map(|(label, raw)| {
                let row_text = render_row([label.as_str(), raw.as_str()]);
                let value = clause.extract(parse_value(raw), &row_text)?;
                Ok(ScheduleEntry {
                    label: label.trim().to_string(),
                    value,
                })
            })
            .collect()
    }
}

/// A benefit table: one row per benefit, one column per plan.
///
/// This is the shape an Integrated Shield benefit summary, a travel policy's
/// Silver/Gold/Platinum table, and a motor policy's comprehensive/third-party
/// comparison all take, and it is *not* a flat schedule — a benefit has a
/// different value under each plan, and flattening it loses which is which.
/// Exposed as a distinct type for exactly that reason, and it is the type a
/// domain crate (`kopitiam-health`) should reach for when it needs "what does
/// Plan B pay for a Class A ward".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenefitTable {
    plans: Vec<String>,
    rows: Vec<BenefitRow>,
}

impl BenefitTable {
    /// The plan/tier names, in column order (the table's header row, minus its
    /// first cell, which labels the benefit column).
    pub fn plans(&self) -> &[String] {
        &self.plans
    }

    /// The benefit rows.
    pub fn rows(&self) -> &[BenefitRow] {
        &self.rows
    }

    /// The value of `benefit` under `plan`, with its citation.
    ///
    /// Both are matched case- and whitespace-insensitively. `None` if either
    /// the benefit or the plan is not in the table — which a caller must
    /// handle as "the document does not say", never as zero.
    pub fn value_for(&self, benefit: &str, plan: &str) -> Option<&ExtractedTerm<ScheduleValue>> {
        let wanted_plan = normalise_label(plan);
        let column = self
            .plans
            .iter()
            .position(|p| normalise_label(p) == wanted_plan)?;
        let wanted_benefit = normalise_label(benefit);
        self.rows
            .iter()
            .find(|row| normalise_label(&row.benefit) == wanted_benefit)
            .and_then(|row| row.values.get(column))
    }

    /// Builds a benefit table from a reconstructed table's headers and rows.
    ///
    /// # Errors
    ///
    /// [`ProvenanceError`] if a row cannot be cited back to its own clause.
    pub(crate) fn from_table(
        clause: &Clause,
        headers: &[String],
        rows: &[Vec<String>],
    ) -> Result<Self, ProvenanceError> {
        let plans = headers.iter().skip(1).map(|h| h.trim().to_string()).collect();

        let rows = rows
            .iter()
            .map(|cells| {
                let row_text = render_row(cells.iter().map(String::as_str));
                let benefit = cells.first().cloned().unwrap_or_default().trim().to_string();
                let values = cells
                    .iter()
                    .skip(1)
                    .map(|raw| clause.extract(parse_value(raw), &row_text))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(BenefitRow { benefit, values })
            })
            .collect::<Result<Vec<_>, ProvenanceError>>()?;

        Ok(Self { plans, rows })
    }
}

/// One benefit, and what it is worth under each plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenefitRow {
    benefit: String,
    values: Vec<ExtractedTerm<ScheduleValue>>,
}

impl BenefitRow {
    /// The benefit's name, as printed.
    pub fn benefit(&self) -> &str {
        &self.benefit
    }

    /// The values, in the same column order as [`BenefitTable::plans`].
    pub fn values(&self) -> &[ExtractedTerm<ScheduleValue>] {
        &self.values
    }
}

/// Renders a table row the way [`crate::ingest`] renders it into clause text,
/// so a citation to the row is a citation to text that really occurs there.
pub(crate) fn render_row<'a>(cells: impl IntoIterator<Item = &'a str>) -> String {
    cells
        .into_iter()
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" | ")
}

fn normalise_label(label: &str) -> String {
    label.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// Types a schedule value from its printed text.
///
/// Total: it never fails, because "failure" here is a *result*, not an error.
/// A value it cannot type comes back as [`ScheduleValue::Unparseable`] with
/// the raw text and the reason — which is exactly what a reader needs.
pub fn parse_value(raw: &str) -> ScheduleValue {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return ScheduleValue::Text(String::new());
    }

    let lower = trimmed.to_lowercase();

    if NIL_WORDS.contains(&lower.as_str()) {
        return ScheduleValue::Nil(trimmed.to_string());
    }
    if UNLIMITED_WORDS.iter().any(|word| lower == *word) {
        return ScheduleValue::Unlimited(trimmed.to_string());
    }

    // No digits at all: prose. Nothing to type, nothing to get wrong.
    if !trimmed.chars().any(|c| c.is_ascii_digit()) {
        return ScheduleValue::Text(trimmed.to_string());
    }

    // Two or more monetary amounts in one cell ("S$500 per claim, S$1,000 in
    // the aggregate") is a compound term. Typing it as its first number would
    // be a confident half-truth, so it is refused.
    if count_number_groups(trimmed) > 1 {
        return ScheduleValue::Unparseable {
            raw: trimmed.to_string(),
            reason: "more than one number in a single value; \
                     typing it as one would discard part of the term"
                .to_string(),
        };
    }

    if trimmed.contains('%') {
        return parse_percentage(trimmed);
    }

    parse_monetary_or_count(trimmed)
}

fn parse_percentage(trimmed: &str) -> ScheduleValue {
    let digits: String = trimmed
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.' || *c == ',')
        .filter(|c| *c != ',')
        .collect();

    match parse_scaled(&digits, 2) {
        Ok(basis_points) => ScheduleValue::Percentage(Percentage::from_basis_points(basis_points)),
        Err(reason) => ScheduleValue::Unparseable {
            raw: trimmed.to_string(),
            reason,
        },
    }
}

fn parse_monetary_or_count(trimmed: &str) -> ScheduleValue {
    let (currency, rest) = split_currency(trimmed);

    let numeric: String = rest
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.' || *c == ',' || *c == '-')
        .filter(|c| *c != ',')
        .collect();

    // Trailing prose after the number ("500 per claim") means the number is
    // real but qualified. Keep the number, and let the citation carry the
    // qualification — the verbatim row is right there.
    let has_currency = !matches!(currency, Currency::Unstated);

    if !has_currency && !rest.trim_start_matches(|c: char| c.is_ascii_digit()).trim().is_empty() {
        // A bare number with words after it: a count of days/visits/claims, or
        // prose we should not guess at.
        if let Ok(count) = numeric.parse::<i64>() {
            return ScheduleValue::Count(count);
        }
    }

    match parse_scaled(&numeric, 2) {
        Ok(cents) if has_currency => ScheduleValue::Money(MonetaryAmount::new(
            Money::from_cents(cents),
            currency,
        )),
        Ok(cents) if cents % 100 == 0 => ScheduleValue::Count(cents / 100),
        Ok(cents) => ScheduleValue::Money(MonetaryAmount::new(
            Money::from_cents(cents),
            Currency::Unstated,
        )),
        Err(reason) => ScheduleValue::Unparseable {
            raw: trimmed.to_string(),
            reason,
        },
    }
}

/// Splits a leading (or ISO-code) currency marker off a value.
fn split_currency(trimmed: &str) -> (Currency, &str) {
    let upper = trimmed.to_ascii_uppercase();

    for code in ISO_CODES {
        if upper.starts_with(code) {
            let rest = &trimmed[code.len()..];
            if rest.starts_with(|c: char| c.is_whitespace() || c.is_ascii_digit()) {
                return (Currency::Iso((*code).to_string()), rest.trim_start());
            }
        }
    }

    for (symbol, code) in UNAMBIGUOUS_SYMBOLS {
        if let Some(rest) = trimmed.strip_prefix(symbol) {
            return (Currency::Iso((*code).to_string()), rest.trim_start());
        }
    }

    for symbol in AMBIGUOUS_SYMBOLS {
        if let Some(rest) = trimmed.strip_prefix(symbol) {
            // Deliberately NOT resolved to a currency. See the module docs.
            return (Currency::Ambiguous((*symbol).to_string()), rest.trim_start());
        }
    }

    (Currency::Unstated, trimmed)
}

/// Parses a decimal string into an exact integer scaled by `10^scale`.
///
/// Refuses (rather than rounds) anything needing more precision than `scale`
/// decimal places: silently dropping a digit of a contractual amount is not an
/// acceptable failure mode.
fn parse_scaled(text: &str, scale: u32) -> Result<i64, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("no digits".to_string());
    }

    let (negative, digits) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };

    let (whole, fraction) = match digits.split_once('.') {
        Some((whole, fraction)) => (whole, fraction),
        None => (digits, ""),
    };

    if whole.is_empty() && fraction.is_empty() {
        return Err("no digits".to_string());
    }
    if digits.matches('.').count() > 1 {
        return Err(format!("{text:?} is not a single decimal number"));
    }
    if fraction.len() as u32 > scale {
        return Err(format!(
            "{text:?} needs more than {scale} decimal places; \
             rounding a contractual figure is not acceptable"
        ));
    }

    let whole: i64 = if whole.is_empty() {
        0
    } else {
        whole
            .parse()
            .map_err(|_| format!("{whole:?} is not a whole number"))?
    };

    let mut fraction_value: i64 = if fraction.is_empty() {
        0
    } else {
        fraction
            .parse()
            .map_err(|_| format!("{fraction:?} is not a fraction"))?
    };
    for _ in fraction.len() as u32..scale {
        fraction_value = fraction_value
            .checked_mul(10)
            .ok_or_else(|| "value overflows".to_string())?;
    }

    let scaled = whole
        .checked_mul(10_i64.pow(scale))
        .and_then(|scaled| scaled.checked_add(fraction_value))
        .ok_or_else(|| format!("{text:?} overflows an i64 at scale {scale}"))?;

    Ok(if negative { -scaled } else { scaled })
}

/// Counts runs of digits separated by non-numeric text, so that
/// `"S$500 per claim, S$1,000 in the aggregate"` is seen to hold two numbers
/// while `"S$1,000.50"` holds one.
fn count_number_groups(text: &str) -> usize {
    let mut groups = 0;
    let mut in_group = false;
    for c in text.chars() {
        if c.is_ascii_digit() {
            if !in_group {
                groups += 1;
                in_group = true;
            }
        } else if !matches!(c, ',' | '.') {
            in_group = false;
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_an_unambiguous_currency_exactly() {
        let ScheduleValue::Money(money) = parse_value("S$150,000") else {
            panic!("expected Money");
        };
        assert_eq!(money.currency().iso(), Some("SGD"));
        assert_eq!(money.amount().cents(), 15_000_000);
        assert_eq!(money.amount().to_decimal_string(), "150000.00");
    }

    #[test]
    fn parses_an_iso_code_and_cents_exactly() {
        let ScheduleValue::Money(money) = parse_value("SGD 1,234.56") else {
            panic!("expected Money");
        };
        assert_eq!(money.currency().iso(), Some("SGD"));
        assert_eq!(money.amount().cents(), 123_456);
    }

    #[test]
    fn a_bare_dollar_sign_is_not_resolved_to_a_currency() {
        // The single most tempting normalisation in this module, and the one
        // that would be inventing a fact: `$` is not USD, and it is not SGD.
        let ScheduleValue::Money(money) = parse_value("$1,500") else {
            panic!("expected Money");
        };
        assert_eq!(money.currency(), &Currency::Ambiguous("$".to_string()));
        assert_eq!(money.currency().iso(), None);
        assert_eq!(money.amount().cents(), 150_000);
    }

    #[test]
    fn nil_is_not_normalised_to_zero() {
        // "Nil excess" and "nil benefit" are not the same statement. Deciding
        // which one a document means is a domain judgment, not ours.
        assert_eq!(parse_value("Nil"), ScheduleValue::Nil("Nil".to_string()));
        assert_eq!(parse_value("N/A"), ScheduleValue::Nil("N/A".to_string()));
        assert!(parse_value("Nil").as_money().is_none());
    }

    #[test]
    fn unlimited_is_kept_as_printed() {
        assert_eq!(
            parse_value("Unlimited"),
            ScheduleValue::Unlimited("Unlimited".to_string())
        );
        assert_eq!(
            parse_value("As charged"),
            ScheduleValue::Unlimited("As charged".to_string())
        );
    }

    #[test]
    fn percentages_are_exact_basis_points_not_floats() {
        assert_eq!(
            parse_value("10%").as_percentage().unwrap().basis_points(),
            1000
        );
        assert_eq!(
            parse_value("12.5%").as_percentage().unwrap().basis_points(),
            1250
        );
        assert_eq!(parse_value("12.5%").as_percentage().unwrap().to_decimal_string(), "12.5");
    }

    #[test]
    fn a_percentage_too_precise_for_basis_points_is_refused_not_rounded() {
        let value = parse_value("3.3333%");
        assert!(value.is_unparseable(), "got {value:?}");
        let ScheduleValue::Unparseable { raw, reason } = value else {
            unreachable!()
        };
        assert_eq!(raw, "3.3333%");
        assert!(reason.contains("decimal places"), "{reason}");
    }

    #[test]
    fn percentage_arithmetic_is_exact() {
        // 10% of S$1,234.56 is 123.456 -> 123.46, exactly, with no float drift.
        let ten_percent = Percentage::from_basis_points(1000);
        let amount = Money::from_cents(123_456);
        assert_eq!(ten_percent.of(amount).unwrap().cents(), 12_346);
    }

    #[test]
    fn a_compound_value_is_refused_rather_than_half_read() {
        // Typing this as "S$500" would discard the aggregate limit entirely,
        // and would look completely convincing while doing so.
        let value = parse_value("S$500 per claim, S$1,000 in the aggregate");
        assert!(value.is_unparseable(), "got {value:?}");
    }

    #[test]
    fn a_count_is_a_count() {
        assert_eq!(parse_value("30 days"), ScheduleValue::Count(30));
        assert_eq!(parse_value("365"), ScheduleValue::Count(365));
    }

    #[test]
    fn prose_stays_prose() {
        assert_eq!(
            parse_value("As per the Schedule"),
            ScheduleValue::Text("As per the Schedule".to_string())
        );
    }

    #[test]
    fn parse_scaled_refuses_to_round_away_a_digit() {
        assert!(parse_scaled("1.234", 2).is_err());
        assert_eq!(parse_scaled("1.23", 2).unwrap(), 123);
        assert_eq!(parse_scaled("1.2", 2).unwrap(), 120);
        assert_eq!(parse_scaled("1", 2).unwrap(), 100);
    }
}
