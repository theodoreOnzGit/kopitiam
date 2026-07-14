//! Synthetic policy documents, and the machinery to turn them into pages.
//!
//! # No real insurer's terms appear anywhere in this crate
//!
//! Every wording below is **fictional and written for the test**. "Kopi Assurance"
//! and the "National Basic Health Scheme" do not exist; the deductibles,
//! co-insurance rates, limits and waiting periods in them are invented figures
//! chosen to make the arithmetic easy to check by hand.
//!
//! This is not fussiness. A plausible-looking fake MediShield Life deductible
//! sitting in a test fixture is a number that will eventually be quoted by
//! somebody, to somebody, as though it were real — and in this domain that is
//! actively dangerous. MediShield Life's actual parameters are set by Singapore's
//! Ministry of Health and revised from time to time; the only correct way to learn
//! them is to read the current scheme document, which is exactly what this crate
//! is for.
//!
//! # Why the fixtures are PDF pages and not strings
//!
//! Because the pipeline under test is the real one. These pages go through
//! `kopitiam_document::reconstruct` (heading detection, paragraph assembly) and
//! `kopitiam_insurance::ingest_pages` (clause segmentation, definition extraction,
//! provenance) exactly as a real PDF's pages would. Feeding the extractor a
//! pre-segmented string would test the rules while quietly skipping the two layers
//! most likely to hand them something unexpected.

// This module is compiled separately into each integration-test binary, and each
// binary uses only the fixtures it needs. That is not dead code; it is the shape
// of Cargo's test harness.
#![allow(dead_code)]

use kopitiam_health::{
    DocumentId, ExtractionConfig, LayerKind, PolicyDocument, PolicyId, PolicyLayer, PolicyTerm,
    read_policy_pages,
};
use kopitiam_pdf::{Page, TextSpan};

/// Body text size. Headings are 16pt, which `kopitiam-document` reads as a level-1
/// heading (a ratio of 1.6 over the body size).
const BODY: f32 = 10.0;
const HEADING: f32 = 16.0;

/// Lays a wording out as PDF pages.
///
/// Lines beginning `#` become 16pt headings; everything else is 10pt body text. A
/// blank line starts a new page, so a fixture can exercise the page-break paths
/// that give clauses their page numbers.
///
/// `y` decreases down the page, matching the PDF convention that
/// `kopitiam-document` sorts by (descending `y` is reading order).
pub fn pages(source: &str) -> Vec<Page> {
    let mut pages = Vec::new();
    let mut spans = Vec::new();
    let mut y = 760.0_f32;

    let flush = |spans: &mut Vec<TextSpan>, pages: &mut Vec<Page>| {
        if !spans.is_empty() {
            pages.push(Page {
                number: pages.len() + 1,
                width: 595.0,
                height: 842.0,
                spans: std::mem::take(spans),
            });
        }
    };

    for line in source.lines() {
        let line = line.trim();
        if line.is_empty() {
            flush(&mut spans, &mut pages);
            y = 760.0;
            continue;
        }

        let (text, size) = match line.strip_prefix("# ") {
            Some(heading) => (heading, HEADING),
            None => (line, BODY),
        };

        spans.push(TextSpan {
            text: text.to_string(),
            x: 50.0,
            y,
            // A rough advance width. Nothing in the reconstruction we rely on
            // depends on it being exact; it only has to be plausible enough not to
            // trip the column-splitting heuristics.
            width: text.chars().count() as f32 * size * 0.5,
            height: size,
            font_size: size,
            ..TextSpan::default()
        });
        y -= size * 2.0;
    }

    flush(&mut spans, &mut pages);
    pages
}

/// Ingests a synthetic wording through the real pipeline and assembles a policy.
pub fn layer(id: &str, name: &str, kind: LayerKind, source: &str) -> PolicyLayer {
    let (document, terms) = read(id, kind, source, ExtractionConfig::new(kind));
    PolicyLayer::new(
        PolicyId::new(id).unwrap(),
        name,
        kind,
        document,
        terms,
    )
}

/// As [`layer`], but with an explicit extraction config (to exercise the
/// declared-currency path).
pub fn layer_with(
    id: &str,
    name: &str,
    kind: LayerKind,
    source: &str,
    config: ExtractionConfig,
) -> PolicyLayer {
    let (document, terms) = read(id, kind, source, config);
    PolicyLayer::new(PolicyId::new(id).unwrap(), name, kind, document, terms)
}

fn read(
    id: &str,
    _kind: LayerKind,
    source: &str,
    config: ExtractionConfig,
) -> (PolicyDocument, Vec<PolicyTerm>) {
    read_policy_pages(
        DocumentId::new(format!("{id}.pdf")).unwrap(),
        &pages(source),
        &config,
    )
    .expect("the synthetic wording must ingest cleanly")
}

// ---------------------------------------------------------------------------
// The fixtures. All fictional. See the module docs.
// ---------------------------------------------------------------------------

/// A fictional universal basic scheme, standing in for the *structure* of
/// MediShield Life without borrowing any of its actual terms.
pub const BASIC_SCHEME: &str = "\
# National Basic Health Scheme (Illustrative Synthetic Scheme)
# Part 1 - Definitions
1.1 \"Claimable Amount\" means the part of a Bill that is eligible for payment under this Scheme.
1.2 \"Policy Year\" means each period of twelve months beginning on the Commencement Date.

# Part 2 - Cost sharing
2.1 The Deductible is S$1,500 for each policy year.
2.2 A co-insurance of 10% applies to the Claimable Amount above the Deductible.
2.3 We will pay up to S$100,000 for each policy year.
";

/// A fictional Integrated Shield Plan whose wording makes its benefit **inclusive
/// of** the basic scheme's payout — so its deductible bites on the whole claimable
/// amount.
pub const SHIELD_INCLUSIVE: &str = "\
# Kopi Assurance Teh Tarik Shield (Illustrative Synthetic Wording)
# Part 1 - Definitions
1.1 \"Claimable Amount\" means the part of a Bill that is eligible for payment under this Plan.
1.2 \"Policy Year\" means each period of twelve months beginning on the Commencement Date.
1.3 \"Hospitalisation\" means admission to a Hospital as an in-patient for at least one night.

# Part 2 - How this Plan works with the Scheme
2.1 The benefit payable under this Plan is inclusive of the amount payable under the Scheme.

# Part 3 - Cost sharing
3.1 The Deductible is S$3,500 for each policy year.
3.2 A co-insurance of 10% applies to the Claimable Amount above the Deductible.
3.3 We will pay up to S$150,000 for each policy year.

# Part 4 - What is not covered
4.1 We will not pay for any Pre-existing Condition.
4.2 A waiting period of 12 months applies to treatment for a Specified Illness.
";

/// The **same plan**, word for word, except that clause 2.1 makes its benefit apply
/// only **in excess of** what the basic scheme pays.
///
/// The two fixtures exist to be run against the identical bill. They give
/// materially different answers, which is the whole reason
/// [`kopitiam_health::IntegrationMode`] is a term this crate refuses to guess at.
pub const SHIELD_EXCESS: &str = "\
# Kopi Assurance Kopi Peng Shield (Illustrative Synthetic Wording)
# Part 1 - Definitions
1.1 \"Claimable Amount\" means the part of a Bill that is eligible for payment under this Plan.
1.2 \"Policy Year\" means each period of twelve months beginning on the Commencement Date.
1.3 \"Hospitalisation\" means admission to a Hospital as an in-patient for at least one night.

# Part 2 - How this Plan works with the Scheme
2.1 The benefit payable under this Plan applies only in excess of the amount payable under the Scheme.

# Part 3 - Cost sharing
3.1 The Deductible is S$3,500 for each policy year.
3.2 A co-insurance of 10% applies to the Claimable Amount above the Deductible.
3.3 We will pay up to S$150,000 for each policy year.

# Part 4 - What is not covered
4.1 We will not pay for any Pre-existing Condition.
4.2 A waiting period of 12 months applies to treatment for a Specified Illness.
";

/// A fictional rider that reimburses the plan's deductible and co-insurance, less a
/// residual co-payment.
pub const RIDER: &str = "\
# Kopi Assurance Kopi-O Rider (Illustrative Synthetic Wording)
# Part 1 - What this Rider does
1.1 This Rider reimburses the Deductible and the co-insurance payable by the Insured under the Plan.
1.2 A co-payment of 5% of the claimable amount applies, capped at S$3,000 for each policy year.
";

/// A plan whose deductible clause never says what the deductible is charged
/// against. The amount is there; the basis is not — and per year versus per claim is
/// not a detail.
pub const SHIELD_UNDERSPECIFIED_DEDUCTIBLE: &str = "\
# Kopi Assurance Vague Shield (Illustrative Synthetic Wording)
# Part 2 - How this Plan works with the Scheme
2.1 The benefit payable under this Plan is inclusive of the amount payable under the Scheme.

# Part 3 - Cost sharing
3.1 The Deductible is S$3,500.
3.2 A co-insurance of 10% applies to the Claimable Amount above the Deductible.
3.3 We will pay up to S$150,000 for each policy year.
";

/// A plan that states no claim limit at all.
///
/// A missing ceiling must never become "no ceiling".
pub const SHIELD_NO_LIMIT: &str = "\
# Kopi Assurance Limitless Shield (Illustrative Synthetic Wording)
# Part 2 - How this Plan works with the Scheme
2.1 The benefit payable under this Plan is inclusive of the amount payable under the Scheme.

# Part 3 - Cost sharing
3.1 The Deductible is S$3,500 for each policy year.
3.2 A co-insurance of 10% applies to the Claimable Amount above the Deductible.
";

/// A plan that states its deductible twice, differently, at the same level of
/// generality. Real wordings do this when a schedule and a body clause are revised
/// out of step.
pub const SHIELD_CONTRADICTORY_DEDUCTIBLE: &str = "\
# Kopi Assurance Two-Minds Shield (Illustrative Synthetic Wording)
# Part 2 - How this Plan works with the Scheme
2.1 The benefit payable under this Plan is inclusive of the amount payable under the Scheme.

# Part 3 - Cost sharing
3.1 The Deductible is S$3,500 for each policy year.
3.2 A co-insurance of 10% applies to the Claimable Amount above the Deductible.
3.3 We will pay up to S$150,000 for each policy year.

# Part 9 - Schedule
9.1 The Deductible is S$2,000 for each policy year.
";

/// A plan whose integration clause says two things that cannot both be true.
pub const SHIELD_CONTRADICTORY_INTEGRATION: &str = "\
# Kopi Assurance Schrodinger Shield (Illustrative Synthetic Wording)
# Part 2 - How this Plan works with the Scheme
2.1 The benefit payable under this Plan is inclusive of the amount payable under the Scheme and applies only in excess of the amount payable under the Scheme.

# Part 3 - Cost sharing
3.1 The Deductible is S$3,500 for each policy year.
3.2 A co-insurance of 10% applies to the Claimable Amount above the Deductible.
3.3 We will pay up to S$150,000 for each policy year.
";

/// A plan that prints its amounts with a bare `$` and never says what currency that
/// is. In a document that also mentions US dollars, that is a genuinely open
/// question.
pub const SHIELD_BARE_DOLLAR: &str = "\
# Kopi Assurance Which Dollar Shield (Illustrative Synthetic Wording)
# Part 2 - How this Plan works with the Scheme
2.1 The benefit payable under this Plan is inclusive of the amount payable under the Scheme.

# Part 3 - Cost sharing
3.1 The Deductible is $3,500 for each policy year.
3.2 A co-insurance of 10% applies to the Claimable Amount above the Deductible.
3.3 We will pay up to $150,000 for each policy year.
";

/// A rider that says what it reimburses but never states its co-payment.
///
/// A missing co-payment clause must never become a 0% co-payment: that would
/// understate what the patient pays, which is the harmful direction to be wrong in.
pub const RIDER_NO_COPAYMENT: &str = "\
# Kopi Assurance Silent Rider (Illustrative Synthetic Wording)
# Part 1 - What this Rider does
1.1 This Rider reimburses the Deductible and the co-insurance payable by the Insured under the Plan.
";

/// A rider whose cover is conditional on facts about the patient's treatment — facts
/// this crate does not have and must not pretend to.
pub const RIDER_CONDITIONAL: &str = "\
# Kopi Assurance Maybe Rider (Illustrative Synthetic Wording)
# Part 1 - What this Rider does
1.1 This Rider reimburses the Deductible and the co-insurance payable by the Insured, provided that the treatment is received from a Panel Specialist.
1.2 A co-payment of 5% of the claimable amount applies, capped at S$3,000 for each policy year.
";

/// A competing plan that defines "Hospitalisation" to **include day surgery**, where
/// the others require an overnight stay.
///
/// Its deductible happens to be the same number as another plan's. It is not the
/// same deductible: it is charged on a different set of events.
pub const SHIELD_RIVAL_WIDER_DEFINITION: &str = "\
# Kopi Assurance Rival Shield (Illustrative Synthetic Wording)
# Part 1 - Definitions
1.1 \"Claimable Amount\" means the part of a Bill that is eligible for payment under this Plan.
1.2 \"Policy Year\" means each period of twelve months beginning on the Commencement Date.
1.3 \"Hospitalisation\" means admission to a Hospital as an in-patient or for day surgery.

# Part 2 - How this Plan works with the Scheme
2.1 The benefit payable under this Plan is inclusive of the amount payable under the Scheme.

# Part 3 - Cost sharing
3.1 The Deductible is S$3,500 for each policy year.
3.2 A co-insurance of 10% applies to the Claimable Amount above the Deductible.
3.3 We will pay up to S$150,000 for each policy year.
";
