//! System dispatch — the hard-coded Rust fallback ladder (temp_ai_design.md §2).
//!
//! This is *level 2* dispatch: per-**task** routing between knowledge, native
//! Rust, local LLM, internet search, and cloud LLM. It is **not** MoE
//! per-token routing (level 1, a model-internal detail) and **not** the
//! determinism boundary (level 3, see [`crate::factquery`]). Don't conflate
//! the three lah.
//!
//! # The one contract you must not break
//!
//! The dispatcher is **deliberately dumb**. It does **not** predict which rung
//! can answer a task — that circular "AI deciding whether to use AI" trap
//! (`CLAUDE.md`'s "AI-dependent workflows" ban) is exactly what we refuse to
//! build. Instead:
//!
//! > **Try each rung in fixed priority order; a rung either answers honestly
//! > from its own coverage, or reports [`Coverage::Indeterminate`]. A *miss* is
//! > the routing signal — never a guess, never a bluff.**
//!
//! So the intelligence does not live in the dispatcher. It lives in each
//! [`Provider`] being honest about what it can and cannot cover — the same
//! `Indeterminate` discipline `kopitiam-finance` / `kopitiam-legal` already
//! use. If a provider bluffs (answers when it shouldn't), the whole ladder is
//! poisoned; that is the single worst thing a provider can do here.
//!
//! # The ladder order (and why internet-search sits where it does)
//!
//! Priority is fixed by [`Rung::priority`], lowest number tried first:
//!
//! | # | [`Rung`] | Network? | Why here |
//! |---|---|---|---|
//! | 0 | [`Rung::ExistingKnowledge`] | offline | already-computed facts, free, instant |
//! | 1 | [`Rung::NativeRust`] | offline | LSP / cargo / parsers — deterministic, free |
//! | 2 | [`Rung::LocalModel`] | offline | local LLM reasoning, no tokens, no network |
//! | 3 | [`Rung::InternetSearch`] | **network** | web retrieval — grounded facts, cheap-ish |
//! | 4 | [`Rung::CloudModel`] | **network** | cloud LLM reasoning — last resort, costs tokens |
//!
//! The ordering is **Offline-First** (`CLAUDE.md`): every offline rung (0–2)
//! is exhausted before *any* network rung (3–4). "Running out of AI tokens
//! should never block knowledge work" — so a local LLM (offline) is always
//! preferred over reaching for the network.
//!
//! Internet-search (the maintainer's added rung) sits **late and optional**:
//! after the offline rungs, and **before** cloud. Two reasons:
//! 1. It needs the network, so Offline-First pushes it past every offline rung.
//! 2. It *retrieves* facts (which can be re-grounded and cited — see
//!    [`crate::factquery`]'s "LLM proposes, Rust disposes"), whereas cloud
//!    *reasons* probabilistically. Grounded retrieval is more trustworthy and
//!    cheaper than cloud reasoning, so it comes first. Its real implementation
//!    will wrap `kopitiam-web` (`SearchProvider`) via `kopitiam-internet-research`;
//!    for now it is a stub — see [`InternetSearchProvider`].
//!
//! Whatever happens, internet-search is **never rung 0**. Reaching for the web
//! before checking what KOPITIAM already knows would violate both Offline-First
//! and the AI Philosophy ("Never ask ... to rediscover information already
//! present inside KOPITIAM").
//!
//! # Escalation logging (§2 guardrail 2)
//!
//! Every fall-through past a deterministic rung is a **gap that layer could not
//! cover**. [`Dispatcher`] records each [`Miss`] into an [`EscalationLog`].
//! Recurring misses on a given rung are the signal for *which deterministic
//! provider to build next* — the ladder teaches you where to extend it. See
//! [`EscalationLog::gaps_by_rung`].

use std::collections::BTreeMap;

/// A task handed to the [`Dispatcher`] to be routed down the ladder.
///
/// Note: this is a *routing request* — the thing being dispatched — **not**
/// [`kopitiam_ontology::EntityKind::Task`] (a tracked unit of work in the
/// knowledge graph). Different concept, same English word. This one is just
/// "here's an ask, find me a rung that can answer it".
///
/// It carries only what a provider needs to *self-assess coverage*. It
/// deliberately carries **no** "intent" / "category" field: the moment you add
/// one, someone will route on it, and you're back to the predictive-classifier
/// trap §2 forbids. Providers inspect the task themselves; they never trust a
/// pre-computed label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Task {
    /// The ask, as posed — natural language or a structured query string.
    pub query: String,
}

impl Task {
    pub fn new(query: impl Into<String>) -> Self {
        Self { query: query.into() }
    }
}

/// A concrete answer produced by one rung of the ladder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    /// The answer content.
    pub content: String,
    /// Which rung produced it — provenance, so a caller (and the escalation
    /// log) always knows whether this came from a deterministic source or a
    /// model. `CLAUDE.md`'s Provenance Standards, applied to routing.
    pub answered_by: Rung,
}

/// A provider's honest verdict on one task: either it covers it, or it doesn't.
///
/// This is the whole point of the ladder — a provider **must** return
/// [`Coverage::Indeterminate`] rather than bluff an answer it isn't sure of.
/// The `reason` on a miss is not decoration: it feeds [`Miss::reason`] and thus
/// the escalation log, so "why did rung 1 keep missing on symbol-type queries"
/// is answerable later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Coverage {
    /// The provider covers this task and answers it — honestly, from its own
    /// coverage.
    Answered(Answer),
    /// The provider cannot cover this task. Carries a short reason for the log.
    /// **This is the routing signal**, not a failure.
    Indeterminate(String),
}

/// One rung of the fallback ladder, in priority order.
///
/// The enum discriminant order here is *not* the priority — use
/// [`Rung::priority`], which is the single source of truth the [`Dispatcher`]
/// sorts by. (Keeping them separate means reordering the enum for readability
/// can never silently change routing.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rung {
    /// Rung 0 — already-computed knowledge (`kopitiam-knowledge` / graph).
    ExistingKnowledge,
    /// Rung 1 — native Rust tooling (rust-analyzer, cargo, pure-Rust parsers).
    NativeRust,
    /// Rung 2 — local LLM reasoning (offline, no tokens).
    LocalModel,
    /// Rung 3 — internet search (network; `kopitiam-web` retrieval). Late/optional.
    InternetSearch,
    /// Rung 4 — cloud LLM reasoning (network, costs tokens). Final fallback.
    CloudModel,
}

impl Rung {
    /// Fixed routing priority, lowest tried first. **This** is the ladder
    /// order; changing it changes routing, so it is the one place to touch.
    pub fn priority(self) -> u8 {
        match self {
            Rung::ExistingKnowledge => 0,
            Rung::NativeRust => 1,
            Rung::LocalModel => 2,
            Rung::InternetSearch => 3,
            Rung::CloudModel => 4,
        }
    }

    /// True for rungs that compute facts deterministically (rungs 0–1). A miss
    /// here is the strongest "build a provider" signal — it means something the
    /// runtime *could* have known deterministically had to escalate.
    pub fn is_deterministic(self) -> bool {
        matches!(self, Rung::ExistingKnowledge | Rung::NativeRust)
    }

    /// True for rungs that need the network (rungs 3–4). Offline-First keeps
    /// these strictly after every offline rung.
    pub fn needs_network(self) -> bool {
        matches!(self, Rung::InternetSearch | Rung::CloudModel)
    }

    /// True for the LLM rungs (local + cloud). A fall-through to one of these
    /// is what §2 calls "escalation to the LLM" — the thing worth logging.
    pub fn is_model(self) -> bool {
        matches!(self, Rung::LocalModel | Rung::CloudModel)
    }

    /// Human-readable rung name for logs.
    pub fn label(self) -> &'static str {
        match self {
            Rung::ExistingKnowledge => "existing-knowledge",
            Rung::NativeRust => "native-rust",
            Rung::LocalModel => "local-model",
            Rung::InternetSearch => "internet-search",
            Rung::CloudModel => "cloud-model",
        }
    }
}

/// One rung of the ladder. Implementors **must** self-report coverage honestly.
///
/// The contract, restated because it is the whole design: given a task a
/// provider is not confident it can answer, it returns
/// [`Coverage::Indeterminate`] — it does **not** produce a plausible-looking
/// answer. The dispatcher's correctness depends entirely on this honesty; a
/// bluffing provider is a bug, not a quality issue.
pub trait Provider {
    /// Which rung this provider occupies. Determines its place in the ladder.
    fn rung(&self) -> Rung;

    /// Short stable name, for the escalation log (`"kopitiam-knowledge"`,
    /// `"rust-analyzer"`, `"brave-search"`, `"local-qwen"`, ...).
    fn name(&self) -> &str;

    /// Attempt the task. Answer only if genuinely covered; otherwise
    /// [`Coverage::Indeterminate`] with a short reason for the log.
    fn try_answer(&self, task: &Task) -> Coverage;
}

/// A recorded fall-through: rung `provider`/`rung` looked at a task and missed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Miss {
    pub rung: Rung,
    pub provider: String,
    pub reason: String,
}

/// The outcome of one [`Dispatcher::dispatch`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchOutcome {
    /// The answer, if any rung covered the task. `None` means every rung
    /// missed — which, in this scaffold (all providers are stubs), is the
    /// normal case. In a real ladder the cloud rung answers, so `None` there
    /// would mean total exhaustion.
    pub answer: Option<Answer>,
    /// Every rung that returned [`Coverage::Indeterminate`], in ladder order,
    /// up to (not including) the rung that answered.
    pub misses: Vec<Miss>,
}

impl DispatchOutcome {
    /// True if the ladder had to skip a deterministic rung to get here — i.e.
    /// something the runtime might have known deterministically forced a climb.
    /// This is the "escalation" §2 says to persist.
    pub fn is_escalation(&self) -> bool {
        self.misses.iter().any(|m| m.rung.is_deterministic())
    }
}

/// An append-only record of every escalation the ladder took.
///
/// §2 guardrail 2: "Persist every escalation." Recurring misses on a rung tell
/// you which deterministic provider to build next — the log is the ladder's
/// own to-do list. This scaffold keeps it in memory; wiring it to
/// `kopitiam-index` for cross-session persistence is a follow-up bead.
#[derive(Debug, Default, Clone)]
pub struct EscalationLog {
    records: Vec<EscalationRecord>,
}

/// One escalation: a task, the rungs that missed on it, and who finally answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EscalationRecord {
    pub task: String,
    pub misses: Vec<Miss>,
    pub answered_by: Option<Rung>,
}

impl EscalationLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a dispatch, but only if it actually escalated (had ≥1 miss).
    /// A task answered by rung 0 first-try is not a gap and does not clutter
    /// the log.
    pub fn record(&mut self, task: &Task, outcome: &DispatchOutcome) {
        if outcome.misses.is_empty() {
            return;
        }
        self.records.push(EscalationRecord {
            task: task.query.clone(),
            misses: outcome.misses.clone(),
            answered_by: outcome.answer.as_ref().map(|a| a.answered_by),
        });
    }

    /// Every recorded escalation, oldest first.
    pub fn records(&self) -> &[EscalationRecord] {
        &self.records
    }

    /// How many times each rung has missed, across all recorded escalations.
    /// The rung with the most misses is the loudest "build me a real provider"
    /// signal. Sorted by rung for stable, testable output.
    pub fn gaps_by_rung(&self) -> BTreeMap<Rung, usize> {
        let mut gaps = BTreeMap::new();
        for record in &self.records {
            for miss in &record.misses {
                *gaps.entry(miss.rung).or_insert(0) += 1;
            }
        }
        gaps
    }
}

/// The hard-coded fallback ladder. Owns its providers (sorted by rung
/// priority) and an [`EscalationLog`].
///
/// Routing lives **here**, in `kopitiam-workflow` — never in the CLI/TUI (§2
/// guardrail 1). Clients call [`Dispatcher::dispatch`]; they do not decide the
/// lane.
pub struct Dispatcher {
    /// Providers, kept sorted ascending by [`Rung::priority`]. Invariant
    /// maintained by [`Dispatcher::new`].
    providers: Vec<Box<dyn Provider>>,
    log: EscalationLog,
}

impl Dispatcher {
    /// Build a dispatcher from any set of providers. They are sorted into
    /// ladder order here, so callers cannot accidentally route out of
    /// priority order by passing them in the wrong sequence.
    pub fn new(mut providers: Vec<Box<dyn Provider>>) -> Self {
        providers.sort_by_key(|p| p.rung().priority());
        Self { providers, log: EscalationLog::new() }
    }

    /// The full stub ladder: one stub per rung, each honestly reporting
    /// `Indeterminate` (nothing is implemented yet). This is the scaffold's
    /// default — swap individual stubs for real providers as they land.
    pub fn with_default_ladder() -> Self {
        Self::new(vec![
            Box::new(ExistingKnowledgeProvider),
            Box::new(NativeRustProvider),
            Box::new(LocalModelProvider),
            Box::new(InternetSearchProvider),
            Box::new(CloudModelProvider),
        ])
    }

    /// Try the task down the ladder. Returns the first rung's answer, recording
    /// every miss above it into the [`EscalationLog`].
    ///
    /// This is temp_ai_design.md §2's `dispatch` function made concrete:
    /// deliberately dumb, no prediction — just try, and let honest misses route.
    pub fn dispatch(&mut self, task: &Task) -> DispatchOutcome {
        let mut misses = Vec::new();
        let mut answer = None;

        for provider in &self.providers {
            match provider.try_answer(task) {
                Coverage::Answered(a) => {
                    answer = Some(a);
                    break;
                }
                Coverage::Indeterminate(reason) => {
                    misses.push(Miss { rung: provider.rung(), provider: provider.name().to_string(), reason });
                }
            }
        }

        let outcome = DispatchOutcome { answer, misses };
        self.log.record(task, &outcome);
        outcome
    }

    /// The escalation log, for inspecting which deterministic providers the
    /// ladder keeps wishing it had.
    pub fn log(&self) -> &EscalationLog {
        &self.log
    }
}

// ---------------------------------------------------------------------------
// Stub providers — one per rung.
//
// Every one honestly reports `Indeterminate`: NOTHING is implemented yet, so
// nothing may answer. This is the scaffold contract — the deliverable is the
// ladder + trait + escalation log, not working providers. Each stub's rustdoc
// records what its *real* implementation will consult, so the next person knows
// where to plug in.
// ---------------------------------------------------------------------------

/// Rung 0 stub. Real version will query the `kopitiam-knowledge` semantic
/// graph (and `kopitiam-search`) for an already-computed answer.
pub struct ExistingKnowledgeProvider;
impl Provider for ExistingKnowledgeProvider {
    fn rung(&self) -> Rung {
        Rung::ExistingKnowledge
    }
    fn name(&self) -> &str {
        "kopitiam-knowledge(stub)"
    }
    fn try_answer(&self, _task: &Task) -> Coverage {
        Coverage::Indeterminate("stub: knowledge graph lookup not wired yet".into())
    }
}

/// Rung 1 stub. Real version will run native Rust tooling — rust-analyzer
/// (`kopitiam-semantic`), `cargo metadata`, pure-Rust parsers (`kopitiam-syntax`).
pub struct NativeRustProvider;
impl Provider for NativeRustProvider {
    fn rung(&self) -> Rung {
        Rung::NativeRust
    }
    fn name(&self) -> &str {
        "native-rust(stub)"
    }
    fn try_answer(&self, _task: &Task) -> Coverage {
        Coverage::Indeterminate("stub: native Rust tooling not wired yet".into())
    }
}

/// Rung 2 stub. Real version will invoke a local model adapter
/// (`kopitiam_ai::LocalAdapter`) — offline, no tokens. `kopitiam-workflow` is
/// the only crate allowed to touch `kopitiam-ai`, so this rung lives here.
pub struct LocalModelProvider;
impl Provider for LocalModelProvider {
    fn rung(&self) -> Rung {
        Rung::LocalModel
    }
    fn name(&self) -> &str {
        "local-model(stub)"
    }
    fn try_answer(&self, _task: &Task) -> Coverage {
        Coverage::Indeterminate("stub: local model adapter not wired yet".into())
    }
}

/// Rung 3 stub. Real version will wrap `kopitiam-web`'s `SearchProvider`
/// (Brave / SearxNG) via `kopitiam-internet-research` (itself scaffold-only for
/// now). Late/optional by Offline-First — see the module docs for why this rung
/// sits after every offline rung and before cloud.
pub struct InternetSearchProvider;
impl Provider for InternetSearchProvider {
    fn rung(&self) -> Rung {
        Rung::InternetSearch
    }
    fn name(&self) -> &str {
        "internet-search(stub)"
    }
    fn try_answer(&self, _task: &Task) -> Coverage {
        Coverage::Indeterminate("stub: kopitiam-web/internet-research not wired yet".into())
    }
}

/// Rung 4 stub. Real version will invoke a cloud model adapter — the final
/// fallback, costs tokens, needs network. In a real ladder this rung
/// (almost) always answers; here it honestly reports `Indeterminate`.
pub struct CloudModelProvider;
impl Provider for CloudModelProvider {
    fn rung(&self) -> Rung {
        Rung::CloudModel
    }
    fn name(&self) -> &str {
        "cloud-model(stub)"
    }
    fn try_answer(&self, _task: &Task) -> Coverage {
        Coverage::Indeterminate("stub: cloud model adapter not wired yet".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test provider that either answers or misses on demand, so we can
    /// exercise the ladder's short-circuit and fall-through behaviour without
    /// real providers.
    struct FakeProvider {
        rung: Rung,
        answers: bool,
    }
    impl Provider for FakeProvider {
        fn rung(&self) -> Rung {
            self.rung
        }
        fn name(&self) -> &str {
            "fake"
        }
        fn try_answer(&self, task: &Task) -> Coverage {
            if self.answers {
                Coverage::Answered(Answer { content: format!("{}:{}", self.rung.label(), task.query), answered_by: self.rung })
            } else {
                Coverage::Indeterminate("fake miss".into())
            }
        }
    }

    #[test]
    fn tries_rungs_in_priority_order_regardless_of_insertion_order() {
        // Insert cloud first, knowledge last — dispatcher must still try
        // knowledge (rung 0) before cloud (rung 4).
        let mut d = Dispatcher::new(vec![
            Box::new(FakeProvider { rung: Rung::CloudModel, answers: true }),
            Box::new(FakeProvider { rung: Rung::ExistingKnowledge, answers: true }),
        ]);
        let outcome = d.dispatch(&Task::new("q"));
        assert_eq!(outcome.answer.unwrap().answered_by, Rung::ExistingKnowledge);
        assert!(outcome.misses.is_empty(), "rung 0 answered first-try, nothing skipped");
    }

    #[test]
    fn a_stub_that_answers_short_circuits_the_ladder() {
        // Rung 0 misses, rung 1 answers → rungs 2..4 never consulted.
        let mut d = Dispatcher::new(vec![
            Box::new(FakeProvider { rung: Rung::ExistingKnowledge, answers: false }),
            Box::new(FakeProvider { rung: Rung::NativeRust, answers: true }),
            Box::new(FakeProvider { rung: Rung::LocalModel, answers: true }),
        ]);
        let outcome = d.dispatch(&Task::new("q"));
        assert_eq!(outcome.answer.unwrap().answered_by, Rung::NativeRust);
        // exactly one miss recorded — rung 0 — and rung 2 was never reached.
        assert_eq!(outcome.misses.len(), 1);
        assert_eq!(outcome.misses[0].rung, Rung::ExistingKnowledge);
    }

    #[test]
    fn a_stub_that_misses_falls_through_to_the_next_rung() {
        // Knowledge + native both miss → escalates to the local model.
        let mut d = Dispatcher::new(vec![
            Box::new(FakeProvider { rung: Rung::ExistingKnowledge, answers: false }),
            Box::new(FakeProvider { rung: Rung::NativeRust, answers: false }),
            Box::new(FakeProvider { rung: Rung::LocalModel, answers: true }),
        ]);
        let outcome = d.dispatch(&Task::new("q"));
        assert_eq!(outcome.answer.as_ref().unwrap().answered_by, Rung::LocalModel);
        assert_eq!(outcome.misses.len(), 2);
        assert!(outcome.is_escalation(), "two deterministic rungs missed — this is a gap");
    }

    #[test]
    fn all_stubs_indeterminate_means_exhausted_and_no_answer() {
        // The scaffold's real default: every rung is a stub, nothing answers.
        let mut d = Dispatcher::with_default_ladder();
        let outcome = d.dispatch(&Task::new("what is the return type of select_adapter"));
        assert!(outcome.answer.is_none(), "all stubs miss → no answer");
        assert_eq!(outcome.misses.len(), 5, "every rung on the ladder missed");
    }

    #[test]
    fn escalation_log_records_gaps_and_points_at_the_next_provider_to_build() {
        let mut d = Dispatcher::with_default_ladder();
        d.dispatch(&Task::new("q1"));
        d.dispatch(&Task::new("q2"));
        // Two dispatches, each missing all five rungs → 2 records, each rung
        // missed twice. The gaps map is the ladder's own build-next list.
        assert_eq!(d.log().records().len(), 2);
        let gaps = d.log().gaps_by_rung();
        assert_eq!(gaps[&Rung::ExistingKnowledge], 2);
        assert_eq!(gaps[&Rung::NativeRust], 2);
        assert_eq!(gaps[&Rung::CloudModel], 2);
    }

    #[test]
    fn a_first_try_answer_is_not_logged_as_an_escalation() {
        let mut d = Dispatcher::new(vec![Box::new(FakeProvider { rung: Rung::ExistingKnowledge, answers: true })]);
        d.dispatch(&Task::new("q"));
        assert!(d.log().records().is_empty(), "answered by rung 0 — no gap, no log entry");
    }

    #[test]
    fn offline_rungs_all_sort_before_network_rungs() {
        // Guards Offline-First at the type level: no network rung may ever
        // out-prioritise an offline rung.
        for offline in [Rung::ExistingKnowledge, Rung::NativeRust, Rung::LocalModel] {
            for network in [Rung::InternetSearch, Rung::CloudModel] {
                assert!(offline.priority() < network.priority(), "{offline:?} must precede {network:?}");
                assert!(network.needs_network());
                assert!(!offline.needs_network());
            }
        }
        // And internet-search precedes cloud (retrieval before reasoning).
        assert!(Rung::InternetSearch.priority() < Rung::CloudModel.priority());
    }
}
