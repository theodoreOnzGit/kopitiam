//! The Context Builder and Workflow Engine for KOPITIAM's Semantic Runtime.
//!
//! This is the top of the Semantic Runtime's dependency stack: it sits
//! above `kopitiam-knowledge`, `kopitiam-index`, `kopitiam-search`,
//! `kopitiam-workspace`, `kopitiam-translation`, and `kopitiam-ai`, and is
//! the *only* crate in the whole platform allowed to invoke a model. See
//! `CLAUDE.md`'s "Semantic Runtime" section for the full architecture.
//!
//! Two concerns are deliberately kept apart:
//!
//! * [`ContextBuilder`] assembles a [`Context`] from `kopitiam-knowledge`
//!   (the semantic graph) and `kopitiam-workspace` (session memory) alone —
//!   it has no dependency on `kopitiam-ai` and cannot accidentally leak a
//!   vendor-specific shape into what "context" means.
//! * [`Workflow`] (and [`WorkflowKind`]'s eight named workflows — `plan`,
//!   `implement`, `translate`, `review`, `summarize`, `verify`,
//!   `document`, `resume`) renders a [`Context`] into a `kopitiam-ai`
//!   request and invokes a [`kopitiam_ai::ModelAdapter`].
//!
//! [`run_workflow`] wires both together into `CLAUDE.md`'s full pipeline:
//! `load state -> collect facts -> build context -> invoke model ->
//! validate -> persist`.
//!
//! # Scaffolded runtime seams (temp_ai_design.md)
//!
//! Sitting alongside the workflow engine are the scaffolded seams of the hybrid
//! AI architecture, each a skeleton with honest-stub behaviour, not a finished
//! build:
//!
//! * [`dispatch`] — the system-dispatch **fallback ladder** (§2): try existing
//!   knowledge → native Rust → local LLM → internet search → cloud LLM, in that
//!   fixed order, escalating only on an honest [`dispatch::Coverage::Indeterminate`]
//!   miss. Deliberately dumb; the routing lives here, never in a client.
//! * [`factquery`] — the determinism-boundary **fact-query seam** (§3, §8.1):
//!   the typed "ask the runtime" API a model proposes against — *LLM proposes,
//!   Rust disposes*. An open design question; the vocabulary is scaffolded and
//!   marked `TODO(decide-bead kopitiam-6ud)`.
//! * [`budget`] + [`context_session`] — progressive/anytime **context
//!   assembly** (§4): a local [`budget::ResourceBudget`] seam (a follow-up wires
//!   it to `kopitiam-resource`) and a single-owner background [`context_session`]
//!   actor, sibling to AID-0028. Context stays `f(task, budget)`, never
//!   `f(wall-clock)`.

mod budget;
mod context;
mod context_session;
mod dispatch;
mod factquery;
mod pipeline;
mod workflow;

pub use budget::{FactBudget, ResourceBudget};
pub use context::{Context, ContextBuilder, DEFAULT_MAX_FACTS};
pub use context_session::{ContextSession, SessionState};
pub use dispatch::{
    Answer, CloudModelProvider, Coverage, DispatchOutcome, Dispatcher, EscalationLog,
    EscalationRecord, ExistingKnowledgeProvider, InternetSearchProvider, LocalModelProvider, Miss,
    NativeRustProvider, Provider, Rung, Task,
};
pub use factquery::{FactAnswer, FactOracle, FactQuery, NullOracle};
pub use pipeline::run_workflow;
pub use workflow::{NamedWorkflow, Workflow, WorkflowKind};
