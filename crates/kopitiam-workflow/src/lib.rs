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

mod context;
mod pipeline;
mod workflow;

pub use context::{Context, ContextBuilder, DEFAULT_MAX_FACTS};
pub use pipeline::run_workflow;
pub use workflow::{NamedWorkflow, Workflow, WorkflowKind};
