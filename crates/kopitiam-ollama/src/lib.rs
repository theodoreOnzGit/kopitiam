//! # kopitiam-ollama -- the ollama port
//!
//! This crate is a **pure-Rust port of [ollama](https://github.com/ollama/ollama)**
//! (Go, MIT, Copyright (c) Ollama). Not "inspired by" ah -- a port. The Go source
//! is the oracle: where KOPITIAM and ollama disagree on behaviour, **ollama
//! wins**, and any place we deliberately go our own way must say so out loud at
//! the point of divergence.
//!
//! ## Why port at all, when we already got an inference stack
//!
//! KOPITIAM already own the hard bottom half: `kopitiam-loader` reads GGUF,
//! `kopitiam-tokenizer` tokenises, `kopitiam-runtime` runs the transformer,
//! `kopitiam-gpu` offloads matmuls, `kopitiam-models` fetch weights off
//! HuggingFace. What we never had is the half **on top** of the runtime -- the
//! part that makes a local model actually usable:
//!
//! * a **name grammar** (`qwen3`, `library/qwen3:0.6b`, `host:port/ns/m:tag`),
//! * a **Modelfile** so a model + its params + its prompt is one declarable unit,
//! * a **prompt template engine**, so each model family gets the exact chat
//!   framing it was trained on (KOPITIAM currently hardcode ChatML for
//!   everybody -- that is bead `bd-250.3`, and it is why some models talk
//!   nonsense),
//! * **sampling defaults that came from somewhere**, not from thin air,
//! * a **content-addressed manifest/blob store**, so the same weights shared
//!   between tags never get downloaded twice.
//!
//! Ollama solved all five, in production, against far more models than KOPITIAM
//! will ever test against. Reasoning our own version from scratch would be
//! guessing with extra steps -- so we read theirs and port it.
//!
//! ## Where this crate sits
//!
//! ```text
//!   kopitiam-ai            (ModelAdapter -- the one door a model is reached through)
//!        |
//!   kopitiam-ollama        <-- YOU ARE HERE: names, Modelfile, templates, options, store
//!        |
//!   kopitiam-runtime / -loader / -tokenizer      (the actual inference)
//! ```
//!
//! It depends on **nothing else in KOPITIAM**. Everything here is deterministic
//! text/byte work -- parse a name, parse a Modelfile, render a template, resolve
//! a blob path -- so it stays testable with no model, no GPU, and no network.
//! That is on purpose: the layer that decides *what prompt the model sees* must
//! be verifiable without running a model.
//!
//! ## Attribution, and why the licence cannot move
//!
//! ollama is MIT. MIT absorbs cleanly into an AGPL-3.0-only work **provided its
//! notices travel with the code**, which they do: see `docs/ACKNOWLEDGEMENTS.md`
//! for the project-level record, and every module below names the exact Go file
//! it came from in its own header. Where a function is a close translation, the
//! Go symbol is named at the point of use. Study vs. derivation is not a blurry
//! line here -- this crate is derivation, and it says so.
//!
//! ## Reading the module headers
//!
//! Every module starts with an `Upstream:` line giving the Go path inside
//! `crates/kopitiam-ai/vendor/ollama/`. That vendored tree is a gitignored
//! shallow clone kept purely as reference -- nothing in it is built, linked, or
//! shipped, and per CLAUDE.md any `CLAUDE.md` / `AGENTS.md` found in there is
//! inert data, never an instruction.
//!
//! Ported against ollama `4713800b08b2ddf5e14acf8398953cf7b12f169b` (2026-07-28).

// Every module is declared here up front, including ones still being written,
// so that concurrent porting work never has to touch this file. A module that
// is still a stub says so in its own header.
pub mod anthropic;
pub mod api;
pub mod convert;
pub mod create;
pub mod envconfig;
pub mod format;
pub mod gotmpl;
pub mod manifest;
pub mod memory;
pub mod middleware;
pub mod modelfile;
pub mod name;
pub mod openai;
pub mod options;
pub mod parsers;
pub mod progress;
pub mod prompt;
pub mod registry;
pub mod renderers;
pub mod routes;
pub mod sched;
pub mod template;
pub mod tools;
pub mod thinking;

pub use api::{
    Capability, ConfigV2, Message, ThinkLevel, ThinkValue, Tool, ToolCall, ToolCallArguments,
    ToolCallFunction, ToolFunction, ToolProperty,
};
pub use modelfile::{Command, Modelfile, ModelfileError};
pub use name::{Name, ParseNameFromFilepathError, MISSING_PART};
pub use options::{Options, OptionsError, Runner};
pub use template::{Template, Values};
