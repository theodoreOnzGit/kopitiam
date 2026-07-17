//! # kopitiam-internet-research
//!
//! Ah, this crate is the **online-research arm** of KOPITIAM's general knowledge
//! engine. When the answer cannot be found from existing knowledge, native Rust,
//! or the local corpus, this is where KOPITIAM go reach out to the open web to
//! gather knowledge — across *any* domain (science, lifestyle, finance, and so
//! on), not one specialised field.
//!
//! Sits above [`kopitiam-web`](https://crates.io/crates/kopitiam-web)'s low-level
//! search adapters: `kopitiam-web` is the plumbing (the search-engine calls, the
//! last-resort fetch in Offline-First), while this crate is meant to be the
//! higher-level *research* layer (frame a question, gather + rank sources,
//! hand back structured findings). That split is not built yet — see the bead.
//!
//! **Status: scaffold only.** Renamed from the old `kopitiam-scientific` stub
//! (an empty `cargo new`), because KOPITIAM is a general knowledge engine for any
//! domain, not a science-specific tool. Real implementation is future work.

// Intentionally empty for now — this is a name-reserving scaffold. The research
// API lands when the internet-research design is worked out (tracked in beads).
