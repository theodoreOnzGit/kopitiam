//! # kopitiam-models -- the model acquisition layer
//!
//! KOPITIAM already got a working, pure-Rust inference stack
//! (`kopitiam-loader` / `-tokenizer` / `-runtime` / `-ai`). What it did NOT
//! have is a way to get the actual model weights (the `.gguf` files) onto
//! disk in the first place. That is this crate's one job: **acquire model
//! files, and prove they are the right bytes, before anyone tries to load
//! them.**
//!
//! It sits BELOW `kopitiam-ai`. It hands verified on-disk file paths up to
//! `kopitiam-loader`; it does not load or run anything itself.
//!
//! ## Autofetch-first, BYO-fallback
//!
//! Two ways a model can end up acquired, and this crate does both through the
//! same door ([`ensure_available`]):
//!
//! * **Autofetch** -- the file is missing, so we pull it over the network from
//!   the URL in the catalog, then verify it.
//! * **Bring-your-own (BYO)** -- you already dropped the correct `.gguf` into
//!   the store yourself (downloaded by hand, copied off a thumbdrive, whatever).
//!   Then [`ensure_available`] just verifies it and hands it back, touching NO
//!   network at all. This is the short-circuit: present-and-correct means we
//!   never even call the fetcher.
//!
//! "Autofetch-first" is about convenience -- it works out of the box (the `net`
//! feature is on by default). "BYO-fallback" is about staying true to
//! KOPITIAM's Offline First rule: turn `net` off and the whole crate still
//! builds and works, just BYO-only, with no HTTP stack compiled at all.
//!
//! ## The verification gate -- non-negotiable
//!
//! Bytes are NEVER trusted just because they landed. Every artifact, whether
//! freshly fetched or BYO, gets streamed through SHA-256 and checked against
//! the catalog's recorded [`Artifact::sha256`]. A mismatch is always fatal
//! ([`Error::ChecksumMismatch`]) -- acquisition will not hand back a path it
//! could not vouch for. This is what makes the returned [`AcquiredModel`]
//! actually mean something.
//!
//! > **Heads-up:** the shipped [`Catalog::builtin`] uses **placeholder**
//! > checksums (64 zeros), because real hashes cannot be computed without
//! > downloading the hundreds-of-MB weights at authoring time. So acquiring a
//! > *built-in* entry will fetch and then deliberately fail the gate until the
//! > real sha256 is recorded. Honesty over a catalog that lies -- see
//! > [`Catalog::builtin`] for the full story and the fix workflow.
//!
//! ## The `Fetcher` seam, and why it exists
//!
//! All network I/O goes through one trait, [`Fetcher`]. [`ensure_available`]
//! takes a `&dyn Fetcher`, so the whole acquire path can be driven in a test
//! with a fake that writes known local bytes and never opens a socket. That
//! keeps the core offline-testable, which is the same instinct you see all over
//! KOPITIAM: `kopitiam-ai`'s `ModelAdapter` is the single boundary a model is
//! reached through, and `kopitiam-loader` stops one step short of depending on
//! `kopitiam-tensor` so the two can settle independently. Here, the acquisition
//! core doesn't know or care *how* bytes arrive -- only that they arrive and
//! then verify.
//!
//! The one real fetcher, [`HttpFetcher`], lives behind the default-on `net`
//! feature and is built on `ureq` + `rustls`.
//!
//! ### The ring/rustls caveat
//!
//! Say it plainly, because it is easy to get wrong: "rustls" does NOT mean
//! "no C". rustls is a pure-Rust TLS *protocol* implementation, but ureq's
//! `rustls` feature picks the `ring` provider (C + perlasm) for the actual
//! crypto. `ring` is accepted on purpose -- it cross-compiles clean to the
//! targets KOPITIAM care about (including Android/aarch64, where OpenSSL
//! famously cannot), and it stays chope-d behind the off-by-default-able `net`
//! feature, so the BYO-only build never compile even one byte of it. Same
//! tradeoff `kopitiam-web` already made; see `docs/ai-decisions/AID-0013`.
//!
//! ## Quick tour
//!
//! ```no_run
//! use kopitiam_models::{Catalog, ModelStore, ensure_available};
//! # #[cfg(feature = "net")]
//! use kopitiam_models::HttpFetcher;
//!
//! # #[cfg(feature = "net")]
//! # fn demo() -> Result<(), kopitiam_models::Error> {
//! let spec = Catalog::find("qwen2.5-0.5b-instruct-q4_0")
//!     .expect("known id");
//! let store = ModelStore::with_default_root()?;      // ~/.cache/kopitiam/models
//! let acquired = ensure_available(&store, &spec, &HttpFetcher::new())?;
//! // `acquired.artifact_paths` are on-disk, verified, ready for the loader.
//! # let _ = acquired;
//! # Ok(())
//! # }
//! ```

mod catalog;
mod error;
mod fetch;
mod store;

pub use catalog::{Architecture, Artifact, Catalog, CatalogProblem, ModelSpec};
pub use error::Error;
pub use fetch::{ensure_available, AcquiredModel, Fetcher};
pub use store::ModelStore;

#[cfg(feature = "net")]
pub use fetch::HttpFetcher;
