//! Cloud model adapters — **scaffold only** (`temp_ai_design.md` §10.3).
//!
//! CLAUDE.md's Offline First pipeline puts cloud AI **last**: "existing
//! knowledge, then native Rust, then local AI, then cloud AI as the final
//! fallback." This module is where that final rung lives — the adapters for
//! Claude / GPT / Gemini, each gated on an API key the user supplies through
//! the environment.
//!
//! # What's built here now, and what's deliberately not
//!
//! **Built:** the [`CloudAdapter`] trait (the vendor-specific seam — which
//! env var holds the key, what the wire name is), the [`CloudVendor`] enum,
//! and [`CloudStub`], a concrete [`crate::ModelAdapter`] that does the
//! **key-detection and availability** half completely and honestly:
//!
//! * **no key in the environment → [`crate::ModelAdapter::complete`] returns
//!   `Err`** carrying [`CloudUnavailable::NoApiKey`]. That is the "no key →
//!   Unavailable" path the design calls for: the dispatcher above sees an
//!   `Err`, treats the rung as absent, and falls through — no network touched.
//! * **key present → `Err` carrying [`CloudUnavailable::NotYetImplemented`]**,
//!   because the actual HTTP call isn't wired yet.
//!
//! **Deliberately NOT built:** the real network layer (ring/rustls per
//! AID-0013, request/response JSON per vendor, SSE token streaming). That is
//! a **follow-up bead** — see this crate's `stream` module and the task
//! notes. Scaffolding the trait + key detection first means the dispatch
//! ladder and the CLI can already *name* and *probe* a cloud rung, and wiring
//! the HTTP in later is a localised change behind [`CloudStub::send`] with no
//! churn to callers.
//!
//! # Why this compiles with zero extra dependencies
//!
//! Nothing here needs a tensor engine or an HTTP client — it's `std::env`
//! and error types only. So, unlike [`crate::LocalAdapter`] (behind the
//! `local` feature so a consumer can skip the runtime), the cloud scaffold is
//! **always compiled**: it adds no weight. The HTTP follow-up will introduce
//! its network deps behind their own feature at that time.

use std::fmt;

use anyhow::Result;

use crate::{CompletionRequest, CompletionResponse, ModelAdapter};

/// Which cloud vendor a [`CloudAdapter`] talks to. Each knows the environment
/// variable its API key lives in, a human-facing name, and a sensible default
/// model id — the minimum a dispatcher needs to *probe* the rung before any
/// network layer exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CloudVendor {
    /// Anthropic's Claude family. Key: `ANTHROPIC_API_KEY`.
    Claude,
    /// OpenAI's GPT family. Key: `OPENAI_API_KEY`.
    Gpt,
    /// Google's Gemini family. Key: `GEMINI_API_KEY`.
    Gemini,
}

impl CloudVendor {
    /// The environment variable this vendor's API key is read from. These are
    /// the vendors' own conventional names, so a user who already exported a
    /// key for another tool needs no extra setup.
    pub fn api_key_env(self) -> &'static str {
        match self {
            CloudVendor::Claude => "ANTHROPIC_API_KEY",
            CloudVendor::Gpt => "OPENAI_API_KEY",
            CloudVendor::Gemini => "GEMINI_API_KEY",
        }
    }

    /// A short human-facing name for notices and logs (`"Claude"`, `"GPT"`,
    /// `"Gemini"`).
    pub fn display_name(self) -> &'static str {
        match self {
            CloudVendor::Claude => "Claude",
            CloudVendor::Gpt => "GPT",
            CloudVendor::Gemini => "Gemini",
        }
    }

    /// The model id used when the caller doesn't pin one. Recorded here so
    /// the eventual HTTP layer has a default to send; not load-bearing while
    /// this is a scaffold. Deliberately a placeholder-grade default per
    /// vendor — the real, pinned model ids land with the HTTP follow-up.
    pub fn default_model(self) -> &'static str {
        match self {
            CloudVendor::Claude => "claude-sonnet-latest",
            CloudVendor::Gpt => "gpt-latest",
            CloudVendor::Gemini => "gemini-latest",
        }
    }
}

/// The vendor-specific seam of a cloud model adapter.
///
/// This is intentionally thin: it captures *only* what differs between
/// vendors that we can implement **without** a network layer — which
/// [`CloudVendor`] this is, hence which env var the key comes from and
/// whether that key is present. The uniform request/response surface stays on
/// [`ModelAdapter`] (every cloud adapter is also a `ModelAdapter`, expressed
/// by the `ModelAdapter` supertrait bound), so the dispatch ladder and the
/// CLI treat a cloud adapter exactly like any other.
///
/// When the HTTP follow-up lands, the per-vendor request shaping and response
/// parsing go behind a method here (or a sibling trait), keeping the wire
/// details out of the shared code.
pub trait CloudAdapter: ModelAdapter {
    /// Which vendor this adapter speaks to.
    fn vendor(&self) -> CloudVendor;

    /// The API key for this vendor, read from its environment variable
    /// ([`CloudVendor::api_key_env`]). `None` when the variable is unset or
    /// empty — an empty key is treated as no key, since it can never
    /// authenticate.
    ///
    /// Provided (not required) so every vendor gets identical, correct
    /// key-detection for free; an implementation should not need to override
    /// it.
    fn api_key(&self) -> Option<String> {
        std::env::var(self.vendor().api_key_env()).ok().filter(|key| !key.is_empty())
    }

    /// Whether this rung is usable *right now*: is there a key? This is the
    /// cheap probe the dispatch ladder (and the CLI) calls before bothering
    /// to route anything to cloud. It does **not** hit the network — a
    /// present key means "worth trying", not "known-good".
    fn is_available(&self) -> bool {
        self.api_key().is_some()
    }
}

/// Why a cloud rung couldn't answer. Both variants are the *honest miss*
/// signal the design's dispatch ladder relies on (`temp_ai_design.md` §2):
/// the provider reports it can't cover the task instead of bluffing, and the
/// dispatcher falls through.
///
/// Implements [`std::error::Error`] so [`CloudStub`] can surface it straight
/// through `anyhow` — a caller streaming the adapter sees it rendered into a
/// [`crate::StreamChunk::Error`] by the default eager `stream` path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloudUnavailable {
    /// No API key found in the environment — the rung is simply not
    /// configured on this machine. Carries the vendor and the exact env var
    /// to set, so a notice can tell the user how to enable it.
    NoApiKey {
        /// The vendor whose key is missing.
        vendor: CloudVendor,
        /// The environment variable to set (e.g. `ANTHROPIC_API_KEY`).
        env_var: &'static str,
    },
    /// A key *is* present, but the real network path isn't built yet — this
    /// is the scaffold being honest that it can't actually reach the vendor.
    /// Removed when the HTTP follow-up bead lands.
    NotYetImplemented {
        /// The vendor that would have been called.
        vendor: CloudVendor,
    },
}

impl fmt::Display for CloudUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CloudUnavailable::NoApiKey { vendor, env_var } => write!(
                f,
                "cloud {} unavailable: no API key — set {} to enable it",
                vendor.display_name(),
                env_var,
            ),
            CloudUnavailable::NotYetImplemented { vendor } => write!(
                f,
                "cloud {} has a key but its HTTP layer isn't built yet \
                 (scaffold only — real network is a follow-up)",
                vendor.display_name(),
            ),
        }
    }
}

impl std::error::Error for CloudUnavailable {}

/// A scaffold [`CloudAdapter`] for one [`CloudVendor`]: does key-detection
/// for real, and stands in for the not-yet-built network layer by returning
/// a clear [`CloudUnavailable`] either way.
///
/// This exists so the dispatch ladder and the CLI can already construct,
/// name, and probe a cloud rung (`is_available()`), and so the "no key →
/// Unavailable" contract is testable today — well before any HTTP is wired.
/// Swapping the stub body of [`CloudStub::send`] for a real request is the
/// whole of the follow-up; nothing outside this file changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CloudStub {
    vendor: CloudVendor,
}

impl CloudStub {
    /// A stub adapter for `vendor`.
    pub fn new(vendor: CloudVendor) -> Self {
        Self { vendor }
    }

    /// Shorthand for the Claude stub.
    pub fn claude() -> Self {
        Self::new(CloudVendor::Claude)
    }

    /// Shorthand for the GPT stub.
    pub fn gpt() -> Self {
        Self::new(CloudVendor::Gpt)
    }

    /// Shorthand for the Gemini stub.
    pub fn gemini() -> Self {
        Self::new(CloudVendor::Gemini)
    }

    /// The single seam the HTTP follow-up replaces. Today it decides the
    /// outcome purely from key presence:
    ///
    /// * no key → `Err(CloudUnavailable::NoApiKey)` (the "no key →
    ///   Unavailable" path);
    /// * key present → `Err(CloudUnavailable::NotYetImplemented)`.
    ///
    /// When the network lands, the key-present branch becomes the real
    /// request/response against the vendor, and this returns
    /// `Ok(CompletionResponse)`. Kept as one method so that change is
    /// localised.
    fn send(&self, _request: &CompletionRequest) -> Result<CompletionResponse, CloudUnavailable> {
        match self.api_key() {
            None => Err(CloudUnavailable::NoApiKey {
                vendor: self.vendor,
                env_var: self.vendor.api_key_env(),
            }),
            Some(_key) => Err(CloudUnavailable::NotYetImplemented { vendor: self.vendor }),
        }
    }
}

impl ModelAdapter for CloudStub {
    fn name(&self) -> &str {
        // Stable per-vendor adapter id for logs/provenance, distinct from the
        // model that answers (CompletionResponse::model) once HTTP exists.
        match self.vendor {
            CloudVendor::Claude => "cloud-claude",
            CloudVendor::Gpt => "cloud-gpt",
            CloudVendor::Gemini => "cloud-gemini",
        }
    }

    /// Returns the [`CloudUnavailable`] from [`CloudStub::send`] as an
    /// `anyhow::Error`. The dispatch ladder reads any `Err` here as "this
    /// rung can't answer" and falls through — exactly the honest-miss
    /// routing signal (`temp_ai_design.md` §2).
    fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        Ok(self.send(request)?)
    }

    // `stream` intentionally uses the trait's eager default: with no HTTP
    // layer there is nothing to stream incrementally, and the default turns
    // `complete`'s `Err` into a single `StreamChunk::Error` — the right
    // shape for a caller. The SSE token-streaming override lands with the
    // HTTP follow-up.
}

impl CloudAdapter for CloudStub {
    fn vendor(&self) -> CloudVendor {
        self.vendor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, StreamChunk};
    use std::sync::Mutex;

    /// Serialises the env-mutating tests: the process environment is global,
    /// and these set/remove a real key var. Cargo runs tests in parallel by
    /// default, so without this two tests could stomp each other's key state.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn vendor_env_vars_are_the_conventional_names() {
        assert_eq!(CloudVendor::Claude.api_key_env(), "ANTHROPIC_API_KEY");
        assert_eq!(CloudVendor::Gpt.api_key_env(), "OPENAI_API_KEY");
        assert_eq!(CloudVendor::Gemini.api_key_env(), "GEMINI_API_KEY");
    }

    /// With the key var cleared, the stub is unavailable and `complete`
    /// returns the NoApiKey miss — the "no key → Unavailable" contract.
    #[test]
    fn no_key_means_unavailable_and_complete_errs_with_nokey() {
        let _guard = ENV_LOCK.lock().unwrap();
        let stub = CloudStub::claude();
        // SAFETY: single-threaded within ENV_LOCK; we only clear the var.
        unsafe {
            std::env::remove_var(CloudVendor::Claude.api_key_env());
        }

        assert!(!stub.is_available());
        let err = stub.complete(&CompletionRequest::new([Message::user("hi")])).unwrap_err();
        let cloud = err.downcast_ref::<CloudUnavailable>().expect("a CloudUnavailable");
        assert!(matches!(cloud, CloudUnavailable::NoApiKey { vendor: CloudVendor::Claude, .. }));
    }

    /// The no-key `complete` error, streamed, becomes exactly one
    /// `StreamChunk::Error` (via the eager default) — nothing else.
    #[test]
    fn no_key_stream_is_a_single_error_chunk() {
        let _guard = ENV_LOCK.lock().unwrap();
        let stub = CloudStub::gpt();
        // SAFETY: single-threaded within ENV_LOCK.
        unsafe {
            std::env::remove_var(CloudVendor::Gpt.api_key_env());
        }

        let chunks: Vec<StreamChunk> =
            stub.stream(&CompletionRequest::new([Message::user("hi")])).iter().collect();
        assert_eq!(chunks.len(), 1);
        assert!(matches!(&chunks[0], StreamChunk::Error(msg) if msg.contains("no API key")));
    }

    /// With a key present, the stub is available but honestly reports the
    /// network layer isn't built — proving key-detection flips both the
    /// availability probe and the miss reason.
    #[test]
    fn key_present_is_available_but_not_yet_implemented() {
        let _guard = ENV_LOCK.lock().unwrap();
        let stub = CloudStub::gemini();
        // SAFETY: single-threaded within ENV_LOCK; restored below.
        unsafe {
            std::env::set_var(CloudVendor::Gemini.api_key_env(), "sk-test-not-real");
        }

        assert!(stub.is_available());
        let err = stub.complete(&CompletionRequest::new([Message::user("hi")])).unwrap_err();
        let cloud = err.downcast_ref::<CloudUnavailable>().expect("a CloudUnavailable");
        assert!(matches!(cloud, CloudUnavailable::NotYetImplemented { vendor: CloudVendor::Gemini }));

        // SAFETY: single-threaded within ENV_LOCK; leave the env as we found it.
        unsafe {
            std::env::remove_var(CloudVendor::Gemini.api_key_env());
        }
    }
}
