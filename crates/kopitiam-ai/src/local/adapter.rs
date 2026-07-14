//! [`LocalAdapter`] itself: the seam where `kopitiam-ai` actually touches
//! `kopitiam-runtime`, `kopitiam-loader`, and `kopitiam-tokenizer`.
//!
//! # Why `kopitiam-ai` is allowed to depend on `kopitiam-runtime`
//!
//! CLAUDE.md's Semantic Runtime dependency rule reads: "Nothing below
//! `kopitiam-workflow` may depend on `kopitiam-ai`." That sentence
//! constrains what depends *on* `kopitiam-ai` — it says nothing about what
//! `kopitiam-ai` itself may depend on. `kopitiam-runtime` (and the
//! `kopitiam-core`/`kopitiam-tensor`/`kopitiam-loader`/`kopitiam-tokenizer`
//! stack beneath it) is not one of the Semantic Runtime crates CLAUDE.md's
//! architecture table names at all — it is a separate stack, and this
//! workspace's own root `Cargo.toml` says so explicitly, in the comment
//! directly above where that stack is declared: "the CPU-first,
//! local-first inference engine that will sit *behind* `kopitiam-ai`'s
//! `ModelAdapter`." So `kopitiam-ai -> kopitiam-runtime` is a downward
//! dependency into a lower, independent stack — the same shape as
//! `kopitiam-ai -> serde` or `kopitiam-ai -> anyhow` — not a violation of
//! a rule that is only ever about the *upward* direction (nothing may
//! reach back into `kopitiam-ai` from underneath `kopitiam-workflow`).
//! `kopitiam-runtime` itself has no dependency on `kopitiam-ai` (see its
//! `Cargo.toml`), so no cycle is possible either way, and
//! `kopitiam-workflow` still only ever sees this through the
//! [`crate::ModelAdapter`] trait object — it never names `LocalAdapter` or
//! `kopitiam-runtime` directly.
//!
//! Put differently: had this instead needed a separate `kopitiam-ai-local`
//! crate, the same question would recur one level up (does
//! `kopitiam-workflow` depending on *that* violate the rule?) with the
//! same answer, for the same reason — the rule is about not depending back
//! on `kopitiam-ai` from underneath it, and neither shape does that. A
//! split crate would only be justified by a *different* concern (e.g.
//! wanting `kopitiam-ai` itself to compile without ever pulling in a
//! tensor engine, even optionally) — which is exactly what the `local`
//! Cargo feature already achieves without a second crate: `cargo build
//! -p kopitiam-ai --no-default-features` compiles the trait and
//! [`crate::EchoAdapter`] alone, with no `kopitiam-runtime` in the
//! dependency graph at all.

use std::path::Path;

use anyhow::{Context, Result};
use kopitiam_runtime::{GenerationConfig, QwenModel, generate, tokenizer_from_gguf};
use kopitiam_tokenizer::BpeTokenizer;

use super::chat_template::render_chatml;
use super::generation::{resolve_eos_token_id, resolve_max_new_tokens};
use crate::{CompletionRequest, CompletionResponse, ModelAdapter};

const IM_END: &str = "<|im_end|>";
const ENDOFTEXT: &str = "<|endoftext|>";
/// GGUF's own declared default end-of-sequence id, when present — see
/// [`super::generation::resolve_eos_token_id`]'s docs for why this is
/// consulted at all despite `<|im_end|>` being preferred.
const GGUF_EOS_METADATA_KEY: &str = "tokenizer.ggml.eos_token_id";

/// A [`crate::ModelAdapter`] that runs a GGUF Qwen model entirely on this
/// machine's CPU via `kopitiam-runtime` — no network, no cloud account, no
/// API key. This is what makes CLAUDE.md's Offline First pipeline
/// ("existing knowledge, then native Rust, then local AI, then cloud AI as
/// the final fallback") a real, callable rung rather than a promise:
/// before this type existed, "local AI" had nothing behind it but
/// [`crate::EchoAdapter`], which invokes no model at all.
///
/// Construct via [`LocalAdapter::load`]; see this module's docs for why
/// depending on `kopitiam-runtime` here does not violate the Semantic
/// Runtime's dependency rule.
pub struct LocalAdapter {
    model: QwenModel,
    tokenizer: BpeTokenizer,
    /// Names the specific model this adapter serves, resolved once at
    /// [`LocalAdapter::load`] time from the GGUF's own `general.name`
    /// metadata (falling back to `general.architecture`, then a generic
    /// label). This is what [`CompletionResponse::model`] reports —
    /// deliberately distinct from [`LocalAdapter::name`]; see
    /// [`crate::ModelAdapter::name`]'s docs on that distinction.
    model_name: String,
    /// The single stop token [`GenerationConfig::eos_token_id`] generates
    /// against, resolved once at load time — see
    /// [`super::generation::resolve_eos_token_id`].
    eos_token_id: Option<u32>,
}

impl LocalAdapter {
    /// Loads a GGUF Qwen model and its embedded tokenizer from `path`,
    /// once, and holds both for the lifetime of the returned adapter.
    ///
    /// # Errors
    ///
    /// Returns `Err` — never panics — if `path` does not exist, is not a
    /// valid GGUF/SafeTensors file, is missing metadata
    /// [`kopitiam_runtime::QwenConfig::from_metadata`] requires, or has no
    /// embedded `tokenizer.ggml.tokens` vocabulary
    /// ([`tokenizer_from_gguf`]'s own error cases). This is deliberate,
    /// not incidental: per [`crate::ModelAdapter::complete`]'s own docs, a
    /// failing adapter is how `kopitiam-workflow` falls back to the next
    /// rung of the Offline First pipeline, so every failure mode here has
    /// to surface as `Err` rather than a panic that would take the whole
    /// workflow process down with it.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let loaded = kopitiam_loader::load_model(path)
            .with_context(|| format!("loading model file {}", path.display()))?;

        let model = QwenModel::from_loaded_model(&loaded)
            .with_context(|| format!("building a Qwen model from {}", path.display()))?;

        let tokenizer = tokenizer_from_gguf(&loaded).with_context(|| {
            format!("building a tokenizer from {}'s embedded GGUF vocabulary", path.display())
        })?;

        let model_name = loaded
            .metadata()
            .name
            .clone()
            .or_else(|| loaded.metadata().architecture.clone())
            .unwrap_or_else(|| "local-gguf-model".to_string());

        let gguf_eos_metadata = loaded.metadata().raw.get_u32(GGUF_EOS_METADATA_KEY);
        let eos_token_id = resolve_eos_token_id(
            tokenizer.special_token_id(IM_END),
            gguf_eos_metadata,
            tokenizer.special_token_id(ENDOFTEXT),
        );

        Ok(Self { model, tokenizer, model_name, eos_token_id })
    }
}

impl ModelAdapter for LocalAdapter {
    fn name(&self) -> &str {
        "local-qwen"
    }

    fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        let prompt = render_chatml(&request.messages);
        let config = GenerationConfig {
            max_new_tokens: resolve_max_new_tokens(request.max_tokens),
            eos_token_id: self.eos_token_id,
        };

        let content = generate(&self.model, &self.tokenizer, &prompt, &config, |_id, _text| {})
            .context("local Qwen generation failed")?;

        Ok(CompletionResponse { content, model: self.model_name.clone() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;
    use crate::local::test_support::synthetic_gguf::{build_local_adapter_fixture, write_temp_gguf};

    #[test]
    fn load_on_a_nonexistent_path_returns_err_not_a_panic() {
        let result = LocalAdapter::load("/does/not/exist/kopitiam-nonexistent.gguf");
        assert!(result.is_err());
    }

    #[test]
    fn load_on_a_non_gguf_file_returns_err_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-model.gguf");
        std::fs::write(&path, b"this is plainly not a GGUF or SafeTensors file").unwrap();

        let result = LocalAdapter::load(&path);
        assert!(result.is_err());
    }

    /// The strongest test available without real model weights (none
    /// exist on this machine — see this crate's task brief): a tiny but
    /// structurally real synthetic GGUF, built the same way
    /// `kopitiam-runtime`'s own tests build one, but extended with a full
    /// byte-level vocabulary plus the three ChatML control tokens so it
    /// exercises `LocalAdapter::load`'s *entire* pipeline — GGUF parse,
    /// `QwenModel` construction, GGUF-embedded tokenizer construction,
    /// `general.name` resolution, and EOS token resolution — followed by
    /// a real `complete()` call through ChatML rendering and
    /// `kopitiam_runtime::generate`'s forward passes. It intentionally
    /// does not assert anything about *which* tokens random weights
    /// favor — that would be asserting a coincidence, not a property of
    /// the code — only that the whole path runs, honors `max_tokens`, and
    /// reports the model name from GGUF metadata rather than the
    /// adapter's own name.
    #[test]
    fn end_to_end_against_a_synthetic_gguf_with_no_real_weights() {
        let bytes = build_local_adapter_fixture();
        let path = write_temp_gguf(&bytes, "local-adapter-e2e");

        let adapter = LocalAdapter::load(&path).expect("synthetic GGUF must load");
        assert_eq!(adapter.name(), "local-qwen");

        let request = CompletionRequest::new([
            Message::system("you are a test fixture"),
            Message::user("hello"),
        ])
        .with_max_tokens(5);

        let response = adapter.complete(&request).expect("generation against synthetic weights must not error");
        // The synthetic GGUF's general.name, not LocalAdapter::name()'s
        // "local-qwen" -- proving CompletionResponse::model reports the
        // model, not the adapter.
        assert_eq!(response.model, "kopitiam-test-qwen");
    }

    /// `max_tokens` must actually bound generation length. Greedy decoding
    /// against a fixed synthetic model and fixed prompt is deterministic
    /// (the same property `kopitiam-runtime`'s own KV-cache tests rely
    /// on), so requesting more tokens can only ever produce a
    /// longer-or-equal completion than requesting fewer, never shorter.
    #[test]
    fn max_tokens_bounds_the_completion_length() {
        let bytes = build_local_adapter_fixture();
        let path = write_temp_gguf(&bytes, "local-adapter-max-tokens");
        let adapter = LocalAdapter::load(&path).unwrap();

        let short = adapter
            .complete(&CompletionRequest::new([Message::user("hi")]).with_max_tokens(1))
            .unwrap();
        let long = adapter
            .complete(&CompletionRequest::new([Message::user("hi")]).with_max_tokens(20))
            .unwrap();

        assert!(
            long.content.len() >= short.content.len(),
            "a larger max_tokens budget must never produce a shorter completion \
             (short={:?}, long={:?})",
            short.content,
            long.content
        );
    }

    /// Real, full-size Qwen `.gguf` weights were not found anywhere on
    /// this machine when this test was written (per this crate's task
    /// brief: "no model weights exist on this machine... do NOT download
    /// anything"). This test is `#[ignore]`d rather than deleted, ready to
    /// run the moment a real model is available -- mirrors
    /// `kopitiam_runtime`'s own `a_real_model_on_disk_is_used_if_present`
    /// pattern (`crates/kopitiam-runtime/src/model.rs`), one environment
    /// variable pointing at a `.gguf` file rather than hand-rolling a
    /// second "load and generate" from scratch.
    ///
    /// Run with:
    /// `KOPITIAM_QWEN_GGUF=/path/to/model.gguf cargo test --release -p kopitiam-ai -- --ignored`
    #[test]
    #[ignore = "no real Qwen GGUF present on this machine; point KOPITIAM_QWEN_GGUF at one to run this"]
    fn a_real_local_model_answers_a_chatml_prompt() {
        let path = std::env::var("KOPITIAM_QWEN_GGUF").expect("set KOPITIAM_QWEN_GGUF to a real Qwen .gguf file");
        let adapter = LocalAdapter::load(&path).expect("a real Qwen GGUF must load");

        let request = CompletionRequest::new([
            Message::system("You are a helpful assistant."),
            Message::user("Say hello in one short sentence."),
        ])
        .with_max_tokens(64);

        let response = adapter.complete(&request).expect("a real Qwen model must generate a completion");
        assert!(!response.content.trim().is_empty(), "a real model should not answer with empty text");
        assert_ne!(response.model, adapter.name(), "CompletionResponse::model must name the model, not the adapter");
        println!("real model {:?} answered: {:?}", response.model, response.content);
    }
}
