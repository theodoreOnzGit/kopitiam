//! Chat renderers -- the exact prompt string each model family was trained on.
//!
//! **Upstream:** `model/renderers/` (ollama, MIT). This file is
//! `model/renderers/renderer.go`; each sibling file names its own Go source in
//! its header.
//!
//! ## Why byte-exactness is the whole point
//!
//! A renderer turns `[Message]` + `[Tool]` into one flat string. Sounds boring
//! lah -- but every model family was **fine-tuned on a specific framing**:
//! Qwen wraps turns in `<|im_start|>role\n ... <|im_end|>`, Gemma uses
//! `<start_of_turn>`, GLM uses `<|user|>` / `<|assistant|>`, DeepSeek uses
//! full-width `<｜User｜>`. Feed a model the wrong framing and **it does not
//! error** -- it just quietly gets worse: weaker instruction-following, tool
//! calls in the wrong syntax, thinking that never closes.
//!
//! That is why the acceptance criterion here is "byte-identical to upstream's
//! fixtures", not "looks about right". The tests in these files are ported
//! straight from ollama's `*_test.go` and they *are* the specification.
//!
//! ## Renderer vs. template -- two different roads
//!
//! [`crate::template`] runs a Go `text/template` that came out of the model's
//! own GGUF. A **renderer** is code, chosen by name from the model's
//! `ConfigV2.renderer` field ([`crate::api::ConfigV2`]). Upstream reaches for a
//! renderer when the framing is too gnarly for a template to express
//! honestly -- interleaved thinking, tool-call state machines, image
//! placeholders that must be numbered across a whole conversation.
//!
//! ## The registry
//!
//! [`register`] lets a caller add or **override** a renderer by name, and
//! [`renderer_for_name`] falls back to the built-in `match` when nothing is
//! registered. Same shape as upstream's `rendererForName`, and the override
//! order matters: registered first, built-in second.
//!
//! ## Ported so far
//!
//! | Renderer name(s) | Rust type | Go file |
//! |---|---|---|
//! | `qwen3-coder` | [`Qwen3CoderRenderer`] | `qwen3coder.go` |
//! | `qwen3.5` | [`Qwen35Renderer`] | `qwen35.go` |
//! | `ornith` | [`Qwen35Renderer`] (preset) | `ornith.go` |
//! | `qwen3-vl-instruct`, `qwen3-vl-thinking` | [`Qwen3VLRenderer`] | `qwen3vl.go` |
//! | `deepseek3.1` | [`DeepSeek3Renderer`] | `deepseek3.go` |
//! | `cogito` | [`CogitoRenderer`] | `cogito.go` |
//! | `glm-4.7` | [`Glm47Renderer`] | `glm47.go` |
//! | `glm-ocr` | [`GlmOcrRenderer`] | `glmocr.go` |
//! | (unnamed upstream) | [`Glm46Renderer`] | `glm46.go` |
//! | `olmo3`, `olmo3.1` | [`Olmo3Renderer`] | `olmo3.go` |
//! | `olmo3-think`, `olmo3-32b-think` | [`Olmo3ThinkRenderer`] | `olmo3_think.go` |
//! | `cohere` | [`CohereRenderer`] | `cohere.go` |
//! | `lfm2`, `lfm2-thinking` | [`Lfm2Renderer`] | `lfm2.go` |
//! | `gemma4`, `gemma4-small`, `gemma4-large` | [`Gemma4Renderer`] | `gemma4.go` |
//! | `functiongemma` | [`FunctionGemmaRenderer`] | `functiongemma.go` |
//!
//! Not yet reached (still upstream-only): `nemotron-3-nano`, `laguna` /
//! `poolside-v1`. Asking for one of those names returns
//! [`RenderError::UnknownRenderer`] -- a loud failure, never a silently wrong
//! prompt.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{OnceLock, RwLock};

use crate::api::{Message, ThinkValue, Tool};

mod cogito;
mod cohere;
mod deepseek3;
mod functiongemma;
mod gemma4;
mod glm46;
mod glm47;
mod glmocr;
mod image_tags;
mod json;
mod lfm2;
mod olmo3;
mod olmo3_think;
mod qwen35;
mod qwen3coder;
mod qwen3vl;

pub use cogito::CogitoRenderer;
pub use cohere::CohereRenderer;
pub use deepseek3::{DeepSeek3Renderer, DeepSeek3Variant};
pub use functiongemma::FunctionGemmaRenderer;
pub use gemma4::Gemma4Renderer;
pub use glm46::Glm46Renderer;
pub use glm47::Glm47Renderer;
pub use glmocr::GlmOcrRenderer;
pub use lfm2::Lfm2Renderer;
pub use olmo3::Olmo3Renderer;
pub use olmo3_think::{Olmo3ThinkRenderer, Olmo3ThinkVariant};
pub use qwen3coder::Qwen3CoderRenderer;
pub use qwen3vl::Qwen3VLRenderer;
pub use qwen35::Qwen35Renderer;

/// The ChatML turn delimiters. Qwen, Olmo and friends all share them.
///
/// **Upstream:** `qwen3coder.go` declares them once (`imStartTag`, `imEndTag`)
/// and the rest of the package borrows them, so they live here for the same
/// reason.
pub(crate) const IM_START_TAG: &str = "<|im_start|>";
/// See [`IM_START_TAG`].
pub(crate) const IM_END_TAG: &str = "<|im_end|>";

/// What can go wrong on the way to a prompt.
///
/// Upstream returns a bare `error`; we name the cases, because "unknown
/// renderer" is a configuration mistake the caller can fix and the others are
/// not.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RenderError {
    /// No renderer registered and no built-in match. **Upstream:**
    /// `fmt.Errorf("unknown renderer %q", name)`.
    ///
    /// Deliberately an error and never a fallback to some "generic" framing:
    /// guessing here produces a prompt that looks fine and makes the model
    /// worse, which is far harder to debug than a failed request.
    #[error("unknown renderer {0:?}")]
    UnknownRenderer(String),
}

/// One model family's prompt framing. **Upstream:** the `Renderer` interface.
pub trait Renderer: Send + Sync {
    /// Build the prompt.
    ///
    /// `think` is three-state on purpose (see [`ThinkValue`]): `None` means the
    /// caller never mentioned thinking, so the renderer uses its own default.
    /// That is *not* the same as `Some(Bool(false))`, and several renderers
    /// branch on the difference.
    fn render(
        &self,
        messages: &[Message],
        tools: &[Tool],
        think: Option<&ThinkValue>,
    ) -> Result<String, RenderError>;

    /// The beginning-of-sequence token this family expects in front of
    /// everything, if any. **Upstream:** `LeadingBOS()`.
    ///
    /// Why it is separate from [`Renderer::render`]: the tokenizer may add BOS
    /// itself, so the caller needs to *ask* what the BOS is and decide, rather
    /// than have it baked into the string and end up with two.
    fn leading_bos(&self) -> &'static str {
        ""
    }
}

/// Build a renderer on demand. **Upstream:** `RendererConstructor`.
pub type RendererConstructor = Box<dyn Fn() -> Box<dyn Renderer> + Send + Sync>;

fn registry() -> &'static RwLock<HashMap<String, RendererConstructor>> {
    static REGISTRY: OnceLock<RwLock<HashMap<String, RendererConstructor>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Global switch: emit `[img-N]` markers instead of a family's native image
/// tokens. **Upstream:** the package-level `var RenderImgTags bool`, set by
/// ollama's server package on init and left `false` everywhere else.
///
/// An `AtomicBool` rather than Go's plain `bool` because Rust will not let us
/// pretend a process-global mutable is race-free. Set it once at startup;
/// flipping it mid-flight would change the framing between two turns of the
/// same conversation, which is exactly the sort of quiet wrongness this module
/// exists to prevent.
static RENDER_IMG_TAGS: AtomicBool = AtomicBool::new(false);

/// Turn `[img-N]` markers on or off globally. See [`RENDER_IMG_TAGS`].
pub fn set_render_img_tags(on: bool) {
    RENDER_IMG_TAGS.store(on, Ordering::Relaxed);
}

/// Are `[img-N]` markers on? See [`RENDER_IMG_TAGS`].
pub fn render_img_tags() -> bool {
    RENDER_IMG_TAGS.load(Ordering::Relaxed)
}

/// Add (or replace) a renderer under `name`. **Upstream:** `Register`.
///
/// Registering over a built-in name wins -- that is upstream's behaviour and
/// upstream's test (`TestOverrideBuiltInRenderer`) pins it.
pub fn register(name: impl Into<String>, constructor: RendererConstructor) {
    if let Ok(mut reg) = registry().write() {
        reg.insert(name.into(), constructor);
    }
}

/// Look up a renderer: registry first, built-ins second.
///
/// **Upstream:** `rendererForName`. Returns `None` for a name nobody knows.
pub fn renderer_for_name(name: &str) -> Option<Box<dyn Renderer>> {
    if let Ok(reg) = registry().read()
        && let Some(ctor) = reg.get(name)
    {
        return Some(ctor());
    }
    built_in_renderer(name)
}

/// The `switch name` half of upstream's `rendererForName`.
///
/// Each arm's flag combination is upstream's, not ours -- e.g. `qwen3.5` gets
/// `emit_empty_think_on_no_think: true` while a bare `Qwen35Renderer` does not,
/// and `ornith` is the same renderer again with
/// `always_render_assistant_think_block` on top.
fn built_in_renderer(name: &str) -> Option<Box<dyn Renderer>> {
    let img = render_img_tags();
    Some(match name {
        "qwen3-coder" => Box::new(Qwen3CoderRenderer),
        "qwen3-vl-instruct" => Box::new(Qwen3VLRenderer {
            is_thinking: false,
            use_img_tags: img,
        }),
        "qwen3-vl-thinking" => Box::new(Qwen3VLRenderer {
            is_thinking: true,
            use_img_tags: img,
        }),
        "qwen3.5" => Box::new(Qwen35Renderer {
            is_thinking: true,
            emit_empty_think_on_no_think: true,
            use_img_tags: img,
            ..Default::default()
        }),
        // `ornith.go` -- literally a Qwen35Renderer with a preset, so it is a
        // preset here too rather than a newtype that forwards every method.
        "ornith" => Box::new(Qwen35Renderer {
            is_thinking: true,
            always_render_assistant_think_block: true,
            emit_empty_think_on_no_think: true,
            use_img_tags: img,
        }),
        "cogito" => Box::new(CogitoRenderer { is_thinking: true }),
        "deepseek3.1" => Box::new(DeepSeek3Renderer {
            is_thinking: true,
            variant: DeepSeek3Variant::DeepSeek31,
        }),
        "olmo3" => Box::new(Olmo3Renderer {
            use_extended_system_message: false,
        }),
        "olmo3.1" => Box::new(Olmo3Renderer {
            use_extended_system_message: true,
        }),
        // Used for Olmo-3-7B-Think and Olmo-3.1-32B-Think (same template).
        "olmo3-think" => Box::new(Olmo3ThinkRenderer {
            variant: Olmo3ThinkVariant::Olmo31Think,
        }),
        // Used for Olmo-3-32B-Think.
        "olmo3-32b-think" => Box::new(Olmo3ThinkRenderer {
            variant: Olmo3ThinkVariant::Olmo3Think32B,
        }),
        "gemma4" | "gemma4-small" => Box::new(Gemma4Renderer {
            use_img_tags: img,
            empty_block_on_nothink: false,
        }),
        "gemma4-large" => Box::new(Gemma4Renderer {
            use_img_tags: img,
            empty_block_on_nothink: true,
        }),
        "functiongemma" => Box::new(FunctionGemmaRenderer),
        "glm-4.7" => Box::new(Glm47Renderer),
        "glm-ocr" => Box::new(GlmOcrRenderer { use_img_tags: img }),
        "lfm2" => Box::new(Lfm2Renderer {
            is_thinking: false,
            use_img_tags: img,
        }),
        "lfm2-thinking" => Box::new(Lfm2Renderer {
            is_thinking: true,
            use_img_tags: img,
        }),
        "cohere" => Box::new(CohereRenderer),
        _ => return None,
    })
}

/// Render with the named renderer. **Upstream:** `RenderWithRenderer`.
pub fn render_with_renderer(
    name: &str,
    messages: &[Message],
    tools: &[Tool],
    think: Option<&ThinkValue>,
) -> Result<String, RenderError> {
    match renderer_for_name(name) {
        Some(r) => r.render(messages, tools, think),
        None => Err(RenderError::UnknownRenderer(name.to_string())),
    }
}

/// The BOS token for a named renderer, or `""` when the name is unknown.
///
/// **Upstream:** `LeadingBOSForRenderer` -- note it swallows the unknown-name
/// case and returns `""`, unlike [`render_with_renderer`], which errors. Kept
/// as-is: an empty BOS is harmless, a wrong prompt is not.
pub fn leading_bos_for_renderer(name: &str) -> String {
    renderer_for_name(name)
        .map(|r| r.leading_bos().to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Rust runs tests in parallel; Go runs them sequentially within a package.
    /// The registry is process-global, so the tests that mutate it have to take
    /// turns or they clobber each other. This is scaffolding for that, nothing
    /// more -- the renderers themselves hold no shared state.
    static REGISTRY_LOCK: Mutex<()> = Mutex::new(());

    fn forget(name: &str) {
        if let Ok(mut reg) = registry().write() {
            reg.remove(name);
        }
    }

    struct MockRenderer;

    impl Renderer for MockRenderer {
        fn render(
            &self,
            _: &[Message],
            _: &[Tool],
            _: Option<&ThinkValue>,
        ) -> Result<String, RenderError> {
            Ok("mock-output".to_string())
        }
    }

    /// Upstream `TestRegisterCustomRenderer`.
    #[test]
    fn a_custom_renderer_can_be_registered_and_used() {
        let _g = REGISTRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        register("custom-renderer", Box::new(|| Box::new(MockRenderer)));
        assert_eq!(
            render_with_renderer("custom-renderer", &[], &[], None).unwrap(),
            "mock-output"
        );
        forget("custom-renderer");
    }

    /// Upstream `TestOverrideBuiltInRenderer` -- a registered name beats the
    /// built-in `match`.
    #[test]
    fn a_registered_name_overrides_the_built_in() {
        let _g = REGISTRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        register("qwen3-coder", Box::new(|| Box::new(MockRenderer)));
        assert_eq!(
            render_with_renderer("qwen3-coder", &[], &[], None).unwrap(),
            "mock-output"
        );
        forget("qwen3-coder");
        // ...and once forgotten, the real one is back.
        assert_ne!(
            render_with_renderer("qwen3-coder", &[], &[], None).unwrap(),
            "mock-output"
        );
    }

    /// Upstream `TestBuiltInRendererStillWorks`.
    #[test]
    fn the_built_in_renderers_produce_something() {
        let _g = REGISTRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let messages = [Message::new("user", "Hello")];
        for name in ["qwen3-coder", "qwen3.5"] {
            let got = render_with_renderer(name, &messages, &[], None).unwrap();
            assert!(!got.is_empty(), "{name} rendered nothing");
        }
    }

    /// Upstream `TestUnknownRendererReturnsError`.
    #[test]
    fn an_unknown_renderer_is_an_error_not_a_guess() {
        assert_eq!(
            render_with_renderer("nonexistent-renderer", &[], &[], None),
            Err(RenderError::UnknownRenderer("nonexistent-renderer".into()))
        );
    }

    /// Upstream `TestLeadingBOSForRenderer`, minus the families not yet ported.
    /// The BOS strings are full-width / unusual on purpose -- copy them, never
    /// retype them.
    #[test]
    fn leading_bos_matches_upstream_per_renderer() {
        let _g = REGISTRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let cases = [
            ("gemma4", "<bos>"),
            ("functiongemma", "<bos>"),
            ("gemma4-small", "<bos>"),
            ("gemma4-large", "<bos>"),
            ("lfm2", "<|startoftext|>"),
            ("lfm2-thinking", "<|startoftext|>"),
            ("deepseek3.1", "<｜begin▁of▁sentence｜>"),
            ("cogito", "<｜begin▁of▁sentence｜>"),
            ("qwen3-coder", ""),
            // Unknown names get "" rather than an error -- upstream's choice.
            ("unknown", ""),
        ];
        for (name, want) in cases {
            assert_eq!(leading_bos_for_renderer(name), want, "{name}");
        }
    }
}
