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
//! | `nemotron-3-nano` | [`Nemotron3NanoRenderer`] | `nemotron3nano.go` |
//! | `laguna` | [`LagunaRenderer`] | `laguna.go` |
//! | `poolside-v1` | [`LagunaV8Renderer`] | `laguna.go` |
//!
//! **The renderer side of the port is now complete** -- every name upstream's
//! `rendererForName` dispatches, we dispatch.
//! `tests::the_renderer_registry_is_at_parity_with_upstream` pins that in both
//! directions: drop a name and it fails, invent one upstream does not have and
//! it fails too.
//!
//! Any *other* name still returns [`RenderError::UnknownRenderer`] -- a loud
//! failure, never a silently wrong prompt.
//!
//! ## Renderers vs. parsers -- two lists, and they do NOT match
//!
//! A model manifest carries **two independent** names: `ConfigV2.renderer`
//! (prompt in) and `ConfigV2.parser` (text out). They are not one namespace, and
//! **upstream itself does not keep them in sync** -- see
//! `tests::the_renderer_and_parser_name_lists_agree_where_upstream_says_they_should`
//! for the full accounting and the current gap list. The short version:
//!
//! * Upstream registers **23** renderer names and **25** parser names, sharing
//!   only **18**. `deepseek3.1` renders but parses as `deepseek3`;
//!   `gemma4-large` / `gemma4-small` render but parse as `gemma4` /
//!   `gemma4-no-thinking`; `harmony`, `ministral` and `passthrough` parse but
//!   never render. Those skews are upstream's design, not drift.
//! * **Our port's gap is different and real**: ten families we can render, we
//!   cannot yet parse. [`crate::parsers`] is being filled in separately; the test
//!   is written so the gap *shrinking* passes and the gap *growing* fails.

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
mod laguna;
mod lfm2;
mod nemotron3nano;
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
pub use laguna::{LagunaRenderer, LagunaV8Renderer};
pub use lfm2::Lfm2Renderer;
pub use nemotron3nano::Nemotron3NanoRenderer;
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
        "nemotron-3-nano" => Box::new(Nemotron3NanoRenderer),
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
        // Two names, two *different* renderers -- v2 and v8 of the same family.
        // See `laguna.rs`: they share a vocabulary and disagree on every newline.
        "laguna" => Box::new(LagunaRenderer),
        "poolside-v1" => Box::new(LagunaV8Renderer),
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

    /// Upstream `TestLeadingBOSForRenderer`, every case.
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
            ("laguna", "〈|EOS|〉"),
            ("poolside-v1", "〈|EOS|〉"),
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

    // -----------------------------------------------------------------------
    // Renderer / parser name reconciliation
    // -----------------------------------------------------------------------
    //
    // Everything below exists because `ConfigV2` carries TWO names -- `renderer`
    // and `parser` -- and a model is only fully supported when BOTH resolve. A
    // family that renders but does not parse produces a correct prompt and then
    // leaks raw `<tool_call>` markup into user-visible content; a family that
    // parses but does not render gets a generic template prompt and quietly
    // degrades. Neither failure raises an error at runtime, which is exactly why
    // it needs a test.
    //
    // These lists are transcribed from the oracle -- ollama
    // `model/renderers/renderer.go` `rendererForName` and
    // `model/parsers/parsers.go` `ParserForName`, pinned at 4713800b08. If you
    // re-vendor a newer ollama, re-transcribe them; that is the whole point.

    /// Every name upstream's `rendererForName` dispatches. 23 of them.
    const UPSTREAM_RENDERER_NAMES: &[&str] = &[
        "qwen3-coder",
        "qwen3-vl-instruct",
        "qwen3-vl-thinking",
        "qwen3.5",
        "ornith",
        "cogito",
        "deepseek3.1",
        "olmo3",
        "olmo3.1",
        "olmo3-think",
        "olmo3-32b-think",
        "nemotron-3-nano",
        "gemma4",
        "gemma4-small",
        "gemma4-large",
        "functiongemma",
        "glm-4.7",
        "glm-ocr",
        "lfm2",
        "lfm2-thinking",
        "laguna",
        "poolside-v1",
        "cohere",
    ];

    /// Every name upstream's `ParserForName` dispatches. 25 of them.
    const UPSTREAM_PARSER_NAMES: &[&str] = &[
        "qwen3",
        "qwen3-thinking",
        "qwen3.5",
        "ornith",
        "qwen3-coder",
        "qwen3-vl-instruct",
        "qwen3-vl-thinking",
        "ministral",
        "passthrough",
        "harmony",
        "cogito",
        "deepseek3",
        "olmo3",
        "olmo3-think",
        "nemotron-3-nano",
        "functiongemma",
        "glm-4.7",
        "gemma4",
        "gemma4-no-thinking",
        "glm-ocr",
        "lfm2",
        "lfm2-thinking",
        "laguna",
        "poolside-v1",
        "cohere",
    ];

    /// Names upstream registers on **both** sides. These are the only ones where
    /// "renderable but not parseable" is a genuine gap rather than upstream's own
    /// naming skew.
    fn upstream_names_on_both_sides() -> Vec<&'static str> {
        UPSTREAM_RENDERER_NAMES
            .iter()
            .copied()
            .filter(|n| UPSTREAM_PARSER_NAMES.contains(n))
            .collect()
    }

    /// Families this port can **render but not yet parse**, as of the last time
    /// somebody looked.
    ///
    /// This is a **ceiling, not a snapshot**: the test asserts the real gap is a
    /// *subset* of this list. [`crate::parsers`] is being filled in on its own
    /// schedule, so a name disappearing from the real gap must not fail the
    /// build -- but a name appearing that is not listed here must, because that
    /// means somebody shipped a renderer without checking the other half.
    ///
    /// When a parser lands, deleting its name from here is optional tidying, not
    /// a required step.
    const KNOWN_RENDERABLE_BUT_UNPARSEABLE: &[&str] = &[
        "cogito",
        "olmo3",
        "olmo3-think",
        "nemotron-3-nano",
        "functiongemma",
        "lfm2",
        "lfm2-thinking",
        "laguna",
        "poolside-v1",
        "cohere",
    ];

    /// The renderer half of the port is complete, and stays complete.
    ///
    /// Equality in both directions on purpose: a missing name is an unsupported
    /// model, and an **extra** name is worse -- an invented framing nothing was
    /// fine-tuned on, which no fixture would ever catch.
    #[test]
    fn the_renderer_registry_is_at_parity_with_upstream() {
        let _g = REGISTRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let missing: Vec<&str> = UPSTREAM_RENDERER_NAMES
            .iter()
            .copied()
            .filter(|n| built_in_renderer(n).is_none())
            .collect();
        assert!(
            missing.is_empty(),
            "upstream renders these and we do not: {missing:?}"
        );

        // The other direction. `built_in_renderer` is a closed `match`, so the
        // only way to enumerate it is to probe -- these are the names a typo or
        // an over-eager alias would plausibly add.
        for invented in [
            "laguna-v8",
            "poolside",
            "nemotron",
            "nemotron3nano",
            "glm-4.6",
            "deepseek3",
            "gemma4-no-thinking",
            "qwen3",
            "harmony",
            "ministral",
            "passthrough",
        ] {
            assert!(
                built_in_renderer(invented).is_none(),
                "{invented:?} is not an upstream renderer name -- \
                 an invented framing is worse than an unsupported model"
            );
        }
    }

    /// The reconciliation itself: state the rule, and fail when the port drifts
    /// **further** from it.
    ///
    /// Three claims, each for its own reason:
    ///
    /// 1. **No new render-without-parse gaps.** Over the 18 names upstream
    ///    registers on both sides, whatever we can render we should be able to
    ///    parse; the exceptions are enumerated in
    ///    [`KNOWN_RENDERABLE_BUT_UNPARSEABLE`] and the check is `⊆`, so a parser
    ///    landing shrinks the gap and passes.
    /// 2. **No parse-without-render at all.** If we parse a name, we must render
    ///    it -- *unless* upstream itself makes that name parser-only
    ///    (`harmony`, `passthrough`, `deepseek3`, ...). Unlike claim 1 this is
    ///    exact, because we are already at renderer parity, so any violation is
    ///    a genuine regression rather than unfinished work.
    /// 3. **Nobody invented a name.** Every name either side dispatches must
    ///    appear in the corresponding upstream list.
    ///
    /// What would make this test wrong: re-vendoring a newer ollama without
    /// re-transcribing `UPSTREAM_RENDERER_NAMES` / `UPSTREAM_PARSER_NAMES`. The
    /// lists are the oracle's, copied by hand; they cannot notice upstream
    /// moving on by themselves.
    #[test]
    fn the_renderer_and_parser_name_lists_agree_where_upstream_says_they_should() {
        let _g = REGISTRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let renders = |n: &str| built_in_renderer(n).is_some();
        let parses = |n: &str| crate::parsers::parser_for_name(n).is_some();

        // (1) Renderable but not parseable, over the shared 18.
        let gap: Vec<&str> = upstream_names_on_both_sides()
            .into_iter()
            .filter(|n| renders(n) && !parses(n))
            .collect();
        let unexpected: Vec<&&str> = gap
            .iter()
            .filter(|n| !KNOWN_RENDERABLE_BUT_UNPARSEABLE.contains(n))
            .collect();
        assert!(
            unexpected.is_empty(),
            "these families gained a renderer with no parser: {unexpected:?}\n\
             A renderable-but-unparseable family produces a correct prompt and then \
             leaks raw tool-call markup into user-visible content. Either land the \
             parser or add the name to KNOWN_RENDERABLE_BUT_UNPARSEABLE with a reason."
        );

        // (2) Parseable but not renderable -- allowed only where upstream is
        // itself parser-only.
        let upstream_parser_only: Vec<&str> = UPSTREAM_PARSER_NAMES
            .iter()
            .copied()
            .filter(|n| !UPSTREAM_RENDERER_NAMES.contains(n))
            .collect();
        let reverse_gap: Vec<&str> = UPSTREAM_PARSER_NAMES
            .iter()
            .copied()
            .filter(|n| parses(n) && !renders(n) && !upstream_parser_only.contains(n))
            .collect();
        assert!(
            reverse_gap.is_empty(),
            "these parse but no longer render: {reverse_gap:?} -- \
             the renderer registry is meant to be at full upstream parity"
        );

        // (3) Nothing invented on either side.
        for name in KNOWN_RENDERABLE_BUT_UNPARSEABLE {
            assert!(
                UPSTREAM_RENDERER_NAMES.contains(name) && UPSTREAM_PARSER_NAMES.contains(name),
                "{name:?} is on the gap list but is not a name upstream \
                 registers on both sides -- the list has gone stale"
            );
        }
    }

    /// The accounting, asserted rather than trusted, so the numbers quoted in
    /// this module's docs cannot silently rot.
    #[test]
    fn upstream_itself_keeps_two_deliberately_different_name_lists() {
        assert_eq!(UPSTREAM_RENDERER_NAMES.len(), 23);
        assert_eq!(UPSTREAM_PARSER_NAMES.len(), 25);
        assert_eq!(upstream_names_on_both_sides().len(), 18);

        // The skews worth knowing by name: the same family under two spellings.
        assert!(UPSTREAM_RENDERER_NAMES.contains(&"deepseek3.1"));
        assert!(UPSTREAM_PARSER_NAMES.contains(&"deepseek3"));
        assert!(!UPSTREAM_PARSER_NAMES.contains(&"deepseek3.1"));
        assert!(UPSTREAM_RENDERER_NAMES.contains(&"gemma4-large"));
        assert!(UPSTREAM_PARSER_NAMES.contains(&"gemma4-no-thinking"));
        // ...and the three that only ever parse.
        for parser_only in ["harmony", "passthrough", "ministral"] {
            assert!(UPSTREAM_PARSER_NAMES.contains(&parser_only));
            assert!(!UPSTREAM_RENDERER_NAMES.contains(&parser_only));
        }
    }
}
