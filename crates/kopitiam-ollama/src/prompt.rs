//! Fitting a conversation into the context window before it is templated.
//!
//! **Upstream:** `server/prompt.go` (ollama, MIT).
//!
//! ## The problem this solves
//!
//! A conversation grows without bound; a context window does not. Something has
//! to go. The naive answers are all wrong in ways that show up as the model
//! acting confused rather than as an error:
//!
//! * **Truncate the raw string** -- you cut mid-token, mid-tag, or mid-turn, and
//!   the model sees a malformed prompt.
//! * **Drop the oldest N messages blindly** -- you can drop the system prompt,
//!   and the model forget who it supposed to be.
//! * **Count characters instead of tokens** -- you off by a factor that varies
//!   per language, so you either waste window or overflow it.
//!
//! Upstream's answer, ported here: **drop whole messages from the front, one at
//! a time, re-render and re-tokenise after each drop, and stop as soon as it
//! fits** -- while always keeping (1) every system message and (2) the latest
//! message, no matter what.
//!
//! ## Why it re-renders every iteration instead of measuring once
//!
//! Because the template is not linear in the messages lah. A chat template can
//! emit a tools preamble, a system block, per-turn markers -- so removing one
//! message does not remove a fixed number of tokens. The only honest way to know
//! whether the prompt fits is to build the actual prompt and count it. That is
//! O(n) renders worst case and upstream accept the cost, because being wrong
//! here is worse than being slow.
//!
//! ## Where the token count comes from
//!
//! This module does **not** tokenise. It takes a [`Tokenize`] callback, because
//! `kopitiam-ollama` deliberately depends on nothing else in KOPITIAM (see
//! `docs/ai-decisions/AID-0055`) and tokenising belong to `kopitiam-tokenizer`.
//! The seam also make the whole thing testable with a fake tokenizer, which is
//! how the tests below pin the drop-from-the-front behaviour without a model.

use crate::api::{Message, ThinkValue, Tool};
use crate::gotmpl::Env;
use crate::template::{Template, Values};

/// Count the tokens in a rendered prompt.
///
/// **Upstream:** `type tokenizeFunc func(context.Context, string) ([]int, error)`
/// -- upstream returns the token ids and takes the length; we only ever need the
/// length, so that is what the callback returns. Returning a count instead of a
/// `Vec<u32>` also spare the caller an allocation per iteration, and there can
/// be many iterations.
pub trait Tokenize {
    /// Number of tokens `s` encodes to. An error here aborts the fit.
    fn count(&self, s: &str) -> Result<usize, String>;
}

impl<F> Tokenize for F
where
    F: Fn(&str) -> Result<usize, String>,
{
    fn count(&self, s: &str) -> Result<usize, String> {
        self(s)
    }
}

/// How many tokens one image is assumed to cost.
///
/// **Upstream:** `imageNumTokens = 768` in `chatPrompt`, with its own comment:
/// *"Clip images are represented as 768 tokens, each an embedding."*
///
/// Upstream also carry a standing `TODO` right above it saying this is **only a
/// truncation heuristic** -- the real media accounting happens further down in
/// the runner, and the number should eventually become projector-aware. Copied
/// verbatim, TODO and all, because a different guess here would just be a
/// *different* wrongness with no evidence behind it.
pub const IMAGE_NUM_TOKENS: usize = 768;

/// What went wrong fitting the prompt.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PromptError {
    #[error("rendering the prompt failed: {0}")]
    Render(String),
    #[error("tokenizing the prompt failed: {0}")]
    Tokenize(String),
}

/// The result of fitting a conversation to the window.
#[derive(Debug, Clone, PartialEq)]
pub struct FittedPrompt {
    /// The rendered prompt, ready for the model.
    pub prompt: String,
    /// Index of the first message that survived. `0` means nothing was dropped.
    ///
    /// Not upstream's (it only logs the count) -- exposed because a caller that
    /// silently drop half a conversation should be able to *tell the user* it
    /// happened. Truncation the user cannot see is how "the model forgot what I
    /// said" becomes an unexplainable bug report.
    pub first_kept: usize,
    /// How many messages were dropped from the front.
    pub dropped: usize,
}

/// Render a conversation into a prompt that fits `num_ctx`.
///
/// **Upstream:** `chatPrompt(ctx, m, tokenize, opts, msgs, tools, think, truncate)`.
///
/// The rules, stated precisely because they are the contract:
///
/// * **System messages are never dropped.** As the front of the conversation get
///   discarded, any system message inside the discarded portion is *collected
///   and re-prepended*. So a system prompt written twenty turns ago still frame
///   the model's twenty-first reply.
/// * **The latest message is never dropped.** Even if it alone overflow the
///   window, it is kept and the prompt goes out oversized -- upstream's
///   `if i == lastMsgIdx` escape. Sending an over-long prompt is a problem the
///   runner can report; sending an *empty* one is a problem nobody can debug.
/// * **`truncate == false` skips the whole search** and renders everything, for
///   callers that want to fail loudly on overflow instead of silently shrinking.
///
/// `num_ctx` of `0` means "not yet resolved" (see
/// [`crate::options::Runner::num_ctx`]), and is treated as **no limit** here
/// rather than as a zero-size window -- truncating everything away because the
/// window size had not been decided yet would be the worst possible reading.
#[allow(clippy::too_many_arguments)]
pub fn chat_prompt(
    template: &Template,
    tokenize: &dyn Tokenize,
    num_ctx: usize,
    msgs: &[Message],
    tools: &[Tool],
    think: Option<&ThinkValue>,
    truncate: bool,
    has_projector: bool,
) -> Result<FittedPrompt, PromptError> {
    if msgs.is_empty() {
        let prompt = render(template, &[], tools, think)?;
        return Ok(FittedPrompt {
            prompt,
            first_kept: 0,
            dropped: 0,
        });
    }

    let last_msg_idx = msgs.len() - 1;
    let mut curr_msg_idx = 0usize;

    // `num_ctx == 0` means "undecided", not "zero tokens" -- see the doc above.
    if truncate && num_ctx > 0 {
        for i in 0..=last_msg_idx {
            // Collect the system messages from the portion we about to skip, so
            // they survive the drop.
            let mut candidate = system_messages(&msgs[..i]);
            candidate.extend_from_slice(&msgs[i..]);

            let p = render(template, &candidate, tools, think)?;
            let mut ctx_len = tokenize.count(&p).map_err(PromptError::Tokenize)?;

            if has_projector {
                for m in &msgs[i..] {
                    ctx_len += IMAGE_NUM_TOKENS * m.images.len();
                }
            }

            if ctx_len <= num_ctx {
                curr_msg_idx = i;
                break;
            }

            // Must always include at least the last message, even oversized.
            if i == last_msg_idx {
                curr_msg_idx = last_msg_idx;
                break;
            }
        }
    }

    let mut final_msgs = system_messages(&msgs[..curr_msg_idx]);
    final_msgs.extend_from_slice(&msgs[curr_msg_idx..]);

    Ok(FittedPrompt {
        prompt: render(template, &final_msgs, tools, think)?,
        first_kept: curr_msg_idx,
        dropped: curr_msg_idx,
    })
}

/// The system messages inside a slice, in order.
///
/// **Upstream:** the inline `for j := range i { if msgs[j].Role == "system" }`
/// loop inside `chatPrompt`, lifted out because it runs twice and a copy-paste
/// divergence between the two would be silent.
fn system_messages(msgs: &[Message]) -> Vec<Message> {
    msgs.iter().filter(|m| m.role == "system").cloned().collect()
}

/// **Upstream:** `renderPrompt(m, msgs, tools, think)`.
///
/// Upstream branch first on `m.Config.Renderer` -- a model-family-specific
/// renderer (`model/renderers/`) take precedence over the generic template path.
/// That branch is **not wired in here yet**: the renderers live in
/// [`crate::renderers`] and are being ported separately. When they land, this is
/// the single function that gain the `if renderer != "" { ... }` arm, so the
/// seam stay in one place rather than smeared across callers.
fn render(
    template: &Template,
    msgs: &[Message],
    tools: &[Tool],
    think: Option<&ThinkValue>,
) -> Result<String, PromptError> {
    let values = Values {
        messages: msgs.to_vec(),
        tools: tools.to_vec(),
        think: think.map(ThinkValue::enabled).unwrap_or(false),
        think_level: think.map(|t| t.level().to_string()).unwrap_or_default(),
        // Upstream: `IsThinkSet: think != nil`. This is the flag that let a
        // template tell "thinking explicitly off" apart from "thinking never
        // mentioned".
        is_think_set: think.is_some(),
        ..Default::default()
    };
    template
        .execute(&values, &Env::default())
        .map_err(|e| PromptError::Render(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One "token" per whitespace-separated word. Crude on purpose: the point of
    /// these tests is the DROP LOGIC, and a fake tokenizer make the arithmetic
    /// legible instead of hiding it behind a real vocabulary.
    fn words() -> impl Tokenize {
        |s: &str| -> Result<usize, String> { Ok(s.split_whitespace().count()) }
    }

    fn tmpl() -> Template {
        // One entry per message, no extra framing, so token counts are obvious.
        Template::parse("{{ range .Messages }}{{ .Role }} {{ .Content }} {{ end }}").unwrap()
    }

    fn msgs(pairs: &[(&str, &str)]) -> Vec<Message> {
        pairs.iter().map(|(r, c)| Message::new(r, c)).collect()
    }

    fn fit(num_ctx: usize, m: &[Message]) -> FittedPrompt {
        chat_prompt(&tmpl(), &words(), num_ctx, m, &[], None, true, false).unwrap()
    }

    #[test]
    fn a_conversation_that_already_fits_is_left_alone() {
        let m = msgs(&[("user", "a"), ("assistant", "b")]);
        let out = fit(100, &m);
        assert_eq!(out.dropped, 0);
        assert_eq!(out.prompt, "user a assistant b ");
    }

    /// The core behaviour: drop from the FRONT, one message at a time, until it
    /// fits.
    #[test]
    fn messages_are_dropped_from_the_front_until_it_fits() {
        let m = msgs(&[
            ("user", "one"),
            ("assistant", "two"),
            ("user", "three"),
            ("assistant", "four"),
        ]);
        // Each message costs 2 "tokens" (role + content). Budget 4 -> 2 messages.
        let out = fit(4, &m);
        assert_eq!(out.dropped, 2);
        assert_eq!(out.prompt, "user three assistant four ");
    }

    /// **The system prompt must survive being dropped.** This is the rule whose
    /// absence make a model "forget who it is" deep into a long conversation.
    #[test]
    fn a_system_message_is_re_prepended_after_the_front_is_dropped() {
        let m = msgs(&[
            ("system", "be-terse"),
            ("user", "one"),
            ("assistant", "two"),
            ("user", "three"),
        ]);
        // Budget only fits the system message plus the last turn.
        let out = fit(4, &m);
        assert!(
            out.prompt.starts_with("system be-terse "),
            "system prompt must survive, got {:?}",
            out.prompt
        );
        assert!(out.prompt.ends_with("user three "), "got {:?}", out.prompt);
        assert!(!out.prompt.contains("one"), "old turns must be gone");
    }

    /// Several system messages must ALL survive, in order.
    #[test]
    fn every_system_message_survives_not_just_the_first() {
        let m = msgs(&[
            ("system", "s-one"),
            ("user", "a"),
            ("system", "s-two"),
            ("user", "b"),
            ("user", "c"),
        ]);
        let out = fit(6, &m);
        let s1 = out.prompt.find("s-one");
        let s2 = out.prompt.find("s-two");
        assert!(s1.is_some() && s2.is_some(), "got {:?}", out.prompt);
        assert!(s1 < s2, "system messages must keep their order");
    }

    /// The escape hatch: even a single message that overflow is kept. An
    /// oversized prompt is a reportable problem; an empty one is not debuggable.
    #[test]
    fn the_latest_message_is_kept_even_when_it_alone_overflows() {
        let m = msgs(&[("user", "one"), ("user", "a b c d e f g h i j")]);
        let out = fit(3, &m);
        assert_eq!(out.dropped, 1);
        assert!(out.prompt.contains("a b c d e f g h i j"));
    }

    #[test]
    fn truncate_false_renders_everything_regardless_of_the_window() {
        let m = msgs(&[("user", "one"), ("assistant", "two"), ("user", "three")]);
        let out = chat_prompt(&tmpl(), &words(), 1, &m, &[], None, false, false).unwrap();
        assert_eq!(out.dropped, 0);
        assert!(out.prompt.contains("one") && out.prompt.contains("three"));
    }

    /// `num_ctx == 0` means "not yet decided", so it must NOT be read as a
    /// zero-size window that truncates everything away.
    #[test]
    fn a_zero_context_length_means_undecided_not_empty() {
        let m = msgs(&[("user", "one"), ("assistant", "two"), ("user", "three")]);
        let out = fit(0, &m);
        assert_eq!(out.dropped, 0, "0 must not be read as a zero-size window");
        assert!(out.prompt.contains("one"));
    }

    /// Images cost tokens the text tokenizer cannot see, so they must be added
    /// to the estimate -- otherwise a picture-heavy chat overflow silently.
    #[test]
    fn images_are_charged_against_the_window_when_a_projector_is_present() {
        let mut m = msgs(&[("user", "one"), ("user", "two")]);
        m[1].images = vec!["<base64>".to_string()];

        // Without a projector the image is free, so this budget fits everything.
        let no_proj = chat_prompt(&tmpl(), &words(), 4, &m, &[], None, true, false).unwrap();
        assert_eq!(no_proj.dropped, 0);

        // With a projector the image costs IMAGE_NUM_TOKENS, blowing the same
        // budget -- so the earlier turn gets dropped to make room.
        let proj = chat_prompt(&tmpl(), &words(), 4, &m, &[], None, true, true).unwrap();
        assert_eq!(proj.dropped, 1, "the image must be charged against the window");
        assert!(IMAGE_NUM_TOKENS > 4, "the fixture only makes sense if it overflows");
    }

    #[test]
    fn an_empty_conversation_renders_without_panicking() {
        let out = fit(100, &[]);
        assert_eq!(out.dropped, 0);
        assert_eq!(out.prompt, "");
    }

    #[test]
    fn a_tokenizer_failure_is_surfaced_not_swallowed() {
        let boom = |_: &str| -> Result<usize, String> { Err("tokenizer exploded".into()) };
        let m = msgs(&[("user", "a"), ("user", "b")]);
        let err = chat_prompt(&tmpl(), &boom, 5, &m, &[], None, true, false).unwrap_err();
        assert_eq!(err, PromptError::Tokenize("tokenizer exploded".into()));
    }

    /// The thinking state must reach the template, and `is_think_set` must
    /// distinguish "explicitly off" from "never mentioned".
    #[test]
    fn thinking_state_is_threaded_through_to_the_template() {
        let t = Template::parse(
            "{{ range .Messages }}{{ end }}{{ if .IsThinkSet }}{{ .ThinkLevel }}{{ else }}unset{{ end }}",
        )
        .unwrap();
        let m = msgs(&[("user", "hi")]);

        let unset = chat_prompt(&t, &words(), 0, &m, &[], None, false, false).unwrap();
        assert_eq!(unset.prompt, "unset");

        let off = chat_prompt(
            &t,
            &words(),
            0,
            &m,
            &[],
            Some(&ThinkValue::Bool(false)),
            false,
            false,
        )
        .unwrap();
        assert_eq!(off.prompt, "", "explicitly off: set, but no level");

        let high = chat_prompt(
            &t,
            &words(),
            0,
            &m,
            &[],
            Some(&ThinkValue::Level(crate::api::ThinkLevel::High)),
            false,
            false,
        )
        .unwrap();
        assert_eq!(high.prompt, "high");
    }
}
