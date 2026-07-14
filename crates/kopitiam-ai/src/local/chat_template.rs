//! ChatML rendering: turning [`crate::Message`]s into the literal prompt
//! text `kopitiam_runtime::generate` tokenizes.
//!
//! # The exact template
//!
//! Qwen (every generation shipped as GGUF to date) is instruction-tuned
//! against ChatML — the same template `llama.cpp` calls
//! `LLM_CHAT_TEMPLATE_CHATML`, whose C++ implementation
//! (`llm_chat_apply_template`'s `chatml` branch) is exactly:
//!
//! ```text
//! for (auto message : chat) {
//!     ss << "<|im_start|>" << message->role << "\n" << message->content << "<|im_end|>\n";
//! }
//! if (add_ass) {
//!     ss << "<|im_start|>assistant\n";
//! }
//! ```
//!
//! [`render_chatml`] is that loop, transliterated: one
//! `<|im_start|>{role}\n{content}<|im_end|>\n` block per message, then an
//! unconditional trailing `<|im_start|>assistant\n` generation prompt
//! (`add_ass`/`add_generation_prompt` is never optional here — every call
//! into [`crate::LocalAdapter::complete`] exists to produce an assistant
//! turn, so there is no scenario where a caller wants the prompt without
//! it).
//!
//! # What this deliberately does NOT do
//!
//! The real `Qwen2.5-Instruct` `tokenizer_config.json` chat template layers
//! one more rule on top of bare ChatML: if `messages[0]` is not a system
//! message, it injects a hardcoded default persona ("You are Qwen, created
//! by Alibaba Cloud, you are a helpful assistant..."). This module does
//! not replicate that, for two reasons. First, `kopitiam-workflow` — the
//! only caller of `kopitiam-ai` anywhere in the platform, per the Semantic
//! Runtime's dependency rule — always renders an explicit
//! `Message::system` first (`kopitiam_workflow::workflow::render_messages`
//! pushes `Message::system(kind.system_preamble())` before anything else),
//! so that branch would never fire for any real request `kopitiam-ai` ever
//! sees. Second, and more importantly: this module's own docs warn that
//! "a wrong chat template produces fluent-sounding garbage that is very
//! hard to attribute" — silently injecting an un-requested, hardcoded
//! persona string is exactly that failure mode, not a safe default. A
//! caller that wants a default persona should render one explicitly as a
//! `Message::system`; this module's job is to render `messages` faithfully,
//! not to guess what a caller meant to include.
//!
//! # Verification
//!
//! [`tests::renders_a_system_user_assistant_conversation_exactly`] pins
//! this module's output against `llama.cpp`'s own `chatml` template test
//! case (`tests/test-chat-template.cpp`, the
//! `"teknium/OpenHermes-2.5-Mistral-7B"` case — that model's template is
//! textually the generic ChatML template above), byte for byte, rather
//! than a fixture this module invented and is merely self-consistent with.

use crate::{Message, Role};

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// The three ChatML role tags this module renders. A free function over
/// [`Role`] rather than a `Display` impl on `Role` itself, since `Role` is
/// a generic `kopitiam-ai` concept shared by every adapter
/// (see [`crate::message`]), while `"system"`/`"user"`/`"assistant"` is
/// ChatML's own vocabulary, not necessarily every future adapter's.
fn role_tag(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

/// Renders `messages` as a ChatML prompt, always ending with the
/// `<|im_start|>assistant\n` generation prompt. See this module's docs for
/// the exact template and what it deliberately omits.
pub(crate) fn render_chatml(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str(IM_START);
        prompt.push_str(role_tag(message.role));
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str(IM_END);
        prompt.push('\n');
    }
    prompt.push_str(IM_START);
    prompt.push_str("assistant\n");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;

    /// Ground truth copied verbatim from `llama.cpp`'s own chatml test
    /// case (see this module's docs) — not re-derived from this module's
    /// own understanding of the template, which is exactly the kind of
    /// "internally consistent but silently wrong" trap a hand-authored
    /// expected string could fall into.
    #[test]
    fn renders_a_system_user_assistant_conversation_exactly() {
        let messages = [
            Message::system("You are a helpful assistant"),
            Message::user("Hello"),
            Message::assistant("Hi there"),
            Message::user("Who are you"),
            Message::assistant("   I am an assistant   "),
            Message::user("Another question"),
        ];

        let expected = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n\
<|im_start|>user\nHello<|im_end|>\n\
<|im_start|>assistant\nHi there<|im_end|>\n\
<|im_start|>user\nWho are you<|im_end|>\n\
<|im_start|>assistant\n   I am an assistant   <|im_end|>\n\
<|im_start|>user\nAnother question<|im_end|>\n\
<|im_start|>assistant\n";

        assert_eq!(render_chatml(&messages), expected);
    }

    /// Table-tests smaller/edge-case conversations that the single big
    /// fixture above does not cover: no messages at all, a single turn,
    /// and empty message content.
    #[test]
    fn renders_edge_case_conversations_exactly() {
        let cases: &[(&[Message], &str)] = &[
            (&[], "<|im_start|>assistant\n"),
            (
                &[Message { role: Role::User, content: String::new() }],
                "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n",
            ),
            (
                &[Message { role: Role::System, content: "be terse".to_string() }],
                "<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>assistant\n",
            ),
            (
                &[
                    Message { role: Role::System, content: "be terse".to_string() },
                    Message { role: Role::User, content: "2+2?".to_string() },
                ],
                "<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n",
            ),
        ];

        for (messages, expected) in cases {
            assert_eq!(render_chatml(messages), *expected, "mismatch rendering {messages:?}");
        }
    }

    /// Content containing characters ChatML's own delimiters are built
    /// from (`<`, `|`, newlines) must pass through untouched — this module
    /// renders literal text, it does not escape or re-parse it.
    #[test]
    fn message_content_is_never_escaped_or_reinterpreted() {
        let messages = [Message::user("line one\nline two <|im_end|> not a real stop tag")];
        assert_eq!(
            render_chatml(&messages),
            "<|im_start|>user\nline one\nline two <|im_end|> not a real stop tag<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
