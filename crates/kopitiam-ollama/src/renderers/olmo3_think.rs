//! **Olmo 3 Think** -- the reasoning Olmo variants. No tools at all.
//!
//! **Upstream:** `model/renderers/olmo3_think.go`. Registered as `olmo3-think`
//! (Olmo-3-7B-Think and Olmo-3.1-32B-Think, same template) and
//! `olmo3-32b-think` (Olmo-3-32B-Think).
//!
//! Special tokens: **`<|im_start|>`**, **`<|im_end|>`**, and a `<think>` that is
//! **always** opened at the end and never closed -- the model closes it itself.
//! No BOS.
//!
//! ## Tools are dropped, on purpose
//!
//! The `tools` argument is ignored, `tool` messages are filtered out, and an
//! assistant message's `tool_calls` are never rendered. These models were not
//! trained with tools, so the honest thing is to leave them out rather than
//! invent a framing. Upstream's fixtures assert exactly this
//! (`7b_tools_ignored`, `7b_tool_calls_and_tool_messages_ignored`).
//!
//! ## The system-message trap
//!
//! A **custom** system message gets a fixed suffix bolted on:
//! *" You do not currently have access to any functions. `<functions></functions>`"*.
//! With **no** custom system message the variant's own default is used and the
//! suffix is **not** added. So the two paths produce genuinely different tails,
//! and swapping them changes what the model believes about its own abilities.

use super::{IM_END_TAG, IM_START_TAG, Message, RenderError, Renderer, ThinkValue, Tool};

/// Which Think checkpoint. **Upstream:** `Olmo3ThinkVariant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Olmo3ThinkVariant {
    /// `allenai/Olmo-3-32B-Think`. **Upstream:** `Olmo3Think32B` (the zero
    /// value, hence `Default`).
    #[default]
    Olmo3Think32B,
    /// `allenai/Olmo-3-7B-Think` and `allenai/Olmo-3.1-32B-Think` -- same
    /// template, and it carries the Ai2 model info. **Upstream:** `Olmo31Think`.
    ///
    /// Upstream's own note: this **diverges from the HuggingFace template**, and
    /// the difference was confirmed with the Olmo team. Do not "fix" it back.
    Olmo31Think,
}

const FUNCTIONS_SUFFIX: &str =
    " You do not currently have access to any functions. <functions></functions>";
const THINK_32B_SYSTEM: &str = "You are a helpful AI assistant.";
const THINK_31_SYSTEM: &str = "You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at https://huggingface.co/allenai.";

/// **Upstream:** `Olmo3ThinkRenderer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Olmo3ThinkRenderer {
    pub variant: Olmo3ThinkVariant,
}

impl Renderer for Olmo3ThinkRenderer {
    fn render(
        &self,
        messages: &[Message],
        _tools: &[Tool],
        _think: Option<&ThinkValue>,
    ) -> Result<String, RenderError> {
        let mut sb = String::new();

        let system_message = messages.iter().find(|m| m.role == "system");
        // Tool messages are dropped along with system ones.
        let filtered: Vec<&Message> = messages
            .iter()
            .filter(|m| m.role != "system" && m.role != "tool")
            .collect();

        sb.push_str(IM_START_TAG);
        sb.push_str("system\n");
        match system_message {
            Some(system) => {
                sb.push_str(&system.content);
                sb.push_str(FUNCTIONS_SUFFIX);
            }
            None => sb.push_str(match self.variant {
                Olmo3ThinkVariant::Olmo3Think32B => THINK_32B_SYSTEM,
                Olmo3ThinkVariant::Olmo31Think => THINK_31_SYSTEM,
            }),
        }
        sb.push_str(IM_END_TAG);
        sb.push('\n');

        for message in filtered {
            match message.role.as_str() {
                "user" => {
                    sb.push_str(IM_START_TAG);
                    sb.push_str("user\n");
                    sb.push_str(&message.content);
                    sb.push_str(IM_END_TAG);
                    sb.push('\n');
                }
                "assistant" => {
                    sb.push_str(IM_START_TAG);
                    sb.push_str("assistant\n");
                    sb.push_str(&message.content);
                    sb.push_str(IM_END_TAG);
                    sb.push('\n');
                }
                _ => {}
            }
        }

        // Always thinking: the generation prompt opens `<think>` and leaves it
        // for the model to close.
        sb.push_str(IM_START_TAG);
        sb.push_str("assistant\n<think>");

        Ok(sb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ToolCall, ToolCallFunction};
    use serde_json::json;

    fn render(variant: Olmo3ThinkVariant, msgs: &[Message], tools: &[Tool]) -> String {
        Olmo3ThinkRenderer { variant }
            .render(msgs, tools, None)
            .unwrap()
    }

    /// Upstream `TestOlmo3ThinkRenderer`, all eight cases.
    #[test]
    fn olmo3_think_matches_the_upstream_fixtures() {
        let hello_tail = "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n<think>";

        assert_eq!(
            render(
                Olmo3ThinkVariant::Olmo31Think,
                &[Message::new("user", "Hello!")],
                &[]
            ),
            format!("<|im_start|>system\n{THINK_31_SYSTEM}<|im_end|>\n{hello_tail}")
        );

        // A custom system message picks up the "no functions" suffix.
        assert_eq!(
            render(
                Olmo3ThinkVariant::Olmo31Think,
                &[
                    Message::new("system", "You are a helpful assistant."),
                    Message::new("user", "Hello!"),
                ],
                &[]
            ),
            format!(
                "<|im_start|>system\nYou are a helpful assistant.{FUNCTIONS_SUFFIX}<|im_end|>\n{hello_tail}"
            )
        );

        assert_eq!(
            render(
                Olmo3ThinkVariant::Olmo3Think32B,
                &[Message::new("user", "Hello!")],
                &[]
            ),
            format!("<|im_start|>system\n{THINK_32B_SYSTEM}<|im_end|>\n{hello_tail}")
        );

        assert_eq!(
            render(
                Olmo3ThinkVariant::Olmo3Think32B,
                &[
                    Message::new("system", "You are a helpful assistant."),
                    Message::new("user", "Hello!"),
                ],
                &[]
            ),
            format!(
                "<|im_start|>system\nYou are a helpful assistant.{FUNCTIONS_SUFFIX}<|im_end|>\n{hello_tail}"
            )
        );

        // Multi-turn.
        assert_eq!(
            render(
                Olmo3ThinkVariant::Olmo31Think,
                &[
                    Message::new("user", "Hello"),
                    Message::new("assistant", "Hi there!"),
                    Message::new("user", "How are you?"),
                ],
                &[]
            ),
            format!(
                "<|im_start|>system\n{THINK_31_SYSTEM}<|im_end|>\n\
                 <|im_start|>user\nHello<|im_end|>\n\
                 <|im_start|>assistant\nHi there!<|im_end|>\n\
                 <|im_start|>user\nHow are you?<|im_end|>\n\
                 <|im_start|>assistant\n<think>"
            )
        );
    }

    /// Upstream's two "ignored" fixtures, kept because they pin a *deliberate
    /// omission* -- the sort of behaviour that quietly regresses when somebody
    /// "adds tool support" without reading the model card.
    #[test]
    fn tools_tool_calls_and_tool_messages_are_all_dropped() {
        let tools: Vec<Tool> = vec![
            serde_json::from_value(json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {"type": "", "properties": {}}
                }
            }))
            .unwrap(),
        ];
        let got = render(
            Olmo3ThinkVariant::Olmo31Think,
            &[Message::new("user", "What is the weather?")],
            &tools,
        );
        assert_eq!(
            got,
            format!(
                "<|im_start|>system\n{THINK_31_SYSTEM}<|im_end|>\n\
                 <|im_start|>user\nWhat is the weather?<|im_end|>\n<|im_start|>assistant\n<think>"
            )
        );

        let got = render(
            Olmo3ThinkVariant::Olmo31Think,
            &[
                Message::new("user", "What is the weather in SF?"),
                Message {
                    role: "assistant".into(),
                    content: "Let me check the weather.".into(),
                    tool_calls: vec![ToolCall {
                        id: "call_1".into(),
                        function: ToolCallFunction {
                            index: 0,
                            name: "get_weather".into(),
                            arguments: serde_json::from_str(r#"{"location":"San Francisco"}"#)
                                .unwrap(),
                        },
                    }],
                    ..Default::default()
                },
                Message::new("tool", r#"{"temperature": 68}"#),
            ],
            &[],
        );
        assert_eq!(
            got,
            format!(
                "<|im_start|>system\n{THINK_31_SYSTEM}<|im_end|>\n\
                 <|im_start|>user\nWhat is the weather in SF?<|im_end|>\n\
                 <|im_start|>assistant\nLet me check the weather.<|im_end|>\n\
                 <|im_start|>assistant\n<think>"
            )
        );
    }
}
