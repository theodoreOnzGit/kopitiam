//! **DeepSeek-V3.1** -- full-width special tokens, thinking opt-**in**.
//!
//! **Upstream:** `model/renderers/deepseek3.go`. Registered as `deepseek3.1`.
//!
//! ## The framing
//!
//! Every special token in this family is written with **full-width** vertical
//! bars (`｜`, U+FF5C) and a **black down-pointing small triangle** (`▁`,
//! U+2581) as the word separator. They are not ASCII lookalikes -- copy them,
//! never retype them, or the tokenizer will split them into gibberish:
//!
//! * `<｜begin▁of▁sentence｜>` -- also the BOS ([`Renderer::leading_bos`])
//! * `<｜end▁of▁sentence｜>`
//! * `<｜User｜>` / `<｜Assistant｜>` (capitalised, no lowercase variant)
//! * `<｜tool▁calls▁begin｜>` / `<｜tool▁call▁begin｜>` / `<｜tool▁sep｜>` /
//!   `<｜tool▁call▁end｜>` / `<｜tool▁calls▁end｜>`
//! * `<｜tool▁output▁begin｜>` / `<｜tool▁output▁end｜>`
//! * the plain-ASCII `<think>` / `</think>` pair
//!
//! ## Things that look wrong and are not
//!
//! * **Thinking needs BOTH sides to agree.** `is_thinking` (the model supports
//!   it) **and** an explicit `think = Some(true)` from the caller. `None` means
//!   off here -- the exact opposite of [`super::glm47`], so do not "harmonise"
//!   them.
//! * **Every assistant turn opens with `</think>`** when thinking is off, and
//!   the *generation prompt* likewise. Closing tag, no opening tag: that is the
//!   family's way of saying "no reasoning this turn".
//! * **`</think>` inside historical content is a cut point.** Content is split
//!   on the **first** `</think>` and only what follows survives -- upstream's
//!   `SplitN(content, "</think>", 2)`. A message reading
//!   `"First</think>Second</think>Final"` renders as
//!   `"Second</think>Final"`, which upstream's own fixture asserts.
//! * **System messages are hoisted to the very front**, joined with `\n\n`, and
//!   never re-emitted in the turn loop.
//! * The BOS is written into the prompt **and** returned by
//!   [`Renderer::leading_bos`]. The caller must not add it twice.

use super::json::{go_arguments, write_go_parameters};
use super::{Message, RenderError, Renderer, ThinkValue, Tool};

/// Which DeepSeek-3 dialect. **Upstream:** `DeepSeek3Variant`. Only one exists
/// today, but upstream models it as an enum, so we do too rather than bake the
/// single case in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeepSeek3Variant {
    /// `deepseek3.1`. **Upstream:** `Deepseek31`.
    #[default]
    DeepSeek31,
}

/// **Upstream:** `DeepSeek3Renderer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeepSeek3Renderer {
    /// Does this model support thinking at all? Necessary but **not
    /// sufficient** -- the caller must ask for it too.
    pub is_thinking: bool,
    pub variant: DeepSeek3Variant,
}

const BOS: &str = "<｜begin▁of▁sentence｜>";
const EOS: &str = "<｜end▁of▁sentence｜>";

impl Renderer for DeepSeek3Renderer {
    fn leading_bos(&self) -> &'static str {
        BOS
    }

    fn render(
        &self,
        messages: &[Message],
        tools: &[Tool],
        think: Option<&ThinkValue>,
    ) -> Result<String, RenderError> {
        let mut sb = String::new();

        // Model must support it AND the caller must ask.
        let thinking = self.is_thinking && think.is_some_and(|t| t.enabled());

        // Hoist every system message to the front, joined with a blank line.
        let mut system_prompt = String::new();
        let mut first_system = true;
        for message in messages {
            if message.role == "system" {
                if first_system {
                    system_prompt.push_str(&message.content);
                    first_system = false;
                } else {
                    system_prompt.push_str("\n\n");
                    system_prompt.push_str(&message.content);
                }
            }
        }

        sb.push_str(BOS);
        sb.push_str(&system_prompt);

        if !tools.is_empty() {
            sb.push_str("\n\n## Tools\nYou have access to the following tools:\n");
            for tool in tools {
                sb.push_str(&format!("\n### {}", tool.function.name));
                sb.push_str(&format!("\nDescription: {}", tool.function.description));
                sb.push_str("\n\nParameters: ");
                write_go_parameters(&tool.function.parameters, &mut sb);
                sb.push('\n');
            }
            sb.push_str("\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n");
            sb.push_str("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>\n\n");
            sb.push_str("Where:\n\n");
            sb.push_str(
                "- `tool_call_name` must be an exact match to one of the available tools\n",
            );
            sb.push_str("- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema\n");
            sb.push_str(
                "- For multiple tool calls, chain them directly without separators or spaces\n",
            );
        }

        let mut is_tool = false;
        let mut is_last_user = false;

        // Which assistant messages count as "the current turn"?
        let last_user_index: Option<usize> = messages.iter().rposition(|m| m.role == "user");

        for (i, message) in messages.iter().enumerate() {
            match message.role.as_str() {
                "user" => {
                    is_tool = false;
                    is_last_user = true;
                    sb.push_str("<｜User｜>");
                    sb.push_str(&message.content);
                }
                "assistant" => {
                    if !message.tool_calls.is_empty() {
                        if is_last_user {
                            sb.push_str("<｜Assistant｜></think>");
                        }
                        is_last_user = false;
                        is_tool = false;

                        sb.push_str(&message.content);

                        sb.push_str("<｜tool▁calls▁begin｜>");
                        for tc in &message.tool_calls {
                            sb.push_str(&format!(
                                "<｜tool▁call▁begin｜>{}<｜tool▁sep｜>",
                                tc.function.name
                            ));
                            sb.push_str(&go_arguments(&tc.function.arguments));
                            sb.push_str("<｜tool▁call▁end｜>");
                        }
                        sb.push_str("<｜tool▁calls▁end｜>");
                        sb.push_str(EOS);
                    } else {
                        if is_last_user {
                            sb.push_str("<｜Assistant｜>");
                            let has_thinking = !message.thinking.is_empty();
                            // Only the CURRENT turn (after the last user
                            // message) may open a real `<think>`.
                            let is_current_turn = last_user_index.is_none_or(|idx| i > idx);
                            if has_thinking && thinking && is_current_turn {
                                sb.push_str("<think>");
                            } else {
                                sb.push_str("</think>");
                            }
                        }
                        is_last_user = false;

                        let mut content = message.content.as_str();
                        if is_tool {
                            is_tool = false;
                        } else if let Some(idx) = content.find("</think>") {
                            // SplitN(.., 2): keep everything after the FIRST
                            // close tag, later ones stay in the text.
                            content = &content[idx + "</think>".len()..];
                        }
                        sb.push_str(content);
                        sb.push_str(EOS);
                    }
                }
                "tool" => {
                    is_last_user = false;
                    is_tool = true;
                    sb.push_str("<｜tool▁output▁begin｜>");
                    sb.push_str(&message.content);
                    sb.push_str("<｜tool▁output▁end｜>");
                }
                _ => {}
            }
        }

        if is_last_user && !is_tool {
            sb.push_str("<｜Assistant｜>");
            sb.push_str(if thinking { "<think>" } else { "</think>" });
        }

        Ok(sb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ToolCall, ToolCallFunction};

    fn renderer() -> DeepSeek3Renderer {
        // Upstream's `deepseek3_test.go` builds exactly this.
        DeepSeek3Renderer {
            is_thinking: true,
            variant: DeepSeek3Variant::DeepSeek31,
        }
    }

    fn call(name: &str, raw_args: &str) -> ToolCall {
        ToolCall {
            id: String::new(),
            function: ToolCallFunction {
                index: 0,
                name: name.into(),
                arguments: serde_json::from_str(raw_args).expect("valid fixture args"),
            },
        }
    }

    fn render(msgs: &[Message], think: bool) -> String {
        renderer()
            .render(msgs, &[], Some(&ThinkValue::Bool(think)))
            .unwrap()
    }

    /// Upstream `TestDeepSeekRenderer`, the non-tool-definition cases.
    #[test]
    fn deepseek3_matches_the_upstream_fixtures() {
        assert_eq!(
            render(&[Message::new("user", "Hello, how are you?")], false),
            "<｜begin▁of▁sentence｜><｜User｜>Hello, how are you?<｜Assistant｜></think>"
        );

        assert_eq!(
            render(
                &[
                    Message::new("system", "You are a helpful assistant."),
                    Message::new("user", "Hello, how are you?"),
                ],
                false
            ),
            "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Hello, how are you?<｜Assistant｜></think>"
        );

        // Several system messages join with a blank line between them.
        assert_eq!(
            render(
                &[
                    Message::new("system", "First instruction"),
                    Message::new("system", "Second instruction"),
                    Message::new("user", "Hello"),
                ],
                false
            ),
            "<｜begin▁of▁sentence｜>First instruction\n\nSecond instruction<｜User｜>Hello<｜Assistant｜></think>"
        );

        assert_eq!(
            render(&[Message::new("user", "Hello, how are you?")], true),
            "<｜begin▁of▁sentence｜><｜User｜>Hello, how are you?<｜Assistant｜><think>"
        );

        assert_eq!(
            render(
                &[
                    Message::new("user", "What is the capital of France?"),
                    Message::new("assistant", "The capital of France is Paris."),
                    Message::new("user", "Fantastic!"),
                ],
                false
            ),
            "<｜begin▁of▁sentence｜><｜User｜>What is the capital of France?<｜Assistant｜></think>The capital of France is Paris.<｜end▁of▁sentence｜><｜User｜>Fantastic!<｜Assistant｜></think>"
        );

        // No messages at all -> just the BOS.
        assert_eq!(render(&[], false), "<｜begin▁of▁sentence｜>");

        // System-only -> BOS plus the system text, no turn markers.
        assert_eq!(
            render(&[Message::new("system", "System instruction")], false),
            "<｜begin▁of▁sentence｜>System instruction"
        );

        // An empty system message contributes nothing.
        assert_eq!(
            render(
                &[Message::new("system", ""), Message::new("user", "Hello")],
                false
            ),
            "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"
        );

        // Special characters ride through untouched -- no escaping here, the
        // content is text, not JSON.
        assert_eq!(
            render(
                &[
                    Message::new("user", "What about <|special|> tokens and \"quotes\"?"),
                    Message::new("assistant", "They're handled normally."),
                ],
                false
            ),
            "<｜begin▁of▁sentence｜><｜User｜>What about <|special|> tokens and \"quotes\"?<｜Assistant｜></think>They're handled normally.<｜end▁of▁sentence｜>"
        );
    }

    /// Upstream's two `</think>` fixtures. The split is on the **first** close
    /// tag only -- later ones stay in the text, which is why the second
    /// assertion keeps a `</think>` in the middle of the output.
    #[test]
    fn historical_content_is_cut_at_the_first_think_close_tag() {
        assert_eq!(
            render(
                &[
                    Message::new("user", "Think about this"),
                    Message::new(
                        "assistant",
                        "I'm thinking about this.</think>The answer is 42."
                    ),
                ],
                false
            ),
            "<｜begin▁of▁sentence｜><｜User｜>Think about this<｜Assistant｜></think>The answer is 42.<｜end▁of▁sentence｜>"
        );

        assert_eq!(
            render(
                &[
                    Message::new("user", "Complex question"),
                    Message::new(
                        "assistant",
                        "First thought</think>Second thought</think>Final answer"
                    ),
                ],
                false
            ),
            "<｜begin▁of▁sentence｜><｜User｜>Complex question<｜Assistant｜></think>Second thought</think>Final answer<｜end▁of▁sentence｜>"
        );
    }

    /// Upstream's tool fixtures: calls, outputs, and the assistant message that
    /// follows a tool output (which must NOT get its own `<｜Assistant｜>`).
    #[test]
    fn tool_calls_outputs_and_the_reply_after_them() {
        let weather_call = || Message {
            role: "assistant".into(),
            tool_calls: vec![call("get_weather", r#"{"location":"Paris"}"#)],
            ..Default::default()
        };

        assert_eq!(
            render(
                &[Message::new("user", "What's the weather?"), weather_call()],
                false
            ),
            "<｜begin▁of▁sentence｜><｜User｜>What's the weather?<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"location\":\"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"
        );

        // Content plus a call: the content goes in front of the calls block.
        assert_eq!(
            render(
                &[
                    Message::new("user", "What's the weather in Paris?"),
                    Message {
                        content: "I'll check the weather for you.".into(),
                        ..weather_call()
                    },
                ],
                false
            ),
            "<｜begin▁of▁sentence｜><｜User｜>What's the weather in Paris?<｜Assistant｜></think>I'll check the weather for you.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"location\":\"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"
        );

        // Two calls chain with no separator; two outputs likewise.
        assert_eq!(
            render(
                &[
                    Message::new("user", "Get weather for Paris and London"),
                    Message {
                        role: "assistant".into(),
                        tool_calls: vec![
                            call("get_weather", r#"{"location":"Paris"}"#),
                            call("get_weather", r#"{"location":"London"}"#),
                        ],
                        ..Default::default()
                    },
                    Message::new("tool", "Paris: 22°C, Sunny"),
                    Message::new("tool", "London: 18°C, Cloudy"),
                ],
                false
            ),
            "<｜begin▁of▁sentence｜><｜User｜>Get weather for Paris and London<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"location\":\"Paris\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"location\":\"London\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>Paris: 22°C, Sunny<｜tool▁output▁end｜><｜tool▁output▁begin｜>London: 18°C, Cloudy<｜tool▁output▁end｜>"
        );

        // `Done` comes straight after the tool output with NO `<｜Assistant｜>`
        // marker -- that is the `is_tool` flag doing its job.
        assert_eq!(
            render(
                &[
                    Message::new("user", "Process data"),
                    Message {
                        role: "assistant".into(),
                        tool_calls: vec![call("process", r#"{"data":"test"}"#)],
                        ..Default::default()
                    },
                    Message::new("tool", "Success"),
                    Message::new("assistant", "Done"),
                ],
                false
            ),
            "<｜begin▁of▁sentence｜><｜User｜>Process data<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>process<｜tool▁sep｜>{\"data\":\"test\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>Success<｜tool▁output▁end｜>Done<｜end▁of▁sentence｜>"
        );
    }

    /// Upstream's "thinking enabled/disabled after tool call" pair -- the only
    /// difference in the whole prompt is the final tag.
    #[test]
    fn thinking_only_changes_the_generation_prompt_after_a_tool_round_trip() {
        let msgs = [
            Message::new("user", "What's the weather in Paris?"),
            Message {
                role: "assistant".into(),
                tool_calls: vec![call("get_weather", r#"{"location":"Paris"}"#)],
                ..Default::default()
            },
            Message::new("tool", "Temperature: 22°C, Sunny"),
            Message::new(
                "assistant",
                "Based on the weather data, it's sunny in Paris.",
            ),
            Message::new("user", "Now tell me about London weather too."),
        ];
        assert!(render(&msgs, true).ends_with("<｜Assistant｜><think>"));
        assert!(render(&msgs, false).ends_with("<｜Assistant｜></think>"));
    }

    /// `None` means OFF for this family -- the opposite of GLM-4.7. Getting
    /// this backwards makes every default request open a `<think>` the model
    /// then has to close on its own.
    #[test]
    fn an_absent_think_value_means_thinking_is_off() {
        let got = renderer()
            .render(&[Message::new("user", "hi")], &[], None)
            .unwrap();
        assert!(got.ends_with("</think>"), "{got}");
    }
}
