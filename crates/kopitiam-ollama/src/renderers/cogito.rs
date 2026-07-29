//! **Cogito** (Deep Cogito) -- DeepSeek's token set, a different assembly.
//!
//! **Upstream:** `model/renderers/cogito.go`. Registered as `cogito`.
//!
//! Special tokens: the same full-width family as
//! [`super::deepseek3`] -- `<｜begin▁of▁sentence｜>` (also the BOS),
//! `<｜end▁of▁sentence｜>`, `<｜User｜>`, `<｜Assistant｜>`,
//! `<｜tool▁calls▁begin｜>` / `<｜tool▁call▁begin｜>` / `<｜tool▁sep｜>` /
//! `<｜tool▁call▁end｜>` / `<｜tool▁calls▁end｜>`, and additionally
//! **`<｜tool▁outputs▁begin｜>` / `<｜tool▁outputs▁end｜>`** wrapping a run of
//! `<｜tool▁output▁begin｜>` ... `<｜tool▁output▁end｜>` entries. Full-width
//! bars (U+FF5C) and U+2581 separators -- copy, never retype.
//!
//! ## Things worth knowing before you touch it
//!
//! * **The identity prompt is always present**, whether or not the caller sent a
//!   system message: *"You are Cogito, an AI assistant created by Deep Cogito,
//!   which is an AI research lab based in San Francisco."* A caller's system
//!   text is appended after it.
//! * **Thinking is opt-in AND changes the system prompt**, not just the tail.
//!   With thinking on, the prompt is prefixed with *"Enable deep thinking
//!   subroutine.\n\n"* and a caller's system text is wrapped in `\n\n` on
//!   **both** sides. That trailing `\n\n` is upstream's, and it is why the
//!   thinking and non-thinking prompts differ by more than one tag.
//! * **`<｜User｜>` writes its own `<｜Assistant｜>`** immediately after the
//!   user's text. So the trailing generation prompt is added only when the last
//!   message was *not* a user message -- that is what `is_last_user` tracks.
//! * **Tools are pretty-printed** with a four-space indent inside a
//!   ```` ```json ```` fence, unlike DeepSeek's compact form.
//! * **Tool-call arguments are compact JSON**, also fenced, and the call names
//!   the literal word `function` before the separator:
//!   `<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather`.

use super::json::{go_arguments, go_tool, indent_go_json};
use super::{Message, RenderError, Renderer, ThinkValue, Tool};

/// **Upstream:** `CogitoRenderer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CogitoRenderer {
    /// Does this model support thinking? The caller must still ask for it.
    pub is_thinking: bool,
}

const BOS: &str = "<｜begin▁of▁sentence｜>";
const EOS: &str = "<｜end▁of▁sentence｜>";
const DEFAULT_PROMPT: &str = "You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.";

impl Renderer for CogitoRenderer {
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

        // Model must support it AND the caller must ask -- same rule as
        // DeepSeek, opposite of GLM-4.7.
        let enable_thinking = self.is_thinking && think.is_some_and(|t| t.enabled());

        let (system_prompt, conversation): (&str, &[Message]) = match messages.first() {
            Some(first) if first.role == "system" => (&first.content, &messages[1..]),
            _ => ("", messages),
        };

        let mut final_system_prompt = if enable_thinking {
            let mut p = format!("Enable deep thinking subroutine.\n\n{DEFAULT_PROMPT}");
            if !system_prompt.is_empty() {
                // Trailing `\n\n` is upstream's, not a typo.
                p.push_str(&format!("\n\n{system_prompt}\n\n"));
            }
            p
        } else {
            let mut p = DEFAULT_PROMPT.to_string();
            if !system_prompt.is_empty() {
                p.push_str(&format!("\n\n{system_prompt}"));
            }
            p
        };

        if !tools.is_empty() {
            // The `else` branch upstream is dead in practice -- the default
            // prompt is never empty -- but kept so the shapes line up.
            if !final_system_prompt.is_empty() {
                final_system_prompt.push_str("\nYou have the following functions available:\n");
            } else {
                final_system_prompt = "You have the following functions available:\n".to_string();
            }
            for tool in tools {
                final_system_prompt.push_str("```json\n");
                final_system_prompt.push_str(&indent_go_json(&go_tool(tool), "    "));
                final_system_prompt.push_str("\n```\n");
            }
        }

        sb.push_str(BOS);
        sb.push_str(&final_system_prompt);

        let mut outputs_open = false;
        let mut is_last_user = false;

        for (i, message) in conversation.iter().enumerate() {
            match message.role.as_str() {
                "user" => {
                    is_last_user = true;
                    sb.push_str("<｜User｜>");
                    sb.push_str(&message.content);
                    sb.push_str("<｜Assistant｜>");
                }
                "assistant" => {
                    is_last_user = false;
                    if !message.tool_calls.is_empty() {
                        sb.push_str(&message.content);
                        sb.push_str("<｜tool▁calls▁begin｜>");
                        for (j, tc) in message.tool_calls.iter().enumerate() {
                            sb.push_str(&format!(
                                "<｜tool▁call▁begin｜>function<｜tool▁sep｜>{}",
                                tc.function.name
                            ));
                            sb.push_str(&format!(
                                "\n```json\n{}\n```",
                                go_arguments(&tc.function.arguments)
                            ));
                            sb.push_str("<｜tool▁call▁end｜>");
                            if j < message.tool_calls.len() - 1 {
                                sb.push('\n');
                            }
                        }
                        sb.push_str("<｜tool▁calls▁end｜>");
                        sb.push_str(EOS);
                    } else {
                        sb.push_str(&message.content);
                        sb.push_str(EOS);
                    }
                }
                "tool" => {
                    is_last_user = false;
                    if !outputs_open {
                        sb.push_str("<｜tool▁outputs▁begin｜>");
                        outputs_open = true;
                    }
                    sb.push_str("<｜tool▁output▁begin｜>");
                    sb.push_str(&message.content);
                    sb.push_str("<｜tool▁output▁end｜>");

                    let has_next_tool = conversation
                        .get(i + 1)
                        .is_some_and(|next| next.role == "tool");
                    if has_next_tool {
                        sb.push('\n');
                    } else {
                        sb.push_str("<｜tool▁outputs▁end｜>");
                        outputs_open = false;
                    }
                }
                _ => {}
            }
        }

        if outputs_open {
            sb.push_str("<｜tool▁outputs▁end｜>");
        }

        // A user turn already wrote its own `<｜Assistant｜>`.
        if !is_last_user {
            sb.push_str("<｜Assistant｜>");
        }

        if enable_thinking {
            sb.push_str("<think>\n");
        }

        Ok(sb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ToolCall, ToolCallFunction};
    use serde_json::json;

    fn renderer() -> CogitoRenderer {
        // Upstream's `cogito_test.go` builds exactly this.
        CogitoRenderer { is_thinking: true }
    }

    fn render(msgs: &[Message], tools: &[Tool], think: bool) -> String {
        renderer()
            .render(msgs, tools, Some(&ThinkValue::Bool(think)))
            .unwrap()
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

    /// Upstream `TestCogitoRenderer`'s tool-definition fixture -- the one that
    /// pins the four-space `MarshalIndent` layout inside the ```json fence.
    #[test]
    fn tool_definitions_are_pretty_printed_inside_a_json_fence() {
        let tools: Vec<Tool> = vec![
            serde_json::from_value(json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "required": ["location"],
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        }
                    }
                }
            }))
            .unwrap(),
        ];
        let got = render(
            &[Message::new("user", "What's the weather like?")],
            &tools,
            false,
        );
        assert_eq!(
            got,
            concat!(
                "<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.\n",
                "You have the following functions available:\n",
                "```json\n",
                "{\n",
                "    \"type\": \"function\",\n",
                "    \"function\": {\n",
                "        \"name\": \"get_weather\",\n",
                "        \"description\": \"Get current weather\",\n",
                "        \"parameters\": {\n",
                "            \"type\": \"object\",\n",
                "            \"required\": [\n",
                "                \"location\"\n",
                "            ],\n",
                "            \"properties\": {\n",
                "                \"location\": {\n",
                "                    \"type\": \"string\",\n",
                "                    \"description\": \"City name\"\n",
                "                }\n",
                "            }\n",
                "        }\n",
                "    }\n",
                "}\n",
                "```\n",
                "<｜User｜>What's the weather like?<｜Assistant｜>",
            )
        );
    }

    /// Upstream `TestCogitoRenderer/assistant with tool calls` and
    /// `/tool response`.
    #[test]
    fn tool_calls_and_outputs_match_upstream() {
        let head = "<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.";

        assert_eq!(
            render(
                &[
                    Message::new("user", "What's the weather in Paris?"),
                    Message {
                        role: "assistant".into(),
                        content: "I'll check the weather in Paris for you.".into(),
                        tool_calls: vec![call("get_weather", r#"{"location":"Paris"}"#)],
                        ..Default::default()
                    },
                ],
                &[],
                false
            ),
            format!(
                "{head}<｜User｜>What's the weather in Paris?<｜Assistant｜>I'll check the weather in Paris for you.\
                 <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n\
                 ```json\n{{\"location\":\"Paris\"}}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜Assistant｜>"
            )
        );

        // A tool output closes its own `<｜tool▁outputs▁...｜>` wrapper, then
        // the generation prompt is appended because the last turn was not a
        // user turn.
        let got = render(
            &[
                Message::new("user", "What's the weather in Paris?"),
                Message {
                    role: "assistant".into(),
                    tool_calls: vec![call("get_weather", r#"{"location":"Paris"}"#)],
                    ..Default::default()
                },
                Message::new("tool", "Temperature: 22°C, Sunny"),
            ],
            &[],
            false,
        );
        assert!(
            got.ends_with("<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>Temperature: 22°C, Sunny<｜tool▁output▁end｜><｜tool▁outputs▁end｜><｜Assistant｜>"),
            "{got}"
        );
    }

    /// Thinking rewrites the SYSTEM prompt, not just the tail -- and wraps a
    /// caller's system text in blank lines on both sides.
    #[test]
    fn thinking_prefixes_the_system_prompt_and_opens_a_think_tag() {
        let got = render(
            &[
                Message::new("system", "Be terse."),
                Message::new("user", "hi"),
            ],
            &[],
            true,
        );
        assert_eq!(
            got,
            format!(
                "<｜begin▁of▁sentence｜>Enable deep thinking subroutine.\n\n{DEFAULT_PROMPT}\n\nBe terse.\n\n<｜User｜>hi<｜Assistant｜><think>\n"
            )
        );
    }
}
