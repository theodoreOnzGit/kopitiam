//! **GLM-4.7** -- `[gMASK]<sop>` prelude, `<|role|>` turn markers, thinking on
//! by default.
//!
//! **Upstream:** `model/renderers/glm47.go`. Registered as `glm-4.7`.
//!
//! ## The framing
//!
//! Special tokens: **`[gMASK]`**, **`<sop>`**, **`<|system|>`**, **`<|user|>`**,
//! **`<|assistant|>`**, **`<|observation|>`** (that last one is where tool
//! results go, not `user`), plus **`<think>` / `</think>`** and the tool-call
//! shape `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`.
//! No BOS token -- the `[gMASK]<sop>` prelude is written into the prompt
//! itself.
//!
//! ## Thinking modes (upstream's own notes, kept because they explain the code)
//!
//! GLM-4.7 has three separate thinking knobs upstream
//! (<https://docs.z.ai/guides/capabilities/thinking-mode>):
//!
//! 1. **Interleaved** -- the model thinks between tool calls and after each tool
//!    result, so reasoning carries across a multi-step tool run.
//! 2. **Preserved** -- reasoning from earlier assistant turns stays in context
//!    (`clear_thinking=false` upstream).
//! 3. **Turn-level** -- whether to reason on *this* turn at all
//!    (`enable_thinking`).
//!
//! **ollama's choices, which we copy exactly:** thinking is **on** by default
//! (`think` of `None` or `true` ends the prompt with `<think>`), reasoning is
//! **always preserved** (every historical `Message::thinking` is re-emitted in
//! its `<think>...</think>` block), and a caller can turn it off per turn with
//! `think = Some(false)`.
//!
//! The tell-tale detail: **an assistant turn with no reasoning still emits a
//! bare `</think>`**. That closing-tag-without-an-opening-tag looks like a bug
//! and is not -- it is how this family signals "this turn did no thinking", and
//! omitting it changes the framing.

use serde_json::Value;

use super::json::{add_spaces_outside_strings, go_tool, go_value};
use super::{Message, RenderError, Renderer, ThinkValue, Tool};
use crate::api::ToolCallArguments;

/// **Upstream:** `GLM47Renderer`. Carries no configuration.
#[derive(Debug, Clone, Copy, Default)]
pub struct Glm47Renderer;

/// `<arg_key>k</arg_key><arg_value>v</arg_value>` for each argument, in the
/// order the model gave them.
///
/// **Upstream:** `renderGLM47ToolArguments`. A **string** value goes in raw
/// (`Tokyo`, not `"Tokyo"`); everything else is JSON-encoded. Quoting the
/// string would train the parser on the wrong shape coming back.
pub(crate) fn render_tool_arguments(args: &ToolCallArguments) -> String {
    let mut sb = String::new();
    for (key, value) in &args.0 {
        sb.push_str(&format!("<arg_key>{key}</arg_key>"));
        let value_str = match value {
            Value::String(s) => s.clone(),
            other => go_value(other),
        };
        sb.push_str(&format!("<arg_value>{value_str}</arg_value>"));
    }
    sb
}

/// The tools preamble shared by GLM-4.6, GLM-4.7 and GLM-OCR.
pub(crate) const GLM_TOOLS_HEADER: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";

/// The one-line call-format instruction GLM-4.7 and GLM-OCR share (GLM-4.6 uses
/// a multi-line variant of its own).
pub(crate) const GLM_INLINE_CALL_FORMAT: &str = "For each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";

impl Renderer for Glm47Renderer {
    fn render(
        &self,
        messages: &[Message],
        tools: &[Tool],
        think: Option<&ThinkValue>,
    ) -> Result<String, RenderError> {
        let mut sb = String::from("[gMASK]<sop>");

        if !tools.is_empty() {
            sb.push_str("<|system|>\n");
            sb.push_str(GLM_TOOLS_HEADER);
            for tool in tools {
                // `formatGLM47ToolJSON` is byte-for-byte `marshalWithSpaces`.
                sb.push_str(&add_spaces_outside_strings(&go_tool(tool)));
                sb.push('\n');
            }
            sb.push_str("</tools>\n\n");
            sb.push_str(GLM_INLINE_CALL_FORMAT);
        }

        // On unless explicitly switched off -- note `None` means ON here, the
        // opposite of the DeepSeek family.
        let think_on = !matches!(think, Some(t) if !t.enabled());

        for (i, message) in messages.iter().enumerate() {
            match message.role.as_str() {
                "user" => {
                    sb.push_str("<|user|>");
                    sb.push_str(&message.content);
                }
                "assistant" => {
                    sb.push_str("<|assistant|>");
                    if !message.thinking.is_empty() {
                        sb.push_str(&format!("<think>{}</think>", message.thinking));
                    } else {
                        // Bare closing tag on purpose -- see the module docs.
                        sb.push_str("</think>");
                    }
                    if !message.content.is_empty() {
                        sb.push_str(&message.content);
                    }
                    for tc in &message.tool_calls {
                        sb.push_str(&format!("<tool_call>{}", tc.function.name));
                        sb.push_str(&render_tool_arguments(&tc.function.arguments));
                        sb.push_str("</tool_call>");
                    }
                }
                "tool" => {
                    // Consecutive tool results share one `<|observation|>`.
                    if i == 0 || messages[i - 1].role != "tool" {
                        sb.push_str("<|observation|>");
                    }
                    sb.push_str("<tool_response>");
                    sb.push_str(&message.content);
                    sb.push_str("</tool_response>");
                }
                "system" => {
                    sb.push_str("<|system|>");
                    sb.push_str(&message.content);
                }
                _ => {}
            }
        }

        sb.push_str("<|assistant|>");
        sb.push_str(if think_on { "<think>" } else { "</think>" });

        Ok(sb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ToolCall, ToolCallFunction};
    use serde_json::json;

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

    fn weather_tool() -> Tool {
        serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "required": ["location"],
                    "properties": {"location": {"type": "string"}}
                }
            }
        }))
        .expect("valid fixture tool")
    }

    /// The exact tools block from upstream's fixtures. Worth its own constant:
    /// it also pins that our Go-shaped JSON emitter agrees with Go's
    /// (`"type": "string"`, not `"type": ["string"]`).
    const TOOLS_BLOCK: &str = concat!(
        "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n",
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
        "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get weather\", ",
        "\"parameters\": {\"type\": \"object\", \"required\": [\"location\"], \"properties\": {\"location\": {\"type\": \"string\"}}}}}\n",
        "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n",
        "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>",
        "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>",
    );

    /// Upstream `TestGLM47Renderer`, all eight cases.
    #[test]
    fn glm47_matches_every_upstream_fixture() {
        /// One upstream fixture row: name, messages, tools, think, expected.
        type Case = (
            &'static str,
            Vec<Message>,
            Vec<Tool>,
            Option<ThinkValue>,
            String,
        );

        let cases: Vec<Case> = vec![
            (
                "basic user message",
                vec![Message::new("user", "Hello")],
                vec![],
                None,
                "[gMASK]<sop><|user|>Hello<|assistant|><think>".into(),
            ),
            (
                "thinking disabled",
                vec![Message::new("user", "Hello")],
                vec![],
                Some(ThinkValue::Bool(false)),
                "[gMASK]<sop><|user|>Hello<|assistant|></think>".into(),
            ),
            (
                "system and user",
                vec![
                    Message::new("system", "You are helpful."),
                    Message::new("user", "Hello"),
                ],
                vec![],
                None,
                "[gMASK]<sop><|system|>You are helpful.<|user|>Hello<|assistant|><think>".into(),
            ),
            (
                "multi-turn conversation",
                vec![
                    Message::new("user", "Hi"),
                    Message::new("assistant", "Hello there"),
                    Message::new("user", "How are you?"),
                ],
                vec![],
                None,
                "[gMASK]<sop><|user|>Hi<|assistant|></think>Hello there<|user|>How are you?<|assistant|><think>".into(),
            ),
            (
                "assistant with reasoning_content",
                vec![
                    Message::new("user", "Answer with reasoning."),
                    Message {
                        role: "assistant".into(),
                        thinking: "Plan.".into(),
                        content: "Done.".into(),
                        ..Default::default()
                    },
                ],
                vec![],
                None,
                "[gMASK]<sop><|user|>Answer with reasoning.<|assistant|><think>Plan.</think>Done.<|assistant|><think>".into(),
            ),
            (
                "tool call with empty content",
                vec![
                    Message::new("user", "Weather?"),
                    Message {
                        role: "assistant".into(),
                        tool_calls: vec![call(
                            "get_weather",
                            r#"{"location": "Tokyo", "unit": "celsius"}"#,
                        )],
                        ..Default::default()
                    },
                    Message::new("tool", r#"{"temperature":22}"#),
                ],
                vec![weather_tool()],
                None,
                format!(
                    "[gMASK]<sop>{TOOLS_BLOCK}<|user|>Weather?<|assistant|></think>\
                     <tool_call>get_weather<arg_key>location</arg_key><arg_value>Tokyo</arg_value>\
                     <arg_key>unit</arg_key><arg_value>celsius</arg_value></tool_call>\
                     <|observation|><tool_response>{{\"temperature\":22}}</tool_response>\
                     <|assistant|><think>"
                ),
            ),
            (
                "tool call with content",
                vec![
                    Message::new("user", "Weather?"),
                    Message {
                        role: "assistant".into(),
                        content: "Let me check".into(),
                        tool_calls: vec![call("get_weather", r#"{"location": "Tokyo"}"#)],
                        ..Default::default()
                    },
                    Message::new("tool", r#"{"temperature":22}"#),
                    Message::new("assistant", "It is 22C."),
                ],
                vec![weather_tool()],
                None,
                format!(
                    "[gMASK]<sop>{TOOLS_BLOCK}<|user|>Weather?<|assistant|></think>Let me check\
                     <tool_call>get_weather<arg_key>location</arg_key><arg_value>Tokyo</arg_value></tool_call>\
                     <|observation|><tool_response>{{\"temperature\":22}}</tool_response>\
                     <|assistant|></think>It is 22C.<|assistant|><think>"
                ),
            ),
            (
                "multiple tool calls and responses",
                vec![
                    Message::new("user", "Compare weather"),
                    Message {
                        role: "assistant".into(),
                        tool_calls: vec![
                            call("get_weather", r#"{"location": "Tokyo"}"#),
                            call("get_weather", r#"{"location": "Paris"}"#),
                        ],
                        ..Default::default()
                    },
                    Message::new("tool", r#"{"temperature":22}"#),
                    Message::new("tool", r#"{"temperature":18}"#),
                ],
                vec![weather_tool()],
                None,
                format!(
                    "[gMASK]<sop>{TOOLS_BLOCK}<|user|>Compare weather<|assistant|></think>\
                     <tool_call>get_weather<arg_key>location</arg_key><arg_value>Tokyo</arg_value></tool_call>\
                     <tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>\
                     <|observation|><tool_response>{{\"temperature\":22}}</tool_response>\
                     <tool_response>{{\"temperature\":18}}</tool_response><|assistant|><think>"
                ),
            ),
            (
                "preserved thinking in multi-turn",
                vec![
                    Message::new("user", "Think step by step"),
                    Message {
                        role: "assistant".into(),
                        thinking: "Let me think...".into(),
                        content: "Here's my answer.".into(),
                        ..Default::default()
                    },
                    Message::new("user", "Continue"),
                ],
                vec![],
                None,
                "[gMASK]<sop><|user|>Think step by step<|assistant|><think>Let me think...</think>Here's my answer.<|user|>Continue<|assistant|><think>".into(),
            ),
        ];

        for (name, msgs, tools, think, want) in cases {
            let got = Glm47Renderer.render(&msgs, &tools, think.as_ref()).unwrap();
            assert_eq!(got, want, "case: {name}");
        }
    }
}
