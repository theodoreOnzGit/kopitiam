//! **GLM-4.6** -- the newline-heavy older sibling of [`super::glm47`].
//!
//! **Upstream:** `model/renderers/glm46.go`.
//!
//! ## Not reachable by name, and that is upstream's doing
//!
//! `rendererForName` has **no arm for GLM-4.6** -- the type exists and is
//! tested, but nothing constructs it from a model config. Ported anyway,
//! because the tests are the spec and the family may come back; a caller who
//! wants it constructs [`Glm46Renderer`] directly.
//!
//! ## What differs from GLM-4.7 (all of it whitespace, all of it load-bearing)
//!
//! Same special tokens -- **`[gMASK]`**, **`<sop>`**, **`<|system|>`**,
//! **`<|user|>`**, **`<|assistant|>`**, **`<|observation|>`**,
//! **`<think>`/`</think>`** -- but:
//!
//! * every role marker is followed by a **newline** (`<|user|>\n`), where 4.7
//!   runs the content straight on;
//! * the tool-call format instruction is **multi-line**, not the one-liner;
//! * a `<think>` block appears **only after the last user message**
//!   (`i > last_user_index`), and an assistant turn with no reasoning there gets
//!   `\n<think></think>` -- an *empty* pair, not 4.7's bare closing tag;
//! * turning thinking off appends `/nothink` to the **user's own text**
//!   (unless they already typed it), rather than only changing the generation
//!   prompt. That is how this family was fine-tuned to be told "skip it".

use serde_json::Value;

use super::json::{go_tool, go_value};
use super::{Message, RenderError, Renderer, ThinkValue, Tool};

/// **Upstream:** `GLM46Renderer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Glm46Renderer;

/// The multi-line call-format instruction, verbatim from `glm46.go`.
const GLM46_CALL_FORMAT: &str = "For each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n<arg_value>{arg-value-1}</arg_value>\n<arg_key>{arg-key-2}</arg_key>\n<arg_value>{arg-value-2}</arg_value>\n...\n</tool_call>";

impl Renderer for Glm46Renderer {
    fn render(
        &self,
        messages: &[Message],
        tools: &[Tool],
        think: Option<&ThinkValue>,
    ) -> Result<String, RenderError> {
        let mut sb = String::from("[gMASK]<sop>");

        // Index of the LAST user message. Note upstream leaves this at 0 when
        // there is no user message at all, which we copy rather than "fix".
        let mut last_user_index = 0usize;
        for (i, m) in messages.iter().enumerate() {
            if m.role == "user" {
                last_user_index = i;
            }
        }

        if !tools.is_empty() {
            sb.push_str("<|system|>\n");
            sb.push_str(super::glm47::GLM_TOOLS_HEADER);
            for tool in tools {
                // GLM-4.6 uses PLAIN `json.Marshal` -- no spaces after `:`/`,`,
                // unlike 4.7. Adding them here would change the prompt.
                sb.push_str(&go_tool(tool));
                sb.push('\n');
            }
            sb.push_str("</tools>\n\n");
            sb.push_str(GLM46_CALL_FORMAT);
        }

        let think_off = matches!(think, Some(t) if !t.enabled());

        for (i, message) in messages.iter().enumerate() {
            match message.role.as_str() {
                "user" => {
                    sb.push_str("<|user|>\n");
                    sb.push_str(&message.content);
                    if think_off && !message.content.ends_with("/nothink") {
                        sb.push_str("/nothink");
                    }
                }
                "assistant" => {
                    sb.push_str("<|assistant|>");
                    if i > last_user_index {
                        if !message.thinking.is_empty() {
                            sb.push_str(&format!("\n<think>{}</think>", message.thinking));
                        } else {
                            sb.push_str("\n<think></think>");
                        }
                    }
                    if !message.content.is_empty() {
                        sb.push('\n');
                        sb.push_str(&message.content);
                    }
                    for tc in &message.tool_calls {
                        sb.push_str(&format!("\n<tool_call>{}\n", tc.function.name));
                        for (key, value) in &tc.function.arguments.0 {
                            sb.push_str(&format!("<arg_key>{key}</arg_key>\n"));
                            let value_str = match value {
                                Value::String(s) => s.clone(),
                                other => go_value(other),
                            };
                            sb.push_str(&format!("<arg_value>{value_str}</arg_value>\n"));
                        }
                        sb.push_str("</tool_call>");
                    }
                }
                "tool" => {
                    if i == 0 || messages[i - 1].role != "tool" {
                        sb.push_str("<|observation|>");
                    }
                    sb.push_str("\n<tool_response>\n");
                    sb.push_str(&message.content);
                    sb.push_str("\n</tool_response>");
                }
                "system" => {
                    sb.push_str("<|system|>\n");
                    sb.push_str(&message.content);
                }
                _ => {}
            }
        }

        sb.push_str("<|assistant|>");
        if think_off {
            sb.push_str("\n<think></think>\n");
        }

        Ok(sb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream `TestGLM46Renderer`, the three cases that are **not** marked
    /// `skip` there. Upstream skips its own tool fixtures with the note *"tool
    /// call ordering not guaranteed yet"* -- our [`super::json`] emitter is
    /// deterministic, but we do not invent expectations upstream itself refuses
    /// to assert.
    #[test]
    fn glm46_matches_the_upstream_fixtures() {
        let got = Glm46Renderer
            .render(&[Message::new("user", "Hello, how are you?")], &[], None)
            .unwrap();
        assert_eq!(
            got,
            "[gMASK]<sop><|user|>\nHello, how are you?<|assistant|>"
        );

        let got = Glm46Renderer
            .render(
                &[
                    Message::new("system", "You are a helpful assistant."),
                    Message::new("user", "Hello, how are you?"),
                ],
                &[],
                None,
            )
            .unwrap();
        assert_eq!(
            got,
            "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\nHello, how are you?<|assistant|>"
        );

        // The assistant turn sits BEFORE the last user message, so its
        // reasoning is dropped -- no `<think>` pair at all.
        let got = Glm46Renderer
            .render(
                &[
                    Message::new("user", "What is the capital of France?"),
                    Message {
                        role: "assistant".into(),
                        thinking: "Let me analyze the request...".into(),
                        content: "The capital of France is Paris.".into(),
                        ..Default::default()
                    },
                    Message::new("user", "Fantastic!"),
                ],
                &[],
                None,
            )
            .unwrap();
        assert_eq!(
            got,
            "[gMASK]<sop><|user|>\nWhat is the capital of France?<|assistant|>\nThe capital of France is Paris.<|user|>\nFantastic!<|assistant|>"
        );
    }

    /// Not an upstream fixture: pins the `/nothink` behaviour, which upstream
    /// states only in code. It must go on the **user's text**, and must not be
    /// doubled up when the user already typed it.
    #[test]
    fn thinking_off_appends_nothink_to_the_user_turn_exactly_once() {
        let got = Glm46Renderer
            .render(
                &[Message::new("user", "hi")],
                &[],
                Some(&ThinkValue::Bool(false)),
            )
            .unwrap();
        assert_eq!(
            got,
            "[gMASK]<sop><|user|>\nhi/nothink<|assistant|>\n<think></think>\n"
        );

        let got = Glm46Renderer
            .render(
                &[Message::new("user", "hi/nothink")],
                &[],
                Some(&ThinkValue::Bool(false)),
            )
            .unwrap();
        assert!(!got.contains("/nothink/nothink"), "{got}");
    }
}
