//! **Qwen3-VL** (instruct and thinking) -- ChatML with images and JSON tool
//! calls.
//!
//! **Upstream:** `model/renderers/qwen3vl.go`. Registered as
//! `qwen3-vl-instruct` (`is_thinking: false`) and `qwen3-vl-thinking`
//! (`is_thinking: true`).
//!
//! ## The framing
//!
//! Special tokens: **`<|im_start|>`**, **`<|im_end|>`**, the thinking pair
//! **`<think>` / `</think>`**, and the vision triple
//! **`<|vision_start|><|image_pad|><|vision_end|>`** -- one per image, at the
//! **front** of the message (same assumption as ollama's `runner.go`). No BOS.
//!
//! ## Where it differs from [`super::qwen35`], which it otherwise resembles
//!
//! * **Tool calls are JSON, not XML.** `<tool_call>\n{"name": "f", "arguments":
//!   {...}}\n</tool_call>` -- and both the tool schemas *and* the arguments go
//!   through the `', '` / `': '` spacing pass, so they read like Python's
//!   `json.dumps` output.
//! * **The system message comes FIRST, tools second** -- the reverse of
//!   Qwen3.5, which appends the caller's system text after the tool block.
//! * **Content is not trimmed.** Qwen3.5 runs `TrimSpace` over every message;
//!   this one does not. So the `<tool_response>` prefix/suffix check for the
//!   `last_query_index` scan runs against untrimmed text here, which is
//!   upstream's behaviour and is copied deliberately rather than "harmonised".
//! * The assistant think-block rule has an extra guard: a `<think>` block is
//!   opened only when this is the **last** message or there is actually some
//!   reasoning to put in it.

use super::image_tags::render_content_with_image_tags;
use super::json::{add_spaces_outside_strings, go_arguments, marshal_tool_with_spaces};
use super::{IM_END_TAG, IM_START_TAG, Message, RenderError, Renderer, ThinkValue, Tool};

/// **Upstream:** `Qwen3VLRenderer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen3VLRenderer {
    /// The model's own default. A caller's explicit `think` overrides it.
    pub is_thinking: bool,
    /// Use `[img-N]` markers instead of the native vision tokens.
    pub use_img_tags: bool,
}

/// The tools preamble, verbatim from `qwen3vl.go`.
const VL_TOOLS_HEADER: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
/// The call-format instruction, verbatim, `<|im_end|>\n` included.
const VL_TOOLS_FOOTER: &str = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";

impl Qwen3VLRenderer {
    /// **Upstream:** `(*Qwen3VLRenderer).renderContent`.
    fn render_content(&self, message: &Message, image_offset: usize) -> (String, usize) {
        if self.use_img_tags {
            return render_content_with_image_tags(
                &message.content,
                message.images.len(),
                image_offset,
            );
        }
        let mut s = String::new();
        for _ in &message.images {
            // TODO(upstream, jmorganca): how an image is spelled differs per
            // backend; this may become a parameter or a plain `[img]`.
            s.push_str("<|vision_start|><|image_pad|><|vision_end|>");
        }
        s.push_str(&message.content);
        (s, image_offset)
    }
}

impl Renderer for Qwen3VLRenderer {
    fn render(
        &self,
        messages: &[Message],
        tools: &[Tool],
        think: Option<&ThinkValue>,
    ) -> Result<String, RenderError> {
        let mut sb = String::new();

        let is_thinking = match think {
            Some(t) => t.enabled(),
            None => self.is_thinking,
        };

        if !tools.is_empty() {
            sb.push_str(IM_START_TAG);
            sb.push_str("system\n");
            // System text FIRST here, unlike Qwen3.5.
            if let Some(first) = messages.first()
                && first.role == "system"
            {
                sb.push_str(&first.content);
                sb.push_str("\n\n");
            }
            sb.push_str(VL_TOOLS_HEADER);
            for tool in tools {
                sb.push('\n');
                sb.push_str(&marshal_tool_with_spaces(tool));
            }
            sb.push_str(VL_TOOLS_FOOTER);
        } else if let Some(first) = messages.first()
            && first.role == "system"
        {
            sb.push_str(IM_START_TAG);
            sb.push_str("system\n");
            sb.push_str(&first.content);
            sb.push_str(IM_END_TAG);
            sb.push('\n');
        }

        // Same "which user message is a real query" scan as Qwen3.5 -- but
        // against UNTRIMMED content. See the module docs.
        let mut multi_step_tool = true;
        let mut last_query_index = messages.len().saturating_sub(1);
        for (i, message) in messages.iter().enumerate().rev() {
            if multi_step_tool && message.role == "user" {
                let (content, _) = self.render_content(message, 0);
                if !(content.starts_with("<tool_response>")
                    && content.ends_with("</tool_response>"))
                {
                    multi_step_tool = false;
                    last_query_index = i;
                }
            }
        }

        let mut image_offset = 0usize;
        for (i, message) in messages.iter().enumerate() {
            let (content, next_image_offset) = self.render_content(message, image_offset);
            image_offset = next_image_offset;

            let last_message = i == messages.len() - 1;
            let prefill = last_message && message.role == "assistant";

            if message.role == "user" || (message.role == "system" && i != 0) {
                sb.push_str(IM_START_TAG);
                sb.push_str(&message.role);
                sb.push('\n');
                sb.push_str(&content);
                sb.push_str(IM_END_TAG);
                sb.push('\n');
            } else if message.role == "assistant" {
                let reasoning = if is_thinking { &message.thinking } else { "" };

                if is_thinking && i > last_query_index && (last_message || !reasoning.is_empty()) {
                    sb.push_str(IM_START_TAG);
                    sb.push_str(&message.role);
                    sb.push_str("\n<think>\n");
                    sb.push_str(reasoning.trim_matches('\n'));
                    if !content.is_empty() {
                        sb.push_str("\n</think>\n\n");
                        sb.push_str(content.trim_start_matches('\n'));
                    }
                } else {
                    sb.push_str(IM_START_TAG);
                    sb.push_str(&message.role);
                    sb.push('\n');
                    sb.push_str(&content);
                }

                for (j, tc) in message.tool_calls.iter().enumerate() {
                    if j > 0 || !content.is_empty() {
                        sb.push('\n');
                    }
                    sb.push_str(&format!(
                        "<tool_call>\n{{\"name\": \"{}\", \"arguments\": ",
                        tc.function.name
                    ));
                    sb.push_str(&add_spaces_outside_strings(&go_arguments(
                        &tc.function.arguments,
                    )));
                    sb.push_str("}\n</tool_call>");
                }

                if !prefill {
                    sb.push_str(IM_END_TAG);
                    sb.push('\n');
                }
            } else if message.role == "tool" {
                if i == 0 || messages[i - 1].role != "tool" {
                    sb.push_str(IM_START_TAG);
                    sb.push_str("user");
                }
                sb.push_str("\n<tool_response>\n");
                sb.push_str(&content);
                sb.push_str("\n</tool_response>");
                if i == messages.len() - 1 || messages[i + 1].role != "tool" {
                    sb.push_str(IM_END_TAG);
                    sb.push('\n');
                }
            }

            if last_message && !prefill {
                sb.push_str(IM_START_TAG);
                sb.push_str("assistant\n");
                if is_thinking {
                    sb.push_str("<think>\n");
                }
                // NOTE: upstream has an `emitEmptyThinkOnNoThink` branch here
                // too, but no registered `qwen3-vl-*` preset ever sets it, so
                // the field is not modelled. If a future preset needs it, add
                // the flag rather than the branch.
            }
        }

        Ok(sb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ToolCall, ToolCallFunction};
    use serde_json::json;

    fn instruct() -> Qwen3VLRenderer {
        Qwen3VLRenderer::default()
    }

    fn thinking() -> Qwen3VLRenderer {
        Qwen3VLRenderer {
            is_thinking: true,
            use_img_tags: false,
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

    #[test]
    fn a_plain_exchange_renders_as_chatml() {
        assert_eq!(
            instruct()
                .render(
                    &[
                        Message::new("system", "You are helpful."),
                        Message::new("user", "Hello"),
                    ],
                    &[],
                    None
                )
                .unwrap(),
            "<|im_start|>system\nYou are helpful.<|im_end|>\n\
             <|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    /// The system message comes BEFORE the tools block here -- the reverse of
    /// Qwen3.5, and the thing most likely to be got backwards when porting.
    #[test]
    fn the_system_message_precedes_the_tools_block() {
        let got = instruct()
            .render(
                &[
                    Message::new("system", "You are helpful."),
                    Message::new("user", "Weather?"),
                ],
                &[weather_tool()],
                None,
            )
            .unwrap();
        let sys = got.find("You are helpful.").expect("system text");
        let tools = got.find("# Tools").expect("tools block");
        assert!(sys < tools, "system must come first:\n{got}");
        assert!(
            got.contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"required\": [\"location\"], \"properties\": {\"location\": {\"type\": \"string\"}}}}}\n</tools>"),
            "{got}"
        );
    }

    /// Tool calls are JSON here, spaced like `json.dumps`.
    #[test]
    fn tool_calls_are_spaced_json_inside_tool_call_tags() {
        let got = instruct()
            .render(
                &[
                    Message::new("user", "Weather?"),
                    Message {
                        role: "assistant".into(),
                        tool_calls: vec![ToolCall {
                            id: String::new(),
                            function: ToolCallFunction {
                                index: 0,
                                name: "get_weather".into(),
                                arguments: serde_json::from_str(
                                    r#"{"location":"Paris","unit":"celsius"}"#,
                                )
                                .unwrap(),
                            },
                        }],
                        ..Default::default()
                    },
                    Message::new("tool", "22C"),
                ],
                &[],
                None,
            )
            .unwrap();
        assert!(
            got.contains(
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\", \"unit\": \"celsius\"}}\n</tool_call>"
            ),
            "{got}"
        );
        assert!(
            got.contains("<|im_start|>user\n<tool_response>\n22C\n</tool_response><|im_end|>\n"),
            "{got}"
        );
    }

    #[test]
    fn the_thinking_variant_opens_a_think_block_in_the_generation_prompt() {
        let got = thinking()
            .render(&[Message::new("user", "Hello")], &[], None)
            .unwrap();
        assert!(got.ends_with("<|im_start|>assistant\n<think>\n"), "{got}");

        // An explicit `false` overrides the model default.
        let got = thinking()
            .render(
                &[Message::new("user", "Hello")],
                &[],
                Some(&ThinkValue::Bool(false)),
            )
            .unwrap();
        assert!(got.ends_with("<|im_start|>assistant\n"), "{got}");
        assert!(!got.contains("<think>"), "{got}");
    }

    #[test]
    fn images_render_as_vision_tokens_at_the_front_of_the_message() {
        let got = instruct()
            .render(
                &[Message {
                    role: "user".into(),
                    content: "describe".into(),
                    images: vec!["a".into(), "b".into()],
                    ..Default::default()
                }],
                &[],
                None,
            )
            .unwrap();
        assert!(
            got.contains("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>describe<|im_end|>\n"),
            "{got}"
        );
    }
}
