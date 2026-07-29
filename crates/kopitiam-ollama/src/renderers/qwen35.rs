//! **Qwen3.5** (and **Ornith**, same renderer with a different preset).
//!
//! **Upstream:** `model/renderers/qwen35.go` + `model/renderers/ornith.go`.
//! Registered as `qwen3.5` and `ornith`.
//!
//! ## The framing
//!
//! Special tokens: **`<|im_start|>`**, **`<|im_end|>`**, the thinking pair
//! **`<think>`** / **`</think>`**, and -- when images go in natively rather than
//! as `[img-N]` markers -- **`<|vision_start|><|image_pad|><|vision_end|>`**,
//! one triple per image, always at the **front** of the message (the same
//! assumption ollama's own `runner.go` makes). No BOS.
//!
//! Where Qwen3.5 differs from Qwen3-Coder, and it is not cosmetic:
//!
//! * **Tools come first, system message second.** The `# Tools` block opens the
//!   system turn, and the caller's system text is appended *after* the
//!   postamble. Upstream's own test asserts the order (`systemIdx > toolsIdx`).
//! * **Tools are JSON, not XML** -- each tool marshalled with spaces after `:`
//!   and `,` so it matches Python's `json.dumps` (see
//!   [`super::json::add_spaces_outside_strings`]). The *call* syntax is still
//!   the XML `<tool_call><function=..><parameter=..>` shape though.
//! * **Thinking is per-turn and position-dependent.** Only assistant messages
//!   *after* the last real user query get a `<think>` block; historical
//!   reasoning from earlier turns is dropped. That is what `last_query_index`
//!   below computes, and it is why the "back to back tool calls" fixture
//!   asserts the old reasoning is **absent**.
//!
//! ## The `multi_step_tool` scan, explained
//!
//! Walking backwards, a user message whose whole content is
//! `<tool_response>...</tool_response>` does **not** count as a real query --
//! it is a tool result wearing a user costume (that is how this family carries
//! tool output). `last_query_index` therefore lands on the last *human* turn,
//! and every assistant message after it is "the current turn" and keeps its
//! reasoning. Break this and a multi-step tool conversation loses its thinking
//! blocks halfway through.

use super::image_tags::render_content_with_image_tags;
use super::json::marshal_tool_with_spaces;
use super::qwen3coder::format_tool_call_argument;
use super::{IM_END_TAG, IM_START_TAG, Message, RenderError, Renderer, ThinkValue, Tool};

/// **Upstream:** `model/renderers/qwen35.go`.
///
/// Plain ASCII, and a single token in Qwen's vocabulary. `crate::parsers::qwen35`
/// reads the same pair back out; the two must agree exactly and nothing checks
/// that they do.
const THINK_OPEN_TAG: &str = "<think>";
/// **Upstream:** `model/renderers/qwen35.go`. Pairs with [`THINK_OPEN_TAG`].
const THINK_CLOSE_TAG: &str = "</think>";

/// The instructions appended after the `<tools>` list.
///
/// **Upstream:** `qwen35ToolPostamble` in `qwen35.go`. Note it **opens with
/// `\n</tools>`** -- closing the list is part of the constant, not the caller's
/// job. Copied byte for byte; do not reflow it.
const QWEN35_TOOL_POSTAMBLE: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// **Upstream:** `Qwen35Renderer`.
///
/// The four flags are upstream's, and the two presets that use them are:
///
/// | Name | `is_thinking` | `always_render_...` | `emit_empty_think_...` |
/// |---|---|---|---|
/// | `qwen3.5` | `true` | `false` | `true` |
/// | `ornith` | `true` | `true` | `true` |
#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen35Renderer {
    /// The model's own default: does it think unless told otherwise? A caller's
    /// explicit `think` argument overrides this.
    pub is_thinking: bool,
    /// Render a `<think>` block on **every** assistant turn, not only the
    /// current one. Ornith wants this; plain Qwen3.5 does not.
    pub always_render_assistant_think_block: bool,
    /// When thinking is off, still open and immediately close an empty block
    /// (`<think>\n\n</think>\n\n`) in the generation prompt. Without it the
    /// model may decide to open one itself and never stop.
    pub emit_empty_think_on_no_think: bool,
    /// Use `[img-N]` markers instead of the native vision tokens.
    pub use_img_tags: bool,
}

impl Qwen35Renderer {
    /// **Upstream:** `(*Qwen35Renderer).renderContent`.
    ///
    /// Non-marker mode puts one `<|vision_start|><|image_pad|><|vision_end|>`
    /// per image at the **front** of the text, and leaves the image counter
    /// alone -- the counter only exists for the `[img-N]` path.
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
            s.push_str("<|vision_start|><|image_pad|><|vision_end|>");
        }
        // TODO(upstream): videos are not supported yet.
        s.push_str(&message.content);
        (s, image_offset)
    }
}

/// Pull the reasoning out of an assistant message. **Upstream:**
/// `splitQwen35ReasoningContent`.
///
/// Two sources, in order:
///
/// 1. If the caller filled [`Message::thinking`] and we are rendering a think
///    block, that wins and the content is untouched.
/// 2. Otherwise look for a `</think>` **inside the content**, which is how
///    reasoning arrives when it was never separated out. Everything before it
///    (after the last `<think>`, if there is one) is the reasoning; everything
///    after has its leading **newlines only** trimmed.
///
/// Returns `(reasoning, remaining_content)`. Reasoning is always trimmed of
/// surrounding whitespace; the remaining content is not (the caller trims).
fn split_reasoning_content(
    content: &str,
    message_thinking: &str,
    is_thinking: bool,
) -> (String, String) {
    if is_thinking && !message_thinking.is_empty() {
        return (message_thinking.trim().to_string(), content.to_string());
    }

    let mut reasoning = String::new();
    let mut content = content.to_string();
    if let Some(idx) = content.find(THINK_CLOSE_TAG) {
        let before = &content[..idx];
        reasoning = match before.rfind(THINK_OPEN_TAG) {
            Some(open) => before[open + THINK_OPEN_TAG.len()..].to_string(),
            None => before.to_string(),
        };
        content = content[idx + THINK_CLOSE_TAG.len()..]
            .trim_start_matches('\n')
            .to_string();
    }

    (reasoning.trim().to_string(), content)
}

impl Renderer for Qwen35Renderer {
    fn leading_bos(&self) -> &'static str {
        ""
    }

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
            sb.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
            for tool in tools {
                sb.push('\n');
                sb.push_str(&marshal_tool_with_spaces(tool));
            }
            sb.push_str(QWEN35_TOOL_POSTAMBLE);
            // The caller's system text goes AFTER the tool instructions.
            if let Some(first) = messages.first()
                && first.role == "system"
            {
                let (system_content, _) = self.render_content(first, 0);
                let system_content = system_content.trim();
                if !system_content.is_empty() {
                    sb.push_str("\n\n");
                    sb.push_str(system_content);
                }
            }
            sb.push_str(IM_END_TAG);
            sb.push('\n');
        } else if let Some(first) = messages.first()
            && first.role == "system"
        {
            let (system_content, _) = self.render_content(first, 0);
            sb.push_str(IM_START_TAG);
            sb.push_str("system\n");
            sb.push_str(system_content.trim());
            sb.push_str(IM_END_TAG);
            sb.push('\n');
        }

        // Which user message is the last REAL query? See the module docs.
        let mut multi_step_tool = true;
        let mut last_query_index = messages.len().saturating_sub(1);
        for (i, message) in messages.iter().enumerate().rev() {
            if multi_step_tool && message.role == "user" {
                let (content, _) = self.render_content(message, 0);
                let content = content.trim();
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
            let content = content.trim().to_string();

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
                let render_think_block = self.always_render_assistant_think_block
                    || (is_thinking && i > last_query_index);
                let (reasoning, content) =
                    split_reasoning_content(&content, &message.thinking, render_think_block);

                sb.push_str(IM_START_TAG);
                sb.push_str(&message.role);
                if render_think_block {
                    sb.push_str("\n<think>\n");
                    sb.push_str(&reasoning);
                    sb.push_str("\n</think>\n\n");
                    sb.push_str(&content);
                } else {
                    sb.push('\n');
                    sb.push_str(&content);
                }

                if !message.tool_calls.is_empty() {
                    for (j, tc) in message.tool_calls.iter().enumerate() {
                        if j == 0 {
                            if !content.trim().is_empty() {
                                sb.push_str("\n\n");
                            }
                        } else {
                            sb.push('\n');
                        }

                        sb.push_str(&format!("<tool_call>\n<function={}>\n", tc.function.name));
                        for (name, value) in &tc.function.arguments.0 {
                            sb.push_str(&format!("<parameter={name}>\n"));
                            sb.push_str(&format_tool_call_argument(value));
                            sb.push_str("\n</parameter>\n");
                        }
                        sb.push_str("</function>\n</tool_call>");
                    }
                }

                if !prefill {
                    sb.push_str(IM_END_TAG);
                    sb.push('\n');
                }
            } else if message.role == "tool" {
                // Tool results ride inside a user block; consecutive ones share
                // it, exactly like Qwen3-Coder.
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

            // Generation prompt, appended after the LAST message unless that
            // message was itself an assistant prefill.
            if last_message && !prefill {
                sb.push_str(IM_START_TAG);
                sb.push_str("assistant\n");
                if is_thinking {
                    sb.push_str("<think>\n");
                } else if self.emit_empty_think_on_no_think {
                    sb.push_str("<think>\n\n</think>\n\n");
                }
            }
        }

        Ok(sb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ToolCall, ToolCallArguments, ToolCallFunction};
    use serde_json::json;

    fn thinking_renderer() -> Qwen35Renderer {
        Qwen35Renderer {
            is_thinking: true,
            ..Default::default()
        }
    }

    fn args(raw: &str) -> ToolCallArguments {
        serde_json::from_str(raw).expect("fixture args must be valid JSON")
    }

    fn call(name: &str, raw_args: &str) -> ToolCall {
        ToolCall {
            id: String::new(),
            function: ToolCallFunction {
                index: 0,
                name: name.into(),
                arguments: args(raw_args),
            },
        }
    }

    fn tool(name: &str, description: &str, props: serde_json::Value, required: &[&str]) -> Tool {
        serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "required": required,
                    "properties": props,
                }
            }
        }))
        .expect("fixture tool must deserialise")
    }

    fn math_tools() -> Vec<Tool> {
        vec![
            tool(
                "add",
                "Add two numbers",
                json!({"a": {"type": "integer"}, "b": {"type": "integer"}}),
                &["a", "b"],
            ),
            tool(
                "multiply",
                "Multiply two numbers",
                json!({"x": {"type": "integer"}, "y": {"type": "integer"}}),
                &["x", "y"],
            ),
        ]
    }

    /// Upstream `TestQwen35RendererUsesXMLToolCallingFormat`.
    #[test]
    fn tools_are_json_but_calls_are_xml_and_the_system_text_comes_after() {
        let msgs = [
            Message::new("system", "You are a helpful assistant."),
            Message::new("user", "What's the weather in Paris?"),
            Message {
                role: "assistant".into(),
                content: "I'll check.".into(),
                tool_calls: vec![call("get_weather", r#"{"location":"Paris"}"#)],
                ..Default::default()
            },
            Message::new("tool", "22C"),
            Message::new("user", "Thanks"),
        ];
        let tools = [tool(
            "get_weather",
            "",
            json!({"location": {"type": "string"}}),
            &["location"],
        )];

        let got = thinking_renderer().render(&msgs, &tools, None).unwrap();

        assert!(got.contains("<tools>"), "{got}");
        assert!(got.contains("<function=example_function_name>"), "{got}");
        assert!(
            got.contains(
                "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>"
            ),
            "{got}"
        );

        let tools_idx = got.find("# Tools").expect("tools block");
        let system_idx = got
            .find("You are a helpful assistant.")
            .expect("system text");
        assert!(system_idx > tools_idx, "system must follow tools:\n{got}");
    }

    /// Upstream `TestQwen35RendererNoThinkPrefill` -- with thinking explicitly
    /// off, `qwen3.5` still opens AND closes an empty block, so the model does
    /// not start one of its own.
    #[test]
    fn thinking_off_still_emits_an_empty_think_block() {
        let renderer = Qwen35Renderer {
            is_thinking: true,
            emit_empty_think_on_no_think: true,
            ..Default::default()
        };
        let msgs = [Message::new("user", "hello")];
        let got = renderer
            .render(&msgs, &[], Some(&ThinkValue::Bool(false)))
            .unwrap();
        assert!(
            got.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "{got}"
        );
    }

    /// Upstream `TestQwen35RendererBackToBackToolCallsAndResponses`. The
    /// interesting assertion is the **negative** one: reasoning from before the
    /// last user query is dropped.
    #[test]
    fn historical_reasoning_is_dropped_and_tool_responses_are_grouped() {
        let msgs = [
            Message::new("system", "You are a helpful assistant."),
            Message::new("user", "Run add and multiply."),
            Message {
                role: "assistant".into(),
                content: "I'll run both now.".into(),
                thinking: "Need to call add and multiply.".into(),
                tool_calls: vec![
                    call("add", r#"{"a":2,"b":3}"#),
                    call("multiply", r#"{"x":4,"y":5}"#),
                ],
                ..Default::default()
            },
            Message::new("tool", "5"),
            Message::new("tool", "20"),
            Message::new("user", "Summarize the results."),
        ];

        let got = thinking_renderer()
            .render(&msgs, &math_tools(), None)
            .unwrap();

        assert!(
            !got.contains("Need to call add and multiply."),
            "historical reasoning leaked:\n{got}"
        );

        assert!(
            got.contains(concat!(
                "<tool_call>\n<function=add>\n<parameter=a>\n2\n</parameter>\n<parameter=b>\n3\n</parameter>\n</function>\n</tool_call>\n",
                "<tool_call>\n<function=multiply>\n<parameter=x>\n4\n</parameter>\n<parameter=y>\n5\n</parameter>\n</function>\n</tool_call>"
            )),
            "{got}"
        );

        assert!(
            got.contains(
                "<|im_start|>user\n<tool_response>\n5\n</tool_response>\n<tool_response>\n20\n</tool_response><|im_end|>"
            ),
            "{got}"
        );

        assert!(got.ends_with("<|im_start|>assistant\n<think>\n"), "{got}");
    }

    /// Upstream `TestQwen35RendererInterleavedThinkingAndTools` -- both
    /// assistant turns sit after the last user query, so BOTH keep reasoning.
    #[test]
    fn reasoning_after_the_last_query_survives_on_every_turn() {
        let msgs = [
            Message::new("system", "You are a helpful assistant."),
            Message::new("user", "Plan a picnic in Paris."),
            Message {
                role: "assistant".into(),
                content: "Checking weather first.".into(),
                thinking: "Need weather before giving advice.".into(),
                tool_calls: vec![call("get_weather", r#"{"location":"Paris"}"#)],
                ..Default::default()
            },
            Message::new("tool", "22C"),
            Message {
                role: "assistant".into(),
                content: "Checking UV too.".into(),
                thinking: "Need UV index for sunscreen advice.".into(),
                tool_calls: vec![call("get_uv", r#"{"location":"Paris"}"#)],
                ..Default::default()
            },
            Message::new("tool", "5"),
        ];
        let tools = vec![
            tool(
                "get_weather",
                "Get weather for a location",
                json!({"location": {"type": "string"}}),
                &["location"],
            ),
            tool(
                "get_uv",
                "Get UV index for a location",
                json!({"location": {"type": "string"}}),
                &["location"],
            ),
        ];

        let got = thinking_renderer().render(&msgs, &tools, None).unwrap();

        assert!(
            got.contains(concat!(
                "<|im_start|>assistant\n<think>\nNeed weather before giving advice.\n</think>\n\n",
                "Checking weather first.\n\n",
                "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>"
            )),
            "{got}"
        );
        assert!(
            got.contains(concat!(
                "<|im_start|>assistant\n<think>\nNeed UV index for sunscreen advice.\n</think>\n\n",
                "Checking UV too.\n\n",
                "<tool_call>\n<function=get_uv>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>"
            )),
            "{got}"
        );
        assert!(got.ends_with("<|im_start|>assistant\n<think>\n"), "{got}");
    }

    /// Upstream `TestQwen35RendererAssistantPrefillWithThinking` -- exact
    /// whole-string match, no generation prompt appended.
    #[test]
    fn an_assistant_prefill_keeps_its_think_block_and_stays_open() {
        let msgs = [
            Message::new("user", "Write two words."),
            Message {
                role: "assistant".into(),
                thinking: "Keep it short.".into(),
                content: "Hello world".into(),
                ..Default::default()
            },
        ];
        let got = thinking_renderer().render(&msgs, &[], None).unwrap();
        assert_eq!(
            got,
            "<|im_start|>user\nWrite two words.<|im_end|>\n\
             <|im_start|>assistant\n<think>\nKeep it short.\n</think>\n\nHello world"
        );
    }

    /// Not an upstream fixture: this pins `split_reasoning_content`'s second
    /// path, where the reasoning arrives glued into the content instead of in
    /// its own field. Getting the newline trim wrong here leaves a blank line
    /// in front of every historical answer.
    #[test]
    fn inline_think_tags_are_split_out_of_the_content() {
        let (reasoning, rest) =
            split_reasoning_content("<think>\nhmm\n</think>\n\nthe answer", "", false);
        assert_eq!(reasoning, "hmm");
        assert_eq!(rest, "the answer");

        // No opening tag: everything before `</think>` is the reasoning.
        let (reasoning, rest) = split_reasoning_content("hmm</think>answer", "", false);
        assert_eq!(reasoning, "hmm");
        assert_eq!(rest, "answer");

        // A filled `thinking` field wins outright when we are rendering a block.
        let (reasoning, rest) = split_reasoning_content("body", "  reasoned  ", true);
        assert_eq!(reasoning, "reasoned");
        assert_eq!(rest, "body");
    }

    /// The `ornith` preset renders a think block on EVERY assistant turn, even
    /// ones before the last user query. That is the whole difference from
    /// `qwen3.5`.
    #[test]
    fn ornith_always_renders_the_assistant_think_block() {
        let renderer = Qwen35Renderer {
            is_thinking: true,
            always_render_assistant_think_block: true,
            emit_empty_think_on_no_think: true,
            use_img_tags: false,
        };
        let msgs = [
            Message::new("user", "first"),
            Message {
                role: "assistant".into(),
                thinking: "old thought".into(),
                content: "old answer".into(),
                ..Default::default()
            },
            Message::new("user", "second"),
        ];
        let got = renderer.render(&msgs, &[], None).unwrap();
        assert!(got.contains("<think>\nold thought\n</think>"), "{got}");
    }

    /// Native vision tokens go in front of the text, one triple per image.
    #[test]
    fn images_render_as_vision_tokens_when_markers_are_off() {
        let renderer = thinking_renderer();
        let msgs = [Message {
            role: "user".into(),
            content: "describe".into(),
            images: vec!["<blob-a>".into(), "<blob-b>".into()],
            ..Default::default()
        }];
        let got = renderer.render(&msgs, &[], None).unwrap();
        assert!(
            got.contains("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>describe<|im_end|>"),
            "{got}"
        );
    }
}
