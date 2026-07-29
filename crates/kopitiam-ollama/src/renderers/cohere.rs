//! **Cohere North / Command A 2026** -- a Jinja template transcribed into Rust,
//! blank lines and all.
//!
//! **Upstream:** `model/renderers/cohere.go`. Registered as `cohere`.
//!
//! Special tokens: **`<BOS_TOKEN>`** (BOS only -- it is *not* written into the
//! prompt, the tokenizer adds it), **`<|START_OF_TURN_TOKEN|>` /
//! `<|END_OF_TURN_TOKEN|>`**, the role markers **`<|SYSTEM_TOKEN|>`,
//! `<|USER_TOKEN|>`, `<|CHATBOT_TOKEN|>`**, and four content wrappers:
//! **`<|START_TEXT|>`/`<|END_TEXT|>`**,
//! **`<|START_THINKING|>`/`<|END_THINKING|>`**,
//! **`<|START_ACTION|>`/`<|END_ACTION|>`** (tool calls),
//! **`<|START_TOOL_RESULT|>`/`<|END_TOOL_RESULT|>`** (tool results).
//!
//! ## Why this file is full of odd whitespace
//!
//! Upstream's ground truth is North-Mini-Code-1.0's `chat_template.jinja`
//! rendered with HuggingFace Jinja semantics. That template has **untrimmed
//! blocks**, so it emits runs of newlines and spaces that look like mistakes --
//! `"\n            \n                "` before a thinking block, the empty
//! `"[\n\n\n\n]"` tools array, the `"\n\n            \n            \"0\": "`
//! inside a tool result. Every one of those is reproduced deliberately. **Do
//! not tidy them.** They are what the model saw during fine-tuning; normalising
//! them changes the prompt.
//!
//! ## Tool-call ids get renumbered
//!
//! The template's `regen_tool_call_ids` default is on, so ids are replaced with
//! sequential indices `"0"`, `"1"`, ... across the **whole** conversation, and a
//! tool result references the index of the call it answers (looked up through
//! the client's original id). A result whose id was never seen falls back to
//! whatever the client sent, which is upstream's behaviour and not obviously
//! right -- but it is the oracle.
//!
//! ## The Available Tools section always renders
//!
//! Even with no tools: `# Available Tools\n```json\n[\n\n\n\n]\n```\n`. Skipping
//! it when the list is empty would change the system turn on **every**
//! toolless request.

use std::collections::HashMap;

use super::json::{add_spaces_outside_strings, go_value, write_go_parameters};
use super::{Message, RenderError, Renderer, ThinkValue, Tool};
use crate::api::{ToolCall, ToolCallArguments};

/// **Upstream:** `CohereRenderer`. Carries no configuration.
#[derive(Debug, Clone, Copy, Default)]
pub struct CohereRenderer;

/// One tool entry, exactly as the template's `tojson` filter writes it.
///
/// **Upstream:** `cohereToolJSON`. Note the trailing `"responses": null` --
/// there is no such field on [`Tool`]; the template emits it unconditionally, so
/// we do too.
fn tool_json(tool: &Tool) -> String {
    let mut params = String::new();
    write_go_parameters(&tool.function.parameters, &mut params);
    format!(
        "{{\"name\": {}, \"description\": {}, \"parameters\": {}, \"responses\": null}}",
        go_value(&serde_json::Value::String(tool.function.name.clone())),
        go_value(&serde_json::Value::String(
            tool.function.description.clone()
        )),
        add_spaces_outside_strings(&params),
    )
}

/// **Upstream:** `writeToolsSection`. The whitespace is the template's.
fn write_tools_section(sb: &mut String, tools: &[Tool]) {
    sb.push_str("# Available Tools\n```json\n[\n");
    if tools.is_empty() {
        sb.push_str("\n\n");
    } else {
        for (i, tool) in tools.iter().enumerate() {
            sb.push_str("\n    ");
            sb.push_str(&tool_json(tool));
            if i < tools.len() - 1 {
                sb.push(',');
            }
            sb.push_str("\n\n");
        }
    }
    sb.push_str("\n]\n```");
}

/// **Upstream:** `writeToolResult`. Again: the blank lines are the template's.
fn write_tool_result(sb: &mut String, call_id: &str, content: &str) {
    let wrapped = add_spaces_outside_strings(&format!(
        "{{\"content\":{}}}",
        go_value(&serde_json::Value::String(content.to_string()))
    ));
    sb.push_str("\n    {\n        \"tool_call_id\": \"");
    sb.push_str(call_id);
    sb.push_str("\",\n        \"results\": {\n\n            \n            \"0\": ");
    sb.push_str(&wrapped);
    sb.push_str("\n\n        },\n        \"is_error\": null\n    }");
}

/// `marshalWithSpaces(arguments)`.
fn arguments_with_spaces(args: &ToolCallArguments) -> String {
    add_spaces_outside_strings(&super::json::go_arguments(args))
}

impl Renderer for CohereRenderer {
    fn leading_bos(&self) -> &'static str {
        "<BOS_TOKEN>"
    }

    fn render(
        &self,
        messages: &[Message],
        tools: &[Tool],
        think: Option<&ThinkValue>,
    ) -> Result<String, RenderError> {
        let mut sb = String::new();

        // The template defaults reasoning ON; only an explicit false turns it
        // off.
        let reasoning = think.is_none_or(|t| t.enabled());

        // The first system message fills the template's platform-instruction
        // slot; everything else is history.
        let (system, rest): (&str, &[Message]) = match messages.first() {
            Some(first) if first.role.eq_ignore_ascii_case("system") => {
                (&first.content, &messages[1..])
            }
            _ => ("", messages),
        };

        sb.push_str("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|START_TEXT|>");
        if !system.is_empty() {
            sb.push_str(system);
            sb.push_str("\n\n\n\n");
        }
        write_tools_section(&mut sb, tools);
        sb.push_str("<|END_TEXT|><|END_OF_TURN_TOKEN|>");

        // Sequential id regeneration -- see the module docs.
        let mut call_index = 0usize;
        let mut call_id_to_index: HashMap<String, String> = HashMap::new();
        // A plain fn rather than a closure: the id map is also read further
        // down, and a `FnMut` closure capturing it would hold the borrow open
        // for the whole loop.
        fn next_call_id(
            tc: &ToolCall,
            call_index: &mut usize,
            call_id_to_index: &mut HashMap<String, String>,
        ) -> String {
            let idx = call_index.to_string();
            *call_index += 1;
            if !tc.id.is_empty() {
                call_id_to_index.entry(tc.id.clone()).or_insert(idx.clone());
            }
            idx
        }

        let mut prefill = false;
        let mut i = 0usize;
        while i < rest.len() {
            let message = &rest[i];
            match message.role.to_lowercase().as_str() {
                "system" => {
                    sb.push_str("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|START_TEXT|>");
                    sb.push_str(&message.content);
                    sb.push_str("<|END_TEXT|><|END_OF_TURN_TOKEN|>");
                }
                "user" => {
                    sb.push_str("<|START_OF_TURN_TOKEN|><|USER_TOKEN|><|START_TEXT|>");
                    sb.push_str(&message.content);
                    sb.push_str("<|END_TEXT|><|END_OF_TURN_TOKEN|>");
                }
                "assistant" | "chatbot" => {
                    sb.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
                    if !message.tool_calls.is_empty() {
                        // Untrimmed Jinja block whitespace. Yes, really.
                        sb.push_str("\n            \n                ");
                        if !message.thinking.is_empty() {
                            sb.push_str("<|START_THINKING|>");
                            sb.push_str(&message.thinking);
                            sb.push_str("<|END_THINKING|>");
                        }
                        sb.push_str("<|START_ACTION|>[");
                        for (j, tc) in message.tool_calls.iter().enumerate() {
                            let args = arguments_with_spaces(&tc.function.arguments);
                            sb.push_str("\n\n    {\"tool_call_id\": \"");
                            sb.push_str(&next_call_id(tc, &mut call_index, &mut call_id_to_index));
                            sb.push_str("\", \"tool_name\": \"");
                            sb.push_str(&tc.function.name);
                            sb.push_str("\", \"parameters\": ");
                            sb.push_str(&args);
                            sb.push('}');
                            if j < message.tool_calls.len() - 1 {
                                sb.push(',');
                            }
                        }
                        sb.push_str("\n\n]<|END_ACTION|><|END_OF_TURN_TOKEN|>");
                    } else {
                        if !message.thinking.is_empty() {
                            sb.push_str("<|START_THINKING|>");
                            sb.push_str(&message.thinking);
                            sb.push_str("<|END_THINKING|>");
                        }
                        sb.push_str("<|START_TEXT|>");
                        sb.push_str(&message.content);
                        if i == rest.len() - 1 {
                            // Prefill: leave the text open for continuation.
                            prefill = true;
                        } else {
                            sb.push_str("<|END_TEXT|><|END_OF_TURN_TOKEN|>");
                        }
                    }
                }
                "tool" => {
                    // Consecutive tool messages merge into ONE result array.
                    sb.push_str("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|START_TOOL_RESULT|>[");
                    let id = call_id_to_index
                        .get(&message.tool_call_id)
                        .cloned()
                        .unwrap_or_else(|| message.tool_call_id.clone());
                    write_tool_result(&mut sb, &id, &message.content);
                    while i + 1 < rest.len() && rest[i + 1].role.eq_ignore_ascii_case("tool") {
                        i += 1;
                        sb.push(',');
                        let id = call_id_to_index
                            .get(&rest[i].tool_call_id)
                            .cloned()
                            .unwrap_or_else(|| rest[i].tool_call_id.clone());
                        write_tool_result(&mut sb, &id, &rest[i].content);
                    }
                    sb.push_str("\n\n]<|END_TOOL_RESULT|><|END_OF_TURN_TOKEN|>");
                }
                _ => {}
            }
            i += 1;
        }

        if !prefill {
            sb.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
            sb.push_str(if reasoning {
                "<|START_THINKING|>"
            } else {
                "<|START_THINKING|><|END_THINKING|>"
            });
        }

        Ok(sb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ToolCallFunction;
    use serde_json::json;

    /// The empty-tools system turn -- upstream's `cohereSystemTurnNoTools`.
    const SYSTEM_TURN_NO_TOOLS: &str = concat!(
        "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|START_TEXT|>",
        "# Available Tools\n```json\n[\n\n\n\n]\n```",
        "<|END_TEXT|><|END_OF_TURN_TOKEN|>",
    );

    fn call(id: &str, name: &str, raw_args: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            function: ToolCallFunction {
                index: 0,
                name: name.into(),
                arguments: serde_json::from_str(raw_args).expect("valid fixture args"),
            },
        }
    }

    /// Upstream `TestCohereRenderUserOnly`.
    #[test]
    fn a_lone_user_message_gets_the_empty_tools_platform_turn() {
        let got = CohereRenderer
            .render(&[Message::new("user", "USERMSG")], &[], None)
            .unwrap();
        assert_eq!(
            got,
            format!(
                "{SYSTEM_TURN_NO_TOOLS}\
                 <|START_OF_TURN_TOKEN|><|USER_TOKEN|><|START_TEXT|>USERMSG<|END_TEXT|><|END_OF_TURN_TOKEN|>\
                 <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_THINKING|>"
            )
        );
    }

    /// Upstream `TestCohereRenderSystemHistoryAndThinking`.
    #[test]
    fn a_system_prompt_is_followed_by_four_newlines_before_the_tools_section() {
        let got = CohereRenderer
            .render(
                &[
                    Message::new("system", "DEVPREAMBLE"),
                    Message::new("user", "Q1"),
                    Message {
                        role: "assistant".into(),
                        content: "A1".into(),
                        thinking: "THINK1".into(),
                        ..Default::default()
                    },
                    Message::new("user", "Q2"),
                ],
                &[],
                None,
            )
            .unwrap();
        assert_eq!(
            got,
            concat!(
                "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|START_TEXT|>",
                "DEVPREAMBLE\n\n\n\n# Available Tools\n```json\n[\n\n\n\n]\n```",
                "<|END_TEXT|><|END_OF_TURN_TOKEN|>",
                "<|START_OF_TURN_TOKEN|><|USER_TOKEN|><|START_TEXT|>Q1<|END_TEXT|><|END_OF_TURN_TOKEN|>",
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_THINKING|>THINK1<|END_THINKING|><|START_TEXT|>A1<|END_TEXT|><|END_OF_TURN_TOKEN|>",
                "<|START_OF_TURN_TOKEN|><|USER_TOKEN|><|START_TEXT|>Q2<|END_TEXT|><|END_OF_TURN_TOKEN|>",
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_THINKING|>",
            )
        );
    }

    /// Upstream `TestCohereRenderReasoningOff`.
    #[test]
    fn reasoning_off_closes_the_thinking_block_immediately() {
        let got = CohereRenderer
            .render(
                &[Message::new("user", "Q")],
                &[],
                Some(&ThinkValue::Bool(false)),
            )
            .unwrap();
        assert!(
            got.ends_with(
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_THINKING|><|END_THINKING|>"
            ),
            "{got}"
        );
    }

    /// Upstream `TestCohereRenderToolFlow`.
    #[test]
    fn the_tool_flow_matches_the_jinja_ground_truth() {
        let tools: Vec<Tool> = vec![
            serde_json::from_value(json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "required": ["city"],
                        "properties": {"city": {"type": "string"}}
                    }
                }
            }))
            .unwrap(),
        ];
        let got = CohereRenderer
            .render(
                &[
                    Message::new("user", "weather in Paris?"),
                    Message {
                        role: "assistant".into(),
                        thinking: "I should call the tool".into(),
                        tool_calls: vec![call("call_x", "get_weather", r#"{"city":"Paris"}"#)],
                        ..Default::default()
                    },
                    Message {
                        role: "tool".into(),
                        tool_call_id: "call_x".into(),
                        content: "15C sunny".into(),
                        ..Default::default()
                    },
                    Message::new("user", "thanks"),
                ],
                &tools,
                None,
            )
            .unwrap();

        assert!(got.contains("# Available Tools\n```json\n[\n\n    {\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"required\": [\"city\"], \"properties\": {\"city\": {\"type\": \"string\"}}}, \"responses\": null}\n\n\n]\n```"), "tools section:\n{got}");
        assert!(got.contains("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\n            \n                <|START_THINKING|>I should call the tool<|END_THINKING|><|START_ACTION|>[\n\n    {\"tool_call_id\": \"0\", \"tool_name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}\n\n]<|END_ACTION|><|END_OF_TURN_TOKEN|>"), "action turn:\n{got}");
        assert!(got.contains("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|START_TOOL_RESULT|>[\n    {\n        \"tool_call_id\": \"0\",\n        \"results\": {\n\n            \n            \"0\": {\"content\": \"15C sunny\"}\n\n        },\n        \"is_error\": null\n    }\n\n]<|END_TOOL_RESULT|><|END_OF_TURN_TOKEN|>"), "result turn:\n{got}");
    }

    /// Upstream `TestCohereRenderMultiToolCallsAndResults` -- ids renumbered
    /// 0,1 and consecutive results merged into one array.
    #[test]
    fn multiple_calls_are_renumbered_and_their_results_merge() {
        let got = CohereRenderer
            .render(
                &[
                    Message::new("user", "go"),
                    Message {
                        role: "assistant".into(),
                        tool_calls: vec![call("a", "t1", "{}"), call("b", "t2", r#"{"x":1}"#)],
                        ..Default::default()
                    },
                    Message {
                        role: "tool".into(),
                        tool_call_id: "a".into(),
                        content: "r1".into(),
                        ..Default::default()
                    },
                    Message {
                        role: "tool".into(),
                        tool_call_id: "b".into(),
                        content: "r2".into(),
                        ..Default::default()
                    },
                ],
                &[],
                None,
            )
            .unwrap();

        assert!(got.contains("<|START_ACTION|>[\n\n    {\"tool_call_id\": \"0\", \"tool_name\": \"t1\", \"parameters\": {}},\n\n    {\"tool_call_id\": \"1\", \"tool_name\": \"t2\", \"parameters\": {\"x\": 1}}\n\n]<|END_ACTION|>"), "{got}");
        assert!(got.contains("<|START_TOOL_RESULT|>[\n    {\n        \"tool_call_id\": \"0\",\n        \"results\": {\n\n            \n            \"0\": {\"content\": \"r1\"}\n\n        },\n        \"is_error\": null\n    },\n    {\n        \"tool_call_id\": \"1\",\n        \"results\": {\n\n            \n            \"0\": {\"content\": \"r2\"}\n\n        },\n        \"is_error\": null\n    }\n\n]<|END_TOOL_RESULT|>"), "{got}");
    }

    /// Upstream `TestCohereRenderAssistantPrefill`.
    #[test]
    fn a_trailing_assistant_message_leaves_the_text_open() {
        let got = CohereRenderer
            .render(
                &[
                    Message::new("user", "Q"),
                    Message::new("assistant", "partial"),
                ],
                &[],
                None,
            )
            .unwrap();
        assert!(
            got.ends_with("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_TEXT|>partial"),
            "{got}"
        );
    }

    /// Upstream `TestCohereRendererRegistered`.
    #[test]
    fn cohere_is_reachable_by_name_and_reports_its_bos() {
        assert!(super::super::renderer_for_name("cohere").is_some());
        assert_eq!(
            super::super::leading_bos_for_renderer("cohere"),
            "<BOS_TOKEN>"
        );
    }
}
