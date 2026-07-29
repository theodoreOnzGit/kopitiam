//! **Qwen3-Coder** -- ChatML turns, tools described in XML, tool calls in XML.
//!
//! **Upstream:** `model/renderers/qwen3coder.go`. Registered as `qwen3-coder`.
//!
//! ## The framing, in full
//!
//! Special tokens: **`<|im_start|>`** and **`<|im_end|>`** ([`IM_START_TAG`] /
//! [`IM_END_TAG`]). No BOS -- [`Renderer::leading_bos`] returns `""`.
//!
//! ```text
//! <|im_start|>system
//! {system}
//!
//! # Tools
//! ... <tools> ... </tools> ... <IMPORTANT> ... </IMPORTANT><|im_end|>
//! <|im_start|>user
//! {text}<|im_end|>
//! <|im_start|>assistant
//! ```
//!
//! Three things that are easy to get wrong and would silently degrade the model:
//!
//! * **Tools are XML, not JSON.** Qwen3-Coder was fine-tuned on
//!   `<function><name>..</name><parameters><parameter>..` rather than a JSON
//!   Schema blob. Hand it JSON and it still answers, just worse.
//! * **Tool-call arguments are printed raw, not quoted.** A string argument goes
//!   in as `Paris`, not `"Paris"` -- see [`format_tool_call_argument`]. Only
//!   objects and arrays keep their JSON.
//! * **Consecutive `tool` messages share one `<|im_start|>user` block**, each
//!   with its own `<tool_response>`, and the block closes **immediately** after
//!   the last `</tool_response>` with no newline in between.
//!
//! ## System messages
//!
//! Only the **first** system message survives; the rest are dropped. If there
//! are tools but no system message at all, upstream injects the reference
//! implementation's default: *"You are Qwen, a helpful AI assistant that can
//! interact with a computer to solve tasks."*

use serde_json::Value;

use super::json::{go_value, write_go_property, write_go_value};
use super::{IM_END_TAG, IM_START_TAG, Message, RenderError, Renderer, ThinkValue, Tool};
use crate::api::{PropertyType, ToolFunctionParameters, ToolProperty};

/// **Upstream:** `Qwen3CoderRenderer`. Carries no configuration.
#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen3CoderRenderer;

/// The default system prompt used when tools are present but the caller gave
/// none. **Upstream:** `qwen3coder.go`, matching the reference implementation.
const QWEN3_CODER_DEFAULT_SYSTEM: &str =
    "You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.";

/// The call-format instructions appended after `</tools>`.
///
/// **Upstream:** the one giant `sb.WriteString(...)` at the end of the tools
/// block in `qwen3coder.go`. Copied verbatim, whitespace included -- the model
/// was trained on exactly this text, so "tidying" it is a bug.
pub(crate) const QWEN_TOOL_POSTAMBLE: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// Print one tool-call argument the way the model expects to read it back.
///
/// **Upstream:** `formatToolCallArgument`.
///
/// The surprise: **a string comes out with no quotes.** `"Paris"` renders as
/// `Paris`. Only maps/slices/arrays keep JSON syntax; everything else goes
/// through Go's `%v`. Quoting a string here would put `"Paris"` inside
/// `<parameter=location>`, which is not what the model was trained to emit and
/// so is not what it will produce on the way back.
pub(crate) fn format_tool_call_argument(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::String(s) => s.clone(),
        Value::Object(_) | Value::Array(_) => go_value(value),
        // Go's `%v` on the bool/number that came out of `encoding/json`.
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
    }
}

/// Print a parameter's declared type inside `<type>...</type>`.
///
/// **Upstream:** `formatToolDefinitionType`. Empty -> `[]`, one type -> the bare
/// name, several -> a JSON array. Note the empty case is `[]` and **not** the
/// omitted-field `null` you would get from `json.Marshal` of a nil slice.
pub(crate) fn format_tool_definition_type(t: &PropertyType) -> String {
    match t.0.len() {
        0 => "[]".to_string(),
        1 => t.0[0].clone(),
        _ => {
            let mut s = String::from("[");
            for (i, name) in t.0.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push('"');
                s.push_str(name);
                s.push('"');
            }
            s.push(']');
            s
        }
    }
}

/// Emit `\n<key>value</key>` for every JSON field of `obj` that the caller has
/// not already rendered by hand.
///
/// **Upstream:** `renderAdditionalKeys`, which marshals the struct, unmarshals
/// it into a `map[string]any`, and ranges that map.
///
/// **Deliberate divergence: we fix the order.** Go's map iteration is
/// *randomised*, so upstream's output is genuinely non-deterministic the moment
/// a property carries more than one extra key. Every upstream fixture happens to
/// have exactly one, which is why nobody noticed. We emit in Go's **struct
/// declaration** order instead -- deterministic, reproducible, and identical to
/// upstream wherever upstream is itself well-defined. A renderer that cannot be
/// tested byte-for-byte is a renderer nobody can trust, so determinism wins here.
fn additional_keys_for_property(p: &ToolProperty) -> String {
    // Handled by the caller: `type`, `description`.
    let mut s = String::new();
    if !p.any_of.is_empty() {
        let mut v = String::from("[");
        for (i, a) in p.any_of.iter().enumerate() {
            if i > 0 {
                v.push(',');
            }
            write_go_property(a, &mut v);
        }
        v.push(']');
        s.push_str(&format!("\n<anyOf>{v}</anyOf>"));
    }
    if let Some(items) = &p.items {
        // A nil `items` is skipped by upstream's `case nil: continue`.
        if !items.is_null() {
            s.push_str(&format!("\n<items>{}</items>", simple_or_json(items)));
        }
    }
    if !p.enum_values.is_empty() {
        let mut v = String::from("[");
        for (i, e) in p.enum_values.iter().enumerate() {
            if i > 0 {
                v.push(',');
            }
            write_go_value(e, &mut v);
        }
        v.push(']');
        s.push_str(&format!("\n<enum>{v}</enum>"));
    }
    if let Some(props) = &p.properties {
        let mut v = String::from("{");
        for (i, (k, val)) in props.iter().enumerate() {
            if i > 0 {
                v.push(',');
            }
            v.push('"');
            v.push_str(k);
            v.push_str("\":");
            write_go_property(val, &mut v);
        }
        v.push('}');
        s.push_str(&format!("\n<properties>{v}</properties>"));
    }
    if !p.required.is_empty() {
        s.push_str(&format!(
            "\n<required>{}</required>",
            go_value(&Value::Array(
                p.required
                    .iter()
                    .map(|r| Value::String(r.clone()))
                    .collect()
            ))
        ));
    }
    s
}

/// The `ToolFunctionParameters` half of [`additional_keys_for_property`].
/// Handled by the caller: `type`, `properties`.
fn additional_keys_for_parameters(p: &ToolFunctionParameters) -> String {
    let mut s = String::new();
    if let Some(defs) = &p.defs
        && !defs.is_null()
    {
        s.push_str(&format!("\n<$defs>{}</$defs>", simple_or_json(defs)));
    }
    if let Some(items) = &p.items
        && !items.is_null()
    {
        s.push_str(&format!("\n<items>{}</items>", simple_or_json(items)));
    }
    if !p.required.is_empty() {
        s.push_str(&format!(
            "\n<required>{}</required>",
            go_value(&Value::Array(
                p.required
                    .iter()
                    .map(|r| Value::String(r.clone()))
                    .collect()
            ))
        ));
    }
    s
}

/// Upstream's `switch v := value.(type)` inside `renderAdditionalKeys`: a map or
/// slice is JSON-encoded, anything else goes through `%v` (so a string loses its
/// quotes).
fn simple_or_json(v: &Value) -> String {
    match v {
        Value::Object(_) | Value::Array(_) => go_value(v),
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Null => String::new(),
    }
}

impl Renderer for Qwen3CoderRenderer {
    fn leading_bos(&self) -> &'static str {
        ""
    }

    fn render(
        &self,
        messages: &[Message],
        tools: &[Tool],
        _think: Option<&ThinkValue>,
    ) -> Result<String, RenderError> {
        let mut sb = String::new();

        // Filter out system messages; the FIRST one wins, the rest vanish.
        let mut system_message = String::new();
        let mut filtered: Vec<&Message> = Vec::with_capacity(messages.len());
        for m in messages {
            if m.role != "system" {
                filtered.push(m);
                continue;
            }
            if system_message.is_empty() {
                system_message = m.content.clone();
            }
        }

        if !system_message.is_empty() || !tools.is_empty() {
            sb.push_str(IM_START_TAG);
            sb.push_str("system\n");

            if system_message.is_empty() {
                system_message = QWEN3_CODER_DEFAULT_SYSTEM.to_string();
            }
            sb.push_str(&system_message);

            if !tools.is_empty() {
                sb.push_str("\n\n# Tools\n\nYou have access to the following functions:\n\n");
                sb.push_str("<tools>");
                for tool in tools {
                    sb.push('\n');
                    sb.push_str("<function>\n");
                    sb.push_str(&format!("<name>{}</name>", tool.function.name));
                    if !tool.function.description.is_empty() {
                        sb.push_str(&format!(
                            "\n<description>{}</description>",
                            tool.function.description
                        ));
                    }
                    sb.push_str("\n<parameters>");

                    for (name, prop) in tool.function.parameters.properties_iter() {
                        sb.push_str("\n<parameter>");
                        sb.push_str(&format!("\n<name>{name}</name>"));

                        if !prop.prop_type.0.is_empty() {
                            sb.push_str(&format!(
                                "\n<type>{}</type>",
                                format_tool_definition_type(&prop.prop_type)
                            ));
                        }

                        if !prop.description.is_empty() {
                            sb.push_str(&format!(
                                "\n<description>{}</description>",
                                prop.description
                            ));
                        }

                        sb.push_str(&additional_keys_for_property(prop));
                        sb.push_str("\n</parameter>");
                    }

                    sb.push_str(&additional_keys_for_parameters(&tool.function.parameters));
                    sb.push_str("\n</parameters>");
                    sb.push_str("\n</function>");
                }
                sb.push_str("\n</tools>");
                sb.push_str(QWEN_TOOL_POSTAMBLE);
            }

            sb.push_str(IM_END_TAG);
            sb.push('\n');
        }

        for (i, message) in filtered.iter().enumerate() {
            let last_message = i == filtered.len() - 1;
            let prefill = last_message && message.role == "assistant";

            match message.role.as_str() {
                "assistant" => {
                    if !message.tool_calls.is_empty() {
                        sb.push_str(IM_START_TAG);
                        sb.push_str("assistant\n");
                        if !message.content.is_empty() {
                            sb.push_str(&message.content);
                            sb.push('\n');
                        }
                        for tc in &message.tool_calls {
                            sb.push_str(&format!("\n<tool_call>\n<function={}>", tc.function.name));
                            for (name, value) in &tc.function.arguments.0 {
                                sb.push_str(&format!(
                                    "\n<parameter={}>\n{}\n</parameter>",
                                    name,
                                    format_tool_call_argument(value)
                                ));
                            }
                            sb.push_str("\n</function>\n</tool_call>");
                        }
                        sb.push_str(IM_END_TAG);
                        sb.push('\n');
                    } else {
                        sb.push_str(IM_START_TAG);
                        sb.push_str("assistant\n");
                        sb.push_str(&message.content);
                        if !prefill {
                            sb.push_str(IM_END_TAG);
                            sb.push('\n');
                        }
                    }
                }
                "tool" => {
                    // Consecutive tool responses share ONE `<|im_start|>user`
                    // block but keep separate `<tool_response>` tags.
                    if i == 0 || filtered[i - 1].role != "tool" {
                        sb.push_str(IM_START_TAG);
                        sb.push_str("user");
                    }
                    sb.push_str("\n<tool_response>\n");
                    sb.push_str(&message.content);
                    sb.push_str("\n</tool_response>");
                    // ...and the block closes only after the LAST of them, with
                    // no newline between `</tool_response>` and `<|im_end|>`.
                    if i == filtered.len() - 1 || filtered[i + 1].role != "tool" {
                        sb.push_str(IM_END_TAG);
                        sb.push('\n');
                    }
                }
                role => {
                    sb.push_str(IM_START_TAG);
                    sb.push_str(role);
                    sb.push('\n');
                    sb.push_str(&message.content);
                    sb.push_str(IM_END_TAG);
                    sb.push('\n');
                }
            }

            if last_message && !prefill {
                sb.push_str(IM_START_TAG);
                sb.push_str("assistant\n");
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

    fn msg(role: &str, content: &str) -> Message {
        Message::new(role, content)
    }

    fn args(raw: &str) -> ToolCallArguments {
        serde_json::from_str(raw).expect("fixture args must be valid JSON")
    }

    fn call(name: &str, raw_args: &str) -> ToolCall {
        ToolCall {
            id: String::new(),
            function: ToolCallFunction {
                index: 0,
                name: name.to_string(),
                arguments: args(raw_args),
            },
        }
    }

    fn tool(raw: serde_json::Value) -> Tool {
        serde_json::from_value(raw).expect("fixture tool must deserialise")
    }

    /// Upstream `TestQwen3CoderRenderer/basic`.
    #[test]
    fn a_plain_system_and_user_exchange_renders_as_chatml() {
        let msgs = [
            msg("system", "You are a helpful assistant."),
            msg("user", "Hello, how are you?"),
        ];
        let got = Qwen3CoderRenderer.render(&msgs, &[], None).unwrap();
        assert_eq!(
            got,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\nHello, how are you?<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    /// Upstream `TestQwen3CoderRenderer/with tools and response` -- the whole
    /// XML tool block, verbatim.
    #[test]
    fn tools_render_as_xml_with_the_full_postamble() {
        let msgs = [
            msg("system", "You are a helpful assistant with access to tools."),
            msg("user", "What is the weather like in San Francisco?"),
            Message {
                role: "assistant".into(),
                content: "I'll check the weather in San Francisco for you.".into(),
                tool_calls: vec![call("get_weather", r#"{"unit":"fahrenheit"}"#)],
                ..Default::default()
            },
            Message {
                role: "tool".into(),
                content: "{\"location\": \"San Francisco, CA\", \"temperature\": 68, \"condition\": \"partly cloudy\", \"humidity\": 65, \"wind_speed\": 12}".into(),
                tool_name: "get_weather".into(),
                ..Default::default()
            },
            msg("user", "That sounds nice! What about New York?"),
        ];
        let tools = [tool(json!({
            "type": "",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "",
                    "required": ["unit"],
                    "properties": {
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature"
                        }
                    }
                }
            }
        }))];

        let got = Qwen3CoderRenderer.render(&msgs, &tools, None).unwrap();
        let want = concat!(
            "<|im_start|>system\n",
            "You are a helpful assistant with access to tools.\n",
            "\n",
            "# Tools\n",
            "\n",
            "You have access to the following functions:\n",
            "\n",
            "<tools>\n",
            "<function>\n",
            "<name>get_weather</name>\n",
            "<description>Get the current weather in a given location</description>\n",
            "<parameters>\n",
            "<parameter>\n",
            "<name>unit</name>\n",
            "<type>string</type>\n",
            "<description>The unit of temperature</description>\n",
            "<enum>[\"celsius\",\"fahrenheit\"]</enum>\n",
            "</parameter>\n",
            "<required>[\"unit\"]</required>\n",
            "</parameters>\n",
            "</function>\n",
            "</tools>\n",
            "\n",
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n",
            "\n",
            "<tool_call>\n",
            "<function=example_function_name>\n",
            "<parameter=example_parameter_1>\n",
            "value_1\n",
            "</parameter>\n",
            "<parameter=example_parameter_2>\n",
            "This is the value for the second parameter\n",
            "that can span\n",
            "multiple lines\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>\n",
            "\n",
            "<IMPORTANT>\n",
            "Reminder:\n",
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n",
            "- Required parameters MUST be specified\n",
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n",
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n",
            "</IMPORTANT><|im_end|>\n",
            "<|im_start|>user\n",
            "What is the weather like in San Francisco?<|im_end|>\n",
            "<|im_start|>assistant\n",
            "I'll check the weather in San Francisco for you.\n",
            "\n",
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=unit>\n",
            "fahrenheit\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call><|im_end|>\n",
            "<|im_start|>user\n",
            "<tool_response>\n",
            "{\"location\": \"San Francisco, CA\", \"temperature\": 68, \"condition\": \"partly cloudy\", \"humidity\": 65, \"wind_speed\": 12}\n",
            "</tool_response><|im_end|>\n",
            "<|im_start|>user\n",
            "That sounds nice! What about New York?<|im_end|>\n",
            "<|im_start|>assistant\n",
        );
        assert_eq!(got, want);
    }

    /// Upstream `TestQwen3CoderRenderer/parallel tool calls` -- two calls in one
    /// assistant turn, then two tool replies sharing one user block.
    #[test]
    fn parallel_tool_calls_and_grouped_responses() {
        let msgs = [
            msg(
                "system",
                "You are a helpful assistant with access to tools.",
            ),
            msg("user", "call double(1) and triple(2)"),
            Message {
                role: "assistant".into(),
                content: "I'll call double(1) and triple(2) for you.".into(),
                tool_calls: vec![
                    call("double", r#"{"number":"1"}"#),
                    call("triple", r#"{"number":"2"}"#),
                ],
                ..Default::default()
            },
            Message {
                role: "tool".into(),
                content: "{\"number\": 2}".into(),
                tool_name: "double".into(),
                ..Default::default()
            },
            Message {
                role: "tool".into(),
                content: "{\"number\": 6}".into(),
                tool_name: "triple".into(),
                ..Default::default()
            },
        ];
        let tools = [
            tool(json!({
                "type": "",
                "function": {
                    "name": "double", "description": "Double a number",
                    "parameters": {"type": "", "properties": {
                        "number": {"type": "string", "description": "The number to double"}
                    }}
                }
            })),
            tool(json!({
                "type": "",
                "function": {
                    "name": "triple", "description": "Triple a number",
                    "parameters": {"type": "", "properties": {
                        "number": {"type": "string", "description": "The number to triple"}
                    }}
                }
            })),
        ];

        let got = Qwen3CoderRenderer.render(&msgs, &tools, None).unwrap();

        assert!(
            got.contains(concat!(
                "<tools>\n",
                "<function>\n",
                "<name>double</name>\n",
                "<description>Double a number</description>\n",
                "<parameters>\n",
                "<parameter>\n",
                "<name>number</name>\n",
                "<type>string</type>\n",
                "<description>The number to double</description>\n",
                "</parameter>\n",
                "</parameters>\n",
                "</function>\n",
                "<function>\n",
                "<name>triple</name>\n",
            )),
            "tool block:\n{got}"
        );

        assert!(got.contains(concat!(
            "<tool_call>\n<function=double>\n<parameter=number>\n1\n</parameter>\n</function>\n</tool_call>\n",
            "<tool_call>\n<function=triple>\n<parameter=number>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>\n",
        )), "calls:\n{got}");

        assert!(
            got.contains(concat!(
                "<|im_start|>user\n",
                "<tool_response>\n{\"number\": 2}\n</tool_response>\n",
                "<tool_response>\n{\"number\": 6}\n</tool_response><|im_end|>\n",
                "<|im_start|>assistant\n",
            )),
            "responses:\n{got}"
        );
    }

    /// Upstream `TestQwen3CoderRenderer/prefill` -- a trailing assistant message
    /// is a *prefill*: no `<|im_end|>`, no fresh assistant header, the model
    /// carries straight on from it.
    #[test]
    fn a_trailing_assistant_message_is_left_open_as_a_prefill() {
        let msgs = [
            msg("system", "You are a helpful assistant."),
            msg("user", "Tell me something interesting."),
            msg(
                "assistant",
                "I'll tell you something interesting about cats",
            ),
        ];
        let got = Qwen3CoderRenderer.render(&msgs, &[], None).unwrap();
        assert_eq!(
            got,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\nTell me something interesting.<|im_end|>\n\
             <|im_start|>assistant\nI'll tell you something interesting about cats"
        );
    }

    /// Upstream `TestQwen3CoderRenderer/complex tool call arguments should
    /// remain json encoded`.
    #[test]
    fn object_arguments_stay_json_while_strings_lose_their_quotes() {
        let msgs = [
            msg("user", "call tool"),
            Message {
                role: "assistant".into(),
                tool_calls: vec![call("echo", r#"{"payload":{"foo":"bar"}}"#)],
                ..Default::default()
            },
            Message {
                role: "tool".into(),
                content: "{\"payload\": {\"foo\": \"bar\"}}".into(),
                tool_name: "echo".into(),
                ..Default::default()
            },
        ];
        let got = Qwen3CoderRenderer.render(&msgs, &[], None).unwrap();
        assert_eq!(
            got,
            "<|im_start|>user\ncall tool<|im_end|>\n\
             <|im_start|>assistant\n\n\
             <tool_call>\n<function=echo>\n<parameter=payload>\n{\"foo\":\"bar\"}\n</parameter>\n</function>\n</tool_call><|im_end|>\n\
             <|im_start|>user\n<tool_response>\n{\"payload\": {\"foo\": \"bar\"}}\n</tool_response><|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    /// Upstream `TestQwen3CoderRendererToolResponseNoTrailingNewline`. The
    /// missing newline is not cosmetic -- it is what the model was trained on.
    #[test]
    fn tool_response_is_immediately_followed_by_im_end() {
        let msgs = [
            msg("user", "call tool"),
            Message {
                role: "assistant".into(),
                tool_calls: vec![call("echo", r#"{"payload":"ok"}"#)],
                ..Default::default()
            },
            Message {
                role: "tool".into(),
                content: "{\"payload\":\"ok\"}".into(),
                tool_name: "echo".into(),
                ..Default::default()
            },
        ];
        let got = Qwen3CoderRenderer.render(&msgs, &[], None).unwrap();
        assert!(!got.contains("</tool_response>\n<|im_end|>"), "{got}");
        assert!(got.contains("</tool_response><|im_end|>"), "{got}");
    }

    /// Upstream `TestFormatToolCallArgument`.
    #[test]
    fn tool_call_arguments_print_the_way_upstream_prints_them() {
        assert_eq!(format_tool_call_argument(&json!("foo")), "foo");
        assert_eq!(
            format_tool_call_argument(&json!({"foo": "bar"})),
            r#"{"foo":"bar"}"#
        );
        assert_eq!(format_tool_call_argument(&json!(1)), "1");
        assert_eq!(format_tool_call_argument(&json!(true)), "true");
        assert_eq!(format_tool_call_argument(&Value::Null), "null");
    }

    /// Upstream `TestQwen3ToolDefinitionTypes`.
    #[test]
    fn tool_definition_types_match_upstream() {
        assert_eq!(
            format_tool_definition_type(&PropertyType(vec!["string".into()])),
            "string"
        );
        assert_eq!(
            format_tool_definition_type(&PropertyType(vec!["string".into(), "number".into()])),
            r#"["string","number"]"#
        );
        assert_eq!(format_tool_definition_type(&PropertyType(vec![])), "[]");
    }

    /// If tools are offered but nobody wrote a system message, upstream injects
    /// the reference implementation's default. Dropping it changes the framing.
    #[test]
    fn tools_without_a_system_message_get_the_default_qwen_persona() {
        let msgs = [msg("user", "hi")];
        let tools = [tool(json!({
            "type": "",
            "function": {"name": "noop", "parameters": {"type": "", "properties": {}}}
        }))];
        let got = Qwen3CoderRenderer.render(&msgs, &tools, None).unwrap();
        assert!(got.starts_with(
            "<|im_start|>system\nYou are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.\n\n# Tools"
        ), "{got}");
    }
}
