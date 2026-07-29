//! **FunctionGemma** -- Gemma turn markers, a function-calling dialect of its
//! own.
//!
//! **Upstream:** `model/renderers/functiongemma.go`. Registered as
//! `functiongemma`.
//!
//! ## The framing
//!
//! BOS: **`<bos>`**, written into the prompt *and* reported by
//! [`Renderer::leading_bos`] -- do not add it twice.
//!
//! Turn markers are the classic Gemma pair **`<start_of_turn>role`** /
//! **`<end_of_turn>`**, with the assistant spelled **`model`** and the system
//! prompt living under the role **`developer`**. On top:
//!
//! * **`<start_function_declaration>` / `<end_function_declaration>`** -- tool
//!   schemas,
//! * **`<start_function_call>` / `<end_function_call>`** -- calls,
//! * **`<start_function_response>` / `<end_function_response>`** -- results,
//! * **`<escape>`** as the string delimiter, on **both** sides of a value
//!   (`<escape>Paris<escape>`). Not `<escape>...</escape>` -- the same token
//!   opens and closes. Values are not escaped inside it, same as Gemma 4's
//!   [`super::gemma4::G4Q`].
//!
//! ## Things worth knowing
//!
//! * **The default system line is `"You can do function calling with the
//!   following functions:"`**, added whenever tools are present -- but **not**
//!   if the caller's own system message is already exactly that text. That
//!   duplicate check is upstream's and it compares the *trimmed* system message.
//! * **A tool result knows its own name by counting.** There is no
//!   `tool_call_id` matching here: the renderer walks back to the last assistant
//!   turn with tool calls and pairs the Nth `tool` message with the Nth call. So
//!   dropping or reordering a tool result silently mislabels the rest.
//! * **`<start_function_response>` is opened by whoever gets there first.** The
//!   assistant turn opens it if a `tool` message follows; otherwise the `tool`
//!   branch opens it. That is the `prev_message_type` state machine, and it is
//!   why the whole exchange ends with `<end_function_response>` and **no**
//!   generation prompt -- the model is expected to carry straight on.
//! * **A property declaration always writes `description:<escape><escape>`**,
//!   even when the description is empty. Upstream's `numeric_arguments` fixture
//!   pins the empty pair; omitting it changes the framing.

use std::collections::BTreeMap;

use serde_json::Value;

use super::{Message, RenderError, Renderer, ThinkValue, Tool};
use crate::api::{ToolCall, ToolProperty};

/// The delimiter that wraps every string. Opens *and* closes -- there is no
/// `</escape>`.
const ESC: &str = "<escape>";

/// **Upstream:** `defaultSystemMessage` in `functiongemma.go`.
const DEFAULT_SYSTEM_MESSAGE: &str = "You can do function calling with the following functions:";

/// **Upstream:** `FunctionGemmaRenderer`. Carries no configuration.
#[derive(Debug, Clone, Copy, Default)]
pub struct FunctionGemmaRenderer;

/// **Upstream:** `formatArgValue`. A string is delimited, a whole float loses
/// its `.0`, maps sort their keys and write them **bare**.
fn format_arg_value(v: &Value) -> String {
    match v {
        Value::String(s) => format!("{ESC}{s}{ESC}"),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => {
            if let Some(f) = n.as_f64()
                && f == (f as i64) as f64
            {
                return format!("{}", f as i64);
            }
            n.to_string()
        }
        Value::Object(m) => {
            // `serde_json::Map` is sorted here, matching Go's `sort.Strings`.
            let mut s = String::from("{");
            for (i, (k, val)) in m.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&format!("{k}:{}", format_arg_value(val)));
            }
            s.push('}');
            s
        }
        Value::Array(a) => {
            let mut s = String::from("[");
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&format_arg_value(item));
            }
            s.push(']');
            s
        }
        // Upstream's `default: fmt.Sprintf("%v", value)` -- a Go `nil` prints
        // as `<nil>`, which is not something any real tool call produces.
        // Emitting the JSON `null` is the sane reading and is stated here
        // rather than silently.
        Value::Null => "null".to_string(),
    }
}

impl FunctionGemmaRenderer {
    /// **Upstream:** `renderToolDeclaration`.
    fn render_tool_declaration(&self, tool: &Tool) -> String {
        let f = &tool.function;
        let mut sb = String::new();
        sb.push_str(&format!(
            "<start_function_declaration>declaration:{}{{",
            f.name
        ));
        sb.push_str(&format!("description:{ESC}{}{ESC}", f.description));

        if f.parameters.has_properties() || !f.parameters.param_type.is_empty() {
            sb.push_str(",parameters:{");
            let mut needs_comma = false;

            if f.parameters.has_properties() {
                sb.push_str("properties:{");
                self.write_properties(&mut sb, f.parameters.properties_iter());
                sb.push('}');
                needs_comma = true;
            }

            if !f.parameters.required.is_empty() {
                if needs_comma {
                    sb.push(',');
                }
                sb.push_str("required:[");
                for (i, r) in f.parameters.required.iter().enumerate() {
                    if i > 0 {
                        sb.push(',');
                    }
                    sb.push_str(&format!("{ESC}{r}{ESC}"));
                }
                sb.push(']');
                needs_comma = true;
            }

            if !f.parameters.param_type.is_empty() {
                if needs_comma {
                    sb.push(',');
                }
                sb.push_str(&format!(
                    "type:{ESC}{}{ESC}",
                    f.parameters.param_type.to_uppercase()
                ));
            }

            sb.push('}');
        }

        sb.push_str("}<end_function_declaration>");
        sb
    }

    /// **Upstream:** `writeProperties`. Keys **sorted**, and only the FIRST type
    /// of a union survives -- this dialect has no union form.
    fn write_properties<'a, I>(&self, sb: &mut String, props: I)
    where
        I: Iterator<Item = (&'a String, &'a ToolProperty)>,
    {
        let sorted: BTreeMap<&String, &ToolProperty> = props.collect();
        let mut first = true;
        for (name, prop) in sorted {
            if !first {
                sb.push(',');
            }
            first = false;
            // The empty-description pair is deliberate -- see the module docs.
            sb.push_str(&format!(
                "{name}:{{description:{ESC}{}{ESC}",
                prop.description
            ));
            if let Some(t) = prop.prop_type.0.first() {
                sb.push_str(&format!(",type:{ESC}{}{ESC}", t.to_uppercase()));
            }
            sb.push('}');
        }
    }

    /// **Upstream:** `formatToolCall`. Argument keys sorted.
    fn format_tool_call(&self, tc: &ToolCall) -> String {
        let sorted: BTreeMap<&String, &Value> = tc.function.arguments.0.iter().collect();
        let mut sb = format!("<start_function_call>call:{}{{", tc.function.name);
        for (i, (key, value)) in sorted.iter().enumerate() {
            if i > 0 {
                sb.push(',');
            }
            sb.push_str(&format!("{key}:{}", format_arg_value(value)));
        }
        sb.push_str("}<end_function_call>");
        sb
    }

    /// Which tool a `tool` message answers, worked out by **position**.
    ///
    /// **Upstream:** the inline loop in the `case "tool"` arm. Walk back to the
    /// nearest assistant turn that made calls, count how many `tool` messages
    /// sit between it and this one, and take that call. No id matching happens
    /// -- so a missing tool result mislabels every result after it.
    fn tool_response_name(&self, messages: &[Message], i: usize) -> String {
        for j in (0..i).rev() {
            if messages[j].role == "assistant" && !messages[j].tool_calls.is_empty() {
                let tool_idx = messages[j + 1..i]
                    .iter()
                    .filter(|m| m.role == "tool")
                    .count();
                return messages[j]
                    .tool_calls
                    .get(tool_idx)
                    .map(|tc| tc.function.name.clone())
                    .unwrap_or_default();
            }
        }
        String::new()
    }
}

impl Renderer for FunctionGemmaRenderer {
    fn leading_bos(&self) -> &'static str {
        "<bos>"
    }

    fn render(
        &self,
        messages: &[Message],
        tools: &[Tool],
        _think: Option<&ThinkValue>,
    ) -> Result<String, RenderError> {
        let mut sb = String::from("<bos>");

        let has_system = messages
            .first()
            .is_some_and(|m| m.role == "system" || m.role == "developer");
        let (system_message, loop_messages): (&str, &[Message]) = if has_system {
            (&messages[0].content, &messages[1..])
        } else {
            ("", messages)
        };

        if !system_message.is_empty() || !tools.is_empty() {
            sb.push_str("<start_of_turn>developer\n");
            if !system_message.is_empty() {
                sb.push_str(system_message.trim());
            }
            if !tools.is_empty() {
                if !system_message.is_empty() {
                    sb.push('\n');
                }
                // Skip the default line if the caller already wrote it.
                if system_message.trim() != DEFAULT_SYSTEM_MESSAGE {
                    sb.push_str(DEFAULT_SYSTEM_MESSAGE);
                }
            }
            for tool in tools {
                sb.push_str(&self.render_tool_declaration(tool));
            }
            sb.push_str("<end_of_turn>\n");
        }

        let mut prev_message_type = "";

        for (i, message) in loop_messages.iter().enumerate() {
            match message.role.as_str() {
                "assistant" => {
                    if prev_message_type != "tool_response" {
                        sb.push_str("<start_of_turn>model\n");
                    }
                    prev_message_type = "";

                    if !message.content.is_empty() {
                        sb.push_str(message.content.trim());
                    }

                    if !message.tool_calls.is_empty() {
                        for tc in &message.tool_calls {
                            sb.push_str(&self.format_tool_call(tc));
                        }
                        // Open the response block ourselves if an answer is
                        // coming; otherwise close the turn.
                        if loop_messages
                            .get(i + 1)
                            .is_some_and(|next| next.role == "tool")
                        {
                            sb.push_str("<start_function_response>");
                            prev_message_type = "tool_call";
                        } else {
                            sb.push_str("<end_of_turn>\n");
                        }
                    } else {
                        sb.push_str("<end_of_turn>\n");
                    }
                }
                "user" => {
                    if prev_message_type != "tool_response" {
                        sb.push_str("<start_of_turn>user\n");
                    }
                    prev_message_type = "";
                    sb.push_str(message.content.trim());
                    sb.push_str("<end_of_turn>\n");
                }
                "tool" => {
                    let tool_name = self.tool_response_name(loop_messages, i);
                    if prev_message_type != "tool_call" {
                        sb.push_str("<start_function_response>");
                    }
                    sb.push_str(&format!(
                        "response:{tool_name}{{{}}}<end_function_response>",
                        format_arg_value(&Value::String(message.content.clone()))
                    ));
                    prev_message_type = "tool_response";
                }
                role => {
                    sb.push_str(&format!("<start_of_turn>{role}\n"));
                    sb.push_str(message.content.trim());
                    sb.push_str("<end_of_turn>\n");
                }
            }
        }

        // No generation prompt after a tool response -- the model continues
        // straight from `<end_function_response>`.
        if prev_message_type != "tool_response" {
            sb.push_str("<start_of_turn>model\n");
        }

        Ok(sb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ToolCallFunction;
    use serde_json::json;

    fn r() -> FunctionGemmaRenderer {
        FunctionGemmaRenderer
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

    fn weather_tool() -> Tool {
        serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City"}}
                }
            }
        }))
        .expect("valid fixture tool")
    }

    const WEATHER_DECL: &str = "<start_function_declaration>declaration:get_weather{description:<escape>Get weather<escape>,parameters:{properties:{city:{description:<escape>City<escape>,type:<escape>STRING<escape>}},type:<escape>OBJECT<escape>}}<end_function_declaration>";

    /// Upstream `TestFunctionGemmaRenderer`'s turn-shape cases.
    #[test]
    fn functiongemma_matches_the_upstream_turn_fixtures() {
        assert_eq!(
            r().render(&[Message::new("user", "Hello!")], &[], None)
                .unwrap(),
            "<bos><start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n"
        );

        // A system message renders under the `developer` role.
        assert_eq!(
            r().render(
                &[
                    Message::new("system", "You are helpful"),
                    Message::new("user", "Hello!"),
                ],
                &[],
                None
            )
            .unwrap(),
            "<bos><start_of_turn>developer\nYou are helpful<end_of_turn>\n<start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n"
        );

        // ...and `developer` is accepted as the incoming role name too.
        assert_eq!(
            r().render(
                &[
                    Message::new("developer", "You are a coding assistant"),
                    Message::new("user", "Hello!"),
                ],
                &[],
                None
            )
            .unwrap(),
            "<bos><start_of_turn>developer\nYou are a coding assistant<end_of_turn>\n<start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n"
        );

        assert_eq!(
            r().render(
                &[
                    Message::new("user", "Hi"),
                    Message::new("assistant", "Hello!"),
                    Message::new("user", "More"),
                ],
                &[],
                None
            )
            .unwrap(),
            "<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n<start_of_turn>user\nMore<end_of_turn>\n<start_of_turn>model\n"
        );

        // No messages at all still gets a generation prompt.
        assert_eq!(
            r().render(&[], &[], None).unwrap(),
            "<bos><start_of_turn>model\n"
        );
    }

    /// Upstream `with_tools` / `custom_system_message_with_tools` /
    /// `developer_role_with_tools`.
    #[test]
    fn tools_append_the_default_line_after_any_custom_system_text() {
        assert_eq!(
            r().render(&[Message::new("user", "Weather?")], &[weather_tool()], None)
                .unwrap(),
            format!(
                "<bos><start_of_turn>developer\n{DEFAULT_SYSTEM_MESSAGE}{WEATHER_DECL}<end_of_turn>\n\
                 <start_of_turn>user\nWeather?<end_of_turn>\n<start_of_turn>model\n"
            )
        );

        assert_eq!(
            r().render(
                &[
                    Message::new("system", "You are a weather expert."),
                    Message::new("user", "Weather?"),
                ],
                &[weather_tool()],
                None
            )
            .unwrap(),
            format!(
                "<bos><start_of_turn>developer\nYou are a weather expert.\n{DEFAULT_SYSTEM_MESSAGE}{WEATHER_DECL}<end_of_turn>\n\
                 <start_of_turn>user\nWeather?<end_of_turn>\n<start_of_turn>model\n"
            )
        );
    }

    /// The duplicate guard: a caller who already wrote the default line does not
    /// get it twice.
    #[test]
    fn the_default_line_is_not_repeated_when_the_caller_supplied_it() {
        let got = r()
            .render(
                &[
                    Message::new("system", DEFAULT_SYSTEM_MESSAGE),
                    Message::new("user", "Weather?"),
                ],
                &[weather_tool()],
                None,
            )
            .unwrap();
        assert_eq!(got.matches(DEFAULT_SYSTEM_MESSAGE).count(), 1, "{got}");
    }

    /// Upstream `tool_call`, `assistant_content_with_tool_call` and
    /// `numeric_arguments`. Note the exchange ends at
    /// `<end_function_response>` with **no** generation prompt.
    #[test]
    fn a_tool_round_trip_ends_without_a_generation_prompt() {
        assert_eq!(
            r().render(
                &[
                    Message::new("user", "Weather?"),
                    Message {
                        role: "assistant".into(),
                        tool_calls: vec![call("get_weather", r#"{"city":"Paris"}"#)],
                        ..Default::default()
                    },
                    Message::new("tool", "Sunny"),
                ],
                &[weather_tool()],
                None
            )
            .unwrap(),
            format!(
                "<bos><start_of_turn>developer\n{DEFAULT_SYSTEM_MESSAGE}{WEATHER_DECL}<end_of_turn>\n\
                 <start_of_turn>user\nWeather?<end_of_turn>\n\
                 <start_of_turn>model\n<start_function_call>call:get_weather{{city:<escape>Paris<escape>}}<end_function_call>\
                 <start_function_response>response:get_weather{{<escape>Sunny<escape>}}<end_function_response>"
            )
        );

        // Content plus a call: the content goes first, undecorated.
        let got = r()
            .render(
                &[
                    Message::new("user", "Weather?"),
                    Message {
                        role: "assistant".into(),
                        content: "Let me check.".into(),
                        tool_calls: vec![call("get_weather", r#"{"city":"Paris"}"#)],
                        ..Default::default()
                    },
                    Message::new("tool", "Sunny"),
                ],
                &[weather_tool()],
                None,
            )
            .unwrap();
        assert!(
            got.contains("<start_of_turn>model\nLet me check.<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>"),
            "{got}"
        );
    }

    /// Upstream `numeric_arguments` -- whole numbers lose their `.0`, and a
    /// property with no description still writes the empty `<escape><escape>`
    /// pair.
    #[test]
    fn numbers_are_bare_and_an_empty_description_still_renders() {
        let add_tool: Tool = serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "add", "description": "Add numbers",
                "parameters": {"type": "object", "properties": {
                    "a": {"type": "number"}, "b": {"type": "number"}
                }}
            }
        }))
        .unwrap();
        let got = r()
            .render(
                &[
                    Message::new("user", "Add"),
                    Message {
                        role: "assistant".into(),
                        tool_calls: vec![call("add", r#"{"a":1.0,"b":2.0}"#)],
                        ..Default::default()
                    },
                    Message::new("tool", "3"),
                ],
                &[add_tool],
                None,
            )
            .unwrap();
        assert!(
            got.contains("properties:{a:{description:<escape><escape>,type:<escape>NUMBER<escape>},b:{description:<escape><escape>,type:<escape>NUMBER<escape>}}"),
            "{got}"
        );
        assert!(
            got.contains("<start_function_call>call:add{a:1,b:2}<end_function_call>"),
            "{got}"
        );
        assert!(
            got.ends_with(
                "<start_function_response>response:add{<escape>3<escape>}<end_function_response>"
            ),
            "{got}"
        );
    }

    /// The positional tool-name pairing, which has no id fallback: the second
    /// `tool` message belongs to the second call.
    #[test]
    fn tool_results_are_paired_with_calls_by_position() {
        let got = r()
            .render(
                &[
                    Message::new("user", "Both"),
                    Message {
                        role: "assistant".into(),
                        tool_calls: vec![call("first", "{}"), call("second", "{}")],
                        ..Default::default()
                    },
                    Message::new("tool", "r1"),
                    Message::new("tool", "r2"),
                ],
                &[],
                None,
            )
            .unwrap();
        assert!(got.contains("response:first{<escape>r1<escape>}"), "{got}");
        assert!(got.contains("response:second{<escape>r2<escape>}"), "{got}");
    }
}
