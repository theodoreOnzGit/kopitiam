//! **LFM2** (Liquid AI) -- ChatML turns with Liquid's own tool wrappers.
//!
//! **Upstream:** `model/renderers/lfm2.go`. Registered as `lfm2` and
//! `lfm2-thinking`.
//!
//! ## The framing
//!
//! BOS: **`<|startoftext|>`** -- and unlike most families it is **also written
//! into the prompt**, at the very front. Turn markers are the usual
//! **`<|im_start|>`** / **`<|im_end|>`**. On top of those:
//!
//! * **`<|tool_list_start|>` / `<|tool_list_end|>`** around the tool schemas,
//! * **`<|tool_call_start|>` / `<|tool_call_end|>`** around the calls,
//! * **`<|tool_response_start|>` / `<|tool_response_end|>`** around a result,
//! * **`<think>` / `</think>`**, reconstructed inline (see below),
//! * **`<image>`** as the native image placeholder (or `[img]` in marker mode).
//!
//! Tool results get their **own `<|im_start|>tool` turn** here -- no dressing
//! up as a user message the way Qwen does it.
//!
//! ## Thinking is rebuilt into the content, then selectively stripped
//!
//! [`Message::thinking`] is a separate field in our vocabulary, but LFM2 was
//! trained on an **inline** `<think>...</think>` prefix. So the renderer glues
//! it back on, in front of both tool calls and content. Then:
//!
//! * if `keep_past_thinking` is off (`is_thinking` false, **or** the caller did
//!   not explicitly ask for thinking), every assistant turn except the **last**
//!   one has everything up to and including its final `</think>` cut away;
//! * the non-thinking renderer never emits the tags at all.
//!
//! What would make this wrong: treating `think = None` as "on". It is not --
//! `keep_past_thinking` needs an explicit `Some(true)`.
//!
//! ## Its own JSON dialect, and this is not a nit
//!
//! `lfm2JSON` upstream sets **`SetEscapeHTML(false)`**, unlike every other
//! renderer here, which use Go's HTML-escaping default. So a tool description
//! containing `<`, `>` or `&` comes out **raw** in an LFM2 prompt and
//! **escaped** in a GLM one. That is why this file carries its own little
//! emitter instead of reusing [`super::json`] -- exactly mirroring why upstream
//! has a separate `lfm2JSON`. Merging the two would silently change one
//! family's prompt.

use serde_json::Value;

use super::image_tags::render_content_with_image_tags;
use super::{IM_END_TAG, IM_START_TAG, Message, RenderError, Renderer, ThinkValue, Tool};
use crate::api::{ToolCall, ToolFunction};

const BOS_TOKEN: &str = "<|startoftext|>";
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TOOL_LIST_START: &str = "<|tool_list_start|>";
const TOOL_LIST_END: &str = "<|tool_list_end|>";
const TOOL_CALL_START: &str = "<|tool_call_start|>";
const TOOL_CALL_END: &str = "<|tool_call_end|>";
const TOOL_RESPONSE_START: &str = "<|tool_response_start|>";
const TOOL_RESPONSE_END: &str = "<|tool_response_end|>";

/// **Upstream:** `LFM2Renderer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Lfm2Renderer {
    /// Is this the `-thinking` checkpoint? Only that one may emit `<think>`
    /// tags at all.
    pub is_thinking: bool,
    /// Use `[img]` markers instead of the native `<image>` placeholder.
    pub use_img_tags: bool,
}

/// A JSON string, **without** Go's HTML escaping. See the module docs for why
/// this exists separately.
fn write_raw_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A JSON value, no HTML escaping, object keys sorted (Go marshals maps sorted).
fn write_raw_value(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => write_raw_string(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_raw_value(item, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_raw_string(k, out);
                out.push(':');
                write_raw_value(&m[*k], out);
            }
            out.push('}');
        }
    }
}

/// One `ToolProperty`, no HTML escaping, **Go declaration order** (`anyOf`,
/// `type`, `items`, `description`, `enum`, `properties`, `required`), every
/// field `omitempty`.
///
/// This is [`super::json::write_go_property`] with the escaping taken out --
/// see the module docs for why the two cannot be one function. Note the field
/// order matters even here: sorting it (which is what handing the struct to a
/// generic value-writer would do) would reorder a JSON Schema the model was
/// trained to read in a fixed order.
fn write_raw_property(p: &crate::api::ToolProperty, out: &mut String) {
    let mut first = true;
    let mut key = |out: &mut String, k: &str| {
        if !first {
            out.push(',');
        }
        first = false;
        write_raw_string(k, out);
        out.push(':');
    };
    out.push('{');
    if !p.any_of.is_empty() {
        key(out, "anyOf");
        out.push('[');
        for (i, a) in p.any_of.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_raw_property(a, out);
        }
        out.push(']');
    }
    if !p.prop_type.is_empty() {
        key(out, "type");
        // `api.PropertyType.MarshalJSON`: one type -> a bare string.
        if p.prop_type.0.len() == 1 {
            write_raw_string(&p.prop_type.0[0], out);
        } else {
            out.push('[');
            for (i, t) in p.prop_type.0.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_raw_string(t, out);
            }
            out.push(']');
        }
    }
    if let Some(items) = &p.items {
        key(out, "items");
        write_raw_value(items, out);
    }
    if !p.description.is_empty() {
        key(out, "description");
        write_raw_string(&p.description, out);
    }
    if !p.enum_values.is_empty() {
        key(out, "enum");
        out.push('[');
        for (i, e) in p.enum_values.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_raw_value(e, out);
        }
        out.push(']');
    }
    if let Some(props) = &p.properties {
        key(out, "properties");
        out.push('{');
        for (i, (k, v)) in props.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_raw_string(k, out);
            out.push(':');
            write_raw_property(v, out);
        }
        out.push('}');
    }
    if !p.required.is_empty() {
        key(out, "required");
        out.push('[');
        for (i, r) in p.required.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_raw_string(r, out);
        }
        out.push(']');
    }
    out.push('}');
}

/// The `', '` / `': '` pass. **Upstream:** the tail of `lfm2JSON`.
///
/// Upstream only inserts the space when the next byte is not already
/// whitespace. Compact JSON never has whitespace outside strings, so this can
/// never actually fire -- but it is copied because a future non-compact input
/// would then behave the same.
fn add_spaces(json: &str) -> String {
    let b = json.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len() + b.len() / 8);
    let (mut in_str, mut esc) = (false, false);
    for (i, &c) in b.iter().enumerate() {
        out.push(c);
        if in_str {
            if esc {
                esc = false;
                continue;
            }
            if c == b'\\' {
                esc = true;
                continue;
            }
            if c == b'"' {
                in_str = false;
            }
            continue;
        }
        if c == b'"' {
            in_str = true;
            continue;
        }
        if (c == b':' || c == b',') && i + 1 < b.len() {
            let next = b[i + 1];
            if next != b' ' && next != b'\n' && next != b'\r' && next != b'\t' {
                out.push(b' ');
            }
        }
    }
    String::from_utf8(out).unwrap_or_else(|_| json.to_string())
}

/// `lfm2JSON(value)` for an arbitrary argument value.
fn lfm2_json_value(v: &Value) -> String {
    let mut s = String::new();
    write_raw_value(v, &mut s);
    add_spaces(&s)
}

/// `lfm2JSON(lfm2ToolSchema(tool))`.
///
/// **Upstream:** `lfm2ToolSchema` hands the template the bare
/// `api.ToolFunction` (name / description / parameters) whenever the tool has a
/// name, on the grounds that *"LFM2 templates are typically fed function-schema
/// objects"*. Only a nameless tool falls back to the full `api.Tool`, which in
/// practice never happens -- so this only implements the function-schema shape
/// and states the omission rather than pretending to cover it.
fn lfm2_tool_schema_json(tool: &Tool) -> String {
    let f: &ToolFunction = &tool.function;
    let mut s = String::from("{");
    write_raw_string("name", &mut s);
    s.push(':');
    write_raw_string(&f.name, &mut s);
    if !f.description.is_empty() {
        s.push(',');
        write_raw_string("description", &mut s);
        s.push(':');
        write_raw_string(&f.description, &mut s);
    }
    s.push(',');
    write_raw_string("parameters", &mut s);
    s.push(':');
    // Go declaration order: type, $defs, items, required, properties.
    s.push('{');
    write_raw_string("type", &mut s);
    s.push(':');
    write_raw_string(&f.parameters.param_type, &mut s);
    if let Some(defs) = &f.parameters.defs {
        s.push_str(",\"$defs\":");
        write_raw_value(defs, &mut s);
    }
    if let Some(items) = &f.parameters.items {
        s.push_str(",\"items\":");
        write_raw_value(items, &mut s);
    }
    if !f.parameters.required.is_empty() {
        s.push_str(",\"required\":[");
        for (i, r) in f.parameters.required.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            write_raw_string(r, &mut s);
        }
        s.push(']');
    }
    s.push_str(",\"properties\":");
    if !f.parameters.has_properties() {
        // Nil `*ToolPropertiesMap` upstream -- and this family's fixture is the
        // one that pins it. See [`super::json`].
        s.push_str("null");
    } else {
        s.push('{');
        for (i, (k, v)) in f.parameters.properties_iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            write_raw_string(k, &mut s);
            s.push(':');
            write_raw_property(v, &mut s);
        }
        s.push('}');
    }
    s.push_str("}}");
    add_spaces(&s)
}

/// `<|tool_call_start|>[name(k=v,...)]<|tool_call_end|>`.
///
/// **Upstream:** `lfm2RenderToolCalls`. Argument keys are **sorted**, arguments
/// separated by a bare `,` (no space), and each value JSON-encoded -- so a
/// string keeps its quotes.
fn render_tool_calls(calls: &[ToolCall]) -> String {
    let mut sb = String::new();
    sb.push_str(TOOL_CALL_START);
    sb.push('[');
    for (i, tc) in calls.iter().enumerate() {
        if i > 0 {
            sb.push(',');
        }
        sb.push_str(&tc.function.name);
        sb.push('(');
        let mut keys: Vec<&String> = tc.function.arguments.0.keys().collect();
        keys.sort();
        for (j, key) in keys.iter().enumerate() {
            if j > 0 {
                sb.push(',');
            }
            let value = tc.function.arguments.get(key).cloned().unwrap_or_default();
            sb.push_str(key);
            sb.push('=');
            sb.push_str(&lfm2_json_value(&value));
        }
        sb.push(')');
    }
    sb.push(']');
    sb.push_str(TOOL_CALL_END);
    sb
}

impl Lfm2Renderer {
    /// **Upstream:** `(*LFM2Renderer).renderMessageContent`.
    fn render_message_content(&self, message: &Message, image_offset: usize) -> String {
        let content = message.content.clone();
        if message.images.is_empty() {
            return content;
        }
        if self.use_img_tags {
            let (content, _) =
                render_content_with_image_tags(&content, message.images.len(), image_offset);
            return content;
        }
        // Native placeholder: only prepend when the caller has not already
        // written one into the text.
        if content.contains("<image>") {
            return content;
        }
        let mut sb = String::new();
        for _ in &message.images {
            sb.push_str("<image>");
        }
        sb.push_str(&content);
        sb
    }
}

impl Renderer for Lfm2Renderer {
    fn leading_bos(&self) -> &'static str {
        BOS_TOKEN
    }

    fn render(
        &self,
        messages: &[Message],
        tools: &[Tool],
        think: Option<&ThinkValue>,
    ) -> Result<String, RenderError> {
        let mut sb = String::from(BOS_TOKEN);

        // The first system message, if any, is combined with the tool list.
        let mut first_system_content = String::new();
        let mut start_idx = 0usize;
        if let Some(first) = messages.first()
            && first.role == "system"
        {
            first_system_content = first.content.clone();
            start_idx = 1;
        }

        if !tools.is_empty() {
            if !first_system_content.is_empty() {
                first_system_content.push('\n');
            }
            first_system_content.push_str("List of tools: ");
            first_system_content.push_str(TOOL_LIST_START);
            first_system_content.push('[');
            for (i, tool) in tools.iter().enumerate() {
                first_system_content.push_str(&lfm2_tool_schema_json(tool));
                if i < tools.len() - 1 {
                    first_system_content.push_str(", ");
                }
            }
            first_system_content.push(']');
            first_system_content.push_str(TOOL_LIST_END);
        }

        if !first_system_content.is_empty() {
            sb.push_str(IM_START_TAG);
            sb.push_str("system\n");
            sb.push_str(&first_system_content);
            sb.push_str(IM_END_TAG);
            sb.push('\n');
        }

        // Needs BOTH: a thinking checkpoint AND an explicit ask.
        let keep_past_thinking = self.is_thinking && think.is_some_and(|t| t.enabled());

        let last_assistant_index = messages
            .iter()
            .enumerate()
            .skip(start_idx)
            .rev()
            .find(|(_, m)| m.role == "assistant")
            .map(|(i, _)| i);

        let mut image_offset: usize = messages[..start_idx].iter().map(|m| m.images.len()).sum();

        for (i, message) in messages.iter().enumerate().skip(start_idx) {
            let last_message = i == messages.len() - 1;
            let prefill = last_message && message.role == "assistant";

            sb.push_str(IM_START_TAG);
            sb.push_str(&message.role);
            sb.push('\n');

            let mut content = self.render_message_content(message, image_offset);
            image_offset += message.images.len();

            if message.role == "assistant"
                && !message.tool_calls.is_empty()
                && !content.contains(TOOL_CALL_START)
            {
                let calls = render_tool_calls(&message.tool_calls);
                content = if content.trim().is_empty() {
                    calls + &content
                } else {
                    calls + "\n" + &content
                };
            }

            // Rebuild the inline think block from the separate field.
            if self.is_thinking
                && message.role == "assistant"
                && !message.thinking.is_empty()
                && !content.contains(THINK_CLOSE)
            {
                content = format!("{THINK_OPEN}{}{THINK_CLOSE}{content}", message.thinking);
            }

            // Drop reasoning from earlier assistant turns unless it is kept.
            if message.role == "assistant"
                && !keep_past_thinking
                && Some(i) != last_assistant_index
                && let Some(idx) = content.rfind(THINK_CLOSE)
            {
                content = content[idx + THINK_CLOSE.len()..].trim().to_string();
            }

            if message.role == "tool" && !content.contains(TOOL_RESPONSE_START) {
                content = format!("{TOOL_RESPONSE_START}{content}{TOOL_RESPONSE_END}");
            }

            sb.push_str(&content);
            if !prefill {
                sb.push_str(IM_END_TAG);
                sb.push('\n');
            }
        }

        let needs_generation_prompt = !messages.last().is_some_and(|m| m.role == "assistant");
        if needs_generation_prompt {
            sb.push_str(IM_START_TAG);
            sb.push_str("assistant\n");
        }

        Ok(sb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ToolCallFunction;
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

    fn bare_tool(name: &str) -> Tool {
        serde_json::from_value(json!({
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}}
        }))
        .expect("valid fixture tool")
    }

    fn plain() -> Lfm2Renderer {
        Lfm2Renderer::default()
    }

    fn thinking() -> Lfm2Renderer {
        Lfm2Renderer {
            is_thinking: true,
            use_img_tags: false,
        }
    }

    /// Upstream `TestLFM2Renderer`'s basic and tool-list cases. The
    /// `"properties": null` is upstream's, and it is what pins our nil-vs-empty
    /// choice in [`super::json`].
    #[test]
    fn lfm2_matches_the_upstream_basic_and_tool_list_fixtures() {
        assert_eq!(
            plain()
                .render(&[Message::new("user", "Hello")], &[], None)
                .unwrap(),
            "<|startoftext|><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );

        assert_eq!(
            plain()
                .render(
                    &[
                        Message::new("system", "You are helpful."),
                        Message::new("user", "Hi"),
                    ],
                    &[],
                    None
                )
                .unwrap(),
            "<|startoftext|><|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );

        assert_eq!(
            plain()
                .render(
                    &[Message::new("user", "Use tools")],
                    &[bare_tool("get_weather")],
                    None
                )
                .unwrap(),
            "<|startoftext|><|im_start|>system\nList of tools: <|tool_list_start|>[{\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": null}}]<|tool_list_end|><|im_end|>\n\
             <|im_start|>user\nUse tools<|im_end|>\n<|im_start|>assistant\n"
        );

        // A first system message and the tool list are joined by ONE newline.
        assert_eq!(
            plain()
                .render(
                    &[
                        Message::new("system", "Follow instructions."),
                        Message::new("user", "Do work"),
                    ],
                    &[bare_tool("tool_a"), bare_tool("tool_b")],
                    None
                )
                .unwrap(),
            "<|startoftext|><|im_start|>system\nFollow instructions.\nList of tools: <|tool_list_start|>[{\"name\": \"tool_a\", \"parameters\": {\"type\": \"object\", \"properties\": null}}, {\"name\": \"tool_b\", \"parameters\": {\"type\": \"object\", \"properties\": null}}]<|tool_list_end|><|im_end|>\n\
             <|im_start|>user\nDo work<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// Upstream's tool-call / tool-response fixtures. Tool results get their own
    /// `tool` turn here -- not a user turn.
    #[test]
    fn tool_calls_and_responses_get_their_own_wrappers() {
        assert_eq!(
            plain()
                .render(
                    &[
                        Message::new("user", "Call a tool"),
                        Message {
                            role: "assistant".into(),
                            tool_calls: vec![call("get_weather", r#"{"location":"Paris"}"#)],
                            ..Default::default()
                        },
                        Message::new("tool", "22C"),
                    ],
                    &[],
                    None
                )
                .unwrap(),
            "<|startoftext|><|im_start|>user\nCall a tool<|im_end|>\n\
             <|im_start|>assistant\n<|tool_call_start|>[get_weather(location=\"Paris\")]<|tool_call_end|><|im_end|>\n\
             <|im_start|>tool\n<|tool_response_start|>22C<|tool_response_end|><|im_end|>\n\
             <|im_start|>assistant\n"
        );

        // Calls come FIRST, then a newline, then the content.
        assert_eq!(
            plain()
                .render(
                    &[
                        Message::new("user", "Call a tool"),
                        Message {
                            role: "assistant".into(),
                            content: "Checking now.".into(),
                            tool_calls: vec![call("get_weather", r#"{"location":"Paris"}"#)],
                            ..Default::default()
                        },
                    ],
                    &[],
                    None
                )
                .unwrap(),
            "<|startoftext|><|im_start|>user\nCall a tool<|im_end|>\n\
             <|im_start|>assistant\n<|tool_call_start|>[get_weather(location=\"Paris\")]<|tool_call_end|>\nChecking now."
        );
    }

    /// Upstream's four thinking fixtures. The pivot is `keep_past_thinking`,
    /// which needs an explicit `Some(true)` -- `None` is NOT enough.
    #[test]
    fn past_reasoning_survives_only_when_thinking_is_explicitly_asked_for() {
        let history = || {
            vec![
                Message::new("user", "Q1"),
                Message {
                    role: "assistant".into(),
                    thinking: "reason1".into(),
                    content: "A1".into(),
                    ..Default::default()
                },
                Message::new("user", "Q2"),
                Message {
                    role: "assistant".into(),
                    thinking: "reason2".into(),
                    content: "A2".into(),
                    ..Default::default()
                },
            ]
        };

        // Off (via `None`): the earlier turn loses its reasoning, the last keeps it.
        assert_eq!(
            thinking().render(&history(), &[], None).unwrap(),
            "<|startoftext|><|im_start|>user\nQ1<|im_end|>\n\
             <|im_start|>assistant\nA1<|im_end|>\n\
             <|im_start|>user\nQ2<|im_end|>\n\
             <|im_start|>assistant\n<think>reason2</think>A2"
        );

        // Explicitly on: both keep it.
        assert_eq!(
            thinking()
                .render(&history(), &[], Some(&ThinkValue::Bool(true)))
                .unwrap(),
            "<|startoftext|><|im_start|>user\nQ1<|im_end|>\n\
             <|im_start|>assistant\n<think>reason1</think>A1<|im_end|>\n\
             <|im_start|>user\nQ2<|im_end|>\n\
             <|im_start|>assistant\n<think>reason2</think>A2"
        );

        // A history with no reasoning at all gets no tags invented for it.
        assert_eq!(
            thinking()
                .render(
                    &[
                        Message::new("user", "Q1"),
                        Message::new("assistant", "A1"),
                        Message::new("user", "Q2"),
                    ],
                    &[],
                    None
                )
                .unwrap(),
            "<|startoftext|><|im_start|>user\nQ1<|im_end|>\n\
             <|im_start|>assistant\nA1<|im_end|>\n\
             <|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// Upstream `thinking_precedes_tool_calls`: reasoning goes in FRONT of the
    /// call block, not after it.
    #[test]
    fn reasoning_is_written_before_the_tool_call_block() {
        let got = thinking()
            .render(
                &[
                    Message::new("user", "Weather?"),
                    Message {
                        role: "assistant".into(),
                        thinking: "Let me check the weather.".into(),
                        tool_calls: vec![call("get_weather", r#"{"location":"Paris"}"#)],
                        ..Default::default()
                    },
                    Message::new("tool", "22C"),
                ],
                &[],
                None,
            )
            .unwrap();
        assert_eq!(
            got,
            "<|startoftext|><|im_start|>user\nWeather?<|im_end|>\n\
             <|im_start|>assistant\n<think>Let me check the weather.</think><|tool_call_start|>[get_weather(location=\"Paris\")]<|tool_call_end|><|im_end|>\n\
             <|im_start|>tool\n<|tool_response_start|>22C<|tool_response_end|><|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    /// The non-thinking checkpoint must NEVER emit `<think>` tags, even when the
    /// caller filled the field.
    #[test]
    fn the_non_thinking_variant_never_emits_think_tags() {
        let got = plain()
            .render(
                &[
                    Message::new("user", "Q"),
                    Message {
                        role: "assistant".into(),
                        thinking: "secret".into(),
                        content: "A".into(),
                        ..Default::default()
                    },
                ],
                &[],
                Some(&ThinkValue::Bool(true)),
            )
            .unwrap();
        assert!(!got.contains("<think>"), "{got}");
        assert!(!got.contains("secret"), "{got}");
    }
}
