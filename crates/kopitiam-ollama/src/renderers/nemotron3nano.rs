//! **Nemotron 3 Nano** -- ChatML turns, XML tool schemas, Python-flavoured JSON.
//!
//! **Upstream:** `model/renderers/nemotron3nano.go`. Registered as
//! `nemotron-3-nano`. No BOS.
//!
//! ## The framing, in full
//!
//! Special tokens: **`<|im_start|>`** / **`<|im_end|>`** ([`IM_START_TAG`] /
//! [`IM_END_TAG`]), plus `<think>` / `</think>` and the `<tool_call>` /
//! `<function=...>` / `<parameter=...>` XML that Qwen3-Coder also uses.
//!
//! ```text
//! \n\n\n<|im_start|>system
//! {system}{tools}<|im_end|>
//!
//! <|im_start|>user
//! {text}<|im_end|>
//!
//! <|im_start|>assistant
//! <think>
//! ```
//!
//! Four things that look like typos but are load-bearing -- copy them, don't
//! tidy them:
//!
//! * **The prompt opens with three bare newlines**, always, before the system
//!   turn. Jinja artefact upstream inherited from the reference template; the
//!   model saw it during fine-tuning, so we emit it.
//! * **The system turn is ALWAYS emitted**, even when there is no system message
//!   and no tools -- you get a bare `<|im_start|>system\n<|im_end|>\n\n`.
//! * **There is a blank line after the system turn** (`<|im_end|>\n\n`) and
//!   another before the generation prompt (an extra `\n` after the loop). Every
//!   *other* turn closes with a single `\n`.
//! * **The generation prompt has two shapes.** Thinking on -> `<think>\n` (open,
//!   the model closes it). Thinking off -> `<think></think>` with **no trailing
//!   newline at all**. Add one and the model starts its answer on a line it was
//!   never trained to.
//!
//! ## Thinking is resolved from the text, not just the flag
//!
//! [`resolve_thinking`] starts from the caller's `think` (absent counts as
//! **on**) and then lets **every** `user`/`system` message override it with an
//! inline `/think` or `/no_think` toggle -- last message wins. The toggles are
//! then **scrubbed out of the system message** by [`sanitize_system_message`],
//! but left untouched in user messages (upstream's asymmetry, and its fixtures
//! pin both halves).
//!
//! The `</think>`-dance inside both functions is not paranoia: a literal
//! `</think>` in the text contains no `/think` substring anyway, but
//! `sanitize_system_message` strips `/think` blindly, so it parks `</think>`
//! under a placeholder first and puts it back after. Skip that and a system
//! message mentioning `</think>` comes out as `<>`.
//!
//! ## Assistant history gets truncated
//!
//! Any assistant turn **before the last user turn** has its reasoning thrown
//! away and replaced with an empty `<think></think>` -- the model is not meant
//! to re-read old chains of thought. See [`format_assistant_content`] and
//! [`format_tool_call_content`], which do that job with subtly different rules
//! (the tool-call one also handles a *dangling* `<think>` with no close).
//!
//! ## Images
//!
//! Unlike the Qwen and Gemma renderers, this one has **no `use_img_tags`
//! switch** -- it always runs [`render_content_with_image_tags`]. Upstream's
//! `TestNemotron3NanoRenderer_Images` asserts it with the global flag untouched.

use serde_json::Value;

use super::image_tags::render_content_with_image_tags;
use super::{IM_END_TAG, IM_START_TAG, Message, RenderError, Renderer, ThinkValue, Tool};
use crate::api::{PropertyType, ToolCall, ToolFunctionParameters, ToolProperty};

/// **Upstream:** `Nemotron3NanoRenderer`. Carries no configuration.
#[derive(Debug, Clone, Copy, Default)]
pub struct Nemotron3NanoRenderer;

/// The call-format instructions appended after `</tools>`.
///
/// **Upstream:** the trailing `sb.WriteString(...)` in `renderTools`
/// (`nemotron3nano.go`). Byte-identical to Qwen3-Coder's
/// [`super::qwen3coder::QWEN_TOOL_POSTAMBLE`] -- deliberately **not** shared
/// with it, because the two families are free to drift and a shared constant
/// would silently change one family's prompt when somebody edits the other.
const NEMOTRON_TOOL_POSTAMBLE: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
/// The "I did not think" marker. Note there is **no whitespace** between the
/// two tags -- upstream writes them as one literal.
const THINK_EMPTY: &str = "<think></think>";

impl Renderer for Nemotron3NanoRenderer {
    fn render(
        &self,
        messages: &[Message],
        tools: &[Tool],
        think: Option<&ThinkValue>,
    ) -> Result<String, RenderError> {
        let mut sb = String::new();
        let mut image_offset = 0usize;

        let enable_thinking = resolve_thinking(messages, think);

        // Only a *leading* system message becomes the header. A later one falls
        // through to the `"user" | "system"` arm of the loop and is rendered as
        // its own turn.
        let (system_message, loop_messages) = match messages.first() {
            Some(first) if first.role == "system" => {
                (sanitize_system_message(&first.content), &messages[1..])
            }
            _ => (String::new(), messages),
        };

        // Where thinking-truncation stops. `-1` upstream; `None` here, and every
        // comparison below is "strictly before the last user turn".
        let last_user_idx = loop_messages.iter().rposition(|m| m.role == "user");
        let should_truncate = |i: usize| last_user_idx.is_some_and(|last| i < last);

        sb.push_str("\n\n\n");
        sb.push_str(IM_START_TAG);
        sb.push_str("system\n");
        sb.push_str(&system_message);

        if !tools.is_empty() {
            if !system_message.is_empty() {
                sb.push_str("\n\n");
            }
            sb.push_str(&render_tools(tools));
        }
        sb.push_str(IM_END_TAG);
        sb.push_str("\n\n");

        for (i, message) in loop_messages.iter().enumerate() {
            match message.role.as_str() {
                "assistant" => {
                    let content = build_content(message);
                    let truncate = should_truncate(i);

                    sb.push_str(IM_START_TAG);
                    sb.push_str("assistant\n");
                    if message.tool_calls.is_empty() {
                        sb.push_str(&format_assistant_content(&content, truncate));
                    } else {
                        sb.push_str(&format_tool_call_content(&content, truncate));
                        write_tool_calls(&mut sb, &message.tool_calls);
                    }
                    sb.push_str(IM_END_TAG);
                    sb.push('\n');
                }
                role @ ("user" | "system") => {
                    sb.push_str(IM_START_TAG);
                    sb.push_str(role);
                    sb.push('\n');
                    sb.push_str(&render_message_content(message, image_offset));
                    image_offset += message.images.len();
                    sb.push_str(IM_END_TAG);
                    sb.push('\n');
                }
                "tool" => {
                    // Consecutive tool results share ONE `<|im_start|>user`
                    // block: the opener is skipped when the previous message was
                    // also a tool, and the closer is skipped when the next one
                    // is. Emit one block each and the model sees a conversation
                    // with turns that never happened.
                    let prev_was_tool = i > 0 && loop_messages[i - 1].role == "tool";
                    let next_is_tool = loop_messages.get(i + 1).is_some_and(|m| m.role == "tool");
                    let content = render_message_content(message, image_offset);
                    image_offset += message.images.len();

                    if !prev_was_tool {
                        sb.push_str(IM_START_TAG);
                        sb.push_str("user\n");
                    }
                    sb.push_str("<tool_response>\n");
                    sb.push_str(&content);
                    sb.push_str("\n</tool_response>\n");

                    if !next_is_tool {
                        sb.push_str(IM_END_TAG);
                        sb.push('\n');
                    }
                }
                // Any other role is passed through verbatim -- upstream's
                // `default` arm, and its `fallback role` fixture pins it. Note
                // this path does NOT sanitise or image-tag the content.
                other => {
                    sb.push_str(IM_START_TAG);
                    sb.push_str(other);
                    sb.push('\n');
                    sb.push_str(&message.content);
                    sb.push_str(IM_END_TAG);
                    sb.push('\n');
                }
            }
        }

        sb.push('\n');

        sb.push_str(IM_START_TAG);
        sb.push_str("assistant\n");
        if enable_thinking {
            // Open and leave it -- the model writes the close itself.
            sb.push_str(THINK_OPEN);
            sb.push('\n');
        } else {
            // Closed immediately, and **no trailing newline**.
            sb.push_str(THINK_EMPTY);
        }

        Ok(sb)
    }
}

/// The `# Tools` block. **Upstream:** `(*Nemotron3NanoRenderer).renderTools`.
///
/// The schema is **XML, not JSON** -- one `<function>` per tool, one
/// `<parameter>` per property, in the caller's insertion order. Everything that
/// needs a JSON-ish value inside a tag goes through [`python_json`], which is
/// Python's `json.dumps` spacing, not Go's.
fn render_tools(tools: &[Tool]) -> String {
    let mut sb = String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>");

    for tool in tools {
        let f = &tool.function;
        sb.push_str("\n<function>\n<name>");
        sb.push_str(&f.name);
        sb.push_str("</name>");

        if !f.description.is_empty() {
            sb.push_str("\n<description>");
            sb.push_str(f.description.trim());
            sb.push_str("</description>");
        }

        sb.push_str("\n<parameters>");
        // `Properties != nil` upstream, so a set-but-EMPTY map takes this branch
        // too -- it just iterates zero times, so the output is the same either
        // way. Kept as `properties_iter()` because the distinction cannot show.
        for (name, prop) in f.parameters.properties_iter() {
            sb.push_str("\n<parameter>\n<name>");
            sb.push_str(name);
            sb.push_str("</name>");

            if !prop.prop_type.is_empty() {
                sb.push_str("\n<type>");
                sb.push_str(&format_property_type(&prop.prop_type));
                sb.push_str("</type>");
            }

            if !prop.description.is_empty() {
                sb.push_str("\n<description>");
                sb.push_str(prop.description.trim());
                sb.push_str("</description>");
            }

            if !prop.enum_values.is_empty() {
                sb.push_str("\n<enum>");
                sb.push_str(&python_json_list(&prop.enum_values));
                sb.push_str("</enum>");
            }

            render_tool_property_extra_keys(&mut sb, prop);
            sb.push_str("\n</parameter>");
        }

        render_tool_parameter_extra_keys(&mut sb, &f.parameters);
        if !f.parameters.required.is_empty() {
            sb.push_str("\n<required>");
            sb.push_str(&python_json_strings(&f.parameters.required));
            sb.push_str("</required>");
        }

        sb.push_str("\n</parameters>");
        sb.push_str("\n</function>");
    }

    sb.push_str("\n</tools>");
    sb.push_str(NEMOTRON_TOOL_POSTAMBLE);
    sb
}

/// A property's declared type, inside `<type>...</type>`.
///
/// **Upstream:** `formatPropertyType`. One type -> the bare name; several ->
/// **single-quoted** and comma-space separated, `['string', 'null']`. That is a
/// Python `repr` of a list, not JSON -- the reference chat template was written
/// in Jinja and printed the list directly, so the model was trained on the
/// Python spelling. Emitting `["string","null"]` here is the kind of "fix" that
/// looks tidier and makes the model worse.
fn format_property_type(t: &PropertyType) -> String {
    if t.0.len() == 1 {
        return t.0[0].clone();
    }
    let quoted: Vec<String> = t.0.iter().map(|v| format!("'{v}'")).collect();
    format!("[{}]", quoted.join(", "))
}

/// The property keys that do not get their own tag above.
///
/// **Upstream:** `renderToolPropertyExtraKeys`, and the order is upstream's:
/// `anyOf`, `items`, `properties`, `required`.
///
/// **The `properties` check is `!= nil`, not "non-empty".** Upstream's field is
/// a `*ToolPropertiesMap`, so a caller who sets an *empty* map still gets
/// `<properties>{}</properties>`, while a caller who never set it gets no tag at
/// all. That distinction is representable since `bd-iut`, so we honour it:
/// `is_some()`, never `has_properties()`.
fn render_tool_property_extra_keys(sb: &mut String, prop: &ToolProperty) {
    if !prop.any_of.is_empty() {
        sb.push_str("\n<anyOf>");
        sb.push_str(&python_json_properties(&prop.any_of));
        sb.push_str("</anyOf>");
    }
    if let Some(items) = &prop.items {
        sb.push_str("\n<items>");
        sb.push_str(&python_json(items));
        sb.push_str("</items>");
    }
    if let Some(props) = &prop.properties {
        sb.push_str("\n<properties>");
        sb.push_str(&python_json_property_map(props));
        sb.push_str("</properties>");
    }
    if !prop.required.is_empty() {
        sb.push_str("\n<required>");
        sb.push_str(&python_json_strings(&prop.required));
        sb.push_str("</required>");
    }
}

/// **Upstream:** `renderToolParameterExtraKeys` -- `$defs` then `items`, both
/// emitted only when present. Note the tag really is `<$defs>`, dollar included.
fn render_tool_parameter_extra_keys(sb: &mut String, params: &ToolFunctionParameters) {
    if let Some(defs) = &params.defs {
        sb.push_str("\n<$defs>");
        sb.push_str(&python_json(defs));
        sb.push_str("</$defs>");
    }
    if let Some(items) = &params.items {
        sb.push_str("\n<items>");
        sb.push_str(&python_json(items));
        sb.push_str("</items>");
    }
}

/// Assistant content with its reasoning block attached.
///
/// **Upstream:** `(*Nemotron3NanoRenderer).buildContent`. Three cases, and the
/// third is the one people miss:
///
/// 1. `thinking` is set -> wrap it in a real `<think>\n...\n</think>\n` block.
/// 2. Otherwise, if the content mentions **neither** `<think>` nor `</think>` ->
///    prepend the empty [`THINK_EMPTY`] marker. Every assistant turn must open
///    with *some* reasoning block; a turn with none is off-distribution.
/// 3. Otherwise the content already carries its own tags -> leave it exactly as
///    it is, half-open tags and all.
fn build_content(message: &Message) -> String {
    let content = &message.content;
    if !message.thinking.is_empty() {
        return format!(
            "{THINK_OPEN}\n{}\n{THINK_CLOSE}\n{content}",
            message.thinking
        );
    }
    if !content.contains(THINK_OPEN) && !content.contains(THINK_CLOSE) {
        return format!("{THINK_EMPTY}{content}");
    }
    content.clone()
}

/// **Upstream:** `formatAssistantContent` -- the no-tool-calls branch.
///
/// Untruncated it is just a trim. Truncated (an older turn) the reasoning is
/// **replaced**, not removed: everything up to and including the last
/// `</think>` becomes [`THINK_EMPTY`]. Only fires when BOTH tags are present,
/// so a dangling `<think>` survives here (unlike in
/// [`format_tool_call_content`], which handles it).
fn format_assistant_content(content: &str, truncate: bool) -> String {
    if !truncate {
        return go_trim_space(content).to_string();
    }

    let mut c = content.to_string();
    if c.contains(THINK_OPEN) && c.contains(THINK_CLOSE) {
        let tail = c.rsplit(THINK_CLOSE).next().unwrap_or("").to_string();
        c = format!("{THINK_EMPTY}{tail}");
    }
    go_trim_space(&c).to_string()
}

/// **Upstream:** `formatToolCallContent` -- the with-tool-calls branch.
///
/// Differences from [`format_assistant_content`], all deliberate:
///
/// * Whitespace-only content collapses to a bare [`THINK_EMPTY`] with **no**
///   trailing newline (the `<tool_call>` follows straight after).
/// * Every other case ends with **one trailing `\n`**, because a `<tool_call>`
///   must start on its own line.
/// * Truncation handles a **dangling `<think>`** with no close: everything from
///   that tag on is dropped. `format_assistant_content` leaves that case alone.
fn format_tool_call_content(content: &str, truncate: bool) -> String {
    if go_trim_space(content).is_empty() {
        return THINK_EMPTY.to_string();
    }

    if !truncate {
        return format!("{}\n", go_trim_space(content));
    }

    let mut c = content.to_string();
    if c.contains(THINK_CLOSE) {
        c = c.rsplit(THINK_CLOSE).next().unwrap_or("").to_string();
    } else if c.contains(THINK_OPEN) {
        c = c.split(THINK_OPEN).next().unwrap_or("").to_string();
    }
    c = format!("{THINK_EMPTY}{}", go_trim_space(&c));

    format!("{}\n", go_trim_space(&c))
}

/// **Upstream:** `writeToolCalls`. Arguments keep the model's own insertion
/// order, and each value goes on **its own line** between the `<parameter=..>`
/// tags -- a multi-line value is legal and the postamble above advertises it.
fn write_tool_calls(sb: &mut String, tool_calls: &[ToolCall]) {
    for tc in tool_calls {
        sb.push_str("<tool_call>\n<function=");
        sb.push_str(&tc.function.name);
        sb.push_str(">\n");
        for (name, value) in tc.function.arguments.0.iter() {
            sb.push_str("<parameter=");
            sb.push_str(name);
            sb.push_str(">\n");
            sb.push_str(&format_arg_value(value));
            sb.push_str("\n</parameter>\n");
        }
        sb.push_str("</function>\n</tool_call>\n");
    }
}

/// **Upstream:** `formatArgValue`. Objects and arrays keep JSON syntax (Python
/// spacing); **everything else goes through Go's `%v`**, so a string argument
/// loses its quotes -- `Paris`, not `"Paris"`.
fn format_arg_value(value: &Value) -> String {
    match value {
        Value::Object(_) | Value::Array(_) => python_json(value),
        Value::String(s) => s.clone(),
        Value::Null => "<nil>".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
    }
}

/// **Upstream:** `renderMessageContent` -- content plus `[img-N]` markers.
fn render_message_content(message: &Message, image_offset: usize) -> String {
    if message.images.is_empty() {
        return message.content.clone();
    }
    render_content_with_image_tags(&message.content, message.images.len(), image_offset).0
}

/// **Upstream:** `(*Nemotron3NanoRenderer).resolveThinking`.
///
/// Absent `think` counts as **on** (`thinkValue == nil || thinkValue.Bool()`),
/// then every `user`/`system` message may flip it with an inline toggle.
/// **Last one wins** -- the loop keeps overwriting, it does not break.
///
/// The `</think>` is stripped before looking for `/think` so that a literal
/// closing tag in the text cannot be mistaken for the toggle. `/no_think` is
/// checked in the `else` branch, so a message containing both turns thinking
/// **on**.
fn resolve_thinking(messages: &[Message], think: Option<&ThinkValue>) -> bool {
    let mut enable = think.is_none_or(ThinkValue::enabled);
    for message in messages {
        if message.role != "user" && message.role != "system" {
            continue;
        }
        let content = &message.content;
        if content.replace(THINK_CLOSE, "").contains("/think") {
            enable = true;
        } else if content.contains("/no_think") {
            enable = false;
        }
    }
    enable
}

/// **Upstream:** `sanitizeSystemMessage`.
///
/// The toggles are consumed by [`resolve_thinking`] and must not reach the
/// model, so they are deleted from the system message -- leaving the surrounding
/// spaces behind, which is why the fixture expects `"A  B  C"` with double
/// spaces. Do not "tidy" that.
///
/// The placeholder dance protects a **literal `</think>`**: `/think` is stripped
/// blindly, so without parking `</think>` first, a system message mentioning it
/// would come back as `<>`. `<_end_think>` is upstream's placeholder; a system
/// message that genuinely contains that string would be mangled, which is a
/// (theoretical) upstream bug we reproduce rather than silently fix.
///
/// Note the asymmetry: **only the leading system message is scrubbed.** A
/// `/no_think` in a user message stays visible to the model, and upstream's
/// `user no think toggle` fixture asserts exactly that.
fn sanitize_system_message(content: &str) -> String {
    const PLACEHOLDER: &str = "<_end_think>";
    content
        .replace(THINK_CLOSE, PLACEHOLDER)
        .replace("/think", "")
        .replace("/no_think", "")
        .replace(PLACEHOLDER, THINK_CLOSE)
}

/// Go's `strings.TrimSpace`, which trims **Unicode** whitespace.
///
/// Rust's `str::trim` uses the `White_Space` property and Go's `unicode.IsSpace`
/// is the same set, so this is a straight alias -- named so the ports read like
/// the Go they came from.
fn go_trim_space(s: &str) -> &str {
    s.trim()
}

// ---------------------------------------------------------------------------
// Python-flavoured JSON
// ---------------------------------------------------------------------------
//
// **Upstream:** `(*Nemotron3NanoRenderer).pythonJSON`, one giant type switch.
//
// This is NOT [`super::json`]'s Go emitter and must not be replaced by it. Two
// differences, both visible in upstream's fixtures:
//
// * **Separators are Python's** -- `", "` and `": "`, not Go's bare `,` / `:`.
// * **No HTML escaping.** Go's `encoding/json` turns `<` into `<`;
//   `strconv.Quote` (which upstream uses for every string here) does not. A tool
//   description saying `values <= 10` therefore appears literally.
//
// Both exist because the reference chat template was Jinja and printed Python
// values. Feed the model Go-shaped JSON here and the schema block stops matching
// what it was fine-tuned on.

/// Quote a string the way Go's `strconv.Quote` does.
///
/// Same as JSON for the cases a tool schema actually contains, and deliberately
/// **without** the `<`/`>`/`&` escaping that `encoding/json` applies. Control
/// characters use Go's short forms where they exist (`\n`, `\t`, ...) and
/// `\u00xx` otherwise -- Go would use `\x00`-style for some, but no fixture
/// reaches that and JSON has no `\x`, so `\u` is the honest spelling here. This
/// is a stated divergence, not an oversight: a tool description containing a raw
/// control character is already broken.
fn python_quote(s: &str, out: &mut String) {
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

/// One arbitrary JSON value, Python-spaced. **Object keys come out sorted** --
/// upstream's `map[string]any` case does `sort.Strings(keys)` explicitly.
fn python_json(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            let mut out = String::new();
            python_quote(s, &mut out);
            out
        }
        Value::Array(a) => python_json_list(a),
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let parts: Vec<String> = keys
                .iter()
                .map(|k| {
                    let mut out = String::new();
                    python_quote(k, &mut out);
                    out.push_str(": ");
                    out.push_str(&python_json(&m[*k]));
                    out
                })
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// `pythonJSON([]any)`.
fn python_json_list(values: &[Value]) -> String {
    let parts: Vec<String> = values.iter().map(python_json).collect();
    format!("[{}]", parts.join(", "))
}

/// `pythonJSON([]string)`.
fn python_json_strings(values: &[String]) -> String {
    let parts: Vec<String> = values
        .iter()
        .map(|s| {
            let mut out = String::new();
            python_quote(s, &mut out);
            out
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

/// `pythonJSON([]api.ToolProperty)` -- used for `anyOf`.
fn python_json_properties(props: &[ToolProperty]) -> String {
    let parts: Vec<String> = props.iter().map(python_json_property).collect();
    format!("[{}]", parts.join(", "))
}

/// `pythonJSON(*api.ToolPropertiesMap)` -- **insertion order kept**, unlike the
/// sorted plain-map case. Upstream ranges `value.All()`, which is the ordered
/// map's iterator.
fn python_json_property_map(props: &indexmap::IndexMap<String, ToolProperty>) -> String {
    let parts: Vec<String> = props
        .iter()
        .map(|(k, v)| {
            let mut out = String::new();
            python_quote(k, &mut out);
            out.push_str(": ");
            out.push_str(&python_json_property(v));
            out
        })
        .collect();
    format!("{{{}}}", parts.join(", "))
}

/// `pythonJSON(api.ToolProperty)` -- Go **declaration** order: `anyOf`, `type`,
/// `items`, `description`, `enum`, `properties`, `required`. Every field is
/// skipped when empty, so an empty property renders `{}`.
///
/// `type` collapses to a bare string when there is exactly one -- same rule as
/// `api.PropertyType.MarshalJSON`, spelled out again here because upstream
/// spells it out again here.
fn python_json_property(p: &ToolProperty) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(7);
    if !p.any_of.is_empty() {
        parts.push(format!("\"anyOf\": {}", python_json_properties(&p.any_of)));
    }
    if !p.prop_type.is_empty() {
        let rendered = if p.prop_type.0.len() == 1 {
            let mut out = String::new();
            python_quote(&p.prop_type.0[0], &mut out);
            out
        } else {
            python_json_strings(&p.prop_type.0)
        };
        parts.push(format!("\"type\": {rendered}"));
    }
    if let Some(items) = &p.items {
        parts.push(format!("\"items\": {}", python_json(items)));
    }
    if !p.description.is_empty() {
        let mut out = String::new();
        python_quote(&p.description, &mut out);
        parts.push(format!("\"description\": {out}"));
    }
    if !p.enum_values.is_empty() {
        parts.push(format!("\"enum\": {}", python_json_list(&p.enum_values)));
    }
    // `!= nil`, so a set-but-empty nested map renders `{}` -- see
    // [`render_tool_property_extra_keys`].
    if let Some(nested) = &p.properties {
        parts.push(format!(
            "\"properties\": {}",
            python_json_property_map(nested)
        ));
    }
    if !p.required.is_empty() {
        parts.push(format!(
            "\"required\": {}",
            python_json_strings(&p.required)
        ));
    }
    format!("{{{}}}", parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ToolCallFunction, ToolFunction};
    use indexmap::IndexMap;
    use serde_json::json;

    fn render(msgs: &[Message], tools: &[Tool], think: Option<ThinkValue>) -> String {
        Nemotron3NanoRenderer
            .render(msgs, tools, think.as_ref())
            .expect("nemotron render never fails")
    }

    fn assistant(content: &str) -> Message {
        Message::new("assistant", content)
    }

    fn tool_call(name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: String::new(),
            function: ToolCallFunction {
                index: 0,
                name: name.to_string(),
                arguments: serde_json::from_str(args).expect("test args must be valid JSON"),
            },
        }
    }

    /// Upstream's `nemotron3NanoReferenceTools()` -- one tool exercising every
    /// branch of the XML schema writer at once.
    fn reference_tools() -> Vec<Tool> {
        let mut props: IndexMap<String, ToolProperty> = IndexMap::new();
        props.insert(
            "query".into(),
            serde_json::from_value(json!({
                "type": "string", "description": "Search query", "enum": ["api", "cli"]
            }))
            .unwrap(),
        );
        props.insert(
            "mode".into(),
            serde_json::from_value(json!({
                "type": ["string", "null"], "description": "Mode",
                "anyOf": [{"type": "string"}, {"type": "number"}]
            }))
            .unwrap(),
        );
        props.insert(
            "payload".into(),
            serde_json::from_value(json!({
                "type": "object", "description": "Payload",
                "properties": {"enabled": {"type": "boolean"}}, "required": ["enabled"]
            }))
            .unwrap(),
        );
        props.insert(
            "tags".into(),
            serde_json::from_value(json!({
                "type": "array", "description": "Tags", "items": {"type": "string"}
            }))
            .unwrap(),
        );

        vec![Tool {
            tool_type: "function".into(),
            items: None,
            function: ToolFunction {
                name: "search_docs".into(),
                description: "Search docs".into(),
                parameters: ToolFunctionParameters {
                    param_type: "object".into(),
                    defs: Some(json!({"shared": {"type": "string"}})),
                    items: None,
                    required: vec!["query".into()],
                    properties: Some(props),
                },
            },
        }]
    }

    /// The `<|im_start|>system ... <|im_end|>\n` block upstream calls
    /// `toolText`, copied byte-for-byte from
    /// `nemotron3nano_reference_test.go`.
    fn tool_text(system: &str) -> String {
        format!(
            "<|im_start|>system\n{system}# Tools\n\nYou have access to the following functions:\n\n\
<tools>\n<function>\n<name>search_docs</name>\n<description>Search docs</description>\n\
<parameters>\n<parameter>\n<name>query</name>\n<type>string</type>\n\
<description>Search query</description>\n<enum>[\"api\", \"cli\"]</enum>\n</parameter>\n\
<parameter>\n<name>mode</name>\n<type>['string', 'null']</type>\n<description>Mode</description>\n\
<anyOf>[{{\"type\": \"string\"}}, {{\"type\": \"number\"}}]</anyOf>\n</parameter>\n\
<parameter>\n<name>payload</name>\n<type>object</type>\n<description>Payload</description>\n\
<properties>{{\"enabled\": {{\"type\": \"boolean\"}}}}</properties>\n\
<required>[\"enabled\"]</required>\n</parameter>\n\
<parameter>\n<name>tags</name>\n<type>array</type>\n<description>Tags</description>\n\
<items>{{\"type\": \"string\"}}</items>\n</parameter>\n\
<$defs>{{\"shared\": {{\"type\": \"string\"}}}}</$defs>\n<required>[\"query\"]</required>\n\
</parameters>\n</function>\n</tools>{NEMOTRON_TOOL_POSTAMBLE}<|im_end|>\n"
        )
    }

    /// Upstream `TestNemotron3NanoRendererMatchesReference`, all 21 cases, in
    /// upstream's order. These fixtures ARE the specification -- upstream
    /// cross-checks them against the real Jinja template under
    /// `VERIFY_JINJA2=1`, so a mismatch here is a wrong prompt, not a style
    /// disagreement.
    #[test]
    fn nemotron_matches_every_upstream_reference_fixture() {
        let t = || Some(ThinkValue::Bool(true));
        let f = || Some(ThinkValue::Bool(false));

        // no system default thinking on
        assert_eq!(
            render(&[Message::new("user", "Hello")], &[], None),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nHello<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // no system explicit thinking off -- note NO trailing newline.
        assert_eq!(
            render(&[Message::new("user", "Hello")], &[], f()),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nHello<|im_end|>\n\n<|im_start|>assistant\n<think></think>"
        );

        // literal endthink does not enable thinking
        assert_eq!(
            render(&[Message::new("user", "literal </think> only")], &[], f()),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nliteral </think> only<|im_end|>\n\n<|im_start|>assistant\n<think></think>"
        );

        // user no think toggle -- and the toggle STAYS in the user text.
        assert_eq!(
            render(&[Message::new("user", "Hello /no_think")], &[], t()),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nHello /no_think<|im_end|>\n\n<|im_start|>assistant\n<think></think>"
        );

        // system think toggle overrides an explicit `false`
        assert_eq!(
            render(
                &[
                    Message::new("system", "Policy /think"),
                    Message::new("user", "Hello")
                ],
                &[],
                f()
            ),
            "\n\n\n<|im_start|>system\nPolicy <|im_end|>\n\n<|im_start|>user\nHello<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // later toggle wins
        assert_eq!(
            render(
                &[
                    Message::new("system", "Policy /no_think"),
                    Message::new("user", "Actually /think")
                ],
                &[],
                f()
            ),
            "\n\n\n<|im_start|>system\nPolicy <|im_end|>\n\n<|im_start|>user\nActually /think<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // system sanitizes toggles but preserves the closing tag -- and leaves
        // the double spaces where the toggles were.
        assert_eq!(
            render(
                &[
                    Message::new("system", "A /think B /no_think C </think>"),
                    Message::new("user", "Hello")
                ],
                &[],
                f()
            ),
            "\n\n\n<|im_start|>system\nA  B  C </think><|im_end|>\n\n<|im_start|>user\nHello<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // assistant plain content adds an empty think block
        assert_eq!(
            render(
                &[Message::new("user", "Hi"), assistant("Hello there")],
                &[],
                t()
            ),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>Hello there<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // assistant reasoning content
        let mut a = assistant("Answer");
        a.thinking = "Need to think".into();
        assert_eq!(
            render(&[Message::new("user", "Hi"), a], &[], t()),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nNeed to think\n</think>\nAnswer<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // assistant preserves existing think tags
        assert_eq!(
            render(
                &[
                    Message::new("user", "Hi"),
                    assistant("<think>kept</think>Answer")
                ],
                &[],
                t()
            ),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>kept</think>Answer<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // tools without a system message
        assert_eq!(
            render(
                &[Message::new("user", "Use a tool")],
                &reference_tools(),
                t()
            ),
            format!(
                "\n\n\n{}\n<|im_start|>user\nUse a tool<|im_end|>\n\n<|im_start|>assistant\n<think>\n",
                tool_text("")
            )
        );

        // system with tools -- the system text and `# Tools` are separated by a
        // blank line.
        assert_eq!(
            render(
                &[
                    Message::new("system", "Follow policy."),
                    Message::new("user", "Use a tool")
                ],
                &reference_tools(),
                t()
            ),
            format!(
                "\n\n\n{}\n<|im_start|>user\nUse a tool<|im_end|>\n\n<|im_start|>assistant\n<think>\n",
                tool_text("Follow policy.\n\n")
            )
        );

        // assistant tool call with content
        let mut a = assistant("Checking now.");
        a.tool_calls = vec![tool_call("get_weather", r#"{"city":"Paris"}"#)];
        assert_eq!(
            render(&[Message::new("user", "Weather?"), a], &[], t()),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n<think></think>Checking now.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // assistant tool call with structured arguments -- Python spacing, and
        // the nested object's keys sorted.
        let mut a = assistant("");
        a.tool_calls = vec![tool_call(
            "create",
            r#"{"payload":{"count":42,"nested":{"value":"ok"}},"tags":["a","b"]}"#,
        )];
        assert_eq!(
            render(&[Message::new("user", "Create data"), a], &[], t()),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nCreate data<|im_end|>\n<|im_start|>assistant\n<think></think>\n<tool_call>\n<function=create>\n<parameter=payload>\n{\"count\": 42, \"nested\": {\"value\": \"ok\"}}\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // assistant tool call truncated with reasoning
        let mut a = assistant("Checking now.");
        a.thinking = "Need weather".into();
        a.tool_calls = vec![tool_call("get_weather", r#"{"city":"Paris"}"#)];
        assert_eq!(
            render(
                &[
                    Message::new("user", "Weather?"),
                    a,
                    Message::new("user", "And tomorrow?")
                ],
                &[],
                t()
            ),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n<think></think>Checking now.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n<|im_start|>user\nAnd tomorrow?<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // assistant tool call truncated, dangling `<think>` only
        let mut a = assistant("<think>draft");
        a.tool_calls = vec![tool_call("get_weather", r#"{"city":"Paris"}"#)];
        assert_eq!(
            render(
                &[
                    Message::new("user", "Weather?"),
                    a,
                    Message::new("user", "And tomorrow?")
                ],
                &[],
                t()
            ),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n<think></think>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n<|im_start|>user\nAnd tomorrow?<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // assistant tool call, empty content
        let mut a = assistant("");
        a.tool_calls = vec![tool_call("get_weather", r#"{"city":"Paris"}"#)];
        assert_eq!(
            render(&[Message::new("user", "Weather?"), a], &[], t()),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n<think></think>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // assistant truncated with a think pair
        assert_eq!(
            render(
                &[
                    Message::new("user", "Hi"),
                    assistant("<think>hidden</think>Visible"),
                    Message::new("user", "Next")
                ],
                &[],
                t()
            ),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>Visible<|im_end|>\n<|im_start|>user\nNext<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // assistant truncated reasoning content -- the `\n` before "Visible"
        // survives, because it came from buildContent's own separator.
        let mut a = assistant("Visible");
        a.thinking = "hidden".into();
        assert_eq!(
            render(
                &[Message::new("user", "Hi"), a, Message::new("user", "Next")],
                &[],
                t()
            ),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>\nVisible<|im_end|>\n<|im_start|>user\nNext<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // assistant truncated plain content
        assert_eq!(
            render(
                &[
                    Message::new("user", "Hi"),
                    assistant("Visible"),
                    Message::new("user", "Next")
                ],
                &[],
                t()
            ),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>Visible<|im_end|>\n<|im_start|>user\nNext<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // assistant truncated empty content
        assert_eq!(
            render(
                &[
                    Message::new("user", "Hi"),
                    assistant(""),
                    Message::new("user", "Next")
                ],
                &[],
                t()
            ),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think><|im_end|>\n<|im_start|>user\nNext<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // consecutive tool messages share ONE user block
        let mut a = assistant("");
        a.tool_calls = vec![tool_call("step", r#"{"value":1}"#)];
        assert_eq!(
            render(
                &[
                    Message::new("user", "Do work"),
                    a,
                    Message::new("tool", "one"),
                    Message::new("tool", "two"),
                ],
                &[],
                t()
            ),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\nDo work<|im_end|>\n<|im_start|>assistant\n<think></think>\n<tool_call>\n<function=step>\n<parameter=value>\n1\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n<|im_start|>user\n<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\n</tool_response>\n<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // an unknown role passes straight through
        assert_eq!(
            render(
                &[Message::new("developer", "Custom role content")],
                &[],
                t()
            ),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>developer\nCustom role content<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );
    }

    /// Upstream `TestNemotron3NanoRenderer_Images` -- and note the global
    /// `[img-N]` switch is NOT consulted, so these hold regardless of it.
    #[test]
    fn images_are_numbered_across_the_whole_conversation() {
        let img = |content: &str, n: usize| Message {
            role: "user".into(),
            content: content.into(),
            images: (0..n).map(|i| format!("img{i}")).collect(),
            ..Default::default()
        };

        assert_eq!(
            render(&[img("Describe this image.", 1)], &[], None),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\n[img-0] Describe this image.<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // An explicit `[img]` placeholder is replaced in place -- no space added.
        assert_eq!(
            render(&[img("[img]Describe this image.", 1)], &[], None),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\n[img-0]Describe this image.<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );

        // The counter keeps running across turns.
        assert_eq!(
            render(
                &[
                    img("Describe the first image.", 1),
                    assistant("It shows something."),
                    img("Compare these.", 2),
                ],
                &[],
                None
            ),
            "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\n[img-0] Describe the first image.<|im_end|>\n<|im_start|>assistant\n<think></think>It shows something.<|im_end|>\n<|im_start|>user\n[img-1][img-2] Compare these.<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
        );
    }

    /// `bd-iut` follow-through. Upstream checks `prop.Properties != nil`, so a
    /// tool author who says "this nested object takes no fields" (an empty map)
    /// must still get a `<properties>{}</properties>` tag -- while one who never
    /// described the object at all gets no tag. Collapsing the two would tell
    /// the model the wrong thing about a real schema.
    #[test]
    fn an_empty_nested_property_map_still_emits_its_tag() {
        let with_empty = ToolProperty {
            prop_type: PropertyType(vec!["object".into()]),
            properties: Some(IndexMap::new()),
            ..Default::default()
        };
        let mut sb = String::new();
        render_tool_property_extra_keys(&mut sb, &with_empty);
        assert_eq!(sb, "\n<properties>{}</properties>");

        let absent = ToolProperty {
            prop_type: PropertyType(vec!["object".into()]),
            properties: None,
            ..Default::default()
        };
        let mut sb = String::new();
        render_tool_property_extra_keys(&mut sb, &absent);
        assert_eq!(sb, "", "an undescribed object must emit no tag at all");
    }

    /// The Python spelling, asserted on its own so a future "let's just use
    /// [`super::json`]" refactor fails loudly instead of quietly shifting every
    /// nemotron tool schema.
    #[test]
    fn python_json_uses_python_separators_and_does_not_html_escape() {
        assert_eq!(python_json(&json!({"b": 1, "a": 2})), r#"{"a": 2, "b": 1}"#);
        assert_eq!(python_json(&json!(["x", "y"])), r#"["x", "y"]"#);
        // Go's encoding/json would give `<`; strconv.Quote does not.
        assert_eq!(python_json(&json!("a < b & c")), r#""a < b & c""#);
    }

    /// A multi-type property is a **Python list repr**, single quotes and all.
    #[test]
    fn a_multi_type_property_renders_in_python_repr_style() {
        assert_eq!(
            format_property_type(&PropertyType(vec!["string".into(), "null".into()])),
            "['string', 'null']"
        );
        assert_eq!(
            format_property_type(&PropertyType(vec!["string".into()])),
            "string"
        );
    }

    /// A string argument goes in **unquoted**; a container keeps its JSON.
    #[test]
    fn tool_call_arguments_print_bare_strings_but_json_containers() {
        assert_eq!(format_arg_value(&json!("Paris")), "Paris");
        assert_eq!(format_arg_value(&json!(1)), "1");
        assert_eq!(format_arg_value(&json!(true)), "true");
        assert_eq!(format_arg_value(&json!({"a": 1})), r#"{"a": 1}"#);
        assert_eq!(format_arg_value(&json!([1, 2])), "[1, 2]");
    }

    /// `nemotron-3-nano` adds no BOS -- the tokenizer owns that.
    #[test]
    fn nemotron_declares_no_leading_bos() {
        assert_eq!(Nemotron3NanoRenderer.leading_bos(), "");
    }
}
