//! **Gemma 4** -- `<|turn>` markers and a bespoke, JSON-ish schema dialect.
//!
//! **Upstream:** `model/renderers/gemma4.go`. Registered as `gemma4`,
//! `gemma4-small` (identical) and `gemma4-large` (adds
//! `empty_block_on_nothink`).
//!
//! ## The framing
//!
//! Special tokens, all of them unusual:
//!
//! * **`<bos>`** -- and it is written **into the prompt**. Gemma 4's tokenizer
//!   config has `add_bos_token=false`, so nothing else will add it.
//!   [`Renderer::leading_bos`] also reports it; a caller that prepends it again
//!   double-BOSes the model.
//! * **`<|turn>role` ... `<turn|>`** as the turn wrapper, and the assistant role
//!   is spelled **`model`**, not `assistant`.
//! * **`<|think|>`** -- a marker in the *system* turn saying reasoning is on.
//! * **`<|channel>thought` ... `<channel|>`** around reasoning content.
//! * **`<|tool>declaration:...<tool|>`**, **`<|tool_call>call:...<tool_call|>`**,
//!   **`<|tool_response>response:...<tool_response|>`**.
//! * **`<|"|>`** as the *string delimiter* inside all of those -- [`G4Q`]. Not a
//!   quote character: a token. Values are **not** escaped inside it, which is
//!   upstream's behaviour (a value containing the delimiter would break the
//!   framing, and upstream does not defend against that either).
//!
//! ## The schema dialect is not JSON, and the details bite
//!
//! A tool declaration reads like
//! `declaration:bash{description:<|"|>Run<|"|>,parameters:{properties:{...},required:[...],type:<|"|>OBJECT<|"|>}}`.
//! Three rules that are easy to miss:
//!
//! 1. **Type names are UPPERCASED** -- `string` becomes `STRING`, `object`
//!    becomes `OBJECT`.
//! 2. **Property keys are sorted alphabetically**, not kept in the order the
//!    caller supplied. Upstream's `searchDeclRef` fixture proves it: inserted
//!    `query, limit, offset`, emitted `limit, offset, query`.
//! 3. **Within one property the field order is fixed**: `description`, then
//!    `enum` (STRING only) / `items` (ARRAY only), then `nullable`, then
//!    `properties` + `required` (OBJECT only), then `type` **last**.
//!
//! ## Top-level vs nested properties are genuinely different, and upstream says so
//!
//! A **top-level** property's multi-type union is stringified into
//! `type:<|"|>['STRING', 'NULL']<|"|>` -- one string containing a Python-looking
//! list. A **nested** one emits a real list, `type:[<|"|>A<|"|>,<|"|>B<|"|>]`.
//! Upstream's own comment calls this *"odd, but we match upstream here"*, and so
//! do we. See [`top_level_type_value`].
//!
//! ## Turn folding
//!
//! `tool` messages never open a turn of their own -- they are folded into the
//! assistant turn that called them. Adjacent assistant messages continue the
//! same `model` turn rather than opening a second one. That is what
//! `continue_same_model_turn` / `continues_into_next` compute, and it is the
//! fiddliest part of the whole file.

use std::collections::BTreeMap;

use serde_json::Value;

use super::image_tags::render_content_with_image_tags;
use super::{Message, RenderError, Renderer, ThinkValue, Tool};
use crate::api::{PropertyType, ToolCall, ToolProperty};

/// Gemma 4's string delimiter. **Upstream:** `g4Q` in `gemma4.go`.
pub(crate) const G4Q: &str = "<|\"|>";

/// **Upstream:** `Gemma4Renderer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Gemma4Renderer {
    /// Use `[img-N]` markers for images.
    pub use_img_tags: bool,
    /// `gemma4-large` only: when thinking is off, put an **empty**
    /// `<|channel>thought\n<channel|>` in the generation prompt so the model
    /// does not start one of its own.
    pub empty_block_on_nothink: bool,
}

/// Strip `<|channel>...<channel|>` blocks out of content, then trim.
///
/// **Upstream:** `stripThinking`, mirroring the HF template's `strip_thinking`
/// macro. Note the dangling-open-tag case: if a `<|channel>` has no closing
/// `<channel|>`, everything from it onward is **dropped**, not kept.
fn strip_thinking(text: &str) -> String {
    let mut result = String::new();
    let mut text = text;
    loop {
        let Some(start) = text.find("<|channel>") else {
            result.push_str(text);
            break;
        };
        result.push_str(&text[..start]);
        let Some(end) = text[start..].find("<channel|>") else {
            break;
        };
        text = &text[start + end + "<channel|>".len()..];
    }
    result.trim().to_string()
}

/// Uppercase a list of JSON-Schema type names. **Upstream:** `normalizeTypeNames`.
fn normalize_type_names(t: &PropertyType) -> Vec<String> {
    t.0.iter().map(|s| s.to_uppercase()).collect()
}

/// The **top-level** `type` value. **Upstream:** `upstreamTypedPropertyTypeValue`.
///
/// One type -> the name **unchanged** (the uppercasing happens later, in
/// `normalize_type_names`). More than one -> a single string that looks like a
/// Python list: `['STRING', 'NULL']`, note the space after the comma and the
/// single quotes. Upstream flags this as odd and keeps it; so do we, because the
/// model was trained on it.
fn top_level_type_value(types: &PropertyType) -> String {
    if types.0.len() == 1 {
        return types.0[0].clone();
    }
    let mut s = String::from("[");
    for (i, t) in types.0.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push('\'');
        s.push_str(&t.to_uppercase());
        s.push('\'');
    }
    s.push(']');
    s
}

/// Collapse `anyOf` into a plain type union, when every branch is a bare
/// type-only property. **Upstream:** `simpleAnyOfTypes`.
///
/// Gemma's declaration format has no `anyOf` construct, so a union that is
/// *only* about types gets lowered into a type list. A union carrying
/// descriptions, enums or nested schemas cannot be lowered and is dropped --
/// upstream's behaviour, and worth knowing before you send Gemma a rich `anyOf`.
fn simple_any_of_types(prop: &ToolProperty) -> Option<PropertyType> {
    if prop.any_of.is_empty() {
        return None;
    }
    let mut out: Vec<String> = Vec::new();
    for branch in &prop.any_of {
        let bare = branch.any_of.is_empty()
            && !branch.prop_type.0.is_empty()
            && branch.items.is_none()
            && branch.description.is_empty()
            && branch.enum_values.is_empty()
            && branch.properties.is_none()
            && branch.required.is_empty();
        if !bare {
            return None;
        }
        for t in &branch.prop_type.0 {
            if !out.contains(t) {
                out.push(t.clone());
            }
        }
    }
    (!out.is_empty()).then_some(PropertyType(out))
}

/// A schema value in Gemma's dialect: `null`, a delimited string, a bare
/// bool/number, or a nested map/array.
///
/// **Upstream:** `formatSchemaValue` (used for the "everything else" keys inside
/// an `items` spec) -- note a **map** written this way delimits its *keys* too,
/// unlike the property writer, which writes keys bare.
fn format_schema_value(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::String(s) => format!("{G4Q}{s}{G4Q}"),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => format_number(n),
        Value::Object(m) => {
            let mut s = String::from("{");
            for (i, (k, val)) in m.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                // Delimited keys here, bare keys in `write_schema_properties`.
                s.push_str(&format!("{G4Q}{k}{G4Q}:{}", format_schema_value(val)));
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
                s.push_str(&format_schema_value(item));
            }
            s.push(']');
            s
        }
    }
}

/// Go's `float64` printing for schema/argument numbers: a whole number loses its
/// `.0`. **Upstream:** the `case float64` arm of `formatArgValue`.
fn format_number(n: &serde_json::Number) -> String {
    if let Some(f) = n.as_f64()
        && f == (f as i64) as f64
    {
        return format!("{}", f as i64);
    }
    n.to_string()
}

/// One tool-call argument value. **Upstream:** `formatArgValue`.
///
/// Same shape as [`format_schema_value`] except a **map writes bare keys**.
fn format_arg_value(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::String(s) => format!("{G4Q}{s}{G4Q}"),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => format_number(n),
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
    }
}

/// How a property's `type` is spelled: as one delimited string, or as a list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TypeStyle {
    /// Top level: a multi-type union collapses to ONE stringified list.
    TopLevel,
    /// Nested: a multi-type union stays a real list.
    Nested,
}

/// Write one property's body (everything between its `{` and `}`).
///
/// **Upstream:** the loop body of `writeSchemaProperties`. Field order is fixed
/// and `type` comes **last** -- see the module docs.
fn write_property_body(sb: &mut String, prop: &ToolProperty, style: TypeStyle) {
    let mut add_comma = false;
    let comma = |sb: &mut String, add_comma: &mut bool| {
        if *add_comma {
            sb.push(',');
        }
        *add_comma = true;
    };

    if !prop.description.is_empty() {
        sb.push_str(&format!("description:{G4Q}{}{G4Q}", prop.description));
        add_comma = true;
    }

    // The effective type list, after lowering a simple `anyOf`.
    let types: PropertyType = if !prop.prop_type.0.is_empty() {
        prop.prop_type.clone()
    } else {
        simple_any_of_types(prop).unwrap_or_default()
    };
    let type_names = match style {
        // The top-level path stringifies first, THEN uppercases the whole
        // string -- which is why `['STRING', 'NULL']` survives as one name.
        TypeStyle::TopLevel if !types.0.is_empty() => {
            vec![top_level_type_value(&types).to_uppercase()]
        }
        TypeStyle::TopLevel => Vec::new(),
        TypeStyle::Nested => normalize_type_names(&types),
    };
    let first_type = type_names.first().map(String::as_str).unwrap_or("");

    match first_type {
        "STRING" if !prop.enum_values.is_empty() => {
            comma(sb, &mut add_comma);
            sb.push_str("enum:[");
            for (i, v) in prop.enum_values.iter().enumerate() {
                if i > 0 {
                    sb.push(',');
                }
                // Go's `%v`: a string loses its JSON quotes here.
                let text = match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                sb.push_str(&format!("{G4Q}{text}{G4Q}"));
            }
            sb.push(']');
        }
        "ARRAY" => {
            if let Some(Value::Object(items)) = &prop.items
                && !items.is_empty()
            {
                comma(sb, &mut add_comma);
                sb.push_str("items:{");
                write_schema_items_spec(sb, items);
                sb.push('}');
            }
        }
        _ => {}
    }

    if first_type == "OBJECT" {
        comma(sb, &mut add_comma);
        sb.push_str("properties:{");
        if let Some(nested) = &prop.properties {
            write_schema_properties(sb, nested.iter(), TypeStyle::Nested);
        }
        sb.push('}');

        if !prop.required.is_empty() {
            comma(sb, &mut add_comma);
            sb.push_str("required:[");
            for (i, r) in prop.required.iter().enumerate() {
                if i > 0 {
                    sb.push(',');
                }
                sb.push_str(&format!("{G4Q}{r}{G4Q}"));
            }
            sb.push(']');
        }
    }

    if !type_names.is_empty() {
        if add_comma {
            sb.push(',');
        }
        if type_names.len() == 1 {
            sb.push_str(&format!("type:{G4Q}{}{G4Q}", type_names[0]));
        } else {
            sb.push_str("type:[");
            for (i, name) in type_names.iter().enumerate() {
                if i > 0 {
                    sb.push(',');
                }
                sb.push_str(&format!("{G4Q}{name}{G4Q}"));
            }
            sb.push(']');
        }
    }
}

/// **Upstream:** `writeSchemaProperties`. Keys sorted, bare (not delimited).
fn write_schema_properties<'a, I>(sb: &mut String, props: I, style: TypeStyle)
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
        sb.push_str(name);
        sb.push_str(":{");
        write_property_body(sb, prop, style);
        sb.push('}');
    }
}

/// The `items:{...}` spec of an ARRAY property.
///
/// **Upstream:** `writeSchemaItemsSpec`. Keys sorted; `properties`, `required`
/// and `type` are special-cased and everything else falls through to
/// [`format_schema_value`]. A `null` value is skipped entirely.
fn write_schema_items_spec(sb: &mut String, items: &serde_json::Map<String, Value>) {
    let mut first = true;
    for (key, value) in items {
        if value.is_null() {
            continue;
        }
        if !first {
            sb.push(',');
        }
        first = false;
        match key.as_str() {
            "properties" => {
                sb.push_str("properties:{");
                if let Value::Object(nested) = value {
                    let parsed: BTreeMap<String, ToolProperty> = nested
                        .iter()
                        .filter_map(|(k, v)| {
                            serde_json::from_value::<ToolProperty>(v.clone())
                                .ok()
                                .map(|p| (k.clone(), p))
                        })
                        .collect();
                    write_schema_properties(sb, parsed.iter(), TypeStyle::Nested);
                }
                sb.push('}');
            }
            "required" => {
                sb.push_str("required:[");
                if let Value::Array(a) = value {
                    let mut n = 0;
                    for item in a {
                        if let Value::String(s) = item {
                            if n > 0 {
                                sb.push(',');
                            }
                            n += 1;
                            sb.push_str(&format!("{G4Q}{s}{G4Q}"));
                        }
                    }
                }
                sb.push(']');
            }
            "type" => {
                let names: Vec<String> = match value {
                    Value::String(s) => vec![s.to_uppercase()],
                    Value::Array(a) => a
                        .iter()
                        .filter_map(|v| v.as_str().map(str::to_uppercase))
                        .collect(),
                    _ => Vec::new(),
                };
                if names.len() == 1 {
                    sb.push_str(&format!("type:{G4Q}{}{G4Q}", names[0]));
                } else if names.len() > 1 {
                    sb.push_str("type:[");
                    for (i, n) in names.iter().enumerate() {
                        if i > 0 {
                            sb.push(',');
                        }
                        sb.push_str(&format!("{G4Q}{n}{G4Q}"));
                    }
                    sb.push(']');
                }
            }
            other => {
                sb.push_str(&format!("{other}:{}", format_schema_value(value)));
            }
        }
    }
}

impl Gemma4Renderer {
    /// **Upstream:** `renderToolDeclaration`.
    fn render_tool_declaration(&self, tool: &Tool) -> String {
        let fn_ = &tool.function;
        let mut sb = String::new();
        sb.push_str(&format!("<|tool>declaration:{}{{", fn_.name));
        sb.push_str(&format!("description:{G4Q}{}{G4Q}", fn_.description));

        // Upstream: `Properties != nil || Type != ""` -- the OUTER guard asks
        // "did anybody describe this tool at all?", so a **set-but-empty**
        // property map opens a `parameters:{}` block while an absent one does
        // not. `is_some()`, therefore, NOT `has_properties()`: the latter is
        // false for `Some(empty)` too and would swallow the distinction. (Go's
        // pointer field makes all three states real -- see [`super::json`].)
        //
        // The INNER guard is upstream's `Properties != nil && Len() > 0`, which
        // IS exactly `has_properties()` -- an empty map contributes no
        // `properties:{}` sub-block.
        if fn_.parameters.properties.is_some() || !fn_.parameters.param_type.is_empty() {
            sb.push_str(",parameters:{");
            let mut needs_comma = false;

            if fn_.parameters.has_properties() {
                sb.push_str("properties:{");
                write_schema_properties(
                    &mut sb,
                    fn_.parameters.properties_iter(),
                    TypeStyle::TopLevel,
                );
                sb.push('}');
                needs_comma = true;
            }

            if !fn_.parameters.required.is_empty() {
                if needs_comma {
                    sb.push(',');
                }
                sb.push_str("required:[");
                for (i, r) in fn_.parameters.required.iter().enumerate() {
                    if i > 0 {
                        sb.push(',');
                    }
                    sb.push_str(&format!("{G4Q}{r}{G4Q}"));
                }
                sb.push(']');
                needs_comma = true;
            }

            if !fn_.parameters.param_type.is_empty() {
                if needs_comma {
                    sb.push(',');
                }
                sb.push_str(&format!(
                    "type:{G4Q}{}{G4Q}",
                    fn_.parameters.param_type.to_uppercase()
                ));
            }

            sb.push('}');
        }

        sb.push_str("}<tool|>");
        sb
    }

    /// **Upstream:** `formatToolCall`. Argument keys are **sorted**.
    fn format_tool_call(&self, tc: &ToolCall) -> String {
        let sorted: BTreeMap<&String, &Value> = tc.function.arguments.0.iter().collect();
        let mut sb = format!("<|tool_call>call:{}{{", tc.function.name);
        for (i, (key, value)) in sorted.iter().enumerate() {
            if i > 0 {
                sb.push(',');
            }
            sb.push_str(&format!("{key}:{}", format_arg_value(value)));
        }
        sb.push_str("}<tool_call|>");
        sb
    }

    /// **Upstream:** `formatToolResponseBlock`.
    fn format_tool_response_block(&self, tool_name: &str, response: &str) -> String {
        format!(
            "<|tool_response>response:{tool_name}{{value:{}}}<tool_response|>",
            format_arg_value(&Value::String(response.to_string()))
        )
    }

    /// Which tool a `tool` message is answering. **Upstream:**
    /// `toolResponseName` -- falls back to the literal `"unknown"`, and a
    /// matching `tool_call_id` overrides whatever `tool_name` said.
    fn tool_response_name(&self, message: &Message, tool_calls: &[ToolCall]) -> String {
        let mut name = if message.tool_name.is_empty() {
            "unknown".to_string()
        } else {
            message.tool_name.clone()
        };
        if !message.tool_call_id.is_empty()
            && let Some(tc) = tool_calls.iter().find(|tc| tc.id == message.tool_call_id)
        {
            name = tc.function.name.clone();
        }
        name
    }

    /// **Upstream:** `renderContent`. `trim` mirrors the Jinja `| trim` that is
    /// applied to non-model content.
    fn render_content(&self, sb: &mut String, msg: &Message, image_offset: &mut usize, trim: bool) {
        let mut content = if trim {
            msg.content.trim().to_string()
        } else {
            msg.content.clone()
        };
        if !msg.images.is_empty() && self.use_img_tags {
            let (c, o) = render_content_with_image_tags(&content, msg.images.len(), *image_offset);
            content = c;
            *image_offset = o;
        }
        sb.push_str(&content);
    }

    fn message_has_content(&self, message: &Message) -> bool {
        !message.content.trim().is_empty() || !message.images.is_empty()
    }

    fn next_non_tool_role(&self, messages: &[Message], idx: usize) -> String {
        messages[idx + 1..]
            .iter()
            .find(|m| m.role != "tool")
            .map(|m| m.role.clone())
            .unwrap_or_default()
    }
}

impl Renderer for Gemma4Renderer {
    fn leading_bos(&self) -> &'static str {
        "<bos>"
    }

    fn render(
        &self,
        messages: &[Message],
        tools: &[Tool],
        think: Option<&ThinkValue>,
    ) -> Result<String, RenderError> {
        let mut sb = String::new();
        let mut image_offset = 0usize;

        // Gemma 4 has `add_bos_token=false`, so the BOS must be in the text.
        sb.push_str("<bos>");

        let has_system_role = messages
            .first()
            .is_some_and(|m| m.role == "system" || m.role == "developer");
        let (system_message, loop_messages): (&str, &[Message]) = if has_system_role {
            (&messages[0].content, &messages[1..])
        } else {
            ("", messages)
        };

        let has_think = think.is_some_and(|t| t.enabled());
        if has_system_role || !tools.is_empty() || has_think {
            sb.push_str("<|turn>system\n");
            if has_think {
                sb.push_str("<|think|>\n");
            }
            if !system_message.is_empty() {
                sb.push_str(system_message.trim());
            }
            for tool in tools {
                sb.push_str(&self.render_tool_declaration(tool));
            }
            sb.push_str("<turn|>\n");
        }

        let last_user_idx: Option<usize> = loop_messages.iter().rposition(|m| m.role == "user");

        let mut prev_message_type = "";
        let mut prev_non_tool_role = String::new();

        for (i, message) in loop_messages.iter().enumerate() {
            if message.role == "tool" {
                // Folded into the assistant turn that called it.
                continue;
            }

            prev_message_type = "";
            let role = if message.role == "assistant" {
                "model"
            } else {
                message.role.as_str()
            };

            let continue_same_model_turn = role == "model" && prev_non_tool_role == "assistant";
            if !continue_same_model_turn {
                sb.push_str(&format!("<|turn>{role}\n"));
            }

            if message.role == "assistant"
                && !message.thinking.is_empty()
                && last_user_idx.is_none_or(|idx| i > idx)
            {
                sb.push_str("<|channel>thought\n");
                sb.push_str(&message.thinking);
                sb.push_str("\n<channel|>");
            }

            if !message.tool_calls.is_empty() {
                for tc in &message.tool_calls {
                    sb.push_str(&self.format_tool_call(tc));
                }
                prev_message_type = "tool_call";
            }

            let mut tool_responses_emitted = false;
            if !message.tool_calls.is_empty() {
                let mut k = i + 1;
                while k < loop_messages.len() && loop_messages[k].role == "tool" {
                    let mut response = String::new();
                    self.render_content(&mut response, &loop_messages[k], &mut image_offset, false);
                    let name = self.tool_response_name(&loop_messages[k], &message.tool_calls);
                    sb.push_str(&self.format_tool_response_block(&name, &response));
                    tool_responses_emitted = true;
                    prev_message_type = "tool_response";
                    k += 1;
                }
            }

            let message_had_content;
            if role == "model" {
                if !message.content.is_empty() || !message.images.is_empty() {
                    let stripped = Message {
                        content: strip_thinking(&message.content),
                        ..message.clone()
                    };
                    self.render_content(&mut sb, &stripped, &mut image_offset, false);
                    message_had_content = self.message_has_content(&stripped);
                } else {
                    message_had_content = false;
                }
            } else {
                self.render_content(&mut sb, message, &mut image_offset, true);
                let trimmed = Message {
                    content: message.content.trim().to_string(),
                    ..message.clone()
                };
                message_had_content = self.message_has_content(&trimmed);
            }

            let next_non_tool_role = self.next_non_tool_role(loop_messages, i);
            let continues_into_next = role == "model"
                && next_non_tool_role == "assistant"
                && (message.tool_calls.is_empty() || tool_responses_emitted);

            if prev_message_type == "tool_call" && !tool_responses_emitted {
                // A call with no answer yet: leave the response block open so
                // the model fills it in.
                sb.push_str("<|tool_response>");
            } else if !continues_into_next
                && !(tool_responses_emitted
                    && !message_had_content
                    && next_non_tool_role.is_empty())
            {
                sb.push_str("<turn|>\n");
            }

            prev_non_tool_role = message.role.clone();
        }

        // Generation prompt.
        if prev_message_type != "tool_response" && prev_message_type != "tool_call" {
            sb.push_str("<|turn>model\n");
            if self.empty_block_on_nothink && !has_think {
                sb.push_str("<|channel>thought\n<channel|>");
            }
        } else if prev_message_type == "tool_response" && has_think {
            sb.push_str("<|channel>thought\n");
        }

        Ok(sb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ToolCallFunction;
    use serde_json::json;

    fn r() -> Gemma4Renderer {
        Gemma4Renderer::default()
    }

    fn tool(raw: serde_json::Value) -> Tool {
        serde_json::from_value(raw).expect("valid fixture tool")
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

    /// Upstream's `bashRefTool` + `bashDeclRef`.
    fn bash_tool() -> Tool {
        tool(json!({
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {"type": "string", "description": "The command"}
                    }
                }
            }
        }))
    }
    const BASH_DECL: &str = "<|tool>declaration:bash{description:<|\"|>Run a command<|\"|>,parameters:{properties:{command:{description:<|\"|>The command<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>command<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>";

    /// Upstream's `TestGemma4RendererMatchesReference` header-block cases.
    #[test]
    fn the_system_turn_appears_only_when_something_needs_it() {
        assert_eq!(
            r().render(&[Message::new("user", "Hello")], &[], None)
                .unwrap(),
            "<bos><|turn>user\nHello<turn|>\n<|turn>model\n"
        );

        assert_eq!(
            r().render(
                &[
                    Message::new("system", "You are helpful."),
                    Message::new("user", "Hi"),
                ],
                &[],
                None
            )
            .unwrap(),
            "<bos><|turn>system\nYou are helpful.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
        );

        // `developer` is an alias for `system` -- and still renders as `system`.
        assert_eq!(
            r().render(
                &[
                    Message::new("developer", "You are helpful."),
                    Message::new("user", "Hi"),
                ],
                &[],
                None
            )
            .unwrap(),
            "<bos><|turn>system\nYou are helpful.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
        );

        // Tools alone are enough to open a system turn.
        assert_eq!(
            r().render(&[Message::new("user", "Hi")], &[bash_tool()], None)
                .unwrap(),
            format!(
                "<bos><|turn>system\n{BASH_DECL}<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
            )
        );

        // So is thinking, on its own.
        assert_eq!(
            r().render(
                &[Message::new("user", "Hi")],
                &[],
                Some(&ThinkValue::Bool(true))
            )
            .unwrap(),
            "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
        );

        // ...and an explicit `false` opens nothing at all.
        assert_eq!(
            r().render(
                &[Message::new("user", "Hi")],
                &[],
                Some(&ThinkValue::Bool(false))
            )
            .unwrap(),
            "<bos><|turn>user\nHi<turn|>\n<|turn>model\n"
        );

        // Everything at once, and the system text is trimmed.
        assert_eq!(
            r().render(
                &[
                    Message::new(
                        "developer",
                        "  Prefer terse answers.\nUse tools when needed.  "
                    ),
                    Message::new("user", "Hi"),
                ],
                &[bash_tool()],
                Some(&ThinkValue::Bool(true))
            )
            .unwrap(),
            format!(
                "<bos><|turn>system\n<|think|>\nPrefer terse answers.\nUse tools when needed.{BASH_DECL}<turn|>\n\
                 <|turn>user\nHi<turn|>\n<|turn>model\n"
            )
        );
    }

    /// Upstream's declaration reference strings, which between them pin the
    /// sorting, the uppercasing and the fixed field order.
    #[test]
    fn tool_declarations_match_the_upstream_reference_strings() {
        let decl = |t: &Tool| r().render_tool_declaration(t);

        // Nested OBJECT property. Note `config` sorts before `name`.
        assert_eq!(
            decl(&tool(json!({
                "type": "function",
                "function": {
                    "name": "create", "description": "Create item",
                    "parameters": {"type": "object", "properties": {
                        "name": {"type": "string", "description": "Name"},
                        "config": {"type": "object", "description": "Config", "properties": {
                            "enabled": {"type": "boolean", "description": "On/off"}
                        }}
                    }}
                }
            }))),
            "<|tool>declaration:create{description:<|\"|>Create item<|\"|>,parameters:{properties:{config:{description:<|\"|>Config<|\"|>,properties:{enabled:{description:<|\"|>On/off<|\"|>,type:<|\"|>BOOLEAN<|\"|>}},type:<|\"|>OBJECT<|\"|>},name:{description:<|\"|>Name<|\"|>,type:<|\"|>STRING<|\"|>}},type:<|\"|>OBJECT<|\"|>}}<tool|>"
        );

        // ARRAY with an items spec.
        assert_eq!(
            decl(&tool(json!({
                "type": "function",
                "function": {
                    "name": "batch", "description": "Run batch",
                    "parameters": {"type": "object", "properties": {
                        "commands": {"type": "array", "description": "Commands", "items": {"type": "string"}}
                    }}
                }
            }))),
            "<|tool>declaration:batch{description:<|\"|>Run batch<|\"|>,parameters:{properties:{commands:{description:<|\"|>Commands<|\"|>,items:{type:<|\"|>STRING<|\"|>},type:<|\"|>ARRAY<|\"|>}},type:<|\"|>OBJECT<|\"|>}}<tool|>"
        );

        // ARRAY with NO items spec -- the `items:` clause disappears entirely.
        assert_eq!(
            decl(&tool(json!({
                "type": "function",
                "function": {
                    "name": "tag", "description": "Tag items",
                    "parameters": {"type": "object", "properties": {
                        "tags": {"type": "array", "description": "Tags"}
                    }}
                }
            }))),
            "<|tool>declaration:tag{description:<|\"|>Tag items<|\"|>,parameters:{properties:{tags:{description:<|\"|>Tags<|\"|>,type:<|\"|>ARRAY<|\"|>}},type:<|\"|>OBJECT<|\"|>}}<tool|>"
        );

        // STRING with an enum, and the same without a description.
        assert_eq!(
            decl(&tool(json!({
                "type": "function",
                "function": {
                    "name": "set_level", "description": "Set level",
                    "parameters": {"type": "object", "properties": {
                        "level": {"type": "string", "enum": ["low", "high"]}
                    }}
                }
            }))),
            "<|tool>declaration:set_level{description:<|\"|>Set level<|\"|>,parameters:{properties:{level:{enum:[<|\"|>low<|\"|>,<|\"|>high<|\"|>],type:<|\"|>STRING<|\"|>}},type:<|\"|>OBJECT<|\"|>}}<tool|>"
        );

        // Properties come out ALPHABETICAL: inserted query/limit/offset.
        assert_eq!(
            decl(&tool(json!({
                "type": "function",
                "function": {
                    "name": "search", "description": "Search",
                    "parameters": {"type": "object", "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "number"},
                        "offset": {"type": "number", "description": "Start offset"}
                    }}
                }
            }))),
            "<|tool>declaration:search{description:<|\"|>Search<|\"|>,parameters:{properties:{limit:{type:<|\"|>NUMBER<|\"|>},offset:{description:<|\"|>Start offset<|\"|>,type:<|\"|>NUMBER<|\"|>},query:{description:<|\"|>Search query<|\"|>,type:<|\"|>STRING<|\"|>}},type:<|\"|>OBJECT<|\"|>}}<tool|>"
        );

        // A nested OBJECT that carries its own `required`.
        assert_eq!(
            decl(&tool(json!({
                "type": "function",
                "function": {
                    "name": "create_user", "description": "Create user",
                    "parameters": {"type": "object", "properties": {
                        "profile": {
                            "type": "object", "description": "Profile",
                            "properties": {
                                "name": {"type": "string", "description": "Name"},
                                "age": {"type": "number", "description": "Age"}
                            },
                            "required": ["name"]
                        }
                    }}
                }
            }))),
            "<|tool>declaration:create_user{description:<|\"|>Create user<|\"|>,parameters:{properties:{profile:{description:<|\"|>Profile<|\"|>,properties:{age:{description:<|\"|>Age<|\"|>,type:<|\"|>NUMBER<|\"|>},name:{description:<|\"|>Name<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>name<|\"|>],type:<|\"|>OBJECT<|\"|>}},type:<|\"|>OBJECT<|\"|>}}<tool|>"
        );

        // No properties at all -> only the `type` survives.
        assert_eq!(
            decl(&tool(json!({
                "type": "function",
                "function": {
                    "name": "raw", "description": "Raw input",
                    "parameters": {"type": "object"}
                }
            }))),
            "<|tool>declaration:raw{description:<|\"|>Raw input<|\"|>,parameters:{type:<|\"|>OBJECT<|\"|>}}<tool|>"
        );
    }

    /// Upstream `typed_property_union_type` -- the stringified union. Upstream
    /// itself calls this odd; it is copied, not corrected.
    #[test]
    fn a_top_level_union_becomes_one_stringified_list() {
        let got = r()
            .render(
                &[Message::new("user", "Hi")],
                &[tool(json!({
                    "type": "function",
                    "function": {
                        "name": "maybe_name", "description": "Test nullable union",
                        "parameters": {"type": "object", "properties": {
                            "name": {"type": ["string", "null"], "description": "Name"}
                        }}
                    }
                }))],
                None,
            )
            .unwrap();
        assert!(
            got.contains("name:{description:<|\"|>Name<|\"|>,type:<|\"|>['STRING', 'NULL']<|\"|>}"),
            "{got}"
        );
    }

    /// Upstream `empty_tool_args` and `thinking_with_tool_calls`: a tool call
    /// with its answer folded into the same `model` turn, and the special
    /// "no `<turn|>` at the very end" case.
    #[test]
    fn tool_calls_and_their_responses_share_the_model_turn() {
        assert_eq!(
            r().render(
                &[
                    Message::new("user", "Go"),
                    Message {
                        role: "assistant".into(),
                        tool_calls: vec![call("bash", "{}")],
                        ..Default::default()
                    },
                    Message {
                        role: "tool".into(),
                        tool_name: "bash".into(),
                        content: "ok".into(),
                        ..Default::default()
                    },
                ],
                &[],
                None
            )
            .unwrap(),
            "<bos><|turn>user\nGo<turn|>\n\
             <|turn>model\n<|tool_call>call:bash{}<tool_call|>\
             <|tool_response>response:bash{value:<|\"|>ok<|\"|>}<tool_response|>"
        );

        // The full agentic round trip, thinking on.
        let got = r()
            .render(
                &[
                    Message::new("system", "You are helpful."),
                    Message::new("user", "List files"),
                    Message {
                        role: "assistant".into(),
                        content: "<|channel>I should use bash<channel|>".into(),
                        tool_calls: vec![call("bash", r#"{"command":"ls"}"#)],
                        ..Default::default()
                    },
                    Message {
                        role: "tool".into(),
                        tool_name: "bash".into(),
                        content: "file1.txt".into(),
                        ..Default::default()
                    },
                    Message::new("assistant", "Here are the files."),
                    Message::new("user", "Thanks"),
                ],
                &[],
                Some(&ThinkValue::Bool(true)),
            )
            .unwrap();
        assert!(
            got.contains(
                "<|turn>model\n<|tool_call>call:bash{command:<|\"|>ls<|\"|>}<tool_call|>\
                 <|tool_response>response:bash{value:<|\"|>file1.txt<|\"|>}<tool_response|>\
                 Here are the files.<turn|>\n"
            ),
            "{got}"
        );
        assert!(
            got.ends_with("<|turn>user\nThanks<turn|>\n<|turn>model\n"),
            "{got}"
        );
    }

    /// Upstream `tool_call_pending_response`: a call with nothing answering it
    /// leaves `<|tool_response>` **open** for the model to complete.
    #[test]
    fn an_unanswered_tool_call_leaves_the_response_block_open() {
        let got = r()
            .render(
                &[
                    Message::new("user", "Go"),
                    Message {
                        role: "assistant".into(),
                        tool_calls: vec![call("bash", r#"{"command":"ls"}"#)],
                        ..Default::default()
                    },
                ],
                &[],
                None,
            )
            .unwrap();
        assert!(got.ends_with("<tool_call|><|tool_response>"), "{got}");
    }

    /// Upstream `strip_thinking_history` -- an inline `<|channel>` block is cut
    /// out of historical assistant content, leaving an empty model turn.
    #[test]
    fn inline_channel_blocks_are_stripped_from_history() {
        assert_eq!(
            r().render(
                &[
                    Message::new("user", "Hi"),
                    Message::new("assistant", "<|channel>just thinking<channel|>"),
                    Message::new("user", "More"),
                ],
                &[],
                None
            )
            .unwrap(),
            "<bos><|turn>user\nHi<turn|>\n<|turn>model\n<turn|>\n<|turn>user\nMore<turn|>\n<|turn>model\n"
        );

        assert_eq!(strip_thinking("a<|channel>x<channel|>b"), "ab");
        // Dangling open tag: everything from it is dropped.
        assert_eq!(strip_thinking("a<|channel>x"), "a");
        assert_eq!(strip_thinking("  padded  "), "padded");
    }

    /// Upstream `sorted_args` / `numeric_arguments` / `boolean_argument`:
    /// argument keys sort, whole floats lose their `.0`, strings are delimited.
    #[test]
    fn tool_call_arguments_are_sorted_and_typed_the_gemma_way() {
        let tc = call("f", r#"{"zeta":1,"alpha":"x","mid":true,"frac":1.5}"#);
        assert_eq!(
            r().format_tool_call(&tc),
            "<|tool_call>call:f{alpha:<|\"|>x<|\"|>,frac:1.5,mid:true,zeta:1}<tool_call|>"
        );
        // A map argument writes BARE keys; the schema writer would delimit them.
        assert_eq!(
            format_arg_value(&json!({"b": 2, "a": "s"})),
            "{a:<|\"|>s<|\"|>,b:2}"
        );
        assert_eq!(
            format_arg_value(&json!([1, "x", null])),
            "[1,<|\"|>x<|\"|>,null]"
        );
    }

    /// `gemma4-large`'s one difference: an empty thought block when thinking is
    /// off, so the model does not open one itself.
    #[test]
    fn the_large_preset_emits_an_empty_thought_block_when_not_thinking() {
        let large = Gemma4Renderer {
            use_img_tags: false,
            empty_block_on_nothink: true,
        };
        let got = large
            .render(&[Message::new("user", "Hi")], &[], None)
            .unwrap();
        assert!(
            got.ends_with("<|turn>model\n<|channel>thought\n<channel|>"),
            "{got}"
        );

        // With thinking on, the block is left for real reasoning.
        let got = large
            .render(
                &[Message::new("user", "Hi")],
                &[],
                Some(&ThinkValue::Bool(true)),
            )
            .unwrap();
        assert!(got.ends_with("<|turn>model\n"), "{got}");
    }

    /// Upstream `adjacent_assistants_continue_same_model_turn`.
    #[test]
    fn adjacent_assistant_messages_continue_one_model_turn() {
        let got = r()
            .render(
                &[
                    Message::new("user", "Hi"),
                    Message::new("assistant", "First."),
                    Message::new("assistant", "Second."),
                ],
                &[],
                None,
            )
            .unwrap();
        // One `<|turn>model` opener for the two assistant messages.
        assert_eq!(got.matches("<|turn>model\n").count(), 2, "{got}");
        assert!(
            got.contains("<|turn>model\nFirst.Second.<turn|>\n"),
            "{got}"
        );
    }

    #[test]
    fn the_bos_is_written_into_the_prompt_and_also_reported() {
        assert_eq!(r().leading_bos(), "<bos>");
        assert!(
            r().render(&[Message::new("user", "Hi")], &[], None)
                .unwrap()
                .starts_with("<bos>")
        );
    }

    /// `bd-iut`, gemma4's copy of the same guard. Upstream:
    /// `Properties != nil || Type != ""`. With no type, whether a `parameters:`
    /// block appears at all turns entirely on set-but-empty vs absent, and no
    /// upstream fixture reaches it -- so it is pinned here instead.
    #[test]
    fn a_typeless_tool_still_declares_parameters_when_its_property_map_is_set() {
        let decl = |parameters: serde_json::Value| {
            r().render_tool_declaration(&tool(json!({
                "type": "function",
                "function": {"name": "noargs", "description": "", "parameters": parameters}
            })))
        };

        assert_eq!(
            decl(json!({"type": "", "properties": {}})),
            "<|tool>declaration:noargs{description:<|\"|><|\"|>,parameters:{}}<tool|>",
            "set-but-empty means `takes no arguments` -- the block must appear"
        );
        assert_eq!(
            decl(json!({"type": ""})),
            "<|tool>declaration:noargs{description:<|\"|><|\"|>}<tool|>",
            "absent means `nobody described this` -- no block at all"
        );
    }
}
