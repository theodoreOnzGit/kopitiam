//! Go-shaped JSON, because the prompt must match byte-for-byte.
//!
//! **Upstream:** `model/renderers/json.go` (`marshalWithSpaces`), plus the
//! `encoding/json` behaviour that every renderer leans on when it does
//! `json.Marshal(tool)` / `json.Marshal(args)`.
//!
//! ## Why we cannot just call `serde_json::to_string`
//!
//! A renderer's whole job is to reproduce the *exact* string a model family was
//! fine-tuned on. The moment we hand a model JSON that differs from what its
//! training data looked like, the model doesn't error -- it just quietly gets
//! worse. So the JSON that goes into a prompt must come out the way Go's
//! `encoding/json` would have written it. Three places where `serde_json`
//! and Go disagree, and all three would bite us:
//!
//! 1. **HTML escaping.** Go escapes `<`, `>` and `&` into their `\u00xx` forms
//!    (`003c`, `003e`, `0026`) by default -- `SetEscapeHTML(true)` is the
//!    default on `json.Marshal`.
//!    `serde_json` leaves them raw. Tool descriptions say
//!    things like *"values <= 10"* often enough for this to matter.
//! 2. **`api.PropertyType` marshals as a bare string when it holds exactly one
//!    type** -- `"type":"string"`, not `"type":["string"]`. That is a custom
//!    `MarshalJSON` upstream (`api/types.go`), and our
//!    [`crate::api::PropertyType`] is `#[serde(transparent)]` over a `Vec`, so
//!    plain serde would always emit the array form.
//! 3. **Field order is declaration order, not sorted.** Go structs marshal in
//!    declaration order; a `map[string]any` marshals **sorted**. Both matter,
//!    in different places, so we do both deliberately below.
//!
//! So this module hand-rolls the emitter for the handful of shapes a renderer
//! ever marshals. Small price, and it means the framing can be tested without
//! a feature flag on `serde_json` deciding whether our prompts are right.
//!
//! ## Known divergences (deliberate, say-so-out-loud)
//!
//! * **`"properties"` when there are none.** Upstream's `Properties` is a
//!   `*ToolPropertiesMap`, so a caller who never set it marshals `null`, while
//!   one who set it and left it empty marshals `{}`. Our [`crate::api`] models it
//!   as a plain `IndexMap` with no nil state, so empty maps to **`null`** -- the
//!   nil case, which is the one upstream's LFM2 fixture actually asserts. The
//!   set-but-empty case is unrepresentable here; no fixture exercises it.
//! * **Extreme floats.** Go's float encoder and Rust's shortest-round-trip
//!   encoder can disagree on the exponent spelling for magnitudes below `1e-6`
//!   or at/above `1e21`. Tool arguments in that range basically don't happen,
//!   and no fixture has one.

use serde_json::Value;

use crate::api::{PropertyType, Tool, ToolFunction, ToolFunctionParameters, ToolProperty};

/// Escape one string the way Go's `encoding/json` does, quotes included.
///
/// The `<` / `>` / `&` bit is the one that catches people out -- Go escapes
/// them so that JSON can be dropped straight into an HTML page without a
/// script-injection hole. Nothing here is HTML, but the model was trained on
/// Go's output, so we copy Go.
fn write_go_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            // Go's htmlSafeSet: these three go out as escapes by default.
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            // U+2028/U+2029 are legal JSON but break JavaScript parsers, so Go
            // escapes them too.
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Marshal an arbitrary JSON value the Go way.
///
/// **Object keys come out sorted**, because that is what Go does when it
/// marshals a `map[string]any` -- and every `any` a renderer touches (tool-call
/// argument values, `items`, `enum` entries) arrived as a Go map. Note this is
/// the *opposite* of what happens to the argument map itself, which keeps
/// insertion order (see [`write_go_arguments`]); the two rules live side by side
/// on purpose.
pub(crate) fn write_go_value(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => write_go_string(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_go_value(item, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            // `serde_json::Map` is a `BTreeMap` here (the `preserve_order`
            // feature is NOT on in this workspace), so iteration is already
            // sorted -- same as Go. Collecting + sorting keeps us right even if
            // somebody flips that feature on later.
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_go_string(k, out);
                out.push(':');
                write_go_value(&m[*k], out);
            }
            out.push('}');
        }
    }
}

/// [`write_go_value`] into a fresh `String`.
pub(crate) fn go_value(v: &Value) -> String {
    let mut s = String::new();
    write_go_value(v, &mut s);
    s
}

/// `api.PropertyType.MarshalJSON` -- **one type marshals as a bare string.**
///
/// `["string"]` becomes `"string"`, `["string","number"]` stays an array. Get
/// this wrong and every tool schema in a GLM / DeepSeek prompt is subtly off.
fn write_property_type(t: &PropertyType, out: &mut String) {
    if t.0.len() == 1 {
        write_go_string(&t.0[0], out);
        return;
    }
    out.push('[');
    for (i, s) in t.0.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_go_string(s, out);
    }
    out.push(']');
}

/// A tiny helper so the struct emitters below read like the Go structs do.
struct ObjWriter<'a> {
    out: &'a mut String,
    first: bool,
}

impl<'a> ObjWriter<'a> {
    fn new(out: &'a mut String) -> Self {
        out.push('{');
        Self { out, first: true }
    }

    fn key(&mut self, k: &str) -> &mut String {
        if !self.first {
            self.out.push(',');
        }
        self.first = false;
        write_go_string(k, self.out);
        self.out.push(':');
        self.out
    }

    fn finish(self) {
        self.out.push('}');
    }
}

/// `api.ToolProperty` -- field order is Go's **declaration** order:
/// `anyOf`, `type`, `items`, `description`, `enum`, `properties`, `required`.
/// Every one of them is `omitempty`, so an empty property marshals `{}`.
pub(crate) fn write_go_property(p: &ToolProperty, out: &mut String) {
    let mut o = ObjWriter::new(out);
    if !p.any_of.is_empty() {
        let out = o.key("anyOf");
        out.push('[');
        for (i, a) in p.any_of.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_go_property(a, out);
        }
        out.push(']');
    }
    if !p.prop_type.is_empty() {
        write_property_type(&p.prop_type, o.key("type"));
    }
    if let Some(items) = &p.items {
        write_go_value(items, o.key("items"));
    }
    if !p.description.is_empty() {
        write_go_string(&p.description, o.key("description"));
    }
    if !p.enum_values.is_empty() {
        let out = o.key("enum");
        out.push('[');
        for (i, e) in p.enum_values.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_go_value(e, out);
        }
        out.push(']');
    }
    if let Some(props) = &p.properties {
        let out = o.key("properties");
        out.push('{');
        for (i, (k, v)) in props.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_go_string(k, out);
            out.push(':');
            write_go_property(v, out);
        }
        out.push('}');
    }
    if !p.required.is_empty() {
        let out = o.key("required");
        out.push('[');
        for (i, r) in p.required.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_go_string(r, out);
        }
        out.push(']');
    }
    o.finish();
}

/// `api.ToolFunctionParameters` -- declaration order `type`, `$defs`, `items`,
/// `required`, `properties`.
///
/// `type` and `properties` have **no** `omitempty` upstream, so they are always
/// present even when empty. `properties` keeps its **insertion** order (it is an
/// ordered map upstream, and a JSON-Schema property list reads wrong when
/// alphabetised).
pub(crate) fn write_go_parameters(p: &ToolFunctionParameters, out: &mut String) {
    let mut o = ObjWriter::new(out);
    write_go_string(&p.param_type, o.key("type"));
    if let Some(defs) = &p.defs {
        write_go_value(defs, o.key("$defs"));
    }
    if let Some(items) = &p.items {
        write_go_value(items, o.key("items"));
    }
    if !p.required.is_empty() {
        let out = o.key("required");
        out.push('[');
        for (i, r) in p.required.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_go_string(r, out);
        }
        out.push(']');
    }
    {
        let out = o.key("properties");
        // Upstream's field is a `*ToolPropertiesMap` with no `omitempty`, so all
        // three states are distinguishable on the wire and we now reproduce them
        // exactly (this was `bd-iut`):
        //
        //   None            -> `null`  ("nobody said what this tool takes")
        //   Some(empty)     -> `{}`    ("this tool takes NO arguments")
        //   Some(populated) -> `{...}`
        //
        // The first is by far the common case, and upstream's own LFM2 fixture
        // asserts `"properties": null`. The middle one matters because a
        // no-argument tool emitted as `null` reads to a model as "I don't know
        // this tool's arguments" rather than "it has none".
        match &p.properties {
            None => out.push_str("null"),
            Some(props) => {
                out.push('{');
                for (i, (k, v)) in props.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_go_string(k, out);
                    out.push(':');
                    write_go_property(v, out);
                }
                out.push('}');
            }
        }
    }
    o.finish();
}

/// `api.ToolFunction` -- `name`, `description` (omitempty), `parameters`.
fn write_go_function(f: &ToolFunction, out: &mut String) {
    let mut o = ObjWriter::new(out);
    write_go_string(&f.name, o.key("name"));
    if !f.description.is_empty() {
        write_go_string(&f.description, o.key("description"));
    }
    write_go_parameters(&f.parameters, o.key("parameters"));
    o.finish();
}

/// `api.Tool` -- `type`, `items` (omitempty), `function`.
pub(crate) fn write_go_tool(t: &Tool, out: &mut String) {
    let mut o = ObjWriter::new(out);
    write_go_string(&t.tool_type, o.key("type"));
    if let Some(items) = &t.items {
        write_go_value(items, o.key("items"));
    }
    write_go_function(&t.function, o.key("function"));
    o.finish();
}

/// [`write_go_tool`] into a fresh `String`.
pub(crate) fn go_tool(t: &Tool) -> String {
    let mut s = String::new();
    write_go_tool(t, &mut s);
    s
}

/// `json.Marshal([]api.Tool)`.
pub(crate) fn go_tools(tools: &[Tool]) -> String {
    let mut s = String::from("[");
    for (i, t) in tools.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        write_go_tool(t, &mut s);
    }
    s.push(']');
    s
}

/// `json.Marshal(api.ToolCallFunctionArguments)` -- **insertion order kept.**
///
/// This is the mirror image of [`write_go_value`]'s sorted objects, and the
/// difference is load-bearing: the argument map is what the *model* emitted, so
/// its order is data, while a nested `any` came out of a Go map and Go sorts
/// those. Alphabetising the outer map would put a different string in front of
/// the model than the one it was trained to see.
pub(crate) fn go_arguments(args: &crate::api::ToolCallArguments) -> String {
    let mut s = String::from("{");
    for (i, (k, v)) in args.0.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        write_go_string(k, &mut s);
        s.push(':');
        write_go_value(v, &mut s);
    }
    s.push('}');
    s
}

/// Add a space after every `:` and `,` that sits **outside** a JSON string.
///
/// **Upstream:** the second half of `marshalWithSpaces` (`json.go`), and --
/// byte-for-byte the same algorithm -- `formatGLM47ToolJSON` (`glm47.go`). Two
/// names upstream, one behaviour; ported once here.
///
/// Why it exists at all: Python's `json.dumps` puts those spaces in by default,
/// so any model family whose chat template was authored in Python was fine-tuned
/// on `{"a": 1, "b": 2}`, not Go's `{"a":1,"b":2}`. This is the shim that makes
/// Go's output look Pythonic.
///
/// Runs over **bytes**, deliberately: every byte of a multi-byte UTF-8 sequence
/// has the high bit set, so it can never be mistaken for an ASCII `"`, `\`, `:`
/// or `,`. That makes the scanner safe on arbitrary text without decoding it.
pub(crate) fn add_spaces_outside_strings(json: &str) -> String {
    let b = json.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len() + b.len() / 8);
    let (mut in_str, mut esc) = (false, false);
    for &c in b {
        if in_str {
            out.push(c);
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
        match c {
            b'"' => {
                in_str = true;
                out.push(c);
            }
            b':' => out.extend_from_slice(b": "),
            b',' => out.extend_from_slice(b", "),
            _ => out.push(c),
        }
    }
    // Only ASCII was ever inserted, and multi-byte sequences were copied whole,
    // so the result is still valid UTF-8.
    String::from_utf8(out).unwrap_or_else(|_| json.to_string())
}

/// `marshalWithSpaces(tool)`.
pub(crate) fn marshal_tool_with_spaces(t: &Tool) -> String {
    add_spaces_outside_strings(&go_tool(t))
}

/// `marshalWithSpaces(tools)` -- olmo3 marshals the whole slice in one go.
pub(crate) fn marshal_tools_with_spaces(tools: &[Tool]) -> String {
    add_spaces_outside_strings(&go_tools(tools))
}

/// Re-indent compact Go JSON the way `json.MarshalIndent(v, "", indent)` would.
///
/// **Upstream:** used by `cogito.go`, which pretty-prints each tool with a
/// four-space indent inside a ```` ```json ```` fence.
///
/// Go's indenter is purely textual -- it walks the compact output and inserts
/// newlines + indentation around `{`/`[`/`,`, and a single space after `:`. An
/// **empty** object or array stays `{}` / `[]` with nothing between the braces,
/// which is the one case a naive implementation gets wrong.
pub(crate) fn indent_go_json(json: &str, indent: &str) -> String {
    let b = json.as_bytes();
    let mut out = String::with_capacity(b.len() * 2);
    let mut depth = 0usize;
    let (mut in_str, mut esc) = (false, false);
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if in_str {
            out.push(c as char);
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' => {
                in_str = true;
                out.push('"');
            }
            b'{' | b'[' => {
                // Peek: an immediately-closing bracket stays on one line.
                let closing = if c == b'{' { b'}' } else { b']' };
                if i + 1 < b.len() && b[i + 1] == closing {
                    out.push(c as char);
                    out.push(closing as char);
                    i += 2;
                    continue;
                }
                depth += 1;
                out.push(c as char);
                out.push('\n');
                out.push_str(&indent.repeat(depth));
            }
            b'}' | b']' => {
                depth -= 1;
                out.push('\n');
                out.push_str(&indent.repeat(depth));
                out.push(c as char);
            }
            b',' => {
                out.push(',');
                out.push('\n');
                out.push_str(&indent.repeat(depth));
            }
            b':' => out.push_str(": "),
            _ => out.push(c as char),
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ToolCallArguments;
    use serde_json::json;

    /// Upstream `json_test.go` `TestMarshalWithSpaces`: a colon or comma inside
    /// a string value must be left completely alone.
    #[test]
    fn spaces_go_in_outside_strings_and_nowhere_else() {
        assert_eq!(
            add_spaces_outside_strings(r#"{"a":1,"b":2}"#),
            r#"{"a": 1, "b": 2}"#
        );
        assert_eq!(
            add_spaces_outside_strings(r#"{"a":"x:y,z"}"#),
            r#"{"a": "x:y,z"}"#
        );
        // An escaped quote must not be read as the end of the string.
        assert_eq!(
            add_spaces_outside_strings(r#"{"a":"he said \"x:y\""}"#),
            r#"{"a": "he said \"x:y\""}"#
        );
        assert_eq!(add_spaces_outside_strings("[]"), "[]");
    }

    #[test]
    fn multibyte_text_survives_the_spacing_pass() {
        // The DeepSeek special tokens are full-width; the scanner must not
        // chop them.
        let s = r#"{"tok":"<｜tool▁sep｜>","n":1}"#;
        assert_eq!(
            add_spaces_outside_strings(s),
            r#"{"tok": "<｜tool▁sep｜>", "n": 1}"#
        );
    }

    #[test]
    fn a_single_property_type_marshals_as_a_bare_string() {
        let mut s = String::new();
        write_property_type(&PropertyType(vec!["string".into()]), &mut s);
        assert_eq!(s, r#""string""#);
        s.clear();
        write_property_type(
            &PropertyType(vec!["string".into(), "number".into()]),
            &mut s,
        );
        assert_eq!(s, r#"["string","number"]"#);
    }

    #[test]
    fn angle_brackets_and_ampersands_are_escaped_like_go_does() {
        let mut s = String::new();
        write_go_string("a < b && c > d", &mut s);
        assert_eq!(s, "\"a \\u003c b \\u0026\\u0026 c \\u003e d\"");
    }

    #[test]
    fn nested_any_values_come_out_sorted_like_a_go_map() {
        assert_eq!(go_value(&json!({"z": 1, "a": 2})), r#"{"a":2,"z":1}"#);
    }

    /// ...while the argument map itself keeps what the model actually said.
    #[test]
    fn tool_call_arguments_keep_insertion_order() {
        let args: ToolCallArguments = serde_json::from_str(r#"{"zeta":1,"alpha":2}"#).unwrap();
        assert_eq!(go_arguments(&args), r#"{"zeta":1,"alpha":2}"#);
    }

    #[test]
    fn a_tool_marshals_in_gos_declaration_order() {
        let tool: Tool = serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "required": ["location"],
                    "properties": {
                        "location": {"type": "string", "description": "City"}
                    }
                }
            }
        }))
        .unwrap();
        assert_eq!(
            go_tool(&tool),
            r#"{"type":"function","function":{"name":"get_weather","description":"Get the weather","parameters":{"type":"object","required":["location"],"properties":{"location":{"type":"string","description":"City"}}}}}"#
        );
    }

    #[test]
    fn empty_containers_stay_on_one_line_when_indenting() {
        assert_eq!(
            indent_go_json(r#"{"a":{},"b":[]}"#, "    "),
            "{\n    \"a\": {},\n    \"b\": []\n}"
        );
    }
}
