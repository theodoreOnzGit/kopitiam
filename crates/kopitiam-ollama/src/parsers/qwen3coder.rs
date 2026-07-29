//! Qwen3-Coder response parser.
//!
//! **Upstream:** `model/parsers/qwen3coder.go` (ollama, MIT).
//!
//! ## What is different about this family
//!
//! Qwen3-Coder has **no thinking**, and its tool calls are not JSON -- they are
//! an XML-*ish* dialect that is not actually valid XML:
//!
//! ```text
//! <tool_call>
//! <function=get_current_temperature>
//! <parameter=location>
//! San Francisco
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Note `<function=name>`, not `<function name="...">`. Upstream repairs that
//! into real XML first ([`transform_to_xml`]) and then feeds it to Go's
//! `encoding/xml`. We keep the repair step **exactly** as upstream wrote it --
//! its output is pinned by upstream's own `TestQwenXMLTransform` -- and then read
//! the repaired document with a small purpose-built reader instead of a general
//! XML parser.
//!
//! **Why not just scan the raw text and skip the XML round trip?** Because the
//! repair step is not identity: escaping the character data and then unescaping
//! it on the way out is what makes `ls && echo "a > b"` survive as a parameter
//! value (upstream issue #12357), and the same round trip is what turns a stray
//! `<a=b>` inside a value into `<a name="b">`. Reproducing those behaviours
//! without reproducing the pipeline would be guessing. Keeping the pipeline and
//! swapping only the XML reader keeps the Pure Rust Core promise (no new
//! dependency) without changing a single observable outcome.
//!
//! ## Types come from the tool schema, not from the text
//!
//! The wire format has no types -- every parameter value arrives as a string. So
//! [`parse_value`] looks the parameter up in the caller's [`Tool`] schema and
//! coerces. That is why `Init` is handed the tools and keeps them.

use serde_json::{Map, Value};

use crate::api::{
    Message, PropertyType, ThinkValue, Tool, ToolCall, ToolCallArguments, ToolCallFunction,
};

use super::{Parsed, Parser, ParserError, chop, emit_unambiguous, overlap};

/// **Upstream:** `toolOpenTag` / `toolCloseTag` in `qwen3coder.go`. Shared with
/// the qwen3.5 and qwen3-vl parsers, exactly as in Go.
pub(super) const TOOL_OPEN_TAG: &str = "<tool_call>";
pub(super) const TOOL_CLOSE_TAG: &str = "</tool_call>";

/// **Upstream:** `qwenParserState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    #[default]
    LookingForToolStart,
    CollectingToolContent,
}

/// **Upstream:** the `qwenEvent` sum type.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum QwenEvent {
    Content(String),
    RawToolCall(String),
}

/// **Upstream:** `Qwen3CoderParser`.
#[derive(Debug, Default)]
pub struct Qwen3CoderParser {
    state: State,
    acc: String,
    tools: Vec<Tool>,
    call_index: usize,
}

impl Qwen3CoderParser {
    /// One step of the machine. **Upstream:** the free function `eat(p)`.
    ///
    /// Returns the events that are now unambiguous plus "did I change state", so
    /// the caller knows whether another step could produce more.
    fn eat(&mut self) -> (Vec<QwenEvent>, bool) {
        let mut events = Vec::new();

        match self.state {
            State::LookingForToolStart => {
                if let Some(idx) = self.acc.find(TOOL_OPEN_TAG) {
                    // Full open tag: emit the content before it, trailing
                    // whitespace trimmed off (it was framing for the tag).
                    let (before, rest) = chop(&self.acc, idx);
                    let before = before.trim_end().to_string();
                    // NOTE: `after` is deliberately NOT left-trimmed here --
                    // upstream leaves the tool body untouched, and the XML reader
                    // is the one that gets to decide about whitespace.
                    let after = rest[TOOL_OPEN_TAG.len()..].to_string();
                    if !before.is_empty() {
                        events.push(QwenEvent::Content(before));
                    }
                    self.acc = after;
                    self.state = State::CollectingToolContent;
                    return (events, true);
                }

                // No full tag. Hold back whatever tail could still become one
                // (plus the whitespace in front of it), emit the rest.
                let overlap_len = overlap(&self.acc, TOOL_OPEN_TAG);
                let unambiguous = emit_unambiguous(&mut self.acc, overlap_len);
                if !unambiguous.is_empty() {
                    events.push(QwenEvent::Content(unambiguous));
                }
                (events, false)
            }

            State::CollectingToolContent => {
                // No overlap check on purpose: a half-parsed tool call is never
                // streamed back, so there is nothing to be eager about. Wait for
                // the whole closing tag.
                let Some(idx) = self.acc.find(TOOL_CLOSE_TAG) else {
                    return (events, false);
                };
                let (before, rest) = chop(&self.acc, idx);
                let raw = before.to_string();
                // Whitespace between the tool call and the content after it is
                // dropped -- otherwise every tool call leaves a stray newline in
                // the reply.
                let after = rest[TOOL_CLOSE_TAG.len()..].trim_start().to_string();
                self.acc = after;
                events.push(QwenEvent::RawToolCall(raw));
                self.state = State::LookingForToolStart;
                (events, true)
            }
        }
    }

    /// **Upstream:** `parseEvents`.
    fn parse_events(&mut self) -> Vec<QwenEvent> {
        let mut all = Vec::new();
        let mut keep_looping = true;
        while keep_looping {
            let (events, again) = self.eat();
            keep_looping = again;
            all.extend(events);
        }
        all
    }

    /// Feed a string through the tool-call machine only, returning content and
    /// calls. Used by [`super::Qwen35Parser`], which owns thinking itself and
    /// delegates everything post-thinking here -- exactly as upstream's
    /// `Qwen35Parser` embeds a `Qwen3CoderParser`.
    pub(super) fn add_content(&mut self, s: &str) -> Result<(String, Vec<ToolCall>), ParserError> {
        self.acc.push_str(s);
        let events = self.parse_events();
        let mut content = String::new();
        let mut calls = Vec::new();
        for event in events {
            match event {
                QwenEvent::RawToolCall(raw) => {
                    let mut call = parse_tool_call(&raw, &self.tools)?;
                    call.function.index = self.call_index;
                    self.call_index += 1;
                    calls.push(call);
                }
                QwenEvent::Content(c) => content.push_str(&c),
            }
        }
        Ok((content, calls))
    }
}

impl Parser for Qwen3CoderParser {
    fn init(
        &mut self,
        tools: Vec<Tool>,
        _last_message: Option<&Message>,
        _think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        self.tools = tools.clone();
        self.call_index = 0;
        // Qwen doesn't modify tools.
        tools
    }

    fn add(&mut self, s: &str, _done: bool) -> Result<Parsed, ParserError> {
        let (content, calls) = self.add_content(s)?;
        Ok(Parsed {
            content,
            thinking: String::new(),
            calls,
        })
    }

    fn preserved_tokens(&self) -> Vec<&'static str> {
        vec![TOOL_OPEN_TAG, TOOL_CLOSE_TAG]
    }

    fn has_tool_support(&self) -> bool {
        true
    }

    fn has_thinking_support(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// The XML-ish tool call body
// ---------------------------------------------------------------------------

/// Parse one tool-call body into a [`ToolCall`].
///
/// **Upstream:** `parseToolCall`.
///
/// `tools` is consulted only to look up **parameter types** -- the wire format
/// carries none, so an untyped parameter stays a string. A tool the caller never
/// offered is still parsed, just with everything as strings; upstream does the
/// same rather than rejecting it.
pub(super) fn parse_tool_call(raw: &str, tools: &[Tool]) -> Result<ToolCall, ParserError> {
    let xml = transform_to_xml(raw);
    let (name, params) = read_function_element(&xml)?;

    let matched = tools.iter().find(|t| t.function.name == name);

    let mut arguments = ToolCallArguments::new();
    for (param_name, param_value) in params {
        // Look up the declared type, flattening an `anyOf` union into one list of
        // candidate types -- upstream does the same, and `parse_value`'s
        // precedence rules then pick the most specific one that parses.
        let mut param_type = PropertyType::default();
        if let Some(tool) = matched
            && let Some(prop) = tool.function.parameters.properties.get(&param_name)
        {
            if prop.any_of.is_empty() {
                param_type = prop.prop_type.clone();
            } else {
                for branch in &prop.any_of {
                    param_type.0.extend(branch.prop_type.0.iter().cloned());
                }
            }
        }
        arguments.set(param_name, parse_value(&param_value, &param_type));
    }

    Ok(ToolCall {
        id: String::new(),
        function: ToolCallFunction {
            index: 0,
            name,
            arguments,
        },
    })
}

/// Coerce a raw parameter string to the most specific declared type that parses.
///
/// **Upstream:** `parseValue`, including its precedence comment. The order is
/// `null -> boolean -> integer -> number -> array -> object -> string`, and it is
/// deliberate: with a declared union of `["string","number"]`, `"123"` becomes
/// the number `123` while `"hello"` stays the string `"hello"`.
///
/// Two upstream quirks that look like bugs and are not:
///
/// * exactly **one** leading and **one** trailing newline are stripped first,
///   because the wire format puts the value on its own line;
/// * `"null"` (case-insensitively) wins over every declared type.
///
/// And one that really is a quirk, faithfully kept: if the declared type is
/// *only* `boolean` and the text is neither `true` nor `false`, the answer is
/// `false`, not an error. Upstream: *"matching reference"*.
///
/// **Divergence, stated:** upstream distinguishes Go `int` from `int64` when a
/// value does or does not fit in an int32. JSON has one number type, so that
/// distinction evaporates on the way into [`serde_json::Value`]. Nothing
/// observable changes -- both spellings serialise to the same digits.
pub(super) fn parse_value(raw: &str, param_type: &PropertyType) -> Value {
    // One leading and one trailing newline, no more. Follows the reference impl.
    let raw = raw.strip_prefix('\n').unwrap_or(raw);
    let raw = raw.strip_suffix('\n').unwrap_or(raw);

    if raw.eq_ignore_ascii_case("null") {
        return Value::Null;
    }

    if param_type.0.is_empty() {
        return Value::String(raw.to_string());
    }

    let has = |t: &str| param_type.0.iter().any(|x| x == t);
    let only_one = param_type.0.len() == 1;

    if has("boolean") {
        match raw.to_ascii_lowercase().as_str() {
            "true" => return Value::Bool(true),
            "false" => return Value::Bool(false),
            _ => {}
        }
        if only_one {
            return Value::Bool(false);
        }
    }

    if has("integer") {
        if let Ok(i) = raw.parse::<i64>() {
            return Value::from(i);
        }
        if only_one {
            return Value::String(raw.to_string());
        }
    }

    if has("number") {
        if let Ok(f) = raw.parse::<f64>() {
            // A number with no fractional part comes back as an integer, matching
            // the reference implementation -- `3.0` becomes `3`.
            if f == f.trunc() && f.is_finite() {
                return Value::from(f as i64);
            }
            return Value::from(f);
        }
        if only_one {
            return Value::String(raw.to_string());
        }
    }

    if has("array") {
        if let Ok(arr) = serde_json::from_str::<Vec<Value>>(raw) {
            return Value::Array(arr);
        }
        if only_one {
            return Value::String(raw.to_string());
        }
    }

    if has("object") {
        if let Ok(obj) = serde_json::from_str::<Map<String, Value>>(raw) {
            return Value::Object(obj);
        }
        if only_one {
            return Value::String(raw.to_string());
        }
    }

    // String always succeeds. If nothing matched and `string` was not even
    // offered, we still fall back to string: the reference implementation would
    // try to parse a Python literal here, and upstream purposefully does not.
    Value::String(raw.to_string())
}

/// Repair qwen's XML-ish tool call into real XML.
///
/// **Upstream:** `transformToXML`, whose output is pinned by upstream's
/// `TestQwenXMLTransform`. Two passes:
///
/// 1. every `<tag=value>` becomes `<tag name="escaped-value">` -- the attribute
///    value is escaped the way Go's `xml.EscapeText` does it, so a stray quote
///    cannot break out of the attribute;
/// 2. everything **between** the resulting `<function>` / `<parameter>` tags is
///    treated as character data and gets `&`, `<`, `>` escaped -- and *only*
///    those three, so newlines and tabs inside a parameter value survive
///    byte-for-byte. That is why upstream hand-writes `escapeTextNode` instead of
///    reusing `xml.EscapeText` (which would turn a newline into `&#xA;`).
///
/// Upstream uses two regexes; we hand-roll the same two matchers rather than
/// take a `regex` dependency. The grammars they accept are reproduced exactly:
///
/// * `<(\w+)=([^>]+)>` where `\w` is ASCII `[0-9A-Za-z_]`;
/// * `</?(?:function|parameter)(?:\s+name="[^"]*")?>`.
fn transform_to_xml(raw: &str) -> String {
    let transformed = rewrite_eq_tags(raw);

    let mut out = String::with_capacity(transformed.len());
    let mut last = 0usize;
    for (start, end) in find_xml_tags(&transformed) {
        if start > last {
            escape_text_node(&mut out, &transformed[last..start]);
        }
        out.push_str(&transformed[start..end]);
        last = end;
    }
    if last < transformed.len() {
        escape_text_node(&mut out, &transformed[last..]);
    }
    out
}

/// Pass 1 of [`transform_to_xml`]: `<tag=value>` -> `<tag name="value">`.
/// **Upstream:** the `qwenTagRegex.ReplaceAllStringFunc` call.
fn rewrite_eq_tags(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = String::with_capacity(raw.len());
    let mut i = 0usize;

    while i < bytes.len() {
        if bytes[i] != b'<' {
            // Copy one whole char so we never split a UTF-8 sequence.
            let ch_len = char_len_at(raw, i);
            out.push_str(&raw[i..i + ch_len]);
            i += ch_len;
            continue;
        }

        // `<` then `\w+` then `=` then `[^>]+` then `>`.
        let mut j = i + 1;
        while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
            j += 1;
        }
        if j == i + 1 || j >= bytes.len() || bytes[j] != b'=' {
            out.push('<');
            i += 1;
            continue;
        }
        let tag = &raw[i + 1..j];
        let val_start = j + 1;
        let Some(rel) = raw[val_start..].find('>') else {
            out.push('<');
            i += 1;
            continue;
        };
        if rel == 0 {
            // `[^>]+` needs at least one character.
            out.push('<');
            i += 1;
            continue;
        }
        let value = &raw[val_start..val_start + rel];
        out.push('<');
        out.push_str(tag);
        out.push_str(" name=\"");
        escape_attr(&mut out, value);
        out.push_str("\">");
        i = val_start + rel + 1;
    }

    out
}

/// Byte length of the UTF-8 char starting at `i`. Always >= 1, so callers cannot
/// loop forever even on input that is somehow not on a boundary.
fn char_len_at(s: &str, i: usize) -> usize {
    s[i..].chars().next().map(char::len_utf8).unwrap_or(1)
}

/// Pass 2's matcher: byte ranges of every `</?(function|parameter)( name="...")?>`.
fn find_xml_tags(s: &str) -> Vec<(usize, usize)> {
    let bytes = s.as_bytes();
    let mut hits = Vec::new();
    let mut i = 0usize;

    while i < bytes.len() {
        if bytes[i] != b'<' {
            i += char_len_at(s, i);
            continue;
        }
        let mut j = i + 1;
        if j < bytes.len() && bytes[j] == b'/' {
            j += 1;
        }
        let name = if s[j..].starts_with("function") {
            "function"
        } else if s[j..].starts_with("parameter") {
            "parameter"
        } else {
            i += 1;
            continue;
        };
        j += name.len();

        // Optional `\s+ name="[^"]*"`.
        let mut k = j;
        if k < bytes.len() && bytes[k].is_ascii_whitespace() {
            while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                k += 1;
            }
            if s[k..].starts_with("name=\"") {
                k += "name=\"".len();
                match s[k..].find('"') {
                    Some(rel) => j = k + rel + 1,
                    None => {
                        i += 1;
                        continue;
                    }
                }
            }
        }

        if j < bytes.len() && bytes[j] == b'>' {
            hits.push((i, j + 1));
            i = j + 1;
        } else {
            i += 1;
        }
    }

    hits
}

/// Escape an XML attribute value the way Go's `xml.EscapeText` does.
///
/// **Provenance:** Go's `encoding/xml` `EscapeText` -- and note it spells a quote
/// `&#34;`, not `&quot;`, which upstream's `TestQwenXMLTransform` asserts
/// literally. Getting that "wrong" would still be valid XML but would not be
/// upstream.
fn escape_attr(out: &mut String, s: &str) {
    for r in s.chars() {
        match r {
            '"' => out.push_str("&#34;"),
            '\'' => out.push_str("&#39;"),
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\t' => out.push_str("&#x9;"),
            '\n' => out.push_str("&#xA;"),
            '\r' => out.push_str("&#xD;"),
            c => out.push(c),
        }
    }
}

/// Escape character data, and **only** `& < >`.
///
/// **Upstream:** `escapeTextNode`, with its comment explaining exactly why
/// `xml.EscapeText` is not used: it would mangle newlines and tabs, and a
/// parameter value is often a multi-line shell command where those matter.
fn escape_text_node(out: &mut String, s: &str) {
    for r in s.chars() {
        match r {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            c => out.push(c),
        }
    }
}

/// Read the repaired document: `<function name="..."> <parameter name="...">value
/// </parameter> ... </function>`.
///
/// **Stands in for:** Go's `xml.Unmarshal` into `XMLFunctionCall`. It is not a
/// general XML parser and does not try to be -- it only has to read documents
/// that [`transform_to_xml`] just produced, where the sole tags are `function`
/// and `parameter`, every other `<` has already become `&lt;`, and attribute
/// values are already escaped.
///
/// **What would make this wrong:** feeding it XML from anywhere else. It has no
/// namespace handling, no CDATA, no comments, no DTD. If a future family needs
/// real XML, take a dependency; do not grow this.
fn read_function_element(xml: &str) -> Result<(String, Vec<(String, String)>), ParserError> {
    let rest = xml.trim_start();
    let rest = rest
        .strip_prefix("<function")
        .ok_or_else(|| ParserError::MalformedToolCall("expected a <function> element".into()))?;
    let (name, mut rest) = read_tag_tail(rest)?;
    let name = name.unwrap_or_default();

    let mut params = Vec::new();
    loop {
        let Some(idx) = rest.find('<') else {
            // No closing `</function>`. Go's decoder calls that a syntax error and
            // so do we -- silently accepting it would hand the caller a tool call
            // the model never finished asking for.
            return Err(ParserError::MalformedToolCall(
                "unclosed <function> element".into(),
            ));
        };
        let tail = &rest[idx..];

        if let Some(after) = tail.strip_prefix("</function>") {
            let _ = after;
            return Ok((name, params));
        }

        if let Some(after) = tail.strip_prefix("<parameter") {
            let (param_name, after) = read_tag_tail(after)?;
            let end = after.find("</parameter>").ok_or_else(|| {
                ParserError::MalformedToolCall("unclosed <parameter> element".into())
            })?;
            let value = unescape(&after[..end]);
            params.push((param_name.unwrap_or_default(), value));
            rest = &after[end + "</parameter>".len()..];
            continue;
        }

        // Anything else is not a tag we know; skip past it, like a decoder
        // skipping an unrecognised element.
        let skip = tail.find('>').map(|p| p + 1).unwrap_or(tail.len());
        rest = &tail[skip..];
    }
}

/// Read the rest of an opening tag after its element name: an optional
/// `name="..."` attribute, then `>`. Returns the attribute value (unescaped) and
/// whatever follows the `>`.
fn read_tag_tail(s: &str) -> Result<(Option<String>, &str), ParserError> {
    let mut rest = s.trim_start();
    let mut name = None;

    while !rest.starts_with('>') {
        if rest.is_empty() {
            return Err(ParserError::MalformedToolCall("unterminated tag".into()));
        }
        let Some(eq) = rest.find('=') else {
            return Err(ParserError::MalformedToolCall(
                "attribute without a value".into(),
            ));
        };
        let attr = rest[..eq].trim();
        let after_eq = rest[eq + 1..].trim_start();
        let after_eq = after_eq.strip_prefix('"').ok_or_else(|| {
            ParserError::MalformedToolCall("attribute value is not quoted".into())
        })?;
        let close = after_eq
            .find('"')
            .ok_or_else(|| ParserError::MalformedToolCall("unterminated attribute".into()))?;
        if attr == "name" {
            name = Some(unescape(&after_eq[..close]));
        }
        rest = after_eq[close + 1..].trim_start();
    }

    Ok((name, &rest[1..]))
}

/// Undo [`escape_attr`] / [`escape_text_node`], plus the numeric forms a
/// conforming XML writer might have produced.
fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        let Some(semi) = tail.find(';') else {
            out.push('&');
            rest = &tail[1..];
            continue;
        };
        let entity = &tail[1..semi];
        let decoded = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            e if e.starts_with("#x") || e.starts_with("#X") => {
                u32::from_str_radix(&e[2..], 16).ok().and_then(char::from_u32)
            }
            e if e.starts_with('#') => e[1..].parse::<u32>().ok().and_then(char::from_u32),
            _ => None,
        };
        match decoded {
            Some(c) => {
                out.push(c);
                rest = &tail[semi + 1..];
            }
            None => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ToolFunction, ToolFunctionParameters, ToolProperty};
    use indexmap::IndexMap;
    use serde_json::json;

    /// Upstream's `tool()` test helper.
    fn tool(name: &str, props: &[(&str, &[&str])]) -> Tool {
        let mut properties = IndexMap::new();
        for (k, types) in props {
            properties.insert(
                (*k).to_string(),
                ToolProperty {
                    prop_type: PropertyType(types.iter().map(|s| (*s).to_string()).collect()),
                    ..Default::default()
                },
            );
        }
        Tool {
            tool_type: "function".into(),
            items: None,
            function: ToolFunction {
                name: name.into(),
                description: String::new(),
                parameters: ToolFunctionParameters {
                    param_type: "object".into(),
                    properties,
                    ..Default::default()
                },
            },
        }
    }

    fn fresh() -> Qwen3CoderParser {
        let mut p = Qwen3CoderParser::default();
        p.init(Vec::new(), None, None);
        p
    }

    /// Collect the raw event stream for one chunk, the way upstream's
    /// `TestQwenParserStreaming` does.
    fn step(p: &mut Qwen3CoderParser, s: &str) -> Vec<QwenEvent> {
        p.acc.push_str(s);
        p.parse_events()
    }

    fn content(s: &str) -> QwenEvent {
        QwenEvent::Content(s.to_string())
    }
    fn raw_call(s: &str) -> QwenEvent {
        QwenEvent::RawToolCall(s.to_string())
    }

    /// Upstream `TestQwenParserStreaming`, "simple message streamed word by word".
    #[test]
    fn a_plain_message_streams_straight_through() {
        let mut p = fresh();
        assert_eq!(step(&mut p, "hi"), vec![content("hi")]);
        assert_eq!(step(&mut p, " there"), vec![content(" there")]);
    }

    /// Upstream `TestQwenParserStreaming`, "content before tool call".
    #[test]
    fn content_before_a_tool_call_is_emitted_with_its_trailing_space_trimmed() {
        let mut p = fresh();
        assert_eq!(step(&mut p, "hi there<tool_call>"), vec![content("hi there")]);
    }

    /// Upstream `TestQwenParserStreaming`, "multiple tool calls in one message".
    #[test]
    fn several_tool_calls_and_the_content_between_them_all_come_out_in_order() {
        let mut p = fresh();
        let got = step(
            &mut p,
            "before1<tool_call>in tool call</tool_call>after1<tool_call>in tool call 2</tool_call>after2",
        );
        assert_eq!(
            got,
            vec![
                content("before1"),
                raw_call("in tool call"),
                content("after1"),
                raw_call("in tool call 2"),
                content("after2"),
            ]
        );
    }

    /// Upstream `TestQwenParserStreaming`, "tool calls with split tags".
    #[test]
    fn a_tag_split_across_chunks_is_buffered_until_it_completes() {
        let mut p = fresh();
        assert_eq!(step(&mut p, "before<tool"), vec![content("before")]);
        assert_eq!(step(&mut p, "_call>in tool call</tool"), vec![]);
        assert_eq!(
            step(&mut p, "_call>af"),
            vec![raw_call("in tool call"), content("af")]
        );
        assert_eq!(step(&mut p, "ter"), vec![content("ter")]);
    }

    /// Upstream `TestQwenParserStreaming`, "tool call tags split character by
    /// character" -- the strongest form of the streaming test.
    #[test]
    fn tags_split_one_character_at_a_time_still_parse() {
        let mut p = fresh();
        let input = "<tool_call>abc</tool_call>";
        let mut all = Vec::new();
        for ch in input.chars() {
            all.extend(step(&mut p, &ch.to_string()));
        }
        assert_eq!(all, vec![raw_call("abc")]);
        // Not one byte of the tags leaked into content.
        assert!(
            !all.iter().any(|e| matches!(e, QwenEvent::Content(_))),
            "no content should have escaped: {all:?}"
        );
    }

    /// Upstream `TestQwenToolParser`, "simple tool call".
    #[test]
    fn the_xml_ish_body_becomes_a_named_call_with_string_arguments() {
        let raw = "<function=get_current_temperature>\n<parameter=location>\nSan Francisco\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function>";
        let call = parse_tool_call(raw, &[]).expect("parse");
        assert_eq!(call.function.name, "get_current_temperature");
        assert_eq!(
            call.function.arguments.get("location"),
            Some(&json!("San Francisco"))
        );
        assert_eq!(call.function.arguments.get("unit"), Some(&json!("celsius")));
    }

    /// Upstream `TestQwenToolParser`, "names with spaces".
    #[test]
    fn names_with_spaces_survive() {
        let raw = "<function=get current temperature>\n<parameter=location with spaces>\nSan Francisco\n</parameter>\n</function>";
        let call = parse_tool_call(raw, &[]).expect("parse");
        assert_eq!(call.function.name, "get current temperature");
        assert_eq!(
            call.function.arguments.get("location with spaces"),
            Some(&json!("San Francisco"))
        );
    }

    /// Upstream `TestQwenToolParser`, "names with quotes" -- documents the
    /// behaviour rather than endorsing it: the quotes end up *in* the name.
    #[test]
    fn quotes_in_a_name_are_escaped_and_come_back_as_part_of_the_name() {
        let raw = "<function=\"get current temperature\">\n<parameter=\"location with spaces\">\nSan Francisco\n</parameter>\n<parameter=\"unit with spaces\">\n\"celsius\"\n</parameter>\n</function>";
        let call = parse_tool_call(raw, &[]).expect("parse");
        assert_eq!(call.function.name, "\"get current temperature\"");
        assert_eq!(
            call.function.arguments.get("\"location with spaces\""),
            Some(&json!("San Francisco"))
        );
        assert_eq!(
            call.function.arguments.get("\"unit with spaces\""),
            Some(&json!("\"celsius\""))
        );
    }

    /// Upstream `TestQwenToolParser`, "tool call with typed parameters" -- the
    /// schema is what turns `"3.14"` into a number.
    ///
    /// The `approx_constant` allow is deliberate: `3.14` here is upstream's
    /// fixture text, not an attempt at pi. Changing it to `PI` would stop testing
    /// what upstream tests.
    #[allow(clippy::approx_constant)]
    #[test]
    fn declared_parameter_types_coerce_the_values() {
        let tools = vec![tool(
            "calculate",
            &[
                ("x", &["number"]),
                ("y", &["integer"]),
                ("enabled", &["boolean"]),
                ("items", &["array"]),
            ],
        )];
        let raw = "<function=calculate>\n<parameter=x>\n3.14\n</parameter>\n<parameter=y>\n42\n</parameter>\n<parameter=enabled>\ntrue\n</parameter>\n<parameter=items>\n[\"a\", \"b\", \"c\"]\n</parameter>\n</function>";
        let call = parse_tool_call(raw, &tools).expect("parse");
        assert_eq!(call.function.arguments.get("x"), Some(&json!(3.14)));
        assert_eq!(call.function.arguments.get("y"), Some(&json!(42)));
        assert_eq!(call.function.arguments.get("enabled"), Some(&json!(true)));
        assert_eq!(
            call.function.arguments.get("items"),
            Some(&json!(["a", "b", "c"]))
        );
    }

    /// Upstream regression test for ollama issue #12357.
    #[test]
    fn ampersands_in_a_parameter_value_survive_the_xml_round_trip() {
        let raw = "<function=exec>\n<parameter=command>\nls && echo \"done\"\n</parameter>\n</function>";
        let call = parse_tool_call(raw, &[]).expect("parse");
        assert_eq!(
            call.function.arguments.get("command"),
            Some(&json!("ls && echo \"done\""))
        );
    }

    /// Upstream `TestQwenToolParser`, "angle brackets in parameter values".
    #[test]
    fn angle_brackets_in_a_parameter_value_survive_the_xml_round_trip() {
        let raw = "<function=exec>\n<parameter=command>\nls && echo \"a > b and a < b\"\n</parameter>\n</function>";
        let call = parse_tool_call(raw, &[]).expect("parse");
        assert_eq!(
            call.function.arguments.get("command"),
            Some(&json!("ls && echo \"a > b and a < b\""))
        );
    }

    /// Upstream `TestQwenToolParser`, "unicode in function names and parameters".
    #[test]
    fn unicode_names_and_values_survive() {
        let raw = "<function=获取天气>\n<parameter=城市>\n北京\n</parameter>\n<parameter=message>\nHello! 你好! 🌟 مرحبا\n</parameter>\n</function>";
        let call = parse_tool_call(raw, &[]).expect("parse");
        assert_eq!(call.function.name, "获取天气");
        assert_eq!(call.function.arguments.get("城市"), Some(&json!("北京")));
        assert_eq!(
            call.function.arguments.get("message"),
            Some(&json!("Hello! 你好! 🌟 مرحبا"))
        );
    }

    /// Upstream `TestQwenXMLTransform`, verbatim -- this pins the repair step's
    /// output byte for byte, including Go's `&#34;` spelling for a quote.
    #[test]
    fn the_xml_repair_step_matches_upstream_byte_for_byte() {
        let cases = [
            (
                "<function=get_current_temperature>\n<parameter=location>\nSan Francisco\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function>",
                "<function name=\"get_current_temperature\">\n<parameter name=\"location\">\nSan Francisco\n</parameter>\n<parameter name=\"unit\">\ncelsius\n</parameter>\n</function>",
            ),
            (
                "<function=\"get current temperature\">\n<parameter=\"location with spaces\">\nSan Francisco\n</parameter>\n<parameter=\"unit with spaces\">\ncelsius\n</parameter>\n</function>",
                "<function name=\"&#34;get current temperature&#34;\">\n<parameter name=\"&#34;location with spaces&#34;\">\nSan Francisco\n</parameter>\n<parameter name=\"&#34;unit with spaces&#34;\">\ncelsius\n</parameter>\n</function>",
            ),
            (
                "<function=get_current_temperature>\n\t\t<parameter=location>\n\t\tSan Francisco & San Jose\n\t\t</parameter>\n\t\t</function>",
                "<function name=\"get_current_temperature\">\n\t\t<parameter name=\"location\">\n\t\tSan Francisco &amp; San Jose\n\t\t</parameter>\n\t\t</function>",
            ),
        ];
        for (raw, want) in cases {
            assert_eq!(transform_to_xml(raw), want, "for input {raw:?}");
        }
    }

    /// Upstream `TestQwen3CoderParserToolCallIndexing`.
    #[test]
    fn parallel_tool_calls_are_indexed_in_emission_order() {
        let mut p = fresh();
        let got = p
            .add(
                "<tool_call><function=first><parameter=a>1</parameter></function></tool_call><tool_call><function=second><parameter=b>2</parameter></function></tool_call>",
                true,
            )
            .expect("add");
        assert_eq!(got.calls.len(), 2);
        assert_eq!(got.calls[0].function.name, "first");
        assert_eq!(got.calls[0].function.index, 0);
        assert_eq!(got.calls[1].function.name, "second");
        assert_eq!(got.calls[1].function.index, 1);
    }

    /// Upstream `TestQwen3CoderParserToolCallIndexResetOnInit`.
    #[test]
    fn init_resets_the_call_index() {
        let mut p = fresh();
        p.add(
            "<tool_call><function=first><parameter=a>1</parameter></function></tool_call>",
            true,
        )
        .expect("add");
        p.init(Vec::new(), None, None);
        let got = p
            .add(
                "<tool_call><function=second><parameter=b>2</parameter></function></tool_call>",
                true,
            )
            .expect("add");
        assert_eq!(got.calls[0].function.index, 0);
    }

    /// Upstream `TestQwenToolCallValueParsing`'s precedence rules, distilled.
    #[test]
    fn value_coercion_follows_upstreams_type_precedence() {
        let t = |types: &[&str]| {
            PropertyType(types.iter().map(|s| (*s).to_string()).collect())
        };

        // null beats every declared type, case-insensitively.
        assert_eq!(parse_value("null", &t(&["string"])), json!(null));
        assert_eq!(parse_value("NULL", &t(&["integer"])), json!(null));
        // No declared type at all -> string.
        assert_eq!(parse_value("123", &PropertyType::default()), json!("123"));
        // Union: the most specific type that parses wins.
        assert_eq!(parse_value("123", &t(&["string", "number"])), json!(123));
        assert_eq!(parse_value("hello", &t(&["string", "number"])), json!("hello"));
        // boolean-only and not a boolean -> false. Upstream: "matching reference".
        assert_eq!(parse_value("maybe", &t(&["boolean"])), json!(false));
        // A whole-valued float comes back as an integer.
        assert_eq!(parse_value("3.0", &t(&["number"])), json!(3));
        // integer-only that will not parse falls back to the raw string.
        assert_eq!(parse_value("12.5", &t(&["integer"])), json!("12.5"));
        // Exactly one leading and one trailing newline are stripped, no more.
        assert_eq!(parse_value("\n\nx\n\n", &PropertyType::default()), json!("\nx\n"));
        assert_eq!(
            parse_value("{\"a\":1}", &t(&["object"])),
            json!({"a": 1})
        );
    }

    /// `anyOf` flattens into one candidate list before precedence is applied.
    #[test]
    fn an_any_of_union_flattens_into_the_candidate_types() {
        let mut properties = IndexMap::new();
        properties.insert(
            "v".to_string(),
            ToolProperty {
                any_of: vec![
                    ToolProperty {
                        prop_type: PropertyType(vec!["integer".into()]),
                        ..Default::default()
                    },
                    ToolProperty {
                        prop_type: PropertyType(vec!["string".into()]),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
        );
        let tools = vec![Tool {
            tool_type: "function".into(),
            items: None,
            function: ToolFunction {
                name: "f".into(),
                description: String::new(),
                parameters: ToolFunctionParameters {
                    param_type: "object".into(),
                    properties,
                    ..Default::default()
                },
            },
        }];
        let call = parse_tool_call("<function=f><parameter=v>7</parameter></function>", &tools)
            .expect("parse");
        assert_eq!(call.function.arguments.get("v"), Some(&json!(7)));
    }

    #[test]
    fn a_body_that_is_not_a_function_element_is_an_error() {
        assert!(parse_tool_call("just some text", &[]).is_err());
    }

    #[test]
    fn an_unclosed_function_element_is_an_error() {
        assert!(parse_tool_call("<function=f><parameter=a>1</parameter>", &[]).is_err());
    }
}
