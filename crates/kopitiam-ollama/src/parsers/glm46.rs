//! GLM-4.6 response parser (and the base for GLM-4.7 and GLM-OCR).
//!
//! **Upstream:** `model/parsers/glm46.go` (ollama, MIT).
//!
//! ## The format
//!
//! Same `<think>` / `<tool_call>` tags as qwen3, but the tool-call *body* is its
//! own little XML dialect with the function name as bare text and the arguments
//! as two parallel lists:
//!
//! ```text
//! <tool_call>get_weather
//! <arg_key>location</arg_key>
//! <arg_value>Singapore</arg_value>
//! </tool_call>
//! ```
//!
//! Keys and values are paired **by position**, which is why a mismatched count is
//! a hard error -- if there are three keys and two values, there is no honest way
//! to decide which argument went missing.
//!
//! ## GLM drops tags, so there is a repair pass
//!
//! GLM models frequently omit an opening or closing `<arg_key>` / `<arg_value>`.
//! [`repair_glm46_xml`] walks the expected four-tag cycle and inserts whatever is
//! missing. It runs **only after** a strict parse has already failed, so a
//! well-formed body is never touched.
//!
//! ## The end-of-stream rule that stops a truncated call from firing
//!
//! If the stream ends while still inside a tool call, [`Glm46Parser`] will finish
//! the call -- but under **stricter** rules than mid-stream
//! ([`validate_final_glm46_tool_call`]): the tool must be one the caller actually
//! declared, every required argument must be present, and no argument name may be
//! empty. Upstream's reason, worth repeating: *repairing a truncated argument
//! could turn partial model output into a mutating tool call*. Deleting the wrong
//! file because the model got cut off mid-token is not a hypothetical.

use crate::api::{Message, PropertyType, ThinkValue, Tool, ToolCall, ToolCallArguments, ToolCallFunction};

use super::qwen3coder::parse_value;
use super::{Parsed, Parser, ParserError, chop, emit_unambiguous, overlap};

/// **Upstream:** the `glm46*Tag` consts.
pub(super) const THINKING_OPEN_TAG: &str = "<think>";
pub(super) const THINKING_CLOSE_TAG: &str = "</think>";
pub(super) const TOOL_OPEN_TAG: &str = "<tool_call>";
pub(super) const TOOL_CLOSE_TAG: &str = "</tool_call>";
const ARG_KEY_OPEN_TAG: &str = "<arg_key>";
const ARG_KEY_CLOSE_TAG: &str = "</arg_key>";
const ARG_VALUE_OPEN_TAG: &str = "<arg_value>";
const ARG_VALUE_CLOSE_TAG: &str = "</arg_value>";

/// **Upstream:** `glm46ParserState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum State {
    #[default]
    LookingForThinkingOpen,
    ThinkingStartedEatingWhitespace,
    CollectingThinking,
    ThinkingDoneEatingWhitespace,
    CollectingContent,
    ToolStartedEatingWhitespace,
    CollectingToolContent,
}

#[derive(Debug, Clone, PartialEq)]
enum Event {
    Content(String),
    Thinking(String),
    RawToolCall(String),
}

/// **Upstream:** `GLM46Parser`.
#[derive(Debug, Default)]
pub struct Glm46Parser {
    pub(super) state: State,
    buffer: String,
    pub(super) tools: Vec<Tool>,
    pub(super) call_index: usize,
}

impl Glm46Parser {
    fn eat_leading_whitespace_and_transition_to(&mut self, next: State) -> (Vec<Event>, bool) {
        let trimmed = self.buffer.trim_start().to_string();
        self.buffer.clear();
        if trimmed.is_empty() {
            return (Vec::new(), false);
        }
        self.state = next;
        self.buffer.push_str(&trimmed);
        (Vec::new(), true)
    }

    /// **Upstream:** `glm46SplitAtTag`. Same shape as the shared
    /// [`super::split_at_tag`], and it is only kept separate because upstream
    /// kept it separate.
    fn split_at_tag(&mut self, tag: &str, trim_after: bool) -> (String, String) {
        super::split_at_tag(&mut self.buffer, tag, trim_after)
    }

    /// **Upstream:** `(*GLM46Parser).eat`.
    fn eat(&mut self) -> (Vec<Event>, bool) {
        let mut events = Vec::new();

        match self.state {
            State::LookingForThinkingOpen => {
                let trimmed = self.buffer.trim_start().to_string();
                if let Some(after) = trimmed.strip_prefix(THINKING_OPEN_TAG) {
                    let after = after.trim_start().to_string();
                    self.buffer.clear();
                    self.buffer.push_str(&after);
                    self.state = if after.is_empty() {
                        State::ThinkingStartedEatingWhitespace
                    } else {
                        State::CollectingThinking
                    };
                    return (events, true);
                }
                if THINKING_OPEN_TAG.starts_with(&trimmed) {
                    // Includes the empty case: still could become `<think>`.
                    return (events, false);
                }
                // No thinking tag. Note the buffer is NOT rewritten to `trimmed`
                // -- the original leading whitespace is content and must survive.
                self.state = State::CollectingContent;
                (events, true)
            }

            State::ThinkingStartedEatingWhitespace => {
                self.eat_leading_whitespace_and_transition_to(State::CollectingThinking)
            }

            State::CollectingThinking => {
                let acc = self.buffer.clone();
                if acc.contains(THINKING_CLOSE_TAG) {
                    let (thinking, remaining) = self.split_at_tag(THINKING_CLOSE_TAG, true);
                    if !thinking.is_empty() {
                        events.push(Event::Thinking(thinking));
                    }
                    self.state = if remaining.is_empty() {
                        State::ThinkingDoneEatingWhitespace
                    } else {
                        State::CollectingContent
                    };
                    return (events, true);
                }
                // Only `</think>` is guarded here -- unlike qwen3, GLM does not
                // treat a `<tool_call>` inside thinking as an early close.
                let overlap_len = overlap(&acc, THINKING_CLOSE_TAG);
                let unambiguous = emit_unambiguous(&mut self.buffer, overlap_len);
                if !unambiguous.is_empty() {
                    events.push(Event::Thinking(unambiguous));
                }
                (events, false)
            }

            State::ThinkingDoneEatingWhitespace => {
                self.eat_leading_whitespace_and_transition_to(State::CollectingContent)
            }

            State::CollectingContent => {
                let acc = self.buffer.clone();
                if acc.contains(TOOL_OPEN_TAG) {
                    let (before, after) = self.split_at_tag(TOOL_OPEN_TAG, true);
                    if !before.is_empty() {
                        events.push(Event::Content(before));
                    }
                    self.state = if after.is_empty() {
                        State::ToolStartedEatingWhitespace
                    } else {
                        State::CollectingToolContent
                    };
                    return (events, true);
                }
                let overlap_len = overlap(&acc, TOOL_OPEN_TAG);
                let unambiguous = emit_unambiguous(&mut self.buffer, overlap_len);
                if !unambiguous.is_empty() {
                    events.push(Event::Content(unambiguous));
                }
                (events, false)
            }

            State::ToolStartedEatingWhitespace => {
                self.eat_leading_whitespace_and_transition_to(State::CollectingToolContent)
            }

            State::CollectingToolContent => {
                // Tool calls are never streamed, so just wait for the close tag.
                if self.buffer.contains(TOOL_CLOSE_TAG) {
                    let (tool_content, _) = self.split_at_tag(TOOL_CLOSE_TAG, true);
                    events.push(Event::RawToolCall(tool_content));
                    self.state = State::CollectingContent;
                    return (events, true);
                }
                (events, false)
            }
        }
    }

    fn parse_events(&mut self) -> Vec<Event> {
        let mut all = Vec::new();
        let mut keep_looping = true;
        while keep_looping {
            let (events, again) = self.eat();
            keep_looping = again;
            all.extend(events);
        }
        all
    }

    /// The stream ended mid-tool-call. **Upstream:** `finalizeToolCall`.
    ///
    /// Any partial `</tool_call>` at the tail is dropped first (it is framing the
    /// model never finished), then the body is validated under the strict
    /// end-of-stream rules. A failure here is a **hard error** for the whole
    /// `add` -- see the module docs for why we would rather fail than guess.
    fn finalize_tool_call(&mut self) -> Result<Event, ParserError> {
        let mut raw = self.buffer.clone();
        let overlap_len = overlap(&raw, TOOL_CLOSE_TAG);
        if overlap_len > 0 {
            let (before, _) = chop(&raw, raw.len() - overlap_len);
            raw = before.trim_end().to_string();
        }

        let parsed = read_glm_tool_call_xml(&escape_glm46_content(&raw))?;
        validate_final_glm46_tool_call(&parsed, &self.tools)?;

        self.buffer.clear();
        self.state = State::CollectingContent;
        Ok(Event::RawToolCall(raw))
    }

    /// Shared by `Glm46Parser`, `Glm47Parser` and `GlmOcrParser` -- the three
    /// differ only in `init`.
    pub(super) fn add_inner(&mut self, s: &str, done: bool) -> Result<Parsed, ParserError> {
        self.buffer.push_str(s);
        let mut events = self.parse_events();

        if done
            && matches!(
                self.state,
                State::ToolStartedEatingWhitespace | State::CollectingToolContent
            )
        {
            events.push(self.finalize_tool_call().map_err(|e| {
                ParserError::MalformedToolCall(format!("incomplete GLM tool call: {e}"))
            })?);
        }

        let mut out = Parsed::default();
        for event in events {
            match event {
                Event::RawToolCall(raw) => {
                    let mut call = parse_glm46_tool_call(&raw, &self.tools)?;
                    call.function.index = self.call_index;
                    self.call_index += 1;
                    out.calls.push(call);
                }
                Event::Thinking(t) => out.thinking.push_str(&t),
                Event::Content(c) => out.content.push_str(&c),
            }
        }
        Ok(out)
    }

    pub(super) fn preserved(&self) -> Vec<&'static str> {
        vec![
            THINKING_OPEN_TAG,
            THINKING_CLOSE_TAG,
            TOOL_OPEN_TAG,
            TOOL_CLOSE_TAG,
            ARG_KEY_OPEN_TAG,
            ARG_KEY_CLOSE_TAG,
            ARG_VALUE_OPEN_TAG,
            ARG_VALUE_CLOSE_TAG,
        ]
    }
}

impl Parser for Glm46Parser {
    fn init(
        &mut self,
        tools: Vec<Tool>,
        _last_message: Option<&Message>,
        _think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        // Upstream ignores both `lastMessage` and `thinkValue` here: GLM-4.6's
        // output carries its own `<think>` opening tag, so the machine starts in
        // `LookingForThinkingOpen` and finds out for itself.
        self.tools = tools.clone();
        self.call_index = 0;
        tools
    }

    fn add(&mut self, s: &str, done: bool) -> Result<Parsed, ParserError> {
        self.add_inner(s, done)
    }

    fn preserved_tokens(&self) -> Vec<&'static str> {
        self.preserved()
    }

    fn has_tool_support(&self) -> bool {
        true
    }

    fn has_thinking_support(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// The tool-call body
// ---------------------------------------------------------------------------

/// The shape upstream unmarshals into. **Upstream:** `GLMToolCallXML`.
#[derive(Debug, Default, PartialEq)]
pub(super) struct GlmToolCallXml {
    /// Character data directly inside `<tool_call>` -- i.e. the function name.
    pub content: String,
    /// Every `<arg_key>` in document order.
    pub keys: Vec<String>,
    /// Every `<arg_value>` in document order.
    pub values: Vec<String>,
}

/// **Upstream:** `parseGLM46ToolCall`.
fn parse_glm46_tool_call(raw: &str, tools: &[Tool]) -> Result<ToolCall, ParserError> {
    let escaped = escape_glm46_content(raw);

    // Strict first; repair only if that fails, so a well-formed body is never
    // rewritten.
    let parsed = match read_glm_tool_call_xml(&escaped) {
        Ok(p) => p,
        Err(first) => read_glm_tool_call_xml(&repair_glm46_xml(&escaped)).map_err(|_| first)?,
    };

    let function_name = parsed.content.trim().to_string();
    if function_name.is_empty() {
        return Err(ParserError::EmptyFunctionName);
    }
    if parsed.keys.len() != parsed.values.len() {
        return Err(ParserError::MalformedToolCall(format!(
            "mismatched arg_key and arg_value counts: {} keys, {} values",
            parsed.keys.len(),
            parsed.values.len()
        )));
    }

    let matched = tools.iter().find(|t| t.function.name == function_name);

    let mut arguments = ToolCallArguments::new();
    for (raw_key, value) in parsed.keys.iter().zip(parsed.values.iter()) {
        let key = raw_key.trim();

        let mut param_type = PropertyType::default();
        if let Some(tool) = matched
            && let Some(prop) = tool.function.parameters.properties.get(key)
        {
            if prop.any_of.is_empty() {
                param_type = prop.prop_type.clone();
            } else {
                for branch in &prop.any_of {
                    param_type.0.extend(branch.prop_type.0.iter().cloned());
                }
            }
        }

        // NOT trimmed here -- `parse_value` owns the one-newline-each-side rule.
        arguments.set(key, parse_value(value, &param_type));
    }

    Ok(ToolCall {
        id: String::new(),
        function: ToolCallFunction {
            index: 0,
            name: function_name,
            arguments,
        },
    })
}

/// End-of-stream validation, deliberately stricter than the mid-stream path.
/// **Upstream:** `validateFinalGLM46ToolCall`.
///
/// At end of stream **only the outer closing tag may be missing**. Everything
/// else must already be there: a declared tool, a non-empty name, matched
/// key/value counts, no empty argument names, and every required argument
/// present. Loosening any of these lets a truncated generation fire a real,
/// possibly mutating, tool call.
fn validate_final_glm46_tool_call(
    parsed: &GlmToolCallXml,
    tools: &[Tool],
) -> Result<(), ParserError> {
    let function_name = parsed.content.trim();
    if function_name.is_empty() {
        return Err(ParserError::EmptyFunctionName);
    }
    if parsed.keys.len() != parsed.values.len() {
        return Err(ParserError::MalformedToolCall(format!(
            "mismatched arg_key and arg_value counts: {} keys, {} values",
            parsed.keys.len(),
            parsed.values.len()
        )));
    }

    let Some(declared) = tools.iter().find(|t| t.function.name == function_name) else {
        return Err(ParserError::MalformedToolCall(format!(
            "tool {function_name:?} is not declared"
        )));
    };

    let mut seen: Vec<&str> = Vec::with_capacity(parsed.keys.len());
    for raw_key in &parsed.keys {
        let key = raw_key.trim();
        if key.is_empty() {
            return Err(ParserError::MalformedToolCall("empty argument name".into()));
        }
        seen.push(key);
    }

    for required in &declared.function.parameters.required {
        if !seen.contains(&required.as_str()) {
            return Err(ParserError::MalformedToolCall(format!(
                "required argument {required:?} is missing for tool {function_name:?}"
            )));
        }
    }
    Ok(())
}

/// Escape `&`, `<`, `>` in text while leaving the four known tags alone.
/// **Upstream:** `escapeGLM46Content`.
///
/// This is what lets an argument value contain raw XML-looking text -- a code
/// snippet, an HTML fragment -- without the model having to escape anything.
/// Byte-wise like upstream: only ASCII characters are ever rewritten, so
/// multi-byte runes pass through untouched.
fn escape_glm46_content(s: &str) -> String {
    let b = s.as_bytes();
    // Bytes, not chars, so a multi-byte rune's continuation bytes pass through
    // untouched. Everything this function *inserts* is ASCII, so the result is
    // still valid UTF-8 -- which is why the `from_utf8` below cannot fail.
    let mut out: Vec<u8> = Vec::with_capacity(s.len());
    let mut in_tag = false;

    for (i, &ch) in b.iter().enumerate() {
        if ch == b'<' {
            let rest = &s[i..];
            if rest.starts_with(ARG_KEY_OPEN_TAG)
                || rest.starts_with(ARG_KEY_CLOSE_TAG)
                || rest.starts_with(ARG_VALUE_OPEN_TAG)
                || rest.starts_with(ARG_VALUE_CLOSE_TAG)
            {
                in_tag = true;
            }
        }

        if in_tag {
            out.push(ch);
            if ch == b'>' {
                in_tag = false;
            }
        } else {
            match ch {
                b'&' => out.extend_from_slice(b"&amp;"),
                b'<' => out.extend_from_slice(b"&lt;"),
                b'>' => out.extend_from_slice(b"&gt;"),
                _ => out.push(ch),
            }
        }
    }

    String::from_utf8(out).unwrap_or_else(|_| s.to_string())
}

/// Read the escaped body of a `<tool_call>`: bare text (the function name) plus
/// `<arg_key>` / `<arg_value>` elements.
///
/// **Stands in for:** Go's `xml.Unmarshal` into `GLMToolCallXML`. Purpose-built,
/// same reasoning as the qwen3-coder reader -- it only ever sees output of
/// [`escape_glm46_content`], where the only surviving `<` characters are the four
/// known tags.
///
/// Returning `Err` is meaningful: it is exactly what triggers
/// [`repair_glm46_xml`]. So "unclosed `<arg_key>`" must fail here rather than be
/// quietly tolerated, or the repair pass never runs.
fn read_glm_tool_call_xml(body: &str) -> Result<GlmToolCallXml, ParserError> {
    let mut out = GlmToolCallXml::default();
    let mut rest = body;

    loop {
        let Some(idx) = rest.find('<') else {
            out.content.push_str(&unescape_xml(rest));
            return Ok(out);
        };
        out.content.push_str(&unescape_xml(&rest[..idx]));
        let tail = &rest[idx..];

        let (open, close, sink): (&str, &str, &mut Vec<String>) =
            if tail.starts_with(ARG_KEY_OPEN_TAG) {
                (ARG_KEY_OPEN_TAG, ARG_KEY_CLOSE_TAG, &mut out.keys)
            } else if tail.starts_with(ARG_VALUE_OPEN_TAG) {
                (ARG_VALUE_OPEN_TAG, ARG_VALUE_CLOSE_TAG, &mut out.values)
            } else {
                return Err(ParserError::MalformedToolCall(format!(
                    "unexpected tag at {:?}",
                    &tail[..tail.len().min(16)]
                )));
            };

        let inner = &tail[open.len()..];
        let Some(end) = inner.find(close) else {
            return Err(ParserError::MalformedToolCall(format!(
                "unclosed {open} element"
            )));
        };
        // A nested opener before the closer means the model dropped a tag --
        // fail, so the repair pass gets its turn.
        let text = &inner[..end];
        if text.contains('<') {
            return Err(ParserError::MalformedToolCall(format!(
                "nested tag inside {open}"
            )));
        }
        sink.push(unescape_xml(text));
        rest = &inner[end + close.len()..];
    }
}

/// Undo [`escape_glm46_content`].
fn unescape_xml(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

/// Which tag the repair walker is expecting next.
/// **Upstream:** `repairPhase` (`phaseArgKeyOpen`, `phaseArgKeyClose`, ...).
///
/// The shared `Arg` prefix mirrors upstream's names so the two can be read side
/// by side; the discriminants are load-bearing too -- the walker does modular
/// arithmetic on them, and "even means an opening tag" is upstream's own
/// `phase%2 == 0`.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    ArgKeyOpen = 0,
    ArgKeyClose = 1,
    ArgValOpen = 2,
    ArgValClose = 3,
}

const TAG_CYCLE: [&str; 4] = [
    ARG_KEY_OPEN_TAG,
    ARG_KEY_CLOSE_TAG,
    ARG_VALUE_OPEN_TAG,
    ARG_VALUE_CLOSE_TAG,
];

impl Phase {
    fn from_index(i: usize) -> Phase {
        match i % 4 {
            0 => Phase::ArgKeyOpen,
            1 => Phase::ArgKeyClose,
            2 => Phase::ArgValOpen,
            _ => Phase::ArgValClose,
        }
    }
    fn index(self) -> usize {
        self as usize
    }
    fn next(self) -> Phase {
        Phase::from_index(self.index() + 1)
    }
    /// Even phases are opening tags.
    fn is_open(self) -> bool {
        self.index().is_multiple_of(2)
    }
    fn tag(self) -> &'static str {
        TAG_CYCLE[self.index()]
    }
}

/// Rebuild well-formed XML from GLM output that dropped tags.
/// **Upstream:** `repairGLM46XML`.
///
/// The expected body is a function name followed by a repeating cycle of
/// `<arg_key> key </arg_key> <arg_value> value </arg_value>`. The walker looks for
/// whichever tag it expects next; when a *different* known tag turns up first, it
/// emits the tags that were skipped and carries on.
///
/// The awkward first block handles a specific real failure: when the very first
/// tag found is not `<arg_key>` (say the model wrote `weather city</arg_key>`),
/// the text in front of it holds **both** the function name and the first key.
/// Function names cannot contain whitespace, so the split goes at the first
/// space. That heuristic is upstream's; it is the only place here that guesses.
fn repair_glm46_xml(s: &str) -> String {
    fn find_next_tag(s: &str) -> Option<(usize, &'static str)> {
        TAG_CYCLE
            .iter()
            .filter_map(|t| s.find(t).map(|i| (i, *t)))
            .min_by_key(|(i, _)| *i)
    }
    fn tag_phase(tag: &str) -> Phase {
        Phase::from_index(TAG_CYCLE.iter().position(|t| *t == tag).unwrap_or(0))
    }

    let mut result = String::with_capacity(s.len() + 32);

    let Some((idx, first_tag)) = find_next_tag(s) else {
        return s.to_string();
    };
    let prefix = &s[..idx];
    let mut s = &s[idx..];

    let mut phase = Phase::ArgKeyOpen;
    if first_tag != ARG_KEY_OPEN_TAG {
        if let Some(sp) = prefix.find(char::is_whitespace) {
            result.push_str(&prefix[..sp]);
            result.push_str(ARG_KEY_OPEN_TAG);
            result.push_str(prefix[sp..].trim_start());
            phase = Phase::ArgKeyClose;
        } else {
            result.push_str(prefix);
        }
    } else {
        result.push_str(prefix);
    }

    while !s.is_empty() {
        let hit = find_next_tag(s);
        let expected = phase.tag();
        let is_open = phase.is_open();

        let Some((mut idx, found)) = hit else {
            if is_open {
                // Expecting an opening tag and nothing is left -- done.
                break;
            }
            result.push_str(s);
            result.push_str(expected);
            phase = phase.next();
            break;
        };

        if found == expected {
            result.push_str(&s[..idx]);
            result.push_str(expected);
            s = &s[idx + expected.len()..];
            phase = phase.next();
            continue;
        }

        let found_phase = tag_phase(found);

        if is_open && idx > 0 {
            // Text sitting where an opening tag should be: the opening tag was
            // dropped. Emit it, then the text, then re-evaluate expecting the
            // matching closing tag.
            result.push_str(expected);
            result.push_str(&s[..idx]);
            phase = phase.next();
            s = &s[idx..];
            continue;
        }

        // Emit the tags that were skipped, until we are back in step.
        while phase != found_phase {
            let tag = phase.tag();
            if phase.is_open() {
                result.push_str(tag);
            } else {
                // One step short of the found tag: the text in front of it is
                // this element's content, so emit it before closing.
                if phase.next() == found_phase && idx > 0 {
                    result.push_str(&s[..idx]);
                    s = &s[idx..];
                    idx = 0;
                }
                result.push_str(tag);
            }
            phase = phase.next();
        }
        // `phase == found_phase` now; loop round and consume it normally.
    }

    // Stopped mid-pair: close whatever is dangling, and pad out the argument so
    // key and value counts still match.
    match phase {
        Phase::ArgKeyClose => {
            result.push_str(ARG_KEY_CLOSE_TAG);
            result.push_str(ARG_VALUE_OPEN_TAG);
            result.push_str(ARG_VALUE_CLOSE_TAG);
        }
        Phase::ArgValOpen => {
            result.push_str(ARG_VALUE_OPEN_TAG);
            result.push_str(ARG_VALUE_CLOSE_TAG);
        }
        Phase::ArgValClose => result.push_str(ARG_VALUE_CLOSE_TAG),
        Phase::ArgKeyOpen => {}
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ToolFunction, ToolFunctionParameters, ToolProperty};
    use indexmap::IndexMap;
    use serde_json::json;

    fn tool(name: &str, props: &[(&str, &str)], required: &[&str]) -> Tool {
        let mut properties = IndexMap::new();
        for (k, t) in props {
            properties.insert(
                (*k).to_string(),
                ToolProperty {
                    prop_type: PropertyType(vec![(*t).to_string()]),
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
                    required: required.iter().map(|s| (*s).to_string()).collect(),
                    ..Default::default()
                },
            },
        }
    }

    fn fresh() -> Glm46Parser {
        let mut p = Glm46Parser::default();
        p.init(Vec::new(), None, None);
        p
    }

    #[test]
    fn thinking_is_recognised_from_its_own_opening_tag() {
        let mut p = fresh();
        let got = p.add("<think>weighing it up</think>the answer", true).expect("add");
        assert_eq!(got.thinking, "weighing it up");
        assert_eq!(got.content, "the answer");
    }

    /// No `<think>` at all: everything is content, and the leading whitespace is
    /// NOT eaten -- upstream's "don't trim, we want to keep the original content".
    #[test]
    fn without_a_thinking_tag_the_original_leading_whitespace_survives() {
        let mut p = fresh();
        let got = p.add("  hello there", true).expect("add");
        assert_eq!(got.content, "  hello there");
        assert!(got.thinking.is_empty());
    }

    #[test]
    fn a_tool_call_pairs_keys_and_values_by_position() {
        let mut p = fresh();
        let got = p
            .add(
                "<tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>Singapore</arg_value>\n<arg_key>unit</arg_key>\n<arg_value>celsius</arg_value>\n</tool_call>",
                true,
            )
            .expect("add");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "get_weather");
        assert_eq!(
            got.calls[0].function.arguments.get("location"),
            Some(&json!("Singapore"))
        );
        assert_eq!(
            got.calls[0].function.arguments.get("unit"),
            Some(&json!("celsius"))
        );
    }

    /// The tool schema is what turns text into a number, same as qwen3-coder --
    /// GLM shares `parse_value` with it.
    #[test]
    fn declared_parameter_types_coerce_the_values() {
        let tools = vec![tool("calc", &[("x", "integer")], &[])];
        let mut p = Glm46Parser::default();
        p.init(tools, None, None);
        let got = p
            .add(
                "<tool_call>calc<arg_key>x</arg_key><arg_value>42</arg_value></tool_call>",
                true,
            )
            .expect("add");
        assert_eq!(got.calls[0].function.arguments.get("x"), Some(&json!(42)));
    }

    /// Raw XML-looking text inside a value survives, because
    /// `escape_glm46_content` escapes everything that is not one of the four
    /// known tags.
    #[test]
    fn xml_looking_text_inside_an_argument_value_survives() {
        let mut p = fresh();
        let got = p
            .add(
                "<tool_call>write<arg_key>body</arg_key><arg_value><p>hi & bye</p></arg_value></tool_call>",
                true,
            )
            .expect("add");
        assert_eq!(
            got.calls[0].function.arguments.get("body"),
            Some(&json!("<p>hi & bye</p>"))
        );
    }

    /// Mismatched counts are a hard error -- there is no honest pairing.
    #[test]
    fn mismatched_key_and_value_counts_are_an_error() {
        let mut p = fresh();
        assert!(
            p.add(
                "<tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value><arg_key>b</arg_key></tool_call>",
                true,
            )
            .is_err()
        );
    }

    /// Upstream's repair pass: a dropped `<arg_key>` opener is reinserted.
    #[test]
    fn a_dropped_arg_key_opening_tag_is_repaired() {
        let mut p = fresh();
        let got = p
            .add(
                "<tool_call>get_weather city</arg_key><arg_value>Singapore</arg_value></tool_call>",
                true,
            )
            .expect("add");
        assert_eq!(got.calls[0].function.name, "get_weather");
        assert_eq!(
            got.calls[0].function.arguments.get("city"),
            Some(&json!("Singapore"))
        );
    }

    /// A dropped `</arg_value>` at the end is closed off by the repair walker.
    #[test]
    fn a_dropped_final_closing_tag_is_repaired() {
        let mut p = fresh();
        let got = p
            .add(
                "<tool_call>f<arg_key>a</arg_key><arg_value>1</tool_call>",
                true,
            )
            .expect("add");
        assert_eq!(got.calls[0].function.arguments.get("a"), Some(&json!("1")));
    }

    /// A well-formed body must NOT be touched by the repair pass -- it never runs.
    #[test]
    fn a_well_formed_body_is_never_rewritten() {
        let good = "f<arg_key>a</arg_key><arg_value>1</arg_value>";
        let parsed = read_glm_tool_call_xml(good).expect("strict parse should succeed");
        assert_eq!(parsed.content.trim(), "f");
        assert_eq!(parsed.keys, vec!["a"]);
        assert_eq!(parsed.values, vec!["1"]);
    }

    /// End-of-stream: only the outer `</tool_call>` may be missing, and the tool
    /// must be declared. This is the guard that stops a truncated generation from
    /// firing a real, possibly mutating, call.
    #[test]
    fn a_truncated_tool_call_only_fires_for_a_declared_tool() {
        // Undeclared -> refused.
        let mut p = fresh();
        assert!(
            p.add(
                "<tool_call>rm_rf<arg_key>path</arg_key><arg_value>/</arg_value>",
                true
            )
            .is_err()
        );

        // Declared, all required args present -> allowed.
        let tools = vec![tool("get_weather", &[("city", "string")], &["city"])];
        let mut p = Glm46Parser::default();
        p.init(tools, None, None);
        let got = p
            .add(
                "<tool_call>get_weather<arg_key>city</arg_key><arg_value>SG</arg_value>",
                true,
            )
            .expect("add");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.arguments.get("city"), Some(&json!("SG")));
    }

    /// ...and a declared tool missing a REQUIRED argument is still refused.
    #[test]
    fn a_truncated_tool_call_missing_a_required_argument_is_refused() {
        let tools = vec![tool("send", &[("to", "string"), ("body", "string")], &["to", "body"])];
        let mut p = Glm46Parser::default();
        p.init(tools, None, None);
        assert!(
            p.add("<tool_call>send<arg_key>to</arg_key><arg_value>bob</arg_value>", true)
                .is_err()
        );
    }

    #[test]
    fn a_thinking_close_tag_split_across_chunks_never_leaks() {
        let mut p = fresh();
        let a = p.add("<think>thought</thi", false).expect("add");
        assert_eq!(a.thinking, "thought");
        let b = p.add("nk>visible", true).expect("add");
        assert_eq!(b.content, "visible");
    }

    #[test]
    fn feeding_one_byte_at_a_time_gives_the_same_answer_as_one_big_chunk() {
        let input = "<think>reasoning</think>Hi <tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>bye";

        let mut whole = fresh();
        let want = whole.add(input, true).expect("add");

        let mut streamed = fresh();
        let mut got = Parsed::default();
        for (i, ch) in input.char_indices() {
            let part = streamed
                .add(&input[i..i + ch.len_utf8()], i + ch.len_utf8() == input.len())
                .expect("add");
            got.content.push_str(&part.content);
            got.thinking.push_str(&part.thinking);
            got.calls.extend(part.calls);
        }

        assert_eq!(got.thinking, want.thinking);
        assert_eq!(got.content, want.content);
        assert_eq!(got.calls.len(), want.calls.len());
    }

    #[test]
    fn preserved_tokens_cover_the_arg_tags_too() {
        let p = Glm46Parser::default();
        for t in [
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<arg_key>",
            "</arg_key>",
            "<arg_value>",
            "</arg_value>",
        ] {
            assert!(p.preserved_tokens().contains(&t), "missing {t}");
        }
    }
}
