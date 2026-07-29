//! Gemma 4 response parser.
//!
//! **Upstream:** `model/parsers/gemma4.go` (ollama, MIT).
//!
//! ## Gemma 4's format, in one look
//!
//! ```text
//! <|channel>thought
//! I should check the weather<channel|><|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|>
//! ```
//!
//! Three things to notice, all of them traps:
//!
//! 1. **The tags are asymmetric.** Thinking opens `<|channel>` and closes
//!    `<channel|>` -- the bar moves to the other side, it is not `</channel>`.
//!    Same for `<|tool_call>` / `<tool_call|>`.
//! 2. **`<|channel>` is followed by a channel NAME**, normally `thought\n`, which
//!    is framing and must be stripped -- and in a stream `thought` and `\n` can
//!    land in different chunks.
//! 3. **The tool-call arguments are not JSON.** Keys are bare, and strings are
//!    wrapped in `<|"|>` ... `<|"|>` instead of double quotes. That last bit is
//!    the whole reason for [`gemma4_args_to_json`]: it exists so a value can
//!    contain raw `"` characters (shell commands, regexes, Windows paths) without
//!    escaping, which is exactly what the model does.
//!
//! ## Why there is a whole repair department down the bottom
//!
//! Gemma 4 truncates. It drops the closing `<|"|>`, forgets the final `}`, and
//! occasionally writes `'single quotes'` instead of the delimiter. Upstream's
//! answer (ollama issue #15315) is a **small, ordered set of candidate repairs**,
//! each of which must still survive the normal conversion and a real JSON parse
//! before it is accepted. The important guardrail: wrapping a bare value in
//! string delimiters is **schema-gated** -- only done when the tool declares that
//! argument as a string. Raw text is otherwise too ambiguous to guess at, and a
//! wrong guess silently changes what the tool is asked to do.

use crate::api::{Message, ThinkValue, Tool, ToolCall, ToolCallArguments, ToolCallFunction, ToolProperty};
use indexmap::IndexMap;

use super::{Parsed, Parser, ParserError, chop, emit_unambiguous, overlap};

/// **Upstream:** the `gemma4*Tag` consts. Note the asymmetry -- the closing tags
/// put the bar on the *right*. `</channel>` and `</tool_call>` do not exist in
/// this family and would never match.
const THINKING_OPEN_TAG: &str = "<|channel>";
const THINKING_CLOSE_TAG: &str = "<channel|>";
const TOOL_CALL_OPEN_TAG: &str = "<|tool_call>";
const TOOL_CALL_CLOSE_TAG: &str = "<tool_call|>";
const TOOL_RESPONSE_TAG: &str = "<|tool_response>";
/// The string delimiter. Both the opener and the closer are this same literal --
/// which is why [`gemma4_args_to_json`] pairs them off left to right, and why an
/// odd count means the model truncated mid-string.
const STRING_DELIMITER: &str = "<|\"|>";
/// The channel name Gemma emits before its reasoning. Stripped as framing.
const THOUGHT_CHANNEL_PREFIX: &str = "thought\n";

/// **Upstream:** `Gemma4ParserState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    #[default]
    CollectingContent,
    CollectingThinking,
    CollectingToolCall,
    /// Straight after a tool call, where stray `<tool_call|>` / `<|tool_response>`
    /// markers get swallowed. See [`Gemma4Parser::eat`] for the trade-off.
    IgnoringPostToolCallNoise,
}

#[derive(Debug, Clone, PartialEq)]
enum Event {
    Thinking(String),
    Content(String),
    ToolCall(Box<ToolCall>),
}

/// **Upstream:** `Gemma4Parser`.
#[derive(Debug, Default)]
pub struct Gemma4Parser {
    state: State,
    buffer: String,
    tools: Vec<Tool>,
    call_index: usize,
    has_thinking_support: bool,
    /// Both the model supports thinking AND the caller asked for it.
    thinking_enabled: bool,
    /// Just entered thinking; the `thought\n` channel name still needs stripping.
    needs_channel_name_strip: bool,
}

impl Gemma4Parser {
    /// `"gemma4"` is `new(true)`; `"gemma4-no-thinking"` is `new(false)`.
    pub fn new(has_thinking_support: bool) -> Self {
        Self {
            has_thinking_support,
            ..Default::default()
        }
    }

    /// **Upstream:** `(*Gemma4Parser).eat`.
    ///
    /// `done` matters here in a way it does not for the qwen families: on the
    /// last chunk there is no "wait for more" option, so the ambiguity checks are
    /// skipped and everything held back gets flushed. Without that, a reply
    /// ending in whitespace would lose it, and a tool call the model never closed
    /// would be dropped entirely.
    fn eat(&mut self, done: bool) -> (Vec<Event>, bool) {
        let mut events = Vec::new();
        if self.buffer.is_empty() {
            return (events, false);
        }
        let buf = self.buffer.clone();

        match self.state {
            State::CollectingContent => {
                if let Some(idx) = buf.find(THINKING_OPEN_TAG) {
                    let (before, rest) = chop(&buf, idx);
                    self.buffer = rest[THINKING_OPEN_TAG.len()..].to_string();
                    self.state = State::CollectingThinking;
                    self.needs_channel_name_strip = true;
                    let before = before.trim_end();
                    if !before.is_empty() {
                        events.push(Event::Content(before.to_string()));
                    }
                    return (events, true);
                }

                if let Some(idx) = buf.find(TOOL_CALL_OPEN_TAG) {
                    let (before, rest) = chop(&buf, idx);
                    self.buffer = rest[TOOL_CALL_OPEN_TAG.len()..].to_string();
                    self.state = State::CollectingToolCall;
                    let before = before.trim_end();
                    if !before.is_empty() {
                        events.push(Event::Content(before.to_string()));
                    }
                    return (events, true);
                }

                if !done {
                    let overlap_len =
                        longest_overlap(&buf, &[THINKING_OPEN_TAG, TOOL_CALL_OPEN_TAG]);
                    if overlap_len > 0 {
                        let unambiguous = emit_unambiguous(&mut self.buffer, overlap_len);
                        if !unambiguous.is_empty() {
                            events.push(Event::Content(unambiguous));
                        }
                        return (events, false);
                    }
                }

                self.buffer.clear();
                if !buf.is_empty() {
                    events.push(Event::Content(buf));
                }
                (events, false)
            }

            State::CollectingThinking => {
                let mut buf = buf;

                // The channel name. `thought` and `\n` can arrive in separate
                // chunks, so a prefix of `thought\n` means "wait", not "no match".
                if self.needs_channel_name_strip {
                    if let Some(after) = buf.strip_prefix(THOUGHT_CHANNEL_PREFIX) {
                        buf = after.to_string();
                        self.buffer = buf.clone();
                        self.needs_channel_name_strip = false;
                    } else if !done && THOUGHT_CHANNEL_PREFIX.starts_with(buf.as_str()) {
                        return (events, false);
                    } else {
                        // A different channel name, or no newline. Leave it alone.
                        self.needs_channel_name_strip = false;
                    }
                }

                if let Some(idx) = buf.find(THINKING_CLOSE_TAG) {
                    let (thinking, rest) = chop(&buf, idx);
                    let thinking = thinking.trim_end().to_string();
                    self.buffer = rest[THINKING_CLOSE_TAG.len()..].trim_start().to_string();
                    self.state = State::CollectingContent;
                    if !thinking.is_empty() {
                        events.push(Event::Thinking(thinking));
                    }
                    return (events, true);
                }

                if !done {
                    let overlap_len = overlap(&buf, THINKING_CLOSE_TAG);
                    self.buffer = buf;
                    let unambiguous = emit_unambiguous(&mut self.buffer, overlap_len);
                    if !unambiguous.is_empty() {
                        events.push(Event::Thinking(unambiguous));
                    }
                    return (events, false);
                }

                // Last chunk: nothing left to wait for, flush it all.
                self.buffer.clear();
                if !buf.is_empty() {
                    events.push(Event::Thinking(buf));
                }
                (events, false)
            }

            State::CollectingToolCall => {
                if let Some(idx) = buf.find(TOOL_CALL_CLOSE_TAG) {
                    let (body, rest) = chop(&buf, idx);
                    self.buffer = rest[TOOL_CALL_CLOSE_TAG.len()..].trim_start().to_string();
                    self.state = State::IgnoringPostToolCallNoise;
                    // A body that will not parse is dropped with a warning
                    // upstream, not raised -- one bad call must not kill the turn.
                    if let Ok(call) = parse_gemma4_tool_call(body, &self.tools) {
                        events.push(Event::ToolCall(Box::new(call)));
                    }
                    return (events, true);
                }

                // The model can hit a stop token before `<tool_call|>`. On the
                // last chunk, try to parse what we have rather than lose the call.
                if done && !buf.is_empty() {
                    self.buffer.clear();
                    self.state = State::CollectingContent;
                    if let Ok(call) = parse_gemma4_tool_call(&buf, &self.tools) {
                        events.push(Event::ToolCall(Box::new(call)));
                    }
                    return (events, false);
                }

                (events, false)
            }

            State::IgnoringPostToolCallNoise => {
                // Gemma 4 sometimes emits extra `<tool_call|>` tags after a valid
                // call, and the newer template uses `<|tool_response>` as a
                // post-call boundary. Both get swallowed here.
                //
                // **The trade-off, stated:** if the model genuinely means to start
                // its next reply with one of those literal strings, we drop it.
                // Upstream took that deal; leaking control tags into assistant
                // content is the worse failure.
                let mut buf = buf.trim_start().to_string();
                loop {
                    if let Some(after) = buf.strip_prefix(TOOL_CALL_CLOSE_TAG) {
                        buf = after.trim_start().to_string();
                    } else if let Some(after) = buf.strip_prefix(TOOL_RESPONSE_TAG) {
                        buf = after.trim_start().to_string();
                    } else {
                        break;
                    }
                }
                self.buffer = buf.clone();

                if buf.is_empty() {
                    return (events, false);
                }

                // Still could grow INTO one of those markers -- keep waiting.
                if TOOL_CALL_CLOSE_TAG.starts_with(buf.as_str())
                    || TOOL_RESPONSE_TAG.starts_with(buf.as_str())
                {
                    if done {
                        self.buffer.clear();
                        self.state = State::CollectingContent;
                    }
                    return (events, false);
                }

                self.state = State::CollectingContent;
                (events, true)
            }
        }
    }

    fn parse_events(&mut self, done: bool) -> Vec<Event> {
        let mut all = Vec::new();
        let mut keep_looping = true;
        while keep_looping {
            let (events, again) = self.eat(done);
            keep_looping = again;
            all.extend(events);
        }
        all
    }
}

impl Parser for Gemma4Parser {
    fn init(
        &mut self,
        tools: Vec<Tool>,
        last_message: Option<&Message>,
        think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        self.tools = tools.clone();
        self.call_index = 0;

        self.thinking_enabled = self.has_thinking_support && think.is_some_and(|t| t.enabled());

        if !self.thinking_enabled {
            self.state = State::CollectingContent;
            return tools;
        }

        // Continuing after a tool result: the model resumes *inside* its channel,
        // with no `<|channel>` to announce it -- and no channel name to strip.
        if last_message.is_some_and(|m| m.role == "tool") {
            self.state = State::CollectingThinking;
            self.needs_channel_name_strip = false;
            return tools;
        }

        let prefill = last_message.is_some_and(|m| m.role == "assistant");
        if prefill && last_message.is_some_and(|m| !m.content.is_empty()) {
            self.state = State::CollectingContent;
            return tools;
        }

        // Otherwise start in content mode and switch when `<|channel>` shows up.
        // With thinking on, the model normally emits it immediately.
        self.state = State::CollectingContent;
        tools
    }

    fn add(&mut self, s: &str, done: bool) -> Result<Parsed, ParserError> {
        self.buffer.push_str(s);
        let events = self.parse_events(done);

        let mut out = Parsed::default();
        for event in events {
            match event {
                Event::ToolCall(call) => out.calls.push(*call),
                Event::Thinking(t) => {
                    // Channel content is silently DISCARDED when thinking is off.
                    // Upstream's choice: the model still emits a channel block, and
                    // the caller who said "no thinking" must not see it.
                    if self.thinking_enabled {
                        out.thinking.push_str(&t);
                    }
                }
                Event::Content(c) => out.content.push_str(&c),
            }
        }

        for call in &mut out.calls {
            call.function.index = self.call_index;
            self.call_index += 1;
        }

        Ok(out)
    }

    fn preserved_tokens(&self) -> Vec<&'static str> {
        vec![
            THINKING_OPEN_TAG,
            THINKING_CLOSE_TAG,
            TOOL_CALL_OPEN_TAG,
            TOOL_CALL_CLOSE_TAG,
            TOOL_RESPONSE_TAG,
            STRING_DELIMITER,
        ]
    }

    fn has_tool_support(&self) -> bool {
        true
    }

    fn has_thinking_support(&self) -> bool {
        self.has_thinking_support
    }
}

/// Longest overlap between the tail of `buf` and the head of any of `tags`.
/// **Upstream:** `longestOverlap`.
fn longest_overlap(buf: &str, tags: &[&str]) -> usize {
    tags.iter().map(|t| overlap(buf, t)).max().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// The tool-call body: `call:NAME{args}`
// ---------------------------------------------------------------------------

/// **Upstream:** `parseGemma4ToolCall`.
///
/// Strict parse first; only if that fails do the repairs get a turn, and each
/// repaired candidate still has to produce valid JSON. That ordering matters --
/// a repair must never change the meaning of a body that was already fine.
pub(super) fn parse_gemma4_tool_call(content: &str, tools: &[Tool]) -> Result<ToolCall, ParserError> {
    let content = content
        .strip_prefix("call:")
        .ok_or_else(|| ParserError::MalformedToolCall("expected 'call:' prefix".into()))?;

    let brace_idx = content
        .find('{')
        .ok_or_else(|| ParserError::MalformedToolCall("expected '{' in tool call".into()))?;

    let tool_name = content[..brace_idx].trim().to_string();
    let args_str = &content[brace_idx..];

    let arguments = match serde_json::from_str::<ToolCallArguments>(&gemma4_args_to_json(args_str)) {
        Ok(a) => a,
        Err(_) => repair_gemma4_tool_call_args(args_str, &tool_name, tools)?,
    };

    Ok(ToolCall {
        id: String::new(),
        function: ToolCallFunction {
            index: 0,
            name: tool_name,
            arguments,
        },
    })
}

/// Turn Gemma's argument dialect into real JSON. **Upstream:** `gemma4ArgsToJSON`.
///
/// Three passes, and the order is the whole trick:
///
/// 1. every `<|"|>...<|"|>` string is lifted out and replaced by a sentinel, so
///    its contents -- which may hold raw `"`, `{`, `,` -- cannot confuse step 2;
/// 2. bare keys get quoted;
/// 3. the sentinels come back as **properly JSON-escaped** strings.
///
/// **What would make this wrong:** quoting keys before lifting the strings out.
/// Then a value like `<|"|>a,b:c<|"|>` would have its `b` quoted as if it were a
/// key, and the JSON parse would fail on text the model got right.
///
/// **Divergence, stated:** upstream's sentinel is `"\x00" + rune(index) +
/// "\x00"`, which would collide with a literal NUL in the model's own output. We
/// keep the identical scheme rather than invent a safer one, so behaviour matches
/// the oracle byte for byte; a NUL in a tool argument would break upstream too.
fn gemma4_args_to_json(s: &str) -> String {
    let mut quoted_strings: Vec<String> = Vec::new();
    let mut text = String::with_capacity(s.len());

    // Pass 1: `(?s)<\|"\|>(.*?)<\|"\|>` -- non-greedy, so pairs are matched left
    // to right and an unpaired trailing delimiter is left alone.
    let mut rest = s;
    while let Some(open) = rest.find(STRING_DELIMITER) {
        let after_open = &rest[open + STRING_DELIMITER.len()..];
        let Some(close) = after_open.find(STRING_DELIMITER) else {
            break;
        };
        text.push_str(&rest[..open]);
        quoted_strings.push(after_open[..close].to_string());
        text.push_str(&sentinel(quoted_strings.len() - 1));
        rest = &after_open[close + STRING_DELIMITER.len()..];
    }
    text.push_str(rest);

    let mut text = quote_gemma4_bare_keys(&text);

    // Pass 3: sentinels back, JSON-escaped.
    for (i, value) in quoted_strings.iter().enumerate() {
        let escaped = serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string());
        text = text.replace(&sentinel(i), &escaped);
    }

    text
}

/// **Upstream:** `"\x00" + string(rune(i)) + "\x00"`.
fn sentinel(i: usize) -> String {
    let marker = char::from_u32(i as u32).unwrap_or('\u{FFFD}');
    format!("\u{0}{marker}\u{0}")
}

/// Put quotes round bare object keys. **Upstream:** `quoteGemma4BareKeys`.
///
/// A key is only recognised straight after `{` or `,` (with optional space), and
/// only when a `:` follows it. Anything inside a real JSON `"..."` string is
/// skipped wholesale, so a colon in a value cannot be mistaken for a key
/// separator.
fn quote_gemma4_bare_keys(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len() + 16);
    let mut i = 0usize;

    while i < b.len() {
        if b[i] == b'"'
            && let Some(end) = gemma4_json_quoted_string_end(s, i)
        {
            out.push_str(&s[i..end]);
            i = end;
            continue;
        }

        if b[i] != b'{' && b[i] != b',' {
            // Copy a whole char, never half a UTF-8 sequence.
            let n = char_len_at(s, i);
            out.push_str(&s[i..i + n]);
            i += n;
            continue;
        }

        out.push(b[i] as char);
        i += 1;

        let space_start = i;
        i = gemma4_skip_space(s, i);
        out.push_str(&s[space_start..i]);

        let key_end = gemma4_bare_key_end(s, i);
        if key_end > i && key_end < b.len() && b[key_end] == b':' {
            out.push('"');
            out.push_str(&s[i..key_end]);
            out.push_str("\":");
            i = key_end + 1;
        }
        // No key here: upstream falls through WITHOUT copying anything, because
        // the `{`/`,` and the whitespace are already written. Do not "fix" this
        // into an else-branch that re-copies.
    }

    out
}

/// End of a bare key: letters, digits and `_`. **Upstream:** `gemma4BareKeyEnd`.
fn gemma4_bare_key_end(s: &str, start: usize) -> usize {
    let mut i = start;
    while i < s.len() {
        let Some(c) = s[i..].chars().next() else { break };
        if !(c == '_' || c.is_alphanumeric()) {
            break;
        }
        i += c.len_utf8();
    }
    i
}

/// Best-effort repair once the strict parse has failed.
/// **Upstream:** `repairGemma4ToolCallArgs`.
fn repair_gemma4_tool_call_args(
    args_str: &str,
    tool_name: &str,
    tools: &[Tool],
) -> Result<ToolCallArguments, ParserError> {
    for candidate in gemma4_repair_candidates(args_str, tool_name, tools) {
        if let Ok(args) = serde_json::from_str::<ToolCallArguments>(&gemma4_args_to_json(&candidate))
        {
            return Ok(args);
        }
    }
    Err(ParserError::MalformedToolCall(
        "repair failed to produce valid JSON arguments".into(),
    ))
}

fn gemma4_tool_properties<'a>(
    tool_name: &str,
    tools: &'a [Tool],
) -> Option<&'a IndexMap<String, ToolProperty>> {
    tools
        .iter()
        .find(|t| t.function.name == tool_name)
        .map(|t| &t.function.parameters.properties)
}

/// The small, ordered set of repairs we are willing to try.
/// **Upstream:** `gemma4RepairCandidates`.
///
/// The guardrail worth understanding: [`repair_gemma4_missing_object_close`] only
/// runs when some *other* repair already fired (or for the schema-gated raw-value
/// candidate). Adding a `}` on its own would happily "fix" bodies that were never
/// truncated, which is how a repair starts inventing tool calls.
fn gemma4_repair_candidates(args_str: &str, tool_name: &str, tools: &[Tool]) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    let mut candidates: Vec<String> = Vec::new();

    let add = |candidate: &str,
               allow_missing_object_close: bool,
               seen: &mut Vec<String>,
               candidates: &mut Vec<String>| {
        let original = candidate.to_string();
        let mut c = repair_gemma4_single_quoted_values(candidate);
        c = repair_gemma4_missing_string_delimiter(&c);
        if allow_missing_object_close || c != original {
            c = repair_gemma4_missing_object_close(&c);
        }
        if !seen.contains(&c) {
            seen.push(c.clone());
            candidates.push(c);
        }
    };

    add(args_str, false, &mut seen, &mut candidates);
    if let Some(raw) = repair_gemma4_raw_terminal_string_value(args_str, tool_name, tools) {
        add(&raw, true, &mut seen, &mut candidates);
    }

    candidates
}

/// Close an unbalanced `<|"|>`. **Upstream:** `repairGemma4MissingStringDelimiter`.
///
/// The delimiter goes in *before* a trailing `}` or `]`, not after it -- a
/// truncated `{command:<|"|>ls}` means the value is `ls` and the brace closes the
/// object, not that the value is `ls}`.
fn repair_gemma4_missing_string_delimiter(s: &str) -> String {
    if s.matches(STRING_DELIMITER).count().is_multiple_of(2) {
        return s.to_string();
    }

    let mut insert_at = gemma4_trim_right_space_index(s);
    let b = s.as_bytes();
    if insert_at > 0 && (b[insert_at - 1] == b'}' || b[insert_at - 1] == b']') {
        insert_at -= 1;
    }

    let mut out = String::with_capacity(s.len() + STRING_DELIMITER.len());
    out.push_str(&s[..insert_at]);
    out.push_str(STRING_DELIMITER);
    out.push_str(&s[insert_at..]);
    out
}

/// Add a final `}` to a truncated object. **Upstream:**
/// `repairGemma4MissingObjectClose`. Purely mechanical -- the caller decides when
/// it is allowed to run.
fn repair_gemma4_missing_object_close(s: &str) -> String {
    if !s.trim_start().starts_with('{') {
        return s.to_string();
    }
    let trimmed_end = gemma4_trim_right_space_index(s);
    if trimmed_end > 0 && s.as_bytes()[trimmed_end - 1] == b'}' {
        return s.to_string();
    }
    format!("{}}}{}", &s[..trimmed_end], &s[trimmed_end..])
}

/// Turn `'single quoted'` values into `<|"|>`-delimited ones.
/// **Upstream:** `repairGemma4SingleQuotedValues`.
///
/// Only after a `:` -- a stray apostrophe inside a value is left alone. A spare
/// delimiter immediately after the closing quote is dropped, because the model
/// sometimes emits both.
fn repair_gemma4_single_quoted_values(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;

    while i < b.len() {
        if s[i..].starts_with(STRING_DELIMITER) {
            let after = &s[i + STRING_DELIMITER.len()..];
            match after.find(STRING_DELIMITER) {
                None => {
                    out.push_str(&s[i..]);
                    break;
                }
                Some(rel) => {
                    let end = i + STRING_DELIMITER.len() + rel + STRING_DELIMITER.len();
                    out.push_str(&s[i..end]);
                    i = end;
                    continue;
                }
            }
        }

        if b[i] == b'"'
            && let Some(end) = gemma4_json_quoted_string_end(s, i)
        {
            out.push_str(&s[i..end]);
            i = end;
            continue;
        }

        if b[i] != b':' {
            let n = char_len_at(s, i);
            out.push_str(&s[i..i + n]);
            i += n;
            continue;
        }

        out.push(':');
        i += 1;

        let space_end = gemma4_skip_space(s, i);
        out.push_str(&s[i..space_end]);
        i = space_end;
        if i >= b.len() || b[i] != b'\'' {
            continue;
        }

        let Some((value, end)) = gemma4_single_quoted_value(s, i) else {
            continue;
        };

        out.push_str(STRING_DELIMITER);
        out.push_str(&value);
        out.push_str(STRING_DELIMITER);
        i = end;
        if s[i..].starts_with(STRING_DELIMITER) {
            i += STRING_DELIMITER.len();
        }
    }

    out
}

/// **Upstream:** `gemma4SingleQuotedValue`. Returns the contents and the index
/// just past the closing quote, or `None` if it never closes.
fn gemma4_single_quoted_value(s: &str, start: usize) -> Option<(String, usize)> {
    let b = s.as_bytes();
    let mut sb = String::new();
    let mut escaped = false;
    let mut i = start + 1;
    while i < b.len() {
        if b[i] == b'\'' && !escaped {
            return Some((sb, i + 1));
        }
        let n = char_len_at(s, i);
        sb.push_str(&s[i..i + n]);
        escaped = b[i] == b'\\' && !escaped;
        if b[i] != b'\\' {
            escaped = false;
        }
        i += n;
    }
    None
}

/// Wrap a bare terminal value in string delimiters -- **only** where the tool
/// schema declares that argument a string.
/// **Upstream:** `repairGemma4RawTerminalStringValue`.
///
/// The schema gate is the point. Without it, `{count:12` would become
/// `{count:<|"|>12<|"|>}` and the tool would be handed the string `"12"` where it
/// expected a number, with nothing anywhere to show the meaning changed.
fn repair_gemma4_raw_terminal_string_value(
    args_str: &str,
    tool_name: &str,
    tools: &[Tool],
) -> Option<String> {
    let props = gemma4_tool_properties(tool_name, tools)?;
    for (key, prop) in props.iter() {
        if !gemma4_property_accepts_string(prop) {
            continue;
        }
        if let Some(repaired) = repair_gemma4_raw_terminal_string_value_for_key(args_str, key, props)
        {
            return Some(repaired);
        }
    }
    None
}

/// **Upstream:** `repairGemma4RawTerminalStringValueForKey`.
fn repair_gemma4_raw_terminal_string_value_for_key(
    s: &str,
    key: &str,
    props: &IndexMap<String, ToolProperty>,
) -> Option<String> {
    let mut search_start = 0usize;
    while search_start < s.len() {
        let value_start = gemma4_find_value_start_for_key(s, key, search_start)?;

        // A value that already starts with a quote, brace, bracket or JSON
        // literal is not raw text -- leave it and look for a later occurrence.
        let value_check = gemma4_skip_space(s, value_start);
        if value_check < s.len() && gemma4_value_starts_structured(s, value_check) {
            search_start = value_start;
            continue;
        }

        let value_end = gemma4_raw_string_value_end(s, value_start, props);
        return Some(format!(
            "{}{}{}{}{}",
            &s[..value_start],
            STRING_DELIMITER,
            &s[value_start..value_end],
            STRING_DELIMITER,
            &s[value_end..]
        ));
    }
    None
}

/// **Upstream:** `gemma4FindValueStartForKey`. Returns the index just after the
/// `:` that follows `key`, skipping over any delimited or quoted strings so a
/// key name appearing inside a value cannot be mistaken for the real thing.
fn gemma4_find_value_start_for_key(s: &str, key: &str, search_start: usize) -> Option<usize> {
    let b = s.as_bytes();
    let mut i = search_start;
    while i < b.len() {
        if s[i..].starts_with(STRING_DELIMITER) {
            let after = &s[i + STRING_DELIMITER.len()..];
            let rel = after.find(STRING_DELIMITER)?;
            i += STRING_DELIMITER.len() + rel + STRING_DELIMITER.len();
            continue;
        }

        if b[i] == b'"'
            && let Some(end) = gemma4_json_quoted_string_end(s, i)
        {
            i = end;
            continue;
        }

        if b[i] != b'{' && b[i] != b',' {
            i += char_len_at(s, i);
            continue;
        }

        let key_start = gemma4_skip_space(s, i + 1);
        if !s[key_start..].starts_with(key) {
            i += 1;
            continue;
        }

        let colon = gemma4_skip_space(s, key_start + key.len());
        if colon < b.len() && b[colon] == b':' {
            return Some(colon + 1);
        }
        i += 1;
    }
    None
}

/// Where a raw (undelimited) string value ends.
/// **Upstream:** `gemma4RawStringValueEnd`.
///
/// It runs to the next `, key:` where `key` is a **declared** property of this
/// tool -- so a comma inside the raw text does not end it. Failing that, to just
/// before a trailing `}`, or to the end.
fn gemma4_raw_string_value_end(
    s: &str,
    start: usize,
    props: &IndexMap<String, ToolProperty>,
) -> usize {
    let b = s.as_bytes();
    let mut i = start;
    while i < b.len() {
        if b[i] != b',' {
            i += char_len_at(s, i);
            continue;
        }
        let key_start = gemma4_skip_space(s, i + 1);
        let key_end = gemma4_bare_key_end(s, key_start);
        if key_end == key_start {
            i += 1;
            continue;
        }
        let colon = gemma4_skip_space(s, key_end);
        if colon < b.len() && b[colon] == b':' && props.contains_key(&s[key_start..key_end]) {
            return i;
        }
        i += 1;
    }

    let end = gemma4_trim_right_space_index(s);
    if end > start && b[end - 1] == b'}' {
        return end - 1;
    }
    s.len()
}

/// **Upstream:** `gemma4ValueStartsStructured`.
fn gemma4_value_starts_structured(s: &str, pos: usize) -> bool {
    if pos >= s.len() {
        return false;
    }
    if s[pos..].starts_with(STRING_DELIMITER) {
        return true;
    }
    let ch = s.as_bytes()[pos];
    matches!(ch, b'\'' | b'"' | b'{' | b'[') || gemma4_looks_like_json_literal_start(ch)
}

/// Index just past the closing `"` of a JSON string starting at `start`, honouring
/// backslash escapes. **Upstream:** `gemma4JSONQuotedStringEnd`.
fn gemma4_json_quoted_string_end(s: &str, start: usize) -> Option<usize> {
    let b = s.as_bytes();
    let mut escaped = false;
    let mut i = start + 1;
    while i < b.len() {
        if b[i] == b'"' && !escaped {
            return Some(i + 1);
        }
        escaped = b[i] == b'\\' && !escaped;
        if b[i] != b'\\' {
            escaped = false;
        }
        i += 1;
    }
    None
}

/// **Upstream:** `gemma4SkipSpace`.
fn gemma4_skip_space(s: &str, mut i: usize) -> usize {
    while i < s.len() {
        let Some(c) = s[i..].chars().next() else { break };
        if !c.is_whitespace() {
            return i;
        }
        i += c.len_utf8();
    }
    i
}

/// **Upstream:** `gemma4TrimRightSpaceIndex`.
fn gemma4_trim_right_space_index(s: &str) -> usize {
    s.trim_end().len()
}

/// Does this property accept a string, directly or through an `anyOf` branch?
/// **Upstream:** `gemma4PropertyAcceptsString`.
fn gemma4_property_accepts_string(prop: &ToolProperty) -> bool {
    if prop.prop_type.0.iter().any(|t| t.eq_ignore_ascii_case("string")) {
        return true;
    }
    prop.any_of.iter().any(gemma4_property_accepts_string)
}

/// **Upstream:** `gemma4LooksLikeJSONLiteralStart` -- `-`, a digit, or the first
/// letter of `true` / `false` / `null`.
fn gemma4_looks_like_json_literal_start(ch: u8) -> bool {
    ch == b'-' || ch.is_ascii_digit() || ch == b't' || ch == b'f' || ch == b'n'
}

fn char_len_at(s: &str, i: usize) -> usize {
    s[i..].chars().next().map(char::len_utf8).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{PropertyType, ToolFunction, ToolFunctionParameters};
    use serde_json::json;

    fn tool(name: &str, props: &[(&str, &str)]) -> Tool {
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
                    ..Default::default()
                },
            },
        }
    }

    fn thinking() -> Gemma4Parser {
        let mut p = Gemma4Parser::new(true);
        p.init(Vec::new(), None, Some(&ThinkValue::Bool(true)));
        p
    }

    fn plain() -> Gemma4Parser {
        let mut p = Gemma4Parser::new(false);
        p.init(Vec::new(), None, None);
        p
    }

    /// Upstream's `TestParserPreservedTokensCoverKnownLlamaServerRegressions`
    /// asserts these two by name -- they are the asymmetric pair that is easy to
    /// mistype as `</tool_call>`.
    #[test]
    fn the_preserved_tokens_include_both_halves_of_the_asymmetric_tool_tag() {
        let p = Gemma4Parser::new(true);
        let toks = p.preserved_tokens();
        assert!(toks.contains(&"<|tool_call>"));
        assert!(toks.contains(&"<tool_call|>"));
    }

    /// Upstream `TestGemma4Parser`, "simple tool call".
    #[test]
    fn a_simple_tool_call_is_parsed() {
        let mut p = plain();
        let got = p
            .add(
                r#"<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|>"#,
                true,
            )
            .expect("add");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "get_weather");
        assert_eq!(
            got.calls[0].function.arguments.get("location"),
            Some(&json!("Paris"))
        );
    }

    /// Upstream `TestParseGemma4ToolCall_UnquotedScalarsKeepStructuredTypes` and
    /// `..._QuotedScalarsStayStrings` -- the delimiter is what decides the type,
    /// not the text.
    #[test]
    fn the_string_delimiter_is_what_decides_a_scalars_type() {
        let bare = parse_gemma4_tool_call("call:foo{n:1,b:true,z:null}", &[]).expect("parse");
        assert_eq!(bare.function.arguments.get("n"), Some(&json!(1)));
        assert_eq!(bare.function.arguments.get("b"), Some(&json!(true)));
        assert_eq!(bare.function.arguments.get("z"), Some(&json!(null)));

        let quoted =
            parse_gemma4_tool_call(r#"call:foo{n:<|"|>1<|"|>,b:<|"|>true<|"|>,z:<|"|>null<|"|>}"#, &[])
                .expect("parse");
        assert_eq!(quoted.function.arguments.get("n"), Some(&json!("1")));
        assert_eq!(quoted.function.arguments.get("b"), Some(&json!("true")));
        assert_eq!(quoted.function.arguments.get("z"), Some(&json!("null")));
    }

    /// Upstream `TestParseGemma4ToolCall_ReferenceImplementationExample`.
    #[test]
    fn the_reference_implementation_example_parses() {
        let call = parse_gemma4_tool_call(
            r#"call:get_current_temperature{detail_level:0,location:<|"|>Paris, France<|"|>,unit:<|"|>celsius<|"|>}"#,
            &[],
        )
        .expect("parse");
        assert_eq!(call.function.name, "get_current_temperature");
        assert_eq!(call.function.arguments.get("detail_level"), Some(&json!(0)));
        assert_eq!(
            call.function.arguments.get("location"),
            Some(&json!("Paris, France"))
        );
        assert_eq!(call.function.arguments.get("unit"), Some(&json!("celsius")));
    }

    /// The reason the delimiter exists at all: a value may contain raw `"`.
    #[test]
    fn a_delimited_value_may_contain_raw_double_quotes() {
        let call = parse_gemma4_tool_call(
            r#"call:exec{command:<|"|>fetch "https://ollama.com" --extract<|"|>}"#,
            &[],
        )
        .expect("parse");
        assert_eq!(
            call.function.arguments.get("command"),
            Some(&json!(r#"fetch "https://ollama.com" --extract"#))
        );
    }

    /// Nested objects and arrays keep working, because bare-key quoting runs
    /// after the strings have been lifted out.
    #[test]
    fn nested_objects_and_arrays_survive_the_conversion() {
        let call = parse_gemma4_tool_call(
            r#"call:process{config:{enabled:true,name:<|"|>test<|"|>},items:[<|"|>a<|"|>,<|"|>b<|"|>]}"#,
            &[],
        )
        .expect("parse");
        assert_eq!(
            call.function.arguments.get("config"),
            Some(&json!({"enabled": true, "name": "test"}))
        );
        assert_eq!(call.function.arguments.get("items"), Some(&json!(["a", "b"])));
    }

    /// Upstream `TestGemma4Parser`, the thinking-plus-tool-call case. `thought\n`
    /// is framing and must not reach the caller.
    #[test]
    fn the_channel_name_is_stripped_from_thinking() {
        let mut p = thinking();
        let got = p
            .add(
                "<|channel>thought\nI need to check the weather<channel|><|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>",
                true,
            )
            .expect("add");
        assert_eq!(got.thinking, "I need to check the weather");
        assert_eq!(got.calls.len(), 1);
        assert!(got.content.is_empty());
    }

    /// ...and it must still be stripped when `thought` and `\n` land in different
    /// chunks, which is the whole reason for the partial-match branch.
    #[test]
    fn the_channel_name_is_stripped_even_when_split_across_chunks() {
        let mut p = thinking();
        let mut thinking_out = String::new();
        for (i, c) in ["<|channel>thou", "ght", "\nreasoning", "<channel|>done"]
            .iter()
            .enumerate()
        {
            let r = p.add(c, i == 3).expect("add");
            thinking_out.push_str(&r.thinking);
        }
        assert_eq!(thinking_out, "reasoning");
    }

    /// With thinking off, channel content is silently DISCARDED -- not leaked
    /// into content, and not reported as thinking.
    #[test]
    fn channel_content_is_discarded_when_thinking_is_off() {
        let mut p = plain();
        let got = p
            .add("<|channel>thought\nsecret reasoning<channel|>visible", true)
            .expect("add");
        assert!(got.thinking.is_empty());
        assert_eq!(got.content, "visible");
    }

    /// Upstream `TestGemma4Parser_StreamingSplitThinkingTag`.
    #[test]
    fn a_thinking_tag_split_across_chunks_never_leaks() {
        let mut p = thinking();
        let (mut content, mut think) = (String::new(), String::new());
        for (i, c) in ["<|chan", "nel>thought\nhmm", "<chan", "nel|>answer"]
            .iter()
            .enumerate()
        {
            let r = p.add(c, i == 3).expect("add");
            content.push_str(&r.content);
            think.push_str(&r.thinking);
        }
        assert_eq!(think, "hmm");
        assert_eq!(content, "answer");
    }

    /// Upstream `TestGemma4Parser_IgnoresExtraToolCallCloseTags`.
    #[test]
    fn a_stray_repeated_close_tag_after_a_tool_call_is_swallowed() {
        for (input, want_content) in [
            (
                r#"<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|><tool_call|>"#,
                "",
            ),
            (
                r#"<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|><tool_call|>Done."#,
                "Done.",
            ),
        ] {
            let mut p = plain();
            let got = p.add(input, true).expect("add");
            assert_eq!(got.calls.len(), 1, "for {input}");
            assert_eq!(got.content, want_content, "for {input}");
        }
    }

    /// Upstream `TestGemma4Parser_IgnoresToolResponseBoundaryAfterToolCall`.
    #[test]
    fn a_tool_response_boundary_after_a_tool_call_is_swallowed() {
        let mut p = plain();
        let got = p
            .add(
                r#"<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|><|tool_response>Done."#,
                true,
            )
            .expect("add");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.content, "Done.");
    }

    /// Upstream `TestGemma4Parser_ToolResponseContinuationStartsInThinking`: after
    /// a tool result the model resumes INSIDE its channel with no `<|channel>`, so
    /// the parser must start there -- and must not try to strip a channel name.
    #[test]
    fn continuing_after_a_tool_result_starts_inside_the_thinking_channel() {
        let mut p = Gemma4Parser::new(true);
        let last = Message::new("tool", "22C");
        p.init(Vec::new(), Some(&last), Some(&ThinkValue::Bool(true)));
        let got = p.add("still reasoning<channel|>the answer", true).expect("add");
        assert_eq!(got.thinking, "still reasoning");
        assert_eq!(got.content, "the answer");
    }

    /// Upstream `TestGemma4Parser_StreamingToolCall`: a tool call whose body is
    /// still arriving must emit nothing at all -- half a call is worse than none.
    #[test]
    fn a_partial_tool_call_emits_nothing_until_it_closes() {
        let mut p = plain();
        let got = p.add("<|tool_call>call:get_", false).expect("add");
        assert!(got.is_empty(), "nothing may escape yet: {got:?}");
    }

    /// The model can stop before `<tool_call|>`. On the last chunk we parse what
    /// we have rather than lose the call.
    #[test]
    fn an_unclosed_tool_call_is_flushed_on_the_final_chunk() {
        let mut p = plain();
        let got = p
            .add(r#"<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}"#, true)
            .expect("add");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "get_weather");
    }

    /// Upstream `TestRepairGemma4MissingStringDelimiter` -- the delimiter goes in
    /// BEFORE a trailing brace.
    #[test]
    fn an_unclosed_string_delimiter_is_closed_before_the_trailing_brace() {
        assert_eq!(
            repair_gemma4_missing_string_delimiter(r#"{command:<|"|>ls}"#),
            r#"{command:<|"|>ls<|"|>}"#
        );
        // Balanced input is left completely alone.
        assert_eq!(
            repair_gemma4_missing_string_delimiter(r#"{a:<|"|>x<|"|>}"#),
            r#"{a:<|"|>x<|"|>}"#
        );
    }

    /// Upstream `TestRepairGemma4MissingObjectClose`.
    #[test]
    fn a_truncated_object_gets_its_closing_brace() {
        assert_eq!(repair_gemma4_missing_object_close("{a:1"), "{a:1}");
        // Already closed, or not an object at all: untouched.
        assert_eq!(repair_gemma4_missing_object_close("{a:1}"), "{a:1}");
        assert_eq!(repair_gemma4_missing_object_close("a:1"), "a:1");
    }

    /// Upstream `TestRepairGemma4SingleQuotedValues`.
    #[test]
    fn single_quoted_values_become_delimited_ones() {
        assert_eq!(
            repair_gemma4_single_quoted_values("{pattern:'abc'}"),
            r#"{pattern:<|"|>abc<|"|>}"#
        );
    }

    /// Upstream `TestParseGemma4ToolCall_RepairsIssue15315Examples`: the real
    /// truncation the repairs exist for.
    #[test]
    fn a_truncated_tool_call_from_issue_15315_is_repaired() {
        let call = parse_gemma4_tool_call(r#"call:bash{command:<|"|>ls}"#, &[]).expect("parse");
        assert_eq!(call.function.name, "bash");
        assert_eq!(call.function.arguments.get("command"), Some(&json!("ls")));
    }

    #[test]
    fn a_single_quoted_value_mixed_with_delimited_ones_is_repaired() {
        let call = parse_gemma4_tool_call(
            r#"call:grep{include:<|"|>*.py<|"|>,pattern:'abc',path:<|"|>/tmp<|"|>}"#,
            &[],
        )
        .expect("parse");
        assert_eq!(call.function.arguments.get("pattern"), Some(&json!("abc")));
        assert_eq!(call.function.arguments.get("include"), Some(&json!("*.py")));
        assert_eq!(call.function.arguments.get("path"), Some(&json!("/tmp")));
    }

    /// The schema gate: a bare terminal value is only wrapped when the tool says
    /// that argument is a string. **This is the guardrail that stops the repairs
    /// from quietly changing what a tool is asked to do.**
    #[test]
    fn wrapping_a_bare_value_is_gated_on_the_tool_schema() {
        let tools = vec![tool("bash", &[("command", "string"), ("path", "string")])];
        let call = parse_gemma4_tool_call(r#"call:bash{path:<|"|>/tmp<|"|>,command:ls"#, &tools)
            .expect("parse");
        assert_eq!(call.function.arguments.get("command"), Some(&json!("ls")));

        // With no schema at all there is nothing to gate on, so the raw-value
        // candidate is never offered.
        assert!(gemma4_repair_candidates(r#"{command:ls"#, "bash", &[]).len() == 1);
    }

    /// Upstream `TestParseGemma4ToolCall_InvalidRawQuotedEscape` -- a Windows
    /// path in a plain JSON string has invalid escapes and must NOT be silently
    /// accepted.
    #[test]
    fn an_invalid_escape_in_a_raw_json_string_is_rejected() {
        assert!(parse_gemma4_tool_call(r#"call:open_file{path:"C:\users\bob\file.txt"}"#, &[]).is_err());
    }

    /// ...but the same path inside the Gemma delimiter is fine, which is exactly
    /// why the delimiter exists.
    #[test]
    fn the_same_windows_path_is_fine_inside_the_gemma_delimiter() {
        let call =
            parse_gemma4_tool_call(r#"call:open_file{path:<|"|>C:\users\bob\file.txt<|"|>}"#, &[])
                .expect("parse");
        assert_eq!(
            call.function.arguments.get("path"),
            Some(&json!(r"C:\users\bob\file.txt"))
        );
    }

    /// Upstream `TestGemma4Parser`, parallel calls.
    #[test]
    fn parallel_tool_calls_are_indexed_in_order() {
        let mut p = plain();
        let got = p
            .add(
                r#"<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|><|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|>"#,
                true,
            )
            .expect("add");
        assert_eq!(got.calls.len(), 2);
        assert_eq!(got.calls[0].function.index, 0);
        assert_eq!(got.calls[1].function.index, 1);
        assert_eq!(
            got.calls[1].function.arguments.get("location"),
            Some(&json!("London"))
        );
    }

    /// Content before a tool call is emitted, trailing whitespace trimmed.
    #[test]
    fn content_before_a_tool_call_is_emitted_first() {
        let mut p = plain();
        let got = p
            .add(
                r#"Let me check that for you. <|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|>"#,
                true,
            )
            .expect("add");
        assert_eq!(got.content, "Let me check that for you.");
        assert_eq!(got.calls.len(), 1);
    }

    /// Upstream `TestGemma4ArgsToJSON`: bare keys get quoted, delimited strings
    /// become properly escaped JSON strings.
    #[test]
    fn the_args_dialect_converts_to_real_json() {
        assert_eq!(
            gemma4_args_to_json(r#"{location:<|"|>Paris<|"|>}"#),
            r#"{"location":"Paris"}"#
        );
        assert_eq!(gemma4_args_to_json("{value:42}"), r#"{"value":42}"#);
        // A raw double quote inside a delimited value gets escaped on the way out.
        assert_eq!(
            gemma4_args_to_json(r#"{q:<|"|>say "hi"<|"|>}"#),
            r#"{"q":"say \"hi\""}"#
        );
    }

    #[test]
    fn a_body_without_the_call_prefix_is_rejected() {
        assert!(parse_gemma4_tool_call("get_weather{a:1}", &[]).is_err());
        assert!(parse_gemma4_tool_call("call:get_weather", &[]).is_err());
    }
}
