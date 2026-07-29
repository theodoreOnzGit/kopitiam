//! LFM2 (Liquid AI) response parser -- `lfm2` and `lfm2-thinking`.
//!
//! **Upstream:** `model/parsers/lfm2.go` (ollama, MIT, Copyright (c) Ollama).
//! Ported against `4713800b08b2ddf5e14acf8398953cf7b12f169b` (2026-07-28).
//!
//! ## The thing that makes LFM2 different: thinking is OPTIONAL per turn
//!
//! Most families that support thinking always open a `<think>` block (or the
//! prompt opens it for them). **LFM2 does not.** The model emits `<think>` only
//! when it actually reasons; a direct answer carries no tag at all.
//!
//! So this parser has an extra opening state,
//! [`State::LookingForThinking`], whose only job is to wait and see which kind
//! of turn this is. It commits to thinking the moment it sees a leading
//! `<think>`, and commits to content the moment it sees anything that cannot
//! become one.
//!
//! **What would make this wrong:** treating the first bytes as thinking on
//! spec. A direct answer would then be silently filed as reasoning and the user
//! would see an empty reply.
//!
//! And the matching end-of-stream rule, which is easy to miss: on the final
//! chunk a partial `"<thi"` **can never complete**, so [`LFM2Parser::add`]
//! commits the buffer as content rather than withholding it forever. Drop that
//! and a reply consisting of the single word `"<th"` disappears.
//!
//! ## `None` means thinking OFF here
//!
//! `thinkingEnabled := p.HasThinkingSupport() && (thinkValue != nil && thinkValue.Bool())`
//!
//! Read it carefully: a `nil` think value is **false** for LFM2. That is the
//! opposite of `qwen3.5` and `cohere`, where `None` means on. Per-family choice,
//! upstream's, not a default we get to unify.
//!
//! ## Tool calls are Python, wrapped in special tokens
//!
//! ```text
//! I'll check.<|tool_call_start|>[get_weather(location="Paris")]<|tool_call_end|>
//! ```
//!
//! Python call syntax again (like olmo3), but a **different** dialect and a
//! genuinely different parser: LFM2 puts several calls in one bracketed list
//! separated by commas, where olmo3 puts one per line. The two literal parsers
//! are therefore NOT shared -- they disagree on the separator, and merging them
//! would break both.
//!
//! ## The bare-tool-call fallback, and its guard rail
//!
//! Some LFM2 checkpoints emit the call with **no `<|tool_call_*|>` wrappers at
//! all**, so the whole thing lands in content. On `done`, if no calls were found
//! and tools were offered, the accumulated content is re-parsed as a call.
//!
//! The guard rail matters as much as the fallback: [`LFM2Parser::tool_calls_allowed`]
//! only accepts the reparse when **every** call names a tool the caller actually
//! offered. Without it, ordinary prose like `[img-0]describe this image` or a
//! model writing `[some_function(x=1)]` inside an explanation would be eaten and
//! turned into a phantom tool call, and the user's reply would vanish. Upstream
//! pins both directions (`TestLFM2Parser_BareToolCallFallback` and
//! `TestLFM2Parser_BareUnknownToolCallDoesNotParse`).

use std::collections::HashSet;

use crate::api::{Message, ThinkValue, Tool, ToolCall, ToolCallArguments, ToolCallFunction};

use super::{Parsed, Parser, ParserError, overlap, trailing_whitespace_len};

/// **Upstream:** the `lfm2*Tag` consts.
const THINKING_OPEN_TAG: &str = "<think>";
const THINKING_CLOSE_TAG: &str = "</think>";
const TOOL_CALL_START_TAG: &str = "<|tool_call_start|>";
const TOOL_CALL_END_TAG: &str = "<|tool_call_end|>";

/// **Upstream:** `LFM2ParserState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    /// Waiting to find out whether this turn reasons or answers directly. See
    /// the module docs -- this state is the whole reason LFM2 needs its own
    /// parser rather than the generic thinking machine.
    #[default]
    LookingForThinking,
    CollectingThinking,
    CollectingContent,
    CollectingToolCalls,
}

/// **Upstream:** `lfm2Event`.
#[derive(Debug, Clone, PartialEq)]
enum Event {
    Thinking(String),
    Content(String),
    ToolCall(Box<ToolCall>),
}

/// **Upstream:** `LFM2Parser`. `has_thinking_support` distinguishes the two
/// registered names: `lfm2` (false) and `lfm2-thinking` (true).
#[derive(Debug, Default)]
pub struct LFM2Parser {
    state: State,
    buffer: String,
    call_index: usize,
    has_thinking_support: bool,
    /// Trim leading whitespace after `<think>`. A flag rather than a one-shot
    /// trim because the whitespace **may span chunks** -- `<think>` then `"\n"`
    /// then `"\n  reasoning"` arrives as three calls, and all of that leading
    /// whitespace has to go.
    needs_thinking_leading_trim: bool,
    /// Same idea for the whitespace after `</think>`.
    needs_content_leading_trim: bool,
    /// The tool names the caller offered. Guards the bare-call fallback.
    tool_names: HashSet<String>,
    has_tools: bool,
}

impl LFM2Parser {
    /// `lfm2` -> `false`, `lfm2-thinking` -> `true`.
    pub fn new(has_thinking_support: bool) -> Self {
        Self {
            has_thinking_support,
            ..Default::default()
        }
    }

    /// **Upstream:** `setInitialState`.
    fn set_initial_state(&mut self, last_message: Option<&Message>, think: Option<&ThinkValue>) {
        let prefill = last_message.is_some_and(|m| m.role == "assistant");

        // Model capability AND request preference. `None` is OFF -- see the
        // module docs; this differs from qwen3.5 and cohere on purpose.
        let thinking_enabled = self.has_thinking_support && think.is_some_and(|t| t.enabled());

        if !thinking_enabled {
            self.state = State::CollectingContent;
            return;
        }

        if prefill && last_message.is_some_and(|m| !m.content.is_empty()) {
            self.state = State::CollectingContent;
            return;
        }

        self.state = State::LookingForThinking;
    }

    /// Would accepting these calls be sane? **Upstream:** `toolCallsAllowed`.
    ///
    /// The rules, in order:
    /// * no calls -> no;
    /// * caller offered no named tools -> **yes**, accept anything (there is
    ///   nothing to check against, and refusing would disable the fallback
    ///   entirely);
    /// * otherwise every call must name an offered tool -- **all of them**, not
    ///   just one. One unknown name rejects the whole batch, because a batch
    ///   with a made-up name is far more likely to be prose than a real call.
    fn tool_calls_allowed(&self, calls: &[ToolCall]) -> bool {
        if calls.is_empty() {
            return false;
        }
        if self.tool_names.is_empty() {
            return true;
        }
        calls
            .iter()
            .all(|c| self.tool_names.contains(&c.function.name))
    }

    /// **Upstream:** `(*LFM2Parser).parseEvents`.
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

    /// **Upstream:** `(*LFM2Parser).eat`.
    fn eat(&mut self) -> (Vec<Event>, bool) {
        if self.buffer.is_empty() {
            return (Vec::new(), false);
        }

        match self.state {
            State::LookingForThinking => {
                // Leading whitespace is ignored either way, so trim before
                // deciding.
                let trimmed = self.buffer.trim_start().to_string();

                if let Some(after) = trimmed.strip_prefix(THINKING_OPEN_TAG) {
                    self.buffer = after.to_string();
                    self.state = State::CollectingThinking;
                    self.needs_thinking_leading_trim = true;
                    return (Vec::new(), true);
                }

                if trimmed.is_empty() || THINKING_OPEN_TAG.starts_with(&trimmed) {
                    // Only whitespace so far, or still could grow into
                    // `<think>`. Wait -- committing now could file a reasoning
                    // turn as content.
                    return (Vec::new(), false);
                }

                // Anything else proves this is a direct answer.
                self.buffer = trimmed;
                self.state = State::CollectingContent;
                (Vec::new(), true)
            }

            State::CollectingThinking => {
                // A `<think>` can still show up here when the state was entered
                // some other way; strip it and re-arm the trim.
                if let Some(after) = self.buffer.strip_prefix(THINKING_OPEN_TAG) {
                    self.buffer = after.to_string();
                    self.needs_thinking_leading_trim = true;
                }

                if self.needs_thinking_leading_trim {
                    let trimmed = self.buffer.trim_start();
                    if trimmed.len() != self.buffer.len() {
                        self.buffer = trimmed.to_string();
                    }
                    // Only disarm once real content has landed -- an all-
                    // whitespace chunk must leave the flag armed so the NEXT
                    // chunk's leading whitespace is trimmed too.
                    if !self.buffer.is_empty() {
                        self.needs_thinking_leading_trim = false;
                    }
                }

                if let Some(idx) = self.buffer.find(THINKING_CLOSE_TAG) {
                    let thinking = self.buffer[..idx].trim_end().to_string();
                    let remaining = self.buffer[idx + THINKING_CLOSE_TAG.len()..]
                        .trim_start()
                        .to_string();
                    self.buffer = remaining.clone();
                    self.state = State::CollectingContent;
                    self.needs_thinking_leading_trim = false;
                    // If nothing followed the close tag yet, whitespace may
                    // still arrive in a later chunk -- arm the content trim.
                    self.needs_content_leading_trim = remaining.is_empty();

                    let mut events = Vec::new();
                    if !thinking.is_empty() {
                        events.push(Event::Thinking(thinking));
                    }
                    return (events, true);
                }

                // Hold back a partial `</think>` AND the whitespace in front of
                // it (which would be trimmed if the tag lands). When there is no
                // partial tag at all, `overlap` is 0 and this degrades to "hold
                // back only the trailing whitespace" -- still necessary, since
                // `</think>` may be the very next thing to arrive.
                let overlap_len = overlap(&self.buffer, THINKING_CLOSE_TAG);
                let split = self.buffer.len() - overlap_len;
                let (before_partial_tag, _) = super::chop(&self.buffer, split);
                let ambiguous_start =
                    before_partial_tag.len() - trailing_whitespace_len(before_partial_tag);
                let (unambiguous, ambiguous) = super::chop(&self.buffer, ambiguous_start);
                let (unambiguous, ambiguous) = (unambiguous.to_string(), ambiguous.to_string());
                self.buffer = ambiguous;

                let mut events = Vec::new();
                if !unambiguous.is_empty() {
                    events.push(Event::Thinking(unambiguous));
                }
                (events, false)
            }

            State::CollectingContent => {
                if self.needs_content_leading_trim {
                    let trimmed = self.buffer.trim_start();
                    if trimmed.len() != self.buffer.len() {
                        self.buffer = trimmed.to_string();
                    }
                    if !self.buffer.is_empty() {
                        self.needs_content_leading_trim = false;
                    }
                }

                if let Some(idx) = self.buffer.find(TOOL_CALL_START_TAG) {
                    let content_before = self.buffer[..idx].trim_end().to_string();
                    self.buffer = self.buffer[idx + TOOL_CALL_START_TAG.len()..].to_string();
                    self.state = State::CollectingToolCalls;
                    let mut events = Vec::new();
                    if !content_before.is_empty() {
                        events.push(Event::Content(content_before));
                    }
                    return (events, true);
                }

                // NOTE: no partial-tag buffering for `<|tool_call_start|>` here.
                // Upstream emits the whole buffer as content, so a split start
                // tag can briefly leak. That is upstream's behaviour and the
                // bare-call fallback is what cleans up after it -- diverging
                // here would make this port disagree with ollama on real output.
                let content = std::mem::take(&mut self.buffer);
                (vec![Event::Content(content)], false)
            }

            State::CollectingToolCalls => {
                if let Some(idx) = self.buffer.find(TOOL_CALL_END_TAG) {
                    let tool_call_content = self.buffer[..idx].to_string();
                    if let Ok(tool_calls) = parse_tool_calls_content(&tool_call_content)
                        && !tool_calls.is_empty()
                    {
                        let mut remaining =
                            self.buffer[idx + TOOL_CALL_END_TAG.len()..].to_string();

                        // Back-to-back calls: `...end|><|tool_call_start|>...`
                        // stays in this state rather than bouncing through
                        // content, which would emit an empty content event.
                        if let Some(next) = remaining.strip_prefix(TOOL_CALL_START_TAG) {
                            remaining = next.to_string();
                        } else {
                            remaining = remaining.trim_start().to_string();
                            self.state = State::CollectingContent;
                        }

                        self.buffer = remaining;
                        let events = tool_calls
                            .into_iter()
                            .map(|tc| Event::ToolCall(Box::new(tc)))
                            .collect();
                        return (events, true);
                    }
                    // Parse failed (upstream logs a warning). Fall through and
                    // wait -- the body may still be arriving.
                }

                (Vec::new(), false)
            }
        }
    }
}

impl Parser for LFM2Parser {
    fn init(
        &mut self,
        tools: Vec<Tool>,
        last_message: Option<&Message>,
        think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        self.tool_names = tools
            .iter()
            .map(|t| t.function.name.clone())
            .filter(|n| !n.is_empty())
            .collect();
        self.call_index = 0;
        self.has_tools = !tools.is_empty();
        // Upstream does not clear the buffer or the trim flags in Init; it
        // relies on always building a fresh parser. Cleared here so a re-`init`
        // cannot inherit half a tag. Stated divergence, no behaviour change for
        // the documented lifecycle.
        self.buffer.clear();
        self.needs_thinking_leading_trim = false;
        self.needs_content_leading_trim = false;
        self.set_initial_state(last_message, think);
        tools
    }

    /// **Upstream:** `(*LFM2Parser).Add`.
    fn add(&mut self, s: &str, done: bool) -> Result<Parsed, ParserError> {
        self.buffer.push_str(s);

        // End of stream: a partial `<think>` prefix can never complete now, so
        // commit the buffered output as a direct answer instead of withholding
        // it forever.
        if done && self.state == State::LookingForThinking {
            let trimmed = self.buffer.trim_start().to_string();
            if !trimmed.starts_with(THINKING_OPEN_TAG) {
                self.buffer = trimmed;
                self.state = State::CollectingContent;
            }
        }

        let mut content = String::new();
        let mut thinking = String::new();
        let mut calls: Vec<ToolCall> = Vec::new();

        for event in self.parse_events() {
            match event {
                Event::ToolCall(tc) => calls.push(*tc),
                Event::Thinking(t) => thinking.push_str(&t),
                Event::Content(c) => content.push_str(&c),
            }
        }

        // The bare-tool-call fallback. See the module docs for why the
        // `tool_calls_allowed` guard is not optional.
        if done && calls.is_empty() && self.has_tools {
            let candidate = content.trim();
            if let Ok(fallback) = parse_tool_calls_content(candidate)
                && self.tool_calls_allowed(&fallback)
            {
                content.clear();
                calls = fallback;
            }
        }

        for call in &mut calls {
            call.function.index = self.call_index;
            self.call_index += 1;
        }

        Ok(Parsed {
            content,
            thinking,
            calls,
        })
    }

    fn preserved_tokens(&self) -> Vec<&'static str> {
        vec![
            THINKING_OPEN_TAG,
            THINKING_CLOSE_TAG,
            TOOL_CALL_START_TAG,
            TOOL_CALL_END_TAG,
        ]
    }

    /// Always `true`, even for plain `lfm2`. **Upstream:** `HasToolSupport`
    /// returns a hard `true` while `HasThinkingSupport` returns the flag -- only
    /// thinking varies between the two registered names.
    fn has_tool_support(&self) -> bool {
        true
    }

    fn has_thinking_support(&self) -> bool {
        self.has_thinking_support
    }
}

// ===========================================================================
// The Python-call parser. Upstream: the bottom half of lfm2.go.
// ===========================================================================

/// Byte-index slice that cannot panic on a bad boundary.
///
/// The scanners below use **byte** indices, exactly as upstream does, because
/// every delimiter they look for (`(`, `)`, `,`, `=`, `'`, `"`, `\`) is ASCII.
/// An index can therefore only land mid-character if the input is malformed in a
/// way upstream would also mishandle; returning `""` degrades instead of
/// panicking inside a generation.
fn slice(s: &str, a: usize, b: usize) -> &str {
    s.get(a..b).unwrap_or("")
}

/// Strip stray wrapper tags, then parse. **Upstream:** `parseToolCallsContent`.
///
/// The `trim_start_matches`-style stripping is upstream's tolerance for
/// malformed output that includes a wrapper tag without its pair.
fn parse_tool_calls_content(content: &str) -> Result<Vec<ToolCall>, ParserError> {
    let content = content.trim();
    let content = content.strip_prefix(TOOL_CALL_START_TAG).unwrap_or(content).trim();
    let content = content.strip_suffix(TOOL_CALL_END_TAG).unwrap_or(content).trim();
    parse_python_style_tool_calls(content)
}

/// Parse `[f(a='1'), g(b=2)]` or a bare `f(a='1')`.
///
/// **Upstream:** `parsePythonStyleToolCalls`. Several calls in ONE bracketed
/// list separated by commas -- which is exactly where this differs from olmo3's
/// one-call-per-line format, and why the two literal parsers are not shared.
///
/// Returns an error (never an empty `Ok`) when nothing parses, because the
/// caller uses the error to decide whether the text was a tool call at all.
fn parse_python_style_tool_calls(content: &str) -> Result<Vec<ToolCall>, ParserError> {
    let mut content = content.trim();

    // `[f(...)]` -> `f(...)`
    if content.starts_with('[') && content.ends_with(']') && content.len() >= 2 {
        content = &content[1..content.len() - 1];
    }

    let mut content = content.to_string();
    let mut tool_calls = Vec::new();

    while !content.is_empty() {
        content = content.trim().to_string();
        if content.is_empty() {
            break;
        }

        // Skip the separating comma left by the previous call.
        if let Some(rest) = content.strip_prefix(',') {
            content = rest.trim().to_string();
            if content.is_empty() {
                break;
            }
        }

        let Some(paren_idx) = content.find('(') else {
            return Err(ParserError::MalformedToolCall(
                "invalid tool call: no opening parenthesis".into(),
            ));
        };

        let func_name = content[..paren_idx].trim().to_string();
        if func_name.is_empty() {
            return Err(ParserError::EmptyFunctionName);
        }

        let Some(close_idx) = find_matching_paren(&content, paren_idx) else {
            return Err(ParserError::MalformedToolCall(
                "invalid tool call: no matching closing parenthesis".into(),
            ));
        };

        let args_str = slice(&content, paren_idx + 1, close_idx).to_string();
        let mut args = ToolCallArguments::new();
        if !args_str.is_empty() {
            parse_python_args(&args_str, &mut args)?;
        }

        tool_calls.push(ToolCall {
            function: ToolCallFunction {
                name: func_name,
                arguments: args,
                ..Default::default()
            },
            ..Default::default()
        });

        content = content[close_idx + 1..].to_string();
    }

    if tool_calls.is_empty() {
        return Err(ParserError::MalformedToolCall("no tool calls found".into()));
    }

    Ok(tool_calls)
}

/// Index of the `)` matching the `(` at `open_idx`, or `None`.
///
/// **Upstream:** `findMatchingParen`. Handles nesting **and quoted strings** --
/// the quote handling is what stops `f(msg="smile :-)")` from closing early on
/// the paren inside the string.
fn find_matching_paren(s: &str, open_idx: usize) -> Option<usize> {
    let b = s.as_bytes();
    let mut depth = 1;
    let mut i = open_idx + 1;
    while i < b.len() && depth > 0 {
        match b[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            q @ (b'\'' | b'"') => {
                // Skip the whole quoted string, honouring backslash escapes.
                i += 1;
                while i < b.len() && b[i] != q {
                    if b[i] == b'\\' && i + 1 < b.len() {
                        i += 1;
                    }
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Parse `key='value', key2=42, key3=[1,2]` into ordered arguments.
///
/// **Upstream:** `parsePythonArgs`.
///
/// Note the whitespace test: upstream writes `unicode.IsSpace(rune(argsStr[i]))`,
/// casting a single **byte** to a rune. So byte `0xA0` becomes U+00A0
/// (no-break space) and counts as whitespace even when it is really a UTF-8
/// continuation byte. `(b as char).is_whitespace()` reproduces that exactly --
/// same property, same per-byte cast. It looks like a bug and arguably is one,
/// but "fixing" it here would make this port and ollama disagree on the same
/// input, which is worse.
fn parse_python_args(args_str: &str, args: &mut ToolCallArguments) -> Result<(), ParserError> {
    let b = args_str.as_bytes();
    let mut i = 0usize;

    while i < b.len() {
        // Skip separators and whitespace.
        while i < b.len() && (b[i] == b',' || (b[i] as char).is_whitespace()) {
            i += 1;
        }
        if i >= b.len() {
            break;
        }

        let key_start = i;
        while i < b.len() && b[i] != b'=' && b[i] != b',' {
            i += 1;
        }
        if i >= b.len() || b[i] != b'=' {
            return Err(ParserError::MalformedToolCall(
                "invalid argument: expected '='".into(),
            ));
        }

        let key = slice(args_str, key_start, i).trim().to_string();
        if key.is_empty() {
            return Err(ParserError::MalformedToolCall(
                "invalid argument: empty key".into(),
            ));
        }
        i += 1; // past '='

        while i < b.len() && (b[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= b.len() {
            return Err(ParserError::MalformedToolCall(
                "invalid argument: missing value".into(),
            ));
        }

        let (value, next) = parse_python_arg_value(args_str, i)?;
        args.set(key, value);
        i = next;

        // Optional trailing comma before the next pair.
        if i < b.len() && b[i] == b',' {
            i += 1;
        }
    }

    Ok(())
}

/// Parse one argument value starting at byte `i`; return it and the next index.
///
/// **Upstream:** `parsePythonArgValue`. Two paths:
///
/// * **quoted** -- consume to the matching quote, honouring `\` escapes, and
///   return the raw inner text as a string. Note the escapes are *not* expanded
///   (upstream does not), so `\n` stays two characters.
/// * **unquoted** -- consume until a comma at depth zero, tracking `()`, `[]`,
///   `{}` and quoted strings separately, then hand the token to
///   [`parse_python_literal`]. The depth tracking is what keeps
///   `config={'a': 1, 'b': 2}` in one piece.
fn parse_python_arg_value(s: &str, mut i: usize) -> Result<(serde_json::Value, usize), ParserError> {
    let b = s.as_bytes();
    if i >= b.len() {
        return Err(ParserError::MalformedToolCall(
            "invalid argument: missing value".into(),
        ));
    }

    if b[i] == b'\'' || b[i] == b'"' {
        let quote = b[i];
        i += 1;
        let start = i;
        while i < b.len() {
            if b[i] == b'\\' && i + 1 < b.len() {
                i += 2;
                continue;
            }
            if b[i] == quote {
                let value = slice(s, start, i).to_string();
                return Ok((serde_json::Value::String(value), i + 1));
            }
            i += 1;
        }
        return Err(ParserError::MalformedToolCall(
            "invalid argument: unterminated string".into(),
        ));
    }

    let start = i;
    let (mut d_paren, mut d_square, mut d_curly) = (0i32, 0i32, 0i32);
    let mut in_string = false;
    let mut quote = 0u8;
    let mut escaped = false;

    while i < b.len() {
        let ch = b[i];
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == b'\\' {
                escaped = true;
            } else if ch == quote {
                in_string = false;
            }
            i += 1;
            continue;
        }

        match ch {
            b'\'' | b'"' => {
                in_string = true;
                quote = ch;
            }
            b'(' => d_paren += 1,
            b')' => d_paren = (d_paren - 1).max(0),
            b'[' => d_square += 1,
            b']' => d_square = (d_square - 1).max(0),
            b'{' => d_curly += 1,
            b'}' => d_curly = (d_curly - 1).max(0),
            b',' if d_paren == 0 && d_square == 0 && d_curly == 0 => {
                let token = slice(s, start, i).trim();
                return Ok((parse_python_literal(token), i));
            }
            _ => {}
        }
        i += 1;
    }

    let token = slice(s, start, i).trim();
    Ok((parse_python_literal(token), i))
}

/// Turn one unquoted token into a JSON value.
///
/// **Upstream:** `parsePythonLiteral`. Order is upstream's and matters:
/// empty -> `""`, then Python/JSON booleans (both casings), then null/None, then
/// integer, then float, then -- only for tokens starting `[` or `{` -- a JSON
/// parse with a **Python-to-JSON rewrite as second attempt**.
///
/// Anything unrecognised falls back to the token as a plain string. Lenient on
/// purpose: better a string argument than a dropped call.
fn parse_python_literal(token: &str) -> serde_json::Value {
    use serde_json::Value;

    match token {
        "" => return Value::String(String::new()),
        "true" | "True" => return Value::Bool(true),
        "false" | "False" => return Value::Bool(false),
        "null" | "None" => return Value::Null,
        _ => {}
    }

    if let Ok(v) = token.parse::<i64>() {
        return Value::Number(v.into());
    }
    if let Ok(v) = token.parse::<f64>()
        && let Some(n) = serde_json::Number::from_f64(v)
    {
        return Value::Number(n);
    }

    if token.starts_with('[') || token.starts_with('{') {
        // First try: it might already be valid JSON.
        if let Ok(parsed) = serde_json::from_str::<Value>(token) {
            return parsed;
        }
        // Second try: rewrite Python syntax into JSON and parse that.
        if let Ok(converted) = python_literal_to_json(token)
            && let Ok(parsed) = serde_json::from_str::<Value>(&converted)
        {
            return parsed;
        }
    }

    Value::String(token.to_string())
}

/// Rewrite a Python collection literal as JSON.
///
/// **Upstream:** `pythonLiteralToJSON`. Two jobs, and both are needed:
///
/// 1. **Single quotes become double quotes.** `{'a': 1}` is not JSON. A `"`
///    found *inside* a single-quoted string is escaped to `\"` on the way out,
///    so `{'msg': 'he said "hi"'}` survives instead of producing broken JSON.
/// 2. **`True`/`False`/`None` become `true`/`false`/`null`**, but **only outside
///    strings** -- which is why it scans identifiers rather than doing a blind
///    `replace`. A blind replace would corrupt the string `'None of these'`.
///
/// Returns `Err` on an unterminated string, so the caller can fall back to
/// treating the token as a plain string rather than emitting garbage.
///
/// Byte-wise like upstream. Multi-byte characters pass through unchanged because
/// every byte of a UTF-8 sequence is >= 0x80 and therefore matches none of the
/// ASCII cases, so the output is still valid UTF-8.
fn python_literal_to_json(s: &str) -> Result<String, ParserError> {
    let b = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len() + b.len() / 8);

    let mut in_string = false;
    let mut quote = 0u8;
    let mut escaped = false;
    let mut i = 0usize;

    while i < b.len() {
        let ch = b[i];

        if in_string {
            if escaped {
                out.push(ch);
                escaped = false;
            } else if ch == b'\\' {
                out.push(ch);
                escaped = true;
            } else if ch == quote {
                out.push(b'"');
                in_string = false;
            } else if quote == b'\'' && ch == b'"' {
                // A double quote inside a single-quoted string has to be
                // escaped, because the string is about to BECOME
                // double-quoted.
                out.extend_from_slice(b"\\\"");
            } else {
                out.push(ch);
            }
            i += 1;
            continue;
        }

        if ch == b'\'' || ch == b'"' {
            in_string = true;
            quote = ch;
            escaped = false;
            out.push(b'"');
            i += 1;
            continue;
        }

        if is_ident_start(ch) {
            let mut j = i + 1;
            while j < b.len() && is_ident_part(b[j]) {
                j += 1;
            }
            match &b[i..j] {
                b"True" => out.extend_from_slice(b"true"),
                b"False" => out.extend_from_slice(b"false"),
                b"None" => out.extend_from_slice(b"null"),
                ident => out.extend_from_slice(ident),
            }
            i = j;
            continue;
        }

        out.push(ch);
        i += 1;
    }

    if in_string {
        return Err(ParserError::MalformedToolCall("unterminated string".into()));
    }

    String::from_utf8(out)
        .map_err(|_| ParserError::MalformedToolCall("literal was not valid utf-8".into()))
}

/// **Upstream:** `isIdentStart` -- ASCII only, deliberately.
fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

/// **Upstream:** `isIdentPart`.
fn is_ident_part(b: u8) -> bool {
    is_ident_start(b) || b.is_ascii_digit()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ToolFunction;
    use serde_json::json;

    fn tool(name: &str) -> Tool {
        Tool {
            tool_type: "function".into(),
            function: ToolFunction {
                name: name.into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Upstream's harness: `&LFM2Parser{hasThinkingSupport: X}` then
    /// `Init([]api.Tool{}, nil, &api.ThinkValue{Value: X})`.
    fn lfm2(has_thinking: bool) -> LFM2Parser {
        let mut p = LFM2Parser::new(has_thinking);
        p.init(Vec::new(), None, Some(&ThinkValue::Bool(has_thinking)));
        p
    }

    /// Upstream `TestLFM2Parser`, the whole table, ported as ground truth.
    #[test]
    fn lfm2_splits_thinking_content_and_python_tool_calls() {
        // (name, input, want_content, want_thinking, has_thinking)
        let cases: &[(&str, &str, &str, &str, bool)] = &[
            ("simple_content", "Hello, how are you?", "Hello, how are you?", "", false),
            (
                "thinking_content",
                "<think>I need to think about this...</think>The answer is 42.",
                "The answer is 42.",
                "I need to think about this...",
                true,
            ),
            // Thinking ENABLED but the model chose not to reason -- no tag at
            // all. This is the case `LookingForThinking` exists for.
            (
                "direct_answer_with_thinking_enabled",
                "The answer is 42.",
                "The answer is 42.",
                "",
                true,
            ),
            (
                "thinking_with_newlines",
                "<think>Let me think:\n- Point 1\n- Point 2</think>\n\nHere's my answer.",
                "Here's my answer.",
                "Let me think:\n- Point 1\n- Point 2",
                true,
            ),
            ("empty_content", "", "", "", false),
            (
                "only_thinking",
                "<think>Just thinking content</think>",
                "",
                "Just thinking content",
                true,
            ),
            (
                "unicode_content",
                "\u{645}\u{631}\u{62D}\u{628}\u{627} \u{4F60}\u{597D}\u{4E16}\u{754C}! \u{1F30D}",
                "\u{645}\u{631}\u{62D}\u{628}\u{627} \u{4F60}\u{597D}\u{4E16}\u{754C}! \u{1F30D}",
                "",
                false,
            ),
            (
                "newlines_and_whitespace",
                "Line 1\n\nLine 3\t\tTabbed content",
                "Line 1\n\nLine 3\t\tTabbed content",
                "",
                false,
            ),
            (
                "thinking_with_unicode",
                "<think>\u{6211}\u{5728}\u{601D}\u{8003}...</think>\u{7B54}\u{6848}\u{662F}42\u{3002}",
                "\u{7B54}\u{6848}\u{662F}42\u{3002}",
                "\u{6211}\u{5728}\u{601D}\u{8003}...",
                true,
            ),
            (
                "thinking_with_special_chars",
                "<think>Let me calculate: 2+2=4 & 3*3=9...</think>The results are correct!",
                "The results are correct!",
                "Let me calculate: 2+2=4 & 3*3=9...",
                true,
            ),
        ];

        for (name, input, want_content, want_thinking, has_thinking) in cases {
            let mut p = lfm2(*has_thinking);
            let got = p.add(input, true).expect("add");
            assert_eq!(&got.content, want_content, "content, case {name}");
            assert_eq!(&got.thinking, want_thinking, "thinking, case {name}");
        }
    }

    #[test]
    fn a_wrapped_python_tool_call_is_extracted_and_content_kept() {
        let mut p = lfm2(false);
        let got = p
            .add(
                r#"I'll check the weather.<|tool_call_start|>[get_weather(location="Paris")]<|tool_call_end|>"#,
                true,
            )
            .expect("add");
        assert_eq!(got.content, "I'll check the weather.");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "get_weather");
        assert_eq!(
            got.calls[0].function.arguments.get("location"),
            Some(&json!("Paris"))
        );
    }

    /// Two back-to-back wrapped calls -- the parser must stay in the tool-call
    /// state rather than bouncing through content and emitting an empty chunk.
    #[test]
    fn back_to_back_wrapped_calls_are_both_found_and_indexed() {
        let mut p = lfm2(false);
        let got = p
            .add(
                r#"Getting weather for both cities.<|tool_call_start|>[get_weather(location="Paris")]<|tool_call_end|><|tool_call_start|>[get_weather(location="London")]<|tool_call_end|>"#,
                true,
            )
            .expect("add");
        assert_eq!(got.content, "Getting weather for both cities.");
        assert_eq!(got.calls.len(), 2);
        assert_eq!(got.calls[0].function.index, 0);
        assert_eq!(got.calls[1].function.index, 1);
        assert_eq!(
            got.calls[1].function.arguments.get("location"),
            Some(&json!("London"))
        );
    }

    /// Several calls inside ONE bracketed list, comma-separated. This is the
    /// dialect difference from olmo3 (one per line).
    #[test]
    fn several_calls_in_one_bracketed_list_are_split_on_top_level_commas() {
        let mut p = lfm2(false);
        let got = p
            .add(
                "Running commands.<|tool_call_start|>[bash(command='ls'),bash(command='pwd')]<|tool_call_end|>",
                true,
            )
            .expect("add");
        assert_eq!(got.calls.len(), 2);
        assert_eq!(got.calls[0].function.arguments.get("command"), Some(&json!("ls")));
        assert_eq!(got.calls[1].function.arguments.get("command"), Some(&json!("pwd")));
    }

    /// The Python-literal rewrite: single quotes, `True`, nested collections.
    #[test]
    fn python_collection_arguments_are_rewritten_into_json() {
        let mut p = lfm2(false);
        let got = p
            .add(
                "Processing data.<|tool_call_start|>[process_data(items=['item1','item2'], config={'enabled': True, 'threshold': 0.95})]<|tool_call_end|>",
                true,
            )
            .expect("add");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(
            got.calls[0].function.arguments.get("items"),
            Some(&json!(["item1", "item2"]))
        );
        assert_eq!(
            got.calls[0].function.arguments.get("config"),
            Some(&json!({"enabled": true, "threshold": 0.95}))
        );
    }

    #[test]
    fn mixed_scalar_argument_types_parse_to_their_json_equivalents() {
        let mut p = lfm2(false);
        let got = p
            .add(
                r#"Processing.<|tool_call_start|>[process(name="test", count=42, enabled=true)]<|tool_call_end|>"#,
                true,
            )
            .expect("add");
        let a = &got.calls[0].function.arguments;
        assert_eq!(a.get("name"), Some(&json!("test")));
        assert_eq!(a.get("count"), Some(&json!(42)));
        assert_eq!(a.get("enabled"), Some(&json!(true)));
    }

    #[test]
    fn a_call_with_no_arguments_yields_an_empty_argument_map() {
        let mut p = lfm2(false);
        let got = p
            .add("Pinging server.<|tool_call_start|>[ping()]<|tool_call_end|>", true)
            .expect("add");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "ping");
        assert!(got.calls[0].function.arguments.is_empty());
    }

    #[test]
    fn a_tool_call_with_no_content_around_it_still_parses() {
        let mut p = lfm2(false);
        let got = p
            .add("<|tool_call_start|>[check()]<|tool_call_end|>", true)
            .expect("add");
        assert!(got.content.is_empty());
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "check");
    }

    #[test]
    fn thinking_can_run_straight_into_a_tool_call_with_no_content_between() {
        let mut p = lfm2(true);
        let got = p
            .add(
                "<think>Let me run this command...</think><|tool_call_start|>[bash(command='ls')]<|tool_call_end|>",
                true,
            )
            .expect("add");
        assert_eq!(got.thinking, "Let me run this command...");
        assert!(got.content.is_empty());
        assert_eq!(got.calls.len(), 1);
    }

    #[test]
    fn thinking_then_content_then_a_tool_call_all_come_out_separately() {
        let mut p = lfm2(true);
        let got = p
            .add(
                r#"<think>Let me check the weather...</think>I'll get that for you.<|tool_call_start|>[get_weather(location="Paris")]<|tool_call_end|>"#,
                true,
            )
            .expect("add");
        assert_eq!(got.thinking, "Let me check the weather...");
        assert_eq!(got.content, "I'll get that for you.");
        assert_eq!(got.calls.len(), 1);
    }

    #[test]
    fn arguments_keep_the_order_the_model_wrote_them() {
        let mut p = lfm2(false);
        let got = p
            .add(
                "Searching.<|tool_call_start|>[search(query='beijing weather', language='zh')]<|tool_call_end|>",
                true,
            )
            .expect("add");
        let keys: Vec<&str> = got.calls[0]
            .function
            .arguments
            .0
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["query", "language"]);
    }

    /// Upstream `TestLFM2Parser_EdgeCases`, ported verbatim.
    #[test]
    fn lfm2_edge_cases_behave_exactly_as_upstream_pins_them() {
        let cases: &[(&str, &str, &str, &str, bool)] = &[
            // Only the FIRST `</think>` closes thinking; later ones are content.
            (
                "multiple_think_close_tags",
                "<think>First thought</think>Second thought</think>Final content",
                "Second thought</think>Final content",
                "First thought",
                true,
            ),
            (
                "empty_thinking_block",
                "<think></think>Just content",
                "Just content",
                "",
                true,
            ),
            (
                "direct_answer_with_leading_whitespace",
                "  \n  Hello there",
                "Hello there",
                "",
                true,
            ),
            // Thinking OFF: a stray `</think>` is just text, not framing.
            (
                "thinking_disabled_with_think_tags",
                "Some content</think>More content",
                "Some content</think>More content",
                "",
                false,
            ),
            // Thinking OFF means content mode from byte zero, so whitespace is
            // NOT trimmed -- there is no framing to strip.
            ("whitespace_only_content", "   \n\t   ", "   \n\t   ", "", false),
        ];

        for (name, input, want_content, want_thinking, has_thinking) in cases {
            let mut p = lfm2(*has_thinking);
            let got = p.add(input, true).expect("add");
            assert_eq!(&got.content, want_content, "content, case {name}");
            assert_eq!(&got.thinking, want_thinking, "thinking, case {name}");
        }
    }

    /// Upstream `TestLFM2Parser_BareToolCallFallback` -- no wrapper tags at all.
    #[test]
    fn a_bare_tool_call_naming_an_offered_tool_is_recovered_on_done() {
        let mut p = LFM2Parser::new(false);
        p.init(vec![tool("get_weather")], None, Some(&ThinkValue::Bool(false)));
        let got = p
            .add(r#"[get_weather(location="Paris")]"#, true)
            .expect("add");
        assert!(got.content.is_empty(), "content should be consumed by the call");
        assert!(got.thinking.is_empty());
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "get_weather");
    }

    /// Upstream `TestLFM2Parser_BareUnknownToolCallDoesNotParse` -- the guard
    /// rail. A call naming a tool nobody offered stays as content.
    #[test]
    fn a_bare_call_naming_an_unoffered_tool_stays_as_content() {
        let mut p = LFM2Parser::new(false);
        p.init(vec![tool("get_weather")], None, Some(&ThinkValue::Bool(false)));
        let input = r#"[unknown_tool(location="Paris")]"#;
        let got = p.add(input, true).expect("add");
        assert_eq!(got.content, input, "content must be preserved");
        assert!(got.calls.is_empty());
    }

    /// Upstream `TestLFM2Parser_ImagePlaceholdersPreserved`. `[img-0]...` looks
    /// bracket-y but has no `(`, so the fallback rejects it and the text
    /// survives. This is the failure mode the guard rail prevents.
    #[test]
    fn image_placeholders_are_never_mistaken_for_tool_calls() {
        for input in ["[img-0]describe this image", "<image>describe this image"] {
            let mut p = LFM2Parser::new(false);
            p.init(vec![tool("bash")], None, Some(&ThinkValue::Bool(false)));
            let got = p.add(input, true).expect("add");
            assert_eq!(got.content, input, "for {input:?}");
            assert!(got.thinking.is_empty());
            assert!(got.calls.is_empty());
        }
    }

    /// With NO tools offered the fallback accepts anything that parses -- there
    /// is nothing to check against.
    #[test]
    fn with_no_tools_offered_the_bare_call_fallback_does_not_run_at_all() {
        let mut p = lfm2(false);
        let got = p.add(r#"[anything(x=1)]"#, true).expect("add");
        // `has_tools` is false, so the fallback is skipped entirely and this
        // stays content -- NOT the same as "tool_names is empty".
        assert_eq!(got.content, "[anything(x=1)]");
        assert!(got.calls.is_empty());
    }

    /// `None` for `think` means OFF for this family, unlike qwen3.5 / cohere.
    #[test]
    fn an_unspecified_think_value_means_thinking_off_for_this_family() {
        let mut p = LFM2Parser::new(true);
        p.init(Vec::new(), None, None);
        let got = p.add("<think>reasoning</think>answer", true).expect("add");
        assert!(got.thinking.is_empty(), "None must not enable thinking");
        assert_eq!(got.content, "<think>reasoning</think>answer");
    }

    /// Upstream `TestLFM2Parser_Streaming`, ported verbatim as ground truth.
    ///
    /// Worth noticing what these fixtures do **not** contain: every one of them
    /// hands `<|tool_call_start|>` over as a **whole chunk**. See
    /// [`the_tool_call_start_tag_is_not_buffered_across_chunks`] for why that is
    /// not a coincidence.
    #[test]
    fn upstreams_streaming_fixtures_all_come_out_right() {
        /// (name, chunks, want_content, want_thinking, want_call_count,
        /// has_thinking).
        type Case<'a> = (&'a str, Vec<&'a str>, &'a str, &'a str, usize, bool);

        let cases: Vec<Case<'_>> = vec![
            ("streaming_simple_content", vec!["Hello, ", "how are ", "you?"], "Hello, how are you?", "", 0, false),
            (
                "streaming_thinking",
                vec!["<think>", "I need to ", "think about this", "...</think>", "The answer is 42."],
                "The answer is 42.",
                "I need to think about this...",
                0,
                true,
            ),
            ("streaming_direct_answer", vec!["The answer ", "is ", "42."], "The answer is 42.", "", 0, true),
            (
                "streaming_tool_call",
                vec!["I'll check weather.", "<|tool_call_start|>", "[get_weather(", "location=\"Paris\")]", "<|tool_call_end|>"],
                "I'll check weather.",
                "",
                1,
                false,
            ),
            (
                "streaming_thinking_with_partial_tag",
                vec!["<think>", "Thinking about this", "...</", "think>", "Done thinking."],
                "Done thinking.",
                "Thinking about this...",
                0,
                true,
            ),
            (
                "streaming_tool_call_with_split_python",
                vec!["Processing.", "<|tool_call_start|>", "[calc(", "x=42, ", "y=24)]", "<|tool_call_end|>"],
                "Processing.",
                "",
                1,
                false,
            ),
            // The OPENING `<think>` split across chunks IS handled -- that is
            // what `LookingForThinking` buffers for.
            (
                "streaming_thinking_split_open_tag",
                vec!["<th", "ink>", "reasoning", "</think>", "answer"],
                "answer",
                "reasoning",
                0,
                true,
            ),
            // A direct answer starting with `<` stays ambiguous until enough
            // arrives to rule out `<think>`.
            (
                "streaming_direct_answer_starting_with_angle",
                vec!["<", "html> is a tag"],
                "<html> is a tag",
                "",
                0,
                true,
            ),
            // Leading whitespace after `<think>` is trimmed even when it lands
            // in its own chunk -- that is what `needs_thinking_leading_trim` is.
            (
                "streaming_thinking_whitespace_after_tag",
                vec!["<think>", "\n\n  ", "Actual thinking content", "</think>", "Response"],
                "Response",
                "Actual thinking content",
                0,
                true,
            ),
            // ...and the same for whitespace after `</think>`.
            (
                "streaming_whitespace_after_close_tag",
                vec!["<think>Thinking</think>", "\n\n\n", "Response content"],
                "Response content",
                "Thinking",
                0,
                true,
            ),
        ];

        for (name, chunks, want_content, want_thinking, want_calls, has_thinking) in cases {
            let mut p = lfm2(has_thinking);
            let mut got = Parsed::default();
            for (i, chunk) in chunks.iter().enumerate() {
                let part = p.add(chunk, i == chunks.len() - 1).expect("add");
                got.content.push_str(&part.content);
                got.thinking.push_str(&part.thinking);
                got.calls.extend(part.calls);
            }
            assert_eq!(got.content, want_content, "content, case {name}");
            assert_eq!(got.thinking, want_thinking, "thinking, case {name}");
            assert_eq!(got.calls.len(), want_calls, "calls, case {name}");
        }
    }

    /// Thinking and content survive byte-at-a-time feeding, which is the part
    /// of the machine that actually buffers.
    #[test]
    fn thinking_and_content_fed_one_byte_at_a_time_give_the_same_answer() {
        let input = "<think>hmm let me see</think>Here you go lah.";

        let mut whole = lfm2(true);
        let want = whole.add(input, true).expect("add");

        let mut p = lfm2(true);
        let mut got = Parsed::default();
        for (i, ch) in input.char_indices() {
            let part = p
                .add(&input[i..i + ch.len_utf8()], i + ch.len_utf8() == input.len())
                .expect("add");
            got.content.push_str(&part.content);
            got.thinking.push_str(&part.thinking);
        }

        assert_eq!(got.thinking, want.thinking);
        assert_eq!(got.content, want.content);
        assert_eq!(got.thinking, "hmm let me see");
        assert_eq!(got.content, "Here you go lah.");
    }

    /// **A REAL UPSTREAM LIMITATION, pinned so nobody "fixes" it by accident.**
    ///
    /// `<|tool_call_start|>` gets **no partial-tag buffering**: the content
    /// state dumps its whole buffer as content every pass (see `eat`). So if the
    /// start tag is split across chunks -- one byte at a time being the extreme
    /// case -- it leaks into content and the tool call is never recognised.
    ///
    /// This port reproduces it deliberately. Every upstream streaming fixture
    /// delivers that tag as one whole chunk, which is exactly why upstream never
    /// noticed: a real tokeniser emits `<|tool_call_start|>` as a **single
    /// special token**, so it cannot split in practice.
    ///
    /// Fixing it would mean holding back `overlap(buf, TOOL_CALL_START_TAG)`
    /// bytes in the content state. That is a one-line change and it is
    /// **tempting and wrong** to make unilaterally: it would make this port and
    /// ollama disagree on the same byte stream, which is the one thing the
    /// golden-oracle rule forbids. Take it upstream first.
    #[test]
    fn the_tool_call_start_tag_is_not_buffered_across_chunks() {
        let input = r#"Checking.<|tool_call_start|>[f(x=1)]<|tool_call_end|>"#;

        // One chunk: works.
        let mut whole = lfm2(false);
        let want = whole.add(input, true).expect("add");
        assert_eq!(want.content, "Checking.");
        assert_eq!(want.calls.len(), 1);

        // One byte at a time: the tag leaks and NO call is found.
        let mut p = lfm2(false);
        let mut got = Parsed::default();
        for (i, ch) in input.char_indices() {
            let part = p
                .add(&input[i..i + ch.len_utf8()], i + ch.len_utf8() == input.len())
                .expect("add");
            got.content.push_str(&part.content);
            got.calls.extend(part.calls);
        }
        assert_eq!(got.content, input, "the whole thing leaks as content");
        assert!(got.calls.is_empty(), "and no tool call is recognised");
    }

    #[test]
    fn a_thinking_close_tag_split_across_chunks_never_leaks() {
        let mut p = lfm2(true);
        let a = p.add("<think>thought</thi", false).expect("add");
        assert_eq!(a.thinking, "thought");
        assert!(a.content.is_empty());
        let b = p.add("nk>visible", true).expect("add");
        assert!(b.thinking.is_empty());
        assert_eq!(b.content, "visible");
    }

    /// A partial `<think>` at end-of-stream can never complete, so it must be
    /// released as content rather than withheld forever.
    #[test]
    fn a_partial_think_tag_at_end_of_stream_is_released_as_content() {
        let mut p = lfm2(true);
        let a = p.add("<th", false).expect("add");
        assert!(a.content.is_empty(), "still ambiguous mid-stream, hold it");
        let b = p.add("", true).expect("drain");
        assert_eq!(b.content, "<th");
    }

    /// A paren inside a quoted argument must not close the call early.
    #[test]
    fn a_parenthesis_inside_a_quoted_argument_does_not_close_the_call() {
        let calls = parse_python_style_tool_calls(r#"[say(msg="smile :-)")]"#).expect("parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments.get("msg"), Some(&json!("smile :-)")));
    }

    /// The Python->JSON rewrite must not touch identifiers inside strings.
    #[test]
    fn the_python_to_json_rewrite_leaves_words_inside_strings_alone() {
        assert_eq!(
            python_literal_to_json("{'a': 'None of these', 'b': None}").expect("convert"),
            r#"{"a": "None of these", "b": null}"#
        );
        // A double quote inside a single-quoted string gets escaped, because
        // the string is about to become double-quoted.
        assert_eq!(
            python_literal_to_json(r#"{'msg': 'he said "hi"'}"#).expect("convert"),
            r#"{"msg": "he said \"hi\""}"#
        );
        assert!(python_literal_to_json("{'a': 'unterminated}").is_err());
    }

    #[test]
    fn a_body_with_no_parenthesis_is_rejected_rather_than_guessed_at() {
        assert!(parse_tool_calls_content("just some prose").is_err());
        assert!(parse_tool_calls_content("[img-0]describe this").is_err());
        assert!(parse_tool_calls_content("").is_err());
    }

    #[test]
    fn lfm2_advertises_its_tags_and_always_supports_tools() {
        let p = lfm2(false);
        assert_eq!(
            p.preserved_tokens(),
            vec!["<think>", "</think>", "<|tool_call_start|>", "<|tool_call_end|>"]
        );
        // Tools always; thinking only for the `lfm2-thinking` name.
        assert!(p.has_tool_support());
        assert!(!p.has_thinking_support());
        assert!(lfm2(true).has_thinking_support());
    }
}
