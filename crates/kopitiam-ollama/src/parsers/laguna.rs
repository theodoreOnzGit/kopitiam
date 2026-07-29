//! Laguna / poolside-v1 response parser.
//!
//! **Upstream:** `model/parsers/laguna.go` (ollama, MIT, Copyright (c) Ollama).
//! Ported against `4713800b08b2ddf5e14acf8398953cf7b12f169b` (2026-07-28).
//!
//! | parser name | struct | difference |
//! |---|---|---|
//! | `laguna` | [`LagunaParser`] | honours an assistant prefill |
//! | `poolside-v1` | [`LagunaV8Parser`] | **ignores** `last_message` entirely |
//!
//! ## The prompt primes the reasoning mode
//!
//! Upstream's own note, and it explains the whole opening dance: the prompt ends
//! with `<think>` when thinking is enabled (so the model's output **begins with
//! reasoning and no opening tag**), or `</think>` when it is not. **Thinking
//! defaults OFF**, matching the chat template -- so `None` means off here.
//!
//! ## Why `LagunaV8Parser` throws `last_message` away
//!
//! The v8 renderer closes any assistant history turn and emits a **fresh**
//! assistant generation prompt, instead of continuing the final assistant
//! message in place. So for v8 a trailing assistant message is history, not a
//! prefill -- honouring it would drop the parser into content mode when the
//! prompt actually primed `<think>`, and the reasoning would be reported as
//! content. That is the entire difference between the two names.
//!
//! ## THREE tool-call syntaxes, all live at once
//!
//! This is what makes laguna the fiddliest family in the port. It accepts:
//!
//! 1. **Tagged, `<arg_key>`/`<arg_value>` body** --
//!    `<tool_call>read<arg_key>path</arg_key><arg_value>/x</arg_value></tool_call>`
//! 2. **Tagged, JSON body** --
//!    `<tool_call>{"name":"read","arguments":{"path":"/x"}}</tool_call>`
//! 3. **Bare JSON with no tags at all** -- a `{"name":..., "arguments":...}`
//!    object sitting in the content stream, sniffed by
//!    [`looks_like_json_tool_call`]. **Only when tools were offered**, which is
//!    the guard that stops ordinary JSON in a code answer from being eaten.
//!
//! Plus a fourth entry point: a `<user>`...`</user>` block, which some
//! checkpoints wrap a call in ([`LagunaParser::parse_tool_alias`]) -- again only
//! when tools were offered, and only when the name resolves.
//!
//! ## Tool-name aliases are real and load-bearing
//!
//! [`resolve_tool_name`] maps `read_file`->`read`, `write_file`->`write`,
//! `edit_file`->`edit`, `web_fetch`->`webfetch`. These models were trained on
//! the long names but agent harnesses register the short ones. The alias is only
//! applied when a tool with the short name was actually offered -- so it can
//! never invent a tool that does not exist.

use crate::api::{Message, PropertyType, ThinkValue, Tool, ToolCall, ToolCallArguments, ToolCallFunction};

use super::qwen3coder::parse_value;
use super::{Parsed, Parser, ParserError, overlap, trailing_whitespace_len};

/// **Upstream:** the `laguna*Tag` consts.
const THINKING_OPEN_TAG: &str = "<think>";
const THINKING_CLOSE_TAG: &str = "</think>";
const TOOL_CALL_OPEN_TAG: &str = "<tool_call>";
const TOOL_CALL_CLOSE_TAG: &str = "</tool_call>";
const USER_OPEN_TAG: &str = "<user>";
const USER_CLOSE_TAG: &str = "</user>";
const ARG_KEY_OPEN_TAG: &str = "<arg_key>";
const ARG_KEY_CLOSE_TAG: &str = "</arg_key>";
const ARG_VALUE_OPEN_TAG: &str = "<arg_value>";
const ARG_VALUE_CLOSE_TAG: &str = "</arg_value>";

/// **Upstream:** `lagunaParserState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    Thinking,
    #[default]
    Content,
    Tool,
}

/// **Upstream:** `LagunaParser`.
#[derive(Debug, Default)]
pub struct LagunaParser {
    state: State,
    buffer: String,
    tools: Vec<Tool>,
    call_index: usize,
    /// Did the caller ask for thinking? Also gates whether thinking text is
    /// **reported** at all -- see [`LagunaParser::add`].
    thinking_enabled: bool,
    /// Always `!thinking_enabled`. Upstream keeps both as separate fields and
    /// then branches on each; kept so the ported conditions read like the Go.
    thinking_suppressed: bool,
    /// One-shot latch: strip at most one redundant leading `<think>` (the prompt
    /// already opened one, but some checkpoints echo it).
    allow_leading_think_open: bool,
    /// Still at the very start of content, so leading whitespace is framing left
    /// over from `</think>` and must be eaten.
    at_content_start: bool,
}

impl LagunaParser {
    /// **Upstream:** `(*LagunaParser).consumeThinking`.
    ///
    /// Thinking ends at `</think>` (consumed) **or** at `<tool_call>`
    /// (consumed too, and the state jumps straight to [`State::Tool`]) -- the
    /// model can go from reasoning into a call without closing the block.
    fn consume_thinking(&mut self, done: bool) -> (bool, String) {
        let mut acc = self.buffer.clone();

        if self.allow_leading_think_open {
            let trimmed = acc.trim_start().to_string();
            if let Some(after) = trimmed.strip_prefix(THINKING_OPEN_TAG) {
                // The model echoed the primed `<think>`; drop it.
                self.buffer = after.trim_start().to_string();
                self.allow_leading_think_open = false;
                return (true, String::new());
            }
            if THINKING_OPEN_TAG.starts_with(&trimmed) && !done {
                // Could still grow into `<think>`; hold it (minus leading
                // whitespace) and wait.
                self.buffer = trimmed;
                return (false, String::new());
            }
            // Reasoning starts here. Drop the leading whitespace the model emits
            // after the primed tag -- its trained format is `<think>\n...`.
            self.buffer = trimmed.clone();
            self.allow_leading_think_open = false;
            acc = trimmed;
        }

        if let Some(idx) = acc.find(THINKING_CLOSE_TAG) {
            let thinking = acc[..idx].trim_end().to_string();
            self.buffer = acc[idx + THINKING_CLOSE_TAG.len()..].trim_start().to_string();
            self.state = State::Content;
            return (true, thinking);
        }

        if let Some(idx) = acc.find(TOOL_CALL_OPEN_TAG) {
            let thinking = acc[..idx].trim_end().to_string();
            // NOTE: no `trim_start` here, unlike the `</think>` branch. The
            // tool-call body's own leading whitespace is handled downstream by
            // `clean_tool_call_raw`.
            self.buffer = acc[idx + TOOL_CALL_OPEN_TAG.len()..].to_string();
            self.state = State::Tool;
            return (true, thinking);
        }

        if done {
            // Stream ended mid-thought. Flush it and fall back to content mode.
            self.buffer.clear();
            self.state = State::Content;
            let acc = acc.trim_end().to_string();
            return (!acc.is_empty(), acc);
        }

        let overlap_len = overlap(&acc, THINKING_CLOSE_TAG).max(overlap(&acc, TOOL_CALL_OPEN_TAG));
        let trailing_len = trailing_whitespace_len(&acc);
        let keep = overlap_len.max(trailing_len);
        if keep > 0 && keep < acc.len() {
            let (emit, hold) = super::chop(&acc, acc.len() - keep);
            let (emit, hold) = (emit.to_string(), hold.to_string());
            self.buffer = hold;
            return (!emit.is_empty(), emit);
        }
        (false, String::new())
    }

    /// **Upstream:** `(*LagunaParser).consumeContent`.
    ///
    /// Returns `(progress, content, calls)`. Long because it is checking, in
    /// order: a `<think>` re-entry, a stray `</think>` to swallow, a
    /// `<tool_call>`, a `<user>`-wrapped call, then a bare JSON call.
    fn consume_content(&mut self, done: bool) -> Result<(bool, String, Vec<ToolCall>), ParserError> {
        if self.at_content_start {
            // Eat the leading whitespace the model emits before content -- its
            // trained format puts a newline after the primed/closed `</think>`.
            let trimmed = self.buffer.trim_start().to_string();
            self.buffer = trimmed;
            if self.buffer.is_empty() {
                return Ok((false, String::new(), Vec::new()));
            }
            self.at_content_start = false;
        }

        let acc = self.buffer.clone();

        // Upstream guards this with `thinkingEnabled || thinkingSuppressed`,
        // which is a tautology (`suppressed == !enabled`), so it always runs.
        // Kept unguarded here rather than ported as an always-true `if`.
        if let Some(idx) = acc.find(THINKING_OPEN_TAG) {
            let content = acc[..idx].to_string();
            self.buffer = acc[idx + THINKING_OPEN_TAG.len()..].trim_start().to_string();
            self.state = State::Thinking;
            self.allow_leading_think_open = false;
            return Ok((true, content, Vec::new()));
        }
        if !done {
            let overlap_len = overlap(&acc, THINKING_OPEN_TAG);
            if overlap_len > 0 && overlap_len < acc.len() {
                let (content, hold) = super::chop(&acc, acc.len() - overlap_len);
                let (content, hold) = (content.to_string(), hold.to_string());
                self.buffer = hold;
                return Ok((!content.is_empty(), content, Vec::new()));
            }
        }

        // A stray `</think>` at the head of content is swallowed, not shown.
        // Upstream writes this block TWICE -- once under `if thinkingEnabled`,
        // once under `if thinkingSuppressed`, with identical bodies. Since one
        // of the two always holds, it is written once here.
        {
            let trimmed = acc.trim_start().to_string();
            if let Some(after) = trimmed.strip_prefix(THINKING_CLOSE_TAG) {
                self.buffer = after.trim_start().to_string();
                return Ok((true, String::new(), Vec::new()));
            }
            if THINKING_CLOSE_TAG.starts_with(&trimmed) && !done {
                return Ok((false, String::new(), Vec::new()));
            }
        }

        if let Some(idx) = acc.find(TOOL_CALL_OPEN_TAG) {
            let content = acc[..idx].trim_end().to_string();
            self.buffer = acc[idx + TOOL_CALL_OPEN_TAG.len()..].to_string();
            self.state = State::Tool;
            return Ok((true, content, Vec::new()));
        }

        // `<user>`-wrapped call. Only when tools were offered -- otherwise a
        // `<user>` in ordinary prose would be swallowed.
        if let Some(idx) = acc.find(USER_OPEN_TAG)
            && !self.tools.is_empty()
        {
            let before = acc[..idx].trim_end().to_string();
            let after_open = &acc[idx + USER_OPEN_TAG.len()..];
            if let Some(close_idx) = after_open.find(USER_CLOSE_TAG) {
                let raw = after_open[..close_idx].to_string();
                if let Some(call) = self.parse_tool_alias(&raw) {
                    self.buffer = after_open[close_idx + USER_CLOSE_TAG.len()..]
                        .trim_start()
                        .to_string();
                    return Ok((true, before, vec![call]));
                }
            } else if !done {
                // The block is still arriving. Emit whatever came before it and
                // keep the `<user>` onwards buffered.
                if idx > 0 {
                    self.buffer = acc[idx..].to_string();
                    return Ok((true, before, Vec::new()));
                }
                return Ok((false, String::new(), Vec::new()));
            }
        }

        if !self.tools.is_empty()
            && let Some(outcome) = self.consume_standalone_json_tool(done)?
        {
            return Ok(outcome);
        }

        if done {
            self.buffer.clear();
            let acc = acc.trim_end().to_string();
            return Ok((!acc.is_empty(), acc, Vec::new()));
        }

        let mut overlap_len =
            overlap(&acc, TOOL_CALL_OPEN_TAG).max(overlap(&acc, USER_OPEN_TAG));
        overlap_len = overlap_len.max(overlap(&acc, THINKING_OPEN_TAG));
        if self.thinking_suppressed {
            overlap_len = overlap_len.max(overlap(&acc, THINKING_CLOSE_TAG));
        }
        let trailing_len = trailing_whitespace_len(&acc);
        let keep = overlap_len.max(trailing_len);
        if keep > 0 && keep < acc.len() {
            let (emit, hold) = super::chop(&acc, acc.len() - keep);
            let (emit, hold) = (emit.to_string(), hold.to_string());
            self.buffer = hold;
            return Ok((!emit.is_empty(), emit, Vec::new()));
        }
        if keep == 0 && !acc.is_empty() {
            self.buffer.clear();
            return Ok((true, acc, Vec::new()));
        }
        Ok((false, String::new(), Vec::new()))
    }

    /// Sniff for a bare `{"name":..., "arguments":...}` sitting in content.
    ///
    /// **Upstream:** `consumeStandaloneJSONTool`. Returns `None` when this is
    /// not a JSON tool call at all (so the caller carries on with its other
    /// checks), and `Some(...)` when it took charge.
    ///
    /// The `!done && !json.Valid(...)` branch is the streaming case: the object
    /// is still arriving, so emit whatever prose came before it and keep
    /// waiting rather than parsing half an object.
    #[allow(clippy::type_complexity)]
    fn consume_standalone_json_tool(
        &mut self,
        done: bool,
    ) -> Result<Option<(bool, String, Vec<ToolCall>)>, ParserError> {
        let acc = self.buffer.clone();
        let Some(json_idx) = acc.find('{') else {
            return Ok(None);
        };

        let before = acc[..json_idx].trim_end().to_string();
        let raw = acc[json_idx..].trim_start().to_string();
        if !looks_like_json_tool_call(&raw, done) {
            return Ok(None);
        }

        let complete = serde_json::from_str::<serde_json::Value>(raw.trim()).is_ok();
        if !done && !complete {
            if !before.is_empty() {
                self.buffer = acc[json_idx..].to_string();
                return Ok(Some((true, before, Vec::new())));
            }
            return Ok(Some((false, String::new(), Vec::new())));
        }

        let mut call = parse_tool_call(&raw, &self.tools)?;
        call.function.index = self.call_index;
        self.call_index += 1;
        self.buffer.clear();
        self.state = State::Content;
        Ok(Some((true, before, vec![call])))
    }

    /// A `<user>`-wrapped call, accepted only if its name resolves to an offered
    /// tool. **Upstream:** `parseToolAlias`.
    ///
    /// Every failure returns `None` rather than an error, on purpose: a `<user>`
    /// block that is not a tool call is ordinary content and must be left alone.
    fn parse_tool_alias(&mut self, raw: &str) -> Option<ToolCall> {
        let cleaned = clean_tool_call_raw(raw);
        let name = tool_call_name(&cleaned)?;
        resolve_tool_name(&name, &self.tools)?;
        let mut call = parse_tool_call(&cleaned, &self.tools).ok()?;
        call.function.index = self.call_index;
        self.call_index += 1;
        Some(call)
    }

    /// **Upstream:** `(*LagunaParser).consumeTool`.
    fn consume_tool(&mut self, done: bool) -> Result<(bool, Option<ToolCall>), ParserError> {
        let acc = self.buffer.clone();

        if let Some(idx) = acc.find(TOOL_CALL_CLOSE_TAG) {
            let raw = acc[..idx].to_string();
            self.buffer = acc[idx + TOOL_CALL_CLOSE_TAG.len()..].trim_start().to_string();
            self.state = State::Content;
            let mut call = parse_tool_call(&raw, &self.tools)?;
            call.function.index = self.call_index;
            self.call_index += 1;
            return Ok((true, Some(call)));
        }

        // Stream ended without `</tool_call>`. Best effort on what is there --
        // a truncated call is still worth reporting if it names a tool.
        if done && !acc.trim().is_empty() {
            self.buffer.clear();
            self.state = State::Content;
            let mut call = parse_tool_call(&acc, &self.tools)?;
            call.function.index = self.call_index;
            self.call_index += 1;
            return Ok((true, Some(call)));
        }

        Ok((false, None))
    }
}

impl Parser for LagunaParser {
    /// **Upstream:** `(*LagunaParser).Init`.
    ///
    /// `None` for `think` means **OFF** -- the chat template's default.
    fn init(
        &mut self,
        tools: Vec<Tool>,
        last_message: Option<&Message>,
        think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        self.tools = tools.clone();
        self.call_index = 0;
        self.buffer.clear();
        self.thinking_enabled = think.is_some_and(|t| t.enabled());
        self.thinking_suppressed = !self.thinking_enabled;
        self.at_content_start = true;

        // Any trailing assistant message is a prefill: the renderer continues
        // that turn in place and has ALREADY written the closing `</think>`, so
        // the model resumes with content. Note this checks only the role -- an
        // empty-content assistant message still counts, unlike most families.
        let assistant_prefill = last_message.is_some_and(|m| m.role == "assistant");

        if self.thinking_enabled && !assistant_prefill {
            self.state = State::Thinking;
            self.allow_leading_think_open = true;
        } else {
            self.state = State::Content;
            self.allow_leading_think_open = false;
        }

        tools
    }

    /// **Upstream:** `(*LagunaParser).Add`.
    ///
    /// Note thinking text is only **reported** when `thinking_enabled`. With
    /// thinking off, a `<think>` block the model emits anyway is parsed and then
    /// **thrown away** -- deliberate, so a suppressed reasoning block does not
    /// leak into the reply.
    fn add(&mut self, s: &str, done: bool) -> Result<Parsed, ParserError> {
        self.buffer.push_str(s);
        let mut out = Parsed::default();

        loop {
            let progress = match self.state {
                State::Thinking => {
                    let (progress, thinking) = self.consume_thinking(done);
                    if self.thinking_enabled {
                        out.thinking.push_str(&thinking);
                    }
                    progress
                }
                State::Content => {
                    let (progress, content, calls) = self.consume_content(done)?;
                    out.content.push_str(&content);
                    out.calls.extend(calls);
                    progress
                }
                State::Tool => {
                    let (progress, call) = self.consume_tool(done)?;
                    if let Some(call) = call {
                        out.calls.push(call);
                    }
                    progress
                }
            };
            if !progress {
                break;
            }
        }

        Ok(out)
    }

    fn preserved_tokens(&self) -> Vec<&'static str> {
        vec![
            THINKING_OPEN_TAG,
            THINKING_CLOSE_TAG,
            TOOL_CALL_OPEN_TAG,
            TOOL_CALL_CLOSE_TAG,
            USER_OPEN_TAG,
            USER_CLOSE_TAG,
            ARG_KEY_OPEN_TAG,
            ARG_KEY_CLOSE_TAG,
            ARG_VALUE_OPEN_TAG,
            ARG_VALUE_CLOSE_TAG,
        ]
    }

    fn has_tool_support(&self) -> bool {
        true
    }

    fn has_thinking_support(&self) -> bool {
        true
    }
}

/// The `poolside-v1` parser. **Upstream:** `LagunaV8Parser`.
///
/// Identical to [`LagunaParser`] except that [`Parser::init`] **drops
/// `last_message` on the floor**. See the module docs: the v8 renderer emits a
/// fresh generation prompt rather than continuing the last assistant turn, so a
/// trailing assistant message is history, not a prefill.
#[derive(Debug, Default)]
pub struct LagunaV8Parser {
    inner: LagunaParser,
}

impl Parser for LagunaV8Parser {
    fn init(
        &mut self,
        tools: Vec<Tool>,
        _last_message: Option<&Message>,
        think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        // `None`, always -- that is the whole point of this type.
        self.inner.init(tools, None, think)
    }

    fn add(&mut self, s: &str, done: bool) -> Result<Parsed, ParserError> {
        self.inner.add(s, done)
    }

    fn preserved_tokens(&self) -> Vec<&'static str> {
        self.inner.preserved_tokens()
    }

    fn has_tool_support(&self) -> bool {
        self.inner.has_tool_support()
    }

    fn has_thinking_support(&self) -> bool {
        self.inner.has_thinking_support()
    }
}

/// Does this look like a bare JSON tool call rather than ordinary JSON?
///
/// **Upstream:** `lagunaLooksLikeJSONToolCall`.
///
/// * must start `{`;
/// * contains `"name"` or `"arguments"` -> **yes**, decided;
/// * on `done` -> **no** (the object is complete and lacks both keys, so it is
///   just JSON);
/// * mid-stream -> a hopeful `{"` / `{\n` / `{\r\n` counts as a maybe, so the
///   parser waits for the keys instead of streaming the object out as content
///   and then being unable to take it back.
fn looks_like_json_tool_call(raw: &str, done: bool) -> bool {
    let trimmed = raw.trim_start();
    if !trimmed.starts_with('{') {
        return false;
    }
    if trimmed.contains("\"name\"") || trimmed.contains("\"arguments\"") {
        return true;
    }
    if done {
        return false;
    }
    trimmed.starts_with("{\"") || trimmed.starts_with("{\n") || trimmed.starts_with("{\r\n")
}

/// Map a model-emitted tool name onto an offered one, honouring the aliases.
///
/// **Upstream:** `lagunaResolveToolName`. Returns `None` when nothing matches --
/// and that `None` is a **guard**, not a failure: it is what stops a `<user>`
/// block or a stray JSON object from becoming a call to a tool nobody offered.
///
/// The alias table is upstream's, hard-coded: these checkpoints were trained on
/// the `*_file` long names while agent harnesses register the short ones. An
/// alias only applies when the short name is actually on offer.
fn resolve_tool_name(name: &str, tools: &[Tool]) -> Option<String> {
    if tools.iter().any(|t| t.function.name == name) {
        return Some(name.to_string());
    }

    let alias = match name {
        "read_file" => "read",
        "write_file" => "write",
        "edit_file" => "edit",
        "web_fetch" => "webfetch",
        _ => return None,
    };

    tools
        .iter()
        .any(|t| t.function.name == alias)
        .then(|| alias.to_string())
}

/// Strip stray `<tool_call>` / `</tool_call>` wrappers off a call body.
///
/// **Upstream:** `cleanLagunaToolCallRaw`. Tolerance for models that emit the
/// tags unpaired or doubled. The order matters and is upstream's:
///
/// 1. peel **every** leading `<tool_call>` (a `while`, not an `if`);
/// 2. cut at the first `</tool_call>`;
/// 3. if a `<tool_call>` still remains inside, prefer whatever came **before**
///    it; only if that is empty do we take what came after.
fn clean_tool_call_raw(raw: &str) -> String {
    let mut raw = raw.trim().to_string();

    while let Some(rest) = raw.strip_prefix(TOOL_CALL_OPEN_TAG) {
        raw = rest.trim().to_string();
    }

    if let Some(idx) = raw.find(TOOL_CALL_CLOSE_TAG) {
        raw = raw[..idx].trim().to_string();
    }

    if let Some(idx) = raw.find(TOOL_CALL_OPEN_TAG) {
        let before = raw[..idx].trim().to_string();
        if !before.is_empty() {
            return before;
        }
        raw = raw[idx + TOOL_CALL_OPEN_TAG.len()..].trim().to_string();
    }

    raw
}

/// Read just the tool name out of a call body, without parsing the arguments.
///
/// **Upstream:** `lagunaToolCallName`. Used by
/// [`LagunaParser::parse_tool_alias`] to decide whether a `<user>` block is even
/// a tool call before committing to it.
///
/// For a JSON body it reads `"name"`. For a tagged body the name is everything
/// up to the first of `<arg_key>`, `{`, or a newline -- in that order.
fn tool_call_name(raw: &str) -> Option<String> {
    let raw = clean_tool_call_raw(raw);

    if raw.starts_with('{') {
        let parsed: serde_json::Value = serde_json::from_str(&raw).ok()?;
        let name = parsed.get("name")?.as_str()?.trim().to_string();
        return (!name.is_empty()).then_some(name);
    }

    let name_end = raw
        .find(ARG_KEY_OPEN_TAG)
        .or_else(|| raw.find('{'))
        .or_else(|| raw.find(['\r', '\n']))
        .unwrap_or(raw.len());
    let name = raw[..name_end].trim().to_string();
    (!name.is_empty()).then_some(name)
}

/// Parse a laguna tool call in any of its shapes.
///
/// **Upstream:** `parseLagunaToolCall`.
///
/// * body starts `{` -> whole thing is `{"name":..., "arguments":{...}}`;
/// * otherwise the name runs to the first `<arg_key>` (or `{`), and what follows
///   is either a JSON argument object or a run of
///   `<arg_key>k</arg_key><arg_value>v</arg_value>` pairs.
///
/// The tagged-pair path types each value against the tool's declared schema via
/// [`parse_value`] -- so `<arg_value>3</arg_value>` becomes the number `3` when
/// the schema says integer and the string `"3"` when it says string. `anyOf`
/// branches are **unioned** into one candidate type list, matching upstream.
fn parse_tool_call(raw: &str, tools: &[Tool]) -> Result<ToolCall, ParserError> {
    let raw = clean_tool_call_raw(raw);

    if raw.starts_with('{') {
        #[derive(serde::Deserialize)]
        struct Json {
            #[serde(default)]
            name: String,
            #[serde(default)]
            arguments: ToolCallArguments,
        }
        let parsed: Json = serde_json::from_str(&raw)?;
        if parsed.name.is_empty() {
            return Err(ParserError::EmptyFunctionName);
        }
        let name = resolve_tool_name(&parsed.name, tools).unwrap_or(parsed.name);
        return Ok(ToolCall {
            function: ToolCallFunction {
                name,
                arguments: parsed.arguments,
                ..Default::default()
            },
            ..Default::default()
        });
    }

    let (name, args_text) = match raw.find(ARG_KEY_OPEN_TAG) {
        Some(idx) => (&raw[..idx], &raw[idx..]),
        None => match raw.find('{') {
            Some(idx) => (&raw[..idx], &raw[idx..]),
            None => (raw.as_str(), ""),
        },
    };
    let name = name.trim().to_string();
    let name = resolve_tool_name(&name, tools).unwrap_or(name);

    let matched_tool = tools.iter().find(|t| t.function.name == name);

    let mut arguments = ToolCallArguments::new();

    if args_text.trim().starts_with('{') {
        arguments = serde_json::from_str(args_text.trim())?;
        return Ok(ToolCall {
            function: ToolCallFunction {
                name,
                arguments,
                ..Default::default()
            },
            ..Default::default()
        });
    }

    for (key, value) in arg_pairs(args_text) {
        let mut param_type = PropertyType(Vec::new());
        if let Some(tool) = matched_tool
            && let Some(prop) = tool.function.parameters.property(&key)
        {
            if !prop.any_of.is_empty() {
                for branch in &prop.any_of {
                    param_type.0.extend(branch.prop_type.0.iter().cloned());
                }
            } else {
                param_type = prop.prop_type.clone();
            }
        }
        arguments.set(key, parse_value(&value, &param_type));
    }

    Ok(ToolCall {
        function: ToolCallFunction {
            name,
            arguments,
            ..Default::default()
        },
        ..Default::default()
    })
}

/// Pull every `<arg_key>k</arg_key> <arg_value>v</arg_value>` pair out of a body.
///
/// **Upstream:** `lagunaArgRE` --
/// `regexp.MustCompile("(?s)<arg_key>(.*?)</arg_key>\\s*<arg_value>(.*?)</arg_value>")`
/// scanned with `FindAllStringSubmatch`.
///
/// Hand-rolled rather than pulling in `regex`, and the details that matter:
///
/// * `(?s)` -- `.` matches newlines, so a **multi-line value is one value**.
///   That is why this scans for the literal close tag instead of stopping at a
///   line end.
/// * `.*?` is **non-greedy**, so each capture stops at the FIRST close tag, not
///   the last. Get this backwards and two arguments merge into one.
/// * `\s*` between the key and value tags allows the whitespace real models put
///   there.
/// * The key is trimmed, the **value is not** -- upstream trims only `match[1]`,
///   and [`parse_value`] does its own newline handling.
///
/// A malformed pair is skipped and scanning resumes at the next `<arg_key>`,
/// which is how the regex engine behaves too.
fn arg_pairs(s: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut pos = 0usize;

    while let Some(k_open_rel) = s[pos..].find(ARG_KEY_OPEN_TAG) {
        let k_open = pos + k_open_rel;
        let key_start = k_open + ARG_KEY_OPEN_TAG.len();

        let Some(k_close_rel) = s[key_start..].find(ARG_KEY_CLOSE_TAG) else {
            break;
        };
        let k_close = key_start + k_close_rel;
        let key = s[key_start..k_close].trim().to_string();

        let after_key = k_close + ARG_KEY_CLOSE_TAG.len();
        let value_region = s[after_key..].trim_start();
        let Some(rest) = value_region.strip_prefix(ARG_VALUE_OPEN_TAG) else {
            // No `<arg_value>` where one was required -- skip this key and look
            // for the next `<arg_key>`, exactly as the regex would.
            pos = key_start;
            continue;
        };

        let Some(v_close_rel) = rest.find(ARG_VALUE_CLOSE_TAG) else {
            break;
        };
        let value = rest[..v_close_rel].to_string();

        out.push((key, value));

        // Resume after the closing `</arg_value>`.
        let consumed_from_after_key = s.len() - after_key - value_region.len();
        pos = after_key
            + consumed_from_after_key
            + ARG_VALUE_OPEN_TAG.len()
            + v_close_rel
            + ARG_VALUE_CLOSE_TAG.len();
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ToolFunction, ToolFunctionParameters, ToolProperty};
    use indexmap::IndexMap;
    use serde_json::json;

    fn tool_with(name: &str, props: &[(&str, &str)]) -> Tool {
        let mut m: IndexMap<String, ToolProperty> = IndexMap::new();
        for (k, ty) in props {
            m.insert(
                (*k).to_string(),
                ToolProperty {
                    prop_type: PropertyType(vec![(*ty).to_string()]),
                    ..Default::default()
                },
            );
        }
        Tool {
            tool_type: "function".into(),
            function: ToolFunction {
                name: name.into(),
                parameters: ToolFunctionParameters {
                    param_type: "object".into(),
                    properties: Some(m),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn laguna(think: bool, tools: Vec<Tool>) -> LagunaParser {
        let mut p = LagunaParser::default();
        p.init(tools, None, Some(&ThinkValue::Bool(think)));
        p
    }

    #[test]
    fn with_thinking_off_everything_is_content() {
        let mut p = laguna(false, Vec::new());
        let got = p.add("Hello, how can I help?", true).expect("add");
        assert_eq!(got.content, "Hello, how can I help?");
        assert!(got.thinking.is_empty());
    }

    /// `None` means thinking OFF for this family -- the chat template's default.
    #[test]
    fn an_unspecified_think_value_means_thinking_off() {
        let mut p = LagunaParser::default();
        p.init(Vec::new(), None, None);
        assert!(!p.thinking_enabled);
        let got = p.add("just an answer", true).expect("add");
        assert_eq!(got.content, "just an answer");
    }

    /// The prompt primed `<think>`, so reasoning starts with NO opening tag.
    #[test]
    fn thinking_starts_with_no_opening_tag_because_the_prompt_primed_it() {
        let mut p = laguna(true, Vec::new());
        let got = p.add("let me reckon</think>\nThe answer.", true).expect("add");
        assert_eq!(got.thinking, "let me reckon");
        assert_eq!(got.content, "The answer.");
    }

    /// ...but a redundant echoed `<think>` is stripped, once.
    #[test]
    fn a_redundant_echoed_think_open_tag_is_stripped() {
        let mut p = laguna(true, Vec::new());
        let got = p
            .add("<think>\nlet me reckon</think>\nThe answer.", true)
            .expect("add");
        assert_eq!(got.thinking, "let me reckon");
        assert_eq!(got.content, "The answer.");
    }

    /// With thinking off, a `<think>` block the model emits anyway is parsed and
    /// then **thrown away** -- it must not leak into the reply.
    #[test]
    fn with_thinking_off_a_think_block_is_swallowed_not_leaked() {
        let mut p = laguna(false, Vec::new());
        let got = p
            .add("before<think>secret reasoning</think>after", true)
            .expect("add");
        assert!(got.thinking.is_empty(), "suppressed thinking must not be reported");
        assert!(!got.content.contains("secret reasoning"));
        assert!(!got.content.contains("<think>"));
        assert_eq!(got.content, "beforeafter");
    }

    /// A stray leading `</think>` in content is swallowed, not shown.
    #[test]
    fn a_stray_think_close_tag_at_the_head_of_content_is_swallowed() {
        let mut p = laguna(false, Vec::new());
        let got = p.add("</think>\nThe answer.", true).expect("add");
        assert_eq!(got.content, "The answer.");
    }

    /// Shape 1: tagged call with `<arg_key>` / `<arg_value>` pairs, typed
    /// against the tool's schema.
    #[test]
    fn a_tagged_call_with_arg_key_pairs_is_typed_against_the_schema() {
        let mut p = laguna(false, vec![tool_with("read", &[("path", "string"), ("line", "integer")])]);
        let got = p
            .add(
                "Reading.<tool_call>read<arg_key>path</arg_key><arg_value>/tmp/x</arg_value><arg_key>line</arg_key><arg_value>3</arg_value></tool_call>",
                true,
            )
            .expect("add");
        assert_eq!(got.content, "Reading.");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "read");
        assert_eq!(got.calls[0].function.arguments.get("path"), Some(&json!("/tmp/x")));
        // Typed as an integer by the schema, NOT the string "3".
        assert_eq!(got.calls[0].function.arguments.get("line"), Some(&json!(3)));
        assert_eq!(got.calls[0].function.index, 0);
    }

    /// Shape 2: tagged call whose body is JSON.
    #[test]
    fn a_tagged_call_with_a_json_body_is_parsed() {
        let mut p = laguna(false, vec![tool_with("read", &[("path", "string")])]);
        let got = p
            .add(r#"<tool_call>{"name":"read","arguments":{"path":"/tmp/x"}}</tool_call>"#, true)
            .expect("add");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "read");
        assert_eq!(got.calls[0].function.arguments.get("path"), Some(&json!("/tmp/x")));
    }

    /// Shape 3: bare JSON with no tags -- accepted only because tools were
    /// offered.
    #[test]
    fn a_bare_json_call_is_recognised_when_tools_were_offered() {
        let mut p = laguna(false, vec![tool_with("read", &[("path", "string")])]);
        let got = p
            .add(r#"{"name":"read","arguments":{"path":"/tmp/x"}}"#, true)
            .expect("add");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "read");
    }

    /// ...and with NO tools offered the very same bytes stay content. This is
    /// the guard that stops a JSON code answer from being eaten.
    #[test]
    fn a_bare_json_object_stays_content_when_no_tools_were_offered() {
        let mut p = laguna(false, Vec::new());
        let input = r#"{"name":"read","arguments":{"path":"/tmp/x"}}"#;
        let got = p.add(input, true).expect("add");
        assert!(got.calls.is_empty());
        assert_eq!(got.content, input);
    }

    /// A JSON object with neither `"name"` nor `"arguments"` is ordinary content
    /// even when tools ARE offered.
    #[test]
    fn ordinary_json_without_the_tool_call_keys_stays_content() {
        let mut p = laguna(false, vec![tool_with("read", &[("path", "string")])]);
        let input = r#"{"total": 42, "ok": true}"#;
        let got = p.add(input, true).expect("add");
        assert!(got.calls.is_empty());
        assert_eq!(got.content, input);
    }

    /// Shape 4: a `<user>`-wrapped call.
    #[test]
    fn a_user_wrapped_call_is_recognised_when_its_name_resolves() {
        let mut p = laguna(false, vec![tool_with("read", &[("path", "string")])]);
        let got = p
            .add(
                "<user>read<arg_key>path</arg_key><arg_value>/tmp/x</arg_value></user>",
                true,
            )
            .expect("add");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "read");
    }

    /// A `<user>` block whose name is NOT an offered tool is left as content.
    #[test]
    fn a_user_block_that_is_not_a_tool_call_is_left_alone() {
        let mut p = laguna(false, vec![tool_with("read", &[("path", "string")])]);
        let got = p.add("<user>hello there</user>", true).expect("add");
        assert!(got.calls.is_empty());
        assert!(got.content.contains("hello there"));
    }

    /// The alias table: the model says `read_file`, the harness offered `read`.
    #[test]
    fn the_long_tool_name_aliases_resolve_to_the_short_offered_ones() {
        let tools = vec![
            tool_with("read", &[]),
            tool_with("write", &[]),
            tool_with("edit", &[]),
            tool_with("webfetch", &[]),
        ];
        for (long, short) in [
            ("read_file", "read"),
            ("write_file", "write"),
            ("edit_file", "edit"),
            ("web_fetch", "webfetch"),
        ] {
            assert_eq!(resolve_tool_name(long, &tools).as_deref(), Some(short));
        }
        // An exact match always wins over the alias table.
        assert_eq!(resolve_tool_name("read", &tools).as_deref(), Some("read"));
        // ...and an alias only applies when the short name is actually offered.
        assert_eq!(resolve_tool_name("read_file", &[]), None);
        assert_eq!(resolve_tool_name("no_such_tool", &tools), None);
    }

    #[test]
    fn an_aliased_name_is_rewritten_in_the_emitted_call() {
        let mut p = laguna(false, vec![tool_with("read", &[("path", "string")])]);
        let got = p
            .add(
                r#"<tool_call>{"name":"read_file","arguments":{"path":"/x"}}</tool_call>"#,
                true,
            )
            .expect("add");
        assert_eq!(got.calls[0].function.name, "read", "the alias must be applied");
    }

    /// Thinking can run straight into a tool call with no `</think>` between.
    #[test]
    fn a_tool_call_ends_thinking_even_with_no_closing_think_tag() {
        let mut p = laguna(true, vec![tool_with("read", &[("path", "string")])]);
        let got = p
            .add(
                "I should look<tool_call>read<arg_key>path</arg_key><arg_value>/x</arg_value></tool_call>",
                true,
            )
            .expect("add");
        assert_eq!(got.thinking, "I should look");
        assert_eq!(got.calls.len(), 1);
    }

    #[test]
    fn two_tool_calls_in_a_row_are_both_found_and_indexed() {
        let mut p = laguna(false, vec![tool_with("read", &[("path", "string")])]);
        let got = p
            .add(
                "<tool_call>read<arg_key>path</arg_key><arg_value>/a</arg_value></tool_call><tool_call>read<arg_key>path</arg_key><arg_value>/b</arg_value></tool_call>",
                true,
            )
            .expect("add");
        assert_eq!(got.calls.len(), 2);
        assert_eq!(got.calls[0].function.index, 0);
        assert_eq!(got.calls[1].function.index, 1);
        assert_eq!(got.calls[1].function.arguments.get("path"), Some(&json!("/b")));
    }

    #[test]
    fn arguments_keep_the_order_the_model_wrote_them() {
        let mut p = laguna(false, vec![tool_with("f", &[("zebra", "string"), ("apple", "string")])]);
        let got = p
            .add(
                "<tool_call>f<arg_key>zebra</arg_key><arg_value>1</arg_value><arg_key>apple</arg_key><arg_value>2</arg_value></tool_call>",
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
        assert_eq!(keys, ["zebra", "apple"]);
    }

    /// The v8 difference: `poolside-v1` throws `last_message` away, so a
    /// trailing assistant message does NOT drop it out of thinking mode.
    #[test]
    fn poolside_v1_ignores_a_trailing_assistant_message_but_laguna_honours_it() {
        let last = Message::new("assistant", "partial");

        // Plain laguna: the prefill wins, so we start in content.
        let mut plain = LagunaParser::default();
        plain.init(Vec::new(), Some(&last), Some(&ThinkValue::Bool(true)));
        let got = plain.add("continuing</think>after", true).expect("add");
        assert!(
            got.thinking.is_empty(),
            "laguna must treat the assistant message as a prefill"
        );

        // v8: last_message is dropped, so we start in thinking.
        let mut v8 = LagunaV8Parser::default();
        v8.init(Vec::new(), Some(&last), Some(&ThinkValue::Bool(true)));
        let got = v8.add("reasoning</think>after", true).expect("add");
        assert_eq!(
            got.thinking, "reasoning",
            "poolside-v1 must ignore the trailing assistant message"
        );
        assert_eq!(got.content, "after");
    }

    /// Laguna counts an assistant message as a prefill on **role alone** --
    /// even with empty content, unlike most other families.
    #[test]
    fn an_empty_assistant_message_still_counts_as_a_prefill_for_laguna() {
        let mut p = LagunaParser::default();
        let last = Message::new("assistant", "");
        p.init(Vec::new(), Some(&last), Some(&ThinkValue::Bool(true)));
        assert_eq!(p.state, State::Content);
    }

    #[test]
    fn a_thinking_close_tag_split_across_chunks_never_leaks() {
        let mut p = laguna(true, Vec::new());
        let a = p.add("thought</thi", false).expect("add");
        assert_eq!(a.thinking, "thought");
        assert!(a.content.is_empty());
        let b = p.add("nk>visible", true).expect("add");
        assert!(b.thinking.is_empty());
        assert_eq!(b.content, "visible");
    }

    /// One character at a time must agree with one big chunk, tool call and all.
    #[test]
    fn feeding_one_character_at_a_time_gives_the_same_answer_as_one_big_chunk() {
        let input = "let me look</think>\nChecking.<tool_call>read<arg_key>path</arg_key><arg_value>/x</arg_value></tool_call>";
        let tools = vec![tool_with("read", &[("path", "string")])];

        let mut whole = laguna(true, tools.clone());
        let want = whole.add(input, true).expect("add");

        let mut p = laguna(true, tools);
        let mut got = Parsed::default();
        for (i, ch) in input.char_indices() {
            let part = p
                .add(&input[i..i + ch.len_utf8()], i + ch.len_utf8() == input.len())
                .expect("add");
            got.content.push_str(&part.content);
            got.thinking.push_str(&part.thinking);
            got.calls.extend(part.calls);
        }

        assert_eq!(got.thinking, want.thinking);
        assert_eq!(got.content, want.content);
        assert_eq!(got.thinking, "let me look");
        assert_eq!(got.content, "Checking.");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls.len(), want.calls.len());
        assert_eq!(got.calls[0].function.arguments.get("path"), Some(&json!("/x")));
    }

    /// The non-greedy captures: two pairs must stay two pairs, not merge.
    #[test]
    fn arg_pairs_are_captured_non_greedily_so_two_pairs_stay_two() {
        let pairs = arg_pairs(
            "<arg_key>a</arg_key><arg_value>1</arg_value><arg_key>b</arg_key><arg_value>2</arg_value>",
        );
        assert_eq!(
            pairs,
            vec![("a".to_string(), "1".to_string()), ("b".to_string(), "2".to_string())]
        );
    }

    /// `(?s)` -- a value may span newlines and stays ONE value.
    #[test]
    fn an_arg_value_may_span_multiple_lines() {
        let pairs = arg_pairs("<arg_key>code</arg_key><arg_value>line1\nline2\n</arg_value>");
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "code");
        assert_eq!(pairs[0].1, "line1\nline2\n");
    }

    /// `\s*` between the key and value tags.
    #[test]
    fn whitespace_between_the_key_and_value_tags_is_allowed() {
        let pairs = arg_pairs("<arg_key>a</arg_key>  \n  <arg_value>1</arg_value>");
        assert_eq!(pairs, vec![("a".to_string(), "1".to_string())]);
    }

    #[test]
    fn clean_tool_call_raw_peels_unpaired_and_doubled_wrappers() {
        assert_eq!(clean_tool_call_raw("  read  "), "read");
        assert_eq!(clean_tool_call_raw("<tool_call>read</tool_call>"), "read");
        // Doubled opener -- the `while` peels both.
        assert_eq!(clean_tool_call_raw("<tool_call><tool_call>read"), "read");
        // A second opener INSIDE: prefer what came before it.
        assert_eq!(clean_tool_call_raw("read<tool_call>junk"), "read");
    }

    #[test]
    fn tool_call_name_reads_both_body_shapes() {
        assert_eq!(tool_call_name(r#"{"name":"read"}"#).as_deref(), Some("read"));
        assert_eq!(
            tool_call_name("read<arg_key>path</arg_key><arg_value>/x</arg_value>").as_deref(),
            Some("read")
        );
        assert_eq!(tool_call_name("read\nrest").as_deref(), Some("read"));
        assert_eq!(tool_call_name(""), None);
        assert_eq!(tool_call_name(r#"{"name":""}"#), None);
    }

    #[test]
    fn looks_like_json_tool_call_only_commits_when_it_should() {
        assert!(looks_like_json_tool_call(r#"{"name":"f"}"#, true));
        assert!(looks_like_json_tool_call(r#"{"arguments":{}}"#, true));
        // Complete but with neither key -> not a call.
        assert!(!looks_like_json_tool_call(r#"{"total":1}"#, true));
        // Mid-stream, a hopeful opening counts as a maybe so we keep waiting.
        assert!(looks_like_json_tool_call("{\"", false));
        assert!(looks_like_json_tool_call("{\n", false));
        // Not JSON at all.
        assert!(!looks_like_json_tool_call("hello", false));
    }

    #[test]
    fn laguna_advertises_all_ten_of_its_tags() {
        let p = laguna(false, Vec::new());
        let toks = p.preserved_tokens();
        assert_eq!(toks.len(), 10);
        for t in ["<think>", "</think>", "<tool_call>", "</tool_call>", "<user>", "</user>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"] {
            assert!(toks.contains(&t), "missing {t}");
        }
        assert!(p.has_tool_support());
        assert!(p.has_thinking_support());
        // v8 forwards everything.
        let v8 = LagunaV8Parser::default();
        assert_eq!(v8.preserved_tokens().len(), 10);
        assert!(v8.has_tool_support());
        assert!(v8.has_thinking_support());
    }
}
