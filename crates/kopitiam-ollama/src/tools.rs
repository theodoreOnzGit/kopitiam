//! Tool-call parsing, plus a few small `server/` odds and ends.
//!
//! **Upstream:** six Go files, all inside
//! `crates/kopitiam-ai/vendor/ollama/` (ollama, MIT, Copyright (c) Ollama),
//! pinned at `4713800b08b2ddf5e14acf8398953cf7b12f169b`:
//!
//! | Section | Go source | What it is |
//! |---|---|---|
//! | 1 | `tools/tools.go` | the streaming tool-call [`Parser`] -- the main event |
//! | 2 | `tools/template.go` | [`parse_tag`], sniff out a model's tool-call tag from its chat template |
//! | 3 | `server/quantization.go` | quant-type validation + `llama-quantize` argv shaping |
//! | 4 | `server/logprob.go` | runner logprobs -> API logprobs |
//! | 5 | `server/model_recommendations.go` | which model can this machine run |
//! | 6 | `server/fixblobs.go` | the legacy `sha256:` -> `sha256-` rename |
//! | 7 | `server/inference_request_log.go` | the pure bits of the debug request log |
//!
//! ## Why six Go files all squeeze into one Rust file
//!
//! Because `lib.rs` is chope-d while the port runs in parallel -- nobody can add
//! a module. So all six land here behind big section banners. Each section names
//! its Go file, each function names its Go counterpart, and every threshold
//! names the upstream line it came from. If this file ever kena split, the
//! banners are the cut lines -- see the note at the bottom of section 7.
//!
//! ## What the tool-call parser actually for
//!
//! A model don't "call a tool". It emit **text**, and somewhere inside that text
//! got something that look like a call. Every model family shape it differently
//! -- Qwen wrap it in `<tool_call>...</tool_call>`, Mistral in
//! `[TOOL_CALLS] [...]`, DeepSeek in `<|tool▁calls▁begin|>`, and plenty of models
//! just emit a bare JSON object with nothing wrapping it at all.
//!
//! Two things make this harder than "just regex the output lah":
//!
//! 1. **It streams.** Tokens come a few bytes at a time, so the tag can kena cut
//!    in half across two chunks (`"<|too"` then `"l▁calls▁begin|>"`), and so can
//!    a tool name (`"say_hello"` then `"_world"`). Commit too early and you fire
//!    `say_hello` when the model meant `say_hello_world` -- wrong tool, wrong
//!    side effect, and nobody see any stack trace.
//! 2. **The tag can be `{`.** For models with no wrapper, the "tag" is just an
//!    opening brace, which also opens every ordinary JSON object the model might
//!    legitimately be writing about. So `{ fmt.Println("hello") }` inside a code
//!    answer must come back to the user as *content*, cannot kena eaten as a
//!    malformed tool call.
//!
//! The [`Parser`] is a three-state machine over those two problems. See
//! [`ToolsState`].
//!
//! ## Attribution
//!
//! ollama is MIT; MIT absorb into KOPITIAM's AGPL-3.0-only provided the notices
//! travel with the code -- see `docs/ACKNOWLEDGEMENTS.md`. This module is
//! **derivation**, not inspiration, and the per-function `Upstream:` lines say
//! exactly which Go symbol each Rust item came from.

use crate::api::{Tool, ToolCall, ToolCallArguments, ToolCallFunction};
use crate::gotmpl;
use crate::memory::{FileType, Kv};
use crate::routes::{Logprob, TokenLogprob};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

// ===========================================================================
// SECTION 1 -- tools/tools.go
//
// The streaming tool-call parser. ~404 lines of Go, and every awkward branch in
// it is there because some model out there does something silly.
// ===========================================================================

/// Where the [`Parser`] up to already. **Upstream:** `type toolsState int`
/// (`tools/tools.go:11`).
///
/// One-way street: `LookingForTag` -> `ToolCalling` -> `Done`. Once `Done`,
/// every later byte is content, full stop -- that is what stop a model that
/// wrote one tool call and then carried on chatting from having its chat
/// re-parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolsState {
    /// Still scanning for the tag. Everything before it is content for the user.
    #[default]
    LookingForTag,
    /// Tag already seen; the buffer kena chewed for complete calls.
    ToolCalling,
    /// Finished. Whatever come next go straight back to the user.
    Done,
}

/// The streaming tool-call parser. **Upstream:** `type Parser struct`
/// (`tools/tools.go:19`).
///
/// Feed it output chunks with [`Parser::add`]; it hand back any **complete**
/// tool calls plus the text that should go to the user. It never return a
/// half-parsed call, and it never hand the user text it still deciding about --
/// both of those are the whole point of buffering.
///
/// ### What would make this wrong
///
/// * Handing back a call before its closing brace arrive -- a caller would
///   invoke a tool with truncated arguments.
/// * Matching a tool name that is a **prefix** of a longer one when the buffer
///   end right there. `say_hello` is a prefix of `say_hello_world`; commit on
///   the short one and you run the wrong tool. [`find_tool`] hold back for
///   exactly this.
/// * Dropping the buffer on the floor when the tag turn out to be false alarm.
///   Anything buffered but not consumed MUST come back as content.
#[derive(Debug, Clone, Default)]
pub struct Parser {
    tag: String,
    tools: Vec<Tool>,

    state: ToolsState,
    /// Raw bytes, not a `String`, because upstream slice it at byte offsets and
    /// we must slice at the same ones. See [`bytes_to_string`] for why that is
    /// safe with valid UTF-8 input.
    buffer: Vec<u8>,
    n: usize,
}

impl Parser {
    /// Build a parser from a model's **chat template** plus the tools on offer.
    ///
    /// **Upstream:** `NewParser(tmpl *template.Template, tools []api.Tool)`
    /// (`tools/tools.go:37`).
    ///
    /// The tag kena sniffed out of the template by [`parse_tag`] -- see section
    /// 2 for why the template is the right place to ask.
    ///
    /// **Divergence:** takes [`gotmpl::Template`] rather than
    /// [`crate::template::Template`], because the tag live in the *raw* parse
    /// tree and `crate::template::Template` graft a `{{ .Response }}` onto its
    /// copy. If you holding one of those, do
    /// `gotmpl::Template::parse(t.raw())` -- the graft is irrelevant to tag
    /// sniffing, but re-parsing keep this function infallible.
    pub fn new(tmpl: &gotmpl::Template, tools: Vec<Tool>) -> Self {
        Self::with_tag(tools, parse_tag(tmpl))
    }

    /// Build a parser with the tag already decided.
    /// **Upstream:** `NewParserWithTag` (`tools/tools.go:41`).
    pub fn with_tag(tools: Vec<Tool>, tag: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            tools,
            state: ToolsState::LookingForTag,
            buffer: Vec::new(),
            n: 0,
        }
    }

    /// The tag being looked for. **Upstream:** `(*Parser).Tag()`
    /// (`tools/tools.go:32`).
    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// Bytes held back, not yet decided. **Upstream:** `(*Parser).GetBuffer()`
    /// (`tools/tools.go:27`).
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// How many calls came out so far -- this is the `Index` the next call will
    /// carry. **Upstream:** the unexported `p.n`.
    pub fn count(&self) -> usize {
        self.n
    }

    /// Current state, mostly for callers who want to know whether to keep
    /// feeding. **Upstream:** the unexported `p.state`.
    pub fn state(&self) -> ToolsState {
        self.state
    }

    /// Push one chunk of model output through.
    ///
    /// **Upstream:** `(*Parser).Add(s string) (calls []api.ToolCall, content string)`
    /// (`tools/tools.go:50`).
    ///
    /// Return `(complete calls, text for the user)`. Both can be empty in the
    /// same call -- that just mean "still buffering, ask me again later".
    ///
    /// ### The `{` / `[` special case (`tools.go:68`)
    ///
    /// When the tag is a bare brace or bracket, upstream only entertain a tool
    /// call if the **first non-whitespace byte of the whole response** is that
    /// brace. Model already said "Sure, let me check:" before the `{`? Then it
    /// is prose containing JSON, not a call -- state go straight to
    /// [`ToolsState::Done`] and everything, buffer included, come back as
    /// content. Without that guard, any model that write *about* JSON get its
    /// answer swallowed.
    pub fn add(&mut self, s: &str) -> (Vec<ToolCall>, String) {
        if self.state == ToolsState::Done {
            return (Vec::new(), s.to_string());
        }

        self.buffer.extend_from_slice(s.as_bytes());

        let mut content = String::new();

        if self.state == ToolsState::LookingForTag {
            let (i, found) = self.find_tag();

            match i {
                // Upstream's `i == -1`: nothing here can ever become a tag, so
                // the whole buffer safe to release.
                None => {
                    content = bytes_to_string(&self.buffer);
                    self.buffer.clear();
                }
                Some(i) => {
                    content = bytes_to_string(&self.buffer[..i]);
                    self.buffer.drain(..i);
                }
            }

            // tools.go:68 -- see the doc comment above.
            if (self.tag == "{" || self.tag == "[") && !content.trim().is_empty() {
                self.state = ToolsState::Done;
                let rest = bytes_to_string(&self.buffer);
                return (Vec::new(), content + &rest);
            }

            if !found {
                return (Vec::new(), content);
            }

            self.state = ToolsState::ToolCalling;
        }

        let mut calls = Vec::new();
        while let Some(call) = self.parse_tool_call() {
            calls.push(call);
        }

        if self.done() {
            self.state = ToolsState::Done;
            // Upstream OVERWRITE `content` here, not append (`tools.go:93`).
            // Only reachable on the `{` / `[` path, where whatever `content`
            // held was whitespace-only or empty -- so the overwrite quietly eat
            // leading whitespace. Faithful on purpose: appending instead would
            // emit a stray newline in front of the reply for every brace-tag
            // model.
            content = bytes_to_string(&self.buffer);
            self.buffer.clear();
        }

        (calls, content)
    }

    /// Anything still buffered that the user is owed.
    ///
    /// **Upstream:** `(*Parser).Content()` (`tools/tools.go:390`). Call it once
    /// the stream finish.
    ///
    /// Empty unless the tag is `{` or `[` **and** no call ever came out -- i.e.
    /// the model opened a brace, never finished a call, and the stream stopped.
    /// That buffered text is the model's real answer, cannot silently drop. For
    /// a wrapper tag like `<tool_call>` upstream return `""`: a dangling
    /// half-tag is model noise, not an answer.
    pub fn content(&self) -> String {
        if self.n > 0 {
            return String::new();
        }
        if self.tag == "{" || self.tag == "[" {
            return bytes_to_string(&self.buffer);
        }
        String::new()
    }

    /// Find the tag in the buffer, or a **partial** tag at its tail.
    ///
    /// **Upstream:** `(*Parser).findTag() (int, bool)` (`tools/tools.go:100`).
    ///
    /// Return `(where, complete)`:
    ///
    /// * `(Some(i), true)` -- the whole tag sit at byte `i`.
    /// * `(Some(i), false)` -- the buffer *end* with the first few bytes of the
    ///   tag, starting at `i`. Might become a tag once more arrive; hold back.
    /// * `(None, false)` -- upstream's `-1`. No tag, no prefix of one; the whole
    ///   buffer is releasable.
    ///
    /// The suffix scan run **longest first** (`for i := max; i > 0; i--`), so a
    /// buffer ending `"<tool_calls><tool_"` report the longest live prefix
    /// instead of the first byte that happen to match.
    fn find_tag(&self) -> (Option<usize>, bool) {
        let tag = self.tag.as_bytes();

        if let Some(i) = index_of(&self.buffer, tag) {
            return (Some(i), true);
        }

        let max = self.buffer.len().min(tag.len());
        for i in (1..=max).rev() {
            if self.buffer.ends_with(&tag[..i]) {
                return (Some(self.buffer.len() - i), false);
            }
        }
        (None, false)
    }

    /// Pull the next **complete** call off the front of the buffer.
    ///
    /// **Upstream:** `(*Parser).parseToolCall()` (`tools/tools.go:119`).
    ///
    /// `None` mean "not yet" -- either no tool name is committable, or its
    /// arguments object has not closed. Never `None`-and-consumed: the buffer
    /// only advance when a call genuinely come out.
    fn parse_tool_call(&mut self) -> Option<ToolCall> {
        // Scoped so the immutable borrow of `self.tools` / `self.buffer` is done
        // before the buffer kena drained below.
        let (name, mut end) = {
            let (tool, end) = find_tool(&self.tools, &self.buffer);
            (tool?.function.name.clone(), end)
        };

        let (args_map, i) = find_arguments(&name, &self.buffer);
        let args_map = args_map?;
        // Upstream take the LATER of the two ends (`tools.go:130`): the
        // arguments object usually close after the name, but with
        // `{"arguments": {...}, "name": "add"}` the name come last. Advancing by
        // the smaller one would re-scan the object and emit it twice.
        if i > end {
            end = i;
        }

        let mut arguments = ToolCallArguments::new();
        for (k, v) in args_map {
            arguments.set(k, v);
        }

        let call = ToolCall {
            id: String::new(),
            function: ToolCallFunction {
                index: self.n,
                name,
                arguments,
            },
        };

        self.n += 1;
        self.buffer.drain(..end);
        Some(call)
    }

    /// The tool-call block already closed or not?
    ///
    /// **Upstream:** `(*Parser).done()` (`tools/tools.go:352`).
    ///
    /// Only ever true for the `{` and `[` tags -- a wrapper tag like
    /// `<tool_call>` return `false` always, because upstream cannot tell a
    /// closing wrapper from more content and would rather keep listening.
    ///
    /// Brace counting is **string-aware**: a `}` inside `"a}b"` don't close
    /// anything, and `\` escape the next byte. Miss that and `{"note": "a}b"}`
    /// look finished one byte early, truncating the JSON.
    fn done(&self) -> bool {
        let (open, close) = match self.tag.as_str() {
            "{" => (b'{', b'}'),
            "[" => (b'[', b']'),
            _ => return false,
        };

        // i64, not usize: upstream's `count` is an `int` and CAN go negative on
        // a buffer that open with a closing brace.
        let mut count: i64 = 0;
        let mut in_string = false;
        let mut escaped = false;

        for &c in &self.buffer {
            if escaped {
                escaped = false;
                continue;
            }
            if c == b'\\' {
                escaped = true;
                continue;
            }
            if c == b'"' {
                in_string = !in_string;
                continue;
            }
            if in_string {
                continue;
            }
            if c == open {
                count += 1;
            } else if c == close {
                count -= 1;
                if count == 0 {
                    return true;
                }
            }
        }

        false
    }
}

/// Match the first tool name in the buffer, holding back on a partial one.
///
/// **Upstream:** `findTool(tools []api.Tool, buf []byte) (*api.Tool, int)`
/// (`tools/tools.go:153`).
///
/// Return `(tool, end-of-name offset)`.
///
/// Two rules, both load-bearing:
///
/// 1. **Bail on a live prefix.** If the buffer's tail is a strict prefix of any
///    tool's name, return `None` -- more bytes might extend it. That is what
///    stop `say_hello` firing when `say_hello_world` still on the way. Upstream
///    only scan back `len(longest tool name)` bytes from the end, since no
///    longer tail could still be a prefix.
/// 2. **Earliest wins; on a tie, longest wins** (`tools.go:190`). Both names
///    starting at the same offset means one is a prefix of the other, and the
///    longer one is what the model actually wrote.
pub fn find_tool<'a>(tools: &'a [Tool], buf: &[u8]) -> (Option<&'a Tool>, usize) {
    if buf.is_empty() {
        return (None, 0);
    }

    let longest = tools
        .iter()
        .map(|t| t.function.name.len())
        .max()
        .unwrap_or(0);

    // Rule 1 -- a partial name at the tail means "wait first".
    for i in 1..=buf.len().min(longest) {
        let tail = &buf[buf.len() - i..];
        for t in tools {
            let name = t.function.name.as_bytes();
            if tail.len() < name.len() && name.starts_with(tail) {
                return (None, 0);
            }
        }
    }

    // Rule 2 -- earliest match, longest on a tie.
    let mut found: Option<&Tool> = None;
    let mut start: Option<usize> = None;
    let mut end = 0usize;

    for t in tools {
        let name = t.function.name.as_bytes();
        let Some(pos) = index_of(buf, name) else {
            continue;
        };

        if let (Some(s), Some(f)) = (start, found) {
            if pos > s {
                continue;
            }
            if pos == s && name.len() <= f.function.name.len() {
                continue;
            }
        }

        found = Some(t);
        start = Some(pos);
        end = pos + name.len();
    }

    match found {
        Some(t) => (Some(t), end),
        None => (None, 0),
    }
}

/// A tool call's arguments, in the order the model emitted them.
///
/// See [`crate::api::ToolCallArguments`] for why the order is load-bearing.
pub type ArgsMap = IndexMap<String, serde_json::Value>;

/// Find the first thing in the buffer that look like arguments for `tool_name`.
///
/// **Upstream:** `findArguments(tool *api.Tool, buffer []byte) (map[string]any, int)`
/// (`tools/tools.go:210`). Takes the name rather than the whole `Tool` because
/// the name is all upstream read off it, and passing `&str` keep the borrow
/// checker out of [`Parser::parse_tool_call`]'s way.
///
/// Return `(args, offset of the closing brace)`. `(None, i)` with a real `i` is
/// a genuine upstream state: "found an object that name a tool, but its
/// arguments unusable" -- the caller must NOT emit a call from it.
///
/// ## How an object kena picked out
///
/// Byte scan tracking brace depth, ignoring braces inside JSON strings and
/// honouring `\` escapes -- the same string-awareness [`Parser::done`] need, and
/// for the same reason. At depth zero the candidate kena JSON-parsed; if it
/// don't parse, the scan reset and keep looking (a `{` in prose is not fatal).
///
/// Once an object parse, [`find_object`] hunt for the arguments in the four
/// shapes models actually emit:
///
/// * `{"name": "...", "arguments": {...}}` -- the common one;
/// * `{"name": "...", "parameters": {...}}` -- same idea, other key;
/// * either of those with the value **as a JSON string** instead of an object;
/// * `{"get_temperature": {...}}` -- the tool's own name as the key.
///
/// Anything else, and the whole parsed object come back as the arguments --
/// upstream's fallback for a model that emit a bare argument bag.
///
/// ## Known gap, upstream's own TODO (`tools.go:206`)
///
/// A call with the arguments key **omitted entirely** -- `{"name": "get_conditions"}`
/// for a tool whose parameters all optional -- is not recognised. Faithfully not
/// fixed here: diverging would make KOPITIAM emit calls ollama don't.
///
/// ## Deliberate divergence: iteration order
///
/// Go walk `map[string]any` in **randomised** order, so on an object with two
/// nested candidates upstream's answer is not reproducible run to run. We walk
/// in the model's emission order via [`IndexMap`], which is deterministic. That
/// is strictly better and cannot disagree on any single-candidate object --
/// which is every real one.
pub fn find_arguments(tool_name: &str, buffer: &[u8]) -> (Option<ArgsMap>, usize) {
    if buffer.is_empty() {
        return (None, 0);
    }

    let mut start: Option<usize> = None;
    let mut braces: i64 = 0;
    let mut in_string = false;
    let mut escaped = false;

    for i in 0..buffer.len() {
        let c = buffer[i];

        if escaped {
            escaped = false;
            continue;
        }
        if c == b'\\' {
            escaped = true;
            continue;
        }
        if c == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }

        if c == b'{' {
            if braces == 0 {
                start = Some(i);
            }
            braces += 1;
        } else if c == b'}' {
            braces -= 1;
            if braces == 0
                && let Some(s) = start
            {
                let object = &buffer[s..=i];

                let data: IndexMap<String, OrderedJson> = match serde_json::from_slice(object) {
                    Ok(d) => d,
                    Err(_) => {
                        // Not valid JSON -- keep looking. `{` turn up in prose
                        // and code all the time.
                        start = None;
                        continue;
                    }
                };

                let (args, found) = find_object(&data, tool_name);
                if found {
                    return (args, i);
                }

                // Upstream's fallback (`tools.go:305`): no recognisable
                // wrapper, so treat the object itself as the arguments.
                return (Some(to_args_map(&data)), i);
            }

            if braces < 0 {
                braces = 0;
            }
        }
    }

    (None, 0)
}

/// Depth-first hunt for an arguments object.
///
/// **Upstream:** the `findObject` closure inside `findArguments`
/// (`tools/tools.go:262`).
///
/// Return `(args, decided)`. `decided == true` with `args == None` is the "this
/// object name a tool but its arguments unusable -- stop looking" verdict, and
/// it matter: without it the walk would descend into a valid-looking sibling and
/// invent a call the model never made.
fn find_object(obj: &IndexMap<String, OrderedJson>, tool_name: &str) -> (Option<ArgsMap>, bool) {
    // An object carrying a "name" key IS the call, whatever else it hold. Note
    // upstream test presence, not type -- `{"name": 7, ...}` still count.
    if obj.contains_key("name") {
        if let Some(args) = find_map("arguments", obj) {
            return (Some(args), true);
        }
        if let Some(args) = find_map("parameters", obj) {
            return (Some(args), true);
        }
        return (None, true);
    }

    // `{"get_temperature": {...}}` -- the tool's own name as the key.
    if let Some(args) = find_map(tool_name, obj) {
        return (Some(args), true);
    }

    for (_k, v) in obj {
        match v {
            OrderedJson::Object(child) => {
                let (args, found) = find_object(child, tool_name);
                if found {
                    return (args, true);
                }
            }
            OrderedJson::Array(items) => {
                for item in items {
                    if let OrderedJson::Object(child) = item {
                        let (args, found) = find_object(child, tool_name);
                        if found {
                            return (args, true);
                        }
                    }
                }
            }
            OrderedJson::Other(_) => {}
        }
    }

    (None, false)
}

/// Read `obj[name]` as an arguments map, accepting a **stringified** one.
///
/// **Upstream:** the `findMap` closure (`tools/tools.go:263`).
///
/// Some models emit `"arguments": "{\"city\": \"Tokyo\"}"` -- the arguments as a
/// JSON *string* rather than a JSON object. Upstream unmarshal the string a
/// second time; if that fail it give up on this key entirely rather than hand
/// back a string where a map belong.
fn find_map(name: &str, obj: &IndexMap<String, OrderedJson>) -> Option<ArgsMap> {
    match obj.get(name)? {
        OrderedJson::Object(m) => Some(to_args_map(m)),
        OrderedJson::Other(v) => {
            let s = v.as_str()?;
            let inner: IndexMap<String, OrderedJson> = serde_json::from_str(s).ok()?;
            Some(to_args_map(&inner))
        }
        OrderedJson::Array(_) => None,
    }
}

/// JSON that remember the order its object keys arrived in.
///
/// `serde_json::Value` back its objects with a `BTreeMap` in this workspace (no
/// `preserve_order` feature -- checked in `Cargo.lock`), so parsing through it
/// would silently re-sort a model's argument keys. See
/// [`crate::api::ToolCallArguments`] for why that is not cosmetic: re-serialise
/// a tool call with the keys shuffled and anything hashing, caching, diffing or
/// replaying it disagree with the model's own output.
///
/// Nested values fall back to `serde_json::Value` once converted out, because
/// [`crate::api::ToolCallArguments`] only hold order at the top level anyway --
/// and so do upstream, whose nested `map[string]any` is unordered too.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum OrderedJson {
    Object(IndexMap<String, OrderedJson>),
    Array(Vec<OrderedJson>),
    Other(serde_json::Value),
}

impl OrderedJson {
    fn to_json(&self) -> serde_json::Value {
        match self {
            OrderedJson::Object(m) => {
                serde_json::Value::Object(m.iter().map(|(k, v)| (k.clone(), v.to_json())).collect())
            }
            OrderedJson::Array(a) => {
                serde_json::Value::Array(a.iter().map(OrderedJson::to_json).collect())
            }
            OrderedJson::Other(v) => v.clone(),
        }
    }
}

fn to_args_map(m: &IndexMap<String, OrderedJson>) -> ArgsMap {
    m.iter().map(|(k, v)| (k.clone(), v.to_json())).collect()
}

/// `bytes.Index`, but returning `Option`. Empty needle match at 0, same as Go.
fn index_of(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Go's `string(someBytes)`, as close as Rust can get.
///
/// A Go string can hold invalid UTF-8; a Rust `String` cannot, so this is lossy
/// where Go is not. **In practice it never lose anything**, and worth knowing
/// why: [`Parser::add`] only ever take `&str`, so the buffer is always valid
/// UTF-8, and every split point is a byte offset of either a tag occurrence, a
/// tool name occurrence, or an ASCII `}` -- all of which are necessarily
/// character boundaries inside valid UTF-8. Feed the buffer raw bytes some other
/// way and that guarantee gone already.
fn bytes_to_string(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

// ===========================================================================
// SECTION 2 -- tools/template.go
//
// Sniffing a model's tool-call tag out of its own chat template.
// ===========================================================================

/// Work out the tool-call tag from a model's chat template.
///
/// **Upstream:** `parseTag(tmpl *template.Template) string`
/// (`tools/template.go:15`).
///
/// ## Why the template know
///
/// The same template that tell a model how to *emit* a tool call also tell us
/// how to *read* one. A template with `{{if .ToolCalls}}<tool_call>...` is
/// literally saying "when got tool calls, they start with `<tool_call>`". So:
/// find the `{{if .ToolCalls}}` branch, take the first non-whitespace text
/// inside it, and that is the tag. No per-model table to maintain, and it work
/// for models nobody heard of yet.
///
/// ## The `{` fallback
///
/// No `.ToolCalls` branch, or no text inside it, give `"{"` -- meaning *"try
/// every JSON object as a possible tool call"*. That is a real strategy, not a
/// failure: plenty of models emit bare JSON with no wrapper at all. The cost is
/// the ambiguity [`Parser::add`] guard against with its first-non-whitespace
/// rule.
///
/// ## Two fiddly bits
///
/// * **Cut at the first `{`** (`template.go:38`). A template whose branch open
///   `{"name": "{{ .Function.Name }}"` has text starting `{"name": "` -- keeping
///   that as the tag would only ever match one exact model. Cutting at `{` leave
///   `""`, which become the `{` fallback: exactly right.
/// * **CRLF kena normalised first** (`template.go:35`). A template saved with
///   Windows line endings must sniff the same tag as the Unix copy, else the
///   same model behave differently depending on how its Modelfile kena saved.
pub fn parse_tag(tmpl: &gotmpl::Template) -> String {
    parse_tag_nodes(tmpl.nodes())
}

/// [`parse_tag`] straight off template source.
///
/// Upstream take a `*template.Template` and treat `nil` as `"{"`; Rust got no
/// nil template, so the equivalent degenerate case is empty source -- which
/// parse to an empty tree and give `"{"` anyway.
pub fn parse_tag_from_source(src: &str) -> Result<String, gotmpl::ParseError> {
    Ok(parse_tag_nodes(gotmpl::Template::parse(src)?.nodes()))
}

fn parse_tag_nodes(nodes: &[gotmpl::parse::Node]) -> String {
    let Some(then) = find_tool_call_node(nodes) else {
        return "{".to_string();
    };

    let Some(text) = find_text_node(then) else {
        return "{".to_string();
    };

    let tag = text.replace("\r\n", "\n");
    // Upstream `strings.Cut(tag, "{")` -- take everything before the first `{`.
    let tag = tag.split('{').next().unwrap_or("");
    let tag = tag.trim();

    if tag.is_empty() {
        "{".to_string()
    } else {
        tag.to_string()
    }
}

/// Find the `{{ if .ToolCalls }}` branch, returning its body.
///
/// **Upstream:** `findToolCallNode(nodes []parse.Node) *parse.IfNode`
/// (`tools/template.go:47`). Upstream hand back the whole `IfNode` but only ever
/// read `.List.Nodes` off it (`template.go:27`), so we return just the body.
///
/// Match on a **field** node whose identifier chain contain `ToolCalls`, so
/// `{{ if .ToolCalls }}` and `{{ if and .ToolCalls .X }}` both hit. Upstream
/// check `*parse.FieldNode` only, so a `$var`-routed reference don't match --
/// copied exactly, quirk and all.
///
/// Recurse through nested `if` / `range` / `with` bodies and their else-bodies,
/// because plenty of templates bury the tool branch inside a message loop.
fn find_tool_call_node(nodes: &[gotmpl::parse::Node]) -> Option<&[gotmpl::parse::Node]> {
    use gotmpl::parse::Node;

    for node in nodes {
        match node {
            Node::If {
                pipe,
                then,
                otherwise,
            } => {
                if pipe_mentions_tool_calls(pipe) {
                    return Some(then);
                }
                if let Some(r) = find_tool_call_node(then) {
                    return Some(r);
                }
                if let Some(r) = find_tool_call_node(otherwise) {
                    return Some(r);
                }
            }
            Node::Range {
                body, otherwise, ..
            }
            | Node::With {
                body, otherwise, ..
            } => {
                if let Some(r) = find_tool_call_node(body) {
                    return Some(r);
                }
                if let Some(r) = find_tool_call_node(otherwise) {
                    return Some(r);
                }
            }
            _ => {}
        }
    }
    None
}

/// **Upstream:** the `isToolCallsNode` closure (`tools/template.go:48`).
fn pipe_mentions_tool_calls(pipe: &gotmpl::parse::Pipeline) -> bool {
    use gotmpl::parse::Arg;
    pipe.cmds.iter().any(|cmd| {
        cmd.args.iter().any(|arg| match arg {
            Arg::Field(path) => path.iter().any(|p| p == "ToolCalls"),
            _ => false,
        })
    })
}

/// First non-whitespace literal text, stopping at the first template construct.
///
/// **Upstream:** `findTextNode(nodes []parse.Node) *parse.TextNode`
/// (`tools/template.go:97`).
///
/// The stopping rule is the subtle part and it is deliberate. An `if`, `range`,
/// `with` or action node **end the search** -- upstream return nil instead of
/// carrying on to the next sibling. So for
/// `{{if .ToolCalls}}{{range .ToolCalls}}{{ . }}{{end}}]{{end}}` the trailing `]`
/// never kena reached, and the tag fall back to `{`. Correct, because that `]`
/// *close* the block; it don't open it, and a closing marker make a useless tag.
///
/// Whitespace-only text nodes kena skipped, not returned -- templates full of
/// cosmetic newlines and none of them is a tag.
fn find_text_node(nodes: &[gotmpl::parse::Node]) -> Option<&str> {
    use gotmpl::parse::Node;

    for node in nodes {
        match node {
            Node::Text(t) => {
                if t.trim().is_empty() {
                    continue;
                }
                return Some(t);
            }
            Node::If {
                then, otherwise, ..
            } => {
                if let Some(t) = find_text_node(then) {
                    return Some(t);
                }
                if let Some(t) = find_text_node(otherwise) {
                    return Some(t);
                }
                return None;
            }
            Node::Range {
                body, otherwise, ..
            }
            | Node::With {
                body, otherwise, ..
            } => {
                if let Some(t) = find_text_node(body) {
                    return Some(t);
                }
                if let Some(t) = find_text_node(otherwise) {
                    return Some(t);
                }
                return None;
            }
            // Go's `case *parse.ActionNode: return nil`. `{{ $x := ... }}` is an
            // ActionNode upstream too, hence Assign landing here.
            Node::Action(_) | Node::Assign { .. } => return None,
            // `{{break}}` / `{{continue}}` are BreakNode/ContinueNode upstream
            // and fall off the end of the switch -- so the loop carry on.
            Node::Continue | Node::Break => {}
        }
    }
    None
}

// ===========================================================================
// SECTION 3 -- server/quantization.go
//
// SCOPE, stated plainly: this port the quantisation TYPE and ARGUMENT logic
// only. The actual requantisation upstream is `exec.Command("llama-quantize")`
// -- it shell out to a llama.cpp binary and then rewrite the output GGUF to put
// back tensors that binary drop. KOPITIAM got no GGUF *writer*, so
// `runLlamaQuantizeCommand`, `restoreEmbeddedCompatibilityTensors`,
// `addDefaultLlavaProjectorType` and `tensorFromFile` are NOT ported. Shelling
// out to a C++ binary would break the Pure Rust Core promise anyway; when
// KOPITIAM quantise, it will be with its own kernels.
//
// What IS here is everything that decide WHAT to quantise to and HOW -- the part
// that carry real knowledge and is verifiable without any model file.
// ===========================================================================

/// Env var telling `llama-quantize` whether to keep llama.cpp compatibility
/// tensors. **Upstream:** `llamaCppCompatEnv` (`server/quantization.go:30`).
pub const LLAMA_CPP_COMPAT_ENV: &str = "OLLAMA_LLAMA_CPP_COMPAT";

/// The pseudo-type meaning "don't requantise, just copy".
/// **Upstream:** the literal `"COPY"` (`server/quantization.go:50`).
pub const COPY_TYPE_NAME: &str = "COPY";

/// Something wrong with a requested quantisation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum QuantizeError {
    /// **Upstream:** `fmt.Errorf("unsupported quantization type: %v", newFileType)`
    /// (`server/quantization.go:37`).
    #[error("unsupported quantization type: {0}")]
    UnsupportedType(String),
}

/// Name the quantisation, or refuse it.
///
/// **Upstream:** the first three lines of `quantize()`
/// (`server/quantization.go:35`), which take `newFileType.String()` and bail
/// when it is empty.
///
/// **Note the upstream guard is dead code.** `FileType.String()`
/// (`fs/ggml/type.go:164`) has a `default: return "unknown"` -- it never return
/// `""`, so that check can never fire. The check that actually protect anything
/// is [`crate::memory::FileType::parse`], which only accept the narrow set
/// ollama willing to produce. Kept here anyway, rewritten against `"unknown"` so
/// it do what upstream *meant*: a file type we cannot name is a file type we
/// must not hand to a quantiser.
pub fn quantize_type_name(new_file_type: FileType) -> Result<&'static str, QuantizeError> {
    let name = new_file_type.name();
    if name == "unknown" {
        return Err(QuantizeError::UnsupportedType(format!(
            "FileType({})",
            new_file_type.0
        )));
    }
    Ok(name)
}

/// Build the `llama-quantize` command line.
///
/// **Upstream:** `llamaQuantizeArgs(arch string, newFileType fsggml.FileType, input, output, typeName string) []string`
/// (`server/quantization.go:277`).
///
/// This is the interesting one: it is where hard-won per-architecture knowledge
/// about *which tensors cannot kena squeezed* live. Every override below is
/// upstream's, with upstream's reasoning, and each one fire **only** for the two
/// K-quant recipes `Q4_K_S` and `Q4_K_M` -- because at Q8_0 the tensor already
/// at or above the floor being protected, and at F16/BF16 nothing kena quantised
/// at all.
///
/// | arch | tensor kept high | why (upstream's own note) |
/// |---|---|---|
/// | `qwen35`, `qwen35moe` | `blk.N.nextn.eh_proj.weight` -> `q8_0` | Qwen3.5 MTP projection combine hidden + embedding states for the draft layer (`quantization.go:283`) |
/// | `gemma3n` | `per_layer_token_embd.weight` -> `f16` | read on **every layer for every token**, not once at input like `token_embd`, so far more quality-sensitive (`quantization.go:294`) |
/// | `deepseek2` | `attn_{k_b,q_a,q_b,v_b,kv_a_mqa}.weight` -> `q8_0` | small, critical matrices in DeepSeek-V2 multi-head latent attention; match published `library/glm-4.7-flash` K-quants (`quantization.go:303`) |
/// | `glmocr`, `glm4` | `token_embd.weight`, `output.weight` -> `f16` | small multimodal OCR model; low-precision embeddings give degenerate text (`quantization.go:319`) |
///
/// Two details not to lose:
///
/// * The gemma3n regex is **anchored** (`^per_layer_token_embd\.weight$`) on
///   purpose -- `--token-embedding-type` would also bump plain `token_embd`,
///   which upstream don't want.
/// * `"COPY"` return immediately with no overrides at all
///   (`quantization.go:279`): nothing kena requantised, so nothing to protect,
///   and passing a `--tensor-type` would be lying about intent.
pub fn llama_quantize_args(
    arch: &str,
    new_file_type: FileType,
    input: &str,
    output: &str,
    type_name: &str,
) -> Vec<String> {
    let mut args = vec!["--allow-requantize".to_string()];

    if type_name == COPY_TYPE_NAME {
        args.push(input.to_string());
        args.push(output.to_string());
        args.push(type_name.to_string());
        return args;
    }

    let is_k_quant = new_file_type == FileType::Q4_K_S || new_file_type == FileType::Q4_K_M;

    if (arch == "qwen35" || arch == "qwen35moe") && is_k_quant {
        args.push("--tensor-type".to_string());
        args.push(r"^blk\.[0-9]+\.nextn\.eh_proj\.weight$=q8_0".to_string());
    }

    if arch == "gemma3n" && is_k_quant {
        args.push("--tensor-type".to_string());
        args.push(r"^per_layer_token_embd\.weight$=f16".to_string());
    }

    if arch == "deepseek2" && is_k_quant {
        for pat in [
            r"attn_k_b\.weight$=q8_0",
            r"attn_q_a\.weight$=q8_0",
            r"attn_q_b\.weight$=q8_0",
            r"attn_v_b\.weight$=q8_0",
            r"attn_kv_a_mqa\.weight$=q8_0",
        ] {
            args.push("--tensor-type".to_string());
            args.push(pat.to_string());
        }
    }

    if (arch == "glmocr" || arch == "glm4") && is_k_quant {
        for pat in [r"^token_embd\.weight$=f16", r"^output\.weight$=f16"] {
            args.push("--tensor-type".to_string());
            args.push(pat.to_string());
        }
    }

    args.push(input.to_string());
    args.push(output.to_string());
    args.push(type_name.to_string());
    args
}

/// Strip any existing `OLLAMA_LLAMA_CPP_COMPAT` and force it to `0`.
///
/// **Upstream:** `disableLlamaCppCompat(env []string) []string`
/// (`server/quantization.go:170`).
///
/// Note the **strip then append** shape: the forced `=0` land at the end, and
/// the entry's original position is not preserved. That is upstream's behaviour
/// and its tests assert the exact ordering, so not a detail to tidy away.
pub fn disable_llama_cpp_compat(env: &[String]) -> Vec<String> {
    let prefix = format!("{LLAMA_CPP_COMPAT_ENV}=");
    let mut out: Vec<String> = env
        .iter()
        .filter(|e| !e.starts_with(&prefix))
        .cloned()
        .collect();
    out.push(format!("{LLAMA_CPP_COMPAT_ENV}=0"));
    out
}

/// The environment `llama-quantize` should run under.
///
/// **Upstream:** `llamaQuantizeEnv(env []string, enableCompat bool) []string`
/// (`server/quantization.go:181`).
///
/// * `enable_compat == false` -> force `=0`, i.e. **validate the GGUF strictly**.
/// * `enable_compat == true` -> just remove any inherited setting and let
///   llama.cpp's own default stand.
///
/// Upstream pass `hasEmbeddedCompatibilityTensors(orig)` here
/// (`quantization.go:128`): a model already carrying compatibility tensors would
/// fail strict validation, so compat mode kena allowed for exactly those.
pub fn llama_quantize_env(env: &[String], enable_compat: bool) -> Vec<String> {
    if !enable_compat {
        return disable_llama_cpp_compat(env);
    }
    let prefix = format!("{LLAMA_CPP_COMPAT_ENV}=");
    env.iter()
        .filter(|e| !e.starts_with(&prefix))
        .cloned()
        .collect()
}

/// This tensor one of llama.cpp's embedded compatibility tensors or not?
///
/// **Upstream:** `isEmbeddedCompatibilityTensor(name string) bool`
/// (`server/create.go:925`) -- it live in `create.go`, but `quantization.go` is
/// its main consumer (lines 44, 128, 222, 231), so it kena ported alongside the
/// code that need it. If `crate::create` end up exporting its own, dedupe onto
/// that one; this is a five-line prefix test, not a design.
///
/// The prefixes are audio (`a.`), multimodal projector (`mm.`), multi-token
/// prediction (`mtp.`), speech (`s.`) and vision (`v.`). llama.cpp's **text**
/// model loader deliberately don't claim these, which is why upstream must put
/// them back by hand after a requantise.
pub fn is_embedded_compatibility_tensor(name: &str) -> bool {
    ["a.", "mm.", "mtp.", "s.", "v."]
        .iter()
        .any(|p| name.starts_with(p))
}

/// This CLIP projector need a `projector_type` filled in or not?
///
/// **Upstream:** `needsDefaultLlavaProjectorType(ggml *fsggml.GGML) bool`
/// (`server/quantization.go:55`).
///
/// True only for a `clip` architecture that got a vision encoder and state
/// **neither** `clip.projector_type` nor `clip.vision.projector_type`. Old LLaVA
/// projectors predate the key; upstream's fix is to write `"mlp"` in
/// (`quantization.go:88`), which is what LLaVA-1.5 actually used.
///
/// Note the asymmetric key lookup, and it is not an accident:
/// `kv.Bool("has_vision_encoder")` go through upstream's architecture-prefixing
/// accessor (so it read `clip.has_vision_encoder`), while
/// `kv["clip.projector_type"]` is a raw map index. [`Kv::boolean`] and
/// [`Kv::value`] mirror that split exactly.
///
/// The *fix* itself is not ported -- rewriting the key mean writing a GGUF, and
/// we got no writer. This predicate is the reusable half.
pub fn needs_default_llava_projector_type(kv: &Kv) -> bool {
    if kv.architecture() != "clip" || !kv.boolean("has_vision_encoder", false) {
        return false;
    }
    if kv.value("clip.projector_type").is_some() {
        return false;
    }
    if kv.value("clip.vision.projector_type").is_some() {
        return false;
    }
    true
}

// ===========================================================================
// SECTION 4 -- server/logprob.go
// ===========================================================================

/// One token's log probability as the **runner** report it.
///
/// **Upstream:** `llm.TokenLogprob` (`llm/server.go:270`). Live here rather than
/// in [`crate::routes`] because it is the inference runner's type, not the HTTP
/// API's -- [`crate::routes::TokenLogprob`] is the wire-facing one, and the
/// whole job of section 4 is turning the first into the second.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunnerTokenLogprob {
    /// The token text itself.
    pub token: String,
    /// Natural log of the probability. Always <= 0.
    pub logprob: f64,
}

/// A generated token's logprob plus its alternatives, as the runner report it.
/// **Upstream:** `llm.Logprob` (`llm/server.go:276`), which embed
/// `llm.TokenLogprob`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunnerLogprob {
    /// The token that was actually generated.
    #[serde(flatten)]
    pub token_logprob: RunnerTokenLogprob,
    /// The alternatives considered. Empty when the caller never asked.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_logprobs: Vec<RunnerTokenLogprob>,
}

/// Runner logprobs -> API logprobs.
///
/// **Upstream:** `toAPILogprobs(logprobs []llm.Logprob) []api.Logprob`
/// (`server/logprob.go:9`).
///
/// The only real work is filling in `bytes` -- see [`string_to_byte_ints`].
pub fn to_api_logprobs(logprobs: &[RunnerLogprob]) -> Vec<Logprob> {
    logprobs
        .iter()
        .map(|lp| Logprob {
            token_logprob: TokenLogprob {
                token: lp.token_logprob.token.clone(),
                bytes: string_to_byte_ints(&lp.token_logprob.token),
                logprob: lp.token_logprob.logprob,
            },
            top_logprobs: lp
                .top_logprobs
                .iter()
                .map(|t| TokenLogprob {
                    token: t.token.clone(),
                    bytes: string_to_byte_ints(&t.token),
                    logprob: t.logprob,
                })
                .collect(),
        })
        .collect()
}

/// A token's **raw UTF-8 bytes**, as integers.
///
/// **Upstream:** `stringToByteInts(s string) []int` (`server/logprob.go:34`).
///
/// This is an OpenAI-compatibility field, and its unit matter: these are
/// **bytes, not code points and not token ids**. A client need them because one
/// token can be a fragment of a multi-byte character -- the string `"é"` alone
/// is unrenderable if the model split it across two tokens, and only the byte
/// values let a client stitch the pieces back.
///
/// Empty string give an empty vec (Go return `nil`, which marshal as
/// `omitempty`-absent; `Vec::is_empty` + `skip_serializing_if` on
/// [`crate::routes::TokenLogprob::bytes`] produce the same wire bytes).
pub fn string_to_byte_ints(s: &str) -> Vec<i64> {
    s.as_bytes().iter().map(|&b| b as i64).collect()
}

// ===========================================================================
// SECTION 5 -- server/model_recommendations.go
//
// SCOPE: the recommendation DATA, its validation, and the refresh policy
// constants. Not ported: the `http.Client` refresh loop, the goroutine, the
// stale-while-revalidate cache and the `~/.ollama/cache` snapshot file -- all
// net + concurrency plumbing, and KOPITIAM is Offline First: the DEFAULT table
// is the offline answer, and it must work with zero network.
// ===========================================================================

/// One "you can run this model" entry.
///
/// **Upstream:** `api.ModelRecommendation` (`api/types.go:807`).
///
/// **Mind the units, three different things here:**
///
/// * `context_length` and `max_output_tokens` are **tokens**, not bytes;
/// * `vram_bytes` is **bytes**, and it is a *decimal* gigabyte in the defaults
///   (`12 * format.GigaByte` = 12_000_000_000, not 12 GiB) -- see
///   [`default_model_recommendations`];
/// * a **cloud** entry carry the two token limits and no VRAM; a **local** entry
///   carry VRAM and no token limits. That split is what
///   [`validate_model_recommendations`] enforce.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRecommendation {
    /// The model name, e.g. `"qwen3.5"` or `"qwen3.5:cloud"`.
    pub model: String,
    /// One line saying what it good for.
    pub description: String,
    /// **Tokens.** Cloud entries only.
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub context_length: i64,
    /// **Tokens.** Cloud entries only.
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub max_output_tokens: i64,
    /// **Bytes.** Local entries only. What the model want on the GPU.
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub vram_bytes: i64,
    /// A subscription tier gate, when the upstream service say so. Never
    /// synthesised -- see [`validate_model_recommendations`].
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub required_plan: String,
}

fn is_zero_i64(v: &i64) -> bool {
    *v == 0
}

/// **Upstream:** `api.ModelRecommendationsResponse` (`api/types.go:802`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRecommendationsResponse {
    /// The list, best first.
    pub recommendations: Vec<ModelRecommendation>,
}

/// Where upstream fetch the live list from.
/// **Upstream:** `modelRecommendationsURL` (`server/model_recommendations.go:23`).
///
/// Recorded, not used. KOPITIAM don't phone ollama.com -- the defaults below are
/// the whole answer offline, per Offline First.
pub const MODEL_RECOMMENDATIONS_URL: &str =
    "https://ollama.com/api/experimental/model-recommendations";

/// **Upstream:** `modelRecommendationsRefreshInterval = 4 * time.Hour`
/// (`server/model_recommendations.go:26`).
pub const REFRESH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(4 * 60 * 60);

/// **Upstream:** `modelRecommendationsFetchTimeout = 3 * time.Second`
/// (`server/model_recommendations.go:27`).
pub const FETCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

/// **Upstream:** `modelRecommendationsReadRefreshCooldown = 5 * time.Second`
/// (`server/model_recommendations.go:28`).
///
/// Throttle the stale-while-revalidate path: a burst of reads trigger at most
/// one background refresh per five seconds.
pub const READ_REFRESH_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(5);

/// Backoff ladder after consecutive refresh failures: 5 min, 15 min, 1 h, 4 h.
///
/// **Upstream:** `modelRecommendationsBackoffSteps`
/// (`server/model_recommendations.go:29`). Index with
/// `min(failures - 1, len - 1)` (`model_recommendations.go:175`), i.e. the last
/// step repeat forever.
pub const BACKOFF_STEPS: [std::time::Duration; 4] = [
    std::time::Duration::from_secs(5 * 60),
    std::time::Duration::from_secs(15 * 60),
    std::time::Duration::from_secs(60 * 60),
    std::time::Duration::from_secs(4 * 60 * 60),
];

/// The offline answer: what upstream ship when it never talk to the net.
///
/// **Upstream:** `defaultModelRecommendations`
/// (`server/model_recommendations.go:367`). Order is upstream's and kena
/// asserted by `TestModelRecommendationsDefaultOrder` -- it is a ranking, not a
/// set.
///
/// **The VRAM figures are DECIMAL gigabytes**, because upstream use
/// `format.GigaByte` = 1000^3 (`format/bytes.go:13`), not `GibiByte`. So
/// `gemma4` say 12_000_000_000 bytes, which is ~11.2 GiB of actual VRAM. Since
/// GPUs kena sold in decimal GB but reported in binary GiB, mixing the two is
/// the classic way to recommend a model that then don't fit. See
/// [`crate::format::human_bytes`] (decimal, for files) vs
/// [`crate::format::human_bytes2`] (binary, for memory) -- and note a VRAM
/// number should be *displayed* with the binary one even though it kena *stored*
/// as a decimal multiple here.
pub fn default_model_recommendations() -> Vec<ModelRecommendation> {
    const GB: i64 = crate::format::GIGABYTE as i64;
    vec![
        ModelRecommendation {
            model: "kimi-k2.6:cloud".into(),
            description:
                "State-of-the-art coding, long-horizon execution, and multimodal agent swarm capability"
                    .into(),
            context_length: 262_144,
            max_output_tokens: 262_144,
            ..Default::default()
        },
        ModelRecommendation {
            model: "glm-5.1:cloud".into(),
            description: "Reasoning and code generation".into(),
            context_length: 202_752,
            max_output_tokens: 131_072,
            ..Default::default()
        },
        ModelRecommendation {
            model: "qwen3.5:cloud".into(),
            description: "Reasoning, coding, and agentic tool use with vision".into(),
            context_length: 262_144,
            max_output_tokens: 32_768,
            ..Default::default()
        },
        ModelRecommendation {
            model: "minimax-m2.7:cloud".into(),
            description: "Fast, efficient coding and real-world productivity".into(),
            context_length: 204_800,
            max_output_tokens: 128_000,
            ..Default::default()
        },
        ModelRecommendation {
            model: "gemma4".into(),
            description: "Reasoning and code generation locally".into(),
            // model_recommendations.go:395 -- 12 * format.GigaByte
            vram_bytes: 12 * GB,
            ..Default::default()
        },
        ModelRecommendation {
            model: "qwen3.5".into(),
            description: "Reasoning, coding, and visual understanding locally".into(),
            // model_recommendations.go:400 -- 14 * format.GigaByte
            vram_bytes: 14 * GB,
            ..Default::default()
        },
    ]
}

/// A recommendation list that cannot kena trusted.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RecommendationError {
    /// **Upstream:** `errors.New("empty recommendations")`
    /// (`model_recommendations.go:316`).
    #[error("empty recommendations")]
    Empty,
    /// **Upstream:** `errors.New("recommendation missing model")`
    /// (`model_recommendations.go:327`).
    #[error("recommendation missing model")]
    MissingModel,
    /// **Upstream:** `fmt.Errorf("duplicate recommendation %q", rec.Model)`
    /// (`model_recommendations.go:330`).
    #[error("duplicate recommendation {0:?}")]
    Duplicate(String),
    /// **Upstream:** `errors.New("no valid recommendations")`
    /// (`model_recommendations.go:342`).
    #[error("no valid recommendations")]
    NoneValid,
}

/// Trim, de-duplicate and sanity-check a recommendation list.
///
/// **Upstream:** `validateModelRecommendations(recs []api.ModelRecommendation)`
/// (`server/model_recommendations.go:314`).
///
/// Note the two different severities -- that is the whole design:
///
/// * A **structural** problem -- empty list, blank model name, duplicate name --
///   reject the **whole list**. A list you cannot trust must not partially
///   replace one you could.
/// * A **cloud entry missing its token limits** drop just that entry
///   (`model_recommendations.go:334`) and the rest still land. Upstream log a
///   warning; we got no logger here, so it is silent -- if a caller need to
///   know, count the entries in versus out.
///
/// `required_plan` is **never synthesised** -- upstream got a whole test for
/// that (`TestValidateModelRecommendationsDoesNotSynthesizeRequiredPlans`).
/// Inventing a plan gate would tell a user they cannot run something they can.
pub fn validate_model_recommendations(
    recs: &[ModelRecommendation],
) -> Result<Vec<ModelRecommendation>, RecommendationError> {
    if recs.is_empty() {
        return Err(RecommendationError::Empty);
    }

    let mut seen = std::collections::HashSet::new();
    let mut valid = Vec::with_capacity(recs.len());

    for rec in recs {
        let rec = ModelRecommendation {
            model: rec.model.trim().to_string(),
            description: rec.description.trim().to_string(),
            required_plan: rec.required_plan.trim().to_string(),
            ..rec.clone()
        };

        if rec.model.is_empty() {
            return Err(RecommendationError::MissingModel);
        }
        if !seen.insert(rec.model.clone()) {
            return Err(RecommendationError::Duplicate(rec.model));
        }

        // A cloud model with no declared limits would let a caller send a
        // request the service must reject -- drop it rather than guess.
        if is_cloud_recommendation(&rec.model)
            && (rec.context_length <= 0 || rec.max_output_tokens <= 0)
        {
            continue;
        }

        valid.push(rec);
    }

    if valid.is_empty() {
        return Err(RecommendationError::NoneValid);
    }

    Ok(valid)
}

/// Cloud-hosted model or local one?
///
/// **Upstream:** `isCloudRecommendation(modelName string) bool`
/// (`server/model_recommendations.go:348`).
///
/// Purely a **name suffix** test -- `:cloud` or `-cloud`. Crude, and upstream
/// know it; a local model somebody name `my-cloud` would kena misread. Kept
/// exactly, because the alternative is a registry lookup and this must run
/// offline.
pub fn is_cloud_recommendation(model_name: &str) -> bool {
    model_name.ends_with(":cloud") || model_name.ends_with("-cloud")
}

/// Spread a retry so a whole fleet of clients don't stampede together.
///
/// **Upstream:** `withJitter(d time.Duration) time.Duration`
/// (`server/model_recommendations.go:352`). Range is **[0.8x, 1.2x]**
/// (`model_recommendations.go:356`), and non-positive durations pass through
/// untouched.
///
/// **Divergence:** upstream call `rand.Float64()` inside. This take the draw as
/// `unit` (expected in `[0, 1)`) instead, so the crate need no RNG dependency
/// and the function stay deterministic and testable. The caller own the
/// randomness -- and for jitter any decent source will do, unlike the registry
/// auth nonce, which need a real CSPRNG.
///
/// `unit` outside `[0, 1)` kena clamped rather than rejected: a bad draw must
/// not be able to push a retry outside the intended band.
pub fn with_jitter(d: std::time::Duration, unit: f64) -> std::time::Duration {
    if d.is_zero() {
        return d;
    }
    let unit = if unit.is_nan() { 0.0 } else { unit.clamp(0.0, 1.0) };
    let factor = 0.8 + unit * 0.4;
    d.mul_f64(factor)
}

// ---------------------------------------------------------------------------
// KOPITIAM ADDITION -- not upstream. Read this before trusting it.
// ---------------------------------------------------------------------------

/// What the caller know about the machine.
///
/// **NOT UPSTREAM.** Deliberately a plain input struct: this crate probe no
/// hardware and depend on nothing else in KOPITIAM (see
/// `docs/ai-decisions/AID-0055`), so whoever know the VRAM fill this in.
///
/// **Units: `vram_bytes` is BYTES.** Same unit as
/// [`ModelRecommendation::vram_bytes`], which is what make the comparison in
/// [`ModelRecommendation::fits`] meaningful. Pass GiB by accident and every
/// model will look like it fit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MachineFacts {
    /// Usable VRAM in **bytes**. Usable, not installed -- minus whatever the
    /// desktop compositor and other processes already holding, else the ladder
    /// will recommend a model that OOM on load.
    pub vram_bytes: i64,
    /// Whether cloud models can kena offered at all. `false` is KOPITIAM's
    /// Offline First default and the honest answer with no network.
    pub cloud_available: bool,
}

impl ModelRecommendation {
    /// This machine can run it or not?
    ///
    /// **NOT UPSTREAM** -- upstream ship `vram_bytes` as advice and leave the
    /// fit decision to its client. This is KOPITIAM's rule, stated out loud so
    /// somebody can argue with it:
    ///
    /// * A **cloud** entry ([`is_cloud_recommendation`]) need no VRAM and fit
    ///   iff [`MachineFacts::cloud_available`].
    /// * A **local** entry fit iff `machine.vram_bytes >= self.vram_bytes`.
    /// * A local entry with `vram_bytes == 0` -- nobody said what it cost --
    ///   fit. Refusing on missing data would hide models from every user the
    ///   moment one registry entry is incomplete.
    ///
    /// ### What would make this wrong
    ///
    /// `vram_bytes` is a **whole-model** figure at whatever quantisation the
    /// registry assumed. It say nothing about context length, and KV cache grow
    /// with context: a model that fit at 4k can OOM at 128k. It also assume full
    /// GPU offload -- partial offload let a bigger model run slowly instead of
    /// not at all, so a `false` here mean "won't fit entirely on the GPU", never
    /// "cannot be run". Use it to rank, not to forbid.
    pub fn fits(&self, machine: &MachineFacts) -> bool {
        if is_cloud_recommendation(&self.model) {
            return machine.cloud_available;
        }
        self.vram_bytes <= machine.vram_bytes
    }
}

/// Filter a recommendation list down to what this machine can actually run,
/// keeping upstream's ranking order.
///
/// **NOT UPSTREAM.** This is the entry point bead `bd-250` (the local model
/// ladder) want: hand it [`default_model_recommendations`] plus a
/// [`MachineFacts`] and get back the ladder for this box, best first. Order kena
/// preserved rather than re-sorted -- upstream's list is already a ranking, and
/// inventing our own ordering would throw away that judgement for nothing.
///
/// Return an empty vec when nothing fit. That is a real answer, not an error: a
/// 4 GB laptop with no network genuinely got nothing on this list.
pub fn recommend_for_machine(
    recs: &[ModelRecommendation],
    machine: &MachineFacts,
) -> Vec<ModelRecommendation> {
    recs.iter().filter(|r| r.fits(machine)).cloned().collect()
}

// ===========================================================================
// SECTION 6 -- server/fixblobs.go
// ===========================================================================

/// The repaired filename for a legacy blob, if it need repairing.
///
/// **Upstream:** the body of the `filepath.Walk` closure in
/// `fixBlobs(dir string) error` (`server/fixblobs.go:11`).
///
/// Split out as a pure function because it is the whole decision, and because
/// [`fix_blobs`] itself cannot kena exercised on Windows -- see there.
///
/// `sha256:abcd` -> `Some("sha256-abcd")`. Everything else -> `None`.
///
/// **Only the exact prefix `sha256` count.** `sha259:5678` kena left alone
/// (upstream test that), and so does anything already using `-`. The rename
/// exist because very old ollama stores wrote a `:` into the filename, which is
/// illegal on Windows and awkward everywhere -- and because only the first `:`
/// kena cut, a digest containing further colons cannot get mangled anyway.
pub fn fixed_blob_name(base_name: &str) -> Option<String> {
    let (typ, sha) = base_name.split_once(':')?;
    if typ != "sha256" {
        return None;
    }
    Some(format!("{typ}-{sha}"))
}

/// Walk `dir` and rename every legacy `sha256:...` entry to `sha256-...`.
///
/// **Upstream:** `fixBlobs(dir string) error` (`server/fixblobs.go:11`).
///
/// **Divergence, and a real one:** upstream rename *during* `filepath.Walk`,
/// mutating the tree it iterating. We collect every path first, then rename.
/// Same result, and it remove a genuine hazard -- a directory rename mid-walk
/// leave the walker holding a stale path on some platforms.
///
/// Directories kena renamed too, exactly like upstream: `filepath.Walk` visit
/// them and the closure don't skip them. Hence the deepest-first ordering below
/// -- rename a parent directory first and every child path collected under it is
/// already invalid.
///
/// **Windows note:** `:` is not a legal filename character on NTFS (it open an
/// alternate data stream), so a store that need this repair cannot exist on
/// Windows in the first place. Upstream's own test skip there. The function
/// still compile and run on Windows and is simply a no-op on any store that
/// could have been created there -- which is why [`fixed_blob_name`] is
/// separately testable.
pub fn fix_blobs(dir: &std::path::Path) -> std::io::Result<()> {
    let mut paths = Vec::new();
    collect_paths(dir, &mut paths)?;

    // Deepest first: a child must kena renamed before its parent directory move
    // out from under it.
    paths.sort_by_key(|p| std::cmp::Reverse(p.components().count()));

    for path in paths {
        let Some(base) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(fixed) = fixed_blob_name(base) else {
            continue;
        };
        let Some(parent) = path.parent() else {
            continue;
        };
        std::fs::rename(&path, parent.join(fixed))?;
    }

    Ok(())
}

fn collect_paths(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_paths(&path, out)?;
        }
        out.push(path);
    }
    Ok(())
}

// ===========================================================================
// SECTION 7 -- server/inference_request_log.go
//
// SCOPE, and it is most of the file: `inferenceRequestLogger` is a gin
// middleware that read the request body, call c.Next(), then write two files and
// four slog lines. The gin handler, the `os.MkdirTemp` lifecycle, the atomic
// counter and every slog call are HTTP + logging plumbing with no logic worth
// porting -- KOPITIAM got neither gin nor this crate's own logger.
//
// Two pure pieces are genuinely reusable and kena ported below. The rest is
// deliberately skipped, and this note is the record of that decision.
// ===========================================================================

/// Squash a route path into something safe as a filename.
///
/// **Upstream:** `sanitizeRouteForFilename(route string) string`
/// (`server/inference_request_log.go:127`).
///
/// Leading `/` kena dropped; `/` alone become `"root"`; then **every** character
/// that is not `[A-Za-z0-9]` become `_`. Aggressive on purpose -- the output
/// must be a legal filename on Windows, macOS and Linux all at once, and the
/// shortest rule satisfying all three is "alphanumerics only".
///
/// Consequence worth knowing: `/api/chat` and `/api-chat` both give `api_chat`.
/// Upstream don't care because the filename kena prefixed with a nanosecond
/// timestamp plus a counter, so collisions cannot happen anyway.
///
/// **Divergence:** upstream range over **runes** and test each against ASCII
/// bounds, so one non-ASCII character become one `_`. We iterate `char`s the
/// same way, so `é` give one `_`, not two. Same as Go; noted because iterating
/// bytes instead would silently differ.
pub fn sanitize_route_for_filename(route: &str) -> String {
    let route = route.strip_prefix('/').unwrap_or(route);
    if route.is_empty() {
        return "root".to_string();
    }
    route
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// The `sh` script that replay a logged request.
///
/// **Upstream:** the `curl` format string at
/// `server/inference_request_log.go:118`.
///
/// The `SCRIPT_DIR` dance is the point: the body file kena referenced **relative
/// to the script**, so the whole log directory can kena moved or copied to
/// another machine and the replay still work. `CDPATH=` kena cleared first
/// because a `CDPATH` set in the user's environment make `cd` print the resolved
/// path to stdout, which would corrupt the command substitution.
///
/// `url` and the `Content-Type` header go through Go's `%q` -- shell-quoted with
/// escapes. [`go_quote`] reproduce it.
pub fn replay_curl_script(
    method: &str,
    url: &str,
    content_type: &str,
    body_filename: &str,
) -> String {
    format!(
        "#!/bin/sh\nSCRIPT_DIR=\"$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\"\ncurl --request {} --url {} --header {} --data-binary @\"${{SCRIPT_DIR}}/{}\"\n",
        method,
        go_quote(url),
        go_quote(&format!("Content-Type: {content_type}")),
        body_filename,
    )
}

/// Go's `%q` for a plain string: double-quoted, with `\` and `"` escaped.
///
/// Narrow on purpose -- upstream only ever apply it to a URL and a MIME type,
/// neither of which contain control characters, so the full Go escape table
/// (`\n`, `\t`, `\x..`, `\u....`) is not reproduced. If this ever kena pointed at
/// arbitrary text, that table must come with it.
fn go_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

// ===========================================================================
// TESTS
//
// Ported from upstream's own tables: tools/tools_test.go (TestParser, TestDone,
// TestContent, TestFindTag, TestFindArguments), tools/template_test.go
// (TestParseTag), server/quantization_test.go (TestLlamaQuantizeArgs,
// TestDisableLlamaCppCompat, TestLlamaQuantizeEnv),
// server/model_recommendations_test.go (the default-order + validate tests) and
// server/fixblobs_test.go (TestFixBlobs).
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ToolFunction, ToolFunctionParameters};
    use serde_json::json;

    // -- helpers -----------------------------------------------------------

    fn tool(name: &str) -> Tool {
        Tool {
            tool_type: "function".into(),
            items: None,
            function: ToolFunction {
                name: name.into(),
                description: String::new(),
                parameters: ToolFunctionParameters::default(),
            },
        }
    }

    /// The exact tool set from upstream's `TestParser` (`tools_test.go:60`).
    /// The overlapping names are the point: `say_hello` is a prefix of
    /// `say_hello_world`.
    fn test_tools() -> Vec<Tool> {
        [
            "get_temperature",
            "get_conditions",
            "say_hello",
            "say_hello_world",
            "get_address",
            "add",
        ]
        .iter()
        .map(|n| tool(n))
        .collect()
    }

    const QWEN: &str = r#"{{if .ToolCalls}}<tool_call>{{range .ToolCalls}}{"name": "{{.Function.Name}}", "arguments": {{.Function.Arguments}}}{{end}}</tool_call>{{end}}"#;
    const DEEPSEEK: &str = "{{if .ToolCalls}}<|tool▁calls▁begin|>{{range .ToolCalls}}<|tool▁call▁begin|>function<|tool▁sep|>get_current_weather\n```json\n{\"location\": \"Tokyo\"}\n```<|tool▁call▁end|>{{end}}<|tool▁calls▁end|><|end▁of▁sentence|>{{end}}";
    const JSON_TMPL: &str = r#"{{if .ToolCalls}}{{range .ToolCalls}}{"name": "{{.Function.Name}}", "arguments": {{.Function.Arguments}}}{{end}}{{end}}"#;
    const MISTRAL: &str = r#"{{if .ToolCalls}}[TOOL_CALLS] [{{range .ToolCalls}}{"name": "{{.Function.Name}}", "arguments": {{.Function.Arguments}}}{{end}}][/TOOL_CALLS]{{end}}"#;
    const LIST: &str = r#"{{if .ToolCalls}}[{{range .ToolCalls}}{"name": "{{.Function.Name}}", "arguments": {{.Function.Arguments}}}{{end}}]{{end}}"#;

    /// One upstream `TestParser` row: feed the chunks, collect calls + content.
    fn run_parser(src: &str, inputs: &[&str]) -> (Vec<ToolCall>, String) {
        let tmpl = gotmpl::Template::parse(src).expect("template parses");
        let mut p = Parser::new(&tmpl, test_tools());
        let mut calls = Vec::new();
        let mut content = String::new();
        for input in inputs {
            let (c, s) = p.add(input);
            calls.extend(c);
            content.push_str(&s);
        }
        (calls, content)
    }

    fn args(pairs: &[(&str, serde_json::Value)]) -> ToolCallArguments {
        let mut a = ToolCallArguments::new();
        for (k, v) in pairs {
            a.set(*k, v.clone());
        }
        a
    }

    fn expect_call(calls: &[ToolCall], i: usize, name: &str, want: &ToolCallArguments) {
        let c = &calls[i];
        assert_eq!(c.function.index, i, "call {i} index");
        assert_eq!(c.function.name, name, "call {i} name");
        // Compare by value, order-insensitively -- upstream's own `argsComparer`
        // (tools_test.go:13) do the same, because a Go map got no order to
        // compare in the first place. Order is asserted separately below.
        let got: std::collections::BTreeMap<_, _> = c.function.arguments.0.iter().collect();
        let expect: std::collections::BTreeMap<_, _> = want.0.iter().collect();
        assert_eq!(got, expect, "call {i} arguments");
    }

    // -- section 1: tools/tools.go -- TestParser ---------------------------

    #[test]
    fn plain_text_comes_straight_back_as_content() {
        let (calls, content) = run_parser(QWEN, &["Hello, how can I help you today?"]);
        assert!(calls.is_empty());
        assert_eq!(content, "Hello, how can I help you today?");
    }

    #[test]
    fn empty_input_produces_nothing() {
        let (calls, content) = run_parser(QWEN, &[""]);
        assert!(calls.is_empty());
        assert_eq!(content, "");
    }

    #[test]
    fn a_wrapped_tool_call_is_parsed_and_its_wrapper_is_not_content() {
        let (calls, content) = run_parser(
            QWEN,
            &[r#"<tool_call>{"name": "get_conditions", "arguments": {"location": "San Francisco"}}</tool_call>"#],
        );
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        expect_call(
            &calls,
            0,
            "get_conditions",
            &args(&[("location", json!("San Francisco"))]),
        );
    }

    #[test]
    fn empty_arguments_still_make_a_call() {
        let (calls, content) = run_parser(
            QWEN,
            &[r#"<tool_call>{"name": "get_conditions", "arguments": {}}</tool_call>"#],
        );
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        expect_call(&calls, 0, "get_conditions", &ToolCallArguments::new());
    }

    #[test]
    fn text_before_a_tool_call_is_kept_as_content() {
        let (calls, content) = run_parser(
            QWEN,
            &[r#"Let me check the weather. <tool_call>{"name": "get_temperature", "arguments": {"city": "New York"}}</tool_call>"#],
        );
        assert_eq!(content, "Let me check the weather. ");
        assert_eq!(calls.len(), 1);
        expect_call(
            &calls,
            0,
            "get_temperature",
            &args(&[("city", json!("New York"))]),
        );
    }

    /// Naming a tool in prose is not calling it -- no tag, no call.
    #[test]
    fn mentioning_a_tool_without_the_tag_is_just_text() {
        let (calls, content) = run_parser(
            QWEN,
            &["Let me say hello to the user. I'll use the say_hello tool. "],
        );
        assert!(calls.is_empty());
        assert_eq!(
            content,
            "Let me say hello to the user. I'll use the say_hello tool. "
        );
    }

    #[test]
    fn a_mistral_style_list_yields_two_indexed_calls() {
        let (calls, content) = run_parser(
            MISTRAL,
            &[r#"[TOOL_CALLS] [{"name": "get_temperature", "arguments": {"city": "London", "format": "fahrenheit"}}, {"name": "get_conditions", "arguments": {"location": "Tokyo"}}][/TOOL_CALLS]"#],
        );
        assert_eq!(content, "");
        assert_eq!(calls.len(), 2);
        expect_call(
            &calls,
            0,
            "get_temperature",
            &args(&[("city", json!("London")), ("format", json!("fahrenheit"))]),
        );
        expect_call(
            &calls,
            1,
            "get_conditions",
            &args(&[("location", json!("Tokyo"))]),
        );
    }

    #[test]
    fn two_wrapped_calls_are_indexed_in_order() {
        let (calls, content) = run_parser(
            QWEN,
            &[r#"Okay, let's call both tools! <tool_call>{"name": "get_temperature", "arguments": {"city": "London", "format": "fahrenheit"}}</tool_call><tool_call>{"name": "get_conditions", "arguments": {"location": "Tokyo"}}</tool_call>"#],
        );
        assert_eq!(content, "Okay, let's call both tools! ");
        assert_eq!(calls.len(), 2);
        expect_call(
            &calls,
            0,
            "get_temperature",
            &args(&[("city", json!("London")), ("format", json!("fahrenheit"))]),
        );
        expect_call(
            &calls,
            1,
            "get_conditions",
            &args(&[("location", json!("Tokyo"))]),
        );
    }

    #[test]
    fn an_empty_argument_call_followed_by_a_real_one_both_land() {
        let (calls, content) = run_parser(
            QWEN,
            &[r#"Let me say hello and check the weather. <tool_call>{"name": "say_hello", "arguments": {}}</tool_call><tool_call>{"name": "get_temperature", "arguments": {"city": "London", "format": "fahrenheit"}}</tool_call>"#],
        );
        assert_eq!(content, "Let me say hello and check the weather. ");
        assert_eq!(calls.len(), 2);
        expect_call(&calls, 0, "say_hello", &ToolCallArguments::new());
        expect_call(
            &calls,
            1,
            "get_temperature",
            &args(&[("city", json!("London")), ("format", json!("fahrenheit"))]),
        );
    }

    #[test]
    fn the_same_tool_twice_with_and_without_arguments() {
        let (calls, content) = run_parser(
            QWEN,
            &[r#"Let me check the weather. <tool_call>{"name": "get_conditions", "arguments": {}}</tool_call><tool_call>{"name": "get_conditions", "arguments": {"location": "Tokyo"}}"#],
        );
        assert_eq!(content, "Let me check the weather. ");
        assert_eq!(calls.len(), 2);
        expect_call(&calls, 0, "get_conditions", &ToolCallArguments::new());
        expect_call(
            &calls,
            1,
            "get_conditions",
            &args(&[("location", json!("Tokyo"))]),
        );
    }

    #[test]
    fn a_deepseek_call_arrives_after_its_thinking_block() {
        let (calls, content) = run_parser(
            DEEPSEEK,
            &["<think>Wait, I need to call a tool</think><|tool▁calls▁begin|><|tool▁call▁begin|>function<|tool▁sep|>get_temperature\n```json\n{\"city\": \"Tokyo\"}\n```<|tool▁call▁end|><|tool▁calls▁end|><|end▁of▁sentence|>"],
        );
        assert_eq!(content, "<think>Wait, I need to call a tool</think>");
        assert_eq!(calls.len(), 1);
        expect_call(
            &calls,
            0,
            "get_temperature",
            &args(&[("city", json!("Tokyo"))]),
        );
    }

    /// The streaming case that justify the whole buffer: the tag itself kena cut
    /// in half across chunks (`"<|too"` then `"l▁calls▁begin"`).
    #[test]
    fn a_deepseek_call_split_across_chunks_reassembles() {
        let (calls, content) = run_parser(
            DEEPSEEK,
            &[
                "<think>Wait",
                ", I need",
                " to call",
                " a tool</think><|too",
                "l▁calls▁begin",
                "|>",
                "<|tool▁call▁begin|>function<|tool▁sep|>get_temperature\n",
                "```json\n",
                "{\"city\": \"Tokyo\"}\n",
                "```",
                "<|tool▁c",
                "all▁end|>",
                "<|tool▁calls▁end|>",
                "<|end▁of▁sentence|>",
            ],
        );
        assert_eq!(content, "<think>Wait, I need to call a tool</think>");
        assert_eq!(calls.len(), 1);
        expect_call(
            &calls,
            0,
            "get_temperature",
            &args(&[("city", json!("Tokyo"))]),
        );
    }

    #[test]
    fn a_bare_json_call_streams_in_a_byte_at_a_time() {
        let (calls, content) = run_parser(
            JSON_TMPL,
            &[
                "{",
                "\"name\": \"get_temperature\",",
                "\"arguments\": {",
                "\"city\": \"Tokyo\"",
                "}",
                "}",
            ],
        );
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        expect_call(
            &calls,
            0,
            "get_temperature",
            &args(&[("city", json!("Tokyo"))]),
        );
    }

    /// Still open at end of stream: nothing emitted, nothing leaked as content.
    #[test]
    fn an_unfinished_json_call_emits_nothing_yet() {
        let (calls, content) = run_parser(
            JSON_TMPL,
            &["{", "\"name\": \"get_temperature\",", "\"arguments\": {"],
        );
        assert!(calls.is_empty());
        assert_eq!(content, "");
    }

    /// A JSON object naming a tool that was never offered is the user's answer,
    /// not a call. Lose this and the whole reply kena swallowed.
    #[test]
    fn json_naming_an_unknown_tool_comes_back_as_content() {
        let (calls, content) = run_parser(
            JSON_TMPL,
            &[
                "{",
                "\"name\": \"search\", ",
                "\"arguments\": {",
                "\"query\": \"What is the capital of Canada?\"",
                "}",
                "}",
            ],
        );
        assert!(calls.is_empty());
        assert_eq!(
            content,
            "{\"name\": \"search\", \"arguments\": {\"query\": \"What is the capital of Canada?\"}}"
        );
    }

    /// Once a non-call JSON object close, the parser is Done -- a real call
    /// arriving after that is content too. Blunt, and upstream's own behaviour.
    #[test]
    fn a_non_call_object_ends_parsing_for_good() {
        let (calls, content) = run_parser(
            JSON_TMPL,
            &[
                "{\"name\": \"jeff\"}",
                "{\"name\": \"get_conditions\", \"arguments\": {\"location\": \"San Francisco\"}}",
            ],
        );
        assert!(calls.is_empty());
        assert_eq!(
            content,
            "{\"name\": \"jeff\"}{\"name\": \"get_conditions\", \"arguments\": {\"location\": \"San Francisco\"}}"
        );
    }

    #[test]
    fn a_non_call_object_ends_parsing_for_good_even_when_split() {
        let (calls, content) = run_parser(
            JSON_TMPL,
            &[
                "{\"name\": \"jeff\"} {",
                "\"name\": \"get_conditions\", \"arguments\": {\"location\": \"San Francisco\"}}",
            ],
        );
        assert!(calls.is_empty());
        assert_eq!(
            content,
            "{\"name\": \"jeff\"} {\"name\": \"get_conditions\", \"arguments\": {\"location\": \"San Francisco\"}}"
        );
    }

    /// The reason the `{` tag need the first-non-whitespace guard: a brace in
    /// code must survive as text.
    #[test]
    fn a_brace_in_code_is_not_a_tool_call() {
        let (calls, content) = run_parser(JSON_TMPL, &["for { fmt.Println(\"hello\") }"]);
        assert!(calls.is_empty());
        assert_eq!(content, "for { fmt.Println(\"hello\") }");
    }

    #[test]
    fn a_bracket_tagged_list_of_two_calls_streams() {
        let (calls, content) = run_parser(
            LIST,
            &[
                "[",
                "{",
                "\"name\": \"get_temperature\", ",
                "\"arguments\": {",
                "\"city\": \"London\"",
                "}",
                "},",
                "{",
                "\"name\": \"get_conditions\", ",
                "\"arguments\": {",
                "\"location\": \"Tokyo\"",
                "}",
                "}]",
            ],
        );
        assert_eq!(content, "");
        assert_eq!(calls.len(), 2);
        expect_call(
            &calls,
            0,
            "get_temperature",
            &args(&[("city", json!("London"))]),
        );
        expect_call(
            &calls,
            1,
            "get_conditions",
            &args(&[("location", json!("Tokyo"))]),
        );
    }

    #[test]
    fn a_bracket_list_missing_its_closing_bracket_still_yields_the_call() {
        let (calls, content) = run_parser(
            LIST,
            &[
                "[{",
                "\"name\": \"get_conditions\", ",
                "\"arguments\": {",
                "\"location\": \"Tokyo\"",
                "}",
                "}",
            ],
        );
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        expect_call(
            &calls,
            0,
            "get_conditions",
            &args(&[("location", json!("Tokyo"))]),
        );
    }

    #[test]
    fn a_bracket_list_naming_an_unknown_tool_yields_nothing() {
        let (calls, content) = run_parser(
            LIST,
            &[
                "[",
                "{",
                "\"name\": \"search\", ",
                "\"arguments\": {",
                "\"query\": \"What is the capital of Canada?\"",
                "}",
                "}",
            ],
        );
        assert!(calls.is_empty());
        assert_eq!(content, "");
    }

    #[test]
    fn a_trailing_bracket_after_a_closed_list_is_ignored() {
        let (calls, content) = run_parser(
            LIST,
            &[
                "[",
                "{",
                "\"name\": \"get_conditions\", ",
                "\"arguments\": {",
                "\"location\": \"Tokyo\"",
                "}",
                "}",
                "]",
                "]",
            ],
        );
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        expect_call(
            &calls,
            0,
            "get_conditions",
            &args(&[("location", json!("Tokyo"))]),
        );
    }

    #[test]
    fn prose_starting_with_a_bracket_is_not_a_tool_call() {
        let (calls, content) = run_parser(LIST, &["[special", " del", "ivery]"]);
        assert!(calls.is_empty());
        assert_eq!(content, "[special delivery]");
    }

    /// The prefix hazard, streamed: `say_hello` must NOT fire while
    /// `say_hello_world` still arriving.
    #[test]
    fn a_tool_name_split_across_chunks_waits_for_the_longer_match() {
        let (calls, content) = run_parser(
            QWEN,
            &[
                "<tool_call>",
                "{",
                "\"name\": \"say_hello",
                "_world\",",
                "\"arguments\": {}}",
                "}",
            ],
        );
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        expect_call(&calls, 0, "say_hello_world", &ToolCallArguments::new());
    }

    #[test]
    fn both_colliding_tool_names_resolve_correctly_in_sequence() {
        let (calls, content) = run_parser(
            QWEN,
            &[
                "<tool_call>",
                "{",
                "\"name\": \"say_hello",
                "_world\",",
                "\"arguments\": {}}",
                "</tool_call>",
                "<tool_call>",
                "{",
                "\"name\": \"say_hello",
                "\",",
                "\"arguments\": {}}",
                "</tool_call>",
            ],
        );
        assert_eq!(content, "");
        assert_eq!(calls.len(), 2);
        expect_call(&calls, 0, "say_hello_world", &ToolCallArguments::new());
        expect_call(&calls, 1, "say_hello", &ToolCallArguments::new());
    }

    /// Buffer end exactly on the ambiguous prefix -- emit nothing, leak nothing.
    #[test]
    fn a_truncated_ambiguous_tool_name_emits_nothing() {
        let (calls, content) = run_parser(QWEN, &[r#"<tool_call>{"name": "say_hello"#]);
        assert!(calls.is_empty());
        assert_eq!(content, "");
    }

    #[test]
    fn colliding_names_in_one_chunk_resolve_shortest_then_longest() {
        let (calls, content) = run_parser(
            QWEN,
            &[r#"<tool_call>{"name": "say_hello", "arguments": {}}</tool_call><tool_call>{"name": "say_hello_world", "arguments": {}}"#],
        );
        assert_eq!(content, "");
        assert_eq!(calls.len(), 2);
        expect_call(&calls, 0, "say_hello", &ToolCallArguments::new());
        expect_call(&calls, 1, "say_hello_world", &ToolCallArguments::new());
    }

    #[test]
    fn the_shorter_colliding_name_alone_resolves_to_itself() {
        let (calls, _) = run_parser(
            QWEN,
            &[r#"<tool_call>{"name": "say_hello", "arguments": {}}</tool_call>"#],
        );
        assert_eq!(calls.len(), 1);
        expect_call(&calls, 0, "say_hello", &ToolCallArguments::new());
    }

    #[test]
    fn the_longer_colliding_name_alone_resolves_to_itself() {
        let (calls, _) = run_parser(
            QWEN,
            &[r#"<tool_call>{"name": "say_hello_world", "arguments": {}}</tool_call>"#],
        );
        assert_eq!(calls.len(), 1);
        expect_call(&calls, 0, "say_hello_world", &ToolCallArguments::new());
    }

    #[test]
    fn a_bare_json_call_for_a_name_sharing_a_prefix_resolves() {
        let (calls, content) = run_parser(
            JSON_TMPL,
            &[
                "{",
                "\"name\": \"get_address\",",
                "\"arguments\": {",
                "\"location\": \"London\"",
                "}",
                "}",
            ],
        );
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        expect_call(
            &calls,
            0,
            "get_address",
            &args(&[("location", json!("London"))]),
        );
    }

    /// Some models put `arguments` before `name`. The buffer must advance past
    /// the LATER of the two, else the object kena parsed twice.
    #[test]
    fn arguments_emitted_before_the_name_still_parse() {
        let (calls, content) = run_parser(
            QWEN,
            &[r#"<tool_call>{"arguments": {"a": "5", "b": "10"}, "name": "add"}</tool_call>"#],
        );
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        expect_call(
            &calls,
            0,
            "add",
            &args(&[("a", json!("5")), ("b", json!("10"))]),
        );
    }

    /// KOPITIAM's one improvement over upstream here: Go re-insert arguments
    /// from a randomised map, we keep the model's emission order.
    #[test]
    fn argument_key_order_survives_exactly_as_the_model_emitted_it() {
        let (calls, _) = run_parser(
            QWEN,
            &[r#"<tool_call>{"name": "get_temperature", "arguments": {"format": "celsius", "city": "Ipoh"}}</tool_call>"#],
        );
        let keys: Vec<&str> = calls[0]
            .function
            .arguments
            .0
            .keys()
            .map(|s| s.as_str())
            .collect();
        assert_eq!(keys, vec!["format", "city"]);
    }

    // -- section 1: TestDone -----------------------------------------------

    fn done_for(tag: &str, buffer: &str) -> bool {
        let mut p = Parser::with_tag(Vec::new(), tag);
        p.buffer = buffer.as_bytes().to_vec();
        p.done()
    }

    #[test]
    fn done_is_false_for_a_wrapper_tag_and_an_empty_buffer() {
        assert!(!done_for("<tool_call>", ""));
    }

    #[test]
    fn done_tracks_brace_and_bracket_balance() {
        assert!(!done_for("{", r#"{"name": "get_weather""#));
        assert!(done_for("{", r#"{"name": "get_weather"}"#));
        assert!(done_for("{", "{}"));
        assert!(!done_for("[", r#"[{"name": "get_weather""#));
        assert!(done_for("[", r#"[{"name": "get_weather"}]"#));
        assert!(done_for("[", "[]"));
    }

    /// Braces inside a JSON string close nothing -- miss this and the object
    /// kena truncated one byte early.
    #[test]
    fn done_ignores_braces_inside_strings() {
        assert!(!done_for("{", r#"{"note": "a}b""#));
        assert!(done_for("{", r#"{"note": "a}b"}"#));
        assert!(!done_for("[", r#"[{"note": "x]y"}"#));
    }

    // -- section 1: TestContent --------------------------------------------

    fn content_for(tag: &str, buffer: &str, n: usize) -> String {
        let mut p = Parser::with_tag(Vec::new(), tag);
        p.buffer = buffer.as_bytes().to_vec();
        p.n = n;
        p.content()
    }

    #[test]
    fn content_returns_a_dangling_json_buffer_but_never_a_dangling_tag() {
        assert_eq!(content_for("{", "", 0), "");
        // A half-written wrapper tag is model noise, not an answer.
        assert_eq!(
            content_for("<tool_call>", r#"<tool_call>{"name": "get_temperature""#, 0),
            ""
        );
        assert_eq!(
            content_for("{", r#"{"name": "get_temperature"}"#, 0),
            r#"{"name": "get_temperature"}"#
        );
        assert_eq!(
            content_for("{", r#"{"hello": "world"}"#, 0),
            r#"{"hello": "world"}"#
        );
        assert_eq!(
            content_for("[", r#"[{"name": "get_temperature"}]"#, 0),
            r#"[{"name": "get_temperature"}]"#
        );
        assert_eq!(
            content_for("{", r#"{ fmt.Println("hello")"#, 0),
            r#"{ fmt.Println("hello")"#
        );
    }

    /// Once a call already came out, the leftovers are the call's own syntax,
    /// not content.
    #[test]
    fn content_is_empty_once_any_call_has_been_emitted() {
        assert_eq!(content_for("{", r#"{"hello": "world"}"#, 1), "");
    }

    // -- section 1: TestFindTag --------------------------------------------

    fn find_tag_for(tag: &str, buffer: &str) -> (Option<usize>, bool) {
        let mut p = Parser::with_tag(Vec::new(), tag);
        p.buffer = buffer.as_bytes().to_vec();
        p.find_tag()
    }

    #[test]
    fn find_tag_locates_a_complete_tag() {
        assert_eq!(find_tag_for("<tool_call>", "<tool_call>"), (Some(0), true));
        assert_eq!(
            find_tag_for("<tool_call>", "text <tool_call>"),
            (Some(5), true)
        );
        assert_eq!(
            find_tag_for("<tool_call>", "<tool_call>{\"name\""),
            (Some(0), true)
        );
        assert_eq!(
            find_tag_for("<tool_call>", "    <tool_call>\n {\"name\": \"bob\"}"),
            (Some(4), true)
        );
        assert_eq!(
            find_tag_for("<tool_calls>", "<tool_calls><tool_call>"),
            (Some(0), true)
        );
    }

    #[test]
    fn find_tag_reports_a_partial_tag_at_the_tail_without_claiming_a_match() {
        assert_eq!(find_tag_for("<tool_call>", "test<"), (Some(4), false));
        assert_eq!(find_tag_for("<tool_call>", "hello <tool_"), (Some(6), false));
    }

    #[test]
    fn find_tag_gives_up_when_nothing_can_become_a_tag() {
        assert_eq!(find_tag_for("<tool_call>", "hello world"), (None, false));
        assert_eq!(find_tag_for("<tool_call>", "<tool>"), (None, false));
        assert_eq!(find_tag_for("<tool_call>", ""), (None, false));
    }

    #[test]
    fn find_tag_handles_the_single_character_tags() {
        assert_eq!(find_tag_for("[", "calling tools: ["), (Some(15), true));
        assert_eq!(find_tag_for("{", "{\"name\": \"bob\""), (Some(0), true));
        assert_eq!(find_tag_for("{", "\n\n{\n\"name\": \"bob\""), (Some(2), true));
    }

    // -- section 1: TestFindArguments --------------------------------------

    fn find_args(tool_name: &str, buffer: &str) -> Option<serde_json::Value> {
        let (args, _) = find_arguments(tool_name, buffer.as_bytes());
        args.map(|m| serde_json::Value::Object(m.into_iter().collect()))
    }

    #[test]
    fn find_arguments_returns_nothing_for_an_empty_or_unclosed_buffer() {
        assert_eq!(find_args("", ""), None);
        assert_eq!(find_args("", "   \n\t  "), None);
        assert_eq!(
            find_args("", r#"{"format": "fahrenheit", "location": "San Francisco""#),
            None
        );
        assert_eq!(
            find_args("", r#"{"name": "test", "arguments": {"data": "some {"#),
            None
        );
    }

    /// Trailing junk after a balanced object kena ignored, not fatal.
    #[test]
    fn find_arguments_stops_at_the_first_balanced_object() {
        assert_eq!(
            find_args("", r#"{"format": "fahrenheit"}}"#),
            Some(json!({"format": "fahrenheit"}))
        );
    }

    /// Unquoted keys are not JSON -- upstream reset and keep scanning.
    #[test]
    fn find_arguments_rejects_invalid_json() {
        assert_eq!(
            find_args("", r#"{format: fahrenheit, location: "San Francisco"}"#),
            None
        );
    }

    #[test]
    fn find_arguments_unwraps_the_common_name_plus_arguments_shape() {
        assert_eq!(
            find_args("", r#"{"name": "get_temperature", "arguments": {"format": "fahrenheit", "location": "San Francisco, CA"}}"#),
            Some(json!({"format": "fahrenheit", "location": "San Francisco, CA"}))
        );
    }

    #[test]
    fn find_arguments_sees_through_special_tokens_and_arrays_and_nesting() {
        assert_eq!(
            find_args("", r#"[tool]get_temperature[args]{"format": "fahrenheit", "location": "San Francisco, CA"}[end]"#),
            Some(json!({"format": "fahrenheit", "location": "San Francisco, CA"}))
        );
        assert_eq!(
            find_args("", r#"[{"name": "get_temperature", "arguments": {"format": "fahrenheit", "location": "San Francisco, CA"}}"#),
            Some(json!({"format": "fahrenheit", "location": "San Francisco, CA"}))
        );
        assert_eq!(
            find_args("", r#"{"function": {"name": "get_temperature", "arguments": {"format": "fahrenheit", "location": "San Francisco, CA"}}}"#),
            Some(json!({"format": "fahrenheit", "location": "San Francisco, CA"}))
        );
        assert_eq!(
            find_args("", r#"get_temperature({"location": "San Francisco, CA"})"#),
            Some(json!({"location": "San Francisco, CA"}))
        );
        assert_eq!(
            find_args("", r#"[{"name": "get_temperature", "arguments": {"location": "San Francisco, CA", "format": "fahrenheit"}}, {"name": "get_weather", "arguments": {"location": "San Francisco, CA", "format": "fahrenheit"}}]"#),
            Some(json!({"location": "San Francisco, CA", "format": "fahrenheit"}))
        );
    }

    #[test]
    fn find_arguments_handles_the_deepseek_shapes() {
        assert_eq!(
            find_args("", "<|tool▁calls▁begin|><|tool▁call▁begin|>function<|tool▁sep|>get_temperature\n```json\n{\"location\": \"Tokyo\"}\n```<|tool▁call▁end|><|tool▁calls▁end|><|end▁of▁sentence|>"),
            Some(json!({"location": "Tokyo"}))
        );
        assert_eq!(
            find_args("", r#""arguments": {"location": "Tokyo"}}</tool_call>"#),
            Some(json!({"location": "Tokyo"}))
        );
    }

    /// The whole reason the brace scan is string-aware. Every one of these got a
    /// brace inside a string that must NOT close the object.
    #[test]
    fn find_arguments_never_closes_an_object_on_a_brace_inside_a_string() {
        let cases: &[(&str, serde_json::Value)] = &[
            (
                r#"{"name": "process_code", "arguments": {"code": "if (x > 0) { return true; }"}}"#,
                json!({"code": "if (x > 0) { return true; }"}),
            ),
            (
                r#"{"name": "send_data", "arguments": {"payload": "{\"nested\": {\"key\": \"value\"}}"}}"#,
                json!({"payload": r#"{"nested": {"key": "value"}}"#}),
            ),
            (
                r#"{"name": "analyze", "arguments": {"text": "The JSON is: {\"key\": \"val{ue}\"}"}}"#,
                json!({"text": r#"The JSON is: {"key": "val{ue}"}"#}),
            ),
            (
                r#"{"name": "test", "arguments": {"query": "find } in text"}} {"name": "other"}"#,
                json!({"query": "find } in text"}),
            ),
            (
                r#"{"name": "search", "arguments": {"pattern": "regex: }"}}"#,
                json!({"pattern": "regex: }"}),
            ),
            (
                r#"{"name": "analyze", "arguments": {"data": "{\"items\": [{\"value\": \"}\"}, {\"code\": \"if (x) { return y; }\"}]}"}}"#,
                json!({"data": r#"{"items": [{"value": "}"}, {"code": "if (x) { return y; }"}]}"#}),
            ),
            (
                r#"{"name": "format", "arguments": {"template": "{\n  \"key\": \"value\"\n}"}}"#,
                json!({"template": "{\n  \"key\": \"value\"\n}"}),
            ),
            (
                r#"{"name": "test", "arguments": {"text": "Unicode: \u007B and \u007D"}}"#,
                json!({"text": "Unicode: { and }"}),
            ),
            (
                r#"{"name": "path", "arguments": {"dir": "C:\\Program Files\\{App}\\"}}"#,
                json!({"dir": r"C:\Program Files\{App}\"}),
            ),
            (
                r#"{"name": "query", "arguments": {"sql": "SELECT * FROM users WHERE name = '{admin}'"}}"#,
                json!({"sql": "SELECT * FROM users WHERE name = '{admin}'"}),
            ),
            (
                r#"{"name": "echo", "arguments": {"msg": "He said \"Hello {World}\" loudly"}}"#,
                json!({"msg": r#"He said "Hello {World}" loudly"#}),
            ),
            (
                r#"{"name": "code", "arguments": {"snippet": "// This is a comment with { and }"}}"#,
                json!({"snippet": "// This is a comment with { and }"}),
            ),
            (
                r#"{"name": "test", "arguments": {"path": "C:\\\\{folder}\\\\"}}"#,
                json!({"path": r"C:\\{folder}\\"}),
            ),
            (
                r#"{"name": "test", "arguments": {"a": "", "b": "{value}"}}"#,
                json!({"a": "", "b": "{value}"}),
            ),
            (
                r#"{"name": "test", "arguments": {"key{": "value", "key}": "value2"}}"#,
                json!({"key{": "value", "key}": "value2"}),
            ),
            (
                r#"{"name": "test", "arguments": {"code": "\tif (true) {\n\t\treturn;\n\t}"}}"#,
                json!({"code": "\tif (true) {\n\t\treturn;\n\t}"}),
            ),
            (
                r#"{"name": "test", "arguments": {"data": "before\u0000{after}"}}"#,
                json!({"data": "before\u{0}{after}"}),
            ),
            (
                r#"{"name": "test", "arguments": {"data": "text with quote at end\\\""}}"#,
                json!({"data": r#"text with quote at end\""#}),
            ),
            (
                r#"{"name": "test", "arguments": {"items": ["{", "}", {"key": "value"}]}}"#,
                json!({"items": ["{", "}", {"key": "value"}]}),
            ),
        ];

        for (buffer, want) in cases {
            assert_eq!(find_args("", buffer), Some(want.clone()), "buffer: {buffer}");
        }
    }

    /// A very long argument value with braces all through -- the scan must not
    /// give up partway.
    #[test]
    fn find_arguments_survives_a_long_string_full_of_braces() {
        let data = "a{b}c".repeat(100);
        let buffer = format!(r#"{{"name": "test", "arguments": {{"data": "{data}"}}}}"#);
        assert_eq!(find_args("", &buffer), Some(json!({ "data": data })));
    }

    /// `arguments` as an ARRAY is not a map, so upstream refuse -- and must,
    /// else a caller get a positional list where it expect named parameters.
    #[test]
    fn find_arguments_refuses_a_non_object_arguments_value() {
        assert_eq!(
            find_args(
                "",
                r#"{"name": "batch", "arguments": ["item1", "item2", "{\"nested\": true}"]}"#
            ),
            None
        );
    }

    #[test]
    fn find_arguments_parses_a_stringified_arguments_or_parameters_value() {
        assert_eq!(
            find_args("", r#"{"name": "get_temperature", "arguments": "{\"format\": \"fahrenheit\", \"location\": \"San Francisco, CA\"}"}"#),
            Some(json!({"format": "fahrenheit", "location": "San Francisco, CA"}))
        );
        assert_eq!(
            find_args("", r#"{"name": "get_temperature", "parameters": "{\"format\": \"fahrenheit\", \"location\": \"San Francisco, CA\"}"}"#),
            Some(json!({"format": "fahrenheit", "location": "San Francisco, CA"}))
        );
    }

    #[test]
    fn find_arguments_accepts_the_tool_name_as_the_key() {
        assert_eq!(
            find_args("get_temperature", r#"{"get_temperature": {"format": "fahrenheit", "location": "San Francisco, CA"}}"#),
            Some(json!({"format": "fahrenheit", "location": "San Francisco, CA"}))
        );
        assert_eq!(
            find_args("get_temperature", r#"{"get_temperature": "{\"format\": \"fahrenheit\", \"location\": \"San Francisco, CA\"}"}"#),
            Some(json!({"format": "fahrenheit", "location": "San Francisco, CA"}))
        );
    }

    // -- section 2: tools/template.go -- TestParseTag ----------------------

    fn tag_of(src: &str) -> String {
        parse_tag_from_source(src).expect("template parses")
    }

    #[test]
    fn a_template_with_no_tool_branch_falls_back_to_the_brace_tag() {
        assert_eq!(tag_of(""), "{");
        assert_eq!(tag_of("{{if .ToolCalls}}{{end}}"), "{");
        assert_eq!(
            tag_of("{{if .ToolCalls}}{{range .ToolCalls}}{{ . }}{{end}}{{end}}"),
            "{"
        );
    }

    #[test]
    fn a_literal_marker_in_the_tool_branch_becomes_the_tag() {
        assert_eq!(tag_of("{{if .ToolCalls}}```json\n{{end}}"), "```json");
        assert_eq!(
            tag_of("{{if .ToolCalls}}Action: ```json{{end}}"),
            "Action: ```json"
        );
        assert_eq!(
            tag_of("{{if .ToolCalls}}<|tool▁calls▁begin|>{{range .ToolCalls}}<|tool▁call▁begin|>functionget_current_weather\n```json\n{\"location\": \"Tokyo\"}\n```<|tool▁call▁end|>\n{{end}}<|tool▁calls▁end|>{{end}}"),
            "<|tool▁calls▁begin|>"
        );
        assert_eq!(
            tag_of(r#"{{if .ToolCalls}}{{range .ToolCalls}}<tool_call>{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}</tool_call>{{end}}{{end}}"#),
            "<tool_call>"
        );
        assert_eq!(
            tag_of("{{if .ToolCalls}}\n{{range .ToolCalls}}<tool_call>{\"name\": \"{{ .Function.Name }}\", \"arguments\": {{ .Function.Arguments }}}</tool_call>{{end}}{{end}}"),
            "<tool_call>"
        );
        assert_eq!(
            tag_of(r#"{{if .ToolCalls}}{{range .ToolCalls}}<tool_call>{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}<tool_call>{{end}}{{end}}"#),
            "<tool_call>"
        );
    }

    #[test]
    fn a_bracket_marker_survives_surrounding_whitespace() {
        assert_eq!(
            tag_of("{{if .ToolCalls}}[{{range .ToolCalls}}{{ . }}{{end}}]{{end}}"),
            "["
        );
        assert_eq!(
            tag_of("{{if .ToolCalls}}\n [ {{range .ToolCalls}}{{ . }}{{end}}]{{end}}"),
            "["
        );
    }

    /// The search STOP at a template construct, so a marker that only show up
    /// after the range never kena taken -- a closing marker make a useless tag.
    #[test]
    fn text_after_a_range_is_never_taken_as_the_tag() {
        assert_eq!(
            tag_of("{{if .ToolCalls}}{{range .ToolCalls}}{{ . }}{{end}}]{{end}}"),
            "{"
        );
    }

    #[test]
    fn whitespace_only_text_is_skipped_not_taken_as_a_tag() {
        assert_eq!(
            tag_of("{{if .ToolCalls}} {{range .ToolCalls}}{{ . }}{{end}}{{end}}"),
            "{"
        );
        assert_eq!(
            tag_of("{{if .ToolCalls}}{{range .ToolCalls}}\n{{ . }}\n{{end}}{{end}}"),
            "{"
        );
    }

    /// Cutting at the first `{` is what turn a bare-JSON template into the `{`
    /// fallback instead of an over-specific literal tag.
    #[test]
    fn a_json_shaped_branch_cuts_back_to_the_brace_tag() {
        assert_eq!(
            tag_of(r#"{{if .ToolCalls}}{{range .ToolCalls}}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}{{end}}{{end}}"#),
            "{"
        );
        assert_eq!(
            tag_of("{{if .ToolCalls}}{{range .ToolCalls}}\n{\"name\": \"{{ .Function.Name }}\", \"arguments\": {{ .Function.Arguments }}}{{end}}{{end}}"),
            "{"
        );
        assert_eq!(
            tag_of("{{if .ToolCalls}}\n{{range .ToolCalls}}\n{\"name\": \"{{ .Function.Name }}\", \"arguments\": {{ .Function.Arguments }}}\r\n{{end}}\r\n{{end}}"),
            "{"
        );
    }

    /// CRLF-saved templates must sniff the same tag as their Unix twins.
    #[test]
    fn crlf_line_endings_do_not_change_the_tag() {
        assert_eq!(
            tag_of("{{if .ToolCalls}}{{range .ToolCalls}}\r\n{\"name\": \"{{ .Function.Name }}\", \"arguments\": {{ .Function.Arguments }}}{{end}}{{end}}"),
            "{"
        );
    }

    #[test]
    fn the_first_text_node_wins_even_with_siblings_after_it() {
        assert_eq!(
            tag_of("{{if .ToolCalls}}First text{{if .Something}}inner{{end}}Second text{{end}}"),
            "First text"
        );
    }

    /// A trailing `[` stay because it is part of the literal marker, not a JSON
    /// opener -- only `{` kena cut.
    #[test]
    fn a_trailing_bracket_stays_part_of_a_literal_tag() {
        assert_eq!(tag_of("{{if .ToolCalls}}functools[{{end}}"), "functools[");
        assert_eq!(
            tag_of("{{if .ToolCalls}}[TOOL_CALL] [{{end}}"),
            "[TOOL_CALL] ["
        );
        assert_eq!(
            tag_of("{{if .ToolCalls}}[TOOL_CALL][{{end}}"),
            "[TOOL_CALL]["
        );
    }

    // -- section 3: server/quantization.go ---------------------------------

    fn quant_args(arch: &str, ft: FileType, type_name: Option<&str>) -> Vec<String> {
        let name = type_name.unwrap_or_else(|| ft.name());
        llama_quantize_args(arch, ft, "in.gguf", "out.gguf", name)
    }

    #[test]
    fn a_plain_architecture_gets_no_tensor_overrides() {
        assert_eq!(
            quant_args("llama", FileType::Q4_K_M, None),
            ["--allow-requantize", "in.gguf", "out.gguf", "Q4_K_M"]
        );
    }

    #[test]
    fn qwen35_k_quants_keep_the_mtp_projection_at_q8() {
        let head = [
            "--allow-requantize",
            "--tensor-type",
            r"^blk\.[0-9]+\.nextn\.eh_proj\.weight$=q8_0",
            "in.gguf",
            "out.gguf",
        ];
        let got = quant_args("qwen35moe", FileType::Q4_K_M, None);
        assert_eq!(got[..5], head);
        assert_eq!(got[5], "Q4_K_M");

        let got = quant_args("qwen35", FileType::Q4_K_S, None);
        assert_eq!(got[..5], head);
        assert_eq!(got[5], "Q4_K_S");
    }

    /// Above the K-quant band nothing need protecting -- F16/BF16 don't kena
    /// quantised at all, and Q8_0 already meet the floor the override enforce.
    #[test]
    fn qwen35_adds_no_override_above_the_k_quant_band() {
        for (ft, name) in [
            (FileType::F16, "F16"),
            (FileType::BF16, "BF16"),
            (FileType::Q8_0, "Q8_0"),
        ] {
            assert_eq!(
                quant_args("qwen35moe", ft, None),
                ["--allow-requantize", "in.gguf", "out.gguf", name]
            );
        }
    }

    #[test]
    fn gemma3n_k_quants_keep_the_per_layer_token_embedding_at_f16() {
        assert_eq!(
            quant_args("gemma3n", FileType::Q4_K_M, None),
            [
                "--allow-requantize",
                "--tensor-type",
                r"^per_layer_token_embd\.weight$=f16",
                "in.gguf",
                "out.gguf",
                "Q4_K_M",
            ]
        );
        assert_eq!(
            quant_args("gemma3n", FileType::Q8_0, None),
            ["--allow-requantize", "in.gguf", "out.gguf", "Q8_0"]
        );
    }

    #[test]
    fn deepseek2_k_quants_keep_all_five_mla_tensors_at_q8() {
        assert_eq!(
            quant_args("deepseek2", FileType::Q4_K_M, None),
            [
                "--allow-requantize",
                "--tensor-type",
                r"attn_k_b\.weight$=q8_0",
                "--tensor-type",
                r"attn_q_a\.weight$=q8_0",
                "--tensor-type",
                r"attn_q_b\.weight$=q8_0",
                "--tensor-type",
                r"attn_v_b\.weight$=q8_0",
                "--tensor-type",
                r"attn_kv_a_mqa\.weight$=q8_0",
                "in.gguf",
                "out.gguf",
                "Q4_K_M",
            ]
        );
    }

    #[test]
    fn glm_ocr_k_quants_keep_input_and_output_embeddings_at_f16() {
        let want = [
            "--allow-requantize",
            "--tensor-type",
            r"^token_embd\.weight$=f16",
            "--tensor-type",
            r"^output\.weight$=f16",
            "in.gguf",
            "out.gguf",
            "Q4_K_M",
        ];
        assert_eq!(quant_args("glmocr", FileType::Q4_K_M, None), want);
        assert_eq!(quant_args("glm4", FileType::Q4_K_M, None), want);
    }

    /// COPY requantise nothing, so it must carry no `--tensor-type` at all.
    #[test]
    fn copy_never_carries_a_tensor_override() {
        for arch in ["gemma3n", "qwen35moe", "deepseek2", "glmocr"] {
            assert_eq!(
                quant_args(arch, FileType::Q4_K_M, Some(COPY_TYPE_NAME)),
                ["--allow-requantize", "in.gguf", "out.gguf", "COPY"]
            );
        }
    }

    #[test]
    fn disabling_compat_strips_the_inherited_setting_and_appends_a_zero() {
        let env = [
            "A=1".to_string(),
            format!("{LLAMA_CPP_COMPAT_ENV}=1"),
            "B=2".to_string(),
        ];
        assert_eq!(
            disable_llama_cpp_compat(&env),
            [
                "A=1".to_string(),
                "B=2".to_string(),
                format!("{LLAMA_CPP_COMPAT_ENV}=0")
            ]
        );
    }

    #[test]
    fn the_quantize_environment_forces_strict_validation_unless_compat_is_allowed() {
        let env = [
            "A=1".to_string(),
            format!("{LLAMA_CPP_COMPAT_ENV}=0"),
            "B=2".to_string(),
        ];
        assert_eq!(
            llama_quantize_env(&env, false),
            [
                "A=1".to_string(),
                "B=2".to_string(),
                format!("{LLAMA_CPP_COMPAT_ENV}=0")
            ]
        );
        assert_eq!(
            llama_quantize_env(&env, true),
            ["A=1".to_string(), "B=2".to_string()]
        );
    }

    #[test]
    fn the_five_compatibility_tensor_prefixes_are_recognised() {
        for name in ["a.enc", "mm.0.weight", "mtp.layer", "s.x", "v.blk.0"] {
            assert!(is_embedded_compatibility_tensor(name), "{name}");
        }
        for name in [
            "blk.0.attn_q.weight",
            "token_embd.weight",
            "output.weight",
            "audio.x",
        ] {
            assert!(!is_embedded_compatibility_tensor(name), "{name}");
        }
    }

    #[test]
    fn quantize_type_name_refuses_a_file_type_it_cannot_name() {
        assert_eq!(quantize_type_name(FileType::Q4_K_M), Ok("Q4_K_M"));
        assert_eq!(
            quantize_type_name(FileType::UNKNOWN),
            Err(QuantizeError::UnsupportedType("FileType(1024)".into()))
        );
    }

    #[test]
    fn a_legacy_clip_projector_with_no_declared_type_needs_the_default() {
        let mut kv = Kv::new();
        kv.insert("general.architecture", "clip");
        kv.insert("clip.has_vision_encoder", true);
        assert!(needs_default_llava_projector_type(&kv));

        // Either key present means the model already said so.
        let mut with_type = kv.clone();
        with_type.insert("clip.projector_type", "mlp");
        assert!(!needs_default_llava_projector_type(&with_type));

        let mut with_vision_type = kv.clone();
        with_vision_type.insert("clip.vision.projector_type", "mlp");
        assert!(!needs_default_llava_projector_type(&with_vision_type));

        // No vision encoder, or not a clip model at all -- not our business.
        let mut no_encoder = Kv::new();
        no_encoder.insert("general.architecture", "clip");
        assert!(!needs_default_llava_projector_type(&no_encoder));

        let mut not_clip = Kv::new();
        not_clip.insert("general.architecture", "llama");
        not_clip.insert("llama.has_vision_encoder", true);
        assert!(!needs_default_llava_projector_type(&not_clip));
    }

    // -- section 4: server/logprob.go --------------------------------------

    #[test]
    fn a_token_carries_its_raw_utf8_bytes_not_its_characters() {
        assert_eq!(string_to_byte_ints(""), Vec::<i64>::new());
        assert_eq!(string_to_byte_ints("hi"), vec![104, 105]);
        // "é" is TWO bytes -- the whole point of the field.
        assert_eq!(string_to_byte_ints("é"), vec![195, 169]);
    }

    #[test]
    fn runner_logprobs_convert_to_api_logprobs_with_bytes_filled_in() {
        let input = vec![RunnerLogprob {
            token_logprob: RunnerTokenLogprob {
                token: "hi".into(),
                logprob: -0.25,
            },
            top_logprobs: vec![
                RunnerTokenLogprob {
                    token: "hi".into(),
                    logprob: -0.25,
                },
                RunnerTokenLogprob {
                    token: "ho".into(),
                    logprob: -1.5,
                },
            ],
        }];

        let got = to_api_logprobs(&input);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].token_logprob.token, "hi");
        assert_eq!(got[0].token_logprob.logprob, -0.25);
        assert_eq!(got[0].token_logprob.bytes, vec![104, 105]);
        assert_eq!(got[0].top_logprobs.len(), 2);
        assert_eq!(got[0].top_logprobs[1].token, "ho");
        assert_eq!(got[0].top_logprobs[1].bytes, vec![104, 111]);
    }

    #[test]
    fn a_logprob_with_no_alternatives_converts_to_an_empty_list() {
        let got = to_api_logprobs(&[RunnerLogprob::default()]);
        assert!(got[0].top_logprobs.is_empty());
        assert!(got[0].token_logprob.bytes.is_empty());
    }

    // -- section 5: server/model_recommendations.go ------------------------

    /// Upstream's `TestModelRecommendationsDefaultOrder` -- the list is a
    /// ranking, so the order is part of the contract.
    #[test]
    fn the_default_recommendations_keep_upstreams_ranking() {
        let recs = default_model_recommendations();
        let names: Vec<&str> = recs.iter().map(|r| r.model.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "kimi-k2.6:cloud",
                "glm-5.1:cloud",
                "qwen3.5:cloud",
                "minimax-m2.7:cloud",
                "gemma4",
                "qwen3.5",
            ]
        );
    }

    /// The VRAM figures are DECIMAL gigabytes, not binary. Get this wrong and
    /// the requirement kena under-reported by ~7%.
    #[test]
    fn the_local_recommendations_state_vram_in_decimal_gigabytes() {
        let recs = default_model_recommendations();
        let gemma4 = recs.iter().find(|r| r.model == "gemma4").unwrap();
        assert_eq!(gemma4.vram_bytes, 12_000_000_000);
        let qwen = recs.iter().find(|r| r.model == "qwen3.5").unwrap();
        assert_eq!(qwen.vram_bytes, 14_000_000_000);
    }

    #[test]
    fn the_defaults_validate_cleanly() {
        let recs = default_model_recommendations();
        assert_eq!(validate_model_recommendations(&recs).unwrap(), recs);
    }

    fn rec(model: &str) -> ModelRecommendation {
        ModelRecommendation {
            model: model.into(),
            ..Default::default()
        }
    }

    #[test]
    fn validation_trims_whitespace_and_drops_cloud_entries_with_no_limits() {
        let input = vec![
            ModelRecommendation {
                model: " good-cloud:cloud ".into(),
                description: " good cloud ".into(),
                context_length: 1024,
                max_output_tokens: 256,
                required_plan: " pro ".into(),
                ..Default::default()
            },
            ModelRecommendation {
                model: "bad-cloud:cloud".into(),
                description: "missing limits".into(),
                ..Default::default()
            },
            ModelRecommendation {
                model: " good-local ".into(),
                description: " good local ".into(),
                vram_bytes: 2 * crate::format::GIGABYTE as i64,
                ..Default::default()
            },
        ];

        let got = validate_model_recommendations(&input).unwrap();
        assert_eq!(
            got,
            vec![
                ModelRecommendation {
                    model: "good-cloud:cloud".into(),
                    description: "good cloud".into(),
                    context_length: 1024,
                    max_output_tokens: 256,
                    required_plan: "pro".into(),
                    ..Default::default()
                },
                ModelRecommendation {
                    model: "good-local".into(),
                    description: "good local".into(),
                    vram_bytes: 2 * crate::format::GIGABYTE as i64,
                    ..Default::default()
                },
            ]
        );
    }

    /// Inventing a plan gate would tell a user they cannot run something they
    /// can. Upstream got a whole test for it; so do we.
    #[test]
    fn validation_never_synthesises_a_required_plan() {
        let input = vec![
            ModelRecommendation {
                model: "kimi-k2.6:cloud".into(),
                description: "coding".into(),
                context_length: 262_144,
                max_output_tokens: 262_144,
                ..Default::default()
            },
            ModelRecommendation {
                model: "minimax-m2.7:cloud".into(),
                description: "custom".into(),
                context_length: 204_800,
                max_output_tokens: 128_000,
                required_plan: "team".into(),
                ..Default::default()
            },
        ];
        let got = validate_model_recommendations(&input).unwrap();
        assert_eq!(got[0].required_plan, "");
        assert_eq!(
            got[1].required_plan, "team",
            "an explicit plan is never overwritten"
        );
    }

    #[test]
    fn a_structurally_broken_list_is_rejected_whole() {
        assert_eq!(
            validate_model_recommendations(&[]),
            Err(RecommendationError::Empty)
        );
        assert_eq!(
            validate_model_recommendations(&[rec("   ")]),
            Err(RecommendationError::MissingModel)
        );
        assert_eq!(
            validate_model_recommendations(&[rec("a"), rec("a")]),
            Err(RecommendationError::Duplicate("a".into()))
        );
        // Every entry droppable -> nothing left to trust.
        assert_eq!(
            validate_model_recommendations(&[rec("x:cloud")]),
            Err(RecommendationError::NoneValid)
        );
    }

    #[test]
    fn a_cloud_model_is_recognised_by_its_name_suffix_only() {
        assert!(is_cloud_recommendation("qwen3.5:cloud"));
        assert!(is_cloud_recommendation("gpt-oss-cloud"));
        assert!(!is_cloud_recommendation("qwen3.5"));
        assert!(!is_cloud_recommendation("cloud-qwen"));
    }

    #[test]
    fn jitter_stays_inside_the_eighty_to_one_hundred_twenty_percent_band() {
        let base = std::time::Duration::from_secs(100);
        assert_eq!(with_jitter(base, 0.0), std::time::Duration::from_secs(80));
        assert_eq!(with_jitter(base, 1.0), std::time::Duration::from_secs(120));
        assert_eq!(with_jitter(base, 0.5), std::time::Duration::from_secs(100));
        // Out-of-range draws kena clamped, never allowed outside the band.
        assert_eq!(with_jitter(base, -5.0), std::time::Duration::from_secs(80));
        assert_eq!(with_jitter(base, 5.0), std::time::Duration::from_secs(120));
        assert!(with_jitter(std::time::Duration::ZERO, 0.5).is_zero());
    }

    #[test]
    fn the_backoff_ladder_is_five_minutes_then_fifteen_then_an_hour_then_four() {
        assert_eq!(BACKOFF_STEPS[0].as_secs(), 300);
        assert_eq!(BACKOFF_STEPS[1].as_secs(), 900);
        assert_eq!(BACKOFF_STEPS[2].as_secs(), 3600);
        assert_eq!(BACKOFF_STEPS[3].as_secs(), 14400);
        assert_eq!(REFRESH_INTERVAL.as_secs(), 14400);
        assert_eq!(FETCH_TIMEOUT.as_secs(), 3);
        assert_eq!(READ_REFRESH_COOLDOWN.as_secs(), 5);
    }

    // -- section 5: the KOPITIAM addition ----------------------------------

    #[test]
    fn an_offline_machine_is_offered_only_the_local_models_that_fit() {
        let recs = default_model_recommendations();

        // 16 GB card, no network: both local models fit, no cloud offered.
        let got = recommend_for_machine(
            &recs,
            &MachineFacts {
                vram_bytes: 16_000_000_000,
                cloud_available: false,
            },
        );
        let names: Vec<&str> = got.iter().map(|r| r.model.as_str()).collect();
        assert_eq!(names, vec!["gemma4", "qwen3.5"]);

        // 12 GB card: gemma4 fit exactly, qwen3.5 (14 GB) don't.
        let got = recommend_for_machine(
            &recs,
            &MachineFacts {
                vram_bytes: 12_000_000_000,
                cloud_available: false,
            },
        );
        let names: Vec<&str> = got.iter().map(|r| r.model.as_str()).collect();
        assert_eq!(
            names,
            vec!["gemma4"],
            "the comparison is >=, so an exact fit fits"
        );

        // A 4 GB laptop with no network get nothing, and that is a real answer.
        assert!(recommend_for_machine(
            &recs,
            &MachineFacts {
                vram_bytes: 4_000_000_000,
                cloud_available: false,
            }
        )
        .is_empty());
    }

    #[test]
    fn cloud_entries_appear_only_when_cloud_is_available_and_ignore_vram() {
        let recs = default_model_recommendations();
        let got = recommend_for_machine(
            &recs,
            &MachineFacts {
                vram_bytes: 0,
                cloud_available: true,
            },
        );
        let names: Vec<&str> = got.iter().map(|r| r.model.as_str()).collect();
        // All four cloud models, plus nothing local (0 bytes of VRAM).
        assert_eq!(
            names,
            vec![
                "kimi-k2.6:cloud",
                "glm-5.1:cloud",
                "qwen3.5:cloud",
                "minimax-m2.7:cloud"
            ]
        );
    }

    /// A local entry with no stated cost kena offered rather than hidden -- an
    /// incomplete registry entry must not make a model invisible.
    #[test]
    fn a_local_model_with_no_stated_vram_is_offered_anyway() {
        let unknown = ModelRecommendation {
            model: "mystery".into(),
            ..Default::default()
        };
        assert!(unknown.fits(&MachineFacts {
            vram_bytes: 0,
            cloud_available: false
        }));
    }

    // -- section 6: server/fixblobs.go -------------------------------------

    #[test]
    fn only_an_exact_sha256_colon_prefix_gets_repaired() {
        assert_eq!(fixed_blob_name("sha256:1234"), Some("sha256-1234".into()));
        assert_eq!(fixed_blob_name("sha256:abcd"), Some("sha256-abcd".into()));
        // Already fixed -- leave it be.
        assert_eq!(fixed_blob_name("sha256-1234"), None);
        // Wrong digest name -- upstream leave it alone, so must we.
        assert_eq!(fixed_blob_name("sha259:5678"), None);
        assert_eq!(fixed_blob_name("manifest"), None);
    }

    /// Only the FIRST colon kena cut, so extra colons ride along in the digest
    /// half instead of mangling the name.
    #[test]
    fn only_the_first_colon_is_cut() {
        assert_eq!(fixed_blob_name("sha256:ab:cd"), Some("sha256-ab:cd".into()));
    }

    /// Exercise the walk itself. `:` is illegal in an NTFS filename, so this can
    /// only run where such a store could have existed -- exactly the condition
    /// upstream's own test skip on.
    #[test]
    #[cfg(not(windows))]
    fn the_walk_repairs_nested_blobs_deepest_first() {
        let root = tempfile::tempdir().expect("tempdir");
        for rel in ["sha256:1234", "sha256-5678", "x/y/sha256:abcd", "x/y/keepme"] {
            let full = root.path().join(rel);
            std::fs::create_dir_all(full.parent().unwrap()).unwrap();
            std::fs::write(&full, b"").unwrap();
        }

        fix_blobs(root.path()).expect("fix_blobs");

        fn walk(dir: &std::path::Path, base: &std::path::Path, out: &mut Vec<String>) {
            for e in std::fs::read_dir(dir).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    walk(&p, base, out);
                } else {
                    out.push(
                        p.strip_prefix(base)
                            .unwrap()
                            .to_string_lossy()
                            .replace('\\', "/"),
                    );
                }
            }
        }

        let mut got = Vec::new();
        walk(root.path(), root.path(), &mut got);
        got.sort();

        assert_eq!(
            got,
            vec!["sha256-1234", "sha256-5678", "x/y/keepme", "x/y/sha256-abcd"]
        );
    }

    // -- section 7: server/inference_request_log.go ------------------------

    #[test]
    fn a_route_becomes_a_filename_safe_alphanumeric_run() {
        assert_eq!(sanitize_route_for_filename("/api/chat"), "api_chat");
        assert_eq!(sanitize_route_for_filename("/api/generate"), "api_generate");
        assert_eq!(sanitize_route_for_filename("/"), "root");
        assert_eq!(sanitize_route_for_filename(""), "root");
        assert_eq!(
            sanitize_route_for_filename("/v1/chat/completions"),
            "v1_chat_completions"
        );
        // One non-ASCII CHARACTER become one underscore, not one per byte.
        assert_eq!(sanitize_route_for_filename("/café"), "caf_");
    }

    #[test]
    fn the_replay_script_references_its_body_relative_to_itself() {
        let s = replay_curl_script(
            "POST",
            "http://127.0.0.1:11434/api/chat",
            "application/json",
            "20260729T120000.000000000Z-000001_api_chat_body.json",
        );
        assert!(s.starts_with("#!/bin/sh\n"));
        // Portability: the body path kena resolved from the script's own dir, so
        // the whole log folder can kena copied elsewhere and still replay.
        assert!(s.contains(r#"SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)""#));
        assert!(s.contains(r#"--url "http://127.0.0.1:11434/api/chat""#));
        assert!(s.contains(r#"--header "Content-Type: application/json""#));
        assert!(s.contains(
            r#"--data-binary @"${SCRIPT_DIR}/20260729T120000.000000000Z-000001_api_chat_body.json""#
        ));
    }

    #[test]
    fn go_quote_escapes_quotes_and_backslashes() {
        assert_eq!(go_quote("plain"), "\"plain\"");
        assert_eq!(go_quote(r#"a"b"#), r#""a\"b""#);
        assert_eq!(go_quote(r"a\b"), r#""a\\b""#);
    }
}
