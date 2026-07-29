//! Response parsers -- the inverse of a renderer.
//!
//! **Upstream:** `model/parsers/parsers.go` (ollama, MIT, Copyright (c) Ollama).
//! Ported against `4713800b08b2ddf5e14acf8398953cf7b12f169b` (2026-07-28).
//!
//! ## What a parser is for
//!
//! A renderer takes [`Message`]s and builds the prompt string the model sees. A
//! **parser** does the reverse trip: it takes whatever the model spat out and
//! splits it back into three separate things --
//!
//! * `content` -- the reply the user should see,
//! * `thinking` -- whatever was inside the model's reasoning tags,
//! * `calls` -- structured [`ToolCall`]s.
//!
//! Every model family frames these differently (`<think>`, `<tool_call>`,
//! `<|tool_call>`, `<|tool calls begin|>`, ...), so there is one parser per
//! family and a registry to pick between them.
//!
//! ## Streaming is the whole point, hor
//!
//! [`Parser::add`] is fed **chunks**, not the whole response. A chunk boundary
//! can land anywhere -- including halfway through `</think>`. So the contract is:
//!
//! > **Emit the moment the input is unambiguous. Buffer while it is not.**
//!
//! Concretely, if the buffer currently ends in `"...here we go</thi"`, the tail
//! `</thi` *might* become `</think>` on the next chunk, or it might become
//! literal text `</thing`. The parser therefore emits `"...here we go"` now and
//! holds `</thi` back. [`overlap`] is the helper that measures exactly how much
//! of the tail is still ambiguous, and [`trailing_whitespace_len`] extends the
//! held-back region over any whitespace immediately before it (because a tag is
//! usually preceded by whitespace that should be trimmed, and once emitted you
//! cannot take it back).
//!
//! **What would make this wrong:** emitting the ambiguous tail early. Do that
//! and a `</think>` split across two chunks leaks `</thi` into `content` and the
//! user sees a broken tag. Holding *too much* back is the safer failure -- it
//! just delays output by one chunk -- but holding back forever is also wrong,
//! since each state must drain whatever it can on every call.
//!
//! ## Why `Box<dyn Parser>` and not an enum
//!
//! Upstream has a runtime `Register(name, constructor)` hook so a caller can
//! plug in a parser that this crate never heard of, and can even *override* a
//! built-in. An enum cannot express "a family we do not know about yet", so we
//! keep the Go shape: a trait object plus a name -> constructor registry. The
//! cost is one virtual call per chunk, which is nothing next to the string work
//! already happening inside.
//!
//! ## Overlap with [`crate::thinking`]
//!
//! [`crate::thinking`] is the *generic* thinking-tag state machine used for
//! models that only need `<think>` / `</think>` stripped and nothing else. The
//! per-family parsers here carry their **own** thinking logic because upstream
//! does -- their tags interact with tool tags (a `<tool_call>` appearing before
//! `</think>` ends thinking early, for one), which the generic machine has no
//! idea about. So the duplication is upstream's, and deliberate.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use crate::api::{Message, ThinkValue, Tool, ToolCall};

pub mod deepseek3;
pub mod gemma4;
pub mod glm46;
pub mod glm47;
pub mod qwen3;
pub mod qwen35;
pub mod qwen3coder;
pub mod qwen3vl;

pub use deepseek3::DeepSeek3Parser;
pub use gemma4::Gemma4Parser;
pub use glm46::Glm46Parser;
pub use glm47::{Glm47Parser, GlmOcrParser};
pub use qwen3::Qwen3Parser;
pub use qwen35::Qwen35Parser;
pub use qwen3coder::Qwen3CoderParser;
pub use qwen3vl::Qwen3VlParser;

/// What one [`Parser::add`] call managed to pull out of the stream.
///
/// **Upstream:** the `(content, thinking, calls, err)` return tuple of
/// `Parser.Add`. Made a struct here because four unnamed returns is how you get
/// `content` and `thinking` swapped at a call site and nobody notices for a
/// month.
///
/// All three fields are **deltas**, not accumulated totals -- whatever became
/// unambiguous since the previous call. An empty `Parsed` is completely normal:
/// it means everything fed in so far is still ambiguous and got buffered.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Parsed {
    pub content: String,
    pub thinking: String,
    pub calls: Vec<ToolCall>,
}

impl Parsed {
    /// Nothing came out this round -- everything is still sitting in the buffer.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty() && self.thinking.is_empty() && self.calls.is_empty()
    }
}

/// What can go wrong while parsing a model's output.
///
/// **Upstream:** plain `error` values. Upstream logs a warning and returns the
/// error up; so do we. Note that a malformed tool call is a **hard error**, not
/// a silent drop -- the caller has to know the model produced something it could
/// not honour.
#[derive(Debug, thiserror::Error)]
pub enum ParserError {
    /// The tool-call payload was not valid JSON. **Upstream:**
    /// `fmt.Errorf("failed to parse JSON: %w", err)`.
    #[error("failed to parse JSON: {0}")]
    Json(#[from] serde_json::Error),

    /// The model asked for a tool but gave no name to call.
    #[error("empty function name")]
    EmptyFunctionName,

    /// The XML-ish tool-call body (qwen3-coder style) did not parse.
    #[error("malformed tool call: {0}")]
    MalformedToolCall(String),
}

/// One model family's response parser.
///
/// **Upstream:** the `Parser` interface in `parsers.go`.
///
/// Lifecycle, and it matters: [`init`](Parser::init) **must** be called before
/// the first [`add`](Parser::add) -- it is what resets the buffer, zeroes the
/// tool-call index, and decides whether the stream starts in thinking mode or
/// content mode. Re-`init` between turns; a parser is not single-use, but it is
/// also not automatically clean.
pub trait Parser: Send {
    /// Set up for one generation. Returns the tools the *renderer* should use --
    /// most families hand back what they were given untouched, but some (harmony)
    /// rename tools, so the return value is not decoration.
    ///
    /// `last_message` is the previous message in the conversation, used to spot
    /// an **assistant prefill** (the caller pre-seeded the assistant's reply), in
    /// which case thinking has already been closed off and the stream starts in
    /// content mode.
    ///
    /// `think` is three-state on purpose -- see [`ThinkValue`]. `None` means the
    /// caller never mentioned thinking, which is *not* the same as `Some(false)`;
    /// each family picks its own default for `None`.
    fn init(
        &mut self,
        tools: Vec<Tool>,
        last_message: Option<&Message>,
        think: Option<&ThinkValue>,
    ) -> Vec<Tool>;

    /// Feed the next chunk. `done` says this is the last one, so any state that
    /// can only drain at the end should drain now.
    fn add(&mut self, s: &str, done: bool) -> Result<Parsed, ParserError>;

    /// Grammar tokens that must survive detokenisation for this parser to see its
    /// own boundaries.
    ///
    /// **Upstream:** `Parser.PreservedTokens`. Why it exists: llama-server will
    /// happily swallow special tokens on the way out, and if `</tool_call>` never
    /// reaches the parser, the tool call is never closed and never emitted. This
    /// list is the parser telling the runtime "do not eat these ah".
    fn preserved_tokens(&self) -> Vec<&'static str> {
        Vec::new()
    }

    fn has_tool_support(&self) -> bool;

    fn has_thinking_support(&self) -> bool;
}

/// Builds a fresh parser. **Upstream:** `ParserConstructor`.
pub type ParserConstructor = fn() -> Box<dyn Parser>;

/// Runtime-registered parsers, checked **before** the built-in list.
///
/// **Upstream:** the package-level `registry` var. Checked first so a caller can
/// override a built-in name, which upstream's `TestOverrideBuiltInParser` relies
/// on.
fn registry() -> &'static RwLock<HashMap<String, ParserConstructor>> {
    static REGISTRY: OnceLock<RwLock<HashMap<String, ParserConstructor>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register (or override) a parser by name. **Upstream:** `Register`.
///
/// A poisoned lock is treated as "no registration happened" rather than a panic
/// -- a wedged registry must not take down a generation that would otherwise
/// have used a built-in parser.
pub fn register(name: &str, constructor: ParserConstructor) {
    if let Ok(mut m) = registry().write() {
        m.insert(name.to_string(), constructor);
    }
}

/// Drop a registration. **Divergence from upstream:** Go has no unregister,
/// because Go's tests inside one package run sequentially and never need to undo
/// a global mutation. Rust runs `#[test]`s in parallel threads of one process,
/// so a test that overrides `"passthrough"` would corrupt a concurrently-running
/// test that looks `"passthrough"` up. This exists purely so the ported registry
/// tests can put the world back.
#[cfg(test)]
fn unregister(name: &str) {
    if let Ok(mut m) = registry().write() {
        m.remove(name);
    }
}

/// Look up a parser by the name in a model's `ConfigV2.parser` field.
///
/// **Upstream:** `ParserForName`. Returns `None` for an unknown name -- upstream
/// returns a nil interface, and the caller's job is to fall back to the generic
/// template path, not to fail.
///
/// The names are **wire values baked into published model manifests**, so they
/// are not ours to tidy. `"ornith"` really is an alias of `"qwen3.5"`, and
/// `"qwen3-thinking"` really is the same struct as `"qwen3"` with two flags
/// flipped.
///
/// **Partial port, stated plainly:** upstream dispatches ~25 names. Still not
/// ported, and therefore falling through to `None` here exactly like an unknown
/// name: `harmony`, `olmo3`, `olmo3-think`, `cogito`, `cohere`, `laguna`,
/// `poolside-v1`, `lfm2`, `lfm2-thinking`, `ministral`, `nemotron-3-nano`,
/// `functiongemma`. `None` is the honest answer -- handing them a passthrough
/// instead would silently leak raw `<tool_call>` markup into user-visible
/// content, which is worse than the caller knowing it has no parser.
pub fn parser_for_name(name: &str) -> Option<Box<dyn Parser>> {
    if let Ok(m) = registry().read()
        && let Some(c) = m.get(name)
    {
        return Some(c());
    }

    Some(match name {
        "qwen3" => Box::new(Qwen3Parser::new(false, false)) as Box<dyn Parser>,
        "qwen3-thinking" => Box::new(Qwen3Parser::new(true, true)),
        "qwen3.5" | "ornith" => Box::new(Qwen35Parser::default()),
        "qwen3-coder" => Box::new(Qwen3CoderParser::default()),
        "qwen3-vl-instruct" => Box::new(Qwen3VlParser::new(false)),
        "qwen3-vl-thinking" => Box::new(Qwen3VlParser::new(true)),
        "deepseek3" => Box::new(DeepSeek3Parser::new(true)),
        "gemma4" => Box::new(Gemma4Parser::new(true)),
        "gemma4-no-thinking" => Box::new(Gemma4Parser::new(false)),
        // NOTE: upstream has no `"glm-4.6"` name -- `Glm46Parser` is only ever
        // reached as the base of the two below. Kept as a public type anyway,
        // because it is where all the behaviour lives.
        "glm-4.7" => Box::new(Glm47Parser::default()),
        "glm-ocr" => Box::new(GlmOcrParser::default()),
        "passthrough" => Box::new(PassthroughParser),
        _ => return None,
    })
}

/// The do-nothing parser. **Upstream:** `PassthroughParser`.
///
/// Everything is content; no thinking, no tools. This is what a model with no
/// special output framing gets.
#[derive(Debug, Default, Clone, Copy)]
pub struct PassthroughParser;

impl Parser for PassthroughParser {
    fn init(
        &mut self,
        tools: Vec<Tool>,
        _last_message: Option<&Message>,
        _think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        // Passthrough doesn't modify tools.
        tools
    }

    fn add(&mut self, s: &str, _done: bool) -> Result<Parsed, ParserError> {
        Ok(Parsed {
            content: s.to_string(),
            ..Default::default()
        })
    }

    fn has_tool_support(&self) -> bool {
        false
    }

    fn has_thinking_support(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Shared helpers. Upstream: the bottom of `parsers.go`.
// ---------------------------------------------------------------------------

/// Split the buffer at the **first** occurrence of `tag`, and leave whatever
/// came after it sitting in the buffer.
///
/// **Upstream:** `splitAtTag(sb *strings.Builder, tag string, trimAfter bool)`.
///
/// Returns `(before, after)` where:
/// * `before` always has its **trailing** whitespace trimmed -- the whitespace
///   sitting right in front of a tag is framing, not content;
/// * `after` gets its **leading** whitespace trimmed only when `trim_after`.
///
/// The buffer is left holding exactly `after`.
///
/// **The one behaviour worth memorising:** if `tag` is *not* present, the buffer
/// is **emptied** and the whole thing comes back as `before`, untrimmed, with
/// `after` empty. So never call this speculatively hoping it is a no-op -- check
/// the tag is there first, which is what every caller does.
pub(crate) fn split_at_tag(sb: &mut String, tag: &str, trim_after: bool) -> (String, String) {
    let Some(idx) = sb.find(tag) else {
        let all = std::mem::take(sb);
        return (all, String::new());
    };

    let before = sb[..idx].trim_end().to_string();
    let after_raw = &sb[idx + tag.len()..];
    let after = if trim_after {
        after_raw.trim_start().to_string()
    } else {
        after_raw.to_string()
    };
    sb.clear();
    sb.push_str(&after);
    (before, after)
}

/// How many bytes of the **tail** of `s` could still turn into the **head** of
/// `delim`.
///
/// **Upstream:** `overlap(s, delim string) int`.
///
/// This is the function that makes streaming work. `overlap("hi</thi", "</think>")`
/// is `5`, meaning the last five bytes of the buffer are ambiguous -- they could
/// grow into the closing tag, or they could turn out to be literal text -- so
/// they must be held back, and only `"hi"` may be emitted.
///
/// Returns `0` when the tail cannot possibly be the start of `delim`, i.e. the
/// whole buffer is safe to emit.
///
/// **Byte-wise on purpose, not char-wise.** Upstream compares raw bytes, and so
/// must we, because a multi-byte delimiter can be split mid-rune across chunks:
/// `overlap("hello\u{1F527}", "\u{1F527}\u{5DE5}\u{5177}")` must return 4 (the
/// byte length of the emoji), which char-wise slicing of `delim` could not
/// express. Comparing bytes also means a partial UTF-8 sequence simply never
/// matches, which is the answer we want anyway.
pub(crate) fn overlap(s: &str, delim: &str) -> usize {
    let (sb, db) = (s.as_bytes(), delim.as_bytes());
    let max = db.len().min(sb.len());
    for i in (1..=max).rev() {
        if sb.ends_with(&db[..i]) {
            return i;
        }
    }
    0
}

/// Length **in bytes** of the whitespace run at the end of `s`.
///
/// **Upstream:** `trailingWhitespaceLen`. Bytes, not chars, because every caller
/// immediately uses it as a slice index into the buffer.
///
/// Used together with [`overlap`]: once the ambiguous tail is found, the region
/// held back is widened backwards over any whitespace in front of it. Reason: if
/// the tail does turn out to be a tag, [`split_at_tag`] would have trimmed that
/// whitespace away -- but by then it would already have been emitted, and you
/// cannot un-emit. So hold it, and let the tag decide.
///
/// Unicode-aware: a non-breaking space (U+00A0, 2 bytes) and an ideographic
/// space (U+3000, 3 bytes) both count, and they count as their **byte** length.
/// Go's `unicode.IsSpace` and Rust's `char::is_whitespace` are both the Unicode
/// `White_Space` property, so the two agree rune for rune -- that is why plain
/// `trim_end()` is a faithful port and not a shortcut.
pub(crate) fn trailing_whitespace_len(s: &str) -> usize {
    s.len() - s.trim_end().len()
}

/// Byte-index split that cannot panic on a bad boundary.
///
/// All the indices in this module come from ASCII tag matches or from
/// [`trailing_whitespace_len`], so they are always on a char boundary in
/// practice. This wrapper exists so a future family whose tags are non-ASCII
/// degrades into "emit nothing this round" instead of panicking inside a
/// generation.
pub(crate) fn chop(s: &str, at: usize) -> (&str, &str) {
    s.split_at_checked(at).unwrap_or((s, ""))
}

/// The buffering step every family repeats: hold back the ambiguous tail, emit
/// the rest.
///
/// Given the accumulated buffer and however many trailing bytes are ambiguous
/// (`overlap_len`, possibly `0`), this widens the held-back region back over any
/// whitespace in front of it, rewrites `buf` to hold **only** the ambiguous part,
/// and returns the unambiguous prefix that is safe to emit right now.
///
/// **Upstream:** this exact eight-line dance appears inline in `qwen3.go`,
/// `qwen3coder.go`, `qwen35.go` and `qwen3vl.go`. Factored out here because four
/// copies of a subtle index calculation is four chances to get it wrong; the
/// behaviour is byte-for-byte the same.
pub(crate) fn emit_unambiguous(buf: &mut String, overlap_len: usize) -> String {
    let acc = std::mem::take(buf);
    let (before_partial_tag, _) = chop(&acc, acc.len().saturating_sub(overlap_len));
    let ambiguous_start = before_partial_tag.len() - trailing_whitespace_len(before_partial_tag);
    let (unambiguous, ambiguous) = chop(&acc, ambiguous_start);
    let unambiguous = unambiguous.to_string();
    buf.push_str(ambiguous);
    unambiguous
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Registry-mutating tests take this so they cannot trip over each other.
    /// See [`unregister`] for why Rust needs this and Go does not.
    static REGISTRY_LOCK: Mutex<()> = Mutex::new(());

    #[derive(Default)]
    struct MockParser;

    impl Parser for MockParser {
        fn init(
            &mut self,
            tools: Vec<Tool>,
            _l: Option<&Message>,
            _t: Option<&ThinkValue>,
        ) -> Vec<Tool> {
            tools
        }
        fn add(&mut self, s: &str, _done: bool) -> Result<Parsed, ParserError> {
            Ok(Parsed {
                content: format!("mock:{s}"),
                ..Default::default()
            })
        }
        fn has_tool_support(&self) -> bool {
            false
        }
        fn has_thinking_support(&self) -> bool {
            false
        }
    }

    #[test]
    fn a_custom_parser_can_be_registered_and_looked_up() {
        let _g = REGISTRY_LOCK.lock().expect("lock");
        register("custom-parser", || Box::new(MockParser));
        let mut p = parser_for_name("custom-parser").expect("registered parser");
        assert_eq!(p.add("test", false).expect("add").content, "mock:test");
        unregister("custom-parser");
    }

    #[test]
    fn every_built_in_parser_name_still_resolves() {
        let _g = REGISTRY_LOCK.lock().expect("lock");
        for name in [
            "passthrough",
            "qwen3",
            "qwen3-thinking",
            "qwen3-coder",
            "qwen3.5",
            "ornith",
            "qwen3-vl-instruct",
            "qwen3-vl-thinking",
            "deepseek3",
            "gemma4",
            "gemma4-no-thinking",
            "glm-4.7",
            "glm-ocr",
        ] {
            assert!(parser_for_name(name).is_some(), "missing built-in {name}");
        }
    }

    /// The families this port has NOT reached must resolve to `None`, not to a
    /// passthrough. A passthrough would leak raw `<tool_call>` markup into what
    /// the user sees; `None` tells the caller plainly there is no parser.
    #[test]
    fn families_this_port_has_not_reached_resolve_to_nothing() {
        let _g = REGISTRY_LOCK.lock().expect("lock");
        for name in [
            "harmony",
            "olmo3",
            "olmo3-think",
            "cogito",
            "cohere",
            "laguna",
            "poolside-v1",
            "lfm2",
            "lfm2-thinking",
            "ministral",
            "nemotron-3-nano",
            "functiongemma",
        ] {
            assert!(
                parser_for_name(name).is_none(),
                "{name} now resolves -- update the doc comment on parser_for_name"
            );
        }
    }

    /// Upstream's `TestParserPreservedTokensCoverKnownLlamaServerRegressions`,
    /// restricted to the families this port has reached. These are real
    /// regressions: when llama-server ate `</tool_call>`, tool calls silently
    /// stopped working.
    #[test]
    fn preserved_tokens_cover_the_known_llama_server_regressions() {
        let p = parser_for_name("qwen3-coder").expect("qwen3-coder");
        for tok in ["<tool_call>", "</tool_call>"] {
            assert!(p.preserved_tokens().contains(&tok), "missing {tok}");
        }
    }

    #[test]
    fn a_registered_name_overrides_the_built_in_of_the_same_name() {
        let _g = REGISTRY_LOCK.lock().expect("lock");
        register("passthrough", || Box::new(MockParser));
        let mut p = parser_for_name("passthrough").expect("passthrough");
        assert_eq!(p.add("test", false).expect("add").content, "mock:test");
        unregister("passthrough");
        // ...and taking the override away restores the real one.
        let mut p = parser_for_name("passthrough").expect("passthrough");
        assert_eq!(p.add("test", false).expect("add").content, "test");
    }

    #[test]
    fn an_unknown_parser_name_resolves_to_nothing() {
        assert!(parser_for_name("nonexistent-parser").is_none());
    }

    /// Upstream's `TestSplitAtTag`, ported verbatim as ground truth.
    #[test]
    fn split_at_tag_matches_upstreams_table() {
        let tag = "<!-- split -->";
        // (input, tag, trim_after, want_before, want_after, want_buffer)
        let cases: &[(&str, &str, bool, &str, &str, &str)] = &[
            (
                "hello <!-- split --> world",
                tag,
                true,
                "hello",
                "world",
                "world",
            ),
            (
                "hello <!-- split -->   world",
                tag,
                false,
                "hello",
                "   world",
                "   world",
            ),
            ("<!-- split -->world", tag, true, "", "world", "world"),
            (
                "<!-- split -->   world",
                tag,
                false,
                "",
                "   world",
                "   world",
            ),
            ("hello <!-- split -->", tag, true, "hello", "", ""),
            ("hello <!-- split -->", tag, false, "hello", "", ""),
            (
                "hello <!-- split --> world <!-- split --> end",
                tag,
                true,
                "hello",
                "world <!-- split --> end",
                "world <!-- split --> end",
            ),
            // No tag at all: the buffer is DRAINED and everything comes back as
            // `before`, untrimmed. Surprises people; it is upstream.
            ("hello world", tag, true, "hello world", "", ""),
            ("", tag, true, "", "", ""),
            ("   \t\n<!-- split -->world", tag, true, "", "world", "world"),
            ("hello<!-- split -->   \t\n", tag, true, "hello", "", ""),
            (
                "hello<!-- split -->   \t\n",
                tag,
                false,
                "hello",
                "   \t\n",
                "   \t\n",
            ),
            (
                "  hello \t\n <!-- split --> \n\t world  ",
                tag,
                true,
                "  hello",
                "world  ",
                "world  ",
            ),
            (
                "text <tag attr=\"value\"> more text",
                "<tag attr=\"value\">",
                true,
                "text",
                "more text",
                "more text",
            ),
        ];

        for (input, tag, trim_after, want_before, want_after, want_buf) in cases {
            let mut sb = input.to_string();
            let (before, after) = split_at_tag(&mut sb, tag, *trim_after);
            assert_eq!(&before, want_before, "before, for input {input:?}");
            assert_eq!(&after, want_after, "after, for input {input:?}");
            assert_eq!(&sb, want_buf, "buffer, for input {input:?}");
        }
    }

    /// Upstream's `TestOverlapFunction`, verbatim -- including the unicode cases
    /// that force byte-wise comparison.
    #[test]
    fn overlap_measures_the_ambiguous_tail_in_bytes() {
        assert_eq!(overlap("hello", "<tool"), 0);
        assert_eq!(overlap("hello<tool", "<tool>"), 5);
        assert_eq!(overlap("hello<to", "<tool>"), 3);
        assert_eq!(overlap("测试🎯<to", "<tool>"), 3);
        assert_eq!(overlap("مرحبا", "<tool>"), 0);
        assert_eq!(overlap("世界<", "<tool>"), 1);
        assert_eq!(overlap("hello🔧", "🔧工具"), "🔧".len());
        assert_eq!(overlap("hello🔧工", "🔧工具"), "🔧工".len());
    }

    /// Upstream's `TestTrailingWhitespaceLen` + `TestTrailingWhitespaceLenUnicode`.
    #[test]
    fn trailing_whitespace_is_measured_in_bytes_not_runes() {
        assert_eq!(trailing_whitespace_len("abc"), 0);
        assert_eq!(trailing_whitespace_len("abc "), 1);
        assert_eq!(trailing_whitespace_len("abc \n"), 2);
        assert_eq!(trailing_whitespace_len(" \n  "), 4);
        assert_eq!(trailing_whitespace_len(" \n abc"), 0);
        assert_eq!(trailing_whitespace_len("测试🎯 "), 1);
        assert_eq!(trailing_whitespace_len("مرحبا\t\n"), 2);
        // Multi-byte whitespace counts its BYTES, not 1.
        assert_eq!(trailing_whitespace_len("Hello\u{00a0}"), 2);
        assert_eq!(trailing_whitespace_len("Hello\u{3000}"), 3);
        assert_eq!(trailing_whitespace_len("Hi\u{00a0}\u{3000}"), 5);
    }

    #[test]
    fn the_passthrough_parser_hands_everything_back_as_content() {
        let mut p = PassthroughParser;
        let got = p.add("anything at all", true).expect("add");
        assert_eq!(got.content, "anything at all");
        assert!(got.thinking.is_empty());
        assert!(got.calls.is_empty());
        assert!(!p.has_tool_support());
        assert!(!p.has_thinking_support());
        assert!(p.preserved_tokens().is_empty());
    }

    #[test]
    fn emit_unambiguous_widens_the_held_back_region_over_whitespace() {
        // "abc \n" + a 4-byte partial "<too": the space AND newline in front of
        // the partial tag must be held back too, because if it completes into a
        // real tag they would have been trimmed.
        let mut buf = String::from("abc \n<too");
        let n = overlap(&buf, "<tool_call>");
        let out = emit_unambiguous(&mut buf, n);
        assert_eq!(out, "abc");
        assert_eq!(buf, " \n<too");
    }
}
