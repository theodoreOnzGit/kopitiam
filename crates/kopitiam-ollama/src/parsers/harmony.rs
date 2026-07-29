//! Harmony -- the response format gpt-oss models speak.
//!
//! **Upstream:** `harmony/harmonyparser.go` (ollama, MIT, Copyright (c) Ollama).
//! Ported against `4713800b08b2ddf5e14acf8398953cf7b12f169b` (2026-07-28).
//!
//! ## Where this file sits, and why it is here
//!
//! Upstream, harmony is its **own top-level Go package**, not one of
//! `model/parsers/`. `ParserForName("harmony")` reaches across and returns
//! `harmony.NewHarmonyMessageHandler()`. We keep it inside `parsers/` because
//! that is the one tree this port owns, and because from the crate's point of
//! view it is just another [`Parser`] implementation. **Divergence, stated
//! plainly:** module location only -- zero behaviour changes.
//!
//! ## Harmony's wire format, in one breath
//!
//! Harmony is not "content plus tags". It is a **stream of framed messages**:
//!
//! ```text
//! <|start|>assistant<|channel|>analysis<|message|>hmm, 2+2...<|end|>
//! <|start|>assistant<|channel|>final<|message|>4<|end|>
//! ^--------^ ^-------------------^ ^---------^ ^-------^
//!   start          header            hdr end     content   end
//! ```
//!
//! So there are TWO machines stacked, exactly like upstream:
//!
//! 1. [`HarmonyParser`] -- purely syntactic. Chews the byte stream into
//!    [`HarmonyEvent`]s (message started, header parsed, content came out,
//!    message ended). It knows nothing about thinking or tools.
//! 2. [`HarmonyMessageHandler`] -- semantic. Reads those events and decides,
//!    from the header's **channel**, whether the content that follows is
//!    thinking, user-visible content, or the JSON body of a tool call.
//!
//! ## The channel is what decides everything, hor
//!
//! | channel | recipient | content goes to |
//! |---|---|---|
//! | `analysis` | none | **thinking** |
//! | `analysis` | `functions.foo` | **tool call** body for `foo` |
//! | `commentary` | none | content |
//! | `commentary` | `functions.foo` | **tool call** body for `foo` |
//! | `final` | -- | content |
//! | anything else | -- | state unchanged (whatever it was) |
//!
//! **What would make this wrong:** treating `analysis` as thinking
//! unconditionally. An `analysis` header that carries a recipient is a *tool
//! call*, and routing its JSON into `thinking` both loses the call and shows the
//! user raw JSON.
//!
//! ## Tool names get renamed, and the rename must be undone
//!
//! Harmony function names have to look like TypeScript identifiers. A user tool
//! called `"get weather"` cannot go on the wire as-is, so [`FunctionNameMap`]
//! converts it to `get_weather` **and remembers the pair**. When the model calls
//! `functions.get_weather`, we map back to `"get weather"` before handing the
//! [`ToolCall`] to the caller. This is why [`Parser::init`] **returns** tools
//! rather than ignoring them -- harmony is the family that made that return
//! value necessary in the first place.
//!
//! The conversion is lossy (`get weather`, `get_weather` and `get-weather` all
//! converge on `get_weather`), so it is **not reversible without the map**. Lose
//! the map and you cannot tell which of the three the model meant -- that is why
//! dupes get `_2`, `_3` suffixes instead of silently colliding.
//!
//! ## Streaming
//!
//! Same contract as every other family: emit the moment it is unambiguous,
//! buffer while it is not. Note one **deliberate difference from the
//! `model/parsers/` families**: harmony's content branch does *not* widen the
//! held-back region over trailing whitespace (no
//! [`trailing_whitespace_len`](super::trailing_whitespace_len) call). Upstream
//! does not, because harmony content is delimited by an explicit `<|end|>` and
//! whitespace in front of it is real content, not framing. Copying the
//! whitespace-widening in from the qwen families would silently eat spaces the
//! user is supposed to see.

use std::collections::HashMap;

use crate::api::{Message, ThinkValue, Tool, ToolCall, ToolCallArguments, ToolCallFunction};

use super::{Parsed, Parser, ParserError, overlap};

/// **Upstream:** the tag fields of `NewHarmonyMessageHandler()`. These are the
/// gpt-oss special tokens, not something we picked.
const MESSAGE_START_TAG: &str = "<|start|>";
const MESSAGE_END_TAG: &str = "<|end|>";
const HEADER_END_TAG: &str = "<|message|>";
/// Marks the channel inside a header. **Upstream:** `parseHeader`.
const CHANNEL_TAG: &str = "<|channel|>";
/// Optional "the body is constrained to this grammar" marker, e.g.
/// `<|constrain|>json`. **Upstream:** `parseHeader`.
const CONSTRAIN_TAG: &str = "<|constrain|>";
/// Harmony namespaces custom tools under `functions.`; built-ins (`python`,
/// `browser.*`) are not namespaced. **Upstream:** `HarmonyMessageHandler.Add`.
const FUNCTIONS_PREFIX: &str = "functions.";

// ---------------------------------------------------------------------------
// Layer 1: the syntactic parser.
// ---------------------------------------------------------------------------

/// **Upstream:** `harmonyParserState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    /// Waiting for `<|start|>`. Anything before it is junk (upstream logs a
    /// warning and throws it away -- so do we, minus the logger).
    #[default]
    LookingForMessageStart,
    /// Inside the header, waiting for `<|message|>`.
    ParsingHeader,
    /// Inside the body, waiting for `<|end|>`.
    ParsingContent,
}

/// What the syntactic layer noticed. **Upstream:** the `HarmonyEvent` interface
/// and its four implementors -- made a proper enum here, since Rust has one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarmonyEvent {
    /// Saw `<|start|>`.
    MessageStart,
    /// Saw `<|message|>`; the header in front of it is parsed.
    HeaderComplete(HarmonyHeader),
    /// A run of body bytes that cannot be part of `<|end|>`.
    ContentEmitted(String),
    /// Saw `<|end|>`.
    MessageEnd,
}

/// A parsed harmony message header. **Upstream:** `HarmonyHeader`.
///
/// Empty string means "absent" in all three fields, matching Go's zero value --
/// harmony never distinguishes an absent channel from an empty one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HarmonyHeader {
    pub role: String,
    pub channel: String,
    /// Who the message is addressed to, e.g. `functions.get_weather`. A
    /// non-empty recipient is what turns a message into a **tool call**.
    pub recipient: String,
}

/// The syntactic layer. **Upstream:** `HarmonyParser`.
#[derive(Debug)]
pub struct HarmonyParser {
    state: State,
    message_start_tag: String,
    message_end_tag: String,
    header_end_tag: String,
    acc: String,
}

impl Default for HarmonyParser {
    /// The gpt-oss tag set. **Upstream:** `NewHarmonyMessageHandler()`.
    fn default() -> Self {
        Self {
            state: State::default(),
            message_start_tag: MESSAGE_START_TAG.to_string(),
            message_end_tag: MESSAGE_END_TAG.to_string(),
            header_end_tag: HEADER_END_TAG.to_string(),
            acc: String::new(),
        }
    }
}

impl HarmonyParser {
    /// Pretend the model already emitted `<|start|>assistant`.
    ///
    /// **Upstream:** `AddImplicitStart`. Needed because the *prompt* ends with
    /// the start tag, so the model's first generated byte is already inside the
    /// header. Without this the parser would sit in
    /// [`State::LookingForMessageStart`] forever and emit nothing at all.
    pub fn add_implicit_start(&mut self) {
        self.acc.push_str("<|start|>assistant");
    }

    /// Same idea, but honouring an **assistant prefill**.
    ///
    /// **Upstream:** `AddImplicitStartOrPrefill`. If the caller pre-seeded the
    /// assistant's reply, the prompt already opened a *specific channel*, so we
    /// must open the same one or the continuation lands in the wrong bucket:
    ///
    /// * prefilled `content` -> the prompt is inside `final`, so seed
    ///   `<|start|>assistant<|channel|>final<|message|>`;
    /// * prefilled `thinking` -> inside `analysis`, seed that instead;
    /// * neither -> plain [`add_implicit_start`](Self::add_implicit_start).
    ///
    /// **What would make this wrong:** seeding the plain start for a
    /// content-prefilled turn. The model continues mid-sentence with no header,
    /// the parser never sees `<|message|>`, and the whole reply is swallowed as
    /// an unterminated header.
    pub fn add_implicit_start_or_prefill(&mut self, last_message: Option<&Message>) {
        if let Some(m) = last_message
            && m.role == "assistant"
        {
            if !m.content.is_empty() {
                self.acc.push_str("<|start|>assistant<|channel|>final<|message|>");
                return;
            } else if !m.thinking.is_empty() {
                self.acc
                    .push_str("<|start|>assistant<|channel|>analysis<|message|>");
                return;
            }
        }
        self.add_implicit_start();
    }

    /// Feed a chunk, get back whatever became unambiguous.
    ///
    /// **Upstream:** `AddContent`. The loop matters: one chunk can carry a whole
    /// message (`<|start|>u<|message|>hi<|end|>`), so we keep eating until the
    /// machine stops making progress. Returning after one step would make the
    /// caller wait for data that is already sitting in the buffer, fully
    /// decided.
    pub fn add_content(&mut self, content: &str) -> Vec<HarmonyEvent> {
        self.acc.push_str(content);

        let mut events = Vec::new();
        let mut keep_looping = true;
        while keep_looping {
            let (new_events, again) = self.eat();
            keep_looping = again;
            events.extend(new_events);
        }
        events
    }

    /// One step of the machine. **Upstream:** `eat`. The `bool` is "call me
    /// again" -- true iff we changed state, since a state change can expose more
    /// decidable input.
    fn eat(&mut self) -> (Vec<HarmonyEvent>, bool) {
        match self.state {
            State::LookingForMessageStart => {
                let Some(idx) = self.acc.find(&self.message_start_tag) else {
                    // No partial-tag cleverness here on purpose: everything
                    // before a start tag is discarded anyway, so there is
                    // nothing to emit early and nothing to protect.
                    return (Vec::new(), false);
                };
                // Upstream logs a warning when `before` is non-empty ("found
                // message start tag in the middle of the content"). We have no
                // logger in this crate, and the byte-level behaviour is the
                // same either way: it is dropped.
                let after = self.acc[idx + self.message_start_tag.len()..].to_string();
                self.acc = after;
                self.state = State::ParsingHeader;
                (vec![HarmonyEvent::MessageStart], true)
            }

            State::ParsingHeader => {
                let Some(idx) = self.acc.find(&self.header_end_tag) else {
                    return (Vec::new(), false);
                };
                let header = self.acc[..idx].to_string();
                let after = self.acc[idx + self.header_end_tag.len()..].to_string();
                self.acc = after;
                self.state = State::ParsingContent;
                (
                    vec![HarmonyEvent::HeaderComplete(parse_header(&header))],
                    true,
                )
            }

            State::ParsingContent => {
                if let Some(idx) = self.acc.find(&self.message_end_tag) {
                    let content = self.acc[..idx].to_string();
                    let after = self.acc[idx + self.message_end_tag.len()..].to_string();
                    self.acc = after;
                    self.state = State::LookingForMessageStart;
                    let mut events = Vec::new();
                    // An empty body emits no content event, only the end. That
                    // is upstream, and a test pins it
                    // (`<|start|>user<|message|><|end|>`).
                    if !content.is_empty() {
                        events.push(HarmonyEvent::ContentEmitted(content));
                    }
                    events.push(HarmonyEvent::MessageEnd);
                    return (events, true);
                }

                let overlap_len = overlap(&self.acc, &self.message_end_tag);
                if overlap_len > 0 {
                    // The tail could still grow into `<|end|>`. Hold exactly
                    // that many bytes, emit the rest. NOTE: no whitespace
                    // widening -- see the module docs for why harmony differs
                    // from the qwen families here.
                    let split = self.acc.len() - overlap_len;
                    let (content, remaining) = super::chop(&self.acc, split);
                    let (content, remaining) = (content.to_string(), remaining.to_string());
                    self.acc = remaining;
                    if content.is_empty() {
                        return (Vec::new(), false);
                    }
                    return (vec![HarmonyEvent::ContentEmitted(content)], false);
                }

                // Nothing ambiguous: the whole buffer is content.
                if self.acc.is_empty() {
                    return (Vec::new(), false);
                }
                let content = std::mem::take(&mut self.acc);
                (vec![HarmonyEvent::ContentEmitted(content)], false)
            }
        }
    }
}

/// Pull `role`, `channel` and `recipient` out of a raw header string.
///
/// **Upstream:** `(*HarmonyParser).parseHeader`.
///
/// The grammar is loose, which is why this is fiddly rather than a regex:
///
/// * the channel is `<|channel|>` glued directly to its name, **no whitespace**
///   (`<|channel|>analysis`), and the name runs to the first whitespace;
/// * the recipient `to=foo` can appear **before or after** the channel tag,
///   which is why it is looked for twice;
/// * a bare `to=foo` with no role at all means role `"tool"` -- the recipient
///   has taken the role slot;
/// * `<|constrain|>` may be glued straight onto the recipient with no space, so
///   the first thing we do is force a space in front of it. Without that,
///   `to=functions.f<|constrain|>json` tokenises as one blob and the recipient
///   comes out as `functions.f<|constrain|>json`.
///
/// Junk tokens between the recipient and the channel are simply ignored
/// (upstream pins this: `assistant to=functions.get_weather abc<|channel|>...`).
///
/// A header with no tokens at all yields the default (all empty). Upstream logs
/// an error and returns the zero value; we return the zero value.
fn parse_header(raw: &str) -> HarmonyHeader {
    let mut header = HarmonyHeader::default();
    let mut raw = raw.to_string();

    // Force a space before `<|constrain|>` so `strings.Fields` sees it as its
    // own token even when the model glued it on. `replacen(.., 1)` is Go's
    // `strings.Replace(.., 1)` -- first occurrence only.
    if raw.contains(CONSTRAIN_TAG) {
        raw = raw.replacen(CONSTRAIN_TAG, &format!(" {CONSTRAIN_TAG}"), 1);
        raw = raw.trim().to_string();
    }

    if let Some(channel_index) = raw.find(CHANNEL_TAG) {
        let before = raw[..channel_index].to_string();
        let after = &raw[channel_index + CHANNEL_TAG.len()..];
        // Channel name = everything up to the first whitespace (or the end).
        let idx = after
            .char_indices()
            .find(|(_, c)| c.is_whitespace())
            .map_or(after.len(), |(i, _)| i);
        header.channel = after[..idx].to_string();
        let after = after[idx..].to_string();
        // Cut the channel tag out and carry on with what is left.
        raw = format!("{before}{after}").trim().to_string();
    }

    // Go's `strings.Fields` splits on runs of Unicode whitespace; Rust's
    // `split_whitespace` is the same job, same White_Space property.
    let mut tokens = raw.split_whitespace();

    let Some(role) = tokens.next() else {
        return header;
    };

    if let Some(recipient) = role.strip_prefix("to=") {
        // No role token at all -- the recipient ate it. Upstream calls this
        // "matches reference code", i.e. it is openai's harmony behaviour, not
        // ollama's invention.
        header.recipient = recipient.to_string();
        header.role = "tool".to_string();
    } else {
        header.role = role.to_string();
    }

    // The recipient may sit AFTER the channel tag instead, so check the next
    // token too -- but only if the role slot did not already supply one.
    if header.recipient.is_empty()
        && let Some(next) = tokens.next()
        && let Some(recipient) = next.strip_prefix("to=")
    {
        header.recipient = recipient.to_string();
    }

    header
}

// ---------------------------------------------------------------------------
// Layer 2: the semantic handler -- this is what `parser_for_name` hands out.
// ---------------------------------------------------------------------------

/// **Upstream:** `harmonyMessageState`. Which bucket the current message body
/// is being poured into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum MessageState {
    #[default]
    Normal,
    Thinking,
    ToolCalling,
}

/// Collects a tool call's body across chunks.
///
/// **Upstream:** `HarmonyToolCallAccumulator`. Deliberately dumb: harmony tool
/// calls arrive **one at a time and always at the end of a message**, so
/// upstream does not even try to parse the JSON until `done`. We keep that --
/// parsing early would just mean re-parsing a half-written object on every
/// chunk and failing every time.
#[derive(Debug, Default)]
pub struct HarmonyToolCallAccumulator {
    acc: String,
    current_tool_name: Option<String>,
}

impl HarmonyToolCallAccumulator {
    /// **Upstream:** `SetToolName`. Called from the header event.
    pub fn set_tool_name(&mut self, name: &str) {
        self.current_tool_name = Some(name.to_string());
    }

    /// **Upstream:** `Add`.
    pub fn add(&mut self, content: &str) {
        self.acc.push_str(content);
    }

    /// Take the accumulated body, keeping the tool name.
    ///
    /// **Upstream:** `Drain` -- and note it clears `acc` but does **not** clear
    /// `currentToolName`. Faithfully kept: draining twice in one generation
    /// would re-report the same tool with an empty body, which then fails JSON
    /// parsing. Upstream only ever drains once (on `done`), and so do we.
    pub fn drain(&mut self) -> (Option<String>, String) {
        let s = std::mem::take(&mut self.acc);
        (self.current_tool_name.clone(), s)
    }

    /// **Upstream:** `Content`.
    pub fn content(&self) -> &str {
        &self.acc
    }
}

/// Two-way map between a caller's tool name and the TypeScript-ish identifier
/// harmony insists on.
///
/// **Upstream:** `FunctionNameMap`.
#[derive(Debug, Default)]
pub struct FunctionNameMap {
    user_to_harmony: HashMap<String, String>,
    harmony_to_user: HashMap<String, String>,
}

impl FunctionNameMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert a caller's tool name and remember the pair.
    ///
    /// **Upstream:** `ConvertAndAdd`. The built-in tools are **exempt from
    /// renaming** -- `browser.open`, `browser.search`, `browser.find` and
    /// `python` are names the model was *trained on*, dots and all. Rename
    /// `browser.search` to `browser_search` and the model no longer recognises
    /// its own built-in. That exact list is upstream's, hard-coded.
    pub fn convert_and_add(&mut self, user_function_name: &str) -> String {
        let mut harmony_function_name = self.derive_name(user_function_name);
        if matches!(
            user_function_name,
            "browser.open" | "browser.search" | "browser.find" | "python"
        ) {
            harmony_function_name = user_function_name.to_string();
        }
        self.user_to_harmony
            .insert(user_function_name.to_string(), harmony_function_name.clone());
        self.harmony_to_user
            .insert(harmony_function_name.clone(), user_function_name.to_string());
        harmony_function_name
    }

    /// Undo a rename. **Upstream:** `OriginalFromConverted`.
    ///
    /// Falls back to the harmony name when there is no mapping (upstream logs a
    /// warning first). That fallback is a **best guess, not a correct answer** --
    /// the conversion is lossy, so without the map there is genuinely no way to
    /// recover which user name was meant. Better to call a plausibly-named tool
    /// than to drop the call.
    pub fn original_from_converted(&self, harmony_function_name: &str) -> String {
        self.harmony_to_user
            .get(harmony_function_name)
            .cloned()
            .unwrap_or_else(|| harmony_function_name.to_string())
    }

    /// Squeeze a name into a valid TypeScript-ish identifier.
    ///
    /// **Upstream:** `convertToValidChars`, limitations and all:
    ///
    /// * space, `-` and `.` become `_`;
    /// * letters, digits, `_` and `$` survive;
    /// * everything else is **deleted**, not replaced;
    /// * nothing left -> `"unnamed"`;
    /// * leading digit -> prepend `_`.
    ///
    /// Upstream notes it does not reject reserved TypeScript keywords and does
    /// not do a real `ID_Start`/`ID_Continue` check. Both limitations are
    /// carried over deliberately -- fixing them would change which names models
    /// see, and nobody has measured what these models were trained on.
    ///
    /// **KNOWN DIVERGENCE from Go, and it is observable -- not theoretical.**
    ///
    /// Go's `unicode.IsLetter` is Unicode **category `L`**. Rust's
    /// `char::is_alphabetic` is the **`Alphabetic` property**, which is `L` plus
    /// `Nl` plus `Other_Alphabetic`. Likewise Go's `unicode.IsDigit` is `Nd`
    /// while Rust's `char::is_numeric` is all of `N`. So characters that *look*
    /// like letters or digits but are formally symbols survive here where
    /// upstream deletes them:
    ///
    /// | input | upstream (Go) | this port |
    /// |---|---|---|
    /// | `"\u{24DE}\u{24DB}\u{24DB}\u{24D0}\u{24DC}\u{24D0}123"` (circled letters) | `_123` | unchanged |
    /// | `"\u{2460}"` (circled one) | `unnamed` | unchanged |
    ///
    /// Upstream's own test pins the first row, so this port **fails that one
    /// fixture on purpose** rather than hand-rolling a category-`L` table, which
    /// would be inventing Unicode data from scratch. Closing the gap exactly
    /// needs a Unicode-general-category dependency, which this crate does not
    /// have and which is the maintainer's call to add.
    ///
    /// **Blast radius, so nobody panics:** a tool name made of circled letters
    /// comes out as a non-ASCII harmony identifier instead of `_123`. Cosmetic
    /// for any realistic tool name, and the two-way [`FunctionNameMap`] still
    /// round-trips it correctly either way -- what goes to the model is what
    /// comes back. Nothing is lost, it is just uglier than upstream's.
    ///
    /// The **leading-digit** check, by contrast, IS exact: Go writes
    /// `unicode.IsDigit(rune(candidate[0]))`, casting the first *byte*, so only
    /// ASCII `0-9` can ever trigger it. `is_ascii_digit` on the first byte is
    /// the same test, not an approximation.
    fn convert_to_valid_chars(&self, user_function_name: &str) -> String {
        let candidate: String = user_function_name
            .chars()
            .filter_map(|c| match c {
                ' ' | '-' | '.' => Some('_'),
                c if c.is_alphabetic() || c.is_numeric() || c == '_' || c == '$' => Some(c),
                _ => None,
            })
            .collect();

        if candidate.is_empty() {
            return "unnamed".to_string();
        }

        match candidate.as_bytes().first() {
            Some(b) if b.is_ascii_digit() => format!("_{candidate}"),
            _ => candidate,
        }
    }

    /// Pick a free name, suffixing `_2`, `_3`, ... on collision.
    ///
    /// **Upstream:** `deriveName`. Counting **starts at 2** on purpose: the
    /// first claimant keeps the bare name, so a run of dupes reads `f`, `f_2`,
    /// `f_3` rather than `f_1`, `f_2`.
    ///
    /// Collisions are real, not theoretical -- `get weather`, `get_weather` and
    /// `get-weather` all convert to the same string.
    fn derive_name(&self, user_function_name: &str) -> String {
        let original_candidate = self.convert_to_valid_chars(user_function_name);
        let mut candidate = original_candidate.clone();

        let mut count = 2;
        while self.harmony_to_user.contains_key(&candidate) {
            candidate = format!("{original_candidate}_{count}");
            count += 1;
        }

        candidate
    }
}

/// The harmony [`Parser`]. **Upstream:** `HarmonyMessageHandler`.
///
/// This is what `parser_for_name("harmony")` returns, and it is what gpt-oss
/// models get.
#[derive(Debug)]
pub struct HarmonyMessageHandler {
    state: MessageState,
    parser: HarmonyParser,
    function_name_map: FunctionNameMap,
    tool_accumulator: HarmonyToolCallAccumulator,
}

impl Default for HarmonyMessageHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl HarmonyMessageHandler {
    /// **Upstream:** `NewHarmonyMessageHandler`.
    ///
    /// Upstream also carries a `convertedTools map[string]struct{}` that is
    /// written in `Init` and **never read anywhere**. Left out here rather than
    /// ported as dead weight -- noted so a future reader diffing against Go does
    /// not think it was missed.
    pub fn new() -> Self {
        Self {
            state: MessageState::Normal,
            parser: HarmonyParser::default(),
            function_name_map: FunctionNameMap::new(),
            tool_accumulator: HarmonyToolCallAccumulator::default(),
        }
    }

    /// Route one chunk's events into the three buckets.
    ///
    /// **Upstream:** `(*HarmonyMessageHandler).AddContent`. Returns
    /// `(content, thinking, tool_content)` -- the first two are finished, the
    /// third still has to go through the accumulator and a JSON parse.
    fn route(&mut self, content: &str) -> (String, String, String) {
        let mut content_sb = String::new();
        let mut thinking_sb = String::new();
        let mut tool_content_sb = String::new();

        for event in self.parser.add_content(content) {
            match event {
                HarmonyEvent::HeaderComplete(header) => match header.channel.as_str() {
                    // `analysis` with a recipient is a TOOL CALL, not thinking.
                    // The recipient is the tool name -- `browser.search` for a
                    // built-in, `functions.calc` for a custom one.
                    "analysis" | "commentary" if !header.recipient.is_empty() => {
                        self.state = MessageState::ToolCalling;
                        self.tool_accumulator.set_tool_name(&header.recipient);
                    }
                    "analysis" => self.state = MessageState::Thinking,
                    "commentary" | "final" => self.state = MessageState::Normal,
                    // Any other channel (including none) leaves the state
                    // alone. Upstream's switch has no default branch, and that
                    // fall-through is load-bearing: a plain `<|start|>user...`
                    // header carries no channel and must keep pouring into
                    // whatever bucket was already open.
                    _ => {}
                },
                HarmonyEvent::ContentEmitted(c) => match self.state {
                    MessageState::Normal => content_sb.push_str(&c),
                    MessageState::Thinking => thinking_sb.push_str(&c),
                    MessageState::ToolCalling => tool_content_sb.push_str(&c),
                },
                HarmonyEvent::MessageEnd => self.state = MessageState::Normal,
                HarmonyEvent::MessageStart => {}
            }
        }

        (content_sb, thinking_sb, tool_content_sb)
    }
}

impl Parser for HarmonyMessageHandler {
    /// **Upstream:** `(*HarmonyMessageHandler).Init`.
    ///
    /// Note this does **not** reset the parser -- it *appends* the implicit
    /// start to the accumulator. A `HarmonyMessageHandler` is therefore one-shot
    /// per generation: build a fresh one (via `parser_for_name`) each turn, do
    /// not re-`init` a used one, or you seed a second `<|start|>` on top of
    /// leftover bytes.
    fn init(
        &mut self,
        tools: Vec<Tool>,
        last_message: Option<&Message>,
        _think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        if last_message.is_some() {
            self.parser.add_implicit_start_or_prefill(last_message);
        } else {
            self.parser.add_implicit_start();
        }

        self.tool_accumulator = HarmonyToolCallAccumulator::default();

        if tools.is_empty() {
            return tools;
        }

        // Rename every tool, and hand the RENAMED list back so the renderer
        // shows the model the same identifiers this parser will later reverse.
        tools
            .into_iter()
            .map(|mut tool| {
                if !tool.function.name.is_empty() {
                    tool.function.name = self.function_name_map.convert_and_add(&tool.function.name);
                }
                tool
            })
            .collect()
    }

    /// **Upstream:** `(*HarmonyMessageHandler).Add`.
    ///
    /// Tool calls are only assembled when `done` is set, because harmony emits
    /// at most one per message and always last -- so there is nothing to gain
    /// from trying earlier, and a partial JSON body would only produce a parse
    /// error per chunk.
    fn add(&mut self, s: &str, done: bool) -> Result<Parsed, ParserError> {
        let (content, thinking, tool_content) = self.route(s);
        if !tool_content.is_empty() {
            self.tool_accumulator.add(&tool_content);
        }

        let mut calls = Vec::new();
        if done {
            let (tool_name, raw) = self.tool_accumulator.drain();
            if let Some(tool_name) = tool_name {
                // Strip the `functions.` namespace but leave built-ins
                // (`python`, `browser.search`) alone -- they were never
                // namespaced and were never renamed.
                let name = tool_name
                    .strip_prefix(FUNCTIONS_PREFIX)
                    .unwrap_or(&tool_name);
                let name = self.function_name_map.original_from_converted(name);

                let arguments: ToolCallArguments = serde_json::from_str(&raw).map_err(|e| {
                    ParserError::HarmonyToolCall {
                        raw: raw.clone(),
                        source: e,
                    }
                })?;

                calls.push(ToolCall {
                    function: ToolCallFunction {
                        name,
                        arguments,
                        ..Default::default()
                    },
                    ..Default::default()
                });
            }
        }

        Ok(Parsed {
            content,
            thinking,
            calls,
        })
    }

    /// **Upstream:** `PreservedTokens`, comment and all: `<|call|>` is an EOG
    /// marker for tool calls, so it is deliberately **left off** the list --
    /// llama-server is supposed to stop on it. Preserve only the structural
    /// tokens this parser needs to see its own boundaries.
    fn preserved_tokens(&self) -> Vec<&'static str> {
        vec![
            MESSAGE_START_TAG,
            MESSAGE_END_TAG,
            HEADER_END_TAG,
            CHANNEL_TAG,
            CONSTRAIN_TAG,
        ]
    }

    fn has_tool_support(&self) -> bool {
        true
    }

    fn has_thinking_support(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parser() -> HarmonyParser {
        HarmonyParser::default()
    }

    /// Upstream `TestHeaderParsing`, ported verbatim as ground truth.
    #[test]
    fn a_header_yields_its_role_channel_and_recipient() {
        // (raw, role, channel, recipient)
        let cases: &[(&str, &str, &str, &str)] = &[
            ("assistant<|channel|>analysis", "assistant", "analysis", ""),
            (
                "assistant<|channel|>analysis to=functions.get_weather",
                "assistant",
                "analysis",
                "functions.get_weather",
            ),
            (
                "assistant to=functions.get_weather<|channel|>analysis",
                "assistant",
                "analysis",
                "functions.get_weather",
            ),
            // No role token at all -- the recipient takes the role slot and the
            // role becomes "tool".
            (
                "to=functions.get_weather<|channel|>analysis",
                "tool",
                "analysis",
                "functions.get_weather",
            ),
            // Junk between recipient and channel is ignored.
            (
                "assistant to=functions.get_weather abc<|channel|>analysis",
                "assistant",
                "analysis",
                "functions.get_weather",
            ),
            (
                "assistant<|channel|>commentary to=functions.get_weather <|constrain|>json",
                "assistant",
                "commentary",
                "functions.get_weather",
            ),
            (
                "assistant to=functions.get_weather<|channel|>commentary <|constrain|>json",
                "assistant",
                "commentary",
                "functions.get_weather",
            ),
            // Constrain tag glued on with no space -- this is why parse_header
            // forces a space in front of it first.
            (
                "assistant<|channel|>commentary to=functions.get_weather<|constrain|>json",
                "assistant",
                "commentary",
                "functions.get_weather",
            ),
            (
                "assistant to=functions.get_weather<|channel|>commentary<|constrain|>json",
                "assistant",
                "commentary",
                "functions.get_weather",
            ),
        ];

        for (raw, role, channel, recipient) in cases {
            let got = parse_header(raw);
            assert_eq!(&got.role, role, "role, for {raw:?}");
            assert_eq!(&got.channel, channel, "channel, for {raw:?}");
            assert_eq!(&got.recipient, recipient, "recipient, for {raw:?}");
        }
    }

    /// Upstream `TestHarmonyParserHeaderEvent`.
    #[test]
    fn the_first_header_event_of_a_stream_carries_the_parsed_header() {
        let cases: &[(&str, bool, &str, &str, &str)] = &[
            (
                "<|start|>user<|message|>What is 2 + 2?<|end|>",
                false,
                "user",
                "",
                "",
            ),
            (
                "<|start|>assistant<|channel|>analysis<|message|>What is 2 + 2?<|end|>",
                false,
                "assistant",
                "analysis",
                "",
            ),
            (
                "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"San Francisco\"}<|call|><|start|>functions.get_weather to=assistant<|message|>{\"sunny\": true, \"temperature\": 20}<|end|>",
                false,
                "assistant",
                "commentary",
                "functions.get_weather",
            ),
            (
                "<|channel|>analysis<|message|>User asks weather in SF.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{\"location\":\"San Francisco, CA\"}<|call|>",
                true,
                "assistant",
                "analysis",
                "",
            ),
        ];

        for (input, implicit_start, role, channel, recipient) in cases {
            let mut p = parser();
            if *implicit_start {
                p.add_implicit_start();
            }
            let events = p.add_content(input);
            assert!(!events.is_empty(), "no events, for {input:?}");
            let header = events
                .iter()
                .find_map(|e| match e {
                    HarmonyEvent::HeaderComplete(h) => Some(h),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("no header event, for {input:?}"));
            assert_eq!(&header.role, role, "role, for {input:?}");
            assert_eq!(&header.channel, channel, "channel, for {input:?}");
            assert_eq!(&header.recipient, recipient, "recipient, for {input:?}");
        }
    }

    /// Upstream `TestHarmonyParserNonStreaming` -- the exact event sequences.
    #[test]
    fn a_whole_message_in_one_chunk_produces_the_full_event_sequence() {
        use HarmonyEvent::*;

        let hdr = |role: &str, channel: &str, recipient: &str| {
            HeaderComplete(HarmonyHeader {
                role: role.into(),
                channel: channel.into(),
                recipient: recipient.into(),
            })
        };

        let cases: Vec<(&str, bool, Vec<HarmonyEvent>)> = vec![
            (
                "<|start|>user<|message|>What is 2 + 2?<|end|>",
                false,
                vec![
                    MessageStart,
                    hdr("user", "", ""),
                    ContentEmitted("What is 2 + 2?".into()),
                    MessageEnd,
                ],
            ),
            (
                "<|start|>assistant<|channel|>analysis<|message|>The answer is 4<|end|>",
                false,
                vec![
                    MessageStart,
                    hdr("assistant", "analysis", ""),
                    ContentEmitted("The answer is 4".into()),
                    MessageEnd,
                ],
            ),
            (
                "<|start|>assistant<|channel|>commentary to=functions.calc<|message|>Computing...<|end|>",
                false,
                vec![
                    MessageStart,
                    hdr("assistant", "commentary", "functions.calc"),
                    ContentEmitted("Computing...".into()),
                    MessageEnd,
                ],
            ),
            // Empty body: NO content event, just start/header/end.
            (
                "<|start|>user<|message|><|end|>",
                false,
                vec![MessageStart, hdr("user", "", ""), MessageEnd],
            ),
            (
                "<|start|>user<|message|>Hello<|end|><|start|>assistant<|message|>Hi!<|end|>",
                false,
                vec![
                    MessageStart,
                    hdr("user", "", ""),
                    ContentEmitted("Hello".into()),
                    MessageEnd,
                    MessageStart,
                    hdr("assistant", "", ""),
                    ContentEmitted("Hi!".into()),
                    MessageEnd,
                ],
            ),
            (
                "<|channel|>analysis<|message|>Thinking about the request<|end|>",
                true,
                vec![
                    MessageStart,
                    hdr("assistant", "analysis", ""),
                    ContentEmitted("Thinking about the request".into()),
                    MessageEnd,
                ],
            ),
        ];

        for (input, implicit_start, want) in cases {
            let mut p = parser();
            if implicit_start {
                p.add_implicit_start();
            }
            assert_eq!(p.add_content(input), want, "for {input:?}");
        }
    }

    /// Upstream `TestHarmonyParserStreaming` -- each step's exact events.
    #[test]
    fn streamed_chunks_emit_the_moment_they_stop_being_ambiguous() {
        use HarmonyEvent::*;

        let hdr = |role: &str, channel: &str, recipient: &str| {
            HeaderComplete(HarmonyHeader {
                role: role.into(),
                channel: channel.into(),
                recipient: recipient.into(),
            })
        };

        /// One streamed chunk and the events it must produce.
        type Step<'a> = (&'a str, Vec<HarmonyEvent>);
        /// (description, implicit_start, steps).
        type Case<'a> = (&'a str, bool, Vec<Step<'a>>);

        let cases: Vec<Case<'_>> = vec![
            (
                "simple message streamed character by character",
                false,
                vec![
                    ("<", vec![]),
                    ("|", vec![]),
                    ("start|>u", vec![MessageStart]),
                    ("ser<|mess", vec![]),
                    (
                        "age|>Hi",
                        vec![hdr("user", "", ""), ContentEmitted("Hi".into())],
                    ),
                    (" there", vec![ContentEmitted(" there".into())]),
                    ("<|e", vec![]),
                    ("nd|>", vec![MessageEnd]),
                ],
            ),
            (
                "message with channel streamed",
                false,
                vec![
                    ("<|start|>assistant", vec![MessageStart]),
                    ("<|chan", vec![]),
                    ("nel|>analysis", vec![]),
                    ("<|message|>", vec![hdr("assistant", "analysis", "")]),
                    ("Thinking", vec![ContentEmitted("Thinking".into())]),
                    ("...", vec![ContentEmitted("...".into())]),
                    ("<|end|>", vec![MessageEnd]),
                ],
            ),
            (
                "message with channel and recipient",
                false,
                vec![
                    (
                        "<|start|>assistant<|channel|>commentary to=functions.calc<|message|>",
                        vec![MessageStart, hdr("assistant", "commentary", "functions.calc")],
                    ),
                    ("{\"x\": 5}", vec![ContentEmitted("{\"x\": 5}".into())]),
                    ("<|end|>", vec![MessageEnd]),
                ],
            ),
            (
                "recipient before channel",
                false,
                vec![
                    (
                        "<|start|>assistant to=functions.calc<|channel|>commentary<|message|>",
                        vec![MessageStart, hdr("assistant", "commentary", "functions.calc")],
                    ),
                    ("{\"x\": 5}", vec![ContentEmitted("{\"x\": 5}".into())]),
                    ("<|end|>", vec![MessageEnd]),
                ],
            ),
            (
                "implicit start with channel",
                true,
                vec![
                    ("<|channel|>thinking", vec![MessageStart]),
                    ("<|message|>", vec![hdr("assistant", "thinking", "")]),
                    (
                        "Processing request",
                        vec![ContentEmitted("Processing request".into())],
                    ),
                    ("<|end|>", vec![MessageEnd]),
                ],
            ),
            (
                "multiple messages streamed",
                false,
                vec![
                    (
                        "<|start|>user<|message|>Hello<|end|>",
                        vec![
                            MessageStart,
                            hdr("user", "", ""),
                            ContentEmitted("Hello".into()),
                            MessageEnd,
                        ],
                    ),
                    ("<|start|>", vec![MessageStart]),
                    ("assistant<|message|>", vec![hdr("assistant", "", "")]),
                    ("Hi!", vec![ContentEmitted("Hi!".into())]),
                    ("<|end|>", vec![MessageEnd]),
                ],
            ),
            (
                "empty message",
                false,
                vec![(
                    "<|start|>system<|message|><|end|>",
                    vec![MessageStart, hdr("system", "", ""), MessageEnd],
                )],
            ),
            // The regression this whole design exists for: a tail that LOOKS
            // like the start of `<|end|>` but turns out to be `<|example|>`.
            // It must be held back, then released intact -- not leaked, not
            // eaten.
            (
                "partial tag that looks like end but isn't",
                false,
                vec![
                    (
                        "<|start|>user<|message|>test<|e",
                        vec![
                            MessageStart,
                            hdr("user", "", ""),
                            ContentEmitted("test".into()),
                        ],
                    ),
                    (
                        "xample|>more",
                        vec![ContentEmitted("<|example|>more".into())],
                    ),
                    ("<|end|>", vec![MessageEnd]),
                ],
            ),
        ];

        for (desc, implicit_start, steps) in cases {
            let mut p = parser();
            if implicit_start {
                p.add_implicit_start();
            }
            for (i, (input, want)) in steps.iter().enumerate() {
                assert_eq!(&p.add_content(input), want, "{desc}: step {i}, {input:?}");
            }
        }
    }

    /// Upstream `TestFunctionConvertToValidChars`.
    #[test]
    fn a_tool_name_is_squeezed_into_a_typescript_identifier() {
        let m = FunctionNameMap::new();
        let cases: &[(&str, &str)] = &[
            ("get weather", "get_weather"),
            ("get-weather", "get_weather"),
            ("get.weather", "get_weather"),
            ("get weather!", "get_weather"),
            ("a\u{1FAE0}bc", "abc"),
            ("\u{1FAE0}", "unnamed"),
            ("123", "_123"),
            ("$", "$"),
            // Weird-but-real Unicode letters survive. Upstream flags this as
            // "we might want ASCII equivalents in future" -- kept as-is.
            ("\u{1D4F8}\u{1D4F5}\u{1D4F5}\u{1D4EA}\u{1D4F6}\u{1D4EA}", "\u{1D4F8}\u{1D4F5}\u{1D4F5}\u{1D4EA}\u{1D4F6}\u{1D4EA}"),
        ];
        for (input, want) in cases {
            assert_eq!(&m.convert_to_valid_chars(input), want, "for {input:?}");
        }
    }

    /// The one upstream fixture this port does **not** match, pinned so the gap
    /// is a recorded fact rather than a surprise. Circled letters
    /// (`\u{24DE}` and friends) are category `So` but carry the
    /// `Other_Alphabetic` property, so Go's `unicode.IsLetter` deletes them and
    /// Rust's `char::is_alphabetic` keeps them.
    ///
    /// Upstream `TestFunctionConvertToValidChars` wants `"_123"` here. Closing
    /// it exactly needs a Unicode-general-category dependency -- see
    /// [`FunctionNameMap::convert_to_valid_chars`] for why that is not this
    /// port's call to make. If someone adds that dependency later, this test is
    /// the one to flip back to upstream's value.
    #[test]
    fn circled_letters_survive_here_but_upstream_deletes_them() {
        let m = FunctionNameMap::new();
        let input = "\u{24DE}\u{24DB}\u{24DB}\u{24D0}\u{24DC}\u{24D0}123";
        let upstream_wants = "_123";
        let we_produce = input;

        assert_eq!(m.convert_to_valid_chars(input), we_produce);
        assert_ne!(
            m.convert_to_valid_chars(input),
            upstream_wants,
            "if this now matches upstream, the divergence is closed -- delete \
             this test and restore the fixture in the table above"
        );
    }

    /// Upstream `TestFunctionConvertAndAdd` -- dupe handling starts at `_2`.
    #[test]
    fn duplicate_tool_names_get_numbered_suffixes_starting_at_two() {
        let cases: &[(&[&str], &[&str])] = &[
            (&["get weather", "get weather"], &["get_weather", "get_weather_2"]),
            (
                &["get weather", "get_weather", "get-weather"],
                &["get_weather", "get_weather_2", "get_weather_3"],
            ),
            (
                &["get weather", "get_weather", "get-weather", "something-different"],
                &["get_weather", "get_weather_2", "get_weather_3", "something_different"],
            ),
            (
                &["a", "a", "b", "a", "a", "b", "a"],
                &["a", "a_2", "b", "a_3", "a_4", "b_2", "a_5"],
            ),
            // Built-ins keep their dots -- the model was trained on them.
            (
                &["browser.open", "python", "not.a.built-in.function", "browser.not_a_real_built_in"],
                &["browser.open", "python", "not_a_built_in_function", "browser_not_a_real_built_in"],
            ),
        ];

        for (inputs, wants) in cases {
            let mut m = FunctionNameMap::new();
            for (input, want) in inputs.iter().zip(wants.iter()) {
                let got = m.convert_and_add(input);
                assert_eq!(&got, want, "for {input:?}");
                assert_eq!(m.user_to_harmony.get(*input), Some(&got.clone()));
                assert_eq!(m.harmony_to_user.get(&got), Some(&input.to_string()));
            }
        }
    }

    // --- the handler, i.e. what `parser_for_name("harmony")` hands out --------

    fn handler() -> HarmonyMessageHandler {
        let mut h = HarmonyMessageHandler::new();
        h.init(Vec::new(), None, None);
        h
    }

    #[test]
    fn an_analysis_channel_without_a_recipient_is_thinking() {
        let mut h = handler();
        let got = h
            .add(
                "<|channel|>analysis<|message|>let me think<|end|><|start|>assistant<|channel|>final<|message|>the answer<|end|>",
                true,
            )
            .expect("add");
        assert_eq!(got.thinking, "let me think");
        assert_eq!(got.content, "the answer");
        assert!(got.calls.is_empty());
    }

    /// The distinction that matters: `analysis` **with** a recipient is a tool
    /// call, not thinking. Get this backwards and the user sees raw JSON.
    #[test]
    fn an_analysis_channel_with_a_recipient_is_a_tool_call_not_thinking() {
        let mut h = HarmonyMessageHandler::new();
        h.init(Vec::new(), None, None);
        let got = h
            .add(
                "<|channel|>analysis to=functions.get_weather<|message|>{\"city\":\"Singapore\"}<|end|>",
                true,
            )
            .expect("add");
        assert!(got.thinking.is_empty());
        assert!(got.content.is_empty());
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "get_weather");
        assert_eq!(
            got.calls[0].function.arguments.get("city"),
            Some(&json!("Singapore"))
        );
    }

    #[test]
    fn a_commentary_tool_call_maps_back_to_the_callers_original_tool_name() {
        let mut h = HarmonyMessageHandler::new();
        // Caller's tool is named "get weather" -- illegal as a harmony
        // identifier, so init renames it, and the reply must be mapped back.
        let tool = Tool {
            tool_type: "function".into(),
            function: crate::api::ToolFunction {
                name: "get weather".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        let renamed = h.init(vec![tool], None, None);
        assert_eq!(renamed[0].function.name, "get_weather");

        let got = h
            .add(
                "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"SG\"}<|end|>",
                true,
            )
            .expect("add");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "get weather");
    }

    /// Arguments must come back in the order the model wrote them -- see
    /// [`ToolCallArguments`].
    #[test]
    fn tool_call_arguments_keep_the_models_key_order() {
        let mut h = HarmonyMessageHandler::new();
        h.init(Vec::new(), None, None);
        let got = h
            .add(
                "<|channel|>commentary to=functions.f<|message|>{\"zebra\":1,\"apple\":2,\"mango\":3}<|end|>",
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
        assert_eq!(keys, ["zebra", "apple", "mango"]);
    }

    /// One byte at a time must give the same answer as one big chunk.
    #[test]
    fn feeding_one_byte_at_a_time_gives_the_same_answer_as_one_big_chunk() {
        let input = "<|channel|>analysis<|message|>thinking hard<|end|><|start|>assistant<|channel|>final<|message|>Hello there<|end|>";

        let mut whole = handler();
        let want = whole.add(input, true).expect("add");

        let mut streamed = handler();
        let mut got = Parsed::default();
        for (i, ch) in input.char_indices() {
            let piece = &input[i..i + ch.len_utf8()];
            let part = streamed
                .add(piece, i + ch.len_utf8() == input.len())
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
    fn an_assistant_content_prefill_opens_the_final_channel() {
        let mut h = HarmonyMessageHandler::new();
        let last = Message::new("assistant", "Sure, here goes:");
        h.init(Vec::new(), Some(&last), None);
        // No header at all in the stream -- the prefill already opened `final`.
        let got = h.add(" and here is the rest.", true).expect("add");
        assert_eq!(got.content, " and here is the rest.");
        assert!(got.thinking.is_empty());
    }

    #[test]
    fn an_assistant_thinking_prefill_opens_the_analysis_channel() {
        let mut h = HarmonyMessageHandler::new();
        let last = Message {
            role: "assistant".into(),
            thinking: "so far I reckon".into(),
            ..Default::default()
        };
        h.init(Vec::new(), Some(&last), None);
        let got = h.add(" ... carry on reasoning", true).expect("add");
        assert_eq!(got.thinking, " ... carry on reasoning");
        assert!(got.content.is_empty());
    }

    /// A tool name with no body is a hard error, not a silent drop -- the caller
    /// has to know the model produced a call it could not honour.
    #[test]
    fn a_tool_call_with_an_unparseable_body_is_a_hard_error() {
        let mut h = handler();
        let err = h
            .add(
                "<|channel|>commentary to=functions.f<|message|>not json at all<|end|>",
                true,
            )
            .expect_err("should fail");
        assert!(matches!(err, ParserError::HarmonyToolCall { .. }));
    }

    #[test]
    fn harmony_advertises_the_structural_tokens_but_not_the_eog_call_marker() {
        let h = handler();
        let toks = h.preserved_tokens();
        for t in ["<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|constrain|>"] {
            assert!(toks.contains(&t), "missing {t}");
        }
        // `<|call|>` is an end-of-generation marker; llama-server is meant to
        // stop on it, so it must NOT be preserved.
        assert!(!toks.contains(&"<|call|>"));
        assert!(h.has_tool_support());
        assert!(h.has_thinking_support());
    }

    #[test]
    fn junk_before_the_first_start_tag_is_thrown_away() {
        let mut p = parser();
        let events = p.add_content("garbage<|start|>user<|message|>hi<|end|>");
        assert_eq!(
            events,
            vec![
                HarmonyEvent::MessageStart,
                HarmonyEvent::HeaderComplete(HarmonyHeader {
                    role: "user".into(),
                    ..Default::default()
                }),
                HarmonyEvent::ContentEmitted("hi".into()),
                HarmonyEvent::MessageEnd,
            ]
        );
    }
}
