//! Thinking tags -- pulling a model's chain-of-thought out of the token stream.
//!
//! **Upstream:** `thinking/parser.go` and `thinking/template.go` (ollama, MIT,
//! Copyright (c) Ollama), ported against `4713800b08b2ddf5e14acf8398953cf7b12f169b`
//! (2026-07-28). Where KOPITIAM and ollama disagree, ollama wins; every place we
//! go our own way says so at the point of divergence.
//!
//! ## The one job
//!
//! A reasoning model don't hand you two fields. It hands you **one stream of
//! text**, and somewhere inside got a pair of tags -- `<think>` ... `</think>`
//! for the Qwen3 family, something else for the next family -- and everything
//! between them is the model thinking out loud, not the answer. Somebody must
//! split the two, and it must happen **while the tokens are still arriving**,
//! because the whole point of streaming is the user sees text early.
//!
//! Two halves live here:
//!
//! * [`Parser`] -- the streaming state machine that does the splitting.
//! * [`infer_tags`] -- sniffs a chat template to work out *which* tags this
//!   model even uses, so nothing is hardcoded per model.
//!
//! ## Why the parser cannot just `split()` and be done
//!
//! Tokens arrive in whatever chunks the runtime feels like. A closing tag can
//! land **straddling two chunks**: `"...abc</th"` then `"ink>def"`. If you emit
//! greedily you leak `</th` into the user's answer; if you buffer everything you
//! kill streaming. So the rule is:
//!
//! > Emit the moment the text is **unambiguous**, buffer only the part that is
//! > still ambiguous.
//!
//! "Still ambiguous" means: the tail of what we hold could yet turn out to be
//! the start of the closing tag. [`overlap`] measures exactly that tail, and it
//! is the whole reason this module is a state machine instead of a one-liner.
//! Note the tail is only a *candidate* -- `"abc</th"` + `"ing>"` is not a
//! closing tag at all, and the buffered `</th` must then be released **as
//! thinking content**, not swallowed. That case is tested (upstream calls it
//! `partial closing tag fakeout`), and it is where a hand-rolled version usually
//! go wrong.
//!
//! **What would make this wrong:** emitting any byte twice, dropping any byte,
//! or reordering them. Concatenate every `thinking` this parser ever returns and
//! concatenate every `content`, and in order they must reconstruct the input
//! exactly -- minus the two tags themselves and the whitespace the states below
//! deliberately eat. Nothing else may go missing.
//!
//! ## Whitespace, and why there are five states not three
//!
//! Models like to write `<think>\n\nthoughts\n\n</think>\n\nAnswer`. Nobody want
//! that leading `\n\n` in their answer lah. So the machine got two extra states
//! whose only job is to eat whitespace -- one right after the opening tag, one
//! right after the closing tag -- and they must *survive across chunks*, since
//! the whitespace and the first real character often arrive separately. See
//! [`ThinkingState`].
//!
//! ```
//! use kopitiam_ollama::thinking::Parser;
//!
//! let mut p = Parser::new("<think>", "</think>");
//!
//! // The thought so far is unambiguous -- out it goes.
//! let a = p.add_content("<think>hmm");
//! assert_eq!((a.thinking.as_str(), a.content.as_str()), ("hmm", ""));
//!
//! // `</th` might be a closing tag, so it stays buffered; ` ok` is safe.
//! let b = p.add_content(" ok</th");
//! assert_eq!((b.thinking.as_str(), b.content.as_str()), (" ok", ""));
//!
//! // Now it disambiguates: it really was the closing tag.
//! let c = p.add_content("ink> world");
//! assert_eq!((c.thinking.as_str(), c.content.as_str()), ("", "world"));
//! ```

use std::fmt;

use crate::gotmpl::{
    Template,
    parse::{Arg, Node, Pipeline},
};

// ---------------------------------------------------------------------------
// The state machine
// ---------------------------------------------------------------------------

/// Where the [`Parser`] currently sits in the stream.
///
/// **Upstream:** `thinking/parser.go`'s `thinkingState` constants -- same five,
/// same order, same meanings. Public here (upstream keeps it package-private)
/// because it is genuinely useful to assert on in tests and to show in a debug
/// UI: knowing the parser is stuck in [`LookingForOpening`] is the fastest way
/// to spot a model whose tags were inferred wrong.
///
/// [`LookingForOpening`]: ThinkingState::LookingForOpening
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingState {
    /// Looking for the opening tag, and so far only whitespace has come in.
    ///
    /// This is the state that decides whether there is any thinking **at all**.
    /// The moment a non-whitespace character shows up that is not the start of
    /// the opening tag, thinking is declared skipped and we jump straight to
    /// [`ThinkingDone`](ThinkingState::ThinkingDone) -- a `<think>` appearing
    /// *later* in the stream is then just ordinary text, never a tag. Sounds
    /// harsh, but it is what stops a model quoting the string `<think>` inside
    /// its answer from hijacking the whole split.
    #[default]
    LookingForOpening,
    /// Opening tag seen, still chewing through the whitespace behind it; no real
    /// thinking character yet.
    ThinkingStartedEatingWhitespace,
    /// Inside the thought. Closing tag not seen yet.
    Thinking,
    /// Closing tag seen, still chewing the whitespace before the real answer.
    ThinkingDoneEatingWhitespace,
    /// Closing tag seen and at least one real answer character emitted. From
    /// here everything is content, passed straight through.
    ThinkingDone,
}

impl fmt::Display for ThinkingState {
    /// **Upstream:** `(thinkingState).String()`. Same spellings, because they
    /// show up in test failure messages and it is nice when ours read the same
    /// as ollama's.
    ///
    /// Upstream also got a `"Unknown"` arm for an out-of-range `int`. A Rust
    /// enum cannot hold a value outside the five, so that arm cannot exist here
    /// -- deliberate divergence, and a strictly safer one.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::LookingForOpening => "LookingForOpening",
            Self::ThinkingStartedEatingWhitespace => "ThinkingStartedEatingWhitespace",
            Self::Thinking => "Thinking",
            Self::ThinkingDoneEatingWhitespace => "ThinkingDoneEatingWhitespace",
            Self::ThinkingDone => "ThinkingDone",
        };
        f.write_str(s)
    }
}

/// What one call to [`Parser::add_content`] managed to release.
///
/// **Deliberate divergence:** upstream `AddContent` returns a bare
/// `(string, string)` and every call site has to remember which one is which.
/// Two same-typed strings in a tuple is exactly the shape that reads fine on the
/// day you write it and gets swapped six months later, so we name them. Pure
/// ergonomics -- behaviour identical, order identical (thinking first).
///
/// Either field can be empty, and usually one of them is. Empty means "nothing
/// released this round", **not** "end of stream".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Emitted {
    /// Chain-of-thought text, safe to show in a "thinking..." pane.
    pub thinking: String,
    /// Answer text, safe to show to the user right now.
    pub content: String,
}

/// The streaming thinking-tag splitter.
///
/// **Upstream:** `thinking.Parser` in `thinking/parser.go`.
///
/// Feed it every chunk the model produces, in order, via
/// [`add_content`](Parser::add_content). It buffers internally only what is
/// still ambiguous.
///
/// **Contract, and what would make it wrong:**
///
/// * Chunks must be fed **in order** and **none skipped**. This thing is a state
///   machine over a single stream -- reuse the same `Parser` for a second
///   response and you inherit a half-eaten state. One response, one `Parser`.
/// * Both tags should be **non-empty**. Upstream's callers all guard with
///   `openingTag != "" && closingTag != ""` before constructing (see
///   `server/routes.go`), so a zero-length tag sits outside the ported
///   behaviour; [`infer_tags`] can never hand you one. Pass one anyway and you
///   get a degenerate-but-safe result, not a panic -- see [`Parser::new`].
/// * Chunk boundaries may fall **anywhere**, including mid-tag. That is the
///   normal case, not the edge case.
///
/// If the runtime already fed the opening tag into the prompt (some templates
/// end the prompt with `<think>`, so the model never emits it), upstream primes
/// the parser by calling `AddContent(openingTag)` once before streaming --
/// `server/routes.go` does exactly that. Same trick works here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parser {
    state: ThinkingState,
    opening_tag: String,
    closing_tag: String,
    /// Everything received but not yet released. Holds only the ambiguous tail
    /// once the machine is in [`ThinkingState::Thinking`].
    acc: String,
}

impl Parser {
    /// Make a parser for one response, given the tag pair for that model.
    ///
    /// Get the tags from [`infer_tags`] -- don't hardcode `<think>`, because
    /// only some families use it.
    ///
    /// **Empty tags are a caller bug, not a panic.** Upstream never builds a
    /// parser with an empty tag (its call sites all check first), so Go's
    /// behaviour there is an accident of `strings.Split` rather than a design.
    /// We do the obvious thing instead: an empty opening tag matches
    /// immediately, an empty closing tag ends the thought immediately. Nobody
    /// should rely on either.
    pub fn new(opening_tag: impl Into<String>, closing_tag: impl Into<String>) -> Self {
        Self {
            state: ThinkingState::default(),
            opening_tag: opening_tag.into(),
            closing_tag: closing_tag.into(),
            acc: String::new(),
        }
    }

    /// The opening tag this parser was built with.
    pub fn opening_tag(&self) -> &str {
        &self.opening_tag
    }

    /// The closing tag this parser was built with.
    pub fn closing_tag(&self) -> &str {
        &self.closing_tag
    }

    /// Where the machine currently sits. Mostly for tests and debugging.
    pub fn state(&self) -> ThinkingState {
        self.state
    }

    /// Everything swallowed but not yet released -- the ambiguous tail.
    ///
    /// Not upstream API (`acc` is private there). Exposed read-only because when
    /// a stream ends mid-thought this is the only place the leftover bytes live,
    /// and a caller that wants to flush them rather than lose them needs to see
    /// them. **Don't** treat it as content: it may be a partial tag.
    pub fn buffered(&self) -> &str {
        &self.acc
    }

    /// Push the next chunk in; get back whatever became unambiguous.
    ///
    /// **Upstream:** `(*Parser).AddContent`.
    ///
    /// Loops internally because a single chunk can carry the machine through
    /// **several** states at once -- `"  <think>abc</think>\n\ndef"` walks all
    /// five in one call -- and a caller should never have to wait for the next
    /// chunk to receive text that is already unambiguous.
    pub fn add_content(&mut self, content: &str) -> Emitted {
        self.acc.push_str(content);

        let mut out = Emitted::default();
        let mut keep_looping = true;
        while keep_looping {
            let (thinking, remaining, again) = self.eat();
            out.thinking.push_str(&thinking);
            out.content.push_str(&remaining);
            keep_looping = again;
        }
        out
    }

    /// One step of the machine: `(thinking, content, keep_looping)`.
    ///
    /// **Upstream:** `eat(s *Parser)`. The third value is `true` iff the state
    /// changed in a way that might immediately release more, so the caller must
    /// go round again.
    fn eat(&mut self) -> (String, String, bool) {
        match self.state {
            ThinkingState::LookingForOpening => {
                let trimmed = self.acc.trim_start();
                if let Some(rest) = trimmed.strip_prefix(self.opening_tag.as_str()) {
                    // `rest` may hold far more than thinking -- the closing tag
                    // could already be in there -- so it goes back into the
                    // buffer and we loop, instead of being returned as thinking.
                    let after = rest.trim_start().to_string();
                    self.state = if after.is_empty() {
                        ThinkingState::ThinkingStartedEatingWhitespace
                    } else {
                        ThinkingState::Thinking
                    };
                    self.acc = after;
                    (String::new(), String::new(), true)
                } else if self.opening_tag.starts_with(trimmed) {
                    // Partial opening tag straddling chunks -- keep swallowing.
                    // Note `acc` is NOT reset here: the untrimmed original must
                    // survive, in case this turns out not to be a tag after all.
                    (String::new(), String::new(), false)
                } else if trimmed.is_empty() {
                    // Whitespace only, so keep swallowing.
                    //
                    // Upstream quirk, ported as-is: this arm is **unreachable**,
                    // because `strings.HasPrefix(openingTag, "")` is true and so
                    // the branch above already caught it. Kept because pinning
                    // the oracle's shape is worth more than tidying it, and if
                    // the branch above ever changes this one stops being dead.
                    (String::new(), String::new(), false)
                } else {
                    // Real content arrived with no opening tag in front of it,
                    // so this response got no thinking. Note we hand back the
                    // **untrimmed** buffer: there were no tags, so that leading
                    // whitespace is the user's text and not ours to eat.
                    self.state = ThinkingState::ThinkingDone;
                    (String::new(), std::mem::take(&mut self.acc), false)
                }
            }
            ThinkingState::ThinkingStartedEatingWhitespace => {
                let trimmed = self.acc.trim_start().to_string();
                self.acc.clear();
                if trimmed.is_empty() {
                    (String::new(), String::new(), false)
                } else {
                    self.state = ThinkingState::Thinking;
                    self.acc = trimmed;
                    (String::new(), String::new(), true)
                }
            }
            ThinkingState::Thinking => {
                if let Some((thought, rest)) = self.acc.split_once(self.closing_tag.as_str()) {
                    let thought = thought.to_string();
                    let remaining = rest.trim_start().to_string();
                    self.acc.clear();
                    self.state = if remaining.is_empty() {
                        // More whitespace might still be coming in the next
                        // chunk, so stay in the eating state.
                        ThinkingState::ThinkingDoneEatingWhitespace
                    } else {
                        ThinkingState::ThinkingDone
                    };
                    (thought, remaining, false)
                } else {
                    let tail = overlap(&self.acc, &self.closing_tag);
                    if tail > 0 {
                        // The last `tail` bytes could be the start of the
                        // closing tag. Release everything before them; hold the
                        // candidate back until the next chunk decides.
                        let split = self.acc.len() - tail;
                        let thought = self.acc[..split].to_string();
                        self.acc.drain(..split);
                        (thought, String::new(), false)
                    } else {
                        // Nothing in here can become a tag -- all thinking.
                        (std::mem::take(&mut self.acc), String::new(), false)
                    }
                }
            }
            ThinkingState::ThinkingDoneEatingWhitespace => {
                let trimmed = self.acc.trim_start().to_string();
                self.acc.clear();
                if !trimmed.is_empty() {
                    self.state = ThinkingState::ThinkingDone;
                }
                (String::new(), trimmed, false)
            }
            ThinkingState::ThinkingDone => (String::new(), std::mem::take(&mut self.acc), false),
        }
    }
}

/// Longest suffix of `s` that is also a prefix of `delim`, in **bytes**.
///
/// **Upstream:** `overlap(s, delim string) int` in `thinking/parser.go`.
///
/// This is the whole trick behind streaming. `overlap("abc</th", "</think>")` is
/// `4`, so `abc` is safe to emit now and `</th` must wait. `overlap("abc", ...)`
/// is `0`, so everything goes out immediately.
///
/// Returns `0` when nothing overlaps. Naive scan from longest to shortest, same
/// as upstream -- tags are a handful of bytes, so no Knuth-Morris-Pratt needed
/// lah, and the naive version is obviously correct.
///
/// ## Why byte comparison is safe on UTF-8
///
/// Go slices bytes; Rust panics if you slice a `&str` mid-character, so this
/// works on `as_bytes()` and returns a byte count. That count is **always a
/// valid char boundary in `s`**, and here is the proof -- it matters because the
/// caller slices `s` at `s.len() - overlap`, so a wrong answer here is a panic
/// there:
///
/// * `delim[..i]` starts at byte 0 of `delim`, which is the start of a
///   character, so its first byte is never a UTF-8 continuation byte.
/// * If `s` ends with those bytes, the byte at `s.len() - i` is that same
///   non-continuation byte, and in valid UTF-8 that position is a char boundary.
/// * A `delim[..i]` that ends mid-character can never match, because `s` is
///   valid UTF-8 and so cannot end with a truncated character.
///
/// The boundary check below is therefore belt-and-braces: it can never fire, it
/// costs nothing, and it means this function cannot hand the caller an index
/// that panics even if the reasoning above is one day proven wrong.
fn overlap(s: &str, delim: &str) -> usize {
    let (sb, db) = (s.as_bytes(), delim.as_bytes());
    let max = db.len().min(sb.len());
    (1..=max)
        .rev()
        .find(|&i| sb.ends_with(&db[..i]) && s.is_char_boundary(sb.len() - i))
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Sniffing the tags out of a chat template
// ---------------------------------------------------------------------------

/// The tag pair a model uses to wrap its thinking.
///
/// Both halves are guaranteed **non-empty** -- see [`infer_tags`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinkingTags {
    /// e.g. `<think>`.
    pub opening: String,
    /// e.g. `</think>`.
    pub closing: String,
}

/// Work out a model's thinking tags by reading its own chat template.
///
/// **Upstream:** `InferTags(t *template.Template)` in `thinking/template.go`.
///
/// ## Why sniff instead of configure
///
/// The tags are not written down anywhere in a GGUF. But the chat template
/// **must** contain them, because the template is what re-serialises a past
/// assistant turn back into the next prompt -- if the model wrapped its thought
/// in `<think>...</think>` when generating, the template has to wrap it the same
/// way when replaying. So the template is the honest source, and reading it
/// means a brand-new model family works with zero config. The same call also
/// decides whether a model advertises
/// [`Capability::Thinking`](crate::api::Capability) at all (upstream
/// `server/images.go`).
///
/// ## The heuristic, exactly
///
/// 1. Walk the parsed template.
/// 2. Find a reference to the field `.Thinking`.
/// 3. Check the **nearest enclosing `range`** iterates over `.Messages`. No
///    enclosing range, or a range over something else, and the hit is ignored --
///    that is what rejects the `{{ if .Thinking }}/think{{ end }}` mode switch at
///    the top of most templates, which is an instruction *to* the model, not a
///    tag.
/// 4. Take the **nearest enclosing node list**, and its **first** and **last**
///    nodes. If they are literal text, trim them and that is the pair.
///
/// So from `<think>{{ .Thinking }}</think>` you get `("<think>", "</think>")`.
/// The "nearest list starts and ends with text" shape is what makes the
/// heuristic mostly self-checking: a hit in the wrong place usually lands in a
/// list that does not start and end with text, and yields nothing.
///
/// **Later hits overwrite earlier ones**, and each half overwrites independently
/// (upstream quirk, pinned in the tests): if the last node of a matching list is
/// not a text node, the previously-found closing tag survives. In practice that
/// is load-bearing -- the `.Thinking` inside `{{ if and $last .Thinking }}` is
/// visited *before* the one inside `<think>{{ .Thinking }}</think>`, and the
/// second is the one you actually want.
///
/// ## Where our AST forces an approximation -- read this before trusting it
///
/// Upstream walks Go's own `text/template/parse` AST with a stack of ancestors.
/// KOPITIAM's engine ([`crate::gotmpl`]) has a **different**, smaller AST, so the
/// *heuristic* is ported, not the node types. Three honest differences:
///
/// * **There is no `ListNode`.** Upstream's "nearest enclosing `ListNode`" is
///   here "the `Vec<Node>` we are currently iterating" -- an `if` body, a `range`
///   body, an `else` body, or the template root. Structurally the same thing; Go
///   just gives it a name. No behavioural difference found against upstream's
///   own test table, including the real qwen3 template.
/// * **No `{{ template }}` / `{{ block }}` / `{{ define }}`.** Our parser rejects
///   them outright, so upstream's `TemplateNode` case has no counterpart. A chat
///   template using sub-templates would fail to parse long before it got here --
///   so this is a loud error upstream of us, not a silent wrong answer.
/// * **Empty node lists are guarded, not indexed.** Upstream does `l.Nodes[0]`
///   with no length check. It cannot actually panic there (any list reached this
///   way holds at least the node containing the field), but we guard anyway,
///   because "cannot happen" is a poor reason to leave an index unchecked.
///
/// ## Return type
///
/// **Deliberate divergence:** upstream returns `(string, string)` and *every*
/// call site immediately checks `openingTag != "" && closingTag != ""`. We fold
/// that check in and return an [`Option`], so a half-inferred pair cannot leak
/// into a [`Parser`] and quietly never match anything. Same decision, made once,
/// in the one place it cannot be forgotten.
pub fn infer_tags(t: &Template) -> Option<ThinkingTags> {
    let (opening, closing) = infer_tags_raw(t);
    if opening.is_empty() || closing.is_empty() {
        return None;
    }
    Some(ThinkingTags { opening, closing })
}

/// [`infer_tags`] without the both-non-empty fold -- exactly upstream's
/// `(string, string)`. Private on purpose, so the tests can pin the oracle's own
/// table including the cases where it returns one empty half.
fn infer_tags_raw(t: &Template) -> (String, String) {
    let mut hunt = TagHunt::default();
    hunt.walk_list(t.nodes(), None);
    (hunt.opening, hunt.closing)
}

/// Walk state for [`infer_tags_raw`].
///
/// **Upstream:** the closures over `ancestors` / `openingTag` / `closingTag`
/// inside `InferTags`. Upstream keeps an explicit ancestor stack and searches it
/// backwards; we thread down the recursion the only two answers that stack was
/// ever asked for (nearest enclosing list, nearest enclosing `range`). Same
/// result, no stack to keep in sync.
#[derive(Default)]
struct TagHunt {
    opening: String,
    closing: String,
}

impl TagHunt {
    /// Visit every node of one list. **That list is the "nearest enclosing
    /// list" for everything directly inside it.**
    ///
    /// `range_over_messages` is `None` when there is no enclosing `range` at
    /// all, and `Some(bool)` for whether the nearest one iterates `.Messages`.
    fn walk_list(&mut self, list: &[Node], range_over_messages: Option<bool>) {
        for n in list {
            self.walk_node(n, list, range_over_messages);
        }
    }

    /// **Upstream:** `templateVisit`, minus the node kinds our AST doesn't have.
    fn walk_node(&mut self, n: &Node, list: &[Node], range_over_messages: Option<bool>) {
        match n {
            // Leaves.
            Node::Text(_) | Node::Continue | Node::Break => {}
            Node::Action(pipe) | Node::Assign { pipe, .. } => {
                self.walk_pipe(pipe, list, range_over_messages);
            }
            // `if` and `with` are both upstream's BranchNode: pipe, then body,
            // then else-body. Order matters -- the pipe is visited FIRST, which
            // is what lets a later hit inside the body overwrite it.
            Node::If {
                pipe,
                then: body,
                otherwise,
            }
            | Node::With {
                pipe,
                body,
                otherwise,
            } => {
                self.walk_pipe(pipe, list, range_over_messages);
                self.walk_list(body, range_over_messages);
                self.walk_list(otherwise, range_over_messages);
            }
            Node::Range {
                pipe,
                body,
                otherwise,
                ..
            } => {
                // This range becomes the nearest one for its OWN pipe too --
                // upstream pushes the RangeNode before descending into its
                // pipeline, so a `.Thinking` sitting inside `{{ range .Thinking }}`
                // is judged against that very range. Faithfully copied, odd as
                // it looks.
                let over_messages = Some(pipe_uses_field(pipe, "Messages"));
                self.walk_pipe(pipe, list, over_messages);
                self.walk_list(body, over_messages);
                self.walk_list(otherwise, over_messages);
            }
        }
    }

    /// **Upstream:** the `PipeNode` / `CommandNode` cases of `templateVisit`,
    /// plus the `FieldNode` arm of `enterFn`.
    fn walk_pipe(&mut self, pipe: &Pipeline, list: &[Node], range_over_messages: Option<bool>) {
        for cmd in &pipe.cmds {
            for arg in &cmd.args {
                match arg {
                    Arg::Field(idents) if idents.first().is_some_and(|i| i == "Thinking") => {
                        self.record(list, range_over_messages);
                    }
                    Arg::Paren(sub) => self.walk_pipe(sub, list, range_over_messages),
                    _ => {}
                }
            }
        }
    }

    /// Found a `.Thinking`; take the tags off the enclosing list if it qualifies.
    fn record(&mut self, list: &[Node], range_over_messages: Option<bool>) {
        // No enclosing range, or the nearest one is not over `.Messages`:
        // upstream bails out of this hit and keeps walking.
        if range_over_messages != Some(true) {
            return;
        }
        // Each half is assigned only when that end of the list is literal text.
        // A non-text end leaves the PREVIOUS value standing -- upstream quirk,
        // deliberately not tidied; see [`infer_tags`].
        if let Some(Node::Text(t)) = list.first() {
            self.opening = t.trim().to_string();
        }
        if let Some(Node::Text(t)) = list.last() {
            self.closing = t.trim().to_string();
        }
    }
}

/// Does this pipeline mention the field `.Name`, where `Name` is `field`?
///
/// **Upstream:** `rangeUsesField(rangeNode, field)`, which walks only the
/// range's own pipeline.
///
/// Only `Arg::Field` counts -- `$.Messages` is a *variable* with a field chain,
/// not a field, and upstream's `FieldNode`-only check ignores it too. That
/// matters, hor: qwen3's range body says `slice $.Messages $i`, and if that
/// counted then the doubly-nested-range case would infer tags it must reject.
fn pipe_uses_field(pipe: &Pipeline, field: &str) -> bool {
    pipe.cmds.iter().any(|cmd| {
        cmd.args.iter().any(|arg| match arg {
            Arg::Field(idents) => idents.first().is_some_and(|i| i == field),
            Arg::Paren(sub) => pipe_uses_field(sub, field),
            _ => false,
        })
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `(chunk, want_thinking, want_content, want_state_after)`.
    type Step<'a> = (&'a str, &'a str, &'a str, ThinkingState);

    /// Drive a fresh `<think>`/`</think>` parser through a chunk script and
    /// check what came out **and** where the machine ended up after each step.
    ///
    /// Checking the state after every step is what upstream's table does, and it
    /// is worth keeping: two different states can emit the same text for one
    /// chunk and then diverge on the next.
    fn drive(steps: &[Step<'_>]) {
        let mut p = Parser::new("<think>", "</think>");
        for (i, (chunk, want_thinking, want_content, want_state)) in steps.iter().enumerate() {
            let got = p.add_content(chunk);
            assert_eq!(
                (got.thinking.as_str(), got.content.as_str()),
                (*want_thinking, *want_content),
                "step {i} fed {chunk:?}",
            );
            assert_eq!(p.state(), *want_state, "state after step {i} fed {chunk:?}");
        }
    }

    // ---- overlap: the streaming primitive ----------------------------------

    #[test]
    fn overlap_finds_the_longest_suffix_that_could_still_become_the_tag() {
        assert_eq!(overlap("abc</th", "</think>"), 4);
        assert_eq!(overlap("ghi</thi", "</think>"), 5);
        assert_eq!(overlap("abc", "</think>"), 0);
        assert_eq!(overlap("</thing>def", "</think>"), 0);
    }

    #[test]
    fn overlap_is_capped_at_the_shorter_of_the_two_strings() {
        assert_eq!(overlap("</think>", "</think>"), 8);
        assert_eq!(overlap("<", "</think>"), 1);
        assert_eq!(overlap("", "</think>"), 0);
    }

    #[test]
    fn a_multibyte_thought_survives_a_buffered_partial_tag() {
        // Regression guard for the byte-index slicing in `eat`: Rust panics on a
        // mid-character slice, so if `overlap` ever returned a non-boundary this
        // test blows up instead of silently corrupting text.
        let mut p = Parser::new("<think>", "</think>");
        let out = p.add_content("<think>\u{601d}\u{8003}\u{4e2d}</th");
        assert_eq!(out.thinking, "\u{601d}\u{8003}\u{4e2d}");
        assert_eq!(p.buffered(), "</th", "only the candidate tag is held back");
        let out = p.add_content("ink>\u{597d}");
        assert_eq!(out.content, "\u{597d}");
    }

    // ---- TestExtractThinking (upstream table) ------------------------------

    #[test]
    fn a_whole_response_in_one_chunk_splits_thought_from_answer() {
        // Upstream `TestExtractThinking`, case 0.
        let mut p = Parser::new("<think>", "</think>");
        let got = p.add_content("<think> internal </think> world");
        assert_eq!(got.thinking, "internal ");
        assert_eq!(got.content, "world");
    }

    #[test]
    fn only_the_first_thinking_block_counts_as_thinking() {
        // Upstream `TestExtractThinking`, case 1. Once the closing tag is seen
        // the parser is done -- a second `<think>` is just text the model wrote.
        let mut p = Parser::new("<think>", "</think>");
        let got = p.add_content("<think>a</think><think>b</think>c");
        assert_eq!(got.thinking, "a");
        assert_eq!(got.content, "<think>b</think>c");
    }

    #[test]
    fn a_response_with_no_tags_at_all_is_pure_content() {
        // Upstream `TestExtractThinking`, case 2.
        let mut p = Parser::new("<think>", "</think>");
        let got = p.add_content("no think");
        assert_eq!(got.thinking, "");
        assert_eq!(got.content, "no think");
    }

    // ---- TestThinkingStreaming (upstream table, one test per case) ---------

    #[test]
    fn content_without_a_thinking_tag_is_never_emitted_twice() {
        // Upstream regression case: an earlier bug moved to ThinkingDone without
        // clearing the buffer, so the first chunk came out again on the second.
        drive(&[
            ("  abc", "", "  abc", ThinkingState::ThinkingDone),
            ("def", "", "def", ThinkingState::ThinkingDone),
        ]);
    }

    #[test]
    fn content_before_a_thinking_tag_nerfs_the_thinking_tag() {
        drive(&[(
            "  abc <think>def</think> ghi",
            "",
            "  abc <think>def</think> ghi",
            ThinkingState::ThinkingDone,
        )]);
    }

    #[test]
    fn an_opening_tag_built_up_across_chunks_still_opens() {
        drive(&[
            ("  <th", "", "", ThinkingState::LookingForOpening),
            ("in", "", "", ThinkingState::LookingForOpening),
            ("k>a", "a", "", ThinkingState::Thinking),
        ]);
    }

    #[test]
    fn a_closing_tag_split_across_chunks_is_buffered_until_it_resolves() {
        drive(&[
            ("<think>abc</th", "abc", "", ThinkingState::Thinking),
            ("ink>def", "", "def", ThinkingState::ThinkingDone),
        ]);
    }

    #[test]
    fn a_partial_closing_tag_that_turns_out_fake_is_released_as_thinking() {
        // The case this whole module exists for. `</th` is held back, then the
        // next chunk proves it was never a tag -- so it must come back out as
        // thinking, not vanish, and the machine must stay in Thinking.
        drive(&[
            ("<think>abc</th", "abc", "", ThinkingState::Thinking),
            ("ing>def", "</thing>def", "", ThinkingState::Thinking),
            ("ghi</thi", "ghi", "", ThinkingState::Thinking),
            ("nk>jkl", "", "jkl", ThinkingState::ThinkingDone),
        ]);
    }

    #[test]
    fn whitespace_between_the_closing_tag_and_the_answer_is_eaten() {
        drive(&[(
            "  <think>abc</think>\n\ndef",
            "abc",
            "def",
            ThinkingState::ThinkingDone,
        )]);
    }

    #[test]
    fn whitespace_after_the_closing_tag_is_eaten_across_a_chunk_boundary() {
        drive(&[
            (
                "  <think>abc</think>",
                "abc",
                "",
                ThinkingState::ThinkingDoneEatingWhitespace,
            ),
            ("\n\ndef", "", "def", ThinkingState::ThinkingDone),
        ]);
    }

    #[test]
    fn whitespace_inside_the_answer_is_left_alone_once_thinking_is_done() {
        // Only the LEADING whitespace of the answer gets eaten. Trailing spaces
        // and later chunks pass through untouched -- eating those would corrupt
        // the model's actual output.
        drive(&[
            (
                "  <think>abc</think>\n\ndef ",
                "abc",
                "def ",
                ThinkingState::ThinkingDone,
            ),
            (" ghi", "", " ghi", ThinkingState::ThinkingDone),
        ]);
    }

    #[test]
    fn a_token_by_token_stream_walks_every_state_in_order() {
        drive(&[
            (
                "<think>",
                "",
                "",
                ThinkingState::ThinkingStartedEatingWhitespace,
            ),
            ("\n", "", "", ThinkingState::ThinkingStartedEatingWhitespace),
            (
                "</think>",
                "",
                "",
                ThinkingState::ThinkingDoneEatingWhitespace,
            ),
            ("\n\n", "", "", ThinkingState::ThinkingDoneEatingWhitespace),
            ("Hi", "", "Hi", ThinkingState::ThinkingDone),
            (" there", "", " there", ThinkingState::ThinkingDone),
        ]);
    }

    #[test]
    fn whitespace_between_the_opening_tag_and_the_thought_is_eaten() {
        drive(&[
            (
                "  <think>   \t ",
                "",
                "",
                ThinkingState::ThinkingStartedEatingWhitespace,
            ),
            (
                "  these are some ",
                "these are some ",
                "",
                ThinkingState::Thinking,
            ),
            (
                "thoughts </think>  ",
                "thoughts ",
                "",
                ThinkingState::ThinkingDoneEatingWhitespace,
            ),
            (
                "  more content",
                "",
                "more content",
                ThinkingState::ThinkingDone,
            ),
        ]);
    }

    // ---- Parser odds and ends ----------------------------------------------

    #[test]
    fn priming_with_the_opening_tag_matches_what_routes_go_does() {
        // Some templates end the prompt with `<think>`, so the model never emits
        // it. Upstream feeds the tag in by hand before streaming starts.
        let mut p = Parser::new("<think>", "</think>");
        let primed = p.add_content("<think>");
        assert_eq!(primed, Emitted::default());
        assert_eq!(p.state(), ThinkingState::ThinkingStartedEatingWhitespace);
        let out = p.add_content("mm</think>yes");
        assert_eq!(out.thinking, "mm");
        assert_eq!(out.content, "yes");
    }

    #[test]
    fn a_non_default_tag_pair_streams_the_same_way() {
        // Nothing is hardcoded to `<think>`; a longer harmony-style tag pair
        // streams identically, including the mid-tag chunk split.
        let mut p = Parser::new("<|channel|>analysis<|message|>", "<|end|>");
        let a = p.add_content("<|channel|>analysis<|message|>weighing it up<|en");
        assert_eq!(a.thinking, "weighing it up");
        assert_eq!(p.buffered(), "<|en");
        let b = p.add_content("d|>Answer.");
        assert_eq!(b.content, "Answer.");
    }

    #[test]
    fn every_byte_fed_in_comes_back_out_exactly_once() {
        // The module-level invariant, checked directly: feed the same response
        // one character at a time and the concatenated halves must rebuild it,
        // minus the two tags and the whitespace the machine is meant to eat.
        let src = "<think>weighing it up</think>\n\nThe answer is 42.";
        let mut p = Parser::new("<think>", "</think>");
        let (mut thinking, mut content) = (String::new(), String::new());
        for ch in src.chars() {
            let out = p.add_content(&ch.to_string());
            thinking.push_str(&out.thinking);
            content.push_str(&out.content);
        }
        assert_eq!(thinking, "weighing it up");
        assert_eq!(content, "The answer is 42.");
        assert_eq!(p.buffered(), "", "nothing left stuck in the buffer");
    }

    #[test]
    fn the_state_names_read_the_same_as_the_go_ones() {
        assert_eq!(
            ThinkingState::LookingForOpening.to_string(),
            "LookingForOpening"
        );
        assert_eq!(
            ThinkingState::ThinkingStartedEatingWhitespace.to_string(),
            "ThinkingStartedEatingWhitespace"
        );
        assert_eq!(ThinkingState::Thinking.to_string(), "Thinking");
        assert_eq!(
            ThinkingState::ThinkingDoneEatingWhitespace.to_string(),
            "ThinkingDoneEatingWhitespace"
        );
        assert_eq!(ThinkingState::ThinkingDone.to_string(), "ThinkingDone");
        assert_eq!(ThinkingState::default(), ThinkingState::LookingForOpening);
    }

    // ---- TestInferThinkingTags (upstream table) ----------------------------

    fn tags_of(src: &str) -> (String, String) {
        let t = Template::parse(src).expect("template must parse");
        infer_tags_raw(&t)
    }

    #[test]
    fn tags_are_inferred_from_a_thinking_field_inside_a_messages_range() {
        // Upstream `TestInferThinkingTags`, case "basic". Note the top-level
        // `{{ if .Thinking }}/think{{ end }}` is a MODE SWITCH sent to the model,
        // not a tag -- it sits outside any range, so it must be ignored.
        let (opening, closing) = tags_of(
            r#"
			{{ if .Thinking}}
				/think
			{{ end }}
			{{- range $i, $_ := .Messages }}
				{{- $last := eq (len (slice $.Messages $i)) 1 -}}
				{{ if and $last .Thinking }}
					<think>{{ .Thinking }}</think>
				{{ end }}
			{{ end }}
		"#,
        );
        assert_eq!(opening, "<think>");
        assert_eq!(closing, "</think>");
    }

    #[test]
    fn a_thinking_field_under_a_range_over_something_else_infers_nothing() {
        // Upstream case "doubly nested range". The NEAREST range is over
        // `.NotMessages`, so the hit is rejected -- even though a `.Messages`
        // range does enclose it further out.
        let (opening, closing) = tags_of(
            r#"
			{{ if .Thinking}}
				/think
			{{ end }}
			{{- range $i, $_ := .Messages }}
				{{- range $j, $_ := .NotMessages }}
					{{- $last := eq (len (slice $.Messages $i)) 1 -}}
					{{ if and $last .Thinking }}
						<think>{{ .Thinking }}</think>
					{{ end }}
				{{ end }}
			{{ end }}
		"#,
        );
        assert_eq!(opening, "");
        assert_eq!(closing, "");
    }

    #[test]
    fn inferred_tags_have_their_surrounding_whitespace_trimmed() {
        // Upstream case "whitespace is trimmed". The tags need not look like
        // tags at all -- whatever literal text brackets `.Thinking` is what the
        // model emits, so it is what we must strip back out.
        let (opening, closing) = tags_of(
            r#"
			{{ if .Thinking}}
				/think
			{{ end }}
			{{- range $i, $_ := .Messages }}
				{{- $last := eq (len (slice $.Messages $i)) 1 -}}
				{{ if and $last .Thinking }}
					Some text before   {{ .Thinking }}    Some text after
				{{ end }}
			{{ end }}
		"#,
        );
        assert_eq!(opening, "Some text before");
        assert_eq!(closing, "Some text after");
    }

    #[test]
    fn the_real_qwen3_chat_template_yields_the_think_tags() {
        // Upstream case "qwen3" -- the actual shipped template, warts and all.
        // This is the one that proves the heuristic survives a real template
        // with tools, tool calls, four roles and trim markers everywhere.
        let (opening, closing) = tags_of(
            r#"
{{- if or .System .Tools .Thinking }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}
{{- if .Thinking }}
/think
{{- else }}
/no_think
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if and $last .Thinking }}
<think>{{ .Thinking }}</think>
{{ end }}
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
			"#,
        );
        assert_eq!(opening, "<think>");
        assert_eq!(closing, "</think>");
    }

    #[test]
    fn a_template_with_no_thinking_field_infers_no_tags() {
        // The common non-reasoning case: nothing to sniff, so no thinking
        // capability. Upstream's `server/images.go` hangs that capability flag
        // off exactly this call.
        let t = Template::parse(
            "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}",
        )
        .expect("template must parse");
        assert_eq!(infer_tags_raw(&t), (String::new(), String::new()));
        assert_eq!(infer_tags(&t), None);
    }

    #[test]
    fn the_folded_api_hands_back_both_tags_together_or_nothing() {
        let t = Template::parse(
            "{{- range $i, $_ := .Messages }}\n<think>{{ .Thinking }}</think>\n{{ end }}",
        )
        .expect("template must parse");
        assert_eq!(
            infer_tags(&t),
            Some(ThinkingTags {
                opening: "<think>".into(),
                closing: "</think>".into(),
            })
        );

        // A half-inferred pair must NOT reach a Parser: an empty closing tag
        // would end every thought at byte zero.
        let half = Template::parse(
            "{{- range $i, $_ := .Messages }}\n<think>{{ .Thinking }}{{ if $i }}x{{ end }}{{ end }}",
        )
        .expect("template must parse");
        let (opening, closing) = infer_tags_raw(&half);
        assert_eq!(opening, "<think>");
        assert_eq!(closing, "", "the list's last node is an if, not text");
        assert_eq!(infer_tags(&half), None);
    }
}
