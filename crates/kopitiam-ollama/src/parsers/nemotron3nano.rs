//! Nemotron 3 Nano response parser.
//!
//! **Upstream:** `model/parsers/nemotron3nano.go` (ollama, MIT, Copyright (c)
//! Ollama). Ported against `4713800b08b2ddf5e14acf8398953cf7b12f169b`
//! (2026-07-28).
//!
//! ## Division of labour
//!
//! This parser owns **thinking only**. Once thinking is done, everything else --
//! content and the XML-ish `<tool_call>` bodies -- is handed to a
//! [`Qwen3CoderParser`], because Nemotron emits qwen3-coder's tool-call format.
//! Upstream holds a `*Qwen3CoderParser` for exactly this; so do we. Upstream's
//! own test file says it plainly: *"Tool call parsing is tested in
//! qwen3coder_test.go since Nemotron delegates to Qwen3CoderParser."*
//!
//! ## The prompt opens `<think>`, so the model usually does not
//!
//! Like cohere and olmo3-think, the generation prompt injects the opening tag,
//! so the stream starts **inside** the thinking block. But some checkpoints emit
//! a redundant `<think>` anyway, so [`Nemotron3NanoParser`] carries a one-shot
//! latch (`maybe_thinking_open_at_bol`) that strips at most one leading
//! `<think>` -- and only at the very beginning, never mid-stream.
//!
//! ## Thinking can end WITHOUT `</think>`
//!
//! A `<tool_call>` seen while still thinking ends thinking too. Upstream picks
//! whichever of `</think>` and `<tool_call>` comes **first** in the buffer, and
//! the two differ in one important way:
//!
//! * `</think>` is **consumed** -- the remainder after it is what carries on;
//! * `<tool_call>` is **kept** -- the remainder *includes* the tag, because the
//!   [`Qwen3CoderParser`] downstream needs to see it to open the call.
//!
//! **What would make this wrong:** consuming `<tool_call>` like a closing tag.
//! The coder parser would then never see the opener, the call would never be
//! recognised, and its raw XML would be handed to the user as content.

use crate::api::{Message, ThinkValue, Tool};

use super::qwen3coder::{Qwen3CoderParser, TOOL_CLOSE_TAG, TOOL_OPEN_TAG};
use super::{Parsed, Parser, ParserError, overlap, trailing_whitespace_len};

/// **Upstream:** the `nemotron*` tag consts.
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
/// Same string as [`TOOL_OPEN_TAG`]; upstream declares its own const for it, and
/// we alias rather than redeclare so the two can never drift apart.
const TOOL_CALL_OPEN: &str = TOOL_OPEN_TAG;

/// **Upstream:** `Nemotron3NanoParserState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    #[default]
    CollectingThinking,
    /// Thinking closed but nothing followed it yet. Eat the whitespace that
    /// arrives before the first real content byte -- it is framing left over
    /// from the close tag, and it may span several chunks.
    SkipWhitespaceAfterThinking,
    CollectingContent,
}

/// **Upstream:** `Nemotron3NanoParser`.
#[derive(Debug, Default)]
pub struct Nemotron3NanoParser {
    state: State,
    buffer: String,
    tool_parser: Qwen3CoderParser,
    /// One-shot latch: a leading `<think>` may be stripped, but only while we
    /// are still at the beginning of the stream.
    maybe_thinking_open_at_bol: bool,
    /// Armed right after a `<think>` is stripped: keep eating leading
    /// whitespace until real thinking content shows up.
    skip_thinking_leading_ws: bool,
}

impl Nemotron3NanoParser {
    /// Return the unambiguous part of the thinking buffer, holding back
    /// whatever could still become a tag.
    ///
    /// **Upstream:** `emitThinking`.
    ///
    /// Two branches, and they treat trailing whitespace **differently** -- which
    /// looks like an oversight but is upstream's, so it is kept:
    ///
    /// * **partial tag present** -- the whitespace in front of it is trimmed off
    ///   the emitted text and **dropped entirely** (the buffer keeps only the
    ///   partial tag). Right when the tag completes, since `</think>` trims that
    ///   whitespace anyway. **Wrong when it does not** -- `"a </thing"` loses the
    ///   space. Upstream has the same hole; diverging would make this port and
    ///   ollama produce different text for the same bytes.
    /// * **no partial tag** -- the trailing whitespace is **held in the buffer**,
    ///   not dropped, because `</think>` may be the very next thing to arrive.
    fn emit_thinking(&mut self) -> String {
        let buf = std::mem::take(&mut self.buffer);

        let max_overlap = overlap(&buf, THINK_CLOSE).max(overlap(&buf, TOOL_CALL_OPEN));

        if max_overlap > 0 {
            let (unambiguous, ambiguous) = super::chop(&buf, buf.len() - max_overlap);
            let out = unambiguous.trim_end().to_string();
            self.buffer.push_str(ambiguous);
            return out;
        }

        let ws_len = trailing_whitespace_len(&buf);
        if ws_len > 0 {
            let (unambiguous, ws) = super::chop(&buf, buf.len() - ws_len);
            let out = unambiguous.to_string();
            self.buffer.push_str(ws);
            return out;
        }

        buf
    }

    /// Strip one leading `<think>` if it is there. **Upstream:**
    /// `stripOpeningThinkTag`.
    ///
    /// Returns `true` only when a tag was actually consumed, which tells the
    /// caller to restart the pass. The three outcomes:
    ///
    /// * tag found -> consume it, disarm the latch, arm the whitespace skip,
    ///   return `true`;
    /// * the buffer so far is **entirely** a prefix of `<think>` (checked as
    ///   `overlap(trimmed, "<think>") == trimmed.len()`) -> still ambiguous,
    ///   keep the latch armed and wait;
    /// * anything else -> there was never a leading tag; disarm the latch for
    ///   good so a `<think>` appearing later in the stream is treated as
    ///   literal text.
    fn strip_opening_think_tag(&mut self) -> bool {
        if !self.maybe_thinking_open_at_bol {
            return false;
        }

        let buf = self.buffer.clone();
        let trimmed = buf.trim_start().to_string();
        if trimmed.is_empty() {
            self.buffer.clear();
            return false;
        }

        if let Some(after) = trimmed.strip_prefix(THINK_OPEN) {
            self.buffer = after.trim_start().to_string();
            self.maybe_thinking_open_at_bol = false;
            self.skip_thinking_leading_ws = true;
            return true;
        }

        if overlap(&trimmed, THINK_OPEN) == trimmed.len() {
            // Still could grow into `<think>`. Keep waiting, but do drop the
            // leading whitespace we already trimmed.
            if trimmed.len() != buf.len() {
                self.buffer = trimmed;
            }
            return false;
        }

        self.maybe_thinking_open_at_bol = false;
        false
    }
}

impl Parser for Nemotron3NanoParser {
    /// **Upstream:** `(*Nemotron3NanoParser).Init`.
    ///
    /// `None` for `think` means **on** for this family. An assistant prefill
    /// *with content* skips thinking entirely, same rule as everywhere else.
    fn init(
        &mut self,
        tools: Vec<Tool>,
        last_message: Option<&Message>,
        think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        self.tool_parser = Qwen3CoderParser::default();
        self.tool_parser.init(tools.clone(), None, None);
        self.buffer.clear();
        self.maybe_thinking_open_at_bol = false;
        self.skip_thinking_leading_ws = false;

        let thinking_enabled = think.is_none_or(|t| t.enabled());
        let prefill = last_message.is_some_and(|m| m.role == "assistant");
        let prefilled_content = last_message.is_some_and(|m| !m.content.is_empty());

        if !thinking_enabled || (prefill && prefilled_content) {
            self.state = State::CollectingContent;
        } else {
            self.state = State::CollectingThinking;
            self.maybe_thinking_open_at_bol = true;
        }

        tools
    }

    /// **Upstream:** `(*Nemotron3NanoParser).Add`.
    fn add(&mut self, s: &str, done: bool) -> Result<Parsed, ParserError> {
        match self.state {
            // Past thinking: the coder parser owns everything, including its own
            // ambiguity buffering for the tool tags.
            State::CollectingContent => return self.tool_parser.add(s, done),

            State::SkipWhitespaceAfterThinking => {
                let s = s.trim_start();
                if s.is_empty() {
                    // Still only whitespace -- stay here so the NEXT chunk's
                    // leading whitespace is eaten too.
                    return Ok(Parsed::default());
                }
                self.state = State::CollectingContent;
                return self.tool_parser.add(s, done);
            }

            State::CollectingThinking => {}
        }

        self.buffer.push_str(s);

        if self.skip_thinking_leading_ws {
            let trimmed = self.buffer.trim_start().to_string();
            self.buffer = trimmed;
            if self.buffer.is_empty() {
                return Ok(Parsed::default());
            }
            self.skip_thinking_leading_ws = false;
        }

        if self.strip_opening_think_tag() {
            // The tag is gone; re-run the pass over what is left. Bounded at one
            // extra level, because the latch is now disarmed.
            return self.add("", done);
        }

        if self.maybe_thinking_open_at_bol {
            let buf = self.buffer.clone();
            let trimmed = buf.trim_start();
            if trimmed.is_empty() || overlap(trimmed, THINK_OPEN) == trimmed.len() {
                // Everything so far could still be a leading `<think>`. Emitting
                // it now would leak `<thi` into thinking.
                if trimmed.len() != buf.len() {
                    self.buffer = trimmed.to_string();
                }
                return Ok(Parsed::default());
            }
        }

        // Thinking ends at whichever comes FIRST: `</think>` or `<tool_call>`.
        let think_idx = self.buffer.find(THINK_CLOSE);
        let tool_idx = self.buffer.find(TOOL_CALL_OPEN);

        let end = match (think_idx, tool_idx) {
            // `</think>` is CONSUMED -- carry on after it.
            (Some(t), None) => Some((t, self.buffer[t + THINK_CLOSE.len()..].trim_start().to_string())),
            (Some(t), Some(k)) if t < k => {
                Some((t, self.buffer[t + THINK_CLOSE.len()..].trim_start().to_string()))
            }
            // `<tool_call>` is KEPT -- the remainder INCLUDES the tag, because
            // the coder parser downstream needs it to open the call.
            (_, Some(k)) => Some((k, self.buffer[k..].to_string())),
            (None, None) => None,
        };

        if let Some((end_idx, remainder)) = end {
            let thinking = self.buffer[..end_idx].trim_end().to_string();
            self.buffer.clear();

            if remainder.is_empty() {
                self.state = State::SkipWhitespaceAfterThinking;
                return Ok(Parsed {
                    thinking,
                    ..Default::default()
                });
            }

            self.state = State::CollectingContent;
            let mut out = self.tool_parser.add(&remainder, done)?;
            out.thinking = thinking;
            return Ok(out);
        }

        let thinking = self.emit_thinking();
        Ok(Parsed {
            thinking,
            ..Default::default()
        })
    }

    fn preserved_tokens(&self) -> Vec<&'static str> {
        vec![THINK_OPEN, THINK_CLOSE, TOOL_OPEN_TAG, TOOL_CLOSE_TAG]
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

    fn nemotron(think: bool) -> Nemotron3NanoParser {
        let mut p = Nemotron3NanoParser::default();
        p.init(Vec::new(), None, Some(&ThinkValue::Bool(think)));
        p
    }

    /// Upstream's harness: one `Add(input, false)` then a drain `Add("", true)`.
    fn run(p: &mut Nemotron3NanoParser, input: &str) -> Parsed {
        let mut got = p.add(input, false).expect("add");
        let drained = p.add("", true).expect("drain");
        got.content.push_str(&drained.content);
        got.thinking.push_str(&drained.thinking);
        got.calls.extend(drained.calls);
        got
    }

    /// Upstream `TestNemotron3NanoParser`, ported verbatim as ground truth.
    #[test]
    fn nemotron_splits_thinking_from_content() {
        // (name, input, want_content, want_thinking, think)
        let cases: &[(&str, &str, &str, &str, bool)] = &[
            (
                "thinking then content",
                "Let me think about this...</think>\nHere is my answer.",
                "Here is my answer.",
                "Let me think about this...",
                true,
            ),
            (
                "thinking with newlines",
                "Step 1: Analyze\nStep 2: Process\nStep 3: Conclude</think>\nThe answer is 42.",
                "The answer is 42.",
                "Step 1: Analyze\nStep 2: Process\nStep 3: Conclude",
                true,
            ),
            // The prompt opened `<think>`, so a bare `</think>` at the very
            // start is an EMPTY thinking block, not stray text.
            (
                "empty thinking block - immediate close",
                "</think>\nHere is my answer.",
                "Here is my answer.",
                "",
                true,
            ),
            // ...but with thinking OFF the very same bytes are literal content.
            (
                "thinking disabled but model outputs think close anyway",
                "</think>\nSome content after spurious tag.",
                "</think>\nSome content after spurious tag.",
                "",
                false,
            ),
            (
                "thinking with only whitespace after close tag",
                "My thoughts...</think>   \n\t\n   Content here.",
                "Content here.",
                "My thoughts...",
                true,
            ),
            (
                "leading open think tag is ignored",
                "<think>\nLet me think about this...</think>\nHere is my answer.",
                "Here is my answer.",
                "Let me think about this...",
                true,
            ),
            (
                "empty explicit think block is ignored",
                "<think></think>\nHere is my answer.",
                "Here is my answer.",
                "",
                true,
            ),
        ];

        for (name, input, want_content, want_thinking, think) in cases {
            let mut p = nemotron(*think);
            let got = run(&mut p, input);
            assert_eq!(&got.content, want_content, "content, case {name}");
            assert_eq!(&got.thinking, want_thinking, "thinking, case {name}");
        }
    }

    /// Tool calls are delegated to the qwen3-coder parser, tags and all.
    #[test]
    fn a_tool_call_after_thinking_is_handed_to_the_coder_parser() {
        let mut p = nemotron(true);
        let got = run(
            &mut p,
            "I should check the weather...</think>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(got.thinking, "I should check the weather...");
        assert!(got.content.is_empty());
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "get_weather");
        assert_eq!(got.calls[0].function.arguments.get("city"), Some(&json!("Paris")));
    }

    #[test]
    fn thinking_then_content_then_a_tool_call_all_come_out_separately() {
        let mut p = nemotron(true);
        let got = run(
            &mut p,
            "Let me think...</think>\nI'll check for you.\n<tool_call>\n<function=search>\n<parameter=query>\ntest\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(got.thinking, "Let me think...");
        assert_eq!(got.content, "I'll check for you.");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "search");
    }

    /// The other way thinking can end: a `<tool_call>` with no `</think>` in
    /// front of it. The tag must be KEPT so the coder parser can open the call.
    #[test]
    fn a_tool_call_ends_thinking_even_with_no_closing_think_tag() {
        let mut p = nemotron(true);
        let got = run(
            &mut p,
            "I should look this up<tool_call>\n<function=search>\n<parameter=query>\nx\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(got.thinking, "I should look this up");
        assert!(got.content.is_empty());
        assert_eq!(got.calls.len(), 1, "the <tool_call> tag must reach the coder parser");
        assert_eq!(got.calls[0].function.name, "search");
    }

    /// Upstream `TestNemotron3NanoParser_Streaming`, the granular cases.
    #[test]
    fn nemotron_streams_correctly_across_awkward_chunk_boundaries() {
        // (name, chunks, want_content, want_thinking, want_call_count)
        type Case<'a> = (&'a str, Vec<&'a str>, &'a str, &'a str, usize);

        let cases: Vec<Case<'_>> = vec![
            ("empty thinking block", vec!["</think>", "\n", "Just content."], "Just content.", "", 0),
            (
                "tool call immediately after think close - no content",
                vec!["Analyzing...", "</think>", "\n", "<tool_call>", "\n<function=test>\n</function>\n", "</tool_call>"],
                "",
                "Analyzing...",
                1,
            ),
            // The redundant OPENING `<think>` split across chunks must still be
            // recognised and stripped.
            (
                "leading open think tag split across chunks",
                vec!["<th", "ink>", "\nThink first", "</think>", "\nDone."],
                "Done.",
                "Think first",
                0,
            ),
        ];

        for (name, chunks, want_content, want_thinking, want_calls) in cases {
            let mut p = nemotron(true);
            let mut got = Parsed::default();
            for chunk in &chunks {
                let part = p.add(chunk, false).expect("add");
                got.content.push_str(&part.content);
                got.thinking.push_str(&part.thinking);
                got.calls.extend(part.calls);
            }
            let drained = p.add("", true).expect("drain");
            got.content.push_str(&drained.content);
            got.thinking.push_str(&drained.thinking);
            got.calls.extend(drained.calls);

            assert_eq!(got.content, want_content, "content, case {name}");
            assert_eq!(got.thinking, want_thinking, "thinking, case {name}");
            assert_eq!(got.calls.len(), want_calls, "calls, case {name}");
        }
    }

    #[test]
    fn a_thinking_close_tag_split_across_chunks_never_leaks() {
        let mut p = nemotron(true);
        let a = p.add("thought</thi", false).expect("add");
        assert_eq!(a.thinking, "thought");
        assert!(a.content.is_empty());
        let b = p.add("nk>visible", false).expect("add");
        assert!(b.thinking.is_empty());
        assert_eq!(b.content, "visible");
    }

    /// One byte at a time must agree with one big chunk.
    #[test]
    fn feeding_one_byte_at_a_time_gives_the_same_answer_as_one_big_chunk() {
        let input = "let me reckon</think>\nThe answer is 4.";

        let mut whole = nemotron(true);
        let want = run(&mut whole, input);

        let mut p = nemotron(true);
        let mut got = Parsed::default();
        for (i, ch) in input.char_indices() {
            let part = p.add(&input[i..i + ch.len_utf8()], false).expect("add");
            got.content.push_str(&part.content);
            got.thinking.push_str(&part.thinking);
        }
        let drained = p.add("", true).expect("drain");
        got.content.push_str(&drained.content);
        got.thinking.push_str(&drained.thinking);

        assert_eq!(got.thinking, want.thinking);
        assert_eq!(got.content, want.content);
        assert_eq!(got.thinking, "let me reckon");
        assert_eq!(got.content, "The answer is 4.");
    }

    /// An assistant prefill with content skips thinking entirely.
    #[test]
    fn an_assistant_content_prefill_starts_the_stream_in_content_mode() {
        let mut p = Nemotron3NanoParser::default();
        let last = Message::new("assistant", "Sure:");
        p.init(Vec::new(), Some(&last), Some(&ThinkValue::Bool(true)));
        let got = run(&mut p, " here you go");
        assert_eq!(got.content, " here you go");
        assert!(got.thinking.is_empty());
    }

    /// `None` means thinking ON for this family.
    #[test]
    fn an_unspecified_think_value_defaults_to_thinking_on() {
        let mut p = Nemotron3NanoParser::default();
        p.init(Vec::new(), None, None);
        let got = run(&mut p, "reasoning</think>\nanswer");
        assert_eq!(got.thinking, "reasoning");
        assert_eq!(got.content, "answer");
    }

    #[test]
    fn nemotron_advertises_both_the_think_and_tool_call_tags() {
        let p = nemotron(true);
        assert_eq!(
            p.preserved_tokens(),
            vec!["<think>", "</think>", "<tool_call>", "</tool_call>"]
        );
        assert!(p.has_tool_support());
        assert!(p.has_thinking_support());
    }
}
