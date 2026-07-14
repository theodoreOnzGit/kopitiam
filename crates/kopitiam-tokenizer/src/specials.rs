//! Special tokens (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, ...) and
//! atomic matching.
//!
//! A special token is control metadata, not natural-language text. If it
//! were run through pre-tokenization and BPE like ordinary text, nothing
//! would stop `"<|endoftext|>"` from being split at, say, `<|` and
//! `endoftext|>` the moment those happen to be more frequent subword
//! boundaries than the whole string -- silently corrupting every prompt
//! template and chat format that relies on the model seeing that exact,
//! single token id. So special tokens must be pulled out of the input
//! *before* pre-tokenization and BPE ever see it, matched as whole,
//! indivisible strings.
//!
//! [`SpecialTokens`] does exactly that: it holds a `content -> id` table
//! and, whenever it is non-empty, a single compiled alternation regex over
//! every registered token's content (longest first, so one special token
//! that is a prefix of another -- e.g. `<|im_start|>` vs a hypothetical
//! `<|im_start|>_extra` -- always resolves to the longest match). Decoding
//! needs no special handling at all: a special token's raw bytes are just
//! the UTF-8 bytes of its content, so once it is registered in
//! [`crate::vocab::Vocab`] like any other token, byte concatenation
//! reconstructs it automatically.

use regex::Regex;

/// Registry of atomically-matched special tokens.
#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    entries: Vec<(String, u32)>,
    /// `None` when no special tokens are registered, so
    /// [`SpecialTokens::split`] can skip straight to the fast "no split
    /// needed" path instead of matching an alternation of zero patterns.
    matcher: Option<Regex>,
}

/// One piece of [`SpecialTokens::split`]'s output: either literal text
/// that still needs pre-tokenization and BPE, or a special token that has
/// already been resolved to its id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment<'a> {
    Text(&'a str),
    Special(u32),
}

impl SpecialTokens {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `content` as a special token with the given `id`,
    /// rebuilding the matcher regex. Does nothing but return `Ok` if the
    /// exact same `(content, id)` pair is registered twice.
    pub fn register(&mut self, content: impl Into<String>, id: u32) {
        let content = content.into();
        if self.entries.iter().any(|(c, i)| c == &content && *i == id) {
            return;
        }
        self.entries.push((content, id));
        self.rebuild_matcher();
    }

    pub fn id_of(&self, content: &str) -> Option<u32> {
        self.entries
            .iter()
            .find(|(c, _)| c == content)
            .map(|(_, id)| *id)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Splits `text` into an alternating sequence of plain-text and
    /// resolved-special-token segments. With no special tokens registered
    /// this is a single `Segment::Text(text)`.
    pub fn split<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        let Some(matcher) = &self.matcher else {
            return vec![Segment::Text(text)];
        };

        let mut segments = Vec::new();
        let mut cursor = 0;
        for m in matcher.find_iter(text) {
            if m.start() > cursor {
                segments.push(Segment::Text(&text[cursor..m.start()]));
            }
            let id = self
                .id_of(m.as_str())
                .expect("matcher is built only from registered content");
            segments.push(Segment::Special(id));
            cursor = m.end();
        }
        if cursor < text.len() {
            segments.push(Segment::Text(&text[cursor..]));
        }
        segments
    }

    fn rebuild_matcher(&mut self) {
        // Longest-first so a special token that is a textual prefix of
        // another always loses to the longer (more specific) one --
        // `regex`'s alternation is leftmost-first, so pattern order here
        // is what decides that, not match length.
        let mut contents: Vec<&str> = self.entries.iter().map(|(c, _)| c.as_str()).collect();
        contents.sort_unstable_by_key(|c| std::cmp::Reverse(c.len()));
        let pattern = contents
            .into_iter()
            .map(regex::escape)
            .collect::<Vec<_>>()
            .join("|");
        self.matcher =
            Some(Regex::new(&pattern).expect("escaped literal alternation is always valid"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_specials_registered_is_a_single_text_segment() {
        let specials = SpecialTokens::new();
        assert_eq!(
            specials.split("hello world"),
            vec![Segment::Text("hello world")]
        );
    }

    #[test]
    fn splits_around_a_special_token() {
        let mut specials = SpecialTokens::new();
        specials.register("<|endoftext|>", 100);
        assert_eq!(
            specials.split("hello<|endoftext|>world"),
            vec![
                Segment::Text("hello"),
                Segment::Special(100),
                Segment::Text("world"),
            ]
        );
    }

    #[test]
    fn special_token_at_start_or_end_produces_no_empty_text_segment() {
        let mut specials = SpecialTokens::new();
        specials.register("<|endoftext|>", 100);
        assert_eq!(
            specials.split("<|endoftext|>hello<|endoftext|>"),
            vec![
                Segment::Special(100),
                Segment::Text("hello"),
                Segment::Special(100),
            ]
        );
    }

    #[test]
    fn back_to_back_special_tokens_produce_no_text_segment_between_them() {
        let mut specials = SpecialTokens::new();
        specials.register("<|im_start|>", 1);
        specials.register("<|im_end|>", 2);
        assert_eq!(
            specials.split("<|im_start|><|im_end|>"),
            vec![Segment::Special(1), Segment::Special(2)]
        );
    }

    #[test]
    fn longest_match_wins_when_one_special_token_prefixes_another() {
        let mut specials = SpecialTokens::new();
        specials.register("<|im_start|>", 1);
        specials.register("<|im_start|>extra", 2);
        assert_eq!(
            specials.split("<|im_start|>extra"),
            vec![Segment::Special(2)]
        );
    }

    #[test]
    fn regex_metacharacters_in_special_token_content_are_treated_literally() {
        let mut specials = SpecialTokens::new();
        specials.register("[SEP]", 5);
        assert_eq!(
            specials.split("a[SEP]b"),
            vec![Segment::Text("a"), Segment::Special(5), Segment::Text("b")]
        );
    }
}
