//! GLM-4.7 and GLM-OCR -- two thin variations on [`Glm46Parser`].
//!
//! **Upstream:** `model/parsers/glm47.go` and `model/parsers/glmocr.go`
//! (ollama, MIT). Both are one-screen files that embed `GLM46Parser` and
//! override `Init`; Go gets that by struct embedding, we get it by holding one.
//!
//! **Divergence, stated:** Go's embedding also inherits the *methods*, so
//! `GLM47Parser` automatically gets `Add`, `PreservedTokens` and friends. Rust
//! has no such inheritance, so each method here forwards by hand. Same
//! behaviour, three more lines each -- and forwarding explicitly means a future
//! change to [`Glm46Parser`] cannot silently change these two without anybody
//! looking at this file.

use crate::api::{Message, ThinkValue, Tool};

use super::glm46::{Glm46Parser, State};
use super::{Parsed, Parser, ParserError};

/// GLM-4.7: [`Glm46Parser`] that already knows thinking has started.
///
/// **Upstream:** `GLM47Parser`. GLM-4.7's *prompt* ends with `<think>`, so the
/// model's output begins **inside** the thinking block with no opening tag to
/// find -- exactly the same situation as qwen3. Leaving it in
/// `LookingForThinkingOpen` would file the whole reasoning block as content.
#[derive(Debug, Default)]
pub struct Glm47Parser {
    inner: Glm46Parser,
}

impl Parser for Glm47Parser {
    fn init(
        &mut self,
        tools: Vec<Tool>,
        _last_message: Option<&Message>,
        think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        self.inner.tools = tools.clone();
        self.inner.call_index = 0;
        // `None` counts as ON here: with thinking enabled the prompt ends with
        // `<think>`, so output starts as thinking content. Note upstream does NOT
        // reset the state in the else-branch -- a parser reused with thinking off
        // keeps whatever state it had. Ported as-is; build a fresh parser per
        // generation (which `parser_for_name` does) and it never bites.
        if think.is_none_or(|t| t.enabled()) {
            self.inner.state = State::CollectingThinking;
        }
        tools
    }

    fn add(&mut self, s: &str, done: bool) -> Result<Parsed, ParserError> {
        self.inner.add_inner(s, done)
    }

    fn preserved_tokens(&self) -> Vec<&'static str> {
        self.inner.preserved()
    }

    fn has_tool_support(&self) -> bool {
        true
    }

    fn has_thinking_support(&self) -> bool {
        true
    }
}

/// GLM-OCR: [`Glm46Parser`] with thinking switched off.
///
/// **Upstream:** `GlmOcrParser`. Note its `Init` sets `tools` but deliberately
/// does **not** reset `callIndex` -- the one difference from `Glm46Parser::init`
/// besides the capability flag. Kept faithful; it only matters if a parser is
/// reused across turns, which the normal path does not do.
#[derive(Debug, Default)]
pub struct GlmOcrParser {
    inner: Glm46Parser,
}

impl Parser for GlmOcrParser {
    fn init(
        &mut self,
        tools: Vec<Tool>,
        _last_message: Option<&Message>,
        _think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        self.inner.tools = tools.clone();
        tools
    }

    fn add(&mut self, s: &str, done: bool) -> Result<Parsed, ParserError> {
        self.inner.add_inner(s, done)
    }

    fn preserved_tokens(&self) -> Vec<&'static str> {
        self.inner.preserved()
    }

    fn has_tool_support(&self) -> bool {
        true
    }

    fn has_thinking_support(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The whole point of GLM-4.7's override: output starts INSIDE thinking, with
    /// no `<think>` to announce it.
    #[test]
    fn glm47_starts_inside_the_thinking_block_with_no_opening_tag() {
        let mut p = Glm47Parser::default();
        p.init(Vec::new(), None, Some(&ThinkValue::Bool(true)));
        let got = p.add("weighing it up</think>the answer", true).expect("add");
        assert_eq!(got.thinking, "weighing it up");
        assert_eq!(got.content, "the answer");
    }

    /// `None` counts as thinking on for this family.
    #[test]
    fn glm47_treats_an_unspecified_think_value_as_on() {
        let mut p = Glm47Parser::default();
        p.init(Vec::new(), None, None);
        let got = p.add("reasoning</think>answer", true).expect("add");
        assert_eq!(got.thinking, "reasoning");
        assert_eq!(got.content, "answer");
    }

    /// Thinking explicitly off: the machine stays in its default
    /// `LookingForThinkingOpen`, so a `<think>` in the output is still honoured
    /// but bare text is content.
    #[test]
    fn glm47_with_thinking_off_treats_bare_text_as_content() {
        let mut p = Glm47Parser::default();
        p.init(Vec::new(), None, Some(&ThinkValue::Bool(false)));
        let got = p.add("just an answer", true).expect("add");
        assert_eq!(got.content, "just an answer");
        assert!(got.thinking.is_empty());
    }

    #[test]
    fn glm47_still_parses_glm_tool_calls() {
        let mut p = Glm47Parser::default();
        p.init(Vec::new(), None, Some(&ThinkValue::Bool(true)));
        let got = p
            .add(
                "done</think><tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>",
                true,
            )
            .expect("add");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.arguments.get("a"), Some(&json!("1")));
    }

    /// GLM-OCR reports no thinking support, and does not start inside a thinking
    /// block regardless of what the caller asks for.
    #[test]
    fn glm_ocr_reports_no_thinking_support_and_starts_in_content() {
        let mut p = GlmOcrParser::default();
        assert!(!p.has_thinking_support());
        p.init(Vec::new(), None, Some(&ThinkValue::Bool(true)));
        let got = p.add("page text", true).expect("add");
        assert_eq!(got.content, "page text");
        assert!(got.thinking.is_empty());
    }

    #[test]
    fn both_variants_preserve_the_same_tokens_as_glm46() {
        let a = Glm47Parser::default();
        let b = GlmOcrParser::default();
        assert_eq!(a.preserved_tokens(), b.preserved_tokens());
        assert!(a.preserved_tokens().contains(&"<arg_value>"));
    }
}
