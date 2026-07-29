//! **Laguna** / **Poolside v1** -- Poolside's XML-tagged turn framing.
//!
//! **Upstream:** `model/renderers/laguna.go`. Two renderers, two names:
//!
//! | Registered as | Rust type | Upstream type | Template fixture |
//! |---|---|---|---|
//! | `laguna` | [`LagunaRenderer`] | `LagunaRenderer` | `laguna_glm_thinking_v5` (a.k.a. v2) |
//! | `poolside-v1` | [`LagunaV8Renderer`] | `LagunaV8Renderer` | `laguna_glm_thinking_v8` |
//!
//! ## The BOS is a **full-width** bracket pair
//!
//! [`LAGUNA_BOS`] is `〈|EOS|〉` -- U+3008 / U+3009 (CJK angle brackets), **not**
//! ASCII `<` `>`, and yes it says *EOS* while being used as the *beginning*-of-
//! sequence token. That is Poolside's own tokenizer vocabulary, not a typo of
//! ours. Copy the constant; never retype it, because the ASCII lookalike is a
//! completely different token and the model will not recognise it.
//!
//! ## v2 vs v8: same vocabulary, opposite whitespace
//!
//! Both use `<system>`, `<user>`, `<assistant>`, `<tool_response>`,
//! `<tool_call>`, `<arg_key>`, `<arg_value>`, `<think>`/`</think>`. The
//! difference is **where the newlines go**, and it is total:
//!
//! * **v2 ([`LagunaRenderer`]) puts content on its own lines** --
//!   `<user>\nHello\n</user>\n`.
//! * **v8 ([`LagunaV8Renderer`]) hugs the tags** -- `<user>Hello</user>\n`, with
//!   the only newline after the closing tag.
//!
//! Which means the two must stay separate implementations. Merging them behind a
//! flag would be a single `if` per newline and the first careless edit silently
//! reframes one of the two families.
//!
//! ## Thinking is signalled by the generation prompt, not the system message
//!
//! Both variants end with `<think>` when thinking is on and `</think>` when it
//! is off -- an **already-closed** reasoning block telling the model to answer
//! directly. It defaults **off**: `think != nil && think.Bool()`, so an absent
//! `think` is off here, unlike several other families where absent means "use my
//! default".
//!
//! ## The default system message
//!
//! With no leading `system` message, both seed [`LAGUNA_DEFAULT_SYSTEM`]. An
//! **explicit but empty** system message is how a caller opts *out* of the
//! header entirely -- upstream's `empty_first_system_opts_out_of_header`
//! fixture. So `Some("")` and `None` mean genuinely different things here.

use serde_json::Value;

use super::json::{add_spaces_outside_strings, go_value, marshal_tool_with_spaces};
use super::{Message, RenderError, Renderer, ThinkValue, Tool};
use crate::api::ToolCall;

/// Poolside's beginning-of-sequence token: **full-width** `〈` U+3008 and `〉`
/// U+3009. **Upstream:** `lagunaBOS`. See the module docs -- the ASCII
/// lookalike is a different token entirely.
pub(crate) const LAGUNA_BOS: &str = "〈|EOS|〉";

const THOUGHT_OPEN: &str = "<think>";
const THOUGHT_CLOSE: &str = "</think>";

/// **Upstream:** `lagunaDefaultSystem` -- lifted from the Laguna chat template,
/// used whenever the request supplies no system message. Verbatim, including the
/// hyphen in "conversationally-fluent".
pub(crate) const LAGUNA_DEFAULT_SYSTEM: &str = "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be helpful to users through natural language conversations.";

/// The `### Tools` preamble, shared by both variants.
///
/// **Upstream:** the three `sb.WriteString` calls after `### Tools\n\n` in both
/// `Render` methods. Identical text in both, so shared here -- the *whitespace
/// around* it is what differs, and that stays at each call site.
const TOOLS_PREAMBLE: &str = "You may call functions to assist with the user query.\nAll available function signatures are listed below:\n<available_tools>\n";

/// The example call shape shown to the model, v2 only.
///
/// **Upstream:** the final `sb.WriteString` of v2's tools block. v8 stops after
/// `</available_tools>` and never shows an example -- that asymmetry is real and
/// upstream's fixtures pin it on both sides.
const V2_TOOL_CALL_EXAMPLE: &str = "<tool_call>function-name\n<arg_key>argument-key</arg_key>\n<arg_value>value-of-argument-key</arg_value>\n</tool_call>";

const V2_THINKING_INSTRUCTIONS: &str = "Wrap your thinking in '<think>', '</think>' tags, followed by a function call. For each function call, return an unescaped XML-like object with function name and arguments within '<tool_call>' and '</tool_call>' tags, like here:\n<think> your thoughts here </think>\n";
const V2_NO_THINKING_INSTRUCTIONS: &str = "For each function call, return an unescaped XML-like object with function name and arguments within '<tool_call>' and '</tool_call>' tags, like here:\n";

/// **Upstream:** `LagunaRenderer` -- the v2 / `laguna` framing. No configuration.
#[derive(Debug, Clone, Copy, Default)]
pub struct LagunaRenderer;

/// **Upstream:** `LagunaV8Renderer` -- the v8 / `poolside-v1` framing. No
/// configuration.
#[derive(Debug, Clone, Copy, Default)]
pub struct LagunaV8Renderer;

/// Pick the header's system text, and say whether it came from the messages.
///
/// **Upstream:** the identical five lines at the top of both `Render` methods.
/// Returns `(system_text, first_message_is_system)`; the second half matters
/// because a leading system message is **consumed** by the header and must be
/// skipped in the main loop, while a *later* system message is rendered as its
/// own `<system>` turn.
fn header_system(messages: &[Message]) -> (&str, bool) {
    match messages.first() {
        Some(first) if first.role == "system" => (first.content.as_str(), true),
        _ => (LAGUNA_DEFAULT_SYSTEM, false),
    }
}

/// Go's `strings.TrimRightFunc(s, unicode.IsSpace)`.
///
/// Right-trim only: leading whitespace in a system message is the caller's, and
/// upstream keeps it.
fn trim_right_space(s: &str) -> &str {
    s.trim_end()
}

/// **Upstream:** `formatLagunaToolCallArgument`.
///
/// A string goes in **raw and unquoted**; everything else is marshalled with
/// [`add_spaces_outside_strings`], so a nested object arrives Python-spaced
/// (`{"mode": "fast"}`). Quoting the string case would put `"Paris"` inside
/// `<arg_value>`, which is not what the model emits and therefore not what it
/// should be shown.
fn format_tool_call_argument(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => add_spaces_outside_strings(&go_value(other)),
    }
}

/// Split an assistant turn into `(content, reasoning)`.
///
/// **Upstream:** `lagunaV2AssistantContent`. v2 only -- v8 never digs reasoning
/// out of the content.
///
/// Rules, in order:
///
/// 1. **No `</think>` in the content at all** -> nothing to extract; both values
///    pass through untouched.
/// 2. Otherwise the content is split on `</think>` and the **last** chunk becomes
///    the visible content, left-trimmed of newlines only.
/// 3. The reasoning is only recovered from the content when the caller supplied
///    **none** -- an explicit `thinking` field always wins. Upstream's
///    `assistant_thinking_metadata_overrides_content_tags` fixture pins that.
/// 4. When recovering, everything after the **last** `<think>` (if there is one)
///    and before the first `</think>` is the reasoning, right-trimmed of
///    newlines and then left-trimmed of newlines.
fn v2_assistant_content(content: &str, reasoning: &str) -> (String, String) {
    let parts: Vec<&str> = content.split(THOUGHT_CLOSE).collect();
    if parts.len() == 1 {
        return (content.to_string(), reasoning.to_string());
    }

    let reasoning = if reasoning.is_empty() {
        let mut before = parts[0].trim_end_matches('\n');
        if let Some(i) = before.rfind(THOUGHT_OPEN) {
            before = &before[i + THOUGHT_OPEN.len()..];
        }
        before.trim_start_matches('\n').to_string()
    } else {
        reasoning.to_string()
    };

    let content = parts[parts.len() - 1].trim_start_matches('\n').to_string();
    (content, reasoning)
}

impl Renderer for LagunaRenderer {
    fn render(
        &self,
        messages: &[Message],
        tools: &[Tool],
        think: Option<&ThinkValue>,
    ) -> Result<String, RenderError> {
        let mut sb = String::from(LAGUNA_BOS);

        // Defaults OFF: an absent `think` is not "use my default" here.
        let thinking_enabled = think.is_some_and(ThinkValue::enabled);

        let (system_message, first_is_system) = header_system(messages);
        let has_system = !system_message.trim().is_empty();

        if has_system || !tools.is_empty() {
            sb.push_str("<system>\n");
            if has_system {
                // The extra `\n` before the text is upstream's, so the header
                // opens with a blank line: `<system>\n\n{text}`.
                sb.push('\n');
                sb.push_str(trim_right_space(system_message));
            }
            if !tools.is_empty() {
                sb.push_str("\n\n### Tools\n\n");
                sb.push_str(TOOLS_PREAMBLE);
                for tool in tools {
                    sb.push_str(&marshal_tool_with_spaces(tool));
                    sb.push('\n');
                }
                sb.push_str("</available_tools>\n\n");
                sb.push_str(if thinking_enabled {
                    V2_THINKING_INSTRUCTIONS
                } else {
                    V2_NO_THINKING_INSTRUCTIONS
                });
                sb.push_str(V2_TOOL_CALL_EXAMPLE);
            }
            sb.push_str("\n</system>\n");
        }

        for (i, message) in messages.iter().enumerate() {
            if i == 0 && first_is_system {
                continue;
            }
            match message.role.as_str() {
                "user" => {
                    sb.push_str("<user>\n");
                    sb.push_str(&message.content);
                    sb.push_str("\n</user>\n");
                }
                "assistant" => {
                    let (content, reasoning) =
                        v2_assistant_content(&message.content, &message.thinking);
                    // A **last** assistant turn with anything in it is a
                    // prefill: leave `<assistant>` open so the model continues
                    // it, and skip the generation prompt entirely below.
                    let is_last = i == messages.len() - 1;
                    let prefill = is_last
                        && (!content.trim().is_empty()
                            || !reasoning.trim().is_empty()
                            || !message.tool_calls.is_empty());

                    sb.push_str("<assistant>\n");

                    // Every assistant turn opens with a reasoning block: a full
                    // `<think>...</think>` when there is reasoning, otherwise a
                    // **bare `</think>`** marking the turn as direct. Omitting
                    // it entirely is off-distribution.
                    let reasoning = reasoning.trim();
                    if reasoning.is_empty() {
                        sb.push_str(THOUGHT_CLOSE);
                        sb.push('\n');
                    } else {
                        sb.push_str(THOUGHT_OPEN);
                        sb.push('\n');
                        sb.push_str(reasoning);
                        sb.push('\n');
                        sb.push_str(THOUGHT_CLOSE);
                        sb.push('\n');
                    }

                    if !content.trim().is_empty() {
                        sb.push_str(content.trim());
                        sb.push('\n');
                    }

                    for tc in &message.tool_calls {
                        write_v2_tool_call(&mut sb, tc);
                    }

                    if !prefill {
                        sb.push_str("</assistant>\n");
                    }
                }
                "tool" => {
                    sb.push_str("<tool_response>\n");
                    sb.push_str(&message.content);
                    sb.push_str("\n</tool_response>\n");
                }
                "system" => {
                    sb.push_str("<system>\n");
                    sb.push_str(&message.content);
                    sb.push_str("\n</system>\n");
                }
                // Any other role is dropped -- upstream's switch has no default
                // arm. Silent, but faithful.
                _ => {}
            }
        }

        // Continue a prefill in place; otherwise open a fresh assistant turn and
        // prime the reasoning mode.
        if messages.last().is_none_or(|m| m.role != "assistant") {
            sb.push_str("<assistant>\n");
            sb.push_str(if thinking_enabled {
                THOUGHT_OPEN
            } else {
                THOUGHT_CLOSE
            });
        }

        Ok(sb)
    }

    fn leading_bos(&self) -> &'static str {
        LAGUNA_BOS
    }
}

/// v2's tool-call block: newline after the name, and after every tag.
fn write_v2_tool_call(sb: &mut String, tc: &ToolCall) {
    sb.push_str("<tool_call>");
    sb.push_str(&tc.function.name);
    sb.push('\n');
    for (name, value) in tc.function.arguments.0.iter() {
        sb.push_str("<arg_key>");
        sb.push_str(name);
        sb.push_str("</arg_key>\n");
        sb.push_str("<arg_value>");
        sb.push_str(&format_tool_call_argument(value));
        sb.push_str("</arg_value>\n");
    }
    sb.push_str("</tool_call>\n");
}

impl Renderer for LagunaV8Renderer {
    fn render(
        &self,
        messages: &[Message],
        tools: &[Tool],
        think: Option<&ThinkValue>,
    ) -> Result<String, RenderError> {
        let mut sb = String::from(LAGUNA_BOS);

        let thinking_enabled = think.is_some_and(ThinkValue::enabled);

        let (system_message, first_is_system) = header_system(messages);
        let has_system = !system_message.trim().is_empty();

        // The extra `|| thinking_enabled` is v8's alone: with thinking on it
        // emits an EMPTY `<system></system>` even with nothing to say. v2 does
        // not. Upstream's `empty_first_system_thinking_enabled` fixture pins it.
        if has_system || !tools.is_empty() || thinking_enabled {
            sb.push_str("<system>");
            if has_system {
                sb.push_str(trim_right_space(system_message));
                if !tools.is_empty() {
                    sb.push_str("\n\n");
                }
            }
            if !tools.is_empty() {
                sb.push_str("### Tools\n\n");
                sb.push_str(TOOLS_PREAMBLE);
                for tool in tools {
                    sb.push_str(&marshal_tool_with_spaces(tool));
                    sb.push('\n');
                }
                // No trailing newline and no example call -- v8 stops here.
                sb.push_str("</available_tools>");
            }
            sb.push_str("</system>\n");
        }

        for (i, message) in messages.iter().enumerate() {
            if i == 0 && first_is_system {
                continue;
            }
            match message.role.as_str() {
                "user" => {
                    sb.push_str("<user>");
                    sb.push_str(&message.content);
                    sb.push_str("</user>\n");
                }
                "assistant" => {
                    sb.push_str("<assistant>");
                    if thinking_enabled {
                        // Note `thinking` goes in **raw** -- no trim, and an
                        // empty one yields `<think></think>`. v8 also ignores
                        // any `<think>` tags already in the content, unlike v2.
                        sb.push_str(THOUGHT_OPEN);
                        sb.push_str(&message.thinking);
                        sb.push_str(THOUGHT_CLOSE);
                    } else {
                        // Thinking off drops the reasoning entirely, even when
                        // the caller supplied some.
                        sb.push_str(THOUGHT_CLOSE);
                    }
                    sb.push_str(&message.content);
                    for tc in &message.tool_calls {
                        write_v8_tool_call(&mut sb, tc);
                    }
                    // v8 always closes; there is no prefill path.
                    sb.push_str("</assistant>\n");
                }
                "tool" => {
                    sb.push_str("<tool_response>");
                    sb.push_str(&message.content);
                    sb.push_str("</tool_response>\n");
                }
                "system" => {
                    sb.push_str("<system>");
                    sb.push_str(&message.content);
                    sb.push_str("</system>\n");
                }
                _ => {}
            }
        }

        // Unconditional -- v8 always opens a fresh turn.
        sb.push_str("<assistant>");
        sb.push_str(if thinking_enabled {
            THOUGHT_OPEN
        } else {
            THOUGHT_CLOSE
        });

        Ok(sb)
    }

    fn leading_bos(&self) -> &'static str {
        LAGUNA_BOS
    }
}

/// v8's tool-call block: **no newlines anywhere inside**.
fn write_v8_tool_call(sb: &mut String, tc: &ToolCall) {
    sb.push_str("<tool_call>");
    sb.push_str(&tc.function.name);
    for (name, value) in tc.function.arguments.0.iter() {
        sb.push_str("<arg_key>");
        sb.push_str(name);
        sb.push_str("</arg_key>");
        sb.push_str("<arg_value>");
        sb.push_str(&format_tool_call_argument(value));
        sb.push_str("</arg_value>");
    }
    sb.push_str("</tool_call>");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ToolCallFunction, ToolFunction, ToolFunctionParameters, ToolProperty};
    use indexmap::IndexMap;
    use serde_json::json;

    fn v2(msgs: &[Message], tools: &[Tool], think: Option<ThinkValue>) -> String {
        LagunaRenderer
            .render(msgs, tools, think.as_ref())
            .expect("laguna render never fails")
    }

    fn v8(msgs: &[Message], tools: &[Tool], think: Option<ThinkValue>) -> String {
        LagunaV8Renderer
            .render(msgs, tools, think.as_ref())
            .expect("laguna v8 render never fails")
    }

    fn tool(name: &str, description: &str, props: &[(&str, &str, &str)]) -> Tool {
        let mut map: IndexMap<String, ToolProperty> = IndexMap::new();
        for (key, ty, desc) in props {
            map.insert(
                (*key).to_string(),
                serde_json::from_value(json!({"type": ty, "description": desc})).unwrap(),
            );
        }
        Tool {
            tool_type: "function".into(),
            items: None,
            function: ToolFunction {
                name: name.into(),
                description: description.into(),
                parameters: ToolFunctionParameters {
                    param_type: "object".into(),
                    defs: None,
                    items: None,
                    required: props.iter().map(|(k, _, _)| (*k).to_string()).collect(),
                    properties: Some(map),
                },
            },
        }
    }

    /// Upstream `lagunaWeatherTool()`.
    fn weather_tool() -> Tool {
        tool(
            "get_weather",
            "Get weather",
            &[("location", "string", "City")],
        )
    }

    /// Upstream `lagunaMathTool()`.
    fn math_tool() -> Tool {
        tool(
            "add",
            "Add numbers",
            &[
                ("a", "number", "First number"),
                ("b", "number", "Second number"),
            ],
        )
    }

    /// Upstream's `lagunaToolJSON` / `lagunaMathToolJSON` constants. If these
    /// two drift, the Go-shaped JSON emitter in [`super::json`] has regressed,
    /// not this renderer.
    const WEATHER_JSON: &str = r#"{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "required": ["location"], "properties": {"location": {"type": "string", "description": "City"}}}}}"#;
    const MATH_JSON: &str = r#"{"type": "function", "function": {"name": "add", "description": "Add numbers", "parameters": {"type": "object", "required": ["a", "b"], "properties": {"a": {"type": "number", "description": "First number"}, "b": {"type": "number", "description": "Second number"}}}}}"#;

    fn assistant_with(content: &str, thinking: &str, calls: Vec<ToolCall>) -> Message {
        Message {
            role: "assistant".into(),
            content: content.into(),
            thinking: thinking.into(),
            tool_calls: calls,
            ..Default::default()
        }
    }

    fn call(name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: String::new(),
            function: ToolCallFunction {
                index: 0,
                name: name.to_string(),
                arguments: serde_json::from_str(args).expect("test args must be valid JSON"),
            },
        }
    }

    fn v2_default_header() -> String {
        format!("{LAGUNA_BOS}<system>\n\n{LAGUNA_DEFAULT_SYSTEM}\n</system>\n")
    }

    fn v8_default_header() -> String {
        format!("{LAGUNA_BOS}<system>{LAGUNA_DEFAULT_SYSTEM}</system>\n")
    }

    /// Upstream `TestLagunaRendererReferenceFlowCoverage` -- the v2 fixtures,
    /// all of them, in upstream's order. Upstream cross-checks these against the
    /// real Jinja template under `VERIFY_JINJA2=1`, so they are the spec.
    #[test]
    fn laguna_v2_matches_every_upstream_fixture() {
        let h = v2_default_header();
        let t = || Some(ThinkValue::Bool(true));
        let f = || Some(ThinkValue::Bool(false));

        // empty_messages -- the header still fires, on the default system text.
        assert_eq!(v2(&[], &[], None), format!("{h}<assistant>\n</think>"));

        // user_only_default -- absent `think` means OFF here.
        assert_eq!(
            v2(&[Message::new("user", "Hello")], &[], None),
            format!("{h}<user>\nHello\n</user>\n<assistant>\n</think>")
        );
        assert_eq!(
            v2(&[Message::new("user", "Hello")], &[], t()),
            format!("{h}<user>\nHello\n</user>\n<assistant>\n<think>")
        );
        assert_eq!(
            v2(&[Message::new("user", "Hello")], &[], f()),
            format!("{h}<user>\nHello\n</user>\n<assistant>\n</think>")
        );

        // first_system_is_header -- and it is right-trimmed.
        assert_eq!(
            v2(
                &[
                    Message::new("system", "Stay concise.\n\n"),
                    Message::new("user", "Hi")
                ],
                &[],
                None
            ),
            format!(
                "{LAGUNA_BOS}<system>\n\nStay concise.\n</system>\n<user>\nHi\n</user>\n<assistant>\n</think>"
            )
        );

        // empty_first_system_opts_out_of_header -- no header AT ALL, which is
        // why `Some("")` and `None` are different requests.
        assert_eq!(
            v2(
                &[Message::new("system", ""), Message::new("user", "Hi")],
                &[],
                None
            ),
            format!("{LAGUNA_BOS}<user>\nHi\n</user>\n<assistant>\n</think>")
        );

        // additional_system -- a LATER system message is its own turn.
        assert_eq!(
            v2(
                &[
                    Message::new("system", "Primary."),
                    Message::new("user", "Hi"),
                    Message::new("system", "Secondary."),
                ],
                &[],
                None
            ),
            format!(
                "{LAGUNA_BOS}<system>\n\nPrimary.\n</system>\n<user>\nHi\n</user>\n\
                 <system>\nSecondary.\n</system>\n<assistant>\n</think>"
            )
        );

        // empty_first_system_with_tools -- the header fires for the tools alone,
        // and the system slot contributes nothing, giving THREE newlines.
        assert_eq!(
            v2(
                &[Message::new("system", ""), Message::new("user", "Weather?")],
                &[weather_tool()],
                None
            ),
            format!(
                "{LAGUNA_BOS}<system>\n\n\n### Tools\n\n{TOOLS_PREAMBLE}{WEATHER_JSON}\n\
                 </available_tools>\n\n{V2_NO_THINKING_INSTRUCTIONS}{V2_TOOL_CALL_EXAMPLE}\
                 \n</system>\n<user>\nWeather?\n</user>\n<assistant>\n</think>"
            )
        );

        // tools_in_header, thinking on -- the instructions change.
        assert_eq!(
            v2(
                &[
                    Message::new("system", "Stay concise."),
                    Message::new("user", "Weather?")
                ],
                &[weather_tool()],
                t()
            ),
            format!(
                "{LAGUNA_BOS}<system>\n\nStay concise.\n\n### Tools\n\n{TOOLS_PREAMBLE}\
                 {WEATHER_JSON}\n</available_tools>\n\n{V2_THINKING_INSTRUCTIONS}\
                 {V2_TOOL_CALL_EXAMPLE}\n</system>\n<user>\nWeather?\n</user>\n<assistant>\n<think>"
            )
        );

        // tools_default
        assert_eq!(
            v2(&[Message::new("user", "Weather?")], &[weather_tool()], None),
            format!(
                "{LAGUNA_BOS}<system>\n\n{LAGUNA_DEFAULT_SYSTEM}\n\n### Tools\n\n{TOOLS_PREAMBLE}\
                 {WEATHER_JSON}\n</available_tools>\n\n{V2_NO_THINKING_INSTRUCTIONS}\
                 {V2_TOOL_CALL_EXAMPLE}\n</system>\n<user>\nWeather?\n</user>\n<assistant>\n</think>"
            )
        );

        // multiple_tools_in_header -- one line each, no separator beyond `\n`.
        assert_eq!(
            v2(
                &[Message::new("user", "Add then report weather")],
                &[weather_tool(), math_tool()],
                None
            ),
            format!(
                "{LAGUNA_BOS}<system>\n\n{LAGUNA_DEFAULT_SYSTEM}\n\n### Tools\n\n{TOOLS_PREAMBLE}\
                 {WEATHER_JSON}\n{MATH_JSON}\n</available_tools>\n\n{V2_NO_THINKING_INSTRUCTIONS}\
                 {V2_TOOL_CALL_EXAMPLE}\n</system>\n<user>\nAdd then report weather\n</user>\n\
                 <assistant>\n</think>"
            )
        );

        // assistant_history -- reasoning block, trimmed content, tool call.
        assert_eq!(
            v2(
                &[
                    Message::new("user", "Add these."),
                    assistant_with(
                        "\nCalling the tool.\n",
                        "Need addition.",
                        vec![call("add", r#"{"a":2,"b":3}"#)]
                    ),
                    Message::new("tool", "5"),
                    Message::new("user", "Thanks"),
                ],
                &[],
                t()
            ),
            format!(
                "{h}<user>\nAdd these.\n</user>\n<assistant>\n<think>\nNeed addition.\n</think>\n\
                 Calling the tool.\n<tool_call>add\n<arg_key>a</arg_key>\n<arg_value>2</arg_value>\n\
                 <arg_key>b</arg_key>\n<arg_value>3</arg_value>\n</tool_call>\n</assistant>\n\
                 <tool_response>\n5\n</tool_response>\n<user>\nThanks\n</user>\n<assistant>\n<think>"
            )
        );

        // assistant_extracts_thinking_from_content
        assert_eq!(
            v2(
                &[
                    Message::new("user", "Explain"),
                    Message::new("assistant", "<think>\nPlan\n</think>\nAnswer\n\n"),
                    Message::new("user", "Next"),
                ],
                &[],
                t()
            ),
            format!(
                "{h}<user>\nExplain\n</user>\n<assistant>\n<think>\nPlan\n</think>\nAnswer\n\
                 </assistant>\n<user>\nNext\n</user>\n<assistant>\n<think>"
            )
        );

        // assistant_thinking_metadata_overrides_content_tags -- the `thinking`
        // field beats whatever the content says.
        assert_eq!(
            v2(
                &[
                    Message::new("user", "Explain"),
                    assistant_with(
                        "<think>Ignore this</think>\nAnswer",
                        "Use metadata.",
                        vec![]
                    ),
                    Message::new("user", "Next"),
                ],
                &[],
                None
            ),
            format!(
                "{h}<user>\nExplain\n</user>\n<assistant>\n<think>\nUse metadata.\n</think>\n\
                 Answer\n</assistant>\n<user>\nNext\n</user>\n<assistant>\n</think>"
            )
        );

        // assistant_whitespace_content_only -- bare `</think>`, no content line.
        assert_eq!(
            v2(
                &[
                    Message::new("user", "Continue"),
                    Message::new("assistant", " \n\t "),
                    Message::new("user", "Next"),
                ],
                &[],
                None
            ),
            format!(
                "{h}<user>\nContinue\n</user>\n<assistant>\n</think>\n</assistant>\n\
                 <user>\nNext\n</user>\n<assistant>\n</think>"
            )
        );

        // assistant_multiple_tool_calls_mixed_args
        assert_eq!(
            v2(
                &[
                    Message::new("user", "Do calls"),
                    assistant_with(
                        "",
                        "",
                        vec![
                            call("echo", r#"{"text":"hello","count":2}"#),
                            call("configure", r#"{"flag":true,"options":{"mode":"fast"}}"#),
                        ]
                    ),
                    Message::new("user", "Done?"),
                ],
                &[],
                None
            ),
            format!(
                "{h}<user>\nDo calls\n</user>\n<assistant>\n</think>\n\
                 <tool_call>echo\n<arg_key>text</arg_key>\n<arg_value>hello</arg_value>\n\
                 <arg_key>count</arg_key>\n<arg_value>2</arg_value>\n</tool_call>\n\
                 <tool_call>configure\n<arg_key>flag</arg_key>\n<arg_value>true</arg_value>\n\
                 <arg_key>options</arg_key>\n<arg_value>{{\"mode\": \"fast\"}}</arg_value>\n\
                 </tool_call>\n</assistant>\n<user>\nDone?\n</user>\n<assistant>\n</think>"
            )
        );
    }

    /// Upstream `TestLagunaRendererAssistantPrefill`.
    ///
    /// v2 leaves a trailing assistant turn **open** so the model continues it,
    /// and emits no generation prompt. Upstream flags this as a **deliberate
    /// divergence from the Jinja template** (`TestLagunaRendererKnownJinja2-
    /// Differences`): the template would close the turn and open a fresh one,
    /// throwing the prefill away. Do not "fix" it back to match the template.
    #[test]
    fn a_trailing_v2_assistant_turn_is_left_open_as_a_prefill() {
        assert_eq!(
            v2(
                &[
                    Message::new("user", "Complete this"),
                    Message::new("assistant", "Partial"),
                ],
                &[],
                None
            ),
            format!(
                "{}<user>\nComplete this\n</user>\n<assistant>\n</think>\nPartial\n",
                v2_default_header()
            )
        );
    }

    /// Upstream `TestLagunaV8RendererReferenceFlowCoverage` -- all of it.
    #[test]
    fn laguna_v8_matches_every_upstream_fixture() {
        let h = v8_default_header();
        let t = || Some(ThinkValue::Bool(true));
        let f = || Some(ThinkValue::Bool(false));

        assert_eq!(v8(&[], &[], None), format!("{h}<assistant></think>"));
        assert_eq!(
            v8(&[Message::new("user", "Hello")], &[], None),
            format!("{h}<user>Hello</user>\n<assistant></think>")
        );
        assert_eq!(
            v8(&[Message::new("user", "Hello")], &[], t()),
            format!("{h}<user>Hello</user>\n<assistant><think>")
        );
        assert_eq!(
            v8(&[Message::new("user", "Hello")], &[], f()),
            format!("{h}<user>Hello</user>\n<assistant></think>")
        );

        assert_eq!(
            v8(
                &[
                    Message::new("system", "Stay concise.\n\n"),
                    Message::new("user", "Hi")
                ],
                &[],
                None
            ),
            format!(
                "{LAGUNA_BOS}<system>Stay concise.</system>\n<user>Hi</user>\n<assistant></think>"
            )
        );

        assert_eq!(
            v8(
                &[Message::new("system", ""), Message::new("user", "Hi")],
                &[],
                None
            ),
            format!("{LAGUNA_BOS}<user>Hi</user>\n<assistant></think>")
        );

        assert_eq!(
            v8(
                &[Message::new("system", ""), Message::new("user", "Weather?")],
                &[weather_tool()],
                None
            ),
            format!(
                "{LAGUNA_BOS}<system>### Tools\n\n{TOOLS_PREAMBLE}{WEATHER_JSON}\n\
                 </available_tools></system>\n<user>Weather?</user>\n<assistant></think>"
            )
        );

        // empty_first_system_thinking_enabled -- v8 ONLY: an empty
        // `<system></system>` appears purely because thinking is on.
        assert_eq!(
            v8(
                &[Message::new("system", ""), Message::new("user", "Hi")],
                &[],
                t()
            ),
            format!("{LAGUNA_BOS}<system></system>\n<user>Hi</user>\n<assistant><think>")
        );

        assert_eq!(
            v8(
                &[
                    Message::new("system", "Primary."),
                    Message::new("user", "Hi"),
                    Message::new("system", "Secondary."),
                ],
                &[],
                None
            ),
            format!(
                "{LAGUNA_BOS}<system>Primary.</system>\n<user>Hi</user>\n\
                 <system>Secondary.</system>\n<assistant></think>"
            )
        );

        assert_eq!(
            v8(
                &[
                    Message::new("system", "Stay concise."),
                    Message::new("user", "Weather?")
                ],
                &[weather_tool()],
                t()
            ),
            format!(
                "{LAGUNA_BOS}<system>Stay concise.\n\n### Tools\n\n{TOOLS_PREAMBLE}{WEATHER_JSON}\n\
                 </available_tools></system>\n<user>Weather?</user>\n<assistant><think>"
            )
        );

        assert_eq!(
            v8(&[Message::new("user", "Weather?")], &[weather_tool()], None),
            format!(
                "{LAGUNA_BOS}<system>{LAGUNA_DEFAULT_SYSTEM}\n\n### Tools\n\n{TOOLS_PREAMBLE}\
                 {WEATHER_JSON}\n</available_tools></system>\n<user>Weather?</user>\n\
                 <assistant></think>"
            )
        );

        assert_eq!(
            v8(
                &[Message::new("user", "Add then report weather")],
                &[weather_tool(), math_tool()],
                None
            ),
            format!(
                "{LAGUNA_BOS}<system>{LAGUNA_DEFAULT_SYSTEM}\n\n### Tools\n\n{TOOLS_PREAMBLE}\
                 {WEATHER_JSON}\n{MATH_JSON}\n</available_tools></system>\n\
                 <user>Add then report weather</user>\n<assistant></think>"
            )
        );

        // assistant_history -- content goes in RAW, whitespace and all.
        assert_eq!(
            v8(
                &[
                    Message::new("user", "Add these."),
                    assistant_with(
                        "\nCalling the tool.\n",
                        "Need addition.",
                        vec![call("add", r#"{"a":2,"b":3}"#)]
                    ),
                    Message::new("tool", "5"),
                    Message::new("user", "Thanks"),
                ],
                &[],
                t()
            ),
            format!(
                "{h}<user>Add these.</user>\n<assistant><think>Need addition.</think>\
                 \nCalling the tool.\n<tool_call>add<arg_key>a</arg_key><arg_value>2</arg_value>\
                 <arg_key>b</arg_key><arg_value>3</arg_value></tool_call></assistant>\n\
                 <tool_response>5</tool_response>\n<user>Thanks</user>\n<assistant><think>"
            )
        );

        // assistant_reasoning_ignored_when_thinking_disabled -- the caller's
        // reasoning is DROPPED, not rendered.
        assert_eq!(
            v8(
                &[
                    Message::new("user", "Explain"),
                    assistant_with("Answer", "Hidden plan.", vec![]),
                    Message::new("user", "Next"),
                ],
                &[],
                None
            ),
            format!(
                "{h}<user>Explain</user>\n<assistant></think>Answer</assistant>\n\
                 <user>Next</user>\n<assistant></think>"
            )
        );

        // assistant_empty_reasoning_when_thinking_enabled -> `<think></think>`.
        assert_eq!(
            v8(
                &[
                    Message::new("user", "Explain"),
                    Message::new("assistant", "Answer"),
                    Message::new("user", "Next"),
                ],
                &[],
                t()
            ),
            format!(
                "{h}<user>Explain</user>\n<assistant><think></think>Answer</assistant>\n\
                 <user>Next</user>\n<assistant><think>"
            )
        );

        // assistant_preserves_content_whitespace -- no trimming at all in v8.
        assert_eq!(
            v8(
                &[
                    Message::new("user", "Explain"),
                    Message::new("assistant", "\nAnswer\n"),
                    Message::new("user", "Next"),
                ],
                &[],
                None
            ),
            format!(
                "{h}<user>Explain</user>\n<assistant></think>\nAnswer\n</assistant>\n\
                 <user>Next</user>\n<assistant></think>"
            )
        );

        assert_eq!(
            v8(
                &[
                    Message::new("user", "Do calls"),
                    assistant_with(
                        "",
                        "",
                        vec![
                            call("echo", r#"{"text":"hello","count":2}"#),
                            call("configure", r#"{"flag":true,"options":{"mode":"fast"}}"#),
                        ]
                    ),
                    Message::new("user", "Done?"),
                ],
                &[],
                None
            ),
            format!(
                "{h}<user>Do calls</user>\n<assistant></think>\
                 <tool_call>echo<arg_key>text</arg_key><arg_value>hello</arg_value>\
                 <arg_key>count</arg_key><arg_value>2</arg_value></tool_call>\
                 <tool_call>configure<arg_key>flag</arg_key><arg_value>true</arg_value>\
                 <arg_key>options</arg_key><arg_value>{{\"mode\": \"fast\"}}</arg_value>\
                 </tool_call></assistant>\n<user>Done?</user>\n<assistant></think>"
            )
        );

        // final_assistant_closes_then_generation_prompt -- v8 has NO prefill
        // path, so a trailing assistant turn is closed and a fresh one opened.
        // Compare with `a_trailing_v2_assistant_turn_is_left_open_as_a_prefill`.
        assert_eq!(
            v8(
                &[
                    Message::new("user", "Complete this"),
                    Message::new("assistant", "Partial"),
                ],
                &[],
                None
            ),
            format!(
                "{h}<user>Complete this</user>\n<assistant></think>Partial</assistant>\n\
                 <assistant></think>"
            )
        );
    }

    /// Upstream `TestLeadingBOSForRenderer` -- both names get the **full-width**
    /// bracket form. Asserted against an escape-coded literal so a copy-paste
    /// that swapped in ASCII `<` `>` fails here rather than in production.
    #[test]
    fn both_laguna_names_lead_with_the_full_width_bos() {
        assert_eq!(LagunaRenderer.leading_bos(), LAGUNA_BOS);
        assert_eq!(LagunaV8Renderer.leading_bos(), LAGUNA_BOS);
        assert_eq!(LAGUNA_BOS, "\u{3008}|EOS|\u{3009}");
    }

    /// The extraction rules, isolated -- easier to reason about than through a
    /// whole prompt.
    #[test]
    fn v2_assistant_content_splits_on_the_last_close_tag() {
        // No close tag -> passthrough.
        assert_eq!(
            v2_assistant_content("plain", ""),
            ("plain".to_string(), String::new())
        );
        // Recovered from the content.
        assert_eq!(
            v2_assistant_content("<think>\nPlan\n</think>\nAnswer", ""),
            ("Answer".to_string(), "Plan".to_string())
        );
        // An explicit reasoning wins, and the content is still split.
        assert_eq!(
            v2_assistant_content("<think>ignored</think>\nAnswer", "given"),
            ("Answer".to_string(), "given".to_string())
        );
        // No opening tag -- everything before the close is the reasoning.
        assert_eq!(
            v2_assistant_content("Plan</think>Answer", ""),
            ("Answer".to_string(), "Plan".to_string())
        );
    }

    /// A string argument goes in bare; anything else is Python-spaced JSON.
    #[test]
    fn laguna_arguments_print_bare_strings_and_spaced_json() {
        assert_eq!(format_tool_call_argument(&json!("Paris")), "Paris");
        assert_eq!(format_tool_call_argument(&json!(2)), "2");
        assert_eq!(format_tool_call_argument(&json!(true)), "true");
        assert_eq!(format_tool_call_argument(&Value::Null), "null");
        assert_eq!(
            format_tool_call_argument(&json!({"mode": "fast"})),
            r#"{"mode": "fast"}"#
        );
    }
}
