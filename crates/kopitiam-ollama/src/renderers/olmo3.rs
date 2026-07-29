//! **Olmo 3 / Olmo 3.1** (Ai2) -- ChatML turns, Python-looking function calls.
//!
//! **Upstream:** `model/renderers/olmo3.go`. Registered as `olmo3` and
//! `olmo3.1`; the only difference between them is the default system message.
//!
//! ## The framing
//!
//! Special tokens: **`<|im_start|>`**, **`<|im_end|>`**, plus the XML-ish
//! **`<functions>...</functions>`** (tool schemas) and
//! **`<function_calls>...</function_calls>`** (calls). No BOS.
//!
//! Two things nobody expects:
//!
//! * **Tool results use the role `environment`**, not `tool` and not `user`:
//!   `<|im_start|>environment\n{...}<|im_end|>`.
//! * **Calls are written as Python-ish source**, not JSON:
//!   `get_weather(location="San Francisco")`. Several calls are separated by a
//!   **newline** inside one `<function_calls>` block. Arguments are
//!   JSON-encoded individually, so a string keeps its quotes here -- the exact
//!   opposite of Qwen3-Coder's raw values.
//!
//! ## The bit that is a real divergence in kind, not degree
//!
//! Upstream **sorts the argument keys** here (`sort.Strings(keys)`) with the
//! comment *"for deterministic output"* -- even though the arguments arrive in
//! insertion order. So `book_flight` renders `from=..., to=...` alphabetically,
//! not in the model's own order. We copy that, because the fixture depends on
//! it, but note it disagrees with how every other renderer treats the same map.
//!
//! ## System messages
//!
//! The **first** system message wins and the rest are dropped. With no system
//! message at all, a default goes in -- `olmo3` gets the short one, `olmo3.1`
//! the long Ai2 identity string -- followed by either the "no functions" or the
//! "here are your functions" sentence. Note both defaults end in a **trailing
//! space**; that space is load-bearing, it is what separates the identity from
//! the functions sentence.

use super::json::{go_value, marshal_tools_with_spaces};
use super::{IM_END_TAG, IM_START_TAG, Message, RenderError, Renderer, ThinkValue, Tool};

/// `olmo3`'s default identity, injected when the caller gives no system message.
///
/// **Upstream:** `model/renderers/olmo3.go:13` (`olmo3DefaultSystemMessage`).
///
/// **The trailing space is real and load-bearing** -- it is in the Go constant,
/// so the rendered prompt has it, so the model was trained with it. Trimming it
/// would look like tidying and would change the token stream.
const OLMO3_DEFAULT_SYSTEM: &str = "You are a helpful function-calling AI assistant. ";
/// `olmo3.1`'s default identity.
///
/// **Upstream:** `model/renderers/olmo3.go:14` (`olmo31DefaultSystemMessage`).
///
/// Trailing space, same reason as above. Note the date cutoff and the weights
/// URL are part of the tuned prompt, not documentation -- do not update them to
/// be "current". Compare `olmo3_think.go:21`, whose otherwise-identical string
/// has **no** trailing space; that difference is upstream's and it is
/// deliberate.
const OLMO31_DEFAULT_SYSTEM: &str = "You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at https://huggingface.co/allenai. ";
const OLMO3_NO_FUNCTIONS: &str = "You do not currently have access to any functions. ";
const OLMO3_WITH_FUNCTIONS: &str = "You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Output any function calls within <function_calls></function_calls> XML tags. Do not make assumptions about what values to plug into functions.";

/// **Upstream:** `Olmo3Renderer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Olmo3Renderer {
    /// `false` -> `olmo3`'s short default system message, `true` -> `olmo3.1`'s
    /// long Ai2 identity one.
    pub use_extended_system_message: bool,
}

impl Renderer for Olmo3Renderer {
    fn render(
        &self,
        messages: &[Message],
        tools: &[Tool],
        _think: Option<&ThinkValue>,
    ) -> Result<String, RenderError> {
        let mut sb = String::new();

        let system_message = messages.iter().find(|m| m.role == "system");
        let filtered: Vec<&Message> = messages.iter().filter(|m| m.role != "system").collect();

        sb.push_str(IM_START_TAG);
        sb.push_str("system\n");
        match system_message {
            Some(system) => {
                sb.push_str(&system.content);
                if !tools.is_empty() {
                    sb.push_str("<functions>");
                    sb.push_str(&marshal_tools_with_spaces(tools));
                    sb.push_str("</functions>");
                }
            }
            None => {
                sb.push_str(if self.use_extended_system_message {
                    OLMO31_DEFAULT_SYSTEM
                } else {
                    OLMO3_DEFAULT_SYSTEM
                });
                if !tools.is_empty() {
                    sb.push_str(OLMO3_WITH_FUNCTIONS);
                    sb.push_str("<functions>");
                    sb.push_str(&marshal_tools_with_spaces(tools));
                    sb.push_str("</functions>");
                } else {
                    sb.push_str(OLMO3_NO_FUNCTIONS);
                    sb.push_str("<functions></functions>");
                }
            }
        }
        sb.push_str(IM_END_TAG);
        sb.push('\n');

        for (i, message) in filtered.iter().enumerate() {
            let last_message = i == filtered.len() - 1;

            match message.role.as_str() {
                "user" => {
                    sb.push_str(IM_START_TAG);
                    sb.push_str("user\n");
                    sb.push_str(&message.content);
                    sb.push_str(IM_END_TAG);
                    sb.push('\n');
                }
                "assistant" => {
                    sb.push_str(IM_START_TAG);
                    sb.push_str("assistant\n");
                    sb.push_str(&message.content);

                    if !message.tool_calls.is_empty() {
                        sb.push_str("<function_calls>");
                        for (j, tc) in message.tool_calls.iter().enumerate() {
                            sb.push_str(&tc.function.name);
                            sb.push('(');
                            // Upstream sorts here -- see the module docs.
                            let mut keys: Vec<&String> = tc.function.arguments.0.keys().collect();
                            keys.sort();
                            for (k, key) in keys.iter().enumerate() {
                                if k > 0 {
                                    sb.push_str(", ");
                                }
                                let val =
                                    tc.function.arguments.get(key).cloned().unwrap_or_default();
                                sb.push_str(&format!("{key}={}", go_value(&val)));
                            }
                            sb.push(')');
                            if j < message.tool_calls.len() - 1 {
                                sb.push('\n');
                            }
                        }
                        sb.push_str("</function_calls>");
                    }

                    // A trailing content-only assistant turn is a prefill and
                    // stays open; anything with tool calls always closes.
                    if !last_message || !message.tool_calls.is_empty() {
                        sb.push_str(IM_END_TAG);
                        sb.push('\n');
                    }
                }
                "tool" => {
                    sb.push_str(IM_START_TAG);
                    sb.push_str("environment\n");
                    sb.push_str(&message.content);
                    sb.push_str(IM_END_TAG);
                    sb.push('\n');
                }
                _ => {}
            }
        }

        let needs_generation_prompt = match filtered.last() {
            Some(last) => {
                !(last.role == "assistant"
                    && last.tool_calls.is_empty()
                    && !last.content.is_empty())
            }
            None => true,
        };
        if needs_generation_prompt {
            sb.push_str(IM_START_TAG);
            sb.push_str("assistant\n");
        }

        Ok(sb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ToolCall, ToolCallFunction};
    use serde_json::json;

    fn call(id: &str, name: &str, raw_args: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            function: ToolCallFunction {
                index: 0,
                name: name.into(),
                arguments: serde_json::from_str(raw_args).expect("valid fixture args"),
            },
        }
    }

    fn weather_tool(description: &str, prop_description: Option<&str>) -> Tool {
        let mut prop = json!({"type": "string"});
        if let Some(d) = prop_description {
            prop["description"] = json!(d);
        }
        let mut function = json!({
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": prop}
            }
        });
        if !description.is_empty() {
            function["description"] = json!(description);
            function["parameters"]["required"] = json!(["location"]);
        }
        serde_json::from_value(json!({"type": "function", "function": function}))
            .expect("valid fixture tool")
    }

    const DEFAULT_SYSTEM_NO_TOOLS: &str = "<|im_start|>system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n";
    const WEATHER_FUNCTIONS: &str = r#"<functions>[{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather", "parameters": {"type": "object", "required": ["location"], "properties": {"location": {"type": "string", "description": "The city"}}}}}]</functions>"#;

    /// Upstream `TestOlmo3Renderer`, all nine cases.
    #[test]
    fn olmo3_matches_the_upstream_fixtures() {
        let r = Olmo3Renderer::default();

        assert_eq!(
            r.render(&[Message::new("user", "Hello!")], &[], None)
                .unwrap(),
            format!(
                "{DEFAULT_SYSTEM_NO_TOOLS}<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
            )
        );

        assert_eq!(
            r.render(
                &[
                    Message::new("system", "You are a helpful assistant."),
                    Message::new("user", "Hello!"),
                ],
                &[],
                None
            )
            .unwrap(),
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
        );

        assert_eq!(
            r.render(
                &[
                    Message::new("system", "You are a helpful assistant."),
                    Message::new("user", "What is the weather?"),
                ],
                &[weather_tool("Get the current weather", Some("The city"))],
                None
            )
            .unwrap(),
            format!(
                "<|im_start|>system\nYou are a helpful assistant.{WEATHER_FUNCTIONS}<|im_end|>\n\
                 <|im_start|>user\nWhat is the weather?<|im_end|>\n<|im_start|>assistant\n"
            )
        );

        assert_eq!(
            r.render(
                &[Message::new("user", "What is the weather?")],
                &[weather_tool("Get the current weather", Some("The city"))],
                None
            )
            .unwrap(),
            format!(
                "<|im_start|>system\n{OLMO3_DEFAULT_SYSTEM}{OLMO3_WITH_FUNCTIONS}{WEATHER_FUNCTIONS}<|im_end|>\n\
                 <|im_start|>user\nWhat is the weather?<|im_end|>\n<|im_start|>assistant\n"
            )
        );

        // Tool call + `environment` reply.
        assert_eq!(
            r.render(
                &[
                    Message::new("system", "You are a helpful assistant."),
                    Message::new("user", "What is the weather in SF?"),
                    Message {
                        role: "assistant".into(),
                        content: "Let me check the weather.".into(),
                        tool_calls: vec![call(
                            "call_1",
                            "get_weather",
                            r#"{"location":"San Francisco"}"#
                        )],
                        ..Default::default()
                    },
                    Message {
                        role: "tool".into(),
                        content: r#"{"temperature": 68}"#.into(),
                        tool_name: "get_weather".into(),
                        ..Default::default()
                    },
                ],
                &[weather_tool("Get the current weather", Some("The city"))],
                None
            )
            .unwrap(),
            format!(
                "<|im_start|>system\nYou are a helpful assistant.{WEATHER_FUNCTIONS}<|im_end|>\n\
                 <|im_start|>user\nWhat is the weather in SF?<|im_end|>\n\
                 <|im_start|>assistant\nLet me check the weather.<function_calls>get_weather(location=\"San Francisco\")</function_calls><|im_end|>\n\
                 <|im_start|>environment\n{{\"temperature\": 68}}<|im_end|>\n\
                 <|im_start|>assistant\n"
            )
        );

        // Multi-turn.
        assert_eq!(
            r.render(
                &[
                    Message::new("system", "You are a helpful assistant."),
                    Message::new("user", "Hello"),
                    Message::new("assistant", "Hi there!"),
                    Message::new("user", "How are you?"),
                ],
                &[],
                None
            )
            .unwrap(),
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\nHi there!<|im_end|>\n\
             <|im_start|>user\nHow are you?<|im_end|>\n\
             <|im_start|>assistant\n"
        );

        // Parallel calls are newline-separated inside ONE block.
        let got = r
            .render(
                &[
                    Message::new("user", "Get weather in SF and NYC"),
                    Message {
                        role: "assistant".into(),
                        tool_calls: vec![
                            call("call_1", "get_weather", r#"{"location":"San Francisco"}"#),
                            call("call_2", "get_weather", r#"{"location":"New York"}"#),
                        ],
                        ..Default::default()
                    },
                    Message::new("tool", r#"{"temperature": 68}"#),
                    Message::new("tool", r#"{"temperature": 55}"#),
                ],
                &[weather_tool("", None)],
                None,
            )
            .unwrap();
        assert!(
            got.contains(
                "<function_calls>get_weather(location=\"San Francisco\")\nget_weather(location=\"New York\")</function_calls><|im_end|>\n"
            ),
            "{got}"
        );
        assert!(
            got.contains("<|im_start|>environment\n{\"temperature\": 68}<|im_end|>\n<|im_start|>environment\n{\"temperature\": 55}<|im_end|>\n"),
            "{got}"
        );

        // Several arguments: sorted, `, `-separated, values JSON-quoted.
        let got = r
            .render(
                &[
                    Message::new("user", "Book a flight"),
                    Message {
                        role: "assistant".into(),
                        tool_calls: vec![call(
                            "call_1",
                            "book_flight",
                            r#"{"from":"SFO","to":"NYC"}"#,
                        )],
                        ..Default::default()
                    },
                ],
                &[],
                None,
            )
            .unwrap();
        assert!(
            got.contains(
                "<function_calls>book_flight(from=\"SFO\", to=\"NYC\")</function_calls><|im_end|>\n"
            ),
            "{got}"
        );

        // A trailing content-only assistant turn is a prefill: no `<|im_end|>`,
        // no generation prompt.
        assert_eq!(
            r.render(
                &[
                    Message::new("user", "Hello"),
                    Message::new("assistant", "Hi there!"),
                ],
                &[],
                None
            )
            .unwrap(),
            format!(
                "{DEFAULT_SYSTEM_NO_TOOLS}<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there!"
            )
        );
    }

    /// `olmo3.1` differs from `olmo3` only in the default identity string.
    #[test]
    fn the_extended_variant_uses_the_ai2_identity_default() {
        let got = Olmo3Renderer {
            use_extended_system_message: true,
        }
        .render(&[Message::new("user", "Hello!")], &[], None)
        .unwrap();
        assert!(got.starts_with(&format!(
            "<|im_start|>system\n{OLMO31_DEFAULT_SYSTEM}{OLMO3_NO_FUNCTIONS}<functions></functions><|im_end|>\n"
        )), "{got}");
    }
}
