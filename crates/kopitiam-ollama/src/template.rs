//! Prompt building -- turning a conversation into the exact string a model sees.
//!
//! **Upstream:** `template/template.go` (ollama, MIT). The Go template *engine*
//! lives in [`crate::gotmpl`]; this module is the ollama-specific layer on top.
//!
//! ## The one job
//!
//! A model does not see "messages". It sees **one string**. Which string depends
//! entirely on how the model was fine-tuned, and getting it wrong does not throw
//! an error -- it just makes the model worse, in ways that look like the model
//! being dumb rather than the harness being wrong. That is what makes this
//! module worth porting exactly rather than approximating.
//!
//! ## Three execution modes, and how one is chosen
//!
//! Upstream's `Execute` branches three ways, and the branch is picked from the
//! template's own text, not from a config flag:
//!
//! 1. **Fill-in-the-middle** -- when the caller supplied both `prompt` and
//!    `suffix`. Used by code-completion models; the template gets `Prompt` and
//!    `Suffix` and nothing else.
//! 2. **Message-aware** -- when the template mentions `messages` anywhere (asked
//!    via [`crate::gotmpl::Template::vars`]). The whole conversation goes in
//!    once, and the template decides its own framing per turn. Every modern
//!    template is this kind.
//! 3. **Legacy** -- everything else. The template only understands
//!    `{{ .System }}{{ .Prompt }}{{ .Response }}`, one exchange at a time, so
//!    upstream **executes it repeatedly**, once per user/assistant pair, and
//!    concatenates. This is how a 2023-era Alpaca template still renders a
//!    twenty-turn conversation.
//!
//! ## Two tricks that look like bugs until you see why
//!
//! **A `{{ .Response }}` is appended to templates that lack one.** At parse time
//! ([`Template::parse`]), if the source mentions neither `messages` nor
//! `response`, upstream grafts a `{{ .Response }}` node onto the end of the
//! tree. Without it, a legacy template has no marker saying "the model's turn
//! starts here", and the truncation below would have nothing to cut at.
//!
//! **The final turn is rendered from a truncated tree.** In legacy mode the last
//! execution uses a copy of the template with everything *after* `{{ .Response }}`
//! deleted ([`truncate_after_response`]). That is what makes the prompt stop
//! precisely where the assistant should begin generating, instead of trailing an
//! end-of-turn marker the model would then have to talk past.
//!
//! ## Deliberate divergences
//!
//! * `Subtree` and `Named` are not ported. `Named` picks a bundled template by
//!   **Levenshtein distance** against 20-odd embedded `.gotmpl` files, which
//!   would mean vendoring those files and a string-distance dependency; KOPITIAM
//!   reads the template out of the GGUF instead. See the port epic bead.
//! * Images are not carried on [`Message`] yet -- upstream's `collate` inserts
//!   `[img-N]` tags, and it carries its own `todo(parthsareen)` about revisiting
//!   that. Ported when the vision path needs it.

use crate::gotmpl::{self, Env, ExecError, ParseError, Value};
use std::collections::BTreeMap;

/// One conversation turn. **Upstream:** `api.Message` reduced to the fields a
/// template can actually reach (`templateMessage` in `template.go`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Message {
    /// `system`, `user`, `assistant`, or `tool`.
    pub role: String,
    pub content: String,
    /// A reasoning model's hidden scratchpad, when the model exposes one.
    pub thinking: String,
    /// Tool calls the assistant emitted, as raw JSON objects.
    pub tool_calls: Vec<serde_json::Value>,
    pub tool_name: String,
    pub tool_call_id: String,
}

impl Message {
    /// Convenience constructor for the common two-field case.
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
            ..Default::default()
        }
    }

    fn to_value(&self) -> Value {
        let mut m = BTreeMap::new();
        m.insert("Role".into(), Value::Str(self.role.clone()));
        m.insert("Content".into(), Value::Str(self.content.clone()));
        m.insert("Thinking".into(), Value::Str(self.thinking.clone()));
        m.insert("ToolName".into(), Value::Str(self.tool_name.clone()));
        m.insert("ToolCallID".into(), Value::Str(self.tool_call_id.clone()));
        m.insert(
            "ToolCalls".into(),
            Value::List(self.tool_calls.iter().map(Value::from_json).collect()),
        );
        Value::Map(m)
    }
}

/// Everything a template render needs. **Upstream:** `type Values struct`.
#[derive(Debug, Clone, Default)]
pub struct Values {
    pub messages: Vec<Message>,
    /// Tool schemas, as raw JSON. Kept as JSON rather than typed because the
    /// `api.Tool` hierarchy is not ported yet -- templates only ever `range`
    /// over these and read `.Function.Name` / `.Description` / `.Parameters`,
    /// which JSON maps serve exactly.
    pub tools: Vec<serde_json::Value>,
    pub prompt: String,
    pub suffix: String,
    pub think: bool,
    /// `"high"` / `"medium"` / `"low"` / `"max"` when the caller asked for a
    /// level rather than a plain on/off.
    pub think_level: String,
    /// Whether the caller **explicitly** set thinking, as opposed to it merely
    /// defaulting off. Upstream's own comment: *"Templates can't see whether
    /// `Think` is nil"* -- so this flag is the seam that lets a template tell
    /// "thinking off" from "thinking not mentioned".
    pub is_think_set: bool,
    /// Force mode 3 even when the template is message-aware. **Upstream:**
    /// `forceLegacy`, an unexported test-only flag. Public here because it is
    /// the only way to test the legacy path against a modern template, and
    /// hiding it would just mean duplicating templates in tests.
    pub force_legacy: bool,
}

/// A parsed prompt template.
#[derive(Debug, Clone)]
pub struct Template {
    inner: gotmpl::Template,
    vars: Vec<String>,
}

impl Template {
    /// Parse a template, grafting on a `{{ .Response }}` if it has neither
    /// `messages` nor `response`.
    ///
    /// **Upstream:** `Parse(s string)`.
    ///
    /// The graft is not cosmetic -- see the module docs. A legacy template with
    /// no response marker would render a prompt that never stops at the
    /// assistant's turn.
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let inner = gotmpl::Template::parse(s)?;
        let vars = inner.vars();

        let inner = if !vars.iter().any(|v| v == "messages" || v == "response") {
            let mut nodes = inner.nodes().clone();
            nodes.push(gotmpl::parse::Node::Action(gotmpl::parse::Pipeline {
                cmds: vec![gotmpl::parse::Command {
                    args: vec![gotmpl::parse::Arg::Field(vec!["Response".to_string()])],
                }],
            }));
            gotmpl::Template::from_nodes(nodes, inner.raw().to_string())
        } else {
            inner
        };

        // Recompute AFTER the graft so `vars` reflects the tree we will run.
        let vars = inner.vars();
        Ok(Self { inner, vars })
    }

    /// **Upstream:** `DefaultTemplate` -- `{{ .Prompt }}`, which then picks up a
    /// grafted `{{ .Response }}`. The fallback for a model whose GGUF carries no
    /// chat template at all.
    pub fn default_template() -> Self {
        Self::parse("{{ .Prompt }}").expect("the default template always parses")
    }

    /// The original source. **Upstream:** `(*Template).String()`.
    pub fn raw(&self) -> &str {
        self.inner.raw()
    }

    /// **Upstream:** `(*Template).Contains(s)`.
    pub fn contains(&self, s: &str) -> bool {
        self.inner.contains(s)
    }

    /// **Upstream:** `(*Template).Vars()`.
    pub fn vars(&self) -> &[String] {
        &self.vars
    }

    /// Does this template take the whole conversation at once?
    ///
    /// The mode-2-vs-mode-3 decision, named rather than left as an inline
    /// `slices.Contains` -- it is the single most consequential branch here.
    pub fn is_message_aware(&self) -> bool {
        self.vars.iter().any(|v| v == "messages")
    }

    /// Render a conversation into the string the model will see.
    ///
    /// **Upstream:** `(*Template).Execute(w io.Writer, v Values)`.
    pub fn execute(&self, v: &Values, env: &Env) -> Result<String, ExecError> {
        let (system, messages) = collate(&v.messages);

        // ---- Mode 1: fill-in-the-middle ----
        if !v.prompt.is_empty() && !v.suffix.is_empty() {
            let mut m = BTreeMap::new();
            m.insert("Prompt".into(), Value::Str(v.prompt.clone()));
            m.insert("Suffix".into(), Value::Str(v.suffix.clone()));
            m.insert("Response".into(), Value::Str(String::new()));
            insert_think(&mut m, v);
            return self.inner.execute(&Value::Map(m), env);
        }

        // ---- Mode 2: message-aware ----
        if !v.force_legacy && self.is_message_aware() {
            let mut m = BTreeMap::new();
            m.insert("System".into(), Value::Str(system));
            m.insert(
                "Messages".into(),
                Value::List(messages.iter().map(Message::to_value).collect()),
            );
            m.insert(
                "Tools".into(),
                if v.tools.is_empty() {
                    // Upstream returns a nil slice when there are no tools, and
                    // Go's template truth makes nil falsy -- `{{ if .Tools }}`
                    // must not fire. An empty list is falsy here too, so the
                    // observable behaviour matches.
                    Value::List(Vec::new())
                } else {
                    Value::List(v.tools.iter().map(Value::from_json).collect())
                },
            );
            m.insert("Response".into(), Value::Str(String::new()));
            insert_think(&mut m, v);
            return self.inner.execute(&Value::Map(m), env);
        }

        // ---- Mode 3: legacy, one execution per exchange ----
        //
        // Walk the turns accumulating (system, prompt, response). A turn is
        // flushed -- i.e. the whole template runs again and appends -- whenever
        // the next message would overwrite something already pending. So a
        // user->assistant->user->assistant conversation produces two full
        // renders plus the truncated tail below.
        let mut out = String::new();
        let mut system = String::new();
        let mut prompt = String::new();
        let mut response = String::new();

        for msg in &messages {
            match msg.role.as_str() {
                "system" => {
                    if !prompt.is_empty() || !response.is_empty() {
                        out.push_str(&self.exec_legacy_turn(&system, &prompt, &response, v, env)?);
                        system.clear();
                        prompt.clear();
                        response.clear();
                    }
                    system = msg.content.clone();
                }
                "user" => {
                    if !response.is_empty() {
                        out.push_str(&self.exec_legacy_turn(&system, &prompt, &response, v, env)?);
                        system.clear();
                        prompt.clear();
                        response.clear();
                    }
                    prompt = msg.content.clone();
                }
                "assistant" => response = msg.content.clone(),
                // Upstream's switch has no other arms -- a `tool` message is
                // silently skipped in legacy mode, because a legacy template has
                // nowhere to put one.
                _ => {}
            }
        }

        // The tail: render what is still pending through a tree truncated at
        // `{{ .Response }}`, so the prompt ends exactly where generation starts.
        let truncated = gotmpl::Template::from_nodes(
            truncate_after_response(self.inner.nodes()),
            self.inner.raw().to_string(),
        );
        let mut m = BTreeMap::new();
        m.insert("System".into(), Value::Str(system));
        m.insert("Prompt".into(), Value::Str(prompt));
        m.insert("Response".into(), Value::Str(response));
        insert_think(&mut m, v);
        out.push_str(&truncated.execute(&Value::Map(m), env)?);

        Ok(out)
    }

    fn exec_legacy_turn(
        &self,
        system: &str,
        prompt: &str,
        response: &str,
        v: &Values,
        env: &Env,
    ) -> Result<String, ExecError> {
        let mut m = BTreeMap::new();
        m.insert("System".into(), Value::Str(system.to_string()));
        m.insert("Prompt".into(), Value::Str(prompt.to_string()));
        m.insert("Response".into(), Value::Str(response.to_string()));
        insert_think(&mut m, v);
        self.inner.execute(&Value::Map(m), env)
    }
}

fn insert_think(m: &mut BTreeMap<String, Value>, v: &Values) {
    m.insert("Think".into(), Value::Bool(v.think));
    m.insert("ThinkLevel".into(), Value::Str(v.think_level.clone()));
    m.insert("IsThinkSet".into(), Value::Bool(v.is_think_set));
}

/// Merge consecutive same-role messages and hoist every system message out.
///
/// **Upstream:** `collate(msgs []api.Message)`.
///
/// Two behaviours worth stating precisely, because both are load-bearing:
///
/// * **Consecutive same-role messages merge**, joined by `"\n\n"`. Models are
///   trained on strictly alternating turns; two user messages in a row is a
///   shape they never saw, and framing them as two turns degrades the reply.
///   `tool` messages are exempt -- each carries its own call id and name, so
///   merging them would destroy the association with its call.
/// * **System messages are collected AND left in place.** They go into the
///   returned `system` string (joined by `"\n\n"`) *and* stay in the message
///   list. That looks like double-counting; it is not. Mode 2 templates read
///   `.System` and skip system entries while ranging; mode 3 reads only
///   `.System`. Removing them from the list would break any template that
///   chooses to render them itself.
fn collate(msgs: &[Message]) -> (String, Vec<Message>) {
    let mut system: Vec<&str> = Vec::new();
    let mut collated: Vec<Message> = Vec::new();

    for m in msgs {
        if m.role == "system" {
            system.push(&m.content);
        }

        match collated.last_mut() {
            Some(last) if last.role == m.role && m.role != "tool" => {
                last.content.push_str("\n\n");
                last.content.push_str(&m.content);
            }
            _ => collated.push(m.clone()),
        }
    }

    (system.join("\n\n"), collated)
}

/// Drop every node that comes **after** the first `{{ .Response }}`, keeping the
/// response node itself.
///
/// **Upstream:** `deleteNode(root.Copy(), fn)` with the `cut` closure in
/// `Execute`. Upstream's predicate returns `false` for the `Response` field node
/// (so it survives) while flipping `cut`, after which every subsequently visited
/// node is deleted. Same shape here, in document order.
///
/// This is what stops a rendered prompt from trailing an end-of-turn marker the
/// model would then have to talk past.
fn truncate_after_response(nodes: &gotmpl::parse::Nodes) -> gotmpl::parse::Nodes {
    let mut cut = false;
    truncate_list(nodes, &mut cut)
}

fn truncate_list(nodes: &gotmpl::parse::Nodes, cut: &mut bool) -> gotmpl::parse::Nodes {
    use gotmpl::parse::Node;
    let mut out = Vec::new();

    for n in nodes {
        if *cut {
            break;
        }
        match n {
            Node::Action(p) => {
                let mentions = pipe_mentions_response(p);
                out.push(n.clone());
                if mentions {
                    *cut = true;
                }
            }
            Node::If {
                pipe,
                then,
                otherwise,
            } => {
                let t = truncate_list(then, cut);
                let o = if *cut {
                    Vec::new()
                } else {
                    truncate_list(otherwise, cut)
                };
                out.push(Node::If {
                    pipe: pipe.clone(),
                    then: t,
                    otherwise: o,
                });
            }
            Node::With {
                pipe,
                body,
                otherwise,
            } => {
                let b = truncate_list(body, cut);
                let o = if *cut {
                    Vec::new()
                } else {
                    truncate_list(otherwise, cut)
                };
                out.push(Node::With {
                    pipe: pipe.clone(),
                    body: b,
                    otherwise: o,
                });
            }
            Node::Range {
                key,
                val,
                pipe,
                body,
                otherwise,
            } => {
                let b = truncate_list(body, cut);
                let o = if *cut {
                    Vec::new()
                } else {
                    truncate_list(otherwise, cut)
                };
                out.push(Node::Range {
                    key: key.clone(),
                    val: val.clone(),
                    pipe: pipe.clone(),
                    body: b,
                    otherwise: o,
                });
            }
            other => out.push(other.clone()),
        }
    }

    out
}

fn pipe_mentions_response(p: &gotmpl::parse::Pipeline) -> bool {
    use gotmpl::parse::Arg;
    p.cmds.iter().any(|c| {
        c.args.iter().any(|a| match a {
            Arg::Field(path) | Arg::Var(_, path) => path.iter().any(|s| s == "Response"),
            Arg::Paren(inner) => pipe_mentions_response(inner),
            _ => false,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(pairs: &[(&str, &str)]) -> Vec<Message> {
        pairs.iter().map(|(r, c)| Message::new(r, c)).collect()
    }

    fn render(src: &str, v: &Values) -> String {
        Template::parse(src)
            .unwrap_or_else(|e| panic!("parse: {e}"))
            .execute(v, &Env::default())
            .unwrap_or_else(|e| panic!("exec: {e}"))
    }

    // ---- collate ----

    #[test]
    fn collate_merges_consecutive_same_role_messages() {
        let (_, out) = collate(&msgs(&[("user", "a"), ("user", "b"), ("assistant", "c")]));
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].content, "a\n\nb");
        assert_eq!(out[1].content, "c");
    }

    /// Tool messages are exempt: each carries its own call id, and merging would
    /// destroy the association with the call it answers.
    #[test]
    fn collate_does_not_merge_tool_messages() {
        let mut a = Message::new("tool", "1");
        a.tool_call_id = "call_a".into();
        let mut b = Message::new("tool", "2");
        b.tool_call_id = "call_b".into();
        let (_, out) = collate(&[a, b]);
        assert_eq!(out.len(), 2, "tool messages must stay separate");
    }

    #[test]
    fn collate_hoists_system_messages_but_leaves_them_in_place() {
        let (system, out) = collate(&msgs(&[("system", "s1"), ("user", "u")]));
        assert_eq!(system, "s1");
        assert_eq!(out.len(), 2, "system stays in the list too");
        assert_eq!(out[0].role, "system");
    }

    #[test]
    fn collate_joins_multiple_system_messages() {
        let (system, _) = collate(&msgs(&[("system", "a"), ("user", "u"), ("system", "b")]));
        assert_eq!(system, "a\n\nb");
    }

    // ---- mode selection ----

    #[test]
    fn a_template_mentioning_messages_is_message_aware() {
        assert!(Template::parse("{{ range .Messages }}{{ .Content }}{{ end }}")
            .unwrap()
            .is_message_aware());
        assert!(!Template::parse("{{ .System }}{{ .Prompt }}")
            .unwrap()
            .is_message_aware());
    }

    /// The graft: a template with neither `messages` nor `response` gets a
    /// `{{ .Response }}` appended, or legacy truncation has nothing to cut at.
    #[test]
    fn a_response_node_is_grafted_onto_templates_that_lack_one() {
        let t = Template::parse("{{ .Prompt }}").unwrap();
        assert!(t.vars().contains(&"response".to_string()));
        // ...and NOT onto ones that already have it.
        let t2 = Template::parse("{{ .Prompt }}{{ .Response }}").unwrap();
        assert_eq!(
            t2.vars().iter().filter(|v| *v == "response").count(),
            1
        );
        // ...nor onto message-aware ones, which manage their own turn framing.
        let t3 = Template::parse("{{ range .Messages }}{{ .Content }}{{ end }}").unwrap();
        assert!(!t3.vars().contains(&"response".to_string()));
    }

    // ---- mode 1: fill in the middle ----

    #[test]
    fn prompt_and_suffix_select_the_fim_branch() {
        let v = Values {
            prompt: "def f(".into(),
            suffix: "return x".into(),
            ..Default::default()
        };
        assert_eq!(
            render("<PRE>{{ .Prompt }}<SUF>{{ .Suffix }}<MID>", &v),
            "<PRE>def f(<SUF>return x<MID>"
        );
    }

    // ---- mode 2: message-aware ----

    #[test]
    fn a_message_aware_template_renders_the_whole_conversation_once() {
        let src = concat!(
            "{{- range .Messages }}<|im_start|>{{ .Role }}\n",
            "{{ .Content }}<|im_end|>\n",
            "{{ end }}<|im_start|>assistant\n"
        );
        let v = Values {
            messages: msgs(&[("system", "be terse"), ("user", "hi")]),
            ..Default::default()
        };
        assert_eq!(
            render(src, &v),
            "<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn a_message_aware_template_can_read_the_hoisted_system_prompt() {
        let src = "[{{ .System }}]{{ range .Messages }}{{ if ne .Role \"system\" }}({{ .Content }}){{ end }}{{ end }}";
        let v = Values {
            messages: msgs(&[("system", "S"), ("user", "u")]),
            ..Default::default()
        };
        assert_eq!(render(src, &v), "[S](u)");
    }

    /// No tools must be FALSY, so `{{ if .Tools }}` does not fire and a
    /// tool-less request never gets a tools preamble it did not ask for.
    #[test]
    fn an_empty_tool_list_is_falsy() {
        let src = "{{ range .Messages }}{{ end }}{{ if .Tools }}TOOLS{{ else }}none{{ end }}";
        let v = Values {
            messages: msgs(&[("user", "hi")]),
            ..Default::default()
        };
        assert_eq!(render(src, &v), "none");
    }

    #[test]
    fn tools_render_from_their_json() {
        let src = "{{ range .Messages }}{{ end }}{{ range .Tools }}{{ .function.name }};{{ end }}";
        let v = Values {
            messages: msgs(&[("user", "hi")]),
            tools: vec![serde_json::json!({"function": {"name": "search"}})],
            ..Default::default()
        };
        assert_eq!(render(src, &v), "search;");
    }

    // ---- mode 3: legacy ----

    /// The heart of legacy mode: the template runs once per exchange and the
    /// results concatenate. A single render would drop all but the last turn.
    #[test]
    fn a_legacy_template_is_executed_once_per_exchange() {
        let src = "[INST] {{ .Prompt }} [/INST] {{ .Response }}";
        let v = Values {
            messages: msgs(&[
                ("user", "q1"),
                ("assistant", "a1"),
                ("user", "q2"),
                ("assistant", "a2"),
                ("user", "q3"),
            ]),
            ..Default::default()
        };
        // Two completed exchanges render in full; the pending third is
        // truncated at `{{ .Response }}`.
        assert_eq!(
            render(src, &v),
            "[INST] q1 [/INST] a1[INST] q2 [/INST] a2[INST] q3 [/INST] "
        );
    }

    /// The truncation: everything after `{{ .Response }}` is dropped on the
    /// final render, so the prompt stops where generation begins instead of
    /// trailing a turn terminator.
    #[test]
    fn the_final_turn_is_truncated_at_the_response_marker() {
        let src = "<user>{{ .Prompt }}</user><bot>{{ .Response }}</bot>";
        let v = Values {
            messages: msgs(&[("user", "q1"), ("assistant", "a1"), ("user", "q2")]),
            ..Default::default()
        };
        let out = render(src, &v);
        assert_eq!(out, "<user>q1</user><bot>a1</bot><user>q2</user><bot>");
        assert!(
            !out.ends_with("</bot>") || out.matches("</bot>").count() == 1,
            "the trailing </bot> must be cut from the final turn"
        );
    }

    /// A grafted `{{ .Response }}` must also be a valid cut point -- this is why
    /// the graft exists at all.
    #[test]
    fn a_grafted_response_still_gives_the_truncation_a_cut_point() {
        let v = Values {
            messages: msgs(&[("user", "q1"), ("assistant", "a1"), ("user", "q2")]),
            ..Default::default()
        };
        assert_eq!(render("### {{ .Prompt }}\n", &v), "### q1\na1### q2\n");
    }

    #[test]
    fn a_legacy_template_sees_the_system_prompt_on_the_turn_it_belongs_to() {
        let src = "{{ if .System }}<sys>{{ .System }}</sys>{{ end }}<u>{{ .Prompt }}</u>{{ .Response }}";
        let v = Values {
            messages: msgs(&[("system", "S"), ("user", "q1")]),
            ..Default::default()
        };
        assert_eq!(render(src, &v), "<sys>S</sys><u>q1</u>");
    }

    /// `force_legacy` must override the message-aware branch -- upstream's own
    /// test hook, and the only way to exercise mode 3 against a modern template.
    #[test]
    fn force_legacy_overrides_a_message_aware_template() {
        let src = "{{ if .Messages }}M{{ else }}L:{{ .Prompt }}{{ .Response }}{{ end }}";
        let base = msgs(&[("user", "q")]);

        let auto = Values {
            messages: base.clone(),
            ..Default::default()
        };
        assert_eq!(render(src, &auto), "M");

        let forced = Values {
            messages: base,
            force_legacy: true,
            ..Default::default()
        };
        assert_eq!(render(src, &forced), "L:q");
    }

    // ---- thinking ----

    #[test]
    fn thinking_flags_reach_the_template() {
        let src = "{{ range .Messages }}{{ end }}{{ if .IsThinkSet }}set:{{ .Think }}/{{ .ThinkLevel }}{{ else }}unset{{ end }}";
        let unset = Values {
            messages: msgs(&[("user", "q")]),
            ..Default::default()
        };
        assert_eq!(render(src, &unset), "unset");

        let set = Values {
            messages: msgs(&[("user", "q")]),
            think: true,
            think_level: "high".into(),
            is_think_set: true,
            ..Default::default()
        };
        assert_eq!(render(src, &set), "set:true/high");
    }

    // ---- default template ----

    #[test]
    fn the_default_template_is_prompt_plus_a_grafted_response() {
        let t = Template::default_template();
        assert_eq!(t.raw(), "{{ .Prompt }}");
        let v = Values {
            messages: msgs(&[("user", "hello")]),
            ..Default::default()
        };
        assert_eq!(t.execute(&v, &Env::default()).unwrap(), "hello");
    }

    // ---- truncation unit ----

    #[test]
    fn truncation_keeps_the_response_node_and_drops_what_follows() {
        let t = gotmpl::Template::parse("a{{ .Response }}b{{ .X }}").unwrap();
        let out = truncate_after_response(t.nodes());
        assert_eq!(out.len(), 2, "text 'a' and the Response action survive");
    }

    #[test]
    fn truncation_reaches_inside_a_conditional() {
        let t = gotmpl::Template::parse("{{ if .A }}x{{ .Response }}y{{ end }}z").unwrap();
        let out = truncate_after_response(t.nodes());
        assert_eq!(out.len(), 1, "the trailing 'z' is cut");
        let gotmpl::parse::Node::If { then, .. } = &out[0] else {
            panic!("expected an If")
        };
        assert_eq!(then.len(), 2, "'y' inside the branch is cut too");
    }

    /// A realistic full render, asserted end to end -- this is what a Qwen-class
    /// model actually receives, and what KOPITIAM currently gets wrong for every
    /// non-ChatML family (bead bd-250.3).
    #[test]
    fn a_full_chatml_conversation_renders_exactly() {
        let src = concat!(
            "{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}",
            "{{- range .Messages }}",
            "{{- if ne .Role \"system\" }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}",
            "{{- end }}<|im_start|>assistant\n"
        );
        let v = Values {
            messages: msgs(&[
                ("system", "You are terse."),
                ("user", "2+2?"),
                ("assistant", "4"),
                ("user", "and 3+3?"),
            ]),
            ..Default::default()
        };
        assert_eq!(
            render(src, &v),
            concat!(
                "<|im_start|>system\nYou are terse.<|im_end|>\n",
                "<|im_start|>user\n2+2?<|im_end|>\n",
                "<|im_start|>assistant\n4<|im_end|>\n",
                "<|im_start|>user\nand 3+3?<|im_end|>\n",
                "<|im_start|>assistant\n"
            )
        );
    }
}
