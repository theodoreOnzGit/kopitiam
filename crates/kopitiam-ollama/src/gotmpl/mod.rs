//! A Go `text/template` subset -- the engine chat templates are written in.
//!
//! **Upstream:** Go's `text/template` + `text/template/parse` (BSD-3-Clause,
//! Copyright (c) 2009 The Go Authors), and the function set ollama layers on top
//! in `template/template.go` (MIT, Copyright (c) Ollama).
//!
//! ## Why KOPITIAM needs a Go template engine, of all things
//!
//! Because that is the language chat templates are already written in. Every
//! GGUF that ships a `tokenizer.chat_template`, every ollama Modelfile
//! `TEMPLATE` block, every `.gotmpl` in ollama's library -- all Go templates.
//! Not liking the choice is beside the point: the templates exist, models were
//! tuned against the prompts they produce, and a model fed the wrong framing
//! quietly gets worse rather than loudly breaking.
//!
//! KOPITIAM's current state is exactly that failure. `kopitiam-ai` hardcodes
//! ChatML for every model (bead `bd-250.3`), so a Gemma gets `<|im_start|>`
//! where it was trained on `<start_of_turn>`, and a Llama 3 gets neither of the
//! header tokens it expects. This module is what lets each model be framed the
//! way it was actually trained.
//!
//! ## Why a reimplementation and not a port
//!
//! Go's `text/template` is ~7000 lines built on reflection over arbitrary Go
//! types -- most of which exists to bridge a static type system into a dynamic
//! template language. KOPITIAM's data is [`Value`], a small closed enum, so all
//! of that machinery evaporates. What remains is the *language semantics*, and
//! those ARE ported faithfully: Go's truth rules, Go's sorted map iteration,
//! Go's `and`/`or` returning values rather than bools, Go's `else if` nesting,
//! Go's trim markers, `missingkey=zero`. Each is called out where it is
//! implemented, because each is a place where "what Rust would do" is wrong.
//!
//! ## What is supported, and what is not
//!
//! Supported: text, `{{ pipeline }}`, `{{ if }}/{{ else if }}/{{ else }}/{{ end }}`,
//! `{{ range }}` (with `$k, $v :=`, `{{ else }}`, `continue`, `break`),
//! `{{ with }}`, variables (`:=` and `=`), field chains, `$` root access, pipes,
//! parenthesised sub-pipelines, string/number/bool/nil literals, `{{- -}}` trim
//! markers, `{{/* comments */}}`.
//!
//! Builtins: `and or not eq ne lt le gt ge len index slice print printf println`,
//! plus ollama's `json currentDate yesterdayDate toTypeScriptType`.
//!
//! **Not supported, and rejected loudly rather than mis-rendered:**
//! `{{ template }}`, `{{ define }}`, `{{ block }}` -- multi-template composition,
//! which no chat template uses. Also absent: method calls on values, and the
//! remaining `fmt` verbs. If a real model's template needs one, that is a bug to
//! fix here, not a reason to fall back to hardcoded ChatML.
//!
//! ## Quick tour
//!
//! ```
//! use kopitiam_ollama::gotmpl::{Template, Value, Env};
//! use std::collections::BTreeMap;
//!
//! let t = Template::parse("{{ if .System }}sys: {{ .System }}\n{{ end }}user: {{ .Prompt }}")?;
//!
//! let mut data = BTreeMap::new();
//! data.insert("System".to_string(), Value::from("be terse"));
//! data.insert("Prompt".to_string(), Value::from("hello"));
//!
//! assert_eq!(t.execute(&Value::Map(data), &Env::default())?, "sys: be terse\nuser: hello");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod exec;
pub mod parse;

pub use exec::{Env, ExecError, Value};
pub use parse::ParseError;

/// A parsed template, ready to execute many times.
#[derive(Debug, Clone, PartialEq)]
pub struct Template {
    nodes: parse::Nodes,
    raw: String,
}

impl Template {
    /// Parse template source.
    pub fn parse(src: &str) -> Result<Self, ParseError> {
        Ok(Self {
            nodes: parse::parse(src)?,
            raw: src.to_string(),
        })
    }

    /// Render against `data`, which becomes both `.` and `$`.
    pub fn execute(&self, data: &Value, env: &Env) -> Result<String, ExecError> {
        exec::execute(&self.nodes, data, env)
    }

    /// The original source. **Upstream:** `(*Template).String()`.
    pub fn raw(&self) -> &str {
        &self.raw
    }

    /// Does the source contain `s`? **Upstream:** `(*Template).Contains(s)`.
    ///
    /// A crude substring test upstream, and deliberately kept crude: it is used
    /// to sniff for markers like `<|im_start|>` in a raw template, where a
    /// structural search would be more precise but also more likely to disagree
    /// with ollama about whether a given template "has" something.
    pub fn contains(&self, s: &str) -> bool {
        self.raw.contains(s)
    }

    /// Every identifier the template references, lowercased, sorted, deduped.
    ///
    /// **Upstream:** `(*Template).Vars()`.
    ///
    /// This drives the single most important decision in the whole template
    /// layer: **whether a template is message-aware**. A template mentioning
    /// `messages` gets the entire conversation in one execution; one that does
    /// not gets executed once per turn in "legacy" mode. See
    /// [`crate::template`] for what hangs off that.
    pub fn vars(&self) -> Vec<String> {
        let mut set = std::collections::BTreeSet::new();
        collect_idents(&self.nodes, &mut set);
        set.into_iter().collect()
    }

    /// The parsed nodes -- for the ollama layer, which rewrites trees.
    pub(crate) fn nodes(&self) -> &parse::Nodes {
        &self.nodes
    }

    /// Build a template from nodes that were rewritten, not parsed.
    pub(crate) fn from_nodes(nodes: parse::Nodes, raw: String) -> Self {
        Self { nodes, raw }
    }
}

/// **Upstream:** `Identifiers(n parse.Node)` -- walk the tree collecting every
/// field and variable name. Lowercased into a set by `Vars()`.
fn collect_idents(nodes: &[parse::Node], out: &mut std::collections::BTreeSet<String>) {
    use parse::Node;
    for n in nodes {
        match n {
            Node::Text(_) | Node::Continue | Node::Break => {}
            Node::Action(p) => collect_pipe(p, out),
            Node::Assign { pipe, .. } => collect_pipe(pipe, out),
            Node::If {
                pipe,
                then,
                otherwise,
            } => {
                collect_pipe(pipe, out);
                collect_idents(then, out);
                collect_idents(otherwise, out);
            }
            Node::With {
                pipe,
                body,
                otherwise,
            }
            | Node::Range {
                pipe,
                body,
                otherwise,
                ..
            } => {
                collect_pipe(pipe, out);
                collect_idents(body, out);
                collect_idents(otherwise, out);
            }
        }
    }
}

fn collect_pipe(p: &parse::Pipeline, out: &mut std::collections::BTreeSet<String>) {
    use parse::Arg;
    for cmd in &p.cmds {
        for a in &cmd.args {
            match a {
                // Upstream returns `n.Ident` for a FieldNode -- the whole chain,
                // so `.Function.Name` contributes both "function" and "name".
                Arg::Field(path) => out.extend(path.iter().map(|s| s.to_lowercase())),
                // And for a VariableNode likewise, INCLUDING the `$name` itself.
                Arg::Var(name, path) => {
                    out.insert(name.to_lowercase());
                    out.extend(path.iter().map(|s| s.to_lowercase()));
                }
                Arg::Paren(inner) => collect_pipe(inner, out),
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn map(pairs: &[(&str, Value)]) -> Value {
        Value::Map(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect::<BTreeMap<_, _>>(),
        )
    }

    fn render(src: &str, data: &Value) -> String {
        Template::parse(src)
            .unwrap_or_else(|e| panic!("parse {src:?}: {e}"))
            .execute(data, &Env::default())
            .unwrap_or_else(|e| panic!("exec {src:?}: {e}"))
    }

    #[test]
    fn a_field_renders() {
        assert_eq!(render("{{ .A }}", &map(&[("A", "x".into())])), "x");
    }

    #[test]
    fn if_picks_the_branch_by_go_truth() {
        let t = "{{ if .A }}yes{{ else }}no{{ end }}";
        assert_eq!(render(t, &map(&[("A", "x".into())])), "yes");
        assert_eq!(render(t, &map(&[("A", "".into())])), "no", "empty string is false");
        assert_eq!(render(t, &map(&[])), "no", "a missing key is false");
    }

    #[test]
    fn else_if_chains_pick_the_first_match() {
        let t = "{{ if eq .R \"system\" }}S{{ else if eq .R \"user\" }}U{{ else }}A{{ end }}";
        assert_eq!(render(t, &map(&[("R", "system".into())])), "S");
        assert_eq!(render(t, &map(&[("R", "user".into())])), "U");
        assert_eq!(render(t, &map(&[("R", "assistant".into())])), "A");
    }

    #[test]
    fn range_binds_dot_to_the_element() {
        let data = map(&[(
            "M",
            Value::List(vec![map(&[("C", "a".into())]), map(&[("C", "b".into())])]),
        )]);
        assert_eq!(render("{{ range .M }}[{{ .C }}]{{ end }}", &data), "[a][b]");
    }

    #[test]
    fn range_with_index_and_value_variables() {
        let data = map(&[("M", Value::List(vec!["a".into(), "b".into()]))]);
        assert_eq!(
            render("{{ range $i, $v := .M }}{{ $i }}={{ $v }};{{ end }}", &data),
            "0=a;1=b;"
        );
    }

    /// Go runs `{{ else }}` on an EMPTY range -- reads backwards, but templates
    /// use it for "no messages yet".
    #[test]
    fn range_else_runs_when_the_subject_is_empty() {
        let data = map(&[("M", Value::List(vec![]))]);
        assert_eq!(render("{{ range .M }}x{{ else }}none{{ end }}", &data), "none");
    }

    #[test]
    fn continue_and_break_control_the_loop() {
        let data = map(&[("M", Value::List(vec![1i64.into(), 2i64.into(), 3i64.into()]))]);
        assert_eq!(
            render("{{ range .M }}{{ if eq . 2 }}{{ continue }}{{ end }}{{ . }}{{ end }}", &data),
            "13"
        );
        assert_eq!(
            render("{{ range .M }}{{ if eq . 3 }}{{ break }}{{ end }}{{ . }}{{ end }}", &data),
            "12"
        );
    }

    /// `=` must write THROUGH to an outer binding. Declaring afresh each
    /// iteration would silently keep only the last system message.
    #[test]
    fn assignment_writes_through_to_the_outer_scope() {
        let data = map(&[("M", Value::List(vec!["a".into(), "b".into()]))]);
        let t = "{{- $s := \"\" }}{{ range .M }}{{ $s = printf \"%s%s\" $s . }}{{ end }}{{ $s }}";
        assert_eq!(render(t, &data), "ab");
    }

    /// ...and `:=` inside the loop must NOT, or the accumulator resets.
    #[test]
    fn declaration_inside_a_loop_does_not_leak_outward() {
        let data = map(&[("M", Value::List(vec!["a".into(), "b".into()]))]);
        let t = "{{- $s := \"start\" }}{{ range .M }}{{ $s := . }}{{ $s }}{{ end }}|{{ $s }}";
        assert_eq!(render(t, &data), "ab|start");
    }

    #[test]
    fn dollar_reaches_the_root_from_inside_a_range() {
        let data = map(&[
            ("System", "S".into()),
            ("M", Value::List(vec!["a".into(), "b".into()])),
        ]);
        assert_eq!(render("{{ range .M }}{{ $.System }}{{ . }}{{ end }}", &data), "SaSb");
    }

    #[test]
    fn with_rebinds_dot_only_when_truthy() {
        let data = map(&[("A", map(&[("B", "inner".into())]))]);
        assert_eq!(render("{{ with .A }}{{ .B }}{{ end }}", &data), "inner");
        assert_eq!(render("{{ with .Z }}x{{ else }}none{{ end }}", &data), "none");
    }

    #[test]
    fn pipes_feed_the_last_argument() {
        let data = map(&[("A", "x".into())]);
        assert_eq!(render("{{ .A | printf \"<%s>\" }}", &data), "<x>");
    }

    #[test]
    fn nested_calls_in_parens_evaluate_inside_out() {
        let data = map(&[("M", Value::List(vec!["a".into(), "b".into(), "c".into()]))]);
        assert_eq!(render("{{ len (slice .M 1) }}", &data), "2");
    }

    /// `and`/`or` return a VALUE, not a bool -- templates print the result.
    #[test]
    fn and_or_return_values_not_bools() {
        let data = map(&[("A", "x".into()), ("B", "y".into()), ("E", "".into())]);
        assert_eq!(render("{{ or .E .B }}", &data), "y");
        assert_eq!(render("{{ and .A .B }}", &data), "y", "and yields the LAST truthy");
        assert_eq!(render("{{ and .E .B }}", &data), "", "and yields the first falsy");
    }

    /// Sorted map iteration -- the reproducibility guarantee. Insert in reverse
    /// and the output must still come out sorted.
    #[test]
    fn maps_range_in_sorted_key_order() {
        let data = map(&[(
            "P",
            map(&[("zeta", "1".into()), ("alpha", "2".into()), ("mid", "3".into())]),
        )]);
        assert_eq!(
            render("{{ range $k, $v := .P }}{{ $k }}={{ $v }},{{ end }}", &data),
            "alpha=2,mid=3,zeta=1,"
        );
    }

    #[test]
    fn trim_markers_shape_the_whitespace() {
        let data = map(&[("A", "x".into())]);
        assert_eq!(render("a\n  {{- .A }}", &data), "ax");
        assert_eq!(render("{{ .A -}}\n  b", &data), "xb");
    }

    #[test]
    fn json_marshals_a_value() {
        let data = map(&[("A", map(&[("k", "v".into())]))]);
        assert_eq!(render("{{ json .A }}", &data), r#"{"k":"v"}"#);
    }

    #[test]
    fn current_and_yesterday_date_respect_the_injected_clock() {
        let t = Template::parse("{{ currentDate }}/{{ yesterdayDate }}").unwrap();
        let env = Env {
            today: Some("2026-03-01".to_string()),
        };
        assert_eq!(t.execute(&Value::Nil, &env).unwrap(), "2026-03-01/2026-02-28");
    }

    /// `missingkey=zero`: a missing field is nil, not an error -- which is what
    /// lets one template serve models that do and do not supply a field.
    #[test]
    fn a_missing_field_is_nil_not_an_error() {
        let data = map(&[]);
        assert_eq!(render("{{ if .Nope }}x{{ end }}", &data), "");
        assert_eq!(render("{{ .A.B.C }}", &map(&[])), "<no value>");
    }

    #[test]
    fn vars_lowercases_and_sorts_every_identifier() {
        let t = Template::parse("{{ if .System }}{{ range .Messages }}{{ .Content }}{{ end }}{{ end }}")
            .unwrap();
        assert_eq!(t.vars(), vec!["content", "messages", "system"]);
    }

    #[test]
    fn vars_sees_through_parens_and_pipes() {
        let t = Template::parse("{{ len (slice $.Messages 1) | printf \"%d\" }}").unwrap();
        assert!(t.vars().contains(&"messages".to_string()));
    }

    /// A realistic ChatML template end to end -- the exact shape of what a
    /// Qwen-family GGUF ships.
    #[test]
    fn a_chatml_style_template_renders_a_whole_conversation() {
        let src = concat!(
            "{{- range .Messages }}<|im_start|>{{ .Role }}\n",
            "{{ .Content }}<|im_end|>\n",
            "{{ end }}<|im_start|>assistant\n"
        );
        let msgs = Value::List(vec![
            map(&[("Role", "system".into()), ("Content", "be terse".into())]),
            map(&[("Role", "user".into()), ("Content", "hi".into())]),
        ]);
        let out = render(src, &map(&[("Messages", msgs)]));
        assert_eq!(
            out,
            "<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// And a Gemma-style one, which is the model family bead bd-250.3 says
    /// KOPITIAM currently mis-frames by forcing ChatML on it.
    #[test]
    fn a_gemma_style_template_renders_its_own_markers() {
        let src = concat!(
            "{{- range .Messages }}<start_of_turn>",
            "{{ if eq .Role \"assistant\" }}model{{ else }}user{{ end }}\n",
            "{{ .Content }}<end_of_turn>\n",
            "{{ end }}<start_of_turn>model\n"
        );
        let msgs = Value::List(vec![
            map(&[("Role", "user".into()), ("Content", "hi".into())]),
            map(&[("Role", "assistant".into()), ("Content", "hello".into())]),
        ]);
        let out = render(src, &map(&[("Messages", msgs)]));
        assert_eq!(
            out,
            "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\nhello<end_of_turn>\n<start_of_turn>model\n"
        );
    }
}
