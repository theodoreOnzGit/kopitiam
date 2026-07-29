//! Values, builtin functions, and the tree-walking executor.
//!
//! **Upstream:** Go's `text/template` execution semantics (BSD-3-Clause,
//! Copyright (c) 2009 The Go Authors), plus ollama's own `funcs` map from
//! `template/template.go` (MIT).
//!
//! ## Two semantics worth knowing before you read the code
//!
//! **Truth is Go's, not Rust's.** `{{ if .X }}` is true when `X` is a non-empty
//! string, a non-zero number, a non-empty list or map, or `true` -- see
//! [`Value::is_true`]. Chat templates lean on this constantly (`{{ if .System }}`
//! means "if there IS a system prompt"), so an ordinary Rust `Option` check in
//! its place would render the wrong branch.
//!
//! **Maps iterate in sorted key order.** Go's `text/template` explicitly sorts
//! map keys before ranging, precisely so template output is reproducible. We get
//! that free by storing maps in a [`BTreeMap`]. That is not an implementation
//! detail to optimise away later -- a tool-schema template that ranges over
//! `.Properties` would otherwise emit a different prompt on every run, and a
//! non-reproducible prompt is a non-reproducible model.

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// A template data value. Deliberately small -- these are the only shapes a
/// chat template ever sees.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Absent or explicitly nil.
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    /// **Sorted** by key -- see the module docs on why that is load-bearing.
    Map(BTreeMap<String, Value>),
}

impl Value {
    /// Go's template `truth` rule. See the module docs.
    pub fn is_true(&self) -> bool {
        match self {
            Value::Nil => false,
            Value::Bool(b) => *b,
            Value::Int(n) => *n != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(v) => !v.is_empty(),
            Value::Map(m) => !m.is_empty(),
        }
    }

    /// How a value prints when an action emits it -- Go's `%v`.
    ///
    /// **`Nil` prints as `<no value>`**, which is Go's rendering of an untyped
    /// nil under `missingkey=zero` (the option ollama sets). It looks alarming
    /// in a prompt, and that is the point: it is upstream's behaviour, and
    /// seeing it means a template referenced a key its caller never supplied.
    /// Quietly printing `""` instead would hide a real template/caller mismatch.
    pub fn render(&self) -> String {
        match self {
            Value::Nil => "<no value>".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Int(n) => n.to_string(),
            Value::Float(f) => format_go_float(*f),
            Value::Str(s) => s.clone(),
            Value::List(v) => {
                // Go prints a slice as `[a b c]`.
                let inner: Vec<String> = v.iter().map(Value::render).collect();
                format!("[{}]", inner.join(" "))
            }
            Value::Map(m) => {
                // Go prints a map as `map[k:v ...]`, keys sorted.
                let inner: Vec<String> =
                    m.iter().map(|(k, v)| format!("{k}:{}", v.render())).collect();
                format!("map[{}]", inner.join(" "))
            }
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Value::Nil => "nil",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "string",
            Value::List(_) => "list",
            Value::Map(_) => "map",
        }
    }

    fn as_number(&self) -> Option<f64> {
        match self {
            Value::Int(n) => Some(*n as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Convenience for building `Values` from JSON-shaped data.
    pub fn from_json(v: &serde_json::Value) -> Value {
        match v {
            serde_json::Value::Null => Value::Nil,
            serde_json::Value::Bool(b) => Value::Bool(*b),
            serde_json::Value::Number(n) => match n.as_i64() {
                Some(i) => Value::Int(i),
                None => Value::Float(n.as_f64().unwrap_or(0.0)),
            },
            serde_json::Value::String(s) => Value::Str(s.clone()),
            serde_json::Value::Array(a) => Value::List(a.iter().map(Value::from_json).collect()),
            serde_json::Value::Object(o) => Value::Map(
                o.iter()
                    .map(|(k, v)| (k.clone(), Value::from_json(v)))
                    .collect(),
            ),
        }
    }

    fn to_json(&self) -> serde_json::Value {
        match self {
            Value::Nil => serde_json::Value::Null,
            Value::Bool(b) => serde_json::Value::Bool(*b),
            Value::Int(n) => serde_json::Value::from(*n),
            Value::Float(f) => serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Value::Str(s) => serde_json::Value::String(s.clone()),
            Value::List(v) => serde_json::Value::Array(v.iter().map(Value::to_json).collect()),
            Value::Map(m) => serde_json::Value::Object(
                m.iter().map(|(k, v)| (k.clone(), v.to_json())).collect(),
            ),
        }
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::Str(s.to_string())
    }
}
impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::Str(s)
    }
}
impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}
impl From<i64> for Value {
    fn from(n: i64) -> Self {
        Value::Int(n)
    }
}

/// Go's `%v` for a float: shortest representation that round-trips, and a bare
/// integer prints without a trailing `.0`.
fn format_go_float(f: f64) -> String {
    if f == f.trunc() && f.abs() < 1e15 {
        format!("{}", f as i64)
    } else {
        format!("{f}")
    }
}

/// A template execution failure.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("template exec: {0}")]
pub struct ExecError(pub String);

fn ex<T>(msg: impl Into<String>) -> Result<T, ExecError> {
    Err(ExecError(msg.into()))
}

/// Knobs the caller controls, so execution stays deterministic in tests.
#[derive(Debug, Clone, Default)]
pub struct Env {
    /// `YYYY-MM-DD` for `currentDate`. `None` reads the system clock.
    ///
    /// Injectable **on purpose**. Upstream's `currentDate` calls `time.Now()`
    /// directly, which makes any template using it untestable and any rendered
    /// prompt unreproducible. KOPITIAM cares about reproducibility, so the clock
    /// becomes an input. Leave it `None` and you get upstream's behaviour
    /// exactly; set it and a prompt can be re-derived a year later.
    pub today: Option<String>,
}

/// Where a `range` body left off.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Flow {
    Normal,
    Continue,
    Break,
}

struct Exec<'a> {
    out: String,
    /// Variable scopes, innermost last.
    scopes: Vec<Vec<(String, Value)>>,
    env: &'a Env,
}

/// Execute a parsed template against `data`.
///
/// `data` becomes both the initial `.` and the `$` root variable, matching Go.
pub fn execute(nodes: &[super::parse::Node], data: &Value, env: &Env) -> Result<String, ExecError> {
    let mut e = Exec {
        out: String::new(),
        scopes: vec![vec![("$".to_string(), data.clone())]],
        env,
    };
    e.walk(nodes, data)?;
    Ok(e.out)
}

impl Exec<'_> {
    fn walk(&mut self, nodes: &[super::parse::Node], dot: &Value) -> Result<Flow, ExecError> {
        use super::parse::Node;
        for n in nodes {
            match n {
                Node::Text(t) => self.out.push_str(t),
                Node::Action(p) => {
                    let v = self.pipeline(p, dot)?;
                    self.out.push_str(&v.render());
                }
                Node::Assign { var, define, pipe } => {
                    let v = self.pipeline(pipe, dot)?;
                    if *define {
                        self.scopes.last_mut().expect("a scope").push((var.clone(), v));
                    } else {
                        // `=` writes through to the nearest existing binding.
                        // This is how a template accumulates a system prompt
                        // across a range body -- declaring a fresh one instead
                        // would reset it every iteration.
                        let mut found = false;
                        for scope in self.scopes.iter_mut().rev() {
                            if let Some(slot) = scope.iter_mut().rev().find(|(n, _)| n == var) {
                                slot.1 = v.clone();
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            return ex(format!("undefined variable {var}"));
                        }
                    }
                }
                Node::If {
                    pipe,
                    then,
                    otherwise,
                } => {
                    let v = self.pipeline(pipe, dot)?;
                    let branch = if v.is_true() { then } else { otherwise };
                    let flow = self.scoped(|s| s.walk(branch, dot))?;
                    if flow != Flow::Normal {
                        return Ok(flow);
                    }
                }
                Node::With {
                    pipe,
                    body,
                    otherwise,
                } => {
                    let v = self.pipeline(pipe, dot)?;
                    let flow = if v.is_true() {
                        // `with` rebinds `.` to the value -- that is the whole
                        // difference from `if`.
                        self.scoped(|s| s.walk(body, &v))?
                    } else {
                        self.scoped(|s| s.walk(otherwise, dot))?
                    };
                    if flow != Flow::Normal {
                        return Ok(flow);
                    }
                }
                Node::Range {
                    key,
                    val,
                    pipe,
                    body,
                    otherwise,
                } => {
                    let subject = self.pipeline(pipe, dot)?;
                    let items: Vec<(Value, Value)> = match &subject {
                        Value::List(v) => v
                            .iter()
                            .enumerate()
                            .map(|(i, x)| (Value::Int(i as i64), x.clone()))
                            .collect(),
                        // BTreeMap gives sorted keys, matching Go -- see module docs.
                        Value::Map(m) => m
                            .iter()
                            .map(|(k, v)| (Value::Str(k.clone()), v.clone()))
                            .collect(),
                        Value::Nil => Vec::new(),
                        Value::Int(n) => (0..*n).map(|i| (Value::Int(i), Value::Int(i))).collect(),
                        other => {
                            return ex(format!("range: cannot range over {}", other.kind()));
                        }
                    };

                    if items.is_empty() {
                        // Go runs the `else` branch when the subject is EMPTY --
                        // reads backwards, but it is how a template says "no
                        // messages yet".
                        let flow = self.scoped(|s| s.walk(otherwise, dot))?;
                        if flow != Flow::Normal {
                            return Ok(flow);
                        }
                        continue;
                    }

                    for (k, v) in items {
                        let flow = self.scoped(|s| {
                            if let Some(kn) = key {
                                s.scopes.last_mut().expect("scope").push((kn.clone(), k.clone()));
                            }
                            if let Some(vn) = val {
                                s.scopes.last_mut().expect("scope").push((vn.clone(), v.clone()));
                            }
                            // `.` is the element regardless of whether variables
                            // were named -- that is what makes `{{ range .X }}
                            // {{ .Field }}{{ end }}` work.
                            s.walk(body, &v)
                        })?;
                        match flow {
                            Flow::Break => break,
                            // `continue` is per-iteration; it must NOT escape.
                            Flow::Continue | Flow::Normal => {}
                        }
                    }
                }
                Node::Continue => return Ok(Flow::Continue),
                Node::Break => return Ok(Flow::Break),
            }
        }
        Ok(Flow::Normal)
    }

    /// Run `f` inside a fresh variable scope, popping it however `f` exits.
    fn scoped<F>(&mut self, f: F) -> Result<Flow, ExecError>
    where
        F: FnOnce(&mut Self) -> Result<Flow, ExecError>,
    {
        self.scopes.push(Vec::new());
        let r = f(self);
        self.scopes.pop();
        r
    }

    fn lookup_var(&self, name: &str) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some((_, v)) = scope.iter().rev().find(|(n, _)| n == name) {
                return Some(v.clone());
            }
        }
        None
    }

    fn pipeline(
        &mut self,
        p: &super::parse::Pipeline,
        dot: &Value,
    ) -> Result<Value, ExecError> {
        let mut carried: Option<Value> = None;
        for cmd in &p.cmds {
            carried = Some(self.command(cmd, dot, carried.take())?);
        }
        Ok(carried.unwrap_or(Value::Nil))
    }

    /// Evaluate one command. A piped-in value is appended as the **final**
    /// argument, which is Go's rule and the reason `{{ .X | printf "%s" }}`
    /// works out to `printf "%s" .X`.
    fn command(
        &mut self,
        cmd: &super::parse::Command,
        dot: &Value,
        piped: Option<Value>,
    ) -> Result<Value, ExecError> {
        use super::parse::Arg;

        if let Arg::Ident(name) = &cmd.args[0] {
            let mut args = Vec::with_capacity(cmd.args.len());
            for a in &cmd.args[1..] {
                args.push(self.arg(a, dot)?);
            }
            if let Some(v) = piped {
                args.push(v);
            }
            return self.call(name, args);
        }

        if cmd.args.len() > 1 {
            return ex(format!(
                "cannot call non-function head of command ({} args)",
                cmd.args.len()
            ));
        }
        let v = self.arg(&cmd.args[0], dot)?;
        // A bare value in a later pipeline stage discards what was piped in --
        // same as Go, which would report "can't give argument to non-function".
        // We are lenient because no real template does it.
        Ok(v)
    }

    fn arg(&mut self, a: &super::parse::Arg, dot: &Value) -> Result<Value, ExecError> {
        use super::parse::Arg;
        Ok(match a {
            Arg::Dot => dot.clone(),
            Arg::Field(path) => field_chain(dot, path)?,
            Arg::Var(name, path) => {
                let base = self
                    .lookup_var(name)
                    .ok_or_else(|| ExecError(format!("undefined variable {name}")))?;
                field_chain(&base, path)?
            }
            Arg::Str(s) => Value::Str(s.clone()),
            Arg::Int(n) => Value::Int(*n),
            Arg::Float(f) => Value::Float(*f),
            Arg::Bool(b) => Value::Bool(*b),
            Arg::Nil => Value::Nil,
            Arg::Ident(name) => self.call(name, Vec::new())?,
            Arg::Paren(p) => self.pipeline(p, dot)?,
        })
    }

    fn call(&mut self, name: &str, args: Vec<Value>) -> Result<Value, ExecError> {
        match name {
            // ---- Go builtins ----
            //
            // `and`/`or` return a VALUE, not a bool: `and` yields the first
            // falsy argument (or the last one), `or` the first truthy. Templates
            // rely on that -- `{{ or .A .B }}` prints A when A is set.
            "and" => {
                let mut last = Value::Bool(true);
                for a in args {
                    if !a.is_true() {
                        return Ok(a);
                    }
                    last = a;
                }
                Ok(last)
            }
            "or" => {
                let mut last = Value::Bool(false);
                for a in args {
                    if a.is_true() {
                        return Ok(a);
                    }
                    last = a;
                }
                Ok(last)
            }
            "not" => Ok(Value::Bool(!one(&args, "not")?.is_true())),
            "eq" => {
                let (first, rest) = args.split_first().ok_or(ExecError("eq: no arguments".into()))?;
                if rest.is_empty() {
                    return ex("eq: need at least two arguments");
                }
                // Go's `eq` is variadic: true if the first equals ANY of the rest.
                Ok(Value::Bool(rest.iter().any(|r| loose_eq(first, r))))
            }
            "ne" => {
                let (a, b) = two(&args, "ne")?;
                Ok(Value::Bool(!loose_eq(a, b)))
            }
            "lt" | "le" | "gt" | "ge" => {
                let (a, b) = two(&args, name)?;
                let ord = compare(a, b, name)?;
                Ok(Value::Bool(match name {
                    "lt" => ord.is_lt(),
                    "le" => ord.is_le(),
                    "gt" => ord.is_gt(),
                    _ => ord.is_ge(),
                }))
            }
            "len" => Ok(Value::Int(match one(&args, "len")? {
                Value::Str(s) => s.chars().count() as i64,
                Value::List(v) => v.len() as i64,
                Value::Map(m) => m.len() as i64,
                Value::Nil => 0,
                other => return ex(format!("len of type {}", other.kind())),
            })),
            "index" => {
                let (first, rest) = args
                    .split_first()
                    .ok_or(ExecError("index: no arguments".into()))?;
                let mut cur = first.clone();
                for k in rest {
                    cur = index_one(&cur, k)?;
                }
                Ok(cur)
            }
            "slice" => {
                let (first, rest) = args
                    .split_first()
                    .ok_or(ExecError("slice: no arguments".into()))?;
                let start = rest.first().and_then(Value::as_number).unwrap_or(0.0) as usize;
                match first {
                    Value::List(v) => {
                        let end = rest
                            .get(1)
                            .and_then(Value::as_number)
                            .map(|f| f as usize)
                            .unwrap_or(v.len());
                        if start > v.len() || end > v.len() || start > end {
                            return ex("slice: index out of range");
                        }
                        Ok(Value::List(v[start..end].to_vec()))
                    }
                    Value::Str(s) => {
                        let ch: Vec<char> = s.chars().collect();
                        let end = rest
                            .get(1)
                            .and_then(Value::as_number)
                            .map(|f| f as usize)
                            .unwrap_or(ch.len());
                        if start > ch.len() || end > ch.len() || start > end {
                            return ex("slice: index out of range");
                        }
                        Ok(Value::Str(ch[start..end].iter().collect()))
                    }
                    other => ex(format!("slice of type {}", other.kind())),
                }
            }
            "print" => Ok(Value::Str(
                args.iter().map(Value::render).collect::<Vec<_>>().join(""),
            )),
            "println" => Ok(Value::Str(format!(
                "{}\n",
                args.iter().map(Value::render).collect::<Vec<_>>().join(" ")
            ))),
            "printf" => {
                let (f, rest) = args
                    .split_first()
                    .ok_or(ExecError("printf: no format".into()))?;
                let Value::Str(f) = f else {
                    return ex("printf: format must be a string");
                };
                Ok(Value::Str(sprintf(f, rest)?))
            }

            // ---- ollama's own funcs (template/template.go `var funcs`) ----
            "json" => {
                let v = one(&args, "json")?;
                Ok(Value::Str(
                    serde_json::to_string(&v.to_json())
                        .map_err(|e| ExecError(format!("json: {e}")))?,
                ))
            }
            // Upstream ignores the format argument too ("accepting it for future
            // use") and always emits YYYY-MM-DD.
            "currentDate" => Ok(Value::Str(self.today(0))),
            "yesterdayDate" => Ok(Value::Str(self.today(-1))),
            // Upstream maps an api.ToolProperty to a TypeScript type and returns
            // "any" for everything else. Without the tool types ported yet, every
            // input lands in that fallback -- faithful for now, and the bead on
            // the port epic tracks the real implementation.
            "toTypeScriptType" => Ok(Value::Str("any".to_string())),

            other => ex(format!("function {other:?} not defined")),
        }
    }

    fn today(&self, day_offset: i64) -> String {
        if let Some(t) = &self.env.today {
            if day_offset == 0 {
                return t.clone();
            }
            // Shift an injected date by parsing it back to a day number.
            if let Some(d) = parse_ymd(t) {
                return ymd_from_days(d + day_offset);
            }
            return t.clone();
        }
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        ymd_from_days(secs.div_euclid(86_400) + day_offset)
    }
}

fn one<'a>(args: &'a [Value], who: &str) -> Result<&'a Value, ExecError> {
    match args {
        [a] => Ok(a),
        _ => ex(format!("{who}: wrong number of arguments ({})", args.len())),
    }
}

fn two<'a>(args: &'a [Value], who: &str) -> Result<(&'a Value, &'a Value), ExecError> {
    match args {
        [a, b] => Ok((a, b)),
        _ => ex(format!("{who}: wrong number of arguments ({})", args.len())),
    }
}

/// Go's `eq` semantics: numbers compare numerically across int/float, everything
/// else compares within its own kind.
fn loose_eq(a: &Value, b: &Value) -> bool {
    match (a.as_number(), b.as_number()) {
        (Some(x), Some(y)) => x == y,
        _ => a == b,
    }
}

fn compare(a: &Value, b: &Value, who: &str) -> Result<std::cmp::Ordering, ExecError> {
    match (a, b) {
        (Value::Str(x), Value::Str(y)) => Ok(x.cmp(y)),
        _ => match (a.as_number(), b.as_number()) {
            (Some(x), Some(y)) => x
                .partial_cmp(&y)
                .ok_or_else(|| ExecError(format!("{who}: incomparable NaN"))),
            _ => ex(format!(
                "{who}: incompatible types {} and {}",
                a.kind(),
                b.kind()
            )),
        },
    }
}

fn index_one(v: &Value, k: &Value) -> Result<Value, ExecError> {
    match (v, k) {
        // Missing key yields the zero value, matching the `missingkey=zero`
        // option ollama sets -- NOT an error.
        (Value::Map(m), Value::Str(s)) => Ok(m.get(s).cloned().unwrap_or(Value::Nil)),
        (Value::List(l), k) => {
            let i = k
                .as_number()
                .ok_or_else(|| ExecError("index: non-numeric list index".into()))?
                as i64;
            if i < 0 || i as usize >= l.len() {
                return ex("index out of range");
            }
            Ok(l[i as usize].clone())
        }
        (Value::Nil, _) => Ok(Value::Nil),
        (other, _) => ex(format!("index of type {}", other.kind())),
    }
}

/// Walk `.A.B.C` from a base value.
///
/// A missing key gives [`Value::Nil`] rather than an error -- ollama parses its
/// templates with `Option("missingkey=zero")`, so this is the oracle's own
/// behaviour, and it is what lets one template serve models that do or do not
/// supply, say, `Thinking`.
fn field_chain(base: &Value, path: &[String]) -> Result<Value, ExecError> {
    let mut cur = base.clone();
    for f in path {
        cur = match cur {
            Value::Map(ref m) => m.get(f).cloned().unwrap_or(Value::Nil),
            Value::Nil => Value::Nil,
            other => return ex(format!("can't evaluate field {f} in type {}", other.kind())),
        };
    }
    Ok(cur)
}

/// A `fmt.Sprintf` subset: `%v %s %d %q %t %f %.Nf %%`.
///
/// Enough for every verb the bundled templates use (`printf "%s\n\n%s"` is the
/// only one in ollama's own set). An unsupported verb is an error rather than a
/// silent passthrough -- a template that renders `%x` literally into a prompt
/// would be a very quiet bug.
fn sprintf(f: &str, args: &[Value]) -> Result<String, ExecError> {
    let mut out = String::new();
    let mut it = f.chars().peekable();
    let mut ai = 0usize;

    while let Some(c) = it.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        if it.peek() == Some(&'%') {
            it.next();
            out.push('%');
            continue;
        }

        // Optional precision, `%.3f`.
        let mut prec: Option<usize> = None;
        if it.peek() == Some(&'.') {
            it.next();
            let mut digits = String::new();
            while it.peek().is_some_and(|d| d.is_ascii_digit()) {
                digits.push(it.next().expect("peeked"));
            }
            prec = digits.parse().ok();
        }

        let Some(verb) = it.next() else {
            return ex("printf: format ends in %");
        };
        let Some(a) = args.get(ai) else {
            return ex(format!("printf: not enough arguments for %{verb}"));
        };
        ai += 1;

        match verb {
            'v' | 's' => out.push_str(&a.render()),
            'd' => match a.as_number() {
                Some(n) => {
                    let _ = write!(out, "{}", n as i64);
                }
                None => return ex("printf: %d needs a number"),
            },
            'f' => match a.as_number() {
                Some(n) => {
                    let _ = write!(out, "{:.*}", prec.unwrap_or(6), n);
                }
                None => return ex("printf: %f needs a number"),
            },
            't' => {
                let _ = write!(out, "{}", a.is_true());
            }
            'q' => {
                let _ = write!(out, "{:?}", a.render());
            }
            other => return ex(format!("printf: unsupported verb %{other}")),
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Civil date arithmetic, for currentDate / yesterdayDate
// ---------------------------------------------------------------------------
//
// Howard Hinnant's `civil_from_days` / `days_from_civil` (public domain, from
// "chrono-Compatible Low-Level Date Algorithms",
// https://howardhinnant.github.io/date_algorithms.html). Ported here rather than
// pulling in `chrono` for two functions -- the crate otherwise has no date
// dependency and KOPITIAM's dependency budget is not free. Valid for the
// proleptic Gregorian calendar, which covers every date a prompt will ever
// carry.

fn ymd_from_days(z: i64) -> String {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

fn parse_ymd(s: &str) -> Option<i64> {
    let mut parts = s.split('-');
    let y: i64 = parts.next()?.parse().ok()?;
    let m: i64 = parts.next()?.parse().ok()?;
    let d: i64 = parts.next()?.parse().ok()?;
    Some(days_from_civil(y, m, d))
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn go_truth_is_not_rust_truth() {
        assert!(!Value::Str(String::new()).is_true());
        assert!(Value::Str("x".into()).is_true());
        assert!(!Value::Int(0).is_true());
        assert!(Value::Int(1).is_true());
        assert!(!Value::List(vec![]).is_true());
        assert!(!Value::Nil.is_true());
        assert!(!Value::Map(BTreeMap::new()).is_true());
    }

    #[test]
    fn printf_covers_the_verbs_templates_use() {
        let a = [Value::Str("x".into()), Value::Str("y".into())];
        assert_eq!(sprintf("%s\n\n%s", &a).unwrap(), "x\n\ny");
        assert_eq!(sprintf("100%%", &[]).unwrap(), "100%");
        assert_eq!(sprintf("%d", &[Value::Int(7)]).unwrap(), "7");
        assert_eq!(sprintf("%.2f", &[Value::Float(1.5)]).unwrap(), "1.50");
        assert!(sprintf("%x", &[Value::Int(7)]).is_err(), "unknown verb must fail");
    }

    /// The date helpers must round-trip, or `yesterdayDate` walks off by a day
    /// at month and year boundaries.
    #[test]
    fn civil_date_arithmetic_round_trips_across_boundaries() {
        for s in ["1970-01-01", "2000-02-29", "2026-07-29", "2027-01-01", "2100-03-01"] {
            let d = parse_ymd(s).unwrap();
            assert_eq!(ymd_from_days(d), s, "round trip of {s}");
        }
        // Leap-day and new-year steps, the two that catch a naive -1.
        assert_eq!(ymd_from_days(parse_ymd("2000-03-01").unwrap() - 1), "2000-02-29");
        assert_eq!(ymd_from_days(parse_ymd("2027-01-01").unwrap() - 1), "2026-12-31");
    }

    #[test]
    fn float_rendering_drops_a_pointless_decimal() {
        assert_eq!(format_go_float(3.0), "3");
        assert_eq!(format_go_float(3.5), "3.5");
    }
}
