//! The virtual machine, and the public embedding API.
//!
//! [`Lua`] *is* the VM. That is deliberate: it means a native function can take
//! `&mut Lua` and do anything the host can do — call back into Lua, read and
//! write globals, build tables — without a second "context" type that is really
//! the same thing wearing a hat.
//!
//! # The execution model in one paragraph
//!
//! A [`Thread`] is a stack of [`Frame`]s. The VM holds a *stack of threads*: the
//! main thread at the bottom, and one for each coroutine currently resumed. A
//! Lua-to-Lua call **pushes a frame** and returns to the dispatch loop — it does
//! not recurse in Rust. That is what makes a coroutine suspendable: `yield` lifts
//! the whole top thread out of the VM and parks it in a [`Coroutine`], with its
//! frames, program counters and stacks exactly as they were. Resuming pushes it
//! back. Nothing is reconstructed, because nothing was unwound. (AID-0007.)
//!
//! # Where the Rust stack *does* get involved
//!
//! A native function that calls back into Lua — `table.sort`'s comparator, a
//! `__index` metamethod, `string.gsub`'s replacement function — runs a **nested**
//! dispatch loop, and that one does sit on the Rust stack. Yielding across it is
//! therefore impossible, and the VM says so with Lua's own error message:
//!
//! ```text
//! attempt to yield across a C-call boundary
//! ```
//!
//! This is not a shortcoming we are apologising for: it is *exactly* the
//! restriction real Lua 5.1 imposes, for exactly the same reason. See
//! [`Lua::run`] and the `base_run_depth` field.

use std::cell::RefCell;
use std::rc::Rc;

use crate::error::{LuaError, Result};
use crate::instr::{Instr, NRes, Proto, UpvalSource};
use crate::value::{
    Closure, CoStatus, Coroutine, LuaStr, NativeFunction, NativeKind, Outcome, Table, Value,
    new_table,
};

/// Resolves a module name to its source, for `require`. Supplied by the host —
/// see [`Lua::set_module_loader`].
pub type ModuleLoader = Rc<dyn Fn(&str) -> Option<String>>;

/// Where `print` writes. See [`Lua::set_output`].
pub type OutputSink = Box<dyn FnMut(&str)>;

/// How deep the Lua call stack may get before we call it a runaway recursion.
///
/// Frames are heap-allocated, so an infinite recursion would otherwise eat all
/// of memory rather than failing cleanly. Real Lua reports `stack overflow`; so
/// do we, and `pcall` can catch it.
const MAX_FRAMES: usize = 200_000;

/// How far an `__index` chain may be followed before we assume it is a cycle.
/// `setmetatable(a, {__index = b}); setmetatable(b, {__index = a})` would
/// otherwise hang.
const MAX_METATABLE_CHAIN: usize = 100;

/// What to do with a call's results when it returns.
///
/// This is the whole of Lua's return protocol, and it composes: `Protected`
/// wraps another mode, so `pcall(pcall, f)` correctly produces `true, true, ...`
/// by prepending a boolean twice.
#[derive(Clone)]
pub(crate) enum ReturnMode {
    /// Push the results into the calling frame's value stack, adjusted to the
    /// count the call site asked for.
    Normal(NRes),
    /// A protected call. On a normal return, prepend `true`; on an error, produce
    /// `false, err` instead of propagating. Either way the result is then handed
    /// to `outer`.
    Protected { handler: Option<Value>, outer: Box<ReturnMode> },
    /// The bottom frame of a [`Lua::run`] invocation: results go to
    /// `self.returned` and the loop stops.
    HostRoot,
    /// The bottom frame of a coroutine: the thread finishes and the results go
    /// to whoever resumed it.
    CoroutineRoot,
}

/// One Lua call.
pub(crate) struct Frame {
    closure: Rc<Closure>,
    pc: usize,
    /// Local variable cells. `None` until the local's declaration executes.
    ///
    /// Every local is its own `Rc<RefCell<Value>>`, so a closure that captures it
    /// captures the *cell*. See the compiler's module docs for why this deletes
    /// an entire category of upvalue bugs.
    slots: Vec<Option<Rc<RefCell<Value>>>>,
    /// The operand stack for this frame.
    stack: Vec<Value>,
    /// Saved stack heights — see [`Instr::Mark`].
    marks: Vec<usize>,
    varargs: Vec<Value>,
    ret: ReturnMode,
}

/// A coroutine's (or the main thread's) call stack.
pub(crate) struct Thread {
    frames: Vec<Frame>,
    /// `None` for the main thread.
    resumed_by: Option<ResumeInfo>,
    /// The [`Lua::run`] nesting depth at which this thread was pushed.
    ///
    /// A yield is legal only when `run_depth` still equals this. If a native has
    /// since started a nested dispatch loop, there is Rust stack between the
    /// yield and the resume, and suspending would strand it — which is precisely
    /// the "C-call boundary" real Lua refuses to cross.
    base_run_depth: usize,
}

struct ResumeInfo {
    co: Rc<RefCell<Coroutine>>,
    /// Where the results of `resume` go in the *resumer*.
    ret: ReturnMode,
    /// `coroutine.wrap` rather than `coroutine.resume`: results come back
    /// without a leading boolean, and errors propagate instead of being returned.
    wrapped: bool,
}

/// A Lua interpreter.
///
/// # Example
///
/// ```
/// use kopitiam_lua::{Lua, Value};
///
/// let mut lua = Lua::new();
/// lua.exec("x = 1 + 2", "=example").unwrap();
/// assert_eq!(lua.get_global("x").as_number(), Some(3.0));
/// ```
///
/// # Injecting Rust into Lua
///
/// This is the interface kvim exists to use: `vim.keymap.set` and friends are
/// Rust functions that Lua config calls.
///
/// ```
/// use kopitiam_lua::{Lua, Value};
///
/// let mut lua = Lua::new();
/// lua.set_global_fn("double", |_lua, args| {
///     let n = args.first().and_then(|v| v.as_number()).unwrap_or(0.0);
///     Ok(vec![Value::Number(n * 2.0)])
/// });
/// assert_eq!(lua.eval("double(21)").unwrap().as_number(), Some(42.0));
/// ```
pub struct Lua {
    /// The thread stack. `threads[0]` is the main thread and always exists.
    threads: Vec<Thread>,
    globals: Rc<RefCell<Table>>,
    /// The single metatable shared by *all* strings. It is what makes
    /// `("x"):upper()` work: indexing a string finds `__index` here, which is
    /// the `string` library table.
    string_meta: Rc<RefCell<Table>>,
    /// `package.loaded` — the module cache `require` consults first.
    loaded: Rc<RefCell<Table>>,
    /// How `require` finds a module's source. The host supplies this; kvim maps
    /// a module name to a file under its own config directory.
    module_loader: Option<ModuleLoader>,
    /// Results of the frame that a [`Lua::run`] loop is waiting on.
    returned: Vec<Value>,
    /// Nesting depth of [`Lua::run`]. See [`Thread::base_run_depth`].
    run_depth: usize,
    /// The line currently executing, so errors can say where they happened.
    current_line: u32,
    current_chunk: String,
    /// Where `print` writes. Captured in tests; `stdout` in production.
    pub(crate) output: Option<OutputSink>,
}

impl Default for Lua {
    fn default() -> Self {
        Lua::new()
    }
}

impl Lua {
    /// A new interpreter with the standard library loaded.
    pub fn new() -> Self {
        let mut lua = Lua {
            threads: vec![Thread {
                frames: Vec::new(),
                resumed_by: None,
                base_run_depth: 0,
            }],
            globals: new_table(),
            string_meta: new_table(),
            loaded: new_table(),
            module_loader: None,
            returned: Vec::new(),
            run_depth: 0,
            current_line: 0,
            current_chunk: "?".to_string(),
            output: None,
        };
        crate::stdlib::install(&mut lua);
        lua
    }

    // ---- Public embedding API. ----

    /// Executes a chunk, returning whatever it returned.
    ///
    /// `chunk_name` appears in error messages. Lua's convention is that a leading
    /// `=` means "use this verbatim" and `@` means "this is a filename"; we keep
    /// the convention so messages look like the ones users already know.
    pub fn exec(&mut self, source: &str, chunk_name: &str) -> Result<Vec<Value>> {
        let block = crate::parser::parse(source, chunk_name)?;
        let proto = crate::compiler::compile(&block, chunk_name)?;
        let closure = Rc::new(Closure { proto, upvalues: Vec::new() });
        self.call_closure_at_root(closure, Vec::new())
    }

    /// Evaluates a single expression.
    ///
    /// A convenience over [`Self::exec`]: it wraps the source in `return (...)`,
    /// so the expression is truncated to exactly one value, which is what a caller
    /// asking for *an* expression's value means.
    pub fn eval(&mut self, expr: &str) -> Result<Value> {
        let results = self.exec(&format!("return ({expr})"), "=eval")?;
        Ok(results.into_iter().next().unwrap_or(Value::Nil))
    }

    /// The globals table (`_G`).
    ///
    /// # Why this is not `&mut Table`
    ///
    /// The brief sketched `globals(&mut self) -> &mut Table`. It cannot be done
    /// honestly: `_G` is a *first-class Lua value*, and Lua code can alias it
    /// (`local g = _G`), store it in another table, or set a metatable on it.
    /// It therefore has to live behind `Rc<RefCell<_>>`, and handing out a
    /// `&mut Table` would either be a lie or would force the globals table to be
    /// something Lua could not touch.
    ///
    /// So the handle is returned instead, and [`Self::get_global`] /
    /// [`Self::set_global`] cover the common cases without any ceremony.
    pub fn globals(&self) -> Rc<RefCell<Table>> {
        self.globals.clone()
    }

    pub fn get_global(&self, name: &str) -> Value {
        self.globals.borrow().raw_get_str(name)
    }

    pub fn set_global(&mut self, name: &str, value: Value) {
        self.globals.borrow_mut().set_str(name, value);
    }

    /// Exposes a Rust function to Lua as a global.
    ///
    /// This is the injection point for a host's API surface. The function
    /// receives `&mut Lua`, so it can call back into Lua, read globals, and
    /// build tables.
    ///
    /// ```
    /// # use kopitiam_lua::{Lua, Value};
    /// let mut lua = Lua::new();
    /// lua.set_global_fn("greet", |_lua, args| {
    ///     let who = args.first().and_then(|v| v.as_lua_string())
    ///         .map(|s| s.to_string_lossy())
    ///         .unwrap_or_else(|| "world".to_string());
    ///     Ok(vec![Value::from(format!("hello, {who}"))])
    /// });
    /// assert_eq!(
    ///     lua.eval("greet('kvim')").unwrap().as_lua_string().unwrap().to_string_lossy(),
    ///     "hello, kvim",
    /// );
    /// ```
    pub fn set_global_fn(
        &mut self,
        name: &str,
        f: impl Fn(&mut Lua, Vec<Value>) -> Result<Vec<Value>> + 'static,
    ) {
        let v = self.create_fn(name, f);
        self.set_global(name, v);
    }

    /// Builds a callable Lua value from a Rust closure, without installing it
    /// anywhere. Use this to put a function inside a table — which is how a
    /// namespaced API like `vim.keymap.set` is built.
    ///
    /// ```
    /// # use kopitiam_lua::{Lua, Value};
    /// # use std::rc::Rc;
    /// let mut lua = Lua::new();
    ///
    /// let keymap = lua.create_table();
    /// let set = lua.create_fn("set", |_lua, _args| Ok(vec![]));
    /// keymap.borrow_mut().set_str("set", set);
    ///
    /// let vim = lua.create_table();
    /// vim.borrow_mut().set_str("keymap", Value::Table(keymap));
    /// lua.set_global("vim", Value::Table(vim));
    ///
    /// lua.exec("vim.keymap.set('n', '<leader>e', 'x')", "=cfg").unwrap();
    /// ```
    pub fn create_fn(
        &mut self,
        name: &str,
        f: impl Fn(&mut Lua, Vec<Value>) -> Result<Vec<Value>> + 'static,
    ) -> Value {
        Value::Native(Rc::new(NativeFunction {
            name: name.to_string(),
            kind: NativeKind::Simple(Box::new(f)),
        }))
    }

    /// A fresh, empty table.
    pub fn create_table(&mut self) -> Rc<RefCell<Table>> {
        new_table()
    }

    /// Calls a Lua value from Rust.
    ///
    /// Note this runs a **nested** dispatch loop, so a coroutine cannot yield
    /// across it (see the module docs).
    pub fn call(&mut self, f: &Value, args: Vec<Value>) -> Result<Vec<Value>> {
        self.call_value(f, args)
    }

    /// Sets how `require` resolves a module name to source code.
    ///
    /// The loader returns the module's source, or `None` if it does not have it
    /// (which becomes Lua's `module 'x' not found` error). kvim will point this
    /// at `~/.kopitiam/kopitiam-neovim/lua/<name>.lua`.
    ///
    /// Results are cached in `package.loaded`, so a module is executed at most
    /// once — as Lua guarantees.
    pub fn set_module_loader(&mut self, f: impl Fn(&str) -> Option<String> + 'static) {
        self.module_loader = Some(Rc::new(f));
    }

    /// Redirects `print`. Without this, `print` goes to stdout.
    pub fn set_output(&mut self, f: impl FnMut(&str) + 'static) {
        self.output = Some(Box::new(f));
    }

    /// Lua's `<`, exposed so that `table.sort`'s default comparator is *the same
    /// code* the `<` operator uses — numbers, strings and `__lt` all behave
    /// identically whether you sort or compare. Reimplementing it in the table
    /// library would be a standing invitation for the two to drift apart.
    pub fn less_than_public(&mut self, a: Value, b: Value) -> Result<bool> {
        self.less_than(a, b)
    }

    /// Converts a value to a string exactly as Lua's `tostring` does, honouring
    /// `__tostring`.
    pub fn tostring(&mut self, v: &Value) -> Result<LuaStr> {
        if let Some(h) = self.metamethod(v, "__tostring") {
            let r = self.call_value(&h, vec![v.clone()])?;
            let first = r.into_iter().next().unwrap_or(Value::Nil);
            return Ok(first
                .as_lua_string()
                .unwrap_or_else(|| LuaStr::from(format!("{first:?}"))));
        }
        Ok(match v {
            Value::Nil => LuaStr::from("nil"),
            Value::Boolean(true) => LuaStr::from("true"),
            Value::Boolean(false) => LuaStr::from("false"),
            Value::Number(n) => LuaStr::from(crate::number::format_number(*n)),
            Value::String(s) => s.clone(),
            other => LuaStr::from(format!("{other:?}")),
        })
    }

    /// The `string` library's table, which doubles as every string's `__index`.
    /// The stdlib installs it here at startup.
    pub(crate) fn set_string_metatable_index(&mut self, string_lib: Value) {
        self.string_meta.borrow_mut().set_str("__index", string_lib);
    }

    pub(crate) fn loaded_table(&self) -> Rc<RefCell<Table>> {
        self.loaded.clone()
    }

    pub(crate) fn module_loader(&self) -> Option<ModuleLoader> {
        self.module_loader.clone()
    }

    // ---- Errors. ----

    /// Raises a Lua error from a native function, with a message.
    ///
    /// The current source position is prepended, exactly as Lua's own `error()`
    /// does at level 1 — so a host function that rejects its arguments produces a
    /// message that points at the offending line of the *config*, not at Rust.
    ///
    /// ```
    /// # use kopitiam_lua::{Lua, Value};
    /// let mut lua = Lua::new();
    /// lua.set_global_fn("must_be_positive", |lua, args| {
    ///     match args.first().and_then(|v| v.as_number()) {
    ///         Some(n) if n > 0.0 => Ok(vec![Value::Number(n)]),
    ///         _ => Err(lua.error("expected a positive number")),
    ///     }
    /// });
    /// // ...and Lua code can catch it like any other error.
    /// let caught = lua.eval("select(2, pcall(must_be_positive, -1))").unwrap();
    /// assert!(caught.as_lua_string().unwrap().to_string_lossy().contains("positive"));
    /// ```
    pub fn error(&self, message: impl Into<String>) -> LuaError {
        self.rt(message)
    }

    /// Raises a Lua error carrying an arbitrary **value** rather than a message.
    ///
    /// This is `error({ code = 404 })`. Lua errors are values, and a host that
    /// wants `pcall` to receive a structured error — a table a config can inspect
    /// — needs this rather than a formatted string.
    pub fn error_value(&self, value: Value) -> LuaError {
        self.rt_value(value)
    }

    /// A runtime error located at the instruction currently executing.
    ///
    /// Lua's messages carry `chunk:line:`, and a config that fails without one is
    /// materially harder to fix, so every VM-raised error gets one.
    pub(crate) fn rt(&self, message: impl Into<String>) -> LuaError {
        let msg = format!("{}:{}: {}", self.current_chunk, self.current_line, message.into());
        LuaError::Runtime { value: Value::from(msg), traceback: self.traceback() }
    }

    /// A runtime error carrying an arbitrary value — this is `error(v)`.
    pub(crate) fn rt_value(&self, value: Value) -> LuaError {
        LuaError::Runtime { value, traceback: self.traceback() }
    }

    /// The current source position, as `error()` at level 1 prefixes it.
    pub(crate) fn where_(&self) -> String {
        format!("{}:{}: ", self.current_chunk, self.current_line)
    }

    fn traceback(&self) -> Vec<String> {
        let Some(t) = self.threads.last() else { return Vec::new() };
        t.frames
            .iter()
            .rev()
            .take(16)
            .map(|f| {
                let p = &f.closure.proto;
                let line = p.lines.get(f.pc.saturating_sub(1)).copied().unwrap_or(0);
                format!("in {} ({}:{})", p.name, p.chunk, line)
            })
            .collect()
    }

    // ---- Frame plumbing. ----

    fn thread(&mut self) -> &mut Thread {
        self.threads.last_mut().expect("the main thread always exists")
    }

    fn frame(&mut self) -> &mut Frame {
        self.thread().frames.last_mut().expect("a frame is executing")
    }

    fn push_value(&mut self, v: Value) {
        self.frame().stack.push(v);
    }

    fn pop_value(&mut self) -> Value {
        self.frame().stack.pop().unwrap_or(Value::Nil)
    }

    /// Pops everything above the most recent mark, and the mark with it.
    fn take_marked(&mut self) -> Vec<Value> {
        let f = self.frame();
        let mark = f.marks.pop().unwrap_or(0);
        f.stack.split_off(mark.min(f.stack.len()))
    }

    fn push_frame(&mut self, closure: Rc<Closure>, args: Vec<Value>, ret: ReturnMode) -> Result<()> {
        let total: usize = self.threads.iter().map(|t| t.frames.len()).sum();
        if total >= MAX_FRAMES {
            return Err(self.rt("stack overflow"));
        }

        let proto = &closure.proto;
        let mut slots: Vec<Option<Rc<RefCell<Value>>>> = vec![None; proto.max_slots];

        // Parameters get their cells here, at frame setup: there is no
        // instruction that could run "before the first one" to create them.
        // Missing arguments are nil, surplus ones are dropped -- or collected as
        // varargs if the function declared `...`.
        for (i, slot) in slots.iter_mut().enumerate().take(proto.num_params) {
            let v = args.get(i).cloned().unwrap_or(Value::Nil);
            *slot = Some(Rc::new(RefCell::new(v)));
        }
        let varargs = if proto.is_vararg && args.len() > proto.num_params {
            args[proto.num_params..].to_vec()
        } else {
            Vec::new()
        };

        self.thread().frames.push(Frame {
            closure,
            pc: 0,
            slots,
            stack: Vec::new(),
            marks: Vec::new(),
            varargs,
            ret,
        });
        Ok(())
    }

    /// Adjusts `vals` to the requested count, padding with nil or truncating.
    fn adjust(mut vals: Vec<Value>, n: NRes) -> Vec<Value> {
        match n {
            NRes::All => vals,
            NRes::Exact(n) => {
                let n = n as usize;
                vals.resize(n, Value::Nil);
                vals
            }
        }
    }

    /// Hands a call's results to whoever was waiting for them.
    fn deliver(&mut self, vals: Vec<Value>, ret: ReturnMode) -> Result<()> {
        match ret {
            ReturnMode::Normal(n) => {
                let vals = Self::adjust(vals, n);
                let f = self.frame();
                f.stack.extend(vals);
                Ok(())
            }
            // A protected call that returned normally: `true` first, then the
            // results. `outer` then does the actual delivery -- which is what
            // makes `pcall(pcall, f)` compose correctly.
            ReturnMode::Protected { outer, .. } => {
                let mut out = vec![Value::Boolean(true)];
                out.extend(vals);
                self.deliver(out, *outer)
            }
            ReturnMode::HostRoot => {
                self.returned = vals;
                Ok(())
            }
            ReturnMode::CoroutineRoot => self.finish_coroutine(Ok(vals)),
        }
    }

    // ---- Calling. ----

    /// Performs a call: a Lua closure pushes a frame; a native runs immediately;
    /// anything else goes looking for `__call`.
    ///
    /// # Why native errors need their own catch
    ///
    /// A Lua closure gets a *frame*, and a frame carries its [`ReturnMode`] — so
    /// when it throws, [`Self::unwind`] pops frames until it finds a
    /// `Protected` one. A native pushes **no frame**. If its `Err` were simply
    /// propagated with `?`, it would sail straight past the `Protected` mode this
    /// very call was given, and `pcall(require, "x")` would not be protected at
    /// all. So an erroring native is routed through [`Self::fail_call`], which
    /// honours `ret` directly.
    fn perform_call(&mut self, func: Value, args: Vec<Value>, ret: ReturnMode) -> Result<()> {
        match func {
            Value::Function(cl) => match self.push_frame(cl, args, ret.clone()) {
                Ok(()) => Ok(()),
                // Even pushing a frame can fail (stack overflow), and that error
                // must still be catchable by the pcall that made this call.
                Err(e) => self.fail_call(e, ret),
            },

            // `nf` is an owned `Rc` handle, independent of anything `self` holds,
            // so the native can take `&mut Lua` and freely mutate globals -- even
            // ones that would drop the function it is currently running. The `Rc`
            // keeps it alive for the duration.
            Value::Native(nf) => {
                let outcome = match &nf.kind {
                    NativeKind::Simple(f) => f(self, args).map(Outcome::Return),
                    NativeKind::Vm(f) => f(self, args),
                };
                match outcome {
                    Ok(o) => self.handle_outcome(o, ret),
                    Err(e) => self.fail_call(e, ret),
                }
            }

            other => {
                // `__call` makes a table callable. Lua passes the table itself as
                // the first argument, so `t(1)` becomes `t.__call(t, 1)`.
                let Some(h) = self.metamethod(&other, "__call") else {
                    let e =
                        self.rt(format!("attempt to call a {} value", other.type_name()));
                    return self.fail_call(e, ret);
                };
                let mut a = Vec::with_capacity(args.len() + 1);
                a.push(other);
                a.extend(args);
                self.perform_call(h, a, ret)
            }
        }
    }

    /// A call failed before any frame of its own existed to unwind.
    ///
    /// If the call was protected, this is where `false, err` is produced.
    /// Otherwise the error is propagated so the *frame* unwinder can look for a
    /// protected frame further out.
    fn fail_call(&mut self, err: LuaError, ret: ReturnMode) -> Result<()> {
        match ret {
            ReturnMode::Protected { handler, outer } => {
                self.deliver_error(err, handler, *outer)
            }
            _ => Err(err),
        }
    }

    /// Turns a caught error into the `false, err` a protected call returns,
    /// running the `xpcall` message handler first if there is one.
    fn deliver_error(
        &mut self,
        err: LuaError,
        handler: Option<Value>,
        outer: ReturnMode,
    ) -> Result<()> {
        let mut errval = err.value();
        if let Some(h) = handler {
            errval = match self.call_value(&h, vec![errval]) {
                Ok(v) => v.into_iter().next().unwrap_or(Value::Nil),
                // An error inside the message handler must not recurse forever.
                Err(_) => Value::from("error in error handling"),
            };
        }
        self.deliver(vec![Value::Boolean(false), errval], outer)
    }

    fn handle_outcome(&mut self, outcome: Outcome, ret: ReturnMode) -> Result<()> {
        match outcome {
            Outcome::Return(vals) => self.deliver(vals, ret),
            // `pcall(f, ...)`: run f under a frame whose return mode catches
            // errors. Note this is a VM-level frame, NOT a nested Rust call --
            // which is why a coroutine can yield across a pcall here (Lua 5.2+
            // behaviour; 5.1 forbade it). A strict superset, so no 5.1 program
            // changes meaning.
            Outcome::Protected { f, args, handler } => {
                self.perform_call(f, args, ReturnMode::Protected { handler, outer: Box::new(ret) })
            }
            Outcome::Resume { co, args, wrapped } => self.do_resume(co, args, wrapped, ret),
            Outcome::Yield(vals) => self.do_yield(vals, ret),
        }
    }

    /// Calls a Lua value from Rust, running a nested dispatch loop.
    ///
    /// Used by natives (`table.sort`'s comparator), by metamethods, and by the
    /// host. The nesting is why a yield cannot cross this.
    pub(crate) fn call_value(&mut self, f: &Value, args: Vec<Value>) -> Result<Vec<Value>> {
        let base_thread = self.threads.len() - 1;
        let base_frames = self.threads[base_thread].frames.len();

        self.perform_call(f.clone(), args, ReturnMode::HostRoot)?;

        // A native may have completed without pushing a frame at all, in which
        // case `deliver(HostRoot)` already stashed the results and there is
        // nothing to run.
        if self.threads.len() == base_thread + 1
            && self.threads[base_thread].frames.len() == base_frames
        {
            return Ok(std::mem::take(&mut self.returned));
        }
        self.run(base_thread, base_frames)
    }

    /// Runs a chunk's closure as the bottom of a fresh dispatch loop.
    fn call_closure_at_root(
        &mut self,
        closure: Rc<Closure>,
        args: Vec<Value>,
    ) -> Result<Vec<Value>> {
        let base_thread = self.threads.len() - 1;
        let base_frames = self.threads[base_thread].frames.len();
        self.push_frame(closure, args, ReturnMode::HostRoot)?;
        self.run(base_thread, base_frames)
    }

    // ---- The dispatch loop. ----

    /// Runs until the frame stack drops back to where it started.
    ///
    /// `run_depth` is bumped for the duration. A coroutine records the depth it
    /// was resumed at, and refuses to yield at any other — see [`Thread`].
    fn run(&mut self, base_thread: usize, base_frames: usize) -> Result<Vec<Value>> {
        self.run_depth += 1;
        let result = self.run_inner(base_thread, base_frames);
        self.run_depth -= 1;
        result
    }

    fn run_inner(&mut self, base_thread: usize, base_frames: usize) -> Result<Vec<Value>> {
        loop {
            if self.threads.len() == base_thread + 1
                && self.threads[base_thread].frames.len() == base_frames
            {
                return Ok(std::mem::take(&mut self.returned));
            }
            if let Err(e) = self.step() {
                // `unwind` either finds a protected frame (and returns Ok, having
                // delivered `false, err`) or runs out of frames to unwind within
                // this loop's floor, in which case the error is ours to propagate.
                self.unwind(e, base_thread, base_frames)?;
            }
        }
    }

    /// Pops frames looking for one that can handle the error.
    fn unwind(&mut self, err: LuaError, base_thread: usize, base_frames: usize) -> Result<()> {
        let mut err = err;
        loop {
            let t = self.threads.len() - 1;

            // Never unwind below the floor of the loop we are running for: those
            // frames belong to a caller further out (or to the host).
            if t == base_thread && self.threads[t].frames.len() <= base_frames {
                return Err(err);
            }

            let Some(frame) = self.threads[t].frames.pop() else {
                // An empty coroutine thread with no root frame -- cannot happen,
                // but treat it as fatal rather than looping.
                return Err(err);
            };

            match frame.ret {
                // Nothing here can handle it; keep going.
                ReturnMode::Normal(_) => {}

                // The bottom of a nested dispatch loop. The native that started
                // it must see the error, so hand it back out of `run`.
                ReturnMode::HostRoot => return Err(err),

                // A `pcall`'s frame. The same delivery path an erroring native
                // takes -- see `fail_call`.
                //
                // Note on `xpcall`: real Lua runs the message handler *before*
                // unwinding, so it can inspect the erroring stack. We run it
                // after. With no `debug` library there is nothing observable that
                // depends on it, but it is a real deviation and is stated rather
                // than hidden.
                ReturnMode::Protected { handler, outer } => {
                    return self.deliver_error(err, handler, *outer);
                }

                ReturnMode::CoroutineRoot => {
                    // The coroutine dies. For `wrap`, the error is re-raised in
                    // the resumer and we keep unwinding there; for `resume`, it
                    // becomes `false, err` and unwinding stops.
                    match self.finish_coroutine(Err(err)) {
                        Ok(()) => return Ok(()),
                        Err(e) => {
                            err = e;
                            continue;
                        }
                    }
                }
            }
        }
    }

    fn step(&mut self) -> Result<()> {
        let (instr, line, chunk) = {
            let t = self.threads.len() - 1;
            let f = self.threads[t].frames.last().expect("a frame is executing");
            let pc = f.pc;
            let p: &Proto = &f.closure.proto;
            (p.code[pc], p.lines[pc], p.chunk.clone())
        };
        self.frame().pc += 1;
        self.current_line = line;
        self.current_chunk = chunk;

        match instr {
            Instr::PushNil => self.push_value(Value::Nil),
            Instr::PushTrue => self.push_value(Value::Boolean(true)),
            Instr::PushFalse => self.push_value(Value::Boolean(false)),
            Instr::PushConst(i) => {
                let v = self.frame().closure.proto.consts[i as usize].clone();
                self.push_value(v);
            }
            Instr::PushVarargs => {
                let v = self.frame().varargs.clone();
                self.frame().stack.extend(v);
            }
            Instr::PushVararg1 => {
                let v = self.frame().varargs.first().cloned().unwrap_or(Value::Nil);
                self.push_value(v);
            }

            Instr::NewLocal(slot) => {
                let v = self.pop_value();
                // A *fresh* cell. This one line is why closures capture correctly
                // and why each loop iteration gets its own variable.
                self.frame().slots[slot as usize] = Some(Rc::new(RefCell::new(v)));
            }
            Instr::GetLocal(slot) => {
                let v = match &self.frame().slots[slot as usize] {
                    Some(c) => c.borrow().clone(),
                    // Reading a slot whose declaration has not run yet. The
                    // compiler makes this unreachable, but nil is the honest
                    // answer rather than a panic.
                    None => Value::Nil,
                };
                self.push_value(v);
            }
            Instr::SetLocal(slot) => {
                let v = self.pop_value();
                match &self.frame().slots[slot as usize] {
                    Some(c) => *c.borrow_mut() = v,
                    None => self.frame().slots[slot as usize] = Some(Rc::new(RefCell::new(v))),
                }
            }
            Instr::GetUpval(i) => {
                let v = self.frame().closure.upvalues[i as usize].borrow().clone();
                self.push_value(v);
            }
            Instr::SetUpval(i) => {
                let v = self.pop_value();
                let cell = self.frame().closure.upvalues[i as usize].clone();
                *cell.borrow_mut() = v;
            }
            Instr::GetGlobal(i) => {
                let k = self.frame().closure.proto.consts[i as usize].clone();
                // Through the metatable-aware path, so `setmetatable(_G, ...)`
                // works -- some configs use it to catch typos.
                let g = Value::Table(self.globals.clone());
                let v = self.index(g, k)?;
                self.push_value(v);
            }
            Instr::SetGlobal(i) => {
                let k = self.frame().closure.proto.consts[i as usize].clone();
                let v = self.pop_value();
                let g = Value::Table(self.globals.clone());
                self.set_index(g, k, v)?;
            }

            Instr::Pop(n) => {
                let f = self.frame();
                let keep = f.stack.len().saturating_sub(n as usize);
                f.stack.truncate(keep);
            }
            Instr::Copy(depth) => {
                let f = self.frame();
                let i = f.stack.len() - 1 - depth as usize;
                let v = f.stack[i].clone();
                f.stack.push(v);
            }

            Instr::NewTable => self.push_value(Value::Table(new_table())),

            Instr::GetIndex => {
                let k = self.pop_value();
                let t = self.pop_value();
                let v = self.index(t, k)?;
                self.push_value(v);
            }
            Instr::SetIndex => {
                let v = self.pop_value();
                let k = self.pop_value();
                let t = self.pop_value();
                self.set_index(t, k, v)?;
            }
            Instr::GetField(i) => {
                let k = self.frame().closure.proto.consts[i as usize].clone();
                let t = self.pop_value();
                let v = self.index(t, k)?;
                self.push_value(v);
            }
            Instr::SetField(i) => {
                let k = self.frame().closure.proto.consts[i as usize].clone();
                let v = self.pop_value();
                let t = self.pop_value();
                self.set_index(t, k, v)?;
            }
            Instr::Method(i) => {
                let k = self.frame().closure.proto.consts[i as usize].clone();
                let obj = self.pop_value();
                let f = self.index(obj.clone(), k)?;
                self.push_value(f);
                // The object becomes the implicit first argument -- and it was
                // evaluated exactly once, which is the entire point of `:`.
                self.push_value(obj);
            }

            Instr::RawSetField(i) => {
                let k = self.frame().closure.proto.consts[i as usize].clone();
                let v = self.pop_value();
                self.raw_set_on_peeked_table(k, v)?;
            }
            Instr::RawSetIndex => {
                let v = self.pop_value();
                let k = self.pop_value();
                self.raw_set_on_peeked_table(k, v)?;
            }
            Instr::RawSetArray(n) => {
                let v = self.pop_value();
                self.raw_set_on_peeked_table(Value::Number(n as f64), v)?;
            }
            Instr::SetListOpen(start) => {
                let vals = self.take_marked();
                let t = self.frame().stack.last().cloned().unwrap_or(Value::Nil);
                let Value::Table(t) = t else {
                    return Err(self.rt("internal error: SetListOpen without a table"));
                };
                for (i, v) in vals.into_iter().enumerate() {
                    t.borrow_mut().raw_set(Value::Number((start as usize + i) as f64), v)?;
                }
            }

            Instr::Add => self.binary_arith(ArithOp::Add)?,
            Instr::Sub => self.binary_arith(ArithOp::Sub)?,
            Instr::Mul => self.binary_arith(ArithOp::Mul)?,
            Instr::Div => self.binary_arith(ArithOp::Div)?,
            Instr::Mod => self.binary_arith(ArithOp::Mod)?,
            Instr::Pow => self.binary_arith(ArithOp::Pow)?,

            Instr::Concat => {
                let b = self.pop_value();
                let a = self.pop_value();
                let v = self.concat(a, b)?;
                self.push_value(v);
            }
            Instr::Neg => {
                let a = self.pop_value();
                let v = match a.as_number() {
                    Some(n) => Value::Number(-n),
                    None => match self.metamethod(&a, "__unm") {
                        // Lua passes the operand twice to __unm, for uniformity
                        // with the binary arithmetic metamethods.
                        Some(h) => self.call_1(&h, vec![a.clone(), a])?,
                        None => {
                            return Err(self.rt(format!(
                                "attempt to perform arithmetic on a {} value",
                                a.type_name()
                            )));
                        }
                    },
                };
                self.push_value(v);
            }
            Instr::Not => {
                let a = self.pop_value();
                self.push_value(Value::Boolean(!a.is_truthy()));
            }
            Instr::Len => {
                let a = self.pop_value();
                let v = self.length(a)?;
                self.push_value(v);
            }

            Instr::Eq => {
                let b = self.pop_value();
                let a = self.pop_value();
                let r = self.values_equal(&a, &b)?;
                self.push_value(Value::Boolean(r));
            }
            Instr::Ne => {
                let b = self.pop_value();
                let a = self.pop_value();
                let r = self.values_equal(&a, &b)?;
                self.push_value(Value::Boolean(!r));
            }
            Instr::Lt => {
                let b = self.pop_value();
                let a = self.pop_value();
                let r = self.less_than(a, b)?;
                self.push_value(Value::Boolean(r));
            }
            Instr::Le => {
                let b = self.pop_value();
                let a = self.pop_value();
                let r = self.less_equal(a, b)?;
                self.push_value(Value::Boolean(r));
            }
            // `a > b` IS `b < a` -- including for the `__lt` metamethod, which
            // receives the operands in that swapped order. Lua does the same.
            Instr::Gt => {
                let b = self.pop_value();
                let a = self.pop_value();
                let r = self.less_than(b, a)?;
                self.push_value(Value::Boolean(r));
            }
            Instr::Ge => {
                let b = self.pop_value();
                let a = self.pop_value();
                let r = self.less_equal(b, a)?;
                self.push_value(Value::Boolean(r));
            }

            Instr::Jump(t) => self.frame().pc = t as usize,
            Instr::JumpIfFalse(t) => {
                let v = self.pop_value();
                if !v.is_truthy() {
                    self.frame().pc = t as usize;
                }
            }
            // `a and b`: if `a` is falsy it IS the result, so leave it and skip
            // `b` entirely. Short-circuiting is not an optimisation here -- `f()
            // and g()` must not call g.
            Instr::AndJump(t) => {
                let keep = !self.frame().stack.last().is_some_and(|v| v.is_truthy());
                if keep {
                    self.frame().pc = t as usize;
                } else {
                    self.pop_value();
                }
            }
            Instr::OrJump(t) => {
                let keep = self.frame().stack.last().is_some_and(|v| v.is_truthy());
                if keep {
                    self.frame().pc = t as usize;
                } else {
                    self.pop_value();
                }
            }

            Instr::Mark => {
                let f = self.frame();
                let h = f.stack.len();
                f.marks.push(h);
            }
            Instr::AdjustTo(n) => {
                let vals = self.take_marked();
                let vals = Self::adjust(vals, NRes::Exact(n));
                self.frame().stack.extend(vals);
            }
            Instr::Call(nres) => {
                let mut region = self.take_marked();
                if region.is_empty() {
                    return Err(self.rt("attempt to call a nil value"));
                }
                let func = region.remove(0);
                self.perform_call(func, region, ReturnMode::Normal(nres))?;
            }
            Instr::Return => {
                let vals = self.take_marked();
                let ret = {
                    let t = self.threads.len() - 1;
                    let frame = self.threads[t].frames.pop().expect("a frame is executing");
                    frame.ret
                };
                self.deliver(vals, ret)?;
            }

            Instr::Closure(i) => {
                let (proto, upvals) = {
                    let f = self.frame();
                    let p = f.closure.proto.protos[i as usize].clone();
                    // Capture the CELLS, not the values. Sharing the `Rc` is what
                    // makes the capture by-reference.
                    let ups = p
                        .upvals
                        .iter()
                        .map(|d| match d.source {
                            UpvalSource::ParentLocal(slot) => f.slots[slot as usize]
                                .clone()
                                .unwrap_or_else(|| Rc::new(RefCell::new(Value::Nil))),
                            UpvalSource::ParentUpval(idx) => {
                                f.closure.upvalues[idx as usize].clone()
                            }
                        })
                        .collect();
                    (p, ups)
                };
                self.push_value(Value::Function(Rc::new(Closure { proto, upvalues: upvals })));
            }

            Instr::ForPrep { base, target } => {
                let step = self.pop_value();
                let limit = self.pop_value();
                let start = self.pop_value();

                let (start, limit, step) = match (
                    start.as_number(),
                    limit.as_number(),
                    step.as_number(),
                ) {
                    (Some(a), Some(b), Some(c)) => (a, b, c),
                    _ => return Err(self.rt("'for' initial value must be a number")),
                };
                if step == 0.0 {
                    return Err(self.rt("'for' step must be non-zero"));
                }

                // Pre-decrement so that `ForLoop`'s unconditional increment lands
                // on the first value. One branch instead of two.
                let cells = [start - step, limit, step];
                for (i, v) in cells.into_iter().enumerate() {
                    self.frame().slots[base as usize + i] =
                        Some(Rc::new(RefCell::new(Value::Number(v))));
                }
                self.frame().pc = target as usize;
            }
            Instr::ForLoop { base, target } => {
                let get = |f: &Frame, i: usize| -> f64 {
                    match &f.slots[base as usize + i] {
                        Some(c) => match &*c.borrow() {
                            Value::Number(n) => *n,
                            _ => 0.0,
                        },
                        None => 0.0,
                    }
                };
                let f = self.frame();
                let (idx, limit, step) = (get(f, 0), get(f, 1), get(f, 2));
                let next = idx + step;

                let go = if step > 0.0 { next <= limit } else { next >= limit };
                if go {
                    if let Some(c) = &self.frame().slots[base as usize] {
                        *c.borrow_mut() = Value::Number(next);
                    }
                    // Push the value for the body's `NewLocal`, which will wrap it
                    // in a FRESH cell -- so a closure made this iteration captures
                    // this iteration's variable, and not a shared one.
                    self.push_value(Value::Number(next));
                    self.frame().pc = target as usize;
                }
            }
            Instr::GenForPrep { base } => {
                let ctl = self.pop_value();
                let state = self.pop_value();
                let iter = self.pop_value();
                for (i, v) in [iter, state, ctl].into_iter().enumerate() {
                    self.frame().slots[base as usize + i] = Some(Rc::new(RefCell::new(v)));
                }
            }
            Instr::GenForTest { base, nvars, target } => {
                let first = {
                    let f = self.frame();
                    let at = f.stack.len() - nvars as usize;
                    f.stack[at].clone()
                };
                if matches!(first, Value::Nil) {
                    // The iterator is exhausted: drop its results and leave.
                    let f = self.frame();
                    let keep = f.stack.len() - nvars as usize;
                    f.stack.truncate(keep);
                    self.frame().pc = target as usize;
                } else {
                    // The first result becomes the next control value.
                    if let Some(c) = &self.frame().slots[base as usize + 2] {
                        *c.borrow_mut() = first;
                    }
                }
            }
        }
        Ok(())
    }

    /// Raw-sets a key on the table sitting on top of the stack, leaving it there.
    /// Table constructors use this: the table is brand new, so `__newindex`
    /// cannot apply.
    fn raw_set_on_peeked_table(&mut self, k: Value, v: Value) -> Result<()> {
        let t = self.frame().stack.last().cloned().unwrap_or(Value::Nil);
        let Value::Table(t) = t else {
            return Err(self.rt("internal error: table constructor lost its table"));
        };
        t.borrow_mut().raw_set(k, v)?;
        Ok(())
    }

    // ---- Metamethods and operators. ----

    /// Looks up a metamethod. Strings share one metatable, so `("x"):upper()`
    /// resolves without every string carrying a pointer.
    pub(crate) fn metamethod(&self, v: &Value, name: &str) -> Option<Value> {
        let mt = match v {
            Value::Table(t) => t.borrow().metatable.clone()?,
            Value::String(_) => self.string_meta.clone(),
            _ => return None,
        };
        let h = mt.borrow().raw_get_str(name);
        if matches!(h, Value::Nil) { None } else { Some(h) }
    }

    /// Calls a value and keeps only its first result — what a metamethod
    /// invocation wants.
    fn call_1(&mut self, f: &Value, args: Vec<Value>) -> Result<Value> {
        Ok(self.call_value(f, args)?.into_iter().next().unwrap_or(Value::Nil))
    }

    /// `t[k]`, honouring `__index` — which may be a table (chain to it) or a
    /// function (call it).
    pub(crate) fn index(&mut self, obj: Value, key: Value) -> Result<Value> {
        let mut current = obj;

        for _ in 0..MAX_METATABLE_CHAIN {
            let handler = match &current {
                Value::Table(t) => {
                    let raw = t.borrow().raw_get(&key);
                    if !matches!(raw, Value::Nil) {
                        return Ok(raw);
                    }
                    let Some(mt) = t.borrow().metatable.clone() else {
                        // A plain table with no metatable: a missing key is nil,
                        // and that is not an error.
                        return Ok(Value::Nil);
                    };
                    let h = mt.borrow().raw_get_str("__index");
                    if matches!(h, Value::Nil) {
                        return Ok(Value::Nil);
                    }
                    h
                }
                Value::String(_) => {
                    let h = self.string_meta.borrow().raw_get_str("__index");
                    if matches!(h, Value::Nil) {
                        return Err(self.rt("attempt to index a string value"));
                    }
                    h
                }
                other => {
                    return Err(self.rt(format!(
                        "attempt to index a {} value",
                        other.type_name()
                    )));
                }
            };

            match handler {
                // `__index = function(t, k)` -- call it and take its answer.
                f @ (Value::Function(_) | Value::Native(_)) => {
                    return self.call_1(&f, vec![current, key]);
                }
                // `__index = someTable` -- look there instead. Looping rather
                // than recursing means a long chain costs no Rust stack.
                other => current = other,
            }
        }
        Err(self.rt("'__index' chain too long; possible loop"))
    }

    /// `t[k] = v`, honouring `__newindex`.
    ///
    /// Note the trigger: `__newindex` fires only when the key is **absent**.
    /// Overwriting an existing key is a plain raw set, and a proxy table that
    /// forgets this appears to work until the first update.
    pub(crate) fn set_index(&mut self, obj: Value, key: Value, val: Value) -> Result<()> {
        let mut current = obj;

        for _ in 0..MAX_METATABLE_CHAIN {
            let handler = match &current {
                Value::Table(t) => {
                    let present = !matches!(t.borrow().raw_get(&key), Value::Nil);
                    let mt = t.borrow().metatable.clone();

                    let h = match mt {
                        Some(mt) if !present => mt.borrow().raw_get_str("__newindex"),
                        _ => Value::Nil,
                    };
                    if matches!(h, Value::Nil) {
                        t.borrow_mut().raw_set(key, val)?;
                        return Ok(());
                    }
                    h
                }
                other => {
                    return Err(self.rt(format!(
                        "attempt to index a {} value",
                        other.type_name()
                    )));
                }
            };

            match handler {
                f @ (Value::Function(_) | Value::Native(_)) => {
                    self.call_value(&f, vec![current, key, val])?;
                    return Ok(());
                }
                other => current = other,
            }
        }
        Err(self.rt("'__newindex' chain too long; possible loop"))
    }

    fn binary_arith(&mut self, op: ArithOp) -> Result<()> {
        let b = self.pop_value();
        let a = self.pop_value();
        let v = self.arith(op, a, b)?;
        self.push_value(v);
        Ok(())
    }

    fn arith(&mut self, op: ArithOp, a: Value, b: Value) -> Result<Value> {
        // Lua coerces strings to numbers in arithmetic: `"10" + 1` is 11.
        // `as_number` is where that lives.
        if let (Some(x), Some(y)) = (a.as_number(), b.as_number()) {
            return Ok(Value::Number(op.apply(x, y)));
        }
        let event = op.event();
        if let Some(h) = self.metamethod(&a, event).or_else(|| self.metamethod(&b, event)) {
            return self.call_1(&h, vec![a, b]);
        }
        // Report the operand that is actually at fault.
        let bad = if a.as_number().is_none() { &a } else { &b };
        Err(self.rt(format!(
            "attempt to perform arithmetic on a {} value",
            bad.type_name()
        )))
    }

    fn concat(&mut self, a: Value, b: Value) -> Result<Value> {
        // Numbers concatenate as their string form: `1 .. "x"` is `"1x"`.
        if let (Some(x), Some(y)) = (a.as_lua_string(), b.as_lua_string()) {
            let mut out = Vec::with_capacity(x.len() + y.len());
            out.extend_from_slice(x.as_bytes());
            out.extend_from_slice(y.as_bytes());
            return Ok(Value::String(LuaStr::from_bytes(&out)));
        }
        if let Some(h) =
            self.metamethod(&a, "__concat").or_else(|| self.metamethod(&b, "__concat"))
        {
            return self.call_1(&h, vec![a, b]);
        }
        let bad = if a.as_lua_string().is_none() { &a } else { &b };
        Err(self.rt(format!("attempt to concatenate a {} value", bad.type_name())))
    }

    fn length(&mut self, a: Value) -> Result<Value> {
        // `__len` on a table is Lua 5.2+; stock 5.1 honours it only for userdata.
        // We honour it, which is a superset: no 5.1 program sets `__len` on a
        // table and expects it to be ignored.
        if let Some(h) = self.metamethod(&a, "__len") {
            return self.call_1(&h, vec![a]);
        }
        match a {
            Value::String(s) => Ok(Value::Number(s.len() as f64)),
            Value::Table(t) => Ok(Value::Number(t.borrow().raw_len() as f64)),
            other => Err(self.rt(format!(
                "attempt to get length of a {} value",
                other.type_name()
            ))),
        }
    }

    fn values_equal(&mut self, a: &Value, b: &Value) -> Result<bool> {
        if a.raw_eq(b) {
            return Ok(true);
        }
        // `__eq` is consulted only when both operands are tables and are not
        // already equal. Values of different types are never equal, and never
        // reach a metamethod -- `1 == "1"` is false, full stop.
        if let (Value::Table(_), Value::Table(_)) = (a, b)
            && let Some(h) =
                self.metamethod(a, "__eq").or_else(|| self.metamethod(b, "__eq"))
        {
            let r = self.call_1(&h, vec![a.clone(), b.clone()])?;
            return Ok(r.is_truthy());
        }
        Ok(false)
    }

    fn less_than(&mut self, a: Value, b: Value) -> Result<bool> {
        match (&a, &b) {
            (Value::Number(x), Value::Number(y)) => return Ok(x < y),
            // Strings compare byte-lexicographically.
            (Value::String(x), Value::String(y)) => return Ok(x.as_bytes() < y.as_bytes()),
            _ => {}
        }
        if let Some(h) = self.metamethod(&a, "__lt").or_else(|| self.metamethod(&b, "__lt")) {
            let r = self.call_1(&h, vec![a, b])?;
            return Ok(r.is_truthy());
        }
        Err(self.rt(format!(
            "attempt to compare {} with {}",
            a.type_name(),
            b.type_name()
        )))
    }

    fn less_equal(&mut self, a: Value, b: Value) -> Result<bool> {
        match (&a, &b) {
            (Value::Number(x), Value::Number(y)) => return Ok(x <= y),
            (Value::String(x), Value::String(y)) => return Ok(x.as_bytes() <= y.as_bytes()),
            _ => {}
        }
        if let Some(h) = self.metamethod(&a, "__le").or_else(|| self.metamethod(&b, "__le")) {
            let r = self.call_1(&h, vec![a, b])?;
            return Ok(r.is_truthy());
        }
        // Lua 5.1's documented fallback: `a <= b` becomes `not (b < a)` when only
        // `__lt` is defined. It is only valid for a total order, which is why 5.4
        // dropped it -- but 5.1 has it, and configs written against 5.1 may rely
        // on defining `__lt` alone.
        if let Some(h) = self.metamethod(&a, "__lt").or_else(|| self.metamethod(&b, "__lt")) {
            let r = self.call_1(&h, vec![b, a])?;
            return Ok(!r.is_truthy());
        }
        Err(self.rt(format!(
            "attempt to compare {} with {}",
            a.type_name(),
            b.type_name()
        )))
    }

    // ---- Coroutines. ----

    /// The coroutine currently running, if any.
    pub(crate) fn current_coroutine(&self) -> Option<Rc<RefCell<Coroutine>>> {
        self.threads.last()?.resumed_by.as_ref().map(|i| i.co.clone())
    }

    fn do_resume(
        &mut self,
        co: Rc<RefCell<Coroutine>>,
        args: Vec<Value>,
        wrapped: bool,
        ret: ReturnMode,
    ) -> Result<()> {
        let status = co.borrow().status;
        if status != CoStatus::Suspended {
            let msg = format!("cannot resume {} coroutine", status.as_str());
            // `wrap` raises; `resume` reports. That asymmetry is Lua's, and it is
            // the whole behavioural difference between the two.
            if wrapped {
                return Err(self.rt(msg));
            }
            return self.deliver(vec![Value::Boolean(false), Value::from(msg)], ret);
        }

        // The resumer is now waiting on us.
        if let Some(info) = self.threads.last().and_then(|t| t.resumed_by.as_ref()) {
            info.co.borrow_mut().status = CoStatus::Normal;
        }

        let started = !co.borrow().frames.is_empty();
        let base_run_depth = self.run_depth;

        if started {
            // Restore the parked frames and hand the resume arguments to the
            // `yield(...)` that is still sitting there waiting for them.
            let (frames, yield_ret) = {
                let mut c = co.borrow_mut();
                c.status = CoStatus::Running;
                (std::mem::take(&mut c.frames), c.yield_ret.take())
            };
            self.threads.push(Thread {
                frames,
                resumed_by: Some(ResumeInfo { co, ret, wrapped }),
                base_run_depth,
            });
            let mode = yield_ret.unwrap_or(ReturnMode::Normal(NRes::All));
            self.deliver(args, mode)?;
        } else {
            let entry = {
                let mut c = co.borrow_mut();
                c.status = CoStatus::Running;
                c.entry.clone()
            };
            let Some(entry) = entry else {
                return Err(self.rt("coroutine has no function to run"));
            };
            self.threads.push(Thread {
                frames: Vec::new(),
                resumed_by: Some(ResumeInfo { co, ret, wrapped }),
                base_run_depth,
            });
            // The body's bottom frame: when it returns, the coroutine is done.
            self.perform_call(entry, args, ReturnMode::CoroutineRoot)?;
        }
        Ok(())
    }

    fn do_yield(&mut self, vals: Vec<Value>, ret: ReturnMode) -> Result<()> {
        if self.threads.len() == 1 {
            return Err(self.rt("attempt to yield from outside a coroutine"));
        }
        // The C-call boundary. If a native started a nested dispatch loop since
        // this coroutine was resumed, there is Rust stack between here and the
        // resume, and lifting the thread out would strand it. Real Lua 5.1 fails
        // here for the same reason, with the same message.
        if self.threads.last().expect("checked len > 1").base_run_depth != self.run_depth {
            return Err(self.rt("attempt to yield across a C-call boundary"));
        }

        let thread = self.threads.pop().expect("checked len > 1");
        let info = thread.resumed_by.expect("a non-main thread was resumed by someone");
        {
            let mut co = info.co.borrow_mut();
            co.status = CoStatus::Suspended;
            co.frames = thread.frames;
            // Remember where the *next* resume's arguments should be delivered:
            // straight back into this `yield(...)` call site.
            co.yield_ret = Some(ret);
        }
        self.restore_resumer_status();

        let out = if info.wrapped {
            vals
        } else {
            let mut v = vec![Value::Boolean(true)];
            v.extend(vals);
            v
        };
        self.deliver(out, info.ret)
    }

    fn finish_coroutine(&mut self, result: Result<Vec<Value>>) -> Result<()> {
        let thread = self.threads.pop().expect("a CoroutineRoot implies a coroutine thread");
        let info = thread.resumed_by.expect("a coroutine thread was resumed by someone");
        {
            let mut co = info.co.borrow_mut();
            co.status = CoStatus::Dead;
            co.frames.clear();
            co.entry = None;
        }
        self.restore_resumer_status();

        match result {
            Ok(vals) => {
                let out = if info.wrapped {
                    vals
                } else {
                    let mut v = vec![Value::Boolean(true)];
                    v.extend(vals);
                    v
                };
                self.deliver(out, info.ret)
            }
            // `wrap` propagates the error into the resumer; `resume` reports it.
            Err(e) => {
                if info.wrapped {
                    // The error is thrown in the resumer -- but the call that
                    // resumed us may itself have been protected (`pcall(gen)`),
                    // and the `wrap` native pushed no frame, so there is nothing
                    // for the frame unwinder to find. The protection lives in
                    // `info.ret`, so honour it here rather than dropping it.
                    self.fail_call(e, info.ret)
                } else {
                    self.deliver(vec![Value::Boolean(false), e.value()], info.ret)
                }
            }
        }
    }

    fn restore_resumer_status(&mut self) {
        if let Some(info) = self.threads.last().and_then(|t| t.resumed_by.as_ref()) {
            info.co.borrow_mut().status = CoStatus::Running;
        }
    }
}

/// The arithmetic operators, and their metamethod names.
#[derive(Clone, Copy)]
pub(crate) enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
}

impl ArithOp {
    fn apply(self, a: f64, b: f64) -> f64 {
        match self {
            ArithOp::Add => a + b,
            ArithOp::Sub => a - b,
            ArithOp::Mul => a * b,
            // Lua 5.1 has one number type, so `/` is ALWAYS float division:
            // `7 / 2` is 3.5, never 3.
            ArithOp::Div => a / b,
            // Lua's `%` is FLOORED, not truncated: `a - floor(a/b)*b`. So
            // `-1 % 3` is 2 in Lua, where Rust's `%` would give -1. Using Rust's
            // operator here would be wrong in a way that only shows up on
            // negative operands -- exactly the kind of silent divergence that
            // makes a config misbehave rather than fail.
            ArithOp::Mod => a - (a / b).floor() * b,
            ArithOp::Pow => a.powf(b),
        }
    }

    fn event(self) -> &'static str {
        match self {
            ArithOp::Add => "__add",
            ArithOp::Sub => "__sub",
            ArithOp::Mul => "__mul",
            ArithOp::Div => "__div",
            ArithOp::Mod => "__mod",
            ArithOp::Pow => "__pow",
        }
    }
}
