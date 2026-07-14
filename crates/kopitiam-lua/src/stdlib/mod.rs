//! The standard library.
//!
//! Scoped to **what a Neovim config actually touches** — `base`, `string`,
//! `table`, `math`, `coroutine`, and `require`. There is no `io`, no `os`, and
//! no `debug`. That is a deliberate boundary, not an oversight: an editor's
//! config has no business opening files or spawning processes, and a Lua VM
//! embedded in an editor is a place where a *small* attack surface is worth more
//! than completeness. Adding them later is easy; taking them away is not.
//!
//! Deviations from stock Lua 5.1 are noted at each function that has one. There
//! are three worth knowing about up front:
//!
//! * `math.random` is **deterministic** by default (a fixed seed). CLAUDE.md
//!   requires deterministic behaviour, and Lua 5.1's own `math.random` is
//!   equally deterministic until you call `math.randomseed`.
//! * `require` resolves modules through a **host-supplied loader**
//!   ([`crate::Lua::set_module_loader`]) rather than a filesystem `package.path`.
//!   kvim points it at its own config directory; nothing here reads the disk on
//!   its own initiative.
//! * `pairs` iterates in **insertion order**, deterministically. Real Lua's order
//!   is unspecified and varies between runs.

use std::cell::RefCell;
use std::rc::Rc;

use crate::error::{LuaError, Result};
use crate::value::{LuaStr, NativeFunction, NativeKind, Outcome, Table, Value};
use crate::vm::Lua;

mod base;
mod coroutine_lib;
mod math_lib;
mod string_lib;
mod table_lib;

/// Installs every library into a fresh interpreter.
pub(crate) fn install(lua: &mut Lua) {
    // `_G` refers to the globals table itself, so `_G.x` and `x` are the same
    // variable. It has to be a genuine self-reference, not a copy.
    let g = lua.globals();
    let gv = Value::Table(g.clone());
    g.borrow_mut().set_str("_G", gv);
    g.borrow_mut().set_str("_VERSION", Value::from("Lua 5.1"));

    base::install(lua);
    string_lib::install(lua);
    table_lib::install(lua);
    math_lib::install(lua);
    coroutine_lib::install(lua);
}

// ---- Helpers shared across the libraries. ----

/// A plain Rust function exposed to Lua. A bare `fn` pointer rather than a
/// closure, so a library's function table can be a `const`-shaped slice.
pub(crate) type LibFn = fn(&mut Lua, Vec<Value>) -> Result<Vec<Value>>;

/// Builds a library table and installs it as a global, in one step.
pub(crate) fn make_lib(
    lua: &mut Lua,
    name: &str,
    fns: &[(&str, LibFn)],
) -> Rc<RefCell<Table>> {
    let t = lua.create_table();
    for (fname, f) in fns {
        let v = lua.create_fn(fname, *f);
        t.borrow_mut().set_str(fname, v);
    }
    lua.set_global(name, Value::Table(t.clone()));
    t
}

/// A native that needs to ask the VM for something only the VM can do — start a
/// protected call, resume a coroutine, yield. See [`Outcome`].
pub(crate) fn vm_fn(
    name: &str,
    f: impl Fn(&mut Lua, Vec<Value>) -> Result<Outcome> + 'static,
) -> Value {
    Value::Native(Rc::new(NativeFunction {
        name: name.to_string(),
        kind: NativeKind::Vm(Box::new(f)),
    }))
}

pub(crate) fn arg(args: &[Value], i: usize) -> Value {
    args.get(i).cloned().unwrap_or(Value::Nil)
}

/// Lua's own wording: `bad argument #1 to 'sub' (number expected, got string)`.
/// Worth matching exactly — it is what users will search for.
fn bad_arg(lua: &Lua, i: usize, fname: &str, expected: &str, got: &Value) -> LuaError {
    lua.rt(format!(
        "bad argument #{} to '{}' ({} expected, got {})",
        i + 1,
        fname,
        expected,
        if matches!(got, Value::Nil) { "no value" } else { got.type_name() }
    ))
}

pub(crate) fn check_number(lua: &Lua, args: &[Value], i: usize, fname: &str) -> Result<f64> {
    let v = arg(args, i);
    // Lua coerces a numeric string here: `string.rep("x", "3")` works.
    v.as_number().ok_or_else(|| bad_arg(lua, i, fname, "number", &v))
}

/// A number that must be a whole one — an index or a count.
pub(crate) fn check_int(lua: &Lua, args: &[Value], i: usize, fname: &str) -> Result<i64> {
    Ok(check_number(lua, args, i, fname)? as i64)
}

pub(crate) fn opt_int(
    lua: &Lua,
    args: &[Value],
    i: usize,
    fname: &str,
    default: i64,
) -> Result<i64> {
    match args.get(i) {
        None | Some(Value::Nil) => Ok(default),
        Some(_) => check_int(lua, args, i, fname),
    }
}

pub(crate) fn check_string(lua: &Lua, args: &[Value], i: usize, fname: &str) -> Result<LuaStr> {
    let v = arg(args, i);
    // Numbers coerce to strings, so `("x"):rep(3)` and `string.len(42)` both work.
    v.as_lua_string().ok_or_else(|| bad_arg(lua, i, fname, "string", &v))
}

pub(crate) fn check_table(
    lua: &Lua,
    args: &[Value],
    i: usize,
    fname: &str,
) -> Result<Rc<RefCell<Table>>> {
    match arg(args, i) {
        Value::Table(t) => Ok(t),
        other => Err(bad_arg(lua, i, fname, "table", &other)),
    }
}

pub(crate) fn check_function(lua: &Lua, args: &[Value], i: usize, fname: &str) -> Result<Value> {
    let v = arg(args, i);
    match v {
        Value::Function(_) | Value::Native(_) => Ok(v),
        other => Err(bad_arg(lua, i, fname, "function", &other)),
    }
}

/// Turns a Lua 1-based index — which may be negative, meaning "from the end" —
/// into a 0-based offset into a byte string.
///
/// This one function is the source of most off-by-ones in string libraries, so
/// it exists once, is used everywhere, and is tested directly.
///
/// ```text
/// len = 5 ("hello")
///   1 ->  1   (first byte)
///  -1 ->  5   (last byte)
///  -5 ->  1
///  -9 ->  0   (clamped: further back than the start)
///   0 ->  0
/// ```
pub(crate) fn str_index(pos: i64, len: usize) -> i64 {
    if pos >= 0 {
        pos
    } else if (-pos) as usize > len {
        0
    } else {
        len as i64 + pos + 1
    }
}
