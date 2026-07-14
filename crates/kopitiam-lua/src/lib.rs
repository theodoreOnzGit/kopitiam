//! # kopitiam-lua — a pure-Rust Lua 5.1 interpreter
//!
//! No C. No bindings. No vendored Lua. This crate is the load-bearing proof of
//! KOPITIAM's **Pure Rust Core** commitment: the whole thing builds with `cargo
//! build` on stable Rust, with one pure-Rust dependency (`indexmap`, justified in
//! `Cargo.toml`).
//!
//! ## Why it exists
//!
//! `kopitiam-neovim` (kvim) is a modal editor that already *discovers* the user's
//! Lua config but cannot *execute* it, because there was no interpreter. This is
//! the interpreter. The target dialect is **Lua 5.1** — the dialect Neovim's
//! LuaJIT implements and that every plugin is written against. Not 5.3, whose
//! integer/float split would silently change division, `%`, and table-key
//! identity.
//!
//! ## Architecture, in one breath
//!
//! ```text
//! source -> lexer -> parser -> AST -> compiler -> bytecode -> VM
//!           ^^^^^    ^^^^^^           ^^^^^^^^               ^^
//!           lexer.rs parser.rs        compiler.rs            vm.rs
//! ```
//!
//! The VM is a **stack machine with an explicit frame stack**, not a
//! tree-walker. That is not a performance decision — it is the only way to get
//! working coroutines, because a coroutine needs a saveable "where am I", and in
//! a tree-walker "where am I" *is* the Rust call stack, which `yield` would have
//! to destroy in order to escape. The full argument, the alternatives considered,
//! and what would make it wrong are recorded in `docs/ai-decisions/AID-0007`.
//!
//! It is deliberately **unoptimised**: every local variable is heap-allocated in
//! its own cell, there is no constant folding, and there is no register
//! allocation. Correct before fast.
//!
//! ## Quick start
//!
//! ```
//! use kopitiam_lua::Lua;
//!
//! let mut lua = Lua::new();
//! lua.exec(r#"
//!     local function fib(n)
//!         if n < 2 then return n end
//!         return fib(n - 1) + fib(n - 2)
//!     end
//!     result = fib(10)
//! "#, "=demo").unwrap();
//!
//! assert_eq!(lua.get_global("result").as_number(), Some(55.0));
//! ```
//!
//! ## Embedding: injecting a Rust API
//!
//! This is what kvim does to provide `vim.opt`, `vim.keymap` and the rest — and
//! it is the case the API is shaped around.
//!
//! ```
//! use kopitiam_lua::{Lua, Value};
//! use std::cell::RefCell;
//! use std::rc::Rc;
//!
//! let mut lua = Lua::new();
//!
//! // Record every keymap the config sets.
//! let recorded: Rc<RefCell<Vec<String>>> = Rc::default();
//!
//! let sink = recorded.clone();
//! let set = lua.create_fn("set", move |_lua, args| {
//!     let lhs = args.get(1).and_then(|v| v.as_lua_string()).unwrap();
//!     sink.borrow_mut().push(lhs.to_string_lossy());
//!     Ok(vec![])
//! });
//!
//! let keymap = lua.create_table();
//! keymap.borrow_mut().set_str("set", set);
//! let vim = lua.create_table();
//! vim.borrow_mut().set_str("keymap", Value::Table(keymap));
//! lua.set_global("vim", Value::Table(vim));
//!
//! lua.exec(r#"vim.keymap.set("n", "<leader>e", "<cmd>Neotree toggle<cr>")"#, "=cfg").unwrap();
//!
//! assert_eq!(recorded.borrow().as_slice(), ["<leader>e"]);
//! ```
//!
//! ## What is and is not here
//!
//! Implemented: the full Lua 5.1 grammar; metatables (`__index`, `__newindex`,
//! `__call`, `__tostring`, `__eq`, `__lt`, `__le`, the arithmetic events,
//! `__concat`, `__len`, `__metatable`); closures with by-reference upvalue
//! capture; multiple returns and multiple assignment; varargs; `pcall`/`error`
//! with arbitrary error values; **coroutines**; Lua patterns; and the `string`,
//! `table` and `math` libraries.
//!
//! Deliberately absent: `io`, `os`, `debug`, and `load`/`loadstring`/`dofile`.
//! An editor config has no business opening files or spawning processes, and a
//! small attack surface is worth more here than completeness. `require` goes
//! through a host-supplied loader ([`Lua::set_module_loader`]) rather than
//! searching a `package.path`, so the host decides what is reachable.
//!
//! ## Known gaps — things a Lua 5.1 program may legitimately expect and not find
//!
//! Stated here rather than discovered later. A known gap is survivable; a
//! silently broken one is not.
//!
//! * **`setfenv` / `getfenv` are not implemented.** They are genuine Lua 5.1
//!   functions. Supporting them means giving every closure its own environment
//!   table instead of one shared globals table, which is a real change to the
//!   value model rather than a missing builtin. Configs rarely use them; some
//!   older plugins do. Calling them yields `attempt to call a nil value`, which
//!   is at least loud.
//! * **Reference cycles are never freed.** Memory is reference-counted, not
//!   traced, so `t.self = t` leaks. `collectgarbage` exists but is a documented
//!   no-op — there is no collector to run. Fine for a config executed once at
//!   startup; not fine for a long-running process churning cyclic data. See
//!   `stdlib::base::collectgarbage`.
//! * **No `goto`.** Correct: `goto` is Lua 5.2. A 5.1 program cannot contain one.
//! * **Performance is unoptimised on purpose.** Every local is a heap allocation;
//!   strings are content-compared rather than interned. Correct before fast.
//!
//! Known deviations from stock 5.1 — all of them supersets or determinism
//! improvements, and each documented at its definition:
//!
//! * `pairs` iterates in insertion order, deterministically (real Lua's order is
//!   unspecified and varies from run to run).
//! * A coroutine may yield across a `pcall` (5.1 forbids it; 5.2 allows it).
//! * `__len` is honoured on tables (5.2 behaviour; 5.1 honours it only for
//!   userdata).
//! * `\xNN` string escapes are accepted (a LuaJIT / 5.2 extension).
//! * `table.sort` is stable (real Lua's quicksort is not).
//! * `xpcall`'s message handler runs *after* unwinding rather than before. With
//!   no `debug` library, nothing observable depends on this.
//!
//! Errors: [`LuaError`] carries a [`Value`], not a string, because Lua's
//! `error()` can throw any value and `pcall` can observe it.

#![forbid(unsafe_code)]

mod ast;
mod compiler;
mod error;
mod instr;
mod lexer;
mod number;
mod parser;
mod pattern;
mod stdlib;
mod value;
mod vm;

pub use error::{LuaError, Result};
pub use value::{CoStatus, Coroutine, LuaStr, Table, Value};
pub use vm::Lua;

/// Parses a chunk without running it — a syntax check.
///
/// Useful to a host that wants to tell a user their config is broken *before*
/// half-executing it and leaving the editor in a partly-configured state.
///
/// ```
/// assert!(kopitiam_lua::check_syntax("local x = 1", "=ok").is_ok());
/// assert!(kopitiam_lua::check_syntax("local x =", "=bad").is_err());
/// ```
pub fn check_syntax(source: &str, chunk_name: &str) -> Result<()> {
    parser::parse(source, chunk_name)?;
    Ok(())
}
