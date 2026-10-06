//! Builds the `vim.*` table and installs it as a global.
//!
//! Every function here is a native Rust closure registered into the VM. They all
//! share one [`VimState`] through an `Rc<RefCell<_>>`, so `vim.opt.number = true`
//! and `vim.keymap.set(...)` and `vim.cmd.colorscheme(...)` all write into the
//! same growing [`Config`](crate::config::Config).
//!
//! The metamethod argument conventions match the VM's (and the scale-model test
//! in `kopitiam-lua/tests/maintainer_config.rs`): `__newindex` receives
//! `(table, key, value)`, `__index` receives `(table, key)`, `__call` receives
//! `(table, ...args)`. A plainly-called native like `vim.keymap.set` receives its
//! arguments starting at index 0.

use std::cell::RefCell;
use std::rc::Rc;

use kopitiam_lua::{Lua, Value};

use crate::config::Action;

/// A Lua chunk that returns a "black hole": a table where every field access
/// yields the same table and every call yields the same table, so
/// `bh.a.b.c(...)` and `bh:setup{...}` all resolve harmlessly to `bh`. `bh` is a
/// captured upvalue (declared `local` then assigned), not a global, so building
/// several black holes never leaks or aliases through the globals table.
const BLACK_HOLE_SRC: &str = "\
    local bh\n\
    bh = setmetatable({}, {\n\
        __index = function() return bh end,\n\
        __call = function() return bh end,\n\
    })\n\
    return bh";

use super::{
    Autocmd, Discovered, UserCommand, VimState, apply_option, lstr, lua_to_string, mode_arg,
    opts_string, record_keymap, resolve_rhs,
};

/// The functions kvim hands out for plugins it implements natively. A config
/// that does `require('telescope.builtin').find_files` gets one of these back,
/// and [`resolve_rhs`] recognises it by identity when it appears as a keymap
/// right-hand side — mapping it onto the corresponding native [`Action`] instead
/// of storing it as an opaque Lua callback.
struct Markers {
    lsp_definition: Value,
    lsp_references: Value,
    lsp_rename: Value,
    lsp_hover: Value,
    find_files: Value,
    buffers: Value,
    help_tags: Value,
    hint_words: Value,
    harpoon_add: Value,
    harpoon_menu: Value,
}

impl Markers {
    /// Every marker paired with the [`Action`] it stands for, for
    /// identity-based lookup in [`resolve_rhs`].
    fn pairs(&self) -> Vec<(Value, Action)> {
        vec![
            (self.lsp_definition.clone(), Action::LspDefinition),
            (self.lsp_references.clone(), Action::LspReferences),
            (self.lsp_rename.clone(), Action::LspRename),
            (self.lsp_hover.clone(), Action::LspHover),
            (self.find_files.clone(), Action::FindFiles),
            (self.buffers.clone(), Action::FindBuffers),
            (self.help_tags.clone(), Action::FindHelp),
            (self.hint_words.clone(), Action::HopWords),
            (self.harpoon_add.clone(), Action::HarpoonAdd),
            (self.harpoon_menu.clone(), Action::HarpoonMenu),
        ]
    }
}

/// A no-op native: the body of every marker. Passed around as a value, never
/// meant to *do* anything if actually called (e.g. from inside a Lua keymap
/// closure) — so calling it is harmless and returns nothing.
fn noop(lua: &mut Lua, name: &str) -> Value {
    lua.create_fn(name, |_, _| Ok(vec![]))
}

fn build_markers(lua: &mut Lua) -> Markers {
    Markers {
        lsp_definition: noop(lua, "vim.lsp.buf.definition"),
        lsp_references: noop(lua, "vim.lsp.buf.references"),
        lsp_rename: noop(lua, "vim.lsp.buf.rename"),
        lsp_hover: noop(lua, "vim.lsp.buf.hover"),
        find_files: noop(lua, "telescope.builtin.find_files"),
        buffers: noop(lua, "telescope.builtin.buffers"),
        help_tags: noop(lua, "telescope.builtin.help_tags"),
        hint_words: noop(lua, "hop.hint_words"),
        harpoon_add: noop(lua, "harpoon.mark.add_file"),
        harpoon_menu: noop(lua, "harpoon.ui.toggle_quick_menu"),
    }
}

/// A value that absorbs any use: `bh.anything` is `bh`, `bh(...)` is `bh`,
/// `bh.a.b.c()` is `bh`. This is what an unsupported `vim.*` field and an
/// unknown `require` resolve to, so plugin-manager boilerplate
/// (`require("lazy").setup{...}`) runs to completion without crashing.
fn build_black_hole(lua: &mut Lua) -> Value {
    match lua.exec(BLACK_HOLE_SRC, "@blackhole") {
        Ok(mut vs) if !vs.is_empty() => vs.remove(0),
        _ => Value::Nil,
    }
}

/// Installs `vim` and the module loader. The single public entry.
pub(super) fn install(lua: &mut Lua, state: &Rc<RefCell<VimState>>, discovered: &Discovered) {
    let markers = Rc::new(build_markers(lua));
    let bh = build_black_hole(lua);

    let vim = lua.create_table();

    // vim.opt / vim.opt_local / vim.opt_global — the full option object.
    for key in ["opt", "opt_local", "opt_global"] {
        let t = make_opt_table(lua, state);
        vim.borrow_mut().set_str(key, Value::Table(t));
    }
    // vim.o / vim.wo / vim.bo / vim.go — the scalar shorthands. Same effect on
    // assignment; reads return nil (kvim does not round-trip them yet).
    for key in ["o", "wo", "bo", "go"] {
        let t = make_scalar_opt_table(lua, state);
        vim.borrow_mut().set_str(key, Value::Table(t));
    }

    // vim.g / vim.b / vim.w — variable namespaces.
    vim.borrow_mut().set_str("g", Value::Table(make_var_table(lua, state, VarScope::Global)));
    vim.borrow_mut().set_str("b", Value::Table(make_var_table(lua, state, VarScope::Buffer)));
    vim.borrow_mut().set_str("w", Value::Table(make_var_table(lua, state, VarScope::Window)));

    vim.borrow_mut().set_str("keymap", Value::Table(make_keymap_table(lua, state, &markers)));
    vim.borrow_mut().set_str("cmd", Value::Table(make_cmd_table(lua, state)));
    vim.borrow_mut().set_str("api", Value::Table(make_api_table(lua, state, &bh)));
    vim.borrow_mut().set_str("fn", Value::Table(make_fn_table(lua, state)));
    vim.borrow_mut().set_str("lsp", Value::Table(make_lsp_table(lua, &markers)));
    vim.borrow_mut().set_str("log", Value::Table(make_log_table(lua)));

    install_misc(lua, &vim, state);
    install_vim_fallback(lua, &vim, state, &bh);

    lua.set_global("vim", Value::Table(vim));

    preseed_modules(lua, &markers);
    install_loader(lua, state, discovered);
}

// --------------------------------------------------------------------------
// vim.opt / vim.o — options
// --------------------------------------------------------------------------

fn make_opt_table(lua: &mut Lua, state: &Rc<RefCell<VimState>>) -> Rc<RefCell<kopitiam_lua::Table>> {
    let opt = lua.create_table();
    let mt = lua.create_table();

    // `vim.opt.number = true` — assignment applies the option.
    let st = state.clone();
    let newindex = lua.create_fn("vim.opt.__newindex", move |lua, args| {
        let name = args.get(1).and_then(|v| v.as_lua_string()).map(|s| s.to_string_lossy());
        if let (Some(name), Some(value)) = (name, args.get(2)) {
            let as_str = lua_to_string(lua, value);
            let mut s = st.borrow_mut();
            if apply_option(&mut s.config.options, &name, value, &as_str).is_err() {
                s.warn(format!("vim.opt.{name}: option not modelled by kvim (ignored)"));
            }
        }
        Ok(vec![])
    });

    // `vim.opt.clipboard:append("unnamedplus")` — reading returns an entry object
    // carrying methods that read-modify-write the comma-list option.
    let st = state.clone();
    let index = lua.create_fn("vim.opt.__index", move |lua, args| {
        let name = args.get(1).and_then(|v| v.as_lua_string()).map(|s| s.to_string_lossy());
        let Some(name) = name else { return Ok(vec![Value::Nil]) };
        Ok(vec![Value::Table(make_opt_entry(lua, &st, &name))])
    });

    mt.borrow_mut().set_str("__newindex", newindex);
    mt.borrow_mut().set_str("__index", index);
    opt.borrow_mut().metatable = Some(mt);
    opt
}

/// The object `vim.opt.<name>` returns: it carries `:append`, `:prepend` and
/// `:remove` for comma-list options, plus `:get()`.
fn make_opt_entry(
    lua: &mut Lua,
    state: &Rc<RefCell<VimState>>,
    name: &str,
) -> Rc<RefCell<kopitiam_lua::Table>> {
    let entry = lua.create_table();
    for (method, op) in [("append", ListOp::Append), ("prepend", ListOp::Prepend), ("remove", ListOp::Remove)] {
        let st = state.clone();
        let name = name.to_string();
        let f = lua.create_fn(&format!("vim.opt.{name}:{method}"), move |lua, args| {
            // args[0] is the entry (self); the value is args[1].
            if let Some(value) = args.get(1) {
                let piece = lua_to_string(lua, value);
                list_modify(&st, &name, &piece, op);
            }
            Ok(vec![])
        });
        entry.borrow_mut().set_str(method, f);
    }
    entry
}

#[derive(Clone, Copy)]
enum ListOp {
    Append,
    Prepend,
    Remove,
}

/// Read-modify-write a comma-list string option (`clipboard`, `spelllang`).
fn list_modify(state: &Rc<RefCell<VimState>>, name: &str, piece: &str, op: ListOp) {
    let mut st = state.borrow_mut();
    let Some(current) = option_string(&st.config.options, name) else {
        st.warn(format!("vim.opt.{name}:append/remove is only supported for comma-list options"));
        return;
    };
    let mut items: Vec<String> = current.split(',').filter(|s| !s.is_empty()).map(str::to_string).collect();
    match op {
        ListOp::Append => items.push(piece.to_string()),
        ListOp::Prepend => items.insert(0, piece.to_string()),
        ListOp::Remove => items.retain(|s| s != piece),
    }
    let joined = items.join(",");
    let _ = apply_option(&mut st.config.options, name, &lstr(&joined), &joined);
}

/// The current value of a string-valued option, or `None` for non-string
/// options (where list operations make no sense).
fn option_string(opts: &crate::config::Options, name: &str) -> Option<String> {
    match name {
        "clipboard" | "cb" => Some(opts.clipboard.clone()),
        "spelllang" | "spl" => Some(opts.spelllang.clone()),
        _ => None,
    }
}

/// `vim.o` / `vim.wo` / `vim.bo`: assignment applies the option; reads are nil.
fn make_scalar_opt_table(
    lua: &mut Lua,
    state: &Rc<RefCell<VimState>>,
) -> Rc<RefCell<kopitiam_lua::Table>> {
    let t = lua.create_table();
    let mt = lua.create_table();
    let st = state.clone();
    let newindex = lua.create_fn("vim.o.__newindex", move |lua, args| {
        let name = args.get(1).and_then(|v| v.as_lua_string()).map(|s| s.to_string_lossy());
        if let (Some(name), Some(value)) = (name, args.get(2)) {
            let as_str = lua_to_string(lua, value);
            let mut s = st.borrow_mut();
            if apply_option(&mut s.config.options, &name, value, &as_str).is_err() {
                s.warn(format!("vim.o.{name}: option not modelled by kvim (ignored)"));
            }
        }
        Ok(vec![])
    });
    mt.borrow_mut().set_str("__newindex", newindex);
    t.borrow_mut().metatable = Some(mt);
    t
}

// --------------------------------------------------------------------------
// vim.g / vim.b / vim.w — variables
// --------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum VarScope {
    Global,
    Buffer,
    Window,
}

fn make_var_table(
    lua: &mut Lua,
    state: &Rc<RefCell<VimState>>,
    scope: VarScope,
) -> Rc<RefCell<kopitiam_lua::Table>> {
    let t = lua.create_table();
    let mt = lua.create_table();

    let st = state.clone();
    let newindex = lua.create_fn("vim.g.__newindex", move |lua, args| {
        let name = args.get(1).and_then(|v| v.as_lua_string()).map(|s| s.to_string_lossy());
        if let (Some(name), Some(value)) = (name, args.get(2)) {
            let as_str = lua_to_string(lua, value);
            let mut s = st.borrow_mut();
            match scope {
                VarScope::Global => {
                    // The load-bearing one: `vim.g.mapleader = " "`. kvim's leader
                    // is a single char, so take the first character of the value.
                    // `vim.g.mapleader = " "` drives the leader; an ordinary
                    // global just gets recorded below.
                    let is_leader = name == "mapleader" || name == "maplocalleader";
                    if let Some(c) = as_str.chars().next().filter(|_| is_leader) {
                        s.config.leader = c;
                    }
                    s.globals.insert(name, as_str);
                }
                VarScope::Buffer => {
                    s.buffer_vars.insert(name, as_str);
                }
                VarScope::Window => {} // recorded nowhere useful yet
            }
        }
        Ok(vec![])
    });

    // Reading a var hands back what was stored, as a string, so a config that
    // does `if vim.g.foo then` after setting it sees the value.
    let st = state.clone();
    let index = lua.create_fn("vim.g.__index", move |_lua, args| {
        let name = args.get(1).and_then(|v| v.as_lua_string()).map(|s| s.to_string_lossy());
        let Some(name) = name else { return Ok(vec![Value::Nil]) };
        let s = st.borrow();
        let store = match scope {
            VarScope::Buffer => &s.buffer_vars,
            _ => &s.globals,
        };
        Ok(vec![store.get(&name).map(|v| lstr(v)).unwrap_or(Value::Nil)])
    });

    mt.borrow_mut().set_str("__newindex", newindex);
    mt.borrow_mut().set_str("__index", index);
    t.borrow_mut().metatable = Some(mt);
    t
}

// --------------------------------------------------------------------------
// vim.keymap.set
// --------------------------------------------------------------------------

fn make_keymap_table(
    lua: &mut Lua,
    state: &Rc<RefCell<VimState>>,
    markers: &Rc<Markers>,
) -> Rc<RefCell<kopitiam_lua::Table>> {
    let keymap = lua.create_table();

    let st = state.clone();
    let mk = markers.clone();
    let set = lua.create_fn("vim.keymap.set", move |_lua, args| {
        let mode = mode_arg(args.first());
        let lhs = args.get(1).and_then(|v| v.as_lua_string()).map(|s| s.to_string_lossy());
        let Some(lhs) = lhs else { return Ok(vec![]) };
        let rhs = args.get(2).cloned().unwrap_or(Value::Nil);
        let desc = opts_string(args.get(3), "desc");
        let buffer_local = matches!(args.get(3), Some(Value::Table(t)) if t.borrow().raw_get_str("buffer").is_truthy());

        let action = resolve_rhs(&st, &mk.pairs(), &rhs);
        // A buffer-local map still binds; kvim has no per-buffer keymap scope
        // yet, so it applies globally and notes the approximation.
        if buffer_local {
            st.borrow_mut().warn(format!(
                "vim.keymap.set: `{lhs}` is buffer-local; applied globally (kvim has no per-buffer maps yet)"
            ));
        }
        record_keymap(&st, mode, lhs, action, desc);
        Ok(vec![])
    });
    keymap.borrow_mut().set_str("set", set);

    // vim.keymap.del: accept and no-op — nothing crashes if a config unbinds.
    let del = lua.create_fn("vim.keymap.del", |_, _| Ok(vec![]));
    keymap.borrow_mut().set_str("del", del);
    keymap
}

// --------------------------------------------------------------------------
// vim.cmd — callable and indexable
// --------------------------------------------------------------------------

fn make_cmd_table(lua: &mut Lua, state: &Rc<RefCell<VimState>>) -> Rc<RefCell<kopitiam_lua::Table>> {
    let cmd = lua.create_table();
    let mt = lua.create_table();

    // `vim.cmd([[ ... ]])` — a whole block; split into lines and apply each.
    let st = state.clone();
    let call = lua.create_fn("vim.cmd.__call", move |_lua, args| {
        if let Some(block) = args.get(1).and_then(|v| v.as_lua_string()) {
            for line in block.to_string_lossy().lines() {
                super::excmd::apply_line(&st, line);
            }
        }
        Ok(vec![])
    });

    // `vim.cmd.colorscheme("gruvbox")` — index gives a function that runs
    // `colorscheme <arg>`.
    let st = state.clone();
    let index = lua.create_fn("vim.cmd.__index", move |lua, args| {
        let name = args.get(1).and_then(|v| v.as_lua_string()).map(|s| s.to_string_lossy());
        let Some(name) = name else { return Ok(vec![Value::Nil]) };
        let st = st.clone();
        Ok(vec![lua.create_fn("vim.cmd.subcommand", move |lua, args| {
            let arg = args.first().map(|v| lua_to_string(lua, v)).unwrap_or_default();
            super::excmd::apply_line(&st, &format!("{name} {arg}"));
            Ok(vec![])
        })])
    });

    mt.borrow_mut().set_str("__call", call);
    mt.borrow_mut().set_str("__index", index);
    cmd.borrow_mut().metatable = Some(mt);
    cmd
}

// --------------------------------------------------------------------------
// vim.api.* — the nvim_ functions
// --------------------------------------------------------------------------

fn make_api_table(
    lua: &mut Lua,
    state: &Rc<RefCell<VimState>>,
    bh: &Value,
) -> Rc<RefCell<kopitiam_lua::Table>> {
    let api = lua.create_table();
    let mt = lua.create_table();

    let st = state.clone();
    let bh = bh.clone();
    let index = lua.create_fn("vim.api.__index", move |lua, args| {
        let name = args.get(1).and_then(|v| v.as_lua_string()).map(|s| s.to_string_lossy());
        let Some(name) = name else { return Ok(vec![Value::Nil]) };
        Ok(vec![api_function(lua, &st, &bh, &name)])
    });

    mt.borrow_mut().set_str("__index", index);
    api.borrow_mut().metatable = Some(mt);
    api
}

/// Builds the native for a specific `vim.api.nvim_*` name. The ones kvim maps
/// return a real closure; everything else returns a black-hole so a chained call
/// (`vim.api.nvim_get_current_buf()`) does not crash.
fn api_function(lua: &mut Lua, state: &Rc<RefCell<VimState>>, bh: &Value, name: &str) -> Value {
    let st = state.clone();
    match name {
        "nvim_set_keymap" => lua.create_fn(name, move |_lua, args| {
            // (mode, lhs, rhs, opts)
            let mode = mode_arg(args.first());
            let lhs = args.get(1).and_then(|v| v.as_lua_string()).map(|s| s.to_string_lossy());
            if let Some(lhs) = lhs {
                let rhs = args.get(2).cloned().unwrap_or(Value::Nil);
                let desc = opts_string(args.get(3), "desc");
                let action = resolve_rhs(&st, &[], &rhs);
                record_keymap(&st, mode, lhs, action, desc);
            }
            Ok(vec![])
        }),
        "nvim_buf_set_keymap" => lua.create_fn(name, move |_lua, args| {
            // (buffer, mode, lhs, rhs, opts) — buffer ignored.
            let mode = mode_arg(args.get(1));
            let lhs = args.get(2).and_then(|v| v.as_lua_string()).map(|s| s.to_string_lossy());
            if let Some(lhs) = lhs {
                let rhs = args.get(3).cloned().unwrap_or(Value::Nil);
                let desc = opts_string(args.get(4), "desc");
                let action = resolve_rhs(&st, &[], &rhs);
                record_keymap(&st, mode, lhs, action, desc);
            }
            Ok(vec![])
        }),
        "nvim_create_autocmd" => lua.create_fn(name, move |lua, args| {
            let events = event_list(args.first());
            let (pattern, action) = match args.get(1) {
                Some(Value::Table(t)) => {
                    let t = t.borrow();
                    let pattern = t.raw_get_str("pattern").as_lua_string().map(|s| s.to_string_lossy()).unwrap_or_default();
                    let cmd = t.raw_get_str("command");
                    let action = if let Some(s) = cmd.as_lua_string() {
                        s.to_string_lossy()
                    } else if t.raw_get_str("callback").is_truthy() {
                        "<lua callback>".to_string()
                    } else {
                        String::new()
                    };
                    (pattern, action)
                }
                _ => (String::new(), String::new()),
            };
            let _ = lua;
            st.borrow_mut().autocmds.push(Autocmd { events, pattern, action });
            Ok(vec![])
        }),
        "nvim_create_user_command" => lua.create_fn(name, move |lua, args| {
            let cmd_name = args.first().and_then(|v| v.as_lua_string()).map(|s| s.to_string_lossy()).unwrap_or_default();
            let definition = args.get(1).map(|v| lua_to_string(lua, v)).unwrap_or_default();
            st.borrow_mut().user_commands.push(UserCommand { name: cmd_name, definition });
            Ok(vec![])
        }),
        "nvim_command" | "nvim_cmd" | "nvim_exec" | "nvim_exec2" => lua.create_fn(name, move |lua, args| {
            if let Some(v) = args.first() {
                let text = lua_to_string(lua, v);
                for line in text.lines() {
                    super::excmd::apply_line(&st, line);
                }
            }
            Ok(vec![])
        }),
        "nvim_set_hl" => lua.create_fn(name, move |_lua, _args| {
            st.borrow_mut().warn(
                "vim.api.nvim_set_hl: highlight overrides are not applied (kvim themes are data)".to_string(),
            );
            Ok(vec![])
        }),
        // Anything else: a black hole, plus a one-line note.
        _ => {
            st.borrow_mut().warn(format!("vim.api.{name} is not supported (no-op)"));
            bh.clone()
        }
    }
}

/// Normalises `nvim_create_autocmd`'s event argument (a string or a list of
/// strings) into a `Vec<String>`.
fn event_list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => vec![s.to_string_lossy()],
        Some(Value::Table(t)) => t
            .borrow()
            .array()
            .iter()
            .filter_map(|v| v.as_lua_string().map(|s| s.to_string_lossy()))
            .collect(),
        _ => Vec::new(),
    }
}

// --------------------------------------------------------------------------
// vim.fn.*
// --------------------------------------------------------------------------

fn make_fn_table(lua: &mut Lua, state: &Rc<RefCell<VimState>>) -> Rc<RefCell<kopitiam_lua::Table>> {
    let fnt = lua.create_table();
    let mt = lua.create_table();
    let st = state.clone();
    let index = lua.create_fn("vim.fn.__index", move |lua, args| {
        let name = args.get(1).and_then(|v| v.as_lua_string()).map(|s| s.to_string_lossy());
        let Some(name) = name else { return Ok(vec![Value::Nil]) };
        Ok(vec![fn_function(lua, &st, &name)])
    });
    mt.borrow_mut().set_str("__index", index);
    fnt.borrow_mut().metatable = Some(mt);
    fnt
}

fn fn_function(lua: &mut Lua, state: &Rc<RefCell<VimState>>, name: &str) -> Value {
    let st = state.clone();
    match name {
        // `vim.fn.stdpath("config")` → kvim's config dir. kvim keeps one dir, so
        // data/cache/state all resolve there too.
        "stdpath" => lua.create_fn(name, move |_lua, _args| {
            let dir = crate::config::Config::dir()
                .map(|d| d.to_string_lossy().to_string())
                .unwrap_or_default();
            Ok(vec![lstr(&dir)])
        }),
        // `vim.fn.expand("~/x")` → expand a leading ~ and drop the vim `%`/`<...>`
        // specials (which have no meaning at config-load time).
        "expand" => lua.create_fn(name, move |lua, args| {
            let raw = args.first().map(|v| lua_to_string(lua, v)).unwrap_or_default();
            let expanded = if let Some(rest) = raw.strip_prefix('~') {
                match std::env::var_os("HOME") {
                    Some(home) => format!("{}{}", home.to_string_lossy(), rest),
                    None => raw,
                }
            } else {
                raw
            };
            Ok(vec![lstr(&expanded)])
        }),
        // `vim.fn.has("nvim")` → 1 for the features kvim honestly provides.
        "has" => lua.create_fn(name, move |lua, args| {
            let what = args.first().map(|v| lua_to_string(lua, v)).unwrap_or_default();
            let yes = matches!(what.as_str(), "nvim" | "unix" | "lua");
            Ok(vec![Value::Number(if yes { 1.0 } else { 0.0 })])
        }),
        // `vim.fn.executable("rg")` → 1 if it is on PATH.
        "executable" => lua.create_fn(name, move |lua, args| {
            let exe = args.first().map(|v| lua_to_string(lua, v)).unwrap_or_default();
            Ok(vec![Value::Number(if on_path(&exe) { 1.0 } else { 0.0 })])
        }),
        "getcwd" => lua.create_fn(name, move |_lua, _args| {
            let cwd = std::env::current_dir().map(|d| d.to_string_lossy().to_string()).unwrap_or_default();
            Ok(vec![lstr(&cwd)])
        }),
        "empty" => lua.create_fn(name, move |_lua, args| {
            let e = match args.first() {
                None | Some(Value::Nil) => true,
                Some(Value::String(s)) => s.is_empty(),
                Some(Value::Number(n)) => *n == 0.0,
                Some(Value::Table(t)) => t.borrow().raw_len() == 0,
                _ => false,
            };
            Ok(vec![Value::Number(if e { 1.0 } else { 0.0 })])
        }),
        _ => {
            let n = name.to_string();
            lua.create_fn(name, move |_lua, _args| {
                st.borrow_mut().warn(format!("vim.fn.{n} is not implemented (returned nil)"));
                Ok(vec![Value::Nil])
            })
        }
    }
}

/// Cheap PATH lookup for `vim.fn.executable`.
fn on_path(exe: &str) -> bool {
    if exe.is_empty() {
        return false;
    }
    let Some(path) = std::env::var_os("PATH") else { return false };
    std::env::split_paths(&path).any(|dir| dir.join(exe).is_file())
}

// --------------------------------------------------------------------------
// vim.lsp.buf.*, vim.log.levels, and the misc top-level functions
// --------------------------------------------------------------------------

fn make_lsp_table(lua: &mut Lua, markers: &Rc<Markers>) -> Rc<RefCell<kopitiam_lua::Table>> {
    let buf = lua.create_table();
    buf.borrow_mut().set_str("definition", markers.lsp_definition.clone());
    buf.borrow_mut().set_str("references", markers.lsp_references.clone());
    buf.borrow_mut().set_str("rename", markers.lsp_rename.clone());
    buf.borrow_mut().set_str("hover", markers.lsp_hover.clone());
    let lsp = lua.create_table();
    lsp.borrow_mut().set_str("buf", Value::Table(buf));
    lsp
}

fn make_log_table(lua: &mut Lua) -> Rc<RefCell<kopitiam_lua::Table>> {
    let levels = lua.create_table();
    for (name, n) in [("TRACE", 0), ("DEBUG", 1), ("INFO", 2), ("WARN", 3), ("ERROR", 4), ("OFF", 5)] {
        levels.borrow_mut().set_str(name, Value::Number(n as f64));
    }
    let log = lua.create_table();
    log.borrow_mut().set_str("levels", Value::Table(levels));
    log
}

/// `vim.notify`, `vim.schedule`, and the other loose top-level functions.
fn install_misc(lua: &mut Lua, vim: &Rc<RefCell<kopitiam_lua::Table>>, state: &Rc<RefCell<VimState>>) {
    let st = state.clone();
    let notify = lua.create_fn("vim.notify", move |lua, args| {
        let msg = args.first().map(|v| lua_to_string(lua, v)).unwrap_or_default();
        st.borrow_mut().notifications.push(msg);
        Ok(vec![])
    });
    vim.borrow_mut().set_str("notify", notify);

    // A once-per-message variant; kvim treats it the same.
    let st = state.clone();
    let notify_once = lua.create_fn("vim.notify_once", move |lua, args| {
        let msg = args.first().map(|v| lua_to_string(lua, v)).unwrap_or_default();
        let mut s = st.borrow_mut();
        if !s.notifications.contains(&msg) {
            s.notifications.push(msg);
        }
        Ok(vec![])
    });
    vim.borrow_mut().set_str("notify_once", notify_once);

    // `vim.schedule(fn)` — defer until the config finishes.
    let st = state.clone();
    let schedule = lua.create_fn("vim.schedule", move |_lua, args| {
        if let Some(f @ (Value::Function(_) | Value::Native(_))) = args.first() {
            st.borrow_mut().scheduled.push(f.clone());
        }
        Ok(vec![])
    });
    vim.borrow_mut().set_str("schedule", schedule);

    // `vim.schedule_wrap(fn)` — return a function that schedules fn when called.
    // Approximated as identity: kvim runs scheduled work synchronously right
    // after load, so wrapping is not observably different for a config.
    let schedule_wrap = lua.create_fn("vim.schedule_wrap", move |_lua, args| {
        Ok(vec![args.into_iter().next().unwrap_or(Value::Nil)])
    });
    vim.borrow_mut().set_str("schedule_wrap", schedule_wrap);

    // `vim.inspect(v)` → a string, for configs that log with it.
    let inspect = lua.create_fn("vim.inspect", move |lua, args| {
        let s = args.first().map(|v| lua_to_string(lua, v)).unwrap_or_default();
        Ok(vec![lstr(&s)])
    });
    vim.borrow_mut().set_str("inspect", inspect);
}

/// The metatable on `vim` itself: any field kvim did not install resolves to a
/// black hole and records a one-line warning, so an unsupported `vim.<x>` never
/// crashes the config.
fn install_vim_fallback(
    lua: &mut Lua,
    vim: &Rc<RefCell<kopitiam_lua::Table>>,
    state: &Rc<RefCell<VimState>>,
    bh: &Value,
) {
    let mt = lua.create_table();
    let st = state.clone();
    let bh = bh.clone();
    let index = lua.create_fn("vim.__index", move |_lua, args| {
        if let Some(name) = args.get(1).and_then(|v| v.as_lua_string()) {
            st.borrow_mut().warn(format!("vim.{} is not supported yet (no-op)", name.to_string_lossy()));
        }
        Ok(vec![bh.clone()])
    });
    mt.borrow_mut().set_str("__index", index);
    vim.borrow_mut().metatable = Some(mt);
}

// --------------------------------------------------------------------------
// require: modules on disk, native plugin markers, black-hole stubs
// --------------------------------------------------------------------------

/// Seeds `package.loaded` with the plugin modules kvim implements natively, so
/// `require('telescope.builtin').find_files` returns a marker rather than a
/// black hole — preserving the keymap → native-action mapping.
fn preseed_modules(lua: &mut Lua, markers: &Rc<Markers>) {
    let telescope_builtin = lua.create_table();
    telescope_builtin.borrow_mut().set_str("find_files", markers.find_files.clone());
    telescope_builtin.borrow_mut().set_str("git_files", markers.find_files.clone());
    telescope_builtin.borrow_mut().set_str("buffers", markers.buffers.clone());
    telescope_builtin.borrow_mut().set_str("help_tags", markers.help_tags.clone());

    let telescope = lua.create_table();
    telescope.borrow_mut().set_str("load_extension", lua.create_fn("telescope.load_extension", |_, _| Ok(vec![])));
    telescope.borrow_mut().set_str("setup", lua.create_fn("telescope.setup", |_, _| Ok(vec![])));

    let hop = lua.create_table();
    hop.borrow_mut().set_str("hint_words", markers.hint_words.clone());
    hop.borrow_mut().set_str("setup", lua.create_fn("hop.setup", |_, _| Ok(vec![])));

    let harpoon_mark = lua.create_table();
    harpoon_mark.borrow_mut().set_str("add_file", markers.harpoon_add.clone());
    let harpoon_ui = lua.create_table();
    harpoon_ui.borrow_mut().set_str("toggle_quick_menu", markers.harpoon_menu.clone());

    if let Ok(Value::Table(loaded)) = lua.eval("package.loaded") {
        let mut l = loaded.borrow_mut();
        l.set_str("telescope.builtin", Value::Table(telescope_builtin));
        l.set_str("telescope", Value::Table(telescope));
        l.set_str("hop", Value::Table(hop));
        l.set_str("harpoon.mark", Value::Table(harpoon_mark));
        l.set_str("harpoon.ui", Value::Table(harpoon_ui));
    }
}

/// The `require` resolver: user modules from disk first, then a black-hole stub
/// for anything unknown — which is how plugin-manager boilerplate degrades.
fn install_loader(lua: &mut Lua, state: &Rc<RefCell<VimState>>, discovered: &Discovered) {
    let modules = discovered.modules.clone();
    let st = state.clone();
    lua.set_module_loader(move |name| {
        if let Some(src) = modules.get(name) {
            return Some(src.clone());
        }
        // Unknown module → a black-hole table, recorded as a stubbed plugin so
        // startup can report what it swallowed.
        let mut s = st.borrow_mut();
        if !s.stubbed_plugins.iter().any(|p| p == name) {
            s.stubbed_plugins.push(name.to_string());
        }
        Some(BLACK_HOLE_SRC.to_string())
    });
}
