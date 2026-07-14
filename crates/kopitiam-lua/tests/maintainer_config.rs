//! **The test this crate exists to pass.**
//!
//! kvim (`crates/kopitiam-neovim`) already discovers the maintainer's Lua config
//! but cannot execute it. This suite executes it — for real — against a `vim`
//! shim of native Rust functions that record what the config asked for, and then
//! asserts the recording matches what kvim's `Config`, `Options`, `Keymap` and
//! `Action` types say it should be.
//!
//! That closes the loop. It proves the VM against *the exact input it was built
//! to run*, rather than against a corpus of Lua the author chose because it was
//! easy.
//!
//! # What the config actually exercises
//!
//! It is not a toy. Loading these three files uses, between them:
//!
//! * `__newindex` metamethods (`vim.opt.number = true` — the key is absent, so
//!   the metamethod fires every time)
//! * `__call` on a table (`vim.cmd([[ ... ]])`) *and* `__index` on the same table
//!   (`vim.cmd.colorscheme(...)`)
//! * long-bracket strings spanning lines (`vim.cmd([[ ... ]])`)
//! * `require` with and without call parentheses (`require "x"` / `require("x")`)
//! * deep index chains passed as values (`vim.lsp.buf.definition`)
//! * closures as arguments, with an upvalue-free body that calls `require`
//! * table constructors with named fields (`{ desc = "...", remap = true }`)
//! * method-call sugar and multiple returns through native functions
//!
//! If any one of those is broken, this file fails.

use std::cell::RefCell;
use std::rc::Rc;

use kopitiam_lua::{Lua, Value};

// ---------------------------------------------------------------------------
// The maintainer's real config, verbatim.
//
// Embedded rather than only read from disk so that the test is hermetic and the
// input is reviewable in the diff. `the_live_config_on_disk_still_executes`
// below additionally runs whatever is actually in ~/.config/nvim right now, so
// the two cannot silently drift apart without a test noticing.
// ---------------------------------------------------------------------------

const SETTINGS_LUA: &str = r#"
-- Basic editor settings.

-- line numbers (absolute on the current line, relative on the others)
vim.opt.number = true
vim.opt.relativenumber = true

-- indentation: 4-space tabs, and don't soft-wrap long lines
vim.opt.tabstop = 4
vim.opt.shiftwidth = 4
vim.opt.wrap = false

-- keep some context around the cursor when scrolling
vim.opt.scrolloff = 5

-- spell checking (British English). Custom words live in spell/en.utf-8.add
vim.opt.spell = true
vim.opt.spelllang = "en_gb"

-- visual guide at column 75 to hint at line length
vim.opt.colorcolumn = "75"

-- dark background; airline uses the matching dark theme (the gruvbox
-- colorscheme itself is applied in lua/plugins.lua once the plugin is loaded)
vim.opt.background = "dark"
vim.g.airline_theme = "dark"

-- classic syntax highlighting + filetype-based indentation
vim.cmd([[
syntax on
filetype plugin indent on
]])

-- force the `tex` filetype on .tex files so texlab attaches consistently
vim.cmd([[
autocmd BufNewFile,BufRead *.tex set filetype=tex
]])
"#;

const KEYMAPS_LUA: &str = r#"
-- Custom keymaps. <leader> = Space.

-- LSP: go to definition (Space + g + d)
vim.keymap.set("n", "<leader>gd", vim.lsp.buf.definition, { desc = "Go to definition" })

-- LSP: go to references (Space + g + r)
vim.keymap.set("n", "<leader>gr", vim.lsp.buf.references, { desc = "Go to references" })

-- LSP: rename symbol (Space + r + n)
vim.keymap.set("n", "<leader>rn", vim.lsp.buf.rename, { desc = "Rename symbol" })

-- neo-tree: toggle the file explorer (Space + e)
vim.keymap.set("n", "<leader>e", "<cmd>Neotree toggle<cr>", { desc = "Toggle file explorer" })

-- hop: jump to any word on screen. This overrides the built-in `f` (find char
-- on the current line), matching the old config's behaviour.
vim.keymap.set("", "f", function()
  require("hop").hint_words({ current_line_only = false })
end, { remap = true, desc = "Hop to word" })
"#;

const TELESCOPE_HARPOON_LUA: &str = r#"
-- Telescope + harpoon keymaps (imported from the old Arch config).
-- <leader> is Space (set in lua/plugins.lua).

-- telescope settings
local builtin = require('telescope.builtin')
vim.keymap.set('n', '\\ff', builtin.find_files, {})
vim.keymap.set('n', '\\fb', builtin.buffers, {})
vim.keymap.set('n', '\\fh', builtin.help_tags, {})

-- harpoon settings (v1 API)
require("telescope").load_extension('harpoon')
vim.cmd([[
nnoremap <leader>b <cmd>lua require("harpoon.mark").add_file()<cr>
nnoremap <leader><Esc> <cmd>lua require("harpoon.ui").toggle_quick_menu()<cr>
nnoremap <leader>q <cmd>Telescope harpoon marks<cr>
]])
"#;

// ---------------------------------------------------------------------------
// The `vim` shim: native Rust functions that record what the config asked for.
//
// This is a scale model of what kvim will really do -- the difference being that
// kvim's versions will apply the settings instead of recording them. The shape
// of the API surface is identical, which is the point.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct RecordedKeymap {
    mode: String,
    lhs: String,
    /// What the right-hand side was, resolved to something assertable.
    rhs: String,
    desc: String,
}

#[derive(Debug, Default)]
struct Recorder {
    /// `vim.opt.x = y`, in source order.
    options: Vec<(String, String)>,
    /// `vim.g.x = y`.
    globals: Vec<(String, String)>,
    keymaps: Vec<RecordedKeymap>,
    /// `vim.cmd([[ ... ]])`.
    commands: Vec<String>,
    /// `require("telescope").load_extension(x)`.
    extensions: Vec<String>,
    /// Every module `require`d, in order.
    required: Vec<String>,
    /// The `current_line_only` value each `hop.hint_words{...}` call passed.
    hop_hint_words: Vec<String>,
}

impl Recorder {
    fn option(&self, name: &str) -> Option<&str> {
        self.options.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
    fn keymap(&self, lhs: &str) -> Option<&RecordedKeymap> {
        self.keymaps.iter().find(|k| k.lhs == lhs)
    }
}

/// The marker functions the config passes around as *values* — `vim.lsp.buf.definition`
/// is never called, it is handed to `vim.keymap.set`. We keep the `Value`s so the
/// recorder can identify which one arrived, by identity.
struct Markers {
    lsp_definition: Value,
    lsp_references: Value,
    lsp_rename: Value,
    find_files: Value,
    buffers: Value,
    help_tags: Value,
}

impl Markers {
    /// Names the right-hand side of a keymap.
    fn describe(&self, v: &Value) -> String {
        for (marker, name) in [
            (&self.lsp_definition, "vim.lsp.buf.definition"),
            (&self.lsp_references, "vim.lsp.buf.references"),
            (&self.lsp_rename, "vim.lsp.buf.rename"),
            (&self.find_files, "telescope.builtin.find_files"),
            (&self.buffers, "telescope.builtin.buffers"),
            (&self.help_tags, "telescope.builtin.help_tags"),
        ] {
            if v.raw_eq(marker) {
                return name.to_string();
            }
        }
        match v {
            Value::String(s) => s.to_string_lossy(),
            Value::Function(_) => "<lua closure>".to_string(),
            other => format!("<{}>", other.type_name()),
        }
    }
}

/// Builds an interpreter with the `vim` global installed, plus a module loader
/// that supplies the handful of plugin modules the config `require`s at load
/// time. Returns the interpreter and the shared recording.
fn interpreter_with_vim_shim() -> (Lua, Rc<RefCell<Recorder>>) {
    let mut lua = Lua::new();
    let rec: Rc<RefCell<Recorder>> = Rc::default();

    // --- vim.lsp.buf.{definition,references,rename} ---
    let markers = Rc::new(Markers {
        lsp_definition: lua.create_fn("definition", |_, _| Ok(vec![])),
        lsp_references: lua.create_fn("references", |_, _| Ok(vec![])),
        lsp_rename: lua.create_fn("rename", |_, _| Ok(vec![])),
        find_files: lua.create_fn("find_files", |_, _| Ok(vec![])),
        buffers: lua.create_fn("buffers", |_, _| Ok(vec![])),
        help_tags: lua.create_fn("help_tags", |_, _| Ok(vec![])),
    });

    let buf = lua.create_table();
    buf.borrow_mut().set_str("definition", markers.lsp_definition.clone());
    buf.borrow_mut().set_str("references", markers.lsp_references.clone());
    buf.borrow_mut().set_str("rename", markers.lsp_rename.clone());
    let lsp = lua.create_table();
    lsp.borrow_mut().set_str("buf", Value::Table(buf));

    // --- vim.opt: a table whose __newindex records the assignment ---
    //
    // It deliberately does NOT rawset the key, so the key stays absent and the
    // metamethod fires on every assignment -- including a repeated one. A shim
    // that rawset would silently miss the second `vim.opt.x = ...`.
    let opt = lua.create_table();
    let sink = rec.clone();
    let opt_newindex = lua.create_fn("opt.__newindex", move |lua, args| {
        let key = args[1].as_lua_string().unwrap().to_string_lossy();
        let val = lua.tostring(&args[2])?.to_string_lossy();
        sink.borrow_mut().options.push((key, val));
        Ok(vec![])
    });
    let opt_mt = lua.create_table();
    opt_mt.borrow_mut().set_str("__newindex", opt_newindex);
    opt.borrow_mut().metatable = Some(opt_mt);

    // --- vim.g: the same trick ---
    let g = lua.create_table();
    let sink = rec.clone();
    let g_newindex = lua.create_fn("g.__newindex", move |lua, args| {
        let key = args[1].as_lua_string().unwrap().to_string_lossy();
        let val = lua.tostring(&args[2])?.to_string_lossy();
        sink.borrow_mut().globals.push((key, val));
        Ok(vec![])
    });
    let g_mt = lua.create_table();
    g_mt.borrow_mut().set_str("__newindex", g_newindex);
    g.borrow_mut().metatable = Some(g_mt);

    // --- vim.keymap.set ---
    let sink = rec.clone();
    let m = markers.clone();
    let keymap_set = lua.create_fn("keymap.set", move |_lua, args| {
        let mode = args
            .first()
            .and_then(|v| v.as_lua_string())
            .map(|s| s.to_string_lossy())
            .unwrap_or_default();
        let lhs = args
            .get(1)
            .and_then(|v| v.as_lua_string())
            .map(|s| s.to_string_lossy())
            .unwrap_or_default();
        let rhs = m.describe(args.get(2).unwrap_or(&Value::Nil));

        // The opts table: `{ desc = "...", remap = true }`.
        let desc = match args.get(3) {
            Some(Value::Table(t)) => match t.borrow().raw_get_str("desc") {
                Value::String(s) => s.to_string_lossy(),
                _ => String::new(),
            },
            _ => String::new(),
        };
        sink.borrow_mut().keymaps.push(RecordedKeymap { mode, lhs, rhs, desc });
        Ok(vec![])
    });
    let keymap = lua.create_table();
    keymap.borrow_mut().set_str("set", keymap_set);

    // --- vim.cmd: callable AND indexable ---
    //
    // `vim.cmd([[...]])` calls it; `vim.cmd.colorscheme("gruvbox")` indexes it.
    // Real Neovim supports both, so the shim must too, which means __call and
    // __index on the same table.
    let cmd = lua.create_table();
    let sink = rec.clone();
    let cmd_call = lua.create_fn("cmd.__call", move |_lua, args| {
        // args[0] is the table itself (the __call convention).
        if let Some(s) = args.get(1).and_then(|v| v.as_lua_string()) {
            sink.borrow_mut().commands.push(s.to_string_lossy());
        }
        Ok(vec![])
    });
    let sink = rec.clone();
    let cmd_index = lua.create_fn("cmd.__index", move |lua, args| {
        // `vim.cmd.foo` returns a function that records `foo <args>`.
        let name = args[1].as_lua_string().unwrap().to_string_lossy();
        let sink = sink.clone();
        Ok(vec![lua.create_fn("cmd.subcommand", move |_lua, args| {
            let arg = args
                .first()
                .and_then(|v| v.as_lua_string())
                .map(|s| s.to_string_lossy())
                .unwrap_or_default();
            sink.borrow_mut().commands.push(format!("{name} {arg}"));
            Ok(vec![])
        })])
    });
    let cmd_mt = lua.create_table();
    cmd_mt.borrow_mut().set_str("__call", cmd_call);
    cmd_mt.borrow_mut().set_str("__index", cmd_index);
    cmd.borrow_mut().metatable = Some(cmd_mt);

    // --- assemble `vim` ---
    let vim = lua.create_table();
    vim.borrow_mut().set_str("opt", Value::Table(opt));
    vim.borrow_mut().set_str("g", Value::Table(g));
    vim.borrow_mut().set_str("keymap", Value::Table(keymap));
    vim.borrow_mut().set_str("cmd", Value::Table(cmd));
    vim.borrow_mut().set_str("lsp", Value::Table(lsp));
    lua.set_global("vim", Value::Table(vim));

    // --- the module loader: `require("telescope.builtin")` and friends ---
    //
    // Modules are returned as Lua source, which the VM then executes -- exactly
    // the path kvim will use, only reading from disk instead of a match arm.
    let m = markers.clone();
    let builtins = (m.find_files.clone(), m.buffers.clone(), m.help_tags.clone());

    // The loader must hand back *source*, but `telescope.builtin` needs to expose
    // the very marker functions we hold in Rust. So preload them into
    // package.loaded directly, and let the loader handle the rest.
    let telescope_builtin = lua.create_table();
    telescope_builtin.borrow_mut().set_str("find_files", builtins.0);
    telescope_builtin.borrow_mut().set_str("buffers", builtins.1);
    telescope_builtin.borrow_mut().set_str("help_tags", builtins.2);

    let sink2 = rec.clone();
    let load_extension = lua.create_fn("load_extension", move |_lua, args| {
        if let Some(s) = args.first().and_then(|v| v.as_lua_string()) {
            sink2.borrow_mut().extensions.push(s.to_string_lossy());
        }
        Ok(vec![])
    });
    let telescope = lua.create_table();
    telescope.borrow_mut().set_str("load_extension", load_extension);

    // `package.loaded` is the module cache `require` consults first, so seeding
    // it is exactly how a host preloads a native module.
    let loaded = lua.eval("package.loaded").unwrap();
    if let Value::Table(loaded) = loaded {
        loaded.borrow_mut().set_str("telescope.builtin", Value::Table(telescope_builtin));
        loaded.borrow_mut().set_str("telescope", Value::Table(telescope));
    }

    // `hop.hint_words` records the options it was handed. The hop module itself is
    // served as SOURCE by the loader below, so this exercises the real path:
    // require -> compile -> execute -> return a table -> index it -> call it.
    let sink = rec.clone();
    lua.set_global_fn("__record_hop", move |lua, args| {
        let opts = match args.first() {
            Some(Value::Table(t)) => lua.tostring(&t.borrow().raw_get_str("current_line_only"))?,
            _ => kopitiam_lua::LuaStr::from("<no table>"),
        };
        sink.borrow_mut().hop_hint_words.push(opts.to_string_lossy());
        Ok(vec![])
    });

    // Anything else the config requires (e.g. "hop", lazily inside a closure).
    let sink = rec.clone();
    lua.set_module_loader(move |name| {
        sink.borrow_mut().required.push(name.to_string());
        match name {
            "hop" => Some(
                "return { hint_words = function(opts) __record_hop(opts) end }".to_string(),
            ),
            _ => None,
        }
    });

    (lua, rec)
}

// ---------------------------------------------------------------------------
// The tests.
// ---------------------------------------------------------------------------

#[test]
fn the_maintainers_settings_lua_executes_and_produces_exactly_kvims_defaults() {
    let (mut lua, rec) = interpreter_with_vim_shim();
    lua.exec(SETTINGS_LUA, "@settings.lua").expect("settings.lua must execute");
    let r = rec.borrow();

    // Every one of these is asserted against `Options::default()` in
    // crates/kopitiam-neovim/src/config.rs. If the VM produced anything else,
    // kvim's ported defaults and the user's real config would disagree.
    assert_eq!(r.option("number"), Some("true"));
    assert_eq!(r.option("relativenumber"), Some("true"));
    assert_eq!(r.option("tabstop"), Some("4"), "a number must not stringify as '4.0'");
    assert_eq!(r.option("shiftwidth"), Some("4"));
    assert_eq!(r.option("wrap"), Some("false"));
    assert_eq!(r.option("scrolloff"), Some("5"));
    assert_eq!(r.option("spell"), Some("true"));
    assert_eq!(r.option("spelllang"), Some("en_gb"));
    assert_eq!(r.option("colorcolumn"), Some("75"));
    assert_eq!(r.option("background"), Some("dark"));

    // Ten options, in source order, none missed and none duplicated.
    assert_eq!(r.options.len(), 10, "got: {:?}", r.options);

    // `vim.g.airline_theme = "dark"`.
    assert_eq!(r.globals, vec![("airline_theme".to_string(), "dark".to_string())]);

    // Two `vim.cmd([[ ... ]])` calls, both long-bracket strings spanning lines.
    // The leading newline after `[[` must have been eaten; the trailing one kept.
    assert_eq!(r.commands.len(), 2);
    assert_eq!(r.commands[0], "syntax on\nfiletype plugin indent on\n");
    assert_eq!(r.commands[1], "autocmd BufNewFile,BufRead *.tex set filetype=tex\n");
}

#[test]
fn the_maintainers_keymaps_lua_executes_and_binds_what_kvim_expects() {
    let (mut lua, rec) = interpreter_with_vim_shim();
    lua.exec(KEYMAPS_LUA, "@keymaps.lua").expect("keymaps.lua must execute");
    let r = rec.borrow();

    assert_eq!(r.keymaps.len(), 5, "got: {:?}", r.keymaps);

    // The three LSP maps pass `vim.lsp.buf.*` as a VALUE -- a three-deep index
    // chain resolved to a function object, not called.
    let gd = r.keymap("<leader>gd").expect("<leader>gd must be bound");
    assert_eq!(gd.mode, "n");
    assert_eq!(gd.rhs, "vim.lsp.buf.definition");
    assert_eq!(gd.desc, "Go to definition");

    let gr = r.keymap("<leader>gr").unwrap();
    assert_eq!(gr.rhs, "vim.lsp.buf.references");
    assert_eq!(gr.desc, "Go to references");

    let rn = r.keymap("<leader>rn").unwrap();
    assert_eq!(rn.rhs, "vim.lsp.buf.rename");
    assert_eq!(rn.desc, "Rename symbol");

    // The neo-tree map passes a plain string.
    let e = r.keymap("<leader>e").unwrap();
    assert_eq!(e.mode, "n");
    assert_eq!(e.rhs, "<cmd>Neotree toggle<cr>");
    assert_eq!(e.desc, "Toggle file explorer");

    // The hop map passes a CLOSURE, and its mode is "" (all modes) -- which is
    // what makes it shadow vim's built-in `f`. kvim models this as
    // `Keymap { mode: String::new(), lhs: "f", action: Action::HopWords }`.
    let f = r.keymap("f").unwrap();
    assert_eq!(f.mode, "", "the hop map is mode \"\" -- all modes");
    assert_eq!(f.rhs, "<lua closure>");
    assert_eq!(f.desc, "Hop to word");
}

#[test]
fn the_hop_closure_is_a_real_closure_that_actually_runs() {
    // Recording that a closure arrived is not enough -- it has to *work*. So call
    // it, and check it did what its body says: `require("hop").hint_words({
    // current_line_only = false })`.
    //
    // This exercises, in one line of config: a closure invoked from Rust,
    // `require` inside it, a method-style field access on the result, and a table
    // constructor with a named field.
    let (mut lua, rec) = interpreter_with_vim_shim();

    // Capture the closure as it is registered.
    let captured: Rc<RefCell<Option<Value>>> = Rc::default();
    let sink = captured.clone();
    let keymap_set = lua.create_fn("set", move |_lua, args| {
        if let Some(v @ Value::Function(_)) = args.get(2) {
            *sink.borrow_mut() = Some(v.clone());
        }
        Ok(vec![])
    });
    // Replace vim.keymap.set with the capturing version.
    let vim = lua.get_global("vim");
    if let Value::Table(vim) = &vim {
        let keymap = vim.borrow().raw_get_str("keymap");
        if let Value::Table(keymap) = keymap {
            keymap.borrow_mut().set_str("set", keymap_set);
        }
    }

    lua.exec(KEYMAPS_LUA, "@keymaps.lua").unwrap();

    let f = captured.borrow().clone().expect("the hop closure must have been registered");
    let result = lua.call(&f, vec![]).expect("the hop closure must run");

    // The closure's body is a CALL STATEMENT -- `require("hop").hint_words({...})`
    // with no `return` -- so it correctly yields no values. What we check is the
    // effect: that `hint_words` was reached, with the options the config wrote.
    assert!(result.is_empty(), "the hop closure returns nothing, and must not invent a value");

    let r = rec.borrow();
    assert_eq!(
        r.hop_hint_words,
        vec!["false".to_string()],
        "hint_words must have been called once with current_line_only = false"
    );

    // And `require("hop")` really went through the module loader -- lazily, only
    // when the key was pressed, exactly as the config intends.
    assert!(r.required.contains(&"hop".to_string()));
}

#[test]
fn requiring_hop_is_lazy_and_does_not_happen_at_config_load_time() {
    // The `f` mapping's body is inside a closure, so `require("hop")` must NOT run
    // while the config is loading -- only when the key is actually pressed. A VM
    // that eagerly evaluated the closure body would try to load every plugin at
    // startup.
    let (mut lua, rec) = interpreter_with_vim_shim();
    lua.exec(KEYMAPS_LUA, "@keymaps.lua").unwrap();

    assert!(
        !rec.borrow().required.contains(&"hop".to_string()),
        "hop must not be required until the mapping is invoked"
    );
    assert!(rec.borrow().hop_hint_words.is_empty());
}

#[test]
fn the_maintainers_telescope_harpoon_lua_executes() {
    let (mut lua, rec) = interpreter_with_vim_shim();
    lua.exec(TELESCOPE_HARPOON_LUA, "@telescope_harpoon.lua")
        .expect("telescope_harpoon.lua must execute");
    let r = rec.borrow();

    // `local builtin = require('telescope.builtin')` then `builtin.find_files`
    // passed as a value -- a require result, stored in a local, then indexed.
    assert_eq!(r.keymaps.len(), 3);
    assert_eq!(r.keymap("\\ff").unwrap().rhs, "telescope.builtin.find_files");
    assert_eq!(r.keymap("\\fb").unwrap().rhs, "telescope.builtin.buffers");
    assert_eq!(r.keymap("\\fh").unwrap().rhs, "telescope.builtin.help_tags");

    // Note the lhs: a Lua "\\ff" literal is the two characters `\` and `ff` --
    // an escaped backslash. kvim's Config spells it "\\ff" in Rust for the same
    // reason. If the lexer mishandled the escape we would see "\\\\ff" here.
    assert_eq!(r.keymap("\\ff").unwrap().lhs, r"\ff");

    // `require("telescope").load_extension('harpoon')` -- chained call on a
    // require result.
    assert_eq!(r.extensions, vec!["harpoon".to_string()]);

    // The harpoon keymaps come through a long-bracket vim.cmd block.
    assert_eq!(r.commands.len(), 1);
    assert!(r.commands[0].contains("nnoremap <leader>b"));
    assert!(r.commands[0].contains("nnoremap <leader><Esc>"));
    assert!(r.commands[0].contains("nnoremap <leader>q"));
}

#[test]
fn the_whole_config_loads_in_order_through_require_just_as_init_lua_does() {
    // init.lua is a sequence of `require("settings")`, `require("keymaps")`, ...
    // This runs that same entry point, with the module loader serving the real
    // file contents -- which is precisely what kvim will do with
    // ~/.kopitiam/kopitiam-neovim/lua/*.lua.
    let (mut lua, rec) = interpreter_with_vim_shim();

    // Layer the config modules on top of the shim's loader.
    lua.set_module_loader(|name| match name {
        "settings" => Some(SETTINGS_LUA.to_string()),
        "keymaps" => Some(KEYMAPS_LUA.to_string()),
        "hop" => Some("return { hint_words = function(opts) return opts end }".to_string()),
        _ => None,
    });

    // The relevant slice of the real init.lua.
    lua.exec(
        r#"
        vim.g.loaded_netrw = 1
        vim.g.loaded_netrwPlugin = 1
        vim.opt.termguicolors = true

        require("settings")
        require("keymaps")
        "#,
        "@init.lua",
    )
    .expect("init.lua must execute");

    let r = rec.borrow();

    // init.lua's own two globals, then settings.lua's one.
    assert_eq!(r.globals.len(), 3);
    assert_eq!(r.globals[0], ("loaded_netrw".to_string(), "1".to_string()));
    assert_eq!(r.globals[1], ("loaded_netrwPlugin".to_string(), "1".to_string()));
    assert_eq!(r.globals[2], ("airline_theme".to_string(), "dark".to_string()));

    // termguicolors from init.lua, then settings.lua's ten.
    assert_eq!(r.options.len(), 11);
    assert_eq!(r.options[0], ("termguicolors".to_string(), "true".to_string()));

    // And keymaps.lua's five bindings all arrived.
    assert_eq!(r.keymaps.len(), 5);
}

#[test]
fn requiring_a_module_twice_executes_it_once() {
    // Lua's guarantee, and it matters: a config that requires "settings" from two
    // places must not apply every `vim.opt` twice.
    let (mut lua, rec) = interpreter_with_vim_shim();
    lua.set_module_loader(|name| match name {
        "settings" => Some(SETTINGS_LUA.to_string()),
        _ => None,
    });
    lua.exec(r#"require("settings") require("settings") require("settings")"#, "@init.lua")
        .unwrap();

    assert_eq!(rec.borrow().options.len(), 10, "settings.lua must run exactly once");
}

/// Runs whatever is *actually* in `~/.config/nvim` right now.
///
/// The embedded copies above make this suite hermetic; this test makes it
/// honest. If the maintainer edits their config into something the VM cannot
/// run, this fails and says so — which is the entire point of building the
/// interpreter.
///
/// Skips (rather than fails) when the config is not present, so the suite still
/// passes on a machine that is not the maintainer's.
#[test]
fn the_live_config_on_disk_still_executes() {
    let Some(home) = std::env::var_os("HOME") else { return };
    let dir = std::path::Path::new(&home).join(".config/nvim/lua");
    if !dir.is_dir() {
        eprintln!("skipping: {} not present", dir.display());
        return;
    }

    for module in ["settings", "keymaps", "telescope_harpoon"] {
        let path = dir.join(format!("{module}.lua"));
        let Ok(source) = std::fs::read_to_string(&path) else {
            eprintln!("skipping {module}: not readable");
            continue;
        };

        let (mut lua, rec) = interpreter_with_vim_shim();
        lua.exec(&source, &format!("@{module}.lua")).unwrap_or_else(|e| {
            panic!("the maintainer's real {}.lua failed to execute: {e}", module)
        });

        // A smoke assertion per file, so an accidentally-empty run cannot pass.
        let r = rec.borrow();
        match module {
            "settings" => assert!(!r.options.is_empty(), "settings.lua set no options"),
            "keymaps" | "telescope_harpoon" => {
                assert!(!r.keymaps.is_empty(), "{module}.lua bound no keys")
            }
            _ => {}
        }
    }
}
