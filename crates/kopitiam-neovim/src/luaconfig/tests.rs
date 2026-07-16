//! Tests for the `vim.*` shim — the config side of the loop the scale-model
//! test in `kopitiam-lua/tests/maintainer_config.rs` opened. There, a `vim` shim
//! *records* what the config asked for; here it *applies* it to a real
//! [`Config`], and these tests assert the config that came out is the one the
//! Lua asked for.

use super::*;
use crate::config::{Action, Background, Config};

/// Runs `init` as the sole `init.lua`, with `modules` reachable via `require`.
fn run_with_modules(init: &str, modules: &[(&str, &str)]) -> LuaRuntime {
    let discovered = Discovered {
        init: Some(init.to_string()),
        modules: modules.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
    };
    LuaRuntime::load(Config::default(), &discovered)
}

/// Runs a single chunk as `init.lua`, no extra modules.
fn run(init: &str) -> LuaRuntime {
    run_with_modules(init, &[])
}

fn keymap<'a>(cfg: &'a Config, lhs: &str) -> &'a crate::config::Keymap {
    cfg.keymaps.iter().find(|k| k.lhs == lhs).unwrap_or_else(|| panic!("no keymap for {lhs}"))
}

// --------------------------------------------------------------------------
// vim.opt / vim.o
// --------------------------------------------------------------------------

#[test]
fn vim_opt_scalar_assignments_land_on_options() {
    // Start from non-maintainer values so we can see the assignment bite, not a
    // coincidence with the default.
    let base = Config::default();
    let discovered = Discovered {
        init: Some(
            r#"
            vim.opt.number = false
            vim.opt.relativenumber = false
            vim.opt.tabstop = 8
            vim.opt.shiftwidth = 2
            vim.opt.wrap = true
            vim.opt.scrolloff = 3
            vim.opt.expandtab = true
            vim.opt.colorcolumn = "100"
            vim.opt.spelllang = "en_us"
            vim.opt.background = "light"
        "#
            .to_string(),
        ),
        modules: Default::default(),
    };
    let cfg = LuaRuntime::load(base, &discovered).config();

    assert!(!cfg.options.number);
    assert!(!cfg.options.relativenumber);
    assert_eq!(cfg.options.tabstop, 8);
    assert_eq!(cfg.options.shiftwidth.resolve(8), 2);
    assert!(cfg.options.wrap);
    assert_eq!(cfg.options.scrolloff, 3);
    assert!(cfg.options.expandtab);
    assert_eq!(cfg.options.colorcolumn, Some(100));
    assert_eq!(cfg.options.spelllang, "en_us");
    assert_eq!(cfg.options.background, Background::Light);
}

#[test]
fn a_number_option_does_not_stringify_as_a_float() {
    // The bug the scale-model test guards: `tabstop = 4` must become 4, never
    // "4.0" and thus a failed parse leaving the default.
    let cfg = run("vim.opt.tabstop = 4").config();
    assert_eq!(cfg.options.tabstop, 4);
}

#[test]
fn vim_o_shorthand_also_sets_options() {
    let cfg = run("vim.o.tabstop = 6\nvim.wo.number = false\nvim.bo.expandtab = true").config();
    assert_eq!(cfg.options.tabstop, 6);
    assert!(!cfg.options.number);
    assert!(cfg.options.expandtab);
}

#[test]
fn vim_opt_append_extends_a_comma_list_option() {
    let cfg = run(r#"vim.opt.clipboard:append("unnamedplus")"#).config();
    assert_eq!(cfg.options.clipboard, "unnamedplus");
    assert_eq!(cfg.options.clipboard_sync_register(), Some('+'));
}

#[test]
fn an_unknown_option_warns_but_does_not_crash() {
    let rt = run("vim.opt.made_up_option = true\nvim.opt.number = true");
    assert!(rt.config().options.number, "the good option after it still applied");
    assert!(rt.warnings().iter().any(|w| w.contains("made_up_option")));
}

// --------------------------------------------------------------------------
// vim.g — the leader is load-bearing
// --------------------------------------------------------------------------

#[test]
fn mapleader_drives_the_leader_key() {
    let cfg = run(r#"vim.g.mapleader = " ""#).config();
    assert_eq!(cfg.leader, ' ');
}

#[test]
fn mapleader_can_be_a_comma() {
    let cfg = run(r#"vim.g.mapleader = ",""#).config();
    assert_eq!(cfg.leader, ',');
}

#[test]
fn a_global_var_can_be_read_back() {
    // `vim.g.x = 1` then reading `vim.g.x` must see it — a config that branches
    // on its own globals relies on this.
    let rt = run(r#"
        vim.g.my_flag = "yes"
        if vim.g.my_flag == "yes" then vim.opt.number = false end
    "#);
    assert!(!rt.config().options.number);
}

// --------------------------------------------------------------------------
// vim.keymap.set
// --------------------------------------------------------------------------

#[test]
fn a_string_ex_command_keymap_registers() {
    // The house-rules example: `<leader>x` → `:w`.
    let rt = run(r#"vim.g.mapleader = " "; vim.opt.number = true; vim.keymap.set("n", "<leader>x", ":w<CR>")"#);
    let cfg = rt.config();
    assert!(cfg.options.number, "the option set alongside also applied");
    let k = keymap(&cfg, "<leader>x");
    assert_eq!(k.mode, "n");
    assert_eq!(k.action, Action::Command("w".to_string()));
}

#[test]
fn a_cmd_wrapped_keymap_maps_to_the_native_action() {
    let cfg = run(r#"vim.keymap.set("n", "<leader>e", "<cmd>Neotree toggle<cr>", { desc = "tree" })"#).config();
    let k = keymap(&cfg, "<leader>e");
    assert_eq!(k.action, Action::FileTreeToggle);
    assert_eq!(k.desc, "tree");
}

#[test]
fn a_plain_string_keymap_feeds_keys() {
    let cfg = run(r#"vim.keymap.set("n", "Y", "y$")"#).config();
    assert_eq!(keymap(&cfg, "Y").action, Action::FeedKeys("y$".to_string()));
}

#[test]
fn a_lua_function_keymap_is_stored_and_fires() {
    // The core of the feature: a function rhs is bound as Action::LuaKeymap, and
    // firing it actually runs the Lua body.
    let mut rt = run(r#"
        vim.g.fired = "no"
        vim.keymap.set("n", "<leader>t", function() vim.g.fired = "yes" end)
    "#);
    let cfg = rt.config();
    let action = keymap(&cfg, "<leader>t").action.clone();
    let Action::LuaKeymap(id) = action else {
        panic!("expected a Lua keymap, got {action:?}");
    };

    // Not fired yet.
    assert_eq!(rt.state.borrow().globals.get("fired").map(String::as_str), Some("no"));

    // Fire it — the closure runs and flips the global.
    rt.fire_keymap(id).expect("firing the Lua keymap must succeed");
    assert_eq!(rt.state.borrow().globals.get("fired").map(String::as_str), Some("yes"));
}

#[test]
fn a_later_keymap_on_the_same_key_wins() {
    let cfg = run(r#"
        vim.keymap.set("n", "gx", "y$")
        vim.keymap.set("n", "gx", "d$")
    "#).config();
    let matches: Vec<_> = cfg.keymaps.iter().filter(|k| k.lhs == "gx").collect();
    assert_eq!(matches.len(), 1, "only the last binding survives");
    assert_eq!(matches[0].action, Action::FeedKeys("d$".to_string()));
}

// --------------------------------------------------------------------------
// vim.cmd
// --------------------------------------------------------------------------

#[test]
fn vim_cmd_colorscheme_sets_the_theme_both_ways() {
    let cfg = run(r#"vim.cmd.colorscheme("evenshade")"#).config();
    assert_eq!(cfg.theme, "evenshade");

    let cfg = run("vim.cmd([[colorscheme tokyonight]])").config();
    assert_eq!(cfg.theme, "tokyonight");
}

#[test]
fn vim_cmd_block_applies_set_and_map_lines() {
    let cfg = run(r#"vim.cmd([[
        set tabstop=2
        set nonumber
        nnoremap <leader>w :w<cr>
        syntax off
    ]])"#).config();
    assert_eq!(cfg.options.tabstop, 2);
    assert!(!cfg.options.number);
    assert!(!cfg.options.syntax);
    assert_eq!(keymap(&cfg, "<leader>w").action, Action::Command("w".to_string()));
}

#[test]
fn vim_cmd_records_an_autocmd() {
    let rt = run("vim.cmd([[autocmd BufNewFile,BufRead *.tex set filetype=tex]])");
    let acs = rt.autocmds();
    assert_eq!(acs.len(), 1);
    assert_eq!(acs[0].events, vec!["BufNewFile", "BufRead"]);
    assert_eq!(acs[0].pattern, "*.tex");
}

// --------------------------------------------------------------------------
// vim.api
// --------------------------------------------------------------------------

#[test]
fn nvim_set_keymap_registers_like_vim_keymap_set() {
    let cfg = run(r#"vim.api.nvim_set_keymap("n", "<leader>z", ":noh<CR>", { noremap = true })"#).config();
    assert_eq!(keymap(&cfg, "<leader>z").action, Action::Command("noh".to_string()));
}

#[test]
fn nvim_create_autocmd_records() {
    let rt = run(r#"vim.api.nvim_create_autocmd("BufWritePre", { pattern = "*.rs", command = "echo 1" })"#);
    let acs = rt.autocmds();
    assert_eq!(acs.len(), 1);
    assert_eq!(acs[0].events, vec!["BufWritePre"]);
    assert_eq!(acs[0].pattern, "*.rs");
}

#[test]
fn an_unknown_nvim_api_is_a_black_hole_not_a_crash() {
    let rt = run(r#"
        local buf = vim.api.nvim_get_current_buf()
        local x = buf.anything.nested()
        vim.opt.number = true
    "#);
    assert!(rt.config().options.number, "config continued past the unknown API");
    assert!(rt.warnings().iter().any(|w| w.contains("nvim_get_current_buf")));
}

// --------------------------------------------------------------------------
// require + plugin-manager boilerplate
// --------------------------------------------------------------------------

#[test]
fn requiring_a_plugin_manager_does_not_crash_and_options_after_it_apply() {
    let rt = run(r#"
        require("lazy").setup({
            { "nvim-telescope/telescope.nvim" },
            { "phaazon/hop.nvim" },
        })
        vim.opt.number = true
    "#);
    assert!(rt.config().options.number, "options after require('lazy').setup still apply");
    assert!(rt.stubbed_plugins().iter().any(|p| p == "lazy"));
}

#[test]
fn a_native_plugin_require_yields_a_marker_that_maps_to_its_action() {
    let cfg = run(r#"
        local builtin = require("telescope.builtin")
        vim.keymap.set("n", "\\ff", builtin.find_files)
        vim.keymap.set("n", "\\fb", builtin.buffers)
    "#).config();
    assert_eq!(keymap(&cfg, "\\ff").action, Action::FindFiles);
    assert_eq!(keymap(&cfg, "\\fb").action, Action::FindBuffers);
}

#[test]
fn the_lsp_buf_functions_map_to_native_lsp_actions() {
    let cfg = run(r#"
        vim.keymap.set("n", "<leader>gd", vim.lsp.buf.definition)
        vim.keymap.set("n", "<leader>rn", vim.lsp.buf.rename)
    "#).config();
    assert_eq!(keymap(&cfg, "<leader>gd").action, Action::LspDefinition);
    assert_eq!(keymap(&cfg, "<leader>rn").action, Action::LspRename);
}

// --------------------------------------------------------------------------
// vim.notify / vim.schedule / vim.fn
// --------------------------------------------------------------------------

#[test]
fn vim_notify_is_collected() {
    let rt = run(r#"vim.notify("hello from config", vim.log.levels.INFO)"#);
    assert!(rt.notifications().iter().any(|n| n.contains("hello from config")));
}

#[test]
fn vim_schedule_runs_after_load() {
    let rt = run(r#"
        vim.g.done = "no"
        vim.schedule(function() vim.g.done = "yes" end)
        -- still "no" at this point in the script, but the runtime runs it after load
    "#);
    assert_eq!(rt.state.borrow().globals.get("done").map(String::as_str), Some("yes"));
}

#[test]
fn vim_fn_has_and_stdpath_are_callable() {
    let rt = run(r#"
        if vim.fn.has("nvim") == 1 then vim.opt.number = false end
        vim.g.cfgdir = vim.fn.stdpath("config")
    "#);
    assert!(!rt.config().options.number, "has('nvim') returned 1");
    assert!(rt.state.borrow().globals.contains_key("cfgdir"));
}

// --------------------------------------------------------------------------
// error handling
// --------------------------------------------------------------------------

#[test]
fn a_malformed_config_is_a_warning_not_a_panic() {
    let rt = run("vim.opt.number = true\nthis is not valid lua @#$");
    // The parse error aborts the file, but it is reported, not fatal.
    assert!(rt.warnings().iter().any(|w| w.contains("init.lua failed")));
}

#[test]
fn a_runtime_error_keeps_what_ran_before_it() {
    let rt = run(r#"
        vim.opt.tabstop = 2
        error("boom")
        vim.opt.tabstop = 9
    "#);
    assert_eq!(rt.config().options.tabstop, 2, "the assignment before the error stuck");
    assert!(rt.warnings().iter().any(|w| w.contains("failed")));
}

// --------------------------------------------------------------------------
// the whole maintainer config, through require, as init.lua does it
// --------------------------------------------------------------------------

#[test]
fn the_maintainers_config_shape_loads_end_to_end() {
    let settings = r#"
        vim.opt.number = true
        vim.opt.relativenumber = true
        vim.opt.tabstop = 4
        vim.opt.shiftwidth = 4
        vim.opt.wrap = false
        vim.opt.scrolloff = 5
        vim.opt.spell = true
        vim.opt.spelllang = "en_gb"
        vim.opt.colorcolumn = "75"
        vim.opt.background = "dark"
        vim.g.airline_theme = "dark"
        vim.cmd([[
        syntax on
        filetype plugin indent on
        ]])
    "#;
    let keymaps = r#"
        vim.keymap.set("n", "<leader>gd", vim.lsp.buf.definition, { desc = "Go to definition" })
        vim.keymap.set("n", "<leader>e", "<cmd>Neotree toggle<cr>", { desc = "Toggle file explorer" })
        vim.keymap.set("", "f", function()
          require("hop").hint_words({ current_line_only = false })
        end, { remap = true, desc = "Hop to word" })
    "#;
    let init = r#"
        vim.g.mapleader = " "
        vim.opt.termguicolors = true
        require("settings")
        require("keymaps")
        require("lazy").setup({})
        vim.cmd.colorscheme("gruvbox")
    "#;

    let rt = run_with_modules(init, &[("settings", settings), ("keymaps", keymaps)]);
    let cfg = rt.config();

    assert_eq!(cfg.leader, ' ');
    assert_eq!(cfg.theme, "gruvbox");
    assert_eq!(cfg.options.tabstop, 4);
    assert_eq!(cfg.options.spelllang, "en_gb");
    assert_eq!(cfg.options.colorcolumn, Some(75));
    assert_eq!(cfg.options.background, Background::Dark);

    assert_eq!(keymap(&cfg, "<leader>gd").action, Action::LspDefinition);
    assert_eq!(keymap(&cfg, "<leader>e").action, Action::FileTreeToggle);

    // The `f` map has a Lua closure rhs and mode "" (all modes), just like the
    // real config, and firing it runs the hop closure without crashing.
    let f = keymap(&cfg, "f");
    assert_eq!(f.mode, "");
    let Action::LuaKeymap(id) = f.action else { panic!("f should be a Lua keymap") };
    let mut rt = rt;
    rt.fire_keymap(id).expect("firing the hop closure must not crash");

    // termguicolors is not modelled, so it warned rather than dying.
    assert!(rt.warnings().iter().any(|w| w.contains("termguicolors")));
    // lazy was stubbed.
    assert!(rt.stubbed_plugins().iter().any(|p| p == "lazy"));
}
