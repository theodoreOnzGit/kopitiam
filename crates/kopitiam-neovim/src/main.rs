//! `kvim` — KOPITIAM's Rust-native, Android-capable modal editor.
//!
//! Everything of substance lives in the [`kopitiam_neovim`] library; this file
//! is the thin shell around it, exactly as `apps/cli` is for the Semantic
//! Runtime. It parses arguments, handles the few things that are not "open the
//! editor" (installing the bundled font, printing where config lives), and
//! otherwise hands control to the UI event loop.
//!
//! # Why the binary lives in this crate and not in `apps/`
//!
//! Every other KOPITIAM client lives under `apps/`, per the
//! "Applications are clients, the platform owns the functionality" rule. This
//! one is a deliberate, maintainer-authorized exception: they asked that
//! `cargo install kopitiam-neovim` yield a working `kvim`, which requires the
//! binary target to be in this crate. See
//! `docs/ai-decisions/AID-0003-kopitiam-neovim-architecture.md`, decision 4.

use std::path::PathBuf;

use kopitiam_neovim::{Config, icons};

const USAGE: &str = "\
kvim — a Rust-native modal editor with batteries included

USAGE:
    kvim [OPTIONS] [FILE]...

OPTIONS:
    --install-font    Install the bundled Nerd Font so devicons render
    --icons <SET>     Force an icon set: nerd | unicode | ascii
    --config-path     Print where kvim looks for its config, and exit
    --version         Print version and exit
    -h, --help        Print this help and exit

Unlike Neovim, kvim needs no plugin manager and no Mason: the plugins are
compiled in, and language servers are fetched as prebuilt static binaries
rather than through npm/pip/go — which is what makes it work on Android.

kvim is part of KOPITIAM. Despite the leading `k`, it is NOT a KDE
application and is not part of the KDE Plasma workspace.

Configuration is optional. With no config file at all, kvim starts with the
maintainer's Neovim setup baked in. Overrides go in the file printed by
--config-path; kvim never reads or writes ~/.config/nvim.
";

fn main() -> anyhow::Result<()> {
    let mut files: Vec<PathBuf> = Vec::new();
    let mut icon_override: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(());
            }
            "--version" => {
                println!("kvim {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--config-path" => return show_config_paths(),
            "--install-font" => return install_font(),
            "--icons" => {
                icon_override = args.next();
            }
            // A lone "--" ends option parsing, so a file may be named "--help".
            // Drain the rest and stop: everything after it is a path.
            "--" => {
                files.extend(args.by_ref().map(PathBuf::from));
                break;
            }
            other if other.starts_with('-') && other.len() > 1 => {
                anyhow::bail!("unknown option {other:?}\n\n{USAGE}");
            }
            other => files.push(PathBuf::from(other)),
        }
    }

    // `--icons` is plumbed through the environment rather than threaded as a
    // parameter, so that it composes with the KVIM_ICONS override that
    // `IconSet::detect` already honours — one code path, not two.
    if let Some(set) = icon_override {
        // SAFETY: single-threaded; nothing has been spawned yet.
        unsafe { std::env::set_var("KVIM_ICONS", set) };
    }

    let config = Config::load()?;
    kopitiam_neovim::ui::run(config, &files)
}

/// Implements `kvim --config-path`: where kvim looks, and what it found.
///
/// Prints what it *found*, not merely where it would look, because the most
/// common configuration bug is a file sitting in a directory nobody is reading.
fn show_config_paths() -> anyhow::Result<()> {
    let Some(dir) = Config::dir() else {
        println!("No home directory could be determined.");
        println!("kvim will run on its built-in defaults, which reproduce the maintainer's Neovim setup.");
        return Ok(());
    };

    println!("kvim directory:  {}", dir.display());

    let config = Config::config_path().expect("dir() resolved, so config_path() must too");
    if config.is_file() {
        println!("config.json:     {} (found)", config.display());
    } else {
        println!("config.json:     {} (absent — using defaults)", config.display());
    }

    let lua = Config::lua_files();
    if lua.is_empty() {
        println!("Lua config:      none found (looked for init.lua and lua/*.lua)");
    } else {
        println!("Lua config:      {} file(s) found, in load order:", lua.len());
        for path in &lua {
            println!("                   {}", path.display());
        }
        println!();
        println!("NOTE: these are found but NOT YET EXECUTED. Running them needs a Lua");
        println!("interpreter, and KOPITIAM is committed to a pure-Rust one (kopitiam-lua)");
        println!("which is not built yet. kvim tells you rather than silently ignoring them.");
    }

    println!();
    println!("With no config at all, kvim's defaults ARE the maintainer's Neovim setup:");
    println!("hybrid line numbers, tabstop/shiftwidth 4, no wrap, scrolloff 5, spell en_gb,");
    println!("colorcolumn 75, gruvbox dark, leader = Space, and their full keymap.");
    println!();
    println!("kvim never reads or writes ~/.config/nvim — that stays yours.");

    Ok(())
}

/// Implements `kvim --install-font`.
///
/// See [`icons`] for why a font, rather than an icon table, is the thing that
/// actually has to ship: a devicon is a private-use codepoint, and a codepoint
/// without a font is a tofu box.
fn install_font() -> anyhow::Result<()> {
    let install = icons::install_font()?;

    println!("Installed JetBrains Mono Nerd Font to {}", install.path.display());
    println!("(OFL-1.1; the license was written alongside it)");

    if let Some(cmd) = install.follow_up {
        println!();
        println!("One more step — run this yourself:");
        println!("    {cmd}");
        if icons::is_termux() {
            // Running it for them would restart the Termux session and kill
            // whatever is issuing the command, so it is theirs to run.
            println!();
            println!("(kvim won't run it for you: on Termux it restarts the session.)");
        }
    }

    println!();
    println!("Then point your terminal at the font and start kvim with:");
    println!("    KVIM_ICONS=nerd kvim");

    Ok(())
}
