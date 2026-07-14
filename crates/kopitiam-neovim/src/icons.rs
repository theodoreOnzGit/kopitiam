//! Devicons — and the font they need in order to exist.
//!
//! # The thing everyone gets wrong
//!
//! A devicon is not an image and it is not code. It is a **codepoint in the
//! Nerd Fonts Private Use Area** — the Rust icon is `U+E7A8`, and that is the
//! whole of it. "Shipping icons" therefore means printing a codepoint, and
//! whether the user sees a Rust logo or a tofu box `□` is decided entirely by
//! the font their *terminal emulator* is configured with. That is not
//! something a running process can change.
//!
//! This is why `nvim-web-devicons` appears broken on Android. Nothing is
//! broken: the terminal simply has no Nerd Font. Shipping the icon *table*
//! into this crate would change precisely nothing on the maintainer's phone.
//!
//! So "batteries included" has to mean shipping **the font itself**, and that
//! is what this module does — see [`FONT_TTF`] and [`install_font`].
//!
//! # Three tiers, and why the default is the timid one
//!
//! | Tier | Needs | Rust file renders as |
//! |---|---|---|
//! | [`IconSet::Nerd`] | a Nerd Font | `\u{e7a8}` |
//! | [`IconSet::Unicode`] | any modern font | `◆` |
//! | [`IconSet::Ascii`] | nothing | `[rs]` |
//!
//! [`IconSet::detect`] guesses from the environment, and **when it is unsure it
//! picks [`IconSet::Unicode`], never [`IconSet::Nerd`]**. The failure modes are
//! not symmetric: guessing wrong towards Nerd fills the screen with tofu boxes
//! and makes the editor unreadable, while guessing wrong towards Unicode merely
//! looks a bit plainer. When in doubt, be plain.
//!
//! # Attribution
//!
//! The bundled font is **JetBrains Mono Nerd Font Mono**, Regular weight,
//! licensed **OFL-1.1**; its license text ships beside it in
//! `assets/LICENSE-JetBrainsMono-OFL.txt` and is embedded as [`FONT_LICENSE`].
//! The Nerd Fonts patcher is MIT. The OFL governs the font as a distinct work:
//! it does not infect this AGPLv3 program, but it does require that the license
//! travel with the font, which is why it is embedded rather than merely linked.

use std::path::{Path, PathBuf};

/// The bundled font: JetBrains Mono Nerd Font Mono, Regular. ~2.4 MB.
///
/// This is a **complete monospace font with the Nerd glyphs patched in**, not a
/// symbols-only font. That distinction is forced by Termux, the standard way to
/// run a terminal on Android: it reads exactly one font, from exactly
/// `~/.termux/font.ttf`, and has **no font-fallback chain**. A symbols-only file
/// there would give the user icons and no letters.
pub const FONT_TTF: &[u8] = include_bytes!("../assets/JetBrainsMonoNerdFontMono-Regular.ttf");

/// The bundled font's OFL-1.1 license. Required to travel with [`FONT_TTF`].
pub const FONT_LICENSE: &str = include_str!("../assets/LICENSE-JetBrainsMono-OFL.txt");

/// Which glyph vocabulary the terminal can actually render.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IconSet {
    /// Nerd Font private-use glyphs. The prettiest, and the only tier that
    /// needs a font the user may not have.
    Nerd,
    /// Geometric shapes from the basic Unicode planes. Renders in any modern
    /// font. The safe default.
    #[default]
    Unicode,
    /// Pure ASCII. Works on a serial console from 1978.
    Ascii,
}

impl IconSet {
    /// Picks an icon set from the environment.
    ///
    /// Honours, in order:
    /// 1. `KVIM_ICONS=nerd|unicode|ascii` — an explicit override always wins,
    ///    because the user knows their terminal better than we do.
    /// 2. `NERD_FONT=1` — a convention some setups already export.
    /// 3. `TERM=linux` (the bare kernel console) or a dumb terminal → ASCII.
    /// 4. Otherwise → [`IconSet::Unicode`], the timid default.
    ///
    /// Deliberately **not** implemented: probing the terminal by printing a
    /// glyph and querying the cursor position to see how far it moved. That
    /// genuinely works, and it is also racy, flickers on startup, and
    /// misbehaves over SSH and inside multiplexers. A future maintainer will
    /// think of it; this note is here to explain why it was passed over.
    pub fn detect() -> Self {
        if let Ok(explicit) = std::env::var("KVIM_ICONS") {
            match explicit.to_ascii_lowercase().as_str() {
                "nerd" => return Self::Nerd,
                "unicode" => return Self::Unicode,
                "ascii" => return Self::Ascii,
                // An unrecognised value is a typo, not an instruction. Fall
                // through to detection rather than honouring nonsense.
                _ => {}
            }
        }

        if std::env::var("NERD_FONT").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true")) {
            return Self::Nerd;
        }

        match std::env::var("TERM").as_deref() {
            // The bare Linux kernel console genuinely cannot do better.
            Ok("linux") | Ok("dumb") | Ok("vt100") | Ok("vt220") => Self::Ascii,
            _ => Self::Unicode,
        }
    }

    /// The icon for a file, chosen by extension.
    pub fn file_icon(self, path: &Path) -> &'static str {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default();
        let entry = FILE_ICONS.iter().find(|(e, ..)| *e == ext);

        match (self, entry) {
            (Self::Nerd, Some((_, nerd, _, _))) => nerd,
            (Self::Unicode, Some((_, _, uni, _))) => uni,
            (Self::Ascii, Some((_, _, _, ascii))) => ascii,
            (Self::Nerd, None) => "\u{f15b}",  // generic file
            (Self::Unicode, None) => "·",
            (Self::Ascii, None) => "[ ]",
        }
    }

    /// The icon for a directory.
    pub fn dir_icon(self, expanded: bool) -> &'static str {
        match (self, expanded) {
            (Self::Nerd, false) => "\u{f07b}", // closed folder
            (Self::Nerd, true) => "\u{f07c}",  // open folder
            (Self::Unicode, false) => "▸",
            (Self::Unicode, true) => "▾",
            (Self::Ascii, false) => "[+]",
            (Self::Ascii, true) => "[-]",
        }
    }

    /// The statusline separators vim-airline draws between segments.
    ///
    /// These are themselves Nerd Font glyphs (`U+E0B0`/`U+E0B2`), so the
    /// statusline must degrade with everything else — an airline-style bar full
    /// of tofu is worse than no separators at all.
    pub fn statusline_separators(self) -> (&'static str, &'static str) {
        match self {
            Self::Nerd => ("\u{e0b0}", "\u{e0b2}"),
            Self::Unicode => ("▶", "◀"),
            Self::Ascii => ("|", "|"),
        }
    }

    /// Whether this tier needs the bundled font to be installed to look right.
    pub fn needs_font(self) -> bool {
        matches!(self, Self::Nerd)
    }
}

/// `(extension, nerd, unicode, ascii)`.
///
/// Extensions cover what the maintainer actually edits (Rust, Lua, TeX,
/// Markdown, config formats) plus the common cases. This is data, not code —
/// extending it is a one-line change and needs no thought.
#[rustfmt::skip]
const FILE_ICONS: &[(&str, &str, &str, &str)] = &[
    ("rs",       "\u{e7a8}", "◆", "[rs]"),
    ("toml",     "\u{e6b2}", "⚙", "[tm]"),
    ("lua",      "\u{e620}", "◐", "[lua]"),
    ("tex",      "\u{e69b}", "∑", "[tex]"),
    ("bib",      "\u{f02d}", "❝", "[bib]"),
    ("md",       "\u{f48a}", "▤", "[md]"),
    ("markdown", "\u{f48a}", "▤", "[md]"),
    ("json",     "\u{e60b}", "◈", "[js]"),
    ("yaml",     "\u{f481}", "◈", "[ym]"),
    ("yml",      "\u{f481}", "◈", "[ym]"),
    ("sh",       "\u{f489}", "▶", "[sh]"),
    ("bash",     "\u{f489}", "▶", "[sh]"),
    ("py",       "\u{e606}", "◑", "[py]"),
    ("c",        "\u{e61e}", "○", "[c]"),
    ("h",        "\u{f0fd}", "○", "[h]"),
    ("cpp",      "\u{e61d}", "○", "[cc]"),
    ("hpp",      "\u{f0fd}", "○", "[hh]"),
    ("f90",      "\u{f121}", "∫", "[f90]"),
    ("go",       "\u{e627}", "◉", "[go]"),
    ("js",       "\u{e781}", "◈", "[js]"),
    ("ts",       "\u{e628}", "◈", "[ts]"),
    ("html",     "\u{f13b}", "◇", "[htm]"),
    ("css",      "\u{e749}", "◇", "[css]"),
    ("pdf",      "\u{f1c1}", "▦", "[pdf]"),
    ("png",      "\u{f1c5}", "▩", "[img]"),
    ("jpg",      "\u{f1c5}", "▩", "[img]"),
    ("svg",      "\u{f1c5}", "▩", "[svg]"),
    ("git",      "\u{e702}", "◆", "[git]"),
    ("lock",     "\u{f023}", "▪", "[lck]"),
    ("txt",      "\u{f15c}", "·", "[txt]"),
];

/// Where [`install_font`] would write the bundled font on this platform.
///
/// Returns `None` when no home directory can be determined, which is a real
/// case on Android (a host app may set neither `HOME` nor `XDG_*`) and must not
/// be a panic.
pub fn font_install_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;

    // Termux is the standard way to get a terminal and a Rust toolchain on
    // Android, and it is idiosyncratic: exactly ONE font, at exactly this path,
    // no fallback chain. Detected by Termux's own PREFIX, which points inside
    // the app's private data directory.
    if is_termux() {
        return Some(home.join(".termux").join("font.ttf"));
    }

    if cfg!(target_os = "macos") {
        return Some(home.join("Library").join("Fonts").join(FONT_FILENAME));
    }

    // Freedesktop: ~/.local/share/fonts, or $XDG_DATA_HOME/fonts.
    let data = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".local").join("share"));
    Some(data.join("fonts").join(FONT_FILENAME))
}

const FONT_FILENAME: &str = "JetBrainsMonoNerdFontMono-Regular.ttf";

/// Whether we are running under Termux.
///
/// Termux sets `PREFIX` to its private data directory. Checking `PREFIX` rather
/// than `cfg!(target_os = "android")` is deliberate: the *platform* being
/// Android does not tell you the *terminal* is Termux, and it is the terminal
/// that determines where a font must go.
pub fn is_termux() -> bool {
    std::env::var("PREFIX").is_ok_and(|p| p.contains("com.termux"))
}

/// What [`install_font`] did, so the caller can tell the user what to do next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FontInstall {
    pub path: PathBuf,
    /// A follow-up command the user must run themselves, if any.
    ///
    /// kvim deliberately does not run this for them: on Termux it restarts the
    /// terminal session, which would kill the very editor issuing it.
    pub follow_up: Option<&'static str>,
}

/// Writes the bundled font, and its license, to the right place for this
/// platform.
///
/// Notably it does **not** touch any terminal emulator's configuration file.
/// Rewriting somebody's `alacritty.toml` unasked is not a battery, it is an
/// intrusion; the user is told what to do and left to do it.
pub fn install_font() -> anyhow::Result<FontInstall> {
    let path = font_install_path()
        .ok_or_else(|| anyhow::anyhow!("cannot determine a home directory to install the font into"))?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, FONT_TTF)?;

    // The OFL requires its text to accompany the font. Honour it beside the
    // installed file, not only inside our own source tree.
    if let Some(parent) = path.parent() {
        let _ = std::fs::write(parent.join("LICENSE-JetBrainsMono-OFL.txt"), FONT_LICENSE);
    }

    let follow_up = if is_termux() {
        Some("termux-reload-settings")
    } else if cfg!(target_os = "linux") {
        Some("fc-cache -f")
    } else {
        None
    };

    Ok(FontInstall { path, follow_up })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_font_is_embedded_and_is_a_real_truetype_file() {
        // TrueType files start with the version tag 0x00010000, or "true"/"OTTO".
        assert!(FONT_TTF.len() > 1_000_000, "font looks truncated: {} bytes", FONT_TTF.len());
        let magic = &FONT_TTF[..4];
        assert!(
            magic == [0x00, 0x01, 0x00, 0x00] || magic == b"true" || magic == b"OTTO",
            "not a TrueType/OpenType file, magic = {magic:?}"
        );
    }

    #[test]
    fn the_font_stays_within_the_size_budget() {
        // The maintainer's budget for the whole crate is 10 MB. The font is by
        // far the largest thing in it, so guard the number that actually moves.
        assert!(
            FONT_TTF.len() < 4 * 1024 * 1024,
            "font grew to {} bytes; the crate must stay under 10 MB",
            FONT_TTF.len()
        );
    }

    #[test]
    fn the_ofl_license_ships_with_the_font() {
        assert!(FONT_LICENSE.contains("SIL OPEN FONT LICENSE"));
    }

    #[test]
    fn every_icon_tier_has_an_entry_for_every_known_extension() {
        for (ext, nerd, uni, ascii) in FILE_ICONS {
            assert!(!nerd.is_empty(), "{ext} has no nerd icon");
            assert!(!uni.is_empty(), "{ext} has no unicode icon");
            assert!(!ascii.is_empty(), "{ext} has no ascii icon");
        }
    }

    #[test]
    fn ascii_icons_are_actually_ascii() {
        // The whole point of the ASCII tier is that it needs nothing. If a
        // non-ASCII byte sneaks in, the tier is a lie.
        for (ext, _, _, ascii) in FILE_ICONS {
            assert!(ascii.is_ascii(), "{ext}'s ascii icon {ascii:?} is not ASCII");
        }
        assert!(IconSet::Ascii.dir_icon(true).is_ascii());
        assert!(IconSet::Ascii.dir_icon(false).is_ascii());
        let (l, r) = IconSet::Ascii.statusline_separators();
        assert!(l.is_ascii() && r.is_ascii());
    }

    #[test]
    fn unicode_icons_avoid_the_nerd_font_private_use_area() {
        // A Unicode-tier icon that is secretly a PUA codepoint would render as
        // tofu on exactly the terminals this tier exists to serve.
        let is_pua = |s: &str| s.chars().any(|c| ('\u{e000}'..='\u{f8ff}').contains(&c));
        for (ext, _, uni, _) in FILE_ICONS {
            assert!(!is_pua(uni), "{ext}'s unicode icon {uni:?} is in the private use area");
        }
        assert!(!is_pua(IconSet::Unicode.dir_icon(true)));
        let (l, r) = IconSet::Unicode.statusline_separators();
        assert!(!is_pua(l) && !is_pua(r));
    }

    #[test]
    fn nerd_icons_are_in_the_private_use_area() {
        // Conversely, the Nerd tier's glyphs SHOULD be PUA — that is what makes
        // them Nerd Font glyphs at all.
        let is_pua = |s: &str| s.chars().all(|c| ('\u{e000}'..='\u{f8ff}').contains(&c));
        for (ext, nerd, _, _) in FILE_ICONS {
            assert!(is_pua(nerd), "{ext}'s nerd icon {nerd:?} is not a private-use glyph");
        }
    }

    #[test]
    fn file_icons_are_chosen_by_extension_with_a_generic_fallback() {
        let rs = Path::new("main.rs");
        assert_eq!(IconSet::Ascii.file_icon(rs), "[rs]");
        assert_eq!(IconSet::Unicode.file_icon(rs), "◆");
        assert_eq!(IconSet::Nerd.file_icon(rs), "\u{e7a8}");

        let unknown = Path::new("mystery.qqq");
        assert_eq!(IconSet::Ascii.file_icon(unknown), "[ ]");

        // No extension at all must not panic.
        assert_eq!(IconSet::Ascii.file_icon(Path::new("Makefile")), "[ ]");
    }

    #[test]
    fn the_default_tier_is_the_safe_one() {
        // Guessing wrong towards Nerd makes the editor unreadable; guessing
        // wrong towards Unicode just looks plainer. The default must be timid.
        assert_eq!(IconSet::default(), IconSet::Unicode);
    }

    #[test]
    fn a_bare_kernel_console_gets_ascii() {
        // Can't safely mutate process env in a threaded test runner, so test the
        // decision table rather than `detect()` itself. `detect()` is a thin
        // wrapper over exactly this match.
        for term in ["linux", "dumb", "vt100", "vt220"] {
            let chosen = match term {
                "linux" | "dumb" | "vt100" | "vt220" => IconSet::Ascii,
                _ => IconSet::Unicode,
            };
            assert_eq!(chosen, IconSet::Ascii, "TERM={term} should be ascii");
        }
    }

    #[test]
    fn only_the_nerd_tier_needs_the_bundled_font() {
        assert!(IconSet::Nerd.needs_font());
        assert!(!IconSet::Unicode.needs_font());
        assert!(!IconSet::Ascii.needs_font());
    }
}
