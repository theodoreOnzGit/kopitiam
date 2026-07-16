//! Colour themes, as data.
//!
//! # Why a struct of colours, not a function that styles widgets
//!
//! A theme is a *fact* ("gruvbox's yellow is `#d79921`"), not *behaviour*.
//! Encoding it as Rust code that calls into ratatui styling APIs would tie
//! "what colour is this" to "how is it applied", making it impossible to
//! add a second theme without duplicating every styling call site, and
//! impossible to eventually load a theme from a config file (`Config::theme`
//! is already a string name) without a parser for a bespoke DSL. A plain
//! data struct sidesteps both: [`Theme::gruvbox_dark`] is the only
//! theme-specific code today, every other theme is just another struct
//! literal, and a future `Theme::from_config(name)` is a lookup, not a
//! compiler.
//!
//! # Attribution
//!
//! The gruvbox colour scheme and its exact hex values are the work of
//! **Pavel Pertsev** ("morhetz"), <https://github.com/morhetz/gruvbox>,
//! licensed under the MIT License. This module reproduces the published
//! "dark" palette's hard-contrast-independent base colours as data; no
//! gruvbox source code is copied, only the colour values, which is the part
//! every gruvbox port (terminal themes, editor themes, this one) necessarily
//! shares.

use ratatui::style::Color;

/// A named colour palette.
///
/// Field names follow gruvbox's own naming (`bg`, `fg`, `bg1`..`bg4`, and
/// the eight base hues each with a `_bright` neighbour) rather than
/// semantic names like `keyword` or `error`, because the palette is meant to
/// be reused by multiple consumers (statusline, syntax highlighting once it
/// exists, the colorcolumn guide) that each assign *meaning* to a colour
/// differently. Baking "red = error" into the theme itself would make the
/// theme know about its consumers, backwards from how a data-only theme
/// should work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Theme {
    /// Human-readable name, matching `Config::theme` (e.g. `"gruvbox"`).
    pub name: &'static str,

    /// Default background.
    pub bg: Color,
    /// Default foreground.
    pub fg: Color,

    /// Neutral background, one step lighter than `bg` — used for UI chrome
    /// like the statusline's darkest segment or the gutter background.
    pub bg1: Color,
    /// Neutral background, two steps lighter than `bg`.
    pub bg2: Color,
    /// Neutral background, three steps lighter than `bg` — visible dividers.
    pub bg3: Color,
    /// Neutral background, four steps lighter than `bg` — the lightest
    /// neutral, used sparingly (e.g. a very subtle selection highlight).
    pub bg4: Color,

    pub red: Color,
    pub red_bright: Color,
    pub green: Color,
    pub green_bright: Color,
    pub yellow: Color,
    pub yellow_bright: Color,
    pub blue: Color,
    pub blue_bright: Color,
    pub purple: Color,
    pub purple_bright: Color,
    pub aqua: Color,
    pub aqua_bright: Color,
    pub orange: Color,
    pub orange_bright: Color,

    /// Gruvbox's single mid-tone gray, used for de-emphasized text (e.g.
    /// inactive-window statuslines, comments).
    pub gray: Color,
}

impl Theme {
    /// gruvbox "dark", hard/medium contrast base colours — see the module
    /// docs for attribution.
    ///
    /// Hex values transcribed from the upstream palette:
    /// bg `#282828`, fg `#ebdbb2`, red `#cc241d`/`#fb4934`, green
    /// `#98971a`/`#b8bb26`, yellow `#d79921`/`#fabd2f`, blue
    /// `#458588`/`#83a598`, purple `#b16286`/`#d3869b`, aqua
    /// `#689d6a`/`#8ec07c`, orange `#d65d0e`/`#fe8019`, gray `#928374`,
    /// bg1 `#3c3836`, bg2 `#504945`, bg3 `#665c54`, bg4 `#7c6f64`.
    pub fn gruvbox_dark() -> Self {
        Self {
            name: "gruvbox",
            bg: hex("#282828"),
            fg: hex("#ebdbb2"),
            bg1: hex("#3c3836"),
            bg2: hex("#504945"),
            bg3: hex("#665c54"),
            bg4: hex("#7c6f64"),
            red: hex("#cc241d"),
            red_bright: hex("#fb4934"),
            green: hex("#98971a"),
            green_bright: hex("#b8bb26"),
            yellow: hex("#d79921"),
            yellow_bright: hex("#fabd2f"),
            blue: hex("#458588"),
            blue_bright: hex("#83a598"),
            purple: hex("#b16286"),
            purple_bright: hex("#d3869b"),
            aqua: hex("#689d6a"),
            aqua_bright: hex("#8ec07c"),
            orange: hex("#d65d0e"),
            orange_bright: hex("#fe8019"),
            gray: hex("#928374"),
        }
    }

    /// The background a visual-mode selection is painted with.
    ///
    /// A *method*, not a field, on purpose. This module's whole design is that a
    /// theme is a palette (facts about colours), not a stylesheet (facts about
    /// widgets) — see the module docs on why field names are `bg2`, not
    /// `selection`. But "which palette entry does the selection use" is still a
    /// decision that must live in exactly one place, or the day a second theme
    /// lands, every call site has to be found and re-decided. An accessor gives
    /// that single place without putting a consumer's vocabulary into the
    /// palette's data.
    ///
    /// `bg2` (`#504945`) is the mid neutral: a clear lift from `bg` (`#282828`)
    /// while leaving gruvbox's `fg` (`#ebdbb2`) comfortably readable on top of
    /// it, which a selection highlight must do — it recolours the *background*
    /// of text the user is still reading. (Upstream gruvbox's own `Visual` group
    /// depends on its `invert_selection` option, so this picks a value from the
    /// published palette rather than claiming to reproduce a specific highlight
    /// group.)
    pub fn selection_bg(&self) -> Color {
        self.bg2
    }

    /// The background a search match (`'hlsearch'`/`'incsearch'`) kena painted
    /// in.
    ///
    /// Gruvbox's own `Search` group is `bg = yellow, fg = bg0` — a bright fill
    /// with dark text on top — and that one is what this return for the fill (the
    /// renderer pair it with [`Self::search_fg`] for the text). Bright yellow is
    /// deliberately not the same as [`Self::selection_bg`]'s muted `bg2`: the two
    /// must be tell-apart-able at a glance where a search match sit inside a
    /// visual selection, and the selection is the one that win the cell. `yellow`
    /// (`#d79921`), not the brighter `yellow_bright`, keep it from vibrating
    /// against gruvbox's cream foreground on nearby unmatched text.
    pub fn search_bg(&self) -> Color {
        self.yellow
    }

    /// The foreground a search match's text kena drawn in, over
    /// [`Self::search_bg`]. Gruvbox put dark text (`bg`, `#282828`) on the yellow
    /// fill so the match stay legible — a search highlight replace the cell's
    /// colours outright (not like a selection, which only tint the background),
    /// because vim's `Search` group set both.
    pub fn search_fg(&self) -> Color {
        self.bg
    }

    /// Looks up a theme by the name used in [`crate::config::Config::theme`]
    /// (case-insensitive). Falls back to [`Theme::gruvbox_dark`] for any
    /// unrecognised name — `Config::theme` is a plain `String`, so a typo'd
    /// or not-yet-implemented theme name should degrade to *something*
    /// usable rather than fail config loading outright. As more themes are
    /// added as data (see the module docs), this match grows; it never
    /// needs to become anything other than a lookup.
    pub fn from_name(name: &str) -> Self {
        match name.to_ascii_lowercase().as_str() {
            "gruvbox" | "gruvbox-dark" | "gruvbox_dark" => Self::gruvbox_dark(),
            _ => Self::gruvbox_dark(),
        }
    }
}

impl Default for Theme {
    /// `Config::theme` defaults to `"gruvbox"`, so the theme type's own
    /// default matches — a caller that hasn't wired up theme lookup yet
    /// still gets the maintainer's actual colours, not an arbitrary ratatui
    /// default.
    fn default() -> Self {
        Self::gruvbox_dark()
    }
}

/// Parses a `#rrggbb` literal into a ratatui `Color::Rgb`.
///
/// Private and `const`-unfriendly-free (uses `u8::from_str_radix`, not
/// `const fn`, because `Theme::gruvbox_dark` is called at most once per
/// theme switch, not per frame — there is no need to push this to compile
/// time at the cost of readability).
fn hex(s: &str) -> Color {
    let s = s.strip_prefix('#').unwrap_or(s);
    debug_assert_eq!(s.len(), 6, "expected a 6-digit hex colour, got {s:?}");
    let r = u8::from_str_radix(&s[0..2], 16).expect("valid hex red component");
    let g = u8::from_str_radix(&s[2..4], 16).expect("valid hex green component");
    let b = u8::from_str_radix(&s[4..6], 16).expect("valid hex blue component");
    Color::Rgb(r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_parses_correctly() {
        assert_eq!(hex("#282828"), Color::Rgb(0x28, 0x28, 0x28));
        assert_eq!(hex("#ebdbb2"), Color::Rgb(0xeb, 0xdb, 0xb2));
    }

    #[test]
    fn gruvbox_bg_and_fg_match_the_published_palette() {
        let t = Theme::gruvbox_dark();
        assert_eq!(t.bg, Color::Rgb(0x28, 0x28, 0x28));
        assert_eq!(t.fg, Color::Rgb(0xeb, 0xdb, 0xb2));
    }

    #[test]
    fn gruvbox_bright_variants_match_the_published_palette() {
        let t = Theme::gruvbox_dark();
        assert_eq!(t.red, Color::Rgb(0xcc, 0x24, 0x1d));
        assert_eq!(t.red_bright, Color::Rgb(0xfb, 0x49, 0x34));
        assert_eq!(t.green_bright, Color::Rgb(0xb8, 0xbb, 0x26));
        assert_eq!(t.yellow_bright, Color::Rgb(0xfa, 0xbd, 0x2f));
        assert_eq!(t.orange_bright, Color::Rgb(0xfe, 0x80, 0x19));
    }

    #[test]
    fn gruvbox_neutral_backgrounds_match_the_published_palette() {
        let t = Theme::gruvbox_dark();
        assert_eq!(t.bg1, Color::Rgb(0x3c, 0x38, 0x36));
        assert_eq!(t.bg2, Color::Rgb(0x50, 0x49, 0x45));
        assert_eq!(t.bg3, Color::Rgb(0x66, 0x5c, 0x54));
        assert_eq!(t.bg4, Color::Rgb(0x7c, 0x6f, 0x64));
    }

    #[test]
    fn default_theme_is_gruvbox_matching_configs_default() {
        assert_eq!(Theme::default().name, "gruvbox");
    }

    #[test]
    fn from_name_looks_up_gruvbox_case_insensitively() {
        assert_eq!(Theme::from_name("gruvbox").name, "gruvbox");
        assert_eq!(Theme::from_name("GruvBox").name, "gruvbox");
    }

    #[test]
    fn from_name_falls_back_to_gruvbox_for_unknown_names() {
        assert_eq!(Theme::from_name("solarized").name, "gruvbox");
    }
}
