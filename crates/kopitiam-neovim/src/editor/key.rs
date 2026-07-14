//! `Key`: the alphabet `Editor::handle_key` reads.
//!
//! This type deliberately does **not** depend on `crossterm`. The UI layer
//! owns the terminal and maps whatever `crossterm::event::KeyEvent` it reads
//! onto this type; everything below `ui` — this whole crate, in fact — stays
//! testable without a terminal at all, which is what makes the keystroke
//! harness in `mod.rs`'s tests possible. See `docs/ai-decisions/AID-0003`,
//! decision on engine/UI separation.

use std::fmt;

/// A physical key, independent of what it happens to mean in the current
/// mode (that meaning is [`Editor::handle_key`](super::Editor::handle_key)'s
/// job to decide).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyCode {
    Char(char),
    Enter,
    Esc,
    Backspace,
    Tab,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    PageUp,
    PageDown,
    Delete,
    F(u8),
}

/// Which modifier keys were held. `shift` is tracked for completeness (e.g.
/// `<S-Tab>`) even though most letter input already encodes case in
/// [`KeyCode::Char`] and so never needs it set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Modifiers {
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
}

/// One key event: the code, plus whatever modifiers were held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Key {
    pub code: KeyCode,
    pub mods: Modifiers,
}

impl Key {
    pub const fn new(code: KeyCode, mods: Modifiers) -> Self {
        Self { code, mods }
    }

    /// A plain, unmodified character key — the common case.
    pub const fn char(c: char) -> Self {
        Self { code: KeyCode::Char(c), mods: Modifiers { ctrl: false, alt: false, shift: false } }
    }

    pub const fn ctrl(c: char) -> Self {
        Self { code: KeyCode::Char(c), mods: Modifiers { ctrl: true, alt: false, shift: false } }
    }

    pub const fn esc() -> Self {
        Self { code: KeyCode::Esc, mods: Modifiers { ctrl: false, alt: false, shift: false } }
    }

    pub const fn enter() -> Self {
        Self { code: KeyCode::Enter, mods: Modifiers { ctrl: false, alt: false, shift: false } }
    }

    pub const fn backspace() -> Self {
        Self { code: KeyCode::Backspace, mods: Modifiers { ctrl: false, alt: false, shift: false } }
    }

    /// The plain character this key types in Insert mode, if any. Control
    /// and navigation keys have none.
    pub fn as_char(self) -> Option<char> {
        match self.code {
            KeyCode::Char(c) if !self.mods.ctrl && !self.mods.alt => Some(c),
            _ => None,
        }
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self.code {
            KeyCode::Char(c) => return write!(f, "{}{c}", if self.mods.ctrl { "C-" } else { "" }),
            KeyCode::Enter => "CR",
            KeyCode::Esc => "Esc",
            KeyCode::Backspace => "BS",
            KeyCode::Tab => "Tab",
            KeyCode::Left => "Left",
            KeyCode::Right => "Right",
            KeyCode::Up => "Up",
            KeyCode::Down => "Down",
            KeyCode::Home => "Home",
            KeyCode::End => "End",
            KeyCode::PageUp => "PageUp",
            KeyCode::PageDown => "PageDown",
            KeyCode::Delete => "Del",
            KeyCode::F(n) => return write!(f, "<F{n}>"),
        };
        write!(f, "<{name}>")
    }
}

/// Parses vim key notation (`"dw"`, `"ci("`, `"<Esc>"`, `"<C-r>"`) into a key
/// sequence. Used both by the `run()` test harness in `mod.rs`'s tests and
/// to compile [`crate::config::Keymap::lhs`] strings into matchable
/// sequences (after substituting `<leader>` — see
/// [`super::Editor::compile_keymaps`], since the *token* `<leader>` is not a
/// physical key at all, just config-file shorthand for whatever key the user
/// configured).
pub fn parse(s: &str) -> Vec<Key> {
    let mut out = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '<' {
            let mut token = String::new();
            let mut closed = false;
            for c2 in chars.by_ref() {
                if c2 == '>' {
                    closed = true;
                    break;
                }
                token.push(c2);
            }
            if closed {
                if let Some(key) = parse_token(&token) {
                    out.push(key);
                    continue;
                }
                // Unknown token: fall back to its literal characters so a
                // typo doesn't silently eat input.
                out.push(Key::char('<'));
                out.extend(token.chars().map(Key::char));
                out.push(Key::char('>'));
                continue;
            }
            out.push(Key::char('<'));
            out.extend(token.chars().map(Key::char));
            continue;
        }
        out.push(Key::char(c));
    }
    out
}

fn parse_token(token: &str) -> Option<Key> {
    let lower = token.to_ascii_lowercase();
    let named = |code: KeyCode| Some(Key::new(code, Modifiers::default()));
    match lower.as_str() {
        "esc" | "escape" => return named(KeyCode::Esc),
        "cr" | "enter" | "return" => return named(KeyCode::Enter),
        "bs" | "backspace" => return named(KeyCode::Backspace),
        "tab" => return named(KeyCode::Tab),
        "left" => return named(KeyCode::Left),
        "right" => return named(KeyCode::Right),
        "up" => return named(KeyCode::Up),
        "down" => return named(KeyCode::Down),
        "home" => return named(KeyCode::Home),
        "end" => return named(KeyCode::End),
        "pageup" => return named(KeyCode::PageUp),
        "pagedown" => return named(KeyCode::PageDown),
        "del" | "delete" => return named(KeyCode::Delete),
        "space" => return Some(Key::char(' ')),
        "leader" => return Some(Key::char(' ')), // fallback; real substitution happens before parse()
        _ => {}
    }
    if let Some(rest) = lower.strip_prefix("f")
        && let Ok(n) = rest.parse::<u8>()
    {
        return named(KeyCode::F(n));
    }
    // Modifier prefixes: C-, A-, S- (possibly combined, e.g. "C-A-x").
    let mut ctrl = false;
    let mut alt = false;
    let mut shift = false;
    let mut rest = token;
    loop {
        let mut prefix_chars = rest.chars();
        match (prefix_chars.next(), prefix_chars.next()) {
            (Some('C'), Some('-')) | (Some('c'), Some('-')) => {
                ctrl = true;
                rest = &rest[2..];
            }
            (Some('A'), Some('-')) | (Some('a'), Some('-')) => {
                alt = true;
                rest = &rest[2..];
            }
            (Some('S'), Some('-')) | (Some('s'), Some('-')) => {
                shift = true;
                rest = &rest[2..];
            }
            _ => break,
        }
    }
    if rest.chars().count() == 1 {
        let c = rest.chars().next().unwrap();
        return Some(Key::new(KeyCode::Char(c), Modifiers { ctrl, alt, shift }));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_characters() {
        assert_eq!(parse("dw"), vec![Key::char('d'), Key::char('w')]);
    }

    #[test]
    fn parses_named_and_modified_tokens() {
        assert_eq!(parse("<Esc>"), vec![Key::esc()]);
        assert_eq!(parse("<CR>"), vec![Key::enter()]);
        assert_eq!(parse("<C-r>"), vec![Key::ctrl('r')]);
    }

    #[test]
    fn mixes_named_tokens_with_plain_chars() {
        assert_eq!(parse("ci(<Esc>"), vec![Key::char('c'), Key::char('i'), Key::char('('), Key::esc()]);
    }
}
