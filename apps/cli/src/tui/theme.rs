//! Shared kopitiam-themed palette and banner for the TUI app.
//!
//! Every view tints itself from this one palette so the whole app reads as a
//! single coffeeshop, and the colours live here (not scattered per view) so a
//! future re-theme is a one-file change. Truecolor terminals render the `Rgb`
//! values exactly; 256/16-colour terminals approximate them — nothing depends
//! on an exact hue.

use ratatui::style::Color;
use ratatui::text::{Line, Span};
use ratatui::style::Style;

/// Title, kopi-o, active accents.
pub const GOLD: Color = Color::Rgb(224, 170, 90);
/// Assistant / body text.
pub const TAN: Color = Color::Rgb(232, 205, 160);
/// Palm fronds / "ok" green.
pub const PALM: Color = Color::Rgb(104, 168, 88);
/// Steam / subtle secondary.
pub const STEAM: Color = Color::Rgb(180, 180, 170);
/// User text / info blue.
pub const USER: Color = Color::Rgb(120, 200, 220);
/// A warm red for failures.
pub const CHILLI: Color = Color::Rgb(210, 100, 80);
/// De-emphasised text and borders.
pub const DIM: Color = Color::DarkGray;

/// The kopitiam banner: two palm trees flanking a steaming cup of kopi-o, each
/// row tinted by its second field. Kept ASCII-only so it renders on any font.
#[rustfmt::skip]
pub const BANNER: &[(&str, Color)] = &[
    (r"   \\ | //          K O P I T I A M   ·   A I          \\ | //   ", GOLD),
    (r"  \ \|/ /        ~ your local kopi-o companion ~        \ \|/ /  ", STEAM),
    (r"   \\|//               (   (   )   )               \\|//   ",       PALM),
    (r"    \|/                 )   )   (                    \|/    ",      PALM),
    (r"     |                .-=============-.                |     ",     PALM),
    (r"     |                |               |               |     ",      GOLD),
    (r"    /|\               |    k o p i    |]             /|\    ",      GOLD),
    (r"   //|\\              |      -o-      |             //|\\   ",      GOLD),
    (r"  ///|\\\              \\_____________/             ///|\\\  ",     PALM),
];

/// Render a footer help line from `(key, description)` pairs: each key bold in
/// gold, its description dim, separated by two spaces. Shared by every view so
/// the footer looks the same everywhere.
pub fn help_line(pairs: &[(&str, &str)]) -> Line<'static> {
    use ratatui::style::Modifier;
    let mut spans: Vec<Span<'static>> = Vec::with_capacity(pairs.len() * 3);
    for (i, (key, desc)) in pairs.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("   ", Style::default().fg(DIM)));
        }
        spans.push(Span::styled(
            format!(" {key} "),
            Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            (*desc).to_string(),
            Style::default().fg(DIM),
        ));
    }
    Line::from(spans)
}
