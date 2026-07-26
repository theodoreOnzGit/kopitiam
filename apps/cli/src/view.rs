//! `kopitiam view <pdf>` — a standalone, on-screen terminal PDF viewer.
//!
//! # What it is
//!
//! This opens a PDF and renders its pages **as images**, straight in the
//! terminal, via the ported MuPDF rasteriser
//! ([`kopitiam_pdf::mupdf::rasterize_page`], reached through
//! [`crate::tui::viewer::MupdfRasterizer`]) and [`ratatui_image`]. It is the
//! command `kmux latex` shells out to for its live LaTeX preview, so it must run
//! standalone against a real terminal — which it does: it owns the terminal
//! (raw mode + alternate screen), runs its own event loop, and restores the
//! terminal on exit or panic.
//!
//! # One shared rendering path
//!
//! There is **no forked viewer logic**. This command drives the very same
//! [`crate::tui::viewer::ViewDoc`] the `kopitiam tui` image mode uses — the same
//! `Pixmap` → `RgbaImage` conversion, the same page/zoom/goto navigation, the
//! same `ratatui-image` display. The only thing this file adds is the terminal
//! lifecycle and an event loop whose `q`/`Esc`/`Ctrl-C` quit the process (rather
//! than returning to a TUI home menu).
//!
//! # Keybindings
//!
//! * `j` / `n` / `→` / `PgDn` — next page; `k` / `p` / `←` / `PgUp` — previous.
//! * `g` then digits then `Enter` — go to a page (clamped into range).
//! * `+` / `-` — zoom in / out (raises / lowers the render dpi).
//! * `r` / `i` / `Tab` — toggle reflow (Markdown) / image modes.
//! * `q` / `Esc` / `Ctrl-C` — quit.
//!
//! # Android / Termux
//!
//! `ratatui-image` auto-detects the terminal's graphics protocol (kitty / sixel
//! / iTerm2) and **falls back to Unicode half-blocks** where none is available —
//! which is what keeps this viewer working under Termux (whose terminal has no
//! graphics protocol). Everything else is crossterm-based with no platform
//! syscalls of our own.

use std::io::{self, Stdout};
use std::time::Duration;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    crossterm::{
        event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    },
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};

use crate::tui::viewer::{MupdfRasterizer, ViewDoc};

/// Arguments for `kopitiam view`.
#[derive(Args, Debug)]
pub struct ViewArgs {
    /// The PDF file to open.
    pub pdf: PathBuf,

    /// Initial 1-based page to open on. Clamped into range if out of bounds.
    #[arg(long, default_value_t = 1)]
    pub page: usize,

    /// Initial render resolution in DPI. Higher is sharper but slower to render;
    /// adjust live with `+` / `-` once open.
    #[arg(long, default_value_t = 150.0)]
    pub dpi: f32,
}

/// Entry point for `kopitiam view`. Opens `args.pdf` in image mode, sets up the
/// terminal (restoring it even on panic), runs the event loop, and restores.
pub fn run(args: ViewArgs) -> Result<()> {
    if !args.pdf.exists() {
        anyhow::bail!("no such file: {}", args.pdf.display());
    }

    // The same ViewDoc + MupdfRasterizer the TUI image mode uses. `open_image`
    // starts in image mode at the requested (clamped) page and dpi; the reflow
    // pipeline stays lazy and only runs if the user toggles to it.
    let mut doc = ViewDoc::open_image(
        args.pdf.clone(),
        Box::new(MupdfRasterizer::new()),
        Some(args.page.max(1)),
        Some(args.dpi),
    );

    let mut terminal = setup_terminal().context("failed to initialise the terminal")?;
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = restore_terminal();
        previous_hook(info);
    }));

    let result = event_loop(&mut terminal, &mut doc);

    restore_terminal().ok();
    let _ = std::panic::take_hook();
    result
}

type Tui = Terminal<CrosstermBackend<Stdout>>;

fn setup_terminal() -> Result<Tui> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(stdout)).map_err(Into::into)
}

/// Undo [`setup_terminal`]. Idempotent and best-effort so it is safe from the
/// panic hook and again on the normal path.
fn restore_terminal() -> Result<()> {
    let mut stdout = io::stdout();
    execute!(stdout, LeaveAlternateScreen)?;
    disable_raw_mode()?;
    Ok(())
}

/// The viewer loop: paint, then handle at most one key per ~30 ms tick. Returns
/// `Ok(())` when the user quits.
fn event_loop(terminal: &mut Tui, doc: &mut ViewDoc) -> Result<()> {
    loop {
        // If the user has toggled to reflow, run its (possibly slow) pipeline
        // once so image mode never pays for text reflow it does not show.
        if doc.needs_reflow_render() {
            doc.render_now();
        }

        terminal.draw(|frame| render(frame, doc))?;

        if event::poll(Duration::from_millis(30))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            if should_quit(key, doc) {
                return Ok(());
            }
            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
            // The transition (Home/Quit/Stay) is meaningless standalone — quit is
            // handled above — so navigation/zoom/goto is all that matters here.
            let _ = doc.on_key(key, ctrl);
        }
    }
}

/// Whether `key` should end the standalone session. `Ctrl-C` always quits; `q`
/// and `Esc` quit only when the document is not capturing text into its search
/// or goto box (so those keys can edit/cancel the box instead).
fn should_quit(key: KeyEvent, doc: &ViewDoc) -> bool {
    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('c')) {
        return true;
    }
    !doc.is_capturing_text() && matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
}

/// Paint the document into the body and a one-line footer of key hints below it.
fn render(frame: &mut Frame, doc: &mut ViewDoc) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(1)])
        .split(frame.area());
    doc.render(frame, chunks[0]);
    frame.render_widget(Paragraph::new(footer_line(&doc.footer_hints())), chunks[1]);
}

/// Build the footer key-hint line from the document's own hints, but with the
/// TUI-specific "Esc home" entry dropped and a standalone "q/Esc quit" appended,
/// so the footer tells the truth about what the keys do here.
fn footer_line(hints: &[(&str, &str)]) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    for (k, v) in hints {
        if *v == "home" {
            continue; // standalone: Esc quits, it does not go "home"
        }
        spans.push(Span::styled(format!(" {k} "), Style::default().add_modifier(Modifier::BOLD)));
        spans.push(Span::raw(format!("{v}   ")));
    }
    spans.push(Span::styled(" q/Esc ", Style::default().add_modifier(Modifier::BOLD)));
    spans.push(Span::raw("quit"));
    Line::from(spans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// A tiny parser wrapping [`ViewArgs`] so its clap derivation (defaults and
    /// flag overrides) is exercised without spinning up a terminal.
    #[derive(Parser)]
    struct Wrap {
        #[command(flatten)]
        args: ViewArgs,
    }

    #[test]
    fn arg_defaults_are_page_1_and_dpi_150() {
        let w = Wrap::parse_from(["kopitiam-view", "paper.pdf"]);
        assert_eq!(w.args.pdf, PathBuf::from("paper.pdf"));
        assert_eq!(w.args.page, 1);
        assert_eq!(w.args.dpi, 150.0);
    }

    #[test]
    fn arg_flags_override_defaults() {
        let w = Wrap::parse_from(["kopitiam-view", "paper.pdf", "--page", "7", "--dpi", "300"]);
        assert_eq!(w.args.page, 7);
        assert_eq!(w.args.dpi, 300.0);
    }

    #[test]
    fn footer_drops_home_and_appends_quit() {
        let line = footer_line(&[("←/→ n/p", "page"), ("Esc", "home")]);
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        // The "home" hint is filtered out; a quit hint is always present.
        assert!(!text.contains("home"));
        assert!(text.contains("quit"));
        assert!(text.contains("page"));
    }

    #[test]
    fn should_quit_respects_ctrl_c_and_plain_q_esc() {
        // A doc not capturing text: q and Esc quit, Ctrl-C quits, other keys don't.
        let doc = ViewDoc::open_image(PathBuf::from("/x/p.pdf"), Box::new(MupdfRasterizer::new()), Some(1), None);
        let plain = |c: KeyCode| KeyEvent::new(c, KeyModifiers::empty());
        assert!(should_quit(plain(KeyCode::Char('q')), &doc));
        assert!(should_quit(plain(KeyCode::Esc), &doc));
        assert!(should_quit(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL), &doc));
        assert!(!should_quit(plain(KeyCode::Char('n')), &doc));
    }
}
