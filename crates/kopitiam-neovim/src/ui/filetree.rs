//! The file-tree sidebar: the *presentation* of [`plugins::filetree::FileTree`].
//!
//! # Where the line is drawn
//!
//! Per `CLAUDE.md` ("never place business logic inside user interfaces"), every
//! question of the form *"what does the tree look like now?"* is answered by the
//! engine, not here. Expanding, collapsing, re-rooting, hiding dotfiles, reading
//! directories lazily, sorting directories before files — all of that is
//! [`FileTree`], and this module never reimplements a line of it. What lives
//! here is exactly the two things a UI owns:
//!
//! 1. **Translation** — a [`KeyPress`] becomes a call on [`FileTree`]. `o` is
//!    [`FileTree::toggle`], `X` is [`FileTree::collapse_all`], `I` is
//!    [`FileTree::toggle_hidden`]. The mapping table is [`FileTreePanel::handle_key`].
//! 2. **Rendering** — [`FileTree::render`]'s flat `Vec<TreeRow>` becomes
//!    terminal cells: indent by `depth`, an icon from [`IconSet`], a highlight
//!    on the selected row.
//!
//! The one piece of state that is genuinely the UI's own is the **selection**
//! (which row the cursor is on). The engine has no opinion about it and should
//! not: two sidebars onto the same tree would each have their own cursor, the
//! same way two vim windows onto one buffer do.
//!
//! # The permission-denied row, and a gap in the engine
//!
//! [`FileTree`] reads directories through `ignore::Walk`, which reports an
//! unreadable directory as an `Err` *entry* — and the engine filters entries
//! with `.filter_map(|e| e.ok())`. The consequence is that expanding a directory
//! you have no permission to read succeeds, yields zero children, and is
//! therefore **indistinguishable from an empty directory**. Silently drawing an
//! empty folder over a permissions error is precisely the kind of quiet lie this
//! project does not ship.
//!
//! Since the engine is finished and not ours to change, the panel probes
//! readability itself, once, at the moment the user asks to expand — a single
//! `read_dir` syscall — and remembers the failure so it can draw an honest error
//! row beneath the folder. This is *not* tree logic (it changes nothing about
//! what the tree contains); it is the panel refusing to render something it
//! cannot vouch for.
//!
//! The clean fix belongs in the engine: `TreeRow` growing an
//! `error: Option<String>`, or `read_children` surfacing the `io::Error` instead
//! of dropping it. That is recorded as follow-up work rather than smuggled in
//! here.
//!
//! [`plugins::filetree::FileTree`]: crate::plugins::filetree::FileTree
//! [`FileTree`]: crate::plugins::filetree::FileTree

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Modifier, Style},
    widgets::Widget,
    Frame,
};

use crate::icons::IconSet;
use crate::plugins::filetree::{FileTree, TreeRow};
use crate::ui::event::{Key, KeyPress};
use crate::ui::overlay::{OpenTarget, OverlayOutcome};
use crate::ui::theme::Theme;

/// One drawable row of the sidebar.
///
/// Almost always a [`TreeRow`] straight from the engine. The exception is
/// [`Row::Error`], which exists only because of the permission gap described in
/// the module docs — it is a row the *panel* invents to tell the truth about a
/// directory the engine could not read.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Row {
    Tree(TreeRow),
    Error { depth: usize, message: String },
}

impl Row {
    fn depth(&self) -> usize {
        match self {
            Row::Tree(row) => row.depth,
            Row::Error { depth, .. } => *depth,
        }
    }

    /// The tree row this is, if it is one — the single place the panel asks
    /// "can I act on the selection?".
    fn as_tree(&self) -> Option<&TreeRow> {
        match self {
            Row::Tree(row) => Some(row),
            Row::Error { .. } => None,
        }
    }
}

/// The sidebar's state: the engine, plus the selection and view state that are
/// the UI's own.
pub struct FileTreePanel {
    tree: FileTree,
    /// Index into [`FileTreePanel::rows`], not into `FileTree::render()` — the
    /// two differ whenever an error row is present.
    selected: usize,
    /// First visible row. Recomputed against the real height on every draw (see
    /// [`FileTreePanel::render`]), because the panel does not know how tall it
    /// is until it is asked to draw.
    scroll_top: usize,
    /// `?` replaces the tree with the keymap table until the next key.
    help: bool,
    /// The configured leader (Space, in the maintainer's config), so that
    /// `<leader>e` closes the tree from *inside* it — the same keystroke that
    /// opened it. Key *sequences* are the UI's business (see `ui/mod.rs`); the
    /// editor's own keymap engine never sees these keys, because the editor is
    /// not the thing with focus.
    leader: char,
    leader_pending: bool,
    /// Directories that could not be read, and why. See the module docs.
    unreadable: BTreeMap<PathBuf, String>,
}

impl FileTreePanel {
    /// The sidebar's preferred width, in columns. neo-tree's default is 40;
    /// 30 is chosen instead because kvim also runs on a phone, where 40 columns
    /// is most of the screen. [`crate::ui::overlay::OverlayPlacement::split`]
    /// clamps it further on a genuinely narrow terminal.
    pub const WIDTH: u16 = 30;

    /// Opens a sidebar rooted at `root`.
    ///
    /// Fails only if the root itself cannot be opened, which the caller reports
    /// on the command line — an editor that dies because `<leader>e` was pressed
    /// in an odd directory would be a poor trade.
    pub fn open(root: &Path, leader: char) -> io::Result<Self> {
        Ok(Self {
            tree: FileTree::new(root)?,
            selected: 0,
            scroll_top: 0,
            help: false,
            leader,
            leader_pending: false,
            unreadable: BTreeMap::new(),
        })
    }

    /// The tree's current root — changes with `u`/`U`/`P` and `C`.
    pub fn root(&self) -> &Path {
        self.tree.root_path()
    }

    /// Every visible row: the engine's flattened tree, with an error row spliced
    /// in under any expanded directory the panel knows it could not read.
    fn rows(&self) -> Vec<Row> {
        let mut out = Vec::new();
        for row in self.tree.render() {
            let unreadable = row.is_dir
                .then(|| self.unreadable.get(&row.path))
                .flatten()
                .filter(|_| row.is_expanded)
                .cloned();
            let depth = row.depth;
            out.push(Row::Tree(row));
            if let Some(message) = unreadable {
                out.push(Row::Error { depth: depth + 1, message });
            }
        }
        out
    }

    /// The path under the cursor, if the cursor is on a real tree row.
    fn selected_row(&self) -> Option<TreeRow> {
        self.rows().get(self.selected).and_then(Row::as_tree).cloned()
    }

    /// Keeps the selection inside the row list after an operation that can
    /// shrink it (collapse, hide dotfiles, re-root).
    fn clamp_selection(&mut self) {
        let len = self.rows().len();
        self.selected = self.selected.min(len.saturating_sub(1));
    }

    /// Expands `path`, first checking that it can actually be read.
    ///
    /// See the module docs: the engine cannot distinguish "empty" from
    /// "forbidden", so the panel asks the filesystem directly rather than draw a
    /// folder it has no evidence is empty. The directory is expanded either way
    /// — an expanded folder with an error row under it says exactly what
    /// happened, whereas refusing to expand would look like a dropped keypress.
    fn expand(&mut self, path: &Path) -> io::Result<()> {
        match std::fs::read_dir(path) {
            Ok(_) => {
                self.unreadable.remove(path);
            }
            Err(e) => {
                self.unreadable.insert(path.to_path_buf(), e.to_string());
            }
        }
        self.tree.expand(path)
    }

    /// Translates one keypress into a call on the engine.
    ///
    /// The table is the maintainer's neo-tree config, which sets NERDTree-style
    /// in-tree mappings — `o t i s O x X R q ? I u U P C` — plus `j`/`k` (and the
    /// arrow keys) to move, `<CR>` as a synonym for `o`, and `<Esc>` / `<leader>e`
    /// to close. Anything else is [`OverlayOutcome::Ignored`]: an unmapped key in
    /// a NERDTree window does nothing, and must not fall through to the editor
    /// (which does not have focus).
    pub fn handle_key(&mut self, key: KeyPress) -> OverlayOutcome {
        // `<leader>e` from inside the tree closes it. The leader is consumed
        // first, before any single-key mapping, so that a leader that happens to
        // collide with a tree key (a user could set `mapleader = "i"`) still
        // begins a sequence rather than opening a split.
        if std::mem::take(&mut self.leader_pending) {
            if key.key == Key::Char('e') {
                return OverlayOutcome::Close;
            }
            // Not `<leader>e`: fall through and let the key mean what it
            // ordinarily means, rather than swallowing it.
        } else if key.key == Key::Char(self.leader) {
            self.leader_pending = true;
            return OverlayOutcome::Ignored;
        }

        // The help screen is modal and trivially dismissible: any key returns to
        // the tree. That key is *not* then acted on — someone reading the help
        // and pressing `X` means "close the help", not "collapse everything".
        if self.help {
            self.help = false;
            return OverlayOutcome::Consumed;
        }

        match key.key {
            Key::Char('j') | Key::Down => self.move_selection(1),
            Key::Char('k') | Key::Up => self.move_selection(-1),

            // `o` / `<CR>`: open a file, or toggle a folder.
            Key::Char('o') | Key::Enter => self.activate(OpenTarget::Current),
            Key::Char('t') => self.open_file(OpenTarget::Tab),
            Key::Char('i') => self.open_file(OpenTarget::HorizontalSplit),
            Key::Char('s') => self.open_file(OpenTarget::VerticalSplit),

            // `O`: expand everything under the node. On a file, NERDTree's `O`
            // simply opens it, so it does that too.
            Key::Char('O') => self.expand_all(),
            // `x`: close the selected node's *parent*, and put the cursor on it.
            Key::Char('x') => self.close_parent(),
            Key::Char('X') => {
                self.tree.collapse_all();
                self.selected = 0;
                self.scroll_top = 0;
                OverlayOutcome::Consumed
            }
            Key::Char('R') => self.refresh(),
            Key::Char('I') => {
                self.tree.toggle_hidden();
                self.clamp_selection();
                OverlayOutcome::Consumed
            }
            // `u`, `U` and `P` all mean "go up to the parent" in NERDTree (they
            // differ only in what they do to the cursor, which kvim does not
            // distinguish yet — see the report).
            Key::Char('u') | Key::Char('U') | Key::Char('P') => self.navigate_up(),
            Key::Char('C') => self.set_root(),

            Key::Char('?') => {
                self.help = true;
                OverlayOutcome::Consumed
            }
            Key::Char('q') | Key::Escape => OverlayOutcome::Close,

            _ => OverlayOutcome::Ignored,
        }
    }

    fn move_selection(&mut self, delta: isize) -> OverlayOutcome {
        let len = self.rows().len();
        if len == 0 {
            return OverlayOutcome::Ignored;
        }
        let next = self.selected.saturating_add_signed(delta).min(len - 1);
        if next == self.selected {
            // Already at an edge: nothing changed, so do not ask for a redraw.
            return OverlayOutcome::Ignored;
        }
        self.selected = next;
        OverlayOutcome::Consumed
    }

    /// `o`/`<CR>`: a directory toggles, a file opens.
    fn activate(&mut self, target: OpenTarget) -> OverlayOutcome {
        let Some(row) = self.selected_row() else { return OverlayOutcome::Ignored };
        if !row.is_dir {
            return OverlayOutcome::OpenPath { path: row.path, target };
        }
        let result = if row.is_expanded {
            self.tree.collapse(&row.path);
            Ok(())
        } else {
            self.expand(&row.path)
        };
        match result {
            Ok(()) => {
                self.clamp_selection();
                OverlayOutcome::Consumed
            }
            Err(e) => OverlayOutcome::Error(format!("{}: {e}", row.path.display())),
        }
    }

    /// `t`/`i`/`s`: these only ever open *files*. NERDTree's split mappings on a
    /// directory do nothing, and doing nothing loudly (a message) beats doing
    /// nothing silently.
    fn open_file(&mut self, target: OpenTarget) -> OverlayOutcome {
        let Some(row) = self.selected_row() else { return OverlayOutcome::Ignored };
        if row.is_dir {
            return OverlayOutcome::Message(format!(
                "{} is a directory — press o to expand it",
                row.name
            ));
        }
        OverlayOutcome::OpenPath { path: row.path, target }
    }

    fn expand_all(&mut self) -> OverlayOutcome {
        let Some(row) = self.selected_row() else { return OverlayOutcome::Ignored };
        if !row.is_dir {
            return OverlayOutcome::OpenPath { path: row.path, target: OpenTarget::Current };
        }
        // `expand_all` walks the subtree itself, reading as it goes, so the
        // panel's readability probe only covers the node the user pointed at.
        // A forbidden directory *deeper* in the subtree is silently skipped by
        // the engine's walker — the same gap the module docs describe, at a
        // depth the panel cannot see without re-walking the tree, which would be
        // reimplementing the engine.
        match self.expand(&row.path).and_then(|()| self.tree.expand_all(&row.path)) {
            Ok(()) => OverlayOutcome::Consumed,
            Err(e) => OverlayOutcome::Error(format!("{}: {e}", row.path.display())),
        }
    }

    /// `x`: collapse the parent of the selected node and move the cursor onto it
    /// — NERDTree's "close the node you are inside".
    fn close_parent(&mut self) -> OverlayOutcome {
        let rows = self.rows();
        let Some(current) = rows.get(self.selected) else { return OverlayOutcome::Ignored };
        let depth = current.depth();

        // The parent is the nearest row *above* the cursor with a smaller depth.
        // Derived from the flattened rows the engine already produced rather than
        // from the path (`path.parent()` would find the right *directory*, but not
        // its *row*, and the cursor has to land on the row).
        let Some(parent_idx) = rows[..self.selected].iter().rposition(|r| r.depth() < depth) else {
            // At the root already: `x` on a top-level node has nothing to close.
            return OverlayOutcome::Ignored;
        };
        let Some(parent) = rows[parent_idx].as_tree() else { return OverlayOutcome::Ignored };

        self.tree.collapse(&parent.path);
        self.selected = parent_idx;
        OverlayOutcome::Consumed
    }

    fn refresh(&mut self) -> OverlayOutcome {
        // Refresh the root, not the selection: NERDTree's `R` is "refresh the
        // root", and a stale cache anywhere below it is exactly what the user is
        // trying to clear. Forget every recorded permission failure at the same
        // time — a `chmod` is one of the things `R` exists to notice.
        self.unreadable.clear();
        let root = self.tree.root_path().to_path_buf();
        match self.tree.refresh(&root) {
            Ok(()) => {
                self.clamp_selection();
                OverlayOutcome::Consumed
            }
            Err(e) => OverlayOutcome::Error(format!("{}: {e}", root.display())),
        }
    }

    fn navigate_up(&mut self) -> OverlayOutcome {
        match self.tree.navigate_up() {
            Ok(true) => {
                self.unreadable.clear();
                self.selected = 0;
                self.scroll_top = 0;
                OverlayOutcome::Consumed
            }
            // Already at `/`. Say so rather than appearing to ignore the key.
            Ok(false) => OverlayOutcome::Message("already at the filesystem root".to_string()),
            Err(e) => OverlayOutcome::Error(e.to_string()),
        }
    }

    fn set_root(&mut self) -> OverlayOutcome {
        let Some(row) = self.selected_row() else { return OverlayOutcome::Ignored };
        if !row.is_dir {
            return OverlayOutcome::Message(format!("{} is not a directory", row.name));
        }
        match self.tree.set_root(&row.path) {
            Ok(()) => {
                self.unreadable.clear();
                self.selected = 0;
                self.scroll_top = 0;
                OverlayOutcome::Consumed
            }
            Err(e) => OverlayOutcome::Error(format!("{}: {e}", row.path.display())),
        }
    }

    /// Scrolls just far enough to keep the selected row on screen. Called from
    /// [`Self::render`], because `height` is not known before then.
    fn scroll_into_view(&mut self, height: usize) {
        if height == 0 {
            return;
        }
        if self.selected < self.scroll_top {
            self.scroll_top = self.selected;
        } else if self.selected >= self.scroll_top + height {
            self.scroll_top = self.selected + 1 - height;
        }
    }

    /// Draws the sidebar, returning where the terminal cursor should sit: on the
    /// selected row when focused, and nowhere (`None`) when it is not — a
    /// blinking cursor in a panel that does not have focus is a lie about where
    /// your keystrokes are going.
    pub fn render(
        &mut self,
        frame: &mut Frame,
        rect: Rect,
        theme: &Theme,
        icons: IconSet,
        focused: bool,
    ) -> Option<(u16, u16)> {
        if rect.width == 0 || rect.height == 0 {
            return None;
        }
        self.scroll_into_view(rect.height as usize);

        let cursor = (!self.help && focused).then(|| {
            let row = (self.selected.saturating_sub(self.scroll_top)) as u16;
            (rect.x, rect.y + row.min(rect.height - 1))
        });

        frame.render_widget(
            FileTreeView { panel: self, theme, icons, focused },
            rect,
        );
        cursor
    }
}

/// The sidebar as a ratatui widget. Borrows the panel rather than owning
/// anything, matching [`crate::ui::textarea::TextArea`]: a view is rebuilt every
/// frame from state that lives elsewhere.
struct FileTreeView<'a> {
    panel: &'a FileTreePanel,
    theme: &'a Theme,
    icons: IconSet,
    focused: bool,
}

impl FileTreeView<'_> {
    /// The divider drawn down the sidebar's right edge, separating it from the
    /// buffer. A Nerd Font is not needed for `│`, but a 1978 serial console does
    /// need `|` — so it degrades with everything else (see [`IconSet`]).
    fn divider(&self) -> &'static str {
        match self.icons {
            IconSet::Ascii => "|",
            IconSet::Nerd | IconSet::Unicode => "│",
        }
    }

    fn render_help(&self, area: Rect, buf: &mut Buffer, style: Style) {
        // Two columns would not fit in 30 cells; one key + one verb does.
        const HELP: &[(&str, &str)] = &[
            ("o/CR", "open / toggle"),
            ("t", "open in new tab"),
            ("i", "open in hsplit"),
            ("s", "open in vsplit"),
            ("O", "expand all below"),
            ("x", "close parent"),
            ("X", "close all"),
            ("R", "refresh"),
            ("I", "toggle hidden"),
            ("u/U/P", "go up a level"),
            ("C", "set as root"),
            ("j/k", "move"),
            ("?", "this help"),
            ("q/Esc", "close tree"),
        ];
        let title = Style::default().fg(self.theme.yellow_bright).bg(self.theme.bg1).add_modifier(Modifier::BOLD);
        buf.set_stringn(area.x, area.y, " keymaps", area.width as usize, title);
        for (i, (key, what)) in HELP.iter().enumerate() {
            let y = area.y + 1 + i as u16;
            if y >= area.y + area.height {
                break;
            }
            buf.set_stringn(area.x, y, format!(" {key:<6}{what}"), area.width as usize, style);
        }
    }
}

impl Widget for FileTreeView<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 {
            return;
        }

        // The sidebar sits one step lighter than the buffer background, which is
        // how neo-tree/NvimTree read as "chrome" rather than "another file".
        let base = Style::default().fg(self.theme.fg).bg(self.theme.bg1);
        buf.set_style(area, base);

        // Reserve the last column for the divider.
        let text_width = area.width.saturating_sub(1);
        let divider_style = Style::default().fg(self.theme.bg3).bg(self.theme.bg1);
        for row in 0..area.height {
            buf.set_stringn(area.x + text_width, area.y + row, self.divider(), 1, divider_style);
        }
        let text_area = Rect { width: text_width, ..area };
        if text_width == 0 {
            return;
        }

        if self.panel.help {
            self.render_help(text_area, buf, base);
            return;
        }

        let dir_style = Style::default().fg(self.theme.blue_bright).bg(self.theme.bg1).add_modifier(Modifier::BOLD);
        let error_style = Style::default().fg(self.theme.red_bright).bg(self.theme.bg1);
        // The selection is highlighted more strongly when the panel has focus:
        // that difference *is* the focus indicator, and it is the only way to see
        // at a glance whether `j` will move the tree or the text.
        let selected_bg = if self.focused { self.theme.bg3 } else { self.theme.bg2 };

        let rows = self.panel.rows();
        for (screen_row, idx) in
            (self.panel.scroll_top..rows.len()).enumerate().take(area.height as usize)
        {
            let y = text_area.y + screen_row as u16;
            let row = &rows[idx];
            let selected = idx == self.panel.selected;

            let (text, mut style) = match row {
                Row::Tree(tree_row) => {
                    let icon = if tree_row.is_dir {
                        self.icons.dir_icon(tree_row.is_expanded)
                    } else {
                        self.icons.file_icon(&tree_row.path)
                    };
                    let indent = "  ".repeat(tree_row.depth);
                    let style = if tree_row.is_dir { dir_style } else { base };
                    (format!("{indent}{icon} {}", tree_row.name), style)
                }
                Row::Error { depth, message } => {
                    let indent = "  ".repeat(*depth);
                    (format!("{indent}! {message}"), error_style)
                }
            };

            if selected {
                style = style.bg(selected_bg);
                // Paint the whole row, not just the text, so the highlight runs
                // to the sidebar's edge instead of stopping at the file name.
                buf.set_style(Rect { y, height: 1, ..text_area }, Style::default().bg(selected_bg));
            }
            buf.set_stringn(text_area.x, y, &text, text_width as usize, style);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};

    /// A tree over: root/{src/{main.rs}, README.md, .hidden}
    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(dir.path().join("README.md"), "# hi\n").unwrap();
        std::fs::write(dir.path().join(".hidden"), "").unwrap();
        dir
    }

    fn panel(root: &Path) -> FileTreePanel {
        FileTreePanel::open(root, ' ').unwrap()
    }

    fn press(c: char) -> KeyPress {
        KeyPress::plain(Key::Char(c))
    }

    fn names(panel: &FileTreePanel) -> Vec<String> {
        panel
            .rows()
            .iter()
            .map(|r| match r {
                Row::Tree(t) => t.name.clone(),
                Row::Error { message, .. } => format!("!{message}"),
            })
            .collect()
    }

    /// Renders the panel and returns its text rows as trimmed strings — the same
    /// thing a user would read off the screen. The last column (the divider) is
    /// excluded; [`divider_is_drawn_down_the_right_edge`] covers that separately.
    fn painted(panel: &mut FileTreePanel, icons: IconSet, focused: bool) -> Vec<String> {
        let backend = TestBackend::new(FileTreePanel::WIDTH, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        let theme = Theme::gruvbox_dark();
        terminal
            .draw(|frame| {
                panel.render(frame, frame.area(), &theme, icons, focused);
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        (0..12)
            .map(|y| {
                (0..FileTreePanel::WIDTH - 1)
                    .map(|x| buf.cell((x, y)).unwrap().symbol().to_string())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect()
    }

    #[test]
    fn divider_is_drawn_down_the_right_edge_and_degrades_to_ascii() {
        let dir = fixture();
        let mut p = panel(dir.path());
        let theme = Theme::gruvbox_dark();
        let mut edge = |icons: IconSet| {
            let mut terminal = Terminal::new(TestBackend::new(FileTreePanel::WIDTH, 4)).unwrap();
            terminal
                .draw(|frame| {
                    p.render(frame, frame.area(), &theme, icons, true);
                })
                .unwrap();
            let buf = terminal.backend().buffer().clone();
            (0..4)
                .map(|y| buf.cell((FileTreePanel::WIDTH - 1, y)).unwrap().symbol().to_string())
                .collect::<String>()
        };
        assert_eq!(edge(IconSet::Ascii), "||||");
        assert_eq!(edge(IconSet::Unicode), "││││");
    }

    #[test]
    fn the_root_and_its_children_render_with_ascii_icons_and_depth_indentation() {
        let dir = fixture();
        let mut p = panel(dir.path());
        let screen = painted(&mut p, IconSet::Ascii, true);

        // The root is expanded (`[-]`), directories sort before files, and the
        // ASCII tier is asserted explicitly because it is the one that has to
        // work on every terminal in existence.
        assert!(screen[0].starts_with("[-] "), "row 0 = {:?}", screen[0]);
        assert_eq!(screen[1], "  [+] src");
        assert_eq!(screen[2], "  [md] README.md");
        // Dotfiles hidden by default.
        assert!(!screen.iter().any(|r| r.contains(".hidden")));
    }

    #[test]
    fn expanding_a_directory_indents_its_children_one_level_further() {
        let dir = fixture();
        let mut p = panel(dir.path());
        p.handle_key(press('j')); // -> src
        assert_eq!(p.handle_key(press('o')), OverlayOutcome::Consumed);

        let screen = painted(&mut p, IconSet::Ascii, true);
        assert_eq!(screen[1], "  [-] src");
        assert_eq!(screen[2], "    [rs] main.rs");
    }

    #[test]
    fn o_on_a_file_asks_the_app_to_open_it() {
        let dir = fixture();
        let mut p = panel(dir.path());
        p.handle_key(press('j'));
        p.handle_key(press('j')); // -> README.md
        assert_eq!(
            p.handle_key(press('o')),
            OverlayOutcome::OpenPath {
                path: dir.path().join("README.md"),
                target: OpenTarget::Current
            }
        );
    }

    #[test]
    fn the_split_and_tab_keys_carry_their_target() {
        let dir = fixture();
        for (key, target) in [
            ('i', OpenTarget::HorizontalSplit),
            ('s', OpenTarget::VerticalSplit),
            ('t', OpenTarget::Tab),
        ] {
            let mut p = panel(dir.path());
            p.handle_key(press('j'));
            p.handle_key(press('j')); // -> README.md
            assert_eq!(
                p.handle_key(press(key)),
                OverlayOutcome::OpenPath { path: dir.path().join("README.md"), target },
                "{key} should open in {target:?}"
            );
        }
    }

    #[test]
    fn the_split_keys_on_a_directory_say_so_rather_than_doing_nothing() {
        let dir = fixture();
        let mut p = panel(dir.path());
        p.handle_key(press('j')); // -> src
        assert!(matches!(p.handle_key(press('s')), OverlayOutcome::Message(_)));
    }

    #[test]
    fn capital_o_expands_the_whole_subtree() {
        let dir = fixture();
        let mut p = panel(dir.path());
        p.handle_key(press('O')); // on the root
        assert!(names(&p).contains(&"main.rs".to_string()));
    }

    #[test]
    fn x_closes_the_parent_and_puts_the_cursor_on_it() {
        let dir = fixture();
        let mut p = panel(dir.path());
        p.handle_key(press('j')); // src
        p.handle_key(press('o')); // expand
        p.handle_key(press('j')); // main.rs
        assert_eq!(p.selected_row().unwrap().name, "main.rs");

        assert_eq!(p.handle_key(press('x')), OverlayOutcome::Consumed);
        assert_eq!(p.selected_row().unwrap().name, "src");
        assert!(!p.selected_row().unwrap().is_expanded);
    }

    #[test]
    fn capital_x_collapses_everything_and_resets_the_cursor() {
        let dir = fixture();
        let mut p = panel(dir.path());
        p.handle_key(press('j'));
        p.handle_key(press('o'));
        p.handle_key(press('j'));
        p.handle_key(press('X'));
        assert_eq!(p.rows().len(), 1, "only the (collapsed) root should remain");
        assert_eq!(p.selected, 0);
    }

    #[test]
    fn capital_i_toggles_dotfiles() {
        let dir = fixture();
        let mut p = panel(dir.path());
        assert!(!names(&p).contains(&".hidden".to_string()));
        p.handle_key(press('I'));
        assert!(names(&p).contains(&".hidden".to_string()));
        p.handle_key(press('I'));
        assert!(!names(&p).contains(&".hidden".to_string()));
    }

    #[test]
    fn capital_r_picks_up_a_file_created_outside_the_editor() {
        let dir = fixture();
        let mut p = panel(dir.path());
        assert!(!names(&p).contains(&"new.txt".to_string()));
        std::fs::write(dir.path().join("new.txt"), "").unwrap();
        assert_eq!(p.handle_key(press('R')), OverlayOutcome::Consumed);
        assert!(names(&p).contains(&"new.txt".to_string()));
    }

    #[test]
    fn u_and_its_synonyms_reroot_at_the_parent() {
        let dir = fixture();
        for key in ['u', 'U', 'P'] {
            let mut p = panel(&dir.path().join("src"));
            assert_eq!(p.root(), dir.path().join("src"));
            assert_eq!(p.handle_key(press(key)), OverlayOutcome::Consumed);
            assert_eq!(p.root(), dir.path(), "{key} should have gone up a level");
        }
    }

    #[test]
    fn c_sets_the_selected_directory_as_the_new_root() {
        let dir = fixture();
        let mut p = panel(dir.path());
        p.handle_key(press('j')); // -> src
        assert_eq!(p.handle_key(press('C')), OverlayOutcome::Consumed);
        assert_eq!(p.root(), dir.path().join("src"));
        assert_eq!(p.selected, 0);
    }

    #[test]
    fn c_on_a_file_refuses_rather_than_rerooting_at_a_file() {
        let dir = fixture();
        let mut p = panel(dir.path());
        p.handle_key(press('j'));
        p.handle_key(press('j')); // -> README.md
        assert!(matches!(p.handle_key(press('C')), OverlayOutcome::Message(_)));
        assert_eq!(p.root(), dir.path());
    }

    #[test]
    fn q_and_escape_close_the_tree() {
        let dir = fixture();
        let mut p = panel(dir.path());
        assert_eq!(p.handle_key(press('q')), OverlayOutcome::Close);
        assert_eq!(p.handle_key(KeyPress::plain(Key::Escape)), OverlayOutcome::Close);
    }

    #[test]
    fn leader_e_closes_the_tree_from_inside_it() {
        let dir = fixture();
        let mut p = panel(dir.path());
        // Space alone is a pending prefix, not a command.
        assert_eq!(p.handle_key(press(' ')), OverlayOutcome::Ignored);
        assert_eq!(p.handle_key(press('e')), OverlayOutcome::Close);
    }

    #[test]
    fn a_leader_that_is_not_followed_by_e_does_not_swallow_the_next_key() {
        let dir = fixture();
        let mut p = panel(dir.path());
        p.handle_key(press(' '));
        // `<leader>j` is not a mapping, so the `j` still moves the cursor.
        assert_eq!(p.handle_key(press('j')), OverlayOutcome::Consumed);
        assert_eq!(p.selected, 1);
    }

    #[test]
    fn question_mark_shows_the_keymap_help_and_the_next_key_dismisses_it() {
        let dir = fixture();
        let mut p = panel(dir.path());
        p.handle_key(press('?'));
        let screen = painted(&mut p, IconSet::Ascii, true);
        assert!(screen[0].contains("keymaps"), "{screen:?}");
        assert!(screen.iter().any(|r| r.contains("open / toggle")));

        // Dismissing must not also *run* the key that dismissed it.
        p.handle_key(press('X'));
        assert!(!p.help);
        assert!(p.rows().len() > 1, "X dismissed the help; it must not also collapse the tree");
    }

    #[test]
    fn j_and_k_stop_at_the_ends_without_asking_for_a_redraw() {
        let dir = fixture();
        let mut p = panel(dir.path());
        assert_eq!(p.handle_key(press('k')), OverlayOutcome::Ignored); // already at the top
        while p.handle_key(press('j')) == OverlayOutcome::Consumed {}
        assert_eq!(p.selected, p.rows().len() - 1);
        assert_eq!(p.handle_key(press('j')), OverlayOutcome::Ignored); // and at the bottom
    }

    #[test]
    fn the_arrow_keys_are_synonyms_for_j_and_k() {
        let dir = fixture();
        let mut p = panel(dir.path());
        p.handle_key(KeyPress::plain(Key::Down));
        assert_eq!(p.selected, 1);
        p.handle_key(KeyPress::plain(Key::Up));
        assert_eq!(p.selected, 0);
    }

    #[test]
    fn an_unreadable_directory_renders_an_honest_error_row() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let forbidden = dir.path().join("forbidden");
        std::fs::create_dir(&forbidden).unwrap();
        std::fs::write(forbidden.join("secret.txt"), "").unwrap();
        std::fs::set_permissions(&forbidden, std::fs::Permissions::from_mode(0o000)).unwrap();

        // root can read anything, so this test proves nothing when run as root.
        if std::fs::read_dir(&forbidden).is_ok() {
            std::fs::set_permissions(&forbidden, std::fs::Permissions::from_mode(0o755)).unwrap();
            return;
        }

        let mut p = panel(dir.path());
        p.handle_key(press('j')); // -> forbidden
        assert_eq!(p.handle_key(press('o')), OverlayOutcome::Consumed);

        let screen = painted(&mut p, IconSet::Ascii, true);
        // The folder is drawn expanded, with the reason it is empty underneath —
        // not as a silently empty directory, and not as a panic.
        assert_eq!(screen[1], "  [-] forbidden");
        assert!(
            screen[2].contains("denied") || screen[2].starts_with("    !"),
            "expected an error row, got {:?}",
            screen[2]
        );

        // Leave the tempdir removable.
        std::fs::set_permissions(&forbidden, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[test]
    fn the_selection_highlight_is_stronger_when_the_panel_has_focus() {
        let dir = fixture();
        let mut p = panel(dir.path());
        let theme = Theme::gruvbox_dark();

        let mut bg_of_row_0 = |focused: bool| {
            let backend = TestBackend::new(FileTreePanel::WIDTH, 6);
            let mut terminal = Terminal::new(backend).unwrap();
            terminal
                .draw(|frame| {
                    p.render(frame, frame.area(), &theme, IconSet::Ascii, focused);
                })
                .unwrap();
            terminal.backend().buffer().cell((0, 0)).unwrap().style().bg
        };
        assert_eq!(bg_of_row_0(true), Some(theme.bg3));
        assert_eq!(bg_of_row_0(false), Some(theme.bg2));
    }

    #[test]
    fn a_long_tree_scrolls_to_keep_the_selection_visible() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..40 {
            std::fs::write(dir.path().join(format!("f{i:02}.txt")), "").unwrap();
        }
        let mut p = panel(dir.path());
        for _ in 0..30 {
            p.handle_key(press('j'));
        }
        let screen = painted(&mut p, IconSet::Ascii, true); // 12 rows tall
        assert!(screen.iter().any(|r| r.contains("f29.txt")), "{screen:?}");
        assert!(!screen.iter().any(|r| r.contains("f00.txt")), "should have scrolled past the top");
    }

    #[test]
    fn rendering_survives_a_terminal_squeezed_to_nothing() {
        let dir = fixture();
        let mut p = panel(dir.path());
        let theme = Theme::gruvbox_dark();
        for (w, h) in [(0, 0), (1, 1), (2, 30), (200, 2)] {
            let mut terminal = Terminal::new(TestBackend::new(w.max(1), h.max(1))).unwrap();
            terminal
                .draw(|frame| {
                    let area = Rect { width: w, height: h, ..frame.area() };
                    p.render(frame, area, &theme, IconSet::Ascii, true);
                })
                .unwrap();
        }
    }
}
