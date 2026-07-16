//! The `:help` manual — kvim's own built-in help, written in Singlish.
//!
//! # Why the help lives here as structured data
//!
//! The manual is not one big string sitting in the middle of `execute_ex`. It
//! is a table of [`HelpTopic`]s — one per section — and [`render`] walks that
//! table once to build both the buffer text *and* the line index that
//! `:help <topic>` jumps to. Keeping the two together in one pass is the whole
//! point: the jump table can never drift out of sync with the text, because
//! neither is written by hand. Add a topic to [`TOPICS`], and it shows up in
//! the index, in the manual body, and as a `:help <topic>` target all at once.
//!
//! # Singlish, but the keys stay exact
//!
//! Per `CLAUDE.md` this session, the *prose* is Colloquial Singapore English —
//! that is the explaining part, meant to read like a friend walking you
//! through kvim, lah. The *key names* (`<C-w>`, `<leader>e`, `:qa`), command
//! names and mode names are NOT Singlish and NOT softened: they must be the
//! exact strings you actually press, or the manual is worse than useless. So
//! `<leader>` is always written `<leader>` (which is Space — the manual says
//! so), pickers are `\ff`/`\fb`/`\fh` to the character, and so on. Singlish is
//! never an excuse to be vague about a keybinding.
//!
//! # Where the facts come from
//!
//! Every keymap quoted here is the maintainer's real config in
//! [`crate::config`] (`Config::keymaps`, `leader`), and every `:` command is
//! one [`super::ex::parse`] actually recognises. If you change a binding
//! there, change the matching line here — an out-of-date help entry is a bug,
//! same as any other.

/// One section of the manual: a heading plus its body lines, addressable by a
/// canonical `id` (and any `aliases`) so `:help <topic>` can land on it.
///
/// `body` lines are stored without a trailing newline; [`render`] joins them.
/// Blank strings in `body` are intentional paragraph breaks.
pub struct HelpTopic {
    /// The canonical topic name, e.g. `"windows"`. This is what `:help windows`
    /// matches on, and what shows up in the index.
    pub id: &'static str,
    /// Extra names that also resolve to this topic, e.g. `"window"`, `"splits"`
    /// for `windows`. Matched case-insensitively, same as `id`.
    pub aliases: &'static [&'static str],
    /// The Singlish heading printed above the section body.
    pub title: &'static str,
    /// The section body, one entry per line.
    pub body: &'static [&'static str],
}

/// The whole manual, section by section. Order here is the order on screen.
///
/// This is the single source of truth for `:help`. Nothing else hard-codes the
/// topic list — the index at the top of the manual and the `:help <topic>`
/// jump table are both derived from this by [`render`].
pub const TOPICS: &[HelpTopic] = &[
    HelpTopic {
        id: "about",
        aliases: &["intro", "start", "index"],
        title: "kvim — the KOPITIAM editor, quick-start ah",
        body: &[
            "Welcome to kvim lah. This one is a modal editor — mostly vim keys,",
            "so if you sabo yourself in insert mode, just tekan <Esc> and you back",
            "in normal mode. Below got all the sections. To jump straight to one,",
            "type `:help <topic>`, e.g. `:help windows` or `:help lsp`.",
            "",
            "Topics you can `:help`:",
            "  about        this page, the overview",
            "  modes        normal / insert / visual / command, how to swap",
            "  motions      how to move the cursor around",
            "  operators    d y c and friends, operator + motion",
            "  textobjects  iw ap i\" — act on a whole thing",
            "  counts       do something N times",
            "  registers    vim's copy-paste clipboards",
            "  macros       record and replay keystrokes with q",
            "  dotrepeat    the . command, redo your last change",
            "  leader       the <leader> (Space) key, where the goodies are",
            "  filetree     the file explorer sidebar",
            "  hop          jump anywhere on screen with f",
            "  pickers      fuzzy-find files / buffers / help",
            "  harpoon      pin your hot files, jump back fast",
            "  align        line up text on a delimiter",
            "  lsp          go-to-definition, hover, rename",
            "  completion   the autocomplete menu",
            "  windows      splits and moving between them",
            "  tmux         hop out into tmux panes",
            "  quit         saving and quitting, all the :q flavours",
            "  shell        run shell commands, filter text through them",
            "",
            "Cannot remember a keymap? Press <leader> (Space) and wait — the",
            "which-key popup will show you what's available. Steady lah.",
        ],
    },
    HelpTopic {
        id: "modes",
        aliases: &["mode"],
        title: "Modes — kvim always in one of these",
        body: &[
            "kvim is modal, so the same key does different things depending on",
            "which mode you inside. Don't blur, the mode is shown bottom-left.",
            "",
            "  normal        the home base. Movement + commands. Press <Esc> from",
            "                anywhere to come back here.",
            "  insert        actually typing text. Enter with i a o (and I A O,",
            "                and c... operators). Leave with <Esc>.",
            "  visual        select text char-by-char. Enter with v.",
            "  visual-line   select whole lines. Enter with V (capital).",
            "  visual-block  select a rectangle/column. Enter with <C-v>.",
            "  command       the `:` command line (ex commands). Enter with `:`.",
            "                Type your command, press <Enter> to run, <Esc> to bail.",
            "",
            "Rule of thumb: when you blur already, just tekan <Esc><Esc> and you",
            "confirm-plus-chop back in normal mode. From there everything works.",
        ],
    },
    HelpTopic {
        id: "motions",
        aliases: &["motion", "movement", "move"],
        title: "Motions — moving the cursor",
        body: &[
            "The standard vim motions all work here, so this part is quite the",
            "same as normal vim lah:",
            "",
            "  h j k l       left / down / up / right",
            "  w b e         word forward / back / end-of-word",
            "  0 ^ $         line start / first non-blank / line end",
            "  gg G          top of file / bottom of file",
            "  { }           paragraph back / forward",
            "  f<x> t<x>     jump to / just-before next <x> on the line (; , repeat)",
            "  % matching bracket",
            "",
            "Want to fly across the screen? Don't need to h-h-h-h damn long —",
            "press f (that's kvim's hop, see `:help hop`) and jump straight there.",
        ],
    },
    HelpTopic {
        id: "operators",
        aliases: &["operator", "edit", "editing"],
        title: "Operators — operator + motion = edit",
        body: &[
            "An operator waits for a motion (or a text object) and acts on",
            "everything in between. This grammar is standard vim, so:",
            "",
            "  d             delete      (dw = delete word, dd = delete line)",
            "  y             yank/copy   (yy = copy line)",
            "  c             change      (cw = change word — delete then insert)",
            "  >  <          indent / un-indent",
            "  =             auto-format the range",
            "  gu gU g~      lowercase / uppercase / swap case",
            "",
            "Mix and match with any motion or text object: `di\"` delete inside",
            "quotes, `ca(` change around parens, `>ap` indent a paragraph. Once",
            "you got the grammar, everything just combines — power sia.",
        ],
    },
    HelpTopic {
        id: "textobjects",
        aliases: &["textobject", "text-objects", "objects"],
        title: "Text objects — act on a whole thing",
        body: &[
            "A text object is 'the whole word', 'the whole paragraph', 'inside",
            "these brackets' — you pair it with an operator. Standard vim:",
            "",
            "  iw aw         inner word / a word (aw grabs the trailing space)",
            "  is as         inner sentence / a sentence",
            "  ip ap         inner paragraph / a paragraph",
            "  i\" a\"  i' a'   inside / around quotes",
            "  i( a( i) a)   inside / around parens (also i{ i[ i< etc.)",
            "  it at         inside / around an HTML/XML tag",
            "",
            "So `ci\"` = change inside the quotes, `dap` = delete a whole",
            "paragraph, `yi(` = yank inside the parens. Very shiok once you used to it.",
        ],
    },
    HelpTopic {
        id: "counts",
        aliases: &["count", "repeat-count"],
        title: "Counts — do it N times",
        body: &[
            "Put a number in front and the thing happens that many times. Standard",
            "vim, no surprise:",
            "",
            "  3j            go down 3 lines",
            "  5dd           delete 5 lines",
            "  2dw           delete 2 words",
            "  d3w           also delete 3 words (count can go on the motion)",
            "",
            "Faster than pressing the same key until your finger pain.",
        ],
    },
    HelpTopic {
        id: "registers",
        aliases: &["register", "clipboard", "yank"],
        title: "Registers — vim's many clipboards",
        body: &[
            "Registers are named clipboards. Prefix with `\"` and a letter to pick",
            "one. Standard vim behaviour:",
            "",
            "  \"ayy          yank this line into register a",
            "  \"ap           paste from register a",
            "  \"0            the yank register (last thing you copied, not deleted)",
            "  \"\"            the unnamed register (the default, what plain p uses)",
            "",
            "Deleted text goes into the numbered registers \"1..\"9, so even if you",
            "delete-then-yank something else, your delete not lost yet — can still",
            "dig it back. Damn useful when you accidentally delete wrong thing.",
        ],
    },
    HelpTopic {
        id: "macros",
        aliases: &["macro", "record"],
        title: "Macros — record and replay",
        body: &[
            "Macros let you record a bunch of keystrokes once and replay them. This",
            "is standard vim, works the same here:",
            "",
            "  q<a>          start recording into register a",
            "  q             stop recording",
            "  @a            replay macro a",
            "  @@            replay the last macro again",
            "  5@a           replay macro a five times",
            "",
            "Best for repetitive donkey work — record the fix once, then just spam",
            "@a down the file. Sit back and relax lah.",
        ],
    },
    HelpTopic {
        id: "dotrepeat",
        aliases: &["dot", "dot-repeat", "."],
        title: "Dot-repeat — the . command",
        body: &[
            "The `.` command repeats your last change. Standard vim, one of the",
            "best things about it:",
            "",
            "  .             redo the last change (edit, not movement)",
            "",
            "So `ciwfoo<Esc>` to change a word to 'foo', then move to the next word",
            "and just press `.` — same change again. No need retype. Confirm plus chop.",
        ],
    },
    HelpTopic {
        id: "leader",
        aliases: &["mapleader", "space", "whichkey", "which-key"],
        title: "The <leader> key — where the goodies hide",
        body: &[
            "In kvim the <leader> key is Space. Whenever a keymap says <leader>,",
            "that means: press Space, then the rest. The maintainer's leader maps:",
            "",
            "  <leader>e     toggle the file explorer sidebar (see `:help filetree`)",
            "  <leader>gd    LSP go-to-definition (see `:help lsp`)",
            "  <leader>gr    LSP find references",
            "  <leader>rn    LSP rename the symbol",
            "  <leader>b     Harpoon: mark this file (see `:help harpoon`)",
            "  <leader><Esc> Harpoon: toggle the quick menu",
            "  <leader>q     Harpoon: fuzzy-find your marks",
            "",
            "Cannot remember? Just press <leader> (Space) and hold horses — the",
            "which-key popup lists everything under it. No need memorise all.",
        ],
    },
    HelpTopic {
        id: "filetree",
        aliases: &["file-tree", "explorer", "neotree", "tree", "sidebar"],
        title: "File tree — the explorer sidebar",
        body: &[
            "kvim got a file-tree sidebar (the neo-tree replacement). Toggle it:",
            "",
            "  <leader>e     open / close the file explorer",
            "",
            "Inside the tree, move with j k like normal, <Enter> to open a file or",
            "expand a folder. To jump focus between the tree and your code window,",
            "use the bare <C-h> / <C-l> window moves (see `:help windows`).",
        ],
    },
    HelpTopic {
        id: "hop",
        aliases: &["jump", "leap", "f"],
        title: "Hop — jump anywhere on screen with f",
        body: &[
            "kvim rebinds f to hop (label jump). Heads up: this DELIBERATELY",
            "overrides vim's built-in f (find char on the line) — the maintainer",
            "wants the screen jump more, so:",
            "",
            "  f             hop: label every word on screen, then you type the",
            "                label letters and cursor teleport straight there.",
            "",
            "No need count how many w to press — see where you want, f, type the",
            "label, done. Damn fast once you catch the rhythm.",
        ],
    },
    HelpTopic {
        id: "pickers",
        aliases: &["picker", "telescope", "fuzzy", "find"],
        title: "Pickers — fuzzy-find things (telescope)",
        body: &[
            "The fuzzy finders (telescope replacement) are on the backslash `\\`",
            "prefix — NOT the leader. Note it's `\\`, not Space:",
            "",
            "  \\ff           find files in the project",
            "  \\fb           find open buffers",
            "  \\fh           find help tags",
            "",
            "Just start typing part of the name, the list narrow down live. Confirm",
            "with <Enter>. No need remember the exact filename — agar-agar also can.",
        ],
    },
    HelpTopic {
        id: "harpoon",
        aliases: &["marks", "pin", "bookmarks"],
        title: "Harpoon — pin your hot files",
        body: &[
            "Harpoon lets you pin the few files you keep jumping between, so you",
            "don't keep fuzzy-finding the same 4 files whole day. The maps:",
            "",
            "  <leader>b     mark (harpoon) the current file",
            "  <leader><Esc> toggle the harpoon quick menu",
            "  <leader>q     fuzzy-find your harpoon marks",
            "",
            "Workflow: open your key files, <leader>b each one, then use the quick",
            "menu to zip between them. Steady pom pi pi.",
        ],
    },
    HelpTopic {
        id: "align",
        aliases: &["easy-align", "easyalign", "ga"],
        title: "Align — line up text on a delimiter",
        body: &[
            "The align feature (vim-easy-align) lines up columns of text on a",
            "delimiter — very useful for tidying up assignments or tables:",
            "",
            "  ga            align on a delimiter (then type the delimiter, e.g. =)",
            "",
            "Select some lines in visual mode first (or give it a motion), press",
            "ga, then the char to align on. Everything snap into neat columns.",
            "Change from messy to shiok in one move.",
        ],
    },
    HelpTopic {
        id: "lsp",
        aliases: &["language-server", "diagnostics", "code"],
        title: "LSP — go-to-definition, hover, rename",
        body: &[
            "kvim talks to language servers (rust-analyzer and friends) for the",
            "smart code stuff. The maintainer's LSP maps:",
            "",
            "  <leader>gd    go to definition",
            "  <leader>gr    find references",
            "  <leader>rn    rename the symbol everywhere",
            "  K             hover docs (type / signature / docs under the cursor)",
            "",
            "K follows Neovim's built-in default (vim.lsp.buf.hover). For the",
            "autocomplete menu that the LSP feeds, see `:help completion`.",
        ],
    },
    HelpTopic {
        id: "completion",
        aliases: &["complete", "autocomplete", "cmp"],
        title: "Completion — the autocomplete menu",
        body: &[
            "As you type, kvim's completion (the blink.cmp replacement) offers",
            "candidates from the LSP, snippets, and words already in the buffer:",
            "",
            "  <C-Space>     force-open the completion menu right now",
            "  <Tab>         accept the highlighted candidate (also jumps snippet",
            "                tabstops when you're inside a snippet)",
            "  <C-n> <C-p>   move down / up the candidate list",
            "  <C-e>         dismiss the menu without accepting anything",
            "",
            "Menu will auto-pop as you type an identifier also. If you just want it",
            "now-now, <C-Space> summon it. Don't paiseh to use it.",
        ],
    },
    HelpTopic {
        id: "windows",
        aliases: &["window", "splits", "split", "panes"],
        title: "Windows — splits and moving between them",
        body: &[
            "Split your screen into windows and jump between them. The <C-w>",
            "prefix owns window commands, and the bare <C-h/j/k/l> move focus:",
            "",
            "  <C-w>s        split horizontally  (also :sp)",
            "  <C-w>v        split vertically    (also :vs)",
            "  <C-w>n        new split with an empty scratch buffer",
            "  <C-w>h/j/k/l  move focus to the split left/down/up/right",
            "  <C-w>o        keep only the current window  (also :only)",
            "  <C-w>c        close the current window       (also :close)",
            "",
            "  <C-h> <C-j> <C-k> <C-l>   (bare, no <C-w>) also move between splits",
            "                           — and hop into tmux at the edge, see",
            "                           `:help tmux`.",
            "",
            "For the whole :sp / :vs / :qa family of commands, see `:help quit`.",
        ],
    },
    HelpTopic {
        id: "tmux",
        aliases: &["tmux-navigator", "panes"],
        title: "Tmux — hop out into tmux panes",
        body: &[
            "If you run kvim inside tmux, the bare <C-h/j/k/l> window moves are",
            "seamless: when you move off the edge of kvim's own splits, kvim hands",
            "focus over to the neighbouring tmux pane automatically.",
            "",
            "  <C-h> <C-j> <C-k> <C-l>   move split OR cross into the tmux pane",
            "                           on that side, whichever is there.",
            "",
            "So navigating your editor splits and your tmux panes feels like one",
            "single thing. kvim detects tmux from $TMUX at startup — no config needed.",
        ],
    },
    HelpTopic {
        id: "quit",
        aliases: &["write", "save", "exit", "q", "wq"],
        title: "Quit & save — all the :q flavours",
        body: &[
            "The `:` command line (press `:` first) does the saving and quitting.",
            "kvim won't let you quit-and-lose unsaved work unless you force with `!`:",
            "",
            "  :w            write (save) this buffer",
            "  :w <file>     write to a specific file",
            "  :q            quit this window (blocks if unsaved — use :q! to force)",
            "  :q!           quit this window, throw away changes",
            "  :wq   :x      write, then quit",
            "  :qa           quit ALL windows / exit kvim (blocks if any unsaved)",
            "  :qa!          exit kvim, throw away everything",
            "  :wa           write all modified buffers",
            "  :wqa  :xa     write all, then exit kvim",
            "",
            "  :sp  :vs      split this window (see `:help windows`)",
            "",
            "Golden rule: no `!` means kvim protects your unsaved work; put `!`",
            "only when you confirm-plus-chop sure you want to lose the changes.",
        ],
    },
    HelpTopic {
        id: "shell",
        aliases: &["bang", "filter", "!", "system"],
        title: "Shell — run commands, filter text through them",
        body: &[
            "kvim can shell out just like vim, all through `sh -c`. Four forms,",
            "and the last one is the power tool:",
            "",
            "  :!<cmd>        run <cmd>, show its output in a scratch buffer.",
            "                 e.g. `:!ls -l`, `:!git status`.",
            "  :r !<cmd>      run <cmd>, read its output INTO the buffer, below",
            "                 the cursor line. e.g. `:r !date`, `:r !echo hi`.",
            "  :{range}!<cmd> FILTER the range: feed those lines to <cmd>'s stdin,",
            "                 replace them with its stdout. The classic one is",
            "                 `:%!sort` (sort the whole file) or `:'<,'>!column -t`.",
            "  !<motion><cmd> the filter OPERATOR. `!` takes a motion — `!ip` for",
            "                 the paragraph, `!5j` for five lines, `!G` to the",
            "                 bottom — then drops you at a `:{range}!` line for you",
            "                 to type the command. `!!` filters just this line.",
            "",
            "Commands run in the folder you started kvim from (like vim's :pwd),",
            "not the file's folder. If a filter command fails (non-zero exit),",
            "kvim leaves your buffer alone and just tells you — no sabo, your text",
            "stays safe.",
        ],
    },
];

/// The manual as a buffer's worth of text, plus the line each section starts on.
///
/// Built in a single pass over [`TOPICS`] so the jump index and the text can
/// never disagree: `sections[i]` is `(topic_id, 0-based line of that section's
/// heading)`, which is exactly what `:help <topic>` needs to position the cursor.
pub struct RenderedHelp {
    /// The full manual text, ready to drop into a scratch buffer.
    pub text: String,
    /// `(canonical topic id, 0-based line of its heading)`, in manual order.
    pub sections: Vec<(&'static str, usize)>,
}

/// A visual divider line under each section heading — long enough to underline
/// the longest title we ship, and kept as one constant so every section is
/// underlined identically.
const RULE: &str = "========================================================================";

/// Renders the whole `:help` manual and its section line-index in one pass.
///
/// See [`RenderedHelp`] for why the text and the index are produced together
/// rather than separately.
pub fn render() -> RenderedHelp {
    let mut lines: Vec<String> = Vec::new();
    let mut sections: Vec<(&'static str, usize)> = Vec::new();

    lines.push("kvim :help — press `:help <topic>` to jump, e.g. `:help lsp`".to_string());
    lines.push(String::new());

    for topic in TOPICS {
        // The heading line is the jump target for this topic.
        sections.push((topic.id, lines.len()));
        lines.push(topic.title.to_string());
        lines.push(RULE.to_string());
        for &body_line in topic.body {
            lines.push(body_line.to_string());
        }
        // One blank line between sections for breathing room.
        lines.push(String::new());
    }

    RenderedHelp { text: lines.join("\n"), sections }
}

/// Resolves a `:help <topic>` argument to a canonical topic id, matching the
/// `id` or any alias case-insensitively.
///
/// Returns `None` when nothing matches, so the caller can fall back to opening
/// the manual at the top (the `about` overview) rather than erroring — a typo'd
/// `:help` topic should still show *some* help, same as real vim.
pub fn resolve(query: &str) -> Option<&'static str> {
    let q = query.trim().to_ascii_lowercase();
    if q.is_empty() {
        return None;
    }
    TOPICS
        .iter()
        .find(|t| t.id.eq_ignore_ascii_case(&q) || t.aliases.iter().any(|a| a.eq_ignore_ascii_case(&q)))
        .map(|t| t.id)
}

/// The 0-based line a topic's heading sits on within a [`RenderedHelp`], or
/// `None` if the topic id is not in the index.
pub fn section_line(sections: &[(&'static str, usize)], topic_id: &str) -> Option<usize> {
    sections.iter().find(|(id, _)| *id == topic_id).map(|(_, line)| *line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_topic_is_a_jump_target() {
        // The index must list exactly the topics, in order — no drift between
        // the rendered text and the jump table.
        let r = render();
        assert_eq!(r.sections.len(), TOPICS.len());
        for (topic, (id, _)) in TOPICS.iter().zip(&r.sections) {
            assert_eq!(topic.id, *id);
        }
    }

    #[test]
    fn section_line_points_at_the_heading() {
        // The recorded line for each topic must actually be that topic's title.
        let r = render();
        let lines: Vec<&str> = r.text.lines().collect();
        for topic in TOPICS {
            let line = section_line(&r.sections, topic.id).expect("topic in index");
            assert_eq!(lines[line], topic.title, "topic {:?} heading misplaced", topic.id);
        }
    }

    #[test]
    fn resolve_matches_id_and_aliases_case_insensitively() {
        assert_eq!(resolve("windows"), Some("windows"));
        assert_eq!(resolve("WINDOWS"), Some("windows"));
        // aliases resolve to the canonical id
        assert_eq!(resolve("splits"), Some("windows"));
        assert_eq!(resolve("neotree"), Some("filetree"));
        assert_eq!(resolve("mapleader"), Some("leader"));
        // unknown / empty fall through
        assert_eq!(resolve("nonsense-topic"), None);
        assert_eq!(resolve(""), None);
    }

    #[test]
    fn manual_carries_the_exact_key_names() {
        // Singlish prose is fine, but the actual keys must survive verbatim.
        let text = render().text;
        for needle in [
            "<leader>e", "<leader>gd", "<leader>gr", "<leader>rn", "<leader>b",
            "<leader><Esc>", "<leader>q", "\\ff", "\\fb", "\\fh", "K",
            "<C-Space>", "<Tab>", "<C-w>", "<C-h>", ":qa", ":wqa", "ga",
        ] {
            assert!(text.contains(needle), "manual must mention key {needle:?} exactly");
        }
    }

    #[test]
    fn manual_reads_as_singlish() {
        // A light guard that the prose kept its Singlish flavour — if someone
        // rewrites it into flat English, at least one of these should still be
        // here, or the session's house style was dropped.
        let text = render().text.to_lowercase();
        let flavour = ["lah", "sia", "shiok", "steady", "paiseh", "confirm plus chop", "sabo"];
        assert!(flavour.iter().any(|w| text.contains(w)), "help prose lost its Singlish");
    }
}
