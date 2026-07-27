//! The **Model Picker**: choose which local model the AI Chat talks to, from
//! inside the TUI itself (`Ctrl-P` while in chat).
//!
//! # Why this exists
//!
//! Before this, the chat's model was decided once at startup by
//! [`crate::adapter::select_adapter`], which reads `KOPITIAM_MODEL_GGUF` /
//! `KOPITIAM_MODEL` from the environment and otherwise looks **only** for
//! [`kopitiam_models::DEFAULT_MODEL_ID`]. Two bad consequences, both of which
//! this module exists to kill:
//!
//! 1. To try another model you must quit, export an env var, and start again.
//! 2. The Models view happily pull *any* catalog row, but chat only ever look
//!    for the default id — so pull the 1.7B and chat still say "no .gguf yet"
//!    while the file sit right there on disk. The picker plus
//!    [`crate::adapter::select_adapter_for`] closes that gap.
//!
//! # Thin client, thick logic
//!
//! `CLAUDE.md` says clients own no business logic, and there is a second reason
//! here: a TUI **cannot be verified headless** — no real terminal, no keystrokes,
//! no paint. So everything that *decides* anything lives in plain functions in
//! the bottom half of this file and is unit-tested:
//!
//! * [`build_choices`] — what is offered (catalog `.gguf` entries, hand-dropped
//!   `.gguf` under the store, the bring-your-own env path), which of them are on
//!   disk, and how each one would be loaded ([`LoadPlan`]).
//! * [`initial_selection`] — where the cursor starts.
//! * [`activate`] — whether Enter on a row can do anything, and the exact
//!   Singlish refusal when it cannot.
//! * [`load_plan`] / [`switch_note`] / [`short_status`] — turning a choice into a
//!   [`SelectedAdapter`] plus the one-liners the user reads about it.
//!
//! [`ModelPickerView`] on top of that is only keys, layout and paint.
//!
//! # What this module does NOT do
//!
//! It never downloads anything. A model not on disk is shown but cannot be
//! activated — the refusal points at Home → Models, which owns acquisition (see
//! [`super::models`]). One code path writes into the model store, not two.
//!
//! # Swapping while a reply is still streaming — safe, and here's why
//!
//! [`kopitiam_ai::LocalAdapter::stream`] hands its worker thread `Arc` clones of
//! the model and tokenizer, so the in-flight turn keeps its weights alive even
//! after the old [`SelectedAdapter`] is dropped. Concretely: the reply being
//! typed out right now finishes on the OLD model, and the NEXT turn uses the new
//! one. That is the contract this picker relies on — if `LocalAdapter::stream`
//! ever starts borrowing from `&self` instead of cloning `Arc`s, swapping
//! mid-stream becomes a use-after-free hazard and this module must then refuse
//! to switch while [`super::chat::ChatView::is_streaming`] is true.

use std::path::{Path, PathBuf};

use kopitiam_ai::{EchoAdapter, LocalAdapter};
use kopitiam_models::{Catalog, ModelSpec, ModelStore};
use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, List, ListItem, ListState, Paragraph, Wrap},
};

use crate::adapter::{FallbackReason, SelectedAdapter, select_adapter_for};

use super::models::{family_label, human_bytes};
use super::theme::{CHILLI, DIM, GOLD, PALM, STEAM, TAN};
use super::{Route, Transition};

/// Env var holding a bring-your-own `.gguf` path. Same name
/// [`crate::adapter::select_adapter`] reads — surfaced as a row so the user can
/// see (and go back to) whatever the environment chose at startup.
const MODEL_PATH_ENV: &str = "KOPITIAM_MODEL_GGUF";

/// How deep to look for hand-dropped `.gguf` under the store root. The store
/// lays models out as `<root>/<id>/<filename>`, so depth 3 covers it with room
/// to spare; the cap exists so a store root pointed at somebody's whole home
/// directory cannot turn the picker into a full-disk walk.
const LOOSE_SCAN_DEPTH: usize = 3;

/// Where a row's `.gguf` came from. Affects labelling and, for the BYO case,
/// how the load is routed — see [`LoadPlan`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Origin {
    /// A model in the built-in [`Catalog`], at its store path.
    Catalog,
    /// A `.gguf` under the store root the catalog does not know about —
    /// somebody dropped it there by hand.
    Loose,
    /// The path in `KOPITIAM_MODEL_GGUF` (bring-your-own).
    Byo,
}

/// How a chosen row gets turned into a [`SelectedAdapter`].
///
/// Two routes, and picking the wrong one is a real bug, so the reasoning is
/// recorded here rather than inline at the call site:
///
/// * [`LoadPlan::ById`] goes through [`crate::adapter::select_adapter_for`],
///   which owns store-path resolution and the echo fallback. This is the normal
///   route for catalog models — it keeps one implementation of "where does model
///   `X` live", shared with the non-TUI `kopitiam ai chat`.
/// * [`LoadPlan::ByPath`] loads an exact file. Needed in two cases: a `.gguf`
///   with no catalog id at all (loose / BYO rows), **and** every row when
///   `KOPITIAM_MODEL_GGUF` is set — because `select_adapter_for` deliberately
///   lets that env path win over the id it is given, so routing by id there
///   would silently load the env's model instead of the one the user just
///   clicked. That would be the worst kind of bug: the UI says one model, the
///   answers come from another.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LoadPlan {
    ById(String),
    ByPath(PathBuf),
}

/// One row in the picker: a model the chat could switch to.
///
/// `present` means **the file exists at `path`** — nothing more. It is not a
/// checksum check and not a promise the weights will load (a truncated or
/// wrong-architecture `.gguf` is `present: true` and still fails on load). This
/// mirrors `crate::adapter`'s selection gate deliberately, see AID-0029: the
/// real test of runnability is `LocalAdapter::load` succeeding, which we only
/// find out when the user picks the row — and when it fails, [`switch_note`]
/// says so in the user's own words instead of pretending nothing happened.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ModelChoice {
    /// Catalog id, or the file name for a loose / BYO `.gguf`.
    pub id: String,
    /// Human label shown in the list.
    pub label: String,
    /// Right-hand detail column (family · size · licence, or the path).
    pub detail: String,
    /// The `.gguf` this row refers to. Empty when the store root could not be
    /// resolved at all — such a row can never be activated.
    pub path: PathBuf,
    pub origin: Origin,
    /// The file is on disk right now. See the type-level note above.
    pub present: bool,
    /// This is the model the chat is using at this moment.
    pub active: bool,
    /// This is the id the environment/default would pick on a fresh start.
    pub is_default: bool,
    /// The catalog entry still carries the placeholder sha256 (all zeros), so
    /// `kopitiam models pull` will download it and then **deliberately fail**
    /// the verify gate. Telling the user to "just pull it" would be wrong
    /// advice, so [`activate`] words the refusal differently for these.
    pub placeholder_checksum: bool,
    /// How to load it if picked.
    pub plan: LoadPlan,
}

/// What pressing Enter on a row means. Split out from key handling so the
/// decision is unit-tested without a terminal.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Activation {
    /// Go load this model and make it the chat's.
    Load { plan: LoadPlan, label: String },
    /// Cannot use this row. The string is a one-line Singlish reason that must
    /// tell the user what to do next — never just "no".
    Refuse(String),
}

/// The picker view's state: the rows, the cursor, and the last notice.
pub struct ModelPickerView {
    choices: Vec<ModelChoice>,
    selected: usize,
    notice: Option<String>,
}

impl ModelPickerView {
    /// Build the picker, with `active` being the `.gguf` the chat is on right
    /// now (`None` when it is running the echo stub).
    ///
    /// Reads the real environment and the real [`ModelStore`]; all the deciding
    /// is delegated to [`build_choices`], which takes both as arguments so tests
    /// drive it with a temp dir instead. The "which id is the default" question
    /// is answered by [`crate::adapter::configured_model_id`] rather than
    /// re-implemented here — one precedence rule, no drift.
    pub fn new(active: Option<&Path>) -> Self {
        let store = ModelStore::with_default_root().ok();
        let byo = std::env::var_os(MODEL_PATH_ENV)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);
        let configured = crate::adapter::configured_model_id();
        let choices = build_choices(store.as_ref(), byo.as_deref(), active, &configured);
        let selected = initial_selection(&choices);
        let notice = no_models_hint(&choices);
        Self { choices, selected, notice }
    }

    pub fn footer_hints(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("↑/↓ j/k", "move"),
            ("Enter", "use this model"),
            ("r", "refresh"),
            ("Esc", "back to chat"),
        ]
    }

    /// Handle one key. Enter asks the router to swap the model; anything that
    /// cannot happen becomes a notice, never a silent no-op.
    pub fn on_key(&mut self, key: KeyEvent) -> Transition {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => Transition::Quit,
            // Back to chat, not Home — the picker is a detour from chat, and
            // `Route::Chat` keeps the existing transcript because the chat state
            // lives on the router, not in the view.
            KeyCode::Esc => Transition::Open(Route::Chat),
            KeyCode::Down | KeyCode::Char('j') => {
                self.selected = step(self.selected, self.choices.len(), 1);
                Transition::Stay
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected = step(self.selected, self.choices.len(), -1);
                Transition::Stay
            }
            KeyCode::Char('r') => {
                let active = self.choices.iter().find(|c| c.active).map(|c| c.path.clone());
                let refreshed = Self::new(active.as_deref());
                self.choices = refreshed.choices;
                self.selected = clamp(self.selected, self.choices.len());
                self.notice = refreshed
                    .notice
                    .or(Some("refreshed lah — re-checked what is on disk".into()));
                Transition::Stay
            }
            KeyCode::Enter => match self.choices.get(self.selected) {
                None => Transition::Stay,
                Some(choice) => match activate(choice) {
                    Activation::Load { plan, label } => Transition::SelectModel { plan, label },
                    Activation::Refuse(why) => {
                        self.notice = Some(why);
                        Transition::Stay
                    }
                },
            },
            _ => Transition::Stay,
        }
    }

    /// Paint the picker: a short header, the model list, then the notice bar.
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let notice_h = if self.notice.is_some() { 3 } else { 0 };
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(2), Constraint::Min(0), Constraint::Length(notice_h)])
            .split(area);

        frame.render_widget(
            Paragraph::new(
                "Pick which local model the chat talk to. Only models already on disk can be \
                 used — to get a new one, go Home → Models and pull first, then press r here.",
            )
            .style(Style::default().fg(STEAM))
            .wrap(Wrap { trim: false }),
            rows[0],
        );

        let items: Vec<ListItem> = self.choices.iter().map(render_row).collect();
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM))
            .title(Span::styled(
                " kopitiam · pick a model ",
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
            ));
        let list = List::new(items).block(block).highlight_symbol("▍ ").highlight_style(
            Style::default().fg(GOLD).add_modifier(Modifier::BOLD | Modifier::REVERSED),
        );
        let mut state = ListState::default();
        state.select(Some(self.selected));
        frame.render_stateful_widget(list, rows[1], &mut state);

        if let Some(notice) = &self.notice {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(DIM));
            frame.render_widget(
                Paragraph::new(Text::from(Line::from(Span::styled(
                    notice.clone(),
                    Style::default().fg(STEAM).add_modifier(Modifier::ITALIC),
                ))))
                .block(block)
                .wrap(Wrap { trim: false }),
                rows[2],
            );
        }
    }
}

/// One list row, tinted by state: gold for the model in use, green for usable,
/// dim for not on disk, chilli for "on disk is not even possible right now".
fn render_row(choice: &ModelChoice) -> ListItem<'static> {
    let (mark, tone) = match (choice.active, choice.present, choice.placeholder_checksum) {
        (true, _, _) => ("◆ in use ", GOLD),
        (false, true, _) => ("● ready  ", PALM),
        (false, false, true) => ("✗ no sha ", CHILLI),
        (false, false, false) => ("○ absent ", DIM),
    };
    let origin = match choice.origin {
        Origin::Catalog => "catalog",
        Origin::Loose => "on-disk",
        Origin::Byo => "env BYO",
    };
    let mut spans = vec![
        Span::styled(mark, Style::default().fg(tone)),
        Span::styled(
            format!("{:<30}", choice.label),
            Style::default().fg(TAN).add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("{origin:<9}"), Style::default().fg(DIM)),
        Span::styled(choice.detail.clone(), Style::default().fg(DIM)),
    ];
    if choice.is_default {
        spans.push(Span::styled("  (default)", Style::default().fg(GOLD)));
    }
    ListItem::new(Line::from(spans))
}

// ---------------------------------------------------------------------------
// The testable part. Nothing below here touches a terminal.
// ---------------------------------------------------------------------------

/// Assemble every model the user could switch to, in a stable order: the
/// bring-your-own env path first (it is what the environment already chose),
/// then the catalog's chat models in catalog order, then any stray `.gguf`
/// found under the store root.
///
/// `store` is `None` when the model cache dir cannot be resolved at all (no
/// `HOME`, no `XDG_CACHE_HOME`). Catalog rows still appear then — so the user
/// see what exists — but with an empty `path` and `present: false`, which
/// [`activate`] turns into a refusal naming that cause. Rows are never silently
/// dropped, because a picker showing nothing teach the user nothing.
///
/// `configured` is the id the environment/default would choose, from
/// [`crate::adapter::configured_model_id`]; it only marks a row "(default)".
pub fn build_choices(
    store: Option<&ModelStore>,
    byo: Option<&Path>,
    active: Option<&Path>,
    configured: &str,
) -> Vec<ModelChoice> {
    let is_active = |path: &Path| active.is_some_and(|a| a == path);
    let mut choices = Vec::new();

    if let Some(path) = byo {
        choices.push(ModelChoice {
            id: file_label(path),
            label: file_label(path),
            detail: format!("{MODEL_PATH_ENV}={}", path.display()),
            present: path.is_file(),
            active: is_active(path),
            is_default: false,
            placeholder_checksum: false,
            plan: LoadPlan::ByPath(path.to_path_buf()),
            path: path.to_path_buf(),
            origin: Origin::Byo,
        });
    }

    for spec in Catalog::builtin().into_iter().filter(is_chat_weights) {
        let path = store.map(|s| spec_path(s, &spec)).unwrap_or_default();
        choices.push(ModelChoice {
            detail: format!(
                "{} · {} · {}",
                family_label(&spec.architecture),
                human_bytes(spec.artifacts.iter().map(|a| a.size_bytes).sum()),
                spec.license
            ),
            present: store.is_some_and(|s| s.is_present(&spec)),
            active: is_active(&path),
            is_default: spec.id == configured,
            placeholder_checksum: has_placeholder_checksum(&spec),
            // See `LoadPlan`: a BYO env path would shadow an id-routed load, so
            // when one is set every row loads by its exact path instead.
            plan: match byo {
                None => LoadPlan::ById(spec.id.clone()),
                Some(_) => LoadPlan::ByPath(path.clone()),
            },
            label: spec.display_name.clone(),
            id: spec.id.clone(),
            path,
            origin: Origin::Catalog,
        });
    }

    if let Some(store) = store {
        let known: Vec<PathBuf> = choices.iter().map(|c| c.path.clone()).collect();
        for path in discover_loose_gguf(store.root(), &known, LOOSE_SCAN_DEPTH) {
            choices.push(ModelChoice {
                id: file_label(&path),
                label: file_label(&path),
                detail: path.display().to_string(),
                present: true, // it was found by walking the filesystem
                active: is_active(&path),
                is_default: false,
                placeholder_checksum: false,
                plan: LoadPlan::ByPath(path.clone()),
                path,
                origin: Origin::Loose,
            });
        }
    }

    choices
}

/// Is this catalog entry actually chat weights?
///
/// The catalog is shared with OCR: alongside the `.gguf` LLMs it carries
/// `tessdata-*` rows whose artifact is a Tesseract `eng.traineddata`. Handing
/// one of those to [`LocalAdapter::load`] can only ever fail (it is not GGUF at
/// all), so offering them in a *chat model* picker is a trap. The test is the
/// artifact extension rather than the id prefix or the architecture: extension
/// is what the loader actually cares about, so a future non-LLM row named
/// anything at all is still excluded.
fn is_chat_weights(spec: &ModelSpec) -> bool {
    !spec.artifacts.is_empty() && spec.artifacts.iter().all(|a| is_gguf(Path::new(&a.filename)))
}

/// Does this entry still carry the catalog's placeholder sha256 (all zeros)?
///
/// Such an entry cannot be acquired: `kopitiam models pull` downloads it and
/// then fails the checksum gate on purpose. Detected by shape (64 zeros) rather
/// than by importing the constant, because `kopitiam_models`' `PLACEHOLDER_SHA256`
/// is private — if the catalog ever exports it, prefer that over this check.
fn has_placeholder_checksum(spec: &ModelSpec) -> bool {
    spec.artifacts
        .iter()
        .any(|a| !a.sha256.is_empty() && a.sha256.chars().all(|c| c == '0'))
}

/// Find `.gguf` files under `root` that the catalog does not already account
/// for — i.e. weights somebody copied in by hand.
///
/// Walks at most `max_depth` directory levels below `root` (0 = only `root`
/// itself) and returns paths sorted, so the picker's order is deterministic
/// across runs and across platforms. Unreadable directories are skipped rather
/// than reported: a stray permission error must not stop the user picking a
/// model that IS readable.
pub fn discover_loose_gguf(root: &Path, known: &[PathBuf], max_depth: usize) -> Vec<PathBuf> {
    let mut found = Vec::new();
    walk_gguf(root, known, max_depth, &mut found);
    found.sort();
    found
}

fn walk_gguf(dir: &Path, known: &[PathBuf], depth_left: usize, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        // `symlink_metadata` does NOT follow the link, which is the point: a
        // symlinked directory inside the store would otherwise be descended into
        // as if it were part of the store. Two ways that bites — a link pointing
        // at, say, `$HOME` silently surfaces unrelated `.gguf` files as if the
        // user had put them there, and a link pointing at an ancestor walks the
        // same tree again. The depth cap bounds the damage either way, so this is
        // not a hang, but "models appear that the user never placed here" is
        // confusing enough to be worth one stat call. Symlinked *files* are still
        // accepted: deliberately pointing at weights elsewhere is a reasonable
        // thing to do, and it is only directory RECURSION that surprises.
        let followable_dir = std::fs::symlink_metadata(&path)
            .map(|m| m.file_type().is_dir())
            .unwrap_or(false);
        if followable_dir {
            if depth_left > 0 {
                walk_gguf(&path, known, depth_left - 1, out);
            }
        } else if path.is_file() && is_gguf(&path) && !known.contains(&path) {
            out.push(path);
        }
    }
}

/// Case-insensitive `.gguf` test — Windows and Termux both show up in this
/// workspace, and a file copied off a phone can easily be `MODEL.GGUF`.
fn is_gguf(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

/// The store path of a spec's first artifact. Our catalog LLMs are
/// single-artifact `.gguf`, and [`LocalAdapter::load`] takes exactly one weights
/// file, so the first artifact is the right one — same rule
/// `crate::adapter`'s path resolution follows.
fn spec_path(store: &ModelStore, spec: &ModelSpec) -> PathBuf {
    spec.artifacts
        .first()
        .map(|a| store.artifact_path(spec, a))
        .unwrap_or_else(|| store.root().join(&spec.id))
}

/// A short label for a bare path: its file name, or the whole path when it has
/// no file name (which would be odd, but must not panic).
fn file_label(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Where the cursor starts: on the model in use, else on the first one that can
/// actually be loaded, else row 0. Never returns an out-of-range index — an
/// empty list gives 0, which the list widget renders as "nothing selected".
pub fn initial_selection(choices: &[ModelChoice]) -> usize {
    choices
        .iter()
        .position(|c| c.active)
        .or_else(|| choices.iter().position(|c| c.present))
        .unwrap_or(0)
}

/// Decide what Enter on this row does.
///
/// Refusals are deliberately actionable — each one names the next step, because
/// this is the screen where a user with no weights on disk gets stuck. In
/// particular a placeholder-checksum row must NOT be told to "pull it": that
/// pull is guaranteed to fail the verify gate, and sending someone to download
/// hundreds of MB for a certain failure is worse than saying nothing.
pub fn activate(choice: &ModelChoice) -> Activation {
    if choice.active {
        return Activation::Refuse(format!("already using {} what — pick another one", choice.label));
    }
    if choice.path.as_os_str().is_empty() {
        return Activation::Refuse(
            "model store cannot be found (no HOME / XDG_CACHE_HOME set) — set one, then press r"
                .into(),
        );
    }
    if !choice.present {
        return Activation::Refuse(if choice.placeholder_checksum {
            format!(
                "{} not on disk, and its catalog entry still got a placeholder checksum — \
                 `models pull` will fail the verify gate. Drop your own .gguf at {} instead.",
                choice.id,
                choice.path.display()
            )
        } else {
            format!(
                "{} not on disk yet — go Home → Models and pull it first, then come back press r",
                choice.id
            )
        });
    }
    Activation::Load { plan: choice.plan.clone(), label: choice.label.clone() }
}

/// A hint for when the picker has nothing usable at all, so the screen is never
/// just a wall of "absent" with no way forward. `None` when at least one model
/// is loadable.
pub fn no_models_hint(choices: &[ModelChoice]) -> Option<String> {
    if choices.iter().any(|c| c.present) {
        return None;
    }
    Some(
        "No model on disk at all leh — chat stay on the echo stub until you pull one \
         (Home → Models, or `kopitiam models pull`)."
            .into(),
    )
}

/// Carry out a [`LoadPlan`] and hand back the adapter the chat should use.
///
/// Never returns an error and never panics: a bad file must leave the user in a
/// working chat with an explanation, not drop them out of the TUI. This and
/// [`load_choice`] are the only effectful functions in this module.
pub fn load_plan(plan: &LoadPlan) -> SelectedAdapter {
    match plan {
        LoadPlan::ById(id) => select_adapter_for(id),
        LoadPlan::ByPath(path) => load_choice(path),
    }
}

/// Load an exact `.gguf`, degrading to the echo stub if it will not load —
/// the same shape [`crate::adapter::select_adapter`] uses, for a path that no
/// catalog id can name.
pub fn load_choice(path: &Path) -> SelectedAdapter {
    match LocalAdapter::load(path) {
        Ok(adapter) => SelectedAdapter::Local {
            adapter: Box::new(adapter),
            source: path.to_path_buf(),
        },
        Err(error) => SelectedAdapter::Echo {
            adapter: EchoAdapter,
            reason: FallbackReason::LoadFailed {
                source: path.to_path_buf(),
                error: format!("{error:#}"),
            },
        },
    }
}

/// The `.gguf` currently powering the chat, if it is a real local model. `None`
/// on the echo stub — there is no file to point at.
pub fn active_source(selected: &SelectedAdapter) -> Option<&Path> {
    match selected {
        SelectedAdapter::Local { source, .. } => Some(source.as_path()),
        SelectedAdapter::Echo { .. } => None,
    }
}

/// The one-line result of a switch, shown in the chat transcript. Must say
/// plainly whether the swap worked **and why not**, because a failed load
/// silently falling back to echo looks exactly like a working model giving
/// nonsense answers — that is the failure mode this whole note exists to stop.
pub fn switch_note(label: &str, selected: &SelectedAdapter) -> String {
    match selected {
        SelectedAdapter::Local { source, .. } => {
            format!("switched to {label} — running on CPU from {}", source.display())
        }
        SelectedAdapter::Echo { reason, .. } => match reason {
            FallbackReason::LoadFailed { source, error } => format!(
                "cannot load {label} ({}): {} — still on the echo stub, no real inference",
                source.display(),
                first_line(error)
            ),
            FallbackReason::NoModelOnDisk { model_id, expected_store_path } => format!(
                "{model_id} not on disk ({}) — still on the echo stub, no real inference",
                expected_store_path.display()
            ),
        },
    }
}

/// A compact status for the chat header: which rung of the Offline-First
/// pipeline is answering, and — when it is not a real model — *why*.
///
/// # The bug this replaces
///
/// The old version consulted [`SelectedAdapter::notice`] only on the local
/// branch and rendered a hardcoded "echo stub — no .gguf yet" for every Echo
/// case. So [`FallbackReason::LoadFailed`] — a file WAS found and the loader
/// rejected it — looked identical to having no file at all, and the loader's
/// real error was thrown away at the UI boundary. That made a broken `.gguf`
/// undiagnosable from inside the TUI while `kopitiam ai chat` printed it fine.
/// **Keep the two cases distinct, and keep the loader's own message in the
/// LoadFailed one** — anything else and the next failure needs another
/// debugging session instead of just being readable on screen.
pub fn short_status(selected: &SelectedAdapter) -> String {
    match selected {
        SelectedAdapter::Local { .. } => selected
            .notice()
            .lines()
            .next()
            .unwrap_or("local model on CPU")
            .to_string(),
        SelectedAdapter::Echo { reason, .. } => match reason {
            FallbackReason::NoModelOnDisk { model_id, .. } => {
                format!("echo stub — no {model_id} on disk (Ctrl-P to pick, Models to pull)")
            }
            FallbackReason::LoadFailed { source, error } => format!(
                "echo stub — {} cannot load: {} (Ctrl-P to pick another)",
                file_label(source),
                first_line(error)
            ),
        },
    }
}

fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or(text)
}

/// Move the cursor by `delta`, wrapping top-to-bottom like the home menu does.
/// Wrapping (rather than saturating) because this list is short and a user
/// scrolling past the end expect to come round again.
fn step(sel: usize, len: usize, delta: isize) -> usize {
    if len == 0 {
        return 0;
    }
    let len_i = len as isize;
    (((sel as isize + delta) % len_i + len_i) % len_i) as usize
}

fn clamp(sel: usize, len: usize) -> usize {
    if len == 0 { 0 } else { sel.min(len - 1) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kopitiam_models::DEFAULT_MODEL_ID;

    /// A store rooted in a fresh temp dir. Returns the tempdir so the caller
    /// keep it alive for the length of the test.
    fn empty_store() -> (tempfile::TempDir, ModelStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::with_root(dir.path());
        (dir, store)
    }

    /// Put a placeholder file where the default model's `.gguf` would live.
    /// Presence is a file-exists check (AID-0029), so contents don't matter.
    fn place_default_model(store: &ModelStore) -> PathBuf {
        let spec = Catalog::find(DEFAULT_MODEL_ID).unwrap();
        let path = spec_path(store, &spec);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not a real gguf, just proving presence").unwrap();
        path
    }

    fn choices_for(store: Option<&ModelStore>) -> Vec<ModelChoice> {
        build_choices(store, None, None, DEFAULT_MODEL_ID)
    }

    /// Every catalog LLM is offered, with presence marked per row.
    #[test]
    fn every_catalog_chat_model_is_offered_with_its_presence_marked() {
        let (_dir, store) = empty_store();
        let present = place_default_model(&store);

        let choices = choices_for(Some(&store));

        let default = choices.iter().find(|c| c.id == DEFAULT_MODEL_ID).expect("default row");
        assert!(default.present, "the model we just wrote must read as present");
        assert!(default.is_default, "and be flagged as what a fresh start would pick");
        assert_eq!(default.path, present);
        assert_eq!(choices.iter().filter(|c| c.present).count(), 1);
        assert!(choices.iter().all(|c| !c.detail.is_empty() && !c.label.is_empty()));
    }

    /// Tesseract `.traineddata` rows share the catalog with the LLMs and are NOT
    /// chat weights — offering one would guarantee a load failure.
    #[test]
    fn tesseract_traineddata_rows_are_not_offered_as_chat_models() {
        let (_dir, store) = empty_store();
        let choices = choices_for(Some(&store));

        assert!(
            !choices.iter().any(|c| c.id.starts_with("tessdata-")),
            "tessdata rows must never appear in a chat model picker"
        );
        // And the filter is by artifact extension, not by id prefix.
        let tess = Catalog::find("tessdata-eng").expect("the catalog still ships tessdata-eng");
        assert!(!is_chat_weights(&tess));
        let llm = Catalog::find(DEFAULT_MODEL_ID).unwrap();
        assert!(is_chat_weights(&llm));
        // Everything offered is a .gguf row.
        assert!(choices.iter().all(|c| is_gguf(&c.path) || c.path.as_os_str().is_empty()));
    }

    /// The two placeholder-sha entries must be flagged, and their refusal must
    /// NOT send the user off to a pull that is designed to fail.
    #[test]
    fn placeholder_checksum_rows_are_flagged_and_refuse_differently() {
        let (_dir, store) = empty_store();
        let choices = choices_for(Some(&store));

        let flagged: Vec<&str> =
            choices.iter().filter(|c| c.placeholder_checksum).map(|c| c.id.as_str()).collect();
        assert!(flagged.contains(&"qwen2.5-0.5b-instruct-q4_0"), "{flagged:?}");
        assert!(flagged.contains(&"llama-3.2-1b-instruct-q4_0"), "{flagged:?}");
        assert!(!flagged.contains(&DEFAULT_MODEL_ID), "the default has a real sha: {flagged:?}");

        let row = choices.iter().find(|c| c.placeholder_checksum).unwrap();
        let why = match activate(row) {
            Activation::Refuse(why) => why,
            other => panic!("expected a refusal, got {other:?}"),
        };
        assert!(why.contains("placeholder checksum"), "{why}");
        assert!(!why.contains("pull it first"), "must not advise a doomed pull: {why}");
    }

    /// A catalog row normally loads **by id**, so store-path resolution stays in
    /// `crate::adapter` and is shared with the non-TUI chat.
    #[test]
    fn catalog_rows_load_by_id_so_resolution_is_not_duplicated() {
        let (_dir, store) = empty_store();
        let choices = choices_for(Some(&store));
        let row = choices.iter().find(|c| c.id == DEFAULT_MODEL_ID).unwrap();
        assert_eq!(row.plan, LoadPlan::ById(DEFAULT_MODEL_ID.to_string()));
    }

    /// ...but when `KOPITIAM_MODEL_GGUF` is set, `select_adapter_for` would let
    /// that path win over any id we pass, so every row must load by path instead
    /// — otherwise the UI shows one model and the answers come from another.
    #[test]
    fn a_byo_env_path_forces_every_row_to_load_by_path() {
        let (dir, store) = empty_store();
        place_default_model(&store);
        let byo = dir.path().join("my-own.gguf");
        std::fs::write(&byo, b"x").unwrap();

        let choices = build_choices(Some(&store), Some(&byo), None, DEFAULT_MODEL_ID);

        assert!(
            choices.iter().all(|c| matches!(c.plan, LoadPlan::ByPath(_))),
            "an env BYO path shadows id-routed loads, so nothing may route by id"
        );
        let default_row = choices.iter().find(|c| c.id == DEFAULT_MODEL_ID).unwrap();
        assert_eq!(default_row.plan, LoadPlan::ByPath(default_row.path.clone()));
    }

    #[test]
    fn no_store_still_lists_the_catalog_but_nothing_is_usable() {
        let choices = choices_for(None);
        assert!(!choices.is_empty());
        assert!(choices.iter().all(|c| !c.present && c.path.as_os_str().is_empty()));
        let why = match activate(&choices[0]) {
            Activation::Refuse(why) => why,
            other => panic!("expected a refusal, got {other:?}"),
        };
        assert!(why.contains("XDG_CACHE_HOME"), "{why}");
    }

    #[test]
    fn the_byo_env_path_leads_the_list_and_is_labelled_as_such() {
        let (dir, store) = empty_store();
        let byo = dir.path().join("my-own.gguf");
        std::fs::write(&byo, b"x").unwrap();

        let choices = build_choices(Some(&store), Some(&byo), None, DEFAULT_MODEL_ID);

        assert_eq!(choices[0].origin, Origin::Byo);
        assert_eq!(choices[0].label, "my-own.gguf");
        assert!(choices[0].present);
        assert!(choices[0].detail.contains(MODEL_PATH_ENV));
    }

    #[test]
    fn a_byo_path_that_does_not_exist_is_shown_but_not_usable() {
        let (_dir, store) = empty_store();
        let byo = PathBuf::from("/does/not/exist/nope.gguf");

        let choices = build_choices(Some(&store), Some(&byo), None, DEFAULT_MODEL_ID);

        assert!(!choices[0].present);
        assert!(matches!(activate(&choices[0]), Activation::Refuse(_)));
    }

    #[test]
    fn hand_dropped_gguf_under_the_store_is_discovered() {
        let (dir, store) = empty_store();
        let known = place_default_model(&store);
        let loose = dir.path().join("nested").join("hand-copied.GGUF");
        std::fs::create_dir_all(loose.parent().unwrap()).unwrap();
        std::fs::write(&loose, b"x").unwrap();
        std::fs::write(dir.path().join("readme.txt"), b"not a model").unwrap();

        let found = discover_loose_gguf(dir.path(), &[known.clone()], LOOSE_SCAN_DEPTH);
        assert_eq!(found, vec![loose.clone()], "only the unknown .gguf, case-insensitively");

        let choices = choices_for(Some(&store));
        let row = choices.iter().find(|c| c.path == loose).expect("the loose row");
        assert_eq!(row.origin, Origin::Loose);
        assert!(row.present);
        assert_eq!(row.plan, LoadPlan::ByPath(loose));
        assert!(!choices.iter().any(|c| c.origin == Origin::Loose && c.path == known));
    }

    #[test]
    fn the_loose_scan_does_not_descend_symlinked_directories() {
        // A symlinked dir inside the store must not be walked: pointing it at a
        // home directory would surface unrelated .gguf files as if the user had
        // put them in the store, and pointing it at an ancestor re-walks the same
        // tree. Symlinked FILES stay allowed — aiming at weights held elsewhere
        // is legitimate; only directory recursion surprises.
        #[cfg(unix)]
        {
            let store = tempfile::tempdir().unwrap();
            let outside = tempfile::tempdir().unwrap();
            std::fs::write(outside.path().join("elsewhere.gguf"), b"x").unwrap();
            std::os::unix::fs::symlink(outside.path(), store.path().join("linked")).unwrap();

            let found = discover_loose_gguf(store.path(), &[], 3);
            assert!(
                found.is_empty(),
                "must not descend a symlinked directory, but found {found:?}"
            );

            // The same file reached by a symlinked FILE is still offered.
            std::os::unix::fs::symlink(
                outside.path().join("elsewhere.gguf"),
                store.path().join("direct.gguf"),
            )
            .unwrap();
            let found = discover_loose_gguf(store.path(), &[], 3);
            assert_eq!(found.len(), 1, "a symlinked .gguf file is still a valid pick");
        }
    }

    #[test]    fn the_loose_scan_respects_its_depth_cap() {
        let dir = tempfile::tempdir().unwrap();
        let deep = dir.path().join("a").join("b").join("c").join("too-deep.gguf");
        std::fs::create_dir_all(deep.parent().unwrap()).unwrap();
        std::fs::write(&deep, b"x").unwrap();

        assert!(discover_loose_gguf(dir.path(), &[], 2).is_empty());
        assert_eq!(discover_loose_gguf(dir.path(), &[], 3), vec![deep]);
    }

    #[test]
    fn the_scan_survives_a_root_that_is_not_there() {
        assert!(discover_loose_gguf(Path::new("/no/such/root"), &[], 3).is_empty());
    }

    #[test]
    fn the_cursor_starts_on_the_model_in_use_then_the_first_usable_one() {
        let (_dir, store) = empty_store();
        let path = place_default_model(&store);

        let choices = choices_for(Some(&store));
        let first_present = choices.iter().position(|c| c.present).unwrap();
        assert_eq!(initial_selection(&choices), first_present);

        let choices = build_choices(Some(&store), None, Some(&path), DEFAULT_MODEL_ID);
        let active_at = choices.iter().position(|c| c.active).unwrap();
        assert_eq!(initial_selection(&choices), active_at);

        let (_dir2, empty) = empty_store();
        assert_eq!(initial_selection(&choices_for(Some(&empty))), 0);
        assert_eq!(initial_selection(&[]), 0);
    }

    #[test]
    fn a_present_row_activates_into_a_load_of_exactly_that_model() {
        let (_dir, store) = empty_store();
        place_default_model(&store);
        let choices = choices_for(Some(&store));
        let row = choices.iter().find(|c| c.present).unwrap();

        match activate(row) {
            Activation::Load { plan, label } => {
                assert_eq!(plan, row.plan);
                assert_eq!(label, row.label);
            }
            other => panic!("a present model must be loadable, got {other:?}"),
        }
    }

    #[test]
    fn an_absent_row_refuses_and_points_at_the_models_view() {
        let (_dir, store) = empty_store();
        let choices = choices_for(Some(&store));
        let row = choices.iter().find(|c| !c.present && !c.placeholder_checksum).unwrap();

        let why = match activate(row) {
            Activation::Refuse(why) => why,
            other => panic!("an absent model must refuse, got {other:?}"),
        };
        assert!(why.contains(&row.id), "the refusal must name the model: {why}");
        assert!(why.contains("Models"), "and where to get it: {why}");
    }

    #[test]
    fn re_picking_the_model_already_in_use_refuses_instead_of_reloading() {
        let (_dir, store) = empty_store();
        let path = place_default_model(&store);
        let choices = build_choices(Some(&store), None, Some(&path), DEFAULT_MODEL_ID);
        let row = choices.iter().find(|c| c.active).unwrap();

        assert!(matches!(activate(row), Activation::Refuse(_)));
    }

    #[test]
    fn the_empty_picker_hint_appears_only_when_nothing_is_usable() {
        let (_dir, empty) = empty_store();
        let hint = no_models_hint(&choices_for(Some(&empty))).expect("a hint when nothing on disk");
        assert!(hint.contains("Models"), "{hint}");

        place_default_model(&empty);
        assert_eq!(no_models_hint(&choices_for(Some(&empty))), None);
    }

    #[test]
    fn loading_a_file_that_is_not_a_gguf_degrades_to_echo_not_a_crash() {
        let dir = tempfile::tempdir().unwrap();
        let bogus = dir.path().join("bogus.gguf");
        std::fs::write(&bogus, b"definitely not GGUF bytes").unwrap();

        let selected = load_plan(&LoadPlan::ByPath(bogus.clone()));

        assert!(!selected.is_local());
        // The user is told the switch did not take, WITH the loader's reason —
        // this is the "self-diagnosing" requirement, not a nice-to-have.
        let note = switch_note("bogus.gguf", &selected);
        assert!(note.contains("cannot load"), "{note}");
        assert!(note.contains("echo stub"), "{note}");
        assert!(note.contains("bogus.gguf"), "{note}");
        assert!(short_status(&selected).contains("bogus.gguf"), "and in the header too");
    }

    #[test]
    fn loading_a_path_that_does_not_exist_degrades_to_echo_too() {
        let selected = load_choice(Path::new("/no/such/model.gguf"));
        assert!(!selected.is_local());
        assert!(matches!(
            selected,
            SelectedAdapter::Echo { reason: FallbackReason::LoadFailed { .. }, .. }
        ));
    }

    /// The header must tell "nothing on disk" apart from "found it but it won't
    /// load", and must carry the loader's own words in the second case. Getting
    /// this wrong is what made a broken `.gguf` undiagnosable from the TUI.
    #[test]
    fn the_status_line_tells_no_model_apart_from_a_broken_model() {
        let nothing = SelectedAdapter::Echo {
            adapter: EchoAdapter,
            reason: FallbackReason::NoModelOnDisk {
                model_id: DEFAULT_MODEL_ID.to_string(),
                expected_store_path: PathBuf::from("/cache/x.gguf"),
            },
        };
        let broken = SelectedAdapter::Echo {
            adapter: EchoAdapter,
            reason: FallbackReason::LoadFailed {
                source: PathBuf::from("/cache/broken.gguf"),
                error: "bad GGUF magic\nsecond line of context".into(),
            },
        };

        let nothing_status = short_status(&nothing);
        let broken_status = short_status(&broken);
        assert_ne!(nothing_status, broken_status);
        assert!(nothing_status.contains(DEFAULT_MODEL_ID), "{nothing_status}");
        assert!(broken_status.contains("bad GGUF magic"), "the loader's own error: {broken_status}");
        assert!(broken_status.contains("broken.gguf"), "and which file: {broken_status}");
        // Both point the user at the way out.
        assert!(nothing_status.contains("Ctrl-P"));
        assert!(broken_status.contains("Ctrl-P"));
        // A multi-line loader error is squeezed onto one line for the header.
        assert!(!broken_status.contains('\n'));
        assert!(!switch_note("x", &broken).contains('\n'));
    }

    #[test]
    fn active_source_is_the_gguf_only_when_a_real_model_is_loaded() {
        let echo = SelectedAdapter::Echo {
            adapter: EchoAdapter,
            reason: FallbackReason::NoModelOnDisk {
                model_id: DEFAULT_MODEL_ID.to_string(),
                expected_store_path: PathBuf::new(),
            },
        };
        assert_eq!(active_source(&echo), None);
    }

    #[test]
    fn the_cursor_wraps_both_ways_and_survives_an_empty_list() {
        assert_eq!(step(0, 3, 1), 1);
        assert_eq!(step(2, 3, 1), 0, "past the end wraps to the top");
        assert_eq!(step(0, 3, -1), 2, "before the top wraps to the end");
        assert_eq!(step(0, 0, 1), 0, "an empty list never panics");
        assert_eq!(clamp(9, 3), 2);
        assert_eq!(clamp(9, 0), 0);
    }

    /// Enter on a usable row must hand the router a `SelectModel` carrying that
    /// exact plan — this is the whole feature in one assertion.
    #[test]
    fn enter_arms_a_model_switch_for_the_selected_row() {
        let (_dir, store) = empty_store();
        place_default_model(&store);
        let choices = choices_for(Some(&store));
        let selected = choices.iter().position(|c| c.present).unwrap();
        let expected = choices[selected].plan.clone();
        let mut view = ModelPickerView { choices, selected, notice: None };

        match view.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::empty())) {
            Transition::SelectModel { plan, .. } => assert_eq!(plan, expected),
            _ => panic!("Enter on a ready model must arm a SelectModel transition"),
        }
    }

    /// Enter on an unusable row must not arm anything, and must leave a notice
    /// behind — the user pressed a key, so something has to answer.
    #[test]
    fn enter_on_an_absent_row_stays_put_and_explains_itself() {
        let (_dir, store) = empty_store();
        let choices = choices_for(Some(&store));
        let mut view = ModelPickerView { choices, selected: 0, notice: None };

        assert!(matches!(
            view.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::empty())),
            Transition::Stay
        ));
        assert!(view.notice.is_some());
    }

    #[test]
    fn esc_goes_back_to_chat_not_home() {
        let mut view = ModelPickerView { choices: Vec::new(), selected: 0, notice: None };
        assert!(matches!(
            view.on_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::empty())),
            Transition::Open(Route::Chat)
        ));
    }
}
