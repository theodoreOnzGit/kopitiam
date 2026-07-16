//! The operator-pending grammar: `[count] ["register] operator [count] motion`.
//!
//! # Why this is a state machine and not a match on key sequences
//!
//! The brief calls this out as the single most important design decision in
//! the crate, and the reason is concrete: `d2w`, `2dw`, `"ad3d`, and `ci(`
//! all have to compose from the *same* pieces (an optional register, one or
//! two counts that multiply together, an operator, and a motion or text
//! object) in different orders and combinations. A design built out of
//! `match` arms on whole key sequences (`"dw" => ..., "d2w" => ..., "2dw" =>
//! ...`) cannot represent that composition — it can only enumerate finitely
//! many sequences someone remembered to write down, and every new operator
//! or motion multiplies the number of arms that need writing. `Pending`
//! instead accumulates *slots* (`register`, `count1`, `operator`, `count2`,
//! and whichever multi-key token is in flight) one key at a time, and a
//! command is complete the moment a *motion* or *text object* or
//! *no-argument command* key arrives, regardless of what filled the slots
//! before it. Every operator, motion and text object this crate will ever
//! grow plugs into the same four slots; nothing here changes shape as the
//! grammar's vocabulary grows.
//!
//! `Pending` does not touch [`crate::text::Buffer`] at all. It only knows
//! about *keys* and the *shape* of a command; resolving a completed command
//! against real buffer contents (applying a motion, resolving a text
//! object's range, running an operator) is [`super::Editor`]'s job. That
//! split is what makes this file's logic testable without a buffer, and
//! keeps [`super::Editor::handle_key`] from having to know the grammar's
//! internals.

use super::key::{Key, KeyCode};
use super::motion::{FindKind, Motion};
use super::operator::Operator;
use super::text_object::{ObjectScope, TextObject};

/// What a completed key sequence turned out to mean. `Pending::feed` returns
/// one of these once the grammar is satisfied; `Editor::handle_key` executes
/// it. Every variant already carries its fully-resolved `count`/`register` —
/// callers never need to look back at `Pending`'s internals.
#[derive(Debug, Clone, PartialEq)]
pub enum GrammarCommand {
    /// A bare motion with no operator: just move the cursor.
    Move { count: Option<usize>, motion: Motion },
    /// An operator applied to a motion's range (`dw`, `2d3w`, `"ayj`).
    OperatorMotion { register: Option<char>, count: Option<usize>, operator: Operator, motion: Motion },
    /// An operator applied to a text object's range (`di(`, `ca{`, `yit`).
    OperatorTextObject { register: Option<char>, operator: Operator, scope: ObjectScope, object: TextObject },
    /// An operator applied to whole lines via its own doubled letter
    /// (`dd`, `guu`, `>>`, `3>>`).
    OperatorLines { register: Option<char>, count: Option<usize>, operator: Operator },
    /// `;`/`,`: repeat the last `f`/`F`/`t`/`T`, forwards or reversed.
    /// `Pending` cannot resolve this to a concrete [`Motion::FindChar`]
    /// itself — see the field comment where this is produced — so `Editor`
    /// looks up its remembered `(kind, target)` and either moves the cursor
    /// (`operator: None`) or runs `operator` over the resulting range, the
    /// same as it would for any other resolved motion.
    RepeatFind { register: Option<char>, count: Option<usize>, operator: Option<Operator>, reverse: bool },
    /// `x`
    DeleteCharForward { register: Option<char>, count: Option<usize> },
    /// `X`
    DeleteCharBackward { register: Option<char>, count: Option<usize> },
    /// `s`
    SubstituteChar { register: Option<char>, count: Option<usize> },
    /// `r{c}`
    ReplaceChar { count: Option<usize>, ch: char },
    /// `~`
    ToggleCaseUnderCursor { count: Option<usize> },
    /// `J`
    JoinLines { count: Option<usize> },
    /// `p`/`P`
    Put { register: Option<char>, count: Option<usize>, before: bool },
    /// `i a I A o O`
    EnterInsert(InsertPos),
    Undo,
    Redo,
    /// `.`
    RepeatLast,
    /// `v`, `V`, `<C-v>`
    EnterVisual(super::VisualKind),
    /// `:`
    EnterCommandLine,
    /// `q{reg}`: start recording. (Stopping an in-progress recording is
    /// handled by `Editor` before a key ever reaches `Pending` — see
    /// `Editor::handle_normal_key`'s docs for why that one case can't live
    /// in this grammar.)
    StartRecording { register: char },
    /// `@{reg}`
    PlayMacro { register: char, count: Option<usize> },
    /// `@@`
    ReplayLastMacro { count: Option<usize> },
    /// `m{a-z}`: set a mark at the cursor.
    SetMark { name: char },
    /// `` `{a-z} `` (`exact = true`, jump to the mark's exact column) or
    /// `'{a-z}` (`exact = false`, jump to the first non-blank of the mark's
    /// line). The back-tick/apostrophe pair `` `` `` / `''` (jump to where
    /// the last jump started) is spelled with `name == '\'' `/'`'`.
    JumpMark { name: char, exact: bool },
    /// `/` or `?`: open the search prompt (`forward` false for `?`).
    StartSearch { forward: bool },
    /// `n`/`N`: repeat the last search, same direction or reversed.
    RepeatSearch { reverse: bool },
    /// `*`/`#`: search for the keyword under the cursor, forward/backward.
    SearchWord { forward: bool },
    /// `zz`/`zt`/`zb`: reposition the viewport around the cursor.
    Scroll(crate::core::ViewportScroll),
    /// `gv`: reselect the last visual selection.
    ReselectVisual,
    /// `ZZ`: write the buffer if modified, then quit (same as `:x`).
    WriteQuit,
    /// `ZQ`: quit without writing, discarding any changes (same as `:q!`).
    QuitForce,
    /// `&`: repeat the last `:s` (substitute) on the current line, dropping
    /// its flags — vim's classic "do that substitution again here" key. Like
    /// [`GrammarCommand::RepeatSearch`], `Pending` has no memory of *what* the
    /// last substitution was (only `Editor` does), so this resolves to a
    /// command for `Editor` to fill in.
    RepeatSubstitute,
    /// `['`/`` [` ``/`]'`/`` ]` ``: jump to the previous (`forward == false`)
    /// or next (`forward == true`) lowercase mark, by buffer position. `exact`
    /// is `true` for the back-tick forms (land on the mark's exact column) and
    /// `false` for the apostrophe forms (land on the first non-blank of the
    /// mark's line) — the same distinction [`GrammarCommand::JumpMark`] draws.
    /// Resolved by `Editor` because the marks live in the buffer, which
    /// `Pending` cannot see (see the module docs).
    JumpBracketMark { forward: bool, exact: bool },
}

/// Where `i a I A o O` place the cursor before switching to Insert mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertPos {
    /// `i`: before the cursor.
    Before,
    /// `a`: after the cursor.
    After,
    /// `I`: at the first non-blank of the line.
    LineStart,
    /// `A`: at the end of the line.
    LineEnd,
    /// `o`: on a new line below.
    NewLineBelow,
    /// `O`: on a new line above.
    NewLineAbove,
}

/// What happened to a key fed into [`Pending::feed`].
#[derive(Debug, Clone, PartialEq)]
pub enum FeedResult {
    /// More keys are needed before the command means anything.
    Continue,
    /// The command is complete.
    Complete(GrammarCommand),
    /// The key sequence is not a valid command (vim would beep and reset to
    /// Normal with nothing pending).
    Invalid,
}

/// What kind of multi-key token is currently being collected. `Fresh` is the
/// state at the very start of a command *and* the state right after an
/// operator has been set (both mean "the next key starts something new" —
/// see the module docs for why that reuse is exactly the point).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    #[default]
    Fresh,
    AwaitingRegisterName,
    AwaitingG,
    AwaitingFind(FindKind),
    AwaitingReplaceChar,
    /// Reached only when an operator is already pending (`di`, `ca`, ...):
    /// waiting for the object letter (`w`, `(`, `"`, `t`, ...).
    AwaitingTextObject(ObjectScope),
    AwaitingMacroRegister,
    AwaitingPlayRegister,
    /// After `m`: waiting for the mark letter to set.
    AwaitingMarkSet,
    /// After `` ` `` (`exact = true`) or `'` (`exact = false`): waiting for
    /// the mark letter to jump to.
    AwaitingMarkJump { exact: bool },
    /// After `z`: waiting for `z`/`t`/`b` (viewport reposition).
    AwaitingZ,
    /// After `Z`: waiting for the second `Z` (`ZZ` = write+quit) or `Q`
    /// (`ZQ` = quit without saving).
    AwaitingBigZ,
    /// After `[` (`forward == false`) or `]` (`forward == true`): waiting for
    /// the bracket-motion's second key (`[`, `]`, `(`, `)`, `{`, `}`, `m`,
    /// `M`, `'`, `` ` ``). See [`Pending::feed_bracket`].
    AwaitingBracket { forward: bool },
}

/// The accumulator itself. See the module docs for the grammar it parses.
#[derive(Debug, Clone, Default)]
pub struct Pending {
    register: Option<char>,
    count1: Option<usize>,
    operator: Option<Operator>,
    count2: Option<usize>,
    state_: Option<State>, // `None` behaves exactly like `Some(Fresh)`; see `state()`.
}

impl Pending {
    pub fn new() -> Self {
        Self::default()
    }

    fn state(&self) -> State {
        self.state_.unwrap_or(State::Fresh)
    }

    /// `true` when nothing has been accumulated at all — the point at which
    /// keys with meanings *outside* this grammar (`v`, `:`, a already-recording
    /// `q`) are safe for `Editor` to intercept before they ever reach `feed`.
    pub fn is_idle(&self) -> bool {
        self.register.is_none() && self.count1.is_none() && self.operator.is_none() && self.count2.is_none() && self.state_.is_none()
    }

    /// The count to report on an aborted/no-op command; also used by
    /// `Editor` to render the pending count/register in a statusline.
    pub fn effective_count(&self) -> Option<usize> {
        match (self.count1, self.count2) {
            (None, None) => None,
            (a, b) => Some(a.unwrap_or(1) * b.unwrap_or(1)),
        }
    }

    /// Abandons a half-typed command.
    ///
    /// `pub(crate)` so the editor can clear a pending operator when a
    /// *non-motion* key arrives -- e.g. Ctrl+D, which is a scroll, not a motion,
    /// and must not be allowed to complete a pending `d`.
    pub(crate) fn reset(&mut self) {
        *self = Pending::default();
    }

    /// Feeds one key. See [`FeedResult`] for what comes back.
    pub fn feed(&mut self, key: Key) -> FeedResult {
        // Esc unconditionally cancels an in-progress command, from any
        // state — the one escape hatch every vim user relies on.
        if key.code == KeyCode::Esc {
            let was_idle = self.is_idle();
            self.reset();
            return if was_idle { FeedResult::Continue } else { FeedResult::Invalid };
        }

        match self.state() {
            State::AwaitingRegisterName => self.feed_register_name(key),
            State::AwaitingG => self.feed_g(key),
            State::AwaitingFind(kind) => self.feed_find_char(key, kind),
            State::AwaitingReplaceChar => self.feed_replace_char(key),
            State::AwaitingTextObject(scope) => self.feed_text_object(key, scope),
            State::AwaitingMacroRegister => self.feed_macro_register(key),
            State::AwaitingPlayRegister => self.feed_play_register(key),
            State::AwaitingMarkSet => self.feed_mark_set(key),
            State::AwaitingMarkJump { exact } => self.feed_mark_jump(key, exact),
            State::AwaitingZ => self.feed_z(key),
            State::AwaitingBigZ => self.feed_big_z(key),
            State::AwaitingBracket { forward } => self.feed_bracket(key, forward),
            State::Fresh => self.feed_fresh(key),
        }
    }

    fn count_slot(&mut self) -> &mut Option<usize> {
        if self.operator.is_none() {
            &mut self.count1
        } else {
            &mut self.count2
        }
    }

    fn feed_fresh(&mut self, key: Key) -> FeedResult {
        if let Some(c) = key.as_char() {
            // Digits accumulate into whichever count slot is active — but a
            // leading `0` (no digits yet in this slot) is the `0` motion
            // (start of line), not the start of a count.
            if c.is_ascii_digit() {
                let is_leading_zero = c == '0' && self.count_slot().is_none();
                if !is_leading_zero {
                    let slot = self.count_slot();
                    let digit = c.to_digit(10).unwrap() as usize;
                    *slot = Some(slot.unwrap_or(0) * 10 + digit);
                    return FeedResult::Continue;
                }
            }
        }

        match key.code {
            KeyCode::Char('"') => {
                self.state_ = Some(State::AwaitingRegisterName);
                FeedResult::Continue
            }
            KeyCode::Char('g') => {
                self.state_ = Some(State::AwaitingG);
                FeedResult::Continue
            }
            KeyCode::Char('d') if self.operator == Some(Operator::Delete) => self.complete_lines(Operator::Delete),
            KeyCode::Char('d') => self.set_operator(Operator::Delete),
            KeyCode::Char('c') if self.operator == Some(Operator::Change) => self.complete_lines(Operator::Change),
            KeyCode::Char('c') => self.set_operator(Operator::Change),
            KeyCode::Char('y') if self.operator == Some(Operator::Yank) => self.complete_lines(Operator::Yank),
            KeyCode::Char('y') => self.set_operator(Operator::Yank),
            KeyCode::Char('>') if self.operator == Some(Operator::Indent) => self.complete_lines(Operator::Indent),
            KeyCode::Char('>') => self.set_operator(Operator::Indent),
            KeyCode::Char('<') if self.operator == Some(Operator::Dedent) => self.complete_lines(Operator::Dedent),
            KeyCode::Char('<') => self.set_operator(Operator::Dedent),

            KeyCode::Char('i') | KeyCode::Char('a') if self.operator.is_some() => {
                let scope = if key.code == KeyCode::Char('i') { ObjectScope::Inner } else { ObjectScope::Around };
                self.state_ = Some(State::AwaitingTextObject(scope));
                FeedResult::Continue
            }
            KeyCode::Char('i') if self.operator.is_none() => self.finish(GrammarCommand::EnterInsert(InsertPos::Before)),
            KeyCode::Char('a') if self.operator.is_none() => self.finish(GrammarCommand::EnterInsert(InsertPos::After)),
            KeyCode::Char('I') if self.operator.is_none() => self.finish(GrammarCommand::EnterInsert(InsertPos::LineStart)),
            KeyCode::Char('A') if self.operator.is_none() => self.finish(GrammarCommand::EnterInsert(InsertPos::LineEnd)),
            KeyCode::Char('o') if self.operator.is_none() => self.finish(GrammarCommand::EnterInsert(InsertPos::NewLineBelow)),
            KeyCode::Char('O') if self.operator.is_none() => self.finish(GrammarCommand::EnterInsert(InsertPos::NewLineAbove)),

            // The classic one-key reflexes, each an alias for its operator+
            // motion long form: `D`=`d$`, `C`=`c$`, `S`=`cc`. `Y` is `y$`
            // here, matching **neovim's** default (neovim remapped `Y` to
            // `y$` in 0.6; older vim's `Y`==`yy` linewise). They only fire at
            // the top level — with an operator already pending, an uppercase
            // letter is not one of these.
            KeyCode::Char('D') if self.operator.is_none() => self.complete_operator_motion(Operator::Delete, Motion::LineEnd),
            KeyCode::Char('C') if self.operator.is_none() => self.complete_operator_motion(Operator::Change, Motion::LineEnd),
            KeyCode::Char('Y') if self.operator.is_none() => self.complete_operator_motion(Operator::Yank, Motion::LineEnd),
            KeyCode::Char('S') if self.operator.is_none() => self.complete_lines(Operator::Change),

            // `Z` opens the two-key quit family: `ZZ` (write + quit) / `ZQ`
            // (quit, discard). `<C-6>`/`<C-^>`/`<C-g>`/`<C-]>` are Ctrl-keys
            // caught in `Editor` before the grammar, not here.
            KeyCode::Char('Z') if self.operator.is_none() => {
                self.state_ = Some(State::AwaitingBigZ);
                FeedResult::Continue
            }

            // Bracket `[`/`]` motion prefixes (`[[`, `]}`, `]m`, `` [` ``, ...).
            // Unconditional (no `operator.is_none()` guard) so they compose as
            // motions after an operator too — `d]}`, `y[[`. The one exception
            // is the `['`/`]'`-style mark jumps, which `feed_bracket` resolves
            // to a standalone jump (see there).
            KeyCode::Char('[') => {
                self.state_ = Some(State::AwaitingBracket { forward: false });
                FeedResult::Continue
            }
            KeyCode::Char(']') => {
                self.state_ = Some(State::AwaitingBracket { forward: true });
                FeedResult::Continue
            }

            KeyCode::Char('f') => {
                self.state_ = Some(State::AwaitingFind(FindKind::To));
                FeedResult::Continue
            }
            KeyCode::Char('F') => {
                self.state_ = Some(State::AwaitingFind(FindKind::ToBack));
                FeedResult::Continue
            }
            KeyCode::Char('t') => {
                self.state_ = Some(State::AwaitingFind(FindKind::Till));
                FeedResult::Continue
            }
            KeyCode::Char('T') => {
                self.state_ = Some(State::AwaitingFind(FindKind::TillBack));
                FeedResult::Continue
            }

            // `<C-r>` (redo) and plain `r` (replace) share the same
            // `KeyCode::Char('r')`; the modifier guard has to be checked
            // first or the plain-`r` arm below would shadow it.
            KeyCode::Char('r') if key.mods.ctrl && self.operator.is_none() => self.finish(GrammarCommand::Redo),
            KeyCode::Char('r') if self.operator.is_none() => {
                self.state_ = Some(State::AwaitingReplaceChar);
                FeedResult::Continue
            }
            KeyCode::Char('x') if self.operator.is_none() => self.finish(GrammarCommand::DeleteCharForward { register: self.register, count: self.effective_count() }),
            KeyCode::Char('X') if self.operator.is_none() => self.finish(GrammarCommand::DeleteCharBackward { register: self.register, count: self.effective_count() }),
            KeyCode::Char('s') if self.operator.is_none() => self.finish(GrammarCommand::SubstituteChar { register: self.register, count: self.effective_count() }),
            KeyCode::Char('~') if self.operator.is_none() => self.finish(GrammarCommand::ToggleCaseUnderCursor { count: self.effective_count() }),
            KeyCode::Char('J') if self.operator.is_none() => self.finish(GrammarCommand::JoinLines { count: self.effective_count() }),
            KeyCode::Char('p') if self.operator.is_none() => self.finish(GrammarCommand::Put { register: self.register, count: self.effective_count(), before: false }),
            KeyCode::Char('P') if self.operator.is_none() => self.finish(GrammarCommand::Put { register: self.register, count: self.effective_count(), before: true }),
            KeyCode::Char('u') if self.operator.is_none() => self.finish(GrammarCommand::Undo),
            KeyCode::Char('.') if self.operator.is_none() => self.finish(GrammarCommand::RepeatLast),
            // `<C-v>` (visual-block) and plain `v` (visual) share
            // `KeyCode::Char('v')`; same ordering requirement as `r` above.
            KeyCode::Char('v') if key.mods.ctrl && self.operator.is_none() => self.finish(GrammarCommand::EnterVisual(super::VisualKind::Blockwise)),
            KeyCode::Char('v') if self.operator.is_none() => self.finish(GrammarCommand::EnterVisual(super::VisualKind::Charwise)),
            KeyCode::Char('V') if self.operator.is_none() => self.finish(GrammarCommand::EnterVisual(super::VisualKind::Linewise)),
            KeyCode::Char(':') if self.operator.is_none() => self.finish(GrammarCommand::EnterCommandLine),
            KeyCode::Char('q') if self.operator.is_none() => {
                self.state_ = Some(State::AwaitingMacroRegister);
                FeedResult::Continue
            }
            KeyCode::Char('@') if self.operator.is_none() => {
                self.state_ = Some(State::AwaitingPlayRegister);
                FeedResult::Continue
            }

            // Marks, search, and viewport commands. These are standalone
            // (operator-free) in this grammar: operator-composed marks/search
            // (`d'a`, `d/pat`) need the *editor* to resolve a mark or pattern
            // to a concrete position, which `Pending` — buffer-free by design
            // (see the module docs) — cannot do. Documented scope cut.
            KeyCode::Char('m') if self.operator.is_none() => {
                self.state_ = Some(State::AwaitingMarkSet);
                FeedResult::Continue
            }
            KeyCode::Char('`') if self.operator.is_none() => {
                self.state_ = Some(State::AwaitingMarkJump { exact: true });
                FeedResult::Continue
            }
            KeyCode::Char('\'') if self.operator.is_none() => {
                self.state_ = Some(State::AwaitingMarkJump { exact: false });
                FeedResult::Continue
            }
            KeyCode::Char('/') if self.operator.is_none() => self.finish(GrammarCommand::StartSearch { forward: true }),
            KeyCode::Char('?') if self.operator.is_none() => self.finish(GrammarCommand::StartSearch { forward: false }),
            KeyCode::Char('n') if self.operator.is_none() => self.finish(GrammarCommand::RepeatSearch { reverse: false }),
            KeyCode::Char('N') if self.operator.is_none() => self.finish(GrammarCommand::RepeatSearch { reverse: true }),
            KeyCode::Char('*') if self.operator.is_none() => self.finish(GrammarCommand::SearchWord { forward: true }),
            KeyCode::Char('#') if self.operator.is_none() => self.finish(GrammarCommand::SearchWord { forward: false }),
            // `&`: redo the last `:s` on this line (flags dropped) — vim's
            // shorthand for `:s`.
            KeyCode::Char('&') if self.operator.is_none() => self.finish(GrammarCommand::RepeatSubstitute),
            KeyCode::Char('z') if self.operator.is_none() => {
                self.state_ = Some(State::AwaitingZ);
                FeedResult::Continue
            }
            // `;`/`,` repeat the last `f`/`F`/`t`/`T`. `Pending` has no
            // memory of what that was — only `Editor` does (see
            // `Editor::last_find`) — so this resolves to a distinct command
            // for `Editor` to fill in, rather than trying to fabricate a
            // `Motion::FindChar` here.
            KeyCode::Char(';') => self.complete_repeat_find(false),
            KeyCode::Char(',') => self.complete_repeat_find(true),

            // Bare motions.
            _ => match simple_motion(key) {
                Some(motion) => self.complete_motion(motion),
                None => self.invalid(),
            },
        }
    }

    fn complete_repeat_find(&mut self, reverse: bool) -> FeedResult {
        let register = self.register;
        let count = self.effective_count();
        let operator = self.operator;
        self.reset();
        FeedResult::Complete(GrammarCommand::RepeatFind { register, count, operator, reverse })
    }

    fn set_operator(&mut self, op: Operator) -> FeedResult {
        self.operator = Some(op);
        self.state_ = Some(State::Fresh);
        FeedResult::Continue
    }

    fn complete_lines(&mut self, op: Operator) -> FeedResult {
        let register = self.register;
        let count = self.effective_count();
        self.reset();
        FeedResult::Complete(GrammarCommand::OperatorLines { register, count, operator: op })
    }

    /// Completes an operator+motion pair directly, without the motion having
    /// arrived as a separate key — the shared tail of the one-key `D`/`C`/`Y`
    /// shortcuts (`d$`/`c$`/`y$`). Carries whatever count/register the user
    /// typed, exactly as if they had spelled the long form.
    fn complete_operator_motion(&mut self, operator: Operator, motion: Motion) -> FeedResult {
        let register = self.register;
        let count = self.effective_count();
        self.reset();
        FeedResult::Complete(GrammarCommand::OperatorMotion { register, count, operator, motion })
    }

    fn complete_motion(&mut self, motion: Motion) -> FeedResult {
        let count = self.effective_count();
        let register = self.register;
        let result = match self.operator {
            Some(operator) => GrammarCommand::OperatorMotion { register, count, operator, motion },
            None => GrammarCommand::Move { count, motion },
        };
        self.reset();
        FeedResult::Complete(result)
    }

    fn finish(&mut self, cmd: GrammarCommand) -> FeedResult {
        self.reset();
        FeedResult::Complete(cmd)
    }

    fn invalid(&mut self) -> FeedResult {
        self.reset();
        FeedResult::Invalid
    }

    fn feed_register_name(&mut self, key: Key) -> FeedResult {
        let Some(c) = key.as_char() else { return self.invalid() };
        self.register = Some(c);
        self.state_ = Some(State::Fresh);
        FeedResult::Continue
    }

    fn feed_g(&mut self, key: Key) -> FeedResult {
        match key.code {
            KeyCode::Char('g') => self.complete_motion(Motion::FileStart),
            KeyCode::Char('e') => self.complete_motion(Motion::WordEndBack),
            KeyCode::Char('E') => self.complete_motion(Motion::WordEndBackBig),
            KeyCode::Char('_') => self.complete_motion(Motion::LastNonBlank),
            // `gj`/`gk` are display-line motions; with `wrap=false` (the
            // maintainer's setting — see `ui::textarea`) a display line is a
            // buffer line, so they are exactly `j`/`k`.
            KeyCode::Char('j') => self.complete_motion(Motion::Down),
            KeyCode::Char('k') => self.complete_motion(Motion::Up),
            KeyCode::Char('v') if self.operator.is_none() => self.finish(GrammarCommand::ReselectVisual),
            KeyCode::Char('u') if self.operator.is_none() => self.set_operator(Operator::LowerCase),
            KeyCode::Char('U') if self.operator.is_none() => self.set_operator(Operator::UpperCase),
            KeyCode::Char('~') if self.operator.is_none() => self.set_operator(Operator::ToggleCase),
            _ => self.invalid(),
        }
    }

    fn feed_mark_set(&mut self, key: Key) -> FeedResult {
        let Some(name) = key.as_char() else { return self.invalid() };
        self.reset();
        FeedResult::Complete(GrammarCommand::SetMark { name })
    }

    fn feed_mark_jump(&mut self, key: Key, exact: bool) -> FeedResult {
        let Some(name) = key.as_char() else { return self.invalid() };
        self.reset();
        FeedResult::Complete(GrammarCommand::JumpMark { name, exact })
    }

    fn feed_z(&mut self, key: Key) -> FeedResult {
        use crate::core::ViewportScroll;
        let req = match key.code {
            KeyCode::Char('z') => ViewportScroll::CenterCursor,
            KeyCode::Char('t') => ViewportScroll::CursorToTop,
            KeyCode::Char('b') => ViewportScroll::CursorToBottom,
            _ => return self.invalid(),
        };
        self.reset();
        FeedResult::Complete(GrammarCommand::Scroll(req))
    }

    fn feed_big_z(&mut self, key: Key) -> FeedResult {
        match key.code {
            KeyCode::Char('Z') => self.finish(GrammarCommand::WriteQuit),
            KeyCode::Char('Q') => self.finish(GrammarCommand::QuitForce),
            _ => self.invalid(),
        }
    }

    /// The second key of a bracket `[`/`]` motion. `forward` is `true` for a
    /// leading `]`, `false` for `[`; the second character then selects which
    /// motion. `<CR>` and non-character keys are invalid here.
    fn feed_bracket(&mut self, key: Key, forward: bool) -> FeedResult {
        let Some(c) = key.as_char() else { return self.invalid() };
        // The mark jumps (`['`/`` [` ``/`]'`/`` ]` ``) resolve to a standalone
        // jump command rather than a `Motion`, because the mark table lives in
        // the buffer and `Pending` cannot read it (see the module docs).
        if c == '\'' || c == '`' {
            let exact = c == '`';
            self.reset();
            return FeedResult::Complete(GrammarCommand::JumpBracketMark { forward, exact });
        }
        let motion = match (forward, c) {
            (false, '[') => Motion::SectionBackward,
            (true, ']') => Motion::SectionForward,
            (false, ']') => Motion::SectionEndBackward,
            (true, '[') => Motion::SectionEndForward,
            (false, '(') => Motion::UnmatchedParenBack,
            (true, ')') => Motion::UnmatchedParenForward,
            (false, '{') => Motion::UnmatchedBraceBack,
            (true, '}') => Motion::UnmatchedBraceForward,
            (false, 'm') => Motion::MethodStartBack,
            (true, 'm') => Motion::MethodStartForward,
            (false, 'M') => Motion::MethodEndBack,
            (true, 'M') => Motion::MethodEndForward,
            _ => return self.invalid(),
        };
        self.complete_motion(motion)
    }

    fn feed_find_char(&mut self, key: Key, kind: FindKind) -> FeedResult {
        let Some(c) = key.as_char() else { return self.invalid() };
        self.complete_motion(Motion::FindChar { kind, target: c })
    }

    fn feed_replace_char(&mut self, key: Key) -> FeedResult {
        let count = self.effective_count();
        let ch = match key.code {
            KeyCode::Enter => '\n',
            _ => match key.as_char() {
                Some(c) => c,
                None => return self.invalid(),
            },
        };
        self.reset();
        FeedResult::Complete(GrammarCommand::ReplaceChar { count, ch })
    }

    fn feed_text_object(&mut self, key: Key, scope: ObjectScope) -> FeedResult {
        let Some(object) = text_object_for(key) else { return self.invalid() };
        let register = self.register;
        let operator = self.operator.expect("AwaitingTextObject only reachable with an operator set");
        self.reset();
        FeedResult::Complete(GrammarCommand::OperatorTextObject { register, operator, scope, object })
    }

    fn feed_macro_register(&mut self, key: Key) -> FeedResult {
        let Some(c) = key.as_char() else { return self.invalid() };
        self.reset();
        FeedResult::Complete(GrammarCommand::StartRecording { register: c })
    }

    fn feed_play_register(&mut self, key: Key) -> FeedResult {
        let count = self.effective_count();
        let result = match key.code {
            KeyCode::Char('@') => GrammarCommand::ReplayLastMacro { count },
            _ => match key.as_char() {
                Some(c) => GrammarCommand::PlayMacro { register: c, count },
                None => return self.invalid(),
            },
        };
        self.reset();
        FeedResult::Complete(result)
    }
}

/// The single-key motions — everything except the multi-key `g`-prefixed
/// and `f`/`F`/`t`/`T` families, which need extra state and are handled
/// directly in [`Pending::feed_fresh`]/[`Pending::feed_g`].
///
/// `pub(crate)` so [`super::Editor`]'s visual-mode handling (a deliberately
/// separate, simpler grammar — see that module's docs) can recognize the
/// same keys without duplicating this table.
pub(crate) fn simple_motion(key: Key) -> Option<Motion> {
    // Arrow keys and Home/End are the named-key equivalents of `h`/`l`/`k`/`j`
    // and `0`/`$`; recognising them here means they compose with operators and
    // counts (`d<Down>`, `3<Right>`) exactly like their letter twins, and work
    // in every mode that goes through this table (Normal, Visual, and — since
    // `Pending` drives operator-pending — after an operator too).
    match key.code {
        KeyCode::Left => return Some(Motion::Left),
        KeyCode::Right => return Some(Motion::Right),
        KeyCode::Up => return Some(Motion::Up),
        KeyCode::Down => return Some(Motion::Down),
        KeyCode::Home => return Some(Motion::LineStart),
        KeyCode::End => return Some(Motion::LineEnd),
        // `<CR>` in Normal mode is the same motion as `+`: down to the first
        // non-blank of the next line. Handling it here means `d<CR>` deletes
        // two lines linewise, exactly like `d+`.
        KeyCode::Enter => return Some(Motion::NextLineFirstNonBlank),
        _ => {}
    }
    let c = key.as_char()?;
    Some(match c {
        'h' => Motion::Left,
        'l' => Motion::Right,
        'k' => Motion::Up,
        'j' => Motion::Down,
        'w' => Motion::WordForward,
        'W' => Motion::WordForwardBig,
        'b' => Motion::WordBackward,
        'B' => Motion::WordBackwardBig,
        'e' => Motion::WordEnd,
        'E' => Motion::WordEndBig,
        '0' => Motion::LineStart,
        '^' => Motion::FirstNonBlank,
        '$' => Motion::LineEnd,
        '|' => Motion::ToColumn,
        '+' => Motion::NextLineFirstNonBlank,
        '-' => Motion::PrevLineFirstNonBlank,
        '_' => Motion::LineDownFirstNonBlank,
        '{' => Motion::ParagraphBackward,
        '}' => Motion::ParagraphForward,
        '(' => Motion::SentenceBackward,
        ')' => Motion::SentenceForward,
        '%' => Motion::MatchPair,
        'H' => Motion::ScreenHigh,
        'M' => Motion::ScreenMid,
        'L' => Motion::ScreenLow,
        'G' => Motion::FileEnd,
        _ => return None,
    })
}

/// Maps the object-designating key after `i`/`a` to a [`TextObject`],
/// including vim's usual delimiter aliases (`)`/`b` for parens, `}`/`B` for
/// braces, `]` for brackets, `>` for angle brackets).
pub(crate) fn text_object_for(key: Key) -> Option<TextObject> {
    let c = key.as_char()?;
    Some(match c {
        'w' => TextObject::Word,
        'W' => TextObject::BigWord,
        '(' | ')' | 'b' => TextObject::Paren,
        '{' | '}' | 'B' => TextObject::Brace,
        '[' | ']' => TextObject::Bracket,
        '<' | '>' => TextObject::Angle,
        '"' => TextObject::DoubleQuote,
        '\'' => TextObject::SingleQuote,
        '`' => TextObject::Backtick,
        't' => TextObject::Tag,
        'p' => TextObject::Paragraph,
        _ => return None,
    })
}
