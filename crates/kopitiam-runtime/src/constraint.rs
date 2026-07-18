//! Grammar-constrained decoding: masking away tokens the model is *not*
//! allowed to emit next, **before** the sampler ever sees the logits.
//!
//! # Why this module exists (the keystone, `temp_ai_design.md` §10.1 #2)
//!
//! A 0.5B local model cannot reliably drive a model-driven agentic tool-loop
//! — it fumbles JSON, invents tool names that don't exist, emits malformed
//! paths. The fix is not "hope it does better lah". The fix is to make the
//! invalid output **physically unreachable**: at every generation step we
//! compute which next tokens keep the output valid (a grammar / JSON schema /
//! an allowed-tool-name set says which), and we set every *other* token's
//! logit to [`f32::NEG_INFINITY`]. After softmax a `-inf` logit has exactly
//! zero probability, so the sampler — greedy or stochastic — simply *cannot*
//! pick it. This turns "fumbles structure sometimes" into "cannot produce
//! anything but valid structure". It is cheap: just a mask over the vocab.
//! This one feature is what makes the small model tool-capable.
//!
//! Provenance: **AID-0045** ("Grammar-constrained decoding — logit masking
//! makes a small model reliably structured").
//!
//! # The two contracts that must never be got wrong
//!
//! **1. Mask BEFORE sampling, not after.** The mask slots into the *front* of
//! the sampling path — before temperature, before top-k, before top-p (see
//! [`crate::sampling::StochasticSampler`]'s pipeline). Masking *after* a token
//! has already been sampled is too late: the invalid token is already chosen,
//! and now you can only reject-and-retry, which is slow and can loop forever
//! on a stubborn model. Masking first means the invalid token never competes
//! in the first place.
//!
//! **2. Disallowed logit -> `-inf`, NOT `0.0`.** A logit is a pre-softmax
//! score, not a probability. Setting it to `0.0` would leave the token a
//! perfectly ordinary, very-much-samplable score (`exp(0) = 1`). Only
//! `-inf` survives every later transform in the pipeline: temperature
//! *divides* the logit (`-inf / t == -inf` for any `t > 0`), top-k/top-p
//! *compare* logits (`-inf` always ranks last), and softmax maps it to
//! `exp(-inf) == 0.0`. So `-inf` is the one value that guarantees the token
//! stays dead no matter what the rest of the sampler does to it. This is the
//! exact same "excluded" marker [`crate::sampling`] already uses for top-k /
//! top-p / min-p, so a constraint mask composes with those stages for free.
//!
//! # What is here, and what is deliberately a later bead
//!
//! Two tractable constraints ship now:
//!
//! * [`AllowedTokens`] — a **fixed allowed-token-set** (e.g. the ids that spell
//!   out a tool-name enum). State-independent: the same set every step.
//! * [`JsonStructure`] — a **simple structural JSON** constraint (balanced
//!   `{}`/`[]`, quoted strings, `:`/`,` only where the grammar allows a
//!   value/key/separator). It is *structural*, not fully lexical — see its own
//!   docs for exactly which subset it enforces and which it leaves to a later
//!   full-CFG bead.
//!
//! The [`TokenConstraint`] trait is the seam both plug into, so a future
//! full context-free-grammar / JSON-schema constraint drops in without
//! touching the masking core or the sampler.

use std::collections::BTreeSet;

use crate::sampling::Sampler;

/// The decode state a [`TokenConstraint`] reasons about when it decides which
/// next tokens are valid.
///
/// Deliberately a *pure function of the tokens generated so far* — no
/// wall-clock, no hidden mutable cursor. That is the same "context = f(state),
/// never f(wall-clock)" discipline the runtime holds everywhere (see
/// `temp_ai_design.md` §4): given the same generated prefix, a constraint
/// must always return the same [`AllowedSet`], so a constrained run is
/// reproducible and testable.
///
/// `generated` is the tokens produced *this decode* (the completion), not the
/// prompt — a structural constraint cares about the shape of the output it is
/// steering, and the prompt is upstream of that.
pub struct DecodeState<'a> {
    /// Token ids emitted so far in this completion, oldest first.
    pub generated: &'a [u32],
}

/// Which next-token ids a constraint permits at the current step.
///
/// Two shapes, because two very different constraints want very different
/// storage:
///
/// * [`AllowedSet::Only`] — a **sparse** set: "only these handful of ids, mask
///   everything else". Right for a fixed tool-name enum over a 150k vocab,
///   where listing the allowed ids is far cheaper than a 150k-entry bool
///   vector.
/// * [`AllowedSet::Mask`] — a **dense** allow-mask, one `bool` per vocab id.
///   Right for a structural constraint that has to test every candidate token
///   anyway (see [`JsonStructure::allowed`]), so the per-id answer is already
///   computed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllowedSet {
    /// Only these ids are allowed; every other id is masked.
    Only(BTreeSet<u32>),
    /// One `bool` per vocab id: `mask[id]` allowed iff `true`. An id past the
    /// end of the vector is treated as not allowed.
    Mask(Vec<bool>),
}

impl AllowedSet {
    /// Is token `id` allowed by this set?
    ///
    /// For [`AllowedSet::Mask`], an `id` at or past the mask length is **not**
    /// allowed — a mask sized to a smaller vocab than the logits row can never
    /// accidentally green-light a token it never considered.
    pub fn contains(&self, id: u32) -> bool {
        match self {
            AllowedSet::Only(set) => set.contains(&id),
            AllowedSet::Mask(mask) => mask.get(id as usize).copied().unwrap_or(false),
        }
    }
}

/// What went wrong when a constraint could not be satisfied.
///
/// Kept as a small crate-local enum (hand-rolled `Display`/`Error`, no
/// `thiserror` dependency — one two-variant enum does not earn a proc-macro
/// crate, per the workspace's "avoid unnecessary dependencies" rule) rather
/// than folded into [`kopitiam_core::Error`], because "the constraint left no
/// valid token" is a decoding-policy fact, not a tensor/model fact, and this
/// crate is the only place it can arise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintError {
    /// After masking, **no** in-range token survived: the constraint forbade
    /// every token the logits row actually has. This is an honest, catchable
    /// error, never a panic — a caller can surface it, widen the constraint,
    /// or stop. It usually means a bug in the constraint (it should always
    /// leave *some* escape hatch, e.g. an EOS/whitespace token) rather than a
    /// bug in the model.
    NoTokenAllowed,
    /// A constraint was constructed with an empty allowed set — rejected at
    /// construction so a vacuous constraint can never silently mask an entire
    /// vocab at decode time.
    EmptyConstraint,
}

impl std::fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstraintError::NoTokenAllowed => {
                write!(f, "constraint masked every token: no valid next token to sample")
            }
            ConstraintError::EmptyConstraint => {
                write!(f, "constraint constructed with an empty allowed-token set")
            }
        }
    }
}

impl std::error::Error for ConstraintError {}

/// A rule that says which next tokens are valid, given what's been decoded so
/// far.
///
/// The whole grammar-constrained-decoding feature hangs off this one method.
/// Implementors range from the dead-simple ([`AllowedTokens`] ignores the
/// state and returns a fixed set) to the stateful ([`JsonStructure`] replays
/// the generated bytes to work out where in the JSON grammar it is). A future
/// full-CFG or JSON-schema constraint is just another `impl` — the masking
/// core ([`mask_logits`]) and the sampler wrapper ([`ConstrainedSampler`])
/// never need to know which one they're driving.
///
/// # Contract
///
/// `allowed` must be a **pure function of `state`**: same generated prefix ->
/// same [`AllowedSet`]. No interior mutability that changes the answer across
/// identical calls, or reproducibility (and the tests that pin it) break.
pub trait TokenConstraint {
    /// The set of token ids that keep the output valid if emitted next.
    fn allowed(&self, state: &DecodeState<'_>) -> AllowedSet;
}

/// The masking step itself: set every logit whose id is **not** in `allowed`
/// to [`f32::NEG_INFINITY`], in place.
///
/// This is the entire keystone in one function. It is deliberately its own
/// standalone, fallible unit so it can be tested to death independently of any
/// sampler or model, and so the "no valid token" case is an honest
/// [`ConstraintError`] a caller must handle — never a panic, never a silently
/// wrong token.
///
/// # Ordering contract (load-bearing — do not move this call)
///
/// Call this on the **raw** logits row, *before* repetition penalty /
/// temperature / top-k / top-p / min-p. See this module's docs: `-inf`
/// survives all of those, so masking first and sampling second is what makes
/// an invalid token unreachable. [`ConstrainedSampler::try_sample`] holds this
/// ordering for you.
///
/// # Errors
///
/// [`ConstraintError::NoTokenAllowed`] if, after masking, **no** in-range
/// token is allowed (the constraint forbade every id the row has). In that
/// case the mask is left applied but there is nothing valid to sample, so the
/// caller must decide what to do — this function will not guess.
pub fn mask_logits(logits: &mut [f32], allowed: &AllowedSet) -> Result<(), ConstraintError> {
    let mut any_allowed = false;
    for (idx, logit) in logits.iter_mut().enumerate() {
        if allowed.contains(idx as u32) {
            any_allowed = true;
        } else {
            *logit = f32::NEG_INFINITY;
        }
    }
    if any_allowed {
        Ok(())
    } else {
        Err(ConstraintError::NoTokenAllowed)
    }
}

/// A **fixed allowed-token-set** constraint: the same ids every step,
/// regardless of decode state.
///
/// The bread-and-butter case for the tool-loop — e.g. "the next token must be
/// one of the ids that spell a valid tool name from the enum", or "pick one of
/// these N candidate-file ids". Because the set never changes, it is a
/// `BTreeSet<u32>` fixed at construction and [`AllowedTokens::allowed`] just
/// hands it back.
///
/// The clone-per-step in [`AllowedTokens::allowed`] is intentional and cheap:
/// a tool-name set is tens of ids, and cloning a small `BTreeSet` once per
/// generated token is nothing next to a whole transformer forward pass. If a
/// future caller pins a *large* fixed set, switch its storage to a shared
/// `Arc<BTreeSet<u32>>` — the [`TokenConstraint`] contract does not change.
#[derive(Debug, Clone)]
pub struct AllowedTokens {
    ids: BTreeSet<u32>,
}

impl AllowedTokens {
    /// Build a fixed-set constraint from an iterator of allowed ids.
    ///
    /// # Errors
    ///
    /// [`ConstraintError::EmptyConstraint`] if `ids` is empty — a constraint
    /// that allows nothing would mask the entire vocab at every step and can
    /// only ever produce [`ConstraintError::NoTokenAllowed`]. Rejecting it here
    /// turns a guaranteed-later failure into an obvious construction-time one.
    pub fn new(ids: impl IntoIterator<Item = u32>) -> Result<Self, ConstraintError> {
        let ids: BTreeSet<u32> = ids.into_iter().collect();
        if ids.is_empty() {
            return Err(ConstraintError::EmptyConstraint);
        }
        Ok(Self { ids })
    }
}

impl TokenConstraint for AllowedTokens {
    fn allowed(&self, _state: &DecodeState<'_>) -> AllowedSet {
        AllowedSet::Only(self.ids.clone())
    }
}

/// A token id -> its byte string, for constraints that must reason about the
/// *bytes* a candidate token would append (not just its id).
///
/// [`JsonStructure`] needs this: whether emitting token 512 keeps the JSON
/// valid depends entirely on what characters token 512 *is* (`":"` vs `,` vs
/// `abc`), which the id alone does not tell you. A real integration hands a
/// view over the model's tokenizer vocabulary here; tests use [`SliceVocab`].
pub trait TokenVocab {
    /// How many token ids exist (`0..token_count()` is the id range).
    fn token_count(&self) -> usize;
    /// The bytes token `id` decodes to, or `None` for an out-of-range /
    /// byte-less control id (e.g. EOS). A `None` token contributes no bytes to
    /// the grammar and is never structurally allowed by [`JsonStructure`] on
    /// its own — permit it explicitly via
    /// [`JsonStructure::with_always_allowed`] if it should be samplable (EOS
    /// usually should, so the model can actually stop).
    fn token_bytes(&self, id: u32) -> Option<&[u8]>;
}

/// The obvious [`TokenVocab`] over an owned `id -> bytes` table. Handy for
/// tests and for any caller that already has the vocabulary as a `Vec`.
#[derive(Debug, Clone)]
pub struct SliceVocab {
    tokens: Vec<Vec<u8>>,
}

impl SliceVocab {
    /// Wrap a `Vec` where index `i` is the byte string of token id `i`.
    pub fn new(tokens: Vec<Vec<u8>>) -> Self {
        Self { tokens }
    }
}

impl TokenVocab for SliceVocab {
    fn token_count(&self) -> usize {
        self.tokens.len()
    }
    fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.tokens.get(id as usize).map(Vec::as_slice)
    }
}

/// Which container we are currently nested inside, for [`JsonMachine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Container {
    Object,
    Array,
}

/// Where the byte-level JSON grammar currently sits — "what may legally come
/// next". See [`JsonMachine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pos {
    /// A value must begin here (document start, after `:`, after `,` in an
    /// array). Legal starters: `{ [ "` or a scalar char.
    ValueStart,
    /// Right after `[`: a value may begin, or the array may close with `]`.
    ArrayStart,
    /// Right after `{`: a `"`-key may begin, or the object may close with `}`.
    ObjectStart,
    /// After `,` inside an object: a `"`-key must begin (no trailing-comma
    /// close).
    KeyStart,
    /// A key string just closed: only `:` may come next.
    Colon,
    /// A value just completed inside a container: `,` or the matching close
    /// bracket.
    AfterValue,
    /// Inside a `"`-string. `escape` tracks a pending backslash; `is_key`
    /// tracks whether closing it lands on [`Pos::Colon`] (a key) or completes
    /// a value.
    InString { escape: bool, is_key: bool },
    /// Inside a scalar run (number / `true` / `false` / `null`). Ends at the
    /// first byte that is not a scalar-continuation char, which is then
    /// re-processed.
    InScalar,
    /// A top-level value completed: the document is done. Only whitespace may
    /// follow.
    Done,
}

/// A streaming, byte-level *structural* JSON checker.
///
/// # What it enforces (and what it does NOT)
///
/// This is the **structural** subset, on purpose (a full JSON CFG is a later
/// bead — see this module's docs):
///
/// * Enforced: balanced `{}` / `[]` nesting, strings opened and closed with
///   `"` (with `\`-escape handling), `:` only between an object key and its
///   value, `,` only between elements, a value expected after `:` / `[` / `,`,
///   no trailing comma, no stray close bracket, nothing but whitespace after a
///   complete top-level value.
/// * **Not** enforced (left to the future full-CFG bead): that a number is
///   *well-formed* (`1.2.3` and `--0` pass — any run of `[A-Za-z0-9.+-]` is
///   accepted as "a scalar"), that a literal is spelled exactly `true` /
///   `false` / `null` (any letter-run passes), and full `\uXXXX` escape-digit
///   validation (the four hex digits are not checked). It guarantees the
///   *shape* is valid JSON, not that every scalar *token* is.
///
/// That subset is exactly what makes a small model's tool-call output
/// *parseable* — brace/quote/separator discipline is where a 0.5B fumbles, and
/// it is what this cheap machine fixes.
#[derive(Debug, Clone)]
struct JsonMachine {
    stack: Vec<Container>,
    pos: Pos,
}

impl JsonMachine {
    fn new() -> Self {
        Self { stack: Vec::new(), pos: Pos::ValueStart }
    }

    /// Feed one byte. Returns `false` if the byte is illegal in the current
    /// state (the machine must not be used further after a `false`).
    fn step(&mut self, b: u8) -> bool {
        loop {
            match self.pos {
                Pos::InString { .. } => return self.string_step(b),
                Pos::InScalar => {
                    if is_scalar_cont(b) {
                        return true; // scalar keeps going, byte consumed
                    }
                    // The scalar ends *before* this byte. Complete the value,
                    // then re-process `b` in the resulting position (it might
                    // be a `,`, a close bracket, or whitespace).
                    self.complete_value();
                    continue;
                }
                _ => return self.consume(b),
            }
        }
    }

    /// Advance a structural (non-string, non-scalar) position by one byte.
    fn consume(&mut self, b: u8) -> bool {
        if is_ws(b) {
            // Whitespace is legal in every structural position, including
            // `Done`. It never changes the position.
            return true;
        }
        match self.pos {
            Pos::ValueStart => self.begin_value(b),
            Pos::ArrayStart => {
                if b == b']' {
                    self.close(Container::Array)
                } else {
                    self.begin_value(b)
                }
            }
            Pos::ObjectStart => match b {
                b'"' => {
                    self.pos = Pos::InString { escape: false, is_key: true };
                    true
                }
                b'}' => self.close(Container::Object),
                _ => false,
            },
            Pos::KeyStart => match b {
                b'"' => {
                    self.pos = Pos::InString { escape: false, is_key: true };
                    true
                }
                _ => false,
            },
            Pos::Colon => {
                if b == b':' {
                    self.pos = Pos::ValueStart;
                    true
                } else {
                    false
                }
            }
            Pos::AfterValue => match b {
                b',' => match self.stack.last() {
                    Some(Container::Object) => {
                        self.pos = Pos::KeyStart;
                        true
                    }
                    Some(Container::Array) => {
                        self.pos = Pos::ValueStart;
                        true
                    }
                    None => false, // a comma at top level is not valid
                },
                b'}' => self.close(Container::Object),
                b']' => self.close(Container::Array),
                _ => false,
            },
            Pos::Done => false, // only whitespace (handled above) may follow
            Pos::InString { .. } | Pos::InScalar => unreachable!("handled in step()"),
        }
    }

    /// Begin a value at `b` from a value-expected position.
    fn begin_value(&mut self, b: u8) -> bool {
        match b {
            b'{' => {
                self.stack.push(Container::Object);
                self.pos = Pos::ObjectStart;
                true
            }
            b'[' => {
                self.stack.push(Container::Array);
                self.pos = Pos::ArrayStart;
                true
            }
            b'"' => {
                self.pos = Pos::InString { escape: false, is_key: false };
                true
            }
            _ if is_scalar_start(b) => {
                self.pos = Pos::InScalar;
                true
            }
            _ => false,
        }
    }

    /// Close the current container, which must match `want`, then treat the
    /// closed container as a completed value.
    fn close(&mut self, want: Container) -> bool {
        match self.stack.last() {
            Some(&top) if top == want => {
                self.stack.pop();
                self.complete_value();
                true
            }
            _ => false,
        }
    }

    /// A value just finished (scalar ended, string value closed, or container
    /// closed): decide where that leaves us.
    fn complete_value(&mut self) {
        self.pos = if self.stack.is_empty() { Pos::Done } else { Pos::AfterValue };
    }

    /// Advance while inside a string.
    fn string_step(&mut self, b: u8) -> bool {
        let Pos::InString { escape, is_key } = self.pos else {
            unreachable!("string_step called outside a string");
        };
        if escape {
            // Any byte after a backslash is a consumed escape payload. We do
            // not validate `\uXXXX` hex digits here (documented as a
            // later-bead gap on JsonMachine).
            self.pos = Pos::InString { escape: false, is_key };
            return true;
        }
        match b {
            b'\\' => {
                self.pos = Pos::InString { escape: true, is_key };
                true
            }
            b'"' => {
                if is_key {
                    self.pos = Pos::Colon;
                } else {
                    self.complete_value();
                }
                true
            }
            // A raw control byte inside a string is invalid JSON; blocking it
            // keeps the constraint honest about string content.
            b'\n' | b'\r' => false,
            _ => true, // ordinary string byte
        }
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

fn is_scalar_start(b: u8) -> bool {
    b == b'-' || b.is_ascii_digit() || b.is_ascii_alphabetic()
}

fn is_scalar_cont(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'+')
}

/// A [`TokenConstraint`] that keeps the model's output structurally valid JSON.
///
/// See [`JsonMachine`] for exactly which subset is enforced. At each step
/// [`JsonStructure::allowed`] replays the already-generated bytes to find the
/// current grammar position, then tests every candidate token: a token is
/// allowed iff feeding its bytes from the current position never hits an
/// illegal transition.
///
/// # `always_allowed` — the escape hatch you almost always need
///
/// A JSON document, once complete, structurally permits *only* whitespace
/// next. If the model can never emit EOS, it can never stop. So control ids
/// (EOS above all) should be registered via
/// [`JsonStructure::with_always_allowed`]: they bypass the structural check and
/// are always permitted, which is also what stops the mask from going empty at
/// `Done` and raising [`ConstraintError::NoTokenAllowed`].
///
/// # Cost
///
/// `allowed` is `O(vocab_size × token_len)` per step (it probes every
/// candidate token). Fine for the scaffold and for the small vocabularies the
/// tool-loop actually constrains; a later bead can make the machine
/// *incremental* (advance by the one chosen token instead of re-probing the
/// whole vocab) if a large-vocab JSON constraint ever needs it. The pure
/// `allowed(&self, state)` shape is kept deliberately so the answer stays a
/// function of state (reproducible), incremental caching being an internal
/// optimisation that must not change it.
pub struct JsonStructure<V: TokenVocab> {
    vocab: V,
    always_allowed: BTreeSet<u32>,
}

impl<V: TokenVocab> JsonStructure<V> {
    /// A JSON-structural constraint over `vocab`, with no always-allowed
    /// control ids. Prefer [`JsonStructure::with_always_allowed`] so the model
    /// has a way to stop.
    pub fn new(vocab: V) -> Self {
        Self { vocab, always_allowed: BTreeSet::new() }
    }

    /// As [`JsonStructure::new`], but `ids` (typically just EOS) always pass
    /// the mask regardless of grammar position. See the type's docs for why
    /// this is almost always needed.
    pub fn with_always_allowed(vocab: V, ids: impl IntoIterator<Item = u32>) -> Self {
        Self { vocab, always_allowed: ids.into_iter().collect() }
    }

    /// Replay `generated` to reconstruct the grammar position. Prior tokens are
    /// assumed to have been valid (they were only sampled because a previous
    /// mask allowed them), so a `false` transition during replay cannot happen
    /// in a real constrained run; we ignore its return here rather than panic,
    /// which keeps `allowed` total even if a caller feeds a hand-built
    /// `generated` that never went through the mask.
    fn replay(&self, generated: &[u32]) -> JsonMachine {
        let mut machine = JsonMachine::new();
        for &id in generated {
            if let Some(bytes) = self.vocab.token_bytes(id) {
                for &b in bytes {
                    machine.step(b);
                }
            }
        }
        machine
    }
}

impl<V: TokenVocab> TokenConstraint for JsonStructure<V> {
    fn allowed(&self, state: &DecodeState<'_>) -> AllowedSet {
        let machine = self.replay(state.generated);
        let count = self.vocab.token_count();
        let mut mask = vec![false; count];
        for id in 0..count as u32 {
            if self.always_allowed.contains(&id) {
                mask[id as usize] = true;
                continue;
            }
            let Some(bytes) = self.vocab.token_bytes(id) else { continue };
            if bytes.is_empty() {
                continue; // an empty token advances the grammar nowhere; not on its own useful
            }
            let mut probe = machine.clone();
            let mut ok = true;
            for &b in bytes {
                if !probe.step(b) {
                    ok = false;
                    break;
                }
            }
            mask[id as usize] = ok;
        }
        AllowedSet::Mask(mask)
    }
}

/// Wraps any [`Sampler`] so the constraint's mask is applied at the **front**
/// of the sampling path — the drop-in integration seam for real decoding.
///
/// # How it plugs in
///
/// [`ConstrainedSampler::try_sample`] does, in order: read the generated-so-far
/// prefix -> ask the [`TokenConstraint`] for the [`AllowedSet`] -> [`mask_logits`]
/// the raw logits (`-inf` on disallowed) -> hand the masked row to the inner
/// [`Sampler`] (temperature/top-k/top-p/... all run *after* the mask, exactly
/// as required) -> record the chosen token so the next step's [`DecodeState`]
/// is correct. Because masking happens before the inner sampler, the sampler
/// can only ever pick an allowed token — greedy's `argmax` skips `-inf`, and
/// stochastic softmax gives `-inf` zero probability.
///
/// # Why `try_sample` is fallible instead of an infallible `Sampler` impl
///
/// [`Sampler::sample`] returns a bare `u32` — it cannot report "the constraint
/// left nothing to sample". That case ([`ConstraintError::NoTokenAllowed`])
/// must be an honest error, not a panic and not a silently-wrong token, so the
/// constrained path is deliberately its own fallible method rather than an
/// `impl Sampler` that would have to swallow the error. Callers drive it
/// through [`crate::generate::generate_constrained`], which threads the error
/// out cleanly.
pub struct ConstrainedSampler<C: TokenConstraint, S: Sampler> {
    constraint: C,
    inner: S,
    generated: Vec<u32>,
}

impl<C: TokenConstraint, S: Sampler> ConstrainedSampler<C, S> {
    /// Wrap `inner` so `constraint` masks its logits every step.
    pub fn new(constraint: C, inner: S) -> Self {
        Self { constraint, inner, generated: Vec::new() }
    }

    /// The tokens this sampler has produced so far this decode.
    pub fn generated(&self) -> &[u32] {
        &self.generated
    }

    /// Forget the generated history so the same sampler can drive a fresh
    /// decode (the constraint restarts from an empty prefix).
    pub fn reset(&mut self) {
        self.generated.clear();
    }

    /// Mask, then sample — the constrained step. See the type's docs for the
    /// exact ordering it guarantees.
    ///
    /// # Errors
    ///
    /// [`ConstraintError::NoTokenAllowed`] if the constraint masked every
    /// in-range token this step (see [`mask_logits`]). The generated history is
    /// left unchanged on error, so a caller may retry with a widened
    /// constraint without corrupting the decode state.
    pub fn try_sample(&mut self, logits: &[f32]) -> Result<u32, ConstraintError> {
        let allowed = {
            let state = DecodeState { generated: &self.generated };
            self.constraint.allowed(&state)
        };
        let mut work = logits.to_vec();
        mask_logits(&mut work, &allowed)?;
        let chosen = self.inner.sample(&work);
        // Defence in depth: masking guarantees the inner sampler cannot return
        // a disallowed token (greedy skips `-inf`, stochastic gives it zero
        // probability). If this ever fires, the invariant in this module's docs
        // has been broken and the whole feature is unsound.
        debug_assert!(
            allowed.contains(chosen),
            "inner sampler returned masked-out token {chosen}; mask-before-sample invariant broken"
        );
        self.generated.push(chosen);
        Ok(chosen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{greedy_argmax, GreedySampler, SamplingConfig, StochasticSampler};

    // -- AllowedSet --

    #[test]
    fn allowed_set_only_contains_listed_ids() {
        let set = AllowedSet::Only([1, 3, 5].into_iter().collect());
        assert!(set.contains(3));
        assert!(!set.contains(2));
    }

    #[test]
    fn allowed_set_mask_treats_out_of_range_ids_as_not_allowed() {
        let set = AllowedSet::Mask(vec![true, false, true]);
        assert!(set.contains(0));
        assert!(!set.contains(1));
        assert!(set.contains(2));
        assert!(!set.contains(9), "an id past the mask length must not be allowed");
    }

    // -- mask_logits: the keystone contract --

    #[test]
    fn mask_forces_only_allowed_ids_to_survive_as_neg_inf() {
        // Vocab of 5; allow only ids 1 and 4.
        let allowed = AllowedSet::Only([1, 4].into_iter().collect());
        let mut logits = vec![10.0, 2.0, 9.0, 8.0, 1.0];
        mask_logits(&mut logits, &allowed).unwrap();
        assert!(logits[0].is_infinite() && logits[0] < 0.0, "disallowed id 0 must be -inf");
        assert_eq!(logits[1], 2.0, "allowed id 1 must be untouched");
        assert!(logits[2].is_infinite() && logits[2] < 0.0, "disallowed id 2 must be -inf");
        assert!(logits[3].is_infinite() && logits[3] < 0.0, "disallowed id 3 must be -inf");
        assert_eq!(logits[4], 1.0, "allowed id 4 must be untouched");
    }

    #[test]
    fn masked_argmax_is_always_in_the_allowed_set() {
        // Id 0 has the highest RAW logit but is disallowed; greedy over the
        // masked row must pick the best *allowed* id (3), never id 0.
        let allowed = AllowedSet::Only([2, 3].into_iter().collect());
        let mut logits = vec![100.0, 50.0, 4.0, 9.0, 7.0];
        mask_logits(&mut logits, &allowed).unwrap();
        let picked = greedy_argmax(&logits);
        assert_eq!(picked, 3, "argmax must land on the highest *allowed* logit, not the masked-out max");
        assert!(allowed.contains(picked));
    }

    #[test]
    fn mask_survives_temperature_scaling_stays_neg_inf() {
        // -inf divided by any positive temperature is still -inf -> this is
        // exactly why we mask to -inf and not 0.0 (0.0/t == 0.0, a very
        // samplable score). Pin it.
        let allowed = AllowedSet::Only([1].into_iter().collect());
        let mut logits = vec![5.0, 5.0];
        mask_logits(&mut logits, &allowed).unwrap();
        let scaled = logits[0] / 0.7;
        assert!(scaled.is_infinite() && scaled < 0.0, "-inf must survive temperature division");
    }

    #[test]
    fn empty_allowed_set_is_an_honest_error_not_a_panic() {
        // No allowed id is in range for this 3-long row -> NoTokenAllowed,
        // returned as an Err, never a panic.
        let allowed = AllowedSet::Only([99].into_iter().collect());
        let mut logits = vec![1.0, 2.0, 3.0];
        assert_eq!(mask_logits(&mut logits, &allowed), Err(ConstraintError::NoTokenAllowed));
    }

    // -- AllowedTokens --

    #[test]
    fn allowed_tokens_rejects_an_empty_set_at_construction() {
        assert_eq!(AllowedTokens::new([]).unwrap_err(), ConstraintError::EmptyConstraint);
    }

    #[test]
    fn allowed_tokens_returns_its_fixed_set_regardless_of_state() {
        let c = AllowedTokens::new([2, 7]).unwrap();
        let a = c.allowed(&DecodeState { generated: &[] });
        let b = c.allowed(&DecodeState { generated: &[2, 2, 2] });
        assert_eq!(a, b, "a fixed-set constraint must ignore decode state");
        assert!(a.contains(2) && a.contains(7) && !a.contains(3));
    }

    // -- ConstrainedSampler: real-sampling integration --

    #[test]
    fn constrained_greedy_sampler_only_ever_emits_allowed_ids() {
        let constraint = AllowedTokens::new([2, 4]).unwrap();
        let mut sampler = ConstrainedSampler::new(constraint, GreedySampler);
        // Id 0 always has the highest raw logit; without the mask greedy would
        // pick it every time. With the mask it must pick among {2, 4}.
        for _ in 0..10 {
            let logits = vec![100.0, 90.0, 3.0, 80.0, 5.0];
            let id = sampler.try_sample(&logits).unwrap();
            assert!(id == 2 || id == 4, "constrained greedy emitted disallowed id {id}");
        }
    }

    #[test]
    fn constrained_stochastic_sampler_only_ever_emits_allowed_ids() {
        let constraint = AllowedTokens::new([1, 3]).unwrap();
        let inner = StochasticSampler::new(SamplingConfig {
            temperature: 1.2,
            top_k: Some(4),
            top_p: Some(0.9),
            seed: 123,
            ..SamplingConfig::default()
        });
        let mut sampler = ConstrainedSampler::new(constraint, inner);
        let mut rng_seed = 7u64;
        for _ in 0..200 {
            // cheap LCG just to vary the logits row
            rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let logits: Vec<f32> = (0..5).map(|k| ((rng_seed >> (k * 8)) & 0xff) as f32 / 25.0).collect();
            let id = sampler.try_sample(&logits).unwrap();
            assert!(id == 1 || id == 3, "constrained stochastic emitted disallowed id {id}");
        }
    }

    #[test]
    fn constrained_sampler_records_its_generated_prefix() {
        let constraint = AllowedTokens::new([2]).unwrap();
        let mut sampler = ConstrainedSampler::new(constraint, GreedySampler);
        sampler.try_sample(&[0.0, 0.0, 1.0]).unwrap();
        sampler.try_sample(&[0.0, 0.0, 1.0]).unwrap();
        assert_eq!(sampler.generated(), &[2, 2]);
        sampler.reset();
        assert_eq!(sampler.generated(), &[] as &[u32]);
    }

    #[test]
    fn constrained_sampler_surfaces_no_token_allowed_as_err() {
        // Allowed id 9 is out of range of a 3-long logits row -> honest Err.
        let constraint = AllowedTokens::new([9]).unwrap();
        let mut sampler = ConstrainedSampler::new(constraint, GreedySampler);
        assert_eq!(sampler.try_sample(&[1.0, 2.0, 3.0]), Err(ConstraintError::NoTokenAllowed));
        assert_eq!(sampler.generated(), &[] as &[u32], "a failed step must not record a token");
    }

    // -- JsonStructure: the structural machine --

    /// A vocab of single structural characters plus a few multi-byte tokens,
    /// so tests can drive the JSON machine by id.
    fn json_test_vocab() -> SliceVocab {
        SliceVocab::new(vec![
            b"{".to_vec(),     // 0
            b"}".to_vec(),     // 1
            b"[".to_vec(),     // 2
            b"]".to_vec(),     // 3
            b":".to_vec(),     // 4
            b",".to_vec(),     // 5
            b"\"".to_vec(),    // 6  bare quote
            b"\"k\"".to_vec(), // 7  a complete "k" string
            b"1".to_vec(),     // 8  a scalar
            b" ".to_vec(),     // 9  whitespace
            b"\"v\"".to_vec(), // 10 a complete "v" string
        ])
    }

    fn allowed_ids(set: &AllowedSet, count: usize) -> Vec<u32> {
        (0..count as u32).filter(|&id| set.contains(id)).collect()
    }

    #[test]
    fn json_document_start_allows_a_value_opener_not_a_close() {
        let c = JsonStructure::new(json_test_vocab());
        let a = c.allowed(&DecodeState { generated: &[] });
        // `{`(0), `[`(2), a string(6,7,10), a scalar(8), whitespace(9) may
        // start; a close `}`(1) `]`(3), `:`(4), `,`(5) may not.
        assert!(a.contains(0) && a.contains(2), "must allow '{{' and '['");
        assert!(a.contains(8), "must allow a scalar start");
        assert!(!a.contains(1) && !a.contains(3), "must not allow a close bracket at start");
        assert!(!a.contains(4) && !a.contains(5), "must not allow ':' or ',' at start");
    }

    #[test]
    fn json_after_open_brace_forces_a_key_or_close_only() {
        let c = JsonStructure::new(json_test_vocab());
        // generated: `{`
        let a = c.allowed(&DecodeState { generated: &[0] });
        assert!(a.contains(7), "after '{{' a \"key\" string is allowed");
        assert!(a.contains(6), "after '{{' a bare quote (start of a key) is allowed");
        assert!(a.contains(1), "after '{{' the object may immediately close with '}}'");
        assert!(!a.contains(0), "after '{{' another '{{' (a non-string value) is NOT a valid key");
        assert!(!a.contains(8), "after '{{' a bare scalar is NOT a valid key");
        assert!(!a.contains(5), "after '{{' a ',' is not valid");
    }

    #[test]
    fn json_after_key_forces_a_colon() {
        let c = JsonStructure::new(json_test_vocab());
        // generated: `{` `"k"`
        let a = c.allowed(&DecodeState { generated: &[0, 7] });
        let non_ws: Vec<u32> = allowed_ids(&a, 11).into_iter().filter(|&id| id != 9).collect();
        assert_eq!(non_ws, vec![4], "after a key only ':' (plus whitespace) is allowed");
    }

    #[test]
    fn json_after_colon_expects_a_value() {
        let c = JsonStructure::new(json_test_vocab());
        // generated: `{` `"k"` `:`
        let a = c.allowed(&DecodeState { generated: &[0, 7, 4] });
        assert!(a.contains(10), "after ':' a string value is allowed");
        assert!(a.contains(8), "after ':' a scalar value is allowed");
        assert!(a.contains(0) && a.contains(2), "after ':' a nested object/array is allowed");
        assert!(!a.contains(1) && !a.contains(4), "after ':' a '}}' or ':' is not allowed");
    }

    #[test]
    fn json_after_value_in_object_allows_comma_or_close_only() {
        let c = JsonStructure::new(json_test_vocab());
        // generated: `{` `"k"` `:` `"v"` -- a *string* value, so the value is
        // definitively complete (unlike a bare scalar, which has no explicit
        // terminator and may still be continued mid-stream -- see
        // `json_mid_scalar_may_continue_or_be_separated`).
        let a = c.allowed(&DecodeState { generated: &[0, 7, 4, 10] });
        assert!(a.contains(5), "after a value, ',' is allowed");
        assert!(a.contains(1), "after a value, '}}' (close object) is allowed");
        assert!(!a.contains(3), "a ']' cannot close an object");
        assert!(!a.contains(10), "a bare second value without a separator is not allowed");
        assert!(!a.contains(8), "a bare scalar value without a separator is not allowed");
    }

    #[test]
    fn json_mid_scalar_may_continue_or_be_separated() {
        // A scalar has no explicit terminator, so after emitting `1` the model
        // may legitimately continue the number (`1` -> `11`) OR terminate it
        // with a separator/close. Both are structurally valid; this pins that
        // streaming-scalar behaviour so nobody "fixes" it into a bug.
        let c = JsonStructure::new(json_test_vocab());
        // generated: `{` `"k"` `:` `1`  (machine is still mid-scalar)
        let a = c.allowed(&DecodeState { generated: &[0, 7, 4, 8] });
        assert!(a.contains(8), "mid-scalar, another digit continues the number");
        assert!(a.contains(5), "mid-scalar, ',' terminates the scalar and separates");
        assert!(a.contains(1), "mid-scalar, '}}' terminates the scalar and closes the object");
        assert!(!a.contains(3), "mid-scalar, ']' still cannot close an *object*");
    }

    #[test]
    fn json_balanced_close_returns_to_done() {
        let c = JsonStructure::new(json_test_vocab());
        // generated: `{` `"k"` `:` `1` `}` -> a complete document.
        let a = c.allowed(&DecodeState { generated: &[0, 7, 4, 8, 1] });
        // Only whitespace may follow a complete top-level value.
        assert!(a.contains(9), "whitespace may follow a complete document");
        assert!(
            !a.contains(5) && !a.contains(0) && !a.contains(1),
            "nothing structural may follow a complete document"
        );
    }

    #[test]
    fn json_empty_array_may_close_immediately() {
        let c = JsonStructure::new(json_test_vocab());
        // generated: `[`
        let a = c.allowed(&DecodeState { generated: &[2] });
        assert!(a.contains(3), "an empty array '[]' must be allowed to close");
        assert!(a.contains(8), "an array may also start with a value");
        assert!(!a.contains(5), "a leading ',' inside a fresh array is not valid");
    }

    #[test]
    fn json_always_allowed_ids_bypass_the_grammar() {
        // Register id 3 (`]`) as always-allowed even though it is structurally
        // invalid at document start; it must come through. This is how EOS is
        // wired in a real decode.
        let c = JsonStructure::with_always_allowed(json_test_vocab(), [3]);
        let a = c.allowed(&DecodeState { generated: &[] });
        assert!(a.contains(3), "an always-allowed id must bypass the structural check");
    }

    #[test]
    fn json_machine_accepts_a_full_nested_document() {
        // Drive the raw machine over a full document to prove the pieces
        // compose: {"k":[1,"v"]}
        let mut m = JsonMachine::new();
        for &b in b"{\"k\":[1,\"v\"]}" {
            assert!(m.step(b), "byte {:?} rejected in a valid document", b as char);
        }
        assert_eq!(m.pos, Pos::Done, "a balanced document must land in Done");
    }

    #[test]
    fn json_machine_rejects_a_mismatched_close() {
        let mut m = JsonMachine::new();
        assert!(m.step(b'{'));
        // A `]` cannot close an object.
        assert!(!m.step(b']'));
    }
}
