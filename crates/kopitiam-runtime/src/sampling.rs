//! Turning a row of logits into a token id.
//!
//! # Two samplers, one trait
//!
//! [`GreedySampler`] (`argmax`) is deterministic and repetitive: it always
//! picks the single highest-scoring token, so the same prompt always
//! produces the same completion, but real conversation needs variety a
//! pure maximum can never produce. [`StochasticSampler`] is the
//! alternative every practical LLM serving stack actually uses —
//! temperature, top-k, top-p (nucleus), min-p, and a repetition penalty,
//! composed as a *pipeline* of logit transforms rather than a pile of
//! `if`-branches. That pipeline shape (not a monolithic "sample" function)
//! is deliberate and mirrors how `llama.cpp` models its own sampler chain:
//! each stage does one well-defined thing to a `[f32]` of per-token scores
//! (mask out excluded tokens with [`f32::NEG_INFINITY`], or rescale
//! surviving ones), stages compose in any order a caller chooses, and each
//! stage is unit-testable in isolation against a hand-computed
//! distribution — see this module's tests for exactly that.
//!
//! # Why a hand-rolled PRNG instead of `rand`
//!
//! Stochastic sampling is meaningless without randomness, but "random" and
//! "deterministic behaviour" (CLAUDE.md's standing requirement) are not in
//! tension here: a PRNG is a pure function of its seed and call count, so
//! seeding it explicitly makes the *whole* generation loop reproducible —
//! same seed, same prompt, same model, same token sequence, forever. That
//! is why [`SamplingConfig::seed`] exists and why it is mandatory rather
//! than "defaults to system entropy": an unseedable sampler cannot be unit
//! tested (there would be no way to assert *which* token comes out), and
//! CLAUDE.md is explicit that AI-adjacent or randomized workflows still
//! owe the platform reproducibility. [`Rng`] is xorshift64* — about a
//! dozen lines, no external crate, good enough statistical quality for
//! sampling a few thousand tokens (it is not used for anything
//! cryptographic) — which is a better fit for this workspace's "avoid
//! unnecessary dependencies" rule than pulling in `rand`'s dependency tree
//! for one struct's worth of functionality.

/// Chooses the next token id from one row of logits (length `vocab_size`).
pub trait Sampler {
    fn sample(&mut self, logits: &[f32]) -> u32;
}

/// Always picks the highest-scoring token: `argmax(logits)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        greedy_argmax(logits)
    }
}

/// `argmax(logits)`, ties broken toward the lowest id (the first maximum
/// encountered) — a `PartialOrd` tie-break rule that is total and
/// deterministic even in the presence of `NaN` (`f32::partial_cmp`'s `None`
/// case), unlike an `assert`-free `.max_by(...)` over raw `f32`, which
/// would panic or silently misbehave on a `NaN` logit.
pub fn greedy_argmax(logits: &[f32]) -> u32 {
    assert!(!logits.is_empty(), "greedy_argmax requires at least one logit");
    let mut best_idx = 0usize;
    let mut best_val = logits[0];
    for (idx, &val) in logits.iter().enumerate().skip(1) {
        if val > best_val {
            best_idx = idx;
            best_val = val;
        }
    }
    best_idx as u32
}

/// A small, seedable, dependency-free PRNG: xorshift64*
/// (Marsaglia/Vigna). Not cryptographic — nothing in this crate needs
/// that — but a fixed seed deterministically reproduces its entire output
/// sequence, which is the one property [`StochasticSampler`] actually
/// needs (see this module's docs).
#[derive(Debug, Clone)]
struct Rng(u64);

impl Rng {
    /// A seed of `0` would leave xorshift's state permanently `0` (its one
    /// fixed point — every subsequent xor-shift of `0` is `0`), so it is
    /// remapped to an arbitrary nonzero constant. Every other seed is used
    /// as-is, so distinct nonzero seeds still produce distinct sequences.
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A uniform `f32` in `[0, 1)`, built from the top 24 bits of
    /// [`Self::next_u64`] (an `f32` mantissa only has 24 bits of precision
    /// including the implicit leading one, so using more source bits than
    /// that would not add resolution, only bias the low bits).
    fn next_unit_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Configuration for [`StochasticSampler`]'s logit-transform pipeline.
///
/// Every threshold is `Option`-or-plain depending on whether "disabled" is
/// a meaningful state: `top_k`/`top_p`/`min_p` default to `None` (that
/// stage of the pipeline is skipped entirely), while `temperature` and
/// `repeat_penalty` always run (a temperature of `1.0` and a repeat
/// penalty of `1.0` are each that stage's identity value, so leaving them
/// at their defaults is equivalent to skipping them, without needing a
/// second `Option` layer).
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Divides every logit before the final softmax. `1.0` (the default)
    /// leaves the distribution's shape unchanged; values below `1.0`
    /// sharpen it toward the top-scoring tokens, values above `1.0`
    /// flatten it toward uniform. `<= 0.0` is treated as "skip stochastic
    /// sampling entirely and fall back to [`greedy_argmax`]" — see
    /// [`StochasticSampler::sample`]'s docs for why that is an exact
    /// equivalence, not an approximation.
    pub temperature: f32,
    /// Keep only the `k` highest-scoring tokens before sampling. `None`
    /// (the default) skips this stage.
    pub top_k: Option<usize>,
    /// Nucleus sampling: keep the smallest prefix of tokens (sorted by
    /// probability, descending) whose cumulative probability is the first
    /// to exceed `p`. `None` (the default) skips this stage.
    pub top_p: Option<f32>,
    /// Keep only tokens whose probability is at least `min_p` times the
    /// single most likely token's probability. `None` (the default) skips
    /// this stage.
    pub min_p: Option<f32>,
    /// Divides a previously-generated token's logit by this value if it
    /// was positive, or multiplies it if negative — the standard
    /// (Keskar et al., 2019, CTRL) repetition penalty. `1.0` (the default)
    /// is the identity: no penalty.
    pub repeat_penalty: f32,
    /// How many of the most recently generated tokens
    /// [`StochasticSampler`] remembers for [`Self::repeat_penalty`]'s
    /// history window.
    pub repeat_last_n: usize,
    /// Seeds [`StochasticSampler`]'s internal PRNG — see this module's
    /// docs for why this is mandatory rather than defaulting to system
    /// entropy.
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            min_p: None,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 0,
        }
    }
}

/// Temperature / top-k / top-p / min-p / repetition-penalty sampling,
/// composed as a pipeline of logit transforms (see this module's docs) and
/// driven by a seeded [`Rng`].
///
/// # Pipeline order
///
/// [`StochasticSampler::sample`] applies, in this order: repetition
/// penalty (needs the *raw* logit scale — see
/// [`apply_repetition_penalty`]'s docs) -> top-k -> top-p -> min-p ->
/// temperature -> softmax -> weighted-random draw. Top-k/top-p/min-p run
/// *before* temperature deliberately: top-p and min-p compute a softmax
/// internally to rank tokens by probability, and temperature changes how
/// flat or peaked that probability distribution is — running them after
/// temperature would make the nucleus/threshold decision depend on a
/// rescaling that has not been "committed to" yet. This is the same
/// ordering `llama.cpp`'s sampler chain uses.
pub struct StochasticSampler {
    config: SamplingConfig,
    rng: Rng,
    /// The most recent [`SamplingConfig::repeat_last_n`] sampled token
    /// ids, oldest first. A `Vec` with an O(n) front-remove is fine here:
    /// `repeat_last_n` is a small, bounded window (tens to low hundreds of
    /// tokens), not the whole generated sequence.
    history: Vec<u32>,
}

impl StochasticSampler {
    pub fn new(config: SamplingConfig) -> Self {
        let rng = Rng::new(config.seed);
        Self { config, rng, history: Vec::new() }
    }
}

impl Sampler for StochasticSampler {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        assert!(!logits.is_empty(), "StochasticSampler::sample requires at least one logit");
        let mut work = logits.to_vec();
        apply_repetition_penalty(&mut work, &self.history, self.config.repeat_penalty);

        // temperature <= 0.0 has no meaningful "divide by temperature"
        // reading (0.0 is a division by zero; negative would invert the
        // distribution's ranking) -- the only sensible interpretation is
        // "as sharp as possible", i.e. plain argmax. This is not a
        // fallback bolted on for safety: it is what makes
        // `StochasticSampler { temperature: 0.0, .. }` an exact drop-in
        // replacement for `GreedySampler` (see
        // `tests::temperature_zero_degrades_exactly_to_greedy`), so a
        // caller can switch between "deterministic" and "stochastic"
        // generation by changing one number instead of swapping sampler
        // types.
        let chosen = if self.config.temperature <= 0.0 {
            greedy_argmax(&work)
        } else {
            if let Some(k) = self.config.top_k {
                apply_top_k(&mut work, k);
            }
            if let Some(p) = self.config.top_p {
                apply_top_p(&mut work, p);
            }
            if let Some(mp) = self.config.min_p {
                apply_min_p(&mut work, mp);
            }
            apply_temperature(&mut work, self.config.temperature);
            sample_from_logits(&work, &mut self.rng)
        };

        self.history.push(chosen);
        let window = self.config.repeat_last_n.max(1);
        if self.history.len() > window {
            self.history.remove(0);
        }
        chosen
    }
}

/// Rescales every logit belonging to a token in `history` — the standard
/// (Keskar et al., 2019, "CTRL") repetition penalty: a positive logit is
/// divided by `penalty`, a negative one is multiplied by it, so `penalty
/// > 1.0` always pushes a previously-seen token's score down regardless of
/// its sign, and `penalty == 1.0` is the identity (every other value in
/// `history` is a no-op, and duplicate ids in `history` are naturally
/// idempotent — applying the same rescale to the same logit twice would
/// double-penalize it, but every id is only visited once here because the
/// loop is driven by [`std::collections::HashSet`]-deduplicated ids, not
/// by `history`'s raw entries).
///
/// Runs on raw logits, *before* top-k/top-p/temperature: those stages
/// reason about relative rank and cumulative probability, both of which
/// the penalty is specifically meant to disturb (nudging a repeated
/// token's rank down); running it after would let a repeated token that
/// already survived top-k/top-p keep its undiminished probability mass.
fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    let seen: std::collections::HashSet<u32> = history.iter().copied().collect();
    for &id in &seen {
        let Some(logit) = logits.get_mut(id as usize) else { continue };
        *logit = if *logit > 0.0 { *logit / penalty } else { *logit * penalty };
    }
}

/// Keeps only the `k` highest-scoring logits, masking every other one to
/// [`f32::NEG_INFINITY`] (excluded from every later stage and from the
/// final sample, since `exp(-inf) == 0.0`). `k == 0` or `k >=
/// logits.len()` is a no-op — there is nothing to remove.
fn apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    // Descending by score; NaN cannot occur in a real logits row (a NaN
    // forward pass is already a bug elsewhere), so `partial_cmp().unwrap()`
    // is a deliberate "this should never happen" panic, not an
    // unhandled-edge-case shortcut.
    order.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    for &idx in &order[k..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}

/// Nucleus sampling: ranks tokens by probability (softmax of `logits`,
/// [`f32::NEG_INFINITY`] entries already contributing zero probability),
/// then keeps the smallest prefix whose cumulative probability is the
/// *first* to exceed `p` — masking every token after that prefix to
/// [`f32::NEG_INFINITY`].
///
/// # The off-by-one this function exists to get right
///
/// The classic bug here is excluding the token whose addition pushes the
/// cumulative sum past `p`. It must be *included*: nucleus sampling's
/// whole definition is "the smallest set of tokens whose probability sums
/// to at least `p`", and excluding the crossing token would make the kept
/// set's probability strictly less than `p` in general — the opposite of
/// what "at least" means. This function's loop adds a token's probability
/// to the running total *before* checking whether the threshold was
/// crossed, and only then decides whether to stop, which is what keeps
/// the crossing token in — see `tests::top_p_includes_the_crossing_token`
/// for the pinned example.
///
/// `p >= 1.0` is a no-op (every token's cumulative probability trivially
/// reaches 1.0, i.e. "keep everything").
fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }
    let probs = softmax_ignoring_masked(logits);
    let mut order: Vec<usize> = (0..logits.len()).filter(|&i| probs[i] > 0.0).collect();
    order.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    let mut cumulative = 0.0f32;
    let mut keep = order.len();
    for (rank, &idx) in order.iter().enumerate() {
        cumulative += probs[idx];
        if cumulative > p {
            keep = rank + 1; // include the crossing token itself.
            break;
        }
    }
    for &idx in &order[keep..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}

/// Keeps only tokens whose probability is at least `min_p` times the
/// single most likely token's probability — a relative floor rather than
/// nucleus sampling's cumulative one. `min_p <= 0.0` is a no-op (every
/// probability is `>= 0.0`, i.e. "keep everything"); the most likely token
/// itself always survives (its own threshold is exactly its own
/// probability).
fn apply_min_p(logits: &mut [f32], min_p: f32) {
    if min_p <= 0.0 {
        return;
    }
    let probs = softmax_ignoring_masked(logits);
    let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
    let threshold = min_p * max_prob;
    for (idx, &pr) in probs.iter().enumerate() {
        if pr < threshold {
            logits[idx] = f32::NEG_INFINITY;
        }
    }
}

/// Divides every (non-masked) logit by `temperature`. Callers only reach
/// this with `temperature > 0.0` — see [`StochasticSampler::sample`]'s
/// docs for why `<= 0.0` is handled as a separate greedy path instead.
fn apply_temperature(logits: &mut [f32], temperature: f32) {
    for logit in logits.iter_mut() {
        if logit.is_finite() {
            *logit /= temperature;
        }
    }
}

/// Softmax over `logits`, treating [`f32::NEG_INFINITY`] entries (this
/// pipeline's "excluded" marker) as exactly zero probability rather than
/// `NaN`. Shifts by the maximum finite logit first for numerical
/// stability, the same trick [`kopitiam_tensor::Tensor::softmax`] uses,
/// reimplemented here on a plain slice because this pipeline runs on a
/// `Vec<f32>` extracted from the model's logits row, not on a
/// [`kopitiam_tensor::Tensor`] directly.
fn softmax_ignoring_masked(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // Every logit is -inf (or empty): no valid token. Returning a
        // uniform distribution is safer than dividing by a zero sum, and
        // this should be unreachable in practice (top-k/top-p/min-p never
        // mask every token — see their own docs).
        let n = logits.len().max(1);
        return vec![1.0 / n as f32; logits.len()];
    }
    let exps: Vec<f32> = logits.iter().map(|&l| if l.is_finite() { (l - max).exp() } else { 0.0 }).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        let n = logits.len().max(1);
        return vec![1.0 / n as f32; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

/// Draws one token id from `logits` via `softmax(logits)` and a uniform
/// random draw from `rng`: walks token ids in order, accumulating
/// probability, and returns the first id whose cumulative probability
/// reaches the draw. Falls back to [`greedy_argmax`] (the highest-
/// probability surviving token) if floating-point summation leaves the
/// cumulative total a hair under `1.0` and the draw landed past it —
/// exceedingly rare, but a `Vec` index panic on a `1 in 2^24` fluke is
/// strictly worse than silently taking the single most likely outcome.
fn sample_from_logits(logits: &[f32], rng: &mut Rng) -> u32 {
    let probs = softmax_ignoring_masked(logits);
    let draw = rng.next_unit_f32();
    let mut cumulative = 0.0f32;
    for (idx, &pr) in probs.iter().enumerate() {
        cumulative += pr;
        if draw < cumulative {
            return idx as u32;
        }
    }
    greedy_argmax(&probs)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- greedy_argmax / GreedySampler (pre-existing) --

    #[test]
    fn greedy_argmax_picks_the_highest_scoring_index() {
        assert_eq!(greedy_argmax(&[0.1, 5.0, -3.0, 2.0]), 1);
    }

    #[test]
    fn greedy_argmax_breaks_ties_toward_the_first_occurrence() {
        assert_eq!(greedy_argmax(&[1.0, 3.0, 3.0, 0.0]), 1);
    }

    #[test]
    fn greedy_argmax_handles_a_single_element() {
        assert_eq!(greedy_argmax(&[42.0]), 0);
    }

    #[test]
    fn greedy_sampler_is_deterministic_across_repeated_calls() {
        let mut sampler = GreedySampler;
        let logits = [0.5, 2.0, 1.0];
        assert_eq!(sampler.sample(&logits), sampler.sample(&logits));
    }

    #[test]
    #[should_panic]
    fn greedy_argmax_rejects_empty_input() {
        greedy_argmax(&[]);
    }

    // -- Rng --

    #[test]
    fn rng_with_the_same_seed_produces_the_same_sequence() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        let seq_a: Vec<u64> = (0..10).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..10).map(|_| b.next_u64()).collect();
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn rng_with_different_seeds_produces_different_sequences() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let seq_a: Vec<u64> = (0..10).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..10).map(|_| b.next_u64()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[test]
    fn rng_zero_seed_does_not_get_stuck_at_the_fixed_point() {
        let mut rng = Rng::new(0);
        let values: Vec<u64> = (0..5).map(|_| rng.next_u64()).collect();
        assert!(values.iter().all(|&v| v != 0), "a stuck-at-zero PRNG would emit all zeros: {values:?}");
    }

    #[test]
    fn rng_unit_f32_stays_within_zero_one() {
        let mut rng = Rng::new(7);
        for _ in 0..1000 {
            let v = rng.next_unit_f32();
            assert!((0.0..1.0).contains(&v), "{v} outside [0, 1)");
        }
    }

    // -- apply_repetition_penalty --

    #[test]
    fn repetition_penalty_of_one_is_a_no_op() {
        let mut logits = vec![1.0, -1.0, 2.0];
        apply_repetition_penalty(&mut logits, &[0, 1, 2], 1.0);
        assert_eq!(logits, vec![1.0, -1.0, 2.0]);
    }

    #[test]
    fn repetition_penalty_divides_positive_and_multiplies_negative_logits() {
        let mut logits = vec![4.0, -4.0, 10.0];
        // Token 0 (positive logit) and token 1 (negative logit) are in
        // history; token 2 is not and must be untouched.
        apply_repetition_penalty(&mut logits, &[0, 1], 2.0);
        assert_eq!(logits[0], 2.0); // 4.0 / 2.0
        assert_eq!(logits[1], -8.0); // -4.0 * 2.0
        assert_eq!(logits[2], 10.0); // untouched
    }

    #[test]
    fn repetition_penalty_ignores_duplicate_history_entries() {
        let mut logits = vec![8.0];
        apply_repetition_penalty(&mut logits, &[0, 0, 0], 2.0);
        assert_eq!(logits[0], 4.0, "repeated history entries must not compound the penalty");
    }

    // -- apply_top_k --

    #[test]
    fn top_k_keeps_exactly_the_k_highest_logits() {
        let mut logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        apply_top_k(&mut logits, 2);
        let kept: Vec<usize> = (0..5).filter(|&i| logits[i].is_finite()).collect();
        assert_eq!(kept, vec![1, 3]); // values 5.0 and 4.0
    }

    #[test]
    fn top_k_zero_is_a_no_op() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_top_k(&mut logits, 0);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn top_k_at_or_above_the_length_is_a_no_op() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_top_k(&mut logits, 3);
        assert!(logits.iter().all(|v| v.is_finite()));
        let mut logits2 = vec![1.0, 2.0, 3.0];
        apply_top_k(&mut logits2, 10);
        assert!(logits2.iter().all(|v| v.is_finite()));
    }

    // -- apply_top_p (the classic off-by-one) --

    #[test]
    fn top_p_includes_the_crossing_token() {
        // Probabilities (after softmax) chosen to land exactly on this
        // module's documented example: [0.5, 0.3, 0.2], p = 0.6.
        // Cumulative after token 0 (0.5) does not exceed 0.6; cumulative
        // after token 1 (0.8) does -> token 1 (the crossing token) must be
        // KEPT, only token 2 excluded.
        let logits = to_logits_from_probs(&[0.5, 0.3, 0.2]);
        let mut work = logits.clone();
        apply_top_p(&mut work, 0.6);
        assert!(work[0].is_finite(), "token 0 (below the threshold on its own) must be kept");
        assert!(work[1].is_finite(), "token 1 (the crossing token) must be kept, not excluded");
        assert!(!work[2].is_finite(), "token 2 (after the crossing point) must be excluded");
    }

    #[test]
    fn top_p_keeps_only_the_single_token_when_it_alone_exceeds_p() {
        let logits = to_logits_from_probs(&[0.9, 0.05, 0.05]);
        let mut work = logits.clone();
        apply_top_p(&mut work, 0.5);
        assert!(work[0].is_finite());
        assert!(!work[1].is_finite());
        assert!(!work[2].is_finite());
    }

    #[test]
    fn top_p_of_one_or_above_is_a_no_op() {
        let logits = to_logits_from_probs(&[0.5, 0.3, 0.2]);
        let mut work = logits.clone();
        apply_top_p(&mut work, 1.0);
        assert!(work.iter().all(|v| v.is_finite()));
    }

    /// Builds a logits row whose softmax is exactly `probs` (which must sum
    /// to 1.0): `logit_i = ln(prob_i)` inverts softmax up to the additive
    /// constant softmax is invariant to, so `softmax(ln(probs)) == probs`
    /// exactly (mod floating-point rounding). This is what lets
    /// `top_p`/`min_p` tests assert against round, hand-checkable
    /// probabilities instead of reverse-engineering what an arbitrary
    /// logits row softmaxes to.
    fn to_logits_from_probs(probs: &[f32]) -> Vec<f32> {
        probs.iter().map(|p| p.ln()).collect()
    }

    // -- apply_min_p --

    #[test]
    fn min_p_keeps_only_tokens_within_the_relative_threshold_of_the_max() {
        // max prob = 0.7; min_p = 0.5 -> threshold = 0.35. 0.2 < 0.35 is
        // excluded, 0.7 and (barely) nothing else survives among these.
        let logits = to_logits_from_probs(&[0.7, 0.2, 0.1]);
        let mut work = logits.clone();
        apply_min_p(&mut work, 0.5);
        assert!(work[0].is_finite());
        assert!(!work[1].is_finite());
        assert!(!work[2].is_finite());
    }

    #[test]
    fn min_p_always_keeps_the_top_token() {
        let logits = to_logits_from_probs(&[0.4, 0.35, 0.25]);
        let mut work = logits.clone();
        apply_min_p(&mut work, 0.99);
        assert!(work[0].is_finite(), "the single most likely token can never be excluded by its own threshold");
    }

    #[test]
    fn min_p_of_zero_or_below_is_a_no_op() {
        let logits = to_logits_from_probs(&[0.7, 0.2, 0.1]);
        let mut work = logits.clone();
        apply_min_p(&mut work, 0.0);
        assert!(work.iter().all(|v| v.is_finite()));
    }

    // -- softmax_ignoring_masked --

    #[test]
    fn softmax_ignoring_masked_treats_negative_infinity_as_zero_probability() {
        let probs = softmax_ignoring_masked(&[1.0, f32::NEG_INFINITY, 1.0]);
        assert_eq!(probs[1], 0.0);
        assert!((probs[0] - 0.5).abs() < 1e-6);
        assert!((probs[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn softmax_ignoring_masked_rows_sum_to_one() {
        let probs = softmax_ignoring_masked(&[2.0, -1.0, 0.5, f32::NEG_INFINITY]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // -- StochasticSampler: the properties the task brief calls for --

    /// `temperature <= 0.0` must be an *exact* drop-in for
    /// [`GreedySampler`], not merely "usually agrees": the two code paths
    /// must never be able to silently disagree about what "greedy" means.
    #[test]
    fn temperature_zero_degrades_exactly_to_greedy() {
        let logits = [0.1, 5.0, -3.0, 2.0, 5.0]; // includes a tie, to exercise tie-breaking too.
        let mut greedy = GreedySampler;
        let mut stochastic = StochasticSampler::new(SamplingConfig { temperature: 0.0, ..SamplingConfig::default() });

        for _ in 0..5 {
            assert_eq!(stochastic.sample(&logits), greedy.sample(&logits));
        }
    }

    #[test]
    fn negative_temperature_also_degrades_to_greedy() {
        let logits = [1.0, 9.0, 2.0];
        let mut sampler = StochasticSampler::new(SamplingConfig { temperature: -1.0, ..SamplingConfig::default() });
        assert_eq!(sampler.sample(&logits), greedy_argmax(&logits));
    }

    #[test]
    fn a_fixed_seed_reproduces_a_fixed_token_sequence() {
        let logits_sequence: Vec<Vec<f32>> =
            (0..20).map(|i| vec![1.0 + i as f32 * 0.1, 2.0, 0.5, 3.0 - i as f32 * 0.05, 1.5]).collect();

        let config = SamplingConfig { temperature: 0.8, top_k: Some(3), seed: 12345, ..SamplingConfig::default() };
        let run = |cfg: SamplingConfig| -> Vec<u32> {
            let mut sampler = StochasticSampler::new(cfg);
            logits_sequence.iter().map(|l| sampler.sample(l)).collect()
        };

        let first = run(config.clone());
        let second = run(config);
        assert_eq!(first, second, "the same seed must reproduce the exact same token sequence");
    }

    #[test]
    fn different_seeds_usually_produce_different_sequences() {
        let logits_sequence: Vec<Vec<f32>> = (0..20).map(|i| vec![1.0, 2.0 + i as f32 * 0.01, 1.5, 1.8]).collect();
        let run = |seed: u64| -> Vec<u32> {
            let cfg = SamplingConfig { temperature: 1.0, seed, ..SamplingConfig::default() };
            let mut sampler = StochasticSampler::new(cfg);
            logits_sequence.iter().map(|l| sampler.sample(l)).collect()
        };
        assert_ne!(run(1), run(2));
    }

    #[test]
    fn stochastic_sampler_only_ever_emits_in_range_token_ids() {
        let vocab = 7usize;
        let cfg = SamplingConfig {
            temperature: 1.2,
            top_k: Some(4),
            top_p: Some(0.9),
            min_p: Some(0.05),
            repeat_penalty: 1.1,
            seed: 999,
            ..SamplingConfig::default()
        };
        let mut sampler = StochasticSampler::new(cfg);
        let mut rng_for_logits = Rng::new(42);
        for _ in 0..200 {
            let logits: Vec<f32> = (0..vocab).map(|_| rng_for_logits.next_unit_f32() * 10.0 - 5.0).collect();
            let id = sampler.sample(&logits);
            assert!((id as usize) < vocab, "sampled id {id} out of range for vocab {vocab}");
        }
    }

    #[test]
    fn repetition_penalty_can_flip_which_token_greedy_decoding_picks() {
        // Token 1 (5.0) narrowly beats token 3 (4.9) on the raw logits, so
        // an unpenalized greedy sampler would pick token 1 forever. Once
        // token 1 has been sampled once (entering the history window), a
        // 2x penalty divides its logit to 2.5 -- well below token 3's
        // untouched 4.9 -- so the very next call must switch. This is
        // computed from a fresh division of the *raw* logit each call
        // (see `apply_repetition_penalty`'s docs on why duplicate history
        // entries do not compound), not a decaying value, so this test
        // checks exactly one flip rather than a multi-call search.
        let mut sampler = StochasticSampler::new(SamplingConfig {
            temperature: 0.0, // greedy, so the effect of the penalty is unambiguous to check
            repeat_penalty: 2.0,
            repeat_last_n: 4,
            ..SamplingConfig::default()
        });
        let logits = [0.1, 5.0, 0.2, 4.9];
        assert_eq!(sampler.sample(&logits), 1, "unpenalized, token 1 has the highest raw logit");
        assert_eq!(
            sampler.sample(&logits),
            3,
            "token 1's history-penalized logit (5.0 / 2.0 = 2.5) must now lose to token 3's untouched 4.9"
        );
    }
}
