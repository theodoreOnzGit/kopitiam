//! Ported from Tesseract `src/lstm/recodebeam.cpp` + `src/lstm/recodebeam.h`
//! (commit db0ec62, Apache-2.0, © 2015 Google Inc., Author: Ray Smith; part of
//! the Tesseract project), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the beam-over-the-recoded-alphabet, the CTC blank/duplicate
//! collapsing, the top-N pruning, the `NodeContinuation`/`TopNState` machinery,
//! the score (log-prob) accumulation, the code-hash duplicate-path removal, and
//! the best-path reassembly through [`UnicharCompress`] follow Tesseract exactly;
//! re-expressed in idiomatic Rust with an index-based lattice arena in place of
//! C++'s borrowed `RecodeNode*` pointers. See docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is — the recode/CTC beam-search decoder (Phase 5)
//!
//! The LSTM network ([`crate::network`]) emits, per timestep, a softmax over the
//! **recoded** alphabet (small code units, plus a CTC null/blank class). This
//! module turns that `[timesteps × code_range]` activation buffer
//! ([`NetworkIO`]) into recognized text. It runs a beam search over the recoder's
//! *valid* code continuations, folding CTC duplicates and dropping nulls, and
//! reassembles multi-unit code sequences back into single unichar ids via
//! [`UnicharCompress::decode_unichar`] — which is *why* CJK works: a Han/Hangul
//! codepoint is several network timesteps that collapse to one id.
//!
//! # The blank/null and CTC folding (the load-bearing correctness bits)
//!
//! `null_char` is the network class of the CTC blank. It is, per Tesseract, the
//! recoder's *encoded* null code (`LSTMRecognizer` passes `encoded_null_char_`):
//! a single-unit code that appears in the recoder's `final_codes` for the empty
//! prefix. In [`RecodeBeamSearch::continue_context`] the null is special-cased so
//! its length-0 completion carries `unichar_id = INVALID_UNICHAR_ID`, and the
//! best-path extractor ([`RecodeBeamSearch::extract_best_path_as_unichar_ids`])
//! then skips nulls and folds trailing `duplicate` nodes — the classic CTC
//! `a a <blank> a a` → `aa` behavior. Adjacent equal codes are the `duplicate`
//! flag (crushed together on the fly), while a null between two equal codes forces
//! them to decode as two separate characters.
//!
//! # Deferred (dictionary/dawg beam, char boxes, secondary beams)
//!
//! The **non-dictionary raw CTC beam** (Tesseract's "no dawg" / top-N path,
//! driven by `Decode(output, 1.0, 0.0, kMinCertainty, nullptr)`) is ported in
//! full. The dawg/dictionary-aware beam (`ContinueDawg`/`PushInitialDawgIfBetter`
//! and the `is_dawg` half of the beam array) is **deferred**: the `is_dawg` beam
//! indices exist (so `beam_index` arithmetic and `kNumBeams` match Tesseract
//! exactly) but are never populated, since a KOPITIAM decode runs without a
//! `Dict`. Char boxes / `WERD_RES` extraction (`ExtractBestPathAsWords`,
//! `InitializeWord`, `calculateCharBoundaries`), the secondary beams
//! (`DecodeSecondaryBeams`), `lstm_choice_mode`/`SaveMostCertainChoices`, and the
//! debug printers are also deferred — Phase 6 drives the raw path first, and the
//! `certs`/`ratings`/`xcoords` this module already returns are what it needs. The
//! reassembly (`extract_best_path_as_unichar_ids`) is ported in full so the CJK
//! path is exercised.

use crate::networkio::NetworkIO;
use crate::unichar::{INVALID_UNICHAR_ID, UnicharId};
use crate::unicharcompress::{RecodedCharID, UnicharCompress};
use crate::unicharset::{UNICHAR_SPACE, Unicharset};

// ---------------------------------------------------------------------------
// Constants (recodebeam.h / recodebeam.cpp)
// ---------------------------------------------------------------------------

/// Clipping value for certainty. Reflects the minimum certainty returned by the
/// extractor. Tesseract: `RecodeBeamSearch::kMinCertainty` (recodebeam.h:254) /
/// `kMinCertainty` (networkio.cpp:30).
pub const K_MIN_CERTAINTY: f32 = -20.0;

/// Probability corresponding to [`K_MIN_CERTAINTY`]. Tesseract: `kMinProb`
/// (networkio.cpp:32) = `exp(kMinCertainty)`.
const K_MIN_PROB: f32 = 2.0611536e-9; // exp(-20.0)

/// Number of different code lengths for which we keep a separate beam.
/// Tesseract: `kNumLengths = RecodedCharID::kMaxCodeLen + 1` (recodebeam.h:256).
pub const K_NUM_LENGTHS: usize = RecodedCharID::MAX_CODE_LEN + 1;

/// Total number of beams: `dawg/nodawg (2) * NC_COUNT * kNumLengths`.
/// Tesseract: `kNumBeams` (recodebeam.h:259).
pub const K_NUM_BEAMS: usize = 2 * NC_COUNT * K_NUM_LENGTHS;

/// The beam width at each code position. Tesseract:
/// `RecodeBeamSearch::kBeamWidths` (recodebeam.cpp:31).
pub const K_BEAM_WIDTHS: [usize; K_NUM_LENGTHS] = [5, 10, 16, 16, 16, 16, 16, 16, 16, 16];

// PermuterType ordinals we use (ratngs.h:235). The dawg permuters are deferred.
/// Tesseract: `NO_PERM` (ratngs.h:236).
const NO_PERM: i32 = 0;
/// Tesseract: `TOP_CHOICE_PERM` (ratngs.h:238).
const TOP_CHOICE_PERM: i32 = 2;

/// What can follow the current node. Tesseract: `enum NodeContinuation`
/// (recodebeam.h:72). See the long comment there for the CTC score-combining
/// rationale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
enum NodeContinuation {
    /// Used just its own score, so anything can follow.
    Anything = 0,
    /// Combined another score without a stand-alone duplicate before, so must be
    /// followed by a stand-alone duplicate.
    OnlyDup = 1,
    /// Combined another score after a stand-alone, so cannot be followed by a
    /// duplicate of the current node.
    NoDup = 2,
}
use NodeContinuation::{Anything as NC_ANYTHING, NoDup as NC_NO_DUP, OnlyDup as NC_ONLY_DUP};

/// Tesseract: `NC_COUNT` (recodebeam.h:80).
const NC_COUNT: usize = 3;

impl NodeContinuation {
    fn from_usize(v: usize) -> NodeContinuation {
        match v {
            0 => NC_ANYTHING,
            1 => NC_ONLY_DUP,
            _ => NC_NO_DUP,
        }
    }
}

/// Top-n status of a code at the current timestep. Tesseract: `enum TopNState`
/// (recodebeam.h:84).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TopNState {
    /// Winner or 2nd.
    Top2,
    /// Runner up in top-n, but not 1st or 2nd.
    TopN,
    /// Not in the top-n.
    AlsoRan,
}
use TopNState::{AlsoRan as TN_ALSO_RAN, Top2 as TN_TOP2, TopN as TN_TOPN};

/// Tesseract: `TN_COUNT` (recodebeam.h:88).
const TN_COUNT: usize = 3;

fn tn_from_usize(v: usize) -> TopNState {
    match v {
        0 => TN_TOP2,
        1 => TN_TOPN,
        _ => TN_ALSO_RAN,
    }
}

// ---------------------------------------------------------------------------
// beams_ index arithmetic (recodebeam.h:261..273)
// ---------------------------------------------------------------------------

/// Tesseract: `LengthFromBeamsIndex` (recodebeam.h:261).
fn length_from_beams_index(index: usize) -> usize {
    index % K_NUM_LENGTHS
}

/// Tesseract: `ContinuationFromBeamsIndex` (recodebeam.h:264).
fn continuation_from_beams_index(index: usize) -> NodeContinuation {
    NodeContinuation::from_usize((index / K_NUM_LENGTHS) % NC_COUNT)
}

/// Tesseract: `IsDawgFromBeamsIndex` (recodebeam.h:267).
fn is_dawg_from_beams_index(index: usize) -> bool {
    index / (K_NUM_LENGTHS * NC_COUNT) > 0
}

/// Computes a `beams_` index from its factors. Tesseract: `BeamIndex`
/// (recodebeam.h:271).
fn beam_index(is_dawg: bool, cont: NodeContinuation, length: usize) -> usize {
    ((is_dawg as usize) * NC_COUNT + cont as usize) * K_NUM_LENGTHS + length
}

/// Converts a probability to a certainty (log prob), clamped at the minimum.
/// Tesseract: `NetworkIO::ProbToCertainty` (networkio.cpp:582).
fn prob_to_certainty(prob: f32) -> f32 {
    if prob > K_MIN_PROB {
        prob.ln()
    } else {
        K_MIN_CERTAINTY
    }
}

// ---------------------------------------------------------------------------
// RecodeNode — a lattice element
// ---------------------------------------------------------------------------

/// Lattice element for the re-encode beam search. Tesseract: `struct RecodeNode`
/// (recodebeam.h:92). The borrowed `prev` pointer becomes an index into the
/// decode-lifetime arena ([`RecodeBeamSearch::arena`]); the owned dawg-position
/// vector is dropped with the deferred dictionary path.
#[derive(Clone, Debug)]
pub struct RecodeNode {
    /// The re-encoded code here = index into the network output.
    pub code: i32,
    /// The decoded unichar id — only valid for the final code of a sequence.
    pub unichar_id: i32,
    /// The permuter type active at this point (dawg permuters deferred).
    permuter: i32,
    /// True if this is the initial dawg state (deferred; always false here).
    start_of_dawg: bool,
    /// True if this is the first node in a dictionary word. Set for provenance
    /// but read only by the deferred dictionary beam.
    #[allow(dead_code)]
    start_of_word: bool,
    /// True if this represents a valid candidate end of word. Set for provenance
    /// but read only by the deferred dictionary beam.
    #[allow(dead_code)]
    end_of_word: bool,
    /// True if `code` is a duplicate of `prev.code`, crushed together via CTC.
    pub duplicate: bool,
    /// Certainty (log prob) of just this position.
    pub certainty: f32,
    /// Total certainty (score) of the path to this position.
    pub score: f32,
    /// The previous node in this chain — an index into the arena.
    prev: Option<usize>,
    /// A hash of all codes in the prefix and this code, for duplicate-path
    /// removal. Tesseract: `code_hash` (recodebeam.h:187).
    code_hash: u64,
}

// ---------------------------------------------------------------------------
// The best-path result
// ---------------------------------------------------------------------------

/// The best path as unichar ids with per-character certainty/rating and the
/// timestep (x-coordinate) at which each character starts. Tesseract:
/// the out-params of `ExtractBestPathAsUnicharIds` (recodebeam.cpp:223).
#[derive(Clone, Debug, Default)]
pub struct BestPath {
    /// One unichar id per recognized character (duplicates/nulls folded out).
    pub unichar_ids: Vec<UnicharId>,
    /// Per-character certainty (minimum log-prob along the character).
    pub certs: Vec<f32>,
    /// Per-character rating (negated sum of log-probs).
    pub ratings: Vec<f32>,
    /// Per-character start timestep, plus a trailing `width` sentinel.
    pub xcoords: Vec<i32>,
}

// ---------------------------------------------------------------------------
// A single timestep's beams
// ---------------------------------------------------------------------------

/// The data for a single time-step position: one heap per `beams_` index.
/// Tesseract: `struct RecodeBeam` (recodebeam.h:279). Each heap holds arena
/// indices; the best-initial-dawg choke point is deferred with the dict path.
#[derive(Clone)]
struct RecodeBeam {
    /// One heap (top-N set of arena indices) per `beams_` index. The heap keeps
    /// the WORST (lowest score) conceptually at the top for O(1) eviction, per
    /// Tesseract's `GenericHeap<KDPairInc>`; membership — not internal order — is
    /// what matters, so a `Vec` scanned for the min reproduces it exactly.
    beams: Vec<Vec<usize>>,
}

impl RecodeBeam {
    fn new() -> Self {
        RecodeBeam {
            beams: vec![Vec::new(); K_NUM_BEAMS],
        }
    }

    /// Tesseract: `RecodeBeam::Clear` (recodebeam.h:281).
    fn clear(&mut self) {
        for beam in &mut self.beams {
            beam.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// RecodeBeamSearch
// ---------------------------------------------------------------------------

/// The entire beam search for recognition of a text line. Tesseract:
/// `class RecodeBeamSearch` (recodebeam.h:194), raw (non-dictionary) subset.
///
/// Construct with [`RecodeBeamSearch::new`], run [`RecodeBeamSearch::decode`]
/// over the network's per-timestep output, then read the result with
/// [`RecodeBeamSearch::extract_best_path_as_unichar_ids`] /
/// [`RecodeBeamSearch::best_path_string`].
pub struct RecodeBeamSearch<'a> {
    /// The encoder/decoder in use. Tesseract: `recoder_` (recodebeam.h:410).
    recoder: &'a UnicharCompress,
    /// All lattice nodes, allocated for the lifetime of a decode. `prev` links
    /// are indices into this arena. Append-only: evicted nodes stay allocated
    /// (harmless) so surviving `prev` indices remain valid, matching the C++
    /// where referenced (previous-timestep) nodes are never freed mid-decode.
    arena: Vec<RecodeNode>,
    /// One [`RecodeBeam`] per timestep. Tesseract: `beam_` (recodebeam.h:412).
    beam: Vec<RecodeBeam>,
    /// Number of timesteps valid in `beam`. Tesseract: `beam_size_`.
    beam_size: usize,
    /// Top-n flag per output code, current timestep. Tesseract: `top_n_flags_`.
    top_n_flags: Vec<TopNState>,
    /// The highest-scoring code. Tesseract: `top_code_`.
    top_code: i32,
    /// The second-highest-scoring code. Tesseract: `second_code_`.
    second_code: i32,
    /// True if adjacent equal chars are NOT to be eliminated. Tesseract:
    /// `is_simple_text_`.
    is_simple_text: bool,
    /// The network class of the null/reject (CTC blank) character. Tesseract:
    /// `null_char_` — the recoder's *encoded* null code.
    null_char: i32,
}

impl<'a> RecodeBeamSearch<'a> {
    /// Borrows `recoder`, expected to outlive the search. `null_char` is the
    /// encoded null (CTC blank) code; `simple_text` disables duplicate collapse.
    ///
    /// Tesseract: `RecodeBeamSearch::RecodeBeamSearch` (recodebeam.cpp:58). The
    /// `Dict*` parameter is omitted — the dictionary beam is deferred, so
    /// `space_delimited_` is effectively always true here.
    pub fn new(recoder: &'a UnicharCompress, null_char: i32, simple_text: bool) -> Self {
        RecodeBeamSearch {
            recoder,
            arena: Vec::new(),
            beam: Vec::new(),
            beam_size: 0,
            top_n_flags: Vec::new(),
            top_code: -1,
            second_code: -1,
            is_simple_text: simple_text,
            null_char,
        }
    }

    /// Decodes the network output with the raw (non-dictionary) settings, exactly
    /// as `LSTMRecognizer::LabelsViaReEncode` drives it:
    /// `Decode(output, 1.0, 0.0, kMinCertainty, nullptr)` (lstmrecognizer.cpp:534).
    pub fn decode(&mut self, output: &NetworkIO) {
        self.decode_with(output, 1.0, 0.0, K_MIN_CERTAINTY as f64);
    }

    /// Decodes with explicit `dict_ratio`/`cert_offset`/`worst_dict_cert`, storing
    /// the lattice internally. Tesseract: `RecodeBeamSearch::Decode`
    /// (recodebeam.cpp:83), `charset`/`lstm_choice_mode` args dropped (deferred).
    pub fn decode_with(
        &mut self,
        output: &NetworkIO,
        dict_ratio: f64,
        cert_offset: f64,
        worst_dict_cert: f64,
    ) {
        self.beam_size = 0;
        self.arena.clear();
        let width = output.width();
        for t in 0..width {
            let outputs = output.f(t);
            self.compute_top_n(outputs, K_BEAM_WIDTHS[0]);
            self.decode_step(outputs, t, dict_ratio, cert_offset, worst_dict_cert);
        }
    }

    /// Fills `top_n_flags_` with the top-n status of each output; sets
    /// `top_code_`/`second_code_`. Tesseract: `ComputeTopN` (recodebeam.cpp:669).
    fn compute_top_n(&mut self, outputs: &[f32], top_n: usize) {
        let n = outputs.len();
        self.top_n_flags = vec![TN_ALSO_RAN; n];
        self.top_code = -1;
        self.second_code = -1;
        // A min-collection kept to size `top_n` of (value, index) — the top-n
        // highest outputs. `heap.PeekTop()` is the minimum key.
        let mut heap: Vec<(f32, usize)> = Vec::new();
        for (i, &v) in outputs.iter().enumerate() {
            let peek_min = heap.iter().map(|e| e.0).fold(f32::INFINITY, f32::min);
            if heap.len() < top_n || v > peek_min {
                heap.push((v, i));
                if heap.len() > top_n {
                    let pos = min_key_pos(&heap, |e| e.0);
                    heap.swap_remove(pos);
                }
            }
        }
        // Pop min-first; the last two popped are the two largest.
        while !heap.is_empty() {
            let pos = min_key_pos(&heap, |e| e.0);
            let (_, idx) = heap.swap_remove(pos);
            if heap.len() > 1 {
                self.top_n_flags[idx] = TN_TOPN;
            } else {
                self.top_n_flags[idx] = TN_TOP2;
                if heap.is_empty() {
                    self.top_code = idx as i32;
                } else {
                    self.second_code = idx as i32;
                }
            }
        }
        self.top_n_flags[self.null_char as usize] = TN_TOP2;
    }

    /// Adds the computation for the current time-step to the beam. Tesseract:
    /// `DecodeStep` (recodebeam.cpp:740). The dict branch and `best_initial_dawgs_`
    /// choke point are deferred.
    fn decode_step(
        &mut self,
        outputs: &[f32],
        t: usize,
        dict_ratio: f64,
        cert_offset: f64,
        worst_dict_cert: f64,
    ) {
        if t == self.beam.len() {
            self.beam.push(RecodeBeam::new());
        }
        self.beam[t].clear();
        self.beam_size = t + 1;
        if t == 0 {
            // The first step can only use singles and initials.
            self.continue_context(
                None,
                beam_index(false, NC_ANYTHING, 0),
                outputs,
                TN_TOP2,
                dict_ratio,
                cert_offset,
                worst_dict_cert,
                t,
            );
        } else {
            // Snapshot the previous step's node-index lists so the current step's
            // heaps (and the arena) can be mutated without aliasing. The indices
            // are stable (arena is append-only), so this is a faithful, cheap
            // stand-in for the C++ borrowed pointers into the frozen prev step.
            let prev = self.beam[t - 1].beams.clone();
            let mut total_beam = 0usize;
            // Work through the scores by group (top-2, top-n, the rest) while the
            // beam is empty, so the top-n results get first crack at the valid
            // codes and we only fall back to the rest if nothing matched.
            let mut tn = 0usize;
            while tn < TN_COUNT && total_beam == 0 {
                let top_n = tn_from_usize(tn);
                for (index, prev_list) in prev.iter().enumerate() {
                    // Working backwards, as Tesseract does.
                    for &prev_idx in prev_list.iter().rev() {
                        self.continue_context(
                            Some(prev_idx),
                            index,
                            outputs,
                            top_n,
                            dict_ratio,
                            cert_offset,
                            worst_dict_cert,
                            t,
                        );
                    }
                }
                total_beam = 0;
                for index in 0..K_NUM_BEAMS {
                    if continuation_from_beams_index(index) == NC_ANYTHING {
                        total_beam += self.beam[t].beams[index].len();
                    }
                }
                tn += 1;
            }
            // Dict best-initial choke point: deferred (no dictionary).
        }
    }

    /// Adds the legal (per recoder) continuations of context `prev` to the
    /// appropriate beams, using `outputs` for scores, restricted to codes with
    /// `top_n_flags_[code] == top_n_flag`. Tesseract: `ContinueContext`
    /// (recodebeam.cpp:888). `charset` enable/disable is dropped (deferred).
    #[allow(clippy::too_many_arguments)]
    fn continue_context(
        &mut self,
        prev_idx: Option<usize>,
        index: usize,
        outputs: &[f32],
        top_n_flag: TopNState,
        dict_ratio: f64,
        cert_offset: f64,
        worst_dict_cert: f64,
        step_t: usize,
    ) {
        let cert_offset = cert_offset as f32;
        let mut prefix = RecodedCharID::new();
        let mut full_code = RecodedCharID::new();
        let length = length_from_beams_index(index);
        let use_dawgs = is_dawg_from_beams_index(index);
        let prev_cont = continuation_from_beams_index(index);
        // Walk back `length` valid codes to rebuild the prefix, skipping the
        // duplicate and null nodes that are not part of the code sequence.
        let mut previous = prev_idx;
        for p in (0..length).rev() {
            // Skip duplicate/null nodes (guarded against a truncated chain).
            while let Some(pi) = previous {
                let n = &self.arena[pi];
                if n.duplicate || n.code == self.null_char {
                    previous = n.prev;
                } else {
                    break;
                }
            }
            let Some(pi) = previous else { break };
            let code = self.arena[pi].code;
            prefix.set(p, code);
            full_code.set(p, code);
            previous = self.arena[pi].prev;
        }

        if let Some(pi) = prev_idx
            && !self.is_simple_text
        {
            let prev_code = self.arena[pi].code;
            let prev_unichar = self.arena[pi].unichar_id;
            if self.top_n_flags[prev_code as usize] == top_n_flag {
                if prev_cont != NC_NO_DUP {
                    let cert = prob_to_certainty(outputs[prev_code as usize]) + cert_offset;
                    self.push_dup_or_no_dawg_if_better(
                        length,
                        true,
                        prev_code,
                        prev_unichar,
                        cert,
                        worst_dict_cert,
                        dict_ratio,
                        use_dawgs,
                        NC_ANYTHING,
                        prev_idx,
                        step_t,
                    );
                }
                if prev_cont == NC_ANYTHING && top_n_flag == TN_TOP2 && prev_code != self.null_char
                {
                    let cert = prob_to_certainty(
                        outputs[prev_code as usize] + outputs[self.null_char as usize],
                    ) + cert_offset;
                    self.push_dup_or_no_dawg_if_better(
                        length,
                        true,
                        prev_code,
                        prev_unichar,
                        cert,
                        worst_dict_cert,
                        dict_ratio,
                        use_dawgs,
                        NC_NO_DUP,
                        prev_idx,
                        step_t,
                    );
                }
            }
            if prev_cont == NC_ONLY_DUP {
                return;
            }
            if prev_code != self.null_char
                && length > 0
                && self.top_n_flags[self.null_char as usize] == top_n_flag
            {
                // Allow nulls within multi-code sequences (the nulls within a
                // sequence are not explicitly part of the code sequence).
                let cert = prob_to_certainty(outputs[self.null_char as usize]) + cert_offset;
                self.push_dup_or_no_dawg_if_better(
                    length,
                    false,
                    self.null_char,
                    INVALID_UNICHAR_ID,
                    cert,
                    worst_dict_cert,
                    dict_ratio,
                    use_dawgs,
                    NC_ANYTHING,
                    prev_idx,
                    step_t,
                );
            }
        }

        // Completions of the current prefix into a whole unichar.
        if let Some(final_codes) = self.recoder.get_final_codes(&prefix) {
            for &code in final_codes.to_vec().iter() {
                if self.top_n_flags[code as usize] != top_n_flag {
                    continue;
                }
                if let Some(pi) = prev_idx
                    && self.arena[pi].code == code
                    && !self.is_simple_text
                {
                    continue;
                }
                let cert = prob_to_certainty(outputs[code as usize]) + cert_offset;
                if cert < K_MIN_CERTAINTY && code != self.null_char {
                    continue;
                }
                full_code.set(length, code);
                let mut unichar_id = self.recoder.decode_unichar(&full_code);
                // Map the length-0 null char to INVALID.
                if length == 0 && code == self.null_char {
                    unichar_id = INVALID_UNICHAR_ID;
                }
                // charset enable/disable (whitelist/blacklist): deferred.
                self.continue_unichar(
                    code,
                    unichar_id,
                    cert,
                    worst_dict_cert,
                    dict_ratio,
                    use_dawgs,
                    NC_ANYTHING,
                    prev_idx,
                    step_t,
                );
                if top_n_flag == TN_TOP2 && code != self.null_char {
                    let mut prob = outputs[code as usize] + outputs[self.null_char as usize];
                    if let Some(pi) = prev_idx {
                        let prev_code = self.arena[pi].code;
                        if prev_cont == NC_ANYTHING
                            && prev_code != self.null_char
                            && ((prev_code == self.top_code && code == self.second_code)
                                || (code == self.top_code && prev_code == self.second_code))
                        {
                            prob += outputs[prev_code as usize];
                        }
                    }
                    let cert = prob_to_certainty(prob) + cert_offset;
                    self.continue_unichar(
                        code,
                        unichar_id,
                        cert,
                        worst_dict_cert,
                        dict_ratio,
                        use_dawgs,
                        NC_ONLY_DUP,
                        prev_idx,
                        step_t,
                    );
                }
            }
        }

        // Non-final extensions of the current prefix (partial multi-unit codes).
        if let Some(next_codes) = self.recoder.get_next_codes(&prefix) {
            for &code in next_codes.to_vec().iter() {
                if self.top_n_flags[code as usize] != top_n_flag {
                    continue;
                }
                if let Some(pi) = prev_idx
                    && self.arena[pi].code == code
                    && !self.is_simple_text
                {
                    continue;
                }
                let cert = prob_to_certainty(outputs[code as usize]) + cert_offset;
                self.push_dup_or_no_dawg_if_better(
                    length + 1,
                    false,
                    code,
                    INVALID_UNICHAR_ID,
                    cert,
                    worst_dict_cert,
                    dict_ratio,
                    use_dawgs,
                    NC_ANYTHING,
                    prev_idx,
                    step_t,
                );
                if top_n_flag == TN_TOP2 && code != self.null_char {
                    let mut prob = outputs[code as usize] + outputs[self.null_char as usize];
                    if let Some(pi) = prev_idx {
                        let prev_code = self.arena[pi].code;
                        if prev_cont == NC_ANYTHING
                            && prev_code != self.null_char
                            && ((prev_code == self.top_code && code == self.second_code)
                                || (code == self.top_code && prev_code == self.second_code))
                        {
                            prob += outputs[prev_code as usize];
                        }
                    }
                    let cert = prob_to_certainty(prob) + cert_offset;
                    self.push_dup_or_no_dawg_if_better(
                        length + 1,
                        false,
                        code,
                        INVALID_UNICHAR_ID,
                        cert,
                        worst_dict_cert,
                        dict_ratio,
                        use_dawgs,
                        NC_ONLY_DUP,
                        prev_idx,
                        step_t,
                    );
                }
            }
        }
    }

    /// Continues for a new unichar (non-dawg path). Tesseract: `ContinueUnichar`
    /// (recodebeam.cpp:1009). The `use_dawgs` branch is deferred (never taken in
    /// the raw decode).
    #[allow(clippy::too_many_arguments)]
    fn continue_unichar(
        &mut self,
        code: i32,
        unichar_id: i32,
        cert: f32,
        _worst_dict_cert: f64,
        dict_ratio: f64,
        use_dawgs: bool,
        cont: NodeContinuation,
        prev_idx: Option<usize>,
        step_t: usize,
    ) {
        if use_dawgs {
            // Dictionary beam deferred.
            return;
        }
        let heap_index = beam_index(false, cont, 0);
        let scaled = (cert as f64 * dict_ratio) as f32;
        self.push_heap_if_better(
            K_BEAM_WIDTHS[0],
            code,
            unichar_id,
            TOP_CHOICE_PERM,
            false,
            false,
            false,
            false,
            scaled,
            prev_idx,
            heap_index,
            step_t,
        );
        // dict_ != nullptr initial-dawg push: deferred.
    }

    /// Adds a partial-unichar or duplicate node to the appropriate beam if there
    /// is room or if better than the current worst. Tesseract:
    /// `PushDupOrNoDawgIfBetter` (recodebeam.cpp:1163). Non-dawg branch only.
    #[allow(clippy::too_many_arguments)]
    fn push_dup_or_no_dawg_if_better(
        &mut self,
        length: usize,
        dup: bool,
        code: i32,
        unichar_id: i32,
        cert: f32,
        _worst_dict_cert: f64,
        dict_ratio: f64,
        use_dawgs: bool,
        cont: NodeContinuation,
        prev_idx: Option<usize>,
        step_t: usize,
    ) {
        let index = beam_index(use_dawgs, cont, length);
        if use_dawgs {
            // Dictionary beam deferred.
            return;
        }
        let cert = (cert as f64 * dict_ratio) as f32;
        if cert >= K_MIN_CERTAINTY || code == self.null_char {
            let permuter = prev_idx
                .map(|pi| self.arena[pi].permuter)
                .unwrap_or(TOP_CHOICE_PERM);
            self.push_heap_if_better(
                K_BEAM_WIDTHS[length],
                code,
                unichar_id,
                permuter,
                false,
                false,
                false,
                dup,
                cert,
                prev_idx,
                index,
                step_t,
            );
        }
    }

    /// Adds a node to `heap` if there is room or if better than the current
    /// worst; updates in place if a matching node already exists. Tesseract:
    /// `PushHeapIfBetter` (recodebeam.cpp:1187) + `UpdateHeapIfMatched`
    /// (recodebeam.cpp:1234).
    #[allow(clippy::too_many_arguments)]
    fn push_heap_if_better(
        &mut self,
        max_size: usize,
        code: i32,
        unichar_id: i32,
        permuter: i32,
        dawg_start: bool,
        word_start: bool,
        end: bool,
        dup: bool,
        cert: f32,
        prev_idx: Option<usize>,
        heap_index: usize,
        step_t: usize,
    ) {
        let prev_score = prev_idx.map(|pi| self.arena[pi].score).unwrap_or(0.0);
        let score = cert + prev_score;
        let heap_len = self.beam[step_t].beams[heap_index].len();
        let worst = self.heap_worst_score(heap_index, step_t);
        if heap_len < max_size || worst.map(|w| score > w).unwrap_or(true) {
            let hash = self.compute_code_hash(code, dup, prev_idx);
            let node = RecodeNode {
                code,
                unichar_id,
                permuter,
                start_of_dawg: dawg_start,
                start_of_word: word_start,
                end_of_word: end,
                duplicate: dup,
                certainty: cert,
                score,
                prev: prev_idx,
                code_hash: hash,
            };
            // UpdateHeapIfMatched: replace an existing matching node if better.
            if let Some(match_idx) =
                self.find_match(heap_index, step_t, code, hash, permuter, dawg_start)
            {
                if score > self.arena[match_idx].score {
                    self.arena[match_idx] = node;
                }
                return;
            }
            let new_idx = self.arena.len();
            self.arena.push(node);
            self.beam[step_t].beams[heap_index].push(new_idx);
            if self.beam[step_t].beams[heap_index].len() > max_size {
                // Pop the worst (minimum-score) element.
                let pos = self.heap_worst_pos(heap_index, step_t);
                self.beam[step_t].beams[heap_index].swap_remove(pos);
            }
        }
    }

    /// The minimum score in a heap (`PeekTop().data().score`), or `None` if empty.
    fn heap_worst_score(&self, heap_index: usize, step_t: usize) -> Option<f32> {
        let heap = &self.beam[step_t].beams[heap_index];
        heap.iter()
            .map(|&i| self.arena[i].score)
            .fold(None, |acc, s| match acc {
                None => Some(s),
                Some(m) => Some(if s < m { s } else { m }),
            })
    }

    /// The position (within the heap vec) of the minimum-score element.
    fn heap_worst_pos(&self, heap_index: usize, step_t: usize) -> usize {
        let heap = &self.beam[step_t].beams[heap_index];
        let mut best_pos = 0;
        let mut best = f32::INFINITY;
        for (k, &i) in heap.iter().enumerate() {
            let s = self.arena[i].score;
            if s < best {
                best = s;
                best_pos = k;
            }
        }
        best_pos
    }

    /// Finds a heap entry matching the given node identity (code, hash, permuter,
    /// dawg-start), for the in-place update. Tesseract: the scan in
    /// `UpdateHeapIfMatched` (recodebeam.cpp:1234).
    fn find_match(
        &self,
        heap_index: usize,
        step_t: usize,
        code: i32,
        hash: u64,
        permuter: i32,
        dawg_start: bool,
    ) -> Option<usize> {
        for &idx in &self.beam[step_t].beams[heap_index] {
            let n = &self.arena[idx];
            if n.code == code
                && n.code_hash == hash
                && n.permuter == permuter
                && n.start_of_dawg == dawg_start
            {
                return Some(idx);
            }
        }
        None
    }

    /// Computes the rolling code-hash for duplicate-path removal. Tesseract:
    /// `ComputeCodeHash` (recodebeam.cpp:1259). Uses wrapping unsigned arithmetic
    /// to match the C++ `uint64_t` overflow behavior exactly.
    fn compute_code_hash(&self, code: i32, dup: bool, prev_idx: Option<usize>) -> u64 {
        let mut hash = prev_idx.map(|pi| self.arena[pi].code_hash).unwrap_or(0);
        if !dup && code != self.null_char {
            let num_classes = self.recoder.code_range() as u64;
            let carry = ((hash >> 32).wrapping_mul(num_classes)) >> 32;
            hash = hash.wrapping_mul(num_classes);
            hash = hash.wrapping_add(carry);
            hash = hash.wrapping_add(code as u64);
        }
        hash
    }

    // -----------------------------------------------------------------------
    // Result extraction
    // -----------------------------------------------------------------------

    /// Backtracks to the best path through the lattice. Tesseract:
    /// `ExtractBestPaths` (recodebeam.cpp:1276) — best node only; the second-best
    /// path and the dawg-validity gate (empty here) are deferred.
    fn extract_best_path(&self) -> Vec<usize> {
        if self.beam_size == 0 {
            return Vec::new();
        }
        let last = &self.beam[self.beam_size - 1];
        let mut best: Option<usize> = None;
        for &c in &[NC_ANYTHING, NC_NO_DUP] {
            // Skips NC_ONLY_DUP, which cannot be a valid end of path.
            for is_dawg in 0..2 {
                let bi = beam_index(is_dawg == 1, c, 0);
                for &node_idx in &last.beams[bi] {
                    // is_dawg heaps are empty in the raw path (dict deferred), so
                    // the dawg-validity scan never runs; kept structurally.
                    let score = self.arena[node_idx].score;
                    match best {
                        None => best = Some(node_idx),
                        Some(b) if score > self.arena[b].score => best = Some(node_idx),
                        _ => {}
                    }
                }
            }
        }
        match best {
            Some(b) => self.extract_path(b),
            None => Vec::new(),
        }
    }

    /// Backtracks from `node`, returning the path in forward order. Tesseract:
    /// `ExtractPath` (recodebeam.cpp:1327).
    fn extract_path(&self, node: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut cur = Some(node);
        while let Some(i) = cur {
            path.push(i);
            cur = self.arena[i].prev;
        }
        path.reverse();
        path
    }

    /// Returns the best path as labels/xcoords, similar to simple CTC. Tesseract:
    /// `ExtractBestPathAsLabels` (recodebeam.cpp:200).
    pub fn extract_best_path_as_labels(&self) -> (Vec<i32>, Vec<i32>) {
        let mut labels = Vec::new();
        let mut xcoords = Vec::new();
        let best = self.extract_best_path();
        let width = best.len();
        let mut t = 0;
        while t < width {
            let label = self.arena[best[t]].code;
            if label != self.null_char {
                labels.push(label);
                xcoords.push(t as i32);
            }
            t += 1;
            while t < width && !self.is_simple_text && self.arena[best[t]].code == label {
                t += 1;
            }
        }
        xcoords.push(width as i32);
        (labels, xcoords)
    }

    /// Returns the best path as unichar ids / certs / ratings / xcoords, skipping
    /// duplicates, nulls, and intermediate (partial) parts. Tesseract:
    /// `ExtractBestPathAsUnicharIds` (recodebeam.cpp:223) over
    /// `ExtractPathAsUnicharIds` (recodebeam.cpp:565). Char-boundary/box output
    /// is deferred.
    pub fn extract_best_path_as_unichar_ids(&self) -> BestPath {
        let best = self.extract_best_path();
        self.extract_path_as_unichar_ids(&best)
    }

    /// The reassembly: folds the raw lattice path into unichar ids. Tesseract:
    /// `ExtractPathAsUnicharIds` (recodebeam.cpp:565), certainty/rating logic
    /// computed in `f64` (matching the C++ `double` locals) and stored as `f32`.
    fn extract_path_as_unichar_ids(&self, best: &[usize]) -> BestPath {
        let mut out = BestPath::default();
        let width = best.len();
        let mut t = 0usize;
        while t < width {
            let mut certainty = 0.0f64;
            let mut rating = 0.0f64;
            // Skip intermediate (partial/null) nodes, accumulating their scores.
            while t < width && self.arena[best[t]].unichar_id == INVALID_UNICHAR_ID {
                let cert = self.arena[best[t]].certainty as f64;
                t += 1;
                if cert < certainty {
                    certainty = cert;
                }
                rating -= cert;
            }
            if t < width {
                let unichar_id = self.arena[best[t]].unichar_id;
                if unichar_id == UNICHAR_SPACE
                    && !out.certs.is_empty()
                    && self.arena[best[t]].permuter != NO_PERM
                {
                    // All the rating/certainty go on the previous character except
                    // for the space itself.
                    if (certainty as f32) < *out.certs.last().unwrap() {
                        *out.certs.last_mut().unwrap() = certainty as f32;
                    }
                    *out.ratings.last_mut().unwrap() += rating as f32;
                    certainty = 0.0;
                    rating = 0.0;
                }
                out.unichar_ids.push(unichar_id);
                out.xcoords.push(t as i32);
                // Consume this node and any trailing duplicates (CTC collapse).
                loop {
                    let node = &self.arena[best[t]];
                    let cert = node.certainty as f64;
                    let perm_no = node.permuter == NO_PERM;
                    t += 1;
                    if cert < certainty || (unichar_id == UNICHAR_SPACE && perm_no) {
                        certainty = cert;
                    }
                    rating -= cert;
                    if !(t < width && self.arena[best[t]].duplicate) {
                        break;
                    }
                }
                out.certs.push(certainty as f32);
                out.ratings.push(rating as f32);
            } else if !out.certs.is_empty() {
                if (certainty as f32) < *out.certs.last().unwrap() {
                    *out.certs.last_mut().unwrap() = certainty as f32;
                }
                *out.ratings.last_mut().unwrap() += rating as f32;
            }
        }
        out.xcoords.push(width as i32);
        out
    }

    /// The best path as a UTF-8 string, via the unicharset. Convenience over
    /// [`RecodeBeamSearch::extract_best_path_as_unichar_ids`] +
    /// `Unicharset::id_to_unichar` (mirrors how the unit test rebuilds `decoded`).
    pub fn best_path_string(&self, unicharset: &Unicharset) -> String {
        let path = self.extract_best_path_as_unichar_ids();
        let mut s = String::new();
        for id in path.unichar_ids {
            if id != INVALID_UNICHAR_ID {
                s.push_str(unicharset.id_to_unichar(id));
            }
        }
        s
    }
}

/// The position of the minimum-key element in a slice.
fn min_key_pos<T, F: Fn(&T) -> f32>(items: &[T], key: F) -> usize {
    let mut best_pos = 0;
    let mut best = f32::INFINITY;
    for (k, item) in items.iter().enumerate() {
        let v = key(item);
        if v < best {
            best = v;
            best_pos = k;
        }
    }
    best_pos
}

/// Convenience: run a raw (non-dictionary) decode and return the unichar ids.
pub fn decode_to_unichar_ids(
    recoder: &UnicharCompress,
    null_char: i32,
    simple_text: bool,
    output: &NetworkIO,
) -> Vec<UnicharId> {
    let mut search = RecodeBeamSearch::new(recoder, null_char, simple_text);
    search.decode(output);
    search.extract_best_path_as_unichar_ids().unichar_ids
}

/// Convenience: run a raw (non-dictionary) decode and return the UTF-8 string.
pub fn decode_to_string(
    recoder: &UnicharCompress,
    null_char: i32,
    simple_text: bool,
    output: &NetworkIO,
    unicharset: &Unicharset,
) -> String {
    let mut search = RecodeBeamSearch::new(recoder, null_char, simple_text);
    search.decode(output);
    search.best_path_string(unicharset)
}

#[cfg(test)]
mod tests;
