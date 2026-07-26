//! Hermetic tests for the recode/CTC beam-search decoder. Every fixture is a
//! hand-built synthetic softmax [`NetworkIO`] over a tiny [`UnicharCompress`]
//! (four single-unit codes + one two-unit CJK-style code), so the whole decode
//! path runs with no downloads. The recoder mirrors Tesseract's convention that
//! the null (CTC blank) is the *encoded* null code (here unit 3), a valid
//! single-unit code in the empty-prefix `final_codes`.

use super::*;
use crate::networkio::NetworkIO;
use crate::unicharcompress::{RecodedCharID, UnicharCompress};
use crate::unicharset::Unicharset;

/// Feature depth = recoder code range (units 0..=5).
const NF: usize = 6;
/// The encoded null / CTC-blank code.
const NULL: i32 = 3;

// Code units: 0=space, 1='a', 2='b', 3=null, 4/5 = the two units of '好'.
fn recoder() -> UnicharCompress {
    let single = |v: i32| {
        let mut c = RecodedCharID::new();
        c.set(0, v);
        c
    };
    let mut codes = vec![
        single(0), // id 0: space
        single(1), // id 1: 'a'
        single(2), // id 2: 'b'
        single(3), // id 3: null unichar (encodes to unit 3)
    ];
    let mut cjk = RecodedCharID::new(); // id 4: '好' = [4, 5]
    cjk.set(0, 4);
    cjk.set(1, 5);
    codes.push(cjk);
    let mut r = UnicharCompress::new();
    r.setup_direct(codes);
    assert_eq!(r.code_range(), NF as i32);
    r
}

fn unicharset() -> Unicharset {
    let text = "5\nNULL 0 Latin 0\na 3 Latin 1\nb 3 Latin 2\n~ 10 Common 3\n\u{597d} 1 Han 4\n";
    Unicharset::load_from_bytes(text.as_bytes(), false).expect("load unicharset")
}

/// One timestep row: the given (code, prob) pairs, everything else 0.
fn row(pairs: &[(usize, f32)]) -> Vec<f32> {
    let mut r = vec![0.0f32; NF];
    for &(i, v) in pairs {
        r[i] = v;
    }
    r
}

fn io(frames: &[Vec<f32>]) -> NetworkIO {
    let width = frames.len();
    let mut data = Vec::with_capacity(width * NF);
    for f in frames {
        data.extend_from_slice(f);
    }
    NetworkIO::from_rows(width, NF, data)
}

fn decode_ids(frames: &[Vec<f32>]) -> Vec<i32> {
    let rec = recoder();
    decode_to_unichar_ids(&rec, NULL, false, &io(frames))
}

fn decode_str(frames: &[Vec<f32>]) -> String {
    let rec = recoder();
    let uni = unicharset();
    decode_to_string(&rec, NULL, false, &io(frames), &uni)
}

// ---------------------------------------------------------------------------

#[test]
fn single_char_decodes() {
    // One clear 'a' frame -> id 1 -> "a".
    let frames = [row(&[(1, 0.9), (NULL as usize, 0.05)])];
    assert_eq!(decode_ids(&frames), vec![1]);
    assert_eq!(decode_str(&frames), "a");
}

#[test]
fn ctc_blank_collapses_repeats() {
    // Classic CTC: a a <blank> a a -> "aa". The null between the two runs of 'a'
    // forces two separate characters; the adjacent equal frames collapse.
    let a = || row(&[(1, 0.9), (NULL as usize, 0.05)]);
    let n = || row(&[(NULL as usize, 0.9), (1, 0.05)]);
    let frames = [a(), a(), n(), a(), a()];
    assert_eq!(decode_ids(&frames), vec![1, 1], "a a <blank> a a -> aa");
    assert_eq!(decode_str(&frames), "aa");
}

#[test]
fn adjacent_equal_without_blank_is_one_char() {
    // Without a separating blank, a run of equal frames is a single character.
    let a = || row(&[(1, 0.9), (NULL as usize, 0.05)]);
    let frames = [a(), a(), a()];
    assert_eq!(decode_ids(&frames), vec![1], "aaa (no blank) -> a");
    assert_eq!(decode_str(&frames), "a");
}

#[test]
fn multi_unit_cjk_reassembles_to_one_id() {
    // The two-unit code [4, 5] must fold back into the single '好' unichar id 4.
    let frames = [
        row(&[(4, 0.9), (NULL as usize, 0.05)]),
        row(&[(5, 0.9), (NULL as usize, 0.05)]),
    ];
    assert_eq!(decode_ids(&frames), vec![4], "[4,5] -> one CJK id");
    assert_eq!(decode_str(&frames), "\u{597d}");
}

#[test]
fn multi_unit_cjk_with_interior_null() {
    // A null inside the multi-code sequence is allowed and dropped: [4] <null> [5]
    // still reassembles to the single '好' id.
    let frames = [
        row(&[(4, 0.9), (NULL as usize, 0.05)]),
        row(&[(NULL as usize, 0.9), (4, 0.05)]),
        row(&[(5, 0.9), (NULL as usize, 0.05)]),
    ];
    assert_eq!(decode_ids(&frames), vec![4]);
    assert_eq!(decode_str(&frames), "\u{597d}");
}

#[test]
fn word_of_several_chars() {
    // "aba": a <blank> b <blank> a, each separated so all three survive.
    let a = || row(&[(1, 0.9), (NULL as usize, 0.05)]);
    let b = || row(&[(2, 0.9), (NULL as usize, 0.05)]);
    let n = || row(&[(NULL as usize, 0.9), (1, 0.02)]);
    let frames = [a(), n(), b(), n(), a()];
    assert_eq!(decode_ids(&frames), vec![1, 2, 1]);
    assert_eq!(decode_str(&frames), "aba");
}

#[test]
fn beam_beats_per_frame_argmax() {
    // A case where per-frame argmax gives a different, lower-joint-probability
    // result than the beam. Frames:
    //   t0: a=0.90 (clear)
    //   t1: b=0.45 > a=0.35 > null=0.20
    // Per-frame argmax is [a, b] -> "ab". But the beam finds that folding t1 into
    // a duplicate 'a' (using the a+null combined score, log(0.55)) outscores
    // starting a fresh 'b' (log(0.9)+log(0.45)), so the top path is just "a".
    let frames = [
        row(&[(1, 0.90), (2, 0.05), (NULL as usize, 0.05)]),
        row(&[(2, 0.45), (1, 0.35), (NULL as usize, 0.20)]),
    ];

    // Per-frame argmax of the raw softmax.
    let argmax: Vec<i32> = frames
        .iter()
        .map(|f| {
            let mut bi = 0usize;
            for i in 1..NF {
                if f[i] > f[bi] {
                    bi = i;
                }
            }
            bi as i32
        })
        .collect();
    assert_eq!(argmax, vec![1, 2], "greedy per-frame argmax is a then b");

    // The beam collapses to a single 'a'.
    assert_eq!(
        decode_ids(&frames),
        vec![1],
        "beam finds the higher-joint 'a'"
    );
    assert_eq!(decode_str(&frames), "a");
}

#[test]
fn empty_input_is_empty_string() {
    let frames: [Vec<f32>; 0] = [];
    assert_eq!(decode_ids(&frames), Vec::<i32>::new());
    assert_eq!(decode_str(&frames), "");
}

#[test]
fn all_blank_input_is_empty_string() {
    let n = || row(&[(NULL as usize, 0.9), (1, 0.02)]);
    let frames = [n(), n(), n()];
    assert_eq!(
        decode_ids(&frames),
        Vec::<i32>::new(),
        "all blanks -> nothing"
    );
    assert_eq!(decode_str(&frames), "");
}

#[test]
fn labels_keep_codes_but_ids_fold() {
    // ExtractBestPathAsLabels keeps the raw (recoded) code labels, dropping only
    // nulls and adjacent duplicates; the unichar-id path folds [4,5] -> one id.
    let frames = [
        row(&[(4, 0.9), (NULL as usize, 0.05)]),
        row(&[(5, 0.9), (NULL as usize, 0.05)]),
    ];
    let rec = recoder();
    let mut search = RecodeBeamSearch::new(&rec, NULL, false);
    search.decode(&io(&frames));
    let (labels, xcoords) = search.extract_best_path_as_labels();
    assert_eq!(labels, vec![4, 5], "labels are the raw code units");
    // xcoords has one entry per label plus the trailing width sentinel.
    assert_eq!(*xcoords.last().unwrap(), frames.len() as i32);
    let ids = search.extract_best_path_as_unichar_ids().unichar_ids;
    assert_eq!(ids, vec![4], "ids reassemble the two units into '好'");
}

#[test]
fn beam_index_arithmetic_round_trips() {
    // Guards the beams_ index packing against the constants.
    assert_eq!(K_NUM_LENGTHS, RecodedCharID::MAX_CODE_LEN + 1);
    assert_eq!(K_NUM_BEAMS, 2 * NC_COUNT * K_NUM_LENGTHS);
    for is_dawg in [false, true] {
        for c in 0..NC_COUNT {
            let cont = NodeContinuation::from_usize(c);
            for length in 0..K_NUM_LENGTHS {
                let idx = beam_index(is_dawg, cont, length);
                assert_eq!(length_from_beams_index(idx), length);
                assert_eq!(continuation_from_beams_index(idx), cont);
                assert_eq!(is_dawg_from_beams_index(idx), is_dawg);
            }
        }
    }
}
