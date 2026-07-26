//! Hermetic tests for the LSTM line driver (Phase 6). A synthetic `.traineddata`
//! buffer is assembled in memory from three components — a tiny
//! `Series[FullyConnected(Softmax)]` network, a 3-entry unicharset, and a
//! matching recoder — so `LstmRecognizer::load` + `recognize_line` run
//! end-to-end without downloading a real model. The scale-to-height +
//! adaptive-black/white normalization is checked directly on `prepare_input`.

use super::*;
use crate::network::NetworkType;
use crate::tessdata::{NUM_ENTRIES, TessdataType};
use crate::test_support::Writer;

// ---------------------------------------------------------------------------
// Synthetic .traineddata assembly
// ---------------------------------------------------------------------------

/// Serialize a recoder's `encoder_` vector (native endian, no swap) exactly as
/// `RecodedCharID::Serialize` + the vector length prefix, for the `LstmRecoder`
/// component. Each code is `self_normalized (u8) · length (u32) · code (i32[])`.
fn build_recoder_component(codes: &[Vec<i32>]) -> Vec<u8> {
    let mut w = Writer::new();
    w.u32(codes.len() as u32);
    for c in codes {
        w.u8(1); // self_normalized
        w.u32(c.len() as u32);
        for &u in c {
            w.i32(u);
        }
    }
    w.bytes
}

/// Build an `Lstm` component: `Series[ FullyConnected(Softmax) ]` with the given
/// dims + row-major weights (rows = `no`, cols = `ni + 1`, trailing bias), then
/// the trailing scalar block (network_str, training_flags = recoding,
/// iteration counters, null_char). The floats after null_char are not read by
/// `load`, so they are omitted.
fn build_lstm_component(ni: i32, no: i32, weights: &[f64], null_char: i32) -> Vec<u8> {
    let mut fc = Writer::new();
    fc.header(NetworkType::Softmax, ni, no, "fc");
    fc.weight_matrix_float(no, ni + 1, weights);

    let mut series = Writer::new();
    series.header(NetworkType::Series, ni, no, "root");
    series.u32(1); // stack size
    series.bytes.extend_from_slice(&fc.bytes);

    // Scalar params (lstmrecognizer.cpp:144): network_str, training_flags,
    // training_iteration, sample_iteration, null_char.
    series.string(""); // network_str_
    series.i32(64); // training_flags_ = TF_COMPRESS_UNICHARSET
    series.i32(0); // training_iteration_
    series.i32(0); // sample_iteration_
    series.i32(null_char); // null_char_
    series.bytes
}

/// A 3-entry unicharset: id0 = space, id1 = "A", id2 = "B".
fn unicharset_text() -> Vec<u8> {
    let mut s = String::new();
    s.push_str("3\n");
    s.push_str("NULL 0 NULL 0\n");
    s.push_str("A 5 0,255,0,255,0,0,0,0,0,0 Latin 3 0 3 A\n");
    s.push_str("B 5 0,255,0,255,0,0,0,0,0,0 Latin 3 0 3 B\n");
    s.into_bytes()
}

/// Assemble a proprietary `.traineddata` buffer (native endian, no swap) with
/// the Lstm / LstmUnicharset / LstmRecoder components at their real slots,
/// mirroring `TessdataManager::Serialize`'s ascending-offset layout.
fn build_traineddata(lstm: &[u8], unicharset: &[u8], recoder: &[u8]) -> Vec<u8> {
    let header_len = 4 + NUM_ENTRIES * 8;
    // Ascending slot order: Lstm(17), LstmUnicharset(21), LstmRecoder(22).
    let present: [(usize, &[u8]); 3] = [
        (TessdataType::Lstm as usize, lstm),
        (TessdataType::LstmUnicharset as usize, unicharset),
        (TessdataType::LstmRecoder as usize, recoder),
    ];
    let mut offset_table = [-1i64; NUM_ENTRIES];
    let mut cursor = header_len as i64;
    for (slot, payload) in present {
        offset_table[slot] = cursor;
        cursor += payload.len() as i64;
    }
    let mut buf = Vec::new();
    buf.extend_from_slice(&(NUM_ENTRIES as u32).to_ne_bytes());
    for &off in &offset_table {
        buf.extend_from_slice(&off.to_ne_bytes());
    }
    for (_, payload) in present {
        buf.extend_from_slice(payload);
    }
    buf
}

// ---------------------------------------------------------------------------
// load + recognize_line, end to end
// ---------------------------------------------------------------------------

#[test]
fn loads_synthetic_recognizer_and_recognizes_two_chars() {
    // Recoder: pass-through-style single-unit codes for ids 0,1,2 -> code_range 3.
    let recoder = build_recoder_component(&[vec![0], vec![1], vec![2]]);
    // Network: ni = 2 (input height), no = code_range + 1 = 4 (CTC null = 3).
    // Softmax weights (rows = no = 4, cols = ni + 1 = 3), trailing bias:
    //   class0 (space): 0
    //   class1 ("A"):   G*x0 - G*x1
    //   class2 ("B"):   G*x1 - G*x0
    //   class3 (null):  0
    // Sign of (x0 - x1) selects A vs B; class0/null never win.
    let g = 5.0;
    #[rustfmt::skip]
    let weights: Vec<f64> = vec![
        0.0, 0.0, 0.0,   // space
        g,  -g,   0.0,   // A
        -g,  g,   0.0,   // B
        0.0, 0.0, 0.0,   // null
    ];
    let lstm = build_lstm_component(2, 4, &weights, 3);
    let uni = unicharset_text();
    let recoder_comp = build_recoder_component(&[vec![0], vec![1], vec![2]]);
    assert_eq!(recoder, recoder_comp);

    let buf = build_traineddata(&lstm, &uni, &recoder_comp);
    let mgr = TessdataManager::from_bytes(&buf).unwrap();
    let rec = LstmRecognizer::load(&mgr).expect("load synthetic recognizer");

    assert_eq!(rec.num_inputs(), 2);
    assert_eq!(rec.num_outputs(), 4);
    assert_eq!(rec.null_char(), 3);
    assert!(rec.is_recoding());
    assert_eq!(rec.unicharset().id_to_unichar(1), "A");

    // A 2-row, 6-col line: left half favors row0 (A), right half favors row1 (B).
    //   row0: 255 255 255   0   0   0
    //   row1:   0   0   0 255 255 255
    let mut pixels = vec![255u8, 255, 255, 0, 0, 0]; // row 0
    pixels.extend_from_slice(&[0u8, 0, 0, 255, 255, 255]); // row 1
    let line = GrayLine::new(6, 2, pixels);

    let text = rec.recognize_line(&line).expect("recognize");
    assert_eq!(text, "AB", "left columns decode A, right columns decode B");
}

#[test]
fn solid_color_line_recognizes_without_error() {
    // A uniform line has no contrast; recognition must still run (contrast is
    // clamped to 1) and not panic. With this network a flat input yields
    // logit1 == logit2 == 0 == logit0, so the beam may pick space/null; we only
    // assert it produces a (possibly empty) string.
    let recoder = build_recoder_component(&[vec![0], vec![1], vec![2]]);
    let weights: Vec<f64> = vec![
        0.0, 0.0, 0.0, //
        5.0, -5.0, 0.0, //
        -5.0, 5.0, 0.0, //
        0.0, 0.0, 0.0,
    ];
    let lstm = build_lstm_component(2, 4, &weights, 3);
    let buf = build_traineddata(&lstm, &unicharset_text(), &recoder);
    let mgr = TessdataManager::from_bytes(&buf).unwrap();
    let rec = LstmRecognizer::load(&mgr).unwrap();

    let line = GrayLine::new(8, 2, vec![128u8; 16]);
    let _ = rec.recognize_line(&line).expect("recognize solid line");
}

#[test]
fn mismatched_network_and_recoder_is_rejected() {
    // Recoder code_range = 3 (codes 0,1,2) requires network no = 4, but the
    // network declares no = 5: the CTC-null invariant must fail the load.
    let recoder = build_recoder_component(&[vec![0], vec![1], vec![2]]);
    let weights: Vec<f64> = vec![0.0; 5 * 3]; // rows = no = 5, cols = ni + 1 = 3
    let lstm = build_lstm_component(2, 5, &weights, 4);
    let buf = build_traineddata(&lstm, &unicharset_text(), &recoder);
    let mgr = TessdataManager::from_bytes(&buf).unwrap();
    let err = LstmRecognizer::load(&mgr).unwrap_err();
    assert_eq!(err.kind(), crate::error::ErrorKind::Format);
}

#[test]
fn missing_lstm_component_errors() {
    // A container with only the unicharset + recoder, no Lstm component.
    let recoder = build_recoder_component(&[vec![0], vec![1], vec![2]]);
    let buf = build_traineddata(&[], &unicharset_text(), &recoder);
    let mgr = TessdataManager::from_bytes(&buf).unwrap();
    // An empty Lstm payload is collapsed to "absent" by the container parser.
    let err = LstmRecognizer::load(&mgr).unwrap_err();
    assert_eq!(err.kind(), crate::error::ErrorKind::Format);
}

// ---------------------------------------------------------------------------
// Input normalization (prepare_input / scale-to-height / black-white)
// ---------------------------------------------------------------------------

#[test]
fn normalization_scales_to_target_height_and_preserves_aspect() {
    // 10x5 image scaled to height 4 -> width round(10*4/5) = 8.
    let line = GrayLine::new(10, 5, vec![100u8; 50]);
    let io = super::prepare_input(4, &line);
    assert_eq!(io.num_features(), 4, "features == target height");
    assert_eq!(io.width(), 8, "timesteps = round(width * target/height)");

    // Doubling the source width doubles the timesteps (aspect ratio preserved).
    let wide = GrayLine::new(20, 5, vec![100u8; 100]);
    let io_wide = super::prepare_input(4, &wide);
    assert_eq!(io_wide.width(), 16);
    assert_eq!(io_wide.num_features(), 4);
}

#[test]
fn normalization_identity_when_already_at_target_height() {
    // height already equals target -> no vertical scale, width unchanged.
    let line = GrayLine::new(6, 4, vec![50u8; 24]);
    let io = super::prepare_input(4, &line);
    assert_eq!(io.width(), 6);
    assert_eq!(io.num_features(), 4);
}

#[test]
fn solid_color_line_yields_uniform_networkio() {
    let line = GrayLine::new(9, 3, vec![200u8; 27]);
    let io = super::prepare_input(4, &line);
    let first = io.f(0)[0];
    for t in 0..io.width() {
        for &v in io.f(t) {
            assert!(
                (v - first).abs() < 1e-6,
                "a solid-color line must normalize to a uniform buffer"
            );
        }
    }
}

#[test]
fn degenerate_line_yields_empty_networkio() {
    let io = super::prepare_input(4, &GrayLine::new(0, 0, Vec::new()));
    assert_eq!(io.width(), 0);
}

// ---------------------------------------------------------------------------
// Optional real-model smoke test (ignored; downloads nothing)
// ---------------------------------------------------------------------------

/// Loads a real `eng.traineddata` (or any LSTM model) and checks the dimensions
/// are sane. Ignored by default; runs only when `KOPITIAM_TEST_TRAINEDDATA`
/// names a file the developer already has.
#[test]
#[ignore = "set KOPITIAM_TEST_TRAINEDDATA to a real LSTM .traineddata to run"]
fn loads_real_traineddata() {
    let path = std::env::var("KOPITIAM_TEST_TRAINEDDATA")
        .expect("KOPITIAM_TEST_TRAINEDDATA must point at a real .traineddata");
    let data = std::fs::read(&path).expect("read traineddata");
    let mgr = TessdataManager::from_bytes(&data).expect("parse traineddata");
    let rec = LstmRecognizer::load(&mgr).expect("load real recognizer");
    // Input height is the scaled line height (typically 36); outputs cover the
    // recoded alphabet plus the CTC null.
    assert!(
        rec.num_inputs() > 0 && rec.num_inputs() <= 128,
        "input height should be a sane line height, got {}",
        rec.num_inputs()
    );
    assert!(
        rec.num_outputs() > 1,
        "a real model has many output classes, got {}",
        rec.num_outputs()
    );
    assert!(
        rec.unicharset().size() > 3,
        "a real unicharset is non-trivial, got {}",
        rec.unicharset().size()
    );
    // A real line can be recognized without panicking (content is arbitrary).
    let line = GrayLine::new(64, rec.num_inputs().max(1) as usize, vec![255u8; 64 * rec.num_inputs().max(1) as usize]);
    let _ = rec.recognize_line(&line).expect("recognize a blank line");
}
