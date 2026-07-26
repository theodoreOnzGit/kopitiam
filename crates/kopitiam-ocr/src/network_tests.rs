//! End-to-end tests for the LSTM recognizer network: deserialize a synthetic
//! (hermetic, no downloads) model buffer and run a forward pass, plus the
//! `#[ignore]`d integration test against a real `.traineddata` file gated on
//! `KOPITIAM_TEST_TRAINEDDATA`.

use crate::network::{Network, NetworkType};
use crate::networkio::NetworkIO;
use crate::test_support::Writer;

/// sigmoid/tanh references matching `Tensor::lstm_cell`.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn series_of_softmax_fully_connected_deserializes_and_forwards() {
    // Series[ FullyConnected(Softmax) ] with ni=2, no=3.
    // Weights [no=3, cols=ni+1=3]; last column of each row is the bias.
    //   row0: [1, 0, 0]   row1: [0, 1, 0]   row2: [0, 0, 0]
    let dense: Vec<f64> = vec![
        1.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0,
    ];
    let mut fc = Writer::new();
    fc.header(NetworkType::Softmax, 2, 3, "fc");
    fc.weight_matrix_float(3, 3, &dense);

    let mut series = Writer::new();
    series.header(NetworkType::Series, 2, 3, "root");
    series.u32(1); // stack size
    series.bytes.extend_from_slice(&fc.bytes);

    let mut fp = crate::TFile::new(&series.bytes);
    let net = Network::deserialize(&mut fp).unwrap();
    assert_eq!(net.network_type(), NetworkType::Series);
    assert_eq!(net.num_inputs(), 2);
    assert_eq!(net.num_outputs(), 3);

    // One timestep input [a, b] = [2, 1]: logits = [2, 1, 0], then softmax.
    let input = NetworkIO::from_rows(1, 2, vec![2.0, 1.0]);
    let out = net.forward(&input).unwrap();
    assert_eq!(out.width(), 1);
    assert_eq!(out.num_features(), 3);

    let logits = [2.0f32, 1.0, 0.0];
    let max = 2.0f32;
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let expected: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    let got = out.f(0);
    for (g, e) in got.iter().zip(&expected) {
        assert!((g - e).abs() < 1e-6, "softmax mismatch: got {g}, want {e}");
    }
    assert!(
        (got.iter().sum::<f32>() - 1.0).abs() < 1e-6,
        "softmax must normalize"
    );
}

#[test]
fn one_timestep_lstm_forward_matches_hand_arithmetic() {
    // Plain 1-D NT_LSTM, ni=1, ns=no=1, so na = ni + ns = 2 and each gate
    // matrix is [ns=1, na+1=3] = [w_in, w_rec, bias].
    let ci = vec![1.0, 0.0, 0.0]; // CI candidate: 1.0 * x
    let gi = vec![0.0, 0.0, 0.25]; // GI: bias 0.25
    let gf = vec![0.0, 0.0, -0.5]; // GF1: bias -0.5
    let go = vec![0.0, 0.0, 0.75]; // GO: bias 0.75

    let mut w = Writer::new();
    w.header(NetworkType::Lstm, 1, 1, "lstm");
    w.i32(2); // na
    for g in [&ci, &gi, &gf, &go] {
        w.weight_matrix_float(1, 3, g);
    }

    let mut fp = crate::TFile::new(&w.bytes);
    let net = Network::deserialize(&mut fp).unwrap();
    assert_eq!(net.network_type(), NetworkType::Lstm);
    assert_eq!(net.num_outputs(), 1);

    // t=0, x=0.5, prev_out=0, prev_state=0.
    let x = 0.5f32;
    let input = NetworkIO::from_rows(1, 1, vec![x]);
    let out = net.forward(&input).unwrap();
    assert_eq!(out.width(), 1);
    assert_eq!(out.num_features(), 1);

    // Gate pre-activations (recurrent term is 0 at t=0):
    let ci_pre = 1.0 * x; // 0.5
    let gi_pre = 0.25;
    let gf_pre = -0.5;
    let go_pre = 0.75;
    let i = sigmoid(gi_pre);
    let f = sigmoid(gf_pre);
    let o = sigmoid(go_pre);
    let g = ci_pre.tanh();
    let cell = f * 0.0 + i * g;
    let hidden = o * cell.tanh();

    assert!(
        (out.f(0)[0] - hidden).abs() < 1e-6,
        "lstm output {} != hand value {hidden}",
        out.f(0)[0]
    );
}

#[test]
fn parallel_concatenates_two_fully_connected_outputs() {
    // Parallel[ Linear(ni=2,no=1), Linear(ni=2,no=2) ] => output depth 3.
    // Sub A weights [1,3]: [1, 1, 0]  -> a0 = x0 + x1
    // Sub B weights [2,3]: [1,0,0 ; 0,1,0] -> b0 = x0, b1 = x1
    let mut a = Writer::new();
    a.header(NetworkType::Linear, 2, 1, "a");
    a.weight_matrix_float(1, 3, &[1.0, 1.0, 0.0]);

    let mut b = Writer::new();
    b.header(NetworkType::Linear, 2, 2, "b");
    b.weight_matrix_float(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    let mut par = Writer::new();
    par.header(NetworkType::Parallel, 2, 3, "par");
    par.u32(2);
    par.bytes.extend_from_slice(&a.bytes);
    par.bytes.extend_from_slice(&b.bytes);

    let mut fp = crate::TFile::new(&par.bytes);
    let net = Network::deserialize(&mut fp).unwrap();
    assert_eq!(net.network_type(), NetworkType::Parallel);
    assert_eq!(net.num_outputs(), 3, "parallel output = sum of sub-outputs");

    // Two timesteps to exercise width handling.
    let input = NetworkIO::from_rows(2, 2, vec![3.0, 4.0, 5.0, 6.0]);
    let out = net.forward(&input).unwrap();
    assert_eq!(out.width(), 2);
    assert_eq!(out.num_features(), 3);
    // t0: [x0+x1, x0, x1] = [7, 3, 4]; t1: [11, 5, 6].
    assert_eq!(out.f(0), &[7.0, 3.0, 4.0]);
    assert_eq!(out.f(1), &[11.0, 5.0, 6.0]);
}

#[test]
fn series_wraps_parallel_then_softmax() {
    // Series[ Parallel[Linear, Linear], Softmax ] — exercises composite wiring.
    let mut a = Writer::new();
    a.header(NetworkType::Linear, 2, 1, "a");
    a.weight_matrix_float(1, 3, &[1.0, 0.0, 0.0]); // a0 = x0

    let mut b = Writer::new();
    b.header(NetworkType::Linear, 2, 1, "b");
    b.weight_matrix_float(1, 3, &[0.0, 1.0, 0.0]); // b0 = x1

    let mut par = Writer::new();
    par.header(NetworkType::Parallel, 2, 2, "par");
    par.u32(2);
    par.bytes.extend_from_slice(&a.bytes);
    par.bytes.extend_from_slice(&b.bytes);

    // Softmax over the 2 parallel outputs.
    let mut sm = Writer::new();
    sm.header(NetworkType::Softmax, 2, 2, "sm");
    sm.weight_matrix_float(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    let mut series = Writer::new();
    series.header(NetworkType::Series, 2, 2, "root");
    series.u32(2);
    series.bytes.extend_from_slice(&par.bytes);
    series.bytes.extend_from_slice(&sm.bytes);

    let mut fp = crate::TFile::new(&series.bytes);
    let net = Network::deserialize(&mut fp).unwrap();
    assert_eq!(net.num_outputs(), 2);

    let input = NetworkIO::from_rows(1, 2, vec![1.0, 0.0]);
    let out = net.forward(&input).unwrap();
    // Parallel -> [x0, x1] = [1, 0]; softmax identity-weighted -> softmax([1,0]).
    let e1 = 1.0f32.exp();
    let e0 = 0.0f32.exp();
    let s = e1 + e0;
    assert!((out.f(0)[0] - e1 / s).abs() < 1e-6);
    assert!((out.f(0)[1] - e0 / s).abs() < 1e-6);
}

/// Integration test against a real `.traineddata` file. Ignored by default and
/// gated on `KOPITIAM_TEST_TRAINEDDATA` (a path to a `*.traineddata`). Downloads
/// nothing: it reads a file the developer already has. Asserts the LSTM network
/// deserializes and its output dimension equals the recoder's code range + 1
/// (the CTC null/blank), the invariant Tesseract's `LSTMRecognizer` relies on.
#[test]
#[ignore = "needs a real model; set KOPITIAM_TEST_TRAINEDDATA=/path/to/xxx.traineddata"]
fn real_traineddata_network_builds_and_output_matches_recoder() {
    let path = match std::env::var("KOPITIAM_TEST_TRAINEDDATA") {
        Ok(p) => p,
        Err(_) => return,
    };
    let bytes = std::fs::read(&path).expect("read traineddata");
    let mgr = crate::TessdataManager::from_bytes(&bytes).expect("parse traineddata");

    let mut lstm_fp = mgr
        .component_reader(crate::TessdataType::Lstm)
        .expect("Lstm component present");
    let net = Network::deserialize(&mut lstm_fp).expect("deserialize LSTM network");
    assert!(net.num_outputs() > 0, "network must have outputs");

    // Cross-check against the recoder's code range, when the recoder is present.
    if let Some(mut rec_fp) = mgr.component_reader(crate::TessdataType::LstmRecoder) {
        let mut recoder = crate::UnicharCompress::new();
        recoder
            .deserialize(&mut rec_fp)
            .expect("deserialize recoder");
        assert_eq!(
            net.num_outputs(),
            recoder.code_range() + 1,
            "softmax outputs should equal recoder code_range + 1 (CTC null)"
        );
    }
}
