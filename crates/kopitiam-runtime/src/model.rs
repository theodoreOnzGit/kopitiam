//! [`QwenModel`]: the concrete Qwen-family transformer, wiring embedding,
//! [`crate::rope::RotaryEmbedding`], [`crate::block::block_forward`] (per
//! layer), the final norm, and the output projection into one
//! [`crate::traits::Model`] implementation.

use kopitiam_core::Result;
use kopitiam_loader::LoadedModel;
use kopitiam_tensor::Tensor;

use crate::block::block_forward;
use crate::config::QwenConfig;
use crate::kv_cache::KvCache;
use crate::linear::linear;
use crate::rope::RotaryEmbedding;
use crate::traits::{CpuBackend, Model};
use crate::weights::ModelWeights;

/// A loaded, ready-to-run Qwen-family transformer.
pub struct QwenModel {
    config: QwenConfig,
    weights: ModelWeights,
    rope: RotaryEmbedding,
    backend: CpuBackend,
}

impl QwenModel {
    /// Builds a [`QwenModel`] from an already-parsed [`LoadedModel`]
    /// (`kopitiam_loader::load_model`'s result): resolves
    /// [`QwenConfig::from_metadata`], loads and dequantizes every weight
    /// tensor via [`crate::weights::ModelWeights::load`], and precomputes
    /// RoPE's rotation tables up to the model's context window.
    pub fn from_loaded_model(model: &LoadedModel) -> Result<Self> {
        let config = QwenConfig::from_metadata(model.metadata())?;
        let weights = ModelWeights::load(model, &config)?;
        let rope = RotaryEmbedding::new(config.rope_dimension_count, config.rope_theta, config.max_context);
        Ok(Self { config, weights, rope, backend: CpuBackend })
    }

    pub fn config(&self) -> &QwenConfig {
        &self.config
    }

    pub fn backend(&self) -> &CpuBackend {
        &self.backend
    }
}

impl Model for QwenModel {
    fn forward(&self, token_ids: &[u32], cache: &mut KvCache) -> Result<Tensor> {
        let ids: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
        let ids_tensor = Tensor::from_i32(ids, [token_ids.len()])?;

        // [seq, hidden]: one embedding row per input token.
        let mut x = self.weights.token_embd.gather_rows(&ids_tensor)?;

        // Read once, before any layer's cache is touched by this call --
        // see crate::attention::attention_forward's docs for why reading
        // this per-layer instead (KvCache::len() reports layer 0's length
        // specifically) is a real KV-cache correctness bug, not a harmless
        // simplification.
        let position_offset = cache.len();

        for (layer_index, layer) in self.weights.layers.iter().enumerate() {
            x = block_forward(
                &x,
                layer,
                &self.rope,
                self.config.n_heads,
                self.config.n_kv_heads,
                self.config.head_dim,
                self.config.norm_eps,
                layer_index,
                position_offset,
                cache,
            )?;
        }

        let x = x.rms_norm(&self.weights.output_norm, self.config.norm_eps)?;
        // logits = x @ output_weight^T -> [seq, vocab_size]. Goes through
        // `linear` (not a bare `matmul`+`transpose`) so a quantized
        // `output.weight` -- usually the single largest weight matrix in a
        // Qwen checkpoint, see `crate::weights::ModelWeights`'s docs --
        // gets the same `Tensor::quantized_matmul` fast path every other
        // projection does, via `linear`'s dtype dispatch.
        linear(&x, &self.weights.output_weight, None)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn max_context(&self) -> usize {
        self.config.max_context
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(self.config.n_layers, self.config.max_context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::greedy_argmax;
    use crate::test_support::synthetic_gguf::{build, tiny_model_bytes, write_temp_gguf, SyntheticModelSpec};
    use crate::traits::Backend;
    use kopitiam_core::DType;

    fn load_tiny() -> QwenModel {
        let bytes = tiny_model_bytes();
        let path = write_temp_gguf(&bytes, "model-tiny");
        let loaded = kopitiam_loader::load_model(&path).unwrap();
        QwenModel::from_loaded_model(&loaded).unwrap()
    }

    /// Builds and loads a full `QwenModel` from `spec`, through the real
    /// GGUF byte format end to end (write -> `kopitiam_loader::load_model`
    /// -> `QwenModel::from_loaded_model`) — the same path
    /// `kopitiam-ai`'s `LocalAdapter` uses on a real model file, just
    /// pointed at a synthetic one.
    fn load_from_spec(spec: &SyntheticModelSpec, disambiguator: &str) -> QwenModel {
        let bytes = build(spec);
        let path = write_temp_gguf(&bytes, disambiguator);
        let loaded = kopitiam_loader::load_model(&path).unwrap();
        QwenModel::from_loaded_model(&loaded).unwrap()
    }

    /// The end-to-end wiring test called for by this crate's task brief:
    /// no real Qwen GGUF is present on this machine (see
    /// `crate::model::tests::a_real_model_on_disk_is_used_if_present`,
    /// which is `#[ignore]`d for exactly that reason), so this builds a
    /// tiny-but-structurally-real synthetic GGUF (2 layers, GQA-shaped,
    /// tied embeddings, random-but-fixed weights) and proves the whole
    /// load -> forward -> logits pipeline runs and produces finite,
    /// correctly-shaped output. It intentionally does not assert anything
    /// about *which* tokens the random weights favor -- that would be
    /// asserting a coincidence, not a property of the code.
    #[test]
    fn synthetic_model_forward_pass_runs_end_to_end_and_produces_finite_logits() {
        let model = load_tiny();
        let mut cache = model.new_cache();

        let prompt = [3u32, 7, 1, 22];
        let logits = model.forward(&prompt, &mut cache).unwrap();
        assert_eq!(logits.shape().dims(), &[prompt.len(), model.vocab_size()]);
        for v in logits.to_vec_f32().unwrap() {
            assert!(v.is_finite(), "logits must be finite, got {v}");
        }
        assert_eq!(cache.len(), prompt.len());

        // Greedy-decode one further step to prove sampling composes with a
        // real forward pass end to end, per this crate's "greedy decode
        // end-to-end" acceptance bar.
        let next = greedy_argmax(&logits.to_vec_f32().unwrap()[(prompt.len() - 1) * model.vocab_size()..]);
        assert!((next as usize) < model.vocab_size());
    }

    /// A model whose GGUF has no separate `output.weight` (tied
    /// embeddings) must still produce logits -- the tied-embedding
    /// fallback in `ModelWeights::load` must actually be wired into the
    /// forward pass, not just present as an unused field.
    #[test]
    fn tied_embeddings_model_still_produces_logits() {
        let spec = SyntheticModelSpec { tie_embeddings: true, ..SyntheticModelSpec::default() };
        let bytes = build(&spec);
        let path = write_temp_gguf(&bytes, "model-tied");
        let loaded = kopitiam_loader::load_model(&path).unwrap();
        let model = QwenModel::from_loaded_model(&loaded).unwrap();
        let mut cache = model.new_cache();

        let logits = model.forward(&[1, 2, 3], &mut cache).unwrap();
        assert_eq!(logits.shape().dims(), &[3, spec.vocab_size]);
        assert!(logits.to_vec_f32().unwrap().iter().all(|v| v.is_finite()));
    }

    /// The KV-cache correctness property named explicitly in this crate's
    /// task brief: decoding token-by-token *with* the cache must produce
    /// bitwise-identical logits to a single forward pass over the whole
    /// sequence *without* it (a fresh cache, called once with all tokens).
    /// This single test is the one most likely to catch a KV-cache bug --
    /// a wrong position offset, an off-by-one in the causal mask, a stale
    /// or duplicated cache entry -- because any of those would perturb
    /// *some* attention score and, without a cache-vs-no-cache oracle,
    /// would otherwise only show up as "the model seems a bit off".
    #[test]
    fn decoding_with_a_kv_cache_matches_a_full_forward_pass_without_one_bit_for_bit() {
        let model = load_tiny();
        let tokens = [5u32, 12, 30, 2, 8];

        // Reference: one forward call over the whole sequence, fresh cache.
        let mut full_cache = model.new_cache();
        let full_logits = model.forward(&tokens, &mut full_cache).unwrap().to_vec_f32().unwrap();

        // Decode: one forward call per token, reusing one cache across calls.
        let mut step_cache = model.new_cache();
        let mut decoded_logits = Vec::new();
        for &token in &tokens {
            let step = model.forward(&[token], &mut step_cache).unwrap();
            decoded_logits.extend(step.to_vec_f32().unwrap());
        }

        assert_eq!(full_cache.len(), step_cache.len());
        assert_eq!(
            full_logits.len(),
            decoded_logits.len(),
            "full-forward and step-by-step decode must produce the same number of logit values"
        );
        for (i, (full, decoded)) in full_logits.iter().zip(&decoded_logits).enumerate() {
            assert_eq!(
                full.to_bits(),
                decoded.to_bits(),
                "logit {i} differs between full-forward ({full}) and cached decode ({decoded}) -- \
                 this is exactly the KV-cache bug this test exists to catch"
            );
        }
    }

    /// Guards against a *different* KV-cache bug than the equivalence test
    /// above: rather than proving "matches a no-cache oracle", this proves
    /// the cache actually accumulates rather than silently resetting or
    /// overwriting -- if it did, `cache.len()` would not grow, and the
    /// third decode step would attend over 1 cached position instead of 3.
    #[test]
    fn the_cache_length_grows_by_exactly_one_position_per_decode_step() {
        let model = load_tiny();
        let mut cache = model.new_cache();
        for (step, &token) in [4u32, 9, 15].iter().enumerate() {
            model.forward(&[token], &mut cache).unwrap();
            assert_eq!(cache.len(), step + 1);
        }
    }

    /// Real, full-size Qwen `.gguf` weights (as opposed to the vocab-only
    /// fixture at `crates/kopitiam-ai/vendor/llama.cpp/models/`) were not
    /// found anywhere on this machine when this test was written (`find ~
    /// -name "*.gguf" -size +100M` and a broader filesystem search both
    /// came back empty). This test is `#[ignore]`d rather than deleted so
    /// it is ready to run the moment a real model is placed at the path
    /// below, without anyone having to reconstruct what "load and greedily
    /// generate a few tokens" should look like from scratch.
    #[test]
    #[ignore = "no real Qwen GGUF present on this machine; point REAL_QWEN_GGUF_PATH at one to run this"]
    fn a_real_model_on_disk_is_used_if_present() {
        let path = std::env::var("REAL_QWEN_GGUF_PATH").expect("set REAL_QWEN_GGUF_PATH to a real Qwen .gguf file");
        let loaded = kopitiam_loader::load_model(path).unwrap();
        let model = QwenModel::from_loaded_model(&loaded).unwrap();
        let mut cache = model.new_cache();

        // A handful of arbitrary, in-range token ids stands in for a real
        // prompt: this test's purpose is proving the pipeline runs on real
        // weights, not testing tokenizer round-tripping (see
        // `crate::gguf_tokenizer` for that).
        let prompt = [1u32, 2, 3, 4];
        let mut logits = model.forward(&prompt, &mut cache).unwrap().to_vec_f32().unwrap();
        for _ in 0..5 {
            let next = greedy_argmax(&logits[logits.len() - model.vocab_size()..]);
            let step = model.forward(&[next], &mut cache).unwrap();
            logits = step.to_vec_f32().unwrap();
        }
    }

    #[test]
    fn dtype_matches_and_config_is_reachable() {
        let model = load_tiny();
        assert_eq!(model.weights.token_embd.dtype(), DType::F32);
        assert_eq!(model.config().n_layers, 2);
        assert_eq!(model.backend().device(), kopitiam_core::Device::Cpu);
    }

    // -- Quantized-weight wiring: the Phase 2 "the fast path is actually
    // -- reachable end to end, not just a library function nobody calls" gate.

    /// A model whose GGUF ships every matmul-operand weight as real `Q8_0`
    /// bytes must load them still `Q8_0` (see `crate::weights::ModelWeights`'s
    /// and `crate::bridge::load_matmul_weight`'s docs) and run a forward
    /// pass end to end through `crate::linear::linear`'s dtype dispatch,
    /// producing finite, correctly-shaped logits.
    #[test]
    fn quantized_matmul_weights_load_natively_and_produce_finite_logits() {
        // tie_embeddings: false, so a separate (quantized) output.weight
        // is actually written -- otherwise it would fall back to a clone
        // of the (always-f32) token embedding table and this test would
        // not exercise the output projection's quantized path at all.
        let spec = SyntheticModelSpec {
            quantize_matmul_weights: true,
            tie_embeddings: false,
            ..SyntheticModelSpec::quantized_benchmark()
        };
        let model = load_from_spec(&spec, "model-quantized");

        assert_eq!(model.weights.layers[0].wq.dtype(), DType::Q8_0);
        assert_eq!(model.weights.output_weight.dtype(), DType::Q8_0);
        // Embeddings are never quantized in this scope (gather_rows needs
        // f32 elementwise access) -- see `crate::bridge::load_matmul_weight`.
        assert_eq!(model.weights.token_embd.dtype(), DType::F32);

        let mut cache = model.new_cache();
        let prompt = [3u32, 7, 1, 22, 9];
        let logits = model.forward(&prompt, &mut cache).unwrap();
        assert_eq!(logits.shape().dims(), &[prompt.len(), model.vocab_size()]);
        assert!(logits.to_vec_f32().unwrap().iter().all(|v| v.is_finite()));
    }

    /// Isolates quantization error from every other source of divergence:
    /// both specs below share one `Xorshift64` stream consumed in the same
    /// call order (see `synthetic_gguf::build`'s docs), so the *only*
    /// difference between the two resulting models is whether each
    /// matmul-operand weight got rounded to `Q8_0` on the way in. This is
    /// not a tight bound (four transformer layers of RMSNorm/softmax/SiLU
    /// nonlinearity compound Q8_0's ~1/127 per-element rounding error
    /// considerably, and the weights themselves are quantized here, unlike
    /// `kopitiam-tensor`'s tighter kernel-level gate which isolates
    /// activation-only error) -- but it is tight enough that a real wiring
    /// bug (a transposed index, a forgotten scale multiply, reading the
    /// wrong block) would blow it up by orders of magnitude and get caught
    /// here.
    #[test]
    fn quantized_and_f32_weights_of_the_same_underlying_values_produce_similar_logits() {
        let f32_spec = SyntheticModelSpec::quantized_benchmark();
        let q_spec = SyntheticModelSpec { quantize_matmul_weights: true, ..SyntheticModelSpec::quantized_benchmark() };

        let f32_model = load_from_spec(&f32_spec, "model-q-cmp-f32");
        let q_model = load_from_spec(&q_spec, "model-q-cmp-q8");

        let prompt = [3u32, 7, 1, 22, 9];
        let mut f32_cache = f32_model.new_cache();
        let logits_f32 = f32_model.forward(&prompt, &mut f32_cache).unwrap().to_vec_f32().unwrap();
        let mut q_cache = q_model.new_cache();
        let logits_q = q_model.forward(&prompt, &mut q_cache).unwrap().to_vec_f32().unwrap();

        assert_eq!(logits_f32.len(), logits_q.len());
        for (f32_v, q_v) in logits_f32.iter().zip(&logits_q) {
            let scale = f32_v.abs().max(1.0);
            assert!(
                (f32_v - q_v).abs() / scale < 1.0,
                "quantized ({q_v}) and f32 ({f32_v}) logits diverged past a sane bound"
            );
        }
    }

    // ---------------------------------------------------------------
    // Benchmark: quantized weights vs. dequantize-to-f32.
    // ---------------------------------------------------------------

    /// Measures what Phase 2 actually bought, so that "faster" and "smaller"
    /// are numbers rather than feelings.
    ///
    /// `#[ignore]`d because it is a measurement, not an assertion — it takes
    /// seconds, and a wall-clock number is not a correctness property and must
    /// never fail CI on a loaded machine. Run it deliberately:
    ///
    /// ```text
    /// cargo test --release -p kopitiam-runtime bench_quantized -- --ignored --nocapture
    /// ```
    ///
    /// # What this does and does not prove
    ///
    /// It compares a Q8_0-weighted model against the same model with its weights
    /// dequantized to `f32`, on a synthetic 4-layer / 256-hidden toy. It is
    /// therefore honest about *ratios* (memory, and whether the fused kernel is
    /// in the right ballpark) and dishonest about *absolutes*: a 4-layer toy on
    /// this desktop tells you nothing about a 7B model on a phone. The number
    /// that actually matters is the memory one, because that is the difference
    /// between the model fitting on the target device and not existing there at
    /// all.
    #[test]
    #[ignore = "measurement, not an assertion; run deliberately with --ignored --nocapture"]
    fn bench_quantized_vs_f32_weights() {
        use std::time::Instant;

        let spec = SyntheticModelSpec::quantized_benchmark();

        let mut f32_spec = spec.clone();
        f32_spec.quantize_matmul_weights = false;
        let mut q_spec = spec.clone();
        q_spec.quantize_matmul_weights = true;

        let f32_bytes = build(&f32_spec);
        let q_bytes = build(&q_spec);

        let f32_model = load_from_spec(&f32_spec, "bench-f32");
        let q_model = load_from_spec(&q_spec, "bench-q8");

        // 32-token prefill, then 32 decode steps — the two phases that behave
        // differently (prefill is compute-bound, decode is memory-bound).
        let prompt: Vec<u32> = (0..32).map(|i| (i % 100) as u32).collect();

        let run = |model: &QwenModel| -> (f64, f64) {
            let mut cache = model.new_cache();
            let t0 = Instant::now();
            model.forward(&prompt, &mut cache).unwrap();
            let prefill = t0.elapsed().as_secs_f64();

            let t1 = Instant::now();
            for i in 0..32u32 {
                model.forward(&[i % 100], &mut cache).unwrap();
            }
            let decode = t1.elapsed().as_secs_f64();
            (prefill, decode)
        };

        // Warm the caches/branch predictors once so the first model measured
        // is not unfairly penalised.
        run(&f32_model);
        run(&q_model);

        let (f32_prefill, f32_decode) = run(&f32_model);
        let (q_prefill, q_decode) = run(&q_model);

        println!();
        println!("=== Kopitiam Runtime: quantized (Q8_0) vs f32 weights ===");
        println!("model: {} layers, hidden {}, vocab {}", spec.n_layers, spec.hidden_size, spec.vocab_size);
        println!();
        println!("GGUF file size on disk");
        println!("  f32 weights : {:>10} bytes", f32_bytes.len());
        println!("  Q8_0 weights: {:>10} bytes  ({:.2}x smaller)",
                 q_bytes.len(), f32_bytes.len() as f64 / q_bytes.len() as f64);
        println!();
        println!("prefill (32 tokens)");
        println!("  f32 : {:>8.2} ms  ({:>7.1} tok/s)", f32_prefill * 1e3, 32.0 / f32_prefill);
        println!("  Q8_0: {:>8.2} ms  ({:>7.1} tok/s)", q_prefill * 1e3, 32.0 / q_prefill);
        println!();
        println!("decode (32 steps, with KV cache)");
        println!("  f32 : {:>8.2} ms  ({:>7.1} tok/s)", f32_decode * 1e3, 32.0 / f32_decode);
        println!("  Q8_0: {:>8.2} ms  ({:>7.1} tok/s)", q_decode * 1e3, 32.0 / q_decode);
        println!();
        println!("NOTE: the memory ratio is the number that matters. A 7B Q4_0 model");
        println!("dequantized to f32 is ~28GB and simply does not fit on a phone; kept");
        println!("quantized it is ~4GB and does. Wall-clock on a 4-layer toy on this");
        println!("desktop says little about a real model on the real target.");

        // The one thing worth asserting: quantized weights really are smaller.
        // That is a property, not a measurement, and it is the whole point.
        assert!(q_bytes.len() < f32_bytes.len(), "quantized weights must be smaller on disk");
    }
}
