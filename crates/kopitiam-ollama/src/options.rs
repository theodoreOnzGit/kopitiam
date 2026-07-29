//! Sampling and runner options -- **the source of every magic number**.
//!
//! **Upstream:** `api/types.go` (`type Options`, `type Runner`,
//! `DefaultOptions()`, `(*Options).FromMap`) (ollama, MIT).
//!
//! ## Why this module is the whole point of the port
//!
//! KOPITIAM's own history says it better than an argument would. `kopitiam ai
//! chat` once answered "hello" with *"I'm a bot, I'm a bot, I'm a chatbot, I'm a
//! chatbot, ..."* for 256 tokens. Cause: greedy argmax, no repetition penalty.
//! The fix needed sampling defaults -- and the reflex was to pick "temperature
//! 0.7, top_p 0.95" out of the air, which would have been an invented number
//! shipped forever. [`Options::default`] below is where those numbers come from
//! instead: values ollama has run against far more models than KOPITIAM will
//! ever test.
//!
//! CLAUDE.md's rule -- *a number with no source is a bug that has not fired yet*
//! -- is enforced here structurally. Every field carry its upstream default in
//! its own doc comment, so nobody downstream ever has to guess again.
//!
//! ## Options vs Runner: when the value can still change
//!
//! Upstream splits the struct in two, and the split is **not** cosmetic:
//!
//! * [`Options`] fields are **per-request**. Change `temperature` between two
//!   messages and the next token already feels it -- nothing reloads.
//! * [`Runner`] fields are **fixed at load time**. `num_ctx` sizes the KV cache,
//!   `num_gpu` decides how many layers go to VRAM. Change one and the model must
//!   be **evicted and reloaded**; you cannot resize a live KV cache.
//!
//! Anything driving a scheduler has to respect that boundary: two requests that
//! differ only in [`Options`] share one loaded model, two that differ in
//! [`Runner`] cannot. Keeping the types separate is what stop that rule from
//! being a comment somebody forgets.

use serde::{Deserialize, Serialize};

/// Options fixed when the model is **loaded**, not per request.
///
/// **Upstream:** `type Runner struct`.
///
/// Changing any of these require a model reload -- see the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Runner {
    /// Context window in tokens. **Upstream default: `envconfig.ContextLength()`,
    /// which is `Uint("OLLAMA_CONTEXT_LENGTH", 0)` -- i.e. `0`.**
    ///
    /// `0` is not "no context"; it means **decide at load time from available
    /// VRAM**. Upstream's own env-var help spells out the ladder it picks from:
    /// *"default: 4k/32k/256k based on VRAM"*. So a caller must treat `0` as
    /// "not yet decided" and resolve it against the model's trained maximum and
    /// the machine, never pass it down as a literal window size.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub num_ctx: u32,

    /// Tokens processed per prompt-eval batch. **Upstream default: 512.**
    ///
    /// Bigger batch = faster prompt ingestion, more transient memory. 512 is
    /// upstream's balance point across the whole model zoo; it is not tuned for
    /// any one machine.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub num_batch: u32,

    /// How many transformer layers to offload to the GPU. **Upstream default: -1.**
    ///
    /// `-1` means **decide dynamically** -- fit as many layers as VRAM allows.
    /// A literal count is an override, and `0` means pure CPU. Signed on
    /// purpose: the sentinel and the count share one field upstream, and
    /// splitting them into an `Option` here would silently diverge the JSON.
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub num_gpu: i32,

    /// Which GPU to treat as primary in a multi-GPU split. **Upstream default:
    /// unset.** `None` means the backend chooses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub main_gpu: Option<i32>,

    /// Memory-map the weights instead of reading them in. **Upstream default:
    /// unset**, i.e. let the backend decide.
    ///
    /// Genuinely three-state, which is why it is an `Option<bool>` upstream too:
    /// unset (backend's call), forced on, forced off. mmap is usually the right
    /// answer, but it hurts on network filesystems and on systems that overcommit
    /// badly, so the override has to exist.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_mmap: Option<bool>,

    /// CPU threads for inference. **Upstream default: 0 -- "let the runtime
    /// decide"** (its own comment). Not "zero threads".
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub num_thread: u32,

    /// Tokens a draft model proposes per speculative-decoding step.
    /// **Upstream default: 4.**
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub draft_num_predict: u32,
}

/// Per-request sampling options.
///
/// **Upstream:** `type Options struct` (which embeds `Runner`; we hold it as a
/// named field, since Rust has no struct embedding. The JSON stays identical
/// because `#[serde(flatten)]` reproduces Go's embedded-field flattening).
///
/// Use [`Options::default`] to get upstream's `DefaultOptions()`, then overlay
/// whatever the caller asked for with [`Options::apply_map`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Options {
    /// Load-time options -- see [`Runner`].
    #[serde(flatten)]
    pub runner: Runner,

    /// Tokens kept from the front of the context when it shifts.
    /// **Upstream default: 4**, with the comment *"set a minimal num_keep to
    /// avoid issues on context shifts"*.
    ///
    /// When the window fills, old tokens get dropped -- but dropping the very
    /// first ones (typically BOS and the start of the system prompt) breaks the
    /// model's framing. Pinning a few is the cheap guard.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub num_keep: u32,

    /// PRNG seed. **Upstream default: -1**, meaning **seed from entropy**, i.e.
    /// the same prompt gives different answers each run.
    ///
    /// **KOPITIAM diverges here, and it is a deliberate divergence, not a slip.**
    /// Reproducibility is one of this project's stated engineering principles, so
    /// a KOPITIAM caller that wants determinism sets a concrete seed rather than
    /// relying on the default. The *default in this struct stays -1*, because the
    /// oracle rule says ollama wins on the value; what diverges is what our
    /// callers do with it, and that decision belongs at the call site where it
    /// can be documented against a specific workflow.
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub seed: i32,

    /// Maximum tokens to generate. **Upstream default: -1** = unlimited (stop
    /// only on an EOS token or a stop string).
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub num_predict: i32,

    /// Sample only from the `k` most likely tokens. **Upstream default: 40.**
    /// `0` disables it.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub top_k: u32,

    /// Nucleus sampling: keep the smallest set of tokens whose probabilities sum
    /// to `p`. **Upstream default: 0.9.**
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub top_p: f32,

    /// Drop tokens below `p` times the top token's probability.
    /// **Upstream default: 0.0 -- i.e. OFF.**
    ///
    /// Worth stating plainly because the absence is easy to misread: upstream's
    /// `DefaultOptions()` simply never mentions `MinP`, so it lands on Go's zero
    /// value. Off is the default, not an oversight we should "fix".
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub min_p: f32,

    /// Locally-typical sampling. **Upstream default: 1.0 -- i.e. OFF** (1.0 keeps
    /// everything).
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub typical_p: f32,

    /// How many recent tokens the repetition penalty looks back over.
    /// **Upstream default: 64.** `0` disables, `-1` means the whole context.
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub repeat_last_n: i32,

    /// Softmax temperature. **Upstream default: 0.8.** Higher = more random;
    /// `0` collapses to greedy argmax.
    ///
    /// 0.8 -- not 0.7, not 1.0 -- is upstream's number, and it is exactly the
    /// kind of value this module exists to stop anyone from inventing.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub temperature: f32,

    /// Penalty applied to tokens already seen within the `repeat_last_n` window.
    /// **Upstream default: 1.1.** `1.0` is no penalty.
    ///
    /// This is the specific knob whose absence produced *"I'm a bot, I'm a bot,
    /// I'm a bot"*. Do not ship it at 1.0 by accident.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub repeat_penalty: f32,

    /// Flat penalty for any token already present. **Upstream default: 0.0 (off).**
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub presence_penalty: f32,

    /// Penalty scaling with how often a token already appeared.
    /// **Upstream default: 0.0 (off).**
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub frequency_penalty: f32,

    /// Strings that end generation when produced. **Upstream default: empty.**
    ///
    /// Usually filled in from the model's Modelfile rather than by the user --
    /// a chat template's turn-terminator (`<|im_end|>`, `<end_of_turn>`) belongs
    /// here, and getting it wrong is how a model starts talking to itself.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
}

impl Default for Runner {
    /// The `Runner` half of **upstream `DefaultOptions()`**.
    fn default() -> Self {
        Self {
            num_ctx: 0, // envconfig.ContextLength() -> Uint("OLLAMA_CONTEXT_LENGTH", 0)
            num_batch: 512,
            num_gpu: -1, // "-1 here indicates that NumGPU should be set dynamically"
            main_gpu: None,
            use_mmap: None,
            num_thread: 0, // "let the runtime decide"
            draft_num_predict: 4,
        }
    }
}

impl Default for Options {
    /// **Upstream:** `DefaultOptions()` in `api/types.go`, value for value.
    ///
    /// If you are reaching for a sampling constant anywhere in KOPITIAM, take it
    /// from here. Do not write a literal.
    fn default() -> Self {
        Self {
            runner: Runner::default(),
            num_keep: 4,
            seed: -1,
            num_predict: -1,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0, // absent from DefaultOptions() -> Go zero value -> off
            typical_p: 1.0,
            repeat_last_n: 64,
            temperature: 0.8,
            repeat_penalty: 1.1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop: Vec::new(),
        }
    }
}

/// What went wrong overlaying a caller's `options` map.
///
/// **Upstream:** the `fmt.Errorf` strings inside `(*Options).FromMap`. The
/// messages are kept close to upstream's wording so a KOPITIAM error is
/// searchable against ollama's issue tracker.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum OptionsError {
    #[error("option {0:?} must be of type integer")]
    NotInteger(String),
    #[error("option {0:?} must be of type boolean")]
    NotBool(String),
    #[error("option {0:?} must be of type float32")]
    NotFloat(String),
    #[error("option {0:?} must be of type array")]
    NotArray(String),
    #[error("option {0:?} must be of an array of strings")]
    NotStringArray(String),
}

impl Options {
    /// Overlay a caller-supplied `{"temperature": 0.2, ...}` map onto `self`.
    ///
    /// **Upstream:** `(*Options).FromMap(m map[string]any)`, which does this by
    /// reflecting over the struct's JSON tags. We match the field set by hand --
    /// more typing, but the compiler then guarantees the mapping stays complete,
    /// whereas the Go version fails silently if a tag is ever misspelled.
    ///
    /// Three upstream behaviours that are easy to get wrong, so they are spelled
    /// out and tested:
    ///
    /// * **An unknown key is NOT an error.** Upstream logs `"invalid option
    ///   provided"` and carries on. So do we -- returned in `unknown` for the
    ///   caller to log, because a hard failure here would break every client that
    ///   sends a newer option to an older server.
    /// * **A `null` value is skipped**, leaving the existing value alone. Not
    ///   reset to zero.
    /// * **A wrong type IS an error**, and it aborts the whole overlay. Better a
    ///   loud failure than a request that quietly runs at the wrong temperature.
    ///
    /// One faithful wart: upstream accepts a JSON number for an integer field by
    /// **truncating** the float (`int64(t)`), so `"top_k": 40.9` becomes `40`
    /// rather than being rejected. Copied, because clients rely on it -- JSON has
    /// no integer type, so a language that serialises `40` as `40.0` must still
    /// work.
    ///
    /// Returns the list of unrecognised keys.
    pub fn apply_map(
        &mut self,
        m: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Vec<String>, OptionsError> {
        use serde_json::Value;

        // Go's `int64(t)` on a float64 truncates toward zero, and so does `as`.
        fn as_int(key: &str, v: &Value) -> Result<i64, OptionsError> {
            v.as_i64()
                .or_else(|| v.as_f64().map(|f| f as i64))
                .ok_or_else(|| OptionsError::NotInteger(key.to_string()))
        }
        fn as_float(key: &str, v: &Value) -> Result<f32, OptionsError> {
            // Upstream requires a `float64` here and rejects anything else --
            // but Go's encoding/json unmarshals EVERY JSON number to float64,
            // so an integer literal like `1` arrives as 1.0 and is accepted.
            // `as_f64` reproduces that: it succeeds for integer JSON too.
            v.as_f64()
                .map(|f| f as f32)
                .ok_or_else(|| OptionsError::NotFloat(key.to_string()))
        }
        fn as_bool(key: &str, v: &Value) -> Result<bool, OptionsError> {
            v.as_bool().ok_or_else(|| OptionsError::NotBool(key.to_string()))
        }

        let mut unknown = Vec::new();

        for (key, val) in m {
            // Upstream: `if val == nil { continue }` -- a null leaves the
            // existing value untouched.
            if val.is_null() {
                continue;
            }

            match key.as_str() {
                // ---- Runner (load-time) ----
                "num_ctx" => self.runner.num_ctx = as_int(key, val)? as u32,
                "num_batch" => self.runner.num_batch = as_int(key, val)? as u32,
                "num_gpu" => self.runner.num_gpu = as_int(key, val)? as i32,
                "main_gpu" => self.runner.main_gpu = Some(as_int(key, val)? as i32),
                "use_mmap" => self.runner.use_mmap = Some(as_bool(key, val)?),
                "num_thread" => self.runner.num_thread = as_int(key, val)? as u32,
                "draft_num_predict" => self.runner.draft_num_predict = as_int(key, val)? as u32,

                // ---- Options (per-request) ----
                "num_keep" => self.num_keep = as_int(key, val)? as u32,
                "seed" => self.seed = as_int(key, val)? as i32,
                "num_predict" => self.num_predict = as_int(key, val)? as i32,
                "top_k" => self.top_k = as_int(key, val)? as u32,
                "top_p" => self.top_p = as_float(key, val)?,
                "min_p" => self.min_p = as_float(key, val)?,
                "typical_p" => self.typical_p = as_float(key, val)?,
                "repeat_last_n" => self.repeat_last_n = as_int(key, val)? as i32,
                "temperature" => self.temperature = as_float(key, val)?,
                "repeat_penalty" => self.repeat_penalty = as_float(key, val)?,
                "presence_penalty" => self.presence_penalty = as_float(key, val)?,
                "frequency_penalty" => self.frequency_penalty = as_float(key, val)?,
                "stop" => {
                    let arr = val
                        .as_array()
                        .ok_or_else(|| OptionsError::NotArray(key.to_string()))?;
                    let mut out = Vec::with_capacity(arr.len());
                    for item in arr {
                        out.push(
                            item.as_str()
                                .ok_or_else(|| OptionsError::NotStringArray(key.to_string()))?
                                .to_string(),
                        );
                    }
                    self.stop = out;
                }

                _ => unknown.push(key.clone()),
            }
        }

        Ok(unknown)
    }

    /// Upstream defaults with a caller's map already overlaid -- the one-liner
    /// a request handler actually wants.
    pub fn from_map(
        m: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<(Self, Vec<String>), OptionsError> {
        let mut o = Self::default();
        let unknown = o.apply_map(m)?;
        Ok((o, unknown))
    }

    /// Would swapping `self` for `other` force the model to be **reloaded**?
    ///
    /// Not an upstream function -- upstream's scheduler compares runner options
    /// inline. Named here because the rule (see module docs: [`Runner`] changes
    /// need a reload, [`Options`] changes do not) is exactly the kind of thing
    /// that rots into a wrong inline comparison, and a named predicate with a
    /// test cannot rot the same way.
    pub fn needs_reload(&self, other: &Options) -> bool {
        self.runner != other.runner
    }
}

// Go's `omitempty` skips a field at its zero value. These predicates reproduce
// it exactly, so a KOPITIAM-serialised Options is byte-comparable with ollama's.
fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}
fn is_zero_i32(v: &i32) -> bool {
    *v == 0
}
fn is_zero_f32(v: &f32) -> bool {
    *v == 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn map(v: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        v.as_object().unwrap().clone()
    }

    /// **The provenance test.** Every one of these numbers is quoted from
    /// `DefaultOptions()` in `api/types.go`. If a future edit drifts one of
    /// them, this fails and the drift has to be argued for rather than
    /// discovered later in a model that repeats itself.
    #[test]
    fn defaults_are_exactly_ollamas_default_options() {
        let o = Options::default();
        assert_eq!(o.num_predict, -1);
        assert_eq!(o.num_keep, 4);
        assert_eq!(o.temperature, 0.8);
        assert_eq!(o.top_k, 40);
        assert_eq!(o.top_p, 0.9);
        assert_eq!(o.typical_p, 1.0);
        assert_eq!(o.repeat_last_n, 64);
        assert_eq!(o.repeat_penalty, 1.1);
        assert_eq!(o.presence_penalty, 0.0);
        assert_eq!(o.frequency_penalty, 0.0);
        assert_eq!(o.seed, -1);
        assert_eq!(o.min_p, 0.0, "min_p is absent upstream -> zero -> off");
        assert!(o.stop.is_empty());

        assert_eq!(o.runner.num_ctx, 0, "0 means decide from VRAM, not no-context");
        assert_eq!(o.runner.num_batch, 512);
        assert_eq!(o.runner.num_gpu, -1, "-1 means decide dynamically");
        assert_eq!(o.runner.num_thread, 0, "0 means let the runtime decide");
        assert_eq!(o.runner.draft_num_predict, 4);
        assert_eq!(o.runner.main_gpu, None);
        assert_eq!(o.runner.use_mmap, None);
    }

    /// The repetition guard specifically -- the setting whose absence caused
    /// "I'm a bot, I'm a bot, ...". A penalty of 1.0 is no penalty at all.
    #[test]
    fn the_repetition_guard_is_actually_on_by_default() {
        let o = Options::default();
        assert!(o.repeat_penalty > 1.0, "repeat_penalty must not default to a no-op");
        assert!(o.repeat_last_n > 0, "a penalty with a zero window is also a no-op");
        assert!(o.temperature > 0.0, "temperature 0 is greedy argmax, the other half of the bug");
    }

    #[test]
    fn apply_map_overlays_only_the_keys_given() {
        let mut o = Options::default();
        let unknown = o
            .apply_map(&map(json!({"temperature": 0.2, "num_ctx": 8192})))
            .unwrap();
        assert!(unknown.is_empty());
        assert_eq!(o.temperature, 0.2);
        assert_eq!(o.runner.num_ctx, 8192);
        // Untouched keys keep the defaults.
        assert_eq!(o.top_k, 40);
        assert_eq!(o.repeat_penalty, 1.1);
    }

    /// Upstream warns and continues on an unknown key. A hard error here would
    /// break every client that speaks a newer option set than the server.
    #[test]
    fn an_unknown_key_is_reported_but_not_fatal() {
        let mut o = Options::default();
        let unknown = o
            .apply_map(&map(json!({"temperature": 0.5, "mirostat_zeta": 0.1})))
            .unwrap();
        assert_eq!(unknown, vec!["mirostat_zeta".to_string()]);
        assert_eq!(o.temperature, 0.5, "the valid key still applied");
    }

    /// `null` means "leave it alone", NOT "reset to zero".
    #[test]
    fn a_null_value_leaves_the_existing_setting_untouched() {
        let mut o = Options::default();
        o.apply_map(&map(json!({"temperature": null}))).unwrap();
        assert_eq!(o.temperature, 0.8);
    }

    #[test]
    fn a_wrong_type_is_a_hard_error() {
        let mut o = Options::default();
        assert_eq!(
            o.apply_map(&map(json!({"top_k": "forty"}))),
            Err(OptionsError::NotInteger("top_k".into()))
        );
        assert_eq!(
            o.apply_map(&map(json!({"temperature": "hot"}))),
            Err(OptionsError::NotFloat("temperature".into()))
        );
        assert_eq!(
            o.apply_map(&map(json!({"use_mmap": 1}))),
            Err(OptionsError::NotBool("use_mmap".into()))
        );
        assert_eq!(
            o.apply_map(&map(json!({"stop": "END"}))),
            Err(OptionsError::NotArray("stop".into()))
        );
        assert_eq!(
            o.apply_map(&map(json!({"stop": ["END", 3]}))),
            Err(OptionsError::NotStringArray("stop".into()))
        );
    }

    /// The faithful wart: JSON has no integer type, so a float must be accepted
    /// for an integer field and truncated, exactly like Go's `int64(t)`.
    #[test]
    fn a_json_float_is_truncated_into_an_integer_field() {
        let mut o = Options::default();
        o.apply_map(&map(json!({"top_k": 40.9, "num_ctx": 4096.0}))).unwrap();
        assert_eq!(o.top_k, 40);
        assert_eq!(o.runner.num_ctx, 4096);
    }

    /// And the mirror: an integer literal must be accepted for a float field,
    /// because Go unmarshals every JSON number to float64.
    #[test]
    fn a_json_integer_is_accepted_for_a_float_field() {
        let mut o = Options::default();
        o.apply_map(&map(json!({"temperature": 1, "top_p": 1}))).unwrap();
        assert_eq!(o.temperature, 1.0);
        assert_eq!(o.top_p, 1.0);
    }

    #[test]
    fn stop_strings_round_trip() {
        let mut o = Options::default();
        o.apply_map(&map(json!({"stop": ["<|im_end|>", "\nUser:"]}))).unwrap();
        assert_eq!(o.stop, vec!["<|im_end|>".to_string(), "\nUser:".to_string()]);
    }

    /// The load-time / per-request boundary, asserted rather than trusted to a
    /// comment: only a `Runner` change forces an evict-and-reload.
    #[test]
    fn only_runner_changes_force_a_reload() {
        let a = Options::default();

        let mut sampling_changed = a.clone();
        sampling_changed.temperature = 0.1;
        sampling_changed.stop = vec!["x".into()];
        assert!(!a.needs_reload(&sampling_changed));

        let mut ctx_changed = a.clone();
        ctx_changed.runner.num_ctx = 8192;
        assert!(a.needs_reload(&ctx_changed));

        let mut gpu_changed = a.clone();
        gpu_changed.runner.num_gpu = 0;
        assert!(a.needs_reload(&gpu_changed));
    }

    /// Go's `omitempty` drops zero-valued fields, and our serialisation must
    /// match or a KOPITIAM request will not look like an ollama one on the wire.
    #[test]
    fn serialisation_omits_empty_like_gos_omitempty() {
        // The defaults already contain both kinds of field -- zero-valued ones
        // (presence_penalty, min_p, stop, main_gpu, num_ctx) that must vanish,
        // and non-zero ones (temperature, num_batch) that must survive.
        let o = Options::default();
        let v = serde_json::to_value(&o).unwrap();
        let obj = v.as_object().unwrap();

        assert!(!obj.contains_key("presence_penalty"), "0.0 must be omitted");
        assert!(!obj.contains_key("min_p"), "0.0 must be omitted");
        assert!(!obj.contains_key("stop"), "empty vec must be omitted");
        assert!(!obj.contains_key("main_gpu"), "None must be omitted");
        assert!(!obj.contains_key("num_ctx"), "0 must be omitted");
        assert!(obj.contains_key("temperature"));

        // Runner is flattened, not nested -- same shape as Go's embedded struct.
        assert!(!obj.contains_key("runner"));
        assert_eq!(obj.get("num_batch"), Some(&json!(512)));
    }

    #[test]
    fn deserialising_a_flattened_payload_fills_the_runner() {
        let o: Options =
            serde_json::from_value(json!({"temperature": 0.3, "num_ctx": 2048})).unwrap();
        assert_eq!(o.temperature, 0.3);
        assert_eq!(o.runner.num_ctx, 2048);
    }
}
