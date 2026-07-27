//! End-to-end tests against **real model weights**, fetched from the network.
//!
//! # What these prove that the unit tests cannot
//!
//! Everything below `kopitiam-runtime` is already covered by unit tests against
//! *synthetic* GGUF fixtures. Those prove the code is self-consistent. They
//! cannot prove the thing that actually keeps breaking: that a **real** model,
//! as published, survives the whole chain —
//!
//! ```text
//! fetch ──▶ load GGUF ──▶ build tokenizer ──▶ build weights ──▶ generate
//! ```
//!
//! The bug that prompted this file is exactly that gap. A SmolLM2-360M file
//! downloaded fine, verified fine, and then died at the tokenizer with
//! *"byte-level vocab has no single-byte token for byte 0x04"*. Every synthetic
//! fixture passed throughout, because the fixtures were built the way the code
//! expects rather than the way HuggingFace actually ships.
//!
//! # Why each stage is asserted separately
//!
//! The whole diagnostic value is in **which stage** fails. A single
//! `assert!(chat_works())` would have told us nothing we did not already know.
//! So each model reports a [`Stage`], and a failure names the stage and carries
//! the underlying error — the same principle as `models inspect`: a test that
//! only says "broken" is barely better than the bug report.
//!
//! # These do NOT run by default
//!
//! They are gated behind `KOPITIAM_NETFETCH=1` because they need the network and
//! hundreds of MB of disk, which no ordinary `cargo test` run should assume.
//! Without the flag every test here prints what it skipped and passes — a silent
//! skip would let the suite look green while proving nothing.
//!
//! # A machine that cannot download is not a broken machine
//!
//! Plenty of the places this suite runs are walled off from `huggingface.co`:
//! the Claude Code container, the institute proxy (`CONNECT ... 403` — crates.io
//! is on the bypass list, HF is not), any air-gapped box. Turning the gate on
//! there must **not** paint the run red. A suite that cries wolf on other
//! people's egress policy is a suite everyone learns to ignore, and an ignored
//! suite catches nothing.
//!
//! So a model whose bytes cannot be obtained is reported `SKIP`, with the
//! underlying error printed in full, and the run stays green. Only a model that
//! was actually *obtained* and then broke turns it red. When **everything**
//! skips, the report says in as many words that the run proved nothing — the
//! skip is loud, just not fatal. See [`Verdict`] and [`acquire`] for exactly
//! where that line is drawn, and what it costs.
//!
//! Two consequences worth knowing:
//!
//! * Weights already in the store are exercised with **no network at all**, so
//!   a bring-your-own `.gguf` on an air-gapped box tests the full chain fine.
//! * `KOPITIAM_NETFETCH_PATHS` uses the **platform's** path separator (`:` on
//!   Unix/Termux, `;` on Windows), so it works on both.
//!
//! # Small models by default
//!
//! A routine run exercises only the **small** catalog weights (see
//! [`MAX_DEFAULT_ARTIFACT_BYTES`]). The big ones are not a different kind of
//! test, just a far more expensive one — SmolLM2-1.7B is 2.7× the file of the
//! 360M and **45×** the wall clock, because the cost is the CPU forward pass. A
//! suite nobody runs because it takes eight minutes proves nothing, so the
//! default stays in the seconds and the big models are opt-in. Anything left out
//! is printed by name with the reason, never silently dropped.
//!
//! ```bash
//! # the small chat weights in the catalog — the routine run
//! KOPITIAM_NETFETCH=1 cargo test --release -p kopitiam-runtime --test netfetch_end_to_end -- --nocapture
//!
//! # include the big ones too
//! KOPITIAM_NETFETCH=1 KOPITIAM_NETFETCH_BIG=1 \
//!   cargo test --release -p kopitiam-runtime --test netfetch_end_to_end -- --nocapture
//!
//! # one model only — runs whatever its size, because you named it
//! KOPITIAM_NETFETCH=1 KOPITIAM_NETFETCH_ONLY=smollm2-1.7b-instruct-q4_k_m \
//!   cargo test --release -p kopitiam-runtime --test netfetch_end_to_end -- --nocapture
//!
//! # a model that is NOT in the catalog (e.g. a Gemma you dropped in by hand).
//! # Needs no network at all — this is the air-gapped path.
//! KOPITIAM_NETFETCH=1 KOPITIAM_NETFETCH_PATHS=/path/to/gemma-3-1b-it-Q4_K_M.gguf \
//!   cargo test --release -p kopitiam-runtime --test netfetch_end_to_end -- --nocapture
//! ```
//!
//! On Windows PowerShell the same thing, remembering `;` separates paths there:
//!
//! ```powershell
//! $env:KOPITIAM_NETFETCH=1
//! $env:KOPITIAM_NETFETCH_PATHS="C:\models\gemma.gguf;C:\models\phi.gguf"
//! cargo test --release -p kopitiam-runtime --test netfetch_end_to_end -- --nocapture
//! ```
//!
//! `--nocapture` matters: the per-stage report is the output, and a passing test
//! that hides it wastes the run.
//!
//! Weights land in the ordinary XDG model store and are `.gitignore`d; nothing
//! is written into the worktree.

use std::path::{Path, PathBuf};

use kopitiam_models::{Catalog, ModelSpec, ModelStore};
use kopitiam_runtime::tokenizer_from_gguf;

/// Env flag that turns these on at all.
const GATE: &str = "KOPITIAM_NETFETCH";
/// Restrict the run to one catalog id.
const ONLY: &str = "KOPITIAM_NETFETCH_ONLY";
/// Extra `.gguf` paths to exercise alongside the catalog — the escape hatch
/// for a model whose HuggingFace coordinates are not in the catalog yet.
///
/// Separated by the **platform's own** `PATH` separator: `:` on Unix/Termux,
/// `;` on Windows. It cannot be a hardcoded `:`, because that tears
/// `C:\models\gemma.gguf` into `C` and `\models\gemma.gguf` and then reports
/// two files that do not exist — which is exactly what it did on the
/// maintainer's Windows box the first time this hatch was used.
const EXTRA_PATHS: &str = "KOPITIAM_NETFETCH_PATHS";

/// How far a model got before something broke.
///
/// Ordered so `<` means "got less far", which is what makes a stage a useful
/// thing to report rather than just a label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    Fetch,
    LoadGguf,
    Tokenizer,
    Weights,
    Generate,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Stage::Fetch => "fetch",
            Stage::LoadGguf => "load-gguf",
            Stage::Tokenizer => "tokenizer",
            Stage::Weights => "weights",
            Stage::Generate => "generate",
        }
    }
}

/// What became of one model.
///
/// Three outcomes, not two, and the third is the whole point of this type:
/// **"this machine is not allowed to download the weights" is not a bug in
/// KOPITIAM**, so it must not be reported as one. The Claude Code container and
/// the institute proxy both refuse `huggingface.co`; a suite that goes red
/// there is a suite people learn to ignore, and an ignored suite catches
/// nothing.
enum Verdict {
    /// Survived the whole chain.
    Passed,
    /// Died at a stage. A real failure — this is what turns the run red.
    Died(Stage, String),
    /// Could not obtain the bytes at all. Reported, not failed.
    NoEgress(String),
}

/// The outcome for one model: how far it got, and what became of it.
struct Outcome {
    label: String,
    reached: Stage,
    verdict: Verdict,
}

impl Outcome {
    /// Only [`Verdict::Died`] is a failure. A skip is deliberately *not* one.
    fn failed(&self) -> bool {
        matches!(self.verdict, Verdict::Died(..))
    }

    fn skipped(&self) -> bool {
        matches!(self.verdict, Verdict::NoEgress(_))
    }

    fn line(&self) -> String {
        match &self.verdict {
            Verdict::Passed => {
                format!("  PASS  {:<34} reached {}", self.label, self.reached.name())
            }
            Verdict::Died(stage, err) => format!(
                "  FAIL  {:<34} died at {}: {}",
                self.label,
                stage.name(),
                first_line(err)
            ),
            // Prints the underlying error in full rather than a tidy "no
            // network": if the real cause is a wrong catalog URL rather than a
            // blocked proxy, that must still be visible to a human reading the
            // output. The run does not go red for it, so the text is the only
            // signal left — do not tidy it away.
            Verdict::NoEgress(err) => format!(
                "  SKIP  {:<34} cannot fetch here: {}",
                self.label,
                first_line(err)
            ),
        }
    }
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or(s)
}

/// Whether a given gate value means "on".
///
/// Split out as a **pure** function of the value on purpose. The obvious way
/// to test the gate is to `set_var`/`remove_var` and re-read it — and that is
/// a trap, because `cargo test` runs every test in this file as threads of one
/// process, so a guard test clearing `KOPITIAM_NETFETCH` yanks it out from
/// under the real run happening concurrently. That actually happened: the full
/// suite printed `SKIPPED` and reported green **with the gate set**, which is
/// precisely the silent-green outcome this file's module docs promise to
/// prevent. Testing the pure function touches no shared state, so it cannot.
fn gate_value_means_on(v: Option<&str>) -> bool {
    v.is_some_and(|v| !v.is_empty() && v != "0")
}

fn enabled() -> bool {
    gate_value_means_on(std::env::var(GATE).ok().as_deref())
}

/// Runs the full chain against one on-disk `.gguf`, stopping at the first
/// failure and reporting where it stopped.
///
/// Deliberately does NOT use `?` to bail out of the test: a Gemma failing must
/// not prevent SmolLM2 being reported, or a single bad model hides the state of
/// every other one.
fn exercise(label: &str, path: &Path) -> Outcome {
    let mut reached = Stage::Fetch;

    let model = match kopitiam_loader::load_model(path) {
        Ok(m) => m,
        Err(e) => {
            return Outcome {
                label: label.into(),
                reached,
                verdict: Verdict::Died(Stage::LoadGguf, e.to_string()),
            };
        }
    };
    reached = Stage::LoadGguf;

    // The stage that actually broke on a real SmolLM2. Built separately from the
    // model — even though `QwenModel::from_loaded_model` could build its own —
    // precisely because it is the historically-failing step and the one whose
    // failure we most need attributed to `tokenizer`, not lumped under `weights`.
    let tokenizer = match tokenizer_from_gguf(&model) {
        Ok(t) => t,
        Err(e) => {
            return Outcome {
                label: label.into(),
                reached,
                verdict: Verdict::Died(Stage::Tokenizer, e.to_string()),
            };
        }
    };
    reached = Stage::Tokenizer;

    // Config + weights + rope in one call. A failure here is genuinely the
    // weights stage (shape mismatch, a tensor the loader cannot dequantize),
    // distinct from the tokenizer failure above.
    let qwen = match kopitiam_runtime::QwenModel::from_loaded_model(&model) {
        Ok(m) => m,
        Err(e) => {
            return Outcome {
                label: label.into(),
                reached,
                verdict: Verdict::Died(Stage::Weights, e.to_string()),
            };
        }
    };
    reached = Stage::Weights;

    match generate_something(&qwen, &tokenizer) {
        Ok(()) => Outcome {
            label: label.into(),
            reached: Stage::Generate,
            verdict: Verdict::Passed,
        },
        Err(e) => Outcome {
            label: label.into(),
            reached,
            verdict: Verdict::Died(Stage::Generate, e),
        },
    }
}

/// Gets one catalog model onto disk, telling "cannot download here" apart from
/// "downloaded and it is wrong".
///
/// Two things are going on, and both matter:
///
/// **Already-present weights never touch the network.** `ensure_available`
/// verifies the on-disk bytes against the catalog sha256 locally and only
/// downloads when the file is missing or wrong. Its `_resolving` sibling, by
/// contrast, re-resolves the sha256 from the hub *first, unconditionally* — so
/// using that one would make an offline box fail even with correct weights
/// already sitting in the store. We only pay for that extra defence-in-depth
/// when we are actually about to download.
///
/// **Only [`kopitiam_models::Error::Http`] becomes a skip.** That variant means
/// the bytes could not be obtained — blocked proxy, no DNS, dead TLS. Everything
/// else means we *did* get bytes, or did not need any, and something is really
/// wrong: a `ChecksumMismatch` is a corrupt download or a silently re-quantized
/// upstream, an `Io` is a broken disk, a `NotFound` is a bad catalog id. Those
/// stay failures. The cost of this split is honest and worth writing down: a
/// catalog URL that 404s also surfaces as `Http`, so it will be skipped rather
/// than failed. We cannot tell the two apart from here without pattern-matching
/// ureq's `Display` text, which is not a contract — so instead the skip line
/// prints the full error, and a 404 stays visible to anyone reading the output.
fn acquire(
    store: &ModelStore,
    spec: &ModelSpec,
    fetcher: &kopitiam_models::HttpFetcher,
) -> Result<PathBuf, Verdict> {
    let path = store.artifact_path(spec, &spec.artifacts[0]);
    let result = if path.is_file() {
        kopitiam_models::ensure_available(store, spec, fetcher)
    } else {
        kopitiam_models::ensure_available_resolving(store, spec, fetcher, None)
    };
    match result {
        Ok(_) => Ok(path),
        Err(kopitiam_models::Error::Http(e)) => Err(Verdict::NoEgress(e)),
        Err(e) => Err(Verdict::Died(Stage::Fetch, e.to_string())),
    }
}

/// Round-trips a prompt through tokenizer + forward pass and insists the model
/// produces *something decodable*.
///
/// Correctness of the text is explicitly NOT asserted — a quantized 360M model
/// answering a prompt has no single right answer, and pinning one would give a
/// test that fails on every upstream requant for no reason. What is asserted is
/// the property that actually regresses silently: the chain runs and yields
/// in-vocabulary, decodable tokens rather than garbage or a panic.
fn generate_something(
    model: &kopitiam_runtime::QwenModel,
    tokenizer: &kopitiam_tokenizer::BpeTokenizer,
) -> Result<(), String> {
    use kopitiam_tokenizer::Tokenizer;

    let prompt = "The capital of France is";
    let ids = tokenizer.encode(prompt).map_err(|e| format!("encoding the prompt: {e}"))?;
    if ids.is_empty() {
        return Err("tokenizer encoded a non-empty prompt to zero tokens".into());
    }

    let round_trip = tokenizer.decode(&ids).map_err(|e| format!("decoding back: {e}"))?;
    if round_trip.trim().is_empty() {
        return Err(format!("prompt {prompt:?} did not survive an encode/decode round trip"));
    }

    let cfg = kopitiam_runtime::GenerationConfig { max_new_tokens: 8, ..Default::default() };
    let out = kopitiam_runtime::generate(model, tokenizer, prompt, &cfg, |_, _| {})
        .map_err(|e| format!("{e}"))?;
    if out.trim().is_empty() {
        return Err("generate produced no text at all".into());
    }
    println!("        prompt {prompt:?} -> {:?}", first_line(&out));
    Ok(())
}

/// The artifact size above which a model is **not** run by default.
///
/// Half a gigabyte, and the line is drawn from measurement rather than taste.
/// On a 14-core ThinkPad the two SmolLM2 entries came in at:
///
/// | model | artifact | whole chain |
/// |---|---|---|
/// | SmolLM2-360M-Instruct Q8_0 | 386 MB | ~10 s |
/// | SmolLM2-1.7B-Instruct Q4_K_M | 1.01 GB | ~453 s |
///
/// A 2.7× file for a **45×** wall-clock is the shape of the problem: the cost is
/// the CPU forward pass, not the download, and it tracks size steeply. Termux on
/// a tablet — a first-class target here — has a fraction of that machine's cores
/// and RAM, so a routine run must stay in the seconds. 512 MB sits cleanly
/// between the two and needs no revisiting for either.
///
/// This is a *default*, not a ban. Naming a model with
/// `KOPITIAM_NETFETCH_ONLY` runs it whatever its size (an explicit request beats
/// a default), and `KOPITIAM_NETFETCH_BIG=1` includes every size.
const MAX_DEFAULT_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;

/// Include models above [`MAX_DEFAULT_ARTIFACT_BYTES`] too.
const INCLUDE_BIG: &str = "KOPITIAM_NETFETCH_BIG";

/// Catalog entries that are chat weights (single `.gguf` artifact), honouring
/// `KOPITIAM_NETFETCH_ONLY`.
///
/// Returns `(runnable, notes)`, where each note is `(id, why it was left out)`.
/// Two things get left out, and **both are reported by name rather than quietly
/// dropped** — an unexplained absence reads as coverage, which is how a suite
/// starts lying about what it proved:
///
/// * **Placeholder checksums.** Those entries are *documented* to fetch hundreds
///   of MB and then be refused by their own gate. That is the catalog saying
///   "this URL was never confirmed", not a runtime defect worth 1.1 GB of
///   somebody's bandwidth to rediscover.
/// * **Big models.** See [`MAX_DEFAULT_ARTIFACT_BYTES`].
fn selected_specs() -> (Vec<ModelSpec>, Vec<(String, String)>) {
    let only = std::env::var(ONLY).ok().filter(|s| !s.is_empty());
    // An explicitly named model is never filtered for size: the caller asked for
    // that one specifically, and silently doing nothing would be the worst
    // possible answer to an explicit request.
    let named = only.is_some();
    let include_big = named || gate_value_means_on(std::env::var(INCLUDE_BIG).ok().as_deref());

    let mut runnable = Vec::new();
    let mut notes = Vec::new();

    for spec in Catalog::builtin() {
        if spec.artifacts.is_empty()
            || !spec.artifacts.iter().all(|a| a.filename.to_ascii_lowercase().ends_with(".gguf"))
        {
            continue; // not chat weights (e.g. the Tesseract `.traineddata` entries)
        }
        if !only.as_ref().is_none_or(|id| &spec.id == id) {
            continue; // narrowed away by KOPITIAM_NETFETCH_ONLY — not worth a note
        }
        if spec.artifacts.iter().any(kopitiam_models::Artifact::is_placeholder) {
            notes.push((
                spec.id.clone(),
                "catalog checksum is still the placeholder — record a real sha256 \
                 (or give it an `hf:` source) to include it"
                    .to_string(),
            ));
            continue;
        }
        let bytes: u64 = spec.artifacts.iter().map(|a| a.size_bytes).sum();
        if !include_big && bytes > MAX_DEFAULT_ARTIFACT_BYTES {
            notes.push((
                spec.id.clone(),
                format!(
                    "{} MB is over the {} MB default cap — set {INCLUDE_BIG}=1, or \
                     {ONLY}={} to run just this one",
                    bytes / (1024 * 1024),
                    MAX_DEFAULT_ARTIFACT_BYTES / (1024 * 1024),
                    spec.id
                ),
            ));
            continue;
        }
        runnable.push(spec);
    }
    (runnable, notes)
}

fn extra_paths() -> Vec<PathBuf> {
    // `split_paths` uses the platform separator and, on Windows, also strips
    // the quotes people put around paths with spaces. Doing it by hand with a
    // literal `:` is what broke `C:\...` — see [`EXTRA_PATHS`].
    std::env::var_os(EXTRA_PATHS)
        .into_iter()
        .flat_map(|v| std::env::split_paths(&v).collect::<Vec<_>>())
        .filter(|p| !p.as_os_str().is_empty())
        .collect()
}

#[test]
fn real_models_survive_fetch_load_tokenize_and_generate() {
    if !enabled() {
        println!(
            "SKIPPED: set {GATE}=1 to run (needs network + hundreds of MB of disk).\n\
             Weights go to the XDG model store and are gitignored."
        );
        return;
    }

    let store = ModelStore::with_default_root()
        .expect("model store root (set HOME or XDG_CACHE_HOME)");
    let fetcher = kopitiam_models::HttpFetcher::new();

    let mut outcomes = Vec::new();

    let (runnable, notes) = selected_specs();
    for (id, why) in &notes {
        println!("  NOTE  {id:<34} not run: {why}");
    }

    for spec in runnable {
        let label = spec.id.clone();
        // Fetch is its own stage: "cannot even download it" and "downloaded but
        // will not load" are completely different problems and must not be
        // reported as the same failure.
        match acquire(&store, &spec, &fetcher) {
            Ok(path) => outcomes.push(exercise(&label, &path)),
            Err(verdict) => outcomes.push(Outcome { label, reached: Stage::Fetch, verdict }),
        }
    }

    for path in extra_paths() {
        let label = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        if !path.is_file() {
            // A hand-named path that is not there is the user's typo, not an
            // egress problem — they said "test THIS file". Stays a failure.
            outcomes.push(Outcome {
                label,
                reached: Stage::Fetch,
                verdict: Verdict::Died(
                    Stage::Fetch,
                    format!("no such file: {}", path.display()),
                ),
            });
            continue;
        }
        outcomes.push(exercise(&label, &path));
    }

    println!("\nnetfetch end-to-end results:");
    for o in &outcomes {
        println!("{}", o.line());
    }

    assert!(
        !outcomes.is_empty(),
        "{GATE} was set but nothing was selected — check {ONLY}/{EXTRA_PATHS}"
    );

    // Say plainly when nothing was actually proven. The gate being on means
    // somebody WANTED a real run, so a run that downloaded nothing must not
    // slip past looking like a success — but it must not go red either, because
    // "your container has no egress" is not a defect in this codebase.
    let skipped = outcomes.iter().filter(|o| o.skipped()).count();
    if skipped == outcomes.len() {
        println!(
            "\nNOTE: all {skipped} model(s) skipped — this machine cannot fetch weights.\n\
             The chain was NOT exercised, so this run proves nothing about the runtime.\n\
             Run it where huggingface.co is reachable, or drop a .gguf in and point\n\
             {EXTRA_PATHS} at it (separator: `{}`) to test with no network at all.",
            if cfg!(windows) { ';' } else { ':' }
        );
    } else if skipped > 0 {
        println!("\nNOTE: {skipped} of {} model(s) skipped — cannot fetch here.", outcomes.len());
    }

    let failed: Vec<&Outcome> = outcomes.iter().filter(|o| o.failed()).collect();
    assert!(
        failed.is_empty(),
        "{} of {} model(s) did not survive the chain:\n{}",
        failed.len(),
        outcomes.len(),
        failed.iter().map(|o| o.line()).collect::<Vec<_>>().join("\n")
    );
}

#[test]
fn the_gate_is_off_by_default_so_an_ordinary_test_run_needs_no_network() {
    // Guards the property that makes this file safe to have in the tree at all.
    // If the gate ever defaults on, `cargo test` starts downloading hundreds of
    // MB on somebody's metered Termux connection with no warning.
    assert!(!gate_value_means_on(None), "the netfetch gate must be OFF when {GATE} is unset");
}

#[test]
fn an_empty_or_zero_gate_value_does_not_count_as_enabled() {
    // `KOPITIAM_NETFETCH=` (exported but empty) and `=0` are both the shell's
    // usual way of saying "no". Treating either as "yes" would surprise someone
    // trying to turn it off.
    //
    // Deliberately does NOT touch the process environment — see
    // [`gate_value_means_on`] for the concurrent-clobber bug that cost us a
    // whole run that looked green while proving nothing.
    for v in ["", "0"] {
        assert!(!gate_value_means_on(Some(v)), "{GATE}={v:?} must not enable the netfetch run");
    }
    assert!(gate_value_means_on(Some("1")), "{GATE}=1 must enable the netfetch run");
}
