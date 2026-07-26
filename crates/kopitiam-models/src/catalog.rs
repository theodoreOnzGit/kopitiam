//! The curated model catalog: what models KOPITIAM knows how to acquire, and
//! everything needed to acquire them and later run them.
//!
//! Two design points worth saying upfront:
//!
//! * **Not Qwen-only.** [`Architecture`] is a proper enum with more than one
//!   family, and [`Catalog::builtin`] ships at least two *different* families
//!   (a Qwen2 one and a Llama one). The acquisition layer must stay
//!   model-agnostic -- the forward pass that eventually runs is picked from
//!   [`Architecture`], not hardcoded to any one vendor.
//! * **Data, not logic.** [`Artifact`], [`ModelSpec`] and [`Architecture`] are
//!   `Serialize`/`Deserialize`, so one fine day the catalog can come from a
//!   JSON/TOML file instead of being baked into the binary. Keep them pure
//!   data hor -- no I/O, no network, no hashing lives in here.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Which model family / architecture this is. Drives which forward pass runs
/// later on, downstream in the runtime -- so it is NOT just a label leh, it is
/// load-bearing.
///
/// [`Architecture::Other`] is the escape hatch for a family the catalog knows
/// about by name but this enum has not grown a dedicated variant for yet. The
/// downstream runtime is free to reject an `Other(..)` it cannot run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Architecture {
    /// Qwen2 / Qwen2.5 family.
    Qwen2,
    /// Meta Llama family (Llama 2 / 3 / 3.1 / 3.2 ...).
    Llama,
    /// Microsoft Phi-3 family.
    Phi3,
    /// Google Gemma family.
    Gemma,
    /// Any family not (yet) given its own variant. The string is the family
    /// name as the catalog knows it.
    Other(String),
}

/// A pointer at a single file inside a HuggingFace repo, kept on an
/// [`Artifact`] so its sha256 can be **auto-resolved from the HF hub** instead
/// of hand-pasted.
///
/// This is pure data -- the *shape* of "which HF file this artifact is". The
/// actual hub call that turns it into a sha256 lives in [`crate::hf`] (behind
/// the `net` feature), exactly like every other network touch in this crate is
/// kept out of the offline core.
///
/// `revision` is a plain string on purpose (a branch/tag like `main`, or a
/// 40-char commit SHA) -- the same value that goes into the `resolve`/`tree`
/// URL. The richer [`crate::hf::Revision`] type (which enforces the
/// reproducible-vs-moving distinction) lives up in `hf`; keeping this one a
/// bare `String` is what stops `catalog` (pure data) from having to depend on
/// `hf` (logic + net).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HfSource {
    /// The `owner/name` repo slug, e.g. `HuggingFaceTB/SmolLM2-360M-Instruct-GGUF`.
    pub repo: String,
    /// Which revision to resolve against: a branch/tag (`main`) or a commit SHA.
    pub revision: String,
    /// The repo-relative path of the file, e.g. `smollm2-360m-instruct-q8_0.gguf`.
    pub file: String,
}

/// One downloadable thing on disk.
///
/// Usually this is the single `.gguf` weights file. But some models need an
/// extra companion file (e.g. a separate tokenizer), so a [`ModelSpec`] holds a
/// *list* of these, and every one of them must land and verify before the model
/// counts as acquired.
///
/// The [`Artifact::sha256`] is the verification gate: after the bytes are on
/// disk (whether freshly fetched or bring-your-own), they get hashed and
/// checked against this exact value. Lowercase hex, always.
///
/// ## Auto-resolved checksums
///
/// [`Artifact::hf`] (optional) records that this artifact comes from a
/// HuggingFace repo. When it is `Some`, the acquire path can **resolve the
/// sha256 from the HF hub** rather than trusting only a hand-pasted value (see
/// [`crate::hf::resolve_hf_sha256`] and [`crate::hf::resolve_spec`]):
///
/// * If [`Artifact::sha256`] is **empty**, it is filled in from the hub -- so a
///   catalog author can add a `{repo, file}` entry with *no* hash and let the
///   machine fetch it.
/// * If [`Artifact::sha256`] is a **real** pinned hash, the resolved one is
///   cross-checked against it (defense in depth) and a mismatch is a hard
///   [`crate::Error::ChecksumMismatch`].
///
/// When `hf` is `None`, nothing changes: the pinned `sha256` is the whole story,
/// exactly as before.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// The local filename to save as, e.g. `"qwen2.5-0.5b-instruct-q4_0.gguf"`.
    /// This is the name inside the store, NOT the last path segment of the URL
    /// (those two can differ).
    pub filename: String,
    /// Where autofetch pull the bytes from.
    pub url: String,
    /// Lowercase-hex sha256 of the expected bytes. The verification gate.
    ///
    /// May be **empty** *only* when [`Artifact::hf`] is `Some` -- that spells
    /// "resolve me from the hub". A non-empty value is always a pinned hash and
    /// must be 64 lowercase-hex chars.
    pub sha256: String,
    /// Expected size in bytes. Handy for progress reporting and a cheap
    /// sanity-check; the sha256 is the real guarantee though. `0` is allowed for
    /// an [`Artifact::hf`] artifact whose size is resolved from the hub.
    pub size_bytes: u64,
    /// Optional HuggingFace origin, enabling sha256 auto-resolution. `None` for a
    /// plain URL artifact (the historical shape). Defaults to `None` when a
    /// serialized catalog omits it, so old data still loads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf: Option<HfSource>,
}

/// A catalog entry: one model, everything needed to acquire it and later run it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Stable key, e.g. `"qwen2.5-0.5b-instruct-q4_0"`. This is what
    /// [`Catalog::find`] and the CLI use to look the model up, so once it is out
    /// there, don't simply change it.
    pub id: String,
    /// Human-friendly name for showing to the user.
    pub display_name: String,
    /// Which family this is -- drives the forward pass later.
    pub architecture: Architecture,
    /// SPDX licence id of the *model weights themselves*, e.g. `"Apache-2.0"` or
    /// `"Llama-3.2-Community"`. This is the model's licence, separate from
    /// KOPITIAM's own AGPL-3.0-only code licence.
    pub license: String,
    /// Every file that must be present and verified. Usually one `.gguf`.
    pub artifacts: Vec<Artifact>,
}

/// The built-in curated catalog.
///
/// A zero-size handle -- all the methods are associated functions. It is a type
/// (not a bare module) so that later a data-driven catalog loaded from a file
/// can live behind the same name without breaking callers.
pub struct Catalog;

impl Catalog {
    /// The whole built-in catalog, one entry per known model.
    ///
    /// # WARNING -- some checksums here are placeholders, on purpose
    ///
    /// The two **SmolLM2** entries (the default [`DEFAULT_MODEL_ID`] and its
    /// larger sibling) carry **real** sha256/size values, read from the
    /// git-LFS pointer published on the official HuggingFaceTB GGUF repos, so
    /// they acquire and verify clean. The small Tesseract `tessdata-*` OCR
    /// entries likewise carry genuine sha256/size (their `.traineddata` files
    /// were pulled and hashed for real).
    ///
    /// The remaining GGUF LLM entries (Qwen2, Llama) still use the sentinel
    /// value `"0000...0000"` (64 zeros), NOT a real hash. Reason: a real
    /// sha256 can only be gotten by actually downloading the hundreds-of-MB
    /// weights file and hashing it, which was not done at authoring time for
    /// those. Better to be honest than to ship a catalog that lies about
    /// hashes it never checked.
    ///
    /// The direct consequence for the *placeholder* entries, and this is by
    /// design: any [`crate::ModelStore::verify`] or [`crate::ensure_available`]
    /// on a Qwen2/Llama entry **will fail with
    /// [`crate::Error::ChecksumMismatch`]** the moment real bytes land --
    /// because real bytes will never hash to 64 zeros. Each carries a `TODO:
    /// record real sha256 after first successful pull`. The workflow is: pull
    /// once, read the `actual` value out of the `ChecksumMismatch` error,
    /// eyeball it against the upstream's published checksum, then paste it in
    /// here. Only then does that entry acquire clean. The SmolLM2 and
    /// `tessdata-*` entries already went through this and carry real hashes.
    ///
    /// The URLs are real HuggingFace GGUF locations. Any entry whose exact
    /// filename we are not 100% sure of carries a `// TODO(verify-url)`.
    ///
    /// This ships **two different families** on purpose (SmolLM2/Llama and
    /// Qwen2), so the model-agnostic promise is not just talk.
    pub fn builtin() -> Vec<ModelSpec> {
        // maintainer will populate: once specific HF-hosted models are named,
        // declare them the clean way via [`crate::hf::HfModel`] (repo / revision
        // / filename / sha256) and `.into_spec()`, ideally pinning a commit
        // [`crate::hf::Revision`] for reproducibility. See bead
        // `kopitiam-56q`. The two hand-written entries below predate that
        // mechanism and stay until the real models + hashes are chosen.
        vec![
            // ---- SmolLM2 family (the DEFAULT) ----------------------------
            // SmolLM2-360M-Instruct, Q8_0 GGUF. Tiny (~369MB), Apache-2.0,
            // HuggingFaceTB. This is the sensible default first pull ([`DEFAULT_MODEL_ID`]) --
            // small and fast enough for local-first / Termux, and the
            // successor to the old Qwen2.5-0.5B default. SmolLM2 is a
            // LLaMA-shaped architecture (`is_llama_config: true` in its
            // upstream configs -- see the vendored reference clone at
            // `crates/kopitiam-models/vendor/smollm`), hence
            // [`Architecture::Llama`].
            //
            // Unlike the Qwen2 / Llama entries below, these two SmolLM2
            // sha256/size values are REAL -- taken from the git-LFS pointer
            // (`oid sha256:`) published on the official HuggingFaceTB GGUF
            // repos -- so they acquire and verify clean.
            ModelSpec {
                id: "smollm2-360m-instruct-q8_0".to_string(),
                display_name: "SmolLM2 360M Instruct (Q8_0)".to_string(),
                architecture: Architecture::Llama,
                license: "Apache-2.0".to_string(),
                artifacts: vec![Artifact {
                    filename: "smollm2-360m-instruct-q8_0.gguf".to_string(),
                    // Official HuggingFaceTB GGUF repo.
                    url: "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf".to_string(),
                    // REAL sha256 -- HF git-LFS oid, cross-checked via the
                    // tree API and the /raw/ LFS pointer. Kept PINNED, and the
                    // `hf` source below makes the acquire path re-resolve it
                    // from the hub and cross-check (defense in depth): if
                    // upstream ever re-quantises, the resolved oid stops matching
                    // this pin and acquisition fails loudly instead of silently.
                    sha256: "48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201".to_string(),
                    size_bytes: 386_404_992,
                    hf: Some(HfSource {
                        repo: "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF".to_string(),
                        revision: "main".to_string(),
                        file: "smollm2-360m-instruct-q8_0.gguf".to_string(),
                    }),
                }],
            },
            // SmolLM2-1.7B-Instruct, Q4_K_M GGUF. The larger local option --
            // still Apache-2.0, still HuggingFaceTB, still LLaMA-shaped.
            ModelSpec {
                id: "smollm2-1.7b-instruct-q4_k_m".to_string(),
                display_name: "SmolLM2 1.7B Instruct (Q4_K_M)".to_string(),
                architecture: Architecture::Llama,
                license: "Apache-2.0".to_string(),
                artifacts: vec![Artifact {
                    filename: "smollm2-1.7b-instruct-q4_k_m.gguf".to_string(),
                    url: "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct-q4_k_m.gguf".to_string(),
                    // REAL sha256 -- HF git-LFS oid, cross-checked via the
                    // tree API and the /raw/ LFS pointer. PINNED + `hf`-resolved
                    // for the same defense-in-depth as the 360M entry above.
                    sha256: "decd2598bc2c8ed08c19adc3c8fdd461ee19ed5708679d1c54ef54a5a30d4f33".to_string(),
                    size_bytes: 1_055_609_536,
                    hf: Some(HfSource {
                        repo: "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF".to_string(),
                        revision: "main".to_string(),
                        file: "smollm2-1.7b-instruct-q4_k_m.gguf".to_string(),
                    }),
                }],
            },
            // ---- Qwen2 family (kept, but NO LONGER the default) ----------
            // Qwen2.5-0.5B-Instruct, Q4_0 GGUF. Small (~350MB), Apache-2.0.
            // Kept available so existing `kopitiam models pull
            // qwen2.5-0.5b-instruct-q4_0` users don't break; it is no longer
            // the default (SmolLM2-360M is). Its sha256 stays a placeholder.
            ModelSpec {
                id: "qwen2.5-0.5b-instruct-q4_0".to_string(),
                display_name: "Qwen2.5 0.5B Instruct (Q4_0)".to_string(),
                architecture: Architecture::Qwen2,
                license: "Apache-2.0".to_string(),
                artifacts: vec![Artifact {
                    filename: "qwen2.5-0.5b-instruct-q4_0.gguf".to_string(),
                    // Official Qwen GGUF repo on HuggingFace.
                    // TODO(verify-url): confirm the exact quant filename in the
                    // repo -- Qwen sometimes name it `-q4_0.gguf` and sometimes
                    // `-fp16.gguf` etc.
                    url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf".to_string(),
                    // TODO: record real sha256 after first successful pull.
                    sha256: PLACEHOLDER_SHA256.to_string(),
                    size_bytes: 352_000_000,
                    hf: None,
                }],
            },
            // ---- Llama family --------------------------------------------
            // Llama-3.2-1B-Instruct, Q4_0 GGUF. Different family on purpose, so
            // the acquisition layer is proven multi-family, not Qwen-shaped.
            // Note the licence is Meta's community licence, NOT Apache-2.0.
            ModelSpec {
                id: "llama-3.2-1b-instruct-q4_0".to_string(),
                display_name: "Llama 3.2 1B Instruct (Q4_0)".to_string(),
                architecture: Architecture::Llama,
                license: "Llama-3.2-Community".to_string(),
                artifacts: vec![Artifact {
                    filename: "llama-3.2-1b-instruct-q4_0.gguf".to_string(),
                    // A community GGUF conversion (Meta's own repo is gated).
                    // TODO(verify-url): confirm repo + exact filename; unsloth
                    // and bartowski both publish Llama-3.2-1B GGUFs and their
                    // quant filenames differ.
                    url: "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf".to_string(),
                    // TODO: record real sha256 after first successful pull.
                    sha256: PLACEHOLDER_SHA256.to_string(),
                    size_bytes: 771_000_000,
                    hf: None,
                }],
            },
            // ---- Tesseract LSTM OCR models -------------------------------
            // These are NOT GGUF LLM weights -- they are Tesseract's float
            // LSTM `.traineddata` files from the `tessdata_best` repo (the
            // most accurate, correct for technical + CJK papers). The store
            // treats a `.traineddata` as just another verified Artifact, so
            // the OCR engine fetches them through the same ModelStore. Family
            // is carried as `Other("tesseract-lstm")` rather than a dedicated
            // enum variant, so every existing exhaustive `match` on
            // `Architecture` (including the CLI's) stays exhaustive. Unlike
            // the two GGUF entries above, these sha256/size values are REAL --
            // computed from an actual pull of the pinned `main` blobs -- so
            // these entries acquire and verify clean.
            ModelSpec {
                id: "tessdata-eng".to_string(),
                display_name: "Tesseract English (LSTM, best)".to_string(),
                architecture: Architecture::Other("tesseract-lstm".to_string()),
                license: "Apache-2.0".to_string(),
                artifacts: vec![Artifact {
                    filename: "eng.traineddata".to_string(),
                    url: "https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata".to_string(),
                    sha256: "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba".to_string(),
                    size_bytes: 15_400_601,
                    hf: None,
                }],
            },
            ModelSpec {
                id: "tessdata-chi_sim".to_string(),
                display_name: "Tesseract Simplified Chinese (LSTM, best)".to_string(),
                architecture: Architecture::Other("tesseract-lstm".to_string()),
                license: "Apache-2.0".to_string(),
                artifacts: vec![Artifact {
                    filename: "chi_sim.traineddata".to_string(),
                    url: "https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_sim.traineddata".to_string(),
                    sha256: "4fef2d1306c8e87616d4d3e4c6c67faf5d44be3342290cf8f2f0f6e3aa7e735b".to_string(),
                    size_bytes: 13_077_423,
                    hf: None,
                }],
            },
            ModelSpec {
                id: "tessdata-jpn".to_string(),
                display_name: "Tesseract Japanese (LSTM, best)".to_string(),
                architecture: Architecture::Other("tesseract-lstm".to_string()),
                license: "Apache-2.0".to_string(),
                artifacts: vec![Artifact {
                    filename: "jpn.traineddata".to_string(),
                    url: "https://github.com/tesseract-ocr/tessdata_best/raw/main/jpn.traineddata".to_string(),
                    sha256: "36bdf9ac823f5911e624c30d0553e890b8abc7c31a65b3ef14da943658c40b79".to_string(),
                    size_bytes: 14_330_109,
                    hf: None,
                }],
            },
        ]
    }

    /// Find one entry by its stable [`ModelSpec::id`]. `None` if no such id.
    ///
    /// Returns a clone (owned [`ModelSpec`]) -- cheap enough, and it frees the
    /// caller from holding a borrow into the built-in list.
    pub fn find(id: &str) -> Option<ModelSpec> {
        Self::builtin().into_iter().find(|spec| spec.id == id)
    }

    /// The id of the **default local model** -- [`DEFAULT_MODEL_ID`], currently
    /// SmolLM2-360M-Instruct (Q8_0). This is the single source of truth for
    /// "which model when nothing is configured": selection code (e.g. the
    /// CLI's `select_adapter`) should reach for this rather than hardcoding an
    /// id, so the default only ever changes in one place -- here. It is
    /// guaranteed to resolve in [`Catalog::builtin`] (there is a test).
    pub fn default_id() -> &'static str {
        DEFAULT_MODEL_ID
    }

    /// The [`ModelSpec`] of the default local model. Convenience over
    /// `Catalog::find(Catalog::default_id())`; always `Some` for the built-in
    /// catalog, so it unwraps safely (see the accompanying test).
    pub fn default_spec() -> ModelSpec {
        Self::find(DEFAULT_MODEL_ID)
            .expect("the default model id must always resolve in the built-in catalog")
    }

    /// Sanity-check a whole catalog and hand back **every** invariant it breaks
    /// -- empty [`Vec`] means the catalog is clean.
    ///
    /// Why bother: the catalog is just data (baked-in today, from a file one
    /// day), and the rest of this crate *silently assumes* things about it that
    /// nothing checks. The nastiest one: a [`ModelSpec`] with an empty
    /// `artifacts` list makes both [`crate::ModelStore::is_present`] and
    /// [`crate::ModelStore::verify`] pass *vacuously* (an `.all()` / a `for`-loop
    /// over nothing), so a bytes-less spec would wrongly count as "acquired"
    /// with not one byte ever on disk. [`Catalog::validate`] is the guardrail
    /// that catches that (and its friends) before it can bite -- run it over a
    /// file-loaded catalog at load time.
    ///
    /// It collects *all* problems rather than bailing on the first, because a
    /// catalog author wants the whole list in one go, not one-at-a-time.
    ///
    /// What it checks:
    ///
    /// * **Cross-spec:** no two specs share an `id` ([`Catalog::find`] returns
    ///   only the first, so a duplicate is a silently-shadowed entry).
    /// * **Per-spec:** everything in [`ModelSpec::validate`] -- empty id, no
    ///   artifacts, duplicate artifact filenames within the spec, empty
    ///   filename/url, and a malformed sha256.
    ///
    /// Take note hor: the shipped [`Catalog::builtin`] validates **clean**. Its
    /// 64-zero placeholder checksums are still *well-formed* lowercase hex, so
    /// they pass the format check by design -- validation is about the entry's
    /// *shape*, not whether the hash is the real one (that is the download-time
    /// [`crate::ModelStore::verify`] gate's job).
    pub fn validate(specs: &[ModelSpec]) -> Vec<CatalogProblem> {
        let mut problems = Vec::new();

        // Cross-spec: duplicate ids. Walk in order and flag the *second* (and
        // later) sighting of any id, so each extra copy gets reported once.
        let mut seen_ids: Vec<&str> = Vec::with_capacity(specs.len());
        for spec in specs {
            if !spec.id.is_empty() && seen_ids.contains(&spec.id.as_str()) {
                problems.push(CatalogProblem::DuplicateModelId {
                    id: spec.id.clone(),
                });
            }
            seen_ids.push(spec.id.as_str());
            problems.append(&mut spec.validate());
        }

        problems
    }
}

impl ModelSpec {
    /// Sanity-check this one spec on its own and hand back **every** problem it
    /// has ([`Vec`] empty means it is fine). Cross-spec checks (like duplicate
    /// ids across the catalog) live in [`Catalog::validate`], not here -- this
    /// one only knows about itself.
    ///
    /// The checks, and why each is load-bearing:
    ///
    /// * **Empty `id`** -> no stable key for [`Catalog::find`] to look it up by.
    /// * **No artifacts** -> the dangerous one: [`crate::ModelStore::is_present`]
    ///   and [`crate::ModelStore::verify`] both pass vacuously on an empty list,
    ///   so this spec would count as "acquired" with zero bytes on disk.
    /// * **Duplicate artifact filenames** -> two artifacts map to the same path
    ///   `<root>/<id>/<filename>`, so one clobbers the other and `verify` ends
    ///   up hashing the same file twice.
    /// * **Empty artifact `filename`** -> nothing to save the bytes as.
    /// * **Empty artifact `url`** -> autofetch has nowhere to pull from. (BYO
    ///   still works, but the catalog entry is incomplete, so we flag it.)
    /// * **Malformed `sha256`** -> not 64 lowercase-hex chars. The verification
    ///   gate compares lowercase hex, so a malformed value can *never* match
    ///   real bytes -- the entry could never acquire clean. The 64-zero
    ///   placeholder is valid *format*, so it passes here by design.
    pub fn validate(&self) -> Vec<CatalogProblem> {
        let mut problems = Vec::new();

        if self.id.is_empty() {
            problems.push(CatalogProblem::EmptyModelId {
                display_name: self.display_name.clone(),
            });
        }

        if self.artifacts.is_empty() {
            problems.push(CatalogProblem::NoArtifacts {
                id: self.id.clone(),
            });
        }

        // Duplicate filenames within this spec. Flag the second (and later)
        // sighting of each filename, so every colliding pair is reported.
        let mut seen_names: Vec<&str> = Vec::with_capacity(self.artifacts.len());
        for a in &self.artifacts {
            if !a.filename.is_empty() && seen_names.contains(&a.filename.as_str()) {
                problems.push(CatalogProblem::DuplicateArtifactFilename {
                    id: self.id.clone(),
                    filename: a.filename.clone(),
                });
            }
            seen_names.push(a.filename.as_str());

            if a.filename.is_empty() {
                problems.push(CatalogProblem::EmptyArtifactFilename {
                    id: self.id.clone(),
                });
            }
            if a.url.is_empty() {
                problems.push(CatalogProblem::EmptyArtifactUrl {
                    id: self.id.clone(),
                    filename: a.filename.clone(),
                });
            }
            // An empty sha256 is allowed ONLY for an HF-sourced artifact -- it
            // means "resolve me from the hub" (see [`Artifact::hf`]). Any other
            // empty, or any non-empty-but-not-64-lowercase-hex value, is
            // malformed: the download-time gate compares lowercase hex, so it
            // could never match real bytes.
            let unresolved_hf = a.sha256.is_empty() && a.hf.is_some();
            if !unresolved_hf && !is_sha256_hex(&a.sha256) {
                problems.push(CatalogProblem::MalformedSha256 {
                    id: self.id.clone(),
                    filename: a.filename.clone(),
                    sha256: a.sha256.clone(),
                });
            }
        }

        problems
    }
}

/// One thing wrong with a catalog entry, as found by [`ModelSpec::validate`] or
/// [`Catalog::validate`].
///
/// Structured (not a bare string) on purpose: a caller can `match` on exactly
/// which invariant kena broken and point the catalog author straight at it. The
/// [`std::fmt::Display`] text is human-facing; the fields are machine-facing.
///
/// This is a *shape*/data-model problem, deliberately separate from
/// [`crate::Error`], which is about the *acquire* path (I/O, HTTP, the
/// download-time checksum gate). Validation is offline and pure-data -- it never
/// touches disk or network.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CatalogProblem {
    /// Two or more specs share this `id`. [`Catalog::find`] returns only the
    /// first, so every later copy is unreachable -- silently shadowed.
    #[error("duplicate model id `{id}` -- later copies are shadowed by Catalog::find")]
    DuplicateModelId {
        /// The id that appears more than once.
        id: String,
    },

    /// A spec has an empty `id`, so there is no stable key to look it up by.
    /// Carries `display_name` because that is the only thing left to identify
    /// the offending entry by.
    #[error("model with display name `{display_name}` has an empty id")]
    EmptyModelId {
        /// The entry's `display_name`, the only remaining handle on it.
        display_name: String,
    },

    /// A spec has zero artifacts. Fatal for correctness: `is_present` and
    /// `verify` both pass vacuously on an empty list, so the model would count
    /// as "acquired" with no bytes ever on disk.
    #[error("model `{id}` has no artifacts -- it would verify as acquired with zero bytes on disk")]
    NoArtifacts {
        /// The offending spec's id.
        id: String,
    },

    /// Two artifacts in the same spec share this `filename`. They map to the
    /// same on-disk path `<root>/<id>/<filename>`, so one clobbers the other.
    #[error("model `{id}` has two artifacts named `{filename}` -- they collide on the same on-disk path")]
    DuplicateArtifactFilename {
        /// The spec the collision is in.
        id: String,
        /// The filename that appears more than once.
        filename: String,
    },

    /// An artifact has an empty `filename` -- nothing to save the bytes as.
    #[error("model `{id}` has an artifact with an empty filename")]
    EmptyArtifactFilename {
        /// The spec the empty-filename artifact is in.
        id: String,
    },

    /// An artifact has an empty `url`, so autofetch has nowhere to pull from.
    /// BYO would still work, but the catalog entry is incomplete.
    #[error("model `{id}` artifact `{filename}` has an empty url -- autofetch has nowhere to pull from")]
    EmptyArtifactUrl {
        /// The spec the artifact is in.
        id: String,
        /// The artifact's filename.
        filename: String,
    },

    /// An artifact's `sha256` is not 64 lowercase-hex chars. The verification
    /// gate compares lowercase hex, so a malformed value can never match real
    /// bytes -- the entry could never acquire clean. The 64-zero placeholder is
    /// valid *format* and does NOT trip this.
    #[error("model `{id}` artifact `{filename}` has a malformed sha256 `{sha256}` -- want 64 lowercase-hex chars")]
    MalformedSha256 {
        /// The spec the artifact is in.
        id: String,
        /// The artifact's filename.
        filename: String,
        /// The offending value as written in the catalog.
        sha256: String,
    },
}

/// Is `s` exactly 64 lowercase-hex chars (`0-9`, `a-f`)? That is the shape a
/// sha256 must take in this catalog -- see [`Artifact::sha256`]. Uppercase is
/// rejected on purpose: the whole crate compares lowercase hex, so an uppercase
/// digest would never match, and we would rather flag it here than let it fail
/// mysteriously at the download-time gate.
fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The 64-zero sentinel used for every catalog checksum until a real pull
/// records the true value. Kept as one named constant so there is exactly one
/// place to grep for, and so nobody mistakes it for a real hash. See the big
/// warning on [`Catalog::builtin`].
const PLACEHOLDER_SHA256: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// The catalog id of the **default local model**: SmolLM2-360M-Instruct
/// (Q8_0), HuggingFaceTB, Apache-2.0. Tiny and fast enough for local-first /
/// Termux, and the successor to the old Qwen2.5-0.5B default. Exposed via
/// [`Catalog::default_id`] so selection code has one place to read the default
/// from; an explicit `KOPITIAM_MODEL` / BYO override still wins over it.
pub const DEFAULT_MODEL_ID: &str = "smollm2-360m-instruct-q8_0";
