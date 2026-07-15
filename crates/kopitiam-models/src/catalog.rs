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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// The local filename to save as, e.g. `"qwen2.5-0.5b-instruct-q4_0.gguf"`.
    /// This is the name inside the store, NOT the last path segment of the URL
    /// (those two can differ).
    pub filename: String,
    /// Where autofetch pull the bytes from.
    pub url: String,
    /// Lowercase-hex sha256 of the expected bytes. The verification gate.
    pub sha256: String,
    /// Expected size in bytes. Handy for progress reporting and a cheap
    /// sanity-check; the sha256 is the real guarantee though.
    pub size_bytes: u64,
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
    /// # WARNING -- the checksums here are placeholders, on purpose
    ///
    /// Every [`Artifact::sha256`] below is the sentinel value
    /// `"0000...0000"` (64 zeros), NOT a real hash. Reason: a real sha256 can
    /// only be gotten by actually downloading the hundreds-of-MB weights file
    /// and hashing it, which cannot be done at authoring time. Better to be
    /// honest than to ship a catalog that lies about hashes it never checked.
    ///
    /// The direct consequence, and this is by design: any [`crate::verify`]
    /// (via [`crate::ModelStore::verify`]) or [`crate::ensure_available`] on a
    /// shipped entry **will fail with [`crate::Error::ChecksumMismatch`]** the
    /// moment real bytes land -- because real bytes will never hash to 64 zeros.
    /// Each entry carries a `TODO: record real sha256 after first successful
    /// pull`. The workflow is: pull once, read the `actual` value out of the
    /// `ChecksumMismatch` error, eyeball it against the upstream's published
    /// checksum, then paste it in here. Only then does that entry acquire clean.
    ///
    /// The URLs are real, plausible HuggingFace GGUF locations. Any entry whose
    /// exact filename we are not 100% sure of carries a `// TODO(verify-url)`.
    ///
    /// This ships **two different families** on purpose (Qwen2 and Llama), so
    /// the model-agnostic promise is not just talk.
    pub fn builtin() -> Vec<ModelSpec> {
        vec![
            // ---- Qwen2 family --------------------------------------------
            // Qwen2.5-0.5B-Instruct, Q4_0 GGUF. Small (~350MB), Apache-2.0,
            // the sensible default first pull.
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
}

/// The 64-zero sentinel used for every catalog checksum until a real pull
/// records the true value. Kept as one named constant so there is exactly one
/// place to grep for, and so nobody mistakes it for a real hash. See the big
/// warning on [`Catalog::builtin`].
const PLACEHOLDER_SHA256: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";
