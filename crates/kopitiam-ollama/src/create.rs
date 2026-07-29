//! # `ollama create` -- how a Modelfile becomes a manifest with layers
//!
//! **Upstream:** `server/create.go`, `server/model.go`, `server/model_resolver.go`,
//! `server/renderer_resolution.go`, the deferred half of `parser/parser.go`
//! (`CreateRequest`, `fileDigestMap`, `filesForModel`, `expandPath`,
//! `rejectMatchingLocalPath`), and the model-loading half of `server/images.go`
//! (`GetModel` + capability inference) -- ollama, MIT, Copyright (c) Ollama.
//! Ported against `4713800b08b2ddf5e14acf8398953cf7b12f169b` (2026-07-28).
//!
//! ## What this module is for
//!
//! [`crate::modelfile`] stop exactly where this module start. It take your
//! `FROM ./model.gguf` / `PARAMETER` / `TEMPLATE` / `SYSTEM` / `MESSAGE` text and
//! give you back a [`Modelfile`] -- a list of commands, nothing more. From there
//! somebody still got to:
//!
//! 1. work out **which actual files on disk** the `FROM` is pointing at
//!    ([`files_for_model`]) -- got a whole priority order for that, wait ah;
//! 2. **hash** every one of them so they become content addresses
//!    ([`file_digest_map`]);
//! 3. turn the rest of the commands into a [`CreateRequest`]
//!    ([`create_request`]);
//! 4. write out the side-car layers (template, system, params, messages,
//!    license), build the config blob, and drop a manifest
//!    ([`create_model`]).
//!
//! That is this module lah. After [`create_model`] returns, the store really got
//! the model -- [`get_model`] can read it back.
//!
//! ## The seam: this crate cannot read GGUF, and that is on purpose
//!
//! Upstream's `create.go` also convert safetensors -> GGUF, quantise, merge
//! split shards, and sniff a GGUF's KV table for the chat template. All of that
//! need a GGUF decoder. `kopitiam-ollama` **depends on nothing else in
//! KOPITIAM** (see the crate docs) -- that is what keeps this layer testable
//! with no model, no GPU, no network -- so it has no decoder and is not getting
//! one.
//!
//! So anywhere upstream reach into `layer.GGML.KV()`, this port instead takes
//! the handful of facts it actually needed as a plain input struct
//! ([`GgufFacts`], [`ProjectorFacts`], and the `architecture` argument to
//! [`apply_architecture_defaults`]). The **decision logic** is ported and
//! tested; the *reading* is the caller's job (`kopitiam-loader` already owns
//! GGUF). What this buys: the rule "a model with `vision.block_count` gets
//! [`Capability::Vision`]" is verifiable in a unit test with no 20 GB file.
//!
//! **What would make this wrong:** a caller that fills [`GgufFacts`] from
//! something other than the model's own GGUF KV table. The struct is documented
//! field-by-field with the exact GGUF key each one must come from -- follow it.
//!
//! ## Not ported, said plainly
//!
//! * **safetensors -> GGUF conversion, quantisation, split-shard merge**
//!   (`convertFromSafetensors`, `quantizeLayer`, `copyLayerWithLlamaQuantize`,
//!   `mergeSplitGGUFLayers`, `convertMTPDraftFromSafetensors`). All need a GGUF
//!   writer plus `llama-quantize`. [`split_gguf_name`] *is* ported, because it
//!   is pure string work and the filename convention is worth preserving.
//! * **The remote / cloud half** (`remoteURL`, `parseFromModel`'s pull,
//!   `parseAndValidateModelRef`'s `modelref` source suffixes). That is
//!   `registry.rs`, owned elsewhere.
//! * **The HTTP handler** (`CreateHandler`, `streamResponse`) -- `routes.rs`.
//! * **`http.DetectContentType` in full.** Only three answers are ported; see
//!   [`detect_content_type`] for exactly which, and what the gap costs.
//!
//! ## `~user` expansion, and why it is a trait
//!
//! Go's `os/user.Lookup` read the platform account database. Rust std has no
//! equivalent, and there is no portable one: Termux has no real `/etc/passwd`
//! entries for other users, and Windows would need `NetUserGetInfo`. So
//! [`expand_path_with`] take a [`UserLookup`], exactly like upstream's
//! `expandPathImpl` take `currentUserFunc` / `lookupUserFunc` -- upstream
//! parameterised those two for the same reason (its own tests mock them).
//! [`SystemUsers`] is the default: it resolve the *current* user's home from the
//! environment, and refuse any other user by name rather than guess. See
//! [`SystemUsers`] for the full reasoning.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::{Capability, ConfigV2, Message};
use crate::manifest::{
    self, Digest, Layer, Sha256Hasher, Store, MEDIA_TYPE_ADAPTER, MEDIA_TYPE_CONFIG,
    MEDIA_TYPE_DRAFT, MEDIA_TYPE_LICENSE, MEDIA_TYPE_MESSAGES, MEDIA_TYPE_MODEL,
    MEDIA_TYPE_PARAMS, MEDIA_TYPE_PROJECTOR, MEDIA_TYPE_PROMPT, MEDIA_TYPE_SYSTEM,
    MEDIA_TYPE_TEMPLATE,
};
use crate::modelfile::{Modelfile, DEPRECATED_PARAMETERS};
use crate::name::Name;
use crate::template::Template;

// ===========================================================================
// Errors
// ===========================================================================

/// Everything that can go wrong on the `create` path.
///
/// **Upstream:** the sentinel `var` block at the top of `server/create.go`
/// (`errNoFilesProvided`, `errOnlyOneAdapterSupported`, `errOnlyGGUFSupported`,
/// `errUnknownType`, `errNeitherFromOrFiles`, `errFilePath`,
/// `errRemoteDraftUnsupported`), plus `parser.ErrModelNotFound` and the
/// `fmt.Errorf`s scattered through `fileDigestMap` / `filesForModel`.
///
/// Wording is kept close to upstream's on purpose -- a KOPITIAM error should
/// still be searchable against ollama's issue tracker.
#[derive(Debug, thiserror::Error)]
pub enum CreateError {
    /// **Upstream:** `errNoFilesProvided`.
    #[error("no files provided to convert")]
    NoFilesProvided,

    /// **Upstream:** `errOnlyOneAdapterSupported`.
    #[error("only one adapter is currently supported")]
    OnlyOneAdapterSupported,

    /// **Upstream:** `errOnlyGGUFSupported`.
    #[error("supplied file was not in GGUF format")]
    OnlyGgufSupported,

    /// **Upstream:** `errUnknownType`.
    #[error("unknown type")]
    UnknownType,

    /// **Upstream:** `errNeitherFromOrFiles`.
    #[error("neither 'from' or 'files' was specified")]
    NeitherFromOrFiles,

    /// **Upstream:** `errFilePath` -- `"file path must be relative"`.
    #[error("file path must be relative: {0}")]
    FilePath(String),

    /// **Upstream:** `parser.ErrModelNotFound`, raised by `filesForModel` when
    /// no glob in the priority ladder matched anything.
    #[error("no safetensors or torch files found")]
    ModelNotFound,

    /// A globbed file resolved (after following symlinks) to somewhere outside
    /// the directory the user pointed at.
    ///
    /// **Upstream:** the `fmt.Errorf("insecure path: %s")` branch in
    /// `fileDigestMap`.
    #[error("insecure path: {0}")]
    InsecurePath(String),

    /// Same escape, but the offending path mentions `.cache`.
    ///
    /// **Upstream:** the `.cache` arm of the same check, hint and all. It fires
    /// when a HuggingFace download left symlinks pointing back into
    /// `~/.cache/huggingface`, and the advice is to re-download with
    /// `--local-dir`.
    #[error(
        "insecure path: {0}\n\nUse --local-dir <dir> when downloading model to disable caching"
    )]
    InsecureCachePath(String),

    /// **Upstream:** `fmt.Errorf("invalid content type: expected %s for %s")`.
    /// Fires for an unresolved git-lfs pointer file: the pointer is text, the
    /// glob wanted binary.
    #[error("invalid content type: expected {expected} for {path}")]
    InvalidContentType {
        /// What [`detect_content_type`] actually said. (Upstream really does
        /// interpolate the *detected* type after the word "expected" -- kept,
        /// so the string a user searches for matches ollama's.)
        expected: String,
        /// The file it said it about, forward-slashed for display.
        path: String,
    },

    /// **Upstream:** `fmt.Errorf("%s must not reference the same local path as FROM: %s")`.
    #[error("{name} must not reference the same local path as FROM: {path}")]
    SameLocalPath {
        /// The command that clashed -- upstream only ever passes `"DRAFT"`.
        name: String,
        /// The offending path, forward-slashed for display.
        path: String,
    },

    /// **Upstream:** `errBadTemplate`, wrapping `template.Parse`'s own error.
    #[error("template: {0}")]
    BadTemplate(String),

    /// **Upstream:** `fmt.Errorf("requires must be a valid semver (e.g. 0.14.0)")`.
    #[error("requires must be a valid semver (e.g. 0.14.0)")]
    BadRequires,

    /// **Upstream:** `fmt.Errorf("unknown parameter '%s'")` in `api.FormatParams`.
    #[error("unknown parameter '{0}'")]
    UnknownParameter(String),

    /// **Upstream:** `fmt.Errorf("invalid %s value %s")` in `api.FormatParams`.
    #[error("invalid {kind} value {value}")]
    BadParameterValue {
        /// `"int"`, `"float"` or `"bool"`, matching upstream's wording.
        kind: &'static str,
        /// What the Modelfile actually said.
        value: String,
    },

    /// `~user` was asked for and this platform cannot answer. See [`SystemUsers`].
    #[error("failed to find user '{0}': looking up another user's home directory is not supported on this platform")]
    UserLookupUnsupported(String),

    /// **Upstream:** `fmt.Errorf("failed to get current user: %w")`.
    #[error("failed to get current user: {0}")]
    NoCurrentUser(String),

    /// A pattern [`go_match`] cannot handle. See its docs for the ported subset.
    #[error("bad glob pattern: {0}")]
    BadPattern(String),

    /// Something on the filesystem said no.
    #[error("{context}: {source}")]
    Io {
        /// What we were trying to do, forward-slashed.
        context: String,
        #[source]
        source: io::Error,
    },

    /// JSON encode/decode of a layer blob failed.
    #[error("{context}: {source}")]
    Json {
        /// What we were trying to do.
        context: String,
        #[source]
        source: serde_json::Error,
    },

    /// The blob store said no.
    #[error(transparent)]
    Manifest(#[from] manifest::ManifestError),
}

/// `Result` with [`CreateError`] pre-filled.
pub type Result<T, E = CreateError> = std::result::Result<T, E>;

fn io_ctx(context: impl Into<String>) -> impl FnOnce(io::Error) -> CreateError {
    move |source| CreateError::Io {
        context: context.into(),
        source,
    }
}

fn json_ctx(context: impl Into<String>) -> impl FnOnce(serde_json::Error) -> CreateError {
    move |source| CreateError::Json {
        context: context.into(),
        source,
    }
}

/// Forward-slash a path, for **display only**.
///
/// Every path that reaches a user's eyes goes through this, because CLAUDE.md
/// require consistent forward-slash output on Windows and Termux alike. Paths
/// handed to `std::fs` stay native -- upstream emit native separators and so do
/// we, otherwise the Windows test expectations (`D:\home\testuser`) would be
/// wrong.
fn disp(p: impl AsRef<Path>) -> String {
    manifest::to_slash(p.as_ref())
}

// ===========================================================================
// Go's `path/filepath`, ported only as far as needed
// ===========================================================================
//
// Why port instead of leaning on `std::path`: the two disagree in ways that
// change behaviour here.
//
// * `std::path::absolute` does NOT lexically remove `..`; Go's `filepath.Abs`
//   calls `Clean`, which does. The security check in `file_digest_map` decides
//   "did this escape" by looking for a leading `..`, so an uncleaned path with
//   `a/../../b` in it would sail straight through.
// * Go's `*` in a glob never crosses a separator, so `**` is just `*` -- ONE
//   level, not recursive. Reach for a `glob` crate and `path/**/*.json`
//   silently becomes a recursive walk, and `files_for_model` starts hashing the
//   whole tree. That is the single easiest way to get this module wrong.

/// `true` for a byte Go's `filepath` treats as a separator.
///
/// **Upstream:** `os.IsPathSeparator`. Windows accepts BOTH `\` and `/`; Unix
/// only `/`. That asymmetry is exactly why [`expand_path_with`] has to check for
/// a leading `/` even after [`is_abs`] said no.
#[inline]
fn is_sep(b: u8) -> bool {
    if cfg!(windows) {
        b == b'\\' || b == b'/'
    } else {
        b == b'/'
    }
}

/// The separator Go's `filepath` **emits**.
const SEP: char = if cfg!(windows) { '\\' } else { '/' };

/// Length of the volume prefix. **Upstream:** `filepath.VolumeName`.
///
/// Unix: always 0. Windows: `2` for a drive letter (`C:`), or the
/// `\\host\share` span for a UNC path. Anything more exotic (`\\?\`, device
/// paths) is **not** ported -- a Modelfile pointing at `\\?\C:\...` would be
/// treated as a plain rooted path. Said out loud because it is a real gap, not
/// an oversight.
fn volume_len(p: &str) -> usize {
    if !cfg!(windows) {
        return 0;
    }
    let b = p.as_bytes();
    if b.len() >= 2 && b[1] == b':' && b[0].is_ascii_alphabetic() {
        return 2;
    }
    // UNC: \\host\share
    if b.len() >= 5 && is_sep(b[0]) && is_sep(b[1]) && !is_sep(b[2]) {
        let mut i = 3;
        while i < b.len() && !is_sep(b[i]) {
            i += 1;
        }
        if i < b.len() {
            i += 1;
            let start = i;
            while i < b.len() && !is_sep(b[i]) {
                i += 1;
            }
            if i > start {
                return i;
            }
        }
    }
    0
}

/// **Upstream:** `filepath.IsAbs`.
///
/// Note what it says on Windows: `/foo` is **not** absolute (no volume), which
/// is why [`expand_path_with`] needs its extra `\` / `/` prefix checks.
fn is_abs(p: &str) -> bool {
    let v = volume_len(p);
    let rest = &p[v..];
    let rooted = rest.as_bytes().first().is_some_and(|b| is_sep(*b));
    if cfg!(windows) {
        v > 0 && rooted
    } else {
        rooted
    }
}

/// **Upstream:** `filepath.Clean` -- the lexical one, no filesystem access.
///
/// Collapse repeated separators, drop `.`, cancel each inner `..` against the
/// element before it, drop a leading `..` on a rooted path, and emit [`SEP`].
/// Empty in, `"."` out.
///
/// **This is load-bearing for security**, not tidiness: [`file_digest_map`]
/// decides "did this file escape the directory" by looking for a leading `..`
/// in a cleaned relative path. Skip the clean and `a/../../b` reads as local.
fn go_clean(p: &str) -> String {
    let vlen = volume_len(p);
    let (vol, rest) = p.split_at(vlen);
    let rooted = rest.as_bytes().first().is_some_and(|b| is_sep(*b));

    let mut out: Vec<&str> = Vec::new();
    for part in rest.split(|c: char| c.is_ascii() && is_sep(c as u8)) {
        match part {
            "" | "." => {}
            ".." => {
                if out.last().is_some_and(|last| *last != "..") {
                    out.pop();
                } else if !rooted {
                    // Go's rule 4: a `..` at the root of a rooted path is dropped.
                    out.push("..");
                }
            }
            other => out.push(other),
        }
    }

    let mut s = String::with_capacity(p.len() + 1);
    // Go normalises the volume's own separators too (`//host/share` -> `\\host\share`).
    for c in vol.chars() {
        s.push(if c.is_ascii() && is_sep(c as u8) { SEP } else { c });
    }
    if rooted {
        s.push(SEP);
    }
    for (i, part) in out.iter().enumerate() {
        if i > 0 {
            s.push(SEP);
        }
        s.push_str(part);
    }
    if s.is_empty() {
        return ".".to_string();
    }
    s
}

/// **Upstream:** `filepath.Join` -- concatenate the non-empty parts, then
/// [`go_clean`].
fn go_join(parts: &[&str]) -> String {
    let kept: Vec<&str> = parts.iter().copied().filter(|p| !p.is_empty()).collect();
    if kept.is_empty() {
        return String::new();
    }
    go_clean(&kept.join(&SEP.to_string()))
}

/// **Upstream:** `filepath.Abs` -- [`go_clean`] if already absolute, otherwise
/// clean of `cwd` joined with the path.
fn go_abs(p: &str) -> Result<String> {
    if is_abs(p) {
        return Ok(go_clean(p));
    }
    let cwd = std::env::current_dir().map_err(io_ctx("get current directory"))?;
    Ok(go_join(&[&cwd.to_string_lossy(), p]))
}

/// **Upstream:** `filepath.Rel(base, target)`.
///
/// Both are cleaned first. Returns `None` when there is no relative route at
/// all -- different volumes, or one rooted and the other not. Upstream returns
/// an error there; `None` says the same thing with less ceremony, and the one
/// call site treats it as "reject".
fn go_rel(base: &str, target: &str) -> Option<String> {
    let base = go_clean(base);
    let target = go_clean(target);
    if paths_eq(&base, &target) {
        return Some(".".to_string());
    }

    let bv = volume_len(&base);
    let tv = volume_len(&target);
    // Windows volumes are case-insensitive; Go compares them that way too.
    if !base[..bv].eq_ignore_ascii_case(&target[..tv]) {
        return None;
    }
    let (b, t) = (&base[bv..], &target[tv..]);
    let b_rooted = b.as_bytes().first().is_some_and(|x| is_sep(*x));
    let t_rooted = t.as_bytes().first().is_some_and(|x| is_sep(*x));
    if b_rooted != t_rooted {
        return None;
    }

    let split = |s: &str| -> Vec<String> {
        s.split(|c: char| c.is_ascii() && is_sep(c as u8))
            .filter(|x| !x.is_empty() && *x != ".")
            .map(str::to_string)
            .collect()
    };
    let bs = split(b);
    let ts = split(t);

    let mut i = 0;
    while i < bs.len() && i < ts.len() && paths_eq(&bs[i], &ts[i]) {
        i += 1;
    }
    // Can't climb out past a `..` whose real name we don't know.
    if bs[i..].iter().any(|x| x == "..") {
        return None;
    }

    let mut parts: Vec<&str> = vec![".."; bs.len() - i];
    parts.extend(ts[i..].iter().map(String::as_str));
    if parts.is_empty() {
        return Some(".".to_string());
    }
    Some(parts.join(&SEP.to_string()))
}

/// Compare two path elements the way the running platform's filesystem would.
///
/// Windows is case-insensitive, Unix is not. Get this backwards on Windows and
/// [`go_rel`] returns a `..`-laden path for a base and target that are the same
/// directory, [`is_local`] then rejects it, and a perfectly good model directory
/// fails with "insecure path".
fn paths_eq(a: &str, b: &str) -> bool {
    if cfg!(windows) {
        a.eq_ignore_ascii_case(b)
    } else {
        a == b
    }
}

/// **Upstream:** `filepath.IsLocal` (Go 1.20+).
///
/// "Local" means: relative, non-empty, and unable to walk out of the directory
/// it is relative to. This is the actual gate in [`file_digest_map`], so the
/// rule, precisely:
///
/// * absolute, or carrying a volume -> **not** local;
/// * empty -> **not** local;
/// * after [`go_clean`], a leading `..` -> **not** local (that is the escape);
/// * on Windows, a reserved device name (`CON`, `NUL`, `COM1`, ...) -> **not**
///   local, because opening it would talk to a device instead of a file.
///
/// **What would make this wrong:** testing the *uncleaned* path. `a/../../b`
/// has no leading `..` until you clean it.
fn is_local(p: &str) -> bool {
    if p.is_empty() || is_abs(p) || volume_len(p) > 0 {
        return false;
    }
    if cfg!(windows) && p.as_bytes().first().is_some_and(|b| is_sep(*b)) {
        // Rooted-but-volumeless (`\foo`) is not "absolute" on Windows, but it
        // is certainly not local either.
        return false;
    }
    let cleaned = go_clean(p);
    if cleaned == ".."
        || (cleaned.len() > 2 && cleaned.starts_with("..") && is_sep(cleaned.as_bytes()[2]))
    {
        return false;
    }
    if cfg!(windows) {
        for part in cleaned.split(|c: char| c.is_ascii() && is_sep(c as u8)) {
            if is_windows_reserved_name(part) {
                return false;
            }
        }
    }
    true
}

/// **Upstream:** `filepath.isReservedName` on Windows.
///
/// The DOS device names, still special-cased by every Windows API forty years
/// on. A trailing extension does not save you: `NUL.txt` is still `NUL`.
fn is_windows_reserved_name(part: &str) -> bool {
    let stem = part.split('.').next().unwrap_or(part);
    let up = stem.to_ascii_uppercase();
    matches!(up.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (up.len() == 4
            && (up.starts_with("COM") || up.starts_with("LPT"))
            && up.as_bytes()[3].is_ascii_digit()
            && up.as_bytes()[3] != b'0')
}

/// Does this pattern contain glob metacharacters? **Upstream:** `hasMeta`.
fn has_meta(p: &str) -> bool {
    p.contains(['*', '?', '[']) || (!cfg!(windows) && p.contains('\\'))
}

/// **Upstream:** `filepath.Match`, applied to a single directory entry name.
///
/// ## The one thing to remember
///
/// **`*` never crosses a path separator.** So `**` is not "recursive" -- it is
/// two stars in a row, which matches exactly what one star would. That is why
/// upstream's `path/**/*.json` reaches precisely **one** level of
/// subdirectories, and why the sentence-transformers case in upstream's
/// `TestFilesForModel` expects `2_Dense/config.json` but nothing deeper.
///
/// ## Ported subset, stated plainly
///
/// `*`, `?` and literal characters. **Not** ported: `[...]` character classes
/// and `\` escaping. No pattern in [`files_for_model`] -- the only caller --
/// uses either, and a half-done character class is worse than an honest error,
/// so [`go_glob`] returns [`CreateError::BadPattern`] the moment it sees a `[`.
fn go_match(pattern: &str, name: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let n: Vec<char> = name.chars().collect();
    // Two-pointer backtracking wildcard match. `star`/`mark` remember where to
    // resume the last `*` if the tail turns out not to fit.
    let (mut pi, mut ni) = (0usize, 0usize);
    let (mut star, mut mark) = (usize::MAX, 0usize);
    while ni < n.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == n[ni]) {
            pi += 1;
            ni += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = pi;
            mark = ni;
            pi += 1;
        } else if star != usize::MAX {
            pi = star + 1;
            mark += 1;
            ni = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

/// **Upstream:** `filepath.Glob`.
///
/// Split off the last segment, glob the directory part (recursively, if it too
/// has metacharacters), then read each resulting directory and match the last
/// segment against every entry name.
///
/// Two upstream behaviours worth naming, both load-bearing:
///
/// * **A directory that does not exist is not an error, it is zero matches.**
///   Upstream swallows `readDir`'s error. [`files_for_model`] leans on that --
///   it tries seven different patterns and expects the misses to be silent.
/// * **Results are sorted by name within each directory** (Go's `readDir`
///   sorts). Kept, so the order of layers in a manifest is reproducible rather
///   than whatever order the filesystem happened to hand back.
fn go_glob(pattern: &str) -> Result<Vec<String>> {
    if pattern.contains('[') {
        return Err(CreateError::BadPattern(pattern.to_string()));
    }
    if !has_meta(pattern) {
        return Ok(if Path::new(pattern).symlink_metadata().is_ok() {
            vec![pattern.to_string()]
        } else {
            Vec::new()
        });
    }

    // Go's `filepath.Split`: cut at the last separator.
    let idx = pattern
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_ascii() && is_sep(*c as u8))
        .map(|(i, _)| i);
    let (dir, file) = match idx {
        Some(i) => (&pattern[..i], &pattern[i + 1..]),
        None => ("", pattern),
    };

    let vlen = volume_len(dir).min(dir.len());
    let dirs: Vec<String> = if dir.is_empty() {
        vec![".".to_string()]
    } else if has_meta(&dir[vlen..]) {
        // Recurse. Upstream guards against `dir == pattern` looping forever;
        // our split always shortens the pattern, so the guard is structural.
        go_glob(dir)?
    } else {
        vec![go_clean(dir)]
    };

    let mut out = Vec::new();
    for d in dirs {
        let Ok(entries) = fs::read_dir(&d) else {
            continue; // missing, or not a directory: zero matches, not an error
        };
        let mut names: Vec<String> = entries
            .filter_map(std::result::Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        for n in names {
            if go_match(file, &n) {
                out.push(go_join(&[&d, &n]));
            }
        }
    }
    Ok(out)
}

/// Resolve every symlink on a path. **Upstream:** `filepath.EvalSymlinks`.
///
/// `std::fs::canonicalize` is the equivalent, with one Windows wart worth
/// knowing: it returns an **extended-length** `\\?\C:\...` path. Left as-is,
/// that prefix makes [`go_rel`] compare `\\?\C:\tmp\x` against `C:\tmp`,
/// conclude "different volumes", and every model directory on Windows would
/// then fail the insecure-path check. So we strip it. This is exactly the kind
/// of platform detail that only shows up as a mystifying test failure, so it
/// lives here where the stripping happens.
fn eval_symlinks(p: &Path) -> Result<String> {
    let c = fs::canonicalize(p).map_err(io_ctx(format!("resolve {}", disp(p))))?;
    let s = c.to_string_lossy().into_owned();
    if cfg!(windows) {
        if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
            return Ok(format!(r"\\{rest}"));
        }
        if let Some(rest) = s.strip_prefix(r"\\?\") {
            return Ok(rest.to_string());
        }
    }
    Ok(s)
}

// ===========================================================================
// Content sniffing
// ===========================================================================

/// How many bytes get sniffed. **Upstream:** `net/http`'s `sniffLen = 512`, and
/// `filesForModel` copies exactly `io.CopyN(&b, f, 512)` before calling it.
///
/// 512 is the WHATWG mime-sniffing standard's "resource header" size, not an
/// arbitrary buffer -- every signature in the table is defined to fit inside it.
pub const SNIFF_LEN: usize = 512;

/// A **deliberately partial** port of Go's `http.DetectContentType`.
///
/// **Upstream:** `net/http/sniff.go`. Only the three answers `filesForModel`
/// actually compares against are ported:
///
/// | Returned | When | Upstream signature |
/// |---|---|---|
/// | `"application/zip"` | starts with `PK\x03\x04` | the zip `exactSig` |
/// | `"text/plain; charset=utf-8"` | empty, or no binary control byte in the window | `textSig` |
/// | `"application/octet-stream"` | anything else | the fallback |
///
/// "Binary control byte" is upstream's `textSig.match` rule exactly: any byte
/// `<= 0x08`, `0x0B`, `0x0E..=0x1A`, or `0x1C..=0x1F`, scanned from the first
/// non-whitespace byte.
///
/// ## What the missing signatures cost, precisely
///
/// Upstream's table also recognises HTML, XML, PDF, PostScript, UTF-16 BOMs,
/// images, audio, video, gzip, rar, wasm and fonts. Each would make upstream
/// return something *specific* where we return one of our three. In
/// [`files_for_model`] every check is an equality test against an expected type,
/// so a missing signature can only ever turn an upstream **rejection** into our
/// **acceptance** -- e.g. a PNG named `model.gguf` is `image/png` upstream
/// (rejected) and `application/octet-stream` here (accepted). It can never do
/// the reverse, which is the safe direction. That is why the subset is
/// acceptable -- but it *is* a real difference, and this is it written down.
///
/// One case that genuinely differs in a way you might meet: a UTF-16 text file.
/// Upstream says `text/plain; charset=utf-16be`; we see the interleaved `0x00`
/// bytes and say `application/octet-stream`. Config files in a model repo are
/// UTF-8 in practice, so it has not bitten -- but it would.
pub fn detect_content_type(data: &[u8]) -> &'static str {
    let data = &data[..data.len().min(SNIFF_LEN)];
    if data.is_empty() {
        // Upstream returns text/plain for an empty body, not octet-stream.
        return "text/plain; charset=utf-8";
    }
    if data.starts_with(b"PK\x03\x04") {
        return "application/zip";
    }
    // `firstNonWS`: upstream skips leading whitespace before running textSig.
    let first_non_ws = data
        .iter()
        .position(|b| !matches!(b, b'\t' | b'\n' | 0x0C | b'\r' | b' '))
        .unwrap_or(data.len());
    for &b in &data[first_non_ws..] {
        if b <= 0x08 || b == 0x0B || (0x0E..=0x1A).contains(&b) || (0x1C..=0x1F).contains(&b) {
            return "application/octet-stream";
        }
    }
    "text/plain; charset=utf-8"
}

/// The media type without its parameters. **Upstream:**
/// `strings.Cut(http.DetectContentType(...), ";")` inside `filesForModel`.
fn detect_content_type_bare(data: &[u8]) -> &'static str {
    match detect_content_type(data) {
        "text/plain; charset=utf-8" => "text/plain",
        other => other,
    }
}

/// Which GGML-family container is this, judged by its 4-byte magic?
///
/// **Upstream:** `fs/ggml.DetectContentType` and the `FILE_MAGIC_*` constants
/// beside it, all compared against `binary.LittleEndian.Uint32(b[:4])`.
///
/// ## The byte order trap, worth knowing before you eyeball a hexdump
///
/// The **old ggml-family magics are byte-reversed on disk, the GGUF one is not.**
/// That is not a mistake in either this code or upstream's -- it is a real
/// artefact of how the formats were written, and it will mislead anyone who
/// assumes the file "spells" its own magic.
///
/// The old formats were produced by `fwrite`-ing the `uint32` constant on a
/// little-endian machine, so the constant's bytes land reversed:
///
/// | Constant | Reads on disk as | Meaning |
/// |---|---|---|
/// | `0x67676d6c` | `l m g g` | `"ggml"` |
/// | `0x67676d66` | `f m g g` | `"ggmf"` |
/// | `0x67676a74` | `t j g g` | `"ggjt"` |
/// | `0x67676c61` | `a l g g` | `"ggla"` |
/// | `0x46554747` | `G G U F` | `"gguf"`, little-endian writer |
/// | `0x47475546` | `F U G G` | `"gguf"`, big-endian writer |
///
/// GGUF deliberately fixed this: its constant is chosen so that a
/// little-endian write produces the literal ASCII `GGUF`, which is why a GGUF
/// file really does start with the four characters you would expect and a
/// `.ggml` file really does start with `lmgg`.
///
/// **What would make this wrong:** "correcting" the constants to match the
/// ASCII, or switching to `from_be_bytes`. Either would stop recognising real
/// files, and the failure would look like "this GGUF is corrupt".
///
/// `None` for anything else, matching upstream's empty string.
pub fn detect_ggml_content_type(b: &[u8]) -> Option<&'static str> {
    if b.len() < 4 {
        return None;
    }
    let magic = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    Some(match magic {
        0x6767_6d6c => "ggml",
        0x6767_6d66 => "ggmf",
        0x6767_6a74 => "ggjt",
        0x6767_6c61 => "ggla",
        0x4655_4747 | 0x4747_5546 => "gguf",
        _ => return None,
    })
}

/// **Upstream:** `server/model.go`'s `detectContentType(r)` -- ask ggml first,
/// then `http.DetectContentType`, then give up with `"unknown"`.
///
/// The ordering matters: a GGUF file is binary, so the HTTP sniffer would only
/// ever call it `application/octet-stream`. Asking ggml first is what lets
/// `ggufLayers` reject a non-GGUF blob with a message that means something.
pub fn detect_blob_content_type(data: &[u8]) -> &'static str {
    if let Some(ct) = detect_ggml_content_type(data) {
        return ct;
    }
    match detect_content_type(data) {
        "application/octet-stream" => "unknown",
        other => other,
    }
}

// ===========================================================================
// expandPath
// ===========================================================================

/// Where `~` and `~user` come from.
///
/// **Upstream:** the `currentUserFunc` / `lookupUserFunc` parameters of
/// `expandPathImpl`. Upstream made these injectable so its own tests could mock
/// them; the same seam here also solves a portability problem Go did not have
/// (see [`SystemUsers`]).
pub trait UserLookup {
    /// Home directory of whoever is running us. **Upstream:** `user.Current`.
    fn current_home(&self) -> Result<PathBuf>;

    /// Home directory of a named user. **Upstream:** `user.Lookup`.
    fn home_of(&self, username: &str) -> Result<PathBuf>;
}

/// The default [`UserLookup`]: the environment for the current user, and an
/// honest refusal for anybody else.
///
/// ## Why `~otheruser` is refused rather than guessed
///
/// Go's `os/user.Lookup` reads the platform account database -- `getpwnam` on
/// Unix, `NetUserGetInfo` on Windows. Rust std has no equivalent and there is no
/// portable substitute:
///
/// * **Termux** (a target this project actually ships to) runs as a single
///   Android app UID. There is no `/etc/passwd` listing other users, so there is
///   genuinely no answer to give.
/// * **Windows** would need a Win32 call, i.e. a new dependency, for a Modelfile
///   syntax nobody has been observed using.
/// * **Guessing** -- "take the parent of my home and join the username" -- is the
///   tempting one, and it is wrong. `/home/bob` is a convention, not a rule:
///   macOS uses `/Users`, some sites use `/export/home`, and a guessed path that
///   happens to exist would silently hash **the wrong directory**. Quietly
///   building a model from someone else's files is a worse outcome than an error.
///
/// So: [`SystemUsers::home_of`] answers for the current user (matched against
/// `$USER` / `$LOGNAME` / `%USERNAME%`, since `~me` should behave like `~`), and
/// returns [`CreateError::UserLookupUnsupported`] otherwise. A host that *can*
/// answer -- a CLI that already links a passwd reader -- implements
/// [`UserLookup`] and passes it to [`expand_path_with`]. Nothing invented, door
/// left open.
///
/// [`SystemUsers::current_home`] reads `$HOME`, then `%USERPROFILE%`, then
/// `%HOMEDRIVE%` + `%HOMEPATH%`. That order is why Termux (which sets only
/// `$HOME`) works unchanged, and why a Windows session without `$HOME` still
/// resolves.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemUsers;

impl UserLookup for SystemUsers {
    fn current_home(&self) -> Result<PathBuf> {
        for key in ["HOME", "USERPROFILE"] {
            if let Some(h) = std::env::var_os(key).filter(|h| !h.is_empty()) {
                return Ok(PathBuf::from(h));
            }
        }
        if let (Some(d), Some(p)) = (
            std::env::var_os("HOMEDRIVE").filter(|h| !h.is_empty()),
            std::env::var_os("HOMEPATH").filter(|h| !h.is_empty()),
        ) {
            let mut s = d.to_string_lossy().into_owned();
            s.push_str(&p.to_string_lossy());
            return Ok(PathBuf::from(s));
        }
        Err(CreateError::NoCurrentUser(
            "neither HOME nor USERPROFILE is set".to_string(),
        ))
    }

    fn home_of(&self, username: &str) -> Result<PathBuf> {
        let me = std::env::var("USER")
            .or_else(|_| std::env::var("LOGNAME"))
            .or_else(|_| std::env::var("USERNAME"))
            .unwrap_or_default();
        if !me.is_empty() && me == username {
            return self.current_home();
        }
        Err(CreateError::UserLookupUnsupported(username.to_string()))
    }
}

/// Resolve a Modelfile path argument to an absolute path.
///
/// **Upstream:** `parser.expandPath(path, relativeDir)`.
///
/// Four cases, in upstream's order:
///
/// 1. **Absolute** -- or, on Windows, merely rooted (`\foo` and `/foo` count) --
///    made absolute as-is. `relative_dir` is ignored, deliberately.
/// 2. **`~` or `~/...`** -> the current user's home.
/// 3. **`~name/...`** -> that user's home. See [`SystemUsers`] for what this
///    platform can and cannot answer.
/// 4. **Anything else** -> joined onto `relative_dir` (the directory the
///    Modelfile itself lives in), then made absolute.
///
/// Every branch ends in [`go_abs`], which cleans -- so the returned path never
/// contains a `.` or an un-cancelled `..`. That is the security-relevant part:
/// a `FROM ../../../../etc` cannot survive as a `..` chain, it resolves to a
/// concrete absolute path that the caller can then check.
pub fn expand_path(path: &str, relative_dir: &str) -> Result<PathBuf> {
    expand_path_with(path, relative_dir, &SystemUsers)
}

/// [`expand_path`] with the user database injected.
///
/// **Upstream:** `parser.expandPathImpl`.
pub fn expand_path_with(path: &str, relative_dir: &str, users: &dyn UserLookup) -> Result<PathBuf> {
    // Upstream: `filepath.IsAbs(path) || strings.HasPrefix(path, "\\") ||
    // strings.HasPrefix(path, "/")`. The two prefix tests exist for Windows,
    // where a volumeless rooted path is not "absolute" but must still not be
    // joined onto relativeDir.
    if is_abs(path) || path.starts_with('\\') || path.starts_with('/') {
        return Ok(PathBuf::from(go_abs(path)?));
    }

    let resolved = if let Some(after_tilde) = path.strip_prefix('~') {
        let (home, tail) = if after_tilde.is_empty() || after_tilde.starts_with('/') {
            (users.current_home()?, after_tilde.to_string())
        } else {
            // Upstream: `strings.SplitN(path[1:], "/", 2)`. Note it splits on
            // `/` ONLY -- even on Windows -- so `~user\docs` parses as one
            // username called `user\docs`, which then fails to look up.
            // Faithful wart, kept: ollama is the oracle.
            let mut parts = after_tilde.splitn(2, '/');
            let user = parts.next().unwrap_or_default();
            let rest = parts.next();
            let home = users.home_of(user)?;
            (home, rest.map(|r| format!("/{r}")).unwrap_or_default())
        };
        go_join(&[&home.to_string_lossy(), &tail])
    } else {
        go_join(&[relative_dir, path])
    };

    Ok(PathBuf::from(go_abs(&resolved)?))
}

// ===========================================================================
// filesForModel
// ===========================================================================

/// Sniff one file's content type, reading at most [`SNIFF_LEN`] bytes.
///
/// **Upstream:** the `detectContentType` closure inside `filesForModel`, which
/// does `io.CopyN(&b, f, 512)` and tolerates `io.EOF` -- a file shorter than 512
/// bytes is fine, you just sniff what is there.
fn sniff_file(path: &str) -> Result<&'static str> {
    use std::io::Read as _;
    let mut f = fs::File::open(path).map_err(io_ctx(format!("open {}", disp(path))))?;
    let mut buf = vec![0u8; SNIFF_LEN];
    let mut filled = 0usize;
    while filled < SNIFF_LEN {
        match f.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(io_ctx(format!("read {}", disp(path)))(e)),
        }
    }
    buf.truncate(filled);
    Ok(detect_content_type_bare(&buf))
}

/// Glob, then content-check every match.
///
/// **Upstream:** the `glob(pattern, contentType)` closure inside
/// `filesForModel`. An empty `content_type` means "don't check" -- upstream uses
/// that for safetensors, with the comment *"some safetensors files do not
/// properly match application/octet-stream"*.
fn glob_checked(pattern: &str, content_type: &str) -> Result<Vec<String>> {
    let matches = go_glob(pattern)?;
    for m in &matches {
        let ct = sniff_file(m)?;
        if !content_type.is_empty() && ct != content_type {
            return Err(CreateError::InvalidContentType {
                expected: ct.to_string(),
                path: disp(m),
            });
        }
    }
    Ok(matches)
}

/// Upstream's `if x, _ := glob(...); len(x) > 0` idiom: a failed probe and an
/// empty probe are the same thing, and the ladder moves on to the next rung.
fn nonempty(r: Result<Vec<String>>) -> Option<Vec<String>> {
    match r {
        Ok(v) if !v.is_empty() => Some(v),
        _ => None,
    }
}

/// Work out which files in `path` actually are the model.
///
/// **Upstream:** `parser.filesForModel(path)`.
///
/// ## The priority ladder, and why it must not be reordered
///
/// A HuggingFace snapshot commonly ships the **same weights twice** -- sharded
/// `model-00001-of-00002.safetensors` *and* a `consolidated.safetensors`, or
/// safetensors *and* the older `pytorch_model.bin`. Taking both would double the
/// disk and the conversion work; taking the wrong one converts from the format
/// upstream considers secondary. So this is strict first-match-wins, not a
/// union:
///
/// 1. `model*.safetensors` -- covers `model.safetensors`,
///    `model-x-of-y.safetensors`, `model.fp32-x-of-y.safetensors`. **If this
///    rung hits, also take `*/model*.safetensors`** (one level down): that is
///    how sentence-transformers ships its `2_Dense/` module weights.
/// 2. `consolidated*.safetensors`
/// 3. `pytorch_model*.bin` -- must sniff as `application/zip`
/// 4. `consolidated*.pth` -- must sniff as `application/zip`
/// 5. `*.gguf` -- must sniff as `application/octet-stream`
/// 6. `*.bin` -- must sniff as `application/octet-stream` (a GGUF named `.bin`,
///    from before the extension settled)
/// 7. nothing matched -> [`CreateError::ModelNotFound`]
///
/// Then, **unconditionally**, the config files go on top: `*.json` and
/// `*/*.json` (both as `text/plain`), `chat_template.jinja`, and
/// `tokenizer.model` -- tried first as `application/octet-stream` at the top
/// level, then as `text/plain` one level down, because Llama-3-style repos put
/// it in a subdirectory.
///
/// ## The git-lfs trap this is really guarding
///
/// The content-type checks are not paranoia. `git clone` of an LFS repo without
/// `git lfs pull` leaves **text pointer files** where the weights should be:
///
/// ```text
/// version https://git-lfs.github.com/spec/v1
/// oid sha256:4d4f...
/// size 4831838448
/// ```
///
/// That sniffs as `text/plain`, so demanding `application/zip` /
/// `application/octet-stream` turns "silently built a model out of 130-byte
/// stubs" into a loud error. Rungs 1 and 2 deliberately *skip* the check
/// (upstream's own comment says some real safetensors do not sniff as
/// octet-stream), so an LFS-stubbed safetensors repo gets caught later, when
/// conversion fails on a 130-byte tensor file.
///
/// **What would make this wrong:** reordering the ladder, unioning instead of
/// first-match-wins, or reading `**` as recursive. See [`go_match`] for that
/// last one -- it is the easy mistake.
pub fn files_for_model(path: &str) -> Result<Vec<String>> {
    let j = |parts: &[&str]| -> String { go_join(parts) };
    let mut files: Vec<String> = Vec::new();

    // Upstream discards the error from each probe (`st, _ := glob(...)`), so a
    // content-type failure on one rung does not abort -- the ladder just moves
    // on and, if nothing else matches, `ModelNotFound` is what the user sees.
    // Upstream's own `TestFilesForModel/"invalid content type for pytorch
    // model"` only asserts *some* error, which is why this is safe to keep.
    let safetensors = glob_checked(&j(&[path, "model*.safetensors"]), "").unwrap_or_default();
    if !safetensors.is_empty() {
        files.extend(safetensors);
        // Nested module weights (sentence-transformers). Note upstream does NOT
        // swallow the error here -- this one propagates.
        files.extend(glob_checked(&j(&[path, "*", "model*.safetensors"]), "")?);
    } else if let Some(hit) = nonempty(glob_checked(&j(&[path, "consolidated*.safetensors"]), "")) {
        files.extend(hit);
    } else if let Some(hit) = nonempty(glob_checked(
        &j(&[path, "pytorch_model*.bin"]),
        "application/zip",
    )) {
        files.extend(hit);
    } else if let Some(hit) = nonempty(glob_checked(
        &j(&[path, "consolidated*.pth"]),
        "application/zip",
    )) {
        files.extend(hit);
    } else if let Some(hit) = nonempty(glob_checked(&j(&[path, "*.gguf"]), "application/octet-stream"))
    {
        files.extend(hit);
    } else if let Some(hit) = nonempty(glob_checked(&j(&[path, "*.bin"]), "application/octet-stream"))
    {
        files.extend(hit);
    } else {
        return Err(CreateError::ModelNotFound);
    }

    // Configuration files. JSON sniffs as text/plain.
    files.extend(glob_checked(&j(&[path, "*.json"]), "text/plain")?);
    // Upstream writes `**/*.json` here, and Go's Match makes that ONE level
    // deep, not recursive -- see `go_match`. bert models need the nested
    // config.json; upstream's own TODO says "merge this with the glob above".
    files.extend(glob_checked(&j(&[path, "**", "*.json"]), "text/plain")?);

    // Transformers stores a tokenizer's default template in this standalone file
    // when it is not embedded in tokenizer_config.json.
    files.extend(glob_checked(&j(&[path, "chat_template.jinja"]), "text/plain")?);

    // tokenizer.model may be an unresolved git-lfs pointer; the octet-stream
    // requirement is what catches that. tokenizer.json is already covered by the
    // *.json glob above.
    if let Some(hit) = nonempty(glob_checked(
        &j(&[path, "tokenizer.model"]),
        "application/octet-stream",
    )) {
        files.extend(hit);
    } else if let Some(hit) =
        nonempty(glob_checked(&j(&[path, "**", "tokenizer.model"]), "text/plain"))
    {
        // Sometimes it lives one level down (e.g. meta-llama/Meta-Llama-3-8B).
        files.extend(hit);
    }

    Ok(files)
}

// ===========================================================================
// fileDigestMap
// ===========================================================================

/// SHA-256 one file, following symlinks first.
///
/// **Upstream:** `parser.digestForFile`, which returns `"sha256:%x"` --
/// [`Digest::as_str`] is byte-for-byte that.
///
/// Resolving symlinks first is not cosmetic: it is what makes two Modelfiles
/// that reach the same weights by different routes produce the same digest, and
/// therefore share one blob instead of storing it twice.
pub fn digest_for_file(filename: &str, hasher: &mut dyn Sha256Hasher) -> Result<Digest> {
    let resolved = eval_symlinks(Path::new(filename))?;
    let f = fs::File::open(&resolved).map_err(io_ctx(format!("open {}", disp(&resolved))))?;
    let (digest, _) =
        manifest::sha256_of_reader(f, hasher).map_err(io_ctx(format!("hash {}", disp(&resolved))))?;
    Ok(digest)
}

/// Map every file of a model to its content address.
///
/// **Upstream:** `parser.fileDigestMap(path)`.
///
/// `path` may be a **single file** (hash it, one entry) or a **directory** (run
/// [`files_for_model`] over it, hash each hit). Keys are the resolved absolute
/// paths, exactly as upstream -- the caller re-keys them to relative names
/// before they go on the wire, which is why `CreateHandler` can then insist
/// every key is a valid relative path.
///
/// ## The security check
///
/// For the directory case only: every globbed file is symlink-resolved and then
/// made relative to `path`. If the result is not [`is_local`] -- i.e. it escaped
/// -- this fails. The `.cache` special case is upstream's, and it exists for a
/// specific real-world shape: `huggingface-cli download` without `--local-dir`
/// fills the target directory with **symlinks into `~/.cache/huggingface/hub/`**.
/// Following those would hash files outside the directory the user pointed at,
/// so upstream refuses and names the exact flag that avoids it.
///
/// **Deliberate divergence:** upstream hashes in parallel via `errgroup` with
/// `GOMAXPROCS-1` workers. We hash sequentially. Reason: this crate takes no
/// threadpool dependency for it, and the work is dominated by disk reads rather
/// than SHA-256 -- on one NVMe device the parallel version mostly queues. A
/// caller who needs it parallel owns the loop; [`digest_for_file`] is public for
/// exactly that.
pub fn file_digest_map(
    path: &str,
    hasher: &mut dyn Sha256Hasher,
) -> Result<BTreeMap<String, Digest>> {
    let meta = fs::metadata(path).map_err(io_ctx(format!("stat {}", disp(path))))?;

    let files: Vec<String> = if meta.is_dir() {
        // Compare against the symlink-resolved directory, not the spelling the
        // user typed -- otherwise a `/tmp` that is itself a symlink (macOS) makes
        // every file look like it escaped.
        let base = eval_symlinks(Path::new(path))?;
        let mut out = Vec::new();
        for f in files_for_model(path)? {
            let resolved = eval_symlinks(Path::new(&f))?;
            let Some(rel) = go_rel(&base, &resolved) else {
                return Err(CreateError::InsecurePath(disp(&resolved)));
            };
            if !is_local(&rel) {
                // Upstream checks `strings.Contains(rel, ".cache")`.
                if rel.contains(".cache") {
                    return Err(CreateError::InsecureCachePath(disp(&rel)));
                }
                return Err(CreateError::InsecurePath(disp(&rel)));
            }
            out.push(resolved);
        }
        out
    } else {
        vec![path.to_string()]
    };

    let mut fl = BTreeMap::new();
    for f in files {
        let digest = digest_for_file(&f, hasher)?;
        fl.insert(f, digest);
    }
    Ok(fl)
}

/// Refuse a `DRAFT` that points at the same place as a `FROM`.
///
/// **Upstream:** `parser.rejectMatchingLocalPath(name, path, existing)`.
///
/// Comparison is by [`canonical_local_path`] -- absolute **and** symlink-
/// resolved -- so `./model.gguf`, `/tmp/x/model.gguf` and a symlink to either
/// all compare equal. Comparing raw strings would let a Modelfile smuggle the
/// same file in twice under two spellings, and the base and draft model would
/// then be the same weights, which defeats the entire point of a draft model.
pub fn reject_matching_local_path(name: &str, path: &str, existing: &[String]) -> Result<()> {
    for candidate in existing {
        if same_local_path(path, candidate)? {
            return Err(CreateError::SameLocalPath {
                name: name.to_string(),
                path: disp(path),
            });
        }
    }
    Ok(())
}

/// **Upstream:** `parser.sameLocalPath`.
fn same_local_path(a: &str, b: &str) -> Result<bool> {
    Ok(paths_eq(&canonical_local_path(a)?, &canonical_local_path(b)?))
}

/// **Upstream:** `parser.canonicalLocalPath` -- `filepath.Abs` then
/// `filepath.EvalSymlinks`.
fn canonical_local_path(path: &str) -> Result<String> {
    let abs = go_abs(path)?;
    eval_symlinks(Path::new(&abs))
}

// ===========================================================================
// CreateRequest
// ===========================================================================

/// A `LICENSE` value: upstream types this `any`, and it really is either shape.
///
/// **Upstream:** `api.CreateRequest.License any`, unpicked by the type switch in
/// `createModel`. A hand-written JSON body may send a bare string; a request
/// built from a Modelfile always sends a list, because a Modelfile may carry
/// several `LICENSE` lines.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LicenseSpec {
    /// One licence, as a plain string.
    One(String),
    /// Several licences. Each becomes its own layer.
    Many(Vec<String>),
}

impl LicenseSpec {
    /// Flatten to the list [`create_model`] actually writes out.
    ///
    /// **Upstream:** the `case string` / `case any` arms of `createModel`'s type
    /// switch. Upstream skips an empty single string (`if l != ""`) but does
    /// **not** filter empties out of a list -- faithful here, wart included.
    fn as_list(&self) -> Vec<&str> {
        match self {
            LicenseSpec::One(s) if s.is_empty() => Vec::new(),
            LicenseSpec::One(s) => vec![s.as_str()],
            LicenseSpec::Many(v) => v.iter().map(String::as_str).collect(),
        }
    }
}

/// Everything `ollama create` needs, once a Modelfile has been resolved against
/// the filesystem.
///
/// **Upstream:** `api.CreateRequest` in `api/types.go`.
///
/// The `files` / `draft_files` / `adapters` maps are **path -> digest**: by the
/// time a request exists, every local file has already been hashed by
/// [`file_digest_map`], so the server never has to trust a path -- it looks the
/// bytes up by content address. That is the whole reason this type sits between
/// the Modelfile and the store.
///
/// **Not ported:** `stream`, `remote_host`, `info`, and the deprecated `name` /
/// `quantization` aliases. `stream` and `remote_host` belong to the HTTP handler
/// and the registry client (both owned elsewhere); `info` is a `map[string]any`
/// override channel that upstream itself notes is "not currently exposed by
/// Modelfiles". `quantize` / `draft_quantize` **are** kept as fields, because
/// they are part of the request's shape -- but the quantisation itself is not
/// ported (see the module docs).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CreateRequest {
    /// The model name being created.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,

    /// Target quantisation, e.g. `"Q4_K_M"`. Empty = leave it alone.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub quantize: String,

    /// Target quantisation for the draft model.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub draft_quantize: String,

    /// A parent **model name** to build on, when `FROM` did not name a local
    /// path. Mutually exclusive with `files` in practice.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub from: String,

    /// Path -> digest for the base model's files.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub files: BTreeMap<String, String>,

    /// Path -> digest for the speculative-decoding draft model's files.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub draft_files: BTreeMap<String, String>,

    /// Path -> digest for LoRA adapters.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub adapters: BTreeMap<String, String>,

    /// The Go prompt template, verbatim from `TEMPLATE`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub template: String,

    /// One or many licences. See [`LicenseSpec`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<LicenseSpec>,

    /// The `SYSTEM` prompt.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub system: String,

    /// `PARAMETER` lines, already coerced to real JSON types by
    /// [`format_params`]. A `BTreeMap` on purpose -- it gets hashed, and Go's
    /// `json.Marshal` sorts map keys. See [`go_json_encode`].
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameters: BTreeMap<String, Value>,

    /// `MESSAGE` lines, priming the conversation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<Message>,

    /// Named chat renderer, from `RENDERER`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub renderer: String,

    /// Named response parser, from `PARSER`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub parser: String,

    /// Minimum runtime version, from `REQUIRES`. Stored **without** the `v`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub requires: String,
}

/// What kind of value a `PARAMETER` name expects.
///
/// **Upstream:** the `reflect.Kind` switch inside `api.FormatParams`, which
/// reflects over the `Options` struct's JSON tags. Rust has no reflection, so
/// the mapping is a table -- more typing, but the test below keeps it honest,
/// whereas Go's version fails silently on a misspelled tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamKind {
    /// Go `int` / `*int` -> a JSON number, parsed as `i64`.
    Int,
    /// Go `float32` -> a JSON number, parsed as `f32` then widened for JSON.
    Float,
    /// Go `bool` / `*bool` -> `true` / `false`.
    Bool,
    /// Go `[]string` -> a JSON array. **Accumulates** across repeated lines.
    StringSlice,
}

/// Every `PARAMETER` name and its type.
///
/// **Upstream:** the JSON tags on `api.Options` (which embeds `api.Runner`), as
/// enumerated by `reflect.VisibleFields`. This table must stay in step with
/// [`crate::options::Options::apply_map`] -- the test
/// `every_parameter_name_is_one_the_options_struct_knows` asserts exactly that,
/// so a drift shows up as a failing test rather than as a Modelfile parameter
/// that silently does nothing.
const PARAM_KINDS: &[(&str, ParamKind)] = &[
    // ---- Runner (load-time) ----
    ("num_ctx", ParamKind::Int),
    ("num_batch", ParamKind::Int),
    ("num_gpu", ParamKind::Int),
    ("main_gpu", ParamKind::Int),
    ("use_mmap", ParamKind::Bool),
    ("num_thread", ParamKind::Int),
    ("draft_num_predict", ParamKind::Int),
    // ---- Options (per-request) ----
    ("num_keep", ParamKind::Int),
    ("seed", ParamKind::Int),
    ("num_predict", ParamKind::Int),
    ("top_k", ParamKind::Int),
    ("top_p", ParamKind::Float),
    ("min_p", ParamKind::Float),
    ("typical_p", ParamKind::Float),
    ("repeat_last_n", ParamKind::Int),
    ("temperature", ParamKind::Float),
    ("repeat_penalty", ParamKind::Float),
    ("presence_penalty", ParamKind::Float),
    ("frequency_penalty", ParamKind::Float),
    ("stop", ParamKind::StringSlice),
];

/// Coerce raw `PARAMETER` strings into the JSON types the params layer stores.
///
/// **Upstream:** `api.FormatParams(map[string][]string)`.
///
/// A Modelfile is all text -- `PARAMETER temperature 0.2` hands you the *string*
/// `"0.2"`. The params blob, though, is JSON with real types, and it gets
/// hashed. So `"0.2"` must become the number `0.2`, or two Modelfiles saying the
/// same thing would produce two different blobs and the store would hold both.
///
/// Unknown names are a **hard error** ([`CreateError::UnknownParameter`]),
/// matching upstream. That is the opposite of
/// [`crate::options::Options::apply_map`], which tolerates unknown keys at
/// request time -- and the asymmetry is deliberate on upstream's part: a typo in
/// a Modelfile should be caught while building the model, whereas a newer client
/// talking to an older server must not be broken by an option that server has
/// never heard of.
///
/// Go's parsing rules, kept exactly: `strconv.ParseBool` accepts
/// `1/t/T/TRUE/true/True/0/f/F/FALSE/false/False` and nothing else -- notably
/// **not** `yes`/`no`/`on`/`off`.
pub fn format_params(params: &BTreeMap<String, Vec<String>>) -> Result<BTreeMap<String, Value>> {
    let mut out = BTreeMap::new();
    for (key, vals) in params {
        let Some((_, kind)) = PARAM_KINDS.iter().find(|(k, _)| k == key) else {
            return Err(CreateError::UnknownParameter(key.clone()));
        };
        let first = vals.first().map(String::as_str).unwrap_or_default();
        let v = match kind {
            ParamKind::Int => {
                Value::from(
                    first
                        .parse::<i64>()
                        .map_err(|_| CreateError::BadParameterValue {
                            kind: "int",
                            value: first.to_string(),
                        })?,
                )
            }
            ParamKind::Float => {
                // Go widens float32 -> float64 for JSON, so 0.2f32 serialises as
                // 0.20000000298023224. Matched on purpose: the params blob is
                // hashed, and different number text is a different digest.
                let f = first
                    .parse::<f32>()
                    .map_err(|_| CreateError::BadParameterValue {
                        kind: "float",
                        value: first.to_string(),
                    })?;
                Value::from(f64::from(f))
            }
            ParamKind::Bool => Value::from(go_parse_bool(first).ok_or_else(|| {
                CreateError::BadParameterValue {
                    kind: "bool",
                    value: first.to_string(),
                }
            })?),
            ParamKind::StringSlice => Value::from(
                vals.iter()
                    .map(|s| Value::from(s.clone()))
                    .collect::<Vec<_>>(),
            ),
        };
        out.insert(key.clone(), v);
    }
    Ok(out)
}

/// **Upstream:** Go's `strconv.ParseBool`. Exactly this set, nothing else --
/// `yes`, `no`, `on` and `off` are all errors, which does surprise people.
fn go_parse_bool(s: &str) -> Option<bool> {
    match s {
        "1" | "t" | "T" | "TRUE" | "true" | "True" => Some(true),
        "0" | "f" | "F" | "FALSE" | "false" | "False" => Some(false),
        _ => None,
    }
}

/// Is this a version string `golang.org/x/mod/semver` would accept?
///
/// **Upstream:** the `semver.IsValid("v" + requires)` check in `CreateRequest`.
/// Takes the string **without** the `v`.
///
/// A **partial** port: `MAJOR[.MINOR[.PATCH]][-prerelease][+build]`, numeric
/// parts with no leading zeros. Enough for the `REQUIRES 0.14.0` shape the field
/// exists for. Prerelease *ordering* is not ported, and does not need to be --
/// nothing here compares two versions, it only validates one.
fn is_valid_semver_suffix(s: &str) -> bool {
    // Build metadata first, then prerelease -- neither joins the numeric check.
    let s = s.split('+').next().unwrap_or(s);
    let (core, pre) = match s.split_once('-') {
        Some((c, p)) => (c, Some(p)),
        None => (s, None),
    };
    if let Some(p) = pre
        && (p.is_empty()
            || p.split('.').any(|id| {
                id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
            }))
    {
        return false;
    }
    let parts: Vec<&str> = core.split('.').collect();
    if parts.is_empty() || parts.len() > 3 {
        return false;
    }
    parts.iter().all(|p| {
        !p.is_empty()
            && p.chars().all(|c| c.is_ascii_digit())
            && (p.len() == 1 || !p.starts_with('0'))
    })
}

/// Turn a parsed [`Modelfile`] into a [`CreateRequest`], resolving every local
/// path against `relative_dir` and hashing whatever it finds.
///
/// **Upstream:** `(Modelfile).CreateRequest(relativeDir)` in `parser/parser.go`
/// -- the half [`crate::modelfile`] deliberately stopped short of, because it is
/// the half that touches the filesystem.
///
/// `relative_dir` is **the directory the Modelfile itself lives in**, not the
/// process working directory. Pass the wrong one and `FROM ./model.gguf`
/// silently resolves somewhere else.
///
/// ## The one subtle branch
///
/// `FROM` is overloaded: it may name a **local path** or a **registry model**.
/// Upstream tells them apart by *trying* the filesystem -- if
/// [`file_digest_map`] comes back "no such file", the argument is taken as a
/// model name and lands in [`CreateRequest::from`]. Any **other** error
/// propagates. So a `FROM ./model.gguf` whose file exists but cannot be read
/// fails loudly instead of being mistaken for a model called `./model.gguf`.
///
/// `DRAFT` is checked against every `FROM` path and vice versa, by
/// [`reject_matching_local_path`]: a draft model that is the same weights as the
/// base model makes speculative decoding pointless.
///
/// Repeated `PARAMETER stop` lines **accumulate**; every other parameter is
/// last-one-wins. That is upstream's `if ks, ok := params[k].([]string)` branch,
/// and a Modelfile with several stop tokens depends on it.
pub fn create_request(
    modelfile: &Modelfile,
    relative_dir: &str,
    hasher: &mut dyn Sha256Hasher,
) -> Result<CreateRequest> {
    create_request_with(modelfile, relative_dir, hasher, &SystemUsers)
}

/// [`create_request`] with the user database injected, for `~user` paths.
///
/// **Upstream:** there is no separate upstream function -- `CreateRequest` calls
/// `expandPath`, which hardwires `user.Current` / `user.Lookup`. The extra seam
/// is ours, for the reason set out on [`SystemUsers`].
pub fn create_request_with(
    modelfile: &Modelfile,
    relative_dir: &str,
    hasher: &mut dyn Sha256Hasher,
    users: &dyn UserLookup,
) -> Result<CreateRequest> {
    let mut req = CreateRequest::default();
    let mut messages: Vec<Message> = Vec::new();
    let mut licenses: Vec<String> = Vec::new();
    let mut params: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut model_paths: Vec<String> = Vec::new();
    let mut draft_paths: Vec<String> = Vec::new();

    for c in &modelfile.commands {
        match c.name.as_str() {
            "model" => {
                let path = expand_path_with(&c.args, relative_dir, users)?;
                let path = path.to_string_lossy().into_owned();
                let digests = match file_digest_map(&path, hasher) {
                    Ok(d) => d,
                    // Upstream: `errors.Is(err, os.ErrNotExist)` -> it was a
                    // model NAME, not a path. Everything else propagates.
                    Err(CreateError::Io { source, .. })
                        if source.kind() == io::ErrorKind::NotFound =>
                    {
                        req.from.clone_from(&c.args);
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                reject_matching_local_path("DRAFT", &path, &draft_paths)?;
                model_paths.push(path);
                for (k, v) in digests {
                    req.files.insert(k, v.as_str().to_string());
                }
            }
            "draft" => {
                let path = expand_path_with(&c.args, relative_dir, users)?;
                let path = path.to_string_lossy().into_owned();
                let digests = file_digest_map(&path, hasher)?;
                reject_matching_local_path("DRAFT", &path, &model_paths)?;
                draft_paths.push(path);
                for (k, v) in digests {
                    req.draft_files.insert(k, v.as_str().to_string());
                }
            }
            "adapter" => {
                let path = expand_path_with(&c.args, relative_dir, users)?;
                let path = path.to_string_lossy().into_owned();
                // Upstream ASSIGNS rather than merges here, so a second ADAPTER
                // replaces the first. Faithful.
                req.adapters = file_digest_map(&path, hasher)?
                    .into_iter()
                    .map(|(k, v)| (k, v.as_str().to_string()))
                    .collect();
            }
            "template" => req.template.clone_from(&c.args),
            "system" => req.system.clone_from(&c.args),
            "license" => licenses.push(c.args.clone()),
            "renderer" => req.renderer.clone_from(&c.args),
            "parser" => req.parser.clone_from(&c.args),
            "requires" => {
                // golang.org/x/mod/semver wants a "v" prefix; upstream adds one
                // to validate, then strips it back off before storing.
                let bare = c.args.strip_prefix('v').unwrap_or(&c.args);
                if !is_valid_semver_suffix(bare) {
                    return Err(CreateError::BadRequires);
                }
                req.requires = bare.to_string();
            }
            "message" => {
                // Upstream: `strings.Cut(c.Args, ": ")`. No match -> the whole
                // string becomes the role and the content is empty. Faithful.
                let (role, msg) = match c.args.split_once(": ") {
                    Some((r, m)) => (r, m),
                    None => (c.args.as_str(), ""),
                };
                messages.push(Message::new(role, msg));
            }
            other => {
                if DEPRECATED_PARAMETERS.contains(&other) {
                    // Upstream prints a warning and drops it. A library has no
                    // business writing to stdout, so we drop it silently --
                    // `Modelfile::deprecated_parameters()` already exists for a
                    // caller that wants to warn.
                    continue;
                }
                params
                    .entry(other.to_string())
                    .or_default()
                    .push(c.args.clone());
            }
        }
    }

    // `format_params` runs once at the end rather than per line. Upstream calls
    // it per command and merges; the observable difference is nil, because the
    // only accumulating kind is `stop`, and gathering the raw strings first then
    // converting yields the same array.
    let formatted = format_params(&params)?;
    if !formatted.is_empty() {
        req.parameters = formatted;
    }
    if !messages.is_empty() {
        req.messages = messages;
    }
    if !licenses.is_empty() {
        req.license = Some(LicenseSpec::Many(licenses));
    }

    Ok(req)
}

// ===========================================================================
// Split GGUF filenames
// ===========================================================================

/// Take apart a llama.cpp split-GGUF filename.
///
/// **Upstream:** `splitGGUFName` in `server/create.go`, whose regex is
/// `^(.*)-(\d{5})-of-(\d{5})\.gguf$`.
///
/// Returns `(prefix, zero_based_index, count)`. Note the **index is shifted**:
/// the filename is one-based (`00001-of-00003`) while the GGUF metadata key
/// `split.no` is zero-based, so this returns `0` for `00001`. Get that backwards
/// and you silently pair shard 1's bytes with shard 2's metadata.
///
/// `00000-of-*` and `*-of-00000` are rejected -- upstream treats a zero in
/// either position as not-a-split-file rather than as shard zero.
///
/// **Deliberate divergence:** upstream uses `regexp`; this crate takes no regex
/// dependency, so the same grammar is hand-parsed. Exactly five digits in each
/// position, exactly as the regex demands -- `-1-of-3.gguf` does **not** match.
pub fn split_gguf_name(name: &str) -> Option<(String, u16, u16)> {
    // Upstream matches against `path.Base(name)`.
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let stem = base.strip_suffix(".gguf")?;
    let (head, count) = stem.rsplit_once("-of-")?;
    let (prefix, index) = head.rsplit_once('-')?;

    let five_digits = |s: &str| -> Option<u16> {
        if s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit()) {
            s.parse::<u16>().ok()
        } else {
            None
        }
    };
    let idx = five_digits(index)?;
    let n = five_digits(count)?;
    if idx == 0 || n == 0 {
        return None;
    }
    Some((prefix.to_string(), idx - 1, n))
}

// ===========================================================================
// Writing the layers
// ===========================================================================

/// Encode a value the way Go's `json.Encoder` would, byte for byte.
///
/// **This is not pedantry -- it decides digests.** Every side-car layer
/// (`params`, `messages`, the config blob) is JSON that gets SHA-256'd, and the
/// digest *is* the blob's address. Emit one byte differently from ollama and the
/// same model built by the two tools lands in two different blobs, a manifest
/// pulled from ollama's registry stops matching what we would have written, and
/// nothing dedupes.
///
/// Three differences between `serde_json::to_vec` and Go, all of which bite:
///
/// 1. **Go HTML-escapes by default.** `json.Encoder` (and `json.Marshal`)
///    replace `<`, `>` and `&` with `<`, `>`, `&` unless you call
///    `SetEscapeHTML(false)`, which upstream never does. This is not a corner
///    case here: stop tokens are things like `<|im_end|>` and `<end_of_turn>`,
///    so essentially **every** params blob is affected. The replacement is safe
///    to do on the whole byte stream, because `<`, `>` and `&` are not JSON
///    structural characters -- they can only ever occur inside a string literal.
/// 2. **Go sorts map keys.** `json.Marshal` of a `map[string]any` emits keys in
///    sorted order. Hence `BTreeMap` on [`CreateRequest::parameters`] rather
///    than [`indexmap::IndexMap`] -- and hence this function refusing to take
///    anything whose key order is not already sorted.
/// 3. **`json.Encoder.Encode` appends a newline.** `json.Marshal` does not.
///    Upstream uses the *Encoder* in `setParameters`, `setMessages` and
///    `createConfigLayer`, so the trailing `\n` is part of the hashed bytes.
///
/// Also escaped by Go: U+2028 and U+2029 (as ` ` / ` `), because they
/// are line terminators in JavaScript. Handled here too, for completeness --
/// they have not been observed in a real Modelfile, but a Chinese or Japanese
/// system prompt pasted out of a web page is exactly where one would show up.
fn go_json_encode<T: Serialize>(v: &T) -> Result<Vec<u8>> {
    let s = serde_json::to_string(v).map_err(json_ctx("encode layer"))?;
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            other => out.push(other),
        }
    }
    // json.Encoder.Encode's trailing newline. Part of the hash.
    out.push('\n');
    Ok(out.into_bytes())
}

/// Drop every layer of one media type from the list, deleting its blob too.
///
/// **Upstream:** `removeLayer(layers, mediatype)`.
///
/// Note upstream's behaviour when the blob will not delete: it logs
/// `"couldn't remove blob"` and **still returns `true`**, i.e. the layer leaves
/// the list regardless. That is the right call -- an undeletable blob is a
/// disk-space problem, whereas keeping a stale layer in the manifest would be a
/// *correctness* problem (the model would carry two templates). Same here: the
/// store error is swallowed, the layer goes.
fn remove_layer(store: &Store, layers: &mut Vec<Layer>, media_type: &str) {
    layers.retain(|l| {
        if l.media_type != media_type {
            return true;
        }
        let _ = store.remove_layer(l);
        false
    });
}

/// Replace the template layer.
///
/// **Upstream:** `setTemplate`. Parses first and refuses a template that will
/// not compile -- [`CreateError::BadTemplate`], which upstream maps to HTTP 400.
/// Validating at build time rather than at chat time is the whole point: a
/// broken `TEMPLATE` should stop `create`, not surface as garbled output three
/// days later.
fn set_template(
    store: &Store,
    layers: &mut Vec<Layer>,
    t: &str,
    hasher: &mut dyn Sha256Hasher,
) -> Result<()> {
    remove_layer(store, layers, MEDIA_TYPE_TEMPLATE);
    Template::parse(t).map_err(|e| CreateError::BadTemplate(e.to_string()))?;
    layers.push(store.new_layer(t.as_bytes(), MEDIA_TYPE_TEMPLATE, hasher)?);
    Ok(())
}

/// Replace the system-prompt layer.
///
/// **Upstream:** `setSystem`. An **empty** system prompt removes the layer
/// without adding a new one -- that is how `SYSTEM ""` clears an inherited
/// system prompt from a `FROM` parent, so the emptiness is meaningful, not a
/// no-op.
fn set_system(
    store: &Store,
    layers: &mut Vec<Layer>,
    s: &str,
    hasher: &mut dyn Sha256Hasher,
) -> Result<()> {
    remove_layer(store, layers, MEDIA_TYPE_SYSTEM);
    if !s.is_empty() {
        layers.push(store.new_layer(s.as_bytes(), MEDIA_TYPE_SYSTEM, hasher)?);
    }
    Ok(())
}

/// Append a licence layer.
///
/// **Upstream:** `setLicense`. Note it **appends** rather than replacing -- a
/// model may legitimately carry several licences, so unlike the template and
/// system layers there is no `removeLayer` first.
fn set_license(
    store: &Store,
    layers: &mut Vec<Layer>,
    l: &str,
    hasher: &mut dyn Sha256Hasher,
) -> Result<()> {
    layers.push(store.new_layer(l.as_bytes(), MEDIA_TYPE_LICENSE, hasher)?);
    Ok(())
}

/// Merge and rewrite the parameters layer.
///
/// **Upstream:** `setParameters`.
///
/// The merge direction is the bit worth remembering: any params layer **already**
/// in the list (inherited from a `FROM` parent) is read, and each of its keys is
/// copied across **only if the new request did not set it**. So a child
/// Modelfile overrides the parent per key, and inherits the rest -- it does not
/// wipe the parent's parameters wholesale.
///
/// If the merged result is empty, the existing layer is left **untouched**
/// (upstream returns early before `removeLayer`), so a Modelfile with no
/// `PARAMETER` lines does not strip its parent's.
fn set_parameters(
    store: &Store,
    layers: &mut Vec<Layer>,
    p: &BTreeMap<String, Value>,
    hasher: &mut dyn Sha256Hasher,
) -> Result<()> {
    let mut merged = p.clone();
    for layer in layers.iter() {
        if layer.media_type != MEDIA_TYPE_PARAMS {
            continue;
        }
        let digest = layer.checked_digest()?;
        let bytes = store.read_blob(&digest)?;
        let existing: BTreeMap<String, Value> =
            serde_json::from_slice(&bytes).map_err(json_ctx("parse existing params layer"))?;
        for (k, v) in existing {
            merged.entry(k).or_insert(v);
        }
    }

    if merged.is_empty() {
        return Ok(());
    }

    remove_layer(store, layers, MEDIA_TYPE_PARAMS);
    let bytes = go_json_encode(&merged)?;
    layers.push(store.new_layer(&bytes, MEDIA_TYPE_PARAMS, hasher)?);
    Ok(())
}

/// Replace the primed-messages layer.
///
/// **Upstream:** `setMessages`, whose own comment says it plainly: *"this leaves
/// the old messages intact if no new messages were specified, which may not be
/// the correct behaviour"*. Kept as-is -- ollama is the oracle, and a model
/// built by both tools must produce the same manifest even where upstream is
/// unsure of itself.
fn set_messages(
    store: &Store,
    layers: &mut Vec<Layer>,
    m: &[Message],
    hasher: &mut dyn Sha256Hasher,
) -> Result<()> {
    if m.is_empty() {
        return Ok(());
    }
    remove_layer(store, layers, MEDIA_TYPE_MESSAGES);
    let bytes = go_json_encode(&m)?;
    layers.push(store.new_layer(&bytes, MEDIA_TYPE_MESSAGES, hasher)?);
    Ok(())
}

/// Build the config layer, stamping every other layer's digest into it.
///
/// **Upstream:** `createConfigLayer`.
///
/// `rootfs.diff_ids` is filled with the digests of all the layers **in order**.
/// The OCI spec wants uncompressed-layer IDs there; ollama just reuses the layer
/// digests, and order matters because the field is a list, not a set. So the
/// config blob's own digest transitively covers every layer -- change any layer
/// and the config digest moves too.
fn create_config_layer(
    store: &Store,
    layers: &[Layer],
    config: &ConfigV2,
    hasher: &mut dyn Sha256Hasher,
) -> Result<Layer> {
    let mut config = config.clone();
    config.rootfs.diff_ids = layers.iter().map(|l| l.digest.clone()).collect();
    let bytes = go_json_encode(&config)?;
    Ok(store.new_layer(&bytes, MEDIA_TYPE_CONFIG, hasher)?)
}

/// Fill in `renderer` / `parser` / default stop tokens from a GGUF architecture.
///
/// **Upstream:** the `switch arch` block inside `createModel`'s
/// `"application/vnd.ollama.image.model"` case, along with its TODO: *"abstract
/// this into a registry/lookup table when multiple models need
/// architecture-based renderer/parser/stop defaults."*
///
/// Split out as its own function here because this crate cannot read the GGUF
/// that supplies `architecture` -- see the module docs on the seam. Caller
/// passes `general.architecture` from the model layer's KV table.
///
/// Nothing is overwritten: each field is only filled if currently empty, which
/// is Go's `cmp.Or`. An explicit `RENDERER` in the Modelfile therefore always
/// beats the architecture default, and the whole block is skipped once both
/// renderer and parser are set.
///
/// | `general.architecture` | renderer | parser | stop |
/// |---|---|---|---|
/// | `gemma4` | `gemma4` | `gemma4` | `["<turn\|>"]` |
/// | `laguna` | `laguna` | `laguna` | -- |
/// | `nemotron_h`, `nemotron_h_moe`, `nemotron_h_omni` | `nemotron-3-nano` | `nemotron-3-nano` | -- |
///
/// The `gemma4` renderer name written here is the **legacy** one, which
/// [`resolve_renderer_name`] later narrows to `gemma4-small` / `gemma4-large`
/// per model size. Writing the narrowed name at create time would be wrong: the
/// manifest must stay valid if the size metadata is later corrected.
pub fn apply_architecture_defaults(
    config: &mut ConfigV2,
    parameters: &mut BTreeMap<String, Value>,
    architecture: &str,
) {
    if !config.renderer.is_empty() && !config.parser.is_empty() {
        return;
    }
    let or = |dst: &mut String, v: &str| {
        if dst.is_empty() {
            *dst = v.to_string();
        }
    };
    match architecture {
        "gemma4" => {
            or(&mut config.renderer, GEMMA4_RENDERER_LEGACY);
            or(&mut config.parser, "gemma4");
            // Upstream only sets this when the Modelfile did not.
            parameters
                .entry("stop".to_string())
                .or_insert_with(|| Value::from(vec![Value::from("<turn|>")]));
        }
        "laguna" => {
            or(&mut config.renderer, "laguna");
            or(&mut config.parser, "laguna");
        }
        "nemotron_h" | "nemotron_h_moe" | "nemotron_h_omni" => {
            or(&mut config.renderer, "nemotron-3-nano");
            or(&mut config.parser, "nemotron-3-nano");
        }
        _ => {}
    }
}

/// Assemble the side-car layers and write the manifest. The end of the road for
/// `create`.
///
/// **Upstream:** `createModel(r, name, baseLayers, config, fn)` in
/// `server/create.go`, minus the GGUF-rewriting half (see the module docs).
///
/// `base_layers` are the weight-bearing layers -- model, projector, adapter,
/// draft -- already in the store, whether they came from `FROM another-model`
/// (reused by digest, no bytes moved) or from freshly hashed local files. This
/// function adds the small text layers around them, builds the config blob, and
/// writes the manifest that ties the lot together. Returns the manifest path.
///
/// Order matters and is upstream's: template, system, licences, parameters,
/// messages, then config. Parameters after licences because the params merge
/// reads any layer already present; config last because it must hash every other
/// layer's digest.
///
/// **What would make this wrong:** calling it with `base_layers` whose blobs are
/// not yet in the store. The manifest would name blobs that do not exist, and
/// the model would fail at load time rather than here.
pub fn create_model(
    store: &Store,
    name: &Name,
    base_layers: Vec<Layer>,
    config: &ConfigV2,
    req: &CreateRequest,
    hasher: &mut dyn Sha256Hasher,
) -> Result<PathBuf> {
    let mut layers = base_layers;

    if !req.template.is_empty() {
        set_template(store, &mut layers, &req.template, hasher)?;
    }
    if !req.system.is_empty() {
        set_system(store, &mut layers, &req.system, hasher)?;
    }
    if let Some(license) = &req.license {
        for l in license.as_list() {
            set_license(store, &mut layers, l, hasher)?;
        }
    }
    set_parameters(store, &mut layers, &req.parameters, hasher)?;
    set_messages(store, &mut layers, &req.messages, hasher)?;

    let config_layer = create_config_layer(store, &layers, config, hasher)?;
    Ok(store.write_manifest(name, config_layer, layers)?)
}

// ===========================================================================
// Renderer resolution
// ===========================================================================
//
// Upstream: server/renderer_resolution.go, in full.

/// The renderer name written into a manifest by `create`. Never used to render.
///
/// **Upstream:** `gemma4RendererLegacy`. It is "legacy" in the sense that it
/// names a *family*, not a template -- [`resolve_renderer_name`] narrows it at
/// load time.
pub const GEMMA4_RENDERER_LEGACY: &str = "gemma4";

/// **Upstream:** `gemma4RendererSmall`. The e2b/e4b prompt shape.
pub const GEMMA4_RENDERER_SMALL: &str = "gemma4-small";

/// **Upstream:** `gemma4RendererLarge`. The 12b/26b/31b prompt shape.
pub const GEMMA4_RENDERER_LARGE: &str = "gemma4-large";

/// Parameter count at which a Gemma 4 model switches to the large template.
///
/// **Upstream:** `gemma4LargeMinParameterCount = 12_000_000_000`, with the
/// comment *"Gemma 4 small templates cover the e2b/e4b family, while 12b/26b/31b
/// use the large template. Default to the small prompt unless the model is
/// clearly in the large range."*
pub const GEMMA4_LARGE_MIN_PARAMETER_COUNT: u64 = 12_000_000_000;

/// Pick the concrete renderer a model should actually use.
///
/// **Upstream:** `resolveRendererName(m *Model)`.
///
/// Everything passes straight through except the legacy `gemma4` name, which is
/// narrowed by size -- see [`resolve_gemma4_renderer`]. An empty
/// `config.renderer` stays empty, which is the signal to fall back to the
/// generic template path rather than a named renderer.
pub fn resolve_renderer_name(model: &Model) -> &str {
    if model.config.renderer.is_empty() {
        return "";
    }
    if model.config.renderer == GEMMA4_RENDERER_LEGACY {
        return resolve_gemma4_renderer(model);
    }
    &model.config.renderer
}

/// Narrow `gemma4` to `gemma4-small` or `gemma4-large`.
///
/// **Upstream:** `resolveGemma4Renderer`.
///
/// Three sources, tried in order, and the order is the point -- the **name** is
/// trusted over the metadata, because a name like `gemma4:e4b` is what the user
/// asked for while `model_type` is derived and can be wrong:
///
/// 1. the short name (`gemma4:e4b`),
/// 2. the full name (`registry.ollama.ai/library/gemma4:12b`),
/// 3. `config.model_type` as a human number (`"12B"`), compared against
///    [`GEMMA4_LARGE_MIN_PARAMETER_COUNT`],
/// 4. failing all of that, **small** -- upstream's deliberate default. Getting
///    this wrong yields a working model with a subtly mis-shaped prompt, which
///    is why the fallback is the more conservative of the two.
pub fn resolve_gemma4_renderer(model: &Model) -> &'static str {
    if let Some(r) = gemma4_renderer_from_name(&model.short_name) {
        return r;
    }
    if let Some(r) = gemma4_renderer_from_name(&model.name) {
        return r;
    }
    if let Some(count) = parse_human_parameter_count(&model.config.model_type) {
        return gemma4_renderer_for_parameter_count(count);
    }
    GEMMA4_RENDERER_SMALL
}

/// **Upstream:** `gemma4RendererForParameterCount`.
pub fn gemma4_renderer_for_parameter_count(parameter_count: u64) -> &'static str {
    if parameter_count >= GEMMA4_LARGE_MIN_PARAMETER_COUNT {
        GEMMA4_RENDERER_LARGE
    } else {
        GEMMA4_RENDERER_SMALL
    }
}

/// Read the size straight out of a model name.
///
/// **Upstream:** `gemma4RendererFromName`. Substring match, case-insensitive:
/// `e2b`/`e4b` -> small, `12b`/`26b`/`31b` -> large, anything else -> `None`.
///
/// Substring, not suffix -- `gemma4:12b-instruct-q4_K_M` must still say large.
pub fn gemma4_renderer_from_name(name: &str) -> Option<&'static str> {
    let lower = name.to_lowercase();
    if lower.contains("e2b") || lower.contains("e4b") {
        Some(GEMMA4_RENDERER_SMALL)
    } else if lower.contains("12b") || lower.contains("26b") || lower.contains("31b") {
        Some(GEMMA4_RENDERER_LARGE)
    } else {
        None
    }
}

/// Turn `"12B"` / `"600M"` / `"1.5K"` back into a number.
///
/// **Upstream:** `parseHumanParameterCount`, the inverse of
/// [`crate::format::human_number`]. The magnitudes are `format.Thousand = 1000`,
/// `Million = Thousand * 1000`, `Billion = Million * 1000` from
/// `format/format.go` -- decimal SI, **not** binary, so `1B` is 1e9 and not 2^30.
///
/// The unit is the last character and is required; anything else returns `None`.
/// Note the deliberate lossiness: `human_number` rounds `"7.6B"` to one decimal,
/// so round-tripping does not recover the exact parameter count -- which is fine,
/// the only consumer compares against a 12e9 threshold.
pub fn parse_human_parameter_count(s: &str) -> Option<u64> {
    let unit = s.chars().last()?;
    let multiplier: f64 = match unit.to_ascii_uppercase() {
        'B' => 1_000_000_000.0,
        'M' => 1_000_000.0,
        'K' => 1_000.0,
        _ => return None,
    };
    let value: f64 = s[..s.len() - unit.len_utf8()].parse().ok()?;
    // Go's `uint64(value * multiplier)` truncates toward zero; a negative value
    // would be undefined behaviour in Go and is simply rejected here.
    let product = value * multiplier;
    if product < 0.0 || !product.is_finite() {
        return None;
    }
    Some(product as u64)
}

/// **Upstream:** `isGemma4Renderer`. Used by
/// `Model::filter_unsupported_capabilities`, so it must cover the legacy name --
/// a manifest written by `create` carries `gemma4`, not the narrowed form.
pub fn is_gemma4_renderer(renderer: &str) -> bool {
    matches!(
        renderer,
        GEMMA4_RENDERER_LEGACY | GEMMA4_RENDERER_SMALL | GEMMA4_RENDERER_LARGE
    )
}

// ===========================================================================
// The loaded model, and what it can do
// ===========================================================================

/// The facts capability inference needs out of a model's GGUF KV table.
///
/// This crate has no GGUF decoder (see the module docs), so the caller reads
/// these and passes them in. **Each field names the exact GGUF key it must come
/// from** -- fill one from somewhere else and the capability set silently goes
/// wrong, which is the worst kind of wrong here because it decides whether a
/// model is ever *offered* tools or vision.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GgufFacts {
    /// GGUF key `general.architecture`. Used to suppress audio on
    /// `nemotron_h_omni`.
    pub architecture: String,
    /// GGUF key `tokenizer.chat_template` -- the raw Jinja template string.
    /// Empty when the model does not carry one.
    pub chat_template: String,
    /// Is GGUF key `pooling_type` present *and valid*? Presence means the model
    /// is an **embedding** model, absence means it does completion. Upstream's
    /// comment: *"If no embedding is specified, we assume the model supports
    /// completion."*
    pub has_pooling_type: bool,
    /// Is GGUF key `vision.block_count` present?
    pub has_vision_block_count: bool,
    /// Is GGUF key `audio.block_count` present?
    pub has_audio_block_count: bool,
}

/// The facts capability inference needs out of a **projector** GGUF.
///
/// One per entry in [`Model::projector_paths`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProjectorFacts {
    /// GGUF key `has_audio_encoder`, **or any key ending in
    /// `.has_audio_encoder`** being true. Upstream scans every key for that
    /// suffix, because different projector families namespace it differently.
    pub has_audio_encoder: bool,
    /// GGUF key `vision.projector_type`. `"gemma3nv"` suppresses the audio
    /// capability even when an audio encoder is present -- see
    /// `projectorSuppressesAudioCapability`.
    pub vision_projector_type: String,
}

/// Which template a capability question is being asked about.
///
/// **Upstream:** `templateCapabilitySource`. Not a stylistic enum -- the same
/// model yields *different* capability sets depending on which template the
/// runtime is going to use, and the logging in `logTemplateSelection` compares
/// all of them side by side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateCapabilitySource {
    /// **Upstream:** `templateCapabilitySelected`. Whatever the runtime will
    /// actually use.
    Selected,
    /// **Upstream:** `templateCapabilityGo`. Force the Modelfile `TEMPLATE`.
    GoTemplate,
    /// **Upstream:** `templateCapabilityChat`. Force the GGUF `chat_template`.
    ChatTemplate,
}

/// A model read back out of the store.
///
/// **Upstream:** `type Model struct` in `server/images.go`.
///
/// **Not ported:** `Digest` is kept, but `Options` is a plain JSON map rather
/// than being folded into [`crate::options::Options`] -- upstream keeps it a map
/// on purpose, *"so that we can see which fields have been specified
/// explicitly"*, and that distinction is exactly what a request-time overlay
/// needs.
#[derive(Debug, Clone)]
pub struct Model {
    /// Fully qualified name, e.g. `registry.ollama.ai/library/qwen3:0.6b`.
    pub name: String,
    /// The shortest unambiguous spelling, e.g. `qwen3:0.6b`.
    pub short_name: String,
    /// The manifest's own digest.
    pub digest: String,
    /// The parsed config blob.
    pub config: ConfigV2,
    /// Blob path of the weights layer, if there is one.
    pub model_path: Option<PathBuf>,
    /// Blob path of the speculative-decoding draft weights.
    pub draft_path: Option<PathBuf>,
    /// `Layer::from` of the model layer -- which model these weights came from.
    pub parent_model: String,
    /// Does the GGUF carry a `tokenizer.chat_template`?
    pub has_chat_template: bool,
    /// Did the manifest carry a Modelfile `TEMPLATE` layer?
    pub has_go_template: bool,
    /// Should the GGUF chat template win over the Go `TEMPLATE`? See
    /// [`should_prefer_chat_template`].
    pub prefer_chat_template: bool,
    /// Blob paths of LoRA adapters.
    pub adapter_paths: Vec<PathBuf>,
    /// Blob paths of multimodal projectors.
    pub projector_paths: Vec<PathBuf>,
    /// The `SYSTEM` prompt.
    pub system: String,
    /// Licence texts.
    pub license: Vec<String>,
    /// Stored parameters, as a raw JSON map. See the struct docs for why.
    pub options: BTreeMap<String, Value>,
    /// Primed conversation.
    pub messages: Vec<Message>,
    /// The prompt template. Defaults to [`Template::default_template`] when the
    /// manifest has no template layer, exactly as upstream.
    pub template: Template,
}

impl Model {
    /// **Upstream:** `(*Model).IsMLX()`.
    pub fn is_mlx(&self) -> bool {
        self.config.model_format == "safetensors"
    }

    /// **Upstream:** `(*Model).isGGUF()`.
    ///
    /// Note the empty string counts as GGUF: manifests written before
    /// `model_format` existed carry no value, and they are all GGUF.
    pub fn is_gguf(&self) -> bool {
        self.config.model_format.is_empty() || self.config.model_format == "gguf"
    }

    /// **Upstream:** `shouldUseHarmony(model)`.
    ///
    /// gpt-oss models are parsed with the Harmony format, but only the ones
    /// whose template actually expects it -- upstream's heuristic is to look for
    /// the two tags Harmony templates nearly always carry.
    ///
    /// **Note:** `routes.rs` (owned by another agent) is upstream's home for
    /// this predicate. It is duplicated here because capability inference cannot
    /// be computed without it, and a create-time module that reached into
    /// `routes` for it would invert the dependency. Worth de-duplicating once
    /// both modules land -- flagged rather than silently forked.
    pub fn should_use_harmony(&self) -> bool {
        matches!(self.config.model_family.as_str(), "gptoss" | "gpt-oss")
            && self.template.contains("<|start|>")
            && self.template.contains("<|end|>")
    }

    /// **Upstream:** `shouldUseGoTemplate(m)`. See the note on
    /// [`Model::should_use_harmony`] about the duplication.
    ///
    /// `env` supplies `OLLAMA_GO_TEMPLATE`. The double call
    /// (`go_template(true)` vs `go_template(false)`) is upstream's trick for
    /// asking *"was the variable set at all?"* -- if the two disagree, it is
    /// unset and the default is what differed.
    pub fn should_use_go_template(&self, env: &crate::envconfig::Env) -> bool {
        if !self.has_go_template {
            return false;
        }
        if go_template_env_set(env) {
            return env.go_template(true);
        }
        !self.prefer_chat_template && env.go_template(true)
    }

    /// **Upstream:** `usesOllamaRenderedChat(m)` -- is the prompt being built by
    /// something other than the GGUF's own chat template?
    pub fn uses_ollama_rendered_chat(&self, env: &crate::envconfig::Env) -> bool {
        !self.config.renderer.is_empty()
            || !self.config.parser.is_empty()
            || self.should_use_harmony()
            || self.should_use_go_template(env)
    }

    /// What this model can do.
    ///
    /// **Upstream:** `(*Model).Capabilities()`, which is
    /// `capabilitiesForTemplate(templateCapabilitySelected, nil)`.
    ///
    /// An **empty** result is upstream's "unknown capabilities for model"
    /// warning case, not an error -- returned as an empty `Vec` here so the
    /// caller can decide.
    pub fn capabilities(
        &self,
        gguf: Option<&GgufFacts>,
        projectors: &[ProjectorFacts],
        env: &crate::envconfig::Env,
    ) -> Vec<Capability> {
        self.capabilities_for_template(TemplateCapabilitySource::Selected, gguf, projectors, env)
    }

    /// **Upstream:** `(*Model).capabilitiesForTemplate(source, f)`.
    ///
    /// The pipeline, in upstream's exact order -- and the order matters, because
    /// `Model::filter_unsupported_capabilities` runs **last** and removes things
    /// the earlier stages added:
    ///
    /// 1. whatever the config blob declares outright,
    /// 2. what the GGUF implies (embedding vs completion, vision, audio, and the
    ///    chat template's own tool/thinking markers),
    /// 3. what the projectors add (vision, and audio if one carries an encoder),
    /// 4. what the Go template's variables imply,
    /// 5. what the named parser supports,
    /// 6. model-family special cases (gpt-oss always thinks),
    /// 7. **subtract** the combinations known to be broken.
    pub fn capabilities_for_template(
        &self,
        source: TemplateCapabilitySource,
        gguf: Option<&GgufFacts>,
        projectors: &[ProjectorFacts],
        env: &crate::envconfig::Env,
    ) -> Vec<Capability> {
        let mut caps: Vec<Capability> = Vec::new();
        self.config_capabilities(&mut caps);
        let arch = self.gguf_capabilities(&mut caps, source, gguf, env);
        projector_capabilities(&mut caps, projectors);
        self.template_capabilities(&mut caps, source, env);
        self.parser_capabilities(&mut caps);
        self.model_family_capabilities(&mut caps);
        self.filter_unsupported_capabilities(&mut caps, &arch);
        caps
    }

    /// **Upstream:** `(*Model).configCapabilities`. Capabilities the manifest
    /// declares outright -- how a published model overrides inference.
    ///
    /// An unrecognised name is **dropped**, not an error: upstream's
    /// `model.Capability(c)` is a string conversion that would happily carry a
    /// nonsense value forward, but every consumer compares against the known
    /// set, so an unknown one can never match anything anyway.
    fn config_capabilities(&self, caps: &mut Vec<Capability>) {
        for c in &self.config.capabilities {
            if let Some(cap) = capability_from_str(c) {
                append_capability(caps, cap);
            }
        }
    }

    /// **Upstream:** `(*Model).ggufCapabilities`. Returns the architecture, for
    /// [`Model::filter_unsupported_capabilities`] to use later.
    fn gguf_capabilities(
        &self,
        caps: &mut Vec<Capability>,
        source: TemplateCapabilitySource,
        gguf: Option<&GgufFacts>,
        env: &crate::envconfig::Env,
    ) -> String {
        if self.model_path.is_none() || !self.is_gguf() {
            return String::new();
        }
        let Some(f) = gguf else {
            // Upstream logs "couldn't open model file" and carries on with what
            // it has. Same here: no facts is not a failure, just less known.
            return String::new();
        };

        match source {
            TemplateCapabilitySource::Selected => {
                // Only consult the chat template if it is the thing that will
                // actually build the prompt.
                if !self.uses_ollama_rendered_chat(env) {
                    chat_template_capabilities(caps, &f.chat_template);
                }
            }
            TemplateCapabilitySource::ChatTemplate => {
                chat_template_capabilities(caps, &f.chat_template);
            }
            TemplateCapabilitySource::GoTemplate => {}
        }

        if f.has_pooling_type {
            append_capability(caps, Capability::Embedding);
        } else {
            // Upstream: "If no embedding is specified, we assume the model
            // supports completion."
            append_capability(caps, Capability::Completion);
        }
        if f.has_vision_block_count {
            append_capability(caps, Capability::Vision);
        }
        if f.has_audio_block_count {
            append_capability(caps, Capability::Audio);
        }

        f.architecture.clone()
    }

    /// **Upstream:** `(*Model).templateCapabilities`.
    ///
    /// The `Selected` arm has a subtlety: if the model *has* a Go template but
    /// the runtime is not going to use it, its variables must not contribute --
    /// otherwise a model would advertise tools it will never be prompted for.
    fn template_capabilities(
        &self,
        caps: &mut Vec<Capability>,
        source: TemplateCapabilitySource,
        env: &crate::envconfig::Env,
    ) {
        match source {
            TemplateCapabilitySource::Selected => {
                if self.has_go_template && !self.should_use_go_template(env) {
                    return;
                }
            }
            TemplateCapabilitySource::GoTemplate => {
                if !self.has_go_template {
                    return;
                }
            }
            TemplateCapabilitySource::ChatTemplate => return,
        }
        for c in go_template_capabilities(&self.template) {
            append_capability(caps, c);
        }
    }

    /// **Upstream:** `(*Model).parserCapabilities`. An unknown parser name
    /// contributes nothing, which is [`crate::parsers::parser_for_name`]
    /// returning `None`.
    fn parser_capabilities(&self, caps: &mut Vec<Capability>) {
        let Some(p) = crate::parsers::parser_for_name(&self.config.parser) else {
            return;
        };
        if p.has_tool_support() {
            append_capability(caps, Capability::Tools);
        }
        if p.has_thinking_support() {
            append_capability(caps, Capability::Thinking);
        }
    }

    /// **Upstream:** `(*Model).modelFamilyCapabilities`. gpt-oss always thinks,
    /// regardless of what its template says.
    fn model_family_capabilities(&self, caps: &mut Vec<Capability>) {
        if matches!(self.config.model_family.as_str(), "gptoss" | "gpt-oss") {
            append_capability(caps, Capability::Thinking);
        }
    }

    /// **Upstream:** `(*Model).filterUnsupportedCapabilities`.
    ///
    /// Subtraction, and it must run last. Two known-broken combinations:
    ///
    /// * **audio** is dropped when [`suppress_audio_capability`] says so;
    /// * **vision** is dropped for a Gemma 4 renderer on a safetensors model.
    fn filter_unsupported_capabilities(&self, caps: &mut Vec<Capability>, arch: &str) {
        if self.suppress_audio_capability(arch) {
            caps.retain(|c| *c != Capability::Audio);
        }
        if is_gemma4_renderer(&self.config.renderer) && self.config.model_format == "safetensors" {
            caps.retain(|c| *c != Capability::Vision);
        }
    }

    /// **Upstream:** `suppressAudioCapability(m, arch)`.
    ///
    /// Two cases, both with a reason attached upstream:
    ///
    /// * Gemma 4 on safetensors -- the MLX path has no audio.
    /// * anything nemotron-h-omni, whether that comes from the GGUF
    ///   architecture, `model_family`, or `model_families`. Upstream's TODO
    ///   explains why: *"expose Nemotron3 audio once llama.cpp can skip or load
    ///   the audio projector safely."* So this is a temporary suppression of a
    ///   real capability, not a statement that the model lacks it -- do not
    ///   "clean it up" by deleting the branch.
    fn suppress_audio_capability(&self, arch: &str) -> bool {
        if is_gemma4_renderer(&self.config.renderer) && self.config.model_format == "safetensors" {
            return true;
        }
        arch == "nemotron_h_omni"
            || self.config.model_family == "nemotron_h_omni"
            || self
                .config
                .model_families
                .iter()
                .any(|f| f == "nemotron_h_omni")
    }

    /// Check the model has everything the caller needs, listing what is missing.
    ///
    /// **Upstream:** `(*Model).CheckCapabilities(want ...)`, which joins one
    /// error per missing capability.
    ///
    /// The qwen3 / deepseek-r1 special case is upstream's and is kept: those two
    /// shipped manifests **before** thinking support existed, so the honest
    /// advice for a missing `thinking` on them is "pull the model again", not
    /// "this model cannot think".
    pub fn check_capabilities(
        &self,
        want: &[Capability],
        gguf: Option<&GgufFacts>,
        projectors: &[ProjectorFacts],
        env: &crate::envconfig::Env,
    ) -> std::result::Result<(), String> {
        let available = self.capabilities(gguf, projectors, env);
        let missing: Vec<Capability> = want
            .iter()
            .copied()
            .filter(|c| !available.contains(c))
            .collect();
        if missing.is_empty() {
            return Ok(());
        }

        let list = missing
            .iter()
            .map(|c| format!("does not support {c}"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut msg = format!("{} {list}", "model missing capabilities:");
        if missing.contains(&Capability::Thinking)
            && (self.config.model_family == "qwen3"
                || Name::parse(&self.name).model == "deepseek-r1")
        {
            msg.push_str(
                ". Pull the model again to get the latest version with full thinking support",
            );
        }
        Err(msg)
    }
}

/// **Upstream:** `appendCapability` -- a set that keeps insertion order.
///
/// Order is not decorative: `CheckCapabilities` and the log lines both render
/// this list, and a stable order keeps those diffable.
fn append_capability(caps: &mut Vec<Capability>, c: Capability) {
    if !caps.contains(&c) {
        caps.push(c);
    }
}

/// Parse a capability name out of a config blob.
///
/// **Upstream:** `model.Capability(c)`, a bare string conversion. The names are
/// **wire values baked into published manifests**, so they are not ours to tidy.
fn capability_from_str(s: &str) -> Option<Capability> {
    Some(match s {
        "completion" => Capability::Completion,
        "tools" => Capability::Tools,
        "insert" => Capability::Insert,
        "vision" => Capability::Vision,
        "embedding" => Capability::Embedding,
        "thinking" => Capability::Thinking,
        "image" => Capability::Image,
        "audio" => Capability::Audio,
        _ => return None,
    })
}

/// **Upstream:** `(*Model).projectorCapabilities`.
///
/// A projector at all means vision. Audio needs an encoder **and** a projector
/// type that does not suppress it.
fn projector_capabilities(caps: &mut Vec<Capability>, projectors: &[ProjectorFacts]) {
    if projectors.is_empty() {
        return;
    }
    append_capability(caps, Capability::Vision);
    for p in projectors {
        if p.has_audio_encoder && !projector_suppresses_audio(p) {
            append_capability(caps, Capability::Audio);
        }
    }
}

/// **Upstream:** `projectorSuppressesAudioCapability`. `gemma3nv` projectors
/// carry an audio encoder that does not work through this path.
fn projector_suppresses_audio(p: &ProjectorFacts) -> bool {
    p.vision_projector_type == "gemma3nv"
}

/// What a GGUF `tokenizer.chat_template` implies.
///
/// **Upstream:** `chatTemplateCapabilities`. Substring heuristics over the Jinja
/// source -- crude, but it is what upstream ships, and it is the oracle.
fn chat_template_capabilities(caps: &mut Vec<Capability>, chat_template: &str) {
    if chat_template.is_empty() {
        return;
    }
    if chat_template_has_tool_support(chat_template) {
        append_capability(caps, Capability::Tools);
    }
    if chat_template_has_thinking_support(chat_template) {
        append_capability(caps, Capability::Thinking);
    }
}

/// **Upstream:** `chatTemplateHasToolSupport`. Does the template so much as
/// mention tools?
pub fn chat_template_has_tool_support(chat_template: &str) -> bool {
    chat_template.contains("tools") || chat_template.contains("tool_call")
}

/// Can the template render a **full** tool round trip -- the call *and* the
/// result coming back?
///
/// **Upstream:** `chatTemplateHasToolRoundTrip`. The long `||` chain is not
/// noise: different model families spell "this message is a tool result"
/// differently (`tool_response`, `tool_results`, `role'] == 'tool'` in four
/// quoting styles, and Llama's `ipython`), and a template that can emit a call
/// but cannot render the reply produces a conversation that dies after one turn.
/// That is why this is a separate, stricter question from
/// [`chat_template_has_tool_support`].
pub fn chat_template_has_tool_round_trip(chat_template: &str) -> bool {
    if !chat_template_has_tool_support(chat_template) {
        return false;
    }
    let tool_calls = chat_template.contains("tool_calls") || chat_template.contains("assistant_tool_call");
    tool_calls
        && (chat_template.contains("tool_response")
            || chat_template.contains("tool_results")
            || chat_template.contains("role'] == 'tool'")
            || chat_template.contains("role'] == \"tool\"")
            || chat_template.contains("role\"] == 'tool'")
            || chat_template.contains("role\"] == \"tool\"")
            || chat_template.contains("message.role == 'tool'")
            || chat_template.contains("message.role == \"tool\"")
            || chat_template.contains("ipython"))
}

/// **Upstream:** `chatTemplateHasThinkingSupport`.
///
/// The obvious case is a `<think>`/`</think>` pair. The second clause covers a
/// real family of Qwen/DeepSeek templates that **strip** prior reasoning by
/// splitting assistant content at `</think>` -- upstream's comment notes
/// llama.cpp can still extract reasoning from those. The two exclusions matter:
/// a template mentioning `reasoning_content` or `<SPECIAL_12>` handles thinking
/// through a different mechanism and must not be caught by this heuristic.
pub fn chat_template_has_thinking_support(chat_template: &str) -> bool {
    if chat_template.contains("<think>") && chat_template.contains("</think>") {
        return true;
    }
    (chat_template.contains("content.split('</think>')")
        || chat_template.contains("content.split(\"</think>\")"))
        && !chat_template.contains("reasoning_content")
        && !chat_template.contains("<SPECIAL_12>")
}

/// What a Modelfile `TEMPLATE`'s variables imply.
///
/// **Upstream:** `goTemplateCapabilities(t)`.
///
/// * a `tools` variable -> [`Capability::Tools`]
/// * a `suffix` variable -> [`Capability::Insert`] (fill-in-the-middle)
/// * inferable thinking tags -> [`Capability::Thinking`]
///
/// ## The graft, and why it is replicated here
///
/// Upstream passes `t.Template`, the **grafted** Go template -- `template.Parse`
/// appends a `{{ .Response }}` node to templates that mention neither `messages`
/// nor `response`, mutating the tree in place. [`crate::template::Template`]
/// keeps its `raw` as the *pre-graft* source and its grafted tree private, so
/// this re-parses and re-applies exactly the same graft before handing the tree
/// to [`crate::thinking::infer_tags`]. Doing it any other way would ask the tag
/// inference about a different tree than upstream does.
fn go_template_capabilities(t: &Template) -> Vec<Capability> {
    let mut caps = Vec::new();
    let vars = t.vars();
    if vars.iter().any(|v| v == "tools") {
        append_capability(&mut caps, Capability::Tools);
    }
    if vars.iter().any(|v| v == "suffix") {
        append_capability(&mut caps, Capability::Insert);
    }
    if let Some(inner) = grafted_gotmpl(t.raw())
        && crate::thinking::infer_tags(&inner).is_some()
    {
        append_capability(&mut caps, Capability::Thinking);
    }
    caps
}

/// Rebuild the grafted `gotmpl` tree from a template's raw source.
///
/// Mirrors [`crate::template::Template::parse`]'s graft exactly. Returns `None`
/// if the source will not parse -- upstream logs *"model template contains
/// errors"* and contributes no capabilities, and `None` says the same thing.
fn grafted_gotmpl(raw: &str) -> Option<crate::gotmpl::Template> {
    use crate::gotmpl::parse::{Arg, Command as TCommand, Node, Pipeline};
    let inner = crate::gotmpl::Template::parse(raw).ok()?;
    let vars = inner.vars();
    if vars.iter().any(|v| v == "messages" || v == "response") {
        return Some(inner);
    }
    let mut nodes = inner.nodes().clone();
    nodes.push(Node::Action(Pipeline {
        cmds: vec![TCommand {
            args: vec![Arg::Field(vec!["Response".to_string()])],
        }],
    }));
    Some(crate::gotmpl::Template::from_nodes(
        nodes,
        inner.raw().to_string(),
    ))
}

/// **Upstream:** `goTemplateHasToolRoundTrip(t)`. The Go-template counterpart of
/// [`chat_template_has_tool_round_trip`] -- needs both `tools` and `toolcalls`
/// variables, plus some way of rendering a tool-role message.
pub fn go_template_has_tool_round_trip(t: &Template) -> bool {
    let v = t.vars();
    if !v.iter().any(|x| x == "tools") || !v.iter().any(|x| x == "toolcalls") {
        return false;
    }
    let raw = t.raw();
    raw.contains(r#"eq .Role "tool""#)
        || raw.contains("tool_response")
        || raw.contains("TOOL_RESULTS")
}

/// **Upstream:** `goTemplateEnvSet()`.
///
/// Ask for the value with two different defaults; if they disagree, nobody set
/// the variable and you were just seeing the defaults. Slightly cryptic, but it
/// is upstream's way of distinguishing "explicitly off" from "not mentioned",
/// which [`Model::should_use_go_template`] genuinely needs.
fn go_template_env_set(env: &crate::envconfig::Env) -> bool {
    env.go_template(true) == env.go_template(false)
}

/// **Upstream:** `hasMoreCapabilities`. Purely a length comparison, which is
/// upstream's simplification -- it is not asking about a superset.
fn has_more_capabilities(candidate: &[Capability], current: &[Capability]) -> bool {
    candidate.len() > current.len()
}

/// **Upstream:** `sameCapabilities`. Set equality, order-insensitive.
fn same_capabilities(candidate: &[Capability], current: &[Capability]) -> bool {
    candidate.len() == current.len() && candidate.iter().all(|c| current.contains(c))
}

/// Should the GGUF's own chat template beat the Modelfile `TEMPLATE`?
///
/// **Upstream:** `shouldPreferChatTemplate`.
///
/// Two ways to say yes:
///
/// 1. **The chat template does strictly more.** Then prefer it -- unless the Go
///    template can do a full tool round trip and the chat template cannot, in
///    which case the extra capability is not worth losing working tool replies.
/// 2. **They do exactly the same things, both including tools**, but only the
///    chat template can complete the round trip.
///
/// Anything else: keep the Go template. The bias toward the Modelfile is
/// deliberate -- it is what the model's publisher explicitly wrote.
pub fn should_prefer_chat_template(
    chat_template: &str,
    chat_template_caps: &[Capability],
    go_template: Option<&Template>,
    go_template_caps: &[Capability],
) -> bool {
    let go_round_trip = go_template.is_some_and(go_template_has_tool_round_trip);
    if has_more_capabilities(chat_template_caps, go_template_caps) {
        return !go_round_trip || chat_template_has_tool_round_trip(chat_template);
    }
    if !same_capabilities(chat_template_caps, go_template_caps)
        || !chat_template_caps.contains(&Capability::Tools)
        || !go_template_caps.contains(&Capability::Tools)
    {
        return false;
    }
    chat_template_has_tool_round_trip(chat_template) && !go_round_trip
}

/// Read a model back out of the store.
///
/// **Upstream:** `GetModel(name)` in `server/images.go` -- the model-loading
/// half only. The registry/network half and the blob store itself live
/// elsewhere.
///
/// Walks the manifest's layers and unpacks each one by media type: weights and
/// projector and adapter layers become **paths** (never read -- they are
/// gigabytes), while template, system, params, messages and licence layers are
/// small enough to slurp.
///
/// `gguf` supplies what a GGUF decoder would have told us. Pass `None` and the
/// model still loads correctly; it simply reports
/// [`Model::has_chat_template`] as `false` and infers fewer capabilities. Pass
/// facts read from the **wrong** file and it will confidently infer nonsense --
/// see [`GgufFacts`].
///
/// **Not ported:** the deprecated `application/vnd.ollama.image.embed` layer,
/// which upstream only logs a warning about and otherwise ignores. Ignoring it
/// silently is the same observable behaviour.
pub fn get_model(
    store: &Store,
    name: &Name,
    gguf: Option<&GgufFacts>,
    env: &crate::envconfig::Env,
) -> Result<Model> {
    let manifest = store.read_manifest(name)?;
    let config = store.read_config(&manifest)?;

    let mut m = Model {
        name: name.to_string(),
        short_name: name.display_shortest(),
        digest: manifest.digest().to_string(),
        config,
        model_path: None,
        draft_path: None,
        parent_model: String::new(),
        has_chat_template: false,
        has_go_template: false,
        prefer_chat_template: false,
        adapter_paths: Vec::new(),
        projector_paths: Vec::new(),
        system: String::new(),
        license: Vec::new(),
        options: BTreeMap::new(),
        messages: Vec::new(),
        template: Template::default_template(),
    };

    for layer in &manifest.layers {
        let digest = layer.checked_digest()?;
        let filename = store.blob_path(&digest);

        match layer.media_type.as_str() {
            MEDIA_TYPE_MODEL => {
                m.model_path = Some(filename);
                m.parent_model.clone_from(&layer.from);
                if m.is_gguf() {
                    m.has_chat_template = gguf.is_some_and(|f| !f.chat_template.is_empty());
                }
            }
            MEDIA_TYPE_DRAFT => m.draft_path = Some(filename),
            MEDIA_TYPE_ADAPTER => m.adapter_paths.push(filename),
            MEDIA_TYPE_PROJECTOR => m.projector_paths.push(filename),
            MEDIA_TYPE_PROMPT | MEDIA_TYPE_TEMPLATE => {
                m.has_go_template = true;
                let bytes = store.read_blob(&digest)?;
                let src = String::from_utf8_lossy(&bytes).into_owned();
                m.template =
                    Template::parse(&src).map_err(|e| CreateError::BadTemplate(e.to_string()))?;
            }
            MEDIA_TYPE_SYSTEM => {
                let bytes = store.read_blob(&digest)?;
                m.system = String::from_utf8_lossy(&bytes).into_owned();
            }
            MEDIA_TYPE_PARAMS => {
                let bytes = store.read_blob(&digest)?;
                m.options =
                    serde_json::from_slice(&bytes).map_err(json_ctx("parse params layer"))?;
            }
            MEDIA_TYPE_MESSAGES => {
                let bytes = store.read_blob(&digest)?;
                m.messages =
                    serde_json::from_slice(&bytes).map_err(json_ctx("parse messages layer"))?;
            }
            MEDIA_TYPE_LICENSE => {
                let bytes = store.read_blob(&digest)?;
                m.license.push(String::from_utf8_lossy(&bytes).into_owned());
            }
            _ => {}
        }
    }

    // Decide whether the GGUF chat template should win. Upstream's guard list,
    // kept whole: only when the env var is unset, the model has BOTH templates,
    // no named renderer/parser is in play, and Harmony is not being used --
    // i.e. only when the choice is genuinely between the two templates.
    if let Some(f) = gguf {
        let uses_harmony = m.should_use_harmony();
        let gguf_caps = {
            let mut c = Vec::new();
            chat_template_capabilities(&mut c, &f.chat_template);
            c
        };
        let go_caps = go_template_capabilities(&m.template);
        if !go_template_env_set(env)
            && m.has_go_template
            && !f.chat_template.is_empty()
            && m.config.renderer.is_empty()
            && m.config.parser.is_empty()
            && !uses_harmony
            && should_prefer_chat_template(
                &f.chat_template,
                &gguf_caps,
                Some(&m.template),
                &go_caps,
            )
        {
            m.prefer_chat_template = true;
        }
    }

    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Sha256;
    use std::io::Write as _;

    // -------------------------------------------------------------------
    // Test helpers
    // -------------------------------------------------------------------

    /// The mock user database from upstream's `TestExpandPath`, homes and all.
    struct MockUsers;

    impl MockUsers {
        fn home(name: &str) -> Option<PathBuf> {
            // Upstream picks `D:/home/...` on Windows and `/home/...` elsewhere,
            // so that the expected paths are absolute on both.
            let root = if cfg!(windows) { "D:/home/" } else { "/home/" };
            match name {
                "testuser" | "anotheruser" => Some(PathBuf::from(go_clean(&format!("{root}{name}")))),
                _ => None,
            }
        }
    }

    impl UserLookup for MockUsers {
        fn current_home(&self) -> Result<PathBuf> {
            Self::home("testuser").ok_or(CreateError::NoCurrentUser("mock".into()))
        }
        fn home_of(&self, username: &str) -> Result<PathBuf> {
            Self::home(username)
                .ok_or_else(|| CreateError::UserLookupUnsupported(username.to_string()))
        }
    }

    fn write(dir: &Path, rel: &str, bytes: &[u8]) {
        let p = dir.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).expect("mkdir");
        }
        let mut f = fs::File::create(&p).expect("create");
        f.write_all(bytes).expect("write");
    }

    /// 512 bytes of `i % 256` -- upstream's own "definitely binary" filler, and
    /// it contains 0x00, so it sniffs as `application/octet-stream`.
    fn binary_filler() -> Vec<u8> {
        (0..512u32).map(|i| (i % 256) as u8).collect()
    }

    const ZIP_HEADER: &[u8] = &[0x50, 0x4B, 0x03, 0x04];

    /// Relative, forward-slashed, sorted -- so an assertion reads the same on
    /// Windows and Termux.
    fn rel_sorted(base: &Path, files: &[String]) -> Vec<String> {
        let base = base.to_string_lossy().into_owned();
        let mut v: Vec<String> = files
            .iter()
            .map(|f| go_rel(&base, f).unwrap_or_else(|| f.clone()).replace('\\', "/"))
            .collect();
        v.sort();
        v
    }

    // -------------------------------------------------------------------
    // Go filepath helpers
    // -------------------------------------------------------------------

    #[test]
    fn clean_cancels_inner_dotdot_and_drops_leading_dotdot_when_rooted() {
        assert_eq!(go_clean("a/b/../c"), go_join(&["a", "c"]));
        assert_eq!(go_clean("a/./b"), go_join(&["a", "b"]));
        assert_eq!(go_clean("a//b"), go_join(&["a", "b"]));
        assert_eq!(go_clean(""), ".");
        assert_eq!(go_clean("."), ".");
        // Not rooted: a leading `..` survives, because it is meaningful.
        assert_eq!(go_clean("../a"), go_join(&["..", "a"]));
        // Rooted: Go's rule 4 drops it -- you cannot climb above the root.
        let rooted = go_clean("/../a");
        assert!(!rooted.contains(".."), "got {rooted}");
    }

    #[test]
    fn is_local_rejects_every_way_out_of_a_directory() {
        assert!(is_local("a"));
        assert!(is_local(&go_join(&["a", "b"])));
        assert!(is_local(&go_join(&["a", "..", "b"])), "cancels inside");

        assert!(!is_local(""), "empty is not local");
        assert!(!is_local(".."), "the escape itself");
        assert!(!is_local("../a"), "leading dotdot");
        // The one that a naive check misses: no LEADING `..` until you clean it.
        assert!(!is_local("a/../../b"), "escape only visible after cleaning");
        assert!(!is_local("/etc/passwd"), "absolute");
    }

    #[cfg(windows)]
    #[test]
    fn is_local_rejects_windows_device_names() {
        assert!(!is_local("NUL"));
        assert!(!is_local("nul.txt"), "an extension does not save you");
        assert!(!is_local("COM1"));
        assert!(is_local("COM0"), "COM0 is not a device");
        assert!(is_local("CONFIG"), "only the exact stem is reserved");
        assert!(!is_local("C:/tmp"), "a volume is not local");
    }

    #[test]
    fn rel_finds_the_route_between_two_paths() {
        let base = go_abs("base").expect("abs");
        assert_eq!(go_rel(&base, &base).as_deref(), Some("."));
        assert_eq!(
            go_rel(&base, &go_join(&[&base, "a", "b"])).as_deref(),
            Some(go_join(&["a", "b"]).as_str())
        );
        // Escaping shows up as a leading `..`, which `is_local` then rejects.
        let sibling = go_join(&[&base, "..", "other", "x"]);
        let rel = go_rel(&base, &sibling).expect("some route");
        assert!(rel.starts_with(".."), "got {rel}");
        assert!(!is_local(&rel));
    }

    /// The fidelity point the whole module turns on: Go's `*` never crosses a
    /// separator, so `**` matches exactly what one `*` would.
    #[test]
    fn a_double_star_is_not_recursive_it_is_just_one_star() {
        assert!(go_match("**", "anything"));
        assert!(go_match("*", "anything"));
        assert!(go_match("model*.safetensors", "model-00001-of-00002.safetensors"));
        assert!(go_match("*.json", "config.json"));
        assert!(!go_match("*.json", "config.jsonl"));
        assert!(go_match("tokenizer.model", "tokenizer.model"));
        assert!(!go_match("model*.safetensors", "consolidated.safetensors"));
        assert!(go_match("consolidated*.pth", "consolidated.00.pth"));
        assert!(go_match("?.gguf", "a.gguf"));
        assert!(!go_match("?.gguf", "ab.gguf"));
    }

    #[test]
    fn glob_sorts_its_matches_and_treats_a_missing_directory_as_zero_matches() {
        let dir = tempfile::tempdir().expect("tempdir");
        for n in ["c.json", "a.json", "b.json"] {
            write(dir.path(), n, b"{}");
        }
        let base = dir.path().to_string_lossy().into_owned();
        let got = go_glob(&go_join(&[&base, "*.json"])).expect("glob");
        assert_eq!(rel_sorted(dir.path(), &got), ["a.json", "b.json", "c.json"]);

        let missing = go_glob(&go_join(&[&base, "nope", "*.json"])).expect("glob");
        assert!(missing.is_empty(), "a missing dir is not an error");
    }

    #[test]
    fn glob_refuses_a_character_class_rather_than_half_supporting_it() {
        let e = go_glob("a/[abc].json").expect_err("must refuse");
        assert!(matches!(e, CreateError::BadPattern(_)), "got {e:?}");
    }

    // -------------------------------------------------------------------
    // Content sniffing
    // -------------------------------------------------------------------

    #[test]
    fn content_sniffing_answers_the_three_types_files_for_model_compares_against() {
        assert_eq!(detect_content_type(ZIP_HEADER), "application/zip");
        assert_eq!(detect_content_type(b""), "text/plain; charset=utf-8");
        assert_eq!(
            detect_content_type(b"{\"config\": true}"),
            "text/plain; charset=utf-8"
        );
        assert_eq!(
            detect_content_type(&binary_filler()),
            "application/octet-stream"
        );
        // Leading whitespace is skipped before the text scan, as upstream does.
        assert_eq!(
            detect_content_type(b"   \n\t{\"a\":1}"),
            "text/plain; charset=utf-8"
        );
        assert_eq!(detect_content_type_bare(b"hello"), "text/plain");
    }

    /// The git-lfs pointer file this whole check exists for.
    #[test]
    fn an_unresolved_git_lfs_pointer_sniffs_as_text_not_as_weights() {
        let pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:4d4f\nsize 4831838448\n";
        assert_eq!(detect_content_type_bare(pointer), "text/plain");
        assert_ne!(detect_content_type_bare(pointer), "application/octet-stream");
    }

    /// Pins the byte-order trap documented on [`detect_ggml_content_type`]: a
    /// GGUF file literally starts `GGUF`, while an old ggml file starts `lmgg`.
    /// Anyone "fixing" the constants to match the ASCII breaks this test, which
    /// is exactly the point of writing it this way.
    #[test]
    fn ggml_magics_are_read_little_endian() {
        assert_eq!(detect_ggml_content_type(b"GGUF"), Some("gguf"));
        assert_eq!(detect_ggml_content_type(b"FUGG"), Some("gguf"), "big-endian writer");
        assert_eq!(detect_ggml_content_type(b"lmgg"), Some("ggml"));
        assert_eq!(detect_ggml_content_type(b"fmgg"), Some("ggmf"));
        assert_eq!(detect_ggml_content_type(b"tjgg"), Some("ggjt"));
        assert_eq!(detect_ggml_content_type(b"algg"), Some("ggla"));
        assert_eq!(
            detect_ggml_content_type(b"ggml"),
            None,
            "the ASCII spelling is NOT the on-disk magic"
        );
        assert_eq!(detect_ggml_content_type(b"nope"), None);
        assert_eq!(detect_ggml_content_type(b"GG"), None, "too short");
        // The blob-level sniffer prefers ggml's answer over the HTTP one.
        assert_eq!(detect_blob_content_type(b"GGUF\x03\x00\x00\x00"), "gguf");
        assert_eq!(detect_blob_content_type(&binary_filler()), "unknown");
    }

    // -------------------------------------------------------------------
    // expandPath -- ported from upstream's parser/expandpath_test.go
    // -------------------------------------------------------------------

    #[test]
    fn expand_path_resolves_tilde_absolute_and_relative_forms() {
        let pwd = std::env::current_dir().expect("cwd").to_string_lossy().into_owned();
        let home = MockUsers::home("testuser").expect("mock home");
        let other = MockUsers::home("anotheruser").expect("mock home");
        let home = home.to_string_lossy().into_owned();
        let other = other.to_string_lossy().into_owned();

        // Upstream's table, with the platform-specific absolute cases folded
        // together via go_join so one table covers Windows and Unix.
        let abs_input = if cfg!(windows) {
            r"D:\absolute\path\to\file"
        } else {
            "/absolute/path/to/file"
        };
        let cases: Vec<(&str, &str, String)> = vec![
            ("~", "", home.clone()),
            (
                "~/myfolder/myfile.txt",
                "",
                go_join(&[&home, "myfolder", "myfile.txt"]),
            ),
            (
                "~anotheruser/docs/file.txt",
                "",
                go_join(&[&other, "docs", "file.txt"]),
            ),
            ("relative/path/to/file", "", go_join(&[&pwd, "relative/path/to/file"])),
            (abs_input, "", go_clean(abs_input)),
            (abs_input, "someotherdir/", go_clean(abs_input)),
            (".", &pwd, pwd.clone()),
            (".", "", pwd.clone()),
            ("somefile", "somedir", go_join(&[&pwd, "somedir", "somefile"])),
        ];

        for (path, rel, want) in cases {
            let got = expand_path_with(path, rel, &MockUsers).expect(path);
            assert_eq!(got.to_string_lossy(), want, "expand_path({path:?}, {rel:?})");
        }
    }

    #[test]
    fn expand_path_errors_on_an_unknown_user() {
        let e = expand_path_with("~nonexistentuser/file.txt", "", &MockUsers)
            .expect_err("must fail");
        assert!(
            matches!(e, CreateError::UserLookupUnsupported(ref u) if u == "nonexistentuser"),
            "got {e:?}"
        );
    }

    /// Traversal attempts do not survive as `..` chains -- `go_abs` cleans them
    /// into a concrete path the caller can then check.
    #[test]
    fn expand_path_cleans_traversal_attempts_into_concrete_paths() {
        for attack in ["../../../../etc/passwd", "a/../../../../etc/passwd", "./../../x"] {
            let got = expand_path_with(attack, "base", &MockUsers).expect(attack);
            let s = got.to_string_lossy().into_owned();
            assert!(!s.contains(".."), "{attack} left a `..` behind: {s}");
            assert!(is_abs(&s), "{attack} did not become absolute: {s}");
        }
    }

    /// `~otheruser` on the real system: refused, never guessed. This is the
    /// documented platform limitation, asserted so it cannot quietly change into
    /// a guess later.
    #[test]
    fn the_default_user_lookup_refuses_other_users_rather_than_guessing() {
        let e = SystemUsers
            .home_of("definitely-not-a-real-user-9f3a")
            .expect_err("must refuse");
        assert!(matches!(e, CreateError::UserLookupUnsupported(_)), "got {e:?}");
    }

    // -------------------------------------------------------------------
    // filesForModel -- ported from upstream's TestFilesForModel
    // -------------------------------------------------------------------

    #[test]
    fn files_for_model_takes_safetensors_and_the_config_files_beside_them() {
        let dir = tempfile::tempdir().expect("tempdir");
        for n in [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "config.json",
            "tokenizer.json",
            "chat_template.jinja",
        ] {
            write(dir.path(), n, b"test content");
        }
        let got = files_for_model(&dir.path().to_string_lossy()).expect("files");
        assert_eq!(
            rel_sorted(dir.path(), &got),
            [
                "chat_template.jinja",
                "config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "tokenizer.json",
            ]
        );
    }

    #[test]
    fn files_for_model_takes_a_binary_tokenizer_model_alongside_tokenizer_json() {
        let dir = tempfile::tempdir().expect("tempdir");
        for n in [
            "model-00001-of-00001.safetensors",
            "config.json",
            "tokenizer.json",
        ] {
            write(dir.path(), n, b"test content");
        }
        write(dir.path(), "tokenizer.model", &binary_filler());
        let got = files_for_model(&dir.path().to_string_lossy()).expect("files");
        assert_eq!(
            rel_sorted(dir.path(), &got),
            [
                "config.json",
                "model-00001-of-00001.safetensors",
                "tokenizer.json",
                "tokenizer.model",
            ]
        );
    }

    /// The case that pins `**` to one level: sentence-transformers module
    /// weights live exactly one directory down, and nothing deeper is wanted.
    #[test]
    fn files_for_model_reaches_exactly_one_level_down_for_module_weights() {
        let dir = tempfile::tempdir().expect("tempdir");
        for n in [
            "model.safetensors",
            "config.json",
            "modules.json",
            "2_Dense/config.json",
            "2_Dense/model.safetensors",
            "3_Dense/config.json",
            "3_Dense/model.safetensors",
        ] {
            write(dir.path(), n, b"test content");
        }
        // Two levels down: must NOT be picked up. If `**` were ever read as
        // recursive, this file would appear and the assertion would fail.
        write(dir.path(), "2_Dense/nested/deep.json", b"{}");

        let got = files_for_model(&dir.path().to_string_lossy()).expect("files");
        assert_eq!(
            rel_sorted(dir.path(), &got),
            [
                "2_Dense/config.json",
                "2_Dense/model.safetensors",
                "3_Dense/config.json",
                "3_Dense/model.safetensors",
                "config.json",
                "model.safetensors",
                "modules.json",
            ]
        );
    }

    #[test]
    fn files_for_model_prefers_sharded_safetensors_over_consolidated_ones() {
        let dir = tempfile::tempdir().expect("tempdir");
        for n in [
            "model-00001-of-00001.safetensors",
            "consolidated.safetensors",
            "config.json",
        ] {
            write(dir.path(), n, b"test content");
        }
        let got = files_for_model(&dir.path().to_string_lossy()).expect("files");
        assert_eq!(
            rel_sorted(dir.path(), &got),
            ["config.json", "model-00001-of-00001.safetensors"],
            "consolidated must be excluded when model*.safetensors exists"
        );
    }

    #[test]
    fn files_for_model_falls_back_to_consolidated_safetensors() {
        let dir = tempfile::tempdir().expect("tempdir");
        for n in ["consolidated.safetensors", "config.json"] {
            write(dir.path(), n, b"test content");
        }
        let got = files_for_model(&dir.path().to_string_lossy()).expect("files");
        assert_eq!(
            rel_sorted(dir.path(), &got),
            ["config.json", "consolidated.safetensors"]
        );
    }

    #[test]
    fn files_for_model_takes_pytorch_bins_that_really_are_zips() {
        let dir = tempfile::tempdir().expect("tempdir");
        for n in [
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
        ] {
            write(dir.path(), n, ZIP_HEADER);
        }
        write(dir.path(), "config.json", br#"{"config": true}"#);
        let got = files_for_model(&dir.path().to_string_lossy()).expect("files");
        assert_eq!(
            rel_sorted(dir.path(), &got),
            [
                "config.json",
                "pytorch_model-00001-of-00002.bin",
                "pytorch_model-00002-of-00002.bin",
            ]
        );
    }

    #[test]
    fn files_for_model_takes_consolidated_pth_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        for n in ["consolidated.00.pth", "consolidated.01.pth"] {
            write(dir.path(), n, ZIP_HEADER);
        }
        write(dir.path(), "config.json", br#"{"config": true}"#);
        let got = files_for_model(&dir.path().to_string_lossy()).expect("files");
        assert_eq!(
            rel_sorted(dir.path(), &got),
            ["config.json", "consolidated.00.pth", "consolidated.01.pth"]
        );
    }

    #[test]
    fn files_for_model_takes_gguf_and_also_bin_files_that_are_really_gguf() {
        for name in ["model.gguf", "model.bin"] {
            let dir = tempfile::tempdir().expect("tempdir");
            write(dir.path(), name, &binary_filler());
            write(dir.path(), "config.json", br#"{"config": true}"#);
            let got = files_for_model(&dir.path().to_string_lossy()).expect("files");
            let mut want = vec!["config.json".to_string(), name.to_string()];
            want.sort();
            assert_eq!(rel_sorted(dir.path(), &got), want);
        }
    }

    #[test]
    fn files_for_model_errors_when_no_rung_of_the_ladder_matches() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "README.md", b"content");
        write(dir.path(), "config.json", b"content");
        let e = files_for_model(&dir.path().to_string_lossy()).expect_err("must fail");
        assert!(matches!(e, CreateError::ModelNotFound), "got {e:?}");
    }

    /// Upstream's "invalid content type for pytorch model" case: the `.bin` is
    /// plain text (an LFS pointer, in the real world), so the zip check misses,
    /// the `*.bin` rung's octet-stream check also misses, and the ladder falls
    /// off the end.
    #[test]
    fn files_for_model_rejects_a_pytorch_bin_that_is_actually_text() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "pytorch_model.bin", b"plain text content");
        write(dir.path(), "config.json", b"plain text content");
        let e = files_for_model(&dir.path().to_string_lossy()).expect_err("must fail");
        assert!(matches!(e, CreateError::ModelNotFound), "got {e:?}");
    }

    // -------------------------------------------------------------------
    // fileDigestMap
    // -------------------------------------------------------------------

    #[test]
    fn file_digest_map_hashes_a_single_file_to_its_content_address() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "model.gguf", b"abc");
        let p = dir.path().join("model.gguf").to_string_lossy().into_owned();

        let mut h = Sha256::new();
        let got = file_digest_map(&p, &mut h).expect("digest map");
        assert_eq!(got.len(), 1);
        // The FIPS 180-4 Appendix B.1 vector for "abc".
        assert_eq!(
            got.values().next().expect("one entry").as_str(),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn file_digest_map_hashes_every_file_of_a_directory_model() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "model.gguf", &binary_filler());
        write(dir.path(), "config.json", br#"{"config": true}"#);

        let mut h = Sha256::new();
        let got = file_digest_map(&dir.path().to_string_lossy(), &mut h).expect("digest map");
        assert_eq!(got.len(), 2);
        assert!(
            got.values().all(|d| d.as_str().starts_with("sha256:")),
            "every value is a content address"
        );
        // Two different files must not collide -- i.e. the hasher really did
        // reset between them.
        let uniq: std::collections::BTreeSet<_> = got.values().map(Digest::as_str).collect();
        assert_eq!(uniq.len(), 2, "the hasher was not reset between files");
    }

    /// The symlink escape the `.cache` hint exists for. Unix only: creating a
    /// symlink on Windows needs Developer Mode or elevation, so the test would
    /// fail for reasons that have nothing to do with the code.
    #[cfg(unix)]
    #[test]
    fn file_digest_map_refuses_a_symlink_that_escapes_into_a_cache() {
        let outside = tempfile::tempdir().expect("tempdir");
        let cache = outside.path().join(".cache").join("huggingface");
        fs::create_dir_all(&cache).expect("mkdir");
        let real = cache.join("weights.gguf");
        fs::write(&real, binary_filler()).expect("write");

        let dir = tempfile::tempdir().expect("tempdir");
        std::os::unix::fs::symlink(&real, dir.path().join("model.gguf")).expect("symlink");

        let mut h = Sha256::new();
        let e = file_digest_map(&dir.path().to_string_lossy(), &mut h).expect_err("must refuse");
        assert!(
            matches!(e, CreateError::InsecureCachePath(_)),
            "expected the .cache hint, got {e:?}"
        );
        assert!(
            e.to_string().contains("--local-dir"),
            "the error must name the flag that fixes it: {e}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn file_digest_map_refuses_a_symlink_that_escapes_anywhere_else() {
        let outside = tempfile::tempdir().expect("tempdir");
        let real = outside.path().join("weights.gguf");
        fs::write(&real, binary_filler()).expect("write");

        let dir = tempfile::tempdir().expect("tempdir");
        std::os::unix::fs::symlink(&real, dir.path().join("model.gguf")).expect("symlink");

        let mut h = Sha256::new();
        let e = file_digest_map(&dir.path().to_string_lossy(), &mut h).expect_err("must refuse");
        assert!(matches!(e, CreateError::InsecurePath(_)), "got {e:?}");
    }

    // -------------------------------------------------------------------
    // format_params
    // -------------------------------------------------------------------

    fn params_of(pairs: &[(&str, &str)]) -> BTreeMap<String, Vec<String>> {
        let mut m: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (k, v) in pairs {
            m.entry((*k).to_string()).or_default().push((*v).to_string());
        }
        m
    }

    #[test]
    fn format_params_coerces_text_into_the_json_types_the_blob_stores() {
        let got = format_params(&params_of(&[
            ("temperature", "0.2"),
            ("top_k", "40"),
            ("use_mmap", "true"),
            ("stop", "<|im_end|>"),
        ]))
        .expect("format");

        assert!(got["temperature"].is_number(), "not the string \"0.2\"");
        assert_eq!(got["top_k"], Value::from(40i64));
        assert_eq!(got["use_mmap"], Value::from(true));
        assert_eq!(got["stop"], Value::from(vec![Value::from("<|im_end|>")]));
    }

    #[test]
    fn format_params_rejects_an_unknown_name_and_a_malformed_value() {
        let e = format_params(&params_of(&[("temprature", "0.2")])).expect_err("typo");
        assert!(
            matches!(e, CreateError::UnknownParameter(ref k) if k == "temprature"),
            "a Modelfile typo must be loud, got {e:?}"
        );

        let e = format_params(&params_of(&[("top_k", "lots")])).expect_err("bad int");
        assert!(matches!(e, CreateError::BadParameterValue { kind: "int", .. }), "got {e:?}");

        // Go's ParseBool does NOT accept these, and neither do we.
        for v in ["yes", "no", "on", "off"] {
            assert!(
                format_params(&params_of(&[("use_mmap", v)])).is_err(),
                "strconv.ParseBool rejects {v:?}"
            );
        }
        assert!(format_params(&params_of(&[("use_mmap", "T")])).is_ok());
    }

    /// The guard against [`PARAM_KINDS`] drifting away from the `Options` struct
    /// it mirrors. Rust has no reflection, so this is what replaces Go's
    /// `reflect.VisibleFields`.
    #[test]
    fn every_parameter_name_is_one_the_options_struct_knows() {
        for (name, kind) in PARAM_KINDS {
            let probe = match kind {
                ParamKind::Int => Value::from(1i64),
                ParamKind::Float => Value::from(1.0f64),
                ParamKind::Bool => Value::from(true),
                ParamKind::StringSlice => Value::from(vec![Value::from("x")]),
            };
            let mut map = serde_json::Map::new();
            map.insert((*name).to_string(), probe);

            let mut opts = crate::options::Options::default();
            let unknown = opts
                .apply_map(&map)
                .unwrap_or_else(|e| panic!("{name} has the wrong type in PARAM_KINDS: {e}"));
            assert!(
                unknown.is_empty(),
                "PARAM_KINDS lists {name:?}, but Options::apply_map does not know it"
            );
        }
    }

    // -------------------------------------------------------------------
    // Go-compatible JSON encoding -- the bit that decides digests
    // -------------------------------------------------------------------

    /// The single most consequential fidelity detail in this module: Go
    /// HTML-escapes `<` and `>`, and essentially every stop token contains both.
    #[test]
    fn json_encoding_html_escapes_stop_tokens_exactly_as_go_does() {
        let mut params: BTreeMap<String, Value> = BTreeMap::new();
        params.insert(
            "stop".to_string(),
            Value::from(vec![Value::from("<|im_end|>")]),
        );
        let bytes = go_json_encode(&params).expect("encode");
        let s = String::from_utf8(bytes).expect("utf8");

        assert_eq!(s, "{\"stop\":[\"\\u003c|im_end|\\u003e\"]}\n");
        assert!(!s.contains('<'), "a raw `<` means a different digest to ollama");
        assert!(s.ends_with('\n'), "json.Encoder.Encode appends a newline");
    }

    #[test]
    fn json_encoding_sorts_keys_and_escapes_ampersands_and_line_separators() {
        let mut m: BTreeMap<String, Value> = BTreeMap::new();
        m.insert("z".into(), Value::from("a&b"));
        m.insert("a".into(), Value::from("x\u{2028}y"));
        let s = String::from_utf8(go_json_encode(&m).expect("encode")).expect("utf8");
        assert_eq!(s, "{\"a\":\"x\\u2028y\",\"z\":\"a\\u0026b\"}\n");
    }

    // -------------------------------------------------------------------
    // create_request -- ported from upstream's parser_test.go
    // -------------------------------------------------------------------

    #[test]
    fn create_request_hashes_a_local_from_and_leaves_a_model_name_alone() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "model.gguf", &binary_filler());
        let mut h = Sha256::new();

        let mf = Modelfile::parse("FROM ./model.gguf\n").expect("parse");
        let req = create_request(&mf, &dir.path().to_string_lossy(), &mut h).expect("request");
        assert_eq!(req.files.len(), 1, "the local file was hashed");
        assert!(req.from.is_empty(), "a local path is not a model name");

        let mf = Modelfile::parse("FROM qwen3:0.6b\n").expect("parse");
        let req = create_request(&mf, &dir.path().to_string_lossy(), &mut h).expect("request");
        assert!(req.files.is_empty());
        assert_eq!(req.from, "qwen3:0.6b", "a missing path is a model name");
    }

    #[test]
    fn create_request_carries_template_system_licence_messages_and_parameters() {
        let mut h = Sha256::new();
        let mf = Modelfile::parse(
            "FROM base\n\
             TEMPLATE \"\"\"{{ .Prompt }}\"\"\"\n\
             SYSTEM you are a kopitiam uncle\n\
             LICENSE MIT\n\
             LICENSE Apache-2.0\n\
             PARAMETER temperature 0.2\n\
             PARAMETER stop <|im_end|>\n\
             PARAMETER stop <|endoftext|>\n\
             MESSAGE user kopi o kosong\n",
        )
        .expect("parse");

        let req = create_request(&mf, "", &mut h).expect("request");
        assert_eq!(req.from, "base");
        assert_eq!(req.template, "{{ .Prompt }}");
        assert_eq!(req.system, "you are a kopitiam uncle");
        assert_eq!(
            req.license,
            Some(LicenseSpec::Many(vec!["MIT".into(), "Apache-2.0".into()]))
        );
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content, "kopi o kosong");
        // Repeated `stop` accumulates; everything else is last-one-wins.
        assert_eq!(
            req.parameters["stop"],
            Value::from(vec![Value::from("<|im_end|>"), Value::from("<|endoftext|>")])
        );
        assert!(req.parameters["temperature"].is_number());
    }

    /// Upstream's `TestCreateRequestDraftFiles`.
    #[test]
    fn create_request_hashes_a_draft_model_separately_from_the_base() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "draft.gguf", &binary_filler());
        let mut h = Sha256::new();

        let mf = Modelfile::parse("FROM base\nDRAFT ./draft.gguf\n").expect("parse");
        let req = create_request(&mf, &dir.path().to_string_lossy(), &mut h).expect("request");
        assert_eq!(req.draft_files.len(), 1);
        assert!(req.files.is_empty(), "the base was a model name, not files");
    }

    /// Upstream's `TestCreateRequestDraftRejectsSameFile`.
    #[test]
    fn create_request_refuses_a_draft_that_is_the_same_file_as_from() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "model.gguf", &binary_filler());
        let mut h = Sha256::new();

        let mf = Modelfile::parse("FROM ./model.gguf\nDRAFT ./model.gguf\n").expect("parse");
        let e = create_request(&mf, &dir.path().to_string_lossy(), &mut h).expect_err("must refuse");
        assert!(
            e.to_string()
                .contains("DRAFT must not reference the same local path as FROM"),
            "got {e}"
        );
    }

    /// Upstream's `TestCreateRequestDraftRejectsSameDirectory` -- the same trap
    /// spelled as a directory rather than a file.
    #[test]
    fn create_request_refuses_a_draft_that_is_the_same_directory_as_from() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "model.gguf", &binary_filler());
        let mut h = Sha256::new();

        let mf = Modelfile::parse("FROM .\nDRAFT .\n").expect("parse");
        let e = create_request(&mf, &dir.path().to_string_lossy(), &mut h).expect_err("must refuse");
        assert!(
            e.to_string()
                .contains("DRAFT must not reference the same local path as FROM"),
            "got {e}"
        );
    }

    #[test]
    fn create_request_validates_requires_as_semver_and_strips_the_v() {
        let mut h = Sha256::new();
        for (input, want) in [("0.14.0", "0.14.0"), ("v1.2", "1.2"), ("2", "2")] {
            let mf = Modelfile::parse(&format!("FROM base\nREQUIRES {input}\n")).expect("parse");
            let req = create_request(&mf, "", &mut h).expect(input);
            assert_eq!(req.requires, want, "REQUIRES {input}");
        }
        for bad in ["not-a-version", "1.2.3.4", "01.2"] {
            let mf = Modelfile::parse(&format!("FROM base\nREQUIRES {bad}\n")).expect("parse");
            assert!(
                matches!(create_request(&mf, "", &mut h), Err(CreateError::BadRequires)),
                "REQUIRES {bad} should be refused"
            );
        }
    }

    #[test]
    fn create_request_drops_deprecated_parameters_instead_of_failing_on_them() {
        let mut h = Sha256::new();
        let mf = Modelfile::parse("FROM base\nPARAMETER mirostat 1\nPARAMETER top_k 40\n")
            .expect("parse");
        let req = create_request(&mf, "", &mut h).expect("request");
        assert!(!req.parameters.contains_key("mirostat"), "deprecated, dropped");
        assert_eq!(req.parameters["top_k"], Value::from(40i64));
    }

    // -------------------------------------------------------------------
    // create_model + get_model, round trip through a real store
    // -------------------------------------------------------------------

    fn test_name() -> Name {
        Name::parse("registry.ollama.ai/library/kopi:latest")
    }

    #[test]
    fn create_model_writes_every_side_car_layer_and_a_config_that_covers_them() {
        let root = tempfile::tempdir().expect("tempdir");
        let store = Store::new(root.path());
        let mut h = Sha256::new();
        let name = test_name();

        let mut req = CreateRequest {
            template: "{{ .Prompt }}".into(),
            system: "you are helpful".into(),
            license: Some(LicenseSpec::Many(vec!["MIT".into()])),
            messages: vec![Message::new("user", "hi")],
            ..Default::default()
        };
        req.parameters
            .insert("temperature".into(), Value::from(0.2f64));

        let config = ConfigV2 {
            model_format: "gguf".into(),
            ..Default::default()
        };
        create_model(&store, &name, Vec::new(), &config, &req, &mut h).expect("create");

        let manifest = store.read_manifest(&name).expect("read back");
        let types: Vec<&str> = manifest.layers.iter().map(|l| l.media_type.as_str()).collect();
        for want in [
            MEDIA_TYPE_TEMPLATE,
            MEDIA_TYPE_SYSTEM,
            MEDIA_TYPE_LICENSE,
            MEDIA_TYPE_PARAMS,
            MEDIA_TYPE_MESSAGES,
        ] {
            assert!(types.contains(&want), "missing {want} in {types:?}");
        }

        // The config's diff_ids must cover every layer, in order -- that is what
        // makes the config digest change whenever any layer does.
        let cfg = store.read_config(&manifest).expect("config");
        let digests: Vec<String> = manifest.layers.iter().map(|l| l.digest.clone()).collect();
        assert_eq!(cfg.rootfs.diff_ids, digests);
    }

    #[test]
    fn get_model_reads_back_exactly_what_create_model_wrote() {
        let root = tempfile::tempdir().expect("tempdir");
        let store = Store::new(root.path());
        let mut h = Sha256::new();
        let name = test_name();
        let env = crate::envconfig::Env::new(BTreeMap::new());

        let req = CreateRequest {
            template: "{{ .Prompt }}".into(),
            system: "kopitiam uncle".into(),
            license: Some(LicenseSpec::One("MIT".into())),
            messages: vec![Message::new("user", "kopi c peng")],
            ..Default::default()
        };
        create_model(&store, &name, Vec::new(), &ConfigV2::default(), &req, &mut h)
            .expect("create");

        let m = get_model(&store, &name, None, &env).expect("get");
        assert_eq!(m.system, "kopitiam uncle");
        assert_eq!(m.license, ["MIT"]);
        assert_eq!(m.messages.len(), 1);
        assert_eq!(m.messages[0].content, "kopi c peng");
        assert!(m.has_go_template);
        assert_eq!(m.template.raw(), "{{ .Prompt }}");
        assert_eq!(m.short_name, "kopi:latest");
    }

    #[test]
    fn set_system_with_an_empty_string_clears_an_inherited_prompt() {
        let root = tempfile::tempdir().expect("tempdir");
        let store = Store::new(root.path());
        let mut h = Sha256::new();
        let mut layers = vec![store
            .new_layer(b"inherited", MEDIA_TYPE_SYSTEM, &mut h)
            .expect("layer")];

        set_system(&store, &mut layers, "", &mut h).expect("clear");
        assert!(layers.is_empty(), "an empty SYSTEM removes the layer");
    }

    #[test]
    fn set_parameters_lets_a_child_override_per_key_and_inherit_the_rest() {
        let root = tempfile::tempdir().expect("tempdir");
        let store = Store::new(root.path());
        let mut h = Sha256::new();

        let parent: BTreeMap<String, Value> = [
            ("temperature".to_string(), Value::from(0.9f64)),
            ("top_k".to_string(), Value::from(10i64)),
        ]
        .into_iter()
        .collect();
        let bytes = go_json_encode(&parent).expect("encode");
        let mut layers = vec![store
            .new_layer(&bytes, MEDIA_TYPE_PARAMS, &mut h)
            .expect("layer")];

        let child: BTreeMap<String, Value> =
            [("temperature".to_string(), Value::from(0.2f64))].into_iter().collect();
        set_parameters(&store, &mut layers, &child, &mut h).expect("merge");

        let layer = layers
            .iter()
            .find(|l| l.media_type == MEDIA_TYPE_PARAMS)
            .expect("params layer");
        let digest = layer.checked_digest().expect("digest");
        let merged: BTreeMap<String, Value> =
            serde_json::from_slice(&store.read_blob(&digest).expect("blob")).expect("json");

        assert_eq!(merged["temperature"], Value::from(0.2f64), "child wins");
        assert_eq!(merged["top_k"], Value::from(10i64), "parent's key survives");
    }

    #[test]
    fn set_template_refuses_a_template_that_will_not_compile() {
        let root = tempfile::tempdir().expect("tempdir");
        let store = Store::new(root.path());
        let mut h = Sha256::new();
        let mut layers = Vec::new();
        let e = set_template(&store, &mut layers, "{{ .Prompt ", &mut h).expect_err("must refuse");
        assert!(matches!(e, CreateError::BadTemplate(_)), "got {e:?}");
        assert!(layers.is_empty(), "a bad template must not be stored");
    }

    // -------------------------------------------------------------------
    // Architecture defaults
    // -------------------------------------------------------------------

    #[test]
    fn architecture_defaults_fill_gaps_but_never_override_the_modelfile() {
        let mut config = ConfigV2::default();
        let mut params = BTreeMap::new();
        apply_architecture_defaults(&mut config, &mut params, "gemma4");
        assert_eq!(config.renderer, GEMMA4_RENDERER_LEGACY);
        assert_eq!(config.parser, "gemma4");
        assert_eq!(params["stop"], Value::from(vec![Value::from("<turn|>")]));

        // An explicit RENDERER wins, and an explicit stop is not clobbered.
        let mut config = ConfigV2 {
            renderer: "mine".into(),
            ..Default::default()
        };
        let mut params: BTreeMap<String, Value> =
            [("stop".to_string(), Value::from(vec![Value::from("X")]))]
                .into_iter()
                .collect();
        apply_architecture_defaults(&mut config, &mut params, "gemma4");
        assert_eq!(config.renderer, "mine");
        assert_eq!(config.parser, "gemma4", "the empty half still gets filled");
        assert_eq!(params["stop"], Value::from(vec![Value::from("X")]));

        let mut config = ConfigV2::default();
        let mut params = BTreeMap::new();
        apply_architecture_defaults(&mut config, &mut params, "nemotron_h_moe");
        assert_eq!(config.renderer, "nemotron-3-nano");
        assert_eq!(config.parser, "nemotron-3-nano");

        let mut config = ConfigV2::default();
        let mut params = BTreeMap::new();
        apply_architecture_defaults(&mut config, &mut params, "llama");
        assert!(config.renderer.is_empty(), "an unknown arch changes nothing");
    }

    // -------------------------------------------------------------------
    // Renderer resolution
    // -------------------------------------------------------------------

    fn model_with(config: ConfigV2, name: &str, short: &str) -> Model {
        Model {
            name: name.to_string(),
            short_name: short.to_string(),
            digest: String::new(),
            config,
            model_path: None,
            draft_path: None,
            parent_model: String::new(),
            has_chat_template: false,
            has_go_template: false,
            prefer_chat_template: false,
            adapter_paths: Vec::new(),
            projector_paths: Vec::new(),
            system: String::new(),
            license: Vec::new(),
            options: BTreeMap::new(),
            messages: Vec::new(),
            template: Template::default_template(),
        }
    }

    fn gemma4_model(name: &str, short: &str, model_type: &str) -> Model {
        model_with(
            ConfigV2 {
                renderer: GEMMA4_RENDERER_LEGACY.into(),
                model_type: model_type.into(),
                ..Default::default()
            },
            name,
            short,
        )
    }

    #[test]
    fn the_renderer_name_passes_through_unless_it_is_the_legacy_gemma4_one() {
        let m = model_with(
            ConfigV2 {
                renderer: "qwen3-coder".into(),
                ..Default::default()
            },
            "x",
            "x",
        );
        assert_eq!(resolve_renderer_name(&m), "qwen3-coder");

        let m = model_with(ConfigV2::default(), "x", "x");
        assert_eq!(resolve_renderer_name(&m), "", "empty stays empty");
    }

    #[test]
    fn gemma4_narrows_by_name_first_then_by_parameter_count_then_defaults_small() {
        // 1. short name wins
        let m = gemma4_model("registry/library/gemma4:12b", "gemma4:e4b", "12B");
        assert_eq!(
            resolve_renderer_name(&m),
            GEMMA4_RENDERER_SMALL,
            "the short name is trusted over the metadata"
        );
        // 2. full name, when the short one says nothing
        let m = gemma4_model("registry/library/gemma4:26b", "gemma4:latest", "");
        assert_eq!(resolve_renderer_name(&m), GEMMA4_RENDERER_LARGE);
        // 3. parameter count
        let m = gemma4_model("gemma4", "gemma4", "12B");
        assert_eq!(resolve_renderer_name(&m), GEMMA4_RENDERER_LARGE);
        let m = gemma4_model("gemma4", "gemma4", "8B");
        assert_eq!(resolve_renderer_name(&m), GEMMA4_RENDERER_SMALL);
        // 4. nothing known -> small, upstream's conservative default
        let m = gemma4_model("gemma4", "gemma4", "");
        assert_eq!(resolve_renderer_name(&m), GEMMA4_RENDERER_SMALL);
        // Substring, not suffix.
        let m = gemma4_model("gemma4", "gemma4:12b-instruct-q4_K_M", "");
        assert_eq!(resolve_renderer_name(&m), GEMMA4_RENDERER_LARGE);
    }

    #[test]
    fn human_parameter_counts_are_decimal_si_not_binary() {
        assert_eq!(parse_human_parameter_count("12B"), Some(12_000_000_000));
        assert_eq!(parse_human_parameter_count("1B"), Some(1_000_000_000));
        assert_eq!(parse_human_parameter_count("600M"), Some(600_000_000));
        assert_eq!(parse_human_parameter_count("1.5K"), Some(1_500));
        assert_eq!(parse_human_parameter_count("7.6b"), Some(7_600_000_000));
        assert_eq!(parse_human_parameter_count(""), None);
        assert_eq!(parse_human_parameter_count("12"), None, "the unit is required");
        assert_eq!(parse_human_parameter_count("XB"), None);
        // The threshold is inclusive, per `>=`.
        assert_eq!(
            gemma4_renderer_for_parameter_count(GEMMA4_LARGE_MIN_PARAMETER_COUNT),
            GEMMA4_RENDERER_LARGE
        );
        assert_eq!(
            gemma4_renderer_for_parameter_count(GEMMA4_LARGE_MIN_PARAMETER_COUNT - 1),
            GEMMA4_RENDERER_SMALL
        );
    }

    #[test]
    fn is_gemma4_renderer_covers_the_legacy_name_too() {
        assert!(is_gemma4_renderer(GEMMA4_RENDERER_LEGACY));
        assert!(is_gemma4_renderer(GEMMA4_RENDERER_SMALL));
        assert!(is_gemma4_renderer(GEMMA4_RENDERER_LARGE));
        assert!(!is_gemma4_renderer("qwen3-coder"));
        assert!(!is_gemma4_renderer(""));
    }

    // -------------------------------------------------------------------
    // Capability inference
    // -------------------------------------------------------------------

    fn env() -> crate::envconfig::Env {
        crate::envconfig::Env::new(BTreeMap::new())
    }

    #[test]
    fn a_gguf_without_a_pooling_type_does_completion_and_one_with_it_embeds() {
        let mut m = model_with(ConfigV2::default(), "x", "x");
        m.model_path = Some(PathBuf::from("weights.gguf"));

        let caps = m.capabilities(Some(&GgufFacts::default()), &[], &env());
        assert!(caps.contains(&Capability::Completion));
        assert!(!caps.contains(&Capability::Embedding));

        let facts = GgufFacts {
            has_pooling_type: true,
            ..Default::default()
        };
        let caps = m.capabilities(Some(&facts), &[], &env());
        assert!(caps.contains(&Capability::Embedding));
        assert!(!caps.contains(&Capability::Completion));
    }

    #[test]
    fn vision_and_audio_come_from_the_gguf_block_counts() {
        let mut m = model_with(ConfigV2::default(), "x", "x");
        m.model_path = Some(PathBuf::from("weights.gguf"));
        let facts = GgufFacts {
            has_vision_block_count: true,
            has_audio_block_count: true,
            ..Default::default()
        };
        let caps = m.capabilities(Some(&facts), &[], &env());
        assert!(caps.contains(&Capability::Vision));
        assert!(caps.contains(&Capability::Audio));
    }

    #[test]
    fn a_chat_template_that_mentions_tools_or_think_tags_grants_those_capabilities() {
        let mut m = model_with(ConfigV2::default(), "x", "x");
        m.model_path = Some(PathBuf::from("weights.gguf"));
        let facts = GgufFacts {
            chat_template: "{% if tools %}...{% endif %}<think></think>".into(),
            ..Default::default()
        };
        let caps = m.capabilities(Some(&facts), &[], &env());
        assert!(caps.contains(&Capability::Tools));
        assert!(caps.contains(&Capability::Thinking));
    }

    #[test]
    fn the_thinking_heuristic_ignores_templates_that_use_a_different_mechanism() {
        assert!(chat_template_has_thinking_support("<think>x</think>"));
        assert!(chat_template_has_thinking_support("content.split('</think>')"));
        assert!(
            !chat_template_has_thinking_support("content.split('</think>') reasoning_content"),
            "reasoning_content means a different mechanism"
        );
        assert!(
            !chat_template_has_thinking_support("content.split('</think>') <SPECIAL_12>"),
            "<SPECIAL_12> means a different mechanism"
        );
        assert!(!chat_template_has_thinking_support("no tags here"));
    }

    #[test]
    fn a_tool_round_trip_needs_both_the_call_and_a_way_to_render_the_reply() {
        assert!(!chat_template_has_tool_round_trip("tools only"));
        assert!(
            !chat_template_has_tool_round_trip("tool_calls but no reply shape"),
            "emitting a call is not enough"
        );
        assert!(chat_template_has_tool_round_trip("tool_calls ... tool_response"));
        assert!(chat_template_has_tool_round_trip("tool_calls ... ipython"));
        for spelling in [
            "role'] == 'tool'",
            "role'] == \"tool\"",
            "role\"] == 'tool'",
            "role\"] == \"tool\"",
            "message.role == 'tool'",
            "message.role == \"tool\"",
        ] {
            assert!(
                chat_template_has_tool_round_trip(&format!("tool_calls {spelling}")),
                "{spelling} must count"
            );
        }
    }

    #[test]
    fn a_go_template_grants_tools_from_its_variables_and_insert_from_suffix() {
        let mut m = model_with(ConfigV2::default(), "x", "x");
        m.has_go_template = true;
        m.template = Template::parse("{{ .Tools }}{{ .Suffix }}{{ .Response }}").expect("parse");
        let caps = m.capabilities(None, &[], &env());
        assert!(caps.contains(&Capability::Tools));
        assert!(caps.contains(&Capability::Insert));
    }

    #[test]
    fn a_projector_grants_vision_and_only_grants_audio_when_it_really_has_one() {
        let m = model_with(ConfigV2::default(), "x", "x");

        let caps = m.capabilities(None, &[ProjectorFacts::default()], &env());
        assert!(caps.contains(&Capability::Vision));
        assert!(!caps.contains(&Capability::Audio));

        let audio = ProjectorFacts {
            has_audio_encoder: true,
            ..Default::default()
        };
        assert!(m.capabilities(None, &[audio], &env()).contains(&Capability::Audio));

        // gemma3nv carries an encoder that does not work through this path.
        let suppressed = ProjectorFacts {
            has_audio_encoder: true,
            vision_projector_type: "gemma3nv".into(),
        };
        assert!(
            !m.capabilities(None, &[suppressed], &env()).contains(&Capability::Audio),
            "a gemma3nv projector must not advertise audio"
        );
    }

    #[test]
    fn gpt_oss_always_thinks_whatever_its_template_says() {
        for family in ["gptoss", "gpt-oss"] {
            let m = model_with(
                ConfigV2 {
                    model_family: family.into(),
                    ..Default::default()
                },
                "x",
                "x",
            );
            assert!(
                m.capabilities(None, &[], &env()).contains(&Capability::Thinking),
                "{family} must think"
            );
        }
    }

    #[test]
    fn the_config_blob_can_declare_capabilities_outright() {
        let m = model_with(
            ConfigV2 {
                capabilities: vec!["tools".into(), "vision".into(), "nonsense".into()],
                ..Default::default()
            },
            "x",
            "x",
        );
        let caps = m.capabilities(None, &[], &env());
        assert!(caps.contains(&Capability::Tools));
        assert!(caps.contains(&Capability::Vision));
        assert_eq!(caps.len(), 2, "an unrecognised name is dropped, not an error");
    }

    /// The subtraction stage, which must run last -- both of these capabilities
    /// were added by an earlier stage and are then taken away.
    #[test]
    fn the_final_filter_removes_capabilities_that_are_known_broken() {
        // nemotron_h_omni: audio suppressed pending llama.cpp support.
        let mut m = model_with(
            ConfigV2 {
                model_family: "nemotron_h_omni".into(),
                ..Default::default()
            },
            "x",
            "x",
        );
        m.model_path = Some(PathBuf::from("w.gguf"));
        let facts = GgufFacts {
            has_audio_block_count: true,
            ..Default::default()
        };
        assert!(
            !m.capabilities(Some(&facts), &[], &env()).contains(&Capability::Audio),
            "nemotron_h_omni audio must be suppressed"
        );

        // Gemma 4 on safetensors: no vision on the MLX path.
        let mut m = model_with(
            ConfigV2 {
                renderer: GEMMA4_RENDERER_LEGACY.into(),
                model_format: "safetensors".into(),
                ..Default::default()
            },
            "x",
            "x",
        );
        m.model_path = Some(PathBuf::from("w.safetensors"));
        let caps = m.capabilities(None, &[ProjectorFacts::default()], &env());
        assert!(!caps.contains(&Capability::Vision), "got {caps:?}");
    }

    #[test]
    fn check_capabilities_names_what_is_missing_and_advises_a_repull_for_qwen3() {
        let m = model_with(ConfigV2::default(), "x", "x");
        assert!(m.check_capabilities(&[], None, &[], &env()).is_ok());

        let e = m
            .check_capabilities(&[Capability::Vision], None, &[], &env())
            .expect_err("missing");
        assert!(e.contains("does not support vision"), "got {e}");

        let m = model_with(
            ConfigV2 {
                model_family: "qwen3".into(),
                ..Default::default()
            },
            "x",
            "x",
        );
        let e = m
            .check_capabilities(&[Capability::Thinking], None, &[], &env())
            .expect_err("missing");
        assert!(e.contains("Pull the model again"), "got {e}");
    }

    #[test]
    fn prefer_the_chat_template_only_when_it_does_more_without_losing_tool_replies() {
        let go = Template::parse("{{ .Tools }}{{ .Response }}").expect("parse");

        // Chat template does strictly more, and the Go one cannot round trip.
        assert!(should_prefer_chat_template(
            "tools <think></think>",
            &[Capability::Tools, Capability::Thinking],
            Some(&go),
            &[Capability::Tools],
        ));

        // Same capabilities, neither does a round trip -> keep the Modelfile's.
        assert!(!should_prefer_chat_template(
            "tools",
            &[Capability::Tools],
            Some(&go),
            &[Capability::Tools],
        ));

        // Same capabilities, only the chat template completes the round trip.
        assert!(should_prefer_chat_template(
            "tools tool_calls tool_response",
            &[Capability::Tools],
            Some(&go),
            &[Capability::Tools],
        ));

        // Fewer capabilities -> never prefer it.
        assert!(!should_prefer_chat_template(
            "tools",
            &[Capability::Tools],
            Some(&go),
            &[Capability::Tools, Capability::Thinking],
        ));
    }

    // -------------------------------------------------------------------
    // Split GGUF filenames
    // -------------------------------------------------------------------

    #[test]
    fn split_gguf_names_are_five_digits_and_the_index_is_shifted_to_zero_based() {
        assert_eq!(
            split_gguf_name("qwen3-00001-of-00003.gguf"),
            Some(("qwen3".to_string(), 0, 3)),
            "filename 00001 is metadata split.no 0"
        );
        assert_eq!(
            split_gguf_name("/blobs/qwen3-00003-of-00003.gguf"),
            Some(("qwen3".to_string(), 2, 3)),
            "matched against the basename"
        );
        // Exactly five digits, exactly as the upstream regex demands.
        assert_eq!(split_gguf_name("m-1-of-3.gguf"), None);
        assert_eq!(split_gguf_name("m-000001-of-000003.gguf"), None);
        // Zero in either position is not-a-split-file, not shard zero.
        assert_eq!(split_gguf_name("m-00000-of-00003.gguf"), None);
        assert_eq!(split_gguf_name("m-00001-of-00000.gguf"), None);
        assert_eq!(split_gguf_name("model.gguf"), None);
        assert_eq!(split_gguf_name("m-00001-of-00003.bin"), None);
    }
}
