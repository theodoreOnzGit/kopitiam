//! Model names -- the grammar behind `qwen3:0.6b`.
//!
//! **Upstream:** `types/model/name.go` (ollama, MIT). This is a close
//! translation; every function below names its Go counterpart.
//!
//! ## What a name actually is
//!
//! One string, four parts, three of which usually stay invisible:
//!
//! ```text
//!     registry.ollama.ai / library / qwen3 : 0.6b
//!     └──── host ──────┘  └─ ns ─┘  └mdl┘  └tag┘
//! ```
//!
//! When you type `qwen3`, ollama fills the other three from [`Name::default_name`]
//! -- host `registry.ollama.ai`, namespace `library`, tag `latest`. So `qwen3`,
//! `library/qwen3`, and `registry.ollama.ai/library/qwen3:latest` are all the
//! same model. That defaulting is the whole reason this module exist: the name
//! the human types and the name the store keys on must be the *same* name, or
//! you end up downloading one model twice under two spellings.
//!
//! ## The parse is separator-driven, not regex-driven
//!
//! Upstream parse right-to-left off the separators, and the ordering trick is
//! worth understanding because it is what make `host:port/ns/model` work:
//!
//! * `/` is illegal inside a tag, so **a `:` that sit after the last `/` must be
//!   the tag separator**, and a `:` before it must be a port. That single
//!   comparison (`last(':') > last('/')`) is the entire disambiguation.
//! * Then cut `/` twice from the right: last segment is the model, the one
//!   before it the namespace, whatever is left is the host.
//!
//! ## "Promised but missing" -- why [`MISSING_PART`] exists
//!
//! If you write `/model` or `model:`, you *promised* a part by typing its
//! separator and then never gave one. Upstream does NOT error there; it fills
//! the hole with the sentinel [`MISSING_PART`] (`"!MISSING!"`), which is
//! deliberately not a valid part, so [`Name::is_valid`] reports false and the
//! bad name shows up legibly in a log instead of as a confusing empty string.
//! We keep that behaviour exactly -- a parse that cannot fail is much easier to
//! call, and validity is a separate question you ask with [`Name::is_valid`].
//!
//! ## Where we deliberately differ from Go
//!
//! * `Name.Filepath()` **panics** upstream when the name is not fully qualified.
//!   Ours returns [`Option<PathBuf>`] instead ([`Name::filepath`]) -- a panic as
//!   a validation channel is not idiomatic Rust, and this path is reachable from
//!   user input.
//! * `ParseNameFromFilepath` upstream splits on the **OS** separator. KOPITIAM
//!   run on Windows *and* Termux, and a manifest tree written on one gets read
//!   on the other, so ours accept **both** `/` and `\`. Cross-platform paths are
//!   a standing rule here, not a nicety.
//! * `LogValue()` / `BaseURL()` are not ported: the first is `log/slog` glue with
//!   no Rust equivalent, the second is one `format!` at the single call site in
//!   the registry client.

use std::fmt;
use std::path::{Path, PathBuf};

/// Sentinel for a name part that was *promised* by a separator but never given.
///
/// **Upstream:** `MissingPart` in `types/model/name.go`.
///
/// Upstream picked this value because it is (a) unlikely to be typed by a human,
/// (b) not a valid part under [`Name::is_valid`], and (c) easy to spot in a log.
/// All three still hold here, so we keep the exact string -- a name that round
/// trips through ollama and KOPITIAM must read the same in both logs.
pub const MISSING_PART: &str = "!MISSING!";

/// Default registry host. **Upstream:** `defaultHost`.
pub const DEFAULT_HOST: &str = "registry.ollama.ai";
/// Default namespace -- ollama's own curated models. **Upstream:** `defaultNamespace`.
pub const DEFAULT_NAMESPACE: &str = "library";
/// Default tag. **Upstream:** `defaultTag`.
pub const DEFAULT_TAG: &str = "latest";
/// Default protocol scheme. **Upstream:** `defaultProtocolScheme`.
pub const DEFAULT_PROTOCOL_SCHEME: &str = "https";

/// Which part of a name we validating -- the rules differ per part.
///
/// **Upstream:** `partKind` in `types/model/name.go`.
///
/// The variant ORDER matters: [`Name::is_fully_qualified`] walk host, namespace,
/// model, tag in exactly this sequence, same as upstream's
/// `parts[i] -> partKind(i)` index trick. Don't reshuffle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PartKind {
    Host,
    Namespace,
    Model,
    Tag,
    /// Not produced by name parsing -- kept because upstream's `isValidPart`
    /// branches on it (a digest is the one other part allowed to contain `:`,
    /// as in `sha256:abcd...`).
    #[allow(dead_code)]
    Digest,
}

/// A parsed model name. **Upstream:** `type Name struct`.
///
/// **A `Name` is not guaranteed valid.** Parsing never fails -- see the module
/// docs on [`MISSING_PART`] -- so ask [`Name::is_valid`] before you trust one.
/// When invalid, treat the field values as undefined, exactly like upstream says.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Name {
    pub host: String,
    pub namespace: String,
    pub model: String,
    pub tag: String,
    pub protocol_scheme: String,
}

impl Name {
    /// A name carrying only the defaults; model and tag-less otherwise.
    ///
    /// **Upstream:** `DefaultName()`.
    pub fn default_name() -> Self {
        Self {
            host: DEFAULT_HOST.to_string(),
            namespace: DEFAULT_NAMESPACE.to_string(),
            model: String::new(),
            tag: DEFAULT_TAG.to_string(),
            protocol_scheme: DEFAULT_PROTOCOL_SCHEME.to_string(),
        }
    }

    /// Parse a name and fill in whatever the user left out. **This is the one
    /// you want** for anything typed by a human.
    ///
    /// **Upstream:** `ParseName(s)` == `Merge(ParseNameBare(s), DefaultName())`.
    ///
    /// ```
    /// # use kopitiam_ollama::Name;
    /// let n = Name::parse("qwen3");
    /// assert_eq!(n.to_string(), "registry.ollama.ai/library/qwen3:latest");
    /// assert_eq!(n.display_shortest(), "qwen3:latest");
    /// ```
    pub fn parse(s: &str) -> Self {
        Self::parse_bare(s).merge(&Self::default_name())
    }

    /// Parse a name and fill in **nothing**. Missing parts stay empty.
    ///
    /// **Upstream:** `ParseNameBare(s)`.
    ///
    /// Use this when the caller's own defaults matter more than ollama's -- e.g.
    /// a KOPITIAM-local registry where `library` is the wrong namespace to
    /// assume. Otherwise use [`Name::parse`].
    pub fn parse_bare(s: &str) -> Self {
        let mut n = Name::default();
        let mut s = s;

        // `/` is an illegal tag character, so a `:` sitting AFTER the last `/`
        // can only be the tag separator -- and one sitting before it can only be
        // a port. `rfind` gives None for absent, and `None < Some(_)` under
        // Option's derived Ord, which is exactly the ordering Go gets from its
        // `-1` sentinel. Same comparison, no sentinel needed.
        if s.rfind(':') > s.rfind('/') {
            let (before, after, _) = cut_promised(s, ':');
            s = before;
            n.tag = after.to_string();
        }

        let (before, after, promised) = cut_promised(s, '/');
        if !promised {
            n.model = before.to_string();
            return n;
        }
        n.model = after.to_string();
        s = before;

        let (before, after, promised) = cut_promised(s, '/');
        if !promised {
            n.namespace = before.to_string();
            return n;
        }
        n.namespace = after.to_string();
        s = before;

        // Whatever is left is the host, optionally scheme-prefixed. Upstream
        // uses `strings.Cut(s, "://")`: on no match the WHOLE string is the
        // host and the scheme stays empty.
        match s.split_once("://") {
            Some((scheme, host)) => {
                n.protocol_scheme = scheme.to_string();
                n.host = host.to_string();
            }
            None => n.host = s.to_string(),
        }

        n
    }

    /// Parse a 4-segment store path (`{host}/{namespace}/{model}/{tag}`) back
    /// into a name. Returns `None` unless it is exactly 4 segments AND fully
    /// qualified.
    ///
    /// **Upstream:** `ParseNameFromFilepath(s)` (which returns a zero `Name`
    /// where we return `None` -- same meaning, better typed).
    ///
    /// This is the inverse of [`Name::filepath`], and it is how the manifest
    /// store enumerate what models are installed: walk the tree, turn each leaf
    /// path back into a name.
    ///
    /// **Cross-platform:** upstream split on the OS separator only. We accept
    /// both `/` and `\`, because a manifest tree synced from Windows must still
    /// enumerate on Termux (and the reverse).
    pub fn parse_from_filepath(s: impl AsRef<Path>) -> Option<Self> {
        let s = s.as_ref().to_string_lossy();
        let parts: Vec<&str> = s.split(['/', '\\']).collect();
        if parts.len() != 4 {
            return None;
        }

        let n = Name {
            host: parts[0].to_string(),
            namespace: parts[1].to_string(),
            model: parts[2].to_string(),
            tag: parts[3].to_string(),
            protocol_scheme: String::new(),
        };

        n.is_fully_qualified().then_some(n)
    }

    /// Fill this name's empty parts from `other`. Non-empty parts of `self` win.
    ///
    /// **Upstream:** `Merge(a, b)` -- `cmp.Or` picks the first non-zero value,
    /// which for a `string` means the first non-empty one.
    ///
    /// Note what it does NOT merge: `model`. A default namespace makes sense; a
    /// default *model* does not, so upstream leaves it alone and so do we.
    #[must_use]
    pub fn merge(mut self, other: &Name) -> Self {
        if self.host.is_empty() {
            self.host = other.host.clone();
        }
        if self.namespace.is_empty() {
            self.namespace = other.namespace.clone();
        }
        if self.tag.is_empty() {
            self.tag = other.tag.clone();
        }
        if self.protocol_scheme.is_empty() {
            self.protocol_scheme = other.protocol_scheme.clone();
        }
        self
    }

    /// The shortest spelling that still identify the model unambiguously.
    ///
    /// **Upstream:** `DisplayShortest()`.
    ///
    /// Drop the host when it is the default one; drop the namespace too when
    /// *both* host and namespace are default. Model and tag always stay -- so
    /// the output is never bare enough to be re-parsed into something else.
    /// This is what you print in a `list` table.
    pub fn display_shortest(&self) -> String {
        let mut sb = String::new();
        if !self.host.eq_ignore_ascii_case(DEFAULT_HOST) {
            sb.push_str(&self.host);
            sb.push('/');
            sb.push_str(&self.namespace);
            sb.push('/');
        } else if !self.namespace.eq_ignore_ascii_case(DEFAULT_NAMESPACE) {
            sb.push_str(&self.namespace);
            sb.push('/');
        }
        sb.push_str(&self.model);
        sb.push(':');
        sb.push_str(&self.tag);
        sb
    }

    /// `{namespace}/{model}` -- **upstream:** `DisplayNamespaceModel()`.
    pub fn display_namespace_model(&self) -> String {
        format!("{}/{}", self.namespace, self.model)
    }

    /// Is every part present and well-formed?
    ///
    /// **Upstream:** `IsValid()`, which today just forwards to
    /// `IsFullyQualified()` -- the digest check was removed upstream and is
    /// planned to come back. Kept as a separate method so that when upstream
    /// restores it, we have the same seam to restore it into.
    pub fn is_valid(&self) -> bool {
        self.is_fully_qualified()
    }

    /// Are host, namespace, model and tag all present and well-formed?
    ///
    /// **Upstream:** `IsFullyQualified()`.
    pub fn is_fully_qualified(&self) -> bool {
        is_valid_part(PartKind::Host, &self.host)
            && is_valid_part(PartKind::Namespace, &self.namespace)
            && is_valid_part(PartKind::Model, &self.model)
            && is_valid_part(PartKind::Tag, &self.tag)
    }

    /// The canonical on-disk path for this name: `{host}/{namespace}/{model}/{tag}`.
    ///
    /// **Upstream:** `Filepath()`, which **panics** on an unqualified name. We
    /// return `None` instead -- see the module docs.
    ///
    /// Every part is validated first, and that validation is load-bearing for
    /// **safety**, not just tidiness: [`is_valid_part`] forbid `/`, `\`, `.` as a
    /// leading char, and everything else outside `[A-Za-z0-9_.-]`, so a name can
    /// never contain `..` and can never escape the store root. A name straight
    /// off the network is untrusted input; this is the gate.
    pub fn filepath(&self) -> Option<PathBuf> {
        if !self.is_fully_qualified() {
            return None;
        }
        let mut p = PathBuf::from(&self.host);
        p.push(&self.namespace);
        p.push(&self.model);
        p.push(&self.tag);
        Some(p)
    }

    /// ASCII case-insensitive equality across all four parts.
    ///
    /// **Upstream:** `EqualFold(o)`. Registries treat names case-insensitively,
    /// so `Qwen3` and `qwen3` are the same model -- but note this is ASCII-only
    /// folding, matching Go's behaviour for these grammars (parts are restricted
    /// to ASCII by [`is_valid_part`] anyway, so full Unicode folding would be
    /// reach for nothing).
    pub fn equal_fold(&self, o: &Name) -> bool {
        self.host.eq_ignore_ascii_case(&o.host)
            && self.namespace.eq_ignore_ascii_case(&o.namespace)
            && self.model.eq_ignore_ascii_case(&o.model)
            && self.tag.eq_ignore_ascii_case(&o.tag)
    }

    /// The registry base URL, `{scheme}://{host}`. **Upstream:** `BaseURL()`.
    pub fn base_url(&self) -> String {
        format!("{}://{}", self.protocol_scheme, self.host)
    }
}

impl fmt::Display for Name {
    /// **Upstream:** `String()`.
    ///
    /// Emits only the parts that are non-empty, so a bare `Name::parse_bare`
    /// result print back as what was typed. Heads-up on an upstream doc/code
    /// mismatch we chose to keep: the Go doc comment claims it returns `""` for
    /// an invalid name, but the code does no such check. Ollama is the oracle,
    /// so we match the **code**, not the comment.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.host.is_empty() {
            write!(f, "{}/", self.host)?;
        }
        if !self.namespace.is_empty() {
            write!(f, "{}/", self.namespace)?;
        }
        f.write_str(&self.model)?;
        if !self.tag.is_empty() {
            write!(f, ":{}", self.tag)?;
        }
        Ok(())
    }
}

/// Error type reserved for a future strict parse. Nothing return it today --
/// upstream's parse is infallible by design -- but the name is exported so
/// downstream `use` lines don't churn when a strict variant lands.
#[derive(Debug, thiserror::Error)]
#[error("unqualified name: {0}")]
pub struct ParseNameFromFilepathError(pub String);

/// Length bounds per part. **Upstream:** `isValidLen`.
///
/// Host gets 350 because a registry host can carry a long DNS name plus a port;
/// everything else gets 80. These are upstream's numbers, not ours -- a name we
/// accept but ollama rejects would produce a store entry ollama cannot read.
fn is_valid_len(kind: PartKind, s: &str) -> bool {
    let max = match kind {
        PartKind::Host => 350,
        _ => 80,
    };
    // Byte length, same as Go's `len(string)`. Parts are ASCII-only anyway (the
    // char check below rejects everything else), so bytes == chars here.
    (1..=max).contains(&s.len())
}

/// Is `s` a well-formed part of the given kind? **Upstream:** `isValidPart`.
///
/// The grammar, stated plainly:
/// * first byte must be alphanumeric or `_` -- so no leading `-`, `.` or `:`,
///   which is what stops a part from ever being `..` or a flag-lookalike;
/// * after that, alphanumeric, `_` and `-` are always fine;
/// * `.` is fine **except in a namespace** (upstream reserves it there);
/// * `:` is fine **only in a host** (the port) **or a digest** (`sha256:...`).
///
/// Everything else -- `/`, `\`, spaces, non-ASCII -- is rejected, which is what
/// makes [`Name::filepath`] safe against traversal.
fn is_valid_part(kind: PartKind, s: &str) -> bool {
    if !is_valid_len(kind, s) {
        return false;
    }
    for (i, c) in s.bytes().enumerate() {
        if i == 0 {
            if !is_alphanumeric_or_underscore(c) {
                return false;
            }
            continue;
        }
        match c {
            b'_' | b'-' => {}
            b'.' => {
                if kind == PartKind::Namespace {
                    return false;
                }
            }
            b':' => {
                if kind != PartKind::Host && kind != PartKind::Digest {
                    return false;
                }
            }
            _ => {
                if !is_alphanumeric_or_underscore(c) {
                    return false;
                }
            }
        }
    }
    true
}

/// **Upstream:** `isAlphanumericOrUnderscore`. ASCII only, deliberately.
fn is_alphanumeric_or_underscore(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Cut `s` at the **last** `sep`, substituting [`MISSING_PART`] for either side
/// that comes back empty.
///
/// **Upstream:** `cutPromised` (which wraps `cutLast`).
///
/// The substitution is the whole point: an empty side means the caller typed a
/// separator and then nothing, i.e. promised a part and didn't deliver. Turning
/// that into a sentinel rather than `""` is what makes `Name::is_valid` able to
/// tell "you never gave a namespace" apart from "you gave a broken one".
fn cut_promised(s: &str, sep: char) -> (&str, &str, bool) {
    match s.rfind(sep) {
        Some(i) => {
            let before = &s[..i];
            let after = &s[i + sep.len_utf8()..];
            (
                if before.is_empty() { MISSING_PART } else { before },
                if after.is_empty() { MISSING_PART } else { after },
                true,
            )
        }
        None => (s, "", false),
    }
}

/// Is `s` a well-formed namespace? **Upstream:** `IsValidNamespace(s)`.
pub fn is_valid_namespace(s: &str) -> bool {
    is_valid_part(PartKind::Namespace, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `part80` / `part350` fixtures from `name_test.go` -- the exact
    /// boundary lengths, so an off-by-one in `is_valid_len` cannot slip past.
    const PART80: &str = "88888888888888888888888888888888888888888888888888888888888888888888888888888888";

    fn part350() -> String {
        "3".repeat(350)
    }

    fn name(host: &str, ns: &str, model: &str, tag: &str) -> Name {
        Name {
            host: host.into(),
            namespace: ns.into(),
            model: model.into(),
            tag: tag.into(),
            protocol_scheme: String::new(),
        }
    }

    /// The upstream `TestParseNameParts` table, ported case for case.
    #[test]
    fn parse_bare_matches_the_upstream_table() {
        let p350 = part350();
        let long_in = format!("{PART80}/{PART80}/{PART80}:{PART80}");
        let long350_in = format!("{p350}/{PART80}/{PART80}:{PART80}");

        let cases: Vec<(&str, Name)> = vec![
            (
                "registry.ollama.ai/library/dolphin-mistral:7b-v2.6-dpo-laser-q6_K",
                name(
                    "registry.ollama.ai",
                    "library",
                    "dolphin-mistral",
                    "7b-v2.6-dpo-laser-q6_K",
                ),
            ),
            ("host/namespace/model:tag", name("host", "namespace", "model", "tag")),
            (
                "host:port/namespace/model:tag",
                name("host:port", "namespace", "model", "tag"),
            ),
            ("host/namespace/model", name("host", "namespace", "model", "")),
            ("host:port/namespace/model", name("host:port", "namespace", "model", "")),
            ("namespace/model", name("", "namespace", "model", "")),
            ("model", name("", "", "model", "")),
            ("h/nn/mm:t", name("h", "nn", "mm", "t")),
            (&long_in, name(PART80, PART80, PART80, PART80)),
            (&long350_in, name(&p350, PART80, PART80, PART80)),
        ];

        for (input, want) in cases {
            assert_eq!(Name::parse_bare(input), want, "parse_bare({input:?})");
        }
    }

    /// The scheme case is separate because it is the only one that populate
    /// `protocol_scheme`, and getting `://` handling wrong is an easy slip.
    #[test]
    fn a_scheme_prefix_is_split_off_the_host() {
        let got = Name::parse_bare("scheme://host:port/namespace/model:tag");
        assert_eq!(
            got,
            Name {
                host: "host:port".into(),
                namespace: "namespace".into(),
                model: "model".into(),
                tag: "tag".into(),
                protocol_scheme: "scheme".into(),
            }
        );
    }

    /// The disambiguation that make `host:port` work at all: a `:` BEFORE the
    /// last `/` is a port, a `:` AFTER it is a tag. Both in one string proves
    /// the rule rather than the accident.
    #[test]
    fn a_colon_before_the_last_slash_is_a_port_not_a_tag() {
        let n = Name::parse_bare("host:5000/ns/model:v1");
        assert_eq!(n.host, "host:5000");
        assert_eq!(n.tag, "v1");

        // No tag at all: the `:` is still the port, and tag stays empty.
        let n = Name::parse_bare("host:5000/ns/model");
        assert_eq!(n.host, "host:5000");
        assert_eq!(n.tag, "");
    }

    #[test]
    fn parse_fills_in_the_defaults() {
        let n = Name::parse("model");
        assert_eq!(n.host, DEFAULT_HOST);
        assert_eq!(n.namespace, DEFAULT_NAMESPACE);
        assert_eq!(n.tag, DEFAULT_TAG);
        assert_eq!(n.protocol_scheme, DEFAULT_PROTOCOL_SCHEME);
        assert!(n.is_valid());
    }

    /// The three spellings that must all mean the same model. If this ever
    /// break, the store will happily hold the same weights twice.
    #[test]
    fn short_and_long_spellings_resolve_to_one_name() {
        let a = Name::parse("qwen3");
        let b = Name::parse("library/qwen3");
        let c = Name::parse("registry.ollama.ai/library/qwen3:latest");
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a.filepath(), c.filepath());
    }

    /// A promised-but-missing part must land on the sentinel and fail validity
    /// -- NOT quietly become an empty string that then gets defaulted away.
    #[test]
    fn a_promised_but_missing_part_becomes_the_sentinel_and_is_invalid() {
        let n = Name::parse_bare("/model");
        assert_eq!(n.namespace, MISSING_PART);
        assert!(!n.is_valid());

        let n = Name::parse_bare("model:");
        assert_eq!(n.tag, MISSING_PART);
        assert!(!n.is_valid());

        // And the sentinel must survive the default-merge: a hole is not the
        // same as an omission, so `parse` must not paper over it.
        let n = Name::parse("model:");
        assert_eq!(n.tag, MISSING_PART);
        assert!(!n.is_valid());
    }

    #[test]
    fn display_round_trips_through_parse() {
        for s in [
            "registry.ollama.ai/library/qwen3:latest",
            "host:5000/ns/model:tag",
            "namespace/model",
            "model",
        ] {
            let n = Name::parse_bare(s);
            assert_eq!(n.to_string(), s, "round trip of {s:?}");
        }
    }

    #[test]
    fn display_shortest_drops_only_the_default_parts() {
        assert_eq!(Name::parse("qwen3").display_shortest(), "qwen3:latest");
        assert_eq!(
            Name::parse("myns/qwen3:0.6b").display_shortest(),
            "myns/qwen3:0.6b"
        );
        assert_eq!(
            Name::parse("myhost/myns/qwen3:0.6b").display_shortest(),
            "myhost/myns/qwen3:0.6b"
        );
        // Non-default host forces the namespace back in even when the namespace
        // itself is the default one -- otherwise `myhost/qwen3` would read as a
        // namespace, not a host.
        assert_eq!(
            Name::parse("myhost/library/qwen3:0.6b").display_shortest(),
            "myhost/library/qwen3:0.6b"
        );
    }

    /// `filepath` and `parse_from_filepath` are inverses **over the four stored
    /// parts only** -- the protocol scheme is not on disk, so it cannot come
    /// back. Upstream has the same gap (`ParseNameFromFilepath` never sets
    /// `ProtocolScheme`), and it is harmless because the scheme is a transport
    /// detail the store has no business remembering. Asserted explicitly so
    /// nobody later "fixes" it by inventing a scheme out of the path.
    #[test]
    fn filepath_and_parse_from_filepath_round_trip_the_four_stored_parts() {
        let n = Name::parse("host/ns/model:tag");
        let p = n.filepath().expect("fully qualified");

        let back = Name::parse_from_filepath(&p).expect("4 valid segments");
        assert_eq!((&back.host, &back.namespace, &back.model, &back.tag),
                   (&n.host, &n.namespace, &n.model, &n.tag));
        assert_eq!(back.protocol_scheme, "", "the scheme is not stored on disk");

        // Against a bare parse (no defaults, so no scheme either) it IS a
        // straight equality.
        let bare = Name::parse_bare("host/ns/model:tag");
        assert_eq!(Name::parse_from_filepath(&p), Some(bare));
    }

    /// Cross-platform: a store tree written on Windows must enumerate on Termux
    /// and vice versa. This is a deliberate divergence from upstream, so it gets
    /// its own test rather than riding along on the round-trip one.
    #[test]
    fn parse_from_filepath_accepts_both_separators() {
        let want = Some(name("host", "ns", "model", "tag"));
        assert_eq!(Name::parse_from_filepath("host/ns/model/tag"), want);
        assert_eq!(Name::parse_from_filepath("host\\ns\\model\\tag"), want);
        // Wrong arity is rejected, not silently padded.
        assert_eq!(Name::parse_from_filepath("host/ns/model"), None);
        assert_eq!(Name::parse_from_filepath("a/host/ns/model/tag"), None);
    }

    /// The security property behind `filepath`: no name that survives validation
    /// can escape the store root.
    #[test]
    fn traversal_and_separator_smuggling_never_produce_a_filepath() {
        for s in [
            "../../etc/passwd",
            "host/../model",
            "host/ns/..:tag",
            "host/ns/model:../..",
        ] {
            let n = Name::parse(s);
            assert!(
                n.filepath().is_none(),
                "{s:?} must not yield a filepath, got {:?}",
                n.filepath()
            );
        }
    }

    #[test]
    fn part_grammar_matches_upstream_rules() {
        // A namespace may not contain `.`; a model and tag may.
        assert!(!is_valid_part(PartKind::Namespace, "my.ns"));
        assert!(is_valid_part(PartKind::Model, "my.model"));
        assert!(is_valid_part(PartKind::Tag, "7b-v2.6-q6_K"));

        // Only a host (or digest) may contain `:`.
        assert!(is_valid_part(PartKind::Host, "host:5000"));
        assert!(!is_valid_part(PartKind::Model, "mo:del"));
        assert!(is_valid_part(PartKind::Digest, "sha256:abc"));

        // Leading char must be alphanumeric or `_`.
        for bad in ["-lead", ".lead", ":lead"] {
            assert!(!is_valid_part(PartKind::Model, bad), "{bad:?}");
        }
        assert!(is_valid_part(PartKind::Model, "_lead"));

        // Length bounds, both edges.
        assert!(!is_valid_part(PartKind::Model, ""));
        assert!(is_valid_part(PartKind::Model, PART80));
        assert!(!is_valid_part(PartKind::Model, &"8".repeat(81)));
        assert!(is_valid_part(PartKind::Host, &part350()));
        assert!(!is_valid_part(PartKind::Host, &"3".repeat(351)));
    }

    #[test]
    fn equal_fold_is_case_insensitive_across_every_part() {
        let a = Name::parse("Registry.Ollama.AI/Library/QWEN3:Latest");
        let b = Name::parse("registry.ollama.ai/library/qwen3:latest");
        assert!(a.equal_fold(&b));
        assert_ne!(a, b, "equal_fold must not imply Eq");
    }

    #[test]
    fn base_url_uses_the_parsed_scheme() {
        assert_eq!(
            Name::parse("qwen3").base_url(),
            "https://registry.ollama.ai"
        );
        assert_eq!(
            Name::parse_bare("http://localhost:11434/ns/m:t").base_url(),
            "http://localhost:11434"
        );
    }
}
