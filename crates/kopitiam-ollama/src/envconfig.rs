//! Runtime knobs read out of the environment -- host, models dir, keep-alive,
//! parallelism, context length, and friends.
//!
//! **Upstream:** `envconfig/config.go` (+ `envconfig/config_test.go`) from
//! ollama (Go, MIT, Copyright (c) Ollama), pinned at
//! `4713800b08b2ddf5e14acf8398953cf7b12f169b` (2026-07-28).
//!
//! ## What this module actually is
//!
//! Every number in here decide runtime behaviour -- how long a model sit in
//! RAM, how many requests run at once, what port we bind, how big the KV cache
//! is. So the defaults are not decoration, they *are* the content, and each one
//! name the upstream line it came from. CLAUDE.md's rule -- *a number with no
//! source is a bug that has not fired yet* -- is enforced field by field below.
//!
//! The other half of the content is the **parsing quirks**. Upstream's helpers
//! have edge cases that look like bugs until you realise they are load-bearing,
//! and every one of them is pinned by a test ported from `config_test.go`:
//!
//! * an **invalid** boolean (`OLLAMA_DEBUG=random`) reads as **`true`**, not as
//!   an error and not as `false` -- see [`Env::bool_var_with_default`];
//! * an **invalid** unsigned int falls back to the default *silently*, and
//!   "invalid" includes `-1`, `0x10` and `0o10`, because Go parse base 10 only;
//! * `OLLAMA_KEEP_ALIVE` accept **either** a Go duration (`5m`, `1h2m3s`) **or**
//!   a bare integer meaning *seconds*, and anything **negative** means
//!   *forever*, while plain `0` mean *unload straight away*. Two very different
//!   things, one env var -- see [`Expiry`];
//! * `OLLAMA_LOAD_TIMEOUT` looks identical but is **not**: there `0` also mean
//!   forever (`<= 0`, not `< 0`). Copy-pasting one into the other is exactly
//!   how you get a load that never gives up;
//! * `OLLAMA_ORIGINS` **always append** the localhost / file / editor origins,
//!   whatever you set. Setting it never *removes* the built-ins, only add to
//!   them. And the split is on `,` with **no trimming**, so `a, b` give you
//!   `"a"` and `" b"`.
//!
//! ## KOPITIAM divergence: we do not read another product's env vars
//!
//! Upstream hardcode `os.Getenv("OLLAMA_...")` everywhere. KOPITIAM is not
//! ollama, so reading `OLLAMA_*` by default would be quietly picking up
//! somebody else's configuration. Two deliberate changes, both here:
//!
//! 1. **The prefix is data, not code.** [`Env`] carry a list of prefixes;
//!    [`Env::from_env`] use [`DEFAULT_PREFIX`] (`KOPITIAM_`). Want byte-exact
//!    upstream behaviour? [`Env::ollama`]. Want both, KOPITIAM winning?
//!    `Env::from_env().with_prefixes(["KOPITIAM_", "OLLAMA_"])`. The
//!    **unprefixed** vars (`HTTP_PROXY`, `CUDA_VISIBLE_DEVICES`,
//!    `LLAMA_ARG_FIT`, ...) stay unprefixed -- they belong to other tools, not
//!    to us -- and are reached through [`Env::raw`].
//! 2. **The core is pure.** Every accessor read from an owned
//!    `BTreeMap<String, String>`, never from `std::env` directly, so the whole
//!    module is testable with no process environment and no `unsafe` set-env
//!    races between parallel tests. [`Env::from_env`] is the one thin wrapper
//!    that snapshot the real environment.
//!
//! ## What is deliberately NOT ported
//!
//! * `AsMap()` / `Values()` -- upstream build a `map[string]EnvVar` whose value
//!   column is formatted with Go's `%v` verb (a `time.Duration` print as
//!   `"5m0s"`, a `[]string` as `"[a b c]"`). That is a **display** concern that
//!   belong to a UI layer, and reproducing Go's verb would be inventing
//!   formatting nobody asked for. The *knowledge* in that table -- what each
//!   var means -- is preserved as [`VAR_DOCS`]; the rendering is the caller's.
//! * The `sync.RWMutex` cache around `~/.ollama/server.json`. Go need it
//!   because `NoCloud()` is a package-level function called from anywhere; we
//!   hand the parsed [`ServerConfig`] in as an argument, so the caller own the
//!   caching and there is no global to race on.
//! * `os.UserHomeDir()`'s **panic**. Upstream `Models()` panic when there is no
//!   home directory. We return [`EnvConfigError::NoHomeDir`] instead -- on
//!   Termux/Android and in stripped containers a missing `HOME` is a normal
//!   Tuesday, not a reason to abort the process.

use std::collections::BTreeMap;
use std::fmt;
use std::net::IpAddr;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// The prefix [`Env::from_env`] use. **Divergence from upstream**, which
/// hardcode `OLLAMA_`: see the module docs for why KOPITIAM must not read
/// another product's environment by default.
pub const DEFAULT_PREFIX: &str = "KOPITIAM_";

/// The dot-directory [`Env::from_env`] look for under `$HOME`. Upstream use
/// `.ollama`; [`Env::ollama`] restore that.
pub const DEFAULT_HOME_DIR_NAME: &str = ".kopitiam";

/// Things that can go wrong. Notice how short this list is -- almost every
/// upstream helper *swallow* a parse failure and fall back to a default, so a
/// bad value is never an error, only a shrug. The one thing we genuinely cannot
/// paper over is "where does this machine keep its home directory".
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EnvConfigError {
    /// No home directory could be worked out from the environment.
    ///
    /// **Divergence:** upstream `Models()` call `os.UserHomeDir()` and `panic`
    /// on failure. A library has no business killing the process over a missing
    /// env var, and on Termux/Android or in a scratch container this really do
    /// happen -- so we hand the problem back to the caller, who can point
    /// `<PREFIX>MODELS` somewhere explicit.
    #[error("cannot work out a home directory: none of [{0}] is set -- set one of them, or set the models directory explicitly")]
    NoHomeDir(String),
}

// ---------------------------------------------------------------------------
// Expiry -- Go's `time.Duration` with an "infinite" sentinel
// ---------------------------------------------------------------------------

/// A duration that might mean *never expire*.
///
/// **Upstream:** the return type of `KeepAlive()` and `LoadTimeout()`, both of
/// which are `time.Duration` (an `int64` of nanoseconds) using
/// `time.Duration(math.MaxInt64)` -- about 292 years -- as the "infinite"
/// sentinel.
///
/// Modelled as an enum here because Rust's [`Duration`] is **unsigned** and
/// cannot carry the sentinel, and because "forever" versus "292 years" is a
/// distinction a scheduler should not have to squint at. The three cases a
/// caller must handle are:
///
/// * [`Expiry::Never`] -- keep it loaded / wait indefinitely;
/// * `After(Duration::ZERO)` -- **not** the same thing as `Never`: for
///   keep-alive it mean *unload the moment the request finish*;
/// * `After(d)` -- the ordinary case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expiry {
    /// Upstream's `time.Duration(math.MaxInt64)`.
    Never,
    /// A finite wait. May be [`Duration::ZERO`].
    After(Duration),
}

impl Expiry {
    /// The same value Go would hold, in nanoseconds, so a ported test table can
    /// be compared number-for-number against upstream's.
    ///
    /// [`Expiry::Never`] is `i64::MAX`, which *is* `math.MaxInt64` -- same
    /// bits, same meaning.
    pub fn as_nanos_i64(&self) -> i64 {
        match self {
            Expiry::Never => i64::MAX,
            Expiry::After(d) => d.as_nanos().min(i64::MAX as u128) as i64,
        }
    }

    /// Is this the "forever" sentinel?
    pub fn is_never(&self) -> bool {
        matches!(self, Expiry::Never)
    }
}

// ---------------------------------------------------------------------------
// LogLevel -- Go's slog.Level
// ---------------------------------------------------------------------------

/// A log level on Go's `log/slog` numeric scale.
///
/// **Upstream:** `LogLevel()` returning `slog.Level`, plus
/// `logutil/logutil.go:12` for `LevelTrace`.
///
/// The scale is what it is because `slog` define it that way: **more negative
/// is more verbose**, and the levels are four apart so a caller can sit between
/// them (`INFO+2`). `KOPITIAM_DEBUG=2` mean TRACE because upstream compute the
/// level as `i * -4` -- so `2 -> -8`, and `-1 -> +4` (WARN), i.e. a *negative*
/// setting makes the log **quieter**. Not a typo, it is upstream's design.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LogLevel(pub i32);

impl LogLevel {
    /// `logutil.LevelTrace` (`logutil/logutil.go:12`).
    pub const TRACE: LogLevel = LogLevel(-8);
    /// `slog.LevelDebug`.
    pub const DEBUG: LogLevel = LogLevel(-4);
    /// `slog.LevelInfo` -- upstream's default when the var is unset.
    pub const INFO: LogLevel = LogLevel(0);
    /// `slog.LevelWarn`.
    pub const WARN: LogLevel = LogLevel(4);
    /// `slog.LevelError`.
    pub const ERROR: LogLevel = LogLevel(8);

    /// The raw `slog` number.
    pub fn as_i32(self) -> i32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Url -- the tiny slice of net/url that Host() needs
// ---------------------------------------------------------------------------

/// Scheme + host + path, and nothing else.
///
/// **Upstream:** the `*url.URL` that `Host()` return. It only ever set
/// `Scheme`, `Host` and `Path` -- never user-info, query or fragment -- so
/// pulling in a whole URL crate to hold three strings would be dependency for
/// its own sake. [`fmt::Display`] reproduce the subset of Go's
/// `(*url.URL).String()` that those three fields exercise, **including** the
/// quirk that a relative path get a `/` inserted before it when the host is
/// non-empty (that is how `https://example.com/ollama` survive the round trip:
/// upstream cut the path at the first `/`, so it is stored *without* the
/// leading slash and put back on the way out).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Url {
    /// `http` or `https` normally, but whatever was before `://` is kept.
    pub scheme: String,
    /// Already `host:port`, already bracketed if it is a literal IPv6 address.
    pub host: String,
    /// Everything after the first `/`, **without** that slash.
    pub path: String,
}

impl fmt::Display for Url {
    /// **Upstream:** `(*url.URL).String()`, restricted to the fields `Host()`
    /// actually populate. Path is emitted as-is: upstream would percent-escape
    /// it, but the only source here is an env var the operator typed, and
    /// silently rewriting their proxy path would be worse than passing it
    /// through.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}://{}", self.scheme, self.host)?;
        if !self.path.is_empty() {
            if !self.path.starts_with('/') && !self.host.is_empty() {
                f.write_str("/")?;
            }
            f.write_str(&self.path)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The env seam
// ---------------------------------------------------------------------------

/// A snapshot of the environment, plus the naming policy for reading it.
///
/// **Upstream:** the whole of `envconfig/config.go`, which is a package of
/// free functions closing over `os.Getenv`. Turning it into a struct is the
/// KOPITIAM divergence explained in the module docs: the prefix become data,
/// and the core become pure.
///
/// Cheap to build, cheap to query -- every accessor is a map lookup plus a bit
/// of string parsing, no caching, no locks. Read them as often as you like.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Env {
    prefixes: Vec<String>,
    home_dir_name: String,
    vars: BTreeMap<String, String>,
}

impl Default for Env {
    /// An empty environment with KOPITIAM's naming policy -- i.e. every
    /// accessor return its documented default. Handy for tests.
    fn default() -> Self {
        Env {
            prefixes: vec![DEFAULT_PREFIX.to_string()],
            home_dir_name: DEFAULT_HOME_DIR_NAME.to_string(),
            vars: BTreeMap::new(),
        }
    }
}

impl Env {
    /// Snapshot the real process environment, KOPITIAM naming.
    ///
    /// This is the **only** function in the module that touch `std::env`, on
    /// purpose: everything below it is pure, so the parsing rules can be tested
    /// exhaustively without `set_var` (which is `unsafe` since Rust 2024 and
    /// races between parallel tests anyway).
    ///
    /// It is a *snapshot*. Change the environment afterwards and this `Env`
    /// will not notice -- which is what you want for a server that must not
    /// change its bind address halfway through a run.
    pub fn from_env() -> Self {
        Env {
            vars: std::env::vars().collect(),
            ..Env::default()
        }
    }

    /// An explicit map, KOPITIAM naming. The workhorse for tests and for
    /// callers who assemble config from somewhere other than the process
    /// environment (a config file, a launcher, a test harness).
    pub fn new(vars: BTreeMap<String, String>) -> Self {
        Env {
            vars,
            ..Env::default()
        }
    }

    /// **Byte-exact upstream behaviour**: prefix `OLLAMA_`, home dir
    /// `.ollama`.
    ///
    /// Use this when you deliberately want to interoperate with an existing
    /// ollama installation -- same env vars, same default model store, so a
    /// model already pulled by ollama is found without re-downloading. Every
    /// test table ported from `config_test.go` below is built through here,
    /// which is what make those tests meaningful as an oracle check.
    pub fn ollama(vars: BTreeMap<String, String>) -> Self {
        Env {
            prefixes: vec!["OLLAMA_".to_string()],
            home_dir_name: ".ollama".to_string(),
            vars,
        }
    }

    /// Replace the prefix search list. First prefix that yield a **non-empty**
    /// value wins, so `["KOPITIAM_", "OLLAMA_"]` mean "ours overrides theirs,
    /// but theirs still works".
    pub fn with_prefixes<I, S>(mut self, prefixes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.prefixes = prefixes.into_iter().map(Into::into).collect();
        self
    }

    /// Replace the dot-directory looked for under `$HOME` (default
    /// [`DEFAULT_HOME_DIR_NAME`]).
    pub fn with_home_dir_name(mut self, name: impl Into<String>) -> Self {
        self.home_dir_name = name.into();
        self
    }

    /// Set one variable. Builder sugar for tests and launchers.
    pub fn with_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.vars.insert(key.into(), value.into());
        self
    }

    /// The prefixes this `Env` search, in order.
    pub fn prefixes(&self) -> &[String] {
        &self.prefixes
    }

    // -- the primitive readers ---------------------------------------------

    /// Read a variable by its **full, unprefixed** name, stripped of
    /// surrounding whitespace and then of surrounding quotes.
    ///
    /// **Upstream:** `Var(key)` --
    /// `strings.Trim(strings.TrimSpace(os.Getenv(key)), "\"'")`.
    ///
    /// The double trim is not belt-and-braces, it is Windows: a value set
    /// through a `.cmd` file or a service manager routinely arrive as
    /// `" C:\models "`, quotes and all. Note the *order* -- space first, then
    /// quotes -- so `" ' value ' "` come out as `" value "` with the **inner**
    /// spaces intact. Upstream's own `TestVar` pin that, and so does ours.
    ///
    /// Use this for variables that are not ours to rename: `HTTP_PROXY`,
    /// `CUDA_VISIBLE_DEVICES`, `HOME`, `LLAMA_ARG_FIT`.
    ///
    /// **Windows fidelity:** environment variable names are case-insensitive on
    /// Windows and Go's `os.Getenv` honour that. Our map is a `BTreeMap`, so on
    /// Windows a case-sensitive miss fall back to a case-insensitive scan.
    /// Elsewhere the lookup stay strictly case-sensitive, because on Unix
    /// `Path` and `PATH` really are two different variables.
    pub fn raw(&self, key: &str) -> String {
        let found = self.vars.get(key).or_else(|| {
            if cfg!(windows) {
                self.vars
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case(key))
                    .map(|(_, v)| v)
            } else {
                None
            }
        });
        match found {
            Some(v) => v.trim().trim_matches(|c| c == '"' || c == '\'').to_string(),
            None => String::new(),
        }
    }

    /// Read a **prefixed** variable, trying each prefix in order.
    ///
    /// `env.var("HOST")` read `KOPITIAM_HOST` by default, `OLLAMA_HOST` under
    /// [`Env::ollama`]. Upstream has no equivalent -- it spell the full name at
    /// every call site -- so this is the seam, not a translation.
    ///
    /// An **empty** value counts as unset and fall through to the next prefix.
    /// That is upstream's convention too: every accessor there is written
    /// `if s := Var(k); s != ""`, so `OLLAMA_HOST=` behave exactly like not
    /// setting it at all.
    pub fn var(&self, suffix: &str) -> String {
        for p in &self.prefixes {
            let v = self.raw(&format!("{p}{suffix}"));
            if !v.is_empty() {
                return v;
            }
        }
        String::new()
    }

    /// A prefixed string variable, empty when unset. **Upstream:** `String(k)`.
    pub fn string(&self, suffix: &str) -> String {
        self.var(suffix)
    }

    /// A prefixed boolean with a caller-supplied default.
    ///
    /// **Upstream:** `BoolWithDefault(k)`.
    ///
    /// **The quirk, and it is deliberate:** an *unparseable* value read as
    /// **`true`**. `KOPITIAM_FLASH_ATTENTION=yes`, `=on`, `=please` -- all
    /// `true`. Only Go's `strconv.ParseBool` vocabulary is understood
    /// (`1 t T TRUE true True 0 f F FALSE false False`), and anything outside
    /// it is read as "the operator clearly meant to turn this on". Upstream's
    /// own `TestBool` pin `"random" -> true`. Do not "fix" this: a flag that
    /// silently stayed off because somebody typed `yes` is a far nastier
    /// support call than one that turned on.
    pub fn bool_var_with_default(&self, suffix: &str, default: bool) -> bool {
        let s = self.var(suffix);
        if s.is_empty() {
            return default;
        }
        parse_go_bool(&s).unwrap_or(true)
    }

    /// A prefixed boolean defaulting to `false`. **Upstream:** `Bool(k)`.
    pub fn bool_var(&self, suffix: &str) -> bool {
        self.bool_var_with_default(suffix, false)
    }

    /// Did the operator set this boolean **explicitly**, whatever they set it
    /// to?
    ///
    /// **Upstream idiom**, not an upstream function: `llm/llama_server.go:608`
    /// write `userSet := enabled == envconfig.FlashAttention(true)` where
    /// `enabled` came from `FlashAttention(false)`. Ask for the flag twice with
    /// opposite defaults; if both answers agree, the answer came from the
    /// environment rather than from a default. `discover/runner.go:418` do the
    /// same for integrated GPUs.
    ///
    /// Worth having as a named function because the trick is invisible at the
    /// call site, and because "unset" versus "explicitly false" genuinely
    /// matter -- it is the difference between "auto-detect" and "the operator
    /// told you no".
    pub fn bool_var_is_set(&self, suffix: &str) -> bool {
        self.bool_var_with_default(suffix, false) == self.bool_var_with_default(suffix, true)
    }

    /// A prefixed unsigned integer, falling back to `default` on anything the
    /// parser dislike.
    ///
    /// **Upstream:** `Uint(key, default)` and `Uint64(key, default)`, which are
    /// the same function twice over -- Go's `uint` is platform-sized, so
    /// upstream's `Uint` narrow a `uint64` on 32-bit targets. We return `u64`
    /// for both. **Deliberate divergence:** it means a 32-bit Android build and
    /// a 64-bit Windows build agree on what `KOPITIAM_MAX_QUEUE=5000000000`
    /// mean, instead of one of them wrapping. Callers that need a narrower type
    /// (e.g. `crate::options::Runner::num_ctx`, a `u32`) do the conversion
    /// themselves, where the truncation is visible.
    ///
    /// "Anything the parser dislike" is wider than it look, because upstream
    /// parse **base 10 only**: `-1`, `0x10`, `0o10`, `1_000` and `string` all
    /// fall back to the default without a peep. Upstream's `TestUint` pin
    /// exactly that list.
    pub fn uint(&self, suffix: &str, default: u64) -> u64 {
        let s = self.var(suffix);
        if s.is_empty() {
            return default;
        }
        parse_go_uint(&s).unwrap_or(default)
    }

    // -- the configured values ---------------------------------------------

    /// Scheme and host we serve on / talk to.
    ///
    /// **Upstream:** `Host()`. Default `http://127.0.0.1:11434` -- loopback,
    /// not `0.0.0.0`, so a fresh install is not on the network by accident.
    ///
    /// The parsing is fussier than it looks, and every branch earn its keep:
    ///
    /// * **no scheme** -> `http`, with one special case: bare `ollama.com`
    ///   become `https://ollama.com:443`, because that host is only ever
    ///   reached over TLS;
    /// * **the default port depend on the scheme** -- `11434` when no scheme
    ///   was given, but `80` for explicit `http://` and `443` for `https://`.
    ///   So `1.2.3.4` and `http://1.2.3.4` do **not** resolve to the same
    ///   thing. That surprise people; it is upstream's behaviour and changing
    ///   it would break every existing `OLLAMA_HOST` in the wild;
    /// * a **path** is kept (`https://example.com/ollama` -> path `ollama`), so
    ///   a reverse proxy can mount us under a sub-path;
    /// * a **port** outside `0..=65535`, or one that is not a number, fall back
    ///   to the scheme's default rather than erroring;
    /// * **IPv6** work with or without brackets: `::1`, `[::1]` and `[::1]:1337`
    ///   all land correctly, and the output is always re-bracketed.
    pub fn host(&self) -> Url {
        let mut default_port = "11434";
        let raw = self.var("HOST");
        // TrimSpace *again* after Var already trimmed, because Var strip the
        // quotes second: ` " 1.2.3.4 " ` come out of Var as ` 1.2.3.4 `.
        let s = raw.trim();

        let (scheme, hostport): (&str, &str) = match s.split_once("://") {
            None => {
                if s == "ollama.com" {
                    ("https", "ollama.com:443")
                } else {
                    ("http", s)
                }
            }
            Some((sch, hp)) => {
                if sch == "http" {
                    default_port = "80";
                } else if sch == "https" {
                    default_port = "443";
                }
                (sch, hp)
            }
        };

        let (hostport, path) = match hostport.split_once('/') {
            Some((hp, p)) => (hp, p),
            None => (hostport, ""),
        };

        let (host, port) = match go_split_host_port(hostport) {
            Some((h, p)) => (h.to_string(), p.to_string()),
            None => {
                // No usable `host:port`. Upstream then try to salvage a bare
                // address: an IP literal (brackets stripped) is normalised
                // through Go's `IP.String()`, anything else non-empty is taken
                // as a hostname, and only a truly empty value fall back to
                // loopback.
                let bare = hostport.trim_matches(|c| c == '[' || c == ']');
                let h = if let Some(ip) = go_parse_ip(bare) {
                    go_ip_string(ip)
                } else if !hostport.is_empty() {
                    hostport.to_string()
                } else {
                    "127.0.0.1".to_string()
                };
                (h, default_port.to_string())
            }
        };

        // `strconv.ParseInt(port, 10, 32)` -- note bit size 32, so a port that
        // overflow an i32 is invalid too, not just one over 65535.
        let port = match port.parse::<i32>() {
            Ok(n) if (0..=65535).contains(&n) => port,
            _ => default_port.to_string(),
        };

        Url {
            scheme: scheme.to_string(),
            host: go_join_host_port(&host, &port),
            path: path.to_string(),
        }
    }

    /// [`Env::host`] with an unspecified bind address swapped for loopback.
    ///
    /// **Upstream:** `ConnectableHost()`. `0.0.0.0` -> `127.0.0.1`, `::` ->
    /// `::1`, everything else untouched.
    ///
    /// This exists because `0.0.0.0` is a perfectly good address to **bind** a
    /// listening socket to and a meaningless address to **connect** to -- and
    /// on Windows connecting to it fail outright rather than quietly meaning
    /// loopback the way it do on Linux. So a client must never dial
    /// [`Env::host`] directly; it dial this.
    pub fn connectable_host(&self) -> Url {
        let mut u = self.host();
        let replacement = go_split_host_port(&u.host).and_then(|(h, p)| {
            let ip = go_parse_ip(h)?;
            if !ip.is_unspecified() {
                return None;
            }
            // Upstream branch on `ip.To4() != nil`, which is true for a real
            // IPv4 address *and* for an IPv4-mapped IPv6 one -- both want the
            // v4 loopback.
            let loop_back = match ip {
                IpAddr::V4(_) => "127.0.0.1",
                IpAddr::V6(v6) if v6.to_ipv4_mapped().is_some() => "127.0.0.1",
                IpAddr::V6(_) => "::1",
            };
            Some(go_join_host_port(loop_back, p))
        });
        if let Some(h) = replacement {
            u.host = h;
        }
        u
    }

    /// CORS origins the HTTP surface should accept.
    ///
    /// **Upstream:** `AllowedOrigins()`.
    ///
    /// **The built-ins are always appended, never replaced.** Setting the var
    /// add to the list; it cannot take `http://localhost` away. That is a
    /// deliberate upstream choice -- the local UI and the editor webviews must
    /// keep working no matter what an operator paste in -- but it does mean
    /// this list is not a lockdown mechanism. If you need one, filter
    /// downstream; do not expect this function to hand you a short list.
    ///
    /// Splitting is on `,` with **no trimming**, matching `strings.Split`, so
    /// `a, b` give `"a"` and `" b"` -- the space is part of the origin and will
    /// never match. Pinned by a test, because it looks like a bug.
    ///
    /// The four spellings per host (`http`/`https` x bare/`:*`) exist because
    /// the CORS `Origin` header carry an explicit port only when it is
    /// non-default, so both forms must be listed. The tail
    /// (`app://`, `file://`, `tauri://`, `vscode-webview://`, `vscode-file://`)
    /// cover desktop shells and editor webviews, which send those schemes.
    pub fn allowed_origins(&self) -> Vec<String> {
        let mut origins: Vec<String> = Vec::new();
        let s = self.var("ORIGINS");
        if !s.is_empty() {
            origins.extend(s.split(',').map(str::to_string));
        }
        for origin in ["localhost", "127.0.0.1", "0.0.0.0"] {
            origins.push(format!("http://{origin}"));
            origins.push(format!("https://{origin}"));
            origins.push(format!("http://{}", go_join_host_port(origin, "*")));
            origins.push(format!("https://{}", go_join_host_port(origin, "*")));
        }
        for extra in [
            "app://*",
            "file://*",
            "tauri://*",
            "vscode-webview://*",
            "vscode-file://*",
        ] {
            origins.push(extra.to_string());
        }
        origins
    }

    /// Where model blobs and manifests live.
    ///
    /// **Upstream:** `Models()`, default `$HOME/.ollama/models`. Ours default
    /// to `$HOME/.kopitiam/models` -- see [`DEFAULT_HOME_DIR_NAME`] -- and
    /// `Env::ollama(..).models()` give the upstream path back, which is how you
    /// share a model store with an existing ollama install instead of
    /// downloading the same weights twice.
    ///
    /// **Always forward slashes**, on every platform, because this string end
    /// up in manifests, logs and beads that get read on Windows *and* on
    /// Termux, and a mixed-separator path (`.../git\checkpoint/...`) is a
    /// documented KOPITIAM rough edge we are not going to add to. On Windows a
    /// backslash in the operator-supplied value is rewritten too; on Unix it is
    /// left alone, because there a backslash is a legal character in a
    /// filename and rewriting it would corrupt the path.
    ///
    /// Returns [`EnvConfigError::NoHomeDir`] instead of panicking -- see the
    /// module docs.
    pub fn models(&self) -> Result<String, EnvConfigError> {
        let s = self.var("MODELS");
        if !s.is_empty() {
            return Ok(slashify(&s));
        }
        let home = self.home_dir()?;
        Ok(format!(
            "{}/{}/models",
            home.trim_end_matches('/'),
            self.home_dir_name
        ))
    }

    /// The home directory, forward-slashed.
    ///
    /// **Upstream:** `os.UserHomeDir()`, which read exactly one variable per
    /// platform (`USERPROFILE` on Windows, `HOME` on Unix, `home` on Plan 9)
    /// and error otherwise.
    ///
    /// **Divergence:** we try both `USERPROFILE` and `HOME`, native one first,
    /// and only then give up. Reason: KOPITIAM run under Git Bash and MSYS on
    /// Windows (where `HOME` is set and `USERPROFILE` may not be inherited)
    /// and under Termux on Android (where `HOME` is
    /// `/data/data/com.termux/files/home` and there is no `USERPROFILE`). One
    /// variable per platform would fail in both of the places we actually
    /// develop. Plan 9's lowercase `home` is not consulted -- Rust does not
    /// target it.
    pub fn home_dir(&self) -> Result<String, EnvConfigError> {
        let candidates: &[&str] = if cfg!(windows) {
            &["USERPROFILE", "HOME"]
        } else {
            &["HOME", "USERPROFILE"]
        };
        for k in candidates {
            let v = self.raw(k);
            if !v.is_empty() {
                return Ok(slashify(&v));
            }
        }
        Err(EnvConfigError::NoHomeDir(candidates.join(", ")))
    }

    /// How long a model stay resident after its last request.
    ///
    /// **Upstream:** `KeepAlive()`. **Default 5 minutes.**
    ///
    /// Accept **two** spellings, tried in that order:
    ///
    /// 1. a Go duration -- `1s`, `5m`, `1h2m3s`, `1.5h`. Units are
    ///    `ns us/µs/μs ms s m h` and nothing else, so `1d`, `1w` and `1y` are
    ///    **not** durations and fall through;
    /// 2. a bare integer, meaning **seconds** -- `60` is one minute.
    ///
    /// Neither works -> the 5 minute default. So `1d` quietly give you five
    /// minutes, not a day. That trap is upstream's and is pinned by a test.
    ///
    /// **Negative mean forever** ([`Expiry::Never`]); plain `0` mean
    /// [`Expiry::After`]`(ZERO)` -- unload as soon as the request finish. Note
    /// `-0` parse as zero, not as negative, so it mean *unload immediately*
    /// too.
    pub fn keep_alive(&self) -> Expiry {
        // 5 * time.Minute
        let mut nanos: i64 = 5 * 60 * 1_000_000_000;
        let s = self.var("KEEP_ALIVE");
        if !s.is_empty() {
            if let Some(d) = parse_go_duration(&s) {
                nanos = d;
            } else if let Ok(n) = s.parse::<i64>() {
                // Go's `time.Duration(n) * time.Second` wrap on overflow rather
                // than saturating; a wrapped value goes negative and therefore
                // reads as "forever", which is upstream's behaviour too.
                nanos = n.wrapping_mul(1_000_000_000);
            }
        }
        if nanos < 0 {
            Expiry::Never
        } else {
            Expiry::After(Duration::from_nanos(nanos as u64))
        }
    }

    /// How long a model load may stall before we give up on it.
    ///
    /// **Upstream:** `LoadTimeout()`. **Default 5 minutes**, same two spellings
    /// as [`Env::keep_alive`].
    ///
    /// **The one difference, and it is the whole reason these are two
    /// functions:** here the test is `<= 0`, not `< 0`. So `0` mean
    /// [`Expiry::Never`] -- *wait forever* -- whereas for keep-alive `0` mean
    /// *unload immediately*. Same var syntax, opposite meaning at zero.
    /// Copy-pasting one implementation into the other give you a load that hang
    /// until the process die, so the check stays spelled out here.
    pub fn load_timeout(&self) -> Expiry {
        let mut nanos: i64 = 5 * 60 * 1_000_000_000;
        let s = self.var("LOAD_TIMEOUT");
        if !s.is_empty() {
            if let Some(d) = parse_go_duration(&s) {
                nanos = d;
            } else if let Ok(n) = s.parse::<i64>() {
                nanos = n.wrapping_mul(1_000_000_000);
            }
        }
        if nanos <= 0 {
            Expiry::Never
        } else {
            Expiry::After(Duration::from_nanos(nanos as u64))
        }
    }

    /// Hosts allowed to serve remote models.
    ///
    /// **Upstream:** `Remotes()`, default `["ollama.com"]`.
    ///
    /// The default is kept **faithful** rather than blanked, because this is an
    /// allow-list, not an action: nothing here cause a network call, it only
    /// decide which host a remote model reference may name. KOPITIAM's
    /// offline-first rule is about what we *do*, not about what we would permit
    /// if asked. A deployment that want no remotes at all set the var to a host
    /// it control -- there is deliberately no "empty means none", since empty
    /// mean "unset" everywhere in this module.
    pub fn remotes(&self) -> Vec<String> {
        let raw = self.var("REMOTES");
        if raw.is_empty() {
            vec!["ollama.com".to_string()]
        } else {
            raw.split(',').map(str::to_string).collect()
        }
    }

    /// Log verbosity, from `<PREFIX>DEBUG`.
    ///
    /// **Upstream:** `LogLevel()`. **Default [`LogLevel::INFO`].**
    ///
    /// Reads as a boolean *first*: `true`/`t`/`1` -> [`LogLevel::DEBUG`].
    /// Failing that, as an integer, and the level become `n * -4`, so `2` ->
    /// [`LogLevel::TRACE`] and `-1` -> [`LogLevel::WARN`]. A **negative** value
    /// therefore make the log **quieter**, which is worth knowing before
    /// somebody sets `DEBUG=-1` expecting more output.
    ///
    /// `false`, `f` and `0` all leave it at INFO.
    pub fn log_level(&self) -> LogLevel {
        let s = self.var("DEBUG");
        if s.is_empty() {
            return LogLevel::INFO;
        }
        if parse_go_bool(&s).unwrap_or(false) {
            return LogLevel::DEBUG;
        }
        // `i, _ := strconv.ParseInt(...)` -- a parse failure leave i == 0,
        // which fall through to INFO.
        match s.parse::<i64>() {
            Ok(i) if i != 0 => {
                LogLevel(i.saturating_mul(-4).clamp(i32::MIN as i64, i32::MAX as i64) as i32)
            }
            _ => LogLevel::INFO,
        }
    }

    // -- the flags, each naming its upstream default -------------------------

    /// **Upstream:** `FlashAttention = BoolWithDefault("OLLAMA_FLASH_ATTENTION")`.
    /// The default is the **caller's** -- upstream pass `false` when reporting
    /// it (`config.go:316`) and use the two-call [`Env::bool_var_is_set`] trick
    /// at `llm/llama_server.go:608` to tell "off" from "unset". Mirrored
    /// faithfully: you must say what you want.
    pub fn flash_attention(&self, default: bool) -> bool {
        self.bool_var_with_default("FLASH_ATTENTION", default)
    }

    /// **Upstream:** `GoTemplate = BoolWithDefault("OLLAMA_GO_TEMPLATE")`, and
    /// every real call site pass **`true`** (`config.go:315`,
    /// `server/routes.go:2358`). Enable Modelfile `TEMPLATE` rendering when the
    /// model carry one.
    pub fn go_template(&self, default: bool) -> bool {
        self.bool_var_with_default("GO_TEMPLATE", default)
    }

    /// **Upstream:** `EnableVulkan = BoolWithDefault("OLLAMA_VULKAN")`, called
    /// with **`true`** at `discover/runner.go:108` and `config.go:360`.
    pub fn enable_vulkan(&self, default: bool) -> bool {
        self.bool_var_with_default("VULKAN", default)
    }

    /// **Upstream:** `EnableIntegratedGPU = BoolWithDefault("OLLAMA_IGPU_ENABLE")`.
    /// `discover/runner.go:418-419` call it with **both** defaults precisely to
    /// detect whether the operator said anything -- see [`Env::bool_var_is_set`].
    pub fn enable_integrated_gpu(&self, default: bool) -> bool {
        self.bool_var_with_default("IGPU_ENABLE", default)
    }

    /// **Upstream:** `DebugLogRequests = Bool("OLLAMA_DEBUG_LOG_REQUESTS")`.
    /// **Default `false`.** Writes inference request bodies to a temp
    /// directory -- prompts included, so it is a privacy switch as much as a
    /// debugging one.
    pub fn debug_log_requests(&self) -> bool {
        self.bool_var("DEBUG_LOG_REQUESTS")
    }

    /// **Upstream:** `NoHistory = Bool("OLLAMA_NOHISTORY")`. **Default `false`**
    /// (i.e. history *is* kept).
    pub fn no_history(&self) -> bool {
        self.bool_var("NOHISTORY")
    }

    /// **Upstream:** `NoPrune = Bool("OLLAMA_NOPRUNE")`. **Default `false`**,
    /// i.e. unreferenced model blobs *are* pruned at startup. Turn it on when
    /// blobs are shared with another tool whose manifests we cannot see.
    pub fn no_prune(&self) -> bool {
        self.bool_var("NOPRUNE")
    }

    /// **Upstream:** `SchedSpread = Bool("OLLAMA_SCHED_SPREAD")`. **Default
    /// `false`** -- pack a model onto as few GPUs as possible. Turn it on to
    /// always spread across every GPU, which cost interconnect bandwidth but
    /// let a model run that would not fit on one card.
    pub fn sched_spread(&self) -> bool {
        self.bool_var("SCHED_SPREAD")
    }

    /// **Upstream:** `UseAuth = Bool("OLLAMA_AUTH")`. **Default `false`.**
    pub fn use_auth(&self) -> bool {
        self.bool_var("AUTH")
    }

    /// **Upstream:** `NoCloudEnv = Bool("OLLAMA_NO_CLOUD")`. **Default
    /// `false`.** Only the environment half -- the file half live in
    /// [`ServerConfig`], and [`Env::no_cloud`] combine them.
    pub fn no_cloud_env(&self) -> bool {
        self.bool_var("NO_CLOUD")
    }

    /// Quantisation type for the K/V cache, e.g. `q8_0`, `q4_0`.
    ///
    /// **Upstream:** `KvCacheType = String("OLLAMA_KV_CACHE_TYPE")`, **default
    /// empty**, and empty mean `f16` -- upstream's own help text say so
    /// (`config.go:317`). So an empty string here is *not* "no cache", it is
    /// "full-precision cache". Do not treat it as a missing value.
    pub fn kv_cache_type(&self) -> String {
        self.string("KV_CACHE_TYPE")
    }

    // -- the numbers, each naming its upstream default -----------------------

    /// Context window in tokens.
    ///
    /// **Upstream:** `ContextLength = Uint("OLLAMA_CONTEXT_LENGTH", 0)` --
    /// **default `0`**.
    ///
    /// `0` do **not** mean "no context". It mean **decide at load time from
    /// available VRAM**; upstream's help text spell out the ladder it choose
    /// from: *"default: 4k/32k/256k based on VRAM"*. So a caller must treat `0`
    /// as "not yet decided" and resolve it against the model's trained maximum
    /// and the machine -- never pass it downstream as a literal window size.
    /// `crate::options::Runner::num_ctx` carry the same warning; the two must
    /// stay consistent.
    pub fn context_length(&self) -> u64 {
        self.uint("CONTEXT_LENGTH", 0)
    }

    /// **Upstream:** `NumParallel = Uint("OLLAMA_NUM_PARALLEL", 1)` --
    /// **default `1`.** How many requests one loaded model serve at once. Each
    /// parallel slot need its own share of the KV cache, so raising this shrink
    /// the effective context per request.
    pub fn num_parallel(&self) -> u64 {
        self.uint("NUM_PARALLEL", 1)
    }

    /// **Upstream:** `MaxRunners = Uint("OLLAMA_MAX_LOADED_MODELS", 0)` --
    /// **default `0`**, meaning *let the scheduler decide from the hardware*,
    /// not *load nothing*. Same "0 is undecided" convention as
    /// [`Env::context_length`].
    pub fn max_runners(&self) -> u64 {
        self.uint("MAX_LOADED_MODELS", 0)
    }

    /// **Upstream:** `MaxQueue = Uint("OLLAMA_MAX_QUEUE", 512)` -- **default
    /// `512`** queued requests before we start rejecting.
    pub fn max_queue(&self) -> u64 {
        self.uint("MAX_QUEUE", 512)
    }

    /// **Upstream:** `MaxTransferStreams = Uint("OLLAMA_MAX_TRANSFER_STREAMS", 4)`
    /// -- **default `4`.** Caps simultaneous body-bearing transfers during
    /// safetensors pull/push so a slow link is not saturated. Upstream note it
    /// has **no effect on GGUF** transfers, which take the legacy path.
    pub fn max_transfer_streams(&self) -> u64 {
        self.uint("MAX_TRANSFER_STREAMS", 4)
    }

    /// **Upstream:** `GpuOverhead = Uint64("OLLAMA_GPU_OVERHEAD", 0)` --
    /// **default `0` bytes.** VRAM held back per GPU, for the display server
    /// and anything else sharing the card.
    pub fn gpu_overhead(&self) -> u64 {
        self.uint("GPU_OVERHEAD", 0)
    }

    /// **Upstream:** `LLMLibrary = String("OLLAMA_LLM_LIBRARY")`. Force a
    /// backend library instead of autodetecting. Empty = autodetect.
    pub fn llm_library(&self) -> String {
        self.string("LLM_LIBRARY")
    }

    /// **Upstream:** `Editor = String("OLLAMA_EDITOR")`. Editor launched for
    /// interactive prompt editing.
    pub fn editor(&self) -> String {
        self.string("EDITOR")
    }

    /// Should cloud features be off?
    ///
    /// **Upstream:** `NoCloud()`, which OR the env var with the
    /// `disable_ollama_cloud` field of `~/.ollama/server.json`.
    ///
    /// **Divergence:** upstream read and cache that file behind a package-level
    /// `sync.RWMutex`. We take the parsed [`ServerConfig`] as an argument, so
    /// there is no global, no lock, no hidden file I/O, and the function stay
    /// pure. Pass `&ServerConfig::default()` if you have no file.
    pub fn no_cloud(&self, config: &ServerConfig) -> bool {
        self.no_cloud_env() || config.disable_cloud
    }

    /// Which of the two switches turned cloud off.
    ///
    /// **Upstream:** `NoCloudSource()`, returning the strings `"none"`,
    /// `"env"`, `"config"`, `"both"`. Modelled as an enum -- the string set was
    /// closed, so Rust can make that a compile-time fact;
    /// [`NoCloudSource::as_str`] give the upstream spelling back for logs.
    pub fn no_cloud_source(&self, config: &ServerConfig) -> NoCloudSource {
        match (self.no_cloud_env(), config.disable_cloud) {
            (true, true) => NoCloudSource::Both,
            (true, false) => NoCloudSource::Env,
            (false, true) => NoCloudSource::Config,
            (false, false) => NoCloudSource::None,
        }
    }

    // -- the unprefixed ones, which belong to other people --------------------

    /// `CUDA_VISIBLE_DEVICES`. **Upstream:** `CudaVisibleDevices`. Not
    /// prefixed -- it is NVIDIA's variable, not ours.
    pub fn cuda_visible_devices(&self) -> String {
        self.raw("CUDA_VISIBLE_DEVICES")
    }

    /// `HIP_VISIBLE_DEVICES` (AMD, by numeric ID). **Upstream:** `HipVisibleDevices`.
    pub fn hip_visible_devices(&self) -> String {
        self.raw("HIP_VISIBLE_DEVICES")
    }

    /// `ROCR_VISIBLE_DEVICES` (AMD, by UUID or numeric ID). **Upstream:** `RocrVisibleDevices`.
    pub fn rocr_visible_devices(&self) -> String {
        self.raw("ROCR_VISIBLE_DEVICES")
    }

    /// `GGML_VK_VISIBLE_DEVICES` (Vulkan, by numeric ID). **Upstream:** `VkVisibleDevices`.
    pub fn vk_visible_devices(&self) -> String {
        self.raw("GGML_VK_VISIBLE_DEVICES")
    }

    /// `GPU_DEVICE_ORDINAL` (AMD, by numeric ID). **Upstream:** `GpuDeviceOrdinal`.
    pub fn gpu_device_ordinal(&self) -> String {
        self.raw("GPU_DEVICE_ORDINAL")
    }

    /// `HSA_OVERRIDE_GFX_VERSION`. **Upstream:** `HsaOverrideGfxVersion`.
    /// Override the gfx target for every detected AMD GPU -- the usual
    /// workaround for a card ROCm has not been told about yet.
    pub fn hsa_override_gfx_version(&self) -> String {
        self.raw("HSA_OVERRIDE_GFX_VERSION")
    }

    /// The full name a [`VarDoc`] refer to under this `Env`'s naming policy --
    /// e.g. `KOPITIAM_HOST`, or `HTTP_PROXY` for the unprefixed ones. The
    /// **first** prefix is used, since that is the one an operator should
    /// reach for.
    pub fn full_name(&self, doc: &VarDoc) -> String {
        if doc.prefixed {
            let p = self.prefixes.first().map(String::as_str).unwrap_or("");
            format!("{p}{}", doc.suffix)
        } else {
            doc.suffix.to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// server.json
// ---------------------------------------------------------------------------

/// The handful of settings that live in a config file rather than the
/// environment.
///
/// **Upstream:** `serverConfigData`, read from `~/.ollama/server.json`.
///
/// The field is renamed on the Rust side only: the JSON key stay
/// `disable_ollama_cloud` so an existing ollama `server.json` is read
/// correctly, while the Rust field say [`ServerConfig::disable_cloud`] because
/// in KOPITIAM there is no "ollama cloud" to name.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default, rename = "disable_ollama_cloud")]
    pub disable_cloud: bool,
}

impl ServerConfig {
    /// Parse, treating **any** problem as "no config".
    ///
    /// **Upstream:** `loadServerConfig()`, which log at debug level and carry
    /// on with a zero value when the file is missing *or* malformed. Same here,
    /// and deliberately so: a typo in an optional config file must not stop the
    /// runtime from starting. Upstream's own `TestNoCloud` pin the
    /// `{invalid json` case as "not disabled".
    pub fn parse(json: &str) -> Self {
        serde_json::from_str(json).unwrap_or_default()
    }
}

/// Where the "cloud is off" decision came from.
/// **Upstream:** the four strings returned by `NoCloudSource()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoCloudSource {
    /// Cloud is not disabled.
    None,
    /// Disabled by the environment variable only.
    Env,
    /// Disabled by `server.json` only.
    Config,
    /// Disabled by both.
    Both,
}

impl NoCloudSource {
    /// The exact string upstream return, for logs that must stay comparable.
    pub fn as_str(&self) -> &'static str {
        match self {
            NoCloudSource::None => "none",
            NoCloudSource::Env => "env",
            NoCloudSource::Config => "config",
            NoCloudSource::Both => "both",
        }
    }
}

impl fmt::Display for NoCloudSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// The documentation table
// ---------------------------------------------------------------------------

/// One row of the "what do these variables do" table.
///
/// **Upstream:** `EnvVar` and `AsMap()`, minus the value column -- see the
/// module docs for why the value column is the caller's job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarDoc {
    /// The name **without** any prefix.
    pub suffix: &'static str,
    /// `true` if this var take the [`Env`]'s prefix; `false` for the ones that
    /// belong to other tools (`HTTP_PROXY`, `CUDA_VISIBLE_DEVICES`, ...) and
    /// therefore must keep their exact name.
    pub prefixed: bool,
    /// Upstream's own help text, near-verbatim -- only the literal `OLLAMA_`
    /// spellings inside the prose are dropped, since the prefix is now
    /// configurable.
    pub description: &'static str,
}

/// Every variable this module read, with upstream's description.
///
/// **Upstream:** the descriptions in `AsMap()` (`config.go:311-363`). Kept as
/// data rather than thrown away with `AsMap`, because the *explanations* are
/// the durable knowledge -- somebody a year from now needs to know what
/// `SCHED_SPREAD` does, not how Go's `%v` verb print a duration.
///
/// Render a row's real name with [`Env::full_name`].
#[rustfmt::skip]
pub const VAR_DOCS: &[VarDoc] = &[
    VarDoc { suffix: "DEBUG", prefixed: true, description: "Show additional debug information (e.g. DEBUG=1; 2 is trace, negative is quieter)" },
    VarDoc { suffix: "DEBUG_LOG_REQUESTS", prefixed: true, description: "Log inference request bodies and replay curl commands to a temp directory" },
    VarDoc { suffix: "GO_TEMPLATE", prefixed: true, description: "Enable Modelfile TEMPLATE based rendering when available" },
    VarDoc { suffix: "FLASH_ATTENTION", prefixed: true, description: "Enable flash attention" },
    VarDoc { suffix: "KV_CACHE_TYPE", prefixed: true, description: "Quantization type for the K/V cache (default: f16)" },
    VarDoc { suffix: "GPU_OVERHEAD", prefixed: true, description: "Reserve a portion of VRAM per GPU (bytes)" },
    VarDoc { suffix: "IGPU_ENABLE", prefixed: true, description: "Enable integrated GPUs" },
    VarDoc { suffix: "VULKAN", prefixed: true, description: "Enable Vulkan support" },
    VarDoc { suffix: "HOST", prefixed: true, description: "IP address for the server (default 127.0.0.1:11434)" },
    VarDoc { suffix: "KEEP_ALIVE", prefixed: true, description: "The duration that models stay loaded in memory (default \"5m\")" },
    VarDoc { suffix: "LLM_LIBRARY", prefixed: true, description: "Set LLM library to bypass autodetection" },
    VarDoc { suffix: "LOAD_TIMEOUT", prefixed: true, description: "How long to allow model loads to stall before giving up (default \"5m\")" },
    VarDoc { suffix: "MAX_LOADED_MODELS", prefixed: true, description: "Maximum number of loaded models per GPU" },
    VarDoc { suffix: "MAX_TRANSFER_STREAMS", prefixed: true, description: "Maximum parallel transfer streams for safetensors model pulls/pushes (default 4)" },
    VarDoc { suffix: "MAX_QUEUE", prefixed: true, description: "Maximum number of queued requests" },
    VarDoc { suffix: "MODELS", prefixed: true, description: "The path to the models directory" },
    VarDoc { suffix: "NO_CLOUD", prefixed: true, description: "Disable cloud features (remote inference and web search)" },
    VarDoc { suffix: "NOHISTORY", prefixed: true, description: "Do not preserve readline history" },
    VarDoc { suffix: "NOPRUNE", prefixed: true, description: "Do not prune model blobs on startup" },
    VarDoc { suffix: "NUM_PARALLEL", prefixed: true, description: "Maximum number of parallel requests" },
    VarDoc { suffix: "ORIGINS", prefixed: true, description: "A comma separated list of allowed origins (the built-in localhost/file/editor origins are always added)" },
    VarDoc { suffix: "SCHED_SPREAD", prefixed: true, description: "Always schedule model across all GPUs" },
    VarDoc { suffix: "CONTEXT_LENGTH", prefixed: true, description: "Context length to use unless otherwise specified (default: 4k/32k/256k based on VRAM)" },
    VarDoc { suffix: "EDITOR", prefixed: true, description: "Path to editor for interactive prompt editing" },
    VarDoc { suffix: "REMOTES", prefixed: true, description: "Allowed hosts for remote models (default \"ollama.com\")" },
    VarDoc { suffix: "AUTH", prefixed: true, description: "Enable authentication between client and server" },
    VarDoc { suffix: "LLAMA_ARG_FIT", prefixed: false, description: "Enable llama.cpp automatic fit of unset memory options (default \"on\")" },
    VarDoc { suffix: "LLAMA_ARG_FIT_TARGET", prefixed: false, description: "Target free VRAM margin per device for llama.cpp fit (MiB)" },
    VarDoc { suffix: "HTTP_PROXY", prefixed: false, description: "HTTP proxy" },
    VarDoc { suffix: "HTTPS_PROXY", prefixed: false, description: "HTTPS proxy" },
    VarDoc { suffix: "NO_PROXY", prefixed: false, description: "No proxy" },
    VarDoc { suffix: "CUDA_VISIBLE_DEVICES", prefixed: false, description: "Set which NVIDIA devices are visible" },
    VarDoc { suffix: "HIP_VISIBLE_DEVICES", prefixed: false, description: "Set which AMD devices are visible by numeric ID" },
    VarDoc { suffix: "ROCR_VISIBLE_DEVICES", prefixed: false, description: "Set which AMD devices are visible by UUID or numeric ID" },
    VarDoc { suffix: "GGML_VK_VISIBLE_DEVICES", prefixed: false, description: "Set which Vulkan devices are visible by numeric ID" },
    VarDoc { suffix: "GPU_DEVICE_ORDINAL", prefixed: false, description: "Set which AMD devices are visible by numeric ID" },
    VarDoc { suffix: "HSA_OVERRIDE_GFX_VERSION", prefixed: false, description: "Override the gfx used for all detected AMD GPUs" },
];

// ---------------------------------------------------------------------------
// Go parsing primitives
// ---------------------------------------------------------------------------

/// Windows-only backslash normalisation. See [`Env::models`] for why this is
/// platform-gated instead of unconditional: on Unix `\` is a legal filename
/// character, so rewriting it there would corrupt a valid path.
fn slashify(p: &str) -> String {
    if cfg!(windows) {
        p.replace('\\', "/")
    } else {
        p.to_string()
    }
}

/// **Upstream:** Go's `strconv.ParseBool`. Exactly these twelve spellings, and
/// `None` for everything else -- note there is no `yes`/`no`/`on`/`off`, and no
/// case-insensitive match beyond the listed forms (`TRue` is **not** accepted).
fn parse_go_bool(s: &str) -> Option<bool> {
    match s {
        "1" | "t" | "T" | "TRUE" | "true" | "True" => Some(true),
        "0" | "f" | "F" | "FALSE" | "false" | "False" => Some(false),
        _ => None,
    }
}

/// **Upstream:** Go's `strconv.ParseUint(s, 10, 64)`.
///
/// Base 10, and **no sign is permitted** -- that is the documented difference
/// between `ParseUint` and `ParseInt`, and it is why upstream's `TestUint` list
/// `-1` as an invalid value. Rust's own `u64::from_str` would accept a leading
/// `+`, so the digits-only check is doing real work, not being pedantic.
/// Underscores are rejected too (Go only allow them when base is 0).
fn parse_go_uint(s: &str) -> Option<u64> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse::<u64>().ok()
}

/// Nanoseconds per unit. **Upstream:** `time`'s `unitMap`.
///
/// Both micro-sign spellings are here on purpose: `µ` is U+00B5 MICRO SIGN and
/// `μ` is U+03BC GREEK SMALL LETTER MU. They look identical and different
/// keyboards produce different ones, so Go accept both and so must we.
fn go_duration_unit(u: &str) -> Option<i64> {
    Some(match u {
        "ns" => 1,
        "us" | "\u{00b5}s" | "\u{03bc}s" => 1_000,
        "ms" => 1_000_000,
        "s" => 1_000_000_000,
        "m" => 60 * 1_000_000_000,
        "h" => 3600 * 1_000_000_000,
        _ => return None,
    })
}

/// **Upstream:** `time.leadingInt`. Digits only, error (here `None`) on
/// overflow. Returns the value and the index just past the last digit.
fn leading_int(b: &[u8], mut i: usize) -> Option<(i64, usize)> {
    let mut x: i64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        x = x.checked_mul(10)?.checked_add((b[i] - b'0') as i64)?;
        i += 1;
    }
    Some((x, i))
}

/// **Upstream:** `time.leadingFraction`.
///
/// Note it **never fails**: once the accumulator would overflow it keeps
/// consuming digits but stops accumulating *and stops growing `scale`*, so the
/// extra precision is dropped rather than the whole parse being rejected. That
/// asymmetry with [`leading_int`] is upstream's, and it is why `0.<50 digits>s`
/// parse fine.
fn leading_fraction(b: &[u8], mut i: usize) -> (i64, f64, usize) {
    let mut x: i64 = 0;
    let mut scale: f64 = 1.0;
    let mut overflow = false;
    while i < b.len() && b[i].is_ascii_digit() {
        if !overflow {
            if x > i64::MAX / 10 {
                overflow = true;
            } else {
                let y = x * 10 + (b[i] - b'0') as i64;
                if y < 0 {
                    overflow = true;
                } else {
                    x = y;
                    scale *= 10.0;
                }
            }
        }
        i += 1;
    }
    (x, scale, i)
}

/// **Upstream:** `time.ParseDuration`, returning nanoseconds.
///
/// Grammar: `[-+]?([0-9]*(\.[0-9]*)?[a-z]+)+`, with the one special case that a
/// bare `0` (or `-0`, `+0`) is accepted without a unit. Everything else
/// **must** carry a unit from [`go_duration_unit`], which is why `1d`, `1w` and
/// `1y` are rejected -- days and up are not fixed-length in Go's model, so the
/// package refuse to guess. `None` on any parse error or overflow, matching Go
/// returning an error; every caller here fall back to its own default.
fn parse_go_duration(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    let mut i = 0usize;
    let mut neg = false;
    if !b.is_empty() && (b[0] == b'-' || b[0] == b'+') {
        neg = b[0] == b'-';
        i = 1;
    }
    // "Special case: if all that is left is '0', this is zero." -- upstream.
    if &s[i..] == "0" {
        return Some(0);
    }
    if i == b.len() {
        return None;
    }

    let mut d: i64 = 0;
    while i < b.len() {
        // The next character must be [0-9.]
        if !(b[i] == b'.' || b[i].is_ascii_digit()) {
            return None;
        }
        let before_int = i;
        let (v_int, ni) = leading_int(b, i)?;
        i = ni;
        let pre = i != before_int;

        let mut frac: i64 = 0;
        let mut scale: f64 = 1.0;
        let mut post = false;
        if i < b.len() && b[i] == b'.' {
            i += 1;
            let before_frac = i;
            let (f, sc, ni) = leading_fraction(b, i);
            frac = f;
            scale = sc;
            i = ni;
            post = i != before_frac;
        }
        if !pre && !post {
            // No digits at all, e.g. ".s"
            return None;
        }

        // The unit runs until the next digit or '.'.
        let unit_start = i;
        while i < b.len() && !(b[i] == b'.' || b[i].is_ascii_digit()) {
            i += 1;
        }
        if i == unit_start {
            return None; // missing unit
        }
        let unit = go_duration_unit(&s[unit_start..i])?;

        let mut v = v_int.checked_mul(unit)?;
        if frac > 0 {
            // Upstream: `v += int64(float64(f) * (float64(unit) / scale))`
            v = v.checked_add((frac as f64 * (unit as f64 / scale)) as i64)?;
            if v < 0 {
                return None; // overflow
            }
        }
        d = d.checked_add(v)?;
        if d < 0 {
            return None; // overflow
        }
    }
    Some(if neg { -d } else { d })
}

/// **Upstream:** `net.SplitHostPort`, collapsed to `Option` because every
/// caller here only care whether it worked.
///
/// Two things that surprise people, both faithful:
///
/// * it does **not** validate the port -- `"a:b"` split happily into
///   `("a", "b")`. Port validation is [`Env::host`]'s job, afterwards;
/// * a bracketed address with no port (`"[::1]"`) is an **error**, not a host
///   with a missing port. That is exactly why [`Env::host`] has a fallback
///   branch that re-parses the bare string as an IP literal.
fn go_split_host_port(hp: &str) -> Option<(&str, &str)> {
    let b = hp.as_bytes();
    // The port starts after the last colon. An address with no colon at all is
    // an error here, which also guarantees `b` is non-empty below.
    let i = hp.rfind(':')?;

    let (host, j, k);
    if b[0] == b'[' {
        // Expect the first ']' just before the last ':'.
        let end = hp.find(']')?;
        if end + 1 == b.len() {
            return None; // missing port
        }
        if end + 1 != i {
            return None; // too many colons, or ']' not followed by the port
        }
        host = &hp[1..end];
        j = 1;
        k = end + 1;
    } else {
        host = &hp[..i];
        if host.contains(':') {
            return None; // too many colons
        }
        j = 0;
        k = 0;
    }
    if hp[j..].contains('[') {
        return None;
    }
    if hp[k..].contains(']') {
        return None;
    }
    Some((host, &hp[i + 1..]))
}

/// **Upstream:** `net.JoinHostPort`. Brackets the host iff it contain a colon,
/// which is how a literal IPv6 address survive being glued to a port.
fn go_join_host_port(host: &str, port: &str) -> String {
    if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

/// **Upstream:** `net.ParseIP`.
///
/// Rust's `IpAddr` parser agree with Go's on everything that reach here:
/// dotted-quad v4 with no leading zeros, full and `::`-compressed v6,
/// `::ffff:1.2.3.4` mapped form, and a rejection of zone suffixes (`%eth0`).
/// Delegating rather than hand-rolling is the right call -- a bespoke IP parser
/// is a bug farm, and `std` is not a dependency.
fn go_parse_ip(s: &str) -> Option<IpAddr> {
    s.parse::<IpAddr>().ok()
}

/// **Upstream:** `(net.IP).String()`.
///
/// The one place Rust and Go disagree, so it is handled here: Go's `String()`
/// call `To4()` first, and `To4()` succeed for an **IPv4-mapped** v6 address --
/// so Go print `::ffff:1.2.3.4` as plain `1.2.3.4`, while Rust's `Display` keep
/// the mapped form. Since this feeds an operator-visible bind address, we
/// follow Go. Note `::1` and `::` are *not* mapped addresses and stay in v6
/// form under both.
fn go_ip_string(ip: IpAddr) -> String {
    match ip {
        IpAddr::V4(v4) => v4.to_string(),
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => v4.to_string(),
            None => v6.to_string(),
        },
    }
}

// ---------------------------------------------------------------------------
// Tests -- upstream's own tables, ported case for case
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// An `Env` with upstream's naming, so a ported table means what it meant
    /// in Go.
    fn ollama_env(pairs: &[(&str, &str)]) -> Env {
        Env::ollama(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        )
    }

    fn with_host(value: &str) -> Env {
        ollama_env(&[("OLLAMA_HOST", value)])
    }

    // -- Host ---------------------------------------------------------------

    /// Ported from upstream `TestHost`, every case.
    #[test]
    fn the_host_table_matches_upstream_case_for_case() {
        let cases: &[(&str, &str, &str)] = &[
            ("empty", "", "http://127.0.0.1:11434"),
            ("only address", "1.2.3.4", "http://1.2.3.4:11434"),
            ("only port", ":1234", "http://:1234"),
            ("address and port", "1.2.3.4:1234", "http://1.2.3.4:1234"),
            ("hostname", "example.com", "http://example.com:11434"),
            (
                "hostname and port",
                "example.com:1234",
                "http://example.com:1234",
            ),
            ("zero port", ":0", "http://:0"),
            ("too large port", ":66000", "http://:11434"),
            ("too small port", ":-1", "http://:11434"),
            ("ipv6 localhost", "[::1]", "http://[::1]:11434"),
            ("ipv6 world open", "[::]", "http://[::]:11434"),
            ("ipv6 no brackets", "::1", "http://[::1]:11434"),
            ("ipv6 + port", "[::1]:1337", "http://[::1]:1337"),
            ("extra space", " 1.2.3.4 ", "http://1.2.3.4:11434"),
            ("extra quotes", "\"1.2.3.4\"", "http://1.2.3.4:11434"),
            (
                "extra space+quotes",
                " \" 1.2.3.4 \" ",
                "http://1.2.3.4:11434",
            ),
            ("extra single quotes", "'1.2.3.4'", "http://1.2.3.4:11434"),
            ("http", "http://1.2.3.4", "http://1.2.3.4:80"),
            ("http port", "http://1.2.3.4:4321", "http://1.2.3.4:4321"),
            ("https", "https://1.2.3.4", "https://1.2.3.4:443"),
            ("https port", "https://1.2.3.4:4321", "https://1.2.3.4:4321"),
            (
                "proxy path",
                "https://example.com/ollama",
                "https://example.com:443/ollama",
            ),
            ("ollama.com", "ollama.com", "https://ollama.com:443"),
        ];
        for (name, value, expect) in cases {
            assert_eq!(with_host(value).host().to_string(), *expect, "case {name}");
        }
    }

    /// Ported from upstream `TestConnectableHost`, every case. The point of
    /// this function: `0.0.0.0` is bindable but not dialable, and on Windows
    /// dialing it fails outright.
    #[test]
    fn an_unspecified_bind_address_becomes_loopback_for_clients() {
        let cases: &[(&str, &str)] = &[
            ("", "http://127.0.0.1:11434"),
            ("127.0.0.1", "http://127.0.0.1:11434"),
            ("127.0.0.1:1234", "http://127.0.0.1:1234"),
            ("0.0.0.0", "http://127.0.0.1:11434"),
            ("0.0.0.0:1234", "http://127.0.0.1:1234"),
            ("[::]", "http://[::1]:11434"),
            ("[::]:1234", "http://[::1]:1234"),
            ("[::1]", "http://[::1]:11434"),
            ("[::1]:1234", "http://[::1]:1234"),
            ("192.168.1.5", "http://192.168.1.5:11434"),
            ("192.168.1.5:8080", "http://192.168.1.5:8080"),
            ("example.com", "http://example.com:11434"),
            ("example.com:1234", "http://example.com:1234"),
            ("https://0.0.0.0:4321", "https://127.0.0.1:4321"),
        ];
        for (value, expect) in cases {
            assert_eq!(
                with_host(value).connectable_host().to_string(),
                *expect,
                "case {value:?}"
            );
        }
    }

    /// The trap worth naming out loud: the default port depend on whether a
    /// scheme was typed, so these two are NOT the same host.
    #[test]
    fn a_bare_address_and_an_http_address_get_different_default_ports() {
        assert_eq!(
            with_host("1.2.3.4").host().to_string(),
            "http://1.2.3.4:11434"
        );
        assert_eq!(
            with_host("http://1.2.3.4").host().to_string(),
            "http://1.2.3.4:80"
        );
    }

    // -- Origins ------------------------------------------------------------

    fn builtin_origins() -> Vec<String> {
        [
            "http://localhost",
            "https://localhost",
            "http://localhost:*",
            "https://localhost:*",
            "http://127.0.0.1",
            "https://127.0.0.1",
            "http://127.0.0.1:*",
            "https://127.0.0.1:*",
            "http://0.0.0.0",
            "https://0.0.0.0",
            "http://0.0.0.0:*",
            "https://0.0.0.0:*",
            "app://*",
            "file://*",
            "tauri://*",
            "vscode-webview://*",
            "vscode-file://*",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Ported from upstream `TestOrigins`, every case. What it pin is that the
    /// configured origins go **first** and the built-ins are **always** there.
    #[test]
    fn configured_origins_are_prepended_and_the_builtins_always_survive() {
        let cases: &[(&str, &[&str])] = &[
            ("", &[]),
            ("http://10.0.0.1", &["http://10.0.0.1"]),
            (
                "http://172.16.0.1,https://192.168.0.1",
                &["http://172.16.0.1", "https://192.168.0.1"],
            ),
            (
                "http://totally.safe,http://definitely.legit",
                &["http://totally.safe", "http://definitely.legit"],
            ),
        ];
        for (value, configured) in cases {
            let mut expect: Vec<String> = configured.iter().map(|s| s.to_string()).collect();
            expect.extend(builtin_origins());
            assert_eq!(
                ollama_env(&[("OLLAMA_ORIGINS", value)]).allowed_origins(),
                expect,
                "case {value:?}"
            );
        }
    }

    /// Looks like a bug, is upstream's behaviour: `strings.Split` do not trim,
    /// so a space after the comma become part of the origin and that origin
    /// will never match a real `Origin` header.
    #[test]
    fn origins_are_split_on_comma_without_trimming() {
        let got = ollama_env(&[("OLLAMA_ORIGINS", "http://a, http://b")]).allowed_origins();
        assert_eq!(&got[..2], &["http://a".to_string(), " http://b".to_string()]);
    }

    // -- Bool / Uint --------------------------------------------------------

    /// Ported from upstream `TestBool`. The last two cases are the important
    /// ones: an unparseable value read as **true**.
    #[test]
    fn an_unparseable_boolean_reads_as_true() {
        let cases: &[(&str, bool)] = &[
            ("", false),
            ("true", true),
            ("false", false),
            ("1", true),
            ("0", false),
            ("random", true),
            ("something", true),
        ];
        for (value, expect) in cases {
            assert_eq!(
                ollama_env(&[("OLLAMA_BOOL", value)]).bool_var("BOOL"),
                *expect,
                "case {value:?}"
            );
        }
    }

    /// Upstream's own idiom for telling "explicitly off" from "never set" --
    /// `llm/llama_server.go:608`.
    #[test]
    fn asking_twice_with_opposite_defaults_reveals_whether_it_was_set() {
        let unset = ollama_env(&[]);
        assert!(!unset.bool_var_is_set("FLASH_ATTENTION"));
        assert!(!unset.flash_attention(false));
        assert!(unset.flash_attention(true));

        let off = ollama_env(&[("OLLAMA_FLASH_ATTENTION", "false")]);
        assert!(off.bool_var_is_set("FLASH_ATTENTION"));
        assert!(
            !off.flash_attention(true),
            "an explicit false must beat the default"
        );
    }

    /// Ported from upstream `TestUint`. Base 10 only -- note `-1`, `0o10` and
    /// `0x10` all fall back to the default rather than erroring.
    #[test]
    fn an_unsigned_int_falls_back_to_its_default_on_anything_non_decimal() {
        let cases: &[(&str, u64)] = &[
            ("0", 0),
            ("1", 1),
            ("1337", 1337),
            ("", 11434),
            ("-1", 11434),
            ("0o10", 11434),
            ("0x10", 11434),
            ("string", 11434),
        ];
        for (value, expect) in cases {
            assert_eq!(
                ollama_env(&[("OLLAMA_UINT", value)]).uint("UINT", 11434),
                *expect,
                "case {value:?}"
            );
        }
    }

    /// Rust's own `u64::from_str` would take `+5`; Go's `ParseUint` refuse a
    /// sign entirely, so we must too.
    #[test]
    fn a_signed_looking_unsigned_int_is_rejected_like_go_does() {
        assert_eq!(ollama_env(&[("OLLAMA_UINT", "+5")]).uint("UINT", 7), 7);
        assert_eq!(ollama_env(&[("OLLAMA_UINT", "1_000")]).uint("UINT", 7), 7);
    }

    // -- Durations ----------------------------------------------------------

    fn keep_alive_of(value: &str) -> i64 {
        ollama_env(&[("OLLAMA_KEEP_ALIVE", value)])
            .keep_alive()
            .as_nanos_i64()
    }

    /// Ported from upstream `TestKeepAlive`, every case, compared in
    /// nanoseconds so the numbers line up with Go's.
    #[test]
    fn the_keep_alive_table_matches_upstream_case_for_case() {
        const SEC: i64 = 1_000_000_000;
        let cases: &[(&str, i64)] = &[
            ("", 5 * 60 * SEC),
            ("1s", SEC),
            ("1m", 60 * SEC),
            ("1h", 3600 * SEC),
            ("5m0s", 5 * 60 * SEC),
            ("1h2m3s", 3600 * SEC + 2 * 60 * SEC + 3 * SEC),
            ("0", 0),
            ("60", 60 * SEC),
            ("120", 2 * 60 * SEC),
            ("3600", 3600 * SEC),
            ("-0", 0),
            ("-1", i64::MAX),
            ("-1m", i64::MAX),
            // invalid values fall back to the 5 minute default
            (" ", 5 * 60 * SEC),
            ("???", 5 * 60 * SEC),
            ("1d", 5 * 60 * SEC),
            ("1y", 5 * 60 * SEC),
            ("1w", 5 * 60 * SEC),
        ];
        for (value, expect) in cases {
            assert_eq!(keep_alive_of(value), *expect, "case {value:?}");
        }
    }

    fn load_timeout_of(value: &str) -> i64 {
        ollama_env(&[("OLLAMA_LOAD_TIMEOUT", value)])
            .load_timeout()
            .as_nanos_i64()
    }

    /// Ported from upstream `TestLoadTimeout`, every case.
    #[test]
    fn the_load_timeout_table_matches_upstream_case_for_case() {
        const SEC: i64 = 1_000_000_000;
        let cases: &[(&str, i64)] = &[
            ("", 5 * 60 * SEC),
            ("1s", SEC),
            ("1m", 60 * SEC),
            ("1h", 3600 * SEC),
            ("5m0s", 5 * 60 * SEC),
            ("1h2m3s", 3600 * SEC + 2 * 60 * SEC + 3 * SEC),
            ("0", i64::MAX),
            ("60", 60 * SEC),
            ("120", 2 * 60 * SEC),
            ("3600", 3600 * SEC),
            ("-0", i64::MAX),
            ("-1", i64::MAX),
            ("-1m", i64::MAX),
            (" ", 5 * 60 * SEC),
            ("???", 5 * 60 * SEC),
            ("1d", 5 * 60 * SEC),
            ("1y", 5 * 60 * SEC),
            ("1w", 5 * 60 * SEC),
        ];
        for (value, expect) in cases {
            assert_eq!(load_timeout_of(value), *expect, "case {value:?}");
        }
    }

    /// The single difference between the two duration vars, stated on its own
    /// so nobody "tidies" them into one function: at zero they mean opposite
    /// things.
    #[test]
    fn zero_means_unload_now_for_keep_alive_but_wait_forever_for_load_timeout() {
        assert_eq!(
            ollama_env(&[("OLLAMA_KEEP_ALIVE", "0")]).keep_alive(),
            Expiry::After(Duration::ZERO)
        );
        assert_eq!(
            ollama_env(&[("OLLAMA_LOAD_TIMEOUT", "0")]).load_timeout(),
            Expiry::Never
        );
    }

    /// Go duration grammar corners that the env tables do not reach.
    #[test]
    fn go_duration_parsing_handles_fractions_signs_and_both_micro_signs() {
        assert_eq!(parse_go_duration("1.5h"), Some(5400 * 1_000_000_000));
        assert_eq!(parse_go_duration("300ms"), Some(300_000_000));
        assert_eq!(parse_go_duration("100ns"), Some(100));
        assert_eq!(parse_go_duration("2us"), Some(2_000));
        assert_eq!(parse_go_duration("2\u{00b5}s"), Some(2_000), "MICRO SIGN");
        assert_eq!(parse_go_duration("2\u{03bc}s"), Some(2_000), "GREEK MU");
        assert_eq!(parse_go_duration("+1m"), Some(60 * 1_000_000_000));
        assert_eq!(parse_go_duration("1h30m"), Some(5400 * 1_000_000_000));
        // A unit is mandatory except for a bare zero.
        assert_eq!(parse_go_duration("0"), Some(0));
        assert_eq!(parse_go_duration("+0"), Some(0));
        assert_eq!(parse_go_duration("1"), None);
        assert_eq!(parse_go_duration(""), None);
        assert_eq!(parse_go_duration("-"), None);
        assert_eq!(parse_go_duration(".s"), None);
        assert_eq!(parse_go_duration("1d"), None, "days are not a Go unit");
    }

    // -- Var ----------------------------------------------------------------

    /// Ported from upstream `TestVar`. The order matters: whitespace is
    /// stripped **before** the quotes, so quoted inner spaces survive.
    #[test]
    fn a_variable_is_trimmed_of_space_then_of_quotes_in_that_order() {
        let cases: &[(&str, &str)] = &[
            ("value", "value"),
            (" value ", "value"),
            (" 'value' ", "value"),
            (" \"value\" ", "value"),
            (" ' value ' ", " value "),
            (" \" value \" ", " value "),
        ];
        for (value, expect) in cases {
            assert_eq!(
                ollama_env(&[("OLLAMA_VAR", value)]).raw("OLLAMA_VAR"),
                *expect,
                "case {value:?}"
            );
        }
    }

    // -- LogLevel / ContextLength -------------------------------------------

    /// Ported from upstream `TestLogLevel`. Note `-1` makes it **quieter**.
    #[test]
    fn the_log_level_table_matches_upstream_case_for_case() {
        let cases: &[(&str, LogLevel)] = &[
            ("", LogLevel::INFO),
            ("false", LogLevel::INFO),
            ("f", LogLevel::INFO),
            ("0", LogLevel::INFO),
            ("true", LogLevel::DEBUG),
            ("t", LogLevel::DEBUG),
            ("1", LogLevel::DEBUG),
            ("2", LogLevel::TRACE),
            ("-1", LogLevel::WARN),
            ("-2", LogLevel::ERROR),
        ];
        for (value, expect) in cases {
            assert_eq!(
                ollama_env(&[("OLLAMA_DEBUG", value)]).log_level(),
                *expect,
                "case {value:?}"
            );
        }
    }

    /// Ported from upstream `TestContextLength`. `0` mean "decide from VRAM",
    /// which is the same contract `options::Runner::num_ctx` carry.
    #[test]
    fn context_length_defaults_to_zero_meaning_decide_from_vram() {
        assert_eq!(ollama_env(&[]).context_length(), 0);
        assert_eq!(
            ollama_env(&[("OLLAMA_CONTEXT_LENGTH", "2048")]).context_length(),
            2048
        );
        assert_eq!(
            crate::options::Runner::default().num_ctx,
            0,
            "must agree with Runner"
        );
    }

    // -- Defaults sweep -----------------------------------------------------

    /// Every default in one place. If one of these move, it moved because
    /// somebody changed a number, and that number decide runtime behaviour.
    #[test]
    fn every_default_matches_the_upstream_line_it_came_from() {
        let e = ollama_env(&[]);
        assert_eq!(e.num_parallel(), 1, "Uint(OLLAMA_NUM_PARALLEL, 1)");
        assert_eq!(e.max_runners(), 0, "Uint(OLLAMA_MAX_LOADED_MODELS, 0)");
        assert_eq!(e.max_queue(), 512, "Uint(OLLAMA_MAX_QUEUE, 512)");
        assert_eq!(
            e.max_transfer_streams(),
            4,
            "Uint(OLLAMA_MAX_TRANSFER_STREAMS, 4)"
        );
        assert_eq!(e.gpu_overhead(), 0, "Uint64(OLLAMA_GPU_OVERHEAD, 0)");
        assert_eq!(e.context_length(), 0, "Uint(OLLAMA_CONTEXT_LENGTH, 0)");
        assert_eq!(e.keep_alive(), Expiry::After(Duration::from_secs(300)));
        assert_eq!(e.load_timeout(), Expiry::After(Duration::from_secs(300)));
        assert_eq!(e.log_level(), LogLevel::INFO);
        assert_eq!(e.remotes(), vec!["ollama.com".to_string()]);
        assert_eq!(e.kv_cache_type(), "", "empty means f16");
        assert_eq!(e.llm_library(), "");
        assert_eq!(e.editor(), "");
        assert!(!e.debug_log_requests());
        assert!(!e.no_history());
        assert!(!e.no_prune());
        assert!(!e.sched_spread());
        assert!(!e.use_auth());
        assert!(!e.no_cloud_env());
        assert_eq!(e.host().to_string(), "http://127.0.0.1:11434");
    }

    #[test]
    fn remotes_split_on_comma_and_default_to_the_one_upstream_host() {
        assert_eq!(ollama_env(&[]).remotes(), vec!["ollama.com".to_string()]);
        assert_eq!(
            ollama_env(&[("OLLAMA_REMOTES", "a.example,b.example")]).remotes(),
            vec!["a.example".to_string(), "b.example".to_string()]
        );
    }

    // -- The KOPITIAM seam --------------------------------------------------

    /// The divergence that matters: by default we read `KOPITIAM_*` and ignore
    /// `OLLAMA_*` entirely. KOPITIAM must not inherit another product's
    /// configuration by accident.
    #[test]
    fn by_default_we_read_our_own_prefix_and_ignore_ollamas() {
        let vars: BTreeMap<String, String> = [
            ("OLLAMA_NUM_PARALLEL".to_string(), "8".to_string()),
            ("KOPITIAM_NUM_PARALLEL".to_string(), "4".to_string()),
        ]
        .into_iter()
        .collect();

        assert_eq!(Env::new(vars.clone()).num_parallel(), 4);
        assert_eq!(Env::ollama(vars).num_parallel(), 8);

        // Ours only -- ollama's is invisible.
        let only_theirs: BTreeMap<String, String> =
            [("OLLAMA_NUM_PARALLEL".to_string(), "8".to_string())]
                .into_iter()
                .collect();
        assert_eq!(
            Env::new(only_theirs).num_parallel(),
            1,
            "an OLLAMA_ var must not configure KOPITIAM"
        );
    }

    /// Opt-in compatibility: ours first, theirs as a fallback.
    #[test]
    fn a_prefix_chain_lets_ours_win_while_theirs_still_works() {
        let both: BTreeMap<String, String> = [
            ("OLLAMA_MAX_QUEUE".to_string(), "8".to_string()),
            ("KOPITIAM_MAX_QUEUE".to_string(), "4".to_string()),
        ]
        .into_iter()
        .collect();
        let chained = Env::new(both).with_prefixes(["KOPITIAM_", "OLLAMA_"]);
        assert_eq!(chained.max_queue(), 4);

        let theirs_only: BTreeMap<String, String> =
            [("OLLAMA_MAX_QUEUE".to_string(), "8".to_string())]
                .into_iter()
                .collect();
        assert_eq!(
            Env::new(theirs_only)
                .with_prefixes(["KOPITIAM_", "OLLAMA_"])
                .max_queue(),
            8
        );
    }

    /// An empty value is "unset", everywhere -- upstream's `if s != ""` idiom
    /// turned into a rule.
    #[test]
    fn an_empty_value_counts_as_unset_and_falls_through() {
        let vars: BTreeMap<String, String> = [
            ("KOPITIAM_MAX_QUEUE".to_string(), String::new()),
            ("OLLAMA_MAX_QUEUE".to_string(), "9".to_string()),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            Env::new(vars)
                .with_prefixes(["KOPITIAM_", "OLLAMA_"])
                .max_queue(),
            9
        );
    }

    // -- Models / home ------------------------------------------------------

    #[test]
    fn the_models_dir_defaults_under_the_home_dir_with_forward_slashes() {
        let unix = Env::new(
            [(
                "HOME".to_string(),
                "/data/data/com.termux/files/home".to_string(),
            )]
            .into_iter()
            .collect(),
        );
        assert_eq!(
            unix.models().unwrap(),
            "/data/data/com.termux/files/home/.kopitiam/models"
        );

        // Upstream naming gives back upstream's path, which is how a shared
        // model store with an existing ollama install is arranged.
        let shared = Env::ollama(
            [("HOME".to_string(), "/home/theo".to_string())]
                .into_iter()
                .collect(),
        );
        assert_eq!(shared.models().unwrap(), "/home/theo/.ollama/models");
    }

    #[test]
    fn an_explicit_models_dir_wins_over_the_home_dir() {
        let e = Env::new(
            [
                ("HOME".to_string(), "/home/theo".to_string()),
                ("KOPITIAM_MODELS".to_string(), "/mnt/big/models".to_string()),
            ]
            .into_iter()
            .collect(),
        );
        assert_eq!(e.models().unwrap(), "/mnt/big/models");
    }

    /// No `HOME`, no `USERPROFILE` -- a normal state on Termux and in stripped
    /// containers. Upstream panics; we hand the problem back.
    #[test]
    fn a_missing_home_is_an_error_not_a_panic() {
        let e = Env::new(BTreeMap::new());
        assert!(matches!(e.models(), Err(EnvConfigError::NoHomeDir(_))));
        // ...and it stops being a problem the moment the caller is explicit.
        assert_eq!(
            e.with_var("KOPITIAM_MODELS", "/srv/models")
                .models()
                .unwrap(),
            "/srv/models"
        );
    }

    /// Windows path handling. The forward-slash rule is a KOPITIAM one (mixed
    /// separators in emitted paths are a known rough edge), so it is asserted
    /// on Windows and the backslash is left alone elsewhere.
    #[test]
    fn windows_paths_come_back_forward_slashed() {
        let e = Env::new(
            [("USERPROFILE".to_string(), "C:\\Users\\theo".to_string())]
                .into_iter()
                .collect(),
        );
        if cfg!(windows) {
            assert_eq!(e.models().unwrap(), "C:/Users/theo/.kopitiam/models");
            assert!(!e.models().unwrap().contains('\\'));
        } else {
            // On Unix `USERPROFILE` is only the fallback, and a backslash is a
            // legal filename character, so it must survive untouched.
            assert_eq!(e.models().unwrap(), "C:\\Users\\theo/.kopitiam/models");
        }
    }

    // -- server.json / no-cloud ---------------------------------------------

    /// Ported from upstream `TestNoCloud`, including the malformed-JSON case.
    #[test]
    fn cloud_can_be_disabled_by_the_env_the_file_or_both() {
        let on = ollama_env(&[]);
        let via_env = ollama_env(&[("OLLAMA_NO_CLOUD", "1")]);
        let none = ServerConfig::default();
        let disabled = ServerConfig::parse(r#"{"disable_ollama_cloud": true}"#);
        let enabled = ServerConfig::parse(r#"{"disable_ollama_cloud": false}"#);
        let broken = ServerConfig::parse("{invalid json");

        assert!(!on.no_cloud(&none));
        assert_eq!(on.no_cloud_source(&none).as_str(), "none");

        assert!(via_env.no_cloud(&none));
        assert_eq!(via_env.no_cloud_source(&none).as_str(), "env");

        assert!(on.no_cloud(&disabled));
        assert_eq!(on.no_cloud_source(&disabled).as_str(), "config");

        assert!(via_env.no_cloud(&disabled));
        assert_eq!(via_env.no_cloud_source(&disabled).as_str(), "both");

        assert!(!on.no_cloud(&enabled));
        assert_eq!(on.no_cloud_source(&enabled).as_str(), "none");

        // Malformed config is ignored, not fatal -- the runtime must still boot.
        assert!(!on.no_cloud(&broken));
        assert_eq!(on.no_cloud_source(&broken).as_str(), "none");
    }

    // -- Docs table ---------------------------------------------------------

    #[test]
    fn every_documented_variable_renders_under_the_configured_prefix() {
        let kopitiam = Env::default();
        let ollama = Env::ollama(BTreeMap::new());
        let host = VAR_DOCS.iter().find(|d| d.suffix == "HOST").unwrap();
        assert_eq!(kopitiam.full_name(host), "KOPITIAM_HOST");
        assert_eq!(ollama.full_name(host), "OLLAMA_HOST");

        let proxy = VAR_DOCS.iter().find(|d| d.suffix == "HTTP_PROXY").unwrap();
        assert_eq!(
            kopitiam.full_name(proxy),
            "HTTP_PROXY",
            "someone else's variable keeps its own name"
        );

        assert!(
            VAR_DOCS.iter().all(|d| !d.description.is_empty()),
            "a knob with no explanation is a knob nobody can use"
        );
    }

    // -- Unprefixed device vars ---------------------------------------------

    #[test]
    fn device_visibility_variables_are_read_unprefixed() {
        let e = Env::new(
            [
                ("CUDA_VISIBLE_DEVICES".to_string(), "0,1".to_string()),
                ("HSA_OVERRIDE_GFX_VERSION".to_string(), "10.3.0".to_string()),
            ]
            .into_iter()
            .collect(),
        );
        assert_eq!(e.cuda_visible_devices(), "0,1");
        assert_eq!(e.hsa_override_gfx_version(), "10.3.0");
        assert_eq!(e.hip_visible_devices(), "");
        assert_eq!(e.rocr_visible_devices(), "");
        assert_eq!(e.vk_visible_devices(), "");
        assert_eq!(e.gpu_device_ordinal(), "");
    }

    /// The one accessor that touches the real process environment. It must not
    /// explode, and it must be reading OUR prefix.
    #[test]
    fn from_env_snapshots_the_real_environment_under_our_prefix() {
        let e = Env::from_env();
        assert_eq!(e.prefixes(), &["KOPITIAM_".to_string()]);
        // No assertion on the values -- they are whatever the machine has.
        // What matters is that building it is infallible.
        let _ = e.host();
    }
}
