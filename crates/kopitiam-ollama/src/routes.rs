//! The HTTP API surface -- request/response types, the route table, and the
//! request-handling logic, minus the server.
//!
//! **Upstream:** `api/types.go` (the wire types) and `server/routes.go` (the
//! endpoint set and every validation / defaulting / error-mapping decision),
//! ollama, MIT, pinned at `4713800b08b2ddf5e14acf8398953cf7b12f169b`.
//!
//! ## Why there is no server in here (read this first)
//!
//! Upstream `routes.go` is gin: `c.ShouldBindJSON`, `c.JSON(status, ...)`,
//! `c.Stream(...)`, goroutines and channels, all tangled together with the
//! actual decisions. **This module deliberately keep only the decisions.** No
//! axum, no hyper, no tokio, no async at all -- this crate got no runtime and
//! picking one is an architecture call that belong to KOPITIAM, not to a port.
//!
//! So the shape here is:
//!
//! ```text
//!   (method, path) --> match_route()  --> Route          "which endpoint?"
//!   Route + typed request             --> handle_*()     "what should happen?"
//!   handle_*() returns                --> Disposition    "do this, reply that"
//! ```
//!
//! A [`Route`] is a plain enum. [`match_route`] is a pure function -- it even
//! tell you 405-vs-404, because upstream set gin's `HandleMethodNotAllowed`.
//! Every handler is a pure function taking a typed request (plus a
//! [`ModelCatalog`] when it need to look a model up) and returning either a
//! typed response or a [`RouteError`] carrying **the exact status code and the
//! exact message string** upstream would have written. Nothing here open a
//! socket, spawn a task, or touch a model.
//!
//! Whatever real server KOPITIAM eventually picks is then a **thin adapter**:
//! read the body, `serde_json::from_slice`, call `match_route`, call the
//! handler, write the status + JSON back. That is the whole point -- the part
//! that is a contract with every ollama client on earth is testable with no
//! network, and the part that is a framework choice stays deferrable.
//!
//! ## What is deliberately NOT ported
//!
//! * **The inference bodies.** Once a request is validated, upstream hands off
//!   to a scheduler, a runner, a tokenizer and a template. Those live
//!   elsewhere (some still being written), so the handlers here stop at
//!   "validated, here is the plan" -- see [`GenerateDisposition`] /
//!   [`ChatDisposition`]. The *decisions before* that point are all here,
//!   because those are the ones clients can observe.
//! * **Cloud passthrough** (`modelSourceCloud`, `proxyCloudJSONRequest`,
//!   `/api/me`, `/api/signout`, the `x/server` bits). Needs an HTTP client and
//!   a network; KOPITIAM is offline-first. The *routes* are still in [`Route`]
//!   so the table is complete and honest, they just got no handler here.
//! * **`GenerateRoutes`' CORS + allowed-hosts middleware.** Pure transport
//!   policy; belong to the adapter that own the listener.
//! * **`Options` / `Runner` / `DefaultOptions` / `FromMap`** -- already ported,
//!   properly, in [`crate::options`]. **`Message` / `Tool` / `ToolCall` /
//!   `ThinkValue` / `Capability` / `ConfigV2`** -- already in [`crate::api`].
//!   This module re-use them, never redefine them.
//! * **`UserResponse`, `WebSearch*`, `WebFetch*`, `ModelRecommendation*`,
//!   `CloudStatus`/`StatusResponse`, `TokenResponse`, `AuthorizationError`** --
//!   all cloud-account types. Left out with the cloud code they serve.
//!
//! ## Two wire-format traps that bite, hor
//!
//! 1. **`api.Duration` and `time.Duration` are NOT the same on the wire.**
//!    [`Duration`] (the `keep_alive` type) has custom marshalling and comes out
//!    as a Go duration *string* (`"5m0s"`), or the bare number `-1`. A plain
//!    `time.Duration` -- every field in [`Metrics`] -- is just an `int64`, so it
//!    marshals as a **nanosecond integer**. Mixing these up silently breaks
//!    `keep_alive` or silently breaks every timing number. See [`Duration`].
//! 2. **Go's `omitempty` does nothing on a struct.** `Details ModelDetails
//!    json:"details,omitempty"` and `ModifiedAt time.Time
//!    json:"modified_at,omitempty"` are **always emitted**, tag
//!    notwithstanding, because `encoding/json` only honour `omitempty` for
//!    empty-able kinds (string, numeric, bool, pointer, map, slice, interface).
//!    Anybody "tidying" a `skip_serializing_if` onto those fields is changing
//!    the wire format. There is a test that fails if they do.
//!
//! ## One divergence, stated once: nil slice vs empty slice
//!
//! Go tells `null` (nil slice) apart from `[]` (empty slice); Rust's `Vec`
//! cannot. Where a field has no `omitempty` and upstream might leave it nil --
//! `EmbeddingResponse.embedding`, `ListResponse.models`, `ModelDetails.families`
//! -- upstream may emit `null` and we always emit `[]`. We accept **both** on
//! the way in, so no client breaks. Modelling every such field as
//! `Option<Vec<_>>` would poison the whole API to preserve a distinction no
//! ollama client actually reads. Where the distinction *does* carry meaning
//! (`ToolFunctionParameters::properties`) [`crate::api`] already keeps it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::api::{Capability, Message, ThinkValue, Tool, ToolCall};
use crate::envconfig::Expiry;
use crate::name::Name;

// ===========================================================================
// Duration -- the keep_alive type, and the reason this module has tests
// ===========================================================================

/// A duration on the wire, in **nanoseconds**, signed.
///
/// **Upstream:** `api.Duration`, which wrap a `time.Duration` purely to hang a
/// custom `MarshalJSON`/`UnmarshalJSON` off it. `keep_alive` ride on this type,
/// which is why it is worth the care.
///
/// Signed `i64` nanoseconds, not [`std::time::Duration`], because the sign is
/// **load-bearing on the wire**: a negative value marshals as the bare number
/// `-1` and Rust's `Duration` is unsigned so it cannot carry that. Use
/// [`Duration::as_expiry`] to hand it to anything that thinks in
/// [`crate::envconfig::Expiry`].
///
/// ## The marshalling, exactly (`(Duration).MarshalJSON`)
///
/// | value | JSON |
/// |---|---|
/// | negative (any) | `-1` -- a bare **number**, not a string |
/// | anything else | a **Go duration string**, e.g. `"5m0s"`, `"42s"`, `"0s"` |
///
/// ## The unmarshalling, exactly (`(*Duration).UnmarshalJSON`)
///
/// | JSON | becomes |
/// |---|---|
/// | number `>= 0` | that many **seconds** (`42` -> 42s, `42.5` -> 42.5s) |
/// | number `< 0` | [`i64::MAX`] ns -- "never expire", ~292 years |
/// | string | `time.ParseDuration`, and if the result is negative, [`i64::MAX`] |
/// | anything else | error |
///
/// So it is **asymmetric on purpose**: `-1` in means "forever", but "forever"
/// out is the string `"2562047h47m16.854775807s"`, not `-1`. Upstream's own
/// `TestDurationMarshalUnmarshal` asserts exactly that round trip.
///
/// **What would make this wrong:** emitting seconds instead of a Go duration
/// string (every ollama client parses `keep_alive` back with
/// `time.ParseDuration`, so `"300"` is a parse error to them); or treating an
/// incoming bare number as nanoseconds instead of seconds (`keep_alive: 5`
/// would become 5ns and the model would unload instantly).
///
/// One upstream line deliberately **not** ported: `UnmarshalJSON` assigns
/// `5 * time.Minute` before its `switch`. Every branch either overwrites it or
/// returns an error, so that default is unobservable -- porting it would just
/// be copying dead code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Duration(pub i64);

impl Duration {
    /// Nanoseconds per second, spelled out because it appears in the seconds
    /// conversion below and a wrong power of ten here is a silent 1000x bug.
    const NANOS_PER_SEC: f64 = 1_000_000_000.0;

    /// The "never expire" sentinel. **Upstream:** `time.Duration(math.MaxInt64)`,
    /// about 292 years -- close enough to forever for a model server lah.
    pub const FOREVER: Duration = Duration(i64::MAX);

    pub fn from_secs(secs: i64) -> Self {
        Duration(secs.saturating_mul(1_000_000_000))
    }

    pub fn as_nanos(self) -> i64 {
        self.0
    }

    /// Is this the "never unload" sentinel?
    pub fn is_forever(self) -> bool {
        self.0 == i64::MAX
    }

    /// Bridge to [`crate::envconfig::Expiry`], which is what the env-var side of
    /// the port speak. A negative duration cannot survive the trip (Rust's
    /// `Duration` is unsigned), so it clamps to zero -- and that is fine,
    /// because the only way a negative value reaches here is somebody
    /// constructing it by hand: everything arriving over the wire has already
    /// been mapped to [`Duration::FOREVER`] by the deserializer below.
    pub fn as_expiry(self) -> Expiry {
        if self.is_forever() {
            Expiry::Never
        } else {
            Expiry::After(std::time::Duration::from_nanos(self.0.max(0) as u64))
        }
    }

    /// Render like Go's `time.Duration.String()` -- `"5m0s"`, `"1.5s"`,
    /// `"300ms"`, `"0s"`. See [`go_duration_string`].
    pub fn to_go_string(self) -> String {
        go_duration_string(self.0)
    }
}

impl std::fmt::Display for Duration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_go_string())
    }
}

impl Serialize for Duration {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // Upstream: `if d.Duration < 0 { return []byte("-1") }` -- a NUMBER, and
        // always exactly -1 no matter how negative the value was.
        if self.0 < 0 {
            s.serialize_i64(-1)
        } else {
            s.serialize_str(&go_duration_string(self.0))
        }
    }
}

impl<'de> Deserialize<'de> for Duration {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;

        // Upstream unmarshals into `any` and switches on the dynamic type, so a
        // JSON number and a JSON string are BOTH accepted and everything else
        // is an error. `serde_json::Value` is the same trick.
        let v = serde_json::Value::deserialize(d)?;
        match v {
            serde_json::Value::Number(n) => {
                let t = n
                    .as_f64()
                    .ok_or_else(|| D::Error::custom("keep_alive: not a number"))?;
                if t < 0.0 {
                    // "negative means never unload"
                    Ok(Duration::FOREVER)
                } else {
                    // Upstream: `time.Duration(t * float64(time.Second))` -- a
                    // bare number is SECONDS, not nanoseconds.
                    Ok(Duration((t * Duration::NANOS_PER_SEC) as i64))
                }
            }
            serde_json::Value::String(s) => {
                let ns = parse_go_duration(&s).ok_or_else(|| {
                    // Upstream returns time.ParseDuration's own error here.
                    D::Error::custom(format!("time: invalid duration {s:?}"))
                })?;
                Ok(if ns < 0 { Duration::FOREVER } else { Duration(ns) })
            }
            // Upstream: `fmt.Errorf("Unsupported type: '%s'", reflect.TypeOf(v))`.
            other => Err(D::Error::custom(format!(
                "Unsupported type: '{}'",
                json_type_name(&other)
            ))),
        }
    }
}

/// Go's `reflect.TypeOf` name for a value that came out of `encoding/json`'s
/// `any` unmarshalling. Only used to reproduce one error string.
fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "<nil>",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "float64",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "[]interface {}",
        serde_json::Value::Object(_) => "map[string]interface {}",
    }
}

// ---------------------------------------------------------------------------
// Go duration formatting -- ported from the Go standard library
// ---------------------------------------------------------------------------

/// Format nanoseconds the way Go's `time.Duration.String()` do.
///
/// **Upstream:** the Go standard library, `src/time/time.go` -- `(Duration).format`
/// with its `fmtFrac` / `fmtInt` helpers. Go's stdlib is **BSD-3-Clause**
/// (Copyright (c) 2009 The Go Authors), which absorb into an AGPL-3.0-only work
/// so long as the notice travel with the code -- which is what this comment is.
/// This is a *translation*, not clean-room study, so it is named here at the
/// point of use as CLAUDE.md require.
///
/// Ported because [`Duration`]'s `MarshalJSON` calls it directly, so `keep_alive`
/// on the wire is literally whatever this function produce. Nothing in Rust's
/// std matches it: `std::time::Duration`'s `Debug` gives `"300ms"` and `"1.5s"`
/// but `"300s"` where Go gives `"5m0s"`, so it cannot be substituted.
///
/// The grammar, and why it look like that:
///
/// * **Under a second**, Go switch to a small unit and print one number:
///   `"999ns"`, `"1.5µs"`, `"300ms"`. The micro sign is U+00B5 MICRO SIGN,
///   which is what Go emit -- **not** U+03BC GREEK SMALL LETTER MU. They look
///   identical and are different bytes; the parser accept both (see
///   `go_duration_unit`) but the formatter only ever writes U+00B5.
/// * **A second or more**, Go always print seconds, and only add minutes /
///   hours when non-zero: `"42s"`, `"5m0s"`, `"1h0m0s"`. That is why
///   `keep_alive: 300` comes back as `"5m0s"` and not `"5m"`.
/// * **It stops at hours.** No days, no weeks -- days are not a fixed length in
///   Go's model, so the package refuse to guess. Same reason `"1d"` doesn't parse.
/// * **Zero is `"0s"`**, and never `"-0s"` -- Go return early before the sign is
///   applied.
/// * Trailing zeros in the fraction are dropped, and so is the `.` when the
///   whole fraction is zero. `1_500_000_000` -> `"1.5s"`, not `"1.500000000s"`.
pub fn go_duration_string(d: i64) -> String {
    let neg = d < 0;
    // Upstream does `u := uint64(d); if neg { u = -u }` -- two's-complement
    // negation on the UNSIGNED value, so i64::MIN maps to 2^63 rather than
    // overflowing. `unsigned_abs` via i128 reproduces that exactly.
    let mut u: u64 = (d as i128).unsigned_abs() as u64;

    const NS_PER_SEC: u64 = 1_000_000_000;
    const NS_PER_MS: u64 = 1_000_000;
    const NS_PER_US: u64 = 1_000;

    if u < NS_PER_SEC {
        // Sub-second: one number, one small unit.
        if u == 0 {
            // Upstream return here, BEFORE the sign block -- so zero is always
            // "0s", never "-0s".
            return "0s".to_string();
        }
        let (prec, unit) = if u < NS_PER_US {
            (0usize, "ns")
        } else if u < NS_PER_MS {
            // U+00B5 MICRO SIGN, per upstream's `copy(buf[w:], "µ")`.
            (3usize, "\u{00b5}s")
        } else {
            (6usize, "ms")
        };
        let (frac, whole) = fmt_frac(u, prec);
        let sign = if neg { "-" } else { "" };
        return format!("{sign}{whole}{frac}{unit}");
    }

    // A second or more: [h][m]SS[.fff]s
    let (frac, secs) = fmt_frac(u, 9);
    u = secs;

    let mut tail = format!("{}{}s", u % 60, frac);
    u /= 60;
    if u > 0 {
        tail = format!("{}m{}", u % 60, tail);
        u /= 60;
        // Stop at hours -- days can be different lengths.
        if u > 0 {
            tail = format!("{u}h{tail}");
        }
    }

    if neg {
        format!("-{tail}")
    } else {
        tail
    }
}

/// **Upstream:** Go's `time.fmtFrac`. Format `v mod 10^prec` as a decimal
/// fraction (`".5"`), dropping trailing zeros and dropping the `.` entirely
/// when the whole fraction is zero. Returns the fraction text plus
/// `v / 10^prec`.
fn fmt_frac(v: u64, prec: usize) -> (String, u64) {
    let mut v = v;
    let mut rev = String::new();
    let mut printing = false;
    for _ in 0..prec {
        let digit = v % 10;
        printing = printing || digit != 0;
        if printing {
            // digit is 0..=9, so this is always a valid ASCII digit.
            rev.push((b'0' + digit as u8) as char);
        }
        v /= 10;
    }
    if printing {
        rev.push('.');
    }
    (rev.chars().rev().collect(), v)
}

// ---------------------------------------------------------------------------
// Go duration PARSING -- duplicated from `crate::envconfig`, on purpose
// ---------------------------------------------------------------------------
//
// Go duration parsing lives in ONE place: `envconfig::parse_go_duration`.
//
// It was briefly duplicated here, because `envconfig.rs` belonged to a different
// porting agent while this module was being written and could not be widened
// from the outside. Both agents flagged the copy instead of quietly forking it,
// and it is now reconciled: those helpers are `pub(crate)` and this module just
// calls them. Two copies of Go's duration grammar WOULD have drifted, and the
// failure mode is nasty -- a `keep_alive` that means different things depending
// on which door the request came through.
use crate::envconfig::parse_go_duration;

// ===========================================================================
// Timestamp -- Go's time.Time on the wire
// ===========================================================================

/// A `created_at` / `modified_at` / `expires_at` timestamp, carried as its
/// **RFC 3339 Nano text**.
///
/// **Upstream:** a plain `time.Time` field. Go's `time.Time.MarshalJSON` emit
/// RFC 3339 with nanosecond precision and trailing fractional zeros trimmed
/// (`2006-01-02T15:04:05.999999999Z07:00`), so a UTC time end with `Z` and a
/// whole-second time carries no fraction at all.
///
/// **Divergence, deliberate:** we hold the formatted string rather than a real
/// instant. This crate has no date-time dependency (no `chrono`, no `time`) and
/// adding one to move some text across a socket is not worth it. Consequence:
/// a timestamp round-trips **byte-exactly** (which is what the wire contract
/// cares about) but this type does no arithmetic. [`Timestamp::from_unix_nanos`]
/// is provided so a caller can actually build one; anything wanting to compare
/// or sort should carry its own epoch value alongside (see
/// [`LoadedModel::expires_at_unix`], which is exactly that).
///
/// [`Timestamp::default`] is Go's **zero time**, `"0001-01-01T00:00:00Z"` --
/// not the epoch, and not an empty string. A `ChatResponse` built without a
/// clock must still serialise the same shape upstream would.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Timestamp(pub String);

impl Default for Timestamp {
    fn default() -> Self {
        // Go's zero `time.Time` is January 1, year 1, 00:00:00 UTC.
        Timestamp("0001-01-01T00:00:00Z".to_string())
    }
}

impl Timestamp {
    /// Build a UTC RFC 3339 Nano timestamp from nanoseconds since the Unix
    /// epoch, matching what Go would emit for `time.Now().UTC()`.
    ///
    /// Negative input (before 1970) is handled, because Go handle it -- the
    /// euclidean division below is why. A plain `%` would give a negative
    /// remainder and a corrupt time-of-day.
    pub fn from_unix_nanos(ns: i64) -> Self {
        let secs = ns.div_euclid(1_000_000_000);
        let nanos = ns.rem_euclid(1_000_000_000) as u32;
        let days = secs.div_euclid(86_400);
        let sod = secs.rem_euclid(86_400);
        let (y, m, d) = civil_from_days(days);
        let (hh, mm, ss) = (sod / 3600, (sod % 3600) / 60, sod % 60);

        let mut s = format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}");
        if nanos != 0 {
            // RFC3339Nano trims trailing zeros, and drops the '.' with them.
            let frac = format!("{nanos:09}");
            s.push('.');
            s.push_str(frac.trim_end_matches('0'));
        }
        s.push('Z');
        Timestamp(s)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Days-since-epoch -> `(year, month, day)`, proleptic Gregorian.
///
/// **Upstream:** Howard Hinnant's `civil_from_days` (public domain,
/// <https://howardhinnant.github.io/date_algorithms.html>) -- the same algorithm
/// Go's `time` package use in `absDate`, written out here because we have no
/// date crate. Era-based, so it stay correct for negative years too.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// ===========================================================================
// StatusError -- the error shape clients actually parse
// ===========================================================================

/// An HTTP status plus ollama's error text. **Upstream:** `api.StatusError`.
///
/// ## The wire shape is weirder than you expect, and it is deliberate here
///
/// Only `ErrorMessage` carries a json tag (`json:"error"`). `StatusCode` and
/// `Status` carry **none**, so Go marshals them under their **Go field names**:
///
/// ```json
/// {"StatusCode":404,"Status":"404 Not Found","error":"model not found"}
/// ```
///
/// Capital S, no snake_case. That very likely started as an oversight upstream,
/// but it is what goes out of `c.JSON(apiError.StatusCode, apiError)` on the
/// remote-model error path, so it is what clients see and therefore what we
/// emit. Renaming these to `status_code` / `status` would be "fixing" a wire
/// contract, which is a breaking change wearing a tidy-up costume.
///
/// Note most error replies are **not** this type -- the ordinary handler path
/// writes a bare `{"error": "..."}` (see [`RouteError`]). This type is for when
/// a `StatusError` is forwarded whole.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatusError {
    /// Go: `StatusCode int` -- **no** json tag, so the key is `StatusCode`.
    #[serde(rename = "StatusCode", default)]
    pub status_code: u16,
    /// Go: `Status string` -- **no** json tag, so the key is `Status`.
    #[serde(rename = "Status", default)]
    pub status: String,
    #[serde(rename = "error", default)]
    pub error_message: String,
}

impl std::fmt::Display for StatusError {
    /// **Upstream:** `(StatusError).Error()`, all four cases.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.status.is_empty(), self.error_message.is_empty()) {
            (false, false) => write!(f, "{}: {}", self.status, self.error_message),
            (false, true) => f.write_str(&self.status),
            (true, false) => f.write_str(&self.error_message),
            // Upstream's own comment: "this should not happen".
            (true, true) => {
                f.write_str("something went wrong, please see the ollama server logs for details")
            }
        }
    }
}

impl std::error::Error for StatusError {}

// ===========================================================================
// Shared response bits
// ===========================================================================

/// Timing and token counts. **Upstream:** `api.Metrics`, embedded into both
/// [`GenerateResponse`] and [`ChatResponse`] (hence `#[serde(flatten)]`).
///
/// **Every duration here is a plain `int64` of NANOSECONDS on the wire**, not a
/// [`Duration`] string. Go's `time.Duration` is just a named `int64` with no
/// `MarshalJSON`, so `encoding/json` emit the raw number. Get this wrong and
/// `total_duration` comes out 10^9 times off, or as a string a client cannot do
/// arithmetic on.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Metrics {
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub total_duration: i64,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub load_duration: i64,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub prompt_eval_count: i32,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub prompt_eval_duration: i64,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub eval_count: i32,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub eval_duration: i64,
}

impl Metrics {
    /// The human summary `ollama run --verbose` print.
    ///
    /// **Upstream:** `(*Metrics).Summary()`, which `fmt.Fprintf` straight to
    /// stderr. **Divergence:** we return the lines instead of printing them --
    /// a library got no business owning somebody's stderr, and returning them
    /// makes the arithmetic testable. Durations render through
    /// [`go_duration_string`] because upstream use `%v` on a `time.Duration`,
    /// and rates are `count / duration_in_seconds`, two decimals.
    pub fn summary_lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        let secs = |ns: i64| ns as f64 / 1e9;
        if self.total_duration > 0 {
            out.push(format!(
                "total duration:       {}",
                go_duration_string(self.total_duration)
            ));
        }
        if self.load_duration > 0 {
            out.push(format!(
                "load duration:        {}",
                go_duration_string(self.load_duration)
            ));
        }
        if self.prompt_eval_count > 0 {
            out.push(format!(
                "prompt eval count:    {} token(s)",
                self.prompt_eval_count
            ));
        }
        if self.prompt_eval_duration > 0 {
            out.push(format!(
                "prompt eval duration: {}",
                go_duration_string(self.prompt_eval_duration)
            ));
            out.push(format!(
                "prompt eval rate:     {:.2} tokens/s",
                self.prompt_eval_count as f64 / secs(self.prompt_eval_duration)
            ));
        }
        if self.eval_count > 0 {
            out.push(format!("eval count:           {} token(s)", self.eval_count));
        }
        if self.eval_duration > 0 {
            out.push(format!(
                "eval duration:        {}",
                go_duration_string(self.eval_duration)
            ));
            out.push(format!(
                "eval rate:            {:.2} tokens/s",
                self.eval_count as f64 / secs(self.eval_duration)
            ));
        }
        out
    }
}

/// What the template actually rendered, for `_debug_render_only`.
/// **Upstream:** `api.DebugInfo`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebugInfo {
    pub rendered_template: String,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub image_count: i32,
}

/// One token and its log probability. **Upstream:** `api.TokenLogprob`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f64,
    /// Go: `[]int`. Byte values in practice, but the wire type is a JSON
    /// integer array, so it is NOT narrowed to `u8` here -- a value outside
    /// 0..=255 must deserialise rather than blow up, same as Go.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bytes: Vec<i64>,
}

/// A generated token's logprob plus, optionally, its alternatives.
/// **Upstream:** `api.Logprob`, which **embeds** `TokenLogprob` -- hence the
/// flatten, so `token` / `logprob` / `bytes` sit at the top level, not nested.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Logprob {
    #[serde(flatten)]
    pub token_logprob: TokenLogprob,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_logprobs: Vec<TokenLogprob>,
}

/// Details about a model. **Upstream:** `api.ModelDetails`.
///
/// Watch the tags: `parent_model` .. `quantization_level` have **no**
/// `omitempty`, so they are always emitted even when empty; only the two
/// lengths are omitted at zero.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDetails {
    #[serde(default)]
    pub parent_model: String,
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub family: String,
    /// See the module header's nil-vs-empty note: Go may emit `null` here, we
    /// always emit `[]`, and we accept both inbound.
    #[serde(default, deserialize_with = "null_as_empty_vec")]
    pub families: Vec<String>,
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub quantization_level: String,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub context_length: i32,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub embedding_length: i32,
}

/// One tensor's metadata, for `show --verbose`. **Upstream:** `api.Tensor`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tensor {
    pub name: String,
    #[serde(rename = "type")]
    pub tensor_type: String,
    #[serde(default, deserialize_with = "null_as_empty_vec")]
    pub shape: Vec<u64>,
}

// ===========================================================================
// Requests and responses
// ===========================================================================

/// `POST /api/generate`. **Upstream:** `api.GenerateRequest`.
///
/// Only `model` and `prompt` are really required; everything else default
/// sensibly. Note which fields carry `omitempty` and which do not -- `model`,
/// `prompt`, `suffix`, `system`, `template` and `options` are **always
/// emitted**, so a serialised request always carries those keys even when
/// empty. That is upstream's shape and clients diff against it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub prompt: String,
    /// Text that comes **after** the insertion point (fill-in-the-middle).
    /// Setting this demands [`Capability::Insert`] from the model.
    #[serde(default)]
    pub suffix: String,
    #[serde(default)]
    pub system: String,
    #[serde(default)]
    pub template: String,
    /// Deprecated upstream. Token ids from a previous response, replayed for a
    /// crude conversational memory. Go's `[]int`, so `i64` here.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub context: Vec<i64>,
    /// `None` means "not stated" and **streams** -- only an explicit `false`
    /// buffers. See [`ResponseMode::from_flag`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// No prompt templating at all. Mutually exclusive with `template`,
    /// `system` and `context` -- see [`handle_generate`].
    #[serde(default, skip_serializing_if = "is_false")]
    pub raw: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<Duration>,
    /// Base64 image payloads. Go's `[]ImageData` is `[][]byte`, which
    /// `encoding/json` renders as base64 **strings** -- so `Vec<String>` here
    /// is the same wire shape, and matches [`crate::api::Message::images`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
    /// Always emitted (no `omitempty`): `null` when unset, which is why this is
    /// an `Option` with no `skip_serializing_if`. Fed to
    /// [`crate::options::Options::apply_map`].
    #[serde(default)]
    pub options: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub think: Option<ThinkValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shift: Option<bool>,
    /// Return the rendered prompt instead of calling the model. The leading
    /// underscore in the wire name is upstream's marker for "debug, unstable".
    #[serde(rename = "_debug_render_only", default, skip_serializing_if = "is_false")]
    pub debug_render_only: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub logprobs: bool,
    /// Valid range **0..=20**, enforced by [`validate_top_logprobs`].
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub top_logprobs: i32,
}

/// `POST /api/chat`. **Upstream:** `api.ChatRequest`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default, deserialize_with = "null_as_empty_vec")]
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<Duration>,
    /// Upstream **embeds** `Tools` (a `[]Tool`) with tag `tools,omitempty`, so
    /// the key is `tools` and it disappears when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(default)]
    pub options: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub think: Option<ThinkValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shift: Option<bool>,
    #[serde(rename = "_debug_render_only", default, skip_serializing_if = "is_false")]
    pub debug_render_only: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub logprobs: bool,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub top_logprobs: i32,
}

/// One chunk (streaming) or the whole answer (buffered) from `/api/generate`.
/// **Upstream:** `api.GenerateResponse`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub remote_model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub remote_host: String,
    /// No `omitempty` -- always emitted, as Go's zero time when unset.
    #[serde(default)]
    pub created_at: Timestamp,
    pub response: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub thinking: String,
    pub done: bool,
    /// `"stop"`, `"length"`, `"load"`, `"unload"`, ... Only on the final chunk.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub done_reason: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub context: Vec<i64>,
    #[serde(flatten)]
    pub metrics: Metrics,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(rename = "_debug_info", default, skip_serializing_if = "Option::is_none")]
    pub debug_info: Option<DebugInfo>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub logprobs: Vec<Logprob>,
}

impl GenerateResponse {
    /// The reply for "prompt was empty, so we only loaded the model".
    /// **Upstream:** `GenerateHandler`'s `done_reason: "load"` branch.
    pub fn load(model: &str, created_at: Timestamp) -> Self {
        Self {
            model: model.to_string(),
            created_at,
            done: true,
            done_reason: "load".to_string(),
            ..Default::default()
        }
    }

    /// The reply for "empty prompt **and** `keep_alive: 0`, so kick the model
    /// out of memory". **Upstream:** the `expireRunner` branch. Note it also
    /// sets `response: ""` explicitly -- same thing as our default.
    pub fn unload(model: &str, created_at: Timestamp) -> Self {
        Self {
            model: model.to_string(),
            created_at,
            done: true,
            done_reason: "unload".to_string(),
            ..Default::default()
        }
    }
}

/// One chunk or the whole answer from `/api/chat`. **Upstream:**
/// `api.ChatResponse`.
///
/// `message` has **no** `omitempty` and is a struct anyway, so it is always
/// emitted -- a streaming delta with no content still carries
/// `"message":{"role":"assistant","content":""}`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    pub model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub remote_model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub remote_host: String,
    #[serde(default)]
    pub created_at: Timestamp,
    #[serde(default)]
    pub message: Message,
    pub done: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub done_reason: String,
    #[serde(rename = "_debug_info", default, skip_serializing_if = "Option::is_none")]
    pub debug_info: Option<DebugInfo>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub logprobs: Vec<Logprob>,
    #[serde(flatten)]
    pub metrics: Metrics,
}

impl ChatResponse {
    /// **Upstream:** `ChatHandler`'s `expireRunner` branch -- note the message
    /// is `{Role: "assistant"}`, i.e. a real assistant message with empty
    /// content, not an absent one.
    pub fn unload(model: &str, created_at: Timestamp) -> Self {
        Self {
            model: model.to_string(),
            created_at,
            message: Message::new("assistant", ""),
            done: true,
            done_reason: "unload".to_string(),
            ..Default::default()
        }
    }
}

/// `POST /api/embed`. **Upstream:** `api.EmbedRequest`.
///
/// `input` is `any` upstream: a bare string, or an array of strings. Anything
/// else is a 400 -- see [`embed_inputs`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbedRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub input: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<Duration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub dimensions: i32,
    #[serde(default)]
    pub options: Option<serde_json::Map<String, serde_json::Value>>,
}

/// **Upstream:** `api.EmbedResponse`. The durations are `time.Duration`, so
/// nanosecond integers -- see [`Metrics`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbedResponse {
    pub model: String,
    #[serde(default, deserialize_with = "null_as_empty_vec")]
    pub embeddings: Vec<Vec<f32>>,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub total_duration: i64,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub load_duration: i64,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub prompt_eval_count: i32,
}

/// `POST /api/embeddings` -- the **older, single-prompt** endpoint, kept
/// because clients still use it. **Upstream:** `api.EmbeddingRequest`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<Duration>,
    #[serde(default)]
    pub options: Option<serde_json::Map<String, serde_json::Value>>,
}

/// **Upstream:** `api.EmbeddingResponse`. Note `f64` here where
/// [`EmbedResponse`] uses `f32` -- that asymmetry is upstream's, not a typo.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    #[serde(default, deserialize_with = "null_as_empty_vec")]
    pub embedding: Vec<f64>,
}

/// `POST /api/create`. **Upstream:** `api.CreateRequest`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CreateRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub quantize: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub draft_quantize: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub from: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub remote_host: String,
    /// Go `map[string]string`, which `encoding/json` emit **sorted by key** --
    /// so `BTreeMap` is the faithful choice, not a hash map.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub files: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub draft_files: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub adapters: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub template: String,
    /// `any` upstream: a string **or** a list of strings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub system: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub renderer: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub parser: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub requires: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub info: Option<serde_json::Map<String, serde_json::Value>>,
    /// **Deprecated** upstream -- use `model`. No `omitempty`, so it is always
    /// emitted, empty or not.
    #[serde(default)]
    pub name: String,
    /// **Deprecated** upstream -- use `quantize`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub quantization: String,
}

/// `DELETE /api/delete`. **Upstream:** `api.DeleteRequest`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeleteRequest {
    #[serde(default)]
    pub model: String,
    /// **Deprecated** upstream. Always emitted. Resolved by [`model_or_name`].
    #[serde(default)]
    pub name: String,
}

/// `POST /api/show`. **Upstream:** `api.ShowRequest`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ShowRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub system: String,
    /// **Deprecated** upstream.
    #[serde(default)]
    pub template: String,
    #[serde(default)]
    pub verbose: bool,
    #[serde(default)]
    pub options: Option<serde_json::Map<String, serde_json::Value>>,
    /// **Deprecated** upstream -- use `model`.
    #[serde(default)]
    pub name: String,
}

/// **Upstream:** `api.ShowResponse`.
///
/// Two fields whose `omitempty` is a **no-op** because Go only honour it for
/// empty-able kinds: `details` (a struct) and `modified_at` (a struct). Both are
/// always emitted. Do not "fix" them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ShowResponse {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub license: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub modelfile: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub parameters: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub template: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub system: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub renderer: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub parser: String,
    /// `details,omitempty` on a struct -- **always emitted**.
    #[serde(default)]
    pub details: ModelDetails,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub remote_model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub remote_host: String,
    /// No `omitempty` -- always emitted, `null` when there is nothing.
    #[serde(default)]
    pub model_info: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projector_info: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tensors: Vec<Tensor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<Capability>,
    /// `modified_at,omitempty` on a struct -- **always emitted**.
    #[serde(default)]
    pub modified_at: Timestamp,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub requires: String,
}

/// `POST /api/copy`. **Upstream:** `api.CopyRequest`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CopyRequest {
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub destination: String,
}

/// `POST /api/pull`. **Upstream:** `api.PullRequest`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PullRequest {
    #[serde(default)]
    pub model: String,
    /// **Deprecated and ignored** upstream -- the comment says so explicitly.
    #[serde(default, skip_serializing_if = "is_false")]
    pub insecure: bool,
    /// **Deprecated and ignored**, but no `omitempty`, so still always emitted.
    #[serde(default)]
    pub username: String,
    /// **Deprecated and ignored**, but no `omitempty`, so still always emitted.
    #[serde(default)]
    pub password: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// **Deprecated** upstream -- use `model`.
    #[serde(default)]
    pub name: String,
}

/// `POST /api/push`. **Upstream:** `api.PushRequest`. Unlike [`PullRequest`],
/// `insecure` / `username` / `password` here are **not** marked deprecated.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PushRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub insecure: bool,
    #[serde(default)]
    pub username: String,
    #[serde(default)]
    pub password: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// **Deprecated** upstream -- use `model`.
    #[serde(default)]
    pub name: String,
}

/// A pull/push progress tick. **Upstream:** `api.ProgressResponse`. This is the
/// value that streams out of `/api/pull`, one JSON object per line.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgressResponse {
    pub status: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub digest: String,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub total: i64,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub completed: i64,
}

/// `GET /api/tags`. **Upstream:** `api.ListResponse`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ListResponse {
    #[serde(default, deserialize_with = "null_as_empty_vec")]
    pub models: Vec<ListModelResponse>,
}

/// One row of `/api/tags`. **Upstream:** `api.ListModelResponse`. `details` and
/// `modified_at` are structs, so their `omitempty` is a no-op -- always emitted.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ListModelResponse {
    pub name: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub remote_model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub remote_host: String,
    #[serde(default)]
    pub modified_at: Timestamp,
    pub size: i64,
    pub digest: String,
    #[serde(default)]
    pub details: ModelDetails,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<Capability>,
}

/// `GET /api/ps`. **Upstream:** `api.ProcessResponse`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProcessResponse {
    #[serde(default, deserialize_with = "null_as_empty_vec")]
    pub models: Vec<ProcessModelResponse>,
}

/// One loaded model in `/api/ps`. **Upstream:** `api.ProcessModelResponse`.
/// Nothing here carries a working `omitempty`, so every key is always present.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProcessModelResponse {
    pub name: String,
    pub model: String,
    pub size: i64,
    pub digest: String,
    #[serde(default)]
    pub details: ModelDetails,
    #[serde(default)]
    pub expires_at: Timestamp,
    pub size_vram: i64,
    pub context_length: i32,
}

// ===========================================================================
// serde helpers -- Go's omitempty, spelled out
// ===========================================================================

fn is_zero_i64(v: &i64) -> bool {
    *v == 0
}
fn is_zero_i32(v: &i32) -> bool {
    *v == 0
}
fn is_false(v: &bool) -> bool {
    !*v
}

/// Accept `null` where a list is expected, yielding an empty `Vec`.
///
/// Go tells a nil slice from an empty one and marshals the first as `null`;
/// Rust's `Vec` cannot, so on the way **in** we treat both the same. See the
/// module header for the full statement of this divergence.
fn null_as_empty_vec<'de, D, T>(d: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Ok(Option::<Vec<T>>::deserialize(d)?.unwrap_or_default())
}

// ===========================================================================
// The route table
// ===========================================================================

/// An HTTP method, only as much of one as the route table need.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Method {
    Get,
    Head,
    Post,
    Put,
    Patch,
    Delete,
    Options,
}

impl Method {
    /// Case-insensitive, because clients are clients. `None` for anything
    /// outside the set -- such a request can only ever be a 404/405 anyway.
    pub fn parse(s: &str) -> Option<Method> {
        Some(match s.to_ascii_uppercase().as_str() {
            "GET" => Method::Get,
            "HEAD" => Method::Head,
            "POST" => Method::Post,
            "PUT" => Method::Put,
            "PATCH" => Method::Patch,
            "DELETE" => Method::Delete,
            "OPTIONS" => Method::Options,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Method::Get => "GET",
            Method::Head => "HEAD",
            Method::Post => "POST",
            Method::Put => "PUT",
            Method::Patch => "PATCH",
            Method::Delete => "DELETE",
            Method::Options => "OPTIONS",
        }
    }
}

/// Every endpoint upstream register in `(*Server).GenerateRoutes`.
///
/// **Upstream:** `server/routes.go`, the `r.GET(...)` / `r.POST(...)` block.
/// The whole table is here -- including the cloud and OpenAI-compatibility
/// routes we do not implement -- because a route table with holes in it lies
/// about what an ollama client may knock on. [`Route::handler`] tell you which
/// core handler each one funnels into, which is the interesting part: several
/// different `/v1/...` paths all end up in `ChatHandler` behind a translating
/// middleware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Route {
    /// `HEAD|GET /` -- replies the literal text `Ollama is running`.
    Root,
    /// `HEAD|GET /api/version`
    Version,
    /// `GET /api/status`
    Status,
    /// `POST /api/pull`
    Pull,
    /// `POST /api/push`
    Push,
    /// `HEAD|GET /api/tags`
    List,
    /// `POST /api/show`
    Show,
    /// `DELETE /api/delete`
    Delete,
    /// `POST /api/me` -- cloud account. Not implemented here.
    Whoami,
    /// `POST /api/signout` -- cloud account. Not implemented here.
    Signout,
    /// `DELETE /api/user/keys/:encodedKey` -- deprecated signout. Not implemented.
    SignoutKey,
    /// `POST /api/create`
    Create,
    /// `POST /api/blobs/:digest`
    CreateBlob,
    /// `HEAD /api/blobs/:digest`
    HeadBlob,
    /// `POST /api/copy`
    Copy,
    /// `POST /api/experimental/web_search` -- cloud proxy. Not implemented.
    WebSearch,
    /// `POST /api/experimental/web_fetch` -- cloud proxy. Not implemented.
    WebFetch,
    /// `GET /api/experimental/model-recommendations`. Not implemented.
    ModelRecommendations,
    /// `GET /api/ps`
    Ps,
    /// `POST /api/generate`
    Generate,
    /// `POST /api/chat`
    Chat,
    /// `POST /api/embed`
    Embed,
    /// `POST /api/embeddings` -- the older single-prompt endpoint.
    Embeddings,
    /// `POST /v1/chat/completions` -- OpenAI shape, translated into chat.
    OpenAiChatCompletions,
    /// `POST /v1/completions` -- OpenAI shape, translated into generate.
    OpenAiCompletions,
    /// `POST /v1/embeddings` -- OpenAI shape, translated into embed.
    OpenAiEmbeddings,
    /// `GET /v1/models` -- OpenAI shape, translated into list.
    OpenAiModels,
    /// `GET /v1/models/:model` -- OpenAI shape, translated into show.
    OpenAiRetrieveModel,
    /// `POST /v1/responses` -- OpenAI Responses shape, translated into chat.
    OpenAiResponses,
    /// `POST /v1/audio/transcriptions` -- OpenAI shape, translated into chat.
    OpenAiTranscriptions,
    /// `POST /v1/messages` -- **Anthropic** Messages shape, translated into chat.
    AnthropicMessages,
}

/// Which core handler a [`Route`] ultimately runs, once any compatibility
/// middleware has translated the body.
///
/// **Upstream:** read straight off the `GenerateRoutes` registrations -- e.g.
/// `r.POST("/v1/messages", ..., middleware.AnthropicMessagesMiddleware(),
/// s.ChatHandler)`. Worth naming, because it is the single fact that explains
/// why the OpenAI and Anthropic surfaces need no separate inference code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Handler {
    Generate,
    Chat,
    Embed,
    Embeddings,
    List,
    Show,
    Pull,
    Push,
    Delete,
    Copy,
    Create,
    CreateBlob,
    HeadBlob,
    Ps,
    Version,
    Status,
    Root,
    /// A cloud-account or cloud-proxy endpoint. Deliberately not ported.
    Cloud,
}

impl Route {
    /// The core handler behind this route.
    pub fn handler(self) -> Handler {
        match self {
            Route::Root => Handler::Root,
            Route::Version => Handler::Version,
            Route::Status => Handler::Status,
            Route::Pull => Handler::Pull,
            Route::Push => Handler::Push,
            Route::List | Route::OpenAiModels => Handler::List,
            Route::Show | Route::OpenAiRetrieveModel => Handler::Show,
            Route::Delete => Handler::Delete,
            Route::Create => Handler::Create,
            Route::CreateBlob => Handler::CreateBlob,
            Route::HeadBlob => Handler::HeadBlob,
            Route::Copy => Handler::Copy,
            Route::Ps => Handler::Ps,
            Route::Generate | Route::OpenAiCompletions => Handler::Generate,
            Route::Chat
            | Route::OpenAiChatCompletions
            | Route::OpenAiResponses
            | Route::OpenAiTranscriptions
            | Route::AnthropicMessages => Handler::Chat,
            Route::Embed | Route::OpenAiEmbeddings => Handler::Embed,
            Route::Embeddings => Handler::Embeddings,
            Route::Whoami
            | Route::Signout
            | Route::SignoutKey
            | Route::WebSearch
            | Route::WebFetch
            | Route::ModelRecommendations => Handler::Cloud,
        }
    }

    /// Is this one of the endpoints this port actually implement? The cloud
    /// routes are in the table for completeness, but they got no handler here.
    pub fn is_implemented(self) -> bool {
        self.handler() != Handler::Cloud
    }
}

/// A matched route plus its single path parameter, if the pattern had one.
///
/// Every parameterised path upstream (`/api/blobs/:digest`, `/v1/models/:model`,
/// `/api/user/keys/:encodedKey`) has **exactly one** parameter, so an `Option`
/// beats a map here -- less ceremony, and nobody can index a key that does not
/// exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteMatch {
    pub route: Route,
    pub param: Option<String>,
}

/// Why a `(method, path)` pair did not match.
///
/// The 405 case exists because upstream set `r.HandleMethodNotAllowed = true`.
/// Without it gin would answer 404 for `PUT /api/chat`, and a client cannot
/// tell "wrong verb" from "wrong server".
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RouteLookupError {
    #[error("404 page not found")]
    NotFound,
    #[error("405 method not allowed")]
    MethodNotAllowed {
        /// Which methods this path *does* accept -- the `Allow` header a proper
        /// 405 owes the client.
        allowed: Vec<Method>,
    },
}

impl RouteLookupError {
    pub fn status(&self) -> u16 {
        match self {
            RouteLookupError::NotFound => 404,
            RouteLookupError::MethodNotAllowed { .. } => 405,
        }
    }
}

/// The full table: pattern -> (methods, route). `:name` is a single-segment
/// wildcard, same as gin.
const TABLE: &[(&str, &[Method], Route)] = &[
    ("/", &[Method::Head, Method::Get], Route::Root),
    ("/api/version", &[Method::Head, Method::Get], Route::Version),
    ("/api/status", &[Method::Get], Route::Status),
    ("/api/pull", &[Method::Post], Route::Pull),
    ("/api/push", &[Method::Post], Route::Push),
    ("/api/tags", &[Method::Head, Method::Get], Route::List),
    ("/api/show", &[Method::Post], Route::Show),
    ("/api/delete", &[Method::Delete], Route::Delete),
    ("/api/me", &[Method::Post], Route::Whoami),
    ("/api/signout", &[Method::Post], Route::Signout),
    ("/api/user/keys/:encodedKey", &[Method::Delete], Route::SignoutKey),
    ("/api/create", &[Method::Post], Route::Create),
    ("/api/blobs/:digest", &[Method::Post], Route::CreateBlob),
    ("/api/blobs/:digest", &[Method::Head], Route::HeadBlob),
    ("/api/copy", &[Method::Post], Route::Copy),
    ("/api/experimental/web_search", &[Method::Post], Route::WebSearch),
    ("/api/experimental/web_fetch", &[Method::Post], Route::WebFetch),
    (
        "/api/experimental/model-recommendations",
        &[Method::Get],
        Route::ModelRecommendations,
    ),
    ("/api/ps", &[Method::Get], Route::Ps),
    ("/api/generate", &[Method::Post], Route::Generate),
    ("/api/chat", &[Method::Post], Route::Chat),
    ("/api/embed", &[Method::Post], Route::Embed),
    ("/api/embeddings", &[Method::Post], Route::Embeddings),
    ("/v1/chat/completions", &[Method::Post], Route::OpenAiChatCompletions),
    ("/v1/completions", &[Method::Post], Route::OpenAiCompletions),
    ("/v1/embeddings", &[Method::Post], Route::OpenAiEmbeddings),
    ("/v1/models", &[Method::Get], Route::OpenAiModels),
    ("/v1/models/:model", &[Method::Get], Route::OpenAiRetrieveModel),
    ("/v1/responses", &[Method::Post], Route::OpenAiResponses),
    ("/v1/audio/transcriptions", &[Method::Post], Route::OpenAiTranscriptions),
    ("/v1/messages", &[Method::Post], Route::AnthropicMessages),
];

/// Map a method + path to a [`Route`]. Pure -- no state, no I/O.
///
/// **Upstream:** gin's router as configured by `GenerateRoutes`, including
/// `HandleMethodNotAllowed`. Two behaviours worth knowing:
///
/// * A **trailing slash** is tolerated (gin redirects it away by default), so
///   `/api/chat/` matches `/api/chat`. The root `/` is not affected.
/// * A **query string** is not part of the path and must already be stripped by
///   the caller -- this function does not look for `?`, because a real HTTP
///   stack has separated them long before here.
///
/// Errors carry the distinction a client needs: [`RouteLookupError::NotFound`]
/// (404) versus [`RouteLookupError::MethodNotAllowed`] (405, with the `Allow`
/// set).
pub fn match_route(method: &str, path: &str) -> Result<RouteMatch, RouteLookupError> {
    let path = normalise_path(path);
    let method = Method::parse(method);

    let mut allowed: Vec<Method> = Vec::new();
    let mut matched: Option<RouteMatch> = None;

    for (pattern, methods, route) in TABLE {
        let Some(param) = match_pattern(pattern, &path) else {
            continue;
        };
        for m in *methods {
            if !allowed.contains(m) {
                allowed.push(*m);
            }
        }
        if matched.is_none() && method.is_some_and(|m| methods.contains(&m)) {
            matched = Some(RouteMatch { route: *route, param });
        }
    }

    match matched {
        Some(m) => Ok(m),
        None if allowed.is_empty() => Err(RouteLookupError::NotFound),
        None => {
            allowed.sort_unstable();
            Err(RouteLookupError::MethodNotAllowed { allowed })
        }
    }
}

/// Strip one trailing slash, the way gin's `RedirectTrailingSlash` effectively
/// do. `"/"` itself stays `"/"` -- it is a real route.
fn normalise_path(path: &str) -> String {
    if path.len() > 1 {
        path.trim_end_matches('/').to_string()
    } else {
        path.to_string()
    }
}

/// Match one gin-style pattern. Returns `Some(param)` on a match, where `param`
/// is `Some(value)` iff the pattern had a `:name` segment.
fn match_pattern(pattern: &str, path: &str) -> Option<Option<String>> {
    if !pattern.contains(':') {
        return if pattern == path { Some(None) } else { None };
    }
    let mut param = None;
    let mut p = pattern.split('/');
    let mut q = path.split('/');
    loop {
        match (p.next(), q.next()) {
            (None, None) => return Some(param),
            (Some(a), Some(b)) => {
                if a.starts_with(':') {
                    // gin will not match an empty segment against `:name`.
                    if b.is_empty() {
                        return None;
                    }
                    param = Some(b.to_string());
                } else if a != b {
                    return None;
                }
            }
            _ => return None,
        }
    }
}

// ===========================================================================
// Errors -- status code + message, exactly as upstream writes them
// ===========================================================================

/// An error reply: an HTTP status plus the message that goes in
/// `{"error": "..."}`.
///
/// **Upstream:** every `c.JSON(status, gin.H{"error": msg})` in `routes.go`.
/// The message strings are reproduced **verbatim**, quirks and all, because
/// clients and test suites match on them. Where upstream use Go's `%q` verb
/// (which wraps the value in double quotes) we use Rust's `{:?}` on a `&str`,
/// which produce the same text for the names models actually have.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct RouteError {
    pub status: u16,
    pub message: String,
}

impl RouteError {
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        RouteError { status, message: message.into() }
    }

    /// `400 Bad Request`.
    pub fn bad_request(message: impl Into<String>) -> Self {
        RouteError::new(400, message)
    }

    /// `404 Not Found`.
    pub fn not_found(message: impl Into<String>) -> Self {
        RouteError::new(404, message)
    }

    /// `500 Internal Server Error`.
    pub fn internal(message: impl Into<String>) -> Self {
        RouteError::new(500, message)
    }

    /// The body a server should write: `{"error": "<message>"}` and nothing
    /// else. **Upstream:** `gin.H{"error": ...}`.
    pub fn to_body(&self) -> serde_json::Value {
        serde_json::json!({ "error": self.message })
    }

    /// **Upstream:** the `"missing request body"` reply every handler give when
    /// `ShouldBindJSON` return `io.EOF`. A separate constructor because every
    /// handler needs it and it is easy to get subtly wrong (it is 400 with that
    /// exact lowercase wording, not "empty body", not 422).
    pub fn missing_request_body() -> Self {
        RouteError::bad_request("missing request body")
    }
}

/// What the model catalogue can fail with when resolving a name.
///
/// **Upstream:** the error switch around `GetModel` in `GenerateHandler` /
/// `ChatHandler` -- `fs.ErrNotExist` -> 404, the literal
/// `errtypes.InvalidModelNameErrMsg` -> 400, anything else -> 500.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CatalogError {
    /// The manifest is not on disk.
    #[error("model not found")]
    NotFound,
    /// **Upstream:** `errtypes.InvalidModelNameErrMsg`, whose text is exactly
    /// `"invalid model name"` -- and upstream compares the error's *string* to
    /// that constant, so the wording is load-bearing.
    #[error("invalid model name")]
    InvalidName,
    #[error("{0}")]
    Other(String),
}

/// The text upstream's `errtypes.InvalidModelNameErrMsg` hold.
pub const INVALID_MODEL_NAME_ERR_MSG: &str = "invalid model name";

/// What scheduling a runner can fail with.
///
/// **Upstream:** the errors `handleScheduleError` switch over -- `errCapabilities`
/// and `errRequired` (`server/routes.go`, `server/images.go`),
/// `context.Canceled`, `ErrMaxQueue` (`server/sched.go`) and `os.ErrNotExist`.
///
/// This is a **local seam**: `crate::sched` is being written separately, so
/// rather than depend on a type that does not exist yet, the handler logic
/// speak this enum and the real scheduler maps into it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScheduleError {
    /// **Upstream:** `fmt.Errorf("model %w", errRequired)` -> `"model is required"`.
    #[error("model is required")]
    ModelRequired,
    /// **Upstream:** `CheckCapabilities` -> `"<name> does not support <a>\n<b>"`.
    #[error("{}", missing_capability_message(.model, .missing))]
    MissingCapabilities { model: String, missing: Vec<Capability> },
    /// **Upstream:** `context.Canceled`, answered with the **non-standard status
    /// 499** (nginx's "client closed request"). Not a typo -- clients rely on it
    /// to tell "you hung up" from a real server error.
    #[error("request canceled")]
    Canceled,
    /// **Upstream:** `ErrMaxQueue`, whose text has a **double space** before
    /// "maximum" and a trailing full stop. Copied exactly.
    #[error("server busy, please try again.  maximum pending requests exceeded")]
    MaxQueue,
    /// **Upstream:** `os.ErrNotExist` reaching the scheduler.
    #[error("model not found")]
    NotFound,
    #[error("{0}")]
    Other(String),
}

/// The word `CheckCapabilities` use for a capability in an error message.
///
/// **Upstream:** the `capToErr` map in `server/images.go`. Note
/// [`Capability::Image`] renders as **`"image generation"`**, not `"image"` --
/// the only place where the error word differs from [`Capability::as_str`]'s
/// wire name. Using the wire name here would produce an error message no
/// upstream test would recognise.
pub fn capability_error_word(c: Capability) -> &'static str {
    match c {
        Capability::Image => "image generation",
        other => other.as_str(),
    }
}

/// **Upstream:** `fmt.Errorf("%s %w", name, err)` wrapped around
/// `fmt.Errorf("%w %w", errCapabilities, errors.Join(errs...))`.
///
/// `errors.Join` joins with a **newline**, so a model missing two capabilities
/// produce a message with an embedded `\n`. Looks odd; it is what upstream
/// emit, and it goes straight into the JSON error string.
fn missing_capability_message(model: &str, missing: &[Capability]) -> String {
    let words: Vec<&str> = missing.iter().copied().map(capability_error_word).collect();
    format!("{model} does not support {}", words.join("\n"))
}

/// Map a scheduler failure onto its status + message.
///
/// **Upstream:** `handleScheduleError`. The statuses, in upstream's order:
/// missing capabilities / missing model name -> **400**; caller went away ->
/// **499**; queue full -> **503**; model not on disk -> **404** with the
/// `try pulling it first` hint; anything else -> **500**.
pub fn map_schedule_error(name: &str, err: &ScheduleError) -> RouteError {
    match err {
        ScheduleError::MissingCapabilities { .. } | ScheduleError::ModelRequired => {
            RouteError::bad_request(err.to_string())
        }
        ScheduleError::Canceled => RouteError::new(499, "request canceled"),
        ScheduleError::MaxQueue => RouteError::new(503, err.to_string()),
        ScheduleError::NotFound => {
            // %q -> Go-quoted, so the name comes back WITH double quotes.
            RouteError::not_found(format!("model {name:?} not found, try pulling it first"))
        }
        ScheduleError::Other(msg) => RouteError::internal(msg.clone()),
    }
}

/// Map a catalogue lookup failure onto its status + message.
///
/// **Upstream:** the `switch` after `GetModel`. `requested` is the name **as the
/// client wrote it**, because that is what upstream interpolate -- echoing the
/// canonicalised name back would confuse somebody who typed `Qwen3`.
pub fn map_catalog_error(requested: &str, err: &CatalogError) -> RouteError {
    match err {
        CatalogError::NotFound => RouteError::not_found(format!("model '{requested}' not found")),
        CatalogError::InvalidName => RouteError::bad_request(INVALID_MODEL_NAME_ERR_MSG),
        CatalogError::Other(msg) => RouteError::internal(msg.clone()),
    }
}

/// The reply for an **empty** `model` on `/api/generate` or `/api/embed`.
///
/// **Upstream:** `modelref.ParseRef("")` returns `ErrModelRequired`, which is
/// neither `errConflictingModelSource` nor `model.ErrUnqualifiedName`, so
/// `writeModelRefParseError` fall through to its **default** branch -- the
/// caller-supplied fallback. Generate and embed pass
/// `(404, "model '%s' not found")`.
///
/// So an empty model name is a **404**, not the 400 you would expect, and the
/// message interpolates the empty string: `model '' not found`. Looks like a
/// bug; it is the contract. And note this is exactly the case where generate
/// and chat diverge -- chat passes `(400, "model is required")`, which is
/// [`empty_model_error_for_chat`].
///
/// An **unqualified but non-empty** name (say `"//"`) is a different error and
/// gives 400 `invalid model name` on *every* endpoint -- see
/// [`CatalogError::InvalidName`]. Do not conflate the two.
pub fn empty_model_error_for_generate(requested: &str) -> RouteError {
    RouteError::not_found(format!("model '{requested}' not found"))
}

/// The reply for an **empty** `model` on `/api/chat` or `/api/embeddings`.
/// **Upstream:** `writeModelRefParseError(c, err, http.StatusBadRequest,
/// "model is required")`. See [`empty_model_error_for_generate`] for the other
/// half of the asymmetry.
pub fn empty_model_error_for_chat() -> RouteError {
    RouteError::bad_request("model is required")
}

// ===========================================================================
// Seams -- the two things a handler needs from the rest of the world
// ===========================================================================

/// The slice of a resolved model that the *routing* logic actually read.
///
/// **Local seam, not an upstream type.** Upstream's `server.Model` carry blob
/// paths, a parsed template, a projector list and more -- none of which any
/// decision in this module touch. Keeping it narrow means the handler logic is
/// testable with a five-line fake, and means `create.rs` / `sched.rs` can own
/// the fat model type without this module having an opinion about it.
///
/// Field by field, why each one is here:
///
/// * `name` -- the canonicalised name the scheduler is asked for.
/// * `config` -- `remote_host` / `remote_model` (the cloud short-circuit),
///   `model_families` (the mllama check), `parser` and `model_family` (the
///   harmony heuristic).
/// * `capabilities` -- thinking / insert / tools / image checks.
/// * `options` -- the Modelfile's own defaults, overlaid *under* the request's.
/// * `system`, `messages` -- prompt assembly defaults.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelSummary {
    pub name: Name,
    pub config: crate::api::ConfigV2,
    pub capabilities: Vec<Capability>,
    pub options: serde_json::Map<String, serde_json::Value>,
    pub system: String,
    pub messages: Vec<Message>,
}

impl ModelSummary {
    pub fn has_capability(&self, c: Capability) -> bool {
        self.capabilities.contains(&c)
    }

    /// Is this a stub pointing at a model on somebody else's server?
    /// **Upstream:** `m.Config.RemoteHost != "" && m.Config.RemoteModel != ""`
    /// -- **both** must be set, one alone does not count.
    pub fn is_remote(&self) -> bool {
        !self.config.remote_host.is_empty() && !self.config.remote_model.is_empty()
    }
}

/// Resolve a client-supplied model string to a [`ModelSummary`].
///
/// **Local seam.** Upstream splits this across `parseAndValidateModelRef`,
/// `getExistingName` (a case-insensitive longest-prefix match over the manifests
/// on disk) and `GetModel`. All three need the filesystem, so they belong to
/// `crate::manifest` / `crate::create`, not here. This trait is the one hole the
/// handlers punch through to reach them.
pub trait ModelCatalog {
    /// The name string is passed through **unmodified** -- implementations do
    /// their own `Name::parse` plus existing-name canonicalisation.
    fn resolve(&self, requested: &str) -> Result<ModelSummary, CatalogError>;
}

/// A currently-loaded model, as the scheduler see it.
///
/// **Local seam** for [`handle_ps`], mirroring the fields `PsHandler` read off
/// `runnerRef`. `expires_at_unix` is seconds since the epoch **as well as** the
/// formatted [`Timestamp`], because the sort is by expiry and [`Timestamp`]
/// deliberately does no arithmetic -- see its docs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoadedModel {
    pub name: Name,
    pub digest: String,
    pub size: i64,
    pub size_vram: i64,
    pub context_length: i32,
    pub expires_at: Timestamp,
    pub expires_at_unix: i64,
    pub details: ModelDetails,
}

// ===========================================================================
// Streaming-vs-buffered
// ===========================================================================

/// Does this request get one JSON object, or a stream of them?
///
/// **Upstream:** every handler's `if req.Stream != nil && !*req.Stream` -- i.e.
/// **streaming is the default**, and only an explicit `"stream": false` turns it
/// off. An absent `stream` and `"stream": true` behave identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseMode {
    /// Newline-delimited JSON, one object per chunk.
    /// **Upstream:** `streamResponse`.
    Stream,
    /// One JSON object, after the whole thing finishes.
    /// **Upstream:** `waitForStream`, or the accumulating loop in
    /// `GenerateHandler` / `ChatHandler`.
    Buffered,
}

impl ResponseMode {
    /// `None` (absent) and `Some(true)` both stream. Only `Some(false)` buffers.
    pub fn from_flag(stream: Option<bool>) -> Self {
        if stream == Some(false) {
            ResponseMode::Buffered
        } else {
            ResponseMode::Stream
        }
    }

    pub fn is_stream(self) -> bool {
        matches!(self, ResponseMode::Stream)
    }

    /// The `Content-Type` a local handler set.
    ///
    /// **Upstream:** `streamResponse` writes `application/x-ndjson`;
    /// `waitForStream` writes `application/json`. Careful, the **remote-proxy**
    /// path in `GenerateHandler`/`ChatHandler` uses
    /// `application/json; charset=utf-8` for the buffered case instead -- that
    /// inconsistency is upstream's, and since we do not port the proxy path it
    /// does not arise here. Do not "unify" them if the proxy ever lands.
    pub fn content_type(self) -> &'static str {
        match self {
            ResponseMode::Stream => "application/x-ndjson",
            ResponseMode::Buffered => "application/json",
        }
    }
}

// ===========================================================================
// Shared validation
// ===========================================================================

/// **Upstream:** the `top_logprobs must be between 0 and 20` check, which
/// appears **twice** in both `GenerateHandler` and `ChatHandler` (once before
/// the model lookup, once after). The duplication is dead code upstream --
/// nothing between the two can change the value -- so it is ported once.
pub fn validate_top_logprobs(n: i32) -> Result<(), RouteError> {
    if !(0..=20).contains(&n) {
        return Err(RouteError::bad_request("top_logprobs must be between 0 and 20"));
    }
    Ok(())
}

/// Resolve the deprecated `name` field against `model`.
///
/// **Upstream:** `cmp.Or(req.Model, req.Name)` -- first non-zero wins, so
/// `model` beats `name` and an empty `model` fall back to `name`.
pub fn model_or_name<'a>(model: &'a str, name: &'a str) -> &'a str {
    if model.is_empty() {
        name
    } else {
        model
    }
}

/// Is this an "unload me" request? **Upstream:** `req.KeepAlive != nil &&
/// req.KeepAlive.Duration == 0`, paired with an empty prompt / no messages.
///
/// It must be an **explicit** `keep_alive: 0`: an absent `keep_alive` mean "use
/// the default", which is 5 minutes, not zero.
fn asks_to_unload(keep_alive: Option<Duration>) -> bool {
    keep_alive == Some(Duration(0))
}

// ===========================================================================
// /api/generate
// ===========================================================================

/// What `/api/generate` decided to do.
#[derive(Debug, Clone, PartialEq)]
pub enum GenerateDisposition {
    /// Empty prompt **and** an explicit `keep_alive: 0`. The adapter should
    /// expire the runner and write this 200 reply. **Upstream:** the
    /// `s.sched.expireRunner(m)` branch.
    Unload(Box<GenerateResponse>),
    /// Everything validated. The adapter schedules a runner using the plan and
    /// runs inference (or, if [`GeneratePlan::load_only`], just replies).
    Infer(Box<GeneratePlan>),
}

/// Everything `/api/generate` worked out before touching a model.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratePlan {
    /// The canonicalised name to schedule.
    pub name: Name,
    /// Capabilities the model must have, in upstream's order: `completion`,
    /// then `insert` if a suffix was given, then `thinking` if the model
    /// advertises it.
    pub capabilities: Vec<Capability>,
    /// Possibly **defaulted** -- see `resolve_think`.
    pub think: Option<ThinkValue>,
    pub response_mode: ResponseMode,
    /// The prompt was empty, so after scheduling the reply is just
    /// `done_reason: "load"` -- **upstream:** `GenerateHandler`'s
    /// `if req.Prompt == ""` branch, which sits *after* `scheduleRunner` on
    /// purpose: the point of an empty prompt is to warm the model up.
    pub load_only: bool,
}

/// Validate and default a `/api/generate` request.
///
/// **Upstream:** `(*Server).GenerateHandler`, everything from the JSON bind down
/// to `scheduleRunner`. The order of the checks is upstream's order, and it is
/// **observable** -- a request that is wrong in two ways gets the first error,
/// so reordering these silently changes which message a client sees.
///
/// In order:
///
/// 1. `top_logprobs` outside 0..=20 -> **400**.
/// 2. An **empty** `model` -> **404 `model '' not found`**, this endpoint's
///    fallback. See [`empty_model_error_for_generate`] for why that is not the
///    400 you would expect.
/// 3. Model lookup -> **404 / 400 / 500** via [`map_catalog_error`].
/// 4. A reference that resolve to a remote stub -> **404** (it is not a model
///    *this* server can run, so upstream hide it rather than explain).
/// 5. Empty prompt + explicit `keep_alive: 0` -> unload, **200**.
/// 6. An image-generation model -> **400**, not supported here.
/// 7. `raw` together with `template` / `system` / `context` -> **400**.
/// 8. Thinking asked of a model that cannot think -> **400**.
/// 9. More than one image for an mllama model -> **400**.
///
/// **Not ported:** the cloud passthrough (between steps 3 and 4 upstream) and
/// the harmony parser selection, which needs the parsed template.
pub fn handle_generate<C: ModelCatalog>(
    catalog: &C,
    req: &GenerateRequest,
) -> Result<GenerateDisposition, RouteError> {
    validate_top_logprobs(req.top_logprobs)?;

    if req.model.is_empty() {
        return Err(empty_model_error_for_generate(&req.model));
    }

    let m = catalog
        .resolve(&req.model)
        .map_err(|e| map_catalog_error(&req.model, &e))?;

    // A stub that points somewhere else is not a model this server can serve.
    if m.is_remote() {
        return Err(RouteError::not_found(format!("model '{}' not found", req.model)));
    }

    // Unload: empty prompt AND an explicit keep_alive of zero.
    if req.prompt.is_empty() && asks_to_unload(req.keep_alive) {
        return Ok(GenerateDisposition::Unload(Box::new(GenerateResponse::unload(
            &req.model,
            Timestamp::default(),
        ))));
    }

    if m.has_capability(Capability::Image) {
        return Err(RouteError::bad_request(
            "image generation models are not currently supported",
        ));
    }

    // Raw means "send my prompt through untouched", so anything that would
    // template it is a contradiction, not a silently-ignored field.
    if req.raw && (!req.template.is_empty() || !req.system.is_empty() || !req.context.is_empty()) {
        return Err(RouteError::bad_request(
            "raw mode does not support template, system, or context",
        ));
    }

    // Upstream order: completion, then insert, then thinking.
    let mut capabilities = vec![Capability::Completion];
    if !req.suffix.is_empty() {
        capabilities.push(Capability::Insert);
    }
    let think = resolve_think(&m, req.think.as_ref(), &req.model, &mut capabilities)?;

    // mllama runners take exactly one image. Checked after scheduling upstream,
    // but it needs nothing from the runner, so it is checked here.
    if m.config.model_families.iter().any(|f| f == "mllama") && req.images.len() > 1 {
        return Err(RouteError::bad_request(
            "this model only supports one image while more than one image requested",
        ));
    }

    Ok(GenerateDisposition::Infer(Box::new(GeneratePlan {
        name: m.name.clone(),
        capabilities,
        think,
        response_mode: ResponseMode::from_flag(req.stream),
        load_only: req.prompt.is_empty(),
    })))
}

/// Thinking: default it on when the model can think, reject it when it cannot.
///
/// **Upstream:** the identical block in both `GenerateHandler` and `ChatHandler`:
/// if the model advertises `thinking`, that capability is demanded and an absent
/// `think` becomes `&api.ThinkValue{Value: true}`; otherwise a truthy `think` is
/// a 400.
///
/// Two things worth saying out loud:
///
/// * A thinking-capable model **thinks by default**. An absent `think` becomes
///   `true`, not `false`. That surprises people whose prompts suddenly grew a
///   reasoning block.
/// * `Think.Bool()` is what decides the rejection, so `"think": false` against a
///   non-thinking model is **fine** -- the caller explicitly said no, and there
///   is nothing to complain about. Only a truthy `think` is a 400.
///
/// **Not ported:** upstream's `relax_thinking` escape hatch, which the Anthropic
/// middleware sets to downgrade the 400 into "just drop the field" so Claude
/// Code keeps working. It is a property of a route we do not implement.
fn resolve_think(
    m: &ModelSummary,
    requested: Option<&ThinkValue>,
    requested_model: &str,
    capabilities: &mut Vec<Capability>,
) -> Result<Option<ThinkValue>, RouteError> {
    if m.has_capability(Capability::Thinking) {
        capabilities.push(Capability::Thinking);
        return Ok(Some(requested.cloned().unwrap_or(ThinkValue::Bool(true))));
    }
    if requested.is_some_and(ThinkValue::enabled) {
        // Go's %q -> the name comes back wrapped in double quotes.
        return Err(RouteError::bad_request(format!(
            "{requested_model:?} does not support thinking"
        )));
    }
    Ok(requested.cloned())
}

/// The 400 upstream give when the scheduler reports a missing `completion`
/// capability on `/api/generate`.
///
/// **Upstream:** `if errors.Is(err, errCapabilityCompletion) { ... "does not
/// support generate" }`. Note the wording is **`generate`**, the endpoint, not
/// `completion`, the capability -- and `/api/chat` says `chat` for the exact
/// same underlying error. Worth its own function so the two cannot drift.
pub fn generate_capability_error(requested_model: &str) -> RouteError {
    RouteError::bad_request(format!("{requested_model:?} does not support generate"))
}

// ===========================================================================
// /api/chat
// ===========================================================================

/// What `/api/chat` decided to do.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatDisposition {
    /// No messages **and** an explicit `keep_alive: 0`. **Upstream:** the
    /// `expireRunner` branch -- which in `ChatHandler` sits **before** the
    /// remote-model block, where `GenerateHandler` puts it **after**. That
    /// asymmetry is upstream's; it means a remote-stub model can be "unloaded"
    /// through `/api/chat` but not through `/api/generate`.
    Unload(Box<ChatResponse>),
    Infer(Box<ChatPlan>),
}

/// Everything `/api/chat` worked out before touching a model.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatPlan {
    pub name: Name,
    /// `completion`, then `tools` if any tool was offered, then `thinking` if
    /// the model advertises it.
    pub capabilities: Vec<Capability>,
    pub think: Option<ThinkValue>,
    pub response_mode: ResponseMode,
}

/// Validate and default a `/api/chat` request.
///
/// **Upstream:** `(*Server).ChatHandler` down to `scheduleRunner`.
///
/// The one place chat genuinely differ from generate, and it is easy to miss:
/// **the fallback for an unparseable model reference is 400 with
/// `"model is required"`, where `/api/generate` gives 404 with
/// `"model '<x>' not found"`.** Same failure, two different answers, because
/// upstream pass a different fallback into `writeModelRefParseError`. An empty
/// `model` is exactly that case -- see [`empty_model_error_for_generate`].
/// Ported faithfully rather than harmonised; clients test against these.
pub fn handle_chat<C: ModelCatalog>(
    catalog: &C,
    req: &ChatRequest,
) -> Result<ChatDisposition, RouteError> {
    validate_top_logprobs(req.top_logprobs)?;

    if req.model.is_empty() {
        return Err(empty_model_error_for_chat());
    }

    let m = catalog
        .resolve(&req.model)
        .map_err(|e| map_catalog_error(&req.model, &e))?;

    // Unload comes BEFORE the remote check here -- see ChatDisposition::Unload.
    if req.messages.is_empty() && asks_to_unload(req.keep_alive) {
        return Ok(ChatDisposition::Unload(Box::new(ChatResponse::unload(
            &req.model,
            Timestamp::default(),
        ))));
    }

    if m.is_remote() {
        return Err(RouteError::not_found(format!("model '{}' not found", req.model)));
    }

    let mut capabilities = vec![Capability::Completion];
    if !req.tools.is_empty() {
        capabilities.push(Capability::Tools);
    }
    let think = resolve_think(&m, req.think.as_ref(), &req.model, &mut capabilities)?;

    Ok(ChatDisposition::Infer(Box::new(ChatPlan {
        name: m.name.clone(),
        capabilities,
        think,
        response_mode: ResponseMode::from_flag(req.stream),
    })))
}

/// **Upstream:** `"%q does not support chat"` -- the chat twin of
/// [`generate_capability_error`].
pub fn chat_capability_error(requested_model: &str) -> RouteError {
    RouteError::bad_request(format!("{requested_model:?} does not support chat"))
}

// ===========================================================================
// /api/embed and /api/embeddings
// ===========================================================================

/// Flatten `EmbedRequest.input` into the list of strings to embed.
///
/// **Upstream:** the `switch i := req.Input.(type)` in `EmbedHandler`. Three
/// accepted shapes and one rejection, and the corners matter:
///
/// | `input` | result |
/// |---|---|
/// | `"hello"` | `["hello"]` |
/// | `""` | **empty list** -- an empty string is dropped, not embedded |
/// | `["a","b"]` | `["a","b"]` |
/// | `[]` | empty list |
/// | absent / `null` | empty list |
/// | `["a", 3]` | **400 `invalid input type`** |
/// | `3`, `{...}`, `true` | **400 `invalid input type`** |
///
/// An empty list is not an error: `EmbedHandler` answers **200** with
/// `{"model": ..., "embeddings": []}` after loading the model, which is how
/// clients warm an embedding model up.
pub fn embed_inputs(req: &EmbedRequest) -> Result<Vec<String>, RouteError> {
    let invalid = || RouteError::bad_request("invalid input type");
    match &req.input {
        None | Some(serde_json::Value::Null) => Ok(Vec::new()),
        Some(serde_json::Value::String(s)) => {
            // Upstream: `if len(i) > 0` -- an empty string contributes nothing.
            if s.is_empty() {
                Ok(Vec::new())
            } else {
                Ok(vec![s.clone()])
            }
        }
        Some(serde_json::Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for v in items {
                // Note: unlike the bare-string case, an empty string INSIDE an
                // array is kept. Upstream only length-checks the scalar branch.
                out.push(v.as_str().ok_or_else(invalid)?.to_string());
            }
            Ok(out)
        }
        Some(_) => Err(invalid()),
    }
}

/// What `/api/embed` decided to do.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbedDisposition {
    /// No usable input. **Upstream:** `c.JSON(http.StatusOK,
    /// api.EmbedResponse{Model: req.Model, Embeddings: [][]float32{}})` --
    /// *after* the runner is scheduled, so the model still gets loaded.
    Empty(Box<EmbedResponse>),
    Infer(Box<EmbedPlan>),
}

/// Everything `/api/embed` worked out before touching a model.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedPlan {
    pub name: Name,
    pub inputs: Vec<String>,
    /// **Upstream:** `[]model.Capability{}` -- embed asks for **no** capability
    /// at all, so an ordinary chat model will happily embed. Deliberate
    /// upstream, not an oversight.
    pub capabilities: Vec<Capability>,
    /// Absent means **truncate** (`truncate == nil || *truncate`), so an
    /// over-long input is silently cut rather than rejected. Only an explicit
    /// `false` turns that into an error.
    pub truncate: bool,
}

/// Validate a `/api/embed` request. **Upstream:** `(*Server).EmbedHandler`.
pub fn handle_embed<C: ModelCatalog>(
    catalog: &C,
    req: &EmbedRequest,
) -> Result<EmbedDisposition, RouteError> {
    // Embed shares generate's 404 fallback, not chat's 400.
    if req.model.is_empty() {
        return Err(empty_model_error_for_generate(&req.model));
    }

    let m = catalog
        .resolve(&req.model)
        .map_err(|e| map_catalog_error(&req.model, &e))?;

    let inputs = embed_inputs(req)?;
    if inputs.is_empty() {
        return Ok(EmbedDisposition::Empty(Box::new(EmbedResponse {
            model: req.model.clone(),
            ..Default::default()
        })));
    }

    Ok(EmbedDisposition::Infer(Box::new(EmbedPlan {
        name: m.name.clone(),
        inputs,
        capabilities: Vec::new(),
        truncate: req.truncate.unwrap_or(true),
    })))
}

/// What `/api/embeddings` decided to do.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingsDisposition {
    /// **Upstream:** *"an empty request loads the model"* -- 200 with an empty
    /// vector, after scheduling.
    Empty(Box<EmbeddingResponse>),
    Infer(Box<EmbeddingsPlan>),
}

/// Everything `/api/embeddings` worked out before touching a model.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingsPlan {
    pub name: Name,
    pub prompt: String,
    pub capabilities: Vec<Capability>,
}

/// Validate a `/api/embeddings` request.
///
/// **Upstream:** `(*Server).EmbeddingsHandler`. Note its model-reference
/// fallback is 400 `"model is required"` (like chat), not 404 (like generate) --
/// and unlike `/api/embed`, it does **not** run `getExistingName`, so upstream
/// does not canonicalise a differently-cased name here.
pub fn handle_embeddings<C: ModelCatalog>(
    catalog: &C,
    req: &EmbeddingRequest,
) -> Result<EmbeddingsDisposition, RouteError> {
    if req.model.is_empty() {
        return Err(empty_model_error_for_chat());
    }

    let m = catalog
        .resolve(&req.model)
        .map_err(|e| map_catalog_error(&req.model, &e))?;

    if req.prompt.is_empty() {
        return Ok(EmbeddingsDisposition::Empty(Box::default()));
    }

    Ok(EmbeddingsDisposition::Infer(Box::new(EmbeddingsPlan {
        name: m.name.clone(),
        prompt: req.prompt.clone(),
        capabilities: Vec::new(),
    })))
}

/// Scale a vector to unit length, rejecting NaN and Inf.
///
/// **Upstream:** `server.normalize`. Two details that are not obvious:
///
/// * The reciprocal is `1 / max(sqrt(sum), 1e-12)`, so an **all-zero vector does
///   not divide by zero** -- it comes back all zero. The `1e-12` floor is
///   upstream's exact constant.
/// * The sum is accumulated in `float32` upstream (`var sum float32`), so this
///   keeps `f32` accumulation too rather than "improving" it to `f64`. Changing
///   it would make our embeddings differ from ollama's in the last bits, and
///   embeddings get compared across implementations.
pub fn normalize(mut vec: Vec<f32>) -> Result<Vec<f32>, RouteError> {
    let mut sum: f32 = 0.0;
    for v in &vec {
        if v.is_nan() || v.is_infinite() {
            return Err(RouteError::internal("embedding contains NaN or Inf values"));
        }
        sum += v * v;
    }
    // Upstream: `float32(1.0 / max(math.Sqrt(float64(sum)), 1e-12))` -- the
    // sqrt and the max happen in float64, only the reciprocal narrows.
    let norm = (1.0 / (sum as f64).sqrt().max(1e-12)) as f32;
    for v in &mut vec {
        *v *= norm;
    }
    Ok(vec)
}

// ===========================================================================
// The model-management endpoints
// ===========================================================================

/// Resolve the model name for `/api/pull`.
///
/// **Upstream:** `PullHandler` -> `parseNormalizePullModelRef(cmp.Or(req.Model,
/// req.Name))`, whose fallback error is **400** with
/// `errtypes.InvalidModelNameErrMsg`.
pub fn handle_pull(req: &PullRequest) -> Result<(Name, ResponseMode), RouteError> {
    let requested = model_or_name(&req.model, &req.name);
    let name = Name::parse(requested);
    if !name.is_valid() {
        return Err(RouteError::bad_request(INVALID_MODEL_NAME_ERR_MSG));
    }
    Ok((name, ResponseMode::from_flag(req.stream)))
}

/// Resolve the model name for `/api/push`.
///
/// **Upstream:** `PushHandler`, which does **not** go through
/// `parseAndValidateModelRef`: it only checks that one of `model` / `name` is
/// non-empty, answering 400 `"model is required"` otherwise, and leaves name
/// validity to `getExistingName` inside the streaming goroutine (so a bad name
/// arrives as a **streamed** error, not a status code).
pub fn handle_push(req: &PushRequest) -> Result<(Name, ResponseMode), RouteError> {
    let requested = model_or_name(&req.model, &req.name);
    if requested.is_empty() {
        return Err(RouteError::bad_request("model is required"));
    }
    Ok((Name::parse(requested), ResponseMode::from_flag(req.stream)))
}

/// Resolve the model name for `DELETE /api/delete`.
///
/// **Upstream:** `DeleteHandler`. Its error wording differ from pull's: an
/// unqualified name gives 400 `name "<x>" is invalid` (Go `%q`, so quoted),
/// while a missing manifest gives 404 `model '<x>' not found` (single quotes).
/// Two quoting styles, two messages, both real.
pub fn handle_delete(req: &DeleteRequest) -> Result<Name, RouteError> {
    let requested = model_or_name(&req.model, &req.name);
    let name = Name::parse(requested);
    if !name.is_valid() {
        return Err(RouteError::bad_request(format!("name {requested:?} is invalid")));
    }
    Ok(name)
}

/// The 404 `DeleteHandler` give once the manifest turn out not to exist.
pub fn delete_not_found(requested: &str) -> RouteError {
    RouteError::not_found(format!("model '{requested}' not found"))
}

/// Resolve the model name for `POST /api/show`.
///
/// **Upstream:** `ShowHandler`'s `if req.Model != "" {} else if req.Name != ""
/// {} else { 400 "model is required" }` -- the same `cmp.Or` in longhand.
pub fn handle_show(req: &ShowRequest) -> Result<String, RouteError> {
    let requested = model_or_name(&req.model, &req.name);
    if requested.is_empty() {
        return Err(RouteError::bad_request("model is required"));
    }
    Ok(requested.to_string())
}

/// Validate `POST /api/copy`. **Upstream:** `CopyHandler`.
///
/// Both messages use Go's `%q`, so the offending name comes back **in double
/// quotes**: `source "" is invalid`. And the destination is checked only after
/// the source passes -- a request with both wrong reports the source.
pub fn handle_copy(req: &CopyRequest) -> Result<(Name, Name), RouteError> {
    let src = Name::parse(&req.source);
    if !src.is_valid() {
        return Err(RouteError::bad_request(format!("source {:?} is invalid", req.source)));
    }
    let dst = Name::parse(&req.destination);
    if !dst.is_valid() {
        return Err(RouteError::bad_request(format!(
            "destination {:?} is invalid",
            req.destination
        )));
    }
    Ok((src, dst))
}

/// The 404 `CopyHandler` give when the source manifest is missing. Note it is
/// `%q` here (`model "x" not found`) where the *generate* path uses single
/// quotes -- upstream is not consistent about this and neither can we be.
pub fn copy_not_found(source: &str) -> RouteError {
    RouteError::not_found(format!("model {source:?} not found"))
}

/// Validate a blob digest for `POST|HEAD /api/blobs/:digest`.
///
/// **Upstream:** `manifest.BlobsPath(digest)` inside `CreateBlobHandler` /
/// `HeadBlobHandler`, whose failure is a **400**. A digest is
/// `sha256:<64 lowercase hex>`; anything else cannot name a file in the blob
/// store, and letting it through would be a path-traversal hole -- which is why
/// the check lives here rather than being trusted from the URL.
pub fn validate_blob_digest(digest: &str) -> Result<(), RouteError> {
    let ok = digest.split_once(':').is_some_and(|(algo, hex)| {
        algo == "sha256"
            && hex.len() == 64
            && hex.bytes().all(|b| b.is_ascii_digit() || b.is_ascii_lowercase() && b <= b'f')
    });
    if ok {
        Ok(())
    } else {
        Err(RouteError::bad_request(format!("invalid digest format {digest:?}")))
    }
}

/// The 400 `CreateBlobHandler` give when the uploaded bytes hash to something
/// other than the digest in the URL. **Upstream:** `digest mismatch, expected
/// %q, got %q`.
pub fn blob_digest_mismatch(expected: &str, got: &str) -> RouteError {
    RouteError::bad_request(format!("digest mismatch, expected {expected:?}, got {got:?}"))
}

/// The 404 `HeadBlobHandler` give for a blob that is not in the store.
pub fn blob_not_found(digest: &str) -> RouteError {
    RouteError::not_found(format!("blob {digest:?} not found"))
}

/// Build the `GET /api/ps` reply. **Upstream:** `(*Server).PsHandler`.
///
/// The sort is the interesting bit: **stable**, descending by `expires_at`
/// (`cmp.Compare(j.ExpiresAt.Unix(), i.ExpiresAt.Unix())` -- note `j` before
/// `i`), so the model that will stay loaded longest is listed first, and models
/// sharing an expiry keep the scheduler's own order. `slices.SortStableFunc`
/// upstream, `sort_by_key` here, which is stable too.
///
/// Also note `name` and `model` are set to the **same** display string upstream;
/// the duplication is for older clients that only read one of them.
pub fn handle_ps(loaded: &[LoadedModel]) -> ProcessResponse {
    let mut keyed: Vec<(std::cmp::Reverse<i64>, ProcessModelResponse)> = loaded
        .iter()
        .map(|v| {
            let display = v.name.display_shortest();
            (
                // Reverse rather than a flipped comparator, so the descending
                // order is obvious instead of hiding in an argument order.
                std::cmp::Reverse(v.expires_at_unix),
                ProcessModelResponse {
                    name: display.clone(),
                    model: display,
                    size: v.size,
                    digest: v.digest.clone(),
                    details: v.details.clone(),
                    expires_at: v.expires_at.clone(),
                    size_vram: v.size_vram,
                    context_length: v.context_length,
                },
            )
        })
        .collect();
    keyed.sort_by_key(|(k, _)| *k);

    ProcessResponse { models: keyed.into_iter().map(|(_, m)| m).collect() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ThinkLevel;
    use serde_json::json;

    // -- Duration ------------------------------------------------------------

    /// Upstream's `TestKeepAliveParsingFromJSON`, case for case.
    #[test]
    fn keep_alive_parses_every_shape_upstream_accepts() {
        const SEC: i64 = 1_000_000_000;
        let of = |raw: &str| -> Option<Duration> {
            serde_json::from_str::<ChatRequest>(raw).expect("must parse").keep_alive
        };

        assert_eq!(of("{ }"), None, "Unset");
        assert_eq!(of(r#"{ "keep_alive": 42 }"#), Some(Duration(42 * SEC)), "Positive Integer");
        assert_eq!(
            of(r#"{ "keep_alive": 42.5 }"#),
            Some(Duration(42_500 * (SEC / 1000))),
            "Positive Float"
        );
        assert_eq!(
            of(r#"{ "keep_alive": "42m" }"#),
            Some(Duration(42 * 60 * SEC)),
            "Positive Integer String"
        );
        assert_eq!(of(r#"{ "keep_alive": -1 }"#), Some(Duration::FOREVER), "Negative Integer");
        assert_eq!(of(r#"{ "keep_alive": -3.14 }"#), Some(Duration::FOREVER), "Negative Float");
        assert_eq!(
            of(r#"{ "keep_alive": "-1m" }"#),
            Some(Duration::FOREVER),
            "Negative Integer String"
        );
    }

    /// Upstream's `TestDurationMarshalUnmarshal`. The negative case is the trap:
    /// `-1` goes out as a bare number and comes back as **forever**, not as -1.
    #[test]
    fn a_negative_duration_round_trips_into_forever() {
        for (name, input, expected) in [
            ("negative duration", Duration(-1), Duration::FOREVER),
            ("positive duration", Duration(42 * 1_000_000_000), Duration(42 * 1_000_000_000)),
            (
                "another positive duration",
                Duration(42 * 60 * 1_000_000_000),
                Duration(42 * 60 * 1_000_000_000),
            ),
            ("zero duration", Duration(0), Duration(0)),
            ("max duration", Duration::FOREVER, Duration::FOREVER),
        ] {
            let bytes = serde_json::to_string(&input).expect("marshal");
            let back: Duration = serde_json::from_str(&bytes).expect("unmarshal");
            assert_eq!(back, expected, "{name}: marshalled as {bytes}");
        }
    }

    #[test]
    fn a_negative_duration_marshals_as_the_bare_number_minus_one() {
        assert_eq!(serde_json::to_string(&Duration(-1)).unwrap(), "-1");
        // ANY negative value, not just -1.
        assert_eq!(serde_json::to_string(&Duration(-999_999)).unwrap(), "-1");
        assert_eq!(serde_json::to_string(&Duration(i64::MIN)).unwrap(), "-1");
    }

    #[test]
    fn a_non_negative_duration_marshals_as_a_go_duration_string() {
        assert_eq!(serde_json::to_string(&Duration(0)).unwrap(), r#""0s""#);
        assert_eq!(serde_json::to_string(&Duration::from_secs(300)).unwrap(), r#""5m0s""#);
        assert_eq!(
            serde_json::to_string(&Duration::FOREVER).unwrap(),
            r#""2562047h47m16.854775807s""#
        );
    }

    #[test]
    fn a_bare_number_keep_alive_means_seconds_not_nanoseconds() {
        let d: Duration = serde_json::from_str("5").unwrap();
        assert_eq!(d, Duration(5_000_000_000), "5 must be five SECONDS");
    }

    #[test]
    fn a_keep_alive_that_is_neither_number_nor_string_is_rejected() {
        for raw in ["true", "null", "{}", "[]"] {
            let e = serde_json::from_str::<Duration>(raw).expect_err(raw);
            assert!(e.to_string().contains("Unsupported type"), "{raw}: {e}");
        }
        // A malformed duration string is an error too -- days are not a Go unit.
        assert!(serde_json::from_str::<Duration>(r#""1d""#).is_err());
    }

    /// Go's `time.Duration.String()`, the cases that differ from anything Rust's
    /// std would produce.
    #[test]
    fn go_duration_strings_match_the_go_formatter() {
        for (ns, want) in [
            (0i64, "0s"),
            (1, "1ns"),
            (999, "999ns"),
            (1_000, "1\u{00b5}s"),
            (1_500, "1.5\u{00b5}s"),
            (1_000_000, "1ms"),
            (300_000_000, "300ms"),
            (1_000_000_000, "1s"),
            (1_500_000_000, "1.5s"),
            (42_000_000_000, "42s"),
            // The one everybody gets wrong: five minutes is "5m0s", not "5m".
            (300_000_000_000, "5m0s"),
            (3_600_000_000_000, "1h0m0s"),
            (3_661_000_000_000, "1h1m1s"),
            (-1, "-1ns"),
            (-300_000_000_000, "-5m0s"),
            (i64::MAX, "2562047h47m16.854775807s"),
        ] {
            assert_eq!(go_duration_string(ns), want, "for {ns} ns");
        }
    }

    /// The micro sign must be U+00B5 MICRO SIGN, which is what Go writes -- not
    /// the visually identical U+03BC GREEK SMALL LETTER MU.
    #[test]
    fn the_formatter_writes_micro_sign_not_greek_mu() {
        let s = go_duration_string(1_500);
        assert!(s.contains('\u{00b5}'), "{s:?} must use U+00B5");
        assert!(!s.contains('\u{03bc}'), "{s:?} must not use U+03BC");
        // ...but the PARSER accepts both, because keyboards differ.
        assert_eq!(parse_go_duration("2\u{00b5}s"), Some(2_000));
        assert_eq!(parse_go_duration("2\u{03bc}s"), Some(2_000));
    }

    /// Every value the formatter emits must parse back to itself. That is the
    /// property `keep_alive` actually depends on, since clients round-trip it.
    #[test]
    fn formatting_then_parsing_is_the_identity_for_non_negative_durations() {
        for ns in [
            0i64,
            1,
            999,
            1_000,
            1_500,
            999_999,
            1_000_000,
            300_000_000,
            1_000_000_000,
            1_500_000_000,
            300_000_000_000,
            3_661_000_000_000,
            i64::MAX,
        ] {
            let s = go_duration_string(ns);
            assert_eq!(parse_go_duration(&s), Some(ns), "{s:?} must parse back to {ns}");
        }
    }

    /// The duplicated parser must behave exactly like `envconfig`'s. Same corpus
    /// as `envconfig`'s own test, so the two copies cannot drift silently.
    #[test]
    fn the_duplicated_parser_handles_fractions_signs_and_both_micro_signs() {
        assert_eq!(parse_go_duration("1.5h"), Some(5400 * 1_000_000_000));
        assert_eq!(parse_go_duration("300ms"), Some(300_000_000));
        assert_eq!(parse_go_duration("100ns"), Some(100));
        assert_eq!(parse_go_duration("2us"), Some(2_000));
        assert_eq!(parse_go_duration("+1m"), Some(60 * 1_000_000_000));
        assert_eq!(parse_go_duration("1h30m"), Some(5400 * 1_000_000_000));
        assert_eq!(parse_go_duration("-1m"), Some(-60 * 1_000_000_000));
        assert_eq!(parse_go_duration("0"), Some(0));
        assert_eq!(parse_go_duration("+0"), Some(0));
        assert_eq!(parse_go_duration("1"), None, "a bare number needs a unit");
        assert_eq!(parse_go_duration(""), None);
        assert_eq!(parse_go_duration("-"), None);
        assert_eq!(parse_go_duration(".s"), None);
        assert_eq!(parse_go_duration("1d"), None, "days are not a Go unit");
        assert_eq!(parse_go_duration("1w"), None, "weeks neither");
    }

    #[test]
    fn a_duration_converts_into_the_envconfig_expiry_vocabulary() {
        assert_eq!(Duration::FOREVER.as_expiry(), Expiry::Never);
        assert_eq!(Duration(0).as_expiry(), Expiry::After(std::time::Duration::ZERO));
        assert_eq!(
            Duration::from_secs(300).as_expiry(),
            Expiry::After(std::time::Duration::from_secs(300))
        );
    }

    // -- Timestamp -----------------------------------------------------------

    #[test]
    fn the_zero_timestamp_is_gos_year_one_not_the_epoch() {
        assert_eq!(Timestamp::default().as_str(), "0001-01-01T00:00:00Z");
        assert_eq!(
            serde_json::to_string(&Timestamp::default()).unwrap(),
            r#""0001-01-01T00:00:00Z""#
        );
    }

    #[test]
    fn timestamps_render_as_rfc3339_nano_with_trailing_zeros_trimmed() {
        assert_eq!(Timestamp::from_unix_nanos(0).as_str(), "1970-01-01T00:00:00Z");
        assert_eq!(
            Timestamp::from_unix_nanos(1_700_000_000_000_000_000).as_str(),
            "2023-11-14T22:13:20Z"
        );
        // A fraction survives, trimmed of trailing zeros...
        assert_eq!(
            Timestamp::from_unix_nanos(1_700_000_000_500_000_000).as_str(),
            "2023-11-14T22:13:20.5Z"
        );
        // ...to full nanosecond precision when it needs it.
        assert_eq!(
            Timestamp::from_unix_nanos(1_700_000_000_123_456_789).as_str(),
            "2023-11-14T22:13:20.123456789Z"
        );
        // Leap day, because that is where date arithmetic dies.
        assert_eq!(
            Timestamp::from_unix_nanos(1_709_164_800_000_000_000).as_str(),
            "2024-02-29T00:00:00Z"
        );
        // Before the epoch: the time-of-day must not go negative.
        assert_eq!(Timestamp::from_unix_nanos(-1_000_000_000).as_str(), "1969-12-31T23:59:59Z");
    }

    // -- StatusError ---------------------------------------------------------

    /// The capitalised keys are the actual wire shape, oversight or not.
    #[test]
    fn a_status_error_marshals_with_gos_untagged_field_names() {
        let e = StatusError {
            status_code: 404,
            status: "404 Not Found".into(),
            error_message: "model not found".into(),
        };
        assert_eq!(
            serde_json::to_value(&e).unwrap(),
            json!({"StatusCode": 404, "Status": "404 Not Found", "error": "model not found"})
        );
        let back: StatusError = serde_json::from_value(serde_json::to_value(&e).unwrap()).unwrap();
        assert_eq!(back, e);
    }

    /// All four branches of `(StatusError).Error()`.
    #[test]
    fn a_status_error_displays_all_four_upstream_ways() {
        let mk = |status: &str, msg: &str| StatusError {
            status_code: 0,
            status: status.into(),
            error_message: msg.into(),
        };
        assert_eq!(mk("404 Not Found", "nope").to_string(), "404 Not Found: nope");
        assert_eq!(mk("404 Not Found", "").to_string(), "404 Not Found");
        assert_eq!(mk("", "nope").to_string(), "nope");
        assert_eq!(
            mk("", "").to_string(),
            "something went wrong, please see the ollama server logs for details"
        );
    }

    // -- Wire format ---------------------------------------------------------

    /// Go's `omitempty` is a **no-op on struct fields**, so `details` and
    /// `modified_at` are always emitted despite carrying the tag. Anybody who
    /// "tidies" a `skip_serializing_if` onto them breaks the wire format, and
    /// this test is the tripwire.
    #[test]
    fn omitempty_on_a_struct_field_does_nothing_so_details_is_always_emitted() {
        let v = serde_json::to_value(ListModelResponse::default()).unwrap();
        let obj = v.as_object().expect("object");
        assert!(obj.contains_key("details"), "details must always be present");
        assert!(obj.contains_key("modified_at"), "modified_at must always be present");

        let v = serde_json::to_value(ShowResponse::default()).unwrap();
        let obj = v.as_object().expect("object");
        assert!(obj.contains_key("details"));
        assert!(obj.contains_key("modified_at"));
        // ...and model_info has no omitempty at all, so it is present as null.
        assert_eq!(v["model_info"], json!(null));
    }

    /// `options` has no `omitempty`, so an absent one is `null` on the wire --
    /// present-and-null, not absent.
    #[test]
    fn an_absent_options_map_serialises_as_null_rather_than_disappearing() {
        let v = serde_json::to_value(GenerateRequest::default()).unwrap();
        assert_eq!(v["options"], json!(null));
        assert!(v.as_object().unwrap().contains_key("options"));
        // The always-emitted strings are there too, empty.
        for k in ["model", "prompt", "suffix", "system", "template"] {
            assert_eq!(v[k], json!(""), "{k} must be emitted even when empty");
        }
        // ...while the omitempty ones are gone.
        for k in ["context", "stream", "raw", "format", "keep_alive", "images", "think"] {
            assert!(!v.as_object().unwrap().contains_key(k), "{k} must be omitted");
        }
    }

    /// A request literal as a real client would send it, checked against the
    /// struct rather than against ourselves.
    #[test]
    fn a_generate_request_deserialises_from_the_literal_wire_json() {
        let raw = r#"{
            "model": "qwen3:0.6b",
            "prompt": "why is the sky blue",
            "stream": false,
            "keep_alive": "10m",
            "think": "high",
            "options": {"temperature": 0.2, "top_k": 40},
            "_debug_render_only": true,
            "top_logprobs": 5
        }"#;
        let r: GenerateRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(r.model, "qwen3:0.6b");
        assert_eq!(r.stream, Some(false));
        assert_eq!(r.keep_alive, Some(Duration(600 * 1_000_000_000)));
        assert_eq!(r.think, Some(ThinkValue::Level(ThinkLevel::High)));
        assert!(r.debug_render_only, "the _debug_render_only key must map through");
        assert_eq!(r.top_logprobs, 5);
        assert_eq!(r.options.as_ref().unwrap()["temperature"], json!(0.2));
    }

    /// `Metrics` durations are NANOSECOND INTEGERS, not `Duration` strings.
    /// Confusing the two is the single easiest way to break this API.
    #[test]
    fn metrics_durations_are_plain_nanosecond_integers_not_duration_strings() {
        let r = GenerateResponse {
            model: "m".into(),
            metrics: Metrics { total_duration: 1_500_000_000, eval_count: 7, ..Default::default() },
            ..Default::default()
        };
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["total_duration"], json!(1_500_000_000i64));
        assert_eq!(v["eval_count"], json!(7));
        // Flattened, so NOT nested under "metrics".
        assert!(!v.as_object().unwrap().contains_key("metrics"));
        // Zero-valued metrics are omitted.
        assert!(!v.as_object().unwrap().contains_key("load_duration"));
    }

    #[test]
    fn a_logprob_flattens_its_embedded_token_logprob() {
        let l = Logprob {
            token_logprob: TokenLogprob {
                token: "hi".into(),
                logprob: -0.5,
                bytes: vec![104, 105],
            },
            top_logprobs: vec![],
        };
        let v = serde_json::to_value(&l).unwrap();
        assert_eq!(v, json!({"token": "hi", "logprob": -0.5, "bytes": [104, 105]}));
        let back: Logprob = serde_json::from_value(v).unwrap();
        assert_eq!(back, l);
    }

    #[test]
    fn a_null_list_deserialises_the_same_as_an_empty_one() {
        let a: EmbeddingResponse = serde_json::from_str(r#"{"embedding": null}"#).unwrap();
        let b: EmbeddingResponse = serde_json::from_str(r#"{"embedding": []}"#).unwrap();
        assert_eq!(a, b);
        let a: ListResponse = serde_json::from_str(r#"{"models": null}"#).unwrap();
        assert!(a.models.is_empty());
    }

    #[test]
    fn a_chat_response_always_carries_a_message_even_when_empty() {
        let v = serde_json::to_value(ChatResponse::default()).unwrap();
        assert_eq!(v["message"], json!({"role": "", "content": ""}));
        assert_eq!(v["created_at"], json!("0001-01-01T00:00:00Z"));
        assert_eq!(v["done"], json!(false));
    }

    #[test]
    fn the_unload_replies_match_upstreams_shape() {
        let g = GenerateResponse::unload("qwen3", Timestamp::default());
        assert_eq!(g.done_reason, "unload");
        assert!(g.done);
        assert_eq!(g.response, "");

        let l = GenerateResponse::load("qwen3", Timestamp::default());
        assert_eq!(l.done_reason, "load");

        let c = ChatResponse::unload("qwen3", Timestamp::default());
        assert_eq!(c.done_reason, "unload");
        // An assistant message with empty content, NOT an absent message.
        assert_eq!(c.message.role, "assistant");
        assert_eq!(c.message.content, "");
    }

    // -- Routing -------------------------------------------------------------

    fn route_of(method: &str, path: &str) -> Route {
        match_route(method, path).expect("must match").route
    }

    #[test]
    fn every_upstream_route_registration_is_in_the_table() {
        assert_eq!(route_of("GET", "/"), Route::Root);
        assert_eq!(route_of("HEAD", "/"), Route::Root);
        assert_eq!(route_of("GET", "/api/version"), Route::Version);
        assert_eq!(route_of("GET", "/api/status"), Route::Status);
        assert_eq!(route_of("POST", "/api/pull"), Route::Pull);
        assert_eq!(route_of("POST", "/api/push"), Route::Push);
        assert_eq!(route_of("GET", "/api/tags"), Route::List);
        assert_eq!(route_of("HEAD", "/api/tags"), Route::List);
        assert_eq!(route_of("POST", "/api/show"), Route::Show);
        assert_eq!(route_of("DELETE", "/api/delete"), Route::Delete);
        assert_eq!(route_of("POST", "/api/me"), Route::Whoami);
        assert_eq!(route_of("POST", "/api/signout"), Route::Signout);
        assert_eq!(route_of("DELETE", "/api/user/keys/abc"), Route::SignoutKey);
        assert_eq!(route_of("POST", "/api/create"), Route::Create);
        assert_eq!(route_of("POST", "/api/copy"), Route::Copy);
        assert_eq!(route_of("POST", "/api/experimental/web_search"), Route::WebSearch);
        assert_eq!(route_of("POST", "/api/experimental/web_fetch"), Route::WebFetch);
        assert_eq!(
            route_of("GET", "/api/experimental/model-recommendations"),
            Route::ModelRecommendations
        );
        assert_eq!(route_of("GET", "/api/ps"), Route::Ps);
        assert_eq!(route_of("POST", "/api/generate"), Route::Generate);
        assert_eq!(route_of("POST", "/api/chat"), Route::Chat);
        assert_eq!(route_of("POST", "/api/embed"), Route::Embed);
        assert_eq!(route_of("POST", "/api/embeddings"), Route::Embeddings);
        assert_eq!(route_of("POST", "/v1/chat/completions"), Route::OpenAiChatCompletions);
        assert_eq!(route_of("POST", "/v1/completions"), Route::OpenAiCompletions);
        assert_eq!(route_of("POST", "/v1/embeddings"), Route::OpenAiEmbeddings);
        assert_eq!(route_of("GET", "/v1/models"), Route::OpenAiModels);
        assert_eq!(route_of("POST", "/v1/responses"), Route::OpenAiResponses);
        assert_eq!(route_of("POST", "/v1/audio/transcriptions"), Route::OpenAiTranscriptions);
        assert_eq!(route_of("POST", "/v1/messages"), Route::AnthropicMessages);
    }

    /// The same path serves two routes on two verbs -- the method has to pick.
    #[test]
    fn the_blob_path_dispatches_on_the_method_and_captures_the_digest() {
        let m = match_route("POST", "/api/blobs/sha256:abc").unwrap();
        assert_eq!(m.route, Route::CreateBlob);
        assert_eq!(m.param.as_deref(), Some("sha256:abc"));

        let m = match_route("HEAD", "/api/blobs/sha256:abc").unwrap();
        assert_eq!(m.route, Route::HeadBlob);
        assert_eq!(m.param.as_deref(), Some("sha256:abc"));
    }

    #[test]
    fn a_parameterised_path_captures_its_one_segment() {
        let m = match_route("GET", "/v1/models/qwen3").unwrap();
        assert_eq!(m.route, Route::OpenAiRetrieveModel);
        assert_eq!(m.param.as_deref(), Some("qwen3"));

        // `/v1/models/` normalises to `/v1/models`, i.e. the LIST route, and the
        // empty segment never matches `:model`.
        let m = match_route("GET", "/v1/models/").unwrap();
        assert_eq!(m, RouteMatch { route: Route::OpenAiModels, param: None });
    }

    /// Upstream set `HandleMethodNotAllowed`, so a known path with the wrong
    /// verb must be 405 -- **not** 404. A client cannot debug "wrong server"
    /// versus "wrong verb" otherwise.
    #[test]
    fn a_known_path_with_the_wrong_method_is_405_not_404() {
        let e = match_route("PUT", "/api/chat").unwrap_err();
        assert_eq!(e.status(), 405);
        assert_eq!(e, RouteLookupError::MethodNotAllowed { allowed: vec![Method::Post] });

        let e = match_route("DELETE", "/api/tags").unwrap_err();
        assert_eq!(
            e,
            RouteLookupError::MethodNotAllowed { allowed: vec![Method::Get, Method::Head] }
        );

        // Both verbs of the blob path are reported.
        let e = match_route("GET", "/api/blobs/sha256:abc").unwrap_err();
        assert_eq!(
            e,
            RouteLookupError::MethodNotAllowed { allowed: vec![Method::Head, Method::Post] }
        );

        // A path nobody serves is a plain 404.
        assert_eq!(match_route("GET", "/api/nope").unwrap_err(), RouteLookupError::NotFound);
        assert_eq!(match_route("GET", "/api/nope").unwrap_err().status(), 404);
    }

    #[test]
    fn a_trailing_slash_is_tolerated_but_the_root_is_still_the_root() {
        assert_eq!(route_of("POST", "/api/chat/"), Route::Chat);
        assert_eq!(route_of("GET", "/"), Route::Root);
    }

    #[test]
    fn an_unknown_method_never_matches_anything() {
        assert!(match_route("BREW", "/api/chat").is_err());
        // ...and a lowercase verb still works, because clients are clients.
        assert_eq!(route_of("post", "/api/chat"), Route::Chat);
    }

    /// The fact that makes the OpenAI/Anthropic surfaces cheap: they are all
    /// ChatHandler behind a translating middleware.
    #[test]
    fn the_v1_routes_all_funnel_into_the_same_few_handlers() {
        assert_eq!(Route::OpenAiChatCompletions.handler(), Handler::Chat);
        assert_eq!(Route::OpenAiResponses.handler(), Handler::Chat);
        assert_eq!(Route::OpenAiTranscriptions.handler(), Handler::Chat);
        assert_eq!(Route::AnthropicMessages.handler(), Handler::Chat);
        assert_eq!(Route::OpenAiCompletions.handler(), Handler::Generate);
        assert_eq!(Route::OpenAiEmbeddings.handler(), Handler::Embed);
        assert_eq!(Route::OpenAiModels.handler(), Handler::List);
        assert_eq!(Route::OpenAiRetrieveModel.handler(), Handler::Show);
    }

    #[test]
    fn the_cloud_routes_are_present_but_marked_unimplemented() {
        for r in [Route::Whoami, Route::Signout, Route::WebSearch, Route::WebFetch] {
            assert!(!r.is_implemented(), "{r:?}");
        }
        assert!(Route::Chat.is_implemented());
    }

    // -- Error mapping -------------------------------------------------------

    #[test]
    fn schedule_errors_map_onto_upstreams_statuses_including_the_odd_499() {
        let e = map_schedule_error("qwen3", &ScheduleError::ModelRequired);
        assert_eq!((e.status, e.message.as_str()), (400, "model is required"));

        let e = map_schedule_error("qwen3", &ScheduleError::Canceled);
        assert_eq!((e.status, e.message.as_str()), (499, "request canceled"));

        let e = map_schedule_error("qwen3", &ScheduleError::MaxQueue);
        assert_eq!(e.status, 503);
        assert_eq!(
            e.message, "server busy, please try again.  maximum pending requests exceeded",
            "the double space before 'maximum' is upstream's"
        );

        let e = map_schedule_error("qwen3", &ScheduleError::NotFound);
        assert_eq!(e.status, 404);
        assert_eq!(e.message, r#"model "qwen3" not found, try pulling it first"#);

        let e = map_schedule_error("qwen3", &ScheduleError::Other("boom".into()));
        assert_eq!((e.status, e.message.as_str()), (500, "boom"));
    }

    /// `errors.Join` puts a NEWLINE between capability names, and `image`
    /// renders as `image generation`. Both are load-bearing wording.
    #[test]
    fn a_missing_capability_message_joins_with_a_newline_and_renames_image() {
        let e = ScheduleError::MissingCapabilities {
            model: "qwen3".into(),
            missing: vec![Capability::Completion, Capability::Tools],
        };
        assert_eq!(e.to_string(), "qwen3 does not support completion\ntools");
        assert_eq!(map_schedule_error("qwen3", &e).status, 400);

        assert_eq!(capability_error_word(Capability::Image), "image generation");
        assert_eq!(capability_error_word(Capability::Tools), "tools");
    }

    #[test]
    fn catalog_errors_map_onto_404_400_and_500() {
        let e = map_catalog_error("qwen3", &CatalogError::NotFound);
        assert_eq!((e.status, e.message.as_str()), (404, "model 'qwen3' not found"));

        let e = map_catalog_error("qwen3", &CatalogError::InvalidName);
        assert_eq!((e.status, e.message.as_str()), (400, "invalid model name"));

        let e = map_catalog_error("qwen3", &CatalogError::Other("disk on fire".into()));
        assert_eq!((e.status, e.message.as_str()), (500, "disk on fire"));
    }

    #[test]
    fn an_error_body_is_exactly_one_error_key() {
        let e = RouteError::missing_request_body();
        assert_eq!(e.status, 400);
        assert_eq!(e.to_body(), json!({"error": "missing request body"}));
    }

    /// The endpoint name, not the capability name -- and the two endpoints must
    /// never drift into saying the same thing.
    #[test]
    fn the_completion_capability_error_names_the_endpoint_not_the_capability() {
        assert_eq!(
            generate_capability_error("qwen3").message,
            r#""qwen3" does not support generate"#
        );
        assert_eq!(chat_capability_error("qwen3").message, r#""qwen3" does not support chat"#);
    }

    // -- Handlers ------------------------------------------------------------

    /// A five-line catalogue, which is the whole point of the seam being narrow.
    struct FakeCatalog(Option<ModelSummary>);

    impl ModelCatalog for FakeCatalog {
        fn resolve(&self, _requested: &str) -> Result<ModelSummary, CatalogError> {
            self.0.clone().ok_or(CatalogError::NotFound)
        }
    }

    fn model_with(caps: &[Capability]) -> FakeCatalog {
        FakeCatalog(Some(ModelSummary {
            name: Name::parse("qwen3:0.6b"),
            capabilities: caps.to_vec(),
            ..Default::default()
        }))
    }

    #[test]
    fn generate_rejects_a_top_logprobs_outside_zero_to_twenty() {
        let cat = model_with(&[Capability::Completion]);
        // Checked FIRST, before the model is even looked at -- so an empty
        // model does not shadow the 400.
        for n in [-1, 21, 100] {
            let req = GenerateRequest { top_logprobs: n, ..Default::default() };
            let e = handle_generate(&cat, &req).unwrap_err();
            assert_eq!(
                (e.status, e.message.as_str()),
                (400, "top_logprobs must be between 0 and 20")
            );
        }
        for n in [0, 1, 20] {
            let req = GenerateRequest {
                model: "qwen3".into(),
                top_logprobs: n,
                prompt: "hi".into(),
                ..Default::default()
            };
            assert!(handle_generate(&cat, &req).is_ok(), "{n} must be allowed");
        }
    }

    #[test]
    fn generate_reports_an_unknown_model_as_404_with_the_name_the_client_typed() {
        let cat = FakeCatalog(None);
        let req =
            GenerateRequest { model: "Qwen3".into(), prompt: "hi".into(), ..Default::default() };
        let e = handle_generate(&cat, &req).unwrap_err();
        assert_eq!((e.status, e.message.as_str()), (404, "model 'Qwen3' not found"));
    }

    /// Empty prompt + explicit `keep_alive: 0` means "kick it out of memory".
    /// An ABSENT keep_alive must NOT trigger it -- that is just a warm-up.
    #[test]
    fn generate_unloads_only_on_an_empty_prompt_with_an_explicit_zero_keep_alive() {
        let cat = model_with(&[Capability::Completion]);

        let req = GenerateRequest {
            model: "qwen3".into(),
            keep_alive: Some(Duration(0)),
            ..Default::default()
        };
        let GenerateDisposition::Unload(resp) = handle_generate(&cat, &req).unwrap() else {
            panic!("expected an unload");
        };
        assert_eq!(resp.done_reason, "unload");

        // Absent keep_alive -> a load, not an unload.
        let req = GenerateRequest { model: "qwen3".into(), ..Default::default() };
        let GenerateDisposition::Infer(plan) = handle_generate(&cat, &req).unwrap() else {
            panic!("expected an infer");
        };
        assert!(plan.load_only, "an empty prompt still warms the model up");

        // A non-empty prompt with keep_alive 0 is an ordinary request.
        let req = GenerateRequest {
            model: "qwen3".into(),
            prompt: "hi".into(),
            keep_alive: Some(Duration(0)),
            ..Default::default()
        };
        let GenerateDisposition::Infer(plan) = handle_generate(&cat, &req).unwrap() else {
            panic!("expected an infer");
        };
        assert!(!plan.load_only);
    }

    #[test]
    fn generate_refuses_raw_mode_combined_with_anything_that_templates() {
        let cat = model_with(&[Capability::Completion]);
        let base = GenerateRequest {
            model: "qwen3".into(),
            prompt: "hi".into(),
            raw: true,
            ..Default::default()
        };
        for req in [
            GenerateRequest { template: "{{ .Prompt }}".into(), ..base.clone() },
            GenerateRequest { system: "be nice".into(), ..base.clone() },
            GenerateRequest { context: vec![1, 2, 3], ..base.clone() },
        ] {
            let e = handle_generate(&cat, &req).unwrap_err();
            assert_eq!(
                (e.status, e.message.as_str()),
                (400, "raw mode does not support template, system, or context")
            );
        }
        // Raw on its own is fine.
        assert!(handle_generate(&cat, &base).is_ok());
    }

    #[test]
    fn generate_refuses_an_image_generation_model() {
        let cat = model_with(&[Capability::Completion, Capability::Image]);
        let req =
            GenerateRequest { model: "sd".into(), prompt: "cat".into(), ..Default::default() };
        let e = handle_generate(&cat, &req).unwrap_err();
        assert_eq!(
            (e.status, e.message.as_str()),
            (400, "image generation models are not currently supported")
        );
    }

    #[test]
    fn a_suffix_demands_the_insert_capability() {
        let cat = model_with(&[Capability::Completion, Capability::Insert]);
        let req = GenerateRequest {
            model: "qwen3".into(),
            prompt: "fn main() {".into(),
            suffix: "}".into(),
            ..Default::default()
        };
        let GenerateDisposition::Infer(plan) = handle_generate(&cat, &req).unwrap() else {
            panic!()
        };
        assert_eq!(plan.capabilities, vec![Capability::Completion, Capability::Insert]);
    }

    /// A thinking-capable model thinks **by default** -- an absent `think`
    /// becomes `true`. That is the surprising half of the rule.
    #[test]
    fn a_thinking_model_thinks_by_default_and_a_plain_model_rejects_thinking() {
        let thinker = model_with(&[Capability::Completion, Capability::Thinking]);
        let req =
            GenerateRequest { model: "qwen3".into(), prompt: "hi".into(), ..Default::default() };
        let GenerateDisposition::Infer(plan) = handle_generate(&thinker, &req).unwrap() else {
            panic!()
        };
        assert_eq!(plan.think, Some(ThinkValue::Bool(true)), "absent think defaults to TRUE");
        assert!(plan.capabilities.contains(&Capability::Thinking));

        // An explicit level survives untouched.
        let req =
            GenerateRequest { think: Some(ThinkValue::Level(ThinkLevel::High)), ..req.clone() };
        let GenerateDisposition::Infer(plan) = handle_generate(&thinker, &req).unwrap() else {
            panic!()
        };
        assert_eq!(plan.think, Some(ThinkValue::Level(ThinkLevel::High)));

        // A non-thinking model: a truthy think is a 400, with Go's %q quoting.
        let plain = model_with(&[Capability::Completion]);
        let req = GenerateRequest {
            model: "qwen3".into(),
            prompt: "hi".into(),
            think: Some(ThinkValue::Bool(true)),
            ..Default::default()
        };
        let e = handle_generate(&plain, &req).unwrap_err();
        assert_eq!((e.status, e.message.as_str()), (400, r#""qwen3" does not support thinking"#));

        // ...but an explicit `think: false` is perfectly fine.
        let req = GenerateRequest { think: Some(ThinkValue::Bool(false)), ..req.clone() };
        assert!(handle_generate(&plain, &req).is_ok(), "explicitly declining is not an error");
    }

    #[test]
    fn an_mllama_model_takes_exactly_one_image() {
        let mut cat = model_with(&[Capability::Completion]);
        if let Some(m) = cat.0.as_mut() {
            m.config.model_families = vec!["mllama".into()];
        }
        let req = GenerateRequest {
            model: "llama3.2-vision".into(),
            prompt: "what".into(),
            images: vec!["a".into(), "b".into()],
            ..Default::default()
        };
        let e = handle_generate(&cat, &req).unwrap_err();
        assert_eq!(
            (e.status, e.message.as_str()),
            (400, "this model only supports one image while more than one image requested")
        );

        let req = GenerateRequest { images: vec!["a".into()], ..req.clone() };
        assert!(handle_generate(&cat, &req).is_ok());
    }

    #[test]
    fn a_remote_stub_model_is_hidden_behind_a_404() {
        let mut cat = model_with(&[Capability::Completion]);
        if let Some(m) = cat.0.as_mut() {
            m.config.remote_host = "https://ollama.com".into();
            m.config.remote_model = "big-model".into();
        }
        let req =
            GenerateRequest { model: "big".into(), prompt: "hi".into(), ..Default::default() };
        let e = handle_generate(&cat, &req).unwrap_err();
        assert_eq!((e.status, e.message.as_str()), (404, "model 'big' not found"));
    }

    /// `remote_host` alone is not a remote model -- upstream demands both.
    #[test]
    fn a_half_configured_remote_stub_is_not_treated_as_remote() {
        let mut cat = model_with(&[Capability::Completion]);
        if let Some(m) = cat.0.as_mut() {
            m.config.remote_host = "https://ollama.com".into();
        }
        let req =
            GenerateRequest { model: "big".into(), prompt: "hi".into(), ..Default::default() };
        assert!(handle_generate(&cat, &req).is_ok());
    }

    #[test]
    fn chat_asks_for_the_tools_capability_only_when_tools_were_offered() {
        let cat = model_with(&[Capability::Completion, Capability::Tools]);
        let req = ChatRequest {
            model: "qwen3".into(),
            messages: vec![Message::new("user", "hi")],
            ..Default::default()
        };
        let ChatDisposition::Infer(plan) = handle_chat(&cat, &req).unwrap() else { panic!() };
        assert_eq!(plan.capabilities, vec![Capability::Completion]);

        let req = ChatRequest { tools: vec![Tool::default()], ..req.clone() };
        let ChatDisposition::Infer(plan) = handle_chat(&cat, &req).unwrap() else { panic!() };
        assert_eq!(plan.capabilities, vec![Capability::Completion, Capability::Tools]);
    }

    #[test]
    fn chat_unloads_on_no_messages_with_an_explicit_zero_keep_alive() {
        let cat = model_with(&[Capability::Completion]);
        let req = ChatRequest {
            model: "qwen3".into(),
            keep_alive: Some(Duration(0)),
            ..Default::default()
        };
        let ChatDisposition::Unload(resp) = handle_chat(&cat, &req).unwrap() else {
            panic!("expected an unload")
        };
        assert_eq!(resp.done_reason, "unload");
        assert_eq!(resp.message.role, "assistant");
    }

    /// Chat's unload check sits BEFORE its remote check, so a remote stub can be
    /// unloaded through `/api/chat` -- where the same request on `/api/generate`
    /// gets a 404, because generate order the two the other way round. Upstream
    /// asymmetry, ported on purpose.
    #[test]
    fn chat_can_unload_a_remote_stub_but_generate_cannot() {
        let mut cat = model_with(&[Capability::Completion]);
        if let Some(m) = cat.0.as_mut() {
            m.config.remote_host = "https://ollama.com".into();
            m.config.remote_model = "big".into();
        }

        let chat =
            ChatRequest { model: "big".into(), keep_alive: Some(Duration(0)), ..Default::default() };
        assert!(matches!(handle_chat(&cat, &chat).unwrap(), ChatDisposition::Unload(_)));

        let generate = GenerateRequest {
            model: "big".into(),
            keep_alive: Some(Duration(0)),
            ..Default::default()
        };
        assert_eq!(handle_generate(&cat, &generate).unwrap_err().status, 404);
    }

    /// An empty `model` is `modelref.ErrModelRequired`, which falls through
    /// `writeModelRefParseError` to **the endpoint's own fallback** -- and the
    /// endpoints disagree. Chat/embeddings say 400 "model is required";
    /// generate/embed say **404 `model '' not found`**, interpolating the empty
    /// string. Looks like a bug upstream; it is the contract.
    #[test]
    fn an_empty_model_gets_a_different_answer_from_chat_than_from_generate() {
        let cat = model_with(&[Capability::Completion]);

        let e = handle_chat(&cat, &ChatRequest::default()).unwrap_err();
        assert_eq!((e.status, e.message.as_str()), (400, "model is required"));

        let e = handle_embeddings(&cat, &EmbeddingRequest::default()).unwrap_err();
        assert_eq!((e.status, e.message.as_str()), (400, "model is required"));

        let e = handle_generate(
            &cat,
            &GenerateRequest { prompt: "hi".into(), ..Default::default() },
        )
        .unwrap_err();
        assert_eq!((e.status, e.message.as_str()), (404, "model '' not found"));

        let e = handle_embed(&cat, &EmbedRequest::default()).unwrap_err();
        assert_eq!((e.status, e.message.as_str()), (404, "model '' not found"));
    }

    /// ...but an **unqualified, non-empty** name is a *different* upstream error
    /// (`model.ErrUnqualifiedName`), which `writeModelRefParseError` answers the
    /// SAME way on every endpoint: 400 `invalid model name`. The two cases must
    /// not be conflated.
    #[test]
    fn an_unqualified_name_is_the_same_400_on_every_endpoint() {
        struct BadName;
        impl ModelCatalog for BadName {
            fn resolve(&self, _: &str) -> Result<ModelSummary, CatalogError> {
                Err(CatalogError::InvalidName)
            }
        }
        let cat = BadName;
        let want = (400, INVALID_MODEL_NAME_ERR_MSG);

        let e = handle_generate(
            &cat,
            &GenerateRequest { model: "//".into(), prompt: "hi".into(), ..Default::default() },
        )
        .unwrap_err();
        assert_eq!((e.status, e.message.as_str()), want);

        let e = handle_chat(&cat, &ChatRequest { model: "//".into(), ..Default::default() })
            .unwrap_err();
        assert_eq!((e.status, e.message.as_str()), want);

        let e = handle_embeddings(
            &cat,
            &EmbeddingRequest { model: "//".into(), ..Default::default() },
        )
        .unwrap_err();
        assert_eq!((e.status, e.message.as_str()), want);
    }

    #[test]
    fn chat_defaults_to_streaming_unless_stream_is_explicitly_false() {
        assert_eq!(ResponseMode::from_flag(None), ResponseMode::Stream);
        assert_eq!(ResponseMode::from_flag(Some(true)), ResponseMode::Stream);
        assert_eq!(ResponseMode::from_flag(Some(false)), ResponseMode::Buffered);
        assert!(ResponseMode::Stream.is_stream());
        assert_eq!(ResponseMode::Stream.content_type(), "application/x-ndjson");
        assert_eq!(ResponseMode::Buffered.content_type(), "application/json");
    }

    // -- Embed ---------------------------------------------------------------

    #[test]
    fn embed_accepts_a_string_or_a_list_of_strings_and_nothing_else() {
        let mk =
            |input: serde_json::Value| EmbedRequest { input: Some(input), ..Default::default() };

        assert_eq!(embed_inputs(&mk(json!("hello"))).unwrap(), vec!["hello".to_string()]);
        assert_eq!(
            embed_inputs(&mk(json!(["a", "b"]))).unwrap(),
            vec!["a".to_string(), "b".to_string()]
        );
        assert!(embed_inputs(&mk(json!([]))).unwrap().is_empty());
        assert!(embed_inputs(&mk(json!(null))).unwrap().is_empty());
        assert!(embed_inputs(&EmbedRequest::default()).unwrap().is_empty());

        // An EMPTY bare string contributes nothing -- upstream's `if len(i) > 0`.
        assert!(embed_inputs(&mk(json!(""))).unwrap().is_empty());
        // ...but an empty string inside an array is kept, because upstream only
        // length-checks the scalar branch.
        assert_eq!(embed_inputs(&mk(json!([""]))).unwrap(), vec![String::new()]);

        for bad in [json!(3), json!(true), json!({"a": 1}), json!(["a", 3])] {
            let e = embed_inputs(&mk(bad.clone())).unwrap_err();
            assert_eq!((e.status, e.message.as_str()), (400, "invalid input type"), "{bad}");
        }
    }

    #[test]
    fn embed_with_no_usable_input_still_answers_200_so_the_model_gets_loaded() {
        let cat = model_with(&[]);
        let req =
            EmbedRequest { model: "nomic".into(), input: Some(json!("")), ..Default::default() };
        let EmbedDisposition::Empty(resp) = handle_embed(&cat, &req).unwrap() else {
            panic!("expected the empty reply")
        };
        assert_eq!(resp.model, "nomic");
        assert!(resp.embeddings.is_empty());
        assert_eq!(serde_json::to_value(&*resp).unwrap()["embeddings"], json!([]));
    }

    /// Embed asks for NO capability at all -- upstream passes an empty slice, so
    /// an ordinary chat model will happily embed.
    #[test]
    fn embed_demands_no_capability_at_all() {
        let cat = model_with(&[]);
        let req = EmbedRequest {
            model: "qwen3".into(),
            input: Some(json!("hello")),
            ..Default::default()
        };
        let EmbedDisposition::Infer(plan) = handle_embed(&cat, &req).unwrap() else { panic!() };
        assert!(plan.capabilities.is_empty());
        assert_eq!(plan.inputs, vec!["hello".to_string()]);
        assert!(plan.truncate, "an absent truncate means TRUNCATE, not error");

        let req = EmbedRequest { truncate: Some(false), ..req.clone() };
        let EmbedDisposition::Infer(plan) = handle_embed(&cat, &req).unwrap() else { panic!() };
        assert!(!plan.truncate);
    }

    #[test]
    fn embeddings_with_an_empty_prompt_just_loads_the_model() {
        let cat = model_with(&[]);
        let req = EmbeddingRequest { model: "nomic".into(), ..Default::default() };
        let EmbeddingsDisposition::Empty(resp) = handle_embeddings(&cat, &req).unwrap() else {
            panic!()
        };
        assert!(resp.embedding.is_empty());
    }

    #[test]
    fn normalising_scales_to_unit_length_and_survives_an_all_zero_vector() {
        let v = normalize(vec![3.0, 4.0]).unwrap();
        assert!((v[0] - 0.6).abs() < 1e-6, "{v:?}");
        assert!((v[1] - 0.8).abs() < 1e-6, "{v:?}");

        // All zeros: the 1e-12 floor stops a divide by zero.
        let v = normalize(vec![0.0, 0.0, 0.0]).unwrap();
        assert_eq!(v, vec![0.0, 0.0, 0.0]);

        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let e = normalize(vec![1.0, bad]).unwrap_err();
            assert_eq!(
                (e.status, e.message.as_str()),
                (500, "embedding contains NaN or Inf values")
            );
        }
    }

    // -- Model management ----------------------------------------------------

    #[test]
    fn the_deprecated_name_field_is_a_fallback_for_model_never_an_override() {
        assert_eq!(model_or_name("a", "b"), "a");
        assert_eq!(model_or_name("", "b"), "b");
        assert_eq!(model_or_name("", ""), "");
    }

    #[test]
    fn pull_falls_back_to_the_deprecated_name_field() {
        let (n, mode) = handle_pull(&PullRequest {
            name: "qwen3".into(),
            stream: Some(false),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(n.model, "qwen3");
        assert_eq!(mode, ResponseMode::Buffered);

        let e = handle_pull(&PullRequest::default()).unwrap_err();
        assert_eq!((e.status, e.message.as_str()), (400, "invalid model name"));
    }

    /// Push does NOT validate the name -- it only checks that one was given.
    /// A bad name reaches the client as a *streamed* error, not a status code.
    #[test]
    fn push_only_checks_that_a_model_was_named_at_all() {
        assert!(handle_push(&PushRequest { model: "qwen3".into(), ..Default::default() }).is_ok());
        let e = handle_push(&PushRequest::default()).unwrap_err();
        assert_eq!((e.status, e.message.as_str()), (400, "model is required"));
    }

    /// Delete and pull word the same failure differently -- `%q` versus `'%s'`.
    #[test]
    fn delete_reports_a_bad_name_with_gos_quoting() {
        let e = handle_delete(&DeleteRequest::default()).unwrap_err();
        assert_eq!((e.status, e.message.as_str()), (400, r#"name "" is invalid"#));

        let e = delete_not_found("qwen3");
        assert_eq!((e.status, e.message.as_str()), (404, "model 'qwen3' not found"));

        assert!(
            handle_delete(&DeleteRequest { model: "qwen3".into(), ..Default::default() }).is_ok()
        );
    }

    #[test]
    fn copy_checks_the_source_before_the_destination() {
        let e =
            handle_copy(&CopyRequest { source: String::new(), destination: "b".into() }).unwrap_err();
        assert_eq!((e.status, e.message.as_str()), (400, r#"source "" is invalid"#));

        let e =
            handle_copy(&CopyRequest { source: "a".into(), destination: String::new() }).unwrap_err();
        assert_eq!((e.status, e.message.as_str()), (400, r#"destination "" is invalid"#));

        // Both wrong -> the SOURCE is reported.
        let e = handle_copy(&CopyRequest::default()).unwrap_err();
        assert!(e.message.starts_with("source"));

        let (s, d) =
            handle_copy(&CopyRequest { source: "a".into(), destination: "b".into() }).unwrap();
        assert_eq!((s.model.as_str(), d.model.as_str()), ("a", "b"));

        assert_eq!(copy_not_found("a").message, r#"model "a" not found"#);
    }

    #[test]
    fn show_needs_a_model_from_either_field() {
        assert_eq!(
            handle_show(&ShowRequest { model: "a".into(), ..Default::default() }).unwrap(),
            "a"
        );
        assert_eq!(
            handle_show(&ShowRequest { name: "b".into(), ..Default::default() }).unwrap(),
            "b"
        );
        let e = handle_show(&ShowRequest::default()).unwrap_err();
        assert_eq!((e.status, e.message.as_str()), (400, "model is required"));
    }

    /// A digest out of the URL is untrusted input that becomes a file path, so
    /// this validation is a path-traversal guard, not a formality.
    #[test]
    fn a_blob_digest_must_be_sha256_plus_sixty_four_lowercase_hex() {
        let good = format!("sha256:{}", "a1b2c3d4".repeat(8));
        assert!(validate_blob_digest(&good).is_ok());

        for bad in [
            "",
            "sha256:",
            "abc",
            "sha256:../../etc/passwd",
            "sha512:0000000000000000000000000000000000000000000000000000000000000000",
            // Uppercase hex is not what the store writes.
            "sha256:A1B2C3D4A1B2C3D4A1B2C3D4A1B2C3D4A1B2C3D4A1B2C3D4A1B2C3D4A1B2C3D4",
            // Out-of-range letters.
            "sha256:z1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4",
            // 63 characters.
            "sha256:a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d",
        ] {
            assert_eq!(validate_blob_digest(bad).unwrap_err().status, 400, "{bad:?}");
        }

        assert_eq!(
            blob_digest_mismatch("sha256:aa", "sha256:bb").message,
            r#"digest mismatch, expected "sha256:aa", got "sha256:bb""#
        );
        assert_eq!(blob_not_found("sha256:aa").status, 404);
    }

    /// Longest-remaining first, and equal expiries keep their original order.
    #[test]
    fn ps_lists_the_longest_lived_model_first_and_is_stable_on_ties() {
        let mk = |model: &str, expires: i64| LoadedModel {
            name: Name::parse(model),
            expires_at_unix: expires,
            expires_at: Timestamp::from_unix_nanos(expires * 1_000_000_000),
            ..Default::default()
        };
        let out = handle_ps(&[mk("a", 100), mk("b", 300), mk("c", 200), mk("d", 300)]);
        // `display_shortest` keeps the tag, so a bare "b" comes back "b:latest".
        let names: Vec<&str> = out.models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["b:latest", "d:latest", "c:latest", "a:latest"],
            "b before d: the tie keeps input order"
        );

        // name and model carry the same display string, for older clients.
        assert_eq!(out.models[0].name, out.models[0].model);
        assert!(handle_ps(&[]).models.is_empty());
    }

    #[test]
    fn a_metrics_summary_reports_rates_in_tokens_per_second() {
        let m = Metrics {
            total_duration: 2_000_000_000,
            load_duration: 500_000_000,
            prompt_eval_count: 10,
            prompt_eval_duration: 1_000_000_000,
            eval_count: 60,
            eval_duration: 2_000_000_000,
        };
        let lines = m.summary_lines();
        assert!(lines.iter().any(|l| l == "total duration:       2s"), "{lines:?}");
        assert!(lines.iter().any(|l| l == "load duration:        500ms"), "{lines:?}");
        assert!(lines.iter().any(|l| l == "prompt eval rate:     10.00 tokens/s"), "{lines:?}");
        assert!(lines.iter().any(|l| l == "eval rate:            30.00 tokens/s"), "{lines:?}");

        // Zero metrics print nothing at all.
        assert!(Metrics::default().summary_lines().is_empty());
    }
}
