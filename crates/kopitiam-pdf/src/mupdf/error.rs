//! Ported from MuPDF `source/fitz/error.c` + `include/mupdf/fitz/context.h`
//! (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the algorithms and numeric
//! behaviour follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # Why this is `Result`, not `setjmp`/`longjmp`
//!
//! MuPDF implements exceptions in C with `setjmp`/`longjmp`: `fz_try` pushes a
//! `jmp_buf` onto a per-context stack, `fz_throw` records a code + formatted
//! message on the `fz_context` and `longjmp`s back to the nearest `fz_try`, and
//! `fz_catch` inspects the recorded code (see the state-machine comment at
//! `error.c:179`). That machinery exists to give C a non-local unwind that it
//! otherwise lacks.
//!
//! Rust already has that unwind: it is `Result` + the `?` operator. So this port
//! does **not** reproduce the `setjmp` stack, the `fz_context.error` slots, or
//! the `fz_try`/`fz_always`/`fz_catch` macros. Instead it ports the part that
//! actually carries meaning across the whole engine -- the **error taxonomy**
//! (`enum fz_error_type` / `FZ_ERROR_*`) and the **message discipline** of
//! `fz_throw` -- as a plain [`Error`] value returned in a [`Result`]. The MuPDF
//! control-flow verbs map onto Rust idioms as follows (this is the port-wide
//! convention; later modules `return Err(..)` / `?` instead of throwing):
//!
//! | MuPDF                          | Rust                                        |
//! |--------------------------------|---------------------------------------------|
//! | `fz_throw(ctx, code, fmt, …)`  | `return Err(Error::<kind>(format!(…)))`      |
//! | `fz_vthrow`                    | `Error::new(kind, msg)`                      |
//! | `fz_try { … } fz_catch { … }`  | `match result { Ok(_) => …, Err(e) => … }`   |
//! | `fz_rethrow(ctx)`              | `?` (propagate the `Err`)                    |
//! | `fz_rethrow_if(ctx, err)`      | `if e.matches(err) { return Err(e); }`       |
//! | `fz_rethrow_unless(ctx, err)`  | `if !e.matches(err) { return Err(e); }`      |
//! | `fz_morph_error(ctx, a, b)`    | [`Error::morph`]                             |
//! | `fz_caught(ctx)`               | [`Error::kind`]                              |
//! | `fz_caught_message(ctx)`       | [`Error::message`]                           |
//! | `fz_ignore_error(ctx)`         | drop the `Err` (only for control signals)    |
//!
//! ## What is deliberately not ported
//!
//! The warning stream (`fz_warn`/`fz_flush_warnings`), the error/warning
//! *callbacks* (`fz_set_error_callback` &c.), the `FZ_VERBOSE_EXCEPTIONS`
//! `…FL` file/line variants, and `errno` capture for `FZ_ERROR_SYSTEM`
//! (`fz_caught_errno`) are context/IO plumbing, not on the text-extraction path.
//! MuPDF formats each message into a fixed 256-byte buffer and truncates
//! (`error.c:350`); we use an unbounded `String`, so no truncation is
//! reproduced. Port these only if a later module actually needs them.

use core::fmt;

// ---------------------------------------------------------------------------
// Error taxonomy
// ---------------------------------------------------------------------------

// MuPDF: enum fz_error_type / FZ_ERROR_x (context.h:218)
//
// The discriminants are pinned to MuPDF's enum values so `kind as i32` is the
// same integer MuPDF passes around as an error `code`. `FZ_ERROR_NONE` (value 0)
// is *not* represented here: it is MuPDF's "no error" sentinel, and in this port
// that role belongs to `Result::Ok`. Every [`ErrorKind`] is therefore a genuine
// error, mirroring the invariant `errcode >= FZ_ERROR_NONE` the C asserts.
/// A ported `FZ_ERROR_*` category. Discriminants match MuPDF's `fz_error_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum ErrorKind {
    /// `FZ_ERROR_GENERIC` -- an unclassified error.
    Generic = 1,
    /// `FZ_ERROR_SYSTEM` -- fatal out-of-memory or syscall error.
    ///
    /// In MuPDF this is the only category that also records `errno`
    /// (`fz_caught_errno`); this port folds any such detail into the message.
    System = 2,
    /// `FZ_ERROR_LIBRARY` -- unclassified error from a third-party library.
    Library = 3,
    /// `FZ_ERROR_ARGUMENT` -- invalid or out-of-range arguments to functions.
    Argument = 4,
    /// `FZ_ERROR_LIMIT` -- failed because of resource or other hard limits.
    Limit = 5,
    /// `FZ_ERROR_UNSUPPORTED` -- tried to use an unsupported feature.
    Unsupported = 6,
    /// `FZ_ERROR_FORMAT` -- syntax or format errors that are **unrecoverable**.
    Format = 7,
    /// `FZ_ERROR_SYNTAX` -- syntax errors that should be diagnosed and ignored
    /// (i.e. recoverable: the document keeps loading past them).
    Syntax = 8,
    /// `FZ_ERROR_TRYLATER` -- try-later progressive-loading signal (internal).
    ///
    /// Not a failure of the data: it means "not enough has streamed in yet".
    TryLater = 9,
    /// `FZ_ERROR_ABORT` -- user-requested abort signal (internal).
    Abort = 10,
    /// `FZ_ERROR_REPAIRED` -- internal flag used when repairing a PDF to avoid
    /// cycles.
    Repaired = 11,
}

impl ErrorKind {
    // MuPDF: fz_error_type_name (error.c:393)
    /// The lowercase category name MuPDF prints for this error
    /// (`fz_error_type_name`).
    pub const fn name(self) -> &'static str {
        match self {
            ErrorKind::Generic => "generic",
            ErrorKind::System => "system",
            ErrorKind::Library => "library",
            ErrorKind::Unsupported => "unsupported",
            ErrorKind::Argument => "argument",
            ErrorKind::Limit => "limit",
            ErrorKind::Format => "format",
            ErrorKind::Syntax => "syntax",
            ErrorKind::TryLater => "trylater",
            ErrorKind::Abort => "abort",
            ErrorKind::Repaired => "repaired",
        }
    }

    /// The integer error `code` MuPDF uses for this category (its `fz_error_type`
    /// value). Kept so `fz_morph_error`/`fz_rethrow_if`-style code comparisons
    /// port across unchanged.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// `FZ_ERROR_TRYLATER`: the progressive-loading retry signal. The one
    /// "recoverable, come back later" category -- callers may re-attempt once
    /// more data is available rather than treat it as a hard failure.
    pub const fn is_try_later(self) -> bool {
        matches!(self, ErrorKind::TryLater)
    }

    /// `FZ_ERROR_ABORT`: the user-requested abort signal.
    pub const fn is_abort(self) -> bool {
        matches!(self, ErrorKind::Abort)
    }

    // MuPDF: fz_report_error / fz_ignore_error CLUSTER guards (error.c:533, 547)
    /// A control signal rather than a real error: `TRYLATER` or `ABORT`.
    ///
    /// MuPDF singles these two out -- under a `CLUSTER` build `fz_report_error`
    /// *aborts* if asked to report one, and `fz_ignore_error` aborts if asked to
    /// swallow anything else. In other words, these are exactly the errors that
    /// are meant to be handled/ignored by control flow, never logged as faults.
    pub const fn is_control_signal(self) -> bool {
        matches!(self, ErrorKind::TryLater | ErrorKind::Abort)
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Error value
// ---------------------------------------------------------------------------

/// A MuPDF exception, reified as a value: a category plus a formatted message.
///
/// This is what `fz_throw` records on the context before it `longjmp`s; here it
/// is simply the `Err` payload of [`Result`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    kind: ErrorKind,
    message: String,
}

impl Error {
    // MuPDF: fz_vthrow (error.c:333) -- code + vsnprintf'd message.
    /// Construct an error of `kind` with an already-formatted `message`.
    ///
    /// This is the `fz_throw`/`fz_vthrow` core. MuPDF's callers pass a printf
    /// format plus varargs; in Rust the caller formats with `format!` and hands
    /// the finished string here.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Error {
            kind,
            message: message.into(),
        }
    }

    /// The error category (`fz_caught`).
    // MuPDF: fz_caught (error.c:284)
    pub const fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The formatted message (`fz_caught_message`).
    // MuPDF: fz_caught_message (error.c:296)
    pub fn message(&self) -> &str {
        &self.message
    }

    /// This error's category name (`fz_error_type_name`).
    pub const fn name(&self) -> &'static str {
        self.kind.name()
    }

    /// Does this error's category equal `kind`? The predicate behind
    /// `fz_rethrow_if` / `fz_rethrow_unless` (which then re-`throw`, i.e. `?`).
    pub const fn matches(&self, kind: ErrorKind) -> bool {
        // (const `==` on the repr'd enum)
        self.kind as i32 == kind as i32
    }

    // MuPDF: fz_morph_error (error.c:372) -- "downgrade" one category to another.
    /// If this error's category is `from`, change it to `to` (used to downgrade
    /// exception severity within what would be a `catch` block). No-op otherwise.
    pub fn morph(&mut self, from: ErrorKind, to: ErrorKind) {
        if self.kind == from {
            self.kind = to;
        }
    }

    /// `FZ_ERROR_TRYLATER`: see [`ErrorKind::is_try_later`].
    pub const fn is_try_later(&self) -> bool {
        self.kind.is_try_later()
    }

    /// `FZ_ERROR_ABORT`: see [`ErrorKind::is_abort`].
    pub const fn is_abort(&self) -> bool {
        self.kind.is_abort()
    }

    /// A control signal (`TRYLATER`/`ABORT`): see
    /// [`ErrorKind::is_control_signal`].
    pub const fn is_control_signal(&self) -> bool {
        self.kind.is_control_signal()
    }
}

/// Per-category constructors mirroring `fz_throw(ctx, FZ_ERROR_x, fmt, …)`.
///
/// Each is the thin wrapper `Error::new(ErrorKind::X, msg)`; callers pre-format
/// the message with `format!` where MuPDF would pass printf varargs.
impl Error {
    // MuPDF: fz_throw (error.c:357) with each FZ_ERROR_x code (context.h:218).
    /// `fz_throw(ctx, FZ_ERROR_GENERIC, …)`.
    pub fn generic(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Generic, message)
    }
    /// `fz_throw(ctx, FZ_ERROR_SYSTEM, …)`.
    pub fn system(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::System, message)
    }
    /// `fz_throw(ctx, FZ_ERROR_LIBRARY, …)`.
    pub fn library(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Library, message)
    }
    /// `fz_throw(ctx, FZ_ERROR_ARGUMENT, …)`.
    pub fn argument(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Argument, message)
    }
    /// `fz_throw(ctx, FZ_ERROR_LIMIT, …)`.
    pub fn limit(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Limit, message)
    }
    /// `fz_throw(ctx, FZ_ERROR_UNSUPPORTED, …)`.
    pub fn unsupported(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Unsupported, message)
    }
    /// `fz_throw(ctx, FZ_ERROR_FORMAT, …)`.
    pub fn format(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Format, message)
    }
    /// `fz_throw(ctx, FZ_ERROR_SYNTAX, …)`.
    pub fn syntax(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Syntax, message)
    }
    /// `fz_throw(ctx, FZ_ERROR_TRYLATER, …)`.
    pub fn try_later(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::TryLater, message)
    }
    /// `fz_throw(ctx, FZ_ERROR_ABORT, …)`.
    pub fn abort(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Abort, message)
    }
    /// `fz_throw(ctx, FZ_ERROR_REPAIRED, …)`.
    pub fn repaired(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Repaired, message)
    }
}

// MuPDF: fz_report_error format "%s error: %s" (error.c:540)
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Matches the string MuPDF logs in fz_report_error:
        //     "%s error: %s", fz_error_type_name(code), message
        write!(f, "{} error: {}", self.kind.name(), self.message)
    }
}

impl std::error::Error for Error {}

/// The port-wide fallible result: `fz_try`/`fz_catch` becomes `Result`, where an
/// [`Error`] `Err` is a thrown exception and `Ok` is `FZ_ERROR_NONE`.
pub type Result<T> = core::result::Result<T, Error>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Every category, its constructor, its MuPDF name, and its enum discriminant
    /// (kept in one table so a drift in any of the three is caught).
    fn table() -> [(Error, ErrorKind, &'static str, i32); 11] {
        [
            (Error::generic("g"), ErrorKind::Generic, "generic", 1),
            (Error::system("s"), ErrorKind::System, "system", 2),
            (Error::library("l"), ErrorKind::Library, "library", 3),
            (Error::argument("a"), ErrorKind::Argument, "argument", 4),
            (Error::limit("li"), ErrorKind::Limit, "limit", 5),
            (Error::unsupported("u"), ErrorKind::Unsupported, "unsupported", 6),
            (Error::format("f"), ErrorKind::Format, "format", 7),
            (Error::syntax("sy"), ErrorKind::Syntax, "syntax", 8),
            (Error::try_later("t"), ErrorKind::TryLater, "trylater", 9),
            (Error::abort("ab"), ErrorKind::Abort, "abort", 10),
            (Error::repaired("r"), ErrorKind::Repaired, "repaired", 11),
        ]
    }

    #[test]
    fn constructors_set_the_right_kind() {
        for (err, kind, _, _) in table() {
            assert_eq!(err.kind(), kind);
        }
    }

    #[test]
    fn kind_names_match_mupdf_fz_error_type_name() {
        // These strings are exactly fz_error_type_name (error.c:393).
        for (err, kind, name, _) in table() {
            assert_eq!(kind.name(), name);
            assert_eq!(err.name(), name);
        }
    }

    #[test]
    fn discriminants_match_mupdf_enum_values() {
        // FZ_ERROR_NONE is 0 and unrepresented; GENERIC starts at 1.
        for (_, kind, _, code) in table() {
            assert_eq!(kind.code(), code);
            assert_eq!(kind as i32, code);
        }
    }

    #[test]
    fn display_matches_fz_report_error_format() {
        // "%s error: %s" of fz_error_type_name + message.
        assert_eq!(
            Error::syntax("unexpected token").to_string(),
            "syntax error: unexpected token"
        );
        assert_eq!(
            Error::format("bad xref").to_string(),
            "format error: bad xref"
        );
        // Every category renders "<name> error: <message>".
        for (err, _, name, _) in table() {
            let shown = err.to_string();
            assert!(shown.starts_with(name), "{shown} should start with {name}");
            assert!(shown.contains(" error: "));
        }
    }

    #[test]
    fn message_is_preserved_and_formattable() {
        // fz_throw takes a printf format; in Rust the caller uses format!.
        let e = Error::argument(format!("page {} out of range 0..{}", 7, 3));
        assert_eq!(e.message(), "page 7 out of range 0..3");
        assert_eq!(e.to_string(), "argument error: page 7 out of range 0..3");
    }

    #[test]
    fn matches_is_the_rethrow_if_predicate() {
        let e = Error::try_later("streaming");
        assert!(e.matches(ErrorKind::TryLater));
        assert!(!e.matches(ErrorKind::Abort));
    }

    #[test]
    fn morph_downgrades_only_the_matching_kind() {
        // fz_morph_error(ctx, FROM, TO): change kind only if it is FROM.
        let mut e = Error::format("unrecoverable");
        e.morph(ErrorKind::Format, ErrorKind::Syntax);
        assert_eq!(e.kind(), ErrorKind::Syntax);
        // Non-matching morph is a no-op.
        e.morph(ErrorKind::Abort, ErrorKind::Generic);
        assert_eq!(e.kind(), ErrorKind::Syntax);
        // Message is untouched by a morph.
        assert_eq!(e.message(), "unrecoverable");
    }

    #[test]
    fn control_signal_predicates_match_cluster_guards() {
        // Exactly TRYLATER and ABORT are control signals.
        for (err, kind, _, _) in table() {
            let expected = matches!(kind, ErrorKind::TryLater | ErrorKind::Abort);
            assert_eq!(err.is_control_signal(), expected);
            assert_eq!(kind.is_control_signal(), expected);
        }
        assert!(Error::try_later("x").is_try_later());
        assert!(!Error::try_later("x").is_abort());
        assert!(Error::abort("x").is_abort());
        assert!(!Error::abort("x").is_try_later());
    }

    #[test]
    fn recoverable_syntax_vs_unrecoverable_format() {
        // MuPDF's comments: SYNTAX "should be diagnosed and ignored" (recoverable),
        // FORMAT is "unrecoverable". They are distinct categories/codes.
        assert_ne!(ErrorKind::Syntax.code(), ErrorKind::Format.code());
        assert!(!ErrorKind::Format.is_control_signal());
        assert!(!ErrorKind::Syntax.is_control_signal());
    }

    #[test]
    fn usable_as_std_error_and_in_result() {
        fn may_fail(ok: bool) -> Result<u32> {
            if ok {
                Ok(42)
            } else {
                Err(Error::limit("too many objects"))
            }
        }
        assert_eq!(may_fail(true).unwrap(), 42);
        let err = may_fail(false).unwrap_err();
        // Exercise the std::error::Error object-safety path.
        let dyn_err: &dyn std::error::Error = &err;
        assert_eq!(dyn_err.to_string(), "limit error: too many objects");
        assert!(dyn_err.source().is_none());
    }
}
