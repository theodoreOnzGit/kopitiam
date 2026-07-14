use std::path::Path;

/// A legacy source language KOPITIAM can translate from, into idiomatic
/// Rust, per the Translation Philosophy in `CLAUDE.md`: understand the
/// mathematics, the algorithm, the ownership, and the scientific
/// assumptions first, and only then produce Rust — never a mechanical
/// syntax transliteration.
///
/// This trait only identifies *which* source files belong to a language;
/// it does not yet parse or translate them. Concrete adapters (C, C++,
/// Fortran, ...) are future work — see the parent epic — layered on top of
/// this trait and [`crate::TranslationState`] once a first legacy codebase
/// is targeted.
pub trait LanguageAdapter {
    /// Stable identifier for this language (e.g. `"c"`, `"cpp"`,
    /// `"fortran"`), recorded in [`crate::TranslationState`] to say which
    /// adapter a given state belongs to.
    fn name(&self) -> &str;

    /// File extensions (without the leading `.`) this adapter's language
    /// uses, e.g. `["c", "h"]`.
    fn file_extensions(&self) -> &[&str];

    /// Whether `path` is a source file this adapter should translate.
    ///
    /// The default implementation matches on [`Self::file_extensions`];
    /// override it if a language needs more than an extension check (e.g.
    /// distinguishing Fortran fixed-form from free-form by content).
    fn claims(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| self.file_extensions().contains(&ext))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubC;

    impl LanguageAdapter for StubC {
        fn name(&self) -> &str {
            "c"
        }

        fn file_extensions(&self) -> &[&str] {
            &["c", "h"]
        }
    }

    #[test]
    fn default_claims_matches_on_extension() {
        let adapter = StubC;
        assert!(adapter.claims(Path::new("solver.c")));
        assert!(adapter.claims(Path::new("solver.h")));
        assert!(!adapter.claims(Path::new("solver.rs")));
        assert!(!adapter.claims(Path::new("solver")));
    }
}
