use std::path::Path;

use anyhow::{Result, bail};
use indexmap::IndexMap;
use kopitiam_index::Store;
use serde::{Deserialize, Serialize};

const STATE_KEY: &str = "translation/state";

/// Where one [`TranslationUnit`] sits in the translation pipeline
/// (`legacy source -> language adapter -> semantic model -> runtime
/// knowledge -> translation workflow -> verification -> persistent
/// translation state`, per `CLAUDE.md`'s Architecture table).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitStatus {
    /// Registered but not yet translated.
    Pending,
    /// A translation workflow is actively working on this unit.
    InProgress,
    /// Rust was produced but has not passed verification.
    Translated,
    /// Rust was produced and verified (tests, benchmarks, or manual
    /// review against the original — see `CLAUDE.md`'s Scientific
    /// Standards) to preserve the original's behavior.
    Verified,
    /// Translation was attempted and abandoned or failed; see
    /// [`TranslationUnit::notes`] for why.
    Failed,
}

/// One legacy source file/module tracked through translation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslationUnit {
    /// Path to the original source, relative to the legacy codebase root.
    pub source_path: String,
    /// Path to the produced Rust, once [`UnitStatus::Translated`] or later.
    pub target_path: Option<String>,
    pub status: UnitStatus,
    /// Free-form context: why a unit is [`UnitStatus::Failed`], validation
    /// strategy notes, or anything else worth preserving per `CLAUDE.md`'s
    /// Scientific Standards (provenance, assumptions, validation).
    pub notes: Option<String>,
}

impl TranslationUnit {
    fn new(source_path: impl Into<String>) -> Self {
        Self { source_path: source_path.into(), target_path: None, status: UnitStatus::Pending, notes: None }
    }
}

/// Persistent state for one legacy-codebase-to-Rust translation effort:
/// which [`crate::LanguageAdapter`] it belongs to, and the status of every
/// [`TranslationUnit`] registered so far.
///
/// Units are kept in an [`IndexMap`] (insertion order), not a `HashMap`,
/// so [`Self::units`] and any report built from it list units in a stable,
/// reproducible order — matching the "deterministic behaviour" engineering
/// principle in `CLAUDE.md`.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TranslationState {
    /// The [`crate::LanguageAdapter::name`] this state was built with, if
    /// any unit has been registered yet.
    pub language: Option<String>,
    units: IndexMap<String, TranslationUnit>,
}

impl TranslationState {
    /// Loads the translation state for `root` from its `.kopitiam`
    /// directory, or returns a fresh, empty state if none has been saved.
    pub fn load(root: &Path) -> Result<Self> {
        let store = Store::open(root)?;
        Ok(store.get_json(STATE_KEY)?.unwrap_or_default())
    }

    /// Persists this state to `root`'s `.kopitiam` directory.
    pub fn save(&self, root: &Path) -> Result<()> {
        let store = Store::open(root)?;
        store.put_json(STATE_KEY, self)
    }

    /// Registers `source_path` as a unit to translate for `language`, in
    /// [`UnitStatus::Pending`]. Re-registering an already-known path is a
    /// no-op that leaves its current status untouched.
    ///
    /// # Errors
    /// Returns an error if `language` disagrees with a `language` this
    /// state was already registering units for — one `TranslationState`
    /// tracks one legacy language at a time.
    pub fn register_unit(&mut self, language: &str, source_path: impl Into<String>) -> Result<()> {
        match &self.language {
            Some(existing) if existing != language => {
                bail!("translation state is tracking language {existing:?}, not {language:?}");
            }
            _ => self.language = Some(language.to_string()),
        }

        let source_path = source_path.into();
        self.units.entry(source_path.clone()).or_insert_with(|| TranslationUnit::new(source_path));
        Ok(())
    }

    /// Transitions `source_path` to [`UnitStatus::Translated`], recording
    /// where the produced Rust landed.
    pub fn mark_translated(&mut self, source_path: &str, target_path: impl Into<String>) -> Result<()> {
        let unit = self.unit_mut(source_path)?;
        unit.status = UnitStatus::Translated;
        unit.target_path = Some(target_path.into());
        Ok(())
    }

    /// Transitions `source_path` to [`UnitStatus::Verified`]. Errors if
    /// the unit was never marked [`UnitStatus::Translated`] first, since a
    /// unit with no produced Rust cannot have been verified.
    pub fn mark_verified(&mut self, source_path: &str) -> Result<()> {
        let unit = self.unit_mut(source_path)?;
        if unit.target_path.is_none() {
            bail!("cannot verify {source_path:?}: it has not been translated yet");
        }
        unit.status = UnitStatus::Verified;
        Ok(())
    }

    /// Transitions `source_path` to [`UnitStatus::Failed`], recording why.
    pub fn mark_failed(&mut self, source_path: &str, reason: impl Into<String>) -> Result<()> {
        let unit = self.unit_mut(source_path)?;
        unit.status = UnitStatus::Failed;
        unit.notes = Some(reason.into());
        Ok(())
    }

    fn unit_mut(&mut self, source_path: &str) -> Result<&mut TranslationUnit> {
        self.units
            .get_mut(source_path)
            .ok_or_else(|| anyhow::anyhow!("no translation unit registered for {source_path:?}"))
    }

    pub fn unit(&self, source_path: &str) -> Option<&TranslationUnit> {
        self.units.get(source_path)
    }

    /// All registered units, in registration order.
    pub fn units(&self) -> impl Iterator<Item = &TranslationUnit> {
        self.units.values()
    }

    /// Units in [`UnitStatus::Translated`] or [`UnitStatus::Verified`] —
    /// i.e. no longer blocking on translation work, whether or not
    /// verification has run yet.
    pub fn completed_count(&self) -> usize {
        self.units
            .values()
            .filter(|u| matches!(u.status, UnitStatus::Translated | UnitStatus::Verified))
            .count()
    }

    /// Units in [`UnitStatus::Pending`] or [`UnitStatus::InProgress`].
    pub fn remaining_count(&self) -> usize {
        self.units.values().filter(|u| matches!(u.status, UnitStatus::Pending | UnitStatus::InProgress)).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registering_twice_is_a_no_op() {
        let mut state = TranslationState::default();
        state.register_unit("c", "solver.c").unwrap();
        state.register_unit("c", "solver.c").unwrap();
        assert_eq!(state.units().count(), 1);
    }

    #[test]
    fn rejects_a_second_language_in_the_same_state() {
        let mut state = TranslationState::default();
        state.register_unit("c", "solver.c").unwrap();
        assert!(state.register_unit("fortran", "other.f90").is_err());
    }

    #[test]
    fn walks_a_unit_through_the_full_lifecycle() {
        let mut state = TranslationState::default();
        state.register_unit("c", "solver.c").unwrap();
        assert_eq!(state.unit("solver.c").unwrap().status, UnitStatus::Pending);
        assert_eq!(state.remaining_count(), 1);
        assert_eq!(state.completed_count(), 0);

        state.mark_translated("solver.c", "solver.rs").unwrap();
        assert_eq!(state.unit("solver.c").unwrap().status, UnitStatus::Translated);
        assert_eq!(state.unit("solver.c").unwrap().target_path.as_deref(), Some("solver.rs"));
        assert_eq!(state.completed_count(), 1);
        assert_eq!(state.remaining_count(), 0);

        state.mark_verified("solver.c").unwrap();
        assert_eq!(state.unit("solver.c").unwrap().status, UnitStatus::Verified);
        assert_eq!(state.completed_count(), 1);
    }

    #[test]
    fn cannot_verify_a_unit_that_was_never_translated() {
        let mut state = TranslationState::default();
        state.register_unit("c", "solver.c").unwrap();
        assert!(state.mark_verified("solver.c").is_err());
    }

    #[test]
    fn mark_failed_records_a_reason() {
        let mut state = TranslationState::default();
        state.register_unit("c", "solver.c").unwrap();
        state.mark_failed("solver.c", "no idiomatic Rust equivalent yet for this BLAS call").unwrap();
        let unit = state.unit("solver.c").unwrap();
        assert_eq!(unit.status, UnitStatus::Failed);
        assert_eq!(unit.notes.as_deref(), Some("no idiomatic Rust equivalent yet for this BLAS call"));
    }

    #[test]
    fn operating_on_an_unregistered_unit_errors() {
        let mut state = TranslationState::default();
        assert!(state.mark_translated("unknown.c", "unknown.rs").is_err());
    }

    #[test]
    fn persists_across_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let mut state = TranslationState::load(dir.path()).unwrap();
        state.register_unit("c", "solver.c").unwrap();
        state.mark_translated("solver.c", "solver.rs").unwrap();
        state.save(dir.path()).unwrap();

        let reloaded = TranslationState::load(dir.path()).unwrap();
        assert_eq!(reloaded.language.as_deref(), Some("c"));
        assert_eq!(reloaded.unit("solver.c").unwrap().status, UnitStatus::Translated);
    }
}
