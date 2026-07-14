use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use redb::{Database, ReadableDatabase, TableDefinition};
use serde::Serialize;
use serde::de::DeserializeOwned;

const KV: TableDefinition<&str, &[u8]> = TableDefinition::new("kv");

/// The name of KOPITIAM's per-project state directory, analogous to `.git`.
///
/// Living inside the project (rather than a platform config directory like
/// XDG's `~/.config` or Windows' `%APPDATA%`) is what makes this
/// automatically cross-platform: it is a plain relative path, so the same
/// code works unmodified on Linux, macOS, Windows, and Android (e.g. under
/// Termux, or inside whatever directory a host app grants a KOPITIAM
/// library build access to) — there is no per-OS branch to get wrong.
pub const PROJECT_DIR_NAME: &str = ".kopitiam";

/// Returns `root`'s `.kopitiam` state directory, without creating it.
pub fn project_dir(root: &Path) -> PathBuf {
    root.join(PROJECT_DIR_NAME)
}

/// Embedded, ACID key-value storage for one project's `.kopitiam` directory.
///
/// Backed by [`redb`] — pure Rust, no C dependency, matching this
/// workspace's Pure Rust Core rule (the vision this crate implements named
/// SQLite, which was rejected for exactly that reason; see the
/// "Semantic Runtime" section of `CLAUDE.md`).
///
/// This type is intentionally low-level (bytes in, bytes out); see
/// [`Store::put_json`] / [`Store::get_json`] for the typed convenience
/// callers actually want, and `kopitiam-workspace` for the project-state
/// type built on top of it.
pub struct Store {
    db: Database,
}

impl Store {
    /// Opens (creating if necessary) the `.kopitiam` directory and its
    /// state database under `root`.
    pub fn open(root: &Path) -> Result<Self> {
        let dir = project_dir(root);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let db_path = dir.join("state.redb");
        let db = Database::create(&db_path).with_context(|| format!("opening {}", db_path.display()))?;
        Ok(Self { db })
    }

    /// Stores raw bytes under `key`, overwriting any previous value.
    pub fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(KV)?;
            table.insert(key, value)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Reads the raw bytes stored under `key`, if any.
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let txn = self.db.begin_read()?;
        let table = match txn.open_table(KV) {
            Ok(table) => table,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(err) => return Err(err.into()),
        };
        Ok(table.get(key)?.map(|value| value.value().to_vec()))
    }

    /// Removes `key`, returning whether it was present.
    pub fn delete(&self, key: &str) -> Result<bool> {
        let txn = self.db.begin_write()?;
        let existed = {
            let mut table = txn.open_table(KV)?;
            table.remove(key)?.is_some()
        };
        txn.commit()?;
        Ok(existed)
    }

    /// Serializes `value` as JSON and stores it under `key`.
    pub fn put_json<T: Serialize>(&self, key: &str, value: &T) -> Result<()> {
        let bytes = serde_json::to_vec(value)?;
        self.put(key, &bytes)
    }

    /// Reads and deserializes the JSON value stored under `key`, if any.
    pub fn get_json<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        match self.get(key)? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[test]
    fn round_trips_raw_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();

        assert_eq!(store.get("missing").unwrap(), None);
        store.put("greeting", b"hello").unwrap();
        assert_eq!(store.get("greeting").unwrap(), Some(b"hello".to_vec()));

        assert!(store.delete("greeting").unwrap());
        assert_eq!(store.get("greeting").unwrap(), None);
        assert!(!store.delete("greeting").unwrap());
    }

    #[test]
    fn round_trips_json() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Example {
            name: String,
            count: u32,
        }

        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        let value = Example {
            name: "kopitiam".to_string(),
            count: 42,
        };
        store.put_json("example", &value).unwrap();
        let back: Example = store.get_json("example").unwrap().unwrap();
        assert_eq!(back, value);
    }

    #[test]
    fn creates_the_dot_kopitiam_directory() {
        let dir = tempfile::tempdir().unwrap();
        Store::open(dir.path()).unwrap();
        assert!(project_dir(dir.path()).is_dir());
    }

    #[test]
    fn persists_across_reopening_the_store() {
        let dir = tempfile::tempdir().unwrap();
        {
            let store = Store::open(dir.path()).unwrap();
            store.put("key", b"value").unwrap();
        }
        let reopened = Store::open(dir.path()).unwrap();
        assert_eq!(reopened.get("key").unwrap(), Some(b"value".to_vec()));
    }
}
