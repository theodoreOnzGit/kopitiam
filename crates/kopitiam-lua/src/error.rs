//! Errors, and why a Lua error is not a string.
//!
//! Lua's `error()` accepts **any value**, not just a message:
//!
//! ```lua
//! error({ code = 404, msg = "not found" })   -- a table
//! local ok, e = pcall(f)                     -- e is that same table
//! ```
//!
//! Plugins rely on this (it is how structured errors are passed around), so
//! [`LuaError::Runtime`] carries a [`Value`] rather than a `String`. Flattening
//! it to a string at the boundary would be lossy in a way `pcall` can observe.
//!
//! Lexer and parser errors, by contrast, genuinely *are* just messages with a
//! source location, and are kept separate so a host can tell "your config has a
//! syntax error" from "your config threw".

use std::fmt;

use crate::value::Value;

/// The result of anything that can fail inside the interpreter.
pub type Result<T> = std::result::Result<T, LuaError>;

/// Everything that can go wrong.
#[derive(Debug, Clone)]
pub enum LuaError {
    /// The source did not lex or parse. `chunk` and `line` locate it.
    Syntax { chunk: String, line: u32, message: String },

    /// A runtime error: a bad operation, or an explicit `error(v)`.
    ///
    /// `value` is whatever was thrown — commonly a string, but not necessarily.
    /// `traceback` is the Lua call stack at the point of the throw, innermost
    /// first, and is best-effort: it exists to make a broken config debuggable,
    /// not to be machine-parsed.
    Runtime { value: Value, traceback: Vec<String> },
}

impl LuaError {
    /// A runtime error carrying a plain message string, with no location
    /// prefix. Used for errors raised by the VM itself (`attempt to index a nil
    /// value`, ...) before the VM has had a chance to attach a position.
    pub(crate) fn runtime(message: impl Into<String>) -> Self {
        LuaError::Runtime { value: Value::from(message.into()), traceback: Vec::new() }
    }

    /// The thrown value, for a runtime error. Syntax errors are converted to a
    /// string value, since `pcall` around a `load()` must be able to see them.
    pub fn value(&self) -> Value {
        match self {
            LuaError::Runtime { value, .. } => value.clone(),
            LuaError::Syntax { .. } => Value::from(self.to_string()),
        }
    }
}

impl fmt::Display for LuaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LuaError::Syntax { chunk, line, message } => {
                write!(f, "{chunk}:{line}: {message}")
            }
            LuaError::Runtime { value, traceback } => {
                // A thrown table has no useful Display, so say what it was
                // rather than printing `table: 0x...` and calling it a message.
                match value {
                    Value::String(s) => write!(f, "{}", s.to_string_lossy())?,
                    Value::Nil => write!(f, "nil")?,
                    other => write!(f, "(error object is a {} value)", other.type_name())?,
                }
                for frame in traceback {
                    write!(f, "\n\t{frame}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for LuaError {}
