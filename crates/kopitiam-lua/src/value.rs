//! Lua's eight types, and the two that carry all the subtlety: strings and
//! tables.
//!
//! # Lua 5.1 has exactly one number type
//!
//! `Number(f64)`. Not an int/float pair — that arrived in Lua 5.3, and adopting
//! it here would change division, `%`, `tostring`, and table-key identity in
//! ways that silently diverge from the dialect Neovim actually runs. `1` and
//! `1.0` are the *same value* and the *same table key*, and that is correct.
//!
//! # Truthiness
//!
//! **Only `nil` and `false` are false.** `0` is true. `""` is true. This is the
//! single most common thing to get wrong when porting from a C-family language,
//! and it fails *silently* — an `if #t == 0` that should have been taken, isn't.
//! It lives in [`Value::is_truthy`] with a test pinning it.

use std::cell::RefCell;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use indexmap::IndexMap;

use crate::error::Result;
use crate::instr::Proto;

/// An immutable Lua string.
///
/// # Why bytes and not `String`
///
/// Lua strings are **arbitrary byte sequences**, not UTF-8. `string.char(200)`
/// is a legal one-byte Lua string that is not valid UTF-8, and `string.byte`,
/// `#s`, and the pattern matcher all index by *byte*. Storing a Rust `String`
/// would make those operations lie about their own semantics. So: bytes.
///
/// # Cheap cloning
///
/// `Rc<[u8]>`, so cloning is a refcount bump — Lua values are copied constantly
/// (every stack push, every table read), and a deep copy per clone would be a
/// disaster.
///
/// # Not interned, and that is deliberate (for now)
///
/// Real Lua interns *every* string in a global table, which makes equality a
/// pointer compare. We hash and compare by content instead. The cost is real
/// but bounded (string equality is the only hot path affected), and the benefit
/// is that constructing a `LuaStr` needs no access to the VM — which keeps
/// `Value: From<&str>` possible and the whole crate far simpler to reason
/// about. Interning is a genuine Phase-2 optimisation; it is not a correctness
/// gap, because content equality and pointer equality agree on an interned
/// pool anyway.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LuaStr(Rc<[u8]>);

impl LuaStr {
    pub fn from_bytes(b: &[u8]) -> Self {
        LuaStr(Rc::from(b))
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The string as UTF-8, replacing invalid sequences. For display and for
    /// handing back to a Rust host that wants a `String`. Lossy by name so that
    /// nobody assumes round-tripping.
    pub fn to_string_lossy(&self) -> String {
        String::from_utf8_lossy(&self.0).into_owned()
    }
}

impl From<&str> for LuaStr {
    fn from(s: &str) -> Self {
        LuaStr::from_bytes(s.as_bytes())
    }
}

impl From<String> for LuaStr {
    fn from(s: String) -> Self {
        LuaStr::from_bytes(s.as_bytes())
    }
}

impl fmt::Debug for LuaStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.to_string_lossy())
    }
}

/// A Lua value.
///
/// `Default` is `Nil`, which is not an arbitrary choice: an absent Lua value IS
/// nil, so `unwrap_or_default()` on a missing argument or table slot gives
/// exactly the right answer.
#[derive(Clone, Default)]
pub enum Value {
    #[default]
    Nil,
    Boolean(bool),
    Number(f64),
    String(LuaStr),
    Table(Rc<RefCell<Table>>),
    /// A Lua closure: compiled code plus its captured upvalues.
    Function(Rc<Closure>),
    /// A Rust function exposed to Lua. This is the injection point kvim uses
    /// for `vim.keymap.set`, `vim.opt`, and friends.
    Native(Rc<NativeFunction>),
    Coroutine(Rc<RefCell<Coroutine>>),
}

impl Value {
    /// **Only `nil` and `false` are falsy.** Everything else — including `0`,
    /// `""`, `0.0`, and NaN — is truthy.
    ///
    /// If you are reading this because a condition is behaving oddly: yes,
    /// really. `if 0 then` is taken in Lua.
    #[inline]
    pub fn is_truthy(&self) -> bool {
        !matches!(self, Value::Nil | Value::Boolean(false))
    }

    /// The name Lua's own error messages use: `nil`, `boolean`, `number`,
    /// `string`, `table`, `function`, `thread`. Note that both `Function` and
    /// `Native` report `"function"` — Lua code cannot tell them apart, and
    /// leaking the distinction would break `type(f) == "function"` checks that
    /// plugins really do make.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Nil => "nil",
            Value::Boolean(_) => "boolean",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Table(_) => "table",
            Value::Function(_) | Value::Native(_) => "function",
            Value::Coroutine(_) => "thread",
        }
    }

    /// Raw equality, with no `__eq` metamethod. This is `rawequal`.
    ///
    /// Tables, functions and coroutines compare by **identity** (pointer), not
    /// contents — two distinct empty tables are not equal. Numbers compare by
    /// value, so `1 == 1.0`. NaN is not equal to itself, as in every IEEE-754
    /// language.
    pub fn raw_eq(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Nil, Value::Nil) => true,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Number(a), Value::Number(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Table(a), Value::Table(b)) => Rc::ptr_eq(a, b),
            (Value::Function(a), Value::Function(b)) => Rc::ptr_eq(a, b),
            (Value::Native(a), Value::Native(b)) => Rc::ptr_eq(a, b),
            (Value::Coroutine(a), Value::Coroutine(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// The value as a number, applying Lua's *implicit string-to-number
    /// coercion*: `"10" + 1` is `11` in Lua, because arithmetic on a string
    /// tries `tonumber` first. Returns `None` if it is not coercible.
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            Value::String(s) => crate::number::parse_number(s.as_bytes()),
            _ => None,
        }
    }

    /// The value as a string, applying Lua's *implicit number-to-string
    /// coercion* (`1 .. "x"` is `"1x"`). Does **not** call `__tostring`; that
    /// is the VM's job, since it may need to call back into Lua.
    pub fn as_lua_string(&self) -> Option<LuaStr> {
        match self {
            Value::String(s) => Some(s.clone()),
            Value::Number(n) => Some(LuaStr::from(crate::number::format_number(*n))),
            _ => None,
        }
    }

    /// The metatable, if any. Only tables carry per-value metatables here.
    ///
    /// Real Lua also allows a shared metatable for *all* strings (that is how
    /// `("x"):upper()` resolves). We special-case that in the VM's index path
    /// rather than storing it per-string, which would cost a pointer on every
    /// string in the program to model a table there is exactly one of.
    pub fn metatable(&self) -> Option<Rc<RefCell<Table>>> {
        match self {
            Value::Table(t) => t.borrow().metatable.clone(),
            _ => None,
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Nil => write!(f, "nil"),
            Value::Boolean(b) => write!(f, "{b}"),
            Value::Number(n) => write!(f, "{}", crate::number::format_number(*n)),
            Value::String(s) => write!(f, "{s:?}"),
            Value::Table(t) => write!(f, "table: {:p}", Rc::as_ptr(t)),
            Value::Function(c) => write!(f, "function: {:p}", Rc::as_ptr(c)),
            Value::Native(n) => write!(f, "function: builtin: {}", n.name),
            Value::Coroutine(c) => write!(f, "thread: {:p}", Rc::as_ptr(c)),
        }
    }
}

// Convenience conversions, so a Rust host can write `Value::from(3)` and
// `Value::from("hi")` rather than spelling out the variant every time. These
// exist for the embedding API's ergonomics, which the brief called out
// explicitly.
impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Boolean(b)
    }
}
impl From<f64> for Value {
    fn from(n: f64) -> Self {
        Value::Number(n)
    }
}
impl From<i64> for Value {
    fn from(n: i64) -> Self {
        Value::Number(n as f64)
    }
}
impl From<usize> for Value {
    fn from(n: usize) -> Self {
        Value::Number(n as f64)
    }
}
impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(LuaStr::from(s))
    }
}
impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(LuaStr::from(s))
    }
}
impl From<LuaStr> for Value {
    fn from(s: LuaStr) -> Self {
        Value::String(s)
    }
}

/// A table key.
///
/// Lua forbids `nil` and NaN as keys (both would break lookup: `nil` because it
/// means "absent", NaN because `NaN ~= NaN` so you could never find it again).
/// Making `Key` a type that *cannot represent them* pushes that rule into the
/// type system instead of leaving it as a runtime check people forget.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Key {
    Boolean(bool),
    /// f64 by bit pattern. [`Key::from_value`] guarantees no NaN, and folds
    /// `-0.0` into `0.0` so that `t[-0.0]` and `t[0]` are the same slot — which
    /// they must be, since `-0.0 == 0.0` is true.
    Number(u64),
    String(LuaStr),
    /// A table, function or coroutine used as a key — hashed and compared by
    /// *identity*, as Lua requires.
    Object(ObjKey),
}

/// An object key: a `Value` that hashes and compares by address.
///
/// # Why this holds the `Value` instead of a raw pointer
///
/// The obvious encoding is `Key::Table(*const RefCell<Table>)`. It is wrong
/// twice over. First, a raw pointer is not an owning handle, so the table used
/// as a key could be dropped while it is still a live key — and the address
/// could then be *reused* by a different table, silently aliasing two distinct
/// keys. Second, `next()` has to hand the key back to Lua as a `Value`, and a
/// `Value` cannot be resurrected from a raw pointer.
///
/// Holding the `Value` fixes both: the `Rc` inside keeps the object alive for
/// exactly as long as it is a key, and `next()` can simply clone it.
#[derive(Clone)]
pub struct ObjKey(Value);

impl ObjKey {
    /// The identity of the referenced object, as an address.
    fn addr(&self) -> usize {
        match &self.0 {
            Value::Table(t) => Rc::as_ptr(t) as usize,
            Value::Function(f) => Rc::as_ptr(f) as usize,
            Value::Native(f) => Rc::as_ptr(f) as usize,
            Value::Coroutine(c) => Rc::as_ptr(c) as usize,
            // `ObjKey` is only ever constructed from the four variants above.
            _ => 0,
        }
    }
}

impl PartialEq for ObjKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.raw_eq(&other.0)
    }
}
impl Eq for ObjKey {}
impl Hash for ObjKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.addr().hash(state);
    }
}

impl Key {
    /// Converts a value to a key, or `None` if Lua forbids it as one
    /// (`nil` and NaN).
    pub fn from_value(v: &Value) -> Option<Key> {
        Some(match v {
            Value::Nil => return None,
            Value::Number(n) if n.is_nan() => return None,
            Value::Boolean(b) => Key::Boolean(*b),
            Value::Number(n) => {
                let n = if *n == 0.0 { 0.0 } else { *n }; // fold -0.0 into 0.0
                Key::Number(n.to_bits())
            }
            Value::String(s) => Key::String(s.clone()),
            Value::Table(_) | Value::Function(_) | Value::Native(_) | Value::Coroutine(_) => {
                Key::Object(ObjKey(v.clone()))
            }
        })
    }
}

/// A Lua table: the language's only data structure.
///
/// # Why two parts
///
/// Lua tables are simultaneously arrays and hash maps, and `#t`, `ipairs`, and
/// `table.insert` all care about the array-like prefix. Keeping a dense `Vec`
/// for keys `1..=n` makes those operations O(1) instead of hashing, which is
/// what makes idiomatic Lua (`t[#t+1] = v` in a loop) not quadratic.
///
/// Invariant: `array` never ends in `Nil`. Interior nils are allowed (setting
/// `t[2] = nil` on a 3-element array leaves a hole), because Lua's `#` is
/// defined as *any* border, not the first one.
pub struct Table {
    array: Vec<Value>,
    /// Insertion-ordered (see the `indexmap` justification in Cargo.toml).
    ///
    /// Absent keys are `Nil` **tombstones** rather than removals. That is not
    /// laziness: Lua explicitly permits `t[k] = nil` *during* a `pairs()`
    /// traversal, and `next` is stateless — it re-finds the previous key and
    /// steps forward. If clearing a key shifted every later key's index, an
    /// in-progress `pairs()` would skip entries. Tombstoning keeps every
    /// surviving key's index stable, so the traversal stays correct.
    map: IndexMap<Key, Value>,
    pub metatable: Option<Rc<RefCell<Table>>>,
}

impl Default for Table {
    fn default() -> Self {
        Table::new()
    }
}

impl Table {
    pub fn new() -> Self {
        Table { array: Vec::new(), map: IndexMap::new(), metatable: None }
    }

    /// A raw read: no `__index`. This is `rawget`.
    pub fn raw_get(&self, key: &Value) -> Value {
        // The array fast path: an integral number in 1..=len.
        if let Value::Number(n) = key
            && let Some(i) = array_index(*n)
            && i <= self.array.len()
        {
            return self.array[i - 1].clone();
        }
        match Key::from_value(key) {
            Some(k) => self.map.get(&k).cloned().unwrap_or(Value::Nil),
            None => Value::Nil,
        }
    }

    /// `rawget` by string key — the overwhelmingly common case (`t.field`),
    /// worth not building a `Value` for.
    pub fn raw_get_str(&self, key: &str) -> Value {
        self.map.get(&Key::String(LuaStr::from(key))).cloned().unwrap_or(Value::Nil)
    }

    /// A raw write: no `__newindex`. This is `rawset`.
    ///
    /// Returns an error for the two keys Lua forbids, rather than silently
    /// dropping the write.
    pub fn raw_set(&mut self, key: Value, value: Value) -> Result<()> {
        if let Value::Number(n) = key
            && let Some(i) = array_index(n)
        {
            if i <= self.array.len() {
                self.array[i - 1] = value;
                // Keep the "no trailing nil" invariant.
                while matches!(self.array.last(), Some(Value::Nil)) {
                    self.array.pop();
                }
                return Ok(());
            }
            if i == self.array.len() + 1 {
                if matches!(value, Value::Nil) {
                    // Clearing one past the end is a no-op, but it may still be
                    // a live key in the hash part (set out of order), so fall
                    // through to the map for the tombstone.
                } else {
                    self.array.push(value);
                    self.migrate_from_map();
                    return Ok(());
                }
            }
        }

        let Some(k) = Key::from_value(&key) else {
            return Err(crate::error::LuaError::runtime(match key {
                Value::Nil => "table index is nil".to_string(),
                _ => "table index is NaN".to_string(),
            }));
        };

        if matches!(value, Value::Nil) {
            // Tombstone rather than remove, so an in-flight `pairs()` is not
            // derailed. Only tombstone a key that exists; inserting a fresh nil
            // would be pure garbage.
            if let Some(slot) = self.map.get_mut(&k) {
                *slot = Value::Nil;
            }
        } else {
            self.map.insert(k, value);
        }
        Ok(())
    }

    /// After appending to the array part, keys `len+1, len+2, ...` may already
    /// be sitting in the hash part because they were assigned out of order
    /// (`t[3]=c; t[1]=a; t[2]=b`). Pull them across so that `#t` and `ipairs`
    /// see the whole run.
    fn migrate_from_map(&mut self) {
        loop {
            let next = Key::Number(((self.array.len() + 1) as f64).to_bits());
            match self.map.get(&next) {
                Some(Value::Nil) | None => break,
                Some(v) => {
                    let v = v.clone();
                    self.array.push(v);
                    // Tombstone rather than shift-remove: same `pairs()`
                    // stability argument as in `raw_set`.
                    if let Some(slot) = self.map.get_mut(&next) {
                        *slot = Value::Nil;
                    }
                }
            }
        }
    }

    /// Convenience for the stdlib and for hosts: `t.name = v`.
    pub fn set_str(&mut self, key: &str, value: Value) {
        // Cannot fail: a string is always a legal key.
        let _ = self.raw_set(Value::from(key), value);
    }

    /// The `#` operator's raw answer (no `__len`).
    ///
    /// Lua defines `#t` on a table with holes as *any* border `n` where
    /// `t[n] ~= nil and t[n+1] == nil`. We return the array part's length,
    /// which is always such a border given our no-trailing-nil invariant. It is
    /// deterministic, which is more than the reference implementation promises.
    pub fn raw_len(&self) -> usize {
        self.array.len()
    }

    /// `next(t, key)` — the stateless iterator underneath `pairs`.
    ///
    /// Walks the array part first, then the hash part, skipping tombstones.
    /// `None` means "no more"; `Some((k, v))` is the next pair.
    pub fn next_key(&self, key: &Value) -> Result<Option<(Value, Value)>> {
        // Where do we resume from? Encode a position as: 0..array.len() are
        // array slots, then array.len()+i are map slots.
        let start = match key {
            Value::Nil => 0,
            Value::Number(n) if array_index(*n).is_some_and(|i| i <= self.array.len()) => {
                array_index(*n).unwrap() // 1-based key i => next position is i
            }
            other => {
                let Some(k) = Key::from_value(other) else {
                    return Err(crate::error::LuaError::runtime("invalid key to 'next'"));
                };
                match self.map.get_index_of(&k) {
                    Some(i) => self.array.len() + i + 1,
                    None => {
                        return Err(crate::error::LuaError::runtime("invalid key to 'next'"));
                    }
                }
            }
        };

        for pos in start..self.array.len() {
            if !matches!(self.array[pos], Value::Nil) {
                return Ok(Some((Value::Number((pos + 1) as f64), self.array[pos].clone())));
            }
        }
        let map_start = start.saturating_sub(self.array.len());
        for i in map_start..self.map.len() {
            let (k, v) = self.map.get_index(i).unwrap();
            if !matches!(v, Value::Nil) {
                return Ok(Some((key_to_value(k), v.clone())));
            }
        }
        Ok(None)
    }

    /// The array part, for `table.concat`/`sort`/`unpack`, which all operate on
    /// the sequence and would otherwise round-trip through `raw_get`.
    pub fn array(&self) -> &[Value] {
        &self.array
    }

    /// `table.insert(t, pos, v)` — shifts the tail up.
    pub fn insert(&mut self, pos: usize, value: Value) {
        if pos >= 1 && pos <= self.array.len() + 1 {
            self.array.insert(pos - 1, value);
            self.migrate_from_map();
        } else {
            let _ = self.raw_set(Value::from(pos), value);
        }
    }

    /// `table.remove(t, pos)` — shifts the tail down, returns what was there.
    pub fn remove(&mut self, pos: usize) -> Value {
        if pos >= 1 && pos <= self.array.len() {
            let v = self.array.remove(pos - 1);
            while matches!(self.array.last(), Some(Value::Nil)) {
                self.array.pop();
            }
            v
        } else {
            Value::Nil
        }
    }

    /// Replaces the array part wholesale — `table.sort`'s write-back.
    pub fn set_array(&mut self, values: Vec<Value>) {
        self.array = values;
    }
}

/// Reconstructs the `Value` a `Key` came from — total, and lossless, which is
/// exactly why [`ObjKey`] stores a `Value` rather than an address.
///
/// `next()` needs this: it must hand the *key* back to Lua on every step of a
/// `pairs()` loop.
fn key_to_value(k: &Key) -> Value {
    match k {
        Key::Boolean(b) => Value::Boolean(*b),
        Key::Number(bits) => Value::Number(f64::from_bits(*bits)),
        Key::String(s) => Value::String(s.clone()),
        Key::Object(o) => o.0.clone(),
    }
}

/// Is this number a usable 1-based array index?
///
/// Must be a positive integer with no fractional part. `2.0` is; `2.5` is not;
/// `0` and negatives are not (Lua arrays are 1-based).
#[inline]
fn array_index(n: f64) -> Option<usize> {
    if n.fract() == 0.0 && n >= 1.0 && n <= (usize::MAX as f64) {
        Some(n as usize)
    } else {
        None
    }
}

/// A compiled Lua function together with the upvalues it closed over.
///
/// `upvalues` are `Rc<RefCell<Value>>` **cells**, shared with the enclosing
/// frame — this is what makes Lua's capture *by reference* rather than by
/// value. Two closures created in the same scope see each other's writes; a
/// closure created in a fresh loop iteration sees a fresh cell. Both fall out of
/// sharing the cell rather than copying the value.
pub struct Closure {
    pub proto: Rc<Proto>,
    pub upvalues: Vec<Rc<RefCell<Value>>>,
}

/// What a native function hands back to the VM.
///
/// Most natives just return values. A few need to do something only the VM can
/// do — start a protected call, resume a coroutine, suspend one — and they
/// cannot do it by calling back into the VM recursively without breaking
/// yieldability (see AID-0007). So they *describe* the action and let the VM
/// perform it on the frame stack.
pub enum Outcome {
    /// Ordinary return.
    Return(Vec<Value>),
    /// `pcall`/`xpcall`: call `f(args)` under a protected frame.
    Protected { f: Value, args: Vec<Value>, handler: Option<Value> },
    /// `coroutine.resume` / a `coroutine.wrap` closure.
    Resume { co: Rc<RefCell<Coroutine>>, args: Vec<Value>, wrapped: bool },
    /// `coroutine.yield`.
    Yield(Vec<Value>),
}

/// A Rust function callable from Lua.
///
/// Two flavours, because the ergonomic one cannot express `pcall`:
///
/// * [`NativeKind::Simple`] — `Fn(&mut Lua, Vec<Value>) -> Result<Vec<Value>>`.
///   This is the *only* kind a host can construct, and it is what kvim will use
///   to inject `vim.keymap.set` and the rest.
/// * [`NativeKind::Vm`] — may additionally request a VM-level action
///   ([`Outcome`]). Reserved for the handful of stdlib functions that are really
///   control flow wearing a function's clothes.
pub struct NativeFunction {
    pub name: String,
    pub kind: NativeKind,
}

/// The ergonomic native: arguments in, values out.
pub type SimpleNative = Box<dyn Fn(&mut crate::Lua, Vec<Value>) -> Result<Vec<Value>>>;

/// The privileged native: may additionally request a VM-level action.
pub type VmNative = Box<dyn Fn(&mut crate::Lua, Vec<Value>) -> Result<Outcome>>;

pub enum NativeKind {
    Simple(SimpleNative),
    Vm(VmNative),
}

/// A coroutine — Lua's `thread` type.
///
/// The whole of AID-0007 exists to make this struct possible: a suspended
/// coroutine is *literally its own call-frame stack*, lifted out of the VM and
/// parked here. Resuming is pushing it back. There is nothing to reconstruct,
/// because nothing was ever unwound.
pub struct Coroutine {
    pub status: CoStatus,
    /// The parked frames. Empty before the first resume, when `entry` holds the
    /// function instead.
    pub(crate) frames: Vec<crate::vm::Frame>,
    pub(crate) entry: Option<Value>,
    /// Where the *next* resume's arguments must be delivered: straight back into
    /// the `coroutine.yield(...)` call site that is still parked in `frames`,
    /// truncated to however many values it asked for.
    ///
    /// It has to be remembered here because the values arrive later, from a
    /// `resume` that has not happened yet — and by then the call site that wants
    /// them is buried inside a suspended frame.
    pub(crate) yield_ret: Option<crate::vm::ReturnMode>,
}

impl Coroutine {
    /// A new, unstarted coroutine wrapping `f`.
    pub(crate) fn new(f: Value) -> Self {
        Coroutine {
            status: CoStatus::Suspended,
            frames: Vec::new(),
            entry: Some(f),
            yield_ret: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoStatus {
    /// Created or yielded; can be resumed.
    Suspended,
    /// Currently executing.
    Running,
    /// Resumed another coroutine, and is waiting for it.
    Normal,
    /// Returned, or errored out.
    Dead,
}

impl CoStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            CoStatus::Suspended => "suspended",
            CoStatus::Running => "running",
            CoStatus::Normal => "normal",
            CoStatus::Dead => "dead",
        }
    }
}

/// Wraps a value in a fresh table `Rc`. Used everywhere a new table is born.
pub fn new_table() -> Rc<RefCell<Table>> {
    Rc::new(RefCell::new(Table::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_nil_and_false_are_falsy() {
        // The single most dangerous thing to get wrong. `0` and `""` are TRUE.
        assert!(!Value::Nil.is_truthy());
        assert!(!Value::Boolean(false).is_truthy());

        assert!(Value::Boolean(true).is_truthy());
        assert!(Value::Number(0.0).is_truthy(), "0 is truthy in Lua");
        assert!(Value::Number(-0.0).is_truthy());
        assert!(Value::Number(f64::NAN).is_truthy());
        assert!(Value::from("").is_truthy(), r#""" is truthy in Lua"#);
        assert!(Value::Table(new_table()).is_truthy());
    }

    #[test]
    fn one_and_one_point_zero_are_the_same_table_key() {
        let t = new_table();
        t.borrow_mut().raw_set(Value::Number(1.0), Value::from("a")).unwrap();
        // Lua 5.1 has one number type, so these cannot be distinct keys.
        assert!(t.borrow().raw_get(&Value::Number(1.0)).raw_eq(&Value::from("a")));
        assert_eq!(t.borrow().raw_len(), 1);
    }

    #[test]
    fn negative_zero_and_zero_are_the_same_key() {
        let t = new_table();
        t.borrow_mut().raw_set(Value::Number(-0.0), Value::from("z")).unwrap();
        assert!(t.borrow().raw_get(&Value::Number(0.0)).raw_eq(&Value::from("z")));
    }

    #[test]
    fn nil_and_nan_are_rejected_as_keys() {
        let t = new_table();
        assert!(t.borrow_mut().raw_set(Value::Nil, Value::from(1i64)).is_err());
        assert!(t.borrow_mut().raw_set(Value::Number(f64::NAN), Value::from(1i64)).is_err());
    }

    #[test]
    fn assigning_nil_removes_a_key() {
        let t = new_table();
        t.borrow_mut().set_str("x", Value::from(1i64));
        assert!(t.borrow().raw_get_str("x").raw_eq(&Value::Number(1.0)));
        t.borrow_mut().set_str("x", Value::Nil);
        assert!(matches!(t.borrow().raw_get_str("x"), Value::Nil));
    }

    #[test]
    fn out_of_order_integer_keys_still_form_a_sequence() {
        // t[3], t[1], t[2] assigned in that order must still give #t == 3.
        let t = new_table();
        for i in [3i64, 1, 2] {
            t.borrow_mut().raw_set(Value::from(i), Value::from(i * 10)).unwrap();
        }
        assert_eq!(t.borrow().raw_len(), 3);
        assert!(t.borrow().raw_get(&Value::from(3i64)).raw_eq(&Value::Number(30.0)));
    }

    #[test]
    fn tables_compare_by_identity_not_contents() {
        let a = Value::Table(new_table());
        let b = Value::Table(new_table());
        assert!(a.raw_eq(&a.clone()));
        assert!(!a.raw_eq(&b), "two distinct empty tables are not equal in Lua");
    }

    #[test]
    fn removing_a_trailing_element_shrinks_the_border() {
        let t = new_table();
        for i in 1i64..=3 {
            t.borrow_mut().raw_set(Value::from(i), Value::from(i)).unwrap();
        }
        t.borrow_mut().raw_set(Value::from(3i64), Value::Nil).unwrap();
        assert_eq!(t.borrow().raw_len(), 2);
    }
}
