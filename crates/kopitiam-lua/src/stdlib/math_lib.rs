//! The `math` library.
//!
//! # `math.random` is deterministic, on purpose
//!
//! CLAUDE.md requires deterministic behaviour, and a config that produced
//! different results on two runs of kvim would be a genuine reproducibility bug.
//! So the generator starts from a **fixed seed** and produces the same sequence
//! every run until `math.randomseed` says otherwise.
//!
//! This is not a deviation from Lua: stock Lua 5.1 calls C's `rand()`, which is
//! *also* deterministic until you call `math.randomseed(os.time())`. The only
//! difference is that we do not offer `os.time()` to seed it with, so the choice
//! to become non-deterministic has to be made explicitly.
//!
//! The generator is xorshift64*, chosen because it is ten lines, has no state to
//! get wrong, and passes the statistical bar for "pick a random colour scheme" —
//! which is the entire cryptographic threat model of an editor config. **It is
//! not suitable for anything security-relevant**, and if that ever changes this
//! comment is the thing to come back to.

use std::cell::Cell;
use std::rc::Rc;

use super::{check_number, opt_int};
use crate::error::Result;
use crate::value::Value;
use crate::vm::Lua;

/// An arbitrary non-zero constant. Any non-zero seed works; xorshift breaks only
/// on zero.
const DEFAULT_SEED: u64 = 0x2545_F491_4F6C_DD1D;

pub(crate) fn install(lua: &mut Lua) {
    let t = super::make_lib(
        lua,
        "math",
        &[
            ("abs", |l, a| one(l, a, "abs", f64::abs)),
            ("ceil", |l, a| one(l, a, "ceil", f64::ceil)),
            ("floor", |l, a| one(l, a, "floor", f64::floor)),
            ("sqrt", |l, a| one(l, a, "sqrt", f64::sqrt)),
            ("sin", |l, a| one(l, a, "sin", f64::sin)),
            ("cos", |l, a| one(l, a, "cos", f64::cos)),
            ("tan", |l, a| one(l, a, "tan", f64::tan)),
            ("asin", |l, a| one(l, a, "asin", f64::asin)),
            ("acos", |l, a| one(l, a, "acos", f64::acos)),
            ("exp", |l, a| one(l, a, "exp", f64::exp)),
            ("log", |l, a| one(l, a, "log", f64::ln)),
            ("log10", |l, a| one(l, a, "log10", f64::log10)),
            ("atan", atan),
            // Lua 5.1 spells the two-argument form `math.atan2`. It is a distinct
            // function in 5.1 (only 5.3 merged it into `atan`), so it must exist
            // under its own name or a 5.1 config calling it gets `nil`.
            ("atan2", atan),
            ("pow", pow),
            ("fmod", fmod),
            ("modf", modf),
            ("max", max),
            ("min", min),
        ],
    );

    t.borrow_mut().set_str("pi", Value::Number(std::f64::consts::PI));
    // `math.huge` is infinity, and is how Lua spells "no limit".
    t.borrow_mut().set_str("huge", Value::Number(f64::INFINITY));

    // The PRNG state, shared between `random` and `randomseed`.
    let state = Rc::new(Cell::new(DEFAULT_SEED));

    let s = state.clone();
    let random = lua.create_fn("random", move |lua, args| random(lua, args, &s));
    t.borrow_mut().set_str("random", random);

    let s = state.clone();
    let randomseed = lua.create_fn("randomseed", move |lua, args| {
        let n = check_number(lua, &args, 0, "randomseed")?;
        // A zero seed would make xorshift produce nothing but zeros forever, so
        // fold it away rather than leaving a silent trap.
        let seed = (n as i64 as u64) ^ DEFAULT_SEED;
        s.set(if seed == 0 { DEFAULT_SEED } else { seed });
        Ok(Vec::new())
    });
    t.borrow_mut().set_str("randomseed", randomseed);
}

/// The one-argument shape shared by most of the library.
fn one(lua: &mut Lua, args: Vec<Value>, name: &str, f: fn(f64) -> f64) -> Result<Vec<Value>> {
    let n = check_number(lua, &args, 0, name)?;
    Ok(vec![Value::Number(f(n))])
}

/// `math.atan(y)` and `math.atan(y, x)`. The two-argument form is C's `atan2`,
/// which Lua 5.1 also exposes separately as `math.atan2`.
fn atan(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let y = check_number(lua, &args, 0, "atan")?;
    Ok(vec![Value::Number(match args.get(1) {
        None | Some(Value::Nil) => y.atan(),
        Some(_) => y.atan2(check_number(lua, &args, 1, "atan")?),
    })])
}

fn pow(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let x = check_number(lua, &args, 0, "pow")?;
    let y = check_number(lua, &args, 1, "pow")?;
    Ok(vec![Value::Number(x.powf(y))])
}

/// `math.fmod` is **truncated**, unlike the `%` operator which is **floored**.
///
/// `math.fmod(-1, 3)` is `-1`; `-1 % 3` is `2`. They are different functions and
/// Lua deliberately offers both. Conflating them is a real and easy bug.
fn fmod(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let x = check_number(lua, &args, 0, "fmod")?;
    let y = check_number(lua, &args, 1, "fmod")?;
    Ok(vec![Value::Number(x % y)])
}

/// `math.modf(3.7)` is `3, 0.7` — the integral and fractional parts, both keeping
/// the sign.
fn modf(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let x = check_number(lua, &args, 0, "modf")?;
    let int = x.trunc();
    Ok(vec![Value::Number(int), Value::Number(x - int)])
}

fn max(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(lua.rt("bad argument #1 to 'max' (number expected, got no value)"));
    }
    let mut m = check_number(lua, &args, 0, "max")?;
    for i in 1..args.len() {
        m = m.max(check_number(lua, &args, i, "max")?);
    }
    Ok(vec![Value::Number(m)])
}

fn min(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(lua.rt("bad argument #1 to 'min' (number expected, got no value)"));
    }
    let mut m = check_number(lua, &args, 0, "min")?;
    for i in 1..args.len() {
        m = m.min(check_number(lua, &args, i, "min")?);
    }
    Ok(vec![Value::Number(m)])
}

/// xorshift64*. See the module docs for why this generator and not another.
fn next_u64(state: &Cell<u64>) -> u64 {
    let mut x = state.get();
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state.set(x);
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// `math.random()` in `[0, 1)`; `math.random(m)` in `[1, m]`;
/// `math.random(m, n)` in `[m, n]`. All inclusive, as Lua specifies.
fn random(lua: &mut Lua, args: Vec<Value>, state: &Cell<u64>) -> Result<Vec<Value>> {
    // 53 bits is exactly the mantissa of an f64, so this fills [0, 1) uniformly
    // without the rounding bias that dividing a 64-bit value would introduce.
    let unit = (next_u64(state) >> 11) as f64 / (1u64 << 53) as f64;

    match args.len() {
        0 => Ok(vec![Value::Number(unit)]),
        _ => {
            let (lo, hi) = if args.len() == 1 {
                (1, opt_int(lua, &args, 0, "random", 1)?)
            } else {
                (
                    opt_int(lua, &args, 0, "random", 1)?,
                    opt_int(lua, &args, 1, "random", 1)?,
                )
            };
            if lo > hi {
                return Err(lua.rt("bad argument #2 to 'random' (interval is empty)"));
            }
            let span = (hi - lo + 1) as f64;
            Ok(vec![Value::Number(lo as f64 + (unit * span).floor())])
        }
    }
}
