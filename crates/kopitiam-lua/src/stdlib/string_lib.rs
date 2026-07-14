//! The `string` library.
//!
//! Also the reason `("x"):upper()` works: the library table is installed as the
//! `__index` of the single metatable shared by every string, so a method call on
//! a string finds it.
//!
//! # Indices are 1-based, and may be negative
//!
//! `s:sub(1, -1)` is the whole string; `s:sub(-3)` is the last three bytes. Every
//! function here routes through [`super::str_index`] so the rule is implemented
//! exactly once.

use std::cell::Cell;
use std::rc::Rc;

use super::{arg, check_int, check_number, check_string, opt_int, str_index};
use crate::error::Result;
use crate::pattern::{self, Capture};
use crate::value::{LuaStr, Value};
use crate::vm::Lua;

pub(crate) fn install(lua: &mut Lua) {
    let t = super::make_lib(
        lua,
        "string",
        &[
            ("len", len),
            ("sub", sub),
            ("upper", upper),
            ("lower", lower),
            ("rep", rep),
            ("reverse", reverse),
            ("byte", byte),
            ("char", char_),
            ("format", format),
            ("find", find),
            ("match", match_),
            ("gmatch", gmatch),
            ("gsub", gsub),
        ],
    );
    // This is what makes `("x"):upper()` and `s:format(...)` resolve.
    lua.set_string_metatable_index(Value::Table(t));
}

fn len(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let s = check_string(lua, &args, 0, "len")?;
    Ok(vec![Value::Number(s.len() as f64)])
}

/// `s:sub(i, j)` — `i` defaults to 1, `j` to -1 (the end).
fn sub(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let s = check_string(lua, &args, 0, "sub")?;
    let len = s.len();

    let mut i = str_index(check_int(lua, &args, 1, "sub")?, len);
    let mut j = str_index(opt_int(lua, &args, 2, "sub", -1)?, len);

    // Clamp into range. An out-of-range slice is an empty string, never an error.
    if i < 1 {
        i = 1;
    }
    if j > len as i64 {
        j = len as i64;
    }
    Ok(vec![if i > j {
        Value::from("")
    } else {
        Value::String(LuaStr::from_bytes(&s.as_bytes()[(i - 1) as usize..j as usize]))
    }])
}

fn upper(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let s = check_string(lua, &args, 0, "upper")?;
    Ok(vec![Value::String(LuaStr::from_bytes(&s.as_bytes().to_ascii_uppercase()))])
}

fn lower(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let s = check_string(lua, &args, 0, "lower")?;
    Ok(vec![Value::String(LuaStr::from_bytes(&s.as_bytes().to_ascii_lowercase()))])
}

fn rep(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let s = check_string(lua, &args, 0, "rep")?;
    let n = check_int(lua, &args, 1, "rep")?;
    if n <= 0 {
        return Ok(vec![Value::from("")]);
    }
    // `("x"):rep(1e9)` should fail cleanly rather than exhaust memory.
    if s.len().saturating_mul(n as usize) > 64 * 1024 * 1024 {
        return Err(lua.rt("resulting string too large"));
    }
    Ok(vec![Value::String(LuaStr::from_bytes(&s.as_bytes().repeat(n as usize)))])
}

fn reverse(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let s = check_string(lua, &args, 0, "reverse")?;
    let mut b = s.as_bytes().to_vec();
    b.reverse();
    Ok(vec![Value::String(LuaStr::from_bytes(&b))])
}

/// `s:byte(i, j)` — the byte values in a range, as multiple numbers.
fn byte(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let s = check_string(lua, &args, 0, "byte")?;
    let len = s.len();
    let i = str_index(opt_int(lua, &args, 1, "byte", 1)?, len).max(1);
    let j = str_index(opt_int(lua, &args, 2, "byte", i)?, len).min(len as i64);

    let mut out = Vec::new();
    for k in i..=j {
        if k >= 1 && (k as usize) <= len {
            out.push(Value::Number(s.as_bytes()[(k - 1) as usize] as f64));
        }
    }
    Ok(out)
}

/// `string.char(65, 66)` is `"AB"`. Values are **bytes**, so 0..255.
fn char_(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for i in 0..args.len() {
        let n = check_int(lua, &args, i, "char")?;
        if !(0..=255).contains(&n) {
            return Err(lua.rt(format!("bad argument #{} to 'char' (value out of range)", i + 1)));
        }
        out.push(n as u8);
    }
    Ok(vec![Value::String(LuaStr::from_bytes(&out))])
}

/// `string.format` — C's `printf`, minus the parts Lua does not have.
///
/// Supports `%d %i %u %c %o %x %X %e %E %f %g %G %s %q %%`, with flags
/// (`-+ #0`), width, and precision.
fn format(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let fmt = check_string(lua, &args, 0, "format")?;
    let f = fmt.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(f.len());
    let mut argi = 1;
    let mut i = 0;

    while i < f.len() {
        if f[i] != b'%' {
            out.push(f[i]);
            i += 1;
            continue;
        }
        i += 1;
        if i >= f.len() {
            return Err(lua.rt("invalid conversion '%' to 'format'"));
        }
        if f[i] == b'%' {
            out.push(b'%');
            i += 1;
            continue;
        }

        // flags
        let mut left = false;
        let mut zero = false;
        let mut plus = false;
        let mut space = false;
        let mut alt = false;
        while i < f.len() {
            match f[i] {
                b'-' => left = true,
                b'0' => zero = true,
                b'+' => plus = true,
                b' ' => space = true,
                b'#' => alt = true,
                _ => break,
            }
            i += 1;
        }
        // width
        let mut width = 0usize;
        while i < f.len() && f[i].is_ascii_digit() {
            width = width * 10 + (f[i] - b'0') as usize;
            i += 1;
        }
        // precision
        let mut precision: Option<usize> = None;
        if i < f.len() && f[i] == b'.' {
            i += 1;
            let mut p = 0usize;
            while i < f.len() && f[i].is_ascii_digit() {
                p = p * 10 + (f[i] - b'0') as usize;
                i += 1;
            }
            precision = Some(p);
        }
        if i >= f.len() {
            return Err(lua.rt("invalid conversion to 'format'"));
        }
        let conv = f[i];
        i += 1;

        let body: Vec<u8> = match conv {
            // Integers. Lua 5.1 has only doubles, so a non-integral value is
            // truncated toward zero -- which is what C's cast does, and what
            // `string.format("%d", 3.7)` therefore gives.
            b'd' | b'i' => {
                let n = check_number(lua, &args, argi, "format")? as i64;
                argi += 1;
                let mut s = n.unsigned_abs().to_string();
                if let Some(p) = precision {
                    while s.len() < p {
                        s.insert(0, '0');
                    }
                }
                let sign = if n < 0 {
                    "-"
                } else if plus {
                    "+"
                } else if space {
                    " "
                } else {
                    ""
                };
                format!("{sign}{s}").into_bytes()
            }
            b'u' => {
                let n = check_number(lua, &args, argi, "format")? as i64;
                argi += 1;
                (n.max(0) as u64).to_string().into_bytes()
            }
            b'c' => {
                let n = check_int(lua, &args, argi, "format")?;
                argi += 1;
                vec![n as u8]
            }
            b'o' => {
                let n = check_number(lua, &args, argi, "format")? as i64;
                argi += 1;
                format!("{:o}", n as u64).into_bytes()
            }
            b'x' | b'X' => {
                let n = check_number(lua, &args, argi, "format")? as i64;
                argi += 1;
                let mut s = if conv == b'x' {
                    format!("{:x}", n as u64)
                } else {
                    format!("{:X}", n as u64)
                };
                if let Some(p) = precision {
                    while s.len() < p {
                        s.insert(0, '0');
                    }
                }
                if alt && n != 0 {
                    s.insert_str(0, if conv == b'x' { "0x" } else { "0X" });
                }
                s.into_bytes()
            }
            b'f' | b'F' => {
                let n = check_number(lua, &args, argi, "format")?;
                argi += 1;
                let p = precision.unwrap_or(6);
                let mut s = format!("{n:.p$}");
                if plus && n >= 0.0 {
                    s.insert(0, '+');
                }
                s.into_bytes()
            }
            b'e' | b'E' => {
                let n = check_number(lua, &args, argi, "format")?;
                argi += 1;
                let p = precision.unwrap_or(6);
                // Rust prints `1e0`; C prints `1.000000e+00`. Rebuild it.
                let s = format!("{n:.p$e}");
                let (mant, exp) = s.split_once('e').expect("Rust emits an exponent");
                let e: i32 = exp.parse().unwrap_or(0);
                let s = format!(
                    "{mant}{}{}{:02}",
                    if conv == b'e' { 'e' } else { 'E' },
                    if e < 0 { '-' } else { '+' },
                    e.abs()
                );
                s.into_bytes()
            }
            b'g' | b'G' => {
                let n = check_number(lua, &args, argi, "format")?;
                argi += 1;
                let s = crate::number::format_g(n, precision.unwrap_or(6));
                if conv == b'G' { s.to_uppercase().into_bytes() } else { s.into_bytes() }
            }
            b's' => {
                let v = arg(&args, argi);
                argi += 1;
                // Through `tostring`, so `__tostring` is honoured.
                let mut s = lua.tostring(&v)?.as_bytes().to_vec();
                if let Some(p) = precision {
                    s.truncate(p);
                }
                s
            }
            // `%q` writes a string that Lua can read back. It is the reason
            // `%q` exists at all, so the escaping has to be exactly right.
            b'q' => {
                let v = arg(&args, argi);
                argi += 1;
                let s = check_string(lua, &[v], 0, "format")?;
                let mut q = vec![b'"'];
                for &c in s.as_bytes() {
                    match c {
                        b'"' => q.extend_from_slice(b"\\\""),
                        b'\\' => q.extend_from_slice(b"\\\\"),
                        b'\n' => q.extend_from_slice(b"\\n"),
                        b'\r' => q.extend_from_slice(b"\\r"),
                        0 => q.extend_from_slice(b"\\0"),
                        c => q.push(c),
                    }
                }
                q.push(b'"');
                q
            }
            other => {
                return Err(lua.rt(format!(
                    "invalid conversion '%{}' to 'format'",
                    other as char
                )));
            }
        };

        // Width padding. `-` left-justifies; `0` pads numbers with zeros.
        if body.len() < width {
            let pad = width - body.len();
            if left {
                out.extend_from_slice(&body);
                out.extend(std::iter::repeat_n(b' ', pad));
            } else if zero && !matches!(conv, b's' | b'q' | b'c') {
                // A zero-padded negative number keeps its sign in front:
                // `%05d` of -42 is `-0042`, not `00-42`.
                let (sign, digits) = match body.first() {
                    Some(&c @ (b'-' | b'+')) => (Some(c), &body[1..]),
                    _ => (None, &body[..]),
                };
                if let Some(c) = sign {
                    out.push(c);
                }
                out.extend(std::iter::repeat_n(b'0', pad));
                out.extend_from_slice(digits);
            } else {
                out.extend(std::iter::repeat_n(b' ', pad));
                out.extend_from_slice(&body);
            }
        } else {
            out.extend_from_slice(&body);
        }
    }

    Ok(vec![Value::String(LuaStr::from_bytes(&out))])
}

/// Turns a capture into the Lua value it should become.
fn capture_value(c: &Capture) -> Value {
    match c {
        Capture::Str(s) => Value::String(LuaStr::from_bytes(s)),
        // A position capture is a NUMBER, not a string.
        Capture::Position(p) => Value::Number(*p as f64),
    }
}

/// The captures a match produced, or the whole match when there were none.
///
/// This "capture zero" rule is shared by `match`, `gmatch` and `gsub`, and
/// getting it wrong in one of them is a classic inconsistency.
fn captures_or_whole(src: &[u8], m: &pattern::Match) -> Vec<Value> {
    if m.captures.is_empty() {
        vec![Value::String(LuaStr::from_bytes(&src[m.start..m.end]))]
    } else {
        m.captures.iter().map(capture_value).collect()
    }
}

/// `s:find(pat, init, plain)` — returns `start, end` (1-based, inclusive), then
/// any captures. `plain` does a literal substring search with no pattern magic.
fn find(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let s = check_string(lua, &args, 0, "find")?;
    let p = check_string(lua, &args, 1, "find")?;
    let src = s.as_bytes();

    let init = str_index(opt_int(lua, &args, 2, "find", 1)?, src.len());
    let init = init.max(1) as usize - 1;
    if init > src.len() {
        return Ok(vec![Value::Nil]);
    }

    if arg(&args, 3).is_truthy() {
        // Plain search: no pattern interpretation at all.
        let pat = p.as_bytes();
        if pat.is_empty() {
            return Ok(vec![
                Value::Number((init + 1) as f64),
                Value::Number(init as f64),
            ]);
        }
        let found = src[init..]
            .windows(pat.len())
            .position(|w| w == pat)
            .map(|off| init + off);
        return Ok(match found {
            Some(at) => vec![
                Value::Number((at + 1) as f64),
                Value::Number((at + pat.len()) as f64),
            ],
            None => vec![Value::Nil],
        });
    }

    match pattern::find(src, p.as_bytes(), init).map_err(|e| lua.rt(e.to_string()))? {
        None => Ok(vec![Value::Nil]),
        Some(m) => {
            let mut out =
                vec![Value::Number((m.start + 1) as f64), Value::Number(m.end as f64)];
            // Unlike `match`, `find` appends only EXPLICIT captures -- there is
            // no capture-zero fallback, because the range is already returned.
            out.extend(m.captures.iter().map(capture_value));
            Ok(out)
        }
    }
}

/// `s:match(pat, init)` — the captures, or the whole match if there are none.
fn match_(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let s = check_string(lua, &args, 0, "match")?;
    let p = check_string(lua, &args, 1, "match")?;
    let src = s.as_bytes();

    let init = str_index(opt_int(lua, &args, 2, "match", 1)?, src.len());
    let init = init.max(1) as usize - 1;
    if init > src.len() {
        return Ok(vec![Value::Nil]);
    }

    match pattern::find(src, p.as_bytes(), init).map_err(|e| lua.rt(e.to_string()))? {
        None => Ok(vec![Value::Nil]),
        Some(m) => Ok(captures_or_whole(src, &m)),
    }
}

/// `s:gmatch(pat)` — an iterator over every match.
///
/// The position is held in a `Cell` captured by the returned closure. That is the
/// whole state: `gmatch` has no `state`/`control` in the generic-for protocol, so
/// it *must* be a closure over mutable state, and Lua's own implementation does
/// the same.
fn gmatch(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let s = check_string(lua, &args, 0, "gmatch")?;
    let p = check_string(lua, &args, 1, "gmatch")?;
    let pos = Rc::new(Cell::new(0usize));

    let iter = lua.create_fn("gmatch_iterator", move |lua, _args| {
        let src = s.as_bytes();
        let at = pos.get();
        if at > src.len() {
            return Ok(vec![Value::Nil]);
        }
        match pattern::find(src, p.as_bytes(), at).map_err(|e| lua.rt(e.to_string()))? {
            None => {
                // Park past the end so a further call cannot loop.
                pos.set(src.len() + 1);
                Ok(vec![Value::Nil])
            }
            Some(m) => {
                // An EMPTY match must still advance, or `("abc"):gmatch("x*")`
                // would spin forever.
                pos.set(if m.end > m.start { m.end } else { m.end + 1 });
                Ok(captures_or_whole(src, &m))
            }
        }
    });
    Ok(vec![iter])
}

/// `s:gsub(pat, repl, n)` — replace, returning the new string and the count.
///
/// `repl` may be a string (with `%0`..`%9` and `%%`), a table (looked up by the
/// first capture), or a function (called with the captures). A table or function
/// that yields `nil` or `false` leaves the original text alone — which is what
/// makes `gsub` usable as a conditional replace.
fn gsub(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let s = check_string(lua, &args, 0, "gsub")?;
    let p = check_string(lua, &args, 1, "gsub")?;
    let repl = arg(&args, 2);
    let max = opt_int(lua, &args, 3, "gsub", i64::MAX)?;

    let src = s.as_bytes();
    let pat = p.as_bytes();
    let anchored = pat.first() == Some(&b'^');

    let mut out: Vec<u8> = Vec::with_capacity(src.len());
    let mut pos = 0usize;
    let mut count: i64 = 0;

    while count < max {
        let Some(m) = pattern::find(src, pat, pos).map_err(|e| lua.rt(e.to_string()))? else {
            break;
        };
        // Everything the pattern skipped over is copied through untouched.
        out.extend_from_slice(&src[pos..m.start]);
        count += 1;

        let whole = &src[m.start..m.end];
        let caps = captures_or_whole(src, &m);

        let replacement: Value = match &repl {
            Value::Table(t) => t.borrow().raw_get(&caps[0]),
            f @ (Value::Function(_) | Value::Native(_)) => {
                lua.call(f, caps.clone())?.into_iter().next().unwrap_or(Value::Nil)
            }
            _ => {
                // A string replacement, with `%n` references.
                let rs = check_string(lua, &args, 2, "gsub")?;
                let rb = rs.as_bytes();
                let mut buf = Vec::with_capacity(rb.len());
                let mut i = 0;
                while i < rb.len() {
                    if rb[i] != b'%' {
                        buf.push(rb[i]);
                        i += 1;
                        continue;
                    }
                    i += 1;
                    match rb.get(i) {
                        None => return Err(lua.rt("invalid use of '%' in replacement string")),
                        Some(b'%') => {
                            buf.push(b'%');
                            i += 1;
                        }
                        Some(&c) if c.is_ascii_digit() => {
                            let n = (c - b'0') as usize;
                            i += 1;
                            if n == 0 {
                                // `%0` is the whole match.
                                buf.extend_from_slice(whole);
                            } else {
                                let Some(cv) = caps.get(n - 1) else {
                                    return Err(lua.rt(format!(
                                        "invalid capture index %{n} in replacement string"
                                    )));
                                };
                                let cs = cv.as_lua_string().ok_or_else(|| {
                                    lua.rt("invalid capture value in replacement string")
                                })?;
                                buf.extend_from_slice(cs.as_bytes());
                            }
                        }
                        Some(_) => {
                            return Err(lua.rt("invalid use of '%' in replacement string"));
                        }
                    }
                }
                Value::String(LuaStr::from_bytes(&buf))
            }
        };

        match replacement {
            // nil or false: keep the original text. This is what lets a lookup
            // table replace only the keys it has.
            Value::Nil | Value::Boolean(false) => out.extend_from_slice(whole),
            other => {
                let rs = other.as_lua_string().ok_or_else(|| {
                    lua.rt(format!(
                        "invalid replacement value (a {})",
                        other.type_name()
                    ))
                })?;
                out.extend_from_slice(rs.as_bytes());
            }
        }

        if m.end > m.start {
            pos = m.end;
        } else {
            // An empty match: emit the byte we are sitting on and step past it,
            // or we would match here forever.
            if m.start < src.len() {
                out.push(src[m.start]);
            }
            pos = m.start + 1;
            if pos > src.len() {
                break;
            }
        }
        if anchored {
            break;
        }
    }

    if pos <= src.len() {
        out.extend_from_slice(&src[pos..]);
    }
    Ok(vec![Value::String(LuaStr::from_bytes(&out)), Value::Number(count as f64)])
}
