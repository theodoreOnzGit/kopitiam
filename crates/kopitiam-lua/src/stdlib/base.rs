//! The base library: the functions that live directly in `_G`.

use super::{arg, check_int, check_string, check_table, opt_int, vm_fn};
use crate::error::Result;
use crate::value::{Outcome, Value};
use crate::vm::Lua;

pub(crate) fn install(lua: &mut Lua) {
    lua.set_global_fn("print", print);
    lua.set_global_fn("type", lua_type);
    lua.set_global_fn("tostring", tostring);
    lua.set_global_fn("tonumber", tonumber);
    lua.set_global_fn("ipairs", ipairs);
    lua.set_global_fn("pairs", pairs);
    lua.set_global_fn("next", next);
    lua.set_global_fn("select", select);
    lua.set_global_fn("unpack", unpack);
    lua.set_global_fn("rawget", rawget);
    lua.set_global_fn("rawset", rawset);
    lua.set_global_fn("rawequal", rawequal);
    lua.set_global_fn("rawlen", rawlen);
    lua.set_global_fn("setmetatable", setmetatable);
    lua.set_global_fn("getmetatable", getmetatable);
    lua.set_global_fn("assert", assert_);
    lua.set_global_fn("error", error);
    lua.set_global_fn("require", require);
    lua.set_global_fn("collectgarbage", collectgarbage);

    // `pcall` and `xpcall` are control flow, not functions: they must run their
    // argument under a frame the VM can unwind to. Hence the `Outcome` form.
    let pcall = vm_fn("pcall", |lua, mut args| {
        if args.is_empty() {
            return Err(lua.rt("bad argument #1 to 'pcall' (value expected)"));
        }
        let f = args.remove(0);
        Ok(Outcome::Protected { f, args, handler: None })
    });
    lua.set_global("pcall", pcall);

    let xpcall = vm_fn("xpcall", |lua, mut args| {
        if args.len() < 2 {
            return Err(lua.rt("bad argument #2 to 'xpcall' (value expected)"));
        }
        let f = args.remove(0);
        let handler = args.remove(0);
        // Lua 5.1's xpcall passes NO arguments to f -- that only arrived in 5.2.
        // Accepting them here would be a superset, but silently doing so would
        // hide a real portability bug in a config, so the extras are dropped as
        // 5.1 does.
        Ok(Outcome::Protected { f, args: Vec::new(), handler: Some(handler) })
    });
    lua.set_global("xpcall", xpcall);

    // `package.loaded` -- the module cache. Exposed because configs sometimes
    // poke at it (`package.loaded["foo"] = nil` to force a reload).
    let package = lua.create_table();
    package.borrow_mut().set_str("loaded", Value::Table(lua.loaded_table()));
    lua.set_global("package", Value::Table(package));
}

fn print(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let mut parts = Vec::with_capacity(args.len());
    for v in &args {
        // Through `tostring`, so `__tostring` is honoured -- `print(obj)` on a
        // table with a `__tostring` must show the friendly form.
        parts.push(lua.tostring(v)?.to_string_lossy());
    }
    let line = parts.join("\t");

    // Take the sink out while calling it: it is `FnMut`, and it may want to look
    // at the Lua state.
    match lua.output.take() {
        Some(mut out) => {
            out(&line);
            lua.output = Some(out);
        }
        None => println!("{line}"),
    }
    Ok(Vec::new())
}

fn lua_type(_lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    Ok(vec![Value::from(arg(&args, 0).type_name())])
}

fn tostring(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let s = lua.tostring(&arg(&args, 0))?;
    Ok(vec![Value::String(s)])
}

/// `tonumber(v)` and `tonumber(s, base)`.
fn tonumber(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let v = arg(&args, 0);

    // With a base, the argument MUST be a string and is parsed in that base.
    if args.len() >= 2 && !matches!(args[1], Value::Nil) {
        let base = check_int(lua, &args, 1, "tonumber")?;
        if !(2..=36).contains(&base) {
            return Err(lua.rt("bad argument #2 to 'tonumber' (base out of range)"));
        }
        let s = check_string(lua, &args, 0, "tonumber")?;
        let text = s.to_string_lossy();
        let text = text.trim();
        if text.is_empty() {
            return Ok(vec![Value::Nil]);
        }
        let (neg, digits) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text.strip_prefix('+').unwrap_or(text)),
        };
        let mut acc: f64 = 0.0;
        if digits.is_empty() {
            return Ok(vec![Value::Nil]);
        }
        for c in digits.chars() {
            match c.to_digit(36) {
                Some(d) if (d as i64) < base => acc = acc * base as f64 + d as f64,
                // Any bad digit makes the whole conversion fail, returning nil.
                _ => return Ok(vec![Value::Nil]),
            }
        }
        return Ok(vec![Value::Number(if neg { -acc } else { acc })]);
    }

    // Without a base: numbers pass through, strings are parsed, everything else
    // is nil. Note `tonumber(true)` is nil, not 1.
    Ok(vec![match v.as_number() {
        Some(n) => Value::Number(n),
        None => Value::Nil,
    }])
}

/// `next(t, key)` — the stateless iterator `pairs` is built on.
fn next(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let t = check_table(lua, &args, 0, "next")?;
    let k = arg(&args, 1);
    let step = t.borrow().next_key(&k).map_err(|e| lua.rt(e.to_string()))?;
    Ok(match step {
        Some((k, v)) => vec![k, v],
        // Lua signals "done" with a single nil.
        None => vec![Value::Nil],
    })
}

/// `pairs(t)` returns `next, t, nil` — the generic-for protocol.
fn pairs(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let t = check_table(lua, &args, 0, "pairs")?;
    // `__pairs` is a Lua 5.2 feature and is deliberately not honoured: a 5.1
    // config cannot rely on it, and quietly supporting it would let a config be
    // written that then fails on real LuaJIT.
    let next = lua.get_global("next");
    Ok(vec![next, Value::Table(t), Value::Nil])
}

/// `ipairs(t)` walks 1, 2, 3, ... and **stops at the first nil**.
///
/// That is not the same as walking every integer key: a table with a hole has an
/// `ipairs` that stops early, which is exactly the behaviour Lua promises and
/// exactly the thing that surprises people.
fn ipairs(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let t = check_table(lua, &args, 0, "ipairs")?;

    let iter = lua.create_fn("ipairs_iterator", |lua, args| {
        let t = check_table(lua, &args, 0, "ipairs")?;
        let i = check_int(lua, &args, 1, "ipairs")? + 1;
        let v = t.borrow().raw_get(&Value::Number(i as f64));
        Ok(if matches!(v, Value::Nil) {
            vec![Value::Nil]
        } else {
            vec![Value::Number(i as f64), v]
        })
    });
    Ok(vec![iter, Value::Table(t), Value::Number(0.0)])
}

/// `select("#", ...)` counts; `select(n, ...)` drops the first n-1.
fn select(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    match arg(&args, 0) {
        Value::String(s) if s.as_bytes() == b"#" => {
            Ok(vec![Value::Number((args.len() - 1) as f64)])
        }
        _ => {
            let n = check_int(lua, &args, 0, "select")?;
            let rest = &args[1..];
            if n < 0 {
                // A negative index counts from the end: select(-1, ...) is the
                // last argument.
                let from = rest.len() as i64 + n;
                if from < 0 {
                    return Err(lua.rt("bad argument #1 to 'select' (index out of range)"));
                }
                return Ok(rest[from as usize..].to_vec());
            }
            if n == 0 {
                return Err(lua.rt("bad argument #1 to 'select' (index out of range)"));
            }
            let from = (n as usize - 1).min(rest.len());
            Ok(rest[from..].to_vec())
        }
    }
}

/// `unpack(t, i, j)` — turns an array into multiple values.
fn unpack(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let t = check_table(lua, &args, 0, "unpack")?;
    let len = t.borrow().raw_len() as i64;
    let i = opt_int(lua, &args, 1, "unpack", 1)?;
    let j = opt_int(lua, &args, 2, "unpack", len)?;

    if i > j {
        return Ok(Vec::new());
    }
    // A guard against `unpack({}, 1, 1e9)` eating all memory.
    if j - i >= 1_000_000 {
        return Err(lua.rt("too many results to unpack"));
    }
    let mut out = Vec::with_capacity((j - i + 1) as usize);
    for k in i..=j {
        out.push(t.borrow().raw_get(&Value::Number(k as f64)));
    }
    Ok(out)
}

fn rawget(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let t = check_table(lua, &args, 0, "rawget")?;
    let v = t.borrow().raw_get(&arg(&args, 1));
    Ok(vec![v])
}

fn rawset(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let t = check_table(lua, &args, 0, "rawset")?;
    t.borrow_mut()
        .raw_set(arg(&args, 1), arg(&args, 2))
        .map_err(|e| lua.rt(e.to_string()))?;
    Ok(vec![Value::Table(t)])
}

fn rawequal(_lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    Ok(vec![Value::Boolean(arg(&args, 0).raw_eq(&arg(&args, 1)))])
}

/// `rawlen` is Lua 5.2, kept because the brief asked for it and because there is
/// otherwise no way to get a table's length past a `__len` metamethod.
fn rawlen(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    Ok(vec![match arg(&args, 0) {
        Value::Table(t) => Value::Number(t.borrow().raw_len() as f64),
        Value::String(s) => Value::Number(s.len() as f64),
        _ => return Err(lua.rt("table or string expected")),
    }])
}

fn setmetatable(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let t = check_table(lua, &args, 0, "setmetatable")?;

    // `__metatable` makes a metatable read-only: it is how a library hides its
    // internals from code that would otherwise reach in and break them.
    if let Some(existing) = t.borrow().metatable.clone()
        && !matches!(existing.borrow().raw_get_str("__metatable"), Value::Nil)
    {
        return Err(lua.rt("cannot change a protected metatable"));
    }

    match arg(&args, 1) {
        Value::Nil => t.borrow_mut().metatable = None,
        Value::Table(mt) => t.borrow_mut().metatable = Some(mt),
        _ => return Err(lua.rt("bad argument #2 to 'setmetatable' (nil or table expected)")),
    }
    Ok(vec![Value::Table(t)])
}

fn getmetatable(_lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let Value::Table(t) = arg(&args, 0) else {
        return Ok(vec![Value::Nil]);
    };
    let Some(mt) = t.borrow().metatable.clone() else {
        return Ok(vec![Value::Nil]);
    };
    // If `__metatable` is set, THAT is what the caller sees -- never the real
    // metatable.
    let protected = mt.borrow().raw_get_str("__metatable");
    Ok(vec![if matches!(protected, Value::Nil) { Value::Table(mt) } else { protected }])
}

/// `assert(v, message)` — returns its arguments if `v` is truthy, throws if not.
///
/// The message may be any value, not just a string, and is thrown as-is.
fn assert_(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    if arg(&args, 0).is_truthy() {
        return Ok(args);
    }
    match args.get(1) {
        Some(m) => Err(lua.rt_value(m.clone())),
        None => Err(lua.rt("assertion failed!")),
    }
}

/// `error(value, level)`.
///
/// Lua's errors carry an arbitrary **value**, not a message. Only when it is a
/// string (and `level` is not 0) does Lua prepend the source position.
fn error(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let v = arg(&args, 0);
    let level = opt_int(lua, &args, 1, "error", 1)?;

    let v = match (&v, level) {
        // Level 0 means "no position information" -- used when the message is
        // already fully formed.
        (_, 0) => v,
        (Value::String(s), _) => {
            Value::from(format!("{}{}", lua.where_(), s.to_string_lossy()))
        }
        // A thrown table stays a table. Prefixing it would destroy it, and
        // structured errors are the entire reason `error` takes a value.
        _ => v,
    };
    Err(lua.rt_value(v))
}

/// `collectgarbage(opt)` — present, and honestly a no-op.
///
/// # There is no garbage collector, because there is no garbage collection
///
/// This VM manages memory by **reference counting** (`Rc`), not by tracing. So
/// there is no collector to step, pause, or restart, and `collectgarbage("collect")`
/// has nothing to do. It exists so that a config which calls it — plenty do,
/// defensively — does not die with `attempt to call a nil value`.
///
/// `collectgarbage("count")` reports 0 rather than a fabricated number. Returning
/// a plausible-looking figure would be worse than useless.
///
/// **The real consequence, stated plainly: reference cycles are never freed.**
/// `local t = {} t.self = t` leaks, where real Lua would collect it. For an
/// editor config — short-lived, small, and run once at startup — that is an
/// acceptable trade for deleting an entire GC from the codebase. It would *not*
/// be acceptable for a long-running process that churns cyclic structures, and if
/// kvim ever grows one, this is the note to come back to.
fn collectgarbage(_lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let opt = arg(&args, 0);
    let opt = opt.as_lua_string().map(|s| s.to_string_lossy()).unwrap_or_default();
    Ok(match opt.as_str() {
        // Lua returns kilobytes-in-use, then the remainder in bytes.
        "count" => vec![Value::Number(0.0), Value::Number(0.0)],
        _ => vec![Value::Number(0.0)],
    })
}

/// `require(name)`.
///
/// Resolution goes through the **host's** loader ([`Lua::set_module_loader`]),
/// not a filesystem search path. That is a deliberate inversion of stock Lua: an
/// embedded interpreter should not decide on its own to read the disk. kvim
/// supplies a loader that maps `require("settings")` to
/// `~/.kopitiam/kopitiam-neovim/lua/settings.lua`, and nothing else can be
/// reached.
///
/// A module is executed at most once; the result is cached in `package.loaded`,
/// as Lua guarantees.
fn require(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let name = check_string(lua, &args, 0, "require")?.to_string_lossy();

    let loaded = lua.loaded_table();
    let cached = loaded.borrow().raw_get_str(&name);
    if !matches!(cached, Value::Nil) {
        return Ok(vec![cached]);
    }

    let Some(loader) = lua.module_loader() else {
        return Err(lua.rt(format!(
            "module '{name}' not found: no module loader is configured"
        )));
    };
    let Some(source) = loader(&name) else {
        return Err(lua.rt(format!("module '{name}' not found")));
    };

    // `@name` is Lua's convention for "this chunk came from a file called name",
    // and it is what makes error messages inside the module readable.
    let results = lua.exec(&source, &format!("@{name}.lua"))?;

    // A module that returns nothing is still loaded. Lua records `true` so that
    // a second `require` is a cache hit rather than a re-execution.
    let mut v = results.into_iter().next().unwrap_or(Value::Nil);
    if matches!(v, Value::Nil) {
        v = Value::Boolean(true);
    }
    loaded.borrow_mut().set_str(&name, v.clone());
    Ok(vec![v])
}
