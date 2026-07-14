//! The `table` library.

use super::{arg, check_int, check_string, check_table, opt_int};
use crate::error::Result;
use crate::value::{LuaStr, Value};
use crate::vm::Lua;

pub(crate) fn install(lua: &mut Lua) {
    let t = super::make_lib(
        lua,
        "table",
        &[("insert", insert), ("remove", remove), ("concat", concat), ("sort", sort)],
    );

    // `table.unpack` is Lua 5.2's spelling; 5.1's is the global `unpack`. We
    // provide BOTH, because the dialect we actually target is the one Neovim
    // runs — LuaJIT with `LUAJIT_ENABLE_LUA52COMPAT`, which has `table.unpack`.
    // Real configs use it, and offering only the 5.1 spelling would reject code
    // that works in the editor we are cloning.
    let unpack = lua.get_global("unpack");
    t.borrow_mut().set_str("unpack", unpack);
}

/// `table.insert(t, v)` appends; `table.insert(t, pos, v)` shifts and inserts.
///
/// The two forms are told apart by **argument count**, which is why passing a nil
/// position silently appends instead of erroring — a quirk of Lua we reproduce
/// rather than "fix", because a config written against it would otherwise behave
/// differently here.
fn insert(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let t = check_table(lua, &args, 0, "insert")?;

    match args.len() {
        0 | 1 => Err(lua.rt("wrong number of arguments to 'insert'")),
        2 => {
            let n = t.borrow().raw_len();
            t.borrow_mut()
                .raw_set(Value::Number((n + 1) as f64), arg(&args, 1))
                .map_err(|e| lua.rt(e.to_string()))?;
            Ok(Vec::new())
        }
        _ => {
            let pos = check_int(lua, &args, 1, "insert")?;
            let n = t.borrow().raw_len() as i64;
            if pos < 1 || pos > n + 1 {
                return Err(lua.rt("bad argument #2 to 'insert' (position out of bounds)"));
            }
            t.borrow_mut().insert(pos as usize, arg(&args, 2));
            Ok(Vec::new())
        }
    }
}

/// `table.remove(t)` pops the last element; `table.remove(t, pos)` removes at a
/// position and shifts the rest down. Returns what was removed.
fn remove(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let t = check_table(lua, &args, 0, "remove")?;
    let n = t.borrow().raw_len() as i64;
    let pos = opt_int(lua, &args, 1, "remove", n)?;

    if n == 0 {
        return Ok(vec![Value::Nil]);
    }
    if pos < 1 || pos > n {
        return Err(lua.rt("bad argument #2 to 'remove' (position out of bounds)"));
    }
    Ok(vec![t.borrow_mut().remove(pos as usize)])
}

/// `table.concat(t, sep, i, j)`.
fn concat(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let t = check_table(lua, &args, 0, "concat")?;
    let sep = match args.get(1) {
        None | Some(Value::Nil) => LuaStr::from(""),
        Some(_) => check_string(lua, &args, 1, "concat")?,
    };
    let n = t.borrow().raw_len() as i64;
    let i = opt_int(lua, &args, 2, "concat", 1)?;
    let j = opt_int(lua, &args, 3, "concat", n)?;

    let mut out: Vec<u8> = Vec::new();
    for k in i..=j {
        let v = t.borrow().raw_get(&Value::Number(k as f64));
        // Only strings and numbers may be concatenated. Anything else is an
        // error rather than a `table: 0x...` sneaking into the output.
        let s = v.as_lua_string().ok_or_else(|| {
            lua.rt(format!("invalid value (at index {k}) in table for 'concat'"))
        })?;
        out.extend_from_slice(s.as_bytes());
        if k < j {
            out.extend_from_slice(sep.as_bytes());
        }
    }
    Ok(vec![Value::String(LuaStr::from_bytes(&out))])
}

/// `table.sort(t, comp)`.
///
/// # Why a hand-written merge sort
///
/// The comparator is a **Lua function**, so calling it can fail (it can `error()`,
/// or hit a `__lt` that throws). Rust's `slice::sort_by` takes a comparator
/// returning `Ordering` with no way to propagate a `Result`, so using it would
/// mean swallowing the error or panicking across the VM — neither acceptable.
///
/// A merge sort threading `Result` through is a dozen lines, is **stable**, and
/// makes `O(n log n)` comparisons even on adversarial input. Real Lua uses a
/// quicksort and is not stable; being stable is a strict improvement and cannot
/// break a program that did not rely on unspecified behaviour.
fn sort(lua: &mut Lua, args: Vec<Value>) -> Result<Vec<Value>> {
    let t = check_table(lua, &args, 0, "sort")?;
    let comp = arg(&args, 1);

    let items: Vec<Value> = t.borrow().array().to_vec();
    let sorted = merge_sort(lua, items, &comp)?;
    t.borrow_mut().set_array(sorted);
    Ok(Vec::new())
}

/// `a < b`, using the comparator if there is one and Lua's own `<` otherwise.
fn less(lua: &mut Lua, a: &Value, b: &Value, comp: &Value) -> Result<bool> {
    if matches!(comp, Value::Nil) {
        // No comparator: use the language's `<`, which handles numbers, strings,
        // and `__lt`. Routing through the VM rather than reimplementing it means
        // `table.sort` and `<` can never disagree.
        return lua.less_than_public(a.clone(), b.clone());
    }
    let r = lua.call(comp, vec![a.clone(), b.clone()])?;
    Ok(r.first().is_some_and(|v| v.is_truthy()))
}

fn merge_sort(lua: &mut Lua, items: Vec<Value>, comp: &Value) -> Result<Vec<Value>> {
    if items.len() <= 1 {
        return Ok(items);
    }
    let mid = items.len() / 2;
    let mut right = items;
    let left = right.drain(..mid).collect::<Vec<_>>();

    let left = merge_sort(lua, left, comp)?;
    let right = merge_sort(lua, right, comp)?;

    let mut out = Vec::with_capacity(left.len() + right.len());
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        // `right[j] < left[i]` rather than `left[i] <= right[j]`: taking from the
        // left on ties is what makes the sort stable, and it needs only the `<`
        // the comparator gives us.
        if less(lua, &right[j], &left[i], comp)? {
            out.push(right[j].clone());
            j += 1;
        } else {
            out.push(left[i].clone());
            i += 1;
        }
    }
    out.extend_from_slice(&left[i..]);
    out.extend_from_slice(&right[j..]);
    Ok(out)
}
