//! The `coroutine` library.
//!
//! Almost nothing happens here. `resume`, `yield` and `wrap` do not *do* anything
//! themselves — they hand the VM an [`Outcome`] describing what they want, and
//! the VM performs it on the thread stack. That indirection is the entire reason
//! coroutines work at all: a native function that tried to suspend by returning
//! through the Rust stack would have destroyed the state it needed to resume.
//! See AID-0007 and [`crate::vm`].
//!
//! # `resume` vs `wrap`
//!
//! They are the same operation with different error handling, and the difference
//! matters:
//!
//! * `coroutine.resume(co, ...)` returns `true, results...` or `false, err`. It
//!   never throws. You must check the boolean.
//! * `coroutine.wrap(f)` returns a plain function. Calling it gives the results
//!   directly — and an error inside the coroutine **propagates into the caller**
//!   rather than being returned. That is what makes a wrapped coroutine usable
//!   as a `for` iterator, where an extra leading `true` would be nonsense.

use std::cell::RefCell;
use std::rc::Rc;

use super::{arg, check_function, vm_fn};
use crate::value::{Coroutine, Outcome, Value};
use crate::vm::Lua;

pub(crate) fn install(lua: &mut Lua) {
    let t = lua.create_table();

    let create = lua.create_fn("create", |lua, args| {
        let f = check_function(lua, &args, 0, "create")?;
        Ok(vec![Value::Coroutine(Rc::new(RefCell::new(Coroutine::new(f))))])
    });
    t.borrow_mut().set_str("create", create);

    // The VM does the work; this just says what is wanted.
    t.borrow_mut().set_str(
        "resume",
        vm_fn("resume", |lua, mut args| {
            if args.is_empty() {
                return Err(lua.rt("bad argument #1 to 'resume' (coroutine expected)"));
            }
            let Value::Coroutine(co) = args.remove(0) else {
                return Err(lua.rt("bad argument #1 to 'resume' (coroutine expected)"));
            };
            Ok(Outcome::Resume { co, args, wrapped: false })
        }),
    );

    t.borrow_mut()
        .set_str("yield", vm_fn("yield", |_lua, args| Ok(Outcome::Yield(args))));

    // `wrap` builds a *closure over the coroutine*. Each call resumes it.
    let wrap = lua.create_fn("wrap", |lua, args| {
        let f = check_function(lua, &args, 0, "wrap")?;
        let co = Rc::new(RefCell::new(Coroutine::new(f)));
        Ok(vec![vm_fn("wrapped coroutine", move |_lua, args| {
            Ok(Outcome::Resume { co: co.clone(), args, wrapped: true })
        })])
    });
    t.borrow_mut().set_str("wrap", wrap);

    let status = lua.create_fn("status", |lua, args| {
        let Value::Coroutine(co) = arg(&args, 0) else {
            return Err(lua.rt("bad argument #1 to 'status' (coroutine expected)"));
        };
        // A coroutine asking about ITSELF is "running"; one that has resumed
        // another and is waiting is "normal". The VM tracks both.
        Ok(vec![Value::from(co.borrow().status.as_str())])
    });
    t.borrow_mut().set_str("status", status);

    let running = lua.create_fn("running", |lua, _args| {
        // In the main thread, `coroutine.running()` is nil. That is how a library
        // detects whether it is inside a coroutine at all.
        Ok(vec![match lua.current_coroutine() {
            Some(co) => Value::Coroutine(co),
            None => Value::Nil,
        }])
    });
    t.borrow_mut().set_str("running", running);

    let isyieldable = lua.create_fn("isyieldable", |lua, _args| {
        Ok(vec![Value::Boolean(lua.current_coroutine().is_some())])
    });
    t.borrow_mut().set_str("isyieldable", isyieldable);

    lua.set_global("coroutine", Value::Table(t));
}
