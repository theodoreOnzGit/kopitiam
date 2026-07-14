//! Coroutines.
//!
//! The whole architecture of this VM exists to make these tests pass (AID-0007).
//! A tree-walking interpreter cannot pass them without either faking it — a
//! `yield` that silently does not suspend — or unwinding the Rust stack, which
//! destroys the state needed to resume.
//!
//! So these are the tests that justify the design, and the ones to run first if
//! anyone ever proposes simplifying the VM back into a tree-walker.

use std::cell::RefCell;
use std::rc::Rc;

use kopitiam_lua::Lua;

fn output(src: &str) -> Vec<String> {
    let out: Rc<RefCell<Vec<String>>> = Rc::default();
    let sink = out.clone();
    let mut lua = Lua::new();
    lua.set_output(move |line| sink.borrow_mut().push(line.to_string()));
    lua.exec(src, "=test").expect("script should run");
    out.borrow().clone()
}

#[test]
fn a_coroutine_suspends_and_resumes() {
    // The irreducible test: execution must stop *inside* the function and later
    // continue from exactly there, with its locals intact.
    assert_eq!(
        output(
            r#"
            local co = coroutine.create(function()
                print("A")
                coroutine.yield()
                print("B")
                coroutine.yield()
                print("C")
            end)
            print("start")
            coroutine.resume(co)
            print("between")
            coroutine.resume(co)
            coroutine.resume(co)
            print("end")
            "#
        ),
        ["start", "A", "between", "B", "C", "end"]
    );
}

#[test]
fn locals_survive_a_suspension() {
    assert_eq!(
        output(
            r#"
            local co = coroutine.create(function()
                local n = 0
                for i = 1, 3 do
                    n = n + i
                    coroutine.yield(n)
                end
                return "done: " .. n
            end)
            print(select(2, coroutine.resume(co)))
            print(select(2, coroutine.resume(co)))
            print(select(2, coroutine.resume(co)))
            print(select(2, coroutine.resume(co)))
            "#
        ),
        ["1", "3", "6", "done: 6"]
    );
}

#[test]
fn values_pass_in_both_directions() {
    // `yield(a)` hands `a` out to `resume`; the NEXT `resume(b)` makes `yield`
    // itself return `b`. Getting this wrong in either direction is easy.
    assert_eq!(
        output(
            r#"
            local co = coroutine.create(function(first)
                print("got:", first)
                local second = coroutine.yield("out1")
                print("got:", second)
                local third = coroutine.yield("out2")
                print("got:", third)
                return "final"
            end)
            print("yielded:", select(2, coroutine.resume(co, "in1")))
            print("yielded:", select(2, coroutine.resume(co, "in2")))
            print("yielded:", select(2, coroutine.resume(co, "in3")))
            "#
        ),
        [
            "got:\tin1",
            "yielded:\tout1",
            "got:\tin2",
            "yielded:\tout2",
            "got:\tin3",
            "yielded:\tfinal",
        ]
    );
}

#[test]
fn multiple_values_pass_in_both_directions() {
    assert_eq!(
        output(
            r#"
            local co = coroutine.create(function(a, b)
                print(a, b)
                local x, y, z = coroutine.yield(1, 2, 3)
                print(x, y, z)
                return "r1", "r2"
            end)
            print(coroutine.resume(co, "a", "b"))
            print(coroutine.resume(co, "x", "y", "z"))
            "#
        ),
        ["a\tb", "true\t1\t2\t3", "x\ty\tz", "true\tr1\tr2"]
    );
}

#[test]
fn status_tracks_the_lifecycle() {
    assert_eq!(
        output(
            r#"
            local co
            co = coroutine.create(function()
                print("inside:", coroutine.status(co))
                coroutine.yield()
            end)
            print("created:", coroutine.status(co))
            coroutine.resume(co)
            print("yielded:", coroutine.status(co))
            coroutine.resume(co)
            print("finished:", coroutine.status(co))
            "#
        ),
        [
            "created:\tsuspended",
            "inside:\trunning",
            "yielded:\tsuspended",
            "finished:\tdead",
        ]
    );
}

#[test]
fn a_coroutine_that_resumed_another_is_normal_not_running() {
    assert_eq!(
        output(
            r#"
            local outer
            local inner = coroutine.create(function()
                print("outer is:", coroutine.status(outer))
                coroutine.yield()
            end)
            outer = coroutine.create(function()
                coroutine.resume(inner)
            end)
            coroutine.resume(outer)
            "#
        ),
        ["outer is:\tnormal"]
    );
}

#[test]
fn resuming_a_dead_coroutine_reports_rather_than_throws() {
    assert_eq!(
        output(
            r#"
            local co = coroutine.create(function() return 1 end)
            print(coroutine.resume(co))
            print(coroutine.resume(co))
            "#
        ),
        ["true\t1", "false\tcannot resume dead coroutine"]
    );
}

#[test]
fn an_error_inside_a_coroutine_comes_back_as_false_plus_message() {
    // `resume` NEVER throws -- it reports. The coroutine dies.
    assert_eq!(
        output(
            r#"
            local co = coroutine.create(function() error("inside", 0) end)
            print(coroutine.resume(co))
            print(coroutine.status(co))
            print("main survived")
            "#
        ),
        ["false\tinside", "dead", "main survived"]
    );
}

#[test]
fn a_runtime_error_inside_a_coroutine_is_also_reported_not_fatal() {
    assert_eq!(
        output(
            r#"
            local co = coroutine.create(function()
                local x = nil
                return x.field
            end)
            local ok, err = coroutine.resume(co)
            print(ok, err:find("index") ~= nil)
            "#
        ),
        ["false\ttrue"]
    );
}

#[test]
fn wrap_returns_values_directly_and_propagates_errors() {
    // `wrap` is `resume` minus the boolean, plus real error propagation. The
    // difference matters: an extra leading `true` would make it useless as an
    // iterator.
    assert_eq!(
        output(
            r#"
            local gen = coroutine.wrap(function()
                coroutine.yield(1)
                coroutine.yield(2)
                return 3
            end)
            print(gen(), gen(), gen())
            "#
        ),
        ["1\t2\t3"]
    );

    // An error inside a wrapped coroutine THROWS in the caller (unlike resume).
    assert_eq!(
        output(
            r#"
            local boom = coroutine.wrap(function() error("wrapped boom", 0) end)
            print(pcall(boom))
            "#
        ),
        ["false\twrapped boom"]
    );
}

#[test]
fn a_wrapped_coroutine_is_a_generic_for_iterator() {
    // This is what coroutines are actually FOR in a plugin. It only works because
    // the generic-for calls its iterator through the ordinary call machinery, so
    // the iterator is free to yield.
    assert_eq!(
        output(
            r#"
            local function range(n)
                return coroutine.wrap(function()
                    for i = 1, n do coroutine.yield(i) end
                end)
            end
            local acc = {}
            for i in range(4) do acc[#acc + 1] = i end
            print(table.concat(acc, ","))
            "#
        ),
        ["1,2,3,4"]
    );
}

#[test]
fn a_coroutine_iterator_yielding_pairs_works_in_a_generic_for() {
    assert_eq!(
        output(
            r#"
            local function entries(t)
                return coroutine.wrap(function()
                    for i, v in ipairs(t) do
                        coroutine.yield(i, v)
                    end
                end)
            end
            for i, v in entries({ "a", "b" }) do print(i, v) end
            "#
        ),
        ["1\ta", "2\tb"]
    );
}

#[test]
fn coroutines_nest() {
    assert_eq!(
        output(
            r#"
            local inner = coroutine.create(function()
                coroutine.yield("from inner")
                return "inner done"
            end)
            local outer = coroutine.create(function()
                local _, v = coroutine.resume(inner)
                coroutine.yield("outer saw: " .. v)
                local _, w = coroutine.resume(inner)
                return "outer done, inner said: " .. w
            end)
            print(select(2, coroutine.resume(outer)))
            print(select(2, coroutine.resume(outer)))
            "#
        ),
        ["outer saw: from inner", "outer done, inner said: inner done"]
    );
}

#[test]
fn a_coroutine_can_yield_from_deep_inside_nested_calls() {
    // The yield is three Lua frames down. A tree-walker would have to unwind all
    // three Rust stack frames to get out -- and could never rebuild them.
    assert_eq!(
        output(
            r#"
            local function c() coroutine.yield("deep") end
            local function b() c() end
            local function a() b() end

            local co = coroutine.create(function()
                a()
                return "resumed all the way back down"
            end)
            print(select(2, coroutine.resume(co)))
            print(select(2, coroutine.resume(co)))
            "#
        ),
        ["deep", "resumed all the way back down"]
    );
}

#[test]
fn a_coroutine_can_yield_from_inside_a_loop_and_an_if() {
    // The program counter, the loop's control cells, and the value stack all have
    // to survive the suspension.
    assert_eq!(
        output(
            r#"
            local co = coroutine.create(function()
                for i = 1, 5 do
                    if i % 2 == 1 then
                        coroutine.yield(i)
                    end
                end
                return "done"
            end)
            local out = {}
            while true do
                local ok, v = coroutine.resume(co)
                if coroutine.status(co) == "dead" then out[#out + 1] = v break end
                out[#out + 1] = v
            end
            print(table.concat(out, ","))
            "#
        ),
        ["1,3,5,done"]
    );
}

#[test]
fn a_coroutine_can_yield_across_a_pcall() {
    // Stock Lua 5.1 REFUSES this ("attempt to yield across a C-call boundary")
    // because its pcall is a C function. Ours implements pcall as a VM-level
    // protected frame, so it works -- which is what Lua 5.2+ does.
    //
    // This is a deliberate SUPERSET: no 5.1 program can depend on the failure, so
    // nothing that works in 5.1 changes meaning. Documented in lib.rs.
    assert_eq!(
        output(
            r#"
            local co = coroutine.create(function()
                local ok, v = pcall(function()
                    coroutine.yield("yielded from inside pcall")
                    return "pcall returned"
                end)
                return ok, v
            end)
            print(select(2, coroutine.resume(co)))
            print(select(2, coroutine.resume(co)))
            "#
        ),
        ["yielded from inside pcall", "true\tpcall returned"]
    );
}

#[test]
fn yielding_from_the_main_thread_is_an_error() {
    let e = Lua::new()
        .exec("coroutine.yield(1)", "=test")
        .expect_err("yield outside a coroutine must fail");
    assert!(e.to_string().contains("outside a coroutine"), "{e}");
}

#[test]
fn yielding_across_a_native_boundary_is_refused_with_luas_own_message() {
    // `table.sort`'s comparator runs in a NESTED dispatch loop, on the Rust stack.
    // Suspending out of it would strand that Rust frame, so it is refused -- which
    // is exactly what real Lua 5.1 does, and with the same message.
    //
    // The important part is that it *errors* rather than silently not yielding. A
    // coroutine that quietly fails to suspend is far worse than one that says it
    // cannot.
    assert_eq!(
        output(
            r#"
            local co = coroutine.create(function()
                local t = { 3, 1, 2 }
                table.sort(t, function(a, b)
                    coroutine.yield()   -- illegal: we are inside a native
                    return a < b
                end)
                return "should not get here"
            end)
            local ok, err = coroutine.resume(co)
            print(ok, err:find("C%-call boundary") ~= nil)
            "#
        ),
        ["false\ttrue"]
    );
}

#[test]
fn yielding_across_a_metamethod_is_refused() {
    // Same rule, same reason: an `__index` function is invoked from inside an
    // opcode, on the Rust stack.
    assert_eq!(
        output(
            r#"
            local t = setmetatable({}, {
                __index = function() coroutine.yield() return 1 end
            })
            local co = coroutine.create(function() return t.anything end)
            local ok, err = coroutine.resume(co)
            print(ok, err:find("C%-call boundary") ~= nil)
            "#
        ),
        ["false\ttrue"]
    );
}

#[test]
fn coroutine_running_is_nil_in_the_main_thread() {
    assert_eq!(
        output(
            r#"
            print(coroutine.running() == nil)
            local co = coroutine.create(function()
                print(coroutine.running() ~= nil)
            end)
            coroutine.resume(co)
            "#
        ),
        ["true", "true"]
    );
}

#[test]
fn a_producer_consumer_pair() {
    // The canonical use, and a decent end-to-end shakedown of the whole mechanism.
    assert_eq!(
        output(
            r#"
            local producer = coroutine.create(function()
                for _, word in ipairs({ "the", "quick", "fox" }) do
                    coroutine.yield(word)
                end
            end)

            local function consume()
                local ok, v = coroutine.resume(producer)
                return ok and v or nil
            end

            local got = {}
            while true do
                local w = consume()
                if not w then break end
                got[#got + 1] = w:upper()
            end
            print(table.concat(got, " "))
            "#
        ),
        ["THE QUICK FOX"]
    );
}

#[test]
fn a_coroutine_survives_thousands_of_round_trips() {
    // Frames are pushed and parked, over and over. If anything leaked or was
    // rebuilt incorrectly, this is where it would show.
    assert_eq!(
        output(
            r#"
            local co = coroutine.wrap(function()
                local n = 0
                while true do
                    n = n + 1
                    coroutine.yield(n)
                end
            end)
            local last = 0
            for _ = 1, 5000 do last = co() end
            print(last)
            "#
        ),
        ["5000"]
    );
}
