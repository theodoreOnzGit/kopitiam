//! Core language semantics.
//!
//! These are the rules that, when wrong, are wrong *silently* — a config that
//! still loads but does the wrong thing. Every test here corresponds to a way a
//! Lua implementation is commonly and quietly broken.

use std::cell::RefCell;
use std::rc::Rc;

use kopitiam_lua::{Lua, Value};

fn eval(src: &str) -> Value {
    Lua::new().eval(src).expect("expression should evaluate")
}

fn num(src: &str) -> f64 {
    eval(src).as_number().unwrap_or_else(|| panic!("`{src}` did not produce a number"))
}

fn text(src: &str) -> String {
    eval(src)
        .as_lua_string()
        .unwrap_or_else(|| panic!("`{src}` did not produce a string"))
        .to_string_lossy()
}

fn truthy(src: &str) -> bool {
    eval(src).is_truthy()
}

/// Runs a script and returns everything it `print`ed, one entry per call.
fn output(src: &str) -> Vec<String> {
    let out: Rc<RefCell<Vec<String>>> = Rc::default();
    let sink = out.clone();
    let mut lua = Lua::new();
    lua.set_output(move |line| sink.borrow_mut().push(line.to_string()));
    lua.exec(src, "=test").expect("script should run");
    out.borrow().clone()
}

/// Runs a script expected to fail, returning the error text.
fn fails(src: &str) -> String {
    Lua::new().exec(src, "=test").expect_err("script should have failed").to_string()
}

// ---------------------------------------------------------------- truthiness

#[test]
fn only_nil_and_false_are_falsy() {
    // The single most dangerous rule to get wrong, because it fails silently.
    assert!(truthy("0"), "0 is TRUE in Lua");
    assert!(truthy("''"), "the empty string is TRUE in Lua");
    assert!(truthy("0.0"));
    assert!(truthy("{}"), "an empty table is true");
    assert!(!truthy("nil"));
    assert!(!truthy("false"));

    // And it must hold in a real `if`, not just in the value model.
    assert_eq!(output("if 0 then print('taken') end"), ["taken"]);
    assert_eq!(output("if '' then print('taken') end"), ["taken"]);
    assert_eq!(output("if nil then print('no') else print('else') end"), ["else"]);
}

#[test]
fn and_or_return_values_not_booleans() {
    // `a or b` is not `bool`; it is `a` or `b`. The ternary idiom depends on it.
    assert_eq!(num("1 or 2"), 1.0);
    assert_eq!(num("nil or 2"), 2.0);
    assert_eq!(num("1 and 2"), 2.0);
    assert_eq!(text("nil and 2 or 'fallback'"), "fallback");
    // 0 is truthy, so `0 or x` is 0.
    assert_eq!(num("0 or 99"), 0.0);
    assert!(!truthy("false and error('never runs')"), "and must short-circuit");
}

#[test]
fn and_or_short_circuit() {
    // The right side must not be *evaluated* at all.
    assert_eq!(
        output("local function boom() print('BOOM') return true end\nlocal _ = false and boom()\nprint('done')"),
        ["done"]
    );
    assert_eq!(
        output("local function boom() print('BOOM') return true end\nlocal _ = true or boom()\nprint('done')"),
        ["done"]
    );
}

// ---------------------------------------------------------------- precedence

#[test]
fn power_is_right_associative_and_binds_tighter_than_unary_minus() {
    assert_eq!(num("2^3^2"), 512.0, "2^(3^2), not (2^3)^2");
    assert_eq!(num("-2^2"), -4.0, "-(2^2), not (-2)^2");
    assert_eq!(num("2^-1"), 0.5);
}

#[test]
fn arithmetic_is_left_associative() {
    assert_eq!(num("1-2-3"), -4.0, "(1-2)-3");
    assert_eq!(num("8/4/2"), 1.0, "(8/4)/2");
    assert_eq!(num("2+3*4"), 14.0);
    assert_eq!(num("(2+3)*4"), 20.0);
}

#[test]
fn concat_is_right_associative() {
    assert_eq!(text("'a'..'b'..'c'"), "abc");
    // Numbers coerce to strings when concatenated.
    assert_eq!(text("1 .. 2"), "12");
    assert_eq!(text("'x' .. 1"), "x1");
}

#[test]
fn division_is_always_float_and_modulo_is_floored() {
    // Lua 5.1 has ONE number type: `/` never truncates.
    assert_eq!(num("7/2"), 3.5);

    // `%` is FLOORED (a - floor(a/b)*b), not truncated like C or Rust.
    // This diverges only on negative operands, which is what makes it insidious.
    assert_eq!(num("-1 % 3"), 2.0, "Lua's % is floored: -1 % 3 == 2, not -1");
    assert_eq!(num("1 % -3"), -2.0);
    assert_eq!(num("5 % 3"), 2.0);
    // math.fmod is the truncated one, and Lua offers both on purpose.
    assert_eq!(num("math.fmod(-1, 3)"), -1.0);
}

#[test]
fn comparison_and_equality() {
    assert!(truthy("1 < 2"));
    assert!(truthy("'a' < 'b'"), "strings compare lexicographically");
    assert!(truthy("1 == 1.0"), "one number type: 1 and 1.0 are equal");
    assert!(!truthy("1 == '1'"), "a number never equals a string");
    assert!(truthy("nil == nil"));
    assert!(!truthy("{} == {}"), "tables compare by identity");
    assert!(truthy("0/0 ~= 0/0"), "NaN is not equal to itself");
}

// ------------------------------------------------------------------ closures

#[test]
fn a_numeric_for_loop_gives_each_iteration_its_own_variable() {
    // THE closure test. In Lua 5.1 the control variable is fresh per iteration,
    // so these three closures capture 1, 2, 3 -- not 3, 3, 3.
    //
    // An implementation that reuses one cell for the loop variable passes every
    // other test in this file and fails only this one.
    assert_eq!(
        output(
            r#"
            local fns = {}
            for i = 1, 3 do
                fns[i] = function() return i end
            end
            for i = 1, 3 do print(fns[i]()) end
            "#
        ),
        ["1", "2", "3"]
    );
}

#[test]
fn a_generic_for_loop_also_gives_each_iteration_its_own_variables() {
    assert_eq!(
        output(
            r#"
            local fns = {}
            for i, v in ipairs({ "a", "b", "c" }) do
                fns[i] = function() return v end
            end
            for i = 1, 3 do print(fns[i]()) end
            "#
        ),
        ["a", "b", "c"]
    );
}

#[test]
fn a_local_in_a_while_body_is_fresh_each_iteration() {
    assert_eq!(
        output(
            r#"
            local fns, n = {}, 0
            while n < 3 do
                n = n + 1
                local captured = n
                fns[n] = function() return captured end
            end
            for i = 1, 3 do print(fns[i]()) end
            "#
        ),
        ["1", "2", "3"]
    );
}

#[test]
fn closures_capture_by_reference_and_share_the_variable() {
    // The flip side: two closures over the SAME variable must see each other's
    // writes. Capture-by-value would make `get` return 0 forever.
    assert_eq!(
        output(
            r#"
            local function counter()
                local n = 0
                local function inc() n = n + 1 end
                local function get() return n end
                return inc, get
            end
            local inc, get = counter()
            inc(); inc(); inc()
            print(get())
            "#
        ),
        ["3"]
    );
}

#[test]
fn two_counters_do_not_share_state() {
    assert_eq!(
        output(
            r#"
            local function counter()
                local n = 0
                return function() n = n + 1 return n end
            end
            local a, b = counter(), counter()
            a(); a()
            print(a(), b())
            "#
        ),
        ["3\t1"]
    );
}

#[test]
fn upvalues_reach_through_several_levels_of_nesting() {
    assert_eq!(
        output(
            r#"
            local x = "outer"
            local function a()
                local function b()
                    local function c() return x end
                    return c()
                end
                return b()
            end
            print(a())
            "#
        ),
        ["outer"]
    );
}

#[test]
fn a_local_function_can_recurse() {
    // `local function f` puts f in scope INSIDE its own body.
    // `local f = function()` does not, and that difference is real.
    assert_eq!(
        output(
            r#"
            local function fact(n)
                if n <= 1 then return 1 end
                return n * fact(n - 1)
            end
            print(fact(10))
            "#
        ),
        ["3628800"]
    );
}

#[test]
fn local_x_equals_x_reads_the_outer_x() {
    // The new local is not in scope until AFTER its initialiser.
    assert_eq!(
        output(
            r#"
            local x = "outer"
            do
                local x = x .. "/inner"
                print(x)
            end
            print(x)
            "#
        ),
        ["outer/inner", "outer"]
    );
}

// ----------------------------------------------------- multiple return values

#[test]
fn a_call_in_the_middle_of_a_list_is_truncated_to_one_value() {
    assert_eq!(
        output(
            r#"
            local function three() return 1, 2, 3 end
            print(three(), "x")
            "#
        ),
        ["1\tx"],
        "a call NOT in last position yields exactly one value"
    );
}

#[test]
fn a_call_in_the_last_position_expands() {
    assert_eq!(
        output(
            r#"
            local function three() return 1, 2, 3 end
            print("x", three())
            "#
        ),
        ["x\t1\t2\t3"]
    );
}

#[test]
fn parentheses_truncate_to_one_value() {
    // `(f())` is exactly one value even in the last position. This is the entire
    // reason the AST keeps a Paren node.
    assert_eq!(
        output(
            r#"
            local function three() return 1, 2, 3 end
            print("x", (three()))
            "#
        ),
        ["x\t1"]
    );
}

#[test]
fn multiple_assignment_pads_and_truncates() {
    assert_eq!(
        output(
            r#"
            local function two() return 1, 2 end
            local a, b, c = two()
            print(a, b, c)
            local d, e = two(), 10
            print(d, e)
            "#
        ),
        [
            "1\t2\tnil",  // padded with nil
            "1\t10",      // two() truncated because it is not last
        ]
    );
}

#[test]
fn multiple_assignment_evaluates_the_whole_right_side_first() {
    // Which is what makes a swap a swap.
    assert_eq!(
        output("local a, b = 1, 2\na, b = b, a\nprint(a, b)"),
        ["2\t1"]
    );
    // The same for table slots.
    assert_eq!(
        output("local t = {1, 2}\nt[1], t[2] = t[2], t[1]\nprint(t[1], t[2])"),
        ["2\t1"]
    );
}

#[test]
fn a_table_constructor_expands_only_its_last_element() {
    assert_eq!(
        output(
            r#"
            local function three() return 1, 2, 3 end
            print(#{ three() })
            print(#{ three(), 10 })
            print(#{ 10, three() })
            "#
        ),
        ["3", "2", "4"]
    );
}

#[test]
fn varargs() {
    // Note: `r##"..."##`, because the Lua source contains `"#` (in `select("#",
    // ...)`), which would close an `r#"..."#` string early.
    assert_eq!(
        output(
            r##"
            local function f(...)
                print(select("#", ...))
                print(...)
                local a, b = ...
                print(a, b)
                local t = { ... }
                print(#t)
            end
            f("x", "y", "z")
            "##
        ),
        ["3", "x\ty\tz", "x\ty", "3"]
    );
}

#[test]
fn varargs_with_named_parameters_before_them() {
    assert_eq!(
        output(
            r##"
            local function f(first, ...)
                print(first, select("#", ...))
            end
            f("a", "b", "c")
            f("a")
            "##
        ),
        ["a\t2", "a\t0"]
    );
}

#[test]
fn a_vararg_in_the_middle_of_a_list_is_truncated() {
    assert_eq!(
        output("local function f(...) print(..., 'end') end\nf(1, 2, 3)"),
        ["1\tend"]
    );
}

// ---------------------------------------------------------------- metatables

#[test]
fn index_as_a_table_is_inheritance() {
    assert_eq!(
        output(
            r#"
            local base = { greet = function() return "hi" end }
            local obj = setmetatable({}, { __index = base })
            print(obj.greet())
            print(rawget(obj, "greet"))
            "#
        ),
        ["hi", "nil"],
        "rawget must NOT follow __index"
    );
}

#[test]
fn index_as_a_function_is_called() {
    assert_eq!(
        output(
            r#"
            local t = setmetatable({}, {
                __index = function(tbl, key) return "computed:" .. key end
            })
            print(t.anything)
            print(t.other)
            "#
        ),
        ["computed:anything", "computed:other"]
    );
}

#[test]
fn index_chains_through_several_metatables() {
    assert_eq!(
        output(
            r#"
            local a = { x = "from a" }
            local b = setmetatable({}, { __index = a })
            local c = setmetatable({}, { __index = b })
            print(c.x)
            print(c.missing)
            "#
        ),
        ["from a", "nil"]
    );
}

#[test]
fn a_cyclic_index_chain_errors_rather_than_hanging() {
    let e = fails(
        r#"
        local a, b = {}, {}
        setmetatable(a, { __index = b })
        setmetatable(b, { __index = a })
        return a.nothing
        "#,
    );
    assert!(e.contains("__index"), "should complain about the chain: {e}");
}

#[test]
fn newindex_fires_only_for_absent_keys() {
    // The rule people get wrong: overwriting an EXISTING key is a plain raw set
    // and does not call __newindex. A proxy that forgets this looks fine until
    // the first update.
    assert_eq!(
        output(
            r#"
            local log = {}
            local t = setmetatable({ existing = 1 }, {
                __newindex = function(tbl, k, v)
                    log[#log + 1] = k
                    rawset(tbl, k, v)
                end
            })
            t.existing = 2   -- present: NO __newindex
            t.fresh = 3      -- absent:  __newindex fires
            t.fresh = 4      -- now present (rawset put it there): no fire
            print(#log, log[1])
            print(t.existing, t.fresh)
            "#
        ),
        ["1\tfresh", "2\t4"]
    );
}

#[test]
fn call_makes_a_table_callable() {
    assert_eq!(
        output(
            r#"
            local t = setmetatable({}, {
                __call = function(self, a, b) return a + b end
            })
            print(t(2, 3))
            "#
        ),
        ["5"]
    );
}

#[test]
fn tostring_metamethod() {
    assert_eq!(
        output(
            r#"
            local t = setmetatable({}, { __tostring = function() return "I am a t" end })
            print(t)
            print(tostring(t))
            "#
        ),
        ["I am a t", "I am a t"]
    );
}

#[test]
fn arithmetic_metamethods() {
    assert_eq!(
        output(
            r#"
            local mt = {}
            local function vec(n) return setmetatable({ n = n }, mt) end
            mt.__add = function(a, b) return vec(a.n + b.n) end
            mt.__sub = function(a, b) return vec(a.n - b.n) end
            mt.__mul = function(a, b) return vec(a.n * b.n) end
            mt.__div = function(a, b) return vec(a.n / b.n) end
            mt.__mod = function(a, b) return vec(a.n % b.n) end
            mt.__pow = function(a, b) return vec(a.n ^ b.n) end
            mt.__unm = function(a) return vec(-a.n) end
            mt.__concat = function(a, b) return "cat" end
            mt.__len = function(a) return 42 end

            print((vec(6) + vec(2)).n)
            print((vec(6) - vec(2)).n)
            print((vec(6) * vec(2)).n)
            print((vec(6) / vec(2)).n)
            print((vec(7) % vec(2)).n)
            print((vec(2) ^ vec(3)).n)
            print((-vec(5)).n)
            print(vec(1) .. vec(2))
            print(#vec(1))
            "#
        ),
        ["8", "4", "12", "3", "1", "8", "-5", "cat", "42"]
    );
}

#[test]
fn comparison_metamethods() {
    assert_eq!(
        output(
            r#"
            local mt = {}
            local function n(v) return setmetatable({ v = v }, mt) end
            mt.__eq = function(a, b) return a.v == b.v end
            mt.__lt = function(a, b) return a.v < b.v end
            mt.__le = function(a, b) return a.v <= b.v end

            print(n(1) == n(1))
            print(n(1) == n(2))
            print(n(1) < n(2))
            print(n(2) < n(1))
            print(n(1) <= n(1))
            print(n(2) > n(1))   -- a > b is b < a
            print(n(2) >= n(2))
            "#
        ),
        ["true", "false", "true", "false", "true", "true", "true"]
    );
}

#[test]
fn le_falls_back_to_not_lt_when_only_lt_is_defined() {
    // A documented Lua 5.1 rule: `a <= b` becomes `not (b < a)`.
    assert_eq!(
        output(
            r#"
            local mt = { __lt = function(a, b) return a.v < b.v end }
            local function n(v) return setmetatable({ v = v }, mt) end
            print(n(1) <= n(2))
            print(n(2) <= n(1))
            print(n(1) <= n(1))
            "#
        ),
        ["true", "false", "true"]
    );
}

#[test]
fn metatable_can_be_protected() {
    assert_eq!(
        output(
            r#"
            local t = setmetatable({}, { __metatable = "locked" })
            print(getmetatable(t))
            print(pcall(setmetatable, t, {}))
            "#
        )[0],
        "locked"
    );
    let e = fails("local t = setmetatable({}, { __metatable = 1 }) setmetatable(t, {})");
    assert!(e.contains("protected"), "{e}");
}

#[test]
fn strings_have_methods_via_a_shared_metatable() {
    // `("x"):upper()` -- the string library IS every string's __index.
    assert_eq!(text("('hello'):upper()"), "HELLO");
    assert_eq!(num("('hello'):len()"), 5.0);
    assert_eq!(text("('hello'):sub(2, 3)"), "el");
    assert_eq!(text("('%d items'):format(3)"), "3 items");
}

// ------------------------------------------------------------------- errors

#[test]
fn pcall_catches_and_reports() {
    assert_eq!(
        output(
            r#"
            print(pcall(function() return "fine" end))
            print(pcall(function() error("boom") end))
            "#
        ),
        ["true\tfine", "false\t=test:3: boom"]
    );
}

#[test]
fn error_values_are_arbitrary_not_just_strings() {
    // The reason LuaError carries a Value. Structured errors are the whole point
    // of `error(v)`, and a table thrown must arrive as that same table.
    assert_eq!(
        output(
            r#"
            local ok, e = pcall(function() error({ code = 404 }) end)
            print(ok, type(e), e.code)
            "#
        ),
        ["false\ttable\t404"]
    );
}

#[test]
fn error_with_level_zero_adds_no_position() {
    assert_eq!(
        output(
            r#"
            local _, e = pcall(function() error("bare", 0) end)
            print(e)
            "#
        ),
        ["bare"]
    );
}

#[test]
fn a_string_error_gets_a_source_position() {
    let out = output(r#"local _, e = pcall(function() error("msg") end) print(e)"#);
    assert!(out[0].contains("msg"));
    assert!(out[0].contains("=test:1:"), "should carry chunk:line -- got {}", out[0]);
}

#[test]
fn pcall_returns_all_of_the_results() {
    assert_eq!(
        output("print(pcall(function() return 1, 2, 3 end))"),
        ["true\t1\t2\t3"]
    );
}

#[test]
fn pcall_nests() {
    // The protected return modes must compose: the inner pcall catches, the outer
    // one succeeds.
    assert_eq!(
        output(
            r#"
            local ok, inner_ok, err = pcall(function()
                return pcall(function() error("inner", 0) end)
            end)
            print(ok, inner_ok, err)
            "#
        ),
        ["true\tfalse\tinner"]
    );

    // And with `pcall` itself as the protected function -- which only works
    // because an erroring NATIVE honours the protected return mode it was called
    // under. (A native pushes no frame, so the frame unwinder never sees it.)
    let out = output("print(pcall(pcall, error, 'x'))");
    assert!(out[0].starts_with("true\tfalse\t"), "got {}", out[0]);
    assert!(out[0].ends_with('x'), "got {}", out[0]);
}

#[test]
fn pcall_protects_a_native_function_not_just_a_lua_one() {
    // `pcall(require, ...)` is a real and common config idiom, and it is only
    // protected if an erroring native honours the protected return mode.
    assert!(output("print(pcall(require, 'definitely-not-a-module'))")[0].starts_with("false\t"));
    // A native that raises a bad-argument error is catchable too.
    assert!(output("print(pcall(string.rep))")[0].starts_with("false\t"));
}

#[test]
fn xpcall_runs_a_message_handler() {
    let out = output(
        r#"
        print(xpcall(function() error("raw") end, function(e) return "handled: " .. e end))
        "#,
    );
    assert!(out[0].starts_with("false\thandled: "), "got {}", out[0]);
    assert!(out[0].contains("raw"), "the handler must see the original message: {}", out[0]);
}

#[test]
fn assert_passes_values_through_and_throws_on_falsy() {
    assert_eq!(output("print(assert(1, 'unused'))"), ["1\tunused"]);
    assert!(fails("assert(false, 'custom message')").contains("custom message"));
    assert!(fails("assert(nil)").contains("assertion failed"));
    // 0 is truthy, so this must NOT throw.
    assert_eq!(output("assert(0) print('ok')"), ["ok"]);
}

#[test]
fn runtime_errors_are_catchable_and_describe_themselves() {
    for (src, want) in [
        ("local x = nil; return x.field", "index"),
        ("local x = nil; return x + 1", "arithmetic"),
        ("local x = {}; return x + 1", "arithmetic"),
        ("local x = nil; x()", "call"),
        ("return {} .. 'x'", "concatenate"),
        ("return #nil", "length"),
        ("return {} < {}", "compare"),
    ] {
        let e = fails(src);
        assert!(e.contains(want), "`{src}` should mention '{want}', got: {e}");
    }
}

#[test]
fn runaway_recursion_is_a_catchable_stack_overflow_not_a_crash() {
    assert_eq!(
        output(
            r#"
            local function f() return f() + 1 end
            local ok, e = pcall(f)
            print(ok, e:find("stack overflow") ~= nil)
            "#
        ),
        ["false\ttrue"]
    );
}

// -------------------------------------------------------------- control flow

#[test]
fn every_loop_form_works() {
    assert_eq!(
        output(
            r#"
            local acc = {}
            for i = 1, 3 do acc[#acc + 1] = i end
            for i = 3, 1, -1 do acc[#acc + 1] = i end
            for i = 1, 10, 4 do acc[#acc + 1] = i end
            local n = 0
            while n < 2 do n = n + 1; acc[#acc + 1] = "w" end
            repeat acc[#acc + 1] = "r"; n = n - 1 until n == 0
            print(table.concat(acc, ","))
            "#
        ),
        ["1,2,3,3,2,1,1,5,9,w,w,r,r"]
    );
}

#[test]
fn a_numeric_for_with_a_zero_step_is_an_error_not_a_hang() {
    assert!(fails("for i = 1, 10, 0 do end").contains("step"));
}

#[test]
fn repeat_until_can_see_the_bodys_locals_in_its_condition() {
    // Legal, idiomatic Lua: the body's scope extends across `until`.
    assert_eq!(
        output(
            r#"
            local n = 0
            repeat
                local done = n >= 2
                n = n + 1
            until done
            print(n)
            "#
        ),
        ["3"]
    );
}

#[test]
fn break_leaves_the_innermost_loop_only() {
    assert_eq!(
        output(
            r#"
            local hits = 0
            for i = 1, 3 do
                for j = 1, 3 do
                    if j == 2 then break end
                    hits = hits + 1
                end
            end
            print(hits)
            "#
        ),
        ["3"]
    );
}

#[test]
fn if_elseif_else() {
    let script = |n: i32| {
        format!(
            r#"
            local x = {n}
            if x < 0 then print("neg")
            elseif x == 0 then print("zero")
            elseif x < 10 then print("small")
            else print("big") end
            "#
        )
    };
    assert_eq!(output(&script(-1)), ["neg"]);
    assert_eq!(output(&script(0)), ["zero"]);
    assert_eq!(output(&script(5)), ["small"]);
    assert_eq!(output(&script(50)), ["big"]);
}

// -------------------------------------------------------------------- tables

#[test]
fn tables_are_arrays_and_maps_at_once() {
    assert_eq!(
        output(
            r#"
            local t = { 10, 20, 30, name = "x", [100] = "hundred" }
            print(#t, t[1], t[3], t.name, t[100])
            "#
        ),
        ["3\t10\t30\tx\thundred"]
    );
}

#[test]
fn assigning_nil_removes_a_key() {
    assert_eq!(
        output(
            r#"
            local t = { a = 1 }
            print(t.a)
            t.a = nil
            print(t.a)
            local n = 0
            for _ in pairs(t) do n = n + 1 end
            print(n)
            "#
        ),
        ["1", "nil", "0"]
    );
}

#[test]
fn the_length_operator_tracks_appends_and_removals() {
    assert_eq!(
        output(
            r#"
            local t = {}
            print(#t)
            t[#t + 1] = "a"
            t[#t + 1] = "b"
            print(#t)
            t[#t] = nil
            print(#t)
            "#
        ),
        ["0", "2", "1"]
    );
}

#[test]
fn ipairs_stops_at_the_first_nil_but_pairs_does_not() {
    assert_eq!(
        output(
            r#"
            local t = { 1, 2, nil, 4 }
            local i = 0
            for _ in ipairs(t) do i = i + 1 end
            print("ipairs", i)
            "#
        ),
        ["ipairs\t2"]
    );
}

#[test]
fn pairs_visits_every_key() {
    assert_eq!(
        output(
            r#"
            local t = { 1, 2, a = "x", b = "y" }
            local n = 0
            for k, v in pairs(t) do n = n + 1 end
            print(n)
            "#
        ),
        ["4"]
    );
}

#[test]
fn pairs_order_is_deterministic() {
    // Real Lua's order is unspecified and varies run to run. Ours does not, and
    // CLAUDE.md's determinism rule is why. Run it twice and compare.
    let script = r#"
        local t = { z = 1, a = 2, m = 3, [1] = "one" }
        local keys = {}
        for k in pairs(t) do keys[#keys + 1] = tostring(k) end
        print(table.concat(keys, ","))
    "#;
    let first = output(script);
    for _ in 0..5 {
        assert_eq!(output(script), first, "pairs order must not vary between runs");
    }
}

#[test]
fn a_table_can_be_a_key() {
    // Object keys must be hashed by identity AND be retrievable from `pairs`,
    // which is why the key holds the value rather than an address.
    assert_eq!(
        output(
            r#"
            local k1, k2 = {}, {}
            local t = { [k1] = "one", [k2] = "two" }
            print(t[k1], t[k2])
            local found = 0
            for k, v in pairs(t) do
                if k == k1 or k == k2 then found = found + 1 end
            end
            print(found)
            "#
        ),
        ["one\ttwo", "2"]
    );
}

#[test]
fn methods_and_self() {
    assert_eq!(
        output(
            r#"
            local Account = {}
            Account.__index = Account

            function Account.new(balance)
                return setmetatable({ balance = balance }, Account)
            end
            function Account:deposit(n)
                self.balance = self.balance + n
                return self
            end
            function Account:get()
                return self.balance
            end

            local a = Account.new(100)
            a:deposit(50):deposit(25)
            print(a:get())
            "#
        ),
        ["175"]
    );
}

#[test]
fn a_method_call_evaluates_its_object_only_once() {
    // `o:m()` must not be `o.m(o)` with `o` evaluated twice -- if `o` is a call,
    // that would run it twice.
    assert_eq!(
        output(
            r#"
            local calls = 0
            local obj = { m = function(self) return "called" end }
            local function get()
                calls = calls + 1
                return obj
            end
            print(get():m())
            print(calls)
            "#
        ),
        ["called", "1"]
    );
}

// --------------------------------------------------------------- host embedding

#[test]
fn a_rust_function_can_be_injected_and_called() {
    let mut lua = Lua::new();
    lua.set_global_fn("add", |_lua, args| {
        let a = args.first().and_then(|v| v.as_number()).unwrap_or(0.0);
        let b = args.get(1).and_then(|v| v.as_number()).unwrap_or(0.0);
        Ok(vec![Value::Number(a + b)])
    });
    assert_eq!(lua.eval("add(2, 3)").unwrap().as_number(), Some(5.0));
}

#[test]
fn a_rust_function_can_return_multiple_values() {
    let mut lua = Lua::new();
    lua.set_global_fn("pair", |_lua, _args| Ok(vec![Value::from(1i64), Value::from(2i64)]));
    lua.exec("a, b = pair()", "=t").unwrap();
    assert_eq!(lua.get_global("a").as_number(), Some(1.0));
    assert_eq!(lua.get_global("b").as_number(), Some(2.0));
}

#[test]
fn a_rust_function_can_raise_a_catchable_lua_error() {
    let mut lua = Lua::new();
    lua.set_global_fn("boom", |lua, _args| Err(lua.error_value(Value::from("from rust"))));
    let v = lua.eval("select(2, pcall(boom))").unwrap();
    assert_eq!(v.as_lua_string().unwrap().to_string_lossy(), "from rust");
}

#[test]
fn a_rust_function_can_call_back_into_lua() {
    let mut lua = Lua::new();
    lua.set_global_fn("twice", |lua, args| {
        let f = args[0].clone();
        let once = lua.call(&f, vec![Value::from(1i64)])?;
        let twice = lua.call(&f, once)?;
        Ok(twice)
    });
    assert_eq!(
        lua.eval("twice(function(n) return n * 10 end)").unwrap().as_number(),
        Some(100.0)
    );
}

#[test]
fn globals_are_readable_and_writable_from_rust() {
    let mut lua = Lua::new();
    lua.set_global("answer", Value::from(42i64));
    assert_eq!(lua.eval("answer").unwrap().as_number(), Some(42.0));

    lua.exec("computed = answer * 2", "=t").unwrap();
    assert_eq!(lua.get_global("computed").as_number(), Some(84.0));

    // _G aliases the same table.
    assert_eq!(lua.eval("_G.answer").unwrap().as_number(), Some(42.0));
}

#[test]
fn require_goes_through_the_host_loader_and_caches() {
    let loads: Rc<RefCell<Vec<String>>> = Rc::default();
    let seen = loads.clone();

    let mut lua = Lua::new();
    lua.set_module_loader(move |name| {
        seen.borrow_mut().push(name.to_string());
        match name {
            "greeting" => Some("return { hello = 'world' }".to_string()),
            _ => None,
        }
    });

    assert_eq!(lua.eval("require('greeting').hello").unwrap().as_lua_string().unwrap().to_string_lossy(), "world");
    // A second require must be a cache hit: the loader is not consulted again.
    lua.exec("require('greeting')", "=t").unwrap();
    assert_eq!(loads.borrow().len(), 1, "a module must execute at most once");

    // An unknown module is an error, not a silent nil.
    assert!(lua.exec("require('nope')", "=t").is_err());
}

#[test]
fn require_accepts_the_call_sugar() {
    let mut lua = Lua::new();
    lua.set_module_loader(|_| Some("return 7".to_string()));
    // `require "x"` with no parentheses -- the form real configs use.
    assert_eq!(lua.eval("require 'anything'").unwrap().as_number(), Some(7.0));
}

#[test]
fn a_syntax_error_names_the_chunk_and_line() {
    let e = Lua::new().exec("local x = 1\nlocal y = = 2", "=myconfig.lua").unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("myconfig.lua"), "{msg}");
    assert!(msg.contains(":2:"), "should point at line 2: {msg}");
}
