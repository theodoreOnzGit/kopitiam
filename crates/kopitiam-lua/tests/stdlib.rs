//! The standard library, exercised through Lua rather than through Rust.
//!
//! The pattern section is the largest, because Lua patterns are where a
//! reimplementation is most likely to be *subtly* wrong — matching almost the
//! right thing, almost always.

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

/// Evaluates one expression and renders it via `tostring`.
fn show(expr: &str) -> String {
    output(&format!("print({expr})")).remove(0)
}

// -------------------------------------------------------------------- base

#[test]
fn type_names_are_luas_names() {
    assert_eq!(show("type(nil)"), "nil");
    assert_eq!(show("type(true)"), "boolean");
    assert_eq!(show("type(1)"), "number");
    assert_eq!(show("type('s')"), "string");
    assert_eq!(show("type({})"), "table");
    assert_eq!(show("type(print)"), "function");
    assert_eq!(show("type(function() end)"), "function");
    assert_eq!(
        show("type(coroutine.create(function() end))"),
        "thread",
        "a coroutine's type is 'thread'"
    );
}

#[test]
fn tostring_uses_luas_number_format() {
    // `%.14g` -- the reason numbers do not print as "1.0".
    assert_eq!(show("tostring(1)"), "1");
    assert_eq!(show("tostring(3.0)"), "3");
    assert_eq!(show("tostring(1.5)"), "1.5");
    assert_eq!(show("tostring(1/3)"), "0.33333333333333");
    assert_eq!(show("tostring(1e20)"), "1e+20");
    assert_eq!(show("tostring(nil)"), "nil");
    assert_eq!(show("tostring(true)"), "true");
    // And concatenation uses the same conversion.
    assert_eq!(show("'' .. 75"), "75");
}

#[test]
fn tonumber() {
    assert_eq!(show("tonumber('42')"), "42");
    assert_eq!(show("tonumber('3.14')"), "3.14");
    assert_eq!(show("tonumber('0xff')"), "255");
    assert_eq!(show("tonumber('  7  ')"), "7");
    assert_eq!(show("tonumber('abc')"), "nil");
    assert_eq!(show("tonumber('10abc')"), "nil", "trailing garbage must fail");
    assert_eq!(show("tonumber(true)"), "nil", "a boolean is not a number");
    assert_eq!(show("tonumber(42)"), "42");
    // With an explicit base.
    assert_eq!(show("tonumber('ff', 16)"), "255");
    assert_eq!(show("tonumber('1010', 2)"), "10");
    assert_eq!(show("tonumber('z', 36)"), "35");
    assert_eq!(show("tonumber('2', 2)"), "nil", "'2' is not a binary digit");
}

#[test]
fn select() {
    assert_eq!(show("select('#')"), "0");
    assert_eq!(show("select('#', 'a', 'b', 'c')"), "3");
    assert_eq!(show("select(2, 'a', 'b', 'c')"), "b\tc");
    assert_eq!(show("select(-1, 'a', 'b', 'c')"), "c");
    // `select('#', ...)` counts nils too -- that is its entire reason to exist.
    assert_eq!(show("select('#', nil, nil)"), "2");
}

#[test]
fn unpack() {
    assert_eq!(show("unpack({ 1, 2, 3 })"), "1\t2\t3");
    assert_eq!(show("unpack({ 1, 2, 3 }, 2)"), "2\t3");
    assert_eq!(show("unpack({ 1, 2, 3 }, 2, 3)"), "2\t3");
    assert_eq!(show("unpack({})"), "");
}

#[test]
fn next_walks_a_table() {
    assert_eq!(show("next({})"), "nil", "an empty table has no next");
    assert_eq!(show("next({ 'a' })"), "1\ta");
}

// ------------------------------------------------------------------ string

#[test]
fn sub_handles_negative_and_out_of_range_indices() {
    assert_eq!(show("('hello'):sub(2, 4)"), "ell");
    assert_eq!(show("('hello'):sub(2)"), "ello");
    assert_eq!(show("('hello'):sub(-3)"), "llo", "negative counts from the end");
    assert_eq!(show("('hello'):sub(-3, -2)"), "ll");
    assert_eq!(show("('hello'):sub(1, -1)"), "hello", "the whole string");
    assert_eq!(show("('hello'):sub(0)"), "hello", "0 clamps to 1");
    assert_eq!(show("('hello'):sub(10)"), "", "past the end is empty, not an error");
    assert_eq!(show("('hello'):sub(4, 2)"), "", "reversed range is empty");
    assert_eq!(show("('hello'):sub(-100, 100)"), "hello", "clamped both ends");
}

#[test]
fn len_rep_upper_lower_reverse() {
    assert_eq!(show("('hello'):len()"), "5");
    assert_eq!(show("#'hello'"), "5");
    assert_eq!(show("('ab'):rep(3)"), "ababab");
    assert_eq!(show("('ab'):rep(0)"), "");
    assert_eq!(show("('aBc'):upper()"), "ABC");
    assert_eq!(show("('aBc'):lower()"), "abc");
    assert_eq!(show("('abc'):reverse()"), "cba");
}

#[test]
fn byte_and_char() {
    assert_eq!(show("('A'):byte()"), "65");
    assert_eq!(show("('ABC'):byte(1, 3)"), "65\t66\t67");
    assert_eq!(show("('ABC'):byte(-1)"), "67");
    assert_eq!(show("string.char(65, 66, 67)"), "ABC");
    assert_eq!(show("string.char()"), "");
    // Round trip through a non-UTF-8 byte -- Lua strings are byte strings.
    assert_eq!(show("string.char(200):byte()"), "200");
}

#[test]
fn format() {
    assert_eq!(show("('%d'):format(42)"), "42");
    assert_eq!(show("('%d'):format(3.7)"), "3", "%d truncates toward zero");
    assert_eq!(show("('%5d'):format(42)"), "   42");
    assert_eq!(show("('%-5d|'):format(42)"), "42   |");
    assert_eq!(show("('%05d'):format(42)"), "00042");
    assert_eq!(show("('%05d'):format(-42)"), "-0042", "the sign stays in front");
    assert_eq!(show("('%s and %s'):format('a', 'b')"), "a and b");
    assert_eq!(show("('%q'):format('he said \"hi\"')"), r#""he said \"hi\"""#);
    assert_eq!(show("('%x'):format(255)"), "ff");
    assert_eq!(show("('%X'):format(255)"), "FF");
    assert_eq!(show("('%08.3f'):format(3.14159)"), "0003.142");
    assert_eq!(show("('%.2f'):format(3.14159)"), "3.14");
    assert_eq!(show("('%e'):format(1234.5)"), "1.234500e+03");
    assert_eq!(show("('%g'):format(0.0001)"), "0.0001");
    assert_eq!(show("('%%'):format()"), "%");
    assert_eq!(show("('%c'):format(65)"), "A");
    // %s goes through tostring, so __tostring is honoured.
    assert_eq!(
        show("('%s'):format(setmetatable({}, { __tostring = function() return 'T' end }))"),
        "T"
    );
}

// ----------------------------------------------------------- Lua patterns

#[test]
fn find_returns_a_range_and_match_returns_text() {
    assert_eq!(show("('hello world'):find('world')"), "7\t11");
    assert_eq!(show("('hello'):find('xyz')"), "nil");
    assert_eq!(show("('hello world'):match('w%a+')"), "world");
    // find's captures come AFTER the range.
    assert_eq!(show("('key=val'):find('(%w+)=(%w+)')"), "1\t7\tkey\tval");
    // match returns the captures instead of the text.
    assert_eq!(show("('key=val'):match('(%w+)=(%w+)')"), "key\tval");
    // With no captures, match returns the whole match.
    assert_eq!(show("('key=val'):match('%w+')"), "key");
}

#[test]
fn find_with_plain_does_no_pattern_interpretation() {
    // Without `plain`, `.` is a wildcard and this finds position 1.
    assert_eq!(show("('a.b'):find('.', 1, false)"), "1\t1");
    // With `plain`, it looks for a literal dot.
    assert_eq!(show("('a.b'):find('.', 1, true)"), "2\t2");
    // The classic use: searching for text that contains magic characters.
    assert_eq!(show("('cost: $5 (approx)'):find('$5 (approx)', 1, true)"), "7\t17");
}

#[test]
fn find_honours_an_init_offset() {
    assert_eq!(show("('aXbXc'):find('X')"), "2\t2");
    assert_eq!(show("('aXbXc'):find('X', 3)"), "4\t4");
    assert_eq!(show("('aXbXc'):find('X', -2)"), "4\t4", "negative init counts back");
}

#[test]
fn character_classes_and_quantifiers() {
    assert_eq!(show("('abc123'):match('%d+')"), "123");
    assert_eq!(show("('abc123'):match('%a+')"), "abc");
    assert_eq!(show("('  padded  '):match('^%s*(.-)%s*$')"), "padded", "the trim idiom");
    assert_eq!(show("('a1!'):match('[%d%a]+')"), "a1");
    assert_eq!(show("('hello'):match('[aeiou]')"), "e");
    assert_eq!(show("('hello'):match('[^aeiou]+')"), "h");
    assert_eq!(show("('x'):match('y*')"), "", "* can match empty");
    assert_eq!(show("('x'):match('y+')"), "nil", "+ cannot");
    assert_eq!(show("('color'):match('colou?r')"), "color");
}

#[test]
fn the_lazy_dash_quantifier_is_not_a_literal_dash() {
    // THE Lua-vs-regex trap, tested through the real API.
    assert_eq!(show(r#"('"a" x "b"'):match('".-"')"#), r#""a""#, "lazy: first quote");
    assert_eq!(show(r#"('"a" x "b"'):match('".*"')"#), r#""a" x "b""#, "greedy: last");
    assert_eq!(show("('<b>hi</b>'):match('<(.-)>')"), "b");
}

#[test]
fn anchors() {
    assert_eq!(show("('hello'):match('^he')"), "he");
    assert_eq!(show("('hello'):match('^el')"), "nil");
    assert_eq!(show("('hello'):match('lo$')"), "lo");
    assert_eq!(show("('hello'):match('^hello$')"), "hello");
}

#[test]
fn magic_characters_must_be_escaped_with_percent() {
    assert_eq!(show("('a.b'):match('%.')"), ".");
    assert_eq!(show("('3+4'):match('%d%+%d')"), "3+4");
    assert_eq!(show("('50%'):match('%d+%%')"), "50%");
    assert_eq!(show("('f(x)'):match('%((%a)%)')"), "x");
}

#[test]
fn back_references_and_balanced_and_frontier() {
    assert_eq!(show("('hello world world'):match('(%w+) %1')"), "world");
    assert_eq!(show("('f(a(b)c)'):match('%b()')"), "(a(b)c)");
    assert_eq!(show("('the cat'):match('%f[%w](%w+)')"), "the");
    // A position capture is a number.
    assert_eq!(show("('hello'):match('()ll()')"), "3\t5");
}

#[test]
fn gmatch_iterates_every_match() {
    assert_eq!(
        output(
            r#"
            local words = {}
            for w in ("the quick brown fox"):gmatch("%a+") do
                words[#words + 1] = w
            end
            print(#words, table.concat(words, "|"))
            "#
        ),
        ["4\tthe|quick|brown|fox"]
    );
}

#[test]
fn gmatch_yields_multiple_captures() {
    assert_eq!(
        output(
            r#"
            local out = {}
            for k, v in ("a=1, b=2, c=3"):gmatch("(%w+)=(%w+)") do
                out[#out + 1] = k .. ":" .. v
            end
            print(table.concat(out, " "))
            "#
        ),
        ["a:1 b:2 c:3"]
    );
}

#[test]
fn gmatch_with_an_empty_match_terminates() {
    // `x*` matches empty everywhere. If the position did not advance, this would
    // hang forever.
    assert_eq!(
        output(
            r#"
            local n = 0
            for _ in ("abc"):gmatch("x*") do n = n + 1 end
            print(n)
            "#
        ),
        ["4"]
    );
}

#[test]
fn gsub_with_a_string_replacement() {
    assert_eq!(show("('hello world'):gsub('o', '0')"), "hell0 w0rld\t2");
    assert_eq!(show("('hello'):gsub('l', 'L', 1)"), "heLlo\t1", "the count limits it");
    assert_eq!(show("('a b c'):gsub(' ', '')"), "abc\t2");
    // Capture references in the replacement.
    assert_eq!(show("('key=val'):gsub('(%w+)=(%w+)', '%2=%1')"), "val=key\t1");
    // %0 is the whole match.
    assert_eq!(show("('abc'):gsub('%a', '[%0]')"), "[a][b][c]\t3");
    // %% is a literal percent.
    assert_eq!(show("('x'):gsub('x', '100%%')"), "100%\t1");
}

#[test]
fn gsub_with_a_function_replacement() {
    assert_eq!(
        output(
            r#"
            local s = ("a1b2"):gsub("%d", function(d) return "<" .. d .. ">" end)
            print(s)
            "#
        ),
        ["a<1>b<2>"]
    );
    // Returning nil or false KEEPS the original text -- the conditional-replace
    // idiom.
    assert_eq!(
        output(
            r#"
            local s = ("abc"):gsub("%a", function(c)
                if c == "b" then return "B" end
                return nil
            end)
            print(s)
            "#
        ),
        ["aBc"]
    );
}

#[test]
fn gsub_with_a_table_replacement() {
    assert_eq!(
        output(
            r#"
            local s = ("$name is $age"):gsub("%$(%w+)", { name = "Ada", age = "36" })
            print(s)
            "#
        ),
        ["Ada is 36"]
    );
    // A key the table lacks leaves the text alone.
    assert_eq!(
        output(r#"print(("$a $b"):gsub("%$(%w+)", { a = "1" }))"#),
        ["1 $b\t2"]
    );
}

#[test]
fn gsub_is_anchored_when_the_pattern_is() {
    assert_eq!(show("('aaa'):gsub('^a', 'X')"), "Xaa\t1", "^ replaces only once");
    assert_eq!(show("('aaa'):gsub('a', 'X')"), "XXX\t3");
}

#[test]
fn gsub_with_an_empty_match_terminates_and_interleaves() {
    assert_eq!(show("('abc'):gsub('x*', '-')"), "-a-b-c-\t4");
}

// ------------------------------------------------------------------- table

#[test]
fn insert_and_remove() {
    assert_eq!(
        output(
            r#"
            local t = {}
            table.insert(t, "a")
            table.insert(t, "b")
            table.insert(t, 1, "first")
            print(table.concat(t, ","), #t)
            print(table.remove(t))
            print(table.remove(t, 1))
            print(table.concat(t, ","), #t)
            "#
        ),
        ["first,a,b\t3", "b", "first", "a\t1"]
    );
}

#[test]
fn remove_from_an_empty_table_is_nil_not_an_error() {
    assert_eq!(show("table.remove({})"), "nil");
}

#[test]
fn concat() {
    assert_eq!(show("table.concat({ 'a', 'b', 'c' })"), "abc");
    assert_eq!(show("table.concat({ 'a', 'b', 'c' }, '-')"), "a-b-c");
    assert_eq!(show("table.concat({ 'a', 'b', 'c' }, '-', 2)"), "b-c");
    assert_eq!(show("table.concat({ 'a', 'b', 'c' }, '-', 2, 2)"), "b");
    assert_eq!(show("table.concat({})"), "");
    // Numbers are allowed; other types are not.
    assert_eq!(show("table.concat({ 1, 2 }, '+')"), "1+2");
    // The extra parens truncate pcall's two results to just the boolean.
    assert_eq!(show("(pcall(table.concat, { {} }))"), "false");
}

#[test]
fn sort_ascending_by_default_and_by_a_comparator() {
    assert_eq!(
        output(
            r#"
            local t = { 3, 1, 4, 1, 5, 9, 2, 6 }
            table.sort(t)
            print(table.concat(t, ","))

            table.sort(t, function(a, b) return a > b end)
            print(table.concat(t, ","))

            local words = { "banana", "apple", "cherry" }
            table.sort(words)
            print(table.concat(words, ","))
            "#
        ),
        ["1,1,2,3,4,5,6,9", "9,6,5,4,3,2,1,1", "apple,banana,cherry"]
    );
}

#[test]
fn sort_uses_the_lt_metamethod() {
    assert_eq!(
        output(
            r#"
            local mt = { __lt = function(a, b) return a.v < b.v end }
            local function n(v) return setmetatable({ v = v }, mt) end
            local t = { n(3), n(1), n(2) }
            table.sort(t)
            print(t[1].v, t[2].v, t[3].v)
            "#
        ),
        ["1\t2\t3"]
    );
}

#[test]
fn an_error_in_a_sort_comparator_propagates() {
    assert_eq!(
        output(
            r#"
            local ok = pcall(table.sort, { 2, 1 }, function() error("bad comparator") end)
            print(ok)
            "#
        ),
        ["false"]
    );
}

// -------------------------------------------------------------------- math

#[test]
fn math_basics() {
    assert_eq!(show("math.floor(3.7)"), "3");
    assert_eq!(show("math.floor(-3.2)"), "-4");
    assert_eq!(show("math.ceil(3.2)"), "4");
    assert_eq!(show("math.abs(-5)"), "5");
    assert_eq!(show("math.max(1, 9, 3)"), "9");
    assert_eq!(show("math.min(1, 9, 3)"), "1");
    assert_eq!(show("math.sqrt(16)"), "4");
    assert_eq!(show("math.huge"), "inf");
    assert_eq!(show("-math.huge"), "-inf");
    assert_eq!(show("math.pow(2, 10)"), "1024");
    assert_eq!(show("math.modf(3.75)"), "3\t0.75");
    assert_eq!(show("('%.4f'):format(math.pi)"), "3.1416");
}

#[test]
fn math_fmod_is_truncated_but_the_operator_is_floored() {
    // Two different functions, and Lua offers both on purpose.
    assert_eq!(show("math.fmod(-1, 3)"), "-1");
    assert_eq!(show("-1 % 3"), "2");
}

#[test]
fn math_random_is_in_range_and_deterministic() {
    // Deterministic by default -- see the math library's module docs. Two fresh
    // interpreters must produce the same sequence.
    let a = output("for _ = 1, 5 do io = nil end for _ = 1, 5 do print(math.random(1, 100)) end");
    let b = output("for _ = 1, 5 do io = nil end for _ = 1, 5 do print(math.random(1, 100)) end");
    assert_eq!(a, b, "math.random must be reproducible across runs");

    // And in range.
    assert_eq!(
        output(
            r#"
            local ok = true
            for _ = 1, 500 do
                local r = math.random(1, 6)
                if r < 1 or r > 6 or r ~= math.floor(r) then ok = false end
            end
            print(ok)
            "#
        ),
        ["true"]
    );

    // math.random() with no arguments is in [0, 1).
    assert_eq!(
        output(
            r#"
            local ok = true
            for _ = 1, 500 do
                local r = math.random()
                if r < 0 or r >= 1 then ok = false end
            end
            print(ok)
            "#
        ),
        ["true"]
    );
}

#[test]
fn math_randomseed_changes_the_sequence() {
    assert_eq!(
        output(
            r#"
            math.randomseed(1)
            local a = math.random(1, 1000000)
            math.randomseed(2)
            local b = math.random(1, 1000000)
            math.randomseed(1)
            local c = math.random(1, 1000000)
            print(a ~= b, a == c)
            "#
        ),
        ["true\ttrue"],
        "a seed must be reproducible, and different seeds must differ"
    );
}

// ------------------------------------------------------------------ absent

#[test]
fn the_lua_51_spellings_neovims_luajit_provides_are_all_present() {
    // Neovim runs LuaJIT with 5.2 compatibility on, so a real config may use
    // either spelling. Offering only one would reject code that works in the
    // editor we are cloning.
    assert_eq!(show("type(unpack)"), "function", "the 5.1 global");
    assert_eq!(show("type(table.unpack)"), "function", "the 5.2 spelling LuaJIT also has");
    assert_eq!(show("table.unpack({ 1, 2 })"), "1\t2");

    // math.atan2 is a distinct function in 5.1 (5.3 merged it into atan).
    assert_eq!(show("type(math.atan2)"), "function");
    assert_eq!(show("('%.4f'):format(math.atan2(1, 1))"), "0.7854");
}

#[test]
fn collectgarbage_exists_and_is_an_honest_no_op() {
    // A config that calls it defensively must not die. But it reports 0 rather
    // than inventing a plausible number -- this VM is reference-counted and has
    // no collector. See the function's docs for the consequence (cycles leak).
    assert_eq!(show("collectgarbage()"), "0");
    assert_eq!(show("collectgarbage('collect')"), "0");
    assert_eq!(show("collectgarbage('count')"), "0\t0");
}

#[test]
fn setfenv_and_getfenv_are_absent_and_that_is_a_known_gap() {
    // Documented in lib.rs. This test exists so the gap cannot be forgotten:
    // if someone implements them, this test fails and forces the docs to be
    // updated too.
    assert_eq!(show("type(setfenv)"), "nil");
    assert_eq!(show("type(getfenv)"), "nil");
}

#[test]
fn goto_is_correctly_rejected_because_it_is_lua_52() {
    let e = Lua::new().exec("do goto skip ::skip:: end", "=t");
    assert!(e.is_err(), "`goto` is not Lua 5.1 syntax and must not parse");
}

#[test]
fn the_dangerous_libraries_are_absent_by_design() {
    // An editor config has no business opening files or spawning processes. If
    // one of these ever appears, it must be a deliberate decision, not a drift --
    // so this test exists to force the conversation.
    for name in ["io", "os", "debug", "loadstring", "load", "dofile", "loadfile"] {
        assert_eq!(
            show(&format!("type({name})")),
            "nil",
            "`{name}` must not exist: see the stdlib module docs"
        );
    }
}
