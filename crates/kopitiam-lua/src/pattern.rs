//! Lua patterns.
//!
//! # These are not regular expressions
//!
//! Lua has its own small pattern language, and reaching for the `regex` crate
//! would be *subtly* wrong rather than obviously wrong — which is worse. The
//! differences that bite:
//!
//! | Lua | Regex | Consequence |
//! |---|---|---|
//! | `%` is the escape | `\` is the escape | `%.` is a literal dot; `\.` is a literal backslash then any char |
//! | `-` is a **lazy** `*` | `-` is a literal, `*?` is lazy | `(.-)` is the single most common Lua idiom and means nothing in regex |
//! | no alternation `\|` | has it | a pattern that "looks like" it should work does not |
//! | no `{n,m}` counts | has them | |
//! | `%b()` balanced match | none | |
//! | `%f[set]` frontier | none | |
//! | classes are `%a %d %s %w` | `\a` is a bell, `\d` `\s` `\w` exist | `%s` vs `\s` silently diverge on what counts as space |
//!
//! So the matcher is implemented here from the Lua 5.1 reference manual's
//! specification (§5.4.1). The recursive-backtracking *structure* follows the
//! algorithm the manual describes and that the reference implementation uses —
//! this is a clean-room implementation from the specification, not a port of
//! `lstrlib.c`, but the algorithm itself is the obvious one and no originality
//! is claimed for it.
//!
//! # Bytes, not characters
//!
//! Everything here indexes by **byte**, because Lua strings are byte strings and
//! `string.find` returns byte offsets. A pattern applied to UTF-8 text matches
//! bytes, exactly as it does in real Lua.

use crate::error::{LuaError, Result};

/// Lua's limit, and ours. Nine so that `%1`..`%9` back-references can address
/// them all.
const MAX_CAPTURES: usize = 32;

/// Guards against patterns whose backtracking would blow the Rust stack —
/// `("a"):rep(100):find("(a*)*b")` and friends. Real Lua has the same guard, for
/// the same reason.
const MAX_RECURSION: usize = 220;

/// What a capture holds.
#[derive(Debug, Clone, PartialEq)]
pub enum Capture {
    /// A `(...)` capture: the matched bytes.
    Str(Vec<u8>),
    /// A `()` capture: the **1-based** position where it appeared. Lua returns
    /// this as a number, not a string.
    Position(usize),
}

/// A successful match: byte range plus captures.
#[derive(Debug, Clone)]
pub struct Match {
    /// Byte index of the first matched byte.
    pub start: usize,
    /// Byte index one past the last matched byte.
    pub end: usize,
    /// Explicit captures. Empty when the pattern has none — callers then treat
    /// the whole match as capture zero.
    pub captures: Vec<Capture>,
}

#[derive(Clone, Copy)]
enum CapLen {
    /// A `(` we have not yet seen the matching `)` for.
    Unfinished,
    /// A `()` position capture.
    Position,
    Len(usize),
}

#[derive(Clone, Copy)]
struct CapState {
    init: usize,
    len: CapLen,
}

struct Matcher<'a> {
    src: &'a [u8],
    pat: &'a [u8],
    level: usize,
    caps: [CapState; MAX_CAPTURES],
    depth: usize,
}

/// Finds the first match of `pat` in `src` at or after byte offset `init`.
///
/// A leading `^` anchors the match to `init` — note that in Lua `^` is only an
/// anchor at the *start of the pattern*; anywhere else it is a literal `^`
/// (outside a set).
pub fn find(src: &[u8], pat: &[u8], init: usize) -> Result<Option<Match>> {
    let anchored = pat.first() == Some(&b'^');
    let pat = if anchored { &pat[1..] } else { pat };

    let mut s = init.min(src.len());
    loop {
        let mut m = Matcher {
            src,
            pat,
            level: 0,
            caps: [CapState { init: 0, len: CapLen::Unfinished }; MAX_CAPTURES],
            depth: 0,
        };
        if let Some(end) = m.do_match(s, 0)? {
            return Ok(Some(Match { start: s, end, captures: m.collect_captures()? }));
        }
        if anchored || s >= src.len() {
            return Ok(None);
        }
        s += 1;
    }
}

fn err<T>(msg: &str) -> Result<T> {
    Err(LuaError::runtime(msg))
}

impl Matcher<'_> {
    fn collect_captures(&self) -> Result<Vec<Capture>> {
        let mut out = Vec::with_capacity(self.level);
        for c in &self.caps[..self.level] {
            out.push(match c.len {
                CapLen::Position => Capture::Position(c.init + 1), // Lua is 1-based
                CapLen::Len(n) => Capture::Str(self.src[c.init..c.init + n].to_vec()),
                CapLen::Unfinished => return err("unfinished capture"),
            });
        }
        Ok(out)
    }

    /// The heart of it: match `pat[p..]` against `src[s..]`, returning where the
    /// match ended.
    ///
    /// Written as a loop with `continue` for the sequential cases and recursion
    /// only where backtracking genuinely needs it, so an ordinary pattern like
    /// `%d+%s*` costs no Rust stack per character.
    fn do_match(&mut self, mut s: usize, mut p: usize) -> Result<Option<usize>> {
        self.depth += 1;
        if self.depth > MAX_RECURSION {
            self.depth -= 1;
            return err("pattern too complex");
        }
        let result = self.do_match_inner(&mut s, &mut p);
        self.depth -= 1;
        result
    }

    fn do_match_inner(&mut self, s: &mut usize, p: &mut usize) -> Result<Option<usize>> {
        loop {
            // Pattern exhausted: we matched.
            if *p >= self.pat.len() {
                return Ok(Some(*s));
            }

            match self.pat[*p] {
                b'(' => {
                    // `()` is a position capture; `(` starts a normal one.
                    return if self.pat.get(*p + 1) == Some(&b')') {
                        self.start_capture(*s, *p + 2, CapLen::Position)
                    } else {
                        self.start_capture(*s, *p + 1, CapLen::Unfinished)
                    };
                }
                b')' => return self.end_capture(*s, *p + 1),

                // `$` is an anchor ONLY as the last character of the pattern.
                // Anywhere else it is a literal `$`.
                b'$' if *p + 1 == self.pat.len() => {
                    return Ok(if *s == self.src.len() { Some(*s) } else { None });
                }

                b'%' if *p + 1 < self.pat.len() => match self.pat[*p + 1] {
                    // %b xy -- balanced match.
                    b'b' => return self.match_balance(*s, *p + 2),

                    // %f[set] -- a frontier: matches the empty string at a
                    // transition from a byte NOT in the set to one that is. Used
                    // for word boundaries: `%f[%w]` is "start of a word".
                    b'f' => {
                        *p += 2;
                        if self.pat.get(*p) != Some(&b'[') {
                            return err("missing '[' after '%f' in pattern");
                        }
                        let ep = self.class_end(*p)?;
                        // The byte before the frontier; treated as \0 at the
                        // start of the subject, which is why `%f[%w]` matches at
                        // position 1.
                        let prev = if *s == 0 { 0 } else { self.src[*s - 1] };
                        let curr = if *s < self.src.len() { self.src[*s] } else { 0 };
                        if !self.match_bracket(prev, *p, ep - 1)
                            && self.match_bracket(curr, *p, ep - 1)
                        {
                            *p = ep;
                            continue;
                        }
                        return Ok(None);
                    }

                    // %1 .. %9 -- back-reference to a previous capture.
                    c if c.is_ascii_digit() => {
                        return self.match_capture(*s, (c - b'0') as usize, *p + 2);
                    }

                    // Anything else after % is a character class, handled below.
                    _ => {}
                },
                _ => {}
            }

            // A single character class, possibly followed by a quantifier.
            let ep = self.class_end(*p)?;
            let matches_here =
                *s < self.src.len() && self.single_match(self.src[*s], *p, ep);

            match self.pat.get(ep) {
                // `?` -- try with, then without. Greedy, but only one step.
                Some(b'?') => {
                    if matches_here
                        && let Some(r) = self.do_match(*s + 1, ep + 1)?
                    {
                        return Ok(Some(r));
                    }
                    *p = ep + 1;
                    continue;
                }
                // `+` -- one or more, greedy. One mandatory match, then `*`.
                Some(b'+') => {
                    return if matches_here {
                        self.max_expand(*s + 1, *p, ep)
                    } else {
                        Ok(None)
                    };
                }
                // `*` -- zero or more, greedy: take as many as possible, then
                // give them back one at a time until the rest of the pattern fits.
                Some(b'*') => return self.max_expand(*s, *p, ep),
                // `-` -- zero or more, LAZY. Not a regex `-`. This is the `(.-)`
                // idiom: take as few as possible.
                Some(b'-') => return self.min_expand(*s, *p, ep),
                _ => {}
            }

            if !matches_here {
                return Ok(None);
            }
            *s += 1;
            *p = ep;
        }
    }

    /// Where the character class starting at `p` ends (one past it).
    fn class_end(&self, p: usize) -> Result<usize> {
        let pat = self.pat;
        if p >= pat.len() {
            return err("malformed pattern (ends with '%')");
        }
        let c = pat[p];
        let mut p = p + 1;

        match c {
            b'%' => {
                if p >= pat.len() {
                    return err("malformed pattern (ends with '%')");
                }
                Ok(p + 1)
            }
            b'[' => {
                if pat.get(p) == Some(&b'^') {
                    p += 1;
                }
                // A `]` as the FIRST member of a set is a literal `]`, not the
                // terminator -- so `[]]` is "the set containing ]". That is why
                // this consumes a byte before testing, rather than testing first.
                loop {
                    if p >= pat.len() {
                        return err("malformed pattern (missing ']')");
                    }
                    let cc = pat[p];
                    p += 1;
                    if cc == b'%' {
                        if p >= pat.len() {
                            return err("malformed pattern (missing ']')");
                        }
                        p += 1; // skip the escaped byte, so `%]` is a literal ]
                    }
                    match pat.get(p) {
                        None => return err("malformed pattern (missing ']')"),
                        Some(b']') => break,
                        Some(_) => {}
                    }
                }
                Ok(p + 1)
            }
            _ => Ok(p),
        }
    }

    /// Does byte `c` belong to the class named by `cl` (the byte after a `%`)?
    ///
    /// An uppercase class letter is the **complement** of the lowercase one, and
    /// a `%` before any non-alphanumeric byte is just that byte escaped — which
    /// is how `%.`, `%%`, and `%(` work.
    fn match_class(c: u8, cl: u8) -> bool {
        let matched = match cl.to_ascii_lowercase() {
            b'a' => c.is_ascii_alphabetic(),
            b'c' => c.is_ascii_control(),
            b'd' => c.is_ascii_digit(),
            b'l' => c.is_ascii_lowercase(),
            b'p' => c.is_ascii_punctuation(),
            // C's isspace, which includes the vertical tab. Rust's
            // `is_ascii_whitespace` does NOT, so spelling it out avoids a
            // one-byte divergence from real Lua.
            b's' => matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c),
            b'u' => c.is_ascii_uppercase(),
            b'w' => c.is_ascii_alphanumeric(),
            b'x' => c.is_ascii_hexdigit(),
            // `%z` is the zero byte. Lua 5.1 only; removed in 5.2 once strings
            // could hold embedded zeros without it.
            b'z' => c == 0,
            // Not a class letter: `%` escaped a literal.
            _ => return cl == c,
        };
        if cl.is_ascii_uppercase() { !matched } else { matched }
    }

    /// `[...]` — `p` is at the `[`, `ec` is at the closing `]`.
    fn match_bracket(&self, c: u8, p: usize, ec: usize) -> bool {
        let pat = self.pat;
        let mut sig = true;
        let mut p = p;

        if pat.get(p + 1) == Some(&b'^') {
            sig = false;
            p += 1;
        }

        loop {
            p += 1;
            if p >= ec {
                break;
            }
            if pat[p] == b'%' {
                p += 1;
                if p < ec && Self::match_class(c, pat[p]) {
                    return sig;
                }
            } else if p + 1 < pat.len() && pat[p + 1] == b'-' && p + 2 < ec {
                // A range, `a-z`. Note a `-` at either end of the set is a
                // literal `-`, which the `p + 2 < ec` bound is what enforces.
                p += 2;
                if pat[p - 2] <= c && c <= pat[p] {
                    return sig;
                }
            } else if pat[p] == c {
                return sig;
            }
        }
        !sig
    }

    fn single_match(&self, c: u8, p: usize, ep: usize) -> bool {
        match self.pat[p] {
            // `.` matches ANY byte, including a newline and a zero. Lua has no
            // "dotall" flag because it does not need one.
            b'.' => true,
            b'%' => Self::match_class(c, self.pat[p + 1]),
            b'[' => self.match_bracket(c, p, ep - 1),
            other => other == c,
        }
    }

    /// Greedy: consume as many as possible, then back off one at a time.
    fn max_expand(&mut self, s: usize, p: usize, ep: usize) -> Result<Option<usize>> {
        let mut i = 0;
        while s + i < self.src.len() && self.single_match(self.src[s + i], p, ep) {
            i += 1;
        }
        loop {
            // `ep + 1` skips the quantifier byte itself.
            if let Some(r) = self.do_match(s + i, ep + 1)? {
                return Ok(Some(r));
            }
            if i == 0 {
                return Ok(None);
            }
            i -= 1;
        }
    }

    /// Lazy (`-`): try the rest of the pattern first, and only consume a byte if
    /// that fails. This is what makes `(.-)` stop at the *first* possible end.
    fn min_expand(&mut self, mut s: usize, p: usize, ep: usize) -> Result<Option<usize>> {
        loop {
            if let Some(r) = self.do_match(s, ep + 1)? {
                return Ok(Some(r));
            }
            if s < self.src.len() && self.single_match(self.src[s], p, ep) {
                s += 1;
            } else {
                return Ok(None);
            }
        }
    }

    fn start_capture(&mut self, s: usize, p: usize, what: CapLen) -> Result<Option<usize>> {
        if self.level >= MAX_CAPTURES {
            return err("too many captures");
        }
        self.caps[self.level] = CapState { init: s, len: what };
        self.level += 1;

        let r = self.do_match(s, p)?;
        if r.is_none() {
            // Backtracking past the `(` -- the capture never happened.
            self.level -= 1;
        }
        Ok(r)
    }

    fn end_capture(&mut self, s: usize, p: usize) -> Result<Option<usize>> {
        let l = self.capture_to_close()?;
        self.caps[l].len = CapLen::Len(s - self.caps[l].init);

        let r = self.do_match(s, p)?;
        if r.is_none() {
            self.caps[l].len = CapLen::Unfinished;
        }
        Ok(r)
    }

    /// The innermost capture still waiting for its `)`.
    fn capture_to_close(&self) -> Result<usize> {
        for i in (0..self.level).rev() {
            if matches!(self.caps[i].len, CapLen::Unfinished) {
                return Ok(i);
            }
        }
        err("invalid pattern capture")
    }

    /// `%1`..`%9` — the text must repeat exactly.
    fn match_capture(&mut self, s: usize, idx: usize, p: usize) -> Result<Option<usize>> {
        if idx == 0 || idx > self.level {
            return err("invalid capture index");
        }
        let cap = self.caps[idx - 1];
        let CapLen::Len(len) = cap.len else {
            return err("invalid capture index");
        };
        if self.src.len() - s >= len && self.src[cap.init..cap.init + len] == self.src[s..s + len]
        {
            self.do_match(s + len, p)
        } else {
            Ok(None)
        }
    }

    /// `%bxy` — matches a balanced run starting at `x` and ending at the `y` that
    /// balances it. `%b()` over `(a(b)c)` matches the whole thing, not `(a(b)`.
    fn match_balance(&mut self, s: usize, p: usize) -> Result<Option<usize>> {
        if p + 1 >= self.pat.len() {
            return err("malformed pattern (missing arguments to '%b')");
        }
        if s >= self.src.len() || self.src[s] != self.pat[p] {
            return Ok(None);
        }
        let (open, close) = (self.pat[p], self.pat[p + 1]);
        let mut depth = 1;
        let mut i = s + 1;

        while i < self.src.len() {
            if self.src[i] == close {
                depth -= 1;
                if depth == 0 {
                    return self.do_match(i + 1, p + 2);
                }
            } else if self.src[i] == open {
                depth += 1;
            }
            i += 1;
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `find`, returning the matched text.
    fn m(src: &str, pat: &str) -> Option<String> {
        find(src.as_bytes(), pat.as_bytes(), 0)
            .unwrap()
            .map(|m| String::from_utf8_lossy(&src.as_bytes()[m.start..m.end]).into_owned())
    }

    /// `find`, returning the captures as strings (position captures as numbers).
    fn caps(src: &str, pat: &str) -> Option<Vec<String>> {
        find(src.as_bytes(), pat.as_bytes(), 0).unwrap().map(|m| {
            m.captures
                .iter()
                .map(|c| match c {
                    Capture::Str(s) => String::from_utf8_lossy(s).into_owned(),
                    Capture::Position(p) => p.to_string(),
                })
                .collect()
        })
    }

    #[test]
    fn literals_and_the_any_class() {
        assert_eq!(m("hello", "ell"), Some("ell".into()));
        assert_eq!(m("hello", "xyz"), None);
        // `.` is ANY byte, so `h.l` matches "hel" (h, e, l).
        assert_eq!(m("hello", "h.l"), Some("hel".into()));
        assert_eq!(m("hello", "h.."), Some("hel".into()));
        // ...but a literal `%.` does not.
        assert_eq!(m("hello", "h%.l"), None);
        // `.` matches a newline too -- no "dotall" flag needed.
        assert_eq!(m("a\nb", "a.b"), Some("a\nb".into()));
    }

    #[test]
    fn character_classes() {
        assert_eq!(m("abc123", "%d+"), Some("123".into()));
        assert_eq!(m("abc123", "%a+"), Some("abc".into()));
        assert_eq!(m("  x", "%s+"), Some("  ".into()));
        assert_eq!(m("a_1", "%w+"), Some("a".into()), "_ is not %w in Lua");
        assert_eq!(m("HELLO", "%u+"), Some("HELLO".into()));
        assert_eq!(m("hello", "%l+"), Some("hello".into()));
        assert_eq!(m("a.b", "%p"), Some(".".into()));
        assert_eq!(m("ff00", "%x+"), Some("ff00".into()));
    }

    #[test]
    fn uppercase_classes_are_complements() {
        assert_eq!(m("abc123", "%D+"), Some("abc".into()));
        assert_eq!(m("abc123", "%A+"), Some("123".into()));
        assert_eq!(m("a b", "%S+"), Some("a".into()));
    }

    #[test]
    fn percent_escapes_a_literal() {
        // `%.` is a literal dot -- the single most common escape in real code.
        assert_eq!(m("a.b", "%."), Some(".".into()));
        assert_eq!(m("abc", "%."), None, "a literal dot does not match 'a'");
        assert_eq!(m("50%", "%%"), Some("%".into()));
        assert_eq!(m("f(x)", "%("), Some("(".into()));
    }

    #[test]
    fn quantifiers() {
        assert_eq!(m("aaa", "a*"), Some("aaa".into()));
        assert_eq!(m("bbb", "a*"), Some("".into()), "* matches empty");
        assert_eq!(m("aaa", "a+"), Some("aaa".into()));
        assert_eq!(m("bbb", "a+"), None, "+ needs at least one");
        assert_eq!(m("color", "colou?r"), Some("color".into()));
        assert_eq!(m("colour", "colou?r"), Some("colour".into()));
    }

    #[test]
    fn the_dash_is_lazy_not_a_literal() {
        // THE Lua-vs-regex trap. `-` is a lazy `*`.
        // Greedy `.*` takes everything to the LAST quote.
        assert_eq!(m(r#""a" and "b""#, r#"".*""#), Some(r#""a" and "b""#.into()));
        // Lazy `.-` stops at the FIRST.
        assert_eq!(m(r#""a" and "b""#, r#"".-""#), Some(r#""a""#.into()));
        // The classic idiom.
        assert_eq!(caps("<tag>body</tag>", "<(.-)>"), Some(vec!["tag".into()]));
    }

    #[test]
    fn sets() {
        assert_eq!(m("hello", "[aeiou]"), Some("e".into()));
        assert_eq!(m("hello", "[^aeiou]+"), Some("h".into()));
        assert_eq!(m("abc123", "[a-c]+"), Some("abc".into()));
        assert_eq!(m("xyz", "[a-c]"), None);
        // A class inside a set.
        assert_eq!(m("a1!", "[%d%a]+"), Some("a1".into()));
        // A `-` at the edge of a set is a literal `-`.
        assert_eq!(m("a-b", "[-]"), Some("-".into()));
        // A `]` first in a set is a literal `]`.
        assert_eq!(m("a]b", "[]]"), Some("]".into()));
        assert_eq!(m("abc", "[^]]+"), Some("abc".into()));
    }

    #[test]
    fn anchors() {
        assert_eq!(m("hello", "^he"), Some("he".into()));
        assert_eq!(m("hello", "^el"), None, "^ anchors to the start");
        assert_eq!(m("hello", "lo$"), Some("lo".into()));
        assert_eq!(m("hello", "he$"), None, "$ anchors to the end");
        assert_eq!(m("hello", "^hello$"), Some("hello".into()));
        // `$` in the middle is a literal.
        assert_eq!(m("a$b", "a$b"), Some("a$b".into()));
    }

    #[test]
    fn captures() {
        assert_eq!(caps("key=value", "(%w+)=(%w+)"), Some(vec!["key".into(), "value".into()]));
        assert_eq!(caps("hello", "(h)(e)"), Some(vec!["h".into(), "e".into()]));
        // Nested.
        assert_eq!(caps("abc", "((a)b)"), Some(vec!["ab".into(), "a".into()]));
        // No captures at all.
        assert_eq!(caps("abc", "abc"), Some(vec![]));
    }

    #[test]
    fn position_captures_are_one_based_numbers() {
        // `()` captures a POSITION, not text -- and Lua counts from 1.
        assert_eq!(caps("hello", "()ll"), Some(vec!["3".into()]));
        assert_eq!(caps("hello", "^()"), Some(vec!["1".into()]));
    }

    #[test]
    fn back_references() {
        // `%1` must match the SAME text the first capture did.
        assert_eq!(caps("abab", "(ab)%1"), Some(vec!["ab".into()]));
        assert_eq!(m("abcd", "(ab)%1"), None);
        // The classic doubled-word finder.
        assert_eq!(caps("the the cat", "(%w+) %1"), Some(vec!["the".into()]));
    }

    #[test]
    fn balanced_match() {
        assert_eq!(m("(a(b)c) rest", "%b()"), Some("(a(b)c)".into()));
        assert_eq!(m("f(x, g(y))", "%b()"), Some("(x, g(y))".into()));
        assert_eq!(m("(unclosed", "%b()"), None);
        assert_eq!(m("{a{b}}", "%b{}"), Some("{a{b}}".into()));
    }

    #[test]
    fn frontier_pattern() {
        // `%f[%w]` matches the empty string at the start of a word.
        assert_eq!(caps("the cat", "%f[%w](%w+)"), Some(vec!["the".into()]));
        // Matching at the very start of the subject works because the byte
        // "before" it is treated as \0.
        assert!(find(b"hello", b"%f[%w]", 0).unwrap().is_some());
        // And there is no frontier in the middle of a word.
        assert!(find(b"hello", b"l%f[%w]", 0).unwrap().is_none());
    }

    #[test]
    fn init_offset_and_repeated_finds() {
        // The mechanism `gmatch` is built on: find, then search again past the end.
        let src = b"a1 b2 c3";
        let mut at = 0;
        let mut found = Vec::new();
        while let Some(m) = find(src, b"%a%d", at).unwrap() {
            found.push(String::from_utf8_lossy(&src[m.start..m.end]).into_owned());
            at = m.end;
        }
        assert_eq!(found, vec!["a1", "b2", "c3"]);
    }

    #[test]
    fn malformed_patterns_error_rather_than_panic() {
        assert!(find(b"x", b"[abc", 0).is_err(), "unclosed set");
        assert!(find(b"x", b"%", 0).is_err(), "trailing %");
        assert!(find(b"x", b"%b", 0).is_err(), "%b needs two chars");
        assert!(find(b"x", b"%f", 0).is_err(), "%f needs a set");

        // An unfinished capture errors only when the match SUCCEEDS with the
        // capture still open -- `(a` against "a". Against "x" the match simply
        // fails, and a failed match is nil, not an error. Real Lua draws the line
        // in exactly this place.
        assert!(find(b"a", b"(a", 0).is_err(), "unfinished capture on a match");
        assert!(
            matches!(find(b"x", b"(a", 0), Ok(None)),
            "a pattern that just does not match is nil, not an error"
        );
    }

    #[test]
    fn pathological_backtracking_errors_rather_than_overflowing_the_stack() {
        // A hostile pattern must not take the process down.
        let src = "a".repeat(200);
        let r = find(src.as_bytes(), b"(a*)*b", 0);
        assert!(r.is_err() || r.unwrap().is_none());
    }

    #[test]
    fn patterns_operate_on_bytes() {
        // A pattern over UTF-8 matches bytes, exactly as in real Lua. "é" is two
        // bytes, so `.` matches only the first of them.
        let src = "é".as_bytes();
        let mm = find(src, b".", 0).unwrap().unwrap();
        assert_eq!(mm.end - mm.start, 1);
    }
}
