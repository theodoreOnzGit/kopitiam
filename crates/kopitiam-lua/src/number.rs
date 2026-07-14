//! Turning numbers into strings and back, exactly the way Lua 5.1 does.
//!
//! This module looks like trivia and is not. Lua 5.1's number-to-string
//! conversion is defined by one C macro:
//!
//! ```c
//! #define LUAI_NUMFFORMAT  "%.14g"
//! ```
//!
//! Every `print(x)`, every `tostring(x)`, every `"" .. x`, and every string key
//! built from a number goes through it. Get it wrong and `print(1)` says `1.0`,
//! `t["1"]` stops matching what the user expects, and a config that formats a
//! number into a vim option (`vim.opt.colorcolumn = 75`) writes `"75.0"`.
//!
//! Rust has no `%g`, so [`format_number`] implements C's `%.14g` directly. The
//! rules, from the C standard:
//!
//! * Let `P` be the precision (14 here) and `X` the decimal exponent of the
//!   value once rounded to `P` significant digits.
//! * If `X < -4` or `X >= P`, use `%e` style (`1e+20`).
//! * Otherwise use `%f` style with `P - 1 - X` digits after the point.
//! * Either way, **strip trailing zeros** in the fraction, and then a trailing
//!   point.
//!
//! That last rule is the one that makes `tostring(3.0) == "3"`.

/// Formats a number the way Lua 5.1's `tostring` does: C's `"%.14g"`.
///
/// ```text
/// 1         -> "1"          (not "1.0" -- the trailing-zero strip)
/// 3.0       -> "3"
/// 1.5       -> "1.5"
/// 1/3       -> "0.33333333333333"   (14 significant digits)
/// 1e20      -> "1e+20"      (exponent >= precision, so %e style)
/// 2^53      -> "9.007199254741e+15"
/// ```
pub fn format_number(v: f64) -> String {
    format_g(v, 14)
}

/// C's `%.*g`. Split out from [`format_number`] so `string.format("%g")` and
/// `%.3g` can share it rather than reimplementing the trailing-zero rules.
pub fn format_g(v: f64, precision: usize) -> String {
    if v.is_nan() {
        // Lua 5.1 prints whatever the platform's printf does; glibc gives
        // "nan"/"-nan". We normalise to "nan" and "-nan" so that behaviour does
        // not vary by platform, which CLAUDE.md's determinism rule requires.
        return if v.is_sign_negative() { "-nan".to_string() } else { "nan".to_string() };
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".to_string() } else { "inf".to_string() };
    }

    // C says a precision of 0 is treated as 1.
    let p = precision.max(1);

    // Get the decimal exponent *after* rounding to `p` significant digits.
    // Doing it via log10 would be wrong at the boundaries: 9.9999e-5 rounds up
    // to 1e-4, which changes which style %g picks. Formatting to exponential
    // first and reading the exponent back gets the rounded answer by
    // construction.
    let sci = format!("{:.*e}", p - 1, v);
    let (mantissa, exp) = sci.split_once('e').expect("Rust always emits an 'e' in {:e}");
    let exp: i32 = exp.parse().expect("Rust always emits a valid exponent");

    if exp < -4 || exp >= p as i32 {
        let mantissa = strip_trailing_zeros(mantissa);
        // C pads the exponent to at least two digits: `1e+20`, `1e+05`.
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", mantissa, sign, exp.abs())
    } else {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        strip_trailing_zeros(&format!("{v:.decimals$}"))
    }
}

/// Removes trailing zeros from a fractional part, then a bare trailing point.
/// `"1.5000"` -> `"1.5"`, `"3.0000"` -> `"3"`, `"100"` -> `"100"` (untouched:
/// there is no point, so those zeros are significant).
fn strip_trailing_zeros(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let s = s.trim_end_matches('0');
    s.strip_suffix('.').unwrap_or(s).to_string()
}

/// Parses a Lua numeric literal, as `tonumber` does.
///
/// Accepts what Lua 5.1 accepts and nothing more:
///
/// * optional surrounding whitespace, optional sign
/// * decimal: `3`, `3.14`, `.5`, `3.`, `1e10`, `1E-2`, `3.14e+2`
/// * hexadecimal **integers**: `0xff`, `0XFF` (Lua 5.1 has no hex floats;
///   `0x1p4` is a 5.2 feature and is correctly rejected here)
///
/// Returns `None` for anything else — including a string with trailing garbage
/// (`"10abc"`), which must fail so that `tonumber("10abc")` is `nil`.
pub fn parse_number(bytes: &[u8]) -> Option<f64> {
    let s = std::str::from_utf8(bytes).ok()?;
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Hex: sign, then 0x, then at least one hex digit, and nothing else.
    let (sign, rest) = match s.as_bytes()[0] {
        b'-' => (-1.0, &s[1..]),
        b'+' => (1.0, &s[1..]),
        _ => (1.0, s),
    };
    if let Some(hex) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
        if hex.is_empty() || !hex.bytes().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        // Lua 5.1 reads hex literals with C's `strtoul`, which wraps at 32/64
        // bits. Accumulating in f64 keeps very long literals finite and
        // deterministic rather than platform-dependent.
        let mut acc = 0.0f64;
        for c in hex.bytes() {
            acc = acc * 16.0 + (c as char).to_digit(16).expect("checked hex digit") as f64;
        }
        return Some(sign * acc);
    }

    // Decimal. Rust's f64 parser accepts `inf`, `NaN`, and `1e5` but Lua does
    // not accept the first two as *literals*, so reject them explicitly:
    // `tonumber("inf")` is nil in Lua 5.1.
    if s.bytes().any(|c| !matches!(c, b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')) {
        return None;
    }
    s.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tostring_drops_the_pointless_decimal_point() {
        // The whole reason this module exists.
        assert_eq!(format_number(1.0), "1");
        assert_eq!(format_number(3.0), "3");
        assert_eq!(format_number(-0.0), "-0");
        assert_eq!(format_number(0.0), "0");
        assert_eq!(format_number(75.0), "75");
    }

    #[test]
    fn fourteen_significant_digits() {
        assert_eq!(format_number(1.0 / 3.0), "0.33333333333333");
        assert_eq!(format_number(1.5), "1.5");
        assert_eq!(format_number(1.23456), "1.23456");
    }

    #[test]
    fn switches_to_exponential_outside_the_g_window() {
        assert_eq!(format_number(1e20), "1e+20");
        assert_eq!(format_number(1e-5), "1e-05");
        assert_eq!(format_number(2f64.powi(53)), "9.007199254741e+15");
        // Just inside the window: exponent 13 < precision 14, so plain digits.
        assert_eq!(format_number(1e13), "10000000000000");
    }

    #[test]
    fn non_finite_numbers_are_platform_independent() {
        assert_eq!(format_number(f64::INFINITY), "inf");
        assert_eq!(format_number(f64::NEG_INFINITY), "-inf");
        assert_eq!(format_number(f64::NAN), "nan");
    }

    #[test]
    fn tonumber_accepts_what_lua_51_accepts() {
        assert_eq!(parse_number(b"3"), Some(3.0));
        assert_eq!(parse_number(b"1.25"), Some(1.25));
        assert_eq!(parse_number(b".5"), Some(0.5));
        assert_eq!(parse_number(b"1e10"), Some(1e10));
        assert_eq!(parse_number(b"1E-2"), Some(0.01));
        assert_eq!(parse_number(b"  42  "), Some(42.0), "surrounding space is allowed");
        assert_eq!(parse_number(b"0xff"), Some(255.0));
        assert_eq!(parse_number(b"0XFF"), Some(255.0));
        assert_eq!(parse_number(b"-0x10"), Some(-16.0));
    }

    #[test]
    fn tonumber_rejects_what_lua_51_rejects() {
        assert_eq!(parse_number(b"10abc"), None, "trailing garbage must fail");
        assert_eq!(parse_number(b""), None);
        assert_eq!(parse_number(b"abc"), None);
        assert_eq!(parse_number(b"inf"), None, "Lua 5.1 does not parse 'inf'");
        assert_eq!(parse_number(b"nan"), None);
        assert_eq!(parse_number(b"0x"), None);
        // Hex floats are Lua 5.2+, not 5.1.
        assert_eq!(parse_number(b"0x1p4"), None);
    }
}
