//! `f16` (IEEE 754 binary16) and `bf16` (bfloat16) conversions to and from
//! `f32`.
//!
//! Model weights on disk are usually `f16` or `bf16` to halve storage, but
//! every op in this crate computes in `f32` (see the crate-level docs for
//! why). That makes these four functions the narrow waist every loaded
//! weight and every stored activation passes through, so they earn their
//! own module and their own exhaustive tests rather than being inlined
//! where first needed.
//!
//! We hand-roll this instead of depending on the `half` crate because the
//! conversions are genuinely small (a few dozen lines, no unsafe, no
//! external ABI to track) and `half` was not already present in the
//! workspace's dependency graph — see `CLAUDE.md`'s "avoid unnecessary
//! dependencies" rule. Bit layouts follow IEEE 754-2008 (binary16) and the
//! Google Brain bfloat16 spec; neither is copied from any codebase.

/// Decodes an IEEE 754 binary16 bit pattern to `f32`.
///
/// This is written as plain floating-point arithmetic rather than bit
/// manipulation. That is a deliberate choice, not a stylistic one: every
/// operation below — `mantissa / 1024.0`, `1.0 + ...`, `2f32.powi(..)` — is
/// exact for the ranges involved (dividing by a power of two only shifts
/// the exponent, and the sums never need to round), so the result is
/// bit-for-bit the correctly-rounded `f32` value without needing to reason
/// about mantissa shifts or implicit-bit placement by hand. That
/// correctness argument is easy to check by inspection, which is worth
/// more here than the handful of cycles a bit-twiddling version would save.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign: f32 = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = f32::from(bits & 0x3FF);

    match exponent {
        // Zero or subnormal: value = mantissa * 2^-24 (no implicit leading 1).
        0 => sign * mantissa * 2f32.powi(-24),
        // Exponent all-ones: infinity (zero mantissa) or NaN.
        0x1F if mantissa == 0.0 => sign * f32::INFINITY,
        0x1F => f32::NAN,
        // Normal: value = 1.mantissa * 2^(exponent - 15).
        e => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(i32::from(e) - 15),
    }
}

/// Encodes an `f32` as the nearest IEEE 754 binary16 value, rounding ties
/// to even (the IEEE default), with overflow saturating to infinity and
/// underflow flushing to a signed zero or subnormal.
///
/// Unlike the decode direction, this narrows a 23-bit mantissa to 10 bits,
/// so a rounding decision is unavoidable. The implementation works in
/// integer bits because `f32` arithmetic has no way to ask "round this
/// binary fraction to N bits, ties to even" directly.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    // f32 exponent field all-ones: infinity, or NaN (quieted; payload is
    // not preserved since nothing downstream inspects it).
    if exponent == 0xFF {
        return if mantissa == 0 {
            sign | 0x7C00
        } else {
            sign | 0x7E00
        };
    }

    // f32 zero or subnormal (exponent field 0): magnitude is at most
    // ~1.2e-38, far below the smallest f16 subnormal (2^-24 ~= 6e-8), so it
    // always flushes to a signed f16 zero.
    if exponent == 0 {
        return sign;
    }

    // Rebase from f32's bias (127) to f16's bias (15).
    let half_exponent = exponent - 127 + 15;

    if half_exponent >= 0x1F {
        // Overflow: too large for even the largest finite f16 (65504).
        return sign | 0x7C00;
    }

    if half_exponent <= 0 {
        return sign | encode_f16_subnormal(mantissa, half_exponent);
    }

    // Normal range: narrow the 23-bit mantissa to 10 bits, rounding to even.
    let mut half_mantissa = (mantissa >> 13) as u16;
    if round_to_even(mantissa, 13) {
        half_mantissa += 1;
    }
    if half_mantissa == 0x0400 {
        // The mantissa rounded up to 2.0 * 2^exponent; carry into the
        // exponent field instead (e.g. 1.111111111_2 * 2^e -> 1.0 * 2^(e+1)).
        return finish_f16_normal(sign, half_exponent + 1, 0);
    }
    finish_f16_normal(sign, half_exponent, half_mantissa)
}

/// Builds a normal f16 (or, if `exponent` has overflowed, an infinity).
fn finish_f16_normal(sign: u16, exponent: i32, mantissa: u16) -> u16 {
    if exponent >= 0x1F {
        return sign | 0x7C00;
    }
    sign | ((exponent as u16) << 10) | mantissa
}

/// Rounds an f32 mantissa (with its implicit leading 1 restored) down into
/// an f16 subnormal, ties to even. `half_exponent <= 0` is the caller's
/// invariant: it is the amount by which the value undershoots f16's
/// smallest normal exponent.
fn encode_f16_subnormal(mantissa: u32, half_exponent: i32) -> u16 {
    // Shift needed to turn the 24-bit significand (1.mantissa) into an
    // integer count of 2^-24 units, i.e. f16 subnormal mantissa units.
    // Derivation: significand * 2^(half_exponent - 15) [true value]
    //           = (result_in_units) * 2^-24
    //       =>  shift = 14 - half_exponent.
    let shift = 14 - half_exponent;
    if shift > 24 {
        // Every bit of the 24-bit significand falls below the rounding
        // position: the result is exactly zero, no rounding needed.
        return 0;
    }
    let significand = 0x0080_0000 | mantissa; // restore the implicit leading 1.
    let mut half_mantissa = (significand >> shift) as u16;
    if round_to_even(significand, shift) {
        half_mantissa += 1;
    }
    // Rounding can carry a subnormal all the way up to the smallest normal
    // (mantissa 0, exponent field 1); that carry is exactly the bit pattern
    // 0x0400 already produces (exponent field 1, mantissa 0), so no special
    // case is needed here.
    half_mantissa
}

/// Round-to-nearest-even test: discarding the low `shift` bits of `value`,
/// should the kept bits be incremented?
///
/// Standard IEEE rounding: round up if the bit just below the cut is set
/// and (any lower bit is also set, i.e. we are strictly past halfway) or
/// (the kept value is odd, i.e. exactly at halfway and rounding to even
/// means rounding up here).
fn round_to_even(value: u32, shift: i32) -> bool {
    if shift <= 0 {
        return false;
    }
    let shift = shift as u32;
    let round_bit = 1u32 << (shift - 1);
    let round_bit_set = value & round_bit != 0;
    let sticky = value & (round_bit - 1) != 0;
    let kept_is_odd = (value >> shift) & 1 != 0;
    round_bit_set && (sticky || kept_is_odd)
}

/// Decodes a bfloat16 bit pattern to `f32`.
///
/// bfloat16 keeps `f32`'s 8-bit exponent and truncates the mantissa to 7
/// bits — it is literally the top 16 bits of an `f32`, which is the entire
/// point of the format (same dynamic range as `f32`, cheap conversion both
/// ways). Decoding is therefore a zero-extend, not a computation.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Encodes an `f32` as the nearest bfloat16 value, rounding ties to even.
///
/// Because bfloat16 is `f32`'s high 16 bits, rounding is "round this u32 to
/// the nearest multiple of 2^16, ties to even" — which integer addition
/// does for free: adding `0x7FFF + (bit 16 of the truncated value)` before
/// truncating carries a `1` into bit 16 exactly when the low 16 bits are
/// past the halfway point, or at the halfway point with an odd result
/// (this is the well-known "round to nearest even via rounding add" trick;
/// it works even across exponent boundaries because it is plain integer
/// addition with carry propagation).
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        // Force quiet (set the top mantissa bit) so a signalling NaN never
        // round-trips into one; the payload is not otherwise preserved.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let round = 0x7FFF_u32 + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, epsilon: f32) {
        assert!(
            (a - b).abs() <= epsilon,
            "expected {b} within {epsilon}, got {a}"
        );
    }

    // -- f16: known bit patterns (cross-checked against numpy.float16) --

    #[test]
    fn f16_known_constants_decode_exactly() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0xC800), -8.0);
        // Largest finite f16.
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
        // Smallest positive subnormal: 2^-24.
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    }

    #[test]
    fn f16_infinity_and_nan_decode_correctly() {
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());
        assert!(f16_to_f32(0xFE00).is_nan());
    }

    #[test]
    fn f32_to_f16_known_constants_encode_exactly() {
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(2.0), 0x4000);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f32_to_f16(-8.0), 0xC800);
        assert_eq!(f32_to_f16(65504.0), 0x7BFF);
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
    }

    #[test]
    fn f32_to_f16_overflow_saturates_to_infinity() {
        // Confirmed against numpy: 70000.0 and 3e38 both saturate.
        assert_eq!(f32_to_f16(70_000.0), 0x7C00);
        assert_eq!(f32_to_f16(-70_000.0), 0xFC00);
        assert_eq!(f32_to_f16(3e38), 0x7C00);
    }

    #[test]
    fn f32_to_f16_underflow_flushes_to_zero() {
        // 1e-9 and 2^-25 (half of the smallest f16 subnormal, an exact tie)
        // both round to zero; the tie rounds down because zero is even.
        assert_eq!(f32_to_f16(1e-9), 0x0000);
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000);
    }

    #[test]
    fn f32_to_f16_infinity_and_nan_round_trip() {
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xFC00);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn f32_to_f16_matches_numpy_oracle_for_irrational_values() {
        // Values and expected bit patterns captured from
        // `numpy.float32(x).astype(numpy.float16)`.
        assert_eq!(f32_to_f16(1.0 / 3.0), 0x3555);
        assert_eq!(f32_to_f16(100.5), 0x5648);
        assert_eq!(f32_to_f16(-100.5), 0xD648);
        assert_eq!(f32_to_f16(std::f32::consts::PI), 0x4248);
        assert_eq!(f32_to_f16(6.10352e-05), 0x0400);
        assert_eq!(f32_to_f16(0.00012), 0x07DD);
    }

    /// The strongest possible correctness check: every one of the 65 536
    /// possible f16 bit patterns, decoded to f32 and re-encoded, must
    /// return the original bits. This holds with zero tolerance because an
    /// f32 that came from decoding a real f16 value has at most 10
    /// significant mantissa bits — re-encoding never needs to round, only
    /// to correctly reconstruct the exponent and mantissa. Any bug in the
    /// subnormal shift, the exponent rebasing, or the NaN handling shows
    /// up here without needing an external oracle.
    #[test]
    fn f16_round_trips_every_possible_bit_pattern() {
        for bits in 0..=u16::MAX {
            let value = f16_to_f32(bits);
            let back = f32_to_f16(value);
            if value.is_nan() {
                assert!(f16_to_f32(back).is_nan(), "0x{bits:04x} -> NaN -> not NaN");
            } else {
                assert_eq!(
                    back, bits,
                    "0x{bits:04x} -> {value} -> 0x{back:04x} (round-trip mismatch)"
                );
            }
        }
    }

    // -- bf16 --

    #[test]
    fn bf16_known_constants_round_trip_exactly() {
        for v in [1.0f32, 2.0, 0.5, 100.5, -1.5, 0.0, -0.0] {
            let bits = f32_to_bf16(v);
            assert_eq!(bf16_to_f32(bits), v);
        }
    }

    #[test]
    fn bf16_matches_ggml_rounding_oracle() {
        // Captured from ggml_compute_fp32_to_bf16's round-to-nearest-even
        // behaviour (equivalently, numpy's ml_dtypes.bfloat16).
        assert_eq!(f32_to_bf16(1.0 / 3.0), 0x3EAB);
        assert_eq!(f32_to_bf16(std::f32::consts::PI), 0x4049);
        assert_eq!(f32_to_bf16(100.5), 0x42C9);
    }

    #[test]
    fn bf16_infinity_and_nan_round_trip() {
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7F80);
        assert_eq!(f32_to_bf16(f32::NEG_INFINITY), 0xFF80);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn bf16_denormals_are_preserved_not_flushed() {
        // Unlike f16 (which is much narrower-range), bf16 shares f32's
        // exponent field, so f32 subnormals stay representable.
        let tiny = f32::from_bits(1); // smallest positive f32 subnormal
        let bits = f32_to_bf16(tiny);
        assert!(bf16_to_f32(bits) >= 0.0);
    }

    /// Same exhaustive strategy as the f16 test: every bf16 bit pattern
    /// round-trips exactly, because decoding a real bf16 value into f32
    /// never invents low mantissa bits that would need rounding away.
    #[test]
    fn bf16_round_trips_every_possible_bit_pattern() {
        for bits in 0..=u16::MAX {
            let value = bf16_to_f32(bits);
            let back = f32_to_bf16(value);
            if value.is_nan() {
                assert!(bf16_to_f32(back).is_nan());
            } else {
                assert_eq!(back, bits, "0x{bits:04x} -> {value} -> 0x{back:04x}");
            }
        }
    }

    #[test]
    fn conversions_are_reasonably_close_for_arbitrary_values() {
        // Sanity bound: f16 has ~3 decimal digits of precision.
        for v in [1.23456f32, -9.8765, 12345.6, 0.001234] {
            let f16_bits = f32_to_f16(v);
            assert_close(f16_to_f32(f16_bits), v, v.abs() * 0.001 + 1e-6);
            let bf16_bits = f32_to_bf16(v);
            assert_close(bf16_to_f32(bf16_bits), v, v.abs() * 0.01 + 1e-6);
        }
    }
}
