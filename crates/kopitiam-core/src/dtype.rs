use std::fmt;

/// The element type of a tensor.
///
/// This enumerates every numeric format the Kopitiam Runtime can hold in a
/// tensor, including the block-quantized formats used by GGUF weights. It
/// deliberately does *not* say how a format is computed on — a kernel may
/// dequantize a [`DType::Q4_0`] block to `f32` before multiplying, or it may
/// have a fused quantized path. That is `kopitiam-kernels`' business; this
/// type only describes what the bytes mean.
///
/// # Why block-quantized formats are dtypes and not a separate concept
///
/// It is tempting to model quantized weights as "a `f32` tensor with a
/// compression scheme attached". That is the wrong abstraction: a Q4_0
/// tensor genuinely cannot be indexed elementwise without decoding a whole
/// block, so pretending it is a float tensor produces an API that lies. By
/// making quantization a property of the element type, every consumer is
/// forced to ask "can I index this?" (see [`DType::is_quantized`]) instead
/// of finding out at runtime.
// The variant names deliberately mirror ggml's own type names (`Q4_0`,
// `Q4_K`, ...) so a reader cross-referencing a GGUF file or ggml source sees
// the identical spelling. That means embracing the underscores ggml uses,
// which `non_camel_case_types` would otherwise flag (`Q4_K` -> `Q4K`);
// renaming for the lint would make these harder, not easier, to match up.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE 754 single precision.
    F32,
    /// IEEE 754 half precision.
    F16,
    /// bfloat16: same exponent range as `f32`, fewer mantissa bits. The
    /// usual format for model weights that must survive `f32` dynamic range
    /// without paying for 32 bits.
    BF16,
    /// Signed 8-bit integer.
    I8,
    /// Signed 32-bit integer, mostly for token ids and indices.
    I32,
    /// 4-bit block quantization, symmetric, 32 elements per block plus one
    /// `f16` scale.
    Q4_0,
    /// 4-bit block quantization, asymmetric: 32 elements per block plus an
    /// `f16` scale and an `f16` minimum.
    Q4_1,
    /// 5-bit block quantization, symmetric, 32 elements per block.
    Q5_0,
    /// 5-bit block quantization, asymmetric, 32 elements per block.
    Q5_1,
    /// 8-bit block quantization, symmetric, 32 elements per block plus one
    /// `f16` scale.
    Q8_0,
    /// ggml "K-quant" 2-bit format. Superblock of 256 elements: 16 sub-block
    /// scales+mins packed 4 bits each, 2-bit quants, an `f16` super-scale and
    /// an `f16` super-min. 84 bytes per superblock.
    Q2_K,
    /// ggml "K-quant" 3-bit format. Superblock of 256: a high-bit mask, 2-bit
    /// low quants, 16 six-bit sub-block scales packed into 12 bytes, and one
    /// `f16` super-scale. 110 bytes per superblock.
    Q3_K,
    /// ggml "K-quant" 4-bit format (the `Q4_K_M` workhorse). Superblock of
    /// 256: `f16` super-scale, `f16` super-min, 8 six-bit sub-block scales and
    /// 8 six-bit sub-block mins packed into 12 bytes, then 4-bit quants. 144
    /// bytes per superblock.
    Q4_K,
    /// ggml "K-quant" 5-bit format. Like [`Self::Q4_K`] plus a 32-byte
    /// high-bit field promoting each 4-bit quant to 5 bits. 176 bytes per
    /// superblock.
    Q5_K,
    /// ggml "K-quant" 6-bit format. Superblock of 256: 4-bit low quants, 2-bit
    /// high quants, 16 signed `i8` sub-block scales and one `f16` super-scale.
    /// 210 bytes per superblock.
    Q6_K,
    /// ggml "K-quant" 8-bit format. Superblock of 256: an `f32` scale, 256
    /// signed `i8` quants and 16 `i16` group sums. 292 bytes per superblock.
    /// Used by ggml only as an intermediate activation type for K-quant dot
    /// products, never as an on-disk weight format; supported here for
    /// completeness. `f32` (not `f16`) scale, unlike every other quant.
    Q8_K,
}

impl DType {
    /// Number of tensor elements encoded per storage block.
    ///
    /// Non-quantized types are trivially one element per "block". Quantized
    /// types pack [`Self::block_size`] elements into [`Self::block_bytes`]
    /// bytes, which is why a quantized tensor's element count must be a
    /// multiple of this.
    pub const fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::I8 | Self::I32 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => 32,
            // ggml K-quants are "super-blocks" of QK_K = 256 elements.
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
        }
    }

    /// Bytes occupied by one block of [`Self::block_size`] elements.
    ///
    /// The quantized numbers below are the on-disk GGUF block layouts:
    /// Q4_0 is a 2-byte `f16` scale plus 32 4-bit weights (16 bytes) = 18;
    /// Q4_1 adds a 2-byte minimum = 20; Q5_0 adds a 4-byte high-bit field to
    /// Q4_0 = 22; Q5_1 likewise on Q4_1 = 24; Q8_0 is a 2-byte scale plus 32
    /// bytes = 34.
    ///
    /// The K-quant super-block sizes below are the `sizeof(block_qX_K)`
    /// static-asserted in ggml's `ggml-common.h` (MIT), with `QK_K = 256`
    /// and `K_SCALE_SIZE = 12`:
    /// * Q2_K = `2*f16 + QK_K/16 + QK_K/4` = 4 + 16 + 64 = **84**
    /// * Q3_K = `f16 + QK_K/8 + QK_K/4 + 12` = 2 + 32 + 64 + 12 = **110**
    /// * Q4_K = `2*f16 + 12 + QK_K/2` = 4 + 12 + 128 = **144**
    /// * Q5_K = `2*f16 + 12 + QK_K/2 + QK_K/8` = 4 + 12 + 128 + 32 = **176**
    /// * Q6_K = `f16 + QK_K/16 + 3*QK_K/4` = 2 + 16 + 192 = **210**
    /// * Q8_K = `f32 + QK_K + QK_K/16*i16` = 4 + 256 + 32 = **292**
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 => 1,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
        }
    }

    /// Whether this type packs multiple elements into a shared block with
    /// its own scale, and therefore cannot be indexed one element at a time.
    pub const fn is_quantized(self) -> bool {
        self.block_size() > 1
    }

    /// Whether this type is a floating-point format that kernels can compute
    /// on directly.
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }

    /// Storage bytes needed to hold `elements` values of this type.
    ///
    /// Returns `None` when `elements` is not a whole number of blocks, which
    /// for a quantized type is not a representable tensor rather than a
    /// rounding question — surfacing it as `None` keeps callers from
    /// silently allocating a buffer that cannot hold what they asked for.
    pub const fn storage_bytes(self, elements: usize) -> Option<usize> {
        let block = self.block_size();
        if !elements.is_multiple_of(block) {
            return None;
        }
        Some(elements / block * self.block_bytes())
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I8 => "i8",
            Self::I32 => "i32",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q2_K => "q2_K",
            Self::Q3_K => "q3_K",
            Self::Q4_K => "q4_K",
            Self::Q5_K => "q5_K",
            Self::Q6_K => "q6_K",
            Self::Q8_K => "q8_K",
        };
        f.write_str(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unquantized_types_are_one_element_per_block() {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8, DType::I32] {
            assert_eq!(dtype.block_size(), 1);
            assert!(!dtype.is_quantized());
        }
    }

    #[test]
    fn storage_bytes_matches_the_plain_element_size_when_unquantized() {
        assert_eq!(DType::F32.storage_bytes(10), Some(40));
        assert_eq!(DType::F16.storage_bytes(10), Some(20));
        assert_eq!(DType::I8.storage_bytes(10), Some(10));
    }

    #[test]
    fn storage_bytes_counts_whole_blocks_when_quantized() {
        // 64 elements = 2 blocks of 32; Q4_0 blocks are 18 bytes each.
        assert_eq!(DType::Q4_0.storage_bytes(64), Some(36));
        assert_eq!(DType::Q8_0.storage_bytes(32), Some(34));
    }

    #[test]
    fn a_partial_quantized_block_is_not_representable() {
        assert_eq!(DType::Q4_0.storage_bytes(33), None);
        assert_eq!(DType::Q4_0.storage_bytes(0), Some(0));
    }

    #[test]
    fn quantized_types_report_themselves_as_such_and_are_not_float() {
        for dtype in [DType::Q4_0, DType::Q4_1, DType::Q5_0, DType::Q5_1, DType::Q8_0] {
            assert!(dtype.is_quantized());
            assert!(!dtype.is_float());
            assert_eq!(dtype.block_size(), 32);
        }
    }

    #[test]
    fn k_quant_types_are_256_element_superblocks_and_not_float() {
        for dtype in [DType::Q2_K, DType::Q3_K, DType::Q4_K, DType::Q5_K, DType::Q6_K, DType::Q8_K] {
            assert!(dtype.is_quantized());
            assert!(!dtype.is_float());
            assert_eq!(dtype.block_size(), 256);
        }
    }

    #[test]
    fn k_quant_superblock_byte_sizes_match_ggml_static_asserts() {
        // These are the sizeof(block_qX_K) values static-asserted in ggml's
        // ggml-common.h with QK_K = 256, K_SCALE_SIZE = 12.
        assert_eq!(DType::Q2_K.block_bytes(), 84);
        assert_eq!(DType::Q3_K.block_bytes(), 110);
        assert_eq!(DType::Q4_K.block_bytes(), 144);
        assert_eq!(DType::Q5_K.block_bytes(), 176);
        assert_eq!(DType::Q6_K.block_bytes(), 210);
        assert_eq!(DType::Q8_K.block_bytes(), 292);
        // One superblock of Q4_K holds 256 elements in 144 bytes.
        assert_eq!(DType::Q4_K.storage_bytes(256), Some(144));
        assert_eq!(DType::Q4_K.storage_bytes(512), Some(288));
        // Not a whole superblock -> not representable.
        assert_eq!(DType::Q4_K.storage_bytes(128), None);
    }
}
