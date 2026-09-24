//! Ported from MuPDF `source/fitz/filter-fax.c` (commit 1ca9d1788, AGPL-3.0,
//! © Artifex Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the algorithm and numeric behaviour follow MuPDF; the
//! code is re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md
//! ("PDF & document-extraction references").
//!
//! # `CCITTFaxDecode` (ITU-T T.4 / T.6)
//!
//! The Group 3/4 fax codecs, as PDF uses them for bilevel scans. This is the
//! filter that stands between a 1980s scanned report and a blank page: a
//! 600-dpi A4 scan is ~35 megapixels of 1-bit data, and CCITT is how it fits
//! in 40 kB.
//!
//! ## The three modes, selected by `/K`
//!
//! | `K` | Mode | What a row is coded against |
//! |---|---|---|
//! | `< 0` | **G4**, pure 2-D | the row above (T.6) |
//! | `0` | **G3 1-D** | run lengths only (T.4 one-dimensional) |
//! | `> 0` | **G3 mixed** | a leading bit per row picks 1-D or 2-D |
//!
//! ## Why two decode tables and not one
//!
//! White and black runs use *different* Huffman tables -- the codes are
//! tuned to the statistics of scanned text, where white runs are long and
//! black runs short. Decoding a black run with the white table yields a
//! plausible-looking wrong number, not an error, which is why
//! [`Faxd::color`] tracking is load-bearing rather than bookkeeping.
//!
//! ## What is NOT implemented, and says so
//!
//! `UNCOMPRESSED` (the T.4 extension escape) is rejected with a format
//! error, exactly as MuPDF does -- it is vanishingly rare and silently
//! mis-decoding it would be worse than failing.

use super::error::{Error, Result};
use super::filter_fax_tables::{
    CfdNode, BLACK_INITIAL_BITS, CF_2D_DECODE, CF_BLACK_DECODE, CF_WHITE_DECODE, CLZ, ERROR, H, LM,
    MASK, P, RM, TWO_D_INITIAL_BITS, UNCOMPRESSED, V0, VL1, VL2, VL3, VR1, VR2, VR3,
    WHITE_INITIAL_BITS,
};

/// Decoder parameters, from the PDF `/DecodeParms` dictionary (§7.4.6).
#[derive(Clone, Copy, Debug)]
pub struct FaxParams {
    /// `/K`: `<0` pure 2-D (G4), `0` pure 1-D, `>0` mixed.
    pub k: i32,
    /// `/EndOfLine`: whether rows are separated by an EOL code.
    pub end_of_line: bool,
    /// `/EncodedByteAlign`: whether each row starts on a byte boundary.
    pub encoded_byte_align: bool,
    /// `/Columns`: pixels per row. PDF default 1728.
    pub columns: u32,
    /// `/Rows`: rows expected, or 0 for "until the data runs out".
    pub rows: u32,
    /// `/BlackIs1`: when false (the PDF default), 0 bits are black.
    pub black_is_1: bool,
}

impl Default for FaxParams {
    fn default() -> Self {
        Self {
            k: 0,
            end_of_line: false,
            encoded_byte_align: false,
            columns: 1728,
            rows: 0,
            black_is_1: false,
        }
    }
}

/// Where the row decoder is in MuPDF's state machine (`filter-fax.c:317`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// Initial, optionally waiting for a leading EOL.
    Init,
    /// Neutral -- waiting for any code.
    Normal,
    /// Got a 1-D makeup code, waiting for the terminating code.
    Makeup,
    /// In horizontal mode, part 1 / part 2.
    H1,
    H2,
    /// All done.
    Done,
}

/// The decoder state -- MuPDF's `fz_faxd`.
struct Faxd<'a> {
    src: &'a [u8],
    pos: usize,

    k: i32,
    end_of_line: bool,
    encoded_byte_align: bool,
    columns: usize,
    rows: usize,

    /// Bit accumulator, MSB-aligned, exactly as MuPDF keeps it.
    word: u32,
    /// How many of the 32 bits are spent.
    bidx: i32,

    stage: Stage,
    /// Current pixel position in the row, `-1` before the row starts.
    a: i32,
    /// Current colour: `false` white, `true` black.
    color: bool,
    /// 1 or 2 -- the dimensionality of the row being decoded.
    dim: i32,
    /// Consecutive EOL codes seen (six in a row is RTC, end of block).
    eolc: i32,

    stride: usize,
    /// The row above, for 2-D coding.
    reference: Vec<u8>,
    /// The row being built.
    current: Vec<u8>,
}

impl<'a> Faxd<'a> {
    fn new(src: &'a [u8], p: &FaxParams) -> Self {
        let columns = p.columns.max(1) as usize;
        let stride = columns.div_ceil(8);
        Self {
            src,
            pos: 0,
            k: p.k,
            end_of_line: p.end_of_line,
            encoded_byte_align: p.encoded_byte_align,
            columns,
            rows: p.rows as usize,
            word: 0,
            bidx: 32,
            stage: Stage::Init,
            a: -1,
            color: false,
            // K < 0 is pure 2-D; otherwise a row starts 1-D until told
            // otherwise by the leading mode bit.
            dim: if p.k < 0 { 2 } else { 1 },
            eolc: 0,
            stride,
            reference: vec![0; stride + 1],
            current: vec![0; stride + 1],
        }
    }

    fn read_byte(&mut self) -> Option<u8> {
        let b = self.src.get(self.pos).copied();
        if b.is_some() {
            self.pos += 1;
        }
        b
    }

    /// MuPDF `eat_bits`.
    fn eat_bits(&mut self, nbits: i32) {
        self.word = if nbits >= 32 { 0 } else { self.word << nbits };
        self.bidx += nbits;
    }

    /// MuPDF `fill_bits`. The longest code is 13 bits, so never read past
    /// what is needed -- over-reading the end of the stream is how a decoder
    /// consumes the *next* image's bytes.
    fn fill_bits(&mut self) -> bool {
        while self.bidx > (32 - 13) {
            match self.read_byte() {
                Some(c) => {
                    self.bidx -= 8;
                    self.word |= u32::from(c) << self.bidx;
                }
                None => return false,
            }
        }
        true
    }

    /// MuPDF `get_code` -- the two-level table walk.
    fn get_code(&mut self, table: &[CfdNode], initial_bits: u32) -> i16 {
        let word = self.word;
        let mut tidx = (word >> (32 - initial_bits)) as usize;
        let mut val = table[tidx].val;
        let mut nbits = i32::from(table[tidx].nbits);

        if nbits > initial_bits as i32 {
            let wordmask = (1u32 << (32 - initial_bits)) - 1;
            tidx = (val as usize) + ((word & wordmask) >> (32 - nbits)) as usize;
            val = table[tidx].val;
            nbits = initial_bits as i32 + i32::from(table[tidx].nbits);
        }

        self.eat_bits(nbits);
        val
    }
}

/// MuPDF `getbit`.
fn getbit(line: &[u8], x: usize) -> bool {
    (line[x >> 3] >> (7 - (x & 7))) & 1 != 0
}

/// MuPDF `setbits` -- fill `[x0, x1)` with 1 bits.
fn setbits(line: &mut [u8], x0: usize, x1: usize) {
    if x1 <= x0 {
        return;
    }
    let (a0, a1) = (x0 >> 3, x1 >> 3);
    let (b0, b1) = (x0 & 7, x1 & 7);
    if a0 == a1 {
        if b1 != 0 {
            line[a0] |= LM[b0] & RM[b1];
        }
    } else {
        line[a0] |= LM[b0];
        for byte in line.iter_mut().take(a1).skip(a0 + 1) {
            *byte = 0xFF;
        }
        if b1 != 0 {
            line[a1] |= RM[b1];
        }
    }
}

/// MuPDF `find_changing` -- the next colour transition at or after `x`.
fn find_changing(line: &[u8], x: i32, w: usize) -> usize {
    let w_i = w as i32;
    let (mut x, m) = if x < 0 {
        (0i32, 0xFFu8)
    } else {
        (x, MASK[(x & 7) as usize])
    };
    let big_w = (w_i >> 3) as usize;
    x >>= 3;
    let mut xi = x as usize;
    let mut a = line[xi];
    let mut b = (a ^ (a >> 1)) & m;

    if xi >= big_w {
        let r = (xi << 3) + CLZ[b as usize] as usize;
        return r.min(w);
    }
    loop {
        if b != 0 {
            return (xi << 3) + CLZ[b as usize] as usize;
        }
        xi += 1;
        if xi >= big_w {
            break;
        }
        let carry = a & 1;
        a = line[xi];
        b = (carry << 7) ^ a ^ (a >> 1);
    }
    // Less than a byte to go; if there are no stray bits we are done.
    if (xi << 3) == w {
        return w;
    }
    let carry = a & 1;
    a = line[xi];
    b = (carry << 7) ^ a ^ (a >> 1);
    let r = (xi << 3) + CLZ[b as usize] as usize;
    r.min(w)
}

/// MuPDF `find_changing_color`.
fn find_changing_color(line: &[u8], x: i32, w: usize, color: bool) -> usize {
    if x >= w as i32 {
        return w;
    }
    let start = if x > 0 || !color { x } else { -1 };
    let mut x = find_changing(line, start, w);
    if x < w && getbit(line, x) != color {
        x = find_changing(line, x as i32, w);
    }
    x
}

impl Faxd<'_> {
    /// MuPDF `dec1d` -- one 1-D run-length code.
    fn dec1d(&mut self) -> Result<()> {
        if self.a == -1 {
            self.a = 0;
        }
        let code = if self.color {
            self.get_code(&CF_BLACK_DECODE, BLACK_INITIAL_BITS)
        } else {
            self.get_code(&CF_WHITE_DECODE, WHITE_INITIAL_BITS)
        };

        if code == UNCOMPRESSED {
            return Err(Error::format("uncompressed data in faxd"));
        }
        if code < 0 {
            return Err(Error::format("negative code in 1d faxd"));
        }
        if self.a + i32::from(code) > self.columns as i32 {
            return Err(Error::format("overflow in 1d faxd"));
        }

        if self.color {
            let (a, c) = (self.a as usize, (self.a + i32::from(code)) as usize);
            setbits(&mut self.current, a, c);
        }
        self.a += i32::from(code);

        // A code below 64 is a terminating code: the run ends and the colour
        // flips. 64 and above is a makeup code, and a terminating code of the
        // SAME colour must follow -- which is why the colour does not flip.
        if code < 64 {
            self.color = !self.color;
            self.stage = Stage::Normal;
        } else {
            self.stage = Stage::Makeup;
        }
        Ok(())
    }

    /// MuPDF `dec2d` -- one 2-D mode code, or a horizontal-mode run.
    fn dec2d(&mut self) -> Result<()> {
        if self.stage == Stage::H1 || self.stage == Stage::H2 {
            if self.a == -1 {
                self.a = 0;
            }
            let code = if self.color {
                self.get_code(&CF_BLACK_DECODE, BLACK_INITIAL_BITS)
            } else {
                self.get_code(&CF_WHITE_DECODE, WHITE_INITIAL_BITS)
            };
            if code == UNCOMPRESSED {
                return Err(Error::format("uncompressed data in faxd"));
            }
            if code < 0 {
                return Err(Error::format("negative code in 2d faxd"));
            }
            if self.a + i32::from(code) > self.columns as i32 {
                return Err(Error::format("overflow in 2d faxd"));
            }
            if self.color {
                let (a, c) = (self.a as usize, (self.a + i32::from(code)) as usize);
                setbits(&mut self.current, a, c);
            }
            self.a += i32::from(code);
            if code < 64 {
                self.color = !self.color;
                self.stage = if self.stage == Stage::H1 {
                    Stage::H2
                } else {
                    Stage::Normal
                };
            }
            return Ok(());
        }

        let code = self.get_code(&CF_2D_DECODE, TWO_D_INITIAL_BITS);
        let w = self.columns;
        // Vertical modes are all "b1 offset by n, then flip"; pass and
        // horizontal are the two special cases.
        let vertical = |d: i32| -> Option<i32> {
            match code {
                V0 => Some(0),
                VR1 => Some(1),
                VR2 => Some(2),
                VR3 => Some(3),
                VL1 => Some(-1),
                VL2 => Some(-2),
                VL3 => Some(-3),
                _ => None,
            }
            .map(|n| n + d)
        };

        match code {
            H => self.stage = Stage::H1,
            P => {
                let b1 = find_changing_color(&self.reference, self.a, w, !self.color);
                let b2 = if b1 >= w {
                    w
                } else {
                    find_changing(&self.reference, b1 as i32, w)
                };
                if self.color {
                    setbits(&mut self.current, self.a.max(0) as usize, b2);
                }
                self.a = b2 as i32;
            }
            _ if vertical(0).is_some() => {
                let delta = vertical(0).expect("matched above");
                let base = find_changing_color(&self.reference, self.a, w, !self.color) as i32;
                let mut b1 = base + delta;
                if b1 >= w as i32 {
                    b1 = w as i32;
                }
                if b1 < 0 {
                    b1 = 0;
                }
                if self.color {
                    setbits(&mut self.current, self.a.max(0) as usize, b1 as usize);
                }
                self.a = b1;
                self.color = !self.color;
            }
            UNCOMPRESSED => return Err(Error::format("uncompressed data in faxd")),
            ERROR => return Err(Error::format("invalid code in 2d faxd")),
            _ => return Err(Error::format("invalid code in 2d faxd")),
        }
        Ok(())
    }
}

/// Decode a `CCITTFaxDecode` stream to packed 1-bit rows, MSB first.
///
/// Returns `(bits, rows)` where `bits` is `rows * ceil(columns/8)` bytes. A
/// set bit means **black** regardless of `/BlackIs1`; the caller expands to
/// 8-bit samples (see [`decode_to_gray`]).
///
/// # Restructured from MuPDF, deliberately
///
/// MuPDF decodes into a fixed 4 kB buffer and resumes, so its control flow is
/// a state machine with `goto loop/eol/rtc`. This crate wants a whole image,
/// so the same state machine is driven by a row loop instead. The decode
/// *steps* are unchanged -- `dec1d`, `dec2d`, `get_code` and the bit
/// accounting are ported line for line -- only the resumption scaffolding is
/// gone.
///
/// # Errors
/// A malformed stream: bad codes, row overflow, or a missing initial EOL when
/// `/EndOfLine` is set.
pub fn decode(src: &[u8], params: &FaxParams) -> Result<(Vec<u8>, usize)> {
    let mut fax = Faxd::new(src, params);
    let stride = fax.stride;
    let mut out: Vec<u8> = Vec::new();
    let mut rows_done = 0usize;

    // An initial EOL is required when /EndOfLine is set. MuPDF warns and
    // hunts for it rather than failing immediately, because encoders get this
    // wrong and the data after it is usually fine.
    if fax.end_of_line {
        fax.fill_bits();
        if (fax.word >> (32 - 12)) != 1 {
            while fax.fill_bits() && (fax.word >> (32 - 12)) != 1 {
                fax.eat_bits(1);
            }
        }
        if (fax.word >> (32 - 12)) != 1 {
            return Err(Error::format("initial EOL not found in faxd"));
        }
    }
    fax.stage = Stage::Normal;

    // A row is finished when `a` reaches `columns`. The outer bound stops a
    // corrupt stream spinning: every iteration either consumes bits or ends
    // the row, and `rows` (when given) caps the total.
    let row_cap = if fax.rows > 0 {
        fax.rows
    } else {
        // No /Rows: bound by what the data could possibly encode. One row
        // needs at least one bit, so this can never truncate a real image.
        src.len() * 8 + 1
    };

    'rows: while rows_done < row_cap {
        fax.a = -1;
        fax.color = false;
        fax.stage = Stage::Normal;
        for b in fax.current.iter_mut() {
            *b = 0;
        }
        if params.k < 0 {
            fax.dim = 2;
        }

        // Decode codes until the row fills.
        loop {
            if !fax.fill_bits() && fax.bidx > 31 {
                // Out of data. A partial row still counts if anything was
                // decoded into it.
                if fax.a > 0 {
                    break;
                }
                break 'rows;
            }

            let peek12 = fax.word >> (32 - 12);
            if peek12 == 0 {
                // Fill bits before an EOL.
                fax.eat_bits(1);
                continue;
            }
            // MuPDF's if/else-if chain (filter-fax.c:597). It is a CHAIN, not
            // a sequence: after consuming an EOL the decoder must go straight
            // to the end-of-row check below. Falling through into dec1d/dec2d
            // instead would reset `eolc` -- destroying the very counter the
            // end-of-block test reads -- and then try to decode the bits after
            // the EOL as a run code.
            if peek12 == 1 {
                fax.eat_bits(12);
                fax.eolc += 1;
                if fax.k > 0 {
                    if fax.a == -1 {
                        fax.a = 0;
                    }
                    fax.dim = if (fax.word >> (32 - 1)) == 1 { 1 } else { 2 };
                    fax.eat_bits(1);
                }
            } else if fax.k > 0 && fax.a == -1 {
                // Mixed mode: a leading bit per row picks 1-D or 2-D.
                fax.a = 0;
                fax.dim = if (fax.word >> (32 - 1)) == 1 { 1 } else { 2 };
                fax.eat_bits(1);
            } else if fax.dim == 1 {
                fax.eolc = 0;
                fax.dec1d()?;
            } else {
                fax.eolc = 0;
                fax.dec2d()?;
            }

            // MuPDF's end-of-row check (filter-fax.c:676), verbatim in
            // structure because the thresholds are not interchangeable:
            //
            //   - no check after a makeup code, nor mid-H-code, because the
            //     row is not in a resolved state there;
            //   - RTC is TWO EOLs for G4 (K<0) and SIX for G3. Using 6 for
            //     G4 means the end-of-block is never recognised and the
            //     decoder runs past the last row into whatever follows --
            //     which is exactly how a 4960-row scan decoded as 4999.
            if matches!(fax.stage, Stage::Makeup | Stage::H1 | Stage::H2) {
                continue;
            }
            if fax.eolc != 0 || fax.a >= fax.columns as i32 {
                if fax.a > 0 {
                    break;
                }
                if fax.eolc == if fax.k < 0 { 2 } else { 6 } {
                    break 'rows;
                }
            }
        }

        out.extend_from_slice(&fax.current[..stride]);
        rows_done += 1;
        std::mem::swap(&mut fax.reference, &mut fax.current);

        if fax.encoded_byte_align {
            // Drop to the next byte boundary.
            let used = fax.bidx & 7;
            if used != 0 {
                fax.eat_bits(8 - used);
            }
        }
        if fax.pos >= src.len() && fax.bidx > 31 {
            break;
        }
    }

    fax.stage = Stage::Done;
    Ok((out, rows_done))
}

/// [`decode`], expanded to one 8-bit grayscale sample per pixel.
///
/// `params.black_is_1` is deliberately **not** consulted: it describes the
/// packed *sample* convention, not which ink is dark. [`decode`] always sets a
/// bit for a black pixel, so black maps to 0 and white to 255 either way. A
/// caller that wants PDF-convention packed samples rather than gray wants
/// [`decode`] plus that inversion, which is what `page_image` does.
///
/// # Errors
/// As [`decode`].
pub fn decode_to_gray(src: &[u8], params: &FaxParams) -> Result<(Vec<u8>, usize, usize)> {
    let (bits, rows) = decode(src, params)?;
    let w = params.columns.max(1) as usize;
    let stride = w.div_ceil(8);
    let mut gray = vec![0u8; w * rows];
    for y in 0..rows {
        let row = &bits[y * stride..(y + 1) * stride];
        for x in 0..w {
            let black = getbit(row, x);
            gray[y * w + x] = if black { 0 } else { 255 };
        }
    }
    Ok((gray, w, rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `FaxParams` for a G4 (pure 2-D) image of the given size.
    fn g4(columns: u32, rows: u32) -> FaxParams {
        FaxParams {
            k: -1,
            columns,
            rows,
            ..FaxParams::default()
        }
    }

    /// An all-white row in G4 is a single V0 against an imaginary white
    /// reference line: the changing element is at `columns`, so one V0 code
    /// (`001` in the 2-D table) fills the row.
    #[test]
    fn a_single_v0_fills_an_all_white_row() {
        // V0 = 0b1 (one bit). Four rows of it, then padding.
        let src = [0b1111_0000u8];
        let (bits, rows) = decode(&src, &g4(8, 4)).expect("decode");
        assert_eq!(rows, 4);
        assert_eq!(bits, vec![0u8; 4], "white rows carry no set bits");
    }

    /// `decode_to_gray` honours the PDF default, where a **clear** bit is
    /// white. Getting this backwards inverts every scan in the corpus, which
    /// is visually obvious but silently wrong if only the packed form is
    /// tested.
    #[test]
    fn white_rows_expand_to_255() {
        let src = [0b1111_0000u8];
        let (gray, w, rows) = decode_to_gray(&src, &g4(8, 4)).expect("decode");
        assert_eq!((w, rows), (8, 4));
        assert!(gray.iter().all(|&g| g == 255));
    }

    /// `setbits` fills a half-open range, including across byte boundaries.
    #[test]
    fn setbits_fills_the_half_open_range() {
        let mut line = [0u8; 3];
        setbits(&mut line, 3, 13);
        // bits 3..13 set: 0b0001_1111, 0b1111_1000, 0
        assert_eq!(line, [0b0001_1111, 0b1111_1000, 0]);

        let mut one = [0u8; 1];
        setbits(&mut one, 2, 5);
        assert_eq!(one, [0b0011_1000]);

        let mut none = [0u8; 1];
        setbits(&mut none, 4, 4);
        assert_eq!(none, [0], "an empty range writes nothing");
    }

    /// `find_changing` reports the next colour transition, and `w` when
    /// there is none.
    #[test]
    fn find_changing_locates_the_transition() {
        // 0b0000_1111: white runs 0..4, black 4..8.
        let line = [0b0000_1111u8, 0x00];
        assert_eq!(find_changing(&line, -1, 16), 4);
        assert_eq!(find_changing(&line, 4, 16), 8);
        let flat = [0u8, 0u8];
        assert_eq!(find_changing(&flat, -1, 16), 16, "no transition -> w");
    }

    /// A malformed stream errors rather than looping or panicking. The row
    /// cap exists for exactly this: without it a stream that never advances
    /// spins forever.
    #[test]
    fn garbage_terminates_rather_than_spinning() {
        let src = [0xFFu8; 64];
        let out = decode(&src, &g4(1728, 0));
        // Either an error or a bounded result -- never a hang.
        if let Ok((_, rows)) = out {
            assert!(rows < 10_000, "bounded even with no /Rows");
        }
    }

    /// Empty input yields no rows rather than an error: a zero-length image
    /// is degenerate, not malformed.
    #[test]
    fn empty_input_decodes_to_no_rows() {
        let (bits, rows) = decode(&[], &g4(64, 0)).expect("decode");
        assert_eq!(rows, 0);
        assert!(bits.is_empty());
    }

    /// RTC is **two** EOLs for G4 and six for G3, and they are not
    /// interchangeable.
    ///
    /// Using six for G4 means the end-of-block is never recognised, the
    /// decoder runs past the last row into whatever follows, and a 4960-row
    /// scan decodes as 4999 rows of increasing garbage. That was a real bug
    /// in this port, caught by comparing against MuPDF; this pins it.
    #[test]
    fn g4_end_of_block_is_two_eols_not_six() {
        // EOL is 000000000001 (12 bits). Two of them, from a clean start.
        // 0000 0000 0001 0000 0000 0001 -> 0x00,0x10,0x01
        let src = [0x00, 0x10, 0x01, 0x00, 0x00, 0x00];
        let (_, rows) = decode(&src, &g4(1728, 0)).expect("decode");
        assert_eq!(rows, 0, "EOFB before any row data yields no rows");
    }
}
