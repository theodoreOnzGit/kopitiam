//! Ported from Tesseract `src/ccutil/unicharcompress.cpp` +
//! `src/ccutil/unicharcompress.h` (commit db0ec62, Apache-2.0, © 2015 Google
//! Inc., Author: Ray Smith), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the [`RecodedCharID`] layout, the serialised recoder
//! format, and the encode/decode/prefix bookkeeping follow Tesseract exactly;
//! the code is re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is — the CJK recoder (the load-bearing bit for chi_sim/jpn)
//!
//! [`UnicharCompress`] "compresses" a unicharset so a neural classifier learns a
//! small alphabet instead of thousands of CJK classes: each unichar id maps to a
//! short **sequence of small codes** ([`RecodedCharID`], up to
//! [`RecodedCharID::MAX_CODE_LEN`] units) and back. This is *why* chi_sim and jpn
//! work — the LSTM emits recoded units (Jamo for Hangul, radical/stroke indices
//! for Han, unicode-sequence codes for Indic, ligature dissection for the rest),
//! and the recoder reassembles them into the original unichar id. A multi-unit
//! CJK codepoint is therefore several network time-steps that
//! [`UnicharCompress::decode_unichar`] folds back into one id.
//!
//! # Read path (deserialise)
//!
//! Only `encoder_` — the id→code table — is serialised; everything else (the
//! reverse `decoder_`, the code range, and the prefix/next/final code lists used
//! by beam search) is **recomputed on load**. [`UnicharCompress::deserialize`]
//! reads the `TESSDATA_LSTM_RECODER` component (a length-prefixed vector of
//! [`RecodedCharID`], each `int8 self_normalized · uint32 length ·
//! int32[length] code`) and then rebuilds the decoder, exactly reproducing the
//! recode indirection that makes multi-unit codepoints reassemble correctly.
//!
//! # Deferred (training / write path)
//!
//! `ComputeEncoding` (which *builds* the recoder at training time from a
//! unicharset + `radical-stroke.txt`), `Serialize`, and `GetEncodingAsString`
//! are the write/training path and are not ported — KOPITIAM loads a recoder, it
//! does not train one. The static [`UnicharCompress::decompose_hangul`] and the
//! Hangul/Han constants are ported (they are pure and testable) so the recode
//! scheme is documented in place. [`UnicharCompress::setup_pass_through`] and
//! [`UnicharCompress::setup_direct`] — the non-training encoder builders shared
//! with the read path — are ported.

use crate::error::{Error, Result};
use crate::serialis::TFile;
use crate::unichar::{INVALID_UNICHAR_ID, UnicharId};
use crate::unicharset::Unicharset;
use std::collections::HashMap;

/// The code for a single recoded unichar id: up to
/// [`RecodedCharID::MAX_CODE_LEN`] small integer units.
///
/// Tesseract: `class RecodedCharID` (unicharcompress.h:34). Equality and hashing
/// consider only `length` and `code[0..length]` (matching `operator==` and
/// `RecodedCharIDHash`, unicharcompress.h:89/101); `self_normalized` and the
/// unused tail of `code` are ignored, so this is safe as a map key.
#[derive(Clone, Copy, Debug)]
pub struct RecodedCharID {
    /// True (1) if this is the master entry for indices mapping to the same code.
    /// `int8_t` for serialisation; not part of identity.
    self_normalized: i8,
    /// Number of units in use in `code`.
    length: u32,
    /// The re-encoded units.
    code: [i32; Self::MAX_CODE_LEN],
}

impl RecodedCharID {
    /// The maximum length of a code. Tesseract: `kMaxCodeLen` (unicharcompress.h:37).
    pub const MAX_CODE_LEN: usize = 9;

    /// An empty code. Tesseract: `RecodedCharID()` (unicharcompress.h:39).
    pub fn new() -> Self {
        RecodedCharID {
            self_normalized: 1,
            length: 0,
            code: [0; Self::MAX_CODE_LEN],
        }
    }

    /// Shorten the code to `length` units. Tesseract: `Truncate` (unicharcompress.h:42).
    pub fn truncate(&mut self, length: u32) {
        self.length = length;
    }

    /// Set the unit at `index`, growing the length if needed.
    ///
    /// Tesseract: `RecodedCharID::Set` (unicharcompress.h:46). Out-of-range
    /// indices are ignored, as in the C.
    pub fn set(&mut self, index: usize, value: i32) {
        if index >= Self::MAX_CODE_LEN {
            return;
        }
        self.code[index] = value;
        if (self.length as usize) <= index {
            self.length = index as u32 + 1;
        }
    }

    /// Set a length-3 code (all Hangul and Han codes are length 3).
    ///
    /// Tesseract: `RecodedCharID::Set3` (unicharcompress.h:57).
    pub fn set3(&mut self, code0: i32, code1: i32, code2: i32) {
        self.length = 3;
        self.code[0] = code0;
        self.code[1] = code1;
        self.code[2] = code2;
    }

    /// Whether the code is empty. Tesseract: `empty` (unicharcompress.h:63).
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Whether this code is the "self-normalising" master entry for indices that
    /// map to the same code (the deserialised `self_normalized_` flag).
    ///
    /// Tesseract: `RecodedCharID::self_normalized_` (unicharcompress.h:114).
    pub fn is_self_normalized(&self) -> bool {
        self.self_normalized != 0
    }

    /// The number of units in use. Tesseract: `length` (unicharcompress.h:67).
    pub fn length(&self) -> u32 {
        self.length
    }

    /// The unit at `index`. Tesseract: `operator()` (unicharcompress.h:70).
    pub fn get(&self, index: usize) -> i32 {
        self.code[index]
    }

    /// Read a code. Tesseract: `RecodedCharID::DeSerialize` (unicharcompress.h:80):
    /// `int8 self_normalized · uint32 length · int32[length] code`.
    pub fn deserialize(fp: &mut TFile<'_>) -> Result<RecodedCharID> {
        let self_normalized = fp.read_u8()? as i8;
        let length = fp.read_u32()?;
        if length as usize > Self::MAX_CODE_LEN {
            return Err(Error::format(format!(
                "recoder: code length {length} exceeds {}",
                Self::MAX_CODE_LEN
            )));
        }
        let mut code = [0i32; Self::MAX_CODE_LEN];
        for slot in code.iter_mut().take(length as usize) {
            *slot = fp.read_i32()?;
        }
        Ok(RecodedCharID {
            self_normalized,
            length,
            code,
        })
    }
}

impl Default for RecodedCharID {
    fn default() -> Self {
        RecodedCharID::new()
    }
}

impl PartialEq for RecodedCharID {
    /// Tesseract: `RecodedCharID::operator==` (unicharcompress.h:89) — length and
    /// used units only.
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }
        self.code[..self.length as usize] == other.code[..other.length as usize]
    }
}

impl Eq for RecodedCharID {}

impl std::hash::Hash for RecodedCharID {
    /// Hashes only `length` and the used units, consistent with [`PartialEq`].
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.length.hash(state);
        self.code[..self.length as usize].hash(state);
    }
}

/// A "compression" of a unicharset: each unichar id ↔ a sequence of small codes.
///
/// Tesseract: `class UnicharCompress` (unicharcompress.h:149).
#[derive(Default)]
pub struct UnicharCompress {
    /// id → code. The only serialised part; the rest is computed on load.
    encoder: Vec<RecodedCharID>,
    /// code → id (reverse of `encoder`).
    decoder: HashMap<RecodedCharID, i32>,
    /// Whether an index is a valid single or start code.
    is_valid_start: Vec<bool>,
    /// Prefix code → valid non-final next codes.
    next_codes: HashMap<RecodedCharID, Vec<i32>>,
    /// Prefix code → valid final codes.
    final_codes: HashMap<RecodedCharID, Vec<i32>>,
    /// 1 + the maximum unit value used anywhere in `encoder`.
    code_range: i32,
}

impl UnicharCompress {
    /// The 1st Hangul unicode. Tesseract: `kFirstHangul` (unicharcompress.h:157).
    pub const FIRST_HANGUL: i32 = 0xac00;
    /// The number of Hangul unicodes. Tesseract: `kNumHangul` (unicharcompress.h:159).
    pub const NUM_HANGUL: i32 = 11172;
    /// Leading-consonant Jamo count. Tesseract: `kLCount` (unicharcompress.h:162).
    pub const L_COUNT: i32 = 19;
    /// Vowel Jamo count. Tesseract: `kVCount` (unicharcompress.h:163).
    pub const V_COUNT: i32 = 21;
    /// Trailing-consonant Jamo count. Tesseract: `kTCount` (unicharcompress.h:164).
    pub const T_COUNT: i32 = 28;

    /// An empty recoder.
    pub fn new() -> Self {
        UnicharCompress::default()
    }

    /// 1 + the maximum unit value used by any code. Tesseract: `code_range`
    /// (unicharcompress.h:181).
    pub fn code_range(&self) -> i32 {
        self.code_range
    }

    /// The number of encoded unichar ids.
    pub fn size(&self) -> usize {
        self.encoder.len()
    }

    /// Encode `unichar_id`, returning its code, or [`None`] if out of range.
    ///
    /// Tesseract: `UnicharCompress::EncodeUnichar` (unicharcompress.cpp:295)
    /// (which returns the code length; here [`None`] is the C's return-0 case).
    pub fn encode_unichar(&self, unichar_id: usize) -> Option<RecodedCharID> {
        self.encoder.get(unichar_id).copied()
    }

    /// Decode `code` back to its unichar id, or [`INVALID_UNICHAR_ID`].
    ///
    /// Tesseract: `UnicharCompress::DecodeUnichar` (unicharcompress.cpp:305).
    pub fn decode_unichar(&self, code: &RecodedCharID) -> UnicharId {
        let len = code.length();
        if len == 0 || len as usize > RecodedCharID::MAX_CODE_LEN {
            return INVALID_UNICHAR_ID;
        }
        self.decoder
            .get(code)
            .copied()
            .unwrap_or(INVALID_UNICHAR_ID)
    }

    /// Whether `code` is a valid start-or-single unit.
    ///
    /// Tesseract: `IsValidFirstCode` (unicharcompress.h:192).
    pub fn is_valid_first_code(&self, code: i32) -> bool {
        code >= 0
            && (code as usize) < self.is_valid_start.len()
            && self.is_valid_start[code as usize]
    }

    /// Valid non-final next units for the given prefix, if any.
    ///
    /// Tesseract: `GetNextCodes` (unicharcompress.h:197).
    pub fn get_next_codes(&self, prefix: &RecodedCharID) -> Option<&[i32]> {
        self.next_codes.get(prefix).map(|v| v.as_slice())
    }

    /// Valid final units for the given prefix, if any.
    ///
    /// Tesseract: `GetFinalCodes` (unicharcompress.h:203).
    pub fn get_final_codes(&self, prefix: &RecodedCharID) -> Option<&[i32]> {
        self.final_codes.get(prefix).map(|v| v.as_slice())
    }

    /// Read a recoder: the `encoder_` vector, then recompute range + decoder.
    ///
    /// Tesseract: `UnicharCompress::DeSerialize` (unicharcompress.cpp:323) over
    /// `TFile::DeSerialize(std::vector<RecodedCharID>&)` (serialis.h:94).
    pub fn deserialize(&mut self, fp: &mut TFile<'_>) -> Result<()> {
        let size = fp.read_u32()?;
        self.encoder.clear();
        if size == 0 {
            // nothing
        } else if size > 50_000_000 {
            return Err(Error::limit(format!(
                "recoder: {size} entries exceeds the 50000000 guard"
            )));
        } else {
            self.encoder.reserve(size as usize);
            for _ in 0..size {
                self.encoder.push(RecodedCharID::deserialize(fp)?);
            }
        }
        self.compute_code_range();
        self.setup_decoder();
        Ok(())
    }

    /// Set up an encoder directly from an id→code vector.
    ///
    /// Tesseract: `UnicharCompress::SetupDirect` (unicharcompress.cpp:247).
    pub fn setup_direct(&mut self, codes: Vec<RecodedCharID>) {
        self.encoder = codes;
        self.compute_code_range();
        self.setup_decoder();
    }

    /// Set up a pass-through encoder that leaves unichars unchanged.
    ///
    /// Tesseract: `UnicharCompress::SetupPassThrough` (unicharcompress.cpp:230).
    pub fn setup_pass_through(&mut self, unicharset: &Unicharset) {
        let mut codes = Vec::with_capacity(unicharset.size() + 1);
        for u in 0..unicharset.size() {
            let mut code = RecodedCharID::new();
            code.set(0, u as i32);
            codes.push(code);
        }
        if !unicharset.has_special_codes() {
            let mut code = RecodedCharID::new();
            code.set(0, unicharset.size() as i32);
            codes.push(code);
        }
        self.setup_direct(codes);
    }

    /// Decompose a Hangul unicode into (leading, vowel, trailing) 0-based Jamo
    /// indices, or [`None`] if outside the Hangul range.
    ///
    /// Tesseract: `UnicharCompress::DecomposeHangul` (unicharcompress.cpp:367).
    pub fn decompose_hangul(unicode: i32) -> Option<(i32, i32, i32)> {
        if unicode < Self::FIRST_HANGUL {
            return None;
        }
        let offset = unicode - Self::FIRST_HANGUL;
        if offset >= Self::NUM_HANGUL {
            return None;
        }
        let n_count = Self::V_COUNT * Self::T_COUNT;
        let leading = offset / n_count;
        let vowel = (offset % n_count) / Self::T_COUNT;
        let trailing = offset % Self::T_COUNT;
        Some((leading, vowel, trailing))
    }

    /// Compute `code_range_` from `encoder_`.
    ///
    /// Tesseract: `UnicharCompress::ComputeCodeRange` (unicharcompress.cpp:383).
    fn compute_code_range(&mut self) {
        let mut code_range = -1;
        for code in &self.encoder {
            for i in 0..code.length() as usize {
                if code.get(i) > code_range {
                    code_range = code.get(i);
                }
            }
        }
        self.code_range = code_range + 1;
    }

    /// Rebuild the decoder, start-code set, and prefix/next/final code lists.
    ///
    /// Tesseract: `UnicharCompress::SetupDecoder` (unicharcompress.cpp:396).
    // The vacant-prefix branch does substantial extra work (it also walks and
    // populates `next_codes`), so the `contains_key`/`insert` pair is not a plain
    // entry-API candidate; kept as-is to mirror the C control flow faithfully.
    #[allow(clippy::map_entry)]
    fn setup_decoder(&mut self) {
        self.cleanup();
        let range = self.code_range.max(0) as usize;
        self.is_valid_start = vec![false; range];
        // Iterate over an immutable snapshot so the maps below can be mutated.
        let encoder = self.encoder.clone();
        for (c, code) in encoder.iter().enumerate() {
            self.decoder.insert(*code, c as i32);
            self.is_valid_start[code.get(0) as usize] = true;
            let mut prefix = *code;
            let mut len = code.length();
            if len == 0 {
                continue;
            }
            len -= 1;
            prefix.truncate(len);
            if !self.final_codes.contains_key(&prefix) {
                self.final_codes
                    .insert(prefix, vec![code.get(len as usize)]);
                while len > 0 {
                    len -= 1;
                    prefix.truncate(len);
                    if let Some(list) = self.next_codes.get_mut(&prefix) {
                        // Reached via multiple code lengths; add if new, then stop.
                        if !list.contains(&code.get(len as usize)) {
                            list.push(code.get(len as usize));
                        }
                        break;
                    } else {
                        self.next_codes.insert(prefix, vec![code.get(len as usize)]);
                    }
                }
            } else {
                let list = self.final_codes.get_mut(&prefix).unwrap();
                if !list.contains(&code.get(len as usize)) {
                    list.push(code.get(len as usize));
                }
            }
        }
    }

    /// Free the computed (non-serialised) state.
    ///
    /// Tesseract: `UnicharCompress::Cleanup` (unicharcompress.cpp:442).
    fn cleanup(&mut self) {
        self.decoder.clear();
        self.is_valid_start.clear();
        self.next_codes.clear();
        self.final_codes.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialise a slice of codes into the `encoder_` on-disk form (native
    /// endian, no swap), matching `RecodedCharID::Serialize` + the vector length
    /// prefix. Used to exercise the deserialise read path hermetically.
    fn serialize_recoder(codes: &[RecodedCharID]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(codes.len() as u32).to_ne_bytes());
        for code in codes {
            buf.push(code.self_normalized as u8);
            buf.extend_from_slice(&code.length.to_ne_bytes());
            for i in 0..code.length() as usize {
                buf.extend_from_slice(&code.get(i).to_ne_bytes());
            }
        }
        buf
    }

    /// A small recoder: ASCII pass-through ids plus two multi-unit CJK-style
    /// codes (length-3, as all Han/Hangul codes are).
    fn sample_codes() -> Vec<RecodedCharID> {
        let mut codes = Vec::new();
        // id 0: space, single unit 0.
        let mut c = RecodedCharID::new();
        c.set(0, 0);
        codes.push(c);
        // id 1,2: single-unit ASCII-like.
        for v in [1, 2] {
            let mut c = RecodedCharID::new();
            c.set(0, v);
            codes.push(c);
        }
        // id 3: multi-unit CJK-style, length 3 (e.g. Han radical/stroke/count).
        let mut c = RecodedCharID::new();
        c.set3(3, 7, 11);
        codes.push(c);
        // id 4: another distinct length-3 code (e.g. Hangul L/V/T).
        let mut c = RecodedCharID::new();
        c.set3(3, 8, 11);
        codes.push(c);
        codes
    }

    #[test]
    fn decode_encode_roundtrip_ascii_and_cjk() {
        let mut rec = UnicharCompress::new();
        rec.setup_direct(sample_codes());
        assert_eq!(rec.size(), 5);
        for id in 0..rec.size() {
            let code = rec.encode_unichar(id).expect("encodable");
            assert_eq!(
                rec.decode_unichar(&code),
                id as UnicharId,
                "decode(encode({id})) must round-trip"
            );
        }
        // Out-of-range id is not encodable.
        assert!(rec.encode_unichar(5).is_none());
    }

    #[test]
    fn distinct_ids_get_distinct_codes() {
        let mut rec = UnicharCompress::new();
        rec.setup_direct(sample_codes());
        // The two length-3 CJK codes differ only in the middle unit but must be
        // distinct sequences that decode to distinct ids.
        let c3 = rec.encode_unichar(3).unwrap();
        let c4 = rec.encode_unichar(4).unwrap();
        assert_ne!(c3, c4);
        assert_ne!(rec.decode_unichar(&c3), rec.decode_unichar(&c4));
        // Every id encodes to a code that decodes back to only itself.
        for id in 0..rec.size() {
            let code = rec.encode_unichar(id).unwrap();
            assert_eq!(rec.decode_unichar(&code), id as UnicharId);
        }
    }

    #[test]
    fn code_range_and_valid_starts() {
        let mut rec = UnicharCompress::new();
        rec.setup_direct(sample_codes());
        // Max unit value used is 11, so range is 12.
        assert_eq!(rec.code_range(), 12);
        // Start units 0,1,2,3 are valid; 7/8/11 appear only in later positions.
        assert!(rec.is_valid_first_code(0));
        assert!(rec.is_valid_first_code(3));
        assert!(!rec.is_valid_first_code(7));
        // The length-3 codes share the prefix [3]; its next-code list has 7 and 8.
        let mut prefix = RecodedCharID::new();
        prefix.set(0, 3);
        let next = rec
            .get_next_codes(&prefix)
            .expect("prefix [3] has next codes");
        assert!(next.contains(&7) && next.contains(&8));
    }

    #[test]
    fn deserialize_matches_setup_direct() {
        let codes = sample_codes();
        let buf = serialize_recoder(&codes);
        let mut fp = TFile::new(&buf);
        let mut rec = UnicharCompress::new();
        rec.deserialize(&mut fp).unwrap();
        assert_eq!(rec.size(), codes.len());
        for id in 0..rec.size() {
            let code = rec.encode_unichar(id).unwrap();
            assert_eq!(rec.decode_unichar(&code), id as UnicharId);
        }
        assert_eq!(rec.code_range(), 12);
    }

    #[test]
    fn deserialize_rejects_overlong_code() {
        // One entry claiming length 10 (> MAX_CODE_LEN = 9).
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_ne_bytes()); // 1 entry
        buf.push(1u8); // self_normalized
        buf.extend_from_slice(&10u32.to_ne_bytes()); // length 10
        let mut fp = TFile::new(&buf);
        let mut rec = UnicharCompress::new();
        assert_eq!(
            rec.deserialize(&mut fp).unwrap_err().kind(),
            crate::error::ErrorKind::Format
        );
    }

    /// Optional real-file smoke test tying the whole Phase-2 read path together:
    /// load the `LSTM_UNICHARSET` + `LSTM_RECODER` components from a real
    /// `.traineddata` via [`TessdataManager`] (exactly as
    /// `LSTMRecognizer::LoadCharsets` does) and round-trip a known character
    /// through the recoder. Ignored by default; downloads nothing.
    #[test]
    #[ignore = "set KOPITIAM_TEST_TRAINEDDATA to a real LSTM .traineddata to run"]
    fn real_traineddata_recoder_roundtrip() {
        use crate::tessdata::{TessdataManager, TessdataType};

        let path = std::env::var("KOPITIAM_TEST_TRAINEDDATA")
            .expect("KOPITIAM_TEST_TRAINEDDATA must point at a real LSTM .traineddata");
        let data = std::fs::read(&path).expect("failed to read traineddata");
        let mgr = TessdataManager::from_bytes(&data).expect("failed to parse traineddata");

        // Load the LSTM unicharset (a text component).
        let mut uni_fp = mgr
            .component_reader(TessdataType::LstmUnicharset)
            .expect("traineddata should contain an lstm-unicharset");
        let unicharset =
            Unicharset::load(&mut uni_fp, false).expect("failed to load lstm-unicharset");
        assert!(unicharset.size() > 3, "unicharset should be non-trivial");

        // Load the recoder (a binary component) and rebuild the decoder.
        let mut rec_fp = mgr
            .component_reader(TessdataType::LstmRecoder)
            .expect("traineddata should contain an lstm-recoder");
        let mut recoder = UnicharCompress::new();
        recoder
            .deserialize(&mut rec_fp)
            .expect("failed to deserialize lstm-recoder");

        // The recoder covers every unichar id (plus possibly a null code).
        assert!(recoder.size() >= unicharset.size());

        // Round-trip a known normal character (skip special codes, whose codes
        // may be shared). Fall back to the space unichar.
        let mut roundtripped = false;
        for cand in ["A", "a", "0", "e", " "] {
            let id = unicharset.unichar_to_id(cand);
            if id == INVALID_UNICHAR_ID {
                continue;
            }
            if id != crate::unicharset::UNICHAR_SPACE
                && (id as usize) < crate::unicharset::SPECIAL_UNICHAR_CODES_COUNT
            {
                continue; // skip Joined/Broken, whose codes can collide
            }
            let code = recoder
                .encode_unichar(id as usize)
                .expect("id should encode");
            assert_eq!(
                recoder.decode_unichar(&code),
                id,
                "recoder must round-trip {cand:?} (id {id})"
            );
            roundtripped = true;
            break;
        }
        assert!(
            roundtripped,
            "expected at least one common character to round-trip"
        );
    }

    #[test]
    fn decompose_hangul_matches_known_values() {
        // 가 (U+AC00) is the first Hangul: L=0, V=0, T=0.
        assert_eq!(UnicharCompress::decompose_hangul(0xAC00), Some((0, 0, 0)));
        // 힣 (U+D7A3) is the last Hangul: L=18, V=20, T=27.
        assert_eq!(
            UnicharCompress::decompose_hangul(0xD7A3),
            Some((18, 20, 27))
        );
        // Non-Hangul returns None.
        assert_eq!(UnicharCompress::decompose_hangul(0x41), None);
        assert_eq!(UnicharCompress::decompose_hangul(0xD7A4), None);
    }
}
