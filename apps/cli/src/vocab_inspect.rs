//! `kopitiam models inspect` — what is actually inside a GGUF's embedded
//! vocabulary.
//!
//! # Why this command exists
//!
//! A model that will not load reports one symptom and hides the cause. The real
//! case that prompted this, on a Termux tablet:
//!
//! ```text
//! error: malformed bpe-vocab model file: byte-level vocab has no single-byte
//!        token for byte 0x04; byte-level BPE requires all 256 bytes as base tokens
//! ```
//!
//! That message names the symptom precisely and still leaves two *opposite*
//! explanations open, which need opposite fixes:
//!
//! 1. **The loader is wrong.** llama.cpp writes `LLAMA_TOKEN_TYPE_BYTE` tokens
//!    as the literal text `<0x04>`. Decoded through the GPT-2 byte-level
//!    alphabet that becomes the six ASCII characters `<`,`0`,`x`,`0`,`4`,`>`,
//!    not byte `0x04` — so the byte is present in the file and we fail to see
//!    it. Fix: decode BYTE-type tokens.
//! 2. **The check is too strict.** HuggingFace BPE trainers only seed the full
//!    256-byte initial alphabet when told to; a vocabulary trained without that
//!    genuinely lacks bytes that never occurred in its corpus. Fix: tolerate
//!    holes instead of demanding all 256.
//!
//! Guessing between those and shipping the wrong one is how a "fix" makes the
//! next failure harder to read. This command answers it from the file:
//! **which** bytes are missing, whether any BYTE-type tokens exist, and what
//! the token-type histogram looks like.
//!
//! # It worked: how the real case was settled (2026-07-27)
//!
//! Run against the actual `smollm2-360m-instruct-q8_0.gguf`, this command said
//! 21 bytes absent, **zero** `<0xNN>` spellings, so explanation (2). The check
//! was relaxed accordingly (see [`kopitiam_tokenizer::BpeTokenizer::missing_byte_tokens`]),
//! and that model now runs end to end. Explanation (1) is still live for other
//! files, which is why both verdicts stay.
//!
//! # Deliberately read-only and deliberately cheap
//!
//! It reads metadata only — it never builds a tokenizer, never loads weights,
//! and never dequantizes. So it works on exactly the model that is failing to
//! load, which is the whole point: a diagnostic that needs the broken thing to
//! work first is no diagnostic.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use kopitiam_loader::LoadedModel;
use kopitiam_tokenizer::byte_map::decode_mapped_token;

/// `llama.cpp`'s `LLAMA_TOKEN_TYPE_*`, as written into
/// `tokenizer.ggml.token_type`. Named here rather than left as bare integers
/// because the whole value of the histogram is being able to read it.
fn token_type_name(t: i64) -> &'static str {
    match t {
        1 => "NORMAL",
        2 => "UNKNOWN",
        3 => "CONTROL",
        4 => "USER_DEFINED",
        5 => "UNUSED",
        6 => "BYTE",
        _ => "unrecognised",
    }
}

/// Options for `kopitiam models inspect`.
#[derive(Args, Debug)]
pub struct InspectArgs {
    /// Path to the `.gguf` file to inspect. Take it straight from
    /// `kopitiam models path <id>`, or from the `file:` line of a load failure.
    pub file: PathBuf,

    /// Print the first N vocabulary entries with their decoded bytes, to eyeball
    /// how the file spells its tokens (mapped `Ġ` form vs raw vs `<0xNN>`).
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub sample: usize,
}

/// What we learned about one GGUF's vocabulary.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct VocabReport {
    pub token_count: usize,
    pub merge_count: usize,
    /// Bytes with no single-byte token, after byte-level decoding. Empty means
    /// the alphabet is complete and the "all 256" requirement is satisfiable.
    pub missing_bytes: Vec<u8>,
    /// `(type_name, count)`, ascending by type id — the histogram that says
    /// whether BYTE-type tokens are in play at all.
    pub type_histogram: Vec<(&'static str, usize)>,
    /// Tokens whose text looks like llama.cpp's `<0xNN>` byte spelling. If this
    /// is non-empty while `missing_bytes` is too, explanation (1) above is the
    /// live one and the loader needs to decode them.
    pub byte_spelled: Vec<(usize, String)>,
    /// Tokens that could not be decoded through the byte-level alphabet at all.
    pub undecodable: Vec<(usize, String)>,
}

impl VocabReport {
    /// The verdict line: which of the two explanations the evidence supports.
    ///
    /// Stated as a conclusion rather than left to the reader because the whole
    /// reason this command exists is that the raw numbers were being guessed at.
    pub fn verdict(&self) -> String {
        match (self.missing_bytes.is_empty(), self.byte_spelled.is_empty()) {
            (true, _) => {
                "alphabet is COMPLETE — all 256 bytes present, so the missing-byte error \
                 did not come from this file's vocabulary"
                    .to_string()
            }
            (false, false) => format!(
                "loader gap: {} byte(s) missing, but {} token(s) are spelled `<0xNN>` — \
                 those are LLAMA_TOKEN_TYPE_BYTE entries the loader is not decoding. \
                 The bytes ARE in the file; teach the loader to read them.",
                self.missing_bytes.len(),
                self.byte_spelled.len()
            ),
            (false, true) => format!(
                "vocabulary is genuinely incomplete: {} byte(s) absent and NO `<0xNN>` \
                 spellings anywhere, so the file really has no token for them. This is \
                 TOLERATED — the tokenizer builds and the model runs; those bytes are \
                 dropped on encode. Nothing to fix unless one of them is reachable from \
                 valid UTF-8 and you care (see BpeTokenizer::missing_byte_tokens).",
                self.missing_bytes.len()
            ),
        }
    }

    /// Human report, one fact per line.
    pub fn to_report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("tokens:  {}\n", self.token_count));
        out.push_str(&format!("merges:  {}\n", self.merge_count));
        out.push_str("token types:\n");
        if self.type_histogram.is_empty() {
            out.push_str("  (no tokenizer.ggml.token_type array in this file)\n");
        }
        for (name, count) in &self.type_histogram {
            out.push_str(&format!("  {name:<13} {count}\n"));
        }
        out.push_str(&format!("missing single-byte tokens: {}\n", self.missing_bytes.len()));
        if !self.missing_bytes.is_empty() {
            let list: Vec<String> =
                self.missing_bytes.iter().map(|b| format!("{b:#04x}")).collect();
            out.push_str(&format!("  {}\n", list.join(" ")));
        }
        if !self.byte_spelled.is_empty() {
            out.push_str(&format!("`<0xNN>`-spelled tokens: {}\n", self.byte_spelled.len()));
            for (id, text) in self.byte_spelled.iter().take(8) {
                out.push_str(&format!("  id {id}: {text}\n"));
            }
        }
        if !self.undecodable.is_empty() {
            out.push_str(&format!("undecodable tokens: {}\n", self.undecodable.len()));
            for (id, text) in self.undecodable.iter().take(8) {
                out.push_str(&format!("  id {id}: {text:?}\n"));
            }
        }
        out.push_str(&format!("\nverdict: {}\n", self.verdict()));
        out
    }
}

/// Is this the `<0xNN>` spelling llama.cpp uses for a BYTE-type token?
///
/// Matched by shape rather than by trusting `token_type`, because a file may
/// carry the spelling without the type array (the type array is optional) — and
/// the spelling is what actually defeats the byte-level decoder.
fn byte_spelling(text: &str) -> Option<u8> {
    let hex = text.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

/// Builds the report from an already-loaded GGUF.
///
/// Split from the I/O so it is unit-testable against synthetic inputs — the
/// verdict logic is the part that must not be wrong, since it is what somebody
/// will act on.
pub fn analyse(tokens: &[String], merges: usize, types: &[i64]) -> VocabReport {
    let mut decoded: Vec<Option<Vec<u8>>> = Vec::with_capacity(tokens.len());
    let mut byte_spelled = Vec::new();
    let mut undecodable = Vec::new();

    for (id, text) in tokens.iter().enumerate() {
        if byte_spelling(text).is_some() {
            byte_spelled.push((id, text.clone()));
        }
        match decode_mapped_token(text) {
            Some(bytes) => decoded.push(Some(bytes)),
            None => {
                undecodable.push((id, text.clone()));
                decoded.push(None);
            }
        }
    }

    // Which of the 256 have a token that is EXACTLY that one byte? That is the
    // question `BpeTokenizer::from_vocab` asks, so ask it identically — a
    // diagnostic that measures something subtly different is worse than none.
    let mut present = [false; 256];
    for bytes in decoded.iter().flatten() {
        if let [b] = bytes[..] {
            present[b as usize] = true;
        }
    }
    let missing_bytes =
        (0u16..=255).map(|b| b as u8).filter(|b| !present[*b as usize]).collect();

    let mut counts: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
    for t in types {
        *counts.entry(*t).or_default() += 1;
    }
    let type_histogram = counts.into_iter().map(|(t, c)| (token_type_name(t), c)).collect();

    VocabReport {
        token_count: tokens.len(),
        merge_count: merges,
        missing_bytes,
        type_histogram,
        byte_spelled,
        undecodable,
    }
}

/// Runs `kopitiam models inspect`.
pub fn run(args: InspectArgs) -> Result<()> {
    let report = inspect_file(&args.file)?;
    print!("{}", report.to_report());

    if args.sample > 0 {
        println!("\nfirst {} tokens:", args.sample);
        let model = load(&args.file)?;
        if let Some(tokens) = model.metadata().raw.get_array("tokenizer.ggml.tokens") {
            for (id, v) in tokens.iter().take(args.sample).enumerate() {
                let text = v.as_str().unwrap_or("<not a string>");
                let decoded = decode_mapped_token(text)
                    .map(|b| format!("{b:02x?}"))
                    .unwrap_or_else(|| "<undecodable>".to_string());
                println!("  {id:>5}: {text:?} -> {decoded}");
            }
        }
    }
    Ok(())
}

fn load(path: &Path) -> Result<LoadedModel> {
    kopitiam_loader::load_model(path)
        .with_context(|| format!("reading GGUF metadata from {}", path.display()))
}

fn inspect_file(path: &Path) -> Result<VocabReport> {
    let model = load(path)?;
    let raw = &model.metadata().raw;

    let tokens: Vec<String> = raw
        .get_array("tokenizer.ggml.tokens")
        .map(|a| a.iter().map(|v| v.as_str().unwrap_or_default().to_string()).collect())
        .unwrap_or_default();
    let merges = raw.get_array("tokenizer.ggml.merges").map(|a| a.len()).unwrap_or(0);
    let types: Vec<i64> = raw
        .get_array("tokenizer.ggml.token_type")
        .map(|a| a.iter().filter_map(|v| v.as_i64().or_else(|| v.as_u64().map(|u| u as i64))).collect())
        .unwrap_or_default();

    Ok(analyse(&tokens, merges, &types))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GPT-2 byte-level spelling of a single byte, so tests can build a
    /// vocabulary the same way a real export does.
    fn mapped(b: u8) -> String {
        match b {
            0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => (b as char).to_string(),
            _ => {
                // The GPT-2 fallback: the nth otherwise-unmapped byte becomes
                // U+0100 + n, in ascending byte order.
                let n = (0u16..b as u16)
                    .filter(|c| !matches!(*c as u8, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF))
                    .count() as u32;
                char::from_u32(0x100 + n).unwrap().to_string()
            }
        }
    }

    fn complete_alphabet() -> Vec<String> {
        (0u16..=255).map(|b| mapped(b as u8)).collect()
    }

    #[test]
    fn a_complete_alphabet_reports_no_missing_bytes() {
        let r = analyse(&complete_alphabet(), 0, &[]);
        assert!(r.missing_bytes.is_empty(), "missing: {:?}", r.missing_bytes);
        assert!(r.verdict().contains("COMPLETE"));
    }

    #[test]
    fn a_hole_with_no_byte_spelling_reads_as_a_genuinely_incomplete_vocab() {
        // The "our check is too strict" case: the byte is simply not there, in
        // any spelling. Acting on this by teaching the loader `<0xNN>` would fix
        // nothing, which is exactly the wrong turn this command prevents.
        let mut v = complete_alphabet();
        v.retain(|t| *t != mapped(0x04));
        let r = analyse(&v, 0, &[]);
        assert_eq!(r.missing_bytes, vec![0x04]);
        assert!(r.byte_spelled.is_empty());
        assert!(r.verdict().contains("genuinely incomplete"), "got: {}", r.verdict());
    }

    #[test]
    fn a_hole_plus_a_byte_spelling_reads_as_a_loader_gap() {
        // The opposite case: the byte IS in the file, spelled `<0x04>`, and the
        // decoder cannot see it. Same symptom, opposite fix.
        let mut v = complete_alphabet();
        v.retain(|t| *t != mapped(0x04));
        v.push("<0x04>".to_string());
        let r = analyse(&v, 0, &[]);
        assert_eq!(r.missing_bytes, vec![0x04]);
        assert_eq!(r.byte_spelled.len(), 1);
        assert!(r.verdict().contains("loader gap"), "got: {}", r.verdict());
    }

    #[test]
    fn byte_spelling_parses_only_the_exact_two_hex_digit_form() {
        assert_eq!(byte_spelling("<0x04>"), Some(0x04));
        assert_eq!(byte_spelling("<0xFF>"), Some(0xFF));
        assert_eq!(byte_spelling("<0x4>"), None, "one digit is not the llama.cpp form");
        assert_eq!(byte_spelling("<0x004>"), None, "three digits is not it either");
        assert_eq!(byte_spelling("0x04"), None);
        assert_eq!(byte_spelling("hello"), None);
    }

    #[test]
    fn the_type_histogram_names_llama_cpp_types() {
        let r = analyse(&complete_alphabet(), 0, &[1, 1, 3, 6, 6, 6]);
        assert_eq!(
            r.type_histogram,
            vec![("NORMAL", 2), ("CONTROL", 1), ("BYTE", 3)]
        );
    }

    #[test]
    fn missing_bytes_are_reported_in_ascending_order() {
        // Ordering matters for reading the output against the loader's own error,
        // which reports the FIRST missing byte it hits while counting up.
        let mut v = complete_alphabet();
        v.retain(|t| *t != mapped(0x04) && *t != mapped(0x00) && *t != mapped(0x7F));
        let r = analyse(&v, 0, &[]);
        assert_eq!(r.missing_bytes, vec![0x00, 0x04, 0x7F]);
    }

    #[test]
    fn an_empty_vocab_reports_every_byte_missing_rather_than_panicking() {
        let r = analyse(&[], 0, &[]);
        assert_eq!(r.missing_bytes.len(), 256);
        assert_eq!(r.token_count, 0);
    }

    #[test]
    fn multi_byte_tokens_do_not_count_as_single_byte_base_tokens() {
        // `id_of(&[b])` matches a token that is EXACTLY that byte. A token that
        // merely CONTAINS it must not be mistaken for the base token, or the
        // report would disagree with the loader it is meant to explain.
        let v = vec![mapped(0x41) + &mapped(0x42)]; // "AB"
        let r = analyse(&v, 0, &[]);
        assert!(r.missing_bytes.contains(&0x41));
        assert!(r.missing_bytes.contains(&0x42));
    }
}
