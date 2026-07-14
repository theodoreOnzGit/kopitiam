//! Loading a HuggingFace `tokenizer.json`.
//!
//! This is the *only* place in the crate that touches the GPT-2
//! byte-to-unicode mapping (see [`crate::byte_map`]): `tokenizer.json`
//! vocab keys and merge strings are written in the mapped alphabet (e.g.
//! `"Ġhello"`), and this module decodes each one back to its canonical raw
//! bytes exactly once, at load time. Everything downstream --
//! [`crate::vocab::Vocab`], [`crate::merges::MergeTable`],
//! [`crate::bpe::BpeTokenizer`] -- works in raw bytes only.
//!
//! Two `tokenizer.json` shapes are handled because both exist in the wild:
//!
//! * `model.merges` as `["tok_a tok_b", ...]` (the classic GPT-2/early
//!   `tokenizers`-library shape: one string per rule, space-separated).
//! * `model.merges` as `[["tok_a", "tok_b"], ...]` (the newer two-element-
//!   array shape some `tokenizers` versions emit instead).
//!
//! `kopitiam-loader`'s GGUF path does *not* go through this module -- GGUF
//! embeds its vocab as metadata arrays with no JSON or byte-mapping
//! involved, so it should call [`crate::BpeTokenizer::from_vocab_and_merges`]
//! directly with bytes it has already decoded itself.

use crate::bpe::BpeTokenizer;
use crate::byte_map::decode_mapped_token;
use crate::vocab::Vocab;
use kopitiam_core::{Error, Result};
use serde_json::Value;

/// Parses a `tokenizer.json` document (already read into memory) into a
/// ready-to-use [`BpeTokenizer`].
///
/// Populates, in addition to the base vocab and merges:
/// * every entry in the top-level `added_tokens` array as a special token
///   (registered via [`crate::BpeTokenizer::add_special_token`]) -- this is
///   where Qwen's `<|endoftext|>`, `<|im_start|>` and `<|im_end|>` actually
///   live in a real file;
/// * `add_prefix_space`, read from the `pre_tokenizer` section (looking
///   inside a `"type": "Sequence"` wrapper if present) and defaulting to
///   `false` if the field is absent.
pub fn from_tokenizer_json(json: &str) -> Result<BpeTokenizer> {
    let root: Value = serde_json::from_str(json).map_err(|e| Error::MalformedModel {
        format: "tokenizer.json",
        reason: format!("invalid JSON: {e}"),
    })?;

    let model = root.get("model").ok_or_else(|| Error::MalformedModel {
        format: "tokenizer.json",
        reason: "missing top-level \"model\" field".to_string(),
    })?;

    let vocab = parse_vocab(model)?;
    let merges = parse_merges(model)?;
    let add_prefix_space = find_add_prefix_space(&root);

    let mut tokenizer =
        BpeTokenizer::from_vocab(vocab, merges)?.with_add_prefix_space(add_prefix_space);

    for (content, id) in parse_added_tokens(&root)? {
        tokenizer.add_special_token(content, id)?;
    }

    Ok(tokenizer)
}

fn parse_vocab(model: &Value) -> Result<Vocab> {
    let entries = model
        .get("vocab")
        .and_then(Value::as_object)
        .ok_or_else(|| Error::MalformedModel {
            format: "tokenizer.json",
            reason: "missing or non-object \"model.vocab\" field".to_string(),
        })?;

    let mut vocab = Vocab::new();
    for (mapped, id_value) in entries {
        let id = id_value
            .as_u64()
            .ok_or_else(|| Error::MalformedModel {
                format: "tokenizer.json",
                reason: format!("vocab entry {mapped:?} has a non-integer id"),
            })?
            .try_into()
            .map_err(|_| Error::MalformedModel {
                format: "tokenizer.json",
                reason: format!("vocab entry {mapped:?} has an id that does not fit in u32"),
            })?;
        let bytes = decode_mapped_token(mapped).ok_or_else(|| Error::MalformedModel {
            format: "tokenizer.json",
            reason: format!(
                "vocab entry {mapped:?} contains a character outside the byte-level alphabet"
            ),
        })?;
        vocab.insert(id, bytes)?;
    }
    Ok(vocab)
}

fn parse_merges(model: &Value) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let raw = model
        .get("merges")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::MalformedModel {
            format: "tokenizer.json",
            reason: "missing or non-array \"model.merges\" field".to_string(),
        })?;

    raw.iter()
        .enumerate()
        .map(|(rank, entry)| parse_one_merge(rank, entry))
        .collect()
}

/// Parses a single merge rule in either the legacy `"a b"` string shape or
/// the `["a", "b"]` two-element-array shape (see module docs).
fn parse_one_merge(rank: usize, entry: &Value) -> Result<(Vec<u8>, Vec<u8>)> {
    let (left, right) = match entry {
        Value::String(s) => {
            // Mapped token strings never contain a literal ASCII space --
            // 0x20 is itself one of the bytes the mapping remaps away from
            // its literal form (see `byte_map`) -- so splitting on the
            // first space unambiguously separates the two sides.
            let mut parts = s.splitn(2, ' ');
            let (Some(l), Some(r), None) = (parts.next(), parts.next(), parts.next()) else {
                return Err(Error::MalformedModel {
                    format: "tokenizer.json",
                    reason: format!("merge rule {rank} ({s:?}) is not \"left right\""),
                });
            };
            (l.to_string(), r.to_string())
        }
        Value::Array(pair) => match pair.as_slice() {
            [Value::String(l), Value::String(r)] => (l.clone(), r.clone()),
            _ => {
                return Err(Error::MalformedModel {
                    format: "tokenizer.json",
                    reason: format!("merge rule {rank} is not a two-string array"),
                });
            }
        },
        other => {
            return Err(Error::MalformedModel {
                format: "tokenizer.json",
                reason: format!(
                    "merge rule {rank} has unsupported JSON shape {other:?}; expected a \
                     \"left right\" string or a [left, right] array"
                ),
            });
        }
    };

    let left_bytes = decode_mapped_token(&left).ok_or_else(|| Error::MalformedModel {
        format: "tokenizer.json",
        reason: format!("merge rule {rank}: left side {left:?} is not byte-level-mapped"),
    })?;
    let right_bytes = decode_mapped_token(&right).ok_or_else(|| Error::MalformedModel {
        format: "tokenizer.json",
        reason: format!("merge rule {rank}: right side {right:?} is not byte-level-mapped"),
    })?;
    Ok((left_bytes, right_bytes))
}

fn parse_added_tokens(root: &Value) -> Result<Vec<(String, u32)>> {
    let Some(entries) = root.get("added_tokens").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    entries
        .iter()
        .map(|entry| {
            let content = entry
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::MalformedModel {
                    format: "tokenizer.json",
                    reason: format!("added_tokens entry {entry:?} has no string \"content\""),
                })?
                .to_string();
            let id = entry
                .get("id")
                .and_then(Value::as_u64)
                .ok_or_else(|| Error::MalformedModel {
                    format: "tokenizer.json",
                    reason: format!("added_tokens entry {entry:?} has no integer \"id\""),
                })?
                .try_into()
                .map_err(|_| Error::MalformedModel {
                    format: "tokenizer.json",
                    reason: format!(
                        "added_tokens entry {entry:?} has an id that does not fit in u32"
                    ),
                })?;
            Ok((content, id))
        })
        .collect()
}

/// Looks for `add_prefix_space` under `pre_tokenizer`, unwrapping a
/// `"type": "Sequence"` wrapper (some `tokenizers` configs compose a
/// `ByteLevel` pre-tokenizer alongside others, e.g. a `Split`) to find it.
/// Defaults to `false` if no `ByteLevel` section is found at all.
fn find_add_prefix_space(root: &Value) -> bool {
    let Some(pre_tokenizer) = root.get("pre_tokenizer") else {
        return false;
    };
    if let Some(v) = pre_tokenizer
        .get("add_prefix_space")
        .and_then(Value::as_bool)
    {
        return v;
    }
    if let Some(list) = pre_tokenizer.get("pretokenizers").and_then(Value::as_array) {
        for entry in list {
            if let Some(v) = entry.get("add_prefix_space").and_then(Value::as_bool) {
                return v;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tokenizer;

    /// A minimal but complete `tokenizer.json`: full byte-level base vocab
    /// (256 single-byte tokens, generated instead of spelled out) plus one
    /// merge, one added special token, and an explicit `add_prefix_space`.
    ///
    /// The special token is given id 257 -- immediately after the base
    /// vocab (256 bytes) and the one merged token (256) -- matching how
    /// real `tokenizer.json` files lay out special tokens contiguously
    /// right after the base vocab, with no gap. `Vocab::len` reports the
    /// highest id plus one (see `vocab.rs`), so a gapped fixture would
    /// make `vocab_size` reflect the gap rather than the token count;
    /// this fixture is deliberately contiguous so that distinction does
    /// not muddy this test.
    fn sample_json() -> String {
        use crate::byte_map::byte_to_unicode;
        let mut vocab = serde_json::Map::new();
        for b in 0u16..=255 {
            let c = byte_to_unicode(b as u8);
            vocab.insert(c.to_string(), serde_json::json!(b));
        }
        // "hi" = byte 'h' (0x68=104) + byte 'i' (0x69=105), merged token id 256.
        let h = byte_to_unicode(b'h').to_string();
        let i = byte_to_unicode(b'i').to_string();
        vocab.insert(format!("{h}{i}"), serde_json::json!(256));

        serde_json::json!({
            "added_tokens": [
                {"id": 257, "content": "<|endoftext|>", "special": true}
            ],
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false
            },
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": [[h, i]]
            }
        })
        .to_string()
    }

    #[test]
    fn loads_vocab_merges_and_special_tokens() {
        let tok = from_tokenizer_json(&sample_json()).expect("valid tokenizer.json");
        assert_eq!(tok.vocab_size(), 258);
        assert_eq!(tok.special_token_id("<|endoftext|>"), Some(257));

        let ids = tok.encode("hi<|endoftext|>").unwrap();
        assert_eq!(ids, vec![256, 257]);
        assert_eq!(tok.decode(&ids).unwrap(), "hi<|endoftext|>");
    }

    #[test]
    fn legacy_string_merge_format_is_also_accepted() {
        use crate::byte_map::byte_to_unicode;
        let mut vocab = serde_json::Map::new();
        for b in 0u16..=255 {
            vocab.insert(byte_to_unicode(b as u8).to_string(), serde_json::json!(b));
        }
        let h = byte_to_unicode(b'h').to_string();
        let i = byte_to_unicode(b'i').to_string();
        vocab.insert(format!("{h}{i}"), serde_json::json!(256));

        let json = serde_json::json!({
            "model": {
                "vocab": vocab,
                "merges": [format!("{h} {i}")]
            }
        })
        .to_string();

        let tok = from_tokenizer_json(&json).expect("legacy merges format");
        assert_eq!(tok.encode("hi").unwrap(), vec![256]);
    }

    #[test]
    fn missing_model_field_is_a_malformed_model_error() {
        let err = from_tokenizer_json("{}").unwrap_err();
        assert!(matches!(err, Error::MalformedModel { .. }));
    }

    #[test]
    fn invalid_json_is_a_malformed_model_error() {
        let err = from_tokenizer_json("not json").unwrap_err();
        assert!(matches!(err, Error::MalformedModel { .. }));
    }
}
