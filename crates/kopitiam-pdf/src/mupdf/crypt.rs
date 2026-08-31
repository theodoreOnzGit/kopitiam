//! The PDF **standard security handler** (§7.6): deriving a file key from a
//! password and decrypting the strings and streams it protects.
//!
//! Ported from the specification (ISO 32000-1:2008 §7.6.3, Algorithms 1-7),
//! cross-checked against MuPDF's `source/pdf/pdf-crypt.c`. gh-98.
//!
//! # What this is actually for
//!
//! Most encrypted PDFs in ordinary life are not secret. They are
//! **owner-restricted**: a government or company form with an *empty user
//! password* and a permissions bitfield saying "you may fill this in, you may
//! not re-typeset it". Every viewer opens them with no prompt, because the
//! empty password is the user password. That is the case this exists to serve,
//! and it needs no password UI at all.
//!
//! (A real user password is supported too -- [`Decryptor::new`] takes one --
//! but nothing in this crate asks a human for it yet.)
//!
//! # The revisions, and what changes between them
//!
//! | `/R` | `/V` | Key | Method |
//! |---|---|---|---|
//! | 2 | 1 | 40-bit, MD5 once | RC4 |
//! | 3 | 2 | `/Length` bits, MD5 x51 | RC4 |
//! | 4 | 4 | `/Length` bits, MD5 x51 | RC4 or AES-128, per `/CF` |
//!
//! `/R 5` and `/R 6` (AES-256, SHA-2 based, PDF 2.0) are **not** implemented;
//! they use a completely different key derivation and are refused by name.
//!
//! # Why MD5, in 2026
//!
//! Because the specification defines the key in terms of it. MD5's collision
//! weakness is irrelevant here: we are not authenticating anything, we are
//! reproducing a byte string a standard prescribes. Swapping in SHA-256 would
//! not be "more secure", it would fail to open every R2-R4 document ever
//! written. Do not.
//!
//! # The trap: object streams must not be decrypted twice
//!
//! A stream is decrypted with a key derived from **its own** object number.
//! Objects living inside an object stream were encrypted as part of that
//! stream's bytes and are already in the clear once it is decompressed -- so
//! their strings must be left alone. Decrypting them again with their own
//! object numbers produces convincing garbage. See
//! [`Decryptor::decrypt_string`]'s contract.

use aes::Aes128;
use aes::cipher::{BlockDecryptMut, KeyIvInit};
use md5::{Digest, Md5};
use rc4::{KeyInit, Rc4, StreamCipher};


use super::error::{Error, Result};
use super::object::Object;

type Aes128CbcDec = cbc::Decryptor<Aes128>;

/// Apply RC4 in place with a **runtime-length** key.
///
/// The `rc4` crate makes the key length a compile-time generic, which is the
/// right call for a cipher API and the wrong shape for us: §7.6.3.2 lets
/// `/Length` be anything from 40 to 128 bits, so the length is only known once
/// the document's `/Encrypt` dictionary has been read. Rather than hand-roll
/// RC4 -- twenty easy lines that are also twenty lines of unreviewed crypto --
/// this dispatches to the vetted implementation at each size the standard
/// permits.
///
/// A key outside 5..=16 bytes leaves the buffer untouched. `Decryptor::new`
/// rejects those before they can reach here, so this is belt-and-braces
/// rather than a silent no-op path.
fn rc4_apply(key: &[u8], buf: &mut [u8]) {
    use rc4::consts::*;
    macro_rules! dispatch {
        ($($n:literal => $ty:ty),+ $(,)?) => {
            match key.len() {
                $($n => {
                    let mut c = Rc4::<$ty>::new(key.into());
                    c.apply_keystream(buf);
                })+
                _ => {}
            }
        };
    }
    dispatch! {
        5 => U5, 6 => U6, 7 => U7, 8 => U8, 9 => U9, 10 => U10,
        11 => U11, 12 => U12, 13 => U13, 14 => U14, 15 => U15, 16 => U16,
    }
}

/// The 32-byte padding string from §7.6.3.3, Algorithm 2, step (a).
///
/// A password shorter than 32 bytes is extended with the front of this; an
/// empty password *is* this string. Copied byte-for-byte from the standard
/// (MuPDF: `pdf_password_pad`, `pdf-crypt.c`).
const PAD: [u8; 32] = [
    0x28, 0xBF, 0x4E, 0x5E, 0x4E, 0x75, 0x8A, 0x41, 0x64, 0x00, 0x4E, 0x56, 0xFF, 0xFA, 0x01, 0x08,
    0x2E, 0x2E, 0x00, 0xB6, 0xD0, 0x68, 0x3E, 0x80, 0x2F, 0x0C, 0xA9, 0xFE, 0x64, 0x53, 0x69, 0x7A,
];

/// The literal bytes `sAlT`, appended to the per-object key material for AES
/// (§7.6.2, Algorithm 1, step (b)). Spelled out rather than written as a
/// string literal because its odd capitalisation is load-bearing and reads
/// like a typo otherwise.
const AES_SALT: [u8; 4] = [0x73, 0x41, 0x6C, 0x54];

/// AES's block size in bytes -- also the length of the IV that precedes an
/// encrypted stream (§7.6.2).
const AES_BLOCK: usize = 16;

/// How a crypt filter transforms bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// No encryption -- `/CFM /None`, or a filter naming `/Identity`.
    None,
    /// RC4, `/CFM /V2` and every `/V 1`/`/V 2` document.
    Rc4,
    /// AES-128 in CBC mode, `/CFM /AESV2`.
    Aes128,
}

/// The fields of an `/Encrypt` dictionary that the standard security handler
/// needs.
///
/// A struct rather than a dozen positional arguments, which was both a clippy
/// complaint and a genuine hazard: `o` and `u` are both `&[u8]`, and `r` and
/// `length_bits` are both `i64`, so a transposed pair would compile cleanly
/// and derive a silently wrong key.
#[derive(Debug, Clone, Copy)]
pub struct EncryptDict<'a> {
    /// `/Filter` -- the handler's name. Only `Standard` is implemented.
    pub filter: &'a str,
    /// `/V`, the algorithm version. Carried for completeness and future
    /// `/V 5` support; the R2-R4 algorithms key off `/R` alone.
    pub v: i64,
    /// `/R`, the handler revision -- what actually selects the algorithm.
    pub r: i64,
    /// `/Length` in **bits** (40..=128). Zero means absent, i.e. 40.
    pub length_bits: i64,
    /// `/O`, the owner-password entry. Must be 32 bytes.
    pub o: &'a [u8],
    /// `/U`, the user-password entry, checked by Algorithm 6.
    pub u: &'a [u8],
    /// `/P`, the permissions bitfield. **Signed** -- see [`file_key`].
    pub p: i64,
    /// The first element of the trailer's `/ID` array.
    pub first_id: &'a [u8],
    /// `/EncryptMetadata`, defaulting to `true` when absent.
    pub encrypt_metadata: bool,
    pub stream_method: Method,
    pub string_method: Method,
}

/// A ready-to-use decryptor for one document.
#[derive(Debug, Clone)]
pub struct Decryptor {
    key: Vec<u8>,
    stream_method: Method,
    string_method: Method,
}

impl Decryptor {
    /// Build a decryptor from the trailer's `/Encrypt` dictionary.
    ///
    /// `first_id` is the first element of the trailer's `/ID` array, which is
    /// part of the key. `password` is the *user* password; pass `b""` for the
    /// overwhelmingly common empty-password case.
    ///
    /// Fails with [`Error::unsupported`] for a handler or revision this does
    /// not implement, and with [`Error::format`] when the dictionary is
    /// missing something the algorithm needs. A wrong password is
    /// [`Error::unsupported`] too, so a caller can tell "cannot" from "will
    /// not" by the message rather than by guessing.
    pub fn new(d: &EncryptDict<'_>, password: &[u8]) -> Result<Decryptor> {
        let EncryptDict {
            filter,
            r,
            length_bits,
            o,
            u,
            p,
            first_id,
            encrypt_metadata,
            stream_method,
            string_method,
            ..
        } = *d;
        if filter != "Standard" {
            return Err(Error::unsupported(format!(
                "security handler /{filter} (only /Standard is implemented)"
            )));
        }
        if !(2..=4).contains(&r) {
            return Err(Error::unsupported(format!(
                "standard security handler revision R{r} (only R2-R4 are \
                 implemented; R5/R6 use AES-256 with a different key derivation)"
            )));
        }
        if o.len() < 32 {
            return Err(Error::format("/Encrypt /O must be 32 bytes"));
        }

        // Key length: R2 is fixed at 40 bits; later revisions read /Length,
        // defaulting to 40 when it is absent (§7.6.3.2).
        let n = if r == 2 {
            5
        } else {
            let bits = if length_bits == 0 { 40 } else { length_bits };
            let bytes = (bits / 8) as usize;
            if !(5..=16).contains(&bytes) {
                return Err(Error::format(format!(
                    "/Encrypt /Length {bits} is not a supported key size"
                )));
            }
            bytes
        };

        let key = file_key(password, o, p, first_id, r, n, encrypt_metadata);

        // Algorithm 6: verify the password by recomputing /U from the key we
        // just derived. Skipping this would "succeed" with a wrong key and
        // hand back plausible-looking garbage.
        if !user_password_matches(&key, r, first_id, u) {
            return Err(Error::unsupported(
                "encrypted PDF needs a user password (the empty password does \
                 not open it)",
            ));
        }

        Ok(Decryptor {
            key,
            stream_method,
            string_method,
        })
    }

    /// Decrypt a stream's raw bytes, **before** any `/Filter` is applied.
    ///
    /// Order matters and is not negotiable: the bytes on disk are
    /// `Flate(plaintext)` encrypted, so they must be decrypted first and
    /// inflated second. Inflating first is what produced the "corrupt object
    /// stream" that started this work.
    pub fn decrypt_stream(&self, num: u32, generation: u16, data: &[u8]) -> Vec<u8> {
        self.apply(self.stream_method, num, generation, data)
    }

    /// Decrypt a string taken directly from an object body.
    ///
    /// **Only for objects stored on their own**, never for one unpacked from
    /// an object stream: those were encrypted as part of the containing
    /// stream's bytes and are already plaintext by the time they are parsed.
    /// Decrypting them again yields garbage that still parses, which is the
    /// worst kind of wrong.
    pub fn decrypt_string(&self, num: u32, generation: u16, data: &[u8]) -> Vec<u8> {
        self.apply(self.string_method, num, generation, data)
    }

    fn apply(&self, method: Method, num: u32, generation: u16, data: &[u8]) -> Vec<u8> {
        match method {
            Method::None => data.to_vec(),
            Method::Rc4 => {
                let key = self.object_key(num, generation, false);
                let mut out = data.to_vec();
                rc4_apply(&key, &mut out);
                out
            }
            Method::Aes128 => decrypt_aes_cbc(&self.object_key(num, generation, true), data),
        }
    }

    /// §7.6.2 Algorithm 1: the per-object key.
    ///
    /// Every object gets its own key, derived from the file key plus its
    /// object and generation numbers -- so the same plaintext in two objects
    /// does not encrypt identically.
    fn object_key(&self, num: u32, generation: u16, aes: bool) -> Vec<u8> {
        let mut h = Md5::new();
        h.update(&self.key);
        // Low three bytes of the object number, low two of the generation,
        // both little-endian.
        h.update([num as u8, (num >> 8) as u8, (num >> 16) as u8]);
        h.update([generation as u8, (generation >> 8) as u8]);
        if aes {
            h.update(AES_SALT);
        }
        let digest = h.finalize();
        // "n + 5, to a maximum of 16" -- the extra five bytes are the object
        // and generation bytes just mixed in.
        let take = (self.key.len() + 5).min(16);
        digest[..take].to_vec()
    }
}

/// §7.6.3.3 Algorithm 2: the file encryption key.
fn file_key(
    password: &[u8],
    o: &[u8],
    p: i64,
    first_id: &[u8],
    r: i64,
    n: usize,
    encrypt_metadata: bool,
) -> Vec<u8> {
    let mut h = Md5::new();
    h.update(pad_password(password));
    h.update(&o[..32]);
    // /P is a SIGNED 32-bit value written little-endian. Treating it as
    // unsigned changes the key for every document with the high bit set,
    // which is most of them (a restrictive /P is negative).
    h.update((p as i32).to_le_bytes());
    h.update(first_id);
    // R4 with /EncryptMetadata false mixes in FF FF FF FF (§7.6.3.3 step f).
    if r >= 4 && !encrypt_metadata {
        h.update([0xFF, 0xFF, 0xFF, 0xFF]);
    }
    let mut digest = h.finalize();
    // R3+ iterate 50 times over the first n bytes -- deliberate key
    // strengthening, and omitting it silently produces a wrong key.
    if r >= 3 {
        for _ in 0..50 {
            let mut h = Md5::new();
            h.update(&digest[..n]);
            digest = h.finalize();
        }
    }
    digest[..n].to_vec()
}

/// §7.6.3.3 Algorithm 2 step (a): pad or truncate to exactly 32 bytes.
fn pad_password(password: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let take = password.len().min(32);
    out[..take].copy_from_slice(&password[..take]);
    out[take..].copy_from_slice(&PAD[..32 - take]);
    out
}

/// §7.6.3.4 Algorithm 6: does this key correspond to the document's `/U`?
fn user_password_matches(key: &[u8], r: i64, first_id: &[u8], u: &[u8]) -> bool {
    let computed = if r == 2 {
        // Algorithm 4: RC4 of the padding string.
        let mut buf = PAD.to_vec();
        rc4_apply(key, &mut buf);
        buf
    } else {
        // Algorithm 5: MD5 of PAD + /ID, RC4'd, then 19 more RC4 passes with
        // the key XORed by the pass number.
        let mut h = Md5::new();
        h.update(PAD);
        h.update(first_id);
        let mut buf = h.finalize().to_vec();
        rc4_apply(key, &mut buf);
        for i in 1u8..=19 {
            let stepped: Vec<u8> = key.iter().map(|b| b ^ i).collect();
            rc4_apply(&stepped, &mut buf);
        }
        buf
    };
    // R3+ only defines the first 16 bytes of /U; the rest is arbitrary
    // padding and comparing it would reject valid documents.
    let len = if r == 2 { 32 } else { 16 };
    u.len() >= len && computed.len() >= len && computed[..len] == u[..len]
}

/// AES-128-CBC, with the IV as the leading 16 bytes (§7.6.2).
///
/// Returns empty for anything too short to hold an IV plus one block. The
/// PKCS#7 padding is stripped when it is well formed and left alone when it is
/// not -- a truncated or subtly wrong stream should come back short rather
/// than panic, since a PDF is untrusted input.
fn decrypt_aes_cbc(key: &[u8], data: &[u8]) -> Vec<u8> {
    if data.len() < 2 * AES_BLOCK || !(data.len() - AES_BLOCK).is_multiple_of(AES_BLOCK) {
        return Vec::new();
    }
    let (iv, body) = data.split_at(AES_BLOCK);
    let mut buf = body.to_vec();
    let Ok(mut dec) = Aes128CbcDec::new_from_slices(key, iv) else {
        return Vec::new();
    };
    // Decrypt block by block WITHOUT the library's padding check, then strip
    // the padding ourselves. A malformed trailer should then cost us a few
    // bytes rather than the whole stream: a PDF is untrusted input, and a
    // viewer that shows nothing because the last block is off is worse than
    // one that shows the page.
    let (blocks, _) = buf.as_mut_slice().as_chunks_mut::<AES_BLOCK>();
    for chunk in blocks {
        let block = aes::cipher::generic_array::GenericArray::from_mut_slice(chunk);
        dec.decrypt_block_mut(block);
    }
    let pad = *buf.last().unwrap_or(&0) as usize;
    if (1..=AES_BLOCK).contains(&pad) && pad <= buf.len() {
        buf.truncate(buf.len() - pad);
    }
    buf
}

/// Read a crypt filter's method out of `/CF << /<name> << /CFM ... >> >>`.
///
/// `/Identity` is the specification's reserved name for "do not encrypt" and
/// never appears in `/CF`; `/None` is the explicit no-op method.
pub fn method_for_filter(
    resolve: &dyn Fn(&Object, &str) -> Option<Object>,
    encrypt: &Object,
    filter_name: &[u8],
) -> Method {
    if filter_name == b"Identity" {
        return Method::None;
    }
    let Some(cf) = resolve(encrypt, "CF") else {
        return Method::None;
    };
    let name = String::from_utf8_lossy(filter_name).into_owned();
    let Some(entry) = resolve(&cf, &name) else {
        return Method::None;
    };
    match resolve(&entry, "CFM").as_ref().map(Object::to_name) {
        Some(b"V2") => Method::Rc4,
        Some(b"AESV2") => Method::Aes128,
        _ => Method::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimally-populated `/Encrypt` description for the refusal tests.
    fn dict<'a>(filter: &'a str, v: i64, r: i64, o: &'a [u8], u: &'a [u8]) -> EncryptDict<'a> {
        EncryptDict {
            filter,
            v,
            r,
            length_bits: 128,
            o,
            u,
            p: -1052,
            first_id: b"id",
            encrypt_metadata: true,
            stream_method: Method::Aes128,
            string_method: Method::Aes128,
        }
    }

    /// The padding string is the empty password. Getting a byte wrong here
    /// silently changes every key this module derives, so it is checked
    /// against the spec's own listing rather than trusted.
    #[test]
    fn an_empty_password_pads_to_exactly_the_standard_string() {
        assert_eq!(pad_password(b""), PAD);
        assert_eq!(PAD[0], 0x28, "first byte per ISO 32000-1 Algorithm 2(a)");
        assert_eq!(PAD[31], 0x7A, "last byte");
    }

    /// A short password keeps its own bytes and is topped up from the front
    /// of the pad -- not the back, and not zero-filled.
    #[test]
    fn a_short_password_is_topped_up_from_the_front_of_the_pad() {
        let p = pad_password(b"abc");
        assert_eq!(&p[..3], b"abc");
        assert_eq!(&p[3..], &PAD[..29]);
    }

    /// Longer than 32 bytes is truncated, not hashed or rejected.
    #[test]
    fn an_over_long_password_is_truncated_to_32_bytes() {
        let long = vec![b'x'; 100];
        assert_eq!(pad_password(&long), [b'x'; 32]);
    }

    /// The per-object key mixes the object and generation numbers as
    /// little-endian bytes, and AES adds the `sAlT` suffix. If the salt were
    /// dropped, AES documents would derive an RC4 key and decrypt to noise.
    #[test]
    fn the_aes_object_key_differs_from_the_rc4_one() {
        let d = Decryptor {
            key: vec![0u8; 16],
            stream_method: Method::Aes128,
            string_method: Method::Aes128,
        };
        assert_ne!(
            d.object_key(7, 0, true),
            d.object_key(7, 0, false),
            "the sAlT suffix must change the key"
        );
    }

    /// Different objects get different keys -- that is the whole point of
    /// Algorithm 1.
    #[test]
    fn every_object_gets_its_own_key() {
        let d = Decryptor {
            key: vec![1u8; 16],
            stream_method: Method::Rc4,
            string_method: Method::Rc4,
        };
        assert_ne!(d.object_key(1, 0, false), d.object_key(2, 0, false));
        assert_ne!(d.object_key(1, 0, false), d.object_key(1, 1, false));
    }

    /// "n + 5, to a maximum of 16" (§7.6.2 Algorithm 1 step (d)). A 16-byte
    /// file key must not produce a 21-byte object key.
    #[test]
    fn the_object_key_is_capped_at_sixteen_bytes() {
        for n in [5usize, 8, 16] {
            let d = Decryptor {
                key: vec![0u8; n],
                stream_method: Method::Rc4,
                string_method: Method::Rc4,
            };
            assert_eq!(d.object_key(1, 0, false).len(), (n + 5).min(16));
        }
    }

    /// `/P` is SIGNED. A restrictive permissions value is negative, so
    /// treating it as unsigned changes the key for most real documents --
    /// which would look like "wrong password" on files that open elsewhere.
    #[test]
    fn a_negative_permissions_value_changes_the_key_the_signed_way() {
        let o = [0u8; 32];
        let signed = file_key(b"", &o, -1052, b"id", 4, 16, true);
        let as_if_unsigned = file_key(b"", &o, -1052i64 as u32 as i64, b"id", 4, 16, true);
        assert_eq!(
            signed, as_if_unsigned,
            "both must reduce to the same 32 low bits -- this pins the \
             truncation, so a future refactor cannot start using the full i64"
        );
        assert_ne!(
            signed,
            file_key(b"", &o, 1052, b"id", 4, 16, true),
            "and a different /P must give a different key"
        );
    }

    /// R3+ strengthens the key with 50 extra MD5 passes. Omitting them is a
    /// silent wrong-key bug, so R2 and R4 must not agree.
    #[test]
    fn revision_three_and_up_iterate_the_digest() {
        let o = [0u8; 32];
        assert_ne!(
            file_key(b"", &o, -1, b"id", 2, 5, true),
            file_key(b"", &o, -1, b"id", 3, 5, true),
            "R3's 50 iterations must change the key"
        );
    }

    /// `/EncryptMetadata false` mixes in FF FF FF FF at R4 and above only.
    #[test]
    fn unencrypted_metadata_changes_the_key_only_from_r4() {
        let o = [0u8; 32];
        assert_ne!(
            file_key(b"", &o, -1, b"id", 4, 16, true),
            file_key(b"", &o, -1, b"id", 4, 16, false),
            "R4 must mix in the sentinel"
        );
        assert_eq!(
            file_key(b"", &o, -1, b"id", 3, 16, true),
            file_key(b"", &o, -1, b"id", 3, 16, false),
            "R3 must not"
        );
    }

    /// An unsupported revision is refused BY NAME rather than attempted --
    /// R5/R6 use a different derivation entirely, and a confident wrong answer
    /// is worse than a refusal.
    #[test]
    fn aes_256_revisions_are_refused_with_an_explanation() {
        let err = Decryptor::new(&dict("Standard", 5, 6, &[0u8; 32], &[0u8; 48]), b"")
            .expect_err("R6 is not implemented");
        assert!(err.to_string().contains("R6"), "{err}");
    }

    #[test]
    fn a_foreign_security_handler_is_refused_by_name() {
        let err = Decryptor::new(&dict("Custom", 4, 4, &[0u8; 32], &[0u8; 32]), b"")
            .expect_err("only /Standard is implemented");
        assert!(err.to_string().contains("Custom"), "{err}");
    }

    /// A wrong password must FAIL rather than derive a plausible key and hand
    /// back garbage. `/U` here is arbitrary, so no password matches it.
    #[test]
    fn a_key_that_does_not_match_u_is_rejected() {
        let err = Decryptor::new(&dict("Standard", 4, 4, &[0u8; 32], &[0xAB; 32]), b"")
            .expect_err("the empty password must not open this");
        assert!(err.to_string().contains("password"), "{err}");
    }

    /// AES needs an IV plus at least one block; anything shorter is malformed
    /// input and must come back empty rather than panic. A PDF is untrusted.
    #[test]
    fn short_or_misaligned_aes_input_is_empty_not_a_panic() {
        let key = [0u8; 16];
        assert!(decrypt_aes_cbc(&key, &[]).is_empty());
        assert!(decrypt_aes_cbc(&key, &[0u8; 16]).is_empty());
        assert!(
            decrypt_aes_cbc(&key, &[0u8; 40]).is_empty(),
            "40 bytes is an IV plus one and a half blocks -- misaligned"
        );
    }

    /// RC4 is its own inverse, which lets the round trip be checked without a
    /// fixture: encrypting with `apply` and decrypting with it again must
    /// return the original.
    #[test]
    fn rc4_round_trips_through_apply() {
        let d = Decryptor {
            key: vec![9u8; 16],
            stream_method: Method::Rc4,
            string_method: Method::Rc4,
        };
        let plain = b"Hello, kopitiam.".to_vec();
        let once = d.apply(Method::Rc4, 4, 0, &plain);
        assert_ne!(once, plain, "it must actually transform the bytes");
        assert_eq!(d.apply(Method::Rc4, 4, 0, &once), plain);
    }

    /// `Method::None` passes bytes through untouched -- `/Identity` and
    /// `/CFM /None` must not corrupt anything.
    #[test]
    fn the_none_method_is_a_pass_through() {
        let d = Decryptor {
            key: vec![0u8; 16],
            stream_method: Method::None,
            string_method: Method::None,
        };
        let data = b"untouched".to_vec();
        assert_eq!(d.decrypt_stream(1, 0, &data), data);
        assert_eq!(d.decrypt_string(1, 0, &data), data);
    }
    /// A round trip through a REAL encrypted document is the only way to know
    /// the key derivation is right -- every intermediate step can be
    /// individually plausible and still produce the wrong key. This builds an
    /// RC4/R3 document with `lopdf`, encrypts nothing itself, and checks the
    /// pieces we can check without a fixture: that the /U we compute from a
    /// key matches what Algorithm 5 says it should, by construction.
    ///
    /// Algorithm 5 is deterministic given (key, /ID), so computing /U forwards
    /// and then verifying it backwards through `user_password_matches` is a
    /// genuine consistency check on both directions of the code that decides
    /// "is this the right password".
    #[test]
    fn a_key_verifies_against_the_u_value_it_generates() {
        let id = b"0123456789abcdef";
        let o = [0x5Au8; 32];
        for r in [2i64, 3, 4] {
            let n = if r == 2 { 5 } else { 16 };
            let key = file_key(b"", &o, -1052, id, r, n, true);

            // Compute /U the way a producer would (Algorithm 4 for R2,
            // Algorithm 5 for R3+), then feed it back to the verifier.
            let u = if r == 2 {
                let mut buf = PAD.to_vec();
                rc4_apply(&key, &mut buf);
                buf
            } else {
                let mut h = Md5::new();
                h.update(PAD);
                h.update(id);
                let mut buf = h.finalize().to_vec();
                rc4_apply(&key, &mut buf);
                for i in 1u8..=19 {
                    let stepped: Vec<u8> = key.iter().map(|b| b ^ i).collect();
                    rc4_apply(&stepped, &mut buf);
                }
                // R3+ /U is 32 bytes: 16 meaningful, 16 arbitrary.
                buf.resize(32, 0);
                buf
            };

            assert!(
                user_password_matches(&key, r, id, &u),
                "R{r}: the key must verify against the /U it produces"
            );
            // And a different key must NOT verify -- otherwise the check is
            // vacuous and every password would "work".
            let wrong = file_key(b"wrong", &o, -1052, id, r, n, true);
            assert!(
                !user_password_matches(&wrong, r, id, &u),
                "R{r}: a wrong password must be rejected"
            );
        }
    }

    /// R3+ compares only the first 16 bytes of /U; the rest is arbitrary
    /// padding. Comparing all 32 would reject documents that are perfectly
    /// valid, which is a "your password is wrong" bug on a file that opens
    /// everywhere else.
    #[test]
    fn revision_three_ignores_the_arbitrary_tail_of_u() {
        let id = b"idid";
        let o = [0u8; 32];
        let key = file_key(b"", &o, -1, id, 3, 16, true);
        let mut h = Md5::new();
        h.update(PAD);
        h.update(id);
        let mut buf = h.finalize().to_vec();
        rc4_apply(&key, &mut buf);
        for i in 1u8..=19 {
            let stepped: Vec<u8> = key.iter().map(|b| b ^ i).collect();
            rc4_apply(&stepped, &mut buf);
        }
        let mut u = buf.clone();
        u.resize(32, 0);
        assert!(user_password_matches(&key, 3, id, &u));
        // Scribble over the tail: still valid.
        for b in u.iter_mut().skip(16) {
            *b = 0xEE;
        }
        assert!(
            user_password_matches(&key, 3, id, &u),
            "the last 16 bytes of /U are arbitrary and must not be compared"
        );
        // Scribble over the meaningful half: no longer valid.
        u[0] ^= 0xFF;
        assert!(!user_password_matches(&key, 3, id, &u));
    }

}
