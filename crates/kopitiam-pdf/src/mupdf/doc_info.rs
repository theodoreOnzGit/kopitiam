//! Document metadata — the `/Info` dictionary (PDF 32000-1:2008 §14.3.3) and
//! the PDF date strings inside it (§7.9.4).
//!
//! # Scope: the PDF half only
//!
//! This reads **what the file says about itself** and stops there. It does not
//! guess a title from cover text, split an `/Author` string into people, or
//! decide that a producer wrote junk in `/Title`. That is bibliographic
//! interpretation, and in this project it belongs to `kovan`
//! (`kovan-literature`), which owns literature metadata and digitisation —
//! kopitiam owns PDF. The division matters because the two answer different
//! questions: this module answers *"what is in the `/Info` dict"*, kovan
//! answers *"who actually wrote this paper"*, and those genuinely differ (an
//! Elsevier PDF's `/Title` is the article PII, not the title).
//!
//! So everything here is faithful and lossless: a caller that wants heuristics
//! layers them on top, and a caller that wants the raw truth already has it.
//!
//! # What this does better than a naive `/Info` read
//!
//! * **Text strings are decoded properly** (§7.9.2.2). A PDF text string is
//!   either UTF-16BE with a `FE FF` byte-order mark, UTF-8 with an `EF BB BF`
//!   BOM (PDF 2.0), or otherwise **PDFDocEncoding** — which is *not* Latin-1.
//!   They agree over ASCII and over most of the upper range, but the
//!   `0x80`-`0x9F` band is where they part company — and that band is exactly
//!   where the typographic punctuation lives. `0x90` is U+2019 RIGHT SINGLE
//!   QUOTATION MARK in PDFDocEncoding (Table D.2) and a C1 control in
//!   Latin-1, so a naive read turns *"Rayleigh’s"* into a control character.
//!   Note it is **also not CP1252/WinAnsi**, which is the easy second mistake:
//!   WinAnsi puts that same quote at `0x92`, where PDFDocEncoding has U+2122
//!   TRADEMARK. Decoding `/Info` with the wrong one of the three silently
//!   corrupts punctuation rather than failing.
//!   [`super::encodings::pdf_doc_unicode`] already holds MuPDF's real table,
//!   so this uses it.
//! * **Dates are parsed in full**, not reduced to a year. `/CreationDate` is
//!   `D:YYYYMMDDHHmmSSOHH'mm'` with every field after the year optional, and
//!   the timezone is real information (a scanned-at-2am timestamp is a
//!   different fact in UTC+8 than in UTC). Callers who only want a year take
//!   [`PdfDate::year`].
//! * **Unknown keys survive.** Producers write their own `/Info` keys
//!   (`/SourceModified`, `/Company`, LaTeX's `/PTEX.Fullbanner`), and a struct
//!   with six fixed fields silently drops them. They land in
//!   [`DocumentInfo::custom`] instead, so nothing in the file is lost.

use super::encodings::pdf_doc_unicode;
use super::object::Object;
use super::xref::PdfDocument;

/// A PDF date (§7.9.4), as written — **not** normalised to UTC.
///
/// Only `year` is required; a producer may stop at any field. Absent fields
/// take their PDF-specified defaults (month/day 1, time 00:00:00), so the
/// struct is always a usable instant, with [`PdfDate::precision`] saying how
/// much of it the file actually stated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PdfDate {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    /// Offset from UTC in minutes, east positive (`+08'00'` -> `480`).
    /// `None` when the file gave no timezone, which per §7.9.4 means unknown —
    /// deliberately not defaulted to UTC, since that would invent a fact.
    pub utc_offset_minutes: Option<i32>,
    /// How many leading fields the file actually supplied.
    pub precision: DatePrecision,
}

/// How much of a [`PdfDate`] the file really stated, as opposed to defaulted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DatePrecision {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
}

/// The `/Info` dictionary's contents. Every field is optional because every
/// `/Info` key is optional — an absent `/Info` yields `DocumentInfo::default()`,
/// which is a legal document, not an error.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DocumentInfo {
    pub title: Option<String>,
    /// `/Author`, verbatim. One string even when it names several people —
    /// splitting it is bibliographic work, and the delimiter is not
    /// standardised (`;`, `,`, ` and `). Left whole on purpose.
    pub author: Option<String>,
    pub subject: Option<String>,
    /// `/Keywords`, verbatim, unsplit — same reasoning as `author`.
    pub keywords: Option<String>,
    /// The application that authored the original document (e.g. `LaTeX`).
    pub creator: Option<String>,
    /// The application that produced the PDF itself (e.g. `pdfTeX-1.40.25`).
    pub producer: Option<String>,
    pub creation_date: Option<PdfDate>,
    pub mod_date: Option<PdfDate>,
    /// `/Trapped`: `Some(true)`/`Some(false)` for `/True`/`/False`, `None` for
    /// `/Unknown` or absent — the PDF default is `/Unknown`, which is
    /// genuinely "not stated", not `false`.
    pub trapped: Option<bool>,
    /// Any `/Info` key not named above, in file order, with its value decoded
    /// as a text string. Non-string values are skipped (there is nothing
    /// meaningful to render), so this never fabricates content.
    pub custom: Vec<(String, String)>,
}

/// The keys [`DocumentInfo`] has dedicated fields for; everything else in the
/// dictionary goes to [`DocumentInfo::custom`].
const KNOWN_KEYS: &[&str] = &[
    "Title",
    "Author",
    "Subject",
    "Keywords",
    "Creator",
    "Producer",
    "CreationDate",
    "ModDate",
    "Trapped",
];

/// Read `doc`'s `/Info` dictionary.
///
/// Never fails: a document with no `/Info`, an `/Info` that is not a dict, or
/// one whose reference dangles all yield an empty [`DocumentInfo`]. Metadata
/// is decoration on a PDF — refusing to open a readable document because its
/// `/Info` is malformed would be the wrong trade, and this is the same stance
/// the rest of the reader takes toward broken optional structure.
pub fn document_info(doc: &PdfDocument) -> DocumentInfo {
    let mut info = DocumentInfo::default();
    let Some(info_ref) = doc.trailer().dict_gets("Info") else {
        return info;
    };
    let Ok(dict) = doc.resolve(info_ref) else {
        return info;
    };
    if !dict.is_dict() {
        return info;
    }

    for i in 0..dict.dict_len() {
        let (Some(key), Some(value)) = (dict.dict_get_key(i), dict.dict_get_val(i)) else {
            continue;
        };
        let key = String::from_utf8_lossy(key).into_owned();
        // Resolve: `/Info` values are usually direct, but nothing forbids an
        // indirect string, and a dangling one must not abort the whole read.
        let value = doc.resolve(value).unwrap_or(Object::Null);
        match key.as_str() {
            "Trapped" => {
                info.trapped = match value.to_name() {
                    b"True" => Some(true),
                    b"False" => Some(false),
                    _ => None, // `/Unknown`, or something unexpected.
                };
            }
            "CreationDate" => info.creation_date = date_field(&value),
            "ModDate" => info.mod_date = date_field(&value),
            _ => {
                let Some(text) = text_field(&value) else {
                    continue;
                };
                match key.as_str() {
                    "Title" => info.title = Some(text),
                    "Author" => info.author = Some(text),
                    "Subject" => info.subject = Some(text),
                    "Keywords" => info.keywords = Some(text),
                    "Creator" => info.creator = Some(text),
                    "Producer" => info.producer = Some(text),
                    _ => {
                        debug_assert!(!KNOWN_KEYS.contains(&key.as_str()));
                        info.custom.push((key, text));
                    }
                }
            }
        }
    }
    info
}

/// A string-valued `/Info` entry, decoded and trimmed. `None` for a
/// non-string, or for a value that is empty once trimmed — an empty `/Title`
/// is not a title, and reporting `Some("")` would make every caller re-check.
fn text_field(value: &Object) -> Option<String> {
    if !value.is_string() {
        return None;
    }
    let text = decode_text_string(value.to_string_bytes());
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn date_field(value: &Object) -> Option<PdfDate> {
    value
        .is_string()
        .then(|| decode_text_string(value.to_string_bytes()))
        .and_then(|s| parse_pdf_date(&s))
}

/// Decode a PDF **text string** (§7.9.2.2) to Rust `String`.
///
/// Three encodings, distinguished by leading byte-order mark:
/// * `FE FF` — UTF-16BE. Unpaired surrogates become U+FFFD rather than an
///   error, since a metadata string is not worth failing a document over.
/// * `EF BB BF` — UTF-8 (PDF 2.0, §7.9.2.2.1). Invalid sequences become
///   U+FFFD.
/// * anything else — **PDFDocEncoding**, byte by byte, via
///   [`super::encodings::pdf_doc_unicode`]. Undefined slots are dropped: the
///   table has genuine holes, and a hole means "no character", not U+FFFD.
pub fn decode_text_string(bytes: &[u8]) -> String {
    if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        return decode_utf16be(rest);
    }
    if let Some(rest) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8_lossy(rest).into_owned();
    }
    bytes.iter().filter_map(|&b| pdf_doc_unicode(b)).collect()
}

fn decode_utf16be(bytes: &[u8]) -> String {
    // An odd trailing byte is malformed; drop it rather than reading past the
    // end. `chunks_exact` does exactly that and keeps the pairing honest.
    let units: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| u16::from_be_bytes(*p))
        .collect();
    char::decode_utf16(units)
        .map(|r| r.unwrap_or(char::REPLACEMENT_CHARACTER))
        .collect()
}

/// Parse a PDF date string (§7.9.4): `D:YYYYMMDDHHmmSSOHH'mm'`.
///
/// The `D:` prefix is required by the spec but widely omitted in the wild, so
/// it is optional here. Every field after the four-digit year is optional, and
/// parsing stops at the first thing that is not what it expects rather than
/// failing — a producer that writes `D:2024` is giving a real, if coarse,
/// answer, and [`PdfDate::precision`] records how coarse.
///
/// Returns `None` only when there is no four-digit year to be had at all.
///
/// The timezone forms accepted are `Z`, `+HH'mm'`, `-HH'mm'`, and the sloppy
/// variants that drop the apostrophes or the minutes — all of which appear in
/// real files.
pub fn parse_pdf_date(s: &str) -> Option<PdfDate> {
    let s = s.trim();
    let s = s.strip_prefix("D:").unwrap_or(s);
    let b = s.as_bytes();

    // Two-digit field at `at`, only if both bytes are ASCII digits.
    let field = |at: usize| -> Option<u8> {
        let pair = b.get(at..at + 2)?;
        if !pair.iter().all(u8::is_ascii_digit) {
            return None;
        }
        std::str::from_utf8(pair).ok()?.parse().ok()
    };

    let year_digits = b.get(0..4)?;
    if !year_digits.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let year: i32 = std::str::from_utf8(year_digits).ok()?.parse().ok()?;

    let mut precision = DatePrecision::Year;
    // Each field is taken only if the previous one was present, so a garbled
    // middle cannot make a later field jump forward into the wrong slot.
    let month = field(4).inspect(|_| precision = DatePrecision::Month);
    let day = month
        .and_then(|_| field(6))
        .inspect(|_| precision = DatePrecision::Day);
    let hour = day
        .and_then(|_| field(8))
        .inspect(|_| precision = DatePrecision::Hour);
    let minute = hour
        .and_then(|_| field(10))
        .inspect(|_| precision = DatePrecision::Minute);
    let second = minute
        .and_then(|_| field(12))
        .inspect(|_| precision = DatePrecision::Second);

    // The timezone begins wherever the numeric run stopped.
    let tz_at = match precision {
        DatePrecision::Year => 4,
        DatePrecision::Month => 6,
        DatePrecision::Day => 8,
        DatePrecision::Hour => 10,
        DatePrecision::Minute => 12,
        DatePrecision::Second => 14,
    };

    Some(PdfDate {
        year,
        // Clamped to legal ranges: a producer writing month 00 or 13 is
        // reporting a broken date, and silently passing it on would push the
        // breakage into every consumer's calendar arithmetic.
        month: month.unwrap_or(1).clamp(1, 12),
        day: day.unwrap_or(1).clamp(1, 31),
        hour: hour.unwrap_or(0).min(23),
        minute: minute.unwrap_or(0).min(59),
        // 60 is legal: a leap second (§7.9.4 defers to ISO 8601).
        second: second.unwrap_or(0).min(60),
        utc_offset_minutes: parse_utc_offset(&s[tz_at.min(s.len())..]),
        precision,
    })
}

/// The trailing timezone of a PDF date. `None` means the file did not say.
fn parse_utc_offset(s: &str) -> Option<i32> {
    let s = s.trim();
    let (sign, rest) = match s.as_bytes().first()? {
        b'Z' => return Some(0),
        b'+' => (1, &s[1..]),
        b'-' => (-1, &s[1..]),
        _ => return None,
    };
    let digits: Vec<u8> = rest.bytes().filter(u8::is_ascii_digit).collect();
    if digits.len() < 2 {
        return None;
    }
    let num = |d: &[u8]| -> i32 { std::str::from_utf8(d).unwrap_or("0").parse().unwrap_or(0) };
    let hours = num(&digits[..2]);
    let minutes = if digits.len() >= 4 {
        num(&digits[2..4])
    } else {
        0
    };
    // Real offsets run -12:00..+14:00; anything past that is corrupt, and a
    // wild offset would silently shift a date by days.
    let total = sign * (hours * 60 + minutes);
    (-12 * 60..=14 * 60).contains(&total).then_some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- text strings ------------------------------------------------------

    /// The divergence this module exists to get right, in the `0x80`-`0x9F`
    /// band where PDFDocEncoding, Latin-1 and WinAnsi all disagree.
    ///
    /// Codes are from PDF 32000-1:2008 Table D.2. In Latin-1 every one of
    /// these is a C1 control, so a Latin-1 read corrupts them outright.
    #[test]
    fn pdfdocencoding_is_not_latin1() {
        assert_eq!(
            decode_text_string(b"Rayleigh\x90s number"),
            "Rayleigh\u{2019}s number",
            "0x90 is RIGHT SINGLE QUOTATION MARK here"
        );
        assert_eq!(decode_text_string(b"\x8Dquoted\x8E"), "\u{201C}quoted\u{201D}");
        assert_eq!(decode_text_string(b"em\x84dash"), "em\u{2014}dash");
        assert_eq!(decode_text_string(b"en\x85dash"), "en\u{2013}dash");
    }

    /// The second, easier mistake: assuming PDFDocEncoding *is* CP1252
    /// because both are "the non-Latin-1 one". They differ in this band, so
    /// picking the wrong one corrupts punctuation silently rather than
    /// failing. `0x92` is the sharpest case.
    #[test]
    fn pdfdocencoding_is_not_cp1252_either() {
        assert_eq!(
            decode_text_string(b"\x92"),
            "\u{2122}",
            "0x92 is TRADEMARK in PDFDocEncoding; CP1252 puts a quote here"
        );
        // Ligatures, which CP1252 has nowhere at all.
        assert_eq!(decode_text_string(b"\x93\x94"), "\u{FB01}\u{FB02}");
    }

    /// The table has genuine holes (0x7F, and the undefined slots); a hole
    /// means "no character", so it is dropped rather than turned into U+FFFD.
    #[test]
    fn undefined_slots_are_dropped_not_replaced() {
        assert_eq!(decode_text_string(b"a\x7Fb"), "ab");
    }

    #[test]
    fn utf16be_bom_is_decoded() {
        // FE FF then "Hi" in UTF-16BE.
        assert_eq!(decode_text_string(b"\xFE\xFF\x00H\x00i"), "Hi");
        // Non-BMP via a surrogate pair: U+1D453 MATHEMATICAL ITALIC SMALL F.
        assert_eq!(
            decode_text_string(b"\xFE\xFF\xD8\x35\xDC\x53"),
            "\u{1D453}"
        );
    }

    /// A truncated UTF-16 string must not panic or read past the end — it is
    /// exactly the kind of thing a damaged file contains.
    #[test]
    fn malformed_utf16_degrades_instead_of_panicking() {
        // Odd trailing byte.
        assert_eq!(decode_text_string(b"\xFE\xFF\x00H\x00"), "H");
        // Unpaired high surrogate.
        assert_eq!(decode_text_string(b"\xFE\xFF\xD8\x35"), "\u{FFFD}");
    }

    #[test]
    fn utf8_bom_is_decoded() {
        assert_eq!(decode_text_string("\u{FEFF}café".as_bytes()), "café");
    }

    // -- dates -------------------------------------------------------------

    #[test]
    fn full_date_with_timezone() {
        let d = parse_pdf_date("D:20240315142530+08'00'").expect("a date");
        assert_eq!(
            (d.year, d.month, d.day, d.hour, d.minute, d.second),
            (2024, 3, 15, 14, 25, 30)
        );
        assert_eq!(d.utc_offset_minutes, Some(480), "SGT is +08:00");
        assert_eq!(d.precision, DatePrecision::Second);
    }

    /// Every field after the year is optional, and the precision must say how
    /// far the file actually went — this is the information kovan's
    /// year-only reduction throws away.
    #[test]
    fn partial_dates_report_their_precision() {
        let y = parse_pdf_date("D:2024").expect("year only");
        assert_eq!(y.precision, DatePrecision::Year);
        assert_eq!((y.year, y.month, y.day), (2024, 1, 1), "defaults per spec");
        assert_eq!(y.utc_offset_minutes, None, "unknown, not assumed UTC");

        assert_eq!(parse_pdf_date("D:202403").unwrap().precision, DatePrecision::Month);
        assert_eq!(parse_pdf_date("D:20240315").unwrap().precision, DatePrecision::Day);
        assert_eq!(
            parse_pdf_date("D:2024031514").unwrap().precision,
            DatePrecision::Hour
        );
    }

    #[test]
    fn timezone_variants_seen_in_the_wild() {
        let z = parse_pdf_date("D:20240315142530Z").unwrap();
        assert_eq!(z.utc_offset_minutes, Some(0));
        // Apostrophes dropped.
        assert_eq!(
            parse_pdf_date("D:20240315142530-0500").unwrap().utc_offset_minutes,
            Some(-300)
        );
        // Hours only.
        assert_eq!(
            parse_pdf_date("D:20240315142530+08").unwrap().utc_offset_minutes,
            Some(480)
        );
        // Nepal, to prove non-hour offsets survive.
        assert_eq!(
            parse_pdf_date("D:20240315142530+05'45'").unwrap().utc_offset_minutes,
            Some(345)
        );
    }

    /// The `D:` prefix is mandated by the spec and routinely omitted.
    #[test]
    fn missing_d_prefix_is_tolerated() {
        assert_eq!(parse_pdf_date("20240315").unwrap().year, 2024);
    }

    #[test]
    fn junk_yields_none() {
        assert!(parse_pdf_date("").is_none());
        assert!(parse_pdf_date("D:").is_none());
        assert!(parse_pdf_date("not a date").is_none());
        assert!(parse_pdf_date("D:20x4").is_none(), "year must be 4 digits");
    }

    /// A broken field must not propagate into consumers' calendar maths, and a
    /// wild offset must not silently shift the date by days.
    #[test]
    fn out_of_range_fields_are_clamped_or_refused() {
        let d = parse_pdf_date("D:20240015").unwrap();
        assert_eq!(d.month, 1, "month 00 clamps into range");
        assert_eq!(
            parse_pdf_date("D:20240315142530+99'00'").unwrap().utc_offset_minutes,
            None,
            "an impossible offset is dropped, not applied"
        );
        // A leap second is legal and must survive.
        assert_eq!(parse_pdf_date("D:20161231235960Z").unwrap().second, 60);
    }

    /// A garbled middle field must not let a later field slide into the wrong
    /// slot — `precision` stops where the damage starts.
    #[test]
    fn a_garbled_field_stops_the_parse_rather_than_shifting_it() {
        let d = parse_pdf_date("D:2024XX15").expect("year still parses");
        assert_eq!(d.precision, DatePrecision::Year);
        assert_eq!(d.day, 1, "the '15' must NOT be read as a day");
    }
}
