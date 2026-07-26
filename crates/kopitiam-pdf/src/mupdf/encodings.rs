//! Ported from MuPDF `source/fitz/encodings.c` + `source/fitz/encodings.h`
//! (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the algorithms and numeric
//! behaviour follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # What this module is
//!
//! `encodings.c` includes the big static `encodings.h` table header. This file
//! ports the subset that defines the **standard PDF simple-font base
//! encodings** -- the 256-entry `code -> glyph-name` / `code -> unicode` tables
//! a PDF font dictionary's `/Encoding` names (`StandardEncoding`,
//! `WinAnsiEncoding`, `MacRomanEncoding`, `PDFDocEncoding`, plus MuPDF's
//! `MacExpertEncoding`). These are ground-truth data: reproduced entry-for-entry
//! from `encodings.h`, not paraphrased.
//!
//! ## Representation matches the C
//!
//! MuPDF stores two shapes and we keep both:
//!
//! * The glyph-name tables are C `const char *[256]` (Adobe glyph names such as
//!   `"space"`, `"Aacute"`). We keep them as [`&'static str`][str]. C uses the
//!   `_notdef` macro (`#define _notdef NULL`) for an undefined slot; we mirror it
//!   with the empty string [`ND`] (a real glyph name is never empty), and the
//!   typed accessor [`BaseEncoding::glyph_name`] turns it back into `None`.
//! * `PDFDocEncoding` is a C `unsigned short[256]` of unicode code points; we
//!   keep it as `[u16; 256]` ([`PDF_DOC`]). A `0` entry is an undefined slot
//!   (MuPDF's comment: "0x0 to 0x17 except \t, \n and \r are really undefined").
//!
//! ## Not ported from `encodings.h` (deliberately)
//!
//! `encodings.h` also carries the legacy single-byte cmaps
//! (`fz_unicode_from_dos_437`, `fz_unicode_from_true_type_glyph_index`, the
//! `iso8859_1` / `iso8859_7` / `koi8u` / `windows_125x` unicode + glyph-name +
//! reverse `*_from_unicode` tables) plus the glyph-name lookup logic and
//! small-caps tables from `encodings.c`. Those are not the PDF simple-font base
//! encodings this task targets; port them if and when a later module needs them.
//! `SymbolEncoding` / `ZapfDingbatsEncoding` have **no table in this C source**
//! (MuPDF resolves them through the built-in Symbol / ZapfDingbats font cmaps,
//! not `encodings.h`), so there is nothing to reproduce here.

// ---------------------------------------------------------------------------
// Undefined-slot sentinel
// ---------------------------------------------------------------------------

/// The `_notdef` sentinel for the glyph-name tables. MuPDF's `encodings.h`
/// opens with `#define _notdef NULL`; since a real Adobe glyph name is never the
/// empty string, we represent that `NULL` as `""`. [`BaseEncoding::glyph_name`]
/// maps it back to `None`.
///
// MuPDF: `#define _notdef NULL` (encodings.h:23)
pub const ND: &str = "";

// ---------------------------------------------------------------------------
// Typed accessor
// ---------------------------------------------------------------------------

/// The glyph-name-based standard PDF simple-font base encodings.
///
/// Each variant selects one 256-entry `code -> glyph-name` table from
/// `encodings.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseEncoding {
    /// `StandardEncoding` (Adobe StandardEncoding) -- [`STANDARD`].
    Standard,
    /// `WinAnsiEncoding` -- [`WIN_ANSI`].
    WinAnsi,
    /// `MacRomanEncoding` -- [`MAC_ROMAN`].
    MacRoman,
    /// `MacExpertEncoding` -- [`MAC_EXPERT`].
    MacExpert,
}

impl BaseEncoding {
    /// The raw 256-entry glyph-name table backing this encoding (with [`ND`] in
    /// undefined slots).
    pub fn table(self) -> &'static [&'static str; 256] {
        match self {
            BaseEncoding::Standard => &STANDARD,
            BaseEncoding::WinAnsi => &WIN_ANSI,
            BaseEncoding::MacRoman => &MAC_ROMAN,
            BaseEncoding::MacExpert => &MAC_EXPERT,
        }
    }

    /// The Adobe glyph name for `code`, or `None` for an undefined (`_notdef`)
    /// slot.
    pub fn glyph_name(self, code: u8) -> Option<&'static str> {
        let name = self.table()[code as usize];
        if name.is_empty() { None } else { Some(name) }
    }
}

/// `PDFDocEncoding`: the unicode code point for `code`, or `None` for an
/// undefined slot (a `0` entry).
///
// MuPDF: fz_unicode_from_pdf_doc_encoding (encodings.h:25)
pub fn pdf_doc_unicode(code: u8) -> Option<char> {
    match PDF_DOC[code as usize] {
        0 => None,
        u => char::from_u32(u as u32),
    }
}

// ---------------------------------------------------------------------------
// PDFDocEncoding: code -> unicode (unsigned short)
// ---------------------------------------------------------------------------

/// `PDFDocEncoding` code -> unicode code point (`0` = undefined slot).
///
// MuPDF: fz_unicode_from_pdf_doc_encoding (encodings.h:25)
pub static PDF_DOC: [u16; 256] = [
    // 0x0 to 0x17 except \t, \n and \r are really undefined
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x02d8, 0x02c7, 0x02c6, 0x02d9, 0x02dd, 0x02db, 0x02da, 0x02dc,
    0x0020, 0x0021, 0x0022, 0x0023, 0x0024, 0x0025, 0x0026, 0x0027,
    0x0028, 0x0029, 0x002a, 0x002b, 0x002c, 0x002d, 0x002e, 0x002f,
    0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037,
    0x0038, 0x0039, 0x003a, 0x003b, 0x003c, 0x003d, 0x003e, 0x003f,
    0x0040, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047,
    0x0048, 0x0049, 0x004a, 0x004b, 0x004c, 0x004d, 0x004e, 0x004f,
    0x0050, 0x0051, 0x0052, 0x0053, 0x0054, 0x0055, 0x0056, 0x0057,
    0x0058, 0x0059, 0x005a, 0x005b, 0x005c, 0x005d, 0x005e, 0x005f,
    0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067,
    0x0068, 0x0069, 0x006a, 0x006b, 0x006c, 0x006d, 0x006e, 0x006f,
    0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077,
    0x0078, 0x0079, 0x007a, 0x007b, 0x007c, 0x007d, 0x007e, 0x0000,
    0x2022, 0x2020, 0x2021, 0x2026, 0x2014, 0x2013, 0x0192, 0x2044,
    0x2039, 0x203a, 0x2212, 0x2030, 0x201e, 0x201c, 0x201d, 0x2018,
    0x2019, 0x201a, 0x2122, 0xfb01, 0xfb02, 0x0141, 0x0152, 0x0160,
    0x0178, 0x017d, 0x0131, 0x0142, 0x0153, 0x0161, 0x017e, 0x0000,
    0x20ac, 0x00a1, 0x00a2, 0x00a3, 0x00a4, 0x00a5, 0x00a6, 0x00a7,
    0x00a8, 0x00a9, 0x00aa, 0x00ab, 0x00ac, 0x0000, 0x00ae, 0x00af,
    0x00b0, 0x00b1, 0x00b2, 0x00b3, 0x00b4, 0x00b5, 0x00b6, 0x00b7,
    0x00b8, 0x00b9, 0x00ba, 0x00bb, 0x00bc, 0x00bd, 0x00be, 0x00bf,
    0x00c0, 0x00c1, 0x00c2, 0x00c3, 0x00c4, 0x00c5, 0x00c6, 0x00c7,
    0x00c8, 0x00c9, 0x00ca, 0x00cb, 0x00cc, 0x00cd, 0x00ce, 0x00cf,
    0x00d0, 0x00d1, 0x00d2, 0x00d3, 0x00d4, 0x00d5, 0x00d6, 0x00d7,
    0x00d8, 0x00d9, 0x00da, 0x00db, 0x00dc, 0x00dd, 0x00de, 0x00df,
    0x00e0, 0x00e1, 0x00e2, 0x00e3, 0x00e4, 0x00e5, 0x00e6, 0x00e7,
    0x00e8, 0x00e9, 0x00ea, 0x00eb, 0x00ec, 0x00ed, 0x00ee, 0x00ef,
    0x00f0, 0x00f1, 0x00f2, 0x00f3, 0x00f4, 0x00f5, 0x00f6, 0x00f7,
    0x00f8, 0x00f9, 0x00fa, 0x00fb, 0x00fc, 0x00fd, 0x00fe, 0x00ff,
];

// ---------------------------------------------------------------------------
// StandardEncoding: code -> glyph name
// ---------------------------------------------------------------------------

/// `StandardEncoding` (Adobe StandardEncoding) code -> glyph name
/// ([`ND`] = `_notdef`).
///
// MuPDF: fz_glyph_name_from_adobe_standard (encodings.h:62)
pub static STANDARD: [&str; 256] = [
    ND, ND, ND, ND, ND, ND, ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    "space", "exclam", "quotedbl", "numbersign", "dollar", "percent",
    "ampersand", "quoteright", "parenleft", "parenright", "asterisk",
    "plus", "comma", "hyphen", "period", "slash", "zero", "one", "two",
    "three", "four", "five", "six", "seven", "eight", "nine", "colon",
    "semicolon", "less", "equal", "greater", "question", "at", "A", "B",
    "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "bracketleft",
    "backslash", "bracketright", "asciicircum", "underscore", "quoteleft",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "braceleft", "bar", "braceright", "asciitilde", ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    "exclamdown", "cent", "sterling", "fraction", "yen", "florin",
    "section", "currency", "quotesingle", "quotedblleft", "guillemotleft",
    "guilsinglleft", "guilsinglright", "fi", "fl", ND, "endash",
    "dagger", "daggerdbl", "periodcentered", ND, "paragraph",
    "bullet", "quotesinglbase", "quotedblbase", "quotedblright",
    "guillemotright", "ellipsis", "perthousand", ND, "questiondown",
    ND, "grave", "acute", "circumflex", "tilde", "macron", "breve",
    "dotaccent", "dieresis", ND, "ring", "cedilla", ND,
    "hungarumlaut", "ogonek", "caron", "emdash", ND, ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    ND, ND, ND, ND, ND, "AE", ND,
    "ordfeminine", ND, ND, ND, ND, "Lslash", "Oslash",
    "OE", "ordmasculine", ND, ND, ND, ND, ND,
    "ae", ND, ND, ND, "dotlessi", ND, ND,
    "lslash", "oslash", "oe", "germandbls", ND, ND, ND,
    ND,
];

// ---------------------------------------------------------------------------
// MacRomanEncoding: code -> glyph name
// ---------------------------------------------------------------------------

/// `MacRomanEncoding` code -> glyph name ([`ND`] = `_notdef`).
///
// MuPDF: fz_glyph_name_from_mac_roman (encodings.h:100)
pub static MAC_ROMAN: [&str; 256] = [
    ND, ND, ND, ND, ND, ND, ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    "space", "exclam", "quotedbl", "numbersign", "dollar", "percent",
    "ampersand", "quotesingle", "parenleft", "parenright", "asterisk",
    "plus", "comma", "hyphen", "period", "slash", "zero", "one", "two",
    "three", "four", "five", "six", "seven", "eight", "nine", "colon",
    "semicolon", "less", "equal", "greater", "question", "at", "A", "B",
    "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "bracketleft",
    "backslash", "bracketright", "asciicircum", "underscore", "grave", "a",
    "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "braceleft",
    "bar", "braceright", "asciitilde", ND, "Adieresis", "Aring",
    "Ccedilla", "Eacute", "Ntilde", "Odieresis", "Udieresis", "aacute",
    "agrave", "acircumflex", "adieresis", "atilde", "aring", "ccedilla",
    "eacute", "egrave", "ecircumflex", "edieresis", "iacute", "igrave",
    "icircumflex", "idieresis", "ntilde", "oacute", "ograve",
    "ocircumflex", "odieresis", "otilde", "uacute", "ugrave",
    "ucircumflex", "udieresis", "dagger", "degree", "cent", "sterling",
    "section", "bullet", "paragraph", "germandbls", "registered",
    "copyright", "trademark", "acute", "dieresis", ND, "AE", "Oslash",
    ND, "plusminus", ND, ND, "yen", "mu", ND, ND,
    ND, ND, ND, "ordfeminine", "ordmasculine", ND,
    "ae", "oslash", "questiondown", "exclamdown", "logicalnot", ND,
    "florin", ND, ND, "guillemotleft", "guillemotright",
    "ellipsis", "space", "Agrave", "Atilde", "Otilde", "OE", "oe",
    "endash", "emdash", "quotedblleft", "quotedblright", "quoteleft",
    "quoteright", "divide", ND, "ydieresis", "Ydieresis", "fraction",
    "currency", "guilsinglleft", "guilsinglright", "fi", "fl", "daggerdbl",
    "periodcentered", "quotesinglbase", "quotedblbase", "perthousand",
    "Acircumflex", "Ecircumflex", "Aacute", "Edieresis", "Egrave",
    "Iacute", "Icircumflex", "Idieresis", "Igrave", "Oacute",
    "Ocircumflex", ND, "Ograve", "Uacute", "Ucircumflex", "Ugrave",
    "dotlessi", "circumflex", "tilde", "macron", "breve", "dotaccent",
    "ring", "cedilla", "hungarumlaut", "ogonek", "caron",
];

// ---------------------------------------------------------------------------
// MacExpertEncoding: code -> glyph name
// ---------------------------------------------------------------------------

/// `MacExpertEncoding` code -> glyph name ([`ND`] = `_notdef`).
///
// MuPDF: fz_glyph_name_from_mac_expert (encodings.h:140)
pub static MAC_EXPERT: [&str; 256] = [
    ND, ND, ND, ND, ND, ND, ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    "space", "exclamsmall", "Hungarumlautsmall", "centoldstyle",
    "dollaroldstyle", "dollarsuperior", "ampersandsmall", "Acutesmall",
    "parenleftsuperior", "parenrightsuperior", "twodotenleader",
    "onedotenleader", "comma", "hyphen", "period", "fraction",
    "zerooldstyle", "oneoldstyle", "twooldstyle", "threeoldstyle",
    "fouroldstyle", "fiveoldstyle", "sixoldstyle", "sevenoldstyle",
    "eightoldstyle", "nineoldstyle", "colon", "semicolon", ND,
    "threequartersemdash", ND, "questionsmall", ND, ND,
    ND, ND, "Ethsmall", ND, ND, "onequarter",
    "onehalf", "threequarters", "oneeighth", "threeeighths", "fiveeighths",
    "seveneighths", "onethird", "twothirds", ND, ND, ND,
    ND, ND, ND, "ff", "fi", "fl", "ffi", "ffl",
    "parenleftinferior", ND, "parenrightinferior", "Circumflexsmall",
    "hypheninferior", "Gravesmall", "Asmall", "Bsmall", "Csmall", "Dsmall",
    "Esmall", "Fsmall", "Gsmall", "Hsmall", "Ismall", "Jsmall", "Ksmall",
    "Lsmall", "Msmall", "Nsmall", "Osmall", "Psmall", "Qsmall", "Rsmall",
    "Ssmall", "Tsmall", "Usmall", "Vsmall", "Wsmall", "Xsmall", "Ysmall",
    "Zsmall", "colonmonetary", "onefitted", "rupiah", "Tildesmall",
    ND, ND, "asuperior", "centsuperior", ND, ND,
    ND, ND, "Aacutesmall", "Agravesmall", "Acircumflexsmall",
    "Adieresissmall", "Atildesmall", "Aringsmall", "Ccedillasmall",
    "Eacutesmall", "Egravesmall", "Ecircumflexsmall", "Edieresissmall",
    "Iacutesmall", "Igravesmall", "Icircumflexsmall", "Idieresissmall",
    "Ntildesmall", "Oacutesmall", "Ogravesmall", "Ocircumflexsmall",
    "Odieresissmall", "Otildesmall", "Uacutesmall", "Ugravesmall",
    "Ucircumflexsmall", "Udieresissmall", ND, "eightsuperior",
    "fourinferior", "threeinferior", "sixinferior", "eightinferior",
    "seveninferior", "Scaronsmall", ND, "centinferior", "twoinferior",
    ND, "Dieresissmall", ND, "Caronsmall", "osuperior",
    "fiveinferior", ND, "commainferior", "periodinferior",
    "Yacutesmall", ND, "dollarinferior", ND, ND,
    "Thornsmall", ND, "nineinferior", "zeroinferior", "Zcaronsmall",
    "AEsmall", "Oslashsmall", "questiondownsmall", "oneinferior",
    "Lslashsmall", ND, ND, ND, ND, ND, ND,
    "Cedillasmall", ND, ND, ND, ND, ND, "OEsmall",
    "figuredash", "hyphensuperior", ND, ND, ND, ND,
    "exclamdownsmall", ND, "Ydieresissmall", ND, "onesuperior",
    "twosuperior", "threesuperior", "foursuperior", "fivesuperior",
    "sixsuperior", "sevensuperior", "ninesuperior", "zerosuperior",
    ND, "esuperior", "rsuperior", "tsuperior", ND, ND,
    "isuperior", "ssuperior", "dsuperior", ND, ND, ND,
    ND, ND, "lsuperior", "Ogoneksmall", "Brevesmall",
    "Macronsmall", "bsuperior", "nsuperior", "msuperior", "commasuperior",
    "periodsuperior", "Dotaccentsmall", "Ringsmall", ND, ND,
    ND, ND,
];

// ---------------------------------------------------------------------------
// WinAnsiEncoding: code -> glyph name
// ---------------------------------------------------------------------------

/// `WinAnsiEncoding` code -> glyph name ([`ND`] = `_notdef`). All unused codes
/// > 32 map to `"bullet"` (matching MuPDF's comment on the C table).
///
// MuPDF: fz_glyph_name_from_win_ansi (encodings.h:193)
pub static WIN_ANSI: [&str; 256] = [
    ND, ND, ND, ND, ND, ND, ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    ND, ND, ND, ND, ND, ND, ND, ND,
    "space", "exclam", "quotedbl", "numbersign", "dollar", "percent",
    "ampersand", "quotesingle", "parenleft", "parenright", "asterisk",
    "plus", "comma", "hyphen", "period", "slash", "zero", "one", "two",
    "three", "four", "five", "six", "seven", "eight", "nine", "colon",
    "semicolon", "less", "equal", "greater", "question", "at", "A", "B",
    "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "bracketleft",
    "backslash", "bracketright", "asciicircum", "underscore", "grave", "a",
    "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "braceleft",
    "bar", "braceright", "asciitilde", "bullet", "Euro", "bullet",
    "quotesinglbase", "florin", "quotedblbase", "ellipsis", "dagger",
    "daggerdbl", "circumflex", "perthousand", "Scaron", "guilsinglleft",
    "OE", "bullet", "Zcaron", "bullet", "bullet", "quoteleft",
    "quoteright", "quotedblleft", "quotedblright", "bullet", "endash",
    "emdash", "tilde", "trademark", "scaron", "guilsinglright", "oe",
    "bullet", "zcaron", "Ydieresis", "space", "exclamdown", "cent",
    "sterling", "currency", "yen", "brokenbar", "section", "dieresis",
    "copyright", "ordfeminine", "guillemotleft", "logicalnot", "hyphen",
    "registered", "macron", "degree", "plusminus", "twosuperior",
    "threesuperior", "acute", "mu", "paragraph", "periodcentered",
    "cedilla", "onesuperior", "ordmasculine", "guillemotright",
    "onequarter", "onehalf", "threequarters", "questiondown", "Agrave",
    "Aacute", "Acircumflex", "Atilde", "Adieresis", "Aring", "AE",
    "Ccedilla", "Egrave", "Eacute", "Ecircumflex", "Edieresis", "Igrave",
    "Iacute", "Icircumflex", "Idieresis", "Eth", "Ntilde", "Ograve",
    "Oacute", "Ocircumflex", "Otilde", "Odieresis", "multiply", "Oslash",
    "Ugrave", "Uacute", "Ucircumflex", "Udieresis", "Yacute", "Thorn",
    "germandbls", "agrave", "aacute", "acircumflex", "atilde", "adieresis",
    "aring", "ae", "ccedilla", "egrave", "eacute", "ecircumflex",
    "edieresis", "igrave", "iacute", "icircumflex", "idieresis", "eth",
    "ntilde", "ograve", "oacute", "ocircumflex", "otilde", "odieresis",
    "divide", "oslash", "ugrave", "uacute", "ucircumflex", "udieresis",
    "yacute", "thorn", "ydieresis",
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_256_entries() {
        assert_eq!(STANDARD.len(), 256);
        assert_eq!(WIN_ANSI.len(), 256);
        assert_eq!(MAC_ROMAN.len(), 256);
        assert_eq!(MAC_EXPERT.len(), 256);
        assert_eq!(PDF_DOC.len(), 256);
    }

    #[test]
    fn win_ansi_known_entries() {
        // Plain ASCII passes through.
        assert_eq!(WIN_ANSI[0x41], "A");
        assert_eq!(BaseEncoding::WinAnsi.glyph_name(0x41), Some("A"));
        assert_eq!(WIN_ANSI[0x20], "space");
        // 0x80 is the Euro sign in WinAnsi.
        assert_eq!(WIN_ANSI[0x80], "Euro");
        assert_eq!(BaseEncoding::WinAnsi.glyph_name(0x80), Some("Euro"));
        // Undefined C0 slot -> _notdef -> None.
        assert_eq!(WIN_ANSI[0x01], ND);
        assert_eq!(BaseEncoding::WinAnsi.glyph_name(0x01), None);
        // "unused > 32" slot maps to bullet.
        assert_eq!(WIN_ANSI[0x81], "bullet");
    }

    #[test]
    fn standard_known_entries() {
        assert_eq!(STANDARD[0x41], "A");
        // StandardEncoding uses "quoteright" at 0x27 (see cross-check below).
        assert_eq!(STANDARD[0x27], "quoteright");
        assert_eq!(STANDARD[0xFA], "oe");
        assert_eq!(STANDARD[0xF6], ND); // undefined slot
        assert_eq!(BaseEncoding::Standard.glyph_name(0x00), None);
    }

    #[test]
    fn mac_roman_known_entries() {
        assert_eq!(MAC_ROMAN[0x41], "A");
        // High-range MacRoman entry.
        assert_eq!(MAC_ROMAN[0x80], "Adieresis");
        assert_eq!(MAC_ROMAN[0xFF], "caron");
        assert_eq!(BaseEncoding::MacRoman.glyph_name(0xA5), Some("bullet"));
    }

    #[test]
    fn mac_expert_known_entries() {
        assert_eq!(MAC_EXPERT[0x20], "space");
        assert_eq!(MAC_EXPERT[0x21], "exclamsmall");
        assert_eq!(BaseEncoding::MacExpert.glyph_name(0x00), None);
    }

    #[test]
    fn pdf_doc_known_entries() {
        // ASCII identity region.
        assert_eq!(PDF_DOC[0x41], 0x0041);
        assert_eq!(pdf_doc_unicode(0x41), Some('A'));
        // PdfDoc-specific: 0x80 is a bullet, 0xA0 is the Euro sign.
        assert_eq!(PDF_DOC[0x80], 0x2022);
        assert_eq!(PDF_DOC[0xA0], 0x20ac);
        assert_eq!(pdf_doc_unicode(0xA0), Some('\u{20ac}'));
        // Undefined slot (0x7F) -> 0 -> None.
        assert_eq!(PDF_DOC[0x7F], 0x0000);
        assert_eq!(pdf_doc_unicode(0x7F), None);
    }

    #[test]
    fn win_ansi_and_mac_roman_deliberately_differ() {
        // 0x80: Euro in WinAnsi, Adieresis in MacRoman.
        assert_ne!(WIN_ANSI[0x80], MAC_ROMAN[0x80]);
        assert_eq!(WIN_ANSI[0x80], "Euro");
        assert_eq!(MAC_ROMAN[0x80], "Adieresis");
        // 0x27: quotesingle in WinAnsi/MacRoman but quoteright in Standard.
        assert_eq!(WIN_ANSI[0x27], "quotesingle");
        assert_eq!(STANDARD[0x27], "quoteright");
        assert_ne!(WIN_ANSI[0x27], STANDARD[0x27]);
    }
}
