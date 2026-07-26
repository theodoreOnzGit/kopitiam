//! Ported from Tesseract `src/ccutil/unicharset.cpp` +
//! `src/ccutil/unicharset.h` (commit db0ec62, Apache-2.0, © 2006 Google Inc.,
//! Author: Thomas Kielbus), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the on-disk text format, the per-char property fields, the
//! script table, the special-code conventions, and the `CHAR_FRAGMENT` grammar
//! follow Tesseract exactly; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! [`Unicharset`] is Tesseract's id↔UTF-8 table: a contiguous set of unichars,
//! each with a numeric [`UnicharId`] in `0..size`, an id→UTF-8 representation,
//! a UTF-8→id lookup ([`crate::unicharmap`]), and per-char properties
//! (isalpha/isdigit/ispunctuation/script/…). The LSTM recogniser loads two of
//! these — the `TESSDATA_LSTM_UNICHARSET` component drives the CJK recoder
//! ([`crate::unicharcompress`]).
//!
//! # On-disk format (the read path)
//!
//! The unicharset component is a **text** blob read line-by-line via
//! [`crate::serialis::TFile::fgets`] (Tesseract: `load_from_file(TFile*)` →
//! `load_via_fgets`, unicharset.cpp:776). The first line is the entry count;
//! each subsequent line is one unichar:
//!
//! ```text
//! <utf8>  <props-hex>  <min_bottom,max_bottom,min_top,max_top,width,width_sd,bearing,bearing_sd,advance,advance_sd>  <script>  <other_case>  <direction>  <mirror>  <normed>  \t# <debug>
//! ```
//!
//! with shorter historical variants (fewer metrics, no direction/mirror/normed),
//! and a special form for the space unichar: `NULL <props-hex> <script>
//! <other_case>`. The trailing `\t# …` debug comment is ignored. This port reads
//! the fields positionally, which reproduces the loader's result for every
//! well-formed line (Tesseract's literal `istringstream` fallback cascade is an
//! implementation detail, not a format).
//!
//! ## Faithful quirk: properties are read as hex
//!
//! Tesseract *writes* the property bitfield in **decimal** for normal lines but
//! *reads* it back with `std::hex` (unicharset.cpp:824). The read path is
//! therefore base-16, and this port matches it: [`Unicharset::load`] parses the
//! property field as hexadecimal. (A latent upstream inconsistency for values
//! ≥ 0xA; reproduced rather than "fixed" so ids/properties match Tesseract.)
//!
//! # Deferred (write path & rarely-used fields)
//!
//! The save/serialise side is not ported (KOPITIAM reads unicharsets). Two
//! read-side simplifications are documented at their call sites: the
//! `encode_string` n-gram de-duplication in `unichar_insert` (only affects
//! multi-codepoint entries that are already encodable — absent from LSTM
//! grapheme unicharsets), and `normed_ids` (needs `encode_string`; the raw
//! `normed` string is still stored).

use crate::error::{Error, Result};
use crate::serialis::TFile;
use crate::unichar::{self, INVALID_UNICHAR, INVALID_UNICHAR_ID, UNICHAR_LEN, UnicharId};
use crate::unicharmap::Unicharmap;

// ---------------------------------------------------------------------------
// Special codes, masks and constants (unicharset.cpp:35..80)
// ---------------------------------------------------------------------------

/// `UNICHAR_SPACE`: the space unichar, always id 0 (unicharset.h:36).
pub const UNICHAR_SPACE: UnicharId = 0;
/// `UNICHAR_JOINED` (unicharset.h:37).
pub const UNICHAR_JOINED: UnicharId = 1;
/// `UNICHAR_BROKEN` (unicharset.h:38).
pub const UNICHAR_BROKEN: UnicharId = 2;
/// `SPECIAL_UNICHAR_CODES_COUNT` (unicharset.h:40).
pub const SPECIAL_UNICHAR_CODES_COUNT: usize = 3;

/// `kSpecialUnicharCodes` (unicharset.cpp:79): the representations of the three
/// special codes, in id order.
pub const SPECIAL_UNICHAR_CODES: [&str; SPECIAL_UNICHAR_CODES_COUNT] = [" ", "Joined", "|Broken|0|1"];

/// The null script name. Tesseract: `UNICHARSET::null_script = "NULL"`
/// (unicharset.cpp:82).
pub const NULL_SCRIPT: &str = "NULL";

// Property bit masks (unicharset.cpp:41).
const ISALPHA_MASK: u32 = 0x1;
const ISLOWER_MASK: u32 = 0x2;
const ISUPPER_MASK: u32 = 0x4;
const ISDIGIT_MASK: u32 = 0x8;
const ISPUNCTUATION_MASK: u32 = 0x10;

// post_load_setup thresholds (unicharset.cpp:50..57).
const MEANLINE_THRESHOLD: i32 = 220;
const MIN_X_HEIGHT_FRACTION: f64 = 0.25;
const MIN_CAP_HEIGHT_FRACTION: f64 = 0.125;

/// `kCleanupMaps` (unicharset.cpp:72): substitutions applied when ingesting a
/// unichar representation — TATWEEL deleted, fi/fl ligatures expanded.
const CLEANUP_MAPS: [(&str, &str); 3] = [
    ("\u{0640}", ""),   // TATWEEL deleted
    ("\u{fb01}", "fi"), // fi ligature -> fi pair
    ("\u{fb02}", "fl"), // fl ligature -> fl pair
];

// CHAR_FRAGMENT grammar (unicharset.cpp:37, unicharset.h:53).
const FRAG_SEPARATOR: u8 = b'|';
const FRAG_NATURAL_FLAG: u8 = b'n';
const FRAG_MIN_LEN: usize = 6;

// ---------------------------------------------------------------------------
// CHAR_FRAGMENT
// ---------------------------------------------------------------------------

/// Meta-information about a unichar that represents a fragment of a character.
///
/// Tesseract: `class CHAR_FRAGMENT` (unicharset.h:50). Fragments are written
/// `|<unichar>|<pos>|<total>` (or `…<pos>n<total>` for a "natural" split).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CharFragment {
    unichar: String,
    natural: bool,
    pos: i16,
    total: i16,
}

impl CharFragment {
    /// The base unichar of the fragment.
    pub fn unichar(&self) -> &str {
        &self.unichar
    }
    /// The fragment's position within the character.
    pub fn pos(&self) -> i16 {
        self.pos
    }
    /// The total number of fragments in the character.
    pub fn total(&self) -> i16 {
        self.total
    }
    /// Whether the fragment was already a separate connected component.
    pub fn is_natural(&self) -> bool {
        self.natural
    }

    /// Parse a fragment representation, or [`None`] if `s` is a regular char.
    ///
    /// Tesseract: `CHAR_FRAGMENT::parse_from_string` (unicharset.cpp:1103).
    pub fn parse_from_string(s: &str) -> Option<CharFragment> {
        let b = s.as_bytes();
        let len = b.len();
        if len < FRAG_MIN_LEN || b[0] != FRAG_SEPARATOR {
            return None; // cannot be a fragment
        }
        let mut ptr = 1usize;
        // Consume the base unichar up to the next separator.
        let mut step = 0usize;
        while ptr + step < len && b[ptr + step] != FRAG_SEPARATOR {
            let s = unichar::utf8_step(&b[ptr + step..]);
            step += if s == 0 { 1 } else { s };
        }
        if step == 0 || step > UNICHAR_LEN {
            return None;
        }
        let unichar = String::from_utf8_lossy(&b[ptr..ptr + step]).into_owned();
        ptr += step;
        let mut pos = 0i32;
        let mut total = 0i32;
        let mut natural = false;
        for i in 0..2 {
            if ptr > len || ptr >= len || b[ptr] != FRAG_SEPARATOR {
                if i == 1 && ptr < len && b[ptr] == FRAG_NATURAL_FLAG {
                    natural = true;
                } else {
                    return None;
                }
            }
            ptr += 1; // move past the separator / natural flag
            let (value, consumed) = strtol(&b[ptr.min(len)..]);
            if consumed == 0 {
                return None;
            }
            if i == 0 {
                pos = value;
            } else {
                total = value;
            }
            ptr += consumed;
        }
        if ptr != len {
            return None;
        }
        Some(CharFragment {
            unichar,
            natural,
            pos: pos as i16,
            total: total as i16,
        })
    }
}

/// Parse a leading base-10 integer (optional sign), returning `(value,
/// bytes_consumed)`. Mimics `strtol(.., 10)` enough for the fragment grammar.
fn strtol(b: &[u8]) -> (i32, usize) {
    let mut i = 0;
    let mut sign = 1i64;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        if b[i] == b'-' {
            sign = -1;
        }
        i += 1;
    }
    let start_digits = i;
    let mut value = 0i64;
    while i < b.len() && b[i].is_ascii_digit() {
        value = value * 10 + (b[i] - b'0') as i64;
        i += 1;
    }
    if i == start_digits {
        return (0, 0); // no digits
    }
    ((sign * value) as i32, i)
}

// ---------------------------------------------------------------------------
// Properties + slot
// ---------------------------------------------------------------------------

/// Per-unichar properties. Tesseract: `UNICHARSET::UNICHAR_PROPERTIES`
/// (unicharset.h:963). The font-metric ranges are parsed and stored but rarely
/// consulted on the recognition read path.
#[derive(Clone, Debug)]
struct UnicharProperties {
    isalpha: bool,
    islower: bool,
    isupper: bool,
    isdigit: bool,
    ispunctuation: bool,
    isngram: bool,
    enabled: bool,
    min_bottom: u8,
    max_bottom: u8,
    min_top: u8,
    max_top: u8,
    width: f32,
    width_sd: f32,
    bearing: f32,
    bearing_sd: f32,
    advance: f32,
    advance_sd: f32,
    script_id: i32,
    other_case: UnicharId,
    direction: i32,
    mirror: UnicharId,
    normed: String,
    fragment: Option<CharFragment>,
}

impl Default for UnicharProperties {
    /// Tesseract: `UNICHAR_PROPERTIES::Init` + `SetRangesOpen` (unicharset.cpp:89).
    fn default() -> Self {
        UnicharProperties {
            isalpha: false,
            islower: false,
            isupper: false,
            isdigit: false,
            ispunctuation: false,
            isngram: false,
            enabled: false,
            min_bottom: 0,
            max_bottom: u8::MAX,
            min_top: 0,
            max_top: u8::MAX,
            width: 0.0,
            width_sd: 0.0,
            bearing: 0.0,
            bearing_sd: 0.0,
            advance: 0.0,
            advance_sd: 0.0,
            script_id: 0,
            other_case: 0,
            direction: 0, // U_LEFT_TO_RIGHT
            mirror: 0,
            normed: String::new(),
            fragment: None,
        }
    }
}

/// A single unichar entry: its UTF-8 representation plus properties.
///
/// Tesseract: `UNICHARSET::UNICHAR_SLOT` (unicharset.h:1025).
#[derive(Clone, Debug)]
struct UnicharSlot {
    representation: String,
    properties: UnicharProperties,
}

// ---------------------------------------------------------------------------
// UNICHARSET
// ---------------------------------------------------------------------------

/// The set of characters used by the engine: an id↔UTF-8 table with per-char
/// properties.
///
/// Tesseract: `class UNICHARSET` (unicharset.h:164).
#[derive(Debug, Default)]
pub struct Unicharset {
    unichars: Vec<UnicharSlot>,
    ids: Unicharmap,
    script_table: Vec<String>,
    old_style_included: bool,
    top_bottom_set: bool,
    script_has_upper_lower: bool,
    script_has_xheight: bool,
    // Convenient script name-to-id mappings, filled by post_load_setup.
    null_sid: i32,
    common_sid: i32,
    latin_sid: i32,
    cyrillic_sid: i32,
    greek_sid: i32,
    han_sid: i32,
    hiragana_sid: i32,
    katakana_sid: i32,
    thai_sid: i32,
    hangul_sid: i32,
    default_sid: i32,
}

impl Unicharset {
    /// An empty unicharset.
    pub fn new() -> Self {
        Unicharset::default()
    }

    /// The number of unichars (ids `0..size`).
    ///
    /// Tesseract: `UNICHARSET::size` (unicharset.h:355).
    pub fn size(&self) -> usize {
        self.unichars.len()
    }

    /// Whether `id` names a unichar in the set (ids are contiguous).
    ///
    /// Tesseract: `UNICHARSET::contains_unichar_id` (unicharset.h:303).
    pub fn contains_unichar_id(&self, id: UnicharId) -> bool {
        id >= 0 && (id as usize) < self.unichars.len()
    }

    /// The UTF-8 representation of `id`, or [`INVALID_UNICHAR`] for
    /// [`INVALID_UNICHAR_ID`].
    ///
    /// Tesseract: `UNICHARSET::id_to_unichar` (unicharset.cpp:279).
    pub fn id_to_unichar(&self, id: UnicharId) -> &str {
        if id == INVALID_UNICHAR_ID {
            return INVALID_UNICHAR;
        }
        &self.unichars[id as usize].representation
    }

    /// The id of `unichar_repr`, or [`INVALID_UNICHAR_ID`] if absent.
    ///
    /// Tesseract: `UNICHARSET::unichar_to_id` (unicharset.cpp:186), including the
    /// CleanupString normalisation of the query.
    pub fn unichar_to_id(&self, unichar_repr: &str) -> UnicharId {
        let cleaned = if self.old_style_included {
            unichar_repr.as_bytes().to_vec()
        } else {
            cleanup_string(unichar_repr.as_bytes())
        };
        if self.ids.contains(&cleaned, cleaned.len()) {
            self.ids.unichar_to_id(&cleaned, cleaned.len())
        } else {
            INVALID_UNICHAR_ID
        }
    }

    /// Whether `unichar_repr` exists in the set.
    ///
    /// Tesseract: `UNICHARSET::contains_unichar` (unicharset.cpp:695).
    pub fn contains_unichar(&self, unichar_repr: &str) -> bool {
        let cleaned = if self.old_style_included {
            unichar_repr.as_bytes().to_vec()
        } else {
            cleanup_string(unichar_repr.as_bytes())
        };
        self.ids.contains(&cleaned, cleaned.len())
    }

    // --- property accessors (unicharset.h:497..) ---

    /// Tesseract: `UNICHARSET::get_isalpha` (unicharset.h:497).
    pub fn get_isalpha(&self, id: UnicharId) -> bool {
        self.prop(id).map(|p| p.isalpha).unwrap_or(false)
    }
    /// Tesseract: `UNICHARSET::get_islower` (unicharset.h:506).
    pub fn get_islower(&self, id: UnicharId) -> bool {
        self.prop(id).map(|p| p.islower).unwrap_or(false)
    }
    /// Tesseract: `UNICHARSET::get_isupper` (unicharset.h:515).
    pub fn get_isupper(&self, id: UnicharId) -> bool {
        self.prop(id).map(|p| p.isupper).unwrap_or(false)
    }
    /// Tesseract: `UNICHARSET::get_isdigit` (unicharset.h:524).
    pub fn get_isdigit(&self, id: UnicharId) -> bool {
        self.prop(id).map(|p| p.isdigit).unwrap_or(false)
    }
    /// Tesseract: `UNICHARSET::get_ispunctuation` (unicharset.h:533).
    pub fn get_ispunctuation(&self, id: UnicharId) -> bool {
        self.prop(id).map(|p| p.ispunctuation).unwrap_or(false)
    }
    /// Tesseract: `UNICHARSET::get_isngram` (unicharset.h:542).
    pub fn get_isngram(&self, id: UnicharId) -> bool {
        self.prop(id).map(|p| p.isngram).unwrap_or(false)
    }

    /// The property bitfield of `id` (alpha/lower/upper/digit/punct).
    ///
    /// Tesseract: `UNICHARSET::get_properties` (unicharset.cpp:615).
    pub fn get_properties(&self, id: UnicharId) -> u32 {
        let mut p = 0;
        if self.get_isalpha(id) {
            p |= ISALPHA_MASK;
        }
        if self.get_islower(id) {
            p |= ISLOWER_MASK;
        }
        if self.get_isupper(id) {
            p |= ISUPPER_MASK;
        }
        if self.get_isdigit(id) {
            p |= ISDIGIT_MASK;
        }
        if self.get_ispunctuation(id) {
            p |= ISPUNCTUATION_MASK;
        }
        p
    }

    /// The dominant character type of `id`: `A`/`a`/`x`/`0`/`p`, or `\0`.
    ///
    /// Tesseract: `UNICHARSET::get_chartype` (unicharset.cpp:635).
    pub fn get_chartype(&self, id: UnicharId) -> char {
        if self.get_isupper(id) {
            'A'
        } else if self.get_islower(id) {
            'a'
        } else if self.get_isalpha(id) {
            'x'
        } else if self.get_isdigit(id) {
            '0'
        } else if self.get_ispunctuation(id) {
            'p'
        } else {
            '\0'
        }
    }

    /// The script id of `id`, or [`null_sid`](Self::null_sid) for invalid.
    ///
    /// Tesseract: `UNICHARSET::get_script` (unicharset.h:681).
    pub fn get_script(&self, id: UnicharId) -> i32 {
        if id == INVALID_UNICHAR_ID {
            return self.null_sid;
        }
        self.unichars[id as usize].properties.script_id
    }

    /// The other-case id of `id`. Tesseract: `get_other_case` (unicharset.h:703).
    pub fn get_other_case(&self, id: UnicharId) -> UnicharId {
        self.prop(id).map(|p| p.other_case).unwrap_or(INVALID_UNICHAR_ID)
    }
    /// The direction of `id`. Tesseract: `get_direction` (unicharset.h:712).
    pub fn get_direction(&self, id: UnicharId) -> i32 {
        self.prop(id).map(|p| p.direction).unwrap_or(10 /* U_OTHER_NEUTRAL */)
    }
    /// The mirror id of `id`. Tesseract: `get_mirror` (unicharset.h:721).
    pub fn get_mirror(&self, id: UnicharId) -> UnicharId {
        self.prop(id).map(|p| p.mirror).unwrap_or(INVALID_UNICHAR_ID)
    }
    /// The normalised representation of `id`. Tesseract: `get_normed_unichar`
    /// (unicharset.h:859).
    pub fn get_normed_unichar(&self, id: UnicharId) -> &str {
        if id == UNICHAR_SPACE {
            return " ";
        }
        &self.unichars[id as usize].properties.normed
    }
    /// Whether `id` is a fragment; the fragment if so.
    ///
    /// Tesseract: `UNICHARSET::get_fragment` (unicharset.h:768).
    pub fn get_fragment(&self, id: UnicharId) -> Option<&CharFragment> {
        if id == INVALID_UNICHAR_ID {
            return None;
        }
        self.unichars[id as usize].properties.fragment.as_ref()
    }

    /// Whether the special codes (space/joined/broken) occupy their ids.
    ///
    /// Tesseract: `UNICHARSET::has_special_codes` (unicharset.h:756).
    pub fn has_special_codes(&self) -> bool {
        self.get_fragment(UNICHAR_BROKEN).is_some()
            && self.id_to_unichar(UNICHAR_BROKEN) == SPECIAL_UNICHAR_CODES[UNICHAR_BROKEN as usize]
    }

    // --- script table (unicharset.cpp:1063) ---

    /// The number of scripts. Tesseract: `get_script_table_size`
    /// (unicharset.h:881).
    pub fn get_script_table_size(&self) -> usize {
        self.script_table.len()
    }

    /// The script name for a script id, or [`NULL_SCRIPT`] if out of range.
    ///
    /// Tesseract: `get_script_from_script_id` (unicharset.h:886).
    pub fn get_script_from_script_id(&self, id: i32) -> &str {
        if id < 0 || id as usize >= self.script_table.len() {
            NULL_SCRIPT
        } else {
            &self.script_table[id as usize]
        }
    }

    /// The id of a script name, or 0 (the null script) if not found.
    ///
    /// Tesseract: `get_script_id_from_name` (unicharset.cpp:1146).
    pub fn get_script_id_from_name(&self, name: &str) -> i32 {
        for (i, s) in self.script_table.iter().enumerate() {
            if s == name {
                return i as i32;
            }
        }
        0
    }

    /// Uniquify and intern `script`, returning its id.
    ///
    /// Tesseract: `UNICHARSET::add_script` (unicharset.cpp:1063).
    fn add_script(&mut self, script: &str) -> i32 {
        for (i, s) in self.script_table.iter().enumerate() {
            if s == script {
                return i as i32;
            }
        }
        self.script_table.push(script.to_string());
        (self.script_table.len() - 1) as i32
    }

    // --- named script id accessors (unicharset.h:916) ---

    /// Tesseract: `null_sid` (unicharset.h:916).
    pub fn null_sid(&self) -> i32 {
        self.null_sid
    }
    /// Tesseract: `common_sid` (unicharset.h:919).
    pub fn common_sid(&self) -> i32 {
        self.common_sid
    }
    /// Tesseract: `latin_sid` (unicharset.h:922).
    pub fn latin_sid(&self) -> i32 {
        self.latin_sid
    }
    /// Tesseract: `cyrillic_sid` (unicharset.h:925).
    pub fn cyrillic_sid(&self) -> i32 {
        self.cyrillic_sid
    }
    /// Tesseract: `greek_sid` (unicharset.h:928).
    pub fn greek_sid(&self) -> i32 {
        self.greek_sid
    }
    /// Tesseract: `han_sid` (unicharset.h:931).
    pub fn han_sid(&self) -> i32 {
        self.han_sid
    }
    /// Tesseract: `hiragana_sid` (unicharset.h:934).
    pub fn hiragana_sid(&self) -> i32 {
        self.hiragana_sid
    }
    /// Tesseract: `katakana_sid` (unicharset.h:937).
    pub fn katakana_sid(&self) -> i32 {
        self.katakana_sid
    }
    /// Tesseract: `thai_sid` (unicharset.h:940).
    pub fn thai_sid(&self) -> i32 {
        self.thai_sid
    }
    /// Tesseract: `hangul_sid` (unicharset.h:943).
    pub fn hangul_sid(&self) -> i32 {
        self.hangul_sid
    }
    /// Tesseract: `default_sid` (unicharset.h:946).
    pub fn default_sid(&self) -> i32 {
        self.default_sid
    }
    /// Whether the set has upper/lower case. Tesseract: `script_has_upper_lower`
    /// (unicharset.h:951).
    pub fn script_has_upper_lower(&self) -> bool {
        self.script_has_upper_lower
    }
    /// Whether the set has an x-height concept. Tesseract: `script_has_xheight`
    /// (unicharset.h:958).
    pub fn script_has_xheight(&self) -> bool {
        self.script_has_xheight
    }
    /// Whether tops/bottoms are useful. Tesseract: `top_bottom_useful`
    /// (unicharset.h:555).
    pub fn top_bottom_useful(&self) -> bool {
        self.top_bottom_set
    }

    // -----------------------------------------------------------------------
    // Loading (the read path)
    // -----------------------------------------------------------------------

    /// Load a unicharset from an in-memory text buffer (e.g. the
    /// `TESSDATA_LSTM_UNICHARSET` component).
    ///
    /// Tesseract: `UNICHARSET::load_from_file(TFile*, skip_fragments)` →
    /// `load_via_fgets` (unicharset.cpp:776/784).
    pub fn load_from_bytes(data: &[u8], skip_fragments: bool) -> Result<Self> {
        let mut fp = TFile::new(data);
        Self::load(&mut fp, skip_fragments)
    }

    /// Load a unicharset by reading lines from `fp`.
    ///
    /// Tesseract: `UNICHARSET::load_via_fgets` (unicharset.cpp:784). The 256-byte
    /// `fgets` buffer of the C is reproduced.
    pub fn load(fp: &mut TFile<'_>, skip_fragments: bool) -> Result<Self> {
        let mut set = Unicharset::new();

        let header = fp
            .fgets(256)
            .ok_or_else(|| Error::format("unicharset: missing size header"))?;
        let unicharset_size: i32 = parse_first_int(&header)
            .ok_or_else(|| Error::format("unicharset: unparseable size header"))?;
        if unicharset_size < 0 {
            return Err(Error::format("unicharset: negative size"));
        }

        for id in 0..unicharset_size {
            let line = fp
                .fgets(256)
                .ok_or_else(|| Error::unexpected_eof("unicharset: truncated before all entries"))?;
            set.load_line(&line, id, unicharset_size, skip_fragments)?;
        }
        set.post_load_setup();
        Ok(set)
    }

    /// Parse and ingest one unichar line.
    ///
    /// Tesseract: the body of the `load_via_fgets` loop (unicharset.cpp:794).
    fn load_line(
        &mut self,
        line: &[u8],
        id: UnicharId,
        unicharset_size: i32,
        skip_fragments: bool,
    ) -> Result<()> {
        // Split on ASCII whitespace; trailing "\t# <debug>" fields are ignored.
        let text = String::from_utf8_lossy(line);
        let fields: Vec<&str> = text.split_whitespace().collect();
        if fields.len() < 2 {
            return Err(Error::format("unicharset: line has too few fields"));
        }
        let unichar = fields[0];
        // Properties are read as HEX (faithful quirk; see module docs).
        let properties = u32::from_str_radix(fields[1], 16)
            .map_err(|_| Error::format("unicharset: bad properties field"))?;

        // Metric/geometry defaults (unicharset.cpp:800).
        let mut min_bottom = 0i32;
        let mut max_bottom = u8::MAX as i32;
        let mut min_top = 0i32;
        let mut max_top = u8::MAX as i32;
        let mut width = 0.0f32;
        let mut width_sd = 0.0f32;
        let mut bearing = 0.0f32;
        let mut bearing_sd = 0.0f32;
        let mut advance = 0.0f32;
        let mut advance_sd = 0.0f32;
        let mut direction = 0i32; // U_LEFT_TO_RIGHT
        let mut other_case = unicharset_size;
        let mut mirror = unicharset_size;
        let mut script = NULL_SCRIPT.to_string();
        let mut normed = String::new();

        // Field 2 is either a comma-separated metrics block or (for the space /
        // shortest forms) the script name.
        if fields.len() > 2 && fields[2].contains(',') {
            let parts: Vec<&str> = fields[2].split(',').collect();
            if parts.len() >= 10 {
                min_bottom = parts[0].parse().unwrap_or(min_bottom);
                max_bottom = parts[1].parse().unwrap_or(max_bottom);
                min_top = parts[2].parse().unwrap_or(min_top);
                max_top = parts[3].parse().unwrap_or(max_top);
                width = parts[4].parse().unwrap_or(width);
                width_sd = parts[5].parse().unwrap_or(width_sd);
                bearing = parts[6].parse().unwrap_or(bearing);
                bearing_sd = parts[7].parse().unwrap_or(bearing_sd);
                advance = parts[8].parse().unwrap_or(advance);
                advance_sd = parts[9].parse().unwrap_or(advance_sd);
                // <script> <other_case> <direction> <mirror> <normed>
                if let Some(f) = fields.get(3) {
                    script = (*f).to_string();
                }
                if let Some(f) = fields.get(4) {
                    other_case = f.parse().unwrap_or(other_case);
                }
                if let Some(f) = fields.get(5) {
                    direction = f.parse().unwrap_or(direction);
                }
                if let Some(f) = fields.get(6) {
                    mirror = f.parse().unwrap_or(mirror);
                }
                if let Some(f) = fields.get(7) {
                    normed = (*f).to_string();
                }
            } else if parts.len() >= 4 {
                // Historical short form: 4 metrics, then script/other_case and
                // optionally direction/mirror.
                min_bottom = parts[0].parse().unwrap_or(min_bottom);
                max_bottom = parts[1].parse().unwrap_or(max_bottom);
                min_top = parts[2].parse().unwrap_or(min_top);
                max_top = parts[3].parse().unwrap_or(max_top);
                if let Some(f) = fields.get(3) {
                    script = (*f).to_string();
                }
                if let Some(f) = fields.get(4) {
                    other_case = f.parse().unwrap_or(other_case);
                }
                if let Some(f) = fields.get(5) {
                    direction = f.parse().unwrap_or(direction);
                }
                if let Some(f) = fields.get(6) {
                    mirror = f.parse().unwrap_or(mirror);
                }
            }
        } else {
            // Shortest form: `<unichar> <props> <script> [<other_case>]`
            // (this is how the space unichar `NULL <props> <script> <oc>` loads).
            if let Some(f) = fields.get(2) {
                script = (*f).to_string();
            }
            if let Some(f) = fields.get(3) {
                other_case = f.parse().unwrap_or(other_case);
            }
        }

        // Skip multi-element fragments if requested (unicharset.cpp:872).
        if skip_fragments
            && CharFragment::parse_from_string(unichar).is_some_and(|frag| frag.total() > 1)
        {
            return Ok(());
        }

        // Insert the unichar (NULL is the space sentinel).
        if unichar == "NULL" {
            self.unichar_insert(" ", false);
        } else {
            self.unichar_insert_backwards_compatible(unichar);
        }

        // Guard: the insert must have produced exactly the expected id. For a
        // well-formed unicharset every line adds one distinct unichar.
        if !self.contains_unichar_id(id) || self.unichars.len() as i32 != id + 1 {
            return Err(Error::format(format!(
                "unicharset: entry {id} ({unichar:?}) did not extend the set as expected"
            )));
        }

        let idx = id as usize;
        self.unichars[idx].properties.isalpha = properties & ISALPHA_MASK != 0;
        self.unichars[idx].properties.islower = properties & ISLOWER_MASK != 0;
        self.unichars[idx].properties.isupper = properties & ISUPPER_MASK != 0;
        self.unichars[idx].properties.isdigit = properties & ISDIGIT_MASK != 0;
        self.unichars[idx].properties.ispunctuation = properties & ISPUNCTUATION_MASK != 0;
        self.unichars[idx].properties.isngram = false;
        let sid = self.add_script(&script);
        self.unichars[idx].properties.script_id = sid;
        self.unichars[idx].properties.enabled = true;
        self.unichars[idx].properties.min_bottom = clip_u8(min_bottom);
        self.unichars[idx].properties.max_bottom = clip_u8(max_bottom);
        self.unichars[idx].properties.min_top = clip_u8(min_top);
        self.unichars[idx].properties.max_top = clip_u8(max_top);
        self.unichars[idx].properties.width = width;
        self.unichars[idx].properties.width_sd = width_sd;
        self.unichars[idx].properties.bearing = bearing;
        self.unichars[idx].properties.bearing_sd = bearing_sd;
        self.unichars[idx].properties.advance = advance;
        self.unichars[idx].properties.advance_sd = advance_sd;
        self.unichars[idx].properties.direction = direction;
        self.unichars[idx].properties.other_case = if other_case < unicharset_size {
            other_case
        } else {
            id
        };
        self.unichars[idx].properties.mirror = if mirror < unicharset_size { mirror } else { id };
        self.unichars[idx].properties.normed = if normed.is_empty() {
            unichar.to_string()
        } else {
            normed
        };
        Ok(())
    }

    /// Add a unichar representation, cleaning it first unless old-style.
    ///
    /// Tesseract: `UNICHARSET::unichar_insert` (unicharset.cpp:654). NOTE: the
    /// `encode_string` n-gram de-duplication (skip an already-encodable n-gram)
    /// is intentionally omitted — it only affects multi-codepoint entries that
    /// decompose into existing unichars, which do not occur in LSTM grapheme
    /// unicharsets. See the module-level "Deferred" note.
    fn unichar_insert(&mut self, unichar_repr: &str, old_style: bool) {
        if old_style {
            self.old_style_included = true;
        }
        let cleaned_bytes = if self.old_style_included {
            unichar_repr.as_bytes().to_vec()
        } else {
            cleanup_string(unichar_repr.as_bytes())
        };
        if cleaned_bytes.is_empty() || self.ids.contains(&cleaned_bytes, cleaned_bytes.len()) {
            return;
        }
        let cleaned = String::from_utf8_lossy(&cleaned_bytes).into_owned();
        let new_id = self.unichars.len() as UnicharId;
        // Script defaults to null; if this is a fragment of a known base unichar,
        // inherit the base's script (unicharset.cpp:687).
        let null_sid = self.add_script(NULL_SCRIPT);
        let frag = CharFragment::parse_from_string(&cleaned);
        let script_id = match &frag {
            Some(f) if self.contains_unichar(f.unichar()) => {
                self.get_script(self.unichar_to_id(f.unichar()))
            }
            _ => null_sid,
        };
        let props = UnicharProperties {
            script_id,
            fragment: frag,
            enabled: true,
            ..Default::default()
        };
        self.unichars.push(UnicharSlot {
            representation: cleaned,
            properties: props,
        });
        self.ids.insert(&cleaned_bytes, new_id);
    }

    /// Add a unichar, preferring clean-style but falling back to old-style if
    /// cleaning would change or drop it.
    ///
    /// Tesseract: `unichar_insert_backwards_compatible` (unicharset.h:288).
    fn unichar_insert_backwards_compatible(&mut self, unichar_repr: &str) {
        let cleaned = cleanup_string(unichar_repr.as_bytes());
        if cleaned.as_slice() != unichar_repr.as_bytes() {
            self.unichar_insert(unichar_repr, true);
        } else {
            let old_size = self.size();
            self.unichar_insert(unichar_repr, false);
            if self.size() == old_size {
                self.unichar_insert(unichar_repr, true);
            }
        }
    }

    /// Compute derived state after loading (sids, case/x-height flags,
    /// top/bottom usefulness, default script).
    ///
    /// Tesseract: `UNICHARSET::post_load_setup` (unicharset.cpp:912). The
    /// `set_normed_ids` step (which needs `encode_string`) is deferred; the raw
    /// `normed` string is retained.
    fn post_load_setup(&mut self) {
        let mut net_case_alphas = 0i32;
        let mut x_height_alphas = 0i32;
        let mut cap_height_alphas = 0i32;
        self.top_bottom_set = false;
        for id in 0..self.unichars.len() as i32 {
            let (min_top, max_top) = {
                let p = &self.unichars[id as usize].properties;
                (p.min_top as i32, p.max_top as i32)
            };
            if min_top > 0 {
                self.top_bottom_set = true;
            }
            if self.get_isalpha(id) {
                if self.get_islower(id) || self.get_isupper(id) {
                    net_case_alphas += 1;
                } else {
                    net_case_alphas -= 1;
                }
                if min_top < MEANLINE_THRESHOLD && max_top < MEANLINE_THRESHOLD {
                    x_height_alphas += 1;
                } else if min_top > MEANLINE_THRESHOLD && max_top > MEANLINE_THRESHOLD {
                    cap_height_alphas += 1;
                }
            }
        }

        self.script_has_upper_lower = net_case_alphas > 0;
        self.script_has_xheight = self.script_has_upper_lower
            || (x_height_alphas as f64 > cap_height_alphas as f64 * MIN_X_HEIGHT_FRACTION
                && cap_height_alphas as f64 > x_height_alphas as f64 * MIN_CAP_HEIGHT_FRACTION);

        self.null_sid = self.get_script_id_from_name(NULL_SCRIPT);
        debug_assert_eq!(self.null_sid, 0, "Tesseract: ASSERT_HOST(null_sid_ == 0)");
        self.common_sid = self.get_script_id_from_name("Common");
        self.latin_sid = self.get_script_id_from_name("Latin");
        self.cyrillic_sid = self.get_script_id_from_name("Cyrillic");
        self.greek_sid = self.get_script_id_from_name("Greek");
        self.han_sid = self.get_script_id_from_name("Han");
        self.hiragana_sid = self.get_script_id_from_name("Hiragana");
        self.katakana_sid = self.get_script_id_from_name("Katakana");
        self.thai_sid = self.get_script_id_from_name("Thai");
        self.hangul_sid = self.get_script_id_from_name("Hangul");

        // Default script: the highest-counting alpha script that is not Common.
        let mut script_counts = vec![0i32; self.script_table.len()];
        for id in 0..self.unichars.len() as i32 {
            if self.get_isalpha(id) {
                script_counts[self.get_script(id) as usize] += 1;
            }
        }
        self.default_sid = 0;
        for s in 1..self.script_table.len() as i32 {
            if script_counts[s as usize] > script_counts[self.default_sid as usize]
                && s != self.common_sid
            {
                self.default_sid = s;
            }
        }
    }

    /// Borrow the properties of `id`, or `None` for [`INVALID_UNICHAR_ID`].
    fn prop(&self, id: UnicharId) -> Option<&UnicharProperties> {
        if id == INVALID_UNICHAR_ID {
            None
        } else {
            self.unichars.get(id as usize).map(|s| &s.properties)
        }
    }
}

/// Clip an int to `0..=255` (Tesseract: `ClipToRange<int>(v, 0, UINT8_MAX)`).
fn clip_u8(v: i32) -> u8 {
    v.clamp(0, u8::MAX as i32) as u8
}

/// Parse the first whitespace-delimited base-10 integer of a line.
fn parse_first_int(line: &[u8]) -> Option<i32> {
    let text = String::from_utf8_lossy(line);
    text.split_whitespace().next()?.parse().ok()
}

/// Clean a unichar representation: apply [`CLEANUP_MAPS`] prefix substitutions.
///
/// Tesseract: `UNICHARSET::CleanupString` (unicharset.cpp:1158).
fn cleanup_string(s: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(s.len());
    let mut pos = 0usize;
    let mut length = s.len() as isize;
    while pos < s.len() && s[pos] != 0 && length > 0 {
        length -= 1;
        let mut matched: Option<usize> = None;
        for (ki, (key, _)) in CLEANUP_MAPS.iter().enumerate() {
            let kb = key.as_bytes();
            let mut m = 0;
            while m < kb.len() && pos + m < s.len() && kb[m] == s[pos + m] {
                m += 1;
            }
            if m == kb.len() {
                matched = Some(ki);
                pos += m;
                break;
            }
        }
        match matched {
            None => {
                result.push(s[pos]);
                pos += 1;
            }
            Some(ki) => result.extend_from_slice(CLEANUP_MAPS[ki].1.as_bytes()),
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic unicharset in the on-disk text format: space + 2 Latin + 2
    /// Han. Property values are written in HEX to match the reader (see the
    /// module docs on the write-decimal/read-hex quirk).
    fn synthetic() -> String {
        // id: 0=space 1=A 2=a 3=标 4=中
        // props: A=isalpha|isupper=0x5, a=isalpha|islower=0x3, Han=isalpha=0x1
        let mut s = String::new();
        s.push_str("5\n");
        s.push_str("NULL 0 NULL 0\n");
        s.push_str("A 5 0,255,0,255,0,0,0,0,0,0 Latin 2 0 2 A\n");
        s.push_str("a 3 0,255,0,255,0,0,0,0,0,0 Latin 1 0 1 a\n");
        s.push_str("标 1 0,255,0,255,0,0,0,0,0,0 Han 4 0 4 标\n");
        s.push_str("中 1 0,255,0,255,0,0,0,0,0,0 Han 3 0 3 中\n");
        s
    }

    #[test]
    fn parses_synthetic_and_roundtrips_ids() {
        let set = Unicharset::load_from_bytes(synthetic().as_bytes(), false).unwrap();
        assert_eq!(set.size(), 5);

        // id -> utf8
        assert_eq!(set.id_to_unichar(UNICHAR_SPACE), " ");
        assert_eq!(set.id_to_unichar(1), "A");
        assert_eq!(set.id_to_unichar(2), "a");
        assert_eq!(set.id_to_unichar(3), "标");
        assert_eq!(set.id_to_unichar(4), "中");
        assert_eq!(set.id_to_unichar(INVALID_UNICHAR_ID), INVALID_UNICHAR);

        // utf8 -> id (round-trips)
        for id in 0..set.size() as UnicharId {
            let repr = set.id_to_unichar(id).to_string();
            assert_eq!(set.unichar_to_id(&repr), id, "roundtrip id {id}");
        }
        assert_eq!(set.unichar_to_id("A"), 1);
        assert_eq!(set.unichar_to_id("标"), 3);
        assert_eq!(set.unichar_to_id("z"), INVALID_UNICHAR_ID);
    }

    #[test]
    fn property_flags_and_chartype() {
        let set = Unicharset::load_from_bytes(synthetic().as_bytes(), false).unwrap();
        // 'A' upper alpha
        assert!(set.get_isalpha(1) && set.get_isupper(1) && !set.get_islower(1));
        assert_eq!(set.get_chartype(1), 'A');
        assert_eq!(set.get_properties(1), ISALPHA_MASK | ISUPPER_MASK);
        // 'a' lower alpha
        assert!(set.get_isalpha(2) && set.get_islower(2) && !set.get_isupper(2));
        assert_eq!(set.get_chartype(2), 'a');
        // '标' alpha, neither case
        assert!(set.get_isalpha(3) && !set.get_isupper(3) && !set.get_islower(3));
        assert_eq!(set.get_chartype(3), 'x');
        // Space has no properties.
        assert_eq!(set.get_properties(UNICHAR_SPACE), 0);
        // other_case wiring: A<->a
        assert_eq!(set.get_other_case(1), 2);
        assert_eq!(set.get_other_case(2), 1);
    }

    #[test]
    fn scripts_and_sids() {
        let set = Unicharset::load_from_bytes(synthetic().as_bytes(), false).unwrap();
        // null script is always id 0.
        assert_eq!(set.null_sid(), 0);
        assert_eq!(set.get_script_from_script_id(0), NULL_SCRIPT);
        // Han script id is set and matches the Han chars.
        assert!(set.han_sid() > 0);
        assert_eq!(set.get_script(3), set.han_sid());
        assert_eq!(set.get_script(4), set.han_sid());
        assert_eq!(set.get_script_from_script_id(set.get_script(3)), "Han");
        // Latin present.
        assert!(set.latin_sid() > 0);
        assert_eq!(set.get_script(1), set.latin_sid());
    }

    #[test]
    fn special_ids() {
        let set = Unicharset::load_from_bytes(synthetic().as_bytes(), false).unwrap();
        assert_eq!(UNICHAR_SPACE, 0);
        assert_eq!(set.id_to_unichar(UNICHAR_SPACE), " ");
        assert!(set.contains_unichar_id(0));
        assert!(set.contains_unichar_id(4));
        assert!(!set.contains_unichar_id(5));
        assert!(!set.contains_unichar_id(INVALID_UNICHAR_ID));
        // This synthetic set has no special (broken) code installed.
        assert!(!set.has_special_codes());
    }

    #[test]
    fn broken_code_is_a_fragment() {
        // "|Broken|0|1" parses as a single-piece fragment (UNICHAR_BROKEN).
        let frag = CharFragment::parse_from_string("|Broken|0|1").expect("should parse");
        assert_eq!(frag.unichar(), "Broken");
        assert_eq!(frag.pos(), 0);
        assert_eq!(frag.total(), 1);
        // A regular char is not a fragment.
        assert!(CharFragment::parse_from_string("A").is_none());
        assert!(CharFragment::parse_from_string("标").is_none());
    }

    #[test]
    fn cleanup_expands_fi_ligature() {
        // U+FB01 (fi) -> "fi".
        assert_eq!(cleanup_string("\u{fb01}".as_bytes()), b"fi");
        // TATWEEL deleted.
        assert_eq!(cleanup_string("\u{0640}".as_bytes()), b"");
        // Ordinary text untouched.
        assert_eq!(cleanup_string("abc标".as_bytes()), "abc标".as_bytes());
    }

    #[test]
    fn rejects_truncated() {
        // Declares 3 entries but only supplies 1.
        let buf = "3\nNULL 0 NULL 0\n";
        let err = Unicharset::load_from_bytes(buf.as_bytes(), false).unwrap_err();
        assert_eq!(err.kind(), crate::error::ErrorKind::UnexpectedEof);
    }
}
