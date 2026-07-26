//! Ported from Tesseract `src/ccutil/tessdatamanager.cpp` +
//! `src/ccutil/tessdatamanager.h` (commit db0ec62, Apache-2.0, © 2009 Google
//! Inc., Author: Daria Antonova), translated to Rust for KOPITIAM
//! (AGPL-3.0-only). Close adaptation: the on-disk container layout, the
//! component-type enumeration, the endianness auto-detection, and the offset-table
//! walk follow Tesseract exactly; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md.
//!
//! # The `.traineddata` container
//!
//! A `.traineddata` file is Tesseract's "proprietary" (non-archive) container: a
//! flat table of component blobs, one slot per [`TessdataType`], addressed by an
//! offset table. On disk (`TessdataManager::LoadMemBuffer`/`Serialize`,
//! `tessdatamanager.cpp:110`/`:189`):
//!
//! ```text
//! ┌───────────────────────────────┐  byte 0
//! │ num_entries : int32            │  (count of offset-table slots)
//! ├───────────────────────────────┤  byte 4
//! │ offset_table : int64[num_entries]                             │
//! │   offset_table[i] = absolute byte offset of component i,      │
//! │                     or -1 if component i is absent            │
//! ├───────────────────────────────┤  byte 4 + num_entries*8
//! │ component payloads, concatenated in ascending slot order      │
//! └───────────────────────────────┘  byte size
//! ```
//!
//! Component `i` occupies `offset_table[i] .. next`, where `next` is the offset of
//! the following **present** slot (offsets of `-1` are skipped), or the end of the
//! file for the last present slot. This port stores each component as a byte
//! **range** into the borrowed buffer, so [`TessdataManager::component`] borrows
//! the payload with no copy — where Tesseract copies every blob into a
//! `std::vector<char> entries_[i]`.
//!
//! # Endianness auto-detection
//!
//! `num_entries` is read host-native first; if it exceeds
//! [`MAX_NUM_TESSDATA_ENTRIES`] (1000) the file must be byte-swapped relative to
//! the host, so the swap flag is set (propagated to the [`TFile`] so the `int64`
//! offset table is read correctly) and `num_entries` is byte-reversed. Component
//! payloads are raw bytes and are never swapped; the swap flag is remembered and
//! handed to any per-component [`TFile`] reader ([`TessdataManager::component_reader`]),
//! exactly as `GetComponent` does (`tessdatamanager.cpp:249`).
//!
//! # Read path only
//!
//! Only container **parsing** is ported. The write/combine side
//! (`Serialize`/`SaveFile`/`CombineDataFiles`/`OverwriteComponents`/
//! `ExtractToFile`), the lazy-`Init`-from-filename path, the libarchive alternate
//! format (`LoadArchiveFile`), and the filename→type suffix matching used only by
//! the combiner (`TessdataTypeFromFileName`) are **not** ported — KOPITIAM reads
//! containers, it does not build them. The [`TessdataType`] ⇄ suffix table itself
//! *is* ported (it is the component enumeration), just not the filename matcher.

use crate::error::{Error, Result};
use crate::serialis::TFile;
use core::ops::Range;
use std::borrow::Cow;

/// The number of component slots (`TESSDATA_NUM_ENTRIES`).
///
/// Tesseract: `TESSDATA_NUM_ENTRIES` (tessdatamanager.h:84) — the terminator of
/// the `TessdataType` enum, i.e. one past the last real type.
pub const NUM_ENTRIES: usize = 24;

/// If a file declares more entries than this even before endianness correction,
/// it is byte-swapped relative to the host.
///
/// Tesseract: `kMaxNumTessdataEntries` (tessdatamanager.h:125).
pub const MAX_NUM_TESSDATA_ENTRIES: u32 = 1000;

/// A component type stored in a `.traineddata` container.
///
/// Discriminants are the exact `TessdataType` enum values
/// (tessdatamanager.h:58); each doubles as the component's slot index in the
/// offset table, so `ty as usize` indexes both the offset table and
/// [`FILE_SUFFIXES`]. Deprecated slots (10–12) are retained so later indices stay
/// correctly aligned on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum TessdataType {
    /// `TESSDATA_LANG_CONFIG` (0) — language config, suffix `config`.
    LangConfig = 0,
    /// `TESSDATA_UNICHARSET` (1) — legacy unicharset, suffix `unicharset`.
    Unicharset = 1,
    /// `TESSDATA_AMBIGS` (2) — ambiguities, suffix `unicharambigs`.
    Ambigs = 2,
    /// `TESSDATA_INTTEMP` (3) — built-in templates, suffix `inttemp`.
    Inttemp = 3,
    /// `TESSDATA_PFFMTABLE` (4) — built-in cutoffs, suffix `pffmtable`.
    Pffmtable = 4,
    /// `TESSDATA_NORMPROTO` (5) — normalization prototypes, suffix `normproto`.
    Normproto = 5,
    /// `TESSDATA_PUNC_DAWG` (6) — punctuation dawg, suffix `punc-dawg`.
    PuncDawg = 6,
    /// `TESSDATA_SYSTEM_DAWG` (7) — system word dawg, suffix `word-dawg`.
    SystemDawg = 7,
    /// `TESSDATA_NUMBER_DAWG` (8) — number dawg, suffix `number-dawg`.
    NumberDawg = 8,
    /// `TESSDATA_FREQ_DAWG` (9) — frequent-word dawg, suffix `freq-dawg`.
    FreqDawg = 9,
    /// `TESSDATA_FIXED_LENGTH_DAWGS` (10, **deprecated**), suffix
    /// `fixed-length-dawgs`.
    FixedLengthDawgs = 10,
    /// `TESSDATA_CUBE_UNICHARSET` (11, **deprecated**), suffix `cube-unicharset`.
    CubeUnicharset = 11,
    /// `TESSDATA_CUBE_SYSTEM_DAWG` (12, **deprecated**), suffix `cube-word-dawg`.
    CubeSystemDawg = 12,
    /// `TESSDATA_SHAPE_TABLE` (13) — shape table, suffix `shapetable`.
    ShapeTable = 13,
    /// `TESSDATA_BIGRAM_DAWG` (14) — bigram dawg, suffix `bigram-dawg`.
    BigramDawg = 14,
    /// `TESSDATA_UNAMBIG_DAWG` (15) — unambiguous dawg, suffix `unambig-dawg`.
    UnambigDawg = 15,
    /// `TESSDATA_PARAMS_MODEL` (16) — params model, suffix `params-model`.
    ParamsModel = 16,
    /// `TESSDATA_LSTM` (17) — the LSTM recognition model, suffix `lstm`.
    Lstm = 17,
    /// `TESSDATA_LSTM_PUNC_DAWG` (18) — LSTM punctuation dawg, suffix
    /// `lstm-punc-dawg`.
    LstmPuncDawg = 18,
    /// `TESSDATA_LSTM_SYSTEM_DAWG` (19) — LSTM system dawg, suffix
    /// `lstm-word-dawg`.
    LstmSystemDawg = 19,
    /// `TESSDATA_LSTM_NUMBER_DAWG` (20) — LSTM number dawg, suffix
    /// `lstm-number-dawg`.
    LstmNumberDawg = 20,
    /// `TESSDATA_LSTM_UNICHARSET` (21) — the LSTM unicharset, suffix
    /// `lstm-unicharset`.
    LstmUnicharset = 21,
    /// `TESSDATA_LSTM_RECODER` (22) — the LSTM recoder, suffix `lstm-recoder`.
    LstmRecoder = 22,
    /// `TESSDATA_VERSION` (23) — the version string, suffix `version`.
    Version = 23,
}

impl TessdataType {
    /// Every component type, in slot order.
    pub const ALL: [TessdataType; NUM_ENTRIES] = [
        TessdataType::LangConfig,
        TessdataType::Unicharset,
        TessdataType::Ambigs,
        TessdataType::Inttemp,
        TessdataType::Pffmtable,
        TessdataType::Normproto,
        TessdataType::PuncDawg,
        TessdataType::SystemDawg,
        TessdataType::NumberDawg,
        TessdataType::FreqDawg,
        TessdataType::FixedLengthDawgs,
        TessdataType::CubeUnicharset,
        TessdataType::CubeSystemDawg,
        TessdataType::ShapeTable,
        TessdataType::BigramDawg,
        TessdataType::UnambigDawg,
        TessdataType::ParamsModel,
        TessdataType::Lstm,
        TessdataType::LstmPuncDawg,
        TessdataType::LstmSystemDawg,
        TessdataType::LstmNumberDawg,
        TessdataType::LstmUnicharset,
        TessdataType::LstmRecoder,
        TessdataType::Version,
    ];

    /// The slot index / offset-table position for this type.
    pub const fn as_index(self) -> usize {
        self as usize
    }

    /// The component type at slot `index`, or `None` if out of range.
    pub const fn from_index(index: usize) -> Option<TessdataType> {
        if index < NUM_ENTRIES {
            Some(TessdataType::ALL[index])
        } else {
            None
        }
    }

    /// The canonical file suffix for this component.
    ///
    /// Tesseract: `kTessdataFileSuffixes[i]` (tessdatamanager.h:91).
    pub const fn suffix(self) -> &'static str {
        FILE_SUFFIXES[self as usize]
    }
}

/// `kTessdataFileSuffixes`: the file suffix for each `TessdataType` slot, in
/// index order (tessdatamanager.h:91).
pub const FILE_SUFFIXES: [&str; NUM_ENTRIES] = [
    "config",             // 0  TESSDATA_LANG_CONFIG
    "unicharset",         // 1  TESSDATA_UNICHARSET
    "unicharambigs",      // 2  TESSDATA_AMBIGS
    "inttemp",            // 3  TESSDATA_INTTEMP
    "pffmtable",          // 4  TESSDATA_PFFMTABLE
    "normproto",          // 5  TESSDATA_NORMPROTO
    "punc-dawg",          // 6  TESSDATA_PUNC_DAWG
    "word-dawg",          // 7  TESSDATA_SYSTEM_DAWG
    "number-dawg",        // 8  TESSDATA_NUMBER_DAWG
    "freq-dawg",          // 9  TESSDATA_FREQ_DAWG
    "fixed-length-dawgs", // 10 TESSDATA_FIXED_LENGTH_DAWGS (deprecated)
    "cube-unicharset",    // 11 TESSDATA_CUBE_UNICHARSET (deprecated)
    "cube-word-dawg",     // 12 TESSDATA_CUBE_SYSTEM_DAWG (deprecated)
    "shapetable",         // 13 TESSDATA_SHAPE_TABLE
    "bigram-dawg",        // 14 TESSDATA_BIGRAM_DAWG
    "unambig-dawg",       // 15 TESSDATA_UNAMBIG_DAWG
    "params-model",       // 16 TESSDATA_PARAMS_MODEL
    "lstm",               // 17 TESSDATA_LSTM
    "lstm-punc-dawg",     // 18 TESSDATA_LSTM_PUNC_DAWG
    "lstm-word-dawg",     // 19 TESSDATA_LSTM_SYSTEM_DAWG
    "lstm-number-dawg",   // 20 TESSDATA_LSTM_NUMBER_DAWG
    "lstm-unicharset",    // 21 TESSDATA_LSTM_UNICHARSET
    "lstm-recoder",       // 22 TESSDATA_LSTM_RECODER
    "version",            // 23 TESSDATA_VERSION
];

/// A parsed `.traineddata` container, borrowing the source bytes.
///
/// Mirrors Tesseract's `TessdataManager` in its loaded state: a swap flag plus,
/// per component slot, the byte range of that component within the file. Absent
/// components (offset `-1`) and present-but-empty components (Tesseract stores a
/// zero-length `entries_[i]`, which every query treats as `.empty()`, i.e.
/// unavailable) are both represented as `None`.
///
/// Tesseract: `class TessdataManager` (tessdatamanager.h:127).
#[derive(Debug)]
pub struct TessdataManager<'a> {
    data: &'a [u8],
    swap: bool,
    /// Byte range of each present, non-empty component; `None` otherwise.
    entries: [Option<Range<usize>>; NUM_ENTRIES],
}

impl<'a> TessdataManager<'a> {
    /// Parse a `.traineddata` container from an in-memory buffer.
    ///
    /// Tesseract: `TessdataManager::LoadMemBuffer` (tessdatamanager.cpp:110),
    /// restricted to the proprietary (non-archive) format. Returns an [`Error`]
    /// for every structural fault the C signals with `return false`.
    pub fn from_bytes(data: &'a [u8]) -> Result<Self> {
        let size = data.len();
        let mut fp = TFile::new(data);

        // Read the entry count, then auto-detect endianness: a count larger than
        // the sane ceiling means the file is byte-swapped relative to the host.
        let mut num_entries = fp.read_u32()?;
        let swap = num_entries > MAX_NUM_TESSDATA_ENTRIES;
        fp.set_swap(swap);
        if swap {
            // Tesseract: ReverseN(&num_entries, sizeof(num_entries)).
            num_entries = num_entries.swap_bytes();
        }
        if num_entries > MAX_NUM_TESSDATA_ENTRIES {
            return Err(Error::limit(format!(
                "traineddata declares {num_entries} entries, exceeding the {MAX_NUM_TESSDATA_ENTRIES} ceiling"
            )));
        }
        let num_entries = num_entries as usize;

        // Read the int64 offset table (honouring the swap flag just set).
        let mut offset_table = vec![0i64; num_entries];
        fp.read_i64_slice(&mut offset_table)?;

        // Walk the offset table, deriving each present component's byte range from
        // its offset and the next present offset. Slots beyond NUM_ENTRIES are
        // ignored (Tesseract loops `i < num_entries && i < TESSDATA_NUM_ENTRIES`).
        let size_i64 = size as i64;
        let mut entries: [Option<Range<usize>>; NUM_ENTRIES] = core::array::from_fn(|_| None);
        let limit = num_entries.min(NUM_ENTRIES);
        for i in 0..limit {
            if offset_table[i] < 0 {
                continue;
            }
            if offset_table[i] > size_i64 {
                return Err(Error::format(format!(
                    "component {i} offset {} past end {size}",
                    offset_table[i]
                )));
            }
            // Default: this component runs to the end of the file.
            let mut entry_size = size_i64 - offset_table[i];
            // Find the next present slot; its offset bounds this component.
            let mut j = i + 1;
            while j < num_entries && offset_table[j] == -1 {
                j += 1;
            }
            if j < num_entries {
                if offset_table[j] < 0 || offset_table[j] > size_i64 {
                    return Err(Error::format(format!(
                        "component {j} offset {} out of range for end {size}",
                        offset_table[j]
                    )));
                }
                entry_size = offset_table[j] - offset_table[i];
            }
            if entry_size < 0 {
                return Err(Error::format(format!(
                    "component {i} has negative size {entry_size}"
                )));
            }
            // Tesseract stores a zero-length blob here, which every query treats
            // as `.empty()` (unavailable); we collapse it to `None`, which is
            // behaviourally identical for the read/query API and keeps
            // `component()` from handing back an empty slice.
            if entry_size > 0 {
                let start = offset_table[i] as usize;
                let len = entry_size as usize;
                entries[i] = Some(start..start + len);
            }
        }

        Ok(TessdataManager {
            data,
            swap,
            entries,
        })
    }

    /// Whether the file was byte-swapped relative to the host.
    ///
    /// Tesseract: `TessdataManager::swap` (tessdatamanager.h:134).
    pub const fn swap(&self) -> bool {
        self.swap
    }

    /// Whether the given component is present (and non-empty).
    ///
    /// Tesseract: `TessdataManager::IsComponentAvailable` (tessdatamanager.h:166),
    /// which is `!entries_[type].empty()`.
    pub fn is_component_available(&self, ty: TessdataType) -> bool {
        self.entries[ty as usize].is_some()
    }

    /// Borrow the raw bytes of the given component, or `None` if absent/empty.
    ///
    /// Tesseract: `TessdataManager::GetComponent` (tessdatamanager.cpp:249), which
    /// opens a `TFile` over `entries_[type]`. Here we hand back the byte slice
    /// directly; see [`TessdataManager::component_reader`] for a [`TFile`] view.
    pub fn component(&self, ty: TessdataType) -> Option<&'a [u8]> {
        self.entries[ty as usize]
            .clone()
            .map(|range| &self.data[range])
    }

    /// A [`TFile`] reader positioned over the given component, with the container's
    /// swap flag applied — the exact object `GetComponent` yields.
    ///
    /// Tesseract: `TessdataManager::GetComponent` (tessdatamanager.cpp:249):
    /// `fp->Open(&entries_[type][0], size); fp->set_swap(swap_)`.
    pub fn component_reader(&self, ty: TessdataType) -> Option<TFile<'a>> {
        self.component(ty).map(|bytes| {
            let mut fp = TFile::new(bytes);
            fp.set_swap(self.swap);
            fp
        })
    }

    /// The component types present in this container, in slot order.
    pub fn present_components(&self) -> Vec<TessdataType> {
        TessdataType::ALL
            .into_iter()
            .filter(|&ty| self.is_component_available(ty))
            .collect()
    }

    /// Whether the base (legacy) Tesseract components are present.
    ///
    /// Tesseract: `TessdataManager::IsBaseAvailable` (tessdatamanager.h:182):
    /// unicharset **and** inttemp both present.
    pub fn is_base_available(&self) -> bool {
        self.is_component_available(TessdataType::Unicharset)
            && self.is_component_available(TessdataType::Inttemp)
    }

    /// Whether the LSTM recognition model is present.
    ///
    /// Tesseract: `TessdataManager::IsLSTMAvailable` (tessdatamanager.h:187).
    pub fn is_lstm_available(&self) -> bool {
        self.is_component_available(TessdataType::Lstm)
    }

    /// The container's version string.
    ///
    /// Tesseract: `TessdataManager::VersionString` (tessdatamanager.cpp:260),
    /// combined with `LoadMemBuffer`'s fallback: when the `TESSDATA_VERSION`
    /// component is absent the version is `"Pre-4.0.0"`. The stored bytes are not
    /// UTF-8 validated by Tesseract, so an invalid sequence is decoded lossily.
    pub fn version(&self) -> Cow<'a, str> {
        match self.component(TessdataType::Version) {
            Some(bytes) => String::from_utf8_lossy(bytes),
            None => Cow::Borrowed("Pre-4.0.0"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Component payloads used by the synthetic container.
    const UNICHARSET: &[u8] = b"UNICHARSET-DATA";
    const LSTM: &[u8] = b"LSTMMODEL";
    const LSTM_UNI: &[u8] = b"LSTM-UNI";
    const LSTM_REC: &[u8] = b"RECODER";
    const VERSION: &[u8] = b"5.0.0-test";

    /// Build a synthetic proprietary `.traineddata` buffer with a full 24-slot
    /// offset table and five present components. If `swapped`, every integer is
    /// byte-reversed relative to the host so the parser's endianness
    /// auto-detection must kick in.
    fn build_synthetic(swapped: bool) -> Vec<u8> {
        let header_len = 4 + NUM_ENTRIES * 8; // int32 count + int64[24] table

        // Present components, in ascending slot order (matching how Serialize
        // lays them out). (slot index, payload).
        let present: [(usize, &[u8]); 5] = [
            (TessdataType::Unicharset as usize, UNICHARSET),
            (TessdataType::Lstm as usize, LSTM),
            (TessdataType::LstmUnicharset as usize, LSTM_UNI),
            (TessdataType::LstmRecoder as usize, LSTM_REC),
            (TessdataType::Version as usize, VERSION),
        ];

        // Assign absolute offsets to the present slots.
        let mut offset_table = [-1i64; NUM_ENTRIES];
        let mut cursor = header_len as i64;
        for (slot, payload) in present {
            offset_table[slot] = cursor;
            cursor += payload.len() as i64;
        }

        // Native or byte-reversed encoders, per `swapped`.
        let enc_u32 = |v: u32| -> [u8; 4] {
            let mut b = v.to_ne_bytes();
            if swapped {
                b.reverse();
            }
            b
        };
        let enc_i64 = |v: i64| -> [u8; 8] {
            let mut b = v.to_ne_bytes();
            if swapped {
                b.reverse();
            }
            b
        };

        let mut buf = Vec::new();
        buf.extend_from_slice(&enc_u32(NUM_ENTRIES as u32));
        for &off in &offset_table {
            buf.extend_from_slice(&enc_i64(off));
        }
        // Payloads are raw bytes, never swapped.
        for (_, payload) in present {
            buf.extend_from_slice(payload);
        }
        assert_eq!(buf.len(), cursor as usize);
        buf
    }

    fn assert_parsed(mgr: &TessdataManager, expect_swap: bool) {
        assert_eq!(mgr.swap(), expect_swap);
        assert_eq!(mgr.version(), "5.0.0-test");

        // Present components and their exact byte ranges.
        assert_eq!(mgr.component(TessdataType::Unicharset), Some(UNICHARSET));
        assert_eq!(mgr.component(TessdataType::Lstm), Some(LSTM));
        assert_eq!(mgr.component(TessdataType::LstmUnicharset), Some(LSTM_UNI));
        assert_eq!(mgr.component(TessdataType::LstmRecoder), Some(LSTM_REC));
        assert_eq!(mgr.component(TessdataType::Version), Some(VERSION));

        // Absent components.
        assert_eq!(mgr.component(TessdataType::LangConfig), None);
        assert_eq!(mgr.component(TessdataType::Inttemp), None);
        assert!(!mgr.is_component_available(TessdataType::Inttemp));

        // Availability predicates.
        assert!(mgr.is_lstm_available());
        assert!(!mgr.is_base_available()); // has unicharset but no inttemp

        // The full present list, in slot order.
        assert_eq!(
            mgr.present_components(),
            vec![
                TessdataType::Unicharset,
                TessdataType::Lstm,
                TessdataType::LstmUnicharset,
                TessdataType::LstmRecoder,
                TessdataType::Version,
            ]
        );
    }

    #[test]
    fn parses_synthetic_container() {
        let buf = build_synthetic(false);
        let mgr = TessdataManager::from_bytes(&buf).unwrap();
        assert_parsed(&mgr, false);
    }

    #[test]
    fn parses_byte_swapped_container() {
        let buf = build_synthetic(true);
        let mgr = TessdataManager::from_bytes(&buf).unwrap();
        assert_parsed(&mgr, true);
    }

    #[test]
    fn component_reader_carries_swap_flag() {
        let buf = build_synthetic(true);
        let mgr = TessdataManager::from_bytes(&buf).unwrap();
        let reader = mgr.component_reader(TessdataType::Lstm).unwrap();
        assert!(reader.swap());
        // A reader for a raw payload sees the unswapped bytes.
        let mut reader = mgr.component_reader(TessdataType::Lstm).unwrap();
        assert_eq!(reader.read_bytes(LSTM.len()).unwrap(), LSTM);
    }

    #[test]
    fn enum_discriminants_and_suffixes_match_header() {
        // Discriminants: the values esp. called out in the brief.
        assert_eq!(TessdataType::Lstm as usize, 17);
        assert_eq!(TessdataType::LstmUnicharset as usize, 21);
        assert_eq!(TessdataType::LstmRecoder as usize, 22);
        assert_eq!(TessdataType::Version as usize, 23);
        assert_eq!(NUM_ENTRIES, 24);

        // Suffixes match kTessdataFileSuffixes exactly.
        assert_eq!(TessdataType::Unicharset.suffix(), "unicharset");
        assert_eq!(TessdataType::Lstm.suffix(), "lstm");
        assert_eq!(TessdataType::LstmUnicharset.suffix(), "lstm-unicharset");
        assert_eq!(TessdataType::LstmRecoder.suffix(), "lstm-recoder");
        assert_eq!(TessdataType::PuncDawg.suffix(), "punc-dawg");
        assert_eq!(TessdataType::SystemDawg.suffix(), "word-dawg");
        assert_eq!(TessdataType::Version.suffix(), "version");

        // ALL / from_index / as_index are mutually consistent for every slot.
        for (i, ty) in TessdataType::ALL.into_iter().enumerate() {
            assert_eq!(ty.as_index(), i);
            assert_eq!(TessdataType::from_index(i), Some(ty));
            assert_eq!(ty.suffix(), FILE_SUFFIXES[i]);
        }
        assert_eq!(TessdataType::from_index(NUM_ENTRIES), None);
    }

    #[test]
    fn missing_version_falls_back_to_pre_4() {
        // A minimal container: 1 present component (unicharset), no version slot.
        // Use a small offset table (num_entries = 2) to exercise the short-table
        // path too.
        let num_entries = 2u32;
        let header_len = 4 + num_entries as usize * 8;
        let payload = b"U";
        let mut offset_table = [-1i64; 2];
        offset_table[TessdataType::Unicharset as usize] = header_len as i64;

        let mut buf = Vec::new();
        buf.extend_from_slice(&num_entries.to_ne_bytes());
        for off in offset_table {
            buf.extend_from_slice(&off.to_ne_bytes());
        }
        buf.extend_from_slice(payload);

        let mgr = TessdataManager::from_bytes(&buf).unwrap();
        assert_eq!(mgr.version(), "Pre-4.0.0");
        assert_eq!(mgr.component(TessdataType::Unicharset), Some(&payload[..]));
        assert!(!mgr.is_lstm_available());
    }

    #[test]
    fn rejects_too_many_entries() {
        // A count that is huge in BOTH endiannesses so the swap correction can't
        // rescue it. 0x00FF00FF reversed is 0xFF00FF00 — both > 1000.
        let n: u32 = 0x00FF_00FF;
        let mut buf = n.to_ne_bytes().to_vec();
        buf.extend_from_slice(&[0u8; 64]);
        let err = TessdataManager::from_bytes(&buf).unwrap_err();
        assert_eq!(err.kind(), crate::error::ErrorKind::Limit);
    }

    #[test]
    fn rejects_offset_past_end() {
        // num_entries = 1, offset_table[0] points beyond the file.
        let num_entries = 1u32;
        let mut buf = Vec::new();
        buf.extend_from_slice(&num_entries.to_ne_bytes());
        buf.extend_from_slice(&9999i64.to_ne_bytes());
        let err = TessdataManager::from_bytes(&buf).unwrap_err();
        assert_eq!(err.kind(), crate::error::ErrorKind::Format);
    }

    #[test]
    fn truncated_header_is_unexpected_eof() {
        // Declares 3 entries but the offset table is cut short.
        let mut buf = Vec::new();
        buf.extend_from_slice(&3u32.to_ne_bytes());
        buf.extend_from_slice(&0i64.to_ne_bytes()); // only 1 of 3 offsets
        let err = TessdataManager::from_bytes(&buf).unwrap_err();
        assert_eq!(err.kind(), crate::error::ErrorKind::UnexpectedEof);
    }

    /// Optional real-file smoke test. Ignored by default; runs only when
    /// `KOPITIAM_TEST_TRAINEDDATA` names a real `eng.traineddata`. Downloads
    /// nothing.
    #[test]
    #[ignore = "set KOPITIAM_TEST_TRAINEDDATA to a real eng.traineddata to run"]
    fn parses_real_traineddata() {
        let path = std::env::var("KOPITIAM_TEST_TRAINEDDATA")
            .expect("KOPITIAM_TEST_TRAINEDDATA must point at a real eng.traineddata");
        let data = std::fs::read(&path).expect("failed to read KOPITIAM_TEST_TRAINEDDATA file");
        let mgr = TessdataManager::from_bytes(&data).expect("failed to parse traineddata");
        assert!(
            mgr.is_component_available(TessdataType::Lstm),
            "real traineddata should contain an LSTM model"
        );
        assert!(
            mgr.is_component_available(TessdataType::LstmUnicharset),
            "real traineddata should contain an LSTM unicharset"
        );
    }
}
