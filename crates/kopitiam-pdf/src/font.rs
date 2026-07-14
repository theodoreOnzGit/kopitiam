//! Font style recovery: turning a PDF `BaseFont` PostScript name (and,
//! when available, its `FontDescriptor`) into a strongly-typed
//! [`FontStyle`] instead of leaving callers to string-match on font names.
//!
//! This module is deliberately split into two halves:
//!
//! * Everything here is a **pure function** -- no `lopdf` types, no I/O, no
//!   PDF parsing. It operates purely on already-extracted strings and
//!   numbers, which makes it exhaustively unit-testable without a PDF
//!   fixture (see the `tests` module below).
//! * [`crate::font_resources`] does the PDF-specific work of walking a
//!   document's font resource dictionaries and `FontDescriptor`s to obtain
//!   the raw inputs this module consumes.
//!
//! # Why prefer the `FontDescriptor` over the name
//!
//! A PDF's `BaseFont` name is a human-authored (or font-generator-authored)
//! label with no enforced structure beyond the subset-prefix convention (see
//! [`strip_subset_prefix`]). Two independent tools can and do choose
//! different naming conventions for the same weight -- "Bold", "Bd", "Black",
//! "Heavy", "Semibold" can all mean roughly the same thing depending on the
//! foundry, and some names carry no style suffix at all even for a bold
//! face (e.g. some auto-subsetted fonts keep only the family name).
//!
//! The `FontDescriptor` dictionary (ISO 32000-1 §9.8.1), by contrast, is a
//! *structured* description with defined semantics:
//!
//! * `/Flags` bit 7 (value `64`) is the `Italic` flag -- an unambiguous
//!   boolean asserted by whoever generated the PDF, not inferred by us.
//! * `/FontWeight` (when present) is a numeric weight on the same 100-900
//!   scale used by OpenType `usWeightClass` and CSS `font-weight`.
//! * `/StemV` (vertical stem width, in thousandths of text space) is a
//!   physical measurement of the glyph's ink -- heavier (bolder) faces have
//!   thicker stems. It is a much weaker signal than `/FontWeight` because
//!   there is no standardized "bold" threshold for it, but the PDF spec
//!   itself gives 106-118 as a typical example for a *non-bold* Times
//!   Roman-class face, so meaningfully larger values are a reasonable (if
//!   soft) indicator of a heavier weight.
//!
//! Because the descriptor encodes intent directly instead of requiring us to
//! guess from a label, [`style_from_descriptor_and_name`] always prefers it,
//! falling back to the name heuristic in [`style_from_name`] only for
//! whichever of bold/italic the descriptor left undetermined (missing
//! `FontDescriptor`, missing `/Flags`, or no weight/stem signal).

/// Structured font identity/style recovered from a PDF font resource.
///
/// Every field is an `Option` because "we could not determine this" and
/// "we determined this to be false/absent" are different facts, and
/// collapsing them would let callers silently mistake missing data for a
/// negative answer (e.g. treating an unresolved font as definitely
/// non-bold). A `FontStyle` is only ever attached to a
/// [`crate::TextSpan`] once its `BaseFont` name was actually resolved from
/// the PDF's font resources -- see `TextSpan::font_style`'s docs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FontStyle {
    /// Family name with the subset prefix removed and (best-effort) the
    /// trailing weight/style suffix stripped, e.g. `"TimesNewRoman"` from
    /// `"ABCDEF+TimesNewRoman-BoldItalic"`. `None` only if no `BaseFont`
    /// name was available at all.
    pub family: Option<String>,
    /// Whether the font renders as bold. See the module docs for how this
    /// is derived from the `FontDescriptor` and/or the font name.
    pub bold: Option<bool>,
    /// Whether the font renders as italic/oblique.
    pub italic: Option<bool>,
}

/// PDF font-weight value (on the 100-900 scale used by `/FontWeight`,
/// OpenType `usWeightClass`, and CSS `font-weight`) at or above which we
/// call a font "bold". 600 (Semibold) rather than the stricter CSS "bold"
/// keyword value of 700, because the practical purpose of this flag is
/// heading/emphasis detection in `kopitiam-document`, where Semibold and
/// heavier weights read as emphasized text.
const BOLD_FONT_WEIGHT_THRESHOLD: f64 = 600.0;

/// Approximate `/StemV` (vertical stem width) threshold above which we
/// treat a font as bold when no `/FontWeight` or `ForceBold` flag is
/// present. This is *not* part of the PDF specification -- there is no
/// standardized StemV-to-weight mapping -- so it is deliberately the
/// weakest and last-consulted signal in [`style_from_descriptor_and_name`].
/// The value is a conservative rule of thumb: regular-weight text faces
/// commonly report StemV in roughly the 60-100 range (the PDF spec's own
/// FontDescriptor example for a non-bold Times-like face uses 106), while
/// true bold cuts of the same family are usually noticeably thicker.
const BOLD_STEM_V_THRESHOLD: f64 = 140.0;

/// Case-insensitive substrings that mark a PostScript font name as bold.
/// Matched against the *whole* remaining name (after subset-prefix
/// stripping), not individual tokens, because subsetters commonly glue
/// weight and slant into one suffix (e.g. `"BoldItalic"`, `"BdIt"`).
const BOLD_MARKERS: &[&str] = &[
    "bold",
    "black",
    "heavy",
    "semibold",
    "demibold",
    "extrabold",
    "ultrabold",
    "bdit",
    "-bd",
];

/// Case-insensitive substrings that mark a PostScript font name as
/// italic/oblique.
const ITALIC_MARKERS: &[&str] = &["italic", "ital", "oblique", "obli"];

/// Additional weight/style vocabulary (beyond [`BOLD_MARKERS`] and
/// [`ITALIC_MARKERS`], which are reused here too) that identifies a
/// trailing name suffix as a style marker worth stripping when deriving
/// [`FontStyle::family`], even though it does not itself imply bold or
/// italic (e.g. `"Regular"`, `"Medium"`, `"Condensed"`).
const NON_BOLD_STYLE_SUFFIX_MARKERS: &[&str] = &[
    "regular", "regu", "medium", "medi", "light", "thin", "condensed", "cond", "narrow", "plain",
    "roman", "mt",
];

/// A PDF font-subsetting embedder is required (ISO 32000-1 §9.6.4.3) to
/// prefix the subset's PostScript name with six uppercase Latin letters and
/// a `"+"`, e.g. `"ABCDEF+TimesNewRoman-BoldItalic"`. The prefix is a
/// per-subset tag chosen by the producing application (so that two
/// different subsets of the same face embedded in one document get
/// distinct tags); it carries no family or style information itself and
/// must be removed before matching family/style vocabulary in the name.
///
/// Returns `name` unchanged if it does not match the six-uppercase-letters
/// `+` pattern (most fonts referenced by name only, e.g. the standard 14,
/// are never subset-tagged).
pub fn strip_subset_prefix(name: &str) -> &str {
    let bytes = name.as_bytes();
    if bytes.len() > 7 && bytes[..6].iter().all(u8::is_ascii_uppercase) && bytes[6] == b'+' {
        &name[7..]
    } else {
        name
    }
}

/// Derive a [`FontStyle`] purely from a `BaseFont` PostScript name, using
/// the naming-convention heuristics described in the module docs. Unlike
/// the descriptor path, this always produces a definite (`Some`) bold and
/// italic answer once a non-empty name is available: PDF font-naming
/// convention requires non-regular styles to be called out explicitly in
/// the name, so the *absence* of a bold/italic marker is itself a (weak
/// but real) negative signal, not an unknown.
pub fn style_from_name(base_font: &str) -> FontStyle {
    let unprefixed = strip_subset_prefix(base_font);
    if unprefixed.is_empty() {
        return FontStyle::default();
    }
    let lower = unprefixed.to_ascii_lowercase();
    let bold = Some(BOLD_MARKERS.iter().any(|marker| lower.contains(marker)));
    let italic = Some(ITALIC_MARKERS.iter().any(|marker| lower.contains(marker)));
    FontStyle {
        family: Some(derive_family(unprefixed, &lower)),
        bold,
        italic,
    }
}

/// Best-effort extraction of the family name from an (already
/// subset-prefix-stripped) `BaseFont` name, by removing a trailing
/// weight/style suffix when one is recognizable.
///
/// Two suffix conventions are handled:
///
/// * The PDF spec's own convention for the standard 14 fonts: family and
///   style separated by a comma, e.g. `"Helvetica,BoldOblique"`.
/// * The far more common convention among embedded/subsetted fonts: family
///   and style separated by the *last* hyphen, e.g.
///   `"NimbusRomNo9L-MediItal"` or `"TimesNewRoman-BoldItalic"` -- but only
///   when the text after that hyphen actually looks like a style suffix
///   (contains bold/italic/weight vocabulary), so that families whose own
///   name happens to contain a hyphen are left intact rather than
///   incorrectly truncated.
fn derive_family(unprefixed: &str, lower: &str) -> String {
    if let Some((base, _style)) = unprefixed.split_once(',') {
        return base.to_string();
    }
    if let Some(idx) = unprefixed.rfind('-') {
        let suffix_lower = &lower[idx + 1..];
        let looks_like_style_suffix = BOLD_MARKERS
            .iter()
            .chain(ITALIC_MARKERS)
            .chain(NON_BOLD_STYLE_SUFFIX_MARKERS)
            .any(|marker| suffix_lower.contains(marker));
        if looks_like_style_suffix {
            return unprefixed[..idx].to_string();
        }
    }
    unprefixed.to_string()
}

/// Descriptor-derived signals used to determine [`FontStyle`], extracted by
/// [`crate::font_resources`] from a PDF `FontDescriptor` dictionary. Kept
/// as a plain struct of `Option`s (no `lopdf` types) so that the merge
/// logic in [`style_from_descriptor_and_name`] stays a pure function.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DescriptorSignals {
    /// `/Flags` bit 7 (Italic), when a `/Flags` entry was present on the
    /// descriptor. `Some(false)` is a genuine "not italic" assertion, not
    /// an unknown -- `/Flags` is meant to be authoritative when present.
    pub italic_flag: Option<bool>,
    /// Whether `/Flags` bit 19 (ForceBold) was set. Only ever `None` or
    /// `Some(true)`: the bit being *unset* does not assert "not bold" (its
    /// spec meaning is "the reader need not synthesize a bold appearance",
    /// which says nothing about whether this font already looks bold).
    pub force_bold_flag: Option<bool>,
    /// Raw `/FontWeight` value, when present (100-900 scale).
    pub font_weight: Option<f64>,
    /// Raw `/StemV` value, when present.
    pub stem_v: Option<f64>,
}

/// Merge descriptor-derived signals with the name-heuristic fallback into a
/// final [`FontStyle`], preferring the descriptor whenever it has an
/// opinion. Bold is decided by the first of these that is available:
/// `/FontWeight` (most authoritative), the `ForceBold` flag, `/StemV`
/// (weakest), then the name heuristic. Italic is decided by `/Flags` bit 7
/// when present, else the name heuristic.
pub(crate) fn style_from_descriptor_and_name(
    base_font: &str,
    descriptor: DescriptorSignals,
) -> FontStyle {
    let name_style = style_from_name(base_font);

    let bold = descriptor
        .font_weight
        .map(|weight| weight >= BOLD_FONT_WEIGHT_THRESHOLD)
        .or(descriptor.force_bold_flag)
        .or(descriptor.stem_v.map(|stem_v| stem_v >= BOLD_STEM_V_THRESHOLD))
        .or(name_style.bold);

    let italic = descriptor.italic_flag.or(name_style.italic);

    FontStyle {
        family: name_style.family,
        bold,
        italic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- strip_subset_prefix --------------------------------------------

    #[test]
    fn strips_six_letter_subset_prefix() {
        assert_eq!(
            strip_subset_prefix("ABCDEF+TimesNewRoman-BoldItalic"),
            "TimesNewRoman-BoldItalic"
        );
    }

    #[test]
    fn leaves_unprefixed_name_untouched() {
        assert_eq!(strip_subset_prefix("Helvetica-Bold"), "Helvetica-Bold");
    }

    #[test]
    fn does_not_strip_lowercase_or_short_prefixes() {
        // Lowercase letters before '+' don't match the subsetting
        // convention (ISO 32000-1 requires uppercase A-Z).
        assert_eq!(strip_subset_prefix("abcdef+Font"), "abcdef+Font");
        // Fewer than six letters before '+' isn't the convention either.
        assert_eq!(strip_subset_prefix("ABC+Font"), "ABC+Font");
    }

    #[test]
    fn does_not_strip_when_name_is_only_the_prefix() {
        // "ABCDEF+" with nothing after it: `bytes.len() > 7` requires at
        // least one character past the '+', so this is left alone.
        assert_eq!(strip_subset_prefix("ABCDEF+"), "ABCDEF+");
    }

    // -- style_from_name: bold/italic detection --------------------------

    #[test]
    fn detects_plain_bold() {
        let style = style_from_name("Helvetica-Bold");
        assert_eq!(style.bold, Some(true));
        assert_eq!(style.italic, Some(false));
        assert_eq!(style.family.as_deref(), Some("Helvetica"));
    }

    #[test]
    fn detects_plain_italic() {
        let style = style_from_name("Times-Italic");
        assert_eq!(style.bold, Some(false));
        assert_eq!(style.italic, Some(true));
        assert_eq!(style.family.as_deref(), Some("Times"));
    }

    #[test]
    fn detects_glued_bold_italic_suffix() {
        let style = style_from_name("ABCDEF+TimesNewRoman-BoldItalic");
        assert_eq!(style.bold, Some(true));
        assert_eq!(style.italic, Some(true));
        assert_eq!(style.family.as_deref(), Some("TimesNewRoman"));
    }

    #[test]
    fn detects_urw_nimbus_abbreviations() {
        // URW's Nimbus fonts (Ghostscript's standard-14 substitutes) use
        // 4-letter abbreviated style suffixes rather than full words.
        let medium = style_from_name("NimbusRomNo9L-Medi");
        assert_eq!(medium.bold, Some(false), "Medium is not Bold");
        assert_eq!(medium.italic, Some(false));
        assert_eq!(medium.family.as_deref(), Some("NimbusRomNo9L"));

        let bold_italic = style_from_name("NimbusRomNo9L-MediItal");
        assert_eq!(bold_italic.italic, Some(true));
        assert_eq!(bold_italic.family.as_deref(), Some("NimbusRomNo9L"));

        let bold = style_from_name("NimbusRomNo9L-Bold");
        assert_eq!(bold.bold, Some(true));
    }

    #[test]
    fn detects_black_and_semibold_as_bold() {
        assert_eq!(style_from_name("Roboto-Black").bold, Some(true));
        assert_eq!(style_from_name("SourceSansPro-Semibold").bold, Some(true));
    }

    #[test]
    fn plain_regular_name_is_definitely_not_bold_or_italic() {
        let style = style_from_name("Arial-Regular");
        assert_eq!(style.bold, Some(false));
        assert_eq!(style.italic, Some(false));
    }

    #[test]
    fn name_with_no_style_suffix_at_all_is_not_bold_or_italic() {
        let style = style_from_name("Calibri");
        assert_eq!(style.bold, Some(false));
        assert_eq!(style.italic, Some(false));
        assert_eq!(style.family.as_deref(), Some("Calibri"));
    }

    #[test]
    fn comma_separated_standard_14_style_suffix() {
        let style = style_from_name("Helvetica,BoldOblique");
        assert_eq!(style.bold, Some(true));
        assert_eq!(style.italic, Some(true));
        assert_eq!(style.family.as_deref(), Some("Helvetica"));
    }

    #[test]
    fn empty_name_yields_default_style() {
        assert_eq!(style_from_name(""), FontStyle::default());
    }

    // -- style_from_descriptor_and_name: descriptor precedence -----------

    #[test]
    fn descriptor_italic_flag_overrides_absent_name_marker() {
        let signals = DescriptorSignals {
            italic_flag: Some(true),
            ..Default::default()
        };
        // Name gives no italic marker at all, but the descriptor knows
        // better and must win.
        let style = style_from_descriptor_and_name("MyFont-Regular", signals);
        assert_eq!(style.italic, Some(true));
    }

    #[test]
    fn descriptor_italic_false_overrides_name_heuristic_guess() {
        // Contrived: name looks italic-ish by substring match, but an
        // explicit descriptor /Flags says otherwise, and the descriptor is
        // authoritative when present.
        let signals = DescriptorSignals {
            italic_flag: Some(false),
            ..Default::default()
        };
        let style = style_from_descriptor_and_name("ItalicsFont", signals);
        assert_eq!(style.italic, Some(false));
    }

    #[test]
    fn missing_descriptor_flags_falls_back_to_name() {
        let style = style_from_descriptor_and_name("Times-Italic", DescriptorSignals::default());
        assert_eq!(style.italic, Some(true));
        assert_eq!(style.bold, Some(false));
    }

    #[test]
    fn font_weight_is_the_most_authoritative_bold_signal() {
        let signals = DescriptorSignals {
            font_weight: Some(700.0),
            ..Default::default()
        };
        // Name doesn't even hint at bold, but /FontWeight settles it.
        let style = style_from_descriptor_and_name("MyFont-Regular", signals);
        assert_eq!(style.bold, Some(true));
    }

    #[test]
    fn font_weight_below_threshold_is_not_bold_even_with_name_marker() {
        let signals = DescriptorSignals {
            font_weight: Some(400.0),
            ..Default::default()
        };
        // The descriptor's numeric weight is authoritative and wins over
        // a (here deliberately misleading) name.
        let style = style_from_descriptor_and_name("MyFont-Bold", signals);
        assert_eq!(style.bold, Some(false));
    }

    #[test]
    fn force_bold_flag_is_used_when_no_font_weight_present() {
        let signals = DescriptorSignals {
            force_bold_flag: Some(true),
            ..Default::default()
        };
        let style = style_from_descriptor_and_name("MyFont-Regular", signals);
        assert_eq!(style.bold, Some(true));
    }

    #[test]
    fn stem_v_is_the_weakest_bold_signal_used_last() {
        let heavy_stem = DescriptorSignals {
            stem_v: Some(180.0),
            ..Default::default()
        };
        assert_eq!(
            style_from_descriptor_and_name("MyFont-Regular", heavy_stem).bold,
            Some(true)
        );

        let light_stem = DescriptorSignals {
            stem_v: Some(80.0),
            ..Default::default()
        };
        assert_eq!(
            style_from_descriptor_and_name("MyFont-Regular", light_stem).bold,
            Some(false)
        );
    }

    #[test]
    fn font_weight_takes_priority_over_force_bold_and_stem_v() {
        let signals = DescriptorSignals {
            font_weight: Some(300.0), // light: definitely not bold
            force_bold_flag: Some(true),
            stem_v: Some(200.0),
            ..Default::default()
        };
        let style = style_from_descriptor_and_name("MyFont", signals);
        assert_eq!(style.bold, Some(false));
    }
}
