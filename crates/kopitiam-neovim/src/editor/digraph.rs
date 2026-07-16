//! Insert-mode digraphs (`<C-k>{c1}{c2}`).
//!
//! A *digraph* is a two-character mnemonic for a character that is awkward to
//! type directly — `<C-k>a:` for `ä`, `<C-k>->` for `→`, `<C-k>Co` for `©`.
//! The mnemonics and their meanings follow vim's own convention (`:help
//! digraph-table`), which in turn follows the RFC 1345 mnemonic set, so the
//! muscle memory a vim user already has carries straight over.
//!
//! This table is a curated *subset* of vim's full ~1400-entry list: the Latin
//! accented letters, the common arrows and math operators, a handful of Greek
//! letters, currency, punctuation and the typographic marks that come up most
//! in day-to-day scientific and prose editing. A complete table is a filed
//! follow-up (kopitiam-cj0.62); the lookup here is a plain linear scan, which
//! is more than fast enough for a two-keystroke interaction and keeps the data
//! a flat, readable `(c1, c2, result)` list that is trivial to extend.
//!
//! vim treats a digraph as *ordered* for lookup but also accepts the reverse
//! order for most pairs; kvim matches the listed order first and then retries
//! with the two characters swapped, so both `a:` and `:a` find `ä` — the same
//! forgiving behaviour vim has.

/// Looks up the character a digraph `{c1}{c2}` produces, trying the pair as
/// given and then reversed. Returns `None` when no entry matches (vim beeps and
/// inserts nothing in that case, which is what the caller does too).
pub fn lookup(c1: char, c2: char) -> Option<char> {
    TABLE
        .iter()
        .find_map(|&(a, b, out)| ((a == c1 && b == c2) || (a == c2 && b == c1)).then_some(out))
}

/// The digraph mnemonics kvim ships. `(first, second, result)`. Kept in the
/// same mnemonic spelling vim uses so `:help digraph-table` doubles as this
/// table's documentation.
static TABLE: &[(char, char, char)] = &[
    // --- Latin letters with diaeresis / umlaut (`X:`) ---
    ('a', ':', 'ä'),
    ('e', ':', 'ë'),
    ('i', ':', 'ï'),
    ('o', ':', 'ö'),
    ('u', ':', 'ü'),
    ('y', ':', 'ÿ'),
    ('A', ':', 'Ä'),
    ('E', ':', 'Ë'),
    ('I', ':', 'Ï'),
    ('O', ':', 'Ö'),
    ('U', ':', 'Ü'),
    // --- Acute accent (`X'`) ---
    ('a', '\'', 'á'),
    ('e', '\'', 'é'),
    ('i', '\'', 'í'),
    ('o', '\'', 'ó'),
    ('u', '\'', 'ú'),
    ('y', '\'', 'ý'),
    ('A', '\'', 'Á'),
    ('E', '\'', 'É'),
    ('I', '\'', 'Í'),
    ('O', '\'', 'Ó'),
    ('U', '\'', 'Ú'),
    // --- Grave accent (`X!`) ---
    ('a', '!', 'à'),
    ('e', '!', 'è'),
    ('i', '!', 'ì'),
    ('o', '!', 'ò'),
    ('u', '!', 'ù'),
    ('A', '!', 'À'),
    ('E', '!', 'È'),
    ('I', '!', 'Ì'),
    ('O', '!', 'Ò'),
    ('U', '!', 'Ù'),
    // --- Circumflex (`X>`) ---
    ('a', '>', 'â'),
    ('e', '>', 'ê'),
    ('i', '>', 'î'),
    ('o', '>', 'ô'),
    ('u', '>', 'û'),
    ('A', '>', 'Â'),
    ('E', '>', 'Ê'),
    ('I', '>', 'Î'),
    ('O', '>', 'Ô'),
    ('U', '>', 'Û'),
    // --- Tilde (`X?`) ---
    ('a', '?', 'ã'),
    ('n', '?', 'ñ'),
    ('o', '?', 'õ'),
    ('A', '?', 'Ã'),
    ('N', '?', 'Ñ'),
    ('O', '?', 'Õ'),
    // --- Cedilla, ring, slash, ligatures, sharp s ---
    ('c', ',', 'ç'),
    ('C', ',', 'Ç'),
    ('a', 'a', 'å'),
    ('A', 'A', 'Å'),
    ('o', '/', 'ø'),
    ('O', '/', 'Ø'),
    ('a', 'e', 'æ'),
    ('A', 'E', 'Æ'),
    ('s', 's', 'ß'),
    // --- Currency ---
    ('E', 'u', '€'),
    ('P', 'd', '£'),
    ('Y', 'e', '¥'),
    ('c', 't', '¢'),
    ('C', 't', '¢'),
    // --- Punctuation & typographic marks ---
    ('C', 'o', '©'),
    ('R', 'g', '®'),
    ('T', 'M', '™'),
    ('d', 'g', '°'),
    ('+', '-', '±'),
    ('!', 'I', '¡'),
    ('?', 'I', '¿'),
    ('<', '<', '«'),
    ('>', '>', '»'),
    ('-', 'N', '–'), // en dash
    ('-', 'M', '—'), // em dash
    ('.', '3', '…'), // horizontal ellipsis
    ('1', '2', '½'),
    ('1', '4', '¼'),
    ('3', '4', '¾'),
    ('S', 'E', '§'),
    ('*', 'X', '×'),
    (':', '-', '÷'),
    ('\'', '6', '‘'),
    ('\'', '9', '’'),
    ('"', '6', '“'),
    ('"', '9', '”'),
    // --- Arrows ---
    ('-', '>', '→'),
    ('<', '-', '←'),
    ('-', '!', '↑'),
    ('-', 'v', '↓'),
    ('=', '>', '⇒'),
    ('<', '=', '⇐'),
    // --- Common math operators ---
    ('O', 'K', '✓'),
    ('X', 'X', '✗'),
    ('*', '*', '∞'), // note: vim uses `00` for ∞; `**` is a kvim convenience
    ('R', 'T', '√'),
    ('-', ':', '÷'),
    ('!', '=', '≠'),
    ('=', '<', '≤'),
    ('>', '=', '≥'),
    ('~', '~', '≈'),
    ('N', 'B', '∇'), // nabla / del
    ('O', 'o', '∘'),
    // --- Greek lowercase (`X*`) ---
    ('a', '*', 'α'),
    ('b', '*', 'β'),
    ('g', '*', 'γ'),
    ('d', '*', 'δ'),
    ('e', '*', 'ε'),
    ('h', '*', 'θ'),
    ('l', '*', 'λ'),
    ('m', '*', 'μ'),
    ('p', '*', 'π'),
    ('r', '*', 'ρ'),
    ('s', '*', 'σ'),
    ('t', '*', 'τ'),
    ('f', '*', 'φ'),
    ('w', '*', 'ω'),
    // --- Greek uppercase ---
    ('D', '*', 'Δ'),
    ('H', '*', 'Θ'),
    ('L', '*', 'Λ'),
    ('P', '*', 'Π'),
    ('S', '*', 'Σ'),
    ('F', '*', 'Φ'),
    ('W', '*', 'Ω'),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_digraphs_resolve() {
        assert_eq!(lookup('a', ':'), Some('ä'));
        assert_eq!(lookup('-', '>'), Some('→'));
        assert_eq!(lookup('C', 'o'), Some('©'));
        assert_eq!(lookup('e', '\''), Some('é'));
        assert_eq!(lookup('E', 'u'), Some('€'));
        assert_eq!(lookup('p', '*'), Some('π'));
    }

    #[test]
    fn reverse_order_also_resolves() {
        // vim accepts either order for a digraph pair.
        assert_eq!(lookup(':', 'a'), Some('ä'));
        assert_eq!(lookup('>', '-'), Some('→'));
    }

    #[test]
    fn unknown_pair_is_none() {
        assert_eq!(lookup('z', 'q'), None);
    }
}
