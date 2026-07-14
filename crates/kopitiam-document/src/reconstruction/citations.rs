use std::sync::LazyLock;

use regex::Regex;

use crate::Citation;

static AUTHOR_YEAR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\([A-Z][\p{L}'-]*(?:\s+(?:and|&)\s+[A-Z][\p{L}'-]*|\s+et al\.)?,?\s+\d{4}[a-z]?\)")
        .unwrap()
});
static NUMBERED_REF: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[\d+(?:,\s*\d+)*\]").unwrap());

/// Detects citations inside already-rendered paragraph text for provenance
/// reporting. Never modifies the text -- references, DOIs, and citation
/// strings must survive verbatim.
pub(super) fn detect(text: &str) -> Vec<Citation> {
    AUTHOR_YEAR
        .find_iter(text)
        .chain(NUMBERED_REF.find_iter(text))
        .map(|m| Citation {
            text: m.as_str().to_string(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_author_year_citation() {
        let found = detect("as shown previously (Cohen et al., 1991).");
        assert_eq!(
            found,
            vec![Citation {
                text: "(Cohen et al., 1991)".to_string()
            }]
        );
    }

    #[test]
    fn detects_numbered_reference() {
        let found = detect("see the discussion in [12].");
        assert_eq!(
            found,
            vec![Citation {
                text: "[12]".to_string()
            }]
        );
    }

    #[test]
    fn plain_prose_has_no_citations() {
        assert!(detect("There is nothing to cite here.").is_empty());
    }
}
