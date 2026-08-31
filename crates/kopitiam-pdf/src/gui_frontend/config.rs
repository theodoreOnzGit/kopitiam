//! What an embedding application turns on, and -- more importantly -- what it
//! can turn **off**.
//!
//! gh-96 Phase 8. The requirement driving this, in the brief's own words:
//! *viewing/navigation must be separable from PDF modification.* A host must
//! be able to open a source document in effectively read-only mode and still
//! get navigation, search, continuous scrolling, selection and page
//! coordinates. Anything that would write to the PDF is a separate decision.
//!
//! That is not a hypothetical. A literature application opening someone's
//! downloaded paper wants every reading feature and absolutely no path by
//! which a stray keystroke rewrites the file -- while an annotating app on
//! the same reader wants the opposite.

/// Which reader features are available.
///
/// Built with a `Default` that matches `kpdf`'s standalone behaviour (all of
/// it), then narrowed:
///
/// ```
/// # use kopitiam_pdf::gui_frontend::PdfReaderConfig;
/// // A read-only literature viewer: everything to read with, nothing that writes.
/// let cfg = PdfReaderConfig::read_only();
/// assert!(cfg.search && cfg.continuous_scroll);
/// assert!(!cfg.annotations && !cfg.forms && !cfg.editing);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PdfReaderConfig {
    /// All pages in one scrolling column. Off means one page at a time.
    pub continuous_scroll: bool,
    /// Whether the thumbnail engine runs at all. Off saves the per-page
    /// thumbnail renders entirely -- worth it for a host that shows no
    /// thumbnail UI.
    pub thumbnails: bool,
    /// `/`, `?`, `n`, `N` and the background search worker.
    pub search: bool,
    /// Reflow (text) mode.
    pub reflow: bool,
    /// Vim-style navigation keys. Off leaves arrows/PageUp/PageDown, so a
    /// host whose own bindings clash does not lose navigation entirely.
    pub vim_keys: bool,
    /// The pen and eraser: **writes annotations into the PDF**.
    pub annotations: bool,
    /// Filling in AcroForm fields: **writes into the PDF**.
    pub forms: bool,
    /// Adding and deleting pages: **writes into the PDF**.
    pub editing: bool,
}

impl Default for PdfReaderConfig {
    /// Everything on -- what `kpdf` is, and the least surprising thing for a
    /// host that has not thought about it yet.
    fn default() -> PdfReaderConfig {
        PdfReaderConfig {
            continuous_scroll: true,
            thumbnails: true,
            search: true,
            reflow: true,
            vim_keys: true,
            annotations: true,
            forms: true,
            editing: true,
        }
    }
}

impl PdfReaderConfig {
    /// Every reading feature, no path that writes to the document.
    ///
    /// The brief's "required capability", as one call. Note what stays *on*:
    /// navigation, search, continuous scroll, selection, page coordinates,
    /// reflow. Read-only restricts what the reader may change, not what it
    /// may show.
    pub fn read_only() -> PdfReaderConfig {
        PdfReaderConfig {
            annotations: false,
            forms: false,
            editing: false,
            ..PdfReaderConfig::default()
        }
    }

    /// Whether any feature that modifies the PDF is enabled.
    ///
    /// The single question the reader asks before offering a save, an undo
    /// stack, or a tool that draws. One predicate rather than three checks
    /// scattered around, so adding a fourth writing feature later cannot miss
    /// a site.
    pub fn can_modify(&self) -> bool {
        self.annotations || self.forms || self.editing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_everything_which_is_what_kpdf_is() {
        let c = PdfReaderConfig::default();
        assert!(c.can_modify());
        assert!(c.continuous_scroll && c.thumbnails && c.search && c.reflow && c.vim_keys);
    }

    /// The brief's required capability: read-only must keep every *reading*
    /// feature. A read-only mode that also disabled search would be a
    /// different, worse product.
    #[test]
    fn read_only_keeps_reading_and_drops_only_writing() {
        let c = PdfReaderConfig::read_only();
        assert!(!c.can_modify(), "nothing may write to the document");
        assert!(!c.annotations && !c.forms && !c.editing);
        assert!(
            c.search && c.continuous_scroll && c.thumbnails && c.reflow && c.vim_keys,
            "read-only restricts what may change, not what may be shown"
        );
    }

    /// `can_modify` must be true if ANY writing feature is on -- a host that
    /// enables only forms still needs a save path.
    #[test]
    fn can_modify_covers_each_writing_feature_alone() {
        for narrow in [
            PdfReaderConfig {
                annotations: true,
                forms: false,
                editing: false,
                ..PdfReaderConfig::read_only()
            },
            PdfReaderConfig {
                annotations: false,
                forms: true,
                editing: false,
                ..PdfReaderConfig::read_only()
            },
            PdfReaderConfig {
                annotations: false,
                forms: false,
                editing: true,
                ..PdfReaderConfig::read_only()
            },
        ] {
            assert!(narrow.can_modify());
        }
    }
}
