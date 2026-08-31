//! The semantic boundary between the reusable reader and its host: what the
//! reader **reports**, as opposed to what it decides.
//!
//! gh-96 Phase 7. This is the narrow waist of the whole extraction, and the
//! rule that keeps it reusable is worth stating before the types:
//!
//! > The reader reports what happened to the **document**. The host decides
//! > what that *means* for the application.
//!
//! So the reader says [`ReaderAction::RegionSelected`] with a page and a
//! rectangle. Whether that region is a graph to digitise, a table to extract,
//! a formula to capture, or a note to file is the host's business entirely --
//! and deliberately unrepresentable here. An action named `DigitiseGraph`
//! would bake one downstream application's vocabulary into a general PDF
//! component, which is exactly the coupling this module exists to prevent.
//!
//! The same rule cuts the other way for policy: the reader emits
//! [`ReaderAction::SaveRequested`] rather than writing a file, and
//! [`ReaderAction::QuitRequested`] rather than closing a window. It does not
//! know where the document came from, whether the host has a file at all, or
//! whether quitting means anything in the host's UI. `kpdf` turns those into
//! a native save dialog and an eframe close; an embedder might turn the same
//! action into a tab close or ignore it outright.

use crate::mupdf::destination::Destination;
use crate::mupdf::geometry::Rect;

/// Something the reader did, or was asked to do, that the host may care
/// about.
///
/// Non-exhaustive: later phases will report more, and a host matching on this
/// should keep a `_` arm rather than break when they do.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ReaderAction {
    /// The visible page changed -- by navigation, by scrolling, or because
    /// the document was replaced under the reader.
    ///
    /// 0-based, like every page index in this crate. Emitted only on an
    /// actual change, so a host can drive a page-number display or a synced
    /// outline from it without filtering repeats itself.
    PageChanged { page: usize },

    /// The reader selected a rectangular region of a page.
    ///
    /// `rect` is in **PDF user space** (y-up, origin bottom-left), not screen
    /// pixels -- so it stays meaningful across zoom, window size and
    /// re-layout, which a screen rect would not. A host that wants pixels can
    /// map it back through
    /// [`geometry::page_to_screen`](super::geometry::page_to_screen).
    RegionSelected { page: usize, rect: Rect },

    /// An annotation was added to a page by the reader's own tools.
    ///
    /// The index is the annotation's position in that page's `/Annots`, which
    /// is how [`annot_edit`](crate::mupdf::annot_edit) addresses them.
    AnnotationCreated { page: usize, index: usize },

    /// An existing annotation was modified or erased.
    AnnotationChanged { page: usize, index: usize },

    /// A link was followed. Both in-document destinations and URIs arrive
    /// here; the reader handles an in-document jump itself and reports it,
    /// but it will **not** open a URI -- opening a browser is host policy,
    /// and a PDF is untrusted input.
    LinkActivated { destination: Destination },

    /// The in-memory document changed and no longer matches whatever the host
    /// loaded it from. A host uses this to drive an "unsaved changes" marker.
    DocumentModified,

    /// The reader was asked to save (`:w`, Ctrl+S).
    ///
    /// The reader does not know where the bytes should go -- it may have been
    /// handed a `Vec<u8>` with no file behind it at all -- so it reports the
    /// request and the host decides: overwrite in place, prompt, or refuse.
    SaveRequested,

    /// The reader was asked to save **to a new location** ("Save as...").
    ///
    /// Distinct from [`SaveRequested`](Self::SaveRequested) because the two
    /// mean different things to a host: one overwrites what it loaded, the
    /// other must ask where. A host with nowhere to write can ignore both.
    SaveAsRequested,

    /// The reader was asked to open a different document.
    ///
    /// The reader has no file picker and no notion of a filesystem -- it is
    /// handed bytes. So this is a request for the *host* to choose something
    /// and call [`load_bytes`](super::reader::PdfReader::load_bytes); a host
    /// with a fixed document simply ignores it.
    OpenRequested,

    /// The reader was asked to close (`:q`, `:wq`). Purely advisory: an
    /// embedder with the reader in a tab may close the tab, or nothing.
    QuitRequested,
}

/// What one [`ui`](super::reader::PdfReader::ui) call produced.
///
/// A struct rather than a bare `Vec` so later phases can add fields (a
/// hovered link, a requested cursor) without breaking every host that matches
/// on it.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct PdfReaderOutput {
    /// In the order they happened.
    pub actions: Vec<ReaderAction>,
}

impl PdfReaderOutput {
    pub fn new() -> PdfReaderOutput {
        PdfReaderOutput::default()
    }

    pub fn push(&mut self, action: ReaderAction) {
        self.actions.push(action);
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Whether any action of this shape was reported. Convenience for the
    /// common `if output.contains(&ReaderAction::SaveRequested)`.
    pub fn contains(&self, action: &ReaderAction) -> bool {
        self.actions.contains(action)
    }

    /// Fold another call's actions in, preserving order. Used when a host
    /// calls several reader surfaces (page pane, sidebars) in one frame and
    /// wants a single list to handle.
    pub fn extend(&mut self, other: PdfReaderOutput) {
        self.actions.extend(other.actions);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_collects_in_order_and_merges() {
        let mut a = PdfReaderOutput::new();
        assert!(a.is_empty());
        a.push(ReaderAction::PageChanged { page: 2 });
        let mut b = PdfReaderOutput::new();
        b.push(ReaderAction::SaveRequested);
        a.extend(b);
        assert_eq!(
            a.actions,
            vec![
                ReaderAction::PageChanged { page: 2 },
                ReaderAction::SaveRequested
            ]
        );
        assert!(a.contains(&ReaderAction::SaveRequested));
        assert!(!a.contains(&ReaderAction::QuitRequested));
    }

    /// A region is reported in PDF user space so it survives zoom and window
    /// resizing. This pins the *documented* contract, which is the part a
    /// downstream application builds on.
    #[test]
    fn a_region_carries_page_space_coordinates() {
        let r = Rect {
            x0: 10.0,
            y0: 20.0,
            x1: 110.0,
            y1: 220.0,
        };
        let a = ReaderAction::RegionSelected { page: 3, rect: r };
        match a {
            ReaderAction::RegionSelected { page, rect } => {
                assert_eq!(page, 3);
                assert_eq!((rect.x1 - rect.x0, rect.y1 - rect.y0), (100.0, 200.0));
            }
            _ => panic!("wrong variant"),
        }
    }
}
