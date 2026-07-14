use crate::{Block, Citation};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Metadata {
    pub source_pages: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Document {
    pub title: Option<String>,
    pub metadata: Metadata,
    pub blocks: Vec<Block>,

    /// The 1-based page each block **starts** on, parallel to [`Self::blocks`]
    /// (`block_pages[i]` is the page of `blocks[i]`).
    ///
    /// # Why this exists
    ///
    /// Reconstruction always *knew* the page — it builds one block list per
    /// page and then flattens them — and until now it simply threw that away.
    /// That made the Document Engine unusable for every consumer that has to
    /// **cite** what it extracted: a citation without a page is not one a
    /// reader can follow, and "provenance" that cannot be checked is a promise
    /// rather than a fact.
    ///
    /// This was found the hard way. The `kopitiam-legal` engine needed page
    /// provenance for statutes, could not get it, and worked around the gap by
    /// calling `reconstruct()` **one page at a time** so the page was known by
    /// construction — which silently gave up the cross-page paragraph merging
    /// (`merge_page_breaks`) that legal text needs constantly, since statutes
    /// split provisions across pages all the time. It paid a real cost to
    /// recover a number this crate already had.
    ///
    /// Every provenance-carrying consumer needs it — legal, insurance, health,
    /// finance, literature all cite by page — so it belongs here rather than
    /// being rediscovered and worked around five more times.
    ///
    /// # A block that spans a page break
    ///
    /// Records the page it **started** on. A paragraph merged across a page
    /// break by [`crate::reconstruct`] therefore reports the earlier page,
    /// which is the right answer for a citation: it is where a reader should
    /// begin looking.
    pub block_pages: Vec<usize>,

    /// Citations detected in paragraph text, for provenance reporting. See
    /// [`Citation`] -- these are pointers into already-rendered text, not
    /// separate content.
    pub citations: Vec<Citation>,
}

impl Document {
    /// The 1-based page that `blocks[index]` starts on.
    ///
    /// Returns `None` for an out-of-range index, and — deliberately — also for
    /// a `Document` constructed without page information (e.g. via
    /// `Default`), rather than guessing page 1. A wrong page in a citation is
    /// worse than an absent one: it sends a reader to the wrong place and looks
    /// authoritative doing it.
    pub fn page_of(&self, index: usize) -> Option<usize> {
        self.block_pages.get(index).copied()
    }

    /// Iterates blocks paired with the page each starts on.
    ///
    /// Yields `None` for the page when a `Document` carries no page
    /// information, so a consumer that requires provenance can detect that
    /// rather than silently attribute everything to page 1.
    pub fn blocks_with_pages(&self) -> impl Iterator<Item = (&Block, Option<usize>)> {
        self.blocks.iter().enumerate().map(|(i, block)| (block, self.page_of(i)))
    }
}
