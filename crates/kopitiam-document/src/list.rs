#[derive(Debug, Clone, PartialEq)]
pub struct List {
    pub ordered: bool,
    pub items: Vec<String>,
    /// Nesting depth of each item, **parallel to `items`** (`depths[i]` is the
    /// indent level of `items[i]`, `0` = top level). Marker-style list nesting
    /// (`kopitiam_token_max.md` task #16): reconstruction infers depth from the
    /// left-edge x-indentation of each item's line and the renderer indents
    /// accordingly.
    ///
    /// # Why a parallel `Vec`, not `Vec<ListItem>`
    ///
    /// `items` stays `Vec<String>` so existing readers that iterate the item
    /// text unchanged (e.g. `kopitiam-insurance`'s clause ingest, which appends
    /// each item as a line) keep compiling and behaving exactly as before. A
    /// flat list built by hand can leave `depths` **empty**, which
    /// [`List::depth`] reads as "every item is top level" — so a `Default`/
    /// hand-constructed list needs no depth bookkeeping.
    ///
    /// Invariant: `depths` is either empty (all top level) or exactly
    /// `items.len()` long. [`List::depth`] upholds the "empty means flat"
    /// reading regardless.
    pub depths: Vec<usize>,
}

impl List {
    /// A flat list (every item at top level), the shape hand-built call sites
    /// and pre-nesting reconstruction produce. `depths` is left empty, which
    /// [`List::depth`] reads as all-zero.
    pub fn flat(ordered: bool, items: Vec<String>) -> Self {
        Self {
            ordered,
            items,
            depths: Vec::new(),
        }
    }

    /// Nesting depth of item `i` (`0` = top level). Returns `0` for any index
    /// when `depths` is empty (a flat list) or out of range, so callers never
    /// have to special-case the flat shape.
    pub fn depth(&self, i: usize) -> usize {
        self.depths.get(i).copied().unwrap_or(0)
    }
}
