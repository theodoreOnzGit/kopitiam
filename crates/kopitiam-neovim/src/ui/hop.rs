//! The hop label-jump overlay's *interaction* state — the small piece that
//! `App` owns while a hop is in flight.
//!
//! # Why hop is not an [`crate::ui::overlay::Overlay`] variant
//!
//! The file tree, pickers and harpoon menu are `Overlay`s: each claims a
//! rectangle (a sidebar or a float) and renders *itself* into that rectangle,
//! knowing nothing about the buffer underneath. Hop is the opposite shape. It
//! paints its labels **onto the word-starts of the buffer**, at the exact
//! screen cells those positions occupy — which means it needs the active
//! window's rectangle, its scroll offset, and the buffer text, all of which
//! `App::render_windows` has already computed and none of which fit the
//! `Overlay::render(frame, rect, ...)` signature. Forcing hop through that
//! seam would mean threading window geometry into an interface designed to be
//! geometry-free.
//!
//! So hop reuses the *focus discipline* the overlay layer established — while
//! a hop is active, keystrokes go here and the editor never sees them — but
//! keeps its own tiny state object and is drawn inline with the buffer. That
//! is the same judgement the overlay module itself records for the file tree:
//! reuse the model that fits, don't contort the code to share a type.

use crate::core::Position;
use crate::plugins::hop::{resolve, Hint, HopResult};

/// One keystroke's outcome while a hop is active.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HopFeed {
    /// The input still matches more than one label; the reduced set is kept
    /// (see [`HopState::visible`]) and the labels are repainted.
    Narrowed,
    /// A label was uniquely typed — jump the cursor here and end the hop.
    Jump(Position),
    /// The input matches no label (a miss), or `<Esc>` was pressed: end the
    /// hop without moving.
    Cancel,
}

/// The live state of a hop: every candidate, the label characters typed so
/// far, and the still-matching subset to highlight.
#[derive(Debug, Clone)]
pub struct HopState {
    hints: Vec<Hint>,
    input: String,
    /// The hints whose labels still match `input` — what the renderer paints.
    /// Starts as the whole set (nothing typed yet narrows nothing).
    visible: Vec<Hint>,
}

impl HopState {
    pub fn new(hints: Vec<Hint>) -> Self {
        let visible = hints.clone();
        Self { hints, input: String::new(), visible }
    }

    /// The hints to paint labels for this frame.
    pub fn visible(&self) -> &[Hint] {
        &self.visible
    }

    /// Feeds one label character.
    pub fn feed(&mut self, c: char) -> HopFeed {
        self.input.push(c);
        match resolve(&self.hints, &self.input) {
            HopResult::Jump(pos) => HopFeed::Jump(pos),
            HopResult::Narrow(remaining) => {
                self.visible = remaining;
                HopFeed::Narrowed
            }
            HopResult::NoMatch => HopFeed::Cancel,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hints() -> Vec<Hint> {
        vec![
            Hint { position: Position::new(0, 0), label: "a".into() },
            Hint { position: Position::new(0, 4), label: "sd".into() },
            Hint { position: Position::new(0, 8), label: "sf".into() },
        ]
    }

    #[test]
    fn a_unique_single_char_label_jumps() {
        let mut hop = HopState::new(hints());
        assert_eq!(hop.feed('a'), HopFeed::Jump(Position::new(0, 0)));
    }

    #[test]
    fn a_prefix_narrows_then_the_second_char_jumps() {
        let mut hop = HopState::new(hints());
        assert_eq!(hop.feed('s'), HopFeed::Narrowed);
        assert_eq!(hop.visible().len(), 2, "only the two 's*' labels remain");
        assert_eq!(hop.feed('f'), HopFeed::Jump(Position::new(0, 8)));
    }

    #[test]
    fn a_miss_cancels() {
        let mut hop = HopState::new(hints());
        assert_eq!(hop.feed('z'), HopFeed::Cancel);
    }
}
