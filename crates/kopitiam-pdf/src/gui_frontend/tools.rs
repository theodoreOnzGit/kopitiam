//! The annotation-tool / forms-mode state machine: which tool page input is
//! currently routed to, and the pure transitions that keep it mutually
//! exclusive with forms mode.

/// Which page-editing tool on-drag/on-click input is currently routed to.
/// Mutually exclusive with forms mode -- see [`select_tool`] and
/// [`toggle_forms_mode`] for why, and for the pure transition logic that
/// enforces it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tool {
    /// Default: clicks/drags do nothing special (the page just scrolls).
    Pan,
    /// Drag to draw an ink stroke; released on drag-stop.
    Draw,
    /// Click or drag over an annotation to delete it.
    Erase,
}

/// Switch to `requested` and turn forms mode off. Forms mode and a drawing
/// tool are deliberately mutually exclusive -- a stray ink stroke while
/// trying to fill in a text field, or a checkbox toggling under the eraser,
/// would both be surprising, and Okular's own forms toggle and its
/// annotation tools don't operate at the same time either.
///
/// Pure state transition over plain fields (no application-struct borrow),
/// so it is unit-tested directly.
pub fn select_tool(tool: &mut Tool, forms_mode: &mut bool, requested: Tool) {
    *tool = requested;
    *forms_mode = false;
}

/// Flip forms mode, and if it is turning on, drop back to [`Tool::Pan`] --
/// same mutual-exclusion reasoning as [`select_tool`].
pub fn toggle_forms_mode(tool: &mut Tool, forms_mode: &mut bool) {
    *forms_mode = !*forms_mode;
    if *forms_mode {
        *tool = Tool::Pan;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_tool_sets_the_requested_tool() {
        let mut tool = Tool::Pan;
        let mut forms_mode = false;
        select_tool(&mut tool, &mut forms_mode, Tool::Draw);
        assert_eq!(tool, Tool::Draw);
    }

    #[test]
    fn selecting_a_tool_exits_forms_mode() {
        // Mutual exclusion, direction one: picking Pen/Eraser while forms
        // mode is on must turn forms mode off, so a click can never be
        // simultaneously "draw ink" and "toggle a checkbox".
        let mut tool = Tool::Pan;
        let mut forms_mode = true;
        select_tool(&mut tool, &mut forms_mode, Tool::Erase);
        assert_eq!(tool, Tool::Erase);
        assert!(!forms_mode);
    }

    #[test]
    fn toggling_forms_mode_on_resets_tool_to_pan() {
        // Mutual exclusion, direction two: turning forms mode on while a
        // drawing tool is active must fall back to Pan, not leave Draw/Erase
        // armed alongside it.
        let mut tool = Tool::Draw;
        let mut forms_mode = false;
        toggle_forms_mode(&mut tool, &mut forms_mode);
        assert!(forms_mode);
        assert_eq!(tool, Tool::Pan);
    }

    #[test]
    fn toggling_forms_mode_off_leaves_tool_at_pan() {
        let mut tool = Tool::Pan;
        let mut forms_mode = true;
        toggle_forms_mode(&mut tool, &mut forms_mode);
        assert!(!forms_mode);
        assert_eq!(tool, Tool::Pan);
    }

    #[test]
    fn toggling_forms_mode_twice_is_idempotent_on_tool() {
        let mut tool = Tool::Pan;
        let mut forms_mode = false;
        toggle_forms_mode(&mut tool, &mut forms_mode);
        toggle_forms_mode(&mut tool, &mut forms_mode);
        assert!(!forms_mode);
        assert_eq!(tool, Tool::Pan);
    }
}
