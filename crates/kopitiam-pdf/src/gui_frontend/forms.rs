//! Forms-mode UI helpers: which visual treatment a form field gets, the
//! colours behind that treatment, and the "does this keypress commit the
//! in-place field editor" predicate an `egui::TextEdit`-based form editor
//! needs.

use crate::mupdf::form::FieldKind;

/// Which visual treatment (if any) a field gets while forms mode is on --
/// and, implicitly, whether a click on it does anything at all. A caller's
/// click dispatch (`KpdfApp::handle_forms_click` in the `kpdf` binary)
/// should mirror this exactly, so the highlight never promises an
/// interaction the click handler doesn't deliver:
///
/// * `Text`, not read-only -- [`FieldHighlight::Editable`]: click opens an
///   in-place editor.
/// * `Checkbox`/`Radio`, not read-only -- [`FieldHighlight::Toggleable`]:
///   click flips it.
/// * any kind, read-only -- [`FieldHighlight::ReadOnly`]: still shown (so
///   the user can see a value is there) but visually muted, since a click
///   just reports "read-only" rather than doing anything.
/// * `Combobox`/`Listbox`, not read-only -- [`FieldHighlight::Unsupported`]:
///   [`crate::mupdf::form::set_field_value`] refuses these unconditionally
///   in this release, so the highlight says "there is a field here" without
///   promising an editor a click will never produce.
/// * `Button`/`Signature`/`Unknown` -- [`FieldHighlight::None`]: no text
///   value, no toggle, nothing to do with a click here, so no highlight at
///   all rather than one that does nothing when clicked.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FieldHighlight {
    Editable,
    Toggleable,
    ReadOnly,
    Unsupported,
    None,
}

pub fn field_highlight_kind(kind: FieldKind, read_only: bool) -> FieldHighlight {
    if read_only {
        return match kind {
            FieldKind::Button | FieldKind::Signature | FieldKind::Unknown => FieldHighlight::None,
            _ => FieldHighlight::ReadOnly,
        };
    }
    match kind {
        FieldKind::Text => FieldHighlight::Editable,
        FieldKind::Checkbox | FieldKind::Radio => FieldHighlight::Toggleable,
        FieldKind::Combobox | FieldKind::Listbox => FieldHighlight::Unsupported,
        FieldKind::Button | FieldKind::Signature | FieldKind::Unknown => FieldHighlight::None,
    }
}

/// Fill + border colour for each [`FieldHighlight`] style. Not unit tested --
/// this is pure styling with no logic to get wrong, and "does this read well
/// on screen" is a human-at-a-display judgment call, not something a gate
/// can check.
pub fn highlight_colors(style: FieldHighlight) -> (egui::Color32, egui::Stroke) {
    match style {
        FieldHighlight::Editable => (
            egui::Color32::from_rgba_unmultiplied(90, 160, 255, 55),
            egui::Stroke::new(
                1.5,
                egui::Color32::from_rgba_unmultiplied(40, 110, 230, 200),
            ),
        ),
        FieldHighlight::Toggleable => (
            egui::Color32::from_rgba_unmultiplied(90, 200, 140, 55),
            egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(40, 150, 90, 200)),
        ),
        FieldHighlight::ReadOnly => (
            egui::Color32::from_rgba_unmultiplied(150, 150, 150, 35),
            egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_unmultiplied(120, 120, 120, 160),
            ),
        ),
        FieldHighlight::Unsupported => (
            egui::Color32::from_rgba_unmultiplied(230, 170, 60, 45),
            egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_unmultiplied(200, 130, 30, 180),
            ),
        ),
        FieldHighlight::None => (egui::Color32::TRANSPARENT, egui::Stroke::NONE),
    }
}

/// Whether an `Enter` keypress (`key`) held with `modifiers` should commit
/// an in-place field editor rather than (for a multiline field) insert a
/// newline. Per the maintainer's explicit call for `kpdf`: "enter saves the
/// thing. Shift-Enter gets a new line" -- so plain `Enter` commits, and only
/// `Shift+Enter` is diverted to a newline. Ctrl/Alt/Cmd held alongside
/// `Enter` are not treated specially either way -- the instruction only
/// distinguished `Shift`, so this predicate does too.
///
/// This is only meaningful for a *multiline* field -- a single-line
/// `egui::TextEdit` has no newline to insert in the first place, so it can
/// keep using its own built-in "Enter surrenders focus" behaviour
/// unmodified.
///
/// Deliberately not implemented via `egui::InputState::consume_key`: that
/// method matches modifiers via `Modifiers::matches_logically`, which is
/// documented to ignore an *extra* `Shift`/`Alt` on the pressed
/// combination -- exactly the distinction this predicate needs to make.
/// Pure over plain `egui::Key`/`egui::Modifiers` (both plain data
/// structures, no display needed to construct), so this exact predicate --
/// the one keyboard-gesture decision made without being able to press the
/// key on a real display -- is covered by a unit test rather than an
/// assumption. See the `tests` module below.
pub fn should_commit_on_enter(key: egui::Key, modifiers: egui::Modifiers) -> bool {
    key == egui::Key::Enter && !modifiers.shift
}

/// Remove a shiftless `Enter` keypress from this frame's input queue, if
/// there is one, and report whether it found one. Meant to be called before
/// a multiline field's `TextEdit` widget runs, so the widget's own default
/// "Enter inserts a newline" handling never sees the key: without this, a
/// plain Enter would both commit (via the caller's check) and still land in
/// the buffer as a newline in the same frame. See [`should_commit_on_enter`]
/// for the actual predicate this wraps -- that is the part covered by a
/// unit test; this thin `egui::Ui` wrapper is not, same as the rest of this
/// module's egui-touching (as opposed to pure) helpers.
pub fn consume_commit_enter(ui: &mut egui::Ui) -> bool {
    ui.input_mut(|i| {
        let mut found = false;
        i.events.retain(|event| {
            let egui::Event::Key {
                key,
                pressed: true,
                modifiers,
                ..
            } = event
            else {
                return true;
            };
            if should_commit_on_enter(*key, *modifiers) {
                found = true;
                false
            } else {
                true
            }
        });
        found
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- field_highlight_kind ---------------------------------------------

    #[test]
    fn field_highlight_kind_matches_handle_forms_click_dispatch() {
        // Every kind kpdf actually acts on while not read-only.
        assert_eq!(
            field_highlight_kind(FieldKind::Text, false),
            FieldHighlight::Editable
        );
        assert_eq!(
            field_highlight_kind(FieldKind::Checkbox, false),
            FieldHighlight::Toggleable
        );
        assert_eq!(
            field_highlight_kind(FieldKind::Radio, false),
            FieldHighlight::Toggleable
        );
        // Combobox/Listbox: `set_field_value` refuses them, so they get a
        // distinct "can't edit this here" style rather than `Editable`.
        assert_eq!(
            field_highlight_kind(FieldKind::Combobox, false),
            FieldHighlight::Unsupported
        );
        assert_eq!(
            field_highlight_kind(FieldKind::Listbox, false),
            FieldHighlight::Unsupported
        );
        // Nothing kpdf can act on -- no highlight at all.
        assert_eq!(
            field_highlight_kind(FieldKind::Button, false),
            FieldHighlight::None
        );
        assert_eq!(
            field_highlight_kind(FieldKind::Signature, false),
            FieldHighlight::None
        );
        assert_eq!(
            field_highlight_kind(FieldKind::Unknown, false),
            FieldHighlight::None
        );
    }

    #[test]
    fn field_highlight_kind_read_only_overrides_every_actionable_kind() {
        for kind in [
            FieldKind::Text,
            FieldKind::Checkbox,
            FieldKind::Radio,
            FieldKind::Combobox,
            FieldKind::Listbox,
        ] {
            assert_eq!(
                field_highlight_kind(kind, true),
                FieldHighlight::ReadOnly,
                "{kind:?} read-only should highlight as ReadOnly"
            );
        }
    }

    #[test]
    fn field_highlight_kind_read_only_button_like_kinds_stay_hidden() {
        // A read-only pushbutton/signature/unknown still has nothing kpdf
        // can do with it -- read-only doesn't turn "no highlight" into
        // "muted highlight" for these.
        for kind in [FieldKind::Button, FieldKind::Signature, FieldKind::Unknown] {
            assert_eq!(field_highlight_kind(kind, true), FieldHighlight::None);
        }
    }

    // -- should_commit_on_enter --------------------------------------------

    #[test]
    fn should_commit_on_enter_plain_enter_commits() {
        assert!(should_commit_on_enter(
            egui::Key::Enter,
            egui::Modifiers::NONE
        ));
    }

    #[test]
    fn should_commit_on_enter_shift_enter_does_not_commit() {
        assert!(!should_commit_on_enter(
            egui::Key::Enter,
            egui::Modifiers::SHIFT
        ));
    }

    #[test]
    fn should_commit_on_enter_shift_plus_other_modifiers_still_does_not_commit() {
        assert!(!should_commit_on_enter(
            egui::Key::Enter,
            egui::Modifiers::SHIFT | egui::Modifiers::CTRL
        ));
    }

    #[test]
    fn should_commit_on_enter_other_modifiers_without_shift_still_commit() {
        // The maintainer's instruction only distinguished Shift; Ctrl/Alt/
        // Cmd alongside a shiftless Enter are not diverted to a newline.
        assert!(should_commit_on_enter(
            egui::Key::Enter,
            egui::Modifiers::CTRL
        ));
    }

    #[test]
    fn should_commit_on_enter_a_different_key_never_commits() {
        assert!(!should_commit_on_enter(egui::Key::A, egui::Modifiers::NONE));
    }
}
