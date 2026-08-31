//! Hit-testing over annotations and form-field widgets in default user
//! space -- "which annotation/field is under this PDF-space point", the
//! question an eraser tool or a forms-mode click handler needs answered
//! before it can act.

use super::geometry::{PageLayout, min_hit_rect, rect_contains};
use crate::mupdf::annot_edit::AnnotRef;
use crate::mupdf::form::FormField;

/// The annotation under `(page_x, page_y)` (default user space), or `None`
/// if `refs` has nothing there -- what an eraser tool needs in order to
/// know which annotation object number to delete.
///
/// Iterates back-to-front (`.rev()`) so an annotation stacked visually on
/// top of another (later in the page's `/Annots` array, per PDF's paint
/// order) wins the hit, matching what is actually seen on screen.
///
/// Pure over [`AnnotRef`]'s public fields, with no document access -- so it
/// is unit-tested directly against hand-built `AnnotRef`s, without needing
/// [`crate::mupdf::annot_edit::page_annot_refs`] (still `todo!()` as of this
/// writing).
pub fn hit_test_annot(page_x: f32, page_y: f32, refs: &[AnnotRef]) -> Option<i32> {
    refs.iter()
        .rev()
        .find(|a| rect_contains(a.rect, page_x, page_y))
        .map(|a| a.num)
}

/// Same idea as [`hit_test_annot`], for form-field widgets. Returns an
/// **index into `fields`** rather than an object number, because
/// [`crate::mupdf::form::set_field_value`]/`toggle_checkbox` take a whole
/// `&FormField`, not a bare handle.
pub fn hit_test_field(page_x: f32, page_y: f32, fields: &[FormField]) -> Option<usize> {
    fields
        .iter()
        .enumerate()
        .rev()
        .find(|(_, f)| rect_contains(f.rect, page_x, page_y))
        .map(|(i, _)| i)
}

/// Same idea as [`hit_test_field`], but first widens each field's rect via
/// [`crate::gui_frontend::geometry::min_hit_rect`] so a hairline-thin or
/// near-zero-size widget is still a comfortable click target at low zoom.
/// Kept as a fallback rather than folded into [`hit_test_field`] itself --
/// a caller should always try the exact rect first, so a click that already
/// lands inside a field's *real* rect is never second-guessed by another,
/// nearby field's widened area.
pub fn hit_test_field_expanded(
    page_x: f32,
    page_y: f32,
    fields: &[FormField],
    layout: PageLayout,
    min_screen_size: f32,
) -> Option<usize> {
    fields
        .iter()
        .enumerate()
        .rev()
        .find(|(_, f)| {
            rect_contains(
                min_hit_rect(f.rect, layout, min_screen_size),
                page_x,
                page_y,
            )
        })
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mupdf::Rect;
    use crate::mupdf::form::FieldKind;

    fn annot(num: i32, rect: Rect) -> AnnotRef {
        AnnotRef {
            num,
            subtype: "Ink".to_string(),
            rect,
        }
    }

    #[test]
    fn hit_test_annot_finds_containing_rect() {
        let refs = vec![
            annot(1, Rect::new(0.0, 0.0, 100.0, 100.0)),
            annot(2, Rect::new(200.0, 200.0, 300.0, 300.0)),
        ];
        assert_eq!(hit_test_annot(50.0, 50.0, &refs), Some(1));
        assert_eq!(hit_test_annot(250.0, 250.0, &refs), Some(2));
    }

    #[test]
    fn hit_test_annot_misses_return_none() {
        let refs = vec![annot(1, Rect::new(0.0, 0.0, 100.0, 100.0))];
        assert_eq!(hit_test_annot(150.0, 150.0, &refs), None);
        assert_eq!(hit_test_annot(150.0, 150.0, &[]), None);
    }

    #[test]
    fn hit_test_annot_boundary_is_inclusive() {
        let refs = vec![annot(1, Rect::new(0.0, 0.0, 100.0, 100.0))];
        assert_eq!(hit_test_annot(0.0, 0.0, &refs), Some(1));
        assert_eq!(hit_test_annot(100.0, 100.0, &refs), Some(1));
    }

    #[test]
    fn hit_test_annot_prefers_the_topmost_overlapping_annotation() {
        // Later entries in `/Annots` paint on top -- a hit inside the
        // overlap must resolve to the *later* (visually topmost) one, not
        // the first match encountered in array order.
        let refs = vec![
            annot(1, Rect::new(0.0, 0.0, 100.0, 100.0)),
            annot(2, Rect::new(50.0, 50.0, 150.0, 150.0)),
        ];
        assert_eq!(hit_test_annot(75.0, 75.0, &refs), Some(2));
    }

    fn field(obj_num: i32, kind: FieldKind, rect: Rect) -> FormField {
        FormField {
            obj_num,
            page_index: 0,
            kind,
            name: format!("field{obj_num}"),
            value: String::new(),
            rect,
            read_only: false,
            on_state: None,
            multiline: false,
            hidden: false,
        }
    }

    #[test]
    fn hit_test_field_finds_containing_rect_and_returns_index() {
        let fields = vec![
            field(10, FieldKind::Checkbox, Rect::new(0.0, 0.0, 20.0, 20.0)),
            field(11, FieldKind::Text, Rect::new(0.0, 100.0, 200.0, 120.0)),
        ];
        assert_eq!(hit_test_field(10.0, 10.0, &fields), Some(0));
        assert_eq!(hit_test_field(100.0, 110.0, &fields), Some(1));
    }

    #[test]
    fn hit_test_field_miss_returns_none() {
        let fields = vec![field(
            10,
            FieldKind::Checkbox,
            Rect::new(0.0, 0.0, 20.0, 20.0),
        )];
        assert_eq!(hit_test_field(500.0, 500.0, &fields), None);
    }

    // -- hit_test_field_expanded ------------------------------------------------

    /// Same test-only stand-in as `geometry::tests::TEST_MIN_HIT_AREA` --
    /// see that constant's comment; the value is `kpdf`'s own UX default,
    /// duplicated here purely as a representative test fixture.
    const TEST_MIN_HIT_AREA: f32 = 16.0;

    fn sample_layout() -> PageLayout {
        PageLayout {
            image_x: 20.0,
            image_y: 40.0,
            image_w: 300.0,
            image_h: 400.0,
            page_w_pts: 612.0,
            page_h_pts: 792.0,
            page_x0: 0.0,
            page_y0: 0.0,
        }
    }

    #[test]
    fn hit_test_field_expanded_catches_a_hairline_field_the_exact_test_misses() {
        let layout = sample_layout();
        // A field rendered as a hairline: 40pt wide but 0.1pt tall. The
        // exact test (`hit_test_field`) must miss a click a few points off
        // the hairline; the expanded test must still catch it.
        let fields = vec![field(
            1,
            FieldKind::Text,
            Rect::new(0.0, 100.0, 40.0, 100.1),
        )];
        assert_eq!(hit_test_field(20.0, 105.0, &fields), None);
        assert_eq!(
            hit_test_field_expanded(20.0, 105.0, &fields, layout, TEST_MIN_HIT_AREA),
            Some(0)
        );
    }

    #[test]
    fn hit_test_field_expanded_still_misses_far_away_clicks() {
        let layout = sample_layout();
        let fields = vec![field(
            1,
            FieldKind::Text,
            Rect::new(0.0, 100.0, 40.0, 100.1),
        )];
        assert_eq!(
            hit_test_field_expanded(500.0, 500.0, &fields, layout, TEST_MIN_HIT_AREA),
            None
        );
    }
}
