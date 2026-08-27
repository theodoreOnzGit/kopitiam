//! Render-dpi zoom model: the dpi range/step a viewer rasterizes pages at,
//! the "percentage" a human reads on a status bar, and the pure accumulator
//! that turns egui's noisy per-frame `zoom_delta()` signal into whole zoom
//! steps. No egui types appear anywhere in this module -- it is plain
//! `f32` arithmetic, usable by any caller that wants the same zoom model
//! without depending on egui at all.

// Same render-dpi range/step as `kopitiam view`
// (apps/cli/src/tui/viewer.rs's DPI_DEFAULT/MIN/MAX/STEP), kept in parity on
// purpose -- there is no reason the two viewers should zoom differently.
pub const DPI_DEFAULT: f32 = 150.0;
pub const DPI_MIN: f32 = 50.0;
pub const DPI_MAX: f32 = 600.0;
pub const DPI_STEP: f32 = 25.0;

/// One [`DPI_STEP`] zoom notch's worth of accumulated Ctrl+scroll input, in
/// the same units as `egui::InputState::zoom_delta()` minus one (so `0.0`
/// means "no zoom this frame", positive means zooming in, negative zooming
/// out). Mouse wheels and trackpads report a Ctrl+scroll gesture as many
/// small per-frame nudges to `zoom_delta()` rather than one clean value, so
/// instead of moving `dpi` by a fraction of a step on every such frame --
/// which would re-rasterize the page on every one of those frames, since a
/// texture cache is typically keyed on `(page, dpi)` -- the raw signal
/// accumulates in a caller-owned accumulator (`KpdfApp::scroll_zoom_accum`
/// in the `kpdf` binary) and is only converted into a whole [`DPI_STEP`]
/// move once enough of it has piled up. See [`zoom_steps_from_zoom_delta`]
/// and its tests for the exact coalescing behaviour. The threshold itself is
/// a reasonable default, not a value derived from measuring a real wheel --
/// whether one flick of the wheel feels like the right number of steps is a
/// human-at-a-display judgment call, not something a gate can check.
pub const ZOOM_DELTA_PER_STEP: f32 = 0.05;

/// Render `dpi` as a percentage of [`DPI_DEFAULT`] -- "150%" is what a
/// reader acts on; the raw "108 dpi" the rasterizer actually uses is not.
/// Rounds to the nearest whole percent.
pub fn zoom_percent(dpi: f32) -> i32 {
    ((dpi / DPI_DEFAULT) * 100.0).round() as i32
}

/// Fold one frame's `zoom_delta` (egui's Ctrl+scroll / pinch-zoom signal,
/// where `1.0` means "no change this frame") into whole [`DPI_STEP`] zoom
/// steps, carrying any leftover in `accum` for the next frame -- see
/// [`ZOOM_DELTA_PER_STEP`] for why this accumulates rather than acting on
/// every frame's raw value directly. Positive steps mean zoom in (matching
/// `KpdfApp::zoom_in`'s direction), negative mean zoom out, `0` means "not
/// enough has accumulated yet".
///
/// Pure and independent of egui's `Context`/`InputState`, so it is unit
/// tested directly without a display -- see the `tests` module below.
pub fn zoom_steps_from_zoom_delta(zoom_delta: f32, accum: &mut f32) -> i32 {
    *accum += zoom_delta - 1.0;
    let steps = (*accum / ZOOM_DELTA_PER_STEP).trunc();
    *accum -= steps * ZOOM_DELTA_PER_STEP;
    steps as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- zoom_percent -------------------------------------------------

    #[test]
    fn zoom_percent_at_default_is_100() {
        assert_eq!(zoom_percent(DPI_DEFAULT), 100);
    }

    #[test]
    fn zoom_percent_scales_with_dpi() {
        assert_eq!(zoom_percent(DPI_DEFAULT * 2.0), 200);
        assert_eq!(zoom_percent(DPI_MIN), 33); // 50/150 = 33.33...%
        assert_eq!(zoom_percent(75.0), 50); // 75/150
    }

    #[test]
    fn zoom_percent_rounds_to_nearest_whole_percent() {
        // 151/150 = 100.666...% -> rounds to 101, not truncates to 100.
        assert_eq!(zoom_percent(151.0), 101);
    }

    // -- zoom_steps_from_zoom_delta -----------------------------------

    #[test]
    fn zoom_delta_of_one_is_a_no_op() {
        let mut accum = 0.0;
        assert_eq!(zoom_steps_from_zoom_delta(1.0, &mut accum), 0);
        assert_eq!(accum, 0.0);
    }

    #[test]
    fn small_zoom_deltas_accumulate_before_stepping() {
        // Each nudge below is well under ZOOM_DELTA_PER_STEP (0.05) on its
        // own -- this is the "many small per-frame events" case the
        // accumulator exists for. No single call should fire a step until
        // enough of them have piled up. The final nudge is deliberately
        // larger than a clean boundary crossing (0.04 + 0.02 = 0.06, not
        // exactly 0.05) so the assertion doesn't hinge on f32 rounding
        // landing on the exact millimetre of the threshold.
        let mut accum = 0.0;
        for _ in 0..4 {
            assert_eq!(zoom_steps_from_zoom_delta(1.01, &mut accum), 0);
        }
        assert_eq!(zoom_steps_from_zoom_delta(1.02, &mut accum), 1);
    }

    #[test]
    fn large_zoom_delta_can_fire_multiple_steps_in_one_call() {
        // A single big jump (e.g. a fast trackpad pinch reported in one
        // frame) should still convert to the right whole number of steps,
        // not just clamp to one.
        let mut accum = 0.0;
        assert_eq!(zoom_steps_from_zoom_delta(1.23, &mut accum), 4); // 0.23 / 0.05 = 4.6 -> 4
        assert!(accum > 0.0 && accum < ZOOM_DELTA_PER_STEP);
    }

    #[test]
    fn zoom_out_direction_is_negative() {
        // Same reasoning as the zoom-in case above: the last nudge is
        // chosen to land clearly past the threshold (-0.06, not exactly
        // -0.05) so the test doesn't depend on f32 rounding at the boundary.
        let mut accum = 0.0;
        for _ in 0..4 {
            assert_eq!(zoom_steps_from_zoom_delta(0.99, &mut accum), 0);
        }
        assert_eq!(zoom_steps_from_zoom_delta(0.98, &mut accum), -1);
    }

    #[test]
    fn leftover_accum_is_never_dropped() {
        // Repeatedly feeding a delta that is *not* an exact multiple of the
        // step threshold must not lose the remainder over many calls --
        // total steps taken should track total input applied. 100 * 0.0011
        // = 0.11 units, deliberately not a clean multiple of
        // ZOOM_DELTA_PER_STEP (0.05) so the expected step count (2, with
        // 0.01 left over) isn't sitting exactly on an f32 rounding boundary
        // the way an exact multiple would be.
        let mut accum = 0.0;
        let mut total_steps = 0;
        for _ in 0..100 {
            total_steps += zoom_steps_from_zoom_delta(1.0011, &mut accum);
        }
        assert_eq!(total_steps, 2);
        assert!(accum > 0.0 && accum < ZOOM_DELTA_PER_STEP);
    }

    #[test]
    fn accum_never_grows_unbounded_once_a_step_is_taken() {
        let mut accum = 0.0;
        zoom_steps_from_zoom_delta(1.23, &mut accum);
        assert!(accum.abs() < ZOOM_DELTA_PER_STEP);
    }
}
