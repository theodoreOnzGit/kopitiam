//! Graphics state that a *reader* uses to tell one curve from another.
//!
//! When a human looks at a figure with three curves on it, they separate the
//! series by exactly three visual cues: colour, line width, and dash pattern.
//! Nothing else. The legend then maps those cues to meanings ("solid black =
//! experiment, dashed red = simulation").
//!
//! That is not a heuristic we invented -- it is how the figure was *authored*,
//! and the cues survive into the PDF verbatim as graphics-state operators
//! (`RG`/`rg`/`G`/`g`/`K`/`k`, `w`, `d`). So series separation is not an
//! inference problem at all: it is a lookup. We reproduce the same graphics
//! state the renderer would, and group paths by it.
//!
//! This is the main reason this crate walks the content stream itself rather
//! than using `pdf-extract`'s `stroke`/`fill` callbacks, which report colour
//! but neither line width nor dash pattern -- see [`crate::content`].

use serde::{Deserialize, Serialize};

/// An RGB colour with components in `[0, 1]`.
///
/// Every PDF colour space we understand is converted to RGB on the way in, so
/// that two paths painted the same colour compare equal even if one was
/// specified as DeviceGray and the other as DeviceRGB. Colour spaces we do not
/// understand (ICCBased, Separation, Indexed, ...) are left at the PDF default
/// of black, which is recorded honestly rather than guessed -- see
/// [`crate::content`]'s handling of `sc`/`scn`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Rgb {
    pub const BLACK: Rgb = Rgb {
        r: 0.0,
        g: 0.0,
        b: 0.0,
    };

    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self {
            r: r.clamp(0.0, 1.0),
            g: g.clamp(0.0, 1.0),
            b: b.clamp(0.0, 1.0),
        }
    }

    pub fn gray(v: f32) -> Self {
        Self::new(v, v, v)
    }

    /// DeviceCMYK -> RGB, using the standard naive conversion from ISO 32000-1
    /// §8.6.4.4 (`R = 1 - min(1, C + K)`, and likewise for G and B).
    ///
    /// This is not colour-managed and does not pretend to be. It is exact
    /// enough for its only purpose here: deciding whether two strokes are *the
    /// same* colour. Two paths authored with identical CMYK land on identical
    /// RGB, which is all series grouping needs.
    pub fn cmyk(c: f32, m: f32, y: f32, k: f32) -> Self {
        Self::new(
            1.0 - (c + k).min(1.0),
            1.0 - (m + k).min(1.0),
            1.0 - (y + k).min(1.0),
        )
    }

    /// Quantise to 8 bits per channel for use as a grouping key.
    ///
    /// Direct float equality would be fragile: a producer may write `0.8` for
    /// one path and `0.800003` for the next after a round-trip through its own
    /// float formatting. 8 bits is the precision the colour will actually be
    /// rendered at anyway, so two colours that quantise alike are two colours
    /// no reader could distinguish.
    fn key(&self) -> (u8, u8, u8) {
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        (q(self.r), q(self.g), q(self.b))
    }

    /// Perceived lightness (Rec. 601 luma), used only to recognise the
    /// near-black that axis furniture is conventionally drawn in.
    pub fn luma(&self) -> f32 {
        0.299 * self.r + 0.587 * self.g + 0.114 * self.b
    }
}

/// How a path was painted. A path can be both filled and stroked (`B`), which
/// is how most solid scatter markers are drawn (filled body, stroked edge).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Paint {
    Stroke,
    Fill,
    FillStroke,
}

impl Paint {
    /// Whether this paint operation puts ink along the path outline.
    pub fn strokes(self) -> bool {
        matches!(self, Paint::Stroke | Paint::FillStroke)
    }

    /// Whether this paint operation puts ink inside the path.
    pub fn fills(self) -> bool {
        matches!(self, Paint::Fill | Paint::FillStroke)
    }
}

/// The visual identity of a series: the cues a reader would use to pick it out
/// of the figure, and the cues we group paths by.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeriesStyle {
    pub paint: Paint,
    /// The colour the ink is actually laid down in: the stroke colour for a
    /// stroked path, the fill colour for a filled one. For `FillStroke` the
    /// fill colour is used, since that is the one that dominates visually on a
    /// small marker.
    pub color: Rgb,
    /// Stroke width in page units (points), already scaled by the CTM. Zero for
    /// a purely filled path.
    pub line_width: f32,
    /// The dash pattern's `on`/`off` lengths in page units. Empty means solid.
    pub dash: Vec<f32>,
}

impl SeriesStyle {
    /// A hashable, quantised identity for grouping.
    ///
    /// Line width is quantised to 0.05 pt and dash lengths to 0.1 pt for the
    /// same reason colour is quantised to 8 bits: producers re-serialise floats,
    /// and differences far below the resolution of the printed page must not
    /// split one series into two.
    pub fn key(&self) -> StyleKey {
        StyleKey {
            paint: self.paint,
            color: self.color.key(),
            line_width: (self.line_width / 0.05).round() as i32,
            dash: self
                .dash
                .iter()
                .map(|d| (d / 0.1).round() as i32)
                .collect(),
        }
    }

    /// Whether this style looks like axis furniture rather than data: thin and
    /// near-black.
    ///
    /// This is only ever used as a *tie-breaker* -- structural elements (spines,
    /// ticks, grid lines) are identified by their geometry, not their colour,
    /// because plenty of real figures draw data in black and grid lines in grey.
    /// Relying on colour alone here would be exactly the kind of silent
    /// wrongness this crate exists to avoid.
    pub fn is_furniture_coloured(&self) -> bool {
        self.color.luma() < 0.35
    }

    /// A short human-readable description, e.g. `"dashed red, 1.5pt"`. Used in
    /// warnings and CSV provenance headers so that a reader holding the printed
    /// figure can tell which curve a row of numbers came from.
    pub fn describe(&self) -> String {
        let dash = if self.dash.is_empty() {
            "solid"
        } else {
            "dashed"
        };
        let (r, g, b) = self.color.key();
        format!(
            "{dash} #{r:02x}{g:02x}{b:02x}, {:.2}pt, {}",
            self.line_width,
            match self.paint {
                Paint::Stroke => "stroked",
                Paint::Fill => "filled",
                Paint::FillStroke => "filled+stroked",
            }
        )
    }
}

/// Quantised, hashable form of a [`SeriesStyle`].
///
/// `Ord` as well as `Hash`, so that series can be grouped in a `BTreeMap` and
/// come out in a stable order. Determinism is a stated requirement (CLAUDE.md,
/// Engineering Principles) and it bites specifically here: series indices appear
/// in the CSV export, so a `HashMap`'s arbitrary iteration order would make two
/// digitisations of the same PDF diff against each other.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StyleKey {
    pub paint: Paint,
    pub color: (u8, u8, u8),
    pub line_width: i32,
    pub dash: Vec<i32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmyk_black_and_white() {
        assert_eq!(Rgb::cmyk(0.0, 0.0, 0.0, 1.0), Rgb::BLACK);
        assert_eq!(Rgb::cmyk(0.0, 0.0, 0.0, 0.0), Rgb::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn nearly_equal_colours_share_a_key() {
        let a = SeriesStyle {
            paint: Paint::Stroke,
            color: Rgb::new(0.8, 0.0, 0.0),
            line_width: 1.5,
            dash: vec![],
        };
        let b = SeriesStyle {
            color: Rgb::new(0.800_003, 0.000_001, 0.0),
            line_width: 1.501,
            ..a.clone()
        };
        assert_eq!(a.key(), b.key());
    }

    #[test]
    fn different_dashes_do_not_share_a_key() {
        let solid = SeriesStyle {
            paint: Paint::Stroke,
            color: Rgb::BLACK,
            line_width: 1.0,
            dash: vec![],
        };
        let dashed = SeriesStyle {
            dash: vec![3.0, 2.0],
            ..solid.clone()
        };
        assert_ne!(solid.key(), dashed.key());
    }

    #[test]
    fn different_widths_do_not_share_a_key() {
        let thin = SeriesStyle {
            paint: Paint::Stroke,
            color: Rgb::BLACK,
            line_width: 0.5,
            dash: vec![],
        };
        let thick = SeriesStyle {
            line_width: 2.0,
            ..thin.clone()
        };
        assert_ne!(thin.key(), thick.key());
    }
}
