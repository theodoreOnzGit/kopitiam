/// Axis-aligned bounding box in PDF user-space units (points; origin at the
/// page's bottom-left, y increasing upward, per the PDF coordinate convention).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    pub fn top(&self) -> f32 {
        self.y + self.height
    }
}
