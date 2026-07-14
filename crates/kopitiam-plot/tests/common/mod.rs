//! Test support: a minimal, pure-Rust PDF writer, and a plot drawer built on it.
//!
//! # Why this exists
//!
//! The single most valuable test for a digitiser is a **round trip**: take data
//! you chose yourself, draw it as a plot, digitise the plot, and assert the
//! numbers come back. That test needs a PDF containing a plot whose underlying
//! data is *known exactly* -- which means we have to author the PDF, because
//! any PDF found in the wild has, by definition, lost the numbers we would be
//! checking against. (That is the entire premise of the crate: the numbers are
//! gone, and only the picture remains.)
//!
//! A valid single-page PDF is about a hundred lines of writer, so we own it
//! outright rather than taking a dependency. It also keeps the test corpus
//! deterministic and offline, both of which CLAUDE.md requires.
//!
//! The PDFs produced here use the Standard-14 font `/Helvetica`, which needs no
//! font embedding: `pdf-extract` carries built-in metrics for the base-14 set,
//! so tick labels come back through `kopitiam-pdf` as real text.

#![allow(dead_code)] // Each integration test file uses a different subset.

/// Builds a valid one-page PDF around a content stream.
pub struct PdfBuilder {
    width: f32,
    height: f32,
    content: String,
}

impl PdfBuilder {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            width,
            height,
            content: String::new(),
        }
    }

    /// Append a raw line of content-stream operators.
    pub fn op(&mut self, line: &str) -> &mut Self {
        self.content.push_str(line);
        self.content.push('\n');
        self
    }

    /// Serialise to PDF bytes, with a correct cross-reference table.
    pub fn build(&self) -> Vec<u8> {
        const OBJECTS: usize = 5;
        let mut out: Vec<u8> = Vec::new();
        let mut offsets = [0usize; OBJECTS + 1];

        // The binary comment on line 2 marks the file as containing binary
        // data, per ISO 32000-1 §7.5.2. Harmless here, and it is what every
        // real producer emits.
        out.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");

        let push = |out: &mut Vec<u8>, offsets: &mut [usize], n: usize, body: &str| {
            offsets[n] = out.len();
            out.extend_from_slice(format!("{n} 0 obj\n{body}\nendobj\n").as_bytes());
        };

        push(
            &mut out,
            &mut offsets,
            1,
            "<< /Type /Catalog /Pages 2 0 R >>",
        );
        push(
            &mut out,
            &mut offsets,
            2,
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        );
        push(
            &mut out,
            &mut offsets,
            3,
            &format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] \
                 /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
                self.width, self.height
            ),
        );
        push(
            &mut out,
            &mut offsets,
            4,
            &format!(
                "<< /Length {} >>\nstream\n{}\nendstream",
                self.content.len(),
                self.content
            ),
        );
        push(
            &mut out,
            &mut offsets,
            5,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        );

        let xref = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n", OBJECTS + 1).as_bytes());
        out.extend_from_slice(b"0000000000 65535 f \n");
        for offset in offsets.iter().skip(1) {
            out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                OBJECTS + 1
            )
            .as_bytes(),
        );
        out
    }
}

/// Linear or base-10 logarithmic axis, matching [`kopitiam_plot::AxisScale`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scale {
    Linear,
    Log10,
}

impl Scale {
    /// Map a data value into the axis's *fraction* of the plot region.
    fn fraction(self, v: f64, lo: f64, hi: f64) -> f64 {
        match self {
            Scale::Linear => (v - lo) / (hi - lo),
            Scale::Log10 => (v.log10() - lo.log10()) / (hi.log10() - lo.log10()),
        }
    }
}

/// How a tick label should be rendered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LabelStyle {
    /// Plain decimal, e.g. `0.25`, formatted with the given precision.
    Decimal(usize),
    /// The scientific `10` with a raised exponent, e.g. `10` + superscript `-3`
    /// -- which is what matplotlib emits by default on a log axis, and which is
    /// therefore the form a real digitiser has to cope with.
    Power10,
}

/// A plot to draw: the region on the page, the data ranges, and the ticks.
///
/// Note that the *drawer* owns the mapping from data to page, and the
/// *digitiser* has to rediscover it. Nothing is shared between them but the
/// PDF, which is the point.
pub struct Plot {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub x_range: (f64, f64),
    pub y_range: (f64, f64),
    pub x_scale: Scale,
    pub y_scale: Scale,
    pub x_ticks: Vec<f64>,
    pub y_ticks: Vec<f64>,
    pub label_style: LabelStyle,
}

impl Plot {
    /// A conventional linear plot on US Letter, ranges 0..1 with ticks every
    /// 0.25 -- the baseline every test starts from.
    pub fn linear_unit_square() -> Self {
        Self {
            x0: 100.0,
            y0: 150.0,
            x1: 500.0,
            y1: 600.0,
            x_range: (0.0, 1.0),
            y_range: (0.0, 1.0),
            x_scale: Scale::Linear,
            y_scale: Scale::Linear,
            x_ticks: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            y_ticks: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            label_style: LabelStyle::Decimal(2),
        }
    }

    /// Map data coordinates to page coordinates -- the *ground truth* affine
    /// (or log-affine) map that the digitiser must recover from the drawing
    /// alone.
    pub fn to_page(&self, x: f64, y: f64) -> (f32, f32) {
        let fx = self
            .x_scale
            .fraction(x, self.x_range.0, self.x_range.1);
        let fy = self
            .y_scale
            .fraction(y, self.y_range.0, self.y_range.1);
        (
            self.x0 + (fx as f32) * (self.x1 - self.x0),
            self.y0 + (fy as f32) * (self.y1 - self.y0),
        )
    }

    /// Draw the frame, the tick marks and the tick labels.
    pub fn draw_axes(&self, pdf: &mut PdfBuilder) {
        // Spines: a bottom x-axis and a left y-axis, 1pt black.
        pdf.op("q 0 0 0 RG 1 w");
        pdf.op(&format!(
            "{} {} m {} {} l S",
            self.x0, self.y0, self.x1, self.y0
        ));
        pdf.op(&format!(
            "{} {} m {} {} l S",
            self.x0, self.y0, self.x0, self.y1
        ));

        // Tick marks: 4pt, pointing outward (down / left), touching the spine.
        for &v in &self.x_ticks {
            let (px, _) = self.to_page(v, self.y_range.0);
            pdf.op(&format!(
                "{px} {} m {px} {} l S",
                self.y0,
                self.y0 - 4.0
            ));
        }
        for &v in &self.y_ticks {
            let (_, py) = self.to_page(self.x_range.0, v);
            pdf.op(&format!(
                "{} {py} m {} {py} l S",
                self.x0,
                self.x0 - 4.0
            ));
        }
        pdf.op("Q");

        for &v in &self.x_ticks {
            let (px, _) = self.to_page(v, self.y_range.0);
            self.draw_label(pdf, v, px, self.y0 - 16.0, Align::Center);
        }
        for &v in &self.y_ticks {
            let (_, py) = self.to_page(self.x_range.0, v);
            self.draw_label(pdf, v, self.x0 - 8.0, py - 3.0, Align::Right);
        }
    }

    fn draw_label(&self, pdf: &mut PdfBuilder, value: f64, x: f32, y: f32, align: Align) {
        const SIZE: f32 = 9.0;
        match self.label_style {
            LabelStyle::Decimal(prec) => {
                let text = format!("{value:.prec$}");
                let w = helvetica_width(&text, SIZE);
                let tx = align.origin(x, w);
                pdf.op(&format!(
                    "BT /F1 {SIZE} Tf {tx} {y} Td ({text}) Tj ET"
                ));
            }
            LabelStyle::Power10 => {
                // `10` at full size, then the exponent at 60% size, raised.
                let exp = value.log10().round() as i32;
                let base = "10";
                let sup = format!("{exp}");
                let sup_size = SIZE * 0.6;
                let w = helvetica_width(base, SIZE) + helvetica_width(&sup, sup_size);
                let tx = align.origin(x, w);
                pdf.op(&format!("BT /F1 {SIZE} Tf {tx} {y} Td ({base}) Tj ET"));
                pdf.op(&format!(
                    "BT /F1 {sup_size} Tf {} {} Td ({sup}) Tj ET",
                    tx + helvetica_width(base, SIZE),
                    y + SIZE * 0.45
                ));
            }
        }
    }

    /// Draw a data series as a stroked polyline, in the standard `m` + `l`*
    /// form every plotting tool emits.
    pub fn draw_line_series(
        &self,
        pdf: &mut PdfBuilder,
        points: &[(f64, f64)],
        rgb: (f32, f32, f32),
        width: f32,
        dash: &[f32],
    ) {
        let dash_op = if dash.is_empty() {
            "[] 0 d".to_string()
        } else {
            let items: Vec<String> = dash.iter().map(|d| d.to_string()).collect();
            format!("[{}] 0 d", items.join(" "))
        };
        pdf.op(&format!(
            "q {} {} {} RG {width} w {dash_op}",
            rgb.0, rgb.1, rgb.2
        ));
        for (i, &(x, y)) in points.iter().enumerate() {
            let (px, py) = self.to_page(x, y);
            pdf.op(&format!("{px} {py} {}", if i == 0 { "m" } else { "l" }));
        }
        pdf.op("S Q");
    }

    /// Draw a scatter series: one small filled square per data point, each its
    /// own closed subpath, painted with `B` (fill+stroke) -- the operator
    /// `pdf-extract`'s own callbacks drop entirely.
    pub fn draw_scatter_series(
        &self,
        pdf: &mut PdfBuilder,
        points: &[(f64, f64)],
        rgb: (f32, f32, f32),
    ) {
        const R: f32 = 2.0;
        pdf.op(&format!(
            "q {} {} {} rg {} {} {} RG 0.5 w",
            rgb.0, rgb.1, rgb.2, rgb.0, rgb.1, rgb.2
        ));
        for &(x, y) in points {
            let (px, py) = self.to_page(x, y);
            pdf.op(&format!(
                "{} {} {} {} re B",
                px - R,
                py - R,
                2.0 * R,
                2.0 * R
            ));
        }
        pdf.op("Q");
    }

    /// Draw a legend entry: a short line sample in the series' style, with its
    /// text to the right. Placed inside the plot region, as real legends are.
    pub fn draw_legend_entry(
        &self,
        pdf: &mut PdfBuilder,
        row: usize,
        text: &str,
        rgb: (f32, f32, f32),
        width: f32,
        dash: &[f32],
    ) {
        let x = self.x1 - 120.0;
        let y = self.y1 - 20.0 - 14.0 * row as f32;
        let dash_op = if dash.is_empty() {
            "[] 0 d".to_string()
        } else {
            let items: Vec<String> = dash.iter().map(|d| d.to_string()).collect();
            format!("[{}] 0 d", items.join(" "))
        };
        pdf.op(&format!(
            "q {} {} {} RG {width} w {dash_op} {x} {y} m {} {y} l S Q",
            rgb.0,
            rgb.1,
            rgb.2,
            x + 24.0
        ));
        pdf.op(&format!(
            "BT /F1 9 Tf {} {} Td ({text}) Tj ET",
            x + 30.0,
            y - 3.0
        ));
    }
}

#[derive(Clone, Copy)]
pub enum Align {
    Center,
    Right,
}

impl Align {
    fn origin(self, anchor: f32, width: f32) -> f32 {
        match self {
            Align::Center => anchor - width / 2.0,
            Align::Right => anchor - width,
        }
    }
}

/// Approximate the advance width of a Helvetica string, in points.
///
/// Only used to place tick labels plausibly (centred under their tick), so that
/// the digitiser's *fallback* path -- calibrating from label centres when tick
/// marks are absent -- is exercised against realistic geometry. Exactness is
/// not required: the primary calibration path uses tick-mark positions, not
/// label positions.
///
/// Widths are the real Helvetica AFM values, in 1/1000 em.
fn helvetica_width(s: &str, size: f32) -> f32 {
    let units: f32 = s
        .chars()
        .map(|c| match c {
            '.' => 278.0,
            '-' => 333.0,
            '0'..='9' => 556.0,
            ' ' => 278.0,
            _ => 556.0,
        })
        .sum();
    units / 1000.0 * size
}
