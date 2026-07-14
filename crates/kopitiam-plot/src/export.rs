//! Getting the numbers out.
//!
//! A digitised plot that nobody can extract the data from has accomplished
//! nothing. CSV is the format every plotting tool, spreadsheet and solver
//! validation harness reads, so it is the one that matters.
//!
//! # Long format, and why
//!
//! Series on one figure rarely share x values -- two experiments sampled at
//! different points, a curve and the scatter it was fitted to -- so a wide
//! `x, y1, y2` layout would force us to invent alignment between columns that
//! do not align. Long format (one row per point, with the series named) makes
//! no such claim, and is what `pandas.read_csv` + `groupby` and gnuplot's
//! `index` both expect anyway.
//!
//! # The provenance header
//!
//! [`to_csv_with_provenance`] prefixes `#` comment lines recording the axis
//! calibration, the tick observations it was fitted from, and every warning.
//! `#` is skipped by `numpy.loadtxt`, `pandas.read_csv(comment='#')` and
//! gnuplot, so the file stays machine-readable while carrying its own audit
//! trail -- which is precisely CLAUDE.md's Scientific Standards requirement
//! that scientific software "always remain explainable". A digitised validation
//! dataset that has been separated from the caveats of its digitisation is
//! exactly the artefact that gets someone into trouble.

use crate::axes::AxisScale;
use crate::digitise::{DigitisedPlot, describe_calibration};

/// Data only: RFC 4180 CSV, one row per recovered point.
///
/// Columns: `series, label, kind, x, y, page_x, page_y`.
///
/// `page_x`/`page_y` travel with every row on purpose. They are the provenance
/// of the row: with the calibration in the header, anyone can recompute `x` and
/// `y` from them and check our arithmetic, or measure the printed figure and
/// check the digitisation itself.
pub fn to_csv(plot: &DigitisedPlot) -> String {
    let mut out = String::from("series,label,kind,x,y,page_x,page_y\n");
    for (i, s) in plot.series.iter().enumerate() {
        let label = s.label.clone().unwrap_or_default();
        let kind = match s.kind {
            crate::series::SeriesKind::Line => "line",
            crate::series::SeriesKind::Scatter => "scatter",
        };
        for p in &s.points {
            out.push_str(&format!(
                "{i},{},{kind},{},{},{},{}\n",
                quote(&label),
                p.x,
                p.y,
                p.page_xy.0,
                p.page_xy.1
            ));
        }
    }
    out
}

/// CSV with a `#`-commented provenance header.
///
/// This is the form to use for anything that will outlive the session that
/// produced it.
pub fn to_csv_with_provenance(plot: &DigitisedPlot, source: &str) -> String {
    let mut out = String::new();
    out.push_str("# Digitised from a printed figure by kopitiam-plot.\n");
    out.push_str("#\n");
    out.push_str("# THESE NUMBERS WERE RECOVERED FROM A PICTURE. They are not the author's\n");
    out.push_str("# original data. Check the warnings below before using them for anything\n");
    out.push_str("# that matters.\n");
    out.push_str("#\n");
    out.push_str(&format!("# source: {source}\n"));
    out.push_str(&format!("# page: {}\n", plot.page));
    out.push_str(&format!(
        "# plot region (PDF user space, pt): x {:.1}..{:.1}, y {:.1}..{:.1}\n",
        plot.region.x,
        plot.region.right(),
        plot.region.y,
        plot.region.top()
    ));

    out.push_str("#\n# calibration:\n");
    out.push_str(&format!(
        "#   {}\n",
        describe_calibration(&plot.axes.x, "x")
    ));
    out.push_str(&format!(
        "#   {}\n",
        describe_calibration(&plot.axes.y, "y")
    ));
    if let Some(t) = &plot.axes.x.title {
        out.push_str(&format!("#   x title: {t}\n"));
    }
    if let Some(t) = &plot.axes.y.title {
        out.push_str(&format!("#   y title: {t}\n"));
    }

    // The tick observations are the *evidence* the calibration rests on. Anyone
    // auditing a digitisation starts here: if a tick was misread, it is visible
    // in this table and nowhere else.
    for (name, axis) in [("x", &plot.axes.x), ("y", &plot.axes.y)] {
        if axis.ticks.is_empty() {
            continue;
        }
        out.push_str(&format!(
            "#\n# {name} tick observations (page coordinate -> value, as printed):\n"
        ));
        for t in &axis.ticks {
            out.push_str(&format!(
                "#   {:>8.2} -> {:<14} \"{}\" [{}]\n",
                t.page,
                t.value,
                t.text,
                match t.source {
                    crate::axes::TickSource::TickMark => "tick mark",
                    crate::axes::TickSource::LabelCentre => "label centre",
                }
            ));
        }
    }

    out.push_str("#\n# series:\n");
    for (i, s) in plot.series.iter().enumerate() {
        out.push_str(&format!(
            "#   [{i}] {} -- {} point(s), {}{}\n",
            s.label.as_deref().unwrap_or("(unlabelled)"),
            s.points.len(),
            s.style.describe(),
            if s.interpolated {
                ", ANCHORS OF A BEZIER PATH"
            } else {
                ""
            }
        ));
    }

    if plot.warnings.is_empty() {
        out.push_str("#\n# warnings: none.\n");
    } else {
        out.push_str("#\n# WARNINGS:\n");
        for w in &plot.warnings {
            out.push_str(&format!("#   ! {w}\n"));
        }
    }

    // Restate the axis scales right above the data: someone reading the numbers
    // must not have to scroll up to discover the y column came off a log axis.
    out.push_str(&format!(
        "#\n# x scale: {}   y scale: {}\n",
        scale_name(plot.axes.x.scale),
        scale_name(plot.axes.y.scale)
    ));

    out.push_str(&to_csv(plot));
    out
}

fn scale_name(scale: AxisScale) -> &'static str {
    match scale {
        AxisScale::Linear => "linear",
        AxisScale::Log10 => "log10",
    }
}

/// Quote a CSV field per RFC 4180, if it needs it.
fn quote(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotes_only_when_needed() {
        assert_eq!(quote("Re=100"), "Re=100");
        assert_eq!(quote("a,b"), "\"a,b\"");
        assert_eq!(quote("say \"hi\""), "\"say \"\"hi\"\"\"");
    }
}
