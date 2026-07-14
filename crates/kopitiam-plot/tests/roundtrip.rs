//! **The round trip.** Take data we chose, draw it as a plot in a PDF we wrote,
//! digitise it, and assert the original numbers come back.
//!
//! This is the only test that can prove the whole pipeline, because it is the
//! only one with ground truth. Every real PDF has, by construction, lost the
//! numbers we would be checking against -- that is why this crate exists. So the
//! ground truth has to be data we own, drawn by a PDF writer we own
//! (`tests/common`), through a mapping the digitiser never gets to see.
//!
//! The tolerances here are deliberately tight. A vector plot carries the
//! producer's own coordinates, so recovery is limited only by the precision
//! those were written at -- roughly f32, i.e. seven significant figures. A
//! digitiser that only managed two or three would be doing something wrong, and
//! a loose tolerance would hide it.

mod common;

use common::{LabelStyle, PdfBuilder, Plot, Scale};
use kopitiam_plot::{AxisScale, SeriesKind, digitise_bytes};

const RED: (f32, f32, f32) = (0.9, 0.1, 0.1);
const BLUE: (f32, f32, f32) = (0.1, 0.2, 0.8);

/// The data every round-trip starts from: a curve with enough structure that a
/// mis-calibration cannot accidentally look right.
fn quadratic() -> Vec<(f64, f64)> {
    (0..=20)
        .map(|i| {
            let x = i as f64 / 20.0;
            (x, 0.1 + 0.8 * x * x)
        })
        .collect()
}

/// Largest absolute error between recovered and expected points.
fn max_abs_error(recovered: &[kopitiam_plot::DataPoint], expected: &[(f64, f64)]) -> f64 {
    assert_eq!(
        recovered.len(),
        expected.len(),
        "recovered {} points, expected {}",
        recovered.len(),
        expected.len()
    );
    recovered
        .iter()
        .zip(expected)
        .map(|(r, e)| (r.x - e.0).abs().max((r.y - e.1).abs()))
        .fold(0.0, f64::max)
}

/// Largest *relative* error, for log axes where absolute error is meaningless.
fn max_rel_error(recovered: &[kopitiam_plot::DataPoint], expected: &[(f64, f64)]) -> f64 {
    assert_eq!(recovered.len(), expected.len());
    recovered
        .iter()
        .zip(expected)
        .map(|(r, e)| ((r.x - e.0) / e.0).abs().max(((r.y - e.1) / e.1).abs()))
        .fold(0.0, f64::max)
}

#[test]
fn recovers_a_linear_plot_to_within_a_millionth() {
    let plot = Plot::linear_unit_square();
    let data = quadratic();

    let mut pdf = PdfBuilder::new(612.0, 792.0);
    plot.draw_axes(&mut pdf);
    plot.draw_line_series(&mut pdf, &data, RED, 1.5, &[]);

    let plots = digitise_bytes(&pdf.build(), 1).expect("must digitise");
    assert_eq!(plots.len(), 1, "exactly one plot on the page");
    let p = &plots[0];

    assert!(
        p.warnings.is_empty(),
        "a clean synthetic plot must produce no warnings, got: {:#?}",
        p.warnings
    );
    assert_eq!(p.axes.x.scale, AxisScale::Linear);
    assert_eq!(p.axes.y.scale, AxisScale::Linear);
    assert_eq!(p.axes.x.ticks.len(), 5);
    assert_eq!(p.axes.y.ticks.len(), 5);

    assert_eq!(p.series.len(), 1, "one series drawn, one recovered");
    let s = &p.series[0];
    assert_eq!(s.kind, SeriesKind::Line);
    assert!(!s.interpolated);

    let err = max_abs_error(&s.points, &data);
    assert!(
        err < 1e-5,
        "round trip must recover the data essentially exactly; max error {err:e}"
    );

    // Provenance: every point must carry the page coordinate it came from, and
    // that coordinate must actually be where the point was drawn.
    for (r, e) in s.points.iter().zip(&data) {
        let (px, py) = plot.to_page(e.0, e.1);
        assert!((r.page_xy.0 - px).abs() < 0.01 && (r.page_xy.1 - py).abs() < 0.01);
    }
}

#[test]
fn recovers_a_log_y_axis() {
    // y spans four decades. Read as linear, this data would come back smooth,
    // plausible and entirely wrong -- which is the failure this test exists for.
    let plot = Plot {
        y_range: (1e-4, 1.0),
        y_scale: Scale::Log10,
        y_ticks: vec![1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        label_style: LabelStyle::Decimal(4),
        ..Plot::linear_unit_square()
    };
    // An exponential curve: a straight line on a log axis.
    let data: Vec<(f64, f64)> = (0..=20)
        .map(|i| {
            let x = i as f64 / 20.0;
            (x, 10f64.powf(-4.0 * x))
        })
        .collect();

    let mut pdf = PdfBuilder::new(612.0, 792.0);
    plot.draw_axes(&mut pdf);
    plot.draw_line_series(&mut pdf, &data, BLUE, 1.0, &[]);

    let plots = digitise_bytes(&pdf.build(), 1).expect("must digitise");
    let p = &plots[0];

    assert_eq!(
        p.axes.y.scale,
        AxisScale::Log10,
        "log axis must be DETECTED, not assumed; warnings: {:#?}",
        p.warnings
    );
    assert_eq!(p.axes.x.scale, AxisScale::Linear, "x is still linear");

    let err = max_rel_error(&p.series[0].points, &data);
    assert!(err < 1e-5, "max relative error {err:e}");
}

#[test]
fn recovers_a_log_axis_labelled_with_powers_of_ten() {
    // matplotlib's default log labels are `10` with a raised exponent, not
    // `0.001`. Read naively that is the number ten at every tick -- which fits a
    // horizontal line perfectly and reports high confidence.
    let plot = Plot {
        y_range: (1e-3, 1.0),
        y_scale: Scale::Log10,
        y_ticks: vec![1e-3, 1e-2, 1e-1, 1.0],
        label_style: LabelStyle::Power10,
        // x ticks would also render as powers of ten under this label style, so
        // give x decade values too and let both axes come back log.
        x_range: (1.0, 1e4),
        x_scale: Scale::Log10,
        x_ticks: vec![1.0, 10.0, 100.0, 1000.0, 10000.0],
        ..Plot::linear_unit_square()
    };
    let data: Vec<(f64, f64)> = (0..=10)
        .map(|i| {
            let e = i as f64 / 10.0;
            (10f64.powf(4.0 * e), 10f64.powf(-3.0 * e))
        })
        .collect();

    let mut pdf = PdfBuilder::new(612.0, 792.0);
    plot.draw_axes(&mut pdf);
    plot.draw_line_series(&mut pdf, &data, RED, 1.0, &[]);

    let plots = digitise_bytes(&pdf.build(), 1).expect("must digitise");
    let p = &plots[0];

    // The tick labels must have been read as 10^n, not as the number 10.
    let values: Vec<f64> = p.axes.y.ticks.iter().map(|t| t.value).collect();
    assert!(
        values.iter().any(|v| (*v - 1e-3).abs() < 1e-12),
        "superscript exponent must be parsed; got tick values {values:?}"
    );
    assert_eq!(p.axes.y.scale, AxisScale::Log10, "{:#?}", p.warnings);
    assert_eq!(p.axes.x.scale, AxisScale::Log10);

    let err = max_rel_error(&p.series[0].points, &data);
    assert!(err < 1e-4, "max relative error {err:e}");
}

#[test]
fn separates_two_series_by_colour_and_dash_and_names_them_from_the_legend() {
    let plot = Plot::linear_unit_square();
    let experiment: Vec<(f64, f64)> = (0..=10)
        .map(|i| {
            let x = i as f64 / 10.0;
            (x, 0.1 + 0.8 * x)
        })
        .collect();
    let simulation: Vec<(f64, f64)> = (0..=10)
        .map(|i| {
            let x = i as f64 / 10.0;
            (x, 0.9 - 0.7 * x)
        })
        .collect();

    let mut pdf = PdfBuilder::new(612.0, 792.0);
    plot.draw_axes(&mut pdf);
    plot.draw_line_series(&mut pdf, &experiment, RED, 1.5, &[]);
    plot.draw_line_series(&mut pdf, &simulation, BLUE, 1.0, &[4.0, 2.0]);
    plot.draw_legend_entry(&mut pdf, 0, "experiment", RED, 1.5, &[]);
    plot.draw_legend_entry(&mut pdf, 1, "simulation", BLUE, 1.0, &[4.0, 2.0]);

    let plots = digitise_bytes(&pdf.build(), 1).expect("must digitise");
    let p = &plots[0];

    assert_eq!(
        p.series.len(),
        2,
        "two styles drawn, two series expected; warnings: {:#?}",
        p.warnings
    );

    let by_label = |name: &str| {
        p.series
            .iter()
            .find(|s| s.label.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("series '{name}' must be named from the legend"))
    };

    // The legend's own sample lines must NOT have been appended to the data --
    // they are drawn in the series' exact style, so a digitiser that ignores
    // legends silently gains two bogus points per series.
    let exp = by_label("experiment");
    assert_eq!(exp.points.len(), experiment.len(), "no legend contamination");
    assert!(max_abs_error(&exp.points, &experiment) < 1e-5);

    let sim = by_label("simulation");
    assert_eq!(sim.points.len(), simulation.len());
    assert!(max_abs_error(&sim.points, &simulation) < 1e-5);

    // And the styles really were what distinguished them.
    assert!(exp.style.dash.is_empty());
    assert!(!sim.style.dash.is_empty(), "the dashed series must be dashed");
}

#[test]
fn recovers_a_scatter_series_as_points_not_a_curve() {
    let plot = Plot::linear_unit_square();
    let measured: Vec<(f64, f64)> = (0..=8)
        .map(|i| {
            let x = i as f64 / 8.0;
            (x, 0.15 + 0.7 * x * x)
        })
        .collect();
    let fitted = quadratic();

    let mut pdf = PdfBuilder::new(612.0, 792.0);
    plot.draw_axes(&mut pdf);
    plot.draw_line_series(&mut pdf, &fitted, BLUE, 1.0, &[]);
    plot.draw_scatter_series(&mut pdf, &measured, RED);

    let plots = digitise_bytes(&pdf.build(), 1).expect("must digitise");
    let p = &plots[0];

    let scatter = p
        .series
        .iter()
        .find(|s| s.kind == SeriesKind::Scatter)
        .unwrap_or_else(|| panic!("scatter must be recognised; got {:#?}", p.series));
    let line = p
        .series
        .iter()
        .find(|s| s.kind == SeriesKind::Line)
        .expect("the fitted line must still be a line");

    // Markers are painted with `B` (fill+stroke) -- one of the five operators
    // pdf-extract drops on the floor. Recovering them at all is the payoff for
    // walking the content stream ourselves.
    assert_eq!(scatter.points.len(), measured.len());
    assert!(
        max_abs_error(&scatter.points, &measured) < 1e-3,
        "marker centres must land on the data"
    );
    assert!(max_abs_error(&line.points, &fitted) < 1e-5);
}

#[test]
fn too_few_ticks_warns_loudly_and_produces_no_numbers() {
    // One labelled x tick. The axis is genuinely uncalibratable, and the ONLY
    // acceptable behaviour is to say so and produce nothing. A confident wrong
    // answer here is how a fabricated validation dataset gets published.
    let plot = Plot {
        x_ticks: vec![0.5],
        ..Plot::linear_unit_square()
    };

    let mut pdf = PdfBuilder::new(612.0, 792.0);
    plot.draw_axes(&mut pdf);
    plot.draw_line_series(&mut pdf, &quadratic(), RED, 1.5, &[]);

    let plots = digitise_bytes(&pdf.build(), 1).expect("must still return the plot");
    let p = &plots[0];

    assert!(!p.axes.x.is_calibrated(), "must not invent an x calibration");
    assert!(p.axes.y.is_calibrated(), "y had 5 ticks and is fine");
    assert!(!p.axes.is_complete());
    assert!(!p.is_clean());

    assert!(
        p.warnings.iter().any(|w| w.contains("cannot calibrate")),
        "must say it cannot calibrate: {:#?}",
        p.warnings
    );
    assert!(
        p.warnings.iter().any(|w| w.contains("NO data values")),
        "must say no values were produced: {:#?}",
        p.warnings
    );

    // The curve was still *found* -- its geometry is evidence -- but no data
    // values were manufactured from it.
    let s = &p.series[0];
    assert!(s.points.is_empty(), "no calibration means no numbers");
    assert!(!s.page_points.is_empty(), "but the geometry is still reported");
}

#[test]
fn two_ticks_cannot_determine_the_scale_and_admits_it() {
    // Two ticks fit linear and log equally well. The scale is undecidable from
    // the figure, and the digitiser must say so rather than quietly picking one.
    //
    // Both ticks are positive on purpose: a tick at zero would *rule out* a log
    // axis (log 0 is undefined), so there would be no ambiguity left to warn
    // about. The dangerous case is the one where both readings remain possible.
    let plot = Plot {
        x_ticks: vec![0.25, 1.0],
        ..Plot::linear_unit_square()
    };

    let mut pdf = PdfBuilder::new(612.0, 792.0);
    plot.draw_axes(&mut pdf);
    plot.draw_line_series(&mut pdf, &quadratic(), RED, 1.5, &[]);

    let plots = digitise_bytes(&pdf.build(), 1).expect("must digitise");
    let p = &plots[0];

    assert!(
        p.warnings
            .iter()
            .any(|w| w.contains("ASSUMED LINEAR") && w.contains("x-axis")),
        "must warn that the x scale was assumed, not determined: {:#?}",
        p.warnings
    );
    // ...and separately, that a 2-point fit has no redundancy to catch a misread
    // label, whichever scale it settled on.
    assert!(
        p.warnings.iter().any(|w| w.contains("no redundancy")),
        "must warn that a 2-tick fit cannot be cross-checked: {:#?}",
        p.warnings
    );
    // It still produces values (linear is the right guess here) -- but the
    // caller has been told, in terms, that they rest on an assumption.
    assert!(p.axes.x.is_calibrated());
    assert!(!p.is_clean());
    // And the values are in fact right, which is exactly why the warning matters:
    // a plausible answer is not the same as a justified one.
    assert!(max_abs_error(&p.series[0].points, &quadratic()) < 1e-5);
}

#[test]
fn a_page_with_no_plot_yields_no_plots() {
    // A table -- two long horizontal rules and some numbers -- must not be
    // digitised as a graph. False positives are their own kind of fabrication.
    let mut pdf = PdfBuilder::new(612.0, 792.0);
    pdf.op("q 0 0 0 RG 1 w");
    pdf.op("100 700 m 500 700 l S");
    pdf.op("100 600 m 500 600 l S");
    pdf.op("Q");
    pdf.op("BT /F1 10 Tf 120 650 Td (Re) Tj ET");
    pdf.op("BT /F1 10 Tf 300 650 Td (1000) Tj ET");

    let plots = digitise_bytes(&pdf.build(), 1).expect("must not error");
    assert!(plots.is_empty(), "a table is not a plot, got {plots:#?}");
}

#[test]
fn exports_csv_with_a_provenance_header() {
    let plot = Plot::linear_unit_square();
    let data = quadratic();
    let mut pdf = PdfBuilder::new(612.0, 792.0);
    plot.draw_axes(&mut pdf);
    plot.draw_line_series(&mut pdf, &data, RED, 1.5, &[]);

    let plots = digitise_bytes(&pdf.build(), 1).unwrap();
    let csv = kopitiam_plot::to_csv_with_provenance(&plots[0], "synthetic.pdf");

    assert!(csv.contains("# source: synthetic.pdf"));
    assert!(csv.contains("RECOVERED FROM A PICTURE"));
    // The calibration and the evidence for it must travel with the numbers.
    assert!(csv.contains("x: linear, fitted from 5 tick(s)"));
    assert!(csv.contains("tick observations"));
    assert!(csv.contains("# warnings: none."));
    assert!(csv.contains("series,label,kind,x,y,page_x,page_y"));

    // The data rows must be parseable and correct.
    let rows: Vec<&str> = csv
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("series,"))
        .filter(|l| !l.trim().is_empty())
        .collect();
    assert_eq!(rows.len(), data.len());

    let first: Vec<&str> = rows[0].split(',').collect();
    let x: f64 = first[3].parse().expect("x must be a number");
    let y: f64 = first[4].parse().expect("y must be a number");
    assert!((x - data[0].0).abs() < 1e-5 && (y - data[0].1).abs() < 1e-5);
    // ...and each row must carry the page coordinate it came from.
    assert!(first[5].parse::<f64>().is_ok());
    assert!(first[6].parse::<f64>().is_ok());
}

#[test]
fn digitised_plots_become_knowledge_graph_entities() {
    let plot = Plot::linear_unit_square();
    let mut pdf = PdfBuilder::new(612.0, 792.0);
    plot.draw_axes(&mut pdf);
    plot.draw_line_series(&mut pdf, &quadratic(), RED, 1.5, &[]);
    plot.draw_legend_entry(&mut pdf, 0, "experiment", RED, 1.5, &[]);

    let plots = digitise_bytes(&pdf.build(), 1).unwrap();
    let (entities, relationships) = kopitiam_plot::to_entities(&plots[0], "paper.pdf");

    assert_eq!(entities.len(), 2, "one Section for the plot, one Fact per series");
    assert_eq!(entities[1].name, "experiment");
    assert_eq!(relationships.len(), 1);
    // The page coordinate must survive into the graph, or a number pulled out of
    // the graph years later is unauditable.
    assert!(entities[1].metadata["points"][0]["page_xy"].is_array());
}

#[test]
fn subplots_on_one_page_are_digitised_independently() {
    // Multi-panel figures are the norm in the literature, and each panel has its
    // own axes and its own calibration.
    let left = Plot::linear_unit_square();
    let right = Plot {
        x0: 100.0,
        y0: 620.0,
        x1: 500.0,
        y1: 760.0,
        y_range: (0.0, 10.0),
        y_ticks: vec![0.0, 5.0, 10.0],
        label_style: LabelStyle::Decimal(1),
        ..Plot::linear_unit_square()
    };

    let mut pdf = PdfBuilder::new(612.0, 792.0);
    left.draw_axes(&mut pdf);
    left.draw_line_series(&mut pdf, &quadratic(), RED, 1.5, &[]);

    let upper: Vec<(f64, f64)> = (0..=10)
        .map(|i| {
            let x = i as f64 / 10.0;
            (x, 10.0 * x)
        })
        .collect();
    right.draw_axes(&mut pdf);
    right.draw_line_series(&mut pdf, &upper, BLUE, 1.0, &[]);

    let plots = digitise_bytes(&pdf.build(), 1).expect("must digitise");
    assert_eq!(plots.len(), 2, "two panels, two plots");

    // Each panel calibrated on its own y range -- 0..1 and 0..10.
    let mut maxima: Vec<f64> = plots
        .iter()
        .map(|p| {
            p.series[0]
                .points
                .iter()
                .map(|pt| pt.y)
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    maxima.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((maxima[0] - 0.9).abs() < 1e-3, "lower panel peaks at 0.9");
    assert!((maxima[1] - 10.0).abs() < 1e-3, "upper panel peaks at 10");
}
