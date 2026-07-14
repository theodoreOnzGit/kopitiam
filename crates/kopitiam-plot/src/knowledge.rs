//! Turning a digitised plot into entities for the semantic graph.
//!
//! CLAUDE.md is explicit that the platform should "not think in terms of files"
//! but in terms of engineering knowledge, and that a PDF is a *scientific
//! paper*, not a blob. A figure inside that paper is a published result, and the
//! series inside the figure are the result's actual content. So a digitised plot
//! must be able to enter the knowledge graph as knowledge -- something the
//! Literature Engine can search, a validation case can cite, and a solver
//! comparison can be run against -- rather than remaining a struct that only
//! this crate understands.
//!
//! The mapping:
//!
//! * The plot itself is a [`EntityKind::Section`] -- "a structural unit of a
//!   document", which is what a figure is.
//! * Each recovered series is a [`EntityKind::Fact`] -- "a deterministic,
//!   tool-derived observation". Which is exactly what it is: no model was
//!   involved in producing it, and running the digitiser again on the same PDF
//!   yields the same numbers.
//! * Each series is `LocatedIn` its plot.
//!
//! Every entity carries `source: "kopitiam-plot"`, and every series entity
//! carries its points *with their page coordinates* and the calibration they
//! were mapped through. A consumer that pulls a digitised curve out of the graph
//! years from now can still answer "where did this number come from?" without
//! the original PDF -- and, if it has the PDF, can verify it.

use kopitiam_ontology::{Entity, EntityKind, Relationship, RelationshipKind};
use serde_json::json;

use crate::digitise::DigitisedPlot;

/// The provider name recorded on every entity this crate emits.
pub const SOURCE: &str = "kopitiam-plot";

/// Convert a digitised plot into graph entities and the edges between them.
///
/// `document` names the source PDF, so a plot recovered from a paper can be
/// traced back to it.
pub fn to_entities(plot: &DigitisedPlot, document: &str) -> (Vec<Entity>, Vec<Relationship>) {
    let name = plot
        .axes
        .y
        .title
        .as_deref()
        .zip(plot.axes.x.title.as_deref())
        .map(|(y, x)| format!("{y} vs {x}"))
        .unwrap_or_else(|| format!("figure on page {}", plot.page));

    let plot_entity = Entity::new(EntityKind::Section, name, SOURCE).with_metadata(json!({
        "document": document,
        "page": plot.page,
        "region": plot.region,
        // The calibration travels with the plot, so a consumer can re-derive
        // any point from its page coordinate without re-reading the PDF.
        "axes": plot.axes,
        "warnings": plot.warnings,
        // A single flag a query can filter on: "give me only the digitisations
        // that carried no caveats".
        "clean": plot.is_clean(),
        "series_count": plot.series.len(),
        "point_count": plot.point_count(),
    }));

    let mut entities = vec![plot_entity.clone()];
    let mut relationships = Vec::new();

    for (i, s) in plot.series.iter().enumerate() {
        let label = s
            .label
            .clone()
            .unwrap_or_else(|| format!("series {i}"));
        let entity = Entity::new(EntityKind::Fact, label, SOURCE).with_metadata(json!({
            "document": document,
            "page": plot.page,
            "kind": s.kind,
            "style": s.style,
            "style_description": s.style.describe(),
            // The points, each with the page coordinate it was recovered from.
            "points": s.points,
            // Retained even when the axes did not calibrate: page geometry is
            // evidence that a curve exists, which is worth recording even when
            // its values are not knowable.
            "page_points": s.page_points,
            "interpolated": s.interpolated,
            "digitised": !s.points.is_empty(),
        }));
        relationships.push(Relationship::new(
            entity.id,
            plot_entity.id,
            RelationshipKind::LocatedIn,
        ));
        entities.push(entity);
    }

    (entities, relationships)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axes::{Axis, AxisScale};
    use crate::digitise::AxisCalibration;
    use crate::geometry::Rect;
    use crate::series::{DataPoint, Series, SeriesKind};
    use crate::style::{Paint, Rgb, SeriesStyle};

    fn axis(title: &str) -> Axis {
        Axis {
            scale: AxisScale::Linear,
            ticks: vec![],
            fit: None,
            title: Some(title.to_string()),
        }
    }

    fn plot() -> DigitisedPlot {
        DigitisedPlot {
            page: 7,
            region: Rect::from_corners(0.0, 0.0, 100.0, 100.0),
            axes: AxisCalibration {
                x: axis("time (s)"),
                y: axis("temperature (K)"),
            },
            series: vec![Series {
                points: vec![DataPoint {
                    x: 1.0,
                    y: 2.0,
                    page_xy: (10.0, 20.0),
                }],
                page_points: vec![(10.0, 20.0)],
                style: SeriesStyle {
                    paint: Paint::Stroke,
                    color: Rgb::BLACK,
                    line_width: 1.0,
                    dash: vec![],
                },
                kind: SeriesKind::Line,
                label: Some("experiment".into()),
                interpolated: false,
            }],
            warnings: vec![],
        }
    }

    #[test]
    fn emits_a_section_for_the_plot_and_a_fact_per_series() {
        let (entities, relationships) = to_entities(&plot(), "paper.pdf");
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].kind, EntityKind::Section);
        // The axis titles name the figure, which is how a human would refer to
        // it -- far more useful in a graph than "figure 3".
        assert_eq!(entities[0].name, "temperature (K) vs time (s)");
        assert_eq!(entities[1].kind, EntityKind::Fact);
        assert_eq!(entities[1].name, "experiment");
        assert_eq!(entities[0].source, SOURCE);

        assert_eq!(relationships.len(), 1);
        assert_eq!(relationships[0].kind, RelationshipKind::LocatedIn);
        assert_eq!(relationships[0].from, entities[1].id);
        assert_eq!(relationships[0].to, entities[0].id);
    }

    #[test]
    fn series_metadata_retains_page_provenance() {
        let (entities, _) = to_entities(&plot(), "paper.pdf");
        let points = &entities[1].metadata["points"];
        assert_eq!(points[0]["x"], 1.0);
        // The page coordinate must survive into the graph, or the number in the
        // graph is unauditable.
        assert_eq!(points[0]["page_xy"][0], 10.0);
        assert_eq!(points[0]["page_xy"][1], 20.0);
    }

    #[test]
    fn falls_back_to_the_page_when_axes_are_untitled() {
        let mut p = plot();
        p.axes.x.title = None;
        let (entities, _) = to_entities(&p, "paper.pdf");
        assert_eq!(entities[0].name, "figure on page 7");
    }
}
