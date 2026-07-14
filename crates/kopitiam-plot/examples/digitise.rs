//! Digitise the plots in a real PDF and print the data as CSV.
//!
//! ```text
//! cargo run --release -p kopitiam-plot --example digitise -- paper.pdf        # survey every page
//! cargo run --release -p kopitiam-plot --example digitise -- paper.pdf 7      # CSV for page 7
//! ```
//!
//! With no page number this surveys the document and reports what it found (and
//! what it was unsure about) without printing data -- which is the right first
//! move on an unfamiliar paper, because it tells you which figures are vector
//! plots the crate can actually recover and which are raster images it cannot.
//!
//! This exists so the crate can be pointed at a real paper in one command. A
//! digitiser you cannot easily aim at your own PDF is a digitiser nobody checks.

use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(pdf) = args.next() else {
        eprintln!("usage: digitise <file.pdf> [page]");
        std::process::exit(2);
    };
    let page: Option<usize> = args.next().and_then(|p| p.parse().ok());

    match page {
        Some(page) => dump_csv(&pdf, page),
        None => survey(&pdf),
    }
}

/// Print the CSV, with its provenance header, for one page.
fn dump_csv(pdf: &str, page: usize) {
    match kopitiam_plot::digitise(Path::new(pdf), page) {
        Ok(plots) if plots.is_empty() => eprintln!("no plot found on page {page}"),
        Ok(plots) => {
            for plot in &plots {
                print!("{}", kopitiam_plot::to_csv_with_provenance(plot, pdf));
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

/// Walk every page and report what was found, without printing data.
fn survey(pdf: &str) {
    let Ok(bytes) = std::fs::read(pdf) else {
        eprintln!("cannot read {pdf}");
        std::process::exit(1);
    };
    let Ok(doc) = lopdf::Document::load_mem(&bytes) else {
        eprintln!("cannot parse {pdf}");
        std::process::exit(1);
    };

    let pages = doc.get_pages().len();
    println!("{pdf}: {pages} page(s)");

    for page in 1..=pages {
        let paths = kopitiam_plot::content::paths_on_page(&doc, page as u32);
        let plots = match kopitiam_plot::digitise_bytes(&bytes, page) {
            Ok(p) => p,
            Err(e) => {
                println!("  page {page}: error: {e}");
                continue;
            }
        };
        if plots.is_empty() {
            // Worth reporting: a page dense with vector paths but no detected
            // plot is either a diagram, or a figure this crate failed on -- and
            // the difference matters to whoever is checking it.
            if paths.len() > 50 {
                println!("  page {page}: {} vector paths, no plot detected", paths.len());
            }
            continue;
        }

        println!("\n  page {page}: {} vector paths -> {} plot(s)", paths.len(), plots.len());
        for plot in &plots {
            println!(
                "    region ({:.0},{:.0}) {:.0}x{:.0}pt, {} series, {} points",
                plot.region.x,
                plot.region.y,
                plot.region.width,
                plot.region.height,
                plot.series.len(),
                plot.point_count()
            );
            println!(
                "      {}",
                kopitiam_plot::describe_calibration(&plot.axes.x, "x")
            );
            println!(
                "      {}",
                kopitiam_plot::describe_calibration(&plot.axes.y, "y")
            );
            if let Some(t) = &plot.axes.x.title {
                println!("      x title: {t}");
            }
            for s in &plot.series {
                println!(
                    "      [{:?}] {} points, label {:?} -- {}",
                    s.kind,
                    s.points.len(),
                    s.label,
                    s.style.describe()
                );
            }
            for w in &plot.warnings {
                println!("      ! {w}");
            }
        }
    }
}
