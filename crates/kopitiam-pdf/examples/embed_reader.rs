//! A minimal third-party egui application embedding the KOPITIAM PDF reader.
//!
//! This example exists to *prove* gh-96's first acceptance criterion, not
//! merely to illustrate it: **a downstream application can embed a working
//! reader using public library APIs, copying nothing from `kpdf.rs`.** If the
//! extraction ever regresses -- a type goes private, the reader starts
//! needing something only the binary has -- this stops compiling, and CI says
//! so.
//!
//! Run it:
//!
//! ```text
//! cargo run --release -p kopitiam-pdf --example embed_reader -- paper.pdf
//! ```
//!
//! Note what is NOT here, because the reader owns it: rasterisation, the
//! render worker, page layout, the continuous viewport, search, thumbnails,
//! vim keys, forms, annotations, caches. And note what IS here, because the
//! *host* owns it: reading a file, deciding what "save" means, and laying
//! out the window. That split is the whole point.

use std::path::PathBuf;

use kopitiam_pdf::gui_frontend::{PdfReader, PdfReaderConfig, ReaderAction};

struct HostApp {
    reader: PdfReader,
    /// Host state the reader knows nothing about -- here, a log of what the
    /// reader reported. A real application would put citations, notes, or a
    /// digitisation queue in this space.
    log: Vec<String>,
}

impl eframe::App for HostApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // The host owns the layout. The reader gets the space the host
        // decides to give it -- here, everything left of a side panel that
        // belongs entirely to this application.
        egui::Panel::right("host-panel")
            .resizable(true)
            .default_size(260.0)
            .show(ui, |ui| {
                ui.heading("Host application");
                ui.label(format!("page {} of {}", self.reader.current_page() + 1, self.reader.page_count()));
                ui.separator();
                ui.label("Reader events:");
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for line in self.log.iter().rev().take(50) {
                        ui.label(line);
                    }
                });
            });

        // `show` gives the reader's own chrome (toolbar, sidebars, find bar)
        // inside whatever Ui it is handed. To arrange those yourself instead,
        // call `pump`, then place `thumbnail_sidebar`/`outline_sidebar`/
        // `find_bar` where you like, and finish with `ui` for the page pane.
        let out = self.reader.show(ui);

        for action in out.actions {
            match action {
                ReaderAction::PageChanged { page } => {
                    self.log.push(format!("moved to page {}", page + 1));
                }
                ReaderAction::RegionSelected { page, rect } => {
                    // A research application would decide here whether this
                    // region is a graph, a table, or a formula. The reader
                    // deliberately has no opinion -- see `gui_frontend::action`.
                    self.log.push(format!(
                        "region on page {}: {:.0}x{:.0} pt",
                        page + 1,
                        rect.x1 - rect.x0,
                        rect.y1 - rect.y0
                    ));
                }
                ReaderAction::LinkActivated { destination } => {
                    self.log.push(format!("followed {destination:?}"));
                }
                // This host opened the document read-only, so there is
                // nothing to save and nowhere to save it. Saying so beats
                // silently ignoring the request.
                ReaderAction::SaveRequested | ReaderAction::SaveAsRequested => {
                    self.reader.set_status("this viewer is read-only");
                }
                ReaderAction::QuitRequested => {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
                _ => {}
            }
        }
    }
}

fn main() -> eframe::Result {
    let Some(path) = std::env::args().nth(1).map(PathBuf::from) else {
        eprintln!("usage: embed_reader <file.pdf>");
        std::process::exit(2);
    };
    // The host reads the file; the reader only ever sees bytes.
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            std::process::exit(1);
        }
    };
    // Read-only: every reading feature, no path by which a keystroke rewrites
    // someone's downloaded paper.
    let mut reader = match PdfReader::open_bytes_with(bytes, PdfReaderConfig::read_only()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            std::process::exit(1);
        }
    };
    reader.set_label(path.display().to_string());

    eframe::run_native(
        "embed_reader",
        eframe::NativeOptions::default(),
        Box::new(|_cc| {
            Ok(Box::new(HostApp {
                reader,
                log: Vec::new(),
            }))
        }),
    )
}
