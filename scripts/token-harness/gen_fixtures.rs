// Synthetic PDF fixture generator for the Task I-0 token-measurement harness.
//
// SPDX-License-Identifier: AGPL-3.0-only
//
// This is a STANDALONE helper. It has no dependency on any workspace crate
// and pulls in no external crates: it hand-writes minimal, uncompressed
// PDF 1.4 bytes (base-14 Helvetica, no font embedding) with a correct
// cross-reference table, computing every byte offset as it emits objects.
// Run it with plain `rustc` (see scripts/token-harness/README for the exact
// command); the resulting small synthetic PDFs are COMMITTED so the harness
// never has to regenerate them at measure time.
//
// The fixtures deliberately reproduce STRUCTURAL ARCHETYPES only (a two-column
// page with a running head + footer page number, a figure caption with
// scattered short labels, a ragged table, and a plain prose control). They are
// invented text, not any real paper -- see kopitiam_token_max.md §2.4 (no
// copyrighted fixtures).
//
// Coordinate system: PDF user space, origin bottom-left, US Letter 612x792.
// The top 10% zone (running heads) is y in [712.8, 792]; the bottom 10% zone
// (footer page numbers) is y in [0, 79.2]. Headers are placed at y=760 and
// footer page numbers at y=36 so the header/footer stripper's zones catch them.

use std::fmt::Write as _;
use std::path::Path;

/// One positioned text run on a page.
struct Item {
    x: f64,
    y: f64,
    size: f64,
    text: String,
}

fn t(x: f64, y: f64, size: f64, text: &str) -> Item {
    Item { x, y, size, text: text.to_string() }
}

/// Escape a literal string for a PDF `(...)` string object.
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '(' => out.push_str("\\("),
            ')' => out.push_str("\\)"),
            _ => out.push(c),
        }
    }
    out
}

/// Build a content stream for one page from its positioned items.
fn content_stream(items: &[Item]) -> String {
    let mut s = String::new();
    for it in items {
        // A fresh text object per run keeps absolute positioning trivial and
        // deterministic (no dependence on prior Td state).
        let _ = write!(
            s,
            "BT /F1 {:.1} Tf {:.1} {:.1} Td ({}) Tj ET\n",
            it.size,
            it.x,
            it.y,
            esc(&it.text)
        );
    }
    s
}

/// Assemble a full PDF from a list of pages (each a list of items) and write
/// it to `path`. Object layout: 1=catalog, 2=pages, 3=font, then per page a
/// page object and a content object.
fn write_pdf(path: &Path, pages: &[Vec<Item>]) {
    let n = pages.len();
    let mut buf: Vec<u8> = Vec::new();
    // Object byte offsets, indexed by object number (index 0 unused / free).
    let obj_count = 3 + 2 * n;
    let mut offsets = vec![0usize; obj_count + 1];

    buf.extend_from_slice(b"%PDF-1.4\n");
    // A binary marker comment: conventional, keeps tools from mis-sniffing.
    buf.extend_from_slice(b"%\xE2\xE3\xCF\xD3\n");

    let emit = |buf: &mut Vec<u8>, offsets: &mut Vec<usize>, num: usize, body: &str| {
        offsets[num] = buf.len();
        buf.extend_from_slice(body.as_bytes());
    };

    // 1: Catalog
    emit(&mut buf, &mut offsets, 1, "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    // 2: Pages
    let mut kids = String::new();
    for i in 0..n {
        let page_obj = 4 + 2 * i;
        let _ = write!(kids, "{} 0 R ", page_obj);
    }
    let pages_obj = format!(
        "2 0 obj\n<< /Type /Pages /Kids [{}] /Count {} >>\nendobj\n",
        kids.trim_end(),
        n
    );
    emit(&mut buf, &mut offsets, 2, &pages_obj);

    // 3: Font (base-14 Helvetica, no embedding)
    emit(
        &mut buf,
        &mut offsets,
        3,
        "3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    // Per-page objects.
    for (i, items) in pages.iter().enumerate() {
        let page_obj = 4 + 2 * i;
        let content_obj = 5 + 2 * i;
        let stream = content_stream(items);
        let page = format!(
            "{} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
             /Resources << /Font << /F1 3 0 R >> >> /Contents {} 0 R >>\nendobj\n",
            page_obj, content_obj
        );
        emit(&mut buf, &mut offsets, page_obj, &page);
        let content = format!(
            "{} 0 obj\n<< /Length {} >>\nstream\n{}endstream\nendobj\n",
            content_obj,
            stream.len(),
            stream
        );
        emit(&mut buf, &mut offsets, content_obj, &content);
    }

    // Cross-reference table.
    let xref_offset = buf.len();
    let total = obj_count + 1; // including free object 0
    let mut xref = format!("xref\n0 {}\n", total);
    xref.push_str("0000000000 65535 f \n");
    for num in 1..=obj_count {
        let _ = write!(xref, "{:010} 00000 n \n", offsets[num]);
    }
    buf.extend_from_slice(xref.as_bytes());

    let trailer = format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
        total, xref_offset
    );
    buf.extend_from_slice(trailer.as_bytes());

    std::fs::write(path, &buf).expect("write pdf");
    eprintln!("wrote {} ({} bytes, {} pages)", path.display(), buf.len(), n);
}

// ---------------------------------------------------------------------------
// Fixture archetypes
// ---------------------------------------------------------------------------

const HEAD_Y: f64 = 760.0; // top 10% zone
const FOOT_Y: f64 = 36.0; // bottom 10% zone
const COL_L: f64 = 72.0;
const COL_R: f64 = 320.0;
const BODY_TOP: f64 = 700.0;
const LEADING: f64 = 16.0;

/// (a) Two-column page with a running head and a footer page number, repeated
/// across several pages so the header/footer stripper's cross-page signature
/// logic engages. After I-C, the running head and bare page numbers should be
/// gone -> bare-page-number count ~0.
fn fixture_two_column() -> Vec<Vec<Item>> {
    let head = "Proceedings of the Synthetic Symposium on Nothing in Particular";
    let left_lines = [
        "The reconstruction pipeline linearises columns into a single",
        "reading order before any block classification runs. Each span",
        "carries a baseline and an x position, and the paragraph builder",
        "walks them top to bottom within a detected column boundary.",
        "A running head repeated on every page carries no information for",
        "an agent that greps the output, yet it is emitted verbatim once",
        "per page unless a header zone strips it before rendering.",
        "This paragraph is ordinary prose and must survive conversion.",
    ];
    let right_lines = [
        "Footers and bare page numbers behave the same way: they recur",
        "in a fixed page zone with a digit-normalised signature, and a",
        "conservative stripper drops them only when they recur across a",
        "minimum number of pages so a short document is left untouched.",
        "The two-column archetype exercises exactly that machinery while",
        "keeping the body text long enough that the recovery ratio stays",
        "honest and well inside the accepted band for a clean document.",
        "None of this text is drawn from any real publication whatsoever.",
    ];
    let mut pages = Vec::new();
    for p in 1..=6 {
        let mut items = Vec::new();
        items.push(t(COL_L, HEAD_Y, 9.0, head));
        // Footer: bare page number, centred-ish.
        items.push(t(300.0, FOOT_Y, 9.0, &format!("{}", p)));
        let mut y = BODY_TOP;
        for line in left_lines {
            items.push(t(COL_L, y, 10.0, line));
            y -= LEADING;
        }
        let mut y = BODY_TOP;
        for line in right_lines {
            items.push(t(COL_R, y, 10.0, line));
            y -= LEADING;
        }
        pages.push(items);
    }
    pages
}

/// (b) A figure: one real "Figure 1." caption plus a scatter of short internal
/// labels (2-3 words, no terminal punctuation) at irregular left edges -- the
/// vector-art-label archetype. After I-D, the labels collapse and only the
/// caption survives -> few short paragraphs.
fn fixture_figure() -> Vec<Vec<Item>> {
    let caption =
        "Figure 1. System architecture of the synthetic digital twin, showing the data ingestion, model, and visualisation tiers described in the text.";
    // Scattered short labels at varied x positions and y rows.
    let labels = [
        "Sensor Gateway", "Message Bus", "Stream Ingest", "Feature Store",
        "Model Runner", "State Cache", "Twin Core", "Sync Engine",
        "Geometry Store", "Mesh Loader", "Render Tier", "View Cache",
        "Alert Rules", "Anomaly Flag", "Ops Console", "Audit Log",
        "Edge Node", "Field Bus", "PLC Bridge", "Historian",
        "Batch Job", "Replay Buf", "Schema Reg", "Type Map",
        "Auth Guard", "Token Vault", "Rate Limit", "Trace Span",
        "Health Ping", "Cold Store",
    ];
    let mut items = Vec::new();
    // Caption near the bottom of the figure region (still above footer zone).
    items.push(t(COL_L, 120.0, 10.0, caption));
    // Scatter labels across the upper page area on an irregular grid.
    let xs = [90.0, 170.0, 250.0, 330.0, 410.0, 490.0];
    let mut i = 0usize;
    let mut y = 680.0;
    while i < labels.len() {
        for &x in xs.iter() {
            if i >= labels.len() {
                break;
            }
            // Jitter x a little so left edges are inconsistent, like vector art.
            let jitter = ((i as f64) * 7.0) % 13.0 - 6.0;
            items.push(t(x + jitter, y, 8.0, labels[i]));
            i += 1;
        }
        y -= 44.0;
    }
    vec![items]
}

/// (c) A ragged table: a header row and three uniform 4-column rows, then a
/// ragged row (a merged/short row with fewer cells). I-E should emit the valid
/// prefix as a table and let the ragged remainder fall through.
fn fixture_ragged_table() -> Vec<Vec<Item>> {
    // Column x positions.
    let cols = [72.0, 200.0, 330.0, 460.0];
    let rows: [[&str; 4]; 4] = [
        ["Component", "Latency ms", "Throughput", "Status"],
        ["Gateway", "12", "8400", "OK"],
        ["Ingest", "31", "6200", "OK"],
        ["Model", "58", "1100", "Degraded"],
    ];
    let mut items = Vec::new();
    // Intro prose so the page is not table-only.
    items.push(t(72.0, 720.0, 10.0, "The measured tiers of the synthetic pipeline are summarised below."));
    let mut y = 680.0;
    for row in rows.iter() {
        for (c, cell) in row.iter().enumerate() {
            items.push(t(cols[c], y, 10.0, cell));
        }
        y -= 20.0;
    }
    // Ragged row: a single merged/short cell spanning, only one value.
    items.push(t(72.0, y, 10.0, "Total across all tiers: 15700 requests per second"));
    y -= 30.0;
    items.push(t(72.0, y, 10.0, "Values above are synthetic and illustrate the ragged-row case only."));
    vec![items]
}

/// (d) Plain single-column prose control: no header, no footer number, no
/// figure, no table. The bare-page-number and short-paragraph counts should
/// be ~0 here regardless of any wave, so it anchors the corpus.
fn fixture_prose() -> Vec<Vec<Item>> {
    let paras = [
        [
            "A digital twin is a synchronised virtual model of a physical system,",
            "kept current by a stream of measurements from that system and used to",
            "reason about states the physical system has not yet reached. The value",
            "of such a model rises with the fidelity of its synchronisation and with",
            "the breadth of the questions an operator can pose against it cheaply.",
        ],
        [
            "This control fixture contains only ordinary prose in a single column.",
            "It carries no running head, no footer page number, no figure labels,",
            "and no tabular material, so every structural stripper in the pipeline",
            "should leave it essentially untouched. Its recovery ratio is therefore",
            "the cleanest signal in the corpus for spotting an accidental regression.",
        ],
        [
            "Because the text reads as continuous sentences with terminal punctuation,",
            "no run of short unpunctuated lines exists for a figure heuristic to catch,",
            "and no digit-only line exists for a page-number rule to strip. A change",
            "that alters this fixture's numbers points at a bug in general prose",
            "handling rather than in any of the targeted structural special cases.",
        ],
    ];
    let mut items = Vec::new();
    let mut y = 720.0;
    for para in paras.iter() {
        for line in para.iter() {
            items.push(t(72.0, y, 11.0, line));
            y -= 16.0;
        }
        y -= 12.0; // paragraph gap
    }
    vec![items]
}

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let dir = Path::new(&out_dir);
    std::fs::create_dir_all(dir).expect("create out dir");

    write_pdf(&dir.join("a_two_column.pdf"), &fixture_two_column());
    write_pdf(&dir.join("b_figure.pdf"), &fixture_figure());
    write_pdf(&dir.join("c_ragged_table.pdf"), &fixture_ragged_table());
    write_pdf(&dir.join("d_prose.pdf"), &fixture_prose());
}
