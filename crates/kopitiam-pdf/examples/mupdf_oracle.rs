//! **Code-to-code harness: kopitiam-pdf vs real MuPDF, same input, three
//! layers compared.**
//!
//! kopitiam-pdf is a translation of MuPDF (`source/fitz` + `source/pdf`,
//! commit `19f1284`, AGPL-3.0, © Artifex). Reading the two codes side by side
//! only catches what the reader thinks to look for; the 0.4.1 blank-scan bug
//! (indirect `/Contents` array never run) sat there for months because no
//! fixture had that shape. This harness asks the upstream program itself, on
//! the SAME file, and diffs the answers -- the llama.cpp "reference oracle"
//! idea (docs/REFERENCE-ORACLE.md) applied to the PDF port.
//!
//! ```sh
//! cargo run --release -p kopitiam-pdf --example mupdf_oracle -- \
//!     --mutool crates/kopitiam-pdf/vendor/mupdf/build/release/mutool \
//!     [--dpi 72] [--max-pages 20] [--tsv out.tsv] [--no-raster] [--no-text] \
//!     [--no-objects] file1.pdf file2.pdf ...
//! ```
//!
//! `--mutool` may also come from the `MUTOOL` env var. How to build the pinned
//! `mutool` is in `docs/mupdf-code-to-code.md`.
//!
//! # What is compared, and the pass criterion for each (fixed BEFORE measuring)
//!
//! 1. **Objects / xref / stream decode.** `mutool run` walks every object
//!    number `1..countObjects()`, classifies it (null/bool/int/real/string/
//!    name/array/dict/stream) and saves each stream's *decoded* bytes
//!    (`readStream`, i.e. `pdf_load_stream`). We resolve the same numbers and
//!    call [`PdfDocument::open_stream_num`]. PASS per object = same kind, and
//!    for a stream, byte-identical decoded data. Filters are lossless, so the
//!    only acceptable tolerance here is zero.
//! 2. **Structured text.** `mutool draw -F stext -O
//!    accurate-bboxes=no,collect-styles=no` (leaving MuPDF's `FZ_STEXT_CLIP`,
//!    which `mutool draw` always sets) vs our
//!    [`page_to_stext`] with the same flag. Every non-space char of MuPDF's is
//!    matched to an unused char of ours with the same code point whose origin
//!    lies within [`MATCH_RADIUS_PT`]. PASS per page = zero unmatched on both
//!    sides AND the non-space text in reading order is identical. The max
//!    origin error over matched chars is reported, not gated.
//! 3. **Raster.** `mutool draw -N -M 0 -r DPI -c rgb` (PPM: no ICC, no
//!    spot simulation -- the mode the port targets) vs our
//!    [`rasterize_page_ex`] -- the *native* kopitiam engine, never the hayro
//!    fallback, because it is the port under test. A pixel is a *gross*
//!    mismatch when any RGB channel differs by more than 128 (one side says
//!    ink, the other paper -- or the ink is the wrong hue); the first version
//!    used luma only, which let red-for-green through (see `compare_raster`). PASS per page = gross mismatches ≤ [`RASTER_GROSS_PASS`] of the
//!    page. Why a tolerance at all: AID-0052 -- our glyph outlines come from
//!    from-spec decoders / skrifa, not FreeType, and the scan converter is not a
//!    fixed-point GEL clone, so anti-aliased edges legitimately differ by
//!    partial coverage. A missing feature (a shading, an inline image, a dash
//!    pattern) moves whole regions and blows straight through 1 %.
//!
//! The corpus is a runtime argument, never a fixture: only openly licensed
//! PDFs may be fed to it for recorded results (see the methodology doc), and
//! nothing it reads is ever copied into the repo.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use kopitiam_pdf::mupdf::draw_device::rasterize_page_ex;
use kopitiam_pdf::mupdf::object::Object;
use kopitiam_pdf::mupdf::structured_text::StextOptions;
use kopitiam_pdf::mupdf::{PdfDocument, page_to_stext};

/// Radius (points) within which a char of ours may match MuPDF's char with the
/// same code point. One point is ~a fifth of an average glyph advance at 10 pt,
/// small enough that two adjacent identical letters cannot cross-match.
const MATCH_RADIUS_PT: f32 = 1.0;

/// Raster pass bound: fraction of page pixels allowed to be gross mismatches.
const RASTER_GROSS_PASS: f64 = 0.01;

/// Gross-mismatch luma threshold (out of 255).
const GROSS_LUMA: i32 = 128;

/// The `mutool run` script: one line per object (`num kind len`) and each
/// stream's decoded bytes saved to `<dir>/<num>.bin`. mujs is ES5 -- no
/// `Math.imul`, no typed arrays -- so the bytes go through a file rather than
/// being hashed in JS.
const OBJECTS_JS: &str = r#"
var doc = Document.openDocument(scriptArgs[0]).asPDF();
var dir = scriptArgs[1];
var n = doc.countObjects();
for (var i = 1; i < n; i++) {
  var kind = "?", len = -1;
  try {
    var o = doc.newIndirect(i, 0);
    if (o.isStream()) {
      kind = "stream";
      try { var b = o.readStream(); len = b.getLength(); b.save(dir + "/" + i + ".bin"); }
      catch (e) { kind = "stream-err"; }
    } else {
      var r = o.resolve();
      if (r.isDictionary()) kind = "dict";
      else if (r.isArray()) kind = "array";
      else if (r.isName()) kind = "name";
      else if (r.isString()) kind = "string";
      else if (r.isInteger()) kind = "int";
      else if (r.isNumber()) kind = "real";
      else if (r.isBoolean()) kind = "bool";
      else if (r.isNull()) kind = "null";
    }
  } catch (e) { kind = "err"; }
  print(i + " " + kind + " " + len);
}
"#;

struct Args {
    mutool: PathBuf,
    dpi: f32,
    max_pages: usize,
    tsv: Option<PathBuf>,
    raster: bool,
    text: bool,
    objects: bool,
    /// Print up to this many unmatched chars per page (debugging aid).
    show_unmatched: usize,
    /// Write `<file>-p<N>-{ours,mupdf}.ppm` here for every raster-failing page.
    dump: Option<PathBuf>,
    /// 1-based first page to compare (default 1).
    first_page: usize,
    files: Vec<PathBuf>,
}

fn parse_args() -> Args {
    let mut a = Args {
        mutool: std::env::var_os("MUTOOL").map(PathBuf::from).unwrap_or_default(),
        dpi: 72.0,
        max_pages: usize::MAX,
        tsv: None,
        raster: true,
        text: true,
        objects: true,
        show_unmatched: 0,
        dump: None,
        first_page: 1,
        files: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(s) = it.next() {
        match s.as_str() {
            "--mutool" => a.mutool = it.next().expect("--mutool PATH").into(),
            "--dpi" => a.dpi = it.next().expect("--dpi N").parse().expect("dpi"),
            "--max-pages" => a.max_pages = it.next().expect("--max-pages N").parse().expect("n"),
            "--tsv" => a.tsv = Some(it.next().expect("--tsv PATH").into()),
            "--no-raster" => a.raster = false,
            "--no-text" => a.text = false,
            "--no-objects" => a.objects = false,
            "--first-page" => a.first_page = it.next().expect("--first-page N").parse().expect("n"),
            "--dump" => a.dump = Some(it.next().expect("--dump DIR").into()),
            "--show-unmatched" => a.show_unmatched = it.next().expect("--show-unmatched N").parse().expect("n"),
            _ => a.files.push(s.into()),
        }
    }
    if a.mutool.as_os_str().is_empty() || a.files.is_empty() {
        eprintln!("usage: mupdf_oracle --mutool PATH [--dpi N] [--max-pages N] [--tsv PATH] [--no-raster|--no-text|--no-objects] FILE.pdf...");
        std::process::exit(2);
    }
    a
}

/// Per-file totals, printed as one markdown row.
#[derive(Default)]
struct FileSummary {
    pages: usize,
    obj_total: usize,
    obj_kind_mismatch: usize,
    streams: usize,
    streams_identical: usize,
    stream_diff_by_filter: HashMap<String, usize>,
    text_pages_pass: usize,
    chars_mupdf: usize,
    chars_missing: usize,
    chars_extra: usize,
    order_mismatch_pages: usize,
    max_origin_err: f32,
    raster_pages_pass: usize,
    worst_gross: f64,
    worst_gross_page: usize,
    mean_abs_sum: f64,
    fallback_glyphs: usize,
    errors: Vec<String>,
}

fn main() {
    let args = parse_args();
    let tmp = std::env::temp_dir().join(format!("kopitiam-mupdf-oracle-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("tmp dir");
    let mut tsv = String::from(
        "file\tpage\tmu_chars\tmissing\textra\torder_ok\tmax_origin_err\tgross_frac\tmean_abs\tfallback_glyphs\tours_ms\tmu_ms\n",
    );
    let mut rows = Vec::new();
    for f in &args.files {
        let name = f.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        eprintln!("== {name}");
        let s = run_file(&args, f, &tmp, &name, &mut tsv);
        rows.push((name, s));
    }
    println!(
        "| file | pages | objects kind-mismatch | streams identical | stream diffs by filter | text pages pass | chars missing / extra (of MuPDF's) | order-mismatch pages | max origin err (pt) | raster pages pass (≤1% gross) | worst gross % (page) | mean |Δluma| | fallback glyphs | errors |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for (name, s) in &rows {
        let mut filt: Vec<_> = s.stream_diff_by_filter.iter().collect();
        filt.sort();
        let filt = filt.iter().map(|(k, v)| format!("{k}:{v}")).collect::<Vec<_>>().join(" ");
        println!(
            "| {name} | {} | {}/{} | {}/{} | {} | {}/{} | {} / {} (of {}) | {} | {:.3} | {}/{} | {:.2} ({}) | {:.2} | {} | {} |",
            s.pages,
            s.obj_kind_mismatch,
            s.obj_total,
            s.streams_identical,
            s.streams,
            if filt.is_empty() { "-".into() } else { filt },
            s.text_pages_pass,
            s.pages,
            s.chars_missing,
            s.chars_extra,
            s.chars_mupdf,
            s.order_mismatch_pages,
            s.max_origin_err,
            s.raster_pages_pass,
            s.pages,
            s.worst_gross * 100.0,
            s.worst_gross_page,
            if s.pages > 0 { s.mean_abs_sum / s.pages as f64 } else { 0.0 },
            s.fallback_glyphs,
            if s.errors.is_empty() { "-".into() } else { s.errors.join("; ") },
        );
    }
    if let Some(p) = &args.tsv {
        std::fs::write(p, tsv).expect("write tsv");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

fn run_file(args: &Args, f: &Path, tmp: &Path, name: &str, tsv: &mut String) -> FileSummary {
    let mut s = FileSummary::default();
    let bytes = match std::fs::read(f) {
        Ok(b) => b,
        Err(e) => {
            s.errors.push(format!("read: {e}"));
            return s;
        }
    };
    let doc = match PdfDocument::open(bytes) {
        Ok(d) => d,
        Err(e) => {
            s.errors.push(format!("ours failed to open: {e}"));
            return s;
        }
    };
    let first = args.first_page.max(1) - 1;
    let n_pages = doc.page_count().min(first.saturating_add(args.max_pages));
    s.pages = n_pages.saturating_sub(first);

    if args.objects {
        compare_objects(args, f, tmp, &doc, &mut s);
    }

    // Per-page layers. MuPDF is asked for all pages in one process each (text,
    // raster) -- much cheaper than one spawn per page.
    let page_range = format!("{}-{n_pages}", first + 1);
    let mu_text = if args.text && n_pages > 0 {
        let out = tmp.join("t.xml");
        let t0 = Instant::now();
        let ok = Command::new(&args.mutool)
            // `mutool draw -F stext` switches on ACCURATE_BBOXES and
            // COLLECT_STYLES on top of CLIP (mudraw.c:872-878). Turn the two
            // extras back off so both engines run the SAME option set -- the
            // style-collecting fake-bold merge would otherwise drop overprinted
            // text that plain CLIP extraction (ours, and MuPDF's API default)
            // keeps, and the diff would be measuring configuration, not code.
            .args(["draw", "-q", "-F", "stext", "-O", "accurate-bboxes=no,collect-styles=no", "-o"])
            .arg(&out)
            .arg(f)
            .arg(&page_range)
            .status()
            .map(|st| st.success())
            .unwrap_or(false);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if ok {
            Some((parse_stext_xml(&std::fs::read_to_string(&out).unwrap_or_default()), ms))
        } else {
            s.errors.push("mutool stext failed".into());
            None
        }
    } else {
        None
    };
    let mu_raster_ms = if args.raster && n_pages > 0 {
        let pat = tmp.join("r%d.ppm");
        let t0 = Instant::now();
        let ok = Command::new(&args.mutool)
            // `-N` (no ICC: MuPDF's own fast colour conversions, the ones the
            // port translates) and `-M 0` (no overprint/spot simulation).
            // Both are MuPDF features the port does not have -- a CMS and
            // spot rendering -- so the oracle runs in the mode the port
            // actually targets; see docs/mupdf-code-to-code.md.
            .args(["draw", "-q", "-N", "-M", "0", "-c", "rgb", "-r"])
            .arg(format!("{}", args.dpi))
            .arg("-o")
            .arg(&pat)
            .arg(f)
            .arg(&page_range)
            .status()
            .map(|st| st.success())
            .unwrap_or(false);
        if !ok {
            s.errors.push("mutool draw failed".into());
        }
        t0.elapsed().as_secs_f64() * 1000.0 / n_pages.max(1) as f64
    } else {
        0.0
    };

    for p in first..n_pages {
        let mut row = (0usize, 0usize, 0usize, true, 0.0f32, f64::NAN, f64::NAN, 0usize, 0.0f64);
        if let Some((pages, _)) = &mu_text {
            let mu = pages.get(p - first).cloned().unwrap_or_default();
            let ours = match page_to_stext(&doc, p, StextOptions { flags: StextOptions::CLIP }) {
                Ok(pg) => pg
                    .blocks
                    .iter()
                    .filter_map(|b| match b {
                        kopitiam_pdf::mupdf::StextBlock::Text(t) => Some(t),
                        _ => None,
                    })
                    .flat_map(|t| t.lines.iter())
                    .flat_map(|l| l.chars.iter())
                    .map(|c| MuChar { c: c.c, x: c.origin.x, y: c.origin.y })
                    .collect::<Vec<_>>(),
                Err(e) => {
                    s.errors.push(format!("p{} stext: {e}", p + 1));
                    Vec::new()
                }
            };
            let m = match_chars(&mu, &ours, args.show_unmatched);
            s.chars_mupdf += m.mu_n;
            s.chars_missing += m.missing;
            s.chars_extra += m.extra;
            if !m.order_ok {
                s.order_mismatch_pages += 1;
            }
            if m.max_err > s.max_origin_err {
                s.max_origin_err = m.max_err;
            }
            if m.missing == 0 && m.extra == 0 && m.order_ok {
                s.text_pages_pass += 1;
            }
            row.0 = m.mu_n;
            row.1 = m.missing;
            row.2 = m.extra;
            row.3 = m.order_ok;
            row.4 = m.max_err;
        }
        if args.raster {
            let t0 = Instant::now();
            let ours = rasterize_page_ex(&doc, p, args.dpi);
            row.8 = t0.elapsed().as_secs_f64() * 1000.0;
            let mu = read_ppm(&tmp.join(format!("r{}.ppm", p + 1)));
            match (ours, mu) {
                (Ok((pix, fb)), Some((w, h, rgb))) => {
                    let (gross, mean) = compare_raster(&pix.samples, pix.w, pix.h, pix.n, &rgb, w, h);
                    s.fallback_glyphs += fb;
                    row.5 = gross;
                    row.6 = mean;
                    row.7 = fb;
                    s.mean_abs_sum += mean;
                    if gross <= RASTER_GROSS_PASS {
                        s.raster_pages_pass += 1;
                    }
                    // Failing pages are dumped; KOPITIAM_DUMP_ALL=1 dumps the passing ones too.
                    let dump_this = gross > RASTER_GROSS_PASS || std::env::var_os("KOPITIAM_DUMP_ALL").is_some();
                    if let Some(dir) = args.dump.as_ref().filter(|_| dump_this) {
                        let _ = std::fs::create_dir_all(dir);
                        let stem = format!("{name}-p{}", p + 1);
                        let _ = write_ppm(&dir.join(format!("{stem}.ours.ppm")), &pix.samples, pix.w, pix.h, pix.n);
                        let _ = std::fs::copy(
                            tmp.join(format!("r{}.ppm", p + 1)),
                            dir.join(format!("{stem}.mupdf.ppm")),
                        );
                    }
                    if gross > s.worst_gross {
                        s.worst_gross = gross;
                        s.worst_gross_page = p + 1;
                    }
                }
                (Err(e), _) => s.errors.push(format!("p{} render: {e}", p + 1)),
                (_, None) => s.errors.push(format!("p{} no mutool raster", p + 1)),
            }
        }
        tsv.push_str(&format!(
            "{name}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.5}\t{:.3}\t{}\t{:.1}\t{:.1}\n",
            p + 1,
            row.0,
            row.1,
            row.2,
            row.3,
            row.4,
            row.5,
            row.6,
            row.7,
            row.8,
            mu_raster_ms
        ));
        eprintln!(
            "  p{:>3}: chars {:>5} miss {:>4} extra {:>4} order {} maxerr {:.3} | gross {:.3}% mean {:.2} fb {}",
            p + 1,
            row.0,
            row.1,
            row.2,
            if row.3 { "ok" } else { "DIFF" },
            row.4,
            row.5 * 100.0,
            row.6,
            row.7
        );
    }
    s
}

fn compare_objects(args: &Args, f: &Path, tmp: &Path, doc: &PdfDocument, s: &mut FileSummary) {
    let dir = tmp.join("objs");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("objs dir");
    let js = tmp.join("objects.js");
    std::fs::write(&js, OBJECTS_JS).expect("write js");
    let out = Command::new(&args.mutool).arg("run").arg(&js).arg(f).arg(&dir).output();
    let Ok(out) = out else {
        s.errors.push("mutool run failed to start".into());
        return;
    };
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(num), Some(kind)) = (it.next(), it.next()) else { continue };
        let Ok(num) = num.parse::<i32>() else { continue };
        s.obj_total += 1;
        let ours_kind = our_kind(doc, num);
        let mu_kind = kind;
        if mu_kind == "stream" {
            s.streams += 1;
            let mu_bytes = std::fs::read(dir.join(format!("{num}.bin"))).unwrap_or_default();
            match doc.open_stream_num(num) {
                Ok(b) if b == mu_bytes => s.streams_identical += 1,
                other => {
                    let filt = last_filter(doc, num);
                    let tag = match other {
                        Ok(_) => format!("{filt}(bytes)"),
                        Err(_) => format!("{filt}(err)"),
                    };
                    *s.stream_diff_by_filter.entry(tag).or_default() += 1;
                }
            }
        } else if ours_kind != mu_kind {
            // MuPDF resolves a free / missing object to null; count ours
            // failing on it as a mismatch too -- that IS a behaviour gap.
            s.obj_kind_mismatch += 1;
            if s.obj_kind_mismatch <= 5 {
                eprintln!("  obj {num}: mupdf {mu_kind} ours {ours_kind}");
            }
        }
    }
}

fn our_kind(doc: &PdfDocument, num: i32) -> &'static str {
    if doc.stream_raw_num(num).is_ok() {
        return "stream";
    }
    match doc.resolve(&Object::Ref { num, generation: 0 }) {
        Ok(o) => match o {
            Object::Null => "null",
            Object::Bool(_) => "bool",
            Object::Int(_) => "int",
            Object::Real(_) => "real",
            Object::String(_) => "string",
            Object::Name(_) => "name",
            Object::Array(_) => "array",
            Object::Dict(_) => "dict",
            _ => "?",
        },
        Err(_) => "err",
    }
}

fn last_filter(doc: &PdfDocument, num: i32) -> String {
    let Ok((_, filter, _)) = doc.stream_raw_num(num) else { return "?".into() };
    let f = doc.resolve(&filter).unwrap_or(Object::Null);
    let last = if f.is_array() {
        f.array_get(f.array_len().saturating_sub(1)).cloned().unwrap_or(Object::Null)
    } else {
        f
    };
    let last = doc.resolve(&last).unwrap_or(Object::Null);
    if last.is_name() {
        String::from_utf8_lossy(last.to_name()).into_owned()
    } else {
        "none".into()
    }
}

#[derive(Clone, Default, Debug)]
struct MuChar {
    c: char,
    x: f32,
    y: f32,
}

/// Parse `mutool draw -F stext` XML into per-page char lists (origin = the
/// `x`/`y` attributes, which are `fz_stext_char.origin`).
fn parse_stext_xml(xml: &str) -> Vec<Vec<MuChar>> {
    let mut pages = Vec::new();
    for page in xml.split("<page ").skip(1) {
        let mut chars = Vec::new();
        for ch in page.split("<char ").skip(1) {
            let attr = |k: &str| -> Option<&str> {
                let pat = format!(" {k}=\"");
                let at = ch.find(&pat).or_else(|| {
                    // `c` is the first attribute, with no leading space.
                    if ch.starts_with(&pat[1..]) { Some(0) } else { None }
                })?;
                let start = at + ch[at..].find('"')? + 1;
                let end = start + ch[start..].find('"')?;
                Some(&ch[start..end])
            };
            let (Some(c), Some(x), Some(y)) = (attr("c"), attr("x"), attr("y")) else { continue };
            let c = unescape_xml(c);
            let Some(c) = c.chars().next() else { continue };
            chars.push(MuChar { c, x: x.parse().unwrap_or(0.0), y: y.parse().unwrap_or(0.0) });
        }
        pages.push(chars);
    }
    pages
}

fn unescape_xml(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        let Some(j) = tail.find(';') else {
            out.push_str(tail);
            return out;
        };
        let ent = &tail[1..j];
        let ch = match ent {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            e if e.starts_with("#x") => u32::from_str_radix(&e[2..], 16).ok().and_then(char::from_u32),
            e if e.starts_with('#') => e[1..].parse().ok().and_then(char::from_u32),
            _ => None,
        };
        match ch {
            Some(c) => out.push(c),
            None => out.push_str(&tail[..=j]),
        }
        rest = &tail[j + 1..];
    }
    out.push_str(rest);
    out
}

struct CharMatch {
    mu_n: usize,
    missing: usize,
    extra: usize,
    order_ok: bool,
    max_err: f32,
}

fn match_chars(mu: &[MuChar], ours: &[MuChar], show: usize) -> CharMatch {
    let mu: Vec<&MuChar> = mu.iter().filter(|c| !c.c.is_whitespace()).collect();
    let ours: Vec<&MuChar> = ours.iter().filter(|c| !c.c.is_whitespace()).collect();
    let mut by_char: HashMap<char, Vec<usize>> = HashMap::new();
    for (i, c) in ours.iter().enumerate() {
        by_char.entry(c.c).or_default().push(i);
    }
    let mut used = vec![false; ours.len()];
    let mut missing = 0;
    let mut max_err = 0.0f32;
    for m in &mu {
        let mut best: Option<(usize, f32)> = None;
        if let Some(cands) = by_char.get(&m.c) {
            for &i in cands {
                if used[i] {
                    continue;
                }
                let d = ((ours[i].x - m.x).powi(2) + (ours[i].y - m.y).powi(2)).sqrt();
                if d <= MATCH_RADIUS_PT && best.is_none_or(|(_, bd)| d < bd) {
                    best = Some((i, d));
                }
            }
        }
        match best {
            Some((i, d)) => {
                used[i] = true;
                max_err = max_err.max(d);
            }
            None => {
                if missing < show {
                    eprintln!("    missing {:?} at ({:.2},{:.2})", m.c, m.x, m.y);
                }
                missing += 1
            }
        }
    }
    let extra = used.iter().filter(|u| !**u).count();
    for (i, _) in used.iter().enumerate().filter(|(_, u)| !**u).take(show) {
        eprintln!("    extra   {:?} at ({:.2},{:.2})", ours[i].c, ours[i].x, ours[i].y);
    }
    let a: String = mu.iter().map(|c| c.c).collect();
    let b: String = ours.iter().map(|c| c.c).collect();
    CharMatch { mu_n: mu.len(), missing, extra, order_ok: a == b, max_err }
}

/// Read a binary PPM (`P6`, maxval 255) as written by `mutool draw -c rgb`.
fn read_ppm(p: &Path) -> Option<(u32, u32, Vec<u8>)> {
    let b = std::fs::read(p).ok()?;
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 && i < b.len() {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < b.len() && b[i] == b'#' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let st = i;
        while i < b.len() && !b[i].is_ascii_whitespace() {
            i += 1;
        }
        fields.push(String::from_utf8_lossy(&b[st..i]).into_owned());
    }
    if fields.len() < 4 || fields[0] != "P6" {
        return None;
    }
    let w: u32 = fields[1].parse().ok()?;
    let h: u32 = fields[2].parse().ok()?;
    let data = b.get(i + 1..)?.to_vec();
    (data.len() >= (w * h * 3) as usize).then_some((w, h, data))
}

/// Gross-mismatch fraction and mean |Δluma| over the union of both rasters
/// (a size mismatch counts the non-overlapping strip as paper on the smaller
/// side, so an off-by-one page size costs a line of pixels, not a crash).
///
/// A pixel is gross when ANY of R, G, B differs by more than [`GROSS_LUMA`].
/// (Amended 2026-09-28, before the feature-corpus measurements: the first
/// version compared luma only, and luma hides hue errors -- pure red and pure
/// green differ by just 74 in luma, so a red square painted where MuPDF
/// paints green passed. Every result in docs/mupdf-code-to-code.md from the
/// "feature corpus" section on uses this per-channel rule.)
fn compare_raster(ours: &[u8], ow: u32, oh: u32, on: u8, mu: &[u8], mw: u32, mh: u32) -> (f64, f64) {
    let w = ow.max(mw) as usize;
    let h = oh.max(mh) as usize;
    let rgb_ours = |x: usize, y: usize| -> [i32; 3] {
        if x >= ow as usize || y >= oh as usize {
            return [255; 3];
        }
        let i = (y * ow as usize + x) * on as usize;
        match on {
            1 | 2 => [ours[i] as i32; 3],
            _ => [ours[i] as i32, ours[i + 1] as i32, ours[i + 2] as i32],
        }
    };
    let rgb_mu = |x: usize, y: usize| -> [i32; 3] {
        if x >= mw as usize || y >= mh as usize {
            return [255; 3];
        }
        let i = (y * mw as usize + x) * 3;
        [mu[i] as i32, mu[i + 1] as i32, mu[i + 2] as i32]
    };
    let luma = |c: [i32; 3]| (c[0] * 77 + c[1] * 151 + c[2] * 28) >> 8;
    let mut gross = 0u64;
    let mut sum = 0u64;
    for y in 0..h {
        for x in 0..w {
            let a = rgb_ours(x, y);
            let b = rgb_mu(x, y);
            sum += (luma(a) - luma(b)).unsigned_abs() as u64;
            if (0..3).any(|k| (a[k] - b[k]).abs() > GROSS_LUMA) {
                gross += 1;
            }
        }
    }
    let n = (w * h).max(1) as f64;
    (gross as f64 / n, sum as f64 / n)
}

/// Write our pixmap as a binary PPM (gray / RGB / RGBA all flattened to RGB).
fn write_ppm(p: &Path, samples: &[u8], w: u32, h: u32, n: u8) -> std::io::Result<()> {
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    for px in samples.chunks(n as usize) {
        match n {
            1 | 2 => out.extend_from_slice(&[px[0], px[0], px[0]]),
            _ => out.extend_from_slice(&px[..3]),
        }
    }
    std::fs::write(p, out)
}
