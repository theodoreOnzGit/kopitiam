//! **Code-to-code check of the `CCITTFaxDecode` port against MuPDF.**
//!
//! Decodes every CCITT-coded image in a PDF with this crate's
//! [`filter_fax`](kopitiam_pdf::mupdf::filter_fax) and writes each as a PBM,
//! so the result can be compared byte-for-byte against the same image decoded
//! by the code it was ported from:
//!
//! ```sh
//! cargo run --example fax_verify -- scan.pdf out/
//! mutool extract -o ref/ scan.pdf        # MuPDF's own decode
//! # or, for an independent third opinion:
//! pdfimages scan.pdf ref/p               # poppler
//! cmp out/img-000.pbm ref/...
//! ```
//!
//! The PDF is a runtime argument rather than a fixture because the documents
//! this was written for are scanned reports under restrictive licences; the
//! harness is reusable, the corpus is not redistributable.

use std::path::PathBuf;

use kopitiam_pdf::mupdf::filter_fax::{decode, FaxParams};
use kopitiam_pdf::mupdf::object::Object;
use kopitiam_pdf::mupdf::xref::PdfDocument;

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(pdf), Some(outdir)) = (args.next(), args.next()) else {
        eprintln!("usage: fax_verify <input.pdf> <outdir>");
        std::process::exit(2);
    };
    let outdir = PathBuf::from(outdir);
    std::fs::create_dir_all(&outdir).expect("create outdir");

    let bytes = std::fs::read(&pdf).expect("read pdf");
    let doc = PdfDocument::open(bytes).expect("open pdf");
    let mut found = 0usize;

    // Walk every object; the CCITT images are whichever streams name the
    // filter. Cheaper than resolving the page tree and it catches images
    // referenced from anywhere, including form XObjects.
    for num in 1..20_000 {
        let Ok((raw, filter, parms)) = doc.stream_raw_num(num as i32) else {
            continue;
        };
        if !mentions_ccitt(&doc, &filter) {
            continue;
        }
        // /Rows is optional in DecodeParms; the image's /Height is the
        // real row count and is what MuPDF's image loader passes in.
        let obj_ref = Object::Ref {
            num: num as i32,
            generation: 0,
        };
        let height = match doc.resolve(&obj_ref) {
            Ok(d) => match doc.resolve_get(&d, "Height") {
                Ok(Object::Int(v)) => v as u32,
                _ => 0,
            },
            Err(_) => 0,
        };
        let mut params = params_from(&doc, &parms);
        if params.rows == 0 {
            params.rows = height;
        }
        match decode(&raw, &params) {
            Ok((bits, rows)) => {
                let w = params.columns as usize;
                let path = outdir.join(format!("obj-{num:04}.pbm"));
                write_pbm(&path, &bits, w, rows);
                println!(
                    "obj {num}: {w}x{rows}  K={}  align={}  -> {}",
                    params.k,
                    params.encoded_byte_align,
                    path.display()
                );
                found += 1;
            }
            Err(e) => println!("obj {num}: DECODE FAILED: {e}"),
        }
    }
    println!("{found} CCITT image(s) decoded");
}

fn mentions_ccitt(doc: &PdfDocument, filter: &Object) -> bool {
    fn is_ccitt(n: &[u8]) -> bool {
        n == b"CCITTFaxDecode" || n == b"CCF"
    }
    match doc.resolve(filter) {
        Ok(Object::Name(n)) => is_ccitt(&n),
        Ok(Object::Array(a)) => a
            .iter()
            .any(|o| matches!(doc.resolve(o), Ok(Object::Name(ref n)) if is_ccitt(&n))),
        _ => false,
    }
}

fn params_from(doc: &PdfDocument, parms: &Object) -> FaxParams {
    let d = match doc.resolve(parms) {
        Ok(Object::Array(a)) => a
            .iter()
            .find_map(|o| match doc.resolve(o) {
                Ok(v @ Object::Dict(_)) => Some(v),
                _ => None,
            })
            .unwrap_or(Object::Null),
        Ok(v) => v,
        _ => Object::Null,
    };
    let int = |k: &str, dflt: i64| match doc.resolve_get(&d, k) {
        Ok(Object::Int(v)) => v,
        Ok(Object::Real(v)) => v as i64,
        _ => dflt,
    };
    let flag = |k: &str| matches!(doc.resolve_get(&d, k), Ok(Object::Bool(true)));
    FaxParams {
        k: int("K", 0) as i32,
        end_of_line: flag("EndOfLine"),
        encoded_byte_align: flag("EncodedByteAlign"),
        // PDF §7.4.6 defaults.
        columns: int("Columns", 1728) as u32,
        rows: int("Rows", 0) as u32,
        black_is_1: flag("BlackIs1"),
    }
}

/// P4 (binary PBM): 1 = black, MSB first — the same convention the decoder
/// emits, so this is a straight copy of the packed rows.
fn write_pbm(path: &std::path::Path, bits: &[u8], w: usize, h: usize) {
    let mut out = format!("P4\n{w} {h}\n").into_bytes();
    out.extend_from_slice(bits);
    std::fs::write(path, out).expect("write pbm");
}
