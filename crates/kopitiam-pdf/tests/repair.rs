//! Broken-file repair (`pdf-repair.c`, ported in `src/mupdf/repair.rs`) and
//! the endstream filter, checked end to end: open the damaged file, then look
//! at what the page *shows* -- its text through `page_to_stext` and a red
//! square through `rasterize_page_native`.
//!
//! Every fixture is synthetic, built right here from nothing (the encrypted
//! cases damage the crate's own MuPDF-made `encrypted-rc4-gen1.pdf`). The
//! expected behaviour of each one was checked against real MuPDF (19f1284)
//! with `mutool draw -F txt`, `mutool draw -r 72 -c rgb` and
//! `mutool show FILE trailer` on byte-identical files, 2026-09-28:
//!
//! | case | mutool |
//! |---|---|
//! | xref offsets all +7 | "repairing PDF document", text + red square |
//! | no xref, no startxref | repairs, trailer `/Size 5 /Root 1 0 R` |
//! | truncated inside the last stream | repairs, `/Info 5 0 R /Root 1 0 R` |
//! | truncated inside the last dict / right after `obj` | **refuses** ("invalid key in dict" / "truncated object") |
//! | `/Length` +40, −30, 99999999, flate −10 (good xref) | draws; "PDF stream Length incorrect" for the short/huge ones |
//! | object 4 redefined later | the NEW content |
//! | objects only in an ObjStm, xref stream garbage | repairs, 1–3 become `o` entries in stream 5 |
//! | no trailer, catalog is object 5 | `/Root 5 0 R /Info 1 0 R` |
//! | encrypted RC4, xref removed / shifted | repairs, red square |
//!
//! Singlish note, hor: "opens" alone proves nothing -- a repair that finds the
//! wrong objects still opens. So every test also checks the content.

use kopitiam_pdf::mupdf::structured_text::StextOptions;
use kopitiam_pdf::mupdf::{PdfDocument, Pixmap, page_to_stext, rasterize_page_native};

const FONT: &str = "<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>";

/// The content every fixture page draws: a red square over (50,50)-(150,150)
/// and one word of Helvetica.
fn content(word: &str) -> Vec<u8> {
    format!("1 0 0 rg 50 50 100 100 re f BT /F1 12 Tf 20 170 Td ({word}) Tj ET").into_bytes()
}

fn stream_obj(dict_extra: &str, length: usize, data: &[u8]) -> Vec<u8> {
    let mut o = format!("<< /Length {length}{dict_extra} >>\nstream\n").into_bytes();
    o.extend_from_slice(data);
    o.extend_from_slice(b"\nendstream");
    o
}

/// Objects 1..4: catalog, pages, page, content stream (with a right /Length).
fn body(word: &str) -> Vec<Vec<u8>> {
    let c = content(word);
    vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources {FONT} /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_obj("", c.len(), &c),
    ]
}

/// `N 0 obj … endobj` for each body, numbered from `start`; returns the bytes
/// and each object's offset within them.
fn objs(bodies: &[Vec<u8>], start: usize) -> (Vec<u8>, Vec<usize>) {
    let mut out = Vec::new();
    let mut offs = Vec::new();
    for (i, b) in bodies.iter().enumerate() {
        offs.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", i + start).as_bytes());
        out.extend_from_slice(b);
        out.extend_from_slice(b"\nendobj\n");
    }
    (out, offs)
}

/// A complete file with a classic xref whose offsets are all off by `shift`.
fn full(bodies: &[Vec<u8>], shift: i64) -> Vec<u8> {
    let head = b"%PDF-1.7\n";
    let (o, offs) = objs(bodies, 1);
    let mut s = head.to_vec();
    s.extend_from_slice(&o);
    let x = s.len();
    s.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", bodies.len() + 1).as_bytes());
    for f in offs {
        s.extend_from_slice(format!("{:010} 00000 n \n", (f + head.len()) as i64 + shift).as_bytes());
    }
    s.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF\n",
            bodies.len() + 1
        )
        .as_bytes(),
    );
    s
}

fn text_of(doc: &PdfDocument) -> String {
    page_to_stext(doc, 0, StextOptions::default())
        .expect("stext")
        .text()
}

fn rgb_at(pix: &Pixmap, x: u32, y: u32) -> [u8; 3] {
    let o = ((y * pix.w + x) * pix.n as u32) as usize;
    [pix.samples[o], pix.samples[o + 1], pix.samples[o + 2]]
}

/// The page shows `word` and the red square, and nothing red outside it.
fn assert_page(doc: &PdfDocument, word: &str) {
    assert_eq!(doc.page_count(), 1);
    let text = text_of(doc);
    assert!(text.contains(word), "expected {word:?} on the page, got {text:?}");
    let pix = rasterize_page_native(doc, 0, 72.0).expect("renders");
    assert_eq!(rgb_at(&pix, 100, 100), [255, 0, 0], "red square");
    assert_eq!(rgb_at(&pix, 10, 10), [255, 255, 255], "white margin");
}

fn open(bytes: Vec<u8>) -> PdfDocument {
    PdfDocument::open(bytes).expect("MuPDF repairs and opens this file, so must we")
}

/// (a) Every xref offset shifted by 7 bytes -- the harness's
/// `feat-broken-xref` case. The xref loads (offsets are in range), so it is
/// the first object lookup that finds `<<` where `1 0 obj` should be and
/// repairs: pdf_cache_object's path, not pdf_init_document's.
#[test]
fn shifted_xref_offsets_repair_on_first_lookup() {
    let doc = open(full(&body("Shifted"), 7));
    assert!(doc.was_repaired());
    assert_page(&doc, "Shifted");
    assert_eq!(doc.trailer().dict_gets("Root").unwrap().to_num(), 1);
}

/// The same file with correct offsets is NOT repaired -- repair is a
/// fallback, not a second opinion.
#[test]
fn a_good_file_is_not_repaired() {
    let doc = open(full(&body("Fine"), 0));
    assert!(!doc.was_repaired());
    assert_page(&doc, "Fine");
}

/// (b) No xref table and no startxref: pdf_init_document's repair. The
/// trailer dict (after a bare `trailer` keyword) still gives /Root.
#[test]
fn no_xref_and_no_startxref() {
    let (o, _) = objs(&body("Noxref"), 1);
    let mut s = b"%PDF-1.7\n".to_vec();
    s.extend_from_slice(&o);
    s.extend_from_slice(b"trailer\n<< /Size 5 /Root 1 0 R >>\n%%EOF\n");
    let doc = open(s);
    assert!(doc.was_repaired());
    assert_page(&doc, "Noxref");
    assert_eq!(doc.trailer().dict_gets("Size").unwrap().to_int(), 5);
}

/// The truncated-file fixtures: objects 1-4 whole, 5 an Info-like dict, 6 a
/// stream that the file will be cut inside.
fn truncatable() -> Vec<u8> {
    let mut b = body("Truncated");
    b.push(b"<< /Producer (x) >>".to_vec());
    b.push(b"<< /Length 100 >>\nstream\nabcdefgh".to_vec());
    let (o, _) = objs(&b, 1);
    let mut s = b"%PDF-1.7\n".to_vec();
    s.extend_from_slice(&o);
    s
}

fn position(hay: &[u8], needle: &[u8]) -> usize {
    hay.windows(needle.len()).position(|w| w == needle).unwrap()
}

/// (c) Truncated inside the last stream: trailer, xref and the stream's end
/// are all gone. The scan runs the stream to EOF, and with no trailer the
/// catalog and the Info dict are found by pdf_repair_trailer's backwards walk
/// -- mutool shows exactly `/Size 7 /Info 5 0 R /Root 1 0 R`.
#[test]
fn truncated_inside_the_last_stream() {
    let t = truncatable();
    let cut = position(&t, b"abcdefgh") + 4;
    let doc = open(t[..cut].to_vec());
    assert_page(&doc, "Truncated");
    let trailer = doc.trailer();
    assert_eq!(trailer.dict_gets("Size").unwrap().to_int(), 7);
    assert_eq!(trailer.dict_gets("Root").unwrap().to_num(), 1);
    assert_eq!(trailer.dict_gets("Info").unwrap().to_num(), 5);
}

/// …but cut inside a *dict* (or right after `obj`), before any trailer gave
/// a /Root, MuPDF gives up ("If we haven't seen a root yet, there is nothing
/// we can do") -- mutool refuses both files. So do we: an honest refusal,
/// same as upstream, rather than a guess.
#[test]
fn truncated_inside_a_dict_before_any_root_is_refused_like_mupdf() {
    let t = truncatable();
    let at = position(&t, b"6 0 obj");
    assert!(PdfDocument::open(t[..at + 14].to_vec()).is_err(), "cut inside the dict");
    assert!(PdfDocument::open(t[..at + 7].to_vec()).is_err(), "cut right after `obj`");
}

/// (d, good xref) A /Length that is too LONG: MuPDF reads it as given and
/// then on to the next `endstream` (here: none, so EOF). The content stream
/// gains some junk operators after its real ops, and still draws.
#[test]
fn too_long_length_with_a_good_xref() {
    let c = content("Longlen");
    let mut b = body("x");
    b[3] = stream_obj("", c.len() + 40, &c);
    let doc = open(full(&b, 0));
    assert!(!doc.was_repaired());
    assert_page(&doc, "Longlen");
}

/// (d, good xref) Too SHORT, and absurdly large (clamped to 0 by
/// pdf_stream_length): the endstream filter reads on to `endstream`. Before
/// the port the short one lost its last 30 bytes and the huge one read as
/// empty.
#[test]
fn too_short_or_absurd_length_with_a_good_xref() {
    let c = content("Shortlen");
    let mut b = body("x");
    b[3] = stream_obj("", c.len() - 30, &c);
    let doc = open(full(&b, 0));
    assert!(!doc.was_repaired());
    assert_page(&doc, "Shortlen");
    let raw = doc.open_stream(&kopitiam_pdf::mupdf::Object::new_indirect(4, 0)).unwrap();
    assert_eq!(raw, c, "exactly the body, the EOL before endstream stripped");

    let mut b = body("x");
    let mut o = b"<< /Length 99999999 >>\nstream\n".to_vec();
    o.extend_from_slice(&c);
    o.extend_from_slice(b"\nendstream");
    b[3] = o;
    assert_page(&open(full(&b, 0)), "Shortlen");
}

/// (d, good xref) A FlateDecode stream 10 bytes short: without the endstream
/// filter the inflater would see a truncated deflate stream.
#[test]
fn flate_stream_with_short_length() {
    let z = deflate(&content("Flatelen"));
    let mut b = body("x");
    b[3] = stream_obj(" /Filter /FlateDecode", z.len() - 10, &z);
    assert_page(&open(full(&b, 0)), "Flatelen");
}

/// (d, repair path) Wrong /Length in a file that also needs repair: the scan
/// measures the stream by its `endstream` and puts that length on the object
/// (pdf_repair_xref_base's "correct stream length"). MuPDF measures up to the
/// `e` of `endstream`, so the EOL before it counts: `mutool show` gives
/// `/Length 67` for the 66-byte "Longlen" body. We match that, one byte and
/// all.
#[test]
fn wrong_lengths_in_a_repaired_file() {
    let z = deflate(&content("Flatelen"));
    let mut b = body("x");
    b[3] = stream_obj(" /Filter /FlateDecode", z.len() - 10, &z);
    let doc = open(full(&b, 3));
    assert!(doc.was_repaired());
    assert_page(&doc, "Flatelen");
    let four = doc.resolve(&kopitiam_pdf::mupdf::Object::new_indirect(4, 0)).unwrap();
    assert_eq!(four.dict_gets("Length").unwrap().to_int(), z.len() as i64 + 1, "corrected");

    let c = content("Longlen");
    let mut b = body("x");
    b[3] = stream_obj("", c.len() + 40, &c);
    let doc = open(full(&b, 3));
    assert_page(&doc, "Longlen");
    let four = doc.resolve(&kopitiam_pdf::mupdf::Object::new_indirect(4, 0)).unwrap();
    assert_eq!(c.len(), 66);
    assert_eq!(four.dict_gets("Length").unwrap().to_int(), 67, "mutool: /Length 67");
}

/// (e) Object 4 defined twice, no xref: the LAST definition wins.
#[test]
fn a_later_definition_wins() {
    let (o1, _) = objs(&body("Old"), 1);
    let c = content("New");
    let mut s = b"%PDF-1.7\n".to_vec();
    s.extend_from_slice(&o1);
    s.extend_from_slice(b"4 0 obj\n");
    s.extend_from_slice(&stream_obj("", c.len(), &c));
    s.extend_from_slice(b"\nendobj\n");
    s.extend_from_slice(b"trailer\n<< /Size 5 /Root 1 0 R >>\n%%EOF\n");
    let doc = open(s);
    assert_page(&doc, "New");
    assert!(!text_of(&doc).contains("Old"));
}

/// (f) Catalog, pages and page live only inside object stream 5, and the
/// xref stream (object 6) is garbage. The scan finds 4, 5, 6 as plain
/// objects; the xref stream's own dict supplies /Root; pdf_repair_obj_stms
/// then adds 1-3 as compressed entries of 5 -- mutool's repaired xref shows
/// `1-3: 5 o`.
#[test]
fn objects_only_inside_an_object_stream_with_a_broken_xref_stream() {
    let b = body("Objstm");
    let mut hdr = String::new();
    let mut data = Vec::new();
    for (i, o) in b.iter().take(3).enumerate() {
        hdr.push_str(&format!("{} {} ", i + 1, data.len()));
        data.extend_from_slice(o);
        data.push(b' ');
    }
    let mut osb = hdr.clone().into_bytes();
    osb.extend_from_slice(&data);

    let mut s = b"%PDF-1.7\n4 0 obj\n".to_vec();
    s.extend_from_slice(&b[3]);
    s.extend_from_slice(b"\nendobj\n5 0 obj\n");
    s.extend_from_slice(&stream_obj(
        &format!(" /Type /ObjStm /N 3 /First {}", hdr.len()),
        osb.len(),
        &osb,
    ));
    s.extend_from_slice(b"\nendobj\n");
    let x = s.len();
    let garbage = b"\x07\x01\x02garbagegarbage";
    s.extend_from_slice(b"6 0 obj\n");
    s.extend_from_slice(&stream_obj(
        " /Type /XRef /Size 7 /Root 1 0 R /W [1 4 2] /Filter /FlateDecode",
        garbage.len(),
        garbage,
    ));
    s.extend_from_slice(format!("\nendobj\nstartxref\n{x}\n%%EOF\n").as_bytes());

    let doc = open(s);
    assert!(doc.was_repaired());
    assert_page(&doc, "Objstm");
    assert_eq!(doc.trailer().dict_gets("Root").unwrap().to_num(), 1);
    assert_eq!(doc.xref_len(), 7);
}

/// (g) No trailer anywhere, no xref: /Root is found by pdf_repair_trailer's
/// backwards walk for a direct `/Type /Catalog` (object 5 here), and /Info
/// by the first object with /Producer -- mutool: `/Root 5 0 R /Info 1 0 R`.
#[test]
fn trailerless_file_finds_the_catalog_by_scanning() {
    let mut b = body("Catalogscan");
    b[0] = b"<< /Producer (none) >>".to_vec();
    b.push(b"<< /Type /Catalog /Pages 2 0 R >>".to_vec());
    let (o, _) = objs(&b, 1);
    let mut s = b"%PDF-1.7\n".to_vec();
    s.extend_from_slice(&o);
    s.extend_from_slice(b"%%EOF\n");
    let doc = open(s);
    assert_page(&doc, "Catalogscan");
    assert_eq!(doc.trailer().dict_gets("Root").unwrap().to_num(), 5);
    assert_eq!(doc.trailer().dict_gets("Info").unwrap().to_num(), 1);
}

/// Junk that holds no object at all: repair's "no objects found" -- an
/// error, never a panic.
#[test]
fn a_file_with_no_objects_is_an_error() {
    assert!(PdfDocument::open(b"%PDF-1.7\nnothing to see here\n%%EOF\n".to_vec()).is_err());
    assert!(PdfDocument::open(Vec::new()).is_err());
}

/// The RC4 fixture made by MuPDF (tests/fixtures/make-encrypted-gen1.py),
/// damaged two ways. Repair must still find /Encrypt (a direct dict in the
/// trailer) and /ID, so the file decrypts and the red square appears --
/// mutool renders both damaged files that way.
#[test]
fn a_damaged_encrypted_file_still_decrypts() {
    let good = include_bytes!("fixtures/encrypted-rc4-gen1.pdf").to_vec();
    let x = position(&good, b"xref\n");
    let t = position(&good, b"trailer");
    let sx = position(&good, b"startxref");

    // Cross-reference and startxref cut out; the trailer dict stays.
    let mut noxref = good[..x].to_vec();
    noxref.extend_from_slice(&good[t..sx]);
    noxref.extend_from_slice(b"%%EOF\n");

    // Every in-use offset shifted by 5.
    let mut shifted = good[..x].to_vec();
    for line in good[x..t].split(|&b| b == b'\n') {
        let l = std::str::from_utf8(line).unwrap();
        if l.len() >= 18 && l.ends_with(" n ") && &l[..10] != "0000000000" {
            let ofs: u64 = l[..10].parse().unwrap();
            shifted.extend_from_slice(format!("{:010}{}", ofs + 5, &l[10..]).as_bytes());
        } else {
            shifted.extend_from_slice(line);
        }
        shifted.push(b'\n');
    }
    shifted.pop();
    shifted.extend_from_slice(&good[t..]);

    for (name, bytes) in [("no xref", noxref), ("shifted", shifted)] {
        let doc = PdfDocument::open(bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(doc.was_decrypted(), "{name}");
        assert!(doc.was_repaired(), "{name}");
        let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
        assert_eq!(rgb_at(&pix, 50, 50), [255, 0, 0], "{name}: decrypted red square");
    }
}

/// After a repair, an object number the scan never saw is simply null --
/// MuPDF's `pdf_resolve_indirect` of an out-of-table number -- not an error
/// and not a second rescan (repair is spent: `repair_attempted`).
#[test]
fn an_unknown_object_after_repair_is_null() {
    let doc = open(full(&body("Once"), 7));
    assert!(doc.was_repaired());
    assert_page(&doc, "Once");
    // Object 99 was never defined anywhere: null, not an error.
    let o = doc.resolve(&kopitiam_pdf::mupdf::Object::new_indirect(99, 0)).unwrap();
    assert!(o.is_null());
}

/// Hostile bytes must never panic: every prefix of a damaged file either
/// opens or returns an error.
#[test]
fn every_prefix_of_a_damaged_file_is_panic_free() {
    let mut b = body("Prefix");
    b.push(b"<< /Type /ObjStm /N 99999999 /First 5 /Length 3 >>\nstream\n1 2\nendstream".to_vec());
    let s = full(&b, 11);
    for cut in (0..s.len()).step_by(7) {
        if let Ok(doc) = PdfDocument::open(s[..cut].to_vec())
            && doc.page_count() > 0
        {
            let _ = page_to_stext(&doc, 0, StextOptions::default());
        }
    }
}

/// Minimal zlib stream (stored blocks) so the test needs no compressor.
fn deflate(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let mut chunks = data.chunks(65535).peekable();
    if chunks.peek().is_none() {
        out.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
    }
    while let Some(chunk) = chunks.next() {
        let last = chunks.peek().is_none();
        let len = chunk.len() as u16;
        out.push(last as u8);
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    out.extend_from_slice(&((b << 16) | a).to_be_bytes());
    out
}
