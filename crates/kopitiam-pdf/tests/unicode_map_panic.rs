//! Regression test for the "missing unicode map and encoding" abort.
//!
//! `pdf-extract` 0.12.0 does not return an error for certain malformed fonts
//! -- it `panic!`s outright, deep inside content-stream decoding
//! (`decode_char` -> `expect("missing unicode map and encoding")`). That panic
//! unwinds straight past every `Result`/`?`/`thiserror`/`anyhow` boundary, so
//! before the fix a single such PDF aborted the whole `kopitiam pdf2md`
//! process (and, in a batch, every remaining PDF with it).
//!
//! The exact trigger, read off `pdf-extract`'s source, is a font that has:
//!   * a `/ToUnicode` CMap that is *present but incomplete* -- it maps some
//!     character code, but NOT the code the page actually draws (so
//!     `unicode_map` is `Some(..)` yet `.get(code)` returns `None`); and
//!   * no `/Encoding`, and a non-core `BaseFont`, so `pdf-extract` builds no
//!     fallback encoding table (`encoding` stays `None`).
//!
//! With both conditions met, `decode_char` reaches
//! `self.encoding.as_ref()...expect("missing unicode map and encoding")` and
//! panics. (Note: a font with *no* `/ToUnicode` at all does NOT panic -- it
//! falls back to `PDFDocEncoding` -- which is why the fixture must supply an
//! incomplete `/ToUnicode`, not omit it.)
//!
//! As with `font_resolution.rs`, the fixture is built in-process with `lopdf`
//! (already a dependency) so the exact bytes under test stay readable.

use lopdf::{Document, Object, Stream, content::Content, content::Operation, dictionary};

/// Minimal `/ToUnicode` CMap that maps ONLY code `0x01` (to U+0058 'X') and
/// nothing else. The page below draws code `0x41` ('A'), which is absent from
/// this map -- exactly the "present but incomplete unicode map" condition.
fn incomplete_to_unicode_cmap() -> Vec<u8> {
    b"/CIDInit /ProcSet findresource begin\n\
      12 dict begin\n\
      begincmap\n\
      /CMapName /Adobe-Identity-UCS def\n\
      /CMapType 2 def\n\
      1 begincodespacerange\n\
      <00> <ff>\n\
      endcodespacerange\n\
      1 beginbfchar\n\
      <01> <0058>\n\
      endbfchar\n\
      endcmap\n\
      CMapName currentdict /CMap defineresource pop\n\
      end\n\
      end"
    .to_vec()
}

/// Build a single-page PDF whose one font triggers `pdf-extract`'s
/// `missing unicode map and encoding` panic (see module docs). The `Tj`
/// string draws byte `0x41`, which the font's incomplete `/ToUnicode` does
/// not cover, while the font carries no `/Encoding` and a non-core
/// `BaseFont`.
fn build_panicking_pdf() -> Vec<u8> {
    let mut doc = Document::with_version("1.5");

    let to_unicode = doc.add_object(Stream::new(dictionary! {}, incomplete_to_unicode_cmap()));

    // Type1, NON-core BaseFont, NO /Encoding, and an incomplete /ToUnicode.
    let font = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "ABCDEF+NoMappingFont",
        "ToUnicode" => to_unicode,
    });

    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! { "F1" => font },
    });

    let content = Content {
        operations: vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 24.into()]),
            Operation::new("Td", vec![72.into(), 700.into()]),
            // A single-byte string 0x41 ('A'): a code the /ToUnicode map lacks.
            Operation::new("Tj", vec![Object::string_literal("A")]),
            Operation::new("ET", vec![]),
        ],
    };
    let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));

    let pages_id = doc.new_object_id();
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Contents" => content_id,
    });
    let pages = dictionary! {
        "Type" => "Pages",
        "Kids" => vec![Object::Reference(page_id)],
        "Count" => 1,
        "Resources" => resources_id,
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages));

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);

    let mut bytes = Vec::new();
    doc.save_to(&mut bytes)
        .expect("saving the fixture PDF must succeed");
    bytes
}

/// The whole point: extraction must return a normal `Err` (our new
/// `UnsupportedFont` variant) rather than letting the `pdf-extract` panic
/// unwind and abort the test process. If the fix regressed, this test would
/// abort with `missing unicode map and encoding` instead of failing an
/// assertion.
#[test]
fn unmapped_font_returns_error_instead_of_aborting() {
    let bytes = build_panicking_pdf();
    let result = kopitiam_pdf::extract_from_bytes(&bytes);
    match result {
        Err(kopitiam_pdf::ExtractError::UnsupportedFont(msg)) => {
            // The recovered panic message should be the real one from
            // pdf-extract, proving we captured the payload rather than
            // inventing a placeholder.
            assert!(
                msg.contains("missing unicode map and encoding"),
                "expected the recovered pdf-extract panic message, got: {msg:?}"
            );
        }
        other => panic!("expected Err(ExtractError::UnsupportedFont(..)), got {other:?}"),
    }
}

/// Per-page recovery proof: a two-page document whose first page is the
/// panicking font and whose second page is a plain, decodable core-font page
/// must still yield the good page. Before the fix, driving the whole document
/// through one `output_doc` call meant the first page's panic discarded
/// everything; now each page is isolated, so the good page survives.
#[test]
fn good_page_survives_alongside_a_panicking_page() {
    let bytes = build_two_page_pdf_with_one_bad_page();
    let pages = kopitiam_pdf::extract_from_bytes(&bytes)
        .expect("a document with at least one good page must extract, not error");

    // Exactly the good page comes back; the panicking page is skipped.
    assert_eq!(
        pages.len(),
        1,
        "expected the single good page to survive; pages = {pages:?}"
    );
    let spans = &pages[0].spans;
    assert!(
        spans.iter().any(|s| s.text.contains("Good")),
        "expected the good page's text to be recovered, got {spans:?}"
    );
}

/// Build a two-page PDF: page 1 uses the panicking unmapped font (as in
/// `build_panicking_pdf`), page 2 uses core-font Helvetica with a normal
/// string, so it decodes cleanly.
fn build_two_page_pdf_with_one_bad_page() -> Vec<u8> {
    let mut doc = Document::with_version("1.5");

    // --- Page 1: the panicking font. ---
    let to_unicode = doc.add_object(Stream::new(dictionary! {}, incomplete_to_unicode_cmap()));
    let bad_font = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "ABCDEF+NoMappingFont",
        "ToUnicode" => to_unicode,
    });
    let bad_resources = doc.add_object(dictionary! {
        "Font" => dictionary! { "F1" => bad_font },
    });
    let bad_content = Content {
        operations: vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 24.into()]),
            Operation::new("Td", vec![72.into(), 700.into()]),
            Operation::new("Tj", vec![Object::string_literal("A")]),
            Operation::new("ET", vec![]),
        ],
    };
    let bad_content_id = doc.add_object(Stream::new(dictionary! {}, bad_content.encode().unwrap()));

    // --- Page 2: a clean core font. ---
    let good_font = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });
    let good_resources = doc.add_object(dictionary! {
        "Font" => dictionary! { "F1" => good_font },
    });
    let good_content = Content {
        operations: vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 24.into()]),
            Operation::new("Td", vec![72.into(), 700.into()]),
            Operation::new("Tj", vec![Object::string_literal("Good")]),
            Operation::new("ET", vec![]),
        ],
    };
    let good_content_id =
        doc.add_object(Stream::new(dictionary! {}, good_content.encode().unwrap()));

    let pages_id = doc.new_object_id();
    let bad_page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Contents" => bad_content_id,
        "Resources" => bad_resources,
    });
    let good_page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Contents" => good_content_id,
        "Resources" => good_resources,
    });
    let pages = dictionary! {
        "Type" => "Pages",
        "Kids" => vec![Object::Reference(bad_page_id), Object::Reference(good_page_id)],
        "Count" => 2,
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages));

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);

    let mut bytes = Vec::new();
    doc.save_to(&mut bytes)
        .expect("saving the fixture PDF must succeed");
    bytes
}
