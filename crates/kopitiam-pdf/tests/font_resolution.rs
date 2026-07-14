//! Integration test for font resource resolution (kopitiam-q4f): a real
//! PDF, built with `lopdf` directly (rather than checked in as an opaque
//! binary fixture, so the exact bytes under test stay readable and
//! maintainable), exercising the full `extract_from_bytes` pipeline end to
//! end -- font resource dictionaries, `FontDescriptor`, and multiple `Tf`
//! font switches within one content stream.
//!
//! `lopdf` is already a direct dependency of `kopitiam-pdf` (see
//! `Cargo.toml`), so building the fixture in-process here needs no new
//! dependency either.

use lopdf::{Document, Object, Stream, content::Content, content::Operation, dictionary};

/// Build a single-page PDF whose content stream selects four different
/// standard-14 fonts (no `FontDescriptor` -- style must come from the name
/// heuristic) and one subset-tagged font with an explicit
/// `FontDescriptor` (style must come from the descriptor, overriding what
/// the plain name would suggest), writing one short `Tj` string in each.
fn build_fixture_pdf() -> Vec<u8> {
    let mut doc = Document::with_version("1.5");

    let helvetica = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });
    let helvetica_bold = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica-Bold",
    });
    let times_italic = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Times-Italic",
    });
    let times_bold_italic = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Times-BoldItalic",
    });

    // A font whose *name* carries no style suffix at all, so the only way
    // to recover its bold+italic style is the FontDescriptor -- proving
    // the descriptor path (not just the name-heuristic path) works
    // end-to-end through real lopdf objects (Integer /Flags, /FontWeight,
    // References, ...), which the pure unit tests in `font.rs` cannot
    // exercise since they hand-construct `DescriptorSignals` directly.
    let descriptor = doc.add_object(dictionary! {
        "Type" => "FontDescriptor",
        "FontName" => "ABCDEF+CustomSans",
        // Bit 7 (value 64): Italic.
        "Flags" => Object::Integer(64),
        "FontWeight" => Object::Integer(700),
    });
    let custom_sans = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "TrueType",
        "BaseFont" => "ABCDEF+CustomSans",
        "FontDescriptor" => descriptor,
    });

    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! {
            "F1" => helvetica,
            "F2" => helvetica_bold,
            "F3" => times_italic,
            "F4" => times_bold_italic,
            "F5" => custom_sans,
        },
    });

    let content = Content {
        operations: vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 24.into()]),
            Operation::new("Td", vec![72.into(), 700.into()]),
            Operation::new("Tj", vec![Object::string_literal("Plain")]),
            Operation::new("Tf", vec!["F2".into(), 24.into()]),
            Operation::new("Td", vec![0.into(), (-30).into()]),
            Operation::new("Tj", vec![Object::string_literal("Heavy")]),
            Operation::new("Tf", vec!["F3".into(), 24.into()]),
            Operation::new("Td", vec![0.into(), (-30).into()]),
            Operation::new("Tj", vec![Object::string_literal("Slanted")]),
            Operation::new("Tf", vec!["F4".into(), 24.into()]),
            Operation::new("Td", vec![0.into(), (-30).into()]),
            Operation::new("Tj", vec![Object::string_literal("BothStyles")]),
            Operation::new("Tf", vec!["F5".into(), 24.into()]),
            Operation::new("Td", vec![0.into(), (-30).into()]),
            Operation::new("Tj", vec![Object::string_literal("Described")]),
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
    doc.save_to(&mut bytes).expect("saving the fixture PDF must succeed");
    bytes
}

fn dict_by_text<'a>(spans: &'a [kopitiam_pdf::TextSpan], text: &str) -> &'a kopitiam_pdf::TextSpan {
    spans
        .iter()
        .find(|span| span.text.contains(text))
        .unwrap_or_else(|| panic!("expected a span containing {text:?}, got {spans:?}"))
}

#[test]
fn real_font_names_and_styles_come_through_a_real_pdf() {
    let bytes = build_fixture_pdf();
    let pages = kopitiam_pdf::extract_from_bytes(&bytes).expect("extraction must succeed");
    assert_eq!(pages.len(), 1);
    let spans = &pages[0].spans;
    assert!(!spans.is_empty(), "expected at least one text span");

    let plain = dict_by_text(spans, "Plain");
    assert_eq!(plain.font_name.as_deref(), Some("Helvetica"));
    assert_eq!(plain.font_style.bold, Some(false));
    assert_eq!(plain.font_style.italic, Some(false));
    assert_eq!(plain.font_style.family.as_deref(), Some("Helvetica"));

    let heavy = dict_by_text(spans, "Heavy");
    assert_eq!(heavy.font_name.as_deref(), Some("Helvetica-Bold"));
    assert_eq!(heavy.font_style.bold, Some(true));
    assert_eq!(heavy.font_style.italic, Some(false));

    let slanted = dict_by_text(spans, "Slanted");
    assert_eq!(slanted.font_name.as_deref(), Some("Times-Italic"));
    assert_eq!(slanted.font_style.bold, Some(false));
    assert_eq!(slanted.font_style.italic, Some(true));

    let both = dict_by_text(spans, "BothStyles");
    assert_eq!(both.font_name.as_deref(), Some("Times-BoldItalic"));
    assert_eq!(both.font_style.bold, Some(true));
    assert_eq!(both.font_style.italic, Some(true));

    // This one's name ("ABCDEF+CustomSans") carries no style suffix at
    // all -- if this came out bold+italic, it can only be because the
    // FontDescriptor's /Flags and /FontWeight were actually read, not
    // because of name-sniffing.
    let described = dict_by_text(spans, "Described");
    assert_eq!(described.font_name.as_deref(), Some("ABCDEF+CustomSans"));
    assert_eq!(described.font_style.bold, Some(true));
    assert_eq!(described.font_style.italic, Some(true));
    assert_eq!(described.font_style.family.as_deref(), Some("CustomSans"));
}
