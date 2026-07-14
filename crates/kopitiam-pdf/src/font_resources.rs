//! PDF-specific glue that resolves the font active for each word
//! `pdf-extract` will hand us, by walking a page's font resource
//! dictionaries and content stream directly with `lopdf`.
//!
//! # Why this needs its own content-stream walk
//!
//! `pdf-extract`'s [`pdf_extract::OutputDev`] trait -- the extraction
//! backend `kopitiam-pdf` builds on ([`crate::extractor`]) -- only tells
//! per-character callbacks the font *size*, never the font *resource* that
//! was active (see `output_character`'s signature). The font resource
//! itself is tracked internally by `pdf-extract`'s content-stream
//! interpreter (`Tf` sets it, `Tj`/`TJ` use it) but never surfaced.
//!
//! Recovering it therefore means re-walking the same content stream
//! ourselves with `lopdf` (which `pdf-extract` already depends on -- see
//! `crates/kopitiam-pdf/Cargo.toml` for why this adds no new dependency),
//! tracking `Tf`/`Tj`/`TJ`/`q`/`Q`/`Do` exactly as `pdf-extract`'s internal
//! `Processor::process_stream` does, and recording which font was active
//! for each text-showing operation.
//!
//! # Alignment strategy
//!
//! Re-deriving font info without duplicating `pdf-extract`'s glyph
//! decoding (character encodings, CID maps, ligatures, ...) requires a
//! trick: `pdf-extract` calls `OutputDev::begin_word` exactly once per
//! `Tj`/`TJ`-string operand it processes (see `show_text` in
//! `pdf-extract`'s source), *before* looking at the string's bytes at all.
//! So instead of trying to count glyphs or bytes, we independently walk the
//! same content stream and, for every `Tj` operand and every string
//! element inside a `TJ` array, push the font resource active at that
//! point into an ordered queue. Because both walks process the exact same
//! operator sequence in the exact same order (see below for the one
//! deliberate divergence), the *n*-th entry in our queue is always the
//! font for the *n*-th `begin_word` call `pdf-extract` will make on that
//! page -- no glyph/byte counting, and no dependency on font encoding,
//! required.
//!
//! [`crate::extractor::PageCollector`] consumes this queue by popping one
//! entry per `begin_word` call it receives, in lock-step.
//!
//! ## What is tracked, and why it is enough
//!
//! `pdf-extract`'s interpreter only calls `show_text` (and therefore only
//! ever calls `begin_word`) from its `Tj` and `TJ` operator handlers, and
//! its font state is affected only by `Tf` (sets the font) and `q`/`Q`
//! (save/restore the whole graphics state, which includes the font,
//! because a `Content Stream`'s `q ... Q` bracket restores text state too
//! -- ISO 32000-1 Table 52). `Do` recurses into a fresh, independent
//! `GraphicsState` for the referenced XObject's content stream (verified
//! against `pdf-extract`'s `process_stream`, which starts every recursive
//! call with a brand new default `GraphicsState` rather than inheriting
//! the caller's), so we mirror that: `walk_content`'s `Do` handler
//! recurses with its own fresh `current_font`/`font_stack`, using the
//! XObject's own `/Resources` (falling back to the caller's, exactly as
//! `pdf-extract` does). Operators that only affect position, color, or
//! clipping are irrelevant to font tracking and are ignored.
//!
//! `pdf-extract` does not special-case the XObject's `/Subtype` before
//! recursing into `Do` -- it treats every named XObject as a content
//! stream, including Image XObjects. We mirror that (for order-alignment
//! fidelity) but, unlike `pdf-extract`, never `unwrap()` the decode: if the
//! bytes are not a valid content stream (as image data usually is not),
//! [`lopdf::content::Content::decode`] fails and we simply contribute no
//! events for that XObject, rather than panicking.
//!
//! ## Graceful degradation
//!
//! If some PDF construct this module does not model causes our walk to
//! diverge from `pdf-extract`'s real one, the queue simply runs out early
//! for the rest of that page: [`crate::extractor::PageCollector`] treats an
//! empty queue as "font unresolved" (honest `None`, never a guess) rather
//! than misattributing a stale font to unrelated words. A self-referential
//! or deeply nested `Do` chain is bounded by
//! [`MAX_XOBJECT_RECURSION_DEPTH`] as a defensive measure `pdf-extract`
//! itself does not take.
//!
//! ## Deliberate divergence: no cross-scope font cache
//!
//! `pdf-extract` caches resolved fonts in a single `HashMap` keyed only by
//! resource name (e.g. `"F1"`), shared across the whole document -- so if
//! an XObject's `/Resources` happens to reuse a name already cached from
//! the page (or a different XObject), it would incorrectly reuse that
//! cached font. We resolve every `Tf` freshly against whichever
//! `/Resources` dictionary is in scope at that point, which is more
//! correct and does not affect operator-order alignment (the cache only
//! changes *which* font is attached, never *how many* text-showing
//! operators occur).

use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;

use lopdf::content::Content;
use lopdf::{Dictionary, Document, Object};

use crate::font::{DescriptorSignals, FontStyle, style_from_descriptor_and_name};

/// `/Flags` bit 7 (`Italic`), ISO 32000-1 Table 123. PDF flag bits are
/// 1-indexed from the least-significant bit, so bit *n* has value
/// `2^(n-1)`.
const ITALIC_FLAG_BIT: i64 = 1 << 6;

/// `/Flags` bit 19 (`ForceBold`), ISO 32000-1 Table 123.
const FORCE_BOLD_FLAG_BIT: i64 = 1 << 18;

/// Recursion cap for `Do` (XObject) traversal, guarding against malformed
/// or self-referential XObjects. `pdf-extract` itself has no such guard;
/// we add one defensively since `kopitiam-pdf` must not crash or hang on
/// hostile input just because it is doing extra work `pdf-extract` skips.
const MAX_XOBJECT_RECURSION_DEPTH: u32 = 32;

/// A font resolved from a PDF font resource dictionary: the raw `BaseFont`
/// PostScript name plus the style derived from it and (when available) its
/// `FontDescriptor`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedFont {
    pub base_font: String,
    pub style: FontStyle,
}

/// Ordered, per-page queue of the font active for each upcoming
/// `begin_word` call. `None` means the font resource could not be
/// resolved for that word (e.g. no `Tf` had been seen yet, or the resource
/// name did not resolve to a font dictionary); it is never a guess.
pub(crate) type WordFontQueue = VecDeque<Option<Rc<ResolvedFont>>>;

/// Build the per-page word-font queues for an entire document. Called once
/// up front by [`crate::extractor::run`], before `pdf-extract` walks the
/// same document, so that each page's queue is ready by the time
/// `PageCollector::begin_page` needs it.
pub(crate) fn font_timelines(doc: &Document) -> BTreeMap<u32, WordFontQueue> {
    let empty_resources = Dictionary::new();
    let mut timelines = BTreeMap::new();
    for (page_num, page_id) in doc.get_pages() {
        let resources = doc
            .get_dictionary(page_id)
            .ok()
            .and_then(|page_dict| inherited_resources(doc, page_dict))
            .unwrap_or(&empty_resources);
        let mut events = Vec::new();
        if let Ok(content_bytes) = doc.get_page_content(page_id) {
            walk_content(doc, &content_bytes, resources, &mut events, 0);
        }
        timelines.insert(page_num, events.into());
    }
    timelines
}

/// Find the page's effective `/Resources` dictionary, walking up the page
/// tree's `/Parent` chain and returning the *first* one found -- mirroring
/// `pdf-extract`'s own `get_inherited` helper exactly (it does not merge
/// resources across levels; it stops at the nearest ancestor that has
/// one), so that `Tf` resolution in [`walk_content`] agrees with what
/// `pdf-extract` itself would resolve.
fn inherited_resources<'a>(doc: &'a Document, page_dict: &'a Dictionary) -> Option<&'a Dictionary> {
    if let Ok(resources_obj) = page_dict.get(b"Resources")
        && let Ok((_, resolved)) = doc.dereference(resources_obj)
        && let Ok(dict) = resolved.as_dict()
    {
        return Some(dict);
    }
    let parent_id = page_dict.get(b"Parent").ok()?.as_reference().ok()?;
    let parent_dict = doc.get_dictionary(parent_id).ok()?;
    inherited_resources(doc, parent_dict)
}

/// Walk one content stream's operators, tracking the active font resource
/// and pushing one queue entry per `Tj`/`TJ`-string text-showing operation
/// -- see the module docs for why this suffices to stay in lock-step with
/// `pdf-extract`'s `begin_word` calls without decoding any text ourselves.
fn walk_content(
    doc: &Document,
    content_bytes: &[u8],
    resources: &Dictionary,
    events: &mut Vec<Option<Rc<ResolvedFont>>>,
    depth: u32,
) {
    if depth > MAX_XOBJECT_RECURSION_DEPTH {
        return;
    }
    let Ok(content) = Content::decode(content_bytes) else {
        return;
    };

    let mut current_font: Option<Rc<ResolvedFont>> = None;
    let mut font_stack: Vec<Option<Rc<ResolvedFont>>> = Vec::new();

    for operation in &content.operations {
        match operation.operator.as_str() {
            "Tf" => {
                if let Some(name) = operation.operands.first().and_then(|o| o.as_name().ok()) {
                    current_font = resolve_font_by_name(doc, resources, name).map(Rc::new);
                }
            }
            "Tj" => {
                if matches!(operation.operands.first(), Some(Object::String(_, _))) {
                    events.push(current_font.clone());
                }
            }
            "TJ" => {
                if let Some(Object::Array(items)) = operation.operands.first() {
                    for item in items {
                        if matches!(item, Object::String(_, _)) {
                            events.push(current_font.clone());
                        }
                    }
                }
            }
            // Text state (including the current font) is part of the
            // graphics state saved/restored by q/Q -- ISO 32000-1 Table
            // 52 -- so it must be tracked here too, or a `q ... Tf ... Q`
            // bracket would leak its font past its `Q`.
            "q" => font_stack.push(current_font.clone()),
            "Q" => {
                if let Some(restored) = font_stack.pop() {
                    current_font = restored;
                }
            }
            "Do" => {
                if let Some(name) = operation.operands.first().and_then(|o| o.as_name().ok())
                    && let Some((xobject_resources, decoded)) =
                        resolve_xobject(doc, resources, name)
                {
                    walk_content(doc, &decoded, xobject_resources, events, depth + 1);
                }
            }
            _ => {}
        }
    }
}

/// Resolve a `Do`-referenced XObject to its decoded content bytes and the
/// `/Resources` dictionary it should be interpreted against (its own, or
/// the caller's if it has none -- matching `pdf-extract`'s `Do` handler).
fn resolve_xobject<'a>(
    doc: &'a Document,
    resources: &'a Dictionary,
    name: &[u8],
) -> Option<(&'a Dictionary, Vec<u8>)> {
    let xobjects = doc.get_dict_in_dict(resources, b"XObject").ok()?;
    let xobject = xobjects.get(name).ok()?;
    let (_, xobject) = doc.dereference(xobject).ok()?;
    let stream = xobject.as_stream().ok()?;

    let xobject_resources = stream
        .dict
        .get(b"Resources")
        .ok()
        .and_then(|r| doc.dereference(r).ok())
        .and_then(|(_, obj)| obj.as_dict().ok())
        .unwrap_or(resources);

    let decoded = stream
        .decompressed_content()
        .unwrap_or_else(|_| stream.content.clone());
    Some((xobject_resources, decoded))
}

/// Resolve a `Tf` resource name (e.g. `b"F1"`) against a `/Resources`
/// dictionary's `/Font` sub-dictionary into a [`ResolvedFont`].
fn resolve_font_by_name(doc: &Document, resources: &Dictionary, name: &[u8]) -> Option<ResolvedFont> {
    let fonts = doc.get_dict_in_dict(resources, b"Font").ok()?;
    let font_dict = doc.get_dict_in_dict(fonts, name).ok()?;
    resolve_font_dict(doc, font_dict)
}

/// Resolve an already-looked-up font dictionary into a [`ResolvedFont`]:
/// its raw `BaseFont` name plus the style derived from its
/// `FontDescriptor` (falling back to name heuristics -- see
/// [`crate::font`]).
fn resolve_font_dict(doc: &Document, font_dict: &Dictionary) -> Option<ResolvedFont> {
    let base_font_bytes = font_dict.get(b"BaseFont").ok()?.as_name().ok()?;
    // PDF Name objects are conventionally ASCII/Latin-1; a lossy decode is
    // an honest best effort for the rare font with non-ASCII bytes in its
    // BaseFont name rather than a failure of the whole span.
    let base_font = String::from_utf8_lossy(base_font_bytes).into_owned();

    let signals = descriptor_for(doc, font_dict)
        .map(|descriptor| descriptor_signals(doc, descriptor))
        .unwrap_or_default();

    let style = style_from_descriptor_and_name(&base_font, signals);
    Some(ResolvedFont { base_font, style })
}

/// Locate a font dictionary's `FontDescriptor`. Simple fonts (Type1,
/// TrueType, MMType1, Type3) carry it directly; composite (Type0) fonts
/// carry it on the single entry of their `/DescendantFonts` array (ISO
/// 32000-1 §9.7.4). Standard-14 fonts referenced by name alone commonly
/// have no `FontDescriptor` at all, in which case this returns `None` and
/// callers fall back to the name heuristic -- an honest "unknown from the
/// descriptor", not a guess.
fn descriptor_for<'a>(doc: &'a Document, font_dict: &'a Dictionary) -> Option<&'a Dictionary> {
    if let Ok(descriptor) = doc.get_dict_in_dict(font_dict, b"FontDescriptor") {
        return Some(descriptor);
    }
    let descendants = font_dict.get(b"DescendantFonts").ok()?;
    let (_, descendants) = doc.dereference(descendants).ok()?;
    let descendants = descendants.as_array().ok()?;
    let descendant = descendants.first()?;
    let (_, descendant) = doc.dereference(descendant).ok()?;
    let descendant_dict = descendant.as_dict().ok()?;
    doc.get_dict_in_dict(descendant_dict, b"FontDescriptor").ok()
}

/// Extract the bold/italic-relevant fields from a `FontDescriptor`
/// dictionary into the pure [`DescriptorSignals`] the style-merge logic in
/// [`crate::font`] consumes.
fn descriptor_signals(doc: &Document, descriptor: &Dictionary) -> DescriptorSignals {
    let flags = descriptor
        .get(b"Flags")
        .ok()
        .and_then(|o| doc.dereference(o).ok())
        .and_then(|(_, o)| o.as_i64().ok());

    let italic_flag = flags.map(|f| f & ITALIC_FLAG_BIT != 0);
    let force_bold_flag = flags.and_then(|f| (f & FORCE_BOLD_FLAG_BIT != 0).then_some(true));

    let font_weight = descriptor
        .get(b"FontWeight")
        .ok()
        .and_then(|o| doc.dereference(o).ok())
        .and_then(|(_, o)| as_number(o));
    let stem_v = descriptor
        .get(b"StemV")
        .ok()
        .and_then(|o| doc.dereference(o).ok())
        .and_then(|(_, o)| as_number(o));

    DescriptorSignals {
        italic_flag,
        force_bold_flag,
        font_weight,
        stem_v,
    }
}

fn as_number(object: &Object) -> Option<f64> {
    match object {
        Object::Integer(i) => Some(*i as f64),
        Object::Real(r) => Some(f64::from(*r)),
        _ => None,
    }
}
