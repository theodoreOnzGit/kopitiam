//! `[img-N]` placeholder substitution, the legacy server-side flavour.
//!
//! **Upstream:** `model/renderers/image_tags.go`.
//!
//! ## What this is for
//!
//! Vision models normally get images spliced in as their own special tokens
//! (`<|vision_start|><|image_pad|><|vision_end|>` for Qwen, `<start_of_image>`
//! for Gemma). But ollama's **older** server path instead put a textual
//! `[img-0]`, `[img-1]`, ... marker in the prompt and let the runner match those
//! markers to the attached image blobs. Some deployments still run that way, so
//! every image-capable renderer carries a `use_img_tags` switch and calls this
//! when it is on.
//!
//! ## The rules, exactly
//!
//! Given the message text, how many images this message carries, and the running
//! image counter (`image_offset`, which keeps numbering across a whole
//! conversation):
//!
//! 1. **No images -> nothing happens**, and the counter does not move.
//! 2. **The text already contains `[img-`** (a *numbered* marker) -> the caller
//!    has done their own numbering, so leave the text completely alone. The
//!    counter still advances by `image_count`, otherwise later messages would
//!    reuse numbers.
//! 3. Otherwise, for each image in turn: if there is an unnumbered `[img]`
//!    placeholder left, replace **the first one** with `[img-N]`; if there is
//!    not, the marker gets **prepended** to the front of the message instead.
//! 4. If anything was prepended and the text is non-empty, a **single space** is
//!    inserted between the markers and the text -- but only when the text starts
//!    with a non-whitespace character. Prepending `[img-0]` onto `" hello"`
//!    must not produce a double space.
//!
//! What would make this wrong: dropping rule 4's whitespace check (the model
//! sees a stray double space), or advancing the counter in rule 1 (every later
//! image is then off by one and points at the wrong blob).

/// **Upstream:** `renderContentWithImageTags`.
///
/// Returns the rewritten content and the **new** image offset.
pub(crate) fn render_content_with_image_tags(
    content: &str,
    image_count: usize,
    image_offset: usize,
) -> (String, usize) {
    if image_count == 0 {
        return (content.to_string(), image_offset);
    }

    // Already numbered by hand -- hands off, but still burn the numbers.
    if content.contains("[img-") {
        return (content.to_string(), image_offset + image_count);
    }

    let mut content = content.to_string();
    let mut prefix = String::new();
    for i in 0..image_count {
        let img_tag = format!("[img-{}]", image_offset + i);
        if let Some(pos) = content.find("[img]") {
            // `strings.Replace(content, "[img]", tag, 1)` -- first occurrence only.
            content.replace_range(pos..pos + "[img]".len(), &img_tag);
        } else {
            prefix.push_str(&img_tag);
        }
    }

    if !prefix.is_empty()
        && !content.is_empty()
        && content.chars().next().is_some_and(|c| !c.is_whitespace())
    {
        prefix.push(' ');
    }

    (prefix + &content, image_offset + image_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixtures are upstream's `TestRenderContentWithImageTags`
    /// (`image_tags_test.go`), copied across verbatim -- they are the spec.
    #[test]
    fn image_tags_follow_the_upstream_placeholder_rules() {
        let cases: &[(&str, &str, usize, usize, &str, usize)] = &[
            (
                "prefixes when there are no placeholders",
                "describe this image",
                2,
                0,
                "[img-0][img-1] describe this image",
                2,
            ),
            (
                "replaces explicit placeholders in order",
                "compare [img] and [img]",
                2,
                3,
                "compare [img-3] and [img-4]",
                5,
            ),
            (
                "prefixes extra images after placeholders are exhausted",
                "compare [img]",
                2,
                0,
                "[img-1] compare [img-0]",
                2,
            ),
            (
                "leaves leftover placeholders when there are fewer images",
                "compare [img] and [img]",
                1,
                0,
                "compare [img-0] and [img]",
                1,
            ),
            (
                "preserves already-numbered placeholders",
                "compare [img-0] and [img-1]",
                2,
                0,
                "compare [img-0] and [img-1]",
                2,
            ),
        ];

        for (name, content, count, offset, want, want_offset) in cases {
            let (got, got_offset) = render_content_with_image_tags(content, *count, *offset);
            assert_eq!(&got, want, "{name}");
            assert_eq!(got_offset, *want_offset, "{name}");
        }
    }

    /// Not an upstream fixture -- this pins rule 1, which upstream states in
    /// code but never asserts. Advancing the counter here would misnumber every
    /// later image in the conversation.
    #[test]
    fn a_message_with_no_images_does_not_move_the_counter() {
        let (got, offset) = render_content_with_image_tags("just text", 0, 7);
        assert_eq!(got, "just text");
        assert_eq!(offset, 7);
    }

    /// Rule 4's whitespace check: no double space when the text already leads
    /// with one.
    #[test]
    fn no_extra_space_is_added_when_the_text_already_starts_with_one() {
        let (got, _) = render_content_with_image_tags(" leading space", 1, 0);
        assert_eq!(got, "[img-0] leading space");
    }
}
