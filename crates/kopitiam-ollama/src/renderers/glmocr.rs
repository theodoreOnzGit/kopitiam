//! **GLM-OCR** -- GLM-4.7's framing, plus images and a lot more newlines.
//!
//! **Upstream:** `model/renderers/glmocr.go`. Registered as `glm-ocr`.
//!
//! Special tokens: **`[gMASK]`**, **`<sop>`**, **`<|system|>`**, **`<|user|>`**,
//! **`<|assistant|>`**, **`<|observation|>`**, **`<think>` / `</think>`**. No
//! BOS.
//!
//! Differences from [`super::glm47::Glm47Renderer`], all deliberate:
//!
//! * **Every role marker is followed by a newline**, and assistant turns are
//!   *closed* with one too.
//! * **Thinking is off unless the caller says otherwise**, and the switch is
//!   three-state: only when `think` is `Some(..)` does the renderer touch it at
//!   all. `Some(false)` appends `/nothink` to the user's text and emits
//!   `<think></think>` in the generation prompt; `None` does **neither**. Treat
//!   `None` as `false` and every default request grows a `/nothink` it should
//!   not have.
//! * Content and reasoning are **trimmed** on assistant turns.
//! * Images flow through the shared `[img-N]` numbering
//!   ([`super::image_tags`]), and the offset **carries across messages** so the
//!   second image in a conversation is `[img-1]`, not `[img-0]` again.

use super::glm47::{GLM_INLINE_CALL_FORMAT, GLM_TOOLS_HEADER, render_tool_arguments};
use super::image_tags::render_content_with_image_tags;
use super::json::{add_spaces_outside_strings, go_tool};
use super::{Message, RenderError, Renderer, ThinkValue, Tool};

/// **Upstream:** `GlmOcrRenderer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GlmOcrRenderer {
    /// Use `[img-N]` markers. When off, image content is passed through
    /// untouched -- this family has no native image token of its own here.
    pub use_img_tags: bool,
}

impl GlmOcrRenderer {
    /// **Upstream:** `(*GlmOcrRenderer).renderContent`.
    fn render_content(&self, message: &Message, image_offset: usize) -> (String, usize) {
        if self.use_img_tags {
            return render_content_with_image_tags(
                &message.content,
                message.images.len(),
                image_offset,
            );
        }
        (message.content.clone(), image_offset)
    }
}

impl Renderer for GlmOcrRenderer {
    fn render(
        &self,
        messages: &[Message],
        tools: &[Tool],
        think: Option<&ThinkValue>,
    ) -> Result<String, RenderError> {
        let mut sb = String::from("[gMASK]<sop>");

        if !tools.is_empty() {
            sb.push_str("<|system|>\n");
            sb.push_str(GLM_TOOLS_HEADER);
            for tool in tools {
                sb.push_str(&add_spaces_outside_strings(&go_tool(tool)));
                sb.push('\n');
            }
            sb.push_str("</tools>\n\n");
            sb.push_str(GLM_INLINE_CALL_FORMAT);
        }

        // Three-state, and the distinction matters -- see the module docs.
        let thinking_explicitly_set = think.is_some();
        let enable_thinking = think.is_some_and(|t| t.enabled());

        let mut image_offset = 0usize;
        for (i, message) in messages.iter().enumerate() {
            match message.role.as_str() {
                "user" => {
                    sb.push_str("<|user|>\n");
                    let (content, next) = self.render_content(message, image_offset);
                    image_offset = next;
                    sb.push_str(&content);
                    if thinking_explicitly_set
                        && !enable_thinking
                        && !message.content.ends_with("/nothink")
                    {
                        sb.push_str("/nothink");
                    }
                }
                "assistant" => {
                    sb.push_str("<|assistant|>\n");
                    if !message.thinking.is_empty() {
                        sb.push_str(&format!("<think>{}</think>", message.thinking.trim()));
                    } else {
                        sb.push_str("<think></think>");
                    }
                    if !message.content.is_empty() {
                        sb.push('\n');
                        sb.push_str(message.content.trim());
                    }
                    for tc in &message.tool_calls {
                        sb.push_str(&format!("\n<tool_call>{}", tc.function.name));
                        sb.push_str(&render_tool_arguments(&tc.function.arguments));
                        sb.push_str("</tool_call>");
                    }
                    sb.push('\n');
                }
                "tool" => {
                    if i == 0 || messages[i - 1].role != "tool" {
                        sb.push_str("<|observation|>");
                    }
                    let (content, next) = self.render_content(message, image_offset);
                    image_offset = next;
                    sb.push_str("\n<tool_response>\n");
                    sb.push_str(&content);
                    sb.push_str("\n</tool_response>\n");
                }
                "system" => {
                    sb.push_str("<|system|>\n");
                    sb.push_str(&message.content);
                    sb.push('\n');
                }
                _ => {}
            }
        }

        sb.push_str("<|assistant|>\n");
        if thinking_explicitly_set && !enable_thinking {
            sb.push_str("<think></think>\n");
        }

        Ok(sb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_msg(role: &str, content: &str, images: &[&str]) -> Message {
        Message {
            role: role.into(),
            content: content.into(),
            images: images.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    /// Upstream `TestGlmOcrRenderer_Images`, all five cases.
    #[test]
    fn glmocr_numbers_images_across_the_whole_conversation() {
        let with_tags = GlmOcrRenderer { use_img_tags: true };

        assert_eq!(
            with_tags
                .render(
                    &[img_msg("user", "Describe this image.", &["img1"])],
                    &[],
                    None
                )
                .unwrap(),
            "[gMASK]<sop><|user|>\n[img-0] Describe this image.<|assistant|>\n"
        );

        assert_eq!(
            with_tags
                .render(
                    &[img_msg("user", "Describe these images.", &["img1", "img2"])],
                    &[],
                    None
                )
                .unwrap(),
            "[gMASK]<sop><|user|>\n[img-0][img-1] Describe these images.<|assistant|>\n"
        );

        // The offset carries across turns: the second image is [img-1].
        assert_eq!(
            with_tags
                .render(
                    &[
                        img_msg("user", "First image", &["img1"]),
                        Message::new("assistant", "Processed."),
                        img_msg("user", "Second image", &["img2"]),
                    ],
                    &[],
                    None
                )
                .unwrap(),
            "[gMASK]<sop><|user|>\n[img-0] First image<|assistant|>\n<think></think>\nProcessed.\n<|user|>\n[img-1] Second image<|assistant|>\n"
        );

        // Markers off -> content untouched even though an image is attached.
        assert_eq!(
            GlmOcrRenderer::default()
                .render(
                    &[img_msg("user", "No image tags expected.", &["img1"])],
                    &[],
                    None
                )
                .unwrap(),
            "[gMASK]<sop><|user|>\nNo image tags expected.<|assistant|>\n"
        );

        assert_eq!(
            with_tags
                .render(&[Message::new("user", "Text only message.")], &[], None)
                .unwrap(),
            "[gMASK]<sop><|user|>\nText only message.<|assistant|>\n"
        );
    }

    /// Not an upstream fixture: pins the three-state `think`. `None` must NOT
    /// behave like `Some(false)` -- otherwise every default request grows a
    /// `/nothink` the caller never asked for.
    #[test]
    fn a_missing_think_value_is_not_the_same_as_thinking_off() {
        let r = GlmOcrRenderer::default();
        let msgs = [Message::new("user", "hi")];

        let unset = r.render(&msgs, &[], None).unwrap();
        assert!(!unset.contains("/nothink"), "{unset}");
        assert!(unset.ends_with("<|assistant|>\n"), "{unset}");

        let off = r
            .render(&msgs, &[], Some(&ThinkValue::Bool(false)))
            .unwrap();
        assert_eq!(
            off,
            "[gMASK]<sop><|user|>\nhi/nothink<|assistant|>\n<think></think>\n"
        );
    }
}
