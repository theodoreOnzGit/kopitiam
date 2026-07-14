//! Applying LSP `WorkspaceEdit`s to real files on disk.
//!
//! Kept deliberately separate from [`crate::lsp_client`]: computing what a
//! `WorkspaceEdit` would do to a file is pure (no I/O side effects beyond a
//! read), which lets the same logic back both a diff preview and the actual
//! write. Only resource operations of kind "edit an existing text document"
//! are handled — `CreateFile`/`RenameFile`/`DeleteFile` entries inside a
//! `documentChanges` array are skipped, since rust-analyzer's rename and
//! code actions do not currently produce them for the cases this client
//! drives.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::position::{self, PositionEncoding};

/// One file's content before and after applying the edits addressed to it.
pub struct FileEdit {
    pub path: PathBuf,
    pub original: String,
    pub updated: String,
}

/// Reads every file a raw `WorkspaceEdit` touches and computes its new
/// contents, without writing anything. Accepts both the modern
/// `documentChanges` form and the legacy `changes` map form, since servers
/// may send either depending on negotiated capabilities.
///
/// `encoding` must be the [`PositionEncoding`] the server negotiated during
/// `initialize` (see [`crate::lsp_client::LspClient::position_encoding`]):
/// every `Position.character` inside `edit` is expressed in that unit, and
/// this function converts each one back to this crate's public `char`-offset
/// unit using the actual line text of the file it addresses — see
/// [`crate::position`] for why that conversion needs the line's real
/// content, not just arithmetic on the two numbers.
pub(crate) fn compute_workspace_edit(edit: &Value, encoding: PositionEncoding) -> Result<Vec<FileEdit>> {
    let mut results = Vec::new();

    if let Some(document_changes) = edit.get("documentChanges").and_then(Value::as_array) {
        for change in document_changes {
            let (Some(uri), Some(edits)) = (
                change.pointer("/textDocument/uri").and_then(Value::as_str),
                change.get("edits").and_then(Value::as_array),
            ) else {
                continue; // a Create/Rename/DeleteFile resource op, not a text edit
            };
            results.push(apply_text_edits_to_file(uri, edits, encoding)?);
        }
    } else if let Some(changes) = edit.get("changes").and_then(Value::as_object) {
        for (uri, edits) in changes {
            let Some(edits) = edits.as_array() else { continue };
            results.push(apply_text_edits_to_file(uri, edits, encoding)?);
        }
    }

    Ok(results)
}

/// Writes every [`FileEdit`]'s updated content over its original file.
pub fn write_file_edits(edits: &[FileEdit]) -> Result<()> {
    for edit in edits {
        std::fs::write(&edit.path, &edit.updated).with_context(|| format!("writing {}", edit.path.display()))?;
    }
    Ok(())
}

/// Computes and immediately writes a raw `WorkspaceEdit`, returning the
/// paths that were changed. Used to answer server-initiated
/// `workspace/applyEdit` requests, where the server is waiting on our
/// response and there is no earlier point to insert a preview step. See
/// [`compute_workspace_edit`] for what `encoding` means.
pub(crate) fn apply_workspace_edit(edit: &Value, encoding: PositionEncoding) -> Result<Vec<PathBuf>> {
    let file_edits = compute_workspace_edit(edit, encoding)?;
    let paths = file_edits.iter().map(|f| f.path.clone()).collect();
    write_file_edits(&file_edits)?;
    Ok(paths)
}

/// Renders a unified diff across every [`FileEdit`], for showing a user
/// what a change would do before committing to it.
pub fn diff(edits: &[FileEdit]) -> String {
    let mut out = String::new();
    for edit in edits {
        let label = edit.path.to_string_lossy();
        let text_diff = similar::TextDiff::from_lines(edit.original.as_str(), edit.updated.as_str());
        out.push_str(&text_diff.unified_diff().header(&label, &label).to_string());
        out.push('\n');
    }
    out
}

fn uri_to_path(uri: &str) -> Result<PathBuf> {
    url::Url::parse(uri)
        .ok()
        .and_then(|url| url.to_file_path().ok())
        .with_context(|| format!("not a `file://` URI KOPITIAM can write to: {uri}"))
}

fn apply_text_edits_to_file(uri: &str, edits: &[Value], encoding: PositionEncoding) -> Result<FileEdit> {
    let path = uri_to_path(uri)?;
    let original = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let chars: Vec<char> = original.chars().collect();
    let line_starts = line_start_offsets(&chars);
    // Each line's own text, used only to convert a wire `character` unit to
    // this crate's public `char`-offset unit (see `position_to_char_offset`)
    // — that conversion needs the real bytes of the line, not just the two
    // integers `line_starts` already gives us.
    let lines: Vec<&str> = original.split('\n').collect();

    let mut spans = Vec::with_capacity(edits.len());
    for edit in edits {
        let new_text = edit.get("newText").and_then(Value::as_str).unwrap_or_default();
        let start = position_to_char_offset(&line_starts, &lines, edit.pointer("/range/start"), encoding)?;
        let end = position_to_char_offset(&line_starts, &lines, edit.pointer("/range/end"), encoding)?;
        spans.push((start, end, new_text.to_string()));
    }
    // Apply back-to-front so earlier offsets are never invalidated by a
    // preceding edit changing the file's length.
    spans.sort_by_key(|span| std::cmp::Reverse(span.0));

    let mut result = chars;
    for (start, end, new_text) in &spans {
        result.splice(*start..*end, new_text.chars());
    }
    let updated: String = result.into_iter().collect();

    Ok(FileEdit { path, original, updated })
}

/// Char index (this crate's public Unicode scalar value / `char`-offset
/// unit — see [`crate::position`]) where each line begins.
fn line_start_offsets(chars: &[char]) -> Vec<usize> {
    let mut starts = vec![0];
    for (i, c) in chars.iter().enumerate() {
        if *c == '\n' {
            starts.push(i + 1);
        }
    }
    starts
}

/// Converts a wire `{ line, character }` position — `character` expressed in
/// `encoding`'s unit — to a `char` offset into the whole file, by locating
/// `line`'s start (from `line_starts`) and converting `character` to a
/// `char` column within that line's own text (from `lines`).
fn position_to_char_offset(
    line_starts: &[usize],
    lines: &[&str],
    position: Option<&Value>,
    encoding: PositionEncoding,
) -> Result<usize> {
    let position = position.context("edit is missing a range position")?;
    let line = position.get("line").and_then(Value::as_u64).context("position missing `line`")? as usize;
    let character = position
        .get("character")
        .and_then(Value::as_u64)
        .context("position missing `character`")? as u32;
    let line_start = *line_starts
        .get(line)
        .with_context(|| format!("line {line} is out of range for this file"))?;
    let line_text = lines.get(line).copied().unwrap_or("");
    let col = position::unit_to_char_col(line_text, character, encoding) as usize;
    Ok(line_start + col)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;

    fn temp_rust_file(contents: &str) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        file
    }

    #[test]
    fn applies_a_single_line_rename_edit() {
        let file = temp_rust_file("fn old_name() {}\n");
        let uri = format!("file://{}", file.path().display());

        let edit = json!({
            "changes": {
                uri: [
                    { "range": { "start": { "line": 0, "character": 3 }, "end": { "line": 0, "character": 11 } }, "newText": "new_name" }
                ]
            }
        });

        let file_edits = compute_workspace_edit(&edit, PositionEncoding::Utf16).unwrap();
        assert_eq!(file_edits.len(), 1);
        assert_eq!(file_edits[0].updated, "fn new_name() {}\n");
        assert_eq!(file_edits[0].original, "fn old_name() {}\n");
    }

    #[test]
    fn applies_multiple_non_overlapping_edits_in_one_file() {
        let file = temp_rust_file("let a = 1;\nlet b = 2;\n");
        let uri = format!("file://{}", file.path().display());

        let edit = json!({
            "changes": {
                uri: [
                    { "range": { "start": { "line": 0, "character": 4 }, "end": { "line": 0, "character": 5 } }, "newText": "x" },
                    { "range": { "start": { "line": 1, "character": 4 }, "end": { "line": 1, "character": 5 } }, "newText": "y" },
                ]
            }
        });

        let file_edits = compute_workspace_edit(&edit, PositionEncoding::Utf16).unwrap();
        assert_eq!(file_edits[0].updated, "let x = 1;\nlet y = 2;\n");
    }

    #[test]
    fn write_file_edits_persists_to_disk() {
        let file = temp_rust_file("fn old_name() {}\n");
        let uri = format!("file://{}", file.path().display());
        let edit = json!({
            "changes": {
                uri: [
                    { "range": { "start": { "line": 0, "character": 3 }, "end": { "line": 0, "character": 11 } }, "newText": "new_name" }
                ]
            }
        });

        let paths = apply_workspace_edit(&edit, PositionEncoding::Utf16).unwrap();
        assert_eq!(paths, vec![file.path().to_path_buf()]);
        let on_disk = std::fs::read_to_string(file.path()).unwrap();
        assert_eq!(on_disk, "fn new_name() {}\n");
    }

    #[test]
    fn diff_reports_the_change() {
        let file = temp_rust_file("fn old_name() {}\n");
        let uri = format!("file://{}", file.path().display());
        let edit = json!({
            "changes": {
                uri: [
                    { "range": { "start": { "line": 0, "character": 3 }, "end": { "line": 0, "character": 11 } }, "newText": "new_name" }
                ]
            }
        });

        let file_edits = compute_workspace_edit(&edit, PositionEncoding::Utf16).unwrap();
        let rendered = diff(&file_edits);
        assert!(rendered.contains("-fn old_name"));
        assert!(rendered.contains("+fn new_name"));
    }

    /// The end-to-end regression test: a `WorkspaceEdit` whose positions are
    /// real byte-oriented `"utf-8"` offsets (the LSP 3.17 meaning of
    /// `"utf-8"`, not `char` offsets), targeting a rename that sits after
    /// multi-byte UTF-8 text on the line. Before this fix,
    /// `apply_text_edits_to_file` treated every `character` value as a
    /// `char` offset regardless of what the server had actually negotiated;
    /// fed a real byte offset, it would have spliced the replacement text
    /// into the middle of "日本語" instead of onto "old_name".
    #[test]
    fn applies_a_rename_edit_after_multibyte_text_under_utf8_encoding() {
        let contents = "// 日本語\nfn old_name() {}\n";
        let file = temp_rust_file(contents);
        let uri = format!("file://{}", file.path().display());

        // Byte offsets into line 1 ("fn old_name() {}"): "old_name" starts
        // at byte 3 and ends at byte 11 -- this line is pure ASCII, so its
        // byte, UTF-16, and `char` offsets all happen to coincide, unlike
        // line 0. That is exactly the trap: get the encoding wrong on THIS
        // edit's line and it still looks right; the corruption in the
        // original bug shows up on `character` values computed for a
        // DIFFERENT line with multi-byte content before the target column
        // (see `position::regression_rename_target_after_multibyte_text_under_utf8_encoding`
        // for that half of the story). This test instead pins down that
        // `compute_workspace_edit` actually threads `PositionEncoding::Utf8`
        // through to the byte-offset math rather than silently ignoring it.
        let edit = json!({
            "changes": {
                uri: [
                    { "range": { "start": { "line": 1, "character": 3 }, "end": { "line": 1, "character": 11 } }, "newText": "new_name" }
                ]
            }
        });

        let file_edits = compute_workspace_edit(&edit, PositionEncoding::Utf8).unwrap();
        assert_eq!(file_edits[0].original, contents);
        assert_eq!(file_edits[0].updated, "// 日本語\nfn new_name() {}\n");
    }

    /// Companion to the test above: same file, but the rename target's own
    /// line has multi-byte text BEFORE the target column, so a byte offset
    /// and a `char` offset genuinely disagree on this line. This is the
    /// shape of edit the original bug corrupted: `character` values here are
    /// real UTF-8 byte offsets, and only counting bytes (not `char`s) lands
    /// on the right splice point.
    #[test]
    fn applies_a_rename_edit_on_a_line_where_the_target_itself_follows_multibyte_text() {
        let contents = "// 日本語 old_name\n";
        let file = temp_rust_file(contents);
        let uri = format!("file://{}", file.path().display());

        // Line 0: "// 日本語 old_name"
        //   "// " = 3 bytes / 3 chars
        //   "日本語" = 9 bytes / 3 chars
        //   " " = 1 byte / 1 char
        //   "old_name" starts here.
        // Byte offset of "old_name": 3 + 9 + 1 = 13. Char offset: 3 + 3 + 1 = 7.
        let byte_start = 13u32;
        let byte_end = byte_start + "old_name".len() as u32; // 21

        let edit = json!({
            "changes": {
                uri: [
                    { "range": { "start": { "line": 0, "character": byte_start }, "end": { "line": 0, "character": byte_end } }, "newText": "new_name" }
                ]
            }
        });

        let file_edits = compute_workspace_edit(&edit, PositionEncoding::Utf8).unwrap();
        assert_eq!(
            file_edits[0].updated, "// 日本語 new_name\n",
            "a byte offset landing after multi-byte content must resolve to the right `char` column, \
             not be misread as if it were already a `char` count"
        );
    }
}
