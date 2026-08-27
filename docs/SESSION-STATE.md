# Session state — kopitiam-pdf 0.2.9

Last updated: 2026-08-27. See `bn list` for what is left; this file records the
**in-flight** state beads cannot express.

## What 0.2.9 is

Version bumped `0.2.8 -> 0.2.9` in `crates/kopitiam-pdf/Cargo.toml`. 0.2.8 is
already live on crates.io (2026-08-27 01:11 UTC), so this is a fresh line.

Already landed and pushed (commits `cfc6192`, `2cbd5ea`, `3836dbc`, `eb361e8`):

* Annotation **rendering** — `annot_run.rs` (consume `/AP`, §12.5.5) +
  `annot_appearance.rs` (synthesise one when the file has none). gh-79, closed.
* Stroker cap/join port — `draw_path.rs`. gh-82, closed.
* `examples/annots.rs` headless harness; `tests/annots.rs` + 2 synthetic fixtures.

In flight for 0.2.9 (this file's subject): the **write** side.

## kpdf is now a binary, not an example

`examples/kpdf.rs` -> `src/bin/kpdf.rs`. `eframe`/`egui`/`rfd` moved from
`[dev-dependencies]` to **optional** `[dependencies]` behind a default-on `kpdf`
feature, with `required-features = ["kpdf"]` on the bin target.

Verified: `cargo tree -e normal` shows **6** egui entries by default and **0**
with `--no-default-features`. So `cargo install kopitiam-pdf` yields a working
reader, while a library consumer (`apps/cli`, `kovan`) can take
`default-features = false` and never compile a GUI toolkit. Maintainer chose
this over plain hard dependencies. Note the direction matters: default-feature
-> non-default later is NOT breaking for library users; hard dep -> optional
would be. Hence optional from the start.

## Parallel agents — one file each, contracts frozen

| Agent | Owns | Depends on |
|---|---|---|
| writer | `src/mupdf/write.rs` | — (**foundation**, others block on it) |
| authoring | `src/mupdf/annot_edit.rs` | `write` |
| forms | `src/mupdf/form.rs` | `write` |
| UI | `src/bin/kpdf.rs` | `annot_edit`, `form` |

Main session owns `Cargo.toml`, `mod.rs`, `tests/`, `docs/`, and **all
formatting**. Agents are forbidden from running any formatter — see CLAUDE.md's
"agents never format" hard rule, added this session after an agent's
`cargo fmt -p kopitiam-pdf` reformatted seven files it did not own.

### Frozen contract — `write.rs`

```rust
pub enum NewObject { Plain(Object), Stream { dict: Object, data: Vec<u8> } }
pub fn write_object(out: &mut Vec<u8>, obj: &Object);
pub fn next_object_number(doc: &PdfDocument) -> i32;
pub fn incremental_update(doc: &PdfDocument, updates: &[(i32, NewObject)]) -> Result<Vec<u8>>;
```

### Frozen contract — `annot_edit.rs`

```rust
pub struct InkStroke { pub points: Vec<(f32, f32)> }
pub struct InkAnnotSpec {
    pub page_index: usize, pub strokes: Vec<InkStroke>,
    pub color: [f32; 3], pub width: f32, pub opacity: f32,
    pub author: Option<String>,
}
pub struct AnnotRef { pub num: i32, pub subtype: String, pub rect: Rect }
pub fn page_annot_refs(doc: &PdfDocument, page_index: usize) -> Vec<AnnotRef>;
pub fn add_ink_annot(doc: &PdfDocument, spec: &InkAnnotSpec) -> Result<Vec<u8>>;
pub fn delete_annot(doc: &PdfDocument, page_index: usize, annot_num: i32) -> Result<Vec<u8>>;
pub struct EditHistory { /* opaque */ }
// new / push / undo / redo / can_undo / can_redo / current
```

### Frozen contract — `form.rs`

```rust
pub enum FieldKind { Text, Checkbox, Radio, Combobox, Listbox, Button, Signature, Unknown }
pub struct FormField {
    pub obj_num: i32, pub page_index: usize, pub kind: FieldKind,
    pub name: String, pub value: String, pub rect: Rect,
    pub read_only: bool, pub on_state: Option<String>,
}
pub fn has_acroform(doc: &PdfDocument) -> bool;
pub fn page_form_fields(doc: &PdfDocument, page_index: usize) -> Vec<FormField>;
pub fn set_field_value(doc: &PdfDocument, field: &FormField, value: &str) -> Result<Vec<u8>>;
pub fn toggle_checkbox(doc: &PdfDocument, field: &FormField) -> Result<Vec<u8>>;
```

## Standing constraints

* **Incremental update only.** Never rewrite bytes before the append point. Two
  things depend on it: a file we cannot fully round-trip stays safe to annotate,
  and `EditHistory` undo is *truncation* (a previous state is a prefix, with its
  xref and `%%EOF` still intact). If any edit path ever rewrites in place, undo
  breaks — that invariant must be stated wherever it is relied on.
* **Always write a real `/AP`.** AP-less annots are what made annotations
  invisible in the first place (gh-79), and `hayro` — our own dependency —
  renders nothing without one.
* **No JavaScript engine for forms.** `pdf_set_field_value`'s
  `ignore_trigger_events` (`form.h:178`) makes read/set/toggle/regenerate
  reachable with no JS. Documented limitation: script-*computed* fields will not
  recalculate.
* **Cross-reader compatibility is the bar.** Output must open in Okular/poppler/
  Acrobat, not just in our parser. `pdftoppm` is installed for checking.
* **MuPDF pin.** New units cite `5fe54ce`; everything older cites `19f1284`.
  Deliberate and documented — AID-0056. Reuse the vendored pin, do not re-check
  upstream.

## Open questions

* This crate has **no settled formatting policy** — `HEAD` does not
  `cargo fmt --check` clean. Until it is decided, the main session's formatting
  pass should be conservative rather than crate-wide reflowing.
* `docs/port-ledger.md` is **stale** w.r.t. the annotation port: regenerating it
  on a machine missing a vendor tree silently drops rows (gh-81), so a lossy
  regeneration was reverted rather than committed.
* Combobox/listbox value setting is out of scope for 0.2.9 (checkbox/radio +
  text only, maintainer's call).
