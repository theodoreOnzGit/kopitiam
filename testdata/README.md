# testdata

Real-world documents used to verify rendering by hand. **Deliberately outside
`crates/`**, so `cargo package` can never sweep them into a published `.crate`
— cargo only packages files inside the package directory, which makes the
exclusion structural rather than a rule someone has to remember.

| File | What it is | Why it's here |
|---|---|---|
| `arxiv-2608.17504v1.pdf` | The maintainer's arXiv paper (LaTeX/pdfTeX) | The only document in-repo with **genuinely embedded** font programs — 12 Type1 `/FontFile`, plus `/FontFile3` CFF including 2 `CIDFontType0` (CID-keyed CFF). Every other fixture uses non-embedded base-14 fonts, which exercise no glyph decoder at all. |

## Why the embedded-font distinction matters

It is easy to run a rendering comparison across the other fixtures, watch it
come back byte-identical, and conclude a font change is safe. That conclusion
would be **worthless**: `radio-form.pdf`, `ink-annots-*.pdf` and the
`scripts/token-harness/fixtures/*.pdf` set contain **zero** `/FontFile*`
references, so no font program is ever parsed and neither the from-spec
decoders nor skrifa run. A null result there measures nothing.

This happened during gh-91 and cost a round of false confidence. If you are
verifying anything touching `glyph*.rs`, check the document actually embeds
fonts first:

```sh
strings FILE.pdf | grep -c FontFile
```

## Provenance

Supplied by the maintainer for this purpose, committed to git with their
explicit say-so and their explicit instruction that it must **not** ship to
crates.io.
