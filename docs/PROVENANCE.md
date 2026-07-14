# Provenance

*A contemporaneous origin note for KOPITIAM. Last reviewed 2026-07-15.*

This document states plainly what KOPITIAM is and how it is built, so the facts
are on the record rather than only inferable from commit metadata. It is a
project origin note, not a legal document.

## What KOPITIAM is

KOPITIAM is a personal, single-maintainer, open-source workbench for knowledge
management, code translation, and documentation. It is a hobby project.

Its subject areas are the maintainer's own domains, technical and personal:

* **Documents and literature** — turning PDFs, Markdown, DOCX and HTML into
  structured, searchable knowledge (parsing, OCR, equation and table extraction,
  plot digitization, bibliography and reference management).
* **Code translation and understanding** — reading legacy sources (C, C++,
  Fortran, Visual Basic, C#, Python) and translating them into idiomatic Rust,
  with semantic code indexing and navigation.
* **Personal-interest corpora** — the maintainer's own knowledge work, including
  bible study, health, housing/finance planning, insurance and legal documents.
* **Developer tooling** — the platform's own engines and interfaces, including
  the `kvim` editor (`kopitiam-neovim`) and the `kmux` terminal multiplexer.

The through-line is a local-first knowledge engine that turns documents, code,
and notes into structured, durable knowledge. The editor and other interfaces
are clients of that engine, not the point of the project.

## How it is built

KOPITIAM is built by one person, on that person's own personal time and personal
hardware, under that person's personal identity. Git authorship is a personal
machine and a personal email address, not any organizational or work address.

The repository contains no third-party confidential material and no data
belonging to any employer or other organization. Its content is the maintainer's
own original work, together with clearly-credited open-source upstreams (see
below).

Development is committed to a **Pure Rust Core**: the platform builds with stable
Rust and Cargo, without a mandatory C, C++, Fortran, or CMake toolchain. Optional
integrations are allowed, but the core stays Cargo-buildable.

## Licensing posture

KOPITIAM is licensed **AGPL-3.0-only**, in its entirety, without exception. Every
crate carries the same license.

Everything KOPITIAM forks, adapts, vendors for reference, or studies is credited
to its upstream authors in [`ACKNOWLEDGEMENTS.md`](ACKNOWLEDGEMENTS.md), with each
upstream's own license recorded and honored. Permissively-licensed upstreams
(MIT, Apache-2.0, BSD) are incorporated with their notices retained; the bundled
font ships under OFL-1.1 as a distinct work alongside the program. Forked code
retains its upstream copyright notices and license texts.

## Provenance record

The authoritative, timestamped record of authorship is the project's **git
history**. Development began on or around **13 July 2026** (the date of the
earliest commit), and every change since is attributed to its author with a
timestamp there.

Design decisions made in the course of development — including decisions an
autonomous agent made on the maintainer's behalf — are recorded as dated
decision records under [`ai-decisions/`](ai-decisions/). Together, the git
history and the decision records are the project's own account of how it came to
be.
