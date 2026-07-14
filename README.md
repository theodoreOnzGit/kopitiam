# kopitiam
A Rust-first knowledge, translation, and documentation workbench that preserves structured knowledge, accelerates code translation, and reduces dependence on AI.

## CLI quick start

```bash
cargo install kopitiam
```

Scan a Rust project and see what the Semantic Runtime learned about it —
real facts from `cargo metadata`, not a guess from reading source:

```bash
cd your-rust-project
kopitiam scan
```

Check back in later, from a fresh shell, without re-explaining anything:

```bash
kopitiam status
```

Rename a symbol project-wide via a live rust-analyzer. This only *previews*
a diff until you add `--apply`:

```bash
kopitiam rename src/lib.rs --line 0 --character 7 --new-name my_new_name
kopitiam rename src/lib.rs --line 0 --character 7 --new-name my_new_name --apply
```

Turn a PDF into semantic Markdown:

```bash
kopitiam pdf2md paper.pdf
```

That's the whole surface for now. For every flag, every subcommand
(`code-actions` included), and the reasoning behind the safety defaults —
written for humans and AI agents alike — see
[`apps/cli/README.md`](apps/cli/README.md). That page is also what
crates.io shows once the `kopitiam` package is published.

---

# KOPITIAM

> **K**nowledge-**O**riented **O**pen-source **P**latform for **I**ntelligent **T**ranslation, **I**ntegrated **A**nalysis and **M**odelling

*An AGPLv3 knowledge, translation, and documentation workbench built in Rust.*

---

## Vision

KOPITIAM is a personal, local-first open-source platform for knowledge management, code translation and understanding, documentation, and working with the maintainer's own corpora.

It is **not** another editor.

It is **not** another AI wrapper.

It is a **knowledge, translation, and documentation workbench** whose purpose is to preserve structured knowledge and make document, translation, and knowledge work more productive, reproducible, and sustainable.

The editor is only one interface into the platform.

The Knowledge Engine is the heart of the project.

---

## Philosophy

### AI is an accelerator, not a dependency.

Cloud AI should make the platform faster.

Cloud AI must never be required for the platform to remain useful.

When possible, KOPITIAM should prefer:

1. Existing knowledge
2. Native Rust algorithms
3. Local AI
4. Cloud AI

The goal is simple:

> **Running out of AI tokens should never stop knowledge work.**

---

### Own your knowledge.

KOPITIAM is built around the idea that a person should own their workflow.

That means:

* owning structured knowledge
* owning documentation
* owning translation rules
* owning semantic abstractions
* owning the software stack

Knowledge should accumulate over time instead of disappearing into AI conversations.

---

### Pure Rust Core

KOPITIAM is committed to a Pure Rust Core.

Core components should compile with stable Rust using Cargo.

The project avoids mandatory C, C++, or Fortran dependencies wherever practical.

Optional integrations may exist, but the platform's essential capabilities should remain portable, reproducible, and easy to build.

```bash
cargo build
```

should be enough.

---

## Personal Theological Repository

Alongside its knowledge and documentation mission, KOPITIAM also serves as the author's personal theological repository: a Rust-native home for biblical text corpora (Hebrew Old Testament, Greek New Testament and Septuagint), morphological data, and study tooling, built under the same Pure Rust Core and knowledge-preservation principles as the rest of the platform.

---

## Long-Term Goals

KOPITIAM aims to become a unified environment for knowledge work.

Eventually it should support:

### Developer Tooling

* Rust-first development
* Crate scaffolding
* Semantic code indexing and navigation
* Refactoring through a live language server
* Testing and benchmarking

### Code Translation

Understand and translate legacy software into idiomatic Rust, including:

* C
* C++
* Fortran
* Visual Basic
* C#
* Python

Translation should preserve program intent rather than syntax, producing
idiomatic Rust rather than a mechanical transliteration.

---

### Knowledge Management

Manage:

* papers
* books
* reports
* notes
* references
* translation memory
* structured facts
* provenance

Knowledge should become searchable, structured, and reusable.

---

### Literature Management

Support:

* PDF import
* OCR
* Markdown conversion
* equation extraction
* table extraction
* figure extraction
* plot digitization
* BibTeX generation
* Typst
* LaTeX

Literature should become part of the knowledge workflow instead of remaining static documents.

---

## Architecture

KOPITIAM is built around reusable engines rather than user interfaces.

```text
                KOPITIAM

        ┌─────────────────────┐
        │  Knowledge Engine   │
        └──────────┬──────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
Semantic     Literature     Translation
 Engine         Engine          Engine
    │              │              │
    └──────────────┼──────────────┘
                   │
           Document Engine
                   │
      ┌────────────┼────────────┐
      │            │            │
     CLI          TUI          GUI
                   │
      Neovim-Compatible Editor
```

The interfaces are clients.

The engines own the functionality.

---

## Project Principles

* Knowledge over chat history.
* Semantic understanding over text processing.
* Rust-first architecture.
* Offline-first workflows.
* Strong typing.
* Explicit APIs.
* Modular crates.
* Long-term maintainability.
* Correctness over convenience.

---

## License

Copyright (C) 2026 Theodore Kay Chen Ong.

KOPITIAM is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.

The intention is to keep the core platform free, open, and community-owned.

### Why AGPLv3, specifically?

This is a deliberate choice, not a default. AGPLv3 is chosen because:

* **It is copyleft.** Anyone who builds on KOPITIAM must share their
  modifications under the same license. Improvements flow back to
  everyone, not just whoever made them.
* **It closes the network loophole plain GPL leaves open.** Ordinary GPL
  only requires sharing source when you *distribute* the software. AGPL
  also requires it when you run a modified version as a network service
  others interact with — so KOPITIAM (or a derivative of it) can't be
  quietly turned into a closed, hosted product without releasing the
  source of what's actually running.
* **Nobody gets to take it away.** The goal is that KOPITIAM, and
  everything built on it, stays open source, permanently. AGPLv3 is the
  strongest widely-used copyleft license available for that purpose.

GPLv3 and AGPLv3 also each carry an explicit mutual-compatibility clause
(Section 13 in both), letting a GPLv3-or-later work be combined into an
AGPLv3 project should the need ever arise.

KOPITIAM is a personal project of Theodore Kay Chen Ong, developed in his
personal capacity using his own resources outside of working hours. It is
not affiliated with, or a work product of, his employer. See
[`NOTICE`](NOTICE) for the full statement.

---

## Current Status

KOPITIAM is in its early design phase.

The initial focus is establishing:

* project architecture
* core workspace
* Knowledge Engine
* Literature Engine
* Translation Engine
* Document Engine

Interfaces such as the editor and GUI will be built on top of these foundations.

---

## Contributing

Contributors are welcome from many disciplines, including:

* Rust
* Compilers and language tooling
* Program translation and porting
* Document and literature processing
* Information retrieval and knowledge management
* Technical writing and documentation
* Natural language processing

Please read the documentation in the `docs/` directory before contributing.

---

## Acknowledgements

KOPITIAM is inspired by the idea that knowledge should accumulate rather than disappear.

A kopitiam is a place where people gather to exchange ideas, solve problems, and learn from one another.

This project aims to provide the same kind of environment for knowledge work.
