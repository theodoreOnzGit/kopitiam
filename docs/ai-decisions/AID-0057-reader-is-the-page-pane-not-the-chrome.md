# AID-0057 — The embeddable reader is the page pane, not the whole chrome

**Status:** Accepted (maintainer's call, taken in-session)
**Date:** 2026-08-31 (SGT)
**Bead:** `bd-rbn` · **GitHub:** gh-96
**Affects:** `kopitiam-pdf::gui_frontend`, phases 5–6 and 12 of the gh-96 brief

---

## Context

gh-96 wants `kpdf`'s reader lifted out of `src/bin/kpdf.rs` into the library so
kovan (and any other egui app) can embed **the same** reader instead of copying
2900 lines of `KpdfApp`. The brief sketches the API as:

```rust
let output = reader.ui(ui);
```

That one line hides a real design fork, and egui is what forces it: **an
`egui::Panel` cannot be created inside a `Ui`.** Panels are `Context`-level —
`SidePanel::show(ctx, ..)`. So a reader handed only a `&mut Ui` physically
cannot build kpdf's thumbnail sidebar, outline sidebar, or find bar. Something
has to give.

## Decision

**The reusable reader is the document pane only.**

* `reader.ui(ui)` paints the pages, and nothing else — it takes a `&mut Ui`
  and stays inside it.
* Thumbnails and outline are exposed as **separate methods the host calls into
  a `Ui` it owns** — `reader.thumbnail_sidebar(ui)`, `reader.outline_sidebar(ui)`
  — so the host decides whether those live in a `SidePanel`, a collapsing
  header, a tab, or nowhere at all.
* The reader still owns the *engines* behind them: thumbnail scheduling and
  caching, outline loading, search. What the host owns is **layout**.

## Alternatives considered

**A. `reader.show(ctx)` — the reader builds its own panels.**
Closer to today's `KpdfApp::ui`, and less work now: kpdf's existing panel code
would move almost verbatim. Rejected because it makes the reader a *window*
rather than a *widget*. A host cannot put it in its own split, cannot place the
sidebar on the other side, and cannot combine it with its own chrome — which is
precisely what kovan needs (a reader beside its literature tooling, not
instead of it). It also silently claims screen real estate the host may have
promised to something else.

**B. Reader takes a `Ui` and emits layout *requests* the host fulfils.**
Most flexible, and keeps one entry point. Rejected as over-engineering for a
problem two explicit methods already solve: it invents a layout protocol,
which is more API surface to document, version and get wrong than
`thumbnail_sidebar(ui)`.

## Consequences

* `kpdf`'s `eframe::App::ui` keeps its `Panel` calls — those are standalone
  application policy, exactly where the brief's Phase 12 says they belong.
* Phase 5 (thumbnails) is shaped by this: the library must expose *request /
  fetch-cached / is-pending*, since the host may paint the sidebar itself and
  never call our method at all.
* The find bar is chrome by the same argument, so it stays in the host; the
  library owns the search worker and the scan state, which it already does
  as of `gui_frontend::search`.

## What would make this wrong

* **If every realistic embedder wants kpdf's exact chrome.** Then two extra
  methods per surface is friction for no gain, and option A's single
  `show(ctx)` would have been the kinder API. The signal would be kovan
  reimplementing a sidebar that looks identical to ours.
* **If the panes turn out to be coupled** — if painting the page pane
  correctly requires knowing the sidebar's width or scroll state, the split
  leaks and the host ends up threading state between two calls that pretend to
  be independent. Watch for the first parameter that exists only to tell one
  method what the other did.
* **If egui gains nested panels**, the constraint that forced this evaporates
  and option A becomes cheap again. Worth re-reading then, though the
  host-owns-layout argument would still stand on its own.
