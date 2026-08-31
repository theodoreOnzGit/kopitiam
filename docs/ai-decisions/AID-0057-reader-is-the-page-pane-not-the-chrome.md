# AID-0057 — The embeddable reader is the page pane, not the whole chrome

**Status:** Accepted — but see **Correction** below: one of the two reasons
given for it was factually wrong. The decision survives on the other.
**Date:** 2026-08-31 (SGT)
**Bead:** `bd-rbn` · **GitHub:** gh-96
**Affects:** `kopitiam-pdf::gui_frontend`, phases 5–6 and 12 of the gh-96 brief

---

## Correction (2026-08-31 SGT, same day)

**The load-bearing technical claim below is false, and this section is the
correction rather than a rewrite, because an AID is never quietly edited into
looking right.**

I wrote that "an `egui::Panel` cannot be created inside a `Ui` — panels are
`Context`-level". That is true of egui's older `SidePanel`/`TopBottomPanel`
API. It is **not** true of egui 0.36, which this workspace pins. Checked
against the vendored source at
`egui-0.36.1/src/containers/panel.rs:420-422`:

```rust
impl Panel {
    /// Show the panel inside a [`Ui`].
    pub fn show<R>(self, ui: &mut Ui, add_contents: impl FnOnce(&mut Ui) -> R) -> InnerResponse<R>
```

`kpdf.rs` has been calling it that way all along (`egui::Panel::top("kpdf-status").show(ui, ..)`),
which is what prompted the check. I should have read the API before asserting
a constraint from it.

**Does the decision change? No — but its support does.** It rested on two
legs, and only one was real:

* ~~Panels cannot live inside a `Ui`, so the reader *cannot* own its chrome.~~
  **False.** It can.
* **The host should own layout.** Still true, and still the actual reason.
  A reader that builds its own panels is a *window*, not a widget: it claims
  screen space the host may have promised elsewhere, and it cannot be placed
  in a host's own split. kovan wants the reader *beside* its literature
  tooling, not instead of it.

So this stays a design preference, not a technical necessity — a weaker
justification than the original text claims, and the maintainer chose between
the options partly on the false premise. Flagged to them explicitly rather
than left buried here.

One thing the correction *gains*: because `show(ui)` is possible after all,
"reader owns its chrome" is no longer an either/or. The page pane stays the
primitive, and an all-in-one convenience that assembles the standard chrome
can be offered **on top of it** for hosts that just want kpdf's layout. That
is strictly better than either original option, and is what was built.

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

**A. The reader builds its own panels.**
Closer to today's `KpdfApp::ui`, and less work now: kpdf's existing panel code
would move almost verbatim. Rejected *as the only entry point* because it makes
the reader a *window* rather than a *widget*. A host cannot put it in its own
split, cannot place the sidebar on the other side, and cannot combine it with
its own chrome — which is precisely what kovan needs (a reader beside its
literature tooling, not instead of it). It also silently claims screen real
estate the host may have promised to something else.

(Originally rejected as *impossible* as well; see the Correction — it is
possible, and is now offered as a convenience layered over the pane API rather
than as the only option.)

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
* ~~**If egui gains nested panels**, the constraint that forced this
  evaporates.~~ It already had them; see the Correction. What remains of this
  point is its tail, which was right: the host-owns-layout argument stands on
  its own, and is now the whole case.
* **The lesson worth keeping**, beyond this decision: I asserted an API
  constraint from memory of an older egui, in a document whose entire job is
  to be trustworthy later. The workspace pins its dependencies and vendors
  sources precisely so a claim like that can be checked in one grep. Check
  first; a confident wrong premise in an AID outlives the session that wrote
  it.
