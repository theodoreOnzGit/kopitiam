# Engineering Journal

The running log of **discoveries, engineering insights, and hard-won knowledge**
that don't belong in an AID (which records a *decision*) or in a crate's rustdoc
(which sits at the point of use), but which the next person — human or agent —
would waste hours re-deriving.

`CLAUDE.md` mandates this journal: *"Maintain an engineering journal documenting
discoveries, translation insights, format and parsing knowledge, and
architectural rationale."* Knowledge endures; this is where the reasoning lands
so it doesn't evaporate into chat history.

## How to use

* One file per entry, named `YYYY-MM-DD-slug.md`.
* Newest insight, not newest edit — an entry is dated when the knowledge was won.
* Cross-link the AIDs / beads / crates it relates to.
* Singlish register, technical precision preserved (workspace rule).

## Entries

| Date | Entry | About |
| --- | --- | --- |
| 2026-07-18 | [Progressive (anytime) context assembly + the self-improving resource probe](2026-07-18-progressive-context-and-self-improving-probe.md) | Two patterns from the hybrid-AI design (epic `kopitiam-ckv`): resource-adaptive context that is usable-immediately-refined-continuously, and a probe that learns its own constants every run |
| 2026-07-18 | [The constrained-oracle + human-judgment-rung pattern, and the capability-tiered model ladder](2026-07-18-constrained-oracle-human-judgment-and-the-model-ladder.md) | Two more patterns from the §10–12 TUI expansion (epic `kopitiam-ckv`): how to get reliable tool use out of a 0.5B (Rust drives, model is a constrained oracle, human owns judgment) and one binary whose model rungs the resource probe lights up per device |
