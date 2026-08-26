# Architecture Decision Records

Decisions the **project** made about its architecture, and why. Permanent
history — an ADR is never deleted, even when a later one overturns it.

`CLAUDE.md` has asked for ADRs since the beginning; this directory is where
they finally live. It sits beside [`../ai-decisions/`](../ai-decisions/), and
the split is about *who decided*, not how important it was:

| | Who decided | What it records |
| --- | --- | --- |
| **ADR** (here) | The maintainer, or the project as a whole | What was decided and why, permanently |
| **AID** ([`../ai-decisions/`](../ai-decisions/)) | An AI, autonomously, in the maintainer's absence | That a judgment call was made *without the maintainer*, so they can confirm or reverse it |

An AID is a thing awaiting review. An ADR is settled. When the maintainer
confirms an AID that turned out to be architecturally load-bearing, it is fair
to promote it to an ADR — but the AID stays where it is, marked confirmed.

## Status values

| Status | Meaning |
| --- | --- |
| **Accepted** | In force. |
| **Superseded** | Overtaken by a later ADR (which must be named). |
| **Reversed** | Tried, then undone, because it turned out wrong. |

## Index

| ID | Decision | Status |
| --- | --- | --- |
| [ADR-0001](ADR-0001-ollama-client-not-port.md) | Talk to `ollama serve` as a client; abandon the in-process port | Accepted |
