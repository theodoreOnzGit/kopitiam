# AID-0018: kopitiam-bibliography — how an unparseable name is handled, where the network seam goes, and why a fifth provenance model exists

* **Status:** Pending review
* **Bead:** `kopitiam-bjo`
* **Date:** 2026-07-14
* **Decided by:** AI (Claude), maintainer absent

## The brief

> Build `crates/kopitiam-bibliography` — KOPITIAM's bibliography and reference
> engine. [...] A citation is a CLAIM ABOUT PROVENANCE, and a wrong one is an
> academic-integrity problem, not a bug. **NEVER FABRICATE A REFERENCE.**

Four decisions in it were genuinely the maintainer's and he was not there to make
them. They are recorded below in descending order of how much damage the wrong
call would do.

---

## Decision 1: a name we cannot split is parsed anyway — but the split is never emitted

**This is the one to scrutinise.** The brief was explicit:

> **when you cannot confidently parse a name, KEEP IT VERBATIM rather than
> mangling it.** Mangling a researcher's name in their own citation is a real
> harm.

### The problem the brief's instruction runs into

`Kim Jong-un` and `John Smith` are **the same shape**: two capitalised tokens,
nothing else to go on. So are `Mao Zedong` and `Alan Turing`. There is no rule —
none, not a cleverer regex, not a longer particle list — that separates them from
the string alone. It requires a lexicon of personal names, which this crate does
not have and cannot acquire offline.

So "keep it verbatim when you cannot confidently parse it" has a literal reading
that destroys the crate: refuse to split *any* two-capitalised-word name, and
`John Smith` — the single most common shape in the entire corpus — yields no
family name, cannot be alphabetised, cannot be keyed, and cannot be matched to an
author-year citation. The bibliography stops working.

### What I did instead

Split it, **label the split as an assumption, and never let the assumption out of
the crate.**

`NameConfidence` has three values:

| Value | When | Trustworthy? |
|---|---|---|
| `Explicit` | the source used a comma — `Waals, J. D.` — and *told* us where the family name ends | yes: no convention was assumed |
| `Conventional` | the *shape* carries the information: an initial (`A. B. C. Smith`), a lower-case particle (`van der Waals`) | yes: the rule is what the shape means |
| `Assumed` | two plain capitalised words, nothing to disambiguate — `John Smith`, `Mao Zedong` | **no** |

And then four rules make an `Assumed` split harmless:

1. `PersonName` **always retains the source literal**. `as_written()` returns the
   name byte-for-byte, whatever we concluded. Nothing is ever destroyed.
2. `Author::family()` returns **`None`** for an `Assumed` name. Nothing downstream
   can misattribute, because nothing downstream is given a family name to
   misattribute with.
3. `Author::sort_key()` falls back to the name **as written**, so `Mao Zedong`
   sorts under *M* — which is right — rather than under a surname we guessed.
4. **BibTeX and Hayagriva emission never reorder an `Assumed` name.** `Mao Zedong`
   goes into a `.bib` file as `Mao Zedong`, never as `Zedong, Mao`.

Rule 4 is the load-bearing one. Reordering would put `Zedong, Mao` in a
bibliography, which BibTeX then typesets as *"M. Zedong"* — **a real person,
renamed, in a published document, by us.** That is the harm. The internal split
being wrong is not; it is invisible and inert.

So: **we may be wrong internally; we are never wrong in public.**

Names we refuse outright and keep as `Author::Literal`:
* anything in Han, Hangul or Kana script — the ordering convention depends on the
  *venue*, not the characters, and the characters do not say which;
* anything a source brace-protected as `{{...}}` — BibTeX's own "do not touch
  this" idiom, honoured literally;
* corporate authors (`Argonne National Laboratory`), which a person-name grammar
  turns into "Laboratory, Argonne National".

### What would make this wrong

* If the maintainer's view is that an `Assumed` split should not be *computed* at
  all — that `John Smith` should have no family name in the API — then rules 2-4
  are insufficient and `PersonName::family()` should be `Option` at the type level
  rather than at the accessor. That is a one-line change and I would make it on
  request; I judged it would make the common case unusable for no additional
  safety, since the split is already unavailable to every consumer.
* If a name lexicon ever becomes acceptable (a bundled CJK-surname list is small
  and offline), `Assumed` could be narrowed considerably. Filed as
  `kopitiam-bjo.1`.

---

## Decision 2: no `kopitiam-web` dependency — the resolution seam is a trait and a string

The brief said the crate has no network and must "leave a clean seam where
`kopitiam-web` [...] could later resolve identifiers. **Do not stub a fake
resolver that returns plausible metadata.**"

I read "clean seam" as *not* "depend on `kopitiam-web`". The crate depends on
`kopitiam-ontology`, `kopitiam-pdf` and `kopitiam-document`, and on nothing that
can open a socket.

The seam is:
* `ReferenceResolver` — a trait, mirroring `kopitiam_web::SearchProvider` and
  `kopitiam_ai::ModelAdapter`;
* `NullResolver` — which **errors** (`ResolveError::Disabled`), exactly as
  `kopitiam-web`'s `NullProvider` errors rather than returning an empty result
  set, and for the identical reason: *"I did not look"* and *"there is nothing
  there"* are different sentences;
* `ResolutionRequest::search_text()` — the query string a future resolver would
  hand to CrossRef or to `kopitiam_web::SearchQuery::new(...)`. One line, in a
  crate above this one.

There is **no** `MockResolver`, and that is deliberate: a stub returning plausible
DOIs would be indistinguishable, to every caller and to the knowledge graph, from
a real one. It would compile, its tests would pass, and it would put fabricated
identifiers — pointing at other people's papers — into a scientist's bibliography.

`ResolutionOutcome` also forces a resolver to return **the evidence, not just the
answer** (`candidate_title`, `candidate_first_author`, `candidate_year`), so that
whoever writes the real resolver must check the candidate against what the
document actually printed before attaching a DOI. A search engine returning a
paper with a similar title is not evidence that it is the same paper.

**What would make this wrong:** if the maintainer wants `kopitiam-bibliography` to
*be* the literature engine rather than its reference layer, the dependency should
go in and `kopitiam-literature` (currently a 14-line stub) should be deleted. I
assumed the layering implied by that stub's existence.

---

## Decision 3: a fifth provenance model, following the insurance pattern — and it should be hoisted

The brief said: *"read `crates/kopitiam-insurance/src/` for the pattern [...]
Do not invent a fourth provenance model."*

I followed the **pattern** exactly — private fields, exactly one constructor, no
`Default`, `#[serde(try_from)]` so deserialisation cannot smuggle in an un-sourced
value — but the **components differ**, and had to:

* `kopitiam-insurance`'s `Provenance` requires a `ClauseId` and a `SectionPath`. A
  bibliographic reference has neither.
* A reference read out of a `.bib` file has a **line**, not a page. So `Locator` is
  an enum (`Page` | `Line`) rather than an `Option<PageNumber>` — encoding a
  file-sourced reference as a page-sourced one whose page we *lost* would be a lie
  about the quality of our own knowledge.
* `Provenance` carries **two** strings: `verbatim` (the printed text, line breaks
  intact — what a reader checks against the page) and `normalised` (the
  de-hyphenated, line-joined text the parser actually consumed — what a reader
  checks against *us*). Reference-list entries are wrapped and hyphenated, so
  these genuinely differ, and keeping only one loses something.

Depending on `kopitiam-insurance` from a bibliography engine would have been
architecturally absurd. So this is the fifth hand-rolled provenance model in the
workspace (insurance, legal, health, web, now bibliography), and **that is the
real finding**: the pattern is right and its repetition is a smell.

**Recommendation, filed as `kopitiam-bjo.2`:** hoist a generic
`Provenance`/`SourceText`/`DocumentId` into `kopitiam-ontology`, which is exactly
where shared vocabulary is supposed to live ("pure data, no logic, no storage").
Domain crates would parameterise the locator. I did not do it because
`kopitiam-ontology` was explicitly off limits to this task.

---

## Decision 4: `RelationshipKind::Custom("cites")`, under protest

"Paper A cites paper B" is the edge the whole crate exists to produce, and
`kopitiam-ontology` has no `Cites` variant.

`kopitiam-ontology`'s own rustdoc, in the docs for `RelationshipKind::Inherits`,
records what happens next: four language adapters were written concurrently, each
reached for a *different* encoding of inheritance (`Custom("inherits")`,
`ImplementedBy`, nearly `DependsOn`), and the shared vocabulary was quietly
defeated. `Inherits` was promoted to a first-class variant precisely so that could
not recur.

**`Cites` is the same case**, and more consequential: it is *the* fundamental
relation of the scientific literature, and if `kopitiam-literature`, a future
citation-analysis tool and a second bibliography importer each invent their own
spelling, the graph cannot answer *"what cites this?"* — which is the single most
valuable question a citation graph exists to answer.

I used `Custom("cites")` because the ontology crate was off limits, and confined
the string to one `const` (`knowledge::CITES`) so there is exactly one line to
change. **Filed as `kopitiam-bjo.3`: promote `RelationshipKind::Cites`.**

---

## What the real paper actually did to this crate

Recorded here because it is the most valuable part, and because it is the
strongest available argument for the practice.

A real published paper found **one fabricated author**. A reference whose
title begins *"Design, fabrication and startup testing..."* — a comma-terminated
word at the head of the title — made the author scanner read `Design` as a fourth
surname. It produced **a researcher who does not exist**, sitting in the `.bib`
file looking exactly as trustworthy as the genuine authors.

Every synthetic test passed. The crate's entire stated purpose is to not do that,
and it did it, and only the real document caught it. The regression test written
for it then immediately exposed a second bug: the indefinite article `A` in
*"A Book About Salt"* was being read as an initial, making a title look like a
name.

This is the same lesson `kopitiam-plot` learned on the same paper in the same
session, and it should probably be written into CLAUDE.md as standing practice:
**a synthetic corpus can only contain the failures you already thought of.**

---

## What would make all of this wrong

* **If a bibliography is expected to be complete rather than honest.** Five of the
  twelve references in the maintainer's own paper are PhD theses that `biblatex`'s
  `ieee` style prints *identically* to books — the "PhD thesis" designator is
  simply dropped. This crate reads them as `@book` and raises an
  `AmbiguousEntryKind` anomaly naming `@phdthesis` as the alternative. If the
  maintainer would rather it guessed `@phdthesis` when the publisher is a
  university, that is a one-line change — but it is a *guess*, it will be wrong for
  university-press monographs, and I judged that a wrong entry type asserted
  confidently is worse than a flagged ambiguity.
* **If `et al.` should be silently completed.** It is not: `AuthorList::is_truncated()`
  records it, and BibTeX emission writes `and others`. Padding the list out would
  require a network.
* **If the eleven references with no DOI should acquire one.** They should not, and
  cannot, offline. `ResolutionRequest::for_reference` marks each of them as work a
  future resolver would do.
