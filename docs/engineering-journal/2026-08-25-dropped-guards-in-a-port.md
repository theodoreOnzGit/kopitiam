# The guard you drop when porting is the one you never had a fixture for

**Date:** 2026-08-25
**About:** GitHub issue [#70](https://github.com/theodoreOnzGit/kopitiam/issues/70), bead `bd-nnk`, `crates/kopitiam-pdf/src/mupdf/xref.rs`
**Related:** [Silent wrongness, and why only a reference oracle finds it](2026-07-28-silent-wrongness-and-the-reference-oracle.md)

## What happen

Open one PDF, ask it the same question two ways, get two different answers:

```
A (resolve straight away):  /Annots array_len=8
B (rasterize first, then resolve): /Annots array_len=3   <-- wrong
```

Same file, same process, same public API. Only the *order of access* different.
The user add annotations in Okular, save, then kopitiam-pdf render the page and
report the pre-annotation structure. No error, no warning, nothing. Just quietly
the old answer.

## The root cause, and why it hide so well

Three separate things, each one correct by itself, combine into a bug:

1. **Incremental update.** Okular/Acrobat don't rewrite the file when you save
   annotations — they append the changed objects plus a new xref section chained
   back via `/Prev`. The superseded object is still sitting there in the original
   body, byte for byte. Nothing erase it.
2. **Object streams don't get rewritten either.** If the superseded object was
   packed inside an `/ObjStm`, that stream *still contains the old copy*. You
   cannot tell from the stream alone which of its objects are still current —
   only the xref knows (PDF 32000-1:2008 §7.5.4, §7.5.8.3).
3. **The cache is keyed by object number and first-write-wins.** `load_obj_stm`
   must decompress the *whole* stream to reach any one object, and our port then
   cached every object it parsed out — unchecked.

So: rasterize touch some unrelated object that happen to live in the same
objstm → whole stream decompress → stale copy of the `/Annots` array land in the
cache first → every later `resolve()` serve the stale one. Resolve *before*
rendering and the correct copy get there first, which is why order decide the
answer.

MuPDF guard exactly this, in `pdf_load_obj_stm` (`source/pdf/pdf-xref.c:2213`):

```c
if ((entry->type == 'o' || entry->type == 'O') && entry->ofs == num)
{ ... store it ... }
else
    pdf_drop_obj(ctx, obj);   /* xref no longer source this object here */
```

Our port drop that `else`. One branch, and it look like defensive noise if you
never seen an incrementally-updated file.

## The lesson (this is the part worth keeping)

**When you port, the branch that look redundant is often the one carrying a case
your fixtures don't have.** Every objstm fixture in this crate is a
single-generation file written in one pass. In that world the guard is a
tautology — every object in stream 4 *is* claimed by stream 4, always. The guard
only earn its keep the moment a second generation exist, and no test had one.

So the shape of the mistake is not "we misread the C". We read it fine. We
dropped a condition that was *unfalsifiable against our own test corpus*, and
the corpus agreed with us for months.

Practical rules that follow:

* **A dropped condition needs a written reason, same as a magic number needs a
  source.** "Simplified away" is not a reason. If upstream check something we
  don't, say in the rustdoc why we don't need to — then the next person can see
  the argument is wrong instead of guessing there was one.
* **Order-dependent answers are always a bug**, never a quirk. `resolve()` after
  `rasterize()` returning something different from `resolve()` before it is the
  cleanest possible signal that a cache got poisoned. Worth testing for
  deliberately: ask the same question two ways round and assert they agree.
* **Generational fixtures are cheap.** The regression test
  (`incremental_update_supersedes_object_stream_copy`) is ~90 lines of hand-built
  PDF and reproduce the real-world failure exactly — 200 instead of 400, the same
  shape as 3 instead of 8. Any format with a supersede/append mechanism (PDF
  incremental update, log-structured stores, git packfiles) deserve at least one
  fixture where generation 2 contradict generation 1.

## Severity note

The reported symptom was cosmetic — annotations invisible. Same mechanism apply
to annotations used as **redactions**, where silently rendering the pre-redaction
structure is much worse than an error. A cache that serve stale objects doesn't
know what the object *means*, so "how bad" depend entirely on which object got
superseded. Treat any stale-read path as high severity by default.
