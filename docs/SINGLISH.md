# SINGLISH.md — KOPITIAM's Singlish style guide (living doc)

This is the maintainer's Singlish register for KOPITIAM, written down so it
**endures** instead of evaporating into chat. It grows: when the maintainer
teaches a word, a phrase, a usage, or corrects something, it gets **logged here**
(see [Lessons from the maintainer](#lessons-from-the-maintainer) at the bottom).

Every agent and session working on KOPITIAM reads this to keep the register
consistent — so the READMEs, AIDs, journal, and doc comments all sound like the
same kopitiam-shop, not like some drift back into plain textbook English.

## The one rule that overrides everything

**Register is Singlish. Precision survives.** Singlish is *how* we say it, never an
excuse to be vague. Every API contract, safety constraint, unit, ownership rule,
and "what would make this wrong" must stay **exactly as unambiguous** as plain
English would be — just said in Singlish. If a point cannot be made precise in
Singlish, make it precise first, Singlish-flavour second. (This is why the AIDs
lean careful — but careful ≠ plain; keep the flavour, hor.)

Write **natural, genuine** Singlish — not a caricature, not mockery. If it reads
like somebody making fun of the accent, it's wrong. It should read like a
Singaporean engineer explaining something to a colleague over kopi.

## Particles (the load-bearing ones)

Sentence-final particles carry tone, not meaning — use them the way a native
speaker would, sparingly and for effect, not sprinkled on every line.

| Particle | Does what | Example |
|---|---|---|
| **lah** | assertion / mild emphasis / "come on" | "The build pass already lah." |
| **leh** | soft question / mild contrast / seeking agreement | "But this one slower leh?" |
| **lor** | resignation / "that's just how it is" | "No GPU driver, fall back to CPU lor." |
| **hor** | seeking agreement / "right?" / flagging a caveat | "This one must pin the version hor, else break." |
| **sia** | exclamation / emphasis (stronger) | "Wah, 900 tests pass sia." |
| **ah** | softener / list-marker / gentle question | "Open the file first ah, then run." |
| **meh** | skeptical question ("really?") | "Can compile on Android meh? Turns out can." |
| **liao** | already / completed | "Published liao." |
| **mah** | "obviously / as you'd expect" | "Of course got test mah." |

## Loanwords + Singlish vocab already used in KOPITIAM

| Word | Origin | Meaning | Used in-repo for |
|---|---|---|---|
| **kamsia** / **kum siah** | Hokkien | thank you | sign-offs, acknowledgements |
| **shiok** | Malay-ish | great, satisfying | "works shiok" |
| **chope** | Singlish | reserve / hold a resource | `chope_read_only()`, "chope behind a feature" |
| **kaypoh** | Hokkien | nosey / busybody | a nosey full-scan (`kaypoh_scan` idea) |
| **makan** | Malay | eat / consume | a fn that consumes/eats input |
| **kopi** | Malay | coffee | the shop identity; `kopi>` chat prompt |
| **swee** | Hokkien | nice, smooth, well-done | "green tests, swee" |
| **steady** | Singlish | solid / reliable / "nice one" | approval |
| **sian** | Hokkien | bored / weary / "sigh" | tedious work |
| **paiseh** | Hokkien | embarrassed / "sorry to trouble" | apologising for a mistake |
| **atas** | Malay | high-class / fancy | over-engineered |
| **bo jio** | Hokkien | "didn't invite me" | (banter only) |

## Grammar patterns (natural Singlish, keep it readable)

- **"Got" = have / exist / did.** "Got test or not?" · "This crate got no internal deps."
- **"Can" = yes / able / OK.** "Publish 0.1.7 can?" — "Can." · "This one can work on Termux."
- **Topic first, comment after.** "This function hor, it never checks the bound."
- **Drop the copula / articles where natural.** "The build clean." · "Terminal froze because still in Terminal mode."
- **Reduplication for emphasis / small.** "small small change" · "check check first."
- **"Already" / "liao" for completed.** "Merged already." · "Reinstalled liao."
- **"Right / correct?" → "hor / is it".** "Must reap the child hor."

Don't force a particle onto every sentence — that's the caricature trap. One well-
placed *lah* or *hor* per few sentences is plenty; the *grammar* and *vocab* carry
the register more than the particles do.

## Function / identifier names

Singlish identifiers are welcome **when they fit the use case** — `chope()` to
reserve, `kaypoh_scan()` for a nosey full-scan, `makan_` prefix where apt. Must
still be a valid, readable Rust identifier; never force a Singlish name where it
makes the code *harder* to read. Judgment call each time.

## Published-crate caveat (worth the maintainer's eventual call)

Crates published to crates.io render their rustdoc on docs.rs and their README on
the crate page — the public, international face. Full Singlish there may lose
overseas readers. Default for now: Singlish everywhere. If the maintainer later
wants published-crate *public API* docs kept in plainer English for reach, that's
a scope refinement — until then, this rule is everywhere.

---

## Lessons from the maintainer

> Log format: each entry is a dated thing the maintainer taught or corrected —
> the word/phrase, what it means, and how to use it right (with a wrong-vs-right
> example if there was a correction). Newest at the top. Append here whenever the
> maintainer teaches Singlish; never overwrite an old lesson.

<!-- Nothing logged yet — teach away, boss, and it lands here. -->
