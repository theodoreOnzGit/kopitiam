# AID-0006: how to fork rmux — and why you should probably not fork all of it

* **Status:** **DECIDED by the maintainer (2026-07-14): option (a), the full fork.** They want KOPITIAM to own the whole stack. My recommendation of (c) upstream-first was heard and overruled — which is their call to make, and consistent with the Pure Rust Core philosophy of owning what you depend on.
* **Bead:** `kopitiam-2yg`
* **Date:** 2026-07-14
* **Decided by:** AI (Claude) proposed; **the maintainer ruled**
* **Implemented:** yes — `crates/kopitiam-mux/`, 2026-07-14

## Resolution — what implementing it taught us

The full fork is built and lives at `crates/kopitiam-mux/`. It builds, tests
(1042 passing) and type-checks clean for all three Android ABIs. The
reconnaissance below was mostly right, and wrong in two ways worth recording,
because both would mislead the next person.

**Right:** the gaps genuinely span six crates plus the root binary, so option
(b) — patching `rmux-os` alone — really was impossible. The `cfg`-gate census is
accurate.

**Wrong #1 — the path problem was overstated by 10x.** The claim below that
hardcoded `/tmp` and `/var/run` infest *"nearly every crate, plus the whole
top-level binary"* is **false**. Almost every one of those grep hits is a **test
fixture**. There are exactly **two** production sites that hardcode a runtime
root: `rmux-ipc::endpoint::socket_root_from_parts` and `rmux-sdk`'s
`private_startup_lock_root`. Both now delegate to one resolver. A third
(`rmux-server::tmux_shim::shim_root`) already honoured `XDG_RUNTIME_DIR`/`TMPDIR`
and was *already correct* on Termux. The `FALLBACK_SOCKET_ROOT` constants in
`rmux-client` and `rmux-server` that made the problem look pervasive are both
`#[cfg(all(test, unix))]`.

This is the same failure the AID congratulates itself for catching, committed
one paragraph later: **I counted grep hits and called it a census.** Reading
them changed the estimate tenfold. The fork was still the right call — but the
path problem was never the reason, and if it had been the deciding factor the
decision would have rested on a bad number.

**Wrong #2 — abstract sockets needed no conversion.** Work item 3 below says to
convert abstract unix sockets to filesystem sockets. That work does not exist.
rmux gates abstract sockets on `target_os = "linux"` and already carries a
`not(linux)` filesystem fallback for every one, so **Android gets filesystem
sockets for free** — the correct action was to *leave those gates alone*, the
exact opposite of the plan. A future maintainer doing a well-meaning "widen all
the linux gates" sweep would silently re-enable abstract sockets on Android and
regress this. That is why it is now documented at length in
`rmux_os::runtime_dir`'s module docs, which is the canonical write-up of every
Android decision in the fork.

**What actually cost the effort** was none of the above: it was the binary
rename. `rmux` → `kmux` broke four runtime *file-name* lookups (the tmux shim,
SDK daemon discovery, two helper resolvers) that are strings, not types — so
nothing failed to compile, features just silently stopped working. The names now
live once, in `rmux_os::host`.

Everything below this line is the original, pre-decision record, preserved
unaltered including the parts it got wrong.

---

This AID records a decision that is **scoped but not executed**. The
reconnaissance is done and is solid; the fork itself is not built, because the
cheap option turned out to be impossible and the remaining option is expensive
enough that you should get to choose it rather than discover it.

## The brief

> Fork rmux. It doesn't run on Android. `rmux-os` does not support Android. I
> want it to support Android and all 3 platforms: Windows, Mac and Linux. So,
> fork rmux, and it becomes `kopitiam-mux`, with a binary running as `kmux`.

## The hypothesis I had, and why it was wrong

rmux is **~325,000 lines across 12 crates**. Its OS abstraction layer,
`rmux-os`, is only ~3,350 of them. The obvious cheap play was:

> Fork **only** `rmux-os`, depend on the published rmux crates for everything
> else, and use Cargo's `[patch.crates-io]` to substitute our Android-capable OS
> layer into rmux's dependency graph. Own 3k lines instead of 325k.

That only works if the Android gaps are genuinely *confined* to `rmux-os`. **I
checked. They are not.** Evidence:

**Where the `target_os = "linux"` cfg gates live** (these silently exclude
Android, whose `target_os` is `"android"` even though it is Linux-kernel):

| Crate | Files |
| --- | --- |
| `rmux-os` | 4 |
| `rmux-ipc` | 3 |
| `rmux-client` | 2 |
| `rmux-server` | 1 |
| `rmux-sdk` | 1 |
| `rmux-pty` | 1 |
| the top-level binary (`src/main.rs`, `src/daemon_main.rs`) | 2 |

**Where the Android-hostile primitives live:**

| Primitive | Crates |
| --- | --- |
| `openpty` / `forkpty` / `/dev/pts` | **`rmux-pty`** — not `rmux-os` |
| `setsid` | `rmux-os`, `rmux-pty` |
| `/proc/` (restricted on Android) | `rmux-os` |
| abstract unix sockets | `rmux-core`, `rmux-ipc`, `rmux-proto`, `rmux-server` |
| hardcoded `/tmp`, `/var/run` | **nearly every crate, plus the whole top-level binary** |
| SysV IPC (`shmget`/`semget`/`msgget`) | **none — nothing to fix here** |

The path assumptions are the killer. Termux is **not FHS**: its root is
`/data/data/com.termux/files/`, and there is no `/tmp` or `/var/run` to speak
of. Every one of those hardcoded paths breaks, and they are spread across the
entire codebase rather than funnelled through one resolver.

So the gaps span **at least six crates plus the root binary**. `rmux-pty` owns
the PTY layer; `rmux-ipc`/`proto`/`server` own the sockets; the paths are
everywhere. No surgical patch reaches them.

**Recording this plainly because it matters:** the correction came from
*checking*, not from reasoning. My hypothesis was plausible, cheap, and wrong,
and I would have shipped it if I had trusted it. This is the second time this
session (see AID-0003's amendment) — the pattern is worth noticing.

## The options that remain

**(a) Full vendor.** Copy all 12 crates into `crates/kopitiam-mux/`. This is
what you literally asked for. It works. It also means **owning 325,000 lines of
someone else's terminal multiplexer, forever** — every upstream bugfix,
security patch, and feature becomes a manual merge, and CLAUDE.md is explicit
about optimizing for a decade of maintenance.

**(c) Upstream-first.** Contribute Android support *back to rmux* and carry a
thin local patch meanwhile. rmux is **MIT OR Apache-2.0** and actively
developed. The bulk of the work is widening `cfg` gates and routing paths
through a resolver — which is *precisely* the kind of PR upstreams accept, and
which benefits every Termux user of rmux, not just you. You would own a patch,
not a fork.

**(b) Surgical fork of `rmux-os` only.** Rejected — see above. It cannot work.

## What was decided

**Nothing was implemented, deliberately.** The choice between (a) and (c) is
genuinely yours: it is a decade-long maintenance commitment either way, and the
two differ by roughly two orders of magnitude in ongoing cost. Guessing at that
unattended is exactly what CLAUDE.md's standing instruction forbids ("If a
request significantly affects architecture, stop and discuss the design first").

**My recommendation is (c), then (a) only if upstream refuses.** The engineering
work is nearly identical in both cases — the same cfg gates, the same path
resolver, the same Termux detection. The difference is purely who carries it
afterwards. Doing it as a fork *first* and upstreaming *later* is strictly worse
than the reverse, because a diverged fork is much harder to upstream than a
clean patch.

If you want (a) regardless — because you want KOPITIAM to own its whole stack,
which is a legitimate and consistent reading of the Pure Rust Core philosophy —
say so and it gets built.

## The engineering work, which is the same either way

1. Widen every `target_os = "linux"` gate that should also cover Android to
   `any(target_os = "linux", target_os = "android")`, with a comment recording
   *why* (Android is Linux-kernel but reports `target_os = "android"` — the
   knowledge that makes the whole bug class visible).
2. Replace every hardcoded `/tmp` and `/var/run` with a runtime-directory
   resolver. Detect Termux via the **`PREFIX` env var containing `com.termux`**,
   *not* `cfg!(target_os = "android")` — the platform being Android does not
   tell you the *terminal* is Termux, and it is the terminal that decides where
   things live. (`crates/kopitiam-neovim/src/icons.rs` already does exactly this
   and explains the reasoning.)
3. Abstract unix sockets → filesystem sockets under the resolved runtime dir.
4. Audit `rmux-pty`'s PTY acquisition against Bionic, and `rmux-os`'s `/proc`
   use against Android's restrictions.
5. Verify with `cargo check --target aarch64-linux-android`. Do not claim
   Android support that has not at least been type-checked.

## What would make this wrong

* If you have **already decided** you want to own the whole stack regardless of
  cost — a defensible position given Pure Rust Core — then stopping to ask was
  wasted time and (a) should just have been built. I judged the 325k-line
  ownership burden too large to assume on your behalf.
* If rmux's maintainers are known to be unresponsive to PRs, (c)'s advantage
  evaporates and (a) is simply correct. I did not check their PR-merge history;
  that is a five-minute investigation that would sharpen this recommendation
  considerably.
* If the Android gaps turn out to be *deeper* than cfg gates and paths — e.g.
  rmux's daemon supervision model fundamentally requires `/proc` semantics
  Android does not provide — then both options get much more expensive, and the
  honest answer might be "kmux needs a different daemon architecture on
  Android." I have not audited the daemon lifecycle in that depth.

---

## Amendment (post-implementation): my reconnaissance was wrong in three ways

The fork is built and Android type-checks on all three targets. In doing it, the
implementer checked my reconnaissance against the compiler rather than trusting
it, and **three of my claims above are wrong.** Recording them, because one of
them would have caused a regression if it had been followed.

**1. "Hardcoded `/tmp` infests nearly every crate" — wrong by roughly 10×.**
Almost every one of those grep hits is a *test fixture*. There are exactly **two**
production sites. I counted matches and reported them as if they were code. That
is the classic way a grep-driven survey overstates a problem.

**2. "Abstract unix sockets → filesystem sockets" — this needed NO work at all,
and doing it would have made things worse.** rmux already gates abstract sockets
on `linux` with a `not(linux)` filesystem fallback. Android reports
`target_os = "android"`, so it **already takes the filesystem path for free**. The
correct action was to *leave those gates alone* — the exact opposite of what I
prescribed. **A naive "widen every `linux` gate to `any(linux, android)`" sweep —
which is what my plan literally said to do — would have regressed this**, pushing
Android onto abstract sockets it cannot use.

That is worth sitting with: my instruction was mechanical ("widen the gates"), and
mechanically following it would have broken the thing the fork exists to fix.

**3. I missed the actual cost entirely: the binary rename.** `rmux` → `kmux` broke
four runtime *file-name* lookups (the tmux shim, SDK discovery, two helper
resolvers). These are **strings, not types**, so they produce silent feature
failures rather than build errors — invisible to exactly the `cargo check` I said
to verify with. The names now live once, in `rmux_os::host`.

## What was actually built

Twelve sub-crates nested under `crates/kopitiam-mux/crates/`, keeping upstream
names so diffs against upstream stay readable — the dominant cost of owning 325k
lines for a decade. `kmux` (25.9 MB) + `kmux-daemon` build and run; an end-to-end
`start-server` → daemon → filesystem socket → `kill-server` cycle works.

`cargo check --target {aarch64,armv7,x86_64}-linux-android` → **Finished, zero
errors** on all three. Six real Android fixes, each driven by a compiler error
rather than by grepping: `/proc`/eventfd/setsid cfg widening; `libc` binds neither
`CLOSE_RANGE_CLOEXEC` nor `nl_langinfo`/`CODESET` on Android; rustix exposes no
`TIOCGPTPEER` binding there (so the `ptsname` path, which is what Termux itself
uses); `rustix::runtime` does not exist on the libc backend.

`rmux_os::runtime_dir` is a pure `runtime_dir_candidates(env, explicit)` plus a
thin impure resolver. Termux is detected via `$PREFIX` containing `com.termux`,
not `cfg!(target_os = "android")`. Termux candidates rank **ahead of** `/tmp`, not
after — a root-owned `/tmp` that exists but is unwritable would otherwise win,
since `canonicalize` succeeding says nothing about writability. On FHS the
candidate list is exactly `["/tmp"]`, byte-identical to upstream.

**4,715 tests pass; 3 fail — and all 3 are pre-existing upstream**, proven by
building pristine `vendor/rmux` and reproducing them with the same assertion, line
and values.

## What still cannot be claimed

**It has never run on a device.** Daemon survival under Android's process
lifecycle and SELinux is the real unknown, and type-checking says nothing about
it. Filed as `kopitiam-5mr` (P1). Do not describe kmux as "working on Android"
until it has actually run there.
