# AID-0007 — Coroutines force a bytecode VM, not a tree-walker

**Status:** Pending review
**Date:** 2026-07-14
**Review bead:** `kopitiam-0pz`
**Work bead:** `kopitiam-cj0.4` (kvim Phase 4 — `kopitiam-lua`)
**Crate:** `crates/kopitiam-lua`

## Context

`kopitiam-lua` is the pure-Rust Lua 5.1 interpreter that lets kvim execute the
config it already discovers (AID-0003 settled *pure Rust, no `mlua`*; that is
not reopened here).

The brief specified a **tree-walking interpreter** ("a register VM is a Phase-2
optimization; CLAUDE.md says correct before fast"), and separately required
**coroutines**, noting that they are the hardest part of a tree-walker and that
the resumability decision "cannot easily be retrofitted, so make it
CONSCIOUSLY".

Those two requirements are in direct tension. This AID records how the tension
was resolved and why the literal instruction was not followed.

## The problem

A coroutine must suspend deep inside a computation and resume there later.

In a tree-walking interpreter, *"where execution currently is"* is encoded in
the **Rust call stack** — `eval_expr` calling `eval_expr` calling `call_function`.
There is no first-class handle on that state. `coroutine.yield()` can therefore
only get back to `coroutine.resume()` by **unwinding** the Rust stack, which
destroys exactly the information needed to resume. This is not a detail that can
be patched later; it is the shape of the interpreter.

The known escapes:

| Approach | Why rejected |
|---|---|
| **One OS thread per coroutine**, handshaking over channels | Forces `Value: Send`, so every `Rc<RefCell<_>>` becomes `Arc<Mutex<_>>` — viral through the whole crate, slower, and it leaks a thread for every coroutine that is never resumed to completion. A language-level control-flow feature should not be a scheduling feature. |
| **Stack switching** (`generator`-style, or hand-rolled context switch) | Requires `unsafe` stack manipulation or a third-party dependency, and is a portability hazard on Android — an explicit kvim target. |
| **CPS / heap-allocated continuations** over the AST | Correct, but every single expression form must become resumable, because a yield can occur inside `f(g())`, inside a table constructor, inside an operand. In practice this rewrites the tree-walker into a state machine anyway, while keeping none of a tree-walker's readability. |
| **Explicit program counter over a linear instruction array** | Chosen. See below. |

## Decision

**Compile the AST to a simple stack-based bytecode and execute it in a VM loop
with an explicit call-frame stack.** No Rust recursion for Lua-to-Lua calls.

The reasoning in one line: **a coroutine needs a saveable "where am I", a
saveable "where am I" is a program counter, and a program counter needs a linear
instruction array.** Bytecode is not an optimization here — it is the
*enabling mechanism for correctness*.

Consequences, all of which are wins:

* A coroutine is just **its own `Vec<Frame>`**. `yield` = pop the frame stack off
  the VM and store it. `resume` = push it back. Nothing to reconstruct.
* Deep Lua recursion cannot blow the **Rust** stack, because Lua calls push a
  `Frame` rather than a Rust stack frame. Determinism, per CLAUDE.md.
* `pcall` becomes a *protected frame* — error handling unwinds the explicit frame
  stack, not the Rust stack. No `catch_unwind`, no panics as control flow.

### This is explicitly NOT the "register VM" CLAUDE.md warns against

The thing "correct before fast" rules out is a *performance* rewrite: register
allocation, constant folding, upvalue index tables, inline caches. This VM has
none of that and deliberately takes the slow-but-obviously-correct option at
every turn — most visibly, **every local variable is heap-allocated as its own
`Rc<RefCell<Value>>` cell**, whether or not it is ever captured. That is a real
per-local allocation, and it is accepted because it makes by-reference upvalue
capture correct *by construction* and deletes Lua's entire open/closed-upvalue
machinery. Making that cheap is a genuine Phase-2 optimisation; it is recorded
in the rustdoc as such.

The bytecode buys **correctness we cannot otherwise have**. The register VM would
buy **speed we do not yet need**. Only the first is in scope.

## The unexpected dividend: Lua 5.1's C-boundary rule falls out for free

Real Lua 5.1 forbids yielding across a C-call boundary:

> `attempt to yield across metamethod/C-call boundary`

...which in practice means you cannot yield out of a metamethod, out of a
comparator passed to `table.sort`, out of a `string.gsub` replacement function,
or out of a `pcall` (5.1 only — 5.2 relaxed the `pcall` case).

In this design, Lua-to-Lua calls go through the frame stack (yieldable), while
natives that re-enter Lua run a **nested** VM loop (not yieldable, because the
Rust stack is now involved). So the boundary the implementation naturally has is
*the same boundary Lua 5.1 actually specifies*. The restriction is enforced with
Lua's own error message rather than faked or silently ignored.

The one place we are deliberately **more permissive than 5.1**: `pcall` is
implemented as a VM-level protected frame, so yielding across `pcall` works here
though it errors in stock 5.1. This is a strict superset — a config that runs on
5.1 runs here — and it is what 5.2+ does anyway. Called out so nobody mistakes it
for an accident.

## What would make this wrong

* **If coroutines turn out not to be needed at all.** The brief asserts plugins
  use them. If kvim's Phase-5 plan is to rewrite *every* plugin natively in Rust
  and never run third-party Lua, then coroutines are dead weight and a plain
  tree-walker would have been ~500 lines less code and easier to read. This is
  the most plausible way this decision is wrong, and it is the maintainer's call,
  not mine — the bead itself notes the VM is needed "only for THEIR CONFIG and
  for third-party plugins not yet rewritten".
* **If the bytecode layer becomes a maintenance tax** that a future contributor
  cannot follow. Mitigated by keeping the instruction set small, naive, and
  documented, and by compiling straight from the AST with no optimisation passes
  — but it is a real cost and worth re-checking in a year.
* **If performance ever matters more than it does today.** Then this decision was
  *right* and merely incomplete: the register-VM and no-cell-for-uncaptured-locals
  work becomes worthwhile. That is a follow-on, not a reversal.

## Alternatives that were NOT considered, deliberately

Adding `mlua`, `rlua`, `hlua`, or a vendored C Lua as a fallback. AID-0003 settled
this. A "helpful" C fallback would quietly become the only tested path.
