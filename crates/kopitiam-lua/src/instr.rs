//! The instruction set.
//!
//! A **stack machine**, not a register machine. See AID-0007 for the full
//! argument; the short version is that coroutines need a saveable "where am I",
//! a saveable "where am I" is a program counter, and a program counter needs a
//! linear instruction array. Bytecode here is a *correctness* mechanism, not a
//! performance one — the instruction set is deliberately naive, and no
//! optimisation pass runs over it.
//!
//! # The one idea worth understanding: marks
//!
//! Lua expression lists are variadic in a way that is decided *at runtime*:
//! `f(a, g())` passes `a` plus however many values `g` returned, and nobody
//! knows how many that is until `g` returns.
//!
//! Rather than thread a count through every instruction, the VM keeps a small
//! stack of **marks**. [`Instr::Mark`] records the current height of the value
//! stack; a later instruction ([`Instr::Call`], [`Instr::Return`],
//! [`Instr::AdjustTo`], [`Instr::SetListOpen`]) pops that mark and takes
//! *everything above it* as its operand list. So `f(a, g())` compiles to:
//!
//! ```text
//! Mark            ; remember where the argument region starts
//! GetGlobal f     ; the callee sits at the mark
//! GetLocal  a
//! Mark            ; g's own call nests, and uses its own mark
//! GetGlobal g
//! Call All        ; pushes however many values g returned
//! Call ...        ; pops the outer mark: callee = stack[mark], args = above it
//! ```
//!
//! Marks nest naturally, cost one `usize` push, and mean no instruction ever
//! has to know a count it cannot know.

use std::rc::Rc;

use crate::value::Value;

/// How many results a call site wants.
///
/// The distinction is not cosmetic: it *is* Lua's multiple-return rule. A call
/// in the middle of an expression list is truncated to exactly one value
/// ([`NRes::Exact(1)`]); the same call in the last position expands to all of
/// them ([`NRes::All`]). The compiler decides which purely from the call's
/// syntactic position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NRes {
    Exact(u32),
    All,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Instr {
    // -- Literals.
    PushNil,
    PushTrue,
    PushFalse,
    /// Push `proto.consts[i]`.
    PushConst(u32),
    /// Push every vararg (`...` in a multi-value position).
    PushVarargs,
    /// Push exactly one vararg, or nil (`...` in a single-value position).
    PushVararg1,

    // -- Variables.
    //
    // Every local lives in a heap cell (`Rc<RefCell<Value>>`), created fresh by
    // `NewLocal` each time the declaration *executes*. That is what makes
    // closures capture by reference, and what makes each loop iteration's
    // variable distinct -- both fall straight out of "the cell is new" without
    // any open/closed-upvalue bookkeeping. It costs an allocation per local,
    // which is the single biggest thing a future optimisation pass would fix.
    /// Pop a value; install it in a **fresh** cell at this slot.
    NewLocal(u32),
    GetLocal(u32),
    /// Pop a value; write it into the slot's **existing** cell — so every
    /// closure that captured it sees the write.
    SetLocal(u32),
    GetUpval(u32),
    SetUpval(u32),
    /// Read/write a global. The index names a string constant.
    GetGlobal(u32),
    SetGlobal(u32),

    // -- Stack shuffling.
    Pop(u32),
    /// Push a copy of the value `n` places below the top (`Copy(0)` duplicates
    /// the top). Used only by multiple assignment, which needs to read the RHS
    /// values without consuming them.
    Copy(u32),

    // -- Tables. The `Raw*` forms are for table *constructors*, which by
    //    definition bypass `__newindex` (the table is brand new and has no
    //    metatable yet), and which leave the table on the stack.
    NewTable,
    /// Pop key, pop table, push `t[k]` — honouring `__index`.
    GetIndex,
    /// Pop value, pop key, pop table — honouring `__newindex`.
    SetIndex,
    /// Pop table, push `t[const]` — honouring `__index`.
    GetField(u32),
    /// Pop value, pop table — honouring `__newindex`.
    SetField(u32),
    /// `o:m` — pop `o`, push `o.m`, then push `o` back as the implicit first
    /// argument. Evaluates `o` exactly once, which is the whole point of `:`.
    Method(u32),
    /// Pop value; raw-set `t[const]` on the table *peeked* below it.
    RawSetField(u32),
    /// Pop value, pop key; raw-set on the table peeked below.
    RawSetIndex,
    /// Pop value; raw-set `t[n]` on the table peeked below.
    RawSetArray(u32),
    /// Consume a mark; raw-append everything above it to the peeked table,
    /// starting at index `n`. This is `{ 1, 2, f() }` — the trailing call
    /// expands.
    SetListOpen(u32),

    // -- Operators.
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Concat,
    Neg,
    Not,
    Len,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // -- Control flow. Targets are ABSOLUTE instruction indices: relative
    //    offsets buy nothing here and make a bytecode dump much harder to read.
    Jump(u32),
    /// Pop; jump if it was falsy.
    JumpIfFalse(u32),
    /// `and`: **peek**; if falsy, jump and *leave the value* (it is the result).
    /// Otherwise pop it and fall through to evaluate the right-hand side.
    AndJump(u32),
    /// `or`: peek; if truthy, jump and leave the value. Otherwise pop.
    OrJump(u32),

    // -- Calls.
    Mark,
    /// Consume a mark. `stack[mark]` is the callee; everything above it is the
    /// arguments.
    Call(NRes),
    /// Consume a mark; return everything above it.
    Return,
    /// Consume a mark; pad with nils or truncate so exactly `n` values sit above
    /// it. This is how `local a, b, c = f()` gets its padding.
    AdjustTo(u32),

    /// Instantiate `proto.protos[i]`, capturing upvalues from the current frame.
    Closure(u32),

    // -- Loops.
    //
    // `for i = a, b, c` keeps three hidden control cells at `base`, and a
    // separate visible cell for `i` that is re-created every iteration (which is
    // exactly why a closure made inside the loop captures that iteration's `i`).
    /// Pop step, limit, start; install them as control cells at `base`; jump to
    /// the loop test.
    ForPrep { base: u32, target: u32 },
    /// Advance the control variable; if still in range, push it (for the body's
    /// `NewLocal`) and jump to the body. Otherwise fall through.
    ForLoop { base: u32, target: u32 },
    /// Pop control, state, iterator; install them as control cells at `base`.
    GenForPrep { base: u32 },
    /// The iterator's `nvars` results are on the stack. If the first is nil, pop
    /// them and jump out. Otherwise save it as the new control value and leave
    /// them for the body's `NewLocal`s.
    GenForTest { base: u32, nvars: u32, target: u32 },
}

/// Where an upvalue comes from, resolved at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpvalSource {
    /// A local of the immediately-enclosing function: share its cell.
    ParentLocal(u32),
    /// An upvalue of the immediately-enclosing function: share the cell it
    /// already shares. This is what lets a name reach up through several levels
    /// of nesting, one hop at a time.
    ParentUpval(u32),
}

#[derive(Debug, Clone)]
pub struct UpvalDesc {
    pub name: String,
    pub source: UpvalSource,
}

/// A compiled function: code, constants, nested functions, and the shape of its
/// frame. Immutable once built, and shared by every closure made from it.
pub struct Proto {
    /// For error messages only.
    pub name: String,
    pub chunk: String,
    pub code: Vec<Instr>,
    /// `lines[i]` is the source line of `code[i]`. Parallel arrays rather than a
    /// field on `Instr` so that `Instr` stays `Copy` and small.
    pub lines: Vec<u32>,
    pub consts: Vec<Value>,
    pub protos: Vec<Rc<Proto>>,
    pub upvals: Vec<UpvalDesc>,
    pub num_params: usize,
    pub is_vararg: bool,
    /// How many local slots a frame needs. The high-water mark of the compiler's
    /// slot allocator.
    pub max_slots: usize,
}
