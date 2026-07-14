//! The abstract syntax tree.
//!
//! Deliberately a plain algebraic data type with no interpreter logic hanging
//! off it: the AST is *what the program says*, and the compiler is *what it
//! means*. Keeping those apart is what lets [`crate::compiler`] target a
//! bytecode VM without the parser knowing or caring (see AID-0007).
//!
//! A few nodes exist purely to preserve distinctions that a naive AST throws
//! away, and each one is load-bearing:
//!
//! * [`Expr::Paren`] — because `(f())` is **not** `f()`. Parentheses truncate a
//!   multi-value expression to exactly one value. An AST that "helpfully" folds
//!   redundant parens away silently changes `local a, b = (f())` from
//!   `a = first, b = nil` into `a, b = both results`.
//! * [`Field`] as an ordered list — because table constructors evaluate their
//!   fields in source order, and `{ [1] = "a", "b" }` genuinely depends on it.
//! * [`Expr::Vararg`] and [`Expr::Call`] are the only *multi-value* expressions,
//!   and the compiler needs to spot them structurally to know whether a list's
//!   last element expands.

/// A sequence of statements, optionally ending in a `return`.
///
/// `return` is a field rather than a [`Stat`] because Lua's grammar only allows
/// it as the **last** statement of a block. Encoding that in the type makes the
/// rule unrepresentable-if-violated instead of a check someone can forget.
#[derive(Debug, Clone, Default)]
pub struct Block {
    pub stats: Vec<Stat>,
    pub ret: Option<Return>,
}

#[derive(Debug, Clone)]
pub struct Return {
    pub exprs: Vec<Expr>,
    pub line: u32,
}

#[derive(Debug, Clone)]
pub enum Stat {
    /// A bare function call used as a statement: `print(x)`. Lua allows *only*
    /// calls here — `x + 1` alone is a syntax error — so this holds an `Expr`
    /// that the parser has already checked is a call.
    Call(Expr),

    /// `local a, b = 1, 2`
    Local { names: Vec<String>, exprs: Vec<Expr>, line: u32 },

    /// `a, b.c, d[e] = 1, 2, 3`
    Assign { targets: Vec<Expr>, exprs: Vec<Expr>, line: u32 },

    /// `if c1 then b1 elseif c2 then b2 else b3 end`
    ///
    /// `elseif` is not a distinct node: it is exactly a second arm. Modelling it
    /// as a nested `If` inside an `else` would be equivalent but would make the
    /// compiler emit a pointless extra jump per arm.
    If { arms: Vec<(Expr, Block)>, else_block: Option<Block> },

    While { cond: Expr, body: Block },

    /// `repeat body until cond`.
    ///
    /// Note the body's locals are **still in scope** in `cond` — `repeat local x
    /// = f() until x` is legal and idiomatic Lua. The compiler must not close
    /// the scope before compiling the condition.
    Repeat { body: Block, cond: Expr },

    /// `for i = start, end, step do ... end`
    NumericFor {
        var: String,
        start: Expr,
        end: Expr,
        /// `None` means the default step of 1.
        step: Option<Expr>,
        body: Block,
        line: u32,
    },

    /// `for k, v in explist do ... end`
    GenericFor { names: Vec<String>, exprs: Vec<Expr>, body: Block, line: u32 },

    /// `do ... end` — a scope, and nothing else.
    Do(Block),

    Break { line: u32 },

    /// `local function f() ... end`
    ///
    /// Distinct from `local f = function() ... end` because the *name is in
    /// scope inside the body*, which is what makes a local recursive function
    /// possible. The compiler declares `f` before compiling the body.
    LocalFunction { name: String, body: FuncBody, line: u32 },
}

/// A function's parameters and body.
#[derive(Debug, Clone)]
pub struct FuncBody {
    pub params: Vec<String>,
    /// Whether the parameter list ended in `...`.
    pub is_vararg: bool,
    pub body: Block,
    /// A human name for error messages and tracebacks (`"vim.keymap.set"`,
    /// `"<anonymous>"`). Carries no semantics.
    pub name: String,
    pub line: u32,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Nil,
    True,
    False,
    Number(f64),
    Str(Vec<u8>),
    /// `...`
    Vararg { line: u32 },
    Function(Box<FuncBody>),

    /// A bare identifier. Whether it is a local, an upvalue, or a global is a
    /// *scoping* question, and scoping is the compiler's job — the parser has no
    /// business guessing.
    Name { name: String, line: u32 },

    /// `t[k]`, and also `t.k` (which is exactly `t["k"]` — the parser desugars
    /// it, because they are the same operation and giving them two nodes would
    /// mean writing every table rule twice).
    Index { obj: Box<Expr>, key: Box<Expr>, line: u32 },

    /// `f(a, b)`
    Call { func: Box<Expr>, args: Vec<Expr>, line: u32 },

    /// `o:m(a)` — *not* sugar for `o.m(a)`. It passes `o` as an implicit first
    /// argument while evaluating `o` exactly once, which `o.m(o, a)` would not
    /// do if `o` were itself a call.
    MethodCall { obj: Box<Expr>, method: String, args: Vec<Expr>, line: u32 },

    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr>, line: u32 },
    Unary { op: UnOp, expr: Box<Expr>, line: u32 },

    /// `{ 1, 2, x = 3, [k] = v }`
    Table { fields: Vec<Field>, line: u32 },

    /// `(e)`. See the module docs: this node must survive to the compiler.
    Paren(Box<Expr>),
}

impl Expr {
    /// Whether this expression can produce **more or fewer than one** value.
    ///
    /// Only calls and `...` can. Everything else is exactly one value. The
    /// compiler branches on this to decide whether the last element of an
    /// expression list expands (`f(g())` passes all of g's results) or is padded
    /// (`f(nil)`).
    ///
    /// `Paren` is deliberately *not* multi-value, which is the whole point of it.
    pub fn is_multi_value(&self) -> bool {
        matches!(self, Expr::Call { .. } | Expr::MethodCall { .. } | Expr::Vararg { .. })
    }
}

/// One entry in a table constructor. Order is preserved; see the module docs.
#[derive(Debug, Clone)]
pub enum Field {
    /// `{ v }` — appended to the array part at the next integer index.
    Positional(Expr),
    /// `{ k = v }` — sugar for `["k"] = v`.
    Named(String, Expr),
    /// `{ [k] = v }`
    Keyed(Expr, Expr),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Concat,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// `and` and `or` are here for the parser's convenience, but they are **not**
    /// ordinary binary operators: they short-circuit, so the compiler emits jumps
    /// rather than evaluating both sides and applying an operation.
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    /// `-x`
    Neg,
    /// `not x`
    Not,
    /// `#x`
    Len,
}
