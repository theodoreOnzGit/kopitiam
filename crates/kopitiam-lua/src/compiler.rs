//! AST to bytecode.
//!
//! This is where *scope* is resolved — the parser deliberately left every name
//! as a bare string, because whether `x` is a local, an upvalue, or a global is
//! a question about the enclosing functions, not about the text.
//!
//! # Upvalues, and why there is no upvalue machinery here
//!
//! Real Lua carries an "open upvalue" list: while a local is still on the stack,
//! closures point *at the stack slot*; when the scope ends, the value is
//! "closed" — copied to the heap — and the pointers are repointed. It is
//! delicate, and it is a performance optimisation.
//!
//! We skip it entirely. **Every local is heap-allocated as its own
//! `Rc<RefCell<Value>>` cell the moment it is declared.** A closure captures the
//! cell, not the value. So:
//!
//! * Capture is by *reference*, always, with no special case.
//! * Two closures made in one scope share a cell and see each other's writes.
//! * A closure made in a *fresh* iteration captures a *fresh* cell, because
//!   `NewLocal` runs again and makes one — which is precisely Lua 5.1's rule
//!   that a loop variable is distinct per iteration.
//!
//! That last point is the classic test, and it passes here for free rather than
//! by careful construction. The price is an allocation per local; see AID-0007.
//!
//! # Resolution order
//!
//! `local` in this function → upvalue (a `local` in some enclosing function) →
//! global. Globals are the fallback, never a declaration: assigning to an
//! undeclared name creates a global, exactly as Lua does.

use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::*;
use crate::error::{LuaError, Result};
use crate::instr::{Instr, NRes, Proto, UpvalDesc, UpvalSource};
use crate::value::{LuaStr, Value};

/// Compiles a parsed chunk into a top-level [`Proto`].
///
/// The chunk itself is a vararg function of no parameters — that is not a trick,
/// it is what the Lua manual says a chunk *is*, and it is why `...` works at file
/// scope.
pub fn compile(block: &Block, chunk_name: &str) -> Result<Rc<Proto>> {
    let mut c = Compiler { funcs: Vec::new(), chunk: chunk_name.to_string() };
    c.push_func("main chunk", &[], true, 0);
    c.block(block)?;
    // Every function ends with an implicit `return` -- emitting it
    // unconditionally is cheaper than checking whether the body already
    // returned on every path, and an unreachable `return` costs nothing.
    c.emit(Instr::Mark, 0);
    c.emit(Instr::Return, 0);
    Ok(Rc::new(c.pop_func()))
}

/// A local variable, as the compiler sees it.
///
/// There is no `depth` field: a block records how many locals existed when it
/// opened and truncates back to that on exit, so "which block owns this local" is
/// implied by its position in the vector rather than stored on it.
struct LocalVar {
    name: String,
    slot: u32,
}

/// The loop currently being compiled, so `break` knows where to jump.
///
/// A stack, because loops nest. Each `break` emits a `Jump` to a placeholder and
/// records the instruction index; the loop patches them all once it knows where
/// its end is.
#[derive(Default)]
struct LoopCtx {
    breaks: Vec<usize>,
}

/// Per-function compile state.
struct FuncState {
    proto: Proto,
    locals: Vec<LocalVar>,
    depth: usize,
    next_slot: u32,
    loops: Vec<LoopCtx>,
    /// Constant de-duplication, keyed by the constant's textual form. Constants
    /// pools are tiny, but a chunk that mentions `"vim"` fifty times should not
    /// carry fifty copies of it.
    const_cache: HashMap<String, u32>,
}

struct Compiler {
    /// The chain of functions being compiled, innermost last. A `Vec` rather
    /// than parent pointers because upvalue resolution walks *up* this chain,
    /// and Rust makes a vector far pleasanter to walk than a linked structure of
    /// `RefCell`s.
    funcs: Vec<FuncState>,
    chunk: String,
}

impl Compiler {
    // ---- Function and scope plumbing. ----

    fn push_func(&mut self, name: &str, params: &[String], is_vararg: bool, line: u32) {
        let proto = Proto {
            name: name.to_string(),
            chunk: self.chunk.clone(),
            code: Vec::new(),
            lines: Vec::new(),
            consts: Vec::new(),
            protos: Vec::new(),
            upvals: Vec::new(),
            num_params: params.len(),
            is_vararg,
            max_slots: 0,
        };
        self.funcs.push(FuncState {
            proto,
            locals: Vec::new(),
            depth: 0,
            next_slot: 0,
            loops: Vec::new(),
            const_cache: HashMap::new(),
        });
        // Parameters occupy slots 0..n, and their cells are created by the VM at
        // frame setup rather than by a `NewLocal` -- there is no instruction that
        // could run "before" the first one.
        for p in params {
            let _ = self.declare_local(p.clone(), line);
        }
    }

    fn pop_func(&mut self) -> Proto {
        self.funcs.pop().expect("push_func/pop_func are balanced").proto
    }

    fn fs(&mut self) -> &mut FuncState {
        self.funcs.last_mut().expect("a function is always being compiled")
    }

    fn emit(&mut self, i: Instr, line: u32) -> usize {
        let fs = self.fs();
        fs.proto.code.push(i);
        fs.proto.lines.push(line);
        fs.proto.code.len() - 1
    }

    /// The index of the next instruction — a jump target.
    fn here(&mut self) -> u32 {
        self.fs().proto.code.len() as u32
    }

    /// Emits a jump whose target is not known yet, returning its index so it can
    /// be patched. Forward jumps (out of an `if`, out of a loop) are the norm, so
    /// this is not an edge case.
    fn emit_jump(&mut self, make: fn(u32) -> Instr, line: u32) -> usize {
        self.emit(make(u32::MAX), line)
    }

    fn patch(&mut self, at: usize, target: u32) {
        let fs = self.fs();
        fs.proto.code[at] = match fs.proto.code[at] {
            Instr::Jump(_) => Instr::Jump(target),
            Instr::JumpIfFalse(_) => Instr::JumpIfFalse(target),
            Instr::AndJump(_) => Instr::AndJump(target),
            Instr::OrJump(_) => Instr::OrJump(target),
            Instr::ForPrep { base, .. } => Instr::ForPrep { base, target },
            Instr::ForLoop { base, .. } => Instr::ForLoop { base, target },
            Instr::GenForTest { base, nvars, .. } => {
                Instr::GenForTest { base, nvars, target }
            }
            other => panic!("patch called on a non-jump instruction: {other:?}"),
        };
    }

    fn constant(&mut self, v: Value) -> u32 {
        // The cache key must distinguish types: the number 1 and the string "1"
        // are different constants, and folding them together would be a
        // spectacular bug.
        let key = match &v {
            Value::Number(n) => format!("n:{}", n.to_bits()),
            Value::String(s) => format!("s:{}", String::from_utf8_lossy(s.as_bytes())),
            other => format!("?:{}", other.type_name()),
        };
        let fs = self.fs();
        if let Some(&i) = fs.const_cache.get(&key) {
            return i;
        }
        fs.proto.consts.push(v);
        let i = (fs.proto.consts.len() - 1) as u32;
        fs.const_cache.insert(key, i);
        i
    }

    fn string_const(&mut self, s: &[u8]) -> u32 {
        self.constant(Value::String(LuaStr::from_bytes(s)))
    }

    fn enter_block(&mut self) -> (usize, u32) {
        let fs = self.fs();
        fs.depth += 1;
        (fs.locals.len(), fs.next_slot)
    }

    /// Leaves a block: forget its locals and hand their slots back.
    ///
    /// Reusing a slot is safe even though a closure may still hold the cell that
    /// *was* there: the closure holds an `Rc` to the cell itself, and the next
    /// `NewLocal` on that slot installs a brand-new cell rather than overwriting
    /// the old one. Slot numbers are frame-local bookkeeping, not identity.
    fn leave_block(&mut self, saved: (usize, u32)) {
        let fs = self.fs();
        fs.depth -= 1;
        fs.locals.truncate(saved.0);
        fs.next_slot = saved.1;
    }

    fn declare_local(&mut self, name: String, _line: u32) -> u32 {
        let fs = self.fs();
        let slot = fs.next_slot;
        fs.next_slot += 1;
        fs.proto.max_slots = fs.proto.max_slots.max(fs.next_slot as usize);
        fs.locals.push(LocalVar { name, slot });
        slot
    }

    /// A slot with no name — the hidden control variables of a `for` loop.
    /// Unnameable by construction, so no Lua code can reach them.
    fn declare_hidden(&mut self, n: u32) -> u32 {
        let fs = self.fs();
        let base = fs.next_slot;
        fs.next_slot += n;
        fs.proto.max_slots = fs.proto.max_slots.max(fs.next_slot as usize);
        base
    }

    // ---- Name resolution. ----

    fn find_local(&self, func: usize, name: &str) -> Option<u32> {
        // Innermost first: an inner `local x` shadows an outer one.
        self.funcs[func].locals.iter().rev().find(|l| l.name == name).map(|l| l.slot)
    }

    /// Finds `name` as an upvalue of function `func`, adding it to `func`'s
    /// upvalue list if it is reachable.
    ///
    /// Recursion here is what makes multi-level capture work: if the name is not
    /// a local of the *immediate* parent, we ask whether the parent can see it as
    /// an upvalue, and if so we capture the parent's upvalue. Each level adds one
    /// hop, so a name three functions up is threaded down through three lists.
    fn resolve_upvalue(&mut self, func: usize, name: &str) -> Option<u32> {
        if func == 0 {
            return None; // the main chunk has no enclosing function
        }
        let parent = func - 1;

        if let Some(slot) = self.find_local(parent, name) {
            return Some(self.add_upvalue(func, name, UpvalSource::ParentLocal(slot)));
        }
        if let Some(idx) = self.resolve_upvalue(parent, name) {
            return Some(self.add_upvalue(func, name, UpvalSource::ParentUpval(idx)));
        }
        None
    }

    fn add_upvalue(&mut self, func: usize, name: &str, source: UpvalSource) -> u32 {
        let upvals = &mut self.funcs[func].proto.upvals;
        // Capturing the same name twice must yield the same index, or the two
        // references would end up pointing at different cells.
        if let Some(i) = upvals.iter().position(|u| u.source == source) {
            return i as u32;
        }
        upvals.push(UpvalDesc { name: name.to_string(), source });
        (upvals.len() - 1) as u32
    }

    fn resolve(&mut self, name: &str) -> Var {
        let current = self.funcs.len() - 1;
        if let Some(slot) = self.find_local(current, name) {
            return Var::Local(slot);
        }
        if let Some(idx) = self.resolve_upvalue(current, name) {
            return Var::Upvalue(idx);
        }
        Var::Global
    }

    // ---- Statements. ----

    fn block(&mut self, b: &Block) -> Result<()> {
        for s in &b.stats {
            self.statement(s)?;
        }
        if let Some(r) = &b.ret {
            // `return f()` propagates ALL of f's results, so the list is open.
            self.expr_list_open(&r.exprs, r.line)?;
            self.emit(Instr::Return, r.line);
        }
        Ok(())
    }

    /// A block with its own scope. Almost every block has one; the exception is
    /// `repeat`, whose condition can still see the body's locals.
    fn scoped_block(&mut self, b: &Block) -> Result<()> {
        let saved = self.enter_block();
        self.block(b)?;
        self.leave_block(saved);
        Ok(())
    }

    fn statement(&mut self, s: &Stat) -> Result<()> {
        match s {
            Stat::Call(e) => {
                // A call statement discards its results.
                self.call_expr(e, NRes::Exact(0))?;
            }

            Stat::Do(b) => self.scoped_block(b)?,

            Stat::Local { names, exprs, line } => {
                // The values are computed BEFORE the names come into scope, so
                // `local x = x` reads the OUTER x. Hence: compile the
                // expressions, then declare.
                self.expr_list_exact(exprs, names.len() as u32, *line)?;
                let slots: Vec<u32> =
                    names.iter().map(|n| self.declare_local(n.clone(), *line)).collect();
                // `NewLocal` pops from the top, so the last name is bound first.
                for slot in slots.into_iter().rev() {
                    self.emit(Instr::NewLocal(slot), *line);
                }
            }

            Stat::LocalFunction { name, body, line } => {
                // The name must be in scope INSIDE the body, or `local function
                // f() return f() end` could not recurse. So: create the cell
                // first (holding nil), compile the closure (which captures that
                // very cell), then assign into it.
                let slot = self.declare_local(name.clone(), *line);
                self.emit(Instr::PushNil, *line);
                self.emit(Instr::NewLocal(slot), *line);
                self.closure(body)?;
                self.emit(Instr::SetLocal(slot), *line);
            }

            Stat::Assign { targets, exprs, line } => self.assign(targets, exprs, *line)?,

            Stat::If { arms, else_block } => {
                let mut end_jumps = Vec::new();

                for (i, (cond, body)) in arms.iter().enumerate() {
                    let line = expr_line(cond);
                    self.expr(cond)?;
                    let skip = self.emit_jump(Instr::JumpIfFalse, line);
                    self.scoped_block(body)?;

                    // No jump-to-end needed after the last arm when there is no
                    // else: control already falls out.
                    let is_last = i == arms.len() - 1;
                    if !is_last || else_block.is_some() {
                        end_jumps.push(self.emit_jump(Instr::Jump, line));
                    }

                    let next = self.here();
                    self.patch(skip, next);
                }

                if let Some(b) = else_block {
                    self.scoped_block(b)?;
                }

                let end = self.here();
                for j in end_jumps {
                    self.patch(j, end);
                }
            }

            Stat::While { cond, body } => {
                let line = expr_line(cond);
                let start = self.here();
                self.expr(cond)?;
                let exit = self.emit_jump(Instr::JumpIfFalse, line);

                self.fs().loops.push(LoopCtx::default());
                self.scoped_block(body)?;
                self.emit(Instr::Jump(start), line);

                let end = self.here();
                self.patch(exit, end);
                self.close_loop(end);
            }

            Stat::Repeat { body, cond } => {
                let line = expr_line(cond);
                let start = self.here();

                self.fs().loops.push(LoopCtx::default());
                // The body's scope stays OPEN across the condition: `repeat local
                // x = f() until x` is legal Lua and reads the body's `x`.
                let saved = self.enter_block();
                self.block(body)?;
                self.expr(cond)?;
                self.leave_block(saved);

                self.emit(Instr::JumpIfFalse(start), line);
                let end = self.here();
                self.close_loop(end);
            }

            Stat::NumericFor { var, start, end, step, body, line } => {
                self.numeric_for(var, start, end, step.as_ref(), body, *line)?;
            }

            Stat::GenericFor { names, exprs, body, line } => {
                self.generic_for(names, exprs, body, *line)?;
            }

            Stat::Break { line } => {
                if self.fs().loops.is_empty() {
                    return Err(self.error(*line, "no loop to break"));
                }
                let j = self.emit_jump(Instr::Jump, *line);
                self.fs().loops.last_mut().expect("checked non-empty").breaks.push(j);
            }
        }
        Ok(())
    }

    /// Pops the innermost loop and points all its `break`s at `end`.
    fn close_loop(&mut self, end: u32) {
        let ctx = self.fs().loops.pop().expect("close_loop matches a push");
        for b in ctx.breaks {
            self.patch(b, end);
        }
    }

    fn numeric_for(
        &mut self,
        var: &str,
        start: &Expr,
        end: &Expr,
        step: Option<&Expr>,
        body: &Block,
        line: u32,
    ) -> Result<()> {
        // The loop's own scope holds three hidden control cells plus the visible
        // variable. Wrapping it in a block means the control cells are gone the
        // moment the loop ends.
        let saved = self.enter_block();

        self.expr(start)?;
        self.expr(end)?;
        match step {
            Some(e) => self.expr(e)?,
            None => {
                // The default step is 1. `constant` interns it, creating it if
                // this chunk has not mentioned 1 before.
                let one = self.constant(Value::Number(1.0));
                self.emit(Instr::PushConst(one), line);
            }
        }

        let base = self.declare_hidden(3);
        let prep = self.emit(Instr::ForPrep { base, target: u32::MAX }, line);

        let body_start = self.here();
        // A FRESH cell for the control variable, every iteration. This is what
        // makes closures created in the loop capture distinct variables -- Lua
        // 5.1's rule, and it costs us exactly nothing to honour.
        let body_scope = self.enter_block();
        let var_slot = self.declare_local(var.to_string(), line);
        self.emit(Instr::NewLocal(var_slot), line);

        self.fs().loops.push(LoopCtx::default());
        self.block(body)?;
        self.leave_block(body_scope);

        let loop_test = self.here();
        self.patch(prep, loop_test);
        self.emit(Instr::ForLoop { base, target: body_start }, line);

        let end_pc = self.here();
        self.close_loop(end_pc);
        self.leave_block(saved);
        Ok(())
    }

    fn generic_for(
        &mut self,
        names: &[String],
        exprs: &[Expr],
        body: &Block,
        line: u32,
    ) -> Result<()> {
        let saved = self.enter_block();

        // `for k, v in pairs(t)` -- the explist yields exactly three values
        // (iterator, state, control), padding with nil if it gave fewer.
        self.expr_list_exact(exprs, 3, line)?;
        let base = self.declare_hidden(3);
        self.emit(Instr::GenForPrep { base }, line);

        let loop_start = self.here();

        // Call the iterator through the ORDINARY call machinery, not a nested
        // Rust call. That is deliberate: it means an iterator function may itself
        // yield (`for x in coroutine.wrap(...)`), which it could not if we
        // re-entered the VM recursively here. See AID-0007.
        self.emit(Instr::Mark, line);
        self.emit(Instr::GetLocal(base), line);
        self.emit(Instr::GetLocal(base + 1), line);
        self.emit(Instr::GetLocal(base + 2), line);
        self.emit(Instr::Call(NRes::Exact(names.len() as u32)), line);

        let test = self.emit(
            Instr::GenForTest { base, nvars: names.len() as u32, target: u32::MAX },
            line,
        );

        let body_scope = self.enter_block();
        // Fresh cells per iteration, same as the numeric loop. `NewLocal` pops,
        // so bind right-to-left.
        let slots: Vec<u32> =
            names.iter().map(|n| self.declare_local(n.clone(), line)).collect();
        for slot in slots.into_iter().rev() {
            self.emit(Instr::NewLocal(slot), line);
        }

        self.fs().loops.push(LoopCtx::default());
        self.block(body)?;
        self.leave_block(body_scope);

        self.emit(Instr::Jump(loop_start), line);

        let end_pc = self.here();
        self.patch(test, end_pc);
        self.close_loop(end_pc);
        self.leave_block(saved);
        Ok(())
    }

    fn assign(&mut self, targets: &[Expr], exprs: &[Expr], line: u32) -> Result<()> {
        // The overwhelmingly common case: one target, one value. Worth its own
        // path -- it avoids a mark, an adjust, and a copy, and it is what every
        // line of a real config looks like (`vim.opt.number = true`).
        if targets.len() == 1 && exprs.len() == 1 {
            return self.assign_single(&targets[0], &exprs[0], line);
        }

        // The general case. Every value is computed BEFORE any assignment
        // happens, which is what makes `a, b = b, a` a swap rather than a
        // clobber.
        self.expr_list_exact(exprs, targets.len() as u32, line)?;

        let n = targets.len() as u32;
        for (i, target) in targets.iter().enumerate() {
            let depth_from_top = n - 1 - i as u32;
            match target {
                Expr::Name { name, line } => match self.resolve(name) {
                    Var::Local(slot) => {
                        self.emit(Instr::Copy(depth_from_top), *line);
                        self.emit(Instr::SetLocal(slot), *line);
                    }
                    Var::Upvalue(idx) => {
                        self.emit(Instr::Copy(depth_from_top), *line);
                        self.emit(Instr::SetUpval(idx), *line);
                    }
                    Var::Global => {
                        let k = self.string_const(name.as_bytes());
                        self.emit(Instr::Copy(depth_from_top), *line);
                        self.emit(Instr::SetGlobal(k), *line);
                    }
                },
                Expr::Index { obj, key, line } => {
                    self.expr(obj)?;
                    self.expr(key)?;
                    // The table and key now sit on top of the value we want, so
                    // reach two deeper for it.
                    self.emit(Instr::Copy(depth_from_top + 2), *line);
                    self.emit(Instr::SetIndex, *line);
                }
                _ => return Err(self.error(line, "cannot assign to this expression")),
            }
        }

        self.emit(Instr::Pop(n), line);
        Ok(())
    }

    fn assign_single(&mut self, target: &Expr, value: &Expr, line: u32) -> Result<()> {
        match target {
            Expr::Name { name, line } => {
                self.expr(value)?;
                match self.resolve(name) {
                    Var::Local(slot) => self.emit(Instr::SetLocal(slot), *line),
                    Var::Upvalue(idx) => self.emit(Instr::SetUpval(idx), *line),
                    Var::Global => {
                        let k = self.string_const(name.as_bytes());
                        self.emit(Instr::SetGlobal(k), *line)
                    }
                };
            }
            Expr::Index { obj, key, line } => {
                self.expr(obj)?;
                // `t.k = v` -- a constant string key, so use the dedicated
                // instruction and skip pushing the key as a value.
                if let Expr::Str(s) = &**key {
                    let k = self.string_const(s);
                    self.expr(value)?;
                    self.emit(Instr::SetField(k), *line);
                } else {
                    self.expr(key)?;
                    self.expr(value)?;
                    self.emit(Instr::SetIndex, *line);
                }
            }
            _ => return Err(self.error(line, "cannot assign to this expression")),
        }
        Ok(())
    }

    // ---- Expressions. ----

    /// Compiles an expression to **exactly one** value.
    fn expr(&mut self, e: &Expr) -> Result<()> {
        match e {
            Expr::Nil => {
                self.emit(Instr::PushNil, 0);
            }
            Expr::True => {
                self.emit(Instr::PushTrue, 0);
            }
            Expr::False => {
                self.emit(Instr::PushFalse, 0);
            }
            Expr::Number(n) => {
                let k = self.constant(Value::Number(*n));
                self.emit(Instr::PushConst(k), 0);
            }
            Expr::Str(s) => {
                let k = self.string_const(s);
                self.emit(Instr::PushConst(k), 0);
            }
            Expr::Vararg { line } => {
                // In a single-value position, `...` is just its first value.
                self.emit(Instr::PushVararg1, *line);
            }
            Expr::Function(body) => self.closure(body)?,
            // Parentheses are exactly the "truncate to one value" operator, and
            // compiling the inner expression in single-value mode IS that.
            Expr::Paren(inner) => self.expr(inner)?,

            Expr::Name { name, line } => match self.resolve(name) {
                Var::Local(slot) => {
                    self.emit(Instr::GetLocal(slot), *line);
                }
                Var::Upvalue(idx) => {
                    self.emit(Instr::GetUpval(idx), *line);
                }
                Var::Global => {
                    let k = self.string_const(name.as_bytes());
                    self.emit(Instr::GetGlobal(k), *line);
                }
            },

            Expr::Index { obj, key, line } => {
                self.expr(obj)?;
                if let Expr::Str(s) = &**key {
                    let k = self.string_const(s);
                    self.emit(Instr::GetField(k), *line);
                } else {
                    self.expr(key)?;
                    self.emit(Instr::GetIndex, *line);
                }
            }

            Expr::Call { .. } | Expr::MethodCall { .. } => {
                // A call in a single-value position is truncated to one result.
                self.call_expr(e, NRes::Exact(1))?;
            }

            Expr::Table { fields, line } => self.table(fields, *line)?,

            Expr::Unary { op, expr, line } => {
                self.expr(expr)?;
                self.emit(
                    match op {
                        UnOp::Neg => Instr::Neg,
                        UnOp::Not => Instr::Not,
                        UnOp::Len => Instr::Len,
                    },
                    *line,
                );
            }

            Expr::Binary { op, lhs, rhs, line } => self.binary(*op, lhs, rhs, *line)?,
        }
        Ok(())
    }

    fn binary(&mut self, op: BinOp, lhs: &Expr, rhs: &Expr, line: u32) -> Result<()> {
        // `and` / `or` are control flow, not arithmetic: the right side must not
        // be evaluated at all if the left settles the answer. `f() or g()` must
        // not call g when f returns something truthy.
        match op {
            BinOp::And => {
                self.expr(lhs)?;
                let j = self.emit_jump(Instr::AndJump, line);
                self.expr(rhs)?;
                let end = self.here();
                self.patch(j, end);
                return Ok(());
            }
            BinOp::Or => {
                self.expr(lhs)?;
                let j = self.emit_jump(Instr::OrJump, line);
                self.expr(rhs)?;
                let end = self.here();
                self.patch(j, end);
                return Ok(());
            }
            _ => {}
        }

        self.expr(lhs)?;
        self.expr(rhs)?;
        // `>` and `>=` get their own instructions rather than being compiled as a
        // swapped `<`/`<=`. Lua 5.1 does swap them, but that also swaps the
        // operands seen by a `__lt` metamethod, which is observable. Keeping them
        // distinct means source order is preserved everywhere.
        self.emit(
            match op {
                BinOp::Add => Instr::Add,
                BinOp::Sub => Instr::Sub,
                BinOp::Mul => Instr::Mul,
                BinOp::Div => Instr::Div,
                BinOp::Mod => Instr::Mod,
                BinOp::Pow => Instr::Pow,
                BinOp::Concat => Instr::Concat,
                BinOp::Eq => Instr::Eq,
                BinOp::Ne => Instr::Ne,
                BinOp::Lt => Instr::Lt,
                BinOp::Le => Instr::Le,
                BinOp::Gt => Instr::Gt,
                BinOp::Ge => Instr::Ge,
                BinOp::And | BinOp::Or => unreachable!("handled above"),
            },
            line,
        );
        Ok(())
    }

    fn closure(&mut self, body: &FuncBody) -> Result<()> {
        self.push_func(&body.name, &body.params, body.is_vararg, body.line);
        self.block(&body.body)?;
        self.emit(Instr::Mark, body.line);
        self.emit(Instr::Return, body.line);
        let proto = self.pop_func();

        let fs = self.fs();
        fs.proto.protos.push(Rc::new(proto));
        let idx = (fs.proto.protos.len() - 1) as u32;
        self.emit(Instr::Closure(idx), body.line);
        Ok(())
    }

    fn table(&mut self, fields: &[Field], line: u32) -> Result<()> {
        self.emit(Instr::NewTable, line);

        let mut array_index: u32 = 1;
        for (i, f) in fields.iter().enumerate() {
            let is_last = i == fields.len() - 1;
            match f {
                Field::Named(name, v) => {
                    let k = self.string_const(name.as_bytes());
                    self.expr(v)?;
                    self.emit(Instr::RawSetField(k), line);
                }
                Field::Keyed(k, v) => {
                    self.expr(k)?;
                    self.expr(v)?;
                    self.emit(Instr::RawSetIndex, line);
                }
                // The LAST positional field expands if it is a call or `...`:
                // `{ f() }` is all of f's results, but `{ f(), 1 }` is only the
                // first. This is the same truncation rule as everywhere else.
                Field::Positional(v) if is_last && v.is_multi_value() => {
                    self.emit(Instr::Mark, line);
                    self.expr_multi(v)?;
                    self.emit(Instr::SetListOpen(array_index), line);
                }
                Field::Positional(v) => {
                    self.expr(v)?;
                    self.emit(Instr::RawSetArray(array_index), line);
                    array_index += 1;
                }
            }
        }
        Ok(())
    }

    /// Compiles a call, asking for `nresults` values back.
    fn call_expr(&mut self, e: &Expr, nresults: NRes) -> Result<()> {
        match e {
            Expr::Call { func, args, line } => {
                self.emit(Instr::Mark, *line);
                self.expr(func)?;
                self.push_args(args, *line)?;
                self.emit(Instr::Call(nresults), *line);
            }
            Expr::MethodCall { obj, method, args, line } => {
                self.emit(Instr::Mark, *line);
                self.expr(obj)?;
                let k = self.string_const(method.as_bytes());
                // `Method` leaves [function, object] -- the object becomes the
                // first argument, and is evaluated only once.
                self.emit(Instr::Method(k), *line);
                self.push_args(args, *line)?;
                self.emit(Instr::Call(nresults), *line);
            }
            _ => return Err(self.error(expr_line(e), "attempt to call a non-function")),
        }
        Ok(())
    }

    /// Pushes call arguments. The last one expands if it can.
    fn push_args(&mut self, args: &[Expr], _line: u32) -> Result<()> {
        for (i, a) in args.iter().enumerate() {
            let is_last = i == args.len() - 1;
            if is_last && a.is_multi_value() {
                self.expr_multi(a)?;
            } else {
                self.expr(a)?;
            }
        }
        Ok(())
    }

    /// Compiles an expression in **multi-value** position: it may leave any
    /// number of values. Only calls and `...` can; everything else leaves one.
    fn expr_multi(&mut self, e: &Expr) -> Result<()> {
        match e {
            Expr::Call { .. } | Expr::MethodCall { .. } => self.call_expr(e, NRes::All),
            Expr::Vararg { line } => {
                self.emit(Instr::PushVarargs, *line);
                Ok(())
            }
            _ => self.expr(e),
        }
    }

    /// An expression list that keeps all of its last element's values, bracketed
    /// by a mark. The consumer (`Return`) pops the mark.
    fn expr_list_open(&mut self, exprs: &[Expr], line: u32) -> Result<()> {
        self.emit(Instr::Mark, line);
        for (i, e) in exprs.iter().enumerate() {
            let is_last = i == exprs.len() - 1;
            if is_last && e.is_multi_value() {
                self.expr_multi(e)?;
            } else {
                self.expr(e)?;
            }
        }
        Ok(())
    }

    /// An expression list adjusted to exactly `n` values — padded with nils if
    /// short, truncated if long. This is `local a, b, c = f()`.
    fn expr_list_exact(&mut self, exprs: &[Expr], n: u32, line: u32) -> Result<()> {
        self.expr_list_open(exprs, line)?;
        self.emit(Instr::AdjustTo(n), line);
        Ok(())
    }

    fn error(&self, line: impl Into<LineLike>, message: &str) -> LuaError {
        LuaError::Syntax {
            chunk: self.chunk.clone(),
            line: line.into().0,
            message: message.to_string(),
        }
    }
}

/// A tiny newtype so `self.error` can take either a `u32` or a `&u32` without
/// two overloads. Not clever, just convenient.
pub struct LineLike(u32);
impl From<u32> for LineLike {
    fn from(v: u32) -> Self {
        LineLike(v)
    }
}
impl From<&u32> for LineLike {
    fn from(v: &u32) -> Self {
        LineLike(*v)
    }
}

/// Where a name lives.
enum Var {
    Local(u32),
    Upvalue(u32),
    /// Not found in any enclosing scope. In Lua that is not an error — it is a
    /// global, resolved at run time against the globals table.
    Global,
}

/// Best-effort source line for an expression, for error messages.
fn expr_line(e: &Expr) -> u32 {
    match e {
        Expr::Name { line, .. }
        | Expr::Index { line, .. }
        | Expr::Call { line, .. }
        | Expr::MethodCall { line, .. }
        | Expr::Binary { line, .. }
        | Expr::Unary { line, .. }
        | Expr::Table { line, .. }
        | Expr::Vararg { line } => *line,
        Expr::Paren(inner) => expr_line(inner),
        _ => 0,
    }
}
