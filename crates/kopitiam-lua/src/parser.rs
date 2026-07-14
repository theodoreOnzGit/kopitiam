//! The Lua 5.1 parser: tokens in, [`Block`] out.
//!
//! # Precedence
//!
//! Operator precedence is done by **precedence climbing**, driven by the table
//! in [`binary_priority`]. Those numbers are not invented — they are Lua 5.1's
//! own, from `lparser.c`, and they encode associativity in a way worth spelling
//! out because it is easy to get subtly wrong:
//!
//! Each operator has a *left* and a *right* priority. The parser consumes an
//! operator only while `left > limit`, and recurses with `limit = right`.
//! Therefore:
//!
//! * `left == right` (e.g. `+` is `6, 6`) gives **left** associativity: after
//!   parsing `1 - 2`, the next `-` has `left = 6`, which is not `> 6`, so it does
//!   not get absorbed into the right operand. `1-2-3` is `(1-2)-3 == -4`.
//! * `left > right` (e.g. `^` is `10, 9`) gives **right** associativity: after
//!   `2 ^`, we recurse with `limit = 9`, and the next `^` has `left = 10 > 9`, so
//!   it *is* absorbed. `2^3^2` is `2^(3^2) == 512`, not `(2^3)^2 == 64`.
//!
//! `..` (`5, 4`) is right-associative for the same reason, and both facts are
//! pinned by tests.
//!
//! Unary operators bind at 8 — tighter than `*` (7) but looser than `^` (10).
//! That is why `-2^2` is `-(2^2) == -4` and not `(-2)^2 == 4`.

use crate::ast::*;
use crate::error::{LuaError, Result};
use crate::lexer::{Lexer, Token, TokenKind};

/// Unary operators bind tighter than every binary operator except `^`.
const UNARY_PRIORITY: u8 = 8;

/// Lua 5.1's operator priority table (`lparser.c`), as `(left, right)`.
///
/// `and`/`or` are in here so the climbing loop treats them uniformly, even
/// though the compiler will emit short-circuiting jumps rather than an operation.
fn binary_priority(op: BinOp) -> (u8, u8) {
    use BinOp::*;
    match op {
        Or => (1, 1),
        And => (2, 2),
        Lt | Gt | Le | Ge | Ne | Eq => (3, 3),
        // Right-associative: right < left.
        Concat => (5, 4),
        Add | Sub => (6, 6),
        Mul | Div | Mod => (7, 7),
        // Right-associative, and the tightest binding of all.
        Pow => (10, 9),
    }
}

fn binary_op(kind: &TokenKind) -> Option<BinOp> {
    use TokenKind as T;
    Some(match kind {
        T::Plus => BinOp::Add,
        T::Minus => BinOp::Sub,
        T::Star => BinOp::Mul,
        T::Slash => BinOp::Div,
        T::Percent => BinOp::Mod,
        T::Caret => BinOp::Pow,
        T::Concat => BinOp::Concat,
        T::Eq => BinOp::Eq,
        T::NotEq => BinOp::Ne,
        T::Less => BinOp::Lt,
        T::LessEq => BinOp::Le,
        T::Greater => BinOp::Gt,
        T::GreaterEq => BinOp::Ge,
        T::And => BinOp::And,
        T::Or => BinOp::Or,
        _ => return None,
    })
}

fn unary_op(kind: &TokenKind) -> Option<UnOp> {
    Some(match kind {
        TokenKind::Minus => UnOp::Neg,
        TokenKind::Not => UnOp::Not,
        TokenKind::Hash => UnOp::Len,
        _ => return None,
    })
}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    chunk: String,
}

/// Parses a complete chunk (a whole file or `load()`ed string).
pub fn parse(source: &str, chunk_name: &str) -> Result<Block> {
    let tokens = Lexer::new(source, chunk_name).tokenize()?;
    let mut p = Parser { tokens, pos: 0, chunk: chunk_name.to_string() };
    let block = p.block()?;
    p.expect(TokenKind::Eof)?;
    Ok(block)
}

impl Parser {
    fn peek(&self) -> &TokenKind {
        &self.tokens[self.pos].kind
    }

    fn line(&self) -> u32 {
        self.tokens[self.pos].line
    }

    fn advance(&mut self) -> TokenKind {
        let k = self.tokens[self.pos].kind.clone();
        // Never step past Eof: the parser's error paths can call `advance` after
        // hitting the end, and running off the array would panic instead of
        // producing the syntax error the user needs to see.
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        k
    }

    /// Consumes the token if it matches. Returns whether it did.
    fn eat(&mut self, kind: TokenKind) -> bool {
        if *self.peek() == kind {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: TokenKind) -> Result<()> {
        if *self.peek() == kind {
            self.advance();
            Ok(())
        } else {
            let found = self.peek().describe();
            self.error(format!("'{}' expected near {found}", describe_expected(&kind)))
        }
    }

    /// Like [`Self::expect`], but for a closing token, and it says where the
    /// thing it is closing *opened*. "'end' expected (to close 'function' at
    /// line 12)" is worth a great deal more than "'end' expected" when a config
    /// is 200 lines long.
    fn expect_close(&mut self, kind: TokenKind, opener: &str, open_line: u32) -> Result<()> {
        if *self.peek() == kind {
            self.advance();
            return Ok(());
        }
        let found = self.peek().describe();
        if self.line() == open_line {
            self.error(format!("'{}' expected near {found}", describe_expected(&kind)))
        } else {
            self.error(format!(
                "'{}' expected (to close '{opener}' at line {open_line}) near {found}",
                describe_expected(&kind)
            ))
        }
    }

    fn expect_name(&mut self) -> Result<String> {
        match self.peek().clone() {
            TokenKind::Name(n) => {
                self.advance();
                Ok(n)
            }
            other => self.error(format!("<name> expected near {}", other.describe())),
        }
    }

    fn error<T>(&self, message: impl Into<String>) -> Result<T> {
        Err(LuaError::Syntax {
            chunk: self.chunk.clone(),
            line: self.line(),
            message: message.into(),
        })
    }

    // ---- Statements. ----

    /// A block: statements until something that closes it.
    fn block(&mut self) -> Result<Block> {
        let mut stats = Vec::new();
        let mut ret = None;

        loop {
            // `return` ends the block, by grammar. Nothing may follow it (except
            // a `;`), which is why it lives outside `stats`.
            if *self.peek() == TokenKind::Return {
                let line = self.line();
                self.advance();
                let exprs = if self.block_ends_here() || *self.peek() == TokenKind::Semi {
                    Vec::new()
                } else {
                    self.expr_list()?
                };
                self.eat(TokenKind::Semi);
                ret = Some(Return { exprs, line });
                break;
            }

            if self.block_ends_here() {
                break;
            }

            // `break` also ends a block in Lua 5.1's grammar. Being lenient here
            // (allowing statements after it) would accept programs stock Lua
            // rejects, so we do not.
            if *self.peek() == TokenKind::Break {
                let line = self.line();
                self.advance();
                self.eat(TokenKind::Semi);
                stats.push(Stat::Break { line });
                break;
            }

            stats.push(self.statement()?);
            self.eat(TokenKind::Semi);
        }

        Ok(Block { stats, ret })
    }

    /// The tokens that can follow a block. `Eof` is included so an unterminated
    /// block reports as a missing `end` rather than looping forever.
    fn block_ends_here(&self) -> bool {
        matches!(
            self.peek(),
            TokenKind::End
                | TokenKind::Else
                | TokenKind::Elseif
                | TokenKind::Until
                | TokenKind::Eof
        )
    }

    fn statement(&mut self) -> Result<Stat> {
        let line = self.line();
        match self.peek().clone() {
            TokenKind::Semi => {
                self.advance();
                // An empty statement. Represent it as an empty `do end` rather
                // than adding a node for nothing.
                Ok(Stat::Do(Block::default()))
            }
            TokenKind::If => self.if_statement(),
            TokenKind::While => {
                self.advance();
                let cond = self.expr()?;
                self.expect(TokenKind::Do)?;
                let body = self.block()?;
                self.expect_close(TokenKind::End, "while", line)?;
                Ok(Stat::While { cond, body })
            }
            TokenKind::Do => {
                self.advance();
                let body = self.block()?;
                self.expect_close(TokenKind::End, "do", line)?;
                Ok(Stat::Do(body))
            }
            TokenKind::For => self.for_statement(),
            TokenKind::Repeat => {
                self.advance();
                let body = self.block()?;
                self.expect_close(TokenKind::Until, "repeat", line)?;
                // The condition is parsed *inside* the body's scope; see the
                // note on `Stat::Repeat`.
                let cond = self.expr()?;
                Ok(Stat::Repeat { body, cond })
            }
            TokenKind::Function => self.function_statement(),
            TokenKind::Local => {
                self.advance();
                if self.eat(TokenKind::Function) {
                    let name = self.expect_name()?;
                    let body = self.func_body(&name, Vec::new(), line)?;
                    Ok(Stat::LocalFunction { name, body, line })
                } else {
                    let mut names = vec![self.expect_name()?];
                    while self.eat(TokenKind::Comma) {
                        names.push(self.expect_name()?);
                    }
                    let exprs =
                        if self.eat(TokenKind::Assign) { self.expr_list()? } else { Vec::new() };
                    Ok(Stat::Local { names, exprs, line })
                }
            }
            _ => self.expr_statement(),
        }
    }

    fn if_statement(&mut self) -> Result<Stat> {
        let line = self.line();
        self.expect(TokenKind::If)?;
        let mut arms = Vec::new();

        let cond = self.expr()?;
        self.expect(TokenKind::Then)?;
        arms.push((cond, self.block()?));

        let mut else_block = None;
        loop {
            match self.peek() {
                TokenKind::Elseif => {
                    self.advance();
                    let cond = self.expr()?;
                    self.expect(TokenKind::Then)?;
                    arms.push((cond, self.block()?));
                }
                TokenKind::Else => {
                    self.advance();
                    else_block = Some(self.block()?);
                    self.expect_close(TokenKind::End, "if", line)?;
                    break;
                }
                _ => {
                    self.expect_close(TokenKind::End, "if", line)?;
                    break;
                }
            }
        }
        Ok(Stat::If { arms, else_block })
    }

    /// Both `for` loops. They share a prefix (`for Name`) and diverge on the
    /// token after it: `=` is numeric, `,`/`in` is generic.
    fn for_statement(&mut self) -> Result<Stat> {
        let line = self.line();
        self.expect(TokenKind::For)?;
        let first = self.expect_name()?;

        if self.eat(TokenKind::Assign) {
            let start = self.expr()?;
            self.expect(TokenKind::Comma)?;
            let end = self.expr()?;
            let step = if self.eat(TokenKind::Comma) { Some(self.expr()?) } else { None };
            self.expect(TokenKind::Do)?;
            let body = self.block()?;
            self.expect_close(TokenKind::End, "for", line)?;
            return Ok(Stat::NumericFor { var: first, start, end, step, body, line });
        }

        let mut names = vec![first];
        while self.eat(TokenKind::Comma) {
            names.push(self.expect_name()?);
        }
        self.expect(TokenKind::In)?;
        let exprs = self.expr_list()?;
        self.expect(TokenKind::Do)?;
        let body = self.block()?;
        self.expect_close(TokenKind::End, "for", line)?;
        Ok(Stat::GenericFor { names, exprs, body, line })
    }

    /// `function a.b.c:m(...) end`
    ///
    /// This is pure sugar, and is desugared here rather than in the compiler so
    /// that the compiler only ever sees assignments:
    ///
    /// ```lua
    /// function a.b.c(x)   -->  a.b.c = function(x) ... end
    /// function a.b:m(x)   -->  a.b.m = function(self, x) ... end
    /// ```
    ///
    /// The implicit `self` of the `:` form is inserted as a real first parameter.
    /// That is exactly what it is — there is no magic to it beyond the name.
    fn function_statement(&mut self) -> Result<Stat> {
        let line = self.line();
        self.expect(TokenKind::Function)?;

        let base = self.expect_name()?;
        let mut target = Expr::Name { name: base.clone(), line };
        let mut display = base;
        let mut implicit_self = Vec::new();

        while self.eat(TokenKind::Dot) {
            let field = self.expect_name()?;
            display = format!("{display}.{field}");
            target = Expr::Index {
                obj: Box::new(target),
                key: Box::new(Expr::Str(field.into_bytes())),
                line,
            };
        }

        if self.eat(TokenKind::Colon) {
            let method = self.expect_name()?;
            display = format!("{display}:{method}");
            target = Expr::Index {
                obj: Box::new(target),
                key: Box::new(Expr::Str(method.into_bytes())),
                line,
            };
            implicit_self.push("self".to_string());
        }

        let body = self.func_body(&display, implicit_self, line)?;
        Ok(Stat::Assign {
            targets: vec![target],
            exprs: vec![Expr::Function(Box::new(body))],
            line,
        })
    }

    /// A statement that starts with an expression: either a call, or the left
    /// side of an assignment.
    ///
    /// Lua does not allow arbitrary expressions as statements — `x + 1` on its
    /// own is a syntax error, because it cannot do anything. Only a call can.
    fn expr_statement(&mut self) -> Result<Stat> {
        let line = self.line();
        let first = self.suffixed_expr()?;

        if *self.peek() == TokenKind::Assign || *self.peek() == TokenKind::Comma {
            let mut targets = vec![first];
            while self.eat(TokenKind::Comma) {
                targets.push(self.suffixed_expr()?);
            }
            self.expect(TokenKind::Assign)?;
            let exprs = self.expr_list()?;

            // Every target must be something you can assign *to*. `f() = 1` and
            // `(a) = 1` are both syntax errors in Lua, and catching them here
            // gives a clear message instead of a baffling one from the compiler.
            for t in &targets {
                if !matches!(t, Expr::Name { .. } | Expr::Index { .. }) {
                    return self.error("cannot assign to this expression");
                }
            }
            return Ok(Stat::Assign { targets, exprs, line });
        }

        match first {
            e @ (Expr::Call { .. } | Expr::MethodCall { .. }) => Ok(Stat::Call(e)),
            _ => self.error("syntax error near unexpected expression"),
        }
    }

    fn func_body(&mut self, name: &str, mut params: Vec<String>, line: u32) -> Result<FuncBody> {
        self.expect(TokenKind::LParen)?;
        let mut is_vararg = false;

        if *self.peek() != TokenKind::RParen {
            loop {
                match self.peek().clone() {
                    TokenKind::Ellipsis => {
                        self.advance();
                        is_vararg = true;
                        // `...` must be last: `function f(..., a)` is nonsense.
                        break;
                    }
                    TokenKind::Name(n) => {
                        self.advance();
                        params.push(n);
                    }
                    other => {
                        return self
                            .error(format!("<name> expected near {}", other.describe()));
                    }
                }
                if !self.eat(TokenKind::Comma) {
                    break;
                }
            }
        }
        self.expect(TokenKind::RParen)?;

        let body = self.block()?;
        self.expect_close(TokenKind::End, "function", line)?;

        Ok(FuncBody { params, is_vararg, body, name: name.to_string(), line })
    }

    // ---- Expressions. ----

    fn expr_list(&mut self) -> Result<Vec<Expr>> {
        let mut out = vec![self.expr()?];
        while self.eat(TokenKind::Comma) {
            out.push(self.expr()?);
        }
        Ok(out)
    }

    fn expr(&mut self) -> Result<Expr> {
        self.sub_expr(0)
    }

    /// Precedence climbing. See the module docs for why the numbers are what
    /// they are; this function is where associativity actually happens.
    fn sub_expr(&mut self, limit: u8) -> Result<Expr> {
        let line = self.line();

        let mut left = if let Some(op) = unary_op(self.peek()) {
            self.advance();
            // Recursing at UNARY_PRIORITY (not 0) is what makes `-2^2` parse as
            // `-(2^2)`: `^` has left priority 10 > 8, so it binds tighter than
            // the negation and gets absorbed into the operand.
            let operand = self.sub_expr(UNARY_PRIORITY)?;
            Expr::Unary { op, expr: Box::new(operand), line }
        } else {
            self.simple_expr()?
        };

        while let Some(op) = binary_op(self.peek()) {
            let (left_prio, right_prio) = binary_priority(op);
            if left_prio <= limit {
                break;
            }
            let op_line = self.line();
            self.advance();
            // Recursing with the operator's RIGHT priority is the associativity
            // knob: for `^` (10, 9) it lets another `^` in, for `+` (6, 6) it
            // does not.
            let right = self.sub_expr(right_prio)?;
            left = Expr::Binary {
                op,
                lhs: Box::new(left),
                rhs: Box::new(right),
                line: op_line,
            };
        }

        Ok(left)
    }

    fn simple_expr(&mut self) -> Result<Expr> {
        let line = self.line();
        Ok(match self.peek().clone() {
            TokenKind::Nil => {
                self.advance();
                Expr::Nil
            }
            TokenKind::True => {
                self.advance();
                Expr::True
            }
            TokenKind::False => {
                self.advance();
                Expr::False
            }
            TokenKind::Number(n) => {
                self.advance();
                Expr::Number(n)
            }
            TokenKind::Str(s) => {
                self.advance();
                Expr::Str(s)
            }
            TokenKind::Ellipsis => {
                self.advance();
                Expr::Vararg { line }
            }
            TokenKind::LBrace => self.table_constructor()?,
            TokenKind::Function => {
                self.advance();
                Expr::Function(Box::new(self.func_body("<anonymous>", Vec::new(), line)?))
            }
            _ => self.suffixed_expr()?,
        })
    }

    /// A name or a parenthesised expression — the only things a suffix chain can
    /// start from. `1:foo()` is not valid Lua, and this is why.
    fn primary_expr(&mut self) -> Result<Expr> {
        let line = self.line();
        match self.peek().clone() {
            TokenKind::Name(n) => {
                self.advance();
                Ok(Expr::Name { name: n, line })
            }
            TokenKind::LParen => {
                self.advance();
                let e = self.expr()?;
                self.expect(TokenKind::RParen)?;
                // The `Paren` wrapper is not decoration: it truncates a
                // multi-value expression to one value. `(f())` is one value even
                // when `f()` returns three.
                Ok(Expr::Paren(Box::new(e)))
            }
            other => self.error(format!("unexpected symbol near {}", other.describe())),
        }
    }

    /// A primary expression followed by any number of suffixes: `.k`, `[k]`,
    /// `(args)`, `:m(args)`, `"str"`, `{table}`.
    fn suffixed_expr(&mut self) -> Result<Expr> {
        let mut e = self.primary_expr()?;
        loop {
            let line = self.line();
            match self.peek().clone() {
                TokenKind::Dot => {
                    self.advance();
                    let name = self.expect_name()?;
                    // `t.k` IS `t["k"]`. Same node.
                    e = Expr::Index {
                        obj: Box::new(e),
                        key: Box::new(Expr::Str(name.into_bytes())),
                        line,
                    };
                }
                TokenKind::LBracket => {
                    self.advance();
                    let key = self.expr()?;
                    self.expect(TokenKind::RBracket)?;
                    e = Expr::Index { obj: Box::new(e), key: Box::new(key), line };
                }
                TokenKind::Colon => {
                    self.advance();
                    let method = self.expect_name()?;
                    let args = self.call_args()?;
                    e = Expr::MethodCall { obj: Box::new(e), method, args, line };
                }
                // The three call spellings. `f"s"` and `f{t}` are Lua's sugar for
                // `f("s")` and `f({t})`, and they are everywhere in real configs
                // -- `require "foo"` is exactly this.
                TokenKind::LParen | TokenKind::Str(_) | TokenKind::LBrace => {
                    let args = self.call_args()?;
                    e = Expr::Call { func: Box::new(e), args, line };
                }
                _ => return Ok(e),
            }
        }
    }

    fn call_args(&mut self) -> Result<Vec<Expr>> {
        match self.peek().clone() {
            // f"literal"
            TokenKind::Str(s) => {
                self.advance();
                Ok(vec![Expr::Str(s)])
            }
            // f{ ... }
            TokenKind::LBrace => Ok(vec![self.table_constructor()?]),
            TokenKind::LParen => {
                self.advance();
                let args = if *self.peek() == TokenKind::RParen {
                    Vec::new()
                } else {
                    self.expr_list()?
                };
                self.expect(TokenKind::RParen)?;
                Ok(args)
            }
            other => self.error(format!("function arguments expected near {}", other.describe())),
        }
    }

    /// `{ 1, 2; x = 3, ["y"] = 4, }`
    ///
    /// Both `,` and `;` separate fields (Lua accepts either, interchangeably),
    /// and a trailing separator is allowed.
    fn table_constructor(&mut self) -> Result<Expr> {
        let line = self.line();
        self.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();

        while *self.peek() != TokenKind::RBrace {
            match self.peek().clone() {
                // `[k] = v`
                TokenKind::LBracket => {
                    self.advance();
                    let k = self.expr()?;
                    self.expect(TokenKind::RBracket)?;
                    self.expect(TokenKind::Assign)?;
                    fields.push(Field::Keyed(k, self.expr()?));
                }
                // `k = v`, but ONLY when an `=` follows. `{ x }` is a positional
                // field holding the value of the variable `x`, and mistaking it
                // for `{ x = x }` would be badly wrong -- so the `=` must be
                // confirmed by lookahead before committing.
                TokenKind::Name(n)
                    if self.tokens[self.pos + 1].kind == TokenKind::Assign =>
                {
                    self.advance();
                    self.advance();
                    fields.push(Field::Named(n, self.expr()?));
                }
                _ => fields.push(Field::Positional(self.expr()?)),
            }

            // `,` and `;` are interchangeable separators; either may also trail.
            if !self.eat(TokenKind::Comma) && !self.eat(TokenKind::Semi) {
                break;
            }
        }

        self.expect_close(TokenKind::RBrace, "{", line)?;
        Ok(Expr::Table { fields, line })
    }
}

/// The text `expect` uses for the token it wanted.
fn describe_expected(kind: &TokenKind) -> String {
    match kind {
        TokenKind::Eof => "<eof>".to_string(),
        other => other.describe().trim_matches('\'').to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Renders an expression as a fully-parenthesised string, so associativity
    /// and precedence can be asserted *structurally* rather than by evaluating
    /// and hoping. A test that only checks `2^3^2 == 512` would also pass if the
    /// parser were right by luck; this checks the shape of the tree.
    fn shape(e: &Expr) -> String {
        match e {
            Expr::Nil => "nil".into(),
            Expr::True => "true".into(),
            Expr::False => "false".into(),
            Expr::Number(n) => crate::number::format_number(*n),
            Expr::Str(s) => format!("{:?}", String::from_utf8_lossy(s)),
            Expr::Vararg { .. } => "...".into(),
            Expr::Name { name, .. } => name.clone(),
            Expr::Function(_) => "function".into(),
            Expr::Paren(e) => format!("paren({})", shape(e)),
            Expr::Index { obj, key, .. } => format!("({}[{}])", shape(obj), shape(key)),
            Expr::Call { func, args, .. } => {
                let a: Vec<_> = args.iter().map(shape).collect();
                format!("call({}, [{}])", shape(func), a.join(", "))
            }
            Expr::MethodCall { obj, method, args, .. } => {
                let a: Vec<_> = args.iter().map(shape).collect();
                format!("method({}, {}, [{}])", shape(obj), method, a.join(", "))
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                format!("({} {:?} {})", shape(lhs), op, shape(rhs))
            }
            Expr::Unary { op, expr, .. } => format!("({:?} {})", op, shape(expr)),
            Expr::Table { fields, .. } => {
                let f: Vec<_> = fields
                    .iter()
                    .map(|f| match f {
                        Field::Positional(e) => shape(e),
                        Field::Named(k, v) => format!("{k}={}", shape(v)),
                        Field::Keyed(k, v) => format!("[{}]={}", shape(k), shape(v)),
                    })
                    .collect();
                format!("{{{}}}", f.join(", "))
            }
        }
    }

    /// Parses `return <expr>` and returns the expression's shape.
    fn expr_shape(src: &str) -> String {
        let b = parse(&format!("return {src}"), "=test").unwrap();
        shape(&b.ret.unwrap().exprs[0])
    }

    fn parse_err(src: &str) -> String {
        parse(src, "=test").unwrap_err().to_string()
    }

    #[test]
    fn power_is_right_associative() {
        // 2^3^2 must be 2^(3^2) = 512, NOT (2^3)^2 = 64.
        assert_eq!(expr_shape("2^3^2"), "(2 Pow (3 Pow 2))");
    }

    #[test]
    fn arithmetic_is_left_associative() {
        // 1-2-3 must be (1-2)-3 = -4, NOT 1-(2-3) = 2.
        assert_eq!(expr_shape("1-2-3"), "((1 Sub 2) Sub 3)");
        assert_eq!(expr_shape("1/2/4"), "((1 Div 2) Div 4)");
    }

    #[test]
    fn concat_is_right_associative() {
        assert_eq!(expr_shape(r#""a".."b".."c""#), r#"("a" Concat ("b" Concat "c"))"#);
    }

    #[test]
    fn unary_binds_tighter_than_multiplication_but_looser_than_power() {
        // -2^2 is -(2^2) = -4. This one bites people.
        assert_eq!(expr_shape("-2^2"), "(Neg (2 Pow 2))");
        // but -a*b is (-a)*b
        assert_eq!(expr_shape("-a*b"), "((Neg a) Mul b)");
        // #t+1 is (#t)+1
        assert_eq!(expr_shape("#t+1"), "((Len t) Add 1)");
        // not a == b  parses as  (not a) == b
        assert_eq!(expr_shape("not a == b"), "((Not a) Eq b)");
    }

    #[test]
    fn the_full_precedence_ladder() {
        // or < and < comparison < .. < +- < */% < unary < ^
        assert_eq!(
            expr_shape("a or b and c"),
            "(a Or (b And c))",
            "and binds tighter than or"
        );
        assert_eq!(
            expr_shape("a and b < c"),
            "(a And (b Lt c))",
            "comparison binds tighter than and"
        );
        assert_eq!(
            expr_shape("a < b .. c"),
            "(a Lt (b Concat c))",
            "concat binds tighter than comparison"
        );
        assert_eq!(
            expr_shape("a .. b + c"),
            "(a Concat (b Add c))",
            "+ binds tighter than concat"
        );
        assert_eq!(
            expr_shape("a + b * c"),
            "(a Add (b Mul c))",
            "* binds tighter than +"
        );
        assert_eq!(
            expr_shape("a * b ^ c"),
            "(a Mul (b Pow c))",
            "^ binds tighter than *"
        );
    }

    #[test]
    fn power_binds_tighter_than_unary_on_its_right_too() {
        // 2^-3 is 2^(-3): after `^` we recurse, and a unary operator is allowed.
        assert_eq!(expr_shape("2^-3"), "(2 Pow (Neg 3))");
    }

    #[test]
    fn parens_survive_to_the_ast() {
        // If Paren were folded away, `(f())` would expand to all of f's results
        // instead of exactly one. The node must exist.
        assert_eq!(expr_shape("(f())"), "paren(call(f, []))");
        assert_eq!(expr_shape("(1+2)*3"), "(paren((1 Add 2)) Mul 3)");
    }

    #[test]
    fn dot_access_is_desugared_to_indexing() {
        assert_eq!(expr_shape("a.b.c"), r#"((a["b"])["c"])"#);
        assert_eq!(expr_shape("a[1]"), "(a[1])");
        // `a.b.c` and `a["b"]["c"]` are the SAME tree -- that is what "desugared"
        // means, and it is why the table rules only need writing once.
        assert_eq!(expr_shape("a.b.c"), expr_shape(r#"a["b"]["c"]"#));
    }

    #[test]
    fn call_sugar() {
        // f"str" and f{t} are calls, and `require "settings"` depends on it.
        assert_eq!(expr_shape(r#"f"str""#), r#"call(f, ["str"])"#);
        assert_eq!(expr_shape("f{1}"), "call(f, [{1}])");
        assert_eq!(expr_shape(r#"require "settings""#), r#"call(require, ["settings"])"#);
        assert_eq!(expr_shape(r#"require("settings")"#), r#"call(require, ["settings"])"#);
    }

    #[test]
    fn method_calls_are_not_field_calls() {
        assert_eq!(expr_shape("o:m(1)"), "method(o, m, [1])");
        assert_eq!(expr_shape("o.m(1)"), r#"call((o["m"]), [1])"#);
        // Chained.
        assert_eq!(
            expr_shape(r#"require("hop").hint_words({})"#),
            r#"call((call(require, ["hop"])["hint_words"]), [{}])"#
        );
    }

    #[test]
    fn table_constructors_in_every_form() {
        assert_eq!(expr_shape("{}"), "{}");
        assert_eq!(expr_shape("{1, 2, 3}"), "{1, 2, 3}");
        assert_eq!(expr_shape("{x = 1}"), "{x=1}");
        assert_eq!(expr_shape(r#"{["y"] = 2}"#), r#"{["y"]=2}"#);
        assert_eq!(expr_shape("{1, x = 2, [3] = 4}"), "{1, x=2, [3]=4}");
        // `;` is a separator too, and a trailing one is allowed.
        assert_eq!(expr_shape("{1; 2;}"), "{1, 2}");
        assert_eq!(expr_shape("{1, 2,}"), "{1, 2}");
        // Order is preserved.
        assert_eq!(expr_shape("{[1] = 'a', 'b'}"), r#"{[1]="a", "b"}"#);
    }

    #[test]
    fn a_bare_name_in_a_table_is_positional_not_named() {
        // `{ x }` is the VALUE of x at position 1 -- not `{ x = x }`.
        assert_eq!(expr_shape("{x}"), "{x}");
        assert_eq!(expr_shape("{x = x}"), "{x=x}");
    }

    #[test]
    fn the_real_hop_keymap_from_the_maintainers_config_parses() {
        // Verbatim from ~/.config/nvim/lua/keymaps.lua -- a closure argument, a
        // chained call, and a table constructor with a named field.
        let src = r#"
vim.keymap.set("", "f", function()
  require("hop").hint_words({ current_line_only = false })
end, { remap = true, desc = "Hop to word" })
"#;
        let b = parse(src, "=keymaps.lua").unwrap();
        assert_eq!(b.stats.len(), 1);
        assert!(matches!(b.stats[0], Stat::Call(_)));
    }

    #[test]
    fn function_declarations_desugar_to_assignments() {
        let b = parse("function a.b.c() end", "=t").unwrap();
        match &b.stats[0] {
            Stat::Assign { targets, .. } => {
                assert_eq!(shape(&targets[0]), r#"((a["b"])["c"])"#);
            }
            other => panic!("expected an assignment, got {other:?}"),
        }
    }

    #[test]
    fn a_method_declaration_gains_an_implicit_self_parameter() {
        let b = parse("function a:m(x) end", "=t").unwrap();
        match &b.stats[0] {
            Stat::Assign { targets, exprs, .. } => {
                assert_eq!(shape(&targets[0]), r#"(a["m"])"#);
                match &exprs[0] {
                    Expr::Function(f) => {
                        assert_eq!(f.params, vec!["self".to_string(), "x".to_string()]);
                        assert_eq!(f.name, "a:m");
                    }
                    other => panic!("expected a function, got {other:?}"),
                }
            }
            other => panic!("expected an assignment, got {other:?}"),
        }
        // The dot form gets no self.
        let b = parse("function a.m(x) end", "=t").unwrap();
        match &b.stats[0] {
            Stat::Assign { exprs, .. } => match &exprs[0] {
                Expr::Function(f) => assert_eq!(f.params, vec!["x".to_string()]),
                other => panic!("expected a function, got {other:?}"),
            },
            other => panic!("expected an assignment, got {other:?}"),
        }
    }

    #[test]
    fn every_statement_form_parses() {
        let src = r#"
local a, b = 1, 2
a = 3
a, b = b, a
if a then b = 1 elseif b then b = 2 else b = 3 end
while a do break end
repeat a = a - 1 until a == 0
for i = 1, 10, 2 do print(i) end
for k, v in pairs(t) do print(k, v) end
do local x = 1 end
local function f() return 1 end
function g() end
function t.h() end
function t:m() end
return a, b
"#;
        let b = parse(src, "=t").unwrap();
        assert!(b.ret.is_some());
        assert_eq!(b.ret.as_ref().unwrap().exprs.len(), 2);
    }

    #[test]
    fn varargs_parse_in_a_vararg_function() {
        let b = parse("local function f(...) return ... end", "=t").unwrap();
        match &b.stats[0] {
            Stat::LocalFunction { body, .. } => {
                assert!(body.is_vararg);
                assert!(body.params.is_empty());
            }
            other => panic!("expected a local function, got {other:?}"),
        }
        let b = parse("local function f(a, ...) end", "=t").unwrap();
        match &b.stats[0] {
            Stat::LocalFunction { body, .. } => {
                assert!(body.is_vararg);
                assert_eq!(body.params, vec!["a".to_string()]);
            }
            other => panic!("expected a local function, got {other:?}"),
        }
    }

    #[test]
    fn a_bare_expression_is_not_a_statement() {
        // Only a call can stand alone. `x + 1` does nothing and is an error.
        assert!(parse("x + 1", "=t").is_err());
        assert!(parse("1", "=t").is_err());
        // But a call is fine.
        assert!(parse("f()", "=t").is_ok());
        assert!(parse("o:m()", "=t").is_ok());
    }

    #[test]
    fn you_cannot_assign_to_a_call() {
        assert!(parse_err("f() = 1").contains("cannot assign"));
    }

    #[test]
    fn return_must_be_the_last_statement_in_a_block() {
        assert!(parse("return 1 print(2)", "=t").is_err());
        // But `return` followed by `end` is fine, and so is a bare `return`.
        assert!(parse("local function f() return end", "=t").is_ok());
        assert!(parse("local function f() return 1; end", "=t").is_ok());
    }

    #[test]
    fn unclosed_blocks_report_the_opening_line() {
        let err = parse_err("function f()\n  local x = 1\n");
        assert!(err.contains("'end' expected"), "{err}");
        assert!(err.contains("line 1"), "should point at the opening line: {err}");
    }
}
