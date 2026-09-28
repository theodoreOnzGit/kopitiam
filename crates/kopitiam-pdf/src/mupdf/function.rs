//! Ported from MuPDF `source/pdf/pdf-function.c` + `source/fitz/device.c`
//! (`fz_eval_function`) (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.),
//! translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation: the
//! algorithms and numeric behaviour follow MuPDF; the code is re-expressed in
//! idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction
//! references").
//!
//! # PDF functions (types 0, 2, 3, 4)
//!
//! A PDF function maps `m` inputs to `n` outputs; shadings, tint transforms
//! (Separation / DeviceN) and transfer functions all go through here. The four
//! kinds, same as `pdf-function.c`:
//!
//! * **Type 0, sampled** -- a table of `Size[0] × … × Size[m-1]` samples of `n`
//!   values each, `BitsPerSample` 1/2/4/8/12/16/24/32, multilinear
//!   interpolation between the table points, `Encode`/`Decode` remapping.
//! * **Type 2, exponential** -- `C0 + x^N · (C1 - C0)`, one input only.
//! * **Type 3, stitching** -- `k` one-input sub-functions spliced along the
//!   input axis at `Bounds`, each sub-interval remapped through `Encode`.
//! * **Type 4, PostScript calculator** -- a tiny stack language, compiled once
//!   to a flat code array at load and interpreted per evaluation.
//!
//! ## MuPDF behaviour kept on purpose (don't "fix" these, hor)
//!
//! * **Clamping is `fz_clamp` -- `x < min ? min : x > max ? max : x`**, so a
//!   NaN input passes straight through, not snapped to `min`.
//! * **Missing `Domain` is not an error.** `m` becomes 1 and the domain is
//!   `[0 0]`, so every input clamps to 0. Missing `Range` on type 0 / type 4 is
//!   also not an error -- the range stays all-zero, and because those two types
//!   clamp their output *unconditionally*, every output comes out 0. Types 2
//!   and 3 only clamp when `Range` was given.
//! * **Stitching without `Encode` maps every sub-input to 0** (the encode
//!   pairs default to `[0 0]`, and `lerp` with `ymin == ymax` returns `ymin`).
//! * **A `true` / `false` literal in calculator code stops execution.** The
//!   lexer turns them into `PDF_TOK_TRUE` / `PDF_TOK_FALSE`, `parse_code`
//!   compiles them to `PS_BOOL` code objects, and `ps_run` has no `PS_BOOL`
//!   case -- it warns "foreign object in calculator function" and returns.
//!   So `{ pop 3 true 4 }` outputs 3, not 4. Surprising, but that is MuPDF.
//! * **Popping the wrong type does not pop.** `ps_pop_int` / `ps_pop_real` on
//!   a bool (and `ps_pop_bool` on a number) return 0 and leave the stack
//!   alone. Stack overflow silently drops the push; underflow yields 0.
//! * **`-1 index` reads the stale slot just above the stack top** (the C stack
//!   is a fixed 100-slot array, zero-initialised to `false` bools, and never
//!   cleared on pop). That slot is always the `-1` operand `index` just
//!   popped, so `-1 index` re-pushes `-1`. We keep the full array so the
//!   read is the same one C does.
//! * **Warnings (`fz_warn`) are dropped**: wrong numbers of Size / Encode /
//!   Decode / C0 / C1 / Bounds entries, sub-function arity mismatches and so
//!   on all just carry on with MuPDF's fallback values.
//!
//! ## Deliberate divergences (only where the C is undefined behaviour)
//!
//! * **`n` without `Range` is capped at `FZ_MAX_COLORS` (32).** MuPDF sets
//!   `n = out` (the caller's count) uncapped; for `out > 32` it would overrun
//!   its fixed `c0`/`c1`/`decode`/`fakeout` arrays. We cap, and
//!   `fz_eval_function`'s zero-fill covers the extra outputs.
//! * **Float → int conversions saturate** (Rust `as`) where C would be UB
//!   (NaN or out-of-range reals in `cvi`, `ps_pop_int`, `floorf` in the sample
//!   lookup); integer arithmetic wraps where C signed overflow is UB.
//! * **Sample table lookups are bounds-checked** (out of range reads 0.0);
//!   they can only be out of range in the UB cases above.
//! * **Nesting depth is bounded for direct objects too.** `pdf_cycle` only
//!   walks (and depth-limits) the chain for indirect objects; we also refuse a
//!   chain longer than `MAX_CYCLE_STACK_DEPTH` (256) of direct ones, so a
//!   hand-built `Object` tree cannot blow the Rust stack.
//! * **The per-document function store (`pdf_find_item` / `pdf_store_item`)**
//!   becomes a cache that lives for one [`PdfFunction::load`] call, keyed by
//!   object number. Like MuPDF's store, a cache hit returns the function as
//!   first loaded, whatever `in`/`out` the second reference asked for. This
//!   is what keeps a shared-sub-function DAG linear instead of exponential.

use std::collections::HashMap;
use std::sync::Arc;

use super::error::{Error, Result};
use super::lex::{Token, lex};
use super::object::Object;
use super::stream::Stream;
use super::xref::PdfDocument;

/// `FZ_MAX_COLORS` -- the most outputs a function may have (`MAX_N`).
const MAX_N: usize = 32;
/// `FZ_MAX_COLORS` -- the most inputs a function may have (`MAX_M`).
const MAX_M: usize = 32;
/// The most sub-functions a stitching function may have.
const MAX_STITCHING: usize = 256;
/// `pdf-object.c`'s `MAX_CYCLE_STACK_DEPTH`, used by `pdf_cycle`.
const MAX_CYCLE_STACK_DEPTH: usize = 256;
/// `MAX_SAMPLE_FUNCTION_SIZE` -- the most floats a sample table may hold.
const MAX_SAMPLE_FUNCTION_SIZE: i32 = 100 << 20;
/// `FZ_RADIAN` from `fitz/system.h`, truncated exactly as MuPDF has it.
const FZ_RADIAN: f32 = 57.295_779_5;
/// `ps_stack.stack[100]`.
const PS_STACK_SIZE: usize = 100;
/// `parse_code`'s nesting limit.
const MAX_PS_NESTING: u32 = 100;

// MuPDF: fz_clamp (geometry.h:156) -- NaN passes through untouched.
#[inline]
fn fz_clamp(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

// MuPDF: lerp (pdf-function.c:124)
#[inline]
fn lerp(x: f32, xmin: f32, xmax: f32, ymin: f32, ymax: f32) -> f32 {
    if xmin == xmax {
        return ymin;
    }
    if ymin == ymax {
        return ymin;
    }
    ymin + (x - xmin) * (ymax - ymin) / (xmax - xmin)
}

// ---------------------------------------------------------------------------
// The loaded function
// ---------------------------------------------------------------------------

/// A loaded PDF function (MuPDF `pdf_function` / `fz_function`).
///
/// Load it once with [`PdfFunction::load`], then [`eval`](PdfFunction::eval)
/// as many times as you like -- evaluation never fails and never panics; bad
/// input degrades the way MuPDF's does (clamps, zeros, early stop).
#[derive(Clone, Debug)]
pub struct PdfFunction {
    /// `fz_function.m` -- number of inputs (1..=32).
    m: usize,
    /// `fz_function.n` -- number of outputs (0..=32).
    n: usize,
    /// `pdf_function.domain` -- `[min, max]` per input.
    domain: [[f32; 2]; MAX_M],
    /// `pdf_function.range` -- `[min, max]` per output; all zero when absent.
    range: [[f32; 2]; MAX_N],
    /// `pdf_function.has_range`.
    has_range: bool,
    kind: Kind,
}

#[derive(Clone, Debug)]
enum Kind {
    Sampled(Sampled),
    Exponential(Exponential),
    Stitching(Stitching),
    PostScript(Arc<Vec<PsCode>>),
}

/// `pdf_function_sa`.
#[derive(Clone, Debug)]
struct Sampled {
    size: [i32; MAX_M],
    encode: [[f32; 2]; MAX_M],
    decode: [[f32; 2]; MAX_N],
    samples: Arc<Vec<f32>>,
}

/// `pdf_function_e`.
#[derive(Clone, Debug)]
struct Exponential {
    n: f32,
    c0: [f32; MAX_N],
    c1: [f32; MAX_N],
}

/// `pdf_function_st` (`k` is `funcs.len()`).
#[derive(Clone, Debug)]
struct Stitching {
    funcs: Vec<Arc<PdfFunction>>,
    bounds: Vec<f32>,
    encode: Vec<f32>,
}

impl PdfFunction {
    // MuPDF: pdf_load_function (pdf-function.c:1565)
    /// Load the function `obj` (a function dict or stream, possibly an indirect
    /// `Ref`). `n_in`/`n_out` are what the CALLER will pass/expect (MuPDF's
    /// `in`/`out`): `n_out` becomes the function's `n` when it has no `Range`.
    ///
    /// Stream-backed functions (type 0 and type 4) need `obj` to be the
    /// indirect reference, since that is how the stream body is found; a direct
    /// dict with no stream fails to load, as in MuPDF.
    pub fn load(doc: &PdfDocument, obj: &Object, n_in: usize, n_out: usize) -> Result<PdfFunction> {
        let mut cycle = Vec::new();
        let mut cache = HashMap::new();
        let f = load_function_imp(doc, obj, n_in, n_out, &mut cycle, &mut cache)?;
        Ok(Arc::unwrap_or_clone(f))
    }

    // MuPDF: pdf_eval_function (pdf-function.c:1433) / fz_eval_function
    // (device.c:1151)
    /// Evaluate: reads `input` (length = the caller's n_in), writes `out`
    /// (length = caller's n_out). A short `input` is padded with zeros; a long
    /// one is truncated to `m`. Outputs past the function's `n` are set to 0;
    /// if `out` is shorter than `n`, the extra outputs are computed and dropped.
    pub fn eval(&self, input: &[f32], out: &mut [f32]) {
        let mut fakein = [0.0f32; MAX_M];
        let input: &[f32] = if input.len() < self.m {
            fakein[..input.len()].copy_from_slice(input);
            &fakein
        } else {
            input
        };

        if out.len() < self.n {
            let mut fakeout = [0.0f32; MAX_N];
            self.eval_raw(input, &mut fakeout);
            let len = out.len();
            out.copy_from_slice(&fakeout[..len]);
        } else {
            self.eval_raw(input, out);
            for o in out[self.n..].iter_mut() {
                *o = 0.0;
            }
        }
    }

    /// The function's own m (inputs).
    pub fn n_in(&self) -> usize {
        self.m
    }

    /// The function's own n (outputs).
    pub fn n_out(&self) -> usize {
        self.n
    }

    /// `func->eval(ctx, func, in, out)`: `input` has at least `m` entries and
    /// `out` at least `n` (guaranteed by [`eval`](PdfFunction::eval)).
    fn eval_raw(&self, input: &[f32], out: &mut [f32]) {
        match &self.kind {
            Kind::Sampled(sa) => self.eval_sample_func(sa, input, out),
            Kind::Exponential(e) => self.eval_exponential_func(e, input, out),
            Kind::Stitching(st) => self.eval_stitching_func(st, input, out),
            Kind::PostScript(code) => self.eval_postscript_func(code, input, out),
        }
    }
}

// ---------------------------------------------------------------------------
// Object access helpers (pdf_dict_get / pdf_array_get_* resolve indirects)
// ---------------------------------------------------------------------------

/// Resolve `o`; a failed resolution reads as null, like MuPDF's
/// `pdf_resolve_indirect` returning `NULL`.
fn resolved(doc: &PdfDocument, o: &Object) -> Object {
    doc.resolve(o).unwrap_or(Object::Null)
}

// MuPDF: pdf_dict_get (resolving)
fn dict_get(doc: &PdfDocument, dict: &Object, key: &str) -> Object {
    match dict.dict_gets(key) {
        Some(v) => resolved(doc, v),
        None => Object::Null,
    }
}

// MuPDF: pdf_array_get_real
fn array_get_real(doc: &PdfDocument, arr: &Object, i: usize) -> f32 {
    match arr.array_get(i) {
        Some(o) => resolved(doc, o).to_real() as f32,
        None => 0.0,
    }
}

// MuPDF: pdf_array_get_int
fn array_get_int(doc: &PdfDocument, arr: &Object, i: usize) -> i32 {
    match arr.array_get(i) {
        Some(o) => resolved(doc, o).to_int() as i32,
        None => 0,
    }
}

// ---------------------------------------------------------------------------
// Common loader
// ---------------------------------------------------------------------------

// MuPDF: pdf_load_function_imp (pdf-function.c:1438)
fn load_function_imp(
    doc: &PdfDocument,
    obj: &Object,
    n_in: usize,
    n_out: usize,
    cycle: &mut Vec<i32>,
    cache: &mut HashMap<i32, Arc<PdfFunction>>,
) -> Result<Arc<PdfFunction>> {
    let num = obj.to_num();

    // MuPDF: pdf_cycle (pdf-object.c:2913) -- walk up the chain of objects
    // being loaded; hitting ourselves (or walking too far) is "recursive".
    if num > 0 {
        let mut depth = 0usize;
        for &up in cycle.iter().rev() {
            if up == num {
                return Err(Error::syntax("recursive function"));
            }
            depth += 1;
            if depth > MAX_CYCLE_STACK_DEPTH {
                return Err(Error::syntax("recursive function"));
            }
        }
    }
    // Not in MuPDF: bound direct-object nesting as well (see module docs).
    if cycle.len() > MAX_CYCLE_STACK_DEPTH {
        return Err(Error::syntax("recursive function"));
    }

    let dict = resolved(doc, obj);
    let ty = dict_get(doc, &dict, "FunctionType").to_int() as i32;
    if !matches!(ty, 0 | 2 | 3 | 4) {
        return Err(Error::syntax(format!("unknown function type ({num} 0 R)")));
    }

    // MuPDF: pdf_find_item -- the store hit comes after the type check.
    if num > 0 {
        if let Some(f) = cache.get(&num) {
            return Ok(f.clone());
        }
    }

    cycle.push(num);
    let loaded = load_function_body(doc, obj, &dict, ty, n_in, n_out, cycle, cache);
    cycle.pop();
    let func = Arc::new(loaded?);

    // MuPDF: pdf_store_item
    if num > 0 {
        cache.insert(num, func.clone());
    }
    Ok(func)
}

/// The part of `pdf_load_function_imp` after the cycle/store checks: the
/// shared Domain/Range reading, then the per-type loader.
#[allow(clippy::too_many_arguments)]
fn load_function_body(
    doc: &PdfDocument,
    obj: &Object,
    dict: &Object,
    ty: i32,
    _n_in: usize,
    n_out: usize,
    cycle: &mut Vec<i32>,
    cache: &mut HashMap<i32, Arc<PdfFunction>>,
) -> Result<PdfFunction> {
    // Required for all.
    let dom = dict_get(doc, dict, "Domain");
    let m = (dom.array_len() / 2).clamp(1, MAX_M);
    let mut domain = [[0.0f32; 2]; MAX_M];
    for (i, d) in domain.iter_mut().enumerate().take(m) {
        d[0] = array_get_real(doc, &dom, i * 2);
        d[1] = array_get_real(doc, &dom, i * 2 + 1);
    }

    // Required for type 0 and type 4, optional otherwise.
    let rng = dict_get(doc, dict, "Range");
    let mut range = [[0.0f32; 2]; MAX_N];
    let has_range;
    let n;
    if rng.is_array() {
        has_range = true;
        n = (rng.array_len() / 2).clamp(1, MAX_N);
        for (i, r) in range.iter_mut().enumerate().take(n) {
            r[0] = array_get_real(doc, &rng, i * 2);
            r[1] = array_get_real(doc, &rng, i * 2 + 1);
        }
    } else {
        has_range = false;
        // Divergence: capped at MAX_N (MuPDF would overrun its arrays).
        n = n_out.min(MAX_N);
    }
    // MuPDF warns here when m != in or n != out, and carries on.

    let mut func = PdfFunction {
        m,
        n,
        domain,
        range,
        has_range,
        kind: Kind::PostScript(Arc::new(Vec::new())),
    };

    func.kind = match ty {
        0 => Kind::Sampled(load_sample_func(doc, obj, dict, &func)?),
        2 => {
            // "exponential functions have at most one input"
            func.m = 1;
            Kind::Exponential(load_exponential_func(doc, dict, &func))
        }
        3 => {
            // "stitching functions have at most one input"
            func.m = 1;
            Kind::Stitching(load_stitching_func(doc, dict, &func, cycle, cache)?)
        }
        _ => Kind::PostScript(Arc::new(load_postscript_func(doc, obj)?)),
    };
    Ok(func)
}

/// `pdf_open_stream(ctx, dict)` for a function object: needs the reference.
fn open_function_stream(doc: &PdfDocument, obj: &Object) -> Result<Vec<u8>> {
    match obj {
        Object::Ref { .. } => doc.open_stream(obj),
        _ => Err(Error::format("function object is not a stream")),
    }
}

// ---------------------------------------------------------------------------
// Type 0: sampled
// ---------------------------------------------------------------------------

// MuPDF: load_sample_func (pdf-function.c:919)
fn load_sample_func(doc: &PdfDocument, obj: &Object, dict: &Object, f: &PdfFunction) -> Result<Sampled> {
    let m = f.m;
    let n = f.n;

    let size_obj = dict_get(doc, dict, "Size");
    if size_obj.array_len() < m {
        return Err(Error::syntax("too few sample function dimension sizes"));
    }
    let mut size = [0i32; MAX_M];
    for (i, s) in size.iter_mut().enumerate().take(m) {
        *s = array_get_int(doc, &size_obj, i);
        if *s <= 0 {
            // "non-positive sample function dimension size"
            *s = 1;
        }
    }

    let bps = dict_get(doc, dict, "BitsPerSample").to_int() as i32;

    let mut encode = [[0.0f32; 2]; MAX_M];
    for i in 0..m {
        encode[i][0] = 0.0;
        encode[i][1] = (size[i] - 1) as f32;
    }
    let enc = dict_get(doc, dict, "Encode");
    if enc.is_array() {
        let ranges = m.min(enc.array_len() / 2);
        for (i, e) in encode.iter_mut().enumerate().take(ranges) {
            e[0] = array_get_real(doc, &enc, i * 2);
            e[1] = array_get_real(doc, &enc, i * 2 + 1);
        }
    }

    let mut decode = [[0.0f32; 2]; MAX_N];
    decode[..n].copy_from_slice(&f.range[..n]);
    let dec = dict_get(doc, dict, "Decode");
    if dec.is_array() {
        let ranges = n.min(dec.array_len() / 2);
        for (i, d) in decode.iter_mut().enumerate().take(ranges) {
            d[0] = array_get_real(doc, &dec, i * 2);
            d[1] = array_get_real(doc, &dec, i * 2 + 1);
        }
    }

    let mut samplecount = n as i32;
    for &s in size.iter().take(m) {
        if samplecount > MAX_SAMPLE_FUNCTION_SIZE / s {
            return Err(Error::syntax("sample function too large"));
        }
        samplecount *= s;
    }
    if samplecount > MAX_SAMPLE_FUNCTION_SIZE {
        return Err(Error::syntax("sample function too large"));
    }

    let data = open_function_stream(doc, obj)?;
    // Don't pre-allocate 400 MB for a table the stream can't possibly fill;
    // a short stream still fails exactly where MuPDF's would.
    let cap_by_data = data.len().saturating_mul(8) / (bps.max(1) as usize) + 1;
    let mut samples = Vec::with_capacity((samplecount as usize).min(cap_by_data));
    let mut stm = Stream::from_vec(data);

    // Read samples.
    for _ in 0..samplecount {
        if stm.is_eof_bits()? {
            return Err(Error::syntax("truncated sample function stream"));
        }
        let s = match bps {
            1 => stm.read_bits(1)? as f32,
            2 => stm.read_bits(2)? as f32 / 3.0,
            4 => stm.read_bits(4)? as f32 / 15.0,
            // fz_read_byte gives EOF (-1) at end; unreachable after the
            // eof check, but keep the C value anyway.
            8 => match stm.read_byte()? {
                Some(b) => b as f32 / 255.0,
                None => -1.0 / 255.0,
            },
            12 => stm.read_bits(12)? as f32 / 4095.0,
            16 => read_uint_be(&mut stm, 2, "premature end of file in int16")? as f32 / 65535.0,
            24 => read_uint_be(&mut stm, 3, "premature end of file in int24")? as f32 / 16_777_215.0,
            32 => read_uint_be(&mut stm, 4, "premature end of file in int32")? as f32 / 4_294_967_295.0,
            _ => return Err(Error::syntax(format!("sample stream bit depth {bps} unsupported"))),
        };
        samples.push(s);
    }

    Ok(Sampled {
        size,
        encode,
        decode,
        samples: Arc::new(samples),
    })
}

// MuPDF: fz_read_uint16 / fz_read_uint24 / fz_read_uint32 (stream-read.c:278)
/// Read `nbytes` big-endian bytes; any EOF among them is a format error.
fn read_uint_be(stm: &mut Stream, nbytes: usize, msg: &str) -> Result<u32> {
    let mut x = 0u32;
    let mut short = false;
    for _ in 0..nbytes {
        match stm.read_byte()? {
            Some(b) => x = (x << 8) | b as u32,
            None => short = true,
        }
    }
    if short {
        return Err(Error::format(msg));
    }
    Ok(x)
}

impl PdfFunction {
    fn sample_at(sa: &Sampled, idx: i64) -> f32 {
        if idx < 0 {
            return 0.0;
        }
        sa.samples.get(idx as usize).copied().unwrap_or(0.0)
    }

    // MuPDF: interpolate_sample (pdf-function.c:1047)
    #[allow(clippy::too_many_arguments)]
    fn interpolate_sample(
        sa: &Sampled,
        scale: &[i64; MAX_M],
        e0: &[i64; MAX_M],
        e1: &[i64; MAX_M],
        efrac: &[f32; MAX_M],
        dim: usize,
        idx: i64,
    ) -> f32 {
        let idx0 = e0[dim] * scale[dim] + idx;
        let idx1 = e1[dim] * scale[dim] + idx;

        let (a, b) = if dim == 0 {
            (Self::sample_at(sa, idx0), Self::sample_at(sa, idx1))
        } else {
            (
                Self::interpolate_sample(sa, scale, e0, e1, efrac, dim - 1, idx0),
                Self::interpolate_sample(sa, scale, e0, e1, efrac, dim - 1, idx1),
            )
        };

        a + (b - a) * efrac[dim]
    }

    // MuPDF: eval_sample_func (pdf-function.c:1070)
    fn eval_sample_func(&self, sa: &Sampled, input: &[f32], out: &mut [f32]) {
        let m = self.m;
        let n = self.n;
        let mut e0 = [0i64; MAX_M];
        let mut e1 = [0i64; MAX_M];
        let mut scale = [0i64; MAX_M];
        let mut efrac = [0.0f32; MAX_M];

        // Encode input coordinates.
        for i in 0..m {
            let d = self.domain[i];
            let mut x = fz_clamp(input[i], d[0], d[1]);
            x = lerp(x, d[0], d[1], sa.encode[i][0], sa.encode[i][1]);
            x = fz_clamp(x, 0.0, (sa.size[i] - 1) as f32);
            // C: `e0[i] = floorf(x)` into an int (saturating here, see docs).
            let f0 = x.floor() as i32;
            let f1 = x.ceil() as i32;
            e0[i] = f0 as i64;
            e1[i] = f1 as i64;
            efrac[i] = x - f0 as f32;
        }

        scale[0] = n as i64;
        for i in 1..m {
            scale[i] = scale[i - 1] * sa.size[i - 1] as i64;
        }

        let nn = n as i64;
        for i in 0..n {
            let ii = i as i64;
            let dec = sa.decode[i];
            let rng = self.range[i];
            if m == 1 {
                let a = Self::sample_at(sa, e0[0] * nn + ii);
                let b = Self::sample_at(sa, e1[0] * nn + ii);

                let ab = a + (b - a) * efrac[0];

                out[i] = lerp(ab, 0.0, 1.0, dec[0], dec[1]);
                out[i] = fz_clamp(out[i], rng[0], rng[1]);
            } else if m == 2 {
                let s0 = nn;
                let s1 = s0 * sa.size[0] as i64;

                let a = Self::sample_at(sa, e0[0] * s0 + e0[1] * s1 + ii);
                let b = Self::sample_at(sa, e1[0] * s0 + e0[1] * s1 + ii);
                let c = Self::sample_at(sa, e0[0] * s0 + e1[1] * s1 + ii);
                let d = Self::sample_at(sa, e1[0] * s0 + e1[1] * s1 + ii);

                let ab = a + (b - a) * efrac[0];
                let cd = c + (d - c) * efrac[0];
                let abcd = ab + (cd - ab) * efrac[1];

                out[i] = lerp(abcd, 0.0, 1.0, dec[0], dec[1]);
                out[i] = fz_clamp(out[i], rng[0], rng[1]);
            } else {
                let x = Self::interpolate_sample(sa, &scale, &e0, &e1, &efrac, m - 1, ii);
                out[i] = lerp(x, 0.0, 1.0, dec[0], dec[1]);
                out[i] = fz_clamp(out[i], rng[0], rng[1]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Type 2: exponential
// ---------------------------------------------------------------------------

// MuPDF: load_exponential_func (pdf-function.c:1141)
fn load_exponential_func(doc: &PdfDocument, dict: &Object, f: &PdfFunction) -> Exponential {
    let n = f.n;
    let big_n = dict_get(doc, dict, "N").to_real() as f32;

    // MuPDF only warns about illegal domains here (non-integer N with a
    // negative domain, negative N with a domain touching zero); eval copes.

    let mut c0 = [0.0f32; MAX_N];
    let mut c1 = [0.0f32; MAX_N];
    for i in 0..n {
        c0[i] = 0.0;
        c1[i] = 1.0;
    }

    let o = dict_get(doc, dict, "C0");
    if o.is_array() {
        let ranges = n.min(o.array_len());
        for (i, c) in c0.iter_mut().enumerate().take(ranges) {
            *c = array_get_real(doc, &o, i);
        }
    }

    let o = dict_get(doc, dict, "C1");
    if o.is_array() {
        let ranges = n.min(o.array_len());
        for (i, c) in c1.iter_mut().enumerate().take(ranges) {
            *c = array_get_real(doc, &o, i);
        }
    }

    Exponential { n: big_n, c0, c1 }
}

/// C's `func->n != (int)func->n` (saturating cast; NaN counts as non-integer).
#[inline]
fn is_non_integer(x: f32) -> bool {
    x != (x as i32) as f32
}

impl PdfFunction {
    // MuPDF: eval_exponential_func (pdf-function.c:1205)
    fn eval_exponential_func(&self, e: &Exponential, input: &[f32], out: &mut [f32]) {
        let x = fz_clamp(input[0], self.domain[0][0], self.domain[0][1]);

        // Default output is zero, which is suitable for violated constraints.
        if (is_non_integer(e.n) && x < 0.0) || (e.n < 0.0 && x == 0.0) {
            for o in out[..self.n].iter_mut() {
                *o = 0.0;
            }
            return;
        }

        let tmp = x.powf(e.n);
        for i in 0..self.n {
            out[i] = e.c0[i] + tmp * (e.c1[i] - e.c0[i]);
            if self.has_range {
                out[i] = fz_clamp(out[i], self.range[i][0], self.range[i][1]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Type 3: stitching
// ---------------------------------------------------------------------------

// MuPDF: load_stitching_func (pdf-function.c:1236)
fn load_stitching_func(
    doc: &PdfDocument,
    dict: &Object,
    f: &PdfFunction,
    cycle: &mut Vec<i32>,
    cache: &mut HashMap<i32, Arc<PdfFunction>>,
) -> Result<Stitching> {
    let fobj = dict_get(doc, dict, "Functions");
    if !fobj.is_array() {
        return Err(Error::syntax("stitching function has no input functions"));
    }

    let k = fobj.array_len();
    if k < 1 {
        return Err(Error::syntax("no sub-functions in stitching function"));
    }
    if k > MAX_STITCHING {
        return Err(Error::syntax("too many sub-functions in stitching function"));
    }

    let mut funcs = Vec::with_capacity(k);
    for i in 0..k {
        let sub = fobj.array_get(i).cloned().unwrap_or(Object::Null);
        let sf = load_function_imp(doc, &sub, 1, f.n, cycle, cache)?;
        // MuPDF warns on sub-function arity mismatches and carries on.
        funcs.push(sf);
    }

    let bobj = dict_get(doc, dict, "Bounds");
    if !bobj.is_array() {
        return Err(Error::syntax("stitching function has no bounds"));
    }
    if bobj.array_len() < k - 1 {
        return Err(Error::syntax("too few subfunction boundaries"));
    }
    let mut bounds = vec![0.0f32; k - 1];
    for i in 0..k - 1 {
        bounds[i] = array_get_real(doc, &bobj, i);
        if i > 0 && bounds[i - 1] > bounds[i] {
            return Err(Error::syntax(format!("subfunction {i} boundary out of range")));
        }
    }

    let mut encode = vec![0.0f32; k * 2];
    let eobj = dict_get(doc, dict, "Encode");
    if eobj.is_array() {
        let ranges = k.min(eobj.array_len() / 2);
        for i in 0..ranges {
            encode[i * 2] = array_get_real(doc, &eobj, i * 2);
            encode[i * 2 + 1] = array_get_real(doc, &eobj, i * 2 + 1);
        }
    }

    Ok(Stitching { funcs, bounds, encode })
}

impl PdfFunction {
    // MuPDF: eval_stitching_func (pdf-function.c:1334)
    fn eval_stitching_func(&self, st: &Stitching, input: &[f32], out: &mut [f32]) {
        let k = st.funcs.len();
        let bounds = &st.bounds;
        let d = self.domain[0];
        let mut x = fz_clamp(input[0], d[0], d[1]);

        let mut i = 0;
        while i < k - 1 {
            if x < bounds[i] {
                break;
            }
            i += 1;
        }

        let (low, high) = if i == 0 && k == 1 {
            (d[0], d[1])
        } else if i == 0 {
            (d[0], bounds[0])
        } else if i == k - 1 {
            (bounds[k - 2], d[1])
        } else {
            (bounds[i - 1], bounds[i])
        };

        x = lerp(x, low, high, st.encode[i * 2], st.encode[i * 2 + 1]);

        st.funcs[i].eval(&[x], &mut out[..self.n]);
    }
}

// ---------------------------------------------------------------------------
// Type 4: PostScript calculator
// ---------------------------------------------------------------------------

/// The `PS_OP_*` operators, in `ps_op_names` order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PsOp {
    Abs,
    Add,
    And,
    Atan,
    Bitshift,
    Ceiling,
    Copy,
    Cos,
    Cvi,
    Cvr,
    Div,
    Dup,
    Eq,
    Exch,
    Exp,
    False,
    Floor,
    Ge,
    Gt,
    Idiv,
    If,
    Ifelse,
    Index,
    Le,
    Ln,
    Log,
    Lt,
    Mod,
    Mul,
    Ne,
    Neg,
    Not,
    Or,
    Pop,
    Return,
    Roll,
    Round,
    Sin,
    Sqrt,
    Sub,
    True,
    Truncate,
    Xor,
}

/// `ps_op_names` (sorted, as MuPDF's binary search needs).
const PS_OP_NAMES: [(&str, PsOp); 43] = [
    ("abs", PsOp::Abs),
    ("add", PsOp::Add),
    ("and", PsOp::And),
    ("atan", PsOp::Atan),
    ("bitshift", PsOp::Bitshift),
    ("ceiling", PsOp::Ceiling),
    ("copy", PsOp::Copy),
    ("cos", PsOp::Cos),
    ("cvi", PsOp::Cvi),
    ("cvr", PsOp::Cvr),
    ("div", PsOp::Div),
    ("dup", PsOp::Dup),
    ("eq", PsOp::Eq),
    ("exch", PsOp::Exch),
    ("exp", PsOp::Exp),
    ("false", PsOp::False),
    ("floor", PsOp::Floor),
    ("ge", PsOp::Ge),
    ("gt", PsOp::Gt),
    ("idiv", PsOp::Idiv),
    ("if", PsOp::If),
    ("ifelse", PsOp::Ifelse),
    ("index", PsOp::Index),
    ("le", PsOp::Le),
    ("ln", PsOp::Ln),
    ("log", PsOp::Log),
    ("lt", PsOp::Lt),
    ("mod", PsOp::Mod),
    ("mul", PsOp::Mul),
    ("ne", PsOp::Ne),
    ("neg", PsOp::Neg),
    ("not", PsOp::Not),
    ("or", PsOp::Or),
    ("pop", PsOp::Pop),
    ("return", PsOp::Return),
    ("roll", PsOp::Roll),
    ("round", PsOp::Round),
    ("sin", PsOp::Sin),
    ("sqrt", PsOp::Sqrt),
    ("sub", PsOp::Sub),
    ("true", PsOp::True),
    ("truncate", PsOp::Truncate),
    ("xor", PsOp::Xor),
];

/// A code-array `psobj` (`PS_BOOL` / `PS_INT` / `PS_REAL` / `PS_OPERATOR` /
/// `PS_BLOCK`). `Unset` is the realloc'd-but-never-written slot at
/// `opptr + 1` of an `if` -- never executed.
#[derive(Clone, Copy, Debug, PartialEq)]
enum PsCode {
    Bool(bool),
    Int(i32),
    Real(f32),
    Op(PsOp),
    Block(usize),
    Unset,
}

/// A stack `psobj` -- only bools, ints and reals ever live on the stack.
#[derive(Clone, Copy, Debug, PartialEq)]
enum PsVal {
    Bool(bool),
    Int(i32),
    Real(f32),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PsType {
    Bool,
    Int,
    Real,
}

impl PsVal {
    fn ty(self) -> PsType {
        match self {
            PsVal::Bool(_) => PsType::Bool,
            PsVal::Int(_) => PsType::Int,
            PsVal::Real(_) => PsType::Real,
        }
    }
}

/// `ps_stack`: a fixed 100-slot array, never cleared on pop (stale slots are
/// observable through `-1 index`, as in C).
struct PsStack {
    stack: [PsVal; PS_STACK_SIZE],
    sp: i32,
}

impl PsStack {
    // MuPDF: ps_init_stack (pdf-function.c:171) -- memset 0 = `false` bools.
    fn new() -> Self {
        PsStack {
            stack: [PsVal::Bool(false); PS_STACK_SIZE],
            sp: 0,
        }
    }

    // MuPDF: ps_overflow (pdf-function.c:177)
    fn overflow(&self, n: i32) -> bool {
        n < 0 || self.sp as i64 + n as i64 >= PS_STACK_SIZE as i64
    }

    // MuPDF: ps_underflow (pdf-function.c:182)
    fn underflow(&self, n: i32) -> bool {
        n < 0 || n > self.sp
    }

    fn top(&self, depth: i32) -> PsVal {
        self.stack[(self.sp - depth) as usize]
    }

    // MuPDF: ps_is_type (pdf-function.c:187)
    fn is_type(&self, t: PsType) -> bool {
        !self.underflow(1) && self.top(1).ty() == t
    }

    // MuPDF: ps_is_type2 (pdf-function.c:192)
    fn is_type2(&self, t: PsType) -> bool {
        !self.underflow(2) && self.top(1).ty() == t && self.top(2).ty() == t
    }

    fn push(&mut self, v: PsVal) {
        self.stack[self.sp as usize] = v;
        self.sp += 1;
    }

    // MuPDF: ps_push_bool (pdf-function.c:198)
    fn push_bool(&mut self, b: bool) {
        if !self.overflow(1) {
            self.push(PsVal::Bool(b));
        }
    }

    // MuPDF: ps_push_int (pdf-function.c:209)
    fn push_int(&mut self, n: i32) {
        if !self.overflow(1) {
            self.push(PsVal::Int(n));
        }
    }

    // MuPDF: ps_push_real (pdf-function.c:220)
    fn push_real(&mut self, n: f32) {
        if !self.overflow(1) {
            // Push 1.0 for NaN, as it's a small known value that won't cause
            // a divide by 0. Same reason as in fz_atof.
            let n = if n.is_nan() { 1.0 } else { n };
            self.push(PsVal::Real(fz_clamp(n, -f32::MAX, f32::MAX)));
        }
    }

    // MuPDF: ps_pop_bool (pdf-function.c:237)
    fn pop_bool(&mut self) -> bool {
        if !self.underflow(1) {
            if let PsVal::Bool(b) = self.top(1) {
                self.sp -= 1;
                return b;
            }
        }
        false
    }

    // MuPDF: ps_pop_int (pdf-function.c:248)
    fn pop_int(&mut self) -> i32 {
        if !self.underflow(1) {
            match self.top(1) {
                PsVal::Int(i) => {
                    self.sp -= 1;
                    return i;
                }
                PsVal::Real(f) => {
                    self.sp -= 1;
                    return f as i32;
                }
                PsVal::Bool(_) => {}
            }
        }
        0
    }

    // MuPDF: ps_pop_real (pdf-function.c:261)
    fn pop_real(&mut self) -> f32 {
        if !self.underflow(1) {
            match self.top(1) {
                PsVal::Int(i) => {
                    self.sp -= 1;
                    return i as f32;
                }
                PsVal::Real(f) => {
                    self.sp -= 1;
                    return f;
                }
                PsVal::Bool(_) => {}
            }
        }
        0.0
    }

    // MuPDF: ps_copy (pdf-function.c:274)
    fn copy(&mut self, n: i32) {
        if !self.underflow(n) && !self.overflow(n) {
            let sp = self.sp as usize;
            let n = n as usize;
            self.stack.copy_within(sp - n..sp, sp);
            self.sp += n as i32;
        }
    }

    // MuPDF: ps_roll (pdf-function.c:284)
    fn roll(&mut self, n: i32, j: i32) {
        if self.underflow(n) || j == 0 || n == 0 {
            return;
        }

        let mut j = j;
        if j >= 0 {
            j %= n;
        } else {
            j = j.wrapping_neg() % n;
            if j != 0 {
                j = n - j;
            }
        }

        let sp = self.sp as usize;
        let nu = n as usize;
        if j * 2 > n {
            for _ in j..n {
                let tmp = self.stack[sp - nu];
                self.stack.copy_within(sp - nu + 1..sp, sp - nu);
                self.stack[sp - 1] = tmp;
            }
        } else {
            for _ in 0..j {
                let tmp = self.stack[sp - 1];
                self.stack.copy_within(sp - nu..sp - 1, sp - nu + 1);
                self.stack[sp - nu] = tmp;
            }
        }
    }

    // MuPDF: ps_index (pdf-function.c:323)
    fn index(&mut self, n: i32) {
        if !self.overflow(1) && !self.underflow(n.wrapping_add(1)) {
            // n >= -1 here; n == -1 reads the stale slot at `sp`.
            let src = (self.sp - n - 1) as usize;
            self.stack[self.sp as usize] = self.stack[src];
            self.sp += 1;
        }
    }
}

/// `DIV_BY_ZERO(a, b, min, max)` -- the sign of the would-be quotient.
#[inline]
fn div_by_zero<T>(a_neg: bool, b_neg: bool, min: T, max: T) -> T {
    if a_neg ^ b_neg { min } else { max }
}

// MuPDF: ps_run (pdf-function.c:333)
fn ps_run(code: &[PsCode], st: &mut PsStack, mut pc: usize) {
    loop {
        let Some(&c) = code.get(pc) else {
            // Cannot happen for parsed code (every block ends in `return`).
            return;
        };
        match c {
            PsCode::Int(i) => {
                st.push_int(i);
                pc += 1;
            }
            PsCode::Real(f) => {
                st.push_real(f);
                pc += 1;
            }
            PsCode::Op(op) => {
                pc += 1;
                match op {
                    PsOp::Abs => {
                        if st.is_type(PsType::Int) {
                            let i = st.pop_int();
                            st.push_int(i.wrapping_abs());
                        } else {
                            let r = st.pop_real();
                            st.push_real(r.abs());
                        }
                    }
                    PsOp::Add => {
                        if st.is_type2(PsType::Int) {
                            let i2 = st.pop_int();
                            let i1 = st.pop_int();
                            st.push_int(i1.wrapping_add(i2));
                        } else {
                            let r2 = st.pop_real();
                            let r1 = st.pop_real();
                            st.push_real(r1 + r2);
                        }
                    }
                    PsOp::And => {
                        if st.is_type2(PsType::Int) {
                            let i2 = st.pop_int();
                            let i1 = st.pop_int();
                            st.push_int(i1 & i2);
                        } else {
                            let b2 = st.pop_bool();
                            let b1 = st.pop_bool();
                            st.push_bool(b1 && b2);
                        }
                    }
                    PsOp::Atan => {
                        let r2 = st.pop_real();
                        let r1 = st.pop_real();
                        let mut r = r1.atan2(r2) * FZ_RADIAN;
                        if r < 0.0 {
                            r += 360.0;
                        }
                        st.push_real(r);
                    }
                    PsOp::Bitshift => {
                        let i2 = st.pop_int();
                        let i1 = st.pop_int();
                        if i2 > 0 && i2 < 32 {
                            st.push_int(((i1 as u32) << i2) as i32);
                        } else if i2 < 0 && i2 > -32 {
                            st.push_int(((i1 as u32) >> -i2) as i32);
                        } else {
                            st.push_int(i1);
                        }
                    }
                    PsOp::Ceiling => {
                        let r = st.pop_real();
                        st.push_real(r.ceil());
                    }
                    PsOp::Copy => {
                        let n = st.pop_int();
                        st.copy(n);
                    }
                    PsOp::Cos => {
                        let r = st.pop_real();
                        st.push_real((r / FZ_RADIAN).cos());
                    }
                    PsOp::Cvi => {
                        let i = st.pop_int();
                        st.push_int(i);
                    }
                    PsOp::Cvr => {
                        let r = st.pop_real();
                        st.push_real(r);
                    }
                    PsOp::Div => {
                        let r2 = st.pop_real();
                        let r1 = st.pop_real();
                        if r2.abs() >= f32::EPSILON {
                            st.push_real(r1 / r2);
                        } else {
                            st.push_real(div_by_zero(r1 < 0.0, r2 < 0.0, -f32::MAX, f32::MAX));
                        }
                    }
                    PsOp::Dup => st.copy(1),
                    PsOp::Eq => {
                        if st.is_type2(PsType::Bool) {
                            let b2 = st.pop_bool();
                            let b1 = st.pop_bool();
                            st.push_bool(b1 == b2);
                        } else if st.is_type2(PsType::Int) {
                            let i2 = st.pop_int();
                            let i1 = st.pop_int();
                            st.push_bool(i1 == i2);
                        } else {
                            let r2 = st.pop_real();
                            let r1 = st.pop_real();
                            st.push_bool(r1 == r2);
                        }
                    }
                    PsOp::Exch => st.roll(2, 1),
                    PsOp::Exp => {
                        let r2 = st.pop_real();
                        let r1 = st.pop_real();
                        st.push_real(r1.powf(r2));
                    }
                    PsOp::False => st.push_bool(false),
                    PsOp::Floor => {
                        let r = st.pop_real();
                        st.push_real(r.floor());
                    }
                    PsOp::Ge => {
                        if st.is_type2(PsType::Int) {
                            let i2 = st.pop_int();
                            let i1 = st.pop_int();
                            st.push_bool(i1 >= i2);
                        } else {
                            let r2 = st.pop_real();
                            let r1 = st.pop_real();
                            st.push_bool(r1 >= r2);
                        }
                    }
                    PsOp::Gt => {
                        if st.is_type2(PsType::Int) {
                            let i2 = st.pop_int();
                            let i1 = st.pop_int();
                            st.push_bool(i1 > i2);
                        } else {
                            let r2 = st.pop_real();
                            let r1 = st.pop_real();
                            st.push_bool(r1 > r2);
                        }
                    }
                    PsOp::Idiv => {
                        let i2 = st.pop_int();
                        let i1 = st.pop_int();
                        if i2 == 0 {
                            st.push_int(div_by_zero(i1 < 0, i2 < 0, i32::MIN, i32::MAX));
                        } else if i1 == i32::MIN && i2 == -1 {
                            st.push_int(i32::MAX);
                        } else {
                            st.push_int(i1 / i2);
                        }
                    }
                    PsOp::Index => {
                        let n = st.pop_int();
                        st.index(n);
                    }
                    PsOp::Le => {
                        if st.is_type2(PsType::Int) {
                            let i2 = st.pop_int();
                            let i1 = st.pop_int();
                            st.push_bool(i1 <= i2);
                        } else {
                            let r2 = st.pop_real();
                            let r1 = st.pop_real();
                            st.push_bool(r1 <= r2);
                        }
                    }
                    PsOp::Ln => {
                        let r = st.pop_real();
                        st.push_real(r.ln());
                    }
                    PsOp::Log => {
                        let r = st.pop_real();
                        st.push_real(r.log10());
                    }
                    PsOp::Lt => {
                        if st.is_type2(PsType::Int) {
                            let i2 = st.pop_int();
                            let i1 = st.pop_int();
                            st.push_bool(i1 < i2);
                        } else {
                            let r2 = st.pop_real();
                            let r1 = st.pop_real();
                            st.push_bool(r1 < r2);
                        }
                    }
                    PsOp::Mod => {
                        let i2 = st.pop_int();
                        let i1 = st.pop_int();
                        if i2 == 0 {
                            st.push_int(div_by_zero(i1 < 0, i2 < 0, i32::MIN, i32::MAX));
                        } else if i1 == i32::MIN && i2 == -1 {
                            st.push_int(0);
                        } else {
                            st.push_int(i1 % i2);
                        }
                    }
                    PsOp::Mul => {
                        if st.is_type2(PsType::Int) {
                            let i2 = st.pop_int();
                            let i1 = st.pop_int();
                            st.push_int(i1.wrapping_mul(i2));
                        } else {
                            let r2 = st.pop_real();
                            let r1 = st.pop_real();
                            st.push_real(r1 * r2);
                        }
                    }
                    PsOp::Ne => {
                        if st.is_type2(PsType::Bool) {
                            let b2 = st.pop_bool();
                            let b1 = st.pop_bool();
                            st.push_bool(b1 != b2);
                        } else if st.is_type2(PsType::Int) {
                            let i2 = st.pop_int();
                            let i1 = st.pop_int();
                            st.push_bool(i1 != i2);
                        } else {
                            let r2 = st.pop_real();
                            let r1 = st.pop_real();
                            st.push_bool(r1 != r2);
                        }
                    }
                    PsOp::Neg => {
                        if st.is_type(PsType::Int) {
                            let i = st.pop_int();
                            st.push_int(i.wrapping_neg());
                        } else {
                            let r = st.pop_real();
                            st.push_real(-r);
                        }
                    }
                    PsOp::Not => {
                        if st.is_type(PsType::Bool) {
                            let b = st.pop_bool();
                            st.push_bool(!b);
                        } else {
                            let i = st.pop_int();
                            st.push_int(!i);
                        }
                    }
                    PsOp::Or => {
                        if st.is_type2(PsType::Bool) {
                            let b2 = st.pop_bool();
                            let b1 = st.pop_bool();
                            st.push_bool(b1 || b2);
                        } else {
                            let i2 = st.pop_int();
                            let i1 = st.pop_int();
                            st.push_int(i1 | i2);
                        }
                    }
                    PsOp::Pop => {
                        if !st.underflow(1) {
                            st.sp -= 1;
                        }
                    }
                    PsOp::Roll => {
                        let i2 = st.pop_int();
                        let i1 = st.pop_int();
                        st.roll(i1, i2);
                    }
                    PsOp::Round => {
                        if !st.is_type(PsType::Int) {
                            let r = st.pop_real();
                            st.push_real(if r >= 0.0 { (r + 0.5).floor() } else { (r - 0.5).ceil() });
                        }
                    }
                    PsOp::Sin => {
                        let r = st.pop_real();
                        st.push_real((r / FZ_RADIAN).sin());
                    }
                    PsOp::Sqrt => {
                        let r = st.pop_real();
                        st.push_real(r.sqrt());
                    }
                    PsOp::Sub => {
                        if st.is_type2(PsType::Int) {
                            let i2 = st.pop_int();
                            let i1 = st.pop_int();
                            st.push_int(i1.wrapping_sub(i2));
                        } else {
                            let r2 = st.pop_real();
                            let r1 = st.pop_real();
                            st.push_real(r1 - r2);
                        }
                    }
                    PsOp::True => st.push_bool(true),
                    PsOp::Truncate => {
                        if !st.is_type(PsType::Int) {
                            let r = st.pop_real();
                            st.push_real(if r >= 0.0 { r.floor() } else { r.ceil() });
                        }
                    }
                    PsOp::Xor => {
                        if st.is_type2(PsType::Bool) {
                            let b2 = st.pop_bool();
                            let b1 = st.pop_bool();
                            st.push_bool(b1 ^ b2);
                        } else {
                            let i2 = st.pop_int();
                            let i1 = st.pop_int();
                            st.push_int(i1 ^ i2);
                        }
                    }
                    PsOp::If => {
                        let b1 = st.pop_bool();
                        if b1 {
                            if let Some(&PsCode::Block(blk)) = code.get(pc + 1) {
                                ps_run(code, st, blk);
                            }
                        }
                        match code.get(pc + 2) {
                            Some(&PsCode::Block(next)) => pc = next,
                            _ => return,
                        }
                    }
                    PsOp::Ifelse => {
                        let b1 = st.pop_bool();
                        let branch = if b1 { code.get(pc + 1) } else { code.get(pc) };
                        if let Some(&PsCode::Block(blk)) = branch {
                            ps_run(code, st, blk);
                        }
                        match code.get(pc + 2) {
                            Some(&PsCode::Block(next)) => pc = next,
                            _ => return,
                        }
                    }
                    PsOp::Return => return,
                }
            }
            // "foreign object in calculator function" -- includes PS_BOOL
            // code objects from `true`/`false` literals (see module docs).
            PsCode::Bool(_) | PsCode::Block(_) | PsCode::Unset => return,
        }
    }
}

/// Write `v` at `code[idx]`, growing the array as `resize_code` would.
fn put_code(code: &mut Vec<PsCode>, idx: usize, v: PsCode) {
    if idx >= code.len() {
        code.resize(idx + 1, PsCode::Unset);
    }
    code[idx] = v;
}

// MuPDF: parse_code (pdf-function.c:733)
fn parse_code(stm: &mut Stream, code: &mut Vec<PsCode>, codeptr: &mut usize, depth: u32) -> Result<()> {
    if depth > MAX_PS_NESTING {
        return Err(Error::syntax("too much nesting in calculator function"));
    }

    loop {
        match lex(stm)? {
            Token::Eof => return Err(Error::syntax("truncated calculator function")),

            // C stores `buf->i` (int64) into an int: truncating conversion.
            Token::Int(i) => {
                put_code(code, *codeptr, PsCode::Int(i as i32));
                *codeptr += 1;
            }
            Token::True => {
                put_code(code, *codeptr, PsCode::Bool(true));
                *codeptr += 1;
            }
            Token::False => {
                put_code(code, *codeptr, PsCode::Bool(false));
                *codeptr += 1;
            }
            Token::Real(f) => {
                put_code(code, *codeptr, PsCode::Real(f as f32));
                *codeptr += 1;
            }

            Token::OpenBrace => {
                let opptr = *codeptr;
                *codeptr += 4;
                if *codeptr >= code.len() {
                    code.resize(*codeptr + 1, PsCode::Unset);
                }

                let ifptr = *codeptr;
                parse_code(stm, code, codeptr, depth + 1)?;

                let mut tok = lex(stm)?;
                let elseptr = if matches!(tok, Token::OpenBrace) {
                    let e = *codeptr;
                    parse_code(stm, code, codeptr, depth + 1)?;
                    tok = lex(stm)?;
                    Some(e)
                } else {
                    None
                };

                let Token::Keyword(kw) = tok else {
                    return Err(Error::syntax("missing keyword in 'if-else' context"));
                };

                if kw == b"if" {
                    if elseptr.is_some() {
                        return Err(Error::syntax("too many branches for 'if'"));
                    }
                    put_code(code, opptr, PsCode::Op(PsOp::If));
                    put_code(code, opptr + 2, PsCode::Block(ifptr));
                    put_code(code, opptr + 3, PsCode::Block(*codeptr));
                } else if kw == b"ifelse" {
                    let Some(elseptr) = elseptr else {
                        return Err(Error::syntax("not enough branches for 'ifelse'"));
                    };
                    put_code(code, opptr, PsCode::Op(PsOp::Ifelse));
                    put_code(code, opptr + 1, PsCode::Block(elseptr));
                    put_code(code, opptr + 2, PsCode::Block(ifptr));
                    put_code(code, opptr + 3, PsCode::Block(*codeptr));
                } else {
                    return Err(Error::syntax(format!(
                        "unknown keyword in 'if-else' context: '{}'",
                        String::from_utf8_lossy(&kw)
                    )));
                }
            }

            Token::CloseBrace => {
                put_code(code, *codeptr, PsCode::Op(PsOp::Return));
                *codeptr += 1;
                return Ok(());
            }

            Token::Keyword(kw) => {
                // MuPDF binary-searches the sorted ps_op_names; an exact
                // lookup finds the same entry.
                let Some(&(_, op)) = PS_OP_NAMES.iter().find(|(name, _)| name.as_bytes() == kw.as_slice()) else {
                    return Err(Error::syntax(format!(
                        "unknown operator: '{}'",
                        String::from_utf8_lossy(&kw)
                    )));
                };
                if op == PsOp::Ifelse {
                    return Err(Error::syntax("illegally positioned ifelse operator in function"));
                }
                if op == PsOp::If {
                    return Err(Error::syntax("illegally positioned if operator in function"));
                }
                put_code(code, *codeptr, PsCode::Op(op));
                *codeptr += 1;
            }

            _ => return Err(Error::syntax("calculator function syntax error")),
        }
    }
}

// MuPDF: load_postscript_func (pdf-function.c:855)
fn load_postscript_func(doc: &PdfDocument, obj: &Object) -> Result<Vec<PsCode>> {
    let data = open_function_stream(doc, obj)?;
    let mut stm = Stream::from_vec(data);

    if !matches!(lex(&mut stm)?, Token::OpenBrace) {
        return Err(Error::syntax("stream is not a calculator function"));
    }

    let mut code = Vec::new();
    let mut codeptr = 0usize;
    parse_code(&mut stm, &mut code, &mut codeptr, 0)?;
    Ok(code)
}

impl PdfFunction {
    // MuPDF: eval_postscript_func (pdf-function.c:893)
    fn eval_postscript_func(&self, code: &[PsCode], input: &[f32], out: &mut [f32]) {
        let mut st = PsStack::new();

        for i in 0..self.m {
            let x = fz_clamp(input[i], self.domain[i][0], self.domain[i][1]);
            st.push_real(x);
        }

        ps_run(code, &mut st, 0);

        for i in (0..self.n).rev() {
            let x = st.pop_real();
            out[i] = fz_clamp(x, self.range[i][0], self.range[i][1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny PDF: object 1 catalog, 2 page tree, then `objs` as
    /// objects 3, 4, … (raw bodies, so streams can carry binary samples),
    /// and one blank page last (the xref layer wants a non-empty page tree).
    fn build(objs: &[Vec<u8>]) -> PdfDocument {
        let page_num = 3 + objs.len();
        let mut bodies: Vec<Vec<u8>> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            format!("<< /Type /Pages /Kids [{page_num} 0 R] /Count 1 >>").into_bytes(),
        ];
        bodies.extend(objs.iter().cloned());
        bodies.push(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] >>".to_vec());
        let mut pdf: Vec<u8> = b"%PDF-1.5\n".to_vec();
        let mut offsets = vec![0usize; bodies.len() + 1];
        for (idx, body) in bodies.iter().enumerate() {
            let num = idx + 1;
            offsets[num] = pdf.len();
            pdf.extend_from_slice(format!("{num} 0 obj\n").as_bytes());
            pdf.extend_from_slice(body);
            pdf.extend_from_slice(b"\nendobj\n");
        }
        let xref = pdf.len();
        let size = bodies.len() + 1;
        pdf.extend_from_slice(format!("xref\n0 {size}\n").as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for off in offsets.iter().skip(1) {
            pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
        );
        PdfDocument::open(pdf).unwrap()
    }

    fn dict(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    fn stream(d: &str, data: &[u8]) -> Vec<u8> {
        let mut v = format!("<< {d} /Length {} >>\nstream\n", data.len()).into_bytes();
        v.extend_from_slice(data);
        v.extend_from_slice(b"\nendstream");
        v
    }

    fn r(num: i64) -> Object {
        Object::new_indirect(num, 0)
    }

    /// Load object 3 of a one-object doc and evaluate it once.
    fn eval1(body: Vec<u8>, n_in: usize, n_out: usize, input: &[f32]) -> Vec<f32> {
        let doc = build(&[body]);
        let f = PdfFunction::load(&doc, &r(3), n_in, n_out).unwrap();
        let mut out = vec![f32::NAN; n_out];
        f.eval(input, &mut out);
        out
    }

    fn ps(domain: &str, range: &str, code: &str) -> Vec<u8> {
        stream(&format!("/FunctionType 4 /Domain [{domain}] /Range [{range}]"), code.as_bytes())
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5 * (1.0 + b.abs())
    }

    // ---- shared plumbing -------------------------------------------------

    #[test]
    fn op_table_is_sorted_like_mupdfs() {
        // MuPDF binary-searches ps_op_names; our exact lookup is only
        // equivalent if the table is strictly sorted.
        for w in PS_OP_NAMES.windows(2) {
            assert!(w[0].0 < w[1].0, "{} !< {}", w[0].0, w[1].0);
        }
    }

    #[test]
    fn fz_clamp_passes_nan_through() {
        assert!(fz_clamp(f32::NAN, 0.0, 1.0).is_nan());
        assert_eq!(fz_clamp(-1.0, 0.0, 1.0), 0.0);
        assert_eq!(fz_clamp(2.0, 0.0, 1.0), 1.0);
    }

    // ---- type 2 ------------------------------------------------------------

    #[test]
    fn exponential_n1_defaults() {
        // No C0/C1: c0 = 0, c1 = 1 -> out = x^1.
        let out = eval1(dict("<< /FunctionType 2 /Domain [0 1] /N 1 >>"), 1, 1, &[0.25]);
        assert_eq!(out, vec![0.25]);
    }

    #[test]
    fn exponential_n2_with_c0_c1() {
        // x = 0.5, N = 2 -> tmp = 0.25.
        // out0 = 0.2 + 0.25*(1.0-0.2) = 0.4; out1 = 0.4 + 0.25*(0.0-0.4) = 0.3.
        let body = dict("<< /FunctionType 2 /Domain [0 1] /N 2 /C0 [0.2 0.4] /C1 [1.0 0.0] >>");
        let doc = build(&[body]);
        let f = PdfFunction::load(&doc, &r(3), 1, 2).unwrap();
        assert_eq!((f.n_in(), f.n_out()), (1, 2));
        let mut out = [0.0; 2];
        f.eval(&[0.5], &mut out);
        assert!(close(out[0], 0.4) && close(out[1], 0.3), "{out:?}");
        // Domain clamp: x = 2 -> 1 -> out = C1.
        f.eval(&[2.0], &mut out);
        assert_eq!(out, [1.0, 0.0]);
        // Wider caller output: zero-filled past n (fz_eval_function).
        let mut out3 = [9.0; 3];
        f.eval(&[1.0], &mut out3);
        assert_eq!(out3, [1.0, 0.0, 0.0]);
        // Narrower caller output: only the first.
        let mut out1 = [9.0; 1];
        f.eval(&[1.0], &mut out1);
        assert_eq!(out1, [1.0]);
        // Empty input is padded with 0 -> out = C0.
        f.eval(&[], &mut out);
        assert!(close(out[0], 0.2) && close(out[1], 0.4));
    }

    #[test]
    fn exponential_range_and_violations() {
        // Range clamps (has_range): 0.9 -> 0.5.
        let out = eval1(dict("<< /FunctionType 2 /Domain [0 1] /Range [0 0.5] /N 1 >>"), 1, 1, &[0.9]);
        assert_eq!(out, vec![0.5]);
        // Non-integer N with negative x -> all zero.
        let out = eval1(dict("<< /FunctionType 2 /Domain [-1 1] /N 0.5 /C0 [0.7] >>"), 1, 1, &[-0.5]);
        assert_eq!(out, vec![0.0]);
        // Negative N with x == 0 -> all zero.
        let out = eval1(dict("<< /FunctionType 2 /Domain [0 1] /N -1 /C0 [0.7] >>"), 1, 1, &[0.0]);
        assert_eq!(out, vec![0.0]);
    }

    #[test]
    fn exponential_missing_domain_clamps_to_zero() {
        // No Domain: m = 1, domain [0 0] -> x = 0 -> out = C0.
        let out = eval1(dict("<< /FunctionType 2 /N 1 /C0 [0.3] /C1 [0.9] >>"), 1, 1, &[0.8]);
        assert!(close(out[0], 0.3), "{out:?}");
    }

    #[test]
    fn unknown_type_is_an_error() {
        let doc = build(&[dict("<< /FunctionType 7 /Domain [0 1] >>")]);
        assert!(PdfFunction::load(&doc, &r(3), 1, 1).is_err());
        // A non-dict reads FunctionType 0 -> sample path -> "too few sizes".
        let doc = build(&[dict("42")]);
        assert!(PdfFunction::load(&doc, &r(3), 1, 1).is_err());
    }

    // ---- type 3 ------------------------------------------------------------

    fn stitch_doc(extra: &str) -> PdfDocument {
        build(&[
            dict(&format!(
                "<< /FunctionType 3 /Domain [0 1] /Functions [4 0 R 5 0 R] /Bounds [0.5] {extra} >>"
            )),
            dict("<< /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [1] /N 1 >>"),
            dict("<< /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >>"),
        ])
    }

    #[test]
    fn stitching_bounds_and_encode() {
        let doc = stitch_doc("/Encode [0 1 0 1]");
        let f = PdfFunction::load(&doc, &r(3), 1, 1).unwrap();
        let ev = |x: f32| {
            let mut o = [0.0];
            f.eval(&[x], &mut o);
            o[0]
        };
        // 0.25 < 0.5 -> f_a over [0, 0.5] -> encoded 0.5 -> 0.5.
        assert_eq!(ev(0.25), 0.5);
        // Exactly on the bound: `in < bounds[0]` is false -> f_b over
        // [0.5, 1] -> encoded 0 -> 1 - 0 = 1.
        assert_eq!(ev(0.5), 1.0);
        // 0.75 -> f_b(0.5) = 0.5.
        assert_eq!(ev(0.75), 0.5);
        // Domain clamping: -3 -> 0 -> f_a(0) = 0; 5 -> 1 -> f_b(1) = 0.
        assert_eq!(ev(-3.0), 0.0);
        assert_eq!(ev(5.0), 0.0);
    }

    #[test]
    fn stitching_without_encode_maps_to_zero() {
        // Encode defaults to [0 0] pairs, lerp returns ymin = 0 -> f_b(0) = 1.
        let doc = stitch_doc("");
        let f = PdfFunction::load(&doc, &r(3), 1, 1).unwrap();
        let mut o = [0.0];
        f.eval(&[0.75], &mut o);
        assert_eq!(o, [1.0]);
        f.eval(&[0.25], &mut o);
        assert_eq!(o, [0.0]);
    }

    #[test]
    fn stitching_hostile() {
        // Self-reference: recursive function.
        let doc = build(&[dict("<< /FunctionType 3 /Domain [0 1] /Functions [3 0 R] /Bounds [] >>")]);
        let e = PdfFunction::load(&doc, &r(3), 1, 1).unwrap_err();
        assert!(e.message().contains("recursive"), "{e:?}");
        // Two-object loop.
        let doc = build(&[
            dict("<< /FunctionType 3 /Domain [0 1] /Functions [4 0 R] /Bounds [] >>"),
            dict("<< /FunctionType 3 /Domain [0 1] /Functions [3 0 R] /Bounds [] >>"),
        ]);
        assert!(PdfFunction::load(&doc, &r(3), 1, 1).is_err());
        // Missing Bounds, too few bounds, decreasing bounds, no functions.
        let sub = dict("<< /FunctionType 2 /Domain [0 1] /N 1 >>");
        for bad in [
            "/Functions [4 0 R 4 0 R]",
            "/Functions [4 0 R 4 0 R 4 0 R] /Bounds [0.5]",
            "/Functions [4 0 R 4 0 R 4 0 R] /Bounds [0.6 0.4]",
            "/Functions [] /Bounds []",
            "/Bounds []",
        ] {
            let doc = build(&[dict(&format!("<< /FunctionType 3 /Domain [0 1] {bad} >>")), sub.clone()]);
            assert!(PdfFunction::load(&doc, &r(3), 1, 1).is_err(), "{bad}");
        }
    }

    #[test]
    fn stitching_shared_dag_stays_linear() {
        // Ten levels, each stitching 2 copies of the next: 2^10 loads without
        // the store cache, 10 with it. Must load, and quickly.
        let mut objs = Vec::new();
        for lvl in 0..10 {
            let next = 4 + lvl;
            objs.push(dict(&format!(
                "<< /FunctionType 3 /Domain [0 1] /Functions [{next} 0 R {next} 0 R] /Bounds [0.5] /Encode [0 1 0 1] >>"
            )));
        }
        objs.push(dict("<< /FunctionType 2 /Domain [0 1] /N 1 >>"));
        let doc = build(&objs);
        let f = PdfFunction::load(&doc, &r(3), 1, 1).unwrap();
        let mut o = [0.0];
        f.eval(&[0.0], &mut o);
        assert_eq!(o, [0.0]);
    }

    // ---- type 0 ------------------------------------------------------------

    #[test]
    fn sampled_1d_8bit() {
        // Size [3], samples 0, 128, 255 (/255). x = 0.25 -> encode [0 2] ->
        // 0.5 -> e0 0, e1 1, frac 0.5 -> 0.5 * 128/255; decode = Range [0 1].
        let body = stream("/FunctionType 0 /Domain [0 1] /Range [0 1] /Size [3] /BitsPerSample 8", &[0, 128, 255]);
        let out = eval1(body.clone(), 1, 1, &[0.25]);
        let a = 0.0f32;
        let b = 128.0f32 / 255.0;
        assert_eq!(out, vec![a + (b - a) * 0.5]);
        // Top end hits the last sample exactly.
        assert_eq!(eval1(body, 1, 1, &[1.0]), vec![1.0]);
        // Encode [2 0] reverses the table: x = 0 -> sample[2] = 1.
        let body = stream(
            "/FunctionType 0 /Domain [0 1] /Range [0 1] /Size [3] /BitsPerSample 8 /Encode [2 0]",
            &[0, 128, 255],
        );
        assert_eq!(eval1(body, 1, 1, &[0.0]), vec![1.0]);
    }

    #[test]
    fn sampled_1d_16bit_decode() {
        // 0x0000, 0xFFFF -> 0, 1; x = 0.5 -> 0.5 -> Decode [10 20] -> 15.
        let body = stream(
            "/FunctionType 0 /Domain [0 1] /Range [0 100] /Size [2] /BitsPerSample 16 /Decode [10 20]",
            &[0, 0, 0xff, 0xff],
        );
        assert_eq!(eval1(body, 1, 1, &[0.5]), vec![15.0]);
    }

    #[test]
    fn sampled_without_range_is_all_zero() {
        // MuPDF clamps type 0 output to the (all-zero) range regardless.
        let body = stream("/FunctionType 0 /Domain [0 1] /Size [2] /BitsPerSample 8", &[0, 255]);
        assert_eq!(eval1(body, 1, 1, &[1.0]), vec![0.0]);
    }

    #[test]
    fn sampled_2d_bilinear() {
        // Size [2 2], n = 1: index = e0*1 + e1*2.
        // a(0,0)=0, b(1,0)=51/255=0.2, c(0,1)=102/255=0.4, d(1,1)=1.
        let body = stream(
            "/FunctionType 0 /Domain [0 1 0 1] /Range [0 1] /Size [2 2] /BitsPerSample 8",
            &[0, 51, 102, 255],
        );
        let doc = build(&[body]);
        let f = PdfFunction::load(&doc, &r(3), 2, 1).unwrap();
        assert_eq!((f.n_in(), f.n_out()), (2, 1));
        let ev = |x: f32, y: f32| {
            let mut o = [0.0];
            f.eval(&[x, y], &mut o);
            o[0]
        };
        // (0.5, 0.5): ab = 0.1, cd = 0.7, abcd = 0.4.
        let (a, b, c, d) = (0.0f32, 51.0f32 / 255.0, 102.0f32 / 255.0, 1.0f32);
        let ab = a + (b - a) * 0.5;
        let cd = c + (d - c) * 0.5;
        assert_eq!(ev(0.5, 0.5), ab + (cd - ab) * 0.5);
        assert!(close(ev(0.5, 0.5), 0.4));
        assert_eq!(ev(1.0, 0.0), b);
        assert_eq!(ev(0.0, 1.0), c);
        // A short input pads y = 0.
        let mut o = [0.0];
        f.eval(&[1.0], &mut o);
        assert_eq!(o[0], b);
    }

    #[test]
    fn sampled_3d_generic_path() {
        // Size [2 2 2], sample(x,y,z) = 30*(x + 2y + 4z)/255; at the centre
        // the trilinear value is the mean = 30*3.5/255.
        let data: Vec<u8> = (0..8).map(|i| 30 * i as u8).collect();
        let body = stream(
            "/FunctionType 0 /Domain [0 1 0 1 0 1] /Range [0 1] /Size [2 2 2] /BitsPerSample 8",
            &data,
        );
        let out = eval1(body.clone(), 3, 1, &[0.5, 0.5, 0.5]);
        assert!(close(out[0], 105.0 / 255.0), "{out:?}");
        let out = eval1(body, 3, 1, &[1.0, 0.0, 1.0]);
        assert!(close(out[0], 150.0 / 255.0), "{out:?}");
    }

    #[test]
    fn sampled_small_bit_depths() {
        // 4-bit: 0x0F -> samples 0, 15/15 = 1.
        let body = stream("/FunctionType 0 /Domain [0 1] /Range [0 1] /Size [2] /BitsPerSample 4", &[0x0f]);
        assert_eq!(eval1(body, 1, 1, &[1.0]), vec![1.0]);
        // 1-bit: 0b0100_0000 -> 0, 1.
        let body = stream("/FunctionType 0 /Domain [0 1] /Range [0 1] /Size [2] /BitsPerSample 1", &[0x40]);
        assert_eq!(eval1(body, 1, 1, &[1.0]), vec![1.0]);
        // 12-bit: 0xFFF, 0x000 in three bytes.
        let body = stream("/FunctionType 0 /Domain [0 1] /Range [0 1] /Size [2] /BitsPerSample 12", &[0xff, 0xf0, 0x00]);
        assert_eq!(eval1(body, 1, 1, &[0.0]), vec![1.0]);
    }

    #[test]
    fn sampled_hostile() {
        let load = |d: &str, data: &[u8]| {
            let doc = build(&[stream(d, data)]);
            PdfFunction::load(&doc, &r(3), 1, 1)
        };
        // Missing Size.
        assert!(load("/FunctionType 0 /Domain [0 1] /Range [0 1] /BitsPerSample 8", &[0]).is_err());
        // Truncated stream.
        assert!(load("/FunctionType 0 /Domain [0 1] /Range [0 1] /Size [4] /BitsPerSample 8", &[0, 1]).is_err());
        // 16-bit with an odd byte count: premature end.
        assert!(load("/FunctionType 0 /Domain [0 1] /Range [0 1] /Size [2] /BitsPerSample 16", &[0, 0, 1]).is_err());
        // Unsupported depth.
        assert!(load("/FunctionType 0 /Domain [0 1] /Range [0 1] /Size [2] /BitsPerSample 3", &[0, 0]).is_err());
        // Huge Size: over MAX_SAMPLE_FUNCTION_SIZE.
        let e = load("/FunctionType 0 /Domain [0 1 0 1] /Range [0 1] /Size [20000 20000] /BitsPerSample 8", &[0])
            .unwrap_err();
        assert!(e.message().contains("too large"), "{e:?}");
        // Huge-but-legal Size with a tiny stream: truncated, not a 400 MB alloc.
        assert!(load("/FunctionType 0 /Domain [0 1] /Range [0 1] /Size [100000000] /BitsPerSample 8", &[0]).is_err());
        // Zero Size is coerced to 1 (a warning in MuPDF), so one sample loads.
        let doc = build(&[stream("/FunctionType 0 /Domain [0 1] /Range [0 1] /Size [0] /BitsPerSample 8", &[255])]);
        let f = PdfFunction::load(&doc, &r(3), 1, 1).unwrap();
        let mut o = [0.0];
        f.eval(&[0.7], &mut o);
        assert_eq!(o, [1.0]);
        // A direct dict has no stream: fails to load.
        let obj = Object::Dict(vec![
            (b"FunctionType".to_vec(), Object::new_int(0)),
            (b"Size".to_vec(), Object::Array(vec![Object::new_int(1)])),
            (b"BitsPerSample".to_vec(), Object::new_int(8)),
        ]);
        assert!(PdfFunction::load(&doc, &obj, 1, 1).is_err());
    }

    // ---- type 4 ------------------------------------------------------------

    fn ps_eval(domain: &str, range: &str, code: &str, input: &[f32], n_out: usize) -> Vec<f32> {
        eval1(ps(domain, range, code), input.len(), n_out, input)
    }

    #[test]
    fn ps_arithmetic() {
        assert_eq!(ps_eval("0 1", "0 10", "{ 2 mul 1 add }", &[0.5], 1), vec![2.0]);
        // Output clamped to Range.
        assert_eq!(ps_eval("0 1", "0 1.5", "{ 2 mul 1 add }", &[0.5], 1), vec![1.5]);
        // Input clamped to Domain: 7 -> 1 -> 3.
        assert_eq!(ps_eval("0 1", "0 10", "{ 2 mul 1 add }", &[7.0], 1), vec![3.0]);
        // atan 1 1 = 45 degrees; 1 -1 = 135.
        let v = ps_eval("0 1", "0 360", "{ pop 1 1 atan }", &[0.0], 1);
        assert!(close(v[0], 45.0), "{v:?}");
        let v = ps_eval("0 1", "0 360", "{ pop -1 -1 atan }", &[0.0], 1);
        assert!(close(v[0], 225.0), "{v:?}");
        // sqrt, exp, round, truncate.
        assert_eq!(ps_eval("0 1", "-10 10", "{ pop 9 sqrt }", &[0.0], 1), vec![3.0]);
        assert_eq!(ps_eval("0 1", "-10 10", "{ pop 2 3 exp }", &[0.0], 1), vec![8.0]);
        assert_eq!(ps_eval("0 1", "-10 10", "{ pop -2.5 round }", &[0.0], 1), vec![-3.0]);
        assert_eq!(ps_eval("0 1", "-10 10", "{ pop -2.7 truncate }", &[0.0], 1), vec![-2.0]);
        // Real divide by zero: sign-of-quotient FLT_MAX, then Range clamp.
        assert_eq!(ps_eval("0 1", "-10 10", "{ pop -1 0 div }", &[0.0], 1), vec![-10.0]);
    }

    #[test]
    fn ps_int_typing() {
        assert_eq!(ps_eval("0 1", "-10 10", "{ pop 7 2 idiv }", &[0.0], 1), vec![3.0]);
        assert_eq!(ps_eval("0 1", "-10 10", "{ pop 7 2 div }", &[0.0], 1), vec![3.5]);
        assert_eq!(ps_eval("0 1", "-10 10", "{ pop 7 2 mod }", &[0.0], 1), vec![1.0]);
        // Int divide by zero -> INT_MAX (as f32 2^31).
        assert_eq!(ps_eval("0 1", "0 3000000000.0", "{ pop 1 0 idiv }", &[0.0], 1), vec![2147483648.0]);
        // Bitwise not of an int: ~5 = -6.
        assert_eq!(ps_eval("0 1", "-10 10", "{ pop 5 not }", &[0.0], 1), vec![-6.0]);
        // bitshift: 1 << 4 = 16; -16 >> 2 is a LOGICAL shift = 0x3FFFFFFC.
        assert_eq!(ps_eval("0 1", "0 100", "{ pop 1 4 bitshift }", &[0.0], 1), vec![16.0]);
        assert_eq!(
            ps_eval("0 1", "0 3000000000.0", "{ pop -16 -2 bitshift }", &[0.0], 1),
            vec![0x3fff_fffc as f32]
        );
        // cvi truncates a real (C float->int), 2.9 -> 2.
        assert_eq!(ps_eval("0 1", "-10 10", "{ pop 2.9 cvi }", &[0.0], 1), vec![2.0]);
        // Int add stays int; int + real becomes real.
        assert_eq!(ps_eval("0 1", "-10 10", "{ pop 2 3 add 0.5 add }", &[0.0], 1), vec![5.5]);
    }

    #[test]
    fn ps_comparisons_and_branches() {
        let code = "{ 0.5 gt { 1 } { 0 } ifelse }";
        assert_eq!(ps_eval("0 1", "0 10", code, &[0.7], 1), vec![1.0]);
        assert_eq!(ps_eval("0 1", "0 10", code, &[0.3], 1), vec![0.0]);
        let code = "{ dup 0.5 lt { pop 0.25 } if }";
        assert_eq!(ps_eval("0 1", "0 10", code, &[0.2], 1), vec![0.25]);
        assert_eq!(ps_eval("0 1", "0 10", code, &[0.8], 1), vec![0.8]);
        // Bool logic from comparisons.
        let code = "{ pop 1 2 lt 3 4 lt and { 5 } { 6 } ifelse }";
        assert_eq!(ps_eval("0 1", "0 10", code, &[0.0], 1), vec![5.0]);
        let code = "{ pop 1 2 eq not { 5 } { 6 } ifelse }";
        assert_eq!(ps_eval("0 1", "0 10", code, &[0.0], 1), vec![5.0]);
        let code = "{ pop 1 2 eq 1 1 eq xor 1 2 gt or { 5 } { 6 } ifelse }";
        assert_eq!(ps_eval("0 1", "0 10", code, &[0.0], 1), vec![5.0]);
        // Nested blocks.
        let code = "{ dup 0.5 gt { 0.75 gt { 3 } { 2 } ifelse } { pop 1 } ifelse }";
        assert_eq!(ps_eval("0 1", "0 10", code, &[0.9], 1), vec![3.0]);
        assert_eq!(ps_eval("0 1", "0 10", code, &[0.6], 1), vec![2.0]);
        assert_eq!(ps_eval("0 1", "0 10", code, &[0.1], 1), vec![1.0]);
    }

    #[test]
    fn ps_bool_literal_stops_execution() {
        // MuPDF quirk: `true` compiles to a PS_BOOL code object, which ps_run
        // treats as foreign and returns -- so 4 is never pushed.
        assert_eq!(ps_eval("0 1", "0 10", "{ pop 3 true 4 }", &[0.0], 1), vec![3.0]);
    }

    #[test]
    fn ps_stack_ops() {
        let d = "0 10 0 10 0 10";
        let r3 = "0 10 0 10 0 10";
        let i = [1.0, 2.0, 3.0];
        assert_eq!(ps_eval(d, r3, "{ exch }", &i, 3), vec![1.0, 3.0, 2.0]);
        assert_eq!(ps_eval(d, r3, "{ 3 1 roll }", &i, 3), vec![3.0, 1.0, 2.0]);
        assert_eq!(ps_eval(d, r3, "{ 3 -1 roll }", &i, 3), vec![2.0, 3.0, 1.0]);
        assert_eq!(ps_eval(d, r3, "{ pop pop dup dup }", &i, 3), vec![1.0, 1.0, 1.0]);
        assert_eq!(ps_eval(d, "0 10 0 10 0 10 0 10", "{ 2 index }", &i, 4), vec![1.0, 2.0, 3.0, 1.0]);
        assert_eq!(
            ps_eval(d, "0 10 0 10 0 10 0 10 0 10", "{ 2 copy }", &i, 5),
            vec![1.0, 2.0, 3.0, 2.0, 3.0]
        );
        // copy with too few items: pop_int consumes the 5, copy is a no-op,
        // leaving [1]; the two missing outputs underflow to 0.
        assert_eq!(ps_eval(d, r3, "{ pop pop 5 copy }", &i, 3), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn ps_hostile_stack() {
        // Overflow: pushes past 99 slots are dropped, never panic.
        let code = format!("{{ {} }}", "1 ".repeat(150));
        assert_eq!(ps_eval("0 1", "0 10", &code, &[0.5], 1), vec![1.0]);
        let code = format!("{{ {} }}", "dup ".repeat(200));
        assert_eq!(ps_eval("0 1", "0 10", &code, &[0.5], 1), vec![0.5]);
        // Underflow everywhere.
        let code = "{ pop pop pop add mul exch roll index copy 3 -7 roll -1 index }";
        let v = ps_eval("0 1", "0 10", code, &[0.5], 1);
        assert!(v[0].is_finite());
        // -1 index copies the stale slot just above the top: after `pop` of
        // the input, `-1` is pushed into slot 0 and popped again by `index`,
        // so slot 0 still holds Int(-1) and that is what gets duplicated.
        assert_eq!(ps_eval("0 1", "-10 10", "{ pop -1 index }", &[0.5], 1), vec![-1.0]);
        // (So `-1 index` always re-pushes the -1 it just consumed.)
        assert_eq!(ps_eval("0 1 0 1", "-10 10 -10 10", "{ -1 index }", &[0.5, 0.5], 2), vec![0.5, -1.0]);
        // Huge roll/index/copy counts.
        let code = "{ 2147483647 1 roll 2147483647 index 2147483647 copy -2147483648 -2147483648 roll }";
        assert_eq!(ps_eval("0 1", "0 10", code, &[0.5], 1), vec![0.5]);
    }

    #[test]
    fn ps_parse_errors() {
        let load = |code: &str| {
            let doc = build(&[ps("0 1", "0 1", code)]);
            PdfFunction::load(&doc, &r(3), 1, 1)
        };
        assert!(load("{ 1 foo }").is_err());
        assert!(load("{ 1 add").is_err());
        assert!(load("1 add").is_err());
        assert!(load("{ 1 if }").is_err());
        assert!(load("{ true { 1 } { 2 } if }").is_err());
        assert!(load("{ true { 1 } ifelse }").is_err());
        assert!(load("{ true { 1 } bogus }").is_err());
        assert!(load("{ /name }").is_err());
        // Nesting: 101 levels below the top one is too deep.
        let deep = format!("{{ {}", "{ ".repeat(101));
        let e = load(&deep).unwrap_err();
        assert!(e.message().contains("nesting"), "{e:?}");
        // 100 levels is fine.
        let ok = format!("{{ {}{} }}", "true { ".repeat(100), "} if ".repeat(100));
        assert!(load(&ok).is_ok());
    }
}
