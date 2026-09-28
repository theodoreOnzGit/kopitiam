//! Ported from MuPDF `source/pdf/pdf-shade.c` (`pdf_load_shading` and the
//! per-type loaders), `source/fitz/shade.c` (`fz_process_shade` for shading
//! types 1-7, the tensor-patch subdivision, `fz_bound_shade`) and
//! `source/fitz/draw-mesh.c` (`fz_paint_shade`, the Gouraud triangle scan
//! converter `fz_paint_triangle` / `paint_scan`) (commit 19f1284, AGPL-3.0,
//! © Artifex Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the algorithms and numeric behaviour follow MuPDF; the
//! code is re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF &
//! document-extraction references").
//!
//! # Why this exists (0.4.2)
//!
//! Before 0.4.2 the interpreter parsed `sh` and shading patterns and painted
//! nothing: every gradient -- chart bars, slide backdrops, logo fills -- came out
//! as whatever was underneath (usually white paper). The feature corpus
//! (scripts/mupdf-feature-corpus.py) measured 50-64 % of the page wrong on
//! the shading files.
//!
//! # How MuPDF draws a shading (and so how this does)
//!
//! 1. **Load** (`pdf_load_shading_dict`): the colour space, `/Background`,
//!    `/BBox`, the function(s). Types 2 and 3 (and 4-7 when they have a
//!    function) SAMPLE the function(s) into a 256-entry table over the domain
//!    (`make_sampled_shade_function`); type 1 samples a 257 x 257 grid.
//! 2. **Mesh** (`fz_process_shade`): every type becomes triangles in device
//!    space. Axial shadings are one quad along the axis (+ extension quads),
//!    radial ones are annuli of `count` segments, meshes are read from the
//!    bit-packed stream, patches are subdivided 3 levels deep.
//! 3. **Paint** (`fz_paint_shade`): with a sampled function, triangles carry
//!    `t * 255` and are Gouraud-filled into a gray+alpha buffer, which is then
//!    mapped through a 256-entry colour look-up table; without one they carry
//!    the colour directly. The result is composited onto the page.
//!
//! Colour conversion goes through [`ColorSpace::to_rgb`] (the port has no CMS;
//! see docs/mupdf-port-coverage.md). A shading carried by a pattern's
//! `/ExtGState` alpha is ignored, exactly as MuPDF warns and ignores it.

use super::error::{Error, Result};
use super::function::PdfFunction;
use super::geometry::{IRect, Matrix, Point, Rect};
use super::object::Object;
use super::resources::{ColorSpace, load_colorspace};
use super::xref::PdfDocument;

/// `FUNSEGS` (pdf-shade.c): the function-based shading grid size.
const FUNSEGS: usize = 256;
/// `HUGENUM` (shade.c): how far linear/radial shadings extend.
const HUGENUM: f32 = 32000.0;
/// `SUBDIV` (shade.c): patch subdivision depth.
const SUBDIV: u32 = 3;
/// MuPDF's `FZ_PI` (truncated, per AID-0051).
const FZ_PI: f32 = 3.141_592_65;

/// The geometry half of a loaded shading (`fz_shade.u`).
#[derive(Clone, Debug)]
enum Kind {
    /// Type 1: sampled `(xdivs+1) x (ydivs+1)` grid of colours.
    Function {
        domain: [f32; 4],
        matrix: Matrix,
        xdivs: usize,
        ydivs: usize,
        vals: Vec<f32>,
    },
    /// Type 2 (`radial == false`) / type 3.
    LinearOrRadial {
        radial: bool,
        coords: [[f32; 3]; 2],
        extend: [bool; 2],
    },
    /// Types 4-7: the stream data plus its decode parameters.
    Mesh { typ: u8, m: MeshParams, data: Vec<u8> },
}

/// `fz_shade.u.m` (the mesh parameters, after pdf_load_mesh_params' fix-ups).
#[derive(Clone, Debug)]
struct MeshParams {
    vprow: i32,
    bpflag: i32,
    bpcoord: i32,
    bpcomp: i32,
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
    c0: Vec<f32>,
    c1: Vec<f32>,
}

/// A loaded shading (`fz_shade`).
#[derive(Clone, Debug)]
pub struct Shade {
    kind: Kind,
    colorspace: ColorSpace,
    /// `fz_colorspace_n(shade->colorspace)`.
    n: usize,
    /// The sampled function table, `256 * stride` floats, when there is one
    /// (`shade->function`); `stride = components + 1` (the last is alpha 1).
    function: Option<Vec<f32>>,
    stride: usize,
    background: Option<Vec<f32>>,
    bbox: Rect,
    matrix: Matrix,
}

impl Shade {
    // MuPDF: pdf_load_shading (pdf-shade.c:500)
    /// Load a shading from a resolved `/Shading` dictionary or stream, or from
    /// a type 2 pattern dictionary (`/PatternType 2`, whose `/Matrix` becomes
    /// the shading matrix). `obj_ref` is the indirect reference when there is
    /// one (mesh shadings are streams and need it for their data).
    pub(crate) fn load(doc: &PdfDocument, obj: &Object, obj_ref: &Object) -> Result<Shade> {
        if obj.dict_gets("PatternType").is_some() {
            let mat = matrix_of(doc, obj, "Matrix");
            let sh_ref = obj
                .dict_gets("Shading")
                .cloned()
                .ok_or_else(|| Error::syntax("missing shading dictionary"))?;
            let sh = doc.resolve(&sh_ref)?;
            return load_shading_dict(doc, &sh, &sh_ref, mat);
        }
        load_shading_dict(doc, obj, obj_ref, Matrix::IDENTITY)
    }

    // MuPDF: fz_bound_shade (shade.c:1125)
    /// Device-space bounds of the shading under `ctm` (possibly infinite).
    pub fn bound(&self, ctm: Matrix) -> Rect {
        let ctm = self.matrix.concat(ctm);
        match &self.kind {
            // "if (shade->type != FZ_LINEAR && shade->type != FZ_RADIAL)"
            Kind::LinearOrRadial { .. } => self.bbox.transform(ctm),
            _ => self.bound_mesh().intersect(self.bbox).transform(ctm),
        }
    }

    // MuPDF: fz_bound_mesh (shade.c:1073)
    fn bound_mesh(&self) -> Rect {
        match &self.kind {
            Kind::Function { domain, matrix, .. } => {
                Rect::new(domain[0], domain[2], domain[1], domain[3]).transform(*matrix)
            }
            Kind::LinearOrRadial { radial: false, .. } => Rect::INFINITE,
            Kind::LinearOrRadial { radial: true, coords, extend } => {
                let (r0, r1) = (coords[0][2], coords[1][2]);
                if extend[0] && r0 >= r1 {
                    return Rect::INFINITE;
                }
                if extend[1] && r0 <= r1 {
                    return Rect::INFINITE;
                }
                let (p0, p1) = (coords[0], coords[1]);
                // Note: MuPDF writes `bbox.y1 = p0.x + r0` (sic) -- kept.
                let mut b = Rect::new(p0[0] - r0, p0[1] - r0, p0[0] + r0, p0[0] + r0);
                if b.x0 > p1[0] - r1 {
                    b.x0 = p1[0] - r1;
                }
                if b.x1 < p1[0] + r1 {
                    b.x1 = p1[0] + r1;
                }
                if b.y0 > p1[1] - r1 {
                    b.y0 = p1[1] - r1;
                }
                if b.y1 < p1[1] + r1 {
                    b.y1 = p1[1] + r1;
                }
                b
            }
            Kind::Mesh { m, .. } => Rect::new(m.x0.min(m.x1), m.y0.min(m.y1), m.x0.max(m.x1), m.y0.max(m.y1)),
        }
    }

    /// The `/Background` colour in DeviceRGB, if the shading has one.
    pub fn background_rgb(&self) -> Option<[f32; 3]> {
        self.background.as_ref().map(|b| self.colorspace.to_rgb(b, [0.0; 3]))
    }

    // MuPDF: fz_paint_shade (draw-mesh.c:248) -- into a fresh premultiplied
    // RGBA patch over `bbox` rather than straight into the page, so the caller
    // can composite it with the fill alpha and any clip mask.
    /// Paint the shading under device matrix `ctm` into an RGBA buffer
    /// covering `bbox` (row-major, premultiplied, 4 bytes per pixel).
    pub fn paint(&self, ctm: Matrix, bbox: IRect) -> Vec<u8> {
        let w = (bbox.x1 - bbox.x0).max(0) as usize;
        let h = (bbox.y1 - bbox.y0).max(0) as usize;
        let local = self.matrix.concat(ctm);
        if let Some(func) = &self.function {
            // Gray + alpha, cleared: "alpha = 1 here, because the shade might
            // not fill the bbox".
            let mut temp = Buf { x: bbox.x0, y: bbox.y0, w: w as i32, h: h as i32, n: 2, samples: vec![0; w * h * 2] };
            self.process(local, bbox, &mut temp);
            // The colour look-up table (non-DeviceN branch of draw-mesh.c).
            let cn = self.stride - 1;
            let mut clut = [[0u8; 4]; 256];
            for (i, entry) in clut.iter_mut().enumerate() {
                let c = &func[i * self.stride..i * self.stride + cn];
                let rgb = self.colorspace_for_function().to_rgb(c, [0.0; 3]);
                for k in 0..3 {
                    entry[k] = (rgb[k] * 255.0) as u8;
                }
                entry[3] = (func[i * self.stride + cn] * 255.0) as u8;
            }
            let mut out = vec![0u8; w * h * 4];
            for (px, o) in temp.samples.chunks(2).zip(out.chunks_mut(4)) {
                let v = px[0] as usize;
                let a = mul255(px[1] as i32, clut[v][3] as i32);
                for k in 0..3 {
                    o[k] = mul255(clut[v][k] as i32, a) as u8;
                }
                o[3] = a as u8;
            }
            out
        } else {
            let mut temp = Buf { x: bbox.x0, y: bbox.y0, w: w as i32, h: h as i32, n: 4, samples: vec![0; w * h * 4] };
            self.process(local, bbox, &mut temp);
            temp.samples
        }
    }

    /// The colour space the function table's outputs are in: the shading's
    /// own, or -- with one function per component -- still the shading's.
    fn colorspace_for_function(&self) -> &ColorSpace {
        &self.colorspace
    }

    // MuPDF: fz_process_shade (shade.c:983) with draw-mesh.c's
    // prepare_mesh_vertex + do_paint_tri as the callbacks.
    fn process(&self, ctm: Matrix, bbox: IRect, dest: &mut Buf) {
        let mut painter = Painter { shade: self, dest, bbox };
        match &self.kind {
            Kind::Function { .. } => painter.type1(ctm),
            Kind::LinearOrRadial { radial: false, .. } => {
                painter.type2(ctm, Rect::new(bbox.x0 as f32, bbox.y0 as f32, bbox.x1 as f32, bbox.y1 as f32))
            }
            Kind::LinearOrRadial { radial: true, .. } => painter.type3(ctm),
            Kind::Mesh { typ, .. } => match typ {
                4 => painter.type4(ctm),
                5 => painter.type5(ctm),
                6 | 7 => painter.type67(ctm, *typ),
                _ => {}
            },
        }
    }
}

// MuPDF: fz_mul255 (geometry.h:38)
fn mul255(a: i32, b: i32) -> i32 {
    let mut x = a * b + 128;
    x += x >> 8;
    x >> 8
}

fn matrix_of(doc: &PdfDocument, dict: &Object, key: &str) -> Matrix {
    let arr = doc.resolve_get(dict, key).unwrap_or(Object::Null);
    if arr.array_len() < 6 {
        return Matrix::IDENTITY;
    }
    let v = |i: usize| arr.array_get(i).and_then(|o| doc.resolve(o).ok()).map_or(0.0, |o| o.to_real() as f32);
    Matrix::new(v(0), v(1), v(2), v(3), v(4), v(5))
}

fn real_at(doc: &PdfDocument, arr: &Object, i: usize) -> f32 {
    arr.array_get(i).and_then(|o| doc.resolve(o).ok()).map_or(0.0, |o| o.to_real() as f32)
}

// MuPDF: pdf_load_shading_dict (pdf-shade.c:343)
fn load_shading_dict(doc: &PdfDocument, dict: &Object, dict_ref: &Object, transform: Matrix) -> Result<Shade> {
    let get = |k: &str| doc.resolve_get(dict, k).unwrap_or(Object::Null);
    let typ = get("ShadingType").to_int();
    let cs_obj = get("ColorSpace");
    if cs_obj.is_null() {
        return Err(Error::syntax("shading colorspace is missing"));
    }
    let colorspace = load_colorspace(doc, &cs_obj, 0);
    let n = colorspace.n();

    let bg = get("Background");
    let background = (!bg.is_null()).then(|| (0..n).map(|i| real_at(doc, &bg, i)).collect());
    let bb = get("BBox");
    let bbox = if bb.is_array() {
        let (a, b, c, d) = (real_at(doc, &bb, 0), real_at(doc, &bb, 1), real_at(doc, &bb, 2), real_at(doc, &bb, 3));
        Rect::new(a.min(c), b.min(d), a.max(c), b.max(d))
    } else {
        Rect::INFINITE
    };

    // The function(s): one n-out, or n one-out (FZ_MAX_COLORS = 32).
    let in_n = if typ == 1 { 2 } else { 1 };
    let fobj = get("Function");
    let mut funcs: Vec<PdfFunction> = Vec::new();
    let fref = dict.dict_gets("Function").cloned().unwrap_or(Object::Null);
    if fobj.is_dict() {
        funcs.push(PdfFunction::load(doc, &fref, in_n, n)?);
    } else if fobj.is_array() {
        let count = fobj.array_len();
        if count != 1 && count != n {
            return Err(Error::syntax("incorrect number of shading functions"));
        }
        if count > 32 {
            return Err(Error::syntax("too many shading functions"));
        }
        for i in 0..count {
            let f = fobj.array_get(i).cloned().unwrap_or(Object::Null);
            funcs.push(PdfFunction::load(doc, &f, in_n, 1)?);
        }
    } else if typ < 4 {
        return Err(Error::syntax("cannot load shading function"));
    }

    let mut shade = Shade {
        kind: Kind::Mesh { typ: 4, m: MeshParams::default_for(0), data: Vec::new() },
        colorspace,
        n,
        function: None,
        stride: 0,
        background,
        bbox,
        matrix: transform,
    };

    match typ {
        1 => shade.kind = load_function_based(doc, dict, &funcs, n)?,
        2 | 3 => {
            let coords_obj = get("Coords");
            let mut coords = [[0.0f32; 3]; 2];
            if typ == 2 {
                coords[0][0] = real_at(doc, &coords_obj, 0);
                coords[0][1] = real_at(doc, &coords_obj, 1);
                coords[1][0] = real_at(doc, &coords_obj, 2);
                coords[1][1] = real_at(doc, &coords_obj, 3);
            } else {
                for (i, c) in coords.iter_mut().enumerate() {
                    for (j, v) in c.iter_mut().enumerate() {
                        *v = real_at(doc, &coords_obj, i * 3 + j);
                    }
                }
            }
            let dom = get("Domain");
            let (d0, d1) = if dom.is_null() { (0.0, 1.0) } else { (real_at(doc, &dom, 0), real_at(doc, &dom, 1)) };
            let ext = get("Extend");
            let bool_at = |i: usize| ext.array_get(i).and_then(|o| doc.resolve(o).ok()).is_some_and(|o| o.to_bool());
            let extend = if ext.is_null() { [false, false] } else { [bool_at(0), bool_at(1)] };
            make_sampled_shade_function(&mut shade, &funcs, d0, d1);
            shade.kind = Kind::LinearOrRadial { radial: typ == 3, coords, extend };
        }
        4..=7 => {
            let m = load_mesh_params(doc, dict, typ as u8);
            if !funcs.is_empty() {
                let (c0, c1) = (m.c0[0], m.c1[0]);
                make_sampled_shade_function(&mut shade, &funcs, c0, c1);
            }
            // pdf_load_compressed_stream: the decoded stream data.
            let data = doc.open_stream(dict_ref).unwrap_or_default();
            shade.kind = Kind::Mesh { typ: typ as u8, m, data };
        }
        _ => return Err(Error::syntax(format!("unknown shading type: {typ}"))),
    }
    Ok(shade)
}

// MuPDF: pdf_load_function_based_shading (pdf-shade.c:77)
fn load_function_based(doc: &PdfDocument, dict: &Object, funcs: &[PdfFunction], n: usize) -> Result<Kind> {
    let dom = doc.resolve_get(dict, "Domain").unwrap_or(Object::Null);
    let (mut x0, mut y0, mut x1, mut y1) = (0.0f32, 0.0f32, 1.0f32, 1.0f32);
    if !dom.is_null() {
        x0 = real_at(doc, &dom, 0);
        x1 = real_at(doc, &dom, 1);
        y0 = real_at(doc, &dom, 2);
        y1 = real_at(doc, &dom, 3);
    }
    let (xdivs, ydivs) = (FUNSEGS, FUNSEGS);
    let matrix = matrix_of(doc, dict, "Matrix");
    let mut vals = Vec::with_capacity((xdivs + 1) * (ydivs + 1) * n);
    if funcs.len() != 1 && funcs.len() != n {
        return Err(Error::syntax("Expected 1 2in, n-out function, or n 2 in, 1-out functions"));
    }
    let mut out = vec![0.0f32; n.max(1)];
    for yy in 0..=ydivs {
        let fy = y0 + (y1 - y0) * yy as f32 / ydivs as f32;
        for xx in 0..=xdivs {
            let fx = x0 + (x1 - x0) * xx as f32 / xdivs as f32;
            if funcs.len() == 1 {
                funcs[0].eval(&[fx, fy], &mut out[..n]);
                vals.extend_from_slice(&out[..n]);
            } else {
                for f in funcs {
                    let mut one = [0.0f32];
                    f.eval(&[fx, fy], &mut one);
                    vals.push(one[0]);
                }
            }
        }
    }
    Ok(Kind::Function { domain: [x0, x1, y0, y1], matrix, xdivs, ydivs, vals })
}

// MuPDF: make_sampled_shade_function + pdf_sample_shade_function
// (pdf-shade.c:28-62): 256 samples of t over [t0, t1], each followed by an
// alpha of 1.
fn make_sampled_shade_function(shade: &mut Shade, funcs: &[PdfFunction], t0: f32, t1: f32) {
    let n = if funcs.len() == 1 { shade.n } else { funcs.len() };
    let stride = n + 1;
    let mut table = Vec::with_capacity(256 * stride);
    let mut out = vec![0.0f32; n.max(1)];
    for i in 0..256 {
        let t = t0 + (i as f32 / 255.0) * (t1 - t0);
        if funcs.len() == 1 {
            funcs[0].eval(&[t], &mut out[..n]);
            table.extend_from_slice(&out[..n]);
        } else {
            for f in funcs {
                let mut one = [0.0f32];
                f.eval(&[t], &mut one);
                table.push(one[0]);
            }
        }
        table.push(1.0);
    }
    shade.function = Some(table);
    shade.stride = stride;
}

impl MeshParams {
    fn default_for(_typ: u8) -> MeshParams {
        MeshParams {
            vprow: 0,
            bpflag: 0,
            bpcoord: 0,
            bpcomp: 0,
            x0: 0.0,
            x1: 1.0,
            y0: 0.0,
            y1: 1.0,
            c0: vec![0.0; 32],
            c1: vec![1.0; 32],
        }
    }
}

// MuPDF: pdf_load_mesh_params (pdf-shade.c:222)
fn load_mesh_params(doc: &PdfDocument, dict: &Object, typ: u8) -> MeshParams {
    let int = |k: &str| doc.resolve_get(dict, k).map(|o| o.to_int() as i32).unwrap_or(0);
    let mut m = MeshParams::default_for(typ);
    m.vprow = int("VerticesPerRow");
    m.bpflag = int("BitsPerFlag");
    m.bpcoord = int("BitsPerCoordinate");
    m.bpcomp = int("BitsPerComponent");
    let dec = doc.resolve_get(dict, "Decode").unwrap_or(Object::Null);
    if dec.array_len() >= 6 {
        let n = 32.min((dec.array_len() - 4) / 2);
        m.x0 = real_at(doc, &dec, 0);
        m.x1 = real_at(doc, &dec, 1);
        m.y0 = real_at(doc, &dec, 2);
        m.y1 = real_at(doc, &dec, 3);
        for i in 0..n {
            m.c0[i] = real_at(doc, &dec, 4 + i * 2);
            m.c1[i] = real_at(doc, &dec, 5 + i * 2);
        }
    }
    if m.vprow < 2 && typ == 5 {
        m.vprow = 2; // "Too few vertices per row"
    }
    if !matches!(m.bpflag, 2 | 4 | 8) && typ != 5 {
        m.bpflag = 8; // "Invalid number of bits per flag"
    }
    if !matches!(m.bpcoord, 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32) {
        m.bpcoord = 8;
    }
    if !matches!(m.bpcomp, 1 | 2 | 4 | 8 | 12 | 16) {
        m.bpcomp = 8;
    }
    m
}

// ---------------------------------------------------------------------------
// The mesh processor (shade.c) and the triangle painter (draw-mesh.c)
// ---------------------------------------------------------------------------

/// A plain pixel buffer with a device origin: `n` bytes per pixel, the last
/// being alpha (both the gray+alpha and the RGBA buffers have alpha).
struct Buf {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    n: usize,
    samples: Vec<u8>,
}

/// `fz_vertex`: a device point plus up to 3 prepared components (`t * 255`
/// for a sampled function, else R, G, B * 255).
#[derive(Clone, Copy, Debug)]
struct Vertex {
    p: Point,
    c: [f32; 3],
}

impl Default for Vertex {
    fn default() -> Vertex {
        Vertex { p: Point::new(0.0, 0.0), c: [0.0; 3] }
    }
}

struct Painter<'a, 'b> {
    shade: &'a Shade,
    dest: &'b mut Buf,
    bbox: IRect,
}

/// MSB-first bit reader over the mesh data (`fz_read_bits` +
/// `fz_is_eof_bits` on an in-memory stream).
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    bits: u32,
    avail: u32,
}

impl Bits<'_> {
    // MuPDF: fz_read_bits (stream.h:542); EOF bytes read as 0xff... no --
    // fz_read_byte returns EOF (-1) whose low bits are all ones; the data is
    // exhausted by then and is_eof stops the caller at the next record.
    fn read(&mut self, n: u32) -> u32 {
        let mut n = n;
        if n <= self.avail {
            self.avail -= n;
            return (self.bits >> self.avail) & ((1u64 << n) - 1) as u32;
        }
        let mut x: u64 = (self.bits & ((1u64 << self.avail) - 1) as u32) as u64;
        n -= self.avail;
        self.avail = 0;
        while n > 8 {
            x = (x << 8) | self.byte() as u64;
            n -= 8;
        }
        if n > 0 {
            self.bits = self.byte();
            self.avail = 8 - n;
            x = (x << n) | (self.bits >> self.avail) as u64;
        }
        x as u32
    }

    fn byte(&mut self) -> u32 {
        let b = self.data.get(self.pos).map_or(0xff, |b| *b as u32);
        self.pos += 1;
        b
    }

    // MuPDF: fz_is_eof_bits (stream.h:640)
    fn eof(&self) -> bool {
        self.pos >= self.data.len() && self.avail == 0
    }

    // MuPDF: read_sample (shade.c:349)
    fn sample(&mut self, bits: i32, min: f32, max: f32) -> f32 {
        let bitscale = 1.0 / (2.0f32.powi(bits) - 1.0);
        min + self.read(bits as u32) as f32 * (max - min) * bitscale
    }
}

impl Painter<'_, '_> {
    /// `painter->ncomp`: 1 with a sampled function, else the space's `n`.
    fn ncomp(&self) -> usize {
        if self.shade.stride > 0 { 1 } else { self.shade.n }
    }

    // MuPDF: prepare_mesh_vertex (draw-mesh.c:185)
    fn prepare(&self, v: &mut Vertex, input: &[f32]) {
        let shade = self.shade;
        if shade.stride > 0 {
            let mut f = input[0];
            if let Kind::Mesh { m, .. } = &shade.kind {
                f = (f - m.c0[0]) / (m.c1[0] - m.c0[0]);
            }
            v.c[0] = f * 255.0;
        } else {
            let rgb = shade.colorspace.to_rgb(input, [0.0; 3]);
            for k in 0..3 {
                v.c[k] = rgb[k] * 255.0;
            }
        }
    }

    fn vertex(&self, ctm: Matrix, x: f32, y: f32, c: &[f32]) -> Vertex {
        let mut v = Vertex { p: Point::new(x, y).transform(ctm), c: [0.0; 3] };
        self.prepare(&mut v, c);
        v
    }

    fn colored(&self, p: Point, c: &[f32]) -> Vertex {
        let mut v = Vertex { p, c: [0.0; 3] };
        self.prepare(&mut v, c);
        v
    }

    // MuPDF: paint_quad (shade.c:36) -- (v0, v1, v3) then (v3, v2, v1).
    fn quad(&mut self, v0: &Vertex, v1: &Vertex, v2: &Vertex, v3: &Vertex) {
        self.tri(v0, v1, v3);
        self.tri(v3, v2, v1);
    }

    fn tri(&mut self, a: &Vertex, b: &Vertex, c: &Vertex) {
        let n = if self.shade.stride > 0 { 1 } else { 3 };
        paint_triangle(self.dest, [a, b, c], n, self.bbox);
    }

    // MuPDF: fz_process_shade_type1 (shade.c:80)
    fn type1(&mut self, ctm: Matrix) {
        let Kind::Function { domain, matrix, xdivs, ydivs, vals } = &self.shade.kind else { return };
        let n = self.shade.n;
        let (x0, x1, y0, y1) = (domain[0], domain[1], domain[2], domain[3]);
        let ctm = matrix.concat(ctm);
        let (xdivs, ydivs) = (*xdivs, *ydivs);
        let mut p = 0usize;
        let mut y = y0;
        for yy in 0..ydivs {
            let yn = y0 + (y1 - y0) * (yy + 1) as f32 / ydivs as f32;
            let mut x = x0;
            let mut v0 = self.vertex(ctm, x, y, &vals[p..p + n]);
            p += n;
            let mut v1 = self.vertex(ctm, x, yn, &vals[p + xdivs * n..p + xdivs * n + n]);
            for xx in 0..xdivs {
                x = x0 + (x1 - x0) * (xx + 1) as f32 / xdivs as f32;
                let vn0 = self.vertex(ctm, x, y, &vals[p..p + n]);
                p += n;
                let vn1 = self.vertex(ctm, x, yn, &vals[p + xdivs * n..p + xdivs * n + n]);
                self.quad(&v0, &vn0, &vn1, &v1);
                v0 = vn0;
                v1 = vn1;
            }
            y = yn;
        }
    }

    // MuPDF: fz_process_shade_type2 (shade.c:123)
    fn type2(&mut self, ctm: Matrix, scissor: Rect) {
        let Kind::LinearOrRadial { coords, extend, .. } = self.shade.kind.clone() else { return };
        let mut p0 = Point::new(coords[0][0], coords[0][1]);
        let mut p1 = Point::new(coords[1][0], coords[1][1]);
        let dir = Point::new(p0.y - p1.y, p1.x - p0.x);
        p0 = p0.transform(ctm);
        p1 = p1.transform(ctm);
        let dir = dir.transform_vector(ctm);
        let theta = dir.y.atan2(dir.x);

        let mut r = if scissor.is_infinite() {
            HUGENUM
        } else {
            let mut x = p0.x - scissor.x0;
            let mut y = p0.y - scissor.y0;
            if x < scissor.x1 - p0.x {
                x = scissor.x1 - p0.x;
            }
            if x < p0.x - scissor.x1 {
                x = p0.x - scissor.x1;
            }
            if x < scissor.x1 - p1.x {
                x = scissor.x1 - p1.x;
            }
            if y < scissor.y1 - p0.y {
                y = scissor.y1 - p0.y;
            }
            if y < p0.y - scissor.y1 {
                y = p0.y - scissor.y1;
            }
            if y < scissor.y1 - p1.y {
                y = scissor.y1 - p1.y;
            }
            x + y
        };
        let on = |p: Point, r: f32, t: f32| Point::new(p.x + t.cos() * r, p.y + t.sin() * r);
        let v0p = on(p0, r, theta);
        let v1p = on(p1, r, theta);
        let v2p = Point::new(2.0 * p0.x - v0p.x, 2.0 * p0.y - v0p.y);
        let v3p = Point::new(2.0 * p1.x - v1p.x, 2.0 * p1.y - v1p.y);
        let (zero, one) = ([0.0f32], [1.0f32]);
        let v0 = self.colored(v0p, &zero);
        let v1 = self.colored(v1p, &one);
        let v2 = self.colored(v2p, &zero);
        let v3 = self.colored(v3p, &one);
        self.quad(&v0, &v2, &v3, &v1);

        if extend[0] || extend[1] {
            let mut d = (p1.x - p0.x).abs();
            let e = (p1.y - p0.y).abs();
            if d < e {
                d = e;
            }
            if d != 0.0 {
                r /= d;
            }
        }
        if extend[0] {
            let e0 = self.colored(Point::new(v0p.x - (p1.x - p0.x) * r, v0p.y - (p1.y - p0.y) * r), &zero);
            let e1 = self.colored(Point::new(v2p.x - (p1.x - p0.x) * r, v2p.y - (p1.y - p0.y) * r), &zero);
            self.quad(&e0, &v0, &v2, &e1);
        }
        if extend[1] {
            let e0 = self.colored(Point::new(v1p.x + (p1.x - p0.x) * r, v1p.y + (p1.y - p0.y) * r), &one);
            let e1 = self.colored(Point::new(v3p.x + (p1.x - p0.x) * r, v3p.y + (p1.y - p0.y) * r), &one);
            self.quad(&e0, &v1, &v3, &e1);
        }
    }

    // MuPDF: fz_paint_annulus (shade.c:223)
    #[allow(clippy::too_many_arguments)]
    fn annulus(&mut self, ctm: Matrix, p0: Point, r0: f32, c0: f32, p1: Point, r1: f32, c1: f32, count: i32) {
        let theta = (p1.y - p0.y).atan2(p1.x - p0.x);
        let step = FZ_PI / count as f32;
        let on = |p: Point, r: f32, t: f32| Point::new(p.x + t.cos() * r, p.y + t.sin() * r).transform(ctm);
        let mut a = 0.0f32;
        for i in 1..=count {
            let b = i as f32 * step;
            let t0 = self.colored(on(p0, r0, theta + a), &[c0]);
            let t1 = self.colored(on(p0, r0, theta + b), &[c0]);
            let t2 = self.colored(on(p1, r1, theta + a), &[c1]);
            let t3 = self.colored(on(p1, r1, theta + b), &[c1]);
            let b0 = self.colored(on(p0, r0, theta - a), &[c0]);
            let b1 = self.colored(on(p0, r0, theta - b), &[c0]);
            let b2 = self.colored(on(p1, r1, theta - a), &[c1]);
            let b3 = self.colored(on(p1, r1, theta - b), &[c1]);
            self.quad(&t0, &t2, &t3, &t1);
            self.quad(&b0, &b2, &b3, &b1);
            a = b;
        }
    }

    // MuPDF: fz_process_shade_type3 (shade.c:269)
    fn type3(&mut self, ctm: Matrix) {
        let Kind::LinearOrRadial { coords, extend, .. } = self.shade.kind.clone() else { return };
        let p0 = Point::new(coords[0][0], coords[0][1]);
        let r0 = coords[0][2];
        let p1 = Point::new(coords[1][0], coords[1][1]);
        let r1 = coords[1][2];
        let mut count = (4.0 * (ctm.expansion() * r0.max(r1)).sqrt()) as i32;
        count = count.clamp(3, 1024);
        if extend[0] {
            let rs = if r0 < r1 { r0 / (r0 - r1) } else { -HUGENUM };
            let e = Point::new(p0.x + (p1.x - p0.x) * rs, p0.y + (p1.y - p0.y) * rs);
            let er = r0 + (r1 - r0) * rs;
            self.annulus(ctm, e, er, 0.0, p0, r0, 0.0, count);
        }
        self.annulus(ctm, p0, r0, 0.0, p1, r1, 1.0, count);
        if extend[1] {
            let rs = if r0 > r1 { r1 / (r1 - r0) } else { -HUGENUM };
            let e = Point::new(p1.x + (p0.x - p1.x) * rs, p1.y + (p0.y - p1.y) * rs);
            let er = r1 + (r0 - r1) * rs;
            self.annulus(ctm, p1, r1, 1.0, e, er, 1.0, count);
        }
    }

    fn read_vertex(&self, bits: &mut Bits, m: &MeshParams, ctm: Matrix, ncomp: usize) -> Vertex {
        let x = bits.sample(m.bpcoord, m.x0, m.x1);
        let y = bits.sample(m.bpcoord, m.y0, m.y1);
        let mut c = [0.0f32; 32];
        for i in 0..ncomp {
            c[i] = bits.sample(m.bpcomp, m.c0[i], m.c1[i]);
        }
        self.vertex(ctm, x, y, &c[..ncomp.max(1)])
    }

    // MuPDF: fz_process_shade_type4 (shade.c:356)
    fn type4(&mut self, ctm: Matrix) {
        let Kind::Mesh { m, data, .. } = &self.shade.kind else { return };
        let (m, data) = (m.clone(), data.clone());
        let ncomp = self.ncomp();
        let mut bits = Bits { data: &data, pos: 0, bits: 0, avail: 0 };
        let (mut va, mut vb, mut vc) = (Vertex::default(), Vertex::default(), Vertex::default());
        let mut first = true;
        while !bits.eof() {
            let mut flag = bits.read(m.bpflag as u32);
            let vd = self.read_vertex(&mut bits, &m, ctm, ncomp);
            if first {
                flag = 0; // "ignoring non-zero edge flags for first vertex"
                first = false;
            }
            match flag {
                1 => {
                    va = vb;
                    vb = vc;
                    vc = vd;
                    self.tri(&va, &vb, &vc);
                }
                2 => {
                    vb = vc;
                    vc = vd;
                    self.tri(&va, &vb, &vc);
                }
                _ => {
                    // 0 (and out-of-range, "ignoring out of range edge flag"):
                    // start a new triangle.
                    va = vd;
                    bits.read(m.bpflag as u32);
                    vb = self.read_vertex(&mut bits, &m, ctm, ncomp);
                    bits.read(m.bpflag as u32);
                    vc = self.read_vertex(&mut bits, &m, ctm, ncomp);
                    self.tri(&va, &vb, &vc);
                }
            }
        }
    }

    // MuPDF: fz_process_shade_type5 (shade.c:442)
    fn type5(&mut self, ctm: Matrix) {
        let Kind::Mesh { m, data, .. } = &self.shade.kind else { return };
        let (m, data) = (m.clone(), data.clone());
        let ncomp = self.ncomp();
        let vprow = m.vprow.max(2) as usize;
        let mut bits = Bits { data: &data, pos: 0, bits: 0, avail: 0 };
        let mut refv: Vec<Vertex> = Vec::new();
        let mut first = true;
        while !bits.eof() {
            let buf: Vec<Vertex> = (0..vprow).map(|_| self.read_vertex(&mut bits, &m, ctm, ncomp)).collect();
            if !first {
                for i in 0..vprow - 1 {
                    self.quad(&refv[i], &refv[i + 1], &buf[i + 1], &buf[i]);
                }
            }
            refv = buf;
            first = false;
        }
    }

    // MuPDF: fz_process_shade_type6 / _type7 (shade.c:760, 858)
    fn type67(&mut self, ctm: Matrix, typ: u8) {
        let Kind::Mesh { m, data, .. } = &self.shade.kind else { return };
        let (m, data) = (m.clone(), data.clone());
        let ncomp = self.ncomp();
        let npts = if typ == 6 { 12 } else { 16 };
        let mut bits = Bits { data: &data, pos: 0, bits: 0, avail: 0 };
        let mut prev: Option<(Vec<Point>, [[f32; 32]; 4])> = None;
        while !bits.eof() {
            let flag = bits.read(m.bpflag as u32);
            let (startpt, startcolor) = if flag == 0 { (0, 0) } else { (4, 2) };
            let mut v = vec![Point::new(0.0, 0.0); npts];
            let mut c = [[0.0f32; 32]; 4];
            for p in v.iter_mut().skip(startpt) {
                let x = bits.sample(m.bpcoord, m.x0, m.x1);
                let y = bits.sample(m.bpcoord, m.y0, m.y1);
                *p = Point::new(x, y).transform(ctm);
            }
            for ci in c.iter_mut().skip(startcolor) {
                for (k, slot) in ci.iter_mut().enumerate().take(ncomp) {
                    *slot = bits.sample(m.bpcomp, m.c0[k], m.c1[k]);
                }
            }
            if flag != 0 {
                let Some((pp, pc)) = &prev else { continue };
                let (idx, cidx): ([usize; 4], [usize; 2]) = match flag {
                    1 => ([3, 4, 5, 6], [1, 2]),
                    2 => ([6, 7, 8, 9], [2, 3]),
                    3 => ([9, 10, 11, 0], [3, 0]),
                    _ => continue,
                };
                for (j, &i) in idx.iter().enumerate() {
                    v[j] = pp[i];
                }
                c[0] = pc[cidx[0]];
                c[1] = pc[cidx[1]];
            }
            let patch = make_tensor_patch(typ, &v, &c);
            self.draw_patch(&patch, SUBDIV, SUBDIV, ncomp);
            prev = Some((v, c));
        }
    }

    // MuPDF: triangulate_patch (shade.c:497)
    fn triangulate_patch(&mut self, p: &Patch, ncomp: usize) {
        let v0 = self.colored(p.pole[0][0], &p.color[0][..ncomp.max(1)]);
        let v1 = self.colored(p.pole[0][3], &p.color[1][..ncomp.max(1)]);
        let v2 = self.colored(p.pole[3][3], &p.color[2][..ncomp.max(1)]);
        let v3 = self.colored(p.pole[3][0], &p.color[3][..ncomp.max(1)]);
        self.quad(&v0, &v1, &v2, &v3);
    }

    // MuPDF: draw_stripe (shade.c:586)
    fn draw_stripe(&mut self, p: &Patch, depth: u32, ncomp: usize) {
        let (s0, s1) = split_stripe(p, ncomp);
        let depth = depth - 1;
        if depth == 0 {
            self.triangulate_patch(&s1, ncomp);
            self.triangulate_patch(&s0, ncomp);
        } else {
            self.draw_stripe(&s1, depth, ncomp);
            self.draw_stripe(&s0, depth, ncomp);
        }
    }

    // MuPDF: draw_patch (shade.c:633)
    fn draw_patch(&mut self, p: &Patch, depth: u32, origdepth: u32, ncomp: usize) {
        let (s0, s1) = split_patch(p, ncomp);
        let depth = depth - 1;
        if depth == 0 {
            self.draw_stripe(&s0, origdepth, ncomp);
            self.draw_stripe(&s1, origdepth, ncomp);
        } else {
            self.draw_patch(&s0, depth, origdepth, ncomp);
            self.draw_patch(&s1, depth, origdepth, ncomp);
        }
    }
}

/// `tensor_patch` (shade.c:490).
#[derive(Clone)]
struct Patch {
    pole: [[Point; 4]; 4],
    color: [[f32; 32]; 4],
}

// MuPDF: split_curve (shade.c:512), for the four poles at `idx`.
fn split_curve(pole: [Point; 4]) -> ([Point; 4], [Point; 4]) {
    let x12 = (pole[1].x + pole[2].x) * 0.5;
    let y12 = (pole[1].y + pole[2].y) * 0.5;
    let mut q0 = [Point::new(0.0, 0.0); 4];
    let mut q1 = [Point::new(0.0, 0.0); 4];
    q0[1] = Point::new((pole[0].x + pole[1].x) * 0.5, (pole[0].y + pole[1].y) * 0.5);
    q1[2] = Point::new((pole[2].x + pole[3].x) * 0.5, (pole[2].y + pole[3].y) * 0.5);
    q0[2] = Point::new((q0[1].x + x12) * 0.5, (q0[1].y + y12) * 0.5);
    q1[1] = Point::new((x12 + q1[2].x) * 0.5, (y12 + q1[2].y) * 0.5);
    q0[3] = Point::new((q0[2].x + q1[1].x) * 0.5, (q0[2].y + q1[1].y) * 0.5);
    q1[0] = q0[3];
    q0[0] = pole[0];
    q1[3] = pole[3];
    (q0, q1)
}

fn midcolor(a: &[f32; 32], b: &[f32; 32], n: usize) -> [f32; 32] {
    let mut c = [0.0f32; 32];
    for i in 0..n {
        c[i] = (a[i] + b[i]) * 0.5;
    }
    c
}

// MuPDF: split_stripe (shade.c:552) -- split the horizontal curves (pole
// columns, stride 4 in the C) into two half-width patches.
fn split_stripe(p: &Patch, n: usize) -> (Patch, Patch) {
    let mut s0 = p.clone();
    let mut s1 = p.clone();
    for col in 0..4 {
        let curve = [p.pole[0][col], p.pole[1][col], p.pole[2][col], p.pole[3][col]];
        let (a, b) = split_curve(curve);
        for row in 0..4 {
            s0.pole[row][col] = a[row];
            s1.pole[row][col] = b[row];
        }
    }
    s0.color[0] = p.color[0];
    s0.color[1] = p.color[1];
    s0.color[2] = midcolor(&p.color[1], &p.color[2], n);
    s0.color[3] = midcolor(&p.color[0], &p.color[3], n);
    s1.color[0] = s0.color[3];
    s1.color[1] = s0.color[2];
    s1.color[2] = p.color[2];
    s1.color[3] = p.color[3];
    (s0, s1)
}

// MuPDF: split_patch (shade.c:608) -- split the vertical curves (each pole
// row, stride 1) into two half-height patches.
fn split_patch(p: &Patch, n: usize) -> (Patch, Patch) {
    let mut s0 = p.clone();
    let mut s1 = p.clone();
    for row in 0..4 {
        let (a, b) = split_curve(p.pole[row]);
        s0.pole[row] = a;
        s1.pole[row] = b;
    }
    s0.color[0] = p.color[0];
    s0.color[1] = midcolor(&p.color[0], &p.color[1], n);
    s0.color[2] = midcolor(&p.color[2], &p.color[3], n);
    s0.color[3] = p.color[3];
    s1.color[0] = s0.color[1];
    s1.color[1] = p.color[1];
    s1.color[2] = p.color[2];
    s1.color[3] = s0.color[2];
    (s0, s1)
}

// MuPDF: compute_tensor_interior (shade.c:655)
#[allow(clippy::too_many_arguments)]
fn tensor_interior(a: Point, b: Point, c: Point, d: Point, e: Point, f: Point, g: Point, h: Point) -> Point {
    let mut x = -4.0 * a.x;
    x += 6.0 * (b.x + c.x);
    x += -2.0 * (d.x + e.x);
    x += 3.0 * (f.x + g.x);
    x += -h.x;
    x /= 9.0;
    let mut y = -4.0 * a.y;
    y += 6.0 * (b.y + c.y);
    y += -2.0 * (d.y + e.y);
    y += 3.0 * (f.y + g.y);
    y += -h.y;
    y /= 9.0;
    Point::new(x, y)
}

// MuPDF: make_tensor_patch (shade.c:681)
fn make_tensor_patch(typ: u8, pt: &[Point], c: &[[f32; 32]; 4]) -> Patch {
    let mut p = Patch { pole: [[Point::new(0.0, 0.0); 4]; 4], color: *c };
    p.pole[0][0] = pt[0];
    p.pole[0][1] = pt[1];
    p.pole[0][2] = pt[2];
    p.pole[0][3] = pt[3];
    p.pole[1][3] = pt[4];
    p.pole[2][3] = pt[5];
    p.pole[3][3] = pt[6];
    p.pole[3][2] = pt[7];
    p.pole[3][1] = pt[8];
    p.pole[3][0] = pt[9];
    p.pole[2][0] = pt[10];
    p.pole[1][0] = pt[11];
    if typ == 6 {
        let q = p.pole;
        p.pole[1][1] = tensor_interior(q[0][0], q[0][1], q[1][0], q[0][3], q[3][0], q[3][1], q[1][3], q[3][3]);
        p.pole[1][2] = tensor_interior(q[0][3], q[0][2], q[1][3], q[0][0], q[3][3], q[3][2], q[1][0], q[3][0]);
        p.pole[2][1] = tensor_interior(q[3][0], q[3][1], q[2][0], q[3][3], q[0][0], q[0][1], q[2][3], q[0][3]);
        p.pole[2][2] = tensor_interior(q[3][3], q[3][2], q[2][3], q[3][0], q[0][3], q[0][2], q[2][0], q[0][0]);
    } else {
        p.pole[1][1] = pt[12];
        p.pole[1][2] = pt[13];
        p.pole[2][2] = pt[14];
        p.pole[2][1] = pt[15];
    }
    p
}

/// `edge_data` (draw-mesh.c): x, dx and the 16.16 fixed-point components.
struct Edge {
    x: f32,
    dx: f32,
    v: [i32; 3],
    dv: [i32; 3],
}

// MuPDF: prepare_edge (draw-mesh.c:78)
fn prepare_edge(vtop: &Vertex, vbot: &Vertex, y: f32, n: usize) -> Edge {
    let r = 1.0 / (vbot.p.y - vtop.p.y);
    let t = (y - vtop.p.y) * r;
    let diff = vbot.p.x - vtop.p.x;
    let mut e = Edge { x: vtop.p.x + diff * t, dx: diff * r, v: [0; 3], dv: [0; 3] };
    for i in 0..n {
        let diff = vbot.c[i] - vtop.c[i];
        e.v[i] = (65536.0 * (vtop.c[i] + diff * t)) as i32;
        e.dv[i] = (65536.0 * diff * r) as i32;
    }
    e
}

// MuPDF: step_edge (draw-mesh.c:96)
fn step_edge(e: &mut Edge, n: usize) {
    e.x += e.dx;
    for i in 0..n {
        e.v[i] = e.v[i].wrapping_add(e.dv[i]);
    }
}

// MuPDF: paint_scan (draw-mesh.c:31) -- `n` components then alpha 255.
#[allow(clippy::too_many_arguments)]
fn paint_scan(pix: &mut Buf, y: i32, fx0: i32, fx1: i32, cx0: i32, cx1: i32, v0: &[i32; 3], v1: &[i32; 3], n: usize) {
    let (fx0, fx1, v0, v1) = if fx0 > fx1 {
        (fx1, fx0, v1, v0)
    } else if fx0 == fx1 {
        return;
    } else {
        (fx0, fx1, v0, v1)
    };
    if fx0 >= cx1 || fx1 <= cx0 {
        return;
    }
    let x0 = fx0.max(cx0);
    let x1 = fx1.min(cx1);
    let w = x1 - x0;
    if w == 0 || y < pix.y || y >= pix.y + pix.h {
        return;
    }
    let div = 1.0f32 / (fx1 - fx0) as f32;
    let mul = (x0 - fx0) as f32;
    let mut c = [0i32; 3];
    let mut dc = [0i32; 3];
    for k in 0..n {
        // C: `dc[k] = (v1[k] - v0[k]) * div; c[k] = v0[k] + dc[k] * mul;` --
        // both evaluated in float and truncated on the store.
        dc[k] = ((v1[k] - v0[k]) as f32 * div) as i32;
        c[k] = (v0[k] as f32 + dc[k] as f32 * mul) as i32;
    }
    let row = (y - pix.y) as usize * pix.w as usize;
    for x in x0..x1 {
        if x < pix.x || x >= pix.x + pix.w {
            for k in 0..n {
                c[k] = c[k].wrapping_add(dc[k]);
            }
            continue;
        }
        let o = (row + (x - pix.x) as usize) * pix.n;
        for k in 0..n {
            pix.samples[o + k] = (c[k] >> 16) as u8;
            c[k] = c[k].wrapping_add(dc[k]);
        }
        pix.samples[o + pix.n - 1] = 255;
    }
}

// MuPDF: fz_paint_triangle (draw-mesh.c:106)
fn paint_triangle(pix: &mut Buf, v: [&Vertex; 3], n: usize, bbox: IRect) {
    let (mut top, mut bot) = (0usize, 0usize);
    if v[1].p.y < v[0].p.y {
        top = 1;
    } else {
        bot = 1;
    }
    if v[2].p.y < v[top].p.y {
        top = 2;
    } else if v[2].p.y > v[bot].p.y {
        bot = 2;
    }
    if v[top].p.y == v[bot].p.y {
        return;
    }
    if v[bot].p.y < bbox.y0 as f32 || v[top].p.y > bbox.y1 as f32 {
        return;
    }
    let mid = 3 ^ top ^ bot;
    let minx = bbox.x0.max(pix.x);
    let maxx = bbox.x1.min(pix.x + pix.w);
    let mut y = (bbox.y0 as f32).max(v[top].p.y).ceil();
    let mut y1 = (bbox.y1 as f32).min(v[mid].p.y).ceil();
    let mut e0 = prepare_edge(v[top], v[bot], y, n);
    if y < y1 {
        let mut e1 = prepare_edge(v[top], v[mid], y, n);
        loop {
            paint_scan(pix, y as i32, e0.x as i32, e1.x as i32, minx, maxx, &e0.v, &e1.v, n);
            step_edge(&mut e0, n);
            step_edge(&mut e1, n);
            y += 1.0;
            if y >= y1 {
                break;
            }
        }
    }
    y1 = (bbox.y1 as f32).min(v[bot].p.y).ceil();
    if y < y1 {
        let mut e1 = prepare_edge(v[mid], v[bot], y, n);
        loop {
            paint_scan(pix, y as i32, e0.x as i32, e1.x as i32, minx, maxx, &e0.v, &e1.v, n);
            y += 1.0;
            if y >= y1 {
                break;
            }
            step_edge(&mut e0, n);
            step_edge(&mut e1, n);
        }
    }
}
