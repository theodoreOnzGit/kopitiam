#!/usr/bin/env bash
#
# skeleton-gen.sh -- Task III-3 (skeleton-first translation), kopitiam_token_max.md §12.
# SPDX-License-Identifier: AGPL-3.0-only
#
# A DETERMINISTIC porting aid (§0.3: "deterministic tool over model call").
# Given a vendored source file, it mechanically emits Rust signature STUBS --
# function signatures + struct definitions with `todo!()` bodies -- so a porting
# agent writes bodies ONLY, instead of transcribing every signature by hand.
#
# WHY (§302, §0.3): on a long port the naming/return-type shape of each symbol
# is a mechanical function of the source. Re-deriving it with a model costs
# tokens twice over: once to generate, and again every later session to keep the
# naming consistent by re-reading earlier output. This tool derives it once,
# deterministically, so the agent spends judgement only on bodies.
#
# It is HONEST about being a heuristic (§ no-silent-caps): the C side is a
# regex/brace scanner, NOT a real C parser. Multi-line macros, function-pointer
# typedefs and unusual declarations may be missed or mis-typed. Every emitted
# skeleton carries a `// SKELETON-GEN: review` banner and an inline found/skipped
# tally; `--report` prints the full breakdown so nothing is silently dropped.
#
# ---------------------------------------------------------------------------
# USAGE
#   scripts/skeleton-gen.sh <source-file>            # Rust stubs to stdout
#   scripts/skeleton-gen.sh --report <source-file>   # found/skipped TSV
#   scripts/skeleton-gen.sh --help
#
#   --commit <sha>   override the provenance commit SHA (default: auto-detect
#                    from a sibling ported .rs header, else a fill-me placeholder)
#
# Language is chosen by extension: .c/.h/.cc/.cpp -> C mode; .py -> Python mode.
#
# ---------------------------------------------------------------------------
# TYPE MAP (C -> Rust)  --  the documented, deterministic guesses
#
#   int int32_t                 -> i32          float           -> f32
#   unsigned unsigned int uint32_t -> u32       double          -> f64
#   short int16_t               -> i16          char            -> i8
#   unsigned short uint16_t     -> u16          signed char     -> i8
#   long int64_t                -> i64          unsigned char   -> u8
#   unsigned long uint64_t      -> u64          int8_t          -> i8
#   long long                   -> i64          uint8_t         -> u8
#   size_t                      -> usize        int64_t         -> i64
#   ptrdiff_t intptr_t          -> isize        uint64_t        -> u64
#   uintptr_t                   -> usize        void (return)   -> (omitted)
#
#   fz_matrix -> Matrix   fz_point -> Point   fz_rect -> Rect
#   fz_irect  -> IRect    fz_quad  -> Quad    fz_context -> (dropped, AID-0051 §5)
#   any other fz_<x>/pdf_<x> -> strip prefix, PascalCase  (fz_text_span -> TextSpan)
#
#   POINTERS (§ "pointers -> references/Option"):
#     T *          (const)  -> &T           T *   (mutable) -> &mut T
#     T **                  -> *mut *mut T  (flagged for review)
#   SLICE / STRING special-cases:
#     unsigned char *       -> &[u8]
#     const char *          -> &str         char * -> &str  (flagged: may be &mut [u8])
#     void *                -> *mut core::ffi::c_void  (flagged)
#   fz_context * params are DROPPED entirely (AID-0051 §5: context -> Rust ownership).
#   A C function-pointer param becomes `name: /* TODO: C fn pointer */ ()` (flagged).
#
# TYPE MAP (Python -> Rust)
#   Python is untyped, so parameters/returns default to the placeholder `Todo`
#   (the porter replaces each). Where a PEP-484 annotation exists it is mapped:
#     int->i64  float->f64  str->String  bool->bool  bytes->Vec<u8>
#     list->Vec<Todo>  dict->Todo  None->()  (anything else -> Todo)
#   `self` -> `&self` receiver; `cls` / no receiver for classmethod/staticmethod.
#
# KNOWN HEURISTIC LIMITATIONS (be sceptical of the output; it is a starting point)
#   * Not a real parser. Brace/paren depth + regex only. Preprocessor macros are
#     skipped wholesale (counted under --report), so macro-defined functions and
#     X-macro tables are invisible.
#   * `char *` mutability, pointer ownership (owned vs borrowed vs out-param), and
#     pointer-to-pointer are guesses -- the porter MUST fix signatures.
#   * Python parameter/return types are unknowable from source; `Todo` is a
#     deliberate compile error so the gap is impossible to overlook.
#   * Comment/string stripping is heuristic: a `//` or `/*` inside a string
#     literal can confuse it (rare in signatures).
#
# DETERMINISM (acceptance §12): same input -> byte-identical output. LC_ALL=C,
# single source-ordered pass, no timestamps / hostnames / absolute paths (the
# source appears by its vendor-relative path only). Prove it:
#   a=$(scripts/skeleton-gen.sh F); b=$(scripts/skeleton-gen.sh F); [ "$a" = "$b" ]
# ---------------------------------------------------------------------------

set -euo pipefail
export LC_ALL=C

PROG=${0##*/}

usage() {
  sed -n '2,120p' "$0" | sed 's/^# \{0,1\}//; s/^#$//'
}

# ----- argument parsing ----------------------------------------------------
mode=skeleton
commit_override=""
src=""
while [ $# -gt 0 ]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --report) mode=report; shift ;;
    --commit) commit_override=${2:-}; shift 2 ;;
    --) shift; src=${1:-}; shift || true ;;
    -*) echo "$PROG: unknown option: $1" >&2; exit 2 ;;
    *) src=$1; shift ;;
  esac
done

if [ -z "$src" ]; then
  echo "$PROG: no source file given (try --help)" >&2
  exit 2
fi
if [ ! -f "$src" ]; then
  echo "$PROG: not a file: $src" >&2
  exit 2
fi

# ----- language + provenance detection -------------------------------------
# Normalise path separators for matching / relative-path extraction.
srcnorm=${src//\\//}
ext=${src##*.}
ext=$(printf '%s' "$ext" | tr '[:upper:]' '[:lower:]')

lang=""
case "$ext" in
  c|h|cc|cpp|cxx|hpp) lang=c ;;
  py) lang=py ;;
  *) echo "$PROG: unsupported extension '.$ext' (want .c/.h/.cc/.cpp or .py)" >&2; exit 2 ;;
esac

# provenance: kind (mupdf | pdf2md | generic), vendor-relative source path,
# optional matching header, license/copyright phrasing, and commit SHA source.
kind=generic
srcrel="$srcnorm"
hdrrel=""
sibling_dir=""
case "$srcnorm" in
  *vendor/mupdf/*)
    kind=mupdf
    srcrel=${srcnorm#*vendor/mupdf/}
    sibling_dir="crates/kopitiam-pdf/src/mupdf"
    # source/fitz/X.c has a companion include/mupdf/fitz/X.h -- cite it if present.
    base=${srcrel##*/}; stem=${base%.*}
    case "$srcrel" in
      source/fitz/*.c)
        cand="crates/kopitiam-pdf/vendor/mupdf/include/mupdf/fitz/$stem.h"
        [ -f "$cand" ] && hdrrel="include/mupdf/fitz/$stem.h" ;;
    esac
    ;;
  *vendor/pdf-to-markdown/*)
    kind=pdf2md
    srcrel=${srcnorm#*vendor/}          # keep pdf-to-markdown/... for clarity
    ;;
esac

# commit SHA: override wins; else scrape a sibling ported header; else placeholder.
commit=""
if [ -n "$commit_override" ]; then
  commit="$commit_override"
elif [ -n "$sibling_dir" ] && [ -d "$sibling_dir" ]; then
  # first (sorted) sibling that carries a `commit <hex>` provenance line
  commit=$(grep -rhoE 'commit [0-9a-f]{7,40}' "$sibling_dir" 2>/dev/null \
             | sort | head -n1 | awk '{print $2}') || true
fi
[ -n "$commit" ] || commit="<COMMIT-SHA — porter fills>"

# short basename for breadcrumbs / todo!() strings (stable, no abs path)
srcbase=${srcrel##*/}
stem=${srcbase%.*}

# ===========================================================================
# The generator. One gawk program; MODE selects the C or Python front-end.
# ===========================================================================
read -r -d '' AWK <<'AWK_EOF' || true
# ---- small helpers --------------------------------------------------------
function trim(s) { gsub(/^[ \t]+/,"",s); gsub(/[ \t]+$/,"",s); return s }
function squeeze(s) { gsub(/[ \t]+/," ",s); return trim(s) }

function is_reserved(w) {
  return (" as break const continue crate dyn else enum extern false fn for if " \
          "impl in let loop match mod move mut pub ref return self Self static " \
          "struct super trait true type unsafe use where while async await dyn box " \
          "abstract become do final macro override priv typeof unsized virtual yield ") \
         ~ (" " w " ")
}
function ident_ok(w) { return is_reserved(w) ? (w "_") : w }

function pascal(s,   n,a,i,out,p) {
  gsub(/[^A-Za-z0-9]+/," ",s); s = trim(s)
  n = split(s, a, / +/); out=""
  for (i=1;i<=n;i++) { p=a[i]; out = out toupper(substr(p,1,1)) substr(p,2) }
  return out
}
# fz_text_span -> TextSpan ; pdf_obj -> Obj ; keeps a mapped-type name findable.
# Canonical AID-0051 §3 names win over the generic PascalCase rule.
function type_name(s) {
  if (s=="fz_matrix") return "Matrix"
  if (s=="fz_point")  return "Point"
  if (s=="fz_rect")   return "Rect"
  if (s=="fz_irect")  return "IRect"
  if (s=="fz_quad")   return "Quad"
  if (s=="fz_context") return "Context"
  sub(/^fz_/,"",s); sub(/^pdf_/,"",s)
  return pascal(s)
}

# ---- C type mapping -------------------------------------------------------
# Returns rust type for a C type string; sets FLAG=1 when the guess is shaky.
function map_base(b) {
  if (b=="int"||b=="int32_t"||b=="signed"||b=="signed int") return "i32"
  if (b=="unsigned"||b=="unsigned int"||b=="uint32_t"||b=="uint") return "u32"
  if (b=="short"||b=="short int"||b=="int16_t") return "i16"
  if (b=="unsigned short"||b=="uint16_t") return "u16"
  if (b=="long"||b=="long int"||b=="int64_t"||b=="long long"||b=="long long int") return "i64"
  if (b=="unsigned long"||b=="uint64_t"||b=="unsigned long long") return "u64"
  if (b=="char") return "i8"
  if (b=="signed char"||b=="int8_t") return "i8"
  if (b=="unsigned char"||b=="uint8_t") return "u8"
  if (b=="size_t") return "usize"
  if (b=="ptrdiff_t"||b=="intptr_t") return "isize"
  if (b=="uintptr_t") return "usize"
  if (b=="float") return "f32"
  if (b=="double") return "f64"
  if (b=="void") return "()"
  if (b=="fz_context") return "Context"
  if (b ~ /^(fz_|pdf_)/) return type_name(b)
  if (b=="") { FLAG=1; return "()" }
  # unknown bare type: PascalCase it and flag for review
  FLAG=1
  return pascal(b)
}

# Full C type -> Rust. `pos` = "ret" or "param" (affects void handling).
function map_ctype(ct, pos,   c, isconst, stars, base, rb) {
  c = " " squeeze(ct) " "
  isconst = (c ~ / const /)
  # drop storage-class / qualifier / attribute-macro noise
  gsub(/ static /," ",c); gsub(/ inline /," ",c); gsub(/ __inline /," ",c)
  gsub(/ extern /," ",c); gsub(/ register /," ",c); gsub(/ auto /," ",c)
  gsub(/ FZ_[A-Z0-9_]+ /," ",c); gsub(/ PDF_[A-Z0-9_]+ /," ",c)
  gsub(/ const /," ",c); gsub(/ struct /," ",c); gsub(/ union /," ",c); gsub(/ enum /," ",c); gsub(/ volatile /," ",c)
  stars = gsub(/\*/,"",c)
  base = squeeze(c)

  # string / slice special cases
  if (stars==1 && base=="unsigned char") return "&[u8]"
  if (stars==1 && base=="char") { if (!isconst) FLAG=1; return "&str" }
  if (stars>=1 && base=="void") { FLAG=1; return isconst ? "*const core::ffi::c_void" : "*mut core::ffi::c_void" }

  rb = map_base(base)
  if (stars==0) {
    if (rb=="()" && pos=="ret") return ""   # void return -> no `-> T`
    return rb
  }
  if (stars==1) return isconst ? ("&" rb) : ("&mut " rb)
  # pointer-to-pointer and deeper: honest, flagged
  FLAG=1
  { c=""; for (i=0;i<stars;i++) c = c "*mut "; return c rb }
}

# split a parameter list on TOP-LEVEL commas (keeps fn-pointer params intact)
function split_params(s, arr,   n,i,ch,depth,cur,cnt) {
  n=length(s); depth=0; cur=""; cnt=0
  for (i=1;i<=n;i++) {
    ch=substr(s,i,1)
    if (ch=="("||ch=="["||ch=="<") depth++
    else if (ch==")"||ch=="]"||ch==">") depth--
    if (ch=="," && depth==0) { cnt++; arr[cnt]=trim(cur); cur="" }
    else cur = cur ch
  }
  if (trim(cur)!="") { cnt++; arr[cnt]=trim(cur) }
  return cnt
}

# one C parameter -> "name: RustType" (or "" when dropped, e.g. fz_context*)
function map_param(p, idx,   name, type, m, rest) {
  p = squeeze(p)
  if (p=="" || p=="void") return ""
  # C function pointer: `ret (*name)(args)`
  if (p ~ /\(\*/) {
    if (match(p, /\(\*[ \t]*([A-Za-z_][A-Za-z0-9_]*)/)) {
      name = substr(p, RSTART+2, RLENGTH-2); gsub(/^[ \t*]+/,"",name)
    } else name = "f" idx
    FLAG=1
    return ident_ok(name) ": /* TODO: C fn pointer */ ()"
  }
  # drop the fz_context pointer entirely (AID-0051 §5)
  if (p ~ /fz_context[ \t]*\*/) return ""
  # trailing identifier is the name (pointer stars stay with the type)
  if (match(p, /[A-Za-z_][A-Za-z0-9_]*[ \t]*$/)) {
    name = trim(substr(p, RSTART, RLENGTH))
    type = trim(substr(p, 1, RSTART-1))
  } else { name=""; type=p }
  # array suffix e.g. `foo bar[4]` -> name captured was bar, but `[4]` sits after;
  # handle by detecting a bracket in the raw param
  if (type=="") { type=name; name="arg" idx }     # unnamed prototype param
  m = map_ctype(type, "param")
  return ident_ok(name) ": " m
}

# ---- emit one function stub -----------------------------------------------
function emit_func(cname, ret_c, params_c, line,   rustname, rname, np, pa, i, pl, one, ret_r, sig) {
  rname = cname; sub(/^fz_/,"",rname); sub(/^pdf_/,"",rname)
  rname = ident_ok(rname)
  np = split_params(params_c, pa)
  pl = ""
  for (i=1;i<=np;i++) {
    one = map_param(pa[i], i-1)
    if (one=="") continue
    pl = (pl=="") ? one : (pl ", " one)
  }
  ret_r = map_ctype(ret_c, "ret")
  sig = "pub fn " rname "(" pl ")"
  if (ret_r!="" && ret_r!="()") sig = sig " -> " ret_r
  OUT = OUT BREAD_PFX cname " (" SRCBASE ":" line ")\n"
  OUT = OUT sig " {\n"
  OUT = OUT "    todo!(\"port from " SRCREL ":" line "\")\n"
  OUT = OUT "}\n\n"
  NFUNC++
}

# ---- emit one struct ------------------------------------------------------
function emit_struct(cname, body, line,   rn, nf, fa, i, fld, ftype, names, nn, j, one, arrsz) {
  rn = type_name(cname)
  OUT = OUT BREAD_PFX cname " (" SRCBASE ":" line ")\n"
  OUT = OUT "#[derive(Clone, Copy, Debug, PartialEq)]\n"
  OUT = OUT "pub struct " rn " {\n"
  nf = split(body, fa, /;/)
  for (i=1;i<=nf;i++) {
    fld = squeeze(fa[i])
    if (fld=="") continue
    if (fld ~ /\(\*/ || fld ~ /:/ ) {           # fn-pointer field or bitfield: punt
      OUT = OUT "    // TODO: " fld "\n"; NFIELDSKIP++; continue
    }
    arrsz = ""
    if (match(fld, /\[[0-9]+\]/)) { arrsz = substr(fld, RSTART+1, RLENGTH-2); sub(/\[[0-9]+\]/,"",fld); fld=squeeze(fld) }
    # split "type a, b, c" -> type + name list
    if (!match(fld, /[A-Za-z_][A-Za-z0-9_ ,*]*$/)) { OUT = OUT "    // TODO: " fld "\n"; NFIELDSKIP++; continue }
    # last space separates type from the (comma) name list; find first name token
    if (match(fld, /[ \t*]+[A-Za-z_][A-Za-z0-9_]*([ \t]*,[ \t]*[A-Za-z_][A-Za-z0-9_]*)*$/)) {
      ftype = trim(substr(fld,1,RSTART))
      names = trim(substr(fld,RSTART))
    } else { OUT = OUT "    // TODO: " fld "\n"; NFIELDSKIP++; continue }
    gsub(/\*/,"",names)
    nn = split(names, na2, /[ \t]*,[ \t]*/)
    for (j=1;j<=nn;j++) {
      one = map_ctype(ftype (fld ~ /\*/ ? " *" : ""), "param")
      if (arrsz!="") one = "[" one "; " arrsz "]"
      OUT = OUT "    pub " ident_ok(trim(na2[j])) ": " one ",\n"
    }
  }
  OUT = OUT "}\n\n"
  NSTRUCT++
}
split("", na2)   # declare na2 as array (gawk hygiene)

# ==================== C FRONT-END ==========================================
# Pass 1 (per line): strip comments + preprocessor, remember original line no.
MODE=="c" {
  line = $0
  out = ""
  n = length(line); i = 1
  while (i <= n) {
    if (INBLK) {
      p = index(substr(line,i), "*/")
      if (p==0) { i = n+1 } else { INBLK=0; i = i + p + 1 }
      continue
    }
    two = substr(line,i,2)
    if (two=="/*") { INBLK=1; i+=2; continue }
    if (two=="//") { break }
    out = out substr(line,i,1); i++
  }
  # preprocessor line?
  if (out ~ /^[ \t]*#/) {
    if (out ~ /^[ \t]*#[ \t]*define[ \t]+[A-Za-z_][A-Za-z0-9_]*\(/) { SKIP["function-like macro (#define)"]++; NSKIP++ }
    if (out ~ /\\[ \t]*$/) CONTPP=1; else CONTPP=0
    code[NR] = ""; maxnr=NR; next
  }
  if (CONTPP) { if (out !~ /\\[ \t]*$/) CONTPP=0; code[NR]=""; maxnr=NR; next }
  code[NR] = out " "     # trailing space = safe inter-line token separator
  maxnr = NR
  next
}

# ==================== PYTHON FRONT-END =====================================
MODE=="py" {
  pyline[NR] = $0
  maxnr = NR
  next
}

# ==================== END: drive the right scanner =========================
END {
  if (MODE=="c") scan_c()
  else           scan_py()
  finish()
}

# ---- C scanner: brace/paren state machine over the stripped stream --------
function scan_c(   ln, s, i, c, state, skipdepth, aggdepth, unit, unit_line, agg_intro, agg_body, agg_line) {
  state=0; skipdepth=0; aggdepth=0; unit=""; unit_line=0
  for (ln=1; ln<=maxnr; ln++) {
    s = code[ln]; if (s=="") continue
    L = length(s)
    for (i=1;i<=L;i++) {
      c = substr(s,i,1)
      if (state==1) {                         # skipping a function/unknown body
        if (c=="{") skipdepth++
        else if (c=="}") { skipdepth--; if (skipdepth<=0) state=0 }
        continue
      }
      if (state==2) {                         # inside an aggregate body
        if (c=="{") { aggdepth++; agg_body = agg_body c }
        else if (c=="}") { aggdepth--; if (aggdepth<=0) { state=3 } else agg_body = agg_body c }
        else agg_body = agg_body c
        continue
      }
      if (state==3) {                         # after aggregate close: read `name;`
        if (c==";") {
          nm = squeeze(unit)
          gsub(/[={].*/,"",nm)
          if (nm=="") { gsub(/[ \t].*/,"",agg_intro2); nm=agg_intro2 }
          if (nm=="") nm = agg_tag
          emit_struct(nm, agg_body, agg_line)
          state=0; unit=""
        } else unit = unit c
        continue
      }
      # state 0 : collecting a top-level unit
      if (c=="{") {
        u = squeeze(unit)
        if (u ~ /\(/) {                       # function definition
          parse_and_emit_func(u, unit_line)
          state=1; skipdepth=1
        } else if (u ~ /(^|[^A-Za-z0-9_])(struct|union|enum)([^A-Za-z0-9_]|$)/) {
          agg_intro = u; agg_body=""; agg_line=unit_line; aggdepth=1; state=2
          agg_tag = agg_intro; sub(/.*(struct|union|enum)[ \t]+/,"",agg_tag); gsub(/[ \t].*/,"",agg_tag)
          agg_intro2 = agg_tag
        } else {                              # initializer / unknown brace block
          if (u!="") { SKIP["non-decl brace block"]++; NSKIP++ }
          state=1; skipdepth=1
        }
        unit=""
      } else if (c==";") {
        u = squeeze(unit)
        if (u!="") {
          if (is_c_proto(u)) parse_and_emit_func(u, unit_line)
          else classify_skip(u)
        }
        unit=""
      } else if (c=="}") {
        unit=""                               # stray; ignore
      } else {
        if (unit=="" && c ~ /[ \t]/) continue
        if (unit=="") unit_line = ln
        unit = unit c
      }
    }
  }
}

# does a `;`-terminated top-level unit look like a function prototype?
function is_c_proto(u) {
  if (u ~ /^typedef/) return 0                 # typedef (incl fn-pointer typedef)
  if (u !~ /\)[ \t]*$/) return 0               # must end at a param list
  if (u !~ /[A-Za-z_][A-Za-z0-9_]*[ \t]*\(/) return 0
  if (u ~ /=/) return 0                        # has an initializer -> a variable
  return 1
}

function classify_skip(u) {
  if (u ~ /^typedef/) SKIP["typedef"]++
  else if (u ~ /(^|[^A-Za-z0-9_])(struct|union|enum)([^A-Za-z0-9_]|$)/) SKIP["struct/enum fwd-decl or var"]++
  else if (u ~ /extern/) SKIP["extern declaration"]++
  else SKIP["other top-level declaration"]++
  NSKIP++
}

# split a function unit "ret ... name(params)" and emit
function parse_and_emit_func(u, line,   op, name, ret, params, depth, i, ch, clo) {
  op = index(u, "(")
  if (op==0) { classify_skip(u); return }
  # matching close paren for the FIRST '('
  depth=0; clo=0
  for (i=op;i<=length(u);i++) { ch=substr(u,i,1); if(ch=="(")depth++; else if(ch==")"){depth--; if(depth==0){clo=i; break}} }
  if (clo==0) { SKIP["unbalanced parens"]++; NSKIP++; return }
  params = substr(u, op+1, clo-op-1)
  head = trim(substr(u, 1, op-1))
  # name = last identifier in head; the rest (incl pointer stars) is the return type
  if (!match(head, /[A-Za-z_][A-Za-z0-9_]*$/)) { SKIP["no function name"]++; NSKIP++; return }
  name = substr(head, RSTART, RLENGTH)
  ret  = trim(substr(head, 1, RSTART-1))
  if (name=="" ) { SKIP["no function name"]++; NSKIP++; return }
  # guard: things like `return foo(x)` can't appear at depth 0, but a stray
  # keyword head should still be skipped
  if (name=="if"||name=="for"||name=="while"||name=="switch"||name=="return"||name=="sizeof") { SKIP["control keyword"]++; NSKIP++; return }
  emit_func(name, ret, params, line)
}

# ---- Python scanner -------------------------------------------------------
function scan_py(   ln, s, indent, m, name, params, deco, i, cont, buf, startln, cls, clsname, clsindent) {
  clsname=""; clsindent=-1
  for (ln=1; ln<=maxnr; ln++) {
    s = pyline[ln]
    if (s ~ /^[ \t]*$/) continue
    if (s ~ /^[ \t]*#/) continue
    # current indent (spaces; tabs counted as 1 for zone purposes)
    indent = match(s, /[^ \t]/) - 1
    # leaving a class body?
    if (clsname!="" && indent <= clsindent && s !~ /^[ \t]*(def|@)/) {
      # close impl block if the dedent is real code
      if (indent <= clsindent) { close_impl(); clsname="" }
    }
    if (s ~ /^[ \t]*@/) continue               # decorator line (kind noted below)
    if (match(s, /^[ \t]*class[ \t]+([A-Za-z_][A-Za-z0-9_]*)/)) {
      if (clsname!="") close_impl()
      clsname = capture(s, "class[ \t]+([A-Za-z_][A-Za-z0-9_]*)", "class[ \t]+")
      clsindent = indent
      open_impl(clsname, ln)
      continue
    }
    if (s ~ /^[ \t]*def[ \t]/) {
      # accumulate a possibly multi-line signature until the ':' that ends it
      buf = s; startln = ln
      while (buf !~ /:[ \t]*(#.*)?$/ && ln < maxnr) { ln++; buf = buf " " pyline[ln] }
      handle_py_def(buf, startln, (clsname!="" && indent > clsindent) ? clsname : "")
      continue
    }
    # other top-level statement inside class -> a field-ish line; note & skip
    if (clsname!="" && indent > clsindent) { }  # ignore bodies
  }
  if (clsname!="") close_impl()
}
function capture(s, re, strip,   t) { match(s, re); t=substr(s,RSTART,RLENGTH); sub(strip,"",t); return trim(t) }
function open_impl(cn, ln,   rn) {
  rn = pascal(cn)
  OUT = OUT BREAD_PFX cn " (" SRCBASE ":" ln ")  [class]\n"
  OUT = OUT "pub struct " rn " {\n    // TODO: fields (Python carries state in __init__)\n}\n\n"
  OUT = OUT "impl " rn " {\n"
  IMPL_OPEN=1; NSTRUCT++
}
function close_impl() { if (IMPL_OPEN) { OUT = OUT "}\n\n"; IMPL_OPEN=0 } }

# map a Python annotation to a Rust type (best-effort); "" -> Todo
function py_type(a,   t) {
  t = trim(a)
  if (t=="int") return "i64"
  if (t=="float") return "f64"
  if (t=="str") return "String"
  if (t=="bool") return "bool"
  if (t=="bytes") return "Vec<u8>"
  if (t=="None") return "()"
  if (t ~ /^[Ll]ist/) return "Vec<Todo>"
  if (t ~ /^[Dd]ict/) return "Todo"
  return "Todo"
}
function handle_py_def(buf, line, inclass,   name, params, rp, ret, np, pa, i, one, pname, pann, recv, indent_pfx, op, clo, depth, c) {
  name = capture(buf, "def[ \t]+([A-Za-z_][A-Za-z0-9_]*)", "def[ \t]+")
  # params between first '(' and its matching ')'
  op = index(buf, "("); if (op==0) return
  depth=0; clo=0
  for (i=op;i<=length(buf);i++){ c=substr(buf,i,1); if(c=="(")depth++; else if(c==")"){depth--; if(depth==0){clo=i;break}} }
  params = substr(buf, op+1, clo-op-1)
  # return annotation after ')'
  rp = substr(buf, clo+1)
  ret = ""
  if (match(rp, /->[ \t]*[^:]+/)) { ret = substr(rp,RSTART+2,RLENGTH-2); sub(/#.*/,"",ret); ret = py_type(ret) }
  np = split_params(params, pa)
  recv=""; pl=""
  for (i=1;i<=np;i++) {
    pname = trim(pa[i]); if (pname=="") continue
    sub(/=.*/,"",pname)                        # drop default value
    pann=""
    if (index(pname,":")>0) { pann=trim(substr(pname,index(pname,":")+1)); pname=trim(substr(pname,1,index(pname,":")-1)) }
    pname = trim(pname); gsub(/[*]/,"",pname)
    if (pname=="self") { recv="&self"; continue }
    if (pname=="cls")  { recv="";      continue }
    if (pname=="")     continue
    one = ident_ok(pname) ": " (pann=="" ? "Todo" : py_type(pann))
    pl = (pl=="") ? one : (pl ", " one)
  }
  indent_pfx = (inclass!="") ? "    " : ""
  if (inclass!="") {
    OUT = OUT indent_pfx "// pdf2md: " STEM "." inclass "." name " (" SRCBASE ":" line ")\n"
  } else {
    OUT = OUT "// pdf2md: " STEM "." name " (" SRCBASE ":" line ")\n"
  }
  sig = indent_pfx "pub fn " ident_ok(name) "("
  if (inclass!="") sig = sig (recv=="" ? "" : (recv (pl=="" ? "" : ", "))) pl
  else             sig = sig pl
  sig = sig ")"
  if (ret!="" && ret!="()") sig = sig " -> " ret
  OUT = OUT sig " {\n" indent_pfx "    todo!(\"port from " SRCREL ":" line "\")\n" indent_pfx "}\n\n"
  NFUNC++
}

# ---- final assembly: header + banner + decls ------------------------------
function finish(   k, skipsum) {
  if (REPORT=="1") {
    printf "source\t%s\n", SRCREL
    printf "language\t%s\n", (MODE=="c" ? "C" : "Python")
    printf "found.functions\t%d\n", NFUNC
    printf "found.structs\t%d\n", NSTRUCT
    printf "skipped.total\t%d\n", NSKIP
    n = asorti(SKIP, order)
    for (k=1;k<=n;k++) printf "skipped.%s\t%d\n", order[k], SKIP[order[k]]
    if (NFIELDSKIP>0) printf "skipped.struct-fields\t%d\n", NFIELDSKIP
    exit 0
  }
  # header
  print HEADER
  # review banner (§ no-silent-caps)
  print "// SKELETON-GEN: review — heuristic extraction (regex/brace, not a real parser)."
  print "// Multi-line macros, fn-pointer types and unusual decls may be missed or"
  print "// mis-typed; every signature and `todo!()` body needs a human pass."
  skipsum=""
  n = asorti(SKIP, order)
  for (k=1;k<=n;k++) skipsum = skipsum (skipsum=="" ? "" : ", ") order[k] ":" SKIP[order[k]]
  printf "// SKELETON-GEN: %d fn / %d struct emitted; %d decl(s) skipped%s.\n", \
         NFUNC, NSTRUCT, NSKIP, (skipsum=="" ? "" : (" [" skipsum "]"))
  print "// SKELETON-GEN: run `scripts/skeleton-gen.sh --report` for the full breakdown.\n"
  printf "%s", OUT
}
AWK_EOF

# ----- build the provenance header (bash side, then hand to awk) -----------
if [ "$kind" = mupdf ]; then
  if [ -n "$hdrrel" ]; then
    src_cite="MuPDF \`$srcrel\` + \`$hdrrel\`"
  else
    src_cite="MuPDF \`$srcrel\`"
  fi
  header="//! Ported from $src_cite
//! (commit $commit, AGPL-3.0, © Artifex Software, Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the algorithms and numeric
//! behaviour follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md (\"PDF & document-extraction references\").
//!
//! Generated by \`scripts/skeleton-gen.sh\` (Task III-3) — SIGNATURE STUBS ONLY."
  bread="// MuPDF:"
elif [ "$kind" = pdf2md ]; then
  header="//! Ported from the vendored MIT reference
//! \`crates/kopitiam-document/vendor/$srcrel\`,
//! translated to Rust for KOPITIAM (AGPL-3.0-only). Per the port conventions in
//! \`docs/ai-decisions/AID-0051-mupdf-port-conventions.md\`: observable behaviour
//! follows the Python reference; the code is re-expressed idiomatically.
//!
//! Generated by \`scripts/skeleton-gen.sh\` (Task III-3) — SIGNATURE STUBS ONLY."
  bread="// pdf2md:"
else
  header="//! Ported from \`$srcrel\` (commit $commit), translated to Rust for KOPITIAM
//! (AGPL-3.0-only). Verify the upstream licence is AGPL-compatible before use.
//!
//! Generated by \`scripts/skeleton-gen.sh\` (Task III-3) — SIGNATURE STUBS ONLY."
  bread="// src:"
fi

report_flag=0
[ "$mode" = report ] && report_flag=1

gawk \
  -v MODE="$lang" \
  -v REPORT="$report_flag" \
  -v SRCREL="$srcrel" \
  -v SRCBASE="$srcbase" \
  -v STEM="$stem" \
  -v HEADER="$header" \
  -v BREAD_PFX="$bread " \
  "$AWK" "$src"
