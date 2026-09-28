#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Write the SYNTHETIC feature corpus for the kopitiam-pdf <-> MuPDF
code-to-code harness (crates/kopitiam-pdf/examples/mupdf_oracle.rs).

Why this exists, hor: after the 0.4.2 image tranche every page of the open
corpus (NRC reports + PHYSOR papers + the arXiv paper) rendered within 1 % of
MuPDF -- yet the coverage map (docs/mupdf-port-coverage.md) still lists
shadings, `gs` alpha, dashes, stencil masks, inline images, patterns, Type3,
... as missing. Those features just do not occur (or occur too small) in that
corpus, so the corpus stopped being able to fail. These files each exercise
ONE feature over a big area, so a missing feature fails the raster gate loudly
and a fixed one passes it.

Every file is generated here from nothing but this script -- no third-party
content, nothing to license. Output is deterministic.

    python3 scripts/mupdf-feature-corpus.py OUTDIR
    target/release/examples/mupdf_oracle --mutool MUTOOL --no-objects OUTDIR/*.pdf
"""

import pathlib
import sys


def pdf(objs, page_dict_extra=b"", resources=b"<< >>", content=b"", mediabox=b"[0 0 200 200]",
        catalog=b"<< /Type /Catalog /Pages 2 0 R >>"):
    """Objects 1..3 are catalog/pages/page, 4 is the content stream; `objs` are
    appended from 5 on (bytes bodies)."""
    body = [
        catalog,
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox " + mediabox + b" /Resources " + resources
        + b" /Contents 4 0 R" + page_dict_extra + b" >>",
        stream(content),
    ] + list(objs)
    out = b"%PDF-1.7\n"
    offs = []
    for i, o in enumerate(body):
        offs.append(len(out))
        out += b"%d 0 obj\n" % (i + 1) + o + b"\nendobj\n"
    x = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(body) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(body) + 1, x)
    return out


def stream(data, extra=b""):
    return b"<< /Length %d" % len(data) + extra + b" >>\nstream\n" + data + b"\nendstream"


FILES = {}

# 1. Line style: dash (d), caps (J), joins (j), miter limit (M).
FILES["line-style"] = pdf([], content=b"""
0 0 1 RG 12 w
0 J [30 15] 0 d 20 170 m 180 170 l S
1 J [] 0 d 20 140 m 180 140 l S
2 J 20 110 m 180 110 l S
0 j 10 M 30 20 m 60 90 l 90 20 l S
1 j 90 20 m 120 90 l 150 20 l S
2 j 140 20 m 170 90 l 190 20 l S
""")

# 2. ExtGState: constant alpha, line width, dash, via `gs`.
FILES["extgstate"] = pdf(
    [],
    resources=b"<< /ExtGState << /A << /ca 0.5 /CA 0.5 >> /B << /LW 10 /D [[20 10] 0] /LC 1 >> >> >>",
    content=b"""
1 0 0 rg 20 20 120 120 re f
q /A gs 0 0 1 rg 60 60 120 120 re f Q
q /B gs 0 0.6 0 RG 20 185 m 180 185 l S Q
""")

# 3. Stencil image mask painted in the fill colour over a blue square.
mask = bytes([0xF0, 0x0F] * 16)  # 16x16, 1 bpc: left/right half stripes per row
FILES["image-mask"] = pdf(
    [stream(mask, b" /Type /XObject /Subtype /Image /Width 16 /Height 16 /ImageMask true /BitsPerComponent 1")],
    resources=b"<< /XObject << /M 5 0 R >> >>",
    content=b"0 0 1 rg 0 0 200 200 re f 1 0 0 rg q 200 0 0 200 0 0 cm /M Do Q",
)

# 4. Inline images: a gray ramp and an inline stencil.
ramp = bytes(range(0, 256, 32)) * 8  # 8x8 gray
FILES["inline-image"] = pdf([], content=(
    b"q 200 0 0 100 0 100 cm BI /W 8 /H 8 /CS /G /BPC 8 ID\n" + ramp + b"\nEI Q\n"
    + b"0 0.5 0 rg q 200 0 0 100 0 0 cm BI /W 16 /H 16 /IM true ID\n" + mask + b"\nEI Q"
))

# 5. Shadings via `sh`: axial (type 2) and radial (type 3), exponential functions.
FILES["shading-sh"] = pdf(
    [],
    resources=b"""<< /Shading <<
 /Ax << /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 200 0] /Extend [true true]
        /Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >> >>
 /Ra << /ShadingType 3 /ColorSpace /DeviceRGB /Coords [100 100 0 100 100 60] /Extend [false false]
        /Function << /FunctionType 2 /Domain [0 1] /C0 [1 1 0] /C1 [0 1 0] /N 1 >> >>
>> >>""",
    content=b"q 0 0 200 100 re W n /Ax sh Q q 0 100 200 100 re W n /Ra sh Q",
)

# 6. Shading pattern as a fill colour (scn /P).
FILES["shading-pattern"] = pdf(
    [],
    resources=b"""<< /Pattern << /P << /PatternType 2 /Shading
 << /ShadingType 2 /ColorSpace /DeviceGray /Coords [0 0 0 200]
    /Function << /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [1] /N 1 >> >> >> >> >>""",
    content=b"/Pattern cs /P scn 20 20 160 160 re f",
)

# 7. Non-rectangular clip: a circle (4 beziers), then a full-page fill.
k = 0.5523 * 80
circ = (b"100 180 m %.2f 180 180 %.2f 180 100 c 180 %.2f %.2f 20 100 20 c %.2f 20 20 %.2f 20 100 c 20 %.2f %.2f 180 100 180 c h"
        % (100 + k, 100 + k, 100 - k, 100 + k, 100 - k, 100 - k, 100 + k, 100 - k))
FILES["clip-path"] = pdf([], content=b"q " + circ + b" W n 1 0 0 rg 0 0 200 200 re f Q")

# 8. Type3 font: two glyphs drawn by content procs.
FILES["type3"] = pdf(
    [
        b"""<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1000 1000] /FontMatrix [0.001 0 0 0.001 0 0]
 /CharProcs << /sq 6 0 R /tri 7 0 R >> /Encoding << /Type /Encoding /Differences [65 /sq /tri] >>
 /FirstChar 65 /LastChar 66 /Widths [1000 1000] /Resources << >> >>""",
        stream(b"1000 0 0 0 1000 1000 d1 0 0 1000 1000 re f"),
        stream(b"1000 0 0 0 1000 1000 d1 0 0 m 1000 0 l 500 1000 l f"),
    ],
    resources=b"<< /Font << /T3 5 0 R >> >>",
    content=b"0 0 1 rg BT /T3 60 Tf 20 100 Td (ABAB) Tj ET",
)

# 9. Text render modes: stroke (1), fill+stroke (2), fill+clip (4).
FILES["text-render-modes"] = pdf(
    [b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"],
    resources=b"<< /Font << /F 5 0 R >> >>",
    content=b"""0 0 1 RG 2 w 1 0 0 rg
BT /F 48 Tf 1 Tr 10 140 Td (Stroke) Tj ET
BT /F 48 Tf 2 Tr 10 80 Td (Both) Tj ET
q BT /F 48 Tf 7 Tr 10 20 Td (Clip) Tj ET 0 0.6 0 rg 0 0 200 70 re f Q""",
)

# 10. Colour spaces for fills: Indexed, Separation, Lab, ICCBased-less CalRGB.
FILES["fill-colorspaces"] = pdf(
    [],
    resources=b"""<< /ColorSpace <<
 /I [/Indexed /DeviceRGB 1 <FF000000FF00>]
 /S [/Separation /Spot /DeviceCMYK << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 1 1 0] /N 1 >>]
 /L [/Lab << /WhitePoint [0.9505 1 1.089] /Range [-100 100 -100 100] >>]
 /C [/CalRGB << /WhitePoint [0.9505 1 1.089] >>] >> >>""",
    content=b"""/I cs 1 sc 0 100 100 100 re f
/S cs 1 sc 100 100 100 100 re f
/L cs 50 60 40 sc 0 0 100 100 re f
/C cs 0.2 0.4 0.8 sc 100 0 100 100 re f""",
)

# 11. CropBox smaller than MediaBox (the page must be the crop).
FILES["cropbox"] = pdf(
    [],
    page_dict_extra=b" /CropBox [50 50 250 250]",
    mediabox=b"[0 0 300 300]",
    content=b"1 0 0 rg 0 0 300 300 re f 0 0 1 rg 50 50 100 100 re f",
)

# 12. Tiling pattern (checkerboard).
FILES["tiling-pattern"] = pdf(
    [stream(b"0 0 0 rg 0 0 10 10 re f 10 10 10 10 re f",
            b" /Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 20 20] /XStep 20 /YStep 20 /Resources << >>")],
    resources=b"<< /Pattern << /T 5 0 R >> >>",
    content=b"/Pattern cs /T scn 0 0 200 200 re f",
)

# 13. Form XObject with /BBox (must clip) and /Matrix.
FILES["form-bbox"] = pdf(
    [stream(b"1 0 0 rg -50 -50 300 300 re f",
            b" /Type /XObject /Subtype /Form /BBox [0 0 100 100] /Matrix [1 0 0 1 50 50]")],
    resources=b"<< /XObject << /F 5 0 R >> >>",
    content=b"/F Do",
)

# 14. Optional content: a layer that is OFF must not paint.
FILES["optional-content"] = pdf(
    [b"<< /Type /OCG /Name (Hidden) >>"],
    resources=b"<< /Properties << /L 5 0 R >> >>",
    content=b"0 1 0 rg 0 0 200 200 re f /OC /L BDC 1 0 0 rg 50 50 100 100 re f EMC",
    catalog=b"<< /Type /Catalog /Pages 2 0 R /OCProperties << /OCGs [5 0 R] /D << /OFF [5 0 R] >> >> >>",
)

# 15. Blend mode + soft mask through ExtGState.
FILES["blend-smask"] = pdf(
    [stream(b"0 0 200 200 re 0 g f 1 g 50 50 100 100 re f",
            b" /Type /XObject /Subtype /Form /BBox [0 0 200 200] /Group << /S /Transparency /CS /DeviceGray >>")],
    resources=b"""<< /ExtGState << /M << /BM /Multiply >>
 /S << /SMask << /Type /Mask /S /Luminosity /G 5 0 R >> >> >> >>""",
    content=b"""0 1 1 rg 0 0 200 100 re f q /M gs 1 1 0 rg 50 0 100 100 re f Q
q /S gs 1 0 0 rg 0 100 200 100 re f Q""",
)

# 16. CMYK fill (no ICC in our port; MuPDF uses lcms2 + its default profile).
FILES["cmyk-fill"] = pdf([], content=b"1 0 0 0 k 0 0 100 200 re f 0 0.5 1 0 k 100 0 100 200 re f")

# 17. Stroke adjust / hairlines: 0-width lines are one device pixel in MuPDF.
FILES["hairline"] = pdf([], content=b"0 w " + b" ".join(b"%d 0 m %d 200 l S" % (x, x) for x in range(5, 200, 10)))


# 18. A damaged file: every xref offset is off by 7 bytes (a common result of
# an editor re-saving with CRLF/LF damage). MuPDF repairs it (pdf-repair.c:
# scan for "N G obj") and renders the red square; a port without repair
# refuses to open the file at all.
_good = pdf([], content=b"1 0 0 rg 50 50 100 100 re f")
_x = _good.index(b"xref\n")
_tbl = _good[_x:].split(b"trailer")[0]
_fixed = b"\n".join(
    (b"%010d 00000 n " % (int(l[:10]) + 7)) if l.endswith(b" n ") else l for l in _tbl.split(b"\n")
)
FILES["broken-xref"] = _good[:_x] + _fixed + b"trailer" + _good[_x:].split(b"trailer", 1)[1]


# 19. Function-based shading (type 1) driven by a PostScript calculator
# function (type 4): exercises pdf-function.c's calculator end to end.
FILES["shading-type1-ps"] = pdf(
    [stream(b"{ 2 copy mul 3 1 roll }", b" /FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1]")],
    resources=b"""<< /Shading << /S << /ShadingType 1 /ColorSpace /DeviceRGB /Domain [0 1 0 1]
 /Matrix [200 0 0 200 0 0] /Function 5 0 R >> >> >>""",
    content=b"/S sh",
)

# 20. Free-form triangle mesh (type 4): two triangles, 8-bit flag/coord/comp.
_mesh = bytes([
    0, 0, 0, 255, 0, 0,      # flag 0, (0,0) red
    0, 255, 0, 0, 255, 0,    # (255,0) green
    0, 0, 255, 0, 0, 255,    # (0,255) blue
    1, 255, 255, 255, 255, 0,  # flag 1 -> triangle (b, c, d): yellow at (255,255)
])
FILES["shading-mesh4"] = pdf(
    [stream(_mesh, b" /ShadingType 4 /ColorSpace /DeviceRGB /BitsPerCoordinate 8 /BitsPerComponent 8"
                   b" /BitsPerFlag 8 /Decode [0 200 0 200 0 1 0 1 0 1]")],
    resources=b"<< /Shading << /M 5 0 R >> >>",
    content=b"/M sh",
)

# 21. Radial shading with both extends, in a Separation space whose tint
# transform (type 2) maps to CMYK.
FILES["shading-radial-separation"] = pdf(
    [],
    resources=b"""<< /Shading << /R << /ShadingType 3
 /ColorSpace [/Separation /Spot /DeviceCMYK << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 0.8 0.9 0] /N 1 >>]
 /Coords [100 100 10 100 100 80] /Extend [true true]
 /Function << /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [1] /N 1 >> >> >> >>""",
    content=b"/R sh",
)

# 22. A DeviceN image (2 inks -> RGB via a sampled type 0 tint function) and
# a Lab image: both were "unsupported colorspace" and drew nothing.
_samp = bytes([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255])  # 2x2 grid of RGB
_devn_img = bytes([0, 0, 255, 0, 0, 255, 255, 255])  # 2x2 pixels, 2 comps
_lab_img = bytes([50, 200, 128, 90, 128, 30, 20, 60, 200, 100, 128, 128])  # 2x2 L*a*b*
FILES["image-devicen-lab"] = pdf(
    [
        stream(_devn_img, b" /Type /XObject /Subtype /Image /Width 2 /Height 2 /BitsPerComponent 8"
                          b" /ColorSpace [/DeviceN [/A /B] /DeviceRGB 7 0 R]"),
        stream(_lab_img, b" /Type /XObject /Subtype /Image /Width 2 /Height 2 /BitsPerComponent 8"
                         b" /ColorSpace [/Lab << /WhitePoint [0.9505 1 1.089] /Range [-128 127 -128 127] >>]"),
        stream(_samp, b" /FunctionType 0 /Domain [0 1 0 1] /Range [0 1 0 1 0 1] /Size [2 2] /BitsPerSample 8"),
    ],
    resources=b"<< /XObject << /D 5 0 R /L 6 0 R >> >>",
    content=b"q 100 0 0 200 0 0 cm /D Do Q q 100 0 0 200 100 0 cm /L Do Q",
)


# 23. Tiling patterns beyond the plain case: an UNCOLOURED (PaintType 2)
# pattern under a rotating+scaling /Matrix painted in red through
# [/Pattern /DeviceRGB], and big text filled with a coloured pattern.
# (A pattern whose cell paints with itself is NOT here: MuPDF's own run
# dies with "exception stack overflow" and draws nothing, so there is no
# oracle -- tests/mupdf_parity.rs checks only that ours terminates.)
FILES["tiling-pattern-2"] = pdf(
    [
        stream(b"0 0 6 6 re f 0 0 0 rg 6 6 6 6 re f",  # the `0 0 0 rg` must be ignored
               b" /Type /Pattern /PatternType 1 /PaintType 2 /TilingType 1 /BBox [0 0 12 12]"
               b" /XStep 12 /YStep 12 /Matrix [1.2 0.7 -0.7 1.2 3 5] /Resources << >>"),
        stream(b"0 0 1 rg 0 0 4 8 re f 1 1 0 rg 4 0 4 8 re f",
               b" /Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 8 8]"
               b" /XStep 8 /YStep 8 /Resources << >>"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ],
    resources=b"<< /Pattern << /U 5 0 R /C 6 0 R >> /Font << /F 7 0 R >>"
              b" /ColorSpace << /PU [/Pattern /DeviceRGB] >> >>",
    content=b"""/PU cs 1 0 0 /U scn 0 0 200 100 re f
/Pattern cs /C scn BT /F 70 Tf 10 120 Td (Wa) Tj ET""",
)

# 24. Image /Mask: a colour-key ARRAY on an 8-bit RGB image (white keyed
# out) and on a 4-bit gray one (range 5..9), and an explicit stencil /Mask
# STREAM -- all over a blue background that must show through. 0.4.1 drew
# all three opaque.
_ck_rgb = bytes([255, 255, 255, 255, 0, 0, 0, 255, 0, 255, 255, 255])  # 2x2
_ck_g4 = bytes([0x05, 0x9F, 0xF7, 0x31])  # 4x2 at 4 bpc: 0,5,9,15 / 15,7,3,1
_st_img = bytes([200, 30, 30] * 4)  # 2x2 red-ish
_st_mask = bytes([0x40, 0x80])  # 2x2 at 1 bpc: row0 = 0 1, row1 = 1 0
FILES["image-mask-keys"] = pdf(
    [
        stream(_ck_rgb, b" /Type /XObject /Subtype /Image /Width 2 /Height 2 /BitsPerComponent 8"
                        b" /ColorSpace /DeviceRGB /Mask [250 255 250 255 250 255]"),
        stream(_ck_g4, b" /Type /XObject /Subtype /Image /Width 4 /Height 2 /BitsPerComponent 4"
                       b" /ColorSpace /DeviceGray /Mask [5 9]"),
        stream(_st_img, b" /Type /XObject /Subtype /Image /Width 2 /Height 2 /BitsPerComponent 8"
                        b" /ColorSpace /DeviceRGB /Mask 8 0 R"),
        stream(_st_mask, b" /Type /XObject /Subtype /Image /Width 2 /Height 2 /ImageMask true"),
    ],
    resources=b"<< /XObject << /A 5 0 R /B 6 0 R /C 7 0 R >> >>",
    content=b"0 0 1 rg 0 0 200 200 re f q 100 0 0 100 0 100 cm /A Do Q"
            b" q 200 0 0 100 0 0 cm /B Do Q q 100 0 0 100 100 100 cm /C Do Q",
)


# --- Transparency tranche (blend modes, soft masks, groups). A backdrop of
# four mid-tone stripes so every blend function sees non-trivial input.
_stripes = (b"1 0.2 0.2 rg 0 0 50 200 re f 0.2 0.8 0.3 rg 50 0 50 200 re f"
            b" 0.3 0.4 0.9 rg 100 0 50 200 re f 0.9 0.9 0.5 rg 150 0 50 200 re f\n")

# 25. The 11 separable blend modes (+ Normal at ca 0.5) as horizontal bands
# across the stripes, via ExtGState /BM. 0.4.1 ignored /BM: every band was
# plain paint.
_sep = [b"Multiply", b"Screen", b"Overlay", b"Darken", b"Lighten", b"ColorDodge",
        b"ColorBurn", b"HardLight", b"SoftLight", b"Difference", b"Exclusion", b"Normal"]
FILES["blend-separable"] = pdf(
    [],
    resources=b"<< /ExtGState << " + b" ".join(
        b"/B%d << /BM /%s%s >>" % (i, m, b" /ca 0.5" if m == b"Normal" else b"")
        for i, m in enumerate(_sep)) + b" >> >>",
    content=_stripes + b"".join(
        b"q /B%d gs 0.6 0.3 0.8 rg 0 %d 200 14 re f Q\n" % (i, 200 - 16 * (i + 1))
        for i in range(len(_sep))),
)

# 26. The 4 non-separable modes, each band with two source colours.
_nonsep = [b"Hue", b"Saturation", b"Color", b"Luminosity"]
FILES["blend-nonseparable"] = pdf(
    [],
    resources=b"<< /ExtGState << " + b" ".join(
        b"/B%d << /BM /%s >>" % (i, m) for i, m in enumerate(_nonsep)) + b" >> >>",
    content=_stripes + b"".join(
        b"q /B%d gs 0.9 0.2 0.5 rg 0 %d 100 40 re f 0.2 0.6 0.9 rg 100 %d 100 40 re f Q\n"
        % (i, 200 - 50 * (i + 1) + 5, 200 - 50 * (i + 1) + 5) for i in range(len(_nonsep))),
)

# 27. Luminosity soft mask with a /BC backdrop: the mask group only paints
# inside its /BBox [20 20 180 180]; outside it the mask is the /BC colour's
# gray (RGB 0.2 0.5 0.8 -> 0.443), inside, gray and coloured rects.
FILES["smask-luminosity-bc"] = pdf(
    [stream(b"0.2 g 20 20 50 160 re f 1 0 0 rg 70 20 50 160 re f 0 0 1 rg 120 20 60 80 re f"
            b" 0.8 g 120 100 60 80 re f",
            b" /Type /XObject /Subtype /Form /BBox [20 20 180 180]"
            b" /Group << /S /Transparency /CS /DeviceRGB >>")],
    resources=b"<< /ExtGState << /S << /SMask << /Type /Mask /S /Luminosity /G 5 0 R"
              b" /BC [0.2 0.5 0.8] >> >> >> >>",
    content=b"0 0 1 rg 0 0 200 200 re f q /S gs 1 0 0 rg 0 0 200 200 re f Q",
)

# 28. Alpha soft mask: the mask is the ALPHA of what its group draws (an
# opaque rect, a ca 0.5 rect, overlapping), not its colour.
FILES["smask-alpha"] = pdf(
    [stream(b"0 0 1 rg 10 10 90 180 re f /H gs 1 1 0 rg 60 60 130 80 re f",
            b" /Type /XObject /Subtype /Form /BBox [0 0 200 200]"
            b" /Group << /S /Transparency >> /Resources << /ExtGState << /H << /ca 0.5 >> >> >>")],
    resources=b"<< /ExtGState << /S << /SMask << /Type /Mask /S /Alpha /G 5 0 R >> >> >> >>",
    content=_stripes + b"q /S gs 0 0.5 0 rg 0 0 200 200 re f Q",
)

# 29. A soft-mask /TR transfer function (1 - x, type 2): the top half is
# painted by the first object after the gs, the bottom half by a second
# one. MuPDF drops the /TR from the gstate once it has been used, so only
# the top half is inverted -- the port replicates that quirk.
FILES["smask-tr"] = pdf(
    [stream(b"0.25 g 0 0 100 200 re f 0.75 g 100 0 100 200 re f",
            b" /Type /XObject /Subtype /Form /BBox [0 0 200 200] /Group << /S /Transparency >>")],
    resources=b"<< /ExtGState << /S << /SMask << /Type /Mask /S /Luminosity /G 5 0 R"
              b" /TR << /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >> >> >> >> >>",
    content=b"1 1 1 rg 0 0 200 200 re f q /S gs 0 0 0 rg 0 100 200 100 re f 0 0 0 rg 0 0 200 100 re f Q",
)

# 30. Transparency groups: the same Form, whose content paints a white
# /Difference rect, drawn NON-isolated (top left: inverts the page stripes
# under it), isolated (/I true, top right: differences against nothing, so
# stays white), both at group alpha ca 0.6 on the Do; plus a plain Normal
# group at ca 0.5 (bottom).
_grp = b"/M gs 1 1 1 rg 10 10 80 80 re f 0 0 0 rg 30 30 40 40 re f"
_grp_res = b" /Resources << /ExtGState << /M << /BM /Difference >> >> >>"
FILES["group-isolation"] = pdf(
    [stream(_grp, b" /Type /XObject /Subtype /Form /BBox [0 0 100 100]"
                  b" /Group << /S /Transparency >>" + _grp_res),
     stream(_grp, b" /Type /XObject /Subtype /Form /BBox [0 0 100 100]"
                  b" /Group << /S /Transparency /I true >>" + _grp_res),
     stream(b"1 0 1 rg 0 0 100 100 re f 0 0 0 rg 25 25 50 50 re f",
            b" /Type /XObject /Subtype /Form /BBox [0 0 100 100] /Group << /S /Transparency >>")],
    resources=b"<< /XObject << /N 5 0 R /I 6 0 R /P 7 0 R >>"
              b" /ExtGState << /GA << /ca 0.6 >> /HA << /ca 0.5 >> >> >>",
    content=_stripes + b"q /GA gs 1 0 0 1 0 100 cm /N Do Q q /GA gs 1 0 0 1 100 100 cm /I Do Q"
            b" q /HA gs 1 0 0 1 50 0 cm /P Do Q",
)

# 31. Images and transparency: an image under a luminosity soft mask from
# the gstate (top), and an image carrying its own /SMask under /BM
# /Multiply (bottom) -- which MuPDF gives a blend group but no gstate mask.
_img = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0])  # 2x2 RGB
_img_sm = bytes([255, 128, 64, 0])  # 2x2 gray alpha
FILES["image-transparency"] = pdf(
    [stream(b"0.2 g 0 0 100 200 re f 0.9 g 100 0 100 200 re f",
            b" /Type /XObject /Subtype /Form /BBox [0 0 200 200] /Group << /S /Transparency >>"),
     stream(_img, b" /Type /XObject /Subtype /Image /Width 2 /Height 2 /BitsPerComponent 8"
                  b" /ColorSpace /DeviceRGB"),
     stream(_img, b" /Type /XObject /Subtype /Image /Width 2 /Height 2 /BitsPerComponent 8"
                  b" /ColorSpace /DeviceRGB /SMask 8 0 R"),
     stream(_img_sm, b" /Type /XObject /Subtype /Image /Width 2 /Height 2 /BitsPerComponent 8"
                     b" /ColorSpace /DeviceGray")],
    resources=b"<< /XObject << /A 6 0 R /B 7 0 R >> /ExtGState << /S << /SMask << /Type /Mask"
              b" /S /Luminosity /G 5 0 R >> >> /M << /BM /Multiply /SMask /None >> >> >>",
    content=_stripes + b"q /S gs 200 0 0 100 0 100 cm /A Do Q q /M gs 200 0 0 100 0 0 cm /B Do Q",
)

# 32. Text under a blend mode (/Difference, top) and under an alpha soft
# mask that covers only the left half of the page (bottom).
FILES["text-transparency"] = pdf(
    [b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
     stream(b"0 0 0 rg 0 0 100 200 re f",
            b" /Type /XObject /Subtype /Form /BBox [0 0 200 200] /Group << /S /Transparency >>")],
    resources=b"<< /Font << /F 5 0 R >> /ExtGState << /D << /BM /Difference >>"
              b" /S << /SMask << /Type /Mask /S /Alpha /G 6 0 R >> >> >> >>",
    content=_stripes + b"q /D gs 1 1 1 rg BT /F 64 Tf 5 120 Td (Diff) Tj ET Q"
            b" q /S gs 0 0 0 rg BT /F 64 Tf 5 30 Td (Mask) Tj ET Q",
)

# 33. A tiling-pattern fill at ca 0.5: pdf_show_path draws the pattern
# inside a Normal transparency group at the fill alpha. 0.4.1 drew it
# opaque (the cell copies the PARENT state's alpha, which is 1).
FILES["pattern-alpha"] = pdf(
    [stream(b"0 0 1 rg 0 0 10 10 re f 1 1 0 rg 10 10 10 10 re f",
            b" /Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 20 20]"
            b" /XStep 20 /YStep 20 /Resources << >>")],
    resources=b"<< /Pattern << /P 5 0 R >> /ExtGState << /H << /ca 0.5 >> >> >>",
    content=_stripes + b"q /H gs /Pattern cs /P scn 20 20 160 160 re f Q",
)

def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "feature-corpus")
    out.mkdir(parents=True, exist_ok=True)
    for name, data in FILES.items():
        (out / f"feat-{name}.pdf").write_bytes(data)
    print(f"wrote {len(FILES)} files to {out}")


if __name__ == "__main__":
    main()
