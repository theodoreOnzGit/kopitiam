#!/usr/bin/env python3
"""Generate `dct-colour.pdf` + `dct-colour.mutool.ppm` -- the DCTDecode colour fixture.

Why: gh-116 -- a CMYK DCTDecode figure (4 components, Adobe APP14
marker with transform 0, CMYK stored PLAIN, no /Decode) rendered as a solid
BLACK rectangle in kopitiam-pdf 0.4.2, where MuPDF paints the figure. zune-jpeg
converts 4-component JPEGs to RGB itself, assuming Adobe-INVERTED storage
(r = c * k / 255), so plain CMYK white (0,0,0,0) came out black. MuPDF instead
takes libjpeg's raw CMYK (it never inverts inside a PDF: pdf-stream.c:164 sets
invert_cmyk = 0), applies /Decode (image.c:726, fz_decode_tile), then converts.

One 96 x 16 pt page, six 16 x 16 px JPEG images side by side, drawn 1:1 at
72 dpi (no scaling). Each image has four 8 x 8 quadrants (one JPEG block each,
quality 100, no chroma subsampling, so the decode is close to exact):

  x   0..16  CMYK, plain storage, Adobe transform 0, no /Decode  (the bug's shape)
  x  16..32  CMYK, Adobe-inverted storage (Photoshop/Pillow), /Decode [1 0 1 0 1 0 1 0]
  x  32..48  YCCK (Adobe transform 2, component ids 1..4), inverted, /Decode [1 0 ...]
  x  48..64  CMYK, plain storage, /Decode [1 0 0 1 0 1 0 0.5] (C inverted, K halved:
             fz_decode_tile's per-component arithmetic, draw-unpack.c:350)
  x  64..80  3-component, samples stored as RGB (no colour transform), JFIF,
             /DecodeParms << /ColorTransform 0 >>  (filter-dct.c:255-279)
  x  80..96  grayscale, /Decode [1 0]

CMYK quadrants (true ink values): white (0,0,0,0), cyan (255,0,0,0),
(0,128,255,0), 50 % black (0,0,0,128). RGB quadrants: red, green, blue, yellow.
Gray quadrants: 0, 85, 170, 255.

The oracle is MuPDF itself: `dct-colour.mutool.ppm` is
    mutool draw -N -M 0 -r 72 -c rgb -o dct-colour.mutool.ppm dct-colour.pdf
at the pinned commit 19f1284 (mutool 1.29.0), and tests/mupdf_parity.rs
compares every pixel to it. Everything here is synthetic -- this script is
the only source of these bytes.

Regenerate with:
    python3 make-dct-colour.py [--mutool PATH/TO/mutool]
(needs Pillow; without --mutool only the PDF is written).
"""

import io
import pathlib
import subprocess
import sys

from PIL import Image

HERE = pathlib.Path(__file__).parent

CMYK_QUADS = [(0, 0, 0, 0), (255, 0, 0, 0), (0, 128, 255, 0), (0, 0, 0, 128)]
RGB_QUADS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
GRAY_QUADS = [0, 85, 170, 255]


def quads(mode, values):
    im = Image.new(mode, (16, 16))
    for i, v in enumerate(values):
        x0, y0 = (i % 2) * 8, (i // 2) * 8
        for y in range(y0, y0 + 8):
            for x in range(x0, x0 + 8):
                im.putpixel((x, y), v)
    return im


def jpeg(im):
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=100, subsampling=0)
    return buf.getvalue()


def inv(t):
    return tuple(255 - v for v in t)


# Pillow writes CMYK JPEGs Adobe-style: it stores 255 - v (JpegImagePlugin
# rawmode "CMYK;I") with an APP14 "Adobe" marker, transform 0, component ids
# 'C','M','Y','K'. So handing it inv(v) stores v itself, i.e. PLAIN CMYK.
def cmyk_plain():
    return jpeg(quads("CMYK", [inv(q) for q in CMYK_QUADS]))


def cmyk_inverted():
    return jpeg(quads("CMYK", CMYK_QUADS))  # stored 255 - v, the Photoshop way


def ycck_inverted():
    """A YCCK JPEG whose decoded CMYK is the INVERTED ink (so /Decode [1 0..]
    restores it), built from Pillow's CMYK writer: store libjpeg's
    cmyk_ycck_convert (jccolor.c) of the inverted ink as if it were plain CMYK,
    then relabel the colour transform (APP14 byte -> 2) and the component ids
    (SOF / SOS -> 1,2,3,4) so libjpeg and zune-jpeg both read YCCK."""
    stored = []
    for c, m, y, k in (inv(q) for q in CMYK_QUADS):
        r, g, b = 255 - c, 255 - m, 255 - y
        yy = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168735892 * r - 0.331264108 * g + 0.5 * b + 128
        cr = 0.5 * r - 0.418687589 * g - 0.081312411 * b + 128
        v = tuple(max(0, min(255, round(t))) for t in (yy, cb, cr)) + (k,)
        stored.append(inv(v))  # Pillow inverts on write
    data = bytearray(jpeg(quads("CMYK", stored)))
    i, patched = 2, set()
    while i < len(data):
        assert data[i] == 0xFF
        m = data[i + 1]
        seg_len = int.from_bytes(data[i + 2:i + 4], "big")
        body = i + 4
        if m == 0xEE and data[body:body + 5] == b"Adobe":
            data[body + 11] = 2  # transform: YCCK
            patched.add("app14")
        elif m == 0xC0:
            n = data[body + 5]
            assert n == 4
            for c in range(4):
                data[body + 6 + 3 * c] = c + 1
            patched.add("sof")
        elif m == 0xDA:
            n = data[body]
            for c in range(n):
                data[body + 1 + 2 * c] = c + 1
            patched.add("sos")
            break
        i += 2 + seg_len
    assert patched == {"app14", "sof", "sos"}, patched
    return bytes(data)


def rgb_untransformed():
    # Mode "YCbCr" is written without a colour conversion (in_color_space
    # JCS_YCbCr), so channels given as R,G,B are stored as-is.
    return jpeg(quads("YCbCr", RGB_QUADS))


def gray():
    return jpeg(quads("L", GRAY_QUADS))


IMAGES = [
    (cmyk_plain(), "/ColorSpace /DeviceCMYK"),
    (cmyk_inverted(), "/ColorSpace /DeviceCMYK /Decode [1 0 1 0 1 0 1 0]"),
    (ycck_inverted(), "/ColorSpace /DeviceCMYK /Decode [1 0 1 0 1 0 1 0]"),
    (cmyk_plain(), "/ColorSpace /DeviceCMYK /Decode [1 0 0 1 0 1 0 0.5]"),
    (rgb_untransformed(), "/ColorSpace /DeviceRGB /DecodeParms << /ColorTransform 0 >>"),
    (gray(), "/ColorSpace /DeviceGray /Decode [1 0]"),
]


def build():
    n = len(IMAGES)
    content = " ".join(f"q 16 0 0 16 {16 * i} 0 cm /Im{i} Do Q" for i in range(n)).encode()
    xobjs = " ".join(f"/Im{i} {5 + i} 0 R" for i in range(n))
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {16 * n} 16] "
         f"/Resources << /XObject << {xobjs} >> >> /Contents 4 0 R >>").encode(),
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
    ]
    for data, extra in IMAGES:
        objs.append(
            (f"<< /Type /XObject /Subtype /Image /Width 16 /Height 16 "
             f"/BitsPerComponent 8 /Filter /DCTDecode {extra} /Length {len(data)} >>\nstream\n").encode()
            + data + b"\nendstream")
    out = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offs = []
    for i, body in enumerate(objs):
        offs.append(len(out))
        out += b"%d 0 obj\n" % (i + 1) + body + b"\nendobj\n"
    x = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, x)
    return out


if __name__ == "__main__":
    pdf = HERE / "dct-colour.pdf"
    pdf.write_bytes(build())
    print("wrote", pdf)
    if "--mutool" in sys.argv:
        mutool = sys.argv[sys.argv.index("--mutool") + 1]
        ppm = HERE / "dct-colour.mutool.ppm"
        subprocess.run([mutool, "draw", "-N", "-M", "0", "-r", "72", "-c", "rgb",
                        "-o", str(ppm), str(pdf)], check=True)
        print("wrote", ppm)
