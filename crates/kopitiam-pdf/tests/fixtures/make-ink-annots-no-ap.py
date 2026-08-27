#!/usr/bin/env python3
"""Generate `ink-annots-no-ap.pdf` -- the regression fixture for gh-79.

Why synthetic and not a real-world file: the bug was found on a maintainer's
own 51-page document, which does not belong in a public repository. This
reproduces the *exact* shape that triggers it, in ~1.5 KB, with coordinates a
test can assert on:

  * `/Ink` annotations carrying `/InkList`, `/C`, `/Border` and `/CA`
  * **no `/AP` anywhere in the file** -- the whole point. Okular writes ink
    annots as pure data and expects the viewer to synthesise the appearance.
    A renderer that draws strictly from `/AP` shows nothing.
  * a classic cross-reference table (no xref streams, no object streams),
    matching the observed producer output.

Regenerate with:  python3 make-ink-annots-no-ap.py
"""

import pathlib

# Page is 200x200pt so the whole thing rasterises small and fast.
# Annot 1: a horizontal-ish stroke, solidly inside the page, blue.
# Annot 2: a near-VERTICAL stroke -- deliberately degenerate, because its /Rect
#          is barely wider than a hairline. That is the case where a stroke
#          centred on the polyline falls outside a tight /Rect, so it catches a
#          renderer that forgets to widen the box (MuPDF expands by lw + 6).
# Annot 3: hidden via /F 2 -- MUST stay invisible. Guards against a fix that
#          simply draws everything in /Annots regardless of flags.
OBJECTS = {
    1: b"<< /Type /Catalog /Pages 2 0 R >>",
    2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
       b"/Resources << >> /Contents 4 0 R /Annots [5 0 R 6 0 R 7 0 R] >>",
    # A single black rule, so the page is not blank even with zero annots --
    # lets a test tell "nothing rendered at all" from "annots missing".
    4: None,  # stream, built below
    5: b"<< /Type /Annot /Subtype /Ink /Rect [20 150 120 170] "
       b"/InkList [[20 150 60 170 100 155 120 165]] "
       b"/C [0 0 1] /CA 1 /Border [0 0 2] /F 4 >>",
    6: b"<< /Type /Annot /Subtype /Ink /Rect [150 40 151 160] "
       b"/InkList [[150 40 151 100 150 160]] "
       b"/C [1 0 0] /CA 1 /Border [0 0 3] /F 4 >>",
    7: b"<< /Type /Annot /Subtype /Ink /Rect [20 40 120 60] "
       b"/InkList [[20 40 120 60]] "
       b"/C [0 1 0] /CA 1 /Border [0 0 2] /F 2 >>",
}

CONTENT = b"0 0 0 RG 1 w 10 190 m 190 190 l S\n"


def build() -> bytes:
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for num in sorted(OBJECTS):
        offsets[num] = len(out)
        if num == 4:
            body = b"<< /Length %d >>\nstream\n" % len(CONTENT) + CONTENT + b"endstream"
        else:
            body = OBJECTS[num]
        out += b"%d 0 obj\n" % num + body + b"\nendobj\n"

    xref_at = len(out)
    n = max(OBJECTS) + 1
    out += b"xref\n0 %d\n" % n
    out += b"0000000000 65535 f \n"
    for num in sorted(OBJECTS):
        out += b"%010d 00000 n \n" % offsets[num]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (n, xref_at)
    return bytes(out)


if __name__ == "__main__":
    path = pathlib.Path(__file__).with_name("ink-annots-no-ap.pdf")
    data = build()
    path.write_bytes(data)
    assert b"/AP" not in data, "fixture must contain no appearance stream"
    print(f"wrote {path} ({len(data)} bytes)")


# ---------------------------------------------------------------------------
# Second fixture: `ink-annots-mixed-ap.pdf`
# ---------------------------------------------------------------------------
# Exercises the *other* half of annotation rendering -- consuming a real /AP
# form XObject (PDF 32000-1 s12.5.5's BBox->Rect mapping) -- and the
# AnnotPass::SynthesizedOnly filter that stops the hayro fallback path from
# double-drawing.
#
#   magenta: HAS /AP  -> drawn by any /AP-capable engine, including hayro
#   cyan   : NO  /AP  -> only ever appears if the appearance is synthesised
#
# The /AP form deliberately uses a BBox that is NOT equal to /Rect, so the
# Algorithm 8.1 scale+translate is actually exercised rather than reducing to
# the identity: BBox is [0 0 50 10] but Rect is [20 150 120 170], i.e. a 2x
# scale in x, 2x in y, plus a translation.

MIXED_OBJECTS = {
    1: b"<< /Type /Catalog /Pages 2 0 R >>",
    2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
       b"/Resources << >> /Contents 4 0 R /Annots [5 0 R 6 0 R] >>",
    4: None,   # page content stream
    5: b"<< /Type /Annot /Subtype /Ink /Rect [20 150 120 170] "
       b"/InkList [[20 150 120 170]] /C [1 0 1] /CA 1 /Border [0 0 2] /F 4 "
       b"/AP << /N 7 0 R >> >>",
    6: b"<< /Type /Annot /Subtype /Ink /Rect [20 40 120 60] "
       b"/InkList [[20 40 60 60 100 45 120 55]] "
       b"/C [0 1 1] /CA 1 /Border [0 0 3] /F 4 >>",
    7: None,   # the /AP form XObject
}

MIXED_CONTENT = b"0 0 0 RG 1 w 10 190 m 190 190 l S\n"
# Magenta stroke in FORM space (BBox 0 0 50 10), mapped onto Rect by the viewer.
AP_CONTENT = b"1 0 1 RG 2 w 1 1 m 49 9 l S\n"


def build_mixed() -> bytes:
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for num in sorted(MIXED_OBJECTS):
        offsets[num] = len(out)
        if num == 4:
            body = b"<< /Length %d >>\nstream\n" % len(MIXED_CONTENT) + MIXED_CONTENT + b"endstream"
        elif num == 7:
            body = (b"<< /Type /XObject /Subtype /Form /BBox [0 0 50 10] "
                    b"/Resources << >> /Length %d >>\nstream\n" % len(AP_CONTENT)
                    + AP_CONTENT + b"endstream")
        else:
            body = MIXED_OBJECTS[num]
        out += b"%d 0 obj\n" % num + body + b"\nendobj\n"

    xref_at = len(out)
    n = max(MIXED_OBJECTS) + 1
    out += b"xref\n0 %d\n" % n
    out += b"0000000000 65535 f \n"
    for num in sorted(MIXED_OBJECTS):
        out += b"%010d 00000 n \n" % offsets[num]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (n, xref_at)
    return bytes(out)


if __name__ == "__main__":
    p2 = pathlib.Path(__file__).with_name("ink-annots-mixed-ap.pdf")
    d2 = build_mixed()
    p2.write_bytes(d2)
    assert d2.count(b"/AP") == 1, "mixed fixture must have exactly one /AP annot"
    print(f"wrote {p2} ({len(d2)} bytes)")
