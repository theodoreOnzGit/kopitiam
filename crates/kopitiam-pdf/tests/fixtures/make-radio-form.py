#!/usr/bin/env python3
"""Generate `radio-form.pdf` -- a radio/checkbox widget fixture.

Built to answer one question with evidence instead of a guess: the maintainer
reports radio buttons are **invisible in kpdf** (though clicking them does
change the value -- Okular shows the change). So the value path works and the
*drawing* path does not.

The suspect is `annot_run::select_ap_state`: a widget's `/AP` `/N` is a
dictionary of appearance states keyed by `/AS`, and when `/AS` is missing that
function only picks a state if the dict has exactly ONE entry. A radio always
has at least two (`/Off` plus an on-state), so a widget with no `/AS` is
skipped and draws nothing.

Two widgets, differing ONLY in whether `/AS` is present, isolate that exactly:

  radio A  -- /AS /Off present   -> should draw its ring
  radio B  -- /AS ABSENT         -> the suspect case
  checkbox -- /AS /Off present   -> control, same shape as A

Every state stream draws a visibly dark ring/box, so "nothing drawn" cannot be
confused with "drew an empty Off appearance".

Regenerate with:  python3 make-radio-form.py
"""

import pathlib

# Appearance streams. Each draws a thick dark ring/box so it is unmistakable in
# a raster, including in the Off state -- real viewers show an unselected radio
# as an empty circle, not as nothing at all.
RING_OFF = b"0 0 0 RG 1.5 w 2 2 16 16 re S\n"
RING_ON = b"0 0 0 RG 1.5 w 2 2 16 16 re S\n0 0 0 rg 6 6 8 8 re f\n"

OBJECTS = {
    1: b"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [10 0 R 20 0 R] "
       b"/DA (/Helv 0 Tf 0 g) >> >>",
    2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << >> "
       b"/Contents 4 0 R /Annots [11 0 R 12 0 R 21 0 R] >>",
    4: None,  # page content

    # --- radio group: parent field + two kid widgets ---------------------
    # /Ff bit 16 (1<<15) = Radio.
    10: b"<< /FT /Btn /Ff 32768 /T (choice) /V /Off /Kids [11 0 R 12 0 R] >>",
    # Radio A: /AS PRESENT.
    11: b"<< /Type /Annot /Subtype /Widget /Parent 10 0 R /F 4 "
        b"/Rect [20 150 40 170] /AS /Off "
        b"/MK << /BC [0 0 0] >> "
        b"/AP << /N << /Off 30 0 R /A 31 0 R >> >> >>",
    # Radio B: /AS ABSENT -- the suspect.
    12: b"<< /Type /Annot /Subtype /Widget /Parent 10 0 R /F 4 "
        b"/Rect [20 110 40 130] "
        b"/MK << /BC [0 0 0] >> "
        b"/AP << /N << /Off 30 0 R /B 31 0 R >> >> >>",

    # --- checkbox control, /AS present ------------------------------------
    20: b"<< /FT /Btn /T (agree) /V /Off /Kids [21 0 R] >>",
    21: b"<< /Type /Annot /Subtype /Widget /Parent 20 0 R /F 4 "
        b"/Rect [20 60 40 80] /AS /Off "
        b"/MK << /BC [0 0 0] >> "
        b"/AP << /N << /Off 30 0 R /Yes 31 0 R >> >> >>",

    30: None,  # shared Off appearance
    31: None,  # shared On appearance
}

CONTENT = b"0 0 0 RG 1 w 10 190 m 190 190 l S\n"


def form_xobject(data: bytes) -> bytes:
    return (b"<< /Type /XObject /Subtype /Form /BBox [0 0 20 20] "
            b"/Resources << >> /Length %d >>\nstream\n" % len(data)
            + data + b"\nendstream")


def build() -> bytes:
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for num in sorted(OBJECTS):
        offsets[num] = len(out)
        if num == 4:
            body = b"<< /Length %d >>\nstream\n" % len(CONTENT) + CONTENT + b"\nendstream"
        elif num == 30:
            body = form_xobject(RING_OFF)
        elif num == 31:
            body = form_xobject(RING_ON)
        else:
            body = OBJECTS[num]
        out += b"%d 0 obj\n" % num + body + b"\nendobj\n"

    xref_at = len(out)
    nums = sorted(OBJECTS)
    out += b"xref\n"
    # Contiguous runs need their own subsection headers.
    run = []
    for n in nums + [None]:
        if run and (n is None or n != run[-1] + 1):
            out += b"%d %d\n" % (run[0], len(run))
            for m in run:
                out += b"%010d 00000 n \n" % offsets[m]
            run = []
        if n is not None:
            run.append(n)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (max(nums) + 1, xref_at)
    return bytes(out)


if __name__ == "__main__":
    path = pathlib.Path(__file__).with_name("radio-form.pdf")
    data = build()
    path.write_bytes(data)
    print(f"wrote {path} ({len(data)} bytes)")
