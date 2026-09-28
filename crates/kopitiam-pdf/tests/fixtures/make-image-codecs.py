#!/usr/bin/env python3
"""Generate the two tiny image-codec payloads `tests/mupdf_parity.rs` embeds.

Before kopitiam-pdf 0.4.2, `JPXDecode` and `JBIG2Decode` images were
"deferred" -- the draw device got an Unsupported error and painted nothing,
where MuPDF (openjpeg / jbig2dec) paints the picture. The code-to-code harness
measured it: 211 JPX figures blank on NUREG/CR-7289. These fixtures pin the
fix with bytes small enough to reason about.

* `jpx-8x8-red-blue.j2k` -- an 8x8 RGB JPEG 2000 codestream, left half pure
  red, right half pure blue, LOSSLESS (reversible 5/3 wavelet), so the decoded
  samples are exact and a test can assert them.
* `jbig2-16x8-left-black.jb2` -- an EMBEDDED JBIG2 stream (no file header,
  exactly what a PDF `/JBIG2Decode` stream holds): a page-information segment
  plus one immediate lossless generic region, 16x8, MMR-coded (T.6 / CCITT
  G4 inside JBIG2, T.88 6.2.6), left 8 columns black, right 8 white. The G4
  bits come from libtiff via Pillow; the JBIG2 segment headers are written by
  hand below per ITU-T T.88 section 7.

Both are synthetic (this script is their only source) and were cross-checked
by rendering a PDF wrapping each with `mutool draw` at the pinned MuPDF
commit -- MuPDF paints red|blue and black|white respectively.

Regenerate with:  python3 make-image-codecs.py   (needs Pillow with openjpeg
and libtiff; the committed outputs are what the tests use).
"""

import io
import pathlib
import struct

from PIL import Image, TiffImagePlugin  # noqa: F401  (plugin registers G4)

HERE = pathlib.Path(__file__).parent


def make_jpx() -> bytes:
    im = Image.new("RGB", (8, 8), (0, 0, 255))
    for y in range(8):
        for x in range(4):
            im.putpixel((x, y), (255, 0, 0))
    buf = io.BytesIO()
    # .j2k = raw codestream (no JP2 box wrapper); irreversible=False = 5/3
    # reversible wavelet, i.e. lossless.
    im.save(buf, format="JPEG2000", codec="j2k", irreversible=False)
    return buf.getvalue()


def g4_strip(im: Image.Image) -> bytes:
    """CCITT G4 bits for a bilevel image, via libtiff; one strip."""
    buf = io.BytesIO()
    im.save(buf, format="TIFF", compression="group4")
    buf.seek(0)
    t = Image.open(buf)
    offs = t.tag_v2[273]
    counts = t.tag_v2[279]
    assert len(offs) == 1, "want a single strip"
    raw = buf.getvalue()
    return raw[offs[0]: offs[0] + counts[0]]


def segment(number: int, seg_type: int, data: bytes, page: int = 1) -> bytes:
    """T.88 7.2 segment header (short form: <= 4 referred segments, none here;
    1-byte page association) followed by the data."""
    flags = seg_type & 0x3F  # page-association-size bit 0 -> 1-byte page
    referred = 0x00  # count 0, no retention bits
    return (
        struct.pack(">I", number)
        + bytes([flags, referred, page])
        + struct.pack(">I", len(data))
        + data
    )


def make_jbig2() -> bytes:
    w, h = 16, 8
    # T.6 codes runs of 0-bits ("white" runs) and 1-bits ("black" runs), and
    # JBIG2's MMR generic region paints a 1-bit black. Pillow's mode "1" keeps
    # the raw pixel value as the bit (MinIsBlack TIFF), so the pixels that must
    # come out BLACK in JBIG2 are the ones set to 1 here -- the opposite of how
    # Pillow itself would display them. (First attempt had it the other way
    # round; mutool rendered it mirrored, which is how this got pinned.)
    im = Image.new("1", (w, h), 0)
    for y in range(h):
        for x in range(8):
            im.putpixel((x, y), 1)
    g4 = g4_strip(im)
    # Page information (type 48), T.88 7.4.8: width, height, x/y resolution,
    # flags (default pixel value 0 = white), striping info (none).
    page_info = struct.pack(">IIII", w, h, 0, 0) + bytes([0x00]) + struct.pack(">H", 0)
    # Immediate lossless generic region (type 39), T.88 7.4.6: region segment
    # info (w, h, x, y, external combination op 0 = OR), then the generic
    # region flags byte with MMR = 1 (no AT pixels in MMR mode), then the data.
    region_info = struct.pack(">IIII", w, h, 0, 0) + bytes([0x00])
    generic = region_info + bytes([0x01]) + g4
    return segment(0, 48, page_info) + segment(1, 39, generic)


def main() -> None:
    (HERE / "jpx-8x8-red-blue.j2k").write_bytes(make_jpx())
    (HERE / "jbig2-16x8-left-black.jb2").write_bytes(make_jbig2())


if __name__ == "__main__":
    main()
