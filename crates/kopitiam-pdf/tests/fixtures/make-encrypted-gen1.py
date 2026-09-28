#!/usr/bin/env python3
"""Generate `encrypted-rc4-gen1.pdf`: an RC4-40 (/R 2) encrypted one-page PDF
whose content stream is object **4 1 obj** -- generation 1, not 0.

Why: before kopitiam-pdf 0.4.2, `PdfDocument::stream_raw_num` decrypted every
stream with generation 0, while strings used the real generation. The
per-object RC4/AES key is MD5(file key || num || gen) (PDF 32000-1 7.6.2,
algorithm 1), so a stream in a non-zero-generation object decrypted to
garbage -- here: a page that should show a red square renders blank.

Two steps, both reproducible:
  1. this script writes the plaintext `gen1.pdf` (content `1 0 0 rg 20 20 60
     60 re f` as `4 1 obj`);
  2. MuPDF (19f1284) encrypts it, preserving generations:
       mutool clean -E rc4-40 -P -4 gen1.pdf encrypted-rc4-gen1.pdf
The committed file is the output of step 2; mutool itself renders it with the
red square at (50, 50).
"""

import pathlib

HERE = pathlib.Path(__file__).parent

content = b"1 0 0 rg 20 20 60 60 re f"
objs = [
    (1, 0, b"<< /Type /Catalog /Pages 2 0 R >>"),
    (2, 0, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
    (3, 0, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 1 R >>"),
    (4, 1, b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream"),
]
out = b"%PDF-1.4\n"
offs = {}
for n, g, body in objs:
    offs[n] = (len(out), g)
    out += b"%d %d obj\n" % (n, g) + body + b"\nendobj\n"
x = len(out)
out += b"xref\n0 5\n0000000000 65535 f \n"
for n in range(1, 5):
    o, g = offs[n]
    out += b"%010d %05d n \n" % (o, g)
out += (b"trailer\n<< /Size 5 /Root 1 0 R /ID [<0123456789abcdef0123456789abcdef> "
        b"<0123456789abcdef0123456789abcdef>] >>\nstartxref\n%d\n%%%%EOF\n" % x)
(HERE / "gen1.pdf").write_bytes(out)
print("now run: mutool clean -E rc4-40 -P -4 gen1.pdf encrypted-rc4-gen1.pdf && rm gen1.pdf")
