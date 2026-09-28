#!/usr/bin/env python3
"""Generate the AES-128 (/V 4 /R 4, /AESV2) fixture -- gh-98's exact shape.

Why: gh-98 was filed on a government form encrypted /Standard /V 4 /R 4
/AESV2 with an EMPTY user password and owner restrictions -- the ordinary
fill-in-but-don't-edit form that every other viewer opens with no prompt.
Before b5241a7 PdfDocument::open refused it. The crypt.rs unit tests pin the
R4 key derivation and /U check, but nothing opened a real AESV2 file end to
end, so this fixture does.

  1. this script writes the plaintext `aes128-plain.pdf` (a red square,
     `1 0 0 rg 20 20 60 60 re f`, plus an /Info /Title string);
  2. MuPDF (19f1284, mutool 1.29.0) encrypts it, deflating the content
     stream (-z) and packing objects into object streams (-Z):
       mutool clean -z -Z -E aes-128 -O owner -P -4 aes128-plain.pdf encrypted-aes128-r4.pdf
     -z exercises decrypt-BEFORE-inflate (inflating ciphertext is what
     produced gh-98's "corrupt object stream"); -Z puts the /Info string
     inside an object stream, which must NOT be decrypted a second time.
The committed file is the output of step 2; `mutool draw -r 72` renders the
red square at (50, 50) and `mutool show` prints the title.
"""

import pathlib

HERE = pathlib.Path(__file__).parent

content = b"1 0 0 rg 20 20 60 60 re f"
objs = [
    b"<< /Type /Catalog /Pages 2 0 R >>",
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R >>",
    b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
    b"<< /Title (AES-128 fixture) >>",
]
out = b"%PDF-1.7\n"
offs = []
for i, body in enumerate(objs):
    offs.append(len(out))
    out += b"%d 0 obj\n" % (i + 1) + body + b"\nendobj\n"
x = len(out)
out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
for o in offs:
    out += b"%010d 00000 n \n" % o
out += (b"trailer\n<< /Size %d /Root 1 0 R /Info 5 0 R /ID [<00112233445566778899aabbccddeeff> "
        b"<00112233445566778899aabbccddeeff>] >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, x))
(HERE / "aes128-plain.pdf").write_bytes(out)
print("now run the mutool clean command in the docstring, then rm aes128-plain.pdf")
