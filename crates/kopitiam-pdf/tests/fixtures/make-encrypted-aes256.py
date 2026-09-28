#!/usr/bin/env python3
"""Generate the AES-256 (/V 5 /R 6, /AESV3) fixtures.

Why: before kopitiam-pdf 0.4.2 the standard security handler refused /R 5
and /R 6 by name, so every file encrypted the Acrobat X+ way -- including
owner-restricted ones with an EMPTY user password, which MuPDF opens with no
prompt -- failed PdfDocument::open. Two cases, both reproducible:

  1. this script writes the plaintext `aes-plain.pdf` (a red square,
     `1 0 0 rg 20 20 60 60 re f`, plus a text string object);
  2. MuPDF (19f1284) encrypts it twice:
       mutool clean -E aes-256 -O owner -P -4 aes-plain.pdf encrypted-aes256-r6.pdf
         (empty USER password: opens as the user)
       mutool clean -E aes-256 -U secret -O "" -P -4 aes-plain.pdf encrypted-aes256-r6-empty-owner.pdf
         (user password "secret", EMPTY owner password: the empty password
         authenticates as the owner, but MuPDF -- "to match Acrobat" --
         refuses an empty owner password unless the user password is empty
         too, pdf-crypt.c:817. mutool refuses it without `-p secret`; so
         must we.)
The committed files are the outputs of step 2; mutool renders the first with
the red square at (50, 50).
"""

import pathlib

HERE = pathlib.Path(__file__).parent

content = b"1 0 0 rg 20 20 60 60 re f"
objs = [
    b"<< /Type /Catalog /Pages 2 0 R >>",
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R >>",
    b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
    b"<< /Title (AES-256 fixture) >>",
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
(HERE / "aes-plain.pdf").write_bytes(out)
print("now run the two mutool clean commands in the docstring, then rm aes-plain.pdf")
