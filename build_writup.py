"""Render WRITEUP.md to WRITEUP.pdf (2-4 pages) with fpdf2 + DejaVu fonts.

Supports the small Markdown subset used by WRITEUP.md:
  headings (#, ##, ###), paragraphs, '- ' bullets, pipe tables, images
  (![alt](path)), and inline **bold** / `code`.
"""

from __future__ import annotations

import pathlib
import re

import matplotlib
from fpdf import FPDF

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "WRITEUP.md"
OUT = ROOT / "WRITEUP.pdf"

_TTF = pathlib.Path(matplotlib.get_data_path()) / "fonts" / "ttf"
REG = _TTF / "DejaVuSans.ttf"
BOLD = _TTF / "DejaVuSans-Bold.ttf"
MONO = _TTF / "DejaVuSansMono.ttf"
MONO_B = _TTF / "DejaVuSansMono-Bold.ttf"

BODY = 9.4
LEAD = 4.9
MARGIN = 16.0


def build():
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_font("DejaVu", "", str(REG))
    pdf.add_font("DejaVu", "B", str(BOLD))
    pdf.add_font("DejaVu", "I", str(REG))
    pdf.add_font("DejaVuMono", "", str(MONO))
    pdf.add_font("DejaVuMono", "B", str(MONO_B))
    pdf.add_page()
    pdf.set_left_margin(MARGIN)
    pdf.set_right_margin(MARGIN)

    lines = SRC.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue

        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            _table(pdf, block)
            pdf.ln(1.5)
            continue

        m = re.match(r"^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)", line)
        if m:
            _image(pdf, m.group("src"), m.group("alt"))
            i += 1
            continue

        if line.startswith("### "):
            _heading(pdf, line[4:], 11.5, lead_before=2.0)
        elif line.startswith("## "):
            _heading(pdf, line[3:], 13.5, lead_before=3.5)
        elif line.startswith("# "):
            _heading(pdf, line[2:], 17.0, lead_before=0.0, center=True)
        elif line.startswith("- "):
            _bullet(pdf, line[2:])
        else:
            _para(pdf, line)
        i += 1

    pdf.output(str(OUT))
    print(f"[saved] {OUT}")


def _rich(pdf, text, size):
    text = text.replace("`", "")
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            pdf.set_font("DejaVu", "B", size)
            pdf.write(LEAD, part[2:-2])
        elif part.startswith("*") and part.endswith("*"):
            pdf.set_font("DejaVu", "I", size)
            pdf.write(LEAD, part[1:-1])
        else:
            pdf.set_font("DejaVu", "", size)
            pdf.write(LEAD, part)


def _para(pdf, text):
    pdf.set_font("DejaVu", "", BODY)
    _rich(pdf, text, BODY)
    pdf.ln(LEAD)
    pdf.ln(0.8)


def _bullet(pdf, text):
    pdf.ln(0.4)
    x0 = MARGIN + 2.0
    pdf.set_x(x0)
    pdf.set_font("DejaVu", "", BODY)
    pdf.write(LEAD, "\u2022  ")
    x1 = pdf.get_x()
    pdf.set_left_margin(x1 + 1.0)
    pdf.set_x(x1 + 1.0)
    _rich(pdf, text, BODY)
    pdf.set_left_margin(MARGIN)
    pdf.ln(LEAD)
    pdf.set_x(MARGIN)


def _heading(pdf, text, size, lead_before=0.0, center=False):
    if lead_before:
        pdf.ln(lead_before)
    pdf.set_font("DejaVu", "B", size)
    if center:
        pdf.multi_cell(0, size * 0.62, text, align="C")
    else:
        pdf.set_x(MARGIN)
        pdf.multi_cell(0, size * 0.62, text)
    pdf.ln(1.5)
    pdf.set_x(MARGIN)


def _image(pdf, src, alt):
    path = (ROOT / src).resolve()
    if not path.exists():
        _para(pdf, f"[missing figure: {src}]")
        return
    w = min(112.0, pdf.w - 2 * MARGIN)
    x = (pdf.w - w) / 2
    try:
        pdf.image(str(path), x=x, w=w)
    except Exception as e:  # noqa: BLE001
        _para(pdf, f"[figure {src} failed: {e}]")
        return
    pdf.ln(1.0)
    if alt:
        pdf.set_font("DejaVu", "I", 8.0)
        pdf.set_x(MARGIN)
        pdf.multi_cell(0, 4.0, alt, align="C")
    pdf.ln(2.5)
    pdf.set_x(MARGIN)


def _table(pdf, block):
    rows = [[c.strip() for c in r.strip("|").split("|")] for r in block]
    rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    widths = [0] * ncol
    for r in rows:
        for j in range(ncol):
            widths[j] = max(widths[j], len(r[j]) if j < len(r) else 0)
    pdf.ln(0.5)
    for idx, r in enumerate(rows):
        cells = [(r[j] if j < len(r) else "").ljust(widths[j]) for j in range(ncol)]
        line = "  ".join(cells)
        pdf.set_font("DejaVuMono", "B" if idx == 0 else "", 7.4)
        pdf.set_x(MARGIN)
        pdf.multi_cell(0, 3.9, line)
    pdf.set_x(MARGIN)


if __name__ == "__main__":
    build()
