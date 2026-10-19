"""Render WRITEUP.md to WRITEUP.pdf (2-4 pages) with fpdf2, DejaVu fonts, and Matplotlib LaTeX math.

Supports:
  - Headings (#, ##, ###)
  - Paragraphs and '- ' bullets
  - Pipe tables
  - Images (![alt](path))
  - Inline LaTeX math ($...$), bold (**...**), and italic (*...*)
  - Display LaTeX math blocks ($$...$$) rendered via Matplotlib mathtext
"""

from __future__ import annotations

import io
import pathlib
import re

import matplotlib
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "WRITEUP.md"
OUT = ROOT / "WRITEUP.pdf"

_TTF = pathlib.Path(matplotlib.get_data_path()) / "fonts" / "ttf"
REG = _TTF / "DejaVuSans.ttf"
BOLD = _TTF / "DejaVuSans-Bold.ttf"
MONO = _TTF / "DejaVuSansMono.ttf"
MONO_B = _TTF / "DejaVuSansMono-Bold.ttf"

BODY = 9.2
LEAD = 4.6
MARGIN = 16.0


def _math_block(pdf, formula: str):
    formula = formula.replace(r"\le", r"\leq").replace(r"\ge", r"\geq")
    fig, ax = plt.subplots(figsize=(6, 0.6))
    ax.text(0.5, 0.5, f"${formula}$", ha="center", va="center", fontsize=15)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)

    im = Image.open(buf)
    w_px, h_px = im.size
    aspect = w_px / max(h_px, 1)
    h_mm = 10.5
    w_mm = h_mm * aspect
    max_w = pdf.w - 2 * MARGIN
    if w_mm > max_w:
        w_mm = max_w
        h_mm = w_mm / aspect

    x = (pdf.w - w_mm) / 2
    pdf.ln(1.5)
    pdf.image(buf, x=x, w=w_mm, h=h_mm)
    pdf.ln(h_mm + 2.0)
    pdf.set_x(MARGIN)


def build():
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(True, margin=14)
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

        if line.startswith("$$") and line.endswith("$$") and len(line) >= 4:
            _math_block(pdf, line[2:-2].strip())
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
            _heading(pdf, line[4:], 11.2, lead_before=2.0)
        elif line.startswith("## "):
            _heading(pdf, line[3:], 13.0, lead_before=3.2)
        elif line.startswith("# "):
            _heading(pdf, line[2:], 16.5, lead_before=0.0, center=True)
        elif line.startswith("- "):
            _bullet(pdf, line[2:])
        else:
            _para(pdf, line)
        i += 1

    pdf.output(str(OUT))
    print(f"[saved] {OUT}")


def _latex_inline(s: str) -> str:
    s = s.replace(r"^{-30}", "⁻³⁰")
    s = s.replace(r"^{-4}", "⁻⁴")
    s = s.replace(r"^{-1}", "⁻¹")
    s = s.replace(r"^{-cn}", "⁻ᶜⁿ")
    s = s.replace(r"^{1.6}", "¹˙⁶")
    s = s.replace(r"^{2.0}", "²˙⁰")
    s = s.replace(r"e^{-cn}", "e⁻ᶜⁿ")
    s = s.replace(r"2^{1.6}", "2¹˙⁶")
    s = s.replace(r"2^{2.0}", "2²˙⁰")
    s = s.replace(r"\{", "{").replace(r"\}", "}")
    s = s.replace(r"\mathcal{U}", "U")
    s = s.replace(r"\mathcal{O}", "O")
    s = s.replace(r"\mathcal", "")
    s = s.replace(r"\mathrm{Var}", "Var")
    s = s.replace(r"\mathrm", "")
    s = s.replace(r"\log", "log")
    s = s.replace(r"\le", "≤").replace(r"\leq", "≤")
    s = s.replace(r"\ge", "≥").replace(r"\geq", "≥")
    s = s.replace(r"\in", "∈")
    s = s.replace(r"\times", "×")
    s = s.replace(r"\cdot", "·")
    s = s.replace(r"\dots", "…")
    s = s.replace(r"\approx", "≈")
    s = s.replace(r"\sim", "~")
    s = s.replace(r"\pm", "±")
    s = s.replace(r"\lambda", "λ")
    s = s.replace(r"\theta", "θ")
    s = s.replace(r"\varepsilon", "ε").replace(r"\eps", "ε")
    s = s.replace(r"\sum", "∑")
    s = s.replace(r"\partial", "∂")
    s = s.replace(r"\langle", "⟨").replace(r"\rangle", "⟩")
    s = s.replace(r"|0\dots0\rangle", "|0…0⟩")
    s = s.replace(r"|0\dots0>", "|0…0⟩")
    s = s.replace(r"|0...0>", "|0…0⟩")
    s = s.replace("{", "").replace("}", "")
    return s


def _rich(pdf, text: str, size: float):
    text = text.replace("`", "")
    tokens = re.split(r"(\$[^$]+\$|\*\*[^*]+\*\*|\*[^*]+\*)", text)
    for part in tokens:
        if not part:
            continue
        if part.startswith("$") and part.endswith("$") and len(part) > 1:
            clean = _latex_inline(part[1:-1])
            pdf.set_font("DejaVu", "I", size)
            pdf.write(LEAD, clean)
        elif part.startswith("**") and part.endswith("**"):
            pdf.set_font("DejaVu", "B", size)
            pdf.write(LEAD, part[2:-2])
        elif part.startswith("*") and part.endswith("*"):
            pdf.set_font("DejaVu", "I", size)
            pdf.write(LEAD, part[1:-1])
        else:
            pdf.set_font("DejaVu", "", size)
            pdf.write(LEAD, part)


def _para(pdf, text: str):
    pdf.set_font("DejaVu", "", BODY)
    _rich(pdf, text, BODY)
    pdf.ln(LEAD)
    pdf.ln(0.8)


def _bullet(pdf, text: str):
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


def _heading(pdf, text: str, size: float, lead_before: float = 0.0, center: bool = False):
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


def _image(pdf, src: str, alt: str):
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


def _table(pdf, block: list[str]):
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
