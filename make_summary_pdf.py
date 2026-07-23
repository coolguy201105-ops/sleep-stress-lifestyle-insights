"""
Convert Research_Summary.md into a styled, multi-page PDF report.

Run:
    python make_summary_pdf.py
"""

from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Research_Summary.md"
OUTPUT = ROOT / "Research_Summary.pdf"

CSS = """
<style>
    @page {
        size: letter;
        margin: 0.7in 0.75in;
    }
    body {
        font-family: Helvetica, Arial, sans-serif;
        font-size: 10pt;
        line-height: 1.35;
        color: #1F2937;
    }
    h1 {
        font-size: 19pt;
        color: #7C3AED;
        margin: 0 0 2px 0;
    }
    h1 + p {
        font-size: 11pt;
        font-weight: bold;
        color: #4B5563;
        margin: 0 0 14px 0;
    }
    h2 {
        font-size: 13pt;
        color: #2563EB;
        background-color: #F5F3FF;
        padding: 4px 9px;
        margin: 16px 0 6px 0;
        border-radius: 4px;
    }
    p, li {
        margin: 3px 0;
    }
    ul, ol {
        margin: 3px 0 8px 18px;
        padding: 0;
    }
    strong {
        color: #111827;
    }
    em {
        color: #4B5563;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 6px 0 10px 0;
        font-size: 9pt;
    }
    th, td {
        border: 1px solid #D1D5DB;
        padding: 4px 7px;
        text-align: left;
    }
    th {
        background-color: #EEF2FF;
        color: #1F2937;
    }
</style>
"""


def build_pdf() -> None:
    markdown_text = SOURCE.read_text(encoding="utf-8")
    body_html = markdown.markdown(markdown_text, extensions=["tables"])
    full_html = f"<html><head>{CSS}</head><body>{body_html}</body></html>"

    with open(OUTPUT, "wb") as pdf_file:
        result = pisa.CreatePDF(src=full_html, dest=pdf_file)

    if result.err:
        raise RuntimeError(f"PDF generation failed with {result.err} error(s).")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
