"""
Convert Project_Proposal.md into a styled, printable one-page PDF.

Run:
    python make_proposal_pdf.py
"""

from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Project_Proposal.md"
OUTPUT = ROOT / "Project_Proposal.pdf"

CSS = """
<style>
    @page {
        size: letter;
        margin: 0.45in 0.6in;
    }
    body {
        font-family: Helvetica, Arial, sans-serif;
        font-size: 8pt;
        line-height: 1.2;
        color: #1F2937;
    }
    h1 {
        font-size: 14pt;
        color: #7C3AED;
        margin: 0 0 1px 0;
    }
    h1 + p {
        font-size: 9pt;
        font-weight: bold;
        color: #4B5563;
        margin: 0 0 6px 0;
    }
    h2 {
        font-size: 9.5pt;
        color: #2563EB;
        background-color: #F5F3FF;
        padding: 2px 7px;
        margin: 6px 0 3px 0;
        border-radius: 4px;
    }
    p, li {
        margin: 1px 0;
    }
    ul, ol {
        margin: 1px 0 4px 16px;
        padding: 0;
    }
    strong {
        color: #111827;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 2px;
        font-size: 7.5pt;
    }
    th, td {
        border: 1px solid #D1D5DB;
        padding: 2px 5px;
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
