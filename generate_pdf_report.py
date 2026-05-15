"""
Generate a PDF version of the Thesis Results Report from Markdown.
Uses markdown + xhtml2pdf (pisa) for rendering.
"""
import sys
import subprocess

# Ensure dependencies
for pkg in ["markdown", "xhtml2pdf"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import markdown
from xhtml2pdf import pisa
from pathlib import Path

MD_PATH = Path(__file__).parent / "Thesis_Results_Report.md"
PDF_PATH = Path(__file__).parent / "Thesis_Results_Report.pdf"

CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 11px; line-height: 1.6; color: #1a1a1a; }
h1 { font-size: 22px; color: #1a365d; border-bottom: 3px solid #2b6cb0; padding-bottom: 8px; margin-top: 30px; }
h2 { font-size: 18px; color: #2b6cb0; border-bottom: 1px solid #bee3f8; padding-bottom: 5px; margin-top: 25px; }
h3 { font-size: 14px; color: #2c5282; margin-top: 18px; }
h4 { font-size: 12px; color: #4a5568; margin-top: 12px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10px; }
th { background-color: #2b6cb0; color: white; padding: 6px 10px; text-align: left; font-weight: bold; }
td { padding: 5px 10px; border-bottom: 1px solid #e2e8f0; }
tr:nth-child(even) td { background-color: #f7fafc; }
code { background-color: #edf2f7; padding: 1px 4px; border-radius: 3px; font-family: 'Consolas', monospace; font-size: 10px; }
pre { background-color: #1a202c; color: #e2e8f0; padding: 12px; border-radius: 6px; font-size: 9px; overflow-x: auto; white-space: pre-wrap; }
blockquote { border-left: 4px solid #2b6cb0; background: #ebf8ff; padding: 8px 14px; margin: 10px 0; font-style: italic; }
strong { color: #1a365d; }
hr { border: none; border-top: 2px solid #e2e8f0; margin: 25px 0; }
"""

def main():
    print(f"Reading: {MD_PATH}")
    md_text = MD_PATH.read_text(encoding="utf-8")

    extensions = ["tables", "fenced_code", "codehilite", "toc", "sane_lists"]
    html_body = markdown.markdown(md_text, extensions=extensions)

    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>{CSS}</style>
</head><body>{html_body}</body></html>"""

    print(f"Writing PDF: {PDF_PATH}")
    with open(PDF_PATH, "wb") as f:
        status = pisa.CreatePDF(full_html, dest=f)

    if status.err:
        print(f"ERROR: PDF generation failed with {status.err} errors.")
        sys.exit(1)
    else:
        size_kb = PDF_PATH.stat().st_size / 1024
        print(f"SUCCESS: PDF generated ({size_kb:.1f} KB) -> {PDF_PATH}")

if __name__ == "__main__":
    main()
