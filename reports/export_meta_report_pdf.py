#!/usr/bin/env python3
"""Export the reviewed Markdown synthesis as a lab-report HTML/PDF.

Reads only the Markdown report; does not load data or execute experiments.
Uses the existing local Chrome PDF renderer, with no network dependencies.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "recorded_salutation_meta_report_2026-09-22.md"
OUTPUT = SOURCE.with_suffix(".html")
PDF = SOURCE.with_suffix(".pdf")
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def inline(text: str) -> str:
    """Render the small inline Markdown subset used by this report."""
    pattern = r"(`[^`]+`|\[[^\]]+\]\([^\)]+\)|\*\*[^*]+\*\*|\*[^*]+\*)"
    parts = []
    for token in re.split(pattern, text):
        if token.startswith("`"):
            parts.append("<code>" + html.escape(token[1:-1]) + "</code>")
        elif token.startswith("["):
            match = re.fullmatch(r"\[([^\]]+)\]\(([^\)]+)\)", token)
            assert match is not None
            label, target = match.groups()
            if not target.startswith("https://"):
                assert (HERE / target).exists(), f"Missing local source: {target}"
            parts.append(f'<a href="{html.escape(target, quote=True)}">{html.escape(label)}</a>')
        elif token.startswith("**"):
            parts.append("<strong>" + html.escape(token[2:-2]) + "</strong>")
        elif token.startswith("*"):
            parts.append("<em>" + html.escape(token[1:-1]) + "</em>")
        else:
            parts.append(html.escape(token))
    return "".join(parts)


EQUATION = """<div class="equation">
<div>ȳ<sub>h</sub> = M + ∑<sub>k=1</sub><sup>3</sup>
[a<sub>k</sub> cos(2πkh/24) + b<sub>k</sub> sin(2πkh/24)] + ε<sub>h</sub></div>
<div>θ̂ = arg min<sub>θ</sub> ∑<sub>h∈O</sub> n<sub>h</sub>
[ȳ<sub>h</sub> − f<sub>θ</sub>(h)]<sup>2</sup></div>
</div>"""


def blocks(markdown: str) -> str:
    """Render paragraphs, headings, tables and lists from the reviewed source."""
    rendered = []
    lines = markdown.strip().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == r"\[":
            while lines[i].strip() != r"\]":
                i += 1
            rendered.append(EQUATION)
            i += 1
        elif line.startswith("### "):
            rendered.append("<h3>" + inline(line[4:]) + "</h3>")
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append(cells)
                i += 1
            rendered.append("<table><thead><tr>" + "".join("<th>" + inline(c) + "</th>" for c in rows[0]) + "</tr></thead><tbody>")
            for row in rows[1:]:
                rendered.append("<tr>" + "".join("<td>" + inline(c) + "</td>" for c in row) + "</tr>")
            rendered.append("</tbody></table>")
        elif re.match(r"(?:\d+\. |\- )", line):
            ordered = line[0].isdigit()
            tag = "ol" if ordered else "ul"
            rendered.append(f"<{tag}>")
            while i < len(lines) and re.match(r"(?:\d+\. |\- )", lines[i].strip()):
                rendered.append("<li>" + inline(re.sub(r"^(?:\d+\. |\- )", "", lines[i].strip())) + "</li>")
                i += 1
            rendered.append(f"</{tag}>")
        else:
            paragraph = [line]
            i += 1
            while i < len(lines) and lines[i].strip():
                paragraph.append(lines[i].strip())
                i += 1
            rendered.append("<p>" + inline(" ".join(paragraph)) + "</p>")
    return "\n".join(rendered)


CSS = """
@page {
  size: A4;
  margin: 17mm 17mm 18mm;
  @bottom-left { content: "Recorded-salutation prediction · 22 September 2026"; font: 8pt Arial; color: #555; }
  @bottom-right { content: counter(page) " / " counter(pages); font: 8pt Arial; color: #555; }
}
* { box-sizing: border-box; }
body { margin: 0; color: #161616; font: 10pt/1.38 Georgia, "Times New Roman", serif; }
h1 { font: bold 19pt/1.15 Arial, sans-serif; margin: 0 0 8pt; }
.meta { font: 9pt/1.4 Arial, sans-serif; border-bottom: 1pt solid #333; padding-bottom: 9pt; margin-bottom: 14pt; }
h2 { font: bold 13pt/1.2 Arial, sans-serif; margin: 17pt 0 7pt; border-bottom: .5pt solid #aaa; padding-bottom: 4pt; break-after: avoid; }
h3 { font: bold 10.5pt/1.25 Arial, sans-serif; margin: 11pt 0 5pt; break-after: avoid; }
p { margin: 0 0 7pt; orphans: 3; widows: 3; }
li { margin: 0 0 7pt; orphans: 3; widows: 3; }
ul, ol { margin: 6pt 0 10pt; padding-left: 18pt; }
table { width: 100%; border-collapse: collapse; margin: 9pt 0 10pt; font: 8.7pt/1.3 Arial, sans-serif; break-inside: avoid; }
th { text-align: left; background: #ededed; border-top: 1pt solid #444; border-bottom: .7pt solid #666; }
td, th { padding: 6pt; vertical-align: top; }
td { border-bottom: .4pt solid #ccc; }
tr { break-inside: avoid; }
a { color: #164b63; text-decoration: none; }
code { font: 8.2pt Menlo, monospace; overflow-wrap: anywhere; }
.equation { text-align: center; font-family: "Times New Roman", serif; font-size: 11pt; line-height: 1.8; margin: 10pt 0; break-inside: avoid; }
.provenance { font: 8pt/1.35 Arial, sans-serif; margin-top: 14pt; padding-top: 6pt; border-top: .5pt solid #bbb; }
"""


def main() -> None:
    source = SOURCE.read_text()
    sections = re.split(r"^## (.+)\n", source, flags=re.MULTILINE)
    section = dict(zip(sections[1::2], sections[2::2]))
    assessment = section["Assessment"].strip().split("\n\n")
    filling = section["New result: filling sensitivity"].strip().split("\n\n")
    cosinor = section["What exactly was the cosinor model?"].strip().split("\n\n")
    # Reorder existing reviewed text; omit the research-style assessment only.
    objective = assessment[2]
    design = filling[0] + "\n\n" + filling[2]
    cosinor_methods = "\n\n".join(cosinor[:-2])
    fill_results = "\n\n".join([filling[1]] + filling[3:-1])
    discussion = assessment[0] + "\n\n" + filling[-1]
    discussion += "\n\n### Trend sensitivity\n\n" + "\n\n".join(cosinor[-2:])
    discussion += "\n\n### Naive Bayes baseline\n\n" + section["Is naive Bayes warranted?"]
    body = ["<h1>Recorded-salutation prediction<br>from wearable data</h1>",
            '<div class="meta">Laboratory report · 22 September 2026<br>Campaign synthesis and filling-sensitivity results · 16 experiment stages</div>']
    for title, content in [
        ("1. Objective", objective),
        ("2. Methods", "### Filling-sensitivity design\n\n" + design + "\n\n### Cosinor model and preprocessing\n\n" + cosinor_methods),
        ("3. Results", "### Campaign findings\n\n" + section["What the campaign establishes"] + "\n\n### Filling sensitivity\n\n" + fill_results),
        ("4. Discussion", discussion),
        ("5. Next steps", section["Revised next steps"]),
        ("6. Limitations and execution record", section["Material limitations and verification record"]),
    ]:
        body.append("<h2>" + title + "</h2>" + blocks(content))
    body.append('<p class="provenance">Source: recorded_salutation_meta_report_2026-09-22.md. '
                'Layout export only; no experiments rerun or metrics recomputed. '
                'Local artifact links require the repository; external sources are linked in the text.</p>')
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Recorded-salutation prediction — laboratory report</title><style>' + CSS + '</style></head><body>' + "\n".join(body) + '</body></html>'
    assert "Selective classification" not in document
    OUTPUT.write_text(document)
    with tempfile.TemporaryDirectory(prefix="salutation-report-chrome-") as profile:
        fresh_pdf = Path(profile) / "report.pdf"
        process = subprocess.Popen([
            str(CHROME), "--headless=new", "--disable-gpu", "--no-sandbox",
            "--disable-background-networking", "--no-first-run",
            f"--user-data-dir={profile}", f"--print-to-pdf={fresh_pdf}",
            "--no-pdf-header-footer", "--virtual-time-budget=10000", OUTPUT.as_uri(),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                data = fresh_pdf.read_bytes() if fresh_pdf.exists() else b""
                if data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF"):
                    # Chrome may remain alive after writing. Validate the new file
                    # before stopping this isolated renderer and replacing the PDF.
                    subprocess.run(["/opt/homebrew/bin/pdfinfo", str(fresh_pdf)],
                                   check=True, timeout=10, stdout=subprocess.DEVNULL)
                    break
                if process.poll() is not None:
                    raise RuntimeError(f"Chrome exited without a complete PDF ({process.returncode})")
                time.sleep(0.2)
            else:
                raise TimeoutError("Chrome did not write a complete PDF within 60 seconds")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        PDF.write_bytes(data)
    manifest = {
        "source": SOURCE.name,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "export_script": Path(__file__).name,
        "html": OUTPUT.name,
        "html_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "pdf": PDF.name,
        "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
        "note": "Lab-report layout derived from reviewed Markdown; no models or metrics recomputed.",
    }
    SOURCE.with_name(SOURCE.stem + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {PDF} ({PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
