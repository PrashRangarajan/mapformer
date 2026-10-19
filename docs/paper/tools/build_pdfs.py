import re, sys, os, markdown, weasyprint
REPO = "/home/prashr/mapformer"
CSS = """
@page { size: A4; margin: 20mm 18mm; @bottom-center { content: counter(page); font-size: 9pt; color: #555; } }
body { font-family: 'DejaVu Serif', Georgia, serif; font-size: 10.5pt; line-height: 1.42; color: #111; }
h1 { font-size: 17pt; line-height: 1.2; margin: 0 0 10pt; } h2 { font-size: 13pt; margin: 16pt 0 6pt; border-bottom: 0.5pt solid #999; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; } h4 { font-size: 10.5pt; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 8.8pt; width: 100%; }
th, td { border: 0.5pt solid #888; padding: 2.5pt 4pt; vertical-align: top; } th { background: #eee; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; } pre { font-size: 7.5pt; background: #f5f5f5; padding: 6pt; white-space: pre-wrap; }
img.fig { width: 100%; margin: 6pt 0 2pt; } blockquote { color: #333; border-left: 2pt solid #bbb; margin-left: 0; padding-left: 8pt; }
"""
def build(srcs, out, title):
    html = []
    for s in srcs:
        t = open(os.path.join(REPO, s)).read()
        h = markdown.markdown(t, extensions=["tables", "fenced_code", "sane_lists"])
        # embed every referenced figure png after the paragraph / list item that names it
        def emb(m):
            block = m.group(0)
            figs = re.findall(r"docs/paper/figs/(fig_[A-Za-z0-9_]+)\.png", block)
            seen = []
            for f in figs:
                if f not in seen: seen.append(f)
            imgs = "".join(f'<img class="fig" src="file://{REPO}/docs/paper/figs/{f}.png">' for f in seen
                           if os.path.exists(f"{REPO}/docs/paper/figs/{f}.png"))
            return block + imgs
        h = re.sub(r"<li>.*?</li>|<p>.*?</p>", lambda m: emb(m) if "docs/paper/figs/fig_" in m.group(0) else m.group(0), h, flags=re.S)
        html.append(h)
        html.append('<div style="page-break-after: always"></div>')
    doc = f"<html><head><meta charset='utf-8'><title>{title}</title><style>{CSS}</style></head><body>{''.join(html)}</body></html>"
    weasyprint.HTML(string=doc, base_url=os.path.join(REPO, os.path.dirname(srcs[0])) + "/").write_pdf(out)
    print(out, os.path.getsize(out) // 1024, "KB")
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
P = "docs/paper/"
build([P + "MAIN_DRAFT1.md"], f"{OUT}/1_main_paper.pdf", "Main paper, draft 1")
build([P + "MAIN_APPENDIX1.md"], f"{OUT}/2_main_paper_appendix.pdf", "Main paper appendix")
build([P + "APPENDIX_theory_v3.md"], f"{OUT}/3_theory_appendix.pdf", "Theory appendix")
build([P + "DRAFT6_workshop.md"], f"{OUT}/4_sign_workshop_paper.pdf", "Sign workshop paper, draft 6")
build([P + "APPENDIX_draft6.md"], f"{OUT}/5_sign_workshop_appendix.pdf", "Sign workshop appendix")
build([P + "OUTLINE_main_v2.md"], f"{OUT}/6_main_paper_outline.pdf", "Main paper outline v2")
build([P + "figs/FIGS_NOTES.md"], f"{OUT}/7_figures_and_captions.pdf", "Figures and captions")
