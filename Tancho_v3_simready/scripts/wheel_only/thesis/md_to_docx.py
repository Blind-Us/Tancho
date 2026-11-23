#!/usr/bin/env python3
"""Build a .docx draft of the 4.6 rewrite: the markdown draft (headings, paragraphs, bullets),
the comparison table and the figures.  Needs python-docx.

    python scripts/wheel_only/thesis/md_to_docx.py <draft.md> <table.md> <fig dir> <out.docx>
"""
import re, sys, glob, os
from docx import Document
from docx.shared import Pt, Cm
from docx.oxml.ns import qn

md, table_md, figdir, out = sys.argv[1:5]
FONT = "DFKai-SB"
d = Document()
for sec in d.sections:
    sec.left_margin = sec.right_margin = Cm(2.5)
st = d.styles["Normal"]; st.font.name = FONT; st.font.size = Pt(12); st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
FIGS = {"4.6.2": "fig_final_pitch10.png", "4.6.3": "fig_final_cmd.png", "4.6.4": "fig_final_push.png"}
CAP = {"fig_final_pitch10.png": "初始俯仰 ±10° 之俯仰偏差與水平位移（最終版，標稱）",
       "fig_final_cmd.png": "速度命令追蹤（最終版，標稱）",
       "fig_final_push.png": "50 ms 推力掃描之最大俯仰偏差（標稱與雜訊 + 20 ms 延遲）"}
cur = None


def flush_fig():
    global cur
    if cur and cur in FIGS:
        f = os.path.join(figdir, FIGS[cur])
        if os.path.exists(f):
            d.add_picture(f, width=Cm(15)); p = d.add_paragraph(CAP[FIGS[cur]]); p.alignment = 1
    cur = None


for line in open(md, encoding="utf-8"):
    line = line.rstrip("\n")
    if not line.strip() or line.startswith(">"):
        continue
    m = re.match(r"^(#+)\s+(.*)", line)
    if m:
        flush_fig()
        level = len(m.group(1)); text = re.sub(r"（圖.*?）", "", m.group(2))
        d.add_heading(text, level=min(level, 3))
        k = re.match(r"(4\.6\.\d)", text); cur = k.group(1) if k else None
        continue
    text = re.sub(r"[`*]", "", line)
    if text.startswith("- "):
        d.add_paragraph(text[2:], style="List Bullet")
    else:
        d.add_paragraph(text)
flush_fig()
rows = [r.strip().strip("|").split(" | ") for r in open(table_md, encoding="utf-8") if r.startswith("|") and not r.startswith("|---")]
d.add_heading("表 LQR、RL 最終版與實機部署之比較", level=3)
t = d.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Table Grid"
for i, r in enumerate(rows):
    for j, c in enumerate(r):
        cell = t.cell(i, j); cell.text = c.strip()
        for p in cell.paragraphs:
            for run in p.runs:
                run.font.size = Pt(9)
d.save(out); print("saved", out)
