#!/usr/bin/env python3
"""Build the minimalist CtrlHGen paper and reproduction briefing.

The rejected card-style deck is deliberately not modified.  This version uses
the paper's original figures/tables and restrained charts rebuilt from the
archived reproduction records.  Visual policy: white background, black/gray
text, one deep-blue accent, Times New Roman for Latin text/numbers/formulas,
and no truncated axes for absolute metrics.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfReader
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

import build_ctrlhgen_teacher_briefing as evidence_source


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "worklogs/figures"
PAPER_FIG = FIG / "paper"
REPRO_FIG = FIG / "briefing-minimal"
STEM = ROOT / "worklogs/CtrlHGen-论文讲解与复现进展-2026-08-13-极简重制版"
PPTX_OUT = STEM.with_suffix(".pptx")
PDF_OUT = STEM.with_suffix(".pdf")
MD_OUT = STEM.with_suffix(".md")

SLIDE_W = Inches(13.333333)
SLIDE_H = Inches(7.5)
CN = "微软雅黑"
EN = "Times New Roman"

WHITE = RGBColor(255, 255, 255)
BLACK = RGBColor(26, 26, 26)
DARK = RGBColor(64, 64, 64)
GRAY = RGBColor(105, 105, 105)
MID = RGBColor(166, 166, 166)
LIGHT = RGBColor(217, 217, 217)
VERY_LIGHT = RGBColor(243, 243, 243)
BLUE = RGBColor(23, 54, 93)
LIGHT_BLUE = RGBColor(221, 230, 242)
RED = RGBColor(156, 0, 6)

NOTES = evidence_source.NOTES

SOURCES = [
    "论文 p.1 Abstract；worklogs/reproduction-report-2026-08-12.md 摘要",
    "论文 p.2 Fig.1；p.4–5 Definition 3.1, Eq.(1–2)",
    "论文 p.3 Fig.2",
    "论文 p.4 Fig.3；p.5–7 Sec.3.2–3.4",
    "论文 p.5 Eq.(3)；p.7 Fig.4；p.15 Appendix A",
    "论文 p.6–7 Eq.(5–8)；p.10 Table 3",
    "论文 Sec.4.1 / Appendix B；综合报告 §3",
    "综合报告 §4–6；worklogs/phase-a/b/c/d*.md",
    "worklogs/figures/briefing-minimal/phase-c-conditional-loss.png；综合报告 §5.2",
    "worklogs/figures/briefing-minimal/phase-c-jaccard-pattern-accuracy.png；综合报告 §5.3–5.4",
    "综合报告 §5.5；worklogs/phase-c-2026-08-10.md",
    "phase-d trainer_state / rollout audit / validation comparison",
    "repaired-test/comparison.json；论文 p.10 Table 3",
    "综合报告 §9, §11",
    "论文 p.7 Fig.4；p.6 Eq.(5)；akgr/metadata/pattern_filtered.csv",
    "论文 Sec.4.1 / Appendix B；综合报告 §3",
    "repaired-test/comparison.json；综合报告 §5.6, §6.4, §9",
]


def set_run_font(run, size: float, color=BLACK, bold=False, italic=False) -> None:
    run.font.name = EN
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    rpr = run._r.get_or_add_rPr()
    for child in list(rpr):
        if child.tag.endswith("}latin") or child.tag.endswith("}ea"):
            rpr.remove(child)
    latin = OxmlElement("a:latin")
    latin.set("typeface", EN)
    east_asian = OxmlElement("a:ea")
    east_asian.set("typeface", CN)
    rpr.insert(0, east_asian)
    rpr.insert(0, latin)


def add_text(
    slide,
    x,
    y,
    w,
    h,
    text,
    size=16,
    color=BLACK,
    bold=False,
    italic=False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin=0,
):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    tf.word_wrap = True
    for index, line in enumerate(text.split("\n")):
        paragraph = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        paragraph.alignment = align
        paragraph.line_spacing = 1.12
        run = paragraph.add_run()
        run.text = line
        set_run_font(run, size=size, color=color, bold=bold, italic=italic)
    return shape


def add_rich(slide, x, y, w, h, segments, size=16, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = valign
    tf.word_wrap = True
    paragraph = tf.paragraphs[0]
    paragraph.alignment = align
    paragraph.line_spacing = 1.12
    for text, color, bold, italic in segments:
        run = paragraph.add_run()
        run.text = text
        set_run_font(run, size=size, color=color, bold=bold, italic=italic)
    return shape


def add_bullets(slide, x, y, w, h, items, size=15, color=BLACK, gap=7):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = True
    for index, item in enumerate(items):
        p = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        p.text = "•  " + item
        p.space_after = Pt(gap)
        p.line_spacing = 1.08
        for run in p.runs:
            set_run_font(run, size=size, color=color)
    return shape


def add_line(slide, x1, y1, x2, y2, color=LIGHT, width=1.0):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    line.line.color.rgb = color
    line.line.width = Pt(width)
    return line


def add_picture_contain(slide, path: Path, x, y, w, h):
    with Image.open(path) as image:
        iw, ih = image.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return slide.shapes.add_picture(
        str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph)
    )


def set_cell(cell, text, size=12.5, color=BLACK, bold=False, align=PP_ALIGN.CENTER, fill=WHITE):
    cell.fill.solid()
    cell.fill.fore_color.rgb = fill
    cell.margin_left = cell.margin_right = Inches(0.05)
    cell.margin_top = cell.margin_bottom = Inches(0.03)
    tf = cell.text_frame
    tf.clear()
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = str(text)
    set_run_font(run, size=size, color=color, bold=bold)


def add_header(slide, number: int, title: str, section: str):
    title_size = 22.0 if len(title) > 18 else 25.5
    add_text(slide, 0.68, 0.28, 10.15, 0.50, title, size=title_size, bold=True)
    short_section = section.replace("Reproduction", "Repro")
    add_text(slide, 10.88, 0.34, 1.77, 0.32, short_section.upper(), size=9.0, color=GRAY, align=PP_ALIGN.RIGHT)
    add_line(slide, 0.68, 0.92, 12.65, 0.92, color=BLUE, width=1.25)
    add_text(slide, 12.40, 7.03, 0.25, 0.16, str(number), size=8.5, color=GRAY, align=PP_ALIGN.RIGHT)


def add_footer(slide, source: str):
    add_line(slide, 0.68, 6.88, 12.65, 6.88, color=LIGHT, width=0.6)
    add_text(slide, 0.68, 6.96, 11.3, 0.18, source, size=8.2, color=GRAY)


def add_notes(slide, notes: dict, source: str):
    slide.notes_slide.notes_text_frame.text = (
        f"【建议用时】{notes['time']}\n\n"
        f"【核心讲述】\n{notes['talk']}\n\n"
        f"【转场】{notes['transition']}\n\n"
        f"【证据来源】{source}\n\n"
        f"【必要限定】{notes['caveat']}\n\n"
        f"【可能追问】{notes['qa']}"
    )


def add_slide(prs, title, section, source, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    add_header(slide, len(prs.slides), title, section)
    add_footer(slide, source)
    add_notes(slide, notes, source)
    return slide


def note_line(slide, text, y=6.18, color=BLACK, size=14.5):
    add_line(slide, 0.84, y, 0.84, y + 0.38, color=BLUE, width=2.4)
    add_text(slide, 1.02, y - 0.01, 11.2, 0.42, text, size=size, color=color, bold=True)


def build_deck(evidence: dict) -> Presentation:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    # 1 — cover
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = WHITE
    add_line(slide, 0.70, 0.68, 12.62, 0.68, color=BLUE, width=1.5)
    add_text(slide, 0.72, 0.90, 4.5, 0.28, "ICLR 2026 · REPRODUCTION BRIEFING", size=10.5, color=GRAY)
    add_text(slide, 0.72, 1.62, 11.6, 1.28, "CtrlHGen：可控逻辑假设生成\n与复现进展", size=34, bold=True)
    add_text(slide, 0.76, 3.35, 10.8, 0.44, "从论文机制到可审计实验链路", size=19, color=BLUE)
    add_line(slide, 0.76, 4.42, 3.00, 4.42, color=BLACK, width=0.8)
    add_text(slide, 0.76, 4.66, 10.7, 0.90,
             "论文：可控溯因 = 执行语义 + 条件遵循\n复现：监督契约 → 实体覆盖 → GRPO 组内信号", size=17, color=DARK)
    add_text(slide, 0.76, 6.24, 6.5, 0.25, "汇报人：bigbiginsect    2026-08-13", size=11.5, color=GRAY)
    add_footer(slide, SOURCES[0])
    add_notes(slide, NOTES[0], SOURCES[0])

    # 2 — task
    slide = add_slide(prs, "任务：从观察实体集合反求可控逻辑查询", "Paper · Task", SOURCES[1], NOTES[1])
    add_picture_contain(slide, PAPER_FIG / "paper-fig1-controllability.png", 0.75, 1.17, 11.85, 4.75)
    note_line(slide, "同一观察 O 可以有多种合理 H；控制 C 决定语义焦点或结构复杂度。", y=6.16)

    # 3 — challenges
    slide = add_slide(prs, "两个核心难题：候选空间收缩与奖励过敏", "Paper · Motivation", SOURCES[2], NOTES[2])
    add_picture_contain(slide, PAPER_FIG / "paper-fig2-challenges.png", 0.82, 1.18, 11.70, 4.70)
    add_rich(slide, 1.05, 6.10, 11.0, 0.42, [
        ("长逻辑：", BLACK, True, False),
        ("候选 55.36 → 12.97，Jaccard 86.2 → 60.6", BLUE, True, False),
        ("    |    单谓词错误：", BLACK, True, False),
        ("Jaccard = 2/45 = 0.044", RED, True, False),
    ], size=14.5, align=PP_ALIGN.CENTER)

    # 4 — framework
    slide = add_slide(prs, "CtrlHGen 总体框架", "Paper · Method", SOURCES[3], NOTES[3])
    add_picture_contain(slide, PAPER_FIG / "paper-fig3-framework.png", 1.00, 1.12, 11.35, 4.82)
    note_line(slide, "数据构造与分解 → 两阶段 SFT → KG 执行反馈与 GRPO。", y=6.18)

    # 5 — decomposition and controls
    slide = add_slide(prs, "子逻辑分解与控制接口", "Paper · Method", SOURCES[4], NOTES[4])
    add_text(slide, 0.85, 1.34, 5.25, 0.32, "课程式增广：从复杂结构抽取同源子任务", size=17, bold=True)
    add_text(slide, 1.10, 2.05, 1.25, 0.45, "inp", size=30, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    add_line(slide, 2.35, 2.32, 3.20, 2.32, color=BLACK, width=1.0)
    add_text(slide, 3.24, 1.76, 2.20, 1.30, "2p  →  O_A\n2p  →  O_B", size=23, bold=True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
    add_text(slide, 1.02, 3.42, 4.72, 0.95, "不是任意造样本：子逻辑来自同一个复杂假设，\n并在 KG 上重新执行得到子观察。", size=15, color=DARK, align=PP_ALIGN.CENTER)
    add_line(slide, 6.25, 1.32, 6.25, 5.72, color=LIGHT, width=0.8)
    add_picture_contain(slide, PAPER_FIG / "paper-fig4-patterns.png", 6.52, 1.30, 5.55, 3.55)
    add_text(slide, 6.58, 5.00, 5.42, 0.62, "五类控制：pattern  ·  实体数  ·  关系数\n指定实体  ·  指定关系", size=15.5, color=BLACK, align=PP_ALIGN.CENTER)
    note_line(slide, "论文只分解 up / 3in / pni / pin / inp 五类复杂 pattern。", y=6.18, size=13.8)

    # 6 — reward
    slide = add_slide(prs, "双奖励 GRPO：语义质量与控制遵循的权衡", "Paper · Method", SOURCES[5], NOTES[5])
    add_text(slide, 0.86, 1.35, 5.05, 0.38, "Reward definition", size=17, bold=True)
    add_text(slide, 0.95, 1.98, 5.10, 1.60,
             "R_sem = Jaccard + 0.5 Dice\n              + 0.5 Overlap\nR_cond = 1[ H satisfies C ]\nR = 0.5 R_sem + 0.5 R_cond", size=17.2, color=BLACK)
    add_bullets(slide, 1.00, 3.86, 4.86, 1.25, [
        "同一 prompt 采样 k = 4 个候选",
        "组内标准化相对奖励；KL 约束策略漂移",
        "关键是组内能否形成可排序差异",
    ], size=14.5)
    add_line(slide, 6.18, 1.28, 6.18, 5.80, color=LIGHT, width=0.8)
    add_picture_contain(slide, PAPER_FIG / "paper-table3-reward-ablation.png", 6.42, 1.42, 6.02, 3.90)
    add_text(slide, 6.60, 5.38, 5.65, 0.46,
             "去掉条件奖励：集合语义略升，但 Accuracy 93.5 → 68.3。", size=14.5, color=RED, bold=True, align=PP_ALIGN.CENTER)
    note_line(slide, "Table 3 表明：语义质量与条件遵循是明确的双目标，而非单一分数。", y=6.18, size=13.7)

    # 7 — scope table
    slide = add_slide(prs, "复现范围与忠实度边界", "Reproduction · Scope", SOURCES[6], NOTES[6])
    rows = [
        ("模型", "12-layer GPT-2", "GPT2_6 / num_layers: 6", "explicit n_layer = 6"),
        ("优化与 LR", "AdamW / 1e-5", "Adam / 5e-5", "Adam(SFT), AdamW(GRPO)"),
        ("训练", "400 + 50 epoch", "WN batch 160", "50 + 50 epoch, seed 42"),
        ("资源与范围", "4 × A6000 / 3 KG / 5 controls", "公开仓库", "1 × L20 / WN18RR / pattern"),
    ]
    table = slide.shapes.add_table(5, 4, Inches(0.80), Inches(1.42), Inches(11.75), Inches(3.92)).table
    widths = [1.35, 3.35, 3.35, 3.70]
    for i, width in enumerate(widths): table.columns[i].width = Inches(width)
    headers = ["项目", "论文正文", "作者代码意图", "本复现"]
    for j, header in enumerate(headers): set_cell(table.cell(0,j), header, 13.5, bold=True, fill=VERY_LIGHT)
    for i, row in enumerate(rows, 1):
        for j, value in enumerate(row):
            set_cell(table.cell(i,j), value, 12.5, color=BLUE if j == 3 else BLACK, bold=(j == 0 or j == 3), fill=WHITE)
    add_text(slide, 0.95, 5.70, 11.35, 0.36,
             "定位：方法与主要现象复现，不是论文绝对数值逐点复刻。", size=18, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    note_line(slide, "设置不完全同分布；后续所有“论文对照”只解释尺度，不解释严格优劣。", y=6.22, size=13.4)

    # 8 — roadmap
    slide = add_slide(prs, "复现路线：从“能跑”到“可信”", "Reproduction · Journey", SOURCES[7], NOTES[7])
    y = 3.52
    add_line(slide, 1.00, y, 12.20, y, color=MID, width=1.1)
    events = [
        (1.05, "A/B", "基础设施\nsmoke", DARK),
        (2.60, "C-I", "loss↓\n指标全 0", RED),
        (4.10, "Fix 1", "监督契约", BLUE),
        (5.60, "C-II/III", "结构与语义\n分离", DARK),
        (7.22, "C-IV", "覆盖扩容", BLUE),
        (8.78, "D-old", "信号弱\nKL 尖峰", RED),
        (10.30, "Fix 2", "fresh + SEP", BLUE),
        (11.85, "Test", "双解码\nfrozen", BLUE),
    ]
    for index, (x, label, body, color) in enumerate(events):
        dot = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(x-.08), Inches(y-.08), Inches(.16), Inches(.16))
        dot.fill.solid(); dot.fill.fore_color.rgb = color; dot.line.color.rgb = color
        upper = index % 2 == 0
        ty = 2.05 if upper else 3.90
        add_text(slide, x-.55, ty, 1.10, 0.26, label, size=13, color=color, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, x-.66, ty+.35, 1.32, 0.62, body, size=12.5, color=BLACK, align=PP_ALIGN.CENTER)
        add_line(slide, x, y-.12 if upper else y+.12, x, ty+.98 if upper else ty-.08, color=LIGHT, width=0.8)
    note_line(slide, "三次关键转折：监督契约 → 实体覆盖 → GRPO 组内信号。", y=6.08, size=16)

    # 9 — contract failure
    slide = add_slide(prs, "转折一：loss 下降，但监督实际上失效", "Reproduction · Diagnosis", SOURCES[8], NOTES[8])
    add_picture_contain(slide, REPRO_FIG / "phase-c-conditional-loss.png", 0.72, 1.22, 7.18, 4.55)
    add_line(slide, 8.05, 1.28, 8.05, 5.86, color=LIGHT, width=0.8)
    add_text(slide, 8.38, 1.42, 3.85, 0.65, "answers | COND | pattern | SEP\ntarget | END", size=16.5, color=BLACK, bold=True, align=PP_ALIGN.CENTER)
    add_line(slide, 9.60, 2.16, 11.50, 2.16, color=BLUE, width=2.0)
    add_text(slide, 8.38, 2.44, 3.75, 0.34, "active labels = target + END", size=14, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, 8.38, 3.12, 3.72, 0.32, "根因", size=15.5, color=RED, bold=True)
    add_bullets(slide, 8.38, 3.53, 3.78, 0.92, [
        "left-padding backend 状态污染",
        "pair mask 错位；target / END 被屏蔽",
    ], size=13.8, color=BLACK)
    add_text(slide, 8.38, 4.72, 3.72, 0.32, "修复", size=15.5, color=BLUE, bold=True)
    add_bullets(slide, 8.38, 5.10, 3.78, 0.72, [
        "训练固定 right padding；生成局部切换",
        "checkpoint contract + Parse / EOS gate",
    ], size=13.3, color=BLACK)
    note_line(slide, "teacher-forced loss 不能替代自由生成健康检查。", y=6.18, color=RED, size=15.2)

    # 10 — structure versus semantics
    slide = add_slide(prs, "转折二：结构控制与集合语义是两个能力维度", "Reproduction · Evidence", SOURCES[9], NOTES[9])
    add_picture_contain(slide, REPRO_FIG / "phase-c-jaccard-pattern-accuracy.png", 0.80, 1.18, 11.75, 4.85)
    add_rich(slide, 1.08, 6.12, 11.0, 0.40, [
        ("C-II → C-III：", BLACK, True, False),
        ("Pattern Accuracy 0.3185 → 0.5883", BLUE, True, False),
        ("，但 Jaccard 0.3299 → 0.2690", RED, True, False),
    ], size=14.2, align=PP_ALIGN.CENTER)

    # 11 — coverage
    slide = add_slide(prs, "最强单变量证据：训练覆盖扩容", "Reproduction · Evidence", SOURCES[10], NOTES[10])
    add_text(slide, 0.82, 1.26, 11.70, 0.48, "train / pattern    1,024  →  8,000    (7.81×)", size=24, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    rows = [
        ("实体词表覆盖", "35.41%", "81.86%", "+46.45 pp"),
        ("test 未见实体 occurrence", "27.70%", "3.34%", "−24.37 pp"),
        ("greedy Jaccard", "0.2690", "0.5969", "+0.3279"),
        ("Pattern Accuracy", "0.5883", "0.9441", "+0.3558"),
    ]
    table = slide.shapes.add_table(5, 4, Inches(1.18), Inches(2.05), Inches(11.00), Inches(3.35)).table
    widths = [4.10, 2.25, 2.25, 2.40]
    for i, width in enumerate(widths): table.columns[i].width = Inches(width)
    for j, header in enumerate(["指标", "C-III", "C-IV", "变化"]):
        set_cell(table.cell(0,j), header, 14, bold=True, fill=VERY_LIGHT)
    for i, row in enumerate(rows, 1):
        for j, value in enumerate(row):
            set_cell(table.cell(i,j), value, 14.5, color=BLUE if j in (2,3) else BLACK, bold=(j in (0,2,3)), fill=WHITE)
    add_text(slide, 1.10, 5.76, 11.15, 0.30,
             "关系始终 22/22 覆盖；最直接解释是实体监督被补足。", size=16, color=BLACK, bold=True, align=PP_ALIGN.CENTER)
    note_line(slide, "边界：样本增多同时带来更多 optimizer steps，仍需 compute-matched 消融。", y=6.20, size=13.3)

    # 12 — grpo repair
    slide = add_slide(prs, "转折三：高 reward 不等于有效 GRPO 信号", "Reproduction · GRPO", SOURCES[11], NOTES[11])
    add_picture_contain(slide, REPRO_FIG / "phase-d-grpo-signal.png", 0.72, 1.15, 7.65, 4.86)
    add_line(slide, 8.48, 1.28, 8.48, 5.95, color=LIGHT, width=0.8)
    add_picture_contain(slide, REPRO_FIG / "phase-d-semantic-gains.png", 8.68, 1.42, 3.88, 3.33)
    add_text(slide, 8.82, 4.95, 3.58, 0.76,
             "Original：复用 SFT 数据，缺 SEP\nRepaired：fresh RL-only + 显式 SEP", size=13.6, color=BLACK, align=PP_ALIGN.CENTER)
    note_line(slide, "repaired 版本组内 reward std 更高、KL 更稳定；full / test 配对区间均高于 0。", y=6.17, size=13.3)

    # 13 — final result
    slide = add_slide(prs, "最终结果：主要现象已复现，绝对差距仍在", "Reproduction · Result", SOURCES[12], NOTES[12])
    add_picture_contain(slide, REPRO_FIG / "final-metric-comparison.png", 0.70, 1.10, 11.90, 4.92)
    gd = evidence["comparison"]["greedy"]["paired_semantic_average"]
    sd = evidence["comparison"]["sampled"]["paired_semantic_average"]
    add_rich(slide, 0.95, 6.10, 11.45, 0.42, [
        (f"Greedy Δmean = {gd['mean']:+.5f}  CI [{gd['ci95_low']:.5f}, {gd['ci95_high']:.5f}]", BLUE, True, False),
        ("    |    ", GRAY, False, False),
        (f"Sampled Δmean = {sd['mean']:+.5f}  CI [{sd['ci95_low']:.5f}, {sd['ci95_high']:.5f}]", BLUE, True, False),
    ], size=13.2, align=PP_ALIGN.CENTER)
    add_text(slide, 1.30, 6.54, 10.75, 0.20,
             "绝对指标统一使用 0–1 纵轴；论文只作非同设置的尺度参照。", size=10.5, color=GRAY, align=PP_ALIGN.CENTER)

    # 14 — conclusion
    slide = add_slide(prs, "结论、证据边界与下一步", "Reproduction · Conclusion", SOURCES[13], NOTES[13])
    columns = [
        (0.82, "已经得到的证据", ["pattern 控制稳定有效", "GRPO 对集合语义产生配对正增益", "监督、覆盖与 RL 信号均可审计"]),
        (4.55, "不能扩大的结论", ["不是论文绝对数值的严格复刻", "单 seed 不是跨训练显著性", "fresh data 与 SEP 尚未拆分"]),
        (8.35, "下一步", ["12 层 depth-only", "多 seed 与 fresh RL shard", "fresh × SEP 2×2", "新 untouched test"]),
    ]
    for index, (x, heading, items) in enumerate(columns):
        if index:
            add_line(slide, x-.42, 1.52, x-.42, 5.70, color=LIGHT, width=0.8)
        add_text(slide, x, 1.50, 3.15, 0.34, heading, size=18, color=BLUE if index==0 else BLACK, bold=True)
        add_bullets(slide, x, 2.20, 3.20, 2.90, items, size=15.2, color=BLACK, gap=13)
    add_text(slide, 1.02, 5.82, 11.25, 0.52,
             "核心方法现象已复现；结构控制已强，实体—关系实例化后的集合语义仍是主要差距。", size=17, color=BLUE, bold=True, align=PP_ALIGN.CENTER)

    # 15 — backup patterns and metrics
    slide = add_slide(prs, "备份：13 种 pattern、指标与奖励量程", "Backup", SOURCES[14], NOTES[14])
    add_picture_contain(slide, PAPER_FIG / "paper-fig4-patterns.png", 0.78, 1.32, 6.32, 3.62)
    add_line(slide, 7.32, 1.38, 7.32, 5.75, color=LIGHT, width=0.8)
    add_text(slide, 7.68, 1.48, 4.65, 1.75,
             "Jaccard = |A∩O| / |A∪O|\nDice = 2|A∩O| / (|A|+|O|)\nOverlap = |A∩O| / min(|A|,|O|)", size=18, color=BLACK)
    add_text(slide, 7.68, 3.73, 4.65, 0.90,
             "R_sem ∈ [0, 2]\nR_cond ∈ [0, 1]", size=18, color=BLUE, bold=True)
    add_text(slide, 7.68, 5.02, 4.58, 0.64,
             "Overlap 在一集合包含另一集合时可为 1，\n因此必须与 Jaccard / Dice 联合解读。", size=13.5, color=RED)
    note_line(slide, "13 类清单来自 pattern_filtered.csv；不包含 pattern_table.csv 中额外的 3p。", y=6.18, size=13.2)

    # 16 — backup config matrix
    slide = add_slide(prs, "备份：论文、代码意图与本复现的设置差异", "Backup", SOURCES[15], NOTES[15])
    rows = [
        ("模型深度", "12 层", "GPT2_6 / num_layers:6", "字段未覆盖 n_layer", "显式 n_layer=6"),
        ("Optimizer", "AdamW", "未写", "实际使用 Adam", "Adam / AdamW(GRPO)"),
        ("SFT LR", "1e-5", "5e-5", "5e-5", "5e-5"),
        ("Batch", "256", "WN:160", "160", "160"),
        ("Epoch", "400+50", "50", "315 resume / total 50", "50+50"),
        ("Warm-up", "50 / 5 epoch", "5 step", "约 5 step", "按 optimizer step"),
        ("算力", "4×A6000", "—", "—", "1×L20"),
    ]
    table = slide.shapes.add_table(8, 5, Inches(0.55), Inches(1.28), Inches(12.22), Inches(4.94)).table
    widths = [1.42, 1.90, 2.68, 3.05, 3.17]
    for i, width in enumerate(widths): table.columns[i].width = Inches(width)
    for j, header in enumerate(["项目", "论文正文", "作者代码意图", "公开代码实际语义", "本复现"]):
        set_cell(table.cell(0,j), header, 12.2, bold=True, fill=VERY_LIGHT)
    for i, row in enumerate(rows, 1):
        for j, value in enumerate(row):
            set_cell(table.cell(i,j), value, 11.4, color=BLUE if j==4 else BLACK, bold=(j in (0,4)), fill=WHITE)
    note_line(slide, "只陈述可验证差异；本复现采用显式、自洽、可审计的实现。", y=6.34, size=13.5)

    # 17 — backup full metrics
    slide = add_slide(prs, "备份：完整双解码指标与测试隔离边界", "Backup", SOURCES[16], NOTES[16])
    headers = ["解码 / 模型", "Jaccard", "Dice", "Overlap", "PA", "Smatch", "Parse", "EOS"]
    rows = []
    for mode, label in (("greedy", "Greedy"), ("sampled", "Sampled")):
        for who, who_label in (("baseline", "SFT parent"), ("candidate", "Repaired")):
            data = evidence["comparison"][mode][who]
            rows.append((
                f"{label} · {who_label}", data["jaccard"], data["dice"], data["overlap"],
                data["condition_accuracy"], data["smatch"], data["parse_ok"], data["eos_rate"],
            ))
    table = slide.shapes.add_table(5, 8, Inches(0.56), Inches(1.28), Inches(12.20), Inches(2.94)).table
    widths = [2.25, 1.35, 1.22, 1.25, 1.12, 1.22, 1.22, 1.15]
    for i, width in enumerate(widths): table.columns[i].width = Inches(width)
    for j, header in enumerate(headers): set_cell(table.cell(0,j), header, 11.5, bold=True, fill=VERY_LIGHT)
    for i, row in enumerate(rows, 1):
        repaired = "Repaired" in row[0]
        for j, value in enumerate(row):
            text = value if j == 0 else f"{value:.5f}"
            set_cell(table.cell(i,j), text, 11.2, color=BLUE if repaired else BLACK, bold=repaired, fill=WHITE)
    add_text(slide, 0.78, 4.68, 5.75, 0.86,
             "conditional-best → epoch 50（训练规则）\nphase-d-parent → epoch 45（研究目标）", size=14.2, color=BLACK, bold=True, align=PP_ALIGN.CENTER)
    add_line(slide, 6.65, 4.60, 6.65, 5.75, color=LIGHT, width=0.8)
    add_text(slide, 6.92, 4.65, 5.20, 0.96,
             "parent bake-off 使用过同一 test split；\n终点未读 test，但端到端不是 untouched holdout。", size=13.6, color=RED, bold=True, align=PP_ALIGN.CENTER)
    note_line(slide, "paired bootstrap：固定模型在 1,664 条记录上的重采样，不是跨 seed 方差。", y=6.14, size=13.3)

    return prs


def write_markdown(prs: Presentation):
    output = [
        "# CtrlHGen：可控逻辑假设生成与复现进展（极简重制版）",
        "",
        "> 主讲第 1–14 页，约 14–15 分钟；第 15–17 页为答疑备份。",
        "",
    ]
    for index, (slide, note, source) in enumerate(zip(prs.slides, NOTES, SOURCES), 1):
        title = next(
            shape.text.splitlines()[0]
            for shape in slide.shapes
            if getattr(shape, "has_text_frame", False) and shape.text.strip()
        )
        output += [
            f"## {index}. {title}", "", f"- 建议用时：{note['time']}", "",
            "### 核心讲述", "", note["talk"], "",
            "### 转场", "", note["transition"], "",
            "### 证据来源", "", source, "",
            "### 必要限定", "", note["caveat"], "",
            "### 可能追问", "", note["qa"], "",
        ]
    MD_OUT.write_text("\n".join(output), encoding="utf-8")


def validate(prs: Presentation):
    assert len(prs.slides) == 17
    assert prs.slide_width == SLIDE_W and prs.slide_height == SLIDE_H
    required = ["【建议用时】", "【核心讲述】", "【转场】", "【证据来源】", "【必要限定】", "【可能追问】"]
    for number, slide in enumerate(prs.slides, 1):
        notes = slide.notes_slide.notes_text_frame.text
        assert all(item in notes for item in required), number
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0, (number, shape.name)
            assert shape.left + shape.width <= SLIDE_W + Inches(0.02), (number, shape.name, "right")
            assert shape.top + shape.height <= SLIDE_H + Inches(0.02), (number, shape.name, "bottom")
    for path in [
        PAPER_FIG / "paper-fig1-controllability.png",
        PAPER_FIG / "paper-fig2-challenges.png",
        PAPER_FIG / "paper-fig3-framework.png",
        PAPER_FIG / "paper-fig4-patterns.png",
        PAPER_FIG / "paper-table3-reward-ablation.png",
        REPRO_FIG / "phase-c-conditional-loss.png",
        REPRO_FIG / "phase-c-jaccard-pattern-accuracy.png",
        REPRO_FIG / "phase-d-grpo-signal.png",
        REPRO_FIG / "phase-d-semantic-gains.png",
        REPRO_FIG / "final-metric-comparison.png",
    ]:
        assert path.exists(), path


def render_pdf_preview(prs: Presentation):
    """Render a review PDF from the same object model, including picture shapes."""
    pages = []
    for slide in prs.slides:
        image = Image.new("RGB", (evidence_source.PREVIEW_W, evidence_source.PREVIEW_H), "white")
        draw = ImageDraw.Draw(image)
        for shape in slide.shapes:
            rect = (
                evidence_source._px(shape.left), evidence_source._px(shape.top),
                evidence_source._px(shape.left + shape.width), evidence_source._px(shape.top + shape.height),
            )
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                with Image.open(io.BytesIO(shape.image.blob)) as source:
                    source = source.convert("RGB")
                    source = source.resize((max(1, rect[2]-rect[0]), max(1, rect[3]-rect[1])), Image.Resampling.LANCZOS)
                    image.paste(source, (rect[0], rect[1]))
                continue
            if shape.has_table:
                evidence_source._draw_table(draw, shape.table, rect)
                continue
            shape_type = getattr(shape.shape_type, "name", str(shape.shape_type))
            if shape_type == "LINE":
                draw.line(rect, fill=evidence_source._rgb(shape.line.color, (160,160,160)), width=max(1, int((shape.line.width.pt if shape.line.width else 1)*120/72)))
                continue
            if shape_type == "AUTO_SHAPE":
                fill = evidence_source._rgb(shape.fill.fore_color, (255,255,255))
                outline = evidence_source._rgb(shape.line.color, (180,180,180))
                if getattr(getattr(shape, "auto_shape_type", None), "name", "") == "OVAL":
                    draw.ellipse(rect, fill=fill, outline=outline, width=1)
                else:
                    draw.rectangle(rect, fill=fill, outline=outline, width=1)
            if getattr(shape, "has_text_frame", False):
                evidence_source._draw_text_frame(draw, shape.text_frame, rect)
        pages.append(image)
    pages[0].save(PDF_OUT, "PDF", resolution=120, save_all=True, append_images=pages[1:], title="CtrlHGen minimalist teacher briefing")
    return pages


def main():
    evidence = evidence_source.load_evidence()
    prs = build_deck(evidence)
    validate(prs)
    write_markdown(prs)
    prs.save(PPTX_OUT)
    pages = render_pdf_preview(prs)
    # Reload the saved file and verify the archive instead of trusting memory.
    assert zipfile.ZipFile(PPTX_OUT).testzip() is None
    reloaded = Presentation(PPTX_OUT)
    assert len(reloaded.slides) == 17
    assert len(PdfReader(PDF_OUT).pages) == 17
    assert all("【证据来源】" in slide.notes_slide.notes_text_frame.text for slide in reloaded.slides)
    assert len(re.findall(r"^## \d+\.", MD_OUT.read_text(encoding="utf-8"), flags=re.M)) == 17
    print(f"Wrote {PPTX_OUT}")
    print(f"Wrote {PDF_OUT}")
    print(f"Wrote {MD_OUT}")
    print(f"Rendered {len(pages)} slides")


if __name__ == "__main__":
    main()
