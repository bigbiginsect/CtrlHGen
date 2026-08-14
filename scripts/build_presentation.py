#!/usr/bin/env python3
"""Build the CtrlHGen paper + reproduction progress presentation (2026-08-12).

Generates: worklogs/reproduction/briefings/CtrlHGen-汇报-2026-08-12.pptx
Usage: python3 scripts/build_presentation.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "worklogs" / "reproduction" / "figures"
OUT = ROOT / "worklogs" / "reproduction" / "briefings" / "CtrlHGen-汇报-2026-08-12.pptx"

# ---------------------------------------------------------------- style ----
DARK = RGBColor(0x1F, 0x3B, 0x6E)     # deep blue bar
ACCENT = RGBColor(0x2E, 0x75, 0xB6)   # accent blue
TEXT = RGBColor(0x26, 0x26, 0x26)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xEE, 0xF3, 0xFA)
GOOD = RGBColor(0x1E, 0x7A, 0x3C)
BAD = RGBColor(0xB0, 0x30, 0x30)
FONT = "Microsoft YaHei"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def set_run(run, text, size=16, bold=False, color=TEXT, italic=False):
    run.text = text
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    f.name = FONT
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        e = rPr.find(qn(tag))
        if e is None:
            e = rPr.makeelement(qn(tag), {})
            rPr.append(e)
        e.set("typeface", FONT)


def add_bar(slide, title, part=None):
    bar = slide.shapes.add_shape(1, 0, 0, SLIDE_W, Inches(0.95))  # 1 = rectangle
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK
    bar.line.fill.background()
    bar.shadow.inherit = False
    tf = bar.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.45)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    if part:
        set_run(p.add_run(), part + "  ", size=15, bold=True, color=RGBColor(0x9D, 0xC3, 0xE6))
    set_run(p.add_run(), title, size=24, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    return bar


def add_footer(slide, idx, total):
    box = slide.shapes.add_textbox(Inches(12.3), Inches(7.08), Inches(0.9), Inches(0.35))
    p = box.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    set_run(p.add_run(), f"{idx} / {total}", size=10, color=GRAY)


def add_bullets(slide, left, top, width, height, items, line_spacing=1.12):
    """items: list of (level, text, opts) where opts may set size/bold/color."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for level, text, opts in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.line_spacing = line_spacing
        p.space_after = Pt(opts.get("space_after", 6))
        prefix = "" if level == 0 else ("• " if level == 1 else "– ")
        if level == 2:
            p.level = 1
        set_run(
            p.add_run(),
            prefix + text,
            size=opts.get("size", 16),
            bold=opts.get("bold", False),
            color=opts.get("color", TEXT),
        )
    return box


def add_picture_fit(slide, path, left, top, max_w, max_h, px_w, px_h):
    """Add picture centered in the (max_w, max_h) box, preserving aspect."""
    ratio = px_w / px_h
    w = max_w
    h = Emu(int(w / ratio))
    if h > max_h:
        h = max_h
        w = Emu(int(h * ratio))
    x = Emu(int(left + (max_w - w) / 2))
    y = Emu(int(top + (max_h - h) / 2))
    pic = slide.shapes.add_picture(str(path), x, y, w, h)
    # thin border
    pic.line.color.rgb = RGBColor(0xC9, 0xC9, 0xC9)
    pic.line.width = Pt(0.75)
    return pic


def add_table(slide, left, top, width, height, rows, col_widths=None,
              header_fill=DARK, font_size=13, highlight_rows=()):
    n_r, n_c = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_r, n_c, left, top, width, height)
    table = shape.table
    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            table.columns[i].width = Emu(int(width * cw / total))
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.margin_top = cell.margin_bottom = Pt(1)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
            bold = r == 0 or r in highlight_rows
            color = RGBColor(0xFF, 0xFF, 0xFF) if r == 0 else TEXT
            set_run(p.add_run(), str(val), size=font_size, bold=bold, color=color)
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_fill
            elif r in highlight_rows:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT
    return table


def add_flow(slide, left, top, boxes, box_w=Inches(3.3), box_h=Inches(1.5),
             gap=Inches(0.55), font_size=13):
    """Horizontal flow: list of (title, [lines]) boxes joined by arrows."""
    x = left
    for i, (title, lines) in enumerate(boxes):
        shp = slide.shapes.add_shape(5, x, top, box_w, box_h)  # 5 = rounded rect
        shp.adjustments[0] = 0.08
        shp.fill.solid()
        shp.fill.fore_color.rgb = LIGHT
        shp.line.color.rgb = ACCENT
        shp.line.width = Pt(1.25)
        shp.shadow.inherit = False
        tf = shp.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = Inches(0.1)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        set_run(p.add_run(), title, size=font_size + 1, bold=True, color=DARK)
        for ln in lines:
            p2 = tf.add_paragraph()
            p2.alignment = PP_ALIGN.CENTER
            p2.line_spacing = 1.0
            set_run(p2.add_run(), ln, size=font_size - 1, color=TEXT)
        if i < len(boxes) - 1:
            ar_h = Inches(0.4)
            ar = slide.shapes.add_shape(
                33, Emu(int(x + box_w + Inches(0.06))),  # 33 = right arrow
                Emu(int(top + (box_h - ar_h) / 2)),
                Emu(int(gap - Inches(0.12))), ar_h)
            ar.fill.solid()
            ar.fill.fore_color.rgb = ACCENT
            ar.line.fill.background()
            ar.shadow.inherit = False
        x = Emu(int(x + box_w + gap))


prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]
TOTAL = 14


def new_slide():
    return prs.slides.add_slide(BLANK)


# ============================================================== S1 title ====
s = new_slide()
bg = s.shapes.add_shape(1, 0, 0, SLIDE_W, SLIDE_H)
bg.fill.solid()
bg.fill.fore_color.rgb = DARK
bg.line.fill.background()
bg.shadow.inherit = False
box = s.shapes.add_textbox(Inches(0.9), Inches(1.7), Inches(11.5), Inches(4.5))
tf = box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.LEFT
set_run(p.add_run(), "论文讲解与复现进展汇报", size=20, bold=True, color=RGBColor(0x9D, 0xC3, 0xE6))
p = tf.add_paragraph()
p.space_before = Pt(18)
set_run(p.add_run(), "Controllable Logical Hypothesis Generation for", size=34, bold=True,
        color=RGBColor(0xFF, 0xFF, 0xFF))
p = tf.add_paragraph()
set_run(p.add_run(), "Abductive Reasoning in Knowledge Graphs", size=34, bold=True,
        color=RGBColor(0xFF, 0xFF, 0xFF))
p = tf.add_paragraph()
p.space_before = Pt(14)
set_run(p.add_run(), "ICLR 2026 · Yisen Gao, Jiaxin Bai, Tianshi Zheng, Yangqiu Song (HKUST) 等", size=16,
        color=RGBColor(0xD6, 0xDF, 0xEC))
p = tf.add_paragraph()
p.space_before = Pt(30)
set_run(p.add_run(), "汇报日期：2026-08-12", size=15, color=RGBColor(0xD6, 0xDF, 0xEC))

# ===================================================== S2 任务背景 =========
s = new_slide()
add_bar(s, "任务背景：知识图谱上的溯因推理", part="论文 · 01")
add_bullets(s, Inches(0.5), Inches(1.2), Inches(12.4), Inches(3.1), [
    (1, "溯因推理（Abductive Reasoning）：为观察到的现象寻找「最可信的解释」，是三大推理类型之一", {}),
    (1, "KG 溯因推理（AbductiveKGR, ACL 2024）：给定观察实体集合 O，自动生成解释这些实体的"
        "复杂逻辑假设 H（一阶逻辑，含 ∃, ∧, ∨, ¬）", {}),
    (1, "核心痛点——缺乏可控性：即便较小的 DBpedia50（约 2.5 万实体）上，平均每个观察也有 "
        "约 50 个合理假设；更大的图上数量爆炸式增长", {"bold": True}),
    (1, "用户实际需要按兴趣/意图过滤后的假设，而非无差别生成", {}),
])
box = s.shapes.add_shape(5, Inches(0.5), Inches(4.35), Inches(12.33), Inches(2.6))
box.adjustments[0] = 0.05
box.fill.solid(); box.fill.fore_color.rgb = LIGHT
box.line.color.rgb = ACCENT; box.line.width = Pt(1)
box.shadow.inherit = False
tf = box.text_frame; tf.word_wrap = True
tf.margin_left = tf.margin_right = Inches(0.25); tf.margin_top = Inches(0.15)
p = tf.paragraphs[0]
set_run(p.add_run(), "本文提出两个可控性维度（论文 Fig. 1）", size=16, bold=True, color=DARK)
p = tf.add_paragraph(); p.space_before = Pt(8)
set_run(p.add_run(), "• 语义内容控制：指定关注侧面——对三个疾病实体的观察，可分别从病理（pathology）、"
                     "治疗（treatment）、易感人群（susceptibles）等侧面生成假设", size=15)
p = tf.add_paragraph(); p.space_before = Pt(6)
set_run(p.add_run(), "• 结构复杂度控制：调整逻辑假设的结构（合取项数量、是否含否定等），"
                     "控制信息粒度与信息密度", size=15)
add_footer(s, 2, TOTAL)

# ============================================== S3 问题定义与两大挑战 =======
s = new_slide()
add_bar(s, "问题定义与两大挑战", part="论文 · 02")
add_bullets(s, Inches(0.5), Inches(1.15), Inches(12.4), Inches(1.7), [
    (1, "可控溯因推理（Definition 3.1）：给定 KG G、观察 O、控制条件 C，生成假设 H 使 "
        "(1) 结论集 [[H]]G 与 O 尽量吻合（最可信解释）；(2) H 满足 C 指定的约束", {}),
    (1, "假设以析取范式表示：H(V?) = ∃V₁..Vk : e₁ ∨ … ∨ eₙ；开放世界假设下仅用观察到的图训练", {}),
])
y = Inches(2.95)
for i, (title, lines) in enumerate([
    ("挑战一：Hypothesis Space Collapse（假设空间坍缩）",
     ["假设变长（1p → 2p → 3p）时，每个观察对应的",
      "合理参考假设数量急剧下降，",
      "模型难以学到复杂逻辑结构"]),
    ("挑战二：Reward Oversensitivity（奖励过度敏感）",
     ["此前方法用 Jaccard 作 RL 奖励：微小错误即断崖",
      "下跌——44/45 个实体正确也只得 2/45 ≈ 0.044，",
      "导致训练不稳定、引导方向错误"]),
]):
    box = s.shapes.add_shape(5, Emu(int(Inches(0.5) + i * Inches(6.35))), y,
                             Inches(6.05), Inches(3.6))
    box.adjustments[0] = 0.05
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = BAD; box.line.width = Pt(1.25)
    box.shadow.inherit = False
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.22); tf.margin_top = Inches(0.18)
    p = tf.paragraphs[0]
    set_run(p.add_run(), title, size=17, bold=True, color=BAD)
    for ln in lines:
        p2 = tf.add_paragraph(); p2.space_before = Pt(8); p2.line_spacing = 1.15
        set_run(p2.add_run(), ln, size=15)
add_footer(s, 3, TOTAL)

# ===================================================== S4 方法总览 ==========
s = new_slide()
add_bar(s, "CtrlHGen 框架总览（论文 Fig. 3）", part="论文 · 03")
add_flow(s, Inches(0.45), Inches(1.35), [
    ("Step 1 数据构建", ["按 13 种逻辑 pattern 采样假设", "子逻辑分解数据增强", "→ 缓解假设空间坍缩"]),
    ("Step 2 条件监督训练", ["12 层 decoder-only Transformer", "阶段一：无条件训练", "阶段二：按条件分别微调"]),
    ("Step 3 GRPO 强化微调", ["平滑语义奖励 + 条件遵循奖励", "组内相对优化（k=4）", "→ 缓解奖励过度敏感"]),
], box_w=Inches(3.75), box_h=Inches(1.95), gap=Inches(0.45), font_size=13)
add_bullets(s, Inches(0.5), Inches(3.65), Inches(12.4), Inches(0.6), [
    (0, "控制条件 C：两类五种形式（Sec 3.3）", {"size": 16, "bold": True, "color": DARK}),
])
add_table(s, Inches(0.5), Inches(4.25), Inches(12.33), Inches(2.6), [
    ["类别", "条件", "含义", "表示"],
    ["语义聚焦", "specific-entity", "假设须包含指定实体", "实体 token"],
    ["语义聚焦", "specific-relation", "假设须包含指定关系", "关系 token"],
    ["结构约束", "pattern", "严格限定逻辑模式（1p/2i/inp…）", "Lisp-like 语言 + 操作符 token"],
    ["结构约束", "entity-number", "假设恰含 n 个实体", "特殊 token [ne]"],
    ["结构约束", "relation-number", "假设恰含 n 个关系", "特殊 token [nr]"],
], col_widths=[2, 2.6, 5, 3.4], font_size=13)
add_footer(s, 4, TOTAL)

# ============================================ S5 关键技术一：子逻辑分解 =====
s = new_slide()
add_bar(s, "关键技术一：子逻辑分解数据增强", part="论文 · 04")
add_bullets(s, Inches(0.5), Inches(1.2), Inches(12.4), Inches(5.6), [
    (1, "动机：复杂 pattern（长链、含析取/否定）的参考假设稀缺 → 训练信号不足（假设空间坍缩）", {}),
    (1, "做法：对复杂模式 P 下的 (H, O) 训练对，递归分解出子模式 P_sub 对应的子假设 H_sub，"
        "在图上执行得到子观察 O_sub，构造额外训练对", {}),
    (2, "{(H_sub, O_sub)} = { ( f(P_sub, H), [[ f(P_sub, H) ]]G ) | P_sub ⊆ P }（论文公式 3）",
        {"color": ACCENT, "bold": True}),
    (1, "例：inp 模式（交-否-投影）可分解为两个 2p 子假设；论文对 up / 3in / pni / pin / inp "
        "五种复杂模式做分解", {}),
    (1, "为什么有效：子假设与原假设结构、语义高度相关，模型可从简单子模式渐进学会复杂逻辑"
        "（类似课程学习）", {}),
    (1, "消融证据（Fig. 5，DBpedia50，pattern 条件）：显著提升 Jaccard，含析取/否定的复杂模式"
        "提升最大，且条件遵循 Accuracy 基本不变 → 提升来自对逻辑结构的理解增强",
        {"bold": True, "color": GOOD}),
])
add_footer(s, 5, TOTAL)

# ======================================== S6 关键技术二：平滑奖励 + GRPO ====
s = new_slide()
add_bar(s, "关键技术二：平滑语义奖励 + GRPO", part="论文 · 05")
box = s.shapes.add_shape(5, Inches(0.5), Inches(1.25), Inches(12.33), Inches(2.5))
box.adjustments[0] = 0.05
box.fill.solid(); box.fill.fore_color.rgb = LIGHT
box.line.color.rgb = ACCENT; box.line.width = Pt(1.25)
box.shadow.inherit = False
tf = box.text_frame; tf.word_wrap = True
tf.margin_left = tf.margin_right = Inches(0.25); tf.margin_top = Inches(0.15)
p = tf.paragraphs[0]
set_run(p.add_run(), "奖励函数设计（公式 5–7）", size=16, bold=True, color=DARK)
p = tf.add_paragraph(); p.space_before = Pt(8); p.line_spacing = 1.2
set_run(p.add_run(), "R_sem = 1.0·Jaccard + 0.5·Dice + 0.5·Overlap   —— Dice/Overlap 更平滑，"
                     "缓解 Jaccard 的断崖式下跌", size=15)
p = tf.add_paragraph(); p.line_spacing = 1.2
set_run(p.add_run(), "R_cond ∈ {0, 1}   —— 二值条件遵循奖励：生成假设满足条件 C 得 1，否则 0", size=15)
p = tf.add_paragraph(); p.line_spacing = 1.2
set_run(p.add_run(), "R̂ = 0.5·R_sem + 0.5·R_cond", size=15, bold=True, color=ACCENT)
add_bullets(s, Inches(0.5), Inches(4.05), Inches(12.4), Inches(2.9), [
    (1, "优化算法 GRPO（Group Relative Policy Optimization）：对每个观察采样一组 k=4 个假设，"
        "组内奖励归一化，配合重要性采样比、KL 惩罚与梯度裁剪", {}),
    (1, "为什么用 GRPO：溯因推理本身要求生成多个可信假设，需要整组质量提升而非单个输出最优", {}),
    (1, "消融结论（Table 3，WN18RR）：仅用 Jaccard 作奖励太严苛、阻碍收敛；去掉条件遵循奖励则 "
        "Accuracy 93.5 → 68.3 暴跌——以微小语义损失换来大幅可控性提升", {"bold": True, "color": GOOD}),
])
add_footer(s, 6, TOTAL)

# ===================================================== S7 论文实验结果 ======
s = new_slide()
add_bar(s, "论文主要实验结果", part="论文 · 06")
add_bullets(s, Inches(0.5), Inches(1.15), Inches(12.4), Inches(1.9), [
    (1, "设置：DBpedia50 / WN18RR / FB15k-237，8:1:1 划分，开放世界增量图；指标为 "
        "Jaccard / Dice / Overlap / 条件遵循 Accuracy / Smatch", {}),
    (1, "主表：加控制条件后语义相似度普遍不降反升（条件提供额外引导）；多数条件遵循率 > 80%，"
        "FB15k-237 条件遵循平均 96.6", {"bold": True}),
])
add_bullets(s, Inches(0.5), Inches(3.0), Inches(6.4), Inches(0.5), [
    (0, "与顶级 LLM 对比（Table 2，FB15k-237，五条件平均）", {"size": 15, "bold": True, "color": DARK}),
])
add_table(s, Inches(0.5), Inches(3.55), Inches(6.4), Inches(3.1), [
    ["模型", "Jaccard", "Accuracy"],
    ["GPT-4o", "2.4", "77.5"],
    ["DeepSeek-V3 + RAG", "5.3", "76.6"],
    ["GPT-5 (Thinking)", "18.7", "92.8"],
    ["CtrlHGen", "64.3", "96.6"],
], col_widths=[3.2, 1.6, 1.6], font_size=13, highlight_rows=(4,))
add_bullets(s, Inches(7.3), Inches(3.55), Inches(5.55), Inches(3.2), [
    (1, "LLM 失败原因：长 2-hop 子图文本难以结构化理解；内部知识与 KG 知识冲突；混淆输入/输出边",
        {"size": 14}),
    (1, "结论：KG 溯因推理上，专用结构化模型显著优于通用 LLM", {"size": 14, "bold": True}),
    (1, "其他消融：子逻辑分解优于 Logic-Gen 增强与"
        " AbductiveKGR+condition token（Jaccard 70.1 vs 69.5 / 68.2）", {"size": 14}),
])
add_footer(s, 7, TOTAL)

# ================================================= S8 复现目标与范围 ========
s = new_slide()
add_bar(s, "复现目标与范围", part="复现 · 01")
add_bullets(s, Inches(0.5), Inches(1.2), Inches(12.4), Inches(3.0), [
    (1, "复现切片：WN18RR 数据集 + pattern 控制条件 + seed 42（论文全貌为 3 个 KG × 无条件及 5 类条件）",
        {"bold": True}),
    (1, "完整流水线：KG 采样去重 → 无条件 SFT → 条件 SFT（answers COND pattern SEP target END）"
        "→ GRPO → greedy / sampled 双解码 frozen test", {}),
    (1, "指标与论文一致：Jaccard / Dice / Overlap / Pattern Accuracy / Smatch，"
        "另加 Parse / EOS 等生成健康指标", {}),
])
box = s.shapes.add_shape(5, Inches(0.5), Inches(4.35), Inches(12.33), Inches(2.55))
box.adjustments[0] = 0.05
box.fill.solid(); box.fill.fore_color.rgb = LIGHT
box.line.color.rgb = ACCENT; box.line.width = Pt(1.25)
box.shadow.inherit = False
tf = box.text_frame; tf.word_wrap = True
tf.margin_left = tf.margin_right = Inches(0.25); tf.margin_top = Inches(0.15)
p = tf.paragraphs[0]
set_run(p.add_run(), "定位声明", size=16, bold=True, color=DARK)
p = tf.add_paragraph(); p.space_before = Pt(8); p.line_spacing = 1.2
set_run(p.add_run(), "• 最终模型：随机初始化的 6 层 GPT-2（hidden 768，12 heads），单卡 NVIDIA L20", size=15)
p = tf.add_paragraph(); p.line_spacing = 1.2
set_run(p.add_run(), "• 论文配置：12 层 Transformer，4×A6000 48GB", size=15)
p = tf.add_paragraph(); p.line_spacing = 1.2
set_run(p.add_run(), "• 因此目标是「方法与核心现象的高质量复现」，而非论文绝对数值的逐点复刻",
        size=15, bold=True, color=ACCENT)
add_footer(s, 8, TOTAL)

# ===================================================== S9 复现历程 ==========
s = new_slide()
add_bar(s, "复现历程：Phase A → D（08-08 ~ 08-12）", part="复现 · 02")
add_bullets(s, Inches(0.45), Inches(1.2), Inches(5.6), Inches(5.7), [
    (0, "Phase A 工程基线", {"size": 15, "bold": True, "color": DARK, "space_after": 2}),
    (1, "严格配置 + 确定性划分 + checkpoint 契约；79 项测试全过", {"size": 13}),
    (0, "Phase B tiny 冒烟", {"size": 15, "bold": True, "color": DARK, "space_after": 2}),
    (1, "最小数据全链路跑通（验证 artifact 契约）", {"size": 13}),
    (0, "Phase C 四轮 SFT 演进", {"size": 15, "bold": True, "color": DARK, "space_after": 2}),
    (1, "C-I 全零（监督失效）→ C-II 修复契约 → C-III 对齐作者超参 → C-IV 扩大训练数据全面跃升",
        {"size": 13}),
    (0, "Phase D 三轮 GRPO", {"size": 15, "bold": True, "color": DARK, "space_after": 2}),
    (1, "原版增益微弱 → 诊断出两处方法缺陷 → repaired full 稳定收敛、增益显著", {"size": 13}),
    (0, "右图：四轮 SFT 的 validation 轨迹", {"size": 13, "color": GRAY}),
])
add_picture_fit(s, FIG / "phase-c-conditional-validation.png",
                Inches(6.25), Inches(1.4), Inches(6.7), Inches(5.3), 2071, 817)
add_footer(s, 9, TOTAL)

# =========================================== S10 关键问题与解决（上） ========
s = new_slide()
add_bar(s, "关键问题与解决方案（上）", part="复现 · 03")
add_bullets(s, Inches(0.45), Inches(1.2), Inches(5.6), Inches(5.7), [
    (0, "① 监督契约失效（最关键 bug）", {"size": 15, "bold": True, "color": BAD, "space_after": 2}),
    (1, "validation 生成把 tokenizer 切成 left padding 后 backend 状态泄漏进 checkpoint，"
        "label mask 错位——target 与 END 全被 mask，监督实际无效（C-I 零结果的根因）", {"size": 13}),
    (1, "修复：训练固定 right padding；active labels 严格等于 target+END；checkpoint v2 拒绝旧存档",
        {"size": 13, "color": GOOD}),
    (0, "② 作者公开材料不自洽", {"size": 15, "bold": True, "color": BAD, "space_after": 2}),
    (1, "论文：12 层 / AdamW / LR 1e-5 / batch 256；作者仓库：GPT2_6 / LR 5e-5 / batch 160，"
        "且 num_layers 字段实际不生效", {"size": 13}),
    (1, "处置：采用显式 n_layer=6 的自洽实现，差异作为解释指标差距的边界", {"size": 13, "color": GOOD}),
    (0, "右图：C-I（灰）conditional loss 下降是假信号", {"size": 13, "color": GRAY}),
])
add_picture_fit(s, FIG / "phase-c-loss-curves.png",
                Inches(6.25), Inches(1.4), Inches(6.7), Inches(5.3), 1810, 834)
add_footer(s, 10, TOTAL)

# =========================================== S11 关键问题与解决（下） ========
s = new_slide()
add_bar(s, "关键问题与解决方案（下）", part="复现 · 04")
add_bullets(s, Inches(0.45), Inches(1.2), Inches(5.6), Inches(5.7), [
    (0, "③ 训练覆盖不足", {"size": 15, "bold": True, "color": BAD, "space_after": 2}),
    (1, "C-III→C-IV 单变量实验：train/pattern 1,024→8,000，实体覆盖率 35%→82%，"
        "test 未见实体占比 27.7%→3.3%，五项指标全面跃升", {"size": 13}),
    (0, "④ train–test 泄漏", {"size": 15, "bold": True, "color": BAD, "space_after": 2}),
    (1, "预检发现一条跨 split 监督 tuple 重合 → 训练前停止；冻结 test/valid，只重采样冲突 train",
        {"size": 13, "color": GOOD}),
    (0, "⑤ GRPO 信号稀疏 + KL 尖峰", {"size": 15, "bold": True, "color": BAD, "space_after": 2}),
    (1, "根因：prompt 缺 SEP + 复用已高度拟合的 SFT 数据（组内 advantage 稀疏）；"
        "修复后增益翻 3–4 倍，KL 稳定在 0.1–0.4", {"size": 13, "color": GOOD}),
    (0, "右图：原流程 KL 跨数量级尖峰 vs repaired 稳定", {"size": 13, "color": GRAY}),
])
add_picture_fit(s, FIG / "phase-d-training-dynamics.png",
                Inches(6.25), Inches(1.4), Inches(6.7), Inches(5.3), 2071, 817)
add_footer(s, 11, TOTAL)

# ===================================================== S12 当前结果 =========
s = new_slide()
add_bar(s, "当前结果：与论文对照（WN18RR + pattern，greedy frozen test）", part="复现 · 05")
add_picture_fit(s, FIG / "final-metric-comparison.png",
                Inches(0.45), Inches(1.15), Inches(6.9), Inches(3.35), 1653, 758)
add_table(s, Inches(7.6), Inches(1.35), Inches(5.35), Inches(3.0), [
    ["设置", "J", "D", "O", "PA", "Sm."],
    ["论文 w/o RL", ".715", ".758", ".837", ".815", ".790"],
    ["我们 SFT", ".603", ".652", ".722", ".943", ".817"],
    ["我们 GRPO", ".637", ".687", ".761", ".954", ".820"],
    ["论文完整版", ".770", ".808", ".868", ".935", ".833"],
], col_widths=[2.3, 1, 1, 1, 1, 1], font_size=12, highlight_rows=(3,))
add_bullets(s, Inches(0.5), Inches(4.75), Inches(12.4), Inches(2.3), [
    (1, "已对齐/超出：Pattern Accuracy 0.954 超过论文 0.935；Smatch 0.820 接近论文 0.833；"
        "Parse 0.99+、EOS 1.0", {"size": 15, "bold": True, "color": GOOD}),
    (1, "未对齐：Jaccard / Dice / Overlap 距论文约 0.13 / 0.12 / 0.11——合理解释为 6 层容量、"
        "更少 SFT epoch、论文未公开的采样与生成参数、复杂 pattern 偏弱（待消融）", {"size": 15}),
    (1, "注意：paired CI ≠ 跨 seed 显著性；test 因 parent 选点并非完全 untouched holdout",
        {"size": 13, "color": GRAY}),
])
add_footer(s, 12, TOTAL)

# ================================================ S13 RL 有效性证据 =========
s = new_slide()
add_bar(s, "RL 有效性证据：GRPO 增益统计显著", part="复现 · 06")
add_bullets(s, Inches(0.45), Inches(1.2), Inches(5.6), Inches(5.7), [
    (1, "相对 SFT parent（epoch 45）paired bootstrap：", {"size": 14}),
    (2, "greedy 三项均值 +0.036，95% CI [0.025, 0.048]", {"size": 14, "bold": True, "color": GOOD}),
    (2, "sampled +0.044，95% CI [0.029, 0.060]", {"size": 14, "bold": True, "color": GOOD}),
    (1, "frozen test 全程仅做单次双解码评估，所有预冻结规则通过（避免选点偏倚）", {"size": 14}),
    (1, "修复（SEP + fresh RL-only 数据）后增益为原流程的 3–4 倍，同时否定了"
        "「6 层太浅训不动 GRPO」的解释", {"size": 14}),
    (1, "训练全程稳定：4,095 秒无中断，KL ∈ [0.09, 0.38]", {"size": 14}),
    (0, "右图：三场景（pilot / full valid / frozen test）增益与 CI", {"size": 13, "color": GRAY}),
])
add_picture_fit(s, FIG / "phase-d-semantic-gains.png",
                Inches(6.25), Inches(1.4), Inches(6.7), Inches(5.3), 1389, 772)
add_footer(s, 13, TOTAL)

# ===================================================== S14 总结与下一步 =====
s = new_slide()
add_bar(s, "总结与下一步", part="复现 · 07")
add_bullets(s, Inches(0.5), Inches(1.2), Inches(6.1), Inches(0.5), [
    (0, "已复现的核心现象", {"size": 17, "bold": True, "color": DARK}),
])
add_bullets(s, Inches(0.5), Inches(1.75), Inches(6.1), Inches(4.6), [
    (1, "控制条件有效：Pattern Accuracy 0.954 超论文，加条件不损害语义质量", {"size": 14}),
    (1, "GRPO 稳定改善集合语义：三项均值增益 CI 严格为正，复现了论文 Table 3 的 RL 增益方向",
        {"size": 14}),
    (1, "子逻辑分解增强、平滑奖励等方法要点均已在流水线中落地", {"size": 14}),
    (1, "工程上建立了完整的数据 / checkpoint / 测试隔离契约流水线（可重建、可审计）", {"size": 14}),
])
add_bullets(s, Inches(6.95), Inches(1.2), Inches(6.0), Inches(0.5), [
    (0, "下一步工作", {"size": 17, "bold": True, "color": DARK}),
])
add_bullets(s, Inches(6.95), Inches(1.75), Inches(6.0), Inches(4.6), [
    (1, "多 seed replicate（目前全部正式结果仅 seed 42）", {"size": 14}),
    (1, "12 层 depth-only 消融（Phase C-V），用新协议评估容量 / 论文忠实度", {"size": 14}),
    (1, "分离 fresh data 与 SEP 两项修复的各自贡献", {"size": 14}),
    (1, "加强复杂 pattern（pin / inp / union）；尝试更大 group 或更高分辨率 reward", {"size": 14}),
    (1, "扩展至 FB15k-237 / DBpedia50 与其余 4 类控制条件", {"size": 14}),
])
box = s.shapes.add_shape(5, Inches(0.5), Inches(6.15), Inches(12.33), Inches(0.85))
box.adjustments[0] = 0.12
box.fill.solid(); box.fill.fore_color.rgb = LIGHT
box.line.color.rgb = ACCENT; box.line.width = Pt(1)
box.shadow.inherit = False
tf = box.text_frame; tf.word_wrap = True
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.margin_left = tf.margin_right = Inches(0.25)
p = tf.paragraphs[0]
set_run(p.add_run(), "总体结论：核心方法现象已成功复现；J/D/O 距论文约 0.11–0.13 的差距主要受模型规模与"
                     "训练实体覆盖限制，缩小差距的路径明确。", size=14, bold=True, color=DARK)
add_footer(s, 14, TOTAL)

# ---------------------------------------------------------------- save -----
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"saved: {OUT}")

# ------------------------------------------------------------- verify ------
check = Presentation(OUT)
n = len(check.slides)
assert n == TOTAL, f"expected {TOTAL} slides, got {n}"
for i, slide in enumerate(check.slides, 1):
    pics = sum(1 for sh in slide.shapes if sh.shape_type == 13)
    print(f"slide {i:2d}: shapes={len(slide.shapes):2d} pictures={pics}")
expected_pics = {9: 1, 10: 1, 11: 1, 12: 1, 13: 1}
for i, slide in enumerate(check.slides, 1):
    pics = sum(1 for sh in slide.shapes if sh.shape_type == 13)
    assert pics == expected_pics.get(i, 0), f"slide {i}: unexpected picture count {pics}"
print("verification passed")
