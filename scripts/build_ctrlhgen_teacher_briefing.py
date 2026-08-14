#!/usr/bin/env python3
"""Build the CtrlHGen paper/reproduction briefing deck.

The deck is intentionally independent from the older presentation builder.  It
reads the compact reproduction artifacts, checks every reported headline value,
adds speaker notes, and writes both the editable PPTX and a Markdown script.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader
from pptx import Presentation
from pptx.chart.data import ChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "worklogs/reproduction/reports/reproduction-report-2026-08-12.md"
PDF = ROOT / "paper/Controllable_Logical_Hypothesis_Generation.pdf"
DATA = ROOT / "worklogs/reproduction/report-data"
OUT_STEM = ROOT / "worklogs/reproduction/briefings/CtrlHGen-论文讲解与复现进展-2026-08-13"
PPTX_OUT = OUT_STEM.with_suffix(".pptx")
SCRIPT_OUT = OUT_STEM.with_suffix(".md")
PDF_OUT = OUT_STEM.with_suffix(".pdf")

SLIDE_W = Inches(13.333333)
SLIDE_H = Inches(7.5)

FONT_CN = "Microsoft YaHei"
FONT_EN = "Arial"

BG = RGBColor(248, 247, 243)
WHITE = RGBColor(255, 255, 255)
NAVY = RGBColor(25, 48, 74)
BLUE = RGBColor(49, 91, 137)
TEAL = RGBColor(34, 139, 128)
TEAL_LIGHT = RGBColor(219, 239, 234)
AMBER = RGBColor(217, 154, 54)
AMBER_LIGHT = RGBColor(248, 236, 210)
RED = RGBColor(188, 72, 72)
RED_LIGHT = RGBColor(246, 224, 222)
INK = RGBColor(37, 46, 55)
MUTED = RGBColor(95, 105, 113)
LINE = RGBColor(210, 213, 211)
PALE_BLUE = RGBColor(224, 234, 244)
GRAY = RGBColor(135, 143, 150)
LIGHT_GRAY = RGBColor(236, 237, 234)


def load_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def approx(actual: float, expected: float, tol: float = 1e-9) -> None:
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=tol), (actual, expected)


def load_evidence() -> dict:
    comparison = load_json("worklogs/reproduction/report-data/phase-d/repaired-test/comparison.json")
    pilot = load_json("worklogs/reproduction/report-data/phase-d/repaired-pilot/validation-comparison.json")
    full = load_json("worklogs/reproduction/report-data/phase-d/repaired-full/validation-comparison.json")
    rollout = load_json("worklogs/reproduction/report-data/phase-d/repaired-full/rollout-signal-audit.json")
    original_state = load_json("worklogs/reproduction/report-data/phase-d/original/trainer_state.json")
    repaired_state = load_json("worklogs/reproduction/report-data/phase-d/repaired-full/trainer_state.json")
    report_text = REPORT.read_text(encoding="utf-8")

    # Verify the compact evidence copies before consuming them.
    checksum_lines = (DATA / "sha256sums.txt").read_text(encoding="utf-8").splitlines()
    checked = 0
    for line in checksum_lines:
        if not line.strip():
            continue
        expected, rel = line.split(maxsplit=1)
        artifact = DATA / rel.lstrip("* ")
        assert artifact.exists(), artifact
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected, artifact
        checked += 1
    assert checked == 26

    # Frozen-test values must all come from the same paired comparison artifact.
    g = comparison["greedy"]
    s = comparison["sampled"]
    approx(g["baseline"]["jaccard"], 0.6030406494773461)
    approx(g["candidate"]["jaccard"], 0.6366699750041087)
    approx(g["paired_semantic_average"]["mean"], 0.03625220739493916)
    approx(g["paired_semantic_average"]["ci95_low"], 0.025062183961685653)
    approx(g["paired_semantic_average"]["ci95_high"], 0.047728655081578523)
    approx(s["paired_semantic_average"]["mean"], 0.044262838708516695)
    approx(s["paired_semantic_average"]["ci95_low"], 0.028635905226668142)
    approx(s["paired_semantic_average"]["ci95_high"], 0.05990859927292234)
    assert comparison["decision"]["health_pass"] is True
    assert comparison["decision"]["post_test_training_or_checkpoint_selection_allowed"] is False
    assert g["paired_semantic_average"]["bootstrap_samples"] == 10_000
    for mode in ("greedy", "sampled"):
        delta = comparison[mode]["delta"]
        approx(
            delta["semantic_average"],
            sum(delta[k] for k in ("jaccard", "dice", "overlap")) / 3,
            tol=1e-12,
        )
        assert comparison[mode]["paired_semantic_average"]["ci95_low"] > 0

    # Sample count is fixed by 13 patterns x 128 records.
    with (ROOT / "akgr/metadata/pattern_filtered.csv").open(encoding="utf-8", newline="") as fh:
        patterns = list(csv.DictReader(fh))
    assert len(patterns) == 13
    assert {r["pattern_abbr"] for r in patterns} == {
        "1p", "2p", "ip", "inp", "up", "2i", "pi", "2in", "pni", "pin", "2u", "3i", "3in"
    }
    test_n = len(patterns) * 128
    assert test_n == 1664

    # Paper Table 3 values are checked against the repository PDF, not only hard-coded.
    paper_page_10 = PdfReader(str(PDF)).pages[9].extract_text() or ""
    normalized = re.sub(r"\s+", " ", paper_page_10)
    for token in ["71.5", "75.8", "83.7", "81.5", "79.0", "77.0", "80.8", "86.8", "93.5", "83.3"]:
        assert token in normalized, f"Paper Table 3 token missing: {token}"

    # Key coverage and intermediate-run values are independently present in the signed-off report.
    for token in ["35.41%", "81.86%", "27.70%", "3.34%", "0.3299", "0.2690", "0.5969", "0.9441"]:
        assert token in report_text, f"Report value missing: {token}"

    old_logs = [x for x in original_state["log_history"] if "reward_std" in x]
    new_logs = [x for x in repaired_state["log_history"] if "reward_std" in x]
    assert max(x["kl"] for x in old_logs) > 1700
    assert 0.09 <= min(x["kl"] for x in new_logs) < max(x["kl"] for x in new_logs) <= 0.38
    approx(rollout["zero_reward_variance_group_rate"], 0.60546875)
    approx(rollout["mean_unique_strings_per_group"], 2.23828125)

    return {
        "comparison": comparison,
        "pilot": pilot,
        "full": full,
        "rollout": rollout,
        "old_logs": old_logs,
        "new_logs": new_logs,
        "patterns": patterns,
        "test_n": test_n,
        "paper": {
            "sft": {"jaccard": .715, "dice": .758, "overlap": .837, "condition_accuracy": .815, "smatch": .790},
            "full": {"jaccard": .770, "dice": .808, "overlap": .868, "condition_accuracy": .935, "smatch": .833},
        },
    }


def set_cell_text(cell, text: str, size: float = 15, color: RGBColor = INK, bold: bool = False,
                  align: PP_ALIGN = PP_ALIGN.CENTER, font: str = FONT_CN) -> None:
    cell.text = ""
    cell.margin_left = Inches(.08)
    cell.margin_right = Inches(.08)
    cell.margin_top = Inches(.04)
    cell.margin_bottom = Inches(.04)
    p = cell.text_frame.paragraphs[0]
    p.alignment = align
    p.space_after = Pt(0)
    r = p.add_run()
    r.text = str(text)
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color


def set_text(shape, text: str, size: float = 20, color: RGBColor = INK, bold: bool = False,
             align: PP_ALIGN = PP_ALIGN.LEFT, font: str = FONT_CN, valign=MSO_ANCHOR.MIDDLE,
             margin: float = .08) -> None:
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    p = tf.paragraphs[0]
    p.alignment = align
    p.space_after = Pt(0)
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color


def add_box(slide, x, y, w, h, fill=WHITE, line=LINE, radius=True, text: str | None = None,
            size=18, color=INK, bold=False, align=PP_ALIGN.LEFT):
    kind = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(1)
    if text is not None:
        set_text(shape, text, size=size, color=color, bold=bold, align=align)
    return shape


def add_text(slide, x, y, w, h, text, size=20, color=INK, bold=False,
             align=PP_ALIGN.LEFT, font=FONT_CN, valign=MSO_ANCHOR.MIDDLE, margin=.02):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    set_text(shape, text, size=size, color=color, bold=bold, align=align, font=font, valign=valign, margin=margin)
    return shape


def add_rich_text(slide, x, y, w, h, runs, size=18, align=PP_ALIGN.LEFT, fill=None,
                  line=None, margin=.12, valign=MSO_ANCHOR.MIDDLE):
    if fill is None:
        shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    else:
        shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        shape.fill.solid(); shape.fill.fore_color.rgb = fill
        shape.line.color.rgb = line or fill
    tf = shape.text_frame
    tf.clear(); tf.word_wrap = True; tf.vertical_anchor = valign
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(margin)
    p = tf.paragraphs[0]; p.alignment = align; p.space_after = Pt(0)
    for text, color, bold, font in runs:
        r = p.add_run(); r.text = text; r.font.name = font; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return shape


def add_bullets(slide, x, y, w, h, items, size=17, color=INK, bullet_color=TEAL,
                spacing=7, level_indent=.22):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame; tf.clear(); tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(.02); tf.margin_top = tf.margin_bottom = Inches(.02)
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(spacing); p.level = 0
        p.left_margin = Inches(level_indent); p.first_line_indent = Inches(-.16)
        r1 = p.add_run(); r1.text = "● "; r1.font.name = FONT_CN; r1.font.size = Pt(size-3); r1.font.color.rgb = bullet_color
        r2 = p.add_run(); r2.text = item; r2.font.name = FONT_CN; r2.font.size = Pt(size); r2.font.color.rgb = color
    return shape


def add_arrow(slide, x1, y1, x2, y2, color=BLUE, width=2.2, dashed=False):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    line.line.color.rgb = color; line.line.width = Pt(width); line.line.end_arrowhead = True
    if dashed:
        line.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return line


def add_tag(slide, x, y, text, fill=PALE_BLUE, color=BLUE, w=None):
    w = w or max(.82, .18 * len(text) + .28)
    return add_box(slide, x, y, w, .34, fill=fill, line=fill, text=text, size=10.5, color=color, bold=True, align=PP_ALIGN.CENTER)


def add_header(slide, number: int, title: str, section: str, backup=False) -> None:
    add_text(slide, .58, .31, 9.95, .48, title, size=27, color=NAVY, bold=True)
    add_text(slide, 10.72, .31, 1.98, .38, "BACKUP" if backup else section, size=10.5,
             color=AMBER if backup else TEAL, bold=True, align=PP_ALIGN.RIGHT, font=FONT_EN)
    accent = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(.58), Inches(.91), Inches(.72), Inches(.055))
    accent.fill.solid(); accent.fill.fore_color.rgb = TEAL; accent.line.fill.background()
    add_text(slide, 12.56, 7.03, .25, .18, str(number), size=9.5, color=MUTED, align=PP_ALIGN.RIGHT, font=FONT_EN)


def add_footer(slide, source: str, label: str) -> None:
    color = BLUE if label == "论文结论" else TEAL if label == "复现实证" else AMBER
    add_tag(slide, .58, 6.87, label, fill=RGBColor(238, 241, 242), color=color, w=.96)
    add_text(slide, 1.66, 6.86, 10.47, .22, source, size=8.6, color=MUTED, font=FONT_CN)


def set_background(slide, color=BG) -> None:
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = color


def add_notes(slide, notes: dict, source: str) -> None:
    body = (
        f"【建议用时】{notes['time']}\n\n"
        f"【核心讲述】\n{notes['talk']}\n\n"
        f"【转场】{notes['transition']}\n\n"
        f"【证据来源】{source}\n\n"
        f"【必要限定】{notes['caveat']}\n\n"
        f"【可能追问】{notes['qa']}"
    )
    tf = slide.notes_slide.notes_text_frame
    tf.text = body


def add_slide(prs, title: str, section: str, source: str, label: str, notes: dict, backup=False):
    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_background(slide)
    add_header(slide, len(prs.slides), title, section, backup=backup)
    add_footer(slide, source, label)
    add_notes(slide, notes, source)
    return slide


def chart_style(chart, legend=True, value_axis=True):
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.name = FONT_CN; chart.legend.font.size = Pt(10)
    chart.chart_title.text_frame.paragraphs[0].runs[0].font.name = FONT_CN if chart.has_title else FONT_CN
    chart.category_axis.tick_labels.font.name = FONT_CN; chart.category_axis.tick_labels.font.size = Pt(10.5)
    chart.category_axis.format.line.color.rgb = LINE
    if value_axis:
        chart.value_axis.tick_labels.font.name = FONT_EN; chart.value_axis.tick_labels.font.size = Pt(9.5)
        chart.value_axis.format.line.color.rgb = LINE
        chart.value_axis.has_major_gridlines = True
        chart.value_axis.major_gridlines.format.line.color.rgb = RGBColor(229, 230, 226)


NOTES = [
    {"time":"0:30", "talk":"今天汇报两件事：先用六页讲清 CtrlHGen 的任务和机制，再重点说明我们如何把一个会跑但监督无效的公开流程，修成可审计、可复核的实验链路。最终定位是主要现象复现，而不是论文绝对数值逐点复刻。", "transition":"先从这个任务究竟在生成什么开始。", "caveat":"不要把 CtrlHGen 说成自然语言解释模型；它生成可执行的一阶逻辑查询。", "qa":"问：为什么值得复现？答：它把生成、逻辑执行与用户控制放进同一闭环，且公开代码与论文设置存在值得核验的差异。"},
    {"time":"1:00", "talk":"KGQA 是给查询找答案；这里反过来，给出观察实体集合，寻找能解释这些实体的查询。控制条件解决多解性：用户既可以限定语义焦点，也可以限定逻辑结构和复杂度。执行生成查询后得到答案集，再与观察比较，因此输出是可验证的。", "transition":"反向生成比正向问答多了两个特别棘手的问题。", "caveat":"“最合理”通过集合相似度近似，不代表恢复唯一真实因果解释。", "qa":"问：一个观察能有多少解释？答：论文称 DBpedia50 平均每个观察约有 50 个合理假设。"},
    {"time":"1:00", "talk":"第一个难点是长逻辑候选空间坍缩：候选数和 Jaccard 随长度下降。第二个难点是奖励过敏：只错一个谓词可能把答案集从目标两项扩成 45 项，Jaccard 只有 0.044，探索阶段几乎没有平滑方向。", "transition":"CtrlHGen 的所有设计都可以映射回这两个难点。", "caveat":"图中百分数按论文原图展示；Jaccard 2/45 是论文示例。", "qa":"问：为什么不用 token-level 奖励？答：任务质量由逻辑执行后的答案集合定义，token 相似并不能保证执行语义。"},
    {"time":"1:00", "talk":"框架分三步。先在 KG 上按 13 个 pattern 构造观察—假设对，并通过子逻辑分解增加复杂结构的学习信号；再做无条件和条件两阶段 SFT；最后让模型生成一组候选，在 KG 执行后以语义和条件奖励做 GRPO。", "transition":"先看第一项创新：如何让复杂逻辑可学。", "caveat":"无条件 SFT 使用增广数据；条件 SFT 直接学习 observation+condition 到 hypothesis。", "qa":"问：KG 在推理时做检索吗？答：这里核心用途是执行生成查询、形成可验证反馈，不是把子图文本塞给模型。"},
    {"time":"0:55", "talk":"子逻辑分解不是任意造样本，而是从同一个复杂假设中抽取结构和语义高度相关的简单子问题。例如 inp 可拆成两个 2p，并在 KG 上分别执行得到子观察。这相当于逻辑课程学习。控制接口则覆盖五类：pattern、实体数、关系数、指定实体、指定关系。", "transition":"有了可控 SFT，论文再用执行奖励做强化。", "caveat":"论文只对 up、3in、pni、pin、inp 五类复杂 pattern 做分解；不是任意 FOL 分解器。", "qa":"问：一个统一模型支持所有控制吗？答：论文 Sec.3.3 更接近按控制类型分别微调，不能过度概括成单模型任意切换。"},
    {"time":"1:10", "talk":"语义奖励把严格的 Jaccard 与更平滑的 Dice、Overlap 组合，再加二值条件遵循。GRPO 对同一 prompt 的四个候选做组内标准化，所以关键不是绝对 reward 高，而是组内能否拉开差异。Table 3 显示：去条件奖励语义略升，但 Accuracy 从 93.5 掉到 68.3；说明这是显式的双目标权衡。", "transition":"下面从论文方法切到我们的复现边界。", "caveat":"Overlap 对完全包含的过生成可给 1；且 R_sem 最大 2、Rcond 最大 1，alpha=0.5 在原始量程并非严格等权。这是基于公式的分析。", "qa":"问：为什么三个集合指标都要报？答：它们容忍度不同，联合观察可避免单一指标误导。"},
    {"time":"0:50", "talk":"我们的范围刻意收窄为 WN18RR+pattern、seed 42、单张 L20。最终采用作者代码意图的自洽 6 层 GPT-2，而论文正文是 12 层、4 张 A6000。由于模型、epoch、优化器和算力均不完全一致，目标是验证方法链路和主要现象。", "transition":"在这个边界下，我们的复现经历了三次关键转折。", "caveat":"不能把 6 层版本称为论文原设置，也不能把作者公开脚本逐行照跑视为可靠基线。", "qa":"问：为何不直接严格照论文？答：论文与公开配置/代码语义互有矛盾，且单卡资源不等价；我们选择可解释、可审计的设置。"},
    {"time":"0:55", "talk":"A/B 先把配置、hash、checkpoint、逐样本评测和 tiny 链路打通。C-I 表面完成但结果全零；修复契约后 C-II 能生成；C-III 对齐作者意图但暴露结构与语义分离；C-IV 扩容找到覆盖瓶颈；Phase D 再修复 RL 数据和 prompt，最终冻结测试。", "transition":"第一处转折解释了为什么 loss 会骗人。", "caveat":"时间线中的每个阶段都有独立 checkpoint 和证据，失败 checkpoint 禁止复用。", "qa":"问：是不是反复看 test 调参？答：训练和 GRPO 终点不读 test，但 parent 的 epoch45/50 bake-off 使用过同一 split，后面会明确列为局限。"},
    {"time":"1:15", "talk":"C-I 的 conditional loss 能降，甚至 overfit 到约 0.006，但 EOS 和 parse 都是零。根因是 generation 临时切 left padding 后，fast-tokenizer backend 状态被保存；pair mask 随后错位，把 target 和 END 全部屏蔽。修复后我们固定训练 right padding、generation 局部 left padding，主动清理 backend，并让 active labels 精确等于 target+END。", "transition":"契约修复让模型会说合法逻辑，但还没解决说得对不对。", "caveat":"C-I 同时存在预算不足，不能把所有失败只归因于实现 bug；但 conditional 零结果有确定的契约根因。", "qa":"问：为什么 loss 还能下降？答：模型在错误的 active token 上优化，低 loss 并不代表目标逻辑受到监督。"},
    {"time":"0:55", "talk":"C-II 修复后 Jaccard 0.3299、PA 0.3185。C-III 改成作者意图版后，PA 升到 0.5883、Smatch 也升，但 Jaccard 反降到 0.2690。模型更会生成指定的形状，却没有正确实例化实体和关系。这说明控制能力和答案集合语义必须分开看。", "transition":"于是我们做了目前最干净的规模比较。", "caveat":"C-II 到 C-III 同时改变层数、optimizer、LR 等，不能归因给单个因素。", "qa":"问：PA 已提高为何 Jaccard 下降？答：pattern 只约束拓扑，具体实体/关系 token 错一个就会改变执行答案集。"},
    {"time":"1:10", "talk":"C-III 到 C-IV 固定模型、优化器、LR、batch、seed 和 50+50 epoch，只把每 pattern 训练样本从 1024 扩到 8000。实体覆盖升到 81.86%，test 未见实体降到 3.34%，Jaccard 和 PA 同时跃升。关系覆盖始终 22/22，因此最直接解释是实体监督终于充分。", "transition":"SFT 到位以后，原始 GRPO 为什么仍只有小幅增益？", "caveat":"这是强单变量证据，但数据增加也带来更多 optimizer steps，尚未做 compute-matched 消融，不能宣称覆盖是唯一原因。", "qa":"问：是否有数据泄漏？答：预检发现一条冲突后训练前停止；冻结 valid/test，只重采样冲突 train，最终跨 split 完整监督重合为零。"},
    {"time":"1:15", "talk":"原 Phase D 复用 SFT 已见数据，且 raw prompt 末尾缺 SEP。completion 容易得到相同 reward，组内优势稀疏；它的绝对 reward 更高却不等于 GRPO 信号更好。repaired 版本联合使用 fresh RL-only 数据和显式 SEP，reward std 提高、KL 稳定，并从 pilot 的不确定趋势升级到 full validation 和 frozen test 的严格正区间。", "transition":"最后把最终模型与 SFT parent 和论文放在同一尺度上。", "caveat":"fresh data 与 SEP 是联合修复，当前不能分离各自贡献；60.55% 的组仍为零 reward 方差，信号并非完全充足。", "qa":"问：旧 KL=1703 是否模型发散？答：参数和 optimizer 均有限，后续窗口回落；属于 reverse-KL 聚合离群，保留现场后从完整 checkpoint 恢复。"},
    {"time":"1:20", "talk":"在同一 1664 条 frozen test 上，repaired GRPO 相对 epoch-45 SFT parent 的 Jaccard、Dice、Overlap 全部提升；greedy 三项均值加 0.03625，sampled 加 0.04426，区间均高于零，同时 PA、Smatch、parse、EOS 没退化。对论文只作尺度参照：我们的 PA 略高、Smatch 接近，但集合指标仍低 0.11 到 0.13。", "transition":"所以最终结论需要同时包含‘做到了什么’和‘还不能说什么’。", "caveat":"paired bootstrap 描述固定模型、固定样本的不确定性，不是跨 seed 显著性；论文设置也不是同分布对照。", "qa":"问：为何 parent 是 epoch45？答：epoch50 是训练规则 best；epoch45 是为 Phase D 研究目标选择的 parent，两者指针和语义分开保留。"},
    {"time":"1:10", "talk":"可以较强地说：条件控制有效，修复后的 GRPO 能继续改善集合语义；监督契约、实体覆盖和组内 RL 信号是本次复现的三个关键因素。不能说已经严格复现论文绝对值。下一步最有信息量的是 depth-only 12 层、多 seed、fresh×SEP 的 2×2 消融，以及重新保留 untouched test。", "transition":"主讲到这里，后面三页作为答疑备份。", "caveat":"当前只有 WN18RR+pattern，不能外推到论文全部数据集与控制类型。", "qa":"问：优先做哪一个实验？答：先做 12 层 depth-only 和多 seed，它们最直接回答剩余绝对差距和稳定性。"},
    {"time":"答疑", "talk":"这页列出复现真实使用的 13 个 pattern，并给出集合指标公式。重点说明 Overlap 的包含偏好，以及论文奖励量程：R_sem 最大 2，R_cond 最大 1。", "transition":"若追问实现忠实度，切到下一页。", "caveat":"13 pattern 来自 pattern_filtered.csv；pattern_table.csv 额外含 3p，不能混用。", "qa":"问：否定在开放世界是否可靠？答：缺失不是假，带否定查询的执行质量受当前可观测图限制。"},
    {"time":"答疑", "talk":"论文、作者仓库表达意图和公开代码实际语义并不完全一致。我们的选择不是逐行复制矛盾脚本，而是明确写入 6 层、Adam 和 5 optimizer-step warm-up，并完整记录语义 hash。", "transition":"若追问最终统计口径，切到最后一页。", "caveat":"对公开材料只陈述可验证差异，不推断作者动机。", "qa":"问：为什么作者的 num_layers 不生效？答：标准 GPT-2 配置使用 n_layer，公开字段 num_layers 不会覆盖它。"},
    {"time":"答疑", "talk":"这页给出 greedy 与 sampled 全量指标，并解释测试隔离。epoch50 是训练期 best，epoch45 是 Phase D parent；parent bake-off 曾使用该 test split，因此最终 paired 改善仍可解释，但整条模型选择链不能称为 untouched holdout。", "transition":"回到结论页结束。", "caveat":"评测 repaired terminal 后已禁止继续训练或 checkpoint 选择。", "qa":"问：CI 能否代表论文差距显著？答：不能，它只针对当前固定模型在 1664 条记录上的配对差异。"},
]


def build_deck(e: dict) -> Presentation:
    prs = Presentation(); prs.slide_width = SLIDE_W; prs.slide_height = SLIDE_H

    # 1 — Cover
    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_background(slide)
    band = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0), Inches(0), SLIDE_W, Inches(.16))
    band.fill.solid(); band.fill.fore_color.rgb = TEAL; band.line.fill.background()
    add_tag(slide, .72, .62, "ICLR 2026 论文 · 复现进展", fill=TEAL_LIGHT, color=TEAL, w=2.05)
    add_text(slide, .72, 1.42, 10.8, 1.25, "CtrlHGen：\n可控逻辑假设生成与复现进展", size=32, color=NAVY, bold=True, valign=MSO_ANCHOR.TOP)
    add_text(slide, .76, 3.02, 8.5, .48, "从论文机制到可审计实验链路", size=20, color=BLUE)
    # Abstract visual: answers -> logic -> execution loop
    add_box(slide, .78, 4.15, 2.25, 1.05, fill=PALE_BLUE, line=PALE_BLUE, text="观察实体集合\n{o1, o2, …}", size=18, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    add_arrow(slide, 3.12, 4.68, 4.05, 4.68, color=BLUE)
    add_box(slide, 4.16, 4.15, 2.35, 1.05, fill=WHITE, line=TEAL, text="生成逻辑假设 H\n投影 · 交 · 并 · 否定", size=16.5, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_arrow(slide, 6.63, 4.68, 7.56, 4.68, color=TEAL)
    add_box(slide, 7.67, 4.15, 2.35, 1.05, fill=TEAL_LIGHT, line=TEAL_LIGHT, text="在知识图谱执行\n[H]G ≈ O", size=18, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    add_box(slide, 10.45, 1.15, 2.15, 4.55, fill=NAVY, line=NAVY)
    add_text(slide, 10.72, 1.55, 1.6, .45, "汇报主线", size=15, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    for i, (num, txt) in enumerate([("01", "论文核心思想"), ("02", "三次复现转折"), ("03", "结果、边界与下一步")]):
        add_text(slide, 10.72, 2.38+i*.92, .42, .35, num, size=13, color=RGBColor(126, 206, 193), bold=True, font=FONT_EN)
        add_text(slide, 11.16, 2.31+i*.92, 1.16, .48, txt, size=13.5, color=WHITE, bold=True)
    add_text(slide, .76, 6.37, 8.5, .32, "汇报人：bigbiginsect  ·  2026-08-13", size=12, color=MUTED)
    cover_source = "论文 p.1 Abstract；worklogs/reproduction/reports/reproduction-report-2026-08-12.md 摘要"
    add_footer(slide, cover_source, "我们的分析")
    add_notes(slide, NOTES[0], cover_source)

    # 2 — Task
    slide = add_slide(prs, "任务：从答案集合反求逻辑查询", "PAPER · TASK", "论文 p.2 Fig.1；p.4–5 Eq.(1–2), Definition 3.1", "论文结论", NOTES[1])
    add_tag(slide, .7, 1.15, "正向：KGQA", w=1.15)
    add_box(slide, .72, 1.65, 3.72, 1.22, fill=WHITE, text="逻辑查询 q", size=21, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    add_arrow(slide, 4.55, 2.26, 5.35, 2.26)
    add_box(slide, 5.48, 1.65, 2.28, 1.22, fill=PALE_BLUE, line=PALE_BLUE, text="答案集合 O", size=21, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    add_tag(slide, .7, 3.35, "反向：可控溯因", fill=TEAL_LIGHT, color=TEAL, w=1.42)
    add_box(slide, .72, 3.85, 3.72, 1.34, fill=TEAL_LIGHT, line=TEAL_LIGHT, text="观察实体 O + 控制 C", size=21, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    add_arrow(slide, 4.55, 4.52, 5.35, 4.52, color=TEAL)
    add_box(slide, 5.48, 3.85, 2.28, 1.34, fill=WHITE, line=TEAL, text="解释查询 H", size=21, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_box(slide, 8.16, 1.34, 4.28, 4.28, fill=WHITE, line=LINE)
    add_text(slide, 8.5, 1.62, 3.58, .42, "两个同时满足的目标", size=18, color=NAVY, bold=True)
    add_rich_text(slide, 8.48, 2.26, 3.58, .78, [("① ", TEAL, True, FONT_CN), ("执行语义", NAVY, True, FONT_CN), ("  [H]G ≈ O", INK, False, FONT_EN)], size=19, fill=TEAL_LIGHT, line=TEAL_LIGHT)
    add_rich_text(slide, 8.48, 3.28, 3.58, .78, [("② ", BLUE, True, FONT_CN), ("条件遵循", NAVY, True, FONT_CN), ("  H 满足 C", INK, False, FONT_CN)], size=19, fill=PALE_BLUE, line=PALE_BLUE)
    add_text(slide, 8.52, 4.45, 3.5, .72, "语义焦点：指定实体 / 关系\n结构粒度：pattern / 实体数 / 关系数", size=15.5, color=MUTED)
    add_rich_text(slide, .72, 5.72, 11.72, .7, [("一句话：", TEAL, True, FONT_CN), ("生成的是可执行、可验证的逻辑程序，而不是自然语言解释。", NAVY, True, FONT_CN)], size=18, fill=WHITE, line=LINE)

    # 3 — Challenges
    slide = add_slide(prs, "论文的两个核心难题", "PAPER · CHALLENGES", "论文 p.3 Fig.2；数值按原图重绘", "论文结论", NOTES[2])
    add_box(slide, .68, 1.22, 6.1, 5.28, fill=WHITE)
    add_tag(slide, .98, 1.5, "难题 1", fill=AMBER_LIGHT, color=AMBER, w=.8)
    add_text(slide, 1.9, 1.43, 4.25, .48, "长逻辑的假设空间坍缩", size=21, color=NAVY, bold=True)
    xs = [1.25, 3.43, 5.58]; cand = [55.36, 44.31, 12.97]; jac = [86.2, 81.8, 60.6]
    for i, x in enumerate(xs):
        h = cand[i] / 55.36 * 1.75
        bar = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(x), Inches(4.52-h), Inches(.62), Inches(h))
        bar.fill.solid(); bar.fill.fore_color.rgb = BLUE; bar.line.fill.background()
        add_text(slide, x-.25, 4.58, 1.15, .35, ["短", "中", "长"][i], size=13, color=MUTED, align=PP_ALIGN.CENTER)
        add_text(slide, x-.18, 4.13-h, 1.02, .32, f"{cand[i]:.2f}", size=12, color=BLUE, bold=True, align=PP_ALIGN.CENTER, font=FONT_EN)
        cy = 2.1 + (86.2-jac[i])*.045
        circ = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(x+.11), Inches(cy), Inches(.38), Inches(.38))
        circ.fill.solid(); circ.fill.fore_color.rgb = AMBER; circ.line.color.rgb = AMBER
        add_text(slide, x-.18, cy-.38, 1.0, .3, f"{jac[i]:.1f}", size=12, color=AMBER, bold=True, align=PP_ALIGN.CENTER, font=FONT_EN)
        if i:
            prev_y = 2.1 + (86.2-jac[i-1])*.045
            line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(xs[i-1]+.49), Inches(prev_y+.19), Inches(x+.11), Inches(cy+.19))
            line.line.color.rgb = AMBER; line.line.width = Pt(2)
    add_text(slide, 1.05, 5.38, 5.3, .58, "候选数 55.36 → 12.97\nJaccard 86.2 → 60.6", size=15, color=INK, bold=True, align=PP_ALIGN.CENTER)
    add_box(slide, 7.05, 1.22, 5.62, 5.28, fill=WHITE)
    add_tag(slide, 7.35, 1.5, "难题 2", fill=RED_LIGHT, color=RED, w=.8)
    add_text(slide, 8.27, 1.43, 3.9, .48, "假设奖励过敏", size=21, color=NAVY, bold=True)
    add_box(slide, 7.58, 2.22, 4.56, 1.02, fill=PALE_BLUE, line=PALE_BLUE, text="目标观察 |O| = 2", size=18, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    add_arrow(slide, 9.86, 3.34, 9.86, 3.91, color=RED)
    add_box(slide, 7.58, 4.02, 4.56, 1.02, fill=RED_LIGHT, line=RED_LIGHT, text="错一个谓词 → |[H]G| = 45", size=17, color=RED, bold=True, align=PP_ALIGN.CENTER)
    add_rich_text(slide, 7.58, 5.35, 4.56, .62, [("Jaccard = ", MUTED, False, FONT_EN), ("2 / 45 = 0.044", RED, True, FONT_EN)], size=20, fill=WHITE, line=LINE, align=PP_ALIGN.CENTER)
    add_text(slide, 7.55, 6.1, 4.64, .26, "微小逻辑错误 → 奖励陡降 → 探索不稳", size=14, color=MUTED, bold=True, align=PP_ALIGN.CENTER)

    # 4 — Framework
    slide = add_slide(prs, "CtrlHGen 总体框架", "PAPER · METHOD", "论文 p.4 Fig.3；p.5–7 Sec.3.2–3.4", "论文结论", NOTES[3])
    stages = [(.72, "01", "数据构造与增广", "KG 上按 pattern 采样\n复杂逻辑 → 同源子逻辑", PALE_BLUE, BLUE),
              (4.55, "02", "两阶段监督学习", "Unconditional SFT\n→ Conditional SFT", TEAL_LIGHT, TEAL),
              (8.38, "03", "GRPO 强化学习", "同一 prompt 生成 4 个候选\nKG 执行 → 双奖励", AMBER_LIGHT, AMBER)]
    for x, num, title, body, fill, accent in stages:
        add_box(slide, x, 1.48, 3.32, 3.65, fill=WHITE, line=accent)
        add_text(slide, x+.25, 1.72, .62, .52, num, size=23, color=accent, bold=True, font=FONT_EN)
        add_text(slide, x+.25, 2.36, 2.82, .48, title, size=19, color=NAVY, bold=True)
        add_text(slide, x+.25, 3.07, 2.82, 1.0, body, size=16, color=INK, align=PP_ALIGN.CENTER)
        add_tag(slide, x+.66, 4.46, ["扩大可学习空间", "先学语言，再学控制", "优化执行语义与遵循"][int(num)-1], fill=fill, color=accent, w=2.0)
    add_arrow(slide, 4.08, 3.25, 4.43, 3.25, color=BLUE)
    add_arrow(slide, 7.91, 3.25, 8.26, 3.25, color=TEAL)
    add_rich_text(slide, 1.3, 5.62, 10.7, .72, [("训练闭环：", TEAL, True, FONT_CN), ("生成 H → 解析 → 在 KG 执行 [H]G → 计算集合与条件奖励", NAVY, True, FONT_CN)], size=18, fill=WHITE, line=LINE, align=PP_ALIGN.CENTER)

    # 5 — Decomposition and controls
    slide = add_slide(prs, "子逻辑分解与控制接口", "PAPER · METHOD", "论文 p.5 Eq.(3)；p.6 Sec.3.3；p.7/p.15", "论文结论", NOTES[4])
    add_box(slide, .7, 1.3, 6.02, 5.15, fill=WHITE)
    add_text(slide, 1.02, 1.56, 5.3, .42, "复杂样本拆成同源的简单课程", size=20, color=NAVY, bold=True)
    add_box(slide, 2.24, 2.18, 2.88, .82, fill=RED_LIGHT, line=RED_LIGHT, text="inp：交 + 否定 + 投影", size=17, color=RED, bold=True, align=PP_ALIGN.CENTER)
    add_arrow(slide, 3.68, 3.05, 3.68, 3.64, color=TEAL)
    add_box(slide, 1.15, 3.78, 2.1, .9, fill=TEAL_LIGHT, line=TEAL_LIGHT, text="2p 子假设 A", size=17, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_box(slide, 4.11, 3.78, 2.1, .9, fill=TEAL_LIGHT, line=TEAL_LIGHT, text="2p 子假设 B", size=17, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_arrow(slide, 2.2, 4.73, 2.2, 5.18, color=BLUE)
    add_arrow(slide, 5.16, 4.73, 5.16, 5.18, color=BLUE)
    add_text(slide, 1.12, 5.22, 2.16, .62, "执行得 Osub,A", size=14, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, 4.08, 5.22, 2.16, .62, "执行得 Osub,B", size=14, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, 1.1, 5.9, 5.25, .3, "不是任意造样本：结构与语义都继承自原假设", size=14, color=MUTED, align=PP_ALIGN.CENTER)
    add_box(slide, 7.0, 1.3, 5.65, 5.15, fill=WHITE)
    add_text(slide, 7.32, 1.56, 4.95, .42, "五类控制条件 C", size=20, color=NAVY, bold=True)
    controls = [("结构", "指定 pattern", TEAL), ("结构", "指定实体数量 [ne]", TEAL), ("结构", "指定关系数量 [nr]", TEAL),
                ("语义", "指定实体 token", BLUE), ("语义", "指定关系 token", BLUE)]
    for i, (kind, textv, colorv) in enumerate(controls):
        y = 2.27 + i*.68
        add_tag(slide, 7.35, y+.04, kind, fill=TEAL_LIGHT if kind=="结构" else PALE_BLUE, color=colorv, w=.62)
        add_box(slide, 8.12, y, 3.9, .5, fill=RGBColor(251,251,249), line=LINE, text=textv, size=15.5, color=INK, bold=True)
    add_rich_text(slide, 7.36, 5.82, 4.65, .4, [("论文边界：", AMBER, True, FONT_CN), ("预定义 13 类逻辑结构", MUTED, False, FONT_CN)], size=13.5, fill=AMBER_LIGHT, line=AMBER_LIGHT, align=PP_ALIGN.CENTER)

    # 6 — Reward and paper evidence
    slide = add_slide(prs, "双奖励 GRPO：在语义与控制之间取舍", "PAPER · METHOD", "论文 p.6–7 Eq.(5–8)；p.10 Table 3；p.15 Appendix B", "论文结论", NOTES[5])
    add_box(slide, .72, 1.3, 5.65, 2.2, fill=WHITE)
    add_text(slide, 1.02, 1.55, 5.02, .38, "执行语义奖励", size=19, color=NAVY, bold=True)
    add_box(slide, 1.0, 2.04, 5.1, .78, fill=PALE_BLUE, line=PALE_BLUE,
            text="R_sem = 1.0·Jaccard + 0.5·Dice\n+ 0.5·Overlap", size=16.2, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, 1.02, 2.9, 5.0, .28, "严格集合一致 + 两个平滑补充指标", size=13.5, color=MUTED, align=PP_ALIGN.CENTER)
    add_box(slide, .72, 3.73, 5.65, 2.06, fill=WHITE)
    add_text(slide, 1.02, 3.98, 5.02, .38, "条件遵循与组内相对优化", size=19, color=NAVY, bold=True)
    add_box(slide, 1.0, 4.43, 5.1, .78, fill=AMBER_LIGHT, line=AMBER_LIGHT,
            text="R_cond = 0 / 1\nk = 4   ·   group normalization", size=15.8, color=AMBER, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, 1.02, 5.27, 5.0, .3, "关键：同组候选能否形成 reward 排序", size=13.5, color=MUTED, align=PP_ALIGN.CENTER)
    add_box(slide, 6.72, 1.3, 5.94, 4.9, fill=WHITE)
    add_text(slide, 7.04, 1.55, 5.25, .42, "论文消融：WN18RR · pattern", size=19, color=NAVY, bold=True)
    rows = [("w/o RL", "71.5", "81.5", MUTED), ("w/o Dice/Overlap", "74.8", "90.3", BLUE),
            ("w/o 条件奖励", "77.5", "68.3", AMBER), ("完整 CtrlHGen", "77.0", "93.5", TEAL)]
    table = slide.shapes.add_table(5, 3, Inches(7.03), Inches(2.2), Inches(5.25), Inches(2.68)).table
    table.columns[0].width=Inches(2.7); table.columns[1].width=Inches(1.25); table.columns[2].width=Inches(1.3)
    for j,t in enumerate(["模型", "Jaccard", "Accuracy"]):
        table.cell(0,j).fill.solid(); table.cell(0,j).fill.fore_color.rgb=NAVY; set_cell_text(table.cell(0,j),t,13,WHITE,True)
    for i,(name,j,a,c) in enumerate(rows,1):
        for k,val in enumerate([name,j,a]):
            table.cell(i,k).fill.solid(); table.cell(i,k).fill.fore_color.rgb=RGBColor(248,248,246) if i%2 else WHITE
            set_cell_text(table.cell(i,k),val,13.5,c if k else INK,bold=(i==4 or k>0))
    add_rich_text(slide, 7.02, 5.18, 5.28, .66, [("去掉条件奖励：", AMBER, True, FONT_CN), ("语义 +0.5 pt\n但 Accuracy −25.2 pt", INK, True, FONT_CN)], size=14.5, fill=AMBER_LIGHT, line=AMBER_LIGHT, align=PP_ALIGN.CENTER)
    add_text(slide, 6.98, 5.93, 5.35, .3, "我们的分析：Overlap 可能对完全包含的过生成过于宽容", size=12.5, color=MUTED, align=PP_ALIGN.CENTER)

    # 7 — Scope
    slide = add_slide(prs, "我们的复现范围与忠实度边界", "REPRO · SCOPE", "worklogs/reproduction/reports/reproduction-report-2026-08-12.md §1, §3", "复现实证", NOTES[6])
    add_rich_text(slide, .72, 1.2, 11.92, .7, [("定位：", TEAL, True, FONT_CN), ("方法与主要现象的高质量复现", NAVY, True, FONT_CN), ("  ≠  论文绝对数值逐点复刻", RED, True, FONT_CN)], size=21, fill=WHITE, line=LINE, align=PP_ALIGN.CENTER)
    cols=[(.72,"论文正文","12 层 GPT-2\nAdamW · LR 1e−5\nbatch 256 · 400+50 epoch\n4 × NVIDIA A6000",PALE_BLUE,BLUE),
          (4.68,"作者代码意图","GPT2_6 / num_layers: 6\nAdam · LR 5e−5\nWN batch 160\n5-step warm-up",AMBER_LIGHT,AMBER),
          (8.64,"本复现最终选择","6 × 768 · 12 heads\nAdam(SFT) / AdamW(GRPO)\n50+50 epoch · seed 42\n单张 NVIDIA L20",TEAL_LIGHT,TEAL)]
    for x,title,body,fill,c in cols:
        add_box(slide,x,2.25,3.28,3.3,fill=WHITE,line=c)
        add_tag(slide,x+.23,2.52,title,fill=fill,color=c,w=2.0)
        add_text(slide,x+.27,3.2,2.74,1.75,body,size=15.5,color=INK,align=PP_ALIGN.CENTER)
    add_text(slide, .92, 5.83, 11.4, .45, "范围：WN18RR · pattern condition · 13 patterns · test 1,664 · single seed", size=17, color=NAVY, bold=True, align=PP_ALIGN.CENTER)

    # 8 — Timeline
    slide = add_slide(prs, "复现路线图：从“能跑”到“可信”", "REPRO · JOURNEY", "worklogs/reproduction/phases/phase-a/b/c/d*.md；综合报告 §4–6", "复现实证", NOTES[7])
    y=3.42
    line=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(.88), Inches(y), Inches(12.35), Inches(y)); line.line.color.rgb=LINE; line.line.width=Pt(3)
    events=[(.95,"A/B","基础设施\n+ tiny smoke",BLUE), (2.55,"C-I","五项全 0",RED), (4.05,"FIX 1","监督契约",TEAL),
            (5.55,"C-II/III","结构与语义\n分离",AMBER), (7.35,"C-IV","覆盖扩容",TEAL), (8.95,"D-old","增益小 / KL 尖峰",RED),
            (10.55,"FIX 2","fresh + SEP",TEAL), (12.05,"TEST","双解码冻结",NAVY)]
    for i,(x,tag,body,c) in enumerate(events):
        circ=slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(x-.16), Inches(y-.16), Inches(.32), Inches(.32)); circ.fill.solid(); circ.fill.fore_color.rgb=c; circ.line.color.rgb=c
        upper=i%2==0
        add_tag(slide,x-.43,1.48 if upper else 4.02,tag,fill=TEAL_LIGHT if c==TEAL else RED_LIGHT if c==RED else PALE_BLUE if c==BLUE else AMBER_LIGHT if c==AMBER else LIGHT_GRAY,color=c,w=.86)
        add_text(slide,x-.62,1.94 if upper else 4.48,1.24,.75,body,size=13.2,color=INK,bold=True,align=PP_ALIGN.CENTER)
        add_arrow(slide,x,3.22 if upper else 3.62,x,2.78 if upper else 4.0,color=c,width=1.4)
    add_rich_text(slide, 2.3, 5.58, 8.75, .64, [("三次转折：", NAVY, True, FONT_CN), ("监督契约 → 实体覆盖 → GRPO 组内信号", TEAL, True, FONT_CN)], size=19, fill=WHITE, line=LINE, align=PP_ALIGN.CENTER)

    # 9 — Contract failure
    slide = add_slide(prs, "转折一：loss 下降，但监督实际上失效", "REPRO · TURN 1", "综合报告 §5.2；akgr/tokenizer.py；tests/test_condition_batch.py", "复现实证", NOTES[8])
    add_tag(slide,.72,1.24,"C-I 现象",fill=RED_LIGHT,color=RED,w=.9)
    add_rich_text(slide,1.78,1.2,4.9,.52,[("train loss ↓ ~0.006   ·   ",NAVY,True,FONT_EN),("EOS = 0   ·   Parse = 0",RED,True,FONT_EN)],size=15.7,fill=WHITE,line=LINE,align=PP_ALIGN.CENTER)
    # token contract
    tokens=[("answers",PALE_BLUE,BLUE), ("COND",LIGHT_GRAY,MUTED), ("pattern",PALE_BLUE,BLUE), ("SEP",TEAL_LIGHT,TEAL), ("target",TEAL_LIGHT,TEAL), ("END",TEAL_LIGHT,TEAL)]
    x=.78
    for name,fill,c in tokens:
        w={"answers":1.02,"COND":.78,"pattern":1.02,"SEP":.68,"target":.92,"END":.70}[name]
        add_box(slide,x,2.12,w,.68,fill=fill,line=fill,text=name,size=13.8,color=c,bold=True,align=PP_ALIGN.CENTER)
        x+=w+.07
    add_text(slide,.82,2.9,3.55,.3,"prompt：全部 mask",size=13,color=MUTED,align=PP_ALIGN.CENTER)
    add_text(slide,4.34,2.9,2.0,.3,"active: target + END",size=10.8,color=TEAL,bold=True,align=PP_ALIGN.CENTER)
    add_box(slide,.72,3.48,5.97,2.3,fill=RED_LIGHT,line=RED_LIGHT)
    add_text(slide,1.0,3.75,5.35,.42,"根因：padding 状态污染 → label mask 错位",size=18,color=RED,bold=True)
    add_bullets(slide,1.0,4.32,5.25,1.15,["generation 临时 left-padding 被 backend 记住","checkpoint 带入错误状态；target 与 END 被屏蔽"],size=15.5,bullet_color=RED,spacing=6)
    add_box(slide,7.02,1.2,5.62,4.58,fill=WHITE,line=TEAL)
    add_text(slide,7.34,1.5,4.95,.42,"修复后的监督契约",size=20,color=NAVY,bold=True)
    fixes=["训练强制 right padding","生成仅在受控作用域 left padding","退出作用域后清理 backend padding","checkpoint v2 校验 tokenizer/hash/END","Parse / EOS 健康门槛后才能晋级"]
    add_bullets(slide,7.38,2.22,4.72,2.85,fixes,size=15.2,bullet_color=TEAL,spacing=8)
    add_tag(slide,8.52,5.18,"修复后：稳定生成可解析、有 EOS 的逻辑",fill=TEAL_LIGHT,color=TEAL,w=3.12)
    add_rich_text(slide,.75,6.05,11.84,.42,[("教训：",RED,True,FONT_CN),("teacher-forced loss 不是自由生成健康度；二者必须分别设 gate。",NAVY,True,FONT_CN)],size=16,fill=WHITE,line=LINE,align=PP_ALIGN.CENTER)

    # 10 — Structure != semantics
    slide = add_slide(prs, "转折二：会生成指定结构，不等于答案集合正确", "REPRO · TURN 2", "综合报告 §5.3；Phase C frozen-test aggregates", "复现实证", NOTES[9])
    add_text(slide,.75,1.2,11.75,.45,"C-II → C-III：控制指标上升，集合语义反向下降",size=19,color=NAVY,bold=True,align=PP_ALIGN.CENTER)
    # slope chart
    x1,x2=2.05,5.25
    for y in [2.25,3.35,4.45]:
        l=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y),Inches(x2),Inches(y)); l.line.color.rgb=RGBColor(232,232,229); l.line.width=Pt(1)
    add_text(slide,1.43,1.82,1.2,.35,"C-II · 12L",size=13,color=MUTED,bold=True,align=PP_ALIGN.CENTER)
    add_text(slide,4.63,1.82,1.2,.35,"C-III · 6L",size=13,color=MUTED,bold=True,align=PP_ALIGN.CENTER)
    metrics=[("Jaccard",.3299,.2690,RED), ("Pattern Accuracy",.3185,.5883,TEAL), ("Smatch",.6028,.6613,BLUE)]
    ybase=[2.4,3.52,4.64]
    for (name,a,b,c),yb in zip(metrics,ybase):
        dy=(b-a)*1.2
        line=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(yb),Inches(x2),Inches(yb-dy)); line.line.color.rgb=c; line.line.width=Pt(3)
        for x,y,v in [(x1,yb,a),(x2,yb-dy,b)]:
            o=slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL,Inches(x-.09),Inches(y-.09),Inches(.18),Inches(.18)); o.fill.solid(); o.fill.fore_color.rgb=c; o.line.color.rgb=c
            add_text(slide,x-.4,y-.46,.8,.28,f"{v:.4f}",size=11.5,color=c,bold=True,align=PP_ALIGN.CENTER,font=FONT_EN)
        add_text(slide,.78,yb-.2,.95,.42,name,size=13,color=INK,bold=True,align=PP_ALIGN.RIGHT,font=FONT_EN)
    add_box(slide,6.55,1.65,5.72,3.65,fill=WHITE,line=LINE)
    add_text(slide,6.9,1.98,5.0,.42,"两种能力，不能混成一个“正确率”",size=19,color=NAVY,bold=True)
    add_box(slide,6.92,2.68,2.18,1.35,fill=TEAL_LIGHT,line=TEAL_LIGHT,text="结构控制\n逻辑拓扑是否匹配",size=16,color=TEAL,bold=True,align=PP_ALIGN.CENTER)
    add_box(slide,9.7,2.68,2.18,1.35,fill=PALE_BLUE,line=PALE_BLUE,text="集合语义\n实体 / 关系是否实例化正确",size=15.5,color=BLUE,bold=True,align=PP_ALIGN.CENTER)
    add_arrow(slide,9.17,3.35,9.62,3.35,color=AMBER,dashed=True)
    add_text(slide,6.98,4.48,4.85,.38,"指定的“形状”正确 ≠ 执行答案集合正确",size=15.5,color=RED,bold=True,align=PP_ALIGN.CENTER)
    add_rich_text(slide,1.25,5.72,10.82,.58,[("限制：",AMBER,True,FONT_CN),("C-II→C-III 同时改变多项设置，不能归因给某一个超参数。",INK,False,FONT_CN)],size=15,fill=AMBER_LIGHT,line=AMBER_LIGHT,align=PP_ALIGN.CENTER)

    # 11 — Coverage chain
    slide = add_slide(prs, "最强单变量证据：训练覆盖扩容", "REPRO · TURN 2", "综合报告 §5.5；C-III→C-IV，seed 42", "复现实证", NOTES[10])
    add_rich_text(slide,.72,1.15,11.92,.6,[("唯一主要变量：",TEAL,True,FONT_CN),("train / pattern  1,024 → 8,000",NAVY,True,FONT_EN),("   (7.81×)",MUTED,False,FONT_EN)],size=19,fill=WHITE,line=LINE,align=PP_ALIGN.CENTER)
    chain=[("实体词表覆盖","35.41%","81.86%","+46.45 pp",TEAL), ("test 未见实体","27.70%","3.34%","−24.37 pp",BLUE),
           ("greedy Jaccard","0.2690","0.5969","+0.3279",TEAL), ("Pattern Accuracy","0.5883","0.9441","+0.3558",TEAL)]
    for i,(label,a,b,d,c) in enumerate(chain):
        x=.78+i*3.05
        add_box(slide,x,2.08,2.62,3.35,fill=WHITE,line=c)
        add_text(slide,x+.18,2.35,2.26,.4,label,size=16,color=NAVY,bold=True,align=PP_ALIGN.CENTER)
        add_text(slide,x+.10,3.15,1.02,.45,a,size=17,color=MUTED,bold=True,align=PP_ALIGN.CENTER,font=FONT_EN)
        add_arrow(slide,x+1.17,3.38,x+1.43,3.38,color=c)
        add_text(slide,x+1.48,3.15,1.02,.45,b,size=17,color=c,bold=True,align=PP_ALIGN.CENTER,font=FONT_EN)
        add_tag(slide,x+.66,4.25,d,fill=TEAL_LIGHT if c==TEAL else PALE_BLUE,color=c,w=1.3)
        if i<3: add_arrow(slide,x+2.68,3.73,x+2.98,3.73,color=AMBER,dashed=True)
    add_rich_text(slide,1.0,5.78,11.3,.52,[("最直接解释：",TEAL,True,FONT_CN),("关系早已 22/22 覆盖；扩容主要补足实体监督。",NAVY,True,FONT_CN)],size=16.5,fill=TEAL_LIGHT,line=TEAL_LIGHT,align=PP_ALIGN.CENTER)
    add_text(slide,1.0,6.4,11.3,.24,"仍需 compute-matched 消融：更多数据也意味着更多 optimizer steps，覆盖不是已证明的唯一原因。",size=11.5,color=MUTED,align=PP_ALIGN.CENTER)

    # 12 — GRPO repair
    slide = add_slide(prs, "转折三：高 reward ≠ 有效 GRPO 信号", "REPRO · TURN 3", "综合报告 §6；phase-d repaired artifacts", "复现实证", NOTES[11])
    add_box(slide,.72,1.2,3.35,3.8,fill=RED_LIGHT,line=RED)
    add_tag(slide,1.02,1.5,"原 Phase D",fill=WHITE,color=RED,w=1.2)
    add_bullets(slide,1.02,2.15,2.78,1.2,["复用 SFT 已见 104k 数据","raw prompt 末尾缺 SEP","绝对 reward 高，但组内 std 低"],size=14.5,bullet_color=RED,spacing=7)
    add_rich_text(slide,1.04,3.76,2.72,.58,[("reward std ",MUTED,False,FONT_EN),("0.122–0.165",RED,True,FONT_EN)],size=15,fill=WHITE,line=WHITE,align=PP_ALIGN.CENTER)
    add_rich_text(slide,1.04,4.42,2.72,.38,[("max KL = ",MUTED,False,FONT_EN),("1703",RED,True,FONT_EN)],size=14,align=PP_ALIGN.CENTER)
    add_arrow(slide,4.24,3.05,4.93,3.05,color=TEAL,width=3)
    add_text(slide,4.18,3.33,.8,.55,"联合修复",size=12,color=TEAL,bold=True,align=PP_ALIGN.CENTER)
    add_box(slide,5.08,1.2,3.35,3.8,fill=TEAL_LIGHT,line=TEAL)
    add_tag(slide,5.38,1.5,"Repaired",fill=WHITE,color=TEAL,w=1.1)
    add_bullets(slide,5.38,2.15,2.78,1.2,["fresh RL-only 104k 数据","prompt 显式以 SEP 结束","组内多样性与方差恢复"],size=14.5,bullet_color=TEAL,spacing=7)
    add_rich_text(slide,5.4,3.76,2.72,.58,[("reward std ",MUTED,False,FONT_EN),("0.207–0.232",TEAL,True,FONT_EN)],size=15,fill=WHITE,line=WHITE,align=PP_ALIGN.CENTER)
    add_rich_text(slide,5.4,4.42,2.72,.38,[("KL ",MUTED,False,FONT_EN),("0.090–0.378",TEAL,True,FONT_EN)],size=14,align=PP_ALIGN.CENTER)
    add_box(slide,8.76,1.2,3.88,3.8,fill=WHITE,line=LINE)
    add_text(slide,9.08,1.5,3.2,.4,"证据逐级升级",size=19,color=NAVY,bold=True,align=PP_ALIGN.CENTER)
    stages=[("Pilot val",e['pilot']['greedy']['paired_semantic_average'],AMBER), ("Full val",e['full']['greedy']['paired_semantic_average'],TEAL), ("Frozen test",e['comparison']['greedy']['paired_semantic_average'],NAVY)]
    for i,(name,d,c) in enumerate(stages):
        y=2.18+i*.82
        add_text(slide,9.02,y,1.02,.28,name,size=12.5,color=INK,bold=True,font=FONT_EN)
        width=max(.08,d['mean']/.04*1.9)
        bar=slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE,Inches(10.12),Inches(y+.03),Inches(width),Inches(.18)); bar.fill.solid(); bar.fill.fore_color.rgb=c; bar.line.fill.background()
        add_text(slide,9.98,y+.22,2.38,.42,
                 f"Δ {d['mean']:+.4f}\nCI [{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}]",
                 size=9.8,color=c,bold=True,font=FONT_EN,align=PP_ALIGN.CENTER)
    add_tag(slide,9.35,4.53,"full / test CI > 0",fill=TEAL_LIGHT,color=TEAL,w=2.45)
    add_rich_text(slide,1.25,5.52,10.8,.55,[("GRPO 的核心：",NAVY,True,FONT_CN),("不是 prompt 的绝对 reward 高，而是同组 completion 之间有可排序的差异。",TEAL,True,FONT_CN)],size=16.5,fill=WHITE,line=LINE,align=PP_ALIGN.CENTER)
    add_text(slide,1.0,6.18,11.3,.25,"边界：fresh data 与 SEP 联合修复，尚未分离贡献；step 0 仍有 60.55% 零 reward-variance 组。",size=11.5,color=MUTED,align=PP_ALIGN.CENTER)

    # 13 — Final results
    slide = add_slide(prs, "最终结果：核心现象已复现，绝对差距仍在", "REPRO · RESULT", "repaired-test/comparison.json；论文 p.10 Table 3", "复现实证", NOTES[12])
    comp=e['comparison']; paper=e['paper']
    cats_sem=["Jaccard","Dice","Overlap"]
    chartdata=ChartData(); chartdata.categories=cats_sem
    chartdata.add_series("SFT parent (epoch 45)",[comp['greedy']['baseline'][k] for k in ['jaccard','dice','overlap']])
    chartdata.add_series("Repaired GRPO",[comp['greedy']['candidate'][k] for k in ['jaccard','dice','overlap']])
    chartdata.add_series("论文 CtrlHGen",[paper['full'][k] for k in ['jaccard','dice','overlap']])
    chart=slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,Inches(.72),Inches(1.55),Inches(6.05),Inches(3.75),chartdata).chart
    chart.has_title=True; chart.chart_title.text_frame.text="集合语义（greedy）"; chart.chart_title.text_frame.paragraphs[0].runs[0].font.name=FONT_CN; chart.chart_title.text_frame.paragraphs[0].runs[0].font.size=Pt(14); chart.chart_title.text_frame.paragraphs[0].runs[0].font.bold=True; chart.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb=NAVY
    chart.value_axis.minimum_scale=.5; chart.value_axis.maximum_scale=.9; chart.value_axis.major_unit=.1
    chart_style(chart)
    for ser,c in zip(chart.series,[BLUE,TEAL,NAVY]): ser.format.fill.solid(); ser.format.fill.fore_color.rgb=c; ser.format.line.color.rgb=c
    chartdata2=ChartData(); chartdata2.categories=["Pattern Acc.","Smatch"]
    chartdata2.add_series("SFT parent",[comp['greedy']['baseline']['condition_accuracy'],comp['greedy']['baseline']['smatch']])
    chartdata2.add_series("Repaired GRPO",[comp['greedy']['candidate']['condition_accuracy'],comp['greedy']['candidate']['smatch']])
    chartdata2.add_series("论文 CtrlHGen",[paper['full']['condition_accuracy'],paper['full']['smatch']])
    chart2=slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,Inches(6.92),Inches(1.55),Inches(5.72),Inches(3.75),chartdata2).chart
    chart2.has_title=True; chart2.chart_title.text_frame.text="控制与结构（greedy）"; chart2.chart_title.text_frame.paragraphs[0].runs[0].font.name=FONT_CN; chart2.chart_title.text_frame.paragraphs[0].runs[0].font.size=Pt(14); chart2.chart_title.text_frame.paragraphs[0].runs[0].font.bold=True; chart2.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb=NAVY
    chart2.value_axis.minimum_scale=.75; chart2.value_axis.maximum_scale=1.0; chart2.value_axis.major_unit=.05
    chart_style(chart2)
    for ser,c in zip(chart2.series,[BLUE,TEAL,NAVY]): ser.format.fill.solid(); ser.format.fill.fore_color.rgb=c; ser.format.line.color.rgb=c
    gd=comp['greedy']['paired_semantic_average']; sd=comp['sampled']['paired_semantic_average']
    add_rich_text(slide,.82,5.52,5.7,.62,[("Greedy 三项均值  ",MUTED,False,FONT_CN),(f"{gd['mean']:+.5f}",TEAL,True,FONT_EN),(f"  CI [{gd['ci95_low']:.5f}, {gd['ci95_high']:.5f}]",INK,False,FONT_EN)],size=14.5,fill=TEAL_LIGHT,line=TEAL_LIGHT,align=PP_ALIGN.CENTER)
    add_rich_text(slide,6.82,5.52,5.7,.62,[("Sampled 三项均值  ",MUTED,False,FONT_CN),(f"{sd['mean']:+.5f}",TEAL,True,FONT_EN),(f"  CI [{sd['ci95_low']:.5f}, {sd['ci95_high']:.5f}]",INK,False,FONT_EN)],size=14.5,fill=TEAL_LIGHT,line=TEAL_LIGHT,align=PP_ALIGN.CENTER)
    add_text(slide,.9,6.28,11.55,.25,"论文仅作尺度参照（训练规模与生成设置未完全披露）；主要差距集中在 J/D/O：−0.133 / −0.121 / −0.107。",size=11.5,color=MUTED,align=PP_ALIGN.CENTER)

    # 14 — Conclusion
    slide = add_slide(prs, "结论、证据边界与下一步", "REPRO · TAKEAWAYS", "综合报告 §9, §11", "我们的分析", NOTES[13])
    sections=[(.72,"我们已经做到",TEAL,TEAL_LIGHT,["可控逻辑生成链路稳定可解析","GRPO 对集合语义产生配对正增益","监督契约、覆盖、RL 信号均可审计"]),
              (4.72,"现在不能声称",RED,RED_LIGHT,["严格复现论文绝对数值","单 seed 等同跨训练显著性","fresh 与 SEP 各自贡献已分离"]),
              (8.72,"下一步优先级",AMBER,AMBER_LIGHT,["12 层 depth-only Phase C-V","多 seed + fresh RL shard","fresh×SEP 2×2 与新 untouched test"])]
    for x,title,c,fill,items in sections:
        add_box(slide,x,1.35,3.62,4.6,fill=WHITE,line=c)
        add_tag(slide,x+.34,1.68,title,fill=fill,color=c,w=1.75)
        add_bullets(slide,x+.36,2.48,2.9,2.5,items,size=15.5,bullet_color=c,spacing=12)
    add_rich_text(slide,1.4,6.18,10.55,.48,[("一句话结论：",NAVY,True,FONT_CN),("核心方法现象已复现；结构能力强，集合语义仍是主要差距。",TEAL,True,FONT_CN)],size=17,fill=WHITE,line=LINE,align=PP_ALIGN.CENTER)

    # 15 — Backup patterns and formulae
    slide = add_slide(prs, "备份：13 种逻辑 pattern 与指标公式", "BACKUP", "akgr/metadata/pattern_filtered.csv；论文 p.6 Eq.(5)", "我们的分析", NOTES[14], backup=True)
    add_text(slide,.72,1.14,7.4,.36,"实际复现使用的 13 类（不含 pattern_table.csv 中额外的 3p）",size=16,color=NAVY,bold=True)
    for i,row in enumerate(e['patterns']):
        col=i%4; rr=i//4; x=.72+col*1.82; y=1.72+rr*1.05
        add_box(slide,x,y,1.55,.78,fill=WHITE,line=TEAL if row['pattern_abbr'] in {'up','3in','pni','pin','inp'} else LINE)
        add_text(slide,x+.12,y+.08,.48,.28,row['pattern_abbr'],size=15,color=TEAL if row['pattern_abbr'] in {'up','3in','pni','pin','inp'} else NAVY,bold=True,font=FONT_EN)
        short=row['pattern_str'].replace(',',' ')
        add_text(slide,x+.12,y+.37,1.3,.25,short,size=8.8,color=MUTED,font=FONT_EN,align=PP_ALIGN.CENTER)
    add_box(slide,8.25,1.28,4.4,4.85,fill=WHITE,line=LINE)
    add_text(slide,8.58,1.57,3.72,.4,"集合指标",size=19,color=NAVY,bold=True)
    formulae=[("Jaccard","|A∩O| / |A∪O|",BLUE), ("Dice","2|A∩O| / (|A|+|O|)",TEAL), ("Overlap","|A∩O| / min(|A|,|O|)",AMBER)]
    for i,(n,f,c) in enumerate(formulae):
        add_tag(slide,8.58,2.2+i*.74,n,fill=PALE_BLUE if c==BLUE else TEAL_LIGHT if c==TEAL else AMBER_LIGHT,color=c,w=.88)
        add_text(slide,9.62,2.18+i*.74,2.5,.36,f,size=14,color=INK,font=FONT_EN)
    add_box(slide,8.58,4.6,3.72,.54,fill=LIGHT_GRAY,line=LIGHT_GRAY,text="R_sem ∈ [0,2]   ·   R_cond ∈ [0,1]",size=14.5,color=NAVY,bold=True,align=PP_ALIGN.CENTER)
    add_text(slide,8.57,5.3,3.76,.5,"Overlap 在一集合完全包含另一集合时为 1，\n可能对严重过生成过于宽容。",size=13,color=RED,bold=True,align=PP_ALIGN.CENTER)

    # 16 — Backup config matrix
    slide = add_slide(prs, "备份：论文、代码意图与本复现的设置差异", "BACKUP", "综合报告 §3；upstream/main@3f658013", "复现实证", NOTES[15], backup=True)
    rows=[("模型深度","12 层","GPT2_6 / num_layers:6","字段可能不覆盖 n_layer","显式 n_layer=6"),
          ("Optimizer","AdamW","未写","代码使用 Adam","Adam / AdamW(GRPO)"),
          ("SFT LR","1e−5","5e−5","5e−5","5e−5"),
          ("Batch","256","WN: 160","160","160"),
          ("Epoch","400 + 50","GPT-2 配置 50","315 恢复但总 epoch 50","50 + 50"),
          ("Warm-up","50 / 5 epoch","warm_up: 5","前 5 optimizer step","前 5 optimizer step"),
          ("算力","4×A6000","—","—","1×L20")]
    headers=["项目","论文正文","作者代码意图","公开代码实际语义","本复现"]
    table=slide.shapes.add_table(len(rows)+1,len(headers),Inches(.7),Inches(1.35),Inches(11.95),Inches(4.9)).table
    widths=[1.35,2.15,2.3,3.2,2.95]
    for i,w in enumerate(widths): table.columns[i].width=Inches(w)
    for j,h in enumerate(headers): table.cell(0,j).fill.solid(); table.cell(0,j).fill.fore_color.rgb=NAVY; set_cell_text(table.cell(0,j),h,12.2,WHITE,True)
    for i,row in enumerate(rows,1):
        for j,v in enumerate(row):
            table.cell(i,j).fill.solid(); table.cell(i,j).fill.fore_color.rgb=TEAL_LIGHT if j==4 else RGBColor(248,248,246) if i%2 else WHITE
            set_cell_text(table.cell(i,j),v,11.5,TEAL if j==4 else INK,bold=(j==0 or j==4))
    add_text(slide,.9,6.42,11.5,.24,"原则：对公开材料只陈述可验证差异；本复现选择“作者意图的自洽实现”，不复制无效字段或零步退出。",size=11.5,color=MUTED,align=PP_ALIGN.CENTER)

    # 17 — Backup full metrics and protocol
    slide = add_slide(prs, "备份：双解码指标与测试隔离边界", "BACKUP", "repaired-test/comparison.json；综合报告 §5.6, §6.4, §9", "复现实证", NOTES[16], backup=True)
    headers=["解码 / 模型","Jaccard","Dice","Overlap","PA","Smatch","Parse","EOS"]
    rows=[]
    for mode,label in [('greedy','Greedy'),('sampled','Sampled')]:
        for who,wlabel in [('baseline','SFT parent'),('candidate','Repaired')]:
            d=e['comparison'][mode][who]
            rows.append((f"{label} · {wlabel}",d['jaccard'],d['dice'],d['overlap'],d['condition_accuracy'],d['smatch'],d['parse_ok'],d['eos_rate']))
    table=slide.shapes.add_table(5,8,Inches(.7),Inches(1.32),Inches(11.95),Inches(2.62)).table
    widths=[2.2,1.32,1.2,1.22,1.1,1.18,1.18,1.1]
    for i,w in enumerate(widths): table.columns[i].width=Inches(w)
    for j,h in enumerate(headers): table.cell(0,j).fill.solid(); table.cell(0,j).fill.fore_color.rgb=NAVY; set_cell_text(table.cell(0,j),h,11.5,WHITE,True,font=FONT_EN if j else FONT_CN)
    for i,row in enumerate(rows,1):
        rep="Repaired" in row[0]
        for j,v in enumerate(row):
            table.cell(i,j).fill.solid(); table.cell(i,j).fill.fore_color.rgb=TEAL_LIGHT if rep else WHITE
            set_cell_text(table.cell(i,j),v if j==0 else f"{v:.5f}",11.2,TEAL if rep and j else INK,bold=rep)
    gd=e['comparison']['greedy']['paired_semantic_average']; sd=e['comparison']['sampled']['paired_semantic_average']
    add_rich_text(slide,.72,4.18,5.72,.58,[("Greedy Δmean ",MUTED,False,FONT_CN),(f"{gd['mean']:+.5f}",TEAL,True,FONT_EN),(f"  [{gd['ci95_low']:.5f}, {gd['ci95_high']:.5f}]",INK,False,FONT_EN)],size=14,fill=TEAL_LIGHT,line=TEAL_LIGHT,align=PP_ALIGN.CENTER)
    add_rich_text(slide,6.63,4.18,5.72,.58,[("Sampled Δmean ",MUTED,False,FONT_CN),(f"{sd['mean']:+.5f}",TEAL,True,FONT_EN),(f"  [{sd['ci95_low']:.5f}, {sd['ci95_high']:.5f}]",INK,False,FONT_EN)],size=14,fill=TEAL_LIGHT,line=TEAL_LIGHT,align=PP_ALIGN.CENTER)
    add_box(slide,.72,5.08,5.72,1.12,fill=WHITE,line=LINE)
    add_text(slide,1.0,5.27,5.15,.74,"conditional-best → epoch 50（训练规则选择）\nphase-d-parent → epoch 45（研究目标 parent）",size=14,color=NAVY,bold=True,align=PP_ALIGN.CENTER)
    add_box(slide,6.63,5.08,5.72,1.12,fill=AMBER_LIGHT,line=AMBER)
    add_text(slide,6.92,5.22,5.15,.82,"parent bake-off 用过同一 test split；\n终点未看 test，但端到端不是 untouched holdout。",size=12.5,color=INK,bold=True,align=PP_ALIGN.CENTER)
    add_text(slide,.95,6.42,11.4,.24,f"CI：固定模型在 {e['test_n']:,} 条记录上的 10,000 次 paired bootstrap；不是跨 seed 方差。",size=11.5,color=MUTED,align=PP_ALIGN.CENTER)

    return prs


def write_markdown(prs: Presentation) -> None:
    titles=[]
    for slide in prs.slides:
        candidates=[]
        for shape in slide.shapes:
            if getattr(shape,"has_text_frame",False) and shape.text.strip():
                candidates.append((shape.top,shape.left,shape.text.strip().splitlines()[0]))
        titles.append(sorted(candidates)[0][2] if candidates else "")
    out=["# CtrlHGen：可控逻辑假设生成与复现进展", "", "> 15 分钟主讲：第 1–14 页；第 15–17 页为答疑备份。", "> PPT 内也嵌入同样的演讲者备注。", ""]
    for i,(slide,title,n) in enumerate(zip(prs.slides,titles,NOTES),1):
        note_text = slide.notes_slide.notes_text_frame.text
        source_match = re.search(r"【证据来源】(.+?)(?:\n\n|$)", note_text, flags=re.S)
        assert source_match, i
        source = source_match.group(1).strip()
        out += [f"## {i}. {title}", "", f"- 建议用时：{n['time']}", "", "### 核心讲述", "", n['talk'], "", "### 转场", "", n['transition'], "", "### 证据来源", "", source, "", "### 必要限定", "", n['caveat'], "", "### 可能追问", "", n['qa'], ""]
    SCRIPT_OUT.write_text("\n".join(out),encoding="utf-8")


def validate(prs: Presentation) -> None:
    assert len(prs.slides)==17
    assert prs.slide_width==SLIDE_W and prs.slide_height==SLIDE_H
    assert len(NOTES)==17
    for idx,slide in enumerate(prs.slides,1):
        note=slide.notes_slide.notes_text_frame.text
        for label in ["【建议用时】","【核心讲述】","【转场】","【证据来源】","【必要限定】","【可能追问】"]:
            assert label in note, (idx,label)
        for shape in slide.shapes:
            assert shape.left >= -Inches(.02) and shape.top >= -Inches(.02), (idx,shape.name,"negative")
            assert shape.left+shape.width <= SLIDE_W+Inches(.02), (idx,shape.name,"right")
            assert shape.top+shape.height <= SLIDE_H+Inches(.02), (idx,shape.name,"bottom")
    assert PDF.exists() and REPORT.exists()


# The desktop PowerPoint automation endpoint is not available in every WSL
# session.  This renderer produces a faithful, reviewable PDF preview directly
# from the generated PPTX object model.  PowerPoint remains the source format;
# the PDF is deliberately labelled a preview in the document metadata.
PREVIEW_DPI = 120
PREVIEW_W = 1600
PREVIEW_H = 900
FONT_DIR = Path("/mnt/c/Windows/Fonts")


def _px(emu: int) -> int:
    return int(round(emu / 914400 * PREVIEW_DPI))


def _rgb(value, default=(0, 0, 0)) -> tuple[int, int, int]:
    try:
        c = value.rgb
        if c is None:
            return default
        return tuple(c)
    except Exception:
        return default


def _font(size_pt: float, bold=False, latin=False):
    if latin:
        path = FONT_DIR / ("arialbd.ttf" if bold else "arial.ttf")
    else:
        path = FONT_DIR / ("msyhbd.ttc" if bold else "msyh.ttc")
    if not path.exists():
        path = FONT_DIR / ("simhei.ttf" if bold else "simsun.ttc")
    return ImageFont.truetype(str(path), max(8, int(round(size_pt * PREVIEW_DPI / 72))))


def _text_width(draw, text, font) -> float:
    if not text:
        return 0
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    lines = []
    for raw in text.split("\n"):
        if not raw:
            lines.append("")
            continue
        current = ""
        for ch in raw:
            trial = current + ch
            if current and _text_width(draw, trial, font) > max_width:
                lines.append(current)
                current = ch
            else:
                current = trial
        if current:
            lines.append(current)
    return lines or [""]


def _paragraph_style(paragraph):
    run = next((r for r in paragraph.runs if r.text), None)
    if run is None:
        return 16.0, False, (37, 46, 55), False
    size = run.font.size.pt if run.font.size else 16.0
    bold = bool(run.font.bold)
    color = _rgb(run.font.color, (37, 46, 55))
    latin = (run.font.name or "") == FONT_EN
    return size, bold, color, latin


def _draw_text_frame(draw, tf, rect) -> None:
    x0, y0, x1, y1 = rect
    left = _px(tf.margin_left or 0); right = _px(tf.margin_right or 0)
    top = _px(tf.margin_top or 0); bottom = _px(tf.margin_bottom or 0)
    ax0, ay0, ax1, ay1 = x0 + left, y0 + top, x1 - right, y1 - bottom
    if ax1 <= ax0 or ay1 <= ay0:
        return
    rendered = []
    for p in tf.paragraphs:
        if not p.text:
            continue
        size, bold, color, latin = _paragraph_style(p)
        font = _font(size, bold=bold, latin=latin)
        lines = _wrap(draw, p.text, font, max(4, ax1 - ax0))
        line_h = int(round(size * PREVIEW_DPI / 72 * 1.18))
        rendered.append((p, lines, font, color, line_h))
    if not rendered:
        return
    total_h = sum(len(lines) * lh for _, lines, _, _, lh in rendered)
    anchor = getattr(tf.vertical_anchor, "name", "MIDDLE") if tf.vertical_anchor else "MIDDLE"
    if anchor == "TOP":
        y = ay0
    elif anchor == "BOTTOM":
        y = max(ay0, ay1 - total_h)
    else:
        y = max(ay0, ay0 + (ay1 - ay0 - total_h) // 2)
    for p, lines, font, color, line_h in rendered:
        alignment = getattr(p.alignment, "name", "LEFT") if p.alignment else "LEFT"
        for line in lines:
            width = _text_width(draw, line, font)
            if alignment == "CENTER":
                x = ax0 + max(0, (ax1 - ax0 - width) / 2)
            elif alignment == "RIGHT":
                x = max(ax0, ax1 - width)
            else:
                x = ax0
            draw.text((x, y), line, font=font, fill=color)
            y += line_h


def _draw_chart(draw, chart, rect) -> None:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=6, fill=(255, 255, 255), outline=(210, 213, 211), width=1)
    title = ""
    try:
        title = chart.chart_title.text_frame.text
    except Exception:
        pass
    title_font = _font(14, bold=True)
    if title:
        tw = _text_width(draw, title, title_font)
        draw.text((x0 + (x1-x0-tw)/2, y0+10), title, font=title_font, fill=tuple(NAVY))
    left, right, top, bottom = x0+58, x1-18, y0+56, y1-76
    categories = [getattr(c, "label", str(c)) for c in chart.plots[0].categories]
    series = list(chart.series)
    values = [list(s.values) for s in series]
    ymin = chart.value_axis.minimum_scale
    ymax = chart.value_axis.maximum_scale
    if ymin is None:
        ymin = min(0, min(min(v) for v in values))
    if ymax is None:
        ymax = max(max(v) for v in values) * 1.08
    colors = []
    for s, fallback in zip(series, [(49,91,137),(34,139,128),(25,48,74)]):
        colors.append(_rgb(s.format.fill.fore_color, fallback))
    for k in range(5):
        yy = bottom - (bottom-top)*k/4
        draw.line((left, yy, right, yy), fill=(229,230,226), width=1)
        val = ymin + (ymax-ymin)*k/4
        font = _font(8.5, latin=True)
        label = f"{val:.2f}"
        draw.text((left-48, yy-7), label, font=font, fill=(95,105,113))
    group_w = (right-left)/max(1,len(categories))
    bar_w = min(24, group_w/(len(series)+1))
    for ci,cat in enumerate(categories):
        cx = left + group_w*(ci+.5)
        for si,svals in enumerate(values):
            val=svals[ci]
            h=(val-ymin)/(ymax-ymin)*(bottom-top)
            bx0=cx+(si-(len(series)-1)/2)*bar_w-bar_w*.42
            draw.rectangle((bx0,bottom-h,bx0+bar_w*.84,bottom),fill=colors[si])
        f=_font(8.5,latin=True)
        tw=_text_width(draw,cat,f)
        draw.text((cx-tw/2,bottom+10),cat,font=f,fill=(37,46,55))
    legend_y=y1-35
    legend_font=_font(8.5)
    total=sum(18+_text_width(draw,s.name,legend_font)+18 for s in series)
    lx=x0+(x1-x0-total)/2
    for s,c in zip(series,colors):
        draw.rectangle((lx,legend_y+3,lx+12,legend_y+13),fill=c)
        draw.text((lx+17,legend_y),s.name,font=legend_font,fill=(37,46,55))
        lx += 18+_text_width(draw,s.name,legend_font)+18


def _draw_table(draw, table, rect) -> None:
    x0,y0,x1,y1=rect
    col_widths=[_px(c.width) for c in table.columns]
    row_heights=[_px(r.height) for r in table.rows]
    # Scale possible round-off to the requested shape box.
    sx=(x1-x0)/max(1,sum(col_widths)); sy=(y1-y0)/max(1,sum(row_heights))
    yy=y0
    for ri,row in enumerate(table.rows):
        rh=row_heights[ri]*sy; xx=x0
        for ci,cell in enumerate(row.cells):
            cw=col_widths[ci]*sx
            fill=_rgb(cell.fill.fore_color,(255,255,255))
            draw.rectangle((xx,yy,xx+cw,yy+rh),fill=fill,outline=(210,213,211),width=1)
            _draw_text_frame(draw,cell.text_frame,(int(xx),int(yy),int(xx+cw),int(yy+rh)))
            xx+=cw
        yy+=rh


def render_pdf_preview(prs: Presentation) -> list[Image.Image]:
    pages=[]
    for slide in prs.slides:
        bg=(248,247,243)
        try:
            bg=_rgb(slide.background.fill.fore_color,bg)
        except Exception:
            pass
        image=Image.new("RGB",(PREVIEW_W,PREVIEW_H),bg)
        draw=ImageDraw.Draw(image)
        for shape in slide.shapes:
            rect=(_px(shape.left),_px(shape.top),_px(shape.left+shape.width),_px(shape.top+shape.height))
            if shape.has_chart:
                _draw_chart(draw,shape.chart,rect)
                continue
            if shape.has_table:
                _draw_table(draw,shape.table,rect)
                continue
            st=getattr(shape.shape_type,"name",str(shape.shape_type))
            if st=="LINE":
                draw.line(rect,fill=_rgb(shape.line.color,(49,91,137)),width=max(1,int((shape.line.width.pt if shape.line.width else 1.5)*PREVIEW_DPI/72)))
                continue
            if st=="AUTO_SHAPE":
                fill=_rgb(shape.fill.fore_color,(255,255,255))
                outline=_rgb(shape.line.color,(210,213,211))
                kind=getattr(getattr(shape,"auto_shape_type",None),"name","")
                if kind=="OVAL":
                    draw.ellipse(rect,fill=fill,outline=outline,width=1)
                elif kind=="ROUNDED_RECTANGLE":
                    draw.rounded_rectangle(rect,radius=max(5,int((rect[3]-rect[1])*.12)),fill=fill,outline=outline,width=1)
                else:
                    draw.rectangle(rect,fill=fill,outline=outline,width=1)
            if getattr(shape,"has_text_frame",False):
                _draw_text_frame(draw,shape.text_frame,rect)
        pages.append(image)
    pages[0].save(PDF_OUT,"PDF",resolution=PREVIEW_DPI,save_all=True,append_images=pages[1:],title="CtrlHGen teacher briefing — PDF preview")
    return pages


def main() -> None:
    evidence=load_evidence()
    prs=build_deck(evidence)
    validate(prs)
    write_markdown(prs)
    prs.save(PPTX_OUT)
    render_pdf_preview(prs)
    print(f"Wrote {PPTX_OUT}")
    print(f"Wrote {PDF_OUT}")
    print(f"Wrote {SCRIPT_OUT}")


if __name__ == "__main__":
    main()
