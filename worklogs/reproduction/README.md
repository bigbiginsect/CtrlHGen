# CtrlHGen 复现工作档案

这里集中保存 2026-08-08 至 2026-08-13 的 CtrlHGen 复现记录、汇报材料和可复核证据。
本页是后续回看或继续工作的统一入口；代码和运行方式以仓库根目录 README 与当前实现为准，
本目录中的日志主要记录当时实际运行的 commit、配置、结果和判断边界。

## 快速入口

| 想做什么 | 从哪里开始 | 用途 |
|---|---|---|
| 快速了解最终结果 | [`reports/reproduction-report-2026-08-12.md`](reports/reproduction-report-2026-08-12.md) | 完整复现报告、最终指标、论文对照、局限与结论 |
| 接手或追溯实验决策 | [`reports/requirement-gap.md`](reports/requirement-gap.md) | 阶段状态、配置契约、验收口径和历史交接信息 |
| 核验最终 frozen-test | [`phases/phase-d-repaired-frozen-test-2026-08-12.md`](phases/phase-d-repaired-frozen-test-2026-08-12.md) | 最终 SFT parent 与 repaired GRPO 的双解码对照 |
| 追溯 Phase C 四轮实验 | [`phases/phase-c-2026-08-10.md`](phases/phase-c-2026-08-10.md) | C-I 至 C-IV 的配置、曲线、指标、证据路径和 SHA256 |
| 准备讲解或汇报 | [`briefings/`](briefings/) | 中文讲稿、PDF、可编辑 PPTX 及含改进方案版本 |
| 重建图表或核对数字 | [`report-data/README.md`](report-data/README.md) | compact 训练记录、来源映射和完整性校验方法 |

## 结论快照

- 主任务为 WN18RR + pattern condition，最终模型是随机初始化的 6 层 GPT-2（hidden 768，12 heads），seed 42。
- 已跑通数据采样、unconditional SFT、conditional SFT、GRPO、greedy/sampled frozen-test 与产物审计。
- 最关键的两次修复是 conditional 监督契约修复，以及 repaired GRPO 中引入 fresh RL-only 数据并补齐 prompt 末尾 `SEP`。
- 最终 repaired GRPO 相对配对 SFT parent 的 Jaccard/Dice/Overlap 均值：greedy `+0.03625`，sampled `+0.04426`；两者的 paired bootstrap 95% CI 均严格大于 0。
- 结果支持“控制条件有效、RL 继续改善集合语义”的主要现象，但由于模型规模和论文披露差异，不应表述为论文绝对数值的逐点复刻。

以上仅是导航摘要；正式引用数值、置信区间和证据边界时，以综合报告和对应阶段日志为准。

## 目录结构

```text
worklogs/reproduction/
├── README.md                 # 本入口
├── reports/                  # 综合报告与执行交接
├── phases/                   # 按时间记录的 Phase A–D 实验日志
├── briefings/                # 讲稿、PDF 与 PPTX
├── figures/                  # 报告图、汇报图和论文截图
└── report-data/              # 用于复核与重建图表的 compact 原始记录
```

## 实验时间线

| 日期 | 记录 | 一句话定位 |
|---|---|---|
| 2026-08-08/09 | [`phase-a`](phases/phase-a-2026-08-08.md) | 建立配置、采样、训练、评估、checkpoint 与测试基础设施 |
| 2026-08-09 | [`phase-b`](phases/phase-b-2026-08-09.md) | WN18RR tiny 端到端 smoke；不构成论文规模复现结论 |
| 2026-08-09 | [`phase-c-preflight`](phases/phase-c-preflight-2026-08-09.md) | 固化验证、checkpoint、选择与确定性策略 |
| 2026-08-09–11 | [`phase-c`](phases/phase-c-2026-08-10.md) | 四轮 SFT；定位监督问题并确认训练覆盖扩容的主要作用 |
| 2026-08-11 | [`phase-d original`](phases/phase-d-2026-08-11.md) | 首轮正式 GRPO；保留为诊断基线，不是最终方法结果 |
| 2026-08-11 | [`phase-d repaired pilot`](phases/phase-d-pilot-2026-08-11.md) | 验证 fresh RL-only 数据与末尾 `SEP` 的联合修复 |
| 2026-08-11 | [`phase-d repaired full`](phases/phase-d-repaired-full-2026-08-11.md) | 13,000-step repaired GRPO validation 确认实验 |
| 2026-08-12 | [`phase-d frozen-test`](phases/phase-d-repaired-frozen-test-2026-08-12.md) | 唯一一次 repaired terminal 双解码 frozen-test |
| 2026-08-12/13 | [`综合报告`](reports/reproduction-report-2026-08-12.md) / [`汇报材料`](briefings/) | 汇总结论、证据边界、论文对照与后续改进方案 |

## 证据如何对应

1. 先从综合报告确认最终叙事、结果和限制。
2. 对有疑问的结论，进入对应 `phases/` 日志核对运行 SHA、配置、过程和原始路径。
3. 需要复算图表或检查关键数值时，使用 `report-data/`；其文件由
   [`sha256sums.txt`](report-data/sha256sums.txt) 索引。
4. 大 checkpoint、逐样本评测和完整 stdout 没有进入 Git；日志中记录的是当时 DSW
   持久盘路径。DSW 当前未开启，本次目录整理也没有访问或改动远端。

从仓库根目录重建综合报告图表：

```bash
python scripts/plot_reproduction_metrics.py
```

重建汇报专用图表或论文截图：

```bash
python scripts/plot_ctrlhgen_briefing_figures.py
python scripts/extract_ctrlhgen_paper_figures.py
```

## 回来查信息时的建议

- 查 commit、checkpoint、run id 或指标名：
  `rg -n '<关键词>' worklogs/reproduction`。
- 查最终方法结果时认准 `repaired` 和 2026-08-12 frozen-test；旧 Phase D 只作诊断对照。
- 历史日志保留当时判断，不用后续结论反向改写；冲突时优先采用日期更晚、证据层级更高的最终报告与 frozen-test 记录。
- 若要继续新实验，在新的工作日志目录中记录，不要把新结果混入这份已冻结的复现档案。
