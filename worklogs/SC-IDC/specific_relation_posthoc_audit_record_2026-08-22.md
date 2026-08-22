# Specific-relation selector 消融与独立图视角审计记录

日期：2026-08-22

最终状态：两项正式实验均完成；无历史测试集重跑，无模型更新，无事后调参。

## 1. 结论先行

这两项审计把 specific-relation 创新拆得更清楚了：

1. K=4 的主要收益来自 exact/semantic verifier 与 condition nominal awareness，不是 branch-supported tie-break 本身。
2. branch-aware 相对完全不看 condition 的 exact-semantic selector，能稳定提高 nominal、branch-supported 和 nonroot-branch-supported，但在原 P2 上完全不改变 Jaccard、Dice、Overlap、exact。
3. branch-aware 相对 nominal-aware 的边际非常小：原 P2 上 branch-supported 只增加 0.00120，95% CI 下界为 0；在 edge-disjoint valid-exclusive 且 reference 非空的 609 条上，两者选择结果在所有报告指标上完全相同。
4. condition-aware 选择并非只在 train 图上有效。冻结选择后切换到 cumulative-valid，branch-aware 相对 condition-blind selector 的 Jaccard 仍为 +0.00146，95% CI 为 [+0.00018, +0.00297]；但相对 nominal-aware 仍近似为零。
5. valid-exclusive 是很强的图分布改变：reference denotation 与 train reference 的平均 Jaccard 仅 0.00226；609 条非空子集也只有 0.00616。因此该视角适合作为敏感性 stress test，不能直接当作同一任务上的常规泛化集。

据此，论文/方案中的创新表述应聚焦为 condition-aware verifier-guided selection。branch-supported 可以保留为可解释约束或安全 tie-break，但当前证据不支持把它单独表述为语义质量提升的主要来源。

## 2. 冻结实现与环境

- 本地与 DSW 精确 Git SHA：cfff1e972024d29de48e9a0339cc61c22b5e39a4
- DSW：dsw-479226-7977d9bccf-t58m2
- GPU：NVIDIA L20，46,068 MiB
- Python 环境：/mnt/workspace/envs/ctrlhgen
- runner：akgr/reproduction/specific_relation_posthoc_audits.py
- tests：tests/test_specific_relation_posthoc_audits.py
- 执行前方案：worklogs/SC-IDC/specific_relation_posthoc_audit_proposal.md
- 关联测试：9 passed in 6.09s
- bootstrap：paired percentile bootstrap，seed 271828，10,000 次

正式输入：

- P2 summary SHA256：21ec03b56840a447cd93ae9bb027f0c2e10cdc5d41184d2427da144aeed1001e
- P2 records SHA256：6d6a159b1d4f5d0d4f76aaa7ef649c5bacd0fa32f00d4472b48d28054d81568d
- P2 records 数：1,664
- 新 graph audit generation seed：161803
- K：4
- sampling：temperature 1.0，top-k 0，top-p 1.0

## 3. 实验 A：specific-relation selector 离线消融

### 3.1 方法

不调用模型，不重新生成候选。每条记录严格复用原 P2 的同一组 K=4 candidates，只替换 selector：

- first-sample：固定第一个 sample。
- likelihood-only：只按模型 mean log probability。
- semantic-only：只按候选在 observation 上的 Jaccard/Dice/Overlap 等权排序，再用 likelihood/hash 打破平局。
- exact-semantic：exact 优先，其次 semantic；完全不读取 condition。
- nominal-aware：exact+nominal > exact > semantic；不读取 branch-supported。
- branch-aware：历史完整规则，exact+branch-supported > exact+nominal > exact > semantic。

正式运行逐条验证 branch-aware selected index 与历史 sample-k4 完全一致，mismatch count 为 0。

### 3.2 全量结果

所有 rate 和集合指标均为 [0,1]；n=1,664。

| selector | Jaccard | Dice | Overlap | exact | nominal | branch | nonroot | parse | EOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| first-sample | 0.58133 | 0.63721 | 0.72516 | 0.23858 | 0.66166 | 0.44892 | 0.26803 | 0.99820 | 1.00000 |
| likelihood-only | 0.61697 | 0.67642 | 0.77419 | 0.24940 | 0.66406 | 0.48197 | 0.29207 | 0.99940 | 1.00000 |
| semantic-only | 0.72873 | 0.78671 | 0.88202 | 0.33834 | 0.66226 | 0.54808 | 0.32031 | 1.00000 | 1.00000 |
| exact-semantic | 0.72873 | 0.78671 | 0.88202 | 0.33834 | 0.66226 | 0.54808 | 0.32031 | 1.00000 | 1.00000 |
| nominal-aware | 0.72873 | 0.78671 | 0.88202 | 0.33834 | 0.67548 | 0.55589 | 0.32572 | 1.00000 | 1.00000 |
| branch-aware | 0.72873 | 0.78671 | 0.88202 | 0.33834 | 0.67548 | 0.55709 | 0.32692 | 1.00000 | 1.00000 |

semantic-only 与 exact-semantic 在本批候选上完全相同，说明最高 semantic candidate 已经总是落在最高 exact tier；这不是一般规律，只是这批冻结 candidates 的经验现象。

### 3.3 主对比：branch-aware 减 exact-semantic

| 指标 | mean delta | 95% CI | win / tie / loss |
|---|---:|---:|---:|
| Jaccard | 0 | [0, 0] | 0 / 1.0000 / 0 |
| Dice | 0 | [0, 0] | 0 / 1.0000 / 0 |
| Overlap | 0 | [0, 0] | 0 / 1.0000 / 0 |
| exact | 0 | [0, 0] | 0 / 1.0000 / 0 |
| nominal | +0.01322 | [+0.00781, +0.01923] | 0.0132 / 0.9868 / 0 |
| branch-supported | +0.00901 | [+0.00481, +0.01382] | 0.0090 / 0.9910 / 0 |
| nonroot-branch | +0.00661 | [+0.00301, +0.01082] | 0.0066 / 0.9934 / 0 |
| parse | 0 | [0, 0] | 0 / 1.0000 / 0 |
| EOS | 0 | [0, 0] | 0 / 1.0000 / 0 |

### 3.4 主对比：branch-aware 减 nominal-aware

| 指标 | mean delta | 95% CI | win / tie / loss |
|---|---:|---:|---:|
| Jaccard / Dice / Overlap / exact | 0 | [0, 0] | 0 / 1.0000 / 0 |
| nominal | 0 | [0, 0] | 0 / 1.0000 / 0 |
| branch-supported | +0.00120 | [0, +0.00301] | 0.0012 / 0.9988 / 0 |
| nonroot-branch | +0.00120 | [0, +0.00301] | 0.0012 / 0.9988 / 0 |
| parse / EOS | 0 | [0, 0] | 0 / 1.0000 / 0 |

解释：condition nominal awareness 有可辨识贡献；branch-supported 只改变约 0.12% 样本，且没有带来新的 denotation 指标提升。

## 4. 实验 B：独立图视角 anti-overfitting audit

### 4.1 图拆分合同

| view | edges | nodes | edge SHA256 |
|---|---:|---:|---|
| train selection | 148,132 | 39,011 | 3e1b6d27979fee08e9ec446368ea5fa449315202b0ede83c14f4815bd4dbd6df |
| valid cumulative | 166,648 | 39,905 | 2ab8a6b63a6494dc46d3986ef054fc8931def05122f5fa7117396a2de5bb602c |
| valid exclusive | 18,516 | 39,905 | 25b21d79f2da16d1fb4a294fa2a5a0650623e7f8e7b74045a10be33620928d07 |

三项合同检查全部通过：

- train edges 与 valid-exclusive edges 严格不相交；
- 二者并集精确等于 cumulative-valid；
- valid-exclusive 保留 cumulative-valid 的全部实体 node universe，避免 negation universe 改变。

### 4.2 样本资格

- 冻结 validation：1,664
- train reference 非空：1,664
- 因 train answers 超过 max-answers 排除：4
- 正式生成：1,660，覆盖率 0.99760
- valid-exclusive reference 非空：609
- valid-exclusive reference 为空：1,051，比例 0.63313

reference 本身的视角漂移：

| 对比 | 全部 1,660 的平均 reference Jaccard | exclusive 非空 609 条 |
|---|---:|---:|
| train vs cumulative-valid | 0.70265 | 0.73615 |
| train vs valid-exclusive | 0.00226 | 0.00616 |

这说明 valid-exclusive 几乎是全新的关系边语义环境。所有 1,660 条上的高 exact 会被大量“双空集合”抬高，因此主要解释必须看 609 条 reference 非空子集。

### 4.3 selection/train 结果

n=1,660；候选生成、candidate audit 和 selected index 冻结都只使用 train view。

| selector | Jaccard | Dice | Overlap | exact | nominal | branch | nonroot | parse | EOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| first-sample | 0.74538 | 0.78062 | 0.83154 | 0.55060 | 0.71627 | 0.57410 | 0.30843 | 0.99880 | 1.00000 |
| likelihood-only | 0.79650 | 0.83399 | 0.88403 | 0.59277 | 0.72229 | 0.61506 | 0.33554 | 1.00000 | 1.00000 |
| semantic/exact-semantic | 0.84169 | 0.87661 | 0.92828 | 0.64277 | 0.72651 | 0.64518 | 0.34458 | 1.00000 | 1.00000 |
| nominal-aware | 0.84169 | 0.87661 | 0.92828 | 0.64277 | 0.75000 | 0.66205 | 0.35361 | 1.00000 | 1.00000 |
| branch-aware | 0.84169 | 0.87661 | 0.92828 | 0.64277 | 0.75000 | 0.66747 | 0.35723 | 1.00000 | 1.00000 |

branch-aware 减 exact-semantic：branch +0.02229，95% CI [+0.01566,+0.02952]；nonroot +0.01265，95% CI [+0.00783,+0.01807]。集合指标和 exact 全部为 0 差值。

branch-aware 减 nominal-aware：branch +0.00542，95% CI [+0.00241,+0.00904]；nonroot +0.00361，95% CI [+0.00120,+0.00663]。集合指标和 exact 全部为 0 差值。

### 4.4 冻结选择后的 cumulative-valid audit

| selector | Jaccard | Dice | Overlap | exact | nominal | branch | nonroot | parse | EOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| first-sample | 0.66045 | 0.71959 | 0.82055 | 0.35000 | 0.71627 | 0.56145 | 0.29578 | 0.99880 | 1.00000 |
| likelihood-only | 0.70776 | 0.77004 | 0.87518 | 0.38313 | 0.72229 | 0.59940 | 0.32108 | 1.00000 | 1.00000 |
| semantic/exact-semantic | 0.74590 | 0.80780 | 0.91666 | 0.41687 | 0.72651 | 0.62590 | 0.32711 | 1.00000 | 1.00000 |
| nominal-aware | 0.74716 | 0.80879 | 0.91731 | 0.41867 | 0.75000 | 0.64217 | 0.33554 | 1.00000 | 1.00000 |
| branch-aware | 0.74736 | 0.80889 | 0.91701 | 0.41928 | 0.75000 | 0.64759 | 0.33916 | 1.00000 | 1.00000 |

branch-aware 减 exact-semantic：

- Jaccard +0.00146，95% CI [+0.00018,+0.00297]
- Dice +0.00109，95% CI [+0.00019,+0.00215]
- Overlap +0.00035，95% CI [-0.00075,+0.00156]
- exact +0.00241，95% CI [0,+0.00542]
- branch +0.02169，95% CI [+0.01506,+0.02892]
- nonroot +0.01205，95% CI [+0.00723,+0.01747]

branch-aware 减 nominal-aware：

- Jaccard +0.00020，95% CI [-0.00030,+0.00090]
- Dice +0.00010，95% CI [-0.00030,+0.00060]
- Overlap -0.00030，95% CI [-0.00090,+0.00000001]
- exact +0.00060，95% CI [0,+0.00181]
- branch +0.00542，95% CI [+0.00241,+0.00904]

因此 condition awareness 在扩图后保留了很小但可辨识的集合质量收益；branch-aware 相对 nominal-aware 没有稳健的集合质量优势。

### 4.5 valid-exclusive audit

全部 1,660 条上 branch-aware 为：

- Jaccard 0.71638，Dice 0.71771，Overlap 0.19940，exact 0.71084
- nominal 0.75000，branch 0.25482，nonroot 0.20843
- parse 1.00000，EOS 1.00000

exact 很高而 Overlap 很低，是 1,051 个 empty references 导致的“双空 exact”现象，不能按表面 exact 解读。

更可信的 reference 非空子集 n=609：

| selector | Jaccard | Dice | Overlap | exact | nominal | branch | nonroot | parse | EOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| first-sample | 0.48878 | 0.49288 | 0.49972 | 0.47291 | 0.75534 | 0.29721 | 0.18062 | 0.99672 | 1.00000 |
| likelihood-only | 0.51440 | 0.51871 | 0.52627 | 0.49754 | 0.75534 | 0.30870 | 0.19212 | 1.00000 | 1.00000 |
| semantic/exact-semantic | 0.53124 | 0.53513 | 0.54351 | 0.51560 | 0.76190 | 0.32348 | 0.19704 | 1.00000 | 1.00000 |
| nominal-aware | 0.53233 | 0.53595 | 0.54351 | 0.51724 | 0.77668 | 0.32677 | 0.20033 | 1.00000 | 1.00000 |
| branch-aware | 0.53233 | 0.53595 | 0.54351 | 0.51724 | 0.77668 | 0.32677 | 0.20033 | 1.00000 | 1.00000 |

branch-aware 减 exact-semantic：

- Jaccard +0.00109，95% CI [-0.00383,+0.00602]
- Dice +0.00082，95% CI [-0.00411,+0.00575]
- Overlap 0，95% CI [-0.00493,+0.00493]
- exact +0.00164，95% CI [-0.00328,+0.00657]
- nominal +0.01478，95% CI [+0.00657,+0.02463]
- branch +0.00328，95% CI [0,+0.00821]

branch-aware 减 nominal-aware：九项报告指标全部为 0，所有 bootstrap CI 都为 [0,0]。

### 4.6 branch-aware exact 转移

train 到 cumulative-valid，1,660 条：

| train exact | cumulative exact | count |
|---:|---:|---:|
| 0 | 0 | 573 |
| 0 | 1 | 20 |
| 1 | 0 | 391 |
| 1 | 1 | 676 |

train exact 中有 676/1,067 保持 exact，391 条因加入 valid edges 后失去 exact。这个下降不仅是 selector overfitting，也包含 reference 与 candidate query 在扩图后的不同 denotation 漂移。

train 到 valid-exclusive，reference 非空 609 条：

| train exact | exclusive exact | count |
|---:|---:|---:|
| 0 | 0 | 131 |
| 0 | 1 | 123 |
| 1 | 0 | 163 |
| 1 | 1 | 192 |

由于 reference train/exclusive 平均 Jaccard 只有 0.00616，这个转移更接近跨边分布 stress test，不能当作同一 target 的保持率。

## 5. 成本、artifact 与异常记录

离线 selector：

- 输出目录：/mnt/workspace/ctrlhgen-runs/specific-relation-selector-ablation-20260822-cfff1e9
- summary SHA256：12ba37f19ac4bff1337fc5ba7ad170bbeeb4383d3c28e378da36bd0841a610c3
- records SHA256：087aca4061f3c0617b8a67e13adc5c81cc0e31d55b51ea51dc7032a753e0f229
- records：1,664
- generation：0

正式 graph audit：

- 输出目录：/mnt/workspace/ctrlhgen-runs/specific-relation-graph-audit-20260822-cfff1e9
- summary SHA256：5629582ae2cbfc463ff59e094ea763bd0f788b1d842142b361ad77921eb7f044
- records SHA256：4bbda1ea353d5c4e0d8585634390fe8fa5b0f75989b833258de504cd2db9674e
- records：1,660
- total wall：67.05 s
- greedy：1,660 sequences，10,708 tokens，6.62 s generation+verification
- K=4 samples：6,640 sequences，42,394 tokens，25.58 s generation+verification

smoke：

- 第一次 smoke 在进入模型/数据前因缺少 CTRLHGEN_DATA_ROOT、CTRLHGEN_CHECKPOINT_ROOT、CTRLHGEN_RUN_ROOT 明确失败；failed status 已保留在 specific-relation-graph-audit-smoke-20260822-cfff1e9。
- 补齐既有冻结路径变量后，第二次 2-record smoke 完成，图合同、checkpoint reload、CUDA generation、三视角 audit 全部通过。
- 正式任务使用新输出目录执行，未复用或覆盖失败目录。

## 6. 最终判断

### Selector 离线消融

通过，但结论是否定性收窄：完整 branch-aware 能保证更高的 branch-supported rate，却没有在原 P2 上提高任何集合 denotation 指标；相对 nominal-aware 的额外作用只覆盖极少样本。

### Anti-overfitting audit

没有发现 condition-aware selector 在扩图后完全失效：相对 condition-blind baseline 的方向保持为正。另一方面，edge-disjoint 非空子集上 branch-aware 与 nominal-aware 完全一致，说明 branch-supported tie-break 没有展示独立跨视角优势。

当前最稳妥的创新主张：

- 强主张：verifier-guided K=4 selection 与 condition-aware exact selection 有实证价值。
- 中等主张：condition-aware selection 在 cumulative-valid 扩图下仍有小幅稳定收益。
- 弱主张：branch-supported 提供可解释、单调的控制质量排序。
- 不应主张：branch-supported 已被证明能独立提高 denotation accuracy 或带来强跨图泛化。
