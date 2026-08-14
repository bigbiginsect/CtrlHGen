# Phase D repaired pilot：fresh RL-only 数据与 SEP 契约

> 状态日期：2026-08-11（Asia/Shanghai）
>
> 代码 commit：`d7da9ac017b8e1aa57a0c586cac9a9ab9db6961d`
>
> DSW 控制目录：`/mnt/workspace/ctrlhgen-runs/phase-d-repaired-pilot-20260811/control`

## 1. 目的与边界

本 pilot 用同一个 6×768、C-IV conditional epoch 45 parent 检验以下联合修复能否恢复有效
GRPO 信号：

1. raw GRPO prompt 显式以 target-boundary `SEP` 结尾，使优化前缀与 SFT/evaluation 的
   causal contract 一致；
2. GRPO 不再复用已训练 50 epoch 的 base SFT train，而是使用从 train graph 新采样的、
   与 merged SFT train 和 validation 按 `query+pattern` 严格去重的 RL-only 数据。

除上述两项和短程预算外，沿用正式 Phase D 的 group 4、batch 32、AdamW、LR `1e-5`
线性衰减、beta `0.1`、epsilon `0.2` 和 reward 权重。pilot 不打开 test artifact，不做 test
evaluation；parent 与 terminal 只在同一 1,664 条 validation 上各做一次 greedy 和 sampled
对照。因此本轮不能把 SEP 和 fresh data 的贡献彼此完全分离，也不能作为正式 test 结论。

## 2. 实现与验证

- 新增 `build_generation_prompt()`，明确区分 raw generation prefix 与依赖 tokenizer pair
  post-processor 的 SFT prompt；正式 `run_grpo()` 也改为使用完整的 `... SEP` 前缀。
- 新增隔离入口 `akgr.reproduction.phase_d_pilot`：fresh data、rollout gate、短程训练、
  validation 比较、bootstrap、模型参数变化和状态文件均自动归档。
- DSW 在精确 commit 上通过相关测试 21/21、全套测试 113/113。
- 终端后独立从四份 validation JSONL 重算，每份均为 1,664 个唯一 record，重算结果与
  metrics JSON 在 `1e-12` 内完全一致；进程正常退出，GPU 回到空闲。

## 3. Fresh RL-only 数据

每个 13 个 pattern 新采样 1,024 条，共 13,312 条，相当于 1,664 个 GRPO optimizer
step 的一次遍历。排除集合包含 204,610 条 merged SFT train 和 1,664 条 validation；没有
读取 test records。

| 项目 | 结果 |
|---|---:|
| fresh records | 13,312 |
| 每 pattern | 1,024 |
| unique queries | 13,312 |
| 内部 query/supervision duplicate | 0 / 0 |
| 与 SFT train 或 validation overlap | 0 |
| 因旧数据重合而剔除 | 3,000 |
| 因新数据内部 query 重复而剔除 | 72 |

数据 SHA256：`296c88cb97336a51e4f769be6b4b9a756210ffd5571e5be846a691234ba6f1a4`。
达到平衡集合用了 18 个补采样 round；这和原 Phase D 在已见 SFT 数据上 reward 饱和、
组内方差不足的诊断一致。

## 4. Optimizer step 0 rollout gate

在 256 个 prompt group、1,024 个 sampled completions 上，用真实 TRL 0.16 generation 与
reward 路径审计：

| 指标 | 结果 | gate |
|---|---:|---:|
| zero reward-variance group rate | 0.63672 | ≤0.70 |
| all strings identical group rate | 0.37109 | 仅报告 |
| mean unique strings / group | 2.23828 | ≥1.50 |
| parse rate | 0.99707 | ≥0.95 |
| EOS rate | 1.00000 | ≥0.98 |
| condition accuracy | 0.98145 | 仅报告 |
| mean combined reward | 2.19424 / 3 | 仅报告 |

全部 gate 通过后才执行第一个 optimizer step。fresh+SEP 明显恢复了可用信号，但仍有
63.7% 的 group 是零 reward 方差，说明 group-relative RL 的信号稀疏问题没有完全消失。

## 5. 训练动力学

| step | LR | reward | reward std | KL | grad norm |
|---:|---:|---:|---:|---:|---:|
| 500 | 6.995e-6 | 2.13935 | 0.22437 | 0.08555 | 13.8540 |
| 1,000 | 3.990e-6 | 2.16164 | 0.22461 | 0.11039 | 5.4551 |
| 1,500 | 9.856e-7 | 2.19775 | 0.22002 | 0.09358 | 9.3347 |
| terminal 1,664 | 0 | 2.11460 | 0.23362 | 0.08245 | — |

训练 524.9 秒正常结束，无 OOM、NaN、中断或大 KL 聚合离群。parent→pilot 的模型参数
relative L2 delta 为 `0.006909`（0.69%），排除了“训练入口完全没有更新”的解释。

## 6. Validation-only 结果

### Greedy

| 指标 | parent | pilot | delta |
|---|---:|---:|---:|
| Jaccard | 0.61964 | 0.62548 | +0.00584 |
| Dice | 0.67458 | 0.68006 | +0.00548 |
| Overlap | 0.74546 | 0.75184 | +0.00638 |
| 三项均值 | 0.67989 | 0.68579 | **+0.00590** |
| Pattern Accuracy | 0.95913 | 0.96154 | +0.00240 |
| Smatch | 0.82834 | 0.82845 | +0.00011 |
| Parse / EOS | 0.99940 / 1.00000 | 0.99940 / 1.00000 | 0 / 0 |

三项均值的 paired bootstrap 10,000 次 95% CI 为 `[-0.00173, 0.01326]`。

### Sampled

| 指标 | parent | pilot | delta |
|---|---:|---:|---:|
| Jaccard | 0.57918 | 0.59286 | +0.01368 |
| Dice | 0.63105 | 0.64454 | +0.01349 |
| Overlap | 0.69896 | 0.71518 | +0.01623 |
| 三项均值 | 0.63640 | 0.65086 | **+0.01446** |
| Pattern Accuracy | 0.96214 | 0.96274 | +0.00060 |
| Smatch | 0.82180 | 0.82454 | +0.00274 |
| Parse / EOS | 0.99880 / 1.00000 | 0.99880 / 1.00000 | 0 / 0 |

三项均值的 paired bootstrap 10,000 次 95% CI 为 `[-0.00006, 0.02905]`。

pilot 达到预注册的 `greedy 三项均值 +0.005 且 sampled 同向、健康指标不退化`，因此自动
分类为 `promising_repaired_phase_d`。但两种解码的 CI 下界仍略跨 0，结论是“修复方向有
效且值得确认”，不是“单次 pilot 已统计显著”。

## 7. 原因判断与下一步

本轮直接否定了“6 层模型太浅，所以 Phase D 根本训不动”作为主要解释：同一个 6 层
parent 在只恢复正确 prompt contract 和未见 RL 数据后，1,664 步内即可在三项 semantic
metric 上一致改善。模型深度仍可能限制 Phase C 的绝对上限、复杂 pattern 和论文配置
忠实度，但不是已观察到 Phase D 低增益的首要瓶颈。

当前原因优先级为：

1. **主要：SFT train 复用导致 group reward 饱和和零 advantage。** conditional SFT 已在
   同一 base train 上训练 50 epoch，GRPO 再使用这些 prompt 时很难得到组内排序信号。
2. **主要且确定：raw GRPO prompt 缺失 SEP，训练和 evaluation 前缀不一致。** SEP 单独
   不能解决已见数据饱和，但缺失 SEP 会让策略优化一个 off-contract 分布。
3. **次要：group 4 下信号依然稀疏。** fresh+SEP 后仍有 63.7% 零方差 group；后续可考虑
   更大的 group 或更有分辨率的 reward，但不应在确认实验中同时改动。
4. **次要：6 层容量。** 它可能解释与论文绝对值的剩余差距，却不能解释本轮前后的巨大
   信号差异。

行动顺序建议：

1. 先补两个独立 RL-data seed 的同预算 validation-only replicate；保持 parent、SEP、
   optimizer/reward 和 1,664 steps 不变。若至少 2/3 seed 的 greedy 与 sampled 三项均值
   同向且健康 gate 通过，再冻结 repaired Phase D 正式方案。
2. 正式 repaired Phase D 使用彼此去重的 fresh shards，并仅在 terminal 做一次 frozen
   test；不要再使用 SFT base train，也不要在确认阶段同时调 group/reward。
3. 暂不把“12 层 Phase C 第五次”作为修复 Phase D 的前置条件。若项目目标升级为追论文
   绝对指标，再单独执行 12 层 depth-only C-V；这应被表述为容量/论文忠实度消融，而不是
   对当前 GRPO 失效的修复。

## 8. 关键证据 SHA256

| artifact | SHA256 |
|---|---|
| `pilot-result.json` | `8dd2f3f4a6cd8b39944786d088447b7a2c61246b0aebe40f88061b60478c89fc` |
| `validation-comparison.json` | `cdbd655b405f85f02c2f9112897ea28eaa389b8209d4f35c83bc7653de0b661d` |
| `rollout-signal-audit.json` | `c3b41c72404e132fdb407272f08c07d0daf17262a8955b0088a189eb6bc38409` |
| `fresh-rl-manifest.json` | `0a56bf61282c5be188febcb221963c644f8eb10ba629d8e43075c719d62b6847` |
| `fresh-rl-train.jsonl` | `296c88cb97336a51e4f769be6b4b9a756210ffd5571e5be846a691234ba6f1a4` |
| terminal model | `800047ca87b3b9a51cead5b44ce026c3365846721cb1adc06f9d0874f1f5405a` |
