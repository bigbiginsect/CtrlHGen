# Phase D repaired full run：fresh RL-only + SEP

> 状态日期：2026-08-11（Asia/Shanghai）
>
> 运行代码：`6b26ce8621d005cf62524c3997c18ea6e03a68ce`
>
> DSW 根目录：`/mnt/workspace/ctrlhgen-runs/phase-d-repaired-full-20260811`

## 1. 结论

repaired Phase D 在同一个 6×768 C-IV conditional epoch 45 parent 上完成 13,000 steps，
没有触发预注册早停。terminal validation 的 greedy 与 sampled Jaccard/Dice/Overlap 全部
改善，三项均值增益分别为 `+0.02884` 和 `+0.03547`，配对 bootstrap 95% CI 均严格大于
0；condition、parse 和 EOS 同时保持健康。

这轮结果强化了 repaired pilot 的原因判断：原 Phase D 低增益的首要瓶颈是复用高度拟合
的 SFT train 导致 group reward 饱和，以及 raw GRPO prompt 缺失 target-boundary `SEP`。
6 层容量不是 GRPO 基本训不动的主要原因。模型深度仍可作为后续论文忠实度/绝对指标消融，
但不再是修复 Phase D 的前置条件。

本轮只使用 validation 做中途 probe 和 terminal 比较，没有打开 test artifact，也没有生成
test-named 文件，因此它仍是 validation-confirmed repaired run，不是新的 frozen-test 结果。

## 2. 冻结设置与数据

- parent：C-IV `conditional-epoch-45`，经 `phase-d-parent` stable pointer；
- prompt：`answers COND pattern SEP`；
- fresh sampling namespace：`phase_d_repaired_full_v1`；
- 额外排除 repaired pilot 的 13,312 条 query；
- GRPO：group 4、batch 32、1 epoch、13,000 steps、AdamW、LR `1e-5` 线性衰减、
  beta `0.1`、epsilon `0.2`；
- reward：`Jaccard + 0.5*Dice + 0.5*Overlap + condition`。

fresh RL-only 数据为 13 pattern × 8,000，共 104,000 条：

| 审计项 | 结果 |
|---|---:|
| unique query / supervision | 104,000 / 104,000 |
| 内部 query / supervision duplicates | 0 / 0 |
| 与 merged SFT train、validation、pilot overlap | 0 |
| forbidden query set | 175,488 |
| 因 forbidden overlap 剔除 | 27,927 |
| 因新集合内部 query duplicate 剔除 | 4,120 |
| sampling rounds | 30 |

fresh JSONL SHA256：`6263a2d0ece6d897e850dbcaf71893865cb038e198dbd95b576d8b1609ad9260`。

## 3. Step 0 rollout gate

在 256 个 group、1,024 个 completions 上：

| 指标 | 结果 | gate |
|---|---:|---:|
| zero reward-variance group rate | 0.60547 | ≤0.70 |
| all strings identical group rate | 0.37891 | 仅报告 |
| mean unique strings / group | 2.23828 | ≥1.50 |
| parse | 0.99707 | ≥0.95 |
| EOS | 1.00000 | ≥0.98 |
| condition | 0.98242 | 仅报告 |
| combined reward | 2.16081 | 仅报告 |

全部 gate 通过后才开始训练。相较 repaired pilot 的 63.7% 零方差率，独立 full fresh shard
为 60.5%；信号仍稀疏，但足以训练。

## 4. 训练与 20 分钟 probes

采样约 4 分钟，GRPO 训练 4,094.9 秒（68.2 分钟），总墙钟约 73.4 分钟。训练全程无
OOM、NaN、中断或旧 Phase D 中的极端 KL 聚合离群：500-step 日志 KL 范围
`[0.0904, 0.3780]`，reward 范围 `[2.1528, 2.2385]`，reward std 范围
`[0.2067, 0.2318]`，grad norm 范围 `[1.456, 19.481]`。

按固定规则每约 20 分钟只在 validation 做 probe：

| checkpoint | greedy 三项均值 delta | 95% CI | sampled delta | 95% CI | 决策 |
|---:|---:|---:|---:|---:|---|
| 6,800 | +0.02631 | [0.01592, 0.03682] | +0.03616 | [0.02069, 0.05200] | continue |
| 10,600 | +0.02445 | [0.01465, 0.03428] | +0.02909 | [0.01388, 0.04464] | continue |
| 13,000 | +0.02884 | [0.01897, 0.03885] | +0.03547 | [0.02017, 0.05091] | completed |

step 10,600 相对 6,800 有小幅回落，但仍远高于停止门槛；terminal 又恢复到接近或高于
早期峰值，因此继续到 13,000 是合理的。

## 5. Terminal validation

### Greedy

| 指标 | parent | repaired terminal | delta |
|---|---:|---:|---:|
| Jaccard | 0.61964 | 0.64536 | +0.02572 |
| Dice | 0.67458 | 0.70185 | +0.02727 |
| Overlap | 0.74546 | 0.77900 | +0.03354 |
| 三项均值 | 0.67989 | 0.70874 | **+0.02884** |
| Pattern Accuracy | 0.95913 | 0.96815 | +0.00901 |
| Smatch | 0.82834 | 0.83028 | +0.00194 |
| Parse / EOS | 0.99940 / 1.00000 | 1.00000 / 1.00000 | +0.00060 / 0 |

三项均值 paired bootstrap 10,000 次 95% CI：`[0.01897, 0.03885]`。

### Sampled

| 指标 | parent | repaired terminal | delta |
|---|---:|---:|---:|
| Jaccard | 0.57918 | 0.61093 | +0.03175 |
| Dice | 0.63105 | 0.66335 | +0.03230 |
| Overlap | 0.69896 | 0.74131 | +0.04235 |
| 三项均值 | 0.63640 | 0.67186 | **+0.03547** |
| Pattern Accuracy | 0.96214 | 0.96695 | +0.00481 |
| Smatch | 0.82180 | 0.82536 | +0.00356 |
| Parse / EOS | 0.99880 / 1.00000 | 1.00000 / 1.00000 | +0.00120 / 0 |

三项均值 paired bootstrap 10,000 次 95% CI：`[0.02017, 0.05091]`。

parent→terminal 参数 relative L2 delta 为 `0.01756`（1.756%）。

## 6. 独立终端审计

- parent/pilot × greedy/sampled 四份 JSONL 各有 1,664 个唯一 record；
- 从 JSONL 独立重算的全部指标与 metrics JSON 在 `1e-12` 内一致；
- terminal safetensors 的 76 个张量全部有限；
- `checkpoint-13000/model.safetensors` 与 `final-model/model.safetensors` SHA256 完全一致；
- 进程正常退出，GPU 释放；
- run 内无名称包含 `test` 的文件，result 明确记录 test artifact access 为 false。

## 7. 关键 SHA256

| artifact | SHA256 |
|---|---|
| `pilot-result.json` | `91720f5e397c9b86a4a87910ec8b52dc72487fba340e3c87e453107bcc31bff3` |
| `validation-comparison.json` | `55d46a78c57ea9ffa1cc35216c34d65be25a2b2e3916e39843706ab114c93899` |
| `rollout-signal-audit.json` | `0a93053fa85b589b8e507e906bece45b7406094705b6dc185f6fd685cbf52d83` |
| `fresh-rl-manifest.json` | `0c5a096b7b3f40534d63dac8460e45939b56614d3bf8fe0bb0b9225a8f394afa` |
| `fresh-rl-train.jsonl` | `6263a2d0ece6d897e850dbcaf71893865cb038e198dbd95b576d8b1609ad9260` |
| terminal model | `674e6dae7e24fb891f4e6b2db6bd3c221ef0c1f1f8d71f8fabfc6aaa50a3eba2` |

## 8. 下一步边界

本轮已经足以把 repaired 6-layer Phase D 作为有效的 validation-confirmed 方法结果。若需要
新的最终 test 数字，应先冻结本 checkpoint 和报告规则，再进行唯一一次 repaired frozen-test；
不要继续根据同一 validation 选择更多 checkpoint。12 层 Phase C-V 现在只服务于容量和论文
绝对值问题，不应用来解释或修复本轮已经解决的 GRPO 信号问题。
