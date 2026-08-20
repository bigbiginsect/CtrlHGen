# Verifier-guided Best-of-N：P1/P2 实验记录

记录日期：2026-08-21（Asia/Shanghai）  
数据集/条件：WN18RR / `specific_relation`  
方案：`worklogs/SC-IDC/post_proposal.md`  
结论：P1 全部门槛通过；随后按冻结合同只运行一次 P2。sealed evaluation 上 K=4 相对 greedy 的
SemAvg 提高 `+0.10028`，exact 提高 `+0.08594`，branch-supported 提高 `+0.06130`。

## 1. 方法与论断边界

本轮冻结 single-target epoch-5 generator，不训练、不更新 optimizer，不运行 SFT/GRPO/reranker。每个
`(O,C)` 以固定 seed 生成候选，用 observable graph 执行候选，并按以下规则确定性选择：

```text
exact + branch-supported
  > exact + nominal
  > exact
  > highest SemAvg
  > model mean log probability
  > canonical hypothesis SHA256
```

选择器只读取 observation、observable graph、condition 和 candidate；不读取 reference hypothesis。结果只能
表述为 WN18RR / `specific_relation` 上的 inference-time decoding improvement，不能表述为 generator training
improvement。observable-graph exactness 不等于隐藏完整 KG 上的逻辑等价；nominal、branch-supported 和
non-root branch-supported 仍是不同强度的控制证据。

正式冻结值：

- seed：`314159`；
- sampling：`temperature=1.0, top_k=0, top_p=1.0`；
- P1：greedy、sample K=1/2/4/8；候选流按 candidate index 派生 seed，较小 K 是较大 K 的严格前缀；
- P2：只使用 K=4，并同时报告同一 sealed split 的 greedy 对照；
- batch size：`32`；max new tokens：`33`；
- checkpoint：`single_target/epoch-5`；tree SHA256
  `61f979aa25cc53eea1a7cea6ce39567e2cbfdc413a220821f6311fa641619cd8`。

## 2. 实现、部署与验证

实现提交：

- `711ad87aba7ac3b4493b1832ee292a52c4d047dc`：正式 P1/P2 入口、逐条 records、hash/lineage、gate 和测试；
- `8e27317934fffdffd2d9b414bf2f8829021e8d1c`：GPU smoke test 后收紧 checkpoint 合同，显式要求
  `stage=ss_csc_single_target` 并绑定 Experiment A summary hash。

正式实验使用 clean detached `8e27317934fffdffd2d9b414bf2f8829021e8d1c`。DSW 为单 NVIDIA L20；正式
运行前显存 1 MiB、utilization 0%，无遗留实验进程。

定向回归在禁用系统外部 pytest plugin 自动加载后为：

```text
32 passed in 3.09s
```

覆盖 `test_verifier_best_of_n.py`、`test_ss_csc.py` 和 `test_sc_idc_core.py`。默认 pytest 自动发现曾被系统级
Hydra/OmegaConf 版本冲突阻断，设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 后项目测试正常；这不是项目测试失败。
随后用 2 条 frozen validation 做 GPU smoke test，checkpoint reload、sampling、transition log-prob、graph
execution、selection 和 summary 均通过。smoke test 不写正式 artifact，也没有打开 sealed evaluation。

## 3. Experiment P1

正式命令：

```bash
python -m akgr.reproduction.verifier_best_of_n p1 \
  --experiment-config akgr/configs/reproduce/wn-specific-relation-full-train-author-aligned.yml \
  --phase2-preflight /mnt/workspace/ctrlhgen-runs/sc-idc-specific-relation-sft-preflight-20260815-af94eb4-a/sft-preflight.json \
  --pair-summary /mnt/workspace/ctrlhgen-runs/ss-csc-abc-20260820-ade65d2/experiment-a/summary.json \
  --checkpoint /mnt/workspace/ctrlhgen-checkpoints/ss-csc-abc-20260820-ade65d2/single_target/epoch-5 \
  --checkpoint-selection /mnt/workspace/ctrlhgen-runs/ss-csc-abc-20260820-ade65d2/experiment-b/single_target/best.json \
  --checkpoint-tree-sha256 61f979aa25cc53eea1a7cea6ce39567e2cbfdc413a220821f6311fa641619cd8 \
  --output-dir /mnt/workspace/ctrlhgen-runs/verifier-best-of-n-p1-20260821-8e27317 \
  --seed 314159 \
  --batch-size 32
```

运行时间 `174.24s`，状态 `passed`。P1 全程记录
`sealed_final_evaluation_loaded=false`。

### 3.1 Original validation（1,664 条，observable valid graph）

| decode | SemAvg | exact | nominal | branch | non-root | parse | EOS | mean unique AST |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 0.70874 | 0.26262 | 0.67368 | 0.50541 | 0.30288 | 0.99880 | 1.00000 | 0.99880 |
| sample K=1 | 0.66162 | 0.24459 | 0.66106 | 0.46214 | 0.28185 | 0.99940 | 1.00000 | 0.99940 |
| sample K=2 | 0.73937 | 0.29147 | 0.66947 | 0.52524 | 0.31971 | 1.00000 | 1.00000 | 1.68329 |
| **sample K=4** | **0.79618** | **0.34796** | **0.68209** | **0.57272** | **0.34075** | **1.00000** | **1.00000** | **2.80469** |
| sample K=8 | 0.83377 | 0.39543 | 0.70673 | 0.61719 | 0.37500 | 1.00000 | 1.00000 | 4.68750 |

K=4 累积 generation+verification wall time 为 `34.84s`，执行 `11,081` 次 graph executions，平均每个
candidate `6.48468` generation tokens。K=8 相应为 `69.84s`、`22,166` 次和 `6.48678` tokens。

### 3.2 Pair validation（1,005 pairs / 2,010 directional prompts，observable train graph）

| decode | SemAvg | exact | nominal | branch | non-root | parse | EOS | bilateral exact AST switch | condition-only switch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 0.98684 | 0.96816 | 0.59104 | 0.52786 | 0.40100 | 0.99900 | 1.00000 | 0.23582 | 0.14229 |
| sample K=1 | 0.94833 | 0.91194 | 0.55124 | 0.46766 | 0.34478 | 0.99701 | 1.00000 | 0.46368 | 0.13035 |
| sample K=2 | 0.99319 | 0.97910 | 0.68557 | 0.63134 | 0.47413 | 0.99851 | 1.00000 | 0.56020 | 0.30149 |
| **sample K=4** | **0.99733** | **0.98905** | **0.81095** | **0.76468** | **0.58706** | **0.99900** | **1.00000** | **0.66667** | **0.48756** |
| sample K=8 | 0.99960 | 0.99701 | 0.88358 | 0.85771 | 0.65871 | 1.00000 | 1.00000 | 0.77413 | 0.61990 |

K=4 wall time 为 `40.80s`，执行 `12,508` 次 graph executions，平均每个 candidate `6.94341`
generation tokens。K=8 相应为 `82.52s`、`25,068` 次和 `6.94502` tokens。

### 3.3 P1 gate

| gate | 结果 |
|---|---|
| original SemAvg 不低于 greedy | 通过：0.79618 > 0.70874 |
| original branch 不低于 greedy | 通过：0.57272 > 0.50541 |
| pair bilateral exact AST switch >= 0.66766 - 0.01 | 通过：0.66667 >= 0.65766 |
| original parse/EOS >= greedy - 0.005 | 通过：1.00000 / 1.00000 |
| pair parse/EOS >= greedy - 0.005 | 通过：0.99900 / 1.00000 |
| K=4 wall time / graph executions 完整记录 | 通过 |
| reference-free selection | 通过 |

总 gate：**passed**，因此允许进入 P2。

### 3.4 P1 artifacts

运行根目录：

```text
/mnt/workspace/ctrlhgen-runs/verifier-best-of-n-p1-20260821-8e27317/
```

| artifact | count | SHA256 |
|---|---:|---|
| `summary.json` | — | `8083e20d75d42dfb6e472e60efbb12efc1b5e9a26e34a2a34e494addeeb86ee3` |
| `original-validation-records.jsonl` | 1,664 | `321a3c1dd4074d28d55be970368e9f7b1b100e2d37dd7f6037b09176107c23d6` |
| `pair-validation-records.jsonl` | 2,010 | `7ce78740e461dd8474132cef33f8e9cfeaa96ca95a5bf6b99638e798050454ea` |

三项 hash 均在 P1 完成后重新读取并验证一致。

## 4. Experiment P2：一次 sealed evaluation

P2 在打开 sealed manifest 前强制验证：P1 status/gate、P1 summary 与 records hashes、相同 clean Git SHA、
checkpoint tree/selection/lineage、preflight hash，以及完整 method contract。没有 P2 seed/K/batch/checkpoint/
selection-rule 调整入口。

正式命令：

```bash
python -m akgr.reproduction.verifier_best_of_n p2 \
  --p1-summary /mnt/workspace/ctrlhgen-runs/verifier-best-of-n-p1-20260821-8e27317/summary.json \
  --output-dir /mnt/workspace/ctrlhgen-runs/verifier-best-of-n-p2-20260821-8e27317
```

运行时间 `42.96s`，状态 `completed`；这是本方案唯一一次 sealed evaluation，不根据结果调参或重跑。

| decode | SemAvg | exact | nominal | branch | non-root | parse | EOS | mean unique AST |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 0.69887 | 0.25240 | 0.65745 | 0.49579 | 0.29567 | 0.99880 | 1.00000 | 0.99880 |
| sample K=1 | 0.64790 | 0.23858 | 0.66166 | 0.44892 | 0.26803 | 0.99820 | 1.00000 | 0.99820 |
| sample K=2 | 0.73966 | 0.29748 | 0.65805 | 0.50661 | 0.29327 | 0.99940 | 1.00000 | 1.70553 |
| **sample K=4** | **0.79915** | **0.33834** | **0.67548** | **0.55709** | **0.32692** | **1.00000** | **1.00000** | **2.88341** |

K=4 相对 greedy：

- SemAvg `+0.10028`；
- exact `+0.08594`；
- nominal `+0.01803`；
- branch-supported `+0.06130`；
- non-root branch-supported `+0.03125`；
- parse `+0.00120`；EOS `+0.00000`。

K=4 wall time 为 `31.79s`，执行 `10,999` 次 graph executions，平均每 candidate `6.39709`
generation tokens；greedy 为 `8.27s`、`2,756` 次和 `6.47115` tokens。质量收益伴随约 4 倍候选生成与执行
开销，不能省略成本报告。

sealed input hashes：

| input | SHA256 |
|---|---|
| `final_evaluation.manifest.json` | `561217a80cc8dcfe6359f30431bfa4567344249e84e7525085aaf0b8cbf6060a` |
| `final_evaluation.conditions.jsonl` | `c9e5e4c948d9429077801cb2f0632be716029f6a6852a5a3fc46213f91e41020` |
| Phase 2 `sft-preflight.json` | `f8484c7050fd456d59a314b9ca205f52cd5ea7dc036c2f2b1ee1f2593fcfa97a` |

P2 运行根目录：

```text
/mnt/workspace/ctrlhgen-runs/verifier-best-of-n-p2-20260821-8e27317/
```

| artifact | count | SHA256 |
|---|---:|---|
| `summary.json` | — | `21ec03b56840a447cd93ae9bb027f0c2e10cdc5d41184d2427da144aeed1001e` |
| `final-evaluation-records.jsonl` | 1,664 | `6d6a159b1d4f5d0d4f76aaa7ef649c5bacd0fa32f00d4472b48d28054d81568d` |

两项 hash 均在 P2 完成后重新读取并验证一致。P2 summary 明确记录
`post_evaluation_tuning_permitted=false`。

## 5. 结论

正式 P1 复现了 post-failure diagnostic 的主要方向，且全部预注册 gate 通过。唯一一次 sealed P2 进一步显示：
冻结 generator 已包含可由 observable-graph verifier 找出的高语义受控候选，K=4 inference selection 同时提升
SemAvg、exact、branch-supported 和 non-root branch-supported，parse/EOS 不退化。

该结果支持把本改动报告为 **Verifier-guided Best-of-4 inference-time decoding improvement**。它不支持关于
generator 参数、训练目标或 predicate necessity 的更强论断，也不消除约 4 倍 generation/execution 成本。

## 6. DSW 收尾

P2 完成并核验后，L20 显存 1 MiB、utilization 0%，没有遗留 verifier/SFT/GRPO 实验进程。正式 P1/P2
实验代码 SHA 均为 `8e27317934fffdffd2d9b414bf2f8829021e8d1c`。

实例停止操作与官方 API 响应记录将在本文部署后执行，并写入 P2 运行根目录下的
`stop-instance-response.json`；调用 StopInstance，不调用 DeleteInstance，不删除持久化数据、checkpoint 或
runs。
