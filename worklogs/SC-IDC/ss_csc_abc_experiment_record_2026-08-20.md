# SS-CSC Experiments A/B/C 实验记录

记录日期：2026-08-20（Asia/Shanghai）  
数据集/条件：WN18RR / `specific_relation`  
结论：Experiment A 通过；Experiment B/C 显示直接控制显著增强，但预注册总 gate 未通过，停止后续 GRPO。

## 1. 结论

天然多解数据路线在数据层面成立。仅使用 SFT base train，104,000/104,000 hypotheses 均在 train graph
精确执行回 observation；冻结得到 8,897 个 train pairs 和 1,005 个 observation-disjoint pair-validation
pairs，超过 5,000 对门槛。

paired supervision 对“condition 是否选择解”非常有效，但带来明显分布/语义 trade-off：

- single-target baseline 的 pair selection accuracy 为 `0.58408`；
- paired-data SFT 提高到 `0.83532`；
- SS-CSC 进一步提高到 `0.85721`，相对 paired-data SFT 为 `+0.02189`，paired 95% CI
  `[+0.00995,+0.03433]`；
- symmetric margin satisfaction 从 paired-data SFT 的 `0.54527` 提高到 SS-CSC 的 `0.62985`，增量
  `+0.08458`，95% CI `[+0.06070,+0.10846]`；
- 但 SS-CSC 原 1,664 条 validation assigned SemAvg 仅 `0.61248`，相对最佳 single-target
  `0.70874` 下降 `-0.09627`，远超 `-0.01` 容忍线。

预注册 gate 中 selection absolute gain 和 original-validation semantic 两项失败，Experiment C 状态为
`failed_gate`。因此不运行可选 Experiment D/GRPO，不打开 sealed final evaluation，也不据此声称 SS-CSC
已成为有效的整体模型改进。

## 2. 代码、配置和运行根目录

实现提交：

- `081e746537dddb1ca7e8ba44e6a84b6c62bce0f1`：实现 A/B/C；
- `ade65d26d0740dce3f67b880852c1f65c38d5345`：修正 branch-marginal 测试夹具；Experiment A 使用该 SHA；
- `45af7901c15d937fd6762d9ec8e7581b81097576`：修正 target tokenization，并启用严格 deterministic
  CUDA math backend；正式 B/C 使用该 SHA。

DSW 定向回归：`34 passed in 3.56s`，覆盖 SS-CSC、SC-IDC executor 和 Phase 2 数据合同。

冻结 pilot config：

```text
akgr/configs/ss_csc/wn-specific-relation-pilot.json
```

主要预注册值：seed 42、每 observation 最多 2 pairs、hash-mod-10 observation split、10 epochs、每 5
epochs 评价、每 epoch 112 optimizer updates、margin `0.2`、`lambda_swap=0.1`。三个分支每 epoch 均看
17,794 个 assigned targets；single-target/paired-SFT 使用 160 examples/batch，SS-CSC 使用 80 pairs/batch。

共同 unconditional parent：

```text
/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-full-train-author-aligned-c4/unconditional-epoch-45
```

权威运行根目录：

```text
/mnt/workspace/ctrlhgen-runs/ss-csc-abc-20260820-ade65d2/
```

checkpoint 根目录：

```text
/mnt/workspace/ctrlhgen-checkpoints/ss-csc-abc-20260820-ade65d2/
```

虽然运行根目录名含 A 的短 SHA，B/C 的 artifact metadata、command 和 checkpoint lineage 均明确记录
完整 `45af790...`，没有把不同代码版本混作同一阶段。

## 3. Experiment A：Full pair-data audit

正式入口：

```bash
python -m akgr.reproduction.ss_csc_data \
  --experiment-config akgr/configs/reproduce/wn-specific-relation-full-train-author-aligned.yml \
  --pilot-config akgr/configs/ss_csc/wn-specific-relation-pilot.json \
  --output-dir /mnt/workspace/ctrlhgen-runs/ss-csc-abc-20260820-ade65d2/experiment-a
```

只读取 base train 和 train graph；没有读取 validation/test/sealed final evaluation。运行约 366.6 秒。

| 项目 | 数量 |
|---|---:|
| raw / exact records | 104,000 / 104,000 |
| unique observations | 66,053 |
| distinct hypotheses | 99,723 |
| 去除的重复 hypotheses | 4,277 |
| 双向 exclusive-relation groups | 9,364 |
| exclusive candidate pairs | 22,380 |
| 双侧 branch-supported candidate pairs | 14,810 |
| 每 observation 最多 2 对后的 frozen pairs | 9,902 |
| pair train / pair validation | 8,897 / 1,005 |
| directional train examples | 17,794 |
| train/validation observation overlap | 0 |

所有 frozen sides 都通过 positive joint relation-value branch marginal。matched-selective side rate 为
`1.0`，但 replacement empty rate 为 `0.78286`，再次证明 matched replacement 不能作为主训练信号；它只保留
为诊断。

数据 gate：`8,897 >= 5,000`，通过。

## 4. Experiment B：Update-matched paired SFT 决策实验

三个分支从同一 unconditional parent 独立初始化 optimizer/scheduler，依次运行，均完成 10 epochs、每 epoch
112 optimizer updates；每 5 epochs 同时评价 pair-validation 和原 1,664 条 frozen validation。正式运行时
设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，关闭 flash/memory-efficient SDP，启用 math SDP 和
`torch.use_deterministic_algorithms(..., warn_only=False)`。

### 4.1 Epoch 轨迹

| branch | epoch | pair selection | symmetric margin | mean prompt margin | 原 val SemAvg | condition acc | parse | EOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single-target | 5 | **0.58408** | 0.01692 | 0.08600 | **0.70874** | 0.67428 | 0.99880 | 1.0 |
| single-target | 10 | 0.57065 | 0.05174 | 0.13968 | 0.70187 | 0.67969 | 0.99880 | 1.0 |
| paired-SFT | 5 | 0.79403 | 0.39801 | 0.42072 | 0.62566 | 0.74038 | 0.99219 | 1.0 |
| paired-SFT | 10 | **0.83532** | **0.54527** | **0.69839** | 0.60282 | 0.77644 | 0.99038 | 1.0 |
| SS-CSC | 5 | 0.81045 | 0.47164 | 0.51043 | 0.62869 | 0.75841 | 0.99459 | 1.0 |
| SS-CSC | 10 | **0.85721** | **0.62985** | **0.82001** | **0.61248** | 0.79748 | 0.99399 | 1.0 |

固定选择规则选中 single-target epoch 5、paired-SFT epoch 10、SS-CSC epoch 10。三者运行时长分别约
232.0、213.1 和 359.5 秒。

paired-SFT 已证明主要增益来自天然多解数据重组；contrastive margin 在同一数据之上仍有小而稳定的直接
控制增量。SS-CSC 相对 paired-SFT 还把原 validation SemAvg 提高约 `+0.00965`，但远不足以恢复相对
single-target 的 `-0.09627` 退化。

### 4.2 选中 checkpoints

| branch | checkpoint | tree SHA256 |
|---|---|---|
| single-target | `single_target/epoch-5` | `61f979aa25cc53eea1a7cea6ce39567e2cbfdc413a220821f6311fa641619cd8` |
| paired-SFT | `paired_sft/epoch-10` | `12f33095300aabf369da6632d734503fd28f592554e86e756e9435549bcd74ee` |
| SS-CSC | `ss_csc/epoch-10` | `c67706f58578b8340ed10259a93d5f7088a121d436b176fabdbe41cab6ac8d50` |

## 5. Experiment C：冻结 pair 的直接解选择评估

Experiment C 对同一 1,005 个 pair-validation pairs 评价三个选中 checkpoints，运行约 56.3 秒。每个 pair
同时输入两个真实 gold conditions；likelihood 指标与 Experiment B 的选择评价一致，另进行 greedy、AST、
denotation 和 branch-support 审计。

| 指标 | single-target | paired-SFT | SS-CSC |
|---|---:|---:|---:|
| assigned target mean logp | -0.30247 | **-0.27473** | -0.27973 |
| swapped target mean logp | -0.38847 | -0.97313 | **-1.09974** |
| pair selection accuracy | 0.58408 | 0.83532 | **0.85721** |
| bilateral selection rate | 0.17413 | 0.67960 | **0.72637** |
| symmetric margin satisfaction | 0.01692 | 0.54527 | **0.62985** |
| greedy assigned adherence | 0.59104 | 0.86368 | **0.89353** |
| paired-condition-only response | 0.50299 | 0.76020 | **0.78209** |
| exact prediction change | 0.25572 | 0.84975 | **0.89751** |
| parsed AST change | 0.25498 | 0.84960 | **0.89741** |
| denotation change | 0.01892 | 0.16833 | **0.19223** |
| pair greedy SemAvg | **0.98684** | 0.94315 | 0.92732 |
| branch-supported controlled value | 0.52786 | 0.77662 | **0.78706** |
| non-root branch-supported value | 0.40100 | 0.49701 | **0.51393** |
| parse / EOS | 0.99900 / 1.0 | 0.99950 / 1.0 | 0.99950 / 1.0 |

直接控制的改善并非只来自复制 condition 或 root/chain 自动通过：SS-CSC 的 non-root branch-supported rate
也最高。但 pair greedy semantic 从 single-target 的 `0.98684` 降到 SS-CSC 的 `0.92732`，与外部原
validation 的更大退化方向一致。

10,000 次 record-level paired bootstrap（seed 42/43）：

| SS-CSC - paired-SFT | delta | paired 95% CI |
|---|---:|---:|
| pair selection accuracy | +0.02189 | [+0.00995, +0.03433] |
| symmetric margin satisfaction | +0.08458 | [+0.06070, +0.10846] |

### 5.1 预注册 gate

| 检查 | 结果 |
|---|---|
| selection gain >= 0.03 | **失败**（0.02189） |
| selection CI lower > 0 | 通过 |
| symmetric-margin gain >= 0.03 | 通过 |
| symmetric-margin CI lower > 0 | 通过 |
| 原 validation SemAvg 不低于最佳对照 -0.01 | **失败**（-0.09627） |
| 原 validation parse/EOS 容忍线 | 通过 |
| greedy nominal 不退化 | 通过 |

总体 gate：**failed**。

## 6. 非权威运行与实现诊断

正式 B 前保留了两次未纳入结果的 single-target 运行：

1. `single_target-missing-cublas`：启动后发现未设置 CuBLAS deterministic workspace；一分钟内停止，未作
   正式比较。
2. `single_target-target-copy-bug`：likelihood helper 错把 tokenizer 第三个位置参数作为 `text_target`，
   实际输入变成 `prompt SEP prompt END`。epoch 5/10 原 validation parse 和 SemAvg 均为 0，触发停止。
   修复为已由 condition-causality evaluator 使用的 `tokenizer(prompt, target)` 合同，并新增序列级回归测试。

两次运行的日志和中间 checkpoints 均保存在运行根目录及 checkpoint 根目录下的 `non-authoritative/`，没有
覆盖或混入正式 artifacts。

## 7. 权威产物与 hashes

| artifact | SHA256 |
|---|---|
| `experiment-a/summary.json` | `8b03e588c779530f93b241b639d2c736470c2b45cc354b956be4d27667269185` |
| `experiment-a/pair-train.jsonl` | `3cc80d9f850c96da053e36347b9c0bf29c84554bfe0a16cf48a9d88a87204426` |
| `experiment-a/pair-validation.jsonl` | `c6e731c0ddac50b05caf181d122a76fd7160e04be5dd108229cc64f2053ffd31` |
| `experiment-a/single-target-train.jsonl` | `353fbf282a54ddb9eaed6b6f176ed3c59ea55b6c98ed69e7e2b4b6500b8b8de6` |
| `experiment-b/single_target/best.json` | `93f334e107b135b8f3f00c9ea42e80ac6ebe420aad0bba7c5f2e26cc0725f41a` |
| `experiment-b/paired_sft/best.json` | `dd536983a20fd2e7cce999e9c087fbeeaaa442fdcac814860e71c916817a2f11` |
| `experiment-b/ss_csc/best.json` | `f792df60eee4175e809332a255df048f6bb023af6630de2eb62e40405fdc493e` |
| `experiment-c/summary.json` | `b4ecd43314c2acbe496e1022436673b48a8c5eeaf3d76372be4fc7c4e33c7e92` |
| `experiment-c/single_target.jsonl` | `212304fc8ec51e2f1c90a327c5ea4e314ba190eef19df072b5b4b4bee25e7d8b` |
| `experiment-c/paired_sft.jsonl` | `1ba88cee33334d5bf757ed03de851c72d69ab36d3ce574ec81d301c85ddc7a29` |
| `experiment-c/ss_csc.jsonl` | `ffef0c27cd66b55a9771e898e1eb5d90b79a6e391f665c9ade3929d3aea47a66` |

## 8. 决策与下一步边界

- 不运行 optional Experiment D/short GRPO；当前失败发生在 SFT 的外部语义保持 gate，RL 会混入新变量。
- 不打开 sealed final evaluation；本轮所有选择与停止均只使用 train-derived pair validation 和既有模型
  validation。
- 数据重组机制成立，contrastive objective 也有可辨识增量，但当前 natural-pair 分布过窄，pair 内语义近乎
  饱和且向原 validation 的迁移明显退化。
- 若继续研究，应先处理 pair train 的 pattern/answer-cardinality reweighting 或将 paired supervision 与
  代表性 single-target data 混合，并预注册 semantic-retention 约束；不能直接把本轮 SS-CSC checkpoint
  接到 GRPO 后宣称方法有效。

## 9. DSW 收尾

实验结束时：DSW checkout 为 clean detached `45af7901c15d937fd6762d9ec8e7581b81097576`；L20 显存
1 MiB、utilization 0%；没有 SS-CSC/SFT/GRPO 实验进程。StopInstance 响应将在文档提交、部署和最终 hash
复核后补记。
