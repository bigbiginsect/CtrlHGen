# CtrlHGen 复现现状、Phase C-IV 结论与执行交接

> 状态日期：2026-08-11（Asia/Shanghai）
>
> 当前分支：`codex/reproduction-pipeline`
>
> 运行原则：本地修改、提交并推送到 `origin`；DSW 只部署并运行精确 commit SHA。

本文是后续 agent 的当前路线入口。前三次 small Phase C 和第四次 full-train 数据规模消融的
配置、曲线、指标、原始证据路径和 SHA256 单独保存在
`worklogs/phase-c-2026-08-10.md`。历史记录不能被改写成后续阶段的预设结论。

## 1. 当前状态

| 阶段 | 状态 | 当前结论 |
|---|---|---|
| Phase A：复现基础设施 | 已完成 | 数据、模型、checkpoint、评估、脚本和自动测试已闭环 |
| Phase B：tiny 与监督契约诊断 | 已完成 | tokenizer padding、conditional label mask 和 checkpoint 契约问题已修复 |
| Phase C-I | 已失败并归档 | 训练预算不足和 conditional 标签错误同时存在，checkpoint 禁止复用 |
| Phase C-II | 已完成但指标不理想 | 12 层 paper-aligned small SFT 可生成有效结构，但远低于论文绝对数值 |
| Phase C-III：author-aligned small | 已完成并审计 | 10+10 pilot 全门槛通过；正式 50+50 和固定 test 完整结束 |
| Phase C-IV：author-scale full train | 已完成并审计 | 8+8 pilot、正式 50+50 和一次 frozen test 全部通过；数据覆盖是当前主要瓶颈 |
| Phase D：GRPO | 建议进入但尚未执行 | 若获得新任务授权，只能从 C-IV healthy `conditional-best` 启动 |
| Phase E：验收报告 | 部分完成 | 已有四次 Phase C 证据，尚缺 Phase D 结果和最终报告 |

第一次失败 run 的任何 checkpoint 均不得使用：

```text
/mnt/workspace/ctrlhgen-runs/repro-wn-pattern-phase-c-seed42-20260809-111557
```

第二次 `repro-wn-pattern-small-v2` 必须保留，用作预注册基线和旧配置兼容性检查；不得
覆盖、续训或把它冒充第三次实验。

## 2. 复现目标和证据边界

主实验仍为 **WN18RR + pattern condition**，对应论文 Table 3 的 SFT-only（`w/o RL`）
与后续 CtrlHGen 对照。固定报告 Jaccard、Dice、Overlap、Pattern Accuracy、Smatch，另行
报告 parse、EOS、长度与解码方式。

Phase C-IV 已完成的目标是：

> 在单张 L20、固定 seed 42、模型和训练动力学完全不变的条件下，只把 base train 从
> 每 pattern 1,024 条扩大到作者 profile 的 8,000 条，验证训练覆盖不足是否是主要瓶颈。

单 seed 结果不能被写成论文绝对数值复现。论文 Table 3 仅作尺度参照：

| 设置 | Jaccard | Dice | Overlap | Pattern Accuracy | Smatch |
|---|---:|---:|---:|---:|---:|
| CtrlHG (w/o RL) | 71.5 | 75.8 | 83.7 | 81.5 | 79.0 |
| CtrlHGen | 77.0 | 80.8 | 86.8 | 93.5 | 83.3 |

不把论文与代码的不一致直接表述为学术不诚实。报告只记录可验证事实、采用的解释和仍未
公开的信息。

## 3. 两次历史 Phase C 的有效结论

### 3.1 Phase C-I

- 12×768×12 随机 GPT-2，unconditional 20 epoch、conditional 10 epoch、LR `1e-5`。
- unconditional 在结束时刚开始形成可解析输出。
- conditional 同时存在确定性的 label-mask 错误：target 和 `END` 被 mask，继续延长该
  checkpoint 的训练没有意义。
- 因此不能把结果简单归因为“只差更多 epoch”，也不能只归因为实现错误。

### 3.2 Phase C-II

- 12×768×12、AdamW、LR `1e-5`、batch 256、全程线性 warm-up/decay。
- unconditional 配置 400 epoch，实际在 epoch 311 后按另行记录的规则停止；conditional
  完成 50 epoch。
- 修复后的生成和监督链路健康，但固定 test 结果仍明显低于论文：

| decoding | Jaccard | Pattern Accuracy | Smatch | Parse | EOS |
|---|---:|---:|---:|---:|---:|
| greedy | 0.3299 | 0.3185 | 0.6028 | 0.9183 | 1.000 |
| sampled | 0.2419 | 0.3191 | 0.5987 | 0.9417 | 1.000 |

这些数值是 Phase C-III 的预注册比较对象，不是第三次实验的调参目标或 early-stop 输入。

## 4. 论文、作者仓库和 v3 的配置审计

审计基于作者仓库 `upstream/main` 的提交
`3f6580134549f24c2461773d4c2a737c37061448`、本地论文 PDF 和 Transformers 4.50.2
实测。

| 项目 | 论文 | 作者仓库表达的意图 | 作者公开代码实际语义 | Phase C-II | Phase C-III 选择 |
|---|---|---|---|---|---|
| 层数 | 未完整披露本地 `hug_model` | `GPT2_6` / `num_layers: 6` | `num_layers` 不覆盖 GPT-2 `n_layer`，标准模板仍为 12 层 | 12 | 明确 `n_layer: 6` |
| hidden / heads | 未完整披露 | 依赖缺失的 `./hug_model` | 注释指向标准 GPT-2 small | 768 / 12 | 768 / 12 |
| optimizer | AdamW | 配置未写 | 代码使用 Adam | AdamW | Adam |
| SFT LR | `1e-5` | `5e-5` | `5e-5` | `1e-5` | `5e-5` |
| batch | 256 | WN18RR 160 | 160 | 256 | 160 |
| epochs | 400 unconditional + 50 conditional | GPT2 配置 50 | conditional 脚本从 315 恢复，但总 epoch 为 50，会零步退出 | 311 + 50 | 50 + 50 |
| warm-up | 50/5 epoch | `warm_up: 5` | GPT2 分支按前 5 optimizer step 执行，之后恒定 | epoch warm-up 后衰减至 0 | 5-step，0.1×→1.0×，之后恒定 |

参考文件：

- `akgr/configs/config-model.yml`
- `akgr/configs/config-train.yml`
- `scripts/train/wn-g2.sh`
- `scripts/cond-train/wn-g2-pattern.sh`
- `akgr/abduction_model/transformer.py`
- `akgr/abduction_model/main.py`

Phase C-III 是“作者意图版”的自洽解释，不是对矛盾公开脚本的逐行复刻。尤其不复制
conditional 零步退出，也不把无效的 `num_layers` 字段当成真实模型层数。

## 5. 当前代码和配置契约

旧配置继续使用 `warmup_epochs`，其文件内容和 semantic hash 不变；small-v2 的固定 hash
为：

```text
0ff4e8455ccf88e882840304ef8e031ae8e0fcb84967e6365bb1a2cf228d16d2
```

新配置显式记录 optimizer 和 scheduler：

```yaml
optimizer:
  name: adam
  betas: [0.9, 0.999]
  eps: 1.0e-8
  weight_decay: 0.0
scheduler:
  name: linear_warmup_constant
  warmup_unit: optimizer_step
  warmup_value: 5
  start_factor: 0.1
```

新旧形式不能混用。训练 history 每个 epoch 必须记录：

- `optimizer_steps`；
- `learning_rate_start` / `learning_rate_end`；
- `optimizer_schedule` 摘要；
- loss、global step、validation、checkpoint 和 best 状态。

训练发现非有限 loss 或实际 optimizer step 数与预期不一致时必须立即失败。

### 5.1 正式配置

```text
akgr/configs/reproduce/wn-pattern-small-author-aligned.yml
experiment: repro-wn-pattern-small-author-aligned-v3
semantic hash: 0fde2887a211b4deaa7075edf97a1d1e9d3ac93466950bb03f9b5957e1a18796
```

关键参数：

| 项目 | 值 |
|---|---|
| 模型 | random GPT-2，6×768，12 heads，tied embeddings |
| batch | micro/effective 160，accumulation 1 |
| unconditional | 50 epoch，LR `5e-5`，merged train |
| conditional | 50 epoch，LR `5e-5`，base train |
| optimizer | Adam，betas 0.9/0.999，eps `1e-8`，weight decay 0 |
| scheduler | 5 optimizer-step linear warm-up，从 0.1× 到 1.0×，之后恒定 |
| validation/checkpoint | 每 5 epoch；parse≥0.90、EOS≥0.98 |
| generation | 与 v2 相同，greedy 与 sampled 分开报告 |

### 5.2 Pilot 配置

```text
akgr/configs/diagnostics/wn-pattern-small-author-pilot.yml
experiment: diagnostic-wn-pattern-small-author-pilot-v1
semantic hash: b417a2fb7fd1284e7d1e9a96f27d88b7d2a056fe4a0550c499a0bc1b6219af40
```

Pilot 使用相同模型和训练动力学，但两阶段各 10 epoch，每 2 epoch 验证和保存，自动健康
门槛为 parse≥0.10、EOS≥0.90。Pilot 只用于排除明显错误，checkpoint 永远不得进入正式 run。

## 6. 固定 small 数据

第三次实验不重新采样，继续使用：

```text
/mnt/workspace/ctrlhgen-data/WN18RR/small/seed-42/0f7d3531958a/sampling-manifest.json
```

| 项目 | 值 |
|---|---:|
| data hash | `0f7d3531958ac6b8804988a2e80f79d4ffc29ac1a88d3a1aeedae300586fe5e7` |
| KG hash | `ab2a80eabd45a86cc1dad1acfc65c20b36a451070737b054e903bb2025dda4a9` |
| base train | 13,312 |
| valid | 1,664 |
| test | 1,664 |
| augmentation train | 12,945 |
| merged unconditional train | 26,257 |

batch 160、drop-last false 时，每 epoch 的预期 optimizer step 数为：

- unconditional merged：`ceil(26257 / 160) = 165`；
- conditional base：`ceil(13312 / 160) = 84`。

任一 manifest hash、schema 或数量不一致都必须停止。不要覆盖已有数据后继续。

## 7. Phase C-III Pilot（历史流程）

### 7.1 部署和环境

```bash
# local
cd /mnt/d/yang_nankai/CtrlHGen
git status --short --branch
git push origin HEAD
git rev-parse HEAD

# DSW
ssh ctrlhgen-dsw
cd /mnt/workspace/CtrlHGen
git status --short --branch
git fetch origin --prune
git switch --detach <exact-sha>
git rev-parse HEAD

source /mnt/workspace/envs/ctrlhgen/bin/activate
export CTRLHGEN_DATA_ROOT=/mnt/workspace/ctrlhgen-data
export CTRLHGEN_CHECKPOINT_ROOT=/mnt/workspace/ctrlhgen-checkpoints
export CTRLHGEN_RUN_ROOT=/mnt/workspace/ctrlhgen-runs
export HF_HOME=/mnt/workspace/cache/huggingface
export TRITON_CACHE_DIR=/mnt/workspace/cache/triton
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PILOT=akgr/configs/diagnostics/wn-pattern-small-author-pilot.yml
```

正式启动前运行完整非 GPU 测试和相关 6-layer GPU synthetic test。Pilot 不允许访问 test。

### 7.2 Pilot 命令

```bash
bash scripts/reproduce/sft-unconditional.sh "$PILOT"

PILOT_UNCOND=/mnt/workspace/ctrlhgen-checkpoints/diagnostic-wn-pattern-small-author-pilot-v1/unconditional-best
bash scripts/reproduce/sft-conditional.sh "$PILOT" "$PILOT_UNCOND"
```

### 7.3 Pilot 通过条件

以下条件必须全部满足：

1. 无 OOM、NaN、标签契约错误或 checkpoint 恢复错误；
2. history 每 epoch 分别记录 165 和 84 optimizer steps；
3. 首个 epoch 的 LR 从 `5e-6` 开始，在第 5 optimizer step 后达到 `5e-5`，之后保持恒定；
4. 两阶段 epoch 10 train loss 均比各自 epoch 1 至少下降 25%；
5. selected validation 达到 parse≥0.10、EOS≥0.90；
6. conditional selected validation 的 Pattern Accuracy≥0.10；
7. 新进程能加载 selected pilot checkpoint，并通过监督/tokenizer 只读诊断。

Pilot 失败时保留 config snapshot、history、validation、checkpoint 和日志并停止。不得自动
延长 epoch、调 test、降低 gate 或把 pilot checkpoint 送入正式实验。

## 8. Phase C-III 正式运行（历史流程）

只有 Pilot 全部通过后才允许开始。正式实验必须从新的随机初始化 unconditional 模型开始：

```bash
CONFIG=akgr/configs/reproduce/wn-pattern-small-author-aligned.yml

bash scripts/reproduce/sft-unconditional.sh "$CONFIG"

UNCOND=/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-small-author-aligned-v3/unconditional-best
bash scripts/reproduce/sft-conditional.sh "$CONFIG" "$UNCOND"
```

只有 parse≥0.90、EOS≥0.98 且 `health_pass=true` 的 selected best 可以进入下一阶段。不得用
final checkpoint、旧 v2 checkpoint 或手工挑选的低-loss checkpoint 绕过 gate。

### 8.1 固定 test

```bash
COND=/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-small-author-aligned-v3/conditional-best

python -m akgr.abduction_model.main \
  --experiment-config "$CONFIG" \
  --mode testing \
  --checkpoint "$COND" \
  --test_split test \
  --overwrite_batchsize 64 \
  --greedy

bash scripts/reproduce/evaluate.sh "$CONFIG" "$COND"
```

greedy 和 sampled 必须分开命名、保留和报告。训练与 best 选择期间不得读取 test。

### 8.2 完成与比较口径

宣布 Phase C-III 执行完成需要：

1. 两阶段均完成且产生健康 selected best；
2. history 的 step/LR 与配置一致；
3. checkpoint 能在新进程严格加载；
4. 固定 1,664 条 test 的 greedy/sampled 均有逐样本 JSONL 和 aggregate CSV；
5. parse、EOS、max-length rate 和五项论文指标齐全；
6. 精确 SHA、命令、起止时间、GPU、显存、manifest/hash 和产物路径已记录。

对 v2 greedy 的预注册比较为：

| 指标 | v2 基线 |
|---|---:|
| Jaccard | 0.3299 |
| Pattern Accuracy | 0.3185 |
| Smatch | 0.6028 |

报告所有差值。单 seed 的小幅提升只能作描述，不能宣称统计显著；没有提升也必须如实记录，
不能在查看 test 后回头选择另一组超参数。

## 9. Phase D 与最终报告

Phase C-III 的历史审计结果为：

- v3 greedy 相对 v2 的 Pattern Accuracy 提高 `0.2698`，Smatch 提高 `0.0585`；
- Jaccard 下降 `0.0609`，Dice 和 Overlap 也下降，不能表述为全面改善；
- parse 从 `0.9183` 变为 `0.9243`，EOS 保持 `1.0`，max-length rate 保持 `0`；
- 当时 Phase D 是否值得执行仍需根据“condition adherence 改善但集合重合下降”的取舍决定。

Phase C-IV 已把这一决策更新为：**值得从 C-IV conditional-best 进入 GRPO，但本任务没有
执行 GRPO。** Phase C-IV 在只扩大训练数据的条件下同时显著提高集合重合和控制能力，
形成比 v3 更合适的 SFT parent。若后续明确授权 Phase D，必须冻结使用
`repro-wn-pattern-full-train-author-aligned-c4/conditional-best`，并与 C-IV SFT-only 配对；
禁止使用 v2、v3 或 pilot checkpoint 作为正式 parent。

最终导师报告至少包括配置来源差异、精确 SHA/环境、数据 hash/count、运行时间和显存、
greedy/sampled 五项指标、v2/v3/C-IV/论文对照、失败项与单 seed 局限。

## 10. 产物纪律与接手顺序

所有大产物保存在：

```text
/mnt/workspace/ctrlhgen-data/
/mnt/workspace/ctrlhgen-checkpoints/
/mnt/workspace/ctrlhgen-runs/
/mnt/workspace/cache/
```

后续 agent 的固定顺序：

1. 阅读本文和 `worklogs/phase-c-2026-08-10.md`；
2. 用 `worklogs/phase-c-2026-08-10.md` 第 8 节的路径/hash 核验 C-IV 原始证据；
   DSW 收尾后应为 `Stopped`，不要为只读文档工作重启实例；
3. 若明确授权 Phase D，先冻结 GRPO 的目标、成功口径和一次 test 契约；
4. 重启 DSW 后核对 checkout、manifest 和 C-IV `conditional-best`，并以它作为唯一正式
   parent；
5. Phase D 完成后与 C-IV SFT-only 配对比较，再形成最终报告。

当前边界是：**四次 Phase C 均已归档；C-IV author-scale full train 已通过 pilot、正式
50+50、一次 frozen test 和完整审计。扩大数据显著改善语义集合指标并保持、强化控制
能力，支持“覆盖不足是当前主要瓶颈”；但仍有论文语义差距和单 seed 局限。Phase D 尚未
执行，论文绝对数值复现结论仍不成立。**

## 11. Phase C-IV 决策摘要

Phase C-IV 保持 v3 的 random GPT-2 6×768、12 heads、Adam、LR `5e-5`、batch 160、
5 optimizer-step warm-up 后恒定、50+50 epoch、WN18RR/pattern/seed 42 和 checkpoint
selection，仅把 base train 从每 pattern 1,024 条扩大为 8,000 条。train-only augmentation
使 unconditional merged count 为 204,610；conditional 固定 104,000 条。跨 split 的完整
监督 tuple 重合为 0，valid/test 与 v3 对应文件逐字节相同。

实体 target 的 train vocabulary 覆盖从 `14,361/40,559 = 35.41%` 提高到
`33,200/40,559 = 81.86%`；valid/test unseen entity occurrence rate 分别从
`28.25%/27.70%` 降至 `3.03%/3.34%`。relation target 始终为 `22/22`，valid/test unseen
均为 0。因此本轮扩大规模确实改变了实体覆盖，而不是 validation/test 或关系词表。

正式 frozen test 为：

| decoding | Jaccard | Dice | Overlap | Pattern Accuracy | Smatch | Parse | EOS | 五项均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 0.5969 | 0.6447 | 0.7149 | 0.9441 | 0.8196 | 0.9952 | 1.0000 | 0.7440 |
| sampled | 0.5517 | 0.5960 | 0.6620 | 0.9459 | 0.8153 | 0.9946 | 1.0000 | 0.7142 |

greedy 相对 v3 的 Jaccard/Dice/Overlap/Pattern Accuracy/Smatch 分别提高
`+0.3279/+0.3454/+0.3656/+0.3558/+0.1583`；相对论文 w/o RL，前三项仍低
`-0.1181/-0.1133/-0.1221`，但 Pattern Accuracy 和 Smatch 已高
`+0.1291/+0.0296`。这构成“覆盖不足是当前主要瓶颈”的强单变量证据，但不表示它是唯一
瓶颈；`pin`、`inp`、union sampled 和长结构仍是后续优化重点。
