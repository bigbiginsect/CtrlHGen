# CtrlHGen 复现现状、Phase C-III 路线与执行交接

> 状态日期：2026-08-10（Asia/Shanghai）
>
> 当前分支：`codex/reproduction-pipeline`
>
> 运行原则：本地修改、提交并推送到 `origin`；DSW 只部署并运行精确 commit SHA。

本文是后续 agent 的当前路线入口。2026-08-09 和 2026-08-10 两次 Phase C 的配置、
曲线、指标、原始证据路径和 SHA256 单独保存在
`worklogs/phase-c-2026-08-10.md`。该记录是历史证据，不应被改写成第三次实验的预设结论。

## 1. 当前状态

| 阶段 | 状态 | 当前结论 |
|---|---|---|
| Phase A：复现基础设施 | 已完成 | 数据、模型、checkpoint、评估、脚本和自动测试已闭环 |
| Phase B：tiny 与监督契约诊断 | 已完成 | tokenizer padding、conditional label mask 和 checkpoint 契约问题已修复 |
| Phase C-I | 已失败并归档 | 训练预算不足和 conditional 标签错误同时存在，checkpoint 禁止复用 |
| Phase C-II | 已完成但指标不理想 | 12 层 paper-aligned small SFT 可生成有效结构，但远低于论文绝对数值 |
| Phase C-III：author-aligned small | 代码与配置已实现，待 pilot/正式运行 | 当前唯一推荐 Phase C 路线 |
| Phase D：GRPO | 阻塞 | 必须等待 Phase C-III 完成和审计，禁止从旧 v2 checkpoint 启动正式 GRPO |
| Phase E：验收报告 | 部分完成 | 已有两次 Phase C 证据，尚缺 v3、GRPO 和最终报告 |

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

本项目当前目标是：

> 在单张 L20、固定 small 数据和 seed 42 下，检验更接近作者公开仓库意图的模型容量与
> 训练动力学，完成可审计的两阶段 SFT；之后才决定是否进入 GRPO。

small 结果不能被写成论文绝对数值复现。论文 Table 3 仅作尺度参照：

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

## 7. Phase C-III Pilot

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

## 8. Phase C-III 正式运行

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

Phase D 当前明确阻塞。Phase C-III 完成后先审计：

- author-aligned 配置是否改善 v2 的 Pattern Accuracy 与集合重合；
- 改善是否伴随 parse/EOS 或长度退化；
- 是否值得从同一个 v3 `conditional-best` 进入 GRPO。

如果进入 Phase D，GRPO 必须使用 v3 同一 selected checkpoint、base train 和 fixed test，
并与 v3 SFT-only 配对。禁止用 v2 checkpoint 作为正式对照的 parent。

最终导师报告至少包括配置来源差异、精确 SHA/环境、数据 hash/count、运行时间和显存、
greedy/sampled 五项指标、v2/v3/论文对照、失败项与单 seed 局限。

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
2. 核对本地、origin、DSW 精确 SHA 与 clean status；
3. 运行测试并核对固定 manifest；
4. 运行隔离 Pilot，逐项审计第 7.3 节；
5. Pilot 通过后从随机初始化启动正式 50+50 epoch；
6. 只用 selected healthy checkpoint 做 fixed test；
7. 更新 Phase C 实验记录，再决定是否解锁 Phase D。

当前边界是：**两次历史 Phase C 已归档；第三次 author-aligned small 的代码和配置已准备
就绪，但仍必须先通过 Pilot。任何 Phase D 或论文绝对数值结论都尚未获得授权或证据。**
