# CtrlHGen 复现现状、路线图与执行交接

> 状态日期：2026-08-10（Asia/Shanghai）
>
> 当前分支：`codex/reproduction-pipeline`
>
> 运行原则：以 `git rev-parse HEAD` 的精确 SHA 为准；本地修改、提交并推送，DSW 只部署该 SHA。

这份文档是后续 agent 的路线入口。Phase C 的两次实际运行、配置、指标轨迹和原始证据
索引已单独记录在 `worklogs/phase-c-2026-08-10.md`；本文第 5 节保留的是运行前计划和
门槛，阅读时应结合该结果日志，不能再把它当成“尚未启动”的当前状态。

## 1. 一页交接结论

| 阶段 | 状态 | 结论 |
|---|---|---|
| Phase A：复现基础设施 | 已完成 | 数据、模型、checkpoint、评估、脚本和自动测试已经闭环 |
| Phase B：tiny 与根因诊断 | 已完成 | 全链路已跑通；旧 Phase C 的 conditional 失败根因已定位并修复 |
| Phase C：small SFT-only | 已完成一次修复后运行 | 无条件于 311 epoch cost-aware early stop，conditional 完成 50 epoch；结果与证据见独立日志 |
| Phase D：small GRPO | 未执行 | 健康 `conditional-best` 已产生；是否继续应先明确 small 相对趋势或论文绝对数值的目标 |
| Phase E：验收报告 | 部分完成 | Phase C 实验记录已形成；尚无 Phase D、多 seed 或最终导师版报告 |

当前保留第一次失败 run、修复诊断和第二次完整 small-v2 SFT-only run。后续 agent 应先读
`worklogs/phase-c-2026-08-10.md`、核对其中 hash，并根据待验证假设决定做分层分析、扩大
数据、重跑 SFT 或进入 GRPO；不应机械地按第 5 节重新启动同一实验。

严禁使用以下失败实验及其中任何 checkpoint：

```text
/mnt/workspace/ctrlhgen-runs/repro-wn-pattern-phase-c-seed42-20260809-111557
```

它不是一个“多训一些 epoch 就能恢复”的模型。旧 conditional 训练实际把 target 和 `END` mask 掉了，继续训练只会进一步拟合错误标签。

## 2. 复现目标与论文对照

导师要求是：跑通代码，并复现论文中至少一个关键实验或指标；如因算力缩小规模，必须说明与论文设置的差异。

本项目的主实验固定为 **WN18RR + pattern condition**，对应论文 Table 3 的 SFT-only（`w/o RL`）与完整 CtrlHGen 对照。主指标是 Pattern Accuracy，同时报告 Jaccard、Dice、Overlap、Smatch。

建议采用的表述是：

> 在单张 L20 和明确缩小的 query 样本规模下，复现 CtrlHGen 的完整训练/评估流程与 SFT→GRPO 的关键趋势。

除非以后补齐论文规模和多随机种子，不能声称复现了论文绝对数值。

论文 Table 3 的比较基准为：

| 设置 | Jaccard | Dice | Overlap | Pattern Accuracy | Smatch |
|---|---:|---:|---:|---:|---:|
| CtrlHG (w/o RL) | 71.5 | 75.8 | 83.7 | 81.5 | 79.0 |
| CtrlHGen | 77.0 | 80.8 | 86.8 | 93.5 | 83.3 |

缩放实验不以落入论文误差范围为硬性成功条件。合理的成功口径是：流程和指标定义一致、实验可审计可重复，并判断 GRPO 相对同一 small SFT checkpoint 是否改善条件遵循率；若没有改善也应如实报告。

## 3. 旧 Phase C 失败的真实根因

最初只看到无条件模型验证 parse 约 11%，conditional 与 test 为 0% EOS、0% parse，因此“随机初始化模型训练预算不足”是一个合理表象，但它不是 conditional 永不终止的主因。

在完整 12 层模型、真实 WN18RR 数据和 checkpoint round-trip 上，最终定位到：

1. validation generation 临时把 fast tokenizer 改为 left padding。
2. Python 层虽恢复为 right padding，Rust tokenizer backend 仍保留 left-padding 状态，随后被 `save_pretrained()` 写入 checkpoint 的 tokenizer JSON。
3. conditional 从该 checkpoint 加载 left-padding tokenizer。训练 pair 与用于构造 label mask 的 prompt 长度不同，mask 错位后覆盖 target 和 `END`，反而保留 source/condition 为 active labels。
4. 旧 conditional checkpoint 的训练 batch 中 `END` active label 数量为 0。即使训到 100 epoch、loss 降到 `0.006`，模型也不可能学会输出 EOS。

因此旧 run 同时存在两个问题：无条件阶段确实训练不足；conditional 阶段则是确定性的监督标签错误。后者才是 0% EOS 的根因。

已完成的修复包括：

- 条件格式固定为 `answers COND condition SEP target END`，`COND` 是独立 token；
- training batch 强制 right padding；generation-only left padding 被限制在作用域内，并显式清理 backend padding 状态；
- active labels 必须严格等于 `target + END`；
- checkpoint format v2 哈希 tokenizer JSON，并验证 padding、特殊 token 和 pair 模板；旧 format-v1 checkpoint 会被拒绝；
- validation 记录 parse、EOS、生成长度和 max-length rate；未通过健康门槛的 checkpoint 不能被选为 best，也不能进入下一阶段；
- best 选择优先 generation-level 指标，validation loss 不能单独选中不可用模型；
- 无条件 SFT 使用 merged augmentation；conditional SFT 和 GRPO 使用 pattern 平衡的 base train；
- `semantic_hash`、`data_hash`、`kg_hash` 分离，调整训练预算不会触发相同数据的重采样。

真实数据的修复后诊断证据位于：

```text
/mnt/workspace/ctrlhgen-runs/diagnostic-wn-pattern-overfit-v7/
/mnt/workspace/ctrlhgen-checkpoints/diagnostic-wn-pattern-overfit-v7/
```

诊断结果：

- conditional 从 epoch 5 开始 EOS rate 为 100%；
- 对 104 条训练样本，teacher-forced token/sequence/EOS accuracy、prompt-only first-token accuracy 与 prompt agreement 均为 1.0；
- 同一训练集 greedy parse/EOS 均为 1.0；
- tiny valid 的无条件 best greedy parse 为 0.731、EOS 为 1.0；conditional best greedy parse 为 0.462、EOS 为 1.0。

这些结果证明监督、保存/加载和生成链路已经恢复；它们是诊断结果，不是正式论文复现指标。

## 4. 当前可复现基线

### 4.1 正式 small 配置

唯一正式配置入口：

```text
akgr/configs/reproduce/wn-pattern-small.yml
```

关键设置：

| 项目 | small 实际设置 | 与论文关系 |
|---|---|---|
| 数据集/condition/seed | WN18RR / pattern / 42 | 主实验一致；论文 seed 未披露 |
| 逻辑模式/最大 observations | 13 类 / 32 | 一致 |
| 样本数（每 pattern） | train/valid/test = 1024/128/128 | 缩小；论文精确数量未披露 |
| 子逻辑增强 | `up, 3in, pni, pin, inp`，仅 train | 复杂类型一致；仅 train 是防泄漏默认决策 |
| 模型 | random GPT-2，12×768，12 heads，tied embeddings | 层数一致；标准 GPT-2 small 宽度是实现假设 |
| SFT effective batch | 256 | 一致；L20 直接 micro-batch 256、accumulation 1 |
| 无条件 SFT | 400 epoch，warmup 50，LR `1e-5` | 一致 |
| conditional SFT | 50 epoch，warmup 5，LR `1e-5` | 一致 |
| validation/checkpoint | 每 10 epoch greedy；每 25 epoch checkpoint | 本复现的可靠性补充 |
| generation | `top_k=0, top_p=1, temperature=1`，max 33 | 按当前官方代码语义固定并记录 |
| 硬件 | 1× NVIDIA L20 | 论文为 4×A6000 48GB |

模型词表包括 40,559 个实体与 22 个方向关系 token。small 只缩小每类 query 数，没有再缩 SFT epoch 或 effective batch。

L20 最复杂 conditional batch 实测形状约 `256 × 52`，峰值显存 19.75 GiB，每个 forward/backward/step 约 0.62 秒。完整 Phase C 预计约 6–8 小时，实际时间必须从日志记录。

### 4.2 固定数据

已生成且无需重采样的 schema-v2 manifest：

```text
/mnt/workspace/ctrlhgen-data/WN18RR/small/seed-42/0f7d3531958a/sampling-manifest.json
```

数量：

| 数据 | 条数 |
|---|---:|
| base train | 13,312 |
| valid | 1,664 |
| test | 1,664 |
| augmentation train | 12,945 |
| merged unconditional train | 26,257 |

manifest 中应核对：

- `data_hash` 前缀：`0f7d3531958a`
- `kg_hash` 前缀：`ab2a80eabd45`
- schema version：2

若 manifest、数量或 hash 不符，停止运行并调查；不要覆盖现有数据后继续。

### 4.3 已完成的基础设施和验证

早期 P0/P1 requirement gap 已落实为代码和测试，主要包括：

- 可安装环境与 DSW 路径约定；
- 显式 GPT-2 config，不再依赖缺失的 `./hug_model`；
- seeded KG split、采样 schema/profile、数据 manifest 和 hash；
- 五类 sub-logic augmentation；
- condition 编码、评分和五项 evaluation；
- 两阶段 SFT、健康 best 指针、format-v2 checkpoint 与新进程恢复；
- GRPO group/reward/config、checkpoint 和 resume；
- `scripts/reproduce/` 的采样、SFT、评估和 GRPO 入口；
- tiny CPU/GPU、真实数据 overfit、tokenizer round-trip 与 stage-transition 自动测试。

功能代码基线曾在 DSW 完成全套测试：`90 passed`。正式运行前仍应在部署后的精确 SHA 上重跑测试或至少运行相关测试，不能只引用这一历史结果。

## 5. Phase C：small SFT-only 正式实验

### 5.1 进入条件

启动前必须同时满足：

1. DSW checkout 干净并处于本次提交的精确 SHA；
2. 第 4.2 节 manifest 可读且 hash/count 一致；
3. 没有把失败 run 的 checkpoint 作为 parent/resume；
4. CUDA、磁盘和外部持久目录正常；
5. 命令、SHA、环境、GPU、日志路径和进程状态有记录。

Phase C 必须从随机初始化的 format-v2 **无条件训练**开始。不要只重跑 conditional。

### 5.2 DSW 环境

```bash
ssh ctrlhgen-dsw
cd /mnt/workspace/CtrlHGen
git status --short --branch
git rev-parse HEAD

source /mnt/workspace/envs/ctrlhgen/bin/activate
export CTRLHGEN_DATA_ROOT=/mnt/workspace/ctrlhgen-data
export CTRLHGEN_CHECKPOINT_ROOT=/mnt/workspace/ctrlhgen-checkpoints
export CTRLHGEN_RUN_ROOT=/mnt/workspace/ctrlhgen-runs
export HF_HOME=/mnt/workspace/cache/huggingface
export TRITON_CACHE_DIR=/mnt/workspace/cache/triton
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CONFIG=akgr/configs/reproduce/wn-pattern-small.yml
```

长任务可以使用 DSW 上可靠的会话管理方式，但必须把 stdout/stderr 和 PID/会话名写入独立 run 记录；不得让两次正式运行写入同一个实验目录。

### 5.3 无条件 SFT

```bash
bash scripts/reproduce/sft-unconditional.sh "$CONFIG"
```

预期 best 指针：

```text
/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-small-v2/unconditional-best
/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-small-v2/unconditional-best.json
```

健康门槛：validation `parse_ok >= 0.90` 且 `eos_rate >= 0.98`。只有被 `unconditional-best.json` 选中并标记 `health_pass=true` 的 checkpoint 才能进入 conditional。

如果训练结束仍没有 best 指针，或所有候选均未过门槛：**停止 Phase C**。保留日志和候选 checkpoint，分析 validation 曲线、EOS/长度/解析错误；不得用 final checkpoint 绕过门槛。

### 5.4 conditional SFT

```bash
UNCOND=/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-small-v2/unconditional-best
bash scripts/reproduce/sft-conditional.sh "$CONFIG" "$UNCOND"
```

预期 best 指针：

```text
/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-small-v2/conditional-best
/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-small-v2/conditional-best.json
```

同样要求 `parse_ok >= 0.90`、`eos_rate >= 0.98` 和 `health_pass=true`。代码还会校验传入 parent 就是被选中的健康 `unconditional-best`，不能手工挑一个 loss 较低但生成无效的 checkpoint。

可在异常时做只读监督诊断：

```bash
python -m akgr.abduction_model.sft_diagnostics \
  --experiment-config "$CONFIG" \
  --checkpoint /path/to/checkpoint \
  --split train
```

诊断不能替代正式 validation/test，也不要在看过 test 后用它调参。

### 5.5 固定测试集评估

先做 greedy preflight，确认结构稳定：

```bash
COND=/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-small-v2/conditional-best
python -m akgr.abduction_model.main \
  --experiment-config "$CONFIG" \
  --mode testing \
  --checkpoint "$COND" \
  --test_split test \
  --overwrite_batchsize 64 \
  --greedy
```

再按配置进行论文式随机采样评估：

```bash
bash scripts/reproduce/evaluate.sh "$CONFIG" "$COND"
```

greedy 和 sampled 结果必须分开命名和报告。Phase C 的正式 SFT-only 表格取固定 test split 的五项指标，同时保存逐样本 JSONL、aggregate CSV、generation 配置和 seed。

### 5.6 Phase C 完成定义

只有以下条件全部满足，才能宣布 Phase C 完成并解锁 Phase D：

- 无条件与 conditional 都存在 selected healthy best；
- 新进程能加载 conditional best；
- greedy test 的 parse/EOS 未出现系统性退化；
- sampled test 完成并保存五项指标及逐样本结果；
- 没有使用 test 结果挑 checkpoint、改 seed 或调超参数；
- SHA、manifest、配置快照、日志、checkpoint 和结果路径可追溯。

## 6. Phase D：GRPO 正式实验

Phase D 不是已完成工作。它当前被 Phase C 的健康 `conditional-best` 明确阻塞；在该门槛满足前，不能启动正式 GRPO。

### 6.1 目标与固定设置

Phase D 从 Phase C 的**同一个 selected conditional best**继续，使用同一个 base train 和固定 test split，与 SFT-only 做配对比较。

`wn-pattern-small.yml` 当前 GRPO 参数为：

| 参数 | 值 |
|---|---:|
| data variant | base |
| group size / `num_generations` | 4 |
| per-device train batch | 32 |
| epochs | 1 |
| learning rate | `1e-5` |
| beta | 0.1 |
| epsilon | 0.2 |
| max completion length | 33 |
| save interval / keep | 100 steps / 2 |
| reward weights | Jaccard 1.0、Dice 0.5、Overlap 0.5、condition 1.0 |
| external reporting | disabled |

这里应记录为 **1 epoch**，不是旧路线图中的“1–3 epoch”。任何 epoch、reward 或 KL 参数调整都必须形成新配置/实验名，不能覆盖首轮结果。

### 6.2 GRPO 前置诊断

正式长任务前先在 held-out train examples 上检查：

- 四个 completion 是否产生非零的组内总 reward 方差；
- semantic 三项和 condition reward 是否都能被计算；
- parse/EOS/长度是否正常；
- prompt/condition 与 Phase C 的 tokenizer contract 一致。

若需要 `--max-steps 10` 的 GPU smoke，必须复制成独立 diagnostic 配置并使用独立 `experiment.name`/输出目录。不要在正式 `repro-wn-pattern-small-v2/grpo` 目录先跑 10 steps 再从头运行，否则 checkpoint 和 trainer state 会污染正式实验。

若 reward 对所有 group 几乎恒定为 0、completion 基本不可解析或 EOS 崩溃，停止并诊断；不要靠延长 epoch 掩盖无学习信号。

### 6.3 正式命令

```bash
COND=/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-small-v2/conditional-best
bash scripts/reproduce/grpo.sh "$CONFIG" "$COND"
```

入口会强制校验 parent 正是 `conditional-best.json` 选中的健康 checkpoint。预期评估 checkpoint：

```text
/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-small-v2/grpo/evaluation-step-<global-step>
```

训练中必须记录 total/component reward、组内 reward 方差、KL、completion 长度、parse/EOS、loss、global step、耗时和显存。如果实现日志尚未覆盖某项，先补足观测性再做长跑。

### 6.4 GRPO 评估与停止条件

对最终 GRPO evaluation checkpoint 使用与 Phase C 完全相同的 fixed test、greedy preflight 和 sampled evaluation。比较表必须配对展示：

- SFT-only conditional best；
- GRPO evaluation checkpoint；
- 论文 `w/o RL` 与 CtrlHGen 数值（仅作尺度参考）。

Pattern Accuracy 是主要趋势指标，Jaccard、Dice、Overlap、Smatch 为次要指标。还应检查 parse/EOS 和长度，避免奖励提升来自格式或长度异常。

遇到以下情况应停止并保留证据，而不是直接调 test：

- group 内 reward 长期无方差；
- parse 或 EOS 相对 SFT 显著崩溃；
- KL 或 completion 长度出现病态增长/塌缩；
- 语义奖励与 condition reward 实现值不一致；
- checkpoint 无法在新进程恢复并复现评估。

首轮 seed 42 完成后再决定是否扩展 seeds 43、44。每个 seed 必须使用独立配置、manifest、实验名和产物目录；不能看过 test 后只报告最好的 seed。

## 7. Phase E：导师验收报告

Phase E 也不是已完成工作。它应在 Phase C 完成后先形成 SFT-only 版本，在 Phase D 完成后补齐最终对照。

最终报告至少包含：

1. 论文设置与实际设置对照：数据集、13 类 pattern、observation 上限、样本量、模型、epoch、batch、generation、GRPO、硬件；
2. 精确 Git SHA、环境版本、CUDA/GPU 和随机种子；
3. 数据来源、split 协议、manifest/hash、各 split 与 augmentation 数量；
4. 可重复命令、运行起止时间、墙钟耗时、显存和产物目录；
5. SFT-only 与 GRPO 的 Jaccard、Dice、Overlap、Pattern Accuracy、Smatch；
6. greedy 与 sampled 结果分开，不能混为一个表；
7. 论文 Table 3 与本次 small 结果的并排对照；
8. GRPO 是否复现“提高 condition adherence”的趋势，以及对语义指标的影响；
9. 单 seed 或多 seed 的统计口径、均值/标准差或置信区间；
10. 失败、未复现项、实现假设和算力缩放局限；
11. checkpoint、日志、逐样本预测、aggregate CSV 和配置快照的路径。

验收层级：

- **代码跑通**：tiny 数据→两阶段 SFT→checkpoint 恢复→test→tiny GRPO→恢复评估全部成功；当前已达到。
- **最低复现（原工程定义）**：修复后的 Phase C 已完成，并在固定 WN18RR test 上报告
  SFT-only 五项指标；但该定义只证明缩小规模流程可审计完成，不等于复现论文绝对数值。
- **推荐复现**：完成 Phase C+D 的配对比较，最好再做 3 seeds；当前尚未达到。

报告结论必须使用“缩小规模流程/趋势复现”措辞。单张 L20、small 数据、单 seed 的结果不能包装成论文绝对数值复现。

## 8. 仍然开放的假设与风险

以下信息论文或作者仓库没有充分披露，后续 agent 不应假装已经确定：

1. 每种 pattern 的论文训练 query 精确数量和原始 seed；当前 small profile 是显式缩放设置。
2. 除 12 层外的原始 `hug_model/config.json` 缺失；当前采用标准 GPT-2 small 宽度并随机初始化。
3. 论文 Table 中 `±` 的精确统计口径；多 seed 时应明确 run-level std，单次 sampled 结果不能冒充多 run 方差。
4. 论文测试 generation 的全部参数；当前同时保留 deterministic greedy 诊断和已固定参数的全分布随机采样。
5. 当前 PyKEEN mapping 会过滤 valid/test 中未出现在 train mapping 的实体相关 triples；这与论文“unseen entities”的文字可能存在差异，必要时做敏感性分析。
6. CUDA memory-efficient attention 不保证 bitwise deterministic；固定 seed 仍需记录库版本和硬件。
7. 仅 train 做 sub-logic augmentation 是防止评估污染的默认决策，论文没有充分说明其他 split 的处理。

若联系作者，优先询问原始 model config、query 数/seed、Table 3 checkpoint、测试 generation config 和误差项定义。新材料必须保存来源与 hash，不能无痕覆盖现有实验。

## 9. DSW 部署与产物纪律

遵循根目录 `AGENTS.md`：本地修改，commit/push 到 `origin`，DSW fetch 后 detached 到精确 SHA；绝不 push `upstream`，也不在本地和 DSW 维护两套手工修改。

部署前：

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
```

如果 DSW checkout 是 dirty，先检查来源，不得直接丢弃。数据、checkpoint、cache 和大日志必须保存在：

```text
/mnt/workspace/ctrlhgen-data/
/mnt/workspace/ctrlhgen-checkpoints/
/mnt/workspace/ctrlhgen-runs/
/mnt/workspace/cache/
```

每个正式 run 至少保留：精确 SHA/status、完整命令、config snapshot、environment/GPU、manifest/hash、stdout/stderr、起止时间、进程状态、metrics、predictions 和 checkpoint 路径。不得把 generated data、checkpoint、日志或凭据提交到 Git。

## 10. 后续 agent 的执行顺序

以下 1--8 是 Phase C 运行前的原计划，现作为历史执行协议保留。Phase C 已实际完成，
后续接手顺序改为：

1. 阅读 `worklogs/phase-c-2026-08-10.md`，核验两次 run 的文件 hash 和配置差异。
2. 先提出待检验假设，再决定做逐 pattern/覆盖率分析、full-scale SFT 或 small GRPO。
3. 若启动新实验，继续遵守本文的 SHA、manifest、健康 best、独立 test 和产物纪律。

原计划如下：

1. 核对本地、origin、DSW 的精确 SHA 和 clean status。
2. 核对 small manifest 的 schema、hash 和数量，不重新采样。
3. 在部署 SHA 上运行测试/最小 preflight，确认 CUDA 和持久目录。
4. 建立独立正式 run 记录，启动 Phase C 无条件 400 epoch。
5. 只在 selected healthy `unconditional-best` 产生后启动 conditional 50 epoch。
6. 只在 selected healthy `conditional-best` 产生后做固定 test 的 greedy 与 sampled 评估。
7. 汇报 Phase C 指标和健康证据；满足第 5.6 节后再按第 6 节进行 GRPO。
8. 完成 Phase D 配对评估，再按第 7 节形成导师验收报告。

当前边界是：**Phase C 的修复后 small SFT-only run 已完成，但绝对指标没有接近论文
参照；Phase D、多 seed、full-scale 验证和最终导师报告均未完成。现有证据支持多种后续
假设，不应把数据规模或任何单一因素预先写成已证实根因。**
