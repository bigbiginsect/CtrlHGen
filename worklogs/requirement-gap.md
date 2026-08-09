# CtrlHGen 复现 Requirement Gap 与后续执行交接

> 状态日期：2026-08-10（Asia/Shanghai）
>
> 当前维护分支：`codex/reproduction-pipeline`（运行前以 `git rev-parse HEAD` 记录精确 SHA）
> 文档目的：说明“当前代码距离跑通和满足导师复现要求还缺什么”，并让后续 agent 不必重新调查即可继续实施。

## 1. 任务目标与推荐结论

导师要求是：

> 复现：跑通代码，并复现论文中至少一个关键实验/指标；如因算力需要缩小规模可以，但必须说明与论文设置的差异。

本项目建议采用“缩小规模关键实验复现”，而不是承诺整篇论文的数值级复现：

1. 先修复并跑通一条完整链路：知识图谱准备 → query sampling → 无条件 SFT → 条件 SFT → checkpoint 恢复 → 测试评估。
2. 以 **WN18RR + pattern condition** 为主实验，复现论文 Table 3 中 `w/o RL` 与完整 `CtrlHGen` 的 GRPO 消融。
3. 以 **Pattern Accuracy** 为主验收指标，同时报告 Jaccard、Dice、Overlap、Smatch。
4. 单张 L20 上缩减每类 query 数和训练 epoch；保留相同数据集、13 种逻辑类型、最大 observation size 32、12 层 decoder-only 模型、奖励定义和评测方法。
5. 如果预算只够完成一个阶段，最低可交付是 Table 3 的 `CtrlHG (w/o RL)` 行对应的缩小规模 SFT-only 指标；更稳妥的交付是再加入 GRPO，并复现“GRPO 提升条件遵循率”的趋势。

这里的“复现”应表述为：**在明确列出的缩放设置下复现实验流程、指标计算和关键趋势**。除非以后使用完整论文设置并做多随机种子实验，否则不要声称复现了论文的绝对数值。

### 1.1 2026-08-10 Phase C 失败的根因与处置

失败 run：`/mnt/workspace/ctrlhgen-runs/repro-wn-pattern-phase-c-seed42-20260809-111557`。
最初观察到无条件 best 的验证 parse 约 11%，conditional 与 test 则是 0% EOS、0% parse。训练预算不足确实解释了无条件模型较弱，但**不是 conditional 永不停止的根因**。

已用完整 12 层模型、真实 WN18RR 数据和 checkpoint round-trip 复现并定位：

1. validation generation 临时把 fast tokenizer 切为 left padding。Python 属性虽然恢复为 right，Rust backend 仍保留最后一次 left-padding 状态；紧接着 `save_pretrained()` 把它写入了 tokenizer JSON。
2. conditional 从无条件 checkpoint 加载后得到 left-padding tokenizer。训练 pair 和用于 label mask 的 prompt 都左对齐，但长度不同，mask 因而覆盖 target 与 `END`，反而把 source/condition 留成 active labels。
3. 直接检查旧 conditional checkpoint 的 16 条训练 batch 得到 `eos labels = 0`；即使训练 100 epoch、loss 降到 `0.006`，greedy 仍必然是 0% EOS。这证明继续加 epoch 无法修复该 run。
4. 修复后，每条 active label 严格等于 `target + END`。相同的 104 条真实数据诊断中，conditional 从 epoch 5 起 EOS rate 即为 100%；最终 checkpoint 在训练集上 teacher-forced token/sequence/EOS accuracy、prompt-only first-token accuracy、greedy parse 与 greedy EOS 全为 100%。

配套修正包括：

- 条件分隔符改为独立 `COND`，固定序列为 `answers COND condition SEP target END`；
- training batch 强制 right padding，generation-only left padding 不得污染 backend；
- checkpoint format v2 哈希 tokenizer JSON 并校验 padding、SEP、EOS pair 模板，旧 checkpoint 会被拒绝；
- validation 记录 EOS rate、生成长度和 max-length rate，未过 parse/EOS 门槛的 checkpoint 不能成为 best，也不能进入下一阶段或 GRPO；
- best 选择优先 generation-level parse/condition 指标，不再让低 validation loss 单独选中不可用模型；
- 无条件阶段使用 merged augmentation，conditional/GRPO 使用按 pattern 平衡的 base train；
- `semantic_hash`、`data_hash`、`kg_hash` 分离，改训练预算不再重建完全相同的数据；
- small 配置恢复论文的 400+50 epoch SFT 日程，只缩放每 pattern 样本数，避免同时缩数据和缩优化步数。

因此，旧 Phase C checkpoint **不可续训或用于 Phase D**。必须从 format-v2 无条件模型重新开始；正式 small run 先以 greedy validation 的 `parse_ok >= 0.90`、`eos_rate >= 0.98` 为阶段门槛。

## 2. 已核实的当前状态

### 2.1 Git 与 DSW

- 本地 checkout：`/mnt/d/yang_nankai/CtrlHGen`
- DSW checkout：`/mnt/workspace/CtrlHGen`
- 本地 `main` 与 DSW detached HEAD 都在 `0225585dc897455b2c216da26f9ace1adc588415`。
- 调查时两边工作区均干净；本地源代码相对 `upstream/main` 没有功能修改，只有仓库卫生、工作说明、论文和占位目录等私有工作仓库改动。
- DSW：Python 3.11.11、PyTorch 2.6.0+cu124、Transformers 4.50.2、TRL 0.16.0、Datasets 3.5.0。
- DSW：1 张 NVIDIA L20，约 46 GB 显存；`/mnt/workspace` 约有 195 GB 可用空间。
- `python -m akgr.abduction_model.main --help` 和 sampling 模块的 `--help` 可以在 DSW 正常导入并显示参数。这只证明 import/参数解析可用，不代表训练链路已跑通。

### 2.2 当前没有复现实验产物

本地和 DSW 的以下仓库目录都只有 `.gitkeep`：

- `sampled_data/`
- `checkpoints/`
- `results/`

DSW 规划的持久化目录 `/mnt/workspace/ctrlhgen-data` 和 `/mnt/workspace/ctrlhgen-checkpoints` 尚不存在。当前没有：

- 已固定的数据划分；
- sampled query JSONL；
- 模型 checkpoint；
- 正式训练日志；
- 逐样本预测；
- 论文指标 CSV。

因此截至本文档日期，项目还不能称为“代码已跑通”。

### 2.3 已做而未继续做的调查动作

- 已阅读当前 README、所有主要配置、sampling/SFT/GRPO/evaluation 入口以及主要 shell scripts。
- 已对照论文实验章节和附录（当前 arXiv v3）：<https://arxiv.org/html/2505.20948>。
- 已做只读或轻量检查，没有下载知识图谱、没有正式采样、没有启动训练、没有占用长时间 GPU。
- 已实测 `pip install --dry-run --no-index -r requirements.txt` 会因 Conda 格式在 `_libgcc_mutex=0.1=main` 处失败。
- 已在 DSW 直接调用 `create_transformer(...)`，确认因缺少 `./hug_model/config.json` 报 `OSError`。

## 3. 论文目标设置与建议缩放设置

论文明确说明：

- 数据集：DBpedia50、WN18RR、FB15k-237；KG 按 8:1:1 构建递增的 train/valid/test 图。
- 13 种预定义逻辑类型；每个 observation 不超过 32 个实体。
- 子逻辑分解应用于 `up, 3in, pni, pin, inp` 五种复杂模式。
- 指标：Jaccard、Dice、Overlap、condition Accuracy、Smatch。
- 模型：12 层 decoder-only Transformer，优化器 AdamW。
- 硬件：4 张 NVIDIA A6000 48GB。
- SFT：batch size 256；无条件阶段 400 epoch、50 epoch warm-up；条件阶段 50 epoch、5 epoch warm-up；learning rate `1e-5`。
- RL：batch size 32；每组 4 个候选；语义奖励权重 `lambda=(1.0, 0.5, 0.5)`；`alpha=0.5` 平衡语义和条件奖励。

建议首轮正式复现配置如下。样本数和 epoch 是算力缩放，不是论文原设置，报告中必须单列：

| 项目 | 论文设置 | 建议首轮设置 | 处理原则 |
|---|---|---|---|
| 数据集 | WN18RR（目标实验） | WN18RR | 保持一致 |
| 逻辑模式 | 13 类 | 13 类 | 保持一致 |
| 子逻辑分解 | 5 类复杂模式 | 相同 5 类 | 保持一致 |
| 最大 observation | 32 | 32 | 保持一致 |
| 每类样本数 | 论文未披露；代码 `full` 也不能可靠代表论文 | train/valid/test = 1024/128/128 | 明确为缩放设置 |
| 模型 | 12 层 decoder-only | 优先保留 12 层 | 保持层数；显存不足再讨论 |
| optimizer | AdamW | AdamW | 保持一致 |
| effective batch | SFT 256，RL 32 | 尽量保持；用 micro-batch + gradient accumulation | 保持有效 batch |
| SFT epoch | 400 + 50 | 400 + 50；只缩样本规模，不再同时缩优化日程 | 保持论文日程 |
| GRPO group size | 4 | 4 | 保持一致 |
| GRPO epoch | 论文正文未完整披露 | 初始 1–3 | 明确记录 |
| GPU | 4×A6000 48GB | 1×L20 46GB | 明确为硬件差异 |
| 随机种子 | 论文未披露 | 至少 1 个固定 seed；最好 42/43/44 三个 | 提升可审计性 |

建议对照的论文数值：

| Table 3 设置 | Jaccard | Dice | Overlap | Accuracy | Smatch |
|---|---:|---:|---:|---:|---:|
| CtrlHG (w/o RL) | 71.5 | 75.8 | 83.7 | 81.5 | 79.0 |
| CtrlHGen | 77.0 | 80.8 | 86.8 | 93.5 | 83.3 |

缩小规模实验不以进入论文误差范围为硬性成功条件。更合理的成功条件是：流程和指标定义一致，实验可重复，并且 GRPO 相对相同缩放设置的 SFT-only 在 Pattern Accuracy 上产生可解释的改善。若趋势未出现，也必须如实报告，不能调参到只保留有利结果。

## 4. Requirement Gap：当前代码距离“跑通”还缺什么

下面 P0 是正式实验开始前必须解决的阻断项；P1 是导师验收前必须完善的复现性问题；P2 是完整性或工程质量问题。

### P0-1：环境安装说明不可执行

**证据**

- README 要求 `pip install -r requirements.txt`。
- `requirements.txt` 是 `conda list --export` 风格，包含 `_libgcc_mutex=0.1=main` 和 `torch=2.6.0+cu118=pypi_0`，不能作为 pip requirements 使用。

**需要实现**

- 提供一种真正可执行且验证过的环境定义：建议增加精简的 `environment.yml` 或 `requirements-pip.txt`，不要机械复制完整 Conda 环境。
- 至少锁定 Python、PyTorch/CUDA、Transformers、TRL、Datasets、PyKEEN、smatch、NLTK、Accelerate、W&B 等直接依赖。
- README 中的安装命令必须在一个干净环境中验证。
- 记录 DSW 实际环境与论文/README Python 3.9 的差异。当前 DSW Python 3.11 可导入代码，但这不是环境可重建证据。

**验收**

- 新环境执行安装命令成功。
- `python -m akgr.abduction_model.main --help`、sampling help 和最小单元测试成功。

### P0-2：模型配置模板 `./hug_model` 缺失

**证据**

- `akgr/abduction_model/transformer.py` 用 `GPT2Config.from_pretrained('./hug_model', ...)`。
- 仓库和 Git 历史中没有 `hug_model/config.json`。
- DSW 实测模型初始化报 `OSError: Can't load the configuration of './hug_model'`。
- `config-model.yml` 写的是 `num_layers: 6`，但 Hugging Face GPT2Config 的有效层数字段是 `n_layer`；实测传入 `num_layers=6` 时 `n_layer` 仍为默认 12。因此当前文件既不能证明实际是 6 层，也不能完整定义论文的 12 层模型。

**需要实现**

- 不再依赖未提交的本地目录，改为在代码或 YAML 中显式构建 GPT2Config。
- 论文只披露了 12 层，未披露所有宽度参数。建议先采用标准 GPT-2 small 默认结构（12 层、hidden size 768、12 heads）并显式记录；如从作者处获得原 `config.json`，再替换并保留来源/hash。
- 明确是随机初始化结构配置，还是加载预训练权重。当前代码只读取 config 后新建 `GPT2LMHeadModel(config)`，看起来是随机初始化，不能悄悄改成预训练 GPT-2。

**验收**

- 模型可离线初始化。
- 日志打印的 `n_layer/n_embd/n_head/vocab_size` 与复现配置一致。
- 保存一份模型配置 JSON 到每个 run 目录。

### P0-3：README 的数据流程与 WN18RR 训练不闭合

**证据**

- `scripts/sample/sample_full.sh` 使用 `-s full`。
- `config-sampling.yml` 的 `full.datasets` 只有 `DBpedia50`。
- README 下一步却运行 `scripts/train/wn-g2.sh`，它需要 `WN18RR-full-32-*.jsonl`。
- `scripts/sample/sample_server_wn.sh` 使用 profile `wn18rr only`，但该 profile 没有 sampling 代码必需的 `datasets` 和 `scale` 字段，会在 `config_sampling[args.scale]['scale']` 处失败。
- FB15k-237 也没有一条闭合的 full sampling 路径。

**需要实现**

- 把 profile 定义统一成明确 schema，例如：`datasets`、每 split 每 pattern 的目标样本数、seed、输出 scale 名称。
- 不要同时让 `scale` 表示配置名、除数和输出文件名。
- 让 WN18RR small/debug/full profile 都能生成训练代码实际寻找的文件名。
- README 为每个支持的数据集给出一条从零开始的命令。

**验收**

- WN18RR tiny profile 生成 train/valid/test JSONL、`stats.txt` 和 KG cache。
- 生成数量与配置逐类一致，文件名与 dataloader 查找路径一致。

### P0-4：sampling 配置中的显式样本数目前未被使用

**证据**

- `config-sampling.yml` 中存在诸如 `train: 16/8000` 的数值。
- `sample_parallel.py` 实际使用 `num_train_edges // scaling_factor` 计算每类数量，没有读取这些 split 数值。

**需要实现**

- sampling profile 直接读取每 split 每 pattern 的目标数量。
- 如需要保留 edge-ratio 模式，必须使用另一个明确字段并在日志中打印最终数量。
- 给采样循环增加最大尝试次数和失败报告，避免某些 pattern 无限循环。

**验收**

- tiny profile 的每类输出数量可由测试精确断言。

### P0-5：知识图谱划分和采样不确定

**证据**

- `load_kg_util.py` 对全量 triples 调用 Pandas `sample()` 重新做 8:1:1 划分，没有 `random_state`。
- Python `random`、DataLoader shuffle、模型训练、generation 也没有统一 seed。
- 第一次构建的 pickle 会固化一次随机划分，但没有 manifest 说明是哪一次。

**需要实现**

- 增加统一 `--seed`，设置 Python、NumPy、PyTorch、CUDA、Pandas sampling、Datasets/Trainer/DataLoader generator。
- 持久化 split manifest：源数据版本、entity/relation/triple 数、split seed、各 split hash。
- 同一数据 cache 不得在不同 seed 之间静默复用；路径或 metadata 必须包含 seed/config hash。
- 记录确定性设置及其性能影响。

**验收**

- 同 seed 两次 tiny sampling 的 split hash 和 JSONL 内容 hash 一致。
- 不同 seed 不会误读旧 cache。

### P0-6：checkpoint 恢复路径使用了字面量花括号

**证据**

- `load_model_by_mode()` 的普通和 tuning `resume_path` 字符串都缺少 `f` 前缀。
- 所有条件训练、测试和多数 RL scripts 都依赖 `-r/--resume_epoch`。

**需要实现**

- 修复路径格式化。
- 在真正加载前打印并校验绝对路径；不存在时给出可操作错误，列出可用 checkpoint，而不是等 `torch.load` 抛异常。
- 统一 unconditional、conditional、RL checkpoint 命名，避免 `resume_epoch` 同时代表不同阶段。

**验收**

- tiny SFT 保存 checkpoint 后，新进程能恢复并完成测试。
- 条件 SFT 能从无条件 checkpoint 初始化。

### P0-7：两阶段 SFT 的 epoch 语义互相冲突

**证据**

- `GPT2_6_act_nt` 在 `config-train.yml` 中只有 `nepoch: 50`。
- README 推荐的无条件脚本因此训练 50 epoch，而论文是 400。
- 条件脚本从 90、315、380 等 epoch 恢复；即使修好路径，`range(last_epoch+1, nepoch+1)` 在 `nepoch=50` 时为空，不会执行条件训练。
- DB 无条件训练脚本本身还带 `-r=380`，但仓库没有此前 checkpoint。

**需要实现**

- 将无条件 SFT 与条件 SFT 配置分开，例如 `unconditional_epochs` 和 `conditional_epochs`。
- 条件阶段的 epoch 应是本阶段 1..N，不应继承无条件 checkpoint 的全局 epoch 作为循环起点。
- checkpoint metadata 同时保存 `stage`、`stage_epoch`、`parent_checkpoint`、`global_step`。
- optimizer 按论文改为 AdamW；是否继承第一阶段 optimizer/scheduler 必须明确。建议条件阶段重新创建 optimizer/scheduler。

**验收**

- tiny run 明确执行 N 个无条件 epoch 和 M 个条件 epoch，日志/文件名没有歧义。

### P0-8：论文核心的子逻辑分解没有进入训练数据流水线

**证据**

- 分解逻辑位于独立的 `akgr/sampling/sample_add.py`，README 没有调用。
- 脚本只保存 `new_src_1.pt/new_tgt_1.pt`，主 dataloader 只读取 `*-a2q.jsonl`，两者没有合并。
- `device_name=1` 导致只处理固定的第 500000–999999 段。
- `allow_pattern_dict=[5,9,10,11]` 与论文的 `[up=5, 3in=13, pni=9, pin=10, inp=4]` 不一致：遗漏 `3in` 和 `inp`，加入了 `2u`。
- `add_id()` 使用硬编码相对文件且没有在 main 中调用。

**需要实现**

- 把分解作为正式、确定性的 dataset transform，而不是设备编号切片脚本。
- 只对论文指定的五类复杂模式执行；为每种分解写单元测试。
- 生成完整 record：`answers/query/pattern_str`，与原训练 JSONL 合并或生成独立 augmented JSONL，并记录原样本 ID/父 pattern。
- 防止 train/valid/test 泄漏：论文描述的 augmentation 应用于训练数据时，sub-observation 必须在对应可见图上计算；首轮建议只增强 train，并在报告中说明。
- 输出增强前后各 pattern 数量和 hash。

**验收**

- tiny 数据中五类目标模式都产生合法 sub-logic 样本。
- dataloader 实际读到增强后的样本；日志显示基础/增强数量。

### P0-9：条件编码至少有一个明确错误

**证据**

- train、valid、test 的 `entitynumber` 分支调用了 `new_extract_sample_to_device_pattern()`，而不是 entity-number extractor。
- 这会让所谓 entity-number control 实际输入 pattern condition。

**需要实现**

- 改为正确的 entity-number extractor。
- 为五类条件分别增加 tokenizer/extractor 测试：pattern、relation-number、entity-number、specific-entity、specific-relation。
- 测试条件 token、target 和 condition-adherence scorer 使用相同定义。

**验收**

- 每种条件给定手工样本时，编码 token 和 accuracy 判定符合论文定义。

### P0-10：evaluation 的 unconditional 分支会引用未定义变量

**证据**

- unconditional extractor 不返回 `condition`。
- `test_loop()` 随后无条件把 `condition_batch=condition` 传入评分函数。

**需要实现**

- unconditional 使用不需要 condition 的评分路径，或显式传 `None` 并由 scorer 正确处理。
- 增加每个 condition 的单 batch evaluation smoke test。
- 保存逐样本 observation、condition、reference、prediction、五项分数、parse failure，而不只是 aggregate CSV。

**验收**

- unconditional 和五种 conditional 测试均至少跑完一个 batch。

### P0-11：GRPO 设置与论文/脚本不一致

**证据**

- 论文每组采样 4 个候选；当前 `GRPOConfig` 没有显式设置 `num_generations`，会依赖 TRL 版本默认值。
- 论文整体奖励在忽略全局正比例缩放后可写作 `[Jaccard, Dice, Overlap, condition] = [1.0, 0.5, 0.5, 1.0]`。
- README 推荐的单卡 WN script 使用 `[1.0, 1.0, 0.5, 0.0]`，把 condition reward 关闭了；这不对应完整 CtrlHGen。
- multi-GPU WN script 的 `[1.0, 0.5, 0.5, 1.0]` 才与论文权重比例一致。
- GRPO 默认 `report_to='wandb'`，没有说明认证、offline 或禁用方案。
- `rl_minibatch/rl_horizon/rl_smatch_factor` 等部分 CLI 参数在 GRPO 路径未使用，scripts 容易造成“参数看似生效但实际无效”。

**需要实现**

- 显式设置 `num_generations=4`。
- 用有名字的奖励配置代替 `eval('[...]')`；禁止不安全 `eval`。
- 在日志中打印各奖励分量的 batch 均值和最终组合。
- 完整实验使用论文比例；消融实验分别显式关闭 RL、Dice/Overlap、condition reward。
- W&B 默认应允许 `--report_to none` 或 offline；正式 run 再按用户意愿启用。
- 清理或验证所有 RL 参数，未使用参数应报错而不是静默忽略。

**验收**

- tiny GRPO 跑完至少若干 step，group size 确认是 4，保存 checkpoint，并可在新进程加载评测。

### P0-12：现有 shell scripts 绑定作者机器，不能直接用于单卡 DSW

**证据**

- scripts 硬编码 `CUDA_VISIBLE_DEVICES=1/2/3/4/6/7` 和 2/4 卡 accelerate。
- DSW 只有物理 GPU 0；例如 `CUDA_VISIBLE_DEVICES=1` 会隐藏唯一 GPU，可能退回 CPU。
- 部分旧 `optim-test` scripts 使用当前 argparse 已不存在的 `--ppo_*` 参数，属于旧 PPO 实验残留。
- 个别 T5 script 含作者本机路径 `/home/data/ywangmy/checkpoint/`。

**需要实现**

- 新增独立、可配置的 `scripts/reproduce/`，不要把旧脚本当作可信实验入口。
- 默认不在脚本内指定物理 GPU 编号；由调用环境设置，单卡内部统一使用逻辑 `cuda:0`。
- 对 argparse 和 shell scripts 做静态一致性检查。
- 明确区分 legacy PPO 与当前论文 GRPO，不把旧脚本纳入复现报告。

**验收**

- DSW 单卡命令日志显示 `cuda:0` 和 L20，不发生 CPU fallback。

## 5. P1：跑通后仍需补齐的复现性要求

### P1-1：生成/解码协议不明确

- 当前测试使用 `do_sample=True, top_p=1.0, top_k=0`，但没有 generation seed，论文也未清楚披露解码协议。
- 首轮建议保留当前 stochastic generation 语义但固定 seed；三 seed 实验分别报告。
- 记录 `do_sample/top_k/top_p/temperature/max_length/num_return_sequences`。
- 不要把参考 hypothesis 当作唯一正确答案；语义指标在 `G_test` 上计算，Smatch 只是参考结构指标。

### P1-2：论文误差项含义无法从代码恢复

- 当前 `stat_scores_by_pattern()` 的 `std` 是样本级分数标准差。
- 论文表格写“平均值 ± 标准差”，但没有明确是样本级、run 级还是标准误；其数值形式也不能从当前仓库确认。
- 我们应优先报告三个 seed 的 run-level mean ± std，同时另外保存样本级 std，避免混淆。

### P1-3：训练中没有可靠 validation/最佳 checkpoint 选择

- `fit()` 中 validation 和 score 保存被注释，只按固定频率保存。
- 缩小实验应恢复 validation loss/指标，预先定义模型选择规则，例如按 validation Pattern Accuracy，Jaccard 作为 tie-breaker。
- 不能根据 test 指标选 checkpoint。

### P1-4：产物写入位置和 metadata 不规范

- 训练当前会在仓库根目录写 `dataloader.pt` 和 `graph_samplers.pt`。
- 应将数据、cache、checkpoint、日志放到 AGENTS.md 规定的持久目录，Git checkout 只保留代码和小型报告。
- 每个 run 必须记录：Git SHA、dirty status、完整命令、配置快照、环境版本、GPU、seed、数据 hash、父 checkpoint、进程状态、开始/结束时间、输出路径。

### P1-5：缺少自动测试

至少应补：

1. pattern/action/query 往返转换；
2. 五类 condition encoder 与 adherence scorer；
3. 五类 sub-logic decomposition；
4. seeded split/sampling determinism；
5. checkpoint save/resume；
6. 一个 CPU 或极小 GPU batch 的 SFT/evaluation；
7. 一个 tiny GRPO smoke（可标记为 GPU integration test）。

## 6. 实施顺序（后续 agent 直接按此推进）

### Phase A：建立可复现基线，不跑长任务

1. 阅读根目录 `AGENTS.md`，确认 local → origin → exact SHA → DSW 流程。
2. `git status --short --branch`，确认没有覆盖用户修改。
3. 建议创建 `codex/reproduction-pipeline` 分支。
4. 增加精简且可安装的环境定义；在 DSW 现有环境先验证直接依赖版本。
5. 修复显式 GPT2 config、seed 基础设施、checkpoint 路径和阶段 epoch 语义。
6. 修复 sampler profile、entity-number condition、unconditional evaluation。
7. 将 sub-logic decomposition 正式接入 dataset pipeline。
8. 修复 GRPO group/reward/reporting，并新增 `scripts/reproduce/`。
9. 增加上述核心单元测试；此阶段不得启动 full sampling 或长训练。

### Phase B：tiny 端到端 smoke（不计作论文实验）

建议 profile：每 pattern train/valid/test = `8/2/2` 或 `16/2/2`，seed 42。

必须完成：

1. 下载并缓存 WN18RR/WordNet；
2. 固定 8:1:1 KG split；
3. 生成 13 类 query；
4. 生成五类 sub-logic augmentation；
5. 无条件 SFT 1 epoch；
6. pattern 条件 SFT 1 epoch；
7. 新进程恢复 checkpoint；
8. 完整 test loop 输出五项指标和逐样本结果；
9. tiny GRPO 至少若干 step；
10. 恢复 GRPO checkpoint 再测试。

只有以上全部成功，才能说“代码链路已跑通”。

### Phase C：缩小规模 SFT-only 正式实验

建议首轮：

- WN18RR；
- 每 pattern 1024/128/128；
- seed 42；
- 12 层模型；
- effective batch 256；
- 无条件 20 epoch + pattern 条件 10 epoch（先依据 smoke 吞吐量确认）；
- 保留 best-validation 与 final checkpoint；
- 在固定 test split 上报告五项指标。

这是 Table 3 `w/o RL` 的缩小版复现，也是最低可接受正式交付。

### Phase D：GRPO 正式实验

- 从 Phase C 选定的 conditional checkpoint 开始；
- group size 4；
- batch 32；
- reward 比例 `[1.0, 0.5, 0.5, 1.0]`；
- 1–3 epoch 起步；
- 同一固定 test split 评测；
- 与 SFT-only 配对比较，主要看 Pattern Accuracy，次要看 Jaccard/Dice/Overlap/Smatch。

如单 seed 趋势合理且预算允许，重复 seeds 43、44。不要在看过 test 后改 seed 或只报告最好 seed。

### Phase E：导师验收报告

最终报告至少包含：

- 论文设置、实际设置和差异原因对照表；
- 精确 Git SHA 与环境；
- 数据来源、split seed、样本数量和 hash；
- 一键命令、运行耗时、GPU 使用；
- SFT-only 与 GRPO 五项指标；
- 论文数值与本次数值对照；
- 趋势是否复现、未复现项及原因分析；
- checkpoint、log、逐样本预测和 aggregate CSV 路径。

## 7. 目标接口建议（尚未实现，不要直接运行）

后续 agent 可以把正式入口收敛到类似接口。以下是目标设计示意，不是当前可用命令：

```bash
python -m akgr.sampling.sample_parallel \
  --profile repro-wn-small \
  --seed 42 \
  --data_root /mnt/workspace/ctrlhgen-data

python -m akgr.abduction_model.main \
  --mode training \
  --stage unconditional \
  --experiment-config akgr/configs/reproduce/wn-pattern-small.yml

python -m akgr.abduction_model.main \
  --mode training \
  --stage conditional \
  --condition pattern \
  --parent-checkpoint /mnt/workspace/ctrlhgen-checkpoints/<run>/best.pth \
  --experiment-config akgr/configs/reproduce/wn-pattern-small.yml

python -m akgr.abduction_model.main \
  --mode optimizing \
  --condition pattern \
  --parent-checkpoint /mnt/workspace/ctrlhgen-checkpoints/<run>/best.pth \
  --experiment-config akgr/configs/reproduce/wn-pattern-small.yml

python -m akgr.abduction_model.main \
  --mode testing \
  --checkpoint /mnt/workspace/ctrlhgen-checkpoints/<run>/best.pth \
  --experiment-config akgr/configs/reproduce/wn-pattern-small.yml
```

一个实验应由单一 YAML 决定模型、数据、stage epoch、optimizer、batch、generation 和 reward；shell script 只负责调用，不再复制互相矛盾的参数。

## 8. 验收口径

### 8.1 “代码跑通”

同时满足以下条件才能判定：

- 从空数据目录开始完成 tiny 数据准备；
- 无条件和条件 SFT 都实际产生 optimizer step；
- checkpoint 能在新进程恢复；
- test loop 完成并生成五项指标；
- GRPO smoke 能训练、保存、恢复和测试；
- 命令、日志、配置、seed 和产物路径完整保存。

只有 import、`--help`、采样成功、单个 loss 或手写 scorer 输出都不算完整跑通。

### 8.2 “导师最低复现要求”

- 已满足“代码跑通”；并且
- 完成 Phase C，在 WN18RR pattern condition 上得到正式固定测试集指标；并且
- 用表格对照论文 Table 3 `w/o RL`，明确缩放差异。

### 8.3 “推荐的稳妥复现”

- 完成 Phase C + D；
- 在同一缩放设置下比较 SFT-only 和 GRPO；
- Pattern Accuracy 为主指标，五项指标全部报告；
- 最好三个 seed；
- 清楚区分“复现实验趋势”和“复现论文绝对数值”。

## 9. 当前未决信息与默认决策

这些信息官方仓库/论文目前没有充分给出，后续不要假装已经知道：

1. 缺失 `hug_model/config.json` 中除层数外的精确模型结构。默认建议显式采用标准 GPT-2 small 宽度、12 层、随机初始化，并在报告中列为实现假设。
2. 论文训练 query 的精确数量和数据随机种子。默认使用本文建议的小规模 profile 和固定 seed。
3. 论文表格 `±` 的统计口径。默认同时报告 run-level 三 seed std 与 sample-level std，明确标签。
4. 论文测试时的精确 generation 参数。默认以当前代码的 sampling 语义为起点，补齐固定 seed 和完整配置记录。
5. 论文是否对 train 以外 split 做 sub-logic augmentation。默认只增强 train，避免评测集污染。

如能联系作者，优先询问：原始 `hug_model/config.json`、三个数据集的 sampled query 数/seed、Table 1/3 checkpoint、测试 generation config、误差项定义。得到作者材料后必须记录来源和 hash，不能覆盖现有实验而不留版本。

## 10. DSW 运行与产物约定

遵循 `AGENTS.md`：本地改代码，commit/push 到 `origin`，DSW fetch 后 detached 到精确 SHA；不要在本地和 DSW 维护两套手工修改，也不要 push `upstream`。

推荐持久路径：

```text
/mnt/workspace/ctrlhgen-data/
/mnt/workspace/ctrlhgen-checkpoints/
/mnt/workspace/ctrlhgen-runs/<run-id>/
/mnt/workspace/cache/huggingface/
/mnt/workspace/cache/triton/
```

建议 run ID：

```text
repro-wn-pattern-<stage>-seed<seed>-<YYYYMMDD-HHMM>
```

每个 run 目录至少保存：

```text
command.sh
git-sha.txt
git-status.txt
environment.txt
gpu.txt
config.yml
data-manifest.json
stdout.log
metrics.csv
predictions.jsonl
checkpoints/ 或 checkpoint 路径清单
status.json
```

长任务启动前必须先记录 exact SHA、命令、日志目录和预期输出；启动后记录 PID/进程状态。数据、checkpoint、cache 和大日志不得提交 Git。

## 11. 后续 agent 的第一步清单

后续 agent 接手后无需重新做论文/仓库总盘点，直接执行：

1. 阅读 `AGENTS.md` 和本文档。
2. 检查 `git status`，确认本文档之外是否有用户新改动。
3. 建立实现计划，先完成 Phase A；不要直接运行 README 的 full scripts。
4. 优先修复 `hug_model`、环境、sampler schema/seed、checkpoint/stage epoch 这四个最前置阻断项。
5. 每修一个 P0 都增加对应自动测试。
6. Phase A 完成并 commit/push/deploy exact SHA 后，才在 DSW 创建持久目录并做 Phase B tiny smoke。
7. 在 tiny 全链路成功前，不启动 1024/128/128 sampling 或正式 GPU 训练。
8. 正式实验前向用户汇报预计样本数、epoch、显存和时间，再启动长任务。

本文档是执行基线。若实现过程中发现新缺口，应在对应 P0/P1 条目下补充“发现、决策、验证证据”，保持后续交接连续。
