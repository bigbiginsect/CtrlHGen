# Pattern 版 Verifier-guided Best-of-N 实验方案

适用范围：WN18RR / `pattern` condition  
定位：复用已完成的 conditional-pattern SFT checkpoint，验证 inference-time verifier 是否能跨 condition kind
带来稳定 decoding 改善  
默认方法：冻结 generator，sample K=4，observable-graph semantic verification + canonical pattern verification  
不包含：新 SFT、GRPO、learned reranker、beam search、动态 K 或 checkpoint 搜索

## 1. 要回答的问题

当前 `specific_relation` 实验已经证明：冻结 conditional generator 后，Best-of-4 graph execution selection 能
显著提高 SemAvg、exact 和有效 relation control。Pattern 版只回答一个更窄的问题：

> 对已经训练好的 `p(H|O,C_pattern)`，同样的 inference-time candidate generation + execution verification
> 是否能在不牺牲 pattern adherence 的前提下提高 denotation semantic quality？

本实验不是重新训练 pattern 模型，也不声称新的 generator 参数或训练算法得到改善。若通过，只能说明该
inference selection 框架能从 `specific_relation` 扩展到第二种 condition kind。

## 2. 既有模型血缘与冻结 checkpoint

复现阶段的 full pattern run 同时包含 unconditional 和 conditional checkpoints。Pattern 版必须使用真正接收
pattern condition 的 conditional checkpoint，不能使用 unconditional checkpoint，也不能把 relation condition
输入 pattern 模型。

冻结 generator：

```text
/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-full-train-author-aligned-c4/
  phase-d-parent -> conditional-epoch-45
```

选择 epoch 45 的理由不是本轮 validation：它已经是复现阶段登记的 Phase D parent，在 epoch 45/50 固定
bake-off 中具有更好的 Jaccard、Dice、Overlap 和五项均值。原训练规则选择的
`conditional-best -> conditional-epoch-50` 保留为历史事实，但本轮不重新比较或选择 checkpoint。

实现必须 fail closed：

1. 解析并验证 `phase-d-parent.json`；
2. 要求 resolved checkpoint 精确为 `conditional-epoch-45`；
3. 要求 checkpoint metadata 为 `stage=conditional`、`condition=pattern`；
4. 校验 config semantic hash、sampling manifest hash 和 checkpoint tree SHA256；
5. 将 pointer、checkpoint tree 和相关输入 hashes 写入正式 summary。

正式实现前在 DSW 重新计算 tree SHA256，并把结果冻结到 command/artifact；不得从 P1 结果改选 epoch 50。

## 3. 数据、condition 和历史隔离边界

配置：

```text
akgr/configs/reproduce/wn-pattern-full-train-author-aligned.yml
```

数据固定为：

- validation：1,664 条，13 patterns × 128；artifact SHA256
  `950dca912f2d602d32894562b6b17f856e350c0158d125bb01d784d51adeed75`；
- historical test：1,664 条，13 patterns × 128；artifact SHA256
  `38bf3b26367be27cf03689a1dd2b168a505024fde14ea065df4ec084c01929a3`；
- sampling manifest SHA256：
  `244ef264ad538df30ad58d72127d7fe3c31f52480dc3ddac882978400c49e460`。

Prompt condition 不是 `2i`、`pin` 等缩写，而是由 reference target 确定性抽取的 canonical action pattern，例如：

```text
i p e p e
```

使用现有合同：

```python
condition_value_from_target("pattern", target)
build_generation_prompt(source, ConditionSpec("pattern", value), ...)
```

重要边界：旧 test 已在 Phase C frozen test 和 epoch-45/50 bake-off 中使用过，因此 Pattern-P2 不能称为新的
sealed/untouched evaluation。它只能称为：

> 在既有 historical test split 上，对已冻结 checkpoint 和已由 validation 冻结的 decoding rule 做一次新的
> paired decoding comparison。

本轮所有规则、gate、K 和 seed 必须在 Pattern-P1 validation 后冻结；Pattern-P2 不允许反向调参或重跑。

## 4. Candidate generation

冻结值与 `specific_relation` 版本对齐：

```text
seed = 314159
do_sample = true
temperature = 1.0
top_k = 0
top_p = 1.0
batch_size = 32
K default = 4
K upper-bound diagnostic = 8
```

生成约束：

1. model、tokenizer 和 vocabulary 全部冻结；
2. greedy 单独运行；
3. sample candidate stream 按 `(seed, dataset, candidate_index)` 派生；
4. K=1/2/4/8 必须共享严格嵌套的 candidate prefixes，不能各自重新随机采样；
5. 不混入 beam、constrained decoding、temperature search 或 adaptive K；
6. 记录每个 candidate 的 prediction、EOS、token count、mean log probability 和 canonical AST hash。

## 5. Pattern-aware verifier

对每个 candidate `H`：

1. 严格 parse action AST；
2. 在对应 observable graph 上执行，得到 `[H]_G`；
3. 以输入 observation `O` 计算 Jaccard、Dice、Overlap；
4. 定义：

```text
SemAvg(H) = (Jaccard + Dice + Overlap) / 3
exact(H)  = ([H]_G == O)
```

5. 从 candidate 本身抽取 canonical pattern：

```text
candidate_pattern(H) = number_to_pattern(H)
pattern_match(H,C) = parse_ok(H) and candidate_pattern(H) == C
```

`pattern_match` 必须要求 parse 成功，不能让 malformed token sequence 仅因表面 operator/token pattern 相同而
通过。Pattern Accuracy 与现有 `condition_accuracy(..., ConditionSpec("pattern", C))` 对齐，但正式实现应增加
canonical AST/round-trip 回归测试。

SMATCH 只作为 evaluation-only reference structural metric：允许在完成选择后计算，但 reference H 不得进入候选
选择、打分或 tie-break。

Pattern verifier 不使用以下 specific-relation 指标：

- nominal relation adherence；
- branch-supported relation；
- non-root branch-supported relation；
- matched relation replacement。

## 6. 冻结主选择规则

主方法使用 semantic-first、condition-preserving 的确定性词典序：

```text
1. exact + pattern_match
2. exact
3. highest SemAvg
4. pattern_match                    # 只在相同 SemAvg 内打破并列
5. model mean log probability
6. canonical hypothesis SHA256
```

解释：

- 已找到 exact candidate 时，优先选择同时满足 requested pattern 的 exact candidate；
- 若 exact candidates 都不匹配 pattern，仍选择 exact，不为 condition 外观主动放弃精确语义；
- 没有 exact candidate 时先最大化 SemAvg；pattern match 只作同 SemAvg tie-break；
- likelihood/hash 只用于完全确定性并列处理。

不得新增加权和、epsilon semantic band 或 pattern-first selector；这些都会引入需要 validation 调参的新自由度。

## 7. 必做 selector 消融

Pattern 版本最重要的证据不是只比较 K=4 与 greedy，而是区分增益来自普通 Best-of-N、execution semantics，
还是 pattern-aware tie-breaking。对同一组冻结 K=4 candidates 离线计算：

1. `first-sample`：candidate 0；
2. `likelihood-only`：最高 mean log probability；
3. `semantic-only`：最高 SemAvg，随后 likelihood/hash；
4. `exact-semantic`：exact > highest SemAvg > likelihood/hash，不读取 C；
5. `pattern-aware`：第 6 节主规则。

所有 selector 必须复用同一 candidate records，不能为某个 selector 重新生成候选。报告：

- `pattern-aware - greedy`；
- `pattern-aware - first-sample`；
- `pattern-aware - exact-semantic`，用于隔离 condition-aware selection 的贡献；
- candidate coverage：`any exact`、`any pattern_match`、`any exact+pattern_match`。

## 8. Experiment Pattern-P1：validation freeze

输入：1,664 条 frozen validation、conditional epoch 45、observable valid graph、seed 314159。

正式 decoding comparison：

1. greedy；
2. sample K=1；
3. sample K=2 + verifier；
4. **sample K=4 + pattern-aware verifier**；
5. sample K=8 + verifier，仅作成本/coverage 上界；
6. 第 7 节 K=4 selector ablations。

必须分别报告，不能只展示聚合均值：

- Jaccard；
- Dice；
- Overlap；
- SemAvg；
- observable-graph exact；
- Pattern Accuracy / pattern match；
- SMATCH（evaluation only）；
- parse、EOS、max-length；
- mean unique AST；
- mean generation tokens；
- generated sequences、graph executions、generation/verification/wall time；
- 13 patterns 的分层结果；
- K=4 与 greedy 的 paired win/tie/loss。

已知 epoch-45 validation 只用于启动 sanity check，不作为新 gate 的目标值：Jaccard `0.6196`、Dice
`0.6746`、Overlap `0.7455`、Pattern Accuracy `0.9591`、SMATCH `0.8283`、parse `0.9994`。若 greedy
与这些值出现无法由精度/实现版本解释的明显偏差，应停止并检查 checkpoint、prompt 和 graph split。

### 8.1 Bootstrap

使用 10,000 次 paired record bootstrap，固定 bootstrap seed `271828`，报告 95% CI：

- K=4 pattern-aware minus greedy：SemAvg、exact、Pattern Accuracy、SMATCH；
- K=4 pattern-aware minus exact-semantic：SemAvg、Pattern Accuracy；
- K=4 pattern-aware minus likelihood-only：SemAvg、Pattern Accuracy。

13-pattern 结果用于描述 heterogeneity，不对 13 个 slice 分别做无校正显著性声明。

### 8.2 Pattern-P1 gate

只有同时满足以下条件才进入 Pattern-P2：

1. K=4 pattern-aware SemAvg 不低于 greedy；
2. K=4 exact rate 不低于 greedy；
3. K=4 Pattern Accuracy 不低于 greedy `-0.005`；
4. K=4 parse 和 EOS 均不低于 greedy `-0.005`；
5. pattern-aware 相对 exact-semantic 的 Pattern Accuracy 不下降；
6. pattern-aware 相对 exact-semantic 的 SemAvg 不低于 `-0.005`；
7. K=4 wall time、generated sequences、tokens 和 graph executions 完整记录；
8. selection 只使用 `O`、observable `G`、`C_pattern` 和 candidate，不读取 reference H；
9. 所有正式输入、checkpoint、records 和 summary hashes 完整且可复核。

Gate 不要求一个事后选定的最小提升幅度。若 gate 失败，停止，不修改 selector 后重跑 P1/P2；失败结果仍记录。

## 9. Experiment Pattern-P2：冻结 historical-test comparison

仅在 P1 gate 通过后运行一次。P2 loader 必须从 passed P1 summary 恢复并强制校验：

- 相同 clean Git SHA；
- checkpoint tree/pointer/config/sampling hashes；
- seed 314159；
- K=4；
- batch size、max tokens、sampling contract；
- selection rule 和 selector implementation version；
- P1 records/summary hashes。

P2 在 1,664 条 historical test 上报告：

1. greedy；
2. sample K=1/2/4 prefix summaries；
3. 冻结 K=4 pattern-aware 主方法；
4. 冻结的 likelihood-only、semantic-only、exact-semantic ablations；
5. 与 P1 相同的全量、逐 pattern、cost 和 paired bootstrap 指标。

不得根据 P2 结果改变任何方法或再跑第二次 test。报告中必须明确：

```text
historical_test_previously_accessed = true
post_evaluation_tuning_permitted = false
```

已知 epoch-45 historical-test greedy sanity values：Jaccard `0.60304`、Dice `0.65173`、Overlap
`0.72159`、Pattern Accuracy `0.94291`、SMATCH `0.81681`、parse `0.99159`、EOS `1.0`。这些值只用于
检测错误 checkpoint/prompt/split，不用于选择或修改方法。

## 10. 实现建议与代码边界

建议新增独立入口：

```text
akgr/reproduction/pattern_verifier_best_of_n.py
tests/test_pattern_verifier_best_of_n.py
```

期望 CLI：

```bash
python -m akgr.reproduction.pattern_verifier_best_of_n p1 \
  --experiment-config akgr/configs/reproduce/wn-pattern-full-train-author-aligned.yml \
  --checkpoint /mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-full-train-author-aligned-c4/phase-d-parent \
  --checkpoint-pointer /mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-full-train-author-aligned-c4/phase-d-parent.json \
  --output-dir /mnt/workspace/ctrlhgen-runs/pattern-best-of-n-p1-<date>-<sha> \
  --seed 314159 \
  --batch-size 32

python -m akgr.reproduction.pattern_verifier_best_of_n p2 \
  --p1-summary /mnt/workspace/ctrlhgen-runs/pattern-best-of-n-p1-<date>-<sha>/summary.json \
  --output-dir /mnt/workspace/ctrlhgen-runs/pattern-best-of-n-p2-<date>-<sha>
```

可以从 `verifier_best_of_n.py` 提取纯通用 helpers（nested sampling、transition log-prob、semantic scores、
atomic JSONL、cost accounting），但必须遵守：

- 不改变已经完成的 `specific_relation` artifact/schema/selection 语义；
- 不让 pattern runner 依赖 specific-relation Phase 2 preflight；
- 不复制 relation nominal/neutralization 逻辑；
- reference target 只进入 evaluation-only SMATCH，不进入 selector；
- 正式 runner 要求 clean Git worktree、独占 output dir、原子 status/summary 写入和异常 traceback。

最低回归测试：

1. canonical pattern extraction 与现有 `condition_accuracy` 一致；
2. malformed prediction 不能 pattern-match；
3. exact+pattern 优先于 exact-only；
4. exact-only 优先于 nonexact pattern-match；
5. 无 exact 时 SemAvg 优先，pattern 只能同分 tie-break；
6. likelihood 后由 canonical hash 确定性打破并列；
7. K=1/2/4/8 candidate prefixes 严格嵌套；
8. reference H 改变不能改变 selected candidate；
9. checkpoint pointer/tree/config/data lineage fail closed；
10. P2 拒绝 failed P1、dirty/different SHA、method/hash 变化或第二个 output reuse。

## 11. Artifacts 和实验记录

每个 stage 至少写出：

```text
status.json
summary.json
records.jsonl
run.log
pid/control metadata
```

每条 record 包含 input ID、source/condition hashes、greedy、全部 sample candidates、各 selector selected index/
reason、semantic/pattern/parse/generation fields。Summary 写出 command、Git SHA、dirty=false、GPU、seed、K、
sampling、checkpoint tree、config/data/KG/manifest hashes、artifact hashes、wall time 和 graph accounting。

完成后新增：

```text
worklogs/SC-IDC/pattern_best_of_n_experiment_record_<date>.md
```

记录实现提交、测试、P1 gate、P2 限制、全部核心指标、artifact paths/hashes、异常与非权威 smoke runs。

## 12. 运行和资源收尾

DSW 当前已停止。接手 agent 应：

1. 本地实现、测试、只提交本任务文件并 push `origin`；
2. 启动 DSW 后检查 dirty state，不得丢弃远端改动；
3. fetch 并 clean detach 到精确实现 SHA；
4. 先运行定向测试和 2-record validation GPU smoke；
5. P1 使用后台进程运行，约每 5 分钟轮询 status/log/GPU；
6. P1 gate 通过才运行一次 P2；明显异常或 gate 失败立即停止后续阶段；
7. 实验完成后提交/push 实验记录，并让 DSW 使用记录提交；
8. 确认 GPU 空闲、无实验进程后，通过官方 PAI DSW StopInstance API 停止实例；
9. 保存脱敏 StopInstance response，并确认 SSH 不再提供服务。

## 13. 成功与失败的解释边界

若 P1/P2 通过，可报告：

> 在既有 WN18RR conditional-pattern SFT generator 上，frozen verifier-guided Best-of-4 decoding 在保持
> pattern adherence/生成健康度的同时改善 observable-graph semantic quality；结合 specific-relation 结果，
> 该 inference selection 思路覆盖两种 condition kinds。

不能报告：

- generator training improvement；
- 新 test 上的首次 sealed generalization；
- pattern_match 等于逻辑正确性；
- observable exact 等于完整隐藏 KG 上的逻辑等价；
- 单 seed、单数据集结果已证明普遍有效；
- 忽略 Best-of-N generation/execution 成本的免费提升。

若 pattern-aware 与 exact-semantic 几乎相同，但两者都显著优于 greedy，应解释为 execution-semantic Best-of-N
有效，而 pattern-aware tie-breaking 的额外贡献有限；不能把普通 Best-of-N 增益全部归因于 pattern verifier。
