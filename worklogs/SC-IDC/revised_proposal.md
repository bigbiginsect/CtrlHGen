# 修订方案：基于天然多解集合的可辨识语义控制

英文暂名：**Solution-Set Contrastive Semantic Control，SS-CSC**  
适用范围：WN18RR / `specific_relation`  
状态：取代 SC-IDC v1 的继续训练路线；原 SC-IDC executor 保留为数据审计与评价组件

## 1. 修订动机

SC-IDC v1 希望把 semantic control 从“condition token 是否出现”提升为“condition 是否位于有边际的
解释分支，并优于匹配替代值”。第一次 paired GRPO 没有显示稳健增益。后续诊断表明：

1. SFT 中从同一 target hypothesis 动态抽取不同 relation conditions，但监督 target 不变，condition
   不需要决定输出即可取得低 loss；
2. original-reward GRPO 仍能显著改善 semantic 和 prompt-swap response，说明 pipeline 与通用 RL
   并未失效；
3. SC-IDC 相对 baseline 只改善少量 likelihood preference，没有形成一致的 behavioural gain；
4. matched replacements 大量执行为空，使 BSS 主要重复 base semantic quality 与 KG sparsity；
5. 现有 train data 天然包含大量同 observation、多 hypothesis、多 relation-set 的 solution groups。

因此修订方向不再是在 original reward 上叠加更复杂的终局奖励，而是把反事实执行用于构造**可辨识的
条件监督**：同一 observation 在不同 relation conditions 下明确对应不同、都能精确解释 observation
的 hypothesis。

## 2. 研究问题

给定 KG (G) 和 observation (O)，定义 train-graph solution set：

\[
\mathcal S_G(O)=\{H:[H]_G=O\}.
\]

现有单 target 监督只提供一个 (H\in\mathcal S_G(O))，然后从 (R(H)) 抽取 condition。新的目标是从
同一个 solution set 中找到：

\[
H_1,H_2\in\mathcal S_G(O),\qquad H_1\ne H_2,
\]

以及可区分的 relation values：

\[
C_1\in R(H_1)\setminus R(H_2),\qquad
C_2\in R(H_2)\setminus R(H_1).
\]

训练目标应使模型在同一 observation 下能够根据 (C_1/C_2) 选择 (H_1/H_2)，而不是只学习
condition 与某个 target token 的弱相关。

本方案仍要求 controlled relation 位于有边际的分支，但不声称 predicate necessity。branch marginal
用于过滤和评价；当前 matched-replacement score 不进入主训练目标。

## 3. 已有可行性证据

train-only exact-observation audit 显示：

| 数据 | unique observations | 双向 exclusive-relation solution groups |
|---|---:|---:|
| SFT base train 104k | 66,053 | 9,253（14.01%） |
| fresh RL train 104k | 66,764 | 10,361（15.52%） |
| 两者只读合并审计 | 104,498 | 21,602（20.67%） |

在 base train 的 208-group deterministic pilot 中：

- 416/416 hypotheses 精确执行回相同 observation；
- 416/416 selected conditions 不出现在配对 hypothesis 中；
- 75.48% hypothesis sides 为 branch-supported；
- 62.98% pairs 两侧都为 branch-supported；
- replacement empty rate 仍为 70.03%。

这支持直接重组现有 base train，而不是先开发新的 condition-fixed symbolic search。fresh RL pool 只证明
潜在覆盖，不在第一版中与 SFT 混用。

## 4. 数据构造

### 4.1 Observation-level grouping

仅使用 SFT base train：

1. 对每条 raw query 在 train graph 上复核 exact denotation；
2. 用排序后的 answer entity set 作为 observation identity；
3. 同组内按 canonical query hash 去重；
4. 计算每条 query 的 unique relation-value set；
5. 枚举 relation sets 双向差集都非空的 hypothesis pairs。

不读取 validation、test 或 sealed final evaluation。第一版不使用 augmented-only records，因为它们的
observation 来自 sublogic decomposition，分布与 base target 不同；也不把既有 fresh RL pool 直接改作
SFT，以保留方法比较和数据 lineage 清晰。

### 4.2 Effective-condition filtering

对候选 (C\in R(H_i)\setminus R(H_j))：

1. 验证 condition 出现在合法 projection relation slot；
2. 联合处理同值 repeated occurrences；
3. 计算最近分支 neutralization 的
   \(\Delta_C^{\mathrm{branch}}\)；
4. 第一版要求 \(\Delta_C^{\mathrm{branch}}>0\)；
5. matched-replacement delta 只记录，不作为保留门槛。

若一个 hypothesis 有多个 exclusive relations，按预注册规则选择 branch delta 最大者，再按 relation token
打破并列。这样使用执行器选择已知有效 condition，但不把随机 replacement sparsity 写进标签。

### 4.3 Pair selection and weighting

每个 observation 最多保留固定数量的 pairs，避免高碰撞 observation 主导训练。推荐第一版：

- 每个 observation 最多 2 个 hypothesis pairs；
- 每个 pair 形成两个有方向的监督样本
  ((O,C_1,H_1,H_2)) 与 ((O,C_2,H_2,H_1))；
- observation-level uniform sampling；
- repeated relation value 不按 occurrence 加权；
- train/validation pair manifests 分别冻结并 hash。

pair validation 必须来自 train-only solution groups 的预留 observation groups，而不是复用原 1,664 条
模型 validation 来选择所有超参数。原 validation 继续承担与既有模型可比的外部指标。

### 4.4 数据门禁

进入训练前必须满足：

- 每条 hypothesis 重新执行后 exact 等于 observation；
- condition exclusivity、branch support 和 query parse 全部通过；
- observation identity 在 pair-train 与 pair-validation 间不重合；
- 与 sealed final evaluation 的既有隔离契约不被放宽；
- 至少 5,000 个两侧 branch-supported 的 train pairs，否则缩小论断并先做数据方法研究。

208-group pilot 的 62.98% 双侧通过率意味着 9,253 个 natural groups 可能提供约 5,800 个 pairs；这是
规划估计，不代替 full audit 后的实际冻结数量。

## 5. 模型目标

### 5.1 Paired SFT

保留标准 conditional autoregressive loss：

\[
\mathcal L_{\mathrm{SFT}}
=-\log p_\theta(H_C\mid O,C).
\]

与 v1 不同，同一 observation 的不同 conditions 现在对应不同 target hypotheses。

### 5.2 Prompt-swap contrastive margin

定义 target token mean log probability：

\[
s_\theta(H\mid O,C)
=\frac1{|H|}\sum_t\log p_\theta(h_t\mid h_{<t},O,C).
\]

对一个 pair 加入对称 margin：

\[
\mathcal L_{\mathrm{swap}}
=\max(0,m-s(H_1\mid O,C_1)+s(H_1\mid O,C_2))
\]

\[
\quad+\max(0,m-s(H_2\mid O,C_2)+s(H_2\mid O,C_1)).
\]

因为 (C_1\notin R(H_2)) 且 (C_2\notin R(H_1))，这里的 prompt swap 不再像 v1 一样对同一个多 relation
target 形成含混监督。

总损失：

\[
\mathcal L=\mathcal L_{\mathrm{SFT}}+\lambda_{\mathrm{swap}}\mathcal L_{\mathrm{swap}}.
\]

首轮只比较一个预注册 margin 和一个保守 \(\lambda_{\mathrm{swap}}\)，不在 validation 上做大网格搜索。

### 5.3 暂不使用在线 SC-IDC reward

第一轮不运行 GRPO，原因是：

- original-reward GRPO 已证明有效，不是当前最需要验证的变量；
- BSS 没有提供足够独立排序；
- paired supervision 可以直接检验 condition 是否成为解选择变量；
- 避免把数据构造、contrastive objective 和新 reward 三个变化同时引入。

只有 paired SFT 通过直接控制 gate 后，才讨论是否在同一 parent 上做 short paired GRPO。

## 6. 实验设计

### Experiment A：Full pair-data audit

这是无模型训练的正式数据实验。报告：

- exact-observation group 数量和 cardinality 分布；
- distinct hypotheses、relation sets 和 exclusive relation support；
- branch-supported condition coverage；
- pattern/topology/answer-cardinality 分层；
- 每个 observation 可形成的 pairs 数量；
- train/pair-validation isolation 与全部 hashes。

### Experiment B：小规模 paired SFT 决策实验

从同一 unconditional parent 分叉，update-matched 比较：

1. **single-target baseline**：现有 uniform-value conditional SFT 规则；
2. **paired-data SFT**：新数据，但只有标准 SFT loss；
3. **SS-CSC**：新数据 + prompt-swap contrastive margin。

先运行 10--15 epochs 或相同 optimizer updates 的小规模 pilot，每 5 epochs 在 pair-validation 和原模型
validation 上评估。第一轮只用一个 seed，明确称为决策实验。

paired-data SFT baseline 用来区分“多解重组本身的收益”和“contrastive objective 的增量”。不能只比较
旧 SFT 与 SS-CSC，否则无法归因。

### Experiment C：冻结 parent 的直接解选择评估

对每个 gold pair 同时输入 (C_1,C_2)，报告：

1. assigned target mean logp；
2. swapped target mean logp；
3. symmetric margin satisfaction；
4. 在 \(H_1/H_2\) 两个候选间的 condition-consistent selection accuracy；
5. greedy 输出的 assigned adherence；
6. paired-condition-only response；
7. AST/denotation change；
8. semantic Jaccard/Dice/Overlap；
9. branch-supported controlled-value rate。

这里 (C_1,C_2) 都有真实 gold alternative，因而比旧 absent-condition stress test 更能评价“condition 是否
选择解”。旧 static-matcher absent audit 继续作为 out-of-reference robustness，不能作为主正确率。

### 可选 Experiment D：短程 GRPO

只有 Experiment B/C 通过后，才从同一 SS-CSC checkpoint 分叉：

- original reward；
- 一个重新设计且先通过 offline ordering gate 的 effective-control reward。

首轮最多 500--1,000 updates，不直接运行 full GRPO。若新增 reward 相对 base 的 pairwise ordering
改变率和 normalized-advantage 改变量仍接近 SC-IDC v1 的 `4.3%/2.4%`，停止。

## 7. 评价门槛

具体阈值在 full data audit 后冻结，但方向性 gate 预先固定：

1. SS-CSC 必须显著提高 paired candidate selection 与 symmetric margin satisfaction；
2. 相对 paired-data SFT，而不是只相对旧 single-target SFT，仍需有增量；
3. 原 validation assigned SemAvg 不允许出现明显退化；建议容忍线先设为最佳对照的 `-0.01`；
4. assigned nominal adherence、parse、EOS 不退化；
5. branch-supported rate 提高不能仅来自复制 condition 或 root/chain 自动通过；
6. 不使用 sealed final evaluation 选择 epoch、margin、lambda 或后续 reward。

若 contrastive 指标改善但自由生成不变，下一步才考虑 constrained decoding；它必须作为单独推理方法报告，
不能把机制保证的 nominal adherence 当成模型学习收益。

## 8. 与 SC-IDC v1 的继承关系

### 保留

- immutable AST 与 graph executor；
- unique relation-value 与 repeated-occurrence joint semantics；
- branch neutralization；
- deterministic manifests、split isolation、checkpoint lineage；
- nominal、branch support、matched selectivity 分开报告的论断边界。

### 放弃或降级

- `base_score × branch_clip × matched_clip` 的在线 BSS reward；
- static global matcher 随机抽三个 replacements 作为训练主信号；
- empty-root/chain 分数与普通 branch score 混成单一有效性结论；
- 只凭自然 assigned condition accuracy 选择 checkpoint；
- 在单 target、多 condition 监督上继续增加 SFT epochs。

### 后续研究项

predicate-level identity intervention、query-local hard replacements 和 constrained decoding 仍有研究价值，但
不是 SS-CSC 第一轮的必要组成。第一轮只回答一个清晰问题：**同 observation 的可验证多解监督，能否让
relation condition 真正成为解选择变量？**

## 9. 论断边界

- 结果只覆盖 WN18RR `specific_relation`；
- exact denotation 是当前 observable train graph 上的等价，不代表完整 KG；
- branch support 不等于 predicate necessity；
- natural solution collisions 的分布可能偏向高频、小 answer-set 或某些 patterns，必须分层报告；
- 单 seed pilot 只作方法决策；稳定模型结论仍需要预注册多 seed；
- pattern 完整复现是 pipeline 正控制，不是 SS-CSC 的模型级 baseline。

## 10. 推荐执行顺序

```text
full train-only solution-set audit
  -> pair manifests + isolation gate
  -> paired-data SFT baseline / SS-CSC short pilot
  -> paired gold condition-selection evaluation
  -> 通过后再决定是否需要 constrained decoding 或 short GRPO
```

这条路线把创新焦点从“设计更复杂的终局奖励”转为“让 semantic condition 在监督数据中可辨识”，同时
复用已经完成的 pattern reproduction、specific-relation checkpoints、执行器和数据资产，避免重复训练
已经验证过的基础链路。

