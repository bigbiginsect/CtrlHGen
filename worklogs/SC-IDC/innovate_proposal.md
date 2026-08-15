## 方案一（调整版）：控制值对齐的反事实指称贡献

英文暂名：**Semantic-Control Interventional Denotation Credit，SC-IDC**

### 1. 核心动机

SC-IDC 针对的基线将“满足控制条件”主要定义为：

- 生成指定的逻辑结构或元素数量；
- 指定实体/关系 token 出现在假设中。

但“条件出现”不等于“条件真正参与了解释”。模型可以将指定条件放入一个不影响最终结论的冗余分支，同时获得很高的语义相似度和条件遵循奖励。

SC-IDC 希望把语义控制从：

> 条件是否出现在假设里

提升为：

> 在保持结构和其他 token 不变时，这个条件是否位于有边际的解释分支中，并优于匹配替代值。

需要特别限定的是：**只重点奖励用户指定的 controlled semantic value，不要求所有实体和关系
槽位都不可替代。** 一个正确假设可以在当前 KG 上存在合理的外延冗余；SC-IDC 不把“每个
predicate 都不可删除”当作正确性的必要条件，也不声称 branch-supported selectivity 等同于
predicate necessity。

---

### 2. 将控制条件分为两类

#### 2.1 硬结构控制

包括：

- logic pattern；
- relation number；
- entity number。

这些条件可以被形式化验证，适合直接编译进 grammar/constrained decoding，使生成分布满足：

\[
\pi_\theta(H\notin\mathcal H(C)\mid O,C)=0.
\]

它们属于“合法假设空间”的定义，不必继续通过二元终局奖励学习。

#### 2.2 有效语义控制

包括：

- specific entity；
- specific relation。

首先保证指定 token 确实出现在合法逻辑位置，即 nominal adherence；然后分别审计它所在最近
逻辑分支的边际贡献，以及它相对匹配替代值的选择性。

因此：

- nominal adherence 负责“条件出现在合法位置”；
- SC-IDC 负责审计条件的 branch support 与 matched selectivity。

---

### 3. 可辨识的控制值采样与 occurrence 规则

specific relation prompt 暴露的是 relation 值 \(C\)，而不是 AST 中的 slot 地址。同一个值即使在
目标中出现多次，模型看到的条件仍然只是同一个 `COND C`；因此训练控制单位必须是 unique
relation value，不能把模型不可区分的重复 occurrence 当成不同监督。

对目标假设 \(H^\star\)，记 relation occurrence 集和 unique value 集为：

\[
Z_R(H^\star)=\{(\mathrm{path}_j,r_j)\}_{j=1}^{m},
\qquad
V_R(H^\star)=\mathrm{Unique}\{r_j\}_{j=1}^{m}.
\]

specific-relation SFT 的每个 epoch 按以下规则采样：

\[
C\sim\mathrm{Uniform}(V_R(H^\star)).
\]

训练条件可以由 `seed + record_id + epoch` 动态派生；validation、test、rollout signal set 和
正式 RL train 的 `(record_id, condition_kind, condition_value)` 必须提前物化、冻结并 hash。
两个 GRPO 分支共享完全相同的 RL condition manifest。

对生成假设 \(H\)，定义所有匹配 occurrence：

\[
\mathrm{Occ}(C,H)
=
\{\mathrm{path}_j:r_j=C\}.
\]

nominal adherence 只判断该集合是否非空。SC-IDC 奖励对全部 occurrence 做联合干预：一个
matched replacement 同时将所有 \(C\) occurrence 替换为同一个 \(C'\)；branch neutralization
则联合中性化这些 occurrence 所属、去重后的最近逻辑分支。逐 occurrence 结果仅用于 attribution，
不作为模型训练时彼此不可辨识的独立控制样本。

如果任务真的要求控制“第几个 slot”，就必须把 path/role 编码进 prompt；这属于新的控制任务，
不在本方案范围内。

---

### 4. 受控值的边际门控与匹配反事实

#### 4.1 matched replacement 不能单独识别装饰分支

原始设想只比较受控值和匹配替代值，但存在如下反例：

\[
H=H_0\lor B_C,\qquad [B_C]_G\subseteq[H_0]_G.
\]

此时包含 \(C\) 的分支对当前结论没有边际贡献，属于本方案希望识别的 control laundering。
然而，若某个匹配替代 \(C'\) 使 \(B_{C'}\) 引入额外假阳性，仍可能出现：

\[
R_{\mathrm{sem}}(H)-R_{\mathrm{sem}}(H_{C'})>0.
\]

正的 matched delta 因此只能说明“当前值优于这个替代值”，不能证明当前值所在分支不是
装饰。因此 SC-IDC 增加独立的逻辑分支边际门控。

#### 4.2 受控值所在最近逻辑分支的中性化边际

对于 \(\mathrm{Occ}(C,H)\) 中的每个 occurrence，沿逻辑树向上寻找最近的
intersection/union 分支祖先，将对应的直接子分支 path 去重后联合中性化：

- intersection 的恒等元为全集；
- union 的恒等元为空集；
- 去重后若一个待中性化 path 是另一个 path 的后代，只保留祖先 path，形成互不嵌套的
  ancestor-most antichain；
- 若任一受控 occurrence 不存在分支祖先，则加入 root marker，联合基线整体退化为空根查询。

记去重后的分支集合为 \(B_C\)，所得内部审计查询为 \(H^{(-B_C)}\)，定义：

\[
\Delta_C^{\mathrm{marg}}
=
R_{\mathrm{sem}}(H)-R_{\mathrm{sem}}(H^{(-B_C)}).
\]

这一量只衡量“包含受控值的最近分支集合”是否有边际贡献，不是 controlled predicate 本身的
删除效应。

#### 4.3 branch support 仍不等于 predicate necessity

即使 branch marginal 和 matched delta 都为正，controlled relation 本身仍可能冗余。例如：

\[
H=(A\land C)\lor D,
\qquad
[A\land C]_G=[A]_G.
\]

当 \(A\land C\) 整体分支对观测必不可少时，删除整个分支会得到正 branch marginal；若某个
\(C'\) 让该分支变差，matched delta 也可以为正。但删除 \(C\) 后 \(A\) 的外延不变，所以
这两个正 delta 仍不能证明 slot necessity 或单 predicate 因果必要性。

因此本方案只将两个正 delta 称为 **branch-supported selectivity**。只有未来能够在保持变量接口
和查询合法性的前提下严格定义 predicate neutralization 时，才额外报告 slot necessity；无法严格
定义的结构应标为不可识别。

#### 4.4 受控值的匹配替代优势

从匹配分布中选择替代值：

\[
C'\sim q_{\mathrm{match}}(r'\mid C,H,G),
\qquad
H^{(C')}=\operatorname{do}_{\mathrm{all}}(H,C\leftarrow C').
\]

在 WN18RR 上不虚构不存在的显式类型信息，而使用 observable graph 可推导的代理：

- relation：正/反方向、边频率区间、头尾实体的有向关系签名、反事实答案基数区间；
- entity：anchor 角色、相邻 projection 可达性、节点度数区间、有向 incident-relation 签名。

具体候选流程固定为：entity 先按派生 seed 确定性预筛最多 256 个，再按图特征保留 16 个；
entity/relation 都从综合排序最优的 8 个中按派生 seed 无放回采样 3 个。候选不足时按固定层级
放宽并记录 fallback，且始终排除原 token。

Phase 2 的 relation 训练 matcher 对每个 \(C\) 只建立一份静态排序：依次按方向 fallback 层级、
edge-frequency log-bin 距离、负的 head/tail endpoint-profile cosine、relation token id 排序。从最优
8 个中，以 `(reward_seed, record_id, canonical_query_hash, C)` 派生 seed，无放回选择 3 个。所有
occurrences 共享每次选出的同一个 \(C'\)，不允许为不同位置分别选择模型无法从 prompt 区分的
替代值。穷尽 fallback 后不足 3 个不同候选时，该 completion 记为 `unscorable`。

替换同时改变所有目标 occurrences，但保持 AST、logic pattern、occurrence 数量和其他 token
不变。matched intervention 用于衡量受控值的选择性，不单独承担“非装饰性”或 necessity 判定。

---

### 5. SC-IDC 双量审计与候选奖励

首先定义基础语义质量：

\[
R_{\mathrm{sem}}(H)=S([H]_G,O).
\]

离线核心固定 \(S\) 为 Jaccard。受控 relation value 的联合匹配替代优势为：

\[
\Delta_C^{\mathrm{match}}(H)
=
R_{\mathrm{sem}}(H)
-
\mathbb E_{C'\sim q_{\mathrm{match}}}
R_{\mathrm{sem}}(H^{(C')}).
\]

SC-IDC 同时保留两个判据：

\[
\mathrm{branch\_supported}
=
\mathbf 1[\Delta_C^{\mathrm{marg}}>\epsilon],
\qquad
\mathrm{matched\_selective}
=
\mathbf 1[\Delta_C^{\mathrm{match}}>\epsilon].
\]

据此区分：

- `branch_nonmarginal`：nominal 为 1，但 branch marginal 不为正；
- `branch_supported_nonselective`：branch marginal 为正，但未识别出正的 matched advantage；
- `branch_supported_selective`：branch marginal 与 matched advantage 均为正；
- `unscorable`：穷尽 fallback 后仍不足 3 个不同合法替代候选。

这四类只在 parse-ok 且 nominal 的 completion 内判定；parse-fail 与 nominal-fail 单独统计。
`unscorable` 具体指穷尽 fallback 后不足 3 个不同替代候选，仍可保留已经得到的 branch delta
作为诊断，但不计算 matched delta。

确定性图执行的主判定固定为 \(\epsilon=0\)，同时报告 \(\epsilon=0.01/0.05\) 的敏感性；归一化
诊断固定使用 \(\tau_{\mathrm{marg}}=\tau_{\mathrm{match}}=0.1\)。

后续若进入训练，可审计如下候选奖励：

\[
R_{\mathrm{BSS}}(H,C)
=
\mathbf 1[\mathrm{Occ}(C,H)\ne\varnothing]\,
w(R_{\mathrm{sem}}(H))\,
\operatorname{clip}
\left(
\frac{\Delta_C^{\mathrm{marg}}(H)-\epsilon}{\tau_{\mathrm{marg}}},
0,1
\right)
\operatorname{clip}
\left(
\frac{\Delta_C^{\mathrm{match}}(H)-\epsilon}{\tau_{\mathrm{match}}},
0,1
\right).
\]

离线信号审计固定取 \(w(R_{\mathrm{sem}})=R_{\mathrm{sem}}\)，并将其作为语义奖励的受控
relation-value 附加项。这里不对 occurrence 或全部槽位求和。该分数只编码
branch-supported selectivity，不能命名为 slot necessity 或严格因果效应；新增奖励系数
\(\alpha_{\mathrm{IDC}}\) 只使用 train-only signal set 选择并在正式训练前冻结。该符号与现有
GRPO 的 KL 系数 \(\beta_{\mathrm{KL}}\) 严格区分。

---

### 6. 非受控槽位如何使用 IDC

对于不属于 \(\mathrm{Occ}(C,H)\) 的其他槽位 \(z\)，仍然可以计算：

\[
\Delta_z
=
R_{\mathrm{sem}}(H)
-
\mathbb E_{z'}R_{\mathrm{sem}}(H_{z\leftarrow z'}).
\]

但它们主要用于：

- 解释每个 relation/entity 的作用；
- 识别 dead branch；
- 诊断冗余或有害谓词；
- 分析不同 operator 的 precision/recall 贡献；
- 构造后续的 hard-negative preference pairs。

不建议优化：

\[
\sum_{z\in H}[\Delta_z]_+,
\]

因为这会偏向“所有 predicate 都必须外延不可替代”的假设。

如果确实希望加入全局正则，更安全的是只弱惩罚明显有害的组件：

\[
R_{\mathrm{harm}}
=
-\lambda_{\mathrm{harm}}
\frac1{|Z(H)|}
\sum_{z\in Z(H)}
\max(0,-\Delta_z),
\qquad
\lambda_{\mathrm{harm}}\ll\alpha_{\mathrm{IDC}}.
\]

这样：

- \(\Delta_z>0\)：有益，但不额外强迫其变得更大；
- \(\Delta_z=0\)：允许合理冗余；
- \(\Delta_z<0\)：说明替换该槽位反而能改善解释，给予轻微惩罚。

第一轮实验甚至可以完全不使用 \(R_{\mathrm{harm}}\)，只把非受控 IDC 作为诊断指标。

---

### 7. Nominal、Branch Support 与 Selectivity 分开评估

SC-IDC 不取代 nominal adherence，也不把 branch support 误称为 predicate necessity。建议同时报告：

1. **Nominal Adherence**

\[
\mathrm{Acc}_{\mathrm{nom}}
=
\Pr(C\text{ 出现在合法位置}).
\]

2. **Branch-Supported Rate**

\[
\mathrm{Acc}_{\mathrm{branch}}
=
\Pr(\Delta_C^{\mathrm{marg}}>\epsilon).
\]

3. **Matched Selectivity**

\[
\mathrm{Acc}_{\mathrm{match}}
=
\Pr(\Delta_C^{\mathrm{match}}>\epsilon).
\]

4. **Branch-Supported Selectivity**

\[
\Pr(
\Delta_C^{\mathrm{marg}}>\epsilon
\land
\Delta_C^{\mathrm{match}}>\epsilon
).
\]

5. **Branch-Nonmarginal Rate**

\[
\Pr(
\mathrm{nominal}=1
\land
\Delta_C^{\mathrm{marg}}\le\epsilon
).
\]

第五项只衡量“条件虽然出现，但其最近逻辑分支集合没有正边际”。在自然 reference/model
outputs 上不能直接把这一比例命名为 laundering，因为图冗余不等于模型有意规避控制。

6. **Labeled Laundering Detection**

只在人工构造、已知受控分支被其他 OR 分支遮蔽的对抗集上报告 detection rate。该指标验证
branch gate 能否发现已知 laundering，不把 gold reference 的自然冗余反向解释成 laundering。

需要明确：任一 delta 为 0 都不代表假设错误；两个 delta 均为正也不证明 predicate necessity。
nominal、branch support、matched selectivity 和 slot necessity 是不同层级的命题。

---

### 8. 验证与正式实验流程

模型级主实验只选择 `specific_relation`，不重新复现其余四种 controls。SC-IDC 的逻辑核心仍可
复用于 entity/relation，但本轮模型结论只覆盖 relation control。

本方案只保留两个真正的研究实验：

1. **Experiment 1：离线奖励信号审计**；
2. **Experiment 2：specific-relation 配对 GRPO**。

合成反例、执行器等价和 manifest 校验属于 preflight tests，不再命名为实验；first/non-first/
repeated 只是在上述阶段复用同一输出得到的分层指标，不再单列 C-pre/C-post。

两条 preflight 工程线没有人为的先后依赖，可以并行完成；它们只在 Experiment 1 前汇合：

```mermaid
flowchart TD
    P1["数据/SFT preflight：parent import、sampler、manifests"] --> S["specific-relation conditional SFT"]
    P2["reward preflight：joint core、反例、回归审计"] --> E1["Experiment 1：离线信号审计"]
    S --> E1
    E1 -->|"有可学习信号且成本可接受"| E2["Experiment 2：配对 GRPO"]
    E1 -->|"无有效信号"| X["停止并修复，不启动 GRPO"]
    E2 --> R["总体结果 + occurrence-topology 分层报告"]
```

| 阶段 | 性质 | 阻断对象 |
|---|---|---|
| 数据/SFT preflight | 工程验收 | conditional SFT |
| reward preflight | 单元、对抗与集成验收 | Experiment 1 / GRPO，不阻断 SFT |
| specific-relation conditional SFT | 两个 GRPO 分支的共同 parent | Experiment 1 |
| Experiment 1 | train-only 离线 pilot 与奖励冻结 | Experiment 2 |
| Experiment 2 | 正式配对训练 | — |
| topology slices | 复用现有输出的诊断 | 不反向调参 |

#### 8.1 数据/SFT preflight

不得放宽通用 hash 校验来复用旧资产，而应实现两个显式导入契约：

1. **import-unconditional-parent**：允许新 specific-relation config 导入冻结的 unconditional
   权重；验证 source stage/config、data/KG、model/tokenizer 和 checkpoint SHA；新 conditional
   SFT 重置 optimizer/scheduler，并记录 source/target config lineage。
2. **condition-agnostic fresh-query rebind**：验证旧 artifact、sampling manifest、data/KG、
   record IDs、去重与 split 排除契约，生成新的 target condition manifests；旧 manifest 不改写。

train condition 按 epoch 从 unique relation values 动态派生；validation、signal、正式 RL train
和最终 evaluation conditions 分别物化、冻结并 hash。最终 evaluation/test manifest 由隔离步骤
生成并封存，训练、signal audit 和 checkpoint 选择不得加载。

#### 8.2 Reward preflight

preflight 必须在最终的 value-level joint core 上覆盖：

- **OR-append laundering**：受控分支被其他 OR 分支遮蔽时，nominal 和原 condition reward 为 1，
  但 joint branch marginal 为 0；同时覆盖单 occurrence 和重复同值 occurrences。
- **嵌套 predicate 冗余**：对
  \(H=(A\land C)\lor D, [A\land C]_G=[A]_G\)，允许 branch marginal 和 matched delta 都为正，
  但输出只能称为 `branch_supported_selective`，不能产生 predicate-necessity 论断。
- **兼容性回归**：保留 Phase 1 executor parity、结构保持和确定性 fixtures；新增 joint
  replacement、ancestor-most antichain、root marker、unique-value weighting 和 fallback 测试。
- **train-graph reference audit**：验证 reference denotation、joint intervention 结构不变量、
  确定性和无 test 访问。scorable coverage 作为报告指标；不能为了追求 100% coverage 而把 matcher
  放宽到失去可比性。

合成反例、结构不变量、执行器等价、确定性或 split 隔离失败时，禁止进入 Experiment 1。自然
reference 的 branch-nonmarginal 比例只作诊断，不设“必须有利”的阈值，也不命名为 laundering。
该 preflight 不依赖模型 checkpoint，因此可先做；但完整 reference audit 不是 conditional SFT 的
逻辑前置条件。

#### 8.3 specific-relation conditional SFT

从通过导入契约的同一 unconditional parent 开始，使用 uniform unique-value train sampler 与固定
validation condition manifest。必须满足：

- source/target config、parent checkpoint、tokenizer、data/KG 和 condition lineage 完整；
- optimizer/scheduler 从新 conditional stage 重置；
- checkpoint 选择只读取固定 validation conditions；
- 无非有限 loss，checkpoint 可重载，并复现 validation prompt identity；
- 完整报告 parse、EOS 和 nominal adherence。

这些是 SFT 阶段指标，不与 SC-IDC reward 的信号门禁混在一起。选定 checkpoint 后冻结为
Experiment 1 和 Experiment 2 的共同 parent。

#### 8.4 Experiment 1：离线奖励信号审计

物化一份 train-graph-only signal set，使其与 conditional SFT train、正式 RL train、validation
和最终 evaluation 的 query/supervision identity 不重合。在冻结 SFT parent 上生成 rollout，并对
同一批 completions 同时计算：

\[
R_{\mathrm{base}}
=R_J+0.5R_D+0.5R_O+R_{\mathrm{nom}},
\qquad
R_{\mathrm{aug}}
=R_{\mathrm{base}}+\alpha_{\mathrm{IDC}}R_{\mathrm{BSS}}.
\]

\(R_{\mathrm{BSS}}\) 使用全部同值 occurrences 的联合干预，固定
\(\tau_{\mathrm{marg}}=\tau_{\mathrm{match}}=0.1\) 且
\(w(R_{\mathrm{sem}})=R_{\mathrm{sem}}\)。parse-fail、nominal-fail 或 unscorable completion 的
SC-IDC 附加项为 0，基础 reward 仍按原规则计算。

首轮预算可继续采用 13 个 pattern 各 16 个 prompt、每组 4 generations，共 208 groups/832
completions；这是 pilot 预算，不宣称为统计上天然充分。训练版 matcher 先用静态图特征选出
\(K=3\) replacements，再执行 joint neutralization 和 replacements；共享 base denotation 并按
query hash 缓存，报告真实执行数和 p50/p95 wall-time。

主要报告：

- nominal、scorable 和含至少两条非相同可评分 completion 的 group 数；
- zero-reward-variance group rate 与非相同 completion reward-tie rate；
- unique executable AST / unique denotation；
- raw \(R_{\mathrm{BSS}}\) 的组内变化、与语义分数的联合分布；
- 由 IDC 引起的 Pareto-harmful inversion：新排序是否偏好一个语义更差且 SC-IDC 也不更好的候选；
- 每个 completion 的额外执行数和时间。

go/no-go 只保留可解释的硬条件：数据与 reward 逐行可复现；不存在 split 泄漏；至少 30 个 groups
含两条非相同、可评分 completion；raw SC-IDC 在这些 groups 的一部分中提供区别于 base reward 的
action-level ordering，且在 base reward 确有 ties 时至少能解开其中一部分；不存在实现错误导致的
Pareto-harmful inversion；实测成本落在本轮预先声明的资源预算内。tie 降幅和成本同时给出
bootstrap 区间或分布，不使用未经依据的 10%、5% 或 6 倍通用阈值。

\(\alpha_{\mathrm{IDC}}\) 可以在预注册的 train-only 小网格中比较，但必须基于 signal-set 的
semantic-control Pareto 表选择并记录规则；不得读取 validation/test 后反向选择。若 signal set
不支持稳定选择，则先固定一个保守系数做单 seed pilot，而不是伪造精确门槛。

#### 8.5 Experiment 2：specific-relation 配对 GRPO

Experiment 1 通过后，从同一冻结 SFT checkpoint 分叉：

- **uniform-value control + original-reward baseline**：保持原 Jaccard/Dice/Overlap 与 nominal
  condition reward；
- **SC-IDC GRPO**：在完全相同的基础 reward 上加入 controlled-relation SC-IDC 项。

两条分支共享 RL prompt/condition manifest、初始化权重、generation 配置、optimizer 配置和每个
seed 的 update 数；唯一方法差异是 SC-IDC reward。主比较采用 update-matched，另外完整报告累计
图执行数和 wall-time。若同一对运行自然产生了足够密集的 checkpoints，可补充一种 compute-matched
曲线；不强制额外训练来同时制造 graph-execution-matched 与 wall-time-matched 两套结果。

先运行一个 seed 的小规模 pilot 验证稳定性；正式模型结论原则上使用至少 3 个独立 seeds，并报告
均值和离散度。若资源只允许单 seed，必须明确称为 pilot，不作稳定增益结论。

比较 semantic Jaccard/Dice/Overlap、nominal adherence、branch-supported rate、matched
selectivity、branch-supported selectivity、branch-nonmarginal rate、parse/EOS、输出复杂度和训练
成本。已知 laundering 只在有标签对抗集上报告 detection。

baseline 不能称为“原始 CtrlHGen baseline”：它采用了 uniform unique-value condition，而原实现是
fixed-first condition。准确名称应保持为 `uniform-value control + original-reward baseline`。

entity control、all-slot IDC 和其他 structural controls 不进入主训练对照，不能从本实验外推其
模型级效果。

#### 8.6 统一的 occurrence-topology 分层

first/non-first/repeated 不是第三个实验。按照 target condition value 在冻结 target query 中的
occurrence topology，把 SFT validation、Experiment 1 和 Experiment 2 的既有输出统一分为：

- 只出现在第一个 relation slot；
- 只出现在 non-first relation slots；
- 在多个位置重复出现。

每组报告 support 和同一套主要指标，不额外生成 rollout、不选择 checkpoint、不修改 reward，也不
向模型提供 oracle path。分组 support 少于 30 时只报告描述统计，不作该组泛化结论。

---

### 9. 主要风险

- 匹配替代分布若不合理，\(\Delta_C^{\mathrm{match}}\) 可能只反映 degree 或稀有度差异。
- branch marginal 归因于最近逻辑分支，不等同于严格的单 predicate 因果贡献。
- 一个必要分支内部仍可能包含冗余 controlled predicate，因此两个正 delta 也不证明 slot necessity。
- 同义或外延等价关系可能让一个合理控制条件得到 \(\Delta_C^{\mathrm{match}}\approx0\)。
- 当前有限 KG 上的冗余不代表完整图上的冗余。
- 多次反事实执行会增加训练成本，需要缓存或限制替代样本数。
- 离线 audit matcher 与 compute-bounded training matcher 的候选分布不同，必须分别冻结并报告。
- effective control 是比论文 lexical control 更强的任务定义，因此必须同时报告 nominal adherence，不能悄悄替换原评价口径。
- 当用户给出多个 semantic values 时，应显式定义 value-level 联合干预，仍然不扩展到全部槽位。

### 最终概括

SC-IDC 的核心不是追求一个“逐 predicate 最小”的假设，而是：

> 保留合理的逻辑冗余，但要求用户明确指定的语义值不仅出现在假设中，而且位于有边际的解释
> 分支中，并相对匹配替代值表现出可审计的选择性。

SC-IDC 把“受控值所在分支集合是否有正边际”和“当前受控值是否优于匹配替代”分开报告。
两者同时成立时称为 branch-supported selective，而不是 strict effective 或 predicate-necessary。
这比原始二元 condition reward 提供更细的控制诊断，同时避免 matched-only 反例，也不会把
整个 abductive objective 偷换成过强的逻辑最小化目标。
