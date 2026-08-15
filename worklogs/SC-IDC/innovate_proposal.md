## 方案一（调整版）：槽位对称的受控反事实指称贡献

英文暂名：**Slot-Symmetric Controlled Interventional Denotation Credit，SC-IDC**

### 1. 核心动机

CtrlHGen 当前将“满足控制条件”主要定义为：

- 生成指定的逻辑结构或元素数量；
- 指定实体/关系 token 出现在假设中。

但“条件出现”不等于“条件真正参与了解释”。模型可以将指定条件放入一个不影响最终结论的冗余分支，同时获得很高的语义相似度和条件遵循奖励。

SC-IDC 希望把语义控制从：

> 条件是否出现在假设里

提升为：

> 在保持其他因素基本不变时，这个条件是否对假设的结论产生了实际且有益的影响。

需要特别限定的是：**只重点奖励用户指定的 controlled slot，不要求所有实体和关系槽位都不可替代。** 一个正确假设可以在当前 KG 上存在合理的外延冗余；SC-IDC 不把“每个 predicate 都不可删除”当作正确性的必要条件。

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

首先保证指定 token 确实出现在合法逻辑位置，即 nominal adherence；然后通过反事实干预判断它是否真正影响假设结论，即 effective adherence。

因此：

- nominal adherence 负责“条件被使用”；
- SC-IDC 负责“条件不是装饰性的”。

---

### 3. 槽位对称的控制条件采样

论文描述的是从目标假设中随机采样一个实体或关系，但当前实现固定选择序列中的第一个合法元素。这会造成明显的位置捷径。

调整后，对于目标假设 \(H^\star\) 的所有 eligible slots：

\[
Z(H^\star)=\{z_1,\ldots,z_m\},
\]

每个 epoch 均匀采样：

\[
z_C\sim\mathrm{Uniform}(Z(H^\star)).
\]

如果离线枚举全部控制条件，则每个条件样本赋予 \(1/m\) 权重，避免长假设因为槽位更多而被过度采样。

目标假设仍采用 canonical serialization，不需要依靠随机交换逻辑分支来制造表面多样性。

---

### 4. 受控槽位的匹配反事实干预

设生成假设为 \(H\)，用户控制条件对应槽位为 \(z_C\)。从匹配分布中选择替代值：

\[
z_C'\sim q_{\mathrm{match}}(z'\mid z_C,H,G).
\]

干预应尽量只改变语义选择，而不改变结构难度。

对于 relation，匹配：

- 关系方向；
- domain/range；
- 局部度数或答案集基数区间；
- 在逻辑树中的位置。

对于 entity，匹配：

- KG 类型；
- anchor/中间节点等逻辑角色；
- 邻域度数；
- 相关关系签名。

构造反事实假设：

\[
H^{(C')}=\operatorname{do}(H,z_C\leftarrow z_C').
\]

它与原假设保持相同的：

- logic pattern；
- entity/relation 数量；
- 变量绑定；
- 自由变量接口；
- 序列长度大致范围。

这比直接删除一个 projection 或 relation 更干净，因为删除可能改变查询层级、复杂度和答案基数分布。

---

### 5. SC-IDC 奖励

首先定义基础语义质量：

\[
R_{\mathrm{sem}}(H)=S([H]_G,O),
\]

其中 \(S\) 可以先使用 Jaccard，或其他经过校准的 precision–recall 指标。

controlled slot 的有益反事实贡献为：

\[
\Delta_C(H)
=
R_{\mathrm{sem}}(H)
-
\mathbb E_{z_C'\sim q_{\mathrm{match}}}
R_{\mathrm{sem}}(H^{(C')}).
\]

定义有效控制奖励：

\[
R_{\mathrm{eff}}(H,C)
=
\mathbf 1[C\in H]\,
w(R_{\mathrm{sem}}(H))\,
\operatorname{clip}
\left(
\frac{\Delta_C(H)-\epsilon}{\tau},
0,1
\right).
\]

其中：

- \(\mathbf 1[C\in H]\) 保证 nominal adherence；
- \(\Delta_C>0\) 表明原控制条件优于匹配替代项；
- \(w(R_{\mathrm{sem}})\) 防止整体质量很差的假设仅凭局部差异获得高控制奖励；
- \(\epsilon\) 排除图执行噪声和极小变化；
- \(\tau\) 用于归一化和截断。

例如可以取：

\[
w(R_{\mathrm{sem}})=R_{\mathrm{sem}},
\]

最终奖励为：

\[
\boxed{
R(H,O,C)
=
R_{\mathrm{sem}}(H)
+
\beta R_{\mathrm{eff}}(H,C)
}
\]

这里不再对所有槽位求和。

---

### 6. 非受控槽位如何使用 IDC

对于其他槽位 \(z\neq z_C\)，仍然可以计算：

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
-\lambda
\frac1{|Z(H)|}
\sum_{z\in Z(H)}
\max(0,-\Delta_z),
\qquad
\lambda\ll\beta.
\]

这样：

- \(\Delta_z>0\)：有益，但不额外强迫其变得更大；
- \(\Delta_z=0\)：允许合理冗余；
- \(\Delta_z<0\)：说明替换该槽位反而能改善解释，给予轻微惩罚。

第一轮实验甚至可以完全不使用 \(R_{\mathrm{harm}}\)，只把非受控 IDC 作为诊断指标。

---

### 7. Nominal 与 Effective adherence 分开评估

SC-IDC 不应完全取代原来的条件遵循指标，而应增加一个更严格的维度。

建议同时报告：

1. **Nominal Adherence**

\[
\mathrm{Acc}_{\mathrm{nom}}
=
\Pr(C\text{ 出现在合法位置}).
\]

2. **Effective Adherence**

\[
\mathrm{Acc}_{\mathrm{eff}}
=
\Pr(\Delta_C>\epsilon).
\]

3. **Control Laundering Rate**

\[
\Pr(
\mathrm{nominal}=1
\land
\Delta_C\le\epsilon
).
\]

第三项专门衡量“条件虽然出现，却没有实际作用”的比例。

需要明确：\(\Delta_C=0\) 不代表假设错误。它只表示条件在当前 KG 上缺乏可识别的额外外延贡献。因此 nominal 和 effective 两种指标不能互相替代。

---

### 8. 最小验证实验

第一步可以不训练，直接在现有 rollout 上离线审计。

#### 实验 A：奖励信息量

比较原 reward 与加入 SC-IDC 后的：

- zero-reward-variance group rate；
- 不同字符串但相同 reward 的比例；
- unique denotation 数量；
- 每次 rollout 的额外执行成本。

SC-IDC 主要应该减少“不同假设、相同终局集合奖励”的 tie，无法解决四条 completion 完全相同的情况。

#### 实验 B：OR-append 对抗集

从一个已经很好解释 \(O\) 的假设开始，添加一个包含指定条件 \(C\)、但被其他 OR 分支完全遮蔽的分支。

预期：

- nominal adherence 仍为 1；
- 原 condition reward 仍为 1；
- \(\Delta_C\approx0\)；
- effective adherence 判为失败。

如果 SC-IDC 无法识别这类案例，方案的核心动机就不成立。

#### 实验 C：槽位位置泛化

分别用：

- first slot；
- non-first slot；
- 全部槽位均匀采样

作为 semantic control，检查原模型是否存在明显 first-slot advantage，以及均匀训练后差距是否缩小。

#### 实验 D：小规模训练

在相同 SFT parent 和相同图执行预算下比较：

- 原始奖励；
- 只保证 nominal control；
- nominal control + controlled-slot IDC；
- all-slot IDC，作为反例消融。

最后一个消融可以验证“全槽位不可替代性”是否确实导致语义或复杂度多样性下降。

---

### 9. 主要风险

- 匹配替代分布若不合理，\(\Delta_C\) 可能只反映 degree 或稀有度差异。
- 同义或外延等价关系可能让一个合理控制条件得到 \(\Delta_C\approx0\)。
- 当前有限 KG 上的冗余不代表完整图上的冗余。
- 多次反事实执行会增加训练成本，需要缓存或限制替代样本数。
- effective control 是比论文 lexical control 更强的任务定义，因此必须同时报告 nominal adherence，不能悄悄替换原评价口径。
- 当用户给出多个 semantic controls 时，应只对这些 controlled slots 计算联合或逐控制 IDC，仍然不扩展到全部槽位。

### 最终概括

SC-IDC 的核心不是追求一个“逐 predicate 最小”的假设，而是：

> 保留合理的逻辑冗余，但要求用户明确指定的语义条件不仅出现在假设中，而且对该假设解释观测的方式产生可验证的实际影响。

这比原始二元 condition reward 更接近“有效可控”，同时不会把整个 abductive objective 偷换成过强的逻辑最小化目标。
