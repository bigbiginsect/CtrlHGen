# SC-IDC Phase 2 实施契约与接手记录

更新日期：2026-08-15

## 文档目的

本文记录 specific-relation 模型实验启动前必须实现的工程契约和验收顺序，供后续 agent 接手。
本轮只修订文档，没有修改训练、审计或奖励代码，也没有据此启动 SFT/GRPO。

Phase 2 的模型级范围仅为 `specific_relation`：

```text
工程契约与 value-level joint core
                 │
                 ▼
       Experiment B（硬门禁）
                 │
                 ▼
specific-relation conditional SFT
                 │
                 ▼
 Experiment A + C-pre（同批 rollout）
                 │
          A 通过 │ A 失败 → 停止
                 ▼
       Experiment D 配对 GRPO
                 │
                 ▼
       Experiment C-post 分析
```

字母不表示执行顺序：B 是机制门禁，A 是奖励信号门禁，C 是复用既有输出的分层诊断，D 是正式
配对 GRPO。C 不训练第三个模型。

不在本阶段声称 entity control、pattern control 或其他 structural controls 已获得模型级验证。

## 已冻结的任务定义

### 1. 控制单位是 unique relation value

prompt 只提供 relation value \(C\)，不提供 AST path 或 slot role。因此：

- SFT train 从目标假设的 unique relation values 中均匀采样；重复 occurrence 不增加该值的采样权重；
- validation、test、rollout signal set 和正式 RL train 的
  `(record_id, condition_kind, condition_value)` 必须物化、冻结并 hash；
- 两个 GRPO 分支必须共享同一份 RL prompt/condition manifest；
- 如果未来要控制具体 slot，必须把 path/role 加入 prompt，并作为另一个任务处理。

train condition 可以通过 `seed + record_id + epoch` 动态派生；所有用于 checkpoint 选择、跨分支
比较或最终报告的 condition 都必须固定。test condition manifest 由隔离的 split-preparation 步骤
生成并封存，训练、signal gate 和 checkpoint 选择不得加载。

### 2. 同值 occurrences 联合干预

若生成假设中 \(C\) 出现多次：

- nominal adherence 判断至少存在一个合法 occurrence；
- matched intervention 将全部 \(C\) occurrences 同时替换为同一个 \(C'\)；
- branch intervention 联合中性化各 occurrence 所属、去重后的最近 `i/u` 直接子分支；
- 嵌套待中性化 path 只保留 ancestor-most path；任一 occurrence 没有 `i/u` 祖先时，联合基线
  整体退化为空根查询；
- 逐 occurrence 结果只用于 attribution，不构成彼此独立的训练样本或 reward 求和项。

### 3. 指标只支持 branch-level 论断

后续代码与报告使用：

- `branch_nonmarginal`；
- `branch_supported_nonselective`；
- `branch_supported_selective`；
- `unscorable`。

Phase 1 的 `strict_effective`、`laundered`、`marginal_only` 是 legacy schema。自然 reference/model
outputs 的 branch-nonmarginal 比例不得称为 laundering；laundering detection 只用于已标注的 OR
遮蔽对抗集。即使 branch marginal 与 matched delta 都为正，也不能声称 predicate necessity。

### 4. 主对照的准确名称

从同一个 uniform-value specific-relation conditional SFT parent 分叉后，对照组名为：

`uniform-value control + original-reward baseline`

它不是原始 CtrlHGen baseline，因为原实现使用 fixed-first condition。两分支唯一的方法差异应是
是否加入 SC-IDC reward；prompt manifest、初始化权重、generation 参数、optimizer 和 seed 相同。

## 当前代码与 Phase 2 之间的缺口

以下是后续实现事项，不表示本轮已经完成：

1. `condition_value_from_target()` 仍是 fixed-first 行为；不得直接用于 Phase 2 unique-value 采样。
2. Phase 1 SC-IDC core 以单 slot/path 为干预单位，输出 legacy 分类字段；训练版需要 value-level
   joint intervention，同时保持旧审计入口和历史 schema 可复现。
3. 通用 checkpoint loader 要求完整 semantic hash 一致，不能直接把旧 unconditional parent 导入
   新 `specific_relation` config。
4. Phase 1 fresh-RL verifier 绑定 source config semantic hash，不能直接把 pattern artifact 当作新
   condition config 的 manifest。
5. 尚无冻结的 validation/RL condition manifests、训练版 compute-bounded matcher、signal-gate
   runner、Phase 2 value-view Experiment B、C-pre/C-post 汇总或 SC-IDC GRPO reward 接入。

不得为绕过第 3、4 项而放宽现有通用 hash 校验。

## 必须新增的显式导入契约

### import-unconditional-parent

允许跨 condition 导入冻结的 unconditional 权重，但必须验证并记录：

- source stage 与 source config hash；
- data hash 与 KG hash；
- model architecture/config identity；
- tokenizer identity；
- checkpoint 文件 SHA；
- target config hash 和导入命令。

导入后必须重置 optimizer/scheduler，不继承旧训练状态。失败时 fail closed，不回退到宽松加载。

### condition-agnostic fresh-query rebind

允许把已验证的 fresh query 数据重新绑定到 target condition config，但必须验证并记录：

- source artifact 与 sampling manifest SHA；
- data/KG identity；
- record IDs；
- query/supervision 去重；
- 与 validation/test 的 split 排除关系；
- source/target config hash；
- 新 condition manifest 的 SHA。

旧 manifest 不改写；rebind 生成新 manifest，并保留完整 lineage。

## 后续实现顺序与验收

### 1. 兼容的数据结构与术语迁移

- 为 value-level condition、joint intervention 和新分类增加独立版本/schema；
- 保持 Phase 1 audit 的旧入口、旧字段和历史 hash 可复现；
- 实现 repeated-value joint replacement、ancestor-most branch antichain 和 root-marker baseline。

验收：Phase 1 原测试和确定性 fixture 不变，新 joint-occurrence 与嵌套反例测试通过。

### 2. 实现两个导入契约

- 实现 `import-unconditional-parent`；
- 实现 fresh-query rebind；
- 覆盖 source/target hash、checkpoint/artifact 篡改、错误 stage、tokenizer/model 不一致和 split
  污染的拒绝测试。

验收：合法导入产生完整 lineage；任一 identity 不匹配都拒绝，现有通用 verifier 不变。

### 3. 冻结 condition manifests

- train 使用可复现的动态 unique-value sampler；
- validation、test、signal 和正式 RL train 分别物化 condition manifest；
- 两个 GRPO 分支只接受同一 RL manifest hash。

验收：重复 relation value 不被 occurrence 数量加权；固定 seed 逐字节一致；不同 split 无 identity
重合；checkpoint 选择始终使用固定 validation conditions；sealed test manifest 在最终评估前未被
训练或调参进程加载。

### 4. 运行 Experiment B 机制门禁

- B0：OR-append laundering 对抗集，同时覆盖单 occurrence 和重复 condition value；
- B1：嵌套冗余反例
  \(H=(A\land C)\lor D,\ [A\land C]_G=[A]_G\)，两个 delta 可正但不得输出 necessity claim；
- B2 legacy view：Phase 1 executor parity、结构保持与确定性 fixture 不变；
- B2 value view：按 unique relation values 加权，审计同值 occurrences 的联合替换与联合中性化。

验收：proposal 中 B0/B1/B2 的所有硬门槛通过。Phase 1 已有的 single-slot 结果只能作为 legacy
regression baseline，不能替代修改 joint core 后的 Experiment B。任一子项失败都不启动 SFT。

### 5. 运行 specific-relation conditional SFT

- 从显式导入的同一 unconditional parent 开始；
- 使用 unique-value train sampler 和固定 validation manifest；
- 保存 source/target config、parent checkpoint 和 condition manifest lineage。

验收：parse/EOS、nominal adherence 与训练 provenance 完整；SFT checkpoint 冻结后才允许生成
Experiment A rollout。

### 6. 实现并运行 Experiment A signal gate 与 C-pre

signal set 必须是 train-graph-only，且与 conditional SFT train、正式 RL train、validation/test 的
query/supervision identity 不重合。冻结：

- \(\tau_{\mathrm{marg}}=\tau_{\mathrm{match}}=0.1\)；
- \(\alpha_{\mathrm{IDC}}\in\{0.25,0.5,1.0\}\)，按 proposal 的预注册门槛选通过条件的最小值；
- 现有 GRPO KL 系数保持 \(\beta_{\mathrm{KL}}=0.1\)，不得与新增奖励系数复用字段或符号；
- 全部同值 occurrences 联合聚合；
- parse-fail、nominal-fail、unscorable 的 SC-IDC 附加项为 0；
- 13 个 pattern 各 16 个 prompts、每 prompt 4 generations，共 208 groups/832 completions；
- relation 静态特征排序后从最佳 8 个确定性采样 \(K=3\)，每个 scorable completion 最多 5 次
  总图执行；
- candidate seed 由 `(reward_seed, record_id, canonical_query_hash, condition_value)` 派生；同一
  condition value 的全部 occurrences 共享每个 replacement；
- base/neutral/replacement denotation 缓存规则和候选表 hash。

验收：nominal/scorable、非相同 completion reward-tie 降幅、semantic ranking inversion、执行成本
和确定性全部达到 proposal 门槛。若任何 \(\alpha_{\mathrm{IDC}}\) 都不通过，停止，不启动 GRPO。

C-pre 复用同一批 208 groups/832 completions，按 target condition value 的 first-only、
non-first-only、repeated occurrence topology 报告分层 support 和指标，不额外生成 rollout、不训练
模型，也不参与 \(\alpha_{\mathrm{IDC}}\) 选择。少于 30 prompts 的分层不作泛化结论。

### 7. 启动 Experiment D 配对 GRPO

Experiment A 通过后才从同一冻结 SFT checkpoint 分叉。报告三个互补视图：

- update-matched：相同 optimizer updates，SC-IDC 承担并报告额外成本；
- graph-execution-matched：按累计图执行数对齐最近的不超预算 checkpoint；
- wall-time-matched：按累计 wall-time 对齐最近的不超预算 checkpoint。

图执行数和 wall-time 分别成轴；不得声称一组运行天然同时满足“相同步数”“相同图执行数”和
“相同 wall-time”。

### 8. 运行 C-post 冻结分层分析

在 D 的两个分支上复用 C-pre 的 topology 定义与冻结 manifests，报告分层 nominal、branch
support、matched selectivity、branch-supported selectivity、branch-nonmarginal、语义质量和
复杂度。C-post 不选择 checkpoint、不修改 reward、不反向调参，也不训练第三个模型。

## 预计涉及的代码范围

后续 agent 应先定位实际调用链再编辑；预计至少涉及：

- condition 采样与实验配置加载；
- checkpoint import/lineage manifest；
- fresh query manifest 验证与 rebind；
- SC-IDC value-level joint core 和 reward adapter；
- conditional SFT、rollout signal gate 与 GRPO runner；
- 对应 unit/integration tests 和运行记录。

不要直接改写 Phase 1 产物或复用其 config hash 充当 target identity。

## 不变量与停止条件

- 原 pattern 复现的配置 hash、checkpoint 和产物不变；
- 通用 hash/checkpoint verifier 不放宽；
- 不访问 test split 调参，不把 test conditions 用于 checkpoint 选择；
- 不覆盖已有 checkpoint、manifest 或运行目录；
- 任何正式训练都记录精确 Git SHA、命令、seed、输入/输出 hashes 和资源成本；
- value-level joint core、cross-config lineage、condition manifest、Experiment B、conditional SFT
  provenance 或 Experiment A 任一硬门槛未通过时，停止在 GRPO 之前。
