# SC-IDC Phase 2 实施契约与接手记录

更新日期：2026-08-15

## 文档职责

本文只记录 Phase 2 的工程状态、依赖和验收 checklist。研究定义、公式、实验问题与报告口径以
`worklogs/SC-IDC/innovate_proposal.md` 为唯一来源；这里不复制另一套 A/B/C/D 实验命名。

本轮模型级范围仅为 `specific_relation`。截至本文更新时，Phase 2 尚未启动 SFT/GRPO；最近两次
提交只修订文档，没有实现下面列出的缺口。

## 最小依赖图

```text
数据/SFT 工作线                         reward 工作线
parent import                           value-level joint core
unique-value sampler                    matcher + reward adapter
frozen condition manifests              adversarial/regression tests
        │                                      │
        ▼                                      │
specific-relation conditional SFT              │
        └──────────────────┬───────────────────┘
                           ▼
                  Experiment 1 离线信号审计
                           │
                           ▼
                  Experiment 2 配对 GRPO
                           │
                           ▼
                 总体 + topology 分层报告
```

两条工作线可以并行。reward preflight 必须在 Experiment 1 前通过，但不人为阻断 conditional SFT；
数据/SFT preflight 只阻断 conditional SFT。

## 已冻结的任务契约

### 1. 控制单位

prompt 只提供 relation value \(C\)，不提供 AST path 或 slot role。因此：

- train 从目标假设的 unique relation values 中均匀采样；重复 occurrence 不增加采样权重；
- validation、signal、正式 RL train 和最终 evaluation 的
  `(record_id, condition_kind, condition_value)` 分别物化、冻结并 hash；
- 两个 GRPO 分支共享同一份 RL condition manifest；
- 需要控制具体 slot 时必须把 path/role 加入 prompt，那是另一个任务。

train condition 可由 `seed + record_id + epoch` 动态派生。所有用于 checkpoint 选择、跨分支比较或
最终报告的 condition 都必须固定；最终 evaluation/test manifest 在隔离步骤中生成和封存。

### 2. 同值 occurrences 联合干预

生成假设中 \(C\) 出现多次时：

- nominal adherence 判断至少存在一个合法 occurrence；
- matched intervention 把全部 \(C\) occurrences 同时替换为同一个 \(C'\)；
- branch intervention 联合中性化去重后的最近 `i/u` 分支；
- 嵌套分支只保留 ancestor-most path；任一 occurrence 无 `i/u` 祖先时使用 root marker；
- 逐 occurrence 结果只用于 attribution，不构成独立训练样本，也不求和进主 reward。

### 3. 论断边界

Phase 2 使用 `branch_nonmarginal`、`branch_supported_nonselective`、
`branch_supported_selective` 和 `unscorable`。Phase 1 的 `strict_effective`、`laundered`、
`marginal_only` 仅作为 legacy schema 保留。

自然 reference/model outputs 的 branch-nonmarginal 不得称为 laundering；laundering detection 只用于
有标签的 OR 遮蔽对抗集。branch marginal 与 matched delta 同时为正也不能声称 predicate necessity。

### 4. 主对照名称

对照组名为 `uniform-value control + original-reward baseline`，不是“原始 CtrlHGen baseline”，
因为原实现使用 fixed-first condition。它与 SC-IDC 分支共享 parent、manifests、训练配置和 seeds，
唯一方法差异是是否加入 SC-IDC reward。

## 当前实现缺口

下列事项均未完成：

1. `condition_value_from_target()` 仍是 fixed-first 行为；
2. Phase 1 SC-IDC core 仍以单 slot/path 为干预单位并输出 legacy 分类；
3. checkpoint loader 仍要求完整 semantic hash 一致，尚无显式 cross-condition parent import；
4. fresh-RL verifier 仍绑定 source config semantic hash，尚无 condition-agnostic rebind；
5. 尚无冻结的 Phase 2 condition manifests；
6. 尚无 compute-bounded relation matcher、value-level reward adapter 和 signal-audit runner；
7. 尚未把 SC-IDC reward 接入 GRPO；
8. 尚未运行 specific-relation conditional SFT 或任何 Phase 2 模型实验。

不得通过放宽现有通用 hash/checkpoint verifier 绕过第 3、4 项。

## 数据/SFT 工作线

### Data-1. 显式导入契约

实现 `import-unconditional-parent`：

- 验证 source stage/config、data/KG、model architecture、tokenizer 和 checkpoint SHA；
- 记录 target config 与导入命令；
- 重置 optimizer/scheduler；
- 任一 identity 不匹配时 fail closed。

实现 `condition-agnostic fresh-query rebind`：

- 验证 source artifact/sampling manifest、data/KG、record IDs、去重和 split 排除关系；
- 生成新的 target config 与 condition manifests，同时保留 source lineage；
- 不改写旧 artifact 或旧 manifest。

验收由合法导入测试和 source/target hash、checkpoint/artifact 篡改、错误 stage、tokenizer/model
不一致、split 污染拒绝测试组成。

### Data-2. Unique-value sampler 与 manifests

- train 使用可复现的动态 unique-value sampler；
- validation、signal、RL train 和最终 evaluation 分别物化 manifests；
- 重复 relation value 不按 occurrence 数过采样；
- 固定 seed 逐字节一致；
- 两个 GRPO 分支拒绝不同的 RL manifest hash；
- 训练与调参进程不能加载封存的最终 evaluation/test manifest。

### Data-3. conditional SFT

数据/SFT preflight 通过后即可启动，不必等待完整 reward reference audit。训练必须：

- 从显式导入的同一 unconditional parent 开始；
- 使用 uniform unique-value train sampler 和固定 validation manifest；
- 保存 parent、source/target config 和 condition lineage；
- 只用 validation 选择 checkpoint；
- 报告 loss、parse、EOS 和 nominal adherence，并验证 checkpoint 可重载。

冻结的 SFT checkpoint 是 Experiment 1 和两个 GRPO 分支的共同 parent。

## Reward 工作线

### Reward-1. Joint core 与兼容迁移

- 为 value-level joint intervention 增加独立 schema/API；
- 保持 Phase 1 audit 入口、legacy schema 和历史 fixtures 可复现；
- 实现 repeated-value joint replacement、joint branch neutralization、ancestor-most antichain 和
  root marker；
- 逐 occurrence attribution 不进入主 reward。

### Reward-2. Matcher 与 reward adapter

- 训练版 relation matcher 只用静态可缓存特征预筛，最终执行 \(K=3\) joint replacements；
- 同一 completion 的全部 \(C\) occurrences 共享每个 \(C'\)；
- base denotation 在基础 reward 与 SC-IDC 之间共享，其他执行按 canonical query hash 缓存；
- parse-fail、nominal-fail、unscorable 的 SC-IDC 附加项为 0；
- 记录 candidate table、fallback、scorable status、执行数与 wall-time。

### Reward-3. 必须通过的 preflight tests

- OR-append 遮蔽：单 occurrence 与 repeated value 均得到 joint branch marginal 0；
- nested redundancy：允许两个 delta 为正，但 schema 不产生 necessity claim；
- Phase 1 executor parity、结构保持与确定性 fixtures 不回归；
- joint replacement 保持 AST、pattern、relation/entity 数量和目标 occurrence 数量；
- unique-value weighting、fallback、root marker 和 antichain 正确；
- train-graph reference audit 不访问 test，结果可复现。

scorable coverage 是报告指标，不强制 reference 上达到 100%；若覆盖不足，应检查 matcher，而不是
自动放宽匹配条件。

## 两个模型实验

### Experiment 1：离线信号审计

必须同时具备冻结 SFT parent 和通过 preflight 的 reward core。按照 proposal 的 train-only signal
set 对同一批 completions 计算 base reward、raw SC-IDC 与候选 augmented rewards。

硬停止条件仅包括：不可复现、split 泄漏、少于 30 个 informative groups、raw SC-IDC 无法在任何
一部分 informative groups 中提供区别于 base reward 的 action-level ordering、出现实现导致的
Pareto-harmful inversion，或成本超出本轮预先声明的资源预算。base reward 存在 ties 时还必须报告
能解开多少；其他比例连同 bootstrap 区间报告，不使用武断的通用百分比门槛。
\(\alpha_{\mathrm{IDC}}\) 只使用该 train-only signal set 选择并冻结。

### Experiment 2：配对 GRPO

- baseline 与 SC-IDC 从同一冻结 SFT checkpoint 分叉；
- 共享 RL manifest、generation/optimizer 配置、每个 seed 的 update 数；
- 主结果 update-matched，并报告累计图执行数和 wall-time；
- compute-matched 曲线只在既有 checkpoints 足够时补充，不为制造三套视图强制追加训练；
- 先做单 seed pilot；正式稳定性结论原则上至少 3 seeds。若只有单 seed，明确标注 pilot。

first-only、non-first-only 和 repeated 是 SFT、Experiment 1、Experiment 2 共用的报告切片，不单列
第三个实验，不参与 checkpoint 或 reward 选择；support 少于 30 时只作描述统计。

## 下一步执行顺序

1. 实现并测试 Data-1/Data-2；完成后即可准备 conditional SFT。
2. 同时实现 Reward-1/Reward-2/Reward-3；先跑便宜的合成反例，再跑 train-graph reference audit。
3. Data-1/Data-2 通过后运行 specific-relation conditional SFT；不等待 Reward-3 的完整 reference audit。
4. 冻结 SFT parent；待 Reward-3 全部通过后运行 Experiment 1。
5. Experiment 1 显示存在可学习信号且成本可接受后，运行单 seed 配对 GRPO pilot。
6. pilot 稳定后再决定是否投入正式多 seed 训练。

## 不变量

- 原 pattern 复现的 config hash、checkpoint 和产物不变；
- 通用 verifier 不放宽；旧 Phase 1 产物不改写；
- 不覆盖已有 checkpoints、manifests 或 run directories；
- 不访问 final evaluation/test 调参；
- 所有正式训练记录精确 Git SHA、命令、seeds、输入/输出 hashes 和资源成本；
- entity、all-slot IDC 和 structural controls 不进入本轮模型结论。
