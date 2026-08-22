# Specific-relation selector 消融与独立图视角审计方案

日期：2026-08-22

状态：执行前冻结。两项实验均为 post-hoc diagnostic，不修改模型、checkpoint、历史 P2 结论或超参数。

## 1. 研究问题

本轮只回答两个问题：

1. P2 的收益究竟来自 K=4 候选本身、似然/语义筛选、nominal condition match，还是来自 branch-supported 这一创新判据？
2. selector 在一个图视角上做出的选择，换到没有参与选择的关系边视角后是否仍然成立，还是主要利用了同一图上的 verifier 偶合？

## 2. 输入冻结

离线 selector 消融只读取已经完成的 specific-relation P2：

- summary SHA256: 21ec03b56840a447cd93ae9bb027f0c2e10cdc5d41184d2427da144aeed1001e
- records SHA256: 6d6a159b1d4f5d0d4f76aaa7ef649c5bacd0fa32f00d4472b48d28054d81568d
- records: 1664
- 每条记录固定使用原有 K=4 candidates，不重新生成。

图视角审计使用 P2 所引用的同一个 P1 checkpoint、配置和冻结 validation manifest。随机种子固定为 161803，K=4，temperature=1、top-k=0、top-p=1。历史 final-evaluation manifest 不再次打开。

## 3. Offline selector ablation

六个 selector：

1. first_sample：固定第一个 sample。
2. likelihood_only：只按 mean log probability；hash 作最终 tie-break。
3. semantic_only：只按 Jaccard/Dice/Overlap 的原始等权均值；之后 likelihood、hash。
4. exact_semantic：先 exact，再 semantic；不读取 condition nominal/branch 信号。
5. nominal_aware：exact+nominal > exact > semantic；不读取 branch-supported。
6. branch_aware：历史规则 exact+branch-supported > exact+nominal > exact > semantic。

正式执行必须逐条验证 branch_aware 复现历史 sample_k4 的 selected index；任何 mismatch 都令任务失败。

主对比为 branch_aware - exact_semantic 和 branch_aware - nominal_aware。另报告对 semantic_only、likelihood_only、first_sample 的对比。

## 4. Independent graph-view anti-overfitting audit

WN18RR 当前 loader 的图不是互相独立的：train 是独占 train 边，valid 是 train 与独占 valid 边的并集。因此固定如下三视角：

- selection/train：只含 train edges。参考假设 H 在该图上的 denotation 作为生成 observation；candidate 生成、验证和 selector 冻结都只使用此视角。
- audit/valid-cumulative：train 加 valid-exclusive edges，用于测扩图后的稳定性。
- audit/valid-exclusive：只含 valid-exclusive edges，与 train edges 严格不相交；保留 cumulative-valid 的完整 node universe，避免 negation 因节点集合变化而混淆。

对每个冻结 validation reference hypothesis，在三个视角分别执行 H，得到视角特定 target。只有 train target 非空且不超过配置 max_answers 的记录进入生成。selection 完成后冻结 selected index，再把完全相同的 hypotheses 放到两个 audit 视角执行。

valid-exclusive 很稀疏，两个空集合可能产生表面 exact。因此同时报告：

- 所有 eligible records；
- valid-exclusive reference denotation 非空的子集。

## 5. 指标与统计

不使用 semavg 作为汇总结论。每个 selector、每个视角分别报告：

- Jaccard、Dice、Overlap；
- exact、nominal、branch-supported、nonroot-branch-supported；
- parse、EOS。

所有 selector 差值采用 paired bootstrap，固定 seed 271828、10,000 次 percentile 95% CI，并给出逐样本 win/tie/loss rate。每个 artifact 记录 count、SHA256、命令、Git SHA、图 edge hash、eligibility/filtering 和运行成本。

## 6. 解释边界

- 这是 validation-only 的事后诊断，不用于继续选择超参数，也不替换历史 P2 estimate。
- valid-exclusive 稳定性下降只能说明图视角敏感，不能单独证明模型过拟合。
- 稳定性较好也不是全图逻辑等价证明。
- 任何后续方法修改都必须另开新 proposal 和新 final-evaluation，而不能回写当前 P2。

## 7. 2026-08-22 执行后补充审计冻结

在不修改上述实验和结论的前提下，补做一个纯离线 candidate-availability audit，用于回答
generator proposal 与 selector capture 的责任分解。该补充不生成新 candidate，不重新打开其他数据，也
不用于重新选择 K 或方法。

输入仍由同一个 P2 summary 固定：

- P1 original validation records 使用嵌套 K=1/2/4/8；
- P2 final-evaluation records 使用同一 K=4 candidate stream 的 K=1/2/4 前缀；
- P1 是主要 availability-vs-K 曲线，P2 仅作 post-hoc confirmation。

新增 pure exact-branch selector：

- exact+branch-supported > exact > semantic；
- 它不含 nominal fallback，用于和 full lexicographic verifier 严格区分。

对每个 K 报告以下 candidate availability、proposal-failure、平均 qualifying candidate count：

- parse-ok、exact、nominal、branch-supported、nonroot-branch-supported；
- exact+nominal、exact+branch-supported、exact+nonroot-branch-supported。

对 first-sample、likelihood-only、semantic-only、exact-semantic、pure exact-branch、
nominal-aware、full branch-aware 分别报告：

- selected qualifying rate；
- 条件 capture rate，即 qualifying candidate 存在时 selector 选中 qualifying candidate 的比例；
- full verifier 相对 likelihood-only 和 exact-semantic 的 rescue/harm rate；
- selector disagreement rate；
- 所选 candidate 的 Jaccard、Dice、Overlap、exact、nominal、branch、nonroot、parse、EOS。

availability 的相邻 K 增量采用 paired bootstrap，seed 271828、10,000 次 percentile 95% CI。full verifier
对 exact+branch 的 capture 是规则构造结果，不能表述成学习能力；应主要解释 availability、baseline miss
和 rescue 数量。
