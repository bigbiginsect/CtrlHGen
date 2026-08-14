# Phase D repaired terminal：唯一一次 frozen-test

> 状态日期：2026-08-12（Asia/Shanghai）
>
> 评测代码：`3e60d37d7baa57d214f118a4e6ca37b2abe87e55`
>
> DSW 目录：`/mnt/workspace/ctrlhgen-runs/phase-d-repaired-frozen-test-20260812`

## 1. 冻结边界

在打开 test 之前冻结：

- parent：C-IV conditional epoch 45，经 `phase-d-parent` pointer；
- repaired checkpoint：full repaired Phase D `checkpoint-13000`；
- terminal model SHA256：
  `674e6dae7e24fb891f4e6b2db6bd3c221ef0c1f1f8d71f8fabfc6aaa50a3eba2`；
- split：固定 1,664 条 test；
- decoding：各执行一次 greedy 和 sampled；
- 通过规则：两种解码的 Jaccard/Dice/Overlap 三项均值 delta 均为正，同时 candidate
  parse/EOS≥0.98、condition≥0.90；
- test 之后禁止继续训练或选择其他 checkpoint。

评测入口要求输出目录不存在、checkpoint 名为 `checkpoint-13000`、trainer global step 为
13,000，且模型 SHA 与冻结值完全一致，否则在 test evaluation 前失败。精确 commit 上全套
测试 122/122 通过后才运行。

## 2. Frozen-test 结果

### Greedy

| 指标 | SFT parent | repaired terminal | delta |
|---|---:|---:|---:|
| Jaccard | 0.60304 | 0.63667 | +0.03363 |
| Dice | 0.65173 | 0.68702 | +0.03529 |
| Overlap | 0.72159 | 0.76142 | +0.03983 |
| 三项均值 | 0.65879 | 0.69504 | **+0.03625** |
| Pattern Accuracy | 0.94291 | 0.95433 | +0.01142 |
| Smatch | 0.81676 | 0.82006 | +0.00330 |
| Parse / EOS | 0.99159 / 1.00000 | 0.99219 / 1.00000 | +0.00060 / 0 |

三项均值 paired bootstrap 10,000 次 95% CI：`[0.02506, 0.04773]`。

### Sampled

| 指标 | SFT parent | repaired terminal | delta |
|---|---:|---:|---:|
| Jaccard | 0.55997 | 0.60490 | +0.04493 |
| Dice | 0.60575 | 0.65023 | +0.04448 |
| Overlap | 0.67287 | 0.71625 | +0.04338 |
| 三项均值 | 0.61286 | 0.65713 | **+0.04426** |
| Pattern Accuracy | 0.94351 | 0.95132 | +0.00781 |
| Smatch | 0.81270 | 0.81681 | +0.00411 |
| Parse / EOS | 0.99038 / 1.00000 | 0.99399 / 1.00000 | +0.00361 / 0 |

三项均值 paired bootstrap 10,000 次 95% CI：`[0.02864, 0.05991]`。

预冻结规则全部满足，自动结论为 `pass`。

## 3. 相对旧 Phase D terminal

旧 Phase D test terminal 使用复用 SFT train、缺失 raw prompt SEP 的原始流程。repaired
terminal 相对它的 test 三项均值增益为：

| decoding | ΔJaccard | ΔDice | ΔOverlap | 三项均值 delta | paired 95% CI |
|---|---:|---:|---:|---:|---:|
| greedy | +0.02347 | +0.02544 | +0.03004 | **+0.02632** | [0.01418, 0.03870] |
| sampled | +0.03868 | +0.03868 | +0.03643 | **+0.03793** | [0.02114, 0.05478] |

因此 repaired 方案不仅相对 SFT parent 有显著正增益，也显著优于旧 Phase D terminal。
greedy Smatch 相对旧 terminal 为 `-0.00090`、parse 为 `-0.00300`，但相对冻结的 SFT
parent 两者均为正，且所有健康 gate 通过；这不改变预冻结 semantic 主结论。

## 4. 独立审计

- parent/repaired × greedy/sampled 四份 JSONL 各 1,664 个唯一 record；
- JSONL 独立重算与 metrics JSON 在 `1e-12` 内一致；
- `result.json` 记录的 10 个输入/输出 artifact 的 size 与 SHA 全部重新核验通过；
- 评测完成后 GPU 回到空闲；
- 没有训练、resume 或 checkpoint 选择操作。

`result.json` SHA256：
`5c3c904e09f2add4ac579c1e0a8ed2942cc7163f27c9a87c0c5d5b21b419a3ea`。

## 5. 最终判断

Phase D repaired 方法在 validation 与唯一一次 frozen-test 上均得到一致、区间严格为正的
semantic 增益。证据支持：原 Phase D 的主要失败来自高度拟合的 SFT 数据复用造成 GRPO
组内 advantage 稀疏，以及 raw generation prompt 缺失 `SEP` 的契约错位；6 层模型容量
不是 GRPO 低增益的主要解释。

至此不应再对 repaired 6-layer terminal 做训练或 checkpoint 筛选。若继续研究 12 层，必须
作为新的容量/论文忠实度实验，不能利用本 test 结果调参后再声称同一个 frozen-test。
