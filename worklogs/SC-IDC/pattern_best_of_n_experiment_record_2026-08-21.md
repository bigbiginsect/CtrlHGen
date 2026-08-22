# Pattern Verifier-guided Best-of-N 实验记录（2026-08-21）

## 1. 结论

Pattern-P1 validation gate 全部通过，随后按冻结合同只运行了一次 Pattern-P2 historical test。P2 上
pattern-aware Best-of-4 相对 greedy：

- Jaccard：`0.60304065 -> 0.69056009`，差值 `+0.08751944`；
- Dice：`0.65172750 -> 0.73923262`，差值 `+0.08750512`；
- Overlap：`0.72159158 -> 0.81750388`，差值 `+0.09591230`；
- observable exact：`0.34495192 -> 0.40384615`，差值 `+0.05889423`，95% CI
  `[+0.04627404, +0.07151442]`；
- Pattern Accuracy：`0.94290865 -> 0.95432692`，差值 `+0.01141827`，95% CI
  `[+0.00360577, +0.01923077]`；
- SMATCH（evaluation only）：`0.81680792 -> 0.82941114`，差值 `+0.01260322`，95% CI
  `[+0.00883525, +0.01658537]`。

三项集合指标方向一致，并非某一个汇总数值或单一指标拉动：Jaccard 和 Dice 都提高约 `0.0875`，Overlap
提高约 `0.0959`。这说明选择后的答案集既减少了并集尺度上的错配，也改善了交集覆盖；但 Overlap 仍明显高于
Jaccard，说明“较小集合被覆盖”仍比“预测集合与观察集合完全一致”容易，不能把 containment 改善等同于逻辑
等价。

pattern-aware 与 exact-semantic 的 Jaccard、Dice、Overlap、exact 完全相同；pattern-aware 额外把 P2
Pattern Accuracy 提高 `+0.00661058`，95% CI `[+0.00300481, +0.01081731]`。因此主要增益来自
execution-semantic Best-of-N，pattern-aware tie-break 在不损失这三项集合指标的前提下提供了较小但可辨识的
控制增益，不能把全部提升归因于 pattern verifier。

本文后续分开使用各指标：Jaccard 是交集除以并集，对额外预测和漏预测都敏感；Dice 是
`2|P∩O|/(|P|+|O|)`，同样衡量集合一致性但对交集权重更高；Overlap 是交集除以较小集合大小，主要回答一侧是否
被另一侧覆盖，不能单独惩罚大范围 superset；observable exact 则要求执行答案集与 observation 完全相等。
当前实现的 Overlap 分母加了 `1e-5`，所以即使 exact，单条 Overlap 也可能略低于 1。PA 只检查 canonical
pattern adherence，SMATCH 只在选择完成后评估参考结构；二者都不等于 denotation correctness。

## 2. 实现、部署与冻结合同

- 实现提交：`86bd5a85a9da9df8cadb8b885287a6f5d98c5a24`
- DSW checkout：formal P1/P2 均为 clean detached 上述 SHA
- 新入口：`akgr/reproduction/pattern_verifier_best_of_n.py`
- 新测试：`tests/test_pattern_verifier_best_of_n.py`
- 定向测试：`15 passed in 6.15s`，运行时设置
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，用于隔离系统 Hydra/OmegaConf pytest plugin 冲突
- GPU：NVIDIA L20；PyTorch `2.6.0+cu124`；CUDA `12.4`
- seed：`314159`；bootstrap seed：`271828`；bootstrap：10,000 paired record resamples
- sampling：temperature `1.0`、top-k `0`、top-p `1.0`、batch size `32`
- nested candidates：P1 `K=1/2/4/8`；P2 冻结 `K=4`
- selector：exact+pattern > exact > 三项集合指标算术均值 > 同均值时 pattern tie-break > mean log-prob >
  canonical SHA256。这里的算术均值是预注册 selector 的内部排序量；下文不把它作为实验结果指标
- selection inputs 仅为 observation、observable graph、pattern condition 与 candidate；reference H 只在所有
  selector index 冻结后进入 SMATCH evaluation

冻结 lineage：

| 项目 | 值 |
|---|---|
| checkpoint pointer | `/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-full-train-author-aligned-c4/phase-d-parent` |
| resolved checkpoint | `conditional-epoch-45` |
| checkpoint tree SHA256 | `16b9f0dbd58e4eaad0ffc00c4d52a0d90522ddeaeb9d611d51f60d3b8679c99c` |
| pointer SHA256 | `b8ae1bdfe3e9161222645a38c02c9e21887189fabbfb7fdca2274145c27fd33a` |
| metadata SHA256 | `0e33477537a199122366c5b38ba9e07da9a8ba1afd82daea830e419cde8322d0` |
| config semantic hash | `d7978678398d197265c879913d834f79fc9a344201832168a6282a60d1aec296` |
| config file SHA256 | `2cc8689f13d242a0e15f5d28b0cbe97d722a5b7aa86a8cbe47ca87f15badeaaa` |
| sampling manifest SHA256 | `244ef264ad538df30ad58d72127d7fe3c31f52480dc3ddac882978400c49e460` |
| validation artifact SHA256 | `950dca912f2d602d32894562b6b17f856e350c0158d125bb01d784d51adeed75` |
| historical test artifact SHA256 | `38bf3b26367be27cf03689a1dd2b168a505024fde14ea065df4ec084c01929a3` |

## 3. Smoke 与异常记录

第一次 2-record smoke 在配置加载前 fail-closed，因为非交互 DSW shell 没有导出
`CTRLHGEN_DATA_ROOT`、`CTRLHGEN_CHECKPOINT_ROOT`、`CTRLHGEN_RUN_ROOT`。该次没有加载数据、checkpoint 或
GPU，也没有产生指标：

```text
/mnt/workspace/ctrlhgen-runs/pattern-best-of-n-smoke-20260821-86bd5a8/
```

补齐仓库规定的 persistent roots 后，在新目录执行正式 2-record GPU smoke，完整通过 greedy、nested K、五类
selector、graph execution、SMATCH 与 lineage 检查；records SHA256 为
`ab867281d84f49eea252857c8eedfd9330192274b45420125cc654bab34ba940`：

```text
/mnt/workspace/ctrlhgen-runs/pattern-best-of-n-smoke-v2-20260821-86bd5a8/
```

后台启动包装命令还出现一次 launcher PID 文件的 shell `&` 优先级竞态，但只读核对确认唯一正式 P1 进程
PID 4004 已正常启动，没有重复运行。runner 自身的 `control.json`、`status.json` 和 `run.log` 不受影响。

## 4. Pattern-P1 validation

正式目录：

```text
/mnt/workspace/ctrlhgen-runs/pattern-best-of-n-p1-20260821-86bd5a8/
```

greedy 与预登记 sanity 数值逐项吻合：Jaccard `0.61963729`、Dice `0.67458461`、Overlap
`0.74545728`、PA `0.95913462`、SMATCH `0.82834079`、parse `0.99939904`、EOS `1.0`，未发现
checkpoint/prompt/split 偏差。

集合语义指标：

| decoding / selector | Jaccard | Dice | Overlap |
|---|---:|---:|---:|
| greedy | 0.61963729 | 0.67458461 | 0.74545728 |
| sample K=1 / first | 0.57908114 | 0.62925807 | 0.69570535 |
| likelihood-only K=4 | 0.61700739 | 0.67155120 | 0.74401737 |
| semantic-only K=4 | 0.68831002 | 0.74329194 | 0.82132216 |
| exact-semantic K=4 | 0.68831002 | 0.74329194 | 0.82132216 |
| **pattern-aware K=4** | **0.68831002** | **0.74329194** | **0.82132216** |
| pattern-aware K=8 diagnostic | 0.72204155 | 0.77664624 | 0.85958243 |

控制、结构与生成健康度：

| decoding / selector | exact | PA | SMATCH | parse | EOS |
|---|---:|---:|---:|---:|---:|
| greedy | 0.34194712 | 0.95913462 | 0.82834079 | 0.99939904 | 1.00000000 |
| sample K=1 / first | 0.31550481 | 0.95853365 | 0.82407211 | 0.99759615 | 1.00000000 |
| sample K=2 / pattern-aware | 0.35516827 | 0.95913462 | 0.83000012 | 0.99939904 | 1.00000000 |
| likelihood-only K=4 | 0.34375000 | 0.95973558 | 0.82839388 | 0.99939904 | 1.00000000 |
| semantic-only K=4 | 0.38822115 | 0.95312500 | 0.83266137 | 0.99939904 | 1.00000000 |
| exact-semantic K=4 | 0.38822115 | 0.95312500 | 0.83266137 | 0.99939904 | 1.00000000 |
| **pattern-aware K=4** | **0.38822115** | **0.96274038** | **0.83458979** | **0.99939904** | **1.00000000** |
| pattern-aware K=8 diagnostic | 0.41526442 | 0.96754808 | 0.83854176 | 1.00000000 | 1.00000000 |

K=4 coverage：any exact `0.38822115`、any pattern-match `0.97956731`、any exact+pattern
`0.38161058`。

P1 的 pattern-aware K=4 相对 greedy：Jaccard `+0.06867273`、Dice `+0.06870733`、Overlap
`+0.07586488`、exact `+0.04627404`。相对 likelihood-only K=4：Jaccard `+0.07130263`、Dice
`+0.07174074`、Overlap `+0.07730479`。相对 exact-semantic，三项集合指标与 exact 均不变，PA 则从
`0.95312500` 提高到 `0.96274038`。

P1 paired bootstrap：

| comparison | metric | mean delta | 95% CI |
|---|---|---:|---:|
| pattern-aware - greedy | exact | +0.04627404 | [+0.03485577, +0.05829327] |
| pattern-aware - greedy | PA | +0.00360577 | [-0.00300481, +0.01081731] |
| pattern-aware - greedy | SMATCH | +0.00624900 | [+0.00332662, +0.00918920] |
| pattern-aware - exact-semantic | PA | +0.00961538 | [+0.00540865, +0.01442308] |
| pattern-aware - likelihood-only | PA | +0.00300481 | [-0.00420673, +0.00961538] |

正式 runner 按预注册方案只对三项集合指标的算术均值做了 paired bootstrap，没有为 Jaccard、Dice、Overlap
分别保存 bootstrap CI。因此本记录只报告三项各自的点估计与差值，不用原来的均值 CI 代替分项 CI。exact、
PA、SMATCH 的 CI 是正式 artifact 中直接保存的结果。若以后补做统计，应从冻结 `records.jsonl` 对三项分别
bootstrap，并明确标记为 post-hoc statistical reanalysis，不能重跑 P2 或改 selector。

Gate 全过：预注册的三项均值与 exact 不低于 greedy；PA/parse/EOS 在 `-0.005` 容差内；相对
exact-semantic 的 PA 不降，三项均值不低于 `-0.005`；K=4 成本齐全；selector reference-free；
lineage/artifact hashes 齐全。因此允许一次性进入 P2，没有根据 P1 修改方法、K、seed 或 selector。这里保留
gate 的历史合同表述，但不把三项均值用作结果解读。

成本：greedy `1,664` sequences、`11,549` tokens、`1,663` graph executions、runner-accounted wall
`6.93s`；K=4 `6,656` sequences、`46,200` tokens、`6,642` executions、`15.01s`；K=8
`13,312` sequences、`92,383` tokens、`13,289` executions、`25.98s`。P1 端到端 wall
`32.27s`。

## 5. Pattern-P2 historical-test comparison

正式目录：

```text
/mnt/workspace/ctrlhgen-runs/pattern-best-of-n-p2-20260821-86bd5a8/
```

P2 由 passed P1 summary 恢复并校验 clean Git SHA、checkpoint/pointer/tree/config/data/manifest hashes、seed、
K、batch size、max tokens、method version、selector、P1 summary/records hashes。P1 consumption marker SHA256：
`8161fd2cf5580c013634cb8bee1195f04ce1318f8885a46d4b95abb859d30fa4`。未运行第二次 test。

greedy 同样逐项吻合预登记 sanity：Jaccard `0.60304065`、Dice `0.65172750`、Overlap `0.72159158`、
PA `0.94290865`、SMATCH `0.81680792`、parse `0.99158654`、EOS `1.0`。

| decoding / selector | Jaccard | Dice | Overlap | exact | PA | SMATCH | parse | EOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 0.60304065 | 0.65172750 | 0.72159158 | 0.34495192 | 0.94290865 | 0.81680792 | 0.99158654 | 1.00000000 |
| first-sample K=1 | 0.56497923 | 0.60782050 | 0.66816456 | 0.32211538 | 0.94711538 | 0.81283705 | 0.99158654 | 1.00000000 |
| pattern-aware K=2 | 0.63774017 | 0.68455360 | 0.75571003 | 0.36778846 | 0.94831731 | 0.82158444 | 0.99399038 | 1.00000000 |
| likelihood-only K=4 | 0.61175413 | 0.66019124 | 0.73044779 | 0.35276442 | 0.94350962 | 0.81913480 | 0.99278846 | 1.00000000 |
| semantic-only K=4 | 0.69056009 | 0.73923262 | 0.81750388 | 0.40384615 | 0.94771635 | 0.82785443 | 0.99579327 | 1.00000000 |
| exact-semantic K=4 | 0.69056009 | 0.73923262 | 0.81750388 | 0.40384615 | 0.94771635 | 0.82785443 | 0.99579327 | 1.00000000 |
| **pattern-aware K=4** | **0.69056009** | **0.73923262** | **0.81750388** | **0.40384615** | **0.95432692** | **0.82941114** | **0.99699519** | **1.00000000** |

K=4 coverage：any exact `0.40384615`、any pattern-match `0.96694712`、any exact+pattern
`0.39663462`。

P2 的分项差值如下：

| comparison | Jaccard | Dice | Overlap | exact | PA | SMATCH | parse |
|---|---:|---:|---:|---:|---:|---:|---:|
| pattern-aware - greedy | +0.08751944 | +0.08750512 | +0.09591230 | +0.05889423 | +0.01141827 | +0.01260322 | +0.00540865 |
| pattern-aware - first-sample | +0.12558086 | +0.13141212 | +0.14933932 | +0.08173077 | +0.00721154 | +0.01657409 | +0.00540865 |
| pattern-aware - likelihood-only | +0.07880596 | +0.07904138 | +0.08705609 | +0.05108173 | +0.01081730 | +0.01027634 | +0.00420673 |
| pattern-aware - exact-semantic | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | +0.00661057 | +0.00155671 | +0.00120192 |

其中 pattern-aware - greedy 的 exact 95% CI 为 `[+0.04627404, +0.07151442]`，PA 95% CI 为
`[+0.00360577, +0.01923077]`，SMATCH 95% CI 为 `[+0.00883525, +0.01658537]`。pattern-aware -
likelihood-only 的 PA 为 `+0.01081731`，95% CI `[+0.00420673, +0.01742788]`。与 P1 相同，正式
artifact 没有保存 Jaccard、Dice、Overlap 的分项 bootstrap CI，故不作伪精确补写。

成本：greedy `1,664` sequences、`11,518` tokens、`1,650` graph executions、runner-accounted wall
`5.16s`；K=4 `6,656` sequences、`46,073` tokens、`6,602` executions、`13.12s`。P2 端到端 wall
`19.43s`。

逐 pattern 正式 summary 含 Jaccard、Dice、Overlap 分项数据，但本次已登记记录最初只摘录了三项均值和 PA，
且 DSW 已按要求停止。为避免从均值反推各分项，本修订不再声称“13 个 pattern 的集合指标均提高”。现有可直接
核验的 slice 结论仅是：PA 在 `pi`（`0.867188 -> 0.851562`）和 `inp`
（`1.000000 -> 0.992188`）有描述性下降，其余 slice 持平或提高。若以后从冻结 P2 `summary.json` 补录逐
pattern 的三项指标，这属于文档补全而不是新实验；不得据此调参、重跑 P2 或作 13 个未校正显著性声明。

```text
historical_test_previously_accessed = true
post_evaluation_tuning_permitted = false
```

## 6. Artifacts 与 hashes

| artifact | count | SHA256 |
|---|---:|---|
| P1 `summary.json` | 1 | `cec9a2a3e9c9d4c8f63691e1e365dad629435ee6d244f0100edea4110b154cd3` |
| P1 `records.jsonl` | 1,664 | `656408652d88cde617f08a11cf85640a00f9a6cece9eb21cd7f9daa40dc402eb` |
| P2 `summary.json` | 1 | `a46bbcb5bf69941345209e1f5c6c41404b65db33029e6f89b4a02f1e4ee5bb93` |
| P2 `records.jsonl` | 1,664 | `723f143c08ee3eb656b19c8dd8cbb75b3458a95711b4d0aedfa64d697674b28f` |

每个正式目录均包含 `status.json`、`summary.json`、`records.jsonl`、`run.log`、`control.json`。P1/P2
完成后 L20 均回到 1 MiB、utilization 0%，没有遗留 pattern verifier、训练或 torchrun 进程。

## 7. 解释边界

结果支持：在既有 WN18RR conditional-pattern SFT generator 上，冻结的 verifier-guided Best-of-4 decoding
同时提高 Jaccard、Dice、Overlap 与 observable exact，并保持/提高整体 pattern adherence、parse 和 EOS；结合
既有 specific-relation 结果，这一 inference selection 思路已覆盖两种 condition kinds。

结果不支持新的 generator training improvement、新 sealed test 首次泛化、完整 KG 逻辑等价、多数据集普适性，
也不能忽略约 4 倍 candidate generation/execution 成本。historical test 以前已被 Phase C frozen test 与 epoch
45/50 bake-off 使用，本轮不允许 post-evaluation tuning 或第二次 P2。

## 8. DSW 收尾

实验记录提交 `697168ce58cf6fe8b8df2cec30aeae57ed167838` 已 push，并在停机前让 DSW clean detached 到该
SHA。最终确认 L20 显存 1 MiB、utilization 0%，无 pattern verifier、训练或 torchrun 进程。

随后从 PID 1 读取标准 `ALIBABA_CLOUD_CREDENTIALS_URI`，只在进程内存中交给官方 credentials SDK；URI、
access key、secret 与 security token 均未打印或写入产物。先用官方 `GetInstance` 确认实例为 `Running`，再用
PAI DSW 2022-01-01 SDK 调用 `StopInstance(save_image=false)`。脱敏响应保存在：

```text
/mnt/workspace/ctrlhgen-runs/pattern-best-of-n-p2-20260821-86bd5a8/stop-instance-response.json
```

```json
{
  "api": "PAI DSW 2022-01-01 StopInstance",
  "before_status": "Running",
  "code": null,
  "deployed_git_sha": "697168ce58cf6fe8b8df2cec30aeae57ed167838",
  "http_status": null,
  "instance_id": "dsw-uhn5s45l2r8qw8n0f5",
  "message": null,
  "preflight_only": false,
  "region": "cn-beijing",
  "request_id": "01A02497-B743-5375-A9A2-6127A04F17E0",
  "requested_at": "2026-08-21T13:51:57.758759+00:00",
  "save_image": false,
  "success": true
}
```

API 成功后等待 15 秒，再以 8 秒连接窗口尝试 SSH，端口 1024 超时，确认实例已停止提供 SSH 服务。调用的是
StopInstance 而不是 DeleteInstance；persistent data、checkpoints 与 runs 均未删除。
