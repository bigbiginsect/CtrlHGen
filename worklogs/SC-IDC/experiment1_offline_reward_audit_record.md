# SC-IDC Experiment 1 离线奖励信号审计与 GRPO 前交接记录

更新时间：2026-08-16（Asia/Shanghai）

## 1. 结论与边界

Experiment 1 已完成，正式判定为 **go to GRPO preparation**。本轮在冻结的
specific-relation SFT parent 上完成 208 groups / 832 completions 的 train-graph-only rollout，
对每条 completion 同时计算原基础奖励、value-level joint SC-IDC 与候选 augmented rewards。

全部硬门禁通过，`alpha_IDC` 已仅基于 Signal set 冻结为 `0.05`。本轮没有启动 GRPO，没有读取
sealed final-evaluation manifest，也没有用 validation/test 选择 reward 或系数。当前进度停在
Experiment 2 配对 GRPO 之前。

这里的正信号只称为 `branch_supported_selectivity`，不声称 controlled predicate necessity；自然
输出中的 `branch_nonmarginal` 也不命名为 laundering。

## 2. 冻结代码、parent 与输入

正式实验代码提交：

```text
dd7285b6807b990f6b4b42ea8a7a90358805477b
```

DSW checkout 在运行时 detached 到该 SHA，工作区 clean。正式 parent：

```text
/mnt/workspace/ctrlhgen-checkpoints/sc-idc-wn-specific-relation-full-sft-v1/conditional-best
```

它解析到 `conditional-epoch-30`，checkpoint tree SHA256 为：

```text
8f172569dbbc0d71409a19f3579663eac53aa266be191277cc4da5f1422c5dda
```

冻结输入：

| 对象 | 数量 / SHA256 |
| --- | --- |
| Signal conditions | 208；`47774daa4bf3a034af4ab58ef0a39b13159958d65d8f2ccfb8a0ce35213e8914` |
| RL-train condition manifest | 103,711；`17bbd14d8b33787433122cca3ff1cc62b878cda1ba384ee86977fe1a09273122` |
| target config semantic hash | `15a4c9e859e22e457fa7e9be0e7e8f43c464268a5588b37f4797259fa51a2db8` |
| KG hash | `ab2a80eabd45a86cc1dad1acfc65c20b36a451070737b054e903bb2025dda4a9` |

Signal set 是 13 patterns 各 16 条，与 conditional SFT train、validation、正式 RL train 和 sealed
final evaluation 的 query/supervision identity 不重合。审计入口只打开 Signal manifest 和其
train-only fresh source；`final_evaluation_manifest_loaded=false`。

## 3. Reward-1 / Reward-2 / Reward-3 实现

本轮新增：

- value-level occurrences、同值 occurrences 联合 replacement；
- 最近 `i/u` 分支联合 neutralization、ancestor-most antichain 与 root marker；
- 只使用 train graph 静态特征的 relation matcher：方向 fallback、edge-frequency log-bin、
  head/tail endpoint-profile cosine、token ID 排序；从最优 8 个确定性无放回取 `K=3`；
- canonical AST 执行缓存、基础 denotation 共享、execution/time accounting；
- Phase 2 四分类与 `R_BSS` reward adapter；
- Experiment 1 rollout、reference preflight、独立 reward rescore、bootstrap、alpha 冻结与
  go/no-go runner。

合成/合同测试覆盖 single/repeated OR 遮蔽、joint replacement、antichain、root marker、结构保持、
确定性 matcher、Phase 1 executor parity 与全部既有回归。DSW 完整 CPU/合同套件结果：

```text
150 passed, 2 deselected in 9.63s
```

train-graph reference preflight 对 208 条 reference 全部 exact、全部 scorable、全部 replacement
结构保持；分类为 191 条 `branch_supported_selective`、17 条 `branch_nonmarginal`。未产生
predicate-necessity 论断。

## 4. 正式 run 与命令

权威 run：

```text
/mnt/workspace/ctrlhgen-runs/sc-idc-experiment1-formal-20260816-2128-dd7285b-a/
```

正式命令（所需环境变量已在 launcher 中指向外部持久化目录）：

```bash
bash scripts/sc-idc/experiment-1-signal-audit.sh \
  /mnt/workspace/CtrlHGen/akgr/configs/reproduce/wn-specific-relation-full-train-author-aligned.yml \
  /mnt/workspace/ctrlhgen-runs/sc-idc-specific-relation-sft-preflight-20260815-af94eb4-a/sft-preflight.json \
  /mnt/workspace/ctrlhgen-checkpoints/sc-idc-wn-specific-relation-full-sft-v1/conditional-best \
  /mnt/workspace/ctrlhgen-runs/sc-idc-experiment1-formal-20260816-2128-dd7285b-a/artifacts
```

首次目录
`/mnt/workspace/ctrlhgen-runs/sc-idc-experiment1-formal-20260816-2126-dd7285b/` 在模型加载前因
launcher 未注入三个 `CTRLHGEN_*_ROOT` 环境变量而 fail-closed。它没有 rollout、GPU 计算或正式
产物，保留作诊断，不是实验输入。权威 run 使用新目录，没有续跑或覆盖失败目录。

## 5. 主要结果

| 指标 | 结果 |
| --- | ---: |
| prompt groups / completions | 208 / 832 |
| parse / EOS | 1.0000 / 1.0000 |
| nominal / scorable | 0.7512 / 0.7512 |
| informative groups | 100 |
| unique completion strings / executable AST / denotation | 455 / 455 / 293 |
| `branch_supported_selective` | 446 |
| `branch_nonmarginal` | 179 |
| raw `R_BSS` positive rate / mean | 0.5361 / 0.46775 |
| informative groups with raw BSS variation | 50 |
| informative groups with ordering distinct from base | 9 |
| base zero-reward-variance group rate | 0.62019 |
| `alpha_IDC=0.05` 后 zero-variance rate | 0.59615 |
| group-rate reduction，bootstrap 95% CI | 0.02404；[0.00481, 0.04808] |
| non-identical base reward tie pairs | 305 / 600 |
| 被 SC-IDC 解开的 base ties | 26 / 305（0.08525） |
| Pareto-harmful inversions | 0 |

所有正 `alpha` 网格点 `{0.05, 0.1, 0.25, 0.5, 1.0}` 都解开同样 26 个 ties，且没有 strict base
ordering reversal 或 Pareto-harmful inversion。预注册选择规则先最大化 tie resolution，再最小化
strict base-order reversal，最后取最小系数，因此冻结 `alpha_IDC=0.05`。

topology 描述统计：

| topology | groups | nominal/scorable | mean raw BSS |
| --- | ---: | ---: | ---: |
| first-only | 83 | 0.74096 | 0.44163 |
| non-first-only | 65 | 0.65385 | 0.37658 |
| repeated | 60 | 0.87083 | 0.60266 |

三个 slice support 都不少于 30，但本轮仍只把它们作为既有 rollout 的描述性分层，不用于调参。

## 6. 成本、复现与产物

总墙钟 14.58 秒，其中 rollout 0.73 秒、首次 reward scoring 3.47 秒。reward graph requests 为
3,332，真实 executions 为 1,703，cache hit rate 为 0.48890；completion reward wall-time p50/p95
为 0.00028 / 0.00775 秒。均低于预声明的一张 L20、3,600 秒、每 completion 最多 5 次图执行预算。

对相同 rollout 使用全新 matcher/cache 做独立二次 reward 复算，去除观测性 wall-time 字段后的
逐行 canonical SHA256 两次一致：

```text
a2cfb6d857a4adcce0246dedaf4ce158d9e7e50bbdb8bfa6e2f6191052806187
```

权威产物：

| 文件 | SHA256 |
| --- | --- |
| `rollouts.jsonl` | `77fdee31e6f38268ac82c997c24334fa935d3d9705ceb40f276663c06f88d029` |
| `completion-audit.jsonl` | `0fcb55bf31161e7022e944641d35a293fa1044c1f908178b7dad8026aaaea019` |
| `summary.json` | `39ca829a9f25e91b23daa92ec23e18a9b18aaa98b70be746c34b56ca803ba403` |
| `alpha-freeze.json` | `bb5ec5874d5439dfab3728ccd14ca35bdab051b3bfd8e483ad73f8a21fd5381f` |

`manifest.json` 中记录了精确命令、代码/配置/KG/parent/Signal/RL identities 与上述 artifact hashes；
终态已逐个重新计算并验证。`alpha-freeze.json` 明确记录 `grpo_started=false`。

## 7. Go/no-go 审阅

以下硬门禁全部通过：

- Signal/reward rows 可复现；
- train-only isolation 成立，sealed final evaluation 未加载；
- informative groups 为 100，超过 30；
- raw SC-IDC 在 9 个 informative groups 提供区别于 base 的 action ordering；
- 原有 base ties 中有 26 对被解开；
- 0 个实现导致的 Pareto-harmful inversion；
- 实测图执行与 wall-time 在预算内。

因此可以进入 **GRPO preparation**，但这不是 GRPO 已经有效的结论，也不是跨 seed 的模型收益
结论。Experiment 2 仍应先做单 seed 配对 pilot。

## 8. GRPO 接手时必须保持的不变量

1. 两个分支必须共同从上述 `conditional-epoch-30` 分叉。
2. 两个分支必须共享 RL manifest SHA
   `17bbd14d8b33787433122cca3ff1cc62b878cda1ba384ee86977fe1a09273122`、generation/optimizer
   配置、seed 和 update 数。
3. baseline 准确命名为 `uniform-value control + original-reward baseline`；唯一方法差异是 SC-IDC
   分支加入 `0.05 * R_BSS`。
4. 先做单 seed pilot；正式稳定性结论原则上至少 3 seeds。主比较 update-matched，同时报告图执行
   数与 wall-time。
5. GRPO/选择阶段禁止打开 final evaluation manifest；只有终态选择冻结后才能进入 sealed final
   evaluation。
6. 当前 core 已提供 reward adapter，但尚未启动 Experiment 2 或生成任何 GRPO checkpoint。

## 9. DSW 停止记录

实验、artifact hash 复核与本文首版提交完成后，按阿里云 PAI DSW 2022-01-01 官方
`StopInstance` API 停止实例 `dsw-uhn5s45l2r8qw8n0f5`，`SaveImage=false`。最终 API 返回和
SSH 失联核验将在停机完成后补入本文的后续提交；实例不会删除，持久化 run/checkpoint 保留。
