# SC-IDC Post-GRPO 条件因果与数据可辨识性诊断

记录日期：2026-08-17（Asia/Shanghai）  
权威诊断代码：`98180ee261f940f9927b85bca140b17e0e146837`  
数据集/条件：WN18RR / `specific_relation`  
资源：单 NVIDIA L20；没有训练、optimizer update 或 sealed final evaluation 访问

## 1. 结论

本轮补齐了此前只覆盖 SFT checkpoints 的反事实条件审计，并直接比较：

1. specific-relation SFT `conditional-epoch-30`；
2. original-reward baseline GRPO terminal；
3. SC-IDC GRPO terminal。

结论需要分成两层：

- **通用 GRPO 有效。** original-reward baseline 相对 SFT parent 不仅将 assigned SemAvg 提高
  `+0.02732`，也显著提高 assigned/absent condition adherence、counterfactual-only response、
  AST change 和 denotation change。此前“conditional SFT 可辨识性弱”仍成立，但它不是
  specific-relation 整条链路无法响应 condition 的充分解释。
- **SC-IDC 没有形成稳健的增量行为控制。** 相对 original-reward baseline，SC-IDC 的 assigned
  reference preferred rate 提高 `+2.524pp`，但 absent adherence 只提高 `+0.601pp` 且 CI 跨 0，
  counterfactual-only response 提高 `+0.901pp` 且区间下界为 0；与此同时 prompt swap 后 exact/AST
  change 分别显著下降约 `2.52/2.58pp`。结合此前 BSS 只改变 `4.3%` groups 排序、replacement
  empty rate `76%` 的结果，不支持继续当前 SC-IDC reward 的多 seed 或完整 GRPO。

本轮同时发现一条不需要从零进行 query search 的后续路线：现有 104,000 条 SFT base train 中已有
`9,253` 个 exact-same-observation groups，能够找到两条不同 hypothesis，且双方各自包含对方没有的
relation；占 base train unique observations 的 `14.01%`。在 208-group reference pilot 中，
`62.98%` pairs 的两侧 controlled branch 都有正边际。因此后续应优先把这些天然 solution sets
重组为可辨识的 paired supervision，而不是继续 reward shaping。

## 2. 为什么需要补这个诊断

2026-08-17 较早的 condition-causality audit 只比较 SFT epoch 10/30/50。它证明 condition token
不是被完全忽略，但没有回答两个正式 GRPO 分支是否改变了 condition 的行为因果作用。

第一次完整实验只报告了 assigned conditions 下的 semantic、nominal 与 BSS 指标。SC-IDC 相对 baseline
的 greedy SemAvg 为 `+0.004574`，但这不足以排除如下可能：SC-IDC 的主要收益不在平均语义，而在
prompt swap 后更强的条件响应。因此本轮固定同一 validation、同一 matcher 和同一 greedy decode，
对三个 checkpoints 做逐记录配对审计。

pattern 复现不重新训练。已有 repaired pattern GRPO 在 validation 和唯一 frozen-test 上均得到严格
为正的集合语义增益，已经证明模型、fresh RL-only 数据、`SEP` prompt 契约和 GRPO pipeline 可用。
本轮不再扩张为新的 pattern 实验。

## 3. 诊断工具修复与验证

原 `sc_idc_condition_causality.py` 严格写死 `expected_stage="conditional"`，因此第一次尝试在模型加载前
拒绝正确标记为 `stage="grpo"` 的正式 checkpoints。修复后：

- 显式允许 `conditional` 与 `grpo`；
- 先读取 metadata 并拒绝其他 stage；
- 仍严格校验 `condition=specific_relation`、config hash、data manifest、model 和 tokenizer；
- summary 新增实际 checkpoint stage。

DSW 定向测试：

```text
12 passed in 2.80s
```

第一次无运行根目录变量的 smoke、以及修复前的 GRPO-stage verifier failure 都发生在正式模型生成前，
保留在非权威目录：

```text
/mnt/workspace/ctrlhgen-runs/sc-idc-postgrpo-causality-20260817-fc31d48/
```

权威目录只使用 clean detached `98180ee...`：

```text
/mnt/workspace/ctrlhgen-runs/sc-idc-postgrpo-causality-20260817-98180ee/
```

## 4. Post-GRPO 条件因果审计

### 4.1 固定输入

| 模型 | checkpoint | tree SHA256 |
|---|---|---|
| SFT parent | `conditional-epoch-30` | `8f172569dbbc0d71409a19f3579663eac53aa266be191277cc4da5f1422c5dda` |
| baseline GRPO | `baseline/evaluation-step-12963` | `b8062047f235eb0af02c38b879ce2afe49e704a901001f934fed0158e96ae414` |
| SC-IDC GRPO | `sc_idc/evaluation-step-12963` | `1495776f5746e9a963bd332368a5bb2853e38a37fefafe5ae49201b645940a40` |

三者共享 1,664 条冻结 validation records。assigned condition 来自原冻结 manifest；counterfactual
condition 是 static matcher 排名最高且不在 reference 中的 relation。该 counterfactual prompt 没有唯一
gold hypothesis，因此 absent semantic 和 response 只作行为诊断，不称为正确率。所有集合指标使用
valid graph，未加载 final-evaluation manifest。

### 4.2 总体结果

| 指标 | SFT parent | baseline GRPO | SC-IDC GRPO |
|---|---:|---:|---:|
| assigned SemAvg | 0.696966 | 0.724287 | 0.728862 |
| absent SemAvg | 0.691534 | 0.710009 | 0.711440 |
| assigned adherence | 0.778245 | 0.818510 | 0.815505 |
| absent adherence | 0.218750 | 0.278245 | 0.284255 |
| counterfactual-only response | 0.085337 | 0.112380 | 0.121394 |
| 换条件后仍保留原 relation | 0.685096 | 0.671274 | 0.664663 |
| exact prediction change | 0.395433 | 0.489784 | 0.464543 |
| parsed AST change | 0.395433 | 0.489784 | 0.463942 |
| denotation change | 0.276442 | 0.334135 | 0.317909 |
| assigned-reference logp margin | 0.402194 | 0.468457 | 0.473743 |
| assigned reference preferred | 0.742188 | 0.745793 | 0.771034 |

这里的 `counterfactual-only` 指新 relation 出现且原 assigned relation 不再出现。它是严格诊断，不是
任务正确率；一个合理 hypothesis 可以同时含有新旧 relation。

### 4.3 SFT parent 到 original-reward baseline

10,000 次 record-level paired bootstrap，seed 42：

| 指标 | delta | paired 95% CI |
|---|---:|---:|
| assigned SemAvg | +0.027322 | [+0.017218, +0.037642] |
| assigned adherence | +0.040264 | [+0.028245, +0.052284] |
| absent adherence | +0.059495 | [+0.046274, +0.073317] |
| counterfactual-only response | +0.027043 | [+0.017428, +0.037260] |
| exact prediction change | +0.094351 | [+0.070913, +0.117788] |
| parsed AST change | +0.094351 | [+0.070313, +0.118990] |
| denotation change | +0.057692 | [+0.034856, +0.081130] |
| 原 relation retention | -0.013822 | [-0.026442, -0.001202] |
| reference logp margin | +0.066264 | [+0.057136, +0.075521] |

这说明 original reward 的 GRPO 不只是恢复语义；它也显著增强了 prompt swap 的行为响应。因此当前
specific-relation SFT 的弱可辨识性是真问题，但“必须先重做 SFT，否则任何 RL 都不会响应条件”过强。

### 4.4 Baseline 到 SC-IDC 的真正增量

| 指标 | delta | paired 95% CI |
|---|---:|---:|
| assigned SemAvg | +0.004574 | [-0.004918, +0.014345] |
| assigned adherence | -0.003005 | [-0.013221, +0.007212] |
| absent adherence | +0.006010 | [-0.006010, +0.018630] |
| counterfactual-only response | +0.009014 | [0.000000, +0.018029] |
| exact prediction change | -0.025240 | [-0.048077, -0.001803] |
| parsed AST change | -0.025841 | [-0.049279, -0.002404] |
| denotation change | -0.016226 | [-0.039063, +0.006611] |
| 原 relation retention | -0.006611 | [-0.018029, +0.004808] |
| reference logp margin | +0.005286 | [-0.002301, +0.012946] |
| assigned reference preferred | +0.025240 | [+0.012019, +0.038462] |

SC-IDC 唯一清楚为正的是“assigned reference 相对 absent prompt 更常获得较高 log probability”。但 mean
margin、absent adherence、counterfactual-only response、denotation response 和 semantic delta 都没有
稳健的正增量；exact/AST change 反而显著下降。因此不能把 likelihood preference 的小幅改善表述为
有效行为控制成功。

baseline 与 SC-IDC 在 assigned/absent prompt 下生成完全相同文本的比例分别为 `67.91%/63.88%`；
两模型确实不同，但差异没有沿 SC-IDC 目标形成一致的行为优势。

## 5. Train-only 数据可辨识性审计

### 5.1 定义

只读取 SFT base train 与 repaired fresh RL train，不读取 validation/test/final evaluation：

- observation identity：排序后的 exact answer entity set；
- hypothesis identity：完整 raw query token tuple；
- relation set：query 中 unique relation values；
- mutually exclusive relation evidence：同一 observation 的两个 hypotheses，其 relation-set difference
  在两个方向都非空。由此可以选出
  (C_1\in R(H_1)\setminus R(H_2)) 与
  (C_2\in R(H_2)\setminus R(H_1))。

### 5.2 结果

| 数据 | records | unique O | distinct-H groups | distinct-relation-set groups | 双向 exclusive groups |
|---|---:|---:|---:|---:|---:|
| SFT base train | 104,000 | 66,053 | 19,917 (30.15%) | 16,285 (24.65%) | **9,253 (14.01%)** |
| fresh RL train | 104,000 | 66,764 | 21,003 (31.46%) | 17,549 (26.29%) | **10,361 (15.52%)** |
| 合并只读审计 | 208,000 | 104,498 | 40,931 (39.17%) | 34,336 (32.86%) | **21,602 (20.67%)** |

合并统计只证明自然多解覆盖，不授权直接把已冻结的 fresh RL pool 并入新 SFT；后续必须重新定义数据
分区和隔离。即使完全不使用 fresh pool，base train 的 9,253 groups 已足以支持小规模 paired pilot。

## 6. 208-group 天然多解 reference pilot

### 6.1 选择规则

从 SFT base train 的 9,253 个 eligible groups 中，按 exact observation SHA256 排序取 208 个。每组按
record ID 排序，取第一对双方 relation-set difference 都非空的 hypotheses；每侧 condition 取只存在于
本 hypothesis 的最小 shifted relation token。

这不是模型 rollout，而是验证现有数据能否构造合法、有效的 paired supervision。

### 6.2 结果

| 指标 | 结果 |
|---|---:|
| observation pairs / hypothesis rows | 208 / 416 |
| reference exact | 416/416 |
| condition 确实不在配对 hypothesis 中 | 416/416 |
| branch-supported rate | 0.754808 |
| matched-selective rate | 0.951923 |
| branch-supported-selective rate | 0.754808 |
| 两侧均 branch-supported 的 pairs | **0.629808** |
| replacement empty rate | **0.700321** |
| graph requests / executions | 2,080 / 1,964 |

`62.98%` pairs 在没有优化 condition 选择规则时，两侧都已有正 branch marginal。正式构造可以在每侧
多个 exclusive relations 中选择 branch-supported value，覆盖预计还可提高；必须在 full train-only
audit 后冻结实际数量。

但 replacement empty rate 仍为 `70.03%`。高 matched-selective rate 因而不能作为数据质量证明，
也不应继续进入训练 reward。天然 pair 的可靠条件是 exact same observation、condition exclusivity 和
branch marginal；matched replacement 只保留诊断。

## 7. 更新后的失败归因

### 保留

1. 原 SFT 中同一 hypothesis 的多个 relation conditions 共享 target，使忽略 condition 成为低损失解。
2. 当前 BSS 与 base reward 高相关、饱和且被空 replacement 支配。
3. 同组不同 replacement sets 和旧 validation graph split 是真实实现问题。

### 修正

1. SFT 弱可辨识性不是 generic GRPO 无法工作的充分原因；original-reward baseline 已显著改善语义和
   直接条件响应。
2. `~60%` zero-variance groups 本身不是禁止 GRPO 的门槛；repaired pattern GRPO 在相近零方差率下
   有稳定收益。对新增 reward，关键是它相对 base 的独立 ordering 和 advantage 改变量。
3. SC-IDC v1 的核心失败应表述为：**缺少独立于 original reward 的增量行为信号**，而不是 reward
   未接入或整个 specific-relation pipeline 失效。

## 8. 决策

- 不补当前 SC-IDC v1 的额外 seeds，不运行当前 reward 的新 full GRPO。
- 不打开 sealed final evaluation。
- 不从头开始 condition-fixed query search；先使用 base train 的天然 exact-observation solution sets。
- 下一阶段先做 paired-data full audit、manifest 和 contrastive SFT 小规模 pilot；SC-IDC executor 作为
  数据过滤与评价工具，不作为在线 reward。
- 新方法与实验设计见 `revised_proposal.md`。

## 9. 权威产物

根目录：

```text
/mnt/workspace/ctrlhgen-runs/sc-idc-postgrpo-causality-20260817-98180ee/
```

| artifact | SHA256 |
|---|---|
| `parent/summary.json` | `28163f47318e269d788a026eff479a8fb0f4e6c4f25af80200cc9291913c38d9` |
| `parent/conditional-epoch-30.jsonl` | `f58c9d6cd209f322b241916ebf4ce5c5b304acb3802f2b01cc307f5654a0f8b6` |
| `baseline/summary.json` | `64d687bd688eb5cb8d824046c2efae4ae916bd0c4f5e7e07fe6e92be28736993` |
| `baseline/evaluation-step-12963.jsonl` | `00076c9bfbb02a67d2e269792fe4659f3bb33c98554b1a22d7935223af0d52d1` |
| `sc_idc/summary.json` | `e8fbfcf4ae868924fe6eaa43c7f62d0fd6e6eb56d0525be0c0b1cf7dbc75cd15` |
| `sc_idc/evaluation-step-12963.jsonl` | `3531023a22fc0b0d878621b5534d324326807ca59806d183fc65d0ad1ac4581f` |
| `paired-analysis.json` | `cb2a036784df229d778cac3e6231833069f40b6a363841635d4b621344ec9d13` |
| `data-identifiability.json` | `935a3759f48c29141bf1676dd8a22b56421c581c9454efeecce5e495141b88eb` |
| `natural-pairs-reference-pilot-summary.json` | `684b7b278cb86f6cae79f42a01a7323727d92c188bf72d4d0483151c28e1985b` |
| `natural-pairs-reference-pilot.jsonl` | `6cd155ebaef4a36ee41c0182c53e594a6136c2040c97a64c4f15ba3b25fb6428` |

## 10. DSW 收尾

停机前核验：

- checkout 为 clean detached `e8201274690cdc444afc490640b85c82994429f2`；
- 本文档和 `revised_proposal.md` 已部署；
- 权威 summary、JSONL、paired analysis 和 data audit 全部存在且已计算 hash；
- NVIDIA L20 显存占用 1 MiB、utilization 0%，没有 GPU compute process；
- 没有 SC-IDC、GRPO、SFT 或 experiment-runner Python 进程。

随后读取实例内 Alibaba Cloud CLI profile 的 `CredentialsURI` 字段，只在内存中交给 Credentials SDK，
使用官方 `alibabacloud_pai_dsw20220101` SDK 调用 PAI DSW 2022-01-01
`StopInstance(save_image=false)`。URI、临时 access key 和 security token 均未打印或写入产物。

无敏感信息的响应已先写入权威 run 的 `stop-instance-response.json`：

```json
{
  "api": "PAI DSW 2022-01-01 StopInstance",
  "before_status": "Running",
  "code": null,
  "deployed_git_sha": "e8201274690cdc444afc490640b85c82994429f2",
  "http_status": 200,
  "instance_id": "dsw-uhn5s45l2r8qw8n0f5",
  "message": null,
  "region": "cn-beijing",
  "request_id": "01A00FDD-8823-529A-9F28-3EA2ABCBCD0D",
  "requested_at": "2026-08-17T21:16:12.107903+08:00",
  "save_image": false,
  "success": true
}
```

API 返回后再次尝试 SSH，端口 1024 在 8 秒连接窗口内超时，确认实例已不再提供 SSH 服务。调用的是
StopInstance 而不是 DeleteInstance；持久化数据、checkpoints 和 run 未删除。
