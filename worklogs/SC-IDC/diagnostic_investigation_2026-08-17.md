# SC-IDC 失败归因与两阶段诊断记录

记录日期：2026-08-17（Asia/Shanghai）  
诊断代码：`fc31d4895905cbecfca776470039f21bd864e9be`  
原 specific-relation SFT 代码：`af94eb41c2396854caa803d9f1362ab8a356e5b9`  
数据集/条件：WN18RR / `specific_relation`  
资源边界：单 NVIDIA L20；不追加多 seed，不打开 sealed final evaluation

## 1. 结论

本轮诊断不支持“SC-IDC reward 未接入”或“当前 tokenizer/label mask 存在致命错误”的解释。没有明确
增益的主要原因是三层因素叠加：

1. conditional SFT 确实学到了一定的条件关联，但没有把条件变成决定输出假设的强控制变量；换成
   reference 中不存在的关系后，约三分之二的输出仍保留原关系，真正只响应新条件的比例只有
   `8.05%--9.62%`。
2. 当前 BSS 奖励与 base semantic reward 高度相关并大量饱和，绝大多数 matched replacement 又因
   KG 稀疏得到空集合；因此高平均 `raw_bss` 并不转化为有效的组内 advantage。
3. signal audit 与 GRPO rollout 的 generation 配置不一致，validation BSS 又错误地在 train graph
   上计算。这两处实现契约问题会削弱 signal audit 的外推和机制指标可信度，但修正 graph split 后
   仍未改变“相对增量很小”的主结论。

因此不应在当前方法上直接重复完整 GRPO。优先级应是先重做可辨识的 conditional supervision，再
重构 matched intervention；两者通过直接诊断 gate 后，才运行一次短程、单 seed 的 paired pilot。

## 2. Epoch 10 checkpoint 核查与重建

原正式 checkpoint 根目录：

```text
/mnt/workspace/ctrlhgen-checkpoints/sc-idc-wn-specific-relation-full-sft-v1/
```

实际只保留 `conditional-epoch-30/45/50`，`conditional-best` 指向 epoch 30。原 epoch 10 未留存；
其他目录中同名 epoch 10 属于 pattern/早期 diagnostic 实验，不能用于本次 specific-relation 对比。

为了得到可比的 epoch 10，切回冻结的原始代码 SHA `af94eb4...`，使用完全相同的 parent、配置、
preflight、数据顺序、seed 和 32,500-step 调度，从头重放并在 epoch 10 checkpoint 与 history 完整
落盘后由外层监控终止。重建没有放宽 preflight 的 code-SHA gate。

```text
checkpoint:
/mnt/workspace/ctrlhgen-checkpoints/sc-idc-diagnostics-20260817-cf5f649/
  reconstructed-epoch10-original-code/
  sc-idc-wn-specific-relation-full-sft-v1/conditional-epoch-10

tree SHA256:
9b8759ebb61fcbeb8fcc54133418b6f5434a5b26edabd1c48c21f0a207f3ab0f
```

重建轨迹与原记录一致：

| epoch | train loss | val loss | condition acc. | Jaccard | Dice | Overlap |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.15391455 | 3.08622569 | 0.73257212 | 0.63877262 | 0.70085533 | 0.79298402 |
| 10 | 0.13649597 | 3.19796883 | 0.73918269 | 0.64405434 | 0.70566109 | 0.79746269 |

重建 checkpoint 必须继续标记为 diagnostic/reconstructed，不能改写旧 run 或冒充原先留存的文件。

## 3. 第一阶段：奖励与评估口径诊断

### 3.1 已修正的实现契约

- `build_grpo_config` 过去硬编码 `temperature=0.9, top_k=50`，而 Experiment 1 signal audit 使用配置中
  的 `temperature=1.0, top_k=0`。现已统一从 experiment config 读取。
- paired validation 的标准语义指标在累计 valid graph 上执行，但 BSS audit 过去传入 train graph；
  现已改为 valid graph。
- 增加只读 offline diagnostics 和 counterfactual condition audit；增加安全的
  `--stop-after-epoch`，但正式 epoch 10 重建仍使用原始 SHA 和外层监控，以保持 frozen preflight。

### 3.2 Experiment 1 冻结输出的离线重分析

输入仍是原 208 groups / 832 completions，没有重新采样。

| 指标 | 结果 |
|---|---:|
| base / augmented zero-variance group rate | 0.62019 / 0.59615 |
| zero-variance rate reduction | 0.02404 |
| pairwise ordering 改变的 groups | 0.04327 |
| 任意 normalized-advantage 数值改变的 groups | 0.29327 |
| group mean advantage L1 改变 `>0.01` | 0.04808 |
| group mean advantage L1 改变 `>0.05` / `>0.10` | 0.02404 / 0.02404 |
| mean group normalized-advantage L1 改变 | 0.02094 |
| base reward 与 raw BSS Pearson（nominal） | 0.71541 |
| branch / matched / 双 clip 饱和率（nominal） | 0.7024 / 0.7968 / 0.7024 |
| positive raw BSS 均值 | 0.87258 |
| replacement 空 denotation 比例 | 0.7600 |
| 三个 replacements 全空的 nominal outputs | 0.6816 |
| empty-root neutralization 比例 | 0.2576 |
| 同组使用多个 replacement sets 的比例 | 0.57059 |
| condition relation 重复出现比例（nominal） | 0.3616 |

这里“任意 advantage 改变”达到 29.3%，但绝大多数是极小的连续值变化；超过 `0.05` 的 group mean
L1 改变仍只有 2.4%，与 zero-variance reduction 相同。真正改变 pairwise ordering 的也只有 4.3%。

更严重的是 matched intervention 的辨识性：76% 的 replacement 查询为空，68.2% 的可评分输出
三个替代关系全部得到空集合。当前奖励大量把“替换关系后查询因图稀疏坍塌”当成 selectivity。
此外，57.1% 的可比较 groups 内部使用了多个 replacement sets；原因是采样 seed 包含生成 query
hash，这给本应共享 prompt 的四个 action 引入了额外比较噪声。

### 3.3 使用 valid graph 重算 BSS

标准 Jaccard/Dice/Overlap 与原报告一致；Smatch 因其内部匹配过程有极小运行波动，不用于本节判断。

| decode | branch | old train-graph raw BSS | valid-graph raw BSS | graph shift |
|---|---|---:|---:|---:|
| greedy | baseline | 0.393969 | 0.447491 | +0.053522 |
| greedy | SC-IDC | 0.395034 | 0.451339 | +0.056305 |
| sampled | baseline | 0.368546 | 0.418781 | +0.050235 |
| sampled | SC-IDC | 0.369777 | 0.421441 | +0.051664 |

在正确的 valid graph 下，SC-IDC 相对 baseline 的 raw BSS 增量为 greedy `+0.003848`、sampled
`+0.002661`；branch-supported rate 均为 `+0.004808`。绝对 BSS 对 graph split 很敏感，但方法相对
差异仍很小，因此评估 bug 没有掩盖一个大幅机制增益。

## 4. 第二阶段：Conditional SFT 反事实条件响应

### 4.1 审计定义

对冻结 validation 的全部 1,664 条记录：

1. assigned prompt 使用 manifest 中、确实存在于 reference 的关系；
2. counterfactual prompt 使用 static matcher 排名最高且不在 reference 中的关系；
3. 比较 reference 的 per-target-token mean log probability；
4. greedy 生成后比较条件遵从、输出 AST、valid-graph denotation 与语义指标。

counterfactual prompt 没有唯一 gold hypothesis，所以其 semantic score 只用于描述响应代价，不能当作
counterfactual 正确率。本审计没有读取 final evaluation manifest。

### 4.2 总体结果

| 指标 | epoch 10 | epoch 30 | epoch 50 |
|---|---:|---:|---:|
| assigned SemAvg | **0.71573** | 0.69697 | 0.68298 |
| assigned condition adherence | 0.73738 | **0.77825** | 0.77344 |
| absent condition adherence | 0.19231 | 0.21875 | 0.23017 |
| 换条件后仍保留原关系 | 0.67608 | 0.68510 | 0.66587 |
| 只遵从新条件且不保留原关系 | 0.08053 | 0.08534 | **0.09615** |
| exact prediction 改变 | 0.35036 | 0.39543 | 0.42308 |
| parsed AST 改变 | 0.34982 | 0.39398 | 0.42273 |
| valid denotation 改变 | 0.25300 | 0.27644 | 0.30950 |
| assigned-reference logp 减 absent-reference logp | 0.26900 | 0.40219 | 0.44394 |
| assigned reference 获得更高 logp 的比例 | 0.73317 | 0.74219 | 0.74519 |

这排除了“模型完全忽略 condition token”的极端假设：assigned prompt 下 reference log probability
明显更高，换条件也会让 35%--42% 的 greedy 输出发生变化。

但控制仍然很弱。epoch 30 换成 absent relation 后，68.5% 的输出保留原关系，只有 8.53% 做到新关系
出现且旧关系消失。epoch 10 到 epoch 30 的额外 20 epochs 损失 `1.876pp` SemAvg，只带来
`+0.481pp` counterfactual-only adherence；到 epoch 50 语义继续恶化，而干净条件切换仍不足 10%。

### 4.3 Topology 观察

- epoch 30 repeated：assigned adherence `0.8571`，但换条件后旧关系保留率 `0.7805`，
  counterfactual-only adherence 仅 `0.0663`。
- epoch 30 non-first-only：counterfactual-only adherence 相对最高，为 `0.1074`，但仍很低。
- epoch 10/30/50 的总体模式一致，不是某个 checkpoint 或 topology 的偶然异常。

这说明模型更多是在 observation 驱动下复现原 hypothesis，再把 condition 当作弱提示；在 repeated
样本上，高自然 condition accuracy 尤其容易由复现包含重复关系的原 hypothesis 获得，并不代表强控制。

## 5. 归因更新

### 已确认的主因

1. **SFT 可辨识性不足。** 同一 target 中动态抽不同关系却始终监督同一个 hypothesis，使忽略或弱用
   condition 成为低损失解。反事实审计证实模型有条件关联，但缺少干净切换能力。
2. **checkpoint 选择错配最终目标。** condition accuracy 词典序优先选择 epoch 30；它牺牲约
   `1.88pp` SemAvg，却只比 epoch 10 增加约 `0.48pp` 的 counterfactual-only response。
3. **BSS 新信息密度过低。** 真正较大的 normalized-advantage 改变只影响约 2.4% groups，且 reward
   与 base reward 的相关性为 0.715。
4. **matched intervention 主要受空集合和饱和支配。** 这削弱了“selectivity”的因果解释。
5. **组内 comparison 含不必要噪声。** 同一 prompt 的 completions 没有共享 replacement set。

### 已确认但不是主因的实现问题

- signal audit 与 GRPO generation 配置不一致；
- validation BSS 使用了错误的 train graph；
- 现有训练记录缺少逐步 KL、entropy、component reward std、有效 group rate 等诊断。

### 当前没有发现

- SC-IDC reward 未传给 trainer；
- 当前 tokenizer、SEP/EOS、label mask 仍有历史致命 bug；
- paired branches 的数据、步数或 parent 不一致。

## 6. 后续建议：不做多 seed 时的最小有效路线

### P0：不要继续运行当前 reward 的完整 GRPO

当前结果已经足以说明瓶颈是信号设计，不是统计 seed 数不足。追加同配方训练只会以约 `1.75x`
wall-time、`4.39x` graph executions 重复一个极弱干预。

### P1：先重做 conditional supervision

1. 为同一 observation 构造多个有效的 `(C, H_C)`，使不同关系条件对应不同 hypothesis；不能继续让
   多个条件总是共享一个 target。
2. 加入 prompt-swap/contrastive margin，例如要求
   `log p(H_C|O,C) > log p(H_C|O,C')`；可配 nominal-but-nonmarginal hard negative。
3. 每 5 epochs 同时报 SemAvg、assigned adherence、absent adherence、original-retention 和
   counterfactual-only adherence。
4. checkpoint 采用 Pareto 选择，不再以自然 condition accuracy 绝对优先。epoch 10 可作为当前旧
   parent 的语义较强基线，但不建议直接把它视作最终解决方案。

建议进入 RL 前至少达到：counterfactual-only response 明显高于当前约 9%，original-retention 明显
低于当前约 67%，同时 assigned SemAvg 相对最佳语义 checkpoint 不出现明显退化。具体数值门槛应在
新数据构造后预注册，不能用 sealed final set 选择。

### P2：重构 SC-IDC reward

1. replacement 改为 query-local hard matching，要求非空、answer-cardinality/难度接近，并报告 fallback；
2. 同一 `(record_id, condition)` 的整个 GRPO group 共用 replacement set；
3. 无 intersection/union 祖先的 root/chain occurrence 标记为 `unidentifiable`，不再把 empty root 当作
   branch-support 证据；
4. 去除 BSS 对 base Jaccard 的再次相乘，或对 base reward residualize，确保新增项能产生独立排序；
5. 对 relation 重复 occurrence 做归一化/上限，避免通过复制 condition 放大奖励；
6. 新 offline gate 至少检查：replacement empty rate、clip saturation、base correlation、pairwise ordering
   改变率，以及 normalized-advantage L1 超阈值的 group rate。

### P3：只做单 seed 的短程决策实验

资源约束下不安排多 seed。推荐顺序：

1. 新 SFT 先做 10--15 epochs 小规模训练与反事实审计；未通过直接停止。
2. 新 reward 只对冻结 rollouts 做离线重评分；若仍只有约 2%--5% groups 获得实质 advantage，停止。
3. 两个 gate 都通过后，做一次相同 parent/manifest/seed 的 500--1,000 update paired pilot：baseline 与
   revised SC-IDC。每 250 steps 评估 direct condition response、valid-graph BSS、SemAvg、KL、entropy、
   unique AST/denotation 和有效 group rate。
4. 只有直接控制指标产生清晰变化且 SemAvg 不退化，才考虑一次单 seed full run。

## 7. 代码验证与产物

相关测试：

```text
47 passed in 5.33s
```

权威诊断根目录：

```text
/mnt/workspace/ctrlhgen-runs/sc-idc-diagnostics-20260817-fc31d48/
```

| 产物 | SHA256 |
|---|---|
| `offline/summary.json` | `7b767579f33d5d84d92ac1d446886aa2df30bce447cde45f4283f4ac86eeccb1` |
| `condition-causality/summary.json` | `24ac5474146cc316bc80d72a46c6d88910f454027831c1ac5bce5601bc9ce5c9` |
| `condition-causality/conditional-epoch-10.jsonl` | `2ce4777a99f22dfa205b01b737f0874a1959910d4a7530d8fb9d69c319d23104` |
| `condition-causality/conditional-epoch-30.jsonl` | `f58c9d6cd209f322b241916ebf4ce5c5b304acb3802f2b01cc307f5654a0f8b6` |
| `condition-causality/conditional-epoch-50.jsonl` | `9417d8d42b26b708c86faab5ef31d37f5f4376b4b4e332117612b5d19aa04c51` |
| `valid-graph-reevaluation/validation-report.json` | `83d8d2b78858839458e15e5f67cc887ffde220c31e9bedd59a0bd51bc343a1ff` |

诊断结束时 DSW checkout 为 clean detached
`fc31d4895905cbecfca776470039f21bd864e9be`，没有 GPU compute process。sealed final evaluation 始终未打开。

## 8. DSW 停止记录

完成产物 hash、Git clean 状态和 GPU 空闲检查后，使用实例内 CredentialsURI 临时凭据和官方
`alibabacloud_pai_dsw20220101` SDK 调用 PAI DSW 2022-01-01 `StopInstance`。凭据仅在内存中交给
Credentials SDK，没有打印、写入 run 或仓库。响应如下：

```text
requested at: 2026-08-17 15:17:19 Asia/Shanghai
before status: Running
region: cn-beijing
HTTP status: 200
Success: true
Code: null
InstanceId: dsw-uhn5s45l2r8qw8n0f5
SaveImage: false
RequestId: 01A00E94-F817-5CA1-A923-BA679F83FB75
```

无敏感信息的响应已在停机前写入：

```text
/mnt/workspace/ctrlhgen-runs/sc-idc-diagnostics-20260817-fc31d48/
  stop-instance-response.json
```

API 成功返回后，从本地连续进行了两次
`ssh -o BatchMode=yes -o ConnectTimeout=5 ctrlhgen-dsw` 探测，两次均在连接阶段超时，确认实例停止
提供 SSH 服务。调用的是 StopInstance 而不是 DeleteInstance，`save_image=false`；持久化数据、run 与
checkpoint 保留。
