# SC-IDC Experiment 2 配对 GRPO 实验记录

记录日期：2026-08-17（Asia/Shanghai）

## 1. 结论摘要

Experiment 2 已完成一对 `seed=42`、update-matched 的 full-train GRPO。两分支从同一个
specific-relation conditional SFT checkpoint 分叉，使用相同的 103,711 条冻结 RL
prompt/condition、相同 generation/optimizer 配置和相同 12,963 个 optimizer updates。唯一方法差异是
SC-IDC 分支在原基础 reward 上加入训练前已冻结的 `0.05 * R_BSS`。

冻结 validation（1,664 条）上的方向是**小幅正向但不确定**：SC-IDC 相对 baseline 的
Jaccard/Dice/Overlap 平均增量为 greedy `+0.004574`、sampled `+0.003636`，对应 paired bootstrap
95% CI 分别为 `[-0.005069, +0.013982]` 与 `[-0.011177, +0.018064]`，均跨 0。整体
branch-supported-selectivity rate 增量为 greedy `+0.003005`、sampled `+0.003606`；nominal
adherence 增量为 greedy `-0.003005`、sampled `+0.004207`。parse/EOS 没有坍塌。

因此本轮判定为 **inconclusive small positive single-seed pilot**，不是稳定增益结论，也不据此打开
sealed final evaluation。正式模型结论仍需要预注册的至少 3 个独立 seeds。SC-IDC 训练 wall-time
是 baseline 的 `1.754x`，图执行数是 `4.389x`；当前小幅、CI 跨 0 的 validation 增量还不足以证明
这项额外成本有稳定收益。

## 2. 身份、隔离与权威路径

| 项目 | 值 |
|---|---|
| training code SHA | `46008be251da1dae8c09d47a65d16e1a2628e99a`（clean detached checkout） |
| validation-only code SHA | `9e1434b00953e6a75c667bbe34ea1a781a6a5948`（clean detached checkout） |
| dataset / condition | `WN18RR` / `specific_relation` |
| seed / reward seed | `42` / `42` |
| alpha_IDC | `0.05`，由 Experiment 1 train-only signal set 预先冻结 |
| authoritative run | `/mnt/workspace/ctrlhgen-runs/sc-idc-experiment2-formal-20260816-2243-46008be-b` |
| checkpoint root | `/mnt/workspace/ctrlhgen-checkpoints/sc-idc-experiment2-formal-20260816-2243-46008be-b` |
| common parent | `/mnt/workspace/ctrlhgen-checkpoints/sc-idc-wn-specific-relation-full-sft-v1/conditional-epoch-30` |
| parent tree SHA256 | `8f172569dbbc0d71409a19f3579663eac53aa266be191277cc4da5f1422c5dda` |
| RL manifest SHA256 | `17bbd14d8b33787433122cca3ff1cc62b878cda1ba384ee86977fe1a09273122` |
| RL condition artifact SHA256 | `53dc2dd17842a0bc3f93a09203e2cb9558bb3c447f1425d0cb12fcdcca615899` |
| fresh source artifact SHA256 | `6263a2d0ece6d897e850dbcaf71893865cb038e198dbd95b576d8b1609ad9260` |
| alpha-freeze SHA256 | `bb5ec5874d5439dfab3728ccd14ca35bdab051b3bfd8e483ad73f8a21fd5381f` |
| validation run | `.../validation-9e1434b-c` |
| validation manifest SHA256 | `58ddb2c3359ac49a6a8153331b89299268df2324ef6308a53aff2bf04da446ac` |
| validation condition artifact SHA256 | `0bcbf68219c6329cff591fc3a6f596b552569b0faea814129ac98b9b3a5b71a4` |
| validation report SHA256 | `4170acb07511fab794b46b534ebd99c65e3acb8dfb975c898f599db1efdaa23a` |

训练、checkpoint metadata、终评报告均记录
`final_evaluation_manifest_loaded=false`。终评入口没有 final-evaluation 参数；本轮只读取 purpose 为
`validation`、consumer 为 `reporting` 的冻结 manifest。sealed final-evaluation manifest 和 artifact
均未打开。

## 3. 配对契约与命令

共同配置为 1 epoch、batch size 32、4 generations、LR `1e-5`、KL beta `0.1`、clip epsilon
`0.2`、最大 completion 33 tokens。基础 reward 是
`jaccard + 0.5*dice + 0.5*overlap + nominal`。baseline 的准确名称是
`uniform-value control + original-reward baseline`，不能称为原始 CtrlHGen baseline；SC-IDC 分支只
增加 `alpha_IDC=0.05` 的 joint value-level branch-supported-selectivity reward。

两分支由 supervisor 在单张 NVIDIA L20 上顺序执行。每个 branch manifest 保存了展开后的命令；核心
形式如下（`BRANCH` 依次为 `baseline`、`sc_idc`）：

```bash
python -m akgr.reproduction.sc_idc_grpo \
  --branch BRANCH \
  --experiment-config akgr/configs/reproduce/wn-specific-relation-full-train-author-aligned.yml \
  --phase2-preflight /mnt/workspace/ctrlhgen-runs/sc-idc-specific-relation-sft-preflight-20260815-af94eb4-a/sft-preflight.json \
  --alpha-freeze /mnt/workspace/ctrlhgen-runs/sc-idc-experiment1-formal-20260816-2128-dd7285b-a/artifacts/alpha-freeze.json \
  --parent-checkpoint /mnt/workspace/ctrlhgen-checkpoints/sc-idc-wn-specific-relation-full-sft-v1/conditional-epoch-30 \
  --run-dir /mnt/workspace/ctrlhgen-runs/sc-idc-experiment2-formal-20260816-2243-46008be-b/BRANCH \
  --checkpoint-dir /mnt/workspace/ctrlhgen-checkpoints/sc-idc-experiment2-formal-20260816-2243-46008be-b/BRANCH \
  --max-steps -1
```

终态比较使用新增加的 validation-only 入口；sampled 解码在两个 branch 前都重置到相同派生 seed：

```bash
python -m akgr.reproduction.sc_idc_grpo_evaluate \
  --experiment-config akgr/configs/reproduce/wn-specific-relation-full-train-author-aligned.yml \
  --phase2-preflight /mnt/workspace/ctrlhgen-runs/sc-idc-specific-relation-sft-preflight-20260815-af94eb4-a/sft-preflight.json \
  --pair-run-dir /mnt/workspace/ctrlhgen-runs/sc-idc-experiment2-formal-20260816-2243-46008be-b \
  --baseline-checkpoint /mnt/workspace/ctrlhgen-checkpoints/sc-idc-experiment2-formal-20260816-2243-46008be-b/baseline/evaluation-step-12963 \
  --sc-idc-checkpoint /mnt/workspace/ctrlhgen-checkpoints/sc-idc-experiment2-formal-20260816-2243-46008be-b/sc_idc/evaluation-step-12963 \
  --output-dir /mnt/workspace/ctrlhgen-runs/sc-idc-experiment2-formal-20260816-2243-46008be-b/validation-9e1434b-c
```

## 4. 训练终态与健康检查

| 指标 | baseline | SC-IDC |
|---|---:|---:|
| dataset count | 103,711 | 103,711 |
| global step / updates | 12,963 | 12,963 |
| completions | 414,816 | 414,816 |
| train runtime（trainer） | 1,544.51 s | 2,710.69 s |
| reward adapter elapsed | 1,547.84 s | 2,715.23 s |
| train loss | 0.036687 | 0.043722 |
| terminal parse rate | 0.999296 | 0.999352 |
| terminal nominal rate | 0.789856 | 0.791582 |
| mean raw BSS | 0 | 0.491595 |
| graph requests | 414,524 | 1,727,991 |
| graph executions | 183,345 | 804,788 |
| cache hits | 231,179 | 923,203 |
| cache evictions | 133,345 | 754,788 |

两分支状态均为 `completed`，supervisor 写入 `active-branch.txt=completed`。baseline 于
2026-08-16 23:06:27、SC-IDC 于 23:52:01（Asia/Shanghai）完成。训练日志未发现 traceback、CUDA
OOM、NaN/Inf 或磁盘耗尽；完成时 `/mnt/workspace` 使用 46%，GPU 空闲。两者 evaluation checkpoint
都由 `load_reproduction_checkpoint(mode="test")` 严格重载成功，且验证了 stage、condition、config、
sampling manifest、tokenizer 和 model config。

| checkpoint | tree SHA256 |
|---|---|
| `baseline/evaluation-step-12963` | `b8062047f235eb0af02c38b879ce2afe49e704a901001f934fed0158e96ae414` |
| `sc_idc/evaluation-step-12963` | `1495776f5746e9a963bd332368a5bb2853e38a37fefafe5ae49201b645940a40` |

## 5. 冻结 validation 配对结果

总体 support 为 1,664；greedy 和 sampled 都对两模型使用同一 record order 和 condition mapping。
下表的 `SemAvg` 是 Jaccard/Dice/Overlap 的算术平均；BSS rate 是 branch-supported-selectivity 的
总体比例，分母包括 non-nominal outputs。

### 5.1 Greedy

| 指标 | baseline | SC-IDC | delta |
|---|---:|---:|---:|
| Jaccard | 0.651683 | 0.653960 | +0.002278 |
| Dice | 0.714384 | 0.718303 | +0.003919 |
| Overlap | 0.806796 | 0.814322 | +0.007526 |
| SemAvg | 0.724287 | 0.728861 | +0.004574 |
| nominal adherence | 0.818510 | 0.815505 | -0.003005 |
| branch-supported rate | 0.575721 | 0.578726 | +0.003005 |
| matched selectivity | 0.604567 | 0.605168 | +0.000601 |
| branch-supported-selectivity | 0.575721 | 0.578726 | +0.003005 |
| branch-nonmarginal | 0.242788 | 0.236779 | -0.006010 |
| mean raw BSS | 0.393969 | 0.395034 | +0.001065 |
| parse / EOS | 0.999399 / 1.000000 | 0.998798 / 1.000000 | -0.000601 / 0 |
| mean generated tokens | 6.730168 | 6.719952 | -0.010216 |

SemAvg paired bootstrap：mean `+0.004574`，95% CI `[-0.005069, +0.013982]`。Nominal paired
bootstrap：mean `-0.003005`，95% CI `[-0.013221, +0.007212]`。

### 5.2 Sampled

| 指标 | baseline | SC-IDC | delta |
|---|---:|---:|---:|
| Jaccard | 0.611772 | 0.611027 | -0.000746 |
| Dice | 0.668958 | 0.671484 | +0.002526 |
| Overlap | 0.750686 | 0.759814 | +0.009128 |
| SemAvg | 0.677139 | 0.680775 | +0.003636 |
| nominal adherence | 0.808894 | 0.813101 | +0.004207 |
| branch-supported rate | 0.534856 | 0.538462 | +0.003606 |
| matched selectivity | 0.558894 | 0.561298 | +0.002404 |
| branch-supported-selectivity | 0.534856 | 0.538462 | +0.003606 |
| branch-nonmarginal | 0.274038 | 0.274639 | +0.000601 |
| mean raw BSS | 0.368546 | 0.369777 | +0.001231 |
| parse / EOS | 0.996995 / 1.000000 | 0.997596 / 1.000000 | +0.000601 / 0 |
| mean generated tokens | 6.701322 | 6.718750 | +0.017428 |

SemAvg paired bootstrap：mean `+0.003636`，95% CI `[-0.011177, +0.018064]`。Nominal paired
bootstrap：mean `+0.004207`，95% CI `[-0.009615, +0.018630]`。

这里没有把 natural branch-nonmarginality 命名为 laundering；labeled laundering detection 只应在
人工标注对抗集报告，本轮没有重新生成或使用该集合。

## 6. Occurrence-topology 分层

分层只复用同一批 validation outputs，不额外生成 rollout、不反向选择 checkpoint 或 alpha。三组
support 均大于 30，但仍受单 seed 和 validation-only 边界约束。

| decode | topology | n | SemAvg delta | paired 95% CI | BSS-rate delta | nominal delta |
|---|---|---:|---:|---:|---:|---:|
| greedy | first-only | 613 | +0.013018 | [-0.002281, +0.028411] | +0.013051 | -0.004894 |
| greedy | non-first-only | 568 | -0.004331 | [-0.021756, +0.013004] | -0.005282 | +0.005282 |
| greedy | repeated | 483 | +0.004330 | [-0.012856, +0.021954] | 0.000000 | -0.010352 |
| sampled | first-only | 613 | -0.004427 | [-0.029130, +0.020146] | -0.008157 | -0.011419 |
| sampled | non-first-only | 568 | +0.019053 | [-0.005475, +0.042671] | +0.021127 | +0.017606 |
| sampled | repeated | 483 | -0.004261 | [-0.032943, +0.024288] | -0.002070 | +0.008282 |

所有 topology 的 SemAvg CI 都跨 0，且 greedy/sample 的方向并不一致；这支持“异质且不确定”，不支持
某个 occurrence position 已获得稳定改善的结论。

## 7. 成本与产物哈希

SC-IDC reward adapter elapsed 比 baseline 多 `1,167.39 s`（`+75.42%`），graph executions 多
621,443 次（`4.389x`），graph requests 为 `4.169x`。这是 update-matched 主比较；本次自然产生的
checkpoint 不用于另行选择 compute-matched 终点，也没有追加训练来制造另一套预算比较。

终评另做四次 train-graph 离线审计，graph executions 分别为 baseline greedy 6,116、baseline
sampled 6,113、SC-IDC greedy 6,085、SC-IDC sampled 6,130；它们是报告成本，不是训练成本。终评从
16:11:27 到 16:12:43 UTC，约 75.45 秒。

| validation artifact | count | SHA256 |
|---|---:|---|
| `baseline-greedy.jsonl` | 1,664 | `ad316bea494ce7b9cc71361515d70455710636ed98fc17039a494db59305df9c` |
| `baseline-sampled.jsonl` | 1,664 | `a014f99161c062a7fcdd2563b7c36da498dd5b27bd8f7d661c87928add16a98e` |
| `sc_idc-greedy.jsonl` | 1,664 | `847b4e7e76a0510e93fa9197a4b618c30b3a0825f91370396d5cd6a0968c58d4` |
| `sc_idc-sampled.jsonl` | 1,664 | `3da436237aa6472930fd8ff0499a80b4b058ef42165fb32c2cda7126cb41ae4f` |

## 8. Smoke tests、启动异常与处置

1. `sc-idc-experiment2-smoke-20260816-2235-06e587b-a`：较早代码的两个 branch 均完成 1 update。
2. `sc-idc-experiment2-smoke-20260816-2240-46008be-a`：正式训练 SHA 的 deterministic smoke，两个
   branch 均完成 1 update 并保存可重载 checkpoint。
3. 正式 SHA 上曾运行全 suite `156 passed`；终评 SHA `9e1434b` 上针对 SC-IDC core、data、GRPO 和
   evaluator 的测试为 `36 passed in 3.82s`。DSW 的全局 pytest plugin 曾因系统 Hydra/OmegaConf
   版本冲突导致 collection 失败；设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 和仓库 `PYTHONPATH` 后
   项目测试通过，这不是实验失败。
4. 首个外层 launcher
   `/mnt/workspace/ctrlhgen-runs/sc-idc-experiment2-formal-20260816-2242-46008be-a.supervisor.log`
   因错误调用不存在的 `/mnt/workspace/envs/ctrlhgen/bin/bash` 在进入 Python 前失败；没有创建 branch
   run、没有 optimizer update。权威正式 run 是尾缀 `...2243-46008be-b`。
5. validation-only 收尾保留两个失败 status 目录：`validation-da752a9` 缺少 runtime-root 环境变量，
   `validation-da752a9-b` 暴露 `_datasets` 返回值解包错误。两者均在生成前 fail-closed；后者由提交
   `9e1434b` 最小修复并经测试。正式终评目录是 `validation-9e1434b-c`。

所有失败产物和日志均保留，没有删除 run/checkpoint，也没有误杀 DSW 系统服务或其他任务。

## 9. 判定与边界

- **训练健康：通过。** 两分支 update-matched、finite、parse/EOS 正常，且 checkpoint round-trip
  通过。
- **方法信号：小幅正向但不确定。** greedy/sample 的总体 SemAvg 与 BSS rate 都小幅增加，但所有
  总体 SemAvg 95% CI 跨 0，topology 结果异质。
- **成本：显著增加。** wall-time `1.754x`、graph executions `4.389x`；当前单 seed 增量不能证明
  成本收益比。
- **外推边界：** 只覆盖 WN18RR、specific-relation、seed 42、当前 static matcher 与冻结
  `alpha_IDC=0.05`；不能外推到 entity control、all-slot IDC、其他 KG 或其他 seeds。
- **后续门槛：** 若继续形成正式模型结论，应在不读取本轮 sealed final evaluation 的前提下预注册并
  运行至少两个额外 seeds，再以同一冻结 validation 分析规则做跨 seed 汇总。当前不应因本轮
  validation 数值重新选择 alpha、parent、checkpoint 或 topology。

## 10. DSW 停止记录

待完成 StopInstance 调用与独立 SSH 停止确认后填写。实例只停止，不删除；`save_image=false`，持久化
run、checkpoint 和数据必须保留。
