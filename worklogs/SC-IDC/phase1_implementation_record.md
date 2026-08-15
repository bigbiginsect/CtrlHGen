# SC-IDC 第一阶段实施记录

更新日期：2026-08-15

## 当前代码基线

- 正式复现配置仍只覆盖 `pattern` condition。
- `specific_entity` / `specific_relation` 已有 prompt 接口，但
  `condition_value_from_target()` 仍固定选择目标中的第一个合法 token；第一阶段没有修改这一行为。
- 已归档的 pattern rollout 不能支持模型级 semantic-control 结论。本阶段只审计机制与
  reference queries。
- 本地准备阶段工作分支为 `innovate/SC-IDC`；截至本地最小验证完成时尚未提交、推送或连接
  DSW，远端执行结果在完成后另行追加。

## 本地已实现

### 纯逻辑核心

新增 `akgr/reproduction/sc_idc.py`：

- action/raw query 到稳定不可变 AST 的双向转换；
- 带结构路径和同类序号的 entity/relation slot 提取；
- 基于 `seed + record_id + epoch + condition kind` 的无状态均匀槽位采样；
- 保持 pattern、entity/relation 数量和其他 token 不变的单槽位替换；
- 与 `GraphSampler` 集合语义等价并支持内部 `EMPTY/UNIVERSE` 的执行器；
- 最近 `i/u` 分支中性化，以及无分支祖先时的空根基线；
- WN18RR 图可观测代理上的 entity/relation 匹配候选、固定层级 fallback 和派生 seed 采样；
- `branch_marginal_delta`、`matched_delta`、分类和 `tau=0.1` 诊断分数。

### Reference 审计入口

新增 `akgr/reproduction/sc_idc_audit.py`，入口：

```bash
python -m akgr.reproduction.sc_idc_audit --self-check
```

正式审计参数包括实验配置、hash-verified fresh-RL manifest、输出目录、seed、每 pattern
抽样数、condition kind 和替代项数量。入口强制使用 train graph，不加载模型/checkpoint，
不访问 test split，并拒绝覆盖已有输出目录。

产物为：

- `slot-audit.jsonl`：槽位路径，原始/中性化/替换指称的基数与 SHA256，原始 delta，匹配特征，
  fallback，结构保持标志和分类；
- `summary.json`：按 condition kind、pattern、branch operator、first/non-first slot 汇总的
  `0/0.01/0.05` 阈值指标、权重契约、fallback、确定性正确率和执行成本；
- `manifest.json`：代码 SHA、dirty 状态、config/data/KG/input/产物 hashes、命令、seed 和
  运行时间。时间信息不写入前两个确定性产物。

### 测试

新增标准库测试 `tests/test_sc_idc_core.py`。本地已通过 17 项测试，覆盖：

- 全部 13 种 pattern 的新旧执行器等价性；
- OR-laundering 反例，其中 matched delta 为正、branch marginal 为零且分类为 `laundered`；
- 有效 union、冗余 intersection、负贡献、单链根查询和 negation；
- 重复 token 的位置区分、entity/relation 替换、结构和槽位数保持；
- 权重和为 1、固定 seed、manifest 哈希/数量篡改、输出覆盖拒绝；
- 两次完整 13-pattern 审计的 `slot-audit.jsonl` 与 `summary.json` 逐字节一致。

已执行并通过：

```bash
python3 -m unittest discover -s tests -p 'test_sc_idc*.py' -v
python3 -m akgr.reproduction.sc_idc_audit --self-check
python3 -m compileall -q akgr tests
git diff --check
```

本地环境缺少 `pytest`、`transformers`、`datasets` 和 `smatch`，因此完整仓库回归需要在 DSW
环境执行；本阶段没有为补齐依赖而下载软件或数据。

## DSW 开启后的接续步骤

在用户明确通知 DSW 已开启之前，不连接、不部署、不读取远端运行目录。收到通知后：

1. 本地复查工作树，提交并推送到 `origin`，记录精确 commit SHA。
2. 检查 DSW checkout 是否干净；若干净，则 fetch 并 detach 到该精确 SHA。若不干净，先审查，
   不丢弃远端改动。
3. 激活 `/mnt/workspace/envs/ctrlhgen/`，运行：

   ```bash
   pytest -m "not gpu and not live_data"
   ```

4. 使用 repaired-full 的 `fresh-rl-manifest.json`，在 train graph 上每个 pattern 抽 16 条，
   共 208 条 reference queries，枚举 entity/relation 全部槽位：

   ```bash
   python -m akgr.reproduction.sc_idc_audit \
     --experiment-config akgr/configs/reproduce/wn-pattern-full-train-author-aligned.yml \
     --fresh-manifest /mnt/workspace/ctrlhgen-runs/phase-d-repaired-full-20260811/control/fresh-rl-manifest.json \
     --split train \
     --condition-kind both \
     --per-pattern 16 \
     --replacements 3 \
     --seed 42 \
     --output-dir /mnt/workspace/ctrlhgen-runs/sc-idc-reference-audit-<run-id>
   ```

5. 验收新旧执行器等价、reference exact/Jaccard=1、结构保持和权重契约均为 100%，无 fallback 后的
   unscorable slot，并在第二个新目录复跑以比较两项确定性产物的字节。

effective/laundering 实际比例只记录为诊断结果，不设置有利方向门槛，也不生成模型能力结论。

## DSW 验证结果

验证日期：2026-08-15

### 分支与代码

- DSW checkout 原先位于 detached HEAD `5514df839a9b7c7b3cf6a12475d5727ed33beba8`，工作树干净。
- 已先切换到 `innovate/SC-IDC` 并快进同步；测试与审计代码 SHA 为
  `562690d058d4f3789865739444ffef6d327d087c`。
- 审计前后 DSW 工作树均为 clean，`HEAD` 与 `origin/innovate/SC-IDC` 一致。

### 完整回归与 self-check

pytest 首次启动时被系统级 Hydra pytest 插件与环境内 OmegaConf 的版本冲突阻断，尚未进入
测试收集。设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、不加载无关的系统插件后，对同一测试集合
重新执行：

```text
137 passed, 2 deselected in 9.41s
```

独立 `python -m akgr.reproduction.sc_idc_audit --self-check` 通过，并确认 OR-laundering 反例满足：

- matched delta `0.16666666666666666 > 0`；
- branch marginal delta `0`；
- 分类为 `laundered`；
- 新旧执行器等价且替换结构全部保持。

### 真实 WN18RR reference 审计

输入 provenance：

- fresh manifest SHA256：`0c5a096b7b3f40534d63dac8460e45939b56614d3bf8fe0bb0b9225a8f394afa`；
- fresh artifact SHA256：`6263a2d0ece6d897e850dbcaf71893865cb038e198dbd95b576d8b1609ad9260`；
- sampling manifest SHA256：`244ef264ad538df30ad58d72127d7fe3c31f52480dc3ddac882978400c49e460`；
- config semantic hash：`d7978678398d197265c879913d834f79fc9a344201832168a6282a60d1aec296`；
- config data hash：`7f2fc89a9c629dedf06e243301e091fe53c4409093ba60e00d1069da591b6c47`；
- config KG hash：`ab2a80eabd45a86cc1dad1acfc65c20b36a451070737b054e903bb2025dda4a9`。

相同 seed 的两次审计分别保存在：

- `/mnt/workspace/ctrlhgen-runs/sc-idc-reference-audit-20260815-562690d-a/`；
- `/mnt/workspace/ctrlhgen-runs/sc-idc-reference-audit-20260815-562690d-b/`。

每次均分层抽取 208 条 reference queries，枚举得到 944 个槽位行和 416 个
record-condition 权重单元，共记录 13,279 次执行。全部单次硬门槛通过：

- 13 种 pattern 全覆盖；
- 新旧执行器等价率 100%；
- reference exact 与 Jaccard=1 比例 100%；
- 替换结构保持率 100%；
- slot 权重契约 100% 正确，最大绝对误差为 0；
- unscorable slot 为 0；
- manifest 记录 `git_dirty=false`，未访问 test artifact。

两次确定性产物逐字节一致：

| artifact | SHA256 |
|---|---|
| `slot-audit.jsonl` | `935f749034aca66dfe929026d3e579e122a42eef4878e5a6a6c36335f61d1e34` |
| `summary.json` | `7ce19613fe0975dfbe14647af508eb84e3be3055cf886b8a5925d93186e7362e` |

包含时间和路径的 manifest 按设计不要求一致，A/B SHA256 分别为
`25c97a39ec32c109dbff7d44461a8a4381a7aac49da82fd094507ada8218848b` 和
`50faf77b25259ddf24be1c9105b351402bc53c6ca17e9e9a6ac36f24c2affb43`。

主阈值 `epsilon=0` 的诊断结果如下，不作为有利方向门槛：

| 分组 | marginal effective | matched selective | strict effective | laundering |
|---|---:|---:|---:|---:|
| overall | 76.44% | 98.04% | 76.44% | 23.56% |
| specific entity | 75.00% | 97.84% | 75.00% | 25.00% |
| specific relation | 77.88% | 98.24% | 77.88% | 22.12% |
| first slot | 87.26% | 98.78% | 87.26% | 12.74% |
| non-first slot | 65.90% | 97.31% | 65.90% | 34.10% |

`epsilon=0.01` 与主阈值结果相同；`epsilon=0.05` 时 strict effective 为 75.84%，laundering
为 24.16%。这些数字只描述 reference query 的逻辑机制，不支持模型 semantic-control 结论。

## Phase 2 术语与证据边界补充

第一阶段已经归档的 JSON schema、测试断言和上表列名保持原样，以保证产物 hash、复现实验记录
和代码 SHA 可核验；这些历史字段不应继续按字面作更强解释。后续文档与 Phase 2 实现统一使用：

| Phase 1 历史字段 | 后续解释 |
|---|---|
| `strict_effective` | `branch_supported_selective`：受控 occurrence 位于有边际的最近逻辑分支中，且当前值优于匹配替代值 |
| `laundered` | 对自然 reference 只解释为 `branch_nonmarginal`；仅在人工标注的 OR 遮蔽对抗样例中计作 laundering detection |
| `marginal_only` | `branch_supported_nonselective`：分支有边际，但未识别出当前值相对匹配替代的正优势 |

因此，上表的 23.56% 是 **branch-nonmarginal reference occurrences**，不是“模型 laundering
比例”，也不能证明 gold query 有意规避控制。反过来，两个 delta 都为正也只支持
branch-supported selectivity，不能证明 controlled predicate 本身不可删除或具有因果必要性。

第一阶段按 occurrence 枚举并在 record-condition 内赋予权重，是 reference attribution 审计，
不是 Phase 2 的 prompt 采样规则。specific-relation prompt 只暴露 relation value；Phase 2 必须按
unique relation values 采样，并对生成假设中该值的全部 occurrences 联合干预。对应代码迁移、
兼容策略和验收顺序记录在 `worklogs/SC-IDC/phase2_contract_handoff.md`，不回写第一阶段历史产物。

## 延期范围

以下内容不属于第一阶段代码变更：

- 硬结构 grammar / constrained decoding；
- semantic SFT；
- GRPO reward 接入；
- semantic-control rollout 的零方差实验；
- all-slot IDC 消融；
- 对现有配置哈希、checkpoint 或训练契约的修改。
