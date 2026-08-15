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

## 延期范围

以下内容不属于第一阶段代码变更：

- 硬结构 grammar / constrained decoding；
- semantic SFT；
- GRPO reward 接入；
- semantic-control rollout 的零方差实验；
- all-slot IDC 消融；
- 对现有配置哈希、checkpoint 或训练契约的修改。
