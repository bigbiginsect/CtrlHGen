# SC-IDC Phase 2 Data / specific-relation SFT preflight 实施记录

更新时间：2026-08-15（Asia/Shanghai）

## 1. 本轮目标与停止边界

本轮完成 proposal / 接手文档中 Phase 2 的 Data-1、Data-2 和 specific-relation SFT 正式启动前验证：

- 固化 specific-relation SFT 的目标数据配置、条件采样和验证集条件；
- 从 Phase 1 pattern checkpoint 导入模型权重，但不继承 optimizer / scheduler 状态；
- 生成并冻结后续 Signal、RL train 和 sealed final evaluation 的条件清单；
- 在 DSW L20 上完成正式启动前的只读 forward smoke；
- 未启动正式 SFT，未执行 backward、optimizer step 或 checkpoint 写入；
- 未启动 Signal / RL / final evaluation。

达到该边界后，DSW 开发机已停止，避免继续计费。

## 2. 冻结代码与主要实现

通过验证并作为本轮训练前冻结基线的代码提交为：

```text
af94eb41c2396854caa803d9f1362ab8a356e5b9
```

该提交已经推送到 `origin/innovate/SC-IDC`。主要新增或修改内容包括：

- `akgr/reproduction/sc_idc_phase2_data.py`：Phase 2 数据重绑定、动态唯一关系条件采样、条件资产和 manifest 冻结；
- `akgr/reproduction/sc_idc_sft_launch_preflight.py`：正式训练前的无更新 launch gate；
- `akgr/configs/reproduce/wn-specific-relation-full-train-author-aligned.yml`：specific-relation 全监督训练配置；
- `scripts/sc-idc/prepare-specific-relation-sft.sh`：生成冻结数据资产；
- `scripts/sc-idc/preflight-specific-relation-sft.sh`：执行启动前 gate；
- `scripts/sc-idc/sft-specific-relation.sh`：正式 SFT 入口。

实现中的关键约束如下：

1. 从 frozen pattern parent 只导入模型权重，specific-relation SFT 使用全新 Adam optimizer 和 scheduler。
2. 条件采样以目标数据 variant 为作用域；同一 query 内按确定性顺序采样一个唯一 relation condition。
3. full-supervision 的跨 split 重叠按既有 split-dedup 合同处理；sealed final evaluation 的 query 不进入正式 RL train 条件池。
4. 数据 preflight manifest 绑定精确 Git SHA、配置语义哈希、数据哈希、KG 哈希和 parent checkpoint 树哈希。
5. launch gate 必须证明没有 backward、optimizer step、checkpoint 写入，且不会加载 sealed final evaluation manifest。

## 3. DSW 环境验证

DSW checkout 以 detached HEAD 部署到上述冻结提交，工作区干净。环境为单卡 NVIDIA L20，Python 3.11.11，PyTorch 2.6.0+cu124。

CPU / 合同测试命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -m "not gpu and not live_data" -q
```

结果：

```text
145 passed, 2 deselected in 6.29s
```

## 4. 冻结输入身份

目标配置：

```text
akgr/configs/reproduce/wn-specific-relation-full-train-author-aligned.yml
```

身份哈希：

| 对象 | SHA256 |
| --- | --- |
| 配置语义 | `15a4c9e859e22e457fa7e9be0e7e8f43c464268a5588b37f4797259fa51a2db8` |
| 目标数据 | `7f2fc89a9c629dedf06e243301e091fe53c4409093ba60e00d1069da591b6c47` |
| KG | `ab2a80eabd45a86cc1dad1acfc65c20b36a451070737b054e903bb2025dda4a9` |

冻结 parent checkpoint：

```text
/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-full-train-author-aligned-c4/unconditional-epoch-45
```

checkpoint tree SHA256：

```text
6a5edf2f6e78f68decf0329a81843769833a0b317c2d99fcb9ca372b3c58fa62
```

## 5. Data-1 / Data-2 结果与确定性复跑

最终权威 preflight 目录：

```text
/mnt/workspace/ctrlhgen-runs/sc-idc-specific-relation-sft-preflight-20260815-af94eb4-a/
/mnt/workspace/ctrlhgen-runs/sc-idc-specific-relation-sft-preflight-20260815-af94eb4-b/
```

两次独立生成的 8 个条件 / manifest 文件逐字节一致。冻结规模：

| 资产 | 条件数 | SHA256 |
| --- | ---: | --- |
| `validation.conditions.jsonl` | 1,664 | `0bcbf68219c6329cff591fc3a6f596b552569b0faea814129ac98b9b3a5b71a4` |
| `validation.manifest.json` | — | `58ddb2c3359ac49a6a8153331b89299268df2324ef6308a53aff2bf04da446ac` |
| `signal.conditions.jsonl` | 208 | `1415485d25b35aac2609848e7668877750a63c0328612efcb0648488f7a3fba1` |
| `signal.manifest.json` | — | `47774daa4bf3a034af4ab58ef0a39b13159958d65d8f2ccfb8a0ce35213e8914` |
| `rl_train.conditions.jsonl` | 103,711 | `53dc2dd17842a0bc3f93a09203e2cb9558bb3c447f1425d0cb12fcdcca615899` |
| `rl_train.manifest.json` | — | `17bbd14d8b33787433122cca3ff1cc62b878cda1ba384ee86977fe1a09273122` |
| `final_evaluation.conditions.jsonl` | 1,664 | `c9e5e4c948d9429077801cb2f0632be716029f6a6852a5a3fc46213f91e41020` |
| `final_evaluation.manifest.json` | — | `561217a80cc8dcfe6359f30431bfa4567344249e84e7525085aaf0b8cbf6060a` |

补充计数：

- fresh full source：104,000 条；
- 因 query 与 sealed final evaluation 重叠而确定性排除：81 条；
- Signal：208 条（16 × 13 patterns）；
- 排除 Signal 后的正式 RL train：103,711 条。

早期两次 gate 曾 fail-closed：第一次暴露 merged-train 内部 query 重复与既有 split-dedup 合同不一致，第二次暴露 isolation 校验作用域使用错误。对应修复分别落在 `f30aad6` 和 `9a96fc5`；失败目录仅保留诊断意义，不能作为后续实验输入。上述 `af94eb4-a` / `af94eb4-b` 才是权威资产。

## 6. specific-relation SFT launch smoke

权威报告：

```text
/mnt/workspace/ctrlhgen-runs/sc-idc-specific-relation-sft-preflight-20260815-af94eb4-a/sft-launch-verification.json
```

报告 SHA256：

```text
a3bba0d719eb149177203625a8b93d2f8166142a6218b8573897a6408397058f
```

验证结果：

- status：`ready`；Git SHA 与冻结提交一致，DSW checkout 干净；
- base train：104,000；validation：1,664；固定 validation conditions：1,664；
- L20 forward-only smoke，batch size 2；
- train loss：`2.2931950092315674`；
- validation loss：`1.9439325332641602`；
- smoke train conditions：`[-5, -4]`；validation conditions：`[-11, -8]`；
- backward：0；optimizer steps：0；checkpoint writes：0；
- fresh optimizer state entries：0；
- sealed final evaluation manifest 未加载。

冻结的正式训练调度为 Adam、microbatch 160、每 epoch 650 optimizer steps、50 epochs、总计 32,500 optimizer steps、5-step warmup、`linear_warmup_constant`。

## 7. 正式 SFT 启动边界

正式训练尚未启动。获准启动时，应先恢复 DSW，并将 checkout **精确 detach 到**：

```text
af94eb41c2396854caa803d9f1362ab8a356e5b9
```

然后执行冻结报告记录的命令：

```bash
bash scripts/sc-idc/sft-specific-relation.sh \
  /mnt/workspace/CtrlHGen/akgr/configs/reproduce/wn-specific-relation-full-train-author-aligned.yml \
  /mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-full-train-author-aligned-c4/unconditional-epoch-45 \
  /mnt/workspace/ctrlhgen-runs/sc-idc-specific-relation-sft-preflight-20260815-af94eb4-a/sft-preflight.json
```

### 精确 SHA 注意事项

本日志与之前未提交的文档修改会形成一个新的、仅文档提交，因此分支 HEAD 将不再等于 `af94eb4`。冻结 preflight 硬绑定 `af94eb41c2396854caa803d9f1362ab8a356e5b9`；后续正式 SFT 必须：

- 使用 detached `af94eb41c2396854caa803d9f1362ab8a356e5b9` 和现有冻结资产；或
- 在新的代码 SHA 上重新执行完整 Data-1 / Data-2 / launch preflight，生成新的权威资产。

不能仅把本次文档提交的新 SHA 当作已验证训练 SHA。

## 8. DSW 停止记录

实例：

```text
dsw-uhn5s45l2r8qw8n0f5
```

在完成验证后通过 PAI DSW `StopInstance` 停止，API 返回 `Success=True`、`Code=0`、`SaveImage=false`。随后两次独立 SSH 检查均超时，确认开发机不再提供连接。实例没有删除，持久化数据、checkpoint 和上述 run 目录保留。

下一步只有在获准启动正式 specific-relation SFT 时才恢复实例；恢复后先核验实例状态、checkout SHA、Git cleanliness 和冻结资产哈希，再启动训练并登记正式 run ID、命令、日志与 checkpoint 输出位置。
