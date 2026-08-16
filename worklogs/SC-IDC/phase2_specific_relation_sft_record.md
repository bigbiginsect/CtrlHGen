# SC-IDC Phase 2 specific-relation SFT 正式运行记录

更新时间：2026-08-16 16:47（Asia/Shanghai）

## 1. 状态

本记录对应 Phase 2 Data-3 的正式 `specific_relation` conditional SFT。训练已于
2026-08-16 15:20:45（Asia/Shanghai）启动，并于 16:42:47 正常结束；最终状态为
**completed / exit code 0**。训练 wall time 为 1 小时 22 分 2 秒。

本次只运行 specific-relation SFT；没有启动 Signal、GRPO 或 sealed final evaluation。

## 2. 冻结身份与启动前核验

正式训练继续使用 preflight 硬绑定的代码提交：

```text
af94eb41c2396854caa803d9f1362ab8a356e5b9
```

DSW checkout 在启动时为 detached HEAD，工作区 clean，没有把当前分支后续的文档提交
`629b94b` 当作训练代码。启动前还确认：

- 实例 `dsw-uhn5s45l2r8qw8n0f5` 可连接，单张 NVIDIA L20 空闲；
- `/mnt/workspace` 可用空间约 121 GiB；
- 没有其他 `akgr.abduction_model.main` 或 specific-relation SFT 进程；
- 正式 run/checkpoint 目标目录均不存在，没有续跑或覆盖旧产物；
- frozen parent、data manifest、SFT preflight 和 launch verification 均存在；
- `sft-launch-verification.json` SHA256 仍为
  `a3bba0d719eb149177203625a8b93d2f8166142a6218b8573897a6408397058f`；
- preflight status 为 `ready`，CUDA 可用，PyTorch 为 `2.6.0+cu124`。

冻结 parent：

```text
/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-full-train-author-aligned-c4/unconditional-epoch-45
```

冻结 SFT preflight：

```text
/mnt/workspace/ctrlhgen-runs/sc-idc-specific-relation-sft-preflight-20260815-af94eb4-a/sft-preflight.json
```

## 3. 正式命令与产物位置

正式命令：

```bash
bash scripts/sc-idc/sft-specific-relation.sh \
  /mnt/workspace/CtrlHGen/akgr/configs/reproduce/wn-specific-relation-full-train-author-aligned.yml \
  /mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-full-train-author-aligned-c4/unconditional-epoch-45 \
  /mnt/workspace/ctrlhgen-runs/sc-idc-specific-relation-sft-preflight-20260815-af94eb4-a/sft-preflight.json
```

控制目录与 stdout/stderr：

```text
run id: sc-idc-specific-relation-sft-formal-20260816-152045-af94eb4
control: /mnt/workspace/ctrlhgen-runs/sc-idc-specific-relation-sft-formal-20260816-152045-af94eb4/
log: /mnt/workspace/ctrlhgen-runs/sc-idc-specific-relation-sft-formal-20260816-152045-af94eb4/stdout-stderr.log
```

训练 history/validation 输出：

```text
/mnt/workspace/ctrlhgen-runs/sc-idc-wn-specific-relation-full-sft-v1/
```

checkpoint 输出：

```text
/mnt/workspace/ctrlhgen-checkpoints/sc-idc-wn-specific-relation-full-sft-v1/
```

launcher PID 和 PGID 都为 `2430`，分别记录在控制目录的 `launcher.pid` 与
`process-group-id.txt`。wrapper 会在终态写入 `exit-code.txt` 和 `finished-at.txt`。

## 4. 正常开跑证据

启动后内部命令为 conditional SFT，parent 和 Phase 2 manifest 路径与冻结命令一致。首轮观察到：

- GPU utilization 约 97%，显存约 21,039 MiB，温度 58--60°C；
- 没有 traceback、OOM、非有限 loss 或进程退出；
- 日志中的 CuBLAS 和 memory-efficient attention deterministic warning 是本项目既知 warning，
  不作为异常；
- epoch 1 已完整落盘，`optimizer_steps=650`、`train_loss=0.3067472056012887`；
- epoch 1 LR 从 `5e-6` 按 5-step warmup 升至 `5e-5`，总调度仍为 50 epochs / 32,500
  optimizer steps。

### 4.1 完整训练审计

训练完整执行 50 epochs / 32,500 个本阶段 optimizer steps。history 恰有 50 行，epoch 为连续的
1--50；每轮均为 650 steps；所有 train loss 有限。train loss 从 `0.3067472056` 降至
`0.0966858599`，降幅 68.48%。继承 parent global step 后，global step 从首轮末的 `58,205`
增加到 `90,055`。

每 5 epochs 使用冻结的 1,664 条 validation conditions 做一次验证，10 次 validation 全部
`health_pass=true`：

| epoch | train loss | validation loss | condition accuracy | Jaccard | Dice | Overlap | Smatch | parse | EOS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.15391 | 3.08623 | 0.73257 | 0.63877 | 0.70086 | 0.79298 | 0.57207 | 0.99579 | 1.00000 |
| 10 | 0.13650 | 3.19797 | 0.73918 | 0.64405 | 0.70566 | 0.79746 | 0.57320 | 0.99760 | 1.00000 |
| 15 | 0.12670 | 3.31249 | 0.75541 | 0.63391 | 0.69600 | 0.78790 | 0.58191 | 1.00000 | 1.00000 |
| 20 | 0.11924 | 3.34986 | 0.76262 | 0.62350 | 0.68637 | 0.78062 | 0.58206 | 0.99820 | 1.00000 |
| 25 | 0.11379 | 3.41967 | 0.77163 | 0.62007 | 0.68402 | 0.77744 | 0.58364 | 0.99940 | 1.00000 |
| 30 | 0.10917 | 3.44601 | 0.77945 | 0.62463 | 0.68707 | 0.77919 | 0.58136 | 0.99880 | 1.00000 |
| 35 | 0.10576 | 3.50038 | 0.77584 | 0.61941 | 0.68268 | 0.77597 | 0.57353 | 0.99820 | 1.00000 |
| 40 | 0.10268 | 3.52606 | 0.77584 | 0.62782 | 0.69158 | 0.78648 | 0.57678 | 1.00000 | 1.00000 |
| 45 | 0.10039 | 3.58877 | 0.77103 | 0.62173 | 0.68604 | 0.78725 | 0.57255 | 0.99940 | 1.00000 |
| 50 | 0.09669 | 3.64097 | 0.77344 | 0.60941 | 0.67327 | 0.76627 | 0.57512 | 1.00000 | 1.00000 |

训练规则按 condition accuracy、parse、Jaccard、validation loss 的固定词典序选择
`conditional-epoch-30`。它不是 final epoch 的别名；`conditional-best` symlink 和
`conditional-best.json` 都指向 epoch 30。保留的实体 checkpoint 为 epoch 30、45、50。

全过程没有 traceback、OOM、NaN、非有限 loss、step 漂移、Git SHA/cleanliness 漂移或训练
进程异常退出。日志中的 CuBLAS 和 memory-efficient attention deterministic warning 与既有运行
一致，不构成本轮异常。

## 5. 巡检与严格重载

Codex heartbeat `sc-idc-specific-relation-sft` 每 20 分钟巡检一次。巡检读取进程、GPU、磁盘、
history、validation/checkpoint 和新增日志，不读取 sealed final evaluation manifest。正式巡检分别
在 epoch 15、28、40 和训练终态完成，均未发现异常；训练阶段 GPU utilization 约 95--97%，显存
约 21.1 GiB。

终态检查确认 exit code 为 0；history 恰为 epoch 1--50；loss 全部有限；每 epoch 650 steps；
10 次 validation 全部健康；`conditional-best` 指针有效。没有执行异常终止或清理训练进程。

训练结束后在新进程中对 `conditional-best` 执行 CPU strict checkpoint reload，结果为 `passed`：

- resolved checkpoint：`conditional-epoch-30`；
- checkpoint tree SHA256：`8f172569dbbc0d71409a19f3579663eac53aa266be191277cc4da5f1422c5dda`；
- stage/condition：`conditional` / `specific_relation`；
- config semantic hash：`15a4c9e859e22e457fa7e9be0e7e8f43c464268a5588b37f4797259fa51a2db8`；
- data manifest SHA256：`244ef264ad538df30ad58d72127d7fe3c31f52480dc3ddac882978400c49e460`；
- condition lineage 匹配，固定 validation condition/contract 都为 1,664 条；
- 模型参数量为 74,499,072；
- `final_evaluation_manifest_loaded=false`。

## 6. 下游冻结 parent

Experiment 1 与后续两个 GRPO 分支必须共同使用本轮训练规则选中的 parent：

```text
/mnt/workspace/ctrlhgen-checkpoints/sc-idc-wn-specific-relation-full-sft-v1/conditional-best
```

当前它解析到：

```text
/mnt/workspace/ctrlhgen-checkpoints/sc-idc-wn-specific-relation-full-sft-v1/conditional-epoch-30
```

不得用 epoch 50 替换该 selection，也不得在下游选择 parent 时读取 sealed final evaluation。

## 7. DSW 停止记录

训练、审计和记录提交完成后，将通过官方 `alibabacloud_pai_dsw20220101` SDK 和实例内
CredentialsURI profile 调用 PAI DSW `StopInstance`，固定 `SaveImage=false`。凭据不会写入日志
或仓库。API 返回与停机复核将在调用后追加。
