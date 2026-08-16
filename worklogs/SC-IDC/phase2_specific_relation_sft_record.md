# SC-IDC Phase 2 specific-relation SFT 正式运行记录

更新时间：2026-08-16（Asia/Shanghai）

## 1. 状态

本记录对应 Phase 2 Data-3 的正式 `specific_relation` conditional SFT。训练已于
2026-08-16 15:20:45（Asia/Shanghai）启动，目前状态为 **running**。最终训练审计、选中
checkpoint、严格重载和 DSW 停止结果将在 run 进入终态后追加到本文。

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

## 5. 巡检与终态收尾

Codex heartbeat `sc-idc-specific-relation-sft` 已设为 ACTIVE，每 20 分钟巡检一次。正常巡检读取
进程、GPU、磁盘、history、validation/checkpoint 和新增日志，不读取 sealed final evaluation
manifest。单轮 history 没有新增但进程和 GPU 活跃时不误判为卡死。

成功终态要求：exit code 为 0；history 恰为 epoch 1--50；loss 全部有限；每 epoch 650 steps；
validation/checkpoint 健康；`conditional-best` 指针有效；并在新进程中完成一次 CPU strict
checkpoint reload。明显异常时只终止本 run 的准确 PGID，保留现场并记录原因。

进入成功或明确失败终态后，heartbeat 将先补全本文并提交、推送到 `origin/innovate/SC-IDC`，
再通过官方 `alibabacloud_pai_dsw20220101` SDK 和实例内 CredentialsURI profile 调用 PAI DSW
`StopInstance`，固定 `SaveImage=false`。凭据不会写入日志或仓库。只读 `GetInstance` 已验证该
调用链可用，并返回当前实例状态 `Running`。停机后 heartbeat 将暂停自身，避免继续巡检。

## 6. 最终结果（待终态补充）

待填写：完成时间、完整 loss/validation 轨迹摘要、best checkpoint、strict reload、异常处置（如有）、
StopInstance 返回与停机复核。
