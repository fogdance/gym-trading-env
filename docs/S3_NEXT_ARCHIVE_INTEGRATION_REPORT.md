# S3 Next — Research Archive V1 集成

日期：2026-10-14。本次按用户最终范围只合并 Archive；不审计或移植 C2 等其他历史分支，不开发 Composition。

## 来源与独立工作区

- Gym 起点：`58708ceb9e10cc2a355511bdf3133710f4459e0e`，`integration/s3-next`。
- Dreamer 配对：`d41bc80ba13b3dc3f5d2c32fcdc800613b2d0b4d`，本次没有源码或提交变更。
- Archive 来源：`b7d5edb30d1dcb2f2d845593a4cef498d923536d`，父基线是 W&B `ad6490675bf853a7793763f95aa7ba7f12fa379e`；只有一个新增提交。
- 临时 Gym 分支：`feat/s3-next-archive-integration`；工作区 `/home/v/Documents/work/research/fg-s3-next-archive-integration/`。Dreamer 在同一工作区 detached 到上述配对提交，只用于原有 CPU 回归。
- 通过验收后普通合并至 Gym `integration/s3-next`，原 Archive 分支及其工作树保持不变。最终提交与实际 Git 输出见集成工作区外置 `baseline-lock.json` 和 `validation/archive_v1/`。

## 实际实现审查

| 能力 | 源码及实际行为 |
|---|---|
| inspect | `s3_archive/inspection.py` + `source.py`：读取冻结 Run/Prepared、FG_FULL_V1、done、真实 checkpoint/receipt 字节哈希；重建双仓历史源码；发现依赖和已确认日志水位。不初始化模型或联网。 |
| archive | `archive.py` + `io.py`：默认 dry-run；独立文件复制、前后哈希、空间预留、staging 原子发布、幂等重试、原子升级及旧版本保留。 |
| verify | `verification.py`：L1 身份/哈希/交易核对；CLI 会写归档自身的验证元数据；严格只读检查使用 `evidence()`。 |
| L2 / L3 | `isolation.py` + `validation_worker.py`：显式 GPU 与匹配 Python、Bubblewrap 私有映射；L2 独立 Actor-only 重评，L3 全状态/Replay 核对及一次更新。仅复制 full 不提升等级，exact_replay_verified 保持未知。此轮未执行 GPU 验证。 |
| publish | `publication.py` + 共用 `research_tracking/milestones.py`：精确 Run/receipt/checkpoint 映射、稳定幂等键、标签并集、脱敏摘要、云端回读；离线保存 PENDING。本轮仅 Offline/Mock 和只读本地映射，不调用线上发布。 |
| 依赖 | 新增原分支 `requirements-archive.txt`，PyYAML/NumPy；实测训练环境 Python 3.11.11、PyYAML 6.0.2、NumPy 1.26.4，Bubblewrap 可用。W&B 继续使用既有 SDK 0.19.1 隔离环境，没有安装或升级共享依赖。 |

## 冲突处理与直接修复

1. `research_tracking/evals.py` 已是 V2。保留共享 `evaluation_metrics`、三个逆行窗口、去重及未知值语义；只补 Archive 所需开仓笔数核对和可核对的交易流水净盈亏。未退回旧 V1 解析器。
2. `sdk_worker.py` 已保留云端名称和标签并内置代理。保留当前实现，最终对原 integration 无 diff；不重复接入旧分支仅合并标签的逻辑。
3. Archive 旧校验器直接累加原 attributions，与 V2 的相同重复记录去重不一致。新增回归先实际报 `Archived historical evaluation does not reconcile`；修正为复用 `unique_events` 后通过。
4. 成功 inspect 的 `--json-out` 未保护独立 Prepared 路径。新增回归先实际返回 0 并写入测试源 Prepared；补充 Prepared/Eval 输出隔离后返回 2，源目录未写入。

Trainer、神经网络、Loss、正式配置、Env、Reward、Oracle、市场分钟、交易规则、原生 Replay、DAgger 与 FullCheckpoint 源码均与起点相同。所有新增代码位于独立 Archive/Tracker/CLI 层，不修改训练接口。

## 真实工件与写入型验证

| 来源 | 本轮实际结果 |
|---|---|
| Tutor 800K | 原 manifest `5cf093b56ca3bb340cca9520b5579105b1cecc9f044262d5a23d17c545c6a474` 与 receipt 匹配；7 ADD、5 TP、2 SL、Reward −3.805870048525721、扣费净盈亏 −2226；V2 逆行窗口 0/1/3，全持仓 4。 |
| 已有 Tutor 标准归档 | 只读 `evidence()` 验证 7,542 个清单文件、历史身份和上述指标。存储等级 FULL 属于原先 GPU 验证成果；本次不把哈希检查宣称为新的 L2/L3 运行验收。 |
| 当前 B2-Full 200K | 完成节点与已有 Actor-only 评价匹配，manifest `b0a67e6ff61df3c56cf1ee904e038f1b0ce00c50fe0e9decb587fbf3b70f0eaf`。ADD/TP/SL/Reward 均为 0；保留 Outcome Adapter 诊断，不伪造教学或盈利结论。 |
| W&B 本地映射 | Tutor Train/Eval `3dc57d410d9b50601120` / `9832f27f1b038b84fb58`；B2 `de770b0df7f952eb0aeb` / `0409dbc8f221ee65a3f8`。通过 trace_identity、tracker_state 和 checkpoint/receipt SHA 确认，未通过名称猜测、未更新云端。 |
| 临时 Tutor L1 | `/tmp/s3_archive_integration_tutor/s3-tutor-cd-800000-735fe1b4e482`；真实 inspect → dry-run → archive → verify → Offline publish → list。独立源码与原始证据已复制，不保留模型字节，实际等级 EVIDENCE、发布 PENDING。 |
| 临时 B2 L1 | `/tmp/s3_archive_integration_b2/s3-b2-200000-c822c0959ad0`；真实 inspect → archive → verify。活跃 Run 的日志只按 200K 已确认水位读取；未写源 Run，未进行训练恢复。 |

原正式 Tutor/B2 的 Run、Prepared、Checkpoint、Replay、Sidecar、评价、冻结 worktree 和已有归档均只读。前后保护记录见 `validation/archive_v1/before.json` 及最终 `protection.json`；活跃训练正常新增日志/Checkpoint不被误判为本次代码写入。

## 测试与命令

| 测试 | 本轮结果 |
|---|---|
| Archive + W&B 全部可用单元/CLI/Mock/Offline | 48 PASS：Archive 26，Tracker/评价 22；真实 Offline SDK 在主机用户环境运行。 |
| A2/Tutor/B2 + FullCheckpoint CPU 模型回归 | 20 PASS，216.13 秒；真实小模型梯度、三模式语义、B2 100/100/100 及恢复。 |
| 市场分钟/Oracle/账户/DAgger/Replay 公共合同 | 44 PASS / 1 SKIP；SKIP 是临时工作树未安装旧 Tutor E2 GPU 10K 收据，未计为通过。 |

首次沙箱组合为 47 项（此时 CLI 主路径测试尚未加入），44 PASS / 3 FAIL：W&B 本地服务启动超时，涉及 OfflineSDK 两项和 prediction-eval Offline 上传。未忽略失败；主机授权后 48 项全部通过。两项修复回归的修复前失败和修复后通过也有真实命令输出。

执行目录为临时集成 Gym。测试 stdout 和真实工件 JSON 收据全部位于公共集成工作区 `validation/archive_v1/`。

```bash
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  /home/v/Documents/work/research/fg-s3-wandb-env/bin/python -m unittest \
  s3_archive.tests.test_archive research_tracking.tests.test_tracker \
  research_tracking.tests.test_eval_v2 research_tracking.tests.test_prediction_evals -v

env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  MPLCONFIGDIR=/tmp/s3_archive_mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest \
  ../dreamerv3/embodied/tests/test_fg_s3_b2.py \
  ../dreamerv3/embodied/tests/test_fg_s3_tutor_full.py \
  ../dreamerv3/embodied/tests/test_fg_full_checkpoint.py -q

env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  MPLCONFIGDIR=/tmp/s3_archive_mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest \
  tests/entry_v1/test_s3_market_minute.py tests/entry_v1/test_fg_s3_b2_outcome.py \
  tests/entry_v1/test_fg_s3_tutor_cd.py tests/entry_v1/test_fg_s3_tutor_e2.py \
  tests/entry_v1/test_fg_s3_v2_pool.py tests/entry_v1/test_fg_s3_v2.py \
  -k 'not small_cpu and not real_native and not actual_gpu and not real_original and not real_gpu' -q
```

## 限制与后续

- 本次 GPU 训练/独立重评/一次模型恢复更新及线上 W&B 发布 NOT_RUN。L2/L3 的实现、等级边界及既有历史证明保留；新工件只授予实际达到的 L1。
- `/tmp` 是测试结果目录，长期归档继续使用已配置的 `/data/logdir/fg_single3/research_archives/`，本次不复制大型正式依赖集到那里。
- C2、市场分钟单独报告和其他历史分支未合并；范围仅 Archive。两个主分支及原研究/训练目录不重置、不删除。
- 验收提交和双仓完整 SHA 由独立 `baseline-lock.json` 锁定。此公共基线可作为后续 Composition F0 的起点；本轮交付后停止，等待 Review。
