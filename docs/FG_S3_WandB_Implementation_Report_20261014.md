# FG S3 W&B 实施收尾报告

2026-10-14。范围为独立只读 Tracker；本次仅导入 Tutor-CD，未导入历史 A2。
原 Tutor 在实施期间自行完成 1M，原始签名收据为 `FORMAL_COMPLETE`。
没有启动、重启或停止 GPU 训练，没有生成新的 DEV 评价。

## 代码与来源身份

开发双 worktree：`/home/v/Documents/work/research/fg-s3-wandb-dev/`，分支
`feat/fg-s3-wandb-tracking`。Gym 基线实际核对为 `61a3e88`，Dreamer 为
`d76a423dd7930756c3609cc6f7197941f583c160`，并非默认采用原仓库 HEAD。

Gym 实现提交：`03310d5` → `ce26db9` → `957befc`；最后一次实现提交包含代理、
SDK 重放同步、完整历史确认和自动启动器。Dreamer 无改动，SHA 如上；无需空提交。
本报告单独提交，最终工作树 HEAD 以 `git rev-parse HEAD` 为准。

被保护的正式 Tutor 双仓源码位于 `fg-s3-tutor-mm-train/`：

- Gym：`3a6f889adf65583a1ab636f244ca05218f19196f`
- Dreamer：`c18a8b5917f73110f9b6ce840495f4da1d7ca1d4`

两个冻结 worktree 仍然 clean；212 个冻结源码/不可变配置、800K manifest/done
和评价文件的前后 SHA256 比较为 `changed=[]`。原 `research/gym-trading-env`
与 `research/dreamerv3` 仍为 `d374e93` / `bb1d123`，用户原有 untracked 文件保留。
未改 `.gitignore`，未读取 `.bz2`，未使用其他实验仓库。

Tracker 放在 `research_tracking/` 和 `tools/fg_s3_wandb.py`，未改冻结入口
`tools/fg_s3_v2_experiment.py`，未改 Gym `src/gym_trading_env/` 或 Dreamer
`dreamerv3/`、`embodied/`。旧模型代码身份扫描范围没有新增文件。
独立环境为 `/home/v/Documents/work/research/fg-s3-wandb-env`，实测 SDK 0.19.1；
未修改 Trainer 的 Conda 环境。自动入口将原环境原样传给原命令。

## 已实现

共享入口支持 `inspect / import / follow / import-evals / sync / auth-check / launch`。
固定 Project 为 `fg-single3-research`；Entity 从参数或环境获取。
支持 online/offline/disabled，Tracker 专用代理和来源锁。
冻结配置、源 G/D SHA、dirty、模型/Oracle/Reward/clock/数据身份、超参数均来自
原签名身份及 prepared 文件；完整脱敏配置作为小 JSON artifact，索引字段进入 Config。
无法认证的旧 metadata 保持 unknown，未用当前源码替换旧 Run 配置。

完整 Checkpoint 的 manifest/done 和匹配 cycle/update 前缀决定 canonical 水位。
未提交数据仅进入单独 `provisional/*` monitor，回滚启动新 segment。
JSONL 按完整行流式解析，按 40 次更新聚合；未定义 Recall 和缺失字段不填零。
派生交易摘要独立于 TRAIN；TRAIN 中 Oracle 接管的 TP 不代表 Actor 自主成绩。
Checkpoint 只上传 metadata 索引，缺失节点标记 unavailable。

所有写入位于独立 Tracker 根目录。未向 SDK 传入模型、Replay、Oracle、行情本体、
原始账户状态或源 diff。SDK 自动代码、控制台和机器统计关闭。
完整 bundle 内容校验仍由 Trainer 执行；Tracker 不重读模型和 Replay 验证可恢复性。

`launch` 仅包装用户明确给出的 start/resume/evaluate 原命令；先启动原命令，再
独立等待身份和完整 Checkpoint，跟踪并在结束后做最后一次收集。
网络故障明确降级 offline，下次恢复连通时同步积压记录。
原命令退出码保留；Tracker 失败不会中断原命令。
它是未来训练的推荐入口；原冻结 CLI 未被自动替换。

## 真实云端结果

用户终端已登录；通过实际 API 核实 Entity `xkm104501-none` 和 Project **PRIVATE**。
代理为用户提供的 `http://127.0.0.1:10808`，仅用于 Tracker/SDK 子进程。

- [Project](https://wandb.ai/xkm104501-none/fg-single3-research)
- [Tutor 1M train](https://wandb.ai/xkm104501-none/fg-single3-research/runs/3dc57d410d9b50601120)
- [Tutor 800K Actor-only DEV](https://wandb.ai/xkm104501-none/fg-single3-research/runs/9832f27f1b038b84fb58)

实际本地 Run：`/data/logdir/fg_single3/s3_v2/TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M_v1`。
源 Run ID `89ab95419f13405aad39881beb299544`；重新发布 namespace
`tutor-cd-reupload-v1`。新 Tracker 目录：
`/data/logdir/fg_single3/experiment_tracking/3dc57d410d9b50601120`。

训练到 **1,000,000 环境步 / 40,000 local updates**。导入 10,000 个 cycle 点与
1,000 个更新聚合点，以及 1 个独立 DEV 点，共 **11,001 个已确认事件**。
完整云端训练历史 11,000 个序号和哈希全部匹配本地不可变 outbox。
SDK 重试跳过 11,000 个已有训练事件，只新发 1 个评价事件。
后续重复 online import 返回空 transport_results，未新增重复点或评价 Run。

实际读回 800K 评价，Parent/Checkpoint/冻结 SHA 一致，Oracle 接管为 false：

| 指标 | 本地认证值 = 云端读回 |
|---|---:|
| 自主 ADD / TP / SL | 7 / 5 / 2 |
| Reward | -3.805870048525721 |
| 闭合交易净盈亏 | -2226 |
| TP 持仓市场分钟 p50 | 74 |
| 首次逆行10点：前5 / 6–15 / 15以后 / 全持仓 | 0 / 1 / 3 / 4 |

这是既有 **800K** 的 FG409 DEV 5000 步评价，并非新做的 1M 评价。
云端仅有 configuration、checkpoint-index、evaluation-summary 和小型 run_table。

## 故障与修复证据

首次 legacy `wandb sync` 超时，实际日志出现 file_stream HTTP 500，云端旧 Run
无已确认历史（last step -1）。用户明确要求删除后，仅删除该云端不完整 Run
`b8f52130d2b181a59b87`；本地原实验和旧 Tracker 导出保留。
SDK 源码注明删除后的 Run ID 不可复用，因此使用显式、稳定的新 namespace。

代理后 legacy bulk 同步仍记录 HTTP 500，随后历史最终可读。
第一次单次读回确认失败；稍后 full scan 与带 keys 的 scan 均读到 11,000 条，
全部哈希匹配。**未证明 HTTP 500 的服务端根因，也未证明采样造成此次失败。**
实现改为完整 HistoryScan、严格序号/哈希确认、有限等待重试及有界错误收据。
Tracker `sync` 复用在线 SDK worker 重放脱敏 outbox，设置 256 KiB file-stream 上限；
保留离线 SDK 文件。此次该 worker 跳过完整训练历史并实际上传评价，
未把“跳过已有训练点”宣称为重新实发 11,000 点的分批性能测试。

CPU/实际 offline SDK 初期暴露 Table summary 在 offline 仍调用服务器能力查询；
已按实际 stack 修复为 offline 保留 JSON 交易小表，online 才创建 W&B Table。
生产 Tutor offline 导出及对应回归均已通过。

## 验证与复现命令

在开发 Gym worktree 执行；完整命令和参数另见 `configs/wandb/README.md`。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/v/Documents/work/research/fg-s3-wandb-env/bin/python \
  -m unittest research_tracking.tests.test_tracker -v
```

最终 **16/16 PASS，9.617 秒**。包含来源/锁/输出边界、半行和缺失前缀、
canonical 与 provisional 回滚、权重/缺失指标、敏感字段、CP 被移除、
重复导入、800K 实际文件、actual offline SDK，以及 Fake Trainer 自动入口。
Fake Trainer 是 CPU 合成 metadata，未作为真实模型训练验收。

```bash
TRACKER_PY=/home/v/Documents/work/research/fg-s3-wandb-env/bin/python
TUTOR_RUN=/data/logdir/fg_single3/s3_v2/TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M_v1
"$TRACKER_PY" tools/fg_s3_wandb.py import --run-dir "$TUTOR_RUN" \
  --publication tutor-cd-reupload-v1 --mode offline
"$TRACKER_PY" tools/fg_s3_wandb.py sync --run-dir "$TUTOR_RUN" \
  --publication tutor-cd-reupload-v1 --mode online --entity xkm104501-none \
  --proxy http://127.0.0.1:10808
"$TRACKER_PY" tools/fg_s3_wandb.py import --run-dir "$TUTOR_RUN" \
  --publication tutor-cd-reupload-v1 --mode online --entity xkm104501-none \
  --proxy http://127.0.0.1:10808
```

实际不可用代理 `http://127.0.0.1:1` 的 bounded follow 返回
`online_status=OFFLINE_FALLBACK`、fallback_count=1，水位及 11,001 个历史事件不变。
恢复正确代理后重复 import 为 `VERIFIED`，fallback_count=0，无新增事件。
这个已结束 Run 没有产生新的未提交尾部；真实运行中 rollback 的验证来自 fixture，
不冒充正在训练的生产回滚验收。

本地真实收据：Tracker 根目录的 `cloud_acceptance.json`、`last_operation.json`、
`tracker_state.json`、各 `transport/result_*.json`；源保护基线
`/tmp/fg_wandb_frozen_baseline_20261014.json`。云端数值由 API 重新读取并断言。
初次上传期间 Tracker 本身为 `ce26db9` + dirty 开发代码，冻结 Trainer SHA 始终
取原身份，未伪装为本次新代码提交。

## PASS / FAIL / NOT_RUN

| 项目 | 状态 |
|---|---|
| CPU/Mock、实际 Tutor 800K 文件、实际 offline SDK | PASS |
| 实际 Tutor 历史和 DEV 云上传、私有 Project、API 逐事件确认 | PASS |
| offline 重放同步、online 同 ID 重连、重复导入幂等 | PASS |
| 真实不可用代理降级 + 恢复连通、保留历史水位 | PASS |
| 新源 Fake Trainer 自动跟踪、原环境/退出码、Tracker 异常隔离 | PASS（CPU/合成源） |
| 原始源码/不可变文件保护、无大文件上传边界 | PASS |
| 首次 legacy bulk sync / 单次即时确认 | FAIL（已记录，最终完整确认通过） |
| 活跃正式 Trainer 的云端 provisional/真实训练恢复回滚 | NOT_RUN（Tutor 已完成） |
| 新 GPU start/resume/evaluate、真实未来 Run 的 launch | NOT_RUN（本任务禁止启动） |
| 新 1M 评价、A2 历史导入、新 GPU 训练、SDK 系统资源补采 | NOT_RUN |

当前无账号/项目授权阻断。未来 Run 通过 README 的 `launch` 模板自动跟踪，
研究预算与正式启动授权仍由原训练入口管理。未增加 Sweep、远程训练或复杂框架。
