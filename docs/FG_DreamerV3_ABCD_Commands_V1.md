# FG ABC 开发与训练命令 V1

日期：2026-10-10；2026-10-11更新。旧 v3_shared_prefix GPU smoke/先导已完成，正式未运行。

ADD Gate P0 补丁已订正，但新学习协议的完整身份与旧 v3 节点不同。本页 v3 恢复命令保留作历史流程参考，不能用于强行继续旧先导；旧产物只读保留。新身份下的运行准备及模型验收仍待完成，正式仍由用户启动。详见 FG_DreamerV3_ADD_Gate_P0_Review_and_Fix_V1.md。

## 环境和只读准备检查

```bash
conda activate dreamerv3
cd /home/v/Documents/work/gym-trading-env
export PYTHONPATH=/home/v/Documents/work/dreamerv3:/home/v/Documents/work/gym-trading-env/src:/home/v/Documents/work/gym-trading-env
export MPLCONFIGDIR=/tmp/fg-matplotlib
python tools/run_fg_abcd.py --budget pilot
python tools/run_fg_abcd.py --budget formal
python tools/report_fg_abcd.py
python tools/watch_fg_abcd.py --once
```

上述命令只打印命令或刷新本地报告，不运行模型。新清单已经生成，`python tools/build_fg_abcd.py` 再次执行会拒绝覆盖。配置唯一入口是 configs/fg_entry_v1/abcd_comparison.yaml；方向、交易费用和时钟继续由 fg.yaml 统一维护。

## 训练线路与固定调度

Gate/报告每 5000 环境步，完整恢复点每 10000 步；包含 reset 观察。每个 fold/seed 的目录如下：

| 目录（位于 /data/logdir/fg_abcd_v1/v3_shared_prefix/FG409/0/） | 用途 |
|---|---|
| all_history | A/C 公共全历史前缀，先导 100K、正式至 1M |
| recent_from_scratch | B，先导 100K、正式至 1M |
| C_recent_from_70000 | C 先导，从 70K 节点近期训练 30K |
| C_recent_from_700000 | C 正式，从 700K 节点近期训练 300K |

A 的正式 700K 节点为 `all_history/ckpt/step_000000700000`，100K/1M 是总目标，不是恢复后再加的预算。先导 A/B 可以一致性校验后继续，C 先导不替代正式近期阶段。旧轮转结果不复用。

恢复相同目录即可；FG_FULL_V1 校验完整学习/数据/代码/运行环境身份，不能用普通 `from_checkpoint` 只载入模型。常规点保留最近两个，分支点和最终点永久保留；不会清理旧版本。保存失败不影响已有有效点，损坏点允许回退到有效点；全部失效时拒绝从零重算。

## 后续授权后的 GPU smoke 与先导

以下是后续启动命令，本轮未执行。必须先完成新协议的 GPU smoke，旧 smoke 不替代新协议验收：

```bash
python tools/run_fg_abcd_smoke.py
python tools/audit_fg_abcd_smoke.py
python tools/run_fg_abcd.py --budget pilot --fold FG409 --seed 0 --execute
```

Smoke 总计 2040 交互，包含 1110→1120 同目录恢复及 920 步近期迁移，GPU-only、正式 batch16×length64。保存最终 checkpoint 的同时保存完整 Replay；近期未到学习阈值时，权重/优化器应与源节点一致。不能把 smoke 当作策略盈利验收。

先导按 A→B→C 串行，C 读取 A 的 70K 受保护节点，不再次预训练。可按 `--method A/B/C` 单独操作，已有匹配节点复用、中断运行恢复；不兼容则拒绝。Actor Gate 未开启时不宣称策略学习成功，不降低门槛、不修改 Reward。

## 正式命令模板

```bash
python tools/run_fg_abcd.py --budget formal --fold FG409 --seed 0
```

输出 A 全历史至 1M、B 近期至 1M、生成 C 收据、C 近期 300K 的模板；C 的 700K 前缀不重复打印为第二条训练命令。其余三个 fold 与 seed1/2 同理。

正式 runner 支持用户显式 `--budget formal --execute`，默认只打印模板。本轮不会执行正式命令；建议完成新版 GPU smoke 和先导诊断后再启动。

C 源收据准备（不训练，要求源节点已存在）：

```bash
python tools/run_fg_abcd.py --budget formal --method C --fold FG409 --seed 0 --create-transfer
```

收据在近期目录 `source_transfer.json`，指向受保护的精确源节点及该节点冻结的 config.yaml，不依赖 A 的 latest 指针。普通 C 中断恢复自身目录，不重复初始化账户；来源数据、参数、代码或依赖改变时停止。

## 日志与评估

```bash
tensorboard --host 0.0.0.0 --logdir /data/logdir/fg_abcd_v1/v3_shared_prefix
```

评估只使用开发验证合约与独立账户。报告使用阶段专属记录，避免先导共享目录的状态被误写为正式状态。学习审计及覆盖报告取阶段精确节点，不能用 A 的 1M 结果代表 C 的 700K 前缀。相同 seed 的 A/C 是配对结果。

生成配置、数据、报告、日志、Replay、checkpoint 均不提交。当前 readiness_audit.json 为 PREPARED_NOT_RUN，最终盲测仍封存。大 Replay 保存耗时、磁盘增长和真实 GPU 恢复尚未验证。

正式训练由用户启动（本轮未执行）：

```bash
python tools/run_fg_abcd.py --budget formal --execute
```

该命令按共享节点组织 36 组正式实验，已有兼容先导 A/B 优先恢复。默认省略 `--execute` 只输出命令；C 的迁移凭据和源节点校验由编排器完成。
