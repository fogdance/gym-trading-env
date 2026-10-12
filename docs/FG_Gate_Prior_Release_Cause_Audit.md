# Prior 仓位释放与 Gate 根因审计

审计来源固定为 Online `step_000000180000`，183 个文件全部通过 FullCheckpoint 校验，manifest SHA256 为 `4979acfb0272ffbc3c96833e5b77fccd66c5d4d0f335bfca48810f764d927a57`。检查开始时 Gym HEAD=`06dd2d3`、Dreamer HEAD=`0b3a641`；重建源码指纹证明源 Checkpoint 对应 Dreamer `09e2e5c`，其后确有策略参数同步修复，不能将当前源码当作旧身份精确恢复。

## 已观察事实与确认根因

原 `Agent.loss` 以 `logit>=0` 计算 Post/Prior 质量，Imagination 使用 `agent.avail_threshold`。原运行阈值同为0.5，故此前日志没有因两处数值不同而虚报；单独下调配置会使 Gate 仍按0.5计算，接线必须修复。

`RSSM._observe` 先由上一 posterior、上一真实动作计算本帧 deter，再结合本帧 tokens 得到 posterior；`RSSM.loss` 的 prior 由该 deter 计算。Prior 没有吸收当前行情/仓位释放观察。Encoder 包含当前真实 `action_mask`，因此高 Post Recall 不能独立证明 Prior 能预测释放。

对 FG009 的整个已训练路径使用现有 `FixedDirectionEntryEnv` 和真实历史 Action 重新执行，逐帧比较 `market_seq`、`position_state`、`action_mask`、Reward、is_first/is_last/is_terminal，并比较完整财务导出状态。结果：20,891 Replay 行、20,870 次行情推进、21个 reset 全部一致；104次退出为91次TP、13次SL。未观察到其他退出原因，不能杜撰其对账证据。资金、账本、成交、归因、Reward状态、未平仓位置和历史权益等39个字段精确一致；重放初始化的随机状态另行记录，不冒充原账户 RNG，原 RNG 的保留由完整恢复测试验证。

本路径没有发现时序、Action、Observation、Reward 或 Reset 错位。确认的学习不足是当前模型在释放正例中的漏判；不能据此断言所有模型都无法预测退出，也不能将尚未观测到的本帧价格塞给 Prior。

## 具体交易证据

| 事件 | Replay索引 / cursor | stepid | 上一帧→本帧 | 真实执行与Reward |
| --- | --- | --- | --- | --- |
| SL30 |159168 / 59|`1190a940d8ad456ba1991519e2e89a81000001c0`|09:59持仓、ADD非法 → 10:00空仓、ADD合法|上一Action=WAIT；本帧Low=1247触及1277−30，平3手，净PnL=−1818，Reward=−3|
| TP5 |159233 / 124|`1190a940d8ad456ba1991519e2e89a8100000201`|11:19持仓、ADD非法 → 11:20空仓、ADD合法|上一Action=WAIT；本帧High=1259，按1252+5平3手，净PnL=282，Reward≈0.412127|

159233 行记录的 ADD 是在11:20已知观察之后选出的动作，在159234行对应下一 bar 开仓；它没有产生159233行的TP或Reward。全部104笔事件都有退出前后各两帧、cursor、stepid、OHLC、成交归因与Reward，见 `cause_verified/fg009_exits.json`；完整逐帧明细见 `fg009_frames.csv`。唯一上一帧空仓而本帧已有完整退出的事件索引166604，已归入空仓ADD后仍空仓类别，未当作持仓释放。

## 正例分层与窗口统计

沿用原审计的179,968个唯一评分位置，真实正例分组互斥且完备，逐行断言每条只落入一组：519持续空仓WAIT + 487持仓后释放 + 3空仓ADD后仍空仓 + 7reset = 1,016。这些分类不是退出原因标签。

旧日志含115个Gate报告，0个完整合格窗口，最大连续次数0；首窗正例不足，其余114窗有足够支持但质量不通过。最新报告到184,000；源Replay与完整账户仅到180,000。两种位置不能混为同一状态。

`train.py` 原逻辑对每批 ADD TP/positives 求和后计算 Recall，未发现平均稀疏 batch Recall 的问题。相同批长度下 Accuracy/Exact 的均值等于总体值；原 FPR 是每批比例的平均，当前修复保留其 Gate 口径，同时补充 FP/negative 计数供核对。旧日志未记录每窗FP/negative原始计数，标记 NOT_RECORDED，不能从显示后的比例反推精确整数。新 CPU 窗口直接记录这两项。

源码依据：`dreamerv3/agent.py` 的 Encoder、`loss`、`_apply_replay_context`、`_predicted_action_mask_info`；`dreamerv3/rssm.py` 的 `_observe/loss`；`embodied/core/driver.py::_step`；`entry_env.py::_can_add/_get_obs/step`；`ActionMaskWarmup.for_single3_1m` 与 `train.py` 的 warmup 聚合。具体行号和差异保存在实施报告的源码证据中。

## 复现与限制

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_single3_gate_cause_audit.py \
  --source /data/logdir/fg_single3/online_after_10k_callback_fix/C1_M1_ONLINE_1M/ckpt/step_000000180000 \
  --audit artifacts/fg_single3_v1/diagnostics/prior_probability_audit_20261012_02 \
  --output artifacts/fg_single3_v1/diagnostics/gate_cause_recheck
```

实测41.1秒；输入、输出和Checkpoint哈希见 `artifacts/fg_single3_v1/diagnostics/gate_repair_20261012_01/cause_verified/`。此次只读取 TRAIN 价格，未读取 DEV/TEST，没有启动GPU训练。
