# FG Replay R0～R2 数据契约

范围及冻结参数见 `FG_Replay_R0_R2_Implementation_Plan.md`。本接口只采集、重放、审计 CPU 数据，不创建 Dreamer Agent。

## 时序与字段

原 `Driver` 先把上次动作交给 Env，取得当前观察/反馈，再询问下一动作。保存的是 `obs_t, action_t, reward_t`，其中 `reward_t` 属于已经执行的前一个动作；不能把它解释为 `action_t` 的即时结果。

| Replay 字段 | 单条形状 | dtype |
|---|---|---|
| market_seq | (60,18) | float32 |
| position_state | (9,) | float32 |
| action_mask | (2,) | float32 |
| action / reward | 标量 | int32 / float32 |
| is_first / is_last / is_terminal | 标量 | bool |
| stepid | (20,) | uint8 |

采样形状为 `(batch_size,65,...)`，前1位是 replay_context，其余64位分别统计。原 Replay 在 sample 时将第0位 `is_first` 设为 True，可能补充 `is_last`；这些采样标注不是金融账户重新初始化的证据。磁盘字段保留真实 Driver 标志。

## 账户与边界

- 每条策略/合约独立运行一个原 Env、Wrapper、Driver 和 Replay 实例；独立 chunk UUID 链隔离账户。
- Env 的1024步截断返回 `is_last=True,is_terminal=False`；Wrapper 续接输出相同游标的 `is_first=True`、零Reward观察，账户资金、仓位、TP去重状态继续。
- Env 截断与 Replay 的1024行存储 chunk 不对齐：初始和续接观察也占 Replay 行。分别记录两种跨界事件。
- `is_last` 行的输出 action 是 Driver 清零的占位；不会作为下一次真实决策执行。Sidecar 在见到下一行后确认 `decision_executed`，避免把占位或数据结束请求误计入决策。
- `DATA_END/DATA_GAP` 保留真实持仓；`pending_reason` 区分右删失与已经触发但未成交的强退。不得重置成干净账户补齐后续数据。

## Sidecar 与身份

每条轨迹保存 `steps.jsonl.gz`、`account_state.json`、`trade_diagnostics.json`、原生 Replay chunks、哈希 manifest 和逐步对账 receipt。

Sidecar 每行对应一条真实 Replay 记录，用20字节 `stepid` 的十六进制字符串精确关联 chunk UUID 和行偏移。字段包括当前游标、信号/观察/执行时间、输入/输出动作、真实Mask、决策是否推进、成交、归因、Reward事件、活跃持仓、标记价、截断和早期逆行诊断。所有非模型字段均在 Sidecar，不改变 Obs。

`replay_trace_hash` 由冻结数据/执行代码身份、策略/种子、Env配置、原字段和语义事件内容组成，排除物理 UUID、路径、保存时间、用时。重建可对比该哈希；物理文件哈希用于检查同一份冻结语料未被修改。重建的随机 UUID 不保证相同。

V1 Reward 仍为原 Env 的 `60/(60+T_minutes)`，数量3、TP5点。离线诊断器的 `v0` 只作为逐步V1核验，其他候选Reward结果不写入 Replay。早期逆行按现有诊断器首次反向10点、前5/6～15根有效bar口径；同bar TP与逆行的顺序歧义单列。MAE使用已观察OHLC的保守上界，不声称真实触价顺序。

## 运行与复用

使用 conda `dreamerv3` 的 Python，在 gym-trading-env 根目录运行。CLI 在导入 Embodied/JAX 前清空CUDA可见设备并固定JAX CPU。

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_replay.py generate \
  --config configs/fg_entry_v1/replay_r0_r2.yaml \
  --output artifacts/fg_replay_r0_r2/v1/FG409/all/run_20261011_01 --workers 4

/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_replay.py audit \
  --dataset artifacts/fg_replay_r0_r2/v1/FG409/all/run_20261011_01 \
  --output artifacts/fg_replay_r0_r2/v1/FG409/all/audit_20261011_01

/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_replay.py reconcile \
  --dataset artifacts/fg_replay_r0_r2/v1/FG409/all/run_20261011_01 \
  --output artifacts/fg_replay_r0_r2/v1/FG409/all/reconcile_20261011_01
```

已有输出拒绝覆盖；`generate --resume` 仅在冻结身份完全一致时复用通过对账的尝试。失败尝试保留，新尝试目录从账户真实起点重新运行，不伪造中间恢复。

生成先写 `manifest.pending.json` 并执行完整数据校验，通过后原子重命名为正式 `manifest.json`。最终校验失败时保留候选 manifest、轨迹和对账凭据，不发布 dataset 完成标记。审计在 CPU 检查及最后一次来源校验通过后才写 `replay_coverage.md` 的 PASS 结论和审计 `manifest.json`；上述检查失败时仅保留已产生的计数与诊断数据。

Dataset manifest 的 `views` 提供各策略和 `all` 的平铺原生 `.npz`目录；这些是原始chunks的硬链接，没有重新串接账户。调用原 `Replay.load(directory=...)` 即可读取。加载器的构造必须提供一个 directory（该版本原Replay的加载依赖此初始化）；后续训练应提供独立的新运行目录，不向冻结来源保存。

```python
# 已在导入 Embodied 前设置 CUDA_VISIBLE_DEVICES=''、JAX_PLATFORMS='cpu'。
import embodied
reader = embodied.Replay(length=65, capacity=sequence_starts,
                         directory=new_runtime_directory, online=False, seed=202)
reader.load(directory=frozen_replay_view)
assert len(reader) == sequence_starts
batch = reader.sample(16)
reader.workers.shutdown(wait=True)
```

容量是有效连续序列起点数，不是chunk数量。R2按整份语料设置容量并断言加载数，禁止最近数据静默替代完整历史。源文件/代码、Obs/Reward版本和账户对账receipt都核验后才登记复用。模型seed与采样seed独立，同一冻结语料可以被不同模型seed共享。

冻结身份包含生成器源码文件哈希。更新源码后，直接核验旧语料会触发 `Frozen input/output changed`；旧语料应使用其登记的源码版本核验，或在新输出目录生成新身份的语料。不要修改或重新签名旧哈希来绕过这一检查。

## 验收证据

测试覆盖原Driver逐步对账、跨1024后TP与账户恢复、原生保存/加载/跨chunk/账户隔离、canonical重建、固定采样复现、反馈错位拒绝、P3空额计时、P2 Mask因果选择、训练哈希和实际交易日隔离，以及未成交强退与右删失区别。

正式72条账户的对账、存储量和实际batch结果见生成产物的 `replay_coverage.md` 及本任务实现验证报告。模型学习表现、GPU训练和Actor监督实验不属于本次验收。
