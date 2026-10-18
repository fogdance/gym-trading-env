# WP04 参数、优化器与恢复证据

实际完整布局 349 个 Ninjax 叶；正向白名单 205 个可变叶 = 67 个 WM 参数 + 134 个原优化器 slots + 4 个共享时钟；其余 144 个逐键 bitwise 冻结。Actor、Critic、slow value、Quality 头、归一化和 Gate 状态都在冻结审计内。原优化器 warmup 首步学习率为 0，使用第二步以后证明变化，不改优化器默认值。

| 参数组 | 实际叶 | 四次更新后变化叶 |
|---|---:|---:|
| `enc/` | 15 | 15 |
| `enc/minute/` | 9 | 9 |
| `enc/context15/` | 2 | 2 |
| `enc/macd5/` | 2 | 2 |
| `enc/fusion/` | 2 | 2 |
| `dyn/` | 27 | 27 |
| `dec/` | 7 | 7 |
| `rew/` | 5 | 5 |
| `con/` | 5 | 5 |
| `avail/` | 8 | 8 |
| `pol/` | 5 | 0 |
| `val/` | 5 | 0 |
| `slowval/` | 5 | 0 |
| `quality_` | 35 | 0 |

以上变化来自有效 15m/MACD 原生合成源；C 最早的 210 分钟恒价样本缺少有效 15m / 非零 MACD，只作为 B 工程证明，不能代替多尺度学习证据。对应完整前后哈希、梯度和每项 Loss 在 [参数摘要](receipts/wp04/E_parameter_summary.json) 及其引用的 C 原始收据。

最终恢复：连续 4 次更新与 2 + cold interpreter + 2 全状态逐字节相同。实际样本 B1/T64/K32；有效目标 256 位置/256 市场分钟、预热 128/128，更新 4，消费批次 4；原 Prefetch 另有一批待消费，Agent batch counter 因而为 5，不冒认第五次更新。Python/NumPy RNG、下一批全部字段/seed、Reader RNG/coverage、schedule、所有 Agent 参数/slots/缓冲和指标都比较。

持久原件：`/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_restore_review_final/wp04_cold0`。见 [最终冷恢复](receipts/wp04/D_cold_restore_evidence.json)、[重新读取原件的审查](receipts/wp04/E_persisted_full_state_audit.json)。旧失败试验使用 pytest 临时目录，不能声称旧临时原件仍存在；其命令、退出码、XML 和日志全部保留。最终原件置于隔离 artifacts，不再依赖临时目录保留期。

Checkpoint 没有 Corpus/NPZ/在线 Replay 副本；runtime.pkl 包含精确恢复需要的一个待消费原生 batch（含输入数组），不能把“无整份 Corpus 副本”写成零行情字节。原 Corpus/index 外部引用在 save/load 重验。合成 Checkpoint 无真实训练来源认证，不可作为 F2/F3 正式基础权重。真实 TRAIN 所有 349 参数/slots 前后哈希相同，更新数 0。
