# 本轮沿用的 E 模型接口

本轮只发布经过真实验收的组合准入；模型实现未改变，原始完整说明见 `docs/S3_E_MODEL_V1_INTERFACE.md`。

| 输入 | dtype / shape（不含 B / T） | 职责 |
| --- | --- | --- |
| `market_seq_e_shared` | float32 / 60×18 | 原 Env 同一行情生成的 E 六通道重锚定输入，经 minute 三层 MLP |
| `market_context_15m` | float32 / 24×7 | 六个有效值+valid，flatten168→Linear128+GELU；无效桶数值归零 |
| `position_state` | float32 / 9 | minute 分支原仓位输入 |
| `action_mask` | float32 / 2 | minute 分支原合法动作输入 |
| 旧 `market_seq` | float32 / 60×18 | 审计保留，不编码或重建 |

size50m 的 minute 输出512与context128拼接，经 fusion Linear512，返回原 RSSM 接受的 `(*reset.shape,512)` tokens；policy 为 B、world model 为 B/T。Encoder 无持久 carry。Decoder 只重建 shared1m/position，不重建15m、mask或旧market_seq。ObsSpace、Replay schema 与来源 binding 的精确字段列表见各 prepared_source_admission / acceptance 收据。

参数子树 `enc/minute/`、`enc/context15/`、`enc/fusion/` 均进入原优化器、policy 参数同步与 FullCheckpoint。真实 CPU/GPU 更新、非零梯度与冷恢复证据见本目录 REPORT/索引。teaching 轴独立选择 A2、Tutor-CD、B2；联合 Tutor-CD+B2 仍未实现。
