# FG Entry V1 最终观测契约（2026-10-10）

按用户最终修订，删除 expiry_remaining，仓位观测为 **9 维**。到期和换主力风险仍按原退出协议处理，期限不进入模型观测。本次不再扩展技术指标，也不恢复其他字段的旧公式。

## 结构与顺序

| Key | Shape | 定义 |
| --- | --- | --- |
| market_seq | (60,18) | 现有 FEATURES_MARKET_OBS_OUTPUT，当前完整分钟及之前 59 根，不足部分补零 |
| position_state | (9,) | 下表固定顺序，float32 |
| action_mask | (2,) | WAIT / ADD_3；WAIT 始终可用 |

| 顺序 | 字段 | 真实来源与计算 |
| --- | --- | --- |
| 0 | direction | 外部人工方向 LONG=+1、SHORT=-1、NO_TRADE=0 |
| 1 | position_ratio | 现有 PositionManager 已成交持仓 size 总和 / 6，与账户手数核对 |
| 2 | pending_ratio | 实际未成交开仓委托预留 / 6；当前同步原子开仓模式无队列，因此为 0 |
| 3 | pos1_size | 投影 1 对应 PositionManager 的真实剩余 size / 3 |
| 4 | pos1_tp_distance | side_sign × (目标执行价 − 当前平仓执行报价) / 5 |
| 5 | pos1_holding_age | log1p(自实际开仓成交以来的自然分钟) / log1p(43200) |
| 6 | pos2_size | 投影 2 对应 PositionManager 的真实剩余 size / 3 |
| 7 | pos2_tp_distance | 与投影 1 同一口径，按该批实际持仓方向计算 |
| 8 | pos2_holding_age | 与投影 1 同一口径 |

不存在的投影三个字段均为 0。TP 距离可为负，不触发 Reward；成交与 Reward 判定仍由原执行引擎处理。执行报价使用现有价差、滑点和 tick 规则：空仓开仓执行价 900、目标 895、当前平仓报价 910 时 distance=(910−895)/5=3。

年龄超过 43200 分钟时可以大于 1；该常数仅为缩放基准，不创建持仓期限。时间元数据直接来自原子成交时记录的 entry_time，实际 size 与 TP 目标直接读取原 PositionManager，不输入原始开仓价格、权益、浮亏金额、累计盈利、历史盈利次数、最大回撤或止损 R 状态。

## 委托与身份核查

当前离线 ADD_3 在下一分钟同步成交完整 3 手或被拒绝；没有可跨分钟等待的开仓委托，不能用剩余额度、退出挂单或 Driver 待执行 action 冒充 pending_ratio。已有部分平仓通过原 PositionManager 部分关闭协议反映真实 size；测试验证 3 手减为 2 手时 size=2/3、总仓位 5/6。当前引擎不支持部分开仓，未为观测另建订单引擎或物理仓位。

每个成交 entry_id 分配一个只用于观测的 obs_slot，在该记录整个生命周期固定。第一批完全退出后第二批保持 pos2，不压缩到 pos1；新批使用空出的 pos1。obs_slot 随原 entry 元数据一起持久化。两个投影不是两个新物理槽，资金、仓位与真实成交仍由原 PositionManager 和账本唯一维护。

## 行情冻结

18 个输出依次为：obs_V_t、obs_I_t、obs_cumVWAP_t、obs_dC_minus_cumVWAP_t、obs_cmp_C_vs_cumVWAP_t、obs_session_high_t、obs_session_low_t、obs_range_t、obs_open_drift_t、obs_bar_dir_t、obs_minute_index_t、obs_session_phase_t、obs_volatility_t、obs_dI_from_yclose_t、obs_pct_chg_from_ref_t、obs_mask_t、obs_weekday_sin_t、obs_weekday_cos_t。

obs_V_t 使用用户指定旧公式：v=log1p(clip(V,0,None))，obs_V_t=tanh(v/5)。其他 17 个字段保持当前已确认实现。内部已有 MACD/ATR 计算不被加入输出或 Dreamer 输入，本阶段不改其业务用途。

## Dreamer 与 A2

原 Dreamer 通过标准 observation dict 接收 position_state，Actor 可以利用仓位信息。因此不再承诺相同市场、不同仓位下的评分绝对不变。后续单独测试：固定行情与模型 carry/随机状态，改变合法仓位观测，对比未经掩码的 WAIT/ADD logits、mask 后概率及合法动作；区分评分变化与动作可用性变化。本阶段不启动训练。

旧 agent_state(4)+entry_state(2,4) 接口移除；观测字段加入环境配置哈希，旧金融检查点明确拒绝与新接口混用。此前 GPU smoke 证明旧接口下训练/恢复链路可运行，不能当作本次新接口的训练结果。

## 验证

36 项环境/行情特征测试通过：空仓形状与字段顺序、多空顺方向距离、负距离无奖励、部分平仓真实手数、退出后不交换投影、再入场与恢复保持身份、超过 30 天年龄不截断，以及原逐笔止盈/Reward 回归。6 项 GPU 环境下的 Dreamer/Gym 接入与账户保存/恢复回归通过；不执行训练。
