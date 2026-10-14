# S3_E_DATA_V1 数学与可见性协议

固定 shape：旧与影子一分钟各 `float32[60,18]`，新增15分钟 `float32[24,7]`。价格锚点为审计 `float64`；可选一分钟 OHLC 审计为 `float32[60,4]`。不得把审计或 manifest 当模型额外字段。

## 来源和完成时间

真实合约、报价单位、原始 OHLCVOI、时区和 dated calendar 在两尺度间完全一致。合约切换重新开始历史；禁止主力连续价和真实合约价混用。沿用 FG 345 市场分钟及预定连续时段，每段按15分钟 `(start,end]` 分桶，逐一要求15个预期唯一合法一分钟槽。休市、周末和日历不允许的时点不会生成补齐行情。

可用性为 `available_at = max(bucket_end, constituent_arrivals)`。只有 `bucket_end <= visible_cutoff` 且 `available_at <= visible_cutoff` 才允许进入上下文。14分钟时不能用最终 H/L/C/V/OI；第15分钟收盘及实际到达后才加入。原 Env 受控 WAIT 推進测试检查这个边界，原 `MarketClock` 决定一分钟有效行窗口。

OHLCVOI 聚合：O=首 O，H=max H，L=min L，C=末 C，V=sum V，OI=末 OI。原价必须有限且大于0，OHLC关系必须合法，V非负，合法分钟OI必须大于0；重复全部拒收。零成交是一根合法分钟，零时长标记不是一分钟。

## 当次决策的共同锚点

优先使用旧观察当前行 `ref_close_t`，但必须找到对应的已观测原始 Close，并证明 source time 和到达时间均不晚于决策 cutoff。旧特征的首日 fallback 只在首个真实分钟已观测后承认；没有合法 ref 证明时使用已经观察到的当前 Close，并记录 `observed_close_fallback`。开盘零时长标记可以提供已知价格，但不能借用之后首根有效分钟 Close。

本窗口中所有15分钟价格：`log(raw_price / P_anchor(t))`，**不对价格列裁剪**。一分钟审计 OHLC 采用完全相同公式。两个高点的对数坐标差等于 `log(H2/H1)`，与共同锚点选择无关。

原18维按原源码计算，包含不同历史行各自的交易日 ref；跨日旧输入不能宣称全窗口同锚。影子视图只重算如下六个参考通道，其他12个通道包括自身相对比值 `obs_open_drift_t` 保留原 float32 字节：

| 通道 | 当前决策共同锚点的公式 | 既有 clip |
|---|---|---|
| obs_cumVWAP_t | log(cumVWAP/P_anchor) | [-1,1] |
| obs_session_high_t | log(session_high/P_anchor) | [-1,1] |
| obs_session_low_t | log(session_low/P_anchor) | [-1,1] |
| obs_pct_chg_from_ref_t | log(Close/P_anchor) | [-1,1] |
| obs_range_t | log1p((session_high-session_low)/P_anchor) | [0,1] |
| obs_dC_minus_cumVWAP_t | sign(Close-cumVWAP)×log1p(abs(Close-cumVWAP)/P_anchor) | [-1,1] |

影子保留既有非线性与 clip，clip 饱和时不可用 exp 精确反推原价；两尺度原价关系由未裁剪的 OHLC 审计验证。`market_seq_e_shared` 是候选数据，不替换生产原18维；模型选择新输入后不能声称旧权重 exact-resume。

## 15分钟量仓与 mask

通道顺序为 `rel_open,rel_high,rel_low,rel_close,vol_relative,oi_change,valid`。

`vol_relative_j = log((V_j+1)/(mean(V_previous_20_compliant_completed_buckets)+1))`。基准只含本桶之前、同一合约、合规完成且当时可见的20个桶。`oi_change_j = log(OI_j/OI_previous_compliant_completed_bucket)`，从不求和 OI。每决策保存实际基准桶 IDs，延迟到达的历史不能追溯改写之前的基准。

量仓 clip 候选为 [-6,6]、[-1,1]；已获用户允许在实际 TRAIN 通过后冻结。这是有限范围和数据分布审查，不是最优阈值研究。最终通过证据见验收报告及产物 schema；没有该证据时只有候选状态。

保留最后24个完整可见桶，最早到最新。前20个量基准不足的桶占据历史位置但整行置零，valid=0；不足24个桶左填整行零。合法 OI 跳变进入诊断，不擅自裁掉真实数据。全部输入必须 finite；不以 nan_to_num 掩盖异常。

## 因果和身份

F03 比较同一历史决策的源前缀、完整源、篡改所有未来 OHLCVOI 三者，检查所有数组和逐时点元数据。全局完整源 hash 可以因未来字节改变而不同，不能将全局 provenance 的变化误认为历史模型输入泄漏。未来 Teacher/Oracle列在入口被丢弃，从未参加特征计算。

schema hash 覆盖公式、字段顺序、shape、clip、时钟约定、八份未修改原 helper 的字节 SHA256，以及 core/schema/reference 的实现 SHA256。产物 manifest 再覆盖实际数组、逐桶、逐决策元数据和来源清单字节。参数/实现变化不得无声覆盖已有目录；每次新目录保留历史证据。
