# FG 5 点止盈历史行为研究：开发区结果与实施记录 V1

日期：2026-10-10。状态：开发区描述性研究完成；未训练 DreamerV3，未解封前向盲测。

## 1. 可以确认的结论

1. 数据中的固定 5 点目标，在不同历史时期具有不同的观测触达速度；同时原始每分钟价格范围和已有 Obs 状态分布也发生变化。不能据此直接认定模型择时规律失效或主力控盘机制变化。
2. OHLC 触价与保守模拟成交存在明显差异。开发验证区 LONG 60 分钟内触价约 45.40%，保守成交约 33.35%；SHORT 分别约 47.07% 与 34.87%。不能拿触价比例冒充正式策略的成交成功率。
3. 2018、2019 的历史 23:00 后价格已按用户要求截断。研究不读取这些价格，跨越已知不可观测区间时删失，因此早期年度上界较宽。不能把这些删失直接计为失败，也不能将早期删失率与后期零删失率的差异解释成行情机制变化。
4. 12 个月窗口的人工方向覆盖并不稳定：2022-12-09 截止的 12 个月全部为 SHORT；2023 年主要为 LONG。窗口实验必须同时列出方向覆盖，否则模型差异可能来自训练方向而非历史有效期。
5. 尚未确认哪一个训练窗口最佳。这里给出后续实验设计，不承诺 DreamerV3 学会等待，也不启动训练。

## 2. 数据、边界及实际执行

审计输入是当前 345 分钟主力导出，24 合约、632175 根；事件计算仅用盲测之前的 19 合约、519195 根。训练截至 2022-12-09；开发验证截至 2024-08-19。2024-08-20 至 2026-10-09 保留前向盲测，其中 2026 尚未追加本地研究导出。

按“真实合约 × 训练/开发验证分区 × 3 种执行口径”拆成 57 个任务，以 4 个 spawn 进程执行，每个进程原生线程数为 1。首轮 worker 计算实际耗时 242.327 秒。完整数据与拒绝候选压缩后约 301.4 MB。再次执行相同命令，57 个任务全部验证哈希后续跑跳过，worker 阶段耗时 0.961 秒；没有重新计算事件。

每个方向及每种口径均有 518634 个实际入场候选，合计 3111804 条事件；另有 3366 条拒绝候选。拒绝记录不计入止盈分母。不同模式属于同一批候选的敏感性分析，不能当作三份独立统计样本。

| mode | success | terminal_failure | right_censored | no_entry |
| --- | --- | --- | --- | --- |
| touch_spread | 821194 | 39375 | 176699 | 1122 |
| conservative_spread | 777831 | 51975 | 207462 | 1122 |
| touch_zero_spread | 848378 | 32606 | 156284 | 1122 |

primary=`touch_spread`：实际模拟开仓 LONG Open+1 / SHORT Open−1，目标顺方向 5 点，仅记录 OHLC 触价；有触价不代表完整三手委托成交。

conservative=`conservative_spread`：沿用既有 Env 单笔撮合约定，入场根只用 Close，后续价格须扣除退出价差后超过目标一个 tick，成交量覆盖完整三手，入场当根同时预留开仓数量；保证金和主力终局优先于未知先后顺序的 TP。以独立新账户单笔探针回放核对，不能等同于多批加仓后的组合风控。

zero spread=`touch_zero_spread`：零价差敏感性，业务 Reward 公式没有改变。

成功理论 Reward=60/(60+自然分钟)，主力资格终局未成功=0，删失为缺失值。不可执行候选单独输出。成功等待时刻为触价根 eob 上界；已知跳空在 Open 满足目标时使用 bob，并排除该根后续最差价格对 MAE 的污染。自然时间跨休市继续计时；有效根数另列。

## 3. 开发验证区速度统计

下表 5m/15m/60m 为自然分钟，1day/5day 为实际交易日收盘。验证区在主力到期前拥有完整当前研究口径随访，无右删失，所以该区间上下界相等。入场时段差异另见 session 分组；一分钟事件重叠，n 不是独立样本数。

| mode | direction | n | 5m_lower | 15m_lower | 60m_lower | 1day_lower | 5day_lower | mean_success_wait | mean_known_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative_spread | LONG | 139885 | 0.0439 | 0.1506 | 0.3335 | 0.6076 | 0.8932 | 2709.4053 | 0.3220 |
| conservative_spread | SHORT | 139885 | 0.0497 | 0.1612 | 0.3487 | 0.6128 | 0.8931 | 2946.8820 | 0.3311 |
| touch_spread | LONG | 139885 | 0.1029 | 0.2616 | 0.4540 | 0.6973 | 0.9189 | 2063.4080 | 0.4228 |
| touch_spread | SHORT | 139885 | 0.1106 | 0.2713 | 0.4707 | 0.6988 | 0.9202 | 2200.9222 | 0.4323 |
| touch_zero_spread | LONG | 139885 | 0.1612 | 0.3433 | 0.5301 | 0.7468 | 0.9334 | 1713.4763 | 0.4896 |
| touch_zero_spread | SHORT | 139885 | 0.1698 | 0.3542 | 0.5447 | 0.7450 | 0.9339 | 1832.9397 | 0.4979 |

mean_success_wait 只包含成功样本且以自然分钟计，长尾持仓会显著拉高均值。mean_known_reward 包含已知终局零奖励但不把删失填零；E[R(T)] 不能用 R(E[T]) 替代。完整时间、MAE 上界、Reward 分布见分布 CSV，不仅比较均值。

这些是重叠独立探针。即使单笔止盈净收益可计算，也不能将所有探针的 account_net_pnl 相加当作可执行账户净收益。成本为用户常数配置，不冒称逐年交易所实际费率。

## 4. 年度比较与缺失价格影响

2018 仅自 4 月开始，2024 仅截至 8 月 19 日；2022 分为训练主体与 12 月开发验证尾段。年度表不能无提示合并为完整同等长度年份。

| role | direction | group | n | contracts | 60m_lower | 60m_upper |
| --- | --- | --- | --- | --- | --- | --- |
| train | LONG | 2018 | 60871 | 3 | 0.1364 | 0.3103 |
| train | LONG | 2019 | 82465 | 4 | 0.1240 | 0.2778 |
| train | LONG | 2020 | 75551 | 4 | 0.3609 | 0.3609 |
| train | LONG | 2021 | 82512 | 4 | 0.6369 | 0.6369 |
| train | LONG | 2022 | 77350 | 3 | 0.4572 | 0.4576 |
| train | SHORT | 2018 | 60871 | 3 | 0.1358 | 0.3091 |
| train | SHORT | 2019 | 82465 | 4 | 0.1122 | 0.2703 |
| train | SHORT | 2020 | 75551 | 4 | 0.3104 | 0.3104 |
| train | SHORT | 2021 | 82512 | 4 | 0.6070 | 0.6070 |
| train | SHORT | 2022 | 77350 | 3 | 0.5100 | 0.5107 |
| validation | LONG | 2022 | 5174 | 1 | 0.4161 | 0.4161 |
| validation | LONG | 2023 | 82767 | 4 | 0.4723 | 0.4723 |
| validation | LONG | 2024 | 51944 | 2 | 0.4285 | 0.4285 |
| validation | SHORT | 2022 | 5174 | 1 | 0.4009 | 0.4009 |
| validation | SHORT | 2023 | 82767 | 4 | 0.4726 | 0.4726 |
| validation | SHORT | 2024 | 51944 | 2 | 0.4745 | 0.4745 |

lower=已确认在时限内成功/n；upper=(时限内成功+时限内尚未解决的删失)/n。这个区间是可识别范围，不是置信区间。若成功晚于时限，或真实终局未成功，则在该时限下已知未成功；删失随访不足的样本保留未知。

例如 2020 LONG 60 分钟触达为 36.09%，2021 为 63.69%；这是该数据和模拟约定下观察到的频率差异。年度只有 2–4 个真实合约，不能把数万分钟当作独立试验给出极窄显著性结论。

## 5. 交割月、季度和相关性控制

| role | direction | group | n | contracts | 60m_lower | 60m_upper | 60m_cluster_ci_low | 60m_cluster_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | LONG | 01 | 143677 | 5 | 0.3637 | 0.4283 | 0.1831 | 0.5460 |
| train | LONG | 05 | 95979 | 4 | 0.3524 | 0.4035 | 0.1231 | 0.5737 |
| train | LONG | 09 | 139093 | 5 | 0.3424 | 0.4079 | 0.1631 | 0.5338 |
| train | SHORT | 01 | 143677 | 5 | 0.3665 | 0.4317 | 0.1909 | 0.5815 |
| train | SHORT | 05 | 95979 | 4 | 0.3383 | 0.3891 | 0.1089 | 0.5596 |
| train | SHORT | 09 | 139093 | 5 | 0.3263 | 0.3938 | 0.1399 | 0.5223 |
| validation | LONG | 01 | 26789 | 1 | 0.5275 | 0.5275 | 0.5275 | 0.5275 |
| validation | LONG | 05 | 53323 | 2 | 0.4024 | 0.4024 | 0.3270 | 0.4791 |
| validation | LONG | 09 | 59773 | 2 | 0.4670 | 0.4670 | 0.4079 | 0.5308 |
| validation | SHORT | 01 | 26789 | 1 | 0.5095 | 0.5095 | 0.5095 | 0.5095 |
| validation | SHORT | 05 | 53323 | 2 | 0.4455 | 0.4455 | 0.3663 | 0.5262 |
| validation | SHORT | 09 | 59773 | 2 | 0.4756 | 0.4756 | 0.4471 | 0.5065 |

完整季度、日/夜盘和波动环境分组位于 group_statistics.csv，维度为 quarter/session/volatility。每行包含 n、真实合约数、交易日数、终局/删失数量、5 个时限的上下界与合约聚类区间。

合约 bootstrap 1000 次、seed=0，抽取整合约贡献；每个时限的 *_cluster_ci_low/high 是“成功率下界的 2.5% 分位、上界的 97.5% 分位”组成的描述性包络。它不是每分钟独立二项区间，也不是 AJ 曲线的置信带。少于 6 个合约标记 insufficient_contract_clusters；验证区只有 5 个合约，所有年度更少，不能做强显著性推断。

另提供 5/20 交易日循环移动块敏感性，保持合约边界，只重采样日级聚合，不拼接或重排价格轨迹。它与合约抽样是两项互补敏感性检查，不冒称已覆盖跨合约共同宏观冲击的全部依赖。

Aalen–Johansen 曲线把止盈和真实终局失败作为竞争事件、数据或窗口删失作为删失。该估计依赖非信息性删失；历史夜盘截断可能违背这一假设，所以主结论优先采用可观测界限，AJ 仅作为带假设的描述性辅助。

## 6. 整体价格波动与已有状态分布

原始价格尺度及 K 线范围统计如下。这些仅为研究描述，未加入 Obs，也未增加任何技术指标。

| year | bars | contracts | trading_days | mean_price | mean_bar_range | median_bar_range | p90_bar_range | mean_abs_open_close |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018 | 61124 | 3 | 179 | 1381.6296 | 1.3585 | 1 | 2 | 0.6573 |
| 2019 | 82752 | 4 | 244 | 1407.9664 | 1.3065 | 1 | 2 | 0.6285 |
| 2020 | 75555 | 4 | 243 | 1604.6067 | 2.0949 | 2 | 4 | 1.0568 |
| 2021 | 82515 | 4 | 243 | 2266.7672 | 4.0098 | 3 | 7 | 2.1980 |
| 2022 | 82528 | 4 | 242 | 1698.1826 | 2.8217 | 2 | 5 | 1.4623 |
| 2023 | 82770 | 4 | 242 | 1663.9301 | 2.6848 | 2 | 4 | 1.3905 |
| 2024 | 51945 | 2 | 153 | 1590.6601 | 2.3723 | 2 | 4 | 1.2490 |

2019 平均每分钟 High−Low 为 1.3065 点，2021 为 4.0098 点，2023 为 2.6848 点。价格水平也不同：固定 5 点并非固定百分比。因此仅比较归一化波动特征，不足以完全控制绝对 5 点任务的难度。

已有 obs_volatility_t 年度均值从 2019 的 0.0650 到 2021 的 0.1277、2023 的 0.1112；这是当前压缩特征的变化，不是直接宣称真实波动率翻倍。全部 18 个字段的年度均值、标准差和固定参考分箱分布分别输出 feature_moments.csv / feature_distribution.csv。

## 7. 条件行为分析

分箱参考仅取最早三个开发合约 CZCE.FG809, CZCE.FG901, CZCE.FG905 的市场特征，不使用后续年度或盲测结果拟合分箱。波动分界为 [0.0553124776139961, 0.0756103658749478]。主条件为交割月、日/夜盘、既有波动档位及 open_drift 符号，方向分别研究。其余 18 维分布全部列出，但不冒称已联合控制完整 18 维状态。

参考权重固定。某年缺少某状态或样本不足 100 时，其参考权重继续保留为未知 [0,1]，不重新归一化为该年另一个市场状态总体。reference_weight_covered 列出支持覆盖比例。

| direction | year | reference_weight_covered | standardized_60m_lower | standardized_60m_upper | conclusion |
| --- | --- | --- | --- | --- | --- |
| LONG | 2018 | 0.9596 | 0.1258 | 0.3360 | descriptive_only_not_full_18D_control |
| LONG | 2019 | 0.9974 | 0.1326 | 0.2888 | descriptive_only_not_full_18D_control |
| LONG | 2020 | 0.9672 | 0.2893 | 0.3221 | descriptive_only_not_full_18D_control |
| LONG | 2021 | 0.8317 | 0.4888 | 0.6571 | descriptive_only_not_full_18D_control |
| LONG | 2022 | 0.7922 | 0.3006 | 0.5088 | descriptive_only_not_full_18D_control |
| LONG | 2023 | 0.8685 | 0.3450 | 0.4765 | descriptive_only_not_full_18D_control |
| LONG | 2024 | 0.5185 | 0.2090 | 0.6905 | descriptive_only_not_full_18D_control |
| SHORT | 2018 | 0.9596 | 0.1232 | 0.3272 | descriptive_only_not_full_18D_control |
| SHORT | 2019 | 0.9974 | 0.1183 | 0.2786 | descriptive_only_not_full_18D_control |
| SHORT | 2020 | 0.9672 | 0.2330 | 0.2658 | descriptive_only_not_full_18D_control |
| SHORT | 2021 | 0.8317 | 0.4328 | 0.6011 | descriptive_only_not_full_18D_control |
| SHORT | 2022 | 0.7922 | 0.3530 | 0.5612 | descriptive_only_not_full_18D_control |
| SHORT | 2023 | 0.8685 | 0.3465 | 0.4780 | descriptive_only_not_full_18D_control |
| SHORT | 2024 | 0.5185 | 0.2428 | 0.7243 | descriptive_only_not_full_18D_control |

2021 的状态标准化范围与 2020 并不相同，但仍受到粗分箱、价格水平、合约/交易日相关性及有限支持影响。2024 的参考支持仅约 51.85%，且是部分年度；宽区间本身说明不宜硬比较。当前证据支持进一步做样本外窗口实验，不足以认定在严格相同状态下市场机制已改变。

## 8. 输出位置及复现

当前事件目录：`artifacts/fg_history_study/v1/events/1fd2a4e5ca616090`。
当前有效统计目录：`artifacts/fg_history_study/v1/events/1fd2a4e5ca616090/statistics/7165dc58dfeb74cf`。其他哈希目录为保留的旧试运行/旧统计版本，不自动混用。

- 逐候选：每任务 events.csv.gz / rejected.csv.gz，包含必需字段、时间与 MAE 边界、删失/终局原因、单笔模拟账务、决策时刻 18 个 Obs、来源哈希。
- 数据覆盖：artifacts/fg_history_study/v1/coverage.json、coverage.csv、internal_gaps.csv。
- 聚合：group_statistics.csv、competing_risk_curves.csv、day_block_sensitivity.csv。
- 分布：duration_distribution.csv、mae_upper_distribution.csv、reward_distribution.csv。删失等待时间分布是观测随访，不冒称最终首次触达时间；删失 MAE 也是部分观察。
- 条件：conditional_standardization.csv、feature_moments.csv、feature_distribution.csv。
- 价格与窗口：market_profile.csv、window_coverage.csv。
- 图表：annual_LONG/SHORT、touch_curves、feature_changes、conditional_changes，各有 PNG/SVG。特征图阴影为描述性标准差，不是置信区间；触达年度图阴影为删失上下界。

```bash
conda activate dreamerv3
PYTHONPATH=src python tools/fg_history_study.py audit
PYTHONPATH=src python tools/fg_history_study.py audit-features
PYTHONPATH=src python tools/fg_history_study.py events --workers 4
PYTHONPATH=src python tools/fg_history_study.py statistics --run artifacts/fg_history_study/v1/events/1fd2a4e5ca616090
PYTHONPATH=src python tools/fg_history_study.py window-coverage
PYTHONPATH=src python tools/fg_history_study.py market-profile
```

工具由 CLI 指定区间封存，不提供 blind 事件开关。研究源代码、配置、测试及文档提交；生成的 CSV/JSON/PNG/SVG 留在忽略目录，不进 Git。

## 9. 验证和限制

19 项事件/并行/审计回归、2 项统计回归、1 项覆盖工具回归通过；另有 48 项真实数据前缀检查通过。统计回归包括终局竞争事件、删失不填零、整合约抽样，以及第五交易日夜盘尚未到该日收盘的时间判断。实际 57 任务全量执行、全量哈希续跑及图表查看已完成。

正式 Env、Obs、Action、Reward 和 DreamerV3 未修改；未启动模型训练。独立交易所历史日历尚未完整审计；整段缺失与历史正常无夜盘仍有识别限制。方向配置中 FG405 在 2023-12-08 至 2023-12-22 的 3795 根为 NO_TRADE，保留人工维护结果，不自动补标签。

完整一分钟 Volume 在该分钟收盘后才可知：用它作为三手撮合门槛是一项离线模拟假设，不证明 Open 时刻有足量盘口。旧合约最后分钟强平遵循既有指定协议，但本研究没有独立验证实盘可以提前获知下一交易日主力切换的公布时点。两者都不能包装成经纪商真实成交证明。

后续窗口实验见 `FG_5_Point_Training_Window_Experiments_V1.md`。本阶段已观察到的统计事实、OHLC 模拟假设和未验证的模型能力假设必须继续分开记录。
