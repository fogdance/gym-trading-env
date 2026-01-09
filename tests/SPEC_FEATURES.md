好的，我们把之前所有“版本漂移 / 互相引用 / 既不是 spec 也不是 env”全部清空，**从目标→数据→Spec→Oracle→Unit→Integration** 重新设计一套“唯一方案”。我下面只讲方案，不再抛一堆代码。

---

## 0. 目标（冻结成一句话）

> **Market + Agent 的 RAW/OBS：所有字段都要测，且数值逐字段正确。**
> Unit 和 Integration 共用同一份固定“正常数据”。
> Integration 必须“跑真实 env”，用 env 输出的 obs/agent_state/market_seq 去对齐校验。

这里的“数值正确”指：**对每一列、每一步**都能与 Spec Oracle 一致（允许 float32 的误差容忍）。

---

## 1. 核心原则（防止再次混乱）

### 原则 A：唯一真相是“Spec（公式+字段列表+裁剪/掩码规则）”

* 不再以 `market_features.py` 或 `agent_features.py` 作为“金标准”，因为那样**改了实现测试也跟着变**。
* 我们写一份 **Spec Oracle（测试侧独立实现）**，作为唯一真相：

  * Unit：`market_features.py` 的输出必须 == Spec Oracle
  * Unit：`agent_features.py` 的输出必须 == Spec Oracle
  * Integration：`env` 输出必须 == Spec Oracle（用同一套数据）

> 这样 env 内部任何 pipeline 改动、字段顺序改动、mask/clip 改动、window padding 改动都会炸。

### 原则 B：Unit 和 Integration 共用同一份“固定正常数据”

* 统一用一个 `MarketDataFactory` 生成 deterministic 数据（同 seed、同 tz、同时间覆盖）。
* 数据工厂能：

  * 输出 DataFrame（unit 用）
  * 写 CSV（integration/env 用）
* 不再“手搓临时 df”。

### 原则 C：Integration 的 oracle 输入要来自“env 实际用的数据”

* 最理想：env/bar_source/store 暴露 `df_1m_used / df_prev_session_used`（否则用工厂原始 df）
* Integration 只信 env 输出 + Spec Oracle，不信 env 内部 intermediate。

---

## 2. 组件拆分（让同一份东西被反复复用）

### 2.1 固定数据工厂（所有测试共用）

**`tests/factories/market_data_factory.py`**

* 生成覆盖足够长的 1min 数据（保证 futures strict_345 对齐不会缺）
* 列：`Open/High/Low/Close/Volume/OpenInterest`
* deterministic：固定 seed，公式生成（sin + drift + 固定 spread）

输出：

* `bundle.df_1m`
* `bundle.df_prev_session`（可选）
* `write_csv(df_1m)` → integration 用

### 2.2 Action 脚本工厂（integration 用）

**`tests/factories/action_script.py`**

* 最小可控开仓/平仓脚本（你已经给了 Action/ForexCode）
* 要覆盖：open/hold/close、无效 close、重复 open、near_eod blocked 等（可分多段脚本）

### 2.3 Spec Oracle（唯一真相）

这是关键：**它不 import 被测模块**，只实现 spec。

* **`tests/oracles/market_spec_oracle.py`**

  * 纯按 spec 公式从 df_1m 生成 `df_market_full`（包含 REQUIRED_MARKET_COLS 的所有 raw+obs）
  * 包含 window 抽取逻辑（padding 规则也写成 spec）
* **`tests/oracles/agent_spec_oracle.py`**

  * 给定“简化的 env state snapshot”（positions、cash、margin、fee、realized、current_price、minutes_to_eod、entries_used、action_result_code…）
  * 计算 agent raw/obs 全字段（按 spec）
  * 注意：它不调用 env 的 helper，也不调用 `agent_features.py`

> 这样任何实现偏离 spec 都会被抓住：
>
> * 单元测试：模块实现偏离 spec 会炸
> * 集成测试：env 输出偏离 spec 会炸

### 2.4 Snapshot 提取器（integration 只做取数，不做计算）

**`tests/utils/env_snapshot.py`**

* 从 env 取：

  * `obs["market_seq"]`
  * `obs["agent_state"]`
  * 同时构造 `AgentSnapshot`（给 agent oracle 用）
* 这里允许“读取 env 内部状态”，但**不允许计算特征**。

---

## 3. 测试层级设计（你要的：全部字段、数值正确）

### 3.1 Unit：只测“特征计算模块是否符合 Spec”

1. `test_market_features_match_spec.py`

* 输入：工厂 df_1m/df_prev_session
* 期望：`build_market_features(df)` 输出的 **每一列** == `MarketSpecOracle(df)` 的对应列
* 覆盖：Market RAW 全字段 + OBS 全字段（逐字段逐行 allclose）

2. `test_agent_features_match_spec.py`

* 输入：构造若干 AgentSnapshot case（flat/long/short/multi entries/invalid action）
* 期望：`agent_features.py` 输出 == `AgentSpecOracle(snapshot)`（逐字段精确）
* 覆盖：Agent RAW 全字段 + OBS 全字段

> Unit 不跑 env，不涉及 step、撮合、止损等复杂路径，只验证“特征公式”正确。

---

### 3.2 Integration：跑真实 env，校验 env 输出 obs 正确（逐字段值）

**`test_env_obs_match_spec.py`**（最重要）

* 用工厂生成 CSV，配置 env：

  * `obs_feature_mode="raw"` 跑一遍
  * `obs_feature_mode="obs"` 再跑一遍
* 每个 step：

  1. 从 env 拿 `obs["market_seq"] / obs["agent_state"]`
  2. 用 `MarketSpecOracle` 基于同一份 df_1m 生成市场全表，再按 spec window 取出预期 `market_seq`
  3. 用 `AgentSpecOracle` 基于 env snapshot 算出预期 `agent_state`
  4. **逐字段 allclose**（并打印差异表：step/field/got/expected）

额外两条你要求的断言也放 integration：

* **行情不受 action 影响**：同一份数据、同一步 index，两个 env（一个执行脚本，一个全 HOLD），`market_seq` 必须完全相同
* **action 只影响 agent**：`agent_state` 允许不同（应该不同），并且各自都要符合 spec

---

## 4. 覆盖清单（表格化承诺）

| 模块                       |        RAW 全字段 |        OBS 全字段 |       数值逐字段正确 |                      使用固定数据 | 跑 env |
| ------------------------ | -------------: | -------------: | ------------: | --------------------------: | ----: |
| Unit: Market vs Spec     |              ✅ |              ✅ |         ✅（全表） |                           ✅ |     ❌ |
| Unit: Agent vs Spec      |              ✅ |              ✅ |     ✅（多 case） | ✅（同 factory 可复用 price/time） |     ❌ |
| Integration: Env vs Spec | ✅（通过 mode=raw） | ✅（通过 mode=obs） | ✅（每 step、每字段） |                           ✅ |     ✅ |

---

## 5. “固定正常数据”怎么定义才不会踩坑

我们定义它必须满足：

* 能触发 futures strict_345 对齐（时间覆盖够长）
* 有 Volume/OpenInterest（不为 0，避免 log/ratio 边界全是 0）
* 有平滑趋势 + 小波动（保证 range/volatility/open_drift 都非零但稳定）
* 不引入 gap / NaN（先做“正常数据”，后续再做异常数据测试）

---

## 6. 执行顺序（重新开始的清单）

**Step 0（先做）：冻结 Spec 文档**

* 列出所有 Market RAW/OBS 字段 + 公式（含 mask/clip/window padding）
* 列出所有 Agent RAW/OBS 字段 + 公式（含归一化、clip、action_result 编码方式）
=====================================================================================================

# Market + Agent Features Spec (Frozen)

## 1) Market RAW (FEATURES_MARKET)

列顺序（必须一致）：

C_t
V_t
I_t
cumVWAP_t
dC_minus_cumVWAP_t
cmp_C_vs_cumVWAP_t
ref_close_t
session_open_t
session_high_t
session_low_t
range_frac_t
bar_dir_t
turnover_t
minute_index_t
limit_up_price_t
limit_down_price_t
dI_from_yclose_t
dP_from_ref_t
pct_chg_from_ref_t
open_drift_t
mask_t
weekday_sin_t
weekday_cos_t

### Futures(strict_345) 生成规则

#### A. strict 345 对齐与 mask
- 对每个 session_id(YYYYMMDD) 构造固定 345 个“分钟收盘 eob 时刻”：
  - 夜盘: [21:00,23:00) 的每分钟 +1min => 21:01..23:00 共120
  - 日盘1: [09:00,10:15) +1min => 09:01..10:15 共75
  - 日盘2: [10:30,11:30) +1min => 10:31..11:30 共60
  - 日盘3: [13:30,15:00) +1min => 13:31..15:00 共90
- 把原 df_1m reindex 到该 clock：
  - mask_t = (Close 在 fillna 前 notna).astype(int)
  - OHLCV/OI 缺失 fill 0.0

#### B. 基础列
- C_t = Close
- H_t = High, L_t = Low（仅 env 内使用，不进 obs）
- V_t = Volume（缺失则 0）
- I_t = OpenInterest（缺失则 0）
- minute_index_t = 0..344
- weekday_sin/cos：由 session_id 解析成日期，weekday in [0..6]，
  sin = sin(2π*weekday/7), cos = cos(2π*weekday/7)

#### C. ref_close_t（参考收盘）
- valid_close = Close where mask_t==1 else NaN
- 对每个 session：
  - first_valid_close = first(valid_close)
  - last_valid_close  = last(valid_close)
- ref_close_t = prev_session.last_valid_close 映射到当前 session
- 若 prev 不存在，则用本 session first_valid_close 兜底
- 若传入 df_prev_session：第一天 ref_close_t 强制使用 prev_session 的 last_valid_close
- 最后 ffill/bfill，缺失填 0

#### D. cumVWAP_t（仅用有效分钟累计）
- valid_V = V_t * mask_t
- cumVWAP = cumsum(C_t*valid_V)/cumsum(valid_V)
- 若分母为 0，用 “有效分钟 C_t expanding mean” 兜底
- 最后缺失填 0

#### E. session_high/low（仅有效分钟更新，之后 ffill）
- H_valid = High where mask_t==1 else NaN
- L_valid = Low  where mask_t==1 else NaN
- session_high_t = cummax(H_valid) per session，然后 ffill，缺失填 0
- session_low_t  = cummin(L_valid) per session，然后 ffill，缺失填 0

#### F. session_open_t
- open_valid = Open where mask_t==1 else NaN
- session_open_t = first(open_valid) per session
- 若缺，用 first_valid_close 兜底
- 缺失填 0

#### G. range_frac_t
- eps=1e-12
- ref_safe = ref_close_t if >eps else (C_t if >eps else eps)
- range_frac_t = ((session_high_t - session_low_t).clip(lower=0))/ref_safe
- 仅 mask==1 的分钟写入；无效分钟为 0

#### H. open_drift_t
- open_drift_t = log(C_t / session_open_safe)，仅 mask==1 写入，否则 0

#### I. bar_dir_t
- raw_dir = sign(C_t - C_{t-1})
- bar_dir_t = raw_dir 仅当 (mask_t==1 且 prev_mask==1)
- minute_index_t==0 强制为 0
- 输出 int

#### J. turnover_t
- turnover_t = cumsum(C_t*(V_t*mask_t)) per session

#### K. limit_up/down
- 若提供 limit_up_pct & limit_down_pct：
  - limit_up_price_t   = ref_close_t*(1+up)
  - limit_down_price_t = ref_close_t*(1-down)
- 否则 0

#### L. dI_from_yclose_t
- valid_oi = I_t where mask==1 else NaN
- I_yclose = prev_session.last_valid_oi 映射到当前 session
- 若 prev 不存在，用本 session first_valid_oi 兜底
- 若传入 df_prev_session：第一天 I_yclose 用 prev_session last_valid_oi
- dI_from_yclose_t = I_t - I_yclose，缺失填 0

#### M. dP / pct
- dP_from_ref_t = C_t - ref_close_t
- pct_chg_from_ref_t = C_t/ref_close_t - 1（inf/nan -> 0）

#### N. RAW mask-zeroing
- 对 FEATURES_MARKET + [H_t,L_t]：
  - 除 mask_t、minute_index_t、weekday_sin_t、weekday_cos_t 外，其余列 *= mask_t
  - 无效分钟全部为 0

---

## 2) Market OBS (FEATURES_MARKET_OBS)

列顺序（必须一致）：

obs_V_t
obs_I_t
obs_cumVWAP_t
obs_dC_minus_cumVWAP_t
obs_cmp_C_vs_cumVWAP_t
obs_session_high_t
obs_session_low_t
obs_range_t
obs_open_drift_t
obs_bar_dir_t
obs_minute_index_t
obs_session_phase_t
obs_volatility_t
obs_dI_from_yclose_t
obs_pct_chg_from_ref_t
obs_mask_t
obs_weekday_sin_t
obs_weekday_cos_t

统一常量：
- eps=1e-12
- LOG_CLIP = 1.0

通用：
- valid = (mask_t>0)
- ref_safe = ref_close_t if >eps else (C_t if >eps else eps)

### 价格类 log-ratio（仅 valid 写入）
obs_x = clip(log(x_safe/ref_safe), [-LOG_CLIP, LOG_CLIP])
- obs_cumVWAP_t, obs_session_high_t, obs_session_low_t, obs_pct_chg_from_ref_t(=log(C/ref))

### 差值类 signed-log（仅 valid 写入）
obs_z = clip(sign(z)*log1p(|z|/ref_safe), [-LOG_CLIP, LOG_CLIP])
- obs_dC_minus_cumVWAP_t

### obs_range_t（非负）
obs_range_t = clip(log1p(range_frac_t), [0, LOG_CLIP])，仅 valid 写入

### obs_open_drift_t
obs_open_drift_t = clip(open_drift_t, [-LOG_CLIP, LOG_CLIP])

### obs_dI_from_yclose_t（稳定缩放）
I_yclose = I_t - dI_from_yclose_t
denom = max(|I_yclose|, eps)
obs_dI = clip(sign(dI)*log1p(|dI|/denom), [-LOG_CLIP, LOG_CLIP])，仅 valid 写入

### 量/持仓（heavy-tail->bounded）
obs_V_t = clip((V/I*100), [0,2])
obs_I_t = tanh(log1p(I)/5)

### 时间类
- obs_mask_t = mask_t（不乘 mask）
- obs_weekday_sin_t/cos_t = weekday_sin_t/cos_t（不乘 mask）
- obs_minute_index_t = minute_index_t / max(minute_index_t per day)，clip[0,1]（不乘 mask）
- obs_session_phase_t:
  - 若 max(minute_index_t) <= 400（futures/345）：
    phase = (minute_index_t >= 120).astype(float)  # night=0, day=1
  - 否则 0
  - 不乘 mask

### obs_volatility_t（rolling std of logret）
- 对每个 session：
  - C_valid = C_t where mask>0 else NaN
  - logret = log(C_valid / shift(C_valid,1))，nan/inf -> 0
  - vol = rolling std(window=30, min_periods=5)
  - obs_volatility_t = tanh(vol / 0.01)
- 最后（与实现一致）会乘 mask（因为不在 NO_MASK 集合）

### OBS mask rule
- 除 {obs_mask_t, obs_minute_index_t, obs_weekday_sin_t, obs_weekday_cos_t, obs_session_phase_t} 外，
  FEATURES_MARKET_OBS 其余列 *= mask_t（无效分钟置 0）

---

## 3) Agent RAW (FEATURES_AGENT)

列顺序：

pos_t
have_long_t
have_short_t
entry_price_t
holding_minutes_t
upnl_t
realized_pnl_step_t
realized_pnl_cum_t
fee_step_t
fee_cum_t
equity_t
max_equity_t
drawdown_t
sigma_entry_t
sl_ticks_t
tp_ticks_t
sl_price_t
tp_price_t
minutes_to_timeout_t

### 规则
- long_lots = sum(long_positions.size)
- short_lots = sum(short_positions.size)
- flat: long_lots==0 && short_lots==0
- pos_t = long_lots - short_lots（flat 强制 0）
- have_long_t = 1 if long_lots>0 else 0（flat 强制 0）
- have_short_t = 1 if short_lots>0 else 0（flat 强制 0）
- entry_price_t / holding_minutes_t：
  - 取“active side”：
    - 仅 long: 用 long 的 size 加权 VWAP 与 age
    - 仅 short: 同理
    - 都有：取 size 大的一侧；相等取 open_step 更晚的一侧
  - flat: 二者为 0
- upnl_t：
  - long: (price-entry)*size*lot_size
  - short:(entry-price)*size*lot_size
  - sum(long)+sum(short)
- equity_t = cash_balance + used_margin + upnl_t
- max_equity_t = max(prev_max_equity, equity_t)
- drawdown_t = max_equity_t - equity_t，floor at 0

---

## 4) Agent OBS (FEATURES_AGENT_OBS)

列顺序：

obs_market_open_t
obs_can_long_open_t
obs_can_short_open_t
obs_can_long_close_t
obs_can_short_close_t
obs_pos_side_t
obs_entries_left_frac_t
obs_minutes_to_eod_frac_t
obs_holding_frac_t
obs_upnl_R_t
obs_realized_today_R_t
obs_equity_frac_t
obs_drawdown_frac_t
obs_action_result_t

### 规则
- obs_market_open_t = 1 if market_open==1 else 0
- obs_can_long_open_t = 1 if can_long_open==1 else 0
- obs_can_short_open_t = 1 if can_short_open==1 else 0
- obs_can_long_close_t = 1 if can_long_close==1 else 0
- obs_can_short_close_t = 1 if can_short_close==1 else 0

- obs_pos_side_t:
  - long only => +1
  - short only => -1
  - else 0

- obs_entries_left_frac_t:
  max_e = max(max_entries_per_day,1)
  used = max(entries_used_today,0)
  left = max(0, max_e-used)
  left/max_e

- obs_minutes_to_eod_frac_t:
  day_len = max(day_len,1)
  min(max(minutes_to_eod,0), day_len) / day_len

- obs_holding_frac_t:
  clip(holding_minutes_t/day_len, [0,1])

- R 归一化（若 R_cash<=0 则 0）：
  obs_upnl_R_t = clip(upnl_t/R_cash, [-5,5])
  obs_realized_today_R_t = clip(realized_today_cash/R_cash, [-10,10])

- equity/drawdown 归一化（若 B0<=0 则 0）：
  obs_equity_frac_t = clip((equity_t-B0)/B0, [-1,1])
  obs_drawdown_frac_t = clip(drawdown_t/B0, [0,1])

- obs_action_result_t:
  code = clamp(action_result_code, [0,max_code])
  max_code = ForexCode enum 最大值
  obs_action_result_t = code/max_code（若 max_code<=0 则 0）
