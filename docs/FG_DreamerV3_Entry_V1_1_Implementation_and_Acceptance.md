# FG Entry V1.1 实施与验收记录

## 冻结契约

以 V1.1、用户补充及最新选择为准。仅 FG，初始资金 300000 CNY；
WAIT=0 / ADD_3=1，同向每笔 3 手，上限 6 手。复用 PositionManager、Ledger、
UserAccounts 和既有原子开平仓。每笔独立 +5 价格点，不设置策略止损或持仓期限。
唯一 Reward 为完整 3 手 TP 成交后的 `60/(60+自然分钟)`，其他状态为零。
不实施文档中的 +1 奖励对照，不训练 JM。

人工维护独立方向 YAML：timezone=Asia/Shanghai，intervals 每条为
contract/start/end/direction(LONG|SHORT)，区间 [start,end)，同合约禁止重叠，
空白区间为 NO_TRADE。不推导方向、不显示未来区间。
反转/失效停止新增，旧仓继续原 TP 和硬风控，全部处理完才启用新方向。

## 成交及账户生命周期

观察完成 bar 后于下一真实 bar 的 open 事件成交。DB 导出保留 bob/eob；
相同时间戳按完成→决策→下一开盘成交的事件顺序处理，不回到信号 bar 成交。
新仓当根仅可成交收盘价穿越 TP 至少一 tick 才确认；老 TP 后续 bar
需至少穿越一 tick，目标价成交、时间保守记 eob。费用/价差/滑点独立记账。
分钟 OHLCVOI 不证明盘口排队或实际 3 手成交，结果明确标注为保守模拟。
硬风险与 TP 顺序不明时硬风险优先。失败提交不产生 TP 事件或奖励。

分块、数据中断和最终数据截止持久化账户，不免费 reset；同一分区续接。
到期和保证金强平单独记录，退出未真正完成不能删除持仓。
checkpoint 保存经济状态、游标、订单、去重、随机状态及配置哈希；重启不重复成交。
三手整笔是分钟撮合默认协议；外部部分成交需按实量累计，完整 TP 前无奖励。

## 阶段及准入

0. 两仓库各自 feat/fg-entry-v1.1 隔离 worktree，原残留不提交、不覆盖。
1. 先环境和确定性测试：LONG/SHORT 的 0→3→6→3→6、1/2/5 分钟 TP、
   双 TP、失败回滚、跨夜、反转、账户续接、到期/保证金、守恒、无未来数据。
2. DB 按真实 FG 合约导出，冻结人工方向及有效合约参数，三基线回放并输出审计。
3. 独立 Dreamer profile；真实/梦境 WAIT 恒合法；checkpoint/恢复测试后训练 smoke。
4. 条件满足才运行 seeds 0/1/2 各 1M 步，验证集选 checkpoint，锁定测试集比较。

基线：always_add；random_add p=.5；rule_add 最近20个有效完整bar的5点回撤/反弹。
四策略共用行情、方向、成交、成本与资金。报告 TP次数、平均等待、速度奖励、
未完成持仓、最大浮亏、净收益(含MTM)、非TP退出及合法ADD时主动WAIT比例。
没有样本外改善证据不声称模型学会择时；不因不收敛改变业务或Reward。
数据按交易日60/20/20时间划分，训练不访问验证/测试；持仓不跨分区续接。
缺人工方向、有效参数或完整训练生命周期时只完成模板/synthetic，不正式训练。

## 证据与进度

- 起点：env 9699286，Dreamer 92af879。
- 原工作树的 JM 候选门控及 Dreamer 本地文件均原地保留。开发 worktree
  位于 /tmp/fg-entry-v1/env 和 /tmp/fg-entry-v1/dreamerv3（需长期保留时另行迁移）。
- 规划阶段原工作树检查：环境32项、Dreamer15项通过；干净基线另行核对。
- 尚未 DB 导出、没有用户维护的方向区间和正式有效成本/到期参数。禁止凭空填入。

后续每阶段在本节追加准确命令、结果、提交及未验证项。

### 阶段 0/1 — 已验证

- 干净基线：环境32项通过(5.65s)，Dreamer15项通过(1.55s)。
- 环境19项专项回归及34项既有账户/Mask回归：53 passed (12.29s)。命令：
  `PYTHONPATH=src python -m pytest -p no:cacheprovider tests/entry_v1
  tests/unit/test_position_manager.py tests/unit/test_ledger_basic.py
  tests/unit/test_atomic_close_p0.py tests/contract/test_target_action_mask.py
  tests/unit/test_episode_policy_multiday.py tests/contract/test_bookkeeping_conservation.py
  tests/integration/test_trading_env_accounting_integration.py`。
- 已复用既有账户初始化和原子开平仓，仅增加可选fill_time与quantity参数；
  旧默认行为保持。部分平仓替换剩余Position对象，避免污染原子回滚快照。
- 一分钟模拟默认3手整笔开仓；部分退出按实量记账、累计完成才领奖。
  不实现实盘订单连接器及部分开仓回报适配。
- 真实DB只读查询已成功：fut_bar_1m_v2有bob/eob/symbol/trading_date；
  目标2027年1月合约的DB名称为CZCE.FG701，51889根，2026-01-19至2026-10-09。
  CZCE.FG609有75317根，2025-09-15至2026-09-14。
  连续标识CZCE.FG/CZCE.FG00/FG01等不允许作为真实交易合约导出。
- 当前未输入人工方向区间，不把行情推导出的多空方向当人工标签。

## 主力数据与人工方向补充

以 `FG_Entry_Data_Export_and_Roll_Audit.md` 为此次实际数据证据及用户确认记录。旧“方向尚未输入”的阶段记录已被 `fg.txt` 的 26 条人工区间取代。已确认只导出 FG 主力至 2025-12-31，并在换主力前按旧合约最后可成交分钟强平，无速度奖励。数据截止仍保留亏损账户。

Dreamer 独立 profile `fg_entry_v1`：两动作、WAIT 常合法、可用性不确定时只回退 WAIT；真实交易方向仍由人工标签给出。新增 Driver/FromGymnasium 检查点保存金融账户、待执行动作、策略 carry 和 reset 标志，历史 profile 默认关闭此项。正式启动要求冻结 FG manifest、train 角色、完整日历/生命周期及所有输入散列，不能依靠旧 guard 的空元数据跳过机制。

提供三个基准、Dreamer 检查点评估与入场频率匹配 Random 的统一逐笔/权益/Reward 输出。未完成真实多 seed 训练与样本外比较前，不宣称 Dreamer 学会等待或具有额外择时能力。

## 当前权威配置及数据（后续用户澄清覆盖早期记录）

唯一人工配置为 `configs/fg_entry_v1/fg.yaml`。已合并修正后的 fg.txt 26 条方向区间与 1.yaml/CZCE_FG 成本。旧原始文件仅是迁移来源，运行不读取它们。派生数据目录中的环境/合约文件为自动生成的冻结快照，不能手改；运行与检查点包含主配置内容散列。

用户确认历史数据全部正常，无中断；生成的日期时段基于该声明，清单明确标注“用户确认完整”，不冒称独立交易所日历审计。用户定义到期为 DB 映射中的“失去主力资格”，因此使用主力切换前旧合约最后可成交分钟退出；无需另填交割到期规则。

FG805 的 5 个本次生成文件已按用户要求删除。当前导出 24 个区段、643929 根，2018-04-10 21:01 至 2025-12-31 15:00。权威冻结数据为 `artifacts/fg_entry_v1/dataset_v3/manifest.json`；train=20180411~20221125，validation=20221128~20240618，test=20240619~20251231。12 个完整且有标签的 train 区段可用，FG809 未配置方向时 NO_TRADE。最后 FG605 不具备数据范围内的主力到期证据，截止只冻结未平仓账户。

新 FG 按 DB 真实交易日和实际分钟计算现有市场特征，避免旧固定 345 分钟时钟裁掉历史 23:00 后夜盘。旧模式仍走原固定时钟。回归总计 66 项通过，包括历史夜盘及未来价格扰动不影响过去特征；Dreamer 相关 13 项通过。

Synthetic CPU 真实训练链路：2 次参数更新、49 个有限标量指标、合法 WAIT；结果 `/tmp/fg-dreamer-smoke-w5sts2_g/result.json`。不代表正式模型或择时能力。

## 2026-10-10：最终 9 维仓位观测

按用户最后修订删除 expiry_remaining，最终输出 market_seq(60,18)、position_state(9)、action_mask(2)。真实 PositionManager 数量、目标执行价、实际成交时间映射到 9 字段；使用 entry_id 固定的观测投影索引，部分退出和再入场不交换仍在持有的另一批。原同步开仓没有待成交队列，pending_ratio=0，不能把剩余额度伪装为挂单。年龄按 log1p(T)/log1p(43200) 缩放且不截断，距离按有向 TP 点差/5，可为负而不计奖励。36 项环境/特征回归、6 项 GPU 环境下的 Dreamer/Gym 及检查点回归通过，git diff --check 通过。本阶段不训练；A2 评分差异作为后续专项测试，不承诺评分独立。详见 docs/FG_Entry_Observation_V1.md。
