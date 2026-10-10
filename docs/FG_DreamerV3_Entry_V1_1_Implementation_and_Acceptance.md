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
