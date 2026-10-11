# FG Single3 Env 执行协议与 E 阶段验收

本实验协议为 `fg_single3_v1`，成交模型为 `minute_tp5_sl30_risk_first_v1`。账户、PositionManager、开平仓原子操作、费用及保证金复用现有实现；旧 `fg_entry_v1` 仍默认最多两批6手、无SL、V1 Reward。

## 配置与观察

通过独立派生配置严格校验：初始300000元，ADD3，max_long=max_short=3，明确启用SL。训练源由FG409/all冻结manifest派生；不覆盖旧主配置、不重新划分TRAIN/validation/test。

市场 `(60,18)`、仓位 `(9,)`、Mask `(2,)` 保持形状。新仓位字段如下：

| 索引 | 语义 |
|---:|---|
| 0 | 给定历史方向 −1/0/+1 |
| 1 | 实际持仓量/3，只有0或1 |
| 2 | 已发生且未成交的SL pending，0或1 |
| 3 | 第一槽实际持仓量/3 |
| 4 | 有符号TP剩余点数/5，使用Env实际可执行Mark |
| 5 | `log1p(自然持仓分钟)/log1p(43200)`，不截断长持仓 |
| 6～8 | 第二槽固定0 |

SL剩余点数可由 `35−5×position_state[4]` 推导；LONG/SHORT含spread的Mark回归已覆盖。pending必须显式观察，因为价格恢复后仍需优先执行已触发SL。

配置、Observation和Reward均有身份哈希；Dataset拒绝同形状的旧6手协议或不同Reward候选混用。旧Checkpoint不能恢复到新协议。

## 成交与反馈

- 动作基于当前已完成观察，下一bar开盘申请入场。当前合法Mask不保证下一bar成交；继续核对执行时方向、截止、价格、资金、量和bar时钟。
- TP是实际入场价有利方向5点；已有仓位用可执行极值超过目标至少一个tick确认，入场同bar只用收盘确认，完整3手成交于TP目标价。
- SL是实际入场价不利方向30点；有效bar的可执行不利极值触发。TP/SL双触达保守SL优先，记录`ambiguous_tp_sl`。不假设已知OHLC内tick顺序。
- 普通SL按止损目标成交；更差开盘跳空按该可执行开盘价成交，记录额外价差。触发时量不足仍保留pending；后续有量的bar按可执行开盘价重试，即使恢复到更好的价格。
- bar总Volume由开仓和退出共同消耗。入场当bar需至少6手Volume才能又开又完整平3手；不足量或原子拒单不创建平仓、奖励或资金释放。
- 风险优先级保留 MARGIN_CALL → MAIN_ROLL → CONTRACT_EXPIRY，高于TP/SL。MAIN_ROLL用原收盘价格协议，其它终局风险用原不利极值协议。SL本身不会终止整个账户。
- 退出事件在bar完成后发布反馈；开盘价参考时点单独登记。`settle_exit`是原子成交原语，`step`负责将完整成交和观察风险事件评分；只在完整步骤结束后保存可恢复快照。
- 新协议完整退出有确定性唯一fill_id；重复ID先拒绝再处理账户。部分退出请求在记账前拒绝。旧协议仍允许其已有部分退出路径。
- TP自然分钟、首次逆行有效bar、Env推进步三个时钟分开。首次逆行含入场bar为第1根；同barSL与early10可叠加。零时长元数据不成交、不增加风险有效bar。
- Env1024截断不平仓、不丢pending/early10；reset重复观察不重复奖励。STATE schema2保存完整金融、Reward、processed fill和RNG；导入核对Reward与财务活动entry身份。
- DATA_END/GAP是中断/删失，未成交退出不虚构净盈亏；保留实际仓位。

## 验证命令与进度

```bash
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
 MPLCONFIGDIR=/tmp/fg-single3-mpl \
 /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -p no:cacheprovider \
 tests/entry_v1/test_fg_single3_env.py tests/entry_v1/test_entry_contract.py -q

env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
 MPLCONFIGDIR=/tmp/fg-single3-mpl \
 /home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_single3.py env-test \
 --output artifacts/fg_single3_v1/env/run_20261011_03
```

真实行情验收使用TRAIN FG809（SHORT）和FG909（LONG），每合约独立账户执行两遍，第二遍在所有1024边界export/import，比较每步观察、动作、Reward、成交/账本摘要和最终完整状态。每步检查0/3手、唯一entry、账本守恒；最终权益与原生TradeRecord真实PnL减费用及未实现PnL对账。

首两次研究运行保留在run_20261011_01/02，用于实现迭代证据。最终有效版本以run_20261011_03的源码字节、输入哈希和收据为准；最终结果在该运行完成后登记。

新增边界测试包括TP/SL双触达、gap费用、pending恢复和Mask、真实非标准退出评分、部分退出/重复fill拒绝、元数据/删失、跨Reward/旧Checkpoint及Dataset混协议拒绝。两项Dataset测试初次因夹具没有落盘CSV失败，补齐输入后13项新测试通过；不是产品运行失败。

## 自审与限度

自审发现并修复了退出metadata可能残留前次pending尝试的问题；现在所有请求经共享纯退出判定生成当次metadata。Oracle将复用同一判定，不另写更乐观的撮合器。新协议导入还检查金融/Reward活动entry一致性。

仅有分钟OHLC和总Volume，成交是公开的保守模拟，不是交易所逐笔顺序或实盘成交证明。E通过也不说明Reward最优、Oracle已证明、Replay全量完成或Agent学会择时。

## 最终真实运行结果

E运行：`run_20261011_03`，状态 **PASS**。

| 合约 | 步数 | TP | SL | MAIN_ROLL | early10 | 净PnL | chunk恢复 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CZCE.FG809 | 28263 | 37 | 15 | 1 | 28 | -17514.0 | 27 |
| CZCE.FG909 | 26545 | 49 | 9 | 1 | 26 | -2982.0 | 25 |

收据SHA256：`b00d4628414852b91224fd29c0b5c9c9573b6d3e51718808fbbe5323760484af`。
训练源content hash：`ffc45db584106b7c348b7f3aee95096acbcb560c31a7d67fa340f30d2e10009a`。
两个账户的持仓集合均为0/3，max_entries=1，最终空仓；每步及完整状态独立对账PASS。未创建Agent、未启动GPU、未做梯度更新。
既有29项Env回归通过；单仓位新增边界测试见测试文件及执行记录。

## E04发布前回归及源码归档

实际命令：`python tools/fg_single3.py env-test --output artifacts/fg_single3_v1/env/run_20261011_04`。44项测试通过，真实FG809/FG909的财务、每步反馈和最终状态哈希与E03一致；27/25次chunk恢复一致。E04收据content hash：`06ca881053ada31f064a7c488f3140dfdcb51c0a8cb1bf95f1fe1d07ee03ad89`。

TP信息缓存只对追加的真实归因增量计算，reset/import清空缓存；该缓存不进入可恢复账户状态。实际旧/新方法在同一事件历史输出一致；2000个TP、100次info查询用时6.026s/0.000408s，属于扩大历史的性能微基准，不作为财务验收证据。财务证据来自E04完整真实重放。

P3的chunk计时使用当前已知财务/市场状态判断恢复后的合法机会，不把CHUNK_END行政mask误认为机会消失；模型发布mask与is_first保持原规则。这一边界错误在R01发现，旧R完成标记已撤销，R02发布前15项Replay回归通过。
