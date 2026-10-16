# WP02 验收矩阵

这是行为与证据的映射，不把旧版本 PASS 或测试收集成功当作当前运行通过。命令、退出码和每个 testcase 保存在 `docs/receipts/wp02_v2/`（交付时复制原始证据）；运行原件在 `artifacts/fg_s3_v2/e_quality_wp02/evidence/`。最终统计见 `WP02_IMPLEMENTATION_REVIEW.md`。

| # | 要求 | 直接证据 / 测试 | 范围 |
| --- | --- | --- | --- |
| 01 | V1 不可冒充课程V2、五字段严格合同 | A07；course contract 全集 | CPU合成 |
| 02 | Reward、Clock、TP/SL、手数和mask冻结 | WP01合同/原生预检；原金融Env测试 | CPU合成 |
| 03 | MACD字节进入实际e_array_hash | A06 | CPU合成 |
| 04 | partial1–4、EMA第五根仅提交一次、hist不乘2 | A01/A03/A06 | CPU合成 |
| 05 | 首值、≥35个此前完整桶、固定价格scale | A02/A06 | CPU合成 |
| 06 | 未来OHLC和到达时间改动不影响过去 | A08真实MarketData前缀对比 | CPU合成 |
| 07 | 重复查询、流式和冷重建一致 | A03/A06/B01 | CPU合成 |
| 08 | 缺/重/乱序/晚到/非有限/负价格拒绝 | A04/A09 | CPU合成 |
| 09 | 夜盘、跨午夜的闭市间隔、午休、官方日、周末/休市 | A05/A09；原Clock回归 | 冻结FG345分钟时钟；虚构午夜交易拒绝 |
| 10 | 零时长开盘元数据不改锚价、不推进EMA | A09 | CPU合成 |
| 11 | Env和教师fork不共享可变市场/账户 | C01/C05；market cache回归 | CPU合成 |
| 12 | 原生Obs→原Replay→真实Agent forward | B01 | 原始持久化Replay字节，fresh参数 |
| 13 | 合法MACD扰动影响token和RSSM logit | B01/Encoder测试 | CPU合成 |
| 14 | valid=0遮蔽输入及bias，Agent正常forward | B01/Encoder valid-gating测试 | CPU合成 |
| 15 | sidecar/future字段不得进入Policy/Replay/Encoder | A07/B01/E07；E Observation回归 | CPU合成 |
| 16 | 同版本模型、优化器、Driver、Replay完整精确恢复 | EM15；driver account checkpoint | CPU合成；没有旧权重迁移 |
| 17 | P0/P1/P2低中概率/P3真实独立账户 | C02五变体 | 原Env/Driver/Replay+完整reconcile |
| 18 | P2同seed复现、不同seed产生不同轨迹 | C04 | CPU合成 |
| 19 | P3按有效分钟延迟，非法mask和forced WAIT不伪标主动负例 | C02/C01/D03；原Single3/Clock测试 | CPU合成 |
| 20 | 价格、费用、TP/SL、Reward和结算真实一致 | C02/D02/B01；原Reward/Env回归 | CPU合成+原生重播 |
| 21 | chunk reset保留持仓、MACD和账户连续性 | B01/C02/D02；EM15 | CPU合成 |
| 22 | L1真实TP同/跨官方日、SL、OTHER和删失有效位 | C01；Quality teacher/patch acceptance回归 | CPU合成 |
| 23 | MAE、early10和underwater独立、退出分钟排除 | C01；原Quality facts回归 | CPU合成 |
| 24 | L1多优质点不等于L2每处ADD | D04 | 小型独立数值案例 |
| 25 | L2 ADD/WAIT值、margin、tie、WAIT tie-break | D01/D04 | 独立原生双动作穷举 |
| 26 | P4真正Quality TP完整后缀、全计划native等价 | D02；build_quality_suffix原生候选分层/计划校验 | 合成+真实作业独立记录 |
| 27 | 低资金、持仓、未知边界和删失不伪标WAIT | D03/D05；原Q budget/censor回归 | CPU合成 |
| 28 | 旧C1 schedule/旧Reward不得冒充Quality P4 | D02/D05；原P4拒绝回归 | CPU合成 |
| 29 | 非TRAIN/资格字符串/自签假证书读取行情前拒绝 | C03/E01/E02/E05 | 禁止价格reader被调用 |
| 30 | 源字节损坏、源身份变化和不安全fork拒绝 | B01/C01/C05；原Quality source guard | CPU合成 |
| 31 | 外层和内层manifest路径穿越、symlink拒绝 | E02/E06/E09 | 禁止外部文件哈希或np.load |
| 32 | 账本改写、不同账户混入，即使重签hash也拒绝 | E10两负例 | 先正例通过，再篡改同一合法Trace |
| 33 | 重签伪造后验仍被真实重算拒绝 | E04 | L1/L2 native重算 |
| 34 | Catalog事实、动作有效位和准入独立，旧Join不放行 | E08 | CPU合成 |
| 35 | 显式合法draft退出0，formal/空模式/缺身份非零，禁止副作用 | E03；WP01原生preflight全集 | 状态/退出码/文件/副作用同时断言 |
| 36 | 唯一市场、账户决策、优化器曝光三分母分开 | C07 | JSON/CSV可复算、会话分母对账 |
| 37 | 原E+Quality四公开入口仍拒绝且不生产产物 | public_guard_commands.json；composition回归 | 四命令实际均exit1 |
| 38 | 来源/配置/Git/source identity、hash和随机性完整 | stage记录；source_review.json；Corpus source/code/model锁 | 真实Corpus最终核验见主报告 |
| 39 | 末两主力资格日真实TRAIN小样及全部账户重播 | real_generate_small_04；real_verify_small_04/real_coverage_small_04收据 | 生成及随附重播PASS；独立verify RUNNING |
| 40 | 无TEST、GPU、新训练、W&B或18合约生产 | 原入口保护审计+CPU环境+选定单合约来源allowlist | 这些训练/全量行为均NOT_RUN |

## 证据解释

新测试文件中的A/B/C/D/E编号是可定位的行为组，参数化变体不被包装成独立统计样本。当前WP02工程结果不证明学生已学到盈利能力，也不证明任意资金域或所有真实候选均完成慢速穷举。严格后缀证明的范围及native抽样清单随p4_plan/corpus_manifest保留。

历史迁移正例固定两端Git SHA；当前V2拒绝旧迁移的负例另行保留。没有通过生产兼容别名、xfail、skip或放宽认证来消除旧断言失败。
