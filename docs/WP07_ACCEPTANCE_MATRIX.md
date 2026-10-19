# WP07 V2：45 个行为的证据矩阵

PASS_CPU 只指实际合成工程行为；不是正式来源、研究效果或全项目 PASS。原始逐 case 清单见 `receipts/wp07/H_STAGE_TEST_INDEX.json`；最终隔离发布版本与复验见 `WP07_COMMIT_LOCK.json`。本矩阵不将旧 WP02/06 PASS 当作本次未运行的真实验收。

| ID | 行为 | 实测证据 | 结果/边界 |
| --- | --- | --- | --- |
| 01 | 明确完成 F3 source | A_contract，完整账本/上游/模型/冻结证据 | PASS_CPU；正式 F3 NOT_RUN |
| 02 | fresh optimizer 和完整非 opt 接续 | A_handoff、A_contract | PASS_CPU |
| 03 | 同 Run exact resume | G_native_cold_binding、H_four_cold | PASS_CPU |
| 04 | 非法来源、TEST 拒绝 | A_contract、H_cli；无 Run 副作用 | PASS_CPU |
| 05 | F3 continuation 不冒充正式 F4 | 合同三模式、原 ONLINE_ADAPT 不变、H_shared Guard | PASS_CPU |
| 06 | 原生 beta=0 真实 Actor 动作 | B_shadow_archive 原 Driver/Env/Replay | PASS_CPU |
| 07 | Teacher 注入 raise 不影响执行 | B_shadow_archive 两段 128 分钟 | PASS_CPU |
| 08 | real mask 合法 | B_shadow_archive、F_gates_identity | PASS_CPU |
| 09 | partial 5m 无未来泄漏/EMA 不重复提交 | H_macd 14 例、H_shared dated-tail 4 例 | PASS_CPU |
| 10 | TP/SL、roll 和完整账户对账 | B 两段、H CLI 原 reconciler | PASS_CPU；非盈利证据 |
| 11 | stepid/Owner/post-step 准确 | C_data_stable、原 WP03 canonical Join | PASS_CPU |
| 12 | pending 不是 WAIT 真值 | C_data_retry，pending valid=false | PASS_CPU |
| 13 | Natural/aux/online 源身份 | C_mixed、H CLI、H_source_exposures、原 DATA-MULTI F3 | PASS_CPU；真实两发布 Corpus NOT_RUN |
| 14 | 窗口不跨账户/episode | C、H_shared WP03 Reader | PASS_CPU |
| 15 | 跨 source/account Pair 拒绝 | C owner 攻击、H_shared、H_f3 canonical label/Pair | PASS_CPU |
| 16 | 两 Reader RNG/游标恢复 | C_data_stable、G/H 四冷恢复 | PASS_CPU |
| 17 | MIXED 实曝不是独立市场 N | 原 Reader audit、Online SourceExposures、H CLI report | PASS_CPU，按源/role 分母 |
| 18 | 只更新 quality_actor 及其 slots | D_actor_delta、H CLI before/after | PASS_CPU |
| 19 | WM/heads/Q/旧 pol/value 冻结 | D_actor_delta、H_f1/f2/f3 owner 回归 | PASS_CPU |
| 20 | 无合法标签零更新 | D_supervised_candidate、H_f3 skip-first/full CP | PASS_CPU |
| 21 | 真正 K 步预热影响输出 | D future-target isolation、H_f1 K/T/随机 carry、H_f3 burn-in | PASS_CPU；真实 K32/T64 支持量 NOT_RUN |
| 22 | 非有限候选原子回滚 | D finite-input overflow、H_f1/f2/f3，slots/counter 字节 | PASS_CPU |
| 23 | L2 主监督、冲突 L1 Pair 过滤 | 原 F3 actor_objective、H CLI conflict 计数、H_f3 | PASS_CPU |
| 24 | real/imag 同一 quality_actor | E_rl_stable_initialization；同 logits、旧 pol 任意扰动 | PASS_CPU |
| 25 | RL 只改 Actor/Value 合法 owner | E_rl_delta、G/H RL slots 冷恢复 | PASS_CPU |
| 26 | 旧 pol 不更新 | E 参数/slots 前后全键比较 | PASS_CPU |
| 27 | imagined mask false-open/fallback | F 原生多步校准、原正式 mask Gate | PASS_CPU；充分校准 NO_GO |
| 28 | 原 Quality TP reward/continue/discount | F 实际数组、E native market_advance 和 332/333 | PASS_CPU；真实 TP/SL WM 预测充分性 NOT_RUN |
| 29 | Critic 有限真实梯度 | E 两次实际 Ninjax/optimizer、Value 参数与 slots 变化 | PASS_CPU |
| 30 | RL_OFF value/slow/norm 冻结 | E 逐 leaf/slot 比较 | PASS_CPU |
| 31 | 探索限于真实合法 mask | F_gates_identity，探索 RNG 保存/恢复、单动作 mask | PASS_CPU |
| 32 | SHADOW 零更新 | B 模型/slots 字节、G/H SHADOW 计数 | PASS_CPU |
| 33 | 技术不充分时 Gate 阻止 RL | F 数值/WM/policy/支持量/伪认证负例，独立 research review | PASS_CPU；正式启用 NO_GO |
| 34 | Prediction/Q 融合关闭 | A model/合同检查、E owner、H CLI eval逐行 diagnostics | PASS_CPU；原单独 Admission 不变 |
| 35 | Teacher-free 确定性 Actor-only eval | H CLI 原 evaluate_native 两次同初态/argmax 比较 | CPU 定向复验及最终发布结果见原始收据；真实考试 NOT_RUN |
| 36 | SHADOW 新进程全状态恢复 | G/H 四冷恢复，64 vs 32+32 | PASS_CPU |
| 37 | online BC 新进程恢复 | G/H 同上，权重/slots/下一 batch/日志 | PASS_CPU |
| 38 | 发布 7 条中断精确恢复 | G/H publication phase，标签无重发或漏发 | PASS_CPU |
| 39 | RL slots 冷恢复 | G/H RL case，同 Critic/normalizer/账户/reader | PASS_CPU |
| 40 | 不同代码/数据/CP 拒绝，不回退旧节点 | A tamper、G core、H_shared/F1/F2/F3 identity 负例 | PASS_CPU |
| 41 | F1/F2/F3、原课程 Guard | H_f1 18 / H_f2 20 / H_f3 26；H_shared 119 | PASS_CPU，集合有重叠不作全项目数 |
| 42 | 原 MACD V3 因果和 dated tail | H_macd、H_shared | PASS_CPU；原算法无改动 |
| 43 | 原在线账户跨夜/资金语义 | B start=330 跨官方日、两段完整 ledger/reconciler | PASS_CPU |
| 44 | 无 GPU 可完成工程 | 所有命令 JAX_PLATFORMS=cpu、CUDA_VISIBLE_DEVICES 为空 | PASS_CPU；GPU NOT_RUN |
| 45 | 真实 TRAIN 未启动如实标注 | H_real_readiness、全部 engineering receipts | NOT_RUN / FORMAL_NO_GO；真实参数更新 0 |

`H_source_exposures` 指 H CLI 报告和四类冷恢复 result 中的 `source_exposures` 字段，不是声称存在同名测试。源层 delivered/预取计数、已消费 loss-eligible 计数、有效 optimizer updates 与真实市场分钟分别报告。
