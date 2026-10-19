# WP08 的 45 项行为验收映射

按任务书行为逐项映射，不把它等同于 45 个测试函数，也不把选定集合称为全项目通过。下表为工程用例；真实受训模型 DEV、统计泛化和 TEST 另列 NOT_RUN。每项命令/exit、JUnit 完整 case 名和执行源码见 [测试索引](WP08_TEST_INDEX.md)。最终冻结补验结果见实施报告发布记录。

| 编号 | 行为 | 直接测试 / 证据 |
| --- | --- | --- |
| 01 | 合法 SYNTHETIC_DEV | test_wp08_contract::test_pure_contract_stable_hash_and_no_io；CLI 完整 plan/run |
| 02 | fold / TEST / 空模式拒绝 | test_train_corpus_cannot_be_dev_and_empty_is_not_draft；test_cli_refusal_no_output_or_training_side_effect |
| 03 | TRAIN Corpus 不冒 DEV | test_actual_training_corpus_manifest_is_not_a_dev_market_source；role-before-price 探针 |
| 04 | E/MACD V3 / Reward / Contract | H_source_exact_reopen；WP01/WP02 MACD/Observation 共享回归；实际 F3/F4 schema 加载 |
| 05 | 官方日及每合约区间 | test_disjoint_contract_windows_are_explicit_and_inside_study；跨不同日期合约隔离；源端拒绝不存在官方日 |
| 06 | 成本与账户一致 | test_native_trade_denominators_risk_fees_and_daily_accounts；H_finite_train_readonly 原生重播 |
| 07 | 文件/许可伪造拒绝 | test_symlink_and_forged_validation_receipt_refused；test_json_cannot_mint_native_validation_admission；corrupt_actual_source_checkpoint |
| 08 | 纯合同 hash 稳定 | test_pure_contract_stable_hash_and_no_io |
| 09 | 真 quality_actor | test_actual_f3_readonly_loading_and_teacher_free_argmax；逐步 verify_policy |
| 10 | Teacher 访问失败探针 | 上项替换两个 Teacher 构造器为失败探针；native_actor_1024 同探针 |
| 11 | 无探索 / Fusion / 训练 | ExamFrozenActor receipt；test_policy_rng_state_and_frozen_mutation_rejection；CLI training/GPU flag 负例 |
| 12 | mask/argmax 对账 | actual_f3_argmax；test_declared_actor_actions_reproduce_and_legal_action_tamper_is_rejected；原生 trace reconcile |
| 13 | hidden carry | test_1024_equals_333_new_process_691_exactly，逐状态/逐行精确比较 |
| 14 | 旧 pol 不影响 | actual_f3_argmax 中将旧 pol 全改 999，再恢复并验证冻结 |
| 15 | 参数 / slots / normalizer 冻结 | test_actual_f4_complete_schema_and_value_preserved（三模式）；verify_frozen；G_exact_cold |
| 16 | 原生 TP | test_native_tp_market_minute_window_boundaries；1024 native account |
| 17 | 原生 SL | test_native_trade_denominators_risk_fees_and_daily_accounts；1024 native account |
| 18 | 跨夜/官方日/周末 | native_actor_1024；tp_crosses_official_day_without_reset_or_forced_exit；源码 dated-tail/MACD 共享回归 |
| 19 | 持仓穿块 / 原生 roll | test_native_main_roll_and_unfinished_trade_are_distinct，1725 分钟；1024 有仓位边界 |
| 20 | 右删失 | test_censored_risk_does_not_invent_negative_sl_or_tp_label；1024 末端未平仓 |
| 21 | partial 5m 因果 / 不重复 EMA | 原 test_wp02_macd.py + test_macd_dated_tail.py；本轮 H_shared 原始 case |
| 22 | 跨合约隔离 | test_contracts_start_separate_accounts_carries_and_rng，两个不同官方日区间 |
| 23 | >512 连续=分块/恢复 | test_1024_equals_333_new_process_691_exactly，1024=333+新进程691 |
| 24 | 真成交分母 | native_trade_denominators，entries=committed ENTRY；含 TP/SL/roll/未平仓 |
| 25 | 同日 / 跨官方日 TP | test_tp_crosses_official_day_without_reset_or_forced_exit；交易 CSV 同日有效标志 |
| 26 | TP 5/15/30/60 | test_native_tp_market_minute_window_boundaries，年龄5、15、30、60、61 |
| 27 | MAE 5/15/全程 | test_adverse_tp_is_not_sl_and_exit_bar_not_underwater；native_trade_denominators |
| 28 | early adverse10 → SL | native_trade_denominators 有真实 SL 正例；adverse_tp_is_not_sl 为 TP 反例；censored_risk 为未知有效位反例 |
| 29 | underwater / 现金风险 | adverse_tp_is_not_sl：7 分钟、11点/660现金；退出 K 线不计持有 close |
| 30 | 净PnL / fee / Reward | native_trade_denominators + 跨官方日 TP，日资金/未实现/开平仓手续费逐项对账 |
| 31 | RIGHT_CENSORED / MAIN_ROLL | native_main_roll（roll=1、删失=0）；censored_risk 以及 1024 未平仓 |
| 32 | 主动 / 强制 WAIT | 决策执行 Join；全源比较、D_accounting 的 eligible/active/forced 分母和 COVERAGE |
| 33 | WAIT=0 交易不是 GO | test_all_wait_and_too_few_trades_never_become_quality_go；公平对照 WAIT TP率 null |
| 34 | ADD 遵从原 mask | 公平 ALWAYS_ADD 全原生重播，illegal_add=0 |
| 35 | 多 seed Random 可复现 | test_baseline_rng_delay_visible_history_and_no_oracle；64=23+新进程41 RNG 恢复；预注册全部3 seed |
| 36 | Delay 无未来 | 上项仅依 observe_row 的有效可见分钟；第15分钟后才ADD |
| 37 | F3 / F4 公平 | test_same_source_full_interval_baselines_and_f3_f4；共同 eval seed，保留各自原训练 seed；F4−F3 分块差值 |
| 38 | P4 不充实时基线 | pure contract P4_ORACLE 拒绝；ExamBaseline NO_ORACLE 负例 |
| 39 | 少样本 UNDERPOWERED | test_all_wait_and_too_few_trades_never_become_quality_go；test_registered_joint_checks_and_market_support_are_both_required |
| 40 | 合法可入时全 WAIT | evidence 测试区分 ALL_WAIT_WHEN_ADD_LEGAL / NO_ELIGIBLE_ADD_STATES |
| 41 | 行情独立数不重复 | 同源 10 账户只计4个 contract-days；独立市场支持测试 |
| 42 | 全新进程恢复 | 1024 cold + random baseline cold；变换 PYTHONHASHSEED 后精确一致 |
| 43 | chunk / 来源损坏拒绝 | test_strict_latest_corrupt_chunk_and_source_change_refuse；严格 latest 无回退，未提交尾块留证 |
| 44 | WP06 + WP07 无回归 | H_final_previous（原 WP07 A–H 33个用例、WP06 短评价2个）；强制共享 Guard / FullCheckpoint / Reader / MACD H_shared |
| 45 | CLI 0训练副作用 / 不覆盖 | test_synthetic_cli_plan_pause_resume_verify_compare_report；flag/path/empty/source负例；状态无agent/optimizer/replay/teacher；原Run文件不变 |

矩阵中的简名对应 `tests/entry_v1/test_wp08_*.py` 的实际函数；文件、参数化 case 和每次结果完整列在测试索引的 JUnit 收集清单中。

## 工程通过不替代的研究验收

| 验收 | 状态 | 原因 |
| --- | --- | --- |
| 正式已受训 F3/F4 的真实 DEV 完整同源考试 | NOT_RUN | 本轮只有已核验的 CPU 合成模型来源；正式模型未提供 |
| 多独立真实合约/年份泛化、统计显著性 | NOT_RUN / UNDERPOWERED | 合成10账户不是10条独立市场证据 |
| 持续优质入场 / 收益达标 | INCONCLUSIVE | 无真实受训学生考试，不推断策略有效 |
| TEST封存考试 / SEALED_TEST_READY | NOT_RUN / NO_GO | WP10 专属，当前入口拒绝 |
