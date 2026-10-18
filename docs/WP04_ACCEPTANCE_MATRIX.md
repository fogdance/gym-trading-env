# WP04 42 项验收对照

PASS 表示此次授权范围内的工程证据。正式真实更新、GPU、研究成绩未授权，不能由工程 PASS 替代。

| # | 合同 | 结论 | 实际证据 |
|---:|---|---|---|
| 1 | 双仓共同 WP03 祖先、隔离 worktree、原树未改 | PASS | repository_baseline / E_repository_protection_audit |
| 2 | 新实际 Corpus/index + 原生 G3，不复用旧 PASS | PASS | G1_native_reverify / G1_native/native_result / G1_coverage_evidence |
| 3 | FOUNDATION_WM 与 WM_NATURAL 分开，错位拒绝 | PASS | A02 / B01 / D02 |
| 4 | 原生 Quality TP + E/MACD V2，金融规则不改 | PASS | E_contracts / E_repository_protection_audit |
| 5 | Fresh 完整模型，不继承旧 1M | PASS | A05 / E_wp04 |
| 6 | 工程/数据/训练/研究状态分层 | PASS | A01 / A04 / read_check / interface lock |
| 7 | 真实 G3 + 外部授权；loaded JSON 不能放行 | PASS_NEGATIVE; FORMAL_START_NOT_RUN | A01/A04 / D03；正式正向启动未授权未运行 |
| 8 | 公开非 F1、Prediction/Q/Decision Guard | PASS | A_guards / E_required_final |
| 9 | 唯一 Reader，T+K+1，后验隔离 | PASS | B01 / C03 |
| 10 | 真实 stepid/action/prevact/reward 对齐 | PASS | B02 / B03 / native G3 |
| 11 | 原 is_first/last/terminal/market_advance、账户隔离 | PASS | B02 / E_contracts / G3 |
| 12 | K 无损失/无梯度，只 T 目标 | PASS | B05 / C04 |
| 13 | 支持不足明确失败，不截断补样 | PASS | B08 |
| 14 | 随机窗每窗新 carry | PASS | B06 / D runtime carry=None |
| 15 | 重复曝光与独立市场分别计数 | PASS | G1_coverage_evidence / C_multiscale_learning / D_persisted_full_state_audit |
| 16 | 重写全部后验不改变 WM loss/grad | PASS | B01 / C03 / E_teacher_loss |
| 17 | Reward 来自 Replay/Env，不来自老师 | PASS | C04 / G3 / reward unit tests |
| 18 | 原 Loss 键与权重，非六个参数前缀 | PASS | B05 / C04 / C evidence |
| 19 | 仅 1m 与 position 两种重建 | PASS | B05 / C04 |
| 20 | 市场分钟折扣和真实终止 | PASS | C04 / test_s3_market_minute / E_legacy_resume |
| 21 | 初始化未来 F2/F3 Heads/Actor | PASS | A05 / 349-leaf layout |
| 22 | enc 非零梯度与参数变化 | PASS | C01 / E_parameter_summary |
| 23 | dyn 梯度与参数变化 | PASS | B03 / C01 |
| 24 | dec 既有目标真实更新 | PASS | B03 / B05 / C01 |
| 25 | rew 非零原生事件学习信号 | PASS | C01 / C04 TP+SL conditional losses |
| 26 | con 有效学习梯度 | PASS | C01 / C04 |
| 27 | avail 更新、post/prior ADD recall/FPR | PASS | C01 / C04 / G1 read |
| 28 | 全部 Actor/Critic/Quality 参数 bitwise 冻结 | PASS | B03/B04 / C01 / D01 |
| 29 | 冻结 slots/动量/weight decay，共享时钟例外 | PASS | B04/B11 / D01 / parameter evidence |
| 30 | 未知前缀、状态、slots 拒绝 | PASS | B07 / D05 |
| 31 | 零 Reward / 无 ADD / 全删失不冒称学习 GO | PASS | C05 / support always research_go=False |
| 32 | 有效多尺度敏感，无效位下隐藏值不敏感 | PASS | C01/C02 / MACD tests |
| 33 | Q/Soft/融合/RL/BC/Pair/DAgger OFF | PASS | A06 / B09 / C03 / E_teacher_loss |
| 34 | GPU/旧 1M 权重未运行 | NOT_RUN_AS_REQUIRED | CPU environment in all command receipts; no weights inherited |
| 35 | 真实冷进程 N 与 M+N-M next batch 相同 | PASS | E_restore_final / E_persisted_full_state_audit |
| 36 | 全部 Agent/opt/RNG/coverage/schedule/metrics 精确 | PASS | E_restore_final / E_persisted_full_state_audit |
| 37 | 外部 Corpus 不整份复制，save/load 重验 | PASS | D04 / D01 / no NPZ/Replay files；仅 pending batch |
| 38 | source/index/stage/spec/model/Reward/code/K/opt 变化拒绝 | PASS | E_valid_identity（先正向恢复，再合法改配置）/ D04/D05 / E_required_final mutation checks |
| 39 | 破损/不同身份不静默 fresh 或回滚 step0 | PASS | D04 / FullCheckpoint external backend |
| 40 | WP03 其余三视图仍有效 | PASS | E_contracts test_wp03_reader / join / admission |
| 41 | WP01/02/03 与共享准入/模型/恢复回归 | PASS | E_contracts246 / E_teacher_loss11 / legacy6 / shared20 / C shared19 |
| 42 | 原始失败、修复、测试清单、SHA 与未运行项 | PASS | WP04_TEST_RESULTS / E_stage_results / commit lock |
