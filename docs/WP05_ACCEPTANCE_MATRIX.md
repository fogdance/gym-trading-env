# WP05 acceptance matrix

54 engineering behavior checks. PASS_ENGINEERING does not mean a real FORMAL F1/F2 positive or research acceptance. Source formal real positive, current native FORMAL G3, independent holdout and GPU successful update remain NOT_RUN/NO_GO. GPU attempt failed before updates.

| ID | Behavior | Engineering result / actual test | Receipt |
|---|---|---|---|
| 01 | 四头执行子集、课程身份不变 | PASS_ENGINEERING; `test_A01_subset_preserves_original_course` | `F_protected_source_audit.json` |
| 02 | 明示 F2 路由，拒绝 legacy 混用 | PASS_ENGINEERING; `test_A02_protocol_and_guard_negative` | `F_contract_transfer.xml` |
| 03 | 原 F1 保持 | PASS_ENGINEERING; `test_B09_K32_T64_four_native_WM_updates_without_actor_forward` | `C_F1_legacy.xml` |
| 04 | 原 Reward/Obs/MACD 合同 | PASS_ENGINEERING; `test_B03_source_failures` | `F_protected_source_audit.json` |
| 05 | 配置非法值/路径拒绝 | PASS_ENGINEERING; `test_A03_settings_negatives` | `F_contract_transfer.xml` |
| 06 | legacy 正式 Guard 保持 | PASS_ENGINEERING; `test_F01_no_authorization_exit_one_no_output` | `F_shared.xml` |
| 07 | F1 完整性、工程不可冒充正式 | PASS_ENGINEERING; `test_B03_source_failures` | `F_formal_records_complete.xml` |
| 08 | 全部 non-opt 逐键等字节 | PASS_ENGINEERING; `test_B01_nonopt_complete_fresh_optimizer_and_mirror` | `B_transfer_receipt.json` |
| 09 | fresh 优化器与计数 | PASS_ENGINEERING; `test_B01_nonopt_complete_fresh_optimizer_and_mirror` | `B_transfer_receipt.json` |
| 10 | 源节点不修改 | PASS_ENGINEERING; `test_B01_nonopt_complete_fresh_optimizer_and_mirror` | `F_repository_protection.json` |
| 11 | shape/dtype 拒绝 | PASS_ENGINEERING; `test_B02_parameter_failure_before_mutation` | `F_contract_transfer.xml` |
| 12 | 缺键、坏 hash/节点拒绝 | PASS_ENGINEERING; `test_B03_source_failures` | `F_contract_transfer.xml` |
| 13 | Policy mirror 同步 | PASS_ENGINEERING; `test_B01_nonopt_complete_fresh_optimizer_and_mirror` | `B_transfer_receipt.json` |
| 14 | 课程/Reward/Clock 来源不符拒绝 | PASS_ENGINEERING; `test_F05_real_read_smoke_cannot_prepare_with_synthetic_F1` | `F_boundary_review.xml` |
| 15 | 阶段计数与 Reader 独立 | PASS_ENGINEERING; `test_E03_skip_checkpoint_preserves_consumed_reader_and_does_not_handoff_again` | `E_skip_evidence.json` |
| 16 | event 7×4 原 CE | PASS_ENGINEERING; `test_C01_each_head_updates_only_its_params_and_slots` | `C_event_delta.json` |
| 17 | touch 4×2 原 BCE | PASS_ENGINEERING; `test_C01_each_head_updates_only_its_params_and_slots` | `C_touch_delta.json` |
| 18 | adverse 4 原 BCE | PASS_ENGINEERING; `test_C01_each_head_updates_only_its_params_and_slots` | `C_adverse_delta.json` |
| 19 | score[0,100] 原数学 | PASS_ENGINEERING; `test_C01_each_head_updates_only_its_params_and_slots` | `C_score_delta.json` |
| 20 | 独立 valid mask | PASS_ENGINEERING; `test_C01_each_head_updates_only_its_params_and_slots` | `F_heads_supported.xml` |
| 21 | 删失 invalid 非 OPEN | PASS_ENGINEERING; `test_C07_invalid_placeholders_safe_no_q_computation` | `F_reader_clock.xml` |
| 22 | mono 双有效且仅市场分钟序列 | PASS_ENGINEERING; `test_C07_invalid_placeholders_safe_no_q_computation` | `F_heads_supported.xml` |
| 23 | score≥80 辅助及指标 | PASS_ENGINEERING; `test_prediction_metrics_compare_to_train_prior_and_keep_censor_missing` | `D_diagnostics.xml` |
| 24 | 无效 NaN/类别，合法目标非法拒绝 | PASS_ENGINEERING; `test_D03_window_and_projection_negatives` | `F_heads_supported.xml` |
| 25 | Q/Soft 无训练路径 | PASS_ENGINEERING; `test_D03_window_and_projection_negatives` | `F_heads_supported.xml` |
| 26 | L2 仅 Sidecar、不进 F2 | PASS_ENGINEERING; `test_D05_L2_and_invalid_Q_values_never_enter_F2_inputs` | `F_boundary_review.xml` |
| 27 | Teacher 扰动不改表征/预测 | PASS_ENGINEERING; `test_C03_teacher_changes_loss_only_and_burnin_is_not_supervised` | `F_heads_supported.xml` |
| 28 | 各头独立真实参数/slot 更新 | PASS_ENGINEERING; `test_C01_each_head_updates_only_its_params_and_slots` | `C_*_delta.json` |
| 29 | 单无效头 params+slots 精确冻结 | PASS_ENGINEERING; `test_C01_each_head_updates_only_its_params_and_slots` | `F_heads_supported.xml` |
| 30 | 全无效跳过且不推进优化器 | PASS_ENGINEERING; `test_C02_all_invalid_skips_with_fresh_or_inherited_momentum` | `E_skip_evidence.json` |
| 31 | WM 全组冻结 | PASS_ENGINEERING; `test_C04_frozen_nonzero_momentum_negative_zero` | `E_inherited_foundation_freeze.json` |
| 32 | Q/Actor/Critic/normalizer 冻结 | PASS_ENGINEERING; `test_C04_frozen_nonzero_momentum_negative_zero` | `E_inherited_foundation_freeze.json` |
| 33 | 共享优化器时钟规范 | PASS_ENGINEERING; `test_C02_all_invalid_skips_with_fresh_or_inherited_momentum` | `E_skip_evidence.json` |
| 34 | 非 finite 整次拒绝/回滚 | PASS_ENGINEERING; `test_C05_nonfinite_rejected_and_all_state_rolled_back` | `F_heads_supported.xml` |
| 35 | 未知叶与动态权限注入拒绝 | PASS_ENGINEERING; `test_C09_positive_allowlist_rejects_injected_wm_rule` | `F_heads_supported.xml` |
| 36 | 权重关闭不激活头 | PASS_ENGINEERING; `test_C08_disabled_weights_do_not_activate_heads` | `F_heads_supported.xml` |
| 37 | F1/legacy 回归 | PASS_ENGINEERING; `test_quality_first_heads_only_updates_and_actor_isolation` | `C_F1_legacy.xml` |
| 38 | K+T+1、Owner/Stepid 原生时序 | PASS_ENGINEERING; `test_D01_unique_reader_modes_rng_and_native_owner` | `F_heads_supported.xml` |
| 39 | NATURAL/MIXED/STRATIFIED | PASS_ENGINEERING; `test_D01_unique_reader_modes_rng_and_native_owner` | `F_heads_supported.xml` |
| 40 | 目标段/启用损失支持不足拒绝、不重采样 | PASS_ENGINEERING; `test_D06_burnin_only_labels_rejected_without_resampling_or_rng_change` | `F_heads_supported.xml` |
| 41 | 市场/账户分母分开 | PASS_ENGINEERING; `test_D02_four_head_read_only_metrics_and_static_ablations` | `F_real_read_02/read_check.json` |
| 42 | 自然评价不混分层 | PASS_ENGINEERING; `test_D01_unique_reader_modes_rng_and_native_owner` | `D_diagnostics.json` |
| 43 | 稀缺类别 null 指标 | PASS_ENGINEERING; `test_D02_four_head_read_only_metrics_and_static_ablations` | `F_heads_supported.xml` |
| 44 | MACD/15m/1m 因果前向 | PASS_ENGINEERING; `test_D04_visible_native_macd_and_context_reach_F2` | `D_visible_multiscale.json` |
| 45 | 4 vs 2+冷进程+2 下一批 | PASS_ENGINEERING; `test_E01_four_vs_two_cold_two_full_exact` | `E_cold_evidence.json` |
| 46 | Agent/Opt/RNG/queue/计数/日程精确 | PASS_ENGINEERING; `test_E01_four_vs_two_cold_two_full_exact` | `E_cold_evidence.json` |
| 47 | 阶段上游 F1 身份固定 | PASS_ENGINEERING; `test_E02_resume_failures_before_model_or_run_writes` | `E_checkpoint.xml` |
| 48 | 来源/教师/index/代码更改拒绝 | PASS_ENGINEERING; `test_C08_open_reader_detects_changed_sidecar_or_chunk` | `F_reader_clock.xml` |
| 49 | 坏最新节点不回退 | PASS_ENGINEERING; `test_E02_resume_failures_before_model_or_run_writes` | `E_checkpoint.xml` |
| 50 | 外部 Corpus 不整份复制 | PASS_ENGINEERING; `test_E01_four_vs_two_cold_two_full_exact` | `E_cold_evidence.json` |
| 51 | F1 不改、恢复不再 handoff | PASS_ENGINEERING; `test_E03_skip_checkpoint_preserves_consumed_reader_and_does_not_handoff_again` | `F_repository_protection.json` |
| 52 | CLI exit/输出/伪认证负例 | PASS_ENGINEERING; `test_F03_loaded_or_forged_native_json_not_authority` | `F_cli.xml` |
| 53 | CPU 与有限真实 TRAIN 只读 | PASS_ENGINEERING; `test_F04_read_only_route_does_not_consume_an_unrelated_F1` | `F_real_read.json` |
| 54 | Review、Roadmap、真实 SHA/NOT_RUN | PASS_ENGINEERING; `test_F06_formal_record_checker_rejects_missing_or_fake_updates` | `WP05_DELIVERY_COMMIT_LOCK.json` |

07: real FORMAL trained-source positive NOT_RUN; flags-only and fake updates rejected. 42: independent TRAIN suffix/contract holdout UNSUPPORTED on current smoke source. 45–51: postcommit final-HEAD actual repeat lives in F_final_recovery_v2.*. 53: actual real TRAIN optimizer updates 0. 54: postcommit commit lock has actual six paired SHAs. No missing research item is declared PASS.
