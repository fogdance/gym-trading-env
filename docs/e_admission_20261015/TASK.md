# 本轮任务与约束

使用 `~/Downloads/FG_S3_E_Data_Model_Reviewed_Synthesis_NO_GO_20261015.zip`，先读 README、REVIEW_REPORT、MANIFEST，继续完成 S3 E 正式训练准入。工作限于 `/home/v/Documents/work/research/`；双仓最终目标均为 `feat/s3-e-model`。

优先以 Git Bundle 保留 Data→Model 合并历史；应用前核对实际 HEAD、dirty、工作树及现有训练，不 reset 或覆盖修改。完成真实 TRAIN 每分钟接入和 A2、Tutor-CD、B2 CPU 验证；SourceAdmission direct_init、完整 Checkpoint 与恢复；生产配置与正式准入；完整 size50m GPU 1K 及恢复。全部对应 P0 通过才允许解除 E 的 waiting_e 并准备 1M。

保留 Reward、Oracle、TP5/SL30、Action Mask、安全 Gate、盲测 TEST 封存。Tutor-CD+B2 仍未实现。不得修改其他实验仓库、历史产物或原 W&B Run。正式 1M 等待用户验收，不启动。

后续用户指令：ZIP 合并后，追加合入 `~/Downloads/FG_S3_E_15m_Causality_Fix_V2.patch`。该补丁适用于 Gym，SHA256 `5416b78d18e34ff74bb7cd33e5f17bb901bbebc81f98a95dabb5dcbbffe7846c`。已先复现回归 FAIL，再应用修复，31 项 E-Data 测试通过；提交 `03c0db3`。新 schema `7f329566f5f93fc4ad3d9826a77552d0a8b8d50eea481cd5b7f3844a425aa1ca`，新 collection `a8bf803a5f882e01ffb9314fd5b1adee99f83dcee1f78be27706132854c635b3`。旧 collection 不覆盖。

最终交付：双仓 SHA 和 diff、真实 CPU/GPU 测试报告及全部失败、FullCheckpoint 恢复证据、完整来源身份、P0 与 EM01–EM18 状态及明确 GO/NO-GO。
