# E+Quality Master Roadmap V9：第一个P0 DATA-QUALITY现场审计

2026-10-18。继承 [V8](FG_S3_E_Quality_Master_Roadmap_20261018_V8.md) 的WP01–WP10及WP04/WP05集成状态，不变更模型、课程数学或训练授权。本轮仅处理并行工作流任务书的第一个P0。

| 工作流 | 本轮实际状态 | 下一条件 |
| --- | --- | --- |
| 第一个P0 DATA-QUALITY | B01用户正式生产、Native G3 GO、独立Reader PASS；质量分布/分母、缺题型、速度成本报告已同步；首批闭环完成 | B02已选2020-08-07/10/11，SELECTED_NOT_RUN；扩大跨合约覆盖尚未完成 |
| 第二个P0 DATA-MULTI | 本轮未开展/未验收 | 工程可合成验证；真实最终验收至少2份已认证Corpus，当前本线仅1份 |
| P1 EVAL-BASE | 本轮未开展/未验收 | 保持Actor-only和Teacher隔离；DEV/TEST另授权 |
| P1 TRAIN-PLAN | 本轮未开展/未验收 | 数据充分性、资源及F1计划，不启动真实训练 |
| WP04/WP05/其他WP | 维持V8及现有集成版本 | DATA成功不替代真实F1/F2/F3训练或研究验收 |

B01：FG809，2018-08-06/07/08，6210物理行、1034独立cursor、3日。Reader覆盖LIMITED：P1无HEADS_VALID 64步窗口；实际SL/右删失0；快速TP/SL候选各仅5个独立位置。

当前共用运行基线为Gym `26121ab257b843c303e9d916c4dcf598eb8b1a97` / Dreamer `0e39dfd498f9146e10f966cfe118a24b20e5143e`。B01生产者仍固定Gym `b0d5ef9af07fbdd0c553dd943ac9eb40c9a84d46` / Dreamer `fd144113ce706e2d8f1ce8e7828ece0799c8bf7a`，不以当前HEAD重签已发布数据。

当前报告：[DATA-QUALITY成果](DATA_QUALITY_B01_AUDIT_20261018.md)、[Corpus覆盖](../CORPUS_COVERAGE_REPORT.md)、[生产计划](../TRAIN_CORPUS_PRODUCTION_PLAN.md)。本轮只读核对125个现场文件SHA并重算标签统计，未重跑生成/G3、未读取TEST、未用GPU或真实优化器、未修改WP04 worktree；其他三批仍NOT_RUN。
