# FG-S3 E+Quality TP Master Roadmap V13

2026-10-19，基于用户主机实际双仓 WP08 开发与测试。历史状态见 [V12](FG_S3_E_Quality_Master_Roadmap_20261019_V12.md)；本表只更新已经核验的工程与研究边界。

| WP | 当前工程状态 | 数据/研究状态 | 下游 |
| --- | --- | --- | --- |
| WP01 | 课程、draft/formal、身份和 Gate 已实现；WP08 保留原判断 | 本轮未授予正式训练权限 | 全链 |
| WP02 | E/MACD V3、P0–P4、L1/L2 原生教材已实现 | 生产批次数量依各自 DATA 收据，WP08 未全量生产 | Reader/DATA |
| WP03 / DATA-MULTI | 唯一 Reader 与多 Corpus F1/F2/F3 接口在公共基线 | 本轮定向 Reader 回归；不重新签旧 Corpus | F1–F4 |
| WP04 | F1 WM-only、burn-in、冻结及 CP 工程完成 | 真实正式 F1 权重本轮未验收 | F2 |
| WP05 | F2 四头与阶段参数接续工程完成 | 真实正式 F2 权重本轮未验收 | F3 |
| WP06 | F3 quality_actor 工程完成，原短诊断定向回归 | 正式受训 F3/策略质量未证明 | F4/WP08 |
| WP07 | SHADOW、监督、同 quality_actor 受控 RL、FullCheckpoint 已集成 | 原 A–H CPU 合成回归；真实 RL Gate 不开放 | WP08 |
| WP08 | A–H CODE_GO；冻结成对版本已通过独立集成树复验，239个选定去重CPU case；详见实施报告 | 独立 DEV 来源有限读取可用；正式受训 Actor DEV 考试 NOT_RUN；策略 INCONCLUSIVE | WP09/WP10 |
| WP09 | 待扩展多合约考试、分层泛化与消融 | 不因 WP08 合成成绩宣称 DATA 或策略成功 | WP10 |
| WP10 | 待封存模型、协议、阈值、seed、基线和一次最终 TEST | SEALED_TEST_FINAL 明确拒绝，未读 TEST | 最终研究审核 |

WP08 复用唯一原生 Env/ActorPolicy/Driver/Trace。独立 EvalSession 不训练，模型加载使用原 FullCheckpoint 来源验证；评价状态包含账户、RSSM 和可见 MACD 构建态。每个合约有自己的连续区间/账户，存储分块不重置。工程、DEV 证据、入场技能、质量/泛化及 TEST 权限分别判定。

下一研究动作依赖真正经认证的 F3/F4 来源及预注册评价计划；默认不自动启动真实 TRAIN 长训练、GPU、18 合约生产或 TEST。DEV/DATA/FORMAL 分级规则保持，日常修改不强制整套 G3。

源码冻结配对：Gym `09898736602c1a7d6c89899bb79fe232d7907923` / Dreamer `b7a693b3c7b0b45eeecdec954818be42d40e67b6`。实际隔离树CPU复验为40+3+1+51项，另有WP06/WP07回归与共享Reader/课程检查；不同集合有重复，清单和继承范围以报告为准。文档归档后的公共HEAD记在本机 `WP08_DELIVERY.json`。

真实命令/退出码、八阶段提交、45 项验收映射、不可变和恢复证据见 [WP08_IMPLEMENTATION_REVIEW.md](WP08_IMPLEMENTATION_REVIEW.md) 及其链接。原Run、课程、金融规则和训练Gate保持其原有认证，WP08工程成功不升级任何研究权限。
