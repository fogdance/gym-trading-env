# FG-S3 E + Quality TP：Master Roadmap V3

日期：2026-10-18。本版更新 DATA 开发与生产状态；[V2](FG_S3_E_Quality_Master_Roadmap_20261016_V2.md) 保留历史设计及验收记录。最终运行版本见 [Producer pin](data_macd_fix_20261018/locks/PRODUCER_PIN.json)，实际集成结果见 [双仓收据](data_macd_fix_20261018/evidence/INTEGRATION_RESULT.json)。

学生仍从零训练 1m/15m/实时 partial 5m MACD 的 E Encoder、RSSM、WM、Quality Heads 与 Quality Actor。老师仍是 Quality V2 L1/L2；Single3 做空三手、TP5、SL30、原生 Quality TP Reward 不变。新旧模型不造兼容；历史训练和 Corpus 保留冻结身份。

## 当前状态与依赖

| 工作包 | 当前状态 | 下一验收及依赖 |
| --- | --- | --- |
| WP01 课程合同 | 已实现；MACD 数学子合同升级 V3，Observation/Course/Corpus 内容身份重锁 | V2 MACD Snapshot/Corpus 不在 V3 中冒名恢复或准入 |
| WP02 原生教材与老师 | 保留原生六行为生产器；日期、外部输出、worker 扩展保留 | 新四批正式生产和独立 G3 均 NOT_RUN |
| WP03 Join 与 Reader | 既有统一 Reader；V3 有限 DEV 小样见修复收据 | DEV 小样只能证明通路；不能替代整批认证或研究覆盖 |
| WP04 FOUNDATION_WM | 用户在独立 worktree 开发；本轮未检查/修改 | 其优化器、冻结、恢复及阶段验收 NOT_RUN；由 WP04 独立交付 |
| WP05 QUALITY_HEADS | 未开始 | WP04 WM、认证教材与有效 L1 目标；只训练 Quality Heads |
| WP06 ACTOR_SUPERVISED | 未开始 | WP04 WM、认证 L2/Pair 教材；只训练 Quality Actor；不以 PredictionAdmission 作为 F3 硬前提 |
| WP07 在线适配 | 未开始 | 阶段交接、Replay/Reader RNG、独立 Gate 和训练授权 |
| WP08 Actor-only 考试 | 未开始 | 独立真实 Actor、冻结配置、隔离评估协议；教师收益不能冒充学生表现 |
| WP09 多合约课程扩展 | DATA 与 WP04 并行；四批范围已列明 | 由用户逐批生产后看质量覆盖；不自动扩到全部18合约 |
| WP10 研究冻结交付 | 未开始 | 完整参数、源码、数据、Checkpoint、恢复与考试证据 |

## DATA 任务

| 任务 | 当前交付 | 正式状态 |
| --- | --- | --- |
| DATA-A 来源、范围与预算 | 保留显式 TRAIN 官方日、`/data/s3_corpus` 外部输出、账户+标签 worker 总预算 | 用户预算20逻辑线程；不再跑20核占用验证 |
| DATA-B MACD 阻断 | V3 第1～4槽持续输出 partial，第5槽只提交一次；认证区间尾部不提交；零时长结束标记不供价格 | 同 session 缺分钟、重复、乱序、晚到仍拒绝；无补造分钟 |
| DATA-C 有限 DEV | 同一 Producer `--mode dev --max-market-steps 4 --workers 1`，真实 TRAIN 后缀及六账户 | DEV_ONLY / NOT_CERTIFIED；Native G3 准入拒绝 DEV 产物 |
| DATA-D 逐批认证 | DATA 每批生成后独立 Native G3、Reader、身份与质量收据 | 新 B01–B04 正式生成/G3/整批Reader NOT_RUN，命令交用户 |
| DATA-E 覆盖 | [覆盖报告](../CORPUS_COVERAGE_REPORT.md) 区分历史 V2、有限 DEV 与正式计划 | 不从小样推出跨合约、跨日期泛化或正式训练 GO |

## 开发、生产与训练边界

- DEV：允许 dirty 开发和有界 TRAIN CPU 调试；使用真实金融与因果特征，不要求每次改动冻结并跑完整 G3。
- DATA：最终发布版本冻结源码、锁定真实 TRAIN 范围；每批完整 G3、独立 Join/Reader/覆盖；正式生产由用户运行。
- FORMAL：阶段训练、优化器、GPU 和隔离评估各自需要授权；DATA 成功不解除训练 Guard。

## 风险及迁移

1. 原日历由用户确认的完整来源生成，其 `dated_sessions` 是实际时间连续段；本轮没有新增交易所独立日历认证，也没有按观察到的跳时补 K 线。
2. FG809 的零时长行由现有 Env 定义为 mark-only。本轮保持金融撮合与市场分钟语义，仅让 MACD 识别可见的认证结束标记。
3. 旧 `63f2f8c/446d1bc` 冻结生产目录和 pin 保留为失败尝试。新版本使用 `/data/s3_corpus/early_train_20261018_cpu20_v2`，禁止覆盖旧工件或重签旧 Corpus。
4. DATA 依赖完整 WP01–03 提交链；集成前核对祖先与 clean 状态，向共用基线快进。WP04 worktree 保持原样，其后续接入由该开发线审查。
5. 旧 FG201 V2 的 G3/Reader/Loss PASS 仅是历史证据；V3 要求新数学身份及教材。真实优化器、GPU、TEST 价格读取和20核运行验证本轮均 NOT_RUN。

## 变更记录与证据

| 版本 | 日期 | 变化 |
| --- | --- | --- |
| V1/V2 | 2026-10-16 | WP01–10 路线、WP03 A–D 与历史 V2 小样证据，详见保留的 V2 |
| V3 | 2026-10-18 | DATA MACD V3 修复、实时 partial 与结束标记回归、DEV/DATA/FORMAL 分层、新生产版本锁与共用基线集成 |

实施与精确验证清单见 [修复报告](../DATA_PRODUCER_FIX_REPORT.md)、[生产计划](../TRAIN_CORPUS_PRODUCTION_PLAN.md)、[原始收据](data_macd_fix_20261018/evidence)。
