# WP04_IMPLEMENTATION_REVIEW

**结论：WP04_CODE_GO / WP04_REAL_DATA_READ_GO；WP04_TRAIN_GO=NO_GO，WP04_MODEL_RESEARCH_GO=NO_GO。** A→E 的工程交付完成；本轮只进行 CPU 合成更新和有限真实 TRAIN 只读。正式真实 F1 基础权重尚未产生。

## 分支、基线与提交

`feat/s3-e-quality-wp04-20261018`，成对独立 worktrees：

- Gym：`/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env`
- Dreamer：`/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/dreamerv3`

从用户指定 WP03 Gym `7366886f3b40bf74aa4485978ffcc97e39d0d53e` / Dreamer `446d1bc99debd2d05c313eafd7d0c414ebd2815c` 建立；原 WP02/WP03 树仍 clean，未合并 integration，未改在用训练树。原命令及复核见 [基线](receipts/wp04/repository_baseline.json) / [最终保护审查](receipts/wp04/E_repository_protection_audit.json)。

| Commit | Gym SHA | Dreamer SHA |
|---|---|---|
| A | `fa75f0151a8081342fbb3adf83e50970ae381bd7` | `d7bcbcd2a189de75cef57122d61219341b11e65d` |
| B | `a74c69883725b24dd2f08c9c19e1717aeca04828` | `a2cb48e5aa115d7a66de41a0efd66073ced1bf4c` |
| C | `f3140367e993e4cd0d502a43c17a782e4098d355` | `09fc1223933f5aad828eb5f016958338b3787520` |
| D | `22fbb521104e173dc81ff354c953e5a2fa3d90a6` | `c6c35b0cb06b0947a9ebf1bc5a068f21b5b134ca` |
| E | 见 [交付锁](WP04_DELIVERY_COMMIT_LOCK.json) | 见同一交付锁 |

E 的完整 SHA 在提交完成后写交付锁，避免自引用 SHA；该锁作为本地 docs 交付文件保留。A–D 源码、E 集成文档及原始结果均已纳入 Git；没有重新导出 patch。

## 实现

A：显式 F1 课程、fresh 完整 E+Quality 初始化和工程/数据/正式训练准入分层；复用原统一 CLI 和 WP03 Source。B：原 wm_only/单优化器，T+K+1 native byte-bound 窗口，K detached burn-in，六 WM 前缀允许更新，其他参数和 slot 精确冻结，非有限候选完整拒绝。C：原每项 Loss、native Reward/Avail 条件诊断、多尺度梯度/消融、学习支持与病理 NO_SUPPORT。D：原 FullCheckpoint 外部 immutable Reader 后端、原 Prefetch/seed、完整计数与精确冷恢复。E：当前源接口、真实 G3/只读、共享回归及逐条源码审查；FORMAL 保持原 10 GiB checkpoint 磁盘保留，合成诊断为 1 GiB。

没有修改金融 Env、Reward、E 特征、MACD、Oracle/RSSM 的算法，没有新 Replay/WM/Optimizer/在线采集系统。Reader Index V3 只增加原生过去执行事件审计元数据；原 Corpus 不改，旧 V2 cache 不冒充本版本，其余 WP03 Reader 视图回归通过。

## 真实测试与负例

**379 个不同的选定用例按最终结果口径 PASS，0 FAIL，0 SKIP；不是全项目测试。** 所有实际命令、退出码、耗时、逐项清单与历次失败见 [测试总表](WP04_TEST_RESULTS.md) / [机器可核对清单](receipts/wp04/E_stage_results.json)。

| 最终集合 | 结果 | exit |
|---|---|---:|
| WP04 A/B/C 原生模型与优化器 | 40 PASS | 0 |
| F1 冷恢复、损坏与合法 K/seed/model/sampling/optimizer/loss/code 改动拒绝（最终 E_restore_final） | 22 PASS | 0 |
| 原 FullCheckpoint/retention | 15 PASS | 0 |
| 必跑 Gate / SourceTransfer / formal guards | 20 PASS | 0 |
| WP01/02/03、账户、时钟、Reward | 246 PASS | 0 |
| E+A2/Tutor/B2 与账户恢复 | 6 PASS | 0 |
| WP03 原 Teacher loss | 11 PASS | 0 |
| E 模型/MACD 共享 | 18 PASS + 修正单项重跑 1 PASS | 原命令1；修正项0 |

负例实跑包括：无授权无 Run 副作用，伪造 native dataclass/loaded JSON，课程/Reward/Observation/fold/legacy Q/Soft，未知参数/slot，账户混入/乱序/stepid错移/action或reward改写/+0↔-0、carry污染、K/T不足，有限权重溢出、继承动量/weight decay，source/index/配置/代码身份变化、损坏最新 bundle、全零 Reward/无 ADD/全部删失及多尺度无效位隐藏值。

失败不是表面消除：保留 Ninjax 参数读取、初始化重入、canonical warmup 首步 0、离散单次采样、不同矩阵形状 float32、tuple/list 身份及坏负例夹具的原记录；修复说明见 [C Review](WP04_C_REVIEW.md)、[D Review](WP04_D_REVIEW.md) 与 [源码反向 Review](WP04_SOURCE_REVIEW.md)。源码审计没有放行旧 WP03→F1 迁移。

## 参数和恢复

实际 349 个 Ninjax 叶，205 个允许变化、144 个精确冻结；有效样本上 enc 15/15、dyn 27/27、dec 7/7、rew 5/5、con 5/5、avail 8/8 叶变化，15m 与 MACD 分支均变化。所有原 Actor/Critic/slowval/35 个 Quality 参数叶及对应 slot 不变。详见 [前后哈希与恢复证据](WP04_PARAMETER_AND_RESTORE_EVIDENCE.md)。

B1/T64/K32/seed202，4 次 CPU 合成更新；连续4与2+冷进程+2的 Agent/优化器/Reader RNG/coverage/Python-NumPy RNG/原 Prefetch queue/下一 batch/seed/计数/schedule/指标 bitwise 相同。最终原件持久保留在本 worktree artifacts；复核脚本再次读取其字节，exit0。K32 只表示有限历史近似，不冒称完整 episode 隐状态。

## G0 / G1 / G2 与真实数据

| 层级 | 本轮结论 | 证据/边界 |
|---|---|---|
| G0 | GO，工程 | 原金融 Env 合成源、真实单优化器更新、冻结和冷恢复；非市场研究结论 |
| G1 | GO，只读 | 本轮原生 G3 1288.11 秒、exit0；六账户各691行，全部 teacher/account 重播通过，TEST打开数0；不是继承旧报告 |
| G2 | NOT_RUN | 没有多日多合约/18合约数据生产 |

真实 source 为 FG201 主力资格末两官方交易日（2021-12-08/09），单合约 4146 物理行、690 独立市场时刻/六账户，原 119 成交=103TP+12SL+4主力换月，原资金/删失与独立有效位保持。有限只读新读取六策略各2窗，768 target +384 burn-in，128 独立 target 市场时刻、768 个 target 账户行；同一市场不能按六账户乘六。所有349叶前后哈希相同，实际真实更新0。见 [新 G3 原件](receipts/wp04/G1_native/native_result.json)、[只读报告](receipts/wp04/G1_read_only/read_check.json)、[覆盖/权限独立复核](receipts/wp04/G1_coverage_evidence.json)。

G1 的 read-check 收据本身是 development/非训练诊断；本报告将其与同内容 Corpus 的本轮实际 G3 执行交叉核对，不把加载的 G3 JSON 转成训练授权。正式 f1-prepare/start/resume 仍要对应 FORMAL 准入与外部授权。

## 合同哈希

| 身份 | SHA256 |
|---|---|
| corpus | `640865b4239fb483e5d98daf18d3104e670ff9b3c1486cb63ec9924cf6e089a9` |
| source | `85405da25efa3c850fb1b14ec821c53987e61ead7f233073d2b7dd85a07e9081` |
| course | `14cadcdaaa7d96bc0f33f491ef38741494957ee8e7423a1ba5bf598fe230baba` |
| source_model_contract | `7152f9de6de4885b0e64c77f003c7a646a9807364c4019034ff73aa2d76cad1b` |
| observation | `5ca333e52049bf5bdfe0bf18f5519e65766363c946648bdd1f45b525558052b8` |
| resolved_size1m_F1_model | `6fbbf79cce3ddabe5471c532c12166287f175b2e78642ae505d9b8e493810b6e` |
| actual_controlled_small_F1_model | `d1afc98ec90425a421760acbc9937cd5f138ce83099a243b5f98cbb2c264a8f2` |
| reader_index | `bd2b6bc7e8095bfd1bd051392cbd86347476d000d503c9a6bfa8b4f63aa73fcb` |

完整协议、维度、时钟白名单、参数矩阵与状态在 [F1 接口锁](WP04_F1_INTERFACE_LOCK.json)。`resolved_size1m_F1_model` 是正式 fresh 解析合同，未对真实数据更新；`actual_controlled_small_F1_model` 是本轮 CPU 工程模型，不冒充正式 size1m 训练结果。

## 未运行和边界

真实 TRAIN 参数更新、GPU、正式授权的 f1-start/resume 端到端、1K/10K/1M pilot、真实 F1 权重、Actor 自主交易效果/泛化、TEST、18合约全量、W&B 发布及外部人工 Review 均 **NOT_RUN**。对应训练/研究 GO 均 **NO_GO**。未经追加授权不得以本轮工程 PASS 开启上述工作。

用户最新 [DEV/DATA/FORMAL 分级](FG_S3_Development_Admission_Levels_20261018.md) 已纳入交付：日常开发可以临时目录、合成/有限 TRAIN、少量定向测试，失败可修复重跑，不要求每次冻结所有 SHA 或重做 G3。DATA 发布和 FORMAL 训练保留严格身份、来源、完整恢复及授权；金融、MACD 因果与后验隔离在三级均必须保持。

下一步可开发 WP05/WP06 工程；真实 F1 基础权重仍需独立授权、冻结配置与正式准备。总体范围 WP01–WP10 未缩减，见 [Master Roadmap V4](FG_S3_E_Quality_Master_Roadmap_20261018_V4.md)。
