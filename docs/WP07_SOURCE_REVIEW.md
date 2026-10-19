# WP07 反向源码 Review

范围：共同基线到 WP07 的运行源码和原始 CPU 收据。由实施者在功能实现后按调用链另做一轮源码复核；不是独立第三方认证，也不是实际 GPU/真实策略研究结论。最终配对版本、diff 文件清单、运行源码摘要和测试结果见 `WP07_COMMIT_LOCK.json`。

| 接口风险 | 实际源码与证据 | 结论 |
| --- | --- | --- |
| R1：执行 Actor 和 imagined 旧 pol 不同 | 原 Agent 只抽取 imagination_loss 和 distribution 扩展点；V2Model 仅显式 quality_online/RL 路由原 F3 distribution；E 同 logits、old-pol 扰动、实际 Actor/Val 更新及 pol 冻结 | CPU PASS；其他模式仍走原 pol，RL_MODES 未扩大 |
| R2：旧 Quality LiveRun 冒充新 F4 | OnlineRun 继承 ActorRun/LiveRun，只复用 HeadsRun.update 与 collect_to；无新优化器、金融或 reward；明确 execution spec | PASS，属于原生命周期的薄协调 |
| R3：绕过原课程/公开 Gate | execution_contract 检查旧课程上界；SHADOW/BC 是 F3 continuation，未声明 ONLINE_ADAPT 已准入；原 Guard 119 共享用例中包含必跑 20 例 | PASS；正式原 Gate 不变 |
| R4：只恢复模型，不恢复 live/historical state | 原 FullCheckpoint native backend 注册完整账户、Driver、Replay、Sampler、历史/在线 Reader、Prefetch、教师发布/消费、phase/Gates/log prefix；四种新进程下一 batch/日志/模型/slots/账户全等 | PASS_CPU；正式同版本 resume 正例 NOT_RUN |
| R5：teacher_enabled=False 仍走 Oracle query | OnlineCollector.context_snapshot 直接原 Env.export_state；identity-only disabled handle；原生动作路径 Teacher constructors/query/state raise 后完成两段 TP/SL/roll | PASS_CPU |
| R6：用当行动作/TP 冒充 L2 | 私有原 NativeQualityTeacher + QualityStrictTeacher；post-step account snapshot/完整 TRAIN suffix；原 canonical _join_row/project_label；Actor 全 WAIT 仍生成 ADD 和 WAIT；pending invalid | PASS_CPU；真实发布来源 NOT_RUN |
| R7：imag mask 无校准直接 ADD | 原 mask policy 阈值与 25000 支持、20 连续报告不变；actual multi-step calibration 绑定 WM 和原生 stepids；无政策/支持/证据 Gate 关闭 | 数值/拒绝 PASS_CPU；研究 WM_VALID NO_GO |
| R8：重复 DATA-MULTI 或强制课程哈希相同 | 原 MultiCorpusF3Stream/原 WP03 Reader；只共用 existing native compatibility helper；每份 source/course/index identity 独立锁定，F3 source 作为 ABI/math anchor | PASS_CPU；真实两个发布 Corpus 仍要求原 published_sources/native G3 |
| R9：改金融 Reward/continue/市场分钟 | Env/rewards/MACD/Oracle/E 源码无 diff；原 imagination loss 数学、原 rew/con/avail；332/333 对真实 market_advance 的 continue/discount 测试 | PASS_CPU，真实金融/WM充分性未研究 |
| 参数被动量/weight-decay 改动 | 原 positive owner allowlist + 全 state finite rollback；D/E 逐 leaf 和 slots，H_f1/f2/f3 含非零历史动量/negative-zero | PASS_CPU |
| 意外缺源自动当 draft | explicit spec_mode；draft 仅 None 源允许结构验证，空串/遗漏/TEST/非法 path/认证字段拒绝；无 Prepared/Run/CP/训练副作用 | H CLI PASS |
| 伪 JSON PASS 自开 RL | 四技术 Gate + 原 in-process native F3 authority + 单独授权与 operator-approved review SHA；审查文件绑定 Run/Gate/WM/原生 F3/evidence，不由 trainer 生成 | 真实负例 PASS；审核解析器正例只是隔离单元测试，正式授权正例 NOT_RUN |
| 最新节点损坏静默降级 | strict_latest=True，不更改其他运行默认策略；最新 pointer/node/hash/identity/commit 校验失败直接报错，外部输入在恢复 Agent 前校验 | G core PASS |
| native UUID 跨解释器无法查找 | pickle 缓存 Python bytes hash 的实际失败已复现；只重建派生 hash，保留全部 native ID/数组；不同 PYTHONHASHSEED + 移除修正 mutant | PASS，未改 stepid 含义 |

## 审查发现及处理

1. 改正 Ninjax discovery/grad 初始化模式造成的模型 schema 差异；不是在正式 loss 中回退旧 pol。
2. Teacher 在构造 fork proof 前先加载同一 seeded initial account；没有删掉 RNG 或 suffix hash。
3. 使用原 DATA-MULTI ABI/math 兼容检查；完整课程 hash 本来含来源，不强迫不同来源同 hash，也不改原文件。
4. 独立 RL 审核封存审核时的 evidence；后续合法 numeric counters 更新不使审核自动失效。每次启用仍验证当前技术 Gate、同一 WM 和 native F3 authority；改变 WM 必须重审。
5. H 冷恢复详细结果单独归档为 H_cold_cases，G 原始收据恢复为其已提交版本；发布复验使用隔离 cwd，不覆盖旧阶段历史。
6. 最终文档生成器一度使用 compact JSON 计算 runtime_hash，与原生 fingerprint 的 JSON 空格约定不同，文档断言 exit 1。已改为复用未修改的原生 helper，重新核对 SHA/全部运行源码摘要及 25 个文档链接，exit 0；见 `receipts/wp07/Release_documentation_audit.json`。只修正文档摘要，未改运行源码、准入或任何测试结果。

## 未验证的边界

正式训练过的真实 F3、两份真实发布 Corpus、真实在线 prepare/start/resume、真实多步 WM/imagined mask 充分校准、真实 RL 研究有效性、GPU 和 TEST 均 NOT_RUN。正式 Gate 保持 NO_GO。任意 Python 调用者都能修改内存/代码；本实现通过来源、权限、原生 proof、不可变发布和严格恢复边界防止正常入口自我认证，不将自签 hash 称为第三方认证。
