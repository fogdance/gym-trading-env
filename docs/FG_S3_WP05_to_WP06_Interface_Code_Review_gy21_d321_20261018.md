# FG-S3 E+Quality — WP05→WP06 实际接口 Code Review

**日期：** 2026-10-18  
**来源：** 用户提供的 `gy21.tar.bz2` / `d321.tar.bz2` 中完整 Git 对象。以 Git ref `feat/s3-e-quality-wp05-20261018` 导出源码，不能使用归档默认 HEAD（默认 HEAD 为旧 `feat/fg-s3b-certified-bc`）。  
**审查性质：** 静态源码核对 + Git 无副作用 merge-tree 试合并；未执行 GPU 神经网络训练、真实市场训练或修改用户仓库。WP05 运行成绩区分“归档报告”与“本轮重新执行”。

## 1. 已验证双仓版本及分叉

| 仓库 | WP04 父提交 | WP05 实际 HEAD | 当前 `integration/s3-next` |
|---|---|---|---|
| Gym | `b7d62113f1e8d4d7ddbdce175acd83568422420e` | `61dd64b5428588868f374389d4f3d79e0845cdac` | `0e4af49` |
| Dreamer | `9715352d4e548b82ba24d3b31070b9fa20d345a6` | `be7ce01ebbe63144e4d335b1603ca73da4fa1c79` | `fd14411` |

两个WP05分支各含A–F六个提交，各自包含其WP04父提交；当前两仓 `integration/s3-next` 已包含到WP03及 DATA CPU20/MACD V3 修复，但没有包含 WP04/WP05 的全部提交。**此处的分叉是当前最大集成风险**。归档工作树正文并非默认 HEAD，必须使用正确 ref。

`git merge-tree --write-tree --messages integration/s3-next feat/s3-e-quality-wp05-20261018` 在归档 Git 对象上的结果：Gym exit=1，唯一报告冲突 `docs/FG_S3_E_Quality_Master_Roadmap_20261018_V3.md` add/add；Dreamer exit=0。**这只证明文本合并可行，不证明课程/运行语义兼容。**

## 2. WP05 已实现的真实F2接口

| 接口 | 真正源码位置 | 结论与WP06可复用物 |
|---|---|---|
| F2 合同、目标、四头白名单 | `dreamerv3/fg_s3_e_quality_heads.py::{execution_contract,quality_configuration,train,features,active_heads,objective}` | `quality_event/ touch/ adverse/ score/` 单独更新；F2独立执行白名单是已冻结课程参数合同的子集；后续F3模仿此模式创建独立 Actor-only 执行白名单 |
| F2预检、配置、Reader及模型构造 | `src/gym_trading_env/research/fg_s3_e_quality_heads.py::{validate_settings,preflight,prepare,start_prepared,create_agent,F2WindowAdapter}` | 已使用合法 `HEADS_VALID` Reader 和原生模型；WP06增 `ACTOR_STRICT` 配置及窗口支持，保留真实来源验证 |
| F1→F2逐键参数接续 | `embodied/run/fg_s3_e_quality_transfer.py::{inspect_f1,transfer_parameters,training_evidence}` | 核查所有 non-opt 参数一致、fresh Opt/RNG/计数；WP06可用相同底层参数复制模式，但源阶段应严格为F1 **或** F2二选一，不直接resume F2运行 |
| F2运行协调/CP | `embodied/run/fg_s3_e_quality_heads.py::{HeadsRun,HeadsCounters}` | 原 `CourseStream`、`FullCheckpoint`、`RuntimeState`、`StepSchedule`、Reader RNG与prefetch；WP06扩展新stage并保存多流状态，不能重写平行Trainer |
| 模型阶段路由 | `dreamerv3/fg_s3_v2_agent.py::_S3V2Model::{__init__,train,policy_distribution}` | 当前仅 `self.f1`、`self.f2`，**尚无 self.f3**；`quality_v2`用`quality_actor`，但 F3训练必须显式路由，不可进入旧 `CONTROLLED_FULL` |
| 行为教师数学 | `dreamerv3/fg_s3_quality_v2.py::{policy_forward,teaching_objective,pair_identity,active_branches}` | 可以复用 masked logits、BC、strict pair 的数学基础；**现有 quality_pair 只比较 L1 score，存在和 L2相反的梯度**，F3需独立一致性过滤或质量损失默认为0 |
| F3 Reader 入口 | `src/gym_trading_env/research/fg_s3_e_quality_reader.py::{EQualityCourseReader.open,sample,state_dict,load_state_dict,coverage,batch_pair_counts}` | `SAMPLING_STAGES`已包含`ACTOR_STRICT`,`ACTOR_QUALITY_PAIR`；`sample(B, effective_length)`返回 `[B,effective_length+1]`，本行label自 t≥1 填入；F3取 `[B,1+K+T]`，使用K causal burn-in，不可把行为实际 action 当教师动作 |
| Teacher合法标签 | `src/gym_trading_env/research/fg_s3_e_quality_supervision.py`、`fg_s3_e_quality_reader.py` | 严格 L2 `teacher_valid`/`teacher_action` 来自 Sidecar Join；invalid 只能mask，不能补 WAIT、不能从P4执行动作推导 |
| 原模型时序基础 | `dreamerv3/fg_s3_e_quality_foundation.py::causal_context` | `1+K+T`，原始 prefix+K历史无梯度预热，目标T；F3应复用 `wm_only(training=False)` 特征并 stop_gradient |

## 3. 本轮发现的必须调整项（与WP06旧预设计的差异）

1. **F2现在具备真正的完整模型接续**：F3不应重复实现F1/F2阶段模型工厂与 optimizer schema；沿用 `create_agent/model_config` 和 `transfer_parameters` 的逐键/新优化器原则。正式来源为F1或F2，而不是固定必须F2。
2. **F2的配置路由已占有`quality_f2_protocol`/`quality_stage=QUALITY_HEADS`**：F3新增显式 `quality_f3_protocol`、`ACTOR_SUPERVISED`，先于旧 legacy-stage 守卫分派；`_S3V2Model.train`增F3分支。新阶段不需要 `PredictionAdmission`。
3. **F2的`features()`复用了WM，但只提取 features/不训练WM**：F3复用 `causal_context` 和 `wm_only(training=False)`，针对实际 `quality_actor` 输出 logits，并核对 mask/unimix仅一次。
4. **F2的状态白名单与跳过更新已工程化**：F3只授权 `quality_actor/`，其参数、Adam slots及共享时钟通过结构识别；无L2标签时整个Actor更新为0，Reader消费位置必须仍推进并可恢复。
5. **严格Pair的真实来源**：`batch_pair_counts()`以 `quality_owner` + reset/segment + L2对比计数；实际窗口无合法Pair时不能从全库有Pair推出本批有Pair。实现要独立计数，而不是把空集当梯度。
6. **旧 `teaching_objective` 的 Quality Pair不安全**：`score_high > score_low` 并不等于 `L2=ADD`；只有L2一致性认可且可证不损坏教师动作时才允许非零权重，建议代码实现开关、默认关，先用 L2 Natural BC+可选 Balanced+Strict Pair。
7. **WP05尚无真实受训F1/F2权重**：本归档 `docs/WP05_IMPLEMENTATION_REVIEW.md` 报告CPU工程验收成功、模型冻结、完整恢复，但真实F2更新=0，GPU尝试失败。因此F3代码可在合成模型上开发，真实F3 handoff须单独进行来源认证。
8. **历史课程MACD V2与主线V3分歧**：当前 `integration/s3-next` 的 `dreamerv3/fg_s3_e_quality_course_contract.py` 将 `FG_S3_E_MACD5M_CAUSAL_V2`更新为 `...V3`，增加合法交易时段尾部规则；WP05分支尚未含此改动。将来同一正式运行的Gym/Dreamer必须共用V3 Observation及签名；早期V2 Corpus仅可用于专门隔离的开发诊断，不可混进V3正式训练身份。
9. **当前不需要旧模型向后兼容**：重训新模型，合并时保留正确性与历史实验证据即可；不要为了支持旧checkpoint给F3设置第二条入口。

## 4. WP05实测证据的独立性边界

归档报告宣称CPU四头更新：四组各5个参数叶改变；349个状态叶中285个冻结，4个共享计数；完全恢复(连续4 vs 2+冷恢复+2)，选定CPU case统计报告为281 PASS。真实FG201两日只读12批，实际训练更新0。GPU沙箱/主机均未成功初始化，`cuInit CUDA_ERROR_UNKNOWN`，**不能称GPU验收通过**。上述为归档报告与源码对应审核，本轮没有重跑JAX模型测试。

## 5. 推荐的集成策略（先整合，再开发下一WP）

- **短期**：建立一次性 `integration/fg-s3-e-quality-current` 双仓测试集成 worktree，以当前 `integration/s3-next` 作父分支、将WP04/WP05双仓功能分支做合并。Gym处理唯一Roadmap add/add冲突为新的V8汇总；Dreamer自动Git合并也必须做源合同验收。不要在训练worktree执行直接reset/checkout/clean。
- **关键语义审计**：统一MACD V3双方课程定义和Gym Producer/Observation/Teacher源身份；运行课程/Reward/Reader/因果特征/F1/F2定向CPU验证，至少合成 F1→F2 接续及 exact resume、真实TRAIN只读小样。若源码或课程需要变更则使用新提交新身份；不得把V2来源重签为V3。失败不合入 `s3-next`。
- **通过后**：双仓各有确定合并SHA、配对记录、选择测试日志；将已验收功能合并进`integration/s3-next`。WP06从这一对HEAD再建立独立feature worktree。不要让WP06再次从WP05旧分叉继续叠加新功能。
- **DATA-MULTI、EVAL-BASE**：完成各自定向验收后再成对/按影响仓库合并；涉及共享Agent/Reader的改动先在集成分支解决，避免覆盖WP06。纯分析/文档无需为了合并而阻断训练代码，但最终总路线图应更新。
- **FORMAL**：只从已冻结的双仓成对提交+课程数据版本启动训练；正在运行的实验继续引用其原始已冻结提交，不能因主线推进而自动升级。

## 6. 本轮结论

- **WP05：CPU CODE_GO（基于报告与静态核对），FORMAL TRAIN / GPU / 研究成绩不等于GO。**
- **WP06：真实接口可识别，具备生成Codex正式开发任务书的条件。**
- **直接无审查合并：NO-GO。** 文本合并风险较小，课程/特征身份V2→V3必须先统一并定向核验。
