# WP03_IMPLEMENTATION_REVIEW

任务书版本日期：2026-10-16；最终提交收尾日期：2026-10-18。

## Summary

双仓分支：`feat/s3-e-quality-wp03-v2-20261016`。
Gym为本报告所在Commit D；Dreamer HEAD `446d1bc99debd2d05c313eafd7d0c414ebd2815c`。提交后的完整双仓锁见[release_lock.json](receipts/wp03/release_lock.json)。

| 状态 | 结论 | 边界 |
| --- | --- | --- |
| WP03_CODE_GO | **GO** | A–D功能与最终177项选定CPU测试通过 |
| WP03_REAL_TRAIN_DATA_GO | **GO** | 本机真实FG201两日6账户原生G3、三方Join、Reader、Agent Loss均实际运行 |
| WP03_CURRICULUM_SUPPORT | **LIMITED** | 请求的合法分层窗口有支持；单合约两日不足以推广训练/泛化 |
| E_QUALITY_FORMAL_TRAIN_GO | **NO_GO** | 公开E+Quality/Prediction/Q/Decision Guard保持；无训练授权 |

只实施WP03，未新增训练器或Replay存储格式。未改真实旧Run、金融Env、Quality Reward、MACD、Oracle；未读TEST、启动GPU/正式训练、生成18合约或读取旧1M权重。

## Evidence observed

先只读核对原WP02成对分支的HEAD/status/worktrees及真实Corpus，读取WP01/02实际提交和报告、WP03任务书/接口Review/Master V1、根及双仓AGENTS。
原WP02 Gym `e57189bad100465c9fee8d9bbb5601f3e5c376ad`、Dreamer `44a6cc62032811501631e83be977f07137dc65b8`均clean；本轮从它们新建配对WP03树，未从落后integration开始。
路径：`/home/v/Documents/work/research/fg-s3-e-quality-wp03-20261016`，兄弟目录`gym-trading-env/`与`dreamerv3/`。
原Git命令、输出、worktree清单见[repository_baseline.json](receipts/wp03/repository_baseline.json)。WP01/02已是本分支祖先，**仍未合并integration**。

原Corpus真实存在于`/home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp02/real_small_04`；84发布文件、239个历史生成源码哈希最初均匹配。历史生成时Gym是`70a947d... + dirty=true`，Dreamer`44a6cc6... + dirty=false`；本轮保留该事实，不改成虚构clean来源。
本轮完整G3约28分14秒，6账户各691 Replay行/691老师核验、各1次完整账户恢复，exit0；不沿用旧WP02 PASS代替运行。
完整重播证据见[native_verification.json](receipts/wp03/native_verification.json)及[native_verify.log](receipts/wp03/native_verify.log)。

## Root cause

原WP02 verify把生成器整仓239份源码及绝对路径与当前代码字典直接等值比较，使开发换树/新增Reader也触发旧Corpus失配。
直接重签旧标签会破坏已有账户/老师来源证明，故本轮保持原Corpus字节，另以相对内容+课程/Teacher/Obs/source身份为新监督视图身份。
开发诊断检查数据正确性；最终来源证明调用冻结原生产器。当前开发HEAD/dirty不再是Reader门槛。

旧Catalog不能直接attach新L1/L2；物理stepid还不足以证明账户Owner；原全零Owner会形成假Pair；旧Q strict排序项不能仅靠valid=False关闭。
这些缺口由投影/Join/非零Owner/显式监督协议处理；没有放开训练Guard。
另按原Driver和真实P1行纠正任务书的action说明：本行action是下次执行提议，incoming_action/Reward是已有反馈；老师按本行post-step观测查询，不整体平移。

## Changes made

| 提交 | Gym完整SHA | 内容 | 本阶段实际测试 |
| --- | --- | --- | --- |
| A | `bb0c9a37774108834db63091686ff861f963c383` | SupervisionSource与分级Admission、原Corpus身份、路径/完整性负例 | A_final 14 passed，exit0 |
| B | `39b8348e7acb165fa6caaa5e74418effe3dc0ee5` | 原FIELDS投影，StudentReplay+Trace+Sidecar账户Join，非零稳定Owner，Q/Soft禁用 | B_final 22 passed，exit0；B_legacy_pairs16 passed，exit0 |
| C | `98bdb40d13c72112adce17774e7a474aa820418a` | episode-safe NATURAL/STRATIFIED/MIXED、惰性元数据索引、有界缓存、完整RNG恢复和coverage | C_final49 passed，exit0 |
| D | 本报告所在提交，完整SHA在release_lock.json | 原Agent CPU Loss-ready、真实G3/Reader、CLI/Guard负例、反向Review修复、最终报告 | D_release_final177 passed，exit0；真实3命令exit0 |

Dreamer仅B需要运行源码提交：`446d1bc99debd2d05c313eafd7d0c414ebd2815c`。其他阶段沿用，没有制造空提交凑数。

主要代码：

- `src/gym_trading_env/research/fg_s3_e_quality_supervision.py`：SupervisionSource/QualitySupervisionAdmission/project_label。development状态明确INTEGRITY_CHECKED_ONLY/NOT_ADMITTED；native-verification才执行真实G3。任何模式training/prepared/checkpoint认证均false。
- `src/gym_trading_env/research/fg_s3_e_quality_reader.py`：原NPZ/gzip流式Join；SQLite只保存元数据/投影/题型，Observation始终读原Replay；稳定episode Owner与逐行账户hash分工；下一执行校验、原reset/terminal边界。
- `src/gym_trading_env/research/fg_s3_e_quality_checks.py`、`tools/fg_s3_e_quality_reader.py`：隔离CPU诊断CLI和原Agent前向证据，无优化器/训练生命周期。
- `src/gym_trading_env/research/fg_s3_quality_catalog_v2.py`：只修复原生未成交strict平局时L1/L2错误耦合；其余认证及legacy函数不变。
- Dreamer `fg_s3_quality_v2.py`、`fg_s3_v2_agent.py`：新协议Q/Soft权重0/有效位拒绝、Owner全零不成Pair，旧无新协议配置默认保留。
- `tests/entry_v1/test_wp03_*.py`、`wp03_fixture.py`：38类命名行为的正负例；旧Quality测试仅补合法非零Owner fixture。
- `tools/wp03_verify.py`：准确命令/退出码/源码字节/日志取证。源码哈希只作本次报告，不参与普通开发准入。

### 开发与正式边界

开发可dirty/新增源码/换worktree/搬未改Corpus；测试C07禁止打开旧路径仍完成恢复。原Corpus和L1/L2身份不重签、不改路径。
普通read不重做Oracle或G3；最终真实数据结论绑定独立G3与同内容身份Reader。development不是课程draft，也不会因字段空/意外缺失自动draft。
WP01课程V2合法显式draft仍CONTRACT_ONLY/NOT_ADMITTED；formal/缺身份/非法路径/伪认证保持非零且无训练产物，已回归。

## Verification performed

最终选定集合：**177 passed / 0 failed / 0 errors / 0 skipped，exit=0**；pytest298.78秒，wrapper303.27秒。
包括WP03全套、WP01 native preflight、原Replay/Single3 Replay、WP02 MACD/Replay、legacy Pair/目标边界、原生严格老师伪造负例，及AGENTS指定共享Gate/SourceAdmission/formal Guards20项。
这是选定依赖集合通过，**不是全项目通过**。实际177个testcase在[executed_testcases.json](receipts/wp03/executed_testcases.json)的D_release_final；准确完整命令在[WP03_TEST_COMMANDS.md](WP03_TEST_COMMANDS.md)。

| 实际执行 | 结果/退出码 | 收据 |
| --- | --- | --- |
| 最终选定CPU回归 | 177通过，0 | [D_release_final](receipts/wp03/D_release_final.json) / [原日志](receipts/wp03/D_release_final.log) |
| 独立共享回归 | 20通过，0；最终集合再次含该20项 | [D_shared](receipts/wp03/D_shared.json) |
| 真实原Corpus verify/G3 | 六账户PASS/G3 GO，0，1694.46秒 | [D_native_G3](receipts/wp03/D_native_G3.json) |
| 最终真实Reader+模型CPU | 10视图、6policy loss、精确恢复，0，65.05秒 | [D_real_reader_final](receipts/wp03/D_real_reader_final.json) |
| 重新统计原生coverage | 与WP02原发布统计一致，0，3.81秒 | [D_actual_coverage](receipts/wp03/D_actual_coverage.json) |

负例覆盖：TEST/DEV/穿越/symlink读前拒绝；更改源/课程/MACD字节；重签错账户/stepid/ordinal/老师；缺失/重复/shift；同价不同账户不能Pair；无双合法/forced WAIT/资金域外/删失无strict标签；零Owner不Pair；缺类不造样；恢复时source/index/stage/spec变更拒绝；失效标签梯度0；Q/Soft暗开拒绝；teacher改动不影响Actor logits/Reward；prepare/start/resume/evaluate仍退出1并报S3_QUALITY_E_NOT_IMPLEMENTED，无新训练输出。
CLI开发正例exit0但只写非认证诊断；空purpose/非法路径/输出目录负例exit2且无产物。38项完整映射见[验收矩阵](WP03_ACCEPTANCE_MATRIX.md)。

真实计数：4146物理行，690unique market cursors、4134已执行账户决策，投影L1 Outcome1543/Score1430，strict ADD171/WAIT1279，资金域外93仍invalid；真实成交119=103TP+12SL+4MAIN_ROLL，右删失0。
原L1/L2与投影各差8源于有证明的prefix/reset/未执行位置失效。全episode strict46743/quality89729与实际模型batch strict295/quality256分开。
所有6个账户完整equity_history62519行、终态MAIN_ROLL、无未平仓；见[真实覆盖报告](WP03_REAL_TRAIN_COVERAGE.md)。

真实CPU模型fresh E+partial5m MACD+Quality，原training_loss_fn/quality_loss，B1,T64，loss=0.48144430、WM loss=602.44445801；
Actor既有纯数值目标=0.79798061，只验证数学/梯度，不作为CONTROLLED_FULL阶段准入。
有效Head梯度非零、Qc1/Qv2梯度严格0，全部无效时loss与梯度0；原参数/优化器字节不变、n_updates=0；第二fresh Agent内存cold restore一致。
[CPU证据](WP03_CPU_FORWARD_LOSS_EVIDENCE.json)保留错位/Q/Soft负例的真实异常；日志中的jax.debug.callback failed是预期拒绝证据，命令最终exit0。

性能：Reader实测558.95行/s、峰值RSS330432512字节；含模型进程2130935808字节；原chunk cache峰值19500711/预算67108864字节；索引10002432字节，原Corpus168506837字节；不把这些当GPU训练吞吐。
G3按实际可用内存/24逻辑核选择6账户workers+11标签workers，合计17进程、worker原生线程1，资源budget原值在G3收据中。

### 身份锁（内容哈希，非认证签名）

| 合同 | 哈希 |
| --- | --- |
| Corpus | `640865b4239fb483e5d98daf18d3104e670ff9b3c1486cb63ec9924cf6e089a9` |
| Supervision内容 | `85405da25efa3c850fb1b14ec821c53987e61ead7f233073d2b7dd85a07e9081` |
| Source lock | `ba359d42ace3439aab97c848dc7781cac635e2612b8aeb24525fce10099a017c` |
| Course V2 | `14cadcdaaa7d96bc0f33f491ef38741494957ee8e7423a1ba5bf598fe230baba` |
| Strict Teacher | `092d822e5975fdece9f181bb7bf27447a0d2e1923aa93d119b01ef869e56bad6` |
| Observation | `5ca333e52049bf5bdfe0bf18f5519e65766363c946648bdd1f45b525558052b8` |
| Model V2 | `7152f9de6de4885b0e64c77f003c7a646a9807364c4019034ff73aa2d76cad1b` |
| Join index | `c145e57ca1f1d17df2bcb48ae286d5fe807d6122ad86de3807207ebda482f1e1` |

### 自审与失败记录

独立反向阅读损失→投影→三方Join→G3及政策→Observation，检查源码保护、时间、Owner、无效梯度、精确恢复和实际文件副作用。
发现并修复诊断Ninjax上下文设置、历史producer只比HEAD的漏洞、原生未成交L1/L2平局耦合；原始失败D_cpu_targeted、D_review_producer_before、D_final_review、D_native_tie_fix均保留exit1。
平价测试曾误判tie，实测原Reward后纠正fixture，不改Reward。修复与证据详见[WP03_SOURCE_REVIEW.md](WP03_SOURCE_REVIEW.md)。最终177项及真实Reader是在这些修复后的字节上重跑。
最终源码与实际被测字节一致，Gym216个/Dreamer86个既有运行源码未变，受控改动函数之外AST与原FIELDS不变。
`git diff --cached --check`完整集合exit2：原任务书Markdown换行空格及原始失败日志/XML尾随空白；这些原文保持不变。限定本轮代码/测试/编写文档的check exit0；准确argv/输出见[diff check](receipts/wp03/D_diff_check.json)。最终clean状态另在release锁记录。

## Risks / not verified

- **覆盖LIMITED**：一个合约主力资格末两日，strict WAIT多，真实删失0，不支持学生已学会/训练容量/政策效果结论。
- 最终native wrapper加历史producer dirty前后保护后，未重复整个28分钟G3；原生生产器及Corpus字节未变，原G3原始结果保留，新增保护有失败→通过测试与实际producer audit。不是把旧WP02报告继承为本轮PASS。
- SQLite及自哈希JSON只承担开发完整性，不防恶意重签成为正式缓存；实际验收新建索引。正式Prepared/Checkpoint/训练来源认证必须后续单独冻结/重建。
- CPU模型为现有small_config+e_target_config的fresh结构烟测；完整training_loss_fn处于HEADS_ONLY，Actor纯数学梯度不是F3阶段实现。正式参数矩阵/优化器更新/冻结与handoff尚未开发。
- **NOT_RUN**：GPU/正式训练/优化器更新、WP04–07正式阶段生命周期、正式Prepared或Checkpoint认证、TEST行情、18合约全量、旧1M权重、全项目测试、W&B上传/发布/合并integration。
- 原始真实Corpus只在本机artifacts，Git只交付代码与审查证据。搬迁可复制未改Corpus，普通Reader不读取原绝对根；旧G3重验仍要求原冻结来源环境。

本包要求的有限真实来源与CPU Reader验收均已运行并通过；上述NOT_RUN是明确留给后续工作包或未授权范围，不能解释为正式训练GO。

## Next recommended step

按[Master Roadmap V2](FG_S3_E_Quality_Master_Roadmap_20261016_V2.md)进入WP04 F1设计/评审，复用本Reader与既有原生模型，先证明WM-only参数更新、冻结和同版本恢复，再独立申请正式训练准入。保留所有旧Run及本次原始工件。
