# WP03 源码反向 Review

## Summary

**WP03_CODE_GO=GO，有限REAL_TRAIN_DATA_GO=GO，CURRICULUM_SUPPORT=LIMITED，FORMAL_TRAIN_GO=NO_GO。**
这是实施后单独进行的源码/证据反向审查，由同一实施代理完成，不声称外部独立审稿人批准。
审查对象为最终被177项回归及真实Reader验证的运行字节；完整SHA在Commit D后的`receipts/wp03/release_lock.json`。

## Evidence observed

从损失倒查：`_S3V2Model.training_loss_fn` → `quality_loss` → 原`FIELDS` → `project_label` → SQLite Join元数据 → 原StudentReplay/Trace/Sidecar → 本轮原生G3。
从政策倒查：Actor eval/Encoder只接原obs_space；未来Outcome/strict/witness留在监督视图。Teacher变更前后actual latent/logits/原WM目标一致。
从来源倒查：84文件均保持原发布字节；开发身份不依赖当前Git或绝对路径。原G3重新执行并认证691×6行及全账户恢复。
从恢复倒查：Reader state包含RNG、source/index、stage/spec、曝光计数；实际下一批逐字段一致；fresh第二Agent从本轮内存snapshot恢复政策输出，n_updates保持0。

## Root cause

原整仓源码等值检查把Reader开发和生成器迁移混合；旧Catalog与新L1/L2结构不能直接attach；全零Owner还会被旧pair_identity当成相同账户。
旧Q-C1损失中除value valid外还有strict排序项，所以只关闭valid位不够；本轮显式新协议同时令Q/Soft权重为0并在真实Agent入口拒绝非法有效位。
原Driver的action是本行观测下下一次将执行的提议，incoming_action才是本行反馈；按旧审计文字平移teacher会错位。真实行与源代码已经纠正这点。

## Changes made

- 数据身份仅绑定原相对内容与合同；普通dirty/换树不触发重新G3。开发报告绝不自升为SOURCE_VERIFIED、Prepared或Checkpoint认证。
- 原始物理Replay字节不改。SQLite只有Join/投影/分层索引，不复制observations，不写新的Replay。完整物理UUID+offset、policy、ordinal、账户hash、mask、cursor/date同时核对。
- 同episode Owner稳定，账户hash逐步核验；不同账户/原生reset隔离。prefix/terminal/admin行无效，下一推进证明decision_executed。
- 原Quality Score/窗口valid独立映射；严格动作只取L2。Q/Soft数值不冒名新后缀，保留只读witness证据。
- NATURAL按合法窗口长度抽样，STRATIFIED/MIXED使用显式权重；缺类报错且不推进RNG/曝光。缓存有界，状态可精确恢复。
- 真实后端HEADS_ONLY CPU init/forward/loss；Actor BC/Pair仅在现有纯数学目标上检查梯度，没有送入CONTROLLED_FULL阶段工厂或请求准入。

### 反向审查发现并修复

| 观察到的失败 | 确认根因 | 最小修复与实际验证 |
| --- | --- | --- |
| `D_cpu_targeted`:10失败/1通过，exit1 | 诊断nj.pure使用modify=False，而实际RSSM的Ninjax.scan需要可修改的私有功能上下文 | 仅诊断改modify=True且丢弃返回状态；实际全部参数/优化器字节前后相同，D_cpu_retry11通过和最终177通过 |
| `D_review_producer_before`:1失败，exit1 | 历史producer只比HEAD，不能发现实际执行的dirty verifier | 只在显式native-verification核对冻结producer实际运行字节；当前开发树不受此限制。拒绝测试先失败后通过，原producer实际audit也PASS |
| `D_final_review`:30通过/1失败，exit1 | 新测试误以为平价必定平局；原生ADD仍承受MAIN_ROLL反馈−0.03 | 不改金融规则，受控volume=1使原合法ADD下一bar确实未成交；严格老师实际证明双动作等值WAIT |
| `D_native_tie_fix`:1失败，exit1 | 上述真实未成交时L1无Outcome，但L2有效；Catalog错误地把L1 invalid_reason一并当成L2无效 | 仅REJECTED+L1无Outcome/无Score+L2完整strict平局WAIT例外；双合法/账户/后缀/有理数witness仍必需。B14完整原Env生成+reconcile通过；假witness/重签伪造负例仍拒绝；最终177通过 |

原始失败日志/XML/命令全部保留，未skip/xfail或改Guard消除失败。这个Catalog修复只涉及WP02新Corpus envelope，原legacy validate_label/attach_quality/soft_teacher不变。

### 原生代码保护与权限

对固定WP02端点比较：Gym 216个既有运行Python文件、Dreamer 86个保持原字节；已有运行改动仅Catalog、Dreamer quality_extensions/pair_identity和新协议batch拒绝检查。
Catalog FIELDS和其余旧函数AST、Dreamer未改动数学/政策函数AST保持一致，见[source_protection](receipts/wp03/source_protection.json)。
金融Env、Reward、MACD、Clock、Native/Strict Teacher、Oracle、原Replay/Driver、FullCheckpoint、正式SourceAdmission与composition require_ready均保持字节。
源码字节审计是本次Review的范围证明，不是以后每次开发的准入门槛。

## Verification performed

最终177 passed / 0 failed / 0 errors / 0 skipped，exit0；包含AGENTS固定共享20项和直接受影响Replay/MACD/恢复合同。
独立G3六账户、实际Reader和原模型Loss、实际coverage重新统计均exit0。所有实际命令见[命令表](WP03_TEST_COMMANDS.md)，38项映射见[验收矩阵](WP03_ACCEPTANCE_MATRIX.md)。
最终运行源码与被测源码逐文件相同，见[tested_source_equality](receipts/wp03/tested_source_equality.json)。

## Risks / not verified

- `read`命令是开发诊断；真实最终数据结论将独立native G3和同内容身份Reader证据合并评估，不伪称每次read均重做G3。
- native G3实际完成后，才为历史producer加了前后dirty保护。未再重复28分钟的全重放；原生验证器/生成字节及Corpus均未变，保存了实际producer audit与针对新保护的负例。原G3收据保持原样，未补写不存在的producer_audit字段。
- 开发SQLite及自哈希JSON提供完整性，不是防恶意重签的正式索引认证。实际验收从原Corpus新建索引；正式SourceAdmission/Prepared/Checkpoint须在WP04–07独立冻结或重建认证。
- 单合约两日、93个资金域外无效、真实右删失样本0；没有研究策略效果验收。当前小模型只是CPU结构/损失烟测，不证明正式网络容量或训练吞吐。
- WP04–07阶段工厂、优化器冻结/更新/handoff与正式来源授权均未实现/NOT_RUN。公开Guard继续S3_QUALITY_E_NOT_IMPLEMENTED。
- NOT_RUN：GPU、正式训练、旧1M读取、TEST、18合约生产、全项目测试、W&B/发布/合并。

## Next recommended step

WP04先设计F1原生WM-only阶段准入、真实参数变化与同版本恢复；采用本Reader，不把CPU纯数值目标或开发缓存当成正式训练授权。
