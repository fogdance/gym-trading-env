# S3 E-Model V1 实施与 CPU 验收报告

## Summary

完成 M0–M3 的 E Encoder、版本化观察/DAgger/native Replay、Decoder目标和受控fresh-init认证开发，等待Review。状态：`E_MODEL_CODE_READY_WAITING_E_DATA_AND_GPU_REVIEW`。

三E组合仍waiting_e，Tutor+B2仍NOT_IMPLEMENTED；正式E Prepared/Run/GPU/DEV交易/W&B online未执行。CPU模型、Source和Checkpoint证据不冒充生产验收。

## Evidence observed

起点为本机干净 integration/s3-next：Gym `cc6ba78f3a2016eb280135f7730d0a1f68d4512e`，Dreamer `eb63de710fd693fad7ac76f797f3d9f39e331b98`。双仓 `feat/s3-e-model` 位于 `fg-s3-e-model-dev/`。任务书完整读取并原文保存至docs，计划与接口均已落盘。

运行代码提交：Gym `da7e59fb228126102e17e264db3751656fa2f500`，Dreamer `f30d8da2523eddad53aeca1540afe1c77d47b248`。冻结验收在两仓干净时执行。最终文档SHA和Diff由工作区外置 `baseline-lock.json` 记录，避免SHA自引用。

开发期间外部将Gym integration推进到 `c3a808437ef277ee87f25b0ae8803e31fa5dca74`，增加online console和Tutor对照文档；本功能保留已锁定起点，没有合并/rebase或修改integration。保护清单区分这次外部变化和本任务写入，未宣称integration完全不变。

资源实测可用内存约12.2GB，项目磁盘/dev/sdd2可用27GB；测试小模型峰值约1.4GB，无swap。仅使用/dev/sdd和/dev/sda所在目录。原运行进程只读观察，未控制/停止；本任务未占用CUDA。

## Root cause / implementation gaps

原代码仅注册simple Encoder；模型工厂固定旧FIELDS，DAgger严格旧字段集合，Decoder自动重建所有非Mask观察。只增加网络类或YAML不能组成E链路。原B2 SourceAdmission仅允许warm-start，不能认证新E完整初始化。这些缺口由实际源码确认，未归因于缓存或历史实验。

## Changes made

- 新双尺度Encoder：1m三层简单MLP +15m Flatten/Linear128/GELU +Concat/Linear；默认size50m的D512，常规随机初始化。保留原self.enc入口及enc参数根。
- E专用schema/白名单。保留旧行情审计，但只编码shared1m/15m/position/Mask，只重建shared1m/position。旧FIELDS/S3_FIELDS、RSSM/Reward/Oracle/市场分钟及教学公式未改。
- 在原Env/Driver/Collector/DAgger/native Replay扩展schema/provider，每步保存E历史窗口、独立provenance和实际执行动作；未来标签不进入Actor。provider必须匹配当前真实合约/决策时刻。
- 保留公开require_ready，抽取共享内部模型/认证核心。新E只允许可信seed/config/schema绑定的direct_init，B2真实head/adapter/schedule齐备；旧B2 warm-start约束不放宽。
- Source/Prepared/model/Run/evaluation绑定检查及原FullCheckpoint覆盖新参数、Opt、Replay、Env/Driver/RNG/调度。CPU Prepared使用独立契约协议；真实E绑定与consumer留待M4/Review。
- 新增三份明确WAITING的目标配置；不改变公共网络presets。模型接口见 [S3_E_MODEL_V1_INTERFACE.md](S3_E_MODEL_V1_INTERFACE.md)。

## Verification performed

提交后冻结验收共58个不同用例：50项配置/来源/观察/Encoder、5项实际神经模型、3项完整Checkpoint恢复，0FAIL/0SKIP。主命令使用原Dreamer Python，显式CUDA_VISIBLE_DEVICES为空、JAX_PLATFORMS=cpu、float32小模型，T64+prefix1。完整argv/env/SHA/dirty及stdout/stderr/JUnit见 [e_model_v1/evidence](e_model_v1/evidence)。

三模式各5次实际联合更新，保留原1000-update optimizer预热；1m、15m、Fusion、Actor参数均变化，Critic冻结。B2测试调度warmup2/ramp2/min-events2，真实跨越调度边界，不注入更新计数。WM三分支梯度有限非零；A2 BC对Encoder梯度严格0。Tutor/B2复用原函数体，AST对照相同。

同一RNG/初始state下改变受控合成15m输入，Actor p_add从约0.5011214变为0.4993399。RSSM当前deter由先前state/action产生，当前观察影响posterior/logit/stoch；测试按实际接口验证，没有修改动力学来满足错误的deter断言。这不是盈利或真实E行情有效性的证据。

三模式各在2次更新/70环境步保存完整Checkpoint，恢复后环境到75步并执行另外3次更新。模型/Opt、policy参数、native Replay下一样本、Driver/Env、Sidecar、Gate、RNG、计数和后续指标逐值一致。错teaching或data identity拒绝。完整包路径与SHA见 [checkpoint_index.json](e_model_v1/evidence/checkpoint_index.json)。

额外回归：26项原模型/FullCheckpoint/context、60项配置/DAgger/市场分钟均通过，另有本地Tracker组合测试通过。不同组有重叠，不汇总成独立用例总数。真实旧来源B2 CPU prepare为PREPARED_NO_TRAINING，认证源模型更新21183、新环境步0；未初始化GPU模型或创建Run。

所有失败保留：

- 初始pytest basetemp父目录缺失导致18项setup错误；修正验证编排目录，不改业务约束。
- 100分钟fixture越出原FG特征时段，被Env拒绝；改用既有90分钟fixture，未改市场时钟。
- 初次模型测试的A2/Tutor传入B2额外监督字段、Ninjax梯度目标包含未访问的Actor、错误比较当前deter；修正测试字段/目标/断言，未放宽模型或改变RSSM。
- 两项旧D0测试使用了不允许的artifact子目录；按原规则在/tmp重跑，60项通过，未扩大目录许可。
- 自审provider合约/时间校验缺口，两项回归修复前FAIL、修复后PASS，并覆盖原生Replay路径。
- 旧SourceTransfer三项fixture缺少显式S3_MARKET_MINUTE，在本分支和未改动integration对照均为2PASS/3FAIL。保留BASELINE_FAIL_3，不放宽认证、不修改无关fixture。
- 环境编排的python命令/PYTHONPATH错误已修正，记录在setup_failures；未写入正式工件。

自审检查了双仓完整Diff、未变的教学方法AST、参数根/同步/恢复、标签隔离、dtype/schema及初始化身份；代码diff whitespace检查通过。
原始pytest日志/JUnit中的行尾空格按原字节保留，evidence路径单独标记为生成记录，不格式化测试输出。

## Risks / not verified

EM14仅CPU来源与签名合同通过，正式E Prepared仍阻断；EM16保留三项已复现的基线失败。EM17真实E端到端、EM18 GPU/DEV Actor-only/W&B online未运行。默认size50m配置已解析，但神经更新证据来自D16/context128的CPU小模型，不能宣称全尺寸训练性能。

独立E-Data已报告21项F01–F11通过、18个TRAIN合约技术就绪；其Review仍pending，也未合入本模型基线。当前fresh-init binding只接受CPU synthetic清单，不能用于真实E正式启动。联调consumer、真实来源绑定与全生命周期验收见 [阻断清单](e_model_v1/E_DATA_BLOCKERS.md)。

## Next recommended step

Review本次双仓Diff、[EM01–EM18状态表](e_model_v1/EM_STATUS.md)、来源认证与Checkpoint收据。确认后单独安排M4真实E-Data联调；M5公开解锁及GPU/正式实验仍需要各自Review/授权。本任务到此暂停。
