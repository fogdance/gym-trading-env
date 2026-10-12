# FG S3 统一 Gate 0.30 修复、来源迁移与 S3-A 准备

日期：2026-10-12。执行范围：`research/fg-s3-dev/{gym-trading-env,dreamerv3}`；集成分支均为 `feat/fg-s3-shared-s3c1`。依据统一 Gate 任务书 V1 及 `FG_S3_Shared_S3C_gy7_d37_Code_Review_20261012.md`。未合并三个算法分支，未机械合并 Replay 分支，未修改两个其他实验仓库或 `.gitignore`。

## 1. 已修复的问题

| Review 项 | 根因及修改 | 验证与边界 |
| --- | --- | --- |
| P0-1：Gate / Imagination 阈值分叉 | 原 `Agent.loss` 固定 `logit>=0`，Imagination 使用配置阈值。现共用 `availability_prediction`，先转 float32；0.50 保留原 logit 零边界，0.30 比较 sigmoid 概率。WAIT 保底仅作用于下游 mask。 | 真实 Online180k B16×65 CPU `Agent.loss`：loss=18.228124618530273，45 项原指标最大差均为0。BF16/float32 边界与真实 availability head 联动测试通过。该零差异结论限定于实际测试输入。 |
| P0-2：direct_init 误用5窗 | 新公共 `fg_s3_gate.py` 为正式协议调用 `ActionMaskWarmup.for_single3_1m`，源/目标均绑定其开启协议；direct_init 也生成完整 policy。 | 实际小网络 Runner：19窗不开、失败清零、重新20窗才开；开启 Prior Accuracy≥0.9915、FPR≤0.004；原双 Recall≥80%、ADD support≥16、25000样本和连续5次回归停训保留。受控指标窗口是控制器验收，非生产 Gate 已达标证明。 |
| P1：180k / S3-A / 0.30 来源混淆 | 采用已审计的**原始0.50 Online180k冻结副本**，经明确的 V2 新实验迁移变为0.30。原节点和旧A不改写。Actor、WM、Critic、joint optimizer、norm/slow value、原累计计数保留；跨阈值清空 Gate window/streak/READY。 | 原完整286项状态实际CPU迁移；旧A1000实际Actor继承及原joint optimizer逐键相等通过。另一个已发布0.30节点的 `agent.pkl` 与原始源逐字节一致，但其 producer 未在本集成分支独立准入，因此**不选为来源**。 |
| AST 占位符及过宽 factory 归一化 | 固定两份经审查 helper AST SHA；仅归一化获准的 Gate 语句、默认 `return self.loss` factory 及其确切调用。 | helper边界改动、任意原loss改动、factory改为report均拒绝；原 producer→当前实现的归一化 AST 与双端文件SHA写入 receipt。未关闭校验。 |
| Resume 来源READY / 新模型CLOSED不一致 | 恢复前先构造空控制器，再由本 Run FullCheckpoint 同时恢复模型与控制器，随后核对两侧 Gate。 | direct/after_bc真实模型恢复通过；同Run不再读取、准入或迁移原来源，配置/来源改动拒绝。 |
| Resume 日志尾部重复 | 原更新日志可能领先最后完整checkpoint；恢复后重复append。 | 原完整日志先归档到 `recovery_logs/`，再恢复合法前缀；真实神经网络/Replay/Prefetch/Consec/carry/optimizer/Gate恢复和日志序列共同验证。 |
| Formal 误收 diagnostic 来源 | 来源和Catalog的diagnostic标志合并；A/B和C入口均拒绝formal消费诊断来源。 | 真实CPU入口的负例通过。 |
| 新BC中途节点缺少0.30身份 | BC 的 `source_identity_hash` 改为共享 SourceAdmission 生成的已准入目标模型身份，绑定目标Agent配置、Gate协议与转换、源参数摘要和代码。 | 新A/B不改原BC算法或checkpoint writer。旧0.50 cache按原源身份核验，新生成cache按目标身份核验；缺少绑定的managed snapshot时新BC不能被推断为旧0.50来源。managed A→C、B入口及旧A来源分别验证。 |

共享 `SourceAdmission` 协议为 `FG_S3_SOURCE_ADMISSION_V2`；C新Run协议为 `FG_SINGLE3_S3C1_CONCURRENT_BC_V4`。历史原算法文件、标签、缓存和BC工件保持原样。已有C Run不能跨源码/协议精确resume；需要新准备和新运行身份。

## 2. 共同来源与缓存

共同原节点：

```text
/data/logdir/fg_single3/online_after_10k_callback_fix/C1_M1_ONLINE_1M/ckpt/step_000000180000
```

实际使用其已有只读冻结副本：

```text
research/fg-s3-dev/gym-trading-env/artifacts/fg_s3_arch_v2/online_180k_source/ckpt/step_000000180000
```

- 源 manifest SHA：`4979acfb0272ffbc3c96833e5b77fccd66c5d4d0f335bfca48810f764d927a57`。
- 源 Agent SHA：`2cc450b328484ddfa489b3c54ef10cbf98395678dbc6ae957fd1a1fe3777b1c3`。
- 来源模型更新21183、batches21399、环境动作180000；180k节点名表示环境步，不手填模型计数。
- 原0.50 Gate CLOSED，reports112，窗口samples51200；新0.30 reports/streak/samples均0，READY=false，原窗口丢弃仅发生于**新Run transfer**。同运行resume保留完整Gate/窗口。
- TeacherCatalog SHA：`47736742ff81e337d10585b86d0709f5fd1bff8d082b01324b267f9f1761f0de`；完整P4 488629行、有效402096、ADD16171、WAIT385925。
- 复用旧冻结训练目录的404片、402096×5120 float32特征，manifest SHA `4f4af313cb53fae18bec4c4f6301c52e65c776943583d031856aeca3b1218ced`。缓存以旧producer身份和全部文件hash认证；加载后的实际全参数摘要仍必须相等。未重新生成或覆盖8.2GB缓存。
- 数据入口 `research/fg-data` 是此前用户授权的数据软链接；只读复用TRAIN/Replay/Oracle，不作为外部代码或Python导入来源。

SourceReceipt明确区分原始来源和0.30目标身份，记录两端阈值、protocol、reset原因、Gate前后状态、代码指纹和模型摘要。原始源不是伪装成0.30 checkpoint。

## 3. 实际验证与收据

收据目录：`artifacts/fg_s3c1_v1/receipts/gate_t030_v1/`。

| 项目 | 真实结果 |
| --- | --- |
| 原0.50 B16×65 loss和45指标 | PASS：`baseline_050/analysis.json`，CPU，无来源梯度更新 |
| Gate / 边界 / AST负例 / READY跨阈值 / 同阈值窗口继承 | PASS：纳入 `final_neural_cpu.xml` |
| 0.30 lambda=0三种来源、BC仅Actor梯度、真实Mask、原RL/WM、单联合optimizer | PASS：真实小网络CPU，`final_neural_cpu.xml`，不作为正式生产训练验收 |
| FullCheckpoint、原Replay时序、teacher、carry、Gate、日志精确恢复 | PASS：真实小网络CPU，`final_neural_cpu.xml` |
| A/B/C准备、原A/B训练器、固定预算、恢复、FD生命周期 | PASS：`final_manager_cpu.xml`，10用例 |
| 最后目标身份绑定 / 中途包拒绝 / A→C / B入口 / FD复测 | PASS：`target_identity_cpu.xml`，3用例；不是额外3个独立覆盖项 |
| 真实180k C模型一联合更新 | PASS：`real_cpu/one_joint_update.json`；B1×65、52有效WAIT、ADD=0、Gate CLOSED、BC effective=0、WM变化、Actor/Critic梯度0、原opt只增1。**不能证明生产ADD监督或READY联合训练通过。** |
| 真实旧A1000→0.30 C模型 | PASS：`real_cpu/after_bc_1000_transfer.json`；11个原Actor键继承变化、原WM/Critic/joint opt/norm不变，不迁入独立BC opt |
| 真实S3-A CPU Actor两步 / 原缓存 / 保存恢复 | PASS：`actor_cpu/summary.json`；B256，每批ADD分别13和10，仅11个 `pol/*` 键改变，原joint计数/optimizer不变，参数/独立opt/sampler精确恢复 |
| 当前执行环境GPU smoke | FAIL（环境设备不可见）：`actor_gpu.stdout.log`；JAX CUDA backend初始化前退出，此沙箱内GPU模型/梯度未执行；主机单独通过见下一行 |
| 主机GPU smoke | PASS：主机实际 `cuda:0` 两步BC与参数/optimizer/sampler保存恢复；`actor_gpu_host/summary.json` 及独立CPU `host_gpu_checkpoint_audit.json` 验证实际节点 |
| 正式A/B/C、DEV/TEST、C生产全量Replay资源/收敛 | NOT_RUN；A只在全部必要准入完成后交给用户d3等待命令 |

模型套件15用例与管理套件10用例通过。最终身份元数据扩展后针对受影响的3入口再测，最后C after_bc恢复与AST负例另存 `final_contract_cpu.xml`。初始失败XML完整保留：`gate.xml`（收据变量覆盖错误和3处测试夹具问题）、`shared_cpu.xml`（此前已加载的收据变量覆盖错误）、`manager_cpu.xml`（在测试期间修改了被绑定的源码，正确拒绝code drift）。修正后PASS，未覆盖失败证据。

CPU命令使用：

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/home/v/Documents/work/research/fg-s3-dev/dreamerv3:/home/v/Documents/work/research/fg-s3-dev/gym-trading-env/src \
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest \
  embodied/tests/test_fg_s3_gate.py embodied/tests/test_fg_s3c_agent.py embodied/tests/test_fg_s3c_resume.py -q
```

该命令在Dreamer worktree执行。Gym管理套件是 `tests/entry_v1/test_fg_s3_experiment.py tests/entry_v1/test_fg_s3c_experiment.py tests/entry_v1/test_fg_s3_feature_lifecycle.py`。完整执行命令、探针文本及各receipt哈希由收据目录保存；`.source.txt` 是审计记录，不是新的训练入口。训练入口仍只有 `tools/fg_s3_experiment.py prepare/start/resume`。

## 4. 旧S3-A归档与新运行

旧Run `/data/logdir/fg_single3/s3/S3A_ORACLE_BC_FROM_ONLINE_180K_seed0`、其所有checkpoint、stdout、metrics和原冻结worktree均保留。小型只读归档索引/receipt/1000批metrics复制到 `old_s3a_archive/`，未复制或删除大型模型与缓存。

旧A实际完成1000 BC；全TRAIN ADD recall=0、预测ADD频率=0、Always-WAIT accuracy≈0.959783；1000/1000批含ADD。训练步数和每批ADD覆盖不能证明ADD分类有效，也不说明Prior Gate被A改进。

新配置：

```text
S3A_ORACLE_BC_T030_FROM_ONLINE_180K
S3B_CERTIFIED_T030_FROM_ONLINE_180K
S3C1_T030_DIRECT_INIT / DIRECT_FULL / AFTER_BC_S3A
```

全部配置0.30，旧配置0.50保持历史用途。A/B预算1000 Actor梯度更新、batch256、seed0；各有独立Run/prepared路径。C不使用特征缓存，使用当前RSSM repfeat；其中after_bc指向新A1000及同一冻结原基线，节点未产生前严格拒绝准备。B/C正式训练不启动。

新A Run：`/data/logdir/fg_single3/s3/S3A_ORACLE_BC_T030_FROM_ONLINE_180K_seed0_20261012`。共用一对冻结worktree：`research/fg-s3-train-t030/{gym-trading-env,dreamerv3}`。双仓冻结SHA、实际prepare收据及状态在本报告末尾的运行交接记录中填写。

## 5. 权限与今晚/明日边界

本执行环境普通tmux访问 `Operation not permitted`；提前申请主机级访问被策略 `sandbox_approval=false / rules=false` 自动拒绝，不存在已批准的tmux访问。普通GPU只读 `nvidia-smi` 显示主机16GB4060Ti及仅图形进程，但Python检查 `/dev/nvidia0`、`/dev/nvidiactl` 不存在，`has_visible_nvidia_gpu=False`，CUDA初始化失败。因此不能把主机GPU或d3已启动状态从CPU测试推断出来。

用户已在d3运行等待 `/tmp/done` 后启动新版A的命令。此文件是**启动授权信号**，只在冻结、CPU prepare和主机GPU smoke实际通过后创建。访问限制未解除而主机probe未提供时，保持不创建，A状态为WAITING_GPU_SMOKE。

今晚必要项：共享问题修复、Source身份、CPU模型/Actor恢复、冻结和CPU prepare、文档和receipt、成功GPU准入后释放信号。明日项：C生产含ADD批的进一步联合更新、完整Replay内存/耗时与GPU资源预检、生产Gate真实连续20窗、正式A完整分类Review。DEV/TEST未授权，不执行；B/C正式训练未授权，不执行。实际新A指标不足时不得更改Loss或采样来伪装成功。
