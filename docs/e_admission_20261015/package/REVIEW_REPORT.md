# S3 方案 E — E-Data × E-Model 双仓合成审查（2026-10-15）

**本次结论：NO-GO，尚不能启动正式 1M。** 本文记录了在**隔离归档副本**中做的代码级合成及其测试；不代表已更新用户 `/home/v/Documents/work/research/` 的实时 Git 仓库，也没有修改正在训练的 Run、启动 GPU 或读取 DEV/TEST。

## 1. 仓库与合成目标

- Gym Data 源：`feat/s3-e-data-v1`，提交 `720a66a`；Model 侧 Gym 基线：`feat/s3-e-model`，提交 `05141ee`。本次在隔离副本里从 Model 基线建立合成分支，合并 Data 历史（**merge，无冲突**），追加 E runtime provider 和来源输入接线。旧双方提交历史保留；不回写用户现场 `feat/s3-e-model`。
- Dreamer Model 源：`feat/s3-e-model`，提交 `f30d8da`。Dreamer 不存在对应的 `s3-e-data-v1` 分支，因为 Data 是**独立的 Gym 仓库**；Dreamer 单独追加 real TRAIN collection 认证和 E runtime provider 的条件接线。
- 提供分别面向 Gym、Dreamer 的两个 `git apply` patch。两仓全部基线、合成 Git SHA 和补丁 SHA 见包外清单；必须在实际机器上重新审查 dirty diff 和 Head，不能静默 `reset`/`merge` 原训练 Worktree。

## 2. 原设计与当前实现逐项审查

| 合同 | E Data | E Model | 本次判断/措施 |
|---|---|---|---|
| 同合约 OHLCVOI、同锚点 | `market_seq_e_shared`、`market_context_15m`、真实 TRAIN 数学核验 | E Encoder 使用 shared1m+15m | **设计一致**；不改已有数学 |
| 15m 最后一个完整可见桶 | `MarketData.snapshot` 用 cutoff 和 available_at | Encoder 含 [24,7] mask | **设计一致**；坚持每个决策现场计算 |
| 旧18维仅审计 | Data 保留 `market_seq` 不改 | Decoder 仅重建 shared1m/position | **一致**；不增加旧18维 Decoder |
| 原单仓 TP5/SL30/Reward | 不触碰 | 不触碰 | **一致**；E 专注新市场输入 |
| 三种组合 A2/Tutor/B2 | 原数据模块不带 teacher | E 模型的三套合成CPU梯度/恢复既有证据 | **部分验证**；尚无真实 TRAIN 端到端 |
| E 数据进入每个 Env 决策 | 原研究导出仅每15分钟取样（32,564决策） | `provider.snapshot(env, obs)` 协议已定义，但原无实际 consumer | **P0缺口**：本次新增复用原 `MarketData` 的 `ETrainProvider.snapshot`，不能读取稀疏导出替代分钟历史 |
| 真实 E 来源签名 | `index_manifest.json` 用 `index_sha256` 与 collection schema，不使用 Replay signed 的 `content_hash` | 原 `make_binding` 只接受 CPU SYNTHETIC | **P0缺口**：新增 real collection 全量 verify + schema/pool/code hash，且只授权 M4 CPU 联调，不解锁正式入口 |
| 实际 Collector/Replay 绑定 | 原 observation wrapper 支持 provider | `LiveRun` 原未传 `observation_schema/e_provider` | **P0缺口**：条件接线到既有 V2Collector，legacy 默认路径不变 |
| SourceAdmission / exact resume | 生产数据集身份不变 | E public `require_ready()` 仍拒绝 waiting_e | **NO-GO**：尚未完成对真实来源的 Prepared/Run/FullCheckpoint 回归与正式准入审计 |
| GPU size50m | CPU E 模型小网络既有验收 | 未提供真实 E GPU smoke/显存/性能 | **NO-GO**：必须至少跑真实数据GPU 1K 和完整恢复 |

### 数据与模型不一致的核心

原 Model `fg_s3_e_source.make_binding()` 对实际 E-Data `S3_E_DATA_V1/index_manifest.json` 会报错，因为它假定所有来源都采用 synthetic signed (`content_hash`) 结构。即使绕过，Collector 也没实际实例化 E Provider、没有每分钟提供 E 两份数组，而且公开三种 E 组合为 `waiting_e`。因此 **Model EM01–EM15 的 CPU 测试不是生产训练准入**。

### 此次已实施（受限合成）

1. Data 逻辑代码、测试与报告并入 Gym Model 基线，无源文件冲突，原 `e_data_v1` 数学文件字节不改。
2. 新增 `e_data_v1/provider.py` 和 Gym `fg_s3_e_provider.py`：读取认证的原 TRAIN pool，校验来源和 collection，并使用预建的每合约 `MarketData` 在真实当前 Env cutoff 计算每一步 E 数组；外部原18维必须完全相等，缺少/错位的行情直接拒收；Source identity 绑定全部市场身份；未实现或开启生产训练。
3. 模型 `make_binding` 增加接受已验证的 `S3_E_DATA_V1` collection index，强制完整 collection 检验与源码hash；`scope=CPU_REAL_TRAIN_M4_NOT_FORMAL_RELEASED` 明确不代表训练授权。
4. 条件接入 `e_data_source_pool`，要求原完整pool SHA与模型准入一致，并逐合约比对经准备的DAgger TRAIN数据文件/Env/合约配置哈希；`LiveRun` 能按 E 模式构造 provider 并传给 V2Collector。**注意公开 `require_ready` 仍会提前阻断 E，此项是后续 M4 联调的接线，不是一次经过现场端到端的生产PASS。**
5. `e_data_v1` 新 Provider 合成因果测试验证0/14/15/40/359/600等真实时间切点、观察相等、跨合约拒绝、篡改拒绝。

## 3. 实际复验结果（本隔离环境）

- `PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 pytest -ra -q -p no:cacheprovider tests/e_data_v1`：**28 PASS、2 SKIP**。两项跳过的是依赖原 Gym Env CPU interpreter 的 F02/F04 集成核对，当前环境没有 `gymnasium`。原 Data 上传报告里的 22 PASS/6 PASS、18合约真实 TRAIN 是用户机器先前的证据，**本隔离环境未复跑真实1.3GB TRAIN 产物**。
- `(cd docs/e_data_v1 && sha256sum -c SHA256SUMS)`：报告和已上传验收证据 hash 校验 PASS。合并的旧 Task Spec Markdown 原有行尾两个空格；`git diff --check` 会提示这些老 Markdown 格式，不是本次新增的 Python 代码问题。
- `python -m py_compile`：新旧接线文件语法通过。
- Gym/Data、Dreamer/Model补丁在各自精确原始 base 上的 `git apply --check` 回归：待补丁生成后复核并记录在交付收据。
- **Dreamer真实模型测试、Model 新改动的真实JAX梯度验证、原生Collector+Replay完整恢复、GPU、DEV Actor-only、正式1M**：`NOT_RUN`。当前隔离环境缺少 `ninjax`、`elements`、`gymnasium`，也没有原作者的 1.3 GB真实 TRAIN 产物、源 Run、GPU。此前模型 CPU 58项是上传分支的历史验收，并非本次改变源码后的新运行证明。

## 4. 阻止正式训练的核心任务（按优先级）

**P0-1 真实E M4全链路**：在用户现场，用冻结原始 TRAIN pool + 完整 E collection，执行真实 `Env→ETrainProvider→V2Collector/Driver→DAgger→native Replay→join→S3V2Agent`，每分钟（包括轮转、新合约、1024截断、reset）做观察/源 `stepid` 与帐号连续性校验；至少测试 A2/Tutor/B2 各自 >=65个序列帧；明确比较原 `market_seq` 没有被改动。不能只复用 sparse stride15 bundle。

**P0-2 生产身份与可恢复性**：对真实 E SourceAdmission `direct_init`、完整 model/runtime/config、Gate CLOSED、训练原 joint optimizer、三教学标签、E provider dataset/代码版本做 source/prepared/model/run/checkpoint/next-sample 全链身份审计；保存→重载→相同下一批次+参数等价。固定两个 Git HEAD、数据schema哈希、环境包版本与授权状态。需清除旧 `BASELINE_FAIL_3` 或证明只属于严格排除的无关测试。

**P0-3 安全解除 waiting_e**：**只有 P0-1/2 完成并独立Review后**，修改唯一 `fg_s3_composition.py::require_ready()` 的能力状态映射及 formal source admission；三个新 E 组合（A2/Tutor/B2）仅在对应各自证据完整时解锁，Tutor+B2继续阻断。不能用环境变量/用户输入 `ready` 绕过。当前工作不解锁，防止误开正式训练。

**P0-4 真GPU smoke**：真实TRAIN、原完整size50m预设、模型+优化器+EMA、15m数组的混合精度检查；1K GPU更新、合法 ADD/WAIT 覆盖、非零 Encoder 三分支梯度与更新、`bc_effective_updates`、`policy pADD`、FullCheckpoint完整恢复、显存峰值、吞吐/1M耗时估计。若GPU OOM，必须作为单独模型配置实验重新审计，不能偷偷降size。

**P1 择时有效性监控**：在1M训练期间阶段性核验真正合法空仓ADD / voluntary WAIT、Adapter OFF/ON（B2）、Tutor pair coverage、Outcome头类别校准、Actor-only自行入场与Always-WAIT基准；训练 Loss 和整体accuracy不能证明会入场。正式 DEV 最后做，盲测 TEST 继续封存。

## 5. 最终验收决策

**当前 = NO-GO FOR FORMAL TRAINING**。不是否认 E-Data 的历史真实TRAIN验收，也不是E神经模型 CPU 测试失败，而是现实中尚缺真实两组件贯通、公共入口授权、full-checkpoint恢复和 GPU full-size 证据。保留旧真实数据与检查点身份，不覆盖既有 B2/Tutor 训练。

**完成上述 P0 后才可 GO**；建议先跑一条正式来源、单个合约的CPU real M4，后GPU 1K，再上线完整1M。完成之前不要改原 Reward、Gate80%、TP5/SL30或为了出现 ADD 放松掩码。
