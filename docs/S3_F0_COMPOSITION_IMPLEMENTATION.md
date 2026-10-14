# S3 Composition Foundation V1（F0）实施与验收

## 范围与基线

仅实现配置组合基础。保留 A2/Tutor-CD/B2 现有算法；E Encoder、15m 数据、Tutor+B2 联合教学、GPU及正式训练均不执行。当前 B2、历史 Tutor、Archive、main 和公共 integration 工作树保持原样。

- Gym 开发起点：`03a625c1661546b9183b9ceeb24119b516a87459`。
- Dreamer 开发起点：`d41bc80ba13b3dc3f5d2c32fcdc800613b2d0b4d`。
- 两仓开发分支：`feat/s3-f0-composition`。
- 工作区：`/home/v/Documents/work/research/fg-s3-f0-dev/{gym-trading-env,dreamerv3}`。
- 任务书已完整读取自原 Gym docs 的 `S3_Composition_Foundation_E_V1_Codex_Design_Reviewed_20261014.md`。
- 开始时两仓干净；最终 SHA、测试日志、保护记录和独立版本锁位于工作区 `validation/`、`baseline-lock.json`。

## 实施顺序

1. Dreamer 单一纯配置解析：八组合、旧 mode 映射、严格规范化和能力检查；不导入 JAX。
2. 保留旧配置 Schema；新增 `FG_S3_V2_COMPOSITION_CONFIG_V1`，新增配置不允许同时写 mode。新 settings 仅保存规范化组合及派生兼容 mode，原 YAML 保存为 experiment 快照。
3. 为 prepare、直接 Prepared 加载、LiveRun、模型工厂和评价添加前置能力/身份检查。拒绝未支持组合时不创建训练工件。
4. 新组合统一记录在 SourceAdmission、Prepared、模型配置、Run、Checkpoint 与评价中；组合 metadata 与 B2 agent_extension 分离。旧 identity 不自动补字段。
5. 保留现成 Encoder/Loss/优化器实现；仅增组合守卫。Tracker 复用同一解析器验证并展示新组合，旧映射不变。
6. 提供八个最小配置片段和三个独立 Legacy 诊断配置；CPU验证解析、拒绝、身份、恢复与现有真实小模型梯度路径。
7. 自审 Diff、保护原工件、提交双仓；交付后等待 Review，不自动合入 integration 或开展 F1。

## 公共接口与状态

原 `FG_S3_V2_DIAGNOSTIC_CONFIG_V1` 与 `mode: a2/tutor/b2/c2` 保持原路由。新配置只写：

```yaml
schema: FG_S3_V2_COMPOSITION_CONFIG_V1
composition:
  encoder: legacy_1m
  teaching: tutor_cd
```

规范化结果包含 encoder、teaching、base_oracle_bc、use_e/use_tutor/use_b2、legacy_mode、support_state、schema_version。双写 mode/composition，即使值一致也拒绝；未知值/缺字段/无效类型拒绝。

| teaching | legacy_1m | e_multiscale_1m15m |
|---|---|---|
| a2 | ready | waiting_e |
| tutor_cd | ready | waiting_e |
| b2 | ready | waiting_e |
| tutor_cd_b2 | waiting_tutor_b2 | waiting_tutor_b2 |

`describe-composition` 只读展示等待状态。训练/准备/恢复/评价入口必须 fail closed。旧 C2 不纳入八组合。新组合独立建 Run；旧 Run 的 exact resume 仍通过原冻结 SHA 与原身份校验。

## C01～C12 实际验收记录

以下 PASS 仅指 F0 指定合同与 CPU 验收范围，不代表正式训练或 E 模型验收。

| ID | 实际验收内容 | 当前结果 |
|---|---|---|
| C01 | 最新SHA/worktree/dirty、mode分派与E缺口 | PASS：起始快照 `validation/before.json`；分派见下节 |
| C02 | 八组合唯一规范化、基础BC不重复 | PASS：8个解析用例；三模式新旧真实模型更新完全一致 |
| C03 | 三 ready、三 waiting_e、两 waiting_tutor_b2 | PASS：解析重算所有派生字段，不能伪造 ready |
| C04 | 未实现组合在 prepare/start/resume 前拒绝且无产物 | PASS：五等待组合的 CLI/直接 prepare、LiveRun、evaluate 前置拒绝 |
| C05 | 旧配置解析、C2与历史身份语义不改 | PASS：旧 Schema 输出无新字段；三份原配置参数一致；冻结树保护记录 |
| C06 | 双写、未知值、缺字段拒绝 | PASS：双写即使一致也拒绝；类型、Schema、伪造 metadata 均拒绝 |
| C07 | Source/Prepared/Model/Run/Checkpoint/evaluate组合一致 | PASS（CPU）：三模式真实数据准入；签名元数据边界与追踪评价测试；B2原生 FullCheckpoint 含组合并拒绝错配；E真实恢复 NOT_RUN |
| C08 | Legacy三模式真实CPU模型/现有公共回归 | PASS（限定）：三模式实际梯度/模型恢复、24项模型/Checkpoint回归、79项公共合同及48项解析/Oracle/缓存测试。另有3项旧迁移 fixture 基线失败，详见下文，未标为通过 |
| C09 | E未来Encoder/Obs/Source解锁契约 | PASS：列明 F1 接入点；E实现/训练/恢复 NOT_RUN |
| C10 | Tutor+B2明确阻断，无联合Loss | PASS：两种组合 fail closed；未改任何教学Loss |
| C11 | Reward/Oracle/市场分钟/动作/Replay时序不变 | PASS：数学实现无Diff；真实Oracle资金证明/DP复核；原Oracle/时序/市场分钟回归 |
| C12 | 原实验保护、双仓提交与测试收据 | PASS：1191个受保护文件、7个工作树及原有分支引用不变；双仓提交/最终状态见独立 `baseline-lock.json`；不合入 integration |

## 实际改动与身份路径

- 单一解析器：`dreamerv3/fg_s3_composition.py`，使用标准库，无 JAX、文件读写或环境变量解锁。
- `read_settings → normalize_config`：新配置由 composition 派生现有 mode；旧 Schema 保留原键。`agent_extensions`、Agent Loss 分派和优化器没有另写实现。
- `prepare → inspect_source`：SourceAdmission 顶层和 `target_model_identity` 保存同一 `resolved_composition`，进入原签名/模型身份哈希。Prepared、model_config 同样保存该字段。原 experiment.yaml 保存原始配置字节。
- `settings_from_prepared / LiveRun`：先核能力，再检查被冻结文件/代码及组合；新模型的内存配置使用 `s3_composition`，Encoder 必须对应当前 `simple`。B2 `agent_extension` 独立保留。
- Run 身份记录规范化组合，FullCheckpoint 沿用完整 Run 身份的原 equality 校验；不新增恢复协议。旧 Run 不补写字段。
- `evaluate`：核实 Prepared/Source/Model/Run/Checkpoint 后加载相同模型扩展，最终签名 receipt 保存组合；未实现组合不创建评价输出。
- Tracker：`research_tracking/composition.py` 读取同一个纯解析器；只处理新 metadata，不导入训练器。W&B 配置记录 `s3/resolved_composition`；评价组合错配拒绝。没有在线上传或修改云端记录。

实际 mode 分派仍在 `fg_s3_v2_experiment.py` 的源/数据配置、`agent_extensions`、`S3V2Agent` 的 policy/loss/train、`LiveRun` 的 Sidecar/统计和 `fg_s3_v2_evaluation.py`；它们接收派生的同一个原 mode。Encoder 的既有选择为 Agent 的 `enc.typ → rssm.Encoder`，F0不增加网络层。

## 真实准备发现并修复的基线阻断

第一次 CPU prepare 被旧 Online180K producer 的 `entry_dataset_env.py` 字节校验拒绝；第二次进入池准备后，被原 Oracle E receipt 的同一文件校验拒绝。不是组合数学错误：当前集成基线已包含 `c14fcdc` 市场缓存及 `9eba965` CSV/DB连接复用，旧 receipt 固定的是优化前读取代码。

已读取实际 Git Diff，并由 producer SHA256 找到历史原字节。例如 Oracle receipt 的 `entry_env.py` 对应 `37f97c48ac95032983ef7840f82aa720e51008fa`，与当前该文件仅有市场读取/缓存优化差异。receipt 中其他 Env/utils/Reward 文件仍逐字节匹配。

最小适配放在公共 `fg_s3_reader_compat.py`：只接受指定的三份 producer 文件 SHA256和完整四文件目标 bundle（包括新缓存模块），并从 Git 对照原 producer 字节。SourceAdmission 只在参数 warm-start 使用该 seam；直接完整 transfer 不放行。Oracle loader 使用同一 helper，保存原/现代码及审查版本，继续核实所有行情/策略/候选哈希、TRAIN whitelist、原资金证明及保存的 C1 DP 最优结果。未知来源、目标 bundle 修改均拒绝。没有改写旧 receipt、重生成标签、修改认证数学或放宽 exact resume。

失败时创建的本工作树 `artifacts/fg_s3_v2/prepared/F0_A2_LEGACY_1K/` 保留，没有有效 manifest；成功准备使用独立 `_v2` 名称，不覆盖失败目录。

## 实际 CPU 准备收据

`validation/real_preparation.json`：三份配置全部 `PREPARED_NO_TRAINING`，耗时约74.3秒。使用真实 market-minute v2 TRAIN Env/Oracle/TeacherCatalog，以及已认证 Online180K Checkpoint（原模型更新数21183）。共同身份：

```text
dataset_sha256       4b004e3a9346a814ae9fbc646fbdf87181c3c8e449e01f7be8a69c4e45140822
reward_params_hash   6a44ade43ea7168318c479ac2b17f1058590259f33ef463035d3ece2f20cca8c
observation_identity a4a205df98c611f282b782b0355729bf35a347afb325c75d9b008d0a2fa413d6
training_hash        ffc45db584106b7c348b7f3aee95096acbcb560c31a7d67fa340f30d2e10009a
```

上述实施过程的准备目录为 `artifacts/fg_s3_v2/prepared/{F0_A2_LEGACY_1K_v2,F0_TUTOR_LEGACY_1K_v2,F0_B2_LEGACY_1K_v2}`。
它们记录当时 SHA/dirty 和实际代码哈希，不伪称正式冻结训练。神经模型未初始化，新增环境步0，Run目录未创建，GPU与1M均 NOT_RUN。交付配置使用独立 `_v3` 实验/准备/Run目录，供提交后干净版本验证；如已执行，其实际收据为 `validation/frozen_preparation.json`，状态和双仓SHA汇总在独立 `baseline-lock.json`。不编辑或覆盖前两次准备。

## 测试命令与结果

工作目录为上述 F0 Gym。模型测试均显式禁用GPU；固定环境：

```bash
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu
export MPLCONFIGDIR=/tmp/s3_f0_mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=src:../dreamerv3
PY=/home/v/miniconda3/envs/dreamerv3/bin/python

# 最终解析/拒绝/读取版本守卫及原Oracle/缓存测试：48 PASS
"$PY" -m pytest tests/entry_v1/test_s3_composition.py \
  tests/entry_v1/test_fg_s3_dagger_d0.py tests/entry_v1/test_entry_market_cache.py \
  -q -o addopts=''

# 三模式真实CPU更新/精确数值一致/模型继续训练：最终3 PASS
"$PY" -m pytest ../dreamerv3/embodied/tests/test_s3_composition.py -q -o addopts=''

# 完整模型/原生Replay-Sidecar-FullCheckpoint回归：24 PASS
"$PY" -m pytest ../dreamerv3/embodied/tests/test_s3_composition.py \
  ../dreamerv3/embodied/tests/test_fg_s3_b2.py \
  ../dreamerv3/embodied/tests/test_fg_s3_tutor_full.py \
  ../dreamerv3/embodied/tests/test_fg_full_checkpoint.py -q -o addopts=''

# 公共配置、DAgger、Outcome、Tutor、市场分钟、原在线模拟：79 PASS，2 GPU用例未运行
"$PY" -m pytest tests/entry_v1/test_s3_composition.py \
  tests/entry_v1/test_fg_s3_v2.py tests/entry_v1/test_fg_s3_v2_pool.py \
  tests/entry_v1/test_fg_s3_b2_outcome.py tests/entry_v1/test_fg_s3_tutor_e2.py \
  tests/entry_v1/test_s3_market_minute.py tests/entry_v1/test_fg_s3_online_simulation.py \
  -k 'not real_gpu and not actual_e2_gpu' -q -o addopts=''

# W&B Mock/Offline与已包含的Archive回归：49 PASS，无在线Run
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  /home/v/Documents/work/research/fg-s3-wandb-env/bin/python -m unittest \
  s3_archive.tests.test_archive research_tracking.tests.test_tracker \
  research_tracking.tests.test_eval_v2 research_tracking.tests.test_prediction_evals \
  research_tracking.tests.test_composition -v
```

实际执行另带 `--junitxml=../validation/<group>.xml`，stdout/stderr保存在同名 `.log`。记录：`reader_contracts_final`、`models_final`、`models_regression`、`contracts`、`tracking_offline.log`。不同组存在重复测试，不将组数相加冒充独立覆盖数量。

三模式数值测试使用原 FG Env 生成的受控行情和原生 Replay batch，在原1000-update学习率预热下执行两次联合更新后证明 Actor/WM 参数变化；旧/新模型所有参数及对应指标逐值一致，Critic等原冻结状态不变。各模式还保存/恢复模型与carry并继续第三次更新。原生 Replay、Sidecar、Driver、模型、计数完整恢复通过 B2共用 FullCheckpoint 测试，含新/旧身份两种情况。上述模型/恢复证据属于真实CPU小模型受控测试，不是正式市场/GPU训练验收。

## FAIL / NOT_RUN 与已修正测试问题

- `test_fg_s3_source_transfer.py`：2 PASS / 3 FAIL。未修改的公共 integration 基线也执行了相同命令，得到相同3项失败：`test_actual_three_modes_inherit_complete_state_and_reject_foreign`、`test_no_inference_source_and_frozen_snapshot`、`test_missing_baseline_requires_authenticated_features`。共同报错 `New S3 admission requires explicit S3_MARKET_MINUTE model semantics`；这些旧测试fixture没有新市场分钟身份。对照记录 `source_transfer.log/xml` 与 `source_transfer_baseline.log/xml`，不将失败标PASS，不为其绕过时钟校验。
- 新测试初版误要求学习率第0次更新立刻改变参数；实际原优化器从0预热，已改为执行两次更新，没有修改学习率。另一新增测试遗漏原 mode扩展，先被 mode守卫拒绝，已补齐有效mode再测试错误Encoder。最终对应测试通过，初始失败日志保留。
- NOT_RUN：GPU训练、正式1M、新Actor-only DEV市场运行、真实E Encoder/E Replay/E FullCheckpoint、Tutor+B2联合教学、W&B在线Run。旧正式训练不停止/不重启；main/integration不修改。
- 独立保护/版本记录以 `baseline-lock.json` 为准，记录实际完整SHA、测试收据哈希与最终状态；不把源码静态相等单独当成真实学习证明。

## F1 接入边界

当前 Encoder 注册只有 `simple`；旧观察的 `market_seq[60,18]` 不代表 E 的观察身份。E需要认证后的 `market_seq_e_shared[60,18]`、`market_context_15m[24,7]`、真实新 Encoder、Decoder/Replay/数据协议及合法 SourceAdmission。现有 B2 architecture extension 仅支持 warm_start；E新初始化的适配留在F1，不能用旧180K Encoder冒充E。

只有 E-Data/E-Model验收通过后才能在源码放行 E+A2/E+Tutor/E+B2；共同教学仍等待F2。F0的CPU身份/拒绝测试不构成真实E模型或E FullCheckpoint验收。
