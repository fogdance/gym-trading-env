# WP01_IMPLEMENTATION_REVIEW

## Summary

**WP01 工程验收：PASS，可进入 WP02 设计 Review。训练准入：NO-GO。**

仅实现统一课程合同、四阶段参数声明、Gym 原生只读 Preflight、身份哈希及自动化测试。
最终选定集合 **241 passed / 0 failed / 0 errors / 0 skipped**，五组 pytest 命令退出码均为 0。
这是选定 CPU 集合通过，不是全项目或研究效果通过。

两个独立开发 worktree 位于本报告旁的 `gym-trading-env/` 和 `dreamerv3/`，分支均为
`feat/s3-e-quality-wp01-20261016`。最终双仓 clean，`integration/s3-next` 的 HEAD 和状态保持原样；未自动合并。

| 提交 | 仓库 | 完整 SHA | 内容 |
| --- | --- | --- | --- |
| A | Dreamer | `347724e3207fca16907facab021302217f47cf6b` | 纯标准库合同、阶段矩阵、身份载荷、单元测试、版本说明 |
| B | Gym | `2904f3e58cd84fb383f9e5a0d2d2aa643fcb10d2` | 原生预检、CLI、示例、文档、负例和组合守卫回归 |

真实版本、源文件哈希、原始日志哈希见 [baseline-lock.json](baseline-lock.json)。跨仓版本锁和最终 Review
放在两个仓库外，避免最终提交 SHA 自引用；运行代码、测试和版本文档均已提交。

## Evidence observed

已阅读任务书和 V2 架构，原文 SHA256 及两仓 HEAD/status/branch/worktree 原始命令见
[源基线记录](evidence/source_baseline.json)，差异、能力和限制见 [BASELINE_AUDIT.md](BASELINE_AUDIT.md)。
当前开发起点是 Gym `6939de5b43200e6f19311804ebf1739b01f12ba5`、Dreamer
`51ed1d094c92fc52a910acda3e2a3421bc71c9e5`，不是文档中的旧归档版本。

代码和运行证据共同确认：

- 原 `fg_s3_composition.py` 仍返回 `waiting_quality_e`，原 `require_ready()` 仍抛 `S3_QUALITY_E_NOT_IMPLEMENTED`。
- 原 Quality 只接受 HEADS_ONLY/CONTROLLED_FULL；HEADS_ONLY 的旧 C1 约束、Prediction/Q/Decision 准入均未修改。
- Reward 校验调用真实 `Single3RewardConfig`，E 核对调用真实 `observation_contract()` 和 `EDataConfig.schema()`，Quality policy 核对现有 Score 协议。
- 提交后独立运行四个公开命令，prepare/start/resume/evaluate 均退出 1 并出现原 guard，未增加现有 artifacts 路径。
- 原生文件保护审计检查 Gym **415** 个、Dreamer **108** 个既有运行源码/配置，全部与开发起点字节相同，见 [source_protection.json](evidence/source_protection.json)。

## Root cause / 功能缺口

这次是功能新增，不是对金融算法或模型效果的修复。已有运行阶段和来源准入不能表达 V2 从零建立的新课程；
E+Quality 本身尚未认证。WP01 用独立的设计合同表达后续能力，保持全部运行阻断，避免把课程声明误认为执行许可。

自审发现并修复了本次新 CLI 的一个只读缺陷：仅设置环境变量不会改变当前 Python 的
`sys.dont_write_bytecode`。新增回归先真实失败（退出码 1），再加一行显式标志后通过。
[失败前日志](evidence/bytecode_before_fix.log) 与 [原命令/退出码](evidence/bytecode_before_fix.json) 保留，未跳过或放宽断言。
原生 pool helper 的真值语义也没有被当作认证；预检另要求 eligibility 是显式 True，字符串 `"false"` 有拒绝回归。

## Changes made

代码归属遵守 AGENTS：Dreamer 纯合同在 `dreamerv3/`，Gym 适配器在
`src/gym_trading_env/research/`，薄 CLI 在 `tools/`。没有新平行子包、教师、Env、特征计算器或训练器；
`artifacts/` 中没有新增独立训练脚本。测试仅运行已有隔离合成 CPU 生命周期。

### Commit A：3 files，536 insertions

- `dreamerv3/dreamerv3/fg_s3_e_quality_course_contract.py`
- `dreamerv3/dreamerv3/tests/test_fg_s3_e_quality_course_contract.py`
- `dreamerv3/docs/FG_S3_E_QUALITY_COURSE_V1.md`

### Commit B：6 files，921 insertions，4 deletions

- `gym-trading-env/src/gym_trading_env/research/fg_s3_e_quality_course_preflight.py`
- `gym-trading-env/tools/fg_s3_e_quality_course_preflight.py`
- `gym-trading-env/tests/entry_v1/test_fg_s3_e_quality_course_preflight.py`
- `gym-trading-env/tests/entry_v1/test_s3_composition.py`：把 E+Quality 加入现有等待组合负例，不修改生产 guard。
- `gym-trading-env/docs/fg_s3_e_quality_course_v1.md`
- `gym-trading-env/docs/fixtures/fg_s3_e_quality_course_v1.example.json`

完整 patch、diffstat、改动路径：
[Dreamer patch](evidence/dreamerv3_diff.patch)、[Dreamer diffstat](evidence/dreamerv3_diffstat.txt)、
[Gym patch](evidence/gym-trading-env_diff.patch)、[Gym diffstat](evidence/gym-trading-env_diffstat.txt)。
Gym 原仓忽略 docs/，这里只显式加入两份本包交付文档，没有修改忽略规则。

### 合同与输出行为

- 协议唯一：`FG_S3_E_QUALITY_COURSE_V1`。固定 E+Quality、direct_init、Quality TP C1、原生时钟、Observation 和交易断言。
- 阶段：FOUNDATION_WM、QUALITY_HEADS、ACTOR_SUPERVISED、ONLINE_ADAPT；派生矩阵不能输入或覆盖。
- FOUNDATION_WM 明确把 avail 与其它 WM 组列为 TRAIN；QUALITY_HEADS 逐类要求有效标签；F3 仅 quality_actor，F2 研究准入不阻断 F3；F4 条件式开放留待 WP07。
- 所有阶段 CONTRACT_ONLY / NOT_ADMITTED，training_enabled=false；WP01 的预测/Score/Q correction 全为 0。
- 教师信息只在 TRAIN sidecar 的声明层，valid、cutoff 和 host provenance 不变成 policy observation。
- 纯模块无 I/O。Gym 使用原 fingerprint/signed/write_json，未建立另一套签名格式。content_hash 只表示完整性。
- 显式合法 draft 可有明确 null 身份并退出 0；formal 即使元数据匹配也退出 1；非法输入/来源/路径退出 2。
- 意外缺失、空字符串、空来源、伪造认证不能降级为 draft。只有合法 draft 可以写一个非认证诊断文件。
- 引用元数据只核验声明的哈希/绑定；不验证或读取底层真实行情、标签、账户轨迹或参数。source_bytes_verified/time_bounds_verified/official_calendar_verified 均为 false。

## Verification performed

### 最终选定测试集合

所有命令的 cwd 都是：`/home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/gym-trading-env`。
使用现有 Dreamer Python，JAX_PLATFORMS=cpu、CUDA_VISIBLE_DEVICES 为空、BLAS/OMP=1、禁字节码与 pytest 缓存。

| 组 | 通过 | 失败 / 错误 / 跳过 | 退出码 | 原始输出 |
| --- | ---: | --- | ---: | --- |
| 合同/CLI/组合（自审后） | 139 | 0 / 0 / 0 | 0 | [final_contract_preflight_after_review.log](evidence/final_contract_preflight_after_review.log) |
| 共享 Gate/SourceAdmission/formal guard | 20 | 0 / 0 / 0 | 0 | [final_shared_guards.log](evidence/final_shared_guards.log) |
| Teacher/Reward/Admission/Clock/E/Replay | 67 | 0 / 0 / 0 | 0 | [final_native_regressions.log](evidence/final_native_regressions.log) |
| Legacy Quality 生命周期与学生更新 | 9 | 0 / 0 / 0 | 0 | [final_quality_lifecycle.log](evidence/final_quality_lifecycle.log) |
| E 三组合/Driver 精确恢复 | 6 | 0 / 0 / 0 | 0 | [final_e_resume.log](evidence/final_e_resume.log) |

[汇总与准确 argv](evidence/verification_summary.json)；每组对应的 `_cases.json` 和 `.xml` 保存实际执行的 testcase。
提交后另行 collection：241 tests collected，退出码 0，见 [完整清单](evidence/final_collection.log) 和
[collection 命令](evidence/final_collection.json)。不是把“收集成功”当成运行通过。

最终命令逐条如下（实际执行原文）：

**合同/CLI/组合（自审后），exit=0**

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/evidence/final_contract_preflight_after_review.xml ../dreamerv3/dreamerv3/tests/test_fg_s3_e_quality_course_contract.py tests/entry_v1/test_fg_s3_e_quality_course_preflight.py tests/entry_v1/test_s3_composition.py
```

**共享 Gate/SourceAdmission/formal guard，exit=0**

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/evidence/final_shared_guards.xml ../dreamerv3/embodied/tests/test_fg_s3_gate.py ../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py ../dreamerv3/embodied/tests/test_action_mask_formal_guards.py
```

**Teacher/Reward/Admission/Clock/E/Replay，exit=0**

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/evidence/final_native_regressions.xml tests/entry_v1/test_fg_s3_quality_teacher_v2.py tests/unit/test_quality_tp_reward_v1.py tests/entry_v1/test_fg_s3_quality_admission_v2.py tests/entry_v1/test_s3_market_minute.py tests/entry_v1/test_fg_s3_e_observation.py tests/entry_v1/test_s3_e_native_env.py tests/entry_v1/test_fg_s3_e_admission.py ../dreamerv3/dreamerv3/tests/test_fg_s3_e_encoder.py tests/entry_v1/test_fg_single3_replay.py
```

**Legacy Quality 生命周期与学生更新，exit=0**

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/evidence/final_quality_lifecycle.xml tests/entry_v1/test_fg_s3_quality_lifecycle_v2.py tests/entry_v1/test_fg_s3_quality_student_v2.py
```

**E 三组合/Driver 精确恢复，exit=0**

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/evidence/final_e_resume.xml ../dreamerv3/embodied/tests/test_fg_s3_e_checkpoint.py ../dreamerv3/embodied/tests/test_driver_account_checkpoint.py
```

### 16 项行为与正负例映射

| 项 | 验收行为 | 实际测试证据 | 结果 |
| --- | --- | --- | --- |
| 1 | 正常草案可解析，全阶段只声明、永不准入 | pure `WP01_01`；native `WP01_01_12` | PASS |
| 2 | legacy encoder、Tutor/B2、旧参数来源拒绝 | pure `WP01_02`；CLI composition 负例 | PASS |
| 3 | 旧 Reward、C2、C1 参数变化拒绝；调用原生 Reward validator | pure `WP01_03`；native spy 与 CLI 负例 | PASS |
| 4 | 两张 E shape、valid/cutoff 缺失拒绝 | pure `WP01_04`；CLI shape 负例 | PASS |
| 5 | WAIT/ADD、3手、TP/SL、Clock 修改拒绝 | pure `WP01_05`；CLI clock/lots | PASS |
| 6 | DEV/TEST 教师与 TEST 来源拒绝 | pure `WP01_06`；native `test_source` 元数据负例（无 TEST 行情读取） | PASS |
| 7 | teacher/suffix/Outcome/Q/host 字段不能进入 policy | pure `WP01_07`；CLI leak | PASS |
| 8 | Oracle 缺来源仍 blocker；伪造 certified 拒绝；有声明也不认证 | pure `WP01_08`；native `WP01_08`、formal forged | PASS |
| 9 | ready/approved/admitted/support_state/resolved 等输入拒绝 | pure `WP01_09`；CLI forged_ready | PASS |
| 10 | 重复 ID、非法概率、缺 seed、负延迟、未知行为拒绝 | pure `WP01_10`；CLI unknown_behavior | PASS |
| 11 | F1/F2 Actor TRAIN、F3 pol 冒名、派生矩阵篡改拒绝 | pure `WP01_11`；CLI forged_matrix | PASS |
| 12 | 仅显式合法 draft 容许 null；formal、缺字段、空身份非零 | pure `WP01_12`；native draft/formal/空来源系列 | PASS |
| 13 | 同输入哈希/字节相同；列表顺序、seed、源哈希改变身份 | pure/native `WP01_13`；最终两个诊断文件字节比较 | PASS |
| 14 | legacy Quality、E+A2/Tutor/B2 既有真实配置保持；CPU 恢复不退化 | native `WP01_14`，legacy lifecycle，EM15 三模式 | PASS |
| 15 | E+Quality 四公开入口、来源和模型工厂在生产产物前拒绝 | 扩展原 test_s3_composition；最终四条 CLI guard | PASS |
| 16 | stdout/可选文件、错误非零、禁止副作用、无 JAX 纯导入 | pure `WP01_16`；native JSON/路径/字节码测试 | PASS |

本包未通过的必需验收项：**无**。下面的 NOT_ADMITTED 是有意保留的执行阻断，不被解释为训练成功。

### 最终 SHA 下的 CLI、文件和负例

提交后重新执行，双仓 SHA 与 dirty=false 已进入真实诊断报告：
[course_contract_receipt.json](evidence/draft_report_final_1/course_contract_receipt.json)。
它是非认证诊断，不能作为训练、Prepared 或 Checkpoint 来源收据。

- course_identity_hash：`77cd735b8bd67e5705fae661e16498220b6599426a0ff293a937e35ea6c9ea25`。
- 两次新目录输出字节完全相同，诊断文件 SHA256：`357c65889a2f00d809f6a33bbe891d8c3ae810447bfac725cfe4add283e15657`。
- 最终 CLI 命令、真实退出码、stdout/stderr 路径见 [cli_final_commands.json](evidence/cli_final_commands.json)。
- four public commands 的 stderr 含 S3_QUALITY_E_NOT_IMPLEMENTED，现有 artifacts 目录集合前后相同。

| 最终 CLI 场景 | 真实退出码 | 文件/状态证据 |
| --- | ---: | --- |
| cli_draft_final | 0 | [stdout](evidence/cli_draft_final.stdout.json) / [stderr](evidence/cli_draft_final.stderr.log) |
| cli_draft_repeat | 0 | [stdout](evidence/cli_draft_repeat.stdout.json) / [stderr](evidence/cli_draft_repeat.stderr.log) |
| cli_formal_missing | 2 | [stdout](evidence/cli_formal_missing.stdout.json) / [stderr](evidence/cli_formal_missing.stderr.log) |
| cli_forged_approval | 2 | [stdout](evidence/cli_forged_approval.stdout.json) / [stderr](evidence/cli_forged_approval.stderr.log) |
| cli_formal_complete | 1 | [stdout](evidence/cli_formal_complete.stdout.json) / [stderr](evidence/cli_formal_complete.stderr.log) |
| cli_formal_mismatch | 2 | [stdout](evidence/cli_formal_mismatch.stdout.json) / [stderr](evidence/cli_formal_mismatch.stderr.log) |
| public_guard_prepare | 1 | [stdout](evidence/public_guard_prepare.stdout.json) / [stderr](evidence/public_guard_prepare.stderr.log) |
| public_guard_start | 1 | [stdout](evidence/public_guard_start.stdout.json) / [stderr](evidence/public_guard_start.stderr.log) |
| public_guard_resume | 1 | [stdout](evidence/public_guard_resume.stdout.json) / [stderr](evidence/public_guard_resume.stderr.log) |
| public_guard_evaluate | 1 | [stdout](evidence/public_guard_evaluate.stdout.json) / [stderr](evidence/public_guard_evaluate.stderr.log) |

### 早期执行与自审证据

以下记录保留，不重复计入 241 项最终集合：

- `baseline_lifecycle`：exit=0；[原记录](evidence/baseline_lifecycle.json)，[原日志](evidence/baseline_lifecycle.log)。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/evidence/baseline_lifecycle.xml tests/entry_v1/test_fg_s3_quality_lifecycle_v2.py
```

- `contract_A`：exit=0；[原记录](evidence/contract_A.json)，[原日志](evidence/contract_A.log)。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/evidence/contract_A.xml ../dreamerv3/dreamerv3/tests/test_fg_s3_e_quality_course_contract.py
```

- `preflight_B_initial`：exit=0；[原记录](evidence/preflight_B_initial.json)，[原日志](evidence/preflight_B_initial.log)。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/evidence/preflight_B_initial.xml tests/entry_v1/test_fg_s3_e_quality_course_preflight.py tests/entry_v1/test_s3_composition.py
```

- `final_contract_preflight`：exit=0；[原记录](evidence/final_contract_preflight.json)，[原日志](evidence/final_contract_preflight.log)。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q --junitxml=/home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/evidence/final_contract_preflight.xml ../dreamerv3/dreamerv3/tests/test_fg_s3_e_quality_course_contract.py tests/entry_v1/test_fg_s3_e_quality_course_preflight.py tests/entry_v1/test_s3_composition.py
```

- `bytecode_before_fix`：exit=1；[原记录](evidence/bytecode_before_fix.json)，[原日志](evidence/bytecode_before_fix.log)。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -o addopts= -q tests/entry_v1/test_fg_s3_e_quality_course_preflight.py::test_WP01_16_cli_disables_bytecode_without_external_python_flags
```

字节码失败是自审中真实复现并修复的缺陷，没有被跳过或隐藏。自审后仅新预检 CLI 的标志及其测试有变化，
已重跑完整受影响集合 139 项；其它组的运行依赖字节未变，原训练 execution_code 不包含这个新增诊断 CLI 和 tests。
测试时 Gym 是基线 HEAD + staged 代码，Dreamer 已在 A 提交；最终提交不改变被测字节。
提交后再次逐文件比较最终源码与被测快照一致，详见
[tested_source_snapshot.json](evidence/tested_source_snapshot.json)、
[final_tested_source_snapshot.json](evidence/final_tested_source_snapshot.json)、[baseline-lock.json](baseline-lock.json)。

自审检查了输入/派生边界、标签隔离、参数矩阵、相对引用、错误退出、写入边界、名称/日志语义、
既有运行源码保护、测试实效和完整 diff；`git diff --check` 退出 0。本次未增加 debug logging、待办占位、xfail 或测试跳过。

## Risks / not verified

**未实现 / 未认证（后续工作包，仍然 NO-GO）：**

- 真实 E-native Corpus、Quality TP strict suffix/Teacher/Sidecar 生成与认证；时间边界、官方日、源市场字节认证。
- FOUNDATION_WM、QUALITY_HEADS、ACTOR_SUPERVISED、ONLINE_ADAPT 的运行实现、优化器冻结更新及阶段 handoff。
- 新学生的梯度、参数变化、Actor 能力、真实交易效果及独立研究验收。
- Prediction、Q、Decision 的新研究认证；任何预测对行动的修正。

**NOT_RUN：**GPU、正式训练、旧 E+Tutor 1M 参数读取/迁移、真实 TRAIN/DEV/TEST 行情访问、全项目测试、
W&B 上传、发布或合并。现有 CPU 回归使用合成原生 Env/Replay/Checkpoint，不能替代真实行情或 GPU 验收。
纯预检测试中禁止 Env 构造、read_csv、np.load 和真实 source loader；参考元数据的 market 文件甚至不存在。

最终草案完整 blockers（13 项）：

- `E_QUALITY_EXECUTION_NOT_IMPLEMENTED`
- `COURSE_STAGES_NOT_IMPLEMENTED`
- `TRAIN_SOURCE_BYTES_NOT_VERIFIED`
- `QUALITY_TP_CORPUS_NOT_CERTIFIED`
- `QUALITY_TP_TEACHER_NOT_CERTIFIED`
- `STAGE_HANDOFF_NOT_IMPLEMENTED`
- `TRAINING_AUTHORIZATION_NOT_GRANTED`
- `MISSING_SOURCE_MANIFEST`
- `MISSING_MARKET_SOURCES`
- `MISSING_E_PROFILE_SCHEMA_SHA256`
- `MISSING_TEACHER_MANIFEST`
- `QUALITY_ORACLE_SCHEDULE_NOT_IMPLEMENTED:P4`
- `MISSING_QUALITY_ORACLE_SOURCE:P4`


## Next recommended step / 回滚与 WP02 接口

WP02 先评审教材类别和有限 TRAIN 小样，复用：
`validate_course_spec` → `course_identity_payload` → 原生 fingerprint。
把真实来源加入草案会改变课程身份，必须重新冻结并另做 Corpus/Teacher 来源认证，不沿用旧 null 身份。
当前 JSON 示例不能交给训练入口执行。WP02 真实素材生成和后续计算另行授权。

本次没有合并 integration，最简单的暂停方式是保留开发分支和全部证据，原训练仍使用原冻结 worktree。
若之后需要回滚代码，在对应开发分支先 revert Gym B，再 revert Dreamer A；不执行 reset/clean、删除工件或重写历史：

```bash
git -C /home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/gym-trading-env revert 2904f3e58cd84fb383f9e5a0d2d2aa643fcb10d2
git -C /home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/dreamerv3 revert 347724e3207fca16907facab021302217f47cf6b
```

以上是回滚说明，本次没有执行回滚或清理。保留所有历史和本次合成测试工件。
