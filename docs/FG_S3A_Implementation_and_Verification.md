# FG S3-A Implementation and Verification

日期：2026-10-12。任务书：[S3-A V1](FG_DreamerV3_S3A_Oracle_Behavior_Cloning_Codex_Implementation_V1.md)。
本报告交付 S3-A D0–D4 的开发、CPU 验证和自审；正式 GPU BC 与 DEV 评价尚未执行。

## Summary

**Actor-only BC 代码和 CPU 验证已完成。** 可以通过独立入口只读加载符合身份的完成版 C1_M1_1M 模型，重建原 Encoder/RSSM 特征，仅更新原 `self.pol`。没有增加网络层、重新搜索 Oracle、修改 Reward，或加入 RL+BC 联合 Loss。

**当前不能启动本任务书规定的正式源模型训练**：已核对的 C1_M1_1M 没有完成 62,500 更新的静态最终 Checkpoint。生产仓库又出现 gate / 阶段迁移修订，重训产物的代码身份必须重新核对。不能将下面的小模型 CPU 证据冒充完整生产模型加载验收。

用户最后指示“开发完自己这部分就行”；本轮没有启动、重启或中断 C1 训练，没有启动正式 S3-A GPU 作业。

| Gate | 交付状态 | 实际证据 |
| --- | --- | --- |
| D0 | 审计完成；生产保护文件出现外部变化，保留 DRIFT 记录 | `D0_initial.json`、`D0_final.json`、`D4_delivery_audit.json`；本 S3-A 两仓差异均为新增文件 |
| D1 | PASS | 完整真实 TRAIN 标签：402,096 / ADD 16,171 / WAIT 385,925；18 项测试 |
| D2 | PASS（CPU 原神经 Agent） | 教师历史、真实 mask、1024 边界、固定 RNG、未来隔离、缓存与原 policy 对齐 |
| D3 | PASS（CPU） | 独立优化器等价；非 Actor / 原优化器 / Norm / Gate 不变；精确断点恢复；更新后 policy |
| D4 | PASS（CPU 工程验收）；D0 生产漂移例外单列 | 12 项 Dreamer 测试、真实 P4 10+10 vs 20 更新、CLI 与自审收据 |
| D5 | NOT_RUN | 完整生产 payload、正式 GPU BC、独立 DEV 闭环均未执行 |

两个独立开发仓库均从当时当前分支建立 `feat/fg-s3a-oracle-bc`：

| 仓库 | 开发基线 HEAD | 功能提交 |
| --- | --- | --- |
| `/home/v/Documents/work/gym-trading-env2` | `79aa9a9664a648fc961277986290874a6eb1c9d1` | `05f5bfc`（D0）、`261d500`（标签）、`3e70f61`（身份与边界） |
| `/home/v/Documents/work/dreamerv3-dev` | `446e6fc9dbc92a1be24d0d44b840e85d4f96010e` | `358b267`（Actor / feature / checkpoint / CLI / 测试） |

最终文档及收据另作交付提交。收据中的 HEAD 是生成时快照，最终 HEAD 用 `git log -1` 核对。

## Evidence observed

### 数据、时序和身份

已经阅读真实 Oracle / Replay 报告及当前源码，而非沿用任务书的历史 d33/gy4 HEAD。原报告：[Oracle](../../gym-trading-env/docs/FG_Single3_Oracle_Validation.md)、[Replay](../../gym-trading-env/docs/FG_Single3_Replay_R0_R2_Coverage.md)。

| 输入 | SHA256 / 身份 |
| --- | --- |
| Replay manifest | `c9e325b17be7bef1c83dff4c562f22bf61bd0d2ad01beb827f40c95fa136a17c` |
| Oracle manifest | `697c01439e86b026054302034bc3dc8d4de0e7abef0c7c11e4794522dd955d01` |
| Coverage manifest | `62b55c0ce0eeb8b9678cec948cd7ba9c0040dbde213ca053af3f461e96fda18d` |
| TRAIN identity | `ffc45db584106b7c348b7f3aee95096acbcb560c31a7d67fa340f30d2e10009a` |
| C1 Reward | `96f34998ce005b7e5f08239dbeba95239cdd2b97e0e2d3596ce73fe173a17aa1` |
| Observation | `4eb0ddfced5a3fdeeed559dbf041e776611fe68a5731374b528f5878859e7dff` |
| 当前标签缓存 manifest | `2f2a9de785bb963fe7e20fdc7c4b418ba7c2d5f27dfe0b1db5517e00e247d39f` |

原 Replay / Oracle 目录分别为生产 Gym 下 `artifacts/fg_single3_v1/replay/run_20261011_01`、`oracle/run_20261011_01`，全程只读。当前标签缓存位于开发 Gym 的 `artifacts/fg_s3a_v1/labels_20261011_v2`，18 个合约分片。旧 v1 收据保留，但生成器身份已变化，当前代码会拒绝旧缓存；不要复用。

逐行原生 Replay / Sidecar / schedule 对账结果：

| 项目 | 实际数量 |
| --- | ---: |
| 全路径 Replay 行 | 488,629 |
| 合法空仓监督行 | 402,096 |
| outgoing ADD | 16,171 |
| outgoing 自主 WAIT | 385,925 |
| 持仓强制 WAIT 排除 | 82,251 |
| NO_TRADE 排除 | 3,795 |
| 行政末行排除 | 469 |
| terminal 排除 | 18 |
| 保留的合法 reset 行 | 408 |
| 下一反馈分钟无效但确实推进执行的合法 WAIT | 5 |
| 错用 lots_before 会漏掉的合法行 | 14,087 |

当前观察对应 Sidecar `lots_after`；本行 `action` 是 outgoing 标签，本行 `incoming_action` / Reward 是此前动作反馈。退出后的当前空仓可以监督。5 条特殊 WAIT 已实际推进，按原定义保留且单列审计。监督按合约隔离 cursor / time / stepid 去重。

ADD 占比 **4.021676%**；Always-WAIT accuracy **95.978324%**，恒定分数 PR-AP 基线 **4.021676%**。没有类别加权或正例重采样。

标签按决策所在年份 / 日夜分层，旧 Coverage 表的 `add_fill` 按下一行成交反馈分层，两者跨日夜和年界时不同：例如标签 day ADD 10,888 / night 5,283；旧成交表为 10,870 / 5,301。依据 `fg_replay_coverage.row_labels()` 的 decision / fill 时序契约，不能把 fill 分层作为 outgoing 标签分层。总 ADD 完全一致。

### 原模型、冻结和推理证据

诊断使用真正的原神经 Agent、随机初始化的小尺寸原架构，不是 fake agent，也不是生产参数：RSSM deter=32 / stoch=2 / classes=4，feature width=40，CPU BF16 计算、float32 缓存。真实 P4 前 1,100 行产生 892 个合法样本（ADD 14 / WAIT 878），明确标记 `diagnostic=true`、`complete=false`；正式源模型禁止这种前缀缓存。

`D4_real_resume_and_policy.json` 证明：

- 208 个状态键中 203 个冻结键逐键哈希不变，改变的 5 个键全部为 `pol/*`。
- 原 joint updates / batches 均保持 0；另一单测事先设置非零旧优化器与 Norm 状态，BC 后也完全不变。
- 10+10 更新恢复与连续 20 更新：全部模型数组、独立优化器 PyTree、采样器及每步日志指标一致。
- 恢复后原 policy 的 32 行概率与更新后 Actor 对齐，最大误差 `5.960464477539063e-08`；相对原 Actor 的最大概率变化 `0.0035840868949890137`。
- 同一教师历史的 latent 在 BC 前后不变；`policy_params` 已同步，`pending_sync=None`。
- JAX 设备为 `TFRT_CPU_0`；没有加载生产 `agent.pkl`。

原生产模型的**元数据兼容性**另有 `D4_production_metadata_compatibility.json`：开发 Dreamer 的 15 个 core 文件与既有源身份吻合，9 项依赖版本吻合，architecture / learning 配置吻合。正式宽度 5120；全量 float32 特征约 8,234,926,080 字节（7.67 GiB），使用 mmap 分片。该收据为 `PASS_METADATA_ONLY`，不是完整模型运行验收。

### 原训练和文件保护

初始训练 PID 1078561、progress 9,500；初始保护 152 个生产文件。历史末次主机核查已记录进程退出，失败收据 `KeyboardInterrupt()`、`training_complete=false`，progress 停留 10,800，最近 Checkpoint 10,000；当时没有 62,500 最终节点或完成 summary。日志不能确定中断触发者。用户随后说明 gate 问题及重训安排，**该说明不等同于本轮已复现 C1 gate 故障根因**。

`D0_final.json` 原样保留全局保护核查 FAIL：8 个生产文件相对初始指纹变化，涉及 offline/gate/阶段迁移和 Gym experiment/evaluation。随后生产 Dreamer HEAD 已到 `3086573`，而本 S3-A 仍基于 `446e6fc`。本任务对两个开发仓库的 diff 全是新增命名空间；原 `agent.py`、JAX wrapper、配置、Env、Reward、Oracle、Replay 未被本任务修改。不能声称生产全局 152 文件始终不变，也不能把外部源码修订自动归入 S3-A。

本轮没有 signal / kill / C1 restart；核心补丁未应用，GPU BC 未启动。最新源文件快照和旧源身份是否仍匹配见 `D4_delivery_audit.json`。

## Root cause / 设计依据

这是新增监督训练功能。原 `Agent.train()` 使用全模型优化器，因此独立 BC 入口通过原 `pol/*` 参数子树求梯度；原 world model、critic、joint optimizer、Norm、Gate 都不作为可更新参数。

开发中确认并修正两个实际问题：

1. 原 `Agent.policy()` 诊断在 ADD 被 mask 时调用 `prob(ADD)`，实测触发 `Action must be valid under its mask`。新 `S3AAgent` 适配器调用原前向与 reset/carry，仅以 masked logits softmax 返回安全 ADD 概率。测试同时证明原路径失败、适配器通过且合法状态 carry / action 精确一致。原模块一行修正留在 `docs/FG_S3A_pending_integration.patch`，未应用；S3-A 运行不依赖应用该补丁。
2. 首轮恢复测试真实失败：通用 `host()` 把 Optax namedtuple 转为普通 tuple，破坏 PyTree。BC 保存改用 `jax.tree.map` 保留类型；`D3_resume_failure.xml` 保留失败，最终 pytest 与 10+10 vs 20 证明恢复通过。

浮点一致性采用明确的 `<=1e-6` 概率容差；carry / action、恢复模型 / 优化器 / 采样器仍要求精确数组一致。不同 JIT 路径的 `2.98e-08` 误差不再被误报为结构不一致。

## Changes made

| 新文件 | 作用 |
| --- | --- |
| Gym `src/gym_trading_env/research/fg_s3a_labels.py` | 严格 TRAIN / C1_P4 / H333 / 原生逐行对账；合法空仓索引与缓存身份 |
| Gym `tools/fg_s3a.py` | CPU 标签构建与审计 CLI |
| Gym `tests/entry_v1/test_fg_s3a_labels.py` | 原 Env / collector 原生边界、错位、外来身份、缓存篡改测试 |
| Dreamer `dreamerv3/fg_s3a_actor.py` | 原 `self.pol`、masked logp、独立 `s3a_bc_opt`、同步、逐键冻结证明、不平衡指标 |
| Dreamer `dreamerv3/fg_s3a_agent.py` | 原 policy 安全概率诊断适配；没有新网络模块 |
| Dreamer `embodied/run/fg_s3a_checkpoint.py` | 完成源准入、原模型精确加载、原子 BC checkpoint、精确恢复、审计 |
| Dreamer `embodied/run/fg_s3a_features.py` | 原 Encoder/RSSM 教师历史提取，固定 seed，全历史前向，合法行 mmap 分片 |
| Dreamer `embodied/run/fg_s3a_train.py` | 等权 BC、全行均匀排列采样、独立预算/日志/身份与恢复 |
| Dreamer `embodied/run/fg_s3a_eval.py` | 后续独立 DEV 原闭环接入与基线接口；本轮未执行 |
| Dreamer `scripts/fg_s3a.py` | features / train / audit / smoke / eval；默认 CPU，GPU 必须显式指定 |
| Dreamer `embodied/tests/test_fg_s3a.py` | 原神经模型前向、独立梯度、冻结、恢复、未来隔离与缓存检查 |

BC 默认 batch=256、lr=4e-5、AGC=0.3、warmup=0、constant schedule，ADD/WAIT 均权重 1。优化子树与 `Optimizer([self.pol])` 一步结果已通过等价测试；不调用原 joint loss / trainer，不借用源 `run.steps`。

源模型必须是 C1 / C1_natural_oracle / 原 Obs / Reward / seed=0 / replay_seed=202、horizon=333；正式 profile=`C1_M1_1M`、protected 62,500 更新、完成 summary 与最终 manifest SHA / 路径绑定。源 payload 键 / shape / dtype / counters 精确匹配；源参数逐字节载入，原优化器状态保留但不会更新。

标签身份绑定冻结 TRAIN、Oracle、每个 trace / reconciliation / 原生文件及生成器代码。特征绑定源 Checkpoint、世界模型参数、标签 manifest、完整历史遍历、RNG seed、dtype、分片哈希、实际代码和依赖。BC 身份再绑定 features、labels、source、优化器、batch、seed/run-id 与采样规则；同一身份恢复只允许延长目标预算。源模型先加载，再恢复 BC；不会用原源 Actor 覆盖恢复权重。

## Verification performed

所有最终收据位于 [development_20261011](../artifacts/fg_s3a_v1/development_20261011)。

| 验证 | 结果 / 收据 |
| --- | --- |
| Gym 标签与身份 | `D1_identity_pytest.xml`：18 passed，11.977s |
| Dreamer 新 BC 5 项 + 现有 mask/context 7 项 | `D2_D4_pytest.xml`：12 passed，36.794s |
| 真实 P4 CLI smoke / 恢复 / 连续对照 | `D4_smoke.stdout`、`D4_resume.stdout`、`D4_continuous.stdout`；均成功 |
| 真实恢复和新 policy | `D4_real_resume_and_policy.json`：PASS；验证脚本 `D4_verify_cpu.py` |
| BC / 标签 CLI 审计 | `D4_cli_results.json`：两个审计 exit=0 |
| 未完成源模型准入 | `D4_inflight.stdout`：预期 exit=1，拒绝 8,000 节点 |
| 原生产架构/依赖 | `D4_production_metadata_compatibility.json`：PASS_METADATA_ONLY |
| 两仓 diff / 保护 / 已保存证明 | `D4_delivery_audit.json`；diff check / pending patch check 通过 |

共 **30 项最终测试通过**。早期 `D1_pytest.xml` / `D2_D3_pytest.xml` 是历史收据，不重复计数。完整原生合成轨迹跨 1024（真实持仓恢复）；另有真实 P4 BF16 CPU 端到端。篡改 stepid / ordinal / action / incoming_action、旧 schema、C0/C2/C3、Obs/Reward/Oracle identity、重复 reset、非法 mask、标签/特征/Checkpoint 文件均拒绝。

未来隔离实测：固定历史动作和 RNG，改写后续 market_seq / outgoing action / Reward，早期特征逐数组完全不变；新增 `oracle_label` 字段被 schema 拒绝。合约独立初始 carry；按实际 `is_first` 复用原 policy reset，1024 行政边界不凭空清仓、不用乱序窗口重建 latent。

自审检查了两个仓库相对基线的全部 diff、原结构与调用契约、负类过滤、identity 和源加载顺序、类型保留、policy 同步、缓存完整性、错误与日志语义。全部实现位于新增文件；个人已有 untracked 文件保留。没有无关修复。

### 可执行 CPU 命令

以下命令从 Gym 开发仓库执行；新输出目录必须不存在。Pytest 使用临时目录，不修改生产输入。单进程 / 原生线程 1，明确禁止使用 GPU。

```bash
GYM_DEV=/home/v/Documents/work/gym-trading-env2
DREAMER_DEV=/home/v/Documents/work/dreamerv3-dev
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
DATA=/home/v/Documents/work/gym-trading-env/artifacts/fg_single3_v1
LABELS="$GYM_DEV/artifacts/fg_s3a_v1/labels_20261011_v2"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$DREAMER_DEV:$GYM_DEV/src"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$GYM_DEV"

"$PY" tools/fg_s3a.py labels-audit --labels "$LABELS"
"$PY" -m pytest -q tests/entry_v1/test_fg_s3a_labels.py
"$PY" -m pytest -q "$DREAMER_DEV/embodied/tests/test_fg_s3a.py" \
  "$DREAMER_DEV/dreamerv3/tests/test_action_mask.py" \
  "$DREAMER_DEV/dreamerv3/tests/test_fg_replay_context.py"

# 可重新构建标签；不搜索 Oracle，也不覆盖已经存在的缓存。
"$PY" tools/fg_s3a.py labels-build \
  --dataset "$DATA/replay/run_20261011_01" \
  --oracle-receipt "$DATA/oracle/run_20261011_01" --candidate C1 \
  --output "$GYM_DEV/artifacts/fg_s3a_v1/labels_recheck_new" \
  --seed 0 --run-id LABEL_RECHECK --max-cpu-workers 1

# 新建随机小尺寸原 Agent，对真实 P4 前缀进行 CPU BF16 工程验证。
SMOKE="$GYM_DEV/artifacts/fg_s3a_v1/smoke_recheck_new"
"$PY" "$DREAMER_DEV/scripts/fg_s3a.py" smoke --labels "$LABELS" \
  --output "$SMOKE" --seed 0 --run-id CPU_RECHECK --device cpu \
  --compute-dtype bfloat16 --max-rows 1100 --shard-rows 256 \
  --batch-size 32 --updates 10 --save-every 10
"$PY" "$DREAMER_DEV/scripts/fg_s3a.py" train --diagnostic --resume \
  --labels "$LABELS" --source-checkpoint "$SMOKE/source/ckpt/step_000000000000" \
  --features "$SMOKE/features" --output "$SMOKE/train" \
  --seed 0 --run-id CPU_RECHECK --device cpu --batch-size 32 \
  --updates 20 --save-every 10
"$PY" "$DREAMER_DEV/scripts/fg_s3a.py" audit --run "$SMOKE/train"

# 复核本次已保存的真实 10+10 vs 20 收据，不再执行训练。
"$PY" artifacts/fg_s3a_v1/development_20261011/D4_verify_cpu.py
```

### 以后完成源模型后的命令（本轮未执行）

**先确认重训最终产物符合下面的 profile / budget / core hashes；若 gate 修订改变源身份，先审阅并对齐开发源码，重新测试，而非删除 identity 检查。** 当前严格支持任务书的完成版 C1_M1_1M。另一个 10K→online 实验不能直接冒充本源模型。

```bash
# 延用上面的路径与 CPU 环境。先仅做源身份、文件完整性检查。
SOURCE_CHECKPOINT=/data/logdir/fg_single3/gate_margin_v2/C1_M1_1M/ckpt/step_000000062500
S3A_RUN_ROOT=/data/logdir/fg_single3/s3a_v1/C1_M1_actor_bc_new
"$PY" "$DREAMER_DEV/scripts/fg_s3a.py" features --dry-run \
  --labels "$LABELS" --source-checkpoint "$SOURCE_CHECKPOINT" \
  --output "$S3A_RUN_ROOT/features" --seed 0 --run-id S3A_C1

# 待用户后续授权及 GPU 资源确认后执行；本轮不执行以下 GPU 命令。
unset JAX_PLATFORMS CUDA_VISIBLE_DEVICES
"$PY" "$DREAMER_DEV/scripts/fg_s3a.py" features --device gpu \
  --labels "$LABELS" --source-checkpoint "$SOURCE_CHECKPOINT" \
  --output "$S3A_RUN_ROOT/features" --seed 0 --run-id S3A_C1 --shard-rows 1024
"$PY" "$DREAMER_DEV/scripts/fg_s3a.py" train --device gpu \
  --labels "$LABELS" --source-checkpoint "$SOURCE_CHECKPOINT" \
  --features "$S3A_RUN_ROOT/features" --output "$S3A_RUN_ROOT/pilot" \
  --seed 0 --run-id S3A_C1 --batch-size 256 --learning-rate 4e-5 \
  --updates 1000 --save-every 100
"$PY" "$DREAMER_DEV/scripts/fg_s3a.py" audit --run "$S3A_RUN_ROOT/pilot"
# 恢复时同一 train 命令增加 --resume；--updates 是总目标，不是追加更新数。
```

正式 `features` 不允许 `--max-rows`；完整提取以后只能使用对应源哈希的缓存。eval 默认固定 seed=0，后续 A/B 与基线必须使用同一 DEV 和 seed；非零 eval seed 的 CLI 参数目前不传入 evaluator，不能声称已经验证多 seed DEV。

## Risks / not verified

**工程诊断成功，模仿效果尚未成立。** 真实小切片 20 更新为 `ALL_WAIT_COLLAPSE`：ADD recall=0 / precision=0 / F1=0，balanced accuracy=0.5，accuracy=98.430493% 与 Always-WAIT 相同；PR-AP=0.0244536，该切片正例先验=0.0156951。没有据此改变权重或采样规则。

未验证完整生产参数 payload 加载、完整 402k 行生产模型 features、GPU JIT / sharding / 显存吞吐、正式 BC 收敛或 DEV 收益。v1 仅支持单设备。Oracle 后验教师历史不等于部署自主历史；分类工程通过不证明学会择时。没有访问 sealed Test，没有开发 S3-B / S3-C。

实际剩余阻碍：

1. 任务书要求的静态完成版 C1_M1_1M Checkpoint / summary 尚未核实存在。
2. 生产 gate / 阶段迁移代码已变；重训若采用新代码或改 profile/budget，其完成产物不能直接通过当前旧基线 core identity，需要按真实产物做兼容审阅与针对性重测。
3. 正式 GPU BC 和独立 DEV 还需要后续执行授权与实测验收。

## Next recommended step

先由 C1 训练任务产生真实完成收据及静态源 Checkpoint；随后运行 `features --dry-run` 审核新源身份。通过后，在新的 S3-A 目录提取完整教师 features，进行 1000 更新 pilot，再做 Original vs BC 的同一独立 DEV 自由闭环对照。

**回答最终问题：加载完成原模型并只训练 Actor 的路径已经实现且在 CPU 原神经 Agent 上验证；当前旧 C1_M1 元数据也匹配。但尚无完成的正式源模型实载证明，且新 gate 重训版本仍需身份对齐，因此不能宣称现在已可直接对任何重训产物开始正式 BC。**
