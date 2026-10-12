# S3-B 实施与 CPU 验证报告

范围：B0–B4 独立代码开发、实际 TRAIN 数据证明与 CPU 接线验收。
S3-A 按已完成依赖处理；Replay 正在训练不阻碍本轮开发交付。
正式 S3 GPU 训练、DEV、S3-C 均 `NOT_RUN`；封存 TEST `NOT_ACCESSED`。

基准与分支：Gym `97551fad6f512a571bd3028777cff18c01c5ab91`；Dreamer
`8952315be94ed543af1b2bcb73e211cba5077fbc`；各自从当前分支创建
`feat/fg-s3b-certified-bc`。最终提交信息见本文末尾。

## 结果与证据层级

| 项目 | 当前结论 | 证据 |
|---|---|---|
| 价值算法与账户资金证明 | 条件证明及当前完整 TRAIN 数据验证通过 | `B1_B2_final.xml`、`B4_actual_forks.json`、`B4_delivery_audit.json` |
| S3-B 权重数据 | 完整索引、396,542 个严格证书；覆盖/ESS 门槛通过 | `values_full_v2`、`weights_full_v1` 的 manifest/audit |
| 原 Actor 接线与隔离 | 真实原神经网络 CPU 验证通过；非 Actor 和原 opt 未变 | `B3_final.xml`、`actor_cpu_interface_v2` |
| DEV 择时改善 | 未验证 | 本轮无正式训练或 DEV 测试 |
| 开发仓库原有文件保护 | 原有源码/配置字节不变 | `B4_protection_drift.json` |
| 生产仓库全局保护哈希 | **DRIFT_OBSERVED**，保留失败收据 | 同一收据；没有把并发字节变化重签为 PASS |

所有收据在 Gym 仓库 `artifacts/fg_s3b_v1/development_20261012/`。
第一轮进度消息误写了 57 个实际分叉；实际收据是 **55 = 18 + 18 + 19**。

## B0：实际源码与 S3-A 接口

完整阅读任务书，核对当前两仓库 HEAD、S3-A 标签/特征/Actor/恢复接口与既有
C1 Oracle/Replay 收据。旧任务书 gy4/d33 SHA 没有被当作现场 HEAD。
本轮只增加 S3-B 命名空间中的代码、配置、测试和收据；没有编辑 S3-A 文件、
原 `agent.py`/RSSM/优化器/训练入口、原配置或原实验目录。

原 Oracle `funds_certificate` 明确只覆盖经支配性剪枝后的 TP-only 路径，且
`arbitrary_losing_paths_funds_irrelevant=False`。`CandidateIndex` 的账户是 fresh reset，
不能作为亏损后实际账户。完整证明设计见 `FG_S3B_Value_Proof_Design.md`。

沿用完成的 S3-A 全量合法空仓索引，402,096 行，ADD 16,171，WAIT 385,925。
当前行是 outgoing 标签，反馈是上一动作；用当前 `lots_after` 而非 `lots_before`。
本轮不重生成教师标签，不运行真实 Oracle O 候选搜索，不新增网络层。

## B1：新条件资金引理与区间 DP

新引理针对**实际 P4 空仓现金 C、一次强制交易、然后 TP-only 最优续行**。
成本恒定、原 Single3 规则不变，扩张的 TRAIN 报价范围为 `[lo,hi]`：

```text
Lq = 3*M*(hi-lo) + 3*(open_fee + max(close_fee, close_today_fee))
B  = Lq + 3*M*hi*margin_rate
C > B                          -> 一次强制分支资金不改变原候选执行
C' = C + 实际 net_trade_pnl > B -> TP-only relaxed 续行可执行
```

P4 当前空仓现金由原生 Sidecar 已完成交易的 **Decimal 净现金**累积重建；核对
每次选中成交、出场归因、C1 float32 反馈及最终 `account_state.json` 账本。
18 合约最低监督现金均为 300,000；最保守 B 为 104,358。收据还逐合约验证
`min_C - Lq > B`，足以覆盖当前监督状态的一次强制亏损后续行。
这是新条件引理，**没有**宣称任意连续亏损账户资金无关。

低资金合成反例：现金 8,000，真实 SL 完成后不足以成交后续更高价格的 TP。
默认返回未知 QADD；实际分叉验证后仅用“局部收益 + 一直 WAIT”下界和
“局部收益 + relaxed suffix”上界认证 WAIT，不能标 `EXACT_Q`。
续行验证检查原 Env 的实际成交与 TP 正现金，不能把请求 ADD 当作已成交。

从冻结候选一次反向计算 suffix 区间与可行 witness，gamma 为 `Fraction(332,333)`，
奖励系数为原 float32。局部指数 `step-t-1`，续行指数 `u-t`；1024 reset
不增加折扣步。所有运算向外舍入，区间相交时限定 Fraction 预算：每合约
4,096 次比较、每次最多 8,192 事件，耗尽不以 epsilon 裁决。

```text
D = [A_lo - W_hi, A_hi - W_lo]
m = D_lo (D_lo > 0)；-D_hi (D_hi < 0)；否则 0
```

不输出区间中点。`EXACT_Q` 表示真实资金可行 witness 达到 relaxed 最优值；
数值仍是舍入误差包围区间。严格并列通过 Fraction/拒单状态等价证明。
重叠、未知、未释放仓位与删失分开；未知 API 值为 `None`，NumPy 存储以
`has_intervals=False` 明确标记不可使用的占位字段。
证书方向与原 P4 教师冲突直接停止合约发布，不改动作或放宽条件。

| 全量证书 | 行数 | 对权重的影响 |
|---|---:|---|
| ADD_BETTER | 16,171 | 正权重，原标签 ADD |
| WAIT_BETTER | 380,371 | 正权重，原标签 WAIT |
| TIE_EXACT | 5,554 | 原行保留，权重 0 |
| UNCERTIFIED / CENSORED | 0 / 0 | 本数据均满足条件；合成反例仍显式覆盖 |

严格证书覆盖率 **98.6187378%**。完整值索引为 `values_full_v2`；更早开发版
`values_full_v1`、初始 benchmark 为历史收据，代码身份变动后不能作为当前 cache。

真实账户分叉：FG001/FG201/FG809 各重放 4,097 个教师推进步，分别 18/18/19
次完整实际账户 ADD/WAIT 分叉；按年份、方向、TP/SL、早期风险、双触、reset
分组并固定种子 202。超过 2,048 局部推进步的候选未纳入该次抽样，分别
27/0/210 条；这些不是全量证书排除行。没有把抽样验证称为全量账户穷举。
短合成序列独立穷举原 Env 的完整账户状态和所有合法动作，不用同一 DP 作金标准。

## B2：数据索引与权重

逐行核对 contract/trace、ordinal/cursor/stepid、原 label、真实 mask 与顺序；
全量原索引保持 402,096 行。训练接入另核对相同 S3-A feature manifest、
每个 shard 的完整 index 及源模型参数指纹；Q/现金/出场只用于权重，不进网络输入。

TRAIN 已证明正边际的全局 P90 = **3.7523039430204257**，
`w=min(3,margin/P90)`；没有重采样、逐合约归一、类别再均衡或 DEV 超参拟合。

| 标签 | 正权重人数 | sum_w | ESS | w P10 / P50 / P90 / P99 |
|---|---:|---:|---:|---|
| ADD | 16,171 | 3,005.060663 | 12,437.318688 | 0.005032 / 0.243061 / 0.262134 / 0.262134 |
| WAIT | 380,371 | 155,496.676877 | 180,882.127087 | 0.016371 / 0.253083 / 1.012189 / 1.854067 |
| 总计 | 396,542 | 158,501.737541 | 186,925.661976 | 0.015509 / 0.250976 / 0.993918 / 1.838507 |

表中 WAIT 分位数包含保留的 5,554 个零权重行。ADD 的权重贡献 **1.896%**，
低于原标签人数占比 4.022%，仍有严重多数 WAIT 的学习风险；未因此改归一化。
最大合约贡献 FG201 **9.8864%**；88 行到达 cap，占全索引 **0.0218853%**。
`audit.json` 保存每合约、年度、方向、session 的 margin/weight 分位数、零比例、
证书与 ESS。ESS 是加权代数指标，不是独立市场样本数。

初始可配置工程门槛：总 ESS >= 2,560，ADD/WAIT 各 ESS >= 32；零类别或不满足
则 `INSUFFICIENT_SUPERVISION_COVERAGE`；单合约贡献 >50% 标 `WEIGHT_CONCENTRATION`。
本数据在这些明确门槛下 PASS；这不是模型择时有效性的证明。

## B3：加权 Actor、实验身份与断点恢复

新 `WeightedActorBC` 继承已完成的 S3-A `ActorBC`，复用原 pol、MaskedCategorical
和已有纯 NLL 内核，只扩展非二值权重验证。`pol/*` 是唯一可微参数实参。
独立实例的 Optax 状态对外命名 `s3b_bc_opt`；内部沿用基类状态访问，没有读取
任何 S3-A BC checkpoint。全零 batch 跳过且不累计更新；非有限权重/更新拒绝发布。
仅对整个 batch 同乘共同正尺度避免 float32 下溢，Loss 分子/分母同比例变化，
不改变全局目标；记录 raw 分母、尺度、类别 sum、ESS 与单一标签批次。

| 模式 | 样本 | 权重 |
|---|---|---|
| S3A_FULL | 全部原合法行 | 1 |
| S3A_MATCHED | 396,542 个严格证书行 | 1 |
| S3B_CERTIFIED | 精确同一批严格证书行 | 冻结全局权重 |

MATCHED/CERTIFIED 使用同 seed、同 permutation sampler、逐 batch 同 index；
三套配置除 mode 外完全相同：batch 256，seed 0，1,000 updates，LR 4e-5，AGC .3。
不同 mode、source/features/labels/values/weights/scale/subset、代码、包、优化器
任一身份不同，恢复拒绝。独立协议 `FG_S3B_BC_V1`；S3-A checkpoint 相互拒绝。
保存原始 typed Optax PyTree、sampler RNG/order/cursor/epoch、更新/跳过计数；
pending 目录成功后原子发布。整个原源实验目录（含日志和所有 checkpoint）
禁止作为 S3-B 输出目录，默认拒绝覆盖。

CPU 验证使用**真实原始神经 Agent**，同时复用 S3-A 真实 P4 的 1,100 行前缀缓存：
892 合法特征，14 ADD、878 WAIT，宽度 40。该小架构源明确为随机原 Agent 诊断，
没有把它称为已训练的生产源模型或使用假 Agent。
每模式最多 6 次 CPU 参数更新仅用于接口验收；不属于正式 S3 研究训练。

6 连续更新与 3+3 恢复：所有模型参数、独立 optimizer、sampler 和每次指标逐字节
一致；MATCHED/CERTIFIED 六个 batch 的 index hash 一致。
208 个状态键中仅 5 个 `pol/*` 改变，203 个冻结；原 joint updates/batches 均 0。
另有原 joint optimizer/Norm/Gate 非零状态测试，证明没有偷偷重置源状态。
推理同步验证重新重建 32 个真实教师历史 latent：latent 最大误差 0，
新 policy 与已更新 Actor 概率最大误差 `5.960464477539063e-08`，相对旧 Actor
最大概率变化 `0.0010491013526916504`。

诊断 CERTIFIED 结果：precision .02222，recall .14286，F1 .03846，AP .02826，
balanced accuracy .52131；accuracy .88789，AlwaysWAIT .98430。
这是接口与冻结验收，**不能报告 DEV 择时改善**。

## B4：测试、性能与自审

16 项新增必要 CPU 回归通过：`B1_B2_final.xml` 10 项、
`B2_future_identity_final.xml` 1 项、`B3_final.xml` 5 项。
复用既有 S3-A 测试收据，没有重跑其完整 Python 测试集。

| 要求 | 最小验证 |
|---|---|
| 多 TP、SL 后资金不足/仍充足 | 原 Env 全账户穷举；低现金 bounds；实际续行成交检查 |
| 拒单、未完成、主力退出、pending SL、双触 | 原 Env 分叉与 fail-closed 单元回归 |
| γ/release/行政时钟、正负反馈组合 | 稀疏 float32 精算；chunk 测试；3 个真实 4097 步前缀 |
| 重叠/并列/预算耗尽/教师冲突 | 区间与 Fraction 回归；禁止 epsilon |
| hash/版本/TEST/partial 混入 | 数据/权重 mutation 拒绝；训练 code/package/cache identity 校验 |
| 未来隔离 | 修改未来 OHLC 改变证书但过去 RSSM 完全不变；旧源哈希立即失效 |
| ones/解析梯度/零与极小权重 | 原 S3-A byte-equivalence、含 unimix 的 2 样本解析导数 |
| 其他参数/原 opt/Norm/Gate | 全参数 hash、非零源状态、推理同步 |
| checkpoint 隔离与恢复 | 新协议、源整目录拒写、断点与连续轨迹一致、外来身份拒绝 |

最终版单合约 512/4096/8192：17.40 / 17.45 / 17.68 秒；完整 suffix DP
0.230 / 0.221 / 0.232 秒。18 合约处理耗时合计 316.02 秒，进程 peak RSS
720,654,336 bytes（约 687 MiB）。每次均完整读取冻结候选并对账整个合约，prefix 只限制
证据输出，所以不能将这些耗时除行数外推全量逐行成本。RSS/全18合约耗时详见
`B4_benchmarks.json`、`B4_delivery_audit.json`。

自审发现并修复：真实续行必须检查实际成交；推理 checkpoint 审计避免重复全量
hash 的二次复杂度；subset 越界拒绝；原源实验整目录保护。相关回归均通过。
历史数据/CPU接口版目录保留，没有删除或重签使其冒充新版本。

## 隔离收据与未来运行条件

737 个基准源码/配置检查中，生产路径观察到 6 项字节漂移：
Gym 的 `C1_M1_10K_then_online_1M.yaml`、`fg_single3_experiment.py`；Dreamer 的
`test_action_mask.py`、`outs.py`、`train.py`、`test_fg_add_gate_training.py`。
归属与触发者未核实。本轮写入集合均为两个开发仓库中的新 S3-B 文件；开发仓库
原有文件未变；没有写这些生产路径、发送信号、kill、重启或调整 GPU 作业。
最初全局哈希审计因此退出 1；`B4_protection_drift.json` 保留前后哈希及
`DRIFT_OBSERVED`，不把全局保护检查报告为 PASS。
生产 protected 10K checkpoint 的全部文件仍与 producer manifest 及中途固定哈希一致。

S3-A 接口已完成。后续正式运行仍需符合其源 admission：完成且 protected 的
C1_M1_1M checkpoint、同版本完整冻结 feature cache。本轮只读到的 OFFLINE_10K
已完成 10,000 updates，且 snapshot 中 4 个 source code hash 与当前开发基准不匹配。
没有加载该生产 agent payload，也没有为了接入而放宽 S3-A 校验。这是**未来正式
运行的接入条件**，不是本次 S3-B 开发阻塞或 S3-A 尚未完成。

已具备从符合这些身份约束的完成源加载原模型、只监督原 Actor 的代码入口；
若后续采用当前 Replay 流程的新 source identity，应先核对/同步对应源实现并
生成同版本冻结缓存。当前不启动任何正式 S3 GPU 训练。

## 可执行命令

以下环境指定 CPU；原数据/模型目录均只读，output 目录必须不存在。

```bash
cd /home/v/Documents/work/gym-trading-env2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/v/Documents/work/gym-trading-env2/src:/home/v/Documents/work/dreamerv3-dev
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/fg_s3b_mpl
PY=/home/v/miniconda3/envs/dreamerv3/bin/python

# 当前交付数据审计
$PY tools/fg_s3b.py values-audit --output artifacts/fg_s3b_v1/values_full_v2
$PY tools/fg_s3b.py weights-audit --output artifacts/fg_s3b_v1/weights_full_v1

# 在新目录复现完整构建；合约成功段支持同 identity 的 --resume
$PY tools/fg_s3b.py values-build --config configs/experiments/fg_s3b/certified_v1.yaml \
  --output artifacts/fg_s3b_v1/values_reproduce
$PY tools/fg_s3b.py weights-build --config configs/experiments/fg_s3b/certified_v1.yaml \
  --values artifacts/fg_s3b_v1/values_reproduce --output artifacts/fg_s3b_v1/weights_reproduce

$PY -m pytest -p no:cacheprovider tests/entry_v1/test_fg_s3b_advantage.py \
  tests/entry_v1/test_fg_s3b_weights.py -q
cd /home/v/Documents/work/dreamerv3-dev
$PY -m pytest -p no:cacheprovider embodied/tests/test_fg_s3b.py -q
```

独立 CLI：Gym `values-build/values-audit/weights-build/weights-audit`；Dreamer
`train/admission-audit/checkpoint-audit`。接口 CPU 测试完整命令在
`B4_actor_cli_commands_v2.json`；复现脚本在同目录 `b4_actor_cpu.py`。

后续 source 到位时先审计（本轮不执行正式训练）：

```bash
cd /home/v/Documents/work/dreamerv3-dev
# 设置为已核对版本的完整源及其完整 S3-A feature cache
SRC=/path/to/completed/C1_M1_1M/ckpt/step_000000062500
FEAT=/path/to/complete/S3A/features
$PY scripts/fg_s3b.py admission-audit --config configs/fg_s3b/S3B_CERTIFIED.yaml \
  --source-checkpoint "$SRC" --feature-cache "$FEAT"

# B5 后续运行配置；本轮 NOT_RUN。三模式顺序运行，各用新目录
$PY scripts/fg_s3b.py train --config configs/fg_s3b/S3A_FULL.yaml \
  --source-checkpoint "$SRC" --feature-cache "$FEAT" --device gpu \
  --output /data/logdir/fg_single3/s3b/S3A_FULL_seed0
$PY scripts/fg_s3b.py train --config configs/fg_s3b/S3A_MATCHED.yaml \
  --source-checkpoint "$SRC" --feature-cache "$FEAT" --device gpu \
  --output /data/logdir/fg_single3/s3b/S3A_MATCHED_seed0
$PY scripts/fg_s3b.py train --config configs/fg_s3b/S3B_CERTIFIED.yaml \
  --source-checkpoint "$SRC" --feature-cache "$FEAT" --device gpu \
  --output /data/logdir/fg_single3/s3b/S3B_CERTIFIED_seed0
# 恢复同目录、同 config/source/features/weights 身份：原命令增加 --resume
```

B5 命令需在新终端设置对应 GPU 可见设备；不要沿用上面 CPU 环境中的空
`CUDA_VISIBLE_DEVICES`。本轮没有执行这些命令。

## 新文件与交付位置

Gym：`fg_s3b_advantage.py`、`fg_s3b_weights.py`、`fg_s3b_verify.py`、
`tools/fg_s3b.py`、两份 `test_fg_s3b_*.py`、`configs/experiments/fg_s3b/certified_v1.yaml`。
Dreamer：`dreamerv3/fg_s3b_bc.py`、`embodied/run/fg_s3b_train.py`、
`fg_s3b_checkpoint.py`、`scripts/fg_s3b.py`、`embodied/tests/test_fg_s3b.py`、三份对照 yaml。

完整本地数据：Gym `artifacts/fg_s3b_v1/values_full_v2/` 与 `weights_full_v1/`，
包含原索引、值/证书、全局匹配索引、完整权重和校准；大 NumPy 文件留本地，
Git 提交代码、metadata、可复现脚本与测试收据。

最终源码提交与文档校验收据在收尾时填入，进度在证明设计文档同步。
