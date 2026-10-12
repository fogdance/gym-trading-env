# S3 配置、准备快照与启动管理

## Summary

新增统一管理入口 `tools/fg_s3_experiment.py`，保留现有 S3-A / S3-B 的独立训练器、
优化器和 Checkpoint 格式。集中 YAML 位于 `configs/experiments/fg_single3/`，
覆盖原生 S3-A 和 FULL / MATCHED / CERTIFIED 三套公平对照。
公共模板引用用户指定的 `dreamerv3/dreamerv3/configs.yaml`。
本轮只开发管理代码并做 CPU 接线验证，未启动正式 S3 GPU 训练。

## Evidence observed

开发基线：Gym `926ac70`，Dreamer `fd20e28`，均位于 `feat/fg-s3b-certified-bc`。
已读取合并后的 `tools/fg_single3_experiment.py`、实验管理模块、公共模板解析器、
S3-A / S3-B 训练入口、源准入、特征缓存和 Checkpoint 实现。

既有 Single3 管理通过 `--config` 选择集中配置，串联离线预训练和在线训练。
既有 S3-A 使用 CLI 参数；既有 S3-B 的 YAML 位于 Dreamer 的 `configs/fg_s3b/`，
两者此前均缺少独立的准备快照和运行锁层。

## Root cause

这是管理接口补全。分散的 CLI 参数和配置无法通过一个准备收据固定 S3 实验的
有效参数、模板、输入及运行位置。新增管理层复用既有计算接口，不改监督目标。

## Changes made

| 文件 | 职责 |
|---|---|
| `tools/fg_s3_experiment.py` | `plan / prepare / start / resume / audit` 与受控子进程启动 |
| `src/gym_trading_env/research/fg_s3_experiment.py` | 配置解析、路径检查、源/数据准入、快照、锁、训练器分派及恢复核对 |
| `configs/experiments/fg_single3/S3A_ORACLE_BC.yaml` | 原生 S3-A，全部合法空仓样本 |
| `configs/experiments/fg_single3/S3A_FULL.yaml` | 公平对照 FULL |
| `configs/experiments/fg_single3/S3A_MATCHED.yaml` | 公平对照 MATCHED |
| `configs/experiments/fg_single3/S3B_CERTIFIED.yaml` | 公平对照 CERTIFIED |

S3 管理串联“冻结教师特征生成/复用 → 原 Actor-only BC”。特征任务完成后进程退出，
训练任务在独立子进程启动。新实验拒绝已有输出目录；同一 logdir 的启动/恢复使用
排他锁，新特征生成也使用独立锁。只读已有完整缓存时无需写缓存目录。
锁句柄传递给实际子进程，即使父启动器退出，仍阻止其他进程并发写入。

公共模板按原 `read_config_yaml` 和 `elements.Config` 解析并复用 presets；网络
字段必须与源配置一致。模型加载继续由既有 `validate_source / load_source` 执行。
独立 BC 优化器继承公共模板的兼容默认项，使用既有 ActorBC 常数 LR、无预热
约束。所有最终优化器参数写入有效配置。未新增网络层或 RL+BC 联合 Loss。

四套 YAML 使用独立的 prepared / logdir；三套公平对照具有同一数据、权重、
冻结特征、seed、batch_size、optimizer 和预算，差异来自已有模式接口。

目录示例：

```text
artifacts/fg_s3_v1/experiments/S3B_CERTIFIED_seed0/
  experiment.requested.yaml  # 原请求字节
  model.template.yaml       # 公共模板字节
  model.defaults.yaml       # 已解析公共默认配置
  source.config.yaml        # 精确源模型配置
  effective_config.yaml     # 解析路径及完整 BC 优化器配置
  input_hashes.json / code_hashes.json
  manifest.json / done      # 最后发布准备收据
  features_binding.json     # start 后绑定实际冻结 feature manifest
  attempts/*.json           # 具体 argv、stdout 哈希及退出码

artifacts/fg_s3_v1/features/C1_M1_1M_seed0/   # 只读复用，独立于 prepared
/data/logdir/fg_single3/s3/S3B_CERTIFIED_seed0/
  run_identity.json / ckpt/ / metrics_attempt_*.jsonl
  training_receipt.json
  managed_snapshot/         # 请求/有效/模型配置、准备收据、哈希和特征绑定
/data/logdir/fg_single3/s3/launch_logs/       # 按实验/阶段/attempt 保存 stdout
```

准备收据先完整核对输入、公共模板和实际代码，再原子发布；失败不会发布 `done`
或准备 PASS。运行 ID 绑定准备收据摘要，S3-B 还绑定有效配置文件哈希。
恢复重新核对快照、源、代码/包版本、特征及真实 Checkpoint，随后调用各自原恢复
逻辑。若启动器在复制运行快照前退出，只有真实 Checkpoint 与当前准备身份一致
时才允许从已有不可变准备快照补全副本。`audit` 核对状态，不补写快照或重签输入。

## Verification performed

新增 `tests/entry_v1/test_fg_s3_experiment.py` 使用现有 S3-B 原神经 Agent 诊断
fixture，在 CPU 上验证两个训练器各自的 prepare → start → resume。
原生 S3-A 实际生成新特征再训练；S3-B 实际复用原冻结特征并加权更新。
每个案例仅 2 次 BC 更新，恢复保持同一 Checkpoint 内容；非 Actor 与原优化器
继续由原冻结审计检查。测试还覆盖运行快照缺失后的校验恢复、中央 YAML 变更、
源目录重叠、模板网络不匹配、准备重复、快照篡改、正式部分特征禁用和对照配置一致。
接线及配置检查 **4 passed in 67.24s**。随后仅完善子进程锁句柄传递，单独验证
实际 Popen 和 fcntl 的锁交接：**1 passed in 1.35s**，没有发送信号或终止进程。
共 5 个新增案例通过；最后的锁补丁按其公共 I/O 契约测试，没有再运行神经训练。

最小验证命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/v/Documents/work/gym-trading-env2/src:/home/v/Documents/work/dreamerv3-dev \
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
MPLCONFIGDIR=/tmp/fg_s3b_mpl \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -p no:cacheprovider \
  tests/entry_v1/test_fg_s3_experiment.py \
  --junitxml=/tmp/fg_s3_experiment_management_tests_20261012_delivery.xml
```

最终测试结果、CLI 解析结果及原有文件哈希核对收据随交付保存于
`artifacts/fg_s3_v1/management_20261012/`。既有 S3-A / S3-B 全套测试未重复运行。
四份集中配置的实际 `plan` 均返回 `CONFIG_ONLY_NOT_ADMITTED`；新 Python 文件 AST
解析通过，原有 34 个 S3-A / S3-B Python / YAML 与本轮基线逐文件字节一致。

## Risks / not verified

正式配置中的源路径是任务书要求的完成 C1_M1_1M 路径，需要后续根据真实产物
核对；本轮没有将诊断源或不同训练阶段的 Checkpoint 当作正式完成源。
正式 GPU、并发 GPU 容量、DEV、封存 TEST 和 S3-C 均未执行。
若特征生成中断留下不完整目录，仍按既有缓存检查拒绝使用；保留失败目录后在
新配置中指定新缓存路径，不覆盖或自动放宽完整性检查。

## Next recommended step

在 Gym 开发仓库运行：

```bash
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
$PY tools/fg_s3_experiment.py plan --config S3A_ORACLE_BC
$PY tools/fg_s3_experiment.py plan --config S3B_CERTIFIED
# 后续正式源准入且获准训练后执行；本轮不执行以下 GPU 阶段。
$PY tools/fg_s3_experiment.py prepare --config S3B_CERTIFIED
CUDA_VISIBLE_DEVICES=0 $PY tools/fg_s3_experiment.py start --config S3B_CERTIFIED
CUDA_VISIBLE_DEVICES=0 $PY tools/fg_s3_experiment.py resume --config S3B_CERTIFIED
$PY tools/fg_s3_experiment.py audit --config S3B_CERTIFIED
# 中央 YAML 后续编辑后，明确恢复旧的保存配置：
CUDA_VISIBLE_DEVICES=0 $PY tools/fg_s3_experiment.py resume \
  --prepared artifacts/fg_s3_v1/experiments/S3B_CERTIFIED_seed0
```

每个实验一份 YAML、独立日志目录。S3-A / S3-B 计算入口继续分开，后续 S3-C
通过新适配器接入该管理层；本轮不实现其计算逻辑。
