# Tutor-CD Full E2：integration/s3-next正式1M准备

按用户要求使用公共 `integration/s3-next`，不使用旧Tutor功能分支作为来源。
本次仅增加正式配置、说明和独立冻结worktree；不改模型、Loss、Reward、Oracle或训练框架，不执行GPU DEV或正式训练。

## 冻结配置

`configs/experiments/fg_s3_v2/TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M.yaml`：

- 新F0 Schema；`encoder=legacy_1m`、`teaching=tutor_cd`，原Tutor执行路径。
- Natural/Balanced BC、Pair Ranking、E2：alpha0.8、Pair lambda0.1、tau1、kappa1、权重上限2；沿用已实现的Full E2参数，不宣称参数最优。
- beta1固定，不混入独立beta下降实验；Critic/RL冻结。
- ALL_TRAIN原18合约；1M真实环境步，100步收集/4次联合更新，共40000次更新。
- 每100K环境步保存完整Checkpoint；batch1、Replay32768、环境轮转片段1000市场分钟、seed202。
- 使用 `/data/logdir/fg_s3_market_minute/20261013_v2/` 的E/O/TeacherCatalog；原已审计Online180K仅参数warm-start。优化器/Gate/Replay/计数按原warm-start协议新建。

新Run：`/data/logdir/fg_single3/s3_v2/TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014`。
准备目录：冻结Gym的 `artifacts/fg_s3_v2/prepared/TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014`。

双仓从integration实际提交建立独立detached训练工作树：
`/home/v/Documents/work/research/fg-s3-tutor-full-train/{gym-trading-env,dreamerv3}`。
最终完整SHA、同名Tag、配置/输入/Oracle身份、实际CPU准备结果在该工作区的独立 `training-lock.json`。正式启动前此锁必须记录 `PREPARED_NO_TRAINING`、18合约、双仓clean和零新增环境步；未执行的项目不得记PASS。

## 准备、启动、恢复

```bash
cd /home/v/Documents/work/research/fg-s3-tutor-full-train/gym-trading-env
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare \
  --config TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M
```

这只运行CPU准入，检查原行情/教师哈希、资金证明和原DP结果，不初始化GPU模型。
代理完成prepare后，用户直接使用配置目录README的W&B launch命令启动，不重复prepare、不编辑有效快照。训练器保持独立于网络；W&B固定Project为 `fg-single3-research`，使用现有代理和offline fallback。

`launch --no-evals`只自动跟踪正式训练；例行Actor-only结果留在本地，Archive按单独授权处理。同Run恢复只将原启动命令中的 `start` 换为 `resume`，其余身份/路径保持不变。

本次不再重复已经验证的GPU评价或完整模型回归；新增配置使用官方 `read_config_yaml/read_settings` 做实际类型、预算、组合和Tutor参数验证。实际18合约CPU准备收据和命令日志保存在冻结工作区的 `validation/`；既有F0 CPU模型/Checkpoint结果属于继承证据。正式1M仍由用户启动。
