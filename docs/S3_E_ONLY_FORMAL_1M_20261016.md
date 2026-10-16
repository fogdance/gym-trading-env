# E-only 正式 1M 准备与验证（2026-10-16）

## 范围与独立工作树

本次只准备 E 编码器与原 Dreamer RL。A2 BC、Tutor、B2、Quality 的教师、
Loss、预测头与动作融合全部关闭，不启动这些实验。GPU 正式训练由用户执行。

开发使用 `fg-s3-e-only-dev-20261016/{gym-trading-env,dreamerv3}`，分支为
`feat/s3-e-only-1m-20261016`。起点分别是已验收的 `integration/s3-next`：

- Gym：`713a70f37a6dc1d0800cc9c9cf239752848d5b75`。
- Dreamer：`6114b557e5cc5a134a23ae5b446b351e4f95cc0a`。

正式训练使用另一个 detached 冻结兄弟工作树
`fg-s3-e-only-train-20261016/{gym-trading-env,dreamerv3}`。最终两仓 SHA、
annotated Tag、clean 状态、准备收据及哈希记录在该目录的 `training-lock.json`。
E-only 模型改动保留在独立功能分支；不修改旧正式训练工作树、Checkpoint 或工件。

其间按用户要求，先独立合入 Gate 测试修复及版本化 AGENTS 规则；当前继承的
已验收集成基线为 Gym `6c78d7c68a751347abdfa1bb213848861ee61be1`、Dreamer
`3264c7ed43ec4ba4ba358d3e650ff3ce54154c73`。运行源码231个文件与原集成版本
字节相同。E-only 的模型与训练改动仍在本独立功能分支，不随规则合并进入集成。

## 实测问题与实现

旧配置解析器不接受 `teaching: none`；旧 Collector 没有关闭教师查询的参数。
新增回归在修改前分别失败于 `Unknown teaching` 与 `unexpected keyword argument
'teacher_enabled'`。原 A2 的 `bc_lambda: 0` 只执行 WM 路径，Actor/Critic 保持冻结，
因此不能用它冒充原 Dreamer RL。

新增 Composition 为 `e_multiscale_1m15m + none`，运行身份为 `mode: e`，
`base_oracle_bc/use_tutor/use_b2` 均为 false。该模式复用现有 C2 的原 Dreamer
RL 分支及优化器边界，使用原 `Agent.loss`，没有复制或设计新的 RL 算法。
配置层和模型层强制零 BC；配置层还拒绝非零 beta、beta schedule 及任何
Tutor/B2/Quality 设置。旧模式保持原路径。

Collector 跳过实际 Oracle 查询，发布明确的 `TEACHING_DISABLED` 无效标签，
沿用真实 Observation/Action/stepid 的 Replay 所属关系检查。原 Oracle 与
TeacherCatalog 收据仅用于现有数据、Reward 和来源绑定；其缓存仍由现有准备
及加载过程读取和认证，不将“没有标签查询”描述为“没有读取任何历史 Oracle 文件”。
这些记录既不接管动作，也不进入 BC 或任何辅助监督。

## 正式配置

配置：`configs/experiments/fg_s3_v2/E_ONLY_MARKET_MINUTE_FORMAL_1M.yaml`。

| 项目 | 冻结值 |
| --- | --- |
| Encoder / 教学 | `e_multiscale_1m15m` / `none` |
| 初始化 | `direct_init`，seed0，canonical `size50m`；不继承旧模型 |
| TRAIN | 原始白名单全部 18 个合约；现有已认证 E 数据 |
| 预算 | 1,000,000 次真实 Env 行情推进；reset 元数据不计数 |
| 更新 | 每 100 次推进做 4 次联合更新，共 40,000 次 |
| Replay / batch | 32,768 / 1；有效序列64，因果前缀1 |
| Checkpoint | 每 100,000 Env 步 FullCheckpoint，保留全部 |
| beta / BC | `0.0 / 0.0`，无调度、无 Oracle 接管或标签查询 |
| Clock / Reward | `S3_MARKET_MINUTE` / 原 C1；不启用 Quality TP Reward |
| 正式 Run | `/data/logdir/fg_single3/s3_v2/E_ONLY_MARKET_MINUTE_FORMAL_1M_v1_20261016` |

原 Single3 可用动作 Gate 完整保留：初始 CLOSED，分类阈值0.30，最少25,000训练
样本、ADD支持至少16、原 accuracy/FPR/fallback/recall门槛、连续20个通过窗口才能
开启。开启前 WM/Availability 学习，Actor/Critic 冻结；开启后执行原想象 RL。
连续回归5个窗口停止训练。Gate 使用真实可用动作监督，不依赖 Oracle 交易标签。
正式预算包含 CLOSED 阶段；不能保证 Gate 在预算内开启，也不能把 WM 更新次数
宣称为 Actor/Critic RL 更新次数。应查看 `rl_actor_updates/rl_critic_updates`。

## 验证状态与证据

实际原始 XML、修改前失败、CPU 模型梯度与恢复收据位于
`docs/receipts/s3_e_only_20261016/`。最终选定84个不同测试用例全部通过：
配置与E模型42项、Native采集／恢复19项、共享边界20项、已有教学模式数值回归3项；
后续针对性复测不重复计数。这不是全项目测试的通过声明。
开发中的真实行情诊断已完成2200步、6次更新及独立进程精确恢复，原始结果保留在
开发工作树的 `artifacts/fg_s3_v2/runs/E_ONLY_NATIVE_CPU_v1_20261016/acceptance.json`。
该中间版本验收后仅补充了正式 E-only 来源范围登记及配置错误文案；模型与采集
训练源码相同，最终版本的42项验证及19项采集／恢复验证另行通过。它不是冻结
版本的GPU验收，也不冒充相同源码身份的正式 prepared。最终正式准备独立执行，
收据记录于工作树之外的 `training-lock.json`。诊断为小型CPU网络，正式为size50m。

CPU 小网络受控 READY 验证不是正式 Gate 准入。已测得 Actor 梯度
`0.0097574387`、Critic `1.4586328268`；E minute/context15/fusion 梯度分别为
`321.7689514 / 48.0522270 / 189.8900604`。对应参数实际改变的 key 数分别为
Actor5、Critic2、E分支9/2/2、slow Critic3。CLOSED 下 Actor/Critic 不变，
任意替换教师标签不改变更新结果；恢复完整模型/优化器/RNG 后下一次更新精确一致。

曾复现一项未修改的基线测试失败：
`embodied/tests/test_fg_s3_gate.py::test_audited_ast_accepts_only_the_reviewed_transition`。
它要求历史 `09e2e5c` 与当前 Agent 经 Gate-only 归一化后语义相同；当前 Agent
已包含市场分钟折扣、E 编码器和统一 policy 接口，因此严格审计拒绝该转移。
未修改的集成基线也复现同一错误 `Source network/training semantics changed`。
已按用户要求修正测试合同：历史正例固定两端完整 SHA，当前负例使用当前基准，
新增跨市场分钟语义必须拒绝的回归。共享20项及合并后针对性2项通过，生产 Agent
和 SourceAdmission 审计未改变。根因及原始失败收据见兄弟 Dreamer 的
`docs/S3_NEXT_GATE_AUDIT_ROOT_CAUSE_20261016.md`；旧失败保留，不计为当前失败
或额外通过数。E-only 使用 `direct_init`，不尝试该历史模型转移。

尚未执行：E-only 的 size50m GPU CLOSED/READY 数值检查、真实 Gate 开启、正式1M、
Actor-only 策略效果、W&B 云端发布。CPU 验证及 `PREPARED_NO_TRAINING` 不代表这些
项目通过，也不保证收益或1M可完成全部研究门槛。

## 正式启动与恢复

正式准备由统一入口的 CPU `prepare` 完成，不能用 `admit-e-*` 的诊断 prepared
替代。冻结之后不编辑源码或有效配置。下面只启动这一个 E-only Run：

```bash
cd /home/v/Documents/work/research/fg-s3-e-only-train-20261016/gym-trading-env
env PYTHONDONTWRITEBYTECODE=1 FG_S3_VISIBLE_GPUS=0 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  WANDB_ENTITY=xkm104501-none WANDB_PROJECT=fg-single3-research WANDB_MODE=online \
  /home/v/Documents/work/research/fg-s3-wandb-env/bin/python tools/fg_s3_wandb.py launch \
  --no-evals \
  --run-dir /data/logdir/fg_single3/s3_v2/E_ONLY_MARKET_MINUTE_FORMAL_1M_v1_20261016 \
  --prepared "$PWD/artifacts/fg_s3_v2/prepared/E_ONLY_MARKET_MINUTE_FORMAL_1M_v1_20261016" \
  --trainer-cwd "$PWD" -- \
  /home/v/miniconda3/envs/dreamerv3/bin/python -u tools/fg_s3_v2_experiment.py start \
  --prepared "$PWD/artifacts/fg_s3_v2/prepared/E_ONLY_MARKET_MINUTE_FORMAL_1M_v1_20261016" \
  --authorize-training
```

中断后只将上述原 Trainer 命令的 `start` 改成 `resume`，继续使用相同冻结
prepared、Run 路径、完整 Checkpoint 与 W&B Run 身份。不要重新 prepare 原目录，
不要在已存在 Run 上再次 `start`。

W&B 复用现有 Tracker，项目固定 `fg-single3-research`，family自动为 `s3-e`；
代理与断网回退只作用于 Tracker，`--no-evals` 排除例行评价。提交命令不等于代理
已经执行训练或检查了云端连接。启动前的磁盘实测 `/dev/sdd2` 可用约15GiB、
`/dev/sda1` 可用约97GiB；正式工件全部进入 `/data/logdir`，不使用其他盘。
