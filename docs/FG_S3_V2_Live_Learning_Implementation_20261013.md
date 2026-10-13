# S3-V2 DAgger：真实在线学习第一批实施结果

日期：2026-10-13。依据 D0 后执行修订，当前工程结论 **GO_WITH_LIMITS**。
第一项关键交付已完成：真实 Replay 来源模型、原 FG TRAIN Env、原 Oracle，
连续 collect→WM/Actor BC→collect；同时完成基本 Actor-only DEV、实用恢复和 C2 接线。
策略效果尚未达成；正式1M准备尚未完成，不宣称完整详细方案全部验收。

## 代码与运行身份

- 开发分支 `feat/fg-s3-v2-dagger`，双仓位于 `research/fg-s3-dev/`。
- 本批执行源码：Gym `f9e050126d6636cd884f70a3090e71c8fbbd46e0`、
  Dreamer `dc6cca7fbbba9f663d70239432322058d5695d0d`。
  后续本批报告提交仅增加配置、测试及收据；各 preparation 保留启动时的真实 HEAD/dirty。
- 新产物：`artifacts/fg_s3_v2/{prepared,runs,checks,receipts}/`。所有 Run 独立，旧输入/Checkpoint 未覆盖。
- 原模型：冻结 Online180k，42,584,132个网络参数，模型更新21183，source环境180000；
  source manifest SHA `4979acfb0272ffbc3c96833e5b77fccd66c5d4d0f335bfca48810f764d927a57`。
  新 S3 环境计数从0开始，模型累计更新保留；shared SourceAdmission 显式0.50→0.30校准/重置Gate。
- TRAIN/Oracle 复用用户建立的只读 `research/fg-data` 链接，Oracle身份
  `03a0d6cccc03f06cb18f032e955bfbbb28d7272ef4de7e336983e174690414ac`。
  没有使用其他实验仓库作为源码/导入路径；只复用已授权输入工件。
- 模型仅用 `/home/v/miniconda3/envs/dreamerv3` 的 CUDA:0。
  真实来源保持 bfloat16；受控小模型检查用float32，不混称真实来源验收。

## 实现与监督数学

1. D0 只增加 subclass hooks。V2Collector 使用同一 Oracle/Driver/MixedExecutionPolicy；
   outgoing native row 到下一 Env 反馈完成后才发布。提交/采样共用锁，UUID+offset绑定教师证明，
   未提交行不能训练。物理 Replay 无 teacher/oracle/Q 字段；训练时由原 join 接入。
2. 原 Agent WM 前向原样提取为 `world_model()`；准入把 helper 完整重新展开，再比较原 AST。
   篡改Reward数学或GPU配置flag的负例被拒绝，没有占位符绕过。原V1默认loss不变。
3. A2=`原WM loss + lambda × mean_valid(-log π(Oracle action|当前repfeat,真实mask))`。
   `repfeat`停止BC梯度，BC只更新原Actor。无有效标签时loss和Actor梯度为0，Actor及其动量也冻结。
   原联合优化器每次仅调用一次。没有独立BC优化器或新增网络层。
4. C2 CLOSED走同一WM-only+BC路径；READY才走原imagination/repval/RL及慢目标更新。
   原Single3阈值0.30、连续20窗、Recall/支持量/退化规则不放宽；本批真实Gate均CLOSED。
5. FullCheckpoint复用原协议，保存Agent/联合优化器、Driver/Env/carry、Replay、已提交/待完成教师、
   混合RNG、Gate、stream/Prefetch、训练carry与独立计数。resume加载本Run，不重做来源transfer。
   配置和源码快照校验；未提交update/cycle日志先归档再回到Checkpoint边界。

## 真实 TRAIN 结果

| Run | Env推进 | WM更新 | 有效BC更新 | 认证标签/总步 | Actor ADD提议 | 实际成交 | TP+5 / SL30 | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A2_GPU_FG201_1K_v2，beta=.5 | 1000 | 8 | 2 | 41/1000 | 57 | 48 | 40 / 7 | 4.143995 |
| A2_GPU_FG201_1K_TEACHER_v1，beta=1 | 1000 | 40 | 40 | 870/1000 | 177 | 104 | 104 / 0 | 100.304389 |
| C2_GPU_FG201_500_v1，beta=.5 | 500 | 4 | 1 | 13/500 | 28 | 27 | 21 / 5 | -2.277705 |

均使用FG201 SHORT TRAIN原行情；不是合成行情验收。RL有效更新全部为0。
接管/混合成交的成绩属于教师指导的TRAIN轨迹，不作为Actor自主表现。
复用历史价格的本批步数来自真实Env推进，不称新增独立市场数据。

A2 .5每250步2更新；250步有效batch有8标签/5 ADD，Actor梯度2.867341，WM梯度306.191559。
750步恢复后的有效batch有4标签，Actor梯度8.100271；没有标签的6个batch不更新Actor。
四个周期WM均改变，两个周期Actor改变；模型21183→21191，后续提议版本21183/21185/21187/21189。

beta=1诊断是在观察TRAIN认证域丢失后另建，未根据DEV调参；每100步4更新，10个周期均WM+Actor改变。
40次WM梯度119.152901–390.774811，Actor梯度0.147464–5.883037，Critic梯度全0；
累计2229次有效训练行消费，其中299 ADD（包含重复采样，不冒充独立新行）。模型21183→21223。
新参数持续进入后续Actor提议，10个实际提议版本被记录；不要求来源Actor已学会交易。

未认证原因：A2 .5的892行持仓/不可主动决策，67行资金低于原证书；C2的447/40行同类原因。
这些样本继续交互及WM学习，不用于BC。beta=1的130行不可主动决策，未发生资金域失效。
Q仍为null。原D0证书未扩展到未经证明的亏损账户。

TP+5的日历分钟p50/p90：A2 .5为7.5/37.3；接管A2为2/4；C2为5/35。
早期逆行10事件分别21/0/13。详细原因、控制次数、Replay字段见 `receipts/real_train_metrics.json`。
Oracle-selected实际行分别22/870/5；其余行包含强制WAIT，不能把全步控制比例当主动决策比例。

## 恢复及独立评价

A2 .5实际在500步保存并退出进程，以另一个GPU进程恢复至1000步。
真实Checkpoint检查保留前500行完整身份，以及边界排队动作的cursor/observation/state/action/Oracle证明/RNG抽样/模型版本。
计数4→8更新、500→1000环境步，无重复插入；初始化节点与真实来源transfer参数digest完全一致。
三Run最终Critic/slow/count/Normalizer/对应优化器digest与初始化相等，来源manifest及agent文件未变。
没有执行同种子不中断GPU全参数位级对照，**不声称已证明该层面的精确确定性**。

Actor-only FG409 DEV，各1000真实推进，独立模型实例，eval argmax，不查询Oracle，不读取原Oracle标签。
TRAIN模型/Replay不参与DEV更新；固定现有TRAIN/DEV划分，TEST未访问。

| Checkpoint | 自主ADD/成交 | TP+5 / SL30 | TP分钟p50/p90 | TP有效bar p50 | 早期逆行10 | Reward | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A2 .5，8更新 | 9/9 | 7/1 | 79/804.6 | 73 | 3 | -0.116903 | 有自主ADD，研究目标未达成；1笔未平仓 |
| 接管A2，40更新 | 0/0 | 0/0 | 无 | 无 | 0 | 0 | **ALL_WAIT**，研究效果FAIL |

40更新模型学习了BC但自主评价退化，尚未证明具体成因。不能把接管104次TP解释为Actor成功。
后续可在TRAIN研究采样/接管课程/探索，保留等权BC基准；不依据DEV追调Reward/Gate。

## 实际命令与收据

工作目录为本Gym开发worktree。命令使用指定Conda，无CPU模型回退：

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py gpu_check

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:../dreamerv3 MPLCONFIGDIR=/tmp/fg_s3_v2_mpl \
  /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q tests/entry_v1/test_fg_s3_v2.py -k 'not real_gpu'

/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare --config A2_GPU_FG201_1K
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start --config A2_GPU_FG201_1K --stop-at 500
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py resume --config A2_GPU_FG201_1K
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py evaluate --config A2_GPU_FG201_1K --eval-steps 1000

/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare --config C2_GPU_FG201_500
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start --config C2_GPU_FG201_500

/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare --config A2_GPU_FG201_1K_TEACHER
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start --config A2_GPU_FG201_1K_TEACHER
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py evaluate --config A2_GPU_FG201_1K_TEACHER --eval-steps 1000

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:../dreamerv3 MPLCONFIGDIR=/tmp/fg_s3_v2_mpl \
  /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q tests/entry_v1/test_fg_s3_v2_artifacts.py \
  --junitxml=artifacts/fg_s3_v2/receipts/actual_gpu_artifacts.xml
```

准备/运行目录不允许覆盖；这些历史命令再次start会拒绝已有目录。
重新运行需另建配置名/目录；已完成Run的resume只验证恢复状态，不自动延长冻结预算。

| 检查 | 结果 | 权威收据及实际范围 |
| --- | --- | --- |
| 实时提交、恢复行身份/采样；AST负例；日志尾部归档 | PASS | 3项pytest，原Env受控90bar夹具，非真实行情生产验收 |
| 小模型GPU梯度/停止BC梯度/空BC动量冻结/C2 CLOSED/受控READY/lambda0教师独立 | PASS | `checks/gpu_1beb984b69/receipt.json`，54,468参数、原Env受控行情；9项自动断言，非9项pytest |
| 原42,584,132模型在FG201采集与多轮梯度 | PASS | 三Run `training_receipt.json`、`updates.jsonl`、`cycles.jsonl`及终端log |
| 真实500→1000恢复、来源与完整冻结状态 | PASS | `receipts/actual_checkpoint_verification.json`、`actual_gpu_artifacts.xml`，1项实际工件pytest，无CPU神经计算 |
| 物理Replay无教师、原因/交易速度统计 | PASS | `receipts/real_train_metrics.json`，真实NPZ字段及Env导出状态 |
| 两个Actor-only DEV引擎运行 | PASS | 两Run `evaluations/actor_only_*/receipt.json`，真实FG409；策略效果单独列FAIL/未达成 |
| 真实生产Gate连续20窗/READY RL | NOT_RUN | 本批样本量不足，真实Gate reports=0；不能用受控READY替代 |
| lambda0与完整R1引擎严格等价、不中断位级恢复对照 | NOT_RUN | 仅证明lambda0教师无关；R1后置，恢复按修订做实用续训 |
| 正式1M、100k资源估计、TEST、B2 | NOT_RUN | 未启动，未伪造Q或预算完成 |

全链终端输出存于 `receipts/*_terminal.log`。模型/Replay大文件留在各Run，不加入Git。
失败夹具和日志保留，没有清理旧目录或修改旧manifest。

## GPU错误与最终Review

独立demo在 `research/diagnostics/cuda_graph_probe.py`，无S3/Dreamer/Env导入。
原 `LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:` 混用Conda libcublas与系统libcublasLt；
graph=3复现原错误，graph=0同样初始化失败。仅清除该进程库路径，两组均前向/反向/3更新PASS。
四份收据及命令在 `research/diagnostics/README.md`；未修改驱动、系统或Conda包。
V2入口仅重启自身进程清除库覆盖；显式CUDA继承原GPU flags，shared准入只白名单该精确谓词扩展。

另一个GPU回归确认冻结Critic零值符号在`optax.apply_updates`后发生2001个-0→+0，数值差/梯度为0。
现对inactive参数选择原值，CLOSED字节检查及真实来源冻结检查均通过。早期失败记录保留。

Review覆盖：原算法数学提取、完整网络/联合优化器继承、BC数学与梯度归属、执行动作/prevact、
UUID教师证明/提交边界、无future字段、阈值迁移与同Run恢复、配置冻结和已提交日志、独立评价。
当前没有已观察但未修复的数据/梯度/恢复阻断；All-WAIT和认证域丢失是已披露的实际限制。
原T030训练worktree仍为 Gym `9752438` / Dreamer `00b82f8`，tracked clean；原仓库tracked clean，
原有untracked文件及.bz2未处理。.gitignore未修改，没有正式GPU训练进程由本批启动。

资源只在sdd/sda。读数样本GPU3494MiB/16380MiB（包含桌面），不是峰值测量；
root约72GiB、/data约179GiB可用，不能用1000步耗时外推完整1M生产预算。

剩余工作：正式1M准备/显式授权入口与长期TRAIN/Replay资源管理；按最新修订后置R1/B2；
TRAIN上的持续学习与自主ADD采样/课程诊断。当前仅受限诊断入口，不能冒充完整正式1M启动命令。
