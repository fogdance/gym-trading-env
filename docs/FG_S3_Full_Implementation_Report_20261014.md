# Tutor-CD Full / B2 Full 实施报告（2026-10-14）

本次实现及必要CPU/Offline/只读真实数据验收完成；新版GPU研究仍须另行授权。800K W&B里程碑由独立项目负责，**本次完全排除**，未创建/修改任何正式里程碑或旧云端Run。

## 冻结代码与来源

- 实现代码Gym：`29f7f93c05af9435c4f99756b1c4d175387b0b9e`；Dreamer：`0318e6d64cf0e5c9d790d3220315cb08e9e13734`。本报告之后仅文档提交的Gym最终SHA以交付时Git HEAD为准；最终准备快照记录实际双仓SHA。
- 集成双工作树：`/home/v/Documents/work/research/fg-s3-full-integration-dev/{gym-trading-env,dreamerv3}`；另有独立Tutor Full/B2 Full分支双工作树，通过Git合并共享实现。
- 基线实际Gym `ad6490675bf853a7793763f95aa7ba7f12fa379e` / Dreamer `d76a423dd7930756c3609cc6f7197941f583c160`，已有Tutor E2/B2及W&B。未机械覆盖或重新设计核心。
- 原Tutor冻结Gym `3a6f889adf65583a1ab636f244ca05218f19196f` / Dreamer `c18a8b5917f73110f9b6ce840495f4da1d7ca1d4`；正式Run `/data/logdir/fg_single3/s3_v2/TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M_v1` 全程只读。212个既有源码/小型工件基线hash检查未变；未删除、移动或覆盖大工件。
- 同一已审计Online180k参数来源：`/home/v/Documents/work/research/fg-s3-dev/gym-trading-env/artifacts/fg_s3_arch_v2/online_180k_source/ckpt/step_000000180000`，manifestSHA `4979acfb0272ffbc3c96833e5b77fccd66c5d4d0f335bfca48810f764d927a57`。warm_start保留NN来源、重建优化器和本地更新计数；原来源21183模型更新是provenance。
- 数据沿用可信市场分钟v2：`/data/logdir/fg_s3_market_minute/20261013_v2/{env/v1,oracle/v1,catalog/v1}`。本轮未更改Reward/Oracle数学/时间语义，无需覆盖重建既有可信数据。新DEV反事实结果单独生成且明确DEV_EVALUATION_ONLY。

## 差异完成矩阵

| ID | 状态/证据范围 | 实际处理 |
|---|---|---|
| T01 | DONE_CPU | 认证/两个动作合法：eligible_teacher，非法标签fail-closed测试 |
| T02 | DONE_CPU | Natural+Balanced数学与原静态Loss保持 |
| T03 | DONE_CPU | 同episode Pair、空类/零pair/跨reset测试 |
| T04 | DONE_CPU | 同masked Actor分布、detach RSSM；真实CPU梯度检查 |
| T05 | DONE_CPU | 现有E2保留，权重上限、P/N类内归一化、StopGrad测试 |
| T06 | CONFIGURED_NOT_RUN_GPU | 新Full E2 kappa=1；旧正式kappa=0不修改 |
| T07 | DONE_CPU | 每100更新Actor分项原始/加权梯度；同批实际WM特征；无第二次优化/WM/RNG更新 |
| T08 | RETAINED_CPU_REGRESSION | 原Env→Driver→Replay→WM+BC+Pair；新Full GPU闭环NOT_RUN |
| T09 | RESEARCH_DEFERRED_BY_SCOPE | beta=1；DAgger退火独立后续实验 |
| T10 | IMPLEMENTED_CPU_PARTIAL_RUNTIME | 合法pADD/raw-final margin；DEV posthoc P/N/误报/候选质量；新版Tutor DEV GPU NOT_RUN |
| T11 | DONE_CPU | 共享S3_MARKET_MINUTE；Oracle/Reward版本冻结 |
| T12 | DONE_CPU | 模型/优化器/采样计数恢复；沿用FullCheckpoint完整协议 |
| B01 | DONE_CPU_AND_REAL_RECORDED_DEV | H60六类/独立early真值、删失、资金/时序/边界认证 |
| B02 | RETAINED | 沿用原两个128头，无新增网络 |
| B03 | DONE_CPU | BC与policy共用同一Outcome变换/一次mask；mix0一致/真实mask/最终logit关系 |
| B04 | DONE_CPU | BC→OutcomeHead/RSSM=0；Adapter/Actor非零；无未来Observation |
| B05 | DONE_CPU | Outcome→Encoder/RSSM/Heads非零，原WM梯度保留 |
| B06 | DONE_CPU | 风险符号、StopGrad和delta±2不变；新增未裁剪/裁剪/实际应用诊断 |
| B07 | DONE_CPU_CONFIGURED_NOT_RUN_GPU | 新配置100/100/100；99/100/101/199/200/事件不足边界与恢复 |
| B08 | DONE_REAL_RECORDED_DEV_POSTHOC | 先封存预测再给标签；5000真实DEV决策、原Env参考认证、NLL/Brier/先验/校准/SL；非新Full性能 |
| B09 | DONE_CPU_AND_RECORDED_DEV | 真实base/final margins、delta/饱和/argmax；旧缺失logits=null，旧DEV变动作0 |
| B10 | RESEARCH_DEFERRED_BY_SCOPE | 保持beta=1，不混入独立DAgger研究 |
| B11 | DONE_REAL_READONLY_REGRESSION | 评价V2前5/6–15/16+/全持仓；800K四窗0/1/3/4；旧收据不改 |
| B12 | RESEARCH_DEFERRED_BY_SCOPE | 不擅自增加Q_WAIT/C2/显式效用Loss |
| B13 | DONE_CPU | 完整FullCheckpoint一起恢复模型/优化器/Replay/Sidecar/Driver/运行carry/计数，再collect5和更新；100预热独立状态恢复 |

## 已确认的原因与修复

- 旧 `early_mae10` 实际统计全持仓FIRST_ADVERSE10。共享纯stdlib `fg_s3_eval_metrics.py` 统一新评价器、观察工具和现有Tracker；旧字段只保留whole-hold含义。缺日志=null，缺有效分钟归因=unclassified，重复事件去重、冲突拒绝；不从时间戳推持仓分钟。
- 旧 `b2/raw_delta_logit` 已裁剪。保留旧名称的旧含义，新增真实未裁剪值/裁剪值/应用值、base/final logits及饱和标志。预热仅改新配置，原MixSchedule计数和保存机制保留。
- 自Review进一步加固模型身份：B2 Checkpoint必须每行都有真实Outcome预测，缺失/模式不匹配直接拒绝；Tracker评价kind必须匹配原模型mode，不能把缺B2诊断误认作Tutor。对应回归和5005条真实行模式核验通过。
- 原CandidateIndex参考校验固定比较C2 reward。改为Env实际candidate，供原C1 DEV认证；不改Oracle优化数学/交易规则。TRAIN OracleStateQuery/OutcomeCatalog的TRAIN限制没有放宽；复用共用账户资金证明，DEV结果不能进入监督sidecar。
- W&B0.19.1 resume传tags会替换已有tags。SDK worker现在合并现有云端tags并保留人工display name；无800K元数据操作。新预测评价拥有独立身份/索引，晚到收据不重写Actor-only历史。
- W&B内部默认代理`http://127.0.0.1:10808`用于认证/SDK/同步；模块直接调用也强制代理，排除环境NO_PROXY=*绕行。原训练子进程环境不变。聚合保留真实B2阶段整数、最后调度计数及实际权重极值。
- 回归中发现旧Tutor fixture取的是缺market_advance的A2旧Replay，改为只读市场分钟Tutor1K Checkpoint；没有给旧数据伪造字段。

## 真实证据与测试

| 验证 | 实测结果 | 范围 |
|---|---|---|
| B2 + Tutor CPU模型合同 | 9 PASS | 小模型、受控原Env/原生Replay；非新GPU验收 |
| 组合FullCheckpoint恢复 | 1 PASS | 模型/优化器/Replay/Sidecar/Driver/carry/计数一起保存恢复；继续collect5和梯度更新，参数完全一致 |
| 公共Oracle/时间/评价/Tutor合同 | 37 PASS / 1 SKIP | SKIP是缺少旧GPU收据的检查，不能算PASS |
| W&B CPU/Mock/真实Offline SDK | 20 PASS，加新模块代理回归1 PASS | 不创建云端Run；SDK本地服务需沙箱外执行 |
| 真实B2 FG409历史预测posthoc | PASS | 5005预测行、5000实际执行决策；来源CP和账户完全复核；无重新GPU推理 |
| Full新配置CPU prepare | PASS（最终v3见prepare收据） | 原Replay来源/可信Oracle/完整冻结配置；不初始化GPU训练 |
| 新Full GPU1K/10K/1M | NOT_RUN | 本任务书未授权 |

B2 CPU真实梯度：Actor `0.6753884554`，RSSM `1143.5872802734`，OutcomeHead `0.4574307501`，Critic `0.0`；每次联合优化1次。100边界测试通过，恢复下一次参数完全相同。组合FullCheckpoint测试使用独立受控2/2预热以短程验证完整状态耦合，正式研究配置仍为100/100/100。

Tutor动态权重/分项梯度使用当前批的实际WM特征及优化前Actor快照，每100更新采样，无第二WM或优化器更新。CPU对比采样开/关的参数和计数一致，分项梯度非零且总Actor梯度与原联合优化一致；Critic与专属状态冻结。

800K只读数学回归：7 ADD / 5 TP / 2 SL，TP市场分钟 `[56,4,221,89,74]`，p50=74，Reward=-3.805870048525721，净盈亏=-2226，逆行窗0/1/3、全持仓4。旧receipt文件SHA `b1665c584aedbd7629494a0fd5ad68fe6e52d1649a03de602687d583530e05f9` 保持不变。没有提交云端里程碑。

历史B2独立DEV预测诊断：

| 指标 | 模型 | 同样本常数先验 |
|---|---:|---:|
| Event NLL | 1.636813 | 1.207422 |
| Event Brier（六类和） | 0.806815 | 0.636732 |
| Early Brier | 0.042433 | 0.021325 |

- 实际认证覆盖5000/5000。六类支持 FAST180/MID572/LATE1502/SL212/OTHER0/OPEN2534；SL Recall2.8302%、Precision2.8986%，OTHER零支持=null。DEV Adapter改动作0，仍All-WAIT。旧缺失raw logits/饱和数据保持null。
- 原模型10K/400更新实际runtime Gym65477b0 / Dreamer a194a22、预热4/16；**上述负结果不是新Full100预热的性能结果**。原CP manifestSHA `28401fdbe97aabebeb518156c2be1d747649e006bc911b6520e9e3b4794fe4fd`。
- 最终本地收据 `artifacts/fg_s3_v2/full_checks/b2_v0_dev_posthoc_v2/receipt.json`，content_hash `2a83ad9bb94bd6e8536ba3cd217eb0687f2cb1e8bd338a242fe4be304f73c305`。新输出独立；没有自动上传旧B2。
- 新版GPU吞吐NOT_RUN。旧B2真实100 cycles合计核心634.08秒，15.77环境步/秒，即10K核心约10.6分钟（不含全部Checkpoint/独立DEV开销）。这只是同硬件参考，新Full应在获授权的短跑中重新测量。

### 可重复验证命令

在集成Gym目录设置：

```bash
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/fg_s3_full_mpl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=src:../dreamerv3
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
"$PY" -m pytest ../dreamerv3/embodied/tests/test_fg_s3_b2.py ../dreamerv3/embodied/tests/test_fg_s3_tutor_full.py -q
"$PY" -m pytest tests/entry_v1/test_fg_s3_b2_outcome.py tests/entry_v1/test_s3_market_minute.py tests/entry_v1/test_fg_s3_full_prediction.py tests/entry_v1/test_fg_s3_tutor_cd.py tests/entry_v1/test_fg_s3_tutor_e2.py -k "not small_cpu and not real_native and not actual_gpu and not real_original" -q
/home/v/Documents/work/research/fg-s3-wandb-env/bin/python -m unittest research_tracking.tests.test_eval_v2 research_tracking.tests.test_prediction_evals research_tracking.tests.test_tracker -v
```

准确已执行分组命令及stdout复制到 `artifacts/fg_s3_v2/full_checks/{cpu_neural,cpu_full_checkpoint,cpu_contracts,tracking_offline}.log`；本报告不把新的全量组合命令伪称为已经一次执行（模型9项和新增完整恢复1项分别执行）。初次Offline受沙箱SDK服务限制失败，提权后通过；初次旧fixture/旧代理断言失败已保留日志并修正。

## 启动、恢复、评价与W&B

最终准备命令是CPU，启动/恢复/Actor评价会用GPU；以下GPU命令本轮未执行，先Review并另行授权。v1/v2开发准备快照保留，最终v3独立Run尚未启动。

```bash
cd /home/v/Documents/work/research/fg-s3-full-integration-dev/gym-trading-env
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
TRACKER_PY=/home/v/Documents/work/research/fg-s3-wandb-env/bin/python
# 最终交付已prepare；只有在新干净源码目录首次准备时执行一次：
"$PY" tools/fg_s3_v2_experiment.py prepare --config B2_FULL_MM_10K
# 获授权后的1K工程运行，仍mix=0：
env WANDB_ENTITY=xkm104501-none WANDB_MODE=online "$TRACKER_PY" tools/fg_s3_wandb.py launch \
  --run-dir "$PWD/artifacts/fg_s3_v2/runs/B2_FULL_MM_10K_v3" \
  --prepared "$PWD/artifacts/fg_s3_v2/prepared/B2_FULL_MM_10K_v3" --trainer-cwd "$PWD" -- \
  "$PY" tools/fg_s3_v2_experiment.py start --config B2_FULL_MM_10K --stop-at 1000
# 同一来源/代码/配置恢复到10K，改launch尾部为：
"$PY" tools/fg_s3_v2_experiment.py resume --config B2_FULL_MM_10K
"$PY" tools/fg_s3_v2_experiment.py evaluate --config B2_FULL_MM_10K --eval-steps 5000
"$PY" tools/fg_s3_v2_experiment.py score-predictions \
  --eval-dir artifacts/fg_s3_v2/runs/B2_FULL_MM_10K_v3/evaluations/actor_only_REPLACE_WITH_ID \
  --output-dir artifacts/fg_s3_v2/runs/B2_FULL_MM_10K_v3/prediction_evaluations/dev_REPLACE_WITH_NEW_ID
env WANDB_ENTITY=xkm104501-none "$TRACKER_PY" tools/fg_s3_wandb.py import-evals --mode online \
  --run-dir "$PWD/artifacts/fg_s3_v2/runs/B2_FULL_MM_10K_v3" \
  --prepared "$PWD/artifacts/fg_s3_v2/prepared/B2_FULL_MM_10K_v3"
# 断网本地outbox保留；后续恢复联网：
env WANDB_ENTITY=xkm104501-none "$TRACKER_PY" tools/fg_s3_wandb.py sync \
  --run-dir "$PWD/artifacts/fg_s3_v2/runs/B2_FULL_MM_10K_v3" \
  --prepared "$PWD/artifacts/fg_s3_v2/prepared/B2_FULL_MM_10K_v3"
```

Tutor使用`TUTOR_CD_FULL_E2_MM_10K`与对应`TUTOR_CD_FULL_E2_MM_10K_v3`路径。所有输入/初始化来源相同，配置、teacher/head目标、Run身份独立；不叠加Loss。训练器不依赖W&B网络；代理已内置无需额外--proxy。

W&B真实只读认证确认私人项目可访问：[fg-single3-research](https://wandb.ai/xkm104501-none/fg-single3-research)。本次未创建Full云端Run，因此无新Full Run URL；800K里程碑不在本任务。

## 自Review与仍未验证

- 已核对两路线共用公共协议、市场分钟、原规则/认证、BC StopGrad、单次优化、Critic专属状态冻结与恢复；旧产物只读，禁止覆盖/热切换。无C2/Q值/额外Replay/额外Oracle训练索引。
- 新Full GPU闭环、10K实际学习效果、Full Actor-only DEV绩效与1M训练均NOT_RUN；没有把受控模型或旧真实录制数据补评分当成新版生产验收。Tutor posthoc分支已经实现，尚无新Full Tutor GPU预测可评价。
- 推荐超参属于研究选择。旧B2预测弱于同样本先验，新Full是否改善须实测。
- 交付结论：实现与CPU/只读数据验收GO_WITH_LIMITS；当前停止，等待下一轮Review/单独GPU授权。
