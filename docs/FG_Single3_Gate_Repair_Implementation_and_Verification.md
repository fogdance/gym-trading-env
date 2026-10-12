# FG Single3 Gate 修复实施与验收

实施依据：`FG_Single3_Availability_Gate_Repair_Before_Resume_Codex_V1.md`。D0–D5完成，结论为GO（可由用户另行授权恢复训练）。完整CPU预检、双路径状态继承和未授权启动拒绝均有真实证据；正式GPU训练未启动。

## 已确认的问题与修复

旧代码的Gate分类固定为`logit>=0`，Imagination由`agent.avail_threshold`控制。旧配置为0.50，两者实际一致；但只修改YAML会让新概率阈值与Gate统计脱节。Dreamer提交`4659d329375ee591aab5116f570602fcf6ae4cf8`增加统一的float32判定，Gate Post/Prior与Imagination共用同一个阈值与`>=`边界；保底WAIT不混入原始分类统计。

同步补充阈值/版本、Post/Prior TP/FN/FP/negative、Post FPR和想象合法性日志。ADD Recall仍按整窗TP/positives计算；原批均值FPR口径和全部质量线不变。真实执行仍使用Env Mask，WAIT=0始终可用。没有新增网络、损失、教师数据、Reward研究或Replay构建。

FG009真实20,891行、104次退出完整对账未发现Action/Observation/Reward/Reset错位。实际Prior使用上一posterior和历史action计算本帧状态，不吸收当前观察；Post吸收包含真实Mask的当前观察。当前Prior漏判集中于释放状态，不能用Post高Recall证明其已经解决。详细交易证据见[原因审计](FG_Gate_Prior_Release_Cause_Audit.md)。

## 来源、文件范围与不变量

起始Gym HEAD=`06dd2d3c6fed7d84997335efc8f9ae78a8020c7f`；起始Dreamer HEAD=`0b3a6419dfef89ac5eb13d06e65e2e576699d7f5`；源180k节点实际代码身份对应`09e2e5c`。原180k节点183个文件已验证，停止日志到184,260不等于可继承184,260步。

Dreamer生产修改仅为`agent.py`共享判定、`train.py`诊断聚合、`action_mask_warmup.py`日志别名、`fg_stage_transfer.py`显式迁移准入。现有`embodied/jax/agent.py`参数发布修复是来源与当前分支之间已存在的差异，明确列入迁移白名单，不静默重置策略缓存。

Gym复用`fg_single3_prior_audit.py`增加可选posterior starts输出；`tools/fg_single3_experiment.py`继续作为唯一训练入口，集中配置新增`C1_M1_GATE_T030_from_180K.yaml`。`research/fg_single3_gate_calibration.py`只负责编排、哈希和单阈值迁移准入，复用现有Agent/FullCheckpoint/Replay/Driver/Env。CPU诊断工具分别负责真实行情对账、概率和窗口预检、原代码零差异检查以及实际恢复，不复制训练系统。

保持C1、3手单仓、TP5/SL30、H333、Imag15、B16/L64/context1、model seed0、Replay seed202、在线1M预算、80% Post/Prior Recall、原Accuracy/Exact/FPR与20窗及连续5次回归停训规则。TRAIN数据范围与Replay完全继承来源；没有读取DEV/TEST价格。

## 验证矩阵（最终状态）

|项|当前结果|真实证据|
|---|---|---|
|T1 半阈值旧边界、极值、BF16|PASS|`test_availability_threshold_keeps_legacy_half_boundary_in_float_and_bf16`|
|T2 0.30共享判定与raw/safe WAIT隔离|PASS|`test_availability_gate_and_imagination_share_raw_threshold_before_safe_wait`|
|T3 真实180k复现|PASS|179,968位置逐点Post/Prior概率差0；0.30 Prior TP839/FN177/FP179；全部候选完整指标保留|
|T4 因果context、历史Action、reset、next-bar与释放对账|PASS（已见路径）|2个已有时序单测通过；FG009全路径、91TP/13SL逐笔一致；未见其他退出原因，相关真实事件NOT_RUN|
|T5 19窗不开/20窗开、支持不足/失败重置、回归规则|PASS（单测与冻结20窗）|已有13个warmup测试与2个训练聚合/切换测试；0.30真实冻结窗口20/20通过，每窗317–651个正例，未改变质量线|
|T6 真实15步Imagination|PASS（扫描、正例与20窗）|每候选323,584动作判定，另有16,384次正例分层抽样；0.30整20窗31,948,800次真实动作判定，fallback/非法/WAIT非法均0|
|T7 全状态双路径迁移|PASS（来源与两个新身份路径）|来源97.7秒、目标双路径131.8秒；全部Agent/Optimizer/账户/Replay/RNG一致，源183文件不变，Gate=false/streak=0；实际证书已生成|
|T8 0.50真实原训练指标零差异|PASS|旧`09e2e5c`真实Model.loss对照，loss=18.228124618530273，原45项指标最大差全部0|
|T9 未授权不启动、DEV/TEST不访问|PASS（共享入口与真实官方CLI）|拒绝授权/清理授权单测通过；新配置只读官方准入PASS；真实未授权CLI在main.py:180以PermissionError拒绝，无Agent/GPU初始化；未读DEV/TEST价格|

新Dreamer2个判定测试+原warmup13个+训练聚合/切换2个：17 passed（1.42秒）。已有context/先验时序2个测试：2 passed（8.04秒）。Gym迁移合同3个测试通过。没有为提交重复运行大型套件。

最小测试命令（使用dreamerv3 Conda Python，`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`）：

```bash
# dreamerv3仓库内
python -m pytest -q -p no:cacheprovider \
  dreamerv3/tests/test_action_mask_contracts.py::test_availability_threshold_keeps_legacy_half_boundary_in_float_and_bf16 \
  dreamerv3/tests/test_action_mask_contracts.py::test_availability_gate_and_imagination_share_raw_threshold_before_safe_wait \
  dreamerv3/tests/test_action_mask_warmup.py embodied/tests/test_fg_add_gate_training.py
python -m pytest -q -p no:cacheprovider \
  dreamerv3/tests/test_action_mask_contracts.py::test_prior_feature_does_not_read_current_observation_and_resets_history \
  dreamerv3/tests/test_action_mask_contracts.py::test_apply_replay_context_shifts_actions_but_not_masks
# gym-trading-env仓库内
python -m pytest -q -p no:cacheprovider tests/entry_v1/test_fg_single3_gate_calibration.py
```

真实神经零差异回归：

```bash
python tools/fg_single3_gate_metric_regression.py \
  --source /data/logdir/fg_single3/online_after_10k_callback_fix/C1_M1_ONLINE_1M/ckpt/step_000000180000 \
  --output artifacts/fg_single3_v1/diagnostics/gate_metric_regression_recheck
```

行情对账、扫描/窗口和双路径恢复的完整命令见三个分项报告。Gym最近一次实际测试stdout与exit=0保存在`targeted_tests.json`；两仓库`git diff --check`通过。自审增加了非有限预检指标拒绝，未放宽质量门槛。

真实扫描795.9秒、TP/SL逐帧对账41.1秒、原指标神经零差异回归74.4秒、正例想象抽样94.4秒。20窗中后14窗并行续算实测5,153.7秒（85.9分钟），此前6窗已实际完成；双路径恢复131.8秒。诊断和模型后端均为CPU，没有进行来源模型优化器更新。

## 审计产物和中间失败

全部产物位于`artifacts/fg_single3_v1/diagnostics/gate_repair_20261012_01/`，基线和各次运行的manifest/analysis保存输入、输出、源码和Checkpoint哈希。已完成证据为`cause_verified`、`scan_verified`、`metric_regression`、`positive_imagination`、`source_cpu_restore_verified`；`windows_verified`完整20窗已结束且exit=0；`migrated_cpu_restore`、`formal_admission.json`、`final_integrity.json`和项目恢复包证书完成最终验收。

保留首轮行情审计JSON中NaN序列化失败、首次恢复检查对象repr地址不稳定、半CPU扫描与原审计不一致、被主动中断并继续利用6个已完成窗口的串行运行。修正各具体原因后使用新目录留存通过证据，没有覆盖失败结果。半CPU数值差异经同批输入对照定位为CPU亲和性/BF16运算及离散采样差异；全24核复核与原扫描逐点相同。该结果不能推断CUDA必然相同。

## 自审与完成边界

源码证据：Dreamer `agent.py:28/35`校验与共用判定、`agent.py:258`真实Post/Prior指标、`agent.py:384`想象Mask、`agent.py:477`因果上下文；`embodied/run/train.py:181/277`窗口聚合和Gate报告；`action_mask_warmup.py:54/64`实际开启余量与回归规则；`fg_stage_transfer.py:27`显式迁移入口。Gym `fg_single3_gate_calibration.py:48/65/90/169/204/347/398`分别约束配置差异、清理旧统计、真实预检、严格身份、完整迁移、CPU恢复证书和授权恢复。

18/20窗口时的主机只读样本：3个CPU worker的累计平均占用为626%/628%/627%，约18.8个逻辑核；父进程RSS约2.14GiB，各worker约2.62–2.65GiB，内存available约7.34GiB。它是单次样本而非峰值；精确输出留存`cpu_resource_sample.json`。未为重复测试或提交另起大计算任务。

CPU完整20窗、原全部质量条件与安全Mask已经通过，已准备、发布新身份完整节点并通过两条恢复路径实测，已签发恢复证书。正式共享入口需要GO、证书、严格身份和用户显式授权，CPU完成不能自动启动GPU。

尚未验证GPU继续更新后的Gate稳定性、FG209/309在线路径、独立DEV泛化或Agent择时有效。新分支Gate实测为NOT_READY、streak=0，必须依真实在线报告重新连续20次达标。工程修复、冻结模型CPU达标与正式Actor有效训练是三个不同结论。

## 最终问题回答

1. **是否解决Gate阻塞？** 统一判定接线已修复；冻结180k模型在0.30下通过全部原开启质量线和20个连续窗口，具备恢复条件。正式在线Gate仍未开启，继续更新后的长期稳定性没有验证，不能把CPU预检说成已经启用Actor/Critic。
2. **推荐阈值？** 本次推荐0.30。0.50有19窗分类质量失败、0.35有5窗Prior Recall低于80%；0.30通过20/20。全量Prior Recall=82.57874%、FPR=0.10003%；窗口Prior Recall最低80.24263%，余量仅0.24263个百分点，明确保留后续不稳定风险。FG009 Recall仍为63.158%，不新增逐合约Gate或专属阈值。
3. **能否安全继承180k？** 能，在已完成的CPU完整恢复范围内。两路径全部模型/优化器/账户/Replay/RNG状态相等，累计环境步180,000、updates21,183保留；只清理旧规则统计，正式streak从0重算。不继承未保存的180k之后4,260步。

工程修复与完整状态继承已验收；冻结CPU模型的Gate可达性已验收；正式GPU训练、在线Gate持续开启、Actor有效更新和独立DEV择时有效性均NOT_RUN。没有把工程通过解释为Agent已学会入场。

## 源码提交与建议恢复命令

Gym代码提交`cba9080`（共享入口、集中配置、完整迁移及可复用CPU诊断）；Dreamer提交`4659d329375ee591aab5116f570602fcf6ae4cf8`（共享概率阈值、日志和显式迁移准入）。报告作为单独文档提交，不改变已冻结源码/配置或恢复身份。

恢复包与新日志节点已准备完成；无需重跑prepare/publish或预训练。以下命令**本阶段未执行**，仅供用户另行决定正式恢复时执行：

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python \
  /home/v/Documents/work/gym-trading-env/tools/fg_single3_experiment.py resume \
  --config /home/v/Documents/work/gym-trading-env/configs/experiments/fg_single3/C1_M1_GATE_T030_from_180K.yaml \
  --authorize-training
```

从新身份的180k节点继续在线训练至总1M环境步；模型不会从0开始，也不会重新做10k离线预训练。原CPU收据、Replay、Source Checkpoint、停止日志和失败诊断均保留。执行后仍须实际记录新Gate20窗和Actor/Critic有效更新，达不到原门槛不能称为Actor完成择时训练。

关联报告：[原因审计](FG_Gate_Prior_Release_Cause_Audit.md)、[完整阈值预检](FG_Gate_Threshold_Preflight.md)、[来源迁移审查](FG_Gate_Source_Migration_Review.md)。全部产物哈希索引见`artifacts/fg_single3_v1/diagnostics/gate_repair_20261012_01/completion_manifest.json`。
