# Tutor-CD Full / B2 Full 实施计划（2026-10-14）

实施依据：`FG_S3_Tutor_B2_Full_Spec_Review_WandB_Milestone_Codex_20261014.md`，以及实际源码。用户最新补充：**800K W&B 里程碑由独立项目处理，本次不开发、不上传、不修改该里程碑。** 历史800K评价只读用于公共评价语义回归。

## 已核验基线与隔离

- 最新共享 Gym：`ad6490675bf853a7793763f95aa7ba7f12fa379e`；Dreamer：`d76a423dd7930756c3609cc6f7197941f583c160`。W&B工作树干净，已有 Tutor E2 和 B2；不依归档推断新HEAD。
- 公共研发双工作树：`/home/v/Documents/work/research/fg-s3-full-integration-dev/{gym-trading-env,dreamerv3}`，分支 `feat/fg-s3-full-integration`。
- 从公共提交派生 Tutor Full / B2 Full 双工作树，分别为 `fg-s3-tutor-full-dev`、`fg-s3-b2-full-dev`。仅通过Git提交集成源码。
- 旧Tutor冻结Gym `3a6f889adf65583a1ab636f244ca05218f19196f` / Dreamer `c18a8b5917f73110f9b6ce840495f4da1d7ca1d4` 及其正式Run、Checkpoint、Replay、Sidecar、收据全程只读。原仓库已有用户未跟踪文件，保持原样。
- W&B复用现有 `research_tracking` 和 `tools/fg_s3_wandb.py`，Project `fg-single3-research`，不新增Tracker。

## 实施顺序和验收

1. **公共评价V2**：共享市场分钟归因，明确前5、6–15、16+和全持仓逆行10点；未知值不伪造0、重复事件不重复计数。更新评价器、观察工具和现有Tracker；旧 `early_mae10` 仅代表legacy全持仓。回归800K真实只读7 ADD / 5 TP / 2 SL，四窗0/1/3/4。
2. **Tutor Full**：保留Natural/Balanced/Pair及现有E2；新增实际权重/困难题影响诊断和低频Actor分项梯度诊断，诊断不得改优化器或随机/Replay状态。独立配置FG201、10K环境步/400更新，alpha=.8、pair=.1、tau=1、kappa=1、wmax=2、beta=1。
3. **B2 Full**：保留OutcomeHead、有符号StopGrad Adapter、原WM与Oracle BC；独立FG201 10K配置，warmup/ramp/min_event_updates均100，保持delta上限2。新增真实base/final logits、未截断/截断/实际修正和阶段统计；确保Checkpoint恢复调度状态。
4. **独立DEV预测评价**：先封存因果预测/实际动作/账户状态，再以原Env/Oracle数学认证反事实ADD结果。独立评价角色，不放宽TRAIN监督准入；输出NLL/Brier、类别支持、early校准、SL识别、同样本常数先验和Adapter决策影响。旧B2真实5005条预测可只读做本地CPU验收，不自动上传旧B2。
5. **W&B联接与快速验收**：扩展现有schema/指标映射以关联Full配置、Checkpoint和独立评价；离线不影响训练。保持已有Run tags/人工命名。CPU最小回归、模型梯度和恢复、Mock/Offline及旧文件hash保护验证，集成后自行Review。

## 已确认的研究参数与边界

- 两种Full首次配置都从同一已审计 Online180k Replay Checkpoint独立初始化；不是从旧Tutor1M迁移。
- Tutor每100更新采样Actor完整参数的分项梯度；报告是优化器裁剪前梯度诊断，不宣称是动量/裁剪后的实际归因。
- B2调度保持现有累计更新语义；有效event更新不足时mix=0，不为了1K smoke强行启用。
- 推荐超参保留可配置性，属于研究选择，不宣称最优或确定性Bug修复。
- beta退火后置；不开发C2、Q值、多周期、B2/Tutor混合Loss或额外Replay系统。
- 本轮只授权CPU/离线及既有数据只读验证，**新GPU训练、10K试训练、正式1M均NOT_RUN，须另外授权**。
- 800K云端里程碑完全排除。本次结束提交双仓SHA、逐项差异矩阵、真实测试命令/结果、新配置和可恢复启动命令，然后暂停等待Review。

## 进度记录

实施完成：公共评价V2、Tutor Full E2诊断、B2 Full预热与DEV预测补评、原Tracker联接/内置代理已提交并集成。详见同目录 `FG_S3_Full_Implementation_Report_20261014.md` 的25项矩阵、真实命令/结果及NOT_RUN边界。

最终配置/准备与Run使用独立v3名称（开发v1准备只读保留）。两路线同为可信Online180k参数warm-start和FG201诊断10K/400更新，beta仍1。用户最后指定W&B内部默认代理为 `http://127.0.0.1:10808`，现已实现，原训练进程环境不变。

CPU神经合同、组合FullCheckpoint恢复、原Env/时间边界、真实旧B2预测补评及Offline Tracker验证已完成；新Full GPU/10K研究/1M均NOT_RUN。停止在交付版本，等待Review，不能将受控或历史录制数据补评解释为新版生产验收。
