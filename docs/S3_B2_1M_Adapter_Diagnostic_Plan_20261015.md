# B2 1M Adapter 诊断执行范围

本轮只做诊断，不增加训练预算，不改变 B2、Tutor 或公共集成基线的算法。
开发工具位于独立 Gym 分支 `feat/s3-b2-adapter-diagnostics`；神经运行时使用
原 B2 正式训练冻结的 Gym/Dreamer 源码，而不是诊断 worktree 的模型模块。

## 已核实来源

- 原 Gym：`3970e6ec69f67fb2d02f30456a077ea5a31fd05d`。
- 原 Dreamer：`5fd4ebdeb590fd8433338300bd82c7f2f520bf76`。
- Run：`/data/logdir/fg_single3/s3_v2/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014`。
- Checkpoint：`ckpt/step_000001000000`，Manifest SHA256：
  `6eda87d3f0d173acc988fbad9e2f2e456498be230e6b0bf24ec74685fca9db57`。
- 已有 ON 评价：`fg-s3-local-evaluations/b2_step_000001000000_20261015_050252/evaluations/actor_only_dcbd7ec78c`。
  收据 SHA256：`f8dc147b4e5f3897c7d09159ba16628918eea5ae93bc2968728f83230ca75b8b`。

## A/B/C 与证据边界

1. **A — OFF 独立闭环**：原生 Driver/Env，FG409 DEV 5000 个实际环境推进，
   原模型完整恢复；只在诊断进程中让 `MixSchedule.read()` 的返回值为零。
   原始 schedule 变量照常读取、初始化和恢复，不改任何 Checkpoint 参数或文件。
   核对全部模型/优化器参数哈希加载前后相等、评价后相等，以及每步实际修正为零。
2. **B — ON 独立闭环**：复用已经完成并密封的同一 Checkpoint 5000 步评价。
   不重新运行，不改写旧收据。ON 轨迹的基础 Actor 提议只是影子动作，不能当作
   OFF 的独立闭环交易数。
3. **C — Oracle 事后对照**：在 CPU 上复用原 CandidateIndex、资金认证、
   `outcome_label` 和 exact suffix Oracle。四个被否决状态分别从其合法空仓账户
   开始实际 ADD→WAIT 到退出，原生 Env 验证索引结果；不是连续四笔策略交易。
   Oracle 是否立即 ADD 使用 `suffix.first[t] == t`，不以整段最优交易表成员代替。
4. **OutcomeHead 质量**：复用原 `score-predictions`，先验证密封的因果预测和
   原动作轨迹账户，再对 5000 个认证 DEV 空仓状态事后赋标签。
   输出事件/早期逆行 NLL、Brier、六类召回及校准。
   同批 DEV 的常数先验仅为描述性对照，不是 TRAIN 基线。

未来行情及 Oracle 标签仅用于事后诊断；不传入 Actor，不参与执行，不进入训练。
结果标记 `DEV_EVALUATION_ONLY`、`eligible_for_training=false`。
全部新输出写入独立 `fg-s3-b2-diagnostics/results/`，不上传 W&B。

原密封预测保存了 ADD−WAIT logit 差和掩码后概率，没有分别保存两个绝对 raw
logit；报告如实列差值，不凭概率反推并宣称取得原始绝对 logits。

## 完成条件

- OFF GPU 实际闭环完成、恢复及全部参数不变检查通过。
- 四个否决都有真实 Env 退出/Reward、H60 Outcome 和原 Oracle 事后选择。
- ON 5000 个认证状态预测评分完成，给出样本数与局限。
- 文档、命令、收据路径及工具正常提交；不改变正式训练、原收据或云端记录。
- 未完成或失败项目明确记录，停止于诊断，不启动新训练。
