# E + Tutor Full：最新1M Checkpoint Actor-only评估

## Summary

已在GPU完成FG409 DEV的5000市场分钟Actor-only评估。结果为全WAIT：实际开仓0、TP0、SL0、Reward0、净PnL0。最高可执行ADD概率为0.4980793297，当前eval argmax没有选择ADD。该窗口没有交易样本，胜率、TP耗时等为未定义，不将0交易记为策略效果通过。

## Evidence observed

| 项目 | 实际结果 |
|---|---|
| 正式Run | E_TUTOR_CD_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261015 |
| 选定模型 | step_000001000000，训练1000000市场分钟/40000联合更新 |
| Checkpoint manifest SHA256 | `17c28209a087c7c1b7f77ea24fa3fccbc2d2227f703b8a91eb502e539f034041` |
| 评估 | CZCE.FG409 DEV，5000真实推进，reset_seed0，eval argmax，beta0 |
| 时间窗口 | 2024-04-08 21:01至2024-04-29 09:51，Asia/Shanghai |
| 原始提案 / 已执行决策 | 5005 / 5000；reset/末尾提案不计市场分钟 |
| 允许ADD的已执行决策 | 5000 |
| ADD / 开仓 / TP / SL / 截尾仓位 | 0 / 0 / 0 / 0 / 0 |
| Reward / 净PnL / 最大权益回撤 | 0 / 0 CNY / 0 CNY |
| 初始 / 期末权益 | 300000 / 300000 CNY |
| ADD概率中位 / P90 / P99 / 最大 | 0.0997103 / 0.1994697 / 0.3780130 / 0.4980793 |
| GPU | cuda:0；直接提权执行，没有先运行设备存在性探测 |
| Oracle / Teacher / TEST | 无Oracle数据读取、无教师指导、TEST价格保持封存 |

原收据的`legal_decisions=5001`包含末尾未执行提案；本报告采用相邻记录的真实`advanced`字段对齐，已执行合法决策分母为5000。对齐证据见[evidence/decision_audit.json](evidence/decision_audit.json)。

## Root cause

评估接线缺口：冻结`fg_s3_v2_evaluation.evaluate`调用普通`load_env`，没有向DEV Env传入E profile；这条入口原本无法给E策略提供两个新增观察字段。新评估适配层补齐该Env构造，仍复用训练版本的原生E实现、模型和原评估循环。

全WAIT的直接数值依据：5000个已执行决策均允许ADD，实际Actor概率全部小于0.5，因此二动作argmax选择WAIT。最高分记录cursor2912：p_ADD=0.4980793297，Tutor最终ADD-WAIT logit差=-0.0076828003。训练为何形成这段概率分布，本轮没有诊断，不能据此归因于Gate、类别比例或Encoder。

## Changes made

只在用户指定的较新`fg-s3-e-review-fix-20261015`工作树新增`tools/fg_s3_v2_evaluate_frozen.py`。这是只读评估启动器，直接调用原冻结评估函数，没有复制评估循环或另建模型。

- 原模型配置、ActorPolicy、模型类、完整Checkpoint验证、Composition和源码身份检查均来自该Run实际冻结版本：Gym `871ca09000712d41408b8f72e0684624dce8659f`、Dreamer `730939defb1bd77b2ceca04212105ec2727c95f9`。
- 启动器只替换评估调用中的DEV Env工厂：沿用原`load_env`的文件/路径守卫，使用DEV专属profile和同一份通过SHA256验证的CSV字节；不以DEV绕过TRAIN准入，不给profile授予训练权限。
- 原生Env提供`market_seq_e_shared[60,18]`、`market_context_15m[24,7]`及原三个字段。未改变E公式、Actor、RSSM、Reward、TP5/SL30、Action Mask或Gate。
- 独立run_view链接指定Checkpoint和原身份，所有评估输出写research新目录。进程内拒绝打开TEST价格或Oracle数据，并拒绝向原Run/prepared打开写入句柄。没有写原训练目录或上传W&B。

适配器源代码SHA记录在[evidence/launch.json](evidence/launch.json)，其额外运行身份单独披露；没有把适配代码冒充成训练时的完整源码身份。

## Verification performed

实际命令及提权参数见[COMMAND.md](COMMAND.md)。GPU进程退出码0，完整运行5000市场分钟，没有额外训练或通用测试。

原冻结评估器完成42个Checkpoint文件的完整校验后加载原Agent，模型更新计数始终为40000。评估前后的Run身份、训练收据、latest指针、Checkpoint manifest和prepared manifest哈希一致。Oracle/TEST禁止访问事件为空。

- [receipt.json](evidence/receipt.json)：原冻结评估器生成的签名收据。
- [verification.json](evidence/verification.json)：规范Checkpoint路径、源身份不变、DEV E输入合同、cuda结果与40000更新计数。
- [metrics.json](evidence/metrics.json)：复用既有指标函数计算的结果和最高ADD分数记录。
- [actions.jsonl](evidence/actions.jsonl)、[account_state.json](evidence/account_state.json)：完整动作和最终账户证据。
- [e_dev_profile.json](evidence/e_dev_profile.json)、[dev_manifest.json](evidence/dev_manifest.json)：DEV专属输入身份与严格时间切分。
- [index.json](evidence/index.json)：所有原始证据来源路径、文件大小及SHA256。

执行原始输出与完整收据也保留在`/home/v/Documents/work/research/fg-s3-e-review-fix-20261015/actor_only_1m_latest_20261015_v1/`，原Checkpoint未复制或修改。

## Risks / not verified

本次仅是FG409 DEV首5000市场分钟、一个最终Checkpoint、确定性argmax策略。全WAIT限制了交易效果评价；无交易时胜率、SL率、TP耗时均未定义。没有读取TEST，没有执行整段DEV、随机策略、阈值调优、其他Checkpoint对比或重新训练。

评估冷加载了完整Agent状态，不等于训练Replay/Driver/优化器继续更新的完整恢复验收。原模型训练源码及此前CSV身份修复均未热合并或改写旧Run身份。

## Next recommended step

本次用户要求的最新Checkpoint Actor-only评估已完成。保留当前结果，后续扩大DEV窗口、对比其他Checkpoint或诊断全WAIT分布，按用户新指令执行。
