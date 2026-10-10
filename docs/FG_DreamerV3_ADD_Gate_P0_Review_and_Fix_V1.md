# FG ADD Gate P0 补丁评审与订正 V1

日期：2026-10-11。评审对象：dreamerv3 已应用的 `dreamerv3_FG_ADD_Gate_P0_fix.patch`。本次依据实际工作树、源码和失败回归修正，不训练模型，不改业务 Reward 或折扣。

## 1. 保留的正确修改

`Agent._availability_metrics()` 增加 action index1 的合法 ADD 数和正确识别数；posterior/prior 分别统计。训练窗口使用计数求和后计算 TP/positives，而不是平均各批次召回。FG 默认开启专项检查，其他配置默认关闭。

这堵住了“整体精度很高、合法 ADD 召回为0仍通过”的验收漏洞。它不证明当前模型已能正确预测合法 ADD，也不会自动纠正已有 Actor。

## 2. 已确认的问题与订正

### P0：第一轮真实更新返回空指标，新增强制索引崩溃

`embodied/jax/agent.py` 的训练输出异步返回：首次 `pending_mets=None`，因此返回的 `mets={}`。原补丁在 `train.py` 无条件读取 `mets['avail/post_add_positives']`，触发 KeyError；近期阶段迁移也会清空 pending metrics。

新增 `embodied/tests/test_fg_add_gate_training.py` 使用真实 train loop 与非神经网络异步指标桩，修复前失败于上述 KeyError。修复后仅在 ADD 检查启用且确已收到指标时汇总并累计 Gate 样本，首次空结果跳过；ADD 检查关闭时保留旧样本计数行为。

### P1：无足够 ADD 标签被误判为模型退化

补丁把 `positives < min_add_positives` 视为失败。Gate 一旦开启，连续样本不足也累加 regression，默认5次后中止。这些报告没有足够证据评价召回。

订正后的状态表：

| 状态 | ADD 标签不足 | 处理 |
|---|---|---|
| Gate 未开启 | 是 | 不允许开启，打断连续通过计数 |
| Gate 已开启，其他六项指标通过 | 是 | 记无法评估 ADD；不计退化，打断退化连续计数；不伪记通过 |
| Gate 已开启，其他指标实际失败 | 是/否 | 继续计退化，达到原门槛仍停止 |
| 标签足够，ADD 召回实际失败 | 否 | 按原退化机制处理 |

新增 `action_mask_warmup/add_support_sufficient` 状态指标并保存于 Gate Checkpoint。旧 Gate 字典缺该诊断字段时可读取；这不等于允许跨训练协议复用完整 Checkpoint。

## 3. 参数与统计边界

保留补丁的 `min_add_positives=16`、posterior/prior recall `0.80`，没有擅自替换为另一组数字。这是当前实验门槛，尚没有校准证据说明80%能满足择时学习需求。

positives 是被 Replay 抽样的标签次数；同一状态可多次出现，因此16不是16个独立交易机会，更不是统计置信度。门槛的业务充分性需结合已有模型诊断、误拒绝分布及固定审核样本另行评价。业务 Reward 不参与这组参数调整。

## 4. 兼容性实测

解析现有 A 先导命令后，新 Gate 配置确为启用 ADD 专项检查。新旧完整身份不同的字段为 `learning` 和 `code_hash`；实际旧 A100K 节点校验返回 `FG full checkpoint identity mismatch`。

这与隔离不同训练协议的要求一致。旧模型、账户、Replay、日志全部保留，可做只读诊断；不能将旧通过位当作新协议认证，也不能改写旧 manifest 强行恢复。原 v3 训练命令不再具有“继续旧100K正式训练”的准入保证，需要后续在新身份下准备运行清单。正式训练仍由用户启动。

## 5. 验证证据

- 新增两项失败回归在原补丁上实际失败：首次 pending metrics 的 KeyError；已开启 Gate 在样本不足时报 RuntimeError。
- 修复后 Gate、真实训练入口异步指标桩、形式准入、完整恢复/迁移回归共30项通过。
- 新增边界回归确认：无 ADD 支持时，真正变差的 prior FPR 仍能触发停止。
- CUDA GPU 实测 ADD 计数：两批数据合计 positives=5、TP=3，汇总 recall=0.6；保留稀少 ADD 分母，未将 WAIT 加入分母。
- 两处修正与原补丁的差异已经自查；没有修改 RSSM/Actor/Critic 目标、Reward、Env 行为、horizon、imag_length 或预算，没有运行新模型训练。

GPU 指标计算与非神经网络训练循环测试不等于新协议下100K模型验收。实际 A/B/C 的召回、动作概率及长期预测诊断仍需完成。
