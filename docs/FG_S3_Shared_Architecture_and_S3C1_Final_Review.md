# 共享架构与 S3-C1 整体代码 Review

> 交付后复查新增 3 项未修复 P2 问题，当前收尾状态见
> [问题清单与复现](FG_S3_Post_Delivery_Review.md)。下文 PASS 保留其历史执行范围。

## Summary

两份正式任务书的 AR0–AR5、C0–C4 已完成 CPU 工程验收。新增 S3-C 数学、
标签身份、模型继承与完整恢复合同一致；无第二套 Oracle 索引或 C 私有迁移器。
正式 S3-C GPU、DEV、封存 TEST 和生产收敛结果均为 NOT_RUN。

本报告与 `FG_S3C1_Implementation_and_Verification.md` 以及
`artifacts/fg_s3c1_v1/receipts/c_final_acceptance_summary.json` 共同核对。
Dreamer 仓库的收据位于兄弟 Gym 仓库。

## Evidence observed

| 版本 | Gym | Dreamer |
| --- | --- | --- |
| 原研究基线 | d374e93 | bb1d123 |
| 共享架构初始功能 | 47c76cd | 2b61ea3 |
| C 原始实现 | 24eeb42 | 21fa6f9 |
| C 依赖收尾 | dd03a9d | b382cd9 |
| 当前公共在线来源扩展 | 5293e978d3f234d5f7c3149f326f1da831012bad | 93a6fb65b493b60f7c1205b8351d7024cbfc326a |

远端 `feat/fg-single3-tp5-sl30-oracle-replay` 已更新并核对：Gym 1ba0a8c
经合并提交 92873be 进入开发分支；Dreamer 09e2e5c 已是当前分支祖先。
唯一冲突是配置 README 两端新增章节，保留 S3 章节和上游 Gate 校准章节。
没有引入 C 模型/损失/stream/recovery 文件的改动，不重复此前通过的测试。

真实 Catalog 共 488629 P4 行，合法 402096，ADD 16171、WAIT 385925；
所有有效行与原 sparse 标签双向对账，所有排除原因完整覆盖。
Manifest SHA 为 47736742ff81e337d10585b86d0709f5fd1bff8d082b01324b267f9f1761f0de。

## Root cause

四次公共接口回访均基于实际输出：Replay 首行合成 reset；完整 Replay 文件集
准入；producer 的完整依赖清单；在线节点的 FG_FULL provenance 与环境步 / 模型
更新单位。修复均在 AR 公共模块，没有增加 C 私有标签判定或参数拆卸接口。

原 V1 A 来源固定 62500，用户明确选择其他来源后，现有 S3 管理入口增加 V2
来源准备，仍调用原 Actor BC。当前最终来源为指定的在线 180k 节点，而非此前
讨论的 10k。旧准备、数据、缓存、Checkpoint 和收据均保留，不重新签名。

## Changes made

共享合法性来自 TeacherCatalog；C 只调用 attach_teacher / SourceAdmission /
完整参数迁移 / native Replay admission。BC 读取当前 posterior repfeat，
经原 feat2tensor 后 stop_gradient，仅额外更新原 pol。原 base loss、网络、
scales、joint optimizer、slowval、update counter 与 carry 路径保留。

监督为所有合法 P4 行的等权 NLL 按 mini-batch 有效行数平均。安全替换 mask / y
仅用于无效 BC 行，真实 Observation / Replay mask 不改写。lambda=0 静态关断。
Gate 关闭时有效 BC 计数为零，不能将此阶段描述为已进行了有效 RL+BC。

S3-C 新实验迁入整个 Agent 与原 joint optimizer、Gate/window、norm/counters；
A→C 使用已更新 Actor 与保留的原联合优化器，独立 BC optimizer 不导入。
恢复只读取本 run FullCheckpoint，SourceAdmission 不再次覆盖已训练模型。

## Verification performed

| 任务书项 | 核对内容与执行证据 | 结论 / 证据层级 |
| --- | --- | --- |
| C1 | 真实全量 P4 Catalog、原 sparse 标签、排除原因；mixed join | PASS：真实 TRAIN 数据与公共合同 |
| C2 | direct_init/direct_full/after_bc_s3a；源码、来源、模型 schema/hash 篡改拒绝 | PASS：真实 CPU 神经模型；正式 A 大包未训练 |
| C3 | teacher 仅 ext-space；policy 拒绝；native Replay 无 teacher 字段 | PASS：神经/stream 测试 + diff |
| C4 | 65→64；consec=0/>0、prefix、reset/last/terminal、prevact | PASS：实际原模型和 Consec |
| C5 | ADD/WAIT、held/no-trade/invalid direction/Natural；±1 shift/action 不符拒绝 | PASS：全量 Catalog + 反例合同 |
| C6 | 无效行先安全 mask/y；无有效行 loss/额外梯度为零且有限 | PASS：实际原 masked distribution |
| C7 | P(ADD)=0.3 与 WAIT 的预期 NLL，包含真实 unimix | PASS：原 Actor 神经计算 |
| C8 | BC 梯度仅 pol；其余模块 BC grad=0；原 WM/RL 梯度保留 | PASS：原网络梯度与联合更新 |
| C9 | CLOSED 仍有 WM 更新、BC 有效更新=0；真实 controller 后开启 | PASS：实际关闭来源 + 受控 Gate 测试 |
| C10 | lambda=0，三来源、两种 Gate，同 batch/carry/PRNG 的完整状态等价 | PASS：真实小模型；重复不增计覆盖数 |
| C11 | keys/shapes/scales 不增；一次 opt/slowval/update | PASS：继承原 train + 参数/计数证据 |
| C12 | direct_init 无 A 包也可 1–2 次 CPU joint 更新 | PASS：受控 READY；真实 TRAIN 小模型 |
| C13 | 真实 A atomic 包的 Actor、non-pol/joint opt、policy/计数继承，BC opt 丢弃 | PASS：真实 A20迁移；另有 READY atomic 夹具2步 |
| C14 | full bundle 与坏 bundle；原 Gate/norm/opt/updates；在线 180k 单位 | PASS：小模型 + 真实生产来源 CPU 全参数迁移 |
| C15 | N=3 对 K=1+restore+2；stepid/teacher/params/opt/carry/queue/sampler/Gate | PASS：实际 FullCheckpoint/RuntimeState/Prefetch |
| C16 | 同 run 恢复不再 admit/transfer；新来源/预算/lambda/代码变化拒绝 | PASS：实际 managed resume 与拒绝测试 |
| C17 | valid/add/wait、zero-batch、Gate/BC 交集、Always-WAIT | PASS：真实 CPU receipt；仅 TRAIN imitation |
| C18 | join→JAX→train→report→CP→restore，float32/BF16 | PASS：4种真实 TRAIN 小模型 CPU CLI；16命令退出0 |
| C19 | A/B算法、标签、特征、旧CP身份与原工作区不覆盖 | PASS：Git/source/hash核对；host进程未由沙箱验证 |
| C20 | 正式 C GPU、DEV、封存 TEST、交易表现 | NOT_RUN；没有静态/合成结果冒充生产通过 |

上表编号对应任务书第7节20条要求，不等于20个独立生产测试。
原 C 20个独立测试案例；17+4+5是执行批次，有重复。后来新增 A 公共来源与
在线计数合同，所有列出的 XML 共22个不同名称，重复执行不新增覆盖。

关键收据：c_final.xml、ready_lambda_resume_final.xml、final_dependency_c_regression.xml、
s3a_shared_source_regression.xml、online_counter_contract.xml；real_cpu_cli_deps_v2_commands.json；
real_dtype_restore_deps_v2.json；online_180k_neural_transfer_receipt.json。
所有这些文件均在 Gym artifacts/fg_s3c1_v1/receipts。

4个实际 CPU run 均2次更新，合法行196（ADD1、WAIT195），Always-WAIT为195/196。
float32/BF16 READY run 有效 BC=2；继承真实原模型/A20 CLOSED 的 run 有效 BC=0。
恢复逐项比较 saved Agent、两队列/current、carry、sampler、Gate、计数、scheduler、reducers。
新在线来源实际 CPU 神经载入286个参数/状态条目全部字节一致，无来源模型梯度训练。
在线计数合同的合成序列化夹具单独标注，不算神经或生产测试。

早期失败保留为 raw/XML：错误测试 key/Ninjax context、无使用模块的 grad target、
运行代码变化触发的 tamper 断言、缺少 portal/cloudpickle 的 inventory。后续替代执行
通过；失败不隐藏、不计入 PASS。源码变更后的旧 run 不修改 Hash 来绕过恢复。

## Risks / not verified

没有执行正式 S3-C GPU、DEV、封存 TEST、收益评估、生产 C 完整梯度更新。
真实生产来源 CPU transfer 是参数继承验证，不是 C 生产收敛验收。
当前用户确认来源模型仍训练中；只复制已完成180k节点，不操作进程或原目录。
旧 C CPU 收据绑定 dd03a9d/b382cd9；新来源扩展保留其数学和恢复文件字节，但旧
code identity 不允许被当作新版本原地 resume。需要新 run 时重新 prepare。

## Next recommended step

用户按 FG_S3A_Online_180K_Startup.md 启动已准备的 S3-A。其BC预算为1000，
源21183原模型更新与180000环境步保留为来源记录，BC从0另计。
未来 C 可以 direct_init/direct_full/after_bc_s3a；不代选后续顺序。
