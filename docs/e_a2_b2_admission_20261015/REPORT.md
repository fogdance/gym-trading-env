# E+A2 / E+B2 正式训练准入报告（2026-10-16）

## Summary

结论：**GO，限定工程训练准入**。E+A2、E+B2 已分别完成当前真实 TRAIN CPU、原 size50m GPU 1K、完整 Checkpoint 独立冷恢复及正式 SourceAdmission 审查，解除这两项 `waiting_e`。E+Tutor-CD 保持既有状态，Tutor-CD+B2 仍为 `S3_TUTOR_B2_NOT_IMPLEMENTED`。正式 1M 未启动。

发布目标为双仓 `integration/s3-next`；保留真实 merge 历史，不改原已完成的 E+Tutor 冻结工作树和产物。Dreamer 发布 merge 为 `13f03938686a34b8f641c698da3198fb2d74b6cd`。Gym 本报告随最终 merge 提交，其实际完整 SHA 用 `git log -1 --format='%H %P' -- docs/e_a2_b2_admission_20261015/REPORT.md` 读取，避免在提交内写入自身 SHA。

## Evidence observed

| 模式 | 当前 CPU | 当前 GPU | 独立冷恢复 | 正式 CPU prepare / 来源审查 |
| --- | --- | --- | --- | --- |
| E+A2 | 2200 市场分钟、6 次联合更新，PASS | size50m，1000 市场分钟 / 1000 次更新，PASS | CPU 与 GPU 均 exact=true | PASS，18 个 TRAIN 合约 |
| E+B2 Full | 2200 市场分钟、6 次联合更新，PASS | size50m，1000 市场分钟 / 1000 次更新，PASS | CPU 与 GPU 均 exact=true | PASS，18 个 TRAIN 合约 |

CPU 使用 FG201、FG301，1100 分钟保存后退出，冷恢复完成至 2200。GPU 使用同两合约，500/500 保存退出，冷进程验证下一批和下一步更新完全相同，父进程从 Checkpoint 恢复至 1000/1000。保存前的 warm probe 与冷进程 probe 不计入已提交训练预算，日志尾部保留恢复对账。原始命令、工作目录、退出码与日志见 [COMMANDS.md](COMMANDS.md)、[CPU 索引](evidence/cpu_evidence_index.json)、[GPU 索引](evidence/gpu_evidence_index.json)。

| GPU 指标 | E+A2 | E+B2 Full |
| --- | --- | --- |
| minute / context15 / fusion 非零梯度范数最大值 | 207.3821 / 3.7591 / 55.2240 | 212.9451 / 3.2927 / 64.6487 |
| 三分支改变参数 key 数 | 9 / 2 / 2 | 9 / 2 / 2 |
| Actor 改变参数 key 数 | 11 | 11 |
| 有效 BC / 正 ADD 监督更新 | 1000 / 918 | 1000 / 918 |
| RL Actor / RL Critic 更新 | 0 / 0 | 0 / 0 |
| Gate CLOSED 更新 | 1000 | 1000 |
| B2 event / early / adapter 改变 key 数 | 不适用 | 5 / 5 / 1 |
| B2 head / adapter 更新计数 | 不适用 | 999 / 899 |
| 训练进程耗时（start+resume） | 687.41 秒 | 722.82 秒 |
| 单任务 GPU 显存采样峰值 | 2196 MiB | 2196 MiB |

两模式 GPU 实际并行，每组 12 个逻辑 CPU，BLAS=1，不预分配显存；四条实际 start/resume 命令均退出 0。资源审计每 5 秒采样，在训练启动之后采集；显存采样峰值不构成瞬时峰值上界。整卡采样峰值 7867 MiB，含既有进程。吞吐为共享单卡且含冷启动编译的实测，不能当作独占 GPU 的正式训练时间承诺。原始资源日志及包装程序均保留在 evidence。

## Root cause

初次 A2/B2 CPU v1 prepare 均退出 1：`E real TRAIN collection mathematical code/schema changed; rebuild required`。逐字段审查确认旧 collection 的 `entry_market_cache.py` 指纹仍是 Env SHA256 加固之前的版本；其余七个 legacy 文件及 E 数学实现一致。此拒绝来自真实身份校验，不通过修改收据或跳过验证解决。

使用既有 `python -m e_data_v1 build/verify`，从原 18 个 TRAIN 合约、488160 行行情生成独立新 collection。资源预算选 13 个 worker，用时 34.11 秒。18 个 bundle 全部 verify PASS；与旧版四种必需生产数组逐文件比较，72/72 个 SHA256 完全相同。stride15 仅用于已有 collection 审计导出，真实训练仍由原 Env 使用同一份行情逐分钟提供 E Observation。没有重写聚合/归一化公式或引入第二套正式行情 Provider。

完整数据身份：

| 项目 | 身份 |
| --- | --- |
| 新 collection | `/home/v/Documents/work/research/fg-s3-e-ab-admission-20261015/e_data_current_v1` |
| index identity | `e39ee50aa1dbdb95e41ed3d151aad27d0dbc3b5879eaca1a5c2acb28dcc01d24` |
| schema SHA256 | `a41c6d3621a56a739d18a296ecdda95d57efcdbb5254a0293a33963d4a39a874` |
| 原完整 source pool SHA256 | `4d9aa8faa5c2ad22a5664280675067c5c18e9fb22e3fee7b9af3bf0f15e6281b` |
| index 文件 SHA256 | `fd53c988132994b06b4039f7d279438db069c35c7ddabb570518dfd25f5ba42f` |
| 实际验收 Oracle identity | `ed3fe6e4f1e6eacbcfdc4d8243c37191211cd62a850a3e59e994e59274a579ef` |

每合约 bundle / 原始文件、Env/Oracle/Reward、代码、解析配置、seed 的完整身份见 [data_source_identity.json](evidence/data_source_identity.json) 和各 CPU/GPU/Formal SourceAdmission、model_config、train_pool、oracle_pool、run_identity 收据。

## Changes made

运行时代码相对冻结验收版只改变 Dreamer 两个文件：`dreamerv3/fg_s3_composition.py` 将验收后的 E+A2/B2 设为 ready；`embodied/run/fg_s3_e_source.py` 增加各自正式来源 scope，保留 Tutor 原 scope。继续拒绝合成来源用于正式训练，继续认证真实 collection、schema、源文件 SHA256、源码、配置与 seed。

Gym 增加独立命名诊断配置和两份正式 1M 配置，更新配置 README 及直接受状态变化影响的 Composition / SourceAdmission 测试。Dreamer 的两处既有合成 Checkpoint 状态断言随之更新；本次没有重新执行该合成套件，用真实 CPU/GPU 全量 Checkpoint 冷恢复作验收。

Encoder、RSSM、Actor/Loss、Reward、Oracle、TP5/SL30、Action Mask、Gate、Replay、原 FullCheckpoint、旧 B2 warm-start 认证实现与已验收代码逐字节相同。自审见 [release_self_review.json](evidence/release_self_review.json)。代码 diff 见 [Gym patch](evidence/release_gym-trading-env.patch)、[Dreamer patch](evidence/release_dreamerv3.patch)。`.gitattributes` 仅为本次原始 evidence 增加逐字节归档规则。

## Verification performed

必要公开接口回归：`tests/entry_v1/test_s3_composition.py tests/entry_v1/test_fg_s3_e_source.py`，**36 PASS / 0 FAIL / 0 SKIP**。包含八组合状态、联合教学拒绝、公开 direct_init、seed/配置/数据错配拒绝、B2 结构及正式合成来源拒绝；JUnit 和 stdout 原样归档。

正式准备使用独立冻结双仓 `fg-s3-e-ab-release-20261016`，从当前已解锁公开路径 CPU prepare；两份 prepared 均通过签名、全部不可变文件 SHA256、解析配置、当前代码和版本比对，各处 E binding 一致。来源为 seed0 / size50m / direct_init，无继承旧 Checkpoint，Gate 初始关闭、累计计数为零。未经 `--authorize-training` 的正式 start 仍拒绝。完整结果见 [formal_evidence_index.json](evidence/formal_evidence_index.json)。

| 冻结用途 | Gym SHA | Dreamer SHA |
| --- | --- | --- |
| 当前 CPU/GPU 原始验收 | `9c7defbdec12f91c6ebadcbb6585ff55f7b18ade` | `131e7ccd4ef0648c8230e088d95e2c6da991557c` |
| 已解锁正式 prepared | `a230fbe98e1c441bca4a9424bd3259ac078150f8` | `930d76446ac86fbb18b65a4f79eac263d89dafc4` |

冻结工作树不提交归档文档、不继续前进 HEAD。integration 的 merge SHA 不冒充原验收或 prepared 的运行身份。正式配置为 1M 市场分钟、每 100 分钟 4 更新，总计 40000，保存间隔 100000，replay32768、batch1、beta1；B2 保留 Full head128、event/early lambda0.1、系数上限2、delta上限2、warmup/ramp/min_event=100。

Checkpoint 最终 manifest SHA256：

| 模式/设备 | 市场分钟 | Manifest SHA256 |
| --- | --- | --- |
| E+A2 CPU | 2200 | `09770bcd499cf1f96375b30c4be91cdb27f4f3e0264bd1e6d85f5b98071e60d3` |
| E+B2 CPU | 2200 | `f34a2c084a011efe50988dee34d3c4fe4c2f0ad204c7b38bc6c0b899e8e50564` |
| E+A2 GPU | 1000 | `9305048ac1bedc30dc2992f441478df01288614846919948254587dafdb79e4c` |
| E+B2 GPU | 1000 | `615f99c01cd9b762f6910bb69dc6de6124529facc7a8c85bf48559d891b10a46` |

完整 Checkpoint 保留在各索引列出的原绝对路径，包含 agent/优化器、collector、driver/Env、Replay 分片、runtime/Sidecar、mask warmup、配置和累计计数。docs 归档 manifest、冷恢复 comparison/probe、policy/参数/优化器指纹，未复制大权重。冷进程实际下一 batch、下一步模型/优化器参数、metrics 与计数全部相同；此比较验证可复现的下一步，并非声称四次运行已经逐状态长程数学证明。

正式 prepared manifest：

- A2: `/home/v/Documents/work/research/fg-s3-e-ab-release-20261016/gym-trading-env/artifacts/fg_s3_v2/prepared/E_A2_MARKET_MINUTE_FORMAL_1M_v1_20261016`；manifest SHA256 `211d5ee450aad2cad77ca58aeaa4ad45fc307f55443a1eaaba324df443862651`；scope `REAL_TRAIN_FORMAL_E_A2_CPU_GPU_ACCEPTED`。
- B2: `/home/v/Documents/work/research/fg-s3-e-ab-release-20261016/gym-trading-env/artifacts/fg_s3_v2/prepared/E_B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261016`；manifest SHA256 `1d5e0bc5366325eff9c0793dc1e9bb5e682079ffff9d6c5a49247cfc4b26b218`；scope `REAL_TRAIN_FORMAL_E_B2_CPU_GPU_ACCEPTED`。

## Risks / not verified

本次 beta=1 为教师执行轨迹；BC 与 Encoder 确实学习，Gate 持续关闭，RL 更新均为零。Actor 提议 ADD 为 149/147，不是自主执行频率。B2 706 次混合非零 delta，argmax 改变为 0。上述不证明 Actor 自主收益或 RL 已启动。未做新的 DEV Actor-only，未读盲测 Test，未上传 W&B，未启动任何正式 1M。

全部 18 合约 collection 身份/数组验证通过，正式 pool 覆盖 18 合约；真实 Env/模型分钟运行覆盖 FG201/FG301，并非对 18 合约每个决策时刻重跑训练或完成新的全时点 15m 因果性证明。沿用已确认的因果修复；本轮无数据数学变更。

失败与取证限制均保留：

1. 两条 v1 CPU prepare 因 collection 源码身份过期退出 1；保留配置和原日志，以新 collection / v2 路径重试成功。
2. 初次数值对照错误要求旧 `--audit-1m` 可选导出文件，引发 AssertionError；改为核对四种必需生产数组，72/72 相同，没有修改生产数据或实现。
3. 一条多文件 tail 只读取证命令不适配，改用 Python 读取；普通 git add 因 docs 被忽略而拒绝，改为本次目录限定的 `git add -f`，未改 ignore 规则。
4. daemon 重启导致两条正式 prepare 原工具退出码不可恢复，记录 unknown，不写 exit0；完整 stdout JSON 与签名 manifest 的 content_hash 一致，独立签名/文件/当前代码审计通过，未重新训练。
5. 恢复后的只读来源审计有 Matplotlib 配置目录不可写提示并自动回落临时缓存，审计退出 0；未修改系统目录。

## Next recommended step

本轮验收、解锁与证据归档完成后停止，等待用户 Review。后续若选择正式 E+A2 或 E+B2 1M，使用上述冻结 release 双仓和其 prepared，保持源码/配置不变，通过原训练入口和现有 W&B `--no-evals` 流程另行授权启动。不会复用当前 integration merge SHA 伪装 frozen resume，不会覆盖历史 E+Tutor/B2 Run。Tutor-CD+B2 不在本轮可启动范围。

相关入口：[当前 EM01–EM18](EM_STATUS.md)、[模型接口](MODEL_INTERFACE.md)、[实际命令](COMMANDS.md)、[SHA256 清单](SHA256SUMS)、[原始文件来源](evidence/archive_source_files.json)。
