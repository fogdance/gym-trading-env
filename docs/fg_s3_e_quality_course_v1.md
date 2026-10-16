# WP01 E + Quality TP 课程合同和只读 Preflight

唯一课程协议是兄弟 Dreamer 中的 `FG_S3_E_QUALITY_COURSE_V1`。Gym 不复制
课程校验规则，只调用纯合同，并交叉核对原生 Reward、E Observation/可见性、
Quality Score 和公开 composition guard。没有新训练模式或平行训练器。

## 运行设计诊断

从独立开发 Gym worktree 执行：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -B \
  tools/fg_s3_e_quality_course_preflight.py \
  --spec docs/fixtures/fg_s3_e_quality_course_v1.example.json
```

默认仅向 stdout 输出 JSON。可选 `--report NEW_DIR` 仅允许合法显式草案把同一
报告写到新建的独立目录，文件名固定为 `course_contract_receipt.json`。
使用现有 `signed/write_json/fingerprint`；`content_hash` 是内容完整性哈希，
**不是身份认证或训练许可**。报告明确 `NON_AUTHENTICATED_DIAGNOSTIC`、
`authenticated=false`、`execution_admitted=false`。

| 输入 | 退出码 | 状态与输出 |
| --- | --- | --- |
| 显式 `spec_mode: draft`，结构合法，待提供生产身份明确为 null | 0 | CONTRACT_ONLY / NOT_ADMITTED；完整 blockers；可选诊断文件 |
| `spec_mode: formal`，结构和声明的元数据匹配 | 1 | CONTRACT_ONLY / NOT_ADMITTED；不写文件，WP01 不认证市场字节、Corpus、Teacher 或阶段来源 |
| 模式缺失、必要字段缺失、空字符串、空来源列表、非法路径、身份不符、伪造认证、重复 JSON 键或非法 JSON | 2 | INVALID_SPEC_OR_SOURCE / NOT_ADMITTED；不创建报告目录 |

formal 缺任何真实来源身份同样退出非零。没有自动降级成 draft 的分支。
即使所有元数据哈希匹配，也不会移除 `waiting_quality_e` 或
`S3_QUALITY_E_NOT_IMPLEMENTED`。缺 Git 来源或导入非兄弟 Dreamer 也不通过。
CLI 仅使用 CPU 环境，不加载市场、模型、Checkpoint、Collector 或 Trainer。

拒绝覆盖目录、symlink 路径、父路径穿越、正式训练位置和
prepared/run/checkpoint/corpus 目录，也不向所引用来源目录写报告。诊断阶段
不生成 `manifest.json`、`training_receipt.json`、`source_admission.json`、Prepared
或 Checkpoint。

## 合同内容与身份

示例是**设计草案**，合约、边界和 seed 是声明，未查询真实数据，也不构成 WP02
教材配比或生产窗口的批准。固定 E+Quality、direct_init、Quality TP C1 Reward
`(beta=.5, rho=.4, L=3.)`、市场分钟、单批 3 手、WAIT=0/ADD=1、TP5/SL30。
Reward 再交给现有 `Single3RewardConfig` 验证，不复制奖励公式。

E policy 输入只有原生的 `market_seq_e_shared [60,18]`、`position_state [9]`、
`action_mask [2]`、`market_context_15m [24,7]`；valid 是最后一个 context 通道。
cutoff、decision_time、数据身份属于 host provenance，不是 policy 字段。
strict-action、Score、Outcome、Soft 和 Q 的有效位/来源仅属于 TRAIN sidecar。
旧 C1 Oracle 数值和旧 1M 参数不能因为字段可读而进入新课程。

输入不能提供派生矩阵。四个阶段声明由 Dreamer 生成：

| 阶段 | 可训练组声明 | 策略与准入 |
| --- | --- | --- |
| FOUNDATION_WM | enc/dyn/dec/rew/con/avail | 无决策毕业资格，其余冻结 |
| QUALITY_HEADS | 每类有效标签对应的 Quality 预测/Q 头 | WM、旧策略和 quality_actor 冻结 |
| ACTOR_SUPERVISED | 仅 quality_actor | 无需 PredictionAdmission，预测/Score/Q 修正为 0 |
| ONLINE_ADAPT | quality_actor；其它组须 WP07 Review 和各自准入 | 选择 quality_actor；Prediction/Q/Decision 和 RL Gate 独立 |

所有阶段在 WP01 均 `training_enabled=false`，所有 logits correction 均为 0。
这些声明不映射成 legacy `HEADS_ONLY/CONTROLLED_FULL`，不更改优化器或网络。

数据合同必须完整提供 TRAIN 合约、含 +08:00 的 `[start,end)` 边界及官方日来源。
每种行为有独立 behavior_id、seed、资金、合约/时间边界。允许的行为声明是
always_wait、always_add、probabilistic、single_delay、quality_oracle_schedule；
延迟使用有效市场分钟，risk_selected 仅是对真实轨迹的采样标签。
最后一种只是未来教材规范，当前 Single3 公开 specs 没有因此开放 Oracle。

draft 必须显式把待提供的 source_manifest、market_sources、
e_profile_schema_sha256、teacher_manifest 和 Oracle source 写为 null。
有引用时，使用 `{path, sha256}`，相对路径基于 spec 所在目录；仅允许 JSON
元数据。TRAIN pool 复用原生 `pool_entries`，核对选中成员及声明的 market、Env
config、metadata 哈希，但**不读取成员行情文件**。Teacher/Oracle 引用仅验证
原生 content_hash 和声明的 TRAIN/Reward/Score 绑定；不代表语料/标签认证。
时间边界、官方日和市场字节仍明确未认证。

`course_identity_hash` 使用现有 fingerprint 对规范化课程及全阶段矩阵计算。
排除诊断 spec_mode；有序列表、seed、源哈希和配置值改变会改变课程身份。
JSON 键顺序不影响课程身份，原始字节哈希 `spec_input_sha256` 单独记录。
同一输入、源码状态下重复报告的哈希和文件字节相同；不放时间戳或输出路径。

## 后续接口与迁移边界

WP02 复用 `validate_course_spec` 和 `course_identity_payload`，冻结新的课程身份，
另行生成并认证 E-native TRAIN 小样。WP03 认证 Teacher/Sidecar；WP04–WP07 实现
真实阶段更新、FullCheckpoint 和签名参数 handoff。它们都未在 WP01 实现。
新课程必须使用新 Run；不能作为旧来源的 resume 或参数迁移许可。

WP01 不修改金融 Env、Reward、E 计算、公开 composition、SourceAdmission、
Quality Admission、LiveRun 或 FullCheckpoint。原 legacy Quality 和
E+A2/Tutor/B2 保持其现有规则，CPU 回归通过不等于研究或训练 GO。
