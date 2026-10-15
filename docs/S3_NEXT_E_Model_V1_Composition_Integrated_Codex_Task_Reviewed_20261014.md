# S3-NEXT — E-Model V1 双尺度市场 Encoder 开发任务书（Composition F0 对齐版）

- **版本**：2026-10-14 / S3-NEXT-E-MODEL-V1-F1 / Review 1
- **状态**：**模型开发 GO；真实 E-Data 联调、公开解锁与 GPU 正式训练需分别验收**。
- **开发仓库**：`/home/v/Documents/work/research/{gym-trading-env,dreamerv3}`，采用 Git Worktree，不直接修改现有原始工作区。
- **已审查的 `integration/s3-next` 参考提交（来自 gy12/d312 上传归档；Codex 须再次检查本机 HEAD）**：Gym `914989b2776a87e4a7715946e266bfa8573ee412`；Dreamer `eb63de710fd693fad7ac76f797f3d9f39e331b98`。
- **基线现状**：已含 A2、Tutor-CD、B2、W&B、Archive、市场分钟/Replay/Checkpoint、Composition F0；F0 目前对 E 三组合返回 `waiting_e`，对 Tutor+B2 两组合返回 `waiting_tutor_b2`。
- **研究哲学**：探索优先、架构简单、结果驱动；**不以兼容旧 Encoder 权重或单变量消融为开发要求**。

## 0. 目标和不可变边界

让 S3 模型在**同一决策时刻**同时观察：

- `market_seq_e_shared`: `float32[60,18]`，E-Data 提供的1分钟统一价格坐标特征；
- `market_context_15m`: `float32[24,7]`，过去24根已结束并已可见的15分钟K线，历史从旧到新，7列为 `rel_open, rel_high, rel_low, rel_close, vol_relative, oi_change, valid`；
- 既有 `position_state[9]`、`action_mask[2]` 与其它合法模型观察信息。

注意 `market_seq_e_shared[60,18]` 并不是60根原始 OHLC：18列包含价位、VWAP、量仓、波动、交易时段和有效性等混合语义。所谓同一价格坐标，是指**其中依赖价格参考点的价位类通道**与15m OHLC使用一致的 `P_anchor(t)`；不可把所有18个通道当作同类价格，亦不可擅自重排18维。

把以上行情融合成一个新市场表示，送入**现有 RSSM → Actor → WAIT/ADD**。不新建交易 Gate、不人工编码双顶/压力位、不新增趋势、MACD、质量评分、时间奖励或 Outcome 目标；不修改 TP5/SL30、3手、Reward、Oracle、交易时钟和合法动作语义。

**E 是 Encoder 轴，不是第四种 teacher/mode**。教学来自已经实现的 Composition F0：`a2`、`tutor_cd`、`b2`。当完成必要集成验证后，用户通过配置选择 `E+A2`、`E+Tutor-CD` 或 `E+B2`，而不是修改 E 模型源码。`tutor_cd_b2`（含 E 联合版本）继续明确不支持，等待后续 F2。

**此次允许模型从头训练，不要求旧180K、Tutor、B2 Checkpoint 的 Encoder 权重可加载。** 新 E 模型**自身**的 FullCheckpoint exact resume 则是必测项。

## 1. 本次审查所确认的实际代码接口

| 仓库/文件 | 当前实际事实 | 对 E 的要求 |
|---|---|---|
| Dreamer `dreamerv3/fg_s3_composition.py` | `resolve_composition()` 用 `encoder` 与 `teaching`；三 E 组合状态 `waiting_e`，`require_ready()` 全路径拒绝 | **不可只修改 YAML**；待真正验收后按原解析器统一释放，不能篡改产物中的 `support_state` |
| Dreamer `dreamerv3/fg_s3_v2_agent.py` | Agent 从 `s3_v2_mode` 映射 A2/Tutor/B2，BC/Tutor/Outcome Loss 和 B2 Adapter 已有实现 | Encoder 无教学模式分支；**本轮不重写任何教学 Loss** |
| Dreamer `dreamerv3/agent.py` | `self.enc` 目前只注册 `simple: rssm.Encoder`；`policy()`、`world_model()`、`_apply_replay_context(recompute_prefix)` 都调用 `self.enc` | 新类型作为唯一可复用 Encoder 入口，避免三处重复融合 |
| Dreamer `dreamerv3/rssm.py` | `Encoder.__call__(carry,obs,reset,training,single=False)` 返回 `(carry,entries,tokens)`；现有 Encoder 用向量 MLP | E Encoder 必须严格同接口，支持 Policy `[B,...]`、训练 `[B,T,...]` |
| Dreamer `dreamerv3/agent.py` | `dec_space` 自动纳入除 `action_mask` 等少数字段外的全部 Observation | **E 要显式选择 Decoder 重建目标**，否则错误地重建旧1m与15m上下文 |
| Gym `fg_replay.py` | `FIELDS` 仅含旧 `market_seq[60,18]`、`position_state[9]` 等 | E 必须引入**版本化扩展**，不能全局修改旧 Replay FIELDS |
| Dreamer `embodied/run/fg_s3_dagger_policy.py` | Observation 字段采取严格集合校验 | E 合法字段要随版本加入；不能放开未来 Oracle/Outcome 标签 |
| Dreamer `embodied/run/fg_s3_source_transfer.py::make_agent()` | Agent ObsSpace 从 `fg_replay.FIELDS` 创建 | E 模式必须选择 E ObsSpace，不能仅仅在 Encoder 类中添加键 |
| Dreamer `embodied/run/fg_s3_source_admission.py::inspect_source()` | 已有 `direct_init` 分支，但正式 S3 Prepare 主要依赖旧来源；B2 `agent_extension` 目前限制 `warm_start` | **新 E 从头初始化的认证流程需要单独实现并实测**，不可伪装为旧参数 warm-start |
| Gym `fg_s3_v2_experiment.py::read_settings/prepare()` | `normalize_config()` 后执行来源准入与 Prepared 签名；旧 `source_checkpoint` 解析无条件走路径 | E fresh-init 需要新且明确的 source/config/seed 协议，保留旧路径行为 |
| Dreamer `embodied/run/fg_s3_v2_train.py::LiveRun` | 准备/源码/组合身份严格校验、独立真实训练及恢复 | E 新 Observation、模型配置、数据身份、Checkpoint 必须共用同一冻结版本 |

上表按**用户上传归档中真实 `integration/s3-next` Git 对象**检查；不代表已经查看到用户正在使用的本地最新未上传提交。Codex 首先复核本机双仓库 Worktree、SHA、dirty 状态和 E-Data 交付状态；若实际代码变化，先说明差异，再按不改变本任务研究目标的方式实现。

## 2. Composition：E 只占 Encoder 轴

沿用 F0 的唯一公开配置接口，**不新增 `mode: e` / `e_teacher` / 第二套 teacher 开关**：

```yaml
# 目标用法示意；不是现阶段保证可 start 的完整实验配置
schema: FG_S3_V2_COMPOSITION_CONFIG_V1
composition:
  encoder: e_multiscale_1m15m
  teaching: b2             # 或 a2 / tutor_cd
```

组合矩阵：

| Encoder | Teaching | 本任务目标状态 |
|---|---|---|
| `legacy_1m` | `a2/tutor_cd/b2` | 现有 ready；所有旧语义和 Checkpoint 保持 |
| `e_multiscale_1m15m` | `a2` | 模型+数据通过全链路验收后可选 |
| `e_multiscale_1m15m` | `tutor_cd` | 模型+Tutor 接口通过后可选 |
| `e_multiscale_1m15m` | `b2` | 模型+Outcome/Adapter+Fresh SourceAdmission 通过后可选 |
| 任意 Encoder | `tutor_cd_b2` | **保持 NOT_IMPLEMENTED**，绝不能自动退回 Tutor 或 B2 |

**解锁分两步**：

1. **M 阶段（当前可做）**：新 E Encoder、Decoder/Obs/Replay/参数/SourceAdmission 的受控 CPU、合成输入测试。正式 `prepare/start/resume/evaluate` 的 E 组合仍须因 E-Data 未验收而 fail closed。不得用配置开关或环境变量绕过 `require_ready()`。
2. **I 阶段（仅 E-Data F01–F11、真实 TRAIN、双仓端到端验收通过后）**：原单一 Composition 解析器内按**代码版本和已认证 E 数据契约**将三种 E 组合标记 ready，更新相关验收；保持 `tutor_cd_b2` 拒绝。无权擅自宣布 `READY`；完成最终解锁前须有 Review 结论。E 未就绪时 `describe-composition` 如实返回 `waiting_e`。

不要改变旧三种组合生成的 normalized metadata 或既有 Run 的 identity。不得因新代码存在就将旧 Run 视作 E 版 Run；E 新版要记录 Encoder 架构配置、Observation/Decoder schema、E-Data schema/manifest hash、两仓 SHA、teaching 模式、source kind、seed。

## 3. 推荐的唯一简单模型架构

```text
At decision t: 经 E-Data 认证的历史观察

 market_seq_e_shared [60,18]  position_state [9]   action_mask [2]
                    \             |                /
                     \            |               /
                        1m Encoder
                   (复用成熟的简单MLP)
                               | z1m
 market_context_15m [24,7]    |
          |                    |
      flatten [168]            |
          |                    |
      Linear → 128             |
          |                    |
         GELU                 |
          | z15m               |
          +------ Concat ------+
                    |
                Linear(D)
                    |
              fused tokens
                    |
              原 RSSM
                    |
              原 Actor
                    |
               WAIT / ADD
```

- 类名建议 `S3EMultiscaleEncoder`，独立文件 `dreamerv3/fg_s3_e_encoder.py`；在 `dreamerv3/agent.py` 原 `self.enc` 映射中按 `config.enc.typ == 'e_multiscale_1m15m'` 创建，Module 实际名称保持 `enc`。
- 1m 支路只消费 `market_seq_e_shared` 和原有合法账户/动作 Mask 信息；允许复用现有 `rssm.Encoder` 或等价的 3-layer MLP，**不强求参数名、形状、从旧模型加载**。不得双重输入旧 `market_seq`。
- 15m 支路 `reshape(...,168) → Linear(128) → GELU`；24根按原时间顺序排列，Mask 为第7列。invalid行前6列强制零；全24根无有效桶时上下文 embedding 置零（不让 bias 伪造市场）。不引入 Attention、CNN、序列分类、双顶检测或额外时序网络。
- 融合 `Concat(z1m,z15m) → Linear(D)`；`D` 默认复用原 Encoder token 尾维度，以保持现有 RSSM 接口简单；**不是兼容旧权重**。采用常规初始化，不做残差加0.05、小幅修正限制或 zero-last-projection。
- `initial / entry_space / truncate / __call__(carry,obs,reset,training,single=False)` 与现有 `rssm.Encoder` 契约一致；新 Encoder 无额外 recurrent state。把时间维正确恢复为 `[B,D]` 或 `[B,T,D]`。
- 数据预处理、价格归一化、量仓裁剪、15m 时间桶、真实时段、anchor 和 as-of 可见性**只由 E-Data V1 提供**，本任务绝不写第二套15m Builder。Model 只严格校验 shape/dtype/finite/mask。
- 可以正常更新 RSSM/Actor 的网络参数，但首版不改变 RSSM 动力学定义、Actor动作空间、原有 teacher/beta/reward 逻辑。

### 3.1 保留原 `self.enc` 三条真实路径

不能只测独立网络前向。必须使用同一个新 Encoder 真正覆盖：

1. `Agent.policy()` 在线选择 WAIT/ADD；
2. `Agent.world_model()` WM 优化；
3. `_apply_replay_context(recompute_prefix)` 的历史 prefix 状态恢复。

Replay 的每条历史 Observation 必须保留**该步实际可见**的 E Context 快照，不能用恢复时“最近24根15m”替换过去窗口。Replay 的 `prevact` 必须是**真实执行动作**，不是 Oracle 标签也不是未成交 Actor 提议。

### 3.2 新 Encoder 的真实参数与梯度

- 检查 Ninjax 模块注册与实际 train/policy 参数 key；所有新分支、融合层应在 `enc/...` 子树，保证 Agent/`S3V2Agent.policy_keys` 包含。不要因为类名不同而出现参数未同步。
- 检查 WM loss 对 1m 分支、15m 分支和 Fusion 的**有限非零梯度**及多步优化前后的真实变化；检查 Actor BC 参数更新（Tutor/B2用现有 Loss）。
- 保留已存在的 `stop_gradient(repfeat)` 等师生梯度隔离；不得声称 A2/Tutor BC 会直接训练新 Encoder。B2 OutcomeHead/Adapter 使用原可训练结构和独立后果标签路径，不从 Actor 观察读取未来结果。
- 不需增加 E-specific reconstruction/pair/ranking/质量 Loss；当前评估重点是架构真实可学习和后续 Actor-only 交易表现。

## 4. E Observation / Replay / Decoder 接线：这是实装，不是 Mock 之外的“后续可选”

**版本化 E Observation 契约**：

```text
legacy market_seq         float32[60,18]  # 旧模型原始特征，供旧实验及审计保留；E 不编码、不重建
market_seq_e_shared       float32[60,18]  # E 的1m Encoder/Decoder目标
market_context_15m        float32[24,7]   # E 的15m Encoder上下文；不增加重建目标
position_state            float32[9]      # 原有
action_mask              float32[2]      # 原有合法动作输入（真实字段名仍为 action_mask）
reward,is_first,is_last,is_terminal,market_advance,...    # 原时钟/训练系统字段
```

这里的 `market_seq` 是否仍物理存储在 E Replay 中由内存和审计需要决定；**不能无声改动旧特征定义或把旧市场特征作为 E Encoder 的额外输入**。推荐 E Replay 内保留旧 1m 供回归审计，并在 E Encoder/Decoder 白名单中排除；确认实际模型 ObsSpace/Decoder 逻辑与存储契约一致。必要时将审计字段存 Replay metadata 而不进 Agent ObsSpace，不能增加无用 Decoder Loss。

- Gym `fg_replay.FIELDS` 为原有公共常量：**不要就地修改**；按 E 版本定义 `E_FIELDS_V1`/Schema 选择，只在 E 新实验启用。
- Env→Collector→DAgger→Replay→Agent 的 Observation 字段、shape、timestamp、source identity 必须一致；Dagger 白名单只增加合法 E 字段，禁止 `teacher_*`、`oracle_*`、`outcome_*` 或未来 OHLC。
- `Agent.__init__` 为 E 创建独立 `enc_space`、`dec_space`（采用显式允许列表，或者审核后的白名单）；**Decoder 必须只重建** `market_seq_e_shared` 及既有允许的账户字段，不重建旧 `market_seq`、`market_context_15m` 或 Mask；reward、con、avail 损失继续走原头部。严格检验 `self.scales` 与模型返回 loss 键一致。
- CPU fixtures 构建 E policy `[B,60,18]`,`[B,24,7]`，train `[B,T,60,18]`,`[B,T,24,7]`；验证左padding、mask全0、有些有效、全部有效、时间边界和 Replay sample。
- 只用合成 fixtures 做网络集成时，不表示15m真实因果已证明。E-Data 使用 `available_at <= visible_cutoff`、交易所合法分钟聚合、同真实合约和共享 anchor 的 F01–F11 由其自身报告证明，模型联调必须复核来源记录。

## 5. 新 E 初始化、SourceAdmission 与 B2 的真实阻断点

### 5.1 禁止把不兼容旧 Checkpoint 装成新模型

旧正式源 `online_180k_source` / A2/Tutor/B2 Checkpoint 都基于旧 Encoder。E 新 Encoder 形状与 ObsSpace/Decoder 目标可以改变，不做旧权重组的暗中丢弃加载，不复用旧完整训练状态，也不把 `warm_start` 标为成功。

**本版优先采用已经存在的 `source_kind='direct_init'` 思路，为 E 创建认证的**新模型从头初始化**：显式 seed、受审查模型 config、TeacherCatalog / Oracle 来源、完整观察 Schema、代码版本、初始化模型身份；不得复用旧来源的参数哈希当作 E 来源。

实际 F0 有两处阻断必须处理：

1. Gym `read_settings/prepare` 当前默认需要旧 `source_checkpoint`，并从 `inspect_source(warm_start)` 继承原网络配置；需实现 E 专属 fresh-init 设置和**可重复、可认证**的模型 config 选择，不影响 legacy 三模式旧路径。`fg_s3_gate.target_config` 限定继承配置几乎不变；fresh-init 可以直接使用显式目标配置，但 Gate、Reward/market-minute 的合法规则仍必须通过。
2. Dreamer `fg_s3_source_admission.inspect_source` 目前限制所有 B2 `agent_extension` 为 `warm_start`。为了 **E+B2 从头训练**，需要在明确匹配 `composition.encoder=e_multiscale_1m15m`、`teaching=b2`、`source_kind=direct_init`、可信 seed/config/数据版本条件下支持新建 B2 模型及新 OutcomeHead/Adapter。旧 B2 warm-start 协议保持不变；不得移除原约束或放宽到任意无凭据的初始化。

E+Tutor 同理需要继续复用现有 Tutor extension/loss；三种 E teacher 必须最终能够分别构造、优化和 exact resume。新 E `direct_init` 路径还需检查 SourceAdmission 的 Gate 初态/参数转移和 FullCheckpoint 初始 state，不允许继承旧180K模型的 counters、optimizer 或 Replay。

### 5.2 Prepared / Run / FullCheckpoint / Evaluate

- Fresh-init 仍走现有 `inspect_source → signed Prepared → LiveRun → FullCheckpoint → evaluate` 受审查生命周期；不可新增绕过签名的小训练器作为正式入口。
- Signed identity 必须包含 E encoder type、具体编码器和 Decoder config、Observation schema、E-Data manifest hash、TeacherCatalog/OutcomeCatalog identity、training pool、两个 Git SHA、seed、source_kind、Reward/clock、Composition normalized identity；本机实际源身份证据为准。
- 新 E Checkpoint 包含新 Encoder 分支和 Fusion、RSSM、Actor、Tutor/B2专属参数、optimizer、Replay、Env/Driver、RNG、配置/数据身份证明；验证同配置短暂停训→exact resume，与连续训练的计数/状态匹配。
- 不能恢复成不同 teaching 或不同 encoder；换 15m schema/时间桶/剪裁规则必须建立新版本和新 Run。旧正式 Checkpoint 不变。
- W&B 直接使用公共 Tracker，记录 encoder、teaching、E-Data schema、双仓 SHA、Actor/Outcome/教学指标；不要重复开发上传器，也不要创建在线正式 Run。

## 6. 本次开发阶段、阻断状态和权限

**M0 — 源码事实与工作区审计**：核实本机当前 `integration/s3-next` 两仓 HEAD、Composition F0、相关 Git 分支、已有 E-Data 进度、B2 训练是否运行，输出事实报告。新 Worktree 从当前已验收的 `integration/s3-next` 基线创建；建议路径：

```text
/home/v/Documents/work/research/fg-s3-e-model-dev/gym-trading-env   # feat/s3-e-model
/home/v/Documents/work/research/fg-s3-e-model-dev/dreamerv3        # feat/s3-e-model
```

用 `git worktree add -b`，两仓分别处理；存在同名分支/Worktree 时只读审计后安全复用，不得强制覆盖。冻结当前基线 SHA 与实际 Dependency Manifest。禁止直接提交到 integration/main。

**M1 — Encoder 模块**：新 `S3EMultiscaleEncoder`、独立 E 编码器配置、严格接口/三路径、可训练参数；合成数据即可验证。

**M2 — Observation/Decoder/Replay**：E 专属字段 Schema 与选择、Encoder/Decoder目标匹配、DAgger/Replay历史快照及标签隔离；可以用 CPU fixture/Mock E provider 验证，但 Model 不能重算高周期行情。

**M3 — Composition/来源接口适配**：实现 E 配置映射和 fresh-init 受审查协议，完成 E+A2/E+Tutor/E+B2 *内部测试*、参数注册、模型更新、Checkpoint 中断恢复；**继续保留正式入口的 `waiting_e` 拒绝**直至 E-Data 全链路验收。仅限测试中可构造内部 `S3V2Agent`/测试 harness，**不能用伪造 `support_state=ready` 的生产配置绕过 F0**。

**M4 — E-Data 联调（后置）**：先审阅独立 E-Data 的 F01–F11、真实 TRAIN 可见性/量仓/价格坐标和版本化清单；完成 Env→Replay→Agent 的真实 FG CPU 端到端与3种教学模式准备/恢复验收。E-Data 尚未验收时标记 `BLOCKED_BY_E_DATA`，不能称已实现 E 的生产组合。

**M5 — 组合解锁（后置，需 Review）**：确认 M1–M4 通过后，再通过唯一 `fg_s3_composition` 能力协议将三种 E 组合正式标为 ready，继续拒绝 Tutor+B2；同步更新 Source/Prepared/Run/Eval/Tracker 相关测试、身份版本。不得通过环境变量或运行时用户字段指定“已实现”。

**正式实验**：E+A2/E+Tutor/E+B2 GPU smoke、正式1M、Actor-only DEV 及 W&B 在线实验都属于后续单独授权；不能因为 M1–M3 PASS 就启动。B2-Full 原训练与线上 W&B 记录不可改动，不抢占 GPU。

## 7. 必须逐项记录的验收（EM01–EM18）

| ID | 验证要求 | 本次通过/阻断定义 |
|---|---|---|
| EM01 | 本机真实双仓集成 SHA、Worktree、dirty、依赖和 E-Data 状态 | 报告当前值、截图/命令输出；不假装归档 SHA 是本地运行 SHA |
| EM02 | 双轴 Composition 八组合及已有三 mode 完全保持 | 旧三 ready、新 E 三 waiting→合格后 ready、Tutor+B2 两 waiting；旧签名不变 |
| EM03 | 新 E Encoder 参数、接口和 [B]/[B,T] token shape | 正确创建，`initial/entry_space/truncate/__call__` 同原接口 |
| EM04 | 1m/15m/Fusion 各真实参与前向 | 合法不同15m改变 fused tokens、RSSM state；受控 Actor logit/p_add 可受影响 |
| EM05 | 无效 Context、Padding、dtype、finite 和标签隔离 | 全无效不产生虚假15m编码；非法形状/非finite/future key 拒绝 |
| EM06 | policy/world_model/recompute_prefix 三路径 | 都使用同一个 E Encoder，历史 Replay 每步同 as-of context；真实 prevact 未变 |
| EM07 | WM梯度与优化 | 两分支、Fusion 有有限非零梯度并实际更新；BC stop_gradient 语义原样 |
| EM08 | policy 参数同步 | 新 enc 所有参数进入 optimizer、Checkpoint、policy_keys，推理与训练同版本 |
| EM09 | Decoder/WM目标 | 重建 `market_seq_e_shared` 和既有允许项，不重建旧market_seq/15m；rew/con/avail finite |
| EM10 | Env/Collector/DAgger/Replay E Observation 链路（Mock） | 版本化 Fields、shape、执行动作正确；未来教师标签不进 Actor Obs |
| EM11 | E+A2 CPU最小模型路径 | 原 BC/Oracle/Actor更新和 Critic 冻结符合设计；不需要真实数据完整训练 |
| EM12 | E+Tutor-CD CPU最小模型路径 | 原 Balanced BC/Pair Loss、诊断与新Encoder兼容，不改 Tutor公式 |
| EM13 | E+B2 CPU最小模型路径 | OutcomeHead/Adapter、调度、真实梯度/参数、预测监督独立，fresh-init合法 |
| EM14 | Fresh SourceAdmission / Prepared 签名 | 新 seed/config/schema 来源认证；拒绝旧 Encoder warm-start 与错配身份 |
| EM15 | FullCheckpoint exact resume | CPU受控连续 vs 中断恢复，模型/Opt/Replay/Env/RNG/计数一致；新E own checkpoint |
| EM16 | Legacy回归及未支持组合拒绝 | 原A2/Tutor/B2、旧C2与原测试不回归；Tutor+B2及未验收E仍fail closed |
| EM17 | 真实FG E-Data完整联调 | **待 E-Data F01–F11 / TRAIN 实测后**；因果、价格坐标、Manifest身份通过才PASS |
| EM18 | GPU短跑、DEV Actor-only、W&B online | **PENDING_USER_AUTHORIZATION**；不可用Mock效果代替真实Actor交易结果 |

**模型代码交付**：EM01–EM16有逐项测试结果；如某项因 E-Data 才能真正验证，清晰标 `BLOCKED_BY_E_DATA`，不可把 TestHarness PASS 冒充正式 `prepare/start/resume` PASS。EM17、EM18未完成时状态 `E_MODEL_CODE_READY_WAITING_E_DATA_AND_GPU_REVIEW`。严禁将 `NOT_RUN/SKIP/BLOCKED` 写成 PASS。

## 8. 风险与开发中禁止的简化

1. **最严重风险：假E接口**——只改 Encoder 或只改 YAML，Env/Replay/Policy 仍只有旧 `market_seq`；必须在 M2/EM10发现并修复。
2. **F0绕过风险**——直接把 `waiting_e` 字段改为 ready 或传 `enable_e=true` 而数据未验收；必须 fail closed。
3. **B2新初始化风险**——现有 `agent_extension` 强制旧 warm-start，E+B2无法真正从头启动；明确更改经认证的 fresh-init 分支，不能静默 drop head/adapter。
4. **Decoder数据泄漏/过量学习风险**——默认 `dec_space` 自动纳入新增 obs 导致同时重建旧1m/15m；E必须显式 Decoder targets。
5. **训练/推理参数不一致**——若新模块参数不在 `enc/` policy_keys，则出现训练变动却推理不更新；测实际 key 和改变动作分布。
6. **SourceAdmission和行情代码 hashes**——现有 ReaderCompat 仅对旧可审核历史 reader 变化成立；修改 Env/数据后不能自动扩白。新 E-Data 有独立身份、hash 和等价证明，不削弱 SourceAdmission。
7. **Decoder恢复/Reward Gate**——新模型无需兼容旧权重，但不可误改 Reward、Clock、Gate。原 Trainer 的完整冻结/恢复仍需可验证。
8. **过度设计**——禁止 Attention/Transformer、额外分类器/指标、自监督任务、新评分模型、提前的 E OFF 消融和每模式一份 Encoder 代码。

## 9. Review 结论（正式送交 Codex 前）

| 设计检查 | 审查判断 |
|---|---|
| 基于最新已上传的 `s3-next` Composition F0 源码 | **PASS**，已识别真实 `waiting_e` 守卫及命名接口 |
| 统一双尺度 Encoder + Concat/Linear + 原 RSSM | **PASS（设计）**，无不必要复杂网络 |
| A2/Tutor/B2 配置化复用，不预设 A2 | **PASS（设计）**，三种教学保持独立 |
| E Observer、Replay、Decoder 真正入模 | **需要 Codex 实施**，具体代码缺口已列明 |
| E 新模型初始化和 B2 扩展 | **需要 Codex 实施**，当前旧代码不支持 E fresh-init |
| E-Data 时间/同价/市场15m约束 | **协议明确，真实验证待 E-Data** |
| 不触碰运行中正式实验 | **硬约束** |
| 旧权重/旧Encoder兼容、提前消融 | **不要求** |
| 正式 E 模式从 YAML 直接启动 | **当前 NO-GO**，必须经过真实 E-Data + Model 一体化验收和显式解锁 |

**批准范围**：基于 `integration/s3-next` 新建 `feat/s3-e-model` 双 Worktree，完成 M0–M3 的代码开发、CPU/Mock/真实模型小步验证并提交；E-Data 如独立验收已就绪，再申请/进入 M4真实联调；M5解锁需Review，GPU/正式训练不在本次授权范围。

## 10. 给 Codex 的执行指令（可直接复制）

> 当前 `integration/s3-next` 已完成Composition F0，Archive与W&B已经合并。现在按本任务书实现 E-Model V1，不重新合并已有功能，不从旧B2/Tutor分支开发。
>
> 先核实本机两仓集成 SHA 与任务书参考 SHA 差异，创建双仓 `feat/s3-e-model` Worktree。在现有 `S3V2Agent + Composition F0` 上实现 `S3EMultiscaleEncoder`：E专用1m `[60,18]` Encoder + 15m `[24,7]` Flatten/Linear128/GELU + Concat/Linear→原RSSM；教学由配置选 A2/Tutor-CD/B2，不要绑定A2。
>
> 必须补足真实 `ObsSpace/Replay/DAgger/Decoder` 接线及 E fresh SourceAdmission（含 E+B2 OutcomeHead），执行 EM01–EM16 中可在E-Data未交付时完成的 CPU 测试；保留原旧代码/实验；任何必须靠E-Data才能验收的项目标 BLOCKED。**不要提前放行 E 的正式 prepare/start/resume/evaluate；不要跳过Composition守卫或启动GPU。**
>
> 保持原RSSM、Actor、Reward、Oracle、市场分钟、Tutor/B2 Loss 不变。提交两个仓库最终SHA、Diff、测试日志、问题清单。完成M0–M3后暂停等待Review。
