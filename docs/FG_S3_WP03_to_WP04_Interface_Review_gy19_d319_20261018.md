# FG-S3 WP03 → WP04：独立源码接口审查（gy19/d319）

**审查日期：2026-10-18。性质：只读静态接口审查 + WP03已有证据核对；非独立GPU/真实行情复测。**

## 1. 最新双仓锁

| 仓库 | 已交付的WP03功能分支HEAD | 相对WP02的提交 |
|---|---|---|
| Gym | `7366886f3b40bf74aa4485978ffcc97e39d0d53e` | `bb0c9a3`→`39b8348`→`98bdb40`→`7366886` |
| Dreamer | `446d1bc99debd2d05c313eafd7d0c414ebd2815c` | `446d1bc` |

均为 `feat/s3-e-quality-wp03-v2-20261016`。注意Git归档当前指向旧 `feat/fg-s3b-certified-bc` HEAD；上述WP03是归档中存在的**命名功能分支 ref**，是本次提取和核对的源码目标。`integration/s3-next`还落后，不得把归档默认HEAD或integration误作WP03基线。归档只携带Git objects，本次通过 `git archive <WP03 ref>` 提取所需源码，没有更改用户仓库。

已检查归档中 `docs/WP03_IMPLEMENTATION_REVIEW.md` 与 `docs/receipts/wp03/{D_release_final,D_native_G3,D_real_reader_final}.json`。记录表明：177项选定CPU测试pass、真实TRAIN FG201两官方日六账户G3重验、三方Join、Reader与fresh Agent Loss smoke已运行（真实本机证据，不代表本次环境重新运行）。报告保留 corpus 内容哈希和真实来源限制：4146 Replay行、690 unique market cursors、119真实成交、投影 strict ADD171 / WAIT1279，基础教材**覆盖 LIMITED**。原始Corpus不在本次归档中。

当前容器缺少 `ninjax`、`embodied`、`elements`、`gymnasium`，因此无法在这里重跑模型CPU测试；对关键8个源码模块做了独立Python AST解析，均通过。不能把报告中177 PASS冒称本次独立执行。

## 2. WP03真实对外接口（替代预设计假设）

- `src/gym_trading_env/research/fg_s3_e_quality_supervision.py`：
  - `SupervisionSource.from_corpus(root, stage='FOUNDATION_WM')`；`inspect_source(source)`；`QualitySupervisionAdmission.preflight(source, purpose='development'|'native-verification', producer_gym=..., log_path=...)`。
  - 返回 `Admission(source, receipt, native_result)`；development状态 `DEVELOPMENT_VALIDATED` / `NOT_ADMITTED`；仅在本次实际执行G3时是 `SUPERVISION_DATA_READY`，但 `training_admitted=False`。**它证明数据，不授予训练权限。**
- `src/gym_trading_env/research/fg_s3_e_quality_reader.py`：
  - `build_join_index(admission, path)` 构建外部 SQLite 元信息索引，原始大数组仍在只读Corpus。
  - `EQualityCourseReader.open(admission, *, stage, sampling_spec, seed, index_path, cache_bytes=...)`。
  - `stage`为**抽样语义** `WM_NATURAL | HEADS_VALID | ACTOR_STRICT | ACTOR_QUALITY_PAIR`，不是课程阶段 `FOUNDATION_WM`；F1对应 `WM_NATURAL`。
  - `sampling_spec`为 `{'mode':'NATURAL','policies':[...可选...]}` 等；F1基线采用NATURAL，后续混合采样需独立实验身份。
  - `sample(batch_size,effective_length)` →字典张量 `[B,effective_length+1,...]`；第0帧上下文，后续真实执行帧有观察、动作、反馈及L1/L2投影；`stepid`为 `[B,T+1,20]`；`consec`和市场时钟一并返回。
  - `coverage()`区分unique market、账户决策和重复抽样曝光；`state_dict()` / `load_state_dict()`提供Reader RNG、索引/来源/阶段/采样合同、覆盖计数的恢复；`close()`关闭SQLite。
- `src/gym_trading_env/research/fg_s3_e_quality_checks.py`：`cpu_model(length,output)`通过**内部** `_S3V2Model`、原 `_make_agent` 构建fresh真实网络；`cpu_forward`调用 `_apply_replay_context`、`training_loss_fn`、`quality_loss`与 `wm_only`，但**不调用优化器**。这是可用的CPU只读smoke范本，不是WP04训练器。
- Dreamer `dreamerv3/fg_s3_v2_agent.py`：`_S3V2Model.wm_only(carry,obs,prevact,training)`和 `Agent.world_model` 可复用；`S3V2Agent` 公共构造仍 `require_ready()`，阻断未认证E+Quality；`quality_loss`在 `HEADS_ONLY` 阶段以 `stop_gradient(base)*0`冻结WM。
- Dreamer `dreamerv3/fg_s3_e_encoder.py`：输入5字段 `market_seq_e_shared[60,18]`, `market_context_15m[24,7]`, `market_macd_5m[4]`, `position_state[9]`, `action_mask[2]`；解码目标仅1m共享序列与position，并非MACD/15m直接重建。

## 3. WP04真实阻断及必须修改的旧方案假设

**B1 — 还不存在F1训练实现（确定）。** `quality_extensions`只接受`HEADS_ONLY/CONTROLLED_FULL`；原 `training_loss_fn` 在quality mode绑定 `quality_loss`；`_S3V2Model.train` 当前没有F1专用分支。不能把WP03 Loss Forward检查写成训练完成。

**B2 — Reader窗口没有完整历史RSSM carry（确定）。** `EQualityCourseReader.sample`从同段随机取 `[T+1]` 行，在`batch`的`consec`数组上没有填写实际连续carry标志（保持零初始化）；`Agent._apply_replay_context`在 `recompute_prefix`/K=1时从初始hidden重构**仅一个**前缀obs，再训练余下部分。窗口不跨段≠模型获得了从真实episode起点的全部记忆。WP04必须实现并清晰记录可认证的有限历史burn-in/截断规则；不得悄悄携带其他随机样本的hidden，也不得声称完整episode后验。建议扩展现有Reader为F1带warmup的窗口和显式有效loss mask，复用原 Agent，禁止第二份Replay；可用受验证的固定K有限上下文，明确这是模型近似，不是完整历史。工程首版也可增加全episode自起始训练对照。

**B3 — FullCheckpoint不能直接绑定WP03 Reader（确定）。** `embodied/run/fg_checkpoint.py::FullCheckpoint.save()`硬取`self._saveables['replay']`，其 `save_replay()`依赖在线Uniform Replay的chunks/locks。WP03却是外部不可变Corpus、SQLite索引和Reader RNG；不得假装它是Uniform Replay。WP04应在现有FullCheckpoint内新增严格的只读外部Reader状态保存分支，保存Reader state+索引/Corpus hash而不复制Corpus，并涵盖训练carry、opt参数、步数与随机源；旧在线Replay逻辑保持原样。若选择新checkpoint适配层，必须证明它复用相同full-state身份、原子写入和恢复校验，不能创建低配模型-only存档。

**B4 — 冻结 optimizer leaf 的实现需扩充（确定）。** `masked_joint_optimizer`按前缀启闭，在Quality旧路径仅HEADS_ONLY冻WM；未知前缀默认激活。F1必须反转为**六组WM显式白名单**：`enc/ dyn/ dec/ rew/ con/ avail/`，所有 `pol/ val/ slowval/ quality_*`及非WM状态冻结，并比较每类参数、优化器动量/weight-decay状态及Ninjax非参数变量，不能只靠把BC/Pair lambda设0。`opt/step`整体共享时钟可前进，但须明示。

**B5 — 参数布局初始化不能只调用wm_only（确定）。** `training_loss_fn`在 `nj.creating()` 下quality模式借 `quality_init_loss`构建全部网络（这一路内部调用 `self.loss()`，可能构建RL相关子模块）；一次性初始化必须在训练步开始前完成，之后纯F1更新禁止再算Actor/Critic的imag/loss。F1 checkpoint应包含未来F2/F3所需完整参数布局，避免切阶段时误将随机新网络伪装为原学生。

**B6 — Stage字面值不同（确定）。** Corpus admission stage= `FOUNDATION_WM`；Reader sampling stage=`WM_NATURAL`；Quality legacy stage=`HEADS_ONLY`；F1新执行stage=`FOUNDATION_WM`。Codex不得无验证直接复用这些不相同的枚举。

**B7 — Reward事件观测与目标（确定）。** 现有WM `reward` 来自真实Replay `obs['reward']`，与L1/L2严格后验标签相互独立；`losses`实际有`dyn/rep`、Decoder两项、`rew`、`con`、`avail_post/prior`等，按已创建Agent真实scales枚举，不能把WM六个参数前缀当作六个loss字段。真实市场事件稀疏、重复账户行情等可造成低loss假象；需分事件验收。

**B8 — 真正数据与授权身份（确定）。** WP03真实G3仅在开发机运行，Corpus未随归档传输；真实source receipt不等于run Prepared，formal guard仍关闭。WP04研发期CPU合成更新允许按用户之前已批准的代码开发测试范围；实际真实TRAIN更新/GPU运行仍另需明确批准。

## 4. 反向Review与 GO/NO-GO

- **WP03→WP04 CODE INTERFACE：GO**，以上接口可据实编写F1任务书；条件是实际WP04 worktree从一对WP03提交开始。
- **WP04现阶段真正WM TRAIN：NOT_IMPLEMENTED / NO-GO**。
- **WP03的小样 DATA：报告证据GO、覆盖LIMITED**；本次未独立复核原Corpus字节。
- **正式E+Quality训练：NO-GO**，不解除原公共prepare/start/resume/evaluate Guard，不读取TEST。

**要求下一次审查：** 附WP04五组提交SHA、完整源码diff、训练/冻结逐参数哈希、Reader与Checkpoint恢复证据、有限历史burn-in测试、课程身份与原市场回报审计。单凭loss有限和CPU Forward不够。
