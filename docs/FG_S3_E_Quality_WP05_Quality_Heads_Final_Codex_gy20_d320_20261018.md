# FG-S3 E+Quality TP — WP05 F2 四预测头训练：Codex 正式开发任务书 V2（源码核对及反向 Review）

**日期**：2026-10-18  
**状态**：`FINAL_DEVELOPMENT_TASK / SOURCE_REVIEWED / IMPLEMENTATION_NOT_STARTED`  
**依据**：`gy20.tar.bz2`、`d320.tar.bz2` 的 WP04功能分支真实源码；上轮《WP05预设计V1》；本轮《WP04→WP05源码接口审查》；Master Roadmap V6。  
**执行目标**：实现 **F2 QUALITY_HEADS**，让来自 WP04 的已初始化完整 E+Quality 学生，使用 WP03 严格认证的 L1 标签训练 `quality_event`、`quality_touch`、`quality_adverse`、`quality_score` 四个预测头；其余网络与全部相关优化器状态严格冻结；支持完整阶段接续、恢复、诊断与独立评估。  
**不是**：重造Reader/Replay/Oracle、修改Reward、实现F3 Actor、开启Q/Soft监督、解锁Quality预测对真实下单的修正、为老E1M保留兼容路径。

## 0. 项目上下文与最新源码身份

| 项目 | 实际内容 |
|---|---|
| Gym WP04 commit | `b7d62113f1e8d4d7ddbdce175acd83568422420e`，分支 `feat/s3-e-quality-wp04-20261018` |
| Dreamer WP04 commit | `9715352d4e548b82ba24d3b31070b9fa20d345a6`，同名分支 |
| F1前驱 | Gym `7366886f...` / Dreamer `446d1bc9...` WP03；不要从旧integration HEAD开发 |
| F1已有 | `_S3V2Model`完整模型、六WM前缀真实合成更新、K步因果burn-in、external Reader FullCheckpoint、精确同Run resume |
| F1证据 | WP04自报告379个不同选定CPU测试PASS；本轮源码静态核对，未重新运行JAX测试 |
| F1缺口 | **正式TRAIN F1优化器更新0；没有已训练的正式F1权重。** F2工程开发不因此阻断；F2正式训练必须等真实F1来源 |
| 现有教材 | 旧FG201两交易日/六账户开发小样；DATA新四批是否已生成需另行取得新的真实收据，不能假定完成 |

工作流：保留当前主线分支与WP04工作树，新建**一对WP05专用worktree**，明确共同父提交。不要求先合入目前尚未更新的`integration/s3-next`。若Codex发现本机实际WP04代码已变，请先打印成对HEAD、diff及与本任务书的差异，并以真实更晚实现去重；**不要强行重做已完成模块**。不得自动`reset/clean/merge`用户在用工作树。

### 0.1 研究目标（不修改）

FG Single3：既定做空方向、空仓 WAIT/ADD，一次ADD 3手，TP+5点、SL−30点；优先尽快TP、低MAE和规避短时逆势10点；跨官方交易日TP仅降质量评分，不改变金融Env平仓。学生输入为1m、15m、严格因果的当前可见partial-5m MACD及账户状态；老师L1/L2利用完整TRAIN后缀作为教学答案，但未来字段不能进入Policy Observation或WM表征。独立Actor只能在后续WP06/08进行成绩验证。

### 0.2 开发、教材发布、正式训练分级（避免无效工程限制）

| 规则 | DEV（本任务默认） | DATA发布 | FORMAL训练 |
|---|---|---|---|
| 测试 | 定向CPU/JAX合成更新；允许必要的**小型GPU模型烟测**，不授权真实长时间训练 | 真实TRAIN生产后最终G3/Reader | 全课程/参数/Checkpoint/来源及评估门槛 |
| 机器资源 | **默认CPU、GPU、RAM完全空闲；优先提高吞吐，避免重复`free -h`、`nvidia-smi`、`lscpu`、`top`等探针** | 明确批次CPU worker预算，可高并发 | 根据已确定配置充分利用可用GPU/CPU，出现真实OOM/异常才诊断 |
| 身份 | 不必每次改动重新冻结所有SHA/重跑G3 | 产物发布须最终锁定 | 精确匹配完整来源、Run和恢复身份 |
| 出错处理 | 直接修复并重跑受影响测试 | 新收据、不可伪冒旧PASS | 按准入与研究协议处理 |
| 始终强制 | 金融Env、因果性、Teacher隔离、真实标签mask、不虚构测试结果 | 同左 | 同左 |

**优化资源使用≠可以擅自开始真实TRAIN GPU训练**。开发工具性能探针按出现异常时执行，而不是每次动作前后轮询。GPU合成烟测只是可选工程验证；默认不要求重复查询使用率。正式耗时训练在用户明确授权后执行。

## 1. 必须修改的现有源代码和需要复用的接口

| 仓库 | 现有文件/符号 | WP05动作 |
|---|---|---|
| Dreamer | `dreamerv3/fg_s3_e_quality_course_contract.py::expected_stage_contract` | **审查而不直接改写**：现有六头`TRAIN_WITH_VALID_LABEL`是有认证有效目标才可训练的上界；F2新增更严格的**四头运行期白名单**，Q没有认证目标因此冻结；保留已发布Corpus课程身份 |
| Dreamer | `dreamerv3/fg_s3_quality_v2.py::quality_extensions` | 增F2显式协议路由，不让新阶段走legacy`HEADS_ONLY`或`CONTROLLED_FULL`；预测数学尽量复用`prediction_objective` |
| Dreamer | `dreamerv3/fg_s3_v2_agent.py::_S3V2Model.__init__/training_loss_fn/train` | 增`f2`识别和训练分派。Nj creating时仍用现有完整`quality_init_loss`；运行时只走F2四头Loss/optimizer。其他模式尤其F1行为不改变 |
| Dreamer | `dreamerv3/fg_s3_v2_agent.py::masked_joint_optimizer` | 可将原F1固定WM白名单逻辑提取为**明确且受stage约束**的通用组白名单，增加逐head有效动态mask；F1原白名单调用与测试保持精确结果 |
| Dreamer | `dreamerv3/fg_s3_e_quality_foundation.py::context` | 复用单个prefix + K步真实WM预热的时间规则；F2 T段用冻结WM提取features；strict stop_gradient无梯度回WM |
| Dreamer | `dreamerv3/fg_s3_quality_v2.py::predict/prediction_objective` | 四头原生event/touch/adverse/score输出及损失，剥离或完全关闭Q的正则/计算路径；valid逐窗口，不能冒认L2/Q训练 |
| Gym | `src/gym_trading_env/research/fg_s3_e_quality_reader.py::EQualityCourseReader` | `HEADS_VALID` + `sample(B,K+T)`，只读取Trainer所需真实Replay及Sidecar；不得拼账户窗口 |
| Gym | `src/gym_trading_env/research/fg_s3_e_quality_supervision.py` | 原Source/Admission/Join/Label投影；新F2准备应指定课程stage但不改变Corpus内容 |
| Gym | `src/gym_trading_env/research/fg_s3_e_quality_foundation.py` | 重用preflight、model_config、execution_identity及F1生命周界；新增明确F2 configuration/adapter + 跨阶段handoff，不在F1函数上塞隐含分支 |
| Dreamer | `embodied/run/fg_s3_e_quality_foundation.py::CourseStream/FoundationRun` | 抽取可复用流/状态接口或新增短小`HeadsRun`协调器；底层模型、Reader、Opt、Checkpoint只有原来一套 |
| Dreamer | `embodied/run/fg_checkpoint.py::FullCheckpoint` | **同Run F2精确resume**继续使用`external_immutable_course_reader`，只增加必要的身份字段；F1→F2是显式参数handoff，不等于原FullCheckpoint.load |
| Gym CLI | `tools/fg_s3_v2_experiment.py` | 添加`f2-prepare`、`f2-probe`、`f2-read-check`、`f2-start`、`f2-resume`；正式入口守卫分离，legacy质量模式Guard不能整体放开 |

新模块命名建议（如已有等价接口应复用）：

- `dreamerv3/fg_s3_e_quality_heads.py`：F2 stage/模型目标/允许更新参数、带valid的预测Loss；
- `embodied/run/fg_s3_e_quality_heads.py`：F2 Run与同Run恢复；
- `gym_trading_env/research/fg_s3_e_quality_heads.py`：F2设置、来源认证、窗口适配、stage handoff；
- 原CLI加入F2子命令；测试仍放现有`tests/entry_v1`和`dreamerv3/tests`。

以上是**建议新模块**而非强制重复创建；禁止出现第二套Env、第二套Teacher、第二套Replay和第二套优化器数学。

### 1.1 必须保护既有Corpus课程身份的理由（反向Review新增）

实际代码`validate_course_spec()`派生`stage_contracts`，`course_identity_payload()`包含这些派生字段；WP03`inspect_source()`将其哈希与不可变Corpus中的`course_identity_hash`严格比较。直接把QUALITY_HEADS六头许可改成四头，会使**此前已认证的教材突然失效**。因此本包采用“**课程权限上界 + F2执行权限子集**”的显式分层：课程V2原声明保持，WP03对Q目标保持invalid，而本F2白名单明确只允许四头更新。对应F2配置和Run Receipt固定白名单hash，任何擅自激活Q都拒绝。这个决定不涉及旧模型兼容；它是在保持**本项目刚生成的新教材**完整可用。

## 2. F1→F2阶段接续：先做正确的数据与模型身份

### 2.1 运行划分：三条不同路径

1. **F2 ENGINEERING DEV**：使用WP04内置小模型/合成源或受控伪F1权重，验证训练和恢复。必须标记`FOUNDATION_NOT_TRAINED`；可将小型模型更新作为工程检验，绝不声称受训市场模型。
2. **F2 FORMAL START**：未来从**真实完成的F1正式Run**取一个明确、完整、认证的最终节点开始新的F2 Run。F1必须有用户授权的正式TRAIN优化器记录、可信Corpus来源、实际model size/架构及指定最终update；只有研发代码GO不够。
3. **F2 EXACT RESUME**：从本F2同Run的完整Checkpoint恢复模型、四头优化器、Reader RNG、Prefetch队列、计数、训练日程、独立校验；不是从F1重复加载。

### 2.2 模型参数如何继承

F1模型已经**完整fresh初始化**所有网络（含未来4个Quality头、两个Q及Quality Actor）。建议F1→F2：

- 验证F1完整Checkpoint的`FG_FULL_V1` manifest/字节、`run_identity.json`，F1正式训练来源和模型身份，以及当前目标网络全部参数的shape/dtype/key。只消费来源完整节点，不把阶段标识从JSON手工改成READY。
- 新建同架构、同E Observation（含partial MACD）的F2 Agent。对**每一个非`opt/`模型参数及正常模型状态**做逐键严格匹配并复制：已训练的`enc/dyn/dec/rew/con/avail`完整继承；F1中尚未训练的四预测头/Quality Actor/Q等保持它们**在F1时的初始化状态**，构成同一学生的自然连续版本。未知/缺失键一律失败。
- **所有F1 optimizer slots与全局优化器时钟不继承**，F2用新优化器状态/学习率日程从0开始；Reader按`HEADS_VALID`新抽样、RNG独立设置、`optimizer_updates`与stage counters从0开始。不得复用F1的Replay/Preset/旧n_updates。
- 在 Agent 内存中显式地复制已验证的参数（可复用sharding/parameter-transfer helper）；**禁止直接通过`agent.load(regex=...)`做F1→F2**，因为现有`load()`同时写入`n_updates/n_batches/n_actions`。
- 根据JAX Agent的`enable_policy`状态同步新训练权重和Policy mirror；旧F1的policy缓存、pending outs/metrics不能跨stage偷带。测试须验证镜像与新F2模型一致，Q/Actor仍冻结。
- 生成只读、明确的`F1_TO_F2_TRANSFER_RECEIPT`：F1 Run身份、source checkpoint和manifest SHA、源模型架构、逐键集合/哈希、接续前后非opt叶hash完全相等、F2新optim/reader/seed/代码身份；不是`same-run resume receipt`。

如果F1/源数据课程标识和当前F2合同存在版本差异，必须核实**是否只变更阶段权限声明**，还是原生OBS/Reward/网络结构确实改变。任何形状或金融语义不一致都拒绝；不能因为我们不兼容旧模型就容许不明权重迁移。F2可以从全新同一版本F1产物接续，不为历史旧E1M做转换。

### 2.3 示例F2配置字段（Codex按实际通道实现，不把此段当作已存在YAML）

```yaml
schema: FG_S3_E_QUALITY_F2_CONFIG_V1
stage: QUALITY_HEADS
sampling_stage: HEADS_VALID
sampling_spec:
  mode: NATURAL            # 或显式MIXED/STRATIFIED
model_init:
  seed: 202
  presets: [size1m]
batch_size: 16
target_steps: 64
burnin_steps: 32
optimizer_updates: 1000
report_every_updates: 100
save_every_updates: 500
device: gpu
corpus_root: /path/to/immutable/train/corpus
source_f1_run: /path/to/completed/f1_run
source_f1_checkpoint: /path/to/completed/f1_run/ckpt/step_...
prepared_dir: /path/to/new/f2_prepared
run_dir: /path/to/new/f2_run
```

数值仅是**配置示意**，不是预先认证的训练预算。正式第一阶段先根据合法样本覆盖和性能Pilot选择B/T/K，资源允许时提高吞吐；不凭示例自动启动GPU训练。

## 3. F2损失的唯一含义与有效性

### 3.1 输入/输出

`Reader.stage='HEADS_VALID'`，`sample(B,K+T)` → 原生`[B,K+T+1]`；首帧为recompute-context，随后K帧因果预热，T帧才计入Loss。K不得跨账户，当前随机窗口不继承其他窗口的隐状态。

模型执行：

```text
Reader native Replay / Teacher Sidecar
     │
     ├─ 只允许原生Observation/Action/Reward/Clock进入冻结WM
     │      E Encoder(1m+15m+partial5m MACD) → RSSM → repfeat
     │                                  stop_gradient(features)
     │
     └─ L1合法teacher targets + 逐窗口valid (仅作为Loss labels)
                                   ↓
                   event / touch / adverse / score
                                   ↓
                     F2 masked original optimizer
                     (四头每头独立有效更新)
```

原`model.wm_only()`中的Reward等WM Loss不得加入本阶段梯度目标；可以`training=False`提取表征但不执行WM optimizer。任何L1/L2值不能进入`obs_space`、Encoder或Actor logits；L2/strict动作即使存在也不得参与F2损失。保留原Quality TP Score数学定义（时间/官方日，不额外将MAE塞进Score）。

### 3.2 预测头及损失

| 前缀 | 预测形状 | 真实目标 | F2训练项 |
|---|---|---|---|
| `quality_event/` | `[B,T,7,4]` | TP / SL / OTHER / OPEN，7独立窗口 | valid-mask CE；可选仅合法事件对的mono |
| `quality_touch/` | `[B,T,4,2]` | 4窗口×2触达事实 | valid-mask BCE |
| `quality_adverse/` | `[B,T,4]` | 4窗口不利变动 | valid-mask BCE |
| `quality_score/` | `[B,T]` | 原生Quality Score[0,100]及valid | `sigmoid(logit)`尺度上的原Huber + ≥80辅助BCE |
| 两个Q头 | `[B,T,2]` | 无认证数值Q目标 | **冻结，Loss权重0，不能因缺失而补0** |
| `quality_actor/` | ADD/WAIT logits | 仅L2或以后Pair教学 | **冻结，F2不训练** |

**必须复用**`dreamerv3/fg_s3_quality_v2.py::prediction_objective()`的事件/风险/评分原生定义；如果现有函数把Q、Actor运算耦合进输出，可以抽出无副作用的`prediction_terms`公共计算，再让旧接口委托它，保留原测例。不允许复制一套具有不同系数或Score公式的新Loss。

### 3.3 严格mask语义

- `quality_event_valid[...,h]`、`quality_touch_valid[...,h,side]`、`quality_adverse_valid[...,h]`、`quality_score_valid`独立。RIGHT_CENSORED或不支持的官方日窗口是INVALID，**不是OPEN、SL或0分**；只有被真实观察到该完整窗口却未发生事件才能算OPEN。
- 在计算CE/BCE/Huber/mono前对无效位置的占位目标**进行无梯度安全化**；不修改真实sidecar原字节。无效位置不允许贡献Loss、grad或统计。包括占位NaN、非法类别索引等负例必须稳定拒绝或安全屏蔽，不得`0×NaN`污染。
- `mono`只比较双方均valid的可比较5个市场分钟窗口；官方日末与下一官方日末不能简单视为与市场分钟序列同一组单调尺度。`open_violation`是诊断，不自动变训练目标。
- 一个batch仅event有效，只有event头可发生变化；只有touch有效仅touch变化；其它同理。全头无valid：**整次训练优化器更新跳过**，不推进`agent.n_updates`、optimizer step或学习率schedule，但Reader已消费的batch应保留明确`skip_no_valid`计数并在checkpoint记录；不要悄悄重复取样直到出现正例导致采样分布隐蔽偏移。若某个头valid>0但lambda配置为0，则该头亦不能更新。
- 合法训练更新：使用显式F2四头白名单和按head有效状态mask。保存原参数/slots快照，应用后恢复所有冻结叶（注意`-0.0`字节），共享Optimizer时钟只在真实F2更新时前进；检查unknown leaves和不受控EMA/Normalizer。
- 非finite候选应**拒绝整个更新**，并恢复参数、slot和Step状态；保存可复现样本来源，不以跳过失败异常后继续报告PASS。

### 3.4 可验证的优化器权限（必须实测）

| 状态/参数前缀 | 允许变化 | 限制 |
|---|---|---|
| `quality_event/`,`quality_touch/`,`quality_adverse/`,`quality_score/` | 仅本head合法有效Loss时 | 单独params + 对应optax slots |
| `quality_q_c1/`, `quality_q_v2/` | **绝不** | F2实际执行白名单永远不含Q；上层原课程条件许可保持不变 |
| `quality_actor/`, `pol/`, `val/`, `slowval/` | **绝不** | 禁止BC、RL、Actor logits correction |
| `enc/`, `dyn/`, `dec/`, `rew/`, `con/`, `avail/` | **绝不** | 原受训市场基础完整冻结，包括BN/EMA（如有） |
| Gate、Fusion、Normalizer、未知前缀 | **绝不** | 必须覆盖全部状态key，未知fail closed |
| `opt/`相关组 | 仅四头合法slots、规定共享时钟 | F2新optimizer，绝不继承F1时钟/动量 |

## 4. F2采样、标签覆盖及课程能力评价

- `HEADS_VALID`视图支持`NATURAL`（用于原始概率基线与首选训练）、`STRATIFIED`、`MIXED`（正负暴露不足时显式采样比例）。调用WP03 Reader原有接口，不新增第二套Catalog/Index/采样器。
- `STRATIFIED`/`MIXED`必须保留实际采样模式、权重、训练曝光量；同一市场cursor在六账户中的六次反事实候选不计作六个独立市场样本。
- 评估至少设**固定TRAIN内部日期/合约切分**（若可用），无教师训练损失回传；自然分布验证至少对比分头指标，不能拿分层训练样本评估概率校准并声称真实市场概率正确。当前FG201两日小样只够smoke，缺失评价指标必须标`UNSUPPORTED`，不可补数据。
- 记录event七窗口四类TP/SL/OTHER/OPEN与valid、touch/adverse逐窗口正负、score真值与≥80数、类别不平衡、严格无效比例、唯一cursor/官方日/合约/真实账户与optimizer实际曝光。
- 学习指标：CE/logloss、合法类Brier、event混淆、touch/adverse BCE及两类都有时的AUC/PR、score MAE/Huber/校准/≥80识别、训练自然采样对照、预测范围有限性及随update轨迹。
- 完整7-window事件预测概率应单独报告未来5、15、60、120、240市场分钟和两个官方日端点；不要把它们全部混成一个单一“TP概率”。
- 多尺度诊断在F2 Forward上做read-only static ablation：MACD valid and visible、15m context以及1m信息；分支zero/shuffle只针对合法消融场景，能改变预测不等于优化交易策略。完整多合同研究留WP09。
- Prediction Research Admission保持关闭：**WP05训练好 ≠ Quality预测可修改Actor动作**；F3可在不依赖F2准入情况下独立学严格ADD/WAIT。

## 5. 开发大功能、小任务及分提交计划（A–F，共6个可review提交）

### WP05.A — F2执行权限合同与模型路由

**范围**：
1. **不修改已发布课程V2的`expected_stage_contract`**：其派生`stage_contracts`参与`course_identity_payload`并影响既有Corpus内容身份。新增F2 **`effective_trainable_groups`仅四头**与Q精确冻结的独立执行合同，并做白名单是上层条件许可子集的测试；只有真正课程语义/数据算法改变才另行决定升级Corpus版本。
2. 在现有`quality_extensions`、`_S3V2Model`加入明示`QUALITY_HEADS`/`quality_f2_protocol`分派，保留WP04 F1、legacy Quality及E+A2/Tutor/B2旧语义。
3. 提供F2显式纯配置校验、Fresh完整model-init、Reader `HEADS_VALID`绑定、Guard分级；合成入口不要引入正式TRAIN来源的强制G3。

**测试**：F2与legacy/WM stage不可混用；Q冻结在F2执行合同中显式化；Reward/Obs/MACD一致；模型大小及全部参数叶枚举；未知stage标识拒绝；无训练授权没有正式Run文件副作用。

### WP05.B — F1→F2跨阶段参数handoff

**范围**：
1. 对F1节点manifest、run身份、完整训练状态/源、model schema、课程身份及F2目标进行严格审计。
2. Fresh F2初始化后按非`opt/`逐叶复制、保留F1 Heads/Actor随机初始化（同学生）；新optimizer/RNG/Reader/训练计数，从零独立。
3. 完整F1→F2 Receipt、字节哈希与失败保护；模拟F1允许但显式`ENGINEERING_ONLY`。

**测试**：非法F1阶段、F1尚未完成、损坏node/缺key、模型尺寸不符、Reward变化、MACD/clock变化、误将F1 optimizer计数带过来、训练身份不符拒绝；non-opt参数逐字节相等，新opt slots与计数fresh且F1 node原字节不变。

### WP05.C — F2四头有效Loss与optimizer更新

**范围**：
1. 复用WP04因果K+T与冻结WM特征；读取WP03唯一Reader，不重造训练数据。
2. 对四预测头使用公共`prediction_objective`数学，valid-mask并按head动态优化器权限更新。严格禁止Q/Soft/L2/Actor项。
3. 每个head单独正/负、zero-valid、混合valid及nonfinite诊断；冻结参数和slots完整核验。

**测试**：event/touch/adverse/score各自独立有效变化；无效head零变化；全zero-valid跳过不改变optimizer clock；mono合法mask；teacher perturb只变loss不变WM表征；F1 WM/Actor/Q全部不变；非法权重/未知prefix拒绝。

### WP05.D — Reader抽样、评估及学习诊断

**范围**：
1. HEADS_VALID NATURAL/MIXED/STRATIFIED，K步burn-in；合法窗口不足明确`INSUFFICIENT_CURRICULUM_SUPPORT`。
2. 记录逐head/逐窗口的自然分母、类支持、有效损失、预测质量、Optimizer曝光、唯一市场/账户分母及训练/验证切分身份。
3. CPU内存批次读取优化、批量处理、缓存和按需JIT；**不反复轮询资源状态**；在资源空闲默认条件下优先提高吞吐，但不降低标签/账户验收。

**测试**：同seed抽样可复现、账户连续性、pair不被错误跨owner使用、后验移位负例、跨窗口删失、少数类null指标，NATURAL评估固定。

### WP05.E — F2同Run FullCheckpoint与Exact Resume

**范围**：
1. 复用`FullCheckpoint(... backend='external_immutable_course_reader')`保存完整F2 Agent/optimizer/new counters/Reader RNG/原Prefetch queue/JAX+Python+NumPy RNG/学习率与报表计划，Corpus保存在外部不重复复制。
2. 正式same-run Resume从同一Run节点恢复；source F1 manifest只作为Run静态上游身份，绝不重新执行F1→F2 handoff。
3. 冷进程做`N` vs `M + crash/restart + (N-M)` bitwise比较；变更原教师文件、Corpus、Index、模型schema、F1源节点、代码identity时拒绝伪Resume。

**测试**：连4步vs2+恢复+2，next-batch bytes/label mask/Head params/optimizer slots/所有冻结参数/学习率/计数/metrics完全一致；latest损坏拒绝，存盘不得写入Corpus。

### WP05.F — 统一CLI、验收、交付与Review

**范围**：
1. `f2-prepare/f2-probe/f2-read-check/f2-start/f2-resume`按F1入口风格实现，原`f1-*`全部保留；公开legacyE+Quality正式Guard不可被统一解除。
2. CPU小合成更新、有限可信TRAIN读取（不产生真实TRAIN优化器更新）、用户明确许可时可做GPU小模型合成smoke；选定受影响回归、失败修复日志。
3. 发布WP05_IMPLEMENTATION_REVIEW.md、WP05_F2_INTERFACE_LOCK.json、源码diff/commit SHA、所需F1来源状态、全参数前后对照、Reader/Checkpoint恢复证据，以及Master Roadmap增量。

**测试**：无用户授权`f2-start/resume`必须拒绝、development诊断不能签正式认证收据、真实训练更新0、无TEST价格读取、无自动数据批产；旧F1生效且数值未回归。

## 6. 必测验收矩阵（54个行为检查，不要求每次DEV开发都全量重跑）

| 类别 | ID | 要求 |
|---|---|---|
| 合同与阶段 | 01–06 | 四头F2执行白名单/Q frozen且原Corpus身份不变、stage分派、F1保持、课程Reward/Obs一致、F2配置负例、legacy Guard |
| Handoff | 07–15 | 来源F1完整/非工程正式资质、source/target非opt逐键相等、fresh新opt计数、源未改、shape/dtype、缺key/坏hash/伪checkpoint、镜像同步、stage-counters隔离 |
| 标签与Loss | 16–27 | 7×4 event、4×2 touch、4 adverse、score[0,100]、mask独立、right-censored非OPEN、合法mono、score>=80、有占位NaN/非法类别、Q/Soft全禁、L2仅sidecar、无未来Leak |
| 更新/冻结 | 28–37 | 每head独立梯度及params变化、单无效head params+slots不变、全无效skip step、WM全组冻结、Q/Actor全冻结、opt shared clock规范、finite拒绝/回滚、未知参数键拒绝、F1回归 |
| Reader/指标 | 38–44 | K/T+1账户时序、natural/mixed/stratified、支持不足、同cursor独立数、自然评价不混采样、分类稀有null、MACD/15m因果feature探针 |
| 冷恢复 | 45–51 | 4vs2+2 next batch、Agent+Opt+Reader RNG/Prefetch/计数、阶段来源身份、Corpus/index/代码变更拒绝、坏最新节点拒绝、no replay copy、F1无副作用 |
| 交付/权限 | 52–54 | CLI DEV/FORMAL退出、CPU回归/真实TRAIN只读、Review与Roadmap/真实SHA和NOT_RUN诚信 |

每项验收关联真实测试函数、命令退出码和必要收据。失败不能通过修改断言、填造Q目标、静默过滤invalid或把真实TRAIN更新改名为smoke来掩盖。日常开发只需定向测试，**最终WP05提交时**跑完整指定验收集合；正式训练启动另行执行FORMAL流程。

## 7. 必须在实施前做的反向Review及建议修正

**R01** 课程六头`TRAIN_WITH_VALID_LABEL`是条件上界，F2无Q认证目标，所以执行白名单只允许四头；**禁止直接改现有`expected_stage_contract`以免改变Corpus内容hash**。  
**R02** F1完成代码并非F1已训练权重：WP05可以开发但正式F2训练不得冒进。  
**R03** F1→F2跨stage复制所有非opt初始化模型参数，优化器从0；不复用`FullCheckpoint.load()`或`agent.load(regex)`。  
**R04** F2主动区分`self.f1`；Quality Configuration只识别旧阶段，必须新增无歧义新协议。  
**R05** 共用original Optax体系，但已有F1固定WM allowlist不能直接传四头；须改成stage-aware且维持F1回归。  
**R06** Head zero-valid不等于loss0就必然参数不动；Adam动量和weight decay/归一化必须精确冻结。  
**R07** 即使没有Q目标，`prediction_objective`当前也计算Q诊断；零乘非有限Q值会出NaN风险，损失提取与防护不可省。  
**R08** 原score不包含MAE惩罚，F2不得改动真值定义；早期逆行单独从adverse/风险标签学习。  
**R09** event跨官方日窗口与5/15市场分钟窗口因果有效域不同，mono不能混配。  
**R10** 开发小样六账户共同行情，重复候选不等于六倍样本；Natural评估和实际训练曝光分开。  
**R11** WP05预测结果不等于可靠入场信号，不能无研究认证就影响`quality_actor`。  
**R12** WP05与WP06是不同角色训练阶段；F3严格L2 BC不应被F2的预测资格门槛阻塞。  
**R13** 资源策略要简单：默认空闲，适度增加batch/数据预取并让GPU执行任务；只有真实OOM、设备错误或吞吐异常再排障，杜绝机械重复探针。  
**R14** 禁止偷偷把DATA另线尚未认证的四批当成可用F2正式课程；只读开发小样可继续用。  
**R15** FullCheckpoint中的Reader、RNG、原Prefetch状态不能丢，否则同Run恢复看似成功、实际抽样曝光已变。  
**R16** 没有真实正式训练就不提供模型性能GO；只记录工程GO、READ_GO、TRAIN_GO、RESEARCH_GO四轴。

## 8. 完成定义与状态

**WP05_CODE_GO** 要求：6小提交全部落地、54项行为映射测试通过、至少四头各自有效的合成参数更新且其它冻结状态逐键一致、F1→F2工程handoff合法、同Run full cold-resume精确、原F1及Reader回归、实际源码SHA与工作树状态记录。**不要求等先完成真实F1 1M训练才能做代码GO**。

**WP05_REAL_DATA_READ_GO**：有限可信TRAIN Corpus只读原生/Reader可取到四类目标，真实label分母与已存源事实相符；如果受数据规模限制，明确`LIMITED`。

**WP05_TRAIN_GO**：另需真实受训F1最终Checkpoint、跨合约/多日充分性审查、训练身份、真实G3/正式来源准入及用户授权。此时才可开展真实TRAIN/GPU长时间训练。

**WP05_MODEL_RESEARCH_GO**：未来在未被训练/调参污染的固定评价数据上，分头合法校准与效果通过事先约定标准，不能由工程loss下降推导。

**当前结论：WP05正式开发任务书 GO；正式真实训练 NO-GO；WP04工作树原样保留。**
