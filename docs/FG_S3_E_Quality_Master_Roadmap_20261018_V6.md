# FG-S3 E+Quality TP — Master Roadmap V6（WP04完成、WP05正式开发、资源规则更新）

**日期**：2026-10-18。**统一项目路线图**，继承 V1–V5 的 WP01–WP10 及并行 DATA 线；新源码基线为 `gy20/d320` 的 WP04功能分支，不是归档默认 HEAD，也不是旧`integration/s3-next`。

## 0. 大方向不变

**学生**：从零训练 Encoder(1m/15m/当前可见partial5m MACD)、RSSM、WM Heads → Quality Event/Touch/Adverse/Score → Quality Actor；不继承旧1M模型，不维护旧Checkpoint ABI。  
**老师**：Quality V2 L1入场结果/质量/逆势风险；L2在完整TRAIN后缀中按Quality TP折扣价值比较立即ADD与WAIT。后验只能产生标签、不得污染学生当前Observation。  
**教材**：WP02生成真实P0–P4、多行为正/负/中性账户轨迹；WP03进行真实三方Join、分窗口有效遮罩和不可变Reader；DATA并行扩充跨合约跨官方日课程。  
**金融目标**：按既定空单方向，WAIT或ADD 3手，TP+5、SL−30，强调快速止盈、低MAE和规避短时逆势10点；跨官方日TP仅降低质量分，金融退出条件不变。  
**最终衡量**：Actor-only在真正隔离的考试集自主择时，优质ADD/合理WAIT、风险/收益/时间/合约稳健性；不把老师预测或P4盈利当学生成绩。

## 1. 依赖图（大功能→小任务，不在WP06结束）

```text
WP01 课程+交易/观察身份 ────────────┐
          ↓                         │
WP02 Producer、E+MACD、L1/L2/P0–P4 │
          ↓                         │
WP03 认证、Join、唯一Reader          │
          ↓                         │
WP04 F1 WM基础（工程已完成）         │
          ├─────────────┐           │
          ↓             ↓           │
WP05 F2 Quality四头    WP06 F3 Quality Actor
          └───────┬─────┘
                  ↓
WP07 F4 阶段接续、受控在线学习
                  ↓
WP08 Actor-only考试/风险/稳健性
                  ↕
WP09 多合约教材/消融/效率（DATA并行）
                  ↓
WP10 研究版冻结、证据及Go/No-Go
```

F2 Prediction Admission不是F3严格动作监督的前置条件。模型学习路径不等于机器准备路径：DATA教材生产可以与WP04–WP06工程开发并行。

## 2. 截至本版真实状态（四轴与来源区分）

| WP | 大功能 | 小任务/实际交付 | 最新状态 | 下游 |
|---|---|---|---|---|
| **01** | 课程合同 | Reward/Observation/权限矩阵、Draft/正式、Guard | 完成。Gym`2904f3e`/Dreamer`347724e`；后续WP02升课程V2。**WP05需增加独立F2四头执行白名单，保护现有课程身份不变** | 全部WP |
| **02** | 学生输入+教材+教师 | E+partialMACD、P0–P4、L1/L2/数据生产器、原生质量结果 | 源码完成：Gym`e57189b`/Dreamer`44a6cc6`；有限FG201真实小样。跨合约扩量另见DATA线 | WP03/09 |
| **03** | 课程认证与Reader | 可信TRAIN来源、3-way Join、NATURAL/MIXED/STRATIFIED、精确Reader RNG、教师损失ready | 完成：Gym`7366886`/Dreamer`446d1bc`；真实小样4146行、690独立市场cursor，**数据规模LIMITED** | WP04/05/06 |
| **04** | F1 完整World Model工程 | 04.A新F1；B六WM真实更新与冻结；C诊断；D Reader+FullCheckpoint；E合成/真实只读集成 | **CODE_GO/REAL_DATA_READ_GO，TRAIN_GO NO-GO**；新Gym`b7d62113f1e8d4d7ddbdce175acd83568422420e`、Dreamer`9715352d4e548b82ba24d3b31070b9fa20d345a6`；报告379个选定CPU PASS；真实TRAIN正式权重**尚无** | WP05/06 |
| **05** | F2 四Quality预测头 | A合同与阶段；B F1→F2；C四头更新/冻结；D采样/校准；E精确恢复；F集成验收 | **本版正式Codex开发任务GO，代码未实施**；四头Event/Touch/Adverse/Score；两个Q头、Actor、WM冻结。F2真实训练需正式F1权重 | WP07，F3可并行 |
| **06** | F3 Quality Actor | 严格L2 ADD/WAIT、Natural/Balanced BC、同Owner Pair、L1质量排序、自主Argmax测试 | 待预设计；不是WP05预测结果的附属头。F3需F1基础但不需要F2 Prediction Admission | WP08/07 |
| **07** | 阶段衔接与受控在线 | F1/F2/F3→F4正式来源链、原RL/Gate、在线真实账户恢复、受控Trainer | 规划；F1→F2最小正确接续已划归WP05，不应重复开发 | WP08 |
| **08** | Actor-only独立考试 | 冻结DEV/TEST使用门槛、客观基线、TP时间/SL/MAE/成本/风险/多合约块统计 | 规划；尚无质量Actor研究结果 | WP09/10 |
| **09** | 教材扩充和消融 | DATA-A-E，多个市场时段/合约、稀缺失败题型、MACD/教师/Loss/采样消融、性能 | **DATA部分开发中，生产阻断未证明解除**；新Producer提交Gym`63f2f8c`与文档`5873d22`在独立worktree；旧记录B01失败`MACD incomplete previous bucket`、B01–B04 NOT_RUN，后续新收据待上传 | 可并行WP04–08 |
| **10** | 完整研究版冻结 | 双仓/Corpus/教师/参数/seed/实验/风险与Go-NoGo | 规划，不表示实盘部署 | — |

### WP04 真实接口落地后的WP05依赖

- `EQualityCourseReader.open(...stage='HEADS_VALID', sampling_spec=..., index_path=...)` + `sample(B,K+T) → [B,K+T+1]`。
- F1 `dreamerv3/fg_s3_e_quality_foundation.py::context/train/model_extensions`：因果K步预热、原World Model/Opt六前缀。
- F1 `embodied/run/fg_s3_e_quality_foundation.py::FoundationRun/CourseStream`：原FullCheckpoint外部课程Reader精确恢复。
- F2需要新独立stage及参数handoff；`expected_stage_contract('QUALITY_HEADS')`现允许六头**有有效标签时**训练，但WP03 Q无认证目标，WP05.A采用**四头运行时白名单+Q冻结**，**不修改参与Corpus hash的派生阶段矩阵**。不同身份、不同运行不能伪装Resume。

## 3. 学生参数学习权限（修订目标）

| 网络组 | F1/WP04 | F2/WP05 | F3/WP06 | F4/WP07 |
|---|---|---|---|---|
| E Encoder、RSSM、dec/rew/con/avail | TRAIN | 精确冻结 | 原则冻结 | 经独立准入可能放开 |
| Quality event/touch/adverse/score | 冻结 | **逐头合法mask下TRAIN** | 原则冻结 | 经独立准入可能放开 |
| Quality Q c1/v2、Soft/Q动作修正 | 冻结 | **冻结** | 冻结 | 需数值目标认证与决策准入 |
| `quality_actor/` | 冻结 | 冻结 | **TRAIN** | TRAIN |
| 原`pol/val/slowval`、Gate与RL | 冻结 | 冻结 | 冻结 | RL/Gate单独证据决定 |

WP05完成不意味着Prediction/Decision Gate开放；用户最终要的是Actor入场能力，不是漂亮的预测Loss。

## 4. 统一开发与资源使用原则（用户新增）

**默认开发机器CPU、GPU、内存完全空闲。** Codex不应重复执行`free -h`、`nvidia-smi`、`lscpu`、`top`、`ps`等例行巡检，也不要为“资源确认”占用开发回合；按任务选择CPU/GPU，允许合成批处理/并发和适度调大batch，目标尽量充分利用算力。**只有真正OOM、设备竞争、吞吐低或用户主动要求检查时才诊断**，并据实际错误针对性调整。不要在默认配置中无故将并发降到极低。对于DATA线用户已经明确的`--workers 20`预算，按该具体任务约束执行。

开发门槛 `DEV`：真实金融/因果/标签隔离必须正确；可以快迭代、直接修补并重跑定向测试、有限TRAIN只读、小合成优化器更新，不必每次全G3和重新锁定所有SHA。  
教材发布 `DATA`：真实TRAIN分批生产，最终G3/Reader/覆盖与产物身份；不能混DEV/TEST未来或把失败改标签。  
正式训练 `FORMAL`：新的完整F1来源、Corpus、模型身份和阶段授权；真实TRAIN较长优化器训练和独立研究结果另请用户授权；未经允许不自动启动长跑。  
任何阶段：**不虚构性能/测试/来源/训练成果**。

## 5. 近期执行次序与交付维护

1. 现在在WP04双仓父提交创建WP05新worktree，按《WP05 F2 正式Codex任务书V2》A–F开发；WP04原工作树保持不变。可以使用GPU做有界合成模型烟测，不自动真实TRAIN长跑。
2. DATA并行修复合法交易时段跨桶下的MACD问题，产生新的真实B01教材收据；未上传证据前仍NOT_RUN，不影响WP05工程路径。避免两个分支重复改同一 Observation/Reader接口。
3. 如果用户希望启动正式F1训练，必须先单独确定真实课程规模、目标批次、Checkpoint位置和授权，再冻结完成后启动F2真实训练。不要因为WP05代码GO就声称学生已经受训。
4. WP05新代码到达后进行WP05→WP06真实接口Review，提早形成F3 Quality Actor任务书；WP06可并行预设计，不依赖F2预测门槛。
5. 每次源码交付，在当前Master Roadmap更新**双仓HEAD、真实完工小任务、选定测试事实、未运行项、四轴GO/NO-GO、下游依赖**。旧V1–V5保留历史，不用旧状态覆盖新状态。

## 6. 版本日志

| 版本 | 变化 |
|---|---|
| V1–V3 | 确立WP01–WP10、WP03真实接口、WP04正式开发范围。 |
| V4 | WP05预设计、DATA跨合约教材提前生产计划。 |
| V5 | 独立DATA生产者与MACD故障、DEV/DATA/FORMAL规则分层。 |
| **V6（本版）** | 从gy20/d320实际确认WP04双仓完成和F1未正式训练；完成WP05 A–F正式开发计划，明确F2四头执行白名单是现有课程条件许可的严格子集、避免更改`course_identity_hash`导致教材失效；禁止F1→F2冒充Resume；明确默认资源空闲、尽量跑满、避免重复硬件探针。 |
