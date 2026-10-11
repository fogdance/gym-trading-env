# FG DreamerV3：单仓位 3 手 + TP5/SL30 + Reward + Oracle Replay 综合实施方案 V1

- 日期：2026-10-11
- 文档性质：**供 Codex 实施的研究/工程任务书**；不代表已实现、不代表 Reward 最优性或训练收益通过验证。
- 建议工作分支：`feat/fg-single3-tp5-sl30-oracle-replay`（从包含已验收 R0–R2 的基线创建；先核对本地分支/未提交改动，禁止覆盖其他人的工作）。
- **唯一研究目标**：给定外部 LONG/SHORT 方向，Agent 在没有适合机会时 WAIT，在可预测的优质位置 ADD 一批 3 手，尽可能快地获得真实 **+5 点 TP**；开仓后迅速逆行 10 点要负反馈，**−30 点 SL 是最严重的单笔退出惩罚并释放唯一仓位**。无须同时持有第二批；Oracle 只是训练资料和参照，不能代替因果预测。
- 工作环境：主仓库 `gym-trading-env`；Dreamer 源码相邻仓库 `dreamerv3`。只修改已经验证必要的代码；机器产物存独立 `artifacts`；研究/审核文档一律 Markdown。

## 0. 核对过的源事实、复用边界

### 0.1 现有代码

1. `src/gym_trading_env/envs/entry_env.py`：`FixedDirectionEntryEnv` 原来校验 `(initial_balance,trade_lot,max_long,max_short)=(300000,3,6,6)`；`_preflight_open()` 限制最多两批；`_compute_stop_loss_price()` 返回 `None`，`stop_loss_enabled=False`；TP = 实际成交价有利方向 +5。Observation 为 `(market_seq(60,18),position_state(9,),action_mask(2,))`。`step()` 从上一已完成观察选择动作，并用**下一执行 bar**入场；目前退出扫描只包含 TP、主力/保证金/到期退出。
2. `src/gym_trading_env/rewards/fast_take_profit_reward.py`：正式 V1 只对完整 3 手 TP 发 `60/(60+T_natural_minutes)`。
3. `src/gym_trading_env/research/fg_reward_candidates.py`：原 V2 的 `timeboxed_tp` 和首次逆行 10 点的离线计分与状态恢复可供复用；**它不是已有 SL30 官方业务 Reward，也不是单仓位已验收代码**。
4. `src/gym_trading_env/research/fg_replay.py`、`fg_replay_coverage.py`、`tools/fg_replay.py`：已完成 R0–R2 的原 Driver、原生 Replay、哈希、Sidecar、reconcile、audit。原 P3 在第一批成交后继续申请第二批，与本项目新语义不同，必须重定义。
5. `docs/replay_contract.md`：磁盘行 `obs_t, action_t, reward_t` 中 `action_t` 是**接下来准备执行的动作**，而 `reward_t` 是**上一步执行的结果**。`stepid` 关联真实 Replay 行与 Sidecar；64 训练步 + context1 实际抽取长度 65。
6. Dreamer `dreamerv3/configs.yaml` 基线 `batch_length=64, replay_context=1, imag_length=15, horizon=333, contdisc=True`；`embodied/run/offline_pretrain.py` 现阶段调用**完整 `agent.train()`，包括 Actor/Critic**。不得把它的结果标成 World-Model-only。

### 0.2 已完成的旧研究不是新数据

- R0–R2 提交 `65da6aa` / 源 Dreamer `64c1a51...` 对原双批 6 手轨迹验证：72 条独立轨迹、1,954,516 行 Replay、1,949,908 有效采样起点、五池真实抽样 PASS；未创建 Agent、未启动 GPU、未做梯度更新。
- 旧 6 手 Replay 满仓比率 96%–99%，无法通过单纯 Reward relabel 转成新单仓位；**只复用软件基础设施，不复用其中的账户轨迹作为新训练集**。
- 原 FG409/all manifest content hash `ffc45db584106b7c348b7f3aee95096acbcb560c31a7d67fa340f30d2e10009a`；18 个训练合约，原训练截止 2024-04-08。执行前核验实际 manifest、合同边界和时间戳；不能将不同 Fold 或 sealed Test 混入。另有用户原单仓位研究方案 `FG_Single_Position_Entry_Reward_Research_Plan_V1.md` 与 V2/V3 离线验收报告，作为本任务参考，而非已冻结参数。
- 手工 LONG/SHORT 历史方向是本研究的**条件输入**；如果其来源含事后知识，只能报告 `conditional_direction_only`，不声称具备实际可部署方向预测能力。

## 1. 不变与变化的业务协议

| 项目 | 单仓位 V1 新协议 | 说明 |
|---|---|---|
| 合约 | 中国 FG 期货、原历史合约/方向清单 | 原始行情和手续费/乘数规则保留 |
| 初始账户 | 300,000 CNY | 与旧基线一致，独立策略独立账户 |
| 每次 ADD | 3 手 | 只开一批；一次活动 `entry_id` |
| 最大仓位 | 3 手 | 持仓时 ADD mask 必须为 0 |
| 动作 | WAIT=0、ADD_3=1 | 不引入 Agent 手动 CLOSE/止损动作 |
| TP | 成交价有利方向 +5 点，自动平仓 | TP 时间以自然分钟计分 |
| SL | 成交价不利方向 −30 点，自动平仓 | 必须是**真实可模拟成交**，不能只在评分器扣分 |
| 第一次逆行 −10 | 实际成交价为基准，前5/6–15根有效bar分级 | 每个 `entry_id` 至多一次 |
| 持仓超时 | 不额外新增时限强平 | 480自然分钟仅是V2止盈**奖励**归零，不是强平 |
| 主力切换/资金风险 | 保留原 Env 的真实执行与终局 | 与正常 TP/SL 分开记录和评分 |
| Env 1024-step chunk | 默认暂保留 | 不可冲掉财务仓位、SL pending、首次逆行标记；存储 chunk 不是同一边界 |
| PBRS / V3容量 | `alpha=0`、`lambda_capacity=0`（初轮） | 只为排除混杂，不宣称以后永远不用 |

### 1.1 Observation 尽量少改，但必须显式版本化

- `market_seq (60,18)`：沿用特征算法，严格不得使用未来 bar。
- `action_mask (2,)`：空仓且可成交时 `[1,1]`；有仓位/不合法 `[1,0]`。区分模型决策时 mask 与下一bar实际拒单。
- `position_state (9,)`：第一版尽量保持 shape 9，以减少 Dreamer 输入端改动；固定仅一个仓位槽，第二槽三个数全部为零。`position_ratio` 建议按**3手最大容量**归一，空仓0、满仓1，不能仍用旧6手作为分母而未注明。记录 TP 目标距离、仓位年龄，并**核对是否需要纳入 SL 剩余距离**：如 SL 固定为成交价−/+30，可由现有成交方向、TP距离和当前价推导则不额外扩形状；若不满足 Mark/价差机制，要给出证明或升级 Obs schema。
- 所有 shape/语义、mask、Reward 与成交模型分别记录版本哈希；**不得加载原6手 Obs/Value/Checkpoint 当成语义等价**。

## 2. Phase E：Env / 撮合 / 账户的真实规则（先实施）

### E0：实施前代码审计

1. Codex 先读取旧 Env、PositionManager/`_long_close`/`_short_close`、`_price`、费用/保证金、`FastTakeProfitReward`、原 SL 通路及 R0–R2 Driver。输出文件/函数级改动清单，确认已有公共模块能复用的部分。
2. 单仓位版用**单独的配置schema/严格校验**驱动；由于旧 `fg.yaml`/派生快照带有“必须 max6、无 SL、V1 Reward”的校验，必须连同主配置/快照生成与 `config_hash` 一起升级。不要绕过校验或只改某处常量后让配置仍宣称6手。
3. 不修改旧双仓位研究配置及冻结 artifacts。必要时把旧、 新限制参数化，且测试证明旧路径行为逐笔不变；若风险较大，则新建明确的单仓位 Env adapter 并最大限度复用原会计原子操作。

### E1：决定性撮合规则

以实际**开仓成交价**为 TP/SL/−10 相对距离基准，LONG 与 SHORT 镜像。仍采用下一执行bar开盘保守撮合。

- **已持仓，在同一有效1m OHLC内只触TP**：遵循既有 TP可成交/Volume/价格规则，按成交协议平仓。
- **只触SL**：若价格可执行且量足，通过原子 `settle_exit(reason="STOP_LOSS")` 完整成交；不允许把触价立即假设成成交。
- **同bar同时触TP+SL**：OHLC 无法提供先后顺序，**默认保守 SL 优先**，写 `ambiguous_tp_sl=true`；单列仅作敏感性对照的 TP 优先分支，不把乐观分支混入正式 Replay。
- **开仓同bar**：价格从bar开盘之后的路径未知。原 TP 必须经当bar收盘确认；SL同bar需采用公开、固定、保守的OHLC可执行协议（可先按风险优先确认），但不能虚构 tick 顺序。相关歧义必须记录，并用合成OHLC回归覆盖。
- **隔夜/跳空穿越SL价**：若下一开盘已处于比SL更差的位置，以保守的可执行价格及费用结算，记录`gap_slippage_points`；不得永远假设恰好在SL价平仓。
- **Volume不足/原子拒单**：不得生成虚假TP/SL、资金回收或Reward；SL触发可形成 `pending STOP_LOSS`，在后续符合成交规则的bar继续尝试，优先于新入场；对pending期间继续不利行情不隐藏账户亏损。资金/主力/到期强退的优先级必须明确并有测试。
- **原策略其它退出**：MAIN_ROLL、CONTRACT_EXPIRY、MARGIN_CALL、DATA_END/GAP和右删失分开；不把非TP/SL统一打0并直接清除持仓。
- 同样行情、相同方向、相同账户状态/动作/随机流，应产生完全一致成交与账户；不允许 Oracle 走另一套更乐观的撮合器。

### E2：账户与边界不变量

- 持仓数量只能 0 或 3 手；同一时间 `len(active_entries)<=1`；不合法 ADD 绝不会成交；自动平仓释放唯一名额。
- 账本借贷守恒、入场/退出唯一ID、成交数量、费用、滑点、保证金、权益轨迹全部对账。止损失败未成交不得提前释放名额。
- 1024 Env 截断不平仓，续接后下一观察的重复时间戳不能被计算成新增决策、额外Reward或风险K线；原生 Replay chunk 与此分开。
- `export_state/import_state` 包含 entry、事件、早期-10 flag、SL-pending、成交归因与RNG；任意中断后恢复结果和不中断逐步完全一致。

**E-Gate：** 对 LONG/SHORT、正常TP、正常SL、同根双触达、同根入场、跳空、零成交量、部分成交/拒单、主力退出、数据尾、重复fill_id、跨1024恢复建立回归；与真实训练期至少一个合约的逐步双实现对账 PASS。失败则禁止 Replay/Oracle 发布。

## 3. Phase R：Reward 数学定义与事件重标记（并行研发，发布前过 E-Gate）

### 3.1 保留已经研究过的 TP 速度曲线

令 `T` 为真实入场成交到完整TP成交的**自然分钟**：

\[
G(T)=\begin{cases}
60/(60+T), & 0\le T\le60\\
[60/(60+T)]\cdot[(480-T)/420],&60<T<480\\
0,&T\ge480.
\end{cases}
\]

关键观测点：5m→0.923077；15m→0.800000；60m→0.5；120m→0.285714；240m→0.114286；480m→0。**5/15分钟不是自动另发固定Bonus**。需要检验用户预期的5/15窗口在真实数据中是否有判别力，不要事后改公式迎合图形。

### 3.2 首次快速逆行 10 点

定义 `k` 为实际入场后**首次**不利触及10点的有效1分钟执行bar序号，入场bar算第1根：

\[
E_{10}(k)=-\beta\cdot \begin{cases}
1,&1\le k\le5\\
\rho,&6\le k\le15\\
0,&k>15\text{ 或未触发}.
\end{cases}
\]

- 同一笔交易最多触发一次。若先触10点、后反弹TP，仍保留已发生的负反馈；不要以事后TP删除该事件。
- 在同一bar既触首次-10又触SL时，明确顺序与评分：风险事件可以与更严重的SL同时发生，但不要重复发两次-10；OHLC顺序不确定需记录。
- 风险触发 `k` 使用有效交易bar；自然分钟 `T` 包含休市。两套时钟不混用。

### 3.3 已成交 SL30 是最大单笔事件惩罚

\[
E_{SL} = -L,\qquad L>\beta\ge0.
\]

正常非事件步 `r_t=0`，正式事件重标记按**真实反馈时点**发布：

\[
\boxed{r_t=\mathbf1_{TP_t}G(T_t)
-\mathbf1_{first\_adverse10_t}\beta w(k_t)
-\mathbf1_{SL_t}L + r_{other\_exit,t}}
\]

SL事件可与早期-10惩罚叠加，所以**最终交易累计负分可能低于−L**；“最大惩罚”特指最严重的**单个事件**。对 MAIN_ROLL 等非标准退出必须先记录真实数量和盈亏，再冻结独立评分：不能把负的强退当成免费释放仓位，也不能凭空奖励未成交退出。右删失标记为未完成，不假装交易已损失/盈利。

### 3.4 候选参数是可检验假设，不是结论

| 候选 | 速度TP `G` | `beta` | `rho` | `L` | 研究问题 |
|---|---|---:|---:|---:|---|
| C0 | 原V2 | 0 | 0.4 | 3 | 只有TP速度+真实SL是否足够 |
| C1 | 原V2 | 0.5 | 0.4 | 3 | 温和早期反馈 |
| C2 | 原V2 | 1.0 | 0.4 | 3 | 用户举例：5根内首次−10固定−1 |
| C3 | 原V2 | 1.0 | 0.4 | 6 | 更大SL是否加强风险识别还是使WAIT最优 |

可追加少量必要参数，但不要在同一历史区间上盲扫后声称最佳。保持 `alpha=0, lambda_capacity=0`，第一批 `H=333`。`L=6` 对5m TP `g≈0.923` 的简化胜率临界约 `6/(6+.923)=86.67%`（未计-10、期限/机会成本）；该阈值用于识别退化风险，**不是确认一切历史机会都应达到该胜率**。

### 3.5 训练反馈对齐和校准

- 原 Env 必须输出成交/提前逆行的明确事件；先保留不可变真实 trace，再派生候选 Reward 视图（同一成交轨迹可复用多个beta/L）。训练时一个 run 只允许**一个** Reward 版本；所有行重标记与原 Driver 时序逐行匹配，reward_t是已执行行为的反馈，不是同一行下一决策action_t的反馈。
- 原始 Reward、各组成、现金PnL、未折扣和折扣回报分别记录；不能用历史未折扣分数高低替代Critic/Actor训练证据。
- 统计每组 `P(TP | -10 early)`、`P(SL | -10 early)`、首次逆行时序/MAE、止盈自然分钟、TP/SL环境步、非标准退出、不同年份/方向的差异；不能仅根据分数图好看而选择beta/L。
- 对照 Always WAIT（0事件分）及闭环P1/P2/P3：评估 Reward 是否把合理的好入场系统性评价为负，是否对错误入场罚得不足；不允许奖励等待本身或人为触发奖励作弊。

**R-Gate：** 纯函数边界测试 + 与 Env 逐笔/逐步真实事件与账户对账 + 恢复/重标记幂等 + Reward版本哈希校验；由人工评审候选得分与退化风险，冻结首轮训练用 Reward 前不启动 GPU。

## 4. Phase O：先找历史最优，再用真实账户生成 Oracle Replay（重点新增）

### 4.1 分清三个角色

- **Oracle**：允许读取**TRAIN内部未来行情**、真实已指定方向和已冻结Reward，给出该历史区间的后验最佳入场路径；不能在真实实盘或验证期当作可执行策略，其收益是带完美先见之明的上界参考。
- **Causal Teacher**：用训练期 Oracle 提供的后验标签做监督研究，但在回放某一步执行动作时，只能读取此前/当前可见的`market_seq`、`position_state`、`action_mask`及已发布交易规则；不得读取future bar、oracle return、未来TP/SL等Sidecar信息。
- **DreamerV3**：训练时使用已经冻结的真实 Replay `(obs,action,reward,mask,flags)`；Oracle动作是**行为数据**，不是Actor自动收到的正确监督标签。是否增加BC损失，须等WM/Critic诊断后另行评审。

### 4.2 构造候选交易结果表（CPU）

对于 TRAIN 中每个 **合法空仓决策点** `(contract, cursor, direction)`：

1. 从该时刻当下可知的信息向**同一个已审计的新 Env**提交ADD，确认下一执行bar是否真的成交/拒单；之后固定WAIT直至真实 TP/SL/非标准退出/数据终点。
2. 记录真实入场/退出时刻、价、费用、`k`有效交易步、`T`自然分钟、首次-10触发bar、事件评分、现金PnL、max adverse、slot释放时间、最终账户。
3. 若拒单/未出场/终局强退/末尾删失，不把它默认为能够TP；按冻结协议处理。
4. 研究候选结果可使用纯函数及索引快速计算，但要通过抽样和边界全集的**第二个原 Env 完整重放**对账；不得用理想化的 `Low<=target` 代替成交。
5. Oracle 的每一个候选都只能读取训练 Fold 的价格，观察/训练输入仍然严格过去可见；全量表含未来的是**标注Sidecar**，不进入Dreamer模型张量。

性能要求：先小合约/小区间验证正确性、复杂度和内存，再批量生成；禁止每个候选都独立加载整个CSV、产生无界O(N²)重复扫描却不做性能评估。若优化结果与原Env结果不同，必须回退到原Env结果。

### 4.3 “全局最优”必须证明求解前提

单仓位、固定交易方向、外生价格和确定性成交下，可以在有足够状态信息的前提下求解半马尔可夫决策问题：

\[
V(t,z)=\max\left\{
\gamma V(t+1,z),\quad
\sum_{j=1}^{d(t,z)}\gamma^{j-1}r_{t+j}
+\gamma^{d(t,z)}V(t+d(t,z),z')
\right\}.
\]

- `z` 是会影响未来合法开仓/成交的**账户状态**（可用权益、保证金、结算结果、pending状态等）。`d` 是从决策到再次真正空仓可决策所经过的**环境步数**，不是自然分钟。边界 `V(end,z)` 必须处理未平仓/强退/删失，而非凭空设0。
- **只有先证明在本研究所有可达状态中资金约束不改变合法动作和交易结果，才允许把 `z` 省去，使用标量 `V(t)` 做精确时间DP。** 如不能证明，必须采用包含账户状态的精确方法，或明确标为`approximate_oracle`并给出误差/可行性审计。不得把贪心“挑最快TP点”称为全局最优。
- 优化目标必须单独声明：`gamma=1`（未折扣训练期累计Reward）或 `gamma=332/333` 等H版本。不同目标可能产生不同路径。冻结Reward前可运行诊断版，但**不要用Oracle产出的高分反过来挑使Oracle最高的Reward参数**。
- Oracle策略只规划本合约本训练段可见的后续，**不能延伸到validation/test**；不可从验证期将来决定训练末尾的ADD。
- 对平分方案固定 tie-break 与RNG身份；独立Brute-force短合成行情检验DP全局最优，若可用精确DP，实际Oracle轨迹按原Env再回放取得最终证据。
- 报告 `oracle_optimality_status` = `exact_proven` / `approximate_bounded` / `heuristic_only`，以及约束、搜索预算、误差界。没有证明就不贴“理论最优”标签。

### 4.4 Oracle 数据不等于好训练数据

- Oracle可能几乎全部选快速TP、规避SL，这会缩小 World Model 的失败经验覆盖；旧随机/Always ADD的亏损样本仍需保留。
- Oracle在训练历史里可后验挑选入场点，但不允许给模型增加 `future_TP_minutes`、`oracle_Q`、`oracle_label` 或 hindsight flag 等观察字段，也不能把未来信息编码到缺失值、账户随机初值、文件顺序或政策ID的**模型输入**。
- 即使没有在Obs写入未来，Oracle轨迹作为行为数据仍会改变状态访问分布，并非无偏样本；单一Oracle序列不能保证 Actor 学会择时。
- Oracle 导出的最优动作/动作价值差额可存在研究Sidecar，**第一阶段不纳入 Actor Loss**。不允许将“Oracle表现好”直接写成“Dreamer学会好入场”。

**O-Gate：** 候选单笔结果与原Env吻合、合成DP最优性检验、完整Oracle Driver重放、训练集边界拒绝、预测输入泄漏扫描、真实账户和回报逐步对账全部PASS。

## 5. Phase D：构造新的 Replay R0–R2 V2（必须包含自然对照）

### 5.1 采集策略

| ID | 执行规则（单仓位；只用当时信息，Oracle除外） | 作用 |
|---|---|---|
| P0 | Always WAIT | 零交易/价值基准，**可少量采集**；正式比较必须有 |
| P1 | 空仓且ADD合法就ADD | 自然风险/失败/止损与入场对照 |
| P2a | 空仓合法按固定 RNG `p=0.1` ADD | 主动等待与不同入场分布 |
| P2b | 同上 `p=0.5` | ADD/WAIT行为覆盖 |
| P3 | 空仓且合法时按有效分钟计数等待15根后ADD，成交/真实退出后重新计时 | 固定延迟对照，不再继续开第二批 |
| P4 | TRAIN 后验 Oracle 的已冻结动作时间表，执行时仍走原Env | 高质量历史完整轨迹，带hindsight标识在**Sidecar** |
| P5（可选、后阶段） | 用历史特征训练的因果Teacher，前向Out-of-Fold预测并在Env闭环执行 | 检验Oracle中的规律是否可预测；不能窥视未来 |

P0 不应凭行数稀释学习分布。P1/P2/P3/P4 用统一真实 Env 与独立账户/Replay stream，禁止“从各策略挑盈利片段接起来”。P4 完整记录其入场前的WAIT、入场后受限WAIT、早期逆行、TP/SL/终局。必须区分P1/P2/P3 **因果采集**与P4 **后验行为**。

### 5.2 数据分组和训练泄漏

1. 所有 Oracle 标注、Teacher fit、Reward比较均只使用明确的 TRAIN 切片。对验证/封存测试，可以**运行已经冻结的因果Teacher或Dreamer做前向执行评价**，但不允许用验证/测试未来价格生成训练动作/标签。
2. `P5` 若实施，训练 Fold 内使用**前向时间分块、跨标签期重叠purge/embargo**生成 out-of-fold 预测；用于下一个时间块的模型不能用该块之后的信息。训练末端无法确定结果的entry标签标右删失而不是提前看验证行情。
3. 多策略重复使用同一行情，是相关账户反事实路径；Unique market minutes必须单独统计，不能将4～6次重放称为独立市场观测。
4. 新manifest包含：原train manifest hash、fold、actual quote data hash、原Env/Reward/Obs/action/SL fill model版本、采集策略/RNG、Oracle目标与最优性状态、replay chunk hashes、Sidecar hashes、reconcile receipts。拒绝旧6手Replay伪装成新数据、拒绝跨版本接续旧Checkpoint。

### 5.3 Audit 必须升级的项目

- 所有64+context1样本的shape、dtype、mask、真实动作-反馈时序、跨真实Replay chunk与Env截断连续性。
- 区分`voluntary_WAIT`与`forced_WAIT`；单仓位 0手/3手占用时间、合法ADD、ADD实际请求/成交/拒单。
- TP计数、止盈自然分钟（≤5、≤15、≤60、60～480、≥480）、SL次数/实际成交损失、早期-10两档、非标准退出和未成交SL/删失；匹配每笔入场ID。
- 每个池和混合池分别抽样**真实**`batch_size=16`, `length=65`，统计训练64位置中有至少1个`legal_ADD / voluntary_WAIT / ADD_fill / TP / SL / early10`的batch占比；单独统计context位置，禁止将重复起点看作新市场证据。
- 按年份/合约/方向/日夜盘统计，报告条件比例、样本数、不同策略对同一行情的相关性、RAM/磁盘、100K/1M预算模型看到的事件分布。
- 采样比不是先拍一个数字。至少比较 `natural only` 与 `natural + P4` 两组，同等训练**更新步数、时间序列数量**；如需事件加权，保持连续序列、保存未加权natural基准、检查加权引入的分布偏差。P4独立池必须可随时关闭，不能污染R0–R2旧结果。

**D-Gate：** 校验全新轨迹、所有new policy Replay视图及全量池、逐行reconcile、固定seed重复采样、旧字段语义不串、GPU前样本覆盖报告PASS；报告中明确正/负事件的覆盖是否充分而不是只报告195万行。

## 6. Phase H：Horizon / Critic 风险审计（CPU 先行）

### 6.1 不要把自然分钟当交互步

对于每笔真实TP/SL，分别记录自然持仓分钟 `T_i`、从当前决策点到真实反馈的环境推进步 `k_i`、跨休市/跨1024/跨主力情况。Dreamer仍用其原有`gamma_H=1-1/H`。

诊断下列`H`：`333`（基线），`1000`，`3000`（诊断，不代表需要同时训练）：

\[
J_H(t)=\sum_{j\ge0}\gamma_H^j r_{t+j}.
\]

- 对一笔已明确SL，`H=333`、`k=1000`时，`−6`对较早决策的直接折扣约`−0.296`；**不能由此推出“模型已经低估SL概率”**。分析者要分开报告优化目标权重与Critic预测误差。
- 对相同**真实账户快照**，比较执行第一步ADD与WAIT，后续按**同一预先冻结的因果行为策略**继续，评价`ΔQ_H`、TP/SL比例和持续时间；这是有限政策的反事实估值，不是假装已求最优Q。
- Oracle的全局ADD/WAIT排序与相同后续策略的反事实排序含义不同，不能混在一张表里称为相同的“最优价值”。
- 检查`H=333`是否因晚SL折扣或过度偏好早TP，系统性改变原业务速度/风险偏好；**改变horizon也会改变Actor/Critic价值目标及World Model continuation目标**，不能只在报表改gamma而称重新训练。

### 6.2 1024不是原生必须长度

原Env人为1024截断与Replay物理1024存储chunk独立；单仓位/止损新实验中先按原值保持可对照，同时审计跨截断持仓及网络`is_first`含义。**禁止在本实验顺便改变1024**；若后续怀疑影响信用分配，另列单变量试验。

**H-Gate：** 每种H的折扣定义、事件步数、oracle目标版本、同状态ADD/WAIT的公平分叉可复现；至少指出边际排序差异及错误来源，不因某个H在旧历史轨迹总分较大就直接选它。

## 7. Phase M：模型训练研究（只在所有数据Gate通过后启动；分清模式）

### 7.1 首先检查现有训练入口和模式

- `embodied/run/offline_pretrain.py` 实际调用 `agent.train()`，**训练完整 Agent**。可以做“联合离线预训练”对照，但**不得宣称它只训练World Model**。
- 如果需要严格 `WM-only`，必须根据 `dreamerv3/agent.py` 的参数分组和optimizer，新增最小且可测试的训练模式，冻结Actor/Critic；编写权重/优化器/随机数状态的更新与未更新断言，不复制另一份完整Dreamer。
- 确认 Replay load 参数和长度（现有R0–R2 native Uniform需要长度65、完整capacity），不能用旧双仓位Replay混合；确认`Gate`验证的**ADD合法类召回率**，不能只看总体 Mask准确率（旧数据极度稀疏时多数类假高准确）。

### 7.2 先做最小对照，不直接大规模ABC

| 实验 | 数据源 | Reward | 训练方式 | 意图 |
|---|---|---|---|---|
| M0 | 新 P1/P2/P3 自然Replay | 冻结C* | 现有full offline或WM-only模式（明确） | 基准 |
| M1 | 相同预算的自然+P4 Oracle Replay | 同一C* | **完全相同模式** | 测Oracle历史入场轨迹是否额外有价值 |
| M2（只有必要时） | 自然+P5因果Teacher Replay | 同一C* | 相同模式 | 检验可预测Teacher相比完美先知行为的效果 |
| M3（奖励对照） | **相同Replay及更新预算** | 另一个已冻结候选C | 相同模式 | 比较Reward帮助学习的效果，不混入Replay差别 |
| M4（H对照） | **同一Replay/Reward** | 同一C* | H333 vs H1000 | Critic长期估值变化，必要时再H3000 |

顺序：**先Env+数据+CPU Reward与Oracle→小规模模型实验**，不是一次启动全部M0～M4。小规模GPU先导必须记录预算（gradient updates、实际训练采样序列、模型启动/训练seed、Replay采样seed），不同offline/online日志`step`含义不能混。用户确认后再执行GPU先导；正式ABC 1M及sealed Test需另行授权。

### 7.3 模型验收指标必须按学习机制分层

1. **World Model**：ADD合法Mask的precision/recall，过/欠预测 TP、SL、early10 的分层误差；位置转移、仓位释放、不同持仓年龄；隐状态对奖励事件的前向概率和校准（先核对 `rew` Head 能否处理当前标签分布）。
2. **Critic**：真实训练期同状态分叉ADD/WAIT策略回报`ΔQ_H`的MAE、方向一致率、校准误差，按0/3手及SL距离/持仓年龄分层；不要把事后Oracle上界当Critic必须达到的真实Q。
3. **Actor**：合法ADD时ADD概率/主动WAIT、是否All-WAIT/All-ADD、不同年份和方向的真实闭环入场时间分布；Mask Gate必须核对合法ADD召回（不能被99%禁开样本掩盖）。
4. **交易结果**：5/15/60自然分钟TP比例、TP/SL胜率、SL实际价差与损失、Reward/有资格交易日、账户净PnL及最大浮亏、回撤、风险敞口、交易频率；对照 P1/P2/P3/P5（如有），不可把Oracle后验收益当部署基线。
5. 不能使用sealed Test验证某个reward/horizon，再继续调参声称是盲测。

**M-Gate：** 即使模型Loss下降，也不得仅凭此宣布择时有效；须至少通过冻结开发验证上相对因果基线的风险/收益与行为诊断。若Oracle增加数据却不改善网络，记录为否定性证据，不预先实施Actor BC。

## 8. 统一验收、失败处理与报告

| Gate | 必交产物（拟定路径；实现后登记实际路径） | 通过标准 |
|---|---|---|
| E | `docs/FG_Single3_Env_Execution_Contract.md` + tests | 单仓位/TP/SL/原子成交/账户/边界真实逐行验证通过 |
| R | `docs/FG_Single3_Reward_Research.md` + events/parameter_scores | 原V2 TP、首次10、SL、非标准退出完整对账，候选期望/折扣/WAIT风险说明 |
| O | `docs/FG_Single3_Oracle_Validation.md` + candidate-trades/oracle manifests | TRAIN未来边界、准确/近似标签、DP短序列穷举验证、原Env路径对账 |
| D | `docs/FG_Single3_Replay_R0_R2_Coverage.md` + dataset manifest | 多策略完整真实轨迹、native Replay length65、真实batch覆盖、Sidecar及隔离审计 |
| H | `docs/FG_Single3_Horizon_Credit_Assignment.md` | H333/1000/3000折扣与真实决策分叉对照、注明优化目标和模型误差之别 |
| M | `docs/FG_Single3_Model_Pilot_Comparison.md` | 有授权的同预算模型先导；若未授权明确 `NOT_RUN` |

实验产物目录建议 `artifacts/fg_single3_v1/{env,reward,oracle,replay,horizon,model}/<run_id>`，每一份都有源码提交、config hash、行情/方向/费用/策略seed、Reward/horizon/version、逐步校验收据和产物哈希。原旧目录全部只读。

**所有自动化必须对如下异常 fail-closed：** 旧6手数据/Checkpoint误加载、新版本obs或reward混用、任何validation/test价格被用于Oracle构造、Oracle寻找未来点使用非原Env撮合或未来信息成为模型观察、Replay跨账户错误拼接、账户重置丢头寸、止损未成交直接释放仓位、Replay event/action错位、未经证明声称全局最优、GPU训练/密封测试未经批准。

### 8.1 Codex 执行顺序与提交纪律

1. 先检查本地`git status`、当前分支、测试基线和相关schema；阅读上述文件、旧R0–R2报告与Reward离线报告。计划模式输出理解与风险。如存在其它任务未提交改动，隔离文件/新工作树，不覆盖。
2. E：实现单仓位 + SL30 并测试；提交只包含必要代码/配置/tests/doc。
3. R：完成版本化 Reward 重标记与真实账户对账；CPU候选对照与报告；**涉及未确认的非标准退出评分时只报告候选，不静默固定**。
4. O：实现训练期Oracle候选结果与优化器、Brute-force小样例验证、真实Env回放。若最优性前提不满足，明确近似结果并请求用户就研究资格作决定，不能继续伪称Oracle。
5. D：适配P0/P1/P2/P3/P4，生成首个小真实合约smoke、审计，再按训练manifest全量生成；验证完整数据identity。性能如有问题先优化正确实现。
6. H：真实事件折扣和同状态反事实报告；不给出未经验证的“333必然失败”结论。
7. CPU阶段完成后，给用户**一份总报告**：版本/提交、Gate结果、每组策略的TP/SL/主动WAIT/真实收益和Replay关键事件覆盖、候选Reward风险、Oracle证据、未解决项、是否具备GPU训练条件。
8. 未得到明确授权时不运行GPU训练、不改变旧ABC/Reward/R0–R2实验、不打开sealed Test；但要把**可复现的GPU先导配置、脚本、训练矩阵和命令**准备好。用户批准后按M0→M1→必要的M3/M4执行，不在第一轮自动跑1M ABC。
9. 对已明确的规则与接口可自行实施、测试、修复、继续下一Gate；只有未预设的重大业务规则（非标准退出的评分、无法实现准确Oracle等）才提交决策。单个Gate失败禁止发布后续正式数据；允许开发诊断与部分证据交付。

## 9. 预计新增/修改文件与CLI契约（按代码核对后可等价调整）

| 类别 | 目标路径/职责 |
|---|---|
| Env | `src/gym_trading_env/envs/entry_env.py` 或新单仓位Adapter：账户限制、Mask、SL30触发/挂起/成交；尽量不改旧Env的冻结语义 |
| Reward | `src/gym_trading_env/rewards/` 新版本 + `research/fg_single3_reward.py`：事件计分、非标准事件、relabelling |
| Oracle | `src/gym_trading_env/research/fg_single3_oracle.py`：候选交易闭环结果、DP/状态优化、最优性与泄漏审计 |
| Replay | 复用/扩展`research/fg_replay.py`,`fg_replay_coverage.py`：P4及单仓位事件计数 |
| CLI | `tools/fg_single3.py`（**计划新增，现不存在**）：`env-test`,`reward-study`,`oracle`,`replay-generate`,`replay-audit`,`horizon-audit`；根据复用情况可继续使用已有CLI，但命令/配置须真实验收 |
| Config | 新 `configs/fg_entry_v1/single3_*.yaml`、`configs/fg_single3_v1/*.yaml`；从正确主配置派生，不覆盖旧yaml |
| Tests | `tests/entry_v1/test_fg_single3_*.py` + 适当integration tests；pytest targeted 真实运行 |
| Reports | Gate E/R/O/D/H 与总报告 Markdown |

### 开发命令示例（命令在实现后才可用，不代表当前仓库已经存在）

```bash
conda activate dreamerv3
export PYTHONPATH=src:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu

# 每一阶段先执行本阶段 targeted pytest，参数和入口由Codex实装后定稿。
python -m pytest -q tests/entry_v1/test_fg_single3_*.py

# 建议新增的统一入口；以 --help 可用和真实test为验收
python tools/fg_single3.py --help
```

CLI运行必须具备：`--config`, `--output`, `--seed`（适用时）, `--workers`（适用时）, `--resume`（可复用时），默认拒绝覆盖；不同研究stage明确子命令和schema，绝不猜测并调用尚未编写的CLI。

## 10. 给 Codex 的总体完成定义

**工程完成 ≠ Agent 学会。** 当前交付的达标定义是：

1. 单仓位TP5/SL30通过真实Env/账户/边界全套测试；6手老分支未变。
2. Reward快速TP、5/15根逆行−10、SL最大事件、非标准退出/删失全部有可追溯事件和版本化重标记；明确候选与尚未选定参数。
3. 至少一个 TRAIN合约的 P1/P2/P3/P4 完整真实Replay与独立重放通过；正式全训练Fold如可在资源预算下完成则发布完整manifest，不能因某个合约未完成便声称全量完成。
4. Oracle全程使用 TRAIN价格、明确最优性资格、同实际Env成交逐笔一致；未来标签未进入模型Obs；保持随机/失败经验。
5. 真实native Replay 65长度可训练，事件/可自主决策覆盖被量化；H333等折扣目标与事件步数被量化；**GPU模型学习与最终收益未做则明确NOT_ESTABLISHED**。
6. 提交可复现的命令、配置、manifest、测试收据及完整Markdown总结，由用户决定是否冻结首个Reward并启动GPU小试。

> Codex 最后应回答的核心问题：**单仓位SL30之后，历史上哪些位置事后更优、这些位置能否被当时可见信息预测、包含Oracle的Replay是否提供更多可学习的入场经验，以及在不改Dreamer算法的情况下下一步最值得检验哪一组Reward和采样条件。**
