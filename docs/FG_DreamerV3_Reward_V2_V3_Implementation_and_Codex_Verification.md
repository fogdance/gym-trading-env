# FG Reward V2/V3：实现交付与 Codex 验证说明

状态：**工作机 E1/E2 验收完成，36 条轨迹 × 32 组参数通过，详细报告已生成；未接入正式训练 Env**。下方保留交付说明，最新实施记录见末节。本任务没有修改原 `FastTakeProfitReward`、`entry_env.py`、DreamerV3、Gate/Checkpoint/Replay。

## 1. 包内可执行代码

| 路径 | 作用 |
|---|---|
| `src/gym_trading_env/research/fg_reward_candidates.py` | V0/V2/V3 逐分钟状态化 Reward 计算核心；stdlib-only |
| `tools/fg_reward_offline.py` | JSONL 轨迹计分；真实 Env 原动作回放；无需模型的 CPU 策略闭环 |
| `tests/unit/test_fg_reward_candidates.py` | 公式、风险窗口、仓位、恢复、无效输入与 PBRS 测试 |
| `examples/fg_reward_synthetic_events.jsonl` | **纯人工构造**的端到端执行样例，不是实际市场数据 |

已经实现的计算包括：

- **V0**（原 Reward）：实际 3 手 `TAKE_PROFIT` 成交事件 `60/(60+T)`，用于与现有 Env 日志对账。
- **V2**：`G(T)` 时间分段止盈（自然分钟 `T>=480` 时为 0）；每个 `entry_id` 首次反向 ≥10 点，在入场后的第 1–5 根有效行情分钟罚 `-beta`，第 6–15 根罚 `-beta*rho`，其余不罚；加可开关的 `alpha*(gamma*Phi_next-Phi_now)`。
- **V3**：V2 加 `-lambda * eligible_market_minute * (active_slots/2)^2`。容量按实际成交后至本根分钟结束前的真实 3 手批次名额计；空仓 / 3 手 / 6 手分别是 0、0.25、1 个基础成本单位。
- 每一步返回拆分分量 `v0,tp_timeboxed,early,potential,capacity,capacity_basis,v2,v3`；记录逐笔事件、歧义、右删失及折扣累计诊断。所有参数尚未冻结，CLI 示例值**不是推荐参数**。

### 奖励公式

\[
G(T)=\begin{cases}
60/(60+T),&T\le60,\\
60/(60+T)\cdot(480-T)/420,&60<T<480,\\
0,&T\ge480.
\end{cases}
\]

\[
r_{V2}=G_t-\beta w(\tau^-_t)+\alpha(\gamma\Phi_{t+1}-\Phi_t),\qquad
r_{V3}=r_{V2}-\lambda q_t(n_t/2)^2.
\]

所有 TP/风险事件按当时 K 线和**已经提交的真实成交**计算，不接受离线虚构的盈利事件。OHLC 内同时支持触达 -10 与完成 TP 时标记 `intrabar_ambiguous`；候选计分采用保守“风险也计入”的处理，需报告敏感性，不能推断分钟内确切顺序。

## 2. 安装与最小执行

代码拷贝到 `gym-trading-env` 对应的相对路径，使用项目原有 Python 环境：

```bash
# 不安装第三方包也可以执行数学单元测试
python -m unittest discover -s tests/unit -p 'test_fg_reward_candidates.py' -v

# 仅合成轨迹 smoke，绝不是历史绩效
python tools/fg_reward_offline.py \
  --trace examples/fg_reward_synthetic_events.jsonl \
  --beta 0.2 --rho 0.3 --lam 0.01 --alpha 0.05 \
  --out /tmp/fg_reward_demo_001
```

### E1：复算 A 的真实交易动作（不调用模型）

使用实际 FG409 开发验证 Env 配置、冻结实验 manifest 和 A 的评估产物：

```bash
python tools/fg_reward_offline.py \
  --env-config /PATH/FG409_validation_env.yaml \
  --manifest /PATH/experiments.json \
  --eval-dir /PATH/A/eval \
  --beta 0.2 --rho 0.3 --lam 0.0002 --alpha 0.05 \
  --out /PATH/new_reward_research/A_v2v3_eval
```

脚本读取原 `policy_actions.csv`，在同一现有 `FixedDirectionEntryEnv` 中逐分钟重新提交原动作，获取原 Env **实际** entry fill 与完整结算 attribution；逐步校验 V0 与 Env 当根 Reward 一致，并比对原 `fills.csv`、`reward_events.csv`。有不一致立即报错，不输出伪造的“成功对账”。不加载 Actor，不运行新推理，不修改原数据与 Checkpoint。

`--manifest` 在真实 FG 研究中强烈建议提供，以调用原 `validate_validation_inputs()` 校验开发期环境及冻结输入 SHA；未提供时，输出显式标记 `frozen_manifest_verified=false`，不得宣称完成冻结数据验收。

### E2：从原 Env CPU 闭环运行无模型策略

```bash
python tools/fg_reward_offline.py \
  --env-config /PATH/FG409_validation_env.yaml \
  --manifest /PATH/experiments.json \
  --policy always_add \
  --beta 0.2 --rho 0.3 --lam 0.0002 --alpha 0.05 \
  --out /PATH/new_reward_research/always_add

# 另分别运行 --policy always_wait、random、delay5、delay15、delay30；
# random 必须固定 --seed，并重复多个 Seed。
```

首次可使用相同的开发验证合约（FG409），后续拓展开发期其它合约。**禁止**为参数选择访问冻结盲测价格。

执行结束输出到全新目录（已经存在的目录会拒绝覆盖）：

- `step_components.csv`：每分钟完整 V0/V2/V3 奖励分解、占用成本、歧义及 entry_ids。
- `trade_components.csv`：逐 `entry_id` 的风险首次触发、累计容量基础量、完成或右删失。
- `snapshot.json`：可验证 SHA 的评分器完整状态，跨 1024 chunk 不会重置事件标志。
- `summary.json`：V0、V2、V3 总分、折扣诊断、TP 计数、输入哈希及原 Env 真实账户 summary（如果使用 Env 模式）。

## 3. 实现的严格边界

- 研究计算器**不定义撮合**。真实成交和完整三手 TP 只能来自原 Env 的 `fills` 与 `attributions`；JSONL 模式调用方要对原始事件真实性负责。
- 风险的 `5/15` 使用**有效分钟 K 线根数**，TP 的 `60/240/480` 使用**自然分钟**。休市分钟不积累容量成本，零时长记录不计入有效行情分钟。
- 按 `entry_id` 保存早期 -10 只处罚一次；允许超过 480 分钟继续持仓；不产生固定止损和新的平仓动作。
- `chunk_end` 只用于日志标记。独立 `FGRewardScorer` 不在非终局 chunk 重置 `active`、`seen`、`potential` 或计数。
- PBRS 理论的望远镜抵消只在**与 Dreamer 相同的折扣/连续性**条件下成立；本计算器的 `gamma` 是统一近似常数，非终局 chunk 的实际模型 Bootstrap 是否连续仍需 Codex 在 Dreamer 侧审计。
- 参数单位是无量纲候选评分，**尚无最佳 beta/rho/lambda/alpha 的实证证据**；长期占用成本与 480 分钟零奖均改变旧业务目标，必须防范 Always WAIT。
- 只做离线评分/回放；**没有**向正式训练动作或训练 Reward 接口注入 V2/V3。不能因此声称 GPU 实验已运行或新 Reward 能改善择时。
- 真正账户的手续费、PnL、浮亏、保证金、未结束仓位由原 Env 提供，不能用候选 Reward 替代；完整 E1/E2 风险报告由 Codex 汇总原账本指标。

## 4. 给 Codex 的验收任务（先验证现有实现，不要重写）

1. 将随包源码放到当前新分支相同相对路径，`git diff` 确认仅新增研究文件。
2. 运行全部单元测试并进行实际项目依赖环境中的一次合成轨迹烟雾测试；不需要 GPU。
3. 对开发期真实 FG409 的 A 先导执行 E1。核对 75 批真实入场、73 批 TP、2 批非 TP；逐步对账 V0 与原 `FastTakeProfitReward`。如数据或配置缺失，明确 FAIL/BLOCKED，不要编造结果。
4. E2 运行 Always ADD / WAIT / Random / Delay 闭环，在相同冻结开发合约、费用、方向、仓位/成交协议下计算 V0/V2/V3。参数敏感性至少包含 beta=0、lambda=0 和弱中强三档，记录源哈希、随机 Seed、每个候选的有效交易日分母。
5. 专项审计两个容易误解的实现取舍：风险与 TP 同一分钟的 OHLC 次序歧义，以及区间容量成本按入场后、结算前计费是否与当前 Env 一致。两者若不匹配，以实际 Env 协议为准修正 adapter，并新增回归测试。
6. 输出唯一 Markdown 报告：逐笔审计、逐分量参数敏感性、负奖励和正奖励发生率、Always WAIT 相对优势风险，以及与原业务账户结果的对照。
7. 在离线校准没有数据结论前，**不得替换**原业务 Reward；不得擅自启动 100K/1M DreamerV3 或访问封存 Test。

**本交付的准确定位：实现已完成并通过本地纯 Python 单测与合成样例；真实历史 Env 回放需要 Codex 在数据齐备的工作机上验收后才能宣称通过。**

## 5. 工作机实施设计与阶段一记录（2026-10-11）

分支：`feat/fg-reward-v2-v3-offline`，直接在 Env 原仓库开发。源码位于 research/tools，产物位于忽略的 artifacts；旧诊断文件、归档和补丁保持原状。

统一研究配置：`configs/fg_reward_offline_v1.yaml`。范围仅为已冻结的 FG409 开发验证合约。32 个候选点为 beta=0/.05/.10/.20，lambda=0/.00005/.0002/.001，alpha=0/.05，rho=.3，gamma=332/333。对照为 Always ADD/WAIT、Delay5/15/30，以及 Random（合法时概率.5）Seed0–29。参数只用于探索，不是推荐正式参数。

阶段一验证：随包12项单测与人工合成 smoke 先通过。新增重叠分钟回归在原实现上失败；新增篡改 Reward entry_id 但保持总和不变的真实 Env 回归也失败。修复后，连同容量、Delay有效分钟、chunk、完整参数重组、右删失/强平未成交和缓存身份回归，共20项通过。仅新增研究模块、CLI、配置、测试及文档。

E1 实际目录：`artifacts/fg_reward_offline_v1/e1_20261011`。31,034次决策，75批开仓，73批TP，2批非TP，V0=22.740204501031855。逐步 Reward、完整成交及归因、权益序列和账户指标与旧日志一致；净收益18,990元，最大浮亏38,460元，费用1,350元。30次1024 chunk的评分器JSON恢复和金融账户连续性检查通过。7个冻结输入和原评估文件运行前后哈希一致。

示例评分 beta=.2、rho=.3、lambda=.0002、alpha=.05 时：G累计20.475290734，early=-1.7，capacity=-6.2047，未折扣potential=+8.553393393；PBRS折扣累计1.797e-16、终局端点项0。仅证明常数折扣的抵消，不证明当前Dreamer的学习/Bootstrap具备策略不变性。实际TP次数仍为73，G(T)=0不改写成功成交身份。

后续批量入口（无需GPU、无模型）：

```bash
conda activate dreamerv3
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python tools/fg_reward_offline.py \
  --study-config configs/fg_reward_offline_v1.yaml \
  --out artifacts/fg_reward_offline_v1/study_20261011
```

已验收E1可按配置中的verified_e1复用，复用前检查实际源码、参数、冻结数据、原日志及状态hash，并记录文件校验收据。若未准备E1，删除研究配置中的该可选项，入口会先执行E1。新目录拒绝覆盖；失败结果保留。

35条E2轨迹由最多4个spawn进程运行，各自独立账户、合约内部顺序推进。每条轨迹保存events.jsonl、参考配置逐分钟分量、逐笔信息、原Env账户/成交记录、评分器快照和输入hash；32组候选复用不变的TP/风险窗口/potential/capacity基础分量，仅增加参数、逐笔与交易日汇总。A极端候选还进行全轨迹逐分钟直接评分对照。

最终唯一研究报告为 `docs/FG_Reward_V2_V3_Offline_Review_Report_V1.md`。报告必须区分未折扣和折扣结果，列出每组分母、正负奖励频率、Always WAIT优势风险、同根歧义的保守/不计首次风险对照以及真实账本。Random分位数仅表示同一FG409行情上的Seed变化。无模型训练、无Test价格访问、无正式Reward替换。

## 6. 工作机阶段二完成记录（2026-10-11）

实际产物：`artifacts/fg_reward_offline_v1/study_20261011`。E1 通过后执行 35 条独立 E2 轨迹（5 个确定性策略、30 个 Random Seed），总计 36 条轨迹和 1,152 条参数汇总，4 个进程。每条均推进 31,034 根有效分钟，覆盖 91 个有资格交易日；终局未完成、右删失、非法 ADD 请求均为 0。运行前后 7 项冻结输入、49 项源码/配置/收据哈希一致。A 全轨迹直接计分与分量重组最大误差为 0，全部轨迹 PBRS 端点残差不超过 1e-10。

A 与独立 Always ADD 的动作、成交、Reward、权益和逐分钟分量五张完整表严格相同。Delay5 的原 V0 为 30.096090550，高于 A 的 22.740204501；但成交批数为 82 对 75，不能把增益全部归因于纯择时。30 个 Random Seed 中有 26 个 V0 不低于 A，均未匹配成交配额。

A 的 8/32 个候选参数下 V3 低于 Always WAIT 的 0 分，均包含网格中较强的容量惩罚。alpha=.05 在不利持仓的固定负 Phi 状态中产生逐分钟正分；A 未折扣 potential 为 +8.553393393，折扣后与终局端点抵消。因此不得把未折扣 shaping 总分解释成真实盈利或训练改善。

详细报告包含 75 笔 A 交易审计、全部 A 参数分量、策略对照、Random 分位数、正负频率、同根歧义、删失和 PBRS 限制。生成入口：

```bash
python tools/report_fg_reward_offline.py \
  --study-dir artifacts/fg_reward_offline_v1/study_20261011 \
  --out docs/NEW_FG_REWARD_OFFLINE_REPORT.md
```

报告入口要求完整 PASS 结果并保留已有报告文件；产物校验表、报告源码与文档 SHA 收据均在 artifacts，研究数据、图片与账户文件不提交 Git。此阶段没有启动 DreamerV3 训练或替换 V0。
