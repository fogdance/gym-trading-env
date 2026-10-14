# B2 1M：Adapter OFF/ON 与四次否决诊断

## 结论与证据边界

同一个 B2 1M Checkpoint 在 FG409 DEV 前 5000 个市场分钟：
Adapter OFF 实际开仓 3 次、2 TP5 / 1 SL30，净盈亏 **−1254**，Reward **−2.0072574484**；
原 Adapter ON 为 0 交易、0 Reward、0 净盈亏。

基础 Actor 确实很少入场；关闭 Adapter 恢复了少量交易，但尚不能解决寻找优质
ADD 的目标。Adapter 本次避免一次 SL30，也拒绝盈利入场；不能把零交易视为
已证明的有效风险筛选，不能只按四次否决判定所有预测错误。

5000 个认证空仓 DEV 状态的 OutcomeHead 事件 NLL/Brier 都未优于同批样本的
经验常数先验，并存在 OPEN_AT_H 分类偏向。证据支持暂停增加预算，先检查
基础 Actor 的入场监督及 Outcome 预测质量；**本轮没有证明导致训练不足的
最终根因，也没有修改训练算法或超参数**。

## 来源与隔离

- 原冻结 Gym：`3970e6ec69f67fb2d02f30456a077ea5a31fd05d`。
- 原冻结 Dreamer：`5fd4ebdeb590fd8433338300bd82c7f2f520bf76`。
- Checkpoint：`/data/logdir/fg_single3/s3_v2/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014/ckpt/step_000001000000`。
- Checkpoint Manifest SHA256：`6eda87d3f0d173acc988fbad9e2f2e456498be230e6b0bf24ec74685fca9db57`。
- 原模型计数：updates=40000、batches=40001、actions=1001016。
- 工具 worktree：`/home/v/Documents/work/research/fg-s3-b2-diagnostics/gym-trading-env`，
  `feat/s3-b2-adapter-diagnostics`，从实际 integration Gym `cc6ba78f3a2016eb280135f7730d0a1f68d4512e` 建立。
- 神经模型使用原冻结双仓库运行时；不修改 Dreamer、integration、正式 Run 或旧评价。

结果根目录：
`/home/v/Documents/work/research/fg-s3-b2-diagnostics/results/b2_1m_20261015_052151`。
所有新结果仅本地保存，无 W&B 操作。未来行情只用于事后标签和 Oracle C，
不参与 Actor 的输入和执行，结果不能作为 TRAIN 教师样本。

## A/B：独立原生 Env 闭环

| 项目 | A：Adapter OFF | B：原 Adapter ON |
| --- | ---: | ---: |
| 实际市场推进 | 5000 | 5000 |
| 已实际执行 ADD / ENTRY | 3 / 3 | 0 / 0 |
| TP5 / SL30 | 2 / 1 | 0 / 0 |
| 未平仓右删失 | 0 | 0 |
| 闭仓净盈亏 | −1254 | 0 |
| Reward | −2.0072574484 | 0 |
| TP 市场分钟 | 110、28 | 无 |
| TP 中位市场分钟 | 69 | 无 |
| 首5 / 6–15 / 16分钟以后首次逆行10点 | 0 / 0 / 1 | 0 / 0 / 0 |
| 合法决策观察数（含预算末尾未执行提议） | 4682 | 5001 |
| 合法观察平均 ADD 概率 | 3.62535% | 3.29085% |
| 生效 Adapter mix | 全部为0 | 全部为1 |

OFF 采用进程内、明确标记的诊断干预：调用原 `MixSchedule.read()` 注册并读取
原变量后返回同形状的零。不改配置、不改调度计数、不改权重，不伪装成原
默认策略。原 Checkpoint 全部 **321 个参数/状态叶**完整恢复且摘要一致，
评价结束后全部摘要仍一致。原模型、优化器、Normalizer 没有被评价更新。
每个实际观察的 mix=0、有效 delta=0、base/final 掩码后概率相等均已断言。

OFF 的真实开仓决策 cursor 为17、1153、2540。cursor18在OFF中已经持仓，
ADD不合法，因此ON轨迹上的4个基础Actor提议不能直接视为4笔OFF闭环交易。
三笔实际闭仓净盈亏分别 +282、−1818、+282；未发生新的研究训练。

平均 ADD 概率不是入场频率。本次评价为 `eval_argmax`，原冻结模型
`policy.unimix=0.05`，两个动作都合法时概率有2.5%的均匀混合底值。
不能把3%左右的平均概率理解为每100步自主交易约3次。

## 四次否决：原生 Env 独立事后验证

| cursor | base ADD−WAIT logit | Adapter修正 | final ADD−WAIT | 独立真实退出 | 市场分钟 | 净盈亏 | H60标签 | 原Oracle立即ADD / 最优后续cursor |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| 17 | +0.446340 | −0.772618 | −0.326278 | TP5 | 110 | +282 | OPEN_AT_H | 否 / 34 |
| 18 | +0.026660 | −0.757091 | −0.730431 | TP5 | 109 | +282 | OPEN_AT_H | 否 / 34 |
| 1153 | +0.070970 | −0.215582 | −0.144611 | SL30 | 184 | −1818 | OPEN_AT_H | 否 / 1218 |
| 2540 | +0.078957 | −0.410678 | −0.331721 | TP5 | 28 | +282 | TP_LATE | 否 / 2553 |

四个状态全部通过原 `reference_candidate` 的实际 ADD→WAIT→退出验证，包括
原生 Reward、退出时序和费用归因。不是仅用合成价格或静态索引宣称通过。
四次全持仓结果为3TP/1SL，**不能相加为一个可执行策略的交易结果**。

所有案例首次5分钟逆行10点标签均为0。1153的第一次逆行在第34个有效bar。
四次裁剪饱和状态均为false。原密封记录只有logit差，没有独立保存两个
绝对 raw logits；上表给出实际保存的差值，不捏造原始绝对值。

六类预测概率（顺序与原 OutcomeHead 完全一致）：

| cursor | TP_FAST | TP_MID | TP_LATE | SL | OTHER | OPEN_AT_H | 首5逆行预测 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | .031275 | .060506 | .097187 | .039384 | .000180 | .771467 | .054149 |
| 18 | .033979 | .060225 | .125165 | .042520 | .000195 | .737916 | .079139 |
| 1153 | .081834 | .209775 | .298268 | .041482 | .000235 | .368406 | .073630 |
| 2540 | .049678 | .195334 | .239975 | .029338 | .002242 | .483433 | .112698 |

| cursor | 正向TP三项合计 | SL项 | OPEN_AT_H项 | 首5逆行项 | 总修正 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | +.086605 | −.038708 | −.767596 | −.052918 | −.772618 |
| 18 | +.096254 | −.041791 | −.734214 | −.077340 | −.757091 |
| 1153 | +.263702 | −.040770 | −.366558 | −.071956 | −.215582 |
| 2540 | +.209301 | −.028835 | −.481008 | −.110136 | −.410678 |

主要负项是H60内未退出，不是SL概率。前三个H60标签确为OPEN_AT_H，包括
184分钟才SL的1153；不能把最终SL直接改成“60分钟内SL”标签。
2540实际28分钟TP，最高预测却为OPEN_AT_H，是这四个案例中的事件分类错误。

C使用原市场分钟C1 Reward、`gamma=332/333`、原资金证明和exact suffix
Oracle，状态t的立即ADD以 `first[t] == t` 判断。四次均WAIT：这说明按原
完整后续机会目标更优的是等待，并不说明四次独立交易都亏损。C不是实际
执行策略，也不是替用户或Actor接管交易。

## 独立 DEV 预测质量

对ON已经密封的因果预测，先按原Driver重放实际动作、账户、mask、Reward及
状态fingerprint，再生成事后标签。5000/5000个已执行状态通过原资金认证。
原Env候选验证抽检24个状态（含事件代表、长路径及chunk边界），不是5000个
全部逐笔Env重放；四次否决另有全部逐笔原Env验证。

| 指标 | OutcomeHead | 同批DEV经验常数先验 |
| --- | ---: | ---: |
| 六类事件NLL | 1.249401 | 1.207422 |
| 六类事件Brier（六维平方误差之和） | .659787 | .636732 |
| 首5逆行NLL | .099962 | .104964 |
| 首5逆行Brier | .021174 | .021325 |

经验先验由同一批DEV标签计算，仅作描述性对照，不是独立训练的基线。
事件最高置信度的10-bin ECE为.100666；首5逆行ECE为.003018。
早期逆行实际正例109/5000（2.18%），较小的二元Brier不能单独证明选点有效。

| 事件 | 真实样本 | 最高概率预测次数 | 分类Recall | 分类Precision |
| --- | ---: | ---: | ---: | ---: |
| TP_FAST（≤5） | 180 | 0 | 0 | 无预测，未定义 |
| TP_MID（6–15） | 572 | 6 | .001748 | .166667 |
| TP_LATE（16–60） | 1502 | 178 | .032623 | .275281 |
| SL（H60内） | 212 | 0 | 0 | 无预测，未定义 |
| OTHER | 0 | 2 | 无真实样本，未定义 | 0 |
| OPEN_AT_H | 2534 | 4814 | .960931 | .505816 |

180个快速TP候选是重叠的独立空仓反事实状态，不是180笔可顺序执行的交易。
这些度量限定于本次FG409 DEV区间，不外推全部合约或TRAIN分布。
不把本次B2与历史Tutor800K的7ADD/5TP/2SL直接当作同源同预算消融。

OFF的完整原生Driver账户重放也通过：5000个执行决策的动作、cursor、chunk
reset、mask、Reward、状态fingerprint及最终账户全部匹配。仅1045个状态
满足现有教师资金证明，319个持仓状态不满足两个真实动作合法，3636个状态
因亏损后资金低于原证明下界而未认证，有效覆盖率20.9%。**没有绕过证明为
这些状态制造标签，也不把“教师未认证”说成“Actor动作非法”。**
OFF认证子集的事件NLL=1.337051、Brier=.716931，同批常数先验为1.208111、
.637222。它与ON的5000状态样本域不同，不能直接按两组NLL高低作策略消融结论。

## 命令与原始收据

运行记录及摘要放在结果根目录。示例变量均指向实际已验证来源；重跑时
必须换新的`--output`路径，已有结果不会覆盖。

```bash
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
GYM=/home/v/Documents/work/research/fg-s3-b2-full-train/gym-trading-env
DIAG=/home/v/Documents/work/research/fg-s3-b2-diagnostics/gym-trading-env
RUN=/data/logdir/fg_single3/s3_v2/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014
PREP="$GYM/artifacts/fg_s3_v2/prepared/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014"
ON=/home/v/Documents/work/research/fg-s3-local-evaluations/b2_step_000001000000_20261015_050252/evaluations/actor_only_dcbd7ec78c
OUT=/home/v/Documents/work/research/fg-s3-b2-diagnostics/results/b2_1m_20261015_052151

cd "$DIAG"
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/s3_b2_diag_mpl WANDB_MODE=disabled \
  "$PY" -u tools/fg_s3_b2_diagnostics.py off --source-gym "$GYM" \
  --prepared "$PREP" --source-run "$RUN" --output "$OUT/adapter_off_view" --steps 5000

cd "$GYM"
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/s3_b2_posthoc_mpl \
  "$PY" -u tools/fg_s3_v2_experiment.py score-predictions \
  --eval-dir "$ON" --output-dir "$OUT/on_prediction_scoring"

cd "$DIAG"
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/s3_b2_posthoc_mpl \
  "$PY" -u tools/fg_s3_b2_diagnostics.py posthoc --source-gym "$GYM" \
  --on-eval "$ON" --scores "$OUT/on_prediction_scoring" --output "$OUT/veto_posthoc"

cd "$GYM"
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/s3_b2_posthoc_mpl \
  "$PY" -u tools/fg_s3_v2_experiment.py score-predictions \
  --eval-dir "$OUT/adapter_off_view/evaluations/actor_only_5adaa54ea9" \
  --output-dir "$OUT/off_prediction_scoring"

cd "$DIAG"
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" -m pytest tests/entry_v1/test_b2_diagnostic_adapter_off.py -q -o addopts=''
```

| 验证 | 实际状态 | 证据 |
| --- | --- | --- |
| 进程内开关及异常退出还原单测 | PASS，1 passed | helper单测，不代表GPU模型验收 |
| 原ON完整账户重放及5000状态评分 | PASS | `on_prediction_scoring/receipt.json` |
| 四次否决原生Env独立重放 | PASS，17/18/1153/2540全部通过 | `veto_posthoc/receipt.json` |
| OFF真实GPU 5000步及全参数不变 | PASS | `adapter_off_view/evaluations/actor_only_5adaa54ea9/receipt.json`、`diagnostic_model_audit.json` |
| OFF完整动作/账户重放 | PASS，5000执行决策精确匹配 | `off_prediction_scoring/receipt.json` |
| 最后只读边界加强后的CPU复核 | PASS，四个案例、Oracle结果及候选文件字节一致 | `veto_posthoc_readonly_checks/receipt.json` |
| 整段Oracle完整Driver交易路径验收 | NOT_RUN | 仅事后suffix索引及上述候选验证 |
| 多DEV合约/多个种子/新训练 | NOT_RUN | 不扩大本轮诊断 |

策略研究目标仍为 **NOT_ESTABLISHED**；工具与核验成功不等于生产策略验收通过。

### 原始结果哈希与自查

| 收据（相对结果根目录） | SHA256 |
| --- | --- |
| `on_prediction_scoring/receipt.json` | `22b87a55ee882ec0e1273b55f77a836b0c08d6211d1cce2c9fa71532ee98ce92` |
| `veto_posthoc/receipt.json` | `873c8ead7b52fbffe6ba8c596245672ba0e29bcb9fda968e6afb4d1ef50a003e` |
| `adapter_off_view/evaluations/actor_only_5adaa54ea9/receipt.json` | `9aae4fa4656359b0cd44ea0fff772ebf45db3ea8215c47161ea57588a05eb4cd` |
| `off_prediction_scoring/receipt.json` | `1be68ecd100c0f053576af00013bbb4df48a3d0526fb7c4462b5d660d9be6849` |
| `veto_posthoc_readonly_checks/receipt.json` | `178200f9eb128dbda39eebf1d8228f0af9986c0df7ec7a17dfbb36cfe74e5f58` |

真实OFF和四次重放运行的工具SHA256为
`b7882dd994a821a57b1c2681a95022022c7befa5c9cafc4b97f20e47bdb72e21`。
最后自查仅加强CLI读写边界：显式设置Python运行时不写bytecode（正式运行
命令已经在进程启动前设置该环境变量）、检查评分输出文件哈希、确保失败时
关闭Env。没有改变模型开关、Oracle数学或统计指标；加强后的相同CPU事后
验证重新执行通过，GPU结果保留原始工具哈希，不改写成新版运行证明。
最终工具SHA256为 `ac7a533d55798d7f504e81d7f69f5653fcbca1eed28434d86b076f2c8d7a3631`。
复核使用上述posthoc命令，仅将输出改为 `$OUT/veto_posthoc_readonly_checks`。
