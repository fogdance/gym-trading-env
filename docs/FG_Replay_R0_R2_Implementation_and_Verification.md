# FG Replay R0～R2 实现与验证

日期：2026-10-11。实施分支：`feat/fg-replay-r0-r2`。实现提交：`65da6aa`。

范围和冻结参数见 [实施计划](FG_Replay_R0_R2_Implementation_Plan.md)，接口和复用命令见 [Replay 数据契约](replay_contract.md)。本次只构造、对账、统计 Replay；无 Agent、梯度更新、GPU、网络结构或 Actor 监督 Loss 变更。

## 实现与证据来源

- `tools/fg_replay.py`：CPU `generate/reconcile/audit` CLI，不接入现有训练入口。
- `src/gym_trading_env/research/fg_replay.py`：冻结源输入、因果策略、原始 Driver 收集、原生 Replay 保存、完整账户及逐行事件、第二个原 Env 重放、不可覆盖的运行与重试身份。
- `src/gym_trading_env/research/fg_replay_coverage.py`：真实原生 Uniform 加载/采样，stepid 关联事件，分层覆盖、上下文/训练位置分开计数、资源与重复率推算。
- `configs/fg_entry_v1/replay_r0_r2.yaml`：FG409/all，四组策略、seed=0；P3 等待15个有效交易分钟后连续合法ADD直至满仓，3手成交不重新等待。

原始 Driver 的输出动作属于下一次执行，当前 Reward 属于已经执行的前一动作。Sidecar 分别保存二者；1024截断后的相同游标观察不计为新行情或新成交。不同策略/合约账户使用独立原生chunk链，跨chunk采样逐步检查身份和游标。

生成前的实际源输入校验曾发现 CSV Date/bob 为无时区时间、manifest 为 `+08:00` 时间，直接比较拒绝了正确的源边界。现按冻结 Env 的 `data_tz` 归一化再比较；新增回归测试验证这一实际失败路径，没有修改源CSV或原Env。

## R0 验证

针对性命令：

```bash
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
  MPLCONFIGDIR=/tmp/fg-replay-mpl \
  /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest \
  -p no:cacheprovider tests/entry_v1/test_fg_replay.py -q
```

实测结果：`9 passed`。覆盖：

1. 原Driver/Env一致；ADD后跨1024、完整导出恢复后TP，Reward时点及去重；原生保存加载、跨存储chunk、独立账户隔离和canonical内容哈希重建。
2. 修改已保存Reward并更新物理文件哈希，仍被逐行重放拒绝。
3. P3满仓释放及同bar满仓后TP释放的计时；重置/元数据不计时；3手成交后持续ADD。
4. P2仅按当前合法Mask抽样，截断占位动作不消费RNG。
5. 输入哈希变化、实际训练交易日越界拒绝；未打开封存价格文件。
6. 源CSV无时区时按冻结Env时区验证。
7. 原生sample的stepid事件关联、上下文/训练位置分离和100K推算。
8. 数据结束前强退未成交：保留6手及pending MAIN_ROLL，统计为未成交终局失败。
9. 原Env合法ADD因下一bar成交量不足而拒单，归入真实反馈行及原因统计。

真实FG809工程验证：4条完整路径各28,291行、MAIN_ROLL终止，逐行对账PASS、账户导出恢复合计108次。P2_p01的2次TP均跨Env截断与Replay chunk；其他组0次TP。四组及全量池各1,000个真实batch均完成原生加载数、字段、连续账户、Mask和固定RNG完整batch复现检查。工程结果单独保存在 `smoke_20261011_01` 和 `smoke_audit_20261011_02`，不计入正式语料。

相同身份 `generate --resume` 实测复用原有4条完成轨迹，dataset manifest的SHA256保持不变。相同目录普通生成实测报 `FileExistsError: Refusing to overwrite an existing Replay run`，拒绝覆盖。

## R1 正式语料

正式生成命令：

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/fg-replay-mpl \
  /home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_replay.py generate \
  --output artifacts/fg_replay_r0_r2/v1/FG409/all/run_20261011_01 --workers 4
```

冻结来源：18个完整训练合约，488,160行，其中488,154有效分钟、6条零时长元数据。训练manifest content_hash为 `ffc45db584106b7c348b7f3aee95096acbcb560c31a7d67fa340f30d2e10009a`。107个输入及执行源码文件登记实际哈希；仓库provenance为gym `65da6aa0770a14a59e8f740e7b679ed23b948145` / Dreamer `64c1a51e2e1fd3bd4442662795ca313a9ed4afe3`。

正式语料目录：`artifacts/fg_replay_r0_r2/v1/FG409/all/run_20261011_01`。
完整 dataset manifest SHA256：`a1049114a84a00c565585ee8abe5cd0155058c7d2aef2bc41b7ef1d15ec5cb9c`。

| 项目 | 正式实测 |
| --- | --- |
| 账户路径 / 逐行对账 | 72 / 全部 PASS |
| Replay 行数 / 有效连续序列起点 | 1,954,516 / 1,949,908 |
| 实际推进交互 / 有效分钟交互 | 1,952,568 / 1,952,544 |
| 唯一推进行情分钟 / 每分钟策略重用 | 488,136 / 4 次 |
| 非推进行 / Env 截断续接与完整账户恢复 | 1,948 / 1,876 次 |
| TP / 非 TP 退出 | 6,234 / 142 |
| TP 跨 Env 截断 / Replay chunk | 1,003 / 992 |
| 终局 / 未访问源行 / 终局未平仓笔数 | 72 条 MAIN_ROLL / 0 / 0 |
| 已选轨迹产物字节（视图硬链接不重复计） | 897,177,668 |

初始行情观察不属于推进交互，因此唯一推进分钟比源有效分钟少18。四组账户访问同一行情，不能把1,952,544次模拟分钟交互解释为独立市场样本。正式语料中真实 ADD 拒单为0；未成交终局和右删失为0，相关边界由R0原Env测试验证，未人为插入正式样本。

### 汇总失败与单条恢复记录

首次汇总的哈希校验阻止发布完整 manifest：2,164个已登记轨迹产物中，P2_p05 × CZCE.FG909 的一个 `.npz` 与登记哈希不一致，实际加载报 `Bad CRC-32 for file 'market_seq.npy'`。这是确认的文件损坏，损坏原因未确认；按用户指示不继续追查。

此前已经启动的单条重建完成于 `attempt_001`，逐行重放对账PASS；`array_hash`、`event_hash`、`replay_trace_hash` 全部与 `attempt_000` 原登记值一致。其余71条复用原完成结果，随后完整语料验证PASS。旧损坏尝试和未完成视图保留，不纳入正式manifest或审计；没有修改旧哈希以放行损坏数据。

恢复凭据：`run_20261011_01/recovery_20261011_01.json`，其中记录原损坏文件的预期/实际SHA256、两个尝试的manifest哈希和内容一致验证。实际正式轨迹以dataset manifest为准，禁止扫描全部历史attempt并混入语料。

## R2 正式覆盖

完整dataset发布后自动接续如下CPU命令，正常退出，返回码0：

```bash
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fg-replay-mpl \
  /home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_replay.py audit \
  --dataset artifacts/fg_replay_r0_r2/v1/FG409/all/run_20261011_01 \
  --output artifacts/fg_replay_r0_r2/v1/FG409/all/audit_20261011_01 \
  --batches 1000 --batch-size 16 --seed 202
```

四组和全量池各1,000 batch，每batch 16条真实原生Uniform序列、length=64+context1。合计80,000条序列、5,200,000个含上下文采样位置；每池额外一个完整batch的固定RNG复现检查不计入这5,000 batch。全部原生加载数、字段形状/dtype、Mask与Sidecar、连续账户/游标检查PASS，账户误连接为0。

| 池 | 实际加载序列起点 | 数组字节 | 跨存储chunk采样序列 | 重复采样起点 | train64 无TP batch比例 |
| --- | --- | --- | --- | --- | --- |
| P1 | 487,477 | 2,147,524,455 | 992 | 254 | 40.8% |
| P2_p01 | 487,477 | 2,147,524,455 | 992 | 254 | 30.4% |
| P2_p05 | 487,477 | 2,147,524,455 | 950 | 254 | 35.7% |
| P3 | 487,477 | 2,147,524,455 | 992 | 254 | 29.0% |
| all | 1,949,908 | 8,590,097,820 | 1,005 | 68 | 32.5% |

全量池序列构成：P1=3,976、P2_p01=4,009、P2_p05=3,954、P3=4,061。各池capacity等于全部有效序列起点，没有FIFO淘汰。审计进程累计峰值RSS为9,483,296,768字节，包括索引、标签、临时batch和前序池。

混合池train64的零事件batch比例：合法ADD决策30.3%、成交32.2%、TP32.5%、前5根首次逆行72.2%、6～15根首次逆行75.6%。语料以6手持仓/受限WAIT为主；不能把高总行数当成稀有决策覆盖充分。

完整分层、年龄、逐笔诊断、context/train64分布及100K/1M推算见 [正式覆盖报告](replay_coverage.md)。按train_ratio=64，100K/1M预算对应推算6,250/62,500次更新、6.4M/64M模型时间位置，预期重复序列起点约2,521/217,677；这些只是推算，没有执行训练。

审计manifest SHA256：`647b33bb72f0f57ffe25bb959aa7494e66935601d10819dbc0f5bc69ce90e864`。
覆盖Markdown SHA256：`d91beb7a8e8f034b32a0ab15139470b0d32d52960533de9f5a6d9d1ff6e6d22e`。
回执记录实际 `jax_platforms=["cpu"]`、`gradient_updates=0`，五池均 `first_batch_reproduced=true`、`discontinuous_sequences=0`。

最终只读核验再次运行 `validate_dataset`：72条及全部冻结输入、选择产物、视图和对账凭据PASS；审计全部产物哈希PASS；五份计数张量均为 `(1000, 2, 28)`。提交到docs的覆盖报告与机器审计目录原Markdown逐字节一致。

## 自审、限制与后续

已检查实现及最终报告diff和 `git diff --cached --check`；实现提交只含本任务7个文件，报告提交只含本文和正式覆盖Markdown。原Env、Reward、Dreamer网络、训练配置和另一个Codex的文件未改动。Replay及机器事件数据留在独立artifacts目录，未提交大体积文件。

实施计划验收项全部有上述证据：原Driver/直接Env逐行一致；跨1024恢复TP及Reward去重；原生保存加载/跨chunk；账户隔离；固定RNG与canonical重建；P2/P3计时；输入变化和训练边界拒绝；正式72条完整路径；五池真实batch与稀有事件统计；仅提交本任务文件。原生保存文件损坏被哈希Gate拒绝，恢复尝试没有绕过Gate。

生成使用4个CPU工作进程，未持续用满24核；五池审计串行以控制内存，原生加载使用16线程，采样核验在单进程执行。并发属于运行资源设置，不改变本次策略、数据身份或统计口径。

本轮数据正确性与学习充分性分开验收。学习是否充分仍为 `NOT_ESTABLISHED`，没有WM/Actor训练或策略收益对照证据。人工方向配置属于当前实验的条件输入；OHLC成交是原Env的保守模拟；MAE仅是已观察OHLC的保守上界。

后续只根据正式覆盖报告决定是否另行增加固定P2种子或明确混合比例，不自动挑盈利轨迹，不推进R3/R4。
