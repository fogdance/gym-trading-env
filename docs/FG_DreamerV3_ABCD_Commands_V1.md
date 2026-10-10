# FG ABC 开发与训练命令 V1

在 `/home/v/Documents/work/gym-trading-env` 执行。所有生成数据不提交。代码直接位于两个原仓库开发分支。

## 统一环境

```bash
conda activate dreamerv3
export PYTHONPATH=/home/v/Documents/work/dreamerv3:/home/v/Documents/work/gym-trading-env/src
export JAX_PLATFORMS=cuda
export MPLCONFIGDIR=/tmp/fg-matplotlib
export PYTHONDONTWRITEBYTECODE=1
```

## 数据清单

```bash
python tools/build_fg_abcd.py --config configs/fg_entry_v1/abcd_comparison.yaml
```

当前 v2_sequential 已生成；再次执行会拒绝覆盖。需要独立新版本时更改实验输出版本并保留旧清单，不删除旧产物。

## 先导准备与启动

先导与正式均采用按完整合约时间顺序推进；当前停止状态下，本轮只准备配置，不执行以下训练启动命令。旧 v1 轮转 checkpoint 不恢复到顺序模式，新版从新路径开始。

先查看冻结命令，不训练：

```bash
python tools/run_fg_abcd.py --budget pilot
```

P0～P3 验收通过后，执行 FG409、seed0 的 A100k、B100k、C70k+30k，日志统一在 `/data/logdir/fg_abcd_v1/v2_sequential/FG409/pilot/`：

```bash
python tools/run_fg_abcd.py --budget pilot --execute
```

也可通过 `--method A`、`--method B` 或 `--method C` 单独执行。GPU 不可用时停止；不退回 CPU。程序记录真实参数变化、gate、当前账户、顺序访问记录、数据覆盖、代码 SHA 和运行元数据。

Actor gate 尚未开启或没有有效参数更新时不宣称政策学习成功，不自动放宽 gate 或修改 Reward。通过学习审计后才运行正式 checkpoint 评估和 30-seed 配额随机诊断。独立账户随机回放采用进程并行，worker 不运行模型。

## 验证回放

只运行三种基线：

```bash
python tools/evaluate_fg_abcd.py \
  --env-config artifacts/fg_abcd_v1/v2_sequential/contracts/CZCE_FG409_env.yaml \
  --output artifacts/fg_abcd_v1/v2_sequential/new_baseline_review
```

评估已通过 gate 的模型时再传 `--model-config RUN/config.yaml --checkpoint RUN/ckpt/CHECKPOINT_FOLDER`。输出必须是新目录；默认 4 个 matched-random worker、30 个 seeds。

## 正式配置模板：本批 NOT_RUN

```bash
python tools/run_fg_abcd.py --budget formal
python tools/run_fg_abcd.py --budget formal --fold FG409 --seed 0 --method C
```

以上只打印 1M 命令模板，不启动训练。`--budget formal --execute` 会拒绝，避免准备阶段自动启动正式预算。

C 必须先完成自己的预训练预算。正式阶段后续获准执行并完成预训练后，可生成经过哈希核对的迁移收据：

```bash
python tools/run_fg_abcd.py --budget formal --fold FG409 --seed 0 --method C --create-transfer
```

不得共享 A 的 Replay 或拿 A 的模型冒充独立 C 预训练。近期恢复仍使用自身同目录 checkpoint；不重复初始化账户。

## TensorBoard

```bash
tensorboard --host 0.0.0.0 --logdir /data/logdir/fg_abcd_v1/v2_sequential
```

## 进度与验收报告

旧 v1 先导已按用户指令停止；v2_sequential 尚未启动。只刷新报告、不启动训练：

```bash
python tools/watch_fg_abcd.py --once
```

持续每 60 秒刷新可省略 `--once`。生成的 `pilot_results.md` 分别列出已保存检查点和最新日志步数；`comparison.md`、`pilot_comparison.json` 汇总各方法的学习审计与验证结果。未通过策略学习验收时明确标记，不能用其结果推断择时能力。

旧日志和 checkpoint 保留；本批未执行 D、1M 正式训练或最终盲测。
