# Action Mask 自动 Warm-up 训练手册

日期：2026-06-12

## 1. 结论

正常训练只启动一次，不再手工拆成 availability warm-up 和 masked actor 两次训练，也
不再管理独立 availability 资产。

```bash
cd ~/Documents/work/dreamerv3

python -m dreamerv3.main \
  --configs action_mask_formal \
  --logdir /data/logdir/action-mask-formal-202606122100
```

程序会在同一个进程、同一个 Agent、同一个 replay 和同一个 logdir 中自动完成：

```text
fresh run
  -> availability warm-up
  -> 连续报告窗口通过门槛
  -> 自动保存切换点 checkpoint
  -> 自动启用 masked actor/value
  -> 正式训练期间持续监控 availability
```

用户只需要在启动前确定最终训练配置和 logdir。模型尺寸、batch、replay、train ratio、
worker、JAX 资源和环境配置都直接按本次正式训练需要设置，不需要额外建立兼容资产。

## 2. 自动状态机

### 2.1 Fresh run：warm-up

`action_mask_formal` 强制从以下状态开始：

```text
agent.avail_actor_enabled = false
action_mask_warmup.enabled = true
```

此阶段：

- world model、decoder、reward、continue 正常训练；
- posterior 和一步 prior availability loss 正常训练；
- policy loss 为 0；
- imagined value loss 为 0；
- replay value loss 为 0；
- 真实 policy 仍使用环境 true mask，不能产生被 mask 禁止的动作。

自动 warm-up 模式若发现 actor 在启动时已开启，会直接报错，禁止绕过门禁。

### 2.2 自动切换：formal

每个 report 周期单独聚合该周期内所有训练 update 的 availability 指标，并使用这个
大样本窗口检查门禁；不使用单个 report batch，避免偶然抽到简单样本导致过早切换。
如果一个周期内累计不足 `25,000` 个训练样本，程序会跨周期继续累计，不把小样本窗口
计为通过或失败。默认要求连续 5 个合格聚合窗口全部通过：

| 指标 | 默认门槛 |
|---|---:|
| posterior accuracy | `>= 0.995` |
| posterior exact accuracy | `>= 0.990` |
| prior accuracy | `>= 0.990` |
| prior exact accuracy | `>= 0.980` |
| prior false-positive rate | `<= 0.005` |
| imagined-mask fallback rate | `<= 0.0001` |

通过后程序在同一进程内执行：

1. 将动态 actor gate 从 `0` 切换为 `1`；
2. 开启 policy、imagined value 和 replay value loss；
3. 立即保存包含切换状态的完整 checkpoint；
4. 继续使用现有 model、optimizer、replay 和 logdir 正式训练。

不会重新初始化模型，不会清空 replay，也不需要第二条启动命令。

### 2.3 正式训练监控

进入 formal 后，availability 监督损失继续训练，防止 encoder/RSSM latent 漂移后
availability head 失效。

默认连续 5 个聚合窗口未通过门槛时，训练抛出错误并停止。这样不会在 availability
已经退化时继续积累错误 actor 更新。

## 3. Checkpoint 与恢复

checkpoint 同时保存：

- 完整 Agent 参数和 optimizer；
- replay；
- 当前 step；
- 动态 actor gate；
- warm-up/formal 阶段、连续通过次数、连续退化次数和切换 step。

使用相同 logdir 重启时：

```text
warm-up checkpoint -> 从已有连续窗口状态继续 warm-up
formal checkpoint  -> 直接恢复 formal，不重新 warm-up
```

启动日志会明确打印：

```text
Action-mask training phase: warm-up
```

或：

```text
Action-mask training phase: formal
```

## 4. 修改正式训练参数

可以直接修改 `action_mask_formal` 中任意本次训练需要的参数，然后使用一个全新 logdir
启动。自动 warm-up 会在最终配置上训练 availability，再自动进入正式 actor 训练。

例如改成 `size50m`、增大 batch、修改 replay 或切换 stage，都不需要先运行另一套
warm-up 命令：

```yaml
action_mask_formal:
  <<: *size50m
  task: gymnasium_CustomTradingEnv-v0
  batch_size: 128
  replay.size: 5e6
  run:
    steps: 5e7
    train_ratio: 64
    envs: 16
    report_every: 300
  env.gymnasium.config_path: data/trading_stage1.yaml
  action_mask_warmup.enabled: True
  agent.avail_actor_enabled: False
```

硬约束只有两条：

1. 新训练使用新 logdir；
2. 自动 warm-up 正常入口保持 `action_mask_warmup.enabled=True` 和
   `agent.avail_actor_enabled=False`。

恢复已有 logdir 时，不应修改模型结构、task、动作/observation schema 或环境合法性
语义；这些变化应作为新训练使用新 logdir。

## 5. 运行观察

`run.report_every` 是墙钟秒数，不是环境 step。它决定 availability gate 的检查频率。
门禁输入会记录为 `action_mask_gate_eval/avail/*`。
`action_mask_warmup/pending_train_samples` 用于观察当前窗口已累计的训练样本量。

查看阶段和切换状态：

```bash
jq -s '
  [.[] | select(has("action_mask_warmup/actor_enabled")) |
   {
     step,
     actor_enabled: ."action_mask_warmup/actor_enabled",
     passed: ."action_mask_warmup/last_passed",
     consecutive_passes: ."action_mask_warmup/consecutive_passes",
     regressions: ."action_mask_warmup/consecutive_regressions",
     switched_step: ."action_mask_warmup/switched_step"
   }] | last
' "$LOGDIR/metrics.jsonl"
```

查看 loss gate：

```bash
jq -s '
  [.[] | select(has("train/avail/ready")) |
   {
     step,
     ready: ."train/avail/ready",
     policy: ."train/loss/policy",
     value: ."train/loss/value",
     repval: ."train/loss/repval"
   }] | last
' "$LOGDIR/metrics.jsonl"
```

warm-up 时 `ready=0` 且三个 actor/value loss 为 0；切换后 `ready=1` 且它们开始正常训练。

## 6. 已验证行为

2026-06-12 自动流程 smoke 已验证：

- fresh run 从 warm-up 启动；
- warm-up 时 policy/value/repval loss 为 0；
- report 达标后同进程自动切换，loss gate 变为 1；
- 门禁基于 report 周期内的训练指标聚合，不会被单个小 report batch 的偶然完美结果
  提前触发；
- 小于默认 `25,000` 个训练样本的窗口不会参与门禁判断；
- 切换点立即保存 checkpoint；
- 相同 logdir 重启后直接恢复 formal；
- 恢复后 replay 保留，policy/value/repval 继续训练。

`/data/logdir/action-mask-overnight-20260611/` 的历史结果仍证明 availability 可以收敛，
但不再是新训练必须加载或管理的启动资产。
