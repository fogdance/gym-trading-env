# Gate 校准的完整状态迁移审查

当前状态：完整20窗预检GO，原始180k及新分支两个独立路径的真实CPU恢复全部通过，恢复证书已生成。正式GPU训练未启动。

## 来源和恢复边界

来源：`/data/logdir/fg_single3/online_after_10k_callback_fix/C1_M1_ONLINE_1M/ckpt/step_000000180000`。

Manifest SHA256：`4979acfb0272ffbc3c96833e5b77fccd66c5d4d0f335bfca48810f764d927a57`。

Agent文件 SHA256：`2cc450b328484ddfa489b3c54ef10cbf98395678dbc6ae957fd1a1fe3777b1c3`。

已停止日志最后到184,260环境步，但可完整继承的受保护节点为180,000步；其后4,260步不能冒充已经恢复。节点含183个已校验文件、176个Replay块、180,000行、179,936个可采样序列，Uniform sampler seed=202。模型计数为updates=21,183、batches=21,399、actions=180,000。

源身份的代码指纹对应Dreamer提交`09e2e5c`。当前已有`0b3a641`参数发布修复与`4659d32`共享判定阈值修复，所以新分支不声称原实验下的精确恢复。

## 最小迁移协议

复用现有`FullCheckpoint`严格恢复机制、`Agent.load_exact/save_exact`、原生Replay、Driver与交易Env。共享入口为`tools/fg_single3_experiment.py`；新配置集中于`configs/experiments/fg_single3/C1_M1_GATE_T030_from_180K.yaml`。

配置只允许两项差异：新`logdir`、`agent.avail_threshold`从0.50到已验收的0.30。官方`elements.Flags`和`_resolve_seed_protocol`已解析真实启动参数并确认没有其他配置差异。Reward、预算、网络、H333/Imag15、批次、Replay、交易规则与种子均继承来源。

允许的生产代码差异仅为`agent.py`共享阈值、`train.py`诊断聚合、`action_mask_warmup.py`日志别名、已存在的`embodied/jax/agent.py`参数发布修复，以及`fg_stage_transfer.py`显式新身份准入。恢复包冻结旧/新源码映射与文件哈希，不修改`FullCheckpoint`的身份规则。

复制整个节点后，仅重写`config.yaml`、`runtime.pkl`、`action_mask_warmup.pkl`及其manifest；其余组件和Replay文件必须与来源逐文件哈希相同。新manifest使用新provenance身份。旧节点、Replay、日志和身份不写入。

保留模型全部参数、联合优化器、策略缓存、pending outputs、RNG、Driver/交易账户/数据指针、原生Replay与采样器、两条prefetch队列、recurrent carry、ratio、report/save schedule以及所有累计训练计数。清空旧阈值warmup聚合与其samples，仅移除普通日志中旧`train/avail/`聚合；保留其他聚合和历史报告总数112。Gate仍为NOT_READY，streak从0重新统计，不将CPU预检的合格次数导入正式训练。

## 已执行的真实恢复证据

`tools/fg_single3_gate_restore_check.py`在`TFRT_CPU_0`构造真实Agent、Driver、Replay、RuntimeState，执行`FullCheckpoint.latest/load`和`Agent.save_exact`；没有进行来源模型梯度更新或行情推进。源节点测试实测97.7秒，状态`REAL_NEURAL_FULL_CPU_RESTORE_PASS`。

|保留对象|语义SHA256|
|---|---|
|完整Agent状态|`1c2b04e27b1fb10e0948c473137c4b43e29e74d01f7c95fdf3043ca20561beec`|
|World Model参数|`a60394134311d7c61477e861b2d70b4d93759b7558b7fee6267be15388ef869a`|
|Actor参数|`2c6068956275477fc4447a51ccad5a0b36ce483ba9ff318392c0db45c494eafe`|
|Critic参数|`8c506f365aa2aab1f60e0f90ae77e04927aa13e3b6b11e9a60968ddb0205acd2`|
|Optimizer状态|`0efc01ae3301cfd4d73cc1cec24a35969d46e92d307578929c82cd0360787d60`|
|完整Driver/账户状态|`6f4925f355b7bd93e593286411a5af7715a3770fd1e03227f282649af483cbfd`|

Replay各数组逐项相等；sampler RNG、Python/NumPy RNG、prefetch及RuntimeState恢复后逐项/语义一致；Actor Gate=false。实测产物：`artifacts/fg_single3_v1/diagnostics/gate_repair_20261012_01/source_cpu_restore_verified/`。

## 完成的新分支与准入证据

`prepare`已完成，状态`FULL_STATE_GATE_BRANCH_PREPARED`，预检`GO`，原20窗最大连续合格=20。`publish`只复制新目录，实测成功，未覆盖旧节点。

两个恢复路径：

- 项目恢复包：`artifacts/fg_single3_v1/experiments/C1_M1_GATE_T030_from_180K_20261012_01/ckpt/step_000000180000`。
- 发布节点：`/data/logdir/fg_single3/gate_threshold_v2/C1_M1_GATE_T030_180K/ckpt/step_000000180000`。

真实双路径神经恢复实测131.8秒，`REAL_NEURAL_FULL_CPU_RESTORE_PASS`，JAX设备`TFRT_CPU_0`。以上完整Agent、World Model、Actor、Critic、Optimizer和账户语义SHA均与来源相同；Replay数组、采样器RNG、Python/NumPy RNG、prefetch与RuntimeState全部通过恢复后相等断言。两个路径保留updates=21,183、batches=21,399、actions/step=180,000；Gate=false、streak=0、reports=112。无来源模型梯度更新或市场推进。

最终再次验证源183个组件及manifest完全不变；两目标除三个获准组件及新manifest之外，全部文件与来源字节哈希一致。证据为`migrated_cpu_restore/analysis.json`、恢复包`cpu_restore_proof.json`以及`final_integrity.json`。

来源identity digest：`a29518d6bbc90928799e3fd700f70993760d7fe22d67cac65bc6ba0915e0d6a2`；目标identity digest：`221ff505cdc1b336e261b58a815da3a63d071a5b94225e453456696f41eaf978`；目标生产代码hash：`d8dc714440e2a3e5799c464f3387de17863e5c6624ba1c4e7dc8995c6293e272`。

`gate_source_transfer.json` SHA256：`5aa5dbc3e2cae760de8fbd49f758987ff990978661b588394febd9be72cd7c8e`。

`cpu_restore_proof.json` SHA256：`e021b952418a2903d596d492529a80c531f84984c3dd7c1966116dedf4f8084d`。

共享`resume`缺少显式`--authorize-training`会在读取任何节点前拒绝。已用真实官方Dreamer CLI执行未授权启动检查：exit=1，报`PermissionError: Gate GPU continuation requires the explicitly authorized shared runner`，发生于`main.py:180`准入阶段，没有创建Agent或进入训练循环。同时对实际配置进行了带临时授权token的只读官方准入检查：PASS，没有调用训练运行器；token退出后清理。证据：`formal_admission.json`。

该限制仅用于本次显式Gate迁移，不改变其他实验原有精确恢复路径。正式继续仍须用户另行授权；CPU验收通过不会自动启动GPU。

## 实际运行命令

本次已执行（前两条CPU准备/文件复制，第三条CPU双路径恢复）：

```bash
python tools/fg_single3_experiment.py prepare --config configs/experiments/fg_single3/C1_M1_GATE_T030_from_180K.yaml
python tools/fg_single3_experiment.py publish --config configs/experiments/fg_single3/C1_M1_GATE_T030_from_180K.yaml
python tools/fg_single3_gate_restore_check.py \
  --checkpoint \
    /home/v/Documents/work/gym-trading-env/artifacts/fg_single3_v1/experiments/C1_M1_GATE_T030_from_180K_20261012_01/ckpt/step_000000180000 \
    /data/logdir/fg_single3/gate_threshold_v2/C1_M1_GATE_T030_180K/ckpt/step_000000180000 \
  --prepared artifacts/fg_single3_v1/experiments/C1_M1_GATE_T030_from_180K_20261012_01 \
  --source-proof artifacts/fg_single3_v1/diagnostics/gate_repair_20261012_01/source_cpu_restore_verified/analysis.json \
  --output artifacts/fg_single3_v1/diagnostics/gate_repair_20261012_01/migrated_cpu_restore
```

实际Python均为`/home/v/miniconda3/envs/dreamerv3/bin/python`，设置`PYTHONDONTWRITEBYTECODE=1`；复制到`/data`使用已授权沙箱外文件写权限，没有使用sudo。恢复包/证书不可覆盖，以上已完成命令不是重复启动命令；复验必须使用新的输出目录。

正式恢复后原在线总预算仍为1M环境步，从180k继续，不重新离线预训练。新Gate必须在真实在线报告中重新连续20次达标；其开启后保留原连续5次回归停训规则。GPU数值差异、持续更新后的稳定性和DEV表现仍待后续授权训练验证。
