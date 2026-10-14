# B2 Full 正式 1M：统一入口与独立冻结准备

用户追加范围：完成 Tutor/B2 Full 后准备 B2 正式 1M，命令由用户在 tmux 执行。开发者不启动 GPU 或正式训练。800K W&B 里程碑由其他项目处理。

## 入口、配置与冻结目录

- 与 A2 共用 `tools/fg_s3_v2_experiment.py`，配置集中在 `configs/experiments/fg_s3_v2/`。没有新增训练入口或第二套 Tracker。
- 配置：`B2_FULL_MARKET_MINUTE_FORMAL_1M.yaml`。
- 独立冻结双树：`/home/v/Documents/work/research/fg-s3-b2-full-train/{gym-trading-env,dreamerv3}`。
- 准备目录：冻结 Gym 的 `artifacts/fg_s3_v2/prepared/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014`。
- 独立正式 Run：`/data/logdir/fg_single3/s3_v2/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014`，只在用户启动时创建。
- 双仓实际 SHA、源码 clean 状态、配置/准备文件 SHA、认证状态和测试收据路径统一记录在集成 Gym 的 `artifacts/fg_s3_v2/full_checks/final_delivery.json`。准备 manifest 自身也保存实际双 SHA 和源码哈希。
- 核对 A2 实际入口后，统一管理规则已加入 research 根目录 `AGENTS.md`；该文件不属于两个子仓库，原机器配置和资源规则保留。

## 冻结研究条件

| 项目 | 设置 |
|---|---|
| 预算 | 1,000,000 次真实 Env 推进，非 BC 更新次数 |
| 收集/更新 | 每 100 环境步 4 次联合更新，共 40,000 次本地更新 |
| Checkpoint | 每 100,000 环境步完整保存，保留原 FullCheckpoint/恢复流程 |
| TRAIN | 原 ALL_TRAIN 18 合约、原 least_progress 调度，正式 slice=1000 |
| 时间 | S3_MARKET_MINUTE，原 C1/TP5/SL30/单仓3手规则 |
| beta | 1.0 固定；不混入后续 DAgger 退火实验 |
| Outcome | H60 六类、独立前5市场分钟 early10；head128，两个 loss 权重各0.1 |
| Adapter | 原正负风险方向、StopGrad、系数和 delta 上限均2 |
| 预热 | warmup100 / ramp100 / min_event_updates100；本地梯度更新计数 |
| 其他学习 | 原 WM + Oracle BC；RL/Critic 始终冻结；不叠加 Tutor Loss |

18合约原输入合计488,160行。1M是原账户调度器上的累计交互预算，完整生命周期结束、持仓合法结清后按原机制重访TRAIN；不是1M条互不重复的新行情。

来源仍为已审计Online180k参数 Checkpoint，manifest文件SHA256：
`4979acfb0272ffbc3c96833e5b77fccd66c5d4d0f335bfca48810f764d927a57`。
路径：`research/fg-s3-dev/gym-trading-env/artifacts/fg_s3_arch_v2/online_180k_source/ckpt/step_000000180000`。
warm_start重新建立本Run优化器和累计计数，来源21183次模型更新只作为来源记录。

可信数据使用 `/data/logdir/fg_s3_market_minute/20261013_v2/{env/v1,oracle/v1,catalog/v1}`。
共享TeacherCatalog内容hash：`3781cf56208c2185fa5c9ebe40ad2e79d14cbdb0bce79067fd67e98b27122076`。
正式准入使用完整aggregate Oracle证据和共享TeacherCatalog；未放宽诊断LiveOracleTraining的限制。

## 必要改动与验证范围

原B2只支持单合约诊断。新代码沿用原18合约PoolOracleQuery和调度器，按真实合约分派原OutcomeCatalog，并在prepare/load/Sidecar join检查各自身份；没有重新实现Oracle、资金证明或标签索引。

- 受控CPU回归5项通过：正式预算/授权/路径/TeacherCatalog约束；原A2轮转；B2跨合约标签分派、Replay/Sidecar/Driver保存恢复后继续采样一致。
- 已有真实CPU神经合同9项和组合FullCheckpoint恢复1项通过：Actor、RSSM、OutcomeHead梯度有效，Critic冻结；单次联合优化，预热状态和下一次更新可恢复。详见Full实施报告和收据。
- 真实18合约CPU检查与最终prepare采用同一认证加载结果。正式配置不变；独立检查selector使用4步slice，75步保存、恢复后继续到80步。结果及每合约有效/排除标签数保存在冻结树 `artifacts/fg_s3_v2/full_checks/b2_full_real_18_contract_cpu/receipt.json`，最终交付索引核验实际PASS后才记录完成。
- 初次检查误要求每合约前4步必有Outcome标签，实际FG405这4步都是 `[1,0]`，按原规则排除为 `TWO_REAL_ACTIONS_REQUIRED`。已修正检查要求；没有放松真实mask或伪造监督。初次失败日志和数据保留在集成树 `full_checks/b2_full_real_18_contract_cpu_v1`。
- 新Full GPU、10K、1M与新Full Actor-only绩效均 **NOT_RUN**。历史B2独立DEV预测评价结果较常数先验差，不能推断100步预热已经改善模型。

真实准备/CPU检查的复现入口（只用于一个尚未准备过的新配置，拒绝覆盖快照）：

```bash
env PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/fg_s3_full_mpl PYTHONPATH=src:../dreamerv3 \
  FG_S3_REAL_B2_PREPARE_CONFIG=B2_FULL_MARKET_MINUTE_FORMAL_1M \
  /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest \
  tests/entry_v1/test_fg_s3_b2_real_preparation.py -s -q
```

该测试调用原prepare，只保留原prepare_pool返回对象供随后的验证使用，避免重复解析18合约。真实模型不初始化，临时selector不修改正式prepared配置；通过不代表新版GPU验收。

## 用户启动、恢复与评价

在tmux中执行，现有W&B包装器自动跟踪。代理 `http://127.0.0.1:10808` 已内置，无需另设；云端不可用时保留offline记录，训练子进程不依赖网络。

```bash
cd /home/v/Documents/work/research/fg-s3-b2-full-train/gym-trading-env
env PYTHONDONTWRITEBYTECODE=1 WANDB_ENTITY=xkm104501-none WANDB_PROJECT=fg-single3-research WANDB_MODE=online \
  /home/v/Documents/work/research/fg-s3-wandb-env/bin/python tools/fg_s3_wandb.py launch \
  --run-dir /data/logdir/fg_single3/s3_v2/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014 \
  --prepared "$PWD/artifacts/fg_s3_v2/prepared/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014" \
  --trainer-cwd "$PWD" -- \
  /home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start \
  --config B2_FULL_MARKET_MINUTE_FORMAL_1M --authorize-training
```

中断恢复：上面同一条命令仅把 `start` 改为 `resume`，保留同一Run及W&B身份。

独立Actor-only DEV：

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py evaluate \
  --config B2_FULL_MARKET_MINUTE_FORMAL_1M --eval-steps 5000
```

随后在同一入口用 `score-predictions --eval-dir <实际评价目录> --output-dir <本Run新的prediction_evaluations子目录>` 补评预测，再用已有W&B `import-evals`导入。断网后的同步使用已有`sync`，指定相同 `--run-dir` 和 `--prepared`。不将Oracle接管成绩当成Actor-only绩效。

Project：<https://wandb.ai/xkm104501-none/fg-single3-research>。正式Run未由开发者启动，因此本轮没有新的Full云端Run URL。
