# S3-Tutor-CD V0 实施记录

结论：**V0 静态教学实现 GO_WITH_LIMITS**。CPU 契约、真实 GPU 1K、500→1000 完整恢复通过；独立 DEV 仍 All-WAIT，**没有自主交易改善证据**。动态课程、10K 和正式 1M 未运行。

## 范围与隔离

- 依据 `FG_S3_Tutor_CD_V0_Codex_Implementation_Plan_Reviewed_20261013.md`。
- 独立双仓：`research/fg-s3-tutor-dev/{gym-trading-env,dreamerv3}`；分支均为 `feat/fg-s3-tutor-cd-v0`。
- 实际基线：Gym `9b23b72cb395002224bd3f87bdc808903b2d998e`，Dreamer `6356747d64963fb5ef55a036d837591f778c9f0a`。相较任务书旧基线，包含已实现的 beta 调度与观察入口；不合入 C2 开发分支。
- A2 正式训练树 `research/fg-s3-v2-train-beta` 和运行 `/data/logdir/fg_single3/s3_v2/A2_GPU_FORMAL_1M_BETA_v1` 不修改。
- 只读复用既有 C1 TRAIN、Oracle、TeacherCatalog、原 Online180k 来源；新的 prepared/run 位于 Tutor 工作树。

## 最小实施顺序

1. Tutor 分支复用 A2 的 WM-only 路线、原 Actor、单次联合优化器；Critic/RL 全部冻结。
2. 同一次 masked Actor 前向计算 Natural CE、Balanced CE（缺类别时回退）、同 episode P/N Pair Ranking；表征 stop-gradient，不改真实 Replay action。
3. 冻结 Tutor 超参，训练与 Actor-only 评价使用同一配置构造；累积指标纳入原 FullCheckpoint。
4. 必要 CPU 数值、梯度边界及真实 Replay/Sidecar 配对覆盖验证。
5. 用户后续允许与 A2 短时共用 GPU（30 分钟内）：执行 1K 真实训练，500 步保存并退出，再独立进程恢复至 1K；不改变 A2。

## 验收记录

| 项目 | 状态 | 证据 |
|---|---|---|
| CPU 数值、Mask、重置边界、配置保护 | PASS | 5 个定向测试通过；原 A2 formal 授权回归 1 个通过 |
| CPU 小模型梯度/冻结/保存恢复 | PASS | Actor 梯度约 0.83，WM 非零；CLOSED→READY fixture→CLOSED 的 Critic 梯度始终 0，参数及专属状态字节不变 |
| 真实 Native Replay PairCoverage | PASS（历史真实数据契约） | FG201 封存 Replay 原生采样 100 批，100 批可配对，共 19,869 对；排除跨 reset 687 对；不等于新 Tutor 训练 |
| 原模式兼容和 Pair-only 优化边界 | PASS | A2、C2 CLOSED 各自与改动前同模式更新字节一致；Natural-only Tutor 与 A2 最大参数差 0；prefix 标签不监督；Pair-only 更新 Actor，无 pair 和全零权重冻结 Actor 动量 |
| GPU 1K + 恢复 | PASS（受限真实训练） | 1000 实际 Env 步，40 次 WM+Actor 更新；20 次为新进程恢复后更新，40/40 可配对，共 11,096 对 |
| Tutor Actor-only DEV 入口与配置加载 | PASS | 实际 GPU FG409 1000 步；无 Oracle/教师接管，加载冻结 Tutor 配置 |
| 自主入场初步效果 | FAIL（当前短跑） | 0 入场、0 TP5、0 SL30、Reward 0、All-WAIT；不解释成教师交易成功 |
| Tutor 10K / 动态课程 / 正式 1M | NOT_RUN | 尚未授权 / 当前阶段不启用 |

没有未解决的已发现代码正确性失败。正式 Gate 连续 20 窗口开启并未执行；CPU READY 是测试 fixture。
CPU 小模型和历史 Replay 契约结果均未冒充真实 GPU 或生产验收。FullCheckpoint 为实际完整边界恢复，未宣称逐字节确定性续跑。

## 数学和身份

单次联合优化：`WM + bc_lambda * (0.8 * NaturalCE + 0.2 * BalancedCE) + 0.1 * PairLoss`。
缺 P/N 时 BalancedCE 回退 NaturalCE；同一 `[B,T]` 序列、同一 `is_first` 分段配对，空 pair 为 0。
同一次 Actor masked distribution 的最终 unimix logits 提供 CE 和 `z=ADD-WAIT`；Actor 表征 stop-gradient。
保留 `bc/nll` 为 NaturalCE；未定义分组/Pair 分数在 JSON 中为 null。`bc/tp` 是分类真阳性，绝非 TP5 交易。

Tutor 超参进入 settings、model_config 的 agent_extension、run identity 和原 FullCheckpoint。
SourceAdmission 的网络基础配置保持独立，训练扩展不覆盖已准入键。来源为原 Online180k（环境步 180000 / 模型更新 21183），不冒充 A2 warm-start。
Tutor 在任何 Gate 状态均禁止 Critic/RL、SlowCritic 和 RL Normalizer 更新；保持原 Actor/WM、原单个 optimizer。

只读输入软链接（代码不链接）：

- `artifacts/fg_s3_arch_v1/teacher_catalog_v1_final` → `research/fg-s3-dev/gym-trading-env/artifacts/fg_s3_arch_v1/teacher_catalog_v1_final`。
- `artifacts/fg_s3_arch_v2/online_180k_source` → `research/fg-s3-dev/gym-trading-env/artifacts/fg_s3_arch_v2/online_180k_source`。
- 行情和 Oracle 继续通过用户既有 `research/fg-data` 数据别名读取；Python 三包实际导入均来自 Tutor 双仓。

收据：`artifacts/fg_s3_v2/tutor_checks/{real_pair_coverage,cpu_neural,cpu_compatibility,gpu_1k_verification}.json`。
已执行命令前缀：

```bash
taskset -c 20-23 nice -n 10 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:../dreamerv3 \
 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=/tmp/fg_s3_v2_mpl \
 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q
```

参数分别为：

- `tests/entry_v1/test_fg_s3_tutor_cd.py -k 'not small_cpu and not pair_only_activation'`：5 PASS（GPU 实跑前执行）。
- `tests/entry_v1/test_fg_s3_tutor_cd.py -k 'small_cpu or pair_only_activation'`：小模型 1 PASS；兼容用例最初断言失败，见下述修正。
- `tests/entry_v1/test_fg_s3_tutor_cd.py -k pair_only_activation`：修正后 1 PASS。
- `tests/entry_v1/test_fg_s3_tutor_cd.py -k actual_gpu_tutor`：1 PASS，CPU 读取并核验真实 GPU Checkpoint，未替代 GPU 计算。
- `tests/entry_v1/test_fg_s3_v2_pool.py -k formal_budget`：原 A2 授权保护 1 PASS。

合计 8 个 Tutor 用例和 1 个原 A2 授权回归分别通过，未重复跑全面套件。
初始测试错误：把 A2 与 C2 不同 Ninjax 随机路径误要求为跨模式字节相同；随后改为各自对照 Git 旧实现。
另一个测试把新建 optimizer 首次 warmup（LR=0）的非零梯度误要求为立即参数移动；保留原 optimizer 配置，以连续两次更新验证。生产 Loss/optimizer 不因此改变。

## GPU 实跑和效果

冻结训练代码：

- Gym `e82e4dc47f97921861f2953616dc5492fc48c247`。
- DreamerV3 `33636c0593392b4f6ff203109ce41bfc831225a9`。
- 后续提交仅补充测试、报告和收据；运行代码哈希不变，prepared/Checkpoint 不修改。

Run：`research/fg-s3-tutor-dev/gym-trading-env/artifacts/fg_s3_v2/runs/TUTOR_CD_V0_GPU_SMOKE_1K_v1`。
独立 prepared：同工作树 `artifacts/fg_s3_v2/prepared/TUTOR_CD_V0_GPU_SMOKE_1K_v1`。
真实 GPU `cuda:0`、原 42,584,132 参数 Replay 模型；模型累计更新 `21183 → 21203 → 21223`。
完整 Checkpoint `step_000000000000 / step_000000000500 / step_000000001000` 保留。
最终 Checkpoint manifest SHA256：`11811c114868e9d134469a83c7dab99799f2b9b4338d655fb56a7056471822d4`。

20 个恢复后更新保持 Actor、Encoder/RSSM 参数继续变化；Critic、SlowCritic、Normalizer 及其 optimizer 状态在两个真实边界间字节不变。
Sampler/Replay、已提交 Sidecar、pending 执行动作、模型/optimizer、累计计数一致；未完成教师没有 native stepid，不进入训练。
原 Replay Source 文件哈希不变，源参数完整继承，仅有已准入的跨阈值 Gate 重置；没有独立 BC optimizer。

| 指标 | 实测 |
|---|---:|
| 实际 TRAIN 环境步 / 联合更新 | 1000 / 40 |
| 可配对更新 / pair 总数 | 40/40（100%） / 11,096 |
| 新采集认证标签 / 资金未证明行 | 870（87%） / 0 |
| Actor 梯度范数范围 | 0.2795–4.6966 |
| WM 梯度范数范围 | 119.3872–390.7748 |
| Critic 梯度 / RL Actor、Critic 更新 | 0 / 0、0 |
| 采集+更新累计核心耗时 / 吞吐 | 111.02 秒 / 9.01 实际环境步每秒 |
| 恢复后 run_to 耗时 | 61.30 秒（不是两进程总耗时） |
| 进程峰值内存 / Tutor 显存 | 3805 MiB / 实时检查约 2190 MiB |

从 12:05 UTC 开始 GPU 模型短跑，到 12:13:04 UTC 完成 Actor-only，共约 8 分钟，包含两次初始化、恢复和评价，低于用户许可的 30 分钟。
吞吐包含冷编译和短时 GPU 共用，不作为长期独占训练吞吐预测。

前4/末4更新（不同自然批次，仅诊断，不是固定验证集）：

| 指标 | 前4均值 | 末4均值 |
|---|---:|---:|
| Natural CE | 0.6500 | 0.3134 |
| Balanced CE | 0.6941 | 0.9792 |
| BC mix | 0.6588 | 0.4466 |
| Pair Loss | 0.6887 | 0.7112 |
| p_ADD(P) / p_ADD(N) | 0.4717 / 0.4695 | 0.1733 / 0.1772 |
| PairAccuracy | 0.7359 | 0.4266 |

40 个采样更新的分类 ADD Recall 3.16%、Precision 15.69%；末4 Recall 0、Precision 未定义（无预测 ADD）。
TRAIN 中 Actor 随机提议 ADD 220 次；实际 104 次入场和 104 次 TP5 均为 beta=1 的 Oracle 执行，**不能计为 Actor 自主交易**。
FG409 Actor-only 实际1000步（1001个含末尾未执行提议的合法决策），入场/TP5/SL30/Reward 全为0，All-WAIT。
独立评价收据：Run 下 `evaluations/actor_only_143eeee1ac/receipt.json`。没有 DEV 认证 Oracle 标签，未报告 DEV Oracle Recall。

实际命令（上述 Tutor Gym 工作树、指定 conda Python、CPU 原生线程各1，GPU 命令前加 `timeout --signal=INT 600 taskset -c 12-19 nice -n 10 env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fg_s3_v2_mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`）：

```bash
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare --config TUTOR_CD_V0_GPU_SMOKE_1K
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start --config TUTOR_CD_V0_GPU_SMOKE_1K --stop-at 500
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py resume --config TUTOR_CD_V0_GPU_SMOKE_1K
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py evaluate --config TUTOR_CD_V0_GPU_SMOKE_1K --eval-steps 1000
```

四次命令均退出码0；日志在 `tutor_checks/gpu_{prepare,start_500,resume_1000,evaluate_1000}.log`。
既有 Run 禁止重复 start/prepare；只可原配置 resume，改超参要新 Run。10K 模板不自动执行。

## 自审和下一步

已审查完整 diff：无 Env/Reward/Oracle/Collector/Replay/Checkpoint 算法或网络结构修改，无私有 Teacher 索引、新 optimizer、课程缓存或采样器。
唯一公共 Agent 新路线为静态 tutor；BC/Pair 只回传原 Actor，WM 原梯度保留，原联合 optimizer 每次一次。
配置身份与训练/评价接线一致；累计 Tutor 计数进入原 Counters/FullCheckpoint；无效组/Pair 指标不伪报为测量值。
原开发双仓保持原状态；原仓已有用户未跟踪文件保留，`.bz2` 未处理，`.gitignore` 未改。
结束只读检查：A2 PID 2838219 继续运行，GPU 只剩 A2 2194 MiB；冻结树 Gym `904bee6f851fd70d96dca48d06951bb466505561`、Dreamer `6356747d64963fb5ef55a036d837591f778c9f0a` 均无 tracked dirty。

下一关键方向：**授权后的静态 10K 研究中观察自然分布分数/校准与 Actor-only 行为**。当前 Pair 供给充足，但没有排序或自主入场改善证据；不因 Loss 下降扩展预算，也不预先增加 PairBank 或动态课程。本阶段停止，不启动 10K、动态 E2 或 Tutor 1M。
