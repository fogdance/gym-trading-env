# S3-V2 A2：beta 递减与 18 合约 GPU 短跑

2026-10-13。结论 **GO_WITH_LIMITS**：当前固定 Online180k 来源的启动准备无已知工程阻断。
完成 beta 调度、真实 18 合约交互和 GPU 同 Run 恢复；Codex未启动正式1M。
不改 WM/BC Loss、网络、Reward、Oracle 数学，不开发 C2。

## 推荐的正式调度

| 已完成的真实环境步 | Oracle beta | 调度 |
| --- | --- | --- |
| 0–10K | 1.00 | Oracle 主导预热 |
| 10K–100K | 1.00 → 0.98 | 线性递减 |
| 100K–1M | 0.98 → 0.90 | 线性递减 |

beta 作用于已认证的主动决策；无效 Oracle 答案仍不进入 BC。
按实际 Env 推进计量，reset、Native Replay 行数及梯度更新不推进 beta。
Driver 先执行上一次动作，再选择下一动作；调度在当前决策处包含刚发生的真实推进，避免晚一个动作。
现有 FullCheckpoint 的 collector.mixing 保存调度点、环境进度、当前 beta、RNG 和待执行决策；恢复校验配置与已提交边界。
按用户最后要求，正式保存间隔为 **100K 环境步**；沿用现有 FullCheckpoint 保留行为。
100K 检查自主 ADD、有效监督和入场效果，持续 ALL_WAIT 或效果恶化时利用已有恢复点调整研究，不机械等待 1M。

## 实际 GPU 结果

使用指定 Conda、原 42,584,132 参数 Replay 模型、ALL_TRAIN 18 合约和原 least_progress 调度器。
诊断窗口 200 bars（正式为 1000）；加速调度为 beta=1 到3600、5200时0.98、8000时0.90。
**8000真实环境步、18/18实际交互、39次真实合约切换**。
5K 保存退出，新 GPU 进程恢复到 8K；累计模型更新 21183→21383→21503。
320 次 WM 更新、307 次有效 BC（215 次含 ADD）；恢复后继续120次更新。
保存参数证明 Actor 与 WM 继续变化，Critic、slow target、Normalizer及Critic优化器状态不变。
Replay/Sidecar 的8039个已提交 owner与执行动作一致；另有一条 pending，未发布到监督。
原来源 manifest与agent字节哈希均保持不变；manifest SHA256为
`4979acfb0272ffbc3c96833e5b77fccd66c5d4d0f335bfca48810f764d927a57`。

| 加速短跑阶段 | 全部环境步的有效标签覆盖 | 合法决策的有效覆盖 | Actor实际接管合法决策 | 自主执行ADD |
| --- | --- | --- | --- | --- |
| 0–3600 | 81.36% | 100% | 0 | 0 |
| 3600–5200 | 85.63% | 100% | 14 | 0 |
| 5200–8000 | 64.71% | 100% | 96 | 6 |

总计6111/8000有效标签（76.39%）；所有6111个合法决策均获认证，资金域外状态0。
未监督步骤的Action Mask不允许ADD；其强制WAIT未伪装成认证BC标签。
302成交、298 TP、SL0、Reward271.0078属于混合TRAIN成绩；4笔尚未完成。
6次自主ADD只证明Actor执行路径真实工作。此次未执行Actor-only评价，未证明独立交易能力改善。
核心短跑吞吐9.20步/秒，收集738.92秒、更新130.49秒；恢复进程峰值RSS3777MiB，首段3869MiB。
正式1000-bar轮转吞吐及长期学习效果尚未实测。

## 冻结版本与正式命令

冻结目录 `/home/v/Documents/work/research/fg-s3-v2-train-beta/`，两仓detached、tracked clean：

- Gym `904bee6f851fd70d96dca48d06951bb466505561`
- Dreamer `6356747d64963fb5ef55a036d837591f778c9f0a`

实际短跑Gym为`8a69bec3a93220cf03e0845b26a70fb4df35b84c`；到冻结Gym仅增加README和正式配置100K保存设置，src/tools字节无变化。
冻结树复用dev的两个只读输入软链接：`teacher_catalog_v1_final`与`online_180k_source`；行情/Oracle沿用授权的`research/fg-data`数据别名。
原research仓库、旧训练树和旧模型/日志保留。
冻结树prepare完成：`artifacts/fg_s3_v2/prepared/A2_GPU_FORMAL_1M_BETA_v1`，18合约、1M环境步、40000联合更新预算、Replay容量8192。
准备manifest SHA256可由其文件直接核对；签名content_hash为`d324ecff779da7a42d2eae9d2bd3b17f3cddce7d8f5f9558506738c7e2d749b0`。
新Run `/data/logdir/fg_single3/s3_v2/A2_GPU_FORMAL_1M_BETA_v1`。
报告最后检查时，用户已启动进程2838219，cwd确认为上述冻结Gym；进程仍在载入，Run目录已经创建，尚未生成训练身份/Checkpoint。
下列命令与用户已执行的启动命令一致，当前Run不要重复start：

```bash
cd /home/v/Documents/work/research/fg-s3-v2-train-beta/gym-trading-env &&
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start \
  --config A2_GPU_FORMAL_1M_BETA --authorize-training
# 同Run恢复：将start换成resume，其余参数相同。
```

## 验证命令与收据

```bash
env PYTHONPATH=src:../dreamerv3 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fg_s3_v2_mpl /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q tests/entry_v1/test_fg_s3_v2_pool.py tests/entry_v1/test_fg_s3_v2.py -k 'beta_schedule or pool_rotation or live_committed or formal_budget' --junitxml=artifacts/fg_s3_v2/receipts/beta_rotation_contracts.xml
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py prepare --config A2_GPU_ALL_TRAIN_BETA_8K
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py start --config A2_GPU_ALL_TRAIN_BETA_8K --stop-at 5000
/home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_v2_experiment.py resume --config A2_GPU_ALL_TRAIN_BETA_8K
env PYTHONPATH=src:../dreamerv3 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fg_s3_v2_mpl /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -q tests/entry_v1/test_fg_s3_v2_artifacts.py -k actual_all_train_beta --junitxml=artifacts/fg_s3_v2/receipts/beta_actual_gpu_artifacts.xml
```

4项控制数据/公共接口测试PASS；1项已保存真实GPU工件验证PASS（无CPU神经计算）。
工件检查脚本首次收集有括号错误，已修复；失败XML保留为`beta_artifact_collection_failure.xml`。
主收据 `artifacts/fg_s3_v2/receipts/beta_all_train_summary.json`，三份`beta_all_train_*_terminal.log`保留真实命令输出。
正式prepare输出在冻结树的`artifacts/fg_s3_v2/receipts/beta_frozen_formal_prepare_terminal.log`。
PASS：真实18合约GPU、beta递减/恢复、有效标签覆盖、动作/Replay绑定、WM/Actor参数学习、来源保持。
NOT_RUN：正式1M完成验收、正式10K/100K学习里程碑、此次Actor-only、长期全部生命周期轮转、C2/消融。
