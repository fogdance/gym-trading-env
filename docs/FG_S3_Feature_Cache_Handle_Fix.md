# S3 特征缓存句柄耗尽修复

## Summary

已修复开发分支管理入口在 BC 完成后的句柄耗尽。缓存绑定/训练身份生成完成后
关闭父进程 mmap，最终 audit 无论成功或异常均关闭本次 mmap。用户已完成的
S3-A 1000 更新包独立 CPU 审计为 TRAINING_CHECKPOINT_VERIFIED，无需因该异常重训。

## Evidence observed

用户堆栈发生于 start→audit→admission→FeatureCache.__init__→np.load/mmap。
实际缓存 C1_M1_ONLINE_180K_seed0 含404分片、808个映射，402096行，宽度5120。
features/train 两个 worker 的退出码均为0，训练收据 status=BC_COMPLETE，BC更新1000。
final node `bc_000000001000`、done 和 managed_snapshot 都已存在。

在隔离 CPU 进程软限制1024的原缓存检查中，单份缓存打开后 /proc/self/fd 为812，
第二份失败 errno=24。此检查在用户“不要复现，直接改”之前已完成，之后只验证修复。
隔离限制是测试条件，不推断用户 tmux 的真实 nofile 数值。当前工具进程原限制为1048576。

## Root cause

Gym `fg_s3_experiment.py::start` 的本地 cache 在调用最终 audit 时仍持有全部
features/index 的 np.memmap；audit 再创建另一份。双份共1616映射超过1024。
原 FeatureCache 没有显式 close，依赖对象引用释放，start 的局部变量尚未释放。
异常属于训练结束后的父进程资源生命周期，不能据此判定模型训练中断或模型包损坏。

## Changes made

新增 `_admission_scope`：finally 关闭当前缓存的 NumPy `_mmap` 并清空映射列表。
start 只在绑定特征和生成运行身份时持有缓存，随后释放再启动 worker；audit 在
正常返回或审计异常时释放缓存。训练 worker 自行读取自己的缓存，没有传递已关闭数组。

未修改 frozen fg_s3a_features.py、A/B算法/标签/配置/BC包协议，未重签旧准备或产物。
修复位于 fg-s3-dev/gym-trading-env；已冻结训练 worktree 仍为5293e978/93a6fb65。
旧准备与实际源码哈希仍严格绑定，后续使用修复版本需要该版本的新准备，不能改旧收据
绕过校验。本次现有1000包已使用原冻结版本独立审计，无需启动 resume 或重新训练。

## Verification performed

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| 新 mmap 所有权回归：worker之前释放、audit成功及异常后释放 | PASS，1 passed，0.418s | 真实 NumPy mmap；worker/checkpoint mock，不是模型训练 |
| 修复 helper + 真实404分片缓存，两次顺序准入 | PASS，两次均4→812→4 | 原缓存与全部文件/标签/代码哈希，受控CPU，未执行模型更新 |
| 已完成1000包的原冻结 CLI底层 audit，软限制1024 | PASS，TRAINING_CHECKPOINT_VERIFIED，frozen_keys=275 | 实际完整atomic包的只读审计；无新神经前向/梯度 |
| 1000 checkpoint manifest 前后 SHA | PASS，不变 | c26351abc7e8e319f49c11017f3db88d431d5fa04c84270272019ee1df9b109a |
| 原训练/研究源码、.gitignore、旧A/B关键模块 | PASS，定向Git/diff核对 | 源码保护，不是生产运行验收 |

Gym 收据目录 `artifacts/fg_s3c1_v1/receipts/fd_exhaustion_fix/` 保存原始stdout、XML
和 verification.json。真实缓存 probe 只替换 admission 的调用编排为原 FeatureCache
构造器，验证新增所有权 helper；没有跳过 FeatureCache 的源/标签/包/文件校验。
未将 helper 或 mocked worker 的通过称为修复后的正式GPU start验收。

真实测试命令（Gym开发目录）：

```bash
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/v/Documents/work/research/fg-s3-dev/dreamerv3:/home/v/Documents/work/research/fg-s3-dev/gym-trading-env/src \
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
$PY -m pytest -o addopts= -q -p no:cacheprovider tests/entry_v1/test_fg_s3_feature_lifecycle.py --junitxml=artifacts/fg_s3c1_v1/receipts/fd_exhaustion_fix/cache_lifecycle_after.xml
```

## Risks / not verified

新GPU训练、DEV、封存TEST为NOT_RUN。用户既有GPU BC已完成，不与本次CPU检查混计。
ALL_WAIT_COLLAPSE 是既有训练收据的分类结果，本修复不改变它；本次只处理句柄。
上轮复查 R1日志恢复、R2诊断标记、R3正式准入仍OPEN，未在此次顺带修改。

## Next recommended step

现有 run 只需独立 audit，tmux 中直接执行，无 nohup，无 resume：

```bash
cd /home/v/Documents/work/research/fg-s3-train/gym-trading-env
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
env PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/v/Documents/work/research/fg-s3-train/dreamerv3:/home/v/Documents/work/research/fg-s3-train/gym-trading-env/src \
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
"$PY" tools/fg_s3_experiment.py audit --config S3A_ORACLE_BC_FROM_ONLINE_180K
```

已完成的同等底层 audit 输出保存在 completed_1000_checkpoint_fresh_audit.stdout。
下一次新实验使用修复后的源码版本和新的准备快照；保留当前运行及其冻结身份。
