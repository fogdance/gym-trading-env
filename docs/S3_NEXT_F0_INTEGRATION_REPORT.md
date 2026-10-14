# S3 Next：F0集成记录（2026-10-14）

用户已授权将完成的F0合入双仓 `integration/s3-next`。两仓在合并前干净，目标均为功能分支的直接祖先；使用普通 `git merge --no-ff`，没有冲突，没有修改功能分支历史。

| 仓库 | 集成起点 | F0提交 | Merge提交 |
|---|---|---|---|
| Gym | `03a625c1661546b9183b9ceeb24119b516a87459` | `5ab3712a235d2dced57e049f447203903a9a3f4d` | `01ccbd4b43b74c0308910a91dc99972931c030e0` |
| Dreamer | `d41bc80ba13b3dc3f5d2c32fcdc800613b2d0b4d` | `632a870d6c7de042f962c327cc984efb03aee2cf` | `eb63de710fd693fad7ac76f797f3d9f39e331b98` |

Merge完成时，两仓 Git tree 均与已验证的对应F0提交完全相同。随后只补充本记录及Git工作流说明；最终完整双仓SHA在工作区外置 `baseline-lock.json`。

## 本次验证

在集成 Gym 工作树实际执行以下CPU检查，GPU始终禁用：

```bash
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  MPLCONFIGDIR=/tmp/s3_f0_merge_mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python \
  -m pytest tests/entry_v1/test_s3_composition.py -q -o addopts=''

env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  /home/v/Documents/work/research/fg-s3-wandb-env/bin/python \
  -m unittest research_tracking.tests.test_composition -v
```

结果：组合解析/拒绝/身份合同 **36 PASS**，W&B组合身份/评价连接 **1 PASS**。原A2/Tutor/B2 CPU实际梯度、模型恢复、原生FullCheckpoint及三份真实TRAIN数据准入，沿用F0已执行收据；本次未重复模型训练，不把源码tree相同本身称为新增数值验证。

实施阶段的三项旧SourceTransfer fixture失败继续保留记录：fixture缺少显式 `S3_MARKET_MINUTE`，未修改的旧集成基线也出现同样失败。本次不放宽时钟校验、不把这些失败改记PASS。详见 `S3_F0_COMPOSITION_IMPLEMENTATION.md`。

## 集成范围与保护

- 保留原A2/Tutor/B2教学实现；新组合身份进入Prepared/Source/Model/Run/Checkpoint/evaluation及既有Tracker。
- 当前3种Legacy组合ready；3种E组合waiting_e；2种联合教学组合waiting_tutor_b2。E网络、E数据与联合Loss仍未实现。
- 包含F0验证过的精确读取版本适配，Oracle候选哈希、资金证明与原DP结果核验保留；不改数学和交易规则。
- 合并前后核验1191个受保护文件及5个非集成旧工作树，包含冻结B2/Tutor及原Archive源码；main与F0分支引用不变。
- 没有修改正式Run、Prepared、Checkpoint、Replay或W&B云端记录；没有停止/重启训练或使用GPU。

实际日志、XML、合并/保护记录保存在工作区 `validation/f0_merge_20261014_225630/`；旧版本锁保存为该目录的 `baseline-before.json`。当前集成基线已更新双仓 `baseline-lock.json`，后续功能从此验收版本建立。
