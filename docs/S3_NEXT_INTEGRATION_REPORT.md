# integration/s3-next 实施与验收

本页记录首次建立基线的验收。随后 Archive V1 已按独立任务集成，最新范围、冲突处理及测试见 `S3_NEXT_ARCHIVE_INTEGRATION_REPORT.md`；最终双仓身份以工作区 `baseline-lock.json` 为准。

## 已实施的分支与工作区

两仓实际创建 `integration/s3-next`，工作区为：

- `/home/v/Documents/work/research/fg-s3-next-integration/gym-trading-env`
- `/home/v/Documents/work/research/fg-s3-next-integration/dreamerv3`

Gym起点 `3970e6ec69f67fb2d02f30456a077ea5a31fd05d`；Dreamer起点 `5fd4ebdeb590fd8433338300bd82c7f2f520bf76`。
最终提交、实际分支/Worktree/status和验收证据在工作区根目录 `baseline-lock.json` 与 `validation/`，避免双仓相互写SHA形成循环提交。

原 main：Gym `1eef479b7c3a2fafc8c9c792f874a671c8fc2b4a`，Dreamer `e01491fad6434b2245a3b8ca201dd7faedcc458c`。本次不更新main或旧功能分支。

## Git与功能核对

| 来源 | 实际关系及处理 |
|---|---|
| Tutor Full / B2 Full / Full Integration | 两仓均为相同起点；直接继承，不重复移植 |
| W&B Gym `ad6490675bf853a7793763f95aa7ba7f12fa379e` | 起点祖先，已有完整Tracker并包含后续Full指标/代理改进 |
| W&B Dreamer `d76a423dd7930756c3609cc6f7197941f583c160` | 起点祖先；W&B SDK仍位于Gym独立环境 |
| Market Minute Dreamer `c18a8b5917f73110f9b6ce840495f4da1d7ca1d4` | 起点祖先，时间语义已包含 |
| Market Minute Gym `bf3c8b8d41ebcf320e77f624a0682ac0d59efc95` | 仅未合入一份数据准备/验收报告，无未合入源码；不重复合并 |
| Archive Gym `b7d5edb30d1dcb2f2d845593a4cef498d923536d` | 未包含，按要求排除；Dreamer没有同名Archive分支 |

全部旧分支完整SHA、全部Worktree/dirty状态见 `validation/before.json`；关键merge-base、祖先关系、未合入提交和文件见 `validation/git_audit.json`。
原根仓库的用户未跟踪文件原样保留，未解压或处理bz2。

真实扩展点：Dreamer `fg_s3_v2_agent.py` 按 `a2/tutor/b2` 分派；Tutor使用原 `fg_s3_tutor_loss.py`，B2使用原 `fg_s3_b2_outcome.py`。公共训练、Replay/Sidecar、Checkpoint与Env/Oracle/Reward源码未改动。

W&B已包含 `tools/fg_s3_wandb.py`、`research_tracking`、稳定Run身份、Checkpoint确认边界、provisional监控、配置/评价关联、offline导出、sync恢复和Launcher异常隔离。Project仍为 `fg-single3-research`，SDK实测 `0.19.1`，模块默认代理仍为 `http://127.0.0.1:10808`。

仅新增一个Mock同步回归：offline发布后关闭状态、模拟同步失败、重新打开并成功重试、再次同步为空；确认Run ID及原source/prepared文件未变化。没有第二套Tracker，也没有新增生产接口。

## 本次实际测试

| 测试 | 结果 | 范围 |
|---|---|---|
| CPU模型与FullCheckpoint | 20 PASS，219.34秒 | 小模型真实CPU梯度、A2构建/策略等价、Tutor分项教学、B2预热/StopGrad/联合优化与完整恢复 |
| 公共合同 | 44 PASS / 1 SKIP | 受控原Env、市场分钟、Oracle、Replay/DAgger/合约池、Tutor数值/账户边界、预算及恢复 |
| W&B现有测试 | 21 PASS，13.945秒 | Mock、真实Offline SDK、启动器、日志/回滚、身份与脱敏、原Trainer环境及异常隔离 |
| 新W&B同步恢复 | 1 PASS，0.099秒 | Mock传输，无云端API访问 |
| 配置/导入 | PASS | A2/Tutor Full/B2 Full独立模式、100/100/100、兄弟工作树导入、实际CPU后端 |

SKIP是新工作区没有历史Tutor E2 GPU 10K Run收据的只读检查，未计为PASS。GPU/重型历史模型案例按命令排除；新版真实GPU、正式训练、云端上传验收均NOT_RUN。

使用既有环境，没有安装或升级依赖。Dreamer实测JAX/JAXLIB 0.4.33、Ninjax 3.5.1、Optax 0.2.4、NumPy 1.26.4；配置/导入收据记录其他实际版本。

### 精确执行命令

在集成Gym目录，模型和公共合同命令均附加：

```bash
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  MPLCONFIGDIR=/tmp/s3_next_mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest \
  ../dreamerv3/embodied/tests/test_fg_s3_b2.py \
  ../dreamerv3/embodied/tests/test_fg_s3_tutor_full.py \
  ../dreamerv3/embodied/tests/test_fg_full_checkpoint.py -q

env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  MPLCONFIGDIR=/tmp/s3_next_mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest \
  tests/entry_v1/test_s3_market_minute.py tests/entry_v1/test_fg_s3_b2_outcome.py \
  tests/entry_v1/test_fg_s3_tutor_cd.py tests/entry_v1/test_fg_s3_tutor_e2.py \
  tests/entry_v1/test_fg_s3_v2_pool.py tests/entry_v1/test_fg_s3_v2.py \
  -k 'not small_cpu and not real_native and not actual_gpu and not real_original and not real_gpu' -q

env PYTHONDONTWRITEBYTECODE=1 /home/v/Documents/work/research/fg-s3-wandb-env/bin/python \
  -m unittest research_tracking.tests.test_tracker research_tracking.tests.test_eval_v2 \
  research_tracking.tests.test_prediction_evals -v

env PYTHONDONTWRITEBYTECODE=1 /home/v/Documents/work/research/fg-s3-wandb-env/bin/python \
  -m unittest research_tracking.tests.test_tracker.TrackerTests.test_offline_sync_retries_after_reopen_without_changing_source -v
```

原21项W&B组合命令先执行；新增1项随后单独执行，未把它们伪称为一次22项组合运行。配置检查的完整命令、stdout及日志SHA见版本锁与validation文件。

## 保护与签收范围

1174份B2冻结源码/不可变Run和Prepared文件在实施前记录哈希；交付前复核。原分支refs、原Worktree状态与main一并检查，实际结果写入 `baseline-lock.json`。活动训练自身追加日志和Checkpoint照常允许。

本次集成只增加Git工作流文档、验收报告和必要同步测试；无算法/网络/训练配置/Reward/Oracle源码修改，无历史迁移、Archive/E/联合教学，无正式云端或GPU训练操作。

结论：公共代码基线的CPU/Mock/Offline验收通过；不能解释为三种策略已经学会交易。下一项功能从版本锁记录的验收后 `integration/s3-next` 建立。
