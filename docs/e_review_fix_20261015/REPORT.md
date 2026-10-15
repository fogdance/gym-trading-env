# E Review 修复与当前训练证据

## Summary

CSV最终身份缺口已在隔离修复分支关闭；21项定向用例最终全部通过。运行中的正式1M继续使用原基线，本修复没有热修改该运行。

工作树：`/home/v/Documents/work/research/fg-s3-e-review-fix-20261015/{gym-trading-env,dreamerv3}`，分支均为`fix/s3-e-market-identity-20261015`。Gym修复基线`871ca09000712d41408b8f72e0684624dce8659f`；Dreamer仍为`730939defb1bd77b2ceca04212105ec2727c95f9`，没有模型代码修改。

## Evidence observed

真实`ETrainAdmission.from_train_pool`验证临时签名pool和collection后，替换CSV：原代码在无缓存、缓存冷加载、已有缓存三种情况下均未拒绝。解析前替换文件还使Env实际Volume从100变为101；原代码也接受无法绑定来源的注入DataFrame。修复前5项全部失败，见[evidence/market_identity_before.log](evidence/market_identity_before.log)。这些是隔离合成行情上的真实准入/Env运行，不冒称真实TRAIN数值验收。

主机只读进程检查发现正式训练PID1034433已经在原`fg-s3-e-model-dev`运行；Tracker PID1034432。原两仓HEAD与干净状态保留，见[evidence/summary.json](evidence/summary.json)。本轮未启动、停止或修改正式训练，未修改W&B与盲测TEST。

## Root cause

准入验证了路径当时的内容；Env随后重新按路径解析CSV，没有把其实际消费内容与该合约收据中的SHA256绑定。Cache虽然计算当前SHA256，却只用于查找Key，因此新字节能形成新的缓存项并沿用旧E身份。只在解析前额外hash路径，也不能保证校验与解析使用相同字节。

## Changes made

- E的CSV解析使用固定的字节快照：先检查收据中的合约/路径/SHA256，再将该同一份字节交给唯一一次CSV解析；`e_market_source`记录实际通过校验的路径与哈希。Cache原有的文件哈希读取保留。非E路径维持原读法。
- Cache在命中或冷加载前也校验收据，不再将当前字节Key等同于来源认证；原有文件变化检查保留。
- E缺少来源收据或通过DataFrame注入无法证明来源时拒绝；旧非E的DataFrame功能保留。
- 在原严格Reader认证中新增本次两个文件的完整精确哈希组合，保留所有旧组合；未知字节、伪造来源和exact resume仍不放宽。见[evidence/runtime_fix.patch](evidence/runtime_fix.patch)。
- 明确共享MarketData构造后按只读对象使用的约定，并验证跨账户step、reset、恢复后其DataFrame、数组、时钟、桶和事件内容哈希稳定。不引入第二套缓存，不修改E计算公式。本轮没有将Pandas对象变成强制不可变容器。

## Verification performed

命令在隔离Gym工作树执行，使用`/home/v/miniconda3/envs/dreamerv3/bin/python`；设置`PYTHONPATH=src:.:../dreamerv3`、`JAX_PLATFORMS=cpu`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`PYTHONDONTWRITEBYTECODE=1`。

```bash
python -m pytest -q \
  tests/entry_v1/test_s3_e_market_identity.py \
  tests/entry_v1/test_s3_e_native_env.py \
  tests/entry_v1/test_entry_market_cache.py \
  tests/entry_v1/test_s3_composition.py::test_reviewed_reader_warm_start_still_rejects_unknown_changes \
  --basetemp=artifacts/fg_s3_v2/checks/e_market_identity_review_20261015 \
  --junitxml=docs/e_review_fix_20261015/evidence/targeted_final.xml
```

结果：21 PASS、0 FAIL、0 ERROR、0 SKIP。覆盖替换CSV三条路径、实际解析字节竞态、注入/缺收据拒绝、E共享内容稳定性、原18维/Oracle身份不变、单次CSV解析与原特征build、非E缓存/账户隔离/冷恢复及严格来源拒绝。自审后把内容哈希测试进一步改为显式调用`reset()`和`reset(options={'resume': state})`，只复验该1项，最终PASS。日志：[targeted_final.log](evidence/targeted_final.log)、[shared_hash_reset_final.log](evidence/shared_hash_reset_final.log)，JUnit与摘要同目录保留。

修复后首轮有2项Collector测试失败，原因是本轮测试输出目录放在允许的`artifacts`/`/tmp`之外。仅修正`--basetemp`后全部通过，没有放宽Collector输出守卫。显式reset自审的首轮夹具又在未完成账户上调用普通reset，原保护正确拒绝；把普通reset移至合法初始边界后，同一项通过，没有改环境账户规则。所有失败日志和JUnit保留；另保留初始证据目录缺失、只读诊断误用Checkpoint组件名称的命令错误，见summary。未运行神经模型训练或重复大套件。

## Risks / not verified

| Review项 | 当前证据与边界 |
|---|---|
| P0 Env最终SHA256 | 隔离分支修复并通过失败前/通过后回归；尚未进入运行中1M源码 |
| P0 GPU1K及恢复 | 原1K smoke中止记录保留。现有正式Run的来源收据显示`cuda:0`、完整size50m（RSSM4096、bfloat16）；已完成500K Checkpoint/20000更新，可复用实际训练证据。未执行独立GPU冷恢复，也没有完整显存峰值审计；不宣称这两项PASS |
| P1共享可变MarketData | 只读约定和正常step/reset/恢复内容哈希回归PASS；不声称能阻止未来代码蓄意修改内部Pandas对象 |
| P1有效Actor学习 | 500K已提交Checkpoint：有正ADD监督的BC更新14189次，有效BC更新19629次；PairCoverage70.885%，Pair更新14177次/1921273对；Gate READY更新0，RL Actor/Critic更新0。只读校验了计数和Gate组件SHA，不冒称完整Checkpoint恢复 |
| Actor自主ADD | Collector保留Actor提案及累计ADD数量，但beta1在教师轨迹上执行。提案频率不等于Actor自主交易成绩；Actor-only评估未执行 |
| 15m未来泄露 | 本轮未改core/schema/reference或公式。原54个真实前缀抽样和已有夜盘/换日/边界/轮换/恢复证据仅按原范围保留；没有将其升级为全部决策时刻的证明，也未补跑用户建议的完整关键时点集合 |

运行证据只读快照与实际文件路径：[formal_run_readonly_snapshot.json](evidence/formal_run_readonly_snapshot.json)。其中500K Checkpoint是固定的已提交边界，另附的live update/cycle行来自仍在运行的任务，时间边界不同，不能混加。Tutor模式在源码中固定`rl=False`，只有C2才允许RL；因此Tutor的RL计数为0符合当前配方。BC与Pair能独立训练Actor，仍不代表策略效果已通过。

## Next recommended step

本次交付独立提交，不自动合并正在运行的`feat/s3-e-model`。若采用修复，应使用新冻结代码、新准备身份和新Run；不能修改旧收据或将不同源码伪装为原Run的exact resume。运行中1M的停止、切换、恢复或额外GPU验收均等待用户明确指令。本报告只对CSV身份修复给出PASS，不为GPU冷恢复、独立Actor交易或整体生产准入补签GO。
