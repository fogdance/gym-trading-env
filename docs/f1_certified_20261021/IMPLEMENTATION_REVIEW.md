# 正式教材认证复用与 F1 落盘修复

## 观察与根因

旧 F1 的 f1-prepare 在 GPU 更新之前重跑了 FG809 全主力期六行为 Native G3。日志显示四行为 PASS，P0/P4 尚未完成，实际模型更新为 0。
旧 start_prepared 又强制调用 native-verification，prepare 成功后仍会第二次重验。发布认证没有被训练入口使用。

FG809 正式生产已于 2026-10-21 13:56 完成六行为 Native G3，每行为 28,291 条完整标签、27 次账户恢复。正式发布 attempt 0002 的 Corpus 身份与 F1 相同：
76a856fb85ea526800bf2cfd6ce51be1ae26c8f7ac70dea0904bba6b4b629550。

## 修改

新分支 fix/f1-certified-corpus-20261021，以 Gym a4c82b96f145bd547e7566c4539411f9abd7f1db、Dreamer 990f61f8b7ac6bad25ff9e41b65624ae13ccf497 为基线。

共享准入新增 certified-corpus，F1/F2/F3 配置可显式绑定 corpus_certification（发布 BATCH_MANIFEST 路径和独立记录的 SHA256）。
训练直接验证原发布批次、身份锁、完整文件清单、原 G3 日志/Pin、六行为完整标签/Trace 收据、Reader 结果及原 TRAIN 权威。
来源内容身份和 ABI 保持不变，不要求当前训练代码 HEAD 等于历史 Producer。
复用证据明确标记 PUBLISHED_NATIVE_G3_REUSED、fresh_native_execution=false，不声称发生新的原生执行。

prepare_join_index 复用已独立验收的原 Reader SQLite/JSON，复制后验证哈希，不新建 Replay 或改写教材。
prepare/start/resume 均选择相同认证路径，启动时重新核对身份，保留外部训练授权和现有进程内准入对象校验。
显式 native-verification 仍供生产认证使用，不隐式退回去重跑 G3。

新 F1 启动脚本将 Prepared、缓存、TMPDIR/TMP/TEMP、Matplotlib/JAX/CUDA 缓存与启动锁放在 /data/logdir。
模型、指标和 Checkpoint 原本已在 /data/logdir。复验日志路径改为 Prepared 同级 preflights。
不改金融 Env、Reward、Teacher、Loss、优化器、FullCheckpoint 或恢复规则；Dreamer 没有源码改动。

旧 F1 专属进程组 2892975 已停止，全部已知计算子进程退出。
约 13 KB 的不完整准入日志保留；3.2 GiB 可重建 G3 临时值缓存已按用户要求删除。
教材、正式生产工作树与已有 Run 不修改。新分支不自动合回 integration/s3-next。

## 真实命令与验证边界

用户要求停止增加测试后，没有再启动测试、prepare 或训练。

此前已经执行的真实 FG809 只读复用检查（JAX_PLATFORMS=cpu、CUDA_VISIBLE_DEVICES 为空）：

~~~python
s = read_settings('configs/experiments/fg_s3_v2/e_quality_f1_fg809_size12m_u10000_20261021.yaml')
p = preflight(s, purpose='certified-corpus')
~~~

实际退出 0，128.758 秒，返回 F1_DATA_READY、ABI 3.1.0、
PUBLISHED_NATIVE_G3_REUSED、fresh_native_execution=false、optimizer_updates=0。
这是已有约 12 GiB 工件的读取/哈希及来源校验，不是 G3 或训练。
随后增加的身份锁一致性与恢复绑定检查仅静态核对；未再次执行真实核验。

用户停止增加测试之前启动过一次合成接口测试：
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONINTMAXSTRDIGITS=0 MPLCONFIGDIR=/tmp/fg_s3_certified_tests_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider tests/entry_v1/test_corpus_certificate_reuse.py --junitxml=docs/f1_certified_20261021/certificate_reuse_checks.xml -q

退出 1，六项均因同一 fixture 缺少 trace_manifest 的 E_QUALITY_SCHEMA 参数而在 setup 报错；参数已修正。按用户指令没有复跑，不声称该测试通过。
原始 XML 保留。F1/F2/F3 最终接口回归、共享 Gate 回归、GPU 更新、训练冷恢复为 NOT_RUN。
git diff --check 和启动脚本语法检查已通过。

## 正式命令

~~~bash
bash /home/v/Documents/work/research/fg-s3-f1-certified-20261021/gym-trading-env/docs/f1_fg809_20261021/RUN_F1.sh start
~~~

新 Run 首次使用 start；已有检查点后中断使用 resume。训练由用户执行。
