# F1 checkpoint source-hash incident — 2026-10-21

## 观察与证据

用户日志在第 3,000 次更新后的 Checkpoint 保存前报错：
SUPERVISION_CONTENT_CHANGED: teacher_sidecar/P0/steps.jsonl.gz。
最新发布指针为 step_000000002500，未产生第 3,000 步节点；updates.jsonl 的后 500 步尚未提交。

23:02 的独立流式 SHA256 和 23:04 的原训练校验函数均得到同一哈希：
af12f5c498555dc3bb2510b32b0644a4435ff9c9a9d6957e58dedc77c7430cc8。
该值同时等于 Corpus Manifest 和正式发布批次的记录。
目标大小 2,029,127,663 字节，mtime/ctime 为 2026-10-21 02:45:59.990666+08:00。
Corpus Manifest 本身 SHA256 仍为 f447818f32fa43d72e8aa6eb0dcffcd1b63f64dac837dc14b6944cf7e49665a7。

用户提供 22:25–22:36 的 sudo journalctl -k 输出：-- No entries --。
因此没有观察到持久文件改动；也不能从空内核日志推断硬件、存储或内存一定正常。
原错误没有保存实际哈希和读取状态，本次未复现原失配，**具体根因未确认**。

## 已修改

fg_history_study.digest 原先一次 read_bytes 读取整个 Sidecar，导致该路径需要单次分配约 1.89 GiB 的 bytes。
改为 4 MiB 分块计算相同的 SHA256，不改变哈希算法、预期哈希、文件内容或接受条件。
这是已确认的大文件内存开销问题的修正，不能据此宣称原失配根因已解决。

inspect_source 在失配时记录期望/实际 SHA256、完整路径、大小、inode、mtime/ctime，供复发时定位；
仍然失败关闭，不自动重试放行，不改写或重签教材。
不修改金融、Teacher、训练 Loss、优化器或 Checkpoint 恢复规则。

## 定向运行核对

仅执行真实目标文件与既有 Checkpoint 的只读核对，没有运行测试集、G3 或 GPU 训练。
命令使用原 Dreamer Python，PYTHONPATH=.:src:../dreamerv3、JAX_PLATFORMS=cpu、CUDA_VISIBLE_DEVICES 为空。

实际调用：
~~~python
observed = fg_history_study.digest(corpus / 'teacher_sidecar/P0/steps.jsonl.gz')
assert observed == manifest['files']['teacher_sidecar/P0/steps.jsonl.gz']
meta = validate_bundle(node, identity)
validate_offline_journal(root, node)
~~~

退出 0：流式哈希与原值完全相同，耗时 2.618 秒。
最新发布 Checkpoint 文件包及 journal 提交前缀通过：
step=2500、optimizer_updates=2500、consumed_batches=2500。
目标位置 1,280,000，burn-in 位置 640,000。
这是文件与已提交历史的核对，不是模型冷恢复或后续 GPU 更新验收；这些仍为 NOT_RUN。

## 恢复

保留原 Prepared、Corpus、Checkpoint、updates.jsonl 和原启动身份锁备份。
新启动锁只更新修复后的训练 Gym 提交；历史 Run 身份不修改，Dreamer 提交不变。
恢复使用原正式命令：
~~~bash
bash /home/v/Documents/work/research/fg-s3-f1-certified-20261021/gym-trading-env/docs/f1_fg809_20261021/RUN_F1.sh resume
~~~

仍由 STRICT_PUBLISHED_LATEST_V2 选择已发布第 2,500 步；按原日志协调流程保留/处理未提交尾部。
不会冒充恢复到未保存的第 3,000 步。用户执行训练，本次未启动。
本修复未自动合并到 integration/s3-next。
