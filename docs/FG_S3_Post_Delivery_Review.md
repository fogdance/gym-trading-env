# FG S3 交付后代码 Review：未修复问题清单

## Summary

复查对象：Gym `3785ad6199581779468d4c425fd02364f1ac6bb5`、Dreamer
`1b7a5120e43c9b872518fbf692bc4bad168381b8`。
主体 AR0–AR5 / C0–C4 已有实现与 CPU 验收收据；本次新增复查发现 **3 项 P2
问题，均未修复**。既有 PASS 表示当时已执行场景的结果，不能覆盖下面的新失败。
按用户“先列出问题”，本次仅保存问题、复现与报告，运行时代码未修改。

## Evidence observed

逐项阅读两份正式任务书、Catalog/adapter、SourceAdmission/transfer、S3CAgent、
S3CRun、原 FullCheckpoint/RuntimeState/streams、A/B 原子协议和管理入口。
原 Agent.train 相对基线仅增加默认 loss factory；C 使用 posterior repfeat、
stop_gradient、原 masked Actor、原联合 optimizer 一次更新。四个 C 运行时文件
与 `b382cd9` 神经验收版本字节相同，六份既有验收 XML SHA 均与收据匹配。
这些是代码与历史证据核对，未当作新的生产运行 PASS。

原研究和冻结训练 worktree 的 tracked 文件保持 clean。训练版本仍为 Gym
`5293e978d3f234d5f7c3149f326f1da831012bad` / Dreamer
`93a6fb65b493b60f7c1205b8351d7024cbfc326a`。原 A/B 冻结模块及 `.gitignore`
的定向字节比较一致；`.bz2` 和其他原有 untracked 文件未操作。

## Root cause

### R1 / P2：Checkpoint 恢复没有恢复更新日志边界

位置：Dreamer `embodied/run/fg_s3c_train.py:246`、`:297`。
`update()` 先追加 `updates.jsonl`，随后按 `save_every` 保存完整状态；resume
恢复 Agent/local counters 后没有处理日志尾部。真实小模型复现中，受控选择
K=1 的已验证 FullCheckpoint，恢复后 `step=1`，日志仍为 `[1,2,3]`。
继续追加会再次记录步骤 2、3。正常保存间隔内的异常中断也经过同一恢复分支。

影响：日志步号、累计指标与实际恢复模型的历史不一致；本次没有发现模型参数
或 sampler/queue 恢复错误。之前 exact-resume 测试比较模型/流/计数，未检查日志。

建议：以实际恢复的 Checkpoint 为提交边界，保留回退前日志的独立归档，恢复
一致的有效日志；覆盖未保存尾部和不完整末行。不得通过重签 Checkpoint 解决。

### R2 / P2：非随机来源遗漏 Catalog 的 diagnostic 标记

位置：Dreamer `embodied/run/fg_s3_source_admission.py:247`、`:252`、`:297`。
只有 direct_init 合并 `catalog.manifest['diagnostic']`；direct_full 使用来源标记，
after_bc_s3a 使用 A 包标记，公共尾部仅处理调用方显式 `diagnostic=True`。
隔离公共合同复现：Catalog diagnostic=true，完整来源 diagnostic=false，
返回 receipt.origin.diagnostic=false。

影响：来源收据遗漏 Catalog 的诊断 provenance，调用方仅检查收据时无法据此
拒绝正式准入。该复现对 `_full` 使用明确 mock，属于协议测试；未执行真实
生产来源 + 诊断 Catalog 的完整正式 prepare/start，不能声称生产 GPU 已越界。

建议：在共享 SourceAdmission 为所有来源统一合并 Catalog/来源/显式诊断标记，
随后回归公共合同，再验证 C 调用方；不得在 C 新写第二套来源判定。

### R3 / P2：S3-A V2 正式准入没有拒绝诊断来源

位置：Gym `src/gym_trading_env/research/fg_s3_experiment.py:329`。
A V2 admission 调用共享 inspect_source 后直接返回 origin，没有检查正式
experiment 的 diagnostic=false 与已审计 origin.diagnostic=true 的冲突。
使用 R1 的实际原神经小模型 FullCheckpoint、完整 Catalog 和标签，只读调用
该入口，输出 experiment_diagnostic=false / source_diagnostic=true，未拒绝。

影响：A V2 来源准入的正式/诊断边界失效。后续模板等检查仍可能独立拒绝该
小模型；本次只证明 admission 边界遗漏，未运行完整正式 prepare/start。
当前选定 180k 收据的 source diagnostic=false、Catalog diagnostic=false，
不触发 R2/R3 的诊断组合；未因此操作现有实验或训练进程。

建议：A V2 消费共享收据的最终诊断标记并 fail closed，保持原 A 算法及旧包不变。

## Changes made

仅新增这份问题报告与新的复现收据；历史验收报告增加指向本报告的状态说明。
运行时、网络、训练 YAML、冻结 worktree、数据、模型和现有收据均未修改。
三个问题状态为 OPEN，未将失败写成修复通过。

## Verification performed

| 新执行 | 结果 | 证据层级 |
| --- | --- | --- |
| R1：真实 CPU 小模型 + 原 FullCheckpoint，额外核对恢复日志 | FAIL，恢复步1 / 日志1,2,3 | 实际原神经运行；受控回退，非生产训练 |
| R2：诊断 Catalog 的 direct_full 标记传播 | FAIL，true 变为 false | 隔离公共协议 fixture，非神经测试 |
| R3：A V2 正式 admission 接收诊断完整来源 | FAIL，未拒绝 | 实际小模型包的只读准入；未执行正式 prepare/start |

R1/R2 共2 failed、0 errors，57.352s；R3 共1 failed、0 errors，0.925s。
没有重跑全部已通过验收，仅执行本次遗漏合同的复现。

收据位于 Gym `artifacts/fg_s3c1_v1/receipts/post_delivery_review/`：
`review_commands_and_findings.json` 保存真实命令、环境、Git HEAD 和文件 SHA；
`review_evidence_checks.json` 保存代码/历史收据核对。原始 stdout/XML 全字节
压缩为 `.raw.gz`，对应 Python 复现源码以 `.source.txt` 保存。第二个命令只读
引用第一个命令已生成的 `/tmp/pytest-of-v/pytest-221/actual_source0` 小模型包；
重新复现需要替换为对应临时 fixture 路径，原执行路径未作伪造。

```bash
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/v/Documents/work/research/fg-s3-dev/dreamerv3:/home/v/Documents/work/research/fg-s3-dev/gym-trading-env/src
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# 以下临时文件的原始源码保存在收据 .source.txt 中。
$PY -m pytest -o addopts= -q -s -p no:cacheprovider /tmp/fg-s3-review-20261012/test_post_delivery_review.py --junitxml=/tmp/fg-s3-review-20261012/review_reproductions.xml
$PY -m pytest -o addopts= -q -s -p no:cacheprovider /tmp/fg-s3-review-20261012/test_post_delivery_a_admission.py --junitxml=/tmp/fg-s3-review-20261012/review_a_admission.xml
```

## Risks / not verified

本次未发现监督数学、单 optimizer 或已审计 Actor/WM/Critic 参数继承的新缺陷；
该结论限于已读源码和既有真实 CPU 证据，不代表所有场景均已穷尽。
正式 C 大模型梯度训练、GPU、DEV、封存 TEST、收益和生产收敛仍为 NOT_RUN。
未检查或操作 host 训练进程，不能据此推断实际 GPU 运行状态。

## Next recommended step

先处理 R2 公共准入合同并回归，再处理 A V2 拒绝边界及 C 日志恢复；只运行
受影响的新增失败复现和相关恢复/管理合同。修复后再更新收尾状态，保持已冻结
训练源码独立，任何训练版本变更使用新的明确版本与准备快照。
