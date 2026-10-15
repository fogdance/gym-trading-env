# E Review → integration/s3-next 合并报告

## Summary

将 `fg-s3-e-review-fix-20261015` 两仓的 `fix/s3-e-market-identity-20261015` 合入各自 `integration/s3-next`，使用双父普通 merge，保留 E-Data→E-Model→Review 的真实提交历史。

## Evidence observed

| 仓库 | 目标原 HEAD | 来源 HEAD |
| --- | --- | --- |
| Gym | c3a808437ef277ee87f25b0ae8803e31fa5dca74 | 1fa4cff970f700536b72041cad27972dacc76958 |
| Dreamer | eb63de710fd693fad7ac76f797f3d9f39e331b98 | 730939defb1bd77b2ceca04212105ec2727c95f9 |

Dreamer 合并提交：`131e7ccd4ef0648c8230e088d95e2c6da991557c`，文件树与来源完全一致。Gym 合并提交为收录本报告的 merge commit，双父依次为上表 Gym 目标原 HEAD、来源 HEAD；可用 `git log -1 --format='%H %P' -- docs/e_review_merge_s3_next_20261015/REPORT.md` 核对完整 SHA。

合并前四个工作树干净。只读进程检查发现目标 Gym 的在线监控 PID 748343；其模块在本次合并中未改变，进程保持运行。在线模拟 PID 45614 使用另一套冻结工作树，未修改或停止。

## Root cause

本次为分支集成。Gym 唯一文本冲突在 `tools/fg_s3_v2_experiment.py` 的命令 choices：目标增加 `monitor-online`，来源增加 `admit-e-prepare/start/resume`。配置 README 的独立段落自动合并。

## Changes made

保留两侧命令并集，保留监控参数、提前返回分支、E 准入和冷恢复参数。未另外修改模型或算法。来源非重叠的 271 个文件、目标非重叠的 3 个文件均经 Git blob 对照原样保留；无未解决冲突。源码、修复报告及 Actor-only 原始证据一并进入目标分支。

## Verification performed

- 最终 9 个唯一用例通过：7 个 E 行情 SHA256 / 同一字节解析 / 注入拒绝 / 共享缓存恢复用例，1 个严格来源认证用例，1 个共享监控入口用例。
- 首轮为 2 PASS / 7 ERROR：测试临时输出位于 `fg-s3-next-integration` 内，触发 `e_data_v1/io.py:47` 的既有隔离守卫。只将临时目录改至 `research/diagnostics/`，仅重跑这 7 项，结果 7 PASS；未放宽守卫。首轮及重跑日志/XML 均保留。
- 4 个 CLI 检查通过：帮助列出两侧命令；E+Tutor 正式配置为 ready；E+A2、E+B2 仍 waiting_e。Tutor-CD+B2 的 `require_ready` 仍返回 `S3_TUTOR_B2_NOT_IMPLEMENTED`。合并入口 Python 语法解析通过。
- 来源及冻结训练两组 Gym / Dreamer 的 HEAD 和干净状态复核通过；Dreamer merge tree 与来源 tree 相同。
- 默认 `git diff --cached --check` 指出原始 `docs/e_data_v1/TASK_SPEC_20261014.md` 三行 Markdown 双空格换行；该文档 blob 与来源一致，保留原件。所有非 docs 文件的普通 diff 检查通过。
- 预检查 `git merge-tree --write-tree` 返回 128（`fatal: unknown rev --write-tree`）；改用普通 `--no-commit` merge，审查并解决唯一冲突。

命令见 [COMMANDS.md](COMMANDS.md)；接口输出、身份对照与测试原文见 [evidence/](evidence/)。

## Risks / not verified

此次只验收分支合并接口，没有重新运行训练、GPU smoke、Checkpoint 恢复或 Actor-only 评价，不改变历史报告的验收结论。没有修改冻结训练 worktree、Run、Checkpoint、W&B 或 main。合并不重写既有 prepared/Run 的身份；既有 Run 继续引用其原冻结代码。

## Next recommended step

在 `integration/s3-next` 核对合并结果与已有 E 评价报告；后续实验另行授权。
