# F3 合入 integration/s3-next — 2026-10-22

本次按用户明确要求合入主线；原 README 与哈希事件报告中的“不自动合并”描述为此前准备阶段的记录。

## 范围与提交

- Gym 主线基线：`ebdf4dc614bc5f5d626e1a310a21b3c89f38713c`。
- F1 哈希修复：`a4b3bb0a0546feaff0b418f3eb1d9d5a1a7cdd92`。
- F3 分支：`2712be1b24903fd443405944dd935cf5f9662120`，包含同一哈希修复的 cherry-pick `28c57c4e29473199570430f79e46340cfdf2e446`。
- Dreamer 无新增变更，保持 `990f61f8b7ac6bad25ff9e41b65624ae13ccf497`。

合入 FG809 F3 配置与正式启动脚本、流式 SHA256 和失配诊断。F3 继续使用独立冻结工作树启动；本次不修改其源码、启动锁或 Prepared，不修改 F1、教材及已有 Run。

## 核对与已有证据

合并两个来源分支无冲突。添加本记录及准备收据前，`git diff --exit-code run/f3-fg809-20261022 --` 退出 0，合并结果与 F3 分支完全一致；`git diff --cached --check` 退出 0。

`PREPARATION_RESULT.json` 是冻结 F3 工作树此前 CPU prepare 的原样归档，记录 `F3_DATA_READY`、退出码 0、正式 F1 第 10,000 步、Corpus ABI 3.1.0、复用发布 G3 和有效 NATURAL 窗口。该记录描述准备时状态，不代表后续训练状态。

本次合并不额外运行测试、G3、GPU 训练或冷恢复，相关新增验收为 NOT_RUN。原哈希失配具体根因仍未确认，详见 F1 事件报告；本次不扩大修复结论。
