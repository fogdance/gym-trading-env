# FG 开发目录迁移（2026-10-10）

FG 开发已回到两个原仓库，均在 `feat/fg-entry-v1.1` 分支直接修改：

- 环境：`/home/v/Documents/work/gym-trading-env`，迁移前 FG 提交 `be62913`。
- Dreamer：`/home/v/Documents/work/dreamerv3`，迁移前 FG 提交 `f0080e5`。

`artifacts/fg_entry_v1/worktrees` 下两份旧工作树已移除。代码放在原仓库的源码、配置和工具目录；artifacts 仅保留数据、日志、检查点等产物。历史文档中的旧工作树路径不再作为开发入口。

## 原目录残留修改备份

切换分支前已保存包括未跟踪文件的 Git stash，没有覆盖或丢弃原修改：

- 环境旧分支 `feat/reward-profit-lock-atr`：`eda711b90e67f3e343bb707d051682456c5dbb69`。
- Dreamer 旧分支 `dev/action-mask-full-chain`：`c8acbcef6e74c6482da7f2c2746ef90473be6d87`。

旧 `1.yaml`、`fg.txt` 亦在环境备份中。运行时统一使用 `configs/fg_entry_v1/fg.yaml`，不依赖上述旧输入文件。

## 数据及历史运行

已有主力导出保留在 `artifacts/fg_entry_v1/recovered_main_export`，没有重新访问数据库导出。基于该导出重新生成的当前冻结数据集为 `artifacts/fg_entry_v1/dataset_v1`，其中配置和清单均指向原仓库。

旧 `recovered_dataset` 和 GPU smoke 产物保留用于历史审计；其中旧工作树路径及旧观察结构不代表当前运行配置，不能直接作为当前数据集或恢复检查点使用。

Dreamer smoke 脚本默认寻找同级 `gym-trading-env`；可用 `FG_ENV_ROOT` 显式指定环境仓库。脚本仍限定 GPU 和少量训练步数。

## 验证

- 原环境目录：`tests/entry_v1`、`tests/utils/test_market_features_match_spec.py`，36 项通过。
- 原 Dreamer 目录：`dreamerv3/tests/test_fg_entry.py`、`embodied/tests/test_driver_account_checkpoint.py`，6 项通过。
- 当前冻结训练清单通过 `validate_fg_entry` 完整性、路径和生命周期准入检查。
- smoke 脚本通过 `bash -n`；两仓库 diff 无空白错误。
- 本次没有启动正式训练或 smoke 训练；当前九维仓位 Obs 的 GPU 训练 smoke 尚未重跑。
