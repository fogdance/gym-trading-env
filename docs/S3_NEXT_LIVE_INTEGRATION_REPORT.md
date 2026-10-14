# S3 Next：Tutor 800K LIVE 集成

日期：2026-10-14。按用户要求，将 `feat/tutor-cd-online-sim` 合入公共研发基线，暂停本次夜盘监控。

## 合并结果

- Gym 合并提交：`4a78feb7f0187752742a512de6d39f47b6ab9f3f`；来源 `9eba965040b6660e6d4655712624353f90990999`，目标原 HEAD `ee03bdd76cdae7d5d46c2fd51286172412d45d68`。
- 合并无冲突，保留已集成的 Archive V1 和账户轮换性能优化。
- Dreamer 来源与目标均为 `d41bc80ba13b3dc3f5d2c32fcdc800613b2d0b4d`，没有需要合并的修改。
- 合并了 6 个文件：冻结配置、统一 CLI 入口、LIVE 适配器、共用 DB 连接、回归测试和配置说明。

复用原 Driver、ActorPolicy、固定方向 Entry Env 与 `continue_data`。读取已完成的 FG701 分钟 K 线，加载固定 Tutor 800K checkpoint，以 eval argmax 决策；三手开平仓记录映射到现有 RPC。过期信号转 WAIT，缺失分钟停止推进，RPC 失败保留账户和 outbox。源码及使用说明见 `configs/experiments/fg_s3_v2/README.md`。

## 本轮验证

| 验证 | 结果 |
| --- | --- |
| LIVE 续接、短仓开平、Driver 恢复、过期信号、RPC 失败 outbox、规则身份和分钟缺失 | 6 PASS，6.193 秒 |
| 行情缓存、市场分钟、合约池、Single3 账户、RPC payload 合同 | 46 PASS，33.426 秒 |
| 统一 CLI help，`simulate-online` / `--check-only` / `--resume-online` | PASS |
| 合并差异空白检查 | PASS |

两组测试均在合并后的集成 Gym 与其兄弟 Dreamer 上执行，关闭 GPU，使用受控行情、测试 Actor/RPC；本轮没有启动 LIVE、发送真实 RPC 或训练。后续文档提交不改变被测源码。

完整命令、环境、退出码、JUnit、stdout 和源码哈希存放在公共工作区 `validation/live_integration_20261014_2114/`。此前 138 项验收保留在旧记录中，没有计入本轮 52 PASS。

## 版本锁与范围

最终双仓 SHA、clean 状态和本轮验收更新到外置 `baseline-lock.json`；上一份锁的原始字节保存在本轮 `baseline-before.json`。在线来源工作树保持原提交且 clean。当前运行程序的源码目录未切换，本轮合并不对其实施重启或迁移。

RPC 回执代表接收服务/队列确认，券商成交与模拟成交仍需独立对账。本轮没有对合并源码执行真实 GPU 推理、实际 RPC 发送或券商成交验证。后续开发从更新后的集成版本锁建立工作树。
