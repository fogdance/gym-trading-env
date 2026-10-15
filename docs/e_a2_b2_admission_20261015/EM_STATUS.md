# EM01–EM18：截至 2026-10-16 的增量状态

旧 `docs/e_model_v1/EM_STATUS.md` 是初始合成验收历史，原样保留；本表说明本轮真实 E+A2/B2 验收补充，不能把沿用项写成此次重跑。

| ID | 当前状态 | 本轮证据或边界 |
| --- | --- | --- |
| EM01 | PASS | 实际 integration 基线、隔离冻结双仓、旧 Run/HEAD 保护 |
| EM02 | PASS_CURRENT | 36 项公开合同含八组合；E+A2/B2 ready，Tutor+B2 仍拒绝 |
| EM03 | 历史合成 PASS；真实路径覆盖 | Encoder 实现不变，当前 CPU/GPU 使用原模块 |
| EM04 | 历史合成 PASS，未单独重跑 | context→tokens/posterior 既有测试；本轮有真实 context 梯度/更新 |
| EM05 | 历史合成 PASS，未单独重跑 | shape/dtype/invalid/future 拒绝实现不变 |
| EM06 | 历史合成 PASS；真实路径覆盖 | policy/WM/prefix 在当前联合更新及 Replay/cold restore 中执行 |
| EM07 | PASS_CURRENT_REAL | CPU/GPU 三分支有限非零梯度与改变参数；教学 AST 合同通过 |
| EM08 | PASS_CURRENT_REAL | 原优化器、policy 同步及 Checkpoint 参数指纹 |
| EM09 | 历史合成 PASS；当前来源合同确认 | Decoder 仅 shared1m/position，未改变 decoder 实现或重跑专门扰动测试 |
| EM10 | PASS_CURRENT_REAL | 原 Env 每分钟完整 E Observation、跨两合约、原 market_seq 精确一致、Replay 恢复 |
| EM11 | PASS_CURRENT_REAL | E+A2 CPU2200/6、GPU1000/1000、Actor 参数改变 |
| EM12 | 既有 E+Tutor 结果沿用 | 不重新训练 Tutor；已完成原 1M 及冻结代码保持不动 |
| EM13 | PASS_CURRENT_REAL | E+B2 CPU/GPU，head/adapter 初始化、更新和混合实际生效 |
| EM14 | PASS_CURRENT_FORMAL | 两模式公开 direct_init 来源、错配拒绝、正式 prepare；合成正式来源仍拒绝 |
| EM15 | PASS_CURRENT_REAL | 两模式 CPU/GPU 原完整 Checkpoint 冷进程下一步 exact；不称正式1M恢复已运行 |
| EM16 | PASS_CURRENT_TARGETED | 本次影响的 Composition+SourceAdmission 36 PASS；未重跑历史整套或旧无关失败 |
| EM17 | PASS_WITH_SCOPE | 18合约/488160原始行 collection verify；72数组数值身份相同；原Env每分钟FG201/FG301运行与正式18合约pool准备；非18合约全时点逐帧遍历 |
| EM18 | GPU_1K_PASS；FORMAL_1M_NOT_RUN | 两模式原size50m GPU1K及冷恢复通过；DEV Actor-only/W&B/正式1M本轮未运行 |

真实命令、所有已知失败、身份、Checkpoint 路径见 [REPORT.md](REPORT.md)、[COMMANDS.md](COMMANDS.md)。工程训练准入 GO 不等于策略效果或 RL 启动通过。
