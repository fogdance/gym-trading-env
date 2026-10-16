# WP03 / WP02 接口审计

## Summary

基于实际 WP02 功能提交开发 WP03 A–D。新配对 worktrees 位于
`fg-s3-e-quality-wp03-20261016`，分支 `feat/s3-e-quality-wp03-v2-20261016`。
Gym base `e57189bad100465c9fee8d9bbb5601f3e5c376ad`；Dreamer base
`44a6cc62032811501631e83be977f07137dc65b8`。建树前原双仓均 clean；未从 integration 开始。

## Evidence observed

已读取 WP03 Final V2、WP02实施报告、gy18/d318接口Review、Master Roadmap V1及双仓AGENTS。
实际 `real_small_04` 在原WP02树存在：84个发布文件、239份生成时源码哈希核对一致。
上述完整性检查不替代本轮原生G3；最终实际运行结果由收据另记。

## Root cause

原 `verify_corpus` 将 `_code_lock()['files']`（绝对路径、整仓239份源码）与生成时字典等值比较。
`source_binding.files` 又进入L1/L2与账户来源身份；直接改写路径会使既有标签身份变化。
因此不能靠删哈希或重签旧标签迁移。新增Reader、搬worktree与教材语义改变也不应混为一谈。
另已确认：旧 `attach_quality` 不消费L1/L2 Envelope；Replay action是执行反馈，capture查询的是当前
post-step观测；decision_executed由下一回调补写；原pair_identity未排除全零owner；旧Q-C1排序项
还会消费strict标签，不能仅关闭Q value valid。

### 与任务书 action 描述的实测差异

进一步读取原 `embodied/core/driver.py::_step` 并查看真实P1行，确认原Replay的`action`是当前观测下
policy提议、将在下一次Env调用执行的动作；当前行的`incoming_action`、Reward与账户变化才是上次
动作的反馈。真实P1第0行 action=1/incoming_action=null，第1行 action=0/incoming_action=1。
因此不把teacher答案当行为action，也不整体平移标签；Join验证下一行incoming_action等于本行action，
并以下一行advanced证明本行decision_executed。任务书/旧审计中“本行action是上一反馈”的描述据源码纠正。

## Changes made

按用户确认，开发允许dirty/新增源码/换worktree。新监督记录以Corpus相对内容清单和合同建立身份，
保留历史生成证明，路径仅作定位。原Corpus与标签不修改；最终一次原生G3在匹配冻结producer上执行，
不在每批或普通源码编辑后重跑。开发诊断不是正式训练、Prepared或Checkpoint认证。
金融Env、Quality Reward、MACD、Oracle及公开训练/Resume Guard不在改动范围。

## Verification performed

每阶段准确命令、实际退出码、执行源码字节及用例见 `receipts/wp03/`。当前文档不声明尚未运行的项目通过。

## Risks / not verified

真实Corpus只覆盖FG201主力资格末两官方日。93个资金域外状态必须继续无效；教材覆盖与训练权限分开。
WP03不启动优化器、GPU或正式训练，不读TEST，不生产18合约，不用旧1M权重，不导出patch。

## Next recommended step

按A来源与分级、B投影/Join、C统一Reader、D真实CPU证据顺序验收，并更新Master Roadmap V2。
