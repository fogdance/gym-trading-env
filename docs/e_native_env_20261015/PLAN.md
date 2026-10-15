# E 原生 Env 集成计划

## 最新范围（用户后续选择）

后续只跑 **E + Tutor-CD Full**，正式1M也只准备这一组。复用已验收Full E2配方：alpha=.8、pair=.1、tau1、kappa1、cap2，每100更新一次原Actor梯度诊断。此前三模式CPU/静态CD证据保留，不再扩展A2/B2 GPU；静态CD(kappa0)的成绩不冒充Full(kappa1)。仅补所选组合的真实联合更新/恢复与size50m1K，原算法和通用组件不再重复测试。正式1M保留准入通过后Review再启动的门槛。

任务：Review 并应用 `FG_S3_E_Env_Native_Integration_Reviewed_20261015.zip`，优先验证原18维等价、Env原生E观察、缓存隔离及合约轮换。用户后续明确最终目标为原生 Env 融合、模型设计一致、smoke 与正式1M准备，因此完成数据合同后继续真实三种模式CPU、SourceAdmission、完整Checkpoint冷恢复、完整size50m GPU 1K及恢复验收。工作仅限 research 下 Model 双仓，历史训练工作树与产物保留；正式1M不启动。

## 基线与补丁 Review

- 实际 Gym `feat/s3-e-model`：`cb5f447893a87bd32b72b9e6bc3818ecc4bafad0`；Dreamer：`ecbdd77795a0abb5bcdbdcb734d078dbcd8947eb`，应用前均干净。
- ZIP SHA256 `04b892f78bb5ca719fae655a1ffe81b00151624c0845619cdd0622852f2fa51e`，包内5个文件校验通过。包内 README、Review 和测试记录已完整读取。
- Gym patch `git apply --check` 通过；Dreamer patch 的 SourceBinding hunk 不适用：本机已有更完整的所有 E Python 模块指纹和 schema 校验。保留本机实现。Dreamer 训练接线 hunk 单独检查通过，可以按原增量应用；不强制套用、不 reset。
- Data→Model merge 与15m因果修复已在实际历史中；目录迁移和 MarketStore 补丁未合入。
- 只读主机核查：在线模拟 PID 45614 仍运行，不停止或修改。应用前保存受保护工作树与已跟踪文件身份。

## 实施与最小验收

1. 分别应用 Gym 全量增量和 Dreamer 原生接线增量，保留已认证来源、现有准入生命周期与旧代码历史。审查 Env `_raw`→既有 feature builder→E 数学→原生 Observation 的实际调用链。
2. 在本机完整 Python 环境运行包内 Env/Cache/E-Data 测试，无 SKIP 才计通过；对新接口发现的实际错误补必要回归后修复。
3. 复跑旧 Single3 Env、Cache、Replay、D0/V2 受影响合同。验证非E字段、Reward、撮合、Action Mask 与账户行为不变。
4. 针对18个已签名 TRAIN 合约比较全部原18维特征及全部决策 E 数组，使用原 Env 已加载 `_raw` 与既有 E 审计实现逐元素比对；核验六通道、15m边界、延迟到达负例、前缀不变性、元数据截止时间。
5. 核验每次 Env 冷加载只解析一次 CSV、只 build 一次原18维；缓存命中不再 build；覆盖原生 E 的跨合约1024截断、账户连续性、冷恢复和 native Replay/Sidecar 对账，不运行模型训练。
6. 对新代码 schema 在隔离目录重新签发 TRAIN collection 并做元数据准入校验，保留旧 collection。CPU 模型/Checkpoint 只作必要已有合同回归，不启动 GPU。
7. 三种教学模式使用独立配置/产物执行真实TRAIN的CPU小模型联合更新与完整Checkpoint冷恢复，核实两分支/融合/Actor/B2头与Adapter参数更新、梯度及policy同步。原合成CPU成绩只作回归，不代替真实验收。
8. 代码冻结后逐模式执行完整size50m GPU 1000市场分钟/1000联合更新及独立进程恢复，保留所有失败、资源/吞吐和Checkpoint收据；不覆盖现有Run、不修改其他实验、不上传W&B。
9. 对应P0全部通过后才评审对应E组合waiting_e解除及正式配置；正式1M只准备、不启动。若任一P0未通过，明确NO-GO并列出阻断，不以prepare或smoke命令充当验收。
10. 自审双仓diff，记录原始命令、失败、修复、测试收据、真实数据来源、最终双仓SHA与未验证项到本docs目录，交用户Review；不自动合并integration/main。
