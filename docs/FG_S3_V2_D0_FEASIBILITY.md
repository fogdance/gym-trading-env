# FG S3-V2 DAgger D0 技术预研与原型验收

日期：2026-10-13。结论：**GO_WITH_LIMITS**，仅允许继续设计下一阶段实施方案。
当前交付是可信查询、真实交互与接口证明，不是完成A2/B2/C2或正式1M训练。
用户选择：资金等条件超出证明时继续交互、明确跳过监督，不实现任意亏损账户的精确回退。

## 1. 实际实现与来源

开发目录仍为 `research/fg-s3-dev/{gym-trading-env,dreamerv3}`；独立分支
`feat/fg-s3-v2-d0-dagger`，从Gym `559d01a`、Dreamer `fe1bca9`创建。
冻结的两套训练worktree及原仓库不修改。仅原Oracle增加显式可选后缀输出；
旧默认返回、Oracle数学、Env/Reward、网络及V1 S3算法保持原样。

复用用户授权的 `research/fg-data` 数据软链接下的原E/O TRAIN输入及候选缓存。
核对父receipt、输入/策略文件、市场及候选hash，并将原Env producer字节hash与
当前开发代码对比。旧外部实验目录不作为代码来源或Python导入路径，不写入。
Python实测：Dreamer、Embodied、Gym都来自上述开发兄弟目录。

首个真实合约：CZCE.FG201，C1 Reward，31260行情行，31259候选，全部SHORT。
Oracle保留完整合约/Gap边界、gamma=332/333、精确比较与WAIT平局规则。
新查询的动作来自 `first[t]==t`，不是复制原P4起始入场表。

当前认证域：真实ACTIVE空仓、规则/市场一致、真实mask双动作合法、账户资金
不少于原证书300000。证书的报价资金上界104358并非自动扩大认证域的许可。
持仓强制WAIT不学习；低于300000的亏损账户不贴旧P4标签；Q字段均为空。
未来行情、教师结果和证明身份只留在TRAIN教师/sidecar，不进入Actor观察或carry。

## 2. 真实运行结果与限制

以下为实际1000次cursor推进；另有初始reset，因此原生记录数为1001。
所有运行均为CPU，固定seed=202。scripted是因果模拟器，不是已训练Actor。

| 模式 | 环境推进 | ADD提议 | Oracle ADD | 执行ADD/成交 | TP | SL | 认证/合法决策 | 未证明资金决策 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WAIT | 1000 | 0 | 125 | 0/0 | 0 | 0 | 1000/1000 | 0 |
| scripted ADD概率0.5 | 1000 | 51 | 7 | 51/51 | 43 | 7 | 43/107 | 64 |
| mixed Oracle概率0.5 | 1000 | 77 | 22 | 51/51 | 44 | 6 | 141/160 | 19 |

mixed全推进标签比例14.1%，合法入场决策认证率88.125%；840个持仓/强制WAIT
明确排除。scripted认证率约40.19%。三个真实片段未发生拒单；期限/成交量拒单、
非法ADD、Gap和平局在使用原Env的受控CPU行情夹具中测试，**不标记为真实行情事件覆盖**。
最后未平仓持仓不伪造退出，不依据这些TRAIN指标宣布Actor交易成功。

原始固定种子重复1000步、400保存后恢复600步：全部语义记录、执行动作、Oracle
标签、最终状态hash相等，恢复保留全部原生UUID/offset。标签完整性校验新增后，
受影响接口回归及新版本mixed保存/恢复再次验证；详见收据索引。

真实Oracle对照10项：空仓ADD cursor8/81/98，WAIT cursor0/2/3，实际TP后资金
300282、300564的账户查询，实际SL后资金298182的未认证查询，以及31259合约终点。
ADD三个cursor不在原P4起始ADD表中。独立原Env逐区间核对成交、反馈及出口；
完整合约动作最优性不声称经过穷举，小型LONG/SHORT原Env穷举另有CPU测试。

原始一次mixed预处理总耗时20.9849s，后缀DP部分0.1528s；1001次查询平均
17.204ms、p95=28.168ms。查询计时包含真实状态认证，但不包含调用前观察/状态构建。
完整状态和prefix哈希随历史增长，**不能用这个短片段外推1M生产吞吐**。

## 3. 实际模型与数据接口证明

经共享SourceAdmission准入原冻结Online180k：来源环境步180000、模型更新21183；
目标threshold=0.30，原Single3 20报告Gate协议保留，Gate CLOSED。
CPU真实模型42,584,132个网络参数，B1×65，无新网络层。

- 100步真实Agent.policy + Oracle beta=1混合：Actor提议73次ADD，实际执行14次，
  成交14次TP。混合行为不能证明原Actor会交易；其能力不是交互启动前提。
- 100步推理未改变训练参数；随后用原Consec/Agent.stream进行一次原联合更新，
  计数21183→21184，Encoder/RSSM及Reward参数变化，Actor/Critic变化为0。
  Actor/Critic梯度均0，WM梯度383.31885，Reward梯度2.18395；原来源文件hash不变。
- 原Replay/Consec/Prefetch产生65行序列；同一个2×65批次容纳新DAgger及历史P4，
  有效段64，二者有效标签分别57/64。P4继续由原TeacherCatalog/adapter认证。
  物理Replay不含教师字段，训练后连接才产生teacher_action/teacher_valid。
- D0集合checkpoint有Env/Driver/Replay/混合RNG/sidecar待完成记录，明确无模型。
  不冒充完整V2 FullCheckpoint或可迁移Dreamer节点；完整注册要求见接口图。

## 4. 自动化结果、失败与复现

收据目录：`artifacts/fg_s3_v2_d0/receipts/`，真实输出目录：
`artifacts/fg_s3_v2_d0/runs/`。`receipt_index.json`保存各文件hash、范围、命令和版本。

| 项目 | 结果及范围 |
| --- | --- |
| 新D0及原Oracle CPU回归 | PASS：11项，`targeted_cpu_final.xml`；原Env受控行情，不能当生产行情验收 |
| 目标与Oracle证明完整性回归 | PASS：`proof_integrity_cpu.xml`，2项受影响接口复测，非额外独立覆盖数 |
| 真实TRAIN多状态Oracle | PASS：`final_oracle_verify/receipt.json`，10项原Env对照 |
| 三模式1000实际推进 | PASS：各Run receipt，原生字段/标签连接验证 |
| 固定种子/400+600恢复 | PASS：`real_repeat_resume.json`及修复后验收收据 |
| 原FullCheckpoint准入、真实policy、WM一次更新 | PASS：`final_model_smoke_03/receipt.json`；不是V2学习目标实现 |
| P4与DAgger原生批次/Prefetch | PASS：模型Run下的`bridge_receipt.json` |
| 真实行情Gap/平局/拒单事件 | NOT_RUN：选定片段无该事件；受控原Env测试PASS单独记录 |
| 任意亏损账户Oracle、可靠Q、边交互边学习、模型级V2精确恢复 | NOT_RUN / 未实现，已明确接口限制 |
| 正式1M、GPU、DEV/TEST、生产Gate连续20窗 | NOT_RUN |

所有开发失败保留，未覆盖：测试夹具最初引用不存在的`env.config_path`、连续时间
跨期货休市导致时钟拒绝；修正为显式配置路径和合法下午时段。真实模型第1次探针
错引用参数摘要模块，第2次将原生batch直接传入Agent.train观察到缺seed。
第3次通过原Agent.stream/Consec修正后PASS。父Oracle receipt的contract摘要无自身
content_hash，现与完整已签名子receipt正文比较。上述不是未修复失败，也不隐去原日志。

```bash
cd /home/v/Documents/work/research/fg-s3-dev/gym-trading-env
env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fg_s3_d0_mpl \
  PYTHONPATH=src:../dreamerv3 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -m pytest \
  tests/entry_v1/test_fg_s3_dagger_d0.py tests/entry_v1/test_fg_single3_oracle.py -q

env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fg_s3_d0_mpl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python tools/fg_s3_dagger_d0.py collect \
  --config configs/experiments/fg_s3_v2/D0_FG201.yaml --mode mixed --env-steps 1000 --save-at 400
```

同入口支持 `--mode wait/scripted`、`verify_oracle`、`model_smoke`；恢复必须给新输出
目录和`--resume-collection <旧Run>/collection_checkpoints/env_000400`。实际完整命令见索引。
工具拒绝超过1000环境步、非TRAIN和覆盖已有输出，强制CPU。

## 5. GO_WITH_LIMITS 的含义与最终自审

可以继续编写A2优先的完整实施方案：可信后缀查询、原Env混合控制、正确执行动作
carry、真实Replay/受保护监督及WM继续学习均已有代码和真实证据。
仍不能直接启动1M：认证域保守且会丢失亏损状态标签；只验证单个SHORT合约；Q未实现；
活跃Replay和延迟sidecar发布存在确定的并发接线问题；WM-only/BC无Gate/C2阶段控制
及模型级完整恢复尚需新协议。细节见 `FG_S3_V2_INTERFACE_MAP.md`、`FG_S3_V2_D0_NEXT.md`。

自审新增目标/证明认证，拒绝只改teacher_action或Oracle记录而保留旧证明。保存恢复
绑定运行/代码/Oracle身份，不能用新版本覆盖旧receipt绕过检查。旧P4标签和Catalog
未改、旧A1000和两套冻结训练目录未改、原Source未改，无GPU训练或DEV/TEST。
本任务到D0为止，等待下一阶段任务书。

## 6. 提交与产物位置

实现版本（后续报告提交不改变源码）：

- Gym：`96e5115fc7f7e1d480898b8a1d9d52ca240e5bec`。
- Dreamer：`f92c096e5c9a81758cfde103b20e680e8f4be09c`。
- 收据索引：`artifacts/fg_s3_v2_d0/receipts/receipt_index.json`。
- 修复后mixed运行：`artifacts/fg_s3_v2_d0/runs/accepted_mixed_1000`。
- 当前版本重复/恢复：`accepted_mixed_repeat_1000`、`accepted_mixed_resume_1000`；
  `accepted_repeat_resume.json`确认全部语义记录相等与原生ID精确保留。
- 已完成A1000/来源节点保留检查：`protected_artifacts.json`。

原Gym HEAD d374e93、原Dreamer bb1d123均tracked clean；
旧冻结训练5293e97/93a6fb6及T030冻结9752438/00b82f8均保持原HEAD、tracked clean。
当前开发分支只增加D0代码/文档/收据。没有创建新的正式训练worktree，
没有执行其他实验的代码，没有删除、改写或覆盖旧模型/训练工件。

原始失败log/XML按字节无损gzip归档为`.raw.gz`并提交；未压缩原件也保留在本地。
压缩/原件hash和解压字节相等证明写入收据索引，不清理或改写失败内容。
