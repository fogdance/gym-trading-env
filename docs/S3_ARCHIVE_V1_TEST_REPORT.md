# S3_ARCHIVE_V1 实施与800K验收报告

日期：2026-10-14。独立分支：`feat/fg-s3-research-archive`；基线为现有W&B追踪器 `ad6490675bf853a7793763f95aa7ba7f12fa379e`。工具入口、操作手册、来源记录和规范均随此分支提交。提交SHA由 `git rev-parse HEAD` 查看。

## 交付结果

- 五个实际CLI：inspect、archive、verify、publish、list；archive默认干跑，模型验证和发布分别显式执行。
- 标准归档：`/data/logdir/fg_single3/research_archives/s3-tutor-cd-800000-735fe1b4e482`。
- schema：`S3_ARCHIVE_V1`；requested/achieved均为 `FULL`（L3）；state=`VERIFIED_FULL`；缺失依赖为空。
- 身份key：`735fe1b4e482d9ccf88583916032a5bd0ff9a7b31f2043bb7de52075f290e150`。
- 原一次性800K归档及源码冻结目录保持原样；新的标准归档不绑定原Run文件，不依赖共享Git对象库。

## 双源码和配置身份

| 项目 | 已核实身份 |
|---|---|
| Gym | `3a6f889adf65583a1ab636f244ca05218f19196f` |
| Dreamer | `c18a8b5917f73110f9b6ce840495f4da1d7ca1d4` |
| 独立源码树快照 | `f92440519975e4ca08ed6fad6cbac8d04376a625098b6658c55ed97dc46f3e60` |
| 解析后配置字节SHA | `d35d92f8a5693e75d8b3b452729acb64bd8e024cbc12e6695c4c4f0b3b39dcc0` |
| Checkpoint manifest字节SHA | `5cf093b56ca3bb340cca9520b5579105b1cecc9f044262d5a23d17c545c6a474` |
| 原评价receipt字节SHA | `b1665c584aedbd7629494a0fd5ad68fe6e52d1649a03de602687d583530e05f9` |

193个执行代码文件与冻结身份一致；准备时两仓库dirty为空。源码采用历史提交完整导出，当前HEAD/工作区只作观察；后来的差异不冒充运行时补丁。独立源码和所有归档内容纳入SHA256清单。

## CPU测试与故障验证

`s3_archive.tests.test_archive`：23项通过；`research_tracking.tests.test_tracker`：16项通过，共39项。命令：

```bash
/home/v/Documents/work/research/fg-s3-wandb-env/bin/python -B -m unittest s3_archive.tests.test_archive -v
/home/v/Documents/work/research/fg-s3-wandb-env/bin/python -B -m unittest research_tracking.tests.test_tracker -v
```

覆盖：done缺失、Checkpoint/receipt哈希不符、错误配置、历史HEAD与当前dirty差异、真正dirty源码、嵌套收据相对路径、笔数与账户不符、哈希篡改、空间不足、复制中变化、日志前缀追加、不稳定恢复依赖、源硬链接/输出重叠、重复归档、原子升级、复制/源码导出/元数据封存中断重试、未验证不得升级、离线待发布、隐私过滤、标签幂等及云端展开字段。

首次普通沙箱中的两项离线SDK测试出现 `ServiceStartTimeoutError`，在正常主机用户环境重跑16项全部通过，没有修改SDK或增加训练环境依赖。

语法解析16个Python文件通过；改动的diff格式检查通过。用户规范副本保留原字节（SHA `d84ed87be768fcd45dd94309c76c05ddff112af5da0c92c0725fc1d54a4c5906`），其中三处Markdown双空格换行不作格式化改写。

## 800K实际隔离验证

先封存L1及完整Checkpoint，再通过原子升级保存6386个来源对象；归档最终包含7542个校验文件。所需载荷计划约5.1GiB，复制前实际可用约144GiB，预留10GiB。

### L2：独立Actor-only评价

GPU `cuda:0`；FG409 DEV、5000实际环境推进、eval_argmax、beta=0。原冻结源码在Bubblewrap私有命名空间执行，Checkpoint指针在临时副本中指向800K。Oracle数据读取尝试为0。

| 指标 | 原结果 | 独立结果 |
|---|---:|---:|
| ADD / 开仓 | 7 | 7 |
| TP5 / SL30 | 5 / 2 | 5 / 2 |
| TP市场分钟中位数 | 74 | 74 |
| Reward | −3.805870048525721 | −3.805870048525721 |
| 扣费净盈亏 | −2226 | −2226 |
| 合法决策 | 4033 | 4033 |

全部比较项精确一致，未用到浮点容差。重评receipt字节SHA：`28b2778f7b0cfdf4919207359df332b138cbd43f4b3e498957e4151e6ebbfbcf`。完整结果在归档 `reports/eval_report.json`。

### L3：完整恢复与一次更新

`reports/restore_report.json` 状态为 `ISOLATED_RESUME_SMOKE_PASS`。模型/优化器、Driver、Collector、Gate、计数器、Runtime、Replay元数据及全部数组检查通过。

- 恢复前：环境步800000、模型/本地更新32000、周期8000。
- 一次受控更新后：环境步800000、模型/本地更新32001；优化器更新有限、损失为有限值。
- 新环境推进0，新优化器更新1；只写临时Run，未调用正式run_to或保存新的正式Checkpoint。
- 执行worker SHA：`693d3df9b93106b6328acd3a2a0d27473dbd93884b0ea68ed6f5139cb5db0508`。
- 恢复日志SHA：`744f0e6a126ab04ea06c6bf0569200e11dcdf755bf90fba6aa4ec3cfd11e0744`。

实际执行脚本、隔离命令、日志、spec/result和独立评价文件已保存在归档 `verification/executions/`，并纳入总SHA256清单；不要求保留临时目录才能查看证明。

原Run及两训练工作树共1712个文件的路径、大小、mtime和inode前后完全一致（998+246+468），没有新增评价目录到原Run。原正式Run仍为已完成1M，未中止、重启或改写。

本次环境Python3.11.11；JAX/JAXLIB0.4.33、NumPy1.26.4及其余冻结核心依赖由原恢复入口检查匹配。后续工具会独立记录验证环境并在启动前拒绝包版本不匹配。

## W&B及幂等发布

Project为私有 `fg-single3-research`。真实关联：

- 训练Run：https://wandb.ai/xkm104501-none/fg-single3-research/runs/3dc57d410d9b50601120
- 原800K评价Run：https://wandb.ai/xkm104501-none/fg-single3-research/runs/9832f27f1b038b84fb58

幂等key：`987e3b38f08a85398dac97baa2213e7ac0f1e59225669b2de8ef8b81183be143`。目标标签为首次观察自主ADD、checkpoint:800k、eval:actor-only、clock:S3_MARKET_MINUTE及archive:S3_ARCHIVE_V1；原标签全部保留。重复发布云端回读通过，两Run关联L3归档身份；new_run_created=false、raw_files_uploaded=false、parent_link_pending=false。

实际W&B将嵌套摘要展开为带点号的叶字段，工具按实际字段逐项回读；早期失败保留PENDING，修复后达到PUBLISHED，不以HTTP成功代替验证。

## 校验与限制

- `SHA256SUMS` SHA：`33f4f60ca7d507c11174e59f6df911070a2afc8fc6a8be50880cbcb6ab9aa3a4`。
- `archive_manifest.json` SHA：`b2a1cc3b32cbfe0073f2c789ac7cdb52d813c3061f5607957946caf1efcbcb5a`。
- 这些身份对应本次最终封存版本；后续追加验证/发布元数据会生成新清单，旧来源字节保持不变。
- L3证明本环境完整恢复及一次受控更新。没有跨硬件逐位确定性或完整后续轨迹对照，`exact_replay_verified=null`。
- Submodule/LFS及未知训练协议明确报告需要适配，不猜源码或恢复能力。GPU验证依赖Linux/Bubblewrap和匹配运行环境。
- 7笔DEV交易是首次观察自主ADD的证据，不构成稳定盈利证明。Tutor-CD Full/B2 Full开发未参与本分支。
