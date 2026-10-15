# Quality-first 合入 integration/s3-next：复核与验收（2026-10-16）

本轮将已评审 Quality-first 分支普通 merge 入 Gym／Dreamer 的 integration/s3-next。[实际收据与逐项测试索引](receipts/s3_next_quality_merge_20261016/README.md)保存全部原始证据。合并前两仓干净；没有删除、重置、变基或覆盖历史分支、训练目录。最终双仓 SHA 以工作区 baseline-lock.json 为准，锁的旧版本保存在 validation/quality-first-merge-20261016/baseline-lock.before.json。

## 起点、来源与冲突

| 仓库 | 合并前 integration | 已验收来源 | 普通 merge 提交 |
| --- | --- | --- | --- |
| Gym | d05b50c990fe472be711a93531f4d9ce4b1fe186 | 9d2bf64f02db742d631a5556f5547c1378cec5cc | c31ed89c3b3100fc46e37459dd909b692e1b417f |
| Dreamer | 13f03938686a34b8f641c698da3198fb2d74b6cd | d33985cb2989dcf2e3fe6dc06584ffb757011c3d | 6403cd68fdcc4e61f9036a7d06f30ac7a4448312 |

真实文本冲突共五个文件：Gym 的配置 README、共享设置／prepare 模块、共享 CLI；Dreamer 的 SourceAdmission 和 LiveRun。逐一保留两侧扩展：E 的 e_binding、内部准入核心和执行钩子，Quality 的独立 prepare／Catalog、CPU diagnostic、四头训练、不可变准入 Policy 与 FullCheckpoint 计数。CLI 同时保留 admit-e-* 和质量评价／补评命令，按对应配置在模型导入前选择 CPU／GPU。

## 复现发现的集成问题

1. 自动合并后的 Reader 清单仍认证旧 Env 哈希，现有严格 Reader 用例真实失败。原验证函数保持不变，仅更新当前已复核的 E＋Quality 精确目标束，绑定 Env、Reward 和 candidate 源码；未知生产者／改动仍拒绝，没有新增兼容回退。
2. E 的 ready 规则与新增 Quality teaching 组合后，把未开发 E＋Quality 标成 ready。新回归复现后恢复 waiting_quality_e，公开准备在产生工件前拒绝。原 E＋A2／Tutor／B2 和 Legacy＋Quality 各自保留状态；Tutor＋B2 继续拒绝。
3. 原 E 负例把已发布 E 的 support_state 再赋为 ready，实际没有修改输入。对合并前 13f0393 的纯解析器运行结果证明该赋值无变化；测试改为实际非法的 waiting_e，保留生产 ready。
4. 首轮把 --basetemp 放在受保护的 fg-s3-next-integration 内，E 数据 bundle 工具按规则拒绝，产生 5 failure＋7 setup error。孤立短路径通过、集成树内长路径失败；源码明确按目录组件保护整棵集成树。保留保护规则，将测试输出移到 /tmp 独立目录复验。
5. 原 EM16 要求新增模式后整个模型类仍与旧类相同，覆盖了 dispatch／optimizer 路由和新增方法。更新为八个原 A2／Tutor／B2 教学及共享 Loss 函数的严格 AST 对比；新增 Quality 路由使用实际前向、梯度、参数和恢复验证，未为旧断言加入生产兼容分支。

## 两侧父提交逐文件 Review

- E 的 _inspect_source 核心 AST 与合并前目标逐字一致；public 入口保留 require_ready、E binding 和 Quality 独立分派。
- Reader 验证函数 AST 与合并前一致，仍要求精确生产者和完整当前目标束；原生 Env 全部方法除显式 Quality Reward 分派的 step 外保持目标 AST。
- 九个 Quality 核心源码文件与已验收来源逐字相同，包含 Teacher、Score、Catalog、双 Q、预测／决策准入及网络数学；未改 B(t,d)、有效监督规则或梯度边界。
- LiveRun 保留 self.composition／records／create_agent／transfer_agent／agent_class 等 E 钩子，Quality CPU 只允许 diagnostic；E acceptance 与生产 prepared 协议互相隔离。
- 新预测准入仍逐个检查 7 个事件、8 个 Touch、4 个 Adverse 窗口；收据 TRAIN 从绑定 Policy 读取；风险仍按完整交易的入场 Fold 归属，单列跨 Fold 完整结果、未观察截止和右删失。
- 231 个运行源码文件逐字节核对当前 Git，Quality 生命周期与真实来源 prepared 的 code 映射完全一致。旧正式 Tutor Full 1M 来源 manifest 哈希仍为 d6b03a8f2e3604885a7550c9bd217471e290c7840f91dd1ecc93bf93658e9098。

## 实际验证与计数

最终按 testcase 身份去重并选择修复后的最后一次实测，共 235 项通过，无未解决失败／跳过。50 项定向用例不重复累计。

| 验证 | 实测结果 | 原始收据 |
| --- | --- | --- |
| 集成问题修复前 | Reader 与未准入组合 2 个目标失败 | before_fix.xml |
| 定向复验 | 50／50 | targeted_v2.xml |
| Native／Quality／E 数据及现有合同 | 198／198，正确独立临时目录 | native_v2.xml |
| 真实 TRAIN E／Legacy 财务、原 Oracle、合约轮换与冷 Replay | 1／1，首轮已通过，161.419 秒，未重复执行 | native.xml 中对应 testcase |
| Quality／E CPU 网络、来源继承、E FullCheckpoint／Encoder | 35 个不同用例全部通过；旧 EM16 断言失败由后续六项 E source 复验覆盖 | models.xml、e_source_v2.xml |
| Quality 共享 CLI 准备、100→resume200、连续200 对照 | 1／1，92.831 秒 | lifecycle.xml |

不是一次 235 项全绿的单条命令；final_test_index.json 为每项记录最终成功收据，失败命令及原始日志全部保留。原生首轮 199 项中 12 项输出隔离失败；修正命令后重跑其余 198 项，首轮真实 E 回放的独立通过结果继续计入，避免重复耗时运行。

Quality CPU Run 为 QUALITY_CPU_ACCEPTANCE_6cc80b0fae，200 Env／4 更新，四预测头有实际梯度和参数变化，原 Actor／Critic／Backbone／双 Q 冻结；恢复与连续运行参数摘要及全部更新指标相等。该路径使用合成原生监督，不代表研究效果。

真实正式 Tutor Full 1M Checkpoint 的 CPU 来源准入和 FG201 Quality prepare 已重新执行；目标为 S3_NEXT_QUALITY_SOURCE_PREPARE_v1_20261016，仅准备，没有构造或训练 GPU 网络。

E 公共 CLI 新 Run `S3_NEXT_E_A2_CPU_MERGE_REVIEW_20261016_v1` 已通过 `PASS_CPU_REAL_TRAIN`：真实 FG201／FG301，沿用既有小 CPU 验收网络与 canonical Single3 Gate，prepare→start1100→独立子进程下一次更新比较→cold resume2200，2,200 Env／6 更新，`cold_resume_exact=true`。三个 E 编码分支均有有限非零梯度，Encoder／Actor 参数实际改变，推理策略与训练参数同步；beta=1 的交易属于教师控制工程验收，不是 Actor-only 策略成绩。完整原始命令及输出为 `e_cpu_commands.json`、`e_prepare.log`、`e_start1100.log`、`e_resume2200.log`。

## 保护、限制和后续

Git 复核仅 integration/s3-next 两个 refs 改变，原 main、全部其他分支／Tag 与来源功能分支保持原 SHA；两侧旧目标和已验收来源均为最终祖先。旧锁落后于实际合并前分支，已原字节保存，新锁绑定本次真实版本与新验收，不沿用旧 PASS 充当当前结果。

本次只作 CPU 工程验收。未进行 GPU、新 Quality 10K／1M、W&B 云端写入或策略收益放行；Quality 预测／Q／决策门禁继续 PENDING，Quality Full 不宣告 GO。E＋Quality 未开发。新正式研究需独立冻结新双仓版本并重新 prepare，历史 prepared／Run／Checkpoint 和既有研究收据保留。
