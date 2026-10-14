# S3 Research Archive V1 — 研究里程碑标准归档规范与 Codex 实施任务

**版本**：`S3_ARCHIVE_V1` · **日期**：2026-10-14 · **适用**：FG Single3 / DreamerV3 的 A2、Tutor-CD、B2、E 与后续实验。  
**W&B 项目**：`fg-single3-research`。采用用户已完成的 W&B 追踪器，不另造一套云端日志系统。  
**目标**：一条命令发起归档；给定 `run_dir + checkpoint + eval receipt` 可回答：谁训练、哪份代码/数据/Oracle/Reward、保存了哪些字节、哪些结果能复核、是否能从归档独立恢复；并在 W&B 精确关联里程碑。  
**边界**：这是通用归档工具的 **实施规范**，不是现成已可运行的本地命令。Codex 需要基于用户当前代码实现和验证。不得停止、修改、清理任何正在运行的 S3 实验；不得启动新 1M。

## 0. 设计取舍

- **原始实验绝不改写**：归档器只读源实验；任何临时文件、锁、状态、云同步缓存都在归档器自己的目录。
- **以事实而非文件名认定身份**：`800k` 名字只供展示。真实身份必须由本地 `run_identity.json`、双仓库真实执行代码身份、已完成的 Checkpoint manifest（含其字节 SHA256）及 Eval `receipt.json` 交叉确认。
- **归档不等于恢复能力**：有模型权重不代表 FullCheckpoint 能恢复 Replay、Oracle sidecar、优化器、Env、RNG；有历史 receipt 不代表能独立重复跑出评价。
- **归档目录不能依赖正在变化的源文件**：避免 `cp -al`、`rsync --link-dest`、普通硬链接、指向可写 run_dir 的符号链接；可使用 reflink Copy-on-Write 克隆（独立写入语义）或普通复制。源文件处于写入中则拒绝不一致快照。
- **不重复造 W&B Client**：本地证据是权威，W&B 只管理脱敏索引、标签和链接；离线时可以推迟 publish。
- **不因归档占用 GPU**：文件和哈希检查默认 CPU/磁盘；独立 GPU 恢复/评价属于可选显式验证，不能干扰正式训练。
- **市场时间**：新 S3 `S3_MARKET_MINUTE`，1 根有效 1 分钟 K 线 = 1 市场分钟；旧实验如时间口径不同，必须保留旧身份，不得篡改。

## 1. 归档等级（不要混用）

| 等级 | 名称 | 需要的实际内容 | 能宣称什么 | 不得宣称什么 |
| --- | --- | --- | --- | --- |
| **L1** | `EVIDENCE` | 身份、解析后配置、双仓库源码证据、Checkpoint manifest 索引、评价收据、校验值 | 当时做了什么、评价报了什么，可追溯 | 模型可以加载、评价可重复、训练可恢复 |
| **L2** | `EVAL` | L1 + 可加载的模型/策略、评价配置、所需输入数据的不可变快照或可核验的内容地址、隔离评价通过收据 | 在明确环境下能独立重做该评价 | 完整训练可从该点准确续跑 |
| **L3** | `FULL` | L2 + 完整 checkpoint 及所有外部恢复依赖（优化器、Replay、Sidecar、Env、RNG、游标、来源数据等），隔离恢复验证通过 | 从归档依赖集可按原代码身份继续训练 | 不经实测声称跨 GPU/驱动绝对逐位确定性 |

每次归档记录 `requested_level`、`achieved_level`、`verification_status`、`missing_dependencies`。不能因为选了 `--level full` 就直接标 L3。L2/L3 如不能验证，归档仍可按实际较低等级保留并报告 `PARTIAL`。

**重要的物理保留策略**：归档等级表示已经证实的能力，不表示是否复制过某个单独文件。对重要的 800K 里程碑，**可以在 L1 阶段就完整复制该已完成 Checkpoint**（建议 CLI 参数 `--preserve-checkpoint`），但在 Replay/Sidecar 等外部恢复依赖及隔离测试未通过前仍只标 L1；这样避免未来原训练清理旧 Checkpoint 而失去模型字节。必须先估算空间并由用户确认，且绝不能中断原训练。

**可恢复验证**应区分：`loadable`（模型可读）、`resume_smoke_pass`（独立目录恢复并至少运行一次受控更新或等价严格恢复契约）、`exact_replay_verified`（如确有更强的确定性逐步对照证据）。L3 至少 `resume_smoke_pass`，且不能依赖正在变化的外部数据。

## 2. 一条稳定的使用接口（建议实现）

```bash
# 只读查看，列出需要复制的对象、缺失依赖、预计体积、Git身份和来源健康度。
python tools/s3_archive.py inspect \
  --run-dir /data/logdir/fg_single3/s3_v2/TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M_v1 \
  --checkpoint 800000 \
  --eval-dir /data/logdir/fg_single3/s3_v2/TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M_v1/evaluations/actor_only_16be8b0985 \
  --json-out /tmp/s3-archive-inspect-800k.json

# 输出归档计划，默认不写入源实验，不创建归档。
python tools/s3_archive.py archive --plan /tmp/s3-archive-inspect-800k.json \
  --level evidence --preserve-checkpoint --dry-run

# 确认空间、依赖和快照一致性后，创建真实归档；首轮建议 L1 立即做，L2/L3 可追加升级。
python tools/s3_archive.py archive --plan /tmp/s3-archive-inspect-800k.json \
  --level evidence --preserve-checkpoint --execute

# 再次独立校验归档中的哈希、身份关联、文件缺失、目录可移植性。
python tools/s3_archive.py verify --archive-dir /data/logdir/fg_single3/research_archives/<archive_id>

# 可选：对隔离恢复 / 独立评价执行更强验证（默认不碰 GPU）。
python tools/s3_archive.py verify --archive-dir /data/logdir/fg_single3/research_archives/<archive_id> --mode resume-smoke

# 复用已完成的 W&B tracer，只发布索引与脱敏里程碑，不上传原始账户或大模型。
python tools/s3_archive.py publish --archive-dir /data/logdir/fg_single3/research_archives/<archive_id> \
  --project fg-single3-research

# 查询归档和状态，包含 offline 待发布里程碑。
python tools/s3_archive.py list --archive-root /data/logdir/fg_single3/research_archives
```

`archive --execute`、`verify --mode resume-smoke`、`publish` 应分开，不在用户只想记录历史时暗中占用 GPU 或联网。实际 CLI 名称可与已完成 W&B 项目目录布局协调，但要保留对应能力。Codex 须在 `--help` 中列出可复制命令。

## 3. 标准目录结构

```text
/data/logdir/fg_single3/research_archives/
  tutor-cd-mm-800k-<identity12>/            # 唯一 archive_id，包含真实来源身份摘要
    README.md                               # 人能读懂的归档说明与恢复限制
    archive_manifest.json                   # 规范化身份、等级、依赖、状态和签名引用
    SHA256SUMS                              # 归档文件 SHA256 清单（自身不纳入）
    reports/
      inspect.json                          # 原始健康检查与依赖发现
      verify.json                           # 最近一次校验结果与执行环境
      restore_report.json                   # 可选隔离恢复的真实结果
      eval_report.json                      # 可选重复评价结果
    provenance/
      run_identity.json
      resolved_config.yaml
      prepared_manifest.json
      dependency_versions.json
      code_identity.json                    # 双仓库SHA、工作树dirty、代码身份哈希
      source_bundle/                        # 精确源码树快照/所需Git Bundle
    checkpoint/
      ...                                    # L1默认索引；重要里程碑可额外保留已完成Checkpoint字节
                                             # 不因复制了模型就自动声明可恢复
    dependencies/
      ...                                    # L3外部Replay/Sidecar/Env/RNG等独立拷贝
    evaluations/
      fg409_5000_actor_only/
        receipt.json                        # 原始字节原封不动
        dev_env.yaml
        account_state.json                  # 仅本地私有归档，不上传W&B
    wandb/
      milestone.json                        # 脱敏里程碑索引
      publish_state.json                    # 幂等key、W&B Run引用/离线待同步
```

源文件存在与否取决于等级和可用性；不应为填满布局创建空文件冒充有效内容。源数据库原始价格、原始行情 CSV/Manifest 若较大，可做不可变内容副本或显式保留内容寻址的只读数据集；L2/L3 **只有这些依赖真正独立可取回且校验通过**才算达标。

**为什么优先用目录而非超大的 tar**：可以按等级升级、增量校验、独立抽取 Checkpoint 和供离线恢复。若需搬家，可另生成 tar.zst 传输包并校验，不能只用传输包代替归档的可审查文件清单。

## 4. `archive_manifest.json` 规范（最小字段）

```json
{
  "schema": "S3_ARCHIVE_V1",
  "archive_id": "tutor-cd-mm-800k-<identity12>",
  "archive_state": "VERIFIED_EVIDENCE",
  "requested_level": "EVIDENCE",
  "achieved_level": "EVIDENCE",
  "created_at_utc": "<ISO8601>",
  "source": {
    "experiment_family": "s3-tutor-cd",
    "local_run_id": "<verified run_identity>",
    "original_run_dir": "<source path - local only>",
    "checkpoint_steps": 800000,
    "checkpoint_manifest_sha256": "<actual byte SHA256>",
    "eval_receipt_sha256": "<actual byte SHA256>",
    "eval_receipt_content_hash": "<receipt content_hash if present>"
  },
  "identity": {
    "gym_git_sha": "<verified>",
    "dreamer_git_sha": "<verified>",
    "gym_dirty": false,
    "dreamer_dirty": false,
    "code_snapshot_sha256": "<actual tree/snapshot hash>",
    "resolved_config_sha256": "<actual>",
    "reward_identity": "<actual>",
    "oracle_identity": "<actual>",
    "train_manifest_identity": "<actual>",
    "clock_protocol": "S3_MARKET_MINUTE"
  },
  "dependencies": [],
  "verification": {
    "hashes_valid": true,
    "source_identity_verified": true,
    "eval_reproduced": null,
    "resume_smoke_pass": null,
    "missing_dependencies": []
  },
  "wandb": {
    "project": "fg-single3-research",
    "parent_train_run_id": null,
    "eval_run_id": null,
    "milestone_publish_state": "PENDING"
  }
}
```

其中示例布尔值仅说明结构：**生成时必须根据实际验证结果填写**，不能凭资料存在就给 `true`。未验证的字段使用 `null` 或解释性 `NOT_RUN`，严禁填 0。完整绝对路径允许留在本地 `manifest`，上传 W&B 时必须另生成脱敏摘要，不把用户路径、环境变量或秘密自动上云。

状态推荐：`INSPECTED`、`STAGING`、`VERIFIED_EVIDENCE`、`VERIFIED_EVAL`、`VERIFIED_FULL`、`PARTIAL`、`FAILED`；`PUBLISHED` 作为 W&B 独立状态，不覆盖归档完整性等级。

## 5. 四步流程的详细规则

### A. `inspect`：只读发现与身份核实

1. 读取训练 `run_identity.json`、**该 Run 的** `config.yaml`、`prepared/manifest.json`、相关 source admission / oracle identity，提取双 Git SHA 和代码身份。不得使用另一次实验的 config，也不得把当前 worktree HEAD 当成历史运行源码。
2. 确定 `ckpt/step_000000800000` 及它的 manifest `done` 完成标志，计算真实文件字节的 SHA256，交叉对照评价收据记录的 `checkpoint_manifest_sha256`。不一致立即阻止发布正式里程碑。
3. 将 `receipt.json` 中的 fold、contract、env_steps、teacher_guidance、beta、policy_mode、计时语义等与用户设定的里程碑判据核对。必要时读取 `account_state.json` 本地独立复算交易笔数/盈亏。
4. 解析 Checkpoint 的实际依赖图：哪些对象包含在 checkpoint，哪些放在 Replay 根目录、teacher Sidecar、Env state、数据库/CSV、训练 runtime 之外；不要仅因为 FullCheckpoint 这个名字就假定文件独立。
5. 检查数据来源文件的版本与内容身份；检查 Git tracked、dirty patch、untracked execution files、submodules 与 Git LFS（若存在），输出源代码可重建性报告。
6. 估算空间：需要复制的总体积、归档盘空闲量、安全余量；重复归档时先比对 archive_id 与来源身份证据，不能覆盖旧 Archive。
7. 明确哪些源文件正被训练进程使用和修改。正在写入的 Replay、Sidecar 等若无法取得一致快照，标注 `SNAPSHOT_UNSAFE`，L3 不可执行。已完成 Checkpoint 原则上可单独保存（仍须确认保存器没有原地改写）。
8. 生成清晰的 `inspect.json`：`ready`, `warnings`, `missing`, `copy_plan`, `estimated_bytes`, `available_bytes`, `proposed_achieved_level`。如果关键身份不符，则 `ready=false`。

### B. `archive`：最小干预复制，原子发布

1. 在归档根目录外或同一文件系统 `.staging/<archive_id>.tmp` 创建 staging；源 run_dir 和源码工作树完全只读。
2. 优先使用只读的已完成 Checkpoint 与原评价文件，复制后比较源与目的完整文件哈希；归档进行中源若改变，报错并停止提升等级。
3. Git 源码必须有**独立可重建版本**：对核实的双 SHA 保留必要 Git Bundle（防止未来分支删除与 `git gc`）或完整有效源树快照；对 dirty 状态还要保存真实 tracked diff 和执行相关未跟踪文件快照。不要只创建 detached worktree 并宣称长期独立存档，因为 Worktree 仍依赖共享 Git 对象库。
4. 可用 reflink 独立拷贝或普通文件复制，禁止将源目录中的可变文件硬链接到归档。不要在复制前假定源日志仍静止：日志可按明确的已确认 checkpoint 水位截取封存，日志其他部分可以不复制，但必须在报告中说明。
5. L2/L3 复制外部依赖时，需要验证一致的时间点/版本身份；并发写入期间不能仅靠顺序 `cp` 宣称一致快照。若无法取得稳定的 Replay/Sidecar 快照，就保持 L1，待停止或有安全快照机制后再升级。
6. `archive_manifest.json` 只记录外部证据身份与派生元数据。将所有需要校验的正式文件纳入 `SHA256SUMS`；自校验文件（例如 `SHA256SUMS` 本身）不放进去，避免循环哈希。最终 `verify` 通过后，才将 `.staging` 原子 `rename` 成正式归档目录（同文件系统）。失败 staging 保留可诊断状态，不用成功的正式命名掩盖失败。
7. 可选设只读文件权限防误操作；**不是安全边界**。建议额外的第二存储设备/异地备份负责抵御磁盘损坏，W&B tags 不能替代模型文件备份。

### C. `verify`：独立证明，不写源 Run

- **L1**：完整 SHA256 检查；确认 checkpoint manifest 与 receipt 交叉 SHA 匹配；双源码身份可重建；冻结配置来源正确；历史交易数与收据一致；`README.md` 写明已达到哪一级别。
- **L2**：额外从归档复制到临时隔离 run_dir，用对应冻结源码和输入数据，独立运行 Actor-only 固定 DEV 评价，与原 receipt 比较关键指标（合法决策、ADD、TP/SL、Reward、市场分钟、净盈亏），并记录允许浮点容差与差异。不能重新使用未来 Oracle 指导。
- **L3**：在独立路径载入归档完整状态（含模型、优化器、Replay、Sidecar、Env、RNG、游标）；验证加载成功、恢复的更新计数/数据水位与冻结快照一致，再受控推进有限更新（可能需要 GPU，须单独许可/确保资源不冲突）；检验新 Run 不写旧 run_dir。若没有资源进行恢复实验，只能记 `PENDING_RESUME_TEST`，不能标 `VERIFIED_FULL`。
- 每次 `verify` 产出 `reports/verify.json`，包含执行环境和测试日志，不改旧文件。重新校验可能在原 Archive 下追加版本化验证报告，更新 manifest 要重封存与重算相应校验；优先把状态单独记在版本化 `verification/` 下以保证核心归档内容不变。

### D. `publish`：W&B 里程碑与索引

1. 复用当前已经完成的 W&B tracker 的真实 Run 身份映射；以 `(W&B project, eval receipt content_hash/byte SHA, checkpoint manifest SHA)` 作幂等键，**绝不能通过 Display Name 猜 Train/Eval Run**。
2. 精确找到 `fg-single3-research` 里对应的 Eval Run；如还未导入，调用既有追踪器支持的 `import-evals` 等实际入口；不存在或身份不符时只生成 `PENDING` 本地记录，不能创建冒名里程碑。
3. 为真实 Eval Run 追加里程碑 Tags，不删除原 tags；同时记录 `archive_id`、`achieved_level`、`checkpoint manifest SHA`、`receipt SHA`、`parent_train_run_id`。Train Run 可追加指向该 Eval Run 的索引（如果训练同步器运行中有冲突则推迟）。
4. 只上传脱敏简要指标与归档 manifest 摘要，不上传完整 `account_state.json`、密码、私钥、CSV、FullCheckpoint、Replay 或未来教师标签。
5. `publish` 重复运行只修复缺漏字段，不创建第二条同一证据的里程碑；网络失败/权限缺失自动生成本地待同步任务，归档不失效。

## 6. Tutor-CD 800K：首个标准归档示例

### 已有用户提供的评价事实

- 本地 run：`TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M_v1`
- Checkpoint：`step_000000800000`
- 评价：`actor_only_16be8b0985`，`CZCE.FG409`，DEV、5000 环境步、`eval_argmax`、无 Oracle 接管。
- `receipt.checkpoint_manifest_sha256`：`5cf093b56ca3bb340cca9520b5579105b1cecc9f044262d5a23d17c545c6a474`
- `receipt.content_hash`：`7a1c6cc66e7c7ec4bc6b3b6f5bb8984a96b7bb3a6bfb3746600a8cd5a15c9368`
- 7 次自主 ADD、5 TP、2 SL、TP 中位数 74 市场分钟、Reward `-3.805870048525721`、交易净盈亏 −2,226。
- 证据含 `receipt.json` 与交易 `account_state.json`。以上均用于 **inspect 核验**，不是来源文件 SHA 的替代品。

### 推荐打的 W&B 标签

对**精确对应的独立 Eval Run** 添加：

```text
milestone:first-observed-autonomous-add
checkpoint:800k
eval:actor-only
clock:S3_MARKET_MINUTE
archive:S3_ARCHIVE_V1
```

`first-observed` 仅表示在已执行、已检查的评价中第一次观察到，**不宣称历史真实最早出现 ADD**。不要标 `profitable`、`best`、`production-ready`。完整记录通过 Eval Run summary/小型 manifest 字段保存。

### 这次应当怎样做

1. **立刻完成 inspect**，核对真实源码与 checkpoint 关系，判断800K完整Checkpoint是否独立，输出磁盘需求。
2. **先做 L1 Evidence Archive，建议同时 `--preserve-checkpoint`**，在确认空间和源Checkpoint已完成稳定后把800K模型字节独立保留下来。Tutor 1M 正在训练也不应因此被停止；若拷贝会造成磁盘风险，先报告空间需求并暂缓重型复制。
3. 如果800K checkpoint已保留完整的独立状态且数据依赖可安全复制，可升级 L2/L3；否则严格停在当前等级，报告缺失原因，不为了追求 FULL 打断训练。
4. 先 publish W&B 评价里程碑，若项目离线则保存待发布任务；publish 不应等待 L3。
5. Tutor-CD Full / B2 Full 开发可继续；开发和归档相互独立，不更改冻结 Tutor 工作树。

## 7. 对 Codex 的实施任务与验收

**任务定位**：开发可复用的 S3 归档器、归档标准文档和800K首个应用。尽量复用现有训练工件格式与已经做好的 W&B tracker，**不开发后台常驻任务、Web服务、统一模型仓库或复杂去重数据库**。

实施顺序：

1. 核实本地最新的 W&B tracer CLI/API、两个 Git 仓库、800K 原始文件、真实运行源 code identity 与 Checkpoint 依赖，形成 `SOURCE_FACTS.md`；若800K依赖有重大不可访问项，优先报告而非猜测。
2. 创建独立归档工具（建议 `tools/s3_archive.py` + 小型 `s3_archive/` 包），位置避开旧训练源码身份扫描路径，不修改冻结训练树。
3. 实现 `inspect`、`archive --dry-run/--execute`、`verify`、`publish`、`list`。默认 `archive` 干跑，必须 `--execute` 真正写入；`publish` 单独联网上传。
4. 加入 `S3_ARCHIVE_V1` manifest、归档等级与状态、`--preserve-checkpoint`（可在L1保留关键模型但不得虚报L3）、源文件依赖核对、源文件前后 SHA、完成检查点/receipt交叉哈希、Git bundle/dirty源码快照、无硬链接/不依赖外部可写源、空间检查、原子 staging、重复运行幂等、清晰故障报告。
5. 实现 800K L1 并尽可能在可验证安全条件下升级 L2/L3；不得将当前运行变化中的 Replay 直接复制后宣称 FULL。必须真实验证、如实报告 achieved level。
6. 使用已有 W&B 项目 Run 索引添加800K tag与脱敏里程碑；如果项目无权限/未导入 Eval Run，保留本地待同步结果，绝不声称已发布。
7. CPU 测试包括：有/无 Checkpoint manifest done、收据SHA不匹配、配置来自错误Run、dirty代码、重复执行、空间不足、源文件在复制中变化、共享 replay不稳定、tag幂等、W&B offline、隐私过滤、验证未通过不能升级等级。
8. 在独立的800K真实数据上执行至少 `inspect` 与 `archive`（用户确认可用磁盘空间后）；运行 `verify` 和可行的 publish；不启动新的 GPU 研究训练。
9. 提交代码与 Git SHA、工具帮助和文档、实际 archive_id/路径、实现等级、真实验证记录、W&B Run URL或 pending 状态。不得仅生成一个报告说可以归档。

### 验收条件

- 可以用不同的 `run_dir + checkpoint + eval` 重复归档 A2、Tutor、B2、E，而非硬编码800K。
- 原 run_dir / 训练 Worktree 的目录内容保持不变；对原训练没有写入、重启、中断；本地 GPU 不被默认调用。
- 实验身份全链路：配置、两代码版本、数据/Oracle/Reward/市场分钟、checkpoint、评价与 W&B Run 能反向追踪；缺失事实如实标记。
- 可生成带校验结果和恢复声明等级的独立 Archive。L3 需隔离恢复 smoke 的实际收据，没有就不标 L3。
- W&B 有真实 Eval Run tag 或明确可同步的 pending file；旧 W&B tags 原样保留；仅允许小型脱敏元数据云同步。
- 归档器失败后可重试，不覆盖已有 Archive，不产生多份同一里程碑的冲突身份。

## 8. 标准人工操作清单

1. 确定研究意义：为什么值得归档、关联哪个指标和 Checkpoint。
2. 找到 **该 Run 的** `run_identity`、配置快照、真实仓库 SHA；找具体 `checkpoint manifest + done` 与 Eval receipt。
3. 运行 `inspect`，确认精确来源，磁盘空间足够；若有不稳定 Replay，先选 L1。
4. 运行 `archive --dry-run` 查看大小和文件列表，确认不写原 run_dir。
5. 运行 `archive --execute`，得到 archive_id；运行 `verify`，检查实际 achieved level。
6. 运行 `publish`，在 W&B 验证真实 Eval Run tags；离线则标 pending。
7. 在归档索引登记 `archive_id`、本地路径、Checkpoint manifest SHA、W&B URI、等级及恢复限制。
8. 需要移机或长期保存时另复制到第二存储位置、重跑 `verify`，避免只有单块磁盘一份数据。

---

**特别声明**：截至本任务书，用户刚完成的 W&B 真实实现代码与本地800K完整Checkpoint并未上传，故无法在此验证已有 W&B 实际API、归档文件大小、800K完整恢复依赖、是否已经在W&B打Tag。Codex必须读取真实源后如实报告。归档工具实现和首个真实归档尚未执行。
