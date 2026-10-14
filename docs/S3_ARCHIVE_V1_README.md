# FG Single3 Research 标准归档操作手册

规范为 `S3_ARCHIVE_V1`。统一入口是 `tools/s3_archive.py`；包在仓库根目录，普通归档不导入训练/JAX、不启动GPU或联网。Project固定为 `fg-single3-research`，发布直接复用 `research_tracking`。

## 环境与入口

在独立归档工作树执行。文件归档需要 `requirements-archive.txt`（PyYAML、NumPy）；匹配的原训练环境已有这些依赖。W&B发布使用已有隔离追踪器环境，不往训练环境安装SDK。

```bash
cd /home/v/Documents/work/research/fg-s3-archive-dev/gym-trading-env
ARCHIVE_PY=/home/v/miniconda3/envs/dreamerv3/bin/python
TRACKER_PY=/home/v/Documents/work/research/fg-s3-wandb-env/bin/python
"$ARCHIVE_PY" -B tools/s3_archive.py --help
```

未安装NumPy的环境仍可检查证据，但pickle恢复水位会明确记为缺失；需要Full时使用匹配的训练Python重新inspect。GPU验证必须通过 `--python` 显式指定原依赖环境；项目和源码原文件都保持只读。

## 通用流程

用实际Run、Checkpoint步数、具体评价目录替换参数，不通过文件名猜实验身份。历史HEAD可与当前开发树不同；归档采用冻结记录中的历史提交和执行文件哈希。

```bash
"$ARCHIVE_PY" -B tools/s3_archive.py inspect \
  --run-dir "$RUN" --checkpoint "$STEP" --eval-dir "$EVAL" \
  --json-out /tmp/s3-archive-plan.json

"$ARCHIVE_PY" -B tools/s3_archive.py archive \
  --plan /tmp/s3-archive-plan.json --level evidence --preserve-checkpoint --dry-run

# 阅读空间需求后执行；默认预留10GiB，不暂停或清理源实验。
"$ARCHIVE_PY" -B tools/s3_archive.py archive \
  --plan /tmp/s3-archive-plan.json --level evidence --preserve-checkpoint --execute

"$ARCHIVE_PY" -B tools/s3_archive.py verify --archive-dir "$ARCHIVE"

# 同一archive_id追加恢复依赖，保留上一版目录；此命令本身不做模型验证。
"$ARCHIVE_PY" -B tools/s3_archive.py archive \
  --plan /tmp/s3-archive-plan.json --level full --execute

# 显式模型验证：默认hashes模式不触发GPU。
"$ARCHIVE_PY" -B tools/s3_archive.py verify --archive-dir "$ARCHIVE" \
  --mode eval --device gpu --python "$ARCHIVE_PY"
"$ARCHIVE_PY" -B tools/s3_archive.py verify --archive-dir "$ARCHIVE" \
  --mode resume-smoke --device gpu --python "$ARCHIVE_PY"

"$TRACKER_PY" -B tools/s3_archive.py publish --archive-dir "$ARCHIVE" \
  --project fg-single3-research --proxy http://127.0.0.1:10808
"$ARCHIVE_PY" -B tools/s3_archive.py list
```

`--prepared`用于自动发现有歧义的prepare快照，`--family`用于显式研究分类（如E2仍保留原mode=tutor），`--publication`选择追踪器已有重发布映射。重要性标签通过inspect的可重复 `--milestone-tag` 明确选择；工具不会把所有新ADD都叫首次观察。

## 等级、目录与证据

| 等级 | 必要验证 |
|---|---|
| L1 EVIDENCE | 双源码、配置、Checkpoint/receipt身份与完整归档SHA256校验，历史结果复算 |
| L2 EVAL | L1加隔离Actor-only评价通过；关键计数及净盈亏精确一致，浮点容差1e-6 |
| L3 FULL | L2加完整状态恢复、Replay数组逐项对照及一次受控优化器更新通过 |

有Checkpoint字节仍可能只达到L1。`requested_level`、`achieved_level`、`verification_status`、`missing_dependencies`分别报告请求与实际结果；复制Full不会自动获得L3。`exact_replay_verified`保持null，数组相等和一次更新不代表跨硬件逐步确定性。

默认归档根目录是 `/data/logdir/fg_single3/research_archives/`。`provenance/source_bundle/`保存独立完整源树和必要执行补丁；`checkpoint/`保存原manifest/done以及选择保留的模型状态；`dependencies/`保存恢复依赖及确认水位日志；`evaluations/`保留原字节证据。`SHA256SUMS`排除所有自身清单，避免循环校验。核心来源字节不修改，验证报告在 `verification/` 版本化保存。

升级利用Linux同文件系统 `renameat2(RENAME_EXCHANGE)` 原子交换完整目录；旧版本在归档根的 `.revisions/` 中继续保留。平台不支持交换时保留staging，不替换旧归档。外部符号链接与源硬链接被拒绝。

验证和发布的元数据通过归档锁及 `.transactions/` 持久日志更新；中断后恢复上一份已封存元数据，失败诊断保留而不删除。GPU执行脚本、隔离命令、日志及重评收据会一并保存在版本化 `verification/`，无需依赖临时验证目录查看验证证据。

隔离验证使用Bubblewrap，数据和源码命名空间只暴露归档副本。签名文件内的旧绝对路径通过私有挂载映射，不改配置或身份。恢复只执行一次更新，不执行正式 `start` 或 `run_to`，不会推进环境或保存新正式Checkpoint。

## 故障、重试与发布

复制失败保留 `.staging/` 和 `.failures/` 诊断。源身份仍匹配时重复原命令，已完成的独立文件重新校验后复用；不自动删除失败目录。源内容变化需重新inspect，冲突文件不覆盖。

W&B `PUBLISHED`与归档等级分开。仅发布脱敏摘要、身份哈希和标签；原账户流水、行情、Replay、Oracle标签及完整模型均不上传。离线可先运行 `publish --mode offline`，随后重跑online命令。映射不存在时复用原追踪器 `import-evals`，身份冲突则保留PENDING，不按Display Name选择Run。重复发布使用同一证据幂等键，并云端回读验证。

## Tutor-CD 800K

使用原Run `TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M_v1`、800000步和评价 `actor_only_16be8b0985`。inspect增加：

```bash
--publication tutor-cd-reupload-v1 \
--milestone-tag milestone:first-observed-autonomous-add
```

真实交付路径、最终等级及报告见 `S3_ARCHIVE_V1_TEST_REPORT.md`。旧一次性归档保留原样，旧 `FULLY_RESTORABLE`声明不自动转换为本标准L3。

## 测试

```bash
"$TRACKER_PY" -B -m unittest s3_archive.tests.test_archive -v
"$TRACKER_PY" -B -m unittest research_tracking.tests.test_tracker -v
```

离线SDK需要本地服务套接字；沙箱阻止启动时在正常主机用户环境重跑。没有新正式训练。归档是单机持久化，不代替第二存储或异地备份；迁移后重新执行verify。
