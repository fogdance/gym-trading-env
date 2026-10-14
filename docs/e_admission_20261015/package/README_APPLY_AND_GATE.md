# FG S3-E — 双仓合成补丁 / Git 历史与训练准入

## 交付状态

- **数据/Data 源**：Gym `feat/s3-e-data-v1` @ `720a66a`；已在隔离环境中合入 **Gym Model** `feat/s3-e-model` @ `05141ee`；合成结果 `5507ace`（隔离副本中的 `review/s3-e-synthesis`，不是用户主机上的现存提交）。
- **模型/Model 源**：Dreamer `feat/s3-e-model` @ `f30d8da`；合成结果 `03ea881`（隔离副本中的 `review/s3-e-model-patched`）。
- **正式训练判定**：**NO-GO**。已补接口，真实 GPU/完整 checkpoint/正式源码准入尚未验收；公开 `waiting_e` 继续阻断，这是有意保留的安全机制。

## 文件清单

- `FG_S3_E_GYM_DATA_TO_MODEL.patch` — 在 Gym `05141ee` 上应用的全量合成修复（包含数据分支和 Provider）。
- `FG_S3_E_DREAMER_MODEL_FIX.patch` — 在 Dreamer `f30d8da` 上应用的增量修复（来源认证、Provider 接线及 Review）。
- `FG_S3_E_GYM_HISTORY.bundle` — Gym **真正的 Data→Model Git merge 历史**和后续修复提交；保留 parent DAG。**推荐优先使用 Bundle**。
- `FG_S3_E_DREAMER_HISTORY.bundle` — Dreamer 的 Model 后续修复与 Review 提交。
- `REVIEW_REPORT.md` — 上述代码级双向 Review、风险矩阵及 GO 条件。

## 执行规则

所有命令需在 **用户主机**的真实仓库执行，并由 Codex 先核对 HEAD、工作树、现有运行与 Worktree。**不要**直接在 `/home/v/Documents/work/gym-trading-env` 和 `/home/v/Documents/work/dreamerv3` 这两个其他研究仓库执行。不要在被冻结的正式训练 Worktree 上编辑。

本次两个 Git 仓库独立；**不存在跨 Gym/Dreamer 的单次 Git Merge**。

### 方式 A — 保留真正 merge 历史（推荐）

在 `/home/v/Documents/work/research/fg-s3-e-model-dev/{gym-trading-env,dreamerv3}` 上执行，`BUNDLE_DIR` 替换为本包解压目录：

```bash
ROOT=/home/v/Documents/work/research/fg-s3-e-model-dev
GYM="$ROOT/gym-trading-env"
DR="$ROOT/dreamerv3"
BUNDLE_DIR=/absolute/path/to/extracted_bundle

# 只读：核对完整 SHA 和未提交改动，出现差异先停
for p in "$GYM" "$DR"; do git -C "$p" status --short;git -C "$p" rev-parse HEAD; done
# Gym HEAD 必须等于 05141eefb438fd0a6038f32419e209f5fc11149f
# Dreamer HEAD 必须等于 f30d8da2523eddad53aeca1540afe1c77d47b248

# 校验通过后，在真正 Model Worktree 内执行：
git -C "$GYM" bundle verify "$BUNDLE_DIR/FG_S3_E_GYM_HISTORY.bundle"
git -C "$DR" bundle verify "$BUNDLE_DIR/FG_S3_E_DREAMER_HISTORY.bundle"
git -C "$GYM" fetch "$BUNDLE_DIR/FG_S3_E_GYM_HISTORY.bundle" \
  refs/heads/review/s3-e-synthesis:refs/heads/review/s3-e-synthesis
git -C "$DR" fetch "$BUNDLE_DIR/FG_S3_E_DREAMER_HISTORY.bundle" \
  refs/heads/review/s3-e-model-patched:refs/heads/review/s3-e-model-patched

# 两仓 source HEAD、status 均与上述完全一致且 bundle 已验证后：
git -C "$GYM" merge --ff-only review/s3-e-synthesis
git -C "$DR" merge --ff-only review/s3-e-model-patched
```

如果当前 Model 分支含有新提交或 dirty diff，禁止强制替换、rebase/reset；改为先 Review 冲突/Cherry-pick；不要照着旧 SHA 直接执行。此合成代码仍不允许正式训练。

### 方式 B — 只应用补丁（不保留 Merge DAG）

上述完整 SHA 和干净状态必须满足。运行：

```bash
git -C "$GYM" apply --check "$BUNDLE_DIR/FG_S3_E_GYM_DATA_TO_MODEL.patch"
git -C "$DR" apply --check "$BUNDLE_DIR/FG_S3_E_DREAMER_MODEL_FIX.patch"
# Codex确认两侧 diff 完整、不会改运行中的源后：
git -C "$GYM" apply "$BUNDLE_DIR/FG_S3_E_GYM_DATA_TO_MODEL.patch"
git -C "$DR" apply "$BUNDLE_DIR/FG_S3_E_DREAMER_MODEL_FIX.patch"
# 随后分别测试、分别创建 Git Commit。
```

**A 与 B 二选一，不要重复执行。** Bundle 用来保留真实合并历史；Patch 用来查看/应用文件差异。

## 已在隔离归档副本运行的验证

- `tests/e_data_v1`: **28 PASS、2 SKIP**（跳过原 Env CPU 引用，与 `gymnasium` 未安装有关）；新 Provider 8 个测试 PASS。
- `docs/e_data_v1/SHA256SUMS`: **PASS**。
- 两份 patch 在精确原始 Model base 下分别 `git apply --check` 与 apply: **PASS**。
- 新 Python 文件 `py_compile`: **PASS**。
- Dreamer JAX/真实 Env/model GPU：**NOT_RUN**（缺依赖及真实1.3GB TRAIN assets，不能伪称通过）。

## 真正 GO 必须另行完成

1. 原 TRAIN pool 与 E 全量 collection 的 hash/schema 和模型来源完全一致；完整真实每分钟 Snapshot E→Env→DAgger→Replay→Actor/WorldModel，包含跨合约和 reset。
2. A2、Tutor、B2 三套 E 模式各自真实 CPU 准入、数据 Join、梯度、full-state 保存与恢复一致。
3. 基于真实证据单独解除三种 `waiting_e`；Tutor+B2 仍不启用。
4. 完整 size50m GPU 1K smoke、显存、耗时、非法动作/ADD-WAIT比例和 FullCheckpoint恢复通过；正式1M可用独立 Run ID 和冻结双仓 SHA。

更多细节在模型补丁中的 `docs/S3_E_DATA_MODEL_M4_REVIEW_20261015.md` 与 Data 补丁中的 `docs/e_data_v1/M4_RUNTIME_PROVIDER_REVIEW.md`。
