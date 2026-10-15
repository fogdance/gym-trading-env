# FG S3 Quality Teacher V1 源码补丁 Review（独立）

**基线**：用户上传 Git 归档 `gy14.tar.bz2 / d314.tar.bz2`，Gym `feat/s3-e-model@871ca09`、Dreamer `feat/s3-e-model@730939d`。此次补丁只涉及Gym新增文件，Dreamer无代码改动；归档并非用户电脑实时工作区或未来未提交更改。

## 1. 覆盖文件（不修改旧文件）

- `src/gym_trading_env/research/fg_s3_quality_score.py`：新TP事件基础分、跨官方交易日评分、退出质量合法域。
- `src/gym_trading_env/research/fg_s3_quality_teacher.py`：原账户完整快照→原Env独立Fork→一次ADD→WAIT至退出/右删失→事实记录、MAE/时间/新TP替换Reward；可选严格Oracle并排记录。
- `tests/entry_v1/test_fg_s3_quality_teacher.py`：评分锚点、跨日、Invalid、独立TP/SL、亏损后真实账户、严Oracle拒证不阻断质量、原状态不变、禁TEST/持仓标注。

## 2. 已实际执行的检查

|检查|结果|证据边界|
|---|---|---|
|patch独立生成|PASS|单独版本、无既有文件修改；`git diff --cached` 只有3个new file|
|在干净 `feat/s3-e-model` 基线应用|PASS|`git apply --check`与`git apply`均真实完成，无冲突|
|Python AST/compile|PASS|三个新增 `.py` 均可解析|
|TP锚点和边界|PASS|11组校验：4/10/30/60/74/120/240/480分钟，d=1/2/6；加448个值域检查|
|完整Pytest原Env集成|**NOT_RUN / BLOCKED_BY_ENV**|本评审容器缺少`gymnasium`，pytest收集时报ModuleNotFoundError；不能宣称真实业务集成PASS|
|生产Env/Replay/Checkpoint改变|NONE|补丁没有触及旧源码文件；未运行训练|
|TRAIN真实行情/亏损后资金路径验证|**NOT_RUN**|上传Git对象不包含真实数据、冻结Env运行资产|
|新Reward正式启用或新Oracle重算|**NOT_IMPLEMENTED**|Stage4单独协议，旧C1未动|
|E Obs clone非污染证明|NOT_ESTABLISHED|V1显式禁止E-profile Fork，防止共享E市场缓存被误写|

## 3. 数学与科学审查

**P0已处理：** 30万元资金认证与单笔质量评分剥离；不会调用`funds_certificate`；`strict_decision_valid=false`时仍可存在 `quality_valid=true`。真实账户是否能够成交、是否强平由Fork原Env决定。没有资金下界伪造“当前最优”。

**P0已处理：** 质量 ≠ 最优动作。`training_action_label_allowed=false`、`add_better_than_wait=null`；TP90分绝不变成 `teacher_action=ADD`；SL失败不代表严格Oracle WAIT。右删失和OTHER退出不以0伪标签监督。

**P0已处理：** 新TP质量无MAE额外罚。原C1早期逆行和SL惩罚不变；新离线Reward仅对已确认TP的旧事件值做差额替换；没有修改原C1 Env Reward/Optimizer。

**P0已处理：** 无未来进Actor观测。后验路径只在fork中消费，返回旁路research结构；无新Collector/Trainer/Replay连接，无法自动发生下游数据泄漏。

**P0已处理：** Fork原有`close()`会向 `state_path` 写盘；补丁**不调用**Fork的`close()`，原Env前后完整export_state hash必须相同。

**P1需进一步验证：** `fork_account`与原Env动态引用、私有账户ID/随机状态在真实FG201/FG409历史上的分离；假定旧`fg_single3_horizon.fork_account`已完成原项目的状态克隆验收，但本环境无法独立重复。

**P1需进一步验证：** 新官方日评分要求真实`trading_date`，人工夹具需补充该字段；缺列时拒绝评分而不是按墙钟日期凑标签。跨交易日跳变 d=0→1属于用户明确规定，而不是bug。

**P1需进一步验证：** 真实候选反事实会读取全段未来，适合TRAIN离线，不适合在线Actor决策；若批量对全TRAIN状态逐笔Fork，性能可能高成本，需未来Catalog批处理和去重缓存，但缓存不得忽略账户资金差异。

**P1明确限制：** 这个补丁没有真实成交后的TP概率分布预测能力，未实现质量头的神经更新，更不能声称已改善Tutor-CD/Full 1M表现。

## 4. 下一步真正的本机验收（不自动执行）

```bash
# 在一组明确指定的、独立的Gym开发worktree中进行，不在历史训练worktree直接apply。
git status --short
git rev-parse HEAD
git apply --check /path/to/FG_S3_Quality_Teacher_V1_Gym.patch
git apply /path/to/FG_S3_Quality_Teacher_V1_Gym.patch
PYTHONPATH=src python -m pytest -p no:cacheprovider -q \
    tests/entry_v1/test_fg_s3_quality_teacher.py \
    tests/entry_v1/test_fg_single3_env.py \
    tests/entry_v1/test_fg_single3_oracle.py
```

还应使用真实 TRAIN FG201 原账户签名状态重复0、1、2次SL的Fork对账；对同一cursor等资金和低资金状态检查Env后果，确认只要合法就有事实标签；记录对照收据而不改旧证书。

## 5. 总评

**Stage1源代码静态评审 = CONDITIONAL_GO_FOR_ISOLATED_CPU_TESTS**。已经生成独立可应用的Gym patch，设计边界清晰，不影响旧Oracle/Reward/模型，但没有原Env运行验收。**NOT_GO_FOR_PRODUCTION_TRAINING**：缺真实项目依赖/行情、自主样本Catalog和质量Head的Loss接线，不可直接用于新的1M训练。
