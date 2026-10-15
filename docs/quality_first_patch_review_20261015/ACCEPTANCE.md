# Quality-first Gym Patch 本机验收（2026-10-15）

## Summary

**PASS_STAGE1 / GO_FOR_ISOLATED_CATALOG_DEVELOPMENT**。原附件的三个 Python 文件逐字节保持不变；通过实际安装依赖的本机 CPU、两套独立 Gym/Dreamer worktree、真实 TRAIN FG201 行情完成验收。没有启动训练、读取 TEST 或修改历史 Run。

附件 ZIP SHA256：`a4f1048995079ede07b3a0a3ea058862a4c7cae5f9d539b3f889fd1a90aa7a33`。源码完整性见 [source_integrity.json](source_integrity.json)。

## Evidence observed

| 验收源码基线 | Gym | Dreamer（本补丁不修改） |
| --- | --- | --- |
| 附件注明的基线 | `871ca09000712d41408b8f72e0684624dce8659f` | `730939defb1bd77b2ceca04212105ec2727c95f9` |
| 当前 integration/s3-next | `c3a808437ef277ee87f25b0ae8803e31fa5dca74` | `eb63de710fd693fad7ac76f797f3d9f39e331b98` |

两个独立 worktree 分别为 `fg-s3-quality-first-patch-review/` 与 `fg-s3-quality-first-integration-review/`。两个基线上各 **41 tests / 0 failures / 0 errors**：附件原测试 14 个、原 Env/Oracle 回归 20 个、补充明确边界 7 个。

- [附件基线测试收据](attachment_base_all_tests.xml)
- [集成基线测试收据](integration_all_tests.xml)
- [附件基线 FG201 完整账户收据](attachment_base_fg201_evidence/receipt.json)
- [最终集成基线 FG201 完整账户收据](fg201_real_account_final/receipt.json)

真实行情来自冻结 TRAIN 清单中的 `CZCE.FG201`：31,260 行、92 个官方交易日；市场 SHA256 `ebad6ed8b214e0a5801b0c16bb7e4784a6cd00ba943bf0c76cc3a571ed0a2e8c`。通过原 `load_train_query()` 认证 Env、候选、旧 Oracle 收据及生产者源码，没有跳过身份检查。

| 真实原生账户轨迹 | cursor | 起始权益（元） | ADD 后果 | 有效市场分钟 | B 分 | 旧 Strict 认证 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 0 次 SL | 0 | 300000.00 | TP，净利 282 | 2 | 100 | 有效 |
| 第 1 次 SL 后 | 47 | 299592.00 | TP，净利 282 | 37 | 95.78125 | FUNDS_OUTSIDE_ORIGINAL_CERTIFICATE |
| 第 2 次 SL 后 | 187 | 298338.00 | TP，净利 282 | 22 | 98.125 | FUNDS_OUTSIDE_ORIGINAL_CERTIFICATE |
| WAIT-only 到相同 cursor | 47 | 300000.00 | TP，净利 282 | 37 | 95.78125 | 有效 |
| WAIT-only 到相同 cursor | 187 | 300000.00 | TP，净利 282 | 22 | 98.125 | 有效 |

亏损账户通过实际 `ADD whenever legal` → 原 Env 止损形成；未编辑余额或以 `empty_at()` 重建账户。每次 SL 净亏 1818 元；累计账户权益还包含前序真实 TP。低资金账户 mask 均为 `[1,1]`。

全部 5 组均与**重新构造、import 完整快照的原 Env**逐项对账：完整最终 export_state、原生退出事件 hash、净 PnL、float32 奖励序列、持仓市场分钟相同。原账户完整状态、因果观察、行情、特征、state_path 哨兵文件均不改变。DEV 调用只做离线诊断，无 action label。

## Root cause

没有观察到附件生产代码缺陷。旧 Strict Oracle 的 30 万元下界是保留 TP 路径的资金证明边界；它与当前原 Env 的真实开仓合法性、单笔成交后果是不同合同。真实亏损账户在下界外仍可合法成交及评分，附件正确保留 `strict_decision_valid=false`、`quality_valid=true`。旧 Oracle 的 `q_add/q_wait` 仍为 null，未补造数值 Q。

## Changes made

仅新增附件的 score、posthoc teacher 和原测试；额外增加 7 项验收边界测试与一个 CPU 验收工具。现有 Env、Reward、Oracle、生产入口、Replay、Checkpoint 和 Dreamer 文件没有改变。

补充测试覆盖：chunk continuation、零时长元数据不计市场分钟、观察预算导致右删失、夜盘到次日上午仍为同官方日、下午到夜盘跨官方日、MAE 单列不扣分、MAIN_ROLL、MARGIN_CALL、真实执行拒单、缺官方日期和 E profile 拒绝认证。测试夹具曾因没有提供官方日期/345 分钟时钟被原 Env 拒绝；修正夹具后上述 7 项通过，未修改生产代码来放宽时钟。

## Verification performed

```bash
env PYTHONPATH=src:../dreamerv3:tests/entry_v1 JAX_PLATFORMS=cpu \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/entry_v1/test_fg_s3_quality_teacher.py \
  tests/entry_v1/test_fg_s3_quality_patch_acceptance.py \
  tests/entry_v1/test_fg_single3_env.py tests/entry_v1/test_fg_single3_oracle.py

env PYTHONPATH=src:../dreamerv3 JAX_PLATFORMS=cpu \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -B \
  tools/fg_s3_quality_patch_acceptance.py \
  --env-receipt /data/logdir/fg_s3_market_minute/20261013_v2/env/v1 \
  --oracle-receipt /data/logdir/fg_s3_market_minute/20261013_v2/oracle/v1 \
  --output <新的隔离输出目录>
```

工具拒绝复用输出目录。实际执行版本一并保存：附件基线为 `acceptance_harness_executed_original.py`；集成最终验收为 `acceptance_harness_final.py`，对应各收据中的工具 SHA256。此前集成试跑期间更新过工具的输出目录失败保护，因此那次中间收据不作为最终源码身份认证；最终验收使用稳定源码重新运行并保存于 `fg201_real_account_final/`。

## Risks / not verified

真实账户对账范围是 FG201 上述 5 个快照，未声称覆盖全部 18 个 TRAIN 合约或极端低保证金账户。原费用为冻结研究配置，日历为 user_confirmed_observed_clock；未追加交易所历史费用认证。OTHER/追保/右删失边界由可控原 Env 夹具验证。E fork 按附件明确拒绝。未训练模型、未证明预测能力、未准入 Actor 影响、未重算新 Reward Oracle、未开启 GPU/10K/1M。

## Next recommended step

用户已确认采用本 ZIP 的 Quality-first 语义：Catalog 使用 `B(t,d)`，MAE 单列；保留原 Full 网络作为后续受控实验，默认只训练和评价辅助头。通过预测评价门槛前不把质量预测接到 Actor 决策。Catalog 的 TRAIN 身份、快照/stepid 联结及泄漏检查须另行验收。
