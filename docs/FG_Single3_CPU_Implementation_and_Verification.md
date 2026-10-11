# FG Single3 CPU 实施与验收汇总

状态：IN_PROGRESS，以下缺失Gate仍未完成。
模型NOT_RUN；gradient_updates=0；择时能力NOT_ESTABLISHED。

报告生成UTC：2026-10-11T07:02:42.706210+00:00。

正式依据：`FG_DreamerV3_Single3_TP5_SL30_Reward_OracleReplay_Execution_Plan_V1.md`；原文件SHA256 `9e3287a5bbe9eec93788275ae5b3ee082677c637b266690bad08732823679744`。业务协议和实施范围见 `FG_Single3_Implementation_Plan_and_Progress.md`。

| stage | actual status | signed receipt content hash |
| --- | --- | --- |
| E | PASS | 06ca881053ada31f064a7c488f3140dfdcb51c0a8cb1bf95f1fe1d07ee03ad89 |
| R | PASS | 8d1203757d3eede1e8785e79c11b4288990582d80d69d23cfb1b35e726e907c4 |
| Input | PASS | 4c44ee52e79c44699c2acf9091b1bebb6608856175414cb584588d320572129b |
| O | PASS | f2a8497913568d9fedec1c2ad568facc32905a948a29d65ca3b58309b3e8e678 |
| D generation | PASS | 2cb543e5167a4c828fb05b67cdca9fda3c0cabdc7961fc28fc7ad8d433aba123 |
| D coverage | PASS | a1cbec5f456a3b3629d36e55f0aa20ebd005053923b29e0630741eb8bdb9c0b9 |
| H | NOT_COMPLETED | - |
| Pilot preparation | NOT_COMPLETED | - |

输入源为冻结18合约TRAIN白名单；保留旧6手Replay、Checkpoint和实验结果。原账户/撮合、Driver/Replay、Uniform采样和Dreamer网络继续复用。所有新价格轨迹来自Single3真实Env；同一自然策略四Reward视图冻结动作后分别执行Env，不把旧6手反馈重新标记为新数据。

实际确认并修复P3跨CHUNK_END的等待计时重置；原Reward R01收据撤销为失败证据，R02重新采集验收。最终验证失败时正式完成收据/PASS报告不发布；失败资料保留。

## 实际测试及命令证据

- E：`/home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -p no:cacheprovider tests/entry_v1/test_fg_single3_env.py tests/entry_v1/test_entry_contract.py -q --tb=short`；returncode=0；输出 `............................................                             [100%]`。
- R：`/home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -p no:cacheprovider tests/entry_v1/test_fg_single3_replay.py tests/entry_v1/test_fg_replay.py -q --tb=short`；returncode=0；输出 `...............                                                          [100%]`。
- O：`/home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -p no:cacheprovider tests/entry_v1/test_fg_single3_oracle.py -q --tb=short`；returncode=0；输出 `.....                                                                    [100%]`。
- D generation：`/home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -p no:cacheprovider tests/entry_v1/test_fg_single3_replay.py tests/entry_v1/test_fg_single3_coverage.py -q --tb=short`；returncode=0；输出 `....                                                                     [100%]`。
- D coverage：`/home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -p no:cacheprovider tests/entry_v1/test_fg_single3_replay.py tests/entry_v1/test_fg_single3_coverage.py -q --tb=short`；returncode=0；输出 `....                                                                     [100%]`。

## 真实产物与范围

18合约198个实际截点，60×18观察与当时行情前缀精确float32相同；随机seed202、日边界及chunk边界单列。这是抽样因果证据，未来Oracle标签只在Sidecar，不在模型张量。

完整新账户378条；29个单Reward Replay池；P0独立。原生chunk、执行源码、输入及逐行对账身份见 `/home/v/Documents/work/gym-trading-env/artifacts/fg_single3_v1/replay/run_20261011_01/manifest.json` 和 `inputs.json`。

实际Replay行10261209、推进10250982；访问唯一市场分钟488136；29池各1000真实B16×65批次。非法ADD=0、6手决策=0。详见 `FG_Single3_Replay_R0_R2_Coverage.md`。

## CPU实测结果的解读

四候选的P1/P2a/P2b/P3共16组完整TRAIN自然策略，未折扣Reward均为负；P0为0。这是需要在先导中检查All-WAIT偏好的数据证据，不证明已训练模型会选择All-WAIT，也不能用P4后验高分反选Reward。

| 自然策略 | 完整TRAIN未折扣raw Reward |
| --- | --- |
| C0_P1 | -881.193493 |
| C0_P2a | -600.68874 |
| C0_P2b | -734.626974 |
| C0_P3 | -638.77413 |
| C1_P1 | -1778.093493 |
| C1_P2a | -1099.68874 |
| C1_P2b | -1524.826974 |
| C1_P3 | -1005.77413 |
| C2_P1 | -2674.993493 |
| C2_P2a | -1598.68874 |
| C2_P2b | -2315.026974 |
| C2_P3 | -1372.77413 |
| C3_P1 | -6899.033493 |
| C3_P2a | -4590.73874 |
| C3_P2b | -6188.166974 |
| C3_P3 | -3965.22413 |

C2只是采集诊断配置。以下为真实1000批次train64位置的曝光，不包含context1；加入P4后的曝光变化不是模型收益或择时能力。

| 池 | TP/batch均值 | 零TP batch比例 | 合法ADD决策/batch均值 | SL/batch均值 |
| --- | --- | --- | --- | --- |
| C2_natural | 9.408 | 0.0 | 75.251 | 2.33 |
| C2_natural_oracle | 14.519 | 0.0 | 227.763 | 1.886 |

本轮真实TRAIN的ADD拒单=0、SL pending观察行=0、终局未平仓=0。未出现的风险分支只有构造Env测试证据，不能声称模型已经覆盖这些状态。

推进分钟统计不含18个合约的初始观察；unvisited_rows=0。多策略及四Reward视图仍是同一历史市场的相关反事实。

Reward候选未选定，不按P4高分选参数；保持horizon333/imag15/原网络作为第一组基线。当前工程证据不证明Agent学会WAIT/ADD或快速TP。

用户确认Reward、完整Replay覆盖和Oracle验收后，才可授权一组Reward的M0/M1同预算先导。先导结果仍需独立评估；本次没有访问验证/封存Test价格。
