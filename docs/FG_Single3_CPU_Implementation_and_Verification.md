# FG Single3 CPU 实施与验收汇总

状态：IN_PROGRESS，以下缺失Gate仍未完成。
模型NOT_RUN；gradient_updates=0；择时能力NOT_ESTABLISHED。

报告生成UTC：2026-10-11T05:19:21.657582+00:00。

正式依据：`FG_DreamerV3_Single3_TP5_SL30_Reward_OracleReplay_Execution_Plan_V1.md`；原文件SHA256 `9e3287a5bbe9eec93788275ae5b3ee082677c637b266690bad08732823679744`。业务协议和实施范围见 `FG_Single3_Implementation_Plan_and_Progress.md`。

| stage | actual status | signed receipt content hash |
| --- | --- | --- |
| E | PASS | 06ca881053ada31f064a7c488f3140dfdcb51c0a8cb1bf95f1fe1d07ee03ad89 |
| R | PASS | 8d1203757d3eede1e8785e79c11b4288990582d80d69d23cfb1b35e726e907c4 |
| Input | PASS | 4c44ee52e79c44699c2acf9091b1bebb6608856175414cb584588d320572129b |
| O | PASS | f2a8497913568d9fedec1c2ad568facc32905a948a29d65ca3b58309b3e8e678 |
| D generation | NOT_COMPLETED | - |
| D coverage | NOT_COMPLETED | - |
| H | NOT_COMPLETED | - |
| Pilot preparation | NOT_COMPLETED | - |

输入源为冻结18合约TRAIN白名单；保留旧6手Replay、Checkpoint和实验结果。原账户/撮合、Driver/Replay、Uniform采样和Dreamer网络继续复用。所有新价格轨迹来自Single3真实Env；同一自然策略四Reward视图冻结动作后分别执行Env，不把旧6手反馈重新标记为新数据。

实际确认并修复P3跨CHUNK_END的等待计时重置；原Reward R01收据撤销为失败证据，R02重新采集验收。最终验证失败时正式完成收据/PASS报告不发布；失败资料保留。

## 实际测试及命令证据

- E：`/home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -p no:cacheprovider tests/entry_v1/test_fg_single3_env.py tests/entry_v1/test_entry_contract.py -q --tb=short`；returncode=0；输出 `............................................                             [100%]`。
- R：`/home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -p no:cacheprovider tests/entry_v1/test_fg_single3_replay.py tests/entry_v1/test_fg_replay.py -q --tb=short`；returncode=0；输出 `...............                                                          [100%]`。
- O：`/home/v/miniconda3/envs/dreamerv3/bin/python -m pytest -p no:cacheprovider tests/entry_v1/test_fg_single3_oracle.py -q --tb=short`；returncode=0；输出 `.....                                                                    [100%]`。

## 真实产物与范围

18合约198个实际截点，60×18观察与当时行情前缀精确float32相同；随机seed202、日边界及chunk边界单列。这是抽样因果证据，未来Oracle标签只在Sidecar，不在模型张量。

Reward候选未选定，不按P4高分选参数；保持horizon333/imag15/原网络作为第一组基线。当前工程证据不证明Agent学会WAIT/ADD或快速TP。

用户确认Reward、完整Replay覆盖和Oracle验收后，才可授权一组Reward的M0/M1同预算先导。先导结果仍需独立评估；本次没有访问验证/封存Test价格。
