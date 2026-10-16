# WP02_IMPLEMENTATION_REVIEW

## Summary

WP02 V2有限范围实施与验收完成；G1/G2/G3均GO，训练权限仍NO-GO。

范围仅为WP02 V2新课程/Observation/Replay/Encoder、P0–P4及L1/L2、有限可信TRAIN教材和只读核验。样本为FG201主力资格末两官方日（2021-12-08、2021-12-09）；没有旧ABI/旧Checkpoint兼容、旧E+Tutor 1M权重、GPU训练、TEST行情或18合约生产。用户最后指示保持当前性能实现，不再优化。

## Evidence observed

复用WP01兄弟worktrees，独立分支 `feat/s3-e-quality-wp02-v2-20261016`。真实integration仍为Gym `6939de5b43200e6f19311804ebf1739b01f12ba5`、Dreamer `51ed1d094c92fc52a910acda3e2a3421bc71c9e5`，均clean，未合并。WP01是开发树祖先，尚未集成integration；原始Git命令/退出码见 `receipts/wp02_v2/pre_release_repositories.json`。

| 阶段 | Gym完整SHA | Dreamer完整SHA |
|---|---|---|
| WP01 | `2904f3e58cd84fb383f9e5a0d2d2aa643fcb10d2` | `347724e3207fca16907facab021302217f47cf6b` |
| A | `e99bb037ac5df0535e67a22f2ebb0dc495a75b1f` | `1d592828addfb42fdedf658e3ebfd35cd67ba8f2` |
| B | `81c577dc73c92264007b042a12bf7e331fbfded2` | `c056184096958059b7758b836b2ada60472e45fa` |
| C | `48a7eba417a9a77c606e90a1d76ddcadb64242fd` | 无独立源码提交 |
| D | `70a947d03047b371f028ff0122ddd5eeb02ab751` | `44a6cc62032811501631e83be977f07137dc65b8` |
| E | 本报告所在最终提交；完整SHA见提交后`receipts/wp02_v2/release_lock.json`及交付回复 | 无新增源码提交 |

A→E为五个Gym主提交；两个独立Git仓库的关联提交另列。没有制造空提交凑数。生成时是D HEAD加未提交E源码；最终提交后核对所有运行源码字节相同，不能把生成伪称为clean E HEAD运行。

## Root cause

新研究模型需要正式升级E输入与教材身份，旧四字段E输入、旧Replay/Checkpoint和旧C1 Oracle不足以表达本次合同。性能热点由cProfile实测定位：反复把完整行情前缀转换为Python对象/JSON，以及完整账户历史的Python深复制。修正仅改变序列化、复制与独立教师查询调度；金融撮合、账户规则和原Quality TP Reward不变。

## Changes made

- A：课程V2、五字段原生E Observation、MACD(12,26,9)完整/partial桶、35个此前完整桶预热和1000*raw/可见Close；hist=DIF-Signal，第5分钟仅提交EMA一次。
- B：原Driver/Replay保存MACD真实字节，真实Dreamer E Encoder增加16维MACD分支、输入及输出valid gating；fresh Agent CPU forward和同版本完整恢复。
- C：P0/P1/P2低中概率/P3真实独立账户；L1通过独立原Env fork输出质量/时间/风险事实，MAE不改变Score或Reward。
- D：原CandidateIndex/OracleStateQuery结构重新建立Quality目标完整后缀、ADD/WAIT两动作值与rational witness、独立资金域证明、L2和P4。旧C1 schedule不能冒名。
- E：受限validate/generate-small/verify/coverage CLI、原可信TRAIN锚与来源/代码/模型合同、独立student Replay与teacher sidecar、完整重播和JSON/CSV覆盖报告。
- 性能：原JSON字节兼容的行情前缀缓存，逐次检查行情是否改变；JSON快照独立复制；教师有界进程池只传历史增量，worker重建全部历史后经原Env恢复/来源/Mask检查。Driver仍顺序推进账户，全部异步结果核对后才可PASS。

[源码与反向Review](WP02_SOURCE_REVIEW.md)、[40组验收映射](WP02_ACCEPTANCE_MATRIX.md)、[真实教材覆盖与完整账户证据](WP02_REAL_TRAIN_COVERAGE.md)供逐项核对。原金融Env另外25个方法AST未变，13份受保护源码逐字节不变（Reward、Score、市场特征、旧Oracle/行为、统一入口、Gate、SourceAdmission、FullCheckpoint、原Driver/Replay与训练入口）。

## Verification performed

下列集合有重叠，不相加冒充全项目测试。准确argv/cwd/CPU环境/退出码、原始log/XML及实际testcase均保存在 `receipts/wp02_v2/`。

| 运行证据 | 实际结果 | 退出码 |
|---|---|---:|
| stage_A_final | 127 passed | 0 |
| stage_B_acceptance / shared | 12 passed,2 deselected / 20 passed | 0 / 0 |
| stage_C_acceptance | 33 passed | 0 |
| stage_D_acceptance | 124 passed | 0 |
| stage_E_release_suite（优化前源码） | 306 passed,0 skipped；不冒充新源码结果 | 0 |
| perf_snapshot_regression | 97 passed | 0 |
| perf_parallel_unit | 5 passed | 0 |
| perf_parallel_acceptance | 114 passed,9 setup errors（旧测试默认Dreamer路径） | 1 |
| perf_legacy_replay_correct_root | 13 passed；包含上述9个用例，明确指定当前兄弟Dreamer树 | 0 |
| stage_E_release_perf_final（当前冻结源码） | 312 passed, 0 failed, 0 errors, 0 skipped | 0 |
| real_generate_small_04 | 六条各691行/691标签、各1次完整恢复，全部PASS；2857.562秒 | 0 |
| real_coverage_small_04 | JSON/CSV与生成结果一致；3.659秒 | 0 |
| real_verify_small_04 | 独立重建原生WAIT前缀/严格后缀、六账户逐行重播及全部标签核对，PASS；1674.025秒 | 0 |

真实命令（从本Gym worktree执行，工具固定CPU及原Dreamer Python环境）：

```bash
python3 tools/wp02_verify.py real_generate_small_04 --cmd tools/fg_s3_e_quality_corpus.py generate-small \
  --source-pool /home/v/Documents/work/research/fg-s3-b2-full-train/gym-trading-env/artifacts/fg_s3_v2/prepared/B2_FULL_MARKET_MINUTE_FORMAL_1M_v1_20261014/train_pool.json \
  --env-receipt /data/logdir/fg_s3_market_minute/20261013_v2/env/v1 \
  --contract CZCE.FG201 --out artifacts/fg_s3_v2/e_quality_wp02/real_small_04
python3 tools/wp02_verify.py real_verify_small_04 --cmd tools/fg_s3_e_quality_corpus.py verify \
  --corpus artifacts/fg_s3_v2/e_quality_wp02/real_small_04
python3 tools/wp02_verify.py real_coverage_small_04 --cmd tools/fg_s3_e_quality_corpus.py coverage \
  --corpus artifacts/fg_s3_v2/e_quality_wp02/real_small_04
env FG_REPLAY_DREAMER_ROOT=/home/v/Documents/work/research/fg-s3-e-quality-wp01-20261016/dreamerv3 \
  python3 tools/wp02_verify.py perf_legacy_replay_correct_root tests/entry_v1/test_fg_replay.py
```

完整测试路径与真实绝对out/corpus参数以对应JSON原件为准；上方相对产物路径是同工作目录的等价写法。

负例覆盖非TRAIN、自签假来源、坏源字节/身份、路径穿越及symlink、未来字段混入、重签伪造事实（串行及并发）、账本改写、不同账户混状态、旧C1/旧Reward/旧Obs冒名、低资金/删失伪WAIT、行情缓存热后篡改、完整历史和Mask不符。显式合法draft才退出0且仅CONTRACT_ONLY/NOT_ADMITTED；formal、空模式、缺必要身份均非零，断言输出文件和禁止副作用。

四个公开prepare/start/resume/evaluate命令实际均exit1，保留`S3_QUALITY_E_NOT_IMPLEMENTED`，受保护产物集合不变。没有绕过E+Quality/Q/Prediction/Decision准入。

失败记录保留：旧历史迁移断言固定两端Git SHA后修正通过（未改生产SourceAdmission）；早期命令测试路径错误；合成日历/时区构造错误。性能回归中9个setup error确认是`FG_REPLAY_DREAMER_ROOT`仍默认旧工作树，指定实际兄弟树后13项全部通过，没有放宽checkout一致性检查。real_small_01与03按用户授权SIGTERM/exit−15保留；real_small_02因运行源码变化exit2/NO_GO保留，均不作为最终真实验收。

完整暂存区`git diff --check`退出2，位置仅为原样保留的pytest日志/XML空白及原生CSV的CRLF；可编辑源码和报告Markdown的同一检查退出0。没有为了格式检查改写原始收据，详见`diff_check_results.json`。

## 真实TRAIN小样与合同身份

来源仅命中既有可信TRAIN pool、原Env收据和原官方交易日元数据；partition中DEV/TEST仅用于边界元数据核对，未读取其行情。原CSV31260行/3640878字节；样本690个市场位置/689次推进/2官方日；此前完整TRAIN用于原生因果预热。末端为认证主力资格退出，不是任意截断或交割到期。

实际119笔交易=103 TP+12 SL+4 MAIN_ROLL；1551个L1候选事实与账户实际交易分开。存在正/负/连续中间质量、早期逆势后仍TP、跨日TP、主动/强制WAIT；真实右删失为0，删失由合成边界验证，不伪造自然比例。93个根资金认证范围之外的L2保持无效，这93个状态的动作认证仍为NO-GO；没有将它们的后缀证明写成通过。G1/G2/G3结论仅说明来源/完整重播/有效位和已证明域（包括全部P4合法决策）通过，不表示任意资金账户的L2已被认证。P4的54笔TP及全部689步计划feedback、结算、费用、最终账户真实一致。

| 合同/数据身份 | Hash |
|---|---|
| TRAIN CSV | `ebad6ed8b214e0a5801b0c16bb7e4784a6cd00ba943bf0c76cc3a571ed0a2e8c` |
| Source lock | `ba359d42ace3439aab97c848dc7781cac635e2612b8aeb24525fce10099a017c` |
| E profile | `8bf93eb5629ddf7cf3992b9547cbc437bed2a8fcfeef74d71fdc81d777919f22` |
| Observation | `5ca333e52049bf5bdfe0bf18f5519e65766363c946648bdd1f45b525558052b8` |
| Model | `7152f9de6de4885b0e64c77f003c7a646a9807364c4019034ff73aa2d76cad1b` |
| Course | `14cadcdaaa7d96bc0f33f491ef38741494957ee8e7423a1ba5bf598fe230baba` |
| Strict teacher | `092d822e5975fdece9f181bb7bf27447a0d2e1923aa93d119b01ef869e56bad6` |
| Corpus | `640865b4239fb483e5d98daf18d3104e670ff9b3c1486cb63ec9924cf6e089a9` |
| Execution source | `fc44122394430de5e63aea9febace0654b19f32a94c7a19d835a6c55cfcee383` |

## Risks / not verified

- **G1 GO**：新课程/原生MACD→Replay→真实Encoder、P0–P4/L1/L2、完整原生重播与严格后缀证明通过。
- **G2 GO**：自然正/负/中间及风险/有效位/三分母完整审计；真实删失为0，不额外造样本。
- **G3 GO（仅本有限可信TRAIN范围）**：来源、官方日、原Quality目标、账户/Origin-Target、严格后缀、独立完整verify及Corpus身份全部通过。
- **训练权限NO-GO**：正式课程Gate未开放；Corpus不是Prepared/Checkpoint认证或开始训练的许可。
- GPU/正式训练、TEST行情、旧权重/旧ABI迁移、18合约全量生产、W&B、WP03训练Join、WP04–WP07课程训练、模型研究效果：**NOT_RUN**。
- 后缀证据是原索引/支配证明、随机与风险分层原生校验及所有选定计划逐步重播；**未把每个真实候选全部慢速穷举**。
- 生成时13个进程，5秒采样约1044% CPU/PSS6.83GB，未持续占满24逻辑线程。用户最后要求保持此版，未继续优化。全量性能尚未认证，不能线性外推一年或若干小时。
- E新数学/Obs/Replay/Checkpoint不支持旧模型迁移；旧Run保留原冻结源码和工件，不伪装同Run resume。

## Next recommended step

在此冻结版本审阅WP02报告和收据。后续按单独任务进入WP03；任何训练、18合约生产或新优化均需后续明确范围。
