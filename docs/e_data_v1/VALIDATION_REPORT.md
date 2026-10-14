# S3-E Data V1 验收报告

日期：2026-10-15。技术状态：**E_DATA_V1_READY_FOR_INTEGRATION**；用户 Review 仍待进行，本轮停在独立数据组件。未开发 E-Model、Context MLP，未合并 integration，未启动 GPU 或正式训练。

## 实现与身份

开发目录：`/home/v/Documents/work/research/fg-s3-e-data-dev/`。Gym 分支 `feat/s3-e-data-v1`；最终全量验收使用已提交、干净的 Gym `591ed63d9ee08389656e2fd357fced3b07c686a9`，Dreamer 为只读 detached `d41bc80ba13b3dc3f5d2c32fcdc800613b2d0b4d`、干净。本报告及证据在该代码提交之后另行提交；最终文档提交可用本 worktree 的 `git rev-parse HEAD` 查询。

创建时 Gym integration 基线为 `03a625c1661546b9183b9ceeb24119b516a87459`。验收时 integration 已前进到 Gym `cc6ba78f3a2016eb280135f7730d0a1f68d4512e`、Dreamer `eb63de710fd693fad7ac76f797f3d9f39e331b98`，后续增加 Composition F0 与 Tutor 正式配置。逐字节核查八份原行情 helper 与当前 Gym integration 相同；本模块保持原创建基线，不主动集成模型或新配置。完整核对见 [source_review.json](evidence/source_review.json)。

任务书原文件 SHA256：`edb41c72871a103358e47e80df8ef817a1e1825c01990b4e1c22b62fa164eaab`。docs 保留原文，包括原 Markdown 的两个空格换行；实施文件的 diff whitespace 检查通过，原文中这些既有空格不作为代码缺陷。

Schema SHA256：`35648d58089a51198a1fe269600d2efb6219ed718567a21d6f091d4c1bf9ccd1`。CPU Python 3.11.11；固定依赖 NumPy 1.26.4、pandas 2.2.3、PyYAML 6.0.2、pytest 9.1.0，完整依赖列表在 source review。系统 python3 是3.10，README 重建命令使用已核实的3.11解释器只读创建独立 venv。

模块入口为 `e_data_v1/core.py`，CLI 为 `python -m e_data_v1 inspect/build/verify`。未修改任何原有生产文件；复用原特征计算器、MarketClock 和稳定文件身份 helper，具体方式见 [一分钟优化复用审查](ONE_MINUTE_COMPUTATION_REUSE.md)。

## CPU 与数学/因果验证

`tests/e_data_v1`：**22 passed，0 errors，0 failures，0 skipped**，7.788秒。[JUnit](evidence/f01_f11.junit.xml)。既有 `tests/entry_v1/test_entry_market_cache.py`：**6 passed，0 skipped**，5.600秒。[JUnit](evidence/original_cache.junit.xml)。最终22项使用 `/tmp` pytest 缓存，无缓存警告；早期测试和既有缓存回归出现仅限本 E worktree 的缓存只读警告，不影响断言。

| 测试 | 实际证据 | 结果 |
|---|---|---|
| F01 共同坐标 | 原价重建、近高点 log 差、1m审计与15m共同锚点；所有真实 bundle 的价格列公式复核 | PASS |
| F02 原18维 | shape/顺序/mask/provenance/输入不变；受控原Env8个观察及真实FG809五个观察的字节哈希相同；错误原观察拒绝 | PASS |
| F03 因果 | 历史前缀/全量/篡改所有未来 OHLCVOI 的数组和逐时点元数据完全相同；额外无效未来与延迟历史回归；18真实合约各3个历史点，共54点 | PASS |
| F04 桶边界 | 第14/15/16分钟桶数0/1/1；真实Env顺序 WAIT 核对 cutoff；延迟到达前不可用 | PASS |
| F05 时段/合约 | 夜盘跨自然日、休息、周末/合成假日 dated calendar、不跨真实合约；连续主力拒绝、换月历史重置 | PASS |
| F06 聚合 | 独立 pandas 聚合核对全部32,500个真实完整桶的六项 OHLCVOI；缺分钟/重复/冲突拒收，合法零成交计数 | PASS |
| F07 量仓时序 | 基准是此前20个合规完整桶、OI末值与前桶比值；乱序重排相同、非正OI/非法OHLC拒绝；TRAIN跳变统计 | PASS |
| F08 shape/mask | 24×7、60×18、float32 finite、旧到新、左填零、warmup整行零 | PASS |
| F09 跨日换锚 | 旧18维保持原值，六个影子通道统一决策锚点；零时长开盘不能偷用未来首Close；真实差异分布如下 | PASS |
| F10 来源 | Teacher/Oracle额外列在入口被丢弃；原观察provenance限制为身份字符串/字符串字段，不接受Teacher载荷；主数据模块不加载Torch/JAX/W&B/DB | PASS |
| F11 隔离/校验 | 正式路径与已有输出拒写、数组篡改失败、collection诊断/manifest校验；11个受保护仓库与6,793个关键文件前后身份一致 | PASS |

真实 FG809 原 Env 核查 raw cursor `[0,14,15,359,28263]`；五个 `market_seq` 哈希均逐字节相同，observation identity 为 `a4a205df98c611f282b782b0355729bf35a347afb325c75d9b008d0a2fa413d6`。完整哈希、54个时点和每合约聚合数见 [real_validation.json](evidence/real_validation.json)。

真实数据入口为冻结 B2-Full prepared 的 `train_pool.json`，SHA256 `4d9aa8faa5c2ad22a5664280675067c5c18e9fb22e3fee7b9af3bf0f15e6281b`。只读使用 TRAIN，不读取 DEV/TEST评价，不重写行情或 prepared。日期日历采用原合约配置及 source SHA 关联；没有另行认证交易所公告或自行推算未来假期。

## TRAIN 数值与冻结参数

全量18合约，488,160源行；其中488,154合法一分钟、6个零时长开盘标记。32,500个合规15m桶；前20个量基准 warmup 共360个，占1.1077%；特征有效桶32,140。共导出32,564个采样决策，stride=15并包含尾部时点。源价格范围1177–3162；已导出有效上下文价格 log 坐标范围约−0.09323–0.11363。

53个拒收桶：48个有数据但缺预期槽、5个空预期槽，共654个已观察合法一分钟未进入15m聚合。原一分钟窗口仍保留这些合法行。没有无效OHLC/OI分钟，没有非正OI或零成交分钟；这不改变模拟测试中的合法零成交支持。6次 `abs(log(OI/OI_prev))>0.1` 只列入诊断，没有偷偷删除。

| 量仓值 | n | min | P01 | P50 | P99 | max | clip / 饱和率 |
|---|---:|---:|---:|---:|---:|---:|---|
| vol_relative raw | 32,140 | −5.056406 | −1.675753 | −0.253667 | 1.540668 | 3.020142 | [-6,6] / 0% |
| oi_change raw | 32,482 | −0.685544 | −0.025177 | −0.000225 | 0.026041 | 0.129726 | [-1,1] / 0% |

依据用户已经确认的“TRAIN通过后冻结”条件，本轮冻结量仓区间。不是最优性声明，不使用DEV/TEST调参。[可直接供 CLI 使用的配置](evidence/frozen_config.json)。OI统计从第二个合规桶开始，volume统计须有20个以前桶，所以样本数不同。

六个参考通道的真实数值差异如下。计数为多次决策窗口中的有效行元素，包含重复观察，不是独立交易或唯一分钟数；有效窗口行元素总数1,951,212。

| 影子相对旧18维 | changed values | 最大绝对差 |
|---|---:|---:|
| obs_cumVWAP_t | 201,894 | 0.0839715721 |
| obs_session_high_t | 201,894 | 0.0839715712 |
| obs_session_low_t | 201,894 | 0.0839715712 |
| obs_pct_chg_from_ref_t | 201,894 | 0.0839715721 |
| obs_range_t | 201,894 | 0.0081474707 |
| obs_dC_minus_cumVWAP_t | 201,894 | 0.0050522201 |

另外12个通道 changed values=0、最大差=0。各合约的 P01/P50/P99、量仓/OI原值、拒收示例和完整差异统计见 [train_diagnostics.json](evidence/train_diagnostics.json)。原18维跨交易日参考坐标不统一的事实仍存在；候选影子解决新E输入坐标，后续模型输入选择及 warm-start 行为需要单独 Review。

## 性能与隔离

最终全量build包含原1m计算、32,564个快照、约1.32GB产物写入及逐bundle验证：内部37.486秒，外部wall38.52秒，15个合约worker、每worker原生线程1、CPU约1102%，最大单进程RSS438,200KiB。[资源记录](evidence/build_resource.txt)。全量独立数学/因果/原Env/性能核查的实际耗时见 [资源记录](evidence/crosscheck_resource.txt)。多进程RSS不是全部worker相加的整机峰值。

同一FG809、2,000个决策，原E逐步查询方式与复用原 MarketClock/预计算数组的 snapshot 循环：13.924秒→2.152秒，约6.47倍。全部数值数组及原有逐时点元数据一致，数值SHA为 `0202d8106a3a8d6062955a6631abd75301bae4e79cf9076a52e24e7d897144f2`；新实现额外包含逐决策基准 provenance。对照仅覆盖 snapshot 循环，不代表整体训练或GPU加速。保存的旧E对照快照是证据文本，[SHA为857176…](evidence/snapshot_before_reuse.py.txt)，不进入运行接口。

受保护范围包含B2/Tutor冻结训练双仓、800K双仓、Archive/W&B源码和E只读Dreamer；源TRAIN数据、配置、原Python解释器、800K归档manifest/SHA清单/Checkpoint manifest。前后6,793项大小/SHA与双仓身份一致。[before](evidence/isolation_before.json)、[after](evidence/isolation_after.json)。不递归重哈希正在训练的Replay、日志或整个Checkpoint字节，不能把其自然更新错误计入数据任务。正式Python没有安装或更新包；隔离CPU引用使用禁止bytecode与单独Matplotlib缓存。

## 实际产物与复核

完整示例产物：`/home/v/Documents/work/research/fg-s3-e-data-dev/e_data_v1/train_build_20261015_final/`，200个文件、1,323,786,656字节。每合约10个payload文件及manifest，根目录 diagnostics/index。全collection校验18个bundle均PASS；index identity `44798d05e2ff3b91c586dece6e089e7452928d1a4752c36d082748b53895ac43`。[校验记录](evidence/collection_verification.json)。

独立实际 inspect：`../e_data_v1/train_inspect_20261015/`，FG809全部桶诊断和5个采样决策通过；尝试重用build输出目录返回exit1、`Output already exists`，没有覆盖产物。旧pilot和首轮build保留为历史原型，不能代替最终schema产物。

自检复现了首轮接口可将 `original_provenance={'Teacher_score':100}` 复制进审计产物。修复为明确身份字段白名单，增加F10拒收回归，并在新目录重新执行最终验证；旧产物不删除、不覆盖。最终与首轮18合约共90个数值NPY文件的SHA一致，修复只改变身份载荷校验及其实现hash，没有改变行情数组。

```bash
cd /home/v/Documents/work/research/fg-s3-e-data-dev/gym-trading-env
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ../.venv-e-data/bin/python -m e_data_v1 verify --bundle ../e_data_v1/train_build_20261015_final
(cd docs/e_data_v1 && sha256sum -c SHA256SUMS)
```

第二条命令校验报告和证据文件；第一条核查整个真实bundle目录。完整重跑命令和调用例见 [README](README.md) 与 `examples/e_data_v1_consumer.py`。本轮没有E网络、生产Replay接线、GPU/训练吞吐或交易效果验收，后续集成仍须新任务授权和Review。机器可读验收见 [acceptance.json](evidence/acceptance.json)。
