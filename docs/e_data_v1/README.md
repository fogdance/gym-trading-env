# S3 E-Data V1

独立 CPU 数据组件，协议为 `S3_E_DATA_V1`。保留原 `market_seq[60,18]`，同时提供候选 `market_seq_e_shared[60,18]` 和已完成、实际可见的 `context_15m[24,7]`。验收状态和真实数据证据见 [VALIDATION_REPORT.md](VALIDATION_REPORT.md)；公式见 [MATH_SCHEMA.md](MATH_SCHEMA.md)，原一分钟优化的复用审查见 [ONE_MINUTE_COMPUTATION_REUSE.md](ONE_MINUTE_COMPUTATION_REUSE.md)。

源码位于独立 `feat/s3-e-data-v1` worktree。Dreamer 仅供只读核查。本组件不构造模型、不启动训练、不接入原 Env/Replay，不向 Archive 或 W&B 写入。完成后等待 Review。

## 安装与运行

从本 Gym worktree 根目录运行，Python 要求 3.11。当前独立环境已建立于相邻的 `.venv-e-data`，固定依赖见 `requirements-e-data.txt`。重新安装时使用独立 venv：

```bash
python3.11 -m venv ../.venv-e-data
../.venv-e-data/bin/python -m pip install -r requirements-e-data.txt
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

CLI 接受已有、只读、eligible TRAIN 的 FG pool manifest。它逐一校验 CSV、Env 配置、合约配置、日历和来源哈希，不创建训练 prepared/Run。每次使用新的输出目录；已有目录拒绝覆盖，正式 Run、冻结源码、Archive/W&B 工作树拒绝写入。

```bash
../.venv-e-data/bin/python -m e_data_v1 inspect --source /absolute/path/train_pool.json --out ../e_data_v1/inspect_new
../.venv-e-data/bin/python -m e_data_v1 build --source /absolute/path/train_pool.json --out ../e_data_v1/build_new --audit-1m
../.venv-e-data/bin/python -m e_data_v1 verify --bundle ../e_data_v1/build_new
../.venv-e-data/bin/python -m e_data_v1 verify --bundle ../e_data_v1/build_new/CZCE.FG809
```

`--contracts CZCE.FG809 CZCE.FG901` 选择真实合约；`--stride 15` 是离线诊断快照采样频率，桶完整性仍逐桶核查。`--stride 1` 可导出每个可见源行的快照；`--max-decisions` 限制诊断量。`--workers` 设置上限，默认按实际可用内存和 CPU 自动选择；worker 原生线程应设为 1。`--config config.json` 仅允许指定 `volume_clip` 和 `oi_clip`，变更会产生不同 schema hash。任何异常使命令失败，不写 READY 状态；调查后在新目录重试，失败目录保留证据。

`inspect` 仅导出诊断，不导出数组，其 collection 没有模型 bundle；`verify` 对 `build` 的单合约 bundle 或整个 collection 校验。`PASS_TRAIN_NUMERICS` 只代表本次数值诊断，完整可集成状态必须由 F01–F11 和真实数据验收报告共同给出。

## API 与未来消费方式

```python
from e_data_v1 import Decision, EDataConfig, MarketData

prepared = MarketData(minute_bars, schedule, EDataConfig(audit_1m=True))
bundle = prepared.build([
    Decision("CZCE.FG809", visible_cutoff, decision_time=decision_time,
             market_seq=original_env_observation,
             original_provenance={"observation_version": "...", "source_sha256": "..."})
])
e_inputs = {
    "market_seq_e_shared": bundle.market_seq_e_shared[0],
    "market_context_15m": bundle.context_15m[0],
}
```

`MarketData` 对普通 CSV 合约只准备一次原特征和索引，然后批量 snapshot；不要每步重新创建。纯函数 `build_e_data_v1(minute_bars, decisions, schedule, config)` 适合一次性批处理并在计算前截断到所请求的最晚可见时点。显式延迟到达的数据使用当时可见前缀重新计算，正确性优先于该异常路径的吞吐。

`minute_bars` 必须包含 Date/DatetimeIndex、`symbol,trading_date,bob,Open,High,Low,Close,Volume,OpenInterest`；可选 `available_at`。仅接收一个显式 `CZCE.FG` 真实合约，换月另建 prepared 对象。`schedule` 必须包含 `timezone="Asia/Shanghai"`、原 FG `segments`、`minutes_per_day=345`、事先给定的 `dated_sessions[{start,end}]`。不推断未来节假日，不以连续主力序列混合不同合约。

提供原 Env 的 `market_seq` 时要求 float32、60×18、逐字节数值一致，失败则拒绝。`original_provenance` 只供观测身份链接；调用方不得把标签或账户流水当身份传入。源码实例见 `examples/e_data_v1_consumer.py`；本例只返回数组，不调用网络。

## 产物与校验

每合约输出 `context_15m.npy`、旧 `market_seq.npy`、候选 `market_seq_e_shared.npy`、float64 `price_anchor.npy`、可选 `audit_1m_ohlc_shared.npy`，以及 `bucket_manifest.json,decision_meta.json,schema.json,provenance.json,events.json,manifest.json`。

根目录 `diagnostics.json` 汇总实际分布、warmup、拒收、量仓跳变、18维数值差异、来源和双仓身份；`index_manifest.json` 绑定诊断和每个 bundle manifest 的 SHA256。单合约 manifest 包含所有产物大小和 SHA256，verify 检查完整文件清单、数组契约、共同锚点、过去量/OI来源和不可变旧18维哈希。移动整个 collection 后仍可使用相对合约成员校验。

桶拒收包括缺槽、重复、冲突、越出日历和无效 OHLCVOI。零成交合法分钟仍计入15个槽；零时长开盘标记不计入。已完成桶不足20个先行量基准时保留其窗口位置，但整行置零且 mask=0；窗口不足24根左填零。诊断中的合法缺槽不会被修补为假行情。

## 重跑验证

```bash
../.venv-e-data/bin/python -m pytest tests/e_data_v1 -q -o cache_dir=/tmp/s3-e-data-pytest
../.venv-e-data/bin/python -m e_data_v1.validate_real --source /absolute/path/train_pool.json --out ../e_data_v1/crosscheck_new --reference-python /home/v/miniconda3/envs/dreamerv3/bin/python
```

真实验证逐一独立核对全部合规桶的 OHLCVOI，比较18个 TRAIN 合约的历史前缀/全量/未来篡改结果，并调用隔离 CPU 原 Env 核对旧观察。原 Env Python 仅用于读取既有代码，不能安装依赖到该训练环境。CPU 测试若原 Env 引用环境不可用会明确 skip；有 skip 时不得宣告对应运行时验收通过。

本轮只读核查发现 integration 后续新增了 Composition F0 和 Tutor 配置，原一分钟特征与市场缓存代码相对本模块创建基线未变。本模块仍保持独立开发，不自动合并新的模型接线或正式训练。
