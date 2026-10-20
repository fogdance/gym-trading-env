# 独立 E-Data 工具的依赖导入修复

状态：**FIXED；定向回归 PASS**。仅修改 `feat/data-perf-20261019`，未合并到 `integration/s3-next`。

## 观察与已确认根因

扩展回归 `test_F10_only_known_market_inputs_and_no_GPU_imports` 失败。原始扩展集合为 53 PASS / 1 FAIL；在未修改的 Gym `9ab7ff7d18a939e48bedb134979e664ec967a087` 上单独重跑，同一断言仍失败。两份原始失败收据均保留。

运行时 import 审计确认，`EDataConfig.schema()` 为读取 `CONTRACT`，直接导入 `gym_trading_env.utils.macd_5m_causal`。Python 先执行 Gym 包和 `utils/__init__.py`，后者导入 `data_downloader → yfinance → cache → peewee → pymysql`。因此仅构建独立研究数据也会加载 `gym_trading_env` 和 `pymysql`。`MarketData.macd_snapshot()` 存在同一直接导入路径。

这是实际依赖泄漏；不能通过删除或放宽依赖隔离断言修复。

## 修改

- 将原生纯计算文件 `utils/macd_5m_causal.py` 纳入现有 `e_data_v1.reference` 审计加载清单；复用同一份源码和算法，通过现有私有命名空间加载，避免执行 Gym 包初始化。
- Schema 和 MACD 快照均使用 `legacy_helper`，并遵循 `EDataConfig.legacy_root`。Schema 保留实际 MACD 源码 SHA，审计清单也包含该文件。
- 扩展既有独立进程回归，同时构建 bundle 和调用 MACD 快照，检查算法合同一致，并保留全部五项禁止导入断言；失败时输出具体模块名。

MACD 实现文件未修改，SHA256 为 `94d8dff4fa8f54a91604cdd02db17ddf30c24402a9658c5d6826d23cbb0d2afe`。没有复制 EMA 或市场规则实现。

## 验证与收据

| 验证 | 结果 |
| --- | --- |
| 修复前扩展回归 | 54 collected，53 PASS / 1 FAIL |
| 未修改上一版单测 | 1 collected，1 FAIL，同一依赖断言 |
| 修复后独立进程定向回归 | 1 PASS，0 FAIL / ERROR / SKIP，1.130 秒 |
| 修复后原扩展集合 | 54 PASS，0 FAIL / ERROR / SKIP，14.999 秒 |
| 修复后 import 审计 | 五项禁止模块的交集为 `[]`；MACD 文件 SHA 与审计源 SHA 一致 |

扩展集合包含 MACD 因果性、EMA/预热、完整分钟及不满五根的认证时段尾部、跨日状态、冷恢复、独立数据合同、运行时 Provider 和原生 Env 集成。上述计数仅代表实际列出的集合，没有运行全项目测试。没有运行 GPU、模型更新或正式训练准入。

原始失败、修复后 XML、导入审计和精确命令均在 [收据目录](receipts/data_perf_dependency_import_20261020/manifest.json)。Manifest 记录实际父 SHA、Dreamer SHA、测试时 dirty 状态、测试源码哈希及每项 collected test。提交后的源码可用其中哈希逐一核对。

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 \
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/entry_v1/test_wp02_macd.py tests/entry_v1/test_macd_dated_tail.py \
  tests/e_data_v1/test_contract.py tests/e_data_v1/test_runtime_provider.py \
  tests/e_data_v1/test_native_integration.py
```

自审：实际修改三个生产文件及一个既有回归；保持依赖隔离断言和原生计算，未改 Gym 初始化、金融 Env、Loss 或训练入口。源码审计身份会随源码及加载清单改变，历史冻结批次及其收据不修改；本修复不授予旧集合新的身份。
