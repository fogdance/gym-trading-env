# E-Data V1 → E-Model V1 的独立消费器评审

原 Data V1 数学、已冻结 Schema、原18维、原 TRAIN 产物不变。

本次只新增运行时 `ETrainProvider`，按**当前每一分钟实际 Env 决策截止**调用原 `MarketData.snapshot`。不能把原 stride15 的32,564个研究快照当成完整训练Replay输入。真实 collection 必须逐文件验证，原TRAIN source pool SHA、schema、原始OHLCVOI、旧18维逐步一致；任何错位拒收，不生成伪数据、不从未来补齐。

- `e_data_v1/provider.py`：完全独立 CPU 环境，可按全合约真实行情预建已验证 `MarketData`。
- `src/gym_trading_env/research/fg_s3_e_provider.py`：仅重导入上述类，**没有复制数学实现**。
- `tests/e_data_v1/test_runtime_provider.py`：纯数据 fixture 对0/14/15/40/359/600分钟作观察一致性/因果和错源拒绝测试。
- Gym S3-V2正式准备要求显式 `e_data_source_pool` 并在 E 实验验证 source pool SHA，防止误读另外一个TRAIN pool。

这份新增 Provider 不是正式 E-Model 解锁：受控CPU环境不能实例化原Gym/Replay/Model，生产M4必须在完整用户机器上做真正的全链测试；请对照 Model 文档 `docs/S3_E_DATA_MODEL_M4_REVIEW_20261015.md` 进行最终审查。
