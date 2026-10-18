# WP03 任务书验收映射

全部对应最终选定集合177个实际testcase（0失败/0错误/0跳过），或明确的真实执行收据。缩写 A01 等为 `test_A01_...`；完整参数化用例见 [实际清单](receipts/wp03/executed_testcases.json)。真实样本没有RIGHT_CENSORED；其拒绝边界由受控负例验证。

| 编号 | 行为 | 证据 | 结果 |
| --- | --- | --- | --- |
| 1 | Corpus/Course/Obs/Reward/Model/Teacher/source一致 | A01,A09 | PASS（本包范围） |
| 2 | 修改Course/source/Chunk/MACD拒绝 | A03,B11,C08 | PASS（本包范围） |
| 3 | 非TRAIN/TEST/穿越/symlink读前拒绝 | A04,A05,A06 | PASS（本包范围） |
| 4 | 旧V1/C1/Utility不得冒名 | A09,B09,D03 | PASS（本包范围） |
| 5 | 自签PASS不能替代native G3 | A07,D11,D13；本轮D_native_G3 | PASS（本包范围） |
| 6 | 原生学生Chunk逐字节一致 | B01,B03,B11 | PASS（本包范围） |
| 7 | Sidecar与Trace gzip一致 | B03 | PASS（本包范围） |
| 8 | 六policy账户独立恢复 | B01；native_verification六个account_restores=1 | PASS（本包范围） |
| 9 | 物理uint8[20] UUID/offset链 | B01,B02[stepid] | PASS（本包范围） |
| 10 | 重复/丢失/ordinal与prefix无效 | B02[ordinal/missing/duplicate],B05 | PASS（本包范围） |
| 11 | 同cursor不同账户Owner互换拒绝 | B02[policy/account],C05 | PASS（本包范围） |
| 12 | L1/L2账户state_id绑定 | B02[shift/account] | PASS（本包范围） |
| 13 | strict值/witness/tie | B07,B14 | PASS（本包范围） |
| 14 | 双合法/forced/terminal/资金域外/删失无效 | B04,B13,C01,B14 | PASS（本包范围） |
| 15 | teacher不覆盖实际行为action | B05 | PASS（本包范围） |
| 16 | post-step/next execution错移拒绝 | B02[executed/incoming_action/shift],B05 | PASS（本包范围） |
| 17 | L1/L2独立valid | B04,B13,B14 | PASS（本包范围） |
| 18 | Quality TP Score原公式，MAE独立 | B12；原validate_corpus_label结算公式 | PASS（本包范围） |
| 19 | event/touch/adverse各窗口valid | B06,D05 | PASS（本包范围） |
| 20 | 风险/快慢TP/同跨日真实 | C02；最终真实course_support与coverage | PASS（本包范围） |
| 21 | MACD字节不变 | B11,C06 | PASS（本包范围） |
| 22 | policy输入不含后验 | D04,D08 | PASS（本包范围） |
| 23 | T+1 prefix/effective stepid | C01,D02 | PASS（本包范围） |
| 24 | ChunkEnd/reset/terminal/跨账户分离 | C01,C05,C10 | PASS（本包范围） |
| 25 | Natural原连续时间窗 | C01,C06 | PASS（本包范围） |
| 26 | Stratified合法窗口/真实题型 | C02,C03；真实10视图course_support | PASS（本包范围） |
| 27 | Mixed seed/权重/状态/来源恢复 | C02,C04,C07,C09 | PASS（本包范围） |
| 28 | strict比例/不足类不造样 | C03；真实Join和Coverage | PASS（本包范围） |
| 29 | 同Owner/reset pair；零Owner拒绝 | B08,C05 | PASS（本包范围） |
| 30 | QualityPair有效位与分差 | C05；真实batch256个quality_pair | PASS（本包范围） |
| 31 | 旧Q/Soft无效且loss禁用 | B09,D03,D06 | PASS（本包范围） |
| 32 | 无效巨值安全投影、NaN拒绝、无效梯度0 | B06,B12,D05,D07 | PASS（本包范围） |
| 33 | Fresh真实Agent CPU finite，无更新 | D01,D08；D_real_reader_final | PASS（本包范围） |
| 34 | WP01 draft/formal和公开Guard | test_fg_s3_e_quality_course_preflight整文件,D12；共享20项 | PASS（本包范围） |
| 35 | RSS/磁盘/吞吐/缓存有界 | C06；真实Coverage性能字段 | PASS（本包范围） |
| 36 | 实际有限TRAIN六账户读入 | D_native_G3,D_real_reader_final,D_actual_coverage | PASS（本包范围） |
| 37 | 三种统计分母，optimizer0 | C01,D01；真实Coverage | PASS（本包范围） |
| 38 | teacher后验改动不影响Actor logits | D04,D08；真实Loss证据 | PASS（本包范围） |

正式GPU训练、WP04–07阶段优化器/冻结/handoff、正式Prepared/Checkpoint认证、旧1M继承、全项目测试、TEST读取、18合约生产均为 **NOT_RUN / 未授权**。CPU Actor项只调用现有纯数值目标，不是F3阶段准入。
