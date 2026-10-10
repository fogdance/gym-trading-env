# FG 主力数据与统一配置验收

唯一人工维护配置为 `configs/fg_entry_v1/fg.yaml`。原 fg.txt 的 26 个方向区间与 1.yaml 的玻璃参数已经迁入；原文件保持原样，仅作为迁移来源，不参与运行。生成的合约配置和环境配置是带哈希的运行快照。

## 已确认规则

- 仅 FG，按 fut_continuous_map_v2 的 CZCE.FG / JUEJIN / trading_date 关联真实主力分钟行情，保留日夜盘、合约代码、bob 与交易日；不复权。
- 导出截止 2025-12-31，交易日和分钟时间均受限。按用户要求排除 CZCE.FG805；只删除本次生成的对应数据，数据库和原始人工文件未修改。
- 有效数据 24 段、643929 根，2018-04-10 21:01 至 2025-12-31 15:00。主力映射与 CSV 有 SHA256 清单。
- 人工方向使用含首尾的 DB 交易日区间，夜盘不按墙上日期匹配。无标签时 NO_TRADE；FG809 的 30656 根和 FG405 的 3795 根无标签。
- 20 吨/手，保证金 10%，开平各 3 元/手，单侧价差 1 点，无额外滑点。使用用户提供的 1.yaml 数值，不采用 DB multiplier=1。
- 用户确认数据完整。历史时段由已有完整数据生成，明确标记为用户确认，未声称经过独立交易所日历审计。
- 到期按用户定义为失去主力资格。在旧主力最后可成交分钟按旧合约价格强平，记费用与真实损益，原因 MAIN_ROLL，Reward=0。流动性不足不能伪造成交。
- 最后数据截止、人工训练窗口和方向切换均不授权免费清仓；保留或冻结未完成账户。

## 时间真实性

新入场只能下一根分钟 Open 成交；该分钟只能用成交之后的 Close 确认止盈，不能使用开仓前后顺序未知的最高/最低价。后续分钟要求穿透止盈价一跳才按目标成交。只有对应 3 手完整真实模拟成交止盈产生 60/(60+T) 奖励。

发现 FG809/901/905/001 共 8 根 bob=eob 的记录：仅标记账户价格，禁止所有成交和奖励；其他分钟长度异常拒绝加载。实际分钟特征通过现有特征计算的可选时钟扩展构建，旧训练的固定 345 分钟行为不变。

## 重现

在独立工作树设置 PYTHONPATH=src，使用 Dreamer Python 环境：

```sh
python tools/fg_entry_v1.py export --config configs/fg_entry_v1/fg.yaml --output NEW_EMPTY_EXPORT
python tools/fg_entry_v1.py build --config configs/fg_entry_v1/fg.yaml --export-manifest EXPORT/export_manifest.json --output NEW_DATASET
python tools/fg_entry_v1.py baselines --config DATASET/CZCE_FG501_test.yaml --output NEW_EVALUATION --seed 0
```

导出只读 DB；禁止覆盖冻结数据集。当前 dataset_v3 按交易日 60/20/20 划分，有 12 个完整可训练主力生命周期。生成 train_dataset.yaml 自动选择这些训练区段；没有主力切换证据的最终 FG605 不伪造到期。

## JM 溯源限制

只读检查 JM 8Y：655099 根，2018-01-02 至 2025-12-02，夜盘 223111 根，文件没有真实合约列。旧审计记录为主力连续，但不能证明具体导出命令。旧 data/export_future.py 查询固定 JM2601；参考 export_main_csv_v2.py 使用主力映射但删掉真实合约字段。本次不训练 JM。

## 验证

环境及共享会计、动作掩码、旧分钟特征兼容性共 68 项回归通过；真实 dataset_v3 的 Dreamer 准入与 Gym reset 验证通过。正式训练和样本外择时结论必须另行记录，不以接口冒烟代替。

## 重启后的重建

临时目录在重启后被清空，已提交分支恢复到原环境仓库 artifacts/fg_entry_v1/worktrees 下。统一配置导出起点修正为保留数据的实际起点 2018-04-10；原默认 2017-01-01 早于 DB 行情范围，新增的每日映射完整性检查会拒绝该缺失前缀。没有放宽区间内的缺失数据检查。重建导出再次核实 24 段、643929 根及相同首尾时间，排除 FG805，截止 2025-12-31 不变。
