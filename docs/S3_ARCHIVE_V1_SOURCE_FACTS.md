# S3归档实际来源与实现依据

2026-10-14。规范完整读取自 `S3_Research_Archive_Standard_V1_20261014.md`，Downloads与研究仓库docs副本SHA256均为 `d84ed87be768fcd45dd94309c76c05ddff112af5da0c92c0725fc1d54a4c5906`。

## 观察事实

- 独立归档分支基线：W&B追踪器 `ad6490675bf853a7793763f95aa7ba7f12fa379e`，包为 `research_tracking/`，入口为 `tools/fg_s3_wandb.py`。
- 原Tutor已完成1M，`training_receipt.json` 为 `FORMAL_COMPLETE`；800K归档不需要停止或恢复原Run。
- Run、Checkpoint、prepare记录的Gym SHA为 `3a6f889adf65583a1ab636f244ca05218f19196f`，Dreamer SHA为 `c18a8b5917f73110f9b6ce840495f4da1d7ca1d4`。先前193个执行文件与历史Git blob一致，准备时dirty为空。
- 800K manifest字节SHA256：`5cf093b56ca3bb340cca9520b5579105b1cecc9f044262d5a23d17c545c6a474`；评价receipt字节SHA：`b1665c584aedbd7629494a0fd5ad68fe6e52d1649a03de602687d583530e05f9`；receipt content_hash：`7a1c6cc66e7c7ec4bc6b3b6f5bb8984a96b7bb3a6bfb3746600a8cd5a15c9368`。
- 真实W&B映射publication为 `tutor-cd-reupload-v1`，训练Run `3dc57d410d9b50601120`，评价Run `9832f27f1b038b84fb58`；由trace_identity及tracker_state核实，不靠名称查找。
- A2、B2、Tutor及E2真实Run均使用 `FG_S3_V2_LIVE_LEARNING_V1` 和 `FG_FULL_V1` 八组件保存格式。B2历史源码SHA与当前B2 HEAD不同；E2旧mode仍为tutor，分类可显式指定。

## 代码证据与设计对应

- `FullCheckpoint.save`会在已完成node上更新protected字段，因此done不能代替manifest前后哈希检查。`load`要求完整八组件，不能仅凭agent.pkl宣布Full。
- Collector恢复依赖journal前缀；800K水位是1,384,113,971字节、commit_seq=800807；更新/周期前缀分别为32000/8000。
- `LiveRun`恢复会校验代码、依赖、来源文件并可能修剪日志，所以只能在私有映射的临时Run中运行。一次受控更新调用原 `LiveRun.update`，不调用正式run_to。
- 原评价入口固定GPU，读取ckpt/latest；验证临时副本明确指向选定800K，避免误评已完成1M。
- 原评价读取真实DEV价格和TRAIN规则快照，没有Oracle指导；隔离worker进一步拒绝打开Oracle数据目录。
- 子manifest嵌入父manifest时，其相对files路径仍属于子目录。父顶层files索引与独立子manifest提供实际路径；不能递归按父目录重定位。
- 原W&B SDK worker恢复Run时传入固定Tags；现扩展为合并云端已有Tags，保护归档里程碑。

## 核验约束

旧归档完成过八组件CPU加载与精确回读，但没有独立重复评价及一次优化器更新。因此它的旧等级不会直接作为S3_ARCHIVE_V1的L3证据。本工具重新产生隔离评价、恢复和哈希报告。

用户选择独立评价和恢复使用GPU，无需等待GPU空闲。普通CLI和CPU测试仍不调用GPU；其他进程不被停止或重启。原工件、旧归档和其他Codex工作树不修改。
