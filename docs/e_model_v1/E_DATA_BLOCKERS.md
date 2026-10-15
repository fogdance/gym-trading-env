# E-Data 联调与生产组合阻断

E-Model 本轮完成 M0–M3 后暂停等待 Review；不自动进入 M4/M5。

## 最新只读事实

开发开始时E-Data已有代码提交，实施期间交付独立报告。只读 `docs/e_data_v1/evidence/acceptance.json` 给出：

- 技术状态 `E_DATA_V1_READY_FOR_INTEGRATION`，Review为 `PENDING_USER_REVIEW`。
- 被验收代码 Gym `a284416aac2138ea419eaf18228d309f2913e4b6`；文档HEAD读取为 `591ed63d9ee08389656e2fd357fced3b07c686a9`；只读Dreamer `d41bc80ba13b3dc3f5d2c32fcdc800613b2d0b4d`。
- Schema `959724f77f214dff7869ead9d83dc3fb8d2700940ab4fc9befe4995e5672d90f`。
- F01–F11：21 PASS/0 SKIP；原cache：6 PASS；18真实TRAIN合约、488160源行、32564导出决策。以上为独立报告证据，本E-Model任务没有重复执行这些数据数学检查。
- 全量产物 `fg-s3-e-data-dev/e_data_v1/train_build_20261015/`；旧pilot schema不能替代上述版本。

## M4 必须解决

1. Review确认独立数据源码、F01–F11/真实TRAIN收据、schema/hash和合法旧行情helper身份。数据当前未合入本模型基线。
2. 将认证E-Data consumer接入本轮provider接口；使用原数据组件，不重写公式。逐决策覆盖不足时拒绝，不能用最近窗口填历史或用stride15样本假装全部分钟。
3. 替换CPU synthetic来源绑定为经Review的真实E-Data绑定，认证Teacher/Outcome/TRAIN关联与同一价格anchor、真实合约及as-of时刻。当前正式E direct_init不能使用CPU收据。
4. 完成真实FG Env→Collector→DAgger→Replay→Agent，以及三教学的正式CPU准备/签名与真实来源Checkpoint恢复检查；模型合成测试不能代替这项验收。
5. 评价使用认证的DEV E观察和相同冻结模型身份。Teacher控制成绩不能称Actor-only成绩。

## M5 / EM18

仅M1–M4全部通过且有Review结论后，才能在唯一Composition解析器解锁三E组合；两Tutor+B2组合继续NOT_IMPLEMENTED。禁止启用参数、环境变量、产物ready字段或CPU receipt来解除守卫。

GPU smoke、正式1M、DEV Actor-only和W&B online均为 `PENDING_USER_AUTHORIZATION`。本任务不合并integration/main、不编辑旧B2/Tutor/Archive/W&B Run。
