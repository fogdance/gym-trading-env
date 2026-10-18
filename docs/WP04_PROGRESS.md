# WP04 实施进度

## A：课程和完整模型

复用WP03来源接口和原_make_agent/quality_init_loss，新F1显式FOUNDATION_WM路由，不是legacy HEADS_ONLY。初始化所有WM、Quality/Actor/Critic及原优化器。F1 ext_space只保留原生时序，后验不进入训练输入。

A_contract：12 passed / 1 deselected，exit0；A_model：1 passed，exit0，实测两个fresh模型相同seed逐键一致且完整布局存在；A_guards：43 passed，exit0，含原公开Guard和legacy Pair。原命令/退出码、XML和源码字节见receipts/wp04。上述是三个实际集合，不是全项目测试。

本阶段未实现优化器训练，非初始化F1调用明确阻断；B阶段补齐该路由。任何真实TRAIN更新、GPU、TEST及全量生产均NOT_RUN。

## B：原生WM更新、窗口与冻结

新增K步真实warmup、原生WM-only单次原优化器路由；固定六组正向白名单，登记实际349个状态叶，冻结其余参数/slot/Ninjax状态。非有限候选恢复原全状态，在host计数/政策发布前拒绝。

B_final：17 passed，exit0（163.17s pytest）；B_byte_review：1 passed，exit0。包含T64/K32的4次CPU合成更新，逐键冻结、真动量和weight decay、warmup梯度0、独立窗口carry、错误账户/时序/输入及不足支持拒绝。共享20项、WP03来源/Join/四Reader视图56项均passed/exit0。精确参数/slot差异见B_WM_update_evidence.json和B_delta_summary.json。

反向审查强化原生字节校验：np.array_equal会把+0/-0视为相等，改用dtype/shape校验后的逐行tobytes，补正负回归。Index V3仅新增原生过去事件/执行元数据，不改原Corpus/Chunk；旧V2索引不冒充本版本，原WP03冻结树不动，其他课程行为已回归。

保留失败：B_smoke为Ninjax要求所有梯度目标均被读取，修复为只读module.values，不评估冻结头或添加假损失；B_targeted暴露单步测试错误，原优化器warmup首步lr=0，经独立原优化器观察确认，保留原配置并按多步验证；B_retry暴露测试把nj.grad初始化重入误认为训练，改按实际初始化生命周期保护。没有放宽WM更新、冻结或来源条件。

本轮真实TRAIN参数更新、GPU、TEST、18合约生产继续NOT_RUN。后续C诊断、D完整冷恢复和E真实只读G3尚未完成。
