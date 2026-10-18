# WP04 实施进度

## A：课程和完整模型

复用WP03来源接口和原_make_agent/quality_init_loss，新F1显式FOUNDATION_WM路由，不是legacy HEADS_ONLY。初始化所有WM、Quality/Actor/Critic及原优化器。F1 ext_space只保留原生时序，后验不进入训练输入。

A_contract：12 passed / 1 deselected，exit0；A_model：1 passed，exit0，实测两个fresh模型相同seed逐键一致且完整布局存在；A_guards：43 passed，exit0，含原公开Guard和legacy Pair。原命令/退出码、XML和源码字节见receipts/wp04。上述是三个实际集合，不是全项目测试。

本阶段未实现优化器训练，非初始化F1调用明确阻断；B阶段补齐该路由。任何真实TRAIN更新、GPU、TEST及全量生产均NOT_RUN。
