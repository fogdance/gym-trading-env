# FG Single3 模型先导准备

状态：NOT_RUN；等待E/R/O/D/H完整CPU验收和用户确认。没有初始化神经Agent、没有启动GPU、没有执行梯度更新。

准备C0/C1/C2/C3各自的M0自然池与M1自然+Oracle池。每组联合离线完整Agent训练2000次梯度更新、model seed0/replay seed202、原size50m网络/horizon333、B16×64+context1、imag15，完整Uniform容量。原有Mask warmup和ADD类召回/支持数将接入offline_pretrain；不是WM-only，不新增Actor BC。

新入口只接受签名的全Fold3手数据、1000真实batch覆盖、576同账户H对照；一份训练run只含一份Reward。启动配置默认authorized=False；必须由用户确认CPU结果后再启用。新Checkpoint身份绑定数据、Reward、Obs、M池、网络/训练配置和源码；复用原FullCheckpoint与RuntimeState保存Agent、预取队列、Replay RNG、carry及Gate；旧6手或其他Reward/池的checkpoint拒绝接续。

正式CPU预检与配置会在D/H产物完成后生成，测试和状态以实测收据为准。工程准备不能作为Agent择时能力验收；能力NOT_ESTABLISHED。
