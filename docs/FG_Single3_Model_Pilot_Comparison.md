# FG Single3 模型先导比较（准备完成，尚未运行）

模型状态：NOT_RUN。尚未选定训练Reward，未初始化神经Agent、未启动GPU、未执行梯度更新；择时能力NOT_ESTABLISHED。

四候选各准备M0自然池、M1自然+P4池，供用户验收后选一组Reward运行。两组网络结构沿用fg_entry_v1/size50m；horizon333、imag15、B16/64+context1；model seed0、Replay seed202；每组2000次梯度更新（2,048,000训练时间位置），完整Uniform容量。

模式是完整Agent联合离线训练，复用现有Mask warmup并检查ADD合法类post/prior召回及真实正类支持数。不是WM-only；不新增Actor BC Loss。Gate未通过时沿用原有Actor/Critic loss门控。

CPU FakeAgent测试证明首次训练关闭Actor，满足Gate后开启，Gate/训练carry/预取队列/Replay RNG复用原FullCheckpoint保存和恢复，不重复更新；不同身份恢复拒绝。真实GPU权重及训练结果尚未验证。

启动脚本默认authorized=False，会在Agent初始化前拒绝启动；只有用户确认Reward、完整Replay覆盖和Oracle结果后才能设置True。旧6手Replay、Checkpoint、不同Reward和不同M池的Checkpoint身份均不能接续。

每个候选配置及预检在此目录；不存在已训练模型结果。
