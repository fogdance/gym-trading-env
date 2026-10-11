# FG Single3 Horizon 与长期信用分配审计

状态：代码与最小CPU测试完成，正式全Fold实测NOT_RUN，等待D-Gate。该文档不能当成已完成H验收。

已实际通过1项同账户分叉/时钟测试：ADD与WAIT从完整财务和Reward状态完全相同的快照出发，只改变第一个动作；随后P1相同，跨chunk及SL真实执行，模板账户未被分叉修改。第一条反馈指数0，1001步反馈按γ^1000折扣。

正式采集：每个TRAIN合约固定seed202抽取8个当前已知合法空仓P1观察，同一cursor用于四候选。每对ADD/WAIT通过原Env到真实终局/TRAIN右删失。576对账户，比较H333/1000/3000及gamma1，保存完整快照和源trace哈希、两分支反馈/最终状态哈希。Δ属于同一个冻结后续P1的有限政策估值，不是最优Q或Critic预测。

实际源码依据：Dreamer `agent.py` continuation在contdisc=True时训练为γ×(1−terminal)；imag_loss显式disc=1，gamma由continuation携带；repl_loss显式γ；lambda_return使用rew[:,1:]、最后值bootstrap。imag_length15、sample64+context1、Env1024均保持；原Replay采样起点和Env chunk的is_first会重置网络隐状态，账户状态却保留。H变化也改变continuation目标，不能仅改报表称为模型换H训练。

完成D后生成实测报告更新本文；不计算模型未运行时的Critic误差，不由折扣大小断言Agent已低估风险。
