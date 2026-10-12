# S3-B P1-B1：固定 TRAIN 平均权重归一化

## Summary

用户两条 P1 指出的是同一问题，本轮合并为一项修复。S3-B V2 使用
`L_batch = sum((w_i / global_mean_weight) * CE_i) / B`，分母来自冻结的采样总体。
S3-A 原内核和文件保持不变；只监督原 `pol/*`，不新增网络、不加入联合 RL Loss。
不重新搜索或生成生产 Oracle、标签、特征、价值证据或权重数组。

两仓库旧版均保存在 `fg-s3b-batchmean-v1-20261012` Git 标签中：Gym `ab39bd8`，
Dreamer `fd20e28`。V1 的实验文件不重签；当前代码、配置和 Checkpoint 使用 V2。

## Evidence observed

读取实际 `fg_s3a_actor.py`、`fg_s3b_bc.py`、权重数据接口、采样、Checkpoint 和
统一管理入口，确认旧 B 通过原 A 闭包计算 `sum_batch(w*CE)/sum_batch(w)`。
原 A 的等权默认情形分母为 B；替换为浮点权重后，该分母随批次变化。

使用保存的真实原神经 Agent CPU 诊断源，固定同一特征/标签，分别从相同参数
和 BC 优化器初态开始，batch size 为 1：

| 旧权重 | Loss | 原始梯度范数 |
|---|---:|---:|
| 1 | 0.6925262808799744 | 1.5790410041809082 |
| 3 | 0.6925262808799744 | 1.5790410041809082 |

实际比例为 **1**，针对要求比例 3 的断言真实失败，收据 `before.json`。
没有把代码推断当作运行证据。

## Root cause

对随机 mini-batch 使用随机分母，期望通常不能与除法交换。尤其 B=1 时，正
权重在分子分母中抵消。旧 B 为稳定 float32 而按当前 `max(w)` 同比缩放，仍
保留了这一随机比值目标。

若均匀抽样总体含 M 个样本，冻结 `mean_w = sum_population(w)/M`，则固定参数
下，每个均匀批次的原始梯度期望等于该总体加权目标梯度。CERTIFIED 采样的是
MATCHED 正权重子集，因此使用 M=396542；FULL 的原权重均值不能用于该采样。
零权重行省略不改变数学上的加权目标，但改变均匀采样时需要的总体平均权重。

## Changes made

Dreamer `WeightedActorBC` 要求显式 `global_mean_weight`，替换其 JIT 更新内核。
沿用原 Actor、真实 Mask、stop-gradient 特征、独立优化器、冻结审计和推理同步。
原 A 闭包没有 Loss 注入点，因此新 B 内核复用相同原 pol/MaskedCategorical 调用，
保留 A 字节不变。权重除以冻结均值后转 float32，越界/下溢明确拒绝；不再以
batch 的权重和或最大值归一化 Loss。batch sum/max 只用于校验和诊断统计。

Gym 新 `fg_s3b_normalization.py` 读取已验证的 V1 权重，新增一个不可覆盖的 V2
metadata 包。它引用旧权重 manifest 与数组，不复制或改写原训练数据。审计重算
总体统计并核对精确索引、均值及 SHA；不以人工填写的均值替代冻结证据。

生产 TRAIN 实际统计：

| 原权重总体 | N | sum_w (float64) | 原权重均值 |
|---|---:|---:|---:|
| FULL | 402096 | 158501.73754065624 | 0.39418879456810374 |
| MATCHED | 396542 | 158501.7375406562 | 0.39970983537848753 |

两次求和的末位差来自 float64 对不同长度数组的归约，原权重字节没有变化。
同时保存均值的 `float.hex()`，避免恢复时含糊比较。

| 训练模式 | 实际采样总体 | 使用的 global_mean_weight |
|---|---|---:|
| S3A_FULL | FULL；全 1 权重 | 1 |
| S3A_MATCHED | MATCHED；全 1 权重 | 1 |
| S3B_CERTIFIED | MATCHED；原冻结权重 | 0.39970983537848753 |

V2 权重协议 `FG_S3B_GLOBAL_MEAN_WEIGHTS_V2`；BC 协议 `FG_S3B_BC_V2`；独立脚本
配置 schema `FG_S3B_TRAIN_CONFIG_V2`。训练身份包含 Loss 版本、采样总体、N、sum_w、
均值与其 hex。Checkpoint 的 identity、状态和冻结证明都绑定同一常量；保存和
恢复检查实例与身份的一致性。V1 Checkpoint、旧 schema、不同均值不能静默恢复。

三套集中对照配置使用新的 `global_mean_v2` 实验名称、prepared_dir 和 logdir。
统一管理准备快照绑定新 normalization 文件、旧 base manifest/scale 和原源模型。
CPU 诊断若只取特征前缀，显式标记其独立且固定的诊断总体，不冒充完整 TRAIN。

## Verification performed

本轮唯一通过案例 **10 项**，包括以下强制检查：

| 检查 | 结果 |
|---|---|
| B=1，固定分母，权重 1 vs 3 | Loss 与原始梯度范数比例 3，rtol=1e-6 |
| 二样本 `[1,3]`，固定均值 1 | 原 MaskedCategorical/unimix 解析梯度及实际 optimizer 更新匹配 |
| B=256 | 4096 样本、seed=202、16 个打乱批次的梯度均值与全量梯度逐数组 allclose，rtol=2e-5、atol=2e-6 |
| 全 1 权重，均值 1 | 原 S3-A 参数、优化器和指标字节/数值完全一致 |
| 断点恢复 | 3+3 与连续 6 更新的参数、BC 优化器和下一批索引一致；原 policy 读取新 Actor |
| 冻结状态 | 非 Actor、非零原 optimizer/Norm/Gate 及原联合计数器不变 |
| 均值/协议篡改 | manifest、实例或保存状态不一致，以及 V1 协议，均拒绝 |
| 全量/子集及中央入口 | 总体均值、对照配置、prepare → start → resume 通过 |

具体收据位于 `artifacts/fg_s3b_v1/p1_global_mean_20261012/`：
梯度 2 项 `17.77s`，其余 B 接线 5 项 `40.36s`，metadata/config 2 项通过；
管理入口首先真实失败（找不到 V2 目录下的旧 `scale.json`），修复为从 base 权重
目录绑定校准后，针对性复测 1 项 `34.17s` 通过。失败 XML 与修复后 XML 都保留。
失败 XML 的原始字节另存 `.raw.xml.gz` 并记录 SHA；阅读版只剔除 pytest 输出的
行末空格，保留原失败状态和堆栈。
没有重复运行 S3-A 全套测试或不受影响的未来行情回归。

## Risks / not verified

验证的是固定参数下、优化器 clipping/Adam 之前的目标梯度；梯度比例 3 不代表
Adam 参数变化也为 3。现有 permutation sampler、AGC 和优化器配置保持原样。
当前只完成原小尺寸神经 Agent CPU 验证；正式 GPU 和 DEV 尚未执行。
生产模型仍执行原源 Checkpoint 准入。V2 不迁移旧 B 的 optimizer 或 Actor 轨迹。

## Next recommended step

```bash
cd /home/v/Documents/work/gym-trading-env2
PY=/home/v/miniconda3/envs/dreamerv3/bin/python
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fg_s3b_mpl

# 当前已生成的固定均值 metadata：仅 CPU 审计。
$PY tools/fg_s3b.py normalization-audit \
  --output artifacts/fg_s3b_v2/weights_global_mean_v2

# 独立新目录复现 metadata；不会重算 Oracle 或改写 V1 数组。
$PY tools/fg_s3b.py normalization-build \
  --base-weights artifacts/fg_s3b_v1/weights_full_v1 \
  --output artifacts/fg_s3b_v2/weights_global_mean_reproduce

$PY tools/fg_s3_experiment.py plan --config S3B_CERTIFIED
# 后续核对正式源后 prepare/start；本轮不启动这些正式训练阶段。
$PY tools/fg_s3_experiment.py prepare --config S3B_CERTIFIED
```

以后新实验使用 V2 配置和独立目录；同一 V2 身份可恢复。旧 B 实验若需继续原
行为，使用两仓库对应 V1 标签的隔离 checkout 和原配置，不在 V2 中强行加载。
