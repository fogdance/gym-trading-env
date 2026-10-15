# S3 E-Model V1：模型与来源接口

本接口描述 M0–M3 模型实现。正式 E 组合仍为 `waiting_e`；CPU 合成数据及内部模型构造不能作为正式 Prepared/Run 使用。

## 配置与初始化

```yaml
schema: FG_S3_V2_COMPOSITION_CONFIG_V1
composition:
  encoder: e_multiscale_1m15m
  teaching: b2  # a2 / tutor_cd / b2
source_kind: direct_init
model_init:
  seed: 0
  presets: [size50m]
```

配置仅表示目标架构。三份 `E_MODEL_V1_*_WAITING.yaml` 的公开 prepare/start/resume/evaluate 仍被拒绝；`e_data_manifest` 是尚待真实联调的占位路径，不是已认证数据。旧三种 Legacy、旧 C2 不添加 E metadata。

`dreamerv3.fg_s3_e_config.resolve_model_init()` 使用官方 YAML reader 与 elements.Config，从原 defaults/size preset 构造完整目标配置。不修改公共 presets、不继承旧 Encoder 参数。默认 size50m：1m/fusion512、15m128、原 RSSM deter4096/hidden512/stoch32/classes32；模型时钟 H333/S3_MARKET_MINUTE、recompute_prefix1。Gate 初态 CLOSED。

## Encoder / Decoder

`S3EMultiscaleEncoder.initial(B)`、`entry_space`、`truncate(...)` 均为空状态；`__call__(carry, obs, reset, training, single=False)` 返回 `(carry, {}, tokens)`。Policy tokens `[B,D]`，训练 tokens `[B,T,D]`。D 来自1m MLP宽度。

- 1m白名单：`market_seq_e_shared float32[60,18]`、`position_state float32[9]`、`action_mask float32[2]`，原简单MLP三层。18列不重排、不按同类价格处理。
- 15m：`market_context_15m float32[24,7]`；invalid 行前6列清零，flatten168→Linear128→GELU，全无效 embedding 置零。
- Concat→常规初始化 Linear(D)，参数路径 `enc/minute/*`、`enc/context15/*`、`enc/fusion/*`。同一 enc 用于 policy/WM/recompute_prefix；无额外 recurrent state。
- Decoder仅重建 shared1m 和 position。旧 `market_seq` 留在观察/Replay作审计，但不编码、不重建；15m仅编码，Mask走原 availability 头。
- 原 RSSM、Actor、rew/con/avail、A2 BC、Tutor Balanced/Pair、B2 Outcome/Adapter/schedule 不改公式。原 BC 的 stop_gradient 继续隔离 Encoder。

## E Observation / Replay / Provider

`fg_s3_e_observation` 定义 `FG_S3_E_OBSERVATION_V1`、`FG_S3_E_REPLAY_MARKET_MINUTE_V1`、`FG_S3_E_DECODER_V1`。E_FIELDS 在旧字段之上增加两份 E 数组；旧 FIELDS/S3_FIELDS 原字节不改。`market_advance` 继续为原 int32扩展字段，reset/元数据不计环境步。

V2Collector 增加 `observation_schema` 与 `e_provider`，旧默认值不变。Provider接口：

```python
provider.identity  # frozen data manifest identity
arrays, metadata = provider.snapshot(decision_env, original_obs)
```

arrays只能包含 `market_seq_e_shared`、`market_context_15m`。metadata必须含真实 source_contract、decision_time、visible_cutoff、data_identity；可含 context_available_at。真实合约与决策时刻必须对齐当前 Env，可见时刻不能超出截止。metadata/provenance只进Sidecar，不进 Actor。

Adapter只校验/传递数组，不实现15m聚合、价格/量仓归一化或新缓存。CPU fixture 是明确标记的人工数组，不证明真实E-Data因果性。每条Replay保存该步实际窗口，教师join校验包括两份E数组的observation hash。保存实际执行动作，原Actor prevact替换逻辑不变。

## 来源与生命周期

公开 `make_agent / inspect_source / prepare_transfer / apply_and_audit_transfer` 和统一CLI保持require_ready。共用神经实现抽到 `_S3V2Model`；内部构造/认证核心仅供当前 CPU 契约测试，保持真实 waiting_e。公开 S3V2Agent仍拒绝等待组合；不新增解锁开关或第二套训练器。

E只能direct_init，必须有整数seed、完整配置及身份绑定。当前 `FG_S3_E_SOURCE_BINDING_V1` 只认证 `FG_S3_E_MODEL_SYNTHETIC_V1` CPU诊断清单；绑定模型配置、Obs/Decoder/Replay contract、数据文件hash、teaching、双仓SHA/dirty及完整执行代码hash。真实E-Data认证需要M4/Review，明确拒绝自动推广CPU收据。

E+B2保留原head/adapter/schedule构造，实际NN参数必须齐全；不能用基础Agent伪装B2。旧B2只允许原warm-start协议，旧参数schema、字节认证和允许扩展规则保持。

Source/Prepared/model/Run/evaluate增加E binding一致性检查，FullCheckpoint沿用原完整组件。CPU Prepared契约使用独立协议 `FG_S3_E_CPU_PREPARED_CONTRACT_V1` 和 `CPU_CONTRACT_ONLY_NOT_PRODUCTION_PREPARED`；不是正式prepare。CPU Checkpoint使用 `FG_S3_E_CPU_FULL_CHECKPOINT_CONTRACT_V1`，保留 waiting_e，验证自身exact resume。GPU/正式Run/DEV交易/W&B online均未运行。
