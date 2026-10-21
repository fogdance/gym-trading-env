# 正式教材与 FG809 F1 改动集成记录

## 范围与提交

本次仅合入本对话使用的两个 Gym 分支，保留当前主线已有实现：

| 分支 | 合入提交 | 内容 |
| --- | --- | --- |
| `fix/corpus-stage-drain-20261021` | `6f1403c2da2bd112d0222b22edb9bb44bd5c2971` | 发布阶段使用已完成 G3 自身的原始 Verifier Pin 认证，解除与当前 Publisher HEAD 的错误绑定 |
| `run/f1-fg809-20261021` | `29e47fb502ea406eacdbbd61eecd984e615e0233` | FG809 单合约、size12m、10,000 次更新的独立配置与正式命令 |
| `run/f1-fg809-20261021` | `3a80e1f16139d01cb63ae66562cfaf7c294eb4ae` | 启动和只读核对共用规范绝对路径与同一组 Verifier 参数 |

Gym 合并前主线：`99e9f81dc91440bd38a04e84adacbb7c2616baa7`，已包含前次教材恢复及来源身份修复。
Dreamer 主线：`990f61f8b7ac6bad25ff9e41b65624ae13ccf497`。两个来源的 Dreamer 提交均已包含在该主线，无新增提交需要合并。

合并没有冲突。新增集成证据前的差异共 9 个文件、360 行增加、3 行删除；未替换主线 Batch3、默认恢复等其他实现。生产模块、回归测试及 F1 配置/启动文件与各自来源逐字节一致。

## 根因与修复边界

教材发布曾将历史已成功 G3 的认证身份与更新后的 Publisher 身份直接比较，导致合法结果发布失败。现改为按密封执行记录、外部 Pin SHA256、仓库身份及干净状态校验原始认证者，同时保留当前 Publisher 认证和全部 Native/Reader/内容校验。

F1 首次启动曾传入包含 `..` 的 Pin 路径，被既有路径合同拒绝。现启动和 status 都传入规范绝对路径，不放宽路径校验。

## 本次集成验证

从集成 Gym 目录执行，真实退出码均为 0：

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 \
  PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider \
  tests/entry_v1/test_corpus_publication_pin.py \
  --junitxml=docs/corpus_resume_fix_20261021/publication_pin_integration_checks.xml -q

bash -n docs/f1_fg809_20261021/RUN_F1.sh
git diff --cached --check
```

6 passed，0 failed，0 errors，0 skipped。覆盖历史/显式执行绑定正例以及 HEAD、dirty、Pin、cwd 篡改拒绝。原始收据为 `publication_pin_integration_checks.xml`。

## 历史执行证据与运行保护

`PUBLICATION_FIX_EVIDENCE.json` 是来源分支真实 FG809 隔离发布验证的原始收据：退出 0、67.563 秒，原 Corpus manifest 和正式 control state 均未改变。本次没有重跑该生产操作。

`../f1_fg809_20261021/CONFIGURATION_CHECK.json` 与 `HISTORICAL_LAUNCH_LOCK.json` 是 F1 冻结工作树准备时的原始证据快照；其中 NOT_RUN/状态字段描述记录当时，不能代表当前用户启动后的进程状态，也不是集成工作树的新认证凭据。

本次仅在集成工作树写入和提交；未修改生产/F1 冻结工作树、release、Corpus 或既有 Run。原正式命令继续指向各自冻结工作树。本次未执行正式 prepare/start/resume、G3、GPU 训练或 TEST 读取；这些运行验证为 NOT_RUN。
