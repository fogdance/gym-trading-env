# 正式教材长分数证据故障：实施 Review

## 结论与真实现场

TRAIN01_FG809_FULL_MAIN 的正式V1生成失败，exit=2，墙钟707.32秒。原始日志的异常来自冻结源码 `fg_s3_e_quality_strict.py:100` 的 `str(wait)`，不是行情校验；Python 3.11.11 默认整数十进制转换上限4300位。同一冻结Reader在 `fg_s3_quality_catalog_v2.py:103` 使用 `Fraction(witness)`，读回端也受该限制影响。

原现场 `/data/s3_corpus/train_full_main_20261018_cpu20_v1/` 保留；无完整Corpus，G3/Reader均NOT_RUN。原始generate.log SHA256为 `2a3a8ec0e2dbd1b3bd21bbf94019f13dd824ffa48154786d92c3af09d054e4b4`，仓库中的失败收据副本逐字节一致。

## 最小修复

发布提交 `9f0f847979e2047a5c9b2d87206c4423ecf111a3` 的 [V2生产包](formal_train_corpus_20261018_v2/README.md) 在生产脚本及其子进程导出 `PYTHONINTMAXSTRDIGITS=0`。保持原十进制精确分数字符串及数学合同；没有改成float近似、截断后缀或更换金融规则。系统Python默认配置未改。

实际Producer仍为Gym `b0d5ef9af07fbdd0c553dd943ac9eb40c9a84d46` / Dreamer `fd144113ce706e2d8f1ce8e7828ece0799c8bf7a`。冻结代码、242份运行源码哈希及18份来源锁未变化。新输出根为 `/data/s3_corpus/train_full_main_20261018_cpu20_v2`，禁止覆盖V1。

每次生产先执行精确数值运行环境预检，覆盖父进程、spawn worker和原生G3所用的新解释器继承模式；最终BATCH_MANIFEST绑定运行环境设置及原始预检收据。

## 验证结果

3个定向测试全部通过，失败0、跳过0：默认4300位限制下复现写出和读回错误；通过用户入口验证32000步精确分数的父/worker/verifier一致；通过真实冻结Reader的`validate_corpus_label`验证长证据可读且原数学值不变。实际分子80677位、分母80719位，字符串往返精确相等，短证据字节不变，无行情文件或优化器访问。

[验证摘要与准确命令](formal_train_corpus_20261018_v2/evidence/PLAN_VALIDATION.json)、[JUnit](formal_train_corpus_20261018_v2/evidence/RUNTIME_REGRESSION.xml)、[原始数值预检](formal_train_corpus_20261018_v2/evidence/RUNTIME_PROBE.json)。脚本bash语法、内嵌Python及发布包SHA清单通过。差异空白检查排除两份逐字节保存的原始失败日志/命令收据，其中尾随空格来自生产现场，不修改原件以迎合格式检查。

## 未验证与后续

完整V2正式生成、G3、Reader及质量覆盖均NOT_RUN；CPU20占核不再测试，无GPU、TEST或真实优化器更新。由用户执行：

```bash
cd /home/v/Documents/work/research/fg-s3-formal-corpus-plan-20261018/gym-trading-env
bash docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh produce TRAIN01_FG809_FULL_MAIN
```

本修复只在独立生产计划分支提交，未合并到s3-next。DATA-MULTI的独立双仓worktree仍在，后续继续原任务；不修改WP04/WP06工作树。
