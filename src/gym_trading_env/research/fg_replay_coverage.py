"""Audit frozen canonical traces and REAL Uniform Replay batches on CPU."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import resource

import numpy as np

from .fg_replay import (FIELDS, SCHEMA, cpu_runtime, digest, iter_sidecar, read_manifest,
                        signed, trace_manifest, validate_dataset, write_json)


COLUMNS = (
    "decision", "valid_decision", "legal_add_mask", "legal_add_decision",
    "add_requested", "illegal_add", "voluntary_wait", "restricted_wait", "full_wait",
    "add_fill", "add_rejected", "tp", "non_tp_exit", "early_first5", "early_next10", "ambiguous_tp",
    "positive_reward", "empty", "lots3", "lots6", "advanced", "valid_minute", "chunk_reset",
    "age_lt15", "age_15_60", "age_60_480", "age_480_1440", "age_ge1440",
)
INDEX = {k: n for n, k in enumerate(COLUMNS)}
AGE_COLUMNS = COLUMNS[-5:]


def minute_age(end, start):
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60


def row_labels(row):
    """Decision labels live at obs_t; fill/TP labels live at feedback_t."""
    decision = bool(row["decision_executed"])
    legal = row["legal_add"]
    action = row["action"]
    values = dict(decision=int(decision), valid_decision=int(row["decision_valid_minute"]),
        legal_add_mask=int(legal), legal_add_decision=int(decision and legal),
        add_requested=int(decision and action == 1), illegal_add=int(decision and action == 1 and not legal),
        voluntary_wait=int(decision and action == 0 and legal),
        restricted_wait=int(decision and action == 0 and not legal),
        full_wait=int(decision and action == 0 and row["lots_after"] == 6),
        add_fill=sum(f["reason"] == "ENTRY" for f in row["fills"]),
        add_rejected=int(row["advanced"] and row["incoming_action"] == 1 and bool(row["reject_reason"])),
        tp=len(row["reward_events"]),
        non_tp_exit=sum(e["reason"] != "TAKE_PROFIT" for e in row["exits"]),
        early_first5=row["diagnostics"]["early_first5"], early_next10=row["diagnostics"]["early_next10"],
        ambiguous_tp=len(row["diagnostics"]["intrabar_ambiguous"]),
        positive_reward=int(row["official_reward"] > 0),
        empty=int(decision and row["lots_after"] == 0), lots3=int(decision and row["lots_after"] == 3),
        lots6=int(decision and row["lots_after"] == 6), advanced=int(row["advanced"]),
        valid_minute=int(row["valid_bar"]), chunk_reset=int(row["reset"] and row["cursor"] > 0))
    if decision:
        for position in row["active_entries"]:
            age = minute_age(row["time"], position["entry_time"])
            if age < 0:
                raise ValueError("Position age precedes the actual fill")
            bucket = int(np.searchsorted([15, 60, 480, 1440], age, side="right"))
            key = AGE_COLUMNS[bucket]
            values[key] = values.get(key, 0) + 1
    return np.array([values.get(k, 0) for k in COLUMNS], dtype=np.uint16)


def distribution(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return dict(count=0, minimum=None, median=None, p90=None, maximum=None)
    return dict(count=len(values), minimum=float(values.min()), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), maximum=float(values.max()))


def corpus_statistics(root, dataset, *, columns=COLUMNS, label_fn=row_labels, trace_schema=SCHEMA,
                      group_fn=None, row_hook=None, trade_finalize=None):
    root = Path(root)
    index = {name: n for n, name in enumerate(columns)}
    summaries = defaultdict(Counter)
    strata = defaultdict(Counter)
    market_reuse = Counter()
    rejection_reasons = Counter()
    labels, chunks = {}, {}
    trajectories, trades = [], []
    trace_bytes = 0
    account_restores = 0
    for trace_index, record in enumerate(dataset["traces"]):
        directory = root / record["directory"]
        trace = trace_manifest(directory, schema=trace_schema)
        receipt = read_manifest(directory / "reconciliation.json")
        account_restores += receipt["account_restores"]
        policy_id = group_fn(record) if group_fn else record["policy_id"]
        matrix = np.zeros((trace["records"], len(columns)), np.uint16)
        opened, active = {}, {}
        for row in iter_sidecar(directory):
            ordinal = row["ordinal"]
            if not 0 <= ordinal < len(matrix):
                raise ValueError("Sidecar ordinal outside the saved Replay")
            values = label_fn(row)
            matrix[ordinal] = values
            if values[index["add_rejected"]]:
                rejection_reasons[(policy_id, row["reject_reason"])] += 1
            if row["valid_bar"]:
                market_reuse[(record["contract"], row["time"])] += 1
            if row["decision_executed"] or row["advanced"]:
                for dimension, value in (("year", row["time"][:4]), ("contract", record["contract"]),
                                         ("direction", row["direction"]), ("session", row["session"])):
                    strata[(policy_id, dimension, value)].update(dict(zip(columns, map(int, values))))
            for fill in row["fills"]:
                if fill["reason"] == "ENTRY":
                    entry_id = fill["entry_id"]
                    if entry_id in opened:
                        raise ValueError("Duplicate opening entry identity")
                    opened[entry_id] = dict(fill, cursor=row["cursor"], stepid=row["stepid"],
                        observed_bars=0, MAE_points=0., contract=record["contract"], policy_id=policy_id)
                    active[entry_id] = opened[entry_id]
            if row["advanced"] and row["bar"]["valid_bar"]:
                for item in active.values():
                    adverse = (float(item["price"]) - row["bar"]["low"] if item["side"] == "LONG"
                               else row["bar"]["high"] - float(item["price"]))
                    item["MAE_points"] = max(item["MAE_points"], adverse, 0.)
                    item["observed_bars"] += 1
            for event in row["exits"]:
                item = active.pop(event["entry_id"])
                item.update(exit_time=event["exit_time"], exit_reason=event["reason"],
                    duration_minutes=minute_age(event["exit_time"], item["time"]),
                    cross_env_chunk=(item["cursor"] - 1) // 1024 != (row["cursor"] - 1) // 1024,
                    cross_replay_chunk=item["stepid"][:32] != row["stepid"][:32])
            if row_hook:
                row_hook(row, opened, active, record)
            last = row
        for item in opened.values():
            if "exit_time" not in item:
                item.update(exit_time=None, exit_reason=trace["terminal_status"],
                    duration_minutes=minute_age(last["time"], item["time"]),
                    cross_env_chunk=False, cross_replay_chunk=False)
            item["status"] = ("success" if item["exit_reason"] == "TAKE_PROFIT" else
                "terminal_failure" if item["exit_time"] else
                "terminal_failure_unfilled" if trace["pending_reason"]
                else "right_censored")
            item["MAE_basis"] = "observed_post_entry_OHLC_conservative_upper_bound"
            if trade_finalize:
                trade_finalize(item, record)
            trades.append(item)
        if sum(v["status"] == "success" for v in opened.values()) != trace["tp_events"]:
            raise ValueError("Coverage trade/TP identity mismatch")
        totals = dict(zip(columns, map(int, matrix.sum(axis=0, dtype=np.int64))))
        if totals["illegal_add"] or totals["tp"] != trace["tp_events"] or totals["decision"] != trace["advanced"]:
            raise ValueError("Coverage does not reconcile with real transitions")
        summaries[policy_id].update(totals)
        summaries[policy_id].update(trajectories=1, records=trace["records"],
            sequence_starts=trace["sequence_starts"], open_entries=trace["open_entries"],
            non_advancing_rows=trace["records"] - trace["advanced"], unvisited_rows=trace["unvisited_rows"])
        labels[trace_index] = matrix
        offset = 0
        for chunk in trace["chunks"]:
            import elements
            key = bytes(elements.UUID(chunk["uuid"]))
            if key in chunks:
                raise ValueError("Independent trajectories share a chunk")
            chunks[key] = (trace_index, offset, chunk["length"])
            offset += chunk["length"]
        if offset != trace["records"]:
            raise ValueError("Trace chunk lengths do not match coverage labels")
        trace_bytes += sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        trajectories.append(dict(policy_id=policy_id, contract=record["contract"],
            records=trace["records"], advanced=trace["advanced"], valid_minutes=trace["valid_minutes"],
            chunk_resets=trace["chunk_resets"], account_restores=receipt["account_restores"],
            TP=trace["tp_events"], non_tp_exits=trace["non_tp_exits"],
            open_entries=trace["open_entries"], unvisited_rows=trace["unvisited_rows"], status=trace["terminal_status"]))
        print(f"Coverage indexed {policy_id} {record['contract']}", flush=True)
    return dict(summaries=dict(summaries), strata=dict(strata), labels=labels, chunks=chunks,
                rejection_reasons=dict(rejection_reasons),
                trajectories=trajectories, trades=trades, trace_bytes=trace_bytes,
                account_restores=account_restores, unique_advanced_minutes=len(market_reuse),
                market_reuse_histogram=dict(sorted(Counter(market_reuse.values()).items())))


def sample_statistics(root, dataset, corpus, pool_id, *, batches=1000, batch_size=16, seed=202,
                      columns=COLUMNS, group_fn=None):
    """Use real Replay.load/sample; Sidecar labels are joined by actual stepid."""
    import embodied
    index = {name: n for n, name in enumerate(columns)}
    members = dataset.get("pool_members", {}).get(pool_id)
    records = ([dataset["traces"][i] for i in members] if members is not None else
               [r for r in dataset["traces"] if pool_id == "all" or r["policy_id"] == pool_id])
    expected = sum(r["sequence_starts"] for r in records)
    if not expected or min(batches, batch_size) <= 0:
        raise ValueError("Positive batch sizes and nonempty Replay are required")
    replay = embodied.Replay(length=65, capacity=expected, online=False,
                             directory=str(Path(root) / dataset["views"][pool_id]), seed=seed)
    try:
        replay.load()
        if len(replay) != expected:
            raise ValueError(f"Native Replay silently omitted sequences: {len(replay)} != {expected}")
        counts = np.zeros((batches, 2, len(columns)), np.int64)
        sampled_starts = set()
        source_counts = Counter()
        fingerprint_batch = hashlib.sha256()
        cross_chunks = 0
        reproducible = False
        for n in range(batches):
            before_rng = copy.deepcopy(replay.sampler.rng.bit_generator.state) if n == 0 else None
            batch = replay.sample(batch_size, mode="train")
            if n == 0:
                replay.sampler.rng.bit_generator.state = before_rng
                same = replay.sample(batch_size, mode="train")
                reproducible = all(np.array_equal(batch[k], same[k]) for k in batch)
                if not reproducible:
                    raise RuntimeError("Fixed Uniform RNG did not reproduce the complete native batch")
            if set(batch) != set(FIELDS) | {"stepid"}:
                raise ValueError("Unexpected model tensor / potential future label in Replay")
            for key, (shape, dtype) in {**FIELDS, "stepid": ((20,), np.dtype("uint8"))}.items():
                if batch[key].shape != (batch_size, 65, *shape) or batch[key].dtype != dtype:
                    raise ValueError(f"Sample field shape/dtype changed: {key}")
            joined = np.empty((batch_size, 65, len(columns)), np.uint16)
            for b, ids in enumerate(batch["stepid"]):
                owners, ordinals = [], []
                sampled_starts.add(ids[0].tobytes())
                used_chunks = set()
                for t, stepid in enumerate(ids):
                    chunk_key = stepid[:16].tobytes()
                    owner, base, length = corpus["chunks"][chunk_key]
                    offset = int.from_bytes(stepid[16:].tobytes(), "big")
                    if offset >= length:
                        raise ValueError("Sample escaped the saved chunk length")
                    ordinal = base + offset
                    joined[b, t] = corpus["labels"][owner][ordinal]
                    owners.append(owner)
                    ordinals.append(ordinal)
                    used_chunks.add(chunk_key)
                if len(set(owners)) != 1 or not np.all(np.diff(ordinals) == 1):
                    raise ValueError("Native sequence connected discontinuous financial accounts")
                trace = dataset["traces"][owners[0]]
                if (owners[0] not in members if members is not None else
                        pool_id != "all" and trace["policy_id"] != pool_id):
                    raise ValueError("Wrong strategy was loaded into a policy pool")
                source_counts[group_fn(trace) if group_fn else trace["policy_id"]] += 1
                cross_chunks += int(len(used_chunks) > 1)
            if not np.array_equal(batch["action_mask"][:, :, 1] > .5, joined[:, :, index["legal_add_mask"]].astype(bool)):
                raise ValueError("Sample Mask labels do not match canonical Sidecar")
            counts[n, 0] = joined[:, :1].sum(axis=(0, 1), dtype=np.int64)
            counts[n, 1] = joined[:, 1:].sum(axis=(0, 1), dtype=np.int64)
            fingerprint_batch.update(batch["stepid"].tobytes())
        ram_bytes = sum(c.nbytes for c in replay.chunks.values())
        return dict(pool_id=pool_id, counts=counts, sequence_starts=expected, capacity=expected,
                    ram_bytes=ram_bytes,
                    audit_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    batches=batches, batch_size=batch_size, seed=seed,
                    sequence_draws=batches * batch_size, unique_sampled_starts=len(sampled_starts),
                    repeated_starts=batches * batch_size - len(sampled_starts), cross_chunk_sequences=cross_chunks,
                    sampled_policy_sequences=dict(source_counts), sample_stepid_hash=fingerprint_batch.hexdigest(),
                    first_batch_reproduced=reproducible, selector=type(replay.sampler).__name__,
                    excluded_verification_batches=1, discontinuous_sequences=0)
    finally:
        replay.workers.shutdown(wait=True)


def projection(sequence_starts, records, interactions, budget, *, train_ratio=64, batch_size=16):
    updates = budget * train_ratio / (batch_size * 64)
    draws = updates * batch_size
    n = sequence_starts
    unique = 1. if n == 1 else -n * math.expm1(draws * math.log1p(-1 / n))
    return dict(interaction_budget=budget, projected_gradient_updates=updates,
        projected_sequence_draws=draws, model_timesteps=budget * train_ratio,
        sampled_rows_with_context=draws * 65, mean_visits_per_sequence_start=draws / n,
        expected_unique_sequence_starts=unique, expected_repeated_starts=draws - unique,
        timestep_uses_per_stored_row=budget * train_ratio / records,
        timestep_uses_per_real_interaction=budget * train_ratio / interactions)


def table(headers, rows):
    def cell(value):
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "/")
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(cell(v) for v in row) + " |" for row in rows]


def write_coverage(path, root, dataset, inputs, corpus, samples):
    totals = Counter()
    for summary in corpus["summaries"].values():
        totals.update(summary)
    source_valid = sum(s["valid_minutes"] for s in inputs["sources"] if s["contract"] in dataset["identity"]["selected_contracts"])
    lines = ["# FG Replay R0～R2 覆盖审计", "", f"数据目录：`{Path(root).resolve()}`。",
        f"Dataset manifest SHA256：`{digest(Path(root) / 'manifest.json')}`。",
        "", "本报告只证明数据正确性和实测覆盖；World Model/Actor 学习充分性未验证。未创建 Agent、未启动 GPU、未进行梯度更新。",
        "", "## 数据与账户完整性", "",
        f"{len(dataset['traces'])} 条账户轨迹全部逐步重放对账 PASS；真实账户导出/恢复 {corpus['account_restores']} 次。",
        "零非法 ADD 请求；每组及混合池均验证连续序列账户身份、dtype、形状和固定 RNG 完整 batch 复现。",
        "", *table(["量", "实测值"], [
            ("冻结源文件有效唯一行情分钟", source_valid),
            ("实际推进访问的唯一行情分钟", corpus["unique_advanced_minutes"]),
            ("模拟有效分钟交互", totals["valid_minute"]), ("全部实际推进交互（含元数据）", totals["advanced"]),
            ("Replay 记录（含重置和终局观察）", totals["records"]),
            ("非推进行（初始、重置、数据终止观察）", totals["non_advancing_rows"]),
            ("Env 截断重置", totals["chunk_reset"]), ("有效序列起点", totals["sequence_starts"]),
            ("原始轨迹文件总字节（硬链接视图不重复计算）", corpus["trace_bytes"]),
            ("实际行情重用次数分布", corpus["market_reuse_histogram"])]),
        "", "唯一行情以 contract+分钟去重。多策略重复模拟是相关反事实账户路径，不是独立市场样本；初始行情行不属于推进交互。",
        "", "## 分策略覆盖", ""]
    names = ["decision", "legal_add_decision", "add_requested", "illegal_add", "add_fill", "add_rejected", "voluntary_wait", "restricted_wait", "full_wait", "tp", "non_tp_exit", "early_first5", "early_next10", "positive_reward"]
    lines += table(["policy", *names], [(p, *[v[k] for k in names]) for p, v in corpus["summaries"].items()])
    lines += ["", "decision 是下一步实际推进的请求，含零时长元数据；valid_decision 单独排除元数据。合法请求仍可能因下一根成交量、方向、价格或资金而拒单。", ""]
    lines += table(["policy", "真实Env拒单原因", "次数"], [
        (*key, count) for key, count in sorted(corpus["rejection_reasons"].items())])
    lines += ["", "ADD请求属于 outgoing action；成交和拒单属于下一条反馈行，分别统计。未推进的终止请求不计为已执行或拒单。", ""]
    state_rows = []
    for p, summary in corpus["summaries"].items():
        state_rows.append((p, summary["empty"] / summary["decision"], summary["lots3"] / summary["decision"],
            summary["lots6"] / summary["decision"], summary["positive_reward"] / summary["advanced"],
            summary["valid_decision"], summary["open_entries"], summary["unvisited_rows"]))
    lines += table(["policy", "空仓占比", "3手占比", "6手占比", "正Reward/推进交互", "有效分钟决策", "终局未平仓笔数", "未访问源行"], state_rows)
    lines += ["", "## 逐轨迹终局与跨 chunk 事件", ""]
    fields = ["policy_id", "contract", "records", "advanced", "chunk_resets", "account_restores", "TP", "non_tp_exits", "open_entries", "unvisited_rows", "status"]
    lines += table(fields, [[r[k] for k in fields] for r in corpus["trajectories"]])
    lines += ["", *table(["policy", "TP跨Env截断", "TP跨Replay chunk", "终局失败", "强退未成交", "右删失", "TP自然分钟median/p90", "逐笔MAE median/p90"], [
        (p, sum(t["status"] == "success" and t["cross_env_chunk"] for t in corpus["trades"] if t["policy_id"] == p),
         sum(t["status"] == "success" and t["cross_replay_chunk"] for t in corpus["trades"] if t["policy_id"] == p),
         *[sum(t["status"] == status for t in corpus["trades"] if t["policy_id"] == p) for status in ("terminal_failure", "terminal_failure_unfilled", "right_censored")],
         {k: v for k, v in distribution([t["duration_minutes"] for t in corpus["trades"] if t["policy_id"] == p and t["status"] == "success"]).items() if k in ("median", "p90")},
         {k: v for k, v in distribution([t["MAE_points"] for t in corpus["trades"] if t["policy_id"] == p]).items() if k in ("median", "p90")})
        for p in corpus["summaries"]]),
        "", "MAE 是持仓期间已观察 OHLC 的保守上界；TP 同 bar 内的先后顺序未知。早期逆行来自原诊断器的首次逆行10点事件，前5/6～15根有效bar分开，绝不改变业务 Reward。", "", "## 持仓年龄与市场分层", ""]
    lines += table(["policy", *AGE_COLUMNS], [(p, *[v[k] for k in AGE_COLUMNS]) for p, v in corpus["summaries"].items()])
    lines += ["", "年龄桶按自然分钟；每个实际推进决策的在仓 entry 分别计数，因此该表的分母是持仓观察次数。", ""]
    strata_metrics = ["advanced", "legal_add_decision", "add_requested", "add_fill", "add_rejected", "voluntary_wait", "full_wait", "tp", "non_tp_exit", "early_first5", "early_next10"]
    lines += table(["policy", "dimension", "value", *strata_metrics], [
        (*key, *[v[k] for k in strata_metrics])
        for key, v in sorted(corpus["strata"].items())])
    lines += ["", *table(["policy", "dimension", "value", "empty", "lots3", "lots6", *AGE_COLUMNS], [
        (*key, v["empty"], v["lots3"], v["lots6"], *[v[k] for k in AGE_COLUMNS])
        for key, v in sorted(corpus["strata"].items())])]
    lines += ["", "## 原生 Uniform batch 实测", "", "每池抽样1,000 batch（以运行参数为准），上下文1位与其余64位分开。所有事件按真实 stepid 关联；额外1个复现检查 batch 不计入下表。", ""]
    lines += table(["pool", "batch数", "容量/序列起点", "Replay数组字节", "进程累计峰值RSS字节", "跨存储chunk序列", "唯一采样起点", "重复起点", "策略序列构成"], [
        (s["pool_id"], s["batches"], s["sequence_starts"], s["ram_bytes"], s["audit_process_peak_rss_bytes"], s["cross_chunk_sequences"], s["unique_sampled_starts"], s["repeated_starts"], s["sampled_policy_sequences"]) for s in samples])
    lines += ["", "Replay数组内存按真实chunk的nbytes汇总；累计峰值RSS来自Linux审计进程，包括索引、标签、临时batch及前序池，不解释为某池独占内存。", ""]
    batch_rows = []
    sparse = []
    for sample in samples:
        for section, section_id in (("context", 0), ("train64", 1)):
            for metric in ("legal_add_mask", "legal_add_decision", "voluntary_wait", "add_fill", "tp", "early_first5", "early_next10", "lots6"):
                values = sample["counts"][:, section_id, INDEX[metric]]
                batch_rows.append((sample["pool_id"], section, metric, int(values.sum()),
                    float(values.mean()), float(np.median(values)), float(np.quantile(values, .9)),
                    int(values.min()), int(values.max()), float((values == 0).mean())))
                if section == "train64" and metric in ("tp", "early_first5", "early_next10") and (values == 0).mean() >= .5:
                    sparse.append((sample["pool_id"], metric, float((values == 0).mean())))
    lines += ["", *table(["pool", "positions", "metric", "total", "mean/batch", "median", "p90", "min", "max", "零事件batch比例"], batch_rows),
        "", "## 100K / 1M 资源推算（不训练）", "", "按现有FG train_ratio=64、batch_size=16、batch_length=64推算；不计 Gate warmup 的额外路径或模型预算，不把推算当作已发生训练。", ""]
    projections = [projection(totals["sequence_starts"], totals["records"], totals["advanced"], b) for b in (100000, 1000000)]
    lines += table(["交互预算", "梯度更新", "序列抽样", "模型时间位置", "含context读取行", "每起点平均访问", "预期不同起点", "预期重复起点", "模型位置/存储行", "模型位置/真实交互"], [
        [r[k] for k in ("interaction_budget", "projected_gradient_updates", "projected_sequence_draws", "model_timesteps", "sampled_rows_with_context", "mean_visits_per_sequence_start", "expected_unique_sequence_starts", "expected_repeated_starts", "timestep_uses_per_stored_row", "timestep_uses_per_real_interaction")] for r in projections])
    lines += ["", "混合池按有效连续起点 Uniform；磁盘完整语料无容量淘汰。复用时模型 seed 和 Replay 抽样 RNG 独立，源 Replay 不写入训练 Checkpoint 或在线数据。", "", "## 覆盖结论与限制", "",
        "数据正确性：PASS。覆盖充分到可以训练出有效 World Model/Actor：NOT_ESTABLISHED。",
        f"实测稀疏项（train64 零事件batch比例≥50%，仅为报告标记，不是模型 Gate）：`{sparse}`。",
        "如需要增加稀有事件曝光，下一项可比较固定额外P2种子或显式混合比例；本轮不筛盈利样本、不拼接孤立时刻、不自动调整策略。",
        "验证/测试价格未读取；方向是现有人工配置的条件输入。OHLC撮合属于原Env保守模拟，不是经纪商实盘成交。"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_dataset(root, output, dreamer_root, *, batches=1000, batch_size=16, seed=202):
    cpu_runtime(dreamer_root)
    dataset = validate_dataset(root)
    inputs = read_manifest(Path(root) / "inputs.json")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    corpus = corpus_statistics(root, dataset)
    samples = []
    for pool_id in dataset["views"]:
        sample = sample_statistics(root, dataset, corpus, pool_id,
                                    batches=batches, batch_size=batch_size, seed=seed)
        np.savez_compressed(output / f"{pool_id}_batch_counts.npz", counts=sample["counts"], columns=np.array(COLUMNS))
        samples.append(sample)
        print(f"Native Uniform audit {pool_id}: sequences={sample['sequence_starts']} batches={batches}", flush=True)
    # Trade-level diagnostic data; the human result/report remains Markdown.
    write_json(output / "trade_diagnostics.json", corpus["trades"])
    import jax
    platforms = [device.platform for device in jax.devices()]
    if not platforms or any(platform != "cpu" for platform in platforms):
        raise RuntimeError(f"Audit used non-CPU devices: {platforms}")
    # Recheck the source corpus before publishing any completion or PASS result.
    validate_dataset(root)
    write_coverage(output / "replay_coverage.md", root, dataset, inputs, corpus, samples)
    receipt = signed(dict(schema="FG_REPLAY_COVERAGE_V1", dataset_manifest_sha256=digest(Path(root) / "manifest.json"),
        batches=batches, batch_size=batch_size, batch_length=64, replay_context=1, sample_seed=seed,
        jax_platforms=platforms, gradient_updates=0,
        audit_code_sha256=digest(__file__),
        pools=[{k: v for k, v in sample.items() if k != "counts"} for sample in samples],
        files={p.name: digest(p) for p in output.iterdir() if p.is_file()}))
    write_json(output / "manifest.json", receipt)
    return receipt
