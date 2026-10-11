"""CPU-only, event-sourced FG Replay construction through the original Driver.

No Agent is constructed. Financial execution and business reward belong solely
to FixedDirectionEntryEnv; audit labels are kept outside model observations.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from gym_trading_env.envs.entry_contract import fingerprint, read_yaml, resolve_path
from .fg_history_study import digest
from .fg_reward_candidates import CandidateConfig, FGRewardScorer


SCHEMA = "FG_REPLAY_R0_R2_V1"
SINGLE3_SCHEMA = "FG_SINGLE3_REPLAY_V1"
FIELDS = {
    "market_seq": ((60, 18), np.dtype("float32")),
    "position_state": ((9,), np.dtype("float32")),
    "action_mask": ((2,), np.dtype("float32")),
    "action": ((), np.dtype("int32")),
    "reward": ((), np.dtype("float32")),
    "is_first": ((), np.dtype("bool")),
    "is_last": ((), np.dtype("bool")),
    "is_terminal": ((), np.dtype("bool")),
}


def cpu_runtime(dreamer_root):
    """Set CPU constraints before importing Embodied (which imports JAX)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    root = str(Path(dreamer_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import embodied
    if Path(embodied.__file__).resolve().parent != Path(root) / "embodied":
        raise ValueError("Embodied was imported from a different Dreamer checkout")
    return embodied


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_manifest(path):
    data = json.loads(Path(path).read_text())
    body = {k: v for k, v in data.items() if k != "content_hash"}
    if data.get("content_hash") != fingerprint(body):
        raise ValueError(f"Manifest integrity mismatch: {path}")
    return data


def signed(data):
    return dict(data, content_hash=fingerprint(data))


def check_files(files):
    for filename, expected in files.items():
        if not Path(filename).is_file() or digest(filename) != expected:
            raise ValueError(f"Frozen input/output changed: {filename}")


def validate_training(manifest_path):
    """Check actual train bytes/days, without opening held-out price files."""
    manifest_path = Path(manifest_path).resolve()
    data = read_manifest(manifest_path)
    roles = data["roles"]
    if (data["product"] != "FG" or not roles["train"] or
            not max(roles["train"]) < min(roles["validation"]) <=
            max(roles["validation"]) < min(roles["test"])):
        raise ValueError("FG roles must be nonempty, disjoint and chronological")
    rows = sorted((r for r in data["files"] if r["role"] == "train" and
                   r["eligible_for_training"]), key=lambda r: r["first_timestamp"])
    if not rows or len({r["contract"] for r in rows}) != len(rows):
        raise ValueError("Expected unique complete training lifecycles")
    files = {str(manifest_path): digest(manifest_path),
             data["direction_path"]: data["direction_sha256"]}
    source = []
    train_days = set(roles["train"])
    heldout_days = set(roles["validation"]) | set(roles["test"])
    for row in rows:
        if not row["complete_lifecycle"] or not row["calendar_complete"]:
            raise ValueError("Incomplete training lifecycle")
        pairs = {row[k]: row[h] for k, h in
                 (("path", "sha256"), ("env_config", "env_sha256"),
                  ("contract_config", "contract_sha256"))}
        check_files(pairs)
        files.update(pairs)
        env = read_yaml(row["env_config"])
        entry = env["entry"]
        if (env["strategy_mode"] != "entry_only" or
                entry["contract"] != row["contract"] or entry.get("state_path") or
                Path(env["trading"]["data_path"]).resolve() != Path(row["path"]).resolve() or
                resolve_path(row["env_config"], entry["contract_config"]) !=
                Path(row["contract_config"]).resolve() or
                resolve_path(row["env_config"], entry.get("strategy_config", entry.get("direction_config"))) !=
                Path(data["direction_path"]).resolve()):
            raise ValueError("Training environment references unfrozen inputs")
        frame = pd.read_csv(row["path"], usecols=["Date", "bob", "trading_date", "symbol"])
        days = pd.to_datetime(frame.trading_date.astype(str)).dt.strftime("%Y%m%d").astype(int)
        times = pd.to_datetime(frame.Date)
        starts = pd.to_datetime(frame.bob)
        timezone = env["trading"].get("data_tz", "Asia/Shanghai")
        times = times.dt.tz_localize(timezone) if times.dt.tz is None else times.dt.tz_convert(timezone)
        starts = starts.dt.tz_localize(timezone) if starts.dt.tz is None else starts.dt.tz_convert(timezone)
        durations = (times - starts).dt.total_seconds()
        if (len(frame) != row["rows"] or set(frame.symbol) != {row["contract"]} or
                not times.is_monotonic_increasing or times.duplicated().any() or
                not set(days) <= train_days or set(days) & heldout_days or
                not durations.isin([0, 60]).all() or
                times.iloc[0] != pd.Timestamp(row["first_timestamp"]) or
                times.iloc[-1] != pd.Timestamp(row["last_timestamp"])):
            raise ValueError("Actual CSV violates frozen training time/row boundaries")
        source.append(dict(contract=row["contract"], rows=len(frame),
                           valid_minutes=int((durations == 60).sum()),
                           metadata_rows=int((durations == 0).sum()),
                           first_trading_day=int(days.min()), last_trading_day=int(days.max())))
    check_files(files)
    return data, rows, files, source


def provenance(dreamer_root):
    """Hash execution dependencies, including dirty bytes rather than HEAD only."""
    project = Path(__file__).resolve().parents[3]
    dreamer = Path(dreamer_root).resolve()
    paths = []
    for directory in (project / "src/gym_trading_env/envs",
                      project / "src/gym_trading_env/utils"):
        paths.extend(directory.rglob("*.py"))
    paths += [project / "src/gym_trading_env/rewards/fast_take_profit_reward.py",
              Path(__file__), project / "tools/fg_replay.py",
              Path(__file__).with_name("fg_reward_candidates.py"),
              dreamer / "embodied/core/driver.py", dreamer / "embodied/core/replay.py",
              dreamer / "embodied/core/chunk.py", dreamer / "embodied/core/selectors.py",
              dreamer / "embodied/core/wrappers.py", dreamer / "embodied/envs/from_gymnasium.py"]
    versions = {}
    for root in (project, dreamer):
        versions[str(root)] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return dict(git_heads=versions, code_files={str(p): digest(p) for p in sorted(set(paths))},
                python=sys.version, numpy=np.__version__, pandas=pd.__version__)


class BehaviorPolicy:
    """Only current real mask and previously observed execution events are used."""

    def __init__(self, spec):
        self.spec = spec
        self.rng = np.random.default_rng(np.random.SeedSequence(spec["seed_components"]))
        self.waited = 0
        self.waiting = spec["kind"] == "capacity_delay"

    def observe(self, *, advanced, valid_bar, interval_slots, lots_after):
        if self.spec["kind"] != "capacity_delay" or not advanced:
            return
        if interval_slots == 2:
            self.waited = 0
            self.waiting = True  # Includes an intra-bar full->free TP release.
        elif self.waiting and valid_bar and lots_after < 6:
            self.waited += 1
        if self.waited >= self.spec["wait_minutes"]:
            self.waiting = False

    def __call__(self, carry, obs):
        active = not bool(obs["is_last"][0])
        legal = bool(obs["action_mask"][0, 1])
        action = 0
        if active and legal:
            if self.spec["kind"] == "always_add":
                action = 1
            elif self.spec["kind"] == "probabilistic":
                action = int(self.rng.random() < self.spec["probability"])
            elif self.spec["kind"] == "capacity_delay":
                action = int(not self.waiting)
            else:
                raise ValueError("Unknown causal behavior policy")
        return carry, {"action": np.array([action], np.int32)}, {}


def policy_specs(config, contract_index):
    specs = []
    for index, original in enumerate(config["policies"]):
        spec = dict(original, seed_components=[int(config["seed"]), contract_index, index])
        if (spec["id"] == "all" or not spec["id"] or
                Path(spec["id"]).name != spec["id"] or spec["id"] in (".", "..")):
            raise ValueError("Policy ID must be a plain unique directory name")
        if spec["kind"] == "probabilistic" and not 0 <= spec["probability"] <= 1:
            raise ValueError("Invalid ADD probability")
        if spec["kind"] == "capacity_delay" and spec["wait_minutes"] <= 0:
            raise ValueError("Delay must contain positive valid minutes")
        if spec["kind"] not in {"always_add", "probabilistic", "capacity_delay"}:
            raise ValueError("Unknown policy")
        specs.append(spec)
    if len({s["id"] for s in specs}) != len(specs):
        raise ValueError("Policy IDs must be unique")
    return specs


def account_core(env):
    return (float(env.net_lots), copy.deepcopy(env.entries), env.ledger.snapshot(),
            env.total_reward, set(env.reward_function.rewarded))


def execute_observed(env, action=None, *, seed=None):
    """Collect evidence around one ORIGINAL Env reset/step; no execution logic."""
    before = env.current_step
    lots = float(env.net_lots)
    legal = bool(env._published_mask[1])
    fills, attrs, rewards = len(env.fills), len(env.attributions), len(env.reward_events)
    signal = str(env._raw.index[before])
    if action is None:
        core = account_core(env)
        continuing = env.status == "CHUNK_END"
        obs, info = env.reset(seed=seed)
        if continuing and core != account_core(env):
            raise RuntimeError("Chunk resume changed the financial account")
        reward, terminal, truncated = 0., False, False
    else:
        obs, reward, terminal, truncated, info = env.step(int(action))
    advanced = env.current_step > before
    current = env.current_step
    new_fills = copy.deepcopy(env.fills[fills:])
    new_exits = copy.deepcopy(env.attributions[attrs:])
    openings = [f for f in new_fills if f["reason"] == "ENTRY"]
    interval_slots = int(lots / 3) + len(openings)
    valid = bool(advanced and env.valid_fill_clock[current])
    bar = env._raw.iloc[current]
    direction = env.direction_at(env._raw.index[current], current)
    unit = dict(bar_open=str(env.bobs[current]), bar_end=str(env._raw.index[current]),
                **{k.lower(): float(bar[k]) for k in ("Open", "High", "Low", "Close")},
                volume=float(bar.Volume), open_interest=float(bar.OpenInterest),
                valid_bar=valid, eligible_market_minute=bool(valid and direction != "NO_TRADE"),
                entries=openings, exits=new_exits, chunk_end=bool(truncated and env.status == "CHUNK_END"),
                true_terminal=bool(terminal))
    row = dict(cursor=current, time=str(env._raw.index[current]),
               signal_time=signal if action is not None else None,
               trading_date=env._trading_day(env._raw.index[current]),
               direction=direction, session="night" if env._raw.index[current].hour >= 21 else "day",
               reset=action is None, advanced=advanced, valid_bar=valid,
               incoming_action=None if action is None else int(action), legal_add_before=legal,
               lots_before=lots, lots_after=float(env.net_lots), interval_slots=interval_slots,
               mark_price=str(env.current_price), active_entries=copy.deepcopy(list(env.entries.values())),
               fills=new_fills, exits=new_exits, reward_events=copy.deepcopy(env.reward_events[rewards:]),
               official_reward=reward, status=info["status"], reject_reason=info.get("reject_reason"),
               bar=unit if advanced else None,
               is_first=action is None, is_last=bool(terminal or truncated), is_terminal=bool(terminal))
    return obs, row


def update_array_hash(hasher, step):
    for key in sorted(FIELDS):
        value = np.asarray(step[key])
        shape, dtype = FIELDS[key]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"Invalid Replay field {key}: {value.shape}/{value.dtype}")
        hasher.update(key.encode() + value.tobytes())


def iter_sidecar(directory):
    with gzip.open(Path(directory) / "steps.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def trace_manifest(directory, *, verify=True, schema=SCHEMA):
    directory = Path(directory)
    data = read_manifest(directory / "manifest.json")
    expected_reward = 'FG_SINGLE3_REWARD_V1' if schema==SINGLE3_SCHEMA else 'V1'
    if schema not in (SCHEMA,SINGLE3_SCHEMA) or data["schema"] != schema or data["reward_version"] != expected_reward:
        raise ValueError("Incompatible Replay/Reward identity")
    if verify:
        check_files({str(directory / p): h for p, h in data["files"].items()})
    return data


def iter_replay_rows(directory, manifest=None):
    directory = Path(directory)
    manifest = manifest or trace_manifest(directory)
    ordinal = 0
    seen = set()
    for n, chunk in enumerate(manifest["chunks"]):
        if chunk["uuid"] in seen or (n and manifest["chunks"][n - 1]["succ"] != chunk["uuid"]):
            raise ValueError("Broken/repeated account chunk chain")
        seen.add(chunk["uuid"])
        with np.load(directory / chunk["path"], allow_pickle=False) as data:
            if set(data.files) != set(FIELDS) | {"stepid"}:
                raise ValueError("Replay field set changed")
            arrays = {k: data[k] for k in data.files}
        for k, (shape, dtype) in {**FIELDS, "stepid": ((20,), np.dtype("uint8"))}.items():
            if arrays[k].shape != (chunk["length"], *shape) or arrays[k].dtype != dtype:
                raise ValueError(f"Saved Replay dtype/shape mismatch: {k}")
        for index in range(chunk["length"]):
            step = {k: v[index] for k, v in arrays.items()}
            # UUID text is Embodied's base32 spelling; compare through its real type.
            import elements
            expected_id = bytes(elements.UUID(chunk["uuid"])) + index.to_bytes(4, "big")
            if step["stepid"].tobytes() != expected_id:
                raise ValueError("Replay stepid is not its actual chunk/offset")
            yield ordinal, step
            ordinal += 1
    if ordinal != manifest["records"]:
        raise ValueError("Saved Replay record count mismatch")


def collect_trace(config_path, spec, directory, identity, dreamer_root, *, env_factory=None,
                  protocol='fg_entry_v1', policy_factory=None, runtime_factory=None):
    """One complete independent Driver stream. Never reload an earlier account."""
    embodied = (runtime_factory or cpu_runtime)(dreamer_root)
    from embodied.envs.from_gymnasium import FromGymnasium
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    import gymnasium as gym
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    raw = env_factory() if env_factory else FixedDirectionEntryEnv(config_path=str(config_path))
    print(f"Collect {spec['id']} {raw.contract}: {len(raw._raw)} source rows, CPU Driver", flush=True)
    single3 = protocol == 'fg_single3_v1'
    if protocol not in ('fg_entry_v1','fg_single3_v1') or raw.single3 != single3:
        raise ValueError('Explicit Replay protocol must match the actual Env')
    policy = (policy_factory or BehaviorPolicy)(spec)
    scorer = raw._new_reward() if single3 else FGRewardScorer(CandidateConfig())

    class Observed(gym.Wrapper):
        def reset(self, *, seed=None, options=None):
            if options is not None:
                raise ValueError("Use a new trace for another account")
            obs, self.last = execute_observed(self.env, seed=seed)
            return obs, self.env._get_info()

        def step(self, action):
            obs, self.last = execute_observed(self.env, action)
            if single3:
                # CHUNK_END hides actions administratively; it is not a loss of
                # the known market/account opportunity after continuation.
                policy.observe_row(self.last, legal_after=self.env._can_add(continuing=True)
                    if self.env.status=='CHUNK_END' else bool(obs['action_mask'][1]))
            else:
                policy.observe(advanced=self.last["advanced"], valid_bar=self.last["valid_bar"],
                               interval_slots=self.last["interval_slots"], lots_after=self.last["lots_after"])
            return obs, self.last["official_reward"], self.last["is_terminal"], self.env.truncated, self.env._get_info()

    observed = Observed(raw)
    env = embodied.wrappers.CheckSpaces(embodied.wrappers.UnifyDtypes(FromGymnasium(observed, reset_seed=0)))
    driver = embodied.Driver([lambda: env], parallel=False)
    replay = embodied.Replay(length=65, capacity=None, directory=str(directory / "replay"),
                             chunksize=1024, online=False, save_wait=True, seed=202)
    state = dict(records=0, advanced=0, valid_minutes=0, chunk_resets=0, pending=None)
    array_hash, event_hash = hashlib.sha256(), hashlib.sha256()

    def emit(row, stream):
        canonical = {k: v for k, v in row.items() if k not in ("stepid", "ordinal")}
        event_hash.update(json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")

    try:
        with gzip.open(directory / "steps.jsonl.gz", "wt", encoding="utf-8") as stream:
            def capture(step, worker):
                row = copy.deepcopy(observed.last)
                if row["advanced"]:
                    diagnostics = scorer.step(row['bar'],raw.terms.multiplier) if single3 else scorer.step(row["bar"])
                    score = diagnostics['total'] if single3 else diagnostics['v0']
                    if not np.isclose(score, row["official_reward"], rtol=0, atol=1e-9):
                        raise RuntimeError("Committed TP events disagree with Env V1 reward")
                    row["diagnostics"] = {k: diagnostics[k] for k in
                        ("early_first5", "early_next10", "penalized", "intrabar_ambiguous")}
                else:
                    row["diagnostics"] = dict(early_first5=0, early_next10=0, penalized=[], intrabar_ambiguous=[])
                row.update(ordinal=state["records"], action=int(step["action"]),
                           legal_add=bool(step["action_mask"][1]),
                           p3_waited=policy.waited, p3_waiting=policy.waiting)
                if hasattr(policy, 'last_diagnostics'):
                    row['policy_diagnostics'] = copy.deepcopy(policy.last_diagnostics)
                replay.add(step, worker=worker)
                chunkid, offset = replay.streams[worker][-1]
                row["stepid"] = (bytes(chunkid) + offset.to_bytes(4, "big")).hex()
                update_array_hash(array_hash, step)
                if state["pending"] is not None:
                    previous = state["pending"]
                    previous.update(decision_executed=row["advanced"], decision_valid_minute=row["valid_bar"])
                    emit(previous, stream)
                state["pending"] = row
                state["records"] += 1
                if state["records"] % 16384 == 0:
                    print(f"Collect {spec['id']} {raw.contract}: {state['records']} records", flush=True)
                state["advanced"] += int(row["advanced"])
                state["valid_minutes"] += int(row["valid_bar"])
                state["chunk_resets"] += int(row["reset"] and row["cursor"] > 0)

            driver.on_step(capture)
            while True:
                driver(policy, steps=1)
                if raw.terminated or raw.status in ("DATA_END", "DATA_GAP"):
                    break
            state["pending"].update(decision_executed=False, decision_valid_minute=False)
            emit(state["pending"], stream)
        replay.save()
        replay.workers.shutdown(wait=True)
        chunks = [dict(path="replay/" + c.filename, uuid=str(c.uuid), succ=str(c.succ), length=c.length)
                  for c in replay.chunks.values() if c.length]
        write_json(directory / "account_state.json", raw.export_state())
        trades = scorer.scorer.trade_rows() if single3 else scorer.trade_rows()
        for trade in trades:
            if trade["status"] == "right_censored" and raw.pending_reason:
                trade["status"] = "terminal_failure_unfilled"
        write_json(directory / "trade_diagnostics.json", trades)
        source_rows = len(raw._raw)
        payload = dict(schema=SINGLE3_SCHEMA if single3 else SCHEMA, identity=identity, policy=spec, config_path=str(config_path),
                       reward_version='FG_SINGLE3_REWARD_V1' if single3 else "V1",
                       reward_params_hash=raw.reward_config.hash if single3 else fingerprint(dict(formula="60/(60+T_minutes)", quantity=3, tp_points=5)),
                       obs_version='FG_SINGLE3_V1_60x18_9_2' if single3 else "FG_ENTRY_V1_60x18_9_2", env_config_hash=raw.config_hash,
                       contract=raw.contract, records=state["records"], advanced=state["advanced"],
                       valid_minutes=state["valid_minutes"], chunk_resets=state["chunk_resets"],
                       sequence_starts=max(0, state["records"] - 64), chunks=chunks,
                       source_rows=source_rows, unvisited_rows=source_rows - raw.current_step - 1,
                       terminal_status=raw.status, terminal=raw.terminated,
                       pending_reason=raw.pending_reason,
                       open_entries=len(raw.entries), open_lots=float(raw.net_lots), total_reward=raw.total_reward,
                       tp_events=sum(e['reason']=='TAKE_PROFIT' for e in raw.attributions) if single3 else len(raw.reward_events),
                       non_tp_exits=sum(e["reason"] != "TAKE_PROFIT" for e in raw.attributions),
                       array_hash=array_hash.hexdigest(), event_hash=event_hash.hexdigest(),
                       elapsed_seconds=time.monotonic() - began,
                       files={str(p.relative_to(directory)): digest(p) for p in directory.rglob("*") if p.is_file()})
        if single3:
            payload.update(protocol=protocol,reward_candidate=raw.reward_config.candidate,
                           fill_model=raw.fill_model, observation_identity=raw.observation_identity,
                           sl_events=sum(e['reason']=='STOP_LOSS' for e in raw.attributions),
                           early10_events=sum(e['reason']=='FIRST_ADVERSE10' for e in raw.reward_events))
        payload["replay_trace_hash"] = fingerprint({k: payload[k] for k in
            ("identity", "policy", "env_config_hash", "reward_version", "array_hash", "event_hash")})
        write_json(directory / "manifest.json", signed(payload))
        return payload
    finally:
        replay.workers.shutdown(wait=True)
        driver.close()


def reconcile_trace(directory, dreamer_root, *, env_factory=None, schema=SCHEMA, runtime_factory=None):
    """Replay frozen actions through a second original Env, checking every row."""
    (runtime_factory or cpu_runtime)(dreamer_root)
    from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
    directory = Path(directory)
    manifest = trace_manifest(directory,schema=schema)
    raw = env_factory() if env_factory else FixedDirectionEntryEnv(config_path=manifest["config_path"])
    if raw.single3 != (schema==SINGLE3_SCHEMA):
        raise ValueError('Replay/Env protocol mismatch')
    hasher, event_hash = hashlib.sha256(), hashlib.sha256()
    pending = None
    sidecar = iter(iter_sidecar(directory))
    count, restores = 0, 0
    try:
        for ordinal, step in iter_replay_rows(directory, manifest):
            row = next(sidecar, None)
            if row is None or row["ordinal"] != ordinal or row["stepid"] != step["stepid"].tobytes().hex():
                raise ValueError("Sidecar row identity mismatch")
            incoming = None if pending is None or pending["is_last"] else pending["action"]
            obs, expected = execute_observed(raw, incoming, seed=0 if ordinal == 0 else None)
            if incoming != row["incoming_action"]:
                raise ValueError("Driver action/feedback alignment mismatch")
            for key in expected:
                if expected[key] != row[key]:
                    raise ValueError(f"Env/Sidecar row {ordinal} differs: {key}")
            for key, value in obs.items():
                np.testing.assert_array_equal(step[key], value, err_msg=f"row {ordinal}: {key}")
            for key in ("is_first", "is_last", "is_terminal"):
                if bool(step[key]) != expected[key]:
                    raise ValueError(f"Driver boundary mismatch: {ordinal}/{key}")
            if step["reward"] != np.float32(expected["official_reward"]):
                raise ValueError("Replay feedback reward shifted or changed")
            if int(step["action"]) != row["action"] or (row["action"] == 1 and not row["legal_add"]):
                raise ValueError("Replay action changed or violates real Mask")
            if pending is not None:
                if (pending["decision_executed"] != row["advanced"] or
                        pending["decision_valid_minute"] != row["valid_bar"]):
                    raise ValueError("Decision denominator includes a reset/non-advancing row")
            update_array_hash(hasher, step)
            canonical = {k: v for k, v in row.items() if k not in ("stepid", "ordinal")}
            event_hash.update(json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            if expected["status"] == "CHUNK_END":
                # Exercise ORIGINAL import_state too, not just in-process reset.
                before = raw.export_state()
                raw.reset(options={"resume": json.loads(json.dumps(before))})
                if raw.export_state() != before:
                    raise RuntimeError("Export/import changed the complete financial state")
                restores += 1
            pending = row
            count += 1
            if count % 16384 == 0:
                print(f"Reconcile {manifest['policy']['id']} {raw.contract}: {count} records", flush=True)
        if next(sidecar, None) is not None or pending["decision_executed"]:
            raise ValueError("Unmatched Sidecar tail")
        expected_final = json.loads((directory / "account_state.json").read_text())
        if raw.export_state() != expected_final:
            raise ValueError("Final account/ledger/TP identity differs")
        if hasher.hexdigest() != manifest["array_hash"] or event_hash.hexdigest() != manifest["event_hash"]:
            raise ValueError("Canonical trace content hash mismatch")
        return dict(status="PASS", records=count, account_restores=restores,
                    replay_trace_hash=manifest["replay_trace_hash"])
    finally:
        raw.close()


def generate_job(job):
    """Spawn-safe job; only this job's attempt directory can be written."""
    check_files(job["frozen_files"])
    base = Path(job["directory"])
    if job["resume"] and base.exists():
        for attempt in sorted(base.glob("attempt_*"), reverse=True):
            if (attempt / "manifest.json").exists() and (attempt / "reconciliation.json").exists():
                trace = trace_manifest(attempt)
                receipt = read_manifest(attempt / "reconciliation.json")
                if trace["identity"] != job["identity"] or trace["policy"] != job["policy"]:
                    raise ValueError("Completed trace resume identity mismatch")
                if receipt["trace_manifest_sha256"] != digest(attempt / "manifest.json"):
                    raise ValueError("Trace reconciliation receipt changed")
                return dict(directory=str(attempt), trace=trace, resumed=True)
    attempt_index = len(list(base.glob("attempt_*"))) if base.exists() else 0
    directory = base / f"attempt_{attempt_index:03}"
    trace = collect_trace(job["env_config"], job["policy"], directory, job["identity"], job["dreamer_root"])
    result = reconcile_trace(directory, job["dreamer_root"])
    check_files(job["frozen_files"])
    write_json(directory / "reconciliation.json", signed(dict(result,
        trace_manifest_sha256=digest(directory / "manifest.json"))))
    return dict(directory=str(directory), trace=trace, resumed=False)


def generate_dataset(config_path, output, dreamer_root, *, workers=1, resume=False, contracts=None):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    cpu_runtime(dreamer_root)
    if workers < 1:
        raise ValueError("workers must be positive")
    config_path = Path(config_path).resolve()
    config = read_yaml(config_path)
    if config.get("schema") != SCHEMA or config.get("seed", -1) < 0:
        raise ValueError("Invalid Replay configuration/seed")
    policy_specs(config, 0)  # Validate before creating a run directory.
    dataset, rows, inputs, sources = validate_training(config["train_manifest"])
    if dataset["content_hash"] != config.get("expected_manifest_hash"):
        raise ValueError("Training manifest differs from the explicitly frozen fold")
    code = provenance(dreamer_root)
    frozen = {**inputs, **code["code_files"], str(config_path): digest(config_path)}
    if contracts:
        if set(contracts) - {r["contract"] for r in rows}:
            raise ValueError("Requested contract is outside the eligible train set")
    selected = [(i, r) for i, r in enumerate(rows) if not contracts or r["contract"] in contracts]
    root = Path(output).resolve()
    identity = dict(schema=SCHEMA, dataset_manifest_hash=dataset["content_hash"],
                    config_sha256=digest(config_path), code_hash=fingerprint(code["code_files"]),
                    selected_contracts=[r["contract"] for _, r in selected], reward_version="V1")
    if root.exists():
        if not resume:
            raise FileExistsError("Refusing to overwrite an existing Replay run")
        started = read_manifest(root / "inputs.json")
        if started["identity"] != identity or started["frozen_files"] != frozen:
            raise ValueError("Replay resume input/code identity mismatch; use a new output")
        if (root / "manifest.json").exists():
            return validate_dataset(root)
    else:
        root.mkdir(parents=True)
        write_json(root / "inputs.json", signed(dict(identity=identity, frozen_files=frozen,
            provenance=code, sources=sources, config=config,
            cpu_environment={k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS")})))
    jobs = []
    for index, row in selected:
        for spec in policy_specs(config, index):
            jobs.append(dict(directory=str(root / "traces" / spec["id"] / row["contract"]),
                env_config=row["env_config"], policy=spec, identity=identity,
                frozen_files=frozen, dreamer_root=str(Path(dreamer_root).resolve()), resume=resume))
    records = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(generate_job, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
            trace = result["trace"]
            records.append(dict(directory=str(Path(result["directory"]).relative_to(root)),
                                manifest_sha256=digest(Path(result["directory"]) / "manifest.json"),
                                receipt_sha256=digest(Path(result["directory"]) / "reconciliation.json"),
                                policy_id=trace["policy"]["id"], contract=trace["contract"],
                                records=trace["records"], sequence_starts=trace["sequence_starts"],
                                replay_trace_hash=trace["replay_trace_hash"]))
            print(f"[{len(records)}/{len(jobs)}] {trace['policy']['id']} {trace['contract']}: "
                  f"rows={trace['records']} TP={trace['tp_events']} "
                  f"status={trace['terminal_status']} reconciled", flush=True)
    check_files(frozen)
    records.sort(key=lambda r: (r["policy_id"], r["contract"]))
    views = {}
    for policy_id in [p["id"] for p in config["policies"]] + ["all"]:
        view = root / "replay" / policy_id
        view.mkdir(parents=True, exist_ok=True)
        for record in records:
            if policy_id != "all" and record["policy_id"] != policy_id:
                continue
            directory = root / record["directory"]
            trace = trace_manifest(directory)
            for chunk in trace["chunks"]:
                source = directory / chunk["path"]
                target = view / source.name
                if target.exists():
                    if digest(target) != trace["files"][chunk["path"]]:
                        raise ValueError("Existing Replay view changed")
                else:
                    os.link(source, target)
        views[policy_id] = str(view.relative_to(root))
    payload = signed(dict(schema=SCHEMA, identity=identity, reward_version="V1", traces=records,
                          views=views, input_manifest_sha256=digest(root / "inputs.json")))
    # Keep the candidate as failure evidence; only a validated manifest marks completion.
    candidate = root / "manifest.pending.json"
    write_json(candidate, payload)
    validated = validate_dataset(root, manifest_path=candidate)
    candidate.replace(root / "manifest.json")
    return validated


def validate_dataset(root, *, manifest_path=None):
    root = Path(root)
    data = read_manifest(root / "manifest.json" if manifest_path is None else Path(manifest_path))
    inputs = read_manifest(root / "inputs.json")
    if (data["schema"] != SCHEMA or data["reward_version"] != "V1" or
            data["identity"] != inputs["identity"] or
            data["input_manifest_sha256"] != digest(root / "inputs.json")):
        raise ValueError("Dataset identity mismatch")
    check_files(inputs["frozen_files"])
    expected_pairs = {(p["id"], c) for p in inputs["config"]["policies"]
                      for c in data["identity"]["selected_contracts"]}
    actual_pairs = [(r["policy_id"], r["contract"]) for r in data["traces"]]
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        raise ValueError("Dataset is missing a planned strategy/contract account")
    seen = set()
    expected_views = {k: set() for k in data["views"]}
    for record in data["traces"]:
        directory = root / record["directory"]
        check_files({str(directory / "manifest.json"): record["manifest_sha256"],
                     str(directory / "reconciliation.json"): record["receipt_sha256"]})
        trace = trace_manifest(directory)
        receipt = read_manifest(directory / "reconciliation.json")
        if (trace["identity"] != data["identity"] or receipt["status"] != "PASS" or
                receipt["trace_manifest_sha256"] != record["manifest_sha256"] or
                trace["replay_trace_hash"] != record["replay_trace_hash"] or
                trace["records"] != record["records"] or
                trace["sequence_starts"] != record["sequence_starts"] or
                trace["policy"]["id"] != record["policy_id"] or trace["contract"] != record["contract"]):
            raise ValueError("Account trace is not reconciled with this dataset")
        for chunk in trace["chunks"]:
            if chunk["uuid"] in seen:
                raise ValueError("Independent accounts share a Replay chunk")
            seen.add(chunk["uuid"])
            name = Path(chunk["path"]).name
            for view_id in (record["policy_id"], "all"):
                expected_views[view_id].add(name)
                check_files({str(root / data["views"][view_id] / name): trace["files"][chunk["path"]]})
    for view_id, names in expected_views.items():
        if {p.name for p in (root / data["views"][view_id]).glob("*.npz")} != names:
            raise ValueError("Replay view has missing or unrelated chunks")
    return data
