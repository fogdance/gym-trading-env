"""FG-only offline export, dataset manifests and comparable entry policies."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from gym_trading_env.envs.entry_contract import (ContractTerms, DirectionSchedule,
    fingerprint, is_fg, read_yaml, timestamp)
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.research.entry_pipeline import file_hash


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def configured_fg_costs(path):
    """Translate the user's existing FG preset using legacy per-side fee semantics."""
    from decimal import Decimal
    raw=read_yaml(path)["CZCE_FG"]
    leverage=Decimal(str(raw["leverage"]))
    if leverage<=0 or Decimal(str(raw["lot_size"]))<=0:
        raise ValueError("Invalid configured FG leverage or multiplier")
    fee=Decimal(str(raw["trading_fee_per_lot"]))
    close=fee if raw["is_round_turn"] else Decimal(0)
    return dict(source_path=str(Path(path).resolve()),source_sha256=file_hash(path),
        cost_basis="user_configured_constant_not_historical_exchange_schedule",
        lot_size=float(raw["lot_size"]),margin_rate=float(1/leverage),
        open_fee=float(fee),close_fee=float(close),close_today_fee=float(close),
        spread=float(raw["spread"]),slippage=0.0)


def trim_fg_export_minutes(frame):
    """FG export policy: retain 23:00 closes; omit 23:01--23:30 historical bars."""
    times=pd.DatetimeIndex([timestamp(t) for t in frame.Date])
    late=(times.hour==23)&(times.minute>0)
    return frame.loc[~late].copy(), int(late.sum())


def report_fg_truncation(frame, audit_path):
    """Log every affected trading day and persist the exact omitted source bars."""
    times=pd.DatetimeIndex([timestamp(t) for t in frame.Date])
    removed=frame.loc[(times.hour==23)&(times.minute>0)].copy()
    removed['truncation_reason']='FG_345_CLOCK_AFTER_23_00'
    audit_path=Path(audit_path)
    removed.to_csv(audit_path,index=False)
    for (contract,day),part in removed.groupby(['symbol','trading_date'],sort=True):
        print(f'[FG truncate] contract={contract} trading_date={day} '
              f'from={part.Date.iloc[0]} to={part.Date.iloc[-1]} rows={len(part)}',flush=True)
    print(f'[FG truncate] removed={len(removed)} kept_23_00=True audit={audit_path}',flush=True)
    return str(audit_path)


def truncate_existing_fg_export(export_manifest, output):
    """Apply the export policy to an existing snapshot without re-querying DB."""
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite a FG export')
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for item in export_manifest['files']:
        if file_hash(item['path'])!=item['sha256']:
            raise ValueError('Source FG export hash mismatch')
        original=pd.read_csv(item['path'])
        frame,removed=trim_fg_export_minutes(original)
        if frame.empty:
            raise ValueError('FG truncation removed an entire main segment')
        target=output/Path(item['path']).name
        frame.to_csv(target,index=False)
        audit=report_fg_truncation(original,target.with_name(target.stem+'_truncated.csv'))
        record=dict(item,path=str(target),sha256=file_hash(target),rows=len(frame),
            first_timestamp=str(frame.Date.iloc[0]),last_timestamp=str(frame.Date.iloc[-1]),
            source_path=item['path'],source_sha256=item['sha256'],removed_late_night_rows=removed,
            truncation_audit=audit,truncation_audit_sha256=file_hash(audit))
        if item.get('main_exit_at') and timestamp(item['main_exit_at'])!=timestamp(frame.Date.iloc[-1]):
            record['main_exit_at']=record['last_timestamp']
        records.append(record)
    result=dict(export_manifest,files=records,minute_clock=345,
        night_cutoff='23:00',removed_late_night_rows=sum(r['removed_late_night_rows'] for r in records))
    write_json(output/'export_manifest.json',result)
    return result


def export_fg(client, start, end, output, provider="JUEJIN", excluded_contracts=()):
    """Read-only main-contract join; preserve identity and reject ambiguous bars."""
    from gym_trading_env.utils.time_contract import to_db_naive_dt
    start, end = timestamp(start), timestamp(end)
    if start >= end:
        raise ValueError("Export requires an ascending time interval")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing export")
    with client._connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT trading_date,symbol FROM fut_continuous_map_v2
                WHERE csymbol=%s AND provider=%s AND trading_date >= %s AND trading_date < %s
                ORDER BY trading_date""",("CZCE.FG",provider,start.date(),end.date()))
            mapping=pd.DataFrame(cur.fetchall())
            if mapping.empty or mapping.trading_date.duplicated().any():
                raise ValueError("Missing or ambiguous daily main mapping")
            cur.execute("""SELECT b.eob AS Date, b.open AS Open, b.high AS High,
                b.low AS Low, b.close AS Close, b.volume AS Volume,
                b.position AS OpenInterest, b.symbol, b.bob, b.trading_date,
                b.source, b.provider
                FROM fut_continuous_map_v2 m JOIN fut_bar_1m_v2 b
                  ON b.trading_date=m.trading_date AND b.symbol=m.symbol
                WHERE m.csymbol=%s AND m.provider=%s
                  AND m.trading_date >= %s AND m.trading_date < %s
                  AND b.eob >= %s AND b.eob < %s
                ORDER BY b.eob""",
                ("CZCE.FG", provider, start.date(), end.date(), to_db_naive_dt(start), to_db_naive_dt(end)))
            frame = pd.DataFrame(cur.fetchall())
    frame=frame[~frame.symbol.isin(excluded_contracts)] if not frame.empty else frame
    original=frame
    frame,removed_late_night_rows=trim_fg_export_minutes(frame) if not frame.empty else (frame,0)
    mapping=mapping[~mapping.symbol.isin(excluded_contracts)]
    if frame.empty or frame.Date.duplicated().any():
        raise ValueError("Missing or ambiguous main-contract minute bars")
    if any(not is_fg(c) for c in frame.symbol):
        raise ValueError("Main mapping contains a non-real FG contract")
    if (frame.groupby("trading_date").symbol.nunique() != 1).any():
        raise ValueError("Multiple main contracts mapped to one trading day")
    covered=set(frame.trading_date.astype(str))
    missing=set(mapping.trading_date.astype(str))-covered
    if missing:
        raise ValueError(f"Mapped trading days have no exported bars: {sorted(missing)}")
    output.mkdir(parents=True, exist_ok=True)
    audit=report_fg_truncation(original,output/'truncated_late_night.csv')
    mapping_path = output/"main_contract_map.csv"
    mapping.to_csv(mapping_path,index=False)
    records=[]
    runs=frame.symbol.ne(frame.symbol.shift()).cumsum()
    for run, part in frame.groupby(runs,sort=False):
        contract=part.symbol.iloc[0]
        path=output/f"{int(run):04d}_{contract.replace('.', '_')}_main_1m.csv"
        part.to_csv(path,index=False)
        records.append(dict(contract=contract,segment=int(run),path=str(path),sha256=file_hash(path),
            rows=len(part),first_timestamp=str(part.Date.iloc[0]),last_timestamp=str(part.Date.iloc[-1])))
    for previous, following in zip(records, records[1:]):
        previous['main_exit_at']=previous['last_timestamp']
        previous['next_main_contract']=following['contract']
    manifest=dict(schema=1,product="FG",main_only=True,
        source="fut_continuous_map_v2 JOIN fut_bar_1m_v2",mapping_provider=provider,
        mapping_path=str(mapping_path),mapping_sha256=file_hash(mapping_path),
        start=str(start),end=str(end),files=records,excluded_contracts=list(excluded_contracts),
        minute_clock=345,night_cutoff='23:00',removed_late_night_rows=removed_late_night_rows,
        truncation_audit=audit,truncation_audit_sha256=file_hash(audit))
    write_json(output/"export_manifest.json",manifest)
    return manifest


def prepare_main_contracts(export_manifest, costs, output):
    """Materialize user-confirmed complete sessions and main-qualification expiry.

    This calendar is an explicit user declaration, not an independent exchange audit.
    The final data cutoff has no main-expiry evidence and cannot liquidate holdings.
    """
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to replace frozen contract metadata")
    output.mkdir(parents=True,exist_ok=True)
    files=export_manifest['files']
    paths=[]
    for i,item in enumerate(files):
        frame=pd.read_csv(item['path'])
        if file_hash(item['path'])!=item['sha256']:
            raise ValueError("Market changed after export")
        times=pd.DatetimeIndex([timestamp(t) for t in frame.Date])
        if times.has_duplicates or not times.is_monotonic_increasing:
            raise ValueError("Invalid frozen market clock")
        if not np.isfinite(frame[['Open','High','Low','Close','Volume','OpenInterest']].to_numpy()).all():
            raise ValueError("Invalid frozen OHLCVOI")
        if (frame[['Open','High','Low','Close']].to_numpy()%1).any():
            raise ValueError("FG market prices disagree with one-point tick")
        runs=np.r_[True, np.diff(times.asi8)!=60*10**9].cumsum()
        sessions=[]
        for run in np.unique(runs):
            selected=times[runs==run]
            sessions.append(dict(start=str(selected[0]-pd.Timedelta(minutes=1)),end=str(selected[-1])))
        terminal=i+1<len(files) and files[i+1]['contract']!=item['contract']
        # Infinity beyond the frozen dataset is a no-expiry sentinel, never a fill.
        deadline=times[-1] if terminal else timestamp('2260-01-01')
        terms={k:costs[k] for k in ('margin_rate','open_fee','close_fee','close_today_fee','spread','slippage')}
        terms.update(start=str(times[0]-pd.Timedelta(minutes=1)),end=str(times[-1]+pd.Timedelta(minutes=1)))
        raw=dict(contract=item['contract'],tick_size=1,lot_size=costs['lot_size'],
            last_open_at=str(deadline),liquidate_at=str(deadline),terms=[terms],sessions=sessions,
            calendar_basis='user_confirmed_complete_market',expiry_basis='main_qualification',
            costs_basis=costs['cost_basis'],source_sha256=item['sha256'])
        path=output/f"{item['contract'].replace('.', '_')}_contract.yaml"
        if path.exists():
            raise ValueError("Repeated main segments need explicit continuation handling")
        path.write_text(yaml.safe_dump(raw))
        paths.append(str(path))
    write_json(output/'preparation.json',dict(schema=1,contracts=paths,
        calendar_independently_verified=False,user_confirmed_complete=True,
        expiry_basis='main_qualification',costs=costs))
    return paths


def split_main_contract_days(frames, split):
    """Keep each main run in one role; boundaries use DB trading dates."""
    if split.get('mode') != 'main_contract_boundaries':
        raise ValueError('FG split requires main_contract_boundaries')
    a, b = (int(pd.Timestamp(split[key]).strftime('%Y%m%d'))
            for key in ('train_end_inclusive', 'validation_end_inclusive'))
    days = sorted({int(day) for frame in frames.values() for day in frame._day})
    if not a < b or a not in days or b not in days:
        raise ValueError('FG split boundaries must be ascending exported trading days')
    roles = dict(train=[d for d in days if d <= a],
                 validation=[d for d in days if a < d <= b],
                 test=[d for d in days if d > b])
    if not all(roles.values()):
        raise ValueError('FG split requires three nonempty roles')
    for contract, frame in frames.items():
        first, last = int(frame._day.min()), int(frame._day.max())
        if any(first <= boundary < last for boundary in (a, b)):
            raise ValueError(f'FG split cuts main-contract lifecycle: {contract}')
    return roles


def build_contract(export_manifest, direction_path, contract_paths, template_path, output):
    """Split by real trading day; never invent labels, costs or contract deadlines."""
    direction_raw=read_yaml(direction_path)
    unified='directions' in direction_raw
    directions=DirectionSchedule(direction_raw['directions'] if unified else direction_raw)
    if not directions.intervals:
        raise ValueError("Human direction intervals are required before dataset construction")
    configs={raw["contract"]:(Path(p).resolve(),raw) for p in contract_paths for raw in [read_yaml(p)]}
    frames={}
    exits={}
    derived_exits={}
    if export_manifest.get('main_only'):
        for previous,following in zip(export_manifest['files'],export_manifest['files'][1:]):
            if previous['contract'] != following['contract']:
                derived_exits[previous['contract']]=previous['last_timestamp']
    for item in export_manifest["files"]:
        if unified and item["contract"] in direction_raw["data"]["excluded_contracts"]:
            raise ValueError("Excluded FG contract cannot enter a dataset")
        if not is_fg(item["contract"]) or file_hash(item["path"]) != item["sha256"]:
            raise ValueError("Export hash or FG product mismatch")
        frame=pd.read_csv(item["path"])
        frame["Date"]=pd.to_datetime(frame.Date)
        if unified and timestamp(frame.Date.max())>=timestamp(direction_raw["data"]["end_inclusive"])+pd.Timedelta(days=1):
            raise ValueError("FG data exceeds the configured cutoff")
        if "trading_date" not in frame or frame.Date.duplicated().any():
            raise ValueError("FG dataset requires unique timestamps and DB trading_date")
        frame["_day"]=pd.to_datetime(frame.trading_date.astype(str), format="mixed").dt.strftime("%Y%m%d").astype(int)
        if item["contract"] in frames:
            raise ValueError("Repeated main-contract segments require explicit continuation handling; refusing to bridge them")
        frames[item["contract"]]=frame.sort_values("Date")
        exits[item["contract"]]=item.get("main_exit_at",derived_exits.get(item["contract"]))
    days=sorted({int(d) for frame in frames.values() for d in frame._day})
    if unified:
        role_days=split_main_contract_days(frames,direction_raw['data']['split'])
    else:
        a,b=int(len(days)*.6),int(len(days)*.8)
        if not 0<a<b<len(days):
            raise ValueError("Not enough trading days for 60/20/20 chronological splits")
        role_days={"train":days[:a],"validation":days[a:b],"test":days[b:]}
    output=Path(output).resolve()
    if (output/"manifest.json").exists():
        raise FileExistsError("Refusing to overwrite a frozen FG dataset")
    output.mkdir(parents=True,exist_ok=True)
    direction_copy=Path(direction_path).resolve() if unified else output/"directions.yaml"
    if not unified:
        direction_copy.write_text(Path(direction_path).read_text())
    records=[]
    for contract,frame in frames.items():
        if contract not in configs:
            raise ValueError(f"Missing effective contract metadata: {contract}")
        _,meta=configs[contract]
        terms=ContractTerms(meta,contract)
        meta_copy=output/f"{contract.replace('.', '_')}_contract.yaml"
        meta_copy.write_text(yaml.safe_dump(meta))
        for role,selected_days in role_days.items():
            part=frame[frame._day.isin(selected_days)].drop(columns="_day")
            if part.empty:
                continue
            first,last=timestamp(part.Date.iloc[0]),timestamp(part.Date.iloc[-1])
            expected=terms.calendar[(terms.calendar>=first)&(terms.calendar<=last)]
            actual=pd.DatetimeIndex([timestamp(t) for t in part.Date])
            coverage=actual.equals(expected)
            label_count=sum(directions.at(contract,t, str(day))!="NO_TRADE" for t,day in zip(actual,part.trading_date))
            main_exit=timestamp(exits[contract]) if exits[contract] else None
            lifecycle=first<=terms.liquidate<=last or main_exit is not None and first<=main_exit<=last
            csv=output/f"{contract.replace('.', '_')}_{role}_1m.csv"
            part.to_csv(csv,index=False)
            template=read_yaml(template_path)
            config=template['environment'] if unified else template
            config['entry'].update(contract=contract,contract_config=str(meta_copy))
            if unified:
                config['entry']['strategy_config']=str(direction_copy)
            else:
                config['entry']['direction_config']=str(direction_copy)
            if main_exit is not None and first<=main_exit<=last:
                config["entry"]["main_exit_at"]=str(main_exit)
            config["trading"].update(future_symbol=contract,currency_pair=contract.split(".")[-1],data_path=str(csv))
            env_path=output/f"{contract.replace('.', '_')}_{role}.yaml"
            env_path.write_text(yaml.safe_dump(config))
            records.append(dict(contract=contract,role=role,path=str(csv),sha256=file_hash(csv),
                env_config=str(env_path),env_sha256=file_hash(env_path),contract_config=str(meta_copy),
                contract_sha256=file_hash(meta_copy),rows=len(part),first_timestamp=str(first),last_timestamp=str(last),
                calendar_complete=coverage,direction_rows=label_count,complete_lifecycle=lifecycle,
                eligible_for_training=role=="train" and coverage and label_count>0 and lifecycle))
    manifest=dict(schema=1,product="FG",fill_model="minute_close_confirm_v1",reward="60/(60+T_minutes)",
                  direction_path=str(direction_copy),direction_sha256=file_hash(direction_copy),
                  roles=role_days,files=records)
    if unified:
        manifest['split']=direction_raw['data']['split']
    manifest["content_hash"]=fingerprint(manifest)
    write_json(output/"manifest.json",manifest)
    selector=dict(strategy_mode="entry_dataset",role="train",manifest_path=str(output/"manifest.json"))
    (output/"train_dataset.yaml").write_text(yaml.safe_dump(selector))
    return manifest


def baseline_action(env, strategy, rng, random_probability=.5):
    if not env._published_mask[1]:
        return 0
    if strategy=="always_add":
        return 1
    if strategy=="random_add":
        return int(rng.random()<random_probability)
    if strategy!="rule_add":
        raise ValueError(f"Unknown baseline: {strategy}")
    i=env.current_step
    if i<19:
        return 0
    history=env._raw.iloc[i-19:i+1]
    direction=env.direction_at(env._raw.index[i])
    return int(history.High.max()-history.Close.iloc[-1]>=5) if direction=="LONG" else int(history.Close.iloc[-1]-history.Low.min()>=5)


def run_policy(config_path, output, *, strategy=None, policy=None, seed=0, random_probability=.5):
    if (strategy is None)==(policy is None) or not 0<=random_probability<=1:
        raise ValueError("Specify one strategy or policy and a valid probability")
    env=FixedDirectionEntryEnv(config_path=str(config_path))
    output=Path(output)
    if (output/"summary.json").exists():
        raise FileExistsError("Refusing to overwrite an evaluation")
    output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    obs=env._get_obs()
    while not env.terminated and env.status not in ("DATA_END","DATA_GAP"):
        action=baseline_action(env,strategy,rng,random_probability) if strategy else int(policy(obs))
        obs,_,_,truncated,_=env.step(action)
        if truncated and env.status=="CHUNK_END":
            obs,_=env.reset()
    # Audit the actual events, not touches or trigger counters.
    ledger_total=env.ledger.total_balance()
    computed=sum(60/(60+e["elapsed_minutes"]) for e in env.reward_events)
    if ledger_total!=300000 or not np.isclose(computed,env.total_reward,rtol=0,atol=1e-10):
        raise RuntimeError("Financial conservation or event Reward audit failed")
    if len({e["entry_id"] for e in env.reward_events})!=len(env.reward_events):
        raise RuntimeError("Duplicate Reward entry")
    for name,rows,columns in (
        ("fills",env.fills,["entry_id","time","side","quantity","price","reason","committed"]),
        ("entry_tp_attribution",env.attributions,["entry_id","side","entry_time","exit_time","entry_price","exit_price","closed_quantity","reason","committed"]),
        ("reward_events",env.reward_events,["entry_id","elapsed_minutes","reward"]),
        ("account_equity",env.equity_history,["time","equity","unrealized_pnl","net_lots","intrabar"])):
        pd.DataFrame(rows) .reindex(columns=list(dict.fromkeys(columns+[k for row in rows for k in row]))).to_csv(output/f"{name}.csv",index=False)
    state=env.export_state()
    write_json(output/"account_state.json",state)
    summary=env._get_info()
    summary.update(strategy=strategy or "dreamerv3",seed=seed,ledger_total=str(ledger_total),
                   direction_sha256=file_hash(env.strategy_path) if env.strategy_path else env.directions.hash,
                   env_config_hash=env.config_hash,simulated_fills=True,
                   legal_add_decisions=env.legal_add_decisions,voluntary_waits=env.voluntary_waits)
    write_json(output/"summary.json",summary)
    env.close()
    return summary
