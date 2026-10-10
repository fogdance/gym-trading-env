"""Audit sequential full-pass budgets using frozen training files only."""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

from gym_trading_env.envs.entry_contract import fingerprint, read_yaml
from gym_trading_env.envs.action import ForexCode
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.research.entry_pipeline import file_hash
from gym_trading_env.research.fg_entry_v1 import write_json


def frozen(path):
    value = json.loads(Path(path).read_text())
    expected = value.pop('content_hash')
    if fingerprint(value) != expected:
        raise ValueError(f'Manifest changed: {path}')
    return value, expected


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', default='artifacts/fg_abcd_v1/v2_sequential/experiment_manifest.json')
    parser.add_argument('--output', default='artifacts/fg_abcd_v1/v2_sequential/sequential_budget_audit.json')
    parser.add_argument('--terminal-probe', action='store_true',
                        help='Seed six lots at the penultimate real bar and execute the final bar; not a full strategy replay.')
    args = parser.parse_args()
    experiment, digest = frozen(args.manifest)
    contracts, stages = {}, []
    for instance in experiment['instances']:
        if instance['budget'] != 'formal' or instance['seed'] != 0:
            continue
        for phase in instance['phases']:
            manifest, _ = frozen(phase['dataset']['manifest'])
            records = [r for r in manifest['files'] if r['role'] == 'train' and r['eligible_for_training']]
            selected = []
            for row in records:
                key = row['env_config']
                if key not in contracts:
                    if file_hash(row['path']) != row['sha256'] or file_hash(key) != row['env_sha256']:
                        raise ValueError(f'Frozen training input changed: {row["contract"]}')
                    frame = pd.read_csv(row['path'])
                    if len(frame) != row['rows']:
                        raise ValueError('Row count mismatch')
                    config = read_yaml(key)
                    chunk = int(config['entry']['chunk_steps'])
                    advanced = len(frame) - 1
                    resets = math.ceil(advanced / chunk) if chunk else 1
                    last = frame.iloc[-1]
                    clock_valid = pd.Timestamp(last.Date) - pd.Timestamp(last.bob) == pd.Timedelta(minutes=1)
                    contracts[key] = dict(contract=row['contract'],rows=len(frame),advance_bars=advanced,
                        same_contract_chunk_steps=chunk,reset_interactions=resets,
                        full_pass_interactions=advanced+resets,
                        final_timestamp=str(last.Date),main_exit_at=config['entry'].get('main_exit_at'),
                        final_volume=float(last.Volume),final_fill_clock_valid=clock_valid,
                        boundary_liquidity_sufficient_for_6_lots=bool(clock_valid and last.Volume >= 6),
                        final_matches_main_exit=bool(pd.Timestamp(last.Date)==pd.Timestamp(config['entry'].get('main_exit_at'))))
                    if args.terminal_probe:
                        env = FixedDirectionEntryEnv(config_path=key)
                        probes = []
                        for side in ('LONG','SHORT'):
                            if env.terminated:
                                env.reset()
                            env.current_step = len(env._raw)-2
                            prior = env._raw.iloc[env.current_step]
                            time = env._raw.index[env.current_step]
                            env._mark(prior.Close,time)
                            for _ in range(2):
                                if env._open_entry(side,prior.Close,time) != ForexCode.SUCCESS:
                                    raise ValueError('Unable to seed six-lot terminal probe')
                            if env.net_lots != 6:
                                raise ValueError('Terminal probe did not start with six lots')
                            _, reward, terminated, truncated, info = env.step(0)
                            passed = bool(terminated and not truncated and env.net_lots==0 and reward==0
                                          and env.status=='MAIN_ROLL' and len(env.attributions)==2
                                          and all(r['reason']=='MAIN_ROLL' for r in env.attributions))
                            probes.append(dict(side=side,passed=passed,reward=reward,
                                status=env.status,remaining_lots=str(env.net_lots),net_pnl=info['account_net_pnl']))
                            if not passed:
                                raise ValueError(f'Terminal execution probe failed: {row["contract"]}/{side}')
                        env.close()
                        contracts[key]['terminal_probes']=probes
                selected.append(contracts[key])
            full_pass = sum(r['full_pass_interactions'] for r in selected)
            budget = phase['steps']
            stages.append(dict(fold=instance['fold'],method=instance['method'],phase=phase['phase'],
                contracts=len(selected),rows=sum(r['rows'] for r in selected),
                advance_bars=sum(r['advance_bars'] for r in selected),
                reset_interactions=sum(r['reset_interactions'] for r in selected),
                full_pass_interactions=full_pass,budget=budget,whole_passes=budget//full_pass,
                remainder_interactions=budget%full_pass,pass_equivalents=budget/full_pass))
    result = dict(experiment_manifest_hash=digest,mode='SEQUENTIAL_COUNTERFACTUAL_ONLY',
        accounting='One reset interaction per same-contract chunk including the first chunk; all other interactions advance one bar.',
        assumptions=['Every lifecycle reaches its configured main exit; no earlier margin/expiry termination or data-end freeze.',
                     'Existing same-contract chunk resume remains; cross-contract least_progress rotation is not used in this calculation.',
                     'Coverage is rollout exposure, not proof of learning or uniform Replay sampling.'],
        final_blind_prices_read=False,stages=stages,contracts=list(contracts.values()))
    result['terminal_probe_basis']='Synthetic six-lot holdings seeded on real penultimate bars; checks terminal execution only, not full-history coverage.'
    write_json(args.output,result)
    print(json.dumps(dict(output=args.output,stages=stages,
        all_budgets_cover_full_pass=all(r['whole_passes']>=1 for r in stages),
        all_terminal_data_checks_pass=all(r['boundary_liquidity_sufficient_for_6_lots'] and r['final_matches_main_exit'] for r in contracts.values())),indent=2))


if __name__ == '__main__':
    main()
