"""Unified full-account evaluation; never reads sealed test prices."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
from pathlib import Path
import numpy as np
import pandas as pd

from gym_trading_env.research.fg_abcd_evaluation import replay_policy,matched_quotas
from gym_trading_env.research.fg_entry_v1 import write_json


def matched_worker(payload):
    config,output,quotas,seed=payload
    return replay_policy(config,output,quotas=quotas,seed=seed)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--env-config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--model-config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--manifest',default='artifacts/fg_abcd_v1/v3_shared_prefix/experiment_manifest.json')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.workers<1:
        raise ValueError('workers must be positive')
    from gym_trading_env.envs.entry_contract import fingerprint
    manifest=json.loads(Path(args.manifest).read_text())
    digest=manifest.pop('content_hash')
    allowed={str(Path(f['validation_env']).resolve()) for f in manifest['folds']}
    if fingerprint(manifest)!=digest or str(Path(args.env_config).resolve()) not in allowed:
        raise ValueError('Evaluation requires a frozen development validation contract')
    if bool(args.model_config)!=bool(args.checkpoint):
        raise ValueError('Provide both model config and checkpoint')
    root=Path(args.output)
    if root.exists():
        raise FileExistsError('Refusing to replace comparison results')
    rows=[]
    for strategy in ('always_add','random_add','rule_add'):
        rows.append(replay_policy(args.env_config,root/strategy,strategy=strategy,seed=args.seed))
    if args.checkpoint:
        import jax
        import elements
        from ruamel import yaml
        from dreamerv3.main import make_agent
        from embodied.run.action_mask_guard import require_formal_actor
        assert all(d.platform=='gpu' for d in jax.devices()), 'GPU-only model evaluation'
        cfg=elements.Config(yaml.YAML(typ='safe').load(Path(args.model_config).read_text()))
        cfg=cfg.update({'task':'gymnasium_FixedDirectionEntryEnv-v0','env.gymnasium.config_path':str(Path(args.env_config).resolve())})
        agent=make_agent(cfg)
        elements.checkpoint.load(args.checkpoint,dict(agent=agent.load))
        require_formal_actor(agent,'ABC full-account evaluation')
        carry=[agent.init_policy(1)]
        def policy(obs,first,reward):
            if first:
                carry[0]=agent.init_policy(1)
            batch={k:np.asarray(v)[None] for k,v in obs.items()}
            batch.update(reward=np.array([reward],np.float32),is_first=np.array([first]),
                         is_last=np.zeros(1,bool),is_terminal=np.zeros(1,bool))
            carry[0],act,_=agent.policy(carry[0],batch,mode='eval')
            return int(act['action'][0])
        dreamer=replay_policy(args.env_config,root/'dreamerv3',policy=policy,seed=args.seed)
        rows.append(dreamer)
        quotas=matched_quotas(root/'dreamerv3/policy_actions.csv')
        matched=[]
        payloads=[(args.env_config,str(root/f'matched_random/{seed}'),quotas,seed) for seed in range(30)]
        # Workers simulate independent Env accounts; no model/GPU inference in them.
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            results=list(pool.map(matched_worker,payloads))
        for result in results:
            result['dreamer_minus_reward_per_day']=dreamer['reward_per_day']-result['reward_per_day']
            result['dreamer_minus_median_tp_minutes']=(dreamer['median_tp_minutes']-result['median_tp_minutes']
                if dreamer['median_tp_minutes'] is not None and result['median_tp_minutes'] is not None else None)
            result['waiting_difference_basis']='SUCCESS_ONLY_WITH_FAILURE_AND_CENSOR_COUNTS_REPORTED'
            for unit,limits in [('minutes',(5,15,60)),('trading_days',(1,5))]:
                for h in limits:
                    field=f'success_rate_{h}_{unit}'
                    result[f'dreamer_minus_{field}']=(dreamer[field]-result[field]
                        if dreamer[field] is not None and result[field] is not None else None)
            result['diagnostic']='trajectory_conditioned_closed_loop'
            matched.append(result)
        pd.DataFrame(matched).to_csv(root/'matched_random.csv',index=False)
        write_json(root/'matched_random_summary.json',dict(seeds=30,
            exact_matches=sum(r['match_status']=='exact_match' for r in matched),
            mean_reward_per_day=float(np.mean([r['reward_per_day'] for r in matched])),
            mean_paired_reward_per_day_delta=float(np.mean([r['dreamer_minus_reward_per_day'] for r in matched])),
            reward_per_day_quantiles=list(map(float,np.percentile([r['reward_per_day'] for r in matched],[5,50,95]))),
            market_confidence_interval=False,superiority_claim='NOT_ESTABLISHED'))
    if len({r['eligible_trading_days'] for r in rows})!=1:
        raise RuntimeError('Strategy-dependent denominator')
    pd.DataFrame(rows).to_csv(root/'episode_metrics.csv',index=False)
    write_json(root/'comparison.json',dict(results=rows,model_status='EVALUATED' if args.checkpoint else 'NOT_RUN'))
    lines=['# ABC validation comparison','','OHLC simulated fills; given manual direction only.','',
           '| Strategy | Reward/day | TP batches | Open lots | Max floating loss | Equity change |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+ ' | '.join(str(r[k]) for k in ('strategy','reward_per_day','completed_batches','open_lots_at_end','max_floating_loss','net_equity_change'))+' |')
    if not args.checkpoint:
        lines.append('\nDreamer and matched random: NOT_RUN. No trained checkpoint supplied.')
    lines.append('\nRandom seed variability is not market uncertainty. Partial matches do not establish equal-frequency timing superiority.')
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(root/'comparison.md')


if __name__=='__main__':
    main()
