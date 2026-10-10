"""Frozen FG development-only offline study; no model imports or training."""
from __future__ import annotations

import csv
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import shutil
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

from .fg_reward_candidates import CandidateConfig, FGRewardScorer
from .entry_pipeline import file_hash
from .fg_abcd_evaluation import validate_validation_inputs
from .fg_entry_v1 import write_json

ROOT=Path(__file__).resolve().parents[3]


def candidate_grid(raw):
    return [CandidateConfig(beta=b,lam=l,alpha=a,rho=raw['rho'],gamma=raw['gamma'])
            for b,l,a in itertools.product(raw['beta'],raw['lam'],raw['alpha'])]


def grid_id(config):
    return f'b{config.beta:g}_l{config.lam:g}_a{config.alpha:g}'


def combine_components(rows, config):
    """Affine rescoring of already observed events; never simulates new fills."""
    col=lambda k:np.asarray([float(r[k]) for r in rows],dtype=np.float64)
    tp=col('tp_timeboxed')
    early=-config.beta*(col('early_first5')+config.rho*col('early_next10'))
    potential=config.alpha*col('potential_basis')
    capacity=-config.lam*col('capacity_basis')
    ambiguity_refund=config.beta*(col('ambiguous_early_first5')+
                                   config.rho*col('ambiguous_early_next10'))
    return dict(v0=col('v0'),tp_timeboxed=tp,early=early,potential=potential,
                capacity=capacity,v2=tp+early+potential,v3=tp+early+potential+capacity,
                ambiguity_refund=ambiguity_refund)


def verify_direct(events, rows, config):
    """Verify every step against the stateful scorer, not just total score."""
    scorer=FGRewardScorer(config)
    direct=[scorer.step(event) for event in events]
    rebuilt=combine_components(rows,config)
    if len(direct)!=len(rows):
        raise ValueError('Trace/component length mismatch')
    worst=0.0
    for key in ('v0','tp_timeboxed','early','potential','capacity','v2','v3'):
        received=np.asarray([r[key] for r in direct])
        if not np.allclose(received,rebuilt[key],rtol=0,atol=1e-12):
            raise ValueError(f'Component reconstruction failed: {key}')
        if len(received):
            worst=max(worst,float(np.max(np.abs(received-rebuilt[key]))))
    return dict(steps=len(direct),max_step_error=worst,
                snapshot_sha256=scorer.snapshot()['sha256'])


def summarize_grid(rows, config, original, *, policy, seed):
    vectors=combine_components(rows,config)
    weights=config.gamma**np.arange(len(rows),dtype=float)
    result=dict(policy=policy,seed=seed,grid_id=grid_id(config),beta=config.beta,
                rho=config.rho,lam=config.lam,alpha=config.alpha,gamma=config.gamma,
                scored_steps=len(rows),**original)
    days=original['eligible_trading_days']
    for name,values in vectors.items():
        result[name]=float(values.sum())
        if name in ('v0','v2','v3'):
            result[f'discounted_{name}']=float(weights@values)
            result[f'{name}_per_day']=float(values.sum()/days) if days else None
            result[f'{name}_positive_steps']=int((values>1e-12).sum())
            result[f'{name}_negative_steps']=int((values<-1e-12).sum())
            result[f'{name}_positive_fraction']=float((values>1e-12).mean()) if len(values) else None
            result[f'{name}_negative_fraction']=float((values<-1e-12).mean()) if len(values) else None
    result['capacity_basis']=float(sum(float(r['capacity_basis']) for r in rows))
    result['early_first5']=int(sum(int(r['early_first5']) for r in rows))
    result['early_next10']=int(sum(int(r['early_next10']) for r in rows))
    result['ambiguous_tp_bars']=sum(bool(json.loads(r['intrabar_ambiguous']))
                                    if isinstance(r['intrabar_ambiguous'],str)
                                    else bool(r['intrabar_ambiguous']) for r in rows)
    result['v3_without_ambiguous_first_risk']=result['v3']+result['ambiguity_refund']
    result['always_wait_beats_v2']=result['v2'] < -1e-12
    result['always_wait_beats_v3']=result['v3'] < -1e-12
    return result


def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows):
    from tools.fg_reward_offline import _write_csv
    _write_csv(path,rows)


def score_saved(directory, configurations):
    directory=Path(directory)
    summary=json.loads((directory/'summary.json').read_text())
    metadata=summary['input_provenance'];original=metadata['original_env']
    rows=read_csv(directory/'step_components.csv')
    trades=read_csv(directory/'trade_components.csv')
    scores=[];trade_scores=[];daily=[]
    _,day_indices=np.unique([r['trading_date'] for r in rows],return_inverse=True)
    dates=sorted({r['trading_date'] for r in rows})
    for config in configurations:
        score=summarize_grid(rows,config,original,policy=metadata['policy'],seed=metadata['seed'])
        scores.append(score)
        vectors=combine_components(rows,config)
        by_day={k:np.bincount(day_indices,weights=v,minlength=len(dates))
                for k,v in vectors.items() if k in ('v0','v2','v3','tp_timeboxed','early','potential','capacity')}
        for i,day in enumerate(dates):
            daily.append(dict(policy=metadata['policy'],seed=metadata['seed'],grid_id=grid_id(config),
                              trading_date=day,**{k:float(v[i]) for k,v in by_day.items()}))
        for trade in trades:
            first=int(trade['adverse_first_bar']) if trade['adverse_first_bar'] else 0
            early=-config.beta*(1 if 0<first<=5 else config.rho if 6<=first<=15 else 0)
            cost=-config.lam*float(trade['capacity_basis'])
            trade_scores.append(dict(policy=metadata['policy'],seed=metadata['seed'],grid_id=grid_id(config),
                entry_id=trade['entry_id'],status=trade['status'],tp_timeboxed=float(trade['timeboxed_tp']),
                early=early,capacity=cost,score_excluding_account_potential=float(trade['timeboxed_tp'])+early+cost))
    write_csv(directory/'parameter_scores.csv',scores)
    write_csv(directory/'parameter_trade_scores.csv',trade_scores)
    write_csv(directory/'parameter_daily_scores.csv',daily)
    return dict(directory=str(directory),summary=summary,scores=scores)


def _worker(job, reference, grid):
    # Import the one public replay runner; no duplicated Env loop.
    if str(ROOT) not in sys.path:
        sys.path.insert(0,str(ROOT))
    from tools.fg_reward_offline import run_single
    started=time.monotonic()
    args=SimpleNamespace(trace=None,**job,**reference)
    summary=run_single(args)
    result=score_saved(args.out,candidate_grid(grid))
    result['seconds']=time.monotonic()-started
    result['summary']=summary
    return result


def validate_saved_replay(directory, reference, frozen):
    """Reuse only frozen logged actions produced by this exact scorer/adapter."""
    directory=Path(directory)
    summary=json.loads((directory/'summary.json').read_text())
    meta=summary['input_provenance']
    if (meta.get('policy')!='logged_actions' or not meta.get('frozen_manifest_verified') or
            meta.get('frozen_input_hashes')!=frozen or
            summary['config']!=vars(CandidateConfig(**reference))):
        raise ValueError('E1 reuse protocol/frozen-input mismatch')
    for path,checksum in (meta['source_hashes']|meta['original_evaluation_hashes']|frozen).items():
        if file_hash(path)!=checksum:
            raise ValueError(f'E1 reuse source/input changed: {path}')
    state=FGRewardScorer.restore(json.loads((directory/'snapshot.json').read_text()),CandidateConfig(**reference))
    if state.summary()['totals']!=summary['totals']:
        raise ValueError('E1 summary does not reconcile with scorer snapshot')
    return summary


def make_figures(output, scores, reference):
    """Standalone research figures; seed ranges are not market confidence CIs."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    selected=[r for r in scores if all(r[k]==reference[k] for k in ('beta','lam'))]
    policies=['logged_actions','always_add','always_wait','delay5','delay15','delay30','random']
    labels=['A replay','Always ADD','WAIT','Delay 5','Delay 15','Delay 30','Random (30 seeds)']
    fig,ax=plt.subplots(figsize=(12,5))
    variants=[('v0',0,'V0'),('v2',0,'V2, alpha=0'),('v3',0,'V3, alpha=0'),
              ('v3',reference['alpha'],f'V3, alpha={reference["alpha"]:g}')]
    for i,(key,alpha,label) in enumerate(variants):
        values=[[r[key] for r in selected if r['policy']==p and r['alpha']==alpha] for p in policies]
        med=np.array([np.median(v) for v in values]);low=np.array([np.percentile(v,2.5) for v in values])
        high=np.array([np.percentile(v,97.5) for v in values])
        ax.bar(np.arange(len(policies))+(i-1.5)*.2,med,width=.2,label=label,
               yerr=np.stack([med-low,high-med]),capsize=2)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(len(labels)),labels,rotation=12)
    ax.set_ylabel('Undiscounted score');ax.set_title('FG409 fixed trajectories; Random shows central 95% seed range')
    ax.legend();fig.tight_layout();fig.savefig(output/'strategy_scores.png',dpi=160);plt.close(fig)
    a=[r for r in scores if r['policy']=='logged_actions']
    betas=sorted({r['beta'] for r in a});lams=sorted({r['lam'] for r in a})
    alphas=sorted({r['alpha'] for r in a})
    surfaces=[np.array([[next(r['v3'] for r in a if r['beta']==b and r['lam']==l and r['alpha']==alpha)
                        for l in lams] for b in betas]) for alpha in alphas]
    limit=max(abs(v).max() for v in surfaces)
    fig,axes=plt.subplots(1,len(alphas),figsize=(11,4),layout='constrained',squeeze=False)
    for ax,alpha,values in zip(axes[0],alphas,surfaces):
        im=ax.imshow(values,cmap='RdBu',vmin=-limit,vmax=limit)
        ax.set_xticks(range(len(lams)),[f'{v:g}' for v in lams]);ax.set_yticks(range(len(betas)),betas)
        ax.set_xlabel('lambda');ax.set_ylabel('beta');ax.set_title(f'A V3 score, alpha={alpha:g}; WAIT=0')
        for i,j in np.ndindex(values.shape):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=list(axes[0]));fig.savefig(output/'A_parameter_surface.png',dpi=160);plt.close(fig)


def run_study(config_path, output):
    """E1 must pass before any E2 jobs. Output dirs are never reused/overwritten."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0,str(ROOT))
    config_path=Path(config_path).resolve();output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite study: {output}')
    raw=yaml.safe_load(config_path.read_text())
    if raw['schema']!=1 or raw['contract']!='CZCE.FG409':
        raise ValueError('This study is restricted to the frozen FG409 development contract')
    reference=raw['reference'];configs=candidate_grid(raw['grid'])
    if any(c.gamma!=reference['gamma'] or c.rho!=reference['rho'] for c in configs):
        raise ValueError('Component reuse requires the same gamma/rho')
    manifest=(ROOT/raw['experiment_manifest']).resolve()
    experiment=json.loads(manifest.read_text())
    fold=next(f for f in experiment['folds'] if f['validation']==raw['contract'])
    env_config=Path(fold['validation_env'])
    frozen=validate_validation_inputs(manifest,env_config)
    evaluation=(ROOT/raw['evaluation_dir']).resolve()
    receipt=evaluation.parent/'evaluation_receipt.json'
    if json.loads(receipt.read_text())['experiment_hash']!=experiment['content_hash']:
        raise ValueError('Original evaluation is not from the frozen experiment')
    source_paths=[config_path,Path(__file__),
        ROOT/'src/gym_trading_env/research/fg_reward_candidates.py',ROOT/'tools/fg_reward_offline.py',receipt]
    for name in ('envs','rewards','utils'):
        source_paths.extend(sorted((ROOT/'src/gym_trading_env'/name).rglob('*.py')))
    sources={str(p):file_hash(p) for p in source_paths}
    output.mkdir(parents=True)
    write_json(output/'study_inputs.json',dict(config=raw,config_path=str(config_path),
        frozen_inputs=frozen,source_hashes=sources,environment=str(env_config),
        grid_points=len(configs),random_probability=.5,training=False,model_inference=False))
    base=dict(manifest=manifest,env_config=env_config)
    job=dict(**base,eval_dir=evaluation,policy=None,seed=0,out=output/'A_logged')
    print('E1: validating frozen A actions',flush=True)
    if raw.get('verified_e1'):
        reused=(ROOT/raw['verified_e1']).resolve()
        validate_saved_replay(reused,reference,frozen)
        checksums={str(p):file_hash(p) for p in reused.iterdir() if p.is_file()}
        shutil.copytree(reused,job['out'])
        write_json(job['out']/'reuse_receipt.json',dict(source=str(reused),files=checksums,
                   validated_before_reuse=True))
        started=time.monotonic()
        initial=score_saved(job['out'],configs);initial['seconds']=time.monotonic()-started
        initial['reused_from']=str(reused)
        for path,checksum in checksums.items():
            if file_hash(path)!=checksum:
                raise ValueError('E1 cache changed while copying')
    else:
        initial=_worker(job,reference,raw['grid'])
    actual=initial['summary'];account=actual['input_provenance']['original_env']
    expected=dict(ADD_fill=75,tp_count=73,non_tp_exits=2,account_net_pnl=18990,max_floating_loss=38460)
    if any(account[k]!=v for k,v in expected.items()) or not np.isclose(
            actual['totals']['v0'],22.740204501031855,rtol=0,atol=1e-8):
        raise ValueError('E1 FG409 acceptance values do not reconcile')
    events=[json.loads(line) for line in (output/'A_logged/events.jsonl').read_text().splitlines()]
    rows=read_csv(output/'A_logged/step_components.csv')
    extreme=CandidateConfig(beta=max(raw['grid']['beta']),lam=max(raw['grid']['lam']),
                           alpha=max(raw['grid']['alpha']),rho=reference['rho'],gamma=reference['gamma'])
    direct=verify_direct(events,rows,extreme)
    write_json(output/'direct_reconstruction_check.json',direct)
    print(f'E1 PASS: 75 entries, 73 TP, 2 non-TP; reconstruction {direct}',flush=True)
    jobs=[dict(**base,eval_dir=None,policy=p,seed=0,out=output/p) for p in raw['policies']]
    jobs += [dict(**base,eval_dir=None,policy='random',seed=s,out=output/f'random_{s:02d}')
             for s in raw['random_seeds']]
    results=[initial]
    workers=min(4,max(1,int(raw['workers'])),os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={pool.submit(_worker,j,reference,raw['grid']):j for j in jobs}
        for future in as_completed(pending):
            result=future.result();results.append(result)
            job=pending[future]
            print(f'E2 {len(results)-1}/{len(jobs)}: {job["policy"]} seed={job["seed"]} '
                  f'{result["seconds"]:.1f}s',flush=True)
    for path,checksum in (frozen|sources).items():
        if file_hash(path)!=checksum:
            raise ValueError(f'Frozen study input/source changed: {path}')
    results.sort(key=lambda r:r['directory'])
    scores=[s for r in results for s in r['scores']]
    write_csv(output/'parameter_scores.csv',scores)
    make_figures(output,scores,reference)
    write_json(output/'study_summary.json',dict(status='PASS',trajectories=len(results),grid_points=len(configs),
        score_rows=len(scores),workers=workers,results=[dict(directory=r['directory'],summary=r['summary'],
        seconds=r['seconds']) for r in results],reconstruction_check=direct,
        frozen_hashes_verified_before_and_after=True))
    print(f'Study PASS: {len(results)} trajectories × {len(configs)} parameter points',flush=True)
