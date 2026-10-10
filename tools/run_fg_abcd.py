"""Prepare or run ABC stages, reusing exact shared nodes; never launches by default."""
import argparse
import json
import os
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
import uuid

from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.entry_pipeline import file_hash
from gym_trading_env.research.fg_entry_v1 import write_json

DEFAULT_MANIFEST='artifacts/fg_abcd_v1/v3_shared_prefix/experiment_manifest.json'


def command(instance, phase, dreamer_root, transfer=''):
    cmd=[sys.executable,'-u','-m','dreamerv3.main','--configs','defaults','fg_entry_v1','fg_entry_dataset_v1',
        '--fg_entry.manifest',phase['dataset']['manifest'],'--env.gymnasium.config_path',phase['dataset']['selector'],
        '--logdir',phase['logdir'],'--experiment_seed',str(instance['seed']),'--dreamer.seed',str(instance['seed']),
        '--run.steps',str(phase['steps']),'--jax.platform','cuda','--jax.prealloc','False',
        '--logger.outputs','jsonl','tensorboard','--run.fg_learning_audit','True',
        '--run.fg_stage_transfer',transfer]
    if phase.get('full_checkpoint'):
        cfg=phase['checkpoint']
        cmd.extend(['--run.fg_full_checkpoint','True',
            '--run.fg_source_steps',str(phase.get('total_step_offset',0)),
            '--run.fg_report_every_steps',str(cfg['report_every_steps']),
            '--run.fg_save_every_steps',str(cfg['save_every_steps']),
            '--run.fg_key_steps',','.join(str(x) for x in cfg['key_steps']),
            '--run.fg_reserve_gib',str(cfg['reserve_gib'])])
    return cmd


def resolve_command(cmd,dreamer_root):
    """Resolve actual training flags without creating an Agent or initializing CUDA."""
    import elements
    from ruamel import yaml
    from dreamerv3.main import _resolve_seed_protocol
    configs=yaml.YAML(typ='safe').load((Path(dreamer_root)/'dreamerv3/configs.yaml').read_text())
    parsed,remaining=elements.Flags(configs=['defaults']).parse_known(cmd[4:])
    cfg=elements.Config(configs['defaults'])
    for name in parsed.configs:
        cfg=cfg.update(configs[name])
    return _resolve_seed_protocol(elements.Flags(cfg).parse(remaining))


def checkpoint_path(phase,required=True):
    root=Path(phase['logdir'])/'ckpt'
    if phase.get('full_checkpoint'):
        point=root/f"step_{phase['steps']:012d}"
        if not point.exists():
            if not required:return None
            raise FileNotFoundError(f'Required exact shared checkpoint is missing: {point}')
        from embodied.run.fg_checkpoint import validate_bundle
        validate_bundle(point)
        return point
    latest=root/'latest'
    if not latest.exists() and not required:return None
    point=root/latest.read_text().strip()
    if not (point/'done').is_file():raise ValueError('Incomplete legacy checkpoint')
    return point


def phase_record_path(instance,phase):
    return Path(phase['logdir'])/'phase_records'/f"{instance['budget']}_{instance['method']}_{phase['phase']}_{phase['steps']}.json"


def phase_metadata(instance,phase):
    filename=phase_record_path(instance,phase) if phase.get('full_checkpoint') else Path(phase['logdir'])/'launch_metadata.json'
    return json.loads(filename.read_text()) if filename.exists() else dict(status='NOT_RUN')


def transfer_path(source,target):
    return (Path(target['logdir'])/'source_transfer.json' if source.get('full_checkpoint')
            else Path(source['logdir'])/'c_transfer_receipt.json')


def transfer_receipt(instance,source,target):
    root=Path(source['logdir']);checkpoint=checkpoint_path(source)
    if int(pickle.loads((checkpoint/'step.pkl').read_bytes()))!=source['steps']:
        raise ValueError('Pretrain node budget has not completed')
    full=bool(source.get('full_checkpoint'))
    source_config=checkpoint/'config.yaml' if full else root/'config.yaml'
    files=[source_config,Path(source['dataset']['manifest']),Path(target['dataset']['manifest'])]
    files.extend(p for p in checkpoint.rglob('*') if p.is_file())
    receipt=dict(method='C',fold=instance['fold'],seed=instance['seed'],source_steps=source['steps'],
        checkpoint=str(checkpoint),source_config=str(source_config),target_manifest=target['dataset']['manifest'],
        input_hashes={str(p):file_hash(p) for p in files},local_recent_steps=0,total_step_offset=source['steps'],
        replay_protocol='EMPTY_RECENT_ONLY',accounts_protocol='NEW_RECENT_WITH_PRETRAIN_ACCOUNTS_ARCHIVED')
    if full:receipt['full_checkpoint']=True
    receipt['content_hash']=fingerprint(receipt)
    path=transfer_path(source,target)
    if path.exists() and json.loads(path.read_text())!=receipt:
        raise ValueError('Refusing to replace an existing C transfer receipt')
    if not path.exists():write_json(path,receipt)
    return str(path)


def completed_node(instance,phase,dreamer_root):
    point=checkpoint_path(phase,required=False)
    if point is None:return None
    if phase.get('full_checkpoint'):
        from embodied.run.fg_checkpoint import provenance,validate_bundle
        cfg=resolve_command(command(instance,phase,dreamer_root),dreamer_root)
        validate_bundle(point,provenance(cfg))
    elif int(pickle.loads((point/'step.pkl').read_bytes()))!=phase['steps']:
        return None
    return point


def learning_at_node(phase):
    point=checkpoint_path(phase)
    if not phase.get('full_checkpoint'):
        return json.loads((Path(phase['logdir'])/'learning_audit.json').read_text())
    from embodied.run.fg_learning_audit import summarize_learning
    data=pickle.loads((point/'agent.pkl').read_bytes())
    gate=pickle.loads((point/'action_mask_warmup.pkl').read_bytes())
    class SavedAgent:
        def save(self):return data
        def get_avail_actor_enabled(self):return gate['actor_enabled']
    destination=Path(phase['logdir'])/'phase_records'/f"learning_{phase['steps']}.json"
    return summarize_learning(SavedAgent(),phase['logdir'],phase['steps'],output_path=destination)


def account_progress(state,records):
    """Actual visited prefixes; early risk exits must not count as full lifecycles."""
    if state['schema']==2:
        result=[]
        for i,row in enumerate(records):
            cycles,prefix=divmod(state['advanced'][i],row['rows']-1)
            result.append(dict(advanced_bars=state['advanced'][i],full_passes=cycles,
                current_prefix_bars=prefix,prefix_counts={row['rows']-1:cycles,prefix:1},
                saved_account=i in state['accounts'] or str(i) in state['accounts']))
        return result
    if state['schema']!=1:
        raise ValueError('Unknown dataset checkpoint schema')
    prefixes={r['contract']:[] for r in records}
    for finished in state['completed']:
        if 'advanced_bars' not in finished:
            raise ValueError('Legacy completed account lacks actual coverage counters')
        prefixes[finished['contract']].append(finished['advanced_bars'])
    active=state['account']
    prefixes[active['contract']].append(active['current_step'])
    result=[]
    for row in records:
        lengths=prefixes[row['contract']]
        if any(n<0 or n>=row['rows'] for n in lengths):
            raise ValueError('Invalid sequential coverage cursor')
        counts={n:lengths.count(n) for n in set(lengths)}
        result.append(dict(advanced_bars=sum(lengths),full_passes=lengths.count(row['rows']-1),
            current_prefix_bars=active['current_step'] if row['contract']==active['contract'] else 0,
            prefix_counts=counts,saved_account=row['contract']==active['contract']))
    return result


def training_coverage(phase, audit=None):
    import pandas as pd
    from gym_trading_env.envs.entry_contract import DirectionSchedule,read_yaml,timestamp
    root=Path(phase['logdir'])
    checkpoint=checkpoint_path(phase)
    driver=pickle.loads((checkpoint/'driver.pkl').read_bytes())
    state=driver['envs'][0]['env']
    manifest=json.loads(Path(phase['dataset']['manifest']).read_text())
    records=[r for r in manifest['files'] if r['role']=='train' and r['eligible_for_training']]
    schedule=DirectionSchedule(read_yaml(manifest['direction_path'])['directions'])
    coverage=[]
    progress=account_progress(state,records)
    for row,observed in zip(records,progress):
        advanced=observed['advanced_bars']
        frame=pd.read_csv(row['path'])
        groups={}
        for length,weight in observed['prefix_counts'].items():
            if not weight:
                continue
            part=frame.iloc[1:length+1]
            for _,bar in part.iterrows():
                year=str(pd.Timestamp(str(bar.trading_date)).year) if not str(bar.trading_date).isdigit() else str(bar.trading_date)[:4]
                direction=schedule.at(row['contract'],timestamp(bar.Date),str(bar.trading_date))
                key=f'{year}/{direction}'
                groups[key]=groups.get(key,0)+weight
        coverage.append(dict(contract=row['contract'],advanced_bars=advanced,full_passes=observed['full_passes'],
            current_prefix_bars=observed['current_prefix_bars'],year_direction_bars=groups,visited=advanced>0,
            saved_account=observed['saved_account']))
    audit=audit or json.loads((root/'learning_audit.json').read_text())
    actual_advanced=sum(r['advanced_bars'] for r in progress)
    result=dict(driver_interactions=int(pickle.loads((checkpoint/'step.pkl').read_bytes())),actual_advanced_bars=actual_advanced,
        contracts=coverage,visited_contracts=sum(r['visited'] for r in coverage),
        learning_updates=audit['updates'],training_samples=audit['updates']*16*64,
        replay_sample_to_advanced_bar_ratio=audit['updates']*16*64/max(1,actual_advanced),
        dataset_checkpoint_schema=state['schema'],
        accounting='Independent per-contract training accounts; not pooled-investor performance')
    destination=(root/'phase_records'/f"coverage_{phase['steps']}.json" if phase.get('full_checkpoint') else root/'training_coverage.json')
    write_json(destination,result)
    return result


def run_phase(instance,phase,dreamer_root,environment,manifest_hash):
    cmd=command(instance,phase,dreamer_root,phase.get('transfer',''))
    print(shlex.join(cmd),flush=True)
    root=Path(phase['logdir'])
    point=completed_node(instance,phase,dreamer_root)
    before=0
    if phase.get('full_checkpoint'):
        from embodied.run.fg_checkpoint import FullCheckpoint,provenance
        cfg=resolve_command(cmd,dreamer_root)
        latest=FullCheckpoint(root/'ckpt',None,provenance(cfg),()).latest()
        before=int(pickle.loads((latest/'step.pkl').read_bytes())) if latest else 0
        if point is None and before>phase['steps']:
            raise ValueError('Training is beyond the requested budget but its exact node is missing')
    metadata=dict(instance=instance,phase=phase,transfer_receipt=phase.get('transfer',''),
        experiment_manifest_hash=manifest_hash,status='RUNNING',
        execution='REUSED_CHECKPOINT' if point else 'RESUMED' if before else 'NEW',
        resume_step=before,code_sha={name:subprocess.check_output(['git','rev-parse','HEAD'],cwd=directory,text=True).strip()
            for name,directory in [('env',Path(__file__).resolve().parents[1]),('dreamer',dreamer_root)]})
    if point is None:
        root.mkdir(parents=True,exist_ok=True)
        write_json(root/'launch_metadata.json',metadata)
        try:
            with (root/'launch.log').open('a') as log:
                subprocess.run(cmd,cwd=dreamer_root,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        except (subprocess.CalledProcessError,KeyboardInterrupt) as error:
            metadata.update(status='STOPPED_BY_USER' if isinstance(error,KeyboardInterrupt) or error.returncode in (-2,130) else 'FAILED',
                returncode=getattr(error,'returncode',130),log=str(root/'launch.log'))
            write_json(root/'launch_metadata.json',metadata)
            write_json(phase_record_path(instance,phase),metadata)
            raise
        point=completed_node(instance,phase,dreamer_root)
        if point is None:raise ValueError('Training returned without its exact completed node')
    metadata.update(status='COMPLETED_BUDGET',checkpoint=str(point),
        physical_interactions_added=0 if metadata['execution']=='REUSED_CHECKPOINT' else phase['steps']-before,
        logical_total_steps=phase['steps']+phase.get('total_step_offset',0))
    metadata['learning_audit']=learning_at_node(phase)
    metadata['training_coverage']=training_coverage(phase,metadata['learning_audit'])
    previous=phase_metadata(instance,phase)
    if previous.get('status')=='COMPLETED_BUDGET' and metadata['execution']=='REUSED_CHECKPOINT':
        return previous
    write_json(phase_record_path(instance,phase),metadata)
    if metadata['execution']!='REUSED_CHECKPOINT':write_json(root/'launch_metadata.json',metadata)
    history=root/'launch_history'/f"{instance['budget']}_{instance['method']}_{phase['phase']}_{phase['steps']}_{uuid.uuid4().hex}.json"
    write_json(history,metadata)
    return metadata


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest',default=DEFAULT_MANIFEST)
    parser.add_argument('--budget',choices=['pilot','formal'],default='pilot')
    parser.add_argument('--method',choices=['A','B','C','all'],default='all')
    parser.add_argument('--fold');parser.add_argument('--seed',type=int)
    parser.add_argument('--create-transfer',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--dreamer-root',default='/home/v/Documents/work/dreamerv3')
    args=parser.parse_args()
    manifest=json.loads(Path(args.manifest).read_text());digest=manifest.pop('content_hash')
    if fingerprint(manifest)!=digest:raise ValueError('Experiment manifest changed')
    selected=[r for r in manifest['instances'] if r['budget']==args.budget and args.method in (r['method'],'all')]
    selected=[r for r in selected if (args.fold is None or args.fold in (r['fold'],r['fold'].split('.')[-1])) and (args.seed is None or args.seed==r['seed'])]
    if not selected:raise ValueError('No matching frozen experiment instances')
    if args.create_transfer and (len(selected)!=1 or selected[0]['method']!='C' or args.execute):
        raise ValueError('Receipt preparation requires one C instance and no training execution')
    environment=dict(os.environ,JAX_PLATFORMS='cuda',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/fg-matplotlib',
        PYTHONPATH=args.dreamer_root+':'+str(Path(__file__).resolve().parents[1]/'src'),
        JAX_COMPILATION_CACHE_DIR='/data/logdir/fg_abcd_v1/jax_cache')
    if args.execute:
        subprocess.run([sys.executable,'-c','import jax; d=jax.devices(); assert d and all(x.platform=="gpu" for x in d); print(d)'],env=environment,check=True)
    planned={}
    for instance in selected:
        source=None
        for frozen in instance['phases']:
            phase=dict(frozen)
            if source:
                phase['transfer']=(transfer_receipt(instance,source,phase) if args.execute or args.create_transfer
                    else str(transfer_path(source,phase)))
            if args.create_transfer:
                source=phase;continue
            if args.execute:
                run_phase(instance,phase,args.dreamer_root,environment,digest)
            elif phase.get('shared_source') and planned.get(phase['logdir'],0)>=phase['steps']:
                print(f"# Reuse protected shared node {phase['steps']} from {phase['logdir']}",flush=True)
            else:
                if source:
                    receipt_cmd=[sys.executable,str(Path(__file__).resolve()),'--manifest',args.manifest,'--budget',args.budget,
                        '--method','C','--fold',instance['fold'],'--seed',str(instance['seed']),'--create-transfer']
                    print(shlex.join(receipt_cmd),flush=True)
                print(shlex.join(command(instance,phase,args.dreamer_root,phase.get('transfer',''))),flush=True)
                planned[phase['logdir']]=max(planned.get(phase['logdir'],0),phase['steps'])
            source=phase
        if args.execute:
            final=instance['phases'][-1];meta=phase_metadata(instance,final)
            if meta['learning_audit']['actual_learning']:
                fold=next(f for f in manifest['folds'] if f['validation']==instance['fold'])
                point=checkpoint_path(final)
                result_root=Path(args.manifest).resolve().parent/f"{args.budget}_evaluation"/instance['fold']/instance['method']/str(instance['seed'])
                receipt=result_root/'evaluation_receipt.json'
                expected=dict(experiment_hash=digest,checkpoint=str(point),agent_hash=file_hash(point/'agent.pkl'))
                if result_root.exists():
                    if not receipt.exists() or json.loads(receipt.read_text())!=expected:
                        raise ValueError('Existing evaluation has no matching complete receipt; refusing overwrite')
                    continue
                evaluation=[sys.executable,str(Path(__file__).with_name('evaluate_fg_abcd.py')),'--manifest',args.manifest,
                    '--env-config',fold['validation_env'],'--model-config',str(point/'config.yaml' if final.get('full_checkpoint') else Path(final['logdir'])/'config.yaml'),
                    '--checkpoint',str(point),'--output',str(result_root),'--seed',str(instance['seed'])]
                subprocess.run(evaluation,env=environment,check=True)
                write_json(receipt,expected)
            else:print(f"{instance['method']}: policy learning gate failed; no formal comparison claimed",flush=True)


if __name__=='__main__':main()
