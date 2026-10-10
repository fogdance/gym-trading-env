"""Frozen whole-lifecycle ABC datasets. Never open sealed blind prices."""
import copy
import json
from pathlib import Path

import pandas as pd
import yaml

from gym_trading_env.envs.entry_contract import DirectionSchedule, fingerprint, read_yaml, timestamp
from .entry_pipeline import file_hash
from .fg_entry_v1 import prepare_main_contracts, write_json


def choose_contracts(records, cutoff, months=None):
    cutoff=pd.Timestamp(cutoff).normalize()
    start=cutoff-pd.DateOffset(months=months)+pd.Timedelta(days=1) if months else None
    return [r for r in records if r['complete_lifecycle']
            and pd.Timestamp(r['last_day']) <= cutoff
            and (start is None or pd.Timestamp(r['first_day']) >= start)]


def build_abcd(config_path):
    config_path=Path(config_path).resolve()
    cfg=read_yaml(config_path)
    output=Path(cfg['output']).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite an ABC experiment')
    master=Path(cfg['master_config']).resolve()
    raw=read_yaml(master)
    schedule=DirectionSchedule(raw['directions'])
    source=json.loads(Path(cfg['export_manifest']).read_text())
    blind=timestamp(cfg['blind_start'])
    selected=[]
    # Filter by metadata BEFORE opening files. Blind identity is not blind price.
    for item in source['files']:
        if timestamp(item['last_timestamp']) >= blind:
            continue
        if timestamp(item['first_timestamp']) >= blind:
            continue
        if item['contract'] in raw['data']['excluded_contracts']:
            continue
        selected.append(item)
    names=[r['contract'] for r in selected]
    if len(set(names)) != len(names):
        raise ValueError('Repeated main segments require explicit continuation')
    output.mkdir(parents=True)
    metas=prepare_main_contracts(dict(source,files=selected),raw['costs'],
                                output/'contracts',explicit_exits=True)
    records=[]
    days_by_contract={}
    for item,meta in zip(selected,metas):
        frame=pd.read_csv(item['path'])
        days=pd.to_datetime(frame.trading_date.astype(str),format='mixed').dt.normalize()
        days_by_contract[item['contract']]=sorted(set(days.dt.strftime('%Y%m%d').astype(int)))
        actual=pd.DatetimeIndex([timestamp(t) for t in frame.Date])
        counts={d:0 for d in ('LONG','SHORT','NO_TRADE')}
        for t,day in zip(actual,frame.trading_date):
            counts[schedule.at(item['contract'],t,str(day))]+=1
        if max(actual)>=blind:
            raise ValueError('Blind price leaked through source metadata')
        env=copy.deepcopy(raw['environment'])
        env['entry'].update(contract=item['contract'],contract_config=meta,strategy_config=str(master))
        complete=bool(item.get('main_exit_at') and item.get('next_main_contract'))
        if complete:
            env['entry']['main_exit_at']=item['main_exit_at']
        env['trading'].update(future_symbol=item['contract'],currency_pair=item['contract'].split('.')[-1],data_path=item['path'])
        env_path=output/'contracts'/f"{item['contract'].replace('.','_')}_env.yaml"
        env_path.write_text(yaml.safe_dump(env))
        records.append(dict(contract=item['contract'],path=item['path'],sha256=item['sha256'],
            env_config=str(env_path),env_sha256=file_hash(env_path),contract_config=meta,
            contract_sha256=file_hash(meta),rows=len(frame),first_day=str(days.min().date()),
            last_day=str(days.max().date()),trading_days=days.nunique(),direction_rows=sum(counts[d] for d in ('LONG','SHORT')),
            direction_counts=counts,complete_lifecycle=complete,calendar_complete=True,
            calendar_basis='user_confirmed_observed_clock',first_timestamp=str(actual[0]),last_timestamp=str(actual[-1])))
    folds=[]
    instances=[]
    coverage=['# ABC whole-lifecycle coverage','', '| Fold | Dataset | Contracts | Bars | Days | LONG | SHORT | NO_TRADE |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for fold in cfg['folds']:
        val=[r for r in records if r['contract']==fold['validation']]
        if len(val)!=1 or not val[0]['complete_lifecycle']:
            raise ValueError('Validation needs one complete real lifecycle')
        val=val[0]
        if pd.Timestamp(val['first_day'])<=pd.Timestamp(fold['cutoff']):
            raise ValueError('Validation overlaps train cutoff')
        datasets={}
        for label,months in [('all',None),('recent',cfg['recent_months'])]:
            train=choose_contracts(records,fold['cutoff'],months)
            if not train:
                raise ValueError('Empty whole-lifecycle training set')
            destination=output/fold['validation'].split('.')[-1]/label
            destination.mkdir(parents=True)
            files=[dict(r,role='train',eligible_for_training=True) for r in train]
            files.append(dict(val,role='validation',eligible_for_training=False))
            roles=dict(train=sorted({d for r in train for d in days_by_contract[r['contract']]}),
                       validation=days_by_contract[val['contract']],test=[int(cfg['blind_start'].replace('-',''))])
            if not max(roles['train'])<min(roles['validation'])<=max(roles['validation'])<min(roles['test']):
                raise ValueError('Role boundaries overlap')
            manifest=dict(schema=1,product='FG',roles=roles,files=files,direction_path=str(master),
                direction_sha256=file_hash(master),blind_price_access=False,
                test_status='SEALED_METADATA_ONLY',fill_model='minute_close_confirm_v1',reward='60/(60+T_minutes)')
            manifest['content_hash']=fingerprint(manifest)
            manifest_path=destination/'manifest.json'
            write_json(manifest_path,manifest)
            selector=destination/'train_dataset.yaml'
            selector.write_text(yaml.safe_dump(dict(strategy_mode='entry_dataset',role='train',
                manifest_path=str(manifest_path),scheduler=cfg['scheduler'])))
            datasets[label]=dict(manifest=str(manifest_path),selector=str(selector),content_hash=manifest['content_hash'])
            coverage.append('| '+ ' | '.join(map(str,[fold['validation'],label,len(train),sum(r['rows'] for r in train),
                sum(r['trading_days'] for r in train),*[sum(r['direction_counts'][d] for r in train) for d in ('LONG','SHORT','NO_TRADE')]]))+' |')
        folds.append(dict(fold,validation_env=val['env_config'],datasets=datasets))
        for budget in ('pilot','formal'):
            spec=cfg[budget]
            if budget=='pilot' and fold['validation']!=spec['fold']:
                continue
            for seed in spec['seeds']:
                for method in 'ABC':
                    phases=[('train','all' if method=='A' else 'recent',spec['steps'])] if method!='C' else [
                        ('pretrain','all',spec['pretrain_steps']),('recent','recent',spec['recent_steps'])]
                    if sum(p[2] for p in phases)!=spec['steps']:
                        raise ValueError('C total budget differs')
                    instances.append(dict(fold=fold['validation'],budget=budget,method=method,seed=seed,status='NOT_RUN',
                        phases=[dict(phase=p,dataset=datasets[d],steps=n,logdir=str(Path(cfg['log_root'])/fold['validation'].split('.')[-1]/budget/method/str(seed)/p)) for p,d,n in phases]))
    result=dict(schema=1,config=str(config_path),config_sha256=file_hash(config_path),
        export_manifest_sha256=file_hash(cfg['export_manifest']),folds=folds,instances=instances,
        blind_price_access=False,conditional_direction_only=True)
    result['content_hash']=fingerprint(result)
    write_json(output/'experiment_manifest.json',result)
    (output/'coverage.md').write_text('\n'.join(coverage)+'\n\nObserved-clock calendar; not independently exchange-verified. Validation FG409 is SHORT only.\n')
    return result
