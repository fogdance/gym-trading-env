"""Quality-specific preparation adapters for the shared S3-V2 lifecycle."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import pickle
from types import SimpleNamespace
import subprocess

import yaml

from dreamerv3.configuration import read_config_yaml
from dreamerv3.fg_s3_composition import identity_fields
from dreamerv3.fg_s3_quality_v2 import quality_extensions
from embodied.run.fg_checkpoint import file_hash,plain
from embodied.run.fg_s3_source_admission import inspect_source
from .fg_replay import read_manifest,write_json,signed,check_files
from .fg_s3_v2_train_pool import PoolOracleQuery,ObservedTrainingPool
from .fg_s3_quality_teacher_v2 import NativeQualityTeacher
from .fg_s3_quality_q_v2 import CertifiedQEngine,QualityOracle,build_quality_suffix
from gym_trading_env.envs.entry_contract import fingerprint,resolve_path
from gym_trading_env.envs.entry_env import FixedDirectionEntryEnv
from gym_trading_env.envs.entry_market_cache import EntryMarketCache
from gym_trading_env.rewards.single3_reward import Single3RewardConfig

PROJECT=Path(__file__).resolve().parents[3]


class QualityPool(PoolOracleQuery):
    @property
    def teacher_identities(self):
        return {q.teacher.identity_hash for q in self.queries.values()}

    def label_for(self,env,state_id):
        label=self.queries[env.contract].last_label
        if label is None or label['state_id'] != state_id:
            raise ValueError('Quality pending decision has no matching native proof')
        return copy.deepcopy(label)

    def close(self):
        for query in self.queries.values(): query.close()


def _member(row,source,settings, *, build=False):
    cache=EntryMarketCache(1)
    def factory(): return FixedDirectionEntryEnv(config_path=row['env_config'],market_cache=cache)
    env=factory()
    try:
        teacher=NativeQualityTeacher(env,factory,fold='TRAIN',source=source)
        engine=None
        if settings['quality_stage']=='CONTROLLED_FULL':
            engine=CertifiedQEngine(factory,max_states=settings['quality_q_max_states'],
                max_seconds=settings['quality_q_max_seconds'],max_suffix_bars=settings['quality_q_max_suffix_bars'])
        path=Path(row['env_config']).with_name('strict_suffix.pkl')
        if build:
            builder=factory()
            try: suffix=build_quality_suffix(builder,source=source)
            finally: builder.close()
            path.write_bytes(pickle.dumps(suffix,protocol=5))
        else:
            suffix=pickle.loads(path.read_bytes())
            if suffix.rules != teacher.rules or suffix.identity['source'] != {
                    **source, 'target_reward_schema':env.reward_config.schema,
                    'candidate_hash':suffix.identity['source']['candidate_hash'],
                    'builder_code_sha256':file_hash(Path(__file__).with_name('fg_s3_quality_q_v2.py'))}:
                raise ValueError('Prepared strict suffix source changed')
        return QualityOracle(teacher,engine,suffix=suffix)
    finally: env.close()


def _build_member_job(args):
    row,source,settings=args
    q=_member(row,source,settings,build=True)
    try: return row['contract'],q.identity
    finally: q.close()


def _derive_rows(settings,directory):
    """Reuse the signed TRAIN whitelist; only a new reward snapshot is derived."""
    if settings.get('quality_fixture_env'):
        if settings['scope'] != 'diagnostic' or settings['source_kind'] != 'direct_init':
            raise ValueError('Synthetic fixture source is diagnostic direct_init only')
        p=Path(settings['quality_fixture_env'])
        cfg=read_config_yaml(p)
        price=str(Path(cfg['trading']['data_path']).resolve())
        contract=str(resolve_path(p,cfg['entry']['contract_config']))
        rows=[dict(contract=cfg['entry']['contract'],role='train',eligible_for_training=True,
            env_config=str(p),env_sha256=file_hash(p),path=price,sha256=file_hash(price),
            contract_config=contract,contract_sha256=file_hash(contract),calendar_complete=False,
            rows=sum(1 for _ in Path(price).open())-1)]
        training_hash=fingerprint(rows)
    else:
        eroot=Path(settings['env_receipt'])
        em=read_manifest(eroot/'manifest.json')
        inputs=read_manifest(eroot/'inputs/inputs.json')
        if (em['status']!='PASS' or em.get('clock_protocol')!='S3_MARKET_MINUTE'
                or em['files']['inputs/inputs.json']!=file_hash(eroot/'inputs/inputs.json')):
            raise ValueError('Quality source needs signed complete market-minute TRAIN inputs')
        rows=inputs['views']['C1']['rows']
        selected=settings['contracts']
        selected=[settings['contract']] if selected is None else selected
        if selected!='ALL_TRAIN':
            if len(set(selected))!=len(selected) or set(selected)-{r['contract'] for r in rows}:
                raise ValueError('Quality contracts escaped the TRAIN whitelist')
            rows=[r for r in rows if r['contract'] in selected]
        training_hash=inputs['original_content_hash']
    derived=[]; frozen={}; member_files={}
    for original in rows:
        if original['role']!='train' or not original['eligible_for_training']:
            raise ValueError('Quality labels cannot read DEV/TEST future prices')
        row=copy.deepcopy(original)
        own_files={row[k]:row[h] for k,h in (('path','sha256'),('env_config','env_sha256'),('contract_config','contract_sha256'))}
        check_files({row[k]:row[h] for k,h in (('path','sha256'),('env_config','env_sha256'),('contract_config','contract_sha256'))})
        folder=directory/'inputs'/row['contract'];folder.mkdir(parents=True,exist_ok=False)
        config=read_config_yaml(row['env_config'])
        config['entry'].pop('state_path',None)  # Counterfactuals never persist old state.
        config['entry']['contract_config']=row['contract_config']
        config['trading']['data_path']=row['path']
        if config['entry'].get('strategy_config'):
            master=Path(config['entry']['strategy_config'])
            old_master=read_config_yaml(master)
            if not settings.get('quality_fixture_env') and inputs['frozen_files'].get(str(master))!=file_hash(master):
                raise ValueError('Unbound original TRAIN strategy snapshot')
            rules=copy.deepcopy(old_master);rules['reward']=settings['reward']
            target=folder/'single3.yaml';target.write_text(yaml.safe_dump(rules,sort_keys=False))
            config['entry']['strategy_config']=str(target)
            frozen[str(master)]=file_hash(master)
            frozen[str(target)]=file_hash(target)
            own_files.update({str(master):file_hash(master),str(target):file_hash(target)})
        else:
            direction=resolve_path(row['env_config'],config['entry']['direction_config'])
            config['entry']['direction_config']=str(direction)
            config['entry']['reward']=settings['reward']
            frozen[str(direction)]=file_hash(direction)
            own_files[str(direction)]=file_hash(direction)
        target=folder/'env.yaml';target.write_text(yaml.safe_dump(config,sort_keys=False))
        frozen[row['env_config']]=row['env_sha256']
        row.update(env_config=str(target),env_sha256=file_hash(target))
        for k,h in (('path','sha256'),('env_config','env_sha256'),('contract_config','contract_sha256')):
            frozen[row[k]]=row[h]
            own_files[row[k]]=row[h]
        derived.append(row)
        member_files[row['contract']]=own_files
    return derived,frozen,training_hash,member_files


def prepare_quality(settings):
    from embodied.run.fg_s3_v2_train import execution_code
    directory=Path(settings['prepared_dir'])
    if directory.exists(): raise FileExistsError('Quality preparation retained; choose a new experiment')
    directory.mkdir(parents=True)
    rows,frozen,training_hash,member_files=_derive_rows(settings,directory)
    prediction_source=None
    if settings['quality_stage']=='CONTROLLED_FULL':
        from .fg_s3_quality_catalog_v2 import require_prediction_admission
        admission_path=Path(settings['quality_prediction_admission_receipt'])
        admission=require_prediction_admission(admission_path)
        if (file_hash(admission_path)!=settings['quality_prediction_admission_sha256'] or
                set(admission['trained_contracts'])!={row['contract'] for row in rows}):
            raise ValueError('Controlled quality source differs from prediction admission TRAIN split')
        prediction_source=dict(admission_receipt=str(admission_path),
            admission_sha256=file_hash(admission_path),checkpoint=admission['checkpoint'],
            checkpoint_manifest_sha256=admission['checkpoint_manifest_sha256'])
    pool=signed(dict(product='FG',files=rows,source_training_hash=training_hash))
    write_json(directory/'train_pool.json',pool)
    selector=dict(strategy_mode='entry_dataset',role='train',manifest_path=str(directory/'train_pool.json'),
                  scheduler=dict(mode='least_progress',chunk_bars=settings['env_slice_bars']))
    (directory/'selector.yaml').write_text(yaml.safe_dump(selector,sort_keys=False))
    sources={r['contract']:dict(files=member_files[r['contract']],training_hash=training_hash,
        max_market_steps=settings['quality_max_market_steps'],
        calendar_complete=bool(r['calendar_complete']),row_hash=fingerprint(r),fold='TRAIN') for r in rows}
    jobs=[(row,sources[row['contract']],settings) for row in rows]
    if len(jobs)>1:
        import os,multiprocessing,psutil
        from concurrent.futures import ProcessPoolExecutor
        workers=max(1,min(len(jobs),os.cpu_count() or 1,psutil.virtual_memory().available//(512*2**20)))
        with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as executor:
            identities=dict(executor.map(_build_member_job,jobs))
    else: identities=dict(_build_member_job(job) for job in jobs)
    query_identity=dict(protocol='FG_S3_V2_ORACLE_POOL_V1',fold='TRAIN',manifest_hash=pool['content_hash'],
                        queries={c:fingerprint(identity) for c,identity in sorted(identities.items())})
    write_json(directory/'quality_pool.json',signed(dict(sources=sources,queries=identities,frozen=frozen)))
    first=FixedDirectionEntryEnv(config_path=rows[0]['env_config'])
    try:
        binding=signed(dict(protocol='FG_S3_QUALITY_TRAIN_BINDING_V1',fold='TRAIN',
            reward_schema=first.reward_config.schema,quality_stage=settings['quality_stage'],
            score_policy='FG_S3_QUALITY_TP_MARKET_DAY_V1',source=dict(training_hash=training_hash,
                dataset_sha256=pool['content_hash'],reward_params_hash=first.reward_config.hash),
            observation_identity=first.observation_identity,diagnostic=settings['scope']=='diagnostic',
            oracle_identity=fingerprint(query_identity),quality_pool_hash=file_hash(directory/'quality_pool.json')))
    finally: first.close()
    if prediction_source is not None:
        binding=signed({**{k:v for k,v in binding.items() if k!='content_hash'},
                        'prediction_source':prediction_source})
    write_json(directory/'quality_binding.json',binding)
    catalog=SimpleNamespace(manifest=binding,manifest_sha256=file_hash(directory/'quality_binding.json'))
    model=None
    if settings['source_kind']=='direct_init':
        if settings['model_preset']!='cpu_smoke': raise ValueError('Quality direct_init requires explicit diagnostic CPU preset')
        from .fg_s3_v2_gpu_check import small_config
        model=small_config()
    spec=inspect_source(settings['source_kind'],catalog=catalog,path=settings['source_checkpoint'],config=model,
        seed=model['seed'] if model else None,diagnostic=settings['scope']=='diagnostic',
        agent_extension=quality_extensions(settings),composition=settings.get('resolved_composition'))
    model=copy.deepcopy(spec['target_config'])
    model.update(batch_size=settings['batch_size'],batch_length=64,report_length=64)
    write_json(directory/'model_config.json',signed(dict(config=plain(model),agent_extension=quality_extensions(settings),**identity_fields(settings))))
    write_json(directory/'source_admission.json',spec)
    write_json(directory/'settings.json',signed(dict(settings=settings)))
    (directory/'experiment.yaml').write_bytes(Path(settings['config_path']).read_bytes())
    files={str(p.relative_to(directory)):file_hash(p) for p in directory.rglob('*') if p.is_file()}
    result=signed(dict(protocol='FG_S3_V2_PREPARED_V1',status='PREPARED_NO_TRAINING',code=execution_code(),
        oracle_identity=fingerprint(query_identity),**identity_fields(settings),
        versions={str(r):subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()
                  for r in (PROJECT,PROJECT.parent/'dreamerv3')},
        dirty={str(r):subprocess.check_output(['git','status','--porcelain'],cwd=r,text=True).splitlines()
               for r in (PROJECT,PROJECT.parent/'dreamerv3')},files=files,
        source_update_offset=0,source_model_updates=spec['origin']['step'],
        clock_protocol='S3_MARKET_MINUTE',new_environment_steps=0,
        training_identity=dict(candidate='C1',pool='QUALITY_TRAIN',teacher_source_kind='quality_native_train',
            **binding['source'],observation_identity=binding['observation_identity']),
        diagnostic=settings['scope']=='diagnostic',budget_unit='environment_advances',formal_1m='NOT_RUN',
        calibration_status='NOT_CALIBRATED',q_fusion_status='ADMITTED' if settings['quality_q_admitted'] else 'FULL_Q_PENDING'))
    write_json(directory/'manifest.json',result)
    return result


def load_quality_pool(settings,directory):
    directory=Path(directory)
    manifest=read_manifest(directory/'manifest.json')
    check_files({str(directory/k):h for k,h in manifest['files'].items()})
    meta=read_manifest(directory/'quality_pool.json')
    check_files(meta['frozen'])
    pool=read_manifest(directory/'train_pool.json')
    queries={row['contract']:_member(row,meta['sources'][row['contract']],settings) for row in pool['files']}
    q=QualityPool(queries,pool['content_hash'])
    if q.identity_hash!=manifest['oracle_identity'] or {c:x.identity for c,x in queries.items()}!=meta['queries']:
        q.close();raise ValueError('Prepared quality pool identity changed')
    return q,ObservedTrainingPool(directory/'selector.yaml')
