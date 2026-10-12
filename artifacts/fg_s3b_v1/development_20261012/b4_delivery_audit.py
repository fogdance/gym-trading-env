"""Current-state B0-B4 evidence audit; no production model deserialization."""
from collections import Counter
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import xml.etree.ElementTree as ET

os.environ['JAX_PLATFORMS']='cpu'; os.environ['CUDA_VISIBLE_DEVICES']=''
GYM=Path(__file__).resolve().parents[3]; DREAMER=Path('/home/v/Documents/work/dreamerv3-dev')
sys.path[:0]=[str(GYM/'src'),str(DREAMER)]
import jax
import numpy as np
import yaml
from embodied.run.fg_s3a_checkpoint import load_source,validate_source
from embodied.run.fg_s3a_features import FeatureCache,teacher_step
from embodied.run.fg_s3a_train import Sampler
from embodied.run.fg_s3b_checkpoint import audit_bc,restore_bc
from embodied.run.fg_s3b_train import TrainingRows
from dreamerv3.fg_s3b_bc import WeightedActorBC
from dreamerv3.fg_s3a_actor import parameter_digests
from gym_trading_env.research.fg_s3a_labels import load_labels
from gym_trading_env.research.fg_s3b_advantage import audit_values
from gym_trading_env.research.fg_s3b_weights import audit_weights
from gym_trading_env.research.fg_replay import digest,iter_replay_rows,read_manifest,signed,write_json

OUT=Path(__file__).parent; ROOT=GYM/'artifacts/fg_s3b_v1'; BASE=ROOT/'actor_cpu_interface_v2'
vm=audit_values(ROOT/'values_full_v2'); wm=audit_weights(ROOT/'weights_full_v1')
assert vm['complete'] and wm['same_supervision_index_as_s3a']
certificates=Counter(); proofs=Counter(); branches=Counter(); funding=[]; elapsed=0.; peak=0
for shard in vm['shards']:
    cm=read_manifest(ROOT/'values_full_v2'/shard['directory']/'manifest.json')
    certificates.update(cm['summary']['certificates']); proofs.update(cm['summary']['proofs'])
    branches.update(cm['summary']['branches']); elapsed+=cm['elapsed_seconds']; peak=max(peak,cm['peak_rss_bytes'])
    cash=Decimal(cm['actual_account']['minimum_supervised_cash'])
    loss=Decimal(cm['funding']['one_interval_loss_upper']); required=Decimal(cm['funding']['required_cash'])
    assert cash>required and cash-loss>required
    funding.append(dict(contract=cm['contract'],actual_min_cash=str(cash),
        one_forced_worst_cash=str(cash-loss),required_cash=str(required),uniform_one_forced_then_TP_only='PASS'))
def receipt(name): return read_manifest(BASE/name/'training_receipt.json')
resumed,continuous,matched,full=[receipt(n) for n in ('certified_resumed','certified_continuous','matched','full')]
def state(r): return pickle.loads((Path(r['checkpoint'])/'state.pkl').read_bytes())
one,two=state(resumed),state(continuous)
assert parameter_digests(one['agent']['params'])==parameter_digests(two['agent']['params'])
for a,b in zip(jax.tree.leaves(one['s3b_bc_opt']),jax.tree.leaves(two['s3b_bc_opt'])):
    np.testing.assert_array_equal(a,b)
assert jax.tree.structure(one['s3b_bc_opt'])==jax.tree.structure(two['s3b_bc_opt'])
for key in ('order','cursor','epoch','rng','rows'):
    if key=='order': np.testing.assert_array_equal(one['sampler'][key],two['sampler'][key])
    else: assert one['sampler'][key]==two['sampler'][key]
assert resumed['classification']==continuous['classification']
def logs(name):
    rows=[]
    for path in (BASE/name).glob('metrics_attempt_*.jsonl'):
        rows += [json.loads(line) for line in path.read_text().splitlines()]
    return sorted(rows,key=lambda r:r['bc_updates'])
lr,lc,lm=logs('certified_resumed'),logs('certified_continuous'),logs('matched')
assert lr==lc and len(lr)==len(lm)==6
assert [r['batch_index_sha256'] for r in lm]==[r['batch_index_sha256'] for r in lc]
assert matched['identity']['feature_subset_sha256']==continuous['identity']['feature_subset_sha256']
for r in (resumed,continuous,matched,full):
    audit_bc(r['checkpoint'],r['identity'])
    assert r['freeze_audit']['frozen_keys']==203 and len(r['freeze_audit']['changed_keys'])==5
    assert not r['original_joint_optimizer_updated'] and not r['formal_gpu_training']
labels=GYM/'artifacts/fg_s3a_v1/labels_20261011_v2'; labels_manifest=load_labels(labels)
source=validate_source(GYM/'artifacts/fg_s3a_v1/smoke_real_v3/source/ckpt/step_000000000000',labels_manifest,diagnostic=True)
cache=FeatureCache(GYM/'artifacts/fg_s3a_v1/smoke_real_v3/features',source,labels)
rows=TrainingRows(cache,ROOT/'weights_full_v1','S3B_CERTIFIED',diagnostic=True)
agent=load_source(source,'/tmp/fg_s3b_policy_audit')
bc=WeightedActorBC(agent); sampler=Sampler(len(rows.indices),0)
probe,probe_index=cache.batch(np.arange(32)); p_old=bc.probabilities(probe,probe_index['mask'])
restore_bc(resumed['checkpoint'],bc,sampler,resumed['identity'])
p_new=bc.probabilities(probe,probe_index['mask']); assert np.max(np.abs(p_new-p_old))>0
trace=Path(labels_manifest['source']['dataset'])/labels_manifest['traces'][0]['directory']
tm=read_manifest(trace/'manifest.json'); carry=agent.init_policy(1)
with agent.n_actions.lock: agent.n_actions.value=0
max_error=0.; max_latent_error=0.; pos=0
for ordinal,step in iter_replay_rows(trace,tm):
    carry,feature,out=teacher_step(agent,carry,step)
    if pos<32 and ordinal==int(cache.index[0][pos]['ordinal']):
        max_latent_error=max(max_latent_error,float(np.max(np.abs(feature-probe[pos]))))
        max_error=max(max_error,float(abs(out['p_add'][0]-p_new[pos])))
        pos+=1
    if pos==32: break
assert pos==32 and max_latent_error==0 and max_error<=1e-6
configs=[yaml.safe_load((DREAMER/'configs/fg_s3b'/f'{m}.yaml').read_text())
         for m in ('S3A_FULL','S3A_MATCHED','S3B_CERTIFIED')]
baseline={k:v for k,v in configs[0].items() if k!='mode'}
assert all({k:v for k,v in c.items() if k!='mode'}==baseline for c in configs)
tests=[]
for filename in ('B1_B2_final.xml','B2_future_identity_final.xml','B3_final.xml'):
    suite=ET.parse(OUT/filename).getroot().find('testsuite')
    assert suite is not None and int(suite.attrib['failures'])==int(suite.attrib['errors'])==0
    tests.append(dict(file=filename,tests=int(suite.attrib['tests']),seconds=float(suite.attrib['time'])))
protected=json.loads((OUT/'B0_baseline.json').read_text())['protected_files']
drift=[p for p,h in protected.items() if digest(p)!=h]
development_files=[p for p in protected if Path(p).is_relative_to(GYM) or Path(p).is_relative_to(DREAMER)]
development_drift=[p for p in drift if p in development_files]
assert not development_drift,development_drift
drift_receipt=dict(status='DRIFT_OBSERVED' if drift else 'PASS',
    original_baseline_preserved=True,first_global_audit_exit_code=1 if drift else 0,
    changes=[dict(path=p,before=protected[p],after=digest(p)) for p in drift],
    attribution='Not determined; these are production paths outside both S3-B development repositories',
    development_existing_file_status='PASS',development_existing_files=len(development_files),
    action='No rollback, re-signing, signal, restart, or write to those paths')
write_json(OUT/'B4_protection_drift.json',signed(drift_receipt))
production=json.loads((OUT/'B4_source_metadata.json').read_text())
assert all(digest(p)==h for p,h in production['protected_files'].items())
forks=read_manifest(OUT/'B4_actual_forks.json')
result=dict(status='CPU_VERIFIED_WITH_PRODUCTION_DRIFT' if drift else 'PASS',scope='B0-B4 code and CPU verification; S3-A is complete; no formal S3 training',
    labels=402096,certificates=dict(certificates),proofs=dict(proofs),branches=dict(branches),
    certified_coverage=wm['summary']['certified_coverage'],matched_rows=wm['matched_rows'],
    weight_scale=read_manifest(ROOT/'weights_full_v1/scale.json')['scale'],
    weight_summary={k:v for k,v in wm['summary'].items() if k!='strata'},funding=funding,
    full_contract_seconds=elapsed,peak_rss_bytes=peak,
    actual_account_forks=sum(len(c['forks']) for c in forks['contracts']),
    data_manifest_hashes=dict(labels=digest(labels/'manifest.json'),values=digest(ROOT/'values_full_v2/manifest.json'),
                              weights=digest(ROOT/'weights_full_v1/manifest.json')),
    original_actor_cpu=dict(feature_rows=cache.rows,matched_feature_rows=len(rows.indices),
        total_model_keys=208,frozen_keys=203,changed_actor_keys=resumed['freeze_audit']['changed_keys'],
        original_updates=0,original_batches=0,bc_updates=6,continuous_vs_resume='BYTE_EXACT',
        all_update_metrics_equal=True,matched_vs_certified_batch_indices='BYTE_EXACT',
        policy_probability_max_error=max_error,past_latent_max_error=max_latent_error,
        probability_max_change=float(np.max(np.abs(p_new-p_old))),
        classification={r['identity']['mode']:r['classification'] for r in (full,matched,continuous)}),
    tests=tests,unique_new_tests=sum(t['tests'] for t in tests),protected_existing_source_files=len(protected),
    protected_source_drift=drift,development_existing_source_files=len(development_files),
    development_existing_source_drift=development_drift,global_protection_status=drift_receipt['status'],
    protected_production_checkpoint='PRODUCER_HASH_AND_MIDPOINT_HASH_PASS',
    source_checkpoint_condition='FUTURE_FORMAL_TRAIN_ADMISSION_ONLY',production_metadata=production,
    formal_gpu_training='NOT_RUN',dev='NOT_RUN',sealed_test='NOT_ACCESSED')
write_json(OUT/'B4_delivery_audit.json',signed(result))
print(json.dumps(dict(status=result['status'],tests=result['unique_new_tests'],source_files=len(protected),
    actor=result['original_actor_cpu']|{'classification':'SEE_RECEIPT'},forks=result['actual_account_forks']),indent=2))
