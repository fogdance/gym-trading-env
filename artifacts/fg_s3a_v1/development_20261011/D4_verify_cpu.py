"""Reproduce the saved real-P4 CPU resume and updated policy proof; no training."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'

import importlib.metadata
import json
import pickle
from pathlib import Path
import sys

GYM=Path('/home/v/Documents/work/gym-trading-env2')
DREAMER=Path('/home/v/Documents/work/dreamerv3-dev')
sys.path[:0]=[str(DREAMER),str(GYM/'src')]

import jax
import numpy as np
from dreamerv3.configuration import read_config_yaml
from dreamerv3.fg_s3a_actor import ActorBC,parameter_digests
from embodied.run.fg_checkpoint import file_hash,learning_spec,plain
from embodied.run.fg_s3a_checkpoint import audit_bc,load_source,restore_bc,validate_source
from embodied.run.fg_s3a_features import FeatureCache,teacher_step
from embodied.run.fg_s3a_train import Sampler
from gym_trading_env.research.fg_replay import iter_replay_rows,read_manifest,signed,write_json
from gym_trading_env.research.fg_s3a_labels import load_labels


def same_tree(a,b):
    assert jax.tree.structure(a)==jax.tree.structure(b)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)


def main():
    receipt=Path(__file__).resolve().parent
    root=GYM/'artifacts/fg_s3a_v1/smoke_real_v3'
    labels=GYM/'artifacts/fg_s3a_v1/labels_20261011_v2'
    resumed=read_manifest(root/'train/training_receipt.json')
    continuous=read_manifest(root/'train_continuous/training_receipt.json')
    assert resumed['identity']==continuous['identity'] and resumed['bc_updates']==continuous['bc_updates']==20
    for run in (resumed,continuous):audit_bc(run['checkpoint'],run['identity'])
    a=pickle.loads((Path(resumed['checkpoint'])/'state.pkl').read_bytes())
    b=pickle.loads((Path(continuous['checkpoint'])/'state.pkl').read_bytes())
    same_tree(a['agent']['params'],b['agent']['params'])
    same_tree(a['s3a_bc_opt'],b['s3a_bc_opt']);same_tree(a['sampler'],b['sampler'])
    initial=read_manifest(receipt/'D4_smoke_10.json')
    logs=lambda p:[json.loads(line) for line in p.read_text().splitlines()]
    joined=logs(root/'train'/initial['log_file'])+logs(root/'train'/resumed['log_file'])
    assert joined==logs(root/'train_continuous'/continuous['log_file'])

    source=validate_source(root/'source/ckpt/step_000000000000',load_labels(labels),diagnostic=True)
    cache=FeatureCache(root/'features',source,labels)
    agent=load_source(source,root/'verification_only',device='cpu')
    bc=ActorBC(agent);sampler=Sampler(cache.rows,0)
    x,index=cache.batch(np.arange(32));before=bc.probabilities(x,index['mask'])
    restore_bc(resumed['checkpoint'],bc,sampler,resumed['identity'])
    after=bc.probabilities(x,index['mask']);probability_delta=float(np.max(np.abs(after-before)))
    assert probability_delta>0
    agent.sync_policy_from_train(reset_counter=True);carry=agent.init_policy(1)
    lm=load_labels(labels);rec=lm['traces'][0]
    trace=Path(lm['source']['dataset'])/rec['directory'];manifest=read_manifest(trace/'manifest.json')
    pos=0;error=0.
    for ordinal,step in iter_replay_rows(trace,manifest):
        carry,feat,out=teacher_step(agent,carry,step)
        if ordinal==int(index[pos]['ordinal']):
            np.testing.assert_array_equal(feat,x[pos])
            error=max(error,abs(float(out['p_add'][0])-float(after[pos])))
            pos+=1
            if pos==len(x):break
    assert pos==32 and error<=1e-6
    audit=bc.freeze_audit()
    proof=signed(dict(status='PASS',scope='RANDOM_SMALL_ORIGINAL_AGENT_REAL_C1_P4_BFLOAT16_CPU',
        source_manifest_sha256=source['manifest_sha256'],feature_manifest_sha256=file_hash(cache.root/'manifest.json'),
        labels_manifest_sha256=file_hash(labels/'manifest.json'),
        updates=20,resume_split=[10,10],all_model_parameters_byte_equal=True,
        independent_optimizer_byte_equal=True,sampler_byte_equal=True,all_update_metrics_equal=True,
        frozen_keys=audit['frozen_keys'],total_keys=audit['total_keys'],changed_keys=audit['changed_keys'],
        old_joint_updates=audit['original_updates'],old_joint_batches=audit['original_batches'],
        updated_policy_probability_max_error=error,actor_probability_change=probability_delta,
        teacher_latents_unchanged_after_BC=True,policy_rows_checked=32,
        jax_devices=[str(d) for d in jax.devices()],production_payload_loaded=False,
        formal_gpu_training=False,dev_evaluated=False,classification=resumed['classification']))
    write_json(receipt/'D4_real_resume_and_policy.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='classification'},indent=2))

    # Metadata only: do not unpickle the running production checkpoint.
    production=Path('/data/logdir/fg_single3/gate_margin_v2/C1_M1_1M')
    tag=json.loads((production/'single3_identity.json').read_text());identity=tag['identity']
    metadata_node=production/'ckpt/step_000000008000'
    config=read_config_yaml(metadata_node/'config.yaml')
    assert config['agent']==identity['agent_config']
    assert plain(learning_spec(config))==identity['learning']
    assert all(file_hash(DREAMER/name)==h for name,h in identity['code_sha256'].items())
    assert all(importlib.metadata.version(name)==v for name,v in identity['packages'].items())
    width=config['agent']['dyn']['rssm']['deter']+config['agent']['dyn']['rssm']['stoch']*config['agent']['dyn']['rssm']['classes']
    write_json(receipt/'D4_production_metadata_compatibility.json',signed(dict(status='PASS_METADATA_ONLY',
        source_identity_hash=tag['identity_hash'],metadata_checkpoint=str(metadata_node),
        config_manifest_sha256=file_hash(metadata_node/'config.yaml'),
        architecture_and_learning_exact=True,core_files_match=len(identity['code_sha256']),
        package_versions_match=len(identity['packages']),parameter_payload_loaded=False,
        full_model_runtime_compatibility_verified=False,feature_width=width,
        projected_full_features_bytes=lm['summary']['counts']['eligible']*width*4)))


if __name__=='__main__':main()
