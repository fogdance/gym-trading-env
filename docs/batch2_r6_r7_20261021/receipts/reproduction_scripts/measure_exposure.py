"""Untimed denominators, verified against every measured native batch hash."""
import json,pathlib,sys
import numpy as np
from batch2_reader_benchmark import open_case,tree_digest
from dreamerv3.fg_s3_e_quality_actor import valid_l2_targets
from dreamerv3.fg_s3_e_quality_heads import model_extensions
from dreamerv3.fg_s3_quality_v2 import pair_identity
base=pathlib.Path(sys.argv[1]);out=base/'docs/batch2_r6_r7_20261021/receipts'
reference=json.loads((out/'performance_final/ALL_RESULTS.json').read_text())
config=json.loads((base/'artifacts/batch2/bench_v1/config.json').read_text());records=[]
for case in reference:
    raw=open_case(config,case['mode']);digests=[];counts=dict(target_positions=0,effective_target_positions=0,
        target_market_minutes=0,natural_valid=0,auxiliary_valid=0,strict_pair_eligible=0,quality_pair_eligible=0)
    try:
        next(raw);next(raw)
        for _ in range(case['timed_batches']):
            data=next(raw);digests.append(tree_digest(data));d={k:v[:,5:] for k,v in data.items()}
            shape=d['is_first'].shape;counts['target_positions']+=int(np.prod(shape))
            counts['target_market_minutes']+=int(d['market_advance'].sum())
            if case['mode'] in ('f1','multi'):valid=np.ones(shape,bool)
            elif case['mode']=='f2':
                cfg=model_extensions(8,4);valid=np.zeros(shape,bool)
                if cfg['quality_outcome_lambda']>0:
                    for key in ['event','touch','adverse']:
                        value=d['quality_'+key+'_valid'];valid|=value.reshape(*shape,-1).any(-1)
                if cfg['quality_score_lambda']>0:valid|=d['quality_score_valid']
                if cfg['quality_mono_lambda']>0:valid|=(d['quality_event_valid'][...,:5].sum(-1)>=2)
            else:
                valid=np.asarray(valid_l2_targets(d,d['action_mask'],d['is_first'],d['is_last'],d['is_terminal']))
                natural=d['f3_natural'];aux=np.where((~natural).any(),~natural,np.ones_like(natural))
                counts['natural_valid']+=int(np.count_nonzero(valid & natural))
                counts['auxiliary_valid']+=int(np.count_nonzero(valid & ~natural))
                av=valid & aux;pos=av & (d['teacher_action']==1);neg=av & (d['teacher_action']==0)
                identity=np.asarray(pair_identity(d,d['is_first']))
                strict=identity & pos[:,:,None] & neg[:,None,:]
                known=d['quality_score_valid'] & d['quality_outcome_valid'] & av
                score=np.where(known,d['quality_score'],0.)
                quality=identity & known[:,:,None] & known[:,None,:] & (score[:,:,None]-score[:,None,:]>=15) & strict
                counts['strict_pair_eligible']+=int(strict.sum());counts['quality_pair_eligible']+=int(quality.sum())
            counts['effective_target_positions']+=int(np.count_nonzero(valid))
        for rep in case['repeats']:
            for version in ['before','after']:assert digests==rep[version]['batch_digests']
        assert counts['effective_target_positions']>0
        records.append(dict(mode=case['mode'],seen=case['seen'],timed_batches=case['timed_batches'],
            counts=counts,all_measured_batch_hashes_match=True,scope='LEGAL_SYNTHETIC_BATCH_BYTES_ONLY',
            artificial_history_contributes_no_training_labels=True))
    finally:raw.close()
(out/'PERFORMANCE_EXPOSURE_DENOMINATORS.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(dict(cases=len(records),all_batch_hashes_match=True,optimizer_updates=0)))
