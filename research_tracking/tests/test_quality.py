"""Quality uses the existing tracker without erasing missing supervision."""
from research_tracking.metrics import aggregate_updates
from research_tracking.identity import FAMILIES


def test_quality_metrics_keep_task_denominators_and_missing_gradient():
    rows=[]
    for i,count,loss in ((1,2,1.),(2,6,3.)):
        rows.append(dict(env_steps=100,local_update=i,agent_update=i,valid_count=8,add_count=2,
            metrics={'quality/score_valid_count':count,'quality/score_loss':loss,
                     'quality/h0_sl_count':i,'quality/grad/soft_norm':None,
                     'opt/quality_actor_grad_norm':.2}))
    metrics=aggregate_updates(rows)
    assert metrics['train/quality/score_valid_count']==8
    assert metrics['train/quality/score_loss']==2.5
    assert metrics['train/quality/h0_sl_count']==3
    assert 'train/quality/grad/soft_norm' not in metrics
    assert metrics['train/optimizer/quality_actor_grad_norm']==.2
    assert FAMILIES['quality_v2']=='s3-quality-v2'
