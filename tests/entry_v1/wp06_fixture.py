"""Only current-contract built-in synthetic sources; never FORMAL authority."""
from functools import lru_cache
from pathlib import Path
import tempfile
from wp05_fixture import foundation
from test_wp06_contract import settings

@lru_cache(maxsize=2)
def source(stage='F1'):
    s=foundation(True)
    if stage=='F1':return s['corpus_root'],s['run_dir']
    from wp05_fixture import engineering_settings
    from gym_trading_env.research.fg_s3_e_quality_heads import preflight
    from gym_trading_env.research.fg_s3_e_quality_reader import build_join_index
    from embodied.run.fg_s3_e_quality_heads import HeadsRun,engineering_permit
    root=Path(tempfile.mkdtemp(prefix='wp06_f2_'))
    f2=engineering_settings(root,target=64,burnin=32,multiscale=True);p=preflight(f2)
    index=root/'join.sqlite';build_join_index(p.source,index)
    run=HeadsRun(p,index,root/'run',_probe=engineering_permit(p));run.run()
    return f2['corpus_root'],str(root/'run')

def engineering_settings(root,*,stage='F1',target=8,burnin=4):
    corpus,run=source(stage);s=settings(Path(corpus))
    s.update(source_stage=stage,source_run=run,source_checkpoint=str(Path(run)/'ckpt/step_000000000004'),
        target_steps=target,burnin_steps=burnin,prepared_dir=str(root/'prepared'),run_dir=str(root/'run'))
    return s
