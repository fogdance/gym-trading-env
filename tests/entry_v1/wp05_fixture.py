"""Current-contract synthetic F1 evidence; never a formal trained foundation."""
from pathlib import Path
from functools import lru_cache
import tempfile
from test_wp05_contract import settings

@lru_cache(maxsize=2)
def foundation(multiscale=False):
 from wp04_fixture import synthetic_foundation
 return synthetic_foundation(Path(tempfile.mkdtemp(prefix='wp05_f1_')),multiscale=multiscale)

def engineering_settings(root,target=8,burnin=4,*,multiscale=False):
 f1=foundation(multiscale)
 s=settings(Path(f1['corpus_root']))
 s.update(target_steps=target,burnin_steps=burnin,prepared_dir=str(root/'prepared'),run_dir=str(root/'run'),source_f1_run=f1['run_dir'],source_f1_checkpoint=str(Path(f1['run_dir'])/'ckpt/step_000000000004'))
 return s
