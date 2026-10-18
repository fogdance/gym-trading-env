"""Immutable WP04 engineering evidence; never a formal trained foundation."""
from pathlib import Path
import json
from test_wp05_contract import settings
BASE=Path('/home/v/Documents/work/research/fg-s3-e-quality-wp04-20261018/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp04/G0_restore_review_final/wp04_cold0')

def engineering_settings(root,target=8,burnin=4):
 f1=json.loads((BASE/'settings.json').read_text())
 s=settings(Path(f1['corpus_root']))
 s.update(target_steps=target,burnin_steps=burnin,prepared_dir=str(root/'prepared'),run_dir=str(root/'run'),source_f1_run=str(BASE/'continuous'),source_f1_checkpoint=str(BASE/'continuous/ckpt/step_000000000004'))
 return s
