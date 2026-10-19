import json
from pathlib import Path
import sys

from wp08_fixture import native_source,ARTIFACTS
from gym_trading_env.research.fg_s3_e_quality_exam_compare import run_case

p=json.loads(Path(sys.argv[1]).read_text());job=json.loads(sys.argv[2])
source=native_source(ARTIFACTS/'C_native_04/source')
root=Path(p['output_dir'])/job['contract']/job['label']
result=run_case(p,source,root,job)
print(json.dumps(result))
