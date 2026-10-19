"""One independent evaluation account in a fresh CPU interpreter."""
import json
from pathlib import Path
import sys

from .fg_s3_e_quality_exam import execute_job
from .fg_s3_e_quality_exam_contract import validate_plan


if __name__=='__main__':
    p=validate_plan(json.loads(Path(sys.argv[1]).read_text()))
    result=execute_job(p,json.loads(sys.argv[2]),resume=sys.argv[3]=='resume',
        stop_at=None if sys.argv[4]=='end' else int(sys.argv[4]))
    print(json.dumps(result,ensure_ascii=False))
