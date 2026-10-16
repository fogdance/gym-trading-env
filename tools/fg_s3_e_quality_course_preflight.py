#!/usr/bin/env python3
"""Read-only E + Quality TP course diagnostics. This command cannot train."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src'), str(PROJECT.parent / 'dreamerv3')]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, help='JSON course with explicit spec_mode: draft or formal')
    parser.add_argument('--report', help='New independent directory; only a valid explicit draft can write diagnostics')
    args = parser.parse_args(argv)
    try:
        from gym_trading_env.research.fg_s3_e_quality_course_preflight import preflight
        report = preflight(args.spec, report_dir=args.report)
        code = report['exit_code']
    except (ValueError, OSError, OverflowError, subprocess.CalledProcessError) as exc:
        code = 2
        report = dict(schema='FG_S3_E_QUALITY_COURSE_PREFLIGHT_V1',
            status='INVALID_SPEC_OR_SOURCE', admission_status='NOT_ADMITTED',
            authenticated=False, execution_admitted=False, exit_code=code, blocking_reasons=[str(exc)])
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
