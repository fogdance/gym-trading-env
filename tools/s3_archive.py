#!/usr/bin/env python3
"""Inspect, archive, verify, publish and list immutable S3 research evidence."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE']='1'

from s3_archive import DEFAULT_ROOT, LEVELS
from s3_archive.io import ArchiveError, atomic_json, fingerprint, independent, read_json


def arguments():
    p=argparse.ArgumentParser(description=__doc__,epilog='Examples: inspect --run-dir RUN --checkpoint 800000 --eval-dir EVAL --json-out PLAN; archive --plan PLAN --level evidence --preserve-checkpoint --execute; verify --archive-dir ARCHIVE; verify --archive-dir ARCHIVE --mode eval --device gpu --python TRAINER_PY; publish --archive-dir ARCHIVE; list')
    sub=p.add_subparsers(dest='command',required=True)
    i=sub.add_parser('inspect',help='Read-only source and dependency inspection')
    i.add_argument('--run-dir',required=True);i.add_argument('--checkpoint',required=True);i.add_argument('--eval-dir',required=True)
    i.add_argument('--prepared');i.add_argument('--family');i.add_argument('--publication');i.add_argument('--json-out')
    i.add_argument('--milestone-tag',action='append',default=[])
    i.add_argument('--archive-root',default=DEFAULT_ROOT)
    a=sub.add_parser('archive',help='Dry-run by default; --execute explicitly preserves bytes')
    a.add_argument('--plan',required=True);a.add_argument('--level',choices=['evidence','eval','full'],default='evidence')
    a.add_argument('--preserve-checkpoint',action='store_true');a.add_argument('--reserve-gib',type=float,default=10)
    m=a.add_mutually_exclusive_group();m.add_argument('--dry-run',action='store_true');m.add_argument('--execute',action='store_true')
    v=sub.add_parser('verify',help='Hash evidence, or explicitly run isolated GPU validation')
    v.add_argument('--archive-dir',required=True);v.add_argument('--mode',choices=['hashes','eval','resume-smoke'],default='hashes')
    v.add_argument('--python');v.add_argument('--device',choices=['gpu'])
    w=sub.add_parser('publish',help='Publish existing-Run milestone or retain an offline task')
    w.add_argument('--archive-dir',required=True);w.add_argument('--project',default='fg-single3-research')
    w.add_argument('--tracker-dir',default='/data/logdir/fg_single3/experiment_tracking');w.add_argument('--publication')
    w.add_argument('--mode',choices=['offline','online'],default='online');w.add_argument('--proxy')
    l=sub.add_parser('list',help='List archive grades and independent publication status')
    l.add_argument('--archive-root',default=DEFAULT_ROOT)
    return p.parse_args()


def main():
    args=arguments()
    if args.command=='inspect':
        from s3_archive.inspection import inspect
        try:
            result=inspect(args.run_dir,args.checkpoint,args.eval_dir,prepared=args.prepared,family=args.family,
                publication=args.publication,archive_root=args.archive_root)
        except Exception as exc:
            from research_tracking.redaction import safe_error
            result=dict(schema='S3_ARCHIVE_V1',status='INSPECTION_FAILED',ready=False,
                warnings=[],missing=[dict(reason=type(exc).__name__,detail=safe_error(exc))],
                copy_plan=[],estimated_bytes=None,available_bytes=None,proposed_achieved_level=None)
            if args.json_out:
                protected=[args.run_dir,args.eval_dir]
                try:
                    raw=read_json(Path(args.run_dir)/'run_identity.json')
                    protected.extend(raw['identity']['versions'])
                except Exception:
                    pass
                if args.prepared:protected.append(args.prepared)
                independent(args.json_out,protected)
                atomic_json(args.json_out,result)
            print(json.dumps(result,indent=2))
            raise SystemExit(2)
        result['milestone_tags']=args.milestone_tag
        result['plan_sha256']=fingerprint(result)
        if args.json_out:
            independent(args.json_out,[args.run_dir,result['source']['prepared_dir'],args.eval_dir,
                *[r['path'] for r in result['code_audit']['repositories'].values()]])
            atomic_json(args.json_out,result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('copy_plan','identity','code_audit','eval_derived','original_receipt')},indent=2,ensure_ascii=False))
        return
    if args.command=='archive':
        from s3_archive.archive import execute,read_plan
        result=execute(read_plan(args.plan),level=args.level.upper(),preserve=args.preserve_checkpoint,
                       execute=args.execute,reserve_gib=args.reserve_gib)
    elif args.command=='verify':
        from s3_archive.verification import verify
        result=verify(args.archive_dir,mode=args.mode,python=args.python,device=args.device)
    elif args.command=='publish':
        from s3_archive.publication import publish
        result=publish(args.archive_dir,tracker_dir=args.tracker_dir,publication=args.publication,
                       mode=args.mode,proxy=args.proxy,project=args.project)
    else:
        result=[]
        for p in sorted(Path(args.archive_root).glob('*/archive_manifest.json')):
            try:
                m=read_json(p,limit=128*1024*1024)
                result.append(dict(archive_id=m['archive_id'],path=str(p.parent),requested_level=m['requested_level'],
                    achieved_level=m['achieved_level'],archive_state=m['archive_state'],
                    verification_status=m['verification_status'],wandb=m['wandb'],
                    missing_dependencies=m['verification']['missing_dependencies']))
            except Exception as exc:
                result.append(dict(path=str(p.parent),status='UNREADABLE',error=str(exc)))
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        from research_tracking.redaction import safe_error
        print(json.dumps(dict(status='FAILED',error_type=type(exc).__name__,error=safe_error(exc))),file=sys.stderr)
        raise SystemExit(2)
