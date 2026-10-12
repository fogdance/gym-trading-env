"""Independent CPU-only certified value evidence and TRAIN weight packages."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('values-build','values-audit','weights-build','weights-audit'):
        p = sub.add_parser(name)
        p.add_argument('--config')
        p.add_argument('--dataset')
        p.add_argument('--oracle-receipt')
        p.add_argument('--s3a-label-index')
        p.add_argument('--output',required=True)
        p.add_argument('--seed',type=int)
        if name in ('weights-build',):
            p.add_argument('--values',required=True)
        if name == 'values-build':
            p.add_argument('--resume',action='store_true')
            p.add_argument('--contract',action='append')
            p.add_argument('--limit',type=int)
    args = parser.parse_args()
    from gym_trading_env.research.fg_replay import digest
    from gym_trading_env.research.fg_s3b_advantage import audit_values,build_values
    from gym_trading_env.research.fg_s3b_weights import audit_weights,build_weight_index
    cfg = {}
    if args.config:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text())
        if cfg.get('schema') != 'FG_S3B_CPU_CONFIG_V1' or set(cfg)-{
            'schema','dataset','oracle_receipt','s3a_label_index','seed','proof','weights'}:
            raise ValueError('Foreign/unknown S3-B CPU configuration')
    def field(name):
        value = getattr(args,name,None)
        if value is None: value = cfg.get(name)
        if value is None: parser.error('Missing --'+name.replace('_','-'))
        return value
    seed = args.seed if args.seed is not None else cfg.get('seed',202)
    if args.command == 'values-build':
        result = build_values(field('dataset'),field('oracle_receipt'),field('s3a_label_index'),args.output,
            seed=seed,resume=args.resume,contracts=args.contract,limit=args.limit,
            proof_config=cfg.get('proof'),config_sha256=digest(args.config) if args.config else None)
    elif args.command == 'values-audit':
        result = audit_values(args.output)
    elif args.command == 'weights-build':
        # Explicit source options must match the value package, even though the
        # proof package itself already binds and verifies every frozen source.
        value = audit_values(args.values)
        for name in ('dataset','oracle_receipt','s3a_label_index'):
            supplied = getattr(args,name,None) or cfg.get(name)
            key = 's3a_labels' if name == 's3a_label_index' else name
            if supplied and Path(supplied).resolve() != Path(value['identity'][key]).resolve():
                raise ValueError('Requested source differs from value proof: '+name)
        result = build_weight_index(args.values,field('s3a_label_index'),args.output,seed=seed,
            weight_config=cfg.get('weights'),config_sha256=digest(args.config) if args.config else None)
    else:
        result = audit_weights(args.output)
    if 'summary' in result:
        summary = result['summary']
    else:
        counts = Counter()
        for shard in result['shards']: counts.update(shard['summary']['certificates'])
        summary = dict(rows=sum(s['rows'] for s in result['shards']),certificates=dict(counts))
    print(json.dumps(dict(protocol=result['protocol'],status=result['status'],complete=result['complete'],
                         summary=summary),ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
