"""Protocol-based read-only discovery; filenames do not establish identities."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import os
from pathlib import Path
import pickle
import re
import shutil

import yaml

from research_tracking.identity import SourceRun
from research_tracking.evals import parse_eval
from . import DEFAULT_ROOT, SCHEMA
from .io import ArchiveError, digest, fingerprint, now, prefix_hash, read_json, relative
from .source import audit


class StateReader(pickle.Unpickler):
    """Only metadata containers/NumPy; never import trainer saveable classes."""
    def find_class(self, module, name):
        allowed = {
            'builtins': {'bytes','bytearray','set','frozenset','complex','slice','dict','list','tuple','str','int','float','bool'},
            'collections': {'deque','OrderedDict','defaultdict','Counter'}, 'uuid': {'UUID'},
            'numpy': {'ndarray','dtype'},
            'numpy.core.multiarray': {'_reconstruct','scalar'},
            'numpy._core.multiarray': {'_reconstruct','scalar'},
            'numpy.core.numeric': {'_frombuffer'}, 'numpy._core.numeric': {'_frombuffer'},
        }
        if name in allowed.get(module,set()):
            return getattr(importlib.import_module(module), name)
        raise ArchiveError('Unsupported metadata pickle global: '+module+'.'+name)


def state(path):
    with Path(path).open('rb') as stream:
        return StateReader(stream).load()


def line_boundary(path, count):
    size = 0
    with Path(path).open('rb') as stream:
        for _ in range(count):
            line = stream.readline()
            if not line.endswith(b'\n'):
                raise ArchiveError('Checkpoint-confirmed log prefix is missing')
            size += len(line)
    return size


def inspect(run_dir, checkpoint, eval_dir, *, prepared=None, family=None, publication=None,
            archive_root=DEFAULT_ROOT):
    source = SourceRun(run_dir, prepared, family=family, publication=publication)
    step = int(str(checkpoint).removeprefix('step_'))
    cp = source.root/'ckpt'/f'step_{step:012d}'
    if not (cp/'done').is_file():
        raise ArchiveError('Selected Checkpoint has no done marker')
    meta = read_json(cp/'manifest.json')
    if (meta.get('protocol') != 'FG_FULL_V1' or meta.get('schema') != 1
            or meta.get('step') != step or meta.get('identity') != source.identity):
        raise ArchiveError('Checkpoint protocol, step or Run identity differs')
    cp_sha = digest(cp/'manifest.json')
    evaluation = Path(eval_dir).resolve()
    receipt = read_json(evaluation/'receipt.json', signed=True)
    if Path(receipt['checkpoint']).resolve() != cp or receipt['checkpoint_manifest_sha256'] != cp_sha:
        raise ArchiveError('Selected evaluation and Checkpoint manifest SHA differ')
    derived = parse_eval(source, evaluation/'receipt.json')
    code = audit(source)
    plan, refs, missing, warnings = {}, {}, [], []

    def add(path, target, kind, expected=None, boundary=None):
        path = Path(path)
        if not path.is_file():
            missing.append(dict(path=str(path), kind=kind, reason='MISSING_FILE'))
            return
        size = path.stat().st_size if boundary is None else boundary
        key = str(target)
        candidate = dict(source=str(path), target=key, kind=kind, expected=expected,
                         bytes=size, prefix=boundary is not None)
        if key in plan and plan[key] != candidate:
            raise ArchiveError('Conflicting dependency target')
        plan[key] = candidate

    def reference(path, expected=None, kind='restore'):
        path = Path(path)
        if not path.is_absolute():
            raise ArchiveError('Dependency identity contains a nonabsolute external reference')
        old = refs.get(str(path))
        if old and expected and old[0] and old[0] != expected:
            raise ArchiveError('Conflicting content identities for dependency '+str(path)+': '+str(old[0])+' / '+expected)
        if old and old[1] == 'evaluation':
            kind = 'evaluation'
        refs[str(path)] = (expected or (old[0] if old else None), kind)

    def walk(value, base, kind, depth=0):
        if isinstance(value, list):
            for item in value:
                walk(item, base, kind, depth+1)
        elif isinstance(value, dict):
            if value.get('role') == 'test':
                return
            for field in ('files', 'frozen_inputs', 'frozen_files', 'provenance_files', 'immutable_files'):
                rows = value.get(field)
                if isinstance(rows, dict):
                    for name, sha in rows.items():
                        if isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha):
                            p = Path(name)
                            # Embedded receipts retain paths relative to THEIR own
                            # directory, not the parent receipt containing them.
                            # The parent's top-level index lists their actual paths.
                            if not p.is_absolute() and depth:
                                continue
                            reference(p if p.is_absolute() else relative(base, name), sha, kind)
            for field, checksum in (('path', 'sha256'), ('env_config', 'env_sha256'),
                                    ('contract_config', 'contract_sha256')):
                if isinstance(value.get(field), str) and value.get(checksum):
                    reference(value[field], value[checksum], kind)
            for field, item in value.items():
                if field not in ('code', 'code_files', 'versions', 'dirty'):
                    walk(item, base, kind, depth+1)

    for p in source.prepared.iterdir():
        if p.is_file():
            add(p, 'provenance/prepared/'+p.name, 'evidence')
    for name in ('run_identity.json', 'config.yaml', 'source_admission.json', 'source_transfer.json'):
        add(source.root/name, 'provenance/'+('resolved_config.yaml' if name == 'config.yaml' else name), 'evidence')
    for p in evaluation.iterdir():
        if p.is_file():
            add(p, 'evaluations/'+evaluation.name+'/'+p.name, 'evidence')
    add(cp/'manifest.json', 'checkpoint/'+cp.name+'/manifest.json', 'evidence', cp_sha)
    add(cp/'done', 'checkpoint/'+cp.name+'/done', 'evidence')
    for name, sha in meta['files'].items():
        add(relative(cp, name), 'checkpoint/'+cp.name+'/'+name, 'checkpoint', sha)
    actual = {str(p.relative_to(cp)) for p in cp.rglob('*') if p.is_file()}
    if actual != set(meta['files'])|{'manifest.json', 'done'}:
        raise ArchiveError('Checkpoint member file set differs')
    try:
        collector, counters = state(cp/'collector.pkl'), state(cp/'v2_counters.pkl')
        boundaries = dict(journal_bytes=collector['journal_bytes'], commit_seq=collector['commit_seq'],
                          updates=counters['all_updates'], cycles=counters['cycles'])
        add(source.root/'rollout/teacher_journal.jsonl', 'dependencies/run/rollout/teacher_journal.jsonl',
            'restore', boundary=boundaries['journal_bytes'])
        for name, count in [('updates', boundaries['updates']), ('cycles', boundaries['cycles'])]:
            add(source.root/(name+'.jsonl'), 'dependencies/run/'+name+'.jsonl', 'restore',
                boundary=line_boundary(source.root/(name+'.jsonl'), count))
    except (OSError, KeyError, ArchiveError, ImportError) as exc:
        boundaries = None
        missing.append(dict(kind='restore', reason='CHECKPOINT_BOUNDARY_UNAVAILABLE', detail=str(exc)))
    settings = source.settings
    for field in ('env_receipt', 'oracle_receipt', 'catalog', 'outcome_receipt', 'outcome_dir'):
        if settings.get(field):
            directory = Path(settings[field])
            kind = 'evaluation' if field == 'env_receipt' else 'restore'
            for p in directory.rglob('*'):
                if p.is_file():
                    reference(p, kind=kind)
    reference(settings['dev_source_manifest'], kind='evaluation')
    for p in source.prepared.glob('*.json'):
        walk(read_json(p), source.prepared, 'restore')
    dev = yaml.safe_load((evaluation/'dev_env.yaml').read_text())
    for p in (dev['trading']['data_path'], dev['entry']['contract_config'], dev['entry']['strategy_config']):
        reference(p, kind='evaluation')
    todo, parsed = list(refs), set()
    while todo:
        path = todo.pop()
        if path in parsed:
            continue
        parsed.add(path)
        p = Path(path)
        if p.is_file() and (p.name == 'manifest.json' or p.name == 'inputs.json'):
            before = set(refs)
            walk(read_json(p), p.parent, refs[path][1])
            todo.extend(set(refs)-before)
    for p, (sha, kind) in refs.items():
        add(p, 'dependencies/filesystem/'+p.lstrip('/'), kind, sha)
    rows = list(plan.values())
    def hash_row(row):
        row['sha256'] = prefix_hash(row['source'], row['bytes'])
        if row['expected'] and row['expected'] != row['sha256']:
            raise ArchiveError('Frozen source dependency hash differs: '+row['source'])
        return row
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        rows = list(pool.map(hash_row, rows))
    required_missing = [r for r in missing if r['kind'] in ('evidence', 'checkpoint')]
    if required_missing:
        raise ArchiveError('Required evidence/Checkpoint members are missing')
    anchor = dict(schema=SCHEMA, local_run_id=source.run_id,
        source_identity_sha256=fingerprint(source.identity), checkpoint_manifest_sha256=cp_sha,
        eval_receipt_sha256=digest(evaluation/'receipt.json'))
    key = fingerprint(anchor)
    archive_id = re.sub('[^a-z0-9-]+', '-', source.family.lower())+'-'+str(step)+'-'+key[:12]
    base = Path(archive_root).resolve()
    nearest = base
    while not nearest.exists():
        nearest = nearest.parent
    available = shutil.disk_usage(nearest).free
    if code['missing']:
        warnings.extend(code['missing'])
    return dict(schema=SCHEMA, archive_id=archive_id, identity_key=key, anchor=anchor, created_at_utc=now(),
        ready=code['verified'], warnings=warnings, missing=missing,
        source=dict(original_run_dir=str(source.root), prepared_dir=str(source.prepared),
            eval_dir=str(evaluation), checkpoint_dir=str(cp), checkpoint_steps=step,
            experiment_family=source.family, local_run_id=source.run_id, publication=publication,
            run_identity_sha256=digest(source.root/'run_identity.json'),
            preparation_sha256=source.identity['preparation_sha256'], **{k:v for k,v in anchor.items() if k.endswith('sha256')},
            eval_receipt_content_hash=receipt.get('content_hash')),
        identity=source.identity, clock_protocol=source.clock, code_audit=code, boundaries=boundaries,
        eval_derived=derived, original_receipt=receipt, copy_plan=rows,
        estimated_bytes=sum(r['bytes'] for r in rows), available_bytes=available,
        proposed_achieved_level='EVIDENCE', archive_root=str(base),
        isolation_roots={k:settings.get(k) for k in ('oracle_receipt', 'env_receipt', 'catalog')},
        source_files_written=False, gpu_used=False, network_used=False)
