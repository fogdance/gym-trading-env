"""Verify retained DATA-MULTI artifacts; no sampling, model, or production run."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
GYM = PACKAGE.parents[2]
DREAMER = GYM.parent / 'dreamerv3'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True).stdout


def exact(left, right):
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            exact(a, b)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.tobytes() == right.tobytes()
    else:
        assert left == right


def main():
    from embodied.run.fg_checkpoint import validate_bundle

    identity = read(HERE / 'SOURCE_IDENTITY.json')
    hashes = identity['source_sha256']
    for relative, expected in hashes.items():
        path = GYM / relative
        assert sha(path) == expected, relative
        ast.parse(path.read_text())
        root, local, commit = GYM, relative, identity['gym']['source_commit']
        if relative.startswith('../dreamerv3/'):
            root = DREAMER
            local = relative.removeprefix('../dreamerv3/')
            commit = identity['dreamer']['source_commit']
        assert hashlib.sha256(git(root, 'show', f'{commit}:{local}')).hexdigest() == expected

    for command_name in ('TARGETED_FINAL.command.json', 'SHARED_FINAL.command.json'):
        command = read(HERE / command_name)
        assert command['exit_code'] == 0
        for relative, expected in command['source_sha256'].items():
            assert sha(GYM / relative) == expected, (command_name, relative)

    contracts = read(HERE / 'UNCHANGED_CONTRACTS.json')
    for item in contracts['files']:
        root = GYM if item['repo'] == 'gym-trading-env' else DREAMER
        original = git(root, 'show', f"{item['baseline']}:{item['path']}")
        assert (root / item['path']).read_bytes() == original
    for item in contracts['training_coordinators']:
        current = ast.parse((DREAMER / item['path']).read_text())
        original = ast.parse(git(DREAMER, 'show',
            f"e8c2688bc817662e5e6fa7e8cee62825bcf80b00:{item['path']}").decode())
        def node(tree):
            return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == item['class_name'])
        assert ast.dump(node(current), include_attributes=False) == ast.dump(node(original), include_attributes=False)

    test_summary = read(HERE / 'TEST_RESULTS.json')
    totals = dict(tests=0, failures=0, errors=0, skipped=0)
    for run in test_summary['runs']:
        cases = ET.parse(HERE / run['receipt']).getroot().findall('.//testcase')
        counts = dict(tests=len(cases), failures=sum(c.find('failure') is not None for c in cases),
                      errors=sum(c.find('error') is not None for c in cases),
                      skipped=sum(c.find('skipped') is not None for c in cases))
        assert all(counts[k] == run[k] for k in counts), run['receipt']
        assert len(run['cases']) == len(cases)
        for key, value in counts.items():
            totals[key] += value
    assert totals == dict(tests=171, failures=0, errors=0, skipped=0)

    binding = dict(code_sha256='src/gym_trading_env/research/fg_s3_e_quality_multi.py',
        probe_code_sha256='src/gym_trading_env/research/fg_s3_e_quality_multi_checks.py',
        f3_code_sha256='src/gym_trading_env/research/fg_s3_e_quality_multi_actor.py',
        f3_adapter_sha256='src/gym_trading_env/research/fg_s3_e_quality_actor.py',
        role_composer_sha256='../dreamerv3/embodied/run/fg_s3_e_quality_actor.py')
    bundles = []
    for manifest in sorted(HERE.glob('*_final/*/ckpt/*/manifest.json')):
        meta = validate_bundle(manifest.parent)
        assert meta['backend'] == 'external_immutable_course_reader'
        assert meta['identity']['data_only'] is True
        for key, relative in binding.items():
            assert meta['identity'][key] == hashes[relative]
        assert not list(manifest.parent.rglob('*.npz'))
        bundles.append(str(manifest.relative_to(PACKAGE)))
    assert len(bundles) == 10
    for mode in ('natural', 'auxiliary'):
        root = HERE / f'{mode}_final'
        left = pickle.loads((root / 'continuous/evidence_6.pkl').read_bytes())
        right = pickle.loads((root / 'split/evidence_6.pkl').read_bytes())
        exact({k: v for k, v in left.items() if k != 'checkpoint'},
              {k: v for k, v in right.items() if k != 'checkpoint'})

    links = []
    generated = {HERE / 'DOCUMENT_VALIDATION.json', HERE / 'SHA256SUMS.json'}
    for markdown in PACKAGE.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', markdown.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            path = (markdown.parent / target.split('#')[0]).resolve()
            assert path.exists() or path in generated, (markdown.name, target)
            links.append(f'{markdown.name}:{target}')
    json_files = list(PACKAGE.rglob('*.json'))
    for path in json_files:
        read(path)
    raw_receipt = 'docs/data_multi_20261018/f3/evidence/DEV_SUFFIX_INVALID_OWNER_EXPECTATION.xml'
    full_check = subprocess.run(['git', 'diff', '--cached', '--check'], cwd=GYM,
                                capture_output=True, text=True)
    authored_check = subprocess.run(['git', 'diff', '--cached', '--check', '--', '.',
        f':(exclude){raw_receipt}'], cwd=GYM, capture_output=True, text=True)
    assert authored_check.returncode == 0, authored_check.stdout + authored_check.stderr
    assert full_check.returncode in (0, 2)
    issues = re.findall(r'^(.+?):[0-9]+: trailing whitespace\.$', full_check.stdout, re.MULTILINE)
    assert all(path == raw_receipt for path in issues)
    result = dict(status='PASS_RETAINED_DELIVERY_VALIDATION',
        observed_utc=datetime.now(timezone.utc).isoformat(),
        tested_source_files_verified=len(hashes), unchanged_contract_files=len(contracts['files']),
        unchanged_training_classes=len(contracts['training_coordinators']),
        junit_totals=totals, checkpoint_bundles=bundles,
        cold_state_comparisons=['natural', 'auxiliary'],
        cold_comparison='All fields including ndarray dtype/shape/bytes; only checkpoint output path excluded.',
        json_files_parsed=len(json_files), local_markdown_links=links,
        staged_whitespace=dict(authored_exit_code=authored_check.returncode,
            full_exit_code=full_check.returncode, original_raw_receipt_exception=raw_receipt,
            raw_receipt_sha256=sha(GYM / raw_receipt),
            output=full_check.stdout, reason='Preserve original failing JUnit bytes, including traceback whitespace.'),
        optimizer_updates=0, production='NOT_RUN',
        note='Verification of retained evidence and exact source bytes; tests are not rerun by this command.')
    (HERE / 'DOCUMENT_VALIDATION.json').write_text(json.dumps(result, indent=2) + '\n')
    inventory = {str(path.relative_to(PACKAGE)): sha(path)
        for path in sorted(PACKAGE.rglob('*')) if path.is_file() and path != HERE / 'SHA256SUMS.json'}
    value = dict(protocol='FG_S3_DATA_MULTI_F3_DELIVERY_FILES_V1',
                 scope='Only this f3 directory; excludes this inventory itself.', files=inventory)
    (HERE / 'SHA256SUMS.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], junit=totals, bundles=len(bundles),
                         source_files=len(hashes), inventoried_files=len(inventory))))


if __name__ == '__main__':
    main()
