"""Record CPU test commands, cases, versions and exits in docs; no trainer."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
package = os.environ.get('FG_WP_RECEIPTS','wp03')
if package not in ('wp03','wp04','wp05','wp06','wp07','wp08'):
    raise SystemExit('Unknown WP receipt package')
out = root / 'docs/receipts' / package
out.mkdir(parents=True, exist_ok=True)
name, *tests = sys.argv[1:]
if not tests or not name.replace('_', '').isalnum():
    raise SystemExit('name and pytest paths required')
if (out / (name + '.json')).exists():
    raise SystemExit('Receipt already exists; use a new attempt name')
env = {**os.environ, 'PYTHONPATH': '.:src:../dreamerv3:tests/entry_v1',
       'JAX_PLATFORMS': 'cpu', 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
       'FG_REPLAY_DREAMER_ROOT': str(root.parent / 'dreamerv3'),
       'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
argv = ['/home/v/miniconda3/envs/dreamerv3/bin/python', '-B', '-m', 'pytest',
        '-p', 'no:cacheprovider', '-o', 'addopts=', '-q',
        '--junitxml=' + str(out / (name + '.xml')), *tests]
if tests[0] == '--cmd':
    argv = ['/home/v/miniconda3/envs/dreamerv3/bin/python', '-B', *tests[1:]]
versions = {}
sources = {}
for role, repo in [('gym', root), ('dreamer', root.parent / 'dreamerv3')]:
    git = lambda *a: subprocess.check_output(['git', '-C', str(repo), *a], text=True).strip()
    versions[role] = dict(head=git('rev-parse', 'HEAD'), dirty=git('status', '--porcelain'),
                          branch=git('branch', '--show-current'))
    # Audit only, never a reader/G3 gate. Includes uncommitted new modules.
    folders=([repo/'src', repo/'tools'] if role == 'gym' else [repo/'dreamerv3'])
    if package in ('wp04','wp05','wp06','wp07','wp08'):folders += [repo/'e_data_v1'] if role=='gym' else [repo/'embodied']
    for folder in folders:
        for p in folder.rglob('*.py'):
            sources[role + '/' + str(p.relative_to(repo))] = hashlib.sha256(p.read_bytes()).hexdigest()
began = time.monotonic()
with (out / (name + '.log')).open('x') as stream:
    run = subprocess.run(argv, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
record = dict(argv=argv, cwd=str(root), environment={k:env[k] for k in env if k in
    ('PYTHONPATH', 'JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES', 'PYTHONDONTWRITEBYTECODE',
     'FG_REPLAY_DREAMER_ROOT', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
    repositories=versions, execution_sources=sources, exit_code=run.returncode,
    seconds=time.monotonic()-began)
(out / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k:v for k,v in record.items() if k != 'execution_sources'}))
print((out / (name + '.log')).read_text()[-7000:])
raise SystemExit(run.returncode)
