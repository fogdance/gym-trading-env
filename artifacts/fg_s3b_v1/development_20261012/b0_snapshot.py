"""Read-only development baseline; never opens held-out price data or agents."""
import hashlib
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).parent
ROOTS = [Path('/home/v/Documents/work')/p for p in
         ('gym-trading-env2', 'dreamerv3-dev', 'gym-trading-env', 'dreamerv3')]

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

repos, files = {}, {}
for root in ROOTS:
    repos[str(root)] = {key: subprocess.check_output(cmd, cwd=root, text=True).strip()
                       for key, cmd in [('head', ['git', 'rev-parse', 'HEAD']),
                                        ('status', ['git', 'status', '--short', '--branch'])]}
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    for name in names:
        p = root/name
        if p.is_file() and p.suffix in ('.py', '.yaml', '.yml', '.toml', '.sh'):
            files[str(p)] = sha(p)
labels = ROOTS[0]/'artifacts/fg_s3a_v1/labels_20261011_v2/manifest.json'
lm = json.loads(labels.read_text())
for p in [labels, Path(lm['source']['dataset'])/'manifest.json',
          Path(lm['source']['oracle_receipt'])/'manifest.json']:
    files[str(p)] = sha(p)
(OUT/'B0_baseline.json').write_text(json.dumps(dict(repos=repos, protected_files=files,
    scope='Existing source/config and immutable S3-A/Oracle manifest; no GPU or process mutation',
    s3a_complete=True, legal_labels=lm['summary']['counts']['eligible']), indent=2)+'\n')
print(json.dumps(dict(repos=repos, protected_files=len(files), legal_labels=lm['summary']['counts']['eligible'])))
