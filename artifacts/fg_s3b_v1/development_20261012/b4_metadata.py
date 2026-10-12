"""Metadata and producer-hash evidence; production agent pickles are NOT loaded."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).parent
DREAMER = Path('/home/v/Documents/work/dreamerv3-dev')
SOURCE = Path('/data/logdir/fg_single3/offline_10k_from_zero/C1_M1_OFFLINE_10K')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

tag = json.loads((SOURCE/'single3_identity.json').read_text())
summary = json.loads((SOURCE/'training_summary.json').read_text())
node = SOURCE/'ckpt/step_000000010000'
m = json.loads((node/'manifest.json').read_text())
files = {str(node/p):sha(node/p) for p in m['files']}
assert all(files[str(node/p)] == h for p,h in m['files'].items())
code = {p:dict(source=h,development=sha(DREAMER/p),match=h == sha(DREAMER/p))
        for p,h in tag['identity']['code_sha256'].items()}
result = dict(status='METADATA_OBSERVED',source=str(node),profile=tag['identity']['profile'],
    identity_hash=tag['identity_hash'],summary_status=summary['status'],last_update=summary['last_update'],
    required_profile='C1_M1_1M',required_update=62500,production_agent_loaded=False,
    source_manifest_sha256=sha(node/'manifest.json'),code_compatibility=code,
    protected_checkpoint_producer_hash_status='PASS',protected_files=files,
    code_mismatches=[p for p,v in code.items() if not v['match']],
    limitation='10K completion is not the required 1M completion; current production core differs from S3-A baseline')
(OUT/'B4_source_metadata.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('code_compatibility','protected_files')},indent=2))
