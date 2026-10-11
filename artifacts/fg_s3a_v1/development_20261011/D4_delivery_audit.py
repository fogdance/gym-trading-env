"""Read-only delivery audit; only this audit receipt is written in dev Gym."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path('/home/v/Documents/work/gym-trading-env2')
DREAMER = Path('/home/v/Documents/work/dreamerv3-dev')
RECEIPTS = ROOT / 'artifacts/fg_s3a_v1/development_20261011'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def main():
    initial = json.loads((RECEIPTS / 'D0_initial.json').read_text())
    changed = {p: dict(initial=h, current=sha(p) if Path(p).is_file() else None)
               for p, h in initial['protected_files'].items()
               if not Path(p).is_file() or sha(p) != h}
    repos = {}
    for root, base in [(ROOT, '79aa9a9664a648fc961277986290874a6eb1c9d1'),
                       (DREAMER, '446e6fc9dbc92a1be24d0d44b840e85d4f96010e')]:
        diff = git(root, 'diff', '--name-status', base)
        assert all(line.split('\t')[0] == 'A' for line in diff.splitlines())
        # Preserve raw pytest failure text, including its traceback whitespace.
        git(root, 'diff', '--check', base, '--', '.', ':(exclude)*.xml')
        repos[str(root)] = dict(baseline=base, head=git(root, 'rev-parse', 'HEAD'),
                               branch=git(root, 'branch', '--show-current'), diff=diff)
    git(DREAMER, 'apply', '--check', 'docs/FG_S3A_pending_integration.patch')
    source = Path('/data/logdir/fg_single3/gate_margin_v2/C1_M1_1M')
    tag = json.loads((source / 'single3_identity.json').read_text())
    matches = {name: sha(DREAMER / name) == h
               for name, h in tag['identity']['code_sha256'].items()}
    assert all(matches.values())
    tests = {}
    for name in ('D1_identity_pytest.xml', 'D2_D4_pytest.xml'):
        suites = ET.parse(RECEIPTS / name).getroot().findall('testsuite')
        tests[name] = {key: sum(int(s.attrib[key]) for s in suites)
                       for key in ('tests', 'failures', 'errors', 'skipped')}
        assert tests[name]['failures'] == tests[name]['errors'] == tests[name]['skipped'] == 0
    proof = json.loads((RECEIPTS / 'D4_real_resume_and_policy.json').read_text())
    assert proof['status'] == 'PASS' and proof['frozen_keys'] == 203
    assert all(proof[k] for k in ('all_model_parameters_byte_equal',
               'independent_optimizer_byte_equal', 'sampler_byte_equal', 'all_update_metrics_equal'))
    result = dict(time=datetime.now(timezone.utc).isoformat(),
        s3a_development_status='CPU_IMPLEMENTATION_AND_REVIEW_COMPLETE',
        production_file_protection_status='DRIFT_OBSERVED' if changed else 'UNCHANGED',
        protected_file_count=len(initial['protected_files']),
        unchanged_protected_count=len(initial['protected_files']) - len(changed),
        changed_production_files_outside_s3a_dev_diff=changed,
        repos=repos, all_dev_changes_additive=True,
        diff_checks='PASS_EXCLUDING_RAW_PYTEST_XML_TRACEBACK_WHITESPACE',
        pending_core_patch='NOT_APPLIED_CHECK_ONLY_PASS',
        old_source_identity_hash=tag['identity_hash'], old_source_core_matches=matches,
        final_source_checkpoint_exists=(source / 'ckpt/step_000000062500/done').is_file(),
        old_source_training_summary_exists=(source / 'training_summary.json').is_file(),
        tests=tests, real_cpu_proof_status=proof['status'],
        production_payload_loaded=False, formal_s3a_gpu_training=False,
        c1_signal_kill_restart_performed=False,
        latest_user_scope='Finish S3-A development only; C1 retraining handled separately',
        files={p.name: sha(p) for p in sorted(RECEIPTS.iterdir())
               if p.is_file() and p.name != 'D4_delivery_audit.json'})
    path = RECEIPTS / 'D4_delivery_audit.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('s3a_development_status',
          'production_file_protection_status', 'protected_file_count',
          'unchanged_protected_count', 'final_source_checkpoint_exists', 'tests')}, indent=2))


if __name__ == '__main__':
    main()
