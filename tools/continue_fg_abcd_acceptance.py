"""Finish smoke and pilot acceptance; deliberately contains no formal launch."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def pilot_command(manifest):
    return [sys.executable, str(REPO / 'tools/run_fg_abcd.py'),
            '--manifest', str(manifest), '--budget', 'pilot', '--execute']


def smoke_alive(pid, logdir):
    if not pid:
        return False
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return False
    return 'run_fg_abcd_smoke.py' in args and str(logdir) in args


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', default='artifacts/fg_abcd_v1/v3_shared_prefix/experiment_manifest.json')
    parser.add_argument('--smoke-logdir', required=True)
    parser.add_argument('--wait-smoke-pid', type=int, default=0)
    parser.add_argument('--execute-pilot', action='store_true')
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    smoke = Path(args.smoke_logdir).resolve()
    output = manifest.parent
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'acceptance.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(pid=os.getpid(), status='WAITING_SMOKE', formal='NOT_RUN_BY_AGENT',
                 smoke_logdir=str(smoke), pilot='NOT_RUN')

    def publish(**updates):
        state.update(updates, updated_at=datetime.datetime.now().astimezone().isoformat())
        temp = output / 'acceptance_status.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(output / 'acceptance_status.json')
        (output / 'acceptance_progress.md').write_text(
            '# ABC acceptance progress\n\n' + '\n'.join(f'- {k}: {v}' for k, v in state.items()) + '\n')
        print(json.dumps(state), flush=True)

    def run(tool, *flags):
        subprocess.run([sys.executable, str(REPO / 'tools' / tool), *map(str, flags)],
                       cwd=REPO, check=True)

    def refresh():
        run('report_fg_abcd.py', '--manifest', manifest)
        run('watch_fg_abcd.py', '--manifest', manifest, '--once')

    publish()
    try:
        while smoke_alive(args.wait_smoke_pid, smoke):
            time.sleep(30)
        run('audit_fg_abcd_smoke.py', '--logdir', smoke, '--output', output / 'smoke_audit.json')
        audit = json.loads((output / 'smoke_audit.json').read_text())
        if audit['status'] != 'PASS':
            raise ValueError('Smoke has not passed')
        publish(status='SMOKE_PASS', smoke=audit)
        if not args.execute_pilot:
            return
        publish(status='PILOT_RUNNING', pilot='RUNNING')
        # This is the only training launch; budget is fixed to pilot.
        process = subprocess.Popen(pilot_command(manifest), cwd=REPO)
        while process.poll() is None:
            refresh()
            time.sleep(30)
        refresh()
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, pilot_command(manifest))
        comparison = json.loads((output / 'pilot_comparison.json').read_text())
        if not comparison['complete'] or comparison['failed']:
            raise ValueError('Pilot acceptance remains incomplete')
        # Budget completion is not evidence of gate or profitable timing.
        publish(status='PILOT_BUDGETS_COMPLETE', pilot=comparison,
                limitation='Check actual_learning and validation per method; no automatic formal launch')
    except BaseException as error:
        publish(status='FAILED', error=f'{type(error).__name__}: {error}',
                limitation='Logs and checkpoints preserved; no automatic restart or rule changes')
        raise


if __name__ == '__main__':
    main()
