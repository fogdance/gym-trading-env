"""Run an explicit original command with an independent tracking thread."""
import subprocess
import sys
import threading

from .io import TrackingError
from .redaction import safe_error


def launch(command, *, cwd, environment, track):
    if not command:
        raise TrackingError('launch requires an explicit original command after --')
    # The trainer starts before any tracking/network initialization. Its
    # environment is captured before this CLI applies its tracker-only proxy.
    producer = subprocess.Popen(command, cwd=cwd, env=environment)
    stop = threading.Event()
    result = {}

    def observer():
        try:
            result['tracking'] = track(stop)
        except Exception as exc:
            result['tracking_error'] = safe_error(exc)
            print('Tracker failed; original command continues: '+safe_error(exc), file=sys.stderr, flush=True)

    thread = threading.Thread(target=observer, name='fg-research-tracker')
    thread.start()
    try:
        code = producer.wait()
    finally:
        stop.set()
        # Drain the final completed Checkpoint/evaluation once. SDK timeouts
        # remain bounded; an unavailable cloud never interrupts the producer.
        thread.join()
    result.update(status='ORIGINAL_COMMAND_FINISHED', producer_returncode=code,
                  trainer_environment_modified=False)
    return result
