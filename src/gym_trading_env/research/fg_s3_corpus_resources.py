"""CPU leases for Corpus accounts, their teachers, and phase preparation.

The broker belongs to one production controller. Idle coordinators keep a
memory reservation, but no CPU lease. Mathematical/Replay code is unchanged.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
import secrets
import socket
import socketserver
import threading
import time

POLICIES = ('P0', 'P1', 'P2_low', 'P2_mid', 'P3', 'P4')
DEFAULT_CPUS = dict(zip(POLICIES, (8, 2, 2, 2, 2, 8)))
ENV = 'FG_CORPUS_BROKER'


def policy_cpus(value):
    if isinstance(value, str):
        try:
            pairs = [part.split('=') for part in value.split(',')]
            value = dict((key, int(count)) for key, count in pairs)
            if len(pairs) != len(value):
                raise ValueError('duplicate policy')
        except (ValueError, TypeError) as exc:
            raise ValueError('policy-cpus requires P0=8,P1=2,P2_low=2,P2_mid=2,P3=2,P4=8') from exc
    if (not isinstance(value, dict) or set(value) != set(POLICIES)
            or any(type(n) is not int or n < 2 for n in value.values())):
        raise ValueError('Each of the six policy CPU budgets must be an integer >= 2 (account + teacher)')
    return {key: value[key] for key in POLICIES}


def allocate(value, capacity):
    """Keep two processes per policy; apportion extra demand by largest remainder."""
    value = policy_cpus(value)
    if type(capacity) is not int or capacity < 12:
        raise ValueError('Insufficient CPU/memory capacity: six policies require at least 12 worker processes')
    if capacity >= sum(value.values()):
        return value
    extra = capacity - 12
    demand = sum(n - 2 for n in value.values())
    base = {key: 2 + extra * (n - 2) // demand for key, n in value.items()}
    order = sorted(POLICIES, key=lambda key: (-((extra * (value[key] - 2)) % demand), POLICIES.index(key)))
    for key in order[:capacity - sum(base.values())]:
        base[key] += 1
    return base


def available_memory():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) * 1024


def process_identity(pid):
    try:
        # comm can contain spaces and parentheses.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), pgid=int(fields[2]), start_ticks=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def group_pss(groups):
    total = 0
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            identity = process_identity(int(path.name))
            if identity and identity['pgid'] in groups:
                total += int(next(line.split()[1] for line in (path/'smaps_rollup').read_text().splitlines()
                                  if line.startswith('Pss:'))) * 1024
        except (OSError, StopIteration):
            continue
    return total


def largest_group_process(groups):
    largest=0
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            identity=process_identity(int(path.name))
            if identity and identity['pgid'] in groups:
                pss=int(next(line.split()[1] for line in (path/'smaps_rollup').read_text().splitlines()
                             if line.startswith('Pss:')))*1024
                largest=max(largest,pss)
        except (OSError,StopIteration):continue
    return largest


class Ledger:
    def __init__(self, cpus, worker_bytes, headroom_bytes, parent_bytes, *, memory=available_memory,
                 resident=group_pss, event=None):
        self.cpus, self.worker_bytes = cpus, worker_bytes
        self.headroom_bytes, self.parent_bytes = headroom_bytes, parent_bytes
        self.memory, self.resident, self.event = memory, resident, event
        self.condition = threading.Condition()
        self.leases, self.parents, self.pending = {}, {}, {}
        self.stopping = False
        self.failure = None

    def register(self, pid):
        with self.condition:
            self.parents[pid] = process_identity(pid)

    def used(self):
        return sum(item['cpus'] for item in self.leases.values())

    def fits(self, cpus):
        groups = {item['pgid'] for item in self.parents.values() if item}
        if self.resident is group_pss:
            # Retain 25% headroom over the largest observed initialized/grown
            # process; never reduce a reservation during a production session.
            self.worker_bytes=max(self.worker_bytes,(5*largest_group_process(groups)+3)//4)
        reserved = self.used() * self.worker_bytes + len(self.parents) * self.parent_bytes
        nonresident = max(0, reserved - self.resident(groups))
        return (self.used() + cpus <= self.cpus
                and self.memory() - nonresident - cpus * self.worker_bytes >= self.headroom_bytes)

    def acquire(self, cpus, name, priority=0):
        if type(cpus) is not int or not 1 <= cpus <= self.cpus:
            raise ValueError('CPU lease exceeds global capacity')
        token = secrets.token_hex(12)
        with self.condition:
            self.pending[token] = dict(cpus=cpus, name=name, priority=priority, since=time.monotonic())
            try:
                # Let simultaneously ready strategy requests enter the priority queue.
                self.condition.wait(.02)
                while not self.stopping:
                    fitting = [(key, value) for key, value in self.pending.items() if self.fits(value['cpus'])]
                    first = min(fitting, key=lambda item: (item[1]['priority'], item[1]['since']), default=None)
                    if first and first[0] == token:
                        self.leases[token] = self.pending.pop(token)
                        self._event('acquire', token)
                        self.condition.notify_all()
                        return token
                    self.condition.wait(.25)
                raise RuntimeError('Corpus controller stopped admissions')
            finally:
                self.pending.pop(token, None)

    def release(self, token, cpus=None):
        with self.condition:
            if token in self.leases:
                if cpus is None or cpus >= self.leases[token]['cpus']:
                    del self.leases[token]
                else:
                    self.leases[token]['cpus'] -= cpus
                self._event('release', token)
            self.condition.notify_all()

    def unregister(self, pid):
        with self.condition:
            self.parents.pop(pid, None)
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()

    def abort(self, reason):
        with self.condition:
            if self.failure is None:self.failure=str(reason)
            self.stop()

    def _event(self, kind, token):
        if self.event:
            self.event(dict(event=kind, token=token, cpu_used=self.used(), cpu_capacity=self.cpus,
                            leases=list(self.leases.values()),worker_memory_bytes=self.worker_bytes,time=time.time()))


class Broker:
    def __init__(self, ledger):
        secret = secrets.token_hex(32)
        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    request = json.loads(self.rfile.readline(65536))
                    if request.pop('secret', None) != secret:
                        raise ValueError('Invalid Corpus broker credential')
                    method = request.pop('method')
                    if method not in ('acquire', 'release', 'abort'):
                        raise ValueError('Unknown broker operation')
                    result = getattr(ledger, method)(**request)
                    response = dict(result=result)
                except Exception as exc:
                    response = dict(error=str(exc))
                try:
                    self.wfile.write((json.dumps(response)+'\n').encode())
                except BrokenPipeError:
                    ledger.stop()
        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
        self.server = Server(('127.0.0.1', 0), Handler)
        self.setting = json.dumps(dict(port=self.server.server_address[1], secret=secret))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def rpc(method, **kwargs):
    setting = json.loads(os.environ[ENV])
    with socket.create_connection(('127.0.0.1', setting['port'])) as sock:
        sock.sendall((json.dumps(dict(secret=setting['secret'], method=method, **kwargs))+'\n').encode())
        with sock.makefile('r') as stream:
            response = json.loads(stream.readline())
    if 'error' in response:
        raise RuntimeError(response['error'])
    return response['result']


_phase = None


def pause_phase():
    global _phase
    if _phase:
        rpc('release', token=_phase)
        _phase = None


def resume_phase():
    global _phase
    if ENV in os.environ and _phase is None:
        _phase = rpc('acquire', cpus=1, name='phase-preparation-or-finalization',
                     priority=int(os.environ.get('FG_CORPUS_PRIORITY', '0')))


@contextmanager
def phase_lease():
    global _phase
    _phase = os.environ.pop('FG_CORPUS_PREP_LEASE', None)
    resume_phase()
    try:
        yield
    finally:
        pause_phase()


def run_policy_jobs(function, jobs, budgets):
    """Each account owns a fresh process and teacher pool, then releases its lease."""
    def run(job):
        name = job.get('name', job.get('spec', {}).get('id'))
        count = budgets[name]
        token = None
        if ENV in os.environ:
            priority = int(os.environ.get('FG_CORPUS_PRIORITY', '0')) + (0 if name in ('P0', 'P4') else 1)
            token = rpc('acquire', cpus=count, name=name, priority=priority)
        try:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as executor:
                return executor.submit(function, {**job, 'label_workers': count-1}).result()
        finally:
            if token:
                rpc('release', token=token)
    # Standalone weighted execution uses the same admission rule for CPU slots.
    if ENV not in os.environ:
        remaining = [sum(budgets.values())]
        condition = threading.Condition()
        original = run
        def run(job):
            name = job.get('name', job.get('spec', {}).get('id'))
            with condition:
                condition.wait_for(lambda: remaining[0] >= budgets[name])
                remaining[0] -= budgets[name]
            try:
                return original(job)
            finally:
                with condition:
                    remaining[0] += budgets[name]
                    condition.notify_all()
    pause_phase()
    try:
        ordered = sorted(jobs, key=lambda job: (job.get('name', job.get('spec', {}).get('id')) not in ('P0', 'P4')))
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [executor.submit(run, job) for job in ordered]
            try:
                return [future.result() for future in as_completed(futures)]
            except BaseException as exc:
                if ENV in os.environ:
                    rpc('abort',reason=f'Corpus strategy failed: {exc}')
                raise
    finally:
        resume_phase()


@contextmanager
def legacy_verification_lease(workers):
    """Only legacy teachers can be released on structured completion, once each.

    Legacy executor account processes remain alive until executor shutdown;
    their reservations are deliberately retained until the subprocess exits.
    """
    pause_phase()
    token = rpc('acquire', cpus=workers, name='legacy-native-g3', priority=-10)
    limits, completed = None, set()
    def progress(line):
        nonlocal limits
        try:
            value = json.loads(line)
        except ValueError:
            return
        if value.get('phase') == 'verify' and limits is None:
            candidate = value['resource_budget']
            if (type(candidate.get('workers')) is not int or not 1<=candidate['workers']<=6
                    or len(candidate.get('label_workers',[]))!=candidate['workers']
                    or any(type(n) is not int or n<0 for n in candidate['label_workers'])
                    or candidate['total_worker_processes']!=candidate['workers']+sum(candidate['label_workers'])
                    or not 1 <= candidate['total_worker_processes'] <= workers):
                raise ValueError('Legacy resource budget differs')
            limits = candidate
            rpc('release', token=token, cpus=workers-limits['total_worker_processes'])
        name = value.get('behavior')
        # When outer workers < 6, slots can be reused by queued accounts.
        if limits and limits['workers'] == 6 and name in POLICIES and name not in completed and 'receipt' in value:
            completed.add(name)
            teachers = limits['label_workers'][POLICIES.index(name)]
            if teachers:
                rpc('release', token=token, cpus=teachers)
    try:
        yield progress
    finally:
        rpc('release', token=token)
        resume_phase()
