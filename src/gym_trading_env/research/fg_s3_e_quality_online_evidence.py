"""Native online account-prefix evidence, distinct from six-behavior Corpus G3."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import digest, read_manifest, signed, write_json, E_QUALITY_SCHEMA
from .fg_s3_e_quality_course_preflight import _local_path

PROTOCOL = 'FG_S3_E_QUALITY_ONLINE_ACCOUNT_EVIDENCE_V1'
_EXECUTED = {}


@dataclass(frozen=True)
class AccountEvidence:
    report: dict


def require_account_execution(evidence):
    execution = _EXECUTED.get(id(evidence))
    if (execution is None or execution[0] is not evidence
            or execution[1] != fingerprint(evidence.report)):
        raise ValueError('ONLINE_ACTUAL_ACCOUNT_RECONCILIATION_REQUIRED')
    return copy.deepcopy(evidence.report)


def prefix_rows_digest(rows):
    # The prefix's last decision is pending at its observation boundary. Native
    # Collector legitimately fills these two outcome flags when collection
    # continues. Its action/mask/account/reward and every earlier row stay sealed.
    rows=copy.deepcopy(rows)
    if rows:
        rows[-1]['decision_executed']=False
        rows[-1]['decision_valid_minute']=False
    return fingerprint(rows)


def prefix_identity(collector):
    """Logical rows/account identity; native replay bytes are checked by reconcile."""
    return dict(rows=len(collector.rows), market_steps=collector.counts['env_steps'],
        rows_sha256=prefix_rows_digest(collector.rows),
        account_sha256=fingerprint(collector.env.export_state()))


def verify_account(collector, factory, directory, *, run_identity, source_identity,
                   verifier_identity, market_step, final):
    from .fg_replay import reconcile_trace
    from .fg_single3_evaluation import embodied_runtime
    directory = _local_path(directory)
    with collector.commit_lock:
        prefix = prefix_identity(collector)
        if market_step <= 0 or prefix['market_steps'] != market_step:
            raise ValueError('ONLINE_ACCOUNT_PREFIX_STEP_MISMATCH')
        legal = all(not row['action'] or row['legal_add'] for row in collector.rows)
        teacher_free = not collector.teacher_enabled and collector.policy.beta == 0 and not collector.query_seconds
        if not legal or not teacher_free:
            raise ValueError('ONLINE_ACCOUNT_ILLEGAL_ACTION_OR_TEACHER_ACCESS')
        trace = collector.snapshot_trace(directory, dict(protocol=PROTOCOL,
            scope='ONLINE_ACCOUNT_NATIVE_RECONCILIATION_NOT_CORPUS_G3',run_identity=run_identity,
            verified_through_market_step=market_step,final=bool(final)))
        result = reconcile_trace(directory,Path(__file__).resolve().parents[4]/'dreamerv3',
            env_factory=factory,schema=E_QUALITY_SCHEMA,runtime_factory=embodied_runtime)
        if (result.get('status') != 'PASS' or result['records'] != prefix['rows']
                or result['replay_trace_hash'] != trace['replay_trace_hash']
                or prefix_identity(collector) != prefix):
            raise ValueError('ONLINE_ACCOUNT_PREFIX_CHANGED_DURING_RECONCILIATION')
    receipt = signed(dict(protocol=PROTOCOL, run_identity=run_identity, source_identity=source_identity,
        verifier_identity=verifier_identity, verified_through_market_step=market_step,
        collection_status='COLLECTION_FINAL_VERIFIED' if final else 'COLLECTION_PREFIX_VERIFIED',
        prefix=prefix,trace_identity=trace['replay_trace_hash'],trace_array_sha256=trace['array_hash'],
        trace_event_sha256=trace['event_hash'],legal=True,teacher_free=True,reconciliation=result,
        trace_manifest_sha256=digest(directory/'manifest.json')))
    write_json(directory/'native_account_receipt.json',receipt)
    report = dict(receipt=receipt, receipt_sha256=digest(directory/'native_account_receipt.json'),
        artifact=str(directory), formal_corpus_g3=False)
    evidence = AccountEvidence(report)
    _EXECUTED[id(evidence)] = (evidence,fingerprint(report))
    return evidence


def portable_account_report(evidence, root):
    result = require_account_execution(evidence)
    result['artifact'] = str(_local_path(result['artifact']).relative_to(_local_path(root)))
    return result


def validate_account_report(report, root, run_identity=None):
    """Strict CP owns restored evidence; re-check all external bytes, never a bool."""
    from .fg_replay import trace_manifest
    if (type(report) is not dict or report.get('formal_corpus_g3') is not False
            or type(report.get('artifact')) is not str or Path(report['artifact']).is_absolute()):
        raise ValueError('ONLINE_INVALID_ACCOUNT_EVIDENCE')
    directory = _local_path(_local_path(root)/report['artifact'])
    if not directory.is_relative_to(_local_path(root)):
        raise ValueError('ONLINE_ACCOUNT_EVIDENCE_ESCAPES_RUN')
    receipt = read_manifest(directory/'native_account_receipt.json')
    if (receipt != report['receipt'] or digest(directory/'native_account_receipt.json') != report['receipt_sha256']
            or receipt.get('protocol') != PROTOCOL
            or run_identity is not None and receipt.get('run_identity') != run_identity
            or receipt.get('collection_status') not in ('COLLECTION_PREFIX_VERIFIED','COLLECTION_FINAL_VERIFIED')
            or receipt.get('reconciliation',{}).get('status') != 'PASS'
            or receipt.get('legal') is not True or receipt.get('teacher_free') is not True):
        raise ValueError('ONLINE_ACCOUNT_EVIDENCE_BINDING_CHANGED')
    trace = trace_manifest(directory,schema=E_QUALITY_SCHEMA)
    if (digest(directory/'manifest.json') != receipt['trace_manifest_sha256']
            or trace['replay_trace_hash'] != receipt['trace_identity']
            or trace['array_hash'] != receipt['trace_array_sha256']
            or trace['event_hash'] != receipt['trace_event_sha256']
            or trace['records'] != receipt['prefix']['rows']
            or trace['advanced'] != receipt['verified_through_market_step']
            or trace['identity']['run_identity'] != receipt['run_identity']
            or fingerprint(__import__('json').loads((directory/'account_state.json').read_text())) != receipt['prefix']['account_sha256']):
        raise ValueError('ONLINE_ACCOUNT_TRACE_OR_PREFIX_CHANGED')
    return copy.deepcopy(receipt)


class FinalAccountState:
    protocol = 'FG_S3_E_QUALITY_ONLINE_FINAL_ACCOUNT_STATE_V1'

    def __init__(self, root, run_identity):
        self.root = Path(root)
        self.identity = run_identity
        self.state = dict(protocol=self.protocol,run_identity=run_identity,status='PENDING',
            next_attempt=0,final_evidence=None,error=None)

    def allocate(self, step, final):
        while True:
            number=self.state['next_attempt'];self.state['next_attempt']+=1
            path=self.root/'collection_checks'/f'step_{step:012d}'/f'attempt_{number:012d}'
            if not path.exists():return path

    def save(self):return copy.deepcopy(self.state)

    def validate_snapshot(self, state=None):
        state = self.state if state is None else state
        if (set(state) != set(self.state) or state.get('protocol') != self.protocol
                or state.get('run_identity') != self.identity
                or state.get('status') not in ('PENDING','VERIFYING','VERIFIED','NO_GO')
                or type(state.get('next_attempt')) is not int or state['next_attempt']<0):
            raise ValueError('ONLINE_FINAL_ACCOUNT_STATE_INVALID')
        if state['status']=='VERIFIED':
            receipt=validate_account_report(state['final_evidence'],self.root,self.identity)
            if receipt['collection_status']!='COLLECTION_FINAL_VERIFIED':
                raise ValueError('ONLINE_FINAL_ACCOUNT_REQUIRES_FULL_RECONCILIATION')

    def load(self, state):
        self.validate_snapshot(state);self.state=copy.deepcopy(state)
