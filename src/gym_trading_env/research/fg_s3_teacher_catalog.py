"""Authenticated complete P4 view over immutable native TRAIN Replay.

The existing S3-A predicates are the only teacher definition. Natural ownership
is authenticated separately; absence from sparse labels is never an exclusion.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import uuid

import elements
import numpy as np

from .fg_replay import (SINGLE3_SCHEMA, check_files, digest, read_manifest,
                        signed, trace_manifest, write_json)
from .fg_s3a_labels import INDEX_DTYPE, aligned_rows, exclusion, load_labels, sources

PROTOCOL = 'FG_S3_TEACHER_CATALOG_V1'
ROW_DTYPE = np.dtype(INDEX_DTYPE.descr + [
    ('label_valid', '?'), ('exclusion_reason', 'U24'),
    ('is_last', '?'), ('is_terminal', '?'), ('replay_action', '<i4'),
    ('position_state', '<f4', (9,))])


class DatasetIdentityError(ValueError):
    pass


def fail(reason, detail=''):
    raise DatasetIdentityError(reason + (': ' + str(detail) if detail else ''))


def stepid_parts(stepid):
    if isinstance(stepid, np.ndarray):
        if stepid.dtype != np.uint8 or stepid.shape != (20,):
            fail('INVALID_STEPID')
        stepid = stepid.tobytes()
    if not isinstance(stepid, bytes) or len(stepid) != 20:
        fail('INVALID_STEPID')
    return stepid[:16], int.from_bytes(stepid[16:], 'big')


def guard_output(output, inputs):
    root = Path(output).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if root == source or root.is_relative_to(source) or source.is_relative_to(root):
            fail('OUTPUT_OVERLAPS_SOURCE', root)
    if root.exists():
        raise FileExistsError(root)
    return root


def _ownership(source, *, diagnostic):
    root = Path(source['dataset'])
    dataset = read_manifest(root/'manifest.json')
    indices = dataset['pool_members'].get('C1_natural_oracle')
    if indices is None:
        if not diagnostic:
            fail('MISSING_NATURAL_ORACLE_POOL')
        indices = dataset['pool_members']['C1_P4']
    expected = {i for i, r in enumerate(dataset['traces']) if r['candidate'] == 'C1'
                and r['policy_id'] in ('P1', 'P2a', 'P2b', 'P3', 'P4')}
    if len(indices) != len(set(indices)) or set(indices) != expected:
        fail('POOL_MEMBERSHIP_MISMATCH')
    ownership, frozen = {}, {}
    for i in indices:
        rec = dataset['traces'][i]
        directory = (root/rec['directory']).resolve()
        if not directory.is_relative_to(root):
            fail('TRACE_ESCAPED_DATASET')
        check_files({str(directory/'manifest.json'): rec['manifest_sha256'],
                     str(directory/'reconciliation.json'): rec['receipt_sha256']})
        t = trace_manifest(directory, schema=SINGLE3_SCHEMA)
        receipt = read_manifest(directory/'reconciliation.json')
        if (t['policy']['id'] != rec['policy_id'] or t['contract'] != rec['contract']
                or t['reward_candidate'] != 'C1' or t['identity'] != rec['trace_identity']
                or t['reward_params_hash'] != source['reward_params_hash']
                or t['obs_version'] != 'FG_SINGLE3_V1_60x18_9_2'
                or t['observation_identity'] != source['observation_identity']
                or t['observation_identity'] != rec['observation_identity']
                or t['protocol'] != 'fg_single3_v1'
                or t['identity'].get('training_hash' if rec['policy_id']=='P4' else 'original_training_hash') != source['training_hash']
                or t['replay_trace_hash'] != rec['replay_trace_hash']
                or t['records'] != rec['records'] or receipt['status'] != 'PASS'
                or receipt['trace_manifest_sha256'] != rec['manifest_sha256']):
            fail('TRACE_IDENTITY_MISMATCH')
        frozen.update({str(directory/'manifest.json'): rec['manifest_sha256'],
                       str(directory/'reconciliation.json'): rec['receipt_sha256']})
        frozen.update({str(directory/p): h for p, h in t['files'].items()})
        total = 0
        for n, chunk in enumerate(t['chunks']):
            key = bytes(elements.UUID(chunk['uuid'])).hex()
            if key in ownership:
                fail('DUPLICATE_CHUNK_OWNER', key)
            if n and t['chunks'][n-1]['succ'] != chunk['uuid']:
                fail('BROKEN_CHUNK_CHAIN')
            length = int(chunk['length'])
            with np.load(directory/chunk['path'], allow_pickle=False) as data:
                actual = data['stepid']
            expected_ids = np.empty((length, 20), np.uint8)
            expected_ids[:, :16] = np.frombuffer(bytes.fromhex(key), np.uint8)
            expected_ids[:, 16:] = np.arange(length, dtype='>u4').view('u1').reshape(length, 4)
            if actual.dtype != np.uint8 or not np.array_equal(actual, expected_ids):
                fail('NATIVE_STEPID_MISMATCH')
            ownership[key] = dict(owner=rec['policy_id'], length=length,
                                  contract=rec['contract'], trace_id=rec['directory'])
            total += length
        if total != t['records']:
            fail('TRACE_LENGTH_MISMATCH')
    return ownership, frozen


def build_catalog(dataset, oracle_receipt, labels, output, *, diagnostic=False):
    lm = load_labels(labels, verify_sources=True)
    source = sources(dataset, oracle_receipt)
    for key in ('dataset_sha256', 'oracle_sha256', 'training_hash', 'reward_params_hash'):
        if source[key] != lm['source'][key]:
            fail('LABEL_SOURCE_MISMATCH', key)
    if source['training_hash'].startswith('synthetic') and not diagnostic:
        fail('SYNTHETIC_REQUIRES_DIAGNOSTIC')
    source['observation_identity'] = lm['traces'][0]['observation_identity']
    root = guard_output(output, (dataset, oracle_receipt, labels))
    owners, frozen = _ownership(source, diagnostic=diagnostic)
    frozen.update(source['frozen_inputs'])
    frozen[str(Path(labels).resolve()/'manifest.json')] = digest(Path(labels)/'manifest.json')
    frozen.update({str(Path(labels).resolve()/p): h for p, h in lm['files'].items()})
    root.parent.mkdir(parents=True, exist_ok=True)
    pending = root.with_name('.'+root.name+'.pending_'+uuid.uuid4().hex)
    pending.mkdir()
    counts, traces, allids = Counter(), [], set()
    label_traces = {r['contract']: r for r in lm['traces']}
    for rec in source['traces']:
        lr = label_traces.pop(rec['contract'])
        sparse = np.load(Path(labels)/lr['labels_file'], mmap_mode='r', allow_pickle=False)
        by_id = {}
        for x in sparse:
            key = x['stepid'].tobytes()
            if key in by_id:
                fail('DUPLICATE_STEPID', key.hex())
            by_id[key] = x
        directory = Path(source['dataset'])/rec['directory']
        t = trace_manifest(directory, schema=SINGLE3_SCHEMA)
        plan = read_manifest(t['policy']['schedule_path'])
        rows, local = [], Counter()
        for row, step in aligned_rows(directory, t, set(plan['add_cursors'])):
            key = step['stepid'].tobytes()
            if key in allids:
                fail('DUPLICATE_STEPID', key.hex())
            allids.add(key)
            owner = owners.get(key[:16].hex())
            if owner is None or owner['owner'] != 'P4' or owner['trace_id'] != rec['directory']:
                fail('P4_OWNER_MISMATCH')
            why = exclusion(row, step)
            local['total_rows'] += 1
            canonical = (row['ordinal'], row['cursor'], step['stepid'], row['action'],
                         step['action_mask'], row['is_first'], row['decision_valid_minute'],
                         int(row['time'][:4]), {'LONG': 1, 'SHORT': -1, 'NO_TRADE': 0}[row['direction']],
                         int(row['session'] == 'night'))
            if why is None:
                if key not in by_id:
                    fail('ELIGIBLE_LABEL_MISSING', key.hex())
                expected = np.array([canonical], dtype=INDEX_DTYPE)[0]
                saved = by_id.pop(key)
                for name in INDEX_DTYPE.names:
                    if not np.array_equal(expected[name], saved[name]):
                        fail('LABEL_FIELD_MISMATCH', name)
                local['valid'] += 1
                local['ADD' if row['action'] else 'WAIT'] += 1
            else:
                if key in by_id:
                    fail('INVALID_LABEL_PRESENT', key.hex())
                local['excluded_'+why] += 1
            # Invalid label is a safe sentinel, distinct from actual replay_action.
            canonical = (*canonical[:3], row['action'] if why is None else 0, *canonical[4:])
            rows.append((*canonical, why is None, why or '', row['is_last'], row['is_terminal'],
                         row['action'], step['position_state']))
        if by_id:
            fail('FOREIGN_LABEL_STEPID')
        if len(rows) != t['records']:
            fail('P4_LENGTH_MISMATCH')
        filename = rec['contract']+'.rows.npy'
        np.save(pending/filename, np.array(rows, dtype=ROW_DTYPE), allow_pickle=False)
        traces.append(dict(contract=rec['contract'], trace_id=rec['directory'],
                           file=filename, rows=len(rows), counts=dict(local)))
        counts.update(local)
        print('catalog', rec['contract'], dict(local), flush=True)
    if label_traces or counts['valid'] != lm['summary']['counts']['eligible']:
        fail('SPARSE_SET_MISMATCH')
    audit = dict(status='PASS', counts=dict(counts), traces=traces,
                 exact_sparse_fields=list(INDEX_DTYPE.names), unique_stepids=len(allids))
    write_json(pending/'audit.json', audit)
    write_json(pending/'ownership.json', signed(dict(chunks=owners)))
    check_files(frozen)
    manifest = signed(dict(protocol=PROTOCOL, diagnostic=diagnostic, source={
        k: source[k] for k in ('dataset', 'oracle_receipt', 'dataset_sha256', 'oracle_sha256',
                              'training_hash', 'reward_params_hash')},
        labels=str(Path(labels).resolve()), labels_manifest_sha256=digest(Path(labels)/'manifest.json'),
        observation_identity=lm['traces'][0]['observation_identity'],
        obs_version='FG_SINGLE3_V1_60x18_9_2', traces=traces, counts=dict(counts),
        frozen_inputs=frozen, code_sha256=digest(__file__),
        files={p.name: digest(p) for p in pending.iterdir()}))
    write_json(pending/'manifest.json', manifest)
    (pending/'done').write_text('')
    pending.rename(root)
    return manifest


@dataclass(frozen=True)
class TeacherLookup:
    owner: str
    status: str
    record: np.void | None
    contract: str
    trace_id: str


class TeacherCatalog:
    def __init__(self, directory, *, verify_sources=True):
        self.root = Path(directory).resolve()
        self.manifest = m = read_manifest(self.root/'manifest.json')
        if m['protocol'] != PROTOCOL or not (self.root/'done').is_file():
            fail('CATALOG_PROTOCOL_OR_COMPLETION')
        if m['code_sha256'] != digest(__file__):
            fail('CATALOG_CODE_CHANGED')
        if {p.name for p in self.root.iterdir()} != set(m['files']) | {'manifest.json', 'done'}:
            fail('CATALOG_FILE_SET')
        check_files({str(self.root/p): h for p, h in m['files'].items()})
        if verify_sources:
            check_files(m['frozen_inputs'])
        self.manifest_sha256 = digest(self.root/'manifest.json')
        self.owners = read_manifest(self.root/'ownership.json')['chunks']
        self.arrays, self.index = [], {}
        counts = Counter()
        for rec in m['traces']:
            a = np.load(self.root/rec['file'], mmap_mode='r', allow_pickle=False)
            if a.dtype != ROW_DTYPE or a.shape != (rec['rows'],):
                fail('CATALOG_ROW_SCHEMA')
            self.arrays.append(a)
            for n, row in enumerate(a):
                key = row['stepid'].tobytes()
                uid, offset = stepid_parts(key)
                owner = self.require_owner(uid, offset)
                if owner['owner'] != 'P4' or owner['trace_id'] != rec['trace_id'] or key in self.index:
                    fail('DUPLICATE_OR_FOREIGN_CATALOG_ROW')
                valid, why = bool(row['label_valid']), str(row['exclusion_reason'])
                if (valid == bool(why) or row['label'] not in (0, 1)
                        or (not valid and row['label'] != 0)
                        or (valid and (not row['mask'].all() or row['label'] != row['replay_action']))):
                    fail('CATALOG_LABEL_INVARIANT')
                self.index[key] = (len(self.arrays)-1, n)
                counts['total_rows'] += 1
                if valid:
                    counts['valid'] += 1
                    counts['ADD' if row['label'] else 'WAIT'] += 1
                else:
                    counts['excluded_'+why] += 1
        if dict(counts) != m['counts']:
            fail('CATALOG_COUNTS_MISMATCH')
        if len(self.index) != sum(x['length'] for x in self.owners.values() if x['owner'] == 'P4'):
            fail('ELIGIBLE_LABEL_MISSING', 'incomplete P4 view')

    @property
    def valid_count(self):
        return self.manifest['counts']['valid']

    def require_owner(self, uuid16, offset):
        if len(uuid16) != 16:
            fail('INVALID_UUID')
        owner = self.owners.get(bytes(uuid16).hex())
        if owner is None:
            fail('UNKNOWN_SOURCE')
        if not 0 <= offset < owner['length']:
            fail('OFFSET_OUT_OF_RANGE')
        return owner

    def lookup(self, stepid):
        uid, offset = stepid_parts(stepid)
        owner = self.require_owner(uid, offset)
        if owner['owner'] != 'P4':
            return TeacherLookup(owner['owner'], 'NATURAL_NONTEACHER', None,
                                 owner['contract'], owner['trace_id'])
        key = bytes(uid)+offset.to_bytes(4, 'big')
        if key not in self.index:
            fail('ELIGIBLE_LABEL_MISSING')
        shard, n = self.index[key]
        row = self.arrays[shard][n]
        status = ('VALID_ADD' if row['label'] else 'VALID_WAIT') if row['label_valid'] else 'EXCLUDED_'+str(row['exclusion_reason'])
        return TeacherLookup('P4', status, row, owner['contract'], owner['trace_id'])

    def summary(self):
        return dict(protocol=PROTOCOL, manifest_sha256=self.manifest_sha256,
                    diagnostic=self.manifest['diagnostic'], counts=self.manifest['counts'],
                    chunks=len(self.owners))


def load_catalog(path, *, verify_sources=True):
    return TeacherCatalog(path, verify_sources=verify_sources)
