"""Read-only runtime adapter from verified E-Data math to E observations.

This module does not authorize model training. It makes the missing M4 consumer
explicit: snapshots are computed at *every* actual Env decision, never pulled
from the sparse 15-minute-stride research export.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import Decision, EDataConfig, MarketData
from .io import sha256, verify_collection, stable_identity
from .sources import pool_entries, read_fg_source
from .schema import identity as fingerprint
from .reference import load_legacy

_VERIFIED = {}


def verified_collection(path):
    """Verify once per process; reuse only while the complete file tree is stable.

    Admission recomputes the same binding several times. The original verifier
    performs the full hashes and mathematical audit on first use. Any inventory,
    inode, size, mtime or ctime change invalidates this process-local result.
    """
    path = Path(path).resolve(strict=True)
    def inventory():
        return tuple((str(p.relative_to(path)), p.is_symlink(), *(
            getattr(p.stat(), k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')))
            for p in sorted(path.rglob('*')))
    before = inventory()
    cached = _VERIFIED.get(path)
    if cached is not None and cached[0] == before:
        return cached[1]
    report = verify_collection(path)
    if report['status'] != 'PASS' or inventory() != before:
        raise ValueError('E collection failed verification or changed during audit')
    _VERIFIED[path] = before, report
    return report


def verify_training_members(prepared_files, original_entries):
    """Prove DAgger's selected pool is a subset of the E-Data TRAIN source.

    Signed manifests may have different bytes/identities because the S3-V2
    prepared pool is a subset; never demand identical manifest SHA instead of
    comparing the immutable underlying market/config files.
    """
    fields = ('contract', 'path', 'sha256', 'env_config', 'env_sha256',
              'contract_config', 'contract_sha256')
    reference = {r['contract']: r for r in original_entries}
    if len(reference) != len(original_entries) or not prepared_files:
        raise ValueError('E source contains duplicate contracts or empty prepared pool')
    used = set()
    for candidate in prepared_files:
        name = candidate['contract']
        if name not in reference or name in used or any(
                candidate.get(k) != reference[name].get(k) for k in fields):
            raise ValueError('Prepared S3-V2 TRAIN pool differs from E source data')
        if candidate.get('role') != 'train' or not candidate.get('eligible_for_training'):
            raise ValueError('E input must be TRAIN eligible')
        used.add(name)
    return tuple(sorted(used))



class ETrainAdmission:
    """Verify signed TRAIN provenance without opening a second market CSV.

    Market OHLCVOI and feature calculation belong to FixedDirectionEntryEnv.
    This is metadata-only; does not grant formal training authorization.
    """

    def __init__(self, *, identity, source_receipt, contracts):
        self.identity = identity
        self.source_receipt = source_receipt
        self.contracts = frozenset(contracts)

    @classmethod
    def from_train_pool(cls, pool_manifest, collection_dir, prepared_manifest, *, contracts=None):
        import json
        collection_dir = Path(collection_dir).resolve(strict=True)
        report = verified_collection(collection_dir)
        if report['status'] != 'PASS':
            raise ValueError('E real TRAIN collection failed verification')
        index = json.loads((collection_dir/'index_manifest.json').read_text())
        original, pool_identity = pool_entries(pool_manifest)
        if pool_identity['sha256'] != index['source_pool_sha256']:
            raise ValueError('E collection/real TRAIN source identity mismatch')
        from . import EDataConfig
        # Schema identity includes implementation-source hashes. Moving the
        # E computation into the Env is a NEW runtime identity: a previously
        # signed collection must not silently authorize that new code.
        expected_schema = EDataConfig().schema(load_legacy()[1])['schema_sha256']
        if index['schema_sha256'] != expected_schema:
            raise ValueError('E implementation identity changed; regenerate and verify a new TRAIN collection')
        # The prepared pool may be a frozen subset. Its market, Env and
        # contract metadata hashes must match the signed original source.
        from gym_trading_env.research.fg_replay import read_manifest
        prepared = read_manifest(prepared_manifest)
        selected = verify_training_members(prepared['files'], original)
        if contracts is not None and contracts != 'ALL_TRAIN' and set(selected) != set(contracts):
            raise ValueError('E prepared contracts differ from declared TRAIN selection')
        if not set(selected).issubset(index['bundles']):
            raise ValueError('E collection lacks a selected TRAIN contract')
        sources = {}
        by_contract = {row['contract']:row for row in original}
        for contract in selected:
            provenance = json.loads((collection_dir/contract/'provenance.json').read_text())
            row = by_contract[contract]
            records = {}
            for role,path_key,hash_key in (('market','path','sha256'),
                    ('env','env_config','env_sha256'),('contract','contract_config','contract_sha256')):
                record = provenance[role]
                if record['path'] != row[path_key] or record['sha256'] != row[hash_key]:
                    raise ValueError('E collection source provenance differs from TRAIN pool')
                records[role] = record
            if provenance.get('master') is not None:
                records['master'] = provenance['master']
            for record in records.values():
                if stable_identity(record['path'])['sha256'] != record['sha256']:
                    raise ValueError('E admitted TRAIN market/config bytes changed')
            sources[contract] = records
        receipt = dict(protocol='FG_S3_E_ENV_OBSERVATION_V1',
            collection_sha256=sha256(collection_dir/'index_manifest.json'),
            collection_identity=index['index_sha256'],
            source_pool_sha256=pool_identity['sha256'],
            schema_sha256=index['schema_sha256'],
            admitted_contracts=list(selected),sources=sources)
        return cls(identity=fingerprint(receipt), source_receipt=receipt,
            contracts=selected)


# Keep the historical research ETrainProvider for read-only comparisons only.
# Formal V2 must use ETrainAdmission and the native Env observation.

class ETrainProvider:
    """Causal real-contract snapshots for current Env decisions, on CPU.

    The schema/collection check is intentionally separate from SourceAdmission:
    even a valid provider is not authorization to unlock an E training run.
    """

    def __init__(self, markets, *, identity, source_receipt):
        if not markets or not isinstance(identity, str) or len(identity) != 64:
            raise ValueError('E provider requires certified nonempty sources and SHA identity')
        if any(contract != market.contract for contract, market in markets.items()):
            raise ValueError('E provider market contract mismatch')
        self.markets = dict(markets)
        self.identity = identity
        self.source_receipt = source_receipt

    @classmethod
    def from_train_pool(cls, pool_manifest, collection_dir, *, contracts=None):
        """Re-open signed original TRAIN bars; never reuse sparse research snapshots.

        Verification of the full collection is deliberately strict (and costly).
        This is done only when admitting a new run, not at each Env step.
        """
        collection_dir = Path(collection_dir).resolve(strict=True)
        report = verified_collection(collection_dir)
        if report['status'] != 'PASS':
            raise ValueError('E real TRAIN collection failed verification')
        import json
        index = json.loads((collection_dir/'index_manifest.json').read_text())
        entries, pool_identity = pool_entries(pool_manifest, contracts or ())
        if pool_identity['sha256'] != index['source_pool_sha256']:
            raise ValueError('E collection and TRAIN pool manifest differ')
        selected = {item['contract'] for item in entries}
        if not selected.issubset(index['bundles']):
            raise ValueError('E collection lacks required TRAIN contracts')
        sources = {s.contract: s for s in (read_fg_source(row, pool_identity) for row in entries)}
        models = {k: MarketData(s.bars, s.schedule, EDataConfig()) for k,s in sources.items()}
        if any(m.config.schema(load_legacy()[1])['schema_sha256']
               != index['schema_sha256'] for m in models.values()):
            raise ValueError('E source mathematical schema differs from accepted collection')
        receipt = dict(protocol='FG_S3_E_REAL_TRAIN_CONSUMER_V1',
            collection_sha256=sha256(collection_dir/'index_manifest.json'),
            collection_identity=index['index_sha256'], source_pool_sha256=pool_identity['sha256'],
            schema_sha256=index['schema_sha256'], sources={k:s.provenance for k,s in sources.items()})
        return cls(models, identity=fingerprint(receipt), source_receipt=receipt)

    def snapshot(self, decision_env, original_obs):
        contract = decision_env.contract
        if contract not in self.markets:
            raise ValueError('E provider refused unknown/currently ineligible real contract')
        if not isinstance(original_obs, dict) or 'market_seq' not in original_obs:
            raise ValueError('E provider needs the actual original 1m observation')
        # The Env has already completed its current observable minute. Never
        # read the next execution bar and never derive a cutoff from sidecar.
        cutoff = pd.Timestamp(decision_env._raw.index[decision_env.current_step])
        source = self.markets[contract]
        decision = Decision(contract, visible_cutoff=cutoff, decision_time=cutoff,
                            market_seq=np.asarray(original_obs['market_seq']))
        context, old, shared, _, _, metadata = source.snapshot(decision)
        if not np.array_equal(old, original_obs['market_seq']):
            raise ValueError('E provider original observation mismatched the Env')
        arrays = {'market_seq_e_shared': shared.copy(), 'market_context_15m': context.copy()}
        host_meta = dict(source_contract=contract, visible_cutoff=metadata['visible_cutoff'],
            decision_time=metadata['decision_time'], data_identity=self.identity,
            context_available_at=metadata['context_available_at'],
            anchor_source=metadata['anchor_source'])
        return arrays, host_meta
