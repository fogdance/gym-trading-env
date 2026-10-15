"""Bounded real TRAIN acceptance using the shared S3 lifecycle.

This diagnostic protocol never declares an E composition production-ready. The
normal prepare/start/resume paths retain their Composition readiness checks.
"""
from pathlib import Path
import hashlib
import json
import uuid

import numpy as np

from dreamerv3.configuration import read_config_yaml
from dreamerv3.fg_s3_composition import (CONFIG_SCHEMA, normalize_config,
    validate_composition, validate_model_composition)
from dreamerv3.fg_s3_e_config import resolve_model_init, e_target_config
from dreamerv3.fg_s3_e_diagnostics import EAcceptanceModel
from dreamerv3.fg_s3a_actor import parameter_digests
from embodied.run.fg_s3_source_admission import _inspect_source
from embodied.run.fg_s3_source_transfer import _make_agent, _apply_and_audit_transfer
from embodied.run.fg_s3_v2_train import LiveRun, execution_code, reconcile_cycles
from embodied.run.fg_s3c_train import reconcile_update_log
from embodied.run.fg_checkpoint import host, validate_bundle, file_hash
from .fg_replay import signed, read_manifest, write_json
from .fg_s3_v2_experiment import PROJECT, _validate_settings, _prepare

PROTOCOL = 'FG_S3_E_REAL_TRAIN_ACCEPTANCE_V1'


class _Admission:
    protocol = PROTOCOL
    inspect_source = staticmethod(_inspect_source)

    @staticmethod
    def composition(settings):
        value = validate_composition(settings.get('resolved_composition'))
        validate_model_composition(value, 'e_multiscale_1m15m', settings['mode'])
        if not value['use_e']:
            raise ValueError('E acceptance requires the E encoder')
        return value

    @staticmethod
    def identities(settings):
        return dict(resolved_composition=_Admission.composition(settings))

    @staticmethod
    def records(settings, *documents):
        value = _Admission.composition(settings)
        if any(validate_composition(d.get('resolved_composition')) != value for d in documents):
            raise ValueError('E acceptance Composition records differ')
        return value


def validate_settings(settings):
    _Admission.composition(settings)
    if (settings.get('e_admission') != PROTOCOL or settings['schema'] != CONFIG_SCHEMA
            or settings['scope'] != 'diagnostic' or settings['source_kind'] != 'direct_init'
            or settings.get('source_checkpoint') is not None or settings['batch_size'] != 1
            or settings['model_init'] != dict(seed=0, presets=['size50m'])
            or settings['contracts'] != ['CZCE.FG201', 'CZCE.FG301']):
        raise ValueError('E acceptance requires the frozen real TRAIN diagnostic contract')
    device = settings['device']
    expected = (2200, 1100, 3, 1024, 4096) if device == 'cpu' else (1000, 100, 100, 250, 2048)
    if device not in ('cpu', 'gpu') or tuple(settings[k] for k in
            ('env_steps', 'collect_every', 'updates_per_collect', 'env_slice_bars', 'replay_capacity')) != expected:
        raise ValueError('E acceptance forbids changing its bounded CPU/GPU budget or network')
    if device == 'gpu':
        receipt = read_manifest(settings['e_cpu_evidence'])
        if (receipt.get('protocol') != PROTOCOL or receipt.get('status') != 'PASS_CPU_REAL_TRAIN'
                or receipt['mode'] != settings['mode'] or not receipt['cold_resume_exact']
                or receipt['code'] != execution_code()):
            raise ValueError('GPU acceptance requires matching completed real CPU acceptance')
        meta = validate_bundle(receipt['checkpoint'], receipt['identity'])
        if file_hash(Path(receipt['checkpoint'])/'manifest.json') != receipt['checkpoint_manifest_sha256']:
            raise ValueError('CPU acceptance checkpoint changed')
        if meta['identity']['e_binding']['data_manifest'] != settings['e_data_manifest']:
            raise ValueError('CPU/GPU acceptance dataset differs')


def read_settings(config):
    path = Path(config)
    if not path.is_file():
        path = PROJECT/'configs/experiments/fg_s3_v2'/f'{path}.yaml'
    path = path.resolve(strict=True)
    settings = _validate_settings(normalize_config(read_config_yaml(path), executable=False),
                                  path, admission=True)
    validate_settings(settings)
    return settings


def acceptance_model_config(settings):
    model = resolve_model_init(settings['model_init'])
    if settings['device'] == 'cpu':
        # Explicit small CPU architecture; never a substitute for size50m GPU.
        from .fg_s3_v2_gpu_check import small_config
        small = e_target_config(small_config())
        # Reduce only the CPU network, retaining the canonical Single3 Gate.
        # The generic GPU unit fixture intentionally disables ADD recall;
        # real TRAIN acceptance must not inherit that diagnostic relaxation.
        small['action_mask_warmup'] = model['action_mask_warmup']
        model = small
    return model


def prepare(config):
    settings = read_settings(config)
    return _prepare(settings, admission=_Admission, model_config=acceptance_model_config(settings))


def _batch_hash(batch):
    h = hashlib.sha256()
    for key, value in sorted(host(batch).items()):
        x = np.asarray(value)
        h.update(key.encode()+str(x.dtype).encode()+str(x.shape).encode()+x.tobytes())
    return h.hexdigest()


class _EAdmissionRun(LiveRun):
    protocol = PROTOCOL
    composition = staticmethod(_Admission.composition)
    identities = staticmethod(_Admission.identities)
    records = staticmethod(_Admission.records)
    create_agent = staticmethod(_make_agent)
    transfer_agent = staticmethod(_apply_and_audit_transfer)
    agent_class = EAcceptanceModel

    def validate_execution(self, settings, preparation):
        validate_settings(settings)
        if preparation['protocol'] != PROTOCOL:
            raise ValueError('Production prepared snapshots cannot enter E acceptance')

    def probe(self):
        """One real next sample/update, then restore the complete original bundle."""
        node = self.cp.latest()
        row = self.update()
        value = dict(checkpoint=str(node), checkpoint_manifest_sha256=file_hash(node/'manifest.json'),
            next_batch_sha256=_batch_hash(self.last_batch),
            next_parameters=parameter_digests(self.agent.params), metrics=row['metrics'],
            agent_counters=self.agent.save_exact()['counters'])
        self.cp.load(node)
        reconcile_update_log(self.root, self.counters.state['all_updates'], self.offset)
        reconcile_cycles(self.root, self.counters.state['cycles'], int(self.step), int(self.agent.n_updates))
        return value

    def finish(self, receipt, cold_resume_exact):
        if receipt['env_steps'] != self.settings['env_steps']:
            probe = self.probe()
            write_json(self.root/'resume_probe.json', signed(probe))
            return receipt
        if not cold_resume_exact:
            raise ValueError('Acceptance requires a separate start/resume process comparison')
        rows = [json.loads(line) for line in self.collector.journal.read_text().splitlines()]
        if len(rows) < 65 or any('e_snapshot' not in row for row in rows):
            raise ValueError('Missing per-minute E DAgger evidence')
        for row in rows:
            if row['e_snapshot']['data_identity'] != self.collector.identity['e_data_identity']:
                raise ValueError('E provider identity drifted within the run')
            if row['execution_result']['advanced'] and row['execution_result']['cursor'] != row['cursor']+1:
                raise ValueError('A real TRAIN advance skipped a minute')
        for a, b in zip(rows, rows[1:]):
            if a['next_state_id'] != b['state_id']:
                raise ValueError('DAgger next-state/account continuity differs')
        if self.collector.counts['rejects']:
            raise ValueError('Illegal executed action in E acceptance')
        contracts = sorted({r['contract'] for r in rows})
        if contracts != self.settings['contracts'] or self.collector.counts['resets'] < 2:
            raise ValueError('Real TRAIN rotation/reset was not exercised')
        updates = [json.loads(line) for line in (self.root/'updates.jsonl').read_text().splitlines()]
        branch = {name: max(r['metrics']['opt/e_'+name+'_grad_norm'] for r in updates)
                  for name in ('minute', 'context15', 'fusion')}
        if not all(np.isfinite(v) and v > 0 for v in branch.values()):
            raise ValueError('An E encoder branch had no finite nonzero gradient')
        transfer = read_manifest(self.root/'source_transfer.json')
        before, after = transfer['params_after'], parameter_digests(self.agent.params)
        prefixes = ['enc/minute/', 'enc/context15/', 'enc/fusion/', 'pol/']
        if self.settings['mode'] == 'b2':
            prefixes += ['b2_event/', 'b2_early/', 'b2_adapter/']
        changed = {p: sum(before[k] != after[k] for k in before if k.startswith(p)) for p in prefixes}
        if not all(changed.values()):
            raise ValueError('An admitted Encoder/Actor/Outcome/Adapter branch did not update')
        self.agent.sync_policy_from_train()
        policy = parameter_digests(self.agent.save_exact()['policy_params'])
        if any(v != after[k] for k, v in policy.items()):
            raise ValueError('Inference policy is not synchronized')
        node = Path(receipt['checkpoint'])
        validate_bundle(node, self.identity)
        status = 'PASS_CPU_REAL_TRAIN' if self.settings['device'] == 'cpu' else 'PASS_GPU_SIZE50M_1K'
        result = signed(dict(protocol=PROTOCOL, status=status, mode=self.settings['mode'],
            identity=self.identity, code=execution_code(), cold_resume_exact=True,
            checkpoint=str(node), checkpoint_manifest_sha256=file_hash(node/'manifest.json'),
            env_steps=int(self.step), model_updates=int(self.agent.n_updates),
            contracts=contracts, chunk_end_rows=sum(r['execution_result']['status']=='CHUNK_END' for r in rows),
            e_rows=len(rows), resets=self.collector.counts['resets'],
            original_market_seq_exact=True, every_minute_provider=True,
            branch_gradients=branch, parameter_changes=changed,
            policy_digests=policy, counters=self.counters.save(), counts=dict(self.collector.counts),
            optimizer_parameters=after, data_binding=self.spec['e_binding'],
            training_receipt_sha256=file_hash(self.root/'training_receipt.json'),
            formal_1m='NOT_RUN', test='SEALED_NOT_READ'))
        write_json(self.root/'acceptance.json', result)
        return result


def run(prepared, run_dir, *, resume=False, stop_at=None):
    runner = _EAdmissionRun(prepared, run_dir, resume=resume)
    try:
        exact = False
        if resume:
            expected = read_manifest(runner.root/'resume_probe.json')
            actual = runner.probe()
            expected.pop('content_hash')
            exact = expected == actual
            write_json(runner.root/'cold_resume_comparison.json', signed(dict(
                exact=exact, expected=expected, actual=actual)))
            if not exact:
                raise ValueError('E cold resume next batch/model/optimizer/RNG/metrics differs')
        receipt = runner.run_to(stop_at)
        return runner.finish(receipt, exact)
    except BaseException as error:
        write_json(runner.root/('failure_'+uuid.uuid4().hex+'.json'), signed(dict(
            error_type=type(error).__name__, error=str(error), env_steps=int(runner.step),
            updates=runner.counters.save())))
        raise
    finally:
        runner.close()
