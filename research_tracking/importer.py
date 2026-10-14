"""Transactional source-prefix staging and independent provisional monitoring."""
import copy

from .evals import parse_eval
from .io import TrackingError, digest
from .metrics import aggregate_updates, cycle_metrics, provisional_updates
from .parsers import JsonLines, watermark


class Importer:
    def __init__(self, store, every=40):
        if type(every) is not int or every < 1:
            raise TrackingError('Aggregation interval must be a positive integer')
        self.store = store
        self.source = store.source
        self.every = every

    def job_config(self, job):
        return dict(parent_training_run_id=self.source.ids['train'], local_run_id=self.source.run_id,
            family=self.source.family, clock_protocol=self.source.clock,
            job_type=job, source='historical_import' if job != 'live_monitor' else 'read_only_follow')

    def checkpoints(self):
        current = self.source.checkpoints()
        indices = self.store.state['checkpoints']
        attachments = []
        present = {c['checkpoint_manifest_sha256'] for c in current}
        for c in current:
            sha = c['checkpoint_manifest_sha256']
            if indices.get(sha) != c:
                indices[sha] = c
                attachments.append(self.store.artifact('checkpoint_'+sha+'.json', c))
        for sha, c in list(indices.items()):
            if sha not in present and c['local_artifact_available']:
                c = dict(c, local_artifact_available=False)
                indices[sha] = c
                attachments.append(self.store.artifact('checkpoint_'+sha+'_unavailable.json', c))
        return current, attachments

    def canonical(self):
        before = copy.deepcopy(self.store.state)
        try:
            checkpoints, attachments = self.checkpoints()
            mark = watermark(self.source, checkpoints)
            state = self.store.state['canonical']
            if mark['local_update'] < state['local_update'] or mark['env_steps'] < state['cycle_env_steps']:
                raise TrackingError('Completed Checkpoint watermark regressed below published canonical history')
            for name in ('updates', 'cycles'):
                saved = JsonLines(self.source.root/(name+'.jsonl'), state[name])
                if saved.changed_prefix():
                    raise TrackingError('Previously confirmed log boundary changed')
                if saved.cursor.get('inode') is not None and saved.path.exists() and saved.path.stat().st_ino != saved.cursor['inode']:
                    saved.verify_prefix()
            if mark['local_update'] == state['local_update']:
                if attachments:
                    self.store.queue('train', self.source.ids['train'], [dict(kind='checkpoint_index_update',
                        metrics={'checkpoint/latest_env_steps': mark['env_steps']})],
                        config=self.job_config('train'), artifacts=attachments)
                self.store.save()
                return mark
            updates = JsonLines(self.source.root/'updates.jsonl', state['updates'])
            cycles = JsonLines(self.source.root/'cycles.jsonl', state['cycles'])
            updates.verify_prefix()
            cycles.verify_prefix()
            last_update = state['local_update']
            last_env = state['cycle_env_steps']

            def update_events():
                nonlocal last_update
                block = []
                for row, cursor in updates.rows():
                    if row['local_update'] > mark['local_update']:
                        break
                    if (row['local_update'] != last_update + 1
                            or row['agent_update'] != self.source.identity['source_update_offset'] + row['local_update']
                            or row['env_steps'] > mark['env_steps']):
                        raise TrackingError('Committed update prefix has a gap or incompatible counters')
                    last_update = row['local_update']
                    updates.accept(cursor)
                    block.append(row)
                    if last_update % self.every == 0:
                        yield dict(kind='canonical_update_window', metrics=aggregate_updates(block))
                        block = []
                if block:
                    yield dict(kind='canonical_update_window', metrics=aggregate_updates(block))
                if last_update != mark['local_update']:
                    raise TrackingError('Checkpointed updates are missing; canonical batch not published')

            def cycle_events():
                nonlocal last_env
                for row, cursor in cycles.rows():
                    if row['env_steps'] > mark['env_steps']:
                        break
                    if row['env_steps'] <= last_env or row['agent_updates'] > mark['agent_update']:
                        raise TrackingError('Committed cycle prefix chronology differs')
                    row['interval_env_steps'] = row['env_steps'] - last_env
                    last_env = row['env_steps']
                    cycles.accept(cursor)
                    yield dict(kind='canonical_cycle', metrics=cycle_metrics(row))
                if last_env != mark['env_steps']:
                    raise TrackingError('Checkpointed cycles are missing; canonical batch not published')

            def ordered():
                iterators = [iter(update_events()), iter(cycle_events())]
                heads = [next(it, None) for it in iterators]
                while any(h is not None for h in heads):
                    choices = [i for i, h in enumerate(heads) if h is not None]
                    i = min(choices, key=lambda j: (heads[j]['metrics']['progress/env_steps'], j))
                    event = heads[i]
                    yield event
                    heads[i] = next(iterators[i], None)

            attachments.insert(0, 'config_preview.json')
            self.store.queue('train', self.source.ids['train'], ordered(),
                config=self.job_config('train'), artifacts=attachments)
            # Source prefix must still match the completed node at promotion.
            if digest(next(c['local_path'] for c in checkpoints
                           if c['env_steps'] == mark['env_steps']) + '/manifest.json') != mark['checkpoint_manifest_sha256']:
                raise TrackingError('Checkpoint metadata changed during staging')
            state.update(local_update=last_update, cycle_env_steps=last_env,
                updates=updates.saved_cursor(hash_prefix=True), cycles=cycles.saved_cursor(hash_prefix=True),
                checkpoint_manifest_sha256=mark['checkpoint_manifest_sha256'])
            self.store.save()
            return mark
        except BaseException:
            self.store.state = before
            # Unreferenced staged files are retained, but never exported or synced.
            raise

    def evaluations(self):
        count = 0
        for path in sorted((self.source.root/'evaluations').glob('*/receipt.json')):
            key = digest(path)
            if key in self.store.state['evaluations']:
                continue
            report = parse_eval(self.source, path)
            artifact = self.store.artifact('eval_'+report['eval_run_id']+'.json', report)
            config = dict(self.job_config('eval'), checkpoint_manifest_sha256=report['checkpoint_manifest_sha256'],
                receipt_sha256=report['receipt_sha256'], eval_report_sha256=report['report_sha256'],
                teacher_guidance=False, beta=0., contract=report['contract'], fold='DEV',
                local_artifact_available=report['local_artifact_available'])
            self.store.queue('eval', report['eval_run_id'], [dict(kind='actor_only_eval', metrics=report['metrics'])],
                config=config, artifacts=('config_preview.json', artifact))
            self.store.state['evaluations'][key] = dict(run_id=report['eval_run_id'],
                checkpoint_manifest_sha256=report['checkpoint_manifest_sha256'], artifact=artifact)
            self.store.save()
            count += 1
        return count

    def monitor(self, mark):
        state = self.store.state['monitor']
        canonical = self.store.state['canonical']
        if not state['initialized']:
            state.update(initialized=True, updates=copy.deepcopy(canonical['updates']),
                cycles=copy.deepcopy(canonical['cycles']), local_update=canonical['local_update'],
                cycle_env_steps=canonical['cycle_env_steps'])
        readers = [JsonLines(self.source.root/(name+'.jsonl'), state[name]) for name in ('updates', 'cycles')]
        replaced = any(reader.cursor.get('inode') is not None and reader.path.exists()
                       and reader.path.stat().st_ino != reader.cursor['inode'] for reader in readers)
        reset = replaced or any(r.changed_prefix() for r in readers)
        if reset:
            state.update(segment=state['segment']+1, updates=copy.deepcopy(canonical['updates']),
                cycles=copy.deepcopy(canonical['cycles']), local_update=canonical['local_update'],
                cycle_env_steps=canonical['cycle_env_steps'])
            readers = [JsonLines(self.source.root/(name+'.jsonl'), state[name]) for name in ('updates', 'cycles')]
            for reader in readers:
                reader.verify_prefix()
        updates, cycles = readers
        segment = state['segment']
        last_update = state['local_update']
        last_env = state['cycle_env_steps']

        def events():
            nonlocal last_update, last_env
            if reset:
                yield dict(kind='monitor_rollback', metrics={'provisional/segment': segment,
                    'provisional/is_provisional': True, 'provisional/recovery_event': 1,
                    'provisional/latest_checkpoint_env_steps': mark['env_steps']})
            block = []
            for row, cursor in updates.rows():
                if row['local_update'] != last_update + 1:
                    raise TrackingError('Provisional update chronology conflicts; refusing ambiguous monitor history')
                last_update = row['local_update']
                updates.accept(cursor)
                if row['local_update'] <= mark['local_update']:
                    continue
                block.append(row)
                if len(block) == self.every:
                    yield dict(kind='provisional_update_window', metrics=provisional_updates(block, segment, mark['env_steps']))
                    block = []
            if block:
                yield dict(kind='provisional_update_window', metrics=provisional_updates(block, segment, mark['env_steps']))
            for row, cursor in cycles.rows():
                if row['env_steps'] <= last_env:
                    raise TrackingError('Provisional cycle chronology conflicts')
                row['interval_env_steps'] = row['env_steps'] - last_env
                last_env = row['env_steps']
                cycles.accept(cursor)
                if row['env_steps'] > mark['env_steps']:
                    yield dict(kind='provisional_cycle', metrics=cycle_metrics(row, provisional=True,
                        segment=segment, checkpoint_env_steps=mark['env_steps']))

        before = copy.deepcopy(self.store.state)
        try:
            self.store.queue('live_monitor', self.source.ids['live_monitor'], events(),
                             config=self.job_config('live_monitor'))
            state.update(local_update=last_update, cycle_env_steps=last_env,
                updates=updates.saved_cursor(), cycles=cycles.saved_cursor())
            self.store.save()
        except BaseException:
            self.store.state = before
            raise
