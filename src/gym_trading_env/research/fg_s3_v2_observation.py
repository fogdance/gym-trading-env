"""Observe one live A2 run; evaluate through its unchanged frozen GPU entry."""
import collections
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .fg_replay import check_files, read_manifest, signed, write_json


NODES = (10000, 50000, 100000, 300000, 1000000)


class JsonTail:
    """Read only complete rows; rebuilding on rollback preserves chronology."""
    def __init__(self, path):
        self.path, self.position, self.inode = Path(path), 0, None

    def rewound(self):
        if not self.path.exists():
            return self.position > 0
        stat = self.path.stat()
        return self.position > stat.st_size or self.inode not in (None, stat.st_ino)

    def rows(self):
        if not self.path.exists():
            return
        with self.path.open('rb') as stream:
            self.inode = self.path.stat().st_ino
            stream.seek(self.position)
            while True:
                line = stream.readline()
                if not line.endswith(b'\n'):
                    return
                self.position = stream.tell()
                yield json.loads(line)


class TrainMetrics:
    def __init__(self):
        self.counts = collections.Counter()

    def observe(self, row):
        if not row['execution_result']['advanced']:
            return
        c = self.counts
        c['env_steps'] += 1
        legal = bool(all(row['action_mask']))
        c['legal_decisions'] += int(legal)
        c['unproven_funds'] += int(row['oracle_reason'] == 'FUNDS_OUTSIDE_ORIGINAL_CERTIFICATE')
        c['actor_add_proposals'] += int(row['actor_proposal'] == 1)
        guided = row['oracle_selected']
        c['actor_controlled_legal'] += int(legal and not guided)
        c['actor_executed_add'] += int(not guided and row['executed_action'] == 1)
        entries = sum(f['reason'] == 'ENTRY' for f in row['execution_result']['fills'])
        c['oracle_guided_entries' if guided else 'actor_entries'] += entries
        if row['teacher_valid']:
            if not (legal and row['oracle_valid'] and row['oracle_certified']
                    and row['teacher_action'] == row['oracle_action']):
                raise ValueError('Observation contains an uncertified supervised decision')
            c['certified_labels'] += 1
            y, p = row['teacher_action'], row['actor_proposal']
            c['teacher_add_labels'] += int(y == 1)
            c[{(1,1):'tp', (1,0):'fn', (0,1):'fp', (0,0):'tn'}[y,p]] += 1

    def result(self):
        c = self.counts
        return dict(**dict(c), label_ratio=c['certified_labels']/max(c['env_steps'],1),
            valid_legal_ratio=c['certified_labels']/c['legal_decisions'] if c['legal_decisions'] else None,
            actor_proposal_oracle_add_recall=c['tp']/(c['tp']+c['fn']) if c['tp']+c['fn'] else None,
            actor_proposal_oracle_add_precision=c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else None,
            metric_scope='Actor proposals on certified TRAIN states; mixed-state distribution',
            guided_trade_success_is_actor_success=False)


def actor_only_metrics(receipt):
    """Derive entry-based outcomes from the existing actual evaluation artifacts."""
    if (receipt['teacher_guidance'] or receipt['beta'] != 0 or receipt['fold'] != 'DEV'
            or receipt.get('tp_target_points',5) != 5):
        raise ValueError('Independent evaluation must be Actor-only DEV')
    output = Path(receipt['output'])
    state = json.loads((output/'account_state.json').read_text())
    dev = read_manifest(output/'dev_manifest.json')
    check_files(dev['frozen_inputs'])
    data = dev['rows'][0]
    actions = [json.loads(s) for s in (output/'actions.jsonl').read_text().splitlines()]
    # The last proposal has not been executed. Proposals before reset/last
    # also have no market advance; reward and actual fills identify outcomes.
    decisions = [a for a,b in zip(actions,actions[1:]) if b['advanced'] and a['mask'][1] and not a['is_last']]
    probabilities = [r['p_add'] for r in decisions]
    tp = [e for e in state['attributions'] if e['reason'] == 'TAKE_PROFIT']
    entries = receipt['entries']
    return dict(checkpoint=receipt['checkpoint'],
        checkpoint_manifest_sha256=receipt['checkpoint_manifest_sha256'],
        evaluation_data_identity=dict(contract=data['contract'],market_sha256=data['sha256'],
            env_config_sha256=data['env_sha256'],contract_sha256=data['contract_sha256'],
            start_cursor=0,reset_seed=0),
        env_steps=receipt['env_steps'], policy_mode=receipt['policy_mode'], beta=0.,
        teacher_guidance=False, autonomous_add_executed=sum(r['action']==1 for r in decisions),
        actual_entries=entries, tp5_points_count=len(tp),
        tp5_points_rate=len(tp)/entries if entries else None,
        tp5_rate_denominator='all actual entries, including failures and right-censored positions',
        right_censored=receipt['right_censored'], sl30=receipt['sl30'],
        sl30_rate=receipt['sl30']/entries if entries else None,
        early_adverse10=receipt['early_mae10'],
        early_adverse10_rate=receipt['early_mae10']/entries if entries else None,
        tp_minutes_p50=receipt['tp_minutes_p50'], tp_minutes_p90=receipt['tp_minutes_p90'],
        tp_effective_bars_p50=receipt['tp_effective_bars_p50'], reward=receipt['reward'],
        all_wait=receipt['all_wait'],
        p_add_quantiles={str(p):float(np.percentile(probabilities,p)) if probabilities else None
                         for p in (0,50,90,99,100)},
        highest_add_scores=sorted(decisions,key=lambda r:r['p_add'],reverse=True)[:20])


class Observer:
    def __init__(self, prepared, run_dir, eval_steps=5000):
        self.prepared, self.root = Path(prepared).resolve(), Path(run_dir).resolve()
        self.identity = read_manifest(self.root/'run_identity.json')['identity']
        self.settings = read_manifest(self.prepared/'settings.json')['settings']
        from .fg_replay import digest
        if digest(self.prepared/'manifest.json') != self.identity['preparation_sha256']:
            raise ValueError('Observation preparation differs from the trained run')
        candidates = [Path(p) for p in self.identity['code']
                      if Path(p).name == 'fg_s3_v2_experiment.py' and Path(p).parent.name == 'tools']
        if len(candidates) != 1 or not 1 <= eval_steps <= 10000:
            raise ValueError('Expected one frozen evaluator and a bounded evaluation')
        self.cli, self.eval_steps = candidates[0], eval_steps
        check_files({str(self.cli):self.identity['code'][str(self.cli)]})
        self.output = self.root/'observations'
        self.output.mkdir(exist_ok=True)
        self.evaluations = {}
        for p in self.output.glob('actor_only_*.json'):
            result = read_manifest(p)
            self.evaluations[result['checkpoint']] = result
        self.reset()

    def reset(self):
        self.teachers = JsonTail(self.root/'rollout/teacher_journal.jsonl')
        self.cycles = JsonTail(self.root/'cycles.jsonl')
        self.updates = JsonTail(self.root/'updates.jsonl')
        self.total, self.window = TrainMetrics(), TrainMetrics()
        self.nodes, self.latest_cycle = {}, None
        self.cycle_nodes, self.update_nodes = {}, {}
        self.learning = collections.Counter()

    def poll(self):
        if any(r.rewound() for r in (self.teachers,self.cycles,self.updates)):
            self.reset()
        for row in self.teachers.rows():
            self.total.observe(row); self.window.observe(row)
            step = self.total.counts['env_steps']
            if row['execution_result']['advanced'] and step in NODES:
                self.nodes[step] = dict(env_steps=step, cumulative=self.total.result(),
                                       interval=self.window.result())
                self.window = TrainMetrics()
        for row in self.updates.rows():
            m = row['metrics']; c = self.learning
            c['wm_updates'] += 1
            c['valid_bc_updates'] += int(row['valid_count'] > 0)
            c['add_bc_updates'] += int(row['add_count'] > 0)
            c['actor_changed_updates'] += int(bool(m['opt/actor_changed']))
            c['wm_changed_updates'] += int(bool(m['opt/wm_changed']))
            step = row['env_steps']
            expected = (step//self.settings['collect_every'])*self.settings['updates_per_collect']
            if step in NODES and row['local_update'] == expected:
                self.update_nodes[step] = dict(c, model_updates=row['agent_update'])
        for row in self.cycles.rows():
            self.latest_cycle = row
            if row['env_steps'] in NODES:
                self.cycle_nodes[row['env_steps']] = row
        ready = []
        for step in sorted(self.nodes.keys() & self.cycle_nodes.keys() & self.update_nodes.keys()):
            node = self.nodes[step]
            node.update(learning=self.update_nodes[step], beta=self.cycle_nodes[step]['beta_schedule']['beta'],
                actor_only_at_this_node=('NO_MODEL_CHECKPOINT_AT_NODE' if step < 100000 else
                                        'SEE_INDEPENDENT_CHECKPOINT_EVALUATIONS'))
            ready.append(node)
        summary = signed(dict(run_dir=str(self.root),run_id=self.identity['run_id'],
            observation_nodes=list(NODES),observations=ready,
            latest_completed_cycle=self.latest_cycle,live_train=self.total.result(),
            actor_only=list(self.evaluations.values()),eval_steps=self.eval_steps,
            training_controlled_by_observer=False,approval_stops=False,
            ten_and_fifty_k_model_snapshots='unavailable in this frozen 100K save run',
            metrics_are_live_not_checkpoint_certified=True))
        write_json(self.output/'summary.json',summary)
        return summary

    def evaluate_latest(self):
        pointer = self.root/'ckpt/latest'
        if not pointer.is_file():
            return
        name = pointer.read_text().strip()
        if Path(name).name != name or not (self.root/'ckpt'/name/'done').is_file():
            return
        node = str(self.root/'ckpt'/name)
        if node in self.evaluations:
            return
        log = self.output/(name+'_actor_only_terminal.log')
        with log.open('a') as stream:
            result = subprocess.run([sys.executable,str(self.cli),'evaluate','--prepared',str(self.prepared),
                '--run-dir',str(self.root),'--eval-steps',str(self.eval_steps)],
                cwd=self.cli.parents[1],stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Actor-only GPU evaluation failed; see '+str(log))
        # The evaluator prints its exact output path; do not guess the model
        # step from the observer trigger while training continues independently.
        text = log.read_text()
        offset = text.rfind('\n{')
        receipt = json.loads(text[offset+1:] if offset >= 0 else text)
        result = actor_only_metrics(receipt)
        if self.evaluations and any(r['evaluation_data_identity'] != result['evaluation_data_identity']
                                   for r in self.evaluations.values()):
            raise ValueError('Independent evaluation data/rules changed across checkpoints')
        result['eval_budget_steps'] = self.eval_steps
        result = signed(result)
        actual_node = result['checkpoint']
        self.evaluations[actual_node] = result
        write_json(self.output/('actor_only_'+Path(actual_node).name+'.json'),result)
        print('Actor-only',Path(actual_node).name,'entries=',result['actual_entries'],
              'TP5=',result['tp5_points_rate'],'ALL_WAIT=',result['all_wait'],flush=True)

    def run(self, watch=False, evaluate=True):
        while True:
            self.poll()
            if evaluate:
                self.evaluate_latest()
                summary = self.poll()
            else:
                summary = self.poll()
            if not watch or (self.latest_cycle and self.latest_cycle['env_steps'] >= 1000000
                    and (not evaluate or any(Path(k).name=='step_000001000000' for k in self.evaluations))):
                return summary
            time.sleep(10)
