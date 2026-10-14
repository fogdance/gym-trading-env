"""CPU fixtures protect the tracker contract; actual input/SDK tests are labelled."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

from research_tracking.evals import parse_eval
from research_tracking.auth import proxy_environment
from research_tracking.identity import SourceRun
from research_tracking.importer import Importer
from research_tracking.io import TrackingError, atomic_json, digest, fingerprint
from research_tracking.metrics import aggregate_updates, cycle_metrics
from research_tracking.launcher import launch
from research_tracking.parsers import JsonLines, watermark
from research_tracking.redaction import redact
from research_tracking.state import Store
from research_tracking.writer import MockWriter, Writer


def signed(body):
    return dict(body, content_hash=fingerprint(body))


class Fixture:
    """Synthetic metadata, not a real model or acceptance of training behavior."""
    def __init__(self, root):
        self.root = root/'source'
        self.prepared = root/'prepared'
        self.root.mkdir(); self.prepared.mkdir()
        self.offset = 7
        model = {'agent': {'rssm': {'deter': 64}, 'avail_threshold': .3}}
        extension = {'s3_v2_mode': 'tutor', 'tutor_alpha': .8}
        (self.root/'config.yaml').write_text(yaml.safe_dump(model))
        settings = dict(run_dir=str(self.root), mode='tutor', scope='diagnostic',
            beta=1., clock_protocol='S3_MARKET_MINUTE', api_key='FAKE_SENSITIVE_VALUE_NEVER_UPLOAD')
        atomic_json(self.prepared/'settings.json', signed({'settings': settings}))
        atomic_json(self.prepared/'model_config.json', signed({'config': model, 'agent_extension': extension}))
        (self.prepared/'experiment.yaml').write_text(yaml.safe_dump(settings))
        files = {p.name:digest(p) for p in self.prepared.iterdir()}
        self.identity = dict(protocol='FG_S3_V2_LIVE_LEARNING_V1', run_id='0123456789abcdef'*2,
            mode='tutor', scope='diagnostic', source_update_offset=self.offset,
            oracle_identity='1'*64, source_model_identity='2'*64,
            training_identity={'reward_params_hash':'3'*64}, code={'protected.py':'4'*64},
            versions={str(root/'gym-trading-env'):'a'*40, str(root/'dreamerv3'):'b'*40},
            model_config_sha256=fingerprint(model), agent_extension=extension,
            immutable_files={str(self.root/'config.yaml'):digest(self.root/'config.yaml')})
        preparation = signed(dict(code=self.identity['code'], versions=self.identity['versions'],
            dirty={k:[] for k in self.identity['versions']}, files=files,
            oracle_identity=self.identity['oracle_identity'], training_identity=self.identity['training_identity'],
            clock_protocol='S3_MARKET_MINUTE'))
        atomic_json(self.prepared/'manifest.json', preparation)
        self.identity['preparation_sha256'] = digest(self.prepared/'manifest.json')
        atomic_json(self.root/'run_identity.json', signed({'identity':self.identity}))
        self.rows = [self.update(i, 100) for i in range(1,5)]
        self.cycles = [self.cycle(100,4)]
        self.write_logs()
        self.checkpoint(0)
        self.checkpoint(100)

    def update(self, update, env):
        return dict(local_update=update, agent_update=self.offset+update, env_steps=env,
            valid_count=10, add_count=2, sampled_stepids=[[999]*20], oracle_add_recall=.5,
            metrics={'bc/nll':.5, 'bc/valid_count':10., 'bc/tp':1., 'bc/fn':1., 'bc/fp':0., 'bc/tn':8.,
                     'tutor/pair_count':4., 'tutor/pair_accuracy':.75, 'tutor/eligible_pos':2.,
                     'tutor/p_add_pos_mean':.6, 'opt/loss':2., 'private_token':123.})

    def cycle(self, env, updates):
        return dict(env_steps=env, agent_updates=self.offset+updates,
            counts={'env_steps':env,'certified_labels':env//2}, gate={'actor_enabled':0.},
            beta_schedule={'beta':1.}, collect_seconds=1., update_seconds=1., replay_items=100)

    def write_logs(self, *, replace=False):
        for name, rows in [('updates',self.rows),('cycles',self.cycles)]:
            path = self.root/(name+'.jsonl')
            if replace:
                path = path.with_suffix('.replacement')
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            if replace:
                os.replace(path, self.root/(name+'.jsonl'))

    def checkpoint(self, step, *, done=True):
        path = self.root/'ckpt'/f'step_{step:012d}'
        path.mkdir(parents=True,exist_ok=True)
        (path/'agent.pkl').write_bytes(b'SENTINEL: NEVER READ THIS FAKE MODEL')
        atomic_json(path/'manifest.json', {'protocol':'FG_FULL_V1','schema':1,'step':step,
            'identity':self.identity,'files':{'agent.pkl':digest(path/'agent.pkl')}})
        if done:(path/'done').touch()
        return path

    def source(self):
        return SourceRun(self.root,self.prepared)

    def evaluation(self):
        p=self.root/'evaluations/actor_only_fixture';p.mkdir(parents=True)
        cp=self.root/'ckpt/step_000000000100'
        r=dict(status='ACTOR_ONLY_DEV_DIAGNOSTIC_COMPLETE',fold='DEV',contract='CZCE.FG409',
            checkpoint=str(cp),checkpoint_manifest_sha256=digest(cp/'manifest.json'),
            teacher_guidance=False,oracle_files_read=False,beta=0.,policy_mode='eval_argmax',
            env_steps=5000,entries=1,actor_add_proposed=1,add_executed=1,tp_count=1,
            tp_target_points=5,sl30=0,early_mae10=0,reward=.9,clock_protocol='S3_MARKET_MINUTE')
        atomic_json(p/'receipt.json',signed(r))
        atomic_json(p/'account_state.json',dict(contract='CZCE.FG409',total_reward=.9,
            attributions=[dict(entry_id='PRIVATE_ACCOUNT_ID',reason='TAKE_PROFIT',held_market_bars=4,
                net_trade_pnl='282.0',gross_trade_pnl='300.0')], reward_events=[]))
        return p/'receipt.json'


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.fixture=Fixture(self.base)
    def tearDown(self):self.temp.cleanup()

    def test_provenance_without_model_reads_and_wrong_frozen_config(self):
        original=Path.open
        def guarded(path,*args,**kwargs):
            if path.name=='agent.pkl' or 'replay' in path.parts:raise AssertionError('model/replay read')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guarded):
            s=self.fixture.source();self.assertEqual(watermark(s)['local_update'],4)
            self.assertEqual(s.configuration()['provenance']['versions']['gym-trading-env'],'a'*40)
        self.assertNotIn('jax',sys.modules);self.assertNotIn('gym_trading_env',sys.modules)
        with self.assertRaises(TrackingError):Store(s,s.root/'tracking')
        independent=self.base/'symlink_tracking';independent.mkdir()
        (independent/s.ids['train']).symlink_to(s.root,target_is_directory=True)
        with self.assertRaises(TrackingError):Store(s,independent)
        (self.fixture.root/'config.yaml').write_text('agent: changed\n')
        with self.assertRaises(TrackingError):self.fixture.source()

    def test_partial_line_committed_boundary_and_repeat_import(self):
        f=self.fixture
        f.rows.extend(f.update(i,200) for i in range(5,9));f.cycles.append(f.cycle(200,8));f.write_logs()
        f.checkpoint(200,done=False)
        with (f.root/'updates.jsonl').open('ab') as stream:stream.write(b'{"local_update":9')
        with Store(f.source(),self.base/'tracking') as store:
            importer=Importer(store,2);mark=importer.canonical();importer.monitor(mark)
            mock=MockWriter();mock.export(store)
            canonical=mock.events[f.source().ids['train']]
            self.assertTrue(all(e['metrics'].get('progress/env_steps',0)<=100 for e in canonical))
            live=mock.events[f.source().ids['live_monitor']]
            self.assertTrue(all(k.startswith('provisional/') for e in live for k in e['metrics']))
            self.assertEqual(store.state['monitor']['local_update'],8)
            before=store.summary()['queued_events'];importer.canonical();importer.monitor(mark)
            self.assertEqual(store.summary()['queued_events'],before)
            text=''.join((store.root/b['file']).read_text() for b in store.state['batches'])
            self.assertNotIn('sampled_stepids',text);self.assertNotIn('private_token',text)

    def test_actual_trainer_style_tail_rollback_keeps_canonical_and_segments_monitor(self):
        f=self.fixture
        f.rows.extend(f.update(i,200) for i in range(5,9));f.cycles.append(f.cycle(200,8));f.write_logs()
        with Store(f.source(),self.base/'tracking') as store:
            im=Importer(store,2);mark=im.canonical();im.monitor(mark)
            canonical_count=sum(b['count'] for b in store.state['batches'] if b['job_type']=='train')
        f.rows=f.rows[:6];f.rows[4]['metrics']['bc/nll']=.7;f.cycles=f.cycles[:1];f.write_logs(replace=True)
        with Store(f.source(),self.base/'tracking') as store:
            im=Importer(store,2);mark=im.canonical();im.monitor(mark)
            self.assertEqual(store.state['monitor']['segment'],1)
            self.assertEqual(store.state['monitor']['local_update'],6)
            self.assertEqual(sum(b['count'] for b in store.state['batches'] if b['job_type']=='train'),canonical_count)
            self.assertEqual(store.state['canonical']['local_update'],4)

    def test_failed_gap_staging_never_publishes_and_confirmed_mutation_rejected(self):
        f=self.fixture;f.rows[2]['local_update']=99;f.write_logs()
        with Store(f.source(),self.base/'tracking') as store:
            with self.assertRaises(TrackingError):Importer(store,2).canonical()
            self.assertEqual(store.state['batches'],[])
        f.rows=[f.update(i,100) for i in range(1,5)];f.write_logs()
        with Store(f.source(),self.base/'tracking') as store:Importer(store,2).canonical()
        f.rows[0]['metrics']['bc/nll']=.123;f.write_logs(replace=True)
        with Store(f.source(),self.base/'tracking') as store:
            with self.assertRaises(TrackingError):Importer(store,2).canonical()

    def test_eval_identity_idempotency_and_small_allowlist(self):
        p=self.fixture.evaluation()
        r=json.loads(p.read_text());r.pop('content_hash');r['all_wait']=False
        r['b2']={'legal_argmax_changed':3,'p_add_final_mean':.2,'api_key':'NEVER_UPLOAD'}
        atomic_json(p,signed(r))
        with Store(self.fixture.source(),self.base/'tracking') as store:
            im=Importer(store);self.assertEqual(im.evaluations(),1);self.assertEqual(im.evaluations(),0)
            batch=store.state['batches'][0];report=json.loads((store.root/batch['artifacts'][1]).read_text())
            self.assertEqual(report['metrics']['eval/closed_trade_net_pnl'],282.)
            self.assertEqual(report['metrics']['eval/entries'],1)
            self.assertEqual(report['metrics']['eval/b2/legal_argmax_changed'],3)
            self.assertFalse(report['metrics']['eval/all_wait'])
            self.assertNotIn('PRIVATE_ACCOUNT_ID',json.dumps(report))
            self.assertNotIn('FAKE_SENSITIVE_VALUE_NEVER_UPLOAD',(store.root/'config_preview.json').read_text())
        r=json.loads(p.read_text());r['teacher_guidance']=True;r=signed({k:v for k,v in r.items() if k!='content_hash'})
        atomic_json(p,r)
        with self.assertRaises(TrackingError):parse_eval(self.fixture.source(),p)

    def test_weighted_metrics_undefined_recall_counters_and_no_missing_zero(self):
        a=self.fixture.update(1,100);b=self.fixture.update(2,100)
        a['metrics']['b2/mix_stage']=0;b['metrics']['b2/mix_stage']=1
        a['metrics']['tutor/difficulty_weight_pos_max']=1.2
        b['metrics']['tutor/difficulty_weight_pos_max']=1.8
        a['metrics']['tutor/pair_accuracy']=1.;a['metrics']['tutor/pair_count']=1.
        b['metrics']['tutor/pair_accuracy']=0.;b['metrics']['tutor/pair_count']=3.
        a['tutor_pair_update_count']=5;b['tutor_pair_update_count']=6
        out=aggregate_updates([a,b]);self.assertEqual(out['train/tutor/pair_accuracy'],.25)
        self.assertEqual(out['train/b2/mix_stage'],1)
        self.assertEqual(out['train/tutor/difficulty_weight_pos_max'],1.8)
        self.assertEqual(out['train/tutor/pair_update_count'],6)
        a['metrics'].update({'b2/count_FAST':1.,'b2/recall_FAST':1.,'b2/event_valid_count':3.,'opt/updates':1.})
        b['metrics'].update({'b2/count_FAST':3.,'b2/recall_FAST':0.,'b2/event_valid_count':5.,'opt/updates':2.})
        out=aggregate_updates([a,b]);self.assertEqual(out['train/b2/count_FAST'],4.)
        self.assertEqual(out['train/b2/recall_FAST'],.25)
        self.assertEqual(out['train/b2/event_valid_count'],8.)
        self.assertEqual(out['raw_train/opt/updates'],2.)
        self.assertIsInstance(out['progress/local_update'],int)
        for row in (a,b):row['oracle_add_recall']=None;row['metrics']['missing']=None;row['metrics']['bad']=float('nan')
        out=aggregate_updates([a,b]);self.assertNotIn('train/oracle_add_recall',out)
        self.assertNotIn('raw_train/missing',out);self.assertNotIn('raw_train/bad',out)
        cycle=self.fixture.cycle(100,4);del cycle['counts']['certified_labels']
        self.assertNotIn('train/teacher_coverage',cycle_metrics(cycle))

    def test_redaction_preserves_provenance_hashes_and_hides_paths_and_secrets(self):
        secret='Z'*50
        out=redact({'API_KEY':secret,'value':'Bearer '+secret,'note':'/home/v/private/run',
                    'source_model_identity':'a'*64,'gym-trading-env':'b'*40,
                    'sampled_stepids':[[1,2]],'private_key':'-----BEGIN PRIVATE KEY-----abc-----END PRIVATE KEY-----'})
        self.assertNotIn(secret,json.dumps(out));self.assertNotIn('/home/v',json.dumps(out))
        self.assertEqual(out['source_model_identity'],'a'*64);self.assertEqual(out['gym-trading-env'],'b'*40)
        self.assertNotIn('sampled_stepids',out)

    def test_source_lock_and_checkpoint_retention_availability(self):
        f=self.fixture;s=f.source()
        with Store(s,self.base/'tracking') as store:
            Importer(store).canonical()
            with self.assertRaises(TrackingError):
                with Store(s,self.base/'tracking'):pass
        f.rows.extend(f.update(i,200) for i in range(5,9));f.cycles.append(f.cycle(200,8));f.write_logs();f.checkpoint(200)
        shutil.move(str(f.root/'ckpt/step_000000000100'),str(self.base/'retained_elsewhere'))
        with Store(s,self.base/'tracking') as store:
            Importer(store).canonical()
            missing=[c for c in store.state['checkpoints'].values() if c['env_steps']==100]
            self.assertFalse(missing[0]['local_artifact_available'])

    def test_disabled_and_transport_failure_preserve_queued_history(self):
        with Store(self.fixture.source(),self.base/'tracking') as store:
            Importer(store).canonical();before=store.summary()['queued_events']
            self.assertEqual(Writer('disabled').export(store),[])
            self.assertEqual(store.summary()['exported_events'],0)
            with patch('research_tracking.writer.subprocess.run',side_effect=TimeoutError('network failure')):
                with self.assertRaises(TrackingError):Writer('offline').export(store)
            self.assertEqual(store.summary()['queued_events'],before)
            self.assertEqual(store.summary()['exported_events'],0)
            self.assertEqual(json.loads((store.root/'last_transport_error.json').read_text())['phase'],'export')

    def test_offline_sync_retries_after_reopen_without_changing_source(self):
        source = self.fixture.source()
        files = [p for root in (self.fixture.root, self.fixture.prepared)
                 for p in root.rglob('*') if p.is_file() and p.suffix != '.pkl']
        before = {str(p): digest(p) for p in files}
        tracking = self.base/'tracking'
        with Store(source, tracking) as store:
            Importer(store).canonical()
            with patch.object(Writer, '_transport', return_value={'status':'OFFLINE'}):
                Writer('offline').export(store)
            ids = [b['run_id'] for b in store.state['batches']]
        with patch('research_tracking.writer.readiness', return_value={
                'status':'PRIVATE_PROJECT_AUTHENTICATED'}):
            with Store(source, tracking) as store:
                with patch.object(Writer, '_transport', side_effect=TrackingError('mock outage')):
                    with self.assertRaisesRegex(TrackingError, 'mock outage'):
                        Writer('online', entity='mock-entity').sync(store)
                self.assertTrue(all(not b['synced'] for b in store.state['batches']))
            with Store(source, tracking) as store:
                def transport(store, batch, mode):
                    self.assertEqual(mode, 'online')
                    return dict(status='ONLINE_VERIFIED', sent_events=batch['count'],
                                skipped_remote_events=0)
                with patch.object(Writer, '_transport', side_effect=transport) as mocked:
                    writer = Writer('online', entity='mock-entity')
                    self.assertEqual(len(writer.sync(store)), len(ids))
                    self.assertEqual(writer.sync(store), [])
                    self.assertEqual(mocked.call_count, len(ids))
                self.assertEqual([b['run_id'] for b in store.state['batches']], ids)
                self.assertTrue(all(b['synced'] for b in store.state['batches']))
            with Store(source, tracking) as store:
                self.assertTrue(all(b['synced'] for b in store.state['batches']))
        self.assertEqual(before, {str(p): digest(p) for p in files})

    def test_tracker_proxy_does_not_mutate_trainer_environment(self):
        original={'NO_PROXY':'internal.example','WANDB_MODE':'online'}
        routed=proxy_environment('http://127.0.0.1:10808',original)
        self.assertNotIn('HTTPS_PROXY',original)
        self.assertEqual(routed['HTTPS_PROXY'],'http://127.0.0.1:10808')
        self.assertIn('127.0.0.1',routed['NO_PROXY'])
        default=proxy_environment(environment={'NO_PROXY':'*','HTTPS_PROXY':'http://unused:1'})
        self.assertEqual(default['HTTPS_PROXY'],'http://127.0.0.1:10808')
        self.assertNotIn('*',default['NO_PROXY'])
        self.assertEqual(default['FG_S3_WANDB_PROXY'],'http://127.0.0.1:10808')
        self.assertNotIn('internal.example',routed['NO_PROXY'])
        with self.assertRaises(TrackingError):proxy_environment('file:///tmp/proxy',original)

    def test_explicit_republication_preserves_source_and_has_stable_new_ids(self):
        original=self.fixture.source()
        replacement=SourceRun(self.fixture.root,self.fixture.prepared,publication='reupload-v1')
        repeat=SourceRun(self.fixture.root,self.fixture.prepared,publication='reupload-v1')
        self.assertEqual(original.identity,replacement.identity)
        self.assertNotEqual(original.ids,replacement.ids)
        self.assertEqual(replacement.ids,repeat.ids)
        receipt=self.fixture.evaluation()
        self.assertNotEqual(parse_eval(original,receipt)['eval_run_id'],parse_eval(replacement,receipt)['eval_run_id'])


class ActualInputTests(unittest.TestCase):
    @unittest.skipUnless(Path('/data/logdir/fg_single3/s3_v2/TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M_v1').exists(),
                         'actual local Tutor source unavailable')
    def test_actual_tutor_800k_read_only_receipt(self):
        s=SourceRun('/data/logdir/fg_single3/s3_v2/TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M_v1')
        p=s.root/'evaluations/actor_only_16be8b0985/receipt.json'
        before={str(x):digest(x) for x in p.parent.iterdir() if x.is_file()}
        report=parse_eval(s,p);m=report['metrics']
        self.assertEqual((m['eval/entries'],m['eval/tp_count'],m['eval/sl30']),(7,5,2))
        self.assertEqual(m['eval/closed_trade_net_pnl'],-2226.)
        self.assertEqual(m['eval/tp_market_minutes_p50'],74.)
        self.assertAlmostEqual(m['eval/reward'],-3.805870048525721)
        self.assertEqual(tuple(m['eval/adverse10_'+k] for k in ('first5','6_15','after15','any_hold')),(0,1,3,4))
        self.assertEqual(before,{str(x):digest(x) for x in p.parent.iterdir() if x.is_file()})


class LauncherTests(unittest.TestCase):
    def test_tracker_failure_does_not_stop_original_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'producer_completed'
            def failed_tracker(stop):raise RuntimeError('simulated network outage')
            command=[sys.executable,'-c','import pathlib,time; time.sleep(.05); pathlib.Path('+repr(str(marker))+').touch()']
            result=launch(command,cwd=tmp,environment=dict(os.environ),track=failed_tracker)
            self.assertEqual(result['producer_returncode'],0)
            self.assertIn('simulated network outage',result['tracking_error'])
            self.assertTrue(marker.exists())


class OfflineSDKTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('wandb') is not None,'isolated W&B SDK is unavailable')
    def test_launch_new_synthetic_source_and_automatic_offline_tracking(self):
        # An actual subprocess/SDK test with fixture metadata, never GPU training.
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);repo=Path(__file__).resolve().parents[2]
            environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HTTP_PROXY='http://127.0.0.1:18888')
            code=('import sys,time,json; from pathlib import Path; sys.path.insert(0,'+repr(str(repo))+'); '
                  'from research_tracking.tests.test_tracker import Fixture; time.sleep(.1); '
                  'Fixture(Path('+repr(tmp)+')).evaluation(); '
                  'import os; (Path('+repr(tmp)+')/"producer_environment.json").write_text('
                  'json.dumps({"HTTP_PROXY":os.environ.get("HTTP_PROXY")}))')
            command=[sys.executable,str(repo/'tools/fg_s3_wandb.py'),'launch','--run-dir',str(base/'source'),
                '--prepared',str(base/'prepared'),'--tracker-dir',str(base/'tracking'),'--mode','offline',
                '--trainer-cwd',tmp,'--poll-seconds','.05','--every','2','--proxy','http://127.0.0.1:10808',
                '--',sys.executable,'-c',code]
            result=subprocess.run(command,env=environment,capture_output=True,text=True,timeout=45)
            self.assertEqual(result.returncode,0,result.stderr[-1500:])
            self.assertIn('ORIGINAL_COMMAND_FINISHED',result.stdout)
            self.assertNotIn('tracking_error',result.stdout)
            state=json.loads(next((base/'tracking').glob('*/tracker_state.json')).read_text())
            self.assertEqual(state['canonical']['local_update'],4)
            self.assertEqual(len(state['evaluations']),1)
            self.assertTrue(all(b['exported'] for b in state['batches']))
            self.assertTrue(all(not b['synced'] for b in state['batches']))
            self.assertEqual(json.loads((base/'producer_environment.json').read_text())['HTTP_PROXY'],
                             'http://127.0.0.1:18888')

    @unittest.skipUnless(importlib.util.find_spec('wandb') is not None,'isolated W&B SDK is unavailable')
    def test_cloud_acknowledgement_uses_full_history_instead_of_sampled_keys(self):
        from research_tracking.sdk_worker import remote_history
        rows=[{'_step':i,'tracker/event_seq':i,'tracker/event_sha256':str(i)} for i in range(3)]
        class Run:
            def scan_history(self, **kwargs):
                return rows[:1] if 'keys' in kwargs else rows
        class Api:
            def run(self, path):return Run()
        _,confirmed=remote_history(Api(),'fixture','fixture',0,2)
        self.assertEqual(confirmed,{i:str(i) for i in range(3)})
        rows.append(dict(rows[0]))
        with self.assertRaises(TrackingError):remote_history(Api(),'fixture','fixture',0,2)

    @unittest.skipUnless(importlib.util.find_spec('wandb') is not None,'isolated W&B SDK is unavailable')
    def test_actual_offline_sdk_and_repeat_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);f=Fixture(base);f.evaluation()
            with Store(f.source(),base/'tracking') as store:
                im=Importer(store,2);im.canonical();im.evaluations()
                result=Writer('offline').export(store)
                self.assertEqual(len(result),2)
                self.assertTrue(all(r['status']=='OFFLINE_EXPORTED' for r in result))
                for r in result:
                    self.assertTrue(list(Path(r['sdk_directory']).glob('*.wandb')))
                im.canonical();im.evaluations();self.assertEqual(Writer('offline').export(store),[])
                self.assertEqual(store.summary()['synced_events'],0)
                for path in (store.root/'sdk').rglob('*'):
                    if path.is_file():
                        data=path.read_bytes()
                        self.assertNotIn(b'FAKE_SENSITIVE_VALUE_NEVER_UPLOAD',data)
                        self.assertNotIn(b'PRIVATE_ACCOUNT_ID',data)
                self.assertNotIn('jax',sys.modules)


if __name__=='__main__':unittest.main()
