"""CPU archive contracts: fixtures are evidence, never restoration claims."""
import copy
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from research_tracking.tests.test_tracker import Fixture, signed
from research_tracking.io import atomic_json, digest, fingerprint
from research_tracking.milestones import merge_tags, summary_fields
from s3_archive.inspection import inspect, state
from s3_archive.archive import execute, read_plan
from s3_archive.io import ArchiveError, stable_copy, metadata_transaction
from s3_archive.verification import evidence, grade, verify
from s3_archive.publication import publish


class ArchiveFixture:
    def __init__(self, base):
        self.base=base;self.f=Fixture(base)
        f=self.f
        versions, code = {}, {}
        for name in ('gym-trading-env','dreamerv3'):
            p=base/name;p.mkdir();(p/'runtime.py').write_text('VALUE = 1\n')
            def git(*args):
                return subprocess.check_output(['git','-C',str(p),*args],stderr=subprocess.PIPE).decode().strip()
            git('init');git('config','user.name','Archive tests');git('config','user.email','archive@example.invalid')
            git('add','runtime.py');git('commit','-m','fixture source')
            versions[str(p)]=git('rev-parse','HEAD');code[str(p/'runtime.py')]=digest(p/'runtime.py')
        f.identity.update(versions=versions,code=code,packages={})
        inputs=base/'inputs';inputs.mkdir()
        for name in ('prices.csv','contract.yaml','strategy.yaml'):
            (inputs/name).write_text('fixture\n')
        env=inputs/'env.yaml'
        env.write_text(yaml.safe_dump({'trading':{'data_path':str(inputs/'prices.csv')},
            'entry':{'contract_config':str(inputs/'contract.yaml'),'strategy_config':str(inputs/'strategy.yaml')}}))
        atomic_json(inputs/'manifest.json',signed({'files':{p.name:digest(p) for p in inputs.iterdir()}}))
        settings=json.loads((f.prepared/'settings.json').read_text())['settings']
        settings.update(env_receipt=str(inputs),oracle_receipt=str(base/'oracle'),catalog=str(base/'catalog'),
                        dev_source_manifest=str(inputs/'manifest.json'))
        atomic_json(f.prepared/'settings.json',signed({'settings':settings}))
        prep=json.loads((f.prepared/'manifest.json').read_text())
        prep.update(code=code,versions=versions,dirty={k:[] for k in versions},
            files={p.name:digest(p) for p in f.prepared.iterdir() if p.name!='manifest.json'})
        atomic_json(f.prepared/'manifest.json',signed({k:v for k,v in prep.items() if k!='content_hash'}))
        f.identity['preparation_sha256']=digest(f.prepared/'manifest.json')
        for name in ('source_admission.json','source_transfer.json'):
            atomic_json(f.root/name,signed({'kind':'fixture'}))
        atomic_json(f.root/'run_identity.json',signed({'identity':f.identity}))
        cp=f.root/'ckpt/step_000000000100'
        (f.root/'rollout').mkdir();journal=f.root/'rollout/teacher_journal.jsonl';journal.write_text('{"commit":1}\n')
        states={'collector':{'journal_bytes':journal.stat().st_size,'commit_seq':1},
                'v2_counters':{'all_updates':4,'cycles':1}}
        for name, value in states.items():(cp/(name+'.pkl')).write_bytes(pickle.dumps(value))
        atomic_json(cp/'manifest.json',dict(protocol='FG_FULL_V1',schema=1,step=100,identity=f.identity,
            files={p.name:digest(p) for p in cp.iterdir() if p.name not in ('done','manifest.json')}))
        self.eval=f.evaluation().parent
        (self.eval/'dev_env.yaml').write_bytes(env.read_bytes())
        self.plan=inspect(f.root,100,self.eval,prepared=f.prepared,archive_root=base/'archives')
        self.plan['plan_sha256']=fingerprint(self.plan)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.fixture=ArchiveFixture(self.base)
        self.plan=self.fixture.plan
    def tearDown(self):self.temp.cleanup()
    def archive(self,**kwargs):
        result=execute(self.plan,execute=True,reserve_gib=0,**kwargs)
        return Path(result['archive_dir'])

    def test_dry_run_and_l1_preservation_are_not_l3(self):
        before={str(p):digest(p) for p in self.fixture.f.root.rglob('*') if p.is_file()}
        result=execute(self.plan,preserve=True,reserve_gib=0)
        self.assertFalse(Path(result['archive_dir']).exists())
        root=self.archive(preserve=True)
        self.assertEqual(evidence(root)['status'],'EVIDENCE_VERIFIED')
        m=json.loads((root/'archive_manifest.json').read_text());self.assertEqual(m['achieved_level'],'EVIDENCE')
        self.assertIsNone(m['verification']['resume_smoke_pass'])
        after={str(p):digest(p) for p in self.fixture.f.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        self.assertTrue((root/'checkpoint/step_000000000100/agent.pkl').exists())

    def test_checkpoint_missing_done_blocks_inspection(self):
        (self.fixture.f.root/'ckpt/step_000000000100/done').unlink()
        with self.assertRaises(ArchiveError):inspect(self.fixture.f.root,100,self.fixture.eval,prepared=self.fixture.f.prepared)

    def test_cli_failed_inspection_writes_ready_false(self):
        f=self.fixture.f;(f.root/'ckpt/step_000000000100/done').unlink()
        out=self.base/'failed-inspect.json';cli=Path(__file__).resolve().parents[2]/'tools/s3_archive.py'
        process=subprocess.run([sys.executable,'-B',str(cli),'inspect','--run-dir',str(f.root),
            '--checkpoint','100','--eval-dir',str(self.fixture.eval),'--prepared',str(f.prepared),
            '--json-out',str(out)],capture_output=True,text=True)
        self.assertEqual(process.returncode,2)
        self.assertFalse(json.loads(process.stdout)['ready'])
        self.assertFalse(json.loads(out.read_text())['ready'])

    def test_cli_inspection_output_cannot_write_prepared_source(self):
        f=self.fixture.f;out=f.prepared/'archive-plan.json'
        cli=Path(__file__).resolve().parents[2]/'tools/s3_archive.py'
        process=subprocess.run([sys.executable,'-B',str(cli),'inspect','--run-dir',str(f.root),
            '--checkpoint','100','--eval-dir',str(self.fixture.eval),'--prepared',str(f.prepared),
            '--json-out',str(out)],capture_output=True,text=True)
        self.assertEqual(process.returncode,2)
        self.assertFalse(out.exists())

    def test_cli_evidence_lifecycle_stays_offline_and_preserves_source(self):
        cli=Path(__file__).resolve().parents[2]/'tools/s3_archive.py';f=self.fixture.f
        def command(*args):
            process=subprocess.run([sys.executable,'-B',str(cli),*map(str,args)],
                capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            return json.loads(process.stdout)
        before={str(p):digest(p) for root in (f.root,f.prepared)
                for p in root.rglob('*') if p.is_file()}
        plan=self.base/'cli-plan.json';archives=self.base/'cli-archives'
        result=command('inspect','--run-dir',f.root,'--checkpoint',100,'--eval-dir',self.fixture.eval,
            '--prepared',f.prepared,'--archive-root',archives,'--json-out',plan)
        self.assertTrue(result['ready'])
        dry=command('archive','--plan',plan,'--preserve-checkpoint','--reserve-gib',0)
        self.assertEqual(dry['status'],'DRY_RUN');self.assertFalse(archives.exists())
        copied=command('archive','--plan',plan,'--preserve-checkpoint','--reserve-gib',0,'--execute')
        self.assertEqual(copied['achieved_level'],'EVIDENCE')
        archive=Path(copied['archive_dir'])
        self.assertEqual(command('verify','--archive-dir',archive)['status'],'EVIDENCE_VERIFIED')
        published=command('publish','--archive-dir',archive,'--mode','offline')
        self.assertEqual(published['status'],'PENDING')
        listed=command('list','--archive-root',archives)
        self.assertEqual(len(listed),1);self.assertEqual(listed[0]['achieved_level'],'EVIDENCE')
        self.assertEqual(listed[0]['wandb']['milestone_publish_state'],'PENDING')
        after={str(p):digest(p) for root in (f.root,f.prepared)
               for p in root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_embedded_receipts_do_not_rebase_relative_file_indices(self):
        inputs=self.base/'inputs';embedded=[]
        for name in ('a','b'):
            p=inputs/'traces'/name;p.mkdir(parents=True);(p/'account_state.json').write_text(name)
            doc=signed({'files':{'account_state.json':digest(p/'account_state.json')}})
            atomic_json(p/'manifest.json',doc);embedded.append(doc)
        doc=json.loads((inputs/'manifest.json').read_text());doc['reports']=embedded
        doc['files'].update({str(p.relative_to(inputs)):digest(p) for p in (inputs/'traces').rglob('*') if p.is_file()})
        atomic_json(inputs/'manifest.json',signed({k:v for k,v in doc.items() if k!='content_hash'}))
        p=inspect(self.fixture.f.root,100,self.fixture.eval,prepared=self.fixture.f.prepared,archive_root=self.base/'archives')
        self.assertFalse(p['missing'])

    def test_receipt_sha_mismatch_and_wrong_config_block(self):
        f=self.fixture.f;path=self.fixture.eval/'receipt.json';r=json.loads(path.read_text());r['checkpoint_manifest_sha256']='0'*64
        atomic_json(path,signed({k:v for k,v in r.items() if k!='content_hash'}))
        with self.assertRaises(ArchiveError):inspect(f.root,100,self.fixture.eval,prepared=f.prepared)
        (f.root/'config.yaml').write_text('agent: foreign\n')
        with self.assertRaises(ValueError):inspect(f.root,100,self.fixture.eval,prepared=f.prepared)

    def test_receipt_opening_count_must_reconcile_with_account(self):
        p=self.fixture.eval/'receipt.json';r=json.loads(p.read_text());r['entries']=2
        atomic_json(p,signed({k:v for k,v in r.items() if k!='content_hash'}))
        with self.assertRaises(ValueError):inspect(self.fixture.f.root,100,self.fixture.eval,prepared=self.fixture.f.prepared)

    def test_v2_duplicate_attributions_remain_one_archived_trade(self):
        path=self.fixture.eval/'account_state.json';account=json.loads(path.read_text())
        account['attributions'].append(copy.deepcopy(account['attributions'][0]))
        atomic_json(path,account)
        self.plan=inspect(self.fixture.f.root,100,self.fixture.eval,
            prepared=self.fixture.f.prepared,archive_root=self.base/'archives')
        self.assertEqual(self.plan['eval_derived']['metric_schema'],'S3_EVAL_METRIC_V2')
        root=self.archive(preserve=True)
        report=evidence(root)
        self.assertEqual(report['historical_metrics']['eval/tp_count'],1)
        self.assertEqual(report['historical_metrics']['eval/closed_trade_net_pnl'],282)

    def test_historical_commit_survives_current_dirty_head(self):
        repo=Path(next(iter(self.fixture.f.identity['versions'])))
        (repo/'runtime.py').write_text('VALUE = 99\n')
        p=inspect(self.fixture.f.root,100,self.fixture.eval,prepared=self.fixture.f.prepared,archive_root=self.base/'archives')
        self.assertTrue(p['ready'])
        root=Path(execute(p,execute=True,reserve_gib=0)['archive_dir'])
        self.assertEqual((root/'provenance/source_bundle'/repo.name/'runtime.py').read_text(),'VALUE = 1\n')
        self.assertEqual((root/'provenance/source_bundle'/(repo.name+'.execution.binary.patch')).read_bytes(),b'')

    def test_verified_original_dirty_execution_bytes_are_snapshotted(self):
        f=self.fixture.f;repo=Path(next(iter(f.identity['versions'])));runtime=repo/'runtime.py'
        runtime.write_text('VALUE = 2\n');f.identity['code'][str(runtime)]=digest(runtime)
        prep=json.loads((f.prepared/'manifest.json').read_text());prep['code']=f.identity['code'];prep['dirty'][str(repo)]=[' M runtime.py']
        atomic_json(f.prepared/'manifest.json',signed({k:v for k,v in prep.items() if k!='content_hash'}))
        f.identity['preparation_sha256']=digest(f.prepared/'manifest.json')
        atomic_json(f.root/'run_identity.json',signed({'identity':f.identity}))
        cp=f.root/'ckpt/step_000000000100';m=json.loads((cp/'manifest.json').read_text());m['identity']=f.identity;atomic_json(cp/'manifest.json',m)
        r=json.loads((self.fixture.eval/'receipt.json').read_text());r['checkpoint_manifest_sha256']=digest(cp/'manifest.json')
        atomic_json(self.fixture.eval/'receipt.json',signed({k:v for k,v in r.items() if k!='content_hash'}))
        p=inspect(f.root,100,self.fixture.eval,prepared=f.prepared,archive_root=self.base/'archives')
        root=Path(execute(p,execute=True,reserve_gib=0)['archive_dir'])
        self.assertEqual((root/'provenance/source_bundle'/repo.name/'runtime.py').read_text(),'VALUE = 2\n')
        self.assertTrue((root/'provenance/source_bundle'/(repo.name+'.execution.binary.patch')).stat().st_size)

    def test_repeat_execution_and_hash_tamper(self):
        root=self.archive(preserve=True)
        result=execute(self.plan,execute=True,preserve=True,reserve_gib=0)
        self.assertEqual(result['status'],'EXISTING_ARCHIVE_VERIFIED')
        self.assertEqual(len(list((self.base/'archives').glob('s3-*'))),1)
        (root/'checkpoint/step_000000000100/agent.pkl').write_bytes(b'corrupt')
        with self.assertRaises(ArchiveError):evidence(root)

    def test_atomic_level_upgrade_preserves_previous_archive(self):
        root=self.archive(preserve=True)
        before=digest(root/'checkpoint/step_000000000100/agent.pkl')
        execute(self.plan,execute=True,level='FULL',reserve_gib=0)
        self.assertEqual(evidence(root)['status'],'EVIDENCE_VERIFIED')
        self.assertEqual(digest(root/'checkpoint/step_000000000100/agent.pkl'),before)
        old=self.base/'archives/.revisions'/root.name/'v1'
        self.assertEqual(evidence(old)['status'],'EVIDENCE_VERIFIED')

    def test_insufficient_space_creates_no_archive(self):
        with patch('s3_archive.archive.shutil.disk_usage',return_value=type('Disk',(),{'free':0})()):
            with self.assertRaises(ArchiveError):self.archive()
        self.assertFalse((self.base/'archives').exists())

    def test_source_change_and_append_only_boundary(self):
        row=next(r for r in self.plan['copy_plan'] if r['source'].endswith('/config.yaml'))
        Path(row['source']).write_text('foreign\n')
        with self.assertRaises(ArchiveError):stable_copy(row,self.base/'copied')
        journal=next(r for r in self.plan['copy_plan'] if r['prefix'])
        with Path(journal['source']).open('ab') as out:out.write(b'new tail\n')
        stable_copy(journal,self.base/'prefix')
        self.assertEqual((self.base/'prefix').stat().st_size,journal['bytes'])

    def test_change_during_copy_is_rejected(self):
        row=next(r for r in self.plan['copy_plan'] if r['source'].endswith('/config.yaml'))
        from s3_archive.io import prefix_hash
        calls=[0]
        def changing(path,size):
            calls[0]+=1
            if calls[0]==2:Path(path).write_bytes(b'x'*size)
            return prefix_hash(path,size)
        with patch('s3_archive.io.prefix_hash',side_effect=changing):
            with self.assertRaises(ArchiveError):stable_copy(row,self.base/'during')
        self.assertFalse((self.base/'during').exists())

    def test_metadata_pickle_cannot_execute_arbitrary_globals(self):
        class Payload:
            def __reduce__(self):return (eval,('1+1',))
        p=self.base/'metadata.pkl';p.write_bytes(pickle.dumps(Payload()))
        with self.assertRaises(ArchiveError):state(p)

    def test_failed_copy_retry_retains_staging_then_completes(self):
        real=stable_copy;count=[0]
        def fail(row,target):
            count[0]+=1
            if count[0]==3:raise OSError('injected transient copy failure')
            return real(row,target)
        with patch('s3_archive.archive.stable_copy',side_effect=fail):
            with self.assertRaises(OSError):self.archive(preserve=True)
        self.assertTrue((self.base/'archives/.staging').exists())
        root=self.archive(preserve=True);self.assertEqual(evidence(root)['status'],'EVIDENCE_VERIFIED')

    def test_partial_source_snapshot_is_completed_on_retry(self):
        p=self.base/'archives/.staging'/(self.plan['archive_id']+'.tmp')/'provenance/source_bundle/gym-trading-env'
        p.mkdir(parents=True)
        root=self.archive(preserve=True)
        self.assertEqual(evidence(root)['status'],'EVIDENCE_VERIFIED')

    def test_failed_metadata_seal_recovers_prior_verified_archive(self):
        root=self.archive(preserve=True);before=(root/'README.md').read_bytes()
        with self.assertRaises(OSError):
            with metadata_transaction(root):
                (root/'README.md').write_text('interrupted metadata')
                raise OSError('injected interruption')
        self.assertEqual((root/'README.md').read_bytes(),before)
        self.assertEqual(evidence(root)['status'],'EVIDENCE_VERIFIED')

    def test_unsafe_restore_dependency_never_raises_grade(self):
        root=self.archive(level='FULL')
        m=json.loads((root/'archive_manifest.json').read_text())
        self.assertEqual(m['achieved_level'],'EVIDENCE');self.assertEqual(m['archive_state'],'PARTIAL')
        m=grade(m,{'status':'ISOLATED_RESUME_SMOKE_PASS'},'resume-smoke')
        self.assertEqual(m['achieved_level'],'EVIDENCE')
        m=grade(m,{'status':'ISOLATED_EVAL_PASS'},'eval');self.assertEqual(m['achieved_level'],'FULL')
        m['verification']['missing_dependencies']=[{'kind':'restore','reason':'SNAPSHOT_UNSAFE'}]
        self.assertEqual(grade(m,{'status':'ISOLATED_RESUME_SMOKE_PASS'},'resume-smoke')['achieved_level'],'EVAL')

    def test_offline_publish_privacy_and_tag_idempotency(self):
        root=self.archive()
        result=publish(root,mode='offline');self.assertEqual(result['status'],'PENDING')
        self.assertEqual(publish(root,mode='offline')['idempotency_key'],result['idempotency_key'])
        text=(root/'wandb/milestone.json').read_text()
        payload=json.loads(text)
        self.assertRegex(payload['idempotency_key'],r'^[0-9a-f]{64}$')
        self.assertEqual(payload['archive_id'],root.name)
        self.assertNotIn(str(self.base),text);self.assertNotIn('FAKE_SENSITIVE',text);self.assertNotIn('PRIVATE_ACCOUNT',text)
        self.assertEqual(merge_tags(['existing'],['existing','milestone']),['existing','milestone'])
        self.assertEqual(evidence(root)['status'],'EVIDENCE_VERIFIED')

    def test_cloud_summary_uses_dotted_leaf_fields(self):
        value={'receipt_sha256':'a'*64,'metrics':{'eval/add_executed':7}}
        self.assertEqual(summary_fields(value,'archive/key'),
            {'archive/key.receipt_sha256':'a'*64,'archive/key.metrics.eval/add_executed':7})

    def test_explicit_model_validation_and_plan_tampering(self):
        root=self.archive(preserve=True)
        with self.assertRaises(ArchiveError):verify(root,mode='eval')
        path=self.base/'plan.json';atomic_json(path,self.plan);self.assertEqual(read_plan(path)['archive_id'],self.plan['archive_id'])
        p=copy.deepcopy(self.plan);p['ready']=False;atomic_json(path,p)
        with self.assertRaises(ArchiveError):read_plan(path)

    def test_unsupported_hardlinks_and_source_overlap(self):
        row=next(r for r in self.plan['copy_plan'] if r['source'].endswith('/config.yaml'))
        target=self.base/'hardlinked';os.link(row['source'],target)
        with self.assertRaises(ArchiveError):stable_copy(row,target)
        p=copy.deepcopy(self.plan);p['archive_root']=str(self.fixture.f.root/'archive')
        with self.assertRaises(ArchiveError):execute(p)

    def test_other_protocol_families_keep_their_identity(self):
        # Family is explicit metadata; E2's historical mode remains tutor.
        for mode, family in [('a2','s3-a2'),('b2','s3-b2'),('tutor','s3-e')]:
            f=self.fixture.f;f.identity['mode']=mode
            atomic_json(f.root/'run_identity.json',signed({'identity':f.identity}))
            cp=f.root/'ckpt/step_000000000100';m=json.loads((cp/'manifest.json').read_text());m['identity']=f.identity;atomic_json(cp/'manifest.json',m)
            r=json.loads((self.fixture.eval/'receipt.json').read_text());r['checkpoint_manifest_sha256']=digest(cp/'manifest.json')
            atomic_json(self.fixture.eval/'receipt.json',signed({k:v for k,v in r.items() if k!='content_hash'}))
            p=inspect(f.root,100,self.fixture.eval,prepared=f.prepared,family=family,archive_root=self.base/'archives')
            self.assertEqual(p['source']['experiment_family'],family)


if __name__=='__main__':unittest.main()
