"""Lock final delivery HEADs outside both repositories, after docs-only merge."""
import datetime,hashlib,json,os,pathlib,subprocess,uuid
base=pathlib.Path(__file__).resolve().parents[2];integration=base.parent.parent/'fg-s3-next-integration'
snapshot=base.parent/'rounds/final-v5';docs=integration/'gym-trading-env/docs/batch2_r6_r7_20261021'
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()
accept=json.loads((docs/'receipts/FINAL_ACCEPTANCE.json').read_text());assert accept['status']=='B2_CODE_GO_INTEGRATED'
lock=json.loads((docs/'EVIDENCE_SHA256.json').read_text())
for name,v in lock['files'].items():assert (docs/name).stat().st_size==v['bytes'] and sha(docs/name)==v['sha256'],name
assert set(lock['files'])=={str(p.relative_to(docs)) for p in docs.rglob('*') if p.is_file() and p.name!='EVIDENCE_SHA256.json'}
audit=json.loads((docs/'receipts/FINAL_PROTECTED_ASSET_AND_SOURCE_AUDIT.json').read_text())
protected=0
for root,v in audit['assets'].items():
    root=pathlib.Path(root);actual={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
    assert actual==v['sha256'];protected+=len(actual)
repositories={};runtime={}
for name in ['gym-trading-env','dreamerv3']:
    target=integration/name;frozen=snapshot/name;feature=base if name=='gym-trading-env' else base.parent/'dreamerv3'
    assert not git(target,'status','--porcelain') and not git(feature,'status','--porcelain')
    assert git(target,'branch','--show-current')=='integration/s3-next'
    assert git(target,'rev-parse','HEAD^{tree}')==git(feature,'rev-parse','HEAD^{tree}')
    folders=['src/gym_trading_env','e_data_v1'] if name=='gym-trading-env' else ['dreamerv3','embodied']
    def files(root):
        values={str(p.relative_to(root)):sha(p) for folder in folders for p in (root/folder).rglob('*.py') if 'tests' not in p.parts}
        if name=='gym-trading-env':values['tools/fg_s3_v2_experiment.py']=sha(root/'tools/fg_s3_v2_experiment.py')
        return values
    a,b=files(target),files(frozen);assert a==b,('DELIVERY_RUNTIME_DIFFERS_FROM_TESTED_SOURCE',name)
    runtime[name]=dict(frozen_test_sha=git(frozen,'rev-parse','HEAD'),runtime_files=a,exact_byte_identity=True,
        code_merge_sha=accept['code_merge_pair'][name])
    repositories[name]=dict(path=str(target),sha=git(target,'rev-parse','HEAD'),branch='integration/s3-next',dirty=False,
        merged_feature_branch='feat/batch2-r6-r7-20261021',feature_head=git(feature,'rev-parse','HEAD'),
        code_merge_sha=accept['code_merge_pair'][name],tested_source_sha=git(frozen,'rev-parse','HEAD'))
old_path=integration/'baseline-lock.json';old=old_path.read_bytes();old_sha=hashlib.sha256(old).hexdigest()
backup=integration/('baseline-lock.before-batch2-r6-r7-20261021.'+old_sha[:12]+'.json')
if backup.exists():assert backup.read_bytes()==old
else:
    with backup.open('xb') as f:f.write(old)
state=dict(schema='S3_DUAL_REPO_BASELINE_V1',baseline='Batch 2 R6 source freeze + R7 exact transactional/versioned Reader',
    status='B2_CODE_GO_CPU_VALIDATED_FORMAL_NOT_RUN',updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    repositories=repositories,previous_baseline=dict(lock=str(backup),sha256=old_sha,
        scope='Historical capabilities/tests only; not counted as reruns on this delivery'),
    runtime_byte_identity=runtime,capabilities=['common_F1_freeze_all_successor_routes','exact_optimizer_slot_ownership',
        'nested_reader_group_transactions','source_role_exact_rollback','self_contained_reader_version_graph',
        'persisted_monotonic_version_counter','independent_NATURAL_AUX_and_multi_source_RNG',
        'exact_F1_F2_F3_cold_restore','strict_published_latest_and_journal','unchanged_V3_Corpus_ABI_3_1_0_and_verifier_pins'],
    tests=dict(scope='Selected CPU synthetic tests; not a full-project or formal-training admission',
        summary=str(docs/'receipts/FINAL_ACCEPTANCE.json'),summary_sha256=sha(docs/'receipts/FINAL_ACCEPTANCE.json'),
        evidence_lock=str(docs/'EVIDENCE_SHA256.json'),evidence_lock_sha256=sha(docs/'EVIDENCE_SHA256.json')),
    test_totals=dict(passed=accept['final_unique_passed'],unique_passed=accept['final_unique_passed'],frozen_code_executions=accept['selected_executions'],
        merge_reexecutions=accept['merge_reexecutions'],total_accepted_executions=accept['total_accepted_executions'],
        failed_final=0,skipped=0,development_failures_preserved=True),
    protected_assets=dict(verified_unchanged_files=protected,audit=str(docs/'receipts/FINAL_PROTECTED_ASSET_AND_SOURCE_AUDIT.json'),
        frozen_producer=audit['frozen_producer'],producer_pin=audit['producer_pin']),
    corpus_compatibility='3.1.0',reader_stream_state_protocol='READER_STREAM_STATE_V2',
    prefetch_pending_state_protocol='PREFETCH_PENDING_STATE_V2',recovery_policy='STRICT_PUBLISHED_LATEST_V2',
    tracking=dict(validation='NOT_RUN_THIS_TASK',sdk_version=None,source_sha256=sha(integration/'gym-trading-env/tools/fg_s3_wandb.py'),
        tracker_started=False,network_uploads=0),
    not_run=['real_published_TRAIN_multi_Corpus_training_integration','production_Native_G3','GPU','long_formal_training','research_performance'],
    TEST='NOT_READ',authorization='No formal training authorization inferred from engineering acceptance')
pending=integration/('.baseline-lock.batch2-'+uuid.uuid4().hex+'.json')
with pending.open('x') as f:
    f.write(json.dumps(state,indent=2)+'\n');f.flush();os.fsync(f.fileno())
os.replace(pending,old_path)
print(json.dumps(dict(status=state['status'],repositories=repositories,unique_passed=accept['final_unique_passed'],
    merge_reexecutions=accept['merge_reexecutions'],protected_files=protected,lock=str(old_path),sha256=sha(old_path))))
