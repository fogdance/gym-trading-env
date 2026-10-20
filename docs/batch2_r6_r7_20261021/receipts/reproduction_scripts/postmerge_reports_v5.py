"""Finalize audit reports after the real integration verification completes."""
import hashlib,json,pathlib,re,shutil,subprocess
base=pathlib.Path(__file__).resolve().parents[2];docs=base/'docs/batch2_r6_r7_20261021';out=docs/'receipts'
integration=base.parent.parent/'fg-s3-next-integration';snapshot=base.parent/'rounds/final-v5'
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def append_once(path,marker,body):
    text=path.read_text().split(marker,1)[0].rstrip()+'\n\n'+marker+'\n\n'+body.rstrip()+'\n'
    path.write_text(text)
accept=json.loads((out/'FINAL_ACCEPTANCE.json').read_text())
assert accept['code_pair']['gym']=='4c881d39d9a482542534ee678e53787547afae89'
assert accept['unique_passed']==330 and accept['failed_final']==accept['skipped_final']==0
validation=integration/'gym-trading-env/docs/batch2_r6_r7_20261021/receipts/MERGE_VALIDATION_V5.json'
v=json.loads(validation.read_text());assert v['exit_code']==0 and v['counts']['failed']==v['counts']['error']==v['counts']['skipped']==0
def canonical(c):
    name=c['classname'];start=name.find('test_')
    return (name[start:] if start>=0 else name,c['name'])
ids=set()
for item in accept['selected']:
    d=json.loads((out/(item['name']+'.json')).read_text());ids.update(map(canonical,d['cases']))
assert {canonical(c) for c in v['cases']}<=ids,'MERGE_ADDED_UNACCOUNTED_TEST_CASE'
for suffix in ['json','log','xml']:shutil.copy2(validation.with_suffix('.'+suffix),out/('MERGE_VALIDATION_V5.'+suffix))
merged=json.loads((out/'INTEGRATION_COMMANDS_V5.json').read_text())['repositories']
for name,p in merged.items():
    root=integration/name;frozen=snapshot/name
    assert git(root,'rev-parse','HEAD')==p['after']
    assert not git(root,'status','--porcelain')
    assert git(root,'rev-parse','HEAD^{tree}')==git(frozen,'rev-parse','HEAD^{tree}')
accept.update(status='B2_CODE_GO_INTEGRATED',code_merge_pair={k:d['after'] for k,d in merged.items()},
    postmerge_validation=dict(file='MERGE_VALIDATION_V5.json',counts=v['counts'],exit_code=v['exit_code'],
        argv=v['argv'],source_pair=v['sources'],same_full_trees_as_frozen_tests=True),
    final_unique_passed=330,merge_reexecutions=len(v['cases']),total_accepted_executions=330+len(v['cases']))
(out/'FINAL_ACCEPTANCE.json').write_text(json.dumps(accept,indent=2)+'\n')
table='| 仓 | 源码普通合并 HEAD |\n|---|---|\n'+'\n'.join(f"| {name} | `{p['after']}` |" for name,p in merged.items())
append_once(docs/'BATCH2_BASELINE_AND_INTEGRATION.md','## 最终代码合并与复验',table+
    f"\n\n两仓 complete tree 与最终冻结测试树逐字节相同。合并后独立复验 {len(v['cases'])} 项 PASS，exit 0，FAIL/ERROR/SKIP 均为 0；原始命令及结果见 `receipts/MERGE_VALIDATION_V5.*`。这些用例是 330 项中的重复执行，不增加不重复用例数。\n\n以上是代码合并 HEAD。随后只交付 docs 的提交与 merge，最终交付双仓 HEAD 另由工作区 `baseline-lock.json` 锁定，并用运行源码字节一致性证据关联到本表。")
append_once(docs/'BATCH2_IMPLEMENTATION_REVIEW.md','## 最终验收状态',
    f"**B2_CODE_GO_INTEGRATED**：最终冻结代码 330 个不重复用例 PASS；集成目录再执行 {len(v['cases'])} 项 PASS，合计 {330+len(v['cases'])} 次验收执行，0 FAIL/ERROR/SKIP。两仓代码合并 tree 与冻结测试 tree 相同。12 个工况各三次，旧/新批次与全 Reader 状态哈希一致，所有新版均有新进程 CP load 比对。\n\n"+table+
    '\n\n真实 TRAIN/G3/GPU/长期训练保持 NOT_RUN；TEST 保持 NOT_READ。最终文档交付 HEAD 和运行源码一致性由工作区版本锁记录。')
append_once(docs/'TEST_EVIDENCE_INDEX.md','## 集成目录复验',
    f"`MERGE_VALIDATION_V5`：{len(v['cases'])} PASS、0 FAIL/ERROR/SKIP，exit 0，实际 {v['seconds']:.2f} 秒。原始 [JSON](receipts/MERGE_VALIDATION_V5.json)、[log](receipts/MERGE_VALIDATION_V5.log)、[JUnit](receipts/MERGE_VALIDATION_V5.xml)。用例均属于上列 330 项，本轮总验收执行数为 {330+len(v['cases'])}，不重复通过用例仍为 330。\n\n代码合并 SHA、精确 tree 继承与 docs-only 交付的对应关系见 BATCH2_BASELINE_AND_INTEGRATION.md 及工作区 baseline-lock.json。")
scripts=out/'reproduction_scripts';scripts.mkdir(exist_ok=True)
for name in ['final_perf_v5.py','finalize_reports_v5.py','merge_code_v5.py','postmerge_reports_v5.py']:
    shutil.copy2(base/'artifacts/batch2'/name,scripts/name)
issues=[]
for p in docs.rglob('*.md'):
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',p.read_text()):
        if target.startswith(('http:','https:','#')):continue
        q=(p.parent/target.split('#')[0]).resolve()
        if not q.exists():issues.append(dict(file=str(p),target=target))
for p in docs.rglob('*.json'):json.loads(p.read_text())
assert not issues,issues
(out/'DOCUMENT_VALIDATION.json').write_text(json.dumps(dict(status='PASS',local_markdown_links_valid=True,
    all_json_parse=True,selected_unique_tests=330,postmerge_reexecutions=len(v['cases']),exit_code=0),indent=2)+'\n')
lock={str(p.relative_to(docs)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(docs.rglob('*'))
    if p.is_file() and p.name!='EVIDENCE_SHA256.json'}
(docs/'EVIDENCE_SHA256.json').write_text(json.dumps(dict(protocol='BATCH2_R6_R7_EVIDENCE_LOCK_V1',
    code_pair=accept['code_pair'],code_merge_pair=accept['code_merge_pair'],files=lock),indent=2)+'\n')
print(json.dumps(dict(status=accept['status'],tests=330,merge_reexecutions=len(v['cases']),
    evidence_files=len(lock),evidence_sha256=digest(docs/'EVIDENCE_SHA256.json'))))
