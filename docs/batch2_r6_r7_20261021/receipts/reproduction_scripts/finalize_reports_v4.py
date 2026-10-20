"""Build reports only from closed test jobs and completed benchmark matrices."""
import datetime,hashlib,json,pathlib,pickle,shutil,statistics,subprocess
base=pathlib.Path(__file__).resolve().parents[2];docs=base/'docs/batch2_r6_r7_20261021';out=docs/'receipts'
snapshot=base.parent/'rounds/final-v4/gym-trading-env'
final_names=['FINAL_V4_'+n for n in ['f1','f2','f3','versions','shared','readers','handoff']]
for name in final_names:
    p=snapshot/'docs/batch2_r6_r7_20261021/receipts'/(name+'.json')
    assert p.exists(),('TEST_STILL_PENDING',name)
    d=json.loads(p.read_text());assert d['exit_code']==0 and not d['counts']['failed'] and not d['counts']['error']
perf=out/'performance_final_v4/ALL_RESULTS.json';assert perf.exists(),'FINAL_PERFORMANCE_STILL_PENDING'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
for round_name in ['r6','r7b','r7v2','final-v1','final-v2','final-v3','final-v4']:
    p=base.parent/'rounds'/round_name/'gym-trading-env/docs/batch2_r6_r7_20261021/receipts'
    if not p.exists():continue
    for meta in p.glob('*.json'):
        value=json.loads(meta.read_text())
        if not isinstance(value,dict) or 'cases' not in value:continue
        for suffix in ['json','log','xml']:
            src=meta.with_suffix('.'+suffix)
            if not src.exists():continue
            dst=out/src.name
            if dst.exists():assert dst.read_bytes()==src.read_bytes(),('RECEIPT_NAME_COLLISION',str(dst))
            else:shutil.copy2(src,dst)
all_records=[];selected=[];test_ids=set()
for p in sorted(out.glob('*.json')):
    v=json.loads(p.read_text())
    if not isinstance(v,dict) or 'cases' not in v:continue
    item=dict(name=p.stem,**{k:v[k] for k in ['argv','cwd','environment','sources','counts','exit_code','seconds']},
        files={f.name:digest(f) for f in [p,p.with_suffix('.log'),p.with_suffix('.xml')] if f.exists()})
    all_records.append(item)
    if p.stem in final_names:
        selected.append(item)
        assert all(not r['status'] for r in v['sources'].values())
        for c in v['cases']:
            assert c['result']=='passed'
            test_ids.add(c['classname']+'::'+c['name'])
record=dict(status='FINAL_FROZEN_CODE_TESTS_PASS',scope='SELECTED_CPU_SYNTHETIC_TESTS_ONLY',
    code_pair={r:git(p,'rev-parse','HEAD') for r,p in [('gym',snapshot),('dreamer',snapshot.parent/'dreamerv3')]},
    selected_executions=sum(sum(v['counts'].values()) for v in selected),unique_passed=len(test_ids),
    failed_final=0,skipped_final=0,selected=selected,all_development_rounds=all_records,
    formal_training='NOT_RUN',production_G3='NOT_RUN',GPU='NOT_RUN',TEST='NOT_READ')
(out/'FINAL_ACCEPTANCE.json').write_text(json.dumps(record,indent=2)+'\n')
lines=['# 测试证据索引','',
    f"最终冻结双仓共 {record['selected_executions']} 次执行、{len(test_ids)} 个不重复用例通过，FAIL/ERROR/SKIP 均为 0。仅限下列集合，不是全项目测试通过。",'',
    '| 最终分组 | PASS | FAIL | SKIP | 退出码 | 秒 | 原始命令/环境/源身份 |',
    '|---|---:|---:|---:|---:|---:|---|']
for v in selected:lines.append(f"| {v['name']} | {v['counts']['passed']} | {v['counts']['failed']} | {v['counts']['skipped']} | {v['exit_code']} | {v['seconds']:.2f} | [JSON](receipts/{v['name']}.json) / [log](receipts/{v['name']}.log) / [JUnit](receipts/{v['name']}.xml) |")
lines+=['','## 研发与前序证据','',
    '以下每行绑定其真实双仓 SHA 和 dirty 状态，不能把旧版本的 PASS 自动计入最终版本。失败轮次保留，修复后通过的最终组在上表。完整逐用例结果见各 JSON/JUnit。','',
    '| 轮次 | PASS | FAIL | ERROR | SKIP | 退出码 | 原始收据 |','|---|---:|---:|---:|---:|---:|---|']
for v in all_records:
    if v['name'] in final_names:continue
    c=v['counts'];lines.append(f"| {v['name']} | {c['passed']} | {c['failed']} | {c['error']} | {c['skipped']} | {v['exit_code']} | [JSON](receipts/{v['name']}.json) / [log](receipts/{v['name']}.log) |")
lines+=['','运行器保留绝对 argv、cwd、CPU affinity、CPU-only 环境和双仓完整 SHA。移机复现时只替换 worktree / 新输出目录；不得复用已有 pytest basetemp，因为 pytest 会清空该测试专用目录。正式 Corpus/Run 目录不得用作 basetemp。','',
    'R6 来源负例的 15 个原始失败、Guard 传播、诊断 cwd 缺失、独立 Run session 比较合同、覆盖基础摘要以及重复恢复队列的失败均单独保留；未通过跳过、xfail 或放宽金融/准入/严格 CP 消除。正式 TRAIN/G3/GPU/长期训练为 NOT_RUN，TEST 为 NOT_READ。','']
(docs/'TEST_EVIDENCE_INDEX.md').write_text('\n'.join(lines))

# Preserve exact cold proofs plus every file hash from each persistent case.
cold=[];cold_out=out/'cold_final_v4';cold_out.mkdir(exist_ok=True)
for phase in ['f1','f2','f3']:
    root=snapshot/'artifacts/batch2'/('final_v4_'+phase+'_tmp');cases={}
    for proof in root.rglob('exact_graph_restore.json'):
        if proof.parent.name not in ('split','ordinary_split','skip_first_split'):continue
        data=json.loads(proof.read_text());run=proof.parent;matches=[]
        for p in (run/'ckpt').glob('step_*/runtime.pkl'):
            graph=pickle.loads(p.read_bytes())['streams'][0]['source']
            if graph['graph_sha256']==data['graph_sha256']:matches.append(str(p))
        assert matches,('RESTORED_GRAPH_PROOF_HAS_NO_SAVED_MEMBER',str(proof))
        label=phase+'_'+run.name
        shutil.copy2(proof,cold_out/(label+'_exact_graph_restore.json'))
        cases.setdefault(run.parent,[]).append(dict(path=str(proof),sha256=digest(proof),
            saved_runtime_members=matches,evidence=data))
    for case,proofs in cases.items():
        files={str(p.relative_to(case)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in case.rglob('*') if p.is_file()}
        cold.append(dict(stage=phase,case_root=str(case),retention='PERMANENT_UNIQUE_BASETEMP_NO_PRUNING',proofs=proofs,files=files))
        for p in case.glob('*.log'):shutil.copy2(p,cold_out/(phase+'_'+p.name))
root=snapshot/'artifacts/batch2/final_v4_versions_tmp'
for p in root.rglob('result.json'):
    v=json.loads(p.read_text())
    if not v.get('exact_graph_restored'):continue
    run=p.parent
    files={str(f.relative_to(run)):dict(bytes=f.stat().st_size,sha256=digest(f)) for f in run.rglob('*') if f.is_file()}
    cold.append(dict(stage='data_'+v['mode'],case_root=str(run),scope=v['scope'],evidence=v,files=files))
assert sum(len(v.get('proofs',[])) for v in cold)==4
(out/'FINAL_V4_COLD_ARTIFACT_INVENTORY.json').write_text(json.dumps(cold,indent=2)+'\n')

results=json.loads(perf.read_text());den={(v['mode'],v['seen']):v for v in json.loads((out/'PERFORMANCE_EXPOSURE_DENOMINATORS.json').read_text())}
rows=[]
for case in results:
    effective=den[(case['mode'],case['seen'])]['counts']['effective_target_positions'];values={}
    for version in ['before','after']:
        samples=[r[version] for r in case['repeats']]
        for s in samples:assert s['golden']==case['repeats'][0]['before']['golden']
        def median(key):return statistics.median(s[key] for s in samples)
        pending=[]
        for s in samples:
            dest=pathlib.Path(s['argv'][s['argv'].index('--out')+1])
            node=(dest/'ckpt/latest').read_text().strip()
            runtime=pickle.loads((dest/'ckpt'/node/'runtime.pkl').read_bytes())
            pending.append(len(pickle.dumps(runtime['streams'][0]['pending'],protocol=5)))
        values[version]=dict(batch_s=median('batches_per_second'),range_batch_s=[min(s['batches_per_second'] for s in samples),max(s['batches_per_second'] for s in samples)],
            rss_MiB=median('peak_rss_bytes')/2**20,snapshot_s=median('snapshot_seconds'),snapshot_MiB=median('snapshot_bytes')/2**20,
            local_mark_ms=median('local_snapshot_seconds')*1000,local_rollback_ms=median('local_rollback_seconds')*1000,
            pending_MiB=statistics.median(pending)/2**20,checkpoint_MiB=median('checkpoint_bytes')/2**20,checkpoint_save_s=median('checkpoint_seconds'),
            cold_load_s=statistics.median(s['cold_load']['seconds'] for s in samples),
            cold_rss_MiB=statistics.median(s['cold_load']['peak_rss_bytes'] for s in samples)/2**20,
            s_per_100k_effective=statistics.median(s['batch_seconds']*100000/effective for s in samples))
    rows.append(dict(mode=case['mode'],seen=case['seen'],batches=case['timed_batches'],cpus=case['cpus'],
        values=values,speedup=values['after']['batch_s']/values['before']['batch_s'],
        exposure=den[(case['mode'],case['seen'])],exact_batch_and_full_state_and_new_process_load=True))
(out/'FINAL_PERFORMANCE_SUMMARY.json').write_text(json.dumps(rows,indent=2)+'\n')
text=['# R7 性能迭代与最终成本','',
    '以下数字来自完整 WP03 Reader → 原阶段 Adapter/角色/多源调度 → Prefetch 循环；不是只测 state_dict，也不是完整模型训练或 Producer 生成速度。所有压力历史均为 DEV_ONLY。真实市场覆盖和真实 TRAIN 训练吞吐为 NOT_RUN。','',
    '## 配置与可比性','',
    '旧版：Gym 5119f1deb8fc17676b85454c20ce7793e190876d / Dreamer cbc1343971786caec979cead3025431690f1591b。新版：Gym e5ff6e1873819a962f4ec3d6639b62b711c977e6 / Dreamer a6aa3e649ca36022b767e2d89d0392868953f5e9。均为干净冻结兄弟 worktree。','',
    '使用同一份两 Corpus、原生六行为、40 分钟合成数据和同一 join 索引。B=4、T=8、K=4、Prefetch=2；双角色工况每角色 B=4。同一工况固定 seed、权重和 CPU affinity，BLAS/OMP 单线程，CPU-only。旧版和新版执行完全相同的 benchmark 文件，其 SHA 在 SOURCE_PAIRS.json 中。','',
    '1k/10k/100k/300k 表示每个 Reader 人工预置的 market 与 account 历史键数，不改变采样候选或可用标签。两批 warm-up 后，计时 N 次消费并等待恢复满 pending，避免把预先填好的队列当作零成本吞吐。小历史 N=16/8，大历史 N=2，每工况独立三次；短窗口存在计时波动，因此同时保留范围，不能外推长期训练。','',
    '旧版三次基线已在完成的 before/after 原矩阵中测量；三个安全收尾轮次只重跑新版，复用完全相同的旧版原始结果，未重新构造基线。最终新版每次也单独启动新 Python 进程 load CP 并核对全 Reader state。','',
    '## 最终端到端吞吐','',
    '| 工况 | 历史键/Reader | old batch/s | new batch/s [min,max] | 倍数 | old/new peak MiB | old/new 秒/10万有效位 | 等价 |',
    '|---|---:|---:|---:|---:|---:|---:|---|']
for r in rows:
    a=r['values']['before'];b=r['values']['after'];lo,hi=b['range_batch_s']
    text.append(f"| {r['mode']} | {r['seen']} | {a['batch_s']:.3f} | {b['batch_s']:.3f} [{lo:.3f},{hi:.3f}] | {r['speedup']:.1f}× | {a['rss_MiB']:.1f}/{b['rss_MiB']:.1f} | {a['s_per_100k_effective']:.1f}/{b['s_per_100k_effective']:.1f} | PASS |")
text+=['','三次值的中位数；RSS 为整个受测子进程的 ru_maxrss，包含导入、Reader、pending、快照和 CP 保存，不是仅数组大小。冷加载进程峰值另见 JSON。有效位按同一批原生字节重放计数，逐批摘要匹配所有 before/after 重复：F1/Multi 为原 WM target_positions；F2 为至少一个已启用监督项有效的 target 位置；F3 为既有 valid_l2_targets。F2 多头/Pair 并不共享同一 loss 分母，此列仅统一吞吐单位。完整计数与 Natural/Aux/Pair 候选见 PERFORMANCE_EXPOSURE_DENOMINATORS.json；人工历史键不提供任何训练标签。','',
    '## 快照、回滚与完整 Checkpoint 成本','',
    '| 工况/历史键 | mark ms old/new | rollback ms old/new | Prefetch save_full s old/new | snapshot MiB old/new | pending MiB old/new | CP MiB old/new | CP save s old/new | 新进程 load s old/new |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    a=r['values']['before'];b=r['values']['after']
    pairs=[('local_mark_ms',3),('local_rollback_ms',3),('snapshot_s',3),('snapshot_MiB',2),('pending_MiB',2),('checkpoint_MiB',2),('checkpoint_save_s',3),('cold_load_s',3)]
    text.append('| '+r['mode']+'/'+str(r['seen'])+' | '+' | '.join(f"{a[k]:.{n}f}/{b[k]:.{n}f}" for k,n in pairs)+' |')
text+=['','mark/rollback 是单 Reader 的精确局部撤销，均另行核对完整状态一致，不计入主 batch/s。pending MiB 是 CP 中队列 (batch,state) 用 pickle protocol 5 单独序列化的字节数；snapshot 是整个 Prefetch 快照。CP 保存包括实际完整 bundle 写盘；load 为独立新进程内 cp.load 的 wall-clock，不含 Python 导入启动，且检查 state_digest。零 optimizer 更新。','',
    '## 实际迭代与保留决定','',
    '1. 6cf302d / e0d5563：实现可嵌套 delta 事务、角色与多源组回滚；Guard 传播故障有原始失败，修复后重测。保留。','',
    '2. 版本图开发首轮：相同 100k、8 批端到端对照见 BEFORE/AFTER_100000_COMMANDS.json 和日志。F1 约 1.635→45.858 batch/s，多源约 0.204→10.034；F2 约 1.659→3.390、F3 Aux 约 0.559→1.507 暴露其覆写 sample 仍保留全量快照。此轮是开发状态，不能代替干净最终版本测试。','',
    '3. 4cfe194 / 99167e9：清除 F2/F3 覆写 sample 中的全量保存，完善 node/pending 字节与 owner 校验、Run V2 接线。897f476 的干净模型冷恢复和首组三次矩阵通过；见 performance_final。保留。','',
    '4. 0094e28 / 03e1710：故障测试先复现 coverage_basis，再绑定完整图摘要；一次导出供两个 CP 成员使用。复测三次矩阵见 performance_final_v2。保留。','',
    '5. 08b5bbe / a6aa3e6：拒绝在已恢复 pending 队列上再次 load/load_full。真实失败先保留，随后重跑最终冻结测试与本页三次矩阵。保留；没有以降低恢复检查换取性能。','',
    '6. e5ff6e1：补充显式回退和废弃分支回收的回归，原代码冷恢复将下一版本号由 3 变成 1。完整图改为精确保存 next_id，并拒绝回退的计数器；重新执行最终冻结测试和三次矩阵。只改状态元数据，保留全部采样语义。','',
    '## 剩余成本与边界','',
    '热路径仍执行原 SQLite 选窗、native 数组、账户/时序审计和阶段支持检查。完整 CP 的图验证仍物化每个保留版本交给原 source validator，因此冷加载可能比旧版更慢；上表如实列出。后续可研究一次构建共享覆盖索引后逐版本验证，但需重新证明跨来源/角色、压缩基础与 pending 的精确语义。本轮没有为缩短冷加载跳过验证。精确覆盖集合和完整持久快照仍按累计覆盖规模占内存；任意 caller 若长期不释放 lease，会保留其历史。','',
    '原始数据与所有真实命令/退出码：receipts/performance_final{,_v2,_v3}/。最终结构化表：receipts/FINAL_PERFORMANCE_SUMMARY.json。最终正式训练吞吐、GPU、真实 TRAIN 多源模型训练和生产 G3 均为 NOT_RUN。','']
(docs/'R7_PERFORMANCE_ITERATIONS.md').write_text('\n'.join(text))
print(json.dumps(dict(unique_passed=len(test_ids),selected_executions=record['selected_executions'],
    performance_cases=len(rows),model_cold_proofs=4,data_cold_cases=sum(v['stage'].startswith('data_') for v in cold))))
