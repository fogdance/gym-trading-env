import copy,json,os,pickle,shutil,subprocess,sys
from pathlib import Path
import pytest

from data_multi_fixture import build_sources,exact
from batch2_cold_worker import make
from gym_trading_env.research.fg_replay import digest


@pytest.fixture(scope='module')
def config_path(tmp_path_factory):return build_sources(tmp_path_factory.mktemp('batch2_cold_sources'))


def worker(config_path,root,mode,amount,stop,resume=False,seen=0):
    argv=[sys.executable,'-B',str(Path(__file__).with_name('batch2_cold_worker.py')),'--config',str(config_path),
          '--root',str(root),'--mode',mode,'--amount',str(amount),'--stop',str(stop),'--seen',str(seen)]
    if resume:argv.append('--resume')
    result=subprocess.run(argv,capture_output=True,text=True,timeout=180)
    root.mkdir(parents=True,exist_ok=True)
    tag='resume' if resume else 'start'
    (root/(tag+'.log')).write_text(result.stdout+result.stderr)
    (root/(tag+'_command.json')).write_text(json.dumps(dict(argv=argv,exit_code=result.returncode),indent=2)+'\n')
    assert result.returncode==0,result.stdout+result.stderr
    return pickle.loads((root/'evidence.pkl').read_bytes())


@pytest.mark.parametrize('mode,amount,seen',[('f1',1,0),('f3_aux',2,0),('multi',4,0),('multi_f3',4,100001)])
def test_new_process_strict_checkpoint_matches_continuous_batches_and_all_states(config_path,tmp_path,mode,amount,seen):
    continuous=worker(config_path,tmp_path/'continuous',mode,amount,8,seen=seen)
    worker(config_path,tmp_path/'split',mode,amount,4,seen=seen)
    restored=worker(config_path,tmp_path/'split',mode,amount,8,resume=True,seen=seen)
    exact(continuous,restored)
    report=json.loads((tmp_path/'split/result.json').read_text())
    assert report['exact_graph_restored'] and report['node_count']<=amount


@pytest.mark.parametrize('attack',['graph_member','cross_member','missing_delta','missing_done','rollback_latest','journal','coverage_basis'])
def test_strict_payload_faults_reject_before_agent_load(config_path,tmp_path,attack):
    root=tmp_path/'run';worker(config_path,root,'multi_f3',4,3)
    node=root/'ckpt/step_000000000003'
    if attack in ('graph_member','cross_member','missing_delta','coverage_basis'):
        name='course_reader' if attack=='cross_member' else 'runtime'
        p=node/(name+'.pkl');state=pickle.loads(p.read_bytes())
        if attack=='cross_member':state['reader_ref']['digest']='f'*64
        elif attack=='graph_member':state['streams'][0]['source']['graph_sha256']='f'*64
        elif attack=='missing_delta':state['streams'][0]['source']['nodes'][-1]['delta'].pop()
        else:
            from gym_trading_env.research.fg_s3_reader_versions import _hash
            graph=state['streams'][0]['source']
            graph['base']['seen'][0]['market'].append(['CZCE.FG2701',99999991])
            graph['graph_sha256']=_hash({k:v for k,v in graph.items() if k!='graph_sha256'})
        p.write_bytes(pickle.dumps(state))
        manifest=json.loads((node/'manifest.json').read_text());manifest['files'][p.name]=digest(p)
        (node/'manifest.json').write_text(json.dumps(manifest))
    elif attack=='missing_done':(node/'done').unlink()
    elif attack=='rollback_latest':(root/'ckpt/latest').write_text('step_000000000000')
    else:
        with (root/'updates.jsonl').open('r+b') as f:f.write(b'CHANGED')
    _,source,prefetch,_,_,_,cp=make(json.loads(config_path.read_text()),'multi_f3',root,4)
    called=[]
    cp.agent.load_exact=lambda state:called.append(state)
    before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):cp.load()
    assert not called
    assert before=={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    source.close()
