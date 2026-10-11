"""Failed final source admission cannot leave a pilot completion report."""
from types import SimpleNamespace

import pytest

from gym_trading_env.research import fg_single3_pilot as pilot
from gym_trading_env.research.fg_replay import signed, write_json


def test_pilot_source_hash_failure_leaves_no_completion_report(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setenv('JAX_PLATFORMS','cpu')
    monkeypatch.syspath_prepend('/home/v/Documents/work/dreamerv3')
    from dreamerv3 import fg_single3_offline, main
    dataset=tmp_path/'dataset';dataset.mkdir()
    coverage=tmp_path/'coverage';coverage.mkdir()
    horizon=tmp_path/'horizon';horizon.mkdir()
    for path in (dataset,coverage,horizon):write_json(path/'manifest.json',signed({'fixture':True}))
    candidates=('C0','C1','C2','C3')
    pools=[c+'_'+p for c in candidates for p in ('natural','natural_oracle')]
    data=dict(full_training_fold=True,traces=[dict(sequence_starts=100)],
              pool_members={p:[0] for p in pools},views={p:p for p in pools})
    views={c:dict(path=str(tmp_path/'view.json'),rows=[dict(env_config=str(tmp_path/'env.yaml'))]) for c in candidates}
    write_json(dataset/'inputs.json',signed(dict(training_views=views)))
    monkeypatch.setattr(pilot,'validate_dataset',lambda _:data)
    monkeypatch.setattr(pilot,'require_coverage',lambda *_:None)
    monkeypatch.setattr(pilot,'require_h',lambda *_:None)
    monkeypatch.setattr(pilot,'cpu_runtime',lambda _:None)
    monkeypatch.setattr(pilot,'source_witness',lambda *_:{})
    monkeypatch.setattr(pilot.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='CPU wiring fixture'))
    # CPU configuration resolution is real; only expensive already-required
    # Gates are fixtures. No price file is opened and no Agent is instantiated.
    monkeypatch.setattr(fg_single3_offline,'validate',lambda *a,**k:dict(identity_hash='fixture'))
    def fail(_):raise ValueError('actual final source hash failure fixture')
    monkeypatch.setattr(pilot,'check_files',fail)
    output=tmp_path/'pilot'
    config=dict(dreamer_root='/home/v/Documents/work/dreamerv3',output_root=str(tmp_path),minimum_free_disk_gib=0)
    with pytest.raises(ValueError,match='final source hash failure'):
        pilot.prepare_pilot(config,dataset,coverage,horizon,output)
    assert not (output/'pilot_comparison.md').exists()
    assert not (output/'manifest.json').exists()
