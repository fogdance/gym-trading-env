from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gym_trading_env.research.fg_replay import signed,write_json
from gym_trading_env.research import fg_s3_e_quality_corpus_cli as cli


def test_dev_clip_counts_market_advances_and_keeps_teacher_suffix():
    lock=signed(dict(start_cursor=0,end_cursor=5,suffix_end_cursor=8,official_days=['2024-04-01'],
        last_time='2024-04-01 09:06:00+08:00'))
    market=SimpleNamespace(frame=pd.DataFrame({'trading_date':['2024-04-01']*6},
        index=pd.date_range('2024-04-01 09:01',periods=6,freq='min',tz='Asia/Shanghai')),
        market_clock_counter=SimpleNamespace(valid=np.array([1,0,1,1,0,1],bool)))
    result=cli._execution_scope(lock,market,'dev',2)
    assert result['end_cursor']==3 and result['suffix_end_cursor']==8
    assert result['execution_mode']=='dev' and result['source_window_end_cursor']==5
    assert lock['end_cursor']==5
    with pytest.raises(ValueError,match='step-limited'):
        cli._execution_scope(lock,market,'data',2)
    with pytest.raises(ValueError,match='positive'):
        cli._execution_scope(lock,market,'dev',None)


def test_dev_corpus_cannot_be_relabelled_as_native_verified(tmp_path):
    write_json(tmp_path/'corpus_manifest.json',signed(dict(protocol=cli.SCHEMA,
        fold='TRAIN',training_admitted=False,execution_mode='dev')))
    with pytest.raises(ValueError,match='DEV_CORPUS_NOT_CERTIFIED'):
        cli.verify_corpus(tmp_path)


def test_debug_preserves_traceback_and_machine_readable_failure(tmp_path,capsys):
    assert cli.main(['generate','--source-pool',str(tmp_path/'missing.json'),
        '--env-receipt',str(tmp_path/'receipt'),'--start-day','2018-08-08',
        '--end-day','2018-08-08','--out',str(tmp_path/'output'),'--debug'])==2
    captured=capsys.readouterr()
    assert 'Traceback' in captured.err and 'NO_GO' in captured.out
    assert not (tmp_path/'output').exists()
