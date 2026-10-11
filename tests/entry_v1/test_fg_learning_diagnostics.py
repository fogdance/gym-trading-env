import numpy as np
import pickle
from gym_trading_env.research.fg_learning_diagnostics import add_confusion, completed_returns, imagined_return


def test_rare_add_recall_exposes_constant_rejection():
    target = np.arange(100000) < 175
    result = add_confusion(target, np.zeros(100000, bool))
    assert result['legal_add'] == 175 and result['FN'] == 175
    assert result['recall'] == 0 and result['false_positive_rate'] == 0
    assert result['precision'] is None
    assert add_confusion(np.zeros(4, bool), np.zeros(4, bool))['recall'] is None


def test_return_survives_nonterminal_chunk_but_stops_at_contract_terminal():
    # Index1 can be is_last/nonterminal; the positive reward at2 must reach0.
    reward = [0, 0, 1, 7, 0, 2, 0]
    terminal = [False, False, True, False, False, True, False]
    values, complete = completed_returns(reward, terminal, .9)
    np.testing.assert_allclose(values[:6], [.9, 1, 0, 1.8, 2, 0])
    assert complete.tolist() == [True]*6 + [False]
    assert np.isnan(values[-1])


def test_imagined_return_preserves_terminal_reward_and_next_state_alignment():
    actual = imagined_return(np.array([[1., 2.], [3., 8.]]),
                            np.array([[.9, .9], [0., .9]]), [10., 10.])
    np.testing.assert_allclose(actual, [1+.9*(2+.9*10), 3])


def test_replay_uses_successor_order_instead_of_storage_order(tmp_path):
    from tools.diagnose_fg_learning import replay_chunks, OBS_KEYS
    replay = tmp_path/'replay'; replay.mkdir()
    chunks = [dict(uuid='second',succ='end',name='b.npz',length=2),
              dict(uuid='first',succ='second',name='a.npz',length=2)]
    (tmp_path/'replay.pkl').write_bytes(pickle.dumps(dict(chunks=chunks)))
    for name, values in [('a.npz',[0,1]),('b.npz',[2,3])]:
        np.savez(replay/name,action=values,**{k:np.array(values) for k in OBS_KEYS})
    actual = replay_chunks(tmp_path)
    assert np.concatenate([x['action'] for x in actual]).tolist() == [0,1,2,3]
