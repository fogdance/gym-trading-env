"""Posthoc-only scoring numerics; these controlled rows are not GPU evidence."""
import numpy as np
import pytest

from gym_trading_env.research.fg_s3_prediction_evaluation import outcome_scores, decision_scores, probability


def test_matched_outcomes_constant_prior_and_missing_rare_class():
    rows = [dict(prediction=dict(event=[.8,.04,.04,.04,.04,.04],early=.1),
                 label=dict(valid_event=True,event=0,valid_early=True,early10=0)),
            dict(prediction=dict(event=[.04,.04,.04,.8,.04,.04],early=.9),
                 label=dict(valid_event=True,event=3,valid_early=True,early10=1)),
            dict(prediction=dict(event=[1/6]*6,early=.5),
                 label=dict(valid_event=False,event=0,valid_early=False,early10=0))]
    m = outcome_scores(rows)
    assert m['event_samples'] == m['early_samples'] == 2
    np.testing.assert_allclose(m['event_nll'],-np.log(.8))
    np.testing.assert_allclose(m['event_brier'],.048)
    np.testing.assert_allclose(m['early_brier'],.01)
    np.testing.assert_allclose(m['event_constant_prior_nll'],np.log(2))
    assert m['classes']['SL']['support'] == 1 and m['classes']['SL']['recall'] == 1
    assert m['classes']['TP_LATE']['recall'] is None
    assert outcome_scores([])['event_nll'] is None
    assert outcome_scores([])['event_calibration']['ece'] is None


def test_unknown_old_logits_and_invalid_prediction_fail_closed():
    m = decision_scores([dict(p_add=.1,b2={'b2/p_add_base':.2,'b2/p_add_final':.1,'b2/delta_logit':-.5})])
    assert m['b2/base_logit_add_minus_wait']['mean'] is None
    assert m['clip_saturation_fraction'] is None
    with pytest.raises(ValueError,match='Invalid sealed'):
        probability(dict(b2={'b2/outcome_probs':[1]*6,'b2/early_prob':.5}))
