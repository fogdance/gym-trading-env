"""Shared public evaluation contract, including the real read-only 800K data."""
import copy
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch,MagicMock
import os

from research_tracking.eval_schema import evaluation_metrics
from research_tracking.sdk_worker import resumed_metadata
from research_tracking.auth import readiness


class EvaluationV2Tests(unittest.TestCase):
    def test_direct_readiness_forces_internal_proxy_without_changing_caller_env(self):
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = json.dumps(
            dict(data=dict(viewer=dict(username='fixture'),project=dict(access='PRIVATE')))).encode()
        with patch.dict(os.environ,{'NO_PROXY':'*'}),patch('research_tracking.auth.credential',return_value='fixture'),\
                patch('research_tracking.auth.urllib.request.build_opener',return_value=opener):
            before = dict(os.environ)
            self.assertEqual(readiness('fixture')['status'],'PRIVATE_PROJECT_AUTHENTICATED')
            request = opener.open.call_args.args[0]
            self.assertEqual(request.host,'127.0.0.1:10808')
            self.assertEqual(request._tunnel_host,'api.wandb.ai')
            self.assertEqual(dict(os.environ),before)

    def test_resume_keeps_existing_metadata(self):
        spec = dict(tags=['train'], display_name='default name')
        cloud = SimpleNamespace(tags=['manual-research-tag'], name='User title')
        self.assertEqual(resumed_metadata(spec, cloud), ('User title', ['manual-research-tag','train']))
        self.assertEqual(resumed_metadata(spec, None), ('default name',['train']))

    def test_market_windows_duplicates_unknown_and_missing_logs(self):
        events = [dict(entry_id=str(b), reason='FIRST_ADVERSE10', effective_bar=b,
                       time='overnight/lunch timestamps are not duration inputs') for b in (5, 6, 15, 16)]
        state = dict(reward_events=events+[copy.deepcopy(events[0])], attributions=[])
        receipt = dict(clock_protocol='S3_MARKET_MINUTE', early_mae10=4)
        m = evaluation_metrics(state, receipt)
        self.assertEqual(tuple(m[k] for k in ('adverse10_first5','adverse10_6_15',
            'adverse10_after15','adverse10_any_hold','adverse10_unclassified')), (1,2,1,4,0))
        self.assertIsNone(m['tp_market_minutes_p50'])
        self.assertEqual(m['closed_trade_net_pnl'], 0)
        state['reward_events'][1].pop('effective_bar')
        m = evaluation_metrics(state, receipt)
        self.assertEqual(m['adverse10_unclassified'],1)
        self.assertFalse(m['adverse10_windows_complete'])
        self.assertIsNone(evaluation_metrics(None, receipt)['adverse10_first5'])
        self.assertIsNone(evaluation_metrics({}, {})['adverse10_any_hold'])
        self.assertEqual(evaluation_metrics(dict(reward_events=[], attributions=[]), receipt={})['adverse10_any_hold'],0)
        state['reward_events'][-1]['effective_bar'] = 2
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            evaluation_metrics(state, receipt)

    def test_actual_tutor_800k_is_derived_without_writing_source(self):
        root = Path('/data/logdir/fg_single3/s3_v2/TUTOR_CD_V0_MARKET_MINUTE_FORMAL_1M_v1'
                    '/evaluations/actor_only_16be8b0985')
        if not root.exists():
            self.skipTest('Historical local evidence not installed')
        receipt = json.loads((root/'receipt.json').read_text())
        state = json.loads((root/'account_state.json').read_text())
        m = evaluation_metrics(state, receipt)
        self.assertEqual(receipt['entries'], 7)
        self.assertEqual((m['tp_count'],m['sl30']), (5,2))
        self.assertEqual(m['tp_market_minutes_p50'],74)
        self.assertEqual(m['closed_trade_net_pnl'],-2226)
        self.assertAlmostEqual(receipt['reward'],-3.805870048525721)
        self.assertEqual(tuple(m[k] for k in ('adverse10_first5','adverse10_6_15',
            'adverse10_after15','adverse10_any_hold')), (0,1,3,4))


if __name__ == '__main__':
    unittest.main()
