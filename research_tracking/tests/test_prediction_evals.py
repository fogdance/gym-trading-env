"""A late independent prediction report must not rewrite Actor-only history."""
import json
from pathlib import Path
import tempfile
import unittest

from research_tracking.tests.test_tracker import Fixture
from research_tracking.io import atomic_json, digest, fingerprint, TrackingError
from research_tracking.importer import Importer
from research_tracking.prediction_evals import parse_prediction_eval
from research_tracking.state import Store


def signed(value):
    return dict(value,content_hash=fingerprint(value))


class PredictionTrackingTests(unittest.TestCase):
    def test_late_report_idempotency_identity_and_offline_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            fixture = Fixture(base)
            actor_path = fixture.evaluation()
            source = fixture.source()
            actor = json.loads(actor_path.read_text())
            path = base/'independent_prediction.json'
            receipt = dict(schema='S3_DEV_PREDICTION_EVAL_V1',status='DEV_PREDICTION_EVALUATION_COMPLETE',
                role='DEV_EVALUATION_ONLY',fold='DEV',eligible_for_training=False,
                oracle_files_read_in_policy=False,policy_observations_modified=False,future_data_read_posthoc=True,
                account_replay_exact=True,clock_protocol=source.clock,source_run_id=source.run_id,
                source_eval_dir=str(actor_path.parent),source_eval_receipt_sha256=digest(actor_path),
                source_eval_receipt_content_hash=actor['content_hash'],evaluation_kind='tutor_scoring',
                checkpoint_manifest_sha256=actor['checkpoint_manifest_sha256'],certified_accounts=2,
                outcome=dict(event_samples=2,event_nll=.4,event_brier=.2,event_constant_prior_nll=.69,
                    classes={'SL':dict(support=0,predictions=0,recall=None,precision=None)}),decision={},
                # Raw account/labels are not part of the transported summary.
                private_account_state={'secret_account_id':'DO_NOT_UPLOAD'},future_candidates=['DO_NOT_UPLOAD'])
            with Store(source,base/'tracking') as store:
                importer = Importer(store,prediction_reports=())
                self.assertEqual(importer.evaluations(),1)
                original = (store.root/store.state['batches'][0]['file']).read_bytes()
                atomic_json(path,signed(receipt))
                importer = Importer(store,prediction_reports=[path])
                self.assertEqual(importer.evaluations(),1)
                self.assertEqual(importer.evaluations(),0)
                self.assertEqual((store.root/store.state['batches'][0]['file']).read_bytes(),original)
                batch = store.state['batches'][-1]
                artifact = (store.root/batch['artifacts'][-1]).read_text()
                self.assertNotIn('DO_NOT_UPLOAD',artifact)
                self.assertEqual(batch['config']['parent_actor_eval_run_id'],store.state['batches'][0]['run_id'])
                from research_tracking.writer import Writer
                Writer('offline').export(store)
                self.assertTrue(all(b['exported'] for b in store.state['batches']))
            receipt['checkpoint_manifest_sha256'] = '0'*64
            atomic_json(path,signed(receipt))
            with self.assertRaisesRegex(TrackingError,'Checkpoint identities'):
                parse_prediction_eval(source,path)


if __name__ == '__main__':unittest.main()
