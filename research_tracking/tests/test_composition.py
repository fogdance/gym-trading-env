"""Synthetic tracking metadata; no cloud/GPU or production training claims."""
import json
from pathlib import Path
import tempfile
import unittest

from research_tracking.tests.test_tracker import Fixture, signed
from research_tracking.io import atomic_json, digest, read_json, TrackingError
from research_tracking.identity import SourceRun
from research_tracking.composition import validate
from research_tracking.evals import parse_eval


class CompositionTrackingTests(unittest.TestCase):
    def test_new_composition_is_verified_displayed_and_eval_linked(self):
        from importlib.util import spec_from_file_location,module_from_spec
        path=Path(__file__).resolve().parents[3]/'dreamerv3/dreamerv3/fg_s3_composition.py'
        spec=spec_from_file_location('_test_composition',path);module=module_from_spec(spec);spec.loader.exec_module(module)
        resolved=module.resolve_composition({'mode':'tutor'})
        with tempfile.TemporaryDirectory() as temp:
            f=Fixture(Path(temp))
            settings=read_json(f.prepared/'settings.json')['settings']
            settings.update(schema=module.CONFIG_SCHEMA,resolved_composition=resolved)
            atomic_json(f.prepared/'settings.json',signed({'settings':settings}))
            model=read_json(f.prepared/'model_config.json');model.pop('content_hash');model['resolved_composition']=resolved
            atomic_json(f.prepared/'model_config.json',signed(model))
            atomic_json(f.prepared/'source_admission.json',signed({'resolved_composition':resolved,
                'target_model_identity':{'resolved_composition':resolved}}))
            prep=read_json(f.prepared/'manifest.json');prep.pop('content_hash');prep['resolved_composition']=resolved
            prep['files']={p.name:digest(p) for p in f.prepared.iterdir() if p.name!='manifest.json'}
            atomic_json(f.prepared/'manifest.json',signed(prep))
            f.identity.update(resolved_composition=resolved,preparation_sha256=digest(f.prepared/'manifest.json'))
            atomic_json(f.root/'run_identity.json',signed({'identity':f.identity}))
            f.checkpoint(0);f.checkpoint(100)
            source=SourceRun(f.root,f.prepared)
            self.assertEqual(source.wandb_configuration()['s3/resolved_composition'],resolved)
            self.assertEqual(source.configuration()['experiment']['resolved_composition'],resolved)
            evaluation=f.evaluation();receipt=read_json(evaluation);receipt.pop('content_hash')
            receipt['resolved_composition']=resolved;atomic_json(evaluation,signed(receipt))
            self.assertEqual(parse_eval(source,evaluation)['resolved_composition'],resolved)
            receipt.pop('resolved_composition');atomic_json(evaluation,signed(receipt))
            with self.assertRaisesRegex(TrackingError,'COMPOSITION_IDENTITY_MISSING'):parse_eval(source,evaluation)
            with self.assertRaisesRegex(TrackingError,'COMPOSITION_IDENTITY_MISMATCH'):
                validate(settings,{'resolved_composition':module.resolve_composition({'mode':'b2'})})
