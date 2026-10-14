"""Load the shared pure evaluator without importing the training packages."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'src/gym_trading_env/research/fg_s3_eval_metrics.py'
spec = spec_from_file_location('_fg_s3_pure_eval_metrics', path)
module = module_from_spec(spec)
spec.loader.exec_module(module)
evaluation_metrics = module.evaluation_metrics
unique_events = module.unique_events
