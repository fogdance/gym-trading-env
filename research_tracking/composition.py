"""Load the single Dreamer pure contract without a JAX/trainer dependency."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from .io import TrackingError


def validate(settings, *documents):
    # Historical records have no composition and keep their original identity.
    if not (settings.get('schema') == 'FG_S3_V2_COMPOSITION_CONFIG_V1' or
            any('resolved_composition' in d or 'composition' in d for d in (settings, *documents))):
        return None
    path = Path(__file__).resolve().parents[2]/'dreamerv3/dreamerv3/fg_s3_composition.py'
    if not path.is_file():
        raise TrackingError('Matching Dreamer composition contract is unavailable')
    spec = spec_from_file_location('_fg_s3_composition', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.validate_records(settings, *documents)
    except ValueError as exc:
        raise TrackingError(str(exc)) from exc
