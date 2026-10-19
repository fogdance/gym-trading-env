"""Thin Gym execution boundary for E+Quality online practice."""
from __future__ import annotations

from dreamerv3.fg_s3_e_quality_online import PROTOCOL, validate_execution, execution_contract


def preflight(spec, *, allow_engineering=False):
    from embodied.run.fg_s3_e_quality_online import inspect_f3
    from .fg_replay import signed
    from gym_trading_env.envs.entry_contract import fingerprint
    s = validate_execution(spec)
    proof, _ = inspect_f3(s, allow_engineering=allow_engineering)
    from .fg_s3_e_quality_supervision import SupervisionSource, QualitySupervisionAdmission
    original = proof['identity']['preparation']
    source = QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(
        original['settings']['corpus_root'], stage='ACTOR_SUPERVISED'))
    if (source.source.course_identity_hash != original['course']
            or source.receipt['content_identity'] != original['source']):
        raise ValueError('ONLINE_F3_CORPUS_OR_COURSE_CHANGED')
    return signed(dict(protocol=PROTOCOL, spec=s, execution_contract=execution_contract(s['execution_mode']),
        source=proof, execution_identity=fingerprint(dict(spec=s, source_manifest=proof['manifest_sha256'])),
        status='DEV_ONLY' if proof['engineering_only'] else 'SOURCE_VERIFIED_NOT_AUTHORIZED',
        training_admitted=False, research_admitted=False))
