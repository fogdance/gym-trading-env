"""Reviewed read-only Corpus ABI registry, independent of consumer Git identity.

This is a compatibility check, never source authentication or training authority.
Historical file/content hashes remain untouched. Hashes below freeze semantic
descriptors, not executable source; changing a descriptor requires ABI review.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from gym_trading_env.envs.entry_contract import fingerprint


COURSE_ABI_VERSION = '3.1.0'
SUPPORTED_VERSIONS = (COURSE_ABI_VERSION,)
POLICIES = ('P0', 'P1', 'P2_low', 'P2_mid', 'P3', 'P4')
_OBSERVATION = 'a7db9f3d49043203aaf45ea41577a2438dddfd08d8904ef16db936ac6a77f317'
_LABELS = '9a7ad4aae4e26fc6441420af43f00341f1b1c45f50f4f8d45bf571eb81c51e52'
_COURSE = 'e0eaa1b93c8a49ec7a4a8a408db81903b615cd7ae7ba50c9ec804283fb5c6e86'
_E_SEMANTIC = 'b0faae7470878f30a972797145a431350e55fd2650c2b998be1c94e0e802943a'
# Reviewed phase2 and phase3 E profiles. Their implementation-source inventories
# differ; their E mathematical descriptor and native arrays were checked equal.
# This list is deliberately exact: arbitrary V3/profile hashes are not trusted.
_E_PROFILES = frozenset((
    '7d82865eb7981d9aeaab1cd4991c5c38521e027c62ff18a6ecc0efeb44636ee6',
    '365c5dd68a561c242e2e8e58455ee143ed8d347c37b10413a6b2467d6f5a7dd7',
))
_E_PROVENANCE = frozenset(('schema_sha256', 'macd_code_sha256',
                         'legacy_sources_sha256', 'implementation_sources_sha256'))


@dataclass(frozen=True)
class CorpusCompatibility:
    course_abi_version: str
    semantic_contract_hash: str
    resolution: str
    e_profile_schema_sha256: str

    def as_dict(self):
        return asdict(self)

    @staticmethod
    def validate_e_profile(claimed, current):
        semantic = {k: v for k, v in current.items() if k not in _E_PROVENANCE}
        if claimed not in _E_PROFILES or fingerprint(semantic) != _E_SEMANTIC:
            raise ValueError('CORPUS_ABI_UNSUPPORTED_E_PROFILE')

    @classmethod
    def resolve(cls, manifest, course, teacher, model):
        """Resolve only a complete reviewed contract, after inventory checks."""
        version = manifest.get('course_abi_version', COURSE_ABI_VERSION)
        if type(version) is not str or version not in SUPPORTED_VERSIONS:
            raise ValueError('CORPUS_ABI_UNSUPPORTED_VERSION')
        if (manifest.get('protocol') != 'FG_S3_E_QUALITY_CORPUS_V3'
                or manifest.get('behaviors') != list(POLICIES)):
            raise ValueError('CORPUS_ABI_UNSUPPORTED_CORPUS')
        observation = manifest['observation']
        semantic = {k: course[k] for k in ('observation', 'trading', 'teacher', 'reward')}
        if (fingerprint(observation) != _OBSERVATION
                or model.get('protocol') != 'FG_S3_E_MODEL_CONTRACT_V2'
                or model.get('observation') != observation or fingerprint(semantic) != _COURSE):
            raise ValueError('CORPUS_ABI_SEMANTIC_CONTRACT_MISMATCH')
        if (teacher.get('protocol') != 'FG_S3_E_QUALITY_TEACHER_V3'
                or teacher.get('value_witness_encoding') != 'FG_S3_EXACT_RATIONAL_HEX_V1'
                or teacher.get('objective') != 'discounted_native_float32_Quality_TP_feedback'
                or teacher.get('gamma_rational') != [332, 333]
                or teacher.get('reward_schema') != course['reward']['schema']
                or teacher.get('fold') != 'TRAIN'):
            raise ValueError('CORPUS_ABI_TEACHER_CONTRACT_MISMATCH')
        from .fg_s3_e_observation import observation_contract
        from .fg_s3_quality_catalog_v2 import FIELDS
        from e_data_v1.schema import EDataConfig
        from e_data_v1.reference import load_legacy
        labels = {k: dict(dtype=str(dtype), shape=list(shape))
                  for k, (dtype, shape) in FIELDS.items()}
        if fingerprint(labels) != _LABELS or observation_contract() != observation:
            raise ValueError('CORPUS_ABI_CONSUMER_CONTRACT_MISMATCH')
        profile = course['data_identity']['e_profile_schema_sha256']
        cls.validate_e_profile(profile, EDataConfig().schema(load_legacy()[1]))
        contract_hash = fingerprint(dict(observation=_OBSERVATION, labels=_LABELS,
            course=_COURSE, e_semantic=_E_SEMANTIC,
            exact_witness='FG_S3_EXACT_RATIONAL_HEX_V1',
            join='NATIVE_ACCOUNT_CONTIGUOUS_POST_STEP_OWNER_V1'))
        return cls(version, contract_hash,
                   'EXPLICIT' if 'course_abi_version' in manifest else 'REVIEWED_V3_MAPPING', profile)
