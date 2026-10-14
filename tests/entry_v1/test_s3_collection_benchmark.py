"""A malformed pool must not produce a misleading performance identity."""
import json
import sys

import pytest
import yaml

from gym_trading_env.envs.entry_contract import fingerprint
from tools.benchmark_s3_collection import main


def test_duplicate_manifest_contract_cannot_stand_in_for_requested_contract(tmp_path, monkeypatch):
    manifest = dict(product='FG', files=[dict(contract='CZCE.FG2701',
        role='train', eligible_for_training=True, env_config='must_not_be_read')] * 2)
    manifest['content_hash'] = fingerprint(manifest)
    path = tmp_path/'pool.json'; path.write_text(json.dumps(manifest))
    selector = tmp_path/'selector.yaml'
    selector.write_text(yaml.safe_dump(dict(manifest_path=str(path))))
    output = tmp_path/'output'
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--selector', str(selector),
        '--baseline-ref', 'HEAD', '--contracts', 'CZCE.FG2701', 'CZCE.FG2702',
        '--output', str(output)])
    with pytest.raises(ValueError, match='distinct eligible TRAIN'):
        main()
    assert not output.exists()
