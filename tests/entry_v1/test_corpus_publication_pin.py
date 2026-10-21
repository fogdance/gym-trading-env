"""Publication authenticates completed G3, independently of publisher HEAD."""
import copy

import pytest

from gym_trading_env.research.fg_replay import digest,signed,write_json
from gym_trading_env.research import fg_s3_corpus_production as production


def fixture(tmp_path, *, modern):
    pin=tmp_path/'historical_pin.json'
    repositories=dict(gym=dict(sha='a'*40,dirty=False),dreamer=dict(sha='b'*40,dirty=False))
    write_json(pin,signed(dict(protocol='FG_S3_E_QUALITY_PRODUCER_PIN_V1',repositories=repositories)))
    sha=digest(pin)
    recorded=dict(gym=str(tmp_path/'removed_verifier/gym-trading-env'),
                  pin=str(tmp_path/'removed_verifier/missing_pin.json'),pin_sha256=sha)
    current=dict(gym=str(tmp_path/'current/gym-trading-env'),pin=str(tmp_path/'current_pin.json'),pin_sha256='c'*64)
    config=dict(producer=current)
    batch=dict(id='SYNTHETIC',adopt=dict(corpus='unused',origin_pin=dict(pin=str(pin),pin_sha256=sha)))
    directory=tmp_path/'native_g3/0001'
    request=dict(config=dict(producer=current),batch=dict(adopt=dict(producer=recorded)))
    if modern:
        request['config']['producer']=recorded
        write_json(directory/'command.json',dict(execution_lock=dict(producer=recorded)))
    write_json(directory/'request.json',request)
    audit={role:dict(head=item['sha'],runtime_diff_exit_code=0,untracked_runtime=[])
           for role,item in repositories.items()}
    audit['operator_producer_pin_sha256']=sha
    native=dict(cwd=recorded['gym'],producer_audit=audit)
    return directory,config,batch,native,pin


@pytest.mark.parametrize('modern',[False,True])
def test_historical_completed_g3_does_not_require_same_publisher_or_old_checkout(tmp_path,modern):
    directory,config,batch,native,pin=fixture(tmp_path,modern=modern)
    result=production.completed_native_pin(native,directory,config,batch,None)
    assert result==dict(pin=str(pin),pin_sha256=digest(pin))


@pytest.mark.parametrize('attack',['head','dirty','pin','cwd'])
def test_publication_rejects_changed_g3_audit_binding(tmp_path,attack):
    directory,config,batch,native,pin=fixture(tmp_path,modern=True)
    changed=copy.deepcopy(native)
    if attack=='head':changed['producer_audit']['gym']['head']='d'*40
    elif attack=='dirty':changed['producer_audit']['gym']['runtime_diff_exit_code']=1
    elif attack=='pin':changed['producer_audit']['operator_producer_pin_sha256']='e'*64
    else:changed['cwd']=str(tmp_path/'foreign_checkout')
    with pytest.raises(ValueError,match='Completed G3'):
        production.completed_native_pin(changed,directory,config,batch,None)
