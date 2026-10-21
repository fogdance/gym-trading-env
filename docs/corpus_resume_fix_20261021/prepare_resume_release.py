"""Freeze the repaired executor and migrate control metadata, never produce."""
import json
from pathlib import Path
import subprocess
import sys

import yaml

from gym_trading_env.research.fg_replay import digest, signed, write_json
from gym_trading_env.research.fg_s3_corpus_production import (
    authenticate, check_environment, load_config, prepare_resume)
from gym_trading_env.research.fg_s3_e_quality_corpus_cli import _code_lock


def main():
    repo=Path.cwd();release=repo.parent/'release'
    original=release/'corpus_production.before_exit_fix.yaml'
    output=release/'corpus_production.yaml'
    if not original.exists():
        with original.open('xb') as stream:
            stream.write(output.read_bytes())
    config=yaml.safe_load(original.read_text())
    heads={}
    for role,root in [('gym',repo),('dreamer',repo.parent/'dreamerv3')]:
        if subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip():
            raise ValueError('Commit the repair before freezing the executor: '+role)
        heads[role]=dict(sha=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),dirty=False)
    pin=release/('PRODUCER_PIN.exit_fix_'+heads['gym']['sha'][:12]+'.json')
    value=signed(dict(protocol='FG_S3_E_QUALITY_PRODUCER_PIN_V1',repositories=heads))
    if pin.exists():
        if json.loads(pin.read_text())!=value:raise ValueError('Existing executor pin differs')
    else:
        write_json(pin,value)
    config['producer']=dict(gym=str(repo),pin=str(pin),pin_sha256=digest(pin))
    def asset(item,kind):
        source=Path(item['path'])
        target=release/'assets'/kind/(item['sha256'][:12]+'_'+source.name)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            if digest(source)!=item['sha256']:raise ValueError('Release asset changed: '+str(source))
            with target.open('xb') as stream:stream.write(source.read_bytes())
        if digest(target)!=item['sha256']:raise ValueError('Release asset copy differs')
        return dict(path=str(target),sha256=item['sha256'])
    for key in ('runtime_probe','coverage_program'):
        config[key]=asset(config[key],'programs')
    for batch in config['batches']:
        if 'plan' in batch:batch['plan']=asset(batch['plan'],'plans')
        if 'adopt' in batch:
            adoption=batch['adopt']
            origin=adoption.get('origin_pin',adoption.get('producer'))
            preserved=asset(dict(path=origin['pin'],sha256=origin['pin_sha256']),'pins')
            adoption.pop('producer',None)
            adoption['origin_pin']=dict(pin=preserved['path'],pin_sha256=preserved['sha256'])
    temporary=output.with_suffix('.resume.tmp')
    temporary.write_text(yaml.safe_dump(config,sort_keys=False))
    normalized,recipe=load_config(temporary)
    check_environment(normalized)
    audit=authenticate(normalized['producer'])
    temporary.replace(output)
    result=prepare_resume(output,original,apply='--apply' in sys.argv)
    evidence=dict(result,release_repositories=heads,current_code=_code_lock(),producer_audit=audit,
        launch_command='bash '+str(release/'RUN_CORPUS.sh')+' resume')
    write_json(release/'RESUME_READY.json',evidence)
    print(json.dumps(dict(status=result['status'],repositories=heads,recipe_hash=recipe,
        preserved_stages=len(result.get('completed_stages_preserved',[])),
        recoverable_generations=result.get('recoverable_generations',[]),
        production_started=False,command=evidence['launch_command']),ensure_ascii=False))


if __name__=='__main__':
    main()
