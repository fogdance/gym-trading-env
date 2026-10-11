"""Prepare frozen M0/M1 commands and CPU admission receipts; never train an Agent."""
from pathlib import Path
import os
import shlex
import subprocess
import sys

import yaml

from .fg_replay import cpu_runtime,check_files,digest,read_manifest,signed,write_json
from .fg_single3 import make_output,source_witness
from .fg_single3_dataset import validate_dataset
from .fg_single3_coverage import require_coverage
from .fg_single3_horizon import require_h


def prepare_pilot(config,dataset,coverage,horizon,output):
    data=validate_dataset(dataset);require_coverage(dataset,coverage);require_h(dataset,coverage,horizon)
    if not data['full_training_fold']:raise ValueError('Pilot requires full TRAIN Gates')
    root=Path(dataset).resolve();inputs=read_manifest(root/'inputs.json');output=make_output(output,config)
    cpu_runtime(config['dreamer_root'])
    import elements
    from dreamerv3.main import _resolve_seed_protocol
    from dreamerv3.fg_single3_offline import validate
    dreamer=Path(config['dreamer_root']).resolve()
    definitions=yaml.safe_load((dreamer/'dreamerv3/configs.yaml').read_text())
    base=elements.Config(definitions['defaults']).update(definitions['fg_entry_v1']).update(definitions['fg_single3_joint_offline'])
    receipts=[]
    codefiles=[Path(__file__),dreamer/'dreamerv3/fg_single3_offline.py',dreamer/'dreamerv3/main.py',
        dreamer/'dreamerv3/fg_entry_guard.py',dreamer/'dreamerv3/configs.yaml',
        dreamer/'embodied/run/offline_pretrain.py',dreamer/'embodied/run/action_mask_warmup.py',
        dreamer/'embodied/run/fg_checkpoint.py',dreamer/'dreamerv3/agent.py']
    provenance=dict(code_files={str(p):digest(p) for p in codefiles})
    snapshots=source_witness(output,provenance)
    tests=[dreamer/'embodied/tests/test_fg_single3_offline.py',dreamer/'dreamerv3/tests/test_fg_entry.py']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
             PYTHONPATH=str(dreamer)+':'+str(Path(__file__).resolve().parents[2]))
    command=[sys.executable,'-m','pytest','-p','no:cacheprovider',*[str(p) for p in tests],'-q','--tb=short']
    result=subprocess.run(command,cwd=dreamer,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode:raise ValueError('CPU pilot wiring tests failed: '+result.stdout)
    print(result.stdout,flush=True)
    for candidate in ('C0','C1','C2','C3'):
        for mode,pool in (('M0',candidate+'_natural'),('M1',candidate+'_natural_oracle')):
            view=inputs['training_views'][candidate]
            capacity=sum(data['traces'][i]['sequence_starts'] for i in data['pool_members'][pool])
            logdir=Path(config['output_root'])/'pilot_runs'/candidate/mode
            flags={'logdir':str(logdir),'fg_entry.manifest':view['path'],
                'env.gymnasium.config_path':view['rows'][0]['env_config'],
                'replay.size':capacity,'run.replay_dir':str(root/data['views'][pool]),
                'fg_single3_offline.dataset':str(root),'fg_single3_offline.coverage':str(Path(coverage).resolve()),
                'fg_single3_offline.horizon':str(Path(horizon).resolve()),'fg_single3_offline.candidate':candidate,
                'fg_single3_offline.pool':pool,'fg_single3_offline.authorized':False}
            resolved=_resolve_seed_protocol(base.update(flags))
            checked=validate(resolved,allow_unapproved=True,persist=False)
            flags['fg_single3_offline.identity_hash']=checked['identity_hash']
            argv=['--configs','defaults','fg_entry_v1','fg_single3_joint_offline']
            for key,value in flags.items():argv += ['--'+key,str(value)]
            directory=output/candidate/mode;directory.mkdir(parents=True)
            # Always blocked until the user reviews CPU Reward/Replay/Oracle.
            script='#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(dreamer))+'\n'
            script+='export PYTHONPATH='+shlex.quote(str(Path(__file__).resolve().parents[2]))+'"${PYTHONPATH:+:$PYTHONPATH}"\n'
            script+=shlex.join([sys.executable,'-m','dreamerv3.main',*argv])+'\n'
            (directory/'run.sh').write_text(script)
            (directory/'config.yaml').write_text(yaml.safe_dump(dict(resolved),sort_keys=False))
            write_json(directory/'preflight.json',signed(dict(status='CPU_PREFLIGHT_PASS',model_status='NOT_RUN',
                candidate=candidate,mode=mode,capacity=capacity,identity=checked,model_seed=0,replay_seed=202,
                gradient_updates_budget=2000,gradient_updates_executed=0,argv=argv)))
            receipts.append(dict(candidate=candidate,mode=mode,pool=pool,capacity=capacity,
                                 directory=str(directory),identity_hash=checked['identity_hash']))
    lines=['# FG Single3 模型先导比较（准备完成，尚未运行）','',
        '模型状态：NOT_RUN。尚未选定训练Reward，未初始化神经Agent、未启动GPU、未执行梯度更新；择时能力NOT_ESTABLISHED。','',
        '四候选各准备M0自然池、M1自然+P4池，供用户验收后选一组Reward运行。两组网络结构沿用fg_entry_v1/size50m；horizon333、imag15、B16/64+context1；model seed0、Replay seed202；每组2000次梯度更新（2,048,000训练时间位置），完整Uniform容量。','',
        '模式是完整Agent联合离线训练，复用现有Mask warmup并检查ADD合法类post/prior召回及真实正类支持数。不是WM-only；不新增Actor BC Loss。Gate未通过时沿用原有Actor/Critic loss门控。','',
        'CPU FakeAgent测试证明首次训练关闭Actor，满足Gate后开启，Gate/训练carry/预取队列/Replay RNG复用原FullCheckpoint保存和恢复，不重复更新；不同身份恢复拒绝。真实GPU权重及训练结果尚未验证。','',
        '启动脚本默认authorized=False，会在Agent初始化前拒绝启动；只有用户确认Reward、完整Replay覆盖和Oracle结果后才能设置True。旧6手Replay、Checkpoint、不同Reward和不同M池的Checkpoint身份均不能接续。','','每个候选配置及预检在此目录；不存在已训练模型结果。']
    (output/'pilot_comparison.md').write_text('\n'.join(lines)+'\n')
    frozen={str(Path(p).resolve()/'manifest.json'):digest(Path(p)/'manifest.json') for p in (dataset,coverage,horizon)}
    check_files({**frozen,**provenance['code_files']})
    receipt=signed(dict(schema='FG_SINGLE3_CPU_V1',stage='PILOT_PREPARED',status='PASS',model_status='NOT_RUN',
        selected_training_reward=None,gradient_updates=0,configurations=receipts,provenance=provenance,
        source_snapshots=snapshots,frozen_inputs=frozen,tests=dict(command=command,returncode=0,output=result.stdout),
        files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}))
    write_json(output/'manifest.pending.json',receipt)
    try:
        check_files({**frozen,**provenance['code_files']});check_files({str(output/p):h for p,h in receipt['files'].items()})
        (output/'manifest.pending.json').replace(output/'manifest.json')
    except BaseException:
        (output/'pilot_comparison.md').replace(output/'pilot_comparison.failed.md')
        raise
    return receipt
