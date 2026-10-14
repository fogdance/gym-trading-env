"""GPU validation in private path aliases, never in the host source Run."""
import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from .io import ArchiveError, atomic_json, digest, independent, now, read_json


def validate(root, mode, python):
    if mode not in ('eval', 'resume-smoke'):
        raise ArchiveError('Unknown validation mode')
    if not shutil.which('bwrap'):
        raise ArchiveError('Bubblewrap unavailable; original Run execution is forbidden')
    manifest = read_json(root/'archive_manifest.json', limit=128*1024*1024)
    if not manifest['checkpoint_preserved']:
        raise ArchiveError('Checkpoint bytes are required for model validation')
    plan = read_json(root/'reports/inspect.json', limit=128*1024*1024)
    work = root.parent/'.validation'/(root.name+'_'+mode+'_'+uuid.uuid4().hex[:10])
    work.mkdir(parents=True, exist_ok=False)
    run = work/'run'; run.mkdir()
    for old, new in [('run_identity.json','run_identity.json'),('resolved_config.yaml','config.yaml'),
                     ('source_admission.json','source_admission.json'),('source_transfer.json','source_transfer.json')]:
        shutil.copy2(root/'provenance'/old, run/new)
    cp = Path(manifest['source']['checkpoint_dir']).name
    shutil.copytree(root/'checkpoint'/cp, run/'ckpt'/cp)
    (run/'ckpt/latest').write_text(cp+'\n')
    if (root/'dependencies/run').exists():
        shutil.copytree(root/'dependencies/run', run, dirs_exist_ok=True)
    roots = plan['code_audit']['repositories']
    for name in roots:
        shutil.copytree(root/'provenance/source_bundle'/name, work/'source'/name)
    prep_alias = manifest['source']['prepared_dir']
    gym_alias = roots['gym-trading-env']['path']
    prep_mount = work/'source/gym-trading-env'/Path(prep_alias).relative_to(gym_alias)
    prep_mount.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root/'provenance/prepared', work/'prepared')
    eval_name = Path(manifest['source']['eval_dir']).name
    shutil.copytree(root/'evaluations'/eval_name, work/'original_evaluation')
    spec = dict(mode=mode, prepared=prep_alias, run_dir=manifest['source']['original_run_dir'],
        steps=plan['original_receipt']['env_steps'], boundaries=plan['boundaries'],
        checkpoint_manifest_sha256=manifest['source']['checkpoint_manifest_sha256'],
        source_identity=plan['identity'], oracle_root=plan['isolation_roots'].get('oracle_receipt'),
        original_eval=str(work/'original_evaluation'), work=str(work))
    env_code = 'import sys,platform,json,importlib.metadata as m; names=json.loads(sys.argv[1]); print(json.dumps({"python":platform.python_version(),"platform":platform.platform(),"packages":{n:m.version(n) for n in names}}))'
    environment = json.loads(subprocess.check_output([str(Path(python).resolve()),'-B','-c',env_code,
        json.dumps(list(plan['identity'].get('packages',{})))],text=True))
    if environment['packages'] != plan['identity'].get('packages',{}):
        raise ArchiveError('Validation environment package versions differ from frozen Run')
    atomic_json(work/'execution_environment.json',environment)
    atomic_json(work/'spec.json', spec)
    shutil.copy2(Path(__file__).with_name('validation_worker.py'), work/'validation_worker.py')
    command = ['bwrap','--ro-bind','/','/','--dev','/dev','--proc','/proc','--unshare-net','--unshare-pid',
               '--tmpfs','/tmp']
    # Entire relevant data/work namespaces contain only the captured dependency closure.
    # This makes an accidental fallback to the original writable data impossible.
    fs = root/'dependencies/filesystem'
    exceptions = [Path(spec['run_dir']),work,*[Path(r['path']) for r in roots.values()]]
    def mount_tree(target,alias):
        if any(p==alias or p.is_relative_to(alias) for p in exceptions):
            command.extend(['--dir',str(alias)])
            for child in target.iterdir():
                mount_tree(child,alias/child.name)
        else:
            command.extend(['--ro-bind',str(target),str(alias)])
    for alias in ('/data/logdir','/home/v/Documents/work'):
        command += ['--tmpfs',alias]
        target = fs/alias.lstrip('/')
        if target.is_dir():
            for child in target.iterdir():
                mount_tree(child,Path(alias)/child.name)
    for name, repo in roots.items():
        command += ['--ro-bind',str(work/'source'/name),repo['path']]
    command += ['--ro-bind',str(work/'prepared'),prep_alias,
                '--bind',str(work),str(work), '--bind',str(run),spec['run_dir']]
    nodes = glob.glob('/dev/nvidia*')
    if not nodes:
        raise ArchiveError('No NVIDIA device nodes visible; no CPU fallback for model validation')
    for node in nodes:
        command += ['--dev-bind',node,node]
    command += ['--chdir',str(work),'--setenv','JAX_PLATFORMS','cuda',
        '--setenv','XLA_PYTHON_CLIENT_PREALLOCATE','false','--setenv','PYTHONDONTWRITEBYTECODE','1',
        '--setenv','MPLCONFIGDIR',str(work/'mpl'),
        '--setenv','OPENBLAS_NUM_THREADS','1','--setenv','OMP_NUM_THREADS','1',
        '--setenv','PYTHONPATH',roots['dreamerv3']['path']+':'+gym_alias+'/src',
        '--unsetenv','LD_LIBRARY_PATH','--',str(Path(python).resolve()),str(work/'validation_worker.py'),str(work/'spec.json')]
    atomic_json(work/'isolation_command.json', dict(argv=command))
    with (work/'stdout.log').open('w') as out:
        process = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
    result_path = work/'result.json'
    result = read_json(result_path) if result_path.exists() else dict(status='VALIDATION_FAILED',
        error='Isolated process exited without a validation receipt')
    result.update(exit_code=process.returncode, validation_dir=str(work), timestamp=now(),
                  worker_sha256=digest(work/'validation_worker.py'), log_sha256=digest(work/'stdout.log'),
                  execution_environment=environment)
    if process.returncode:
        result['status'] = 'VALIDATION_FAILED'
    return result
