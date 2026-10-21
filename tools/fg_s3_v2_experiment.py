#!/usr/bin/env python3
"""Prepare, train, evaluate S3-V2 and simulate frozen Actors on live DB bars."""
import argparse
import json
import os
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
# The CUDA-enabled environment ships a coherent set of NVIDIA libraries.
# A host CUDA LD_LIBRARY_PATH overrides some of them and failed cuBLAS model
# initialization in the real GPU check. Restart only this process with its
# normal package library search; never mutate the shared environment/system.
if os.environ.get('LD_LIBRARY_PATH'):
    clean = dict(os.environ)
    clean.pop('LD_LIBRARY_PATH')
    clean['FG_S3_V2_CUDA_ENV_CLEANED'] = '1'
    os.execve(sys.executable, [sys.executable, *sys.argv], clean)
sys.path.insert(0, str(PROJECT))  # Existing E mathematical/audit package.
sys.path.insert(0, str(PROJECT/'src'))
sys.path.insert(0, str(PROJECT.parent/'dreamerv3'))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fg_s3_v2_mpl')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'start', 'resume', 'evaluate', 'report', 'gpu_check',
                                     'admit-e-prepare', 'admit-e-start', 'admit-e-resume',
                                     'f1-prepare', 'f1-start', 'f1-resume', 'f1-probe', 'f1-read-check',
                                     'f2-prepare', 'f2-start', 'f2-resume', 'f2-probe', 'f2-read-check',
                                     'f3-prepare', 'f3-start', 'f3-resume', 'f3-probe', 'f3-read-check', 'f3-evaluate',
                                     'online-prepare','online-start','online-resume','online-probe','online-evaluate','online-report','online-rl-review-packet',
                                     'exam-plan','exam-run','exam-resume','exam-verify','exam-compare','exam-report',
                                     'score-predictions', 'simulate-online', 'monitor-online',
                                     'describe-composition','evaluate-source','evaluate-reference','compare-predictions'])
    p.add_argument('--config')
    p.add_argument('--prepared')
    p.add_argument('--source-checkpoint',help='Completed historical source for read-only reference evaluation')
    p.add_argument('--eval-device',choices=['cpu','gpu'],default='cpu')
    p.add_argument('--run-dir')
    p.add_argument('--stop-at', type=int)
    p.add_argument('--verify-resume-only', action='store_true',
                   help='Internal bounded E cold-resume next-update probe; publishes no checkpoint')
    p.add_argument('--eval-steps', type=int, default=1000)
    p.add_argument('--holdout-contract',help='Whitelisted TRAIN contract excluded from quality-head training')
    p.add_argument('--diagnostic-components',help='Read-only HEADS_ONLY evaluation fork: outcome,score,q in that order')
    p.add_argument('--diagnostic-q-admission',help='Verified Q source copied into a private evaluation model; also use for its baseline')
    p.add_argument('--time-fold',help='Pretraining-frozen official-day window for compare-predictions')
    p.add_argument('--eval-dir', help='Existing sealed Actor-only evaluation for CPU post-processing')
    p.add_argument('--reference-dir',help='Posthoc reference predictions on the same sealed causal tape')
    p.add_argument('--output-dir', help='New evaluation/scoring directory or online monitor observations')
    p.add_argument('--runner-pid', type=int, help='Online runner PID for optional process health checks')
    p.add_argument('--interval', type=float, default=30, help='Online monitor sampling interval in seconds')
    p.add_argument('--once', action='store_true', help='Sample the online monitor once')
    p.add_argument('--json', action='store_true', help='Print online console events as JSON')
    p.add_argument('--until', help='Optional online monitor end time (ISO); default is continuous')
    p.add_argument('--check-only', action='store_true', help='Online DB/model/RPC preflight; emits no orders')
    p.add_argument('--resume-online', action='store_true', help='Restore the same online account and policy carry')
    p.add_argument('--authorize-training', action='store_true',
                   help='Explicit authorization for formal 1M and any new Quality GPU start/resume')
    p.add_argument('--source-purpose', choices=['development','native-verification'],default='development',
                   help='F1 preparation: engineering integrity or an actual native G3 invocation')
    p.add_argument('--producer-gym',help='Frozen sibling WP02 Gym/Dreamer verifier for F1/F2/F3 native source verification')
    p.add_argument('--producer-lock',help='Reviewed frozen verifier pin JSON; requires its external SHA256')
    p.add_argument('--producer-lock-sha256',help='Externally approved SHA256 of the verifier pin JSON')
    p.add_argument('--execution-mode',choices=['SHADOW','ONLINE_SUPERVISED','ONLINE_RL_CONTROLLED'],default='ONLINE_SUPERVISED',
                   help='Built-in bounded synthetic online-probe mode only')
    p.add_argument('--authorize-rl',action='store_true',help='Separate reviewed formal RL authorization; never implied by BC authorization')
    p.add_argument('--rl-review',help='Independent operator-reviewed RL document, not generated by the trainer')
    p.add_argument('--rl-review-sha256',help='Explicit operator-approved hash of the independent RL review')
    args = p.parse_args()
    if (args.producer_lock is None) != (args.producer_lock_sha256 is None):
        p.error('--producer-lock and --producer-lock-sha256 must be supplied together')
    if args.producer_lock is not None:
        native_commands = {stage+'-'+action for stage in ('f1','f2','f3')
                           for action in ('prepare','read-check','start','resume')}
        native_commands.update(('online-prepare','online-start','online-resume'))
        if args.command not in native_commands:
            p.error('producer pin flags are only accepted by F1/F2/F3 and online native source routes')
        if (not args.command.startswith('online-') and args.command.endswith(('prepare','read-check'))
                and args.source_purpose!='native-verification'):
            p.error('producer pin flags require --source-purpose native-verification')
    if args.command.startswith('exam-'):
        forbidden=('--authorize-training','--authorize-rl','--rl-review','--rl-review-sha256','--prepared',
            '--run-dir','--execution-mode','--diagnostic-components','--diagnostic-q-admission',
            '--resume-online','--verify-resume-only','--holdout-contract','--source-checkpoint',
            '--producer-gym','--source-purpose')
        if any(a.split('=',1)[0] in forbidden for a in sys.argv[1:]) or args.eval_device!='cpu':
            p.error('exam commands are CPU read-only evaluation; no training, fusion, online or GPU flags')
        os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
        from gym_trading_env.research import fg_s3_e_quality_exam as exam
        if args.command in ('exam-plan','exam-run'):
            if not args.config or args.eval_dir or args.output_dir or args.command=='exam-plan' and args.stop_at is not None:
                p.error('exam-plan/run requires --config; output is frozen in that config; run may pause with --stop-at')
            result=exam.create_plan(args.config) if args.command=='exam-plan' else exam.run(args.config,stop_at=args.stop_at)
        elif args.command=='exam-resume':
            if not args.eval_dir or args.config or args.output_dir:p.error('exam-resume requires --eval-dir and optional --stop-at')
            result=exam.run(directory=args.eval_dir,resume=True,stop_at=args.stop_at)
        else:
            if not args.eval_dir or args.config or args.stop_at is not None:p.error('exam postprocessing requires --eval-dir')
            if args.command in ('exam-verify','exam-compare'):
                if not args.output_dir:p.error('exam-verify/compare requires a new --output-dir')
                result=getattr(exam,'verify' if args.command=='exam-verify' else 'compare')(args.eval_dir,args.output_dir)
            else:result=exam.report(args.eval_dir,args.output_dir)
        print(json.dumps({k:v for k,v in result.items() if k not in ('models','jobs','evidence','source')},ensure_ascii=False,indent=2))
        return 0
    if args.command.startswith('online-'):
        os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
        if ((args.authorize_rl or args.rl_review or args.rl_review_sha256) and args.command not in ('online-start','online-resume')):
            p.error('separate RL authorization is only accepted by online-start/resume')
        if (args.authorize_rl or args.rl_review or args.rl_review_sha256) and args.command!='online-resume':
            p.error('RL review and authorization require online-resume after a technical pause')
        if args.diagnostic_components or args.diagnostic_q_admission or args.resume_online:
            p.error('online execution forbids prediction/Q overrides and legacy resume flags')
        from gym_trading_env.research import fg_s3_e_quality_online as online
        if args.command=='online-prepare':
            if not args.config or args.prepared or args.run_dir or args.authorize_training:p.error('online-prepare requires --config; draft accepts --output-dir for a non-authenticated diagnostic only')
            result=online.prepare(args.config,diagnostic_output=args.output_dir,producer_gym=args.producer_gym,
                producer_lock=args.producer_lock,producer_lock_sha256=args.producer_lock_sha256)
        elif args.command in ('online-start','online-resume'):
            if not args.prepared or args.config or args.run_dir or args.output_dir:p.error('online-start/resume requires --prepared; Run and source are frozen')
            result=online.start_prepared(args.prepared,resume=args.command=='online-resume',authorized=args.authorize_training,
                producer_gym=args.producer_gym,producer_lock=args.producer_lock,producer_lock_sha256=args.producer_lock_sha256,
                stop_at=args.stop_at,rl_review_path=args.rl_review,
                rl_review_sha256=args.rl_review_sha256,authorize_rl=args.authorize_rl)
        elif args.command=='online-rl-review-packet':
            if (not args.run_dir or not args.output_dir or args.config or args.prepared
                    or args.authorize_training or args.authorize_rl):
                p.error('online-rl-review-packet requires --run-dir and a new external --output-dir; it grants no authorization')
            result=online.rl_review_packet(args.run_dir,args.output_dir)
        elif args.command=='online-probe':
            if not args.output_dir or args.config or args.prepared or args.run_dir or args.authorize_training or args.authorize_rl or args.eval_device!='cpu':p.error('online-probe accepts a new --output-dir and --execution-mode for fixed bounded CPU synthetic data only')
            from gym_trading_env.research.fg_s3_e_quality_online_checks import cpu_probe
            result=cpu_probe(args.output_dir,args.execution_mode)
        elif args.command=='online-evaluate':
            if not args.run_dir or not args.source_checkpoint or not args.output_dir or args.config or args.prepared or args.authorize_training:p.error('online-evaluate requires --run-dir --source-checkpoint --output-dir for finite deterministic Actor-only reads')
            result=online.evaluate_checkpoint(args.run_dir,args.source_checkpoint,args.output_dir,steps=args.eval_steps)
        else:
            if not args.run_dir or args.config or args.prepared or args.authorize_training:p.error('online-report requires --run-dir and optional new --output-dir')
            result=online.report(args.run_dir,args.output_dir)
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    if args.command.startswith('f3-'):
        os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
        if args.command in ('f3-start','f3-resume'):
            if not args.authorize_training:raise ValueError('F3_EXTERNAL_TRAINING_AUTHORIZATION_REQUIRED')
            if not args.prepared or args.config or args.run_dir:p.error('f3-start/resume requires --prepared only; run path is frozen')
            from gym_trading_env.research.fg_replay import read_manifest
            device=read_manifest(Path(args.prepared)/'settings.json')['settings']['device']
            if device=='gpu':os.environ['JAX_PLATFORMS']='cuda';os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            from gym_trading_env.research.fg_s3_e_quality_actor import start_prepared
            result=start_prepared(args.prepared,resume=args.command=='f3-resume',authorized=True,producer_gym=args.producer_gym,producer_lock=args.producer_lock,producer_lock_sha256=args.producer_lock_sha256,stop_at=args.stop_at)
        elif args.command=='f3-probe':
            if not args.output_dir or args.config or args.prepared or args.run_dir or args.authorize_training:p.error('f3-probe accepts a new --output-dir and --eval-device for built-in synthetic data only')
            if args.eval_device=='gpu':os.environ['JAX_PLATFORMS']='cuda';os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            from gym_trading_env.research.fg_s3_e_quality_actor_checks import probe
            result=probe(args.output_dir,device=args.eval_device)
        elif args.command=='f3-read-check':
            if not args.config or not args.output_dir or args.prepared or args.run_dir or args.authorize_training:p.error('f3-read-check requires --config --output-dir without training authorization')
            from gym_trading_env.research.fg_s3_e_quality_actor_checks import read_check
            result=read_check(args.config,args.output_dir,purpose=args.source_purpose,producer_gym=args.producer_gym,producer_lock=args.producer_lock,producer_lock_sha256=args.producer_lock_sha256)
        elif args.command=='f3-evaluate':
            if (not args.run_dir or not args.source_checkpoint or not args.output_dir or args.config or args.prepared
                    or args.authorize_training or args.diagnostic_components or args.diagnostic_q_admission):
                p.error('f3-evaluate requires --run-dir --source-checkpoint --output-dir; prediction/teacher overrides are forbidden')
            from gym_trading_env.research.fg_s3_e_quality_actor_checks import evaluate_checkpoint
            result=evaluate_checkpoint(args.run_dir,args.source_checkpoint,args.output_dir,steps=args.eval_steps)
        else:
            if not args.config or args.prepared or args.run_dir or args.authorize_training:p.error('f3-prepare requires --config without training authorization')
            from gym_trading_env.research.fg_s3_e_quality_actor import prepare
            result=prepare(args.config,purpose=args.source_purpose,producer_gym=args.producer_gym,producer_lock=args.producer_lock,producer_lock_sha256=args.producer_lock_sha256)
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    if args.command.startswith('f2-'):
        os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
        if args.command in ('f2-start','f2-resume'):
            if not args.authorize_training:raise ValueError('F2_EXTERNAL_TRAINING_AUTHORIZATION_REQUIRED')
            if not args.prepared or args.config or args.run_dir:p.error('f2-start/resume requires --prepared only; run path is frozen')
            from gym_trading_env.research.fg_replay import read_manifest
            device=read_manifest(Path(args.prepared)/'settings.json')['settings']['device']
            if device=='gpu':os.environ['JAX_PLATFORMS']='cuda';os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            from gym_trading_env.research.fg_s3_e_quality_heads import start_prepared
            result=start_prepared(args.prepared,resume=args.command=='f2-resume',authorized=True,producer_gym=args.producer_gym,producer_lock=args.producer_lock,producer_lock_sha256=args.producer_lock_sha256,stop_at=args.stop_at)
        elif args.command=='f2-probe':
            if not args.output_dir or args.config or args.prepared or args.authorize_training:p.error('f2-probe accepts a new --output-dir and --eval-device for built-in synthetic data only')
            if args.eval_device=='gpu':os.environ['JAX_PLATFORMS']='cuda';os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            from gym_trading_env.research.fg_s3_e_quality_heads_checks import probe
            result=probe(args.output_dir,device=args.eval_device)
        elif args.command=='f2-read-check':
            if not args.config or not args.output_dir or args.prepared or args.authorize_training:p.error('f2-read-check requires --config --output-dir without training authorization')
            from gym_trading_env.research.fg_s3_e_quality_heads_checks import read_check
            result=read_check(args.config,args.output_dir,purpose=args.source_purpose,producer_gym=args.producer_gym,producer_lock=args.producer_lock,producer_lock_sha256=args.producer_lock_sha256)
        else:
            if not args.config or args.prepared or args.authorize_training:p.error('f2-prepare requires --config without training authorization')
            from gym_trading_env.research.fg_s3_e_quality_heads import prepare
            result=prepare(args.config,purpose=args.source_purpose,producer_gym=args.producer_gym,producer_lock=args.producer_lock,producer_lock_sha256=args.producer_lock_sha256)
        print(json.dumps(result,indent=2,ensure_ascii=False));return
    if args.command.startswith('f1-'):
        # F1 has a separate stage admission, never a global Composition READY.
        # All preparation is CPU; authorization is checked before any run I/O.
        os.environ['JAX_PLATFORMS']='cpu'
        os.environ['CUDA_VISIBLE_DEVICES']=''
        if args.command in ('f1-start','f1-resume'):
            if not args.authorize_training:
                raise ValueError('F1_EXTERNAL_TRAINING_AUTHORIZATION_REQUIRED')
            if not args.prepared or args.config or args.run_dir:
                p.error('f1-start/resume requires --prepared; run path is frozen in its settings')
            from gym_trading_env.research.fg_s3_e_quality_foundation import read_settings,start_prepared
            from gym_trading_env.research.fg_replay import read_manifest
            device=read_manifest(Path(args.prepared)/'settings.json')['settings']['device']
            if device=='gpu':
                os.environ['JAX_PLATFORMS']='cuda';os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            result=start_prepared(args.prepared,resume=args.command=='f1-resume',authorized=True,
                producer_gym=args.producer_gym,producer_lock=args.producer_lock,producer_lock_sha256=args.producer_lock_sha256,stop_at=args.stop_at)
            print(json.dumps(result,indent=2,ensure_ascii=False));return
        if args.command=='f1-probe':
            if not args.output_dir or args.config or args.prepared or args.authorize_training:
                p.error('f1-probe only accepts --output-dir for the built-in CPU synthetic source')
            from gym_trading_env.research.fg_s3_e_quality_foundation_checks import cpu_probe
            print(json.dumps(cpu_probe(args.output_dir),indent=2));return
        if args.command=='f1-read-check':
            if not args.config or not args.output_dir or args.prepared or args.authorize_training:
                p.error('f1-read-check requires --config --output-dir and cannot authorize training')
            from gym_trading_env.research.fg_s3_e_quality_foundation_checks import read_check
            result=read_check(args.config,args.output_dir,purpose=args.source_purpose,producer_gym=args.producer_gym,producer_lock=args.producer_lock,producer_lock_sha256=args.producer_lock_sha256)
            print(json.dumps(dict(status=result['status'],actual_optimizer_updates=0)));return
        if not args.config or args.prepared or args.authorize_training:
            p.error('f1-prepare requires --config and cannot authorize training')
        from gym_trading_env.research.fg_s3_e_quality_foundation import prepare as prepare_f1
        result=prepare_f1(args.config,purpose=args.source_purpose,producer_gym=args.producer_gym,producer_lock=args.producer_lock,producer_lock_sha256=args.producer_lock_sha256)
        print(json.dumps(result,indent=2,ensure_ascii=False))
        return
    if args.verify_resume_only and (args.command != 'admit-e-resume' or args.stop_at is not None):
        p.error('--verify-resume-only requires admit-e-resume without --stop-at')
    if args.diagnostic_components and args.command!='evaluate':p.error('--diagnostic-components requires evaluate')
    if args.diagnostic_q_admission and args.command!='evaluate':p.error('--diagnostic-q-admission requires evaluate')
    if args.time_fold and args.command!='compare-predictions':p.error('--time-fold requires compare-predictions')
    if (args.check_only or args.resume_online) and args.command != 'simulate-online':
        p.error('--check-only and --resume-online require simulate-online')
    if args.command == 'monitor-online':
        if not args.run_dir:
            p.error('monitor-online requires --run-dir')
        from gym_trading_env.research.fg_s3_online_monitor import monitor_online
        monitor_online(args.run_dir, output_dir=args.output_dir, runner_pid=args.runner_pid,
                       interval=args.interval, once=args.once, compact=not args.json, until=args.until)
        return
    # Composition preflight precedes trainer imports and CUDA selection. The
    # shared pure resolver is also used by direct prepare/runtime/model callers.
    from dreamerv3.configuration import read_config_yaml
    from dreamerv3.fg_s3_composition import normalize_config, resolve_composition, settings_composition
    if args.command in ('prepare', 'start', 'resume', 'evaluate', 'report', 'describe-composition'):
        if args.config:
            path = Path(args.config)
            if not path.is_file():
                path = PROJECT/'configs/experiments/fg_s3_v2'/f'{args.config}.yaml'
            raw = read_config_yaml(path)
            normalize_config(raw, executable=args.command != 'describe-composition')
            if args.command == 'describe-composition':
                print(json.dumps(dict(resolved_composition=resolve_composition(raw),
                    legacy_c2=raw.get('mode') == 'c2'), indent=2))
                return
        elif args.command == 'describe-composition':
            p.error('describe-composition requires --config')
        elif args.prepared:
            settings_composition(json.loads((Path(args.prepared)/'settings.json').read_text())['settings'])
    admission = args.command.startswith('admit-e-')
    admission_device = None
    if admission:
        if args.authorize_training:
            p.error('E acceptance cannot authorize formal training')
        if args.command == 'admit-e-prepare':
            if not args.config or args.prepared:
                p.error('admit-e-prepare requires --config')
            path = Path(args.config)
            if not path.is_file():
                path = PROJECT/'configs/experiments/fg_s3_v2'/f'{args.config}.yaml'
            admission_device = read_config_yaml(path)['device']
        else:
            if not args.prepared or args.config:
                p.error('admit-e-start/resume require --prepared')
            admission_device = json.loads((Path(args.prepared)/'settings.json').read_text())['settings']['device']
    device=admission_device or 'gpu'
    if args.command in ('start','resume','evaluate'):
        selected=raw if args.config else json.loads((Path(args.prepared)/'settings.json').read_text())['settings']
        device=selected.get('device','gpu')
    if args.command in ('evaluate-source','evaluate-reference'): device=args.eval_device
    os.environ['JAX_PLATFORMS'] = 'cpu' if args.command in ('prepare','report','score-predictions','compare-predictions','admit-e-prepare') or device=='cpu' else 'cuda'
    os.environ.pop('JAX_PLATFORM_NAME', None)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    import gym_trading_env, dreamerv3, embodied
    if (Path(gym_trading_env.__file__).resolve().parent != PROJECT/'src/gym_trading_env'
            or Path(dreamerv3.__file__).resolve().parent != PROJECT.parent/'dreamerv3/dreamerv3'
            or Path(embodied.__file__).resolve().parent != PROJECT.parent/'dreamerv3/embodied'):
        raise ValueError('Foreign research Python imports')
    from gym_trading_env.research.fg_s3_v2_experiment import prepare, read_settings, settings_from_prepared
    if admission:
        from gym_trading_env.research import fg_s3_e_admission
        if args.command == 'admit-e-prepare':
            result = fg_s3_e_admission.prepare(args.config)
        else:
            settings = fg_s3_e_admission.read_manifest(Path(args.prepared)/'settings.json')['settings']
            result = fg_s3_e_admission.run(args.prepared, args.run_dir or settings['run_dir'],
                resume=args.command == 'admit-e-resume', stop_at=args.stop_at,
                verify_resume_only=args.verify_resume_only)
    elif args.command=='evaluate-reference':
        if not args.eval_dir or not args.source_checkpoint or not args.output_dir:
            p.error('evaluate-reference requires --eval-dir, --source-checkpoint and new --output-dir')
        from gym_trading_env.research.fg_s3_quality_prediction_evaluation import reference_predictions
        result=reference_predictions(args.eval_dir,args.source_checkpoint,args.output_dir,device=args.eval_device)
    elif args.command=='compare-predictions':
        if not args.eval_dir or not args.reference_dir or not args.output_dir:
            p.error('compare-predictions requires posthoc --eval-dir, --reference-dir and new --output-dir')
        from gym_trading_env.research.fg_s3_quality_prediction_evaluation import compare_prediction_receipts
        result=compare_prediction_receipts(args.eval_dir,args.reference_dir,args.output_dir,time_fold=args.time_fold)
    elif args.command=='evaluate-source':
        if not args.prepared or not args.source_checkpoint or not args.output_dir:
            p.error('evaluate-source requires --prepared, --source-checkpoint and new --output-dir')
        from gym_trading_env.research.fg_s3_v2_evaluation import evaluate_source
        result=evaluate_source(args.prepared,args.source_checkpoint,args.output_dir,args.eval_steps,args.eval_device)
    elif args.command == 'simulate-online':
        if not args.config:
            p.error('simulate-online requires --config')
        from gym_trading_env.research.fg_s3_online_simulation import simulate_online
        result = simulate_online(args.config, run_dir=args.run_dir,
                                 resume=args.resume_online, check_only=args.check_only)
    elif args.command == 'score-predictions':
        if not args.eval_dir or not args.output_dir:
            p.error('score-predictions requires --eval-dir and a new --output-dir')
        from gym_trading_env.research.fg_s3_prediction_evaluation import score_predictions
        result = score_predictions(args.eval_dir, args.output_dir)
    elif args.command == 'prepare':
        if not args.config or args.prepared:
            p.error('prepare requires a configuration')
        result = prepare(args.config)
    elif args.command == 'gpu_check':
        from gym_trading_env.research.fg_s3_v2_gpu_check import check_gpu_model
        result = check_gpu_model()
    else:
        if not args.config and not args.prepared:
            p.error('Select --config or --prepared')
        settings = read_settings(args.config) if args.config else settings_from_prepared(args.prepared)
        prepared = args.prepared or settings['prepared_dir']
        frozen = settings_from_prepared(prepared)
        if settings != frozen:
            raise ValueError('Selected configuration differs from the prepared snapshot')
        run_dir = args.run_dir or settings['run_dir']
        if args.command == 'report':
            from gym_trading_env.research.fg_replay import read_manifest
            result = read_manifest(Path(run_dir)/'training_receipt.json')
        elif args.command == 'evaluate':
            from gym_trading_env.research.fg_s3_v2_evaluation import evaluate
            components=args.diagnostic_components.split(',') if args.diagnostic_components else None
            result = evaluate(prepared, run_dir, args.eval_steps,args.output_dir,args.holdout_contract,components,args.diagnostic_q_admission)
        else:
            from embodied.run.fg_s3_v2_train import LiveRun
            run = LiveRun(prepared, run_dir, resume=args.command == 'resume',
                          authorize_training=args.authorize_training)
            try:
                result = run.run_to(args.stop_at)
            except BaseException as e:
                from gym_trading_env.research.fg_replay import signed, write_json
                write_json(run.root/'failure.json', signed(dict(error_type=type(e).__name__, error=str(e),
                    env_steps=int(run.step), updates=run.counters.save())))
                raise
            finally:
                run.close()
    print(json.dumps({k: v for k, v in result.items() if k not in
                     ('identity', 'code', 'files', 'immutable_files', 'dirty')}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
