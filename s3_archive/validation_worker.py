"""Runs only inside the archive private namespace with the recorded trainer code."""
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import traceback
from decimal import Decimal


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str)+'\n')


def signature(value):
    import numpy as np
    import jax
    if isinstance(value, (np.ndarray, jax.Array)):
        a = np.asarray(value)
        return ('array',str(a.dtype),a.shape,hashlib.sha256(a.tobytes()).hexdigest())
    if isinstance(value, dict):
        return ('dict',tuple(sorted((str(k),signature(v)) for k,v in value.items())))
    if isinstance(value,(list,tuple)):
        return ('sequence',tuple(signature(v) for v in value))
    if isinstance(value,np.generic):
        return signature(np.asarray(value))
    if isinstance(value,(str,int,float,bool,bytes,type(None))):
        return (type(value).__name__,value)
    if hasattr(value,'__dict__'):
        return (type(value).__module__,type(value).__qualname__,signature(value.__dict__))
    return (type(value).__module__,type(value).__qualname__,hashlib.sha256(pickle.dumps(value,protocol=5)).hexdigest())


def main(spec):
    import jax
    import numpy as np
    from embodied.run.fg_checkpoint import file_hash, save_replay
    from embodied.run.fg_s3_v2_train import execution_code, LiveRun
    assert all(d.platform=='gpu' for d in jax.devices()), 'GPU validation requires CUDA'
    assert execution_code()==spec['source_identity']['code'], 'Execution source identity differs'
    work, run_dir = Path(spec['work']), Path(spec['run_dir'])
    node = run_dir/'ckpt'/(run_dir/'ckpt/latest').read_text().strip()
    assert file_hash(node/'manifest.json')==spec['checkpoint_manifest_sha256']
    if spec['mode']=='eval':
        forbidden = Path(spec['oracle_root']).resolve() if spec.get('oracle_root') else None
        attempts = []
        def audit(event, args):
            if event=='open' and forbidden is not None and isinstance(args[0],(str,bytes)):
                p = Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
                if p.is_relative_to(forbidden):
                    attempts.append(str(p))
                    raise PermissionError('Actor-only validation cannot access Oracle data')
        sys.addaudithook(audit)
        from gym_trading_env.research.fg_s3_v2_evaluation import evaluate
        actual = evaluate(spec['prepared'],run_dir,spec['steps'])
        original = json.loads((Path(spec['original_eval'])/'receipt.json').read_text())
        comparisons = {}
        for key in ('env_steps','legal_decisions','actor_add_proposed','add_executed','entries',
                    'tp_count','sl30','right_censored','clock_protocol','policy_mode',
                    'teacher_guidance','oracle_files_read','beta','checkpoint_manifest_sha256'):
            comparisons[key] = dict(original=original.get(key), reproduced=actual.get(key),
                                    matched=original.get(key)==actual.get(key))
        for key in ('reward','tp_market_minutes_p50','tp_market_minutes_p90'):
            a,b = original.get(key),actual.get(key)
            matched = a==b if a is None or b is None else math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-6)
            comparisons[key] = dict(original=a,reproduced=b,matched=matched)
        old_state = json.loads((Path(spec['original_eval'])/'account_state.json').read_text())
        new_state = json.loads((Path(actual['output'])/'account_state.json').read_text())
        def net(s):
            return sum((Decimal(str(a['net_trade_pnl'])) for a in s['attributions']),Decimal(0))
        comparisons['net_pnl'] = dict(original=str(net(old_state)), reproduced=str(net(new_state)),
                                      matched=net(old_state)==net(new_state))
        passed = not attempts and all(r['matched'] for r in comparisons.values())
        return dict(status='ISOLATED_EVAL_PASS' if passed else 'ISOLATED_EVAL_DIFFERENCES',
            source_identity_verified=True, gpu_used=True, devices=[str(d) for d in jax.devices()],
            oracle_read_attempts=attempts, float_tolerance=dict(atol=1e-6,rtol=1e-6),
            comparisons=comparisons, reproduced_receipt=actual,
            reproduced_receipt_sha256=file_hash(Path(actual['output'])/'receipt.json'))
    run = LiveRun(spec['prepared'],run_dir,resume=True,authorize_training=True)
    try:
        expected = lambda name: pickle.loads((node/(name+'.pkl')).read_bytes())
        checks = {}
        checks['agent'] = signature(expected('agent'))==signature(run.agent.save_exact())
        checks['driver'] = signature(expected('driver'))==signature(run.collector.driver.save())
        checks['collector'] = signature(expected('collector'))==signature(run.collector.save())
        checks['gate'] = signature(expected('action_mask_warmup'))==signature(run.gate.save())
        checks['counters'] = signature(expected('v2_counters'))==signature(run.counters.save())
        runtime = expected('runtime')
        actual_runtime = run.runtime.save()
        # Original RuntimeState defers starting prefetch until Agent is restored.
        # Its saved start flag is preserved in resume_started during this interval.
        assert run.runtime.resume_started == runtime['started']
        actual_runtime['started'] = run.runtime.resume_started
        checks['runtime'] = signature(runtime)==signature(actual_runtime)
        replay = expected('replay')
        checks['replay_metadata'] = signature(replay)==signature(save_replay(run.collector.replay,work/'replay_roundtrip'))
        for chunk in replay['chunks']:
            with np.load(node/'replay'/chunk['name'],allow_pickle=False) as a, np.load(work/'replay_roundtrip'/chunk['name'],allow_pickle=False) as b:
                assert a.files==b.files
                assert all(signature(a[k])==signature(b[k]) for k in a.files)
        checks['replay_arrays'] = True
        checks['step'] = expected('step')==int(run.step)==run.collector.counts['env_steps']
        assert all(checks.values()), str(checks)
        assert int(run.agent.n_updates)==spec['source_identity']['source_update_offset']+spec['boundaries']['updates']
        assert run.counters.state['cycles']==spec['boundaries']['cycles']
        before = dict(env_steps=int(run.step), model_updates=int(run.agent.n_updates),
                      local_updates=run.counters.state['all_updates'])
        row = run.update()
        after = dict(env_steps=int(run.step), model_updates=int(run.agent.n_updates),
                     local_updates=run.counters.state['all_updates'])
        assert after['env_steps']==before['env_steps']
        assert after['model_updates']==before['model_updates']+1
        assert after['local_updates']==before['local_updates']+1
        assert np.isfinite(row['metrics']['opt/loss'])
        return dict(status='ISOLATED_RESUME_SMOKE_PASS', source_identity_verified=True,
            gpu_used=True, devices=[str(d) for d in jax.devices()], checks=checks,
            before=before, after=after, training_updates_executed=1, environment_advances_executed=0,
            original_run_written=False, exact_replay_verified=None)
    finally:
        run.close()


if __name__=='__main__':
    spec = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = main(spec)
    except BaseException as exc:
        result = dict(status='VALIDATION_FAILED', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        write(Path(spec['work'])/'result.json',result)
        raise
    write(Path(spec['work'])/'result.json',result)
    print(json.dumps(result,default=str),flush=True)
