"""Frozen original quality_actor; no training lifecycle or teacher creation."""
from pathlib import Path
import copy

from .fg_s3_e_quality_exam_contract import identity
from .fg_replay import digest,signed,check_files
from .fg_single3_evaluation import ActorPolicy

_LOADED_ACTORS={}


class ExamFrozenActor:
    def __init__(self,agent,source,*,source_files=None,evaluation_seed=None):
        from dreamerv3.fg_s3a_actor import parameter_digests
        from embodied.run.fg_checkpoint import plain
        self.agent=agent;self.source=copy.deepcopy(source);self.source_files=source_files or {}
        from dreamerv3.fg_s3_e_quality_actor import PROTOCOL
        if agent.config.get('s3_v2_mode')!='quality_v2' or agent.config.get('quality_f3_protocol')!=PROTOCOL:
            raise ValueError('EXAM_CURRENT_E_QUALITY_ACTOR_REQUIRED')
        forbidden=('quality_decision_admitted','quality_prediction_admitted','quality_q_admitted',
                   'quality_fuse_score','quality_fuse_outcome')
        if any(agent.model.quality[k] for k in forbidden):raise ValueError('EXAM_TEACHER_FREE_POLICY_REQUIRED')
        agent.sync_policy_from_train(reset_counter=True)
        agent.configure_evaluation_seed(int(agent.config.seed) if evaluation_seed is None else evaluation_seed)
        self.policy=ActorPolicy(agent)
        self.before=parameter_digests(agent.params)
        self.policy_before=parameter_digests(agent.policy_params)
        self.counters_before={k:v for k,v in agent.save()['counters'].items() if k!='actions'}
        config=plain(agent.config);config.pop('logdir',None)
        self.receipt=signed(dict(protocol='FG_S3_E_QUALITY_EXAM_FROZEN_ACTOR_V1',source=source,
            actor='quality_actor',mode='masked_argmax',teacher_used=False,exploration_rate=0.,
            prediction_fusion=False,q_fusion=False,training_updates=0,
            evaluation_seed=agent.save_policy_state()['seed'],
            parameter_digests=self.before,policy_digests=self.policy_before,
            training_counters=self.counters_before,model_config=config,
            observation=__import__('gym_trading_env.research.fg_s3_e_observation',fromlist=['observation_contract']).observation_contract()))

    @classmethod
    def from_f3_checkpoint(cls,run,checkpoint,output,*,engineering=False,evaluation_seed=None):
        return cls._load('F3',run,checkpoint,output,engineering=engineering,evaluation_seed=evaluation_seed)

    @classmethod
    def from_f4_checkpoint(cls,run,checkpoint,output,*,engineering=False,evaluation_seed=None):
        return cls._load('F4',run,checkpoint,output,engineering=engineering,evaluation_seed=evaluation_seed)

    @classmethod
    def _load(cls,kind,run,checkpoint,output,*,engineering,evaluation_seed):
        from embodied.run.fg_s3_e_quality_exam import inspect_exam_actor
        from .fg_s3_e_quality_actor import create_agent
        from dreamerv3.fg_s3a_actor import parameter_digests
        source,state,settings=inspect_exam_actor(kind,run,checkpoint,engineering=engineering)
        settings['device']='cpu'
        agent=create_agent(settings,output,controlled_small=source['engineering_only'],execution_mode=source['execution_mode'])
        if (set(agent.params)!=set(state['params']) or any(agent.params[k].shape!=v.shape
                or agent.params[k].dtype!=v.dtype for k,v in state['params'].items())):
            raise ValueError('EXAM_FULL_CHECKPOINT_PARAMETER_SCHEMA_MISMATCH')
        initial_files={}
        if kind=='F4':
            import pickle
            from embodied.run.fg_checkpoint import validate_bundle
            initial=Path(run)/'ckpt/step_000000000001'
            initial_meta=validate_bundle(initial,source['source_identity'])
            first=pickle.loads((initial/'agent.pkl').read_bytes())['params']
            if (first.keys()!=state['params'].keys() or any(first[k].shape!=v.shape or first[k].dtype!=v.dtype
                    for k,v in state['params'].items())):raise ValueError('EXAM_F4_INITIAL_SCHEMA_MISMATCH')
            mode=source['execution_mode'];changed=[]
            for k,v in state['params'].items():
                if first[k].tobytes()==v.tobytes():continue
                owner=agent.model.f3_state_owner[k]
                allowed=mode!='SHADOW' and (owner in ('CLOCK','quality_actor/') or mode=='ONLINE_RL_CONTROLLED' and (
                    k.startswith(('val/','slowval/','slowval_count/','retnorm/','valnorm/','advnorm/'))
                    or k.startswith('opt/') and 'val/' in k))
                # Native OnlineRun explicitly writes this control state before
                # RL updates; it is not an optimized model parameter. Preserve
                # it exactly, with its original gate/engineering provenance.
                if k=='avail_actor_gate/value' and mode=='ONLINE_RL_CONTROLLED':
                    import numpy as np
                    control=float(np.asarray(v))
                    ident=source['source_identity']
                    gates=pickle.loads((Path(checkpoint)/'gates.pkl').read_bytes())
                    valid_probe=ident['controlled_small'] and ident.get('numeric_probe') is True
                    valid_formal=not ident['controlled_small'] and gates['formal_admitted'] and all(gates['ready'].values())
                    allowed=control in (0.,1.) and (control==0. or valid_probe or valid_formal)
                if not allowed:raise ValueError('EXAM_F4_SOURCE_FROZEN_STATE_CHANGED: '+k)
                changed.append(k)
            source=signed({**{k:v for k,v in source.items() if k!='content_hash'},'source_freeze':dict(initial_manifest_sha256=digest(initial/'manifest.json'),
                changed_leaves=changed,all_frozen_bytes_equal=True,
                native_control_state={k:float(state['params'][k]) for k in changed if k=='avail_actor_gate/value'})})
            initial_files={str(initial/k):v for k,v in initial_meta['files'].items()}
            initial_files[str(initial/'manifest.json')]=digest(initial/'manifest.json')
        agent.load_exact(state)
        if parameter_digests(agent.params)!=parameter_digests(state['params']):raise ValueError('EXAM_EXACT_MODEL_LOAD_FAILED')
        root=Path(run);node=Path(checkpoint)
        files={**initial_files,**{str(node/k):v for k,v in source['member_hashes'].items()}}
        files.update({str(root/'run_identity.json'):source['run_identity_sha256'],str(node/'manifest.json'):source['checkpoint_manifest_sha256']})
        result=cls(agent,source,source_files=files,evaluation_seed=evaluation_seed)
        import weakref
        key=id(result)
        _LOADED_ACTORS[key]=(weakref.ref(result,lambda _: _LOADED_ACTORS.pop(key,None)),
            identity(result.receipt),identity(result.source))
        return result

    def require_mode(self,plan):
        if plan['exam_mode']=='DEV_EXAM':
            proof=_LOADED_ACTORS.get(id(self))
            if (proof is None or proof[0]() is not self or proof[1:]!=(identity(self.receipt),identity(self.source))
                    or self.source.get('engineering_only') is not False):
                raise ValueError('EXAM_AUTHENTICATED_FORMAL_ACTOR_REQUIRED_NOT_JSON')

    @property
    def last_diagnostics(self):return self.policy.last_diagnostics
    waited=0
    waiting=False

    def __call__(self,carry,obs):
        import numpy as np
        carry,action,out=self.policy(carry,obs)
        diagnostics=self.last_diagnostics.get('quality',{})
        if (any(diagnostics.get(k)!=0 for k in ('quality/delta_O','quality/delta_S','quality/delta_Q'))
                or not obs['is_last'][0] and int(action['action'][0])!=diagnostics.get('quality/argmax_action')):
            raise ValueError('EXAM_POLICY_NOT_PURE_QUALITY_ARGMAX')
        if not bool(np.asarray(obs['action_mask'])[0,int(action['action'][0])]):raise ValueError('EXAM_ILLEGAL_ACTOR_ACTION')
        return carry,action,out

    def observe_row(self,row,*,legal_after):self.policy.observe_row(row,legal_after=legal_after)

    def verify_frozen(self):
        from dreamerv3.fg_s3a_actor import parameter_digests
        check_files(self.source_files)
        if (self.before!=parameter_digests(self.agent.params) or self.policy_before!=parameter_digests(self.agent.policy_params)
                or self.counters_before!={k:v for k,v in self.agent.save()['counters'].items() if k!='actions'}
                or self.agent.pending_sync is not None
                or self.agent.save_policy_state()['seed']!=self.receipt['evaluation_seed']):
            raise ValueError('EXAM_MODEL_OR_TRAINING_STATE_CHANGED')
        return dict(all_parameter_bytes_equal=True,policy_bytes_equal=True,training_counters_equal=True,
            source_files_unchanged=True,training_updates=0)

    def save(self):
        return dict(actor_identity=identity(self.receipt),inference=self.agent.save_policy_state(),
            diagnostics=copy.deepcopy(self.last_diagnostics))

    def load(self,state):
        if state['actor_identity']!=identity(self.receipt):raise ValueError('EXAM_ACTOR_RESTORE_IDENTITY_MISMATCH')
        self.verify_frozen();self.agent.load_policy_state(state['inference'])
        self.policy.last_diagnostics=copy.deepcopy(state['diagnostics'])
