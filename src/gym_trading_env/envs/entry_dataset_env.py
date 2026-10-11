"""Cycle complete FG train lifecycles; chunks keep the current financial account."""
import copy
import json
import numpy as np
from pathlib import Path
import gymnasium as gym
from .entry_contract import read_yaml, fingerprint, resolve_path
from .entry_env import FixedDirectionEntryEnv


class FGEntryDatasetEnv(gym.Env):
    metadata={'render_modes':[]}

    def __init__(self,config_path,render_mode=None):
        raw=read_yaml(config_path)
        if raw.get('strategy_mode')!='entry_dataset' or raw.get('role')!='train':
            raise ValueError('FG dataset environment requires the explicit train role')
        manifest=json.loads(Path(raw['manifest_path']).read_text())
        expected=manifest.pop('content_hash')
        if fingerprint(manifest)!=expected or manifest['product']!='FG':
            raise ValueError('Invalid frozen FG manifest')
        self.manifest_hash=expected
        self.scheduler=raw.get('scheduler',{})
        self.rotating=self.scheduler.get('mode')=='least_progress'
        if self.scheduler and not self.rotating:
            raise ValueError('Unknown FG account scheduler')
        self.rows=([r['rows'] for r in manifest['files'] if r['role']=='train' and r['eligible_for_training']]
                   if self.rotating else [])
        if self.rotating and (any(n<=1 for n in self.rows) or int(self.scheduler.get('chunk_bars',0))<=0):
            raise ValueError('Account rotation requires positive lifecycle lengths and chunk size')
        self.paths=[r['env_config'] for r in manifest['files'] if r['role']=='train' and r['eligible_for_training']]
        if not self.paths:
            raise ValueError('No complete FG train lifecycles')
        self.index=0
        self.completed=[]
        self.accounts={}
        self.advanced=[0]*len(self.paths)
        self.repeats=[0]*len(self.paths)
        self.order=list(range(len(self.paths)))
        self.scheduler_seed=None
        self.slice_bars=0
        self.env=FixedDirectionEntryEnv(config_path=self.paths[0])
        self.action_space,self.observation_space=self.env.action_space,self.env.observation_space
        self.observation_identity=self.env.observation_identity
        # Equal shapes do not imply equal max-lots/pending/reward semantics.
        for path in self.paths:
            item=read_yaml(path)
            if item['entry'].get('protocol','fg_entry_v1')!=self.env.protocol:
                raise ValueError('FG dataset mixes observation protocols')
            if self.env.single3:
                from gym_trading_env.rewards.single3_reward import Single3RewardConfig
                entry=item['entry']
                reward=(read_yaml(resolve_path(path,entry['strategy_config']))['reward']
                        if entry.get('strategy_config') else entry['reward'])
                if Single3RewardConfig(**reward).hash!=self.env.reward_config.hash:
                    raise ValueError('FG dataset mixes reward candidates')

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if options and 'resume' in options:
            self.import_state(options['resume'])
            return self.env._get_obs(),self.env._get_info()
        if self.rotating:
            if self.scheduler_seed is None:
                self.scheduler_seed=0 if seed is None else int(seed)
                self.order=list(map(int,np.random.default_rng(self.scheduler_seed).permutation(len(self.paths))))
            if self.env.terminated or self.env.status=='CHUNK_END':
                self._rotate()
            return self.env.reset(seed=seed)
        if self.env.terminated:
            self.completed.append(dict(contract=self.env.contract,metrics=self.env._get_info(),
                                       advanced_bars=self.env.current_step))
            self.env.close()
            self.index=(self.index+1)%len(self.paths)
            self.env=FixedDirectionEntryEnv(config_path=self.paths[self.index])
            if self.env.observation_space!=self.observation_space:
                raise ValueError('FG dataset observation contract changed')
        return self.env.reset(seed=seed)

    def step(self,action):
        before=self.env.current_step
        result=self.env.step(action)
        if self.rotating:
            delta=self.env.current_step-before
            self.advanced[self.index]+=delta
            self.slice_bars+=delta
            if not result[2] and not result[3] and self.slice_bars>=int(self.scheduler['chunk_bars']):
                self.env.status,self.env.truncated='CHUNK_END',True
                obs,reward,terminated,_,info=result
                result=(obs,reward,terminated,True,info)
        return result

    def _rotate(self):
        if self.env.terminated:
            if self.env.entries:
                raise ValueError('Cannot repeat a lifecycle with unresolved positions')
            self.completed.append(dict(contract=self.env.contract,metrics=self.env._get_info()))
            self.repeats[self.index]+=1
        self.accounts[self.index]=self.env.export_state()
        target=min(self.order,key=lambda i:self.advanced[i]/self.rows[i])
        if target!=self.index:
            self.env.close()
            self.env=FixedDirectionEntryEnv(config_path=self.paths[target])
            if self.env.observation_space!=self.observation_space:
                raise ValueError('FG dataset observation contract changed')
            if target in self.accounts:
                self.env.import_state(self.accounts[target])
            else:
                self.env.reset(seed=self.scheduler_seed+1000003*target)
            self.index=target
        self.slice_bars=0

    def export_state(self):
        if self.rotating:
            accounts=copy.deepcopy(self.accounts)
            accounts[self.index]=self.env.export_state()
            return dict(schema=2,manifest_hash=self.manifest_hash,index=self.index,
                completed=copy.deepcopy(self.completed),accounts=accounts,advanced=self.advanced.copy(),
                repeats=self.repeats.copy(),order=self.order.copy(),scheduler_seed=self.scheduler_seed,
                slice_bars=self.slice_bars,scheduler=copy.deepcopy(self.scheduler))
        return dict(schema=1,manifest_hash=self.manifest_hash,index=self.index,
            completed=copy.deepcopy(self.completed),account=self.env.export_state())

    def import_state(self,state):
        if self.rotating:
            if (state['schema']!=2 or state['manifest_hash']!=self.manifest_hash
                    or state['scheduler']!=self.scheduler or sorted(state['order'])!=list(range(len(self.paths)))
                    or len(state['advanced'])!=len(self.paths) or len(state['repeats'])!=len(self.paths)
                    or any(n<0 for n in state['advanced'])
                    or not 0<=state['index']<len(self.paths)):
                raise ValueError('FG scheduler checkpoint mismatch')
            if self.index!=state['index']:
                self.env.close()
                self.env=FixedDirectionEntryEnv(config_path=self.paths[state['index']])
            accounts={int(k):copy.deepcopy(v) for k,v in state['accounts'].items()}
            self.env.import_state(accounts[state['index']])
            self.index=state['index']
            self.accounts=accounts
            for key in ('completed','advanced','repeats','order','scheduler_seed','slice_bars'):
                setattr(self,key,copy.deepcopy(state[key]))
            return
        if state['schema']!=1 or state['manifest_hash']!=self.manifest_hash or not 0<=state['index']<len(self.paths):
            raise ValueError('FG dataset checkpoint mismatch')
        if self.index!=state['index']:
            self.env.close()
            self.env=FixedDirectionEntryEnv(config_path=self.paths[state['index']])
        self.env.import_state(state['account'])
        self.index=state['index']
        self.completed=copy.deepcopy(state['completed'])

    def close(self):
        self.env.close()
