"""Cycle complete FG train lifecycles; chunks keep the current financial account."""
import copy
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from .entry_contract import read_yaml, fingerprint
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
        self.paths=[r['env_config'] for r in manifest['files'] if r['role']=='train' and r['eligible_for_training']]
        if not self.paths:
            raise ValueError('No complete FG train lifecycles')
        self.index=0
        self.order_mode=manifest.get('split',{}).get('training_order',{}).get('mode','sequential')
        self.order_seed=manifest.get('split',{}).get('training_order',{}).get('seed',0)
        if self.order_mode not in ('sequential','shuffle_each_epoch'):
            raise ValueError('Unsupported FG training order')
        self.rng=np.random.default_rng(self.order_seed)
        self.order=list(range(len(self.paths)))
        self.cursor=self.epoch=0
        self.initialized=False
        self.completed=[]
        self.env=FixedDirectionEntryEnv(config_path=self.paths[0])
        self.action_space,self.observation_space=self.env.action_space,self.env.observation_space

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if options and 'resume' in options:
            self.import_state(options['resume'])
            return self.env._get_obs(),self.env._get_info()
        if not self.initialized:
            self.rng=np.random.default_rng(self.order_seed if seed is None else seed)
            self._new_order()
            self.initialized=True
            self._select(self.order[0])
        if self.env.terminated:
            self.completed.append(dict(contract=self.env.contract,metrics=self.env._get_info()))
            self.cursor+=1
            if self.cursor==len(self.paths):
                self.cursor=0
                self.epoch+=1
                self._new_order()
            self._select(self.order[self.cursor],force=True)
        return self.env.reset(seed=seed)

    def _new_order(self):
        self.order=(self.rng.permutation(len(self.paths)).tolist()
                    if self.order_mode=='shuffle_each_epoch' else list(range(len(self.paths))))

    def _select(self,index,force=False):
        if force or self.index!=index:
            self.env.close()
            self.env=FixedDirectionEntryEnv(config_path=self.paths[index])
            if self.env.observation_space!=self.observation_space:
                raise ValueError('FG dataset observation contract changed')
        self.index=index

    def step(self,action):
        return self.env.step(action)

    def export_state(self):
        return dict(schema=2,manifest_hash=self.manifest_hash,index=self.index,
            order_mode=self.order_mode,order=copy.deepcopy(self.order),cursor=self.cursor,
            epoch=self.epoch,initialized=self.initialized,rng_state=copy.deepcopy(self.rng.bit_generator.state),
            completed=copy.deepcopy(self.completed),account=self.env.export_state())

    def import_state(self,state):
        if state['schema'] not in (1,2) or state['manifest_hash']!=self.manifest_hash or not 0<=state['index']<len(self.paths):
            raise ValueError('FG dataset checkpoint mismatch')
        if state['schema']==1:
            if self.order_mode!='sequential':
                raise ValueError('Shuffled FG checkpoint requires sampler state')
            order=list(range(len(self.paths)))
            cursor=state['index']
        else:
            order,cursor=state['order'],state['cursor']
            if (state['order_mode']!=self.order_mode or sorted(order)!=list(range(len(self.paths)))
                    or not 0<=cursor<len(order) or order[cursor]!=state['index'] or state['epoch']<0):
                raise ValueError('FG dataset checkpoint sampler mismatch')
        if self.index!=state['index']:
            self.env.close()
            self.env=FixedDirectionEntryEnv(config_path=self.paths[state['index']])
        self.env.import_state(state['account'])
        self.index=state['index']
        self.completed=copy.deepcopy(state['completed'])
        self.order,self.cursor=copy.deepcopy(order),cursor
        self.epoch=state.get('epoch',0)
        self.initialized=state.get('initialized',True)
        if state['schema']==2:
            self.rng.bit_generator.state=copy.deepcopy(state['rng_state'])

    def close(self):
        self.env.close()
