"""Cycle complete FG train lifecycles; chunks keep the current financial account."""
import copy
import json
from pathlib import Path
import gymnasium as gym
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
        self.completed=[]
        self.env=FixedDirectionEntryEnv(config_path=self.paths[0])
        self.action_space,self.observation_space=self.env.action_space,self.env.observation_space

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if options and 'resume' in options:
            self.import_state(options['resume'])
            return self.env._get_obs(),self.env._get_info()
        if self.env.terminated:
            self.completed.append(dict(contract=self.env.contract,metrics=self.env._get_info()))
            self.env.close()
            self.index=(self.index+1)%len(self.paths)
            self.env=FixedDirectionEntryEnv(config_path=self.paths[self.index])
            if self.env.observation_space!=self.observation_space:
                raise ValueError('FG dataset observation contract changed')
        return self.env.reset(seed=seed)

    def step(self,action):
        return self.env.step(action)

    def export_state(self):
        return dict(schema=1,manifest_hash=self.manifest_hash,index=self.index,
            completed=copy.deepcopy(self.completed),account=self.env.export_state())

    def import_state(self,state):
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
