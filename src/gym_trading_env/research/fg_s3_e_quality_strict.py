"""Quality TP complete-suffix values in a proven native account domain.

Reuses the audited indexed optimizer and its full native replay. The extra
ADD branch owns its own post-settlement funding certificate. This is a TRAIN
posterior teacher, not permission to train or to enable learned Q correction.
"""
from decimal import Decimal
from fractions import Fraction
import copy

import numpy as np
from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import signed,digest
from .fg_s3_quality_q_v2 import build_quality_suffix,GAMMA
from .fg_single3_replay import Single3BehaviorPolicy

PROTOCOL='FG_S3_E_QUALITY_STRICT_SUFFIX_V2'


class QualityStrictTeacher:
    def __init__(self,env,*,source,start_cursor=0):
        if source.get('fold')!='TRAIN' or env.reward_config.schema!='FG_SINGLE3_REWARD_MARKET_MINUTE_TP_QUALITY_V1':
            raise ValueError('Strict E Quality teacher requires explicit TRAIN and native Quality TP')
        if getattr(env,'_e_profile',None) is None:
            raise ValueError('Strict E teacher requires native E V2')
        boundaries=[t for t in (env.main_exit_at,env.terms.liquidate) if t is not None]
        if not boundaries:raise ValueError('Strict suffix has no authenticated native terminal boundary')
        boundary=min(boundaries)
        if boundary not in env._raw.index:
            raise ValueError('Strict suffix boundary outside observed TRAIN; right-censored')
        true_end=int(env._raw.index.get_loc(boundary))
        initial=copy.deepcopy(env.export_state())
        try:
            self.query=build_quality_suffix(env,source=source,start_cursor=start_cursor)
        finally:env.import_state(initial)
        if self.query.end!=true_end:
            raise ValueError('Strict suffix ended at an unproven gap or truncated source')
        self.start=start_cursor
        self.e_identity=env.e_observation_identity
        self.profile_identity=env._e_profile.identity
        self.clock=env.market_clock_counter
        self._values={self.query.end:Fraction(0)}
        self.identity=dict(protocol=PROTOCOL,fold='TRAIN',reward_schema=env.reward_config.schema,
            score_policy='FG_S3_QUALITY_TP_MARKET_DAY_V1',
            base_identity=self.query.identity_hash,e_identity=self.e_identity,
            profile_identity=self.profile_identity,start_cursor=start_cursor,
            suffix_boundary_identity=dict(cursor=true_end,time=str(boundary),kind='NATIVE_MAIN_ROLL_OR_CONTRACT_EXIT'),
            gamma_rational=[332,333],objective='discounted_native_float32_Quality_TP_feedback',
            tie_break='WAIT',code_sha256=digest(__file__),
            funding_domain='original_floor_for_root; independently_proven_post_ADD_quote_envelope')
        self.identity_hash=fingerprint(self.identity)

    def _value(self,cursor):
        if cursor in self._values:return self._values[cursor]
        total=Fraction(0);t=self.query.first[cursor]
        while t is not None:
            row=self.query.candidates[t]
            for step,reward in row['rewards']['C1']:
                total+=GAMMA**self.clock.reward_exponent(cursor,step)*Fraction.from_float(reward)
            t=self.query.first[row['release_cursor']]
        self._values[cursor]=total
        return total

    def query_values(self,env,mask):
        result=dict(protocol=PROTOCOL,teacher_identity=self.identity_hash,
            strict_action=None,teacher_valid=False,tie=None,action_value_add=None,
            action_value_wait=None,margin=None,reason='NO_PROOF',
            account_snapshot_hash=fingerprint(env.export_state()),
            suffix_boundary_identity=self.identity['suffix_boundary_identity'])
        if (env.e_observation_identity!=self.e_identity or env._e_profile.identity!=self.profile_identity):
            raise ValueError('Strict E source/observation identity differs')
        answer=self.query.query(env,mask)
        if not answer['valid'] or not self.start<=env.current_step<self.query.end:
            return signed({**result,'reason':answer['reason'] if not answer['valid'] else 'OUTSIDE_SUFFIX_ROOT'})
        t=env.current_step;row=self.query.candidates.get(t)
        if row is None:return signed({**result,'reason':'CANDIDATE_NOT_COVERED'})
        wait=GAMMA**self.clock.advance(t,t+1)*self._value(t+1)
        funding=None
        if not row.get('filled'):
            if not row.get('eligible'):return signed({**result,'reason':'CANDIDATE_NOT_ELIGIBLE'})
            add=wait
        else:
            if not row.get('exit') or row.get('right_censored'):
                return signed({**result,'reason':'ADD_RIGHT_CENSORED'})
            after=env._calculate_equity()+Decimal(row['exit']['net_trade_pnl'])
            bound=Decimal(self.query.proof['required_equity_upper_bound'])
            if after<=bound:
                return signed({**result,'reason':'POST_ADD_FUNDS_NOT_PROVEN'})
            funding=dict(post_add_cash_lower_bound=str(after),quote_equity_upper_bound=str(bound),
                future_retained_TP_cash_positive=True,constant_terms=True)
            release=row['release_cursor']
            add=sum((GAMMA**self.clock.reward_exponent(t,s)*Fraction.from_float(v)
                     for s,v in row['rewards']['C1']),Fraction(0))
            add+=GAMMA**self.clock.advance(t,release)*self._value(release)
        action=int(add>wait)
        if action!=answer['action']:
            raise ValueError('Strict paired values disagree with the certified native optimum')
        return signed({**result,'teacher_valid':True,'strict_action':action,'tie':add==wait,
            'action_value_add':float(add),'action_value_wait':float(wait),'margin':float(add-wait),
            'value_rational_witness':[str(wait),str(add)],'forced_add_funding_proof':funding,
            'reason':'NATIVE_QUALITY_TP_COMPLETE_SUFFIX'})

    def plan(self):
        return signed(dict(protocol=PROTOCOL,fold='TRAIN',reward_schema=self.identity['reward_schema'],
            score_policy=self.identity['score_policy'],teacher_identity=self.identity_hash,
            suffix_boundary_identity=self.identity['suffix_boundary_identity'],
            gamma_rational=[332,333],add_cursors=self.query.plan['add_cursors'],
            native_validation=self.query.identity['native_candidate_validation'],
            objective=self.identity['objective'],training_admitted=False))


class QualityP4Policy(Single3BehaviorPolicy):
    def __init__(self,spec,*,env,teacher):
        if spec.get('kind')!='quality_oracle_schedule' or spec.get('teacher_identity')!=teacher.identity_hash:
            raise ValueError('P4 requires the new authenticated Quality strict teacher; old C1 rejected')
        super().__init__(spec)
        self.env=env;self.teacher=teacher

    def __call__(self,carry,obs):
        active=not bool(obs['is_last'][0]);legal=bool(obs['action_mask'][0,1])
        action=0
        if active and legal:
            value=self.teacher.query_values(self.env,self.env._published_mask)
            if not value['teacher_valid']:raise ValueError('P4_NO_PROOF: '+value['reason'])
            action=value['strict_action']
        return carry,{'action':np.array([action],np.int32)},{}
