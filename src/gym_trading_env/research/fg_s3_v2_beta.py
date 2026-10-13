"""Oracle probability by completed real Env advances, independent of updates."""
import copy
import math


class BetaSchedule:
    protocol = 'FG_S3_V2_BETA_SCHEDULE_V1'

    def __init__(self, initial_beta, points=None):
        if (isinstance(initial_beta, bool) or not isinstance(initial_beta, (int, float))
                or not math.isfinite(initial_beta) or not 0 <= initial_beta <= 1):
            raise ValueError('Initial beta must be a finite probability')
        self.initial_beta = float(initial_beta)
        self.points = copy.deepcopy(points)
        if points is not None:
            if not isinstance(points, list) or len(points) < 2:
                raise ValueError('Beta schedule requires at least two [Env advances, beta] points')
            previous_step, previous_beta = -1, self.initial_beta
            for point in points:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError('Invalid beta schedule point')
                step, beta = point
                if (type(step) is not int or step <= previous_step
                        or isinstance(beta, bool) or not isinstance(beta, (int, float))
                        or not math.isfinite(beta) or not 0 <= beta <= previous_beta):
                    raise ValueError('Beta points require increasing Env advances and nonincreasing probabilities')
                previous_step, previous_beta = step, beta
            if points[0] != [0, initial_beta]:
                raise ValueError('Beta schedule must begin at [0, initial beta]')
        self.env_steps = 0

    def value(self, env_steps):
        if type(env_steps) is not int or env_steps < 0:
            raise ValueError('Beta progress must count completed real Env advances')
        if self.points is None:
            return self.initial_beta
        for (left, start), (right, end) in zip(self.points, self.points[1:]):
            if env_steps <= right:
                return float(start + (end - start) * (env_steps - left) / (right - left))
        return float(self.points[-1][1])

    @property
    def beta(self):
        return self.value(self.env_steps)

    def advance(self, env_steps):
        beta = self.value(env_steps)
        if env_steps < self.env_steps:
            raise ValueError('Beta progress cannot rewind without checkpoint restoration')
        self.env_steps = env_steps
        return beta

    def save(self):
        return dict(protocol=self.protocol, initial_beta=self.initial_beta,
                    points=copy.deepcopy(self.points), env_steps=self.env_steps, beta=self.beta)

    def load(self, state, env_steps):
        if (state['protocol'] != self.protocol or state['initial_beta'] != self.initial_beta
                or state['points'] != self.points or state['env_steps'] != env_steps
                or state['beta'] != self.value(env_steps)):
            raise ValueError('Beta schedule differs from the committed Env boundary or configuration')
        self.env_steps = env_steps
