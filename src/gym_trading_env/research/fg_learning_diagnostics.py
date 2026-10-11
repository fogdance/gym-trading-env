"""Read-only FG learning diagnostics; no changes to trading or training rules."""
import numpy as np


def distribution(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return dict(count=0, mean=None, std=None, minimum=None, p05=None,
                    median=None, p95=None, maximum=None)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite diagnostic values')
    return dict(count=len(values), mean=float(values.mean()), std=float(values.std()),
                minimum=float(values.min()), p05=float(np.quantile(values, .05)),
                median=float(np.median(values)), p95=float(np.quantile(values, .95)),
                maximum=float(values.max()))


def add_confusion(target, predicted):
    target, predicted = np.broadcast_arrays(np.asarray(target, bool), np.asarray(predicted, bool))
    tp = int((target & predicted).sum()); fn = int((target & ~predicted).sum())
    fp = int((~target & predicted).sum()); tn = int((~target & ~predicted).sum())
    return dict(samples=target.size, legal_add=tp+fn, illegal_add=fp+tn,
                TP=tp, FN=fn, FP=fp, TN=tn,
                recall=tp/(tp+fn) if tp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                false_positive_rate=fp/(fp+tn) if fp+tn else None)


def completed_returns(reward, terminal, gamma):
    """V(obs[t]) target starts at reward[t+1]; unfinished tails stay censored.

    A financial terminal ends return accumulation; a nonterminal chunk/reset
    does not. This is an offline behavior-return diagnostic, not a new target.
    """
    reward = np.asarray(reward, np.float64)
    terminal = np.asarray(terminal, bool)
    if reward.shape != terminal.shape or reward.ndim != 1:
        raise ValueError('Expected aligned one-dimensional reward/terminal')
    values = np.full(len(reward), np.nan)
    complete = np.zeros(len(reward), bool)
    future = 0.; known = False
    for t in range(len(reward)-1, -1, -1):
        if terminal[t]:
            values[t] = 0.; complete[t] = True
            future = 0.; known = True
        elif known and t+1 < len(reward):
            future = reward[t+1] + gamma * future
            values[t] = future; complete[t] = True
    return values, complete


def imagined_return(reward, continuation, final_value):
    """r(next) + con(next)*V(next), with continuation already discounted."""
    reward, continuation = np.asarray(reward), np.asarray(continuation)
    if reward.shape != continuation.shape or reward.ndim != 2:
        raise ValueError('Expected [batch, horizon] predictions')
    value = np.asarray(final_value, np.float64).copy()
    for t in range(reward.shape[1]-1, -1, -1):
        value = reward[:, t] + continuation[:, t] * value
    return value
