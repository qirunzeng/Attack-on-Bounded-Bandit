"""Original Xu learner models (paper Sections 5.1/5.3), no offline log.

UCB initializes arms in index order and uses sqrt(log(T)/N). TS samples from
Beta(1,1) priors immediately; genuine target rewards are Bernoulli(mu_K).
The vector corruption schedule is fixed before any actions are observed.
"""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def trajectory(rewards, offsets, T, is_ts, rng, C1, C2, mu):
    K = len(offsets)-1
    counts = np.zeros(K, dtype=np.int64)
    sums = np.zeros(K)
    stages = np.zeros((3,K), dtype=np.int64)
    modifications = 0
    magnitude = 0.
    for step in range(T):
        arm = 0
        if not is_ts and step<K:
            arm = step
        else:
            best = -math.inf
            for i in range(K):
                if is_ts:
                    value = rng.beta(sums[i]+1., counts[i]-sums[i]+1.)
                else:
                    value = sums[i]/counts[i]+math.sqrt(math.log(T)/counts[i])
                if value>best:
                    best, arm = value, i
        if is_ts and arm==K-1:
            raw = 1. if rng.random()<mu else 0.
        else:
            raw = rewards[rng.integers(offsets[arm],offsets[arm+1])]
        stage = 0 if step<C1 else 1 if step<C1+C2 else 2
        reward = 0. if stage==0 else (1. if arm==K-1 else 0.) if stage==1 else raw
        modifications += reward != raw
        magnitude += abs(reward-raw)
        counts[arm] += 1
        sums[arm] += reward
        stages[stage,arm] += 1
    return counts, modifications, magnitude, stages[0], stages[1], stages[2]


def simulate(reward_arrays, T, learner, seed, C1, C2, mu):
    if learner not in {'UCB','TS'} or C1<0 or C2<0 or C1+C2>=T:
        raise ValueError('Invalid original-setting simulation')
    if learner=='TS' and any(not np.isin(a,[0.,1.]).all() for a in reward_arrays[:-1]):
        raise ValueError('Original TS requires Bernoulli non-target rewards')
    rewards = np.concatenate(reward_arrays)
    offsets = np.r_[0,np.cumsum([len(a) for a in reward_arrays])]
    return trajectory(rewards, offsets, T, learner=='TS', np.random.default_rng(seed), C1, C2, mu)
