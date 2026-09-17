"""Actual sequential UCB/Gaussian-TS trajectories, accelerated with Numba.

The JIT and Python reference consume identical PCG64 streams. No certificate
or expected count is substituted for learner decisions.
"""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def trajectory(counts, sums, rewards, offsets, T, learner, attack, rng,
               sigma=0.5, phase=0, phase_two=-1,
               target_mean=-1.0):
    # learner: 0 UCB, 1 Gaussian TS; attack: 0 none, 1 clipped, 2 two-phase.
    # -1 retains the empirical reward source and its original random stream.
    if not math.isfinite(target_mean) or (target_mean != -1.0 and not 0.0 <= target_mean <= 1.0):
        raise ValueError('target_mean must be finite and in [0, 1]')
    K = len(counts)
    T0 = int(counts.sum())
    online = np.zeros(K, dtype=np.int64)
    modifications = 0
    magnitude = 0.0
    if phase_two < 0:
        phase_two = phase  # Backward-compatible legacy equal-phase calls.
    phase1_counts = np.zeros(K, dtype=np.int64)
    phase2_counts = np.zeros(K, dtype=np.int64)
    post_counts = np.zeros(K, dtype=np.int64)
    for step in range(T - T0):
        t = T0 + step + 1
        best = -math.inf
        arm = 0
        if counts[K - 1] == 0:
            # Online initialization: target first, then other arms in order.
            arm = K - 1
        else:
            unobserved = -1
            for i in range(K):
                if counts[i] == 0:
                    unobserved = i
                    break
            if unobserved >= 0:
                arm = unobserved
            else:
                for i in range(K):
                    value = sums[i] / counts[i]
                    if learner == 0:
                        value += 3.0 * sigma * math.sqrt(math.log(t) / counts[i])
                    else:
                        value += rng.normal() / math.sqrt(counts[i])
                    if value > best:  # first maximum, matching numpy.argmax
                        best, arm = value, i
        if target_mean >= 0.0 and arm == K - 1:
            reward0 = 1.0 if rng.random() < target_mean else 0.0
        else:
            reward0 = rewards[rng.integers(offsets[arm], offsets[arm + 1])]
        reward = reward0
        if attack == 1 and arm != K - 1:
            reward = 0.0
        elif attack == 2:
            if step < phase:
                reward = 0.0
            elif step < phase + phase_two:
                reward = 1.0 if arm == K - 1 else 0.0
        if reward != reward0:
            modifications += 1
            magnitude += abs(reward - reward0)
        counts[arm] += 1
        sums[arm] += reward
        online[arm] += 1
        if attack == 2:
            if step < phase:
                phase1_counts[arm] += 1
            elif step < phase + phase_two:
                phase2_counts[arm] += 1
            else:
                post_counts[arm] += 1
    return online, modifications, magnitude, phase1_counts, phase2_counts, post_counts


def simulate(clean_sum, allocation, reward_arrays, T, learner, attack, seed, phase=0,
             warm_start=True, phase_two=None, diagnostics=False, target_mean=None):
    """Run an attack, optionally replacing genuine target rewards by Bernoulli(p).

    The caller supplies the clean log; ``target_mean`` only changes online
    genuine rewards. ``None`` preserves empirical resampling and its RNG calls.
    """
    target_p = -1.0 if target_mean is None else float(target_mean)
    if target_mean is not None and (not math.isfinite(target_p) or not 0.0 <= target_p <= 1.0):
        raise ValueError('target_mean must be finite and in [0, 1]')
    K = len(clean_sum)
    n = np.asarray(allocation, dtype=np.int64)
    if not warm_start and (np.any(n) or np.any(clean_sum)):
        raise ValueError('Online-only baselines cannot receive offline observations')
    counts = np.full(K, 5 if warm_start else 0, dtype=np.int64) + n
    sums = np.asarray(clean_sum, dtype=float).copy()
    sums[-1] += n[-1]
    if counts.sum() >= T:
        raise ValueError('Allocation leaves no online horizon')
    rewards = np.concatenate(reward_arrays)
    offsets = np.r_[0, np.cumsum([len(a) for a in reward_arrays])]
    result = trajectory(counts, sums, rewards, offsets, T, int(learner == 'TS'),
                        attack, np.random.default_rng(seed), phase=phase,
                        phase_two=-1 if phase_two is None else phase_two,
                        target_mean=target_p)
    return result if diagnostics else result[:3]
