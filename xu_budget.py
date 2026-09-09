"""Xu et al. (2021), supplementary A.2/A.4, with integer rounding.

Use with their UCB index or Beta--Bernoulli TS in xu_simulation.py.
No constants or probability bounds are derived here. These are sufficient
constructions, not minimum successful budgets.
"""
import math

VERSION = 'xu2021-original-settings-v1'
SOURCE = 'https://proceedings.neurips.cc/paper_files/paper/2021/file/be315e7f05e9f13629031915fe87ad44-Supplemental.pdf'


def budget(learner, K, T, mu):
    if learner not in {'UCB', 'TS'} or K < 2 or T <= K:
        raise ValueError('Invalid learner, horizon or arm count')
    if not math.isfinite(mu) or not 0 <= mu < 1:
        raise ValueError('The cited formulas require 0 <= mu < 1')
    info = dict(version=VERSION, learner=learner, K=K, T=T, mu=mu,
                source=SOURCE, source_section='A.2' if learner=='UCB' else 'A.4',
                learner_setting='mean + sqrt(log(T)/N)' if learner=='UCB' else 'Beta(sum+1, N-sum+1)',
                target_feedback='deterministic mu_K' if learner=='UCB' else 'Bernoulli(mu_K)')
    if mu == 0:
        return dict(info, C1=None, C2=None, total=None, feasible=False, reason='undefined at mu_K=0')
    if learner == 'UCB':
        # A.2 uses equal C1/K counts. Round C1 upward to a multiple of K.
        C1 = K * math.ceil(math.log(T) / mu**2)
        C2 = math.ceil(mu / (1-mu) * C1)
    else:
        C1 = math.ceil(4 * math.log(T) / mu**2)
        spread = (K-1) * math.sqrt(C1 * math.log(T))
        n1, n2 = (C1-spread)/K, (C1+spread)/K
        info.update(n1=n1, n2=n2)
        if n1 <= 0:
            return dict(info, C1=C1, C2=None, total=None, feasible=False,
                        reason='A.4 phase-one lower count bound is nonpositive')
        beta = (n1+1)/(n2+1)
        n3 = 1/(beta*(1-2.**(1-K)))
        waiting = n3*(beta/(1+beta))**(1-K)*math.log(T)
        promotion = 2*mu/(1-mu)*C1
        C2 = math.ceil(waiting + promotion)
        info.update(beta=beta, n3=n3, waiting_term=waiting, promotion_term=promotion)
    total = C1+C2
    return dict(info, C1=C1, C2=C2, total=total, feasible=total<T,
                reason='' if total<T else 'original budget reaches/exceeds T')
