"""Accelerated Direct grid search for the paper's [0,1] environment.

Uses the same 4097-point grid and five 1025-point refinements as bandit.py.
The Gaussian target envelope is minimized over integer genuine counts exactly,
instead of repeatedly invoking a continuous numerical optimizer.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def ts_value(m, n, non, T, K, n0, mu, delta):
    t = K*n0 + non + n + m - n0 + 1
    gamma = math.sqrt(2*math.log(math.pi**2*K*t*t/(3*delta)))
    return (n + m*mu)/(n+m) - gamma/math.sqrt(n+m)


@njit(cache=True)
def target_min(n, non, T, K, n0, mu, ts, sigma, delta):
    last = T - (K-1)*n0 - non - n - 1
    if last < n0:
        return -math.inf
    if not ts:
        d = n+last
        return (n+last*mu)/d + 3*sigma*math.sqrt(math.log(T)/d)
    # f(d)=mu+n(1-mu)/d-gamma(d+c)/sqrt(d), d=n+m.
    # Its derivative has the sign of h(d)-n(1-mu), with
    # h=sqrt(d)*(gamma/2-2d/(gamma*(d+c))). h is strictly increasing:
    # 4 gamma^3 sqrt(d) h' = gamma^4-8 gamma^2 q+(16+8 gamma^2)q^2 > 0,
    # q=d/(d+c). Thus f is unimodal, including its integer restriction.
    lo, hi = n0, last
    while lo < hi:
        mid = (lo+hi)//2
        if ts_value(mid, n, non, T, K, n0, mu, delta) <= ts_value(mid+1, n, non, T, K, n0, mu, delta):
            hi = mid
        else:
            lo = mid+1
    best = min(ts_value(n0,n,non,T,K,n0,mu,delta), ts_value(last,n,non,T,K,n0,mu,delta))
    for m in range(max(n0,lo-3), min(last,lo+3)+1):
        best = min(best,ts_value(m,n,non,T,K,n0,mu,delta))
    return best


@njit(cache=True)
def evaluate(z, means, T, n0, mu, ts, sigma, delta):
    K = len(means)
    n = np.zeros(K,dtype=np.int64)
    max_n = T-K*n0-1
    bonus = math.sqrt(2*math.log(math.pi**2*K*T*T/(3*delta))) if ts else 3*sigma*math.sqrt(math.log(T))
    for i in range(K-1):
        lo,hi=0,max_n
        upper = n0*means[i]/(n0+hi)
        upper += bonus/math.sqrt(n0+hi) if ts else 3*sigma*math.sqrt(math.log(T)/(n0+hi))
        if upper > z:
            return n, False
        while lo < hi:
            mid=(lo+hi)//2
            upper = n0*means[i]/(n0+mid)
            upper += bonus/math.sqrt(n0+mid) if ts else 3*sigma*math.sqrt(math.log(T)/(n0+mid))
            if upper <= z:
                hi=mid
            else:
                lo=mid+1
        n[i]=lo
    non=int(n.sum())
    lo,hi=0,max_n-non
    if hi < 0 or target_min(hi,non,T,K,n0,mu,ts,sigma,delta) < z+1e-12:
        return n,False
    # Raising n improves the target envelope and shrinks its genuine-count
    # domain. For these parameters gamma(t)^2 > 4 and t >= n+m.
    while lo < hi:
        mid=(lo+hi)//2
        if target_min(mid,non,T,K,n0,mu,ts,sigma,delta) >= z+1e-12:
            hi=mid
        else:
            lo=mid+1
    n[-1]=lo
    return n,True


@njit(cache=True)
def search(means,T,n0,mu,ts,sigma=.5,delta=.05):
    K=len(means)
    left=0.
    right=target_min(T-K*n0-1,0,T,K,n0,mu,ts,sigma,delta)-1e-12
    best_n=np.zeros(K,dtype=np.int64)
    best_cost=T+1
    best_target=T+1
    best_z=math.inf
    stages=np.zeros((6,5))
    for stage in range(6):
        size=4097 if stage==0 else 1025
        grid=np.linspace(left,right,size)
        stage_cost=T+1
        stage_target=T+1
        stage_z=math.inf
        stage_index=-1
        for i in range(size):
            z=grid[i]
            n,ok=evaluate(z,means,T,n0,mu,ts,sigma,delta)
            if not ok:
                continue
            cost=int(n.sum())
            if (cost,n[-1],z) < (stage_cost,stage_target,stage_z):
                stage_cost,stage_target,stage_z=cost,n[-1],z
                stage_index=i
            if (cost,n[-1],z) < (best_cost,best_target,best_z):
                best_cost,best_target,best_z=cost,n[-1],z
                best_n=n.copy()
        stages[stage]=np.array([left,right,size,stage_cost,stage_index])
        if stage_index < 0:
            break
        left=grid[max(0,stage_index-1)]
        right=grid[min(size-1,stage_index+1)]
    return best_n,best_z,stages
