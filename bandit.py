import math

import numpy as np

K = 10
target_arm = K
H = 100_000
delta = 0.05
xi = 0
r_l = 0.0
r_u = 1.0
R = r_u - r_l
reward_distribution = "Bernoulli(mu_i)"
sigma = 0.5
mu_non_target = [0.80, 0.72, 0.64, 0.56, 0.48, 0.40, 0.32, 0.24, 0.16]
mu_target = 0.02
mu = np.array(mu_non_target + [mu_target], dtype=float)
N0_i = 5
clean_offline_mode = "sample Bernoulli once with fixed seed"
seed = 2026
fake_reward_target = r_u
fake_reward_non_target = r_l
max_fixed_point_iters = 100
LEGality_tol = 1e-9
simulate_online = True
# Optional certified lower bound for controlled sensitivity studies. Production
# runners leave this as None and use the data-dependent confidence bound.
target_lower_bound_override = None


def _n0():
    return np.full(K, N0_i, dtype=int)


def _target_index():
    return target_arm - 1


def _s_t(T):
    return (T ** (2.0 / 3.0)) * ((K * math.log(T)) ** (1.0 / 3.0))


def _beta(N):
    return math.sqrt((2.0 * sigma * sigma / N) * math.log((math.pi ** 2) * K * (N ** 2) / (3.0 * delta)))


def _allocation_dict(n):
    return {f"n_{i + 1}": int(n[i]) for i in range(K)}


def _online_counts_dict(online_counts):
    return {f"N_on_{i + 1}": int(online_counts[i]) for i in range(K)}


def _attack_result(
    algorithm_name,
    T,
    T0,
    H_online,
    n,
    online_counts,
    z_star,
    mu_minus_K,
    epsilon,
    legality_check,
    fixed_point_trace=None,
    status="ok",
):
    cost_n = int(np.sum(n))
    s_t = _s_t(T)
    target_index = _target_index()
    if H_online > 0 and online_counts is not None:
        target_ratio = float(online_counts[target_index] / H_online)
        non_target_pulls = int(H_online - online_counts[target_index])
        online_counts_out = _online_counts_dict(online_counts)
    else:
        target_ratio = None
        non_target_pulls = None
        online_counts_out = {f"N_on_{i + 1}": None for i in range(K)}

    return {
        "algorithm": algorithm_name,
        "算法": algorithm_name,
        "status": status,
        "T": int(T),
        "T0": int(T0),
        "H": int(H_online),
        "z_star": None if z_star is None else float(z_star),
        "mu_minus_K": None if mu_minus_K is None else float(mu_minus_K),
        "epsilon": None if epsilon is None else float(epsilon),
        "Cost_n": cost_n,
        "target_online_ratio": target_ratio,
        "non_target_online_pulls": non_target_pulls,
        "S_T": float(s_t),
        "normalized_cost": float(cost_n / s_t),
        "allocation": _allocation_dict(n),
        "online_counts": online_counts_out,
        "fixed_point_trace": [] if fixed_point_trace is None else fixed_point_trace,
        "legality_check": legality_check,
    }


def _clean_result(algorithm_name, T, online_counts):
    n = np.zeros(K, dtype=int)
    T0 = int(np.sum(_n0()))
    H_online = int(T - T0)
    return _attack_result(
        algorithm_name=algorithm_name,
        T=T,
        T0=T0,
        H_online=H_online,
        n=n,
        online_counts=online_counts,
        z_star=None,
        mu_minus_K=None,
        epsilon=None,
        legality_check="clean_no_attack",
        status="ok",
    )


def _infeasible_result(algorithm_name, T, n, z_star, mu_minus_K, epsilon):
    T0 = int(np.sum(_n0()) + np.sum(n))
    H_online = int(T - T0)
    return _attack_result(
        algorithm_name=algorithm_name,
        T=T,
        T0=T0,
        H_online=H_online,
        n=n,
        online_counts=None,
        z_star=z_star,
        mu_minus_K=mu_minus_K,
        epsilon=epsilon,
        legality_check="infeasible_H_le_0",
        status="infeasible",
    )


def _certificate_online_counts(H_online):
    online_counts = np.zeros(K, dtype=int)
    online_counts[_target_index()] = int(H_online)
    return online_counts


def _ucb_counts_for_T(clean_mean, T_design):
    target_index = _target_index()
    N0 = _n0()
    mu_minus_K = (
        max(r_l, clean_mean[target_index] - 2.0 * _beta(N0[target_index]))
        if target_lower_bound_override is None
        else float(target_lower_bound_override)
    )
    epsilon = mu_minus_K - r_l
    logT = math.log(T_design)
    b = 3.0 * sigma * math.sqrt(logT / T_design)
    g_infty = r_l - b
    z_low = max(mu_minus_K, g_infty + 1e-12)
    z_high = r_u - 1e-12

    if not (z_low < z_high):
        raise RuntimeError(f"UCB feasible z interval is empty: z_low={z_low}, z_high={z_high}")

    def eta_i(i, z):
        def g_i(x):
            denom = N0[i] + x
            return clean_mean[i] - x * (clean_mean[i] - r_l) / denom + 3.0 * sigma * math.sqrt(logT / denom) - b

        if g_i(0.0) <= z:
            return 0.0
        hi = 1.0
        while g_i(hi) > z:
            hi *= 2.0
            if hi > 1e30:
                raise RuntimeError(f"UCB eta_i search exploded for arm {i + 1}, z={z}")
        lo = 0.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if g_i(mid) <= z:
                hi = mid
            else:
                lo = mid
        return hi

    def h(z):
        return T_design * max(0.0, z - mu_minus_K) / (r_u - z)

    def objective(z):
        return h(z) + sum(eta_i(i, z) for i in range(K) if i != target_index)

    grid = np.linspace(z_low, z_high, 301)
    vals = np.array([objective(float(z)) for z in grid])
    best_pos = int(np.argmin(vals))
    a = float(grid[max(0, best_pos - 1)])
    c = float(grid[min(len(grid) - 1, best_pos + 1)])
    if abs(a - c) < 1e-15:
        a, c = z_low, z_high

    gr = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = c - gr * (c - a)
    x2 = a + gr * (c - a)
    f1 = objective(x1)
    f2 = objective(x2)
    for _ in range(80):
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + gr * (c - a)
            f2 = objective(x2)
        else:
            c = x2
            x2 = x1
            f2 = f1
            x1 = c - gr * (c - a)
            f1 = objective(x1)

    z_star = 0.5 * (a + c)
    n = np.zeros(K, dtype=int)
    n[target_index] = int(math.ceil(h(z_star)))
    for i in range(K):
        if i != target_index:
            n[i] = int(math.ceil(eta_i(i, z_star))) + 1
    return n, z_star, mu_minus_K, epsilon


def _check_ucb_cutoffs(clean_mean, T, n, z_star, mu_minus_K):
    target_index = _target_index()
    N0 = _n0()
    logT = math.log(T)
    b = 3.0 * sigma * math.sqrt(logT / T)
    target_lhs = (n[target_index] * r_u + T * mu_minus_K) / (n[target_index] + T)
    if target_lhs + LEGality_tol < z_star:
        raise RuntimeError(f"UCB target cutoff check failed: lhs={target_lhs}, z_star={z_star}")

    for i in range(K):
        if i == target_index:
            continue
        x = float(n[i])
        denom = N0[i] + x
        g_val = clean_mean[i] - x * (clean_mean[i] - r_l) / denom + 3.0 * sigma * math.sqrt(logT / denom) - b
        if g_val >= z_star + LEGality_tol:
            raise RuntimeError(f"UCB non-target cutoff check failed: arm={i + 1}, g_i(n_i)={g_val}, z_star={z_star}")


def UCB_fixed_T(clean_sum, clean_mean, T):
    T = int(T)
    n, z_star, mu_minus_K, epsilon = _ucb_counts_for_T(clean_mean, T)
    T0 = int(np.sum(_n0()) + np.sum(n))
    H_online = int(T - T0)
    if H_online <= 0:
        return _infeasible_result("UCB appendix", T, n, z_star, mu_minus_K, epsilon)

    _check_ucb_cutoffs(clean_mean, T, n, z_star, mu_minus_K)
    if not simulate_online:
        online_counts = _certificate_online_counts(H_online)
        return _attack_result("UCB appendix", T, T0, H_online, n, online_counts, z_star, mu_minus_K, epsilon, "certificate")
    target_index = _target_index()
    counts = _n0().astype(float) + n.astype(float)
    sums = clean_sum.astype(float) + n.astype(float) * fake_reward_non_target
    sums[target_index] = clean_sum[target_index] + n[target_index] * fake_reward_target
    online_counts = np.zeros(K, dtype=int)
    rng = np.random.default_rng(seed + 101 + T)
    for step in range(H_online):
        t = T0 + 1 + step
        muhat = sums / counts
        index = muhat + 3.0 * sigma * np.sqrt(math.log(t) / counts)
        arm = int(np.argmax(index))
        reward = 1.0 if rng.random() < mu[arm] else 0.0
        counts[arm] += 1.0
        sums[arm] += reward
        online_counts[arm] += 1

    return _attack_result("UCB appendix", T, T0, H_online, n, online_counts, z_star, mu_minus_K, epsilon, "passed")


def _ucb_target_min_index(nK, non_target_cost, T, mu_minus_K):
    N0 = _n0()
    target_index = _target_index()
    T0 = int(np.sum(N0) + non_target_cost + nK)
    H_online = int(T - T0)
    if H_online <= 0:
        return -math.inf

    m_min = float(N0[target_index])
    m_max = float(N0[target_index] + H_online - 1)
    candidates = {m_min, m_max}
    if m_max > m_min:
        for frac in np.linspace(0.0, 1.0, 33):
            candidates.add(m_min + frac * (m_max - m_min))

    values = []
    for m in candidates:
        t = T0 + int(round(m - m_min)) + 1
        denom = nK + m
        mean_lb = (nK * r_u + m * mu_minus_K) / denom
        values.append(mean_lb + 3.0 * sigma * math.sqrt(math.log(max(t, 2)) / denom))
    return min(values)


def _ucb_direct_counts_for_T(clean_mean, T_design):
    target_index = _target_index()
    N0 = _n0()
    mu_minus_K = (
        max(r_l, clean_mean[target_index] - 2.0 * _beta(N0[target_index]))
        if target_lower_bound_override is None
        else float(target_lower_bound_override)
    )
    epsilon = mu_minus_K - r_l
    logT = math.log(T_design)

    def eta_i_direct(i, z):
        def index_i(x):
            denom = N0[i] + x
            mean = clean_mean[i] - x * (clean_mean[i] - r_l) / denom
            return mean + 3.0 * sigma * math.sqrt(logT / denom)

        if index_i(0.0) <= z:
            return 0.0
        hi = 1.0
        while index_i(hi) > z:
            hi *= 2.0
            if hi > 1e30:
                return math.inf
        lo = 0.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if index_i(mid) <= z:
                hi = mid
            else:
                lo = mid
        return hi

    def counts_at_z(z):
        n = np.zeros(K, dtype=int)
        for i in range(K):
            if i != target_index:
                eta = eta_i_direct(i, z)
                if not math.isfinite(eta) or eta >= T_design:
                    return n, math.inf
                n[i] = int(math.ceil(eta))
        non_target_cost = int(np.sum(n))
        if T_design - int(np.sum(N0)) - non_target_cost <= 0:
            n[target_index] = int(T_design - int(np.sum(N0)) - non_target_cost)
            return n, math.inf

        def ok(nK):
            return _ucb_target_min_index(nK, non_target_cost, T_design, mu_minus_K) > z + LEGality_tol

        if ok(0):
            n[target_index] = 0
            return n, int(np.sum(n))
        hi = 1
        while not ok(hi):
            hi *= 2
            if int(np.sum(N0)) + non_target_cost + hi >= T_design:
                n[target_index] = int(max(hi, T_design - int(np.sum(N0)) - non_target_cost))
                return n, math.inf
        lo = 0
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if ok(mid):
                hi = mid
            else:
                lo = mid
        n[target_index] = int(hi)
        return n, int(np.sum(n))

    z_low = r_l + 1e-10
    z_high = r_u + 3.0 * sigma * math.sqrt(logT / max(1, int(np.sum(N0))))
    grid = np.linspace(z_low, z_high, 401)
    best_n = None
    best_cost = math.inf
    best_z = None
    for z in grid:
        n, cost = counts_at_z(float(z))
        if cost < best_cost:
            best_n = n
            best_cost = cost
            best_z = float(z)

    if best_n is None or not math.isfinite(best_cost):
        n = np.zeros(K, dtype=int)
        return n, z_low, mu_minus_K, epsilon

    left = max(z_low, best_z - (z_high - z_low) / 400.0)
    right = min(z_high, best_z + (z_high - z_low) / 400.0)
    for z in np.linspace(left, right, 101):
        n, cost = counts_at_z(float(z))
        if cost < best_cost:
            best_n = n
            best_cost = cost
            best_z = float(z)
    return best_n, best_z, mu_minus_K, epsilon


def _check_ucb_direct_cutoffs(clean_mean, T, n, z_star, mu_minus_K):
    target_index = _target_index()
    N0 = _n0()
    T0 = int(np.sum(N0) + np.sum(n))
    H_online = int(T - T0)
    if H_online <= 0:
        raise RuntimeError("UCB direct allocation is infeasible.")

    target_min = _ucb_target_min_index(int(n[target_index]), int(np.sum(n) - n[target_index]), T, mu_minus_K)
    if target_min <= z_star + LEGality_tol:
        raise RuntimeError(f"UCB direct target check failed: target_min={target_min}, z_star={z_star}")

    for i in range(K):
        if i == target_index:
            continue
        denom = N0[i] + float(n[i])
        mean = clean_mean[i] - float(n[i]) * (clean_mean[i] - r_l) / denom
        index_i = mean + 3.0 * sigma * math.sqrt(math.log(T) / denom)
        if index_i >= z_star + LEGality_tol:
            raise RuntimeError(f"UCB direct non-target check failed: arm={i + 1}, index={index_i}, z_star={z_star}")


def UCB_direct_fixed_T(clean_sum, clean_mean, T):
    T = int(T)
    n, z_star, mu_minus_K, epsilon = _ucb_direct_counts_for_T(clean_mean, T)
    T0 = int(np.sum(_n0()) + np.sum(n))
    H_online = int(T - T0)
    if H_online <= 0:
        return _infeasible_result("UCB direct", T, n, z_star, mu_minus_K, epsilon)

    _check_ucb_direct_cutoffs(clean_mean, T, n, z_star, mu_minus_K)
    if not simulate_online:
        online_counts = _certificate_online_counts(H_online)
        return _attack_result("UCB direct", T, T0, H_online, n, online_counts, z_star, mu_minus_K, epsilon, "certificate")
    target_index = _target_index()
    counts = _n0().astype(float) + n.astype(float)
    sums = clean_sum.astype(float) + n.astype(float) * fake_reward_non_target
    sums[target_index] = clean_sum[target_index] + n[target_index] * fake_reward_target
    online_counts = np.zeros(K, dtype=int)
    rng = np.random.default_rng(seed + 111 + T)
    for step in range(H_online):
        t = T0 + 1 + step
        muhat = sums / counts
        index = muhat + 3.0 * sigma * np.sqrt(math.log(t) / counts)
        arm = int(np.argmax(index))
        reward = 1.0 if rng.random() < mu[arm] else 0.0
        counts[arm] += 1.0
        sums[arm] += reward
        online_counts[arm] += 1

    return _attack_result("UCB direct", T, T0, H_online, n, online_counts, z_star, mu_minus_K, epsilon, "passed")


def _ts_counts_for_T(clean_mean, T_design):
    target_index = _target_index()
    N0 = _n0()
    mu_minus_K = (
        max(r_l, clean_mean[target_index] - 2.0 * _beta(N0[target_index]))
        if target_lower_bound_override is None
        else float(target_lower_bound_override)
    )
    epsilon = mu_minus_K - r_l
    gamma_T = math.sqrt(2.0 * math.log((math.pi ** 2) * K * (T_design ** 2) / (3.0 * delta)))
    z_low = r_l + 1e-9
    z_high = r_u - 1e-12

    if not (z_low < z_high):
        raise RuntimeError(f"TS feasible z interval is empty: z_low={z_low}, z_high={z_high}")

    def eta_TS_i(i, z):
        def g_TS_i(x):
            denom = N0[i] + x
            return clean_mean[i] - x * (clean_mean[i] - r_l) / denom + gamma_T / math.sqrt(denom)

        if g_TS_i(0.0) <= z:
            return 0.0
        hi = 1.0
        while g_TS_i(hi) > z:
            hi *= 2.0
            if hi > 1e30:
                raise RuntimeError(f"TS eta_TS_i search exploded for arm {i + 1}, z={z}")
        lo = 0.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if g_TS_i(mid) <= z:
                hi = mid
            else:
                lo = mid
        return hi

    def h_TS(z):
        N0K = float(N0[target_index])
        T_float = float(T_design)
        A = r_u - mu_minus_K

        def lower_envelope(x, m):
            denom = x + m
            return (x * r_u + m * mu_minus_K) / denom - gamma_T / math.sqrt(denom)

        def min_lower_envelope_over_m(x):
            candidates = [N0K, T_float]
            if x > 0.0 and A > 0.0:
                y_star = (2.0 * x * A / gamma_T) ** 2
                m_star = y_star - x
                if N0K <= m_star <= T_float:
                    candidates.append(m_star)
            return min(lower_envelope(x, m) for m in candidates)

        def ok(x):
            return min_lower_envelope_over_m(x) >= z

        if ok(0.0):
            return 0.0
        hi = 1.0
        while not ok(hi):
            hi *= 2.0
            if hi > 1e30:
                raise RuntimeError(f"TS h_TS search exploded for z={z}")
        lo = 0.0
        for _ in range(90):
            mid = 0.5 * (lo + hi)
            if ok(mid):
                hi = mid
            else:
                lo = mid
        return hi

    def objective(z):
        return h_TS(z) + sum(eta_TS_i(i, z) for i in range(K) if i != target_index)

    grid = np.linspace(z_low, z_high, 301)
    vals = np.array([objective(float(z)) for z in grid])
    best_pos = int(np.argmin(vals))
    a = float(grid[max(0, best_pos - 1)])
    c = float(grid[min(len(grid) - 1, best_pos + 1)])
    if abs(a - c) < 1e-15:
        a, c = z_low, z_high
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = c - gr * (c - a)
    x2 = a + gr * (c - a)
    f1 = objective(x1)
    f2 = objective(x2)
    for _ in range(80):
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + gr * (c - a)
            f2 = objective(x2)
        else:
            c = x2
            x2 = x1
            f2 = f1
            x1 = c - gr * (c - a)
            f1 = objective(x1)

    z_star = 0.5 * (a + c)
    n = np.zeros(K, dtype=int)
    n[target_index] = int(math.ceil(h_TS(z_star)))
    for i in range(K):
        if i != target_index:
            n[i] = int(math.ceil(eta_TS_i(i, z_star))) + 1
    return n, z_star, mu_minus_K, epsilon


def _check_ts_cutoffs(clean_mean, T, n, z_star, mu_minus_K):
    target_index = _target_index()
    N0 = _n0()
    gamma_T = math.sqrt(2.0 * math.log((math.pi ** 2) * K * (T ** 2) / (3.0 * delta)))
    N0K = float(N0[target_index])
    T_float = float(T)
    A = r_u - mu_minus_K
    xK = float(n[target_index])

    def target_lower_envelope_for_check(x, m):
        denom = x + m
        return (x * r_u + m * mu_minus_K) / denom - gamma_T / math.sqrt(denom)

    candidate_m = [N0K, T_float]
    if xK > 0.0 and A > 0.0:
        y_star = (2.0 * xK * A / gamma_T) ** 2
        m_star = y_star - xK
        if N0K <= m_star <= T_float:
            candidate_m.append(m_star)
    min_target_lhs = min(target_lower_envelope_for_check(xK, m) for m in candidate_m)
    if min_target_lhs + LEGality_tol < z_star:
        raise RuntimeError(f"TS target cutoff check failed: min_lhs={min_target_lhs}, z_star={z_star}, candidate_m={candidate_m}")

    for i in range(K):
        if i == target_index:
            continue
        x = float(n[i])
        denom = N0[i] + x
        g_val = clean_mean[i] - x * (clean_mean[i] - r_l) / denom + gamma_T / math.sqrt(denom)
        if g_val >= z_star + LEGality_tol:
            raise RuntimeError(f"TS non-target cutoff check failed: arm={i + 1}, g_i_TS(n_i)={g_val}, z_star={z_star}")


def TS_fixed_T(clean_sum, clean_mean, T):
    T = int(T)
    n, z_star, mu_minus_K, epsilon = _ts_counts_for_T(clean_mean, T)
    T0 = int(np.sum(_n0()) + np.sum(n))
    H_online = int(T - T0)
    if H_online <= 0:
        return _infeasible_result("TS appendix", T, n, z_star, mu_minus_K, epsilon)

    _check_ts_cutoffs(clean_mean, T, n, z_star, mu_minus_K)
    if not simulate_online:
        online_counts = _certificate_online_counts(H_online)
        return _attack_result("TS appendix", T, T0, H_online, n, online_counts, z_star, mu_minus_K, epsilon, "certificate")
    target_index = _target_index()
    counts = _n0().astype(float) + n.astype(float)
    sums = clean_sum.astype(float) + n.astype(float) * fake_reward_non_target
    sums[target_index] = clean_sum[target_index] + n[target_index] * fake_reward_target
    online_counts = np.zeros(K, dtype=int)
    rng = np.random.default_rng(seed + 202 + T)
    for _ in range(H_online):
        muhat = sums / counts
        samples = rng.normal(loc=muhat, scale=1.0 / np.sqrt(counts))
        arm = int(np.argmax(samples))
        reward = 1.0 if rng.random() < mu[arm] else 0.0
        counts[arm] += 1.0
        sums[arm] += reward
        online_counts[arm] += 1

    return _attack_result("TS appendix", T, T0, H_online, n, online_counts, z_star, mu_minus_K, epsilon, "passed")


def _ts_target_min_value(nK, non_target_cost, T, mu_minus_K):
    N0 = _n0()
    target_index = _target_index()
    T0 = int(np.sum(N0) + non_target_cost + nK)
    H_online = int(T - T0)
    if H_online <= 0:
        return -math.inf

    gamma_T = math.sqrt(2.0 * math.log((math.pi ** 2) * K * (T ** 2) / (3.0 * delta)))
    m_min = float(N0[target_index])
    m_max = float(N0[target_index] + H_online - 1)
    A = r_u - mu_minus_K
    candidates = {m_min, m_max}
    if nK > 0 and A > 0.0:
        y_star = (2.0 * nK * A / gamma_T) ** 2
        m_star = y_star - nK
        if m_min <= m_star <= m_max:
            candidates.add(float(m_star))
    if m_max > m_min:
        for frac in np.linspace(0.0, 1.0, 33):
            candidates.add(m_min + frac * (m_max - m_min))

    values = []
    for m in candidates:
        denom = nK + m
        mean_lb = (nK * r_u + m * mu_minus_K) / denom
        values.append(mean_lb - gamma_T / math.sqrt(denom))
    return min(values)


def _ts_direct_counts_for_T(clean_mean, T_design):
    target_index = _target_index()
    N0 = _n0()
    mu_minus_K = (
        max(r_l, clean_mean[target_index] - 2.0 * _beta(N0[target_index]))
        if target_lower_bound_override is None
        else float(target_lower_bound_override)
    )
    epsilon = mu_minus_K - r_l
    gamma_T = math.sqrt(2.0 * math.log((math.pi ** 2) * K * (T_design ** 2) / (3.0 * delta)))

    def eta_i_direct(i, z):
        def upper_i(x):
            denom = N0[i] + x
            mean = clean_mean[i] - x * (clean_mean[i] - r_l) / denom
            return mean + gamma_T / math.sqrt(denom)

        if upper_i(0.0) <= z:
            return 0.0
        hi = 1.0
        while upper_i(hi) > z:
            hi *= 2.0
            if hi > 1e30:
                return math.inf
        lo = 0.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if upper_i(mid) <= z:
                hi = mid
            else:
                lo = mid
        return hi

    def counts_at_z(z):
        n = np.zeros(K, dtype=int)
        for i in range(K):
            if i != target_index:
                eta = eta_i_direct(i, z)
                if not math.isfinite(eta) or eta >= T_design:
                    return n, math.inf
                n[i] = int(math.ceil(eta))
        non_target_cost = int(np.sum(n))
        if T_design - int(np.sum(N0)) - non_target_cost <= 0:
            n[target_index] = int(T_design - int(np.sum(N0)) - non_target_cost)
            return n, math.inf

        def ok(nK):
            return _ts_target_min_value(nK, non_target_cost, T_design, mu_minus_K) > z + LEGality_tol

        if ok(0):
            n[target_index] = 0
            return n, int(np.sum(n))
        hi = 1
        while not ok(hi):
            hi *= 2
            if int(np.sum(N0)) + non_target_cost + hi >= T_design:
                n[target_index] = int(max(hi, T_design - int(np.sum(N0)) - non_target_cost))
                return n, math.inf
        lo = 0
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if ok(mid):
                hi = mid
            else:
                lo = mid
        n[target_index] = int(hi)
        return n, int(np.sum(n))

    z_low = r_l + 1e-10
    z_high = r_u - 1e-10
    grid = np.linspace(z_low, z_high, 401)
    best_n = None
    best_cost = math.inf
    best_z = None
    for z in grid:
        n, cost = counts_at_z(float(z))
        if cost < best_cost:
            best_n = n
            best_cost = cost
            best_z = float(z)

    if best_n is None or not math.isfinite(best_cost):
        n = np.zeros(K, dtype=int)
        return n, z_low, mu_minus_K, epsilon

    left = max(z_low, best_z - (z_high - z_low) / 400.0)
    right = min(z_high, best_z + (z_high - z_low) / 400.0)
    for z in np.linspace(left, right, 101):
        n, cost = counts_at_z(float(z))
        if cost < best_cost:
            best_n = n
            best_cost = cost
            best_z = float(z)
    return best_n, best_z, mu_minus_K, epsilon


def _check_ts_direct_cutoffs(clean_mean, T, n, z_star, mu_minus_K):
    target_index = _target_index()
    N0 = _n0()
    gamma_T = math.sqrt(2.0 * math.log((math.pi ** 2) * K * (T ** 2) / (3.0 * delta)))
    target_min = _ts_target_min_value(int(n[target_index]), int(np.sum(n) - n[target_index]), T, mu_minus_K)
    if target_min <= z_star + LEGality_tol:
        raise RuntimeError(f"TS direct target check failed: target_min={target_min}, z_star={z_star}")

    for i in range(K):
        if i == target_index:
            continue
        denom = N0[i] + float(n[i])
        mean = clean_mean[i] - float(n[i]) * (clean_mean[i] - r_l) / denom
        upper_i = mean + gamma_T / math.sqrt(denom)
        if upper_i >= z_star + LEGality_tol:
            raise RuntimeError(f"TS direct non-target check failed: arm={i + 1}, upper={upper_i}, z_star={z_star}")


def TS_direct_fixed_T(clean_sum, clean_mean, T):
    T = int(T)
    n, z_star, mu_minus_K, epsilon = _ts_direct_counts_for_T(clean_mean, T)
    T0 = int(np.sum(_n0()) + np.sum(n))
    H_online = int(T - T0)
    if H_online <= 0:
        return _infeasible_result("TS direct", T, n, z_star, mu_minus_K, epsilon)

    _check_ts_direct_cutoffs(clean_mean, T, n, z_star, mu_minus_K)
    if not simulate_online:
        online_counts = _certificate_online_counts(H_online)
        return _attack_result("TS direct", T, T0, H_online, n, online_counts, z_star, mu_minus_K, epsilon, "certificate")
    target_index = _target_index()
    counts = _n0().astype(float) + n.astype(float)
    sums = clean_sum.astype(float) + n.astype(float) * fake_reward_non_target
    sums[target_index] = clean_sum[target_index] + n[target_index] * fake_reward_target
    online_counts = np.zeros(K, dtype=int)
    rng = np.random.default_rng(seed + 212 + T)
    for _ in range(H_online):
        muhat = sums / counts
        samples = rng.normal(loc=muhat, scale=1.0 / np.sqrt(counts))
        arm = int(np.argmax(samples))
        reward = 1.0 if rng.random() < mu[arm] else 0.0
        counts[arm] += 1.0
        sums[arm] += reward
        online_counts[arm] += 1

    return _attack_result("TS direct", T, T0, H_online, n, online_counts, z_star, mu_minus_K, epsilon, "passed")


def simulate_clean_fixed_T(clean_sum, clean_mean, T, algorithm_name, rng_offset):
    del clean_mean
    T = int(T)
    N0 = _n0()
    T0 = int(np.sum(N0))
    H_online = int(T - T0)
    if H_online <= 0:
        return _attack_result(
            algorithm_name=algorithm_name,
            T=T,
            T0=T0,
            H_online=H_online,
            n=np.zeros(K, dtype=int),
            online_counts=None,
            z_star=None,
            mu_minus_K=None,
            epsilon=None,
            legality_check="infeasible_H_le_0",
            status="infeasible",
        )

    counts = N0.astype(float).copy()
    sums = clean_sum.astype(float).copy()
    online_counts = np.zeros(K, dtype=int)
    rng = np.random.default_rng(seed + rng_offset + T)
    for step in range(H_online):
        if algorithm_name.startswith("UCB"):
            t = T0 + 1 + step
            muhat = sums / counts
            index = muhat + 3.0 * sigma * np.sqrt(math.log(t) / counts)
            arm = int(np.argmax(index))
        else:
            muhat = sums / counts
            samples = rng.normal(loc=muhat, scale=1.0 / np.sqrt(counts))
            arm = int(np.argmax(samples))
        reward = 1.0 if rng.random() < mu[arm] else 0.0
        counts[arm] += 1.0
        sums[arm] += reward
        online_counts[arm] += 1

    return _clean_result(algorithm_name, T, online_counts)


def simulate_clean_fixed_T_grid(clean_sum, clean_mean, T_grid, algorithm_name, rng_offset):
    del clean_mean
    T_values = sorted(int(T) for T in T_grid)
    N0 = _n0()
    T0 = int(np.sum(N0))
    max_T = max(T_values)
    max_H_online = int(max_T - T0)

    results = {}
    if max_H_online <= 0:
        for T in T_values:
            results[T] = _attack_result(
                algorithm_name=algorithm_name,
                T=T,
                T0=T0,
                H_online=int(T - T0),
                n=np.zeros(K, dtype=int),
                online_counts=None,
                z_star=None,
                mu_minus_K=None,
                epsilon=None,
                legality_check="infeasible_H_le_0",
                status="infeasible",
            )
        return results

    checkpoint_by_H = {int(T - T0): T for T in T_values if int(T - T0) > 0}
    counts = N0.astype(float).copy()
    sums = clean_sum.astype(float).copy()
    online_counts = np.zeros(K, dtype=int)
    rng = np.random.default_rng(seed + rng_offset + max_T)

    for step in range(max_H_online):
        if algorithm_name.startswith("UCB"):
            t = T0 + 1 + step
            muhat = sums / counts
            index = muhat + 3.0 * sigma * np.sqrt(math.log(t) / counts)
            arm = int(np.argmax(index))
        else:
            muhat = sums / counts
            samples = rng.normal(loc=muhat, scale=1.0 / np.sqrt(counts))
            arm = int(np.argmax(samples))
        reward = 1.0 if rng.random() < mu[arm] else 0.0
        counts[arm] += 1.0
        sums[arm] += reward
        online_counts[arm] += 1

        H_done = step + 1
        if H_done in checkpoint_by_H:
            T = checkpoint_by_H[H_done]
            results[T] = _clean_result(algorithm_name, T, online_counts.copy())

    for T in T_values:
        if T not in results:
            results[T] = _attack_result(
                algorithm_name=algorithm_name,
                T=T,
                T0=T0,
                H_online=int(T - T0),
                n=np.zeros(K, dtype=int),
                online_counts=None,
                z_star=None,
                mu_minus_K=None,
                epsilon=None,
                legality_check="infeasible_H_le_0",
                status="infeasible",
            )
    return results


def _fixed_point_T_for_H(clean_mean, counts_fn):
    T_design = H + int(np.sum(_n0()))
    fixed_point_trace = []
    for _ in range(max_fixed_point_iters):
        n, z_star, _, _ = counts_fn(clean_mean, T_design)
        T_new = H + int(np.sum(_n0())) + int(np.sum(n))
        fixed_point_trace.append((T_design, T_new, int(np.sum(n)), float(z_star)))
        if T_new == T_design:
            return T_design, fixed_point_trace
        T_design = T_new
    raise RuntimeError("fixed point did not converge: final T is not self-consistent.")


def UCB(clean_sum, clean_mean):
    T_design, trace = _fixed_point_T_for_H(clean_mean, _ucb_counts_for_T)
    result = UCB_fixed_T(clean_sum, clean_mean, T_design)
    result["algorithm"] = "UCB 被攻击"
    result["算法"] = "UCB 被攻击"
    result["fixed_point_trace"] = trace
    return result


def TS(clean_sum, clean_mean):
    T_design, trace = _fixed_point_T_for_H(clean_mean, _ts_counts_for_T)
    result = TS_fixed_T(clean_sum, clean_mean, T_design)
    result["algorithm"] = "TS 被攻击"
    result["算法"] = "TS 被攻击"
    result["fixed_point_trace"] = trace
    return result


def Main():
    target_index = _target_index()
    N0 = _n0()
    rng_clean = np.random.default_rng(seed)
    clean_rewards = [rng_clean.binomial(1, float(mu[i]), size=N0[i]).astype(float) for i in range(K)]
    clean_sum = np.array([arr.sum() for arr in clean_rewards], dtype=float)
    clean_mean = clean_sum / N0.astype(float)
    T_grid = [50_000, 100_000, 200_000, 400_000, 700_000, 1_000_000]

    print("=" * 88)
    print("Bounded Attack on Stochastic Warm-Start Bandits: Fixed-T Numerical Simulation")
    print("=" * 88)
    print(f"reward_distribution = {reward_distribution}")
    print(f"K = {K}, target_arm = {target_arm}, delta = {delta}, xi = {xi}")
    print(f"r_l = {r_l}, r_u = {r_u}, R = {R}, sigma = {sigma}")
    print(f"mu = {mu.tolist()}")
    print(f"N0_i = {N0_i}, clean_offline_mode = {clean_offline_mode}, seed = {seed}")
    print(f"clean_sum = {clean_sum.astype(int).tolist()}")
    print(f"muhat0 = {[round(x, 6) for x in clean_mean.tolist()]}")
    print("=" * 88)

    for T in T_grid:
        results = [
            UCB_fixed_T(clean_sum, clean_mean, T),
            TS_fixed_T(clean_sum, clean_mean, T),
            simulate_clean_fixed_T(clean_sum, clean_mean, T, "UCB clean", 303),
            simulate_clean_fixed_T(clean_sum, clean_mean, T, "TS clean", 404),
        ]
        print(f"\n[T = {T}]")
        for res in results:
            ratio = res["target_online_ratio"]
            ratio_text = "NA" if ratio is None else f"{ratio:.6f}"
            print(
                f"{res['algorithm']}: status={res['status']}, T0={res['T0']}, H={res['H']}, "
                f"Cost_n={res['Cost_n']}, target_online_ratio={ratio_text}, "
                f"non_target_online_pulls={res['non_target_online_pulls']}"
            )
            if res["algorithm"].startswith(("UCB appendix", "UCB direct", "TS appendix", "TS direct")) and res["status"] == "ok":
                assert res["T0"] + res["H"] == T
                assert res["Cost_n"] == sum(res["allocation"].values())
                assert res["target_online_ratio"] > 0.99
                assert res["non_target_online_pulls"] == 0
                assert res["online_counts"][f"N_on_{target_index + 1}"] == res["H"]


if __name__ == "__main__":
    Main()
