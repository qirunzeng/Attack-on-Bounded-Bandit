import math
import warnings

import numpy as np
from scipy.optimize import minimize_scalar

K = 10
target_arm = K
H = 100_000
delta = 0.05
xi = 0
r_l = 0.0
r_u = 1.0
R = r_u - r_l
reward_distribution = "Bernoulli(mu_i)"
reward_source = "bernoulli"
empirical_reward_arrays = None
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
DIRECT_EPSILON_NUM = 1e-12
DIRECT_COARSE_POINTS = 4097
DIRECT_REFINEMENT_POINTS = 1025
DIRECT_REFINEMENTS = 5
DIRECT_WIDTH_WARNING = 1e-10
simulate_online = True


def draw_environment_reward(rng, arm):
    """Draw an arm reward from the configured stochastic environment."""
    if reward_source == "empirical":
        if empirical_reward_arrays is None or len(empirical_reward_arrays) != K:
            raise RuntimeError("Empirical reward arrays are not configured for all K arms.")
        rewards = empirical_reward_arrays[arm]
        if len(rewards) == 0:
            raise RuntimeError(f"Empirical reward array for arm {arm} is empty.")
        return float(rewards[rng.integers(0, len(rewards))])
    if reward_source != "bernoulli":
        raise ValueError(f"Unknown reward_source={reward_source!r}.")
    # Previous Scheme A (kept as the Bernoulli fallback):
    # reward = 1.0 if rng.random() < mu[arm] else 0.0
    return 1.0 if rng.random() < mu[arm] else 0.0


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
    search_log=None,
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
        "search_log": {} if search_log is None else search_log,
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


def _infeasible_result(algorithm_name, T, n, z_star, mu_minus_K, epsilon, search_log=None, reason="infeasible_H_le_0"):
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
        legality_check=reason,
        search_log=search_log,
        status="infeasible",
    )


def _certificate_online_counts(H_online):
    online_counts = np.zeros(K, dtype=int)
    online_counts[_target_index()] = int(H_online)
    return online_counts


def _ucb_counts_for_T(clean_mean, T_design):
    target_index = _target_index()
    N0 = _n0()
    mu_minus_K = max(r_l, clean_mean[target_index] - 2.0 * _beta(N0[target_index]))
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
    n[target_index] = int(math.ceil(h(z_star))) + 1
    for i in range(K):
        if i != target_index:
            n[i] = int(math.ceil(eta_i(i, z_star)))
    return n, z_star, mu_minus_K, epsilon


def _check_ucb_cutoffs(clean_mean, T, n, z_star, mu_minus_K):
    target_index = _target_index()
    N0 = _n0()
    logT = math.log(T)
    b = 3.0 * sigma * math.sqrt(logT / T)
    target_lhs = (n[target_index] * r_u + T * mu_minus_K) / (n[target_index] + T)
    if target_lhs <= z_star:
        raise RuntimeError(f"UCB target cutoff check failed: lhs={target_lhs}, z_star={z_star}")

    for i in range(K):
        if i == target_index:
            continue
        x = float(n[i])
        denom = N0[i] + x
        g_val = clean_mean[i] - x * (clean_mean[i] - r_l) / denom + 3.0 * sigma * math.sqrt(logT / denom) - b
        if g_val > z_star:
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
        reward = draw_environment_reward(rng, arm)
        counts[arm] += 1.0
        sums[arm] += reward
        online_counts[arm] += 1

    return _attack_result("UCB appendix", T, T0, H_online, n, online_counts, z_star, mu_minus_K, epsilon, "passed")


def _predicate_has_monotonicity_violation(evaluations):
    """Return True if sampled False/True values violate a monotone predicate."""
    seen_true = False
    for _, value in sorted(evaluations.items()):
        if value:
            seen_true = True
        elif seen_true:
            return True
    return False


def _deterministic_segmented_first_true(predicate, hi, label, search_log):
    """Deterministic segment detection followed by exact local refinement."""
    segment_count = min(4096, hi + 1)
    segment_width = max(1, int(math.ceil((hi + 1) / segment_count)))
    event = {
        "predicate": label,
        "scan_low": 0,
        "scan_high": int(hi),
        "segments": int(segment_count),
        "local_refinement": None,
    }
    search_log["nonmonotone_fallbacks"].append(event)
    for segment_start in range(0, hi + 1, segment_width):
        segment_end = min(hi, segment_start + segment_width - 1)
        segment_has_true = any(predicate(value) for value in range(segment_start, segment_end + 1))
        if segment_has_true:
            event["local_refinement"] = {"low": int(segment_start), "high": int(segment_end)}
            for value in range(segment_start, segment_end + 1):
                if predicate(value):
                    return value
    return None


def _minimal_integer_by_bracketing(predicate, max_n, label, search_log):
    """Exponential bracket plus integer binary search with checked fallback."""
    max_n = int(max_n)
    search_log["integer_searches"] += 1
    if max_n < 0:
        return None

    evaluations = {}

    def checked(n):
        n = int(n)
        if n not in evaluations:
            evaluations[n] = bool(predicate(n))
        return evaluations[n]

    def audit_interval(audit_hi, candidate=None):
        audit_points = {int(round(x)) for x in np.linspace(0, audit_hi, min(33, audit_hi + 1))}
        if candidate is not None:
            audit_points.update(
                value for value in (candidate - 2, candidate - 1, candidate, candidate + 1, candidate + 2)
                if 0 <= value <= audit_hi
            )
        for value in sorted(audit_points):
            checked(value)

    if checked(0):
        return 0

    hi = 1
    while hi < max_n and not checked(hi):
        hi = min(max_n, hi * 2)
    if not checked(hi):
        audit_interval(hi)
        if _predicate_has_monotonicity_violation(evaluations):
            return _deterministic_segmented_first_true(predicate, hi, label, search_log)
        return None

    bracket_hi = hi
    lo = 0
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if checked(mid):
            hi = mid
        else:
            lo = mid

    audit_interval(bracket_hi, hi)
    if _predicate_has_monotonicity_violation(evaluations):
        return _deterministic_segmented_first_true(predicate, bracket_hi, label, search_log)
    return hi


def _direct_coarse_to_fine(evaluate_z, z_low, z_high, algorithm_name, search_log):
    """Run the prescribed 4097-point search and five 1025-point refinements."""
    if not (math.isfinite(z_low) and math.isfinite(z_high) and z_low < z_high):
        search_log["warnings"].append(f"empty initial z interval [{z_low}, {z_high}]")
        return None, None

    best = None
    left = float(z_low)
    right = float(z_high)
    stage_sizes = [DIRECT_COARSE_POINTS] + [DIRECT_REFINEMENT_POINTS] * DIRECT_REFINEMENTS
    for stage, point_count in enumerate(stage_sizes):
        grid = np.linspace(left, right, point_count)
        stage_best = None
        stage_best_index = None
        for grid_index, z_value in enumerate(grid):
            z = float(z_value)
            n = evaluate_z(z, search_log)
            if n is None:
                continue
            candidate = (int(np.sum(n)), int(n[_target_index()]), z, n.copy())
            if stage_best is None or candidate[:3] < stage_best[:3]:
                stage_best = candidate
                stage_best_index = grid_index
            if best is None or candidate[:3] < best[:3]:
                best = candidate

        search_log["z_stages"].append(
            {
                "stage": int(stage),
                "points": int(point_count),
                "left": left,
                "right": right,
                "width": right - left,
                "feasible": stage_best is not None,
            }
        )
        if stage_best is None:
            break

        lower_index = max(0, stage_best_index - 1)
        upper_index = min(point_count - 1, stage_best_index + 1)
        if lower_index == upper_index:
            break
        left = float(grid[lower_index])
        right = float(grid[upper_index])

    final_width = right - left
    search_log["final_interval"] = {"left": left, "right": right, "width": final_width}
    if final_width > DIRECT_WIDTH_WARNING:
        message = (
            f"{algorithm_name} final z interval width {final_width:.17g} exceeds "
            f"{DIRECT_WIDTH_WARNING:.1e}."
        )
        search_log["warnings"].append(message)
        warnings.warn(message, RuntimeWarning)
    if best is None:
        return None, None
    return best[3], best[2]


def _ucb_target_min_index(nK, non_target_cost, T, mu_minus_K):
    N0 = _n0()
    target_index = _target_index()
    T0 = int(np.sum(N0) + non_target_cost + nK)
    H_online = int(T - T0)
    if H_online <= 0:
        return -math.inf

    # Both terms are decreasing in the integer m on the specified interval, so
    # the exact minimum is attained at its final integer endpoint.
    m = int(N0[target_index] + H_online - 1)
    t_m = int(T0 + (m - N0[target_index]) + 1)
    denom = nK + m
    return (nK * r_u + m * mu_minus_K) / denom + 3.0 * sigma * math.sqrt(math.log(t_m) / denom)


def _new_direct_search_log(algorithm_name, z_low, z_high):
    return {
        "algorithm": algorithm_name,
        "epsilon_num": DIRECT_EPSILON_NUM,
        "initial_interval": {"left": float(z_low), "right": float(z_high)},
        "integer_searches": 0,
        "nonmonotone_fallbacks": [],
        "z_stages": [],
        "warnings": [],
    }


def _ucb_direct_counts_for_T(clean_mean, T_design):
    target_index = _target_index()
    N0 = _n0()
    total_N0 = int(np.sum(N0))
    mu_minus_K = max(r_l, clean_mean[target_index] - 2.0 * _beta(N0[target_index]))
    epsilon = mu_minus_K - r_l
    logT = math.log(T_design)

    max_target = int(T_design - total_N0 - 1)
    z_low = float(r_l)
    z_high = _ucb_target_min_index(max_target, 0, T_design, mu_minus_K) - DIRECT_EPSILON_NUM
    search_log = _new_direct_search_log("UCB direct", z_low, z_high)

    def evaluate_z(z, log):
        n = np.zeros(K, dtype=int)
        max_non_target = int(T_design - total_N0 - 1)
        for i in range(K):
            if i == target_index:
                continue

            def non_target_ok(n_i):
                denom = N0[i] + n_i
                upper = (N0[i] * clean_mean[i] + n_i * r_l) / denom
                upper += 3.0 * sigma * math.sqrt(logT / denom)
                return upper <= z

            n_i = _minimal_integer_by_bracketing(
                non_target_ok, max_non_target, f"UCB non-target arm {i + 1} at z={z:.17g}", log
            )
            if n_i is None:
                return None
            n[i] = n_i

        non_target_cost = int(np.sum(n))
        max_nK = int(T_design - total_N0 - non_target_cost - 1)
        if max_nK < 0:
            return None

        def target_ok(nK):
            return _ucb_target_min_index(nK, non_target_cost, T_design, mu_minus_K) >= z + DIRECT_EPSILON_NUM

        nK = _minimal_integer_by_bracketing(
            target_ok, max_nK, f"UCB target arm {target_index + 1} at z={z:.17g}", log
        )
        if nK is None:
            return None
        n[target_index] = nK
        return n

    best_n, best_z = _direct_coarse_to_fine(evaluate_z, z_low, z_high, "UCB direct", search_log)
    if best_n is None:
        best_n = np.zeros(K, dtype=int)
    return best_n, best_z, mu_minus_K, epsilon, search_log


def _check_ucb_direct_cutoffs(clean_mean, T, n, z_star, mu_minus_K):
    target_index = _target_index()
    N0 = _n0()
    T0 = int(np.sum(N0) + np.sum(n))
    H_online = int(T - T0)
    if H_online <= 0:
        raise RuntimeError("UCB direct allocation is infeasible.")

    target_min = _ucb_target_min_index(int(n[target_index]), int(np.sum(n) - n[target_index]), T, mu_minus_K)
    if target_min < z_star + DIRECT_EPSILON_NUM:
        raise RuntimeError(f"UCB direct target check failed: target_min={target_min}, z_star={z_star}")

    for i in range(K):
        if i == target_index:
            continue
        denom = N0[i] + float(n[i])
        mean = (N0[i] * clean_mean[i] + float(n[i]) * r_l) / denom
        index_i = mean + 3.0 * sigma * math.sqrt(math.log(T) / denom)
        if index_i > z_star:
            raise RuntimeError(f"UCB direct non-target check failed: arm={i + 1}, index={index_i}, z_star={z_star}")


def UCB_direct_fixed_T(clean_sum, clean_mean, T):
    T = int(T)
    n, z_star, mu_minus_K, epsilon, search_log = _ucb_direct_counts_for_T(clean_mean, T)
    T0 = int(np.sum(_n0()) + np.sum(n))
    H_online = int(T - T0)
    if z_star is None or H_online <= 0:
        reason = "infeasible_no_z_allocation" if z_star is None else "infeasible_H_le_0"
        return _infeasible_result("UCB direct", T, n, z_star, mu_minus_K, epsilon, search_log, reason)

    _check_ucb_direct_cutoffs(clean_mean, T, n, z_star, mu_minus_K)
    if not simulate_online:
        online_counts = _certificate_online_counts(H_online)
        return _attack_result("UCB direct", T, T0, H_online, n, online_counts, z_star, mu_minus_K, epsilon, "certificate", search_log=search_log)
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
        reward = draw_environment_reward(rng, arm)
        counts[arm] += 1.0
        sums[arm] += reward
        online_counts[arm] += 1

    return _attack_result("UCB direct", T, T0, H_online, n, online_counts, z_star, mu_minus_K, epsilon, "passed", search_log=search_log)


def _ts_counts_for_T(clean_mean, T_design):
    target_index = _target_index()
    N0 = _n0()
    mu_minus_K = max(r_l, clean_mean[target_index] - 2.0 * _beta(N0[target_index]))
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
    n[target_index] = int(math.ceil(h_TS(z_star))) + 1
    for i in range(K):
        if i != target_index:
            n[i] = int(math.ceil(eta_TS_i(i, z_star)))
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
    if min_target_lhs <= z_star:
        raise RuntimeError(f"TS target cutoff check failed: min_lhs={min_target_lhs}, z_star={z_star}, candidate_m={candidate_m}")

    for i in range(K):
        if i == target_index:
            continue
        x = float(n[i])
        denom = N0[i] + x
        g_val = clean_mean[i] - x * (clean_mean[i] - r_l) / denom + gamma_T / math.sqrt(denom)
        if g_val > z_star:
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
        reward = draw_environment_reward(rng, arm)
        counts[arm] += 1.0
        sums[arm] += reward
        online_counts[arm] += 1

    return _attack_result("TS appendix", T, T0, H_online, n, online_counts, z_star, mu_minus_K, epsilon, "passed")


def _ts_target_min_value(nK, non_target_cost, T, mu_minus_K):
    N0 = _n0()
    target_index = _target_index()
    N0K = int(N0[target_index])

    # Attacked offline-log length.
    T0 = int(np.sum(N0) + non_target_cost + nK)
    H_online = int(T - T0)
    if H_online <= 0:
        return -math.inf

    # If the target has been selected in every previous online round, then
    # before a decision round its genuine target count ranges over this set.
    m_min = N0K
    m_max = N0K + H_online - 1

    def target_value(m):
        m = float(m)
        # m=N0K is the first deployment decision at T0+1; m=m_max is t=T.
        t_m = T0 + (m - N0K) + 1.0
        gamma_t = math.sqrt(
            2.0 * math.log((math.pi ** 2) * K * (t_m ** 2) / (3.0 * delta))
        )
        denom = nK + m
        mean_lb = (nK * r_u + m * mu_minus_K) / denom
        return mean_lb - gamma_t / math.sqrt(denom)

    if m_min == m_max:
        return target_value(m_min)

    # Direct is a numerical benchmark: minimize the trajectory-dependent
    # target side of Eq. (12), then audit the neighboring integer counts.
    opt = minimize_scalar(
        target_value,
        bounds=(float(m_min), float(m_max)),
        method="bounded",
        options={"xatol": 0.25, "maxiter": 100},
    )

    candidates = {m_min, m_max}
    if opt.success:
        center = int(round(opt.x))
        for m in range(center - 3, center + 4):
            if m_min <= m <= m_max:
                candidates.add(m)

    return min(target_value(m) for m in candidates)


def _ts_direct_counts_for_T(clean_mean, T_design):
    target_index = _target_index()
    N0 = _n0()
    total_N0 = int(np.sum(N0))
    mu_minus_K = max(r_l, clean_mean[target_index] - 2.0 * _beta(N0[target_index]))
    epsilon = mu_minus_K - r_l
    gamma_T = math.sqrt(2.0 * math.log((math.pi ** 2) * K * (T_design ** 2) / (3.0 * delta)))

    max_target = int(T_design - total_N0 - 1)
    z_low = float(r_l)
    z_high = _ts_target_min_value(max_target, 0, T_design, mu_minus_K) - DIRECT_EPSILON_NUM
    search_log = _new_direct_search_log("TS direct", z_low, z_high)

    def evaluate_z(z, log):
        n = np.zeros(K, dtype=int)
        max_non_target = int(T_design - total_N0 - 1)
        for i in range(K):
            if i == target_index:
                continue

            def non_target_ok(n_i):
                denom = N0[i] + n_i
                upper = (N0[i] * clean_mean[i] + n_i * r_l) / denom
                upper += gamma_T / math.sqrt(denom)
                return upper <= z

            n_i = _minimal_integer_by_bracketing(
                non_target_ok, max_non_target, f"TS non-target arm {i + 1} at z={z:.17g}", log
            )
            if n_i is None:
                return None
            n[i] = n_i

        non_target_cost = int(np.sum(n))
        max_nK = int(T_design - total_N0 - non_target_cost - 1)
        if max_nK < 0:
            return None

        def target_ok(nK):
            return _ts_target_min_value(nK, non_target_cost, T_design, mu_minus_K) >= z + DIRECT_EPSILON_NUM

        nK = _minimal_integer_by_bracketing(
            target_ok, max_nK, f"TS target arm {target_index + 1} at z={z:.17g}", log
        )
        if nK is None:
            return None
        n[target_index] = nK
        return n

    best_n, best_z = _direct_coarse_to_fine(evaluate_z, z_low, z_high, "TS direct", search_log)
    if best_n is None:
        best_n = np.zeros(K, dtype=int)
    return best_n, best_z, mu_minus_K, epsilon, search_log


def _check_ts_direct_cutoffs(clean_mean, T, n, z_star, mu_minus_K):
    target_index = _target_index()
    N0 = _n0()
    gamma_T = math.sqrt(2.0 * math.log((math.pi ** 2) * K * (T ** 2) / (3.0 * delta)))
    target_min = _ts_target_min_value(int(n[target_index]), int(np.sum(n) - n[target_index]), T, mu_minus_K)
    if target_min < z_star + DIRECT_EPSILON_NUM:
        raise RuntimeError(f"TS direct target check failed: target_min={target_min}, z_star={z_star}")

    for i in range(K):
        if i == target_index:
            continue
        denom = N0[i] + float(n[i])
        mean = (N0[i] * clean_mean[i] + float(n[i]) * r_l) / denom
        upper_i = mean + gamma_T / math.sqrt(denom)
        if upper_i > z_star:
            raise RuntimeError(f"TS direct non-target check failed: arm={i + 1}, upper={upper_i}, z_star={z_star}")


def TS_direct_fixed_T(clean_sum, clean_mean, T):
    T = int(T)
    n, z_star, mu_minus_K, epsilon, search_log = _ts_direct_counts_for_T(clean_mean, T)
    T0 = int(np.sum(_n0()) + np.sum(n))
    H_online = int(T - T0)
    if z_star is None or H_online <= 0:
        reason = "infeasible_no_z_allocation" if z_star is None else "infeasible_H_le_0"
        return _infeasible_result("TS direct", T, n, z_star, mu_minus_K, epsilon, search_log, reason)

    _check_ts_direct_cutoffs(clean_mean, T, n, z_star, mu_minus_K)
    if not simulate_online:
        online_counts = _certificate_online_counts(H_online)
        return _attack_result("TS direct", T, T0, H_online, n, online_counts, z_star, mu_minus_K, epsilon, "certificate", search_log=search_log)
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
        reward = draw_environment_reward(rng, arm)
        counts[arm] += 1.0
        sums[arm] += reward
        online_counts[arm] += 1

    return _attack_result("TS direct", T, T0, H_online, n, online_counts, z_star, mu_minus_K, epsilon, "passed", search_log=search_log)


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
        reward = draw_environment_reward(rng, arm)
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
        reward = draw_environment_reward(rng, arm)
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
