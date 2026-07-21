import csv
import math
from pathlib import Path

import numpy as np

import bandit
import mlrunner

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_FILE = PROJECT_DIR / "results" / "delta_sweep_results.csv"

T_FIXED = 200_000
K = 10
MULTIPLIERS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
NUM_REPEATS = 10
SEED = 2026


def configure(mu, delta_k, repeat_id):
    bandit.K = K
    bandit.target_arm = K
    bandit.delta = 0.05
    bandit.xi = 0
    bandit.r_l = 0.0
    bandit.r_u = 1.0
    bandit.R = 1.0
    bandit.sigma = 0.5
    bandit.mu = np.asarray(mu, dtype=float)
    bandit.mu_non_target = bandit.mu[:-1].tolist()
    bandit.mu_target = float(delta_k)
    bandit.N0_i = 5
    bandit.seed = SEED + 10_000 * repeat_id
    bandit.fake_reward_target = 1.0
    bandit.fake_reward_non_target = 0.0
    bandit.simulate_online = False
    # This controlled study supplies the target-side lower certificate directly.
    bandit.target_lower_bound_override = float(delta_k)


def sample_clean_non_targets(metadata, delta_k, repeat_id):
    rng = np.random.default_rng(SEED + 100_000 * repeat_id)
    clean_sum = np.array([
        rng.choice(rewards, size=5, replace=len(rewards) < 5).sum()
        for rewards in metadata["selected_reward_arrays"]
    ], dtype=float)
    # Use a deterministic target arm at the prescribed gap, so its genuine
    # empirical average equals the supplied target-side certificate.
    clean_sum[-1] = 5.0 * delta_k
    return clean_sum, clean_sum / 5.0


def xu_native_cost(delta_k):
    if delta_k <= 0.0:
        return math.inf
    phase = math.ceil(K * math.log(T_FIXED) / (delta_k**2))
    return float(2 * phase)


def main():
    metadata = mlrunner.load_movielens(K)
    s_t = T_FIXED ** (2.0 / 3.0) * (K * math.log(T_FIXED)) ** (1.0 / 3.0)
    rows = []
    for repeat_id in range(NUM_REPEATS):
        for multiplier in MULTIPLIERS:
            delta_k = multiplier * s_t / T_FIXED
            mu = np.asarray(metadata["mu"], dtype=float).copy()
            mu[-1] = delta_k
            configure(mu, delta_k, repeat_id)
            clean_sum, clean_mean = sample_clean_non_targets(metadata, delta_k, repeat_id)
            for result in (
                bandit.UCB_fixed_T(clean_sum, clean_mean, T_FIXED),
                bandit.UCB_direct_fixed_T(clean_sum, clean_mean, T_FIXED),
                bandit.TS_fixed_T(clean_sum, clean_mean, T_FIXED),
                bandit.TS_direct_fixed_T(clean_sum, clean_mean, T_FIXED),
            ):
                rows.append({
                    "algorithm": result["algorithm"],
                    "repeat": repeat_id,
                    "multiplier": multiplier,
                    "Delta_K": delta_k,
                    "S_T_over_T": s_t / T_FIXED,
                    "status": result["status"],
                    "cost": result["Cost_n"],
                })
            xu_cost = xu_native_cost(delta_k)
            rows.append({
                "algorithm": "Xu2021 observation-free UCB",
                "repeat": repeat_id,
                "multiplier": multiplier,
                "Delta_K": delta_k,
                "S_T_over_T": s_t / T_FIXED,
                "status": "ok" if xu_cost < T_FIXED else "infeasible",
                "cost": "" if not math.isfinite(xu_cost) else xu_cost,
            })
            rows.append({
                "algorithm": "Xu2021 observation-free TS",
                "repeat": repeat_id,
                "multiplier": multiplier,
                "Delta_K": delta_k,
                "S_T_over_T": s_t / T_FIXED,
                "status": "ok" if xu_cost < T_FIXED else "infeasible",
                "cost": "" if not math.isfinite(xu_cost) else xu_cost,
            })

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    bandit.target_lower_bound_override = None
    print(f"wrote {RESULTS_FILE}")
    print(f"T={T_FIXED}, K={K}, S_T/T={s_t / T_FIXED:.8f}")


if __name__ == "__main__":
    main()
