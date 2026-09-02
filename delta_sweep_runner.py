import csv
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

import bandit
import mlrunner

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_FILE = mlrunner.RESULTS_DIR / "delta_sweep_results.csv"

T_FIXED = 200_000
K = 10
MULTIPLIERS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
NUM_REPEATS = 10
SEED = 2026


def configure(mu, reward_arrays, repeat_id):
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
    bandit.mu_target = float(bandit.mu[-1])
    bandit.reward_source = "empirical"
    bandit.empirical_reward_arrays = reward_arrays
    bandit.N0_i = 5
    bandit.seed = SEED + 10_000 * repeat_id
    bandit.fake_reward_target = 1.0
    bandit.fake_reward_non_target = 0.0
    bandit.simulate_online = True


def sample_clean_non_targets(metadata, delta_k, repeat_id):
    clean_sum, _ = mlrunner.sample_clean_warm_start(metadata, repeat_id, K)
    clean_sum[-1] = 5.0 * delta_k
    return clean_sum, clean_sum / 5.0


def xu_native_cost(delta_k):
    if delta_k <= 0.0:
        return math.inf
    phase = math.ceil(K * math.log(T_FIXED) / (delta_k**2))
    return float(2 * phase)


def simulate_xu(delta_k, reward_arrays, clean_sum, repeat_id, learner):
    native_cost = xu_native_cost(delta_k)
    H_base = T_FIXED - 5 * K
    if not math.isfinite(native_cost) or native_cost >= H_base:
        return native_cost, "infeasible", None

    c1 = int(native_cost // 2)
    c2 = c1
    rng = np.random.default_rng(SEED + 700_000 * repeat_id + (0 if learner == "UCB" else 1))
    counts = np.full(K, 5, dtype=int)
    sums = np.asarray(clean_sum, dtype=float).copy()
    target_pulls = 0
    for deployment_step in range(1, H_base + 1):
        means = sums / counts
        if learner == "UCB":
            arm = int(np.argmax(means + np.sqrt(math.log(T_FIXED) / counts)))
        else:
            arm = int(np.argmax(rng.normal(loc=means, scale=1.0 / np.sqrt(counts))))
        if deployment_step <= c1:
            reward = 0.0
        elif deployment_step <= c1 + c2:
            reward = 1.0 if arm == K - 1 else 0.0
        else:
            reward = mlrunner.draw_empirical_reward(reward_arrays, rng, arm)
        counts[arm] += 1
        sums[arm] += reward
        target_pulls += int(arm == K - 1)
    return native_cost, "simulated", target_pulls / H_base


def main():
    metadata = mlrunner.load_movielens(K)
    s_t = T_FIXED ** (2.0 / 3.0) * (K * math.log(T_FIXED)) ** (1.0 / 3.0)
    rows = []
    with tqdm(total=NUM_REPEATS * len(MULTIPLIERS), desc="Delta sweep", unit="case", dynamic_ncols=True) as progress:
        for repeat_id in range(NUM_REPEATS):
            for multiplier in MULTIPLIERS:
                progress.set_postfix(repeat=repeat_id + 1, multiplier=multiplier)
                delta_k = multiplier * s_t / T_FIXED
                mu = np.asarray(metadata["mu"], dtype=float).copy()
                mu[-1] = delta_k
                reward_arrays = list(metadata["selected_reward_arrays"])
                reward_arrays[-1] = np.array([delta_k], dtype=float)
                configure(mu, reward_arrays, repeat_id)
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
                        "target_online_ratio": result["target_online_ratio"],
                        "search_log_json": json.dumps(result["search_log"], sort_keys=True),
                    })
                for learner in ("UCB", "TS"):
                    xu_cost, xu_status, xu_ratio = simulate_xu(delta_k, reward_arrays, clean_sum, repeat_id, learner)
                    rows.append({
                        "algorithm": f"Xu2021 observation-free {learner}",
                        "repeat": repeat_id,
                        "multiplier": multiplier,
                        "Delta_K": delta_k,
                        "S_T_over_T": s_t / T_FIXED,
                        "status": xu_status,
                        "cost": "" if not math.isfinite(xu_cost) else xu_cost,
                        "target_online_ratio": "" if xu_ratio is None else xu_ratio,
                        "search_log_json": "",
                    })
                progress.update(1)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {RESULTS_FILE}")
    print(f"T={T_FIXED}, K={K}, S_T/T={s_t / T_FIXED:.8f}")


if __name__ == "__main__":
    main()
