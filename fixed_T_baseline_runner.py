import csv
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

import mlrunner

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = mlrunner.RESULTS_DIR
RESULTS_FILE = RESULTS_DIR / "ml_fixed_T_baseline_results.csv"

NUM_REPEATS = mlrunner.NUM_REPEATS
T_GRID = mlrunner.T_GRID
K = mlrunner.K
SEED = mlrunner.seed
SIGMA = mlrunner.sigma
DELTA = mlrunner.delta
R_L = mlrunner.r_l
JUN_MARGIN = 0.01


def beta(K_value, n):
    n = max(1, int(n))
    return math.sqrt((2.0 * SIGMA * SIGMA / n) * math.log((math.pi**2) * K_value * (n**2) / (3.0 * DELTA)))


def simulate_jun_ucb(reward_arrays, clean_sum, T, repeat_id):
    rng = np.random.default_rng(SEED + 500_000 * repeat_id + T)
    counts = np.full(K, mlrunner.N0_i, dtype=int)
    sums = np.asarray(clean_sum, dtype=float).copy()
    H_base = T - mlrunner.N0_i * K
    if H_base <= 0:
        raise ValueError(f"T={T} does not leave any online deployment after the shared warm-start.")
    target_pulls = 0
    attack_count = 0
    requested_cost = 0.0
    clipped_cost = 0.0

    for deployment_step in range(1, H_base + 1):
        t = mlrunner.N0_i * K + deployment_step
        means = sums / counts
        index = means + 3.0 * SIGMA * np.sqrt(math.log(t) / counts)
        arm = int(np.argmax(index))

        # Previous Scheme A:
        # reward0 = 1.0 if rng.random() < mu[arm] else 0.0
        reward0 = mlrunner.draw_empirical_reward(reward_arrays, rng, arm)
        alpha_req = 0.0
        if arm != K - 1 and counts[K - 1] > 0:
            target_mean = sums[K - 1] / counts[K - 1]
            threshold = target_mean - 2.0 * beta(K, counts[K - 1]) - JUN_MARGIN
            alpha_req = max(0.0, sums[arm] + reward0 - threshold * (counts[arm] + 1))
        alpha = min(alpha_req, max(0.0, reward0 - R_L))
        requested_cost += alpha_req
        clipped_cost += alpha_req - alpha
        if alpha > 0:
            attack_count += 1
        reward = reward0 - alpha
        counts[arm] += 1
        sums[arm] += reward
        if arm == K - 1:
            target_pulls += 1

    return {
        "algorithm": "Jun2018 clipped UCB",
        "repeat": repeat_id,
        "T": T,
        "status": "ok" if target_pulls / H_base >= 0.99 else "failed",
        "native_cost": attack_count,
        "target_ratio": (mlrunner.N0_i + target_pulls) / T,
        "target_online_ratio": target_pulls / H_base,
        "H_base": H_base,
        "target_pulls": target_pulls,
        "non_target_pulls": H_base - target_pulls,
        "clipped_cost": clipped_cost,
        "requested_cost": requested_cost,
    }


def simulate_zuo_ts(reward_arrays, clean_sum, T, repeat_id):
    rng = np.random.default_rng(SEED + 600_000 * repeat_id + T)
    counts = np.full(K, mlrunner.N0_i, dtype=int)
    sums = np.asarray(clean_sum, dtype=float).copy()
    H_base = T - mlrunner.N0_i * K
    if H_base <= 0:
        raise ValueError(f"T={T} does not leave any online deployment after the shared warm-start.")
    target_pulls = 0
    attack_count = 0
    requested_cost = 0.0
    clipped_cost = 0.0
    ts_const = math.sqrt(8.0 * math.log((math.pi**2) * K / (3.0 * DELTA)))

    for deployment_step in range(1, H_base + 1):
        means = sums / counts
        samples = rng.normal(loc=means, scale=1.0 / np.sqrt(counts))
        arm = int(np.argmax(samples))

        # Previous Scheme A:
        # reward0 = 1.0 if rng.random() < mu[arm] else 0.0
        reward0 = mlrunner.draw_empirical_reward(reward_arrays, rng, arm)
        alpha_req = 0.0
        if arm != K - 1 and counts[K - 1] > 0:
            target_mean = sums[K - 1] / counts[K - 1]
            post_count = counts[arm] + 1
            exp_gap = math.exp(min(post_count, 20))
            threshold = target_mean - 2.0 * beta(K, counts[K - 1]) - 4.0 * exp_gap - ts_const
            alpha_req = max(0.0, sums[arm] + reward0 - threshold * post_count)
        alpha = min(alpha_req, max(0.0, reward0 - R_L))
        requested_cost += alpha_req
        clipped_cost += alpha_req - alpha
        if alpha > 0:
            attack_count += 1
        reward = reward0 - alpha
        counts[arm] += 1
        sums[arm] += reward
        if arm == K - 1:
            target_pulls += 1

    return {
        "algorithm": "Zuo2024 clipped TS",
        "repeat": repeat_id,
        "T": T,
        "status": "ok" if target_pulls / H_base >= 0.99 else "failed",
        "native_cost": attack_count,
        "target_ratio": (mlrunner.N0_i + target_pulls) / T,
        "target_online_ratio": target_pulls / H_base,
        "H_base": H_base,
        "target_pulls": target_pulls,
        "non_target_pulls": H_base - target_pulls,
        "clipped_cost": clipped_cost,
        "requested_cost": requested_cost,
    }


def xu2021_phase_cost(mu_target, reward_arrays, clean_sum, T, repeat_id, learner):
    phase = math.ceil(K * math.log(T) / (mu_target**2))
    c1 = phase
    c2 = phase
    native_cost = 2 * phase
    H_base = T - mlrunner.N0_i * K
    feasible = native_cost < H_base
    target_pulls = ""
    non_target_pulls = ""
    target_ratio = ""
    target_online_ratio = ""
    post_phase_ratio = ""
    if feasible:
        rng = np.random.default_rng(SEED + 700_000 * repeat_id + T + (0 if learner == "UCB" else 1))
        counts = np.full(K, mlrunner.N0_i, dtype=int)
        successes = np.asarray(clean_sum, dtype=float).copy()
        target_pulls = 0
        post_target_pulls = 0
        for deployment_step in range(1, H_base + 1):
            if learner == "UCB":
                means = successes / counts
                arm = int(np.argmax(means + np.sqrt(math.log(T) / counts)))
            else:
                means = successes / counts
                arm = int(np.argmax(rng.normal(loc=means, scale=1.0 / np.sqrt(counts))))
            if deployment_step <= c1:
                reward = 0.0
            elif deployment_step <= c1 + c2:
                reward = 1.0 if arm == K - 1 else 0.0
            else:
                # Previous Scheme A:
                # reward = 1.0 if rng.random() < mu[arm] else 0.0
                reward = mlrunner.draw_empirical_reward(reward_arrays, rng, arm)
            counts[arm] += 1
            successes[arm] += reward
            if arm == K - 1:
                target_pulls += 1
                if deployment_step > native_cost:
                    post_target_pulls += 1
        non_target_pulls = H_base - target_pulls
        target_ratio = (mlrunner.N0_i + target_pulls) / T
        target_online_ratio = target_pulls / H_base
        post_phase_ratio = post_target_pulls / (H_base - native_cost)
    return {
        "algorithm": f"Xu2021 observation-free {learner}",
        "repeat": repeat_id,
        "T": T,
        "status": "simulated" if feasible else "infeasible",
        "native_cost": float(native_cost),
        "H_base": H_base,
        "C1": c1,
        "C2": c2,
        "target_ratio": target_ratio,
        "target_online_ratio": target_online_ratio,
        "post_phase_ratio": post_phase_ratio,
        "target_pulls": target_pulls,
        "non_target_pulls": non_target_pulls,
        "clipped_cost": 0.0,
        "requested_cost": float(native_cost),
    }


def write_results(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm",
        "repeat",
        "T",
        "status",
        "native_cost",
        "H_base",
        "C1",
        "C2",
        "target_ratio",
        "target_online_ratio",
        "post_phase_ratio",
        "target_pulls",
        "non_target_pulls",
        "clipped_cost",
        "requested_cost",
    ]
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def Main():
    metadata = mlrunner.load_movielens()
    mu = metadata["mu"]
    reward_arrays = metadata["selected_reward_arrays"]
    rows = []
    with tqdm(total=NUM_REPEATS * len(T_GRID), desc="Fixed-T baselines", unit="case", dynamic_ncols=True) as progress:
        for repeat_id in range(NUM_REPEATS):
            clean_sum, _ = mlrunner.sample_clean_warm_start(metadata, repeat_id, K)
            for T in T_GRID:
                progress.set_postfix(repeat=repeat_id + 1, T=T)
                rows.append(simulate_jun_ucb(reward_arrays, clean_sum, T, repeat_id))
                rows.append(simulate_zuo_ts(reward_arrays, clean_sum, T, repeat_id))
                rows.append(xu2021_phase_cost(float(mu[-1]), reward_arrays, clean_sum, T, repeat_id, "UCB"))
                rows.append(xu2021_phase_cost(float(mu[-1]), reward_arrays, clean_sum, T, repeat_id, "TS"))
                progress.update(1)

    for row in rows:
        row.setdefault("post_phase_ratio", "")
        row.setdefault("C1", "")
        row.setdefault("C2", "")

    write_results(rows)
    print(f"wrote {RESULTS_FILE}")
    for T in T_GRID:
        print(f"\n[T = {T}]")
        for algorithm in [
            "Jun2018 clipped UCB",
            "Zuo2024 clipped TS",
            "Xu2021 observation-free UCB",
            "Xu2021 observation-free TS",
        ]:
            vals = [row for row in rows if row["T"] == T and row["algorithm"] == algorithm]
            costs = np.array([float(row["native_cost"]) for row in vals], dtype=float)
            ratios = [float(row["target_ratio"]) for row in vals if row["target_ratio"] != ""]
            ratio_text = "NA" if not ratios else f"{np.mean(ratios):.4f}"
            print(f"{algorithm}: mean_native_cost={costs.mean():.1f}, mean_target_ratio={ratio_text}")


if __name__ == "__main__":
    Main()
