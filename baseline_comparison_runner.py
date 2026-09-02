import csv
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

import bandit
import mlrunner

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = mlrunner.RESULTS_DIR
RESULTS_FILE = RESULTS_DIR / "baseline_comparison_results.csv"

T_FIXED = 200_000
K_GRID = [5, 10, 15, 20, 30, 40, 50]
NUM_REPEATS = 10
SEED = 2026

R_L = 0.0
R_U = 1.0
SIGMA = 0.5
DELTA = 0.05
N0_I = 5
JUN_MARGIN = 0.01


def beta(K, n):
    n = max(1, int(n))
    return math.sqrt((2.0 * SIGMA * SIGMA / n) * math.log((math.pi ** 2) * K * (n ** 2) / (3.0 * DELTA)))


def configure_offline(K, metadata, repeat_id):
    mu = metadata["mu"]
    bandit.K = K
    bandit.target_arm = K
    bandit.delta = DELTA
    bandit.xi = 0
    bandit.r_l = R_L
    bandit.r_u = R_U
    bandit.R = R_U - R_L
    bandit.sigma = SIGMA
    bandit.mu = np.asarray(mu, dtype=float)
    bandit.mu_non_target = bandit.mu[:-1].tolist()
    bandit.mu_target = float(bandit.mu[-1])
    bandit.reward_distribution = mlrunner.reward_distribution
    bandit.reward_source = "empirical"
    bandit.empirical_reward_arrays = metadata["selected_reward_arrays"]
    bandit.N0_i = N0_I
    bandit.seed = SEED + 10_000 * repeat_id + K
    bandit.fake_reward_target = R_U
    bandit.fake_reward_non_target = R_L
    bandit.simulate_online = True


def sample_clean_warm_start(metadata, K, repeat_id):
    return mlrunner.sample_clean_warm_start(metadata, repeat_id, K)


def offline_original(K, mu, metadata, repeat_id, algorithm):
    configure_offline(K, metadata, repeat_id)
    clean_sum, clean_mean = sample_clean_warm_start(metadata, K, repeat_id)
    if algorithm == "UCB":
        result = bandit.UCB_direct_fixed_T(clean_sum, clean_mean, T_FIXED)
    else:
        result = bandit.TS_direct_fixed_T(clean_sum, clean_mean, T_FIXED)
    target_online_pulls = result["online_counts"][f"N_on_{K}"]
    target_total_ratio = None
    if target_online_pulls is not None:
        target_total_ratio = (
            N0_I + result["allocation"][f"n_{K}"] + target_online_pulls
        ) / T_FIXED
    return {
        "algorithm": f"Our offline {algorithm}",
        "repeat": repeat_id,
        "K": K,
        "T": T_FIXED,
        "status": result["status"],
        "cost": float(result["Cost_n"]),
        "native_cost": float(result["Cost_n"]),
        "cost_definition": "injected samples",
        "target_ratio": target_total_ratio,
        "target_online_ratio": result["target_online_ratio"],
        "target_pulls": target_online_pulls,
        "non_target_pulls": result["non_target_online_pulls"],
        "clipped_cost": 0.0,
        "requested_cost": float(result["Cost_n"]),
        "search_log_json": json.dumps(result["search_log"], sort_keys=True),
    }


def simulate_jun_ucb(K, reward_arrays, clean_sum, repeat_id):
    rng = np.random.default_rng(SEED + 200_000 * repeat_id + K)
    counts = np.full(K, N0_I, dtype=int)
    sums = np.asarray(clean_sum, dtype=float).copy()
    H_base = T_FIXED - N0_I * K
    target_pulls = 0
    attack_count = 0
    requested_cost = 0.0
    clipped_cost = 0.0

    for deployment_step in range(1, H_base + 1):
        t = N0_I * K + deployment_step
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
        "K": K,
        "T": T_FIXED,
        "status": "ok" if target_pulls / H_base >= 0.99 else "failed",
        "cost": attack_count,
        "native_cost": attack_count,
        "cost_definition": "online reward-lowering modification count clipped at r_l",
        "target_ratio": (N0_I + target_pulls) / T_FIXED,
        "target_online_ratio": target_pulls / H_base,
        "H_base": H_base,
        "target_pulls": target_pulls,
        "non_target_pulls": H_base - target_pulls,
        "clipped_cost": clipped_cost,
        "requested_cost": requested_cost,
    }


def simulate_zuo_ts(K, reward_arrays, clean_sum, repeat_id):
    rng = np.random.default_rng(SEED + 300_000 * repeat_id + K)
    counts = np.full(K, N0_I, dtype=int)
    sums = np.asarray(clean_sum, dtype=float).copy()
    H_base = T_FIXED - N0_I * K
    target_pulls = 0
    attack_count = 0
    requested_cost = 0.0
    clipped_cost = 0.0
    ts_const = math.sqrt(8.0 * math.log((math.pi ** 2) * K / (3.0 * DELTA)))

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
        "K": K,
        "T": T_FIXED,
        "status": "ok" if target_pulls / H_base >= 0.99 else "failed",
        "cost": attack_count,
        "native_cost": attack_count,
        "cost_definition": "online reward-lowering modification count clipped at r_l",
        "target_ratio": (N0_I + target_pulls) / T_FIXED,
        "target_online_ratio": target_pulls / H_base,
        "H_base": H_base,
        "target_pulls": target_pulls,
        "non_target_pulls": H_base - target_pulls,
        "clipped_cost": clipped_cost,
        "requested_cost": requested_cost,
    }


def xu2021_phase_cost(K, mu, reward_arrays, clean_sum, repeat_id, learner):
    phase = math.ceil(K * math.log(T_FIXED) / (float(mu[-1]) ** 2))
    c1 = phase
    c2 = phase
    native_cost = c1 + c2
    H_base = T_FIXED - N0_I * K
    feasible = native_cost < H_base
    if not feasible:
        return {
            "algorithm": f"Xu2021 observation-free {learner}",
            "repeat": repeat_id,
            "K": K,
            "T": T_FIXED,
            "status": "infeasible",
            "cost": float(native_cost),
            "native_cost": float(native_cost),
            "cost_definition": "corrupted online deployment rounds; not simulated because C1+C2 is not strictly below H_base",
            "H_base": H_base,
            "C1": c1,
            "C2": c2,
            "target_ratio": "",
            "target_online_ratio": "",
            "post_phase_ratio": "",
            "target_pulls": "",
            "non_target_pulls": "",
            "clipped_cost": 0.0,
            "requested_cost": float(native_cost),
            "search_log_json": "",
        }

    target_pulls = ""
    non_target_pulls = ""
    target_ratio = ""
    target_online_ratio = ""
    post_phase_ratio = ""

    if feasible:
        rng = np.random.default_rng(SEED + 400_000 * repeat_id + 1_000 * K + (0 if learner == "UCB" else 1))
        counts = np.full(K, N0_I, dtype=int)
        successes = np.asarray(clean_sum, dtype=float).copy()
        target_index = K - 1
        target_pulls = 0
        post_target_pulls = 0

        for deployment_step in range(1, H_base + 1):
            if learner == "UCB":
                means = successes / counts
                index = means + np.sqrt(math.log(T_FIXED) / counts)
                arm = int(np.argmax(index))
            else:
                means = successes / counts
                samples = rng.normal(loc=means, scale=1.0 / np.sqrt(counts))
                arm = int(np.argmax(samples))

            if deployment_step <= c1:
                reward = 0.0
            elif deployment_step <= c1 + c2:
                reward = 1.0 if arm == target_index else 0.0
            else:
                # Previous Scheme A:
                # mean = float(mu[arm])
                # reward = 1.0 if rng.random() < mean else 0.0
                reward = mlrunner.draw_empirical_reward(reward_arrays, rng, arm)

            counts[arm] += 1
            successes[arm] += reward
            if arm == target_index:
                target_pulls += 1
                if deployment_step > native_cost:
                    post_target_pulls += 1

        non_target_pulls = H_base - target_pulls
        target_ratio = (N0_I + target_pulls) / T_FIXED
        target_online_ratio = target_pulls / H_base
        post_phase_ratio = post_target_pulls / (H_base - native_cost)

    return {
        "algorithm": f"Xu2021 observation-free {learner}",
        "repeat": repeat_id,
        "K": K,
        "T": T_FIXED,
        "status": "infeasible" if not feasible else "simulated",
        "cost": float(native_cost),
        "native_cost": float(native_cost),
        "cost_definition": "corrupted online deployment rounds",
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
        "search_log_json": "",
    }


def write_results(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm",
        "repeat",
        "K",
        "T",
        "status",
        "cost",
        "native_cost",
        "cost_definition",
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
        "search_log_json",
    ]
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def Main():
    rows = []
    instances = {K: mlrunner.load_movielens(K) for K in K_GRID}
    with tqdm(total=NUM_REPEATS * len(K_GRID), desc="Baseline comparison", unit="case", dynamic_ncols=True) as progress:
        for repeat_id in range(NUM_REPEATS):
            for K in K_GRID:
                progress.set_postfix(repeat=repeat_id + 1, K=K)
                metadata = instances[K]
                mu = metadata["mu"]
                reward_arrays = metadata["selected_reward_arrays"]
                clean_sum, _ = sample_clean_warm_start(metadata, K, repeat_id)
                rows.append(offline_original(K, mu, metadata, repeat_id, "UCB"))
                rows.append(offline_original(K, mu, metadata, repeat_id, "TS"))
                rows.append(simulate_jun_ucb(K, reward_arrays, clean_sum, repeat_id))
                rows.append(simulate_zuo_ts(K, reward_arrays, clean_sum, repeat_id))
                rows.append(xu2021_phase_cost(K, mu, reward_arrays, clean_sum, repeat_id, "UCB"))
                rows.append(xu2021_phase_cost(K, mu, reward_arrays, clean_sum, repeat_id, "TS"))
                progress.update(1)

    for row in rows:
        row.setdefault("post_phase_ratio", "")
        row.setdefault("C1", "")
        row.setdefault("C2", "")
        row.setdefault("search_log_json", "")
    write_results(rows)
    print("Online baseline comparison")
    print(f"T = {T_FIXED}, K_grid = {K_GRID}, repeats = {NUM_REPEATS}")
    print(f"instance = {mlrunner.DATASET_LABEL}, target_mu = {instances[K_GRID[0]]['mu'][-1]:.12g}")
    print(f"wrote {RESULTS_FILE}")
    for K in K_GRID:
        print(f"\n[K = {K}]")
        for algorithm in [
            "Our offline UCB",
            "Jun2018 clipped UCB",
            "Our offline TS",
            "Zuo2024 clipped TS",
            "Xu2021 observation-free UCB",
        ]:
            vals = [row for row in rows if row["K"] == K and row["algorithm"] == algorithm]
            costs = np.array([float(row["cost"]) for row in vals if row["cost"] != ""], dtype=float)
            ratios = [float(row["target_ratio"]) for row in vals if row["target_ratio"] != ""]
            ratio_text = "NA" if not ratios else f"{np.mean(ratios):.4f}"
            cost_text = "NA" if costs.size == 0 else f"{costs.mean():.1f}"
            print(f"{algorithm}: mean_cost={cost_text}, mean_target_ratio={ratio_text}")


if __name__ == "__main__":
    Main()
