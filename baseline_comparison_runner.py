import csv
import math
from pathlib import Path

import numpy as np

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
    bandit.target_lower_bound_override = None
    bandit.simulate_online = False


def sample_clean_warm_start(metadata, K, repeat_id):
    rng = np.random.default_rng(SEED + 100_000 * repeat_id + K)
    clean_sum = np.array([
        rng.choice(rewards, size=N0_I, replace=len(rewards) < N0_I).sum()
        for rewards in metadata["selected_reward_arrays"]
    ], dtype=float)
    clean_mean = clean_sum / float(N0_I)
    return clean_sum, clean_mean


def offline_original(K, mu, metadata, repeat_id, algorithm):
    configure_offline(K, metadata, repeat_id)
    clean_sum, clean_mean = sample_clean_warm_start(metadata, K, repeat_id)
    if algorithm == "UCB":
        result = bandit.UCB_direct_fixed_T(clean_sum, clean_mean, T_FIXED)
    else:
        result = bandit.TS_direct_fixed_T(clean_sum, clean_mean, T_FIXED)
    return {
        "algorithm": f"Our offline {algorithm}",
        "repeat": repeat_id,
        "K": K,
        "T": T_FIXED,
        "status": result["status"],
        "cost": float(result["Cost_n"]),
        "native_cost": float(result["Cost_n"]),
        "cost_definition": "injected samples",
        "target_ratio": result["target_online_ratio"],
        "target_pulls": result["online_counts"][f"N_on_{K}"],
        "non_target_pulls": result["non_target_online_pulls"],
        "clipped_cost": 0.0,
        "requested_cost": float(result["Cost_n"]),
    }


def initial_arm(t, K):
    return t - 1 if t <= K else None


def simulate_jun_ucb(K, reward_arrays, repeat_id):
    rng = np.random.default_rng(SEED + 200_000 * repeat_id + K)
    counts = np.zeros(K, dtype=int)
    sums = np.zeros(K, dtype=float)
    target_pulls = 0
    attack_count = 0
    requested_cost = 0.0
    clipped_cost = 0.0

    for t in range(1, T_FIXED + 1):
        init = initial_arm(t, K)
        if init is not None:
            arm = init
        else:
            means = sums / np.maximum(counts, 1)
            index = means + 3.0 * SIGMA * np.sqrt(math.log(t) / np.maximum(counts, 1))
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
        "status": "ok" if target_pulls / T_FIXED >= 0.99 else "failed",
        "cost": attack_count,
        "native_cost": attack_count,
        "cost_definition": "online reward-lowering modification count clipped at r_l",
        "target_ratio": target_pulls / T_FIXED,
        "target_pulls": target_pulls,
        "non_target_pulls": T_FIXED - target_pulls,
        "clipped_cost": clipped_cost,
        "requested_cost": requested_cost,
    }


def simulate_zuo_ts(K, reward_arrays, repeat_id):
    rng = np.random.default_rng(SEED + 300_000 * repeat_id + K)
    counts = np.zeros(K, dtype=int)
    sums = np.zeros(K, dtype=float)
    target_pulls = 0
    attack_count = 0
    requested_cost = 0.0
    clipped_cost = 0.0
    ts_const = math.sqrt(8.0 * math.log((math.pi ** 2) * K / (3.0 * DELTA)))

    for t in range(1, T_FIXED + 1):
        init = initial_arm(t, K)
        if init is not None:
            arm = init
        else:
            means = sums / np.maximum(counts, 1)
            samples = rng.normal(loc=means, scale=1.0 / np.sqrt(np.maximum(counts, 1)))
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
        "status": "ok" if target_pulls / T_FIXED >= 0.99 else "failed",
        "cost": attack_count,
        "native_cost": attack_count,
        "cost_definition": "online reward-lowering modification count clipped at r_l",
        "target_ratio": target_pulls / T_FIXED,
        "target_pulls": target_pulls,
        "non_target_pulls": T_FIXED - target_pulls,
        "clipped_cost": clipped_cost,
        "requested_cost": requested_cost,
    }


def xu2021_phase_cost(K, mu, reward_arrays, repeat_id, learner):
    phase = math.ceil(K * math.log(T_FIXED) / (float(mu[-1]) ** 2))
    c1 = phase
    c2 = phase
    native_cost = c1 + c2
    sample_equivalent_cost = K * phase + K * phase
    feasible = native_cost < T_FIXED
    if not feasible:
        return {
            "algorithm": f"Xu2021 observation-free {learner}",
            "repeat": repeat_id,
            "K": K,
            "T": T_FIXED,
            "status": "infeasible",
            "cost": "",
            "native_cost": float(native_cost),
            "cost_definition": "corrupted rounds; not simulated because native cost exceeds horizon",
            "target_ratio": "",
            "post_phase_ratio": "",
            "target_pulls": "",
            "non_target_pulls": "",
            "clipped_cost": 0.0,
            "requested_cost": float(native_cost),
        }

    target_pulls = ""
    non_target_pulls = ""
    target_ratio = ""
    post_phase_ratio = ""

    if feasible:
        rng = np.random.default_rng(SEED + 400_000 * repeat_id + 1_000 * K + (0 if learner == "UCB" else 1))
        counts = np.zeros(K, dtype=int)
        successes = np.zeros(K, dtype=float)
        target_index = K - 1
        target_pulls = 0
        post_target_pulls = 0

        for t in range(1, T_FIXED + 1):
            if learner == "UCB":
                init = initial_arm(t, K)
                if init is not None:
                    arm = init
                else:
                    means = successes / np.maximum(counts, 1)
                    index = means + np.sqrt(math.log(T_FIXED) / np.maximum(counts, 1))
                    arm = int(np.argmax(index))
            else:
                samples = rng.beta(successes + 1.0, counts - successes + 1.0)
                arm = int(np.argmax(samples))

            if t <= c1:
                reward = 0.0
            elif t <= native_cost:
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
                if t > native_cost:
                    post_target_pulls += 1

        non_target_pulls = T_FIXED - target_pulls
        target_ratio = target_pulls / T_FIXED
        post_phase_ratio = post_target_pulls / (T_FIXED - native_cost)

    return {
        "algorithm": f"Xu2021 observation-free {learner}",
        "repeat": repeat_id,
        "K": K,
        "T": T_FIXED,
        "status": "infeasible" if not feasible else "simulated",
        "cost": float(sample_equivalent_cost),
        "native_cost": float(native_cost),
        "cost_definition": "corrupted rounds; sample-equivalent shown as K times rounds",
        "target_ratio": target_ratio,
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
        "K",
        "T",
        "status",
        "cost",
        "native_cost",
        "cost_definition",
        "target_ratio",
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
    rows = []
    instances = {K: mlrunner.load_movielens(K) for K in K_GRID}
    for repeat_id in range(NUM_REPEATS):
        for K in K_GRID:
            metadata = instances[K]
            mu = metadata["mu"]
            reward_arrays = metadata["selected_reward_arrays"]
            rows.append(offline_original(K, mu, metadata, repeat_id, "UCB"))
            rows.append(offline_original(K, mu, metadata, repeat_id, "TS"))
            rows.append(simulate_jun_ucb(K, reward_arrays, repeat_id))
            rows.append(simulate_zuo_ts(K, reward_arrays, repeat_id))
            rows.append(xu2021_phase_cost(K, mu, reward_arrays, repeat_id, "UCB"))
            rows.append(xu2021_phase_cost(K, mu, reward_arrays, repeat_id, "TS"))

    for row in rows:
        row.setdefault("post_phase_ratio", "")
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
