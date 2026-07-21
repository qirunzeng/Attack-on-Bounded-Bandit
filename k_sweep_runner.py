import csv
import json
from pathlib import Path

import numpy as np

import bandit
import mlrunner

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = mlrunner.RESULTS_DIR
RESULTS_FILE = RESULTS_DIR / "k_sweep_results.csv"

T_FIXED = 200_000
K_GRID = [5, 10, 15, 20, 30, 40, 50]
NUM_REPEATS = 10
SEED = 2026

R_L = 0.0
R_U = 1.0
SIGMA = 0.5
DELTA = 0.05
N0_I = 5


def configure(K, metadata, repeat_id):
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


def add_decomposition(row, K):
    allocation = row["allocation"]
    target_cost = int(allocation[f"n_{K}"])
    non_target_total = int(sum(allocation[f"n_{i}"] for i in range(1, K)))
    row["K"] = K
    row["repeat"] = row["repeat"]
    row["target_cost"] = target_cost
    row["non_target_total_cost"] = non_target_total
    row["non_target_avg_cost"] = non_target_total / float(K - 1)
    return row


def flatten(row):
    allocation = row["allocation"]
    out = {
        "algorithm": row["algorithm"],
        "repeat": row["repeat"],
        "K": row["K"],
        "status": row["status"],
        "T": row["T"],
        "T0": row["T0"],
        "H": row["H"],
        "Cost_n": row["Cost_n"],
        "target_cost": row["target_cost"],
        "non_target_total_cost": row["non_target_total_cost"],
        "non_target_avg_cost": f"{row['non_target_avg_cost']:.12g}",
        "S_T": f"{row['S_T']:.12g}",
        "normalized_cost": f"{row['normalized_cost']:.12g}",
        "z_star": "" if row["z_star"] is None else f"{row['z_star']:.12g}",
        "allocation_json": json.dumps(allocation, sort_keys=True),
    }
    for i in range(1, row["K"] + 1):
        out[f"n_{i}"] = allocation[f"n_{i}"]
    return out


def write_results(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    max_K = max(row["K"] for row in rows)
    fieldnames = [
        "algorithm",
        "repeat",
        "K",
        "status",
        "T",
        "T0",
        "H",
        "Cost_n",
        "target_cost",
        "non_target_total_cost",
        "non_target_avg_cost",
        "S_T",
        "normalized_cost",
        "z_star",
        "allocation_json",
    ]
    fieldnames.extend(f"n_{i}" for i in range(1, max_K + 1))
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flatten(row) for row in rows)


def validate(rows):
    for row in rows:
        if row["status"] != "ok":
            continue
        assert row["T0"] + row["H"] == row["T"], row
        assert row["Cost_n"] == row["target_cost"] + row["non_target_total_cost"], row
        assert abs(row["non_target_avg_cost"] * (row["K"] - 1) - row["non_target_total_cost"]) < 1e-9, row


def Main():
    rows = []
    instances = {K: mlrunner.load_movielens(K) for K in K_GRID}
    for repeat_id in range(NUM_REPEATS):
        for K in K_GRID:
            metadata = instances[K]
            configure(K, metadata, repeat_id)
            clean_sum, clean_mean = sample_clean_warm_start(metadata, K, repeat_id)
            for result in [
                bandit.UCB_fixed_T(clean_sum, clean_mean, T_FIXED),
                bandit.UCB_direct_fixed_T(clean_sum, clean_mean, T_FIXED),
                bandit.TS_fixed_T(clean_sum, clean_mean, T_FIXED),
                bandit.TS_direct_fixed_T(clean_sum, clean_mean, T_FIXED),
            ]:
                result["repeat"] = repeat_id
                rows.append(add_decomposition(result, K))

    validate(rows)
    write_results(rows)

    print("Bounded offline attack K-sweep")
    print(f"T = {T_FIXED}, K_grid = {K_GRID}, repeats = {NUM_REPEATS}")
    print(f"instance = {mlrunner.DATASET_LABEL}, N0_i = {N0_I}")
    print(f"target_mu = {instances[K_GRID[0]]['mu'][-1]:.12g}")
    print(f"wrote {RESULTS_FILE}")
    for K in K_GRID:
        print(f"\n[K = {K}]")
        for algorithm in ["UCB direct", "TS direct"]:
            vals = [row for row in rows if row["K"] == K and row["algorithm"] == algorithm]
            costs = np.array([row["Cost_n"] for row in vals], dtype=float)
            target = np.array([row["target_cost"] for row in vals], dtype=float)
            non_target = np.array([row["non_target_avg_cost"] for row in vals], dtype=float)
            print(
                f"{algorithm}: mean_total={costs.mean():.1f}, "
                f"mean_target={target.mean():.1f}, mean_non_target_avg={non_target.mean():.1f}"
            )


if __name__ == "__main__":
    Main()
