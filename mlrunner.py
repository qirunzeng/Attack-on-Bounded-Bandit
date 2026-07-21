import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import bandit

K = 10
target_arm = K
delta = 0.05
xi = 0
r_l = 0.0
r_u = 1.0
R = r_u - r_l

binary_like_threshold = 4
reward_distribution = "Bernoulli(mu_i), mu_i = MovieLens positive-rating rate"
sigma = 0.5
N0_i = 5
clean_offline_mode = "sample N0_i binary MovieLens ratings with fixed seed"
seed = 2026

fake_reward_target = r_u
fake_reward_non_target = r_l
max_fixed_point_iters = 100
LEGality_tol = 1e-9
T_GRID = [100_000, 250_000, 400_000, 550_000, 700_000, 850_000, 1_000_000]
NUM_REPEATS = 10
TARGET_MIN_COUNT = 1

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "ml-1m"
RATINGS_FILE = DATA_DIR / "ratings.dat"
MOVIES_FILE = DATA_DIR / "movies.dat"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "ml_fixed_T_results.csv"

movie_selection_mode = "top_K_minus_1_by_rating_count_plus_lowest_positive_mu_target"


def load_movielens(K_value=K):
    data_dir = DATA_DIR
    ratings_file = RATINGS_FILE
    movies_file = MOVIES_FILE

    if not ratings_file.exists():
        alt_data_dir = PROJECT_DIR / "m1-1m"
        alt_ratings_file = alt_data_dir / "ratings.dat"
        alt_movies_file = alt_data_dir / "movies.dat"
        if alt_ratings_file.exists():
            data_dir = alt_data_dir
            ratings_file = alt_ratings_file
            movies_file = alt_movies_file
        else:
            raise FileNotFoundError(
                f"Cannot find ratings.dat at {RATINGS_FILE} or {PROJECT_DIR / 'm1-1m' / 'ratings.dat'}."
            )

    movie_titles = {}
    if movies_file.exists():
        with movies_file.open("r", encoding="latin-1", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("::")
                if len(parts) >= 2:
                    movie_titles[int(parts[0])] = parts[1]

    movie_binary_rewards = defaultdict(list)
    with ratings_file.open("r", encoding="latin-1", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) != 4:
                raise ValueError(f"Bad ratings.dat row: {line!r}")
            movie_id = int(parts[1])
            rating = int(parts[2])
            movie_binary_rewards[movie_id].append(1 if rating >= binary_like_threshold else 0)

    movie_stats = []
    for movie_id, rewards in movie_binary_rewards.items():
        count_i = len(rewards)
        sum_i = int(sum(rewards))
        mu_i = sum_i / count_i
        movie_stats.append((movie_id, count_i, sum_i, mu_i))

    if len(movie_stats) < K_value:
        raise ValueError(f"MovieLens has {len(movie_stats)} movies, fewer than K={K_value}.")

    target_candidates = [row for row in movie_stats if row[3] > 0]
    if not target_candidates:
        raise ValueError("No MovieLens movie has positive empirical positive-rating rate.")

    target_row = min(target_candidates, key=lambda row: (row[3], -row[1], row[0]))
    non_target_candidates = [row for row in movie_stats if row[0] != target_row[0]]
    selected_by_count = sorted(non_target_candidates, key=lambda row: (-row[1], row[0]))[: K_value - 1]
    selected_by_count.append(target_row)
    selected_arms = sorted(selected_by_count, key=lambda row: (-row[3], -row[1], row[0]))
    selected_movie_ids = [row[0] for row in selected_arms]
    selected_counts = np.array([row[1] for row in selected_arms], dtype=int)
    selected_binary_sums = np.array([row[2] for row in selected_arms], dtype=int)
    mu = np.array([row[3] for row in selected_arms], dtype=float)

    if not np.all(mu[:-1] >= mu[1:] - 1e-15):
        raise RuntimeError("Selected MovieLens arms are not sorted by descending mu_i.")
    if K_value == K and target_arm != K:
        raise RuntimeError("This runner uses the paper convention target arm = K.")

    selected_reward_arrays = []
    for movie_id in selected_movie_ids:
        rewards_i = np.array(movie_binary_rewards[movie_id], dtype=float)
        selected_reward_arrays.append(rewards_i)
    metadata = {
        "data_dir": str(data_dir),
        "movie_titles": movie_titles,
        "selected_movie_ids": selected_movie_ids,
        "selected_counts": selected_counts,
        "selected_binary_sums": selected_binary_sums,
        "selected_reward_arrays": selected_reward_arrays,
        "mu": mu,
        "target_min_count": TARGET_MIN_COUNT,
        "target_selection": "smallest positive empirical mean",
    }
    return metadata


def sample_clean_warm_start(metadata, repeat_id):
    rng_clean = np.random.default_rng(seed + repeat_id)
    clean_rewards = []
    for rewards_i in metadata["selected_reward_arrays"]:
        replace_i = len(rewards_i) < N0_i
        clean_rewards.append(rng_clean.choice(rewards_i, size=N0_i, replace=replace_i))
    clean_sum = np.array([arr.sum() for arr in clean_rewards], dtype=float)
    clean_mean = clean_sum / float(N0_i)
    return clean_sum, clean_mean


def configure_bandit(mu, repeat_id):
    bandit.K = K
    bandit.target_arm = target_arm
    bandit.delta = delta
    bandit.xi = xi
    bandit.r_l = r_l
    bandit.r_u = r_u
    bandit.R = R
    bandit.reward_distribution = reward_distribution
    bandit.sigma = sigma
    bandit.mu_non_target = mu[:-1].tolist()
    bandit.mu_target = float(mu[-1])
    bandit.mu = mu.copy()
    bandit.N0_i = N0_i
    bandit.clean_offline_mode = clean_offline_mode
    bandit.seed = seed + repeat_id * 10_000
    bandit.fake_reward_target = fake_reward_target
    bandit.fake_reward_non_target = fake_reward_non_target
    bandit.target_lower_bound_override = None
    bandit.max_fixed_point_iters = max_fixed_point_iters
    bandit.LEGality_tol = LEGality_tol
    bandit.simulate_online = False


def flatten_result(res):
    allocation = res["allocation"]
    online_counts = res["online_counts"]
    row = {
        "algorithm": res["algorithm"],
        "repeat": res["repeat"],
        "status": res["status"],
        "T": res["T"],
        "T0": res["T0"],
        "H": res["H"],
        "Cost_n": res["Cost_n"],
        "target_online_ratio": "" if res["target_online_ratio"] is None else f"{res['target_online_ratio']:.12g}",
        "non_target_online_pulls": "" if res["non_target_online_pulls"] is None else res["non_target_online_pulls"],
        "S_T": f"{res['S_T']:.12g}",
        "normalized_cost": f"{res['normalized_cost']:.12g}",
        "z_star": "" if res["z_star"] is None else f"{res['z_star']:.12g}",
        "mu_minus_K": "" if res["mu_minus_K"] is None else f"{res['mu_minus_K']:.12g}",
        "epsilon": "" if res["epsilon"] is None else f"{res['epsilon']:.12g}",
        "legality_check": res["legality_check"],
        "allocation_json": json.dumps(allocation, sort_keys=True),
        "online_counts_json": json.dumps(online_counts, sort_keys=True),
    }
    for i in range(K):
        row[f"n_{i + 1}"] = allocation[f"n_{i + 1}"]
        row[f"N_on_{i + 1}"] = online_counts[f"N_on_{i + 1}"]
    return row


def write_results(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm",
        "repeat",
        "status",
        "T",
        "T0",
        "H",
        "Cost_n",
        "target_online_ratio",
        "non_target_online_pulls",
        "S_T",
        "normalized_cost",
        "z_star",
        "mu_minus_K",
        "epsilon",
        "legality_check",
        "allocation_json",
        "online_counts_json",
    ]
    fieldnames.extend([f"n_{i + 1}" for i in range(K)])
    fieldnames.extend([f"N_on_{i + 1}" for i in range(K)])
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flatten_result(row) for row in rows)


def validate_results(rows):
    for res in rows:
        if res["status"] != "ok":
            continue
        assert res["T0"] + res["H"] == res["T"], res
        assert res["Cost_n"] == sum(res["allocation"].values()), res
        if res["algorithm"] in {"UCB appendix", "UCB direct", "TS appendix", "TS direct"}:
            assert res["target_online_ratio"] > 0.99, res
            assert res["non_target_online_pulls"] == 0, res

    by_T_rep = {}
    for res in rows:
        by_T_rep.setdefault((res["T"], res["repeat"]), set()).add(res["algorithm"])
    required = {"UCB appendix", "UCB direct", "TS appendix", "TS direct", "UCB clean", "TS clean"}
    for key, algorithms in by_T_rep.items():
        assert algorithms == required, (key, algorithms)


def Main():
    metadata = load_movielens()

    rows = []
    clean_summaries = []
    for repeat_id in range(NUM_REPEATS):
        configure_bandit(metadata["mu"], repeat_id)
        clean_sum, clean_mean = sample_clean_warm_start(metadata, repeat_id)
        clean_summaries.append(clean_sum.astype(int).tolist())
        clean_ucb_rows = bandit.simulate_clean_fixed_T_grid(clean_sum, clean_mean, T_GRID, "UCB clean", 303)
        clean_ts_rows = bandit.simulate_clean_fixed_T_grid(clean_sum, clean_mean, T_GRID, "TS clean", 404)
        for T in T_GRID:
            repeat_rows = [
                bandit.UCB_fixed_T(clean_sum, clean_mean, T),
                bandit.UCB_direct_fixed_T(clean_sum, clean_mean, T),
                bandit.TS_fixed_T(clean_sum, clean_mean, T),
                bandit.TS_direct_fixed_T(clean_sum, clean_mean, T),
                clean_ucb_rows[T],
                clean_ts_rows[T],
            ]
            for row in repeat_rows:
                row["repeat"] = repeat_id
            rows.extend(repeat_rows)

    validate_results(rows)
    write_results(rows)

    print("=" * 100)
    print("Bounded Attack on Stochastic Warm-Start Bandits: MovieLens-1M Fixed-T Sweep")
    print("=" * 100)
    print(f"data_dir = {metadata['data_dir']}")
    print(f"movie_selection_mode = {movie_selection_mode}")
    print("target_selection = smallest positive empirical mean")
    print(f"reward_distribution = {reward_distribution}")
    print(f"binary_like_threshold = {binary_like_threshold}")
    print(f"K = {K}, target_arm = {target_arm}, delta = {delta}, xi = {xi}")
    print(f"r_l = {r_l}, r_u = {r_u}, R = {R}, sigma = {sigma}")
    print(f"N0_i = {N0_i}, clean_offline_mode = {clean_offline_mode}, seed = {seed}")
    print(f"T_grid = {T_GRID}, repeats = {NUM_REPEATS}")
    print("-" * 100)
    print(f"target_gap_mu_minus_rl = {metadata['mu'][-1] - r_l:.6f}")
    for T in T_GRID:
        print(f"hard_regime_ratio_Delta_over_ST_over_T(T={T}) = {(metadata['mu'][-1] - r_l) / (bandit._s_t(T) / T):.6f}")
    print("-" * 100)
    print("Selected MovieLens arms, sorted by descending mu_i:")
    for idx, movie_id in enumerate(metadata["selected_movie_ids"], start=1):
        title = metadata["movie_titles"].get(movie_id, "<title not found>")
        marker = "  <-- target arm K" if idx == target_arm else ""
        print(
            f"arm_{idx}: movie_id={movie_id}, title={title}, "
            f"rating_count={int(metadata['selected_counts'][idx - 1])}, "
            f"binary_sum={int(metadata['selected_binary_sums'][idx - 1])}, "
            f"mu_{idx}={metadata['mu'][idx - 1]:.6f}{marker}"
        )
    print("-" * 100)
    print(f"first_repeat_clean_sum = {clean_summaries[0]}")
    print(f"wrote {RESULTS_FILE}")
    print("=" * 100)

    for T in T_GRID:
        print(f"\n[T = {T}]")
        for algorithm in ["UCB appendix", "UCB direct", "UCB clean", "TS appendix", "TS direct", "TS clean"]:
            alg_rows = [row for row in rows if row["T"] == T and row["algorithm"] == algorithm]
            costs = np.array([row["Cost_n"] for row in alg_rows], dtype=float)
            ratios = np.array([row["target_online_ratio"] for row in alg_rows if row["target_online_ratio"] is not None], dtype=float)
            target_pulls = np.array([row["online_counts"][f"N_on_{target_arm}"] for row in alg_rows], dtype=float)
            ratio_text = "NA" if len(ratios) == 0 else f"{ratios.mean():.6f}"
            print(
                f"{algorithm}: n={len(alg_rows)}, mean_cost={costs.mean():.1f}, std_cost={costs.std(ddof=1):.1f}, "
                f"mean_target_ratio={ratio_text}, mean_target_pulls={target_pulls.mean():.1f}"
            )


if __name__ == "__main__":
    Main()
