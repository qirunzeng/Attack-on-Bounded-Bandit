import csv
import json
import os
import sys
from collections import defaultdict
from functools import lru_cache
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

# Previous Scheme A (kept for reproducibility/reference):
# binary_like_threshold = 4
# reward_distribution = "Bernoulli(mu_i), mu_i = MovieLens positive-rating rate"
reward_distribution = "empirical MovieLens rating distribution normalized to [0, 1]"
sigma = 0.5
N0_i = 5
# Previous Scheme A:
# clean_offline_mode = "sample N0_i binary MovieLens ratings with fixed seed"
clean_offline_mode = "sample N0_i normalized empirical MovieLens ratings with fixed seed"
seed = 2026

fake_reward_target = r_u
fake_reward_non_target = r_l
max_fixed_point_iters = 100
LEGality_tol = 1e-9
# Previous 1M-horizon grid:
# T_GRID = [100_000, 250_000, 400_000, 550_000, 700_000, 850_000, 1_000_000]
# The MovieLens-25M experiments now extend the learner horizon to 25M.
T_GRID = [
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_500_000,
    5_000_000,
    10_000_000,
    15_000_000,
    20_000_000,
    25_000_000,
]
NUM_REPEATS = 10
TARGET_MIN_COUNT = 1

PROJECT_DIR = Path(__file__).resolve().parent
# Previous MovieLens-1M-only paths:
# DATA_DIR = PROJECT_DIR / "ml-1m"
# RATINGS_FILE = DATA_DIR / "ratings.dat"
# MOVIES_FILE = DATA_DIR / "movies.dat"
MOVIELENS_DATASET = os.environ.get("MOVIELENS_DATASET", "25m").lower().removeprefix("ml-")
if MOVIELENS_DATASET not in {"1m", "25m"}:
    raise ValueError("MOVIELENS_DATASET must be '1m', 'ml-1m', '25m', or 'ml-25m'.")
DATASET_DIR_NAME = f"ml-{MOVIELENS_DATASET}"
DATASET_LABEL = f"MovieLens-{MOVIELENS_DATASET.upper()}"
DATA_DIR = PROJECT_DIR / DATASET_DIR_NAME
RATINGS_FILE = DATA_DIR / ("ratings.dat" if MOVIELENS_DATASET == "1m" else "ratings.csv")
MOVIES_FILE = DATA_DIR / ("movies.dat" if MOVIELENS_DATASET == "1m" else "movies.csv")
RESULTS_DIR = PROJECT_DIR / "results" / DATASET_DIR_NAME
RESULTS_FILE = RESULTS_DIR / "ml_fixed_T_results.csv"

movie_selection_mode = "top_K_minus_1_by_rating_count_plus_lowest_positive_mu_target"


def _dataset_paths(dataset):
    data_dir = PROJECT_DIR / f"ml-{dataset}"
    ratings_file = data_dir / ("ratings.dat" if dataset == "1m" else "ratings.csv")
    movies_file = data_dir / ("movies.dat" if dataset == "1m" else "movies.csv")
    if not ratings_file.exists():
        raise FileNotFoundError(f"Cannot find {ratings_file}.")
    return data_dir, ratings_file, movies_file


def _rating_bounds(dataset):
    return (1.0, 5.0) if dataset == "1m" else (0.5, 5.0)


@lru_cache(maxsize=None)
def _load_movielens_base(dataset):
    """Read a MovieLens release once; compact half-star values into bytearrays."""
    data_dir, ratings_file, movies_file = _dataset_paths(dataset)
    rating_min, rating_max = _rating_bounds(dataset)

    movie_titles = {}
    if movies_file.exists() and dataset == "1m":
        with movies_file.open("r", encoding="latin-1", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("::", 2)
                if len(parts) >= 2:
                    movie_titles[int(parts[0])] = parts[1]
    elif movies_file.exists():
        with movies_file.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                movie_titles[int(row["movieId"])] = row["title"]

    # Store rating*2 as one byte.  This keeps all 25M empirical rewards in
    # roughly 25 MB instead of retaining 25M Python float objects.
    movie_encoded_rewards = defaultdict(bytearray)
    movie_count_and_encoded_sum = defaultdict(lambda: [0, 0])

    def add_rating(movie_id, rating):
        encoded = int(round(2.0 * rating))
        if abs(encoded / 2.0 - rating) > 1e-12 or not (2.0 * rating_min <= encoded <= 2.0 * rating_max):
            raise ValueError(f"Unsupported MovieLens rating {rating!r} for {dataset}.")
        movie_encoded_rewards[movie_id].append(encoded)
        stats = movie_count_and_encoded_sum[movie_id]
        stats[0] += 1
        stats[1] += encoded

    if dataset == "1m":
        with ratings_file.open("r", encoding="latin-1", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("::")
                if len(parts) != 4:
                    raise ValueError(f"Bad ratings.dat row: {line!r}")
                add_rating(int(parts[1]), float(parts[2]))
    else:
        with ratings_file.open("r", encoding="utf-8", newline="") as f:
            header = f.readline().rstrip("\r\n")
            if header != "userId,movieId,rating,timestamp":
                raise ValueError(f"Unexpected ratings.csv header: {header!r}")
            # ratings.csv contains four numeric, unquoted columns.  Avoid the
            # per-row DictReader overhead across 25M rows; movies.csv still
            # uses the full CSV parser because titles can contain commas.
            for line in f:
                _, movie_id_text, rating_text, _ = line.rstrip("\r\n").split(",")
                movie_id = int(movie_id_text)
                encoded = int(round(2.0 * float(rating_text)))
                if not (2.0 * rating_min <= encoded <= 2.0 * rating_max):
                    raise ValueError(f"Unsupported MovieLens rating {rating_text!r} for {dataset}.")
                movie_encoded_rewards[movie_id].append(encoded)
                stats = movie_count_and_encoded_sum[movie_id]
                stats[0] += 1
                stats[1] += encoded

    return {
        "data_dir": data_dir,
        "movie_titles": movie_titles,
        "movie_encoded_rewards": dict(movie_encoded_rewards),
        "movie_count_and_encoded_sum": dict(movie_count_and_encoded_sum),
        "rating_min": rating_min,
        "rating_max": rating_max,
    }


def load_movielens(K_value=K, dataset=None):
    dataset = MOVIELENS_DATASET if dataset is None else dataset.lower().removeprefix("ml-")
    if dataset not in {"1m", "25m"}:
        raise ValueError("dataset must be '1m', 'ml-1m', '25m', or 'ml-25m'.")
    base = _load_movielens_base(dataset)
    rating_min = base["rating_min"]
    rating_range = base["rating_max"] - rating_min

    movie_stats = []
    for movie_id, (count_i, encoded_sum_i) in base["movie_count_and_encoded_sum"].items():
        normalized_sum_i = encoded_sum_i / 2.0 - count_i * rating_min
        normalized_sum_i /= rating_range
        mu_i = normalized_sum_i / count_i
        movie_stats.append((movie_id, count_i, normalized_sum_i, mu_i))

    if len(movie_stats) < K_value:
        raise ValueError(f"MovieLens has {len(movie_stats)} movies, fewer than K={K_value}.")

    target_candidates = [row for row in movie_stats if row[1] >= TARGET_MIN_COUNT and row[3] > 0]
    if not target_candidates:
        raise ValueError("No MovieLens movie has a positive normalized empirical mean.")

    target_row = min(target_candidates, key=lambda row: (row[3], -row[1], row[0]))
    non_target_candidates = [row for row in movie_stats if row[0] != target_row[0]]
    selected_by_count = sorted(non_target_candidates, key=lambda row: (-row[1], row[0]))[: K_value - 1]
    selected_by_count.append(target_row)
    selected_arms = sorted(selected_by_count, key=lambda row: (-row[3], -row[1], row[0]))
    selected_movie_ids = [row[0] for row in selected_arms]
    selected_counts = np.array([row[1] for row in selected_arms], dtype=int)
    selected_reward_sums = np.array([row[2] for row in selected_arms], dtype=float)
    mu = np.array([row[3] for row in selected_arms], dtype=float)

    if not np.all(mu[:-1] >= mu[1:] - 1e-15):
        raise RuntimeError("Selected MovieLens arms are not sorted by descending mu_i.")
    if K_value == K and target_arm != K:
        raise RuntimeError("This runner uses the paper convention target arm = K.")

    selected_reward_arrays = []
    for movie_id in selected_movie_ids:
        encoded = base["movie_encoded_rewards"][movie_id]
        ratings_i = np.frombuffer(encoded, dtype=np.uint8).astype(float) / 2.0
        rewards_i = (ratings_i - rating_min) / rating_range
        selected_reward_arrays.append(rewards_i)
    metadata = {
        "dataset": dataset,
        "dataset_label": f"MovieLens-{dataset.upper()}",
        "data_dir": str(base["data_dir"]),
        "movie_titles": base["movie_titles"],
        "selected_movie_ids": selected_movie_ids,
        "selected_counts": selected_counts,
        "selected_reward_sums": selected_reward_sums,
        "selected_reward_arrays": selected_reward_arrays,
        "mu": mu,
        "rating_min": base["rating_min"],
        "rating_max": base["rating_max"],
        "reward_source": "empirical_normalized_rating_with_replacement",
        "target_min_count": TARGET_MIN_COUNT,
        "target_selection": "smallest positive empirical mean",
    }
    return metadata


def draw_empirical_reward(reward_arrays, rng, arm):
    rewards = reward_arrays[arm]
    if len(rewards) == 0:
        raise ValueError(f"Arm {arm} has no empirical rewards.")
    return float(rewards[rng.integers(0, len(rewards))])


def sample_clean_warm_start(metadata, repeat_id):
    rng_clean = np.random.default_rng(seed + repeat_id)
    clean_rewards = []
    for rewards_i in metadata["selected_reward_arrays"]:
        replace_i = len(rewards_i) < N0_i
        clean_rewards.append(rng_clean.choice(rewards_i, size=N0_i, replace=replace_i))
    clean_sum = np.array([arr.sum() for arr in clean_rewards], dtype=float)
    clean_mean = clean_sum / float(N0_i)
    return clean_sum, clean_mean


def configure_bandit(metadata, repeat_id):
    mu = metadata["mu"]
    bandit.K = K
    bandit.target_arm = target_arm
    bandit.delta = delta
    bandit.xi = xi
    bandit.r_l = r_l
    bandit.r_u = r_u
    bandit.R = R
    bandit.reward_distribution = reward_distribution
    bandit.reward_source = "empirical"
    bandit.empirical_reward_arrays = metadata["selected_reward_arrays"]
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    metadata = load_movielens()

    rows = []
    clean_summaries = []
    for repeat_id in range(NUM_REPEATS):
        configure_bandit(metadata, repeat_id)
        clean_sum, clean_mean = sample_clean_warm_start(metadata, repeat_id)
        clean_summaries.append(clean_sum.tolist())
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
    print(f"Bounded Attack on Stochastic Warm-Start Bandits: {DATASET_LABEL} Fixed-T Sweep")
    print("=" * 100)
    print(f"data_dir = {metadata['data_dir']}")
    print(f"movie_selection_mode = {movie_selection_mode}")
    print("target_selection = smallest positive empirical mean")
    print(f"reward_distribution = {reward_distribution}")
    print(f"reward_source = {metadata['reward_source']}")
    print(f"rating_normalization = (rating - {metadata['rating_min']}) / "
          f"({metadata['rating_max']} - {metadata['rating_min']})")
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
            f"normalized_reward_sum={metadata['selected_reward_sums'][idx - 1]:.6f}, "
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
