import csv
import math
from collections import defaultdict
from pathlib import Path

import mlrunner

PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR
K_RESULTS_FILE = mlrunner.RESULTS_DIR / "k_sweep_results.csv"
BASELINE_RESULTS_FILE = mlrunner.RESULTS_DIR / "baseline_comparison_results.csv"
FIG_FILE = ROOT_DIR / "fig" / "k_sweep_cost_decomposition.tex"
TS_FIG_FILE = ROOT_DIR / "fig" / "k_sweep_cost_decomposition_ts.tex"
UCB_COST_FIG_FILE = ROOT_DIR / "fig" / "ucb_cost_experiments.tex"
UCB_RATIO_FIG_FILE = ROOT_DIR / "fig" / "ucb_ratio_experiments.tex"
TS_COST_FIG_FILE = ROOT_DIR / "fig" / "ts_cost_experiments.tex"
TS_RATIO_FIG_FILE = ROOT_DIR / "fig" / "ts_ratio_experiments.tex"


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the corresponding experiment runner first.")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = "target_online_ratio" if path == K_RESULTS_FILE else "H_base"
        if required not in (reader.fieldnames or []):
            raise RuntimeError(f"Stale result schema in {path}; rerun the corresponding experiment runner.")
        return list(reader)


def mean_std(values):
    mean = sum(values) / len(values)
    if len(values) <= 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def aggregate_construction(rows):
    grouped = defaultdict(list)
    s_t = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key_base = (row["algorithm"], int(row["K"]))
        grouped[key_base + ("total",)].append(float(row["Cost_n"]))
        grouped[key_base + ("target",)].append(float(row["target_cost"]))
        grouped[key_base + ("non_target_avg",)].append(float(row["non_target_avg_cost"]))
        # N_K/T counts the clean, injected, and actually selected target-arm
        # observations in the learner's fixed internal horizon.
        clean_total = float(row["T0"]) - float(row["Cost_n"])
        clean_target = clean_total / int(row["K"])
        target_online_ratio = float(row["target_online_ratio"])
        target_total = clean_target + float(row["target_cost"]) + target_online_ratio * float(row["H"])
        grouped[key_base + ("target_total_ratio",)].append(target_total / float(row["T"]))
        grouped[key_base + ("target_online_ratio",)].append(target_online_ratio)
        s_t[int(row["K"])] = float(row["S_T"])
    return {key: mean_std(values) for key, values in grouped.items()}, s_t


def aggregate_baseline(rows):
    cost = defaultdict(list)
    ratio = defaultdict(list)
    post_ratio = defaultdict(list)
    for row in rows:
        if row.get("status") == "infeasible":
            continue
        K = int(row["K"])
        algorithm = row["algorithm"]
        cost[(algorithm, K)].append(float(row["native_cost"]))
        if row["target_ratio"] != "":
            ratio[(algorithm, K)].append(float(row["target_ratio"]))
        if row.get("post_phase_ratio", "") != "":
            post_ratio[(algorithm, K)].append(float(row["post_phase_ratio"]))
    return (
        {key: mean_std(values) for key, values in cost.items()},
        {key: mean_std(values) for key, values in ratio.items()},
        {key: mean_std(values) for key, values in post_ratio.items()},
    )


def construction_coords(stats, algorithm, metric, which):
    points = []
    for alg, K, met in stats:
        if alg != algorithm or met != metric:
            continue
        mean, std = stats[(alg, K, met)]
        if which == "mean":
            y = mean
        elif which == "upper":
            y = mean + std
        elif which == "lower":
            y = max(0.0, mean - std)
        else:
            raise ValueError(which)
        points.append((K, y))
    points.sort()
    return " ".join(f"({x},{y:.8g})" for x, y in points)


def simple_coords(stats, algorithm, which):
    points = []
    for alg, K in stats:
        if alg != algorithm:
            continue
        mean, std = stats[(alg, K)]
        if which == "mean":
            y = mean
        elif which == "upper":
            y = mean + std
        elif which == "lower":
            y = max(0.0, mean - std)
        else:
            raise ValueError(which)
        points.append((K, y))
    points.sort()
    return " ".join(f"({x},{y:.8g})" for x, y in points)


def theory_coords(stats, s_t, algorithms):
    scale = 0.0
    for algorithm in algorithms:
        for alg, K, metric in stats:
            if alg == algorithm and metric == "total":
                mean, std = stats[(alg, K, metric)]
                scale = max(scale, (mean + std) / s_t[K])
    return " ".join(f"({K},{scale * s_t[K]:.8g})" for K in sorted(s_t))


def theory_plot(stats, s_t, algorithms):
    return rf"""
\addplot[
    color=black!55,
    dashed,
    thick,
    mark=none
] coordinates {{{theory_coords(stats, s_t, algorithms)}}};
"""


def shaded(name, color, mark, mean, upper, lower, legend, style="solid", add_legend=True):
    if not mean.strip():
        return ""
    legend_line = rf"\addlegendentry{{{legend}}}" if add_legend else ""
    return rf"""
\addplot[name path={name}upper, draw=none, forget plot] coordinates {{{upper}}};
\addplot[name path={name}lower, draw=none, forget plot] coordinates {{{lower}}};
\addplot[{color}, fill opacity=0.08, draw=none, forget plot]
    fill between[of={name}upper and {name}lower];
\addplot[
    color={color},
    opacity=0.76,
    thick,
    {style},
    mark={mark},
    mark options={{fill=white, draw={color}, fill opacity=0.50, draw opacity=0.76}}
] coordinates {{{mean}}};
{legend_line}
"""


def construction_target_plot(stats, algorithm_prefix, prefix):
    direct = f"{algorithm_prefix} direct"
    appendix = f"{algorithm_prefix} appendix"
    return (
        shaded(
            f"{prefix}directtarget",
            "myblue",
            "*",
            construction_coords(stats, direct, "target", "mean"),
            construction_coords(stats, direct, "target", "upper"),
            construction_coords(stats, direct, "target", "lower"),
            "Original target",
        )
        + shaded(
            f"{prefix}apptarget",
            "myorange",
            "triangle*",
            construction_coords(stats, appendix, "target", "mean"),
            construction_coords(stats, appendix, "target", "upper"),
            construction_coords(stats, appendix, "target", "lower"),
            "Appendix target",
        )
    )


def construction_non_target_plot(stats, algorithm_prefix, prefix):
    direct = f"{algorithm_prefix} direct"
    appendix = f"{algorithm_prefix} appendix"
    return (
        shaded(
            f"{prefix}directnt",
            "mygreen",
            "square*",
            construction_coords(stats, direct, "non_target_avg", "mean"),
            construction_coords(stats, direct, "non_target_avg", "upper"),
            construction_coords(stats, direct, "non_target_avg", "lower"),
            "Original non-target avg.",
            "solid",
        )
        + shaded(
            f"{prefix}appnt",
            "mypurple",
            "diamond*",
            construction_coords(stats, appendix, "non_target_avg", "mean"),
            construction_coords(stats, appendix, "non_target_avg", "upper"),
            construction_coords(stats, appendix, "non_target_avg", "lower"),
            "Appendix non-target avg.",
            "solid",
        )
    )


def construction_cost_plot(stats, baseline_cost_stats, algorithm_prefix, prefix, add_legend):
    direct = f"{algorithm_prefix} direct"
    appendix = f"{algorithm_prefix} appendix"
    if algorithm_prefix == "UCB":
        baseline = (
            simple_plot_key(
                baseline_cost_stats,
                f"{prefix}jun",
                "black!65",
                "o",
                "Jun2018 clipped UCB",
                "Heuristic",
                add_legend=add_legend,
            )
            + simple_plot_key(
                baseline_cost_stats,
                f"{prefix}xu",
                "mypurple",
                "o",
                "Xu2021 observation-free UCB",
                r"\citet{xu2021observation}",
                add_legend=add_legend,
            )
        )
    else:
        baseline = (
            simple_plot_key(
                baseline_cost_stats,
                f"{prefix}zuo",
                "black!65",
                "o",
                "Zuo2024 clipped TS",
                "Heuristic",
                add_legend=add_legend,
            )
            + simple_plot_key(
                baseline_cost_stats,
                f"{prefix}xu",
                "mypurple",
                "o",
                "Xu2021 observation-free TS",
                r"\citet{xu2021observation}",
                add_legend=add_legend,
            )
        )
    return (
        baseline
        + shaded(
            f"{prefix}apptotal",
            "myorange",
            "o",
            construction_coords(stats, appendix, "total", "mean"),
            construction_coords(stats, appendix, "total", "upper"),
            construction_coords(stats, appendix, "total", "lower"),
            "Threshold total",
            add_legend=add_legend,
        )
        + shaded(
            f"{prefix}directtotal",
            "myblue",
            "x",
            construction_coords(stats, direct, "total", "mean"),
            construction_coords(stats, direct, "total", "upper"),
            construction_coords(stats, direct, "total", "lower"),
            "Direct total",
            add_legend=add_legend,
        )
        + shaded(
            f"{prefix}apptarget",
            "myorange",
            "triangle*",
            construction_coords(stats, appendix, "target", "mean"),
            construction_coords(stats, appendix, "target", "upper"),
            construction_coords(stats, appendix, "target", "lower"),
            "Threshold target",
            add_legend=add_legend,
        )
        + shaded(
            f"{prefix}directtarget",
            "myblue",
            "diamond*",
            construction_coords(stats, direct, "target", "mean"),
            construction_coords(stats, direct, "target", "upper"),
            construction_coords(stats, direct, "target", "lower"),
            "Direct target",
            add_legend=add_legend,
        )
        + shaded(
            f"{prefix}appnt",
            "myorange",
            "square*",
            construction_coords(stats, appendix, "non_target_avg", "mean"),
            construction_coords(stats, appendix, "non_target_avg", "upper"),
            construction_coords(stats, appendix, "non_target_avg", "lower"),
            "Threshold non-target avg.",
            add_legend=add_legend,
        )
        + shaded(
            f"{prefix}directnt",
            "myblue",
            "+",
            construction_coords(stats, direct, "non_target_avg", "mean"),
            construction_coords(stats, direct, "non_target_avg", "upper"),
            construction_coords(stats, direct, "non_target_avg", "lower"),
            "Direct non-target avg.",
            add_legend=add_legend,
        )
    )


def simple_plot(stats, name, color, mark, legend, style="solid", add_legend=True):
    return simple_plot_key(stats, name, color, mark, legend, legend, style, add_legend)


def simple_plot_key(stats, name, color, mark, algorithm, legend, style="solid", add_legend=True):
    return shaded(
        name,
        color,
        mark,
        simple_coords(stats, algorithm, "mean"),
        simple_coords(stats, algorithm, "upper"),
        simple_coords(stats, algorithm, "lower"),
        legend,
        style,
        add_legend,
    )


def cost_axis_block(body, ylabel, ticks):
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[
    width=\linewidth,
    height=3.8cm,
    xlabel={{}},
    ylabel={{{{\small {ylabel}}}}},
    ymin=0,
    tick align=outside,
    tick pos=left,
    xtick={{{ticks}}},
    xticklabel style={{font=\scriptsize}},
    yticklabel style={{font=\scriptsize}},
    axis line style={{black}},
]
{body}
\end{{axis}}
\end{{tikzpicture}}"""


def shared_cost_legend():
    return r"""\begin{tikzpicture}
\begin{axis}[
    hide axis,
    xmin=0,
    xmax=1,
    ymin=0,
    ymax=1,
    width=0.96\linewidth,
    height=2.0cm,
    legend columns=5,
    legend style={
        draw opacity=1,
        fill opacity=0.55,
        text opacity=1,
        font=\fontsize{5}{6}\selectfont,
        row sep=0pt,
        column sep=2pt,
        inner xsep=2pt,
        inner ysep=1pt,
        at={(0.5,0)},
        anchor=center
    },
]
\addlegendimage{black!65, thick, mark=square*, mark options={fill=white, draw=black!65}}
\addlegendentry{Heuristic}
\addlegendimage{myorange, thick, mark=o, mark options={fill=white, draw=myorange}}
\addlegendentry{Threshold total}
\addlegendimage{myblue, thick, mark=x, mark options={draw=myblue}}
\addlegendentry{Direct total}
\addlegendimage{myorange, thick, mark=triangle*, mark options={fill=white, draw=myorange}}
\addlegendentry{Threshold target}
\addlegendimage{myblue, thick, mark=diamond*, mark options={fill=white, draw=myblue}}
\addlegendentry{Direct target}
\addlegendimage{myorange, thick, mark=square*, mark options={fill=white, draw=myorange}}
\addlegendentry{Threshold non-target avg.}
\addlegendimage{myblue, thick, mark=+, mark options={draw=myblue}}
\addlegendentry{Direct non-target avg.}
\addlegendimage{black!55, dashed, thick, mark=none}
\addlegendentry{$S_T$}
\end{axis}
\end{tikzpicture}"""


def axis_block(body, ylabel, ymax, ticks, legend_pos="north west"):
    ymax_line = "" if ymax is None else f"    ymax={ymax},\n"
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[
    width=\linewidth,
    height=3.8cm,
    xlabel={{}},
    ylabel={{{{\small {ylabel}}}}},
    ymin=0,
{ymax_line}    tick align=outside,
    tick pos=left,
    xtick={{{ticks}}},
    xticklabel style={{font=\scriptsize}},
    yticklabel style={{font=\scriptsize}},
    axis line style={{black}},
    legend pos={legend_pos},
    legend style={{
        fill opacity=0.55,
        draw opacity=1,
        text opacity=1,
        font=\fontsize{{5}}{{6}}\selectfont,
        row sep=0pt,
        inner xsep=2pt,
        inner ysep=1pt
    }},
]
{body}
\end{{axis}}
\end{{tikzpicture}}"""


def axis_block_no_legend(body, ylabel, ymax, ticks):
    ymax_line = "" if ymax is None else f"    ymax={ymax},\n"
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[
    width=\linewidth,
    height=3.8cm,
    xlabel={{}},
    ylabel={{{{\small {ylabel}}}}},
    ymin=0,
{ymax_line}    tick align=outside,
    tick pos=left,
    xtick={{{ticks}}},
    xticklabel style={{font=\scriptsize}},
    yticklabel style={{font=\scriptsize}},
    axis line style={{black}},
]
{body}
\end{{axis}}
\end{{tikzpicture}}"""


def shared_ratio_legend():
    return r"""\begin{tikzpicture}
\begin{axis}[
    hide axis,
    xmin=0,
    xmax=1,
    ymin=0,
    ymax=1,
    width=0.90\linewidth,
    height=1.8cm,
    legend columns=3,
    legend style={
        draw opacity=1,
        fill opacity=0.55,
        text opacity=1,
        font=\fontsize{5}{6}\selectfont,
        row sep=0pt,
        column sep=4pt,
        inner xsep=2pt,
        inner ysep=1pt,
        at={(0.5,0)},
        anchor=center
    },
]
\addlegendimage{myorange, thick, mark=square*, mark options={fill=white, draw=myorange}}
\addlegendentry{Threshold $N_K/T$}
\addlegendimage{myblue, thick, mark=x, mark options={draw=myblue}}
\addlegendentry{Direct $N_K/T$}
\addlegendimage{myorange, thick, solid, mark=o, mark options={fill=white, draw=myorange}}
\addlegendentry{Threshold $N_K^{\on}/H$}
\addlegendimage{myblue, thick, solid, mark=triangle*, mark options={fill=white, draw=myblue}}
\addlegendentry{Direct $N_K^{\on}/H$}
\addlegendimage{black!65, thick, mark=square*, mark options={fill=white, draw=black!65}}
\addlegendentry{Heuristic $N_K/T$}
\end{axis}
\end{tikzpicture}"""


def combined_legend():
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[
    hide axis,
    xmin=0, xmax=1, ymin=0, ymax=1,
    width=0.98\linewidth,
    height=2.35cm,
    legend columns=5,
    legend style={{
        draw opacity=1,
        fill opacity=0.55,
        text opacity=1,
        font=\fontsize{{5}}{{6}}\selectfont,
        row sep=0pt,
        column sep=2pt,
        inner xsep=2pt,
        inner ysep=1pt,
        at={{(0.5,0.5)}},
        anchor=center,
    }},
]
\addlegendimage{{black!65, thick, mark=o, mark options={{fill=white, draw=black!65}}}}
\addlegendentry{{Heuristic cost}}
\addlegendimage{{black!65, thick, mark=square*, mark options={{fill=white, draw=black!65}}}}
\addlegendentry{{Heuristic $N_K/T$}}
\addlegendimage{{myorange, thick, mark=o, mark options={{fill=white, draw=myorange}}}}
\addlegendentry{{Threshold total}}
\addlegendimage{{myblue, thick, mark=x, mark options={{draw=myblue}}}}
\addlegendentry{{Direct total}}
\addlegendimage{{myorange, thick, mark=triangle*, mark options={{fill=white, draw=myorange}}}}
\addlegendentry{{Threshold target}}
\addlegendimage{{myblue, thick, mark=diamond*, mark options={{fill=white, draw=myblue}}}}
\addlegendentry{{Direct target}}
\addlegendimage{{myorange, thick, mark=square*, mark options={{fill=white, draw=myorange}}}}
\addlegendentry{{Threshold non-target avg. / $N_K/T$}}
\addlegendimage{{myblue, thick, mark=+, mark options={{draw=myblue}}}}
\addlegendentry{{Direct non-target avg. / $N_K/T$}}
\addlegendimage{{myorange, thick, mark=o, mark options={{fill=white, draw=myorange}}}}
\addlegendentry{{Threshold $N_K^{{\on}}/H$}}
\addlegendimage{{myblue, thick, mark=triangle*, mark options={{fill=white, draw=myblue}}}}
\addlegendentry{{Direct $N_K^{{\on}}/H$}}
\addlegendimage{{black!55, dashed, thick, mark=none}}
\addlegendentry{{$S_T$}}
\end{{axis}}
\end{{tikzpicture}}"""


def metric_legend(kind):
    if kind == "cost":
        entries = r"""
\addlegendimage{black!65, thick, mark=o, mark options={fill=white, draw=black!65}}\addlegendentry{Heuristic}
\addlegendimage{myorange, thick, mark=o, mark options={fill=white, draw=myorange}}\addlegendentry{Threshold total}
\addlegendimage{myblue, thick, mark=x, mark options={draw=myblue}}\addlegendentry{Direct total}
\addlegendimage{myorange, thick, mark=triangle*, mark options={fill=white, draw=myorange}}\addlegendentry{Threshold target}
\addlegendimage{myblue, thick, mark=diamond*, mark options={fill=white, draw=myblue}}\addlegendentry{Direct target}
\addlegendimage{myorange, thick, mark=square*, mark options={fill=white, draw=myorange}}\addlegendentry{Threshold non-target avg.}
\addlegendimage{myblue, thick, mark=+, mark options={draw=myblue}}\addlegendentry{Direct non-target avg.}
\addlegendimage{black!55, dashed, thick, mark=none}\addlegendentry{Scaled $S_T$}
"""
        columns, height = 4, "2.0cm"
    else:
        entries = r"""
\addlegendimage{black!45, thick, mark=x}\addlegendentry{Clean $N_K/T$}
\addlegendimage{black!65, thick, mark=square*, mark options={fill=white, draw=black!65}}\addlegendentry{Heuristic $N_K/T$}
\addlegendimage{myorange, thick, mark=square*, mark options={fill=white, draw=myorange}}\addlegendentry{Threshold $N_K/T$}
\addlegendimage{myblue, thick, mark=x, mark options={draw=myblue}}\addlegendentry{Direct $N_K/T$}
\addlegendimage{myorange, thick, mark=o, mark options={fill=white, draw=myorange}}\addlegendentry{Threshold $N_K^{\on}/H$}
\addlegendimage{myblue, thick, mark=triangle*, mark options={fill=white, draw=myblue}}\addlegendentry{Direct $N_K^{\on}/H$}
"""
        columns, height = 3, "2.0cm"
    return rf"""\begin{{tikzpicture}}\begin{{axis}}[
hide axis, xmin=0, xmax=1, ymin=0, ymax=1, width=0.96\linewidth, height={height},
legend columns={columns}, legend style={{draw=black!40, fill=white, font=\fontsize{{5}}{{6}}\selectfont,
column sep=2pt, row sep=0pt, inner xsep=2pt, inner ysep=1pt, at={{(0.5,0.5)}}, anchor=center}}]
{entries}\end{{axis}}\end{{tikzpicture}}"""


def write_figure(k_rows, baseline_rows):
    construction_stats, s_t = aggregate_construction(k_rows)
    baseline_cost_stats, baseline_ratio_stats, baseline_post_ratio_stats = aggregate_baseline(baseline_rows)
    k_values = sorted({int(row["K"]) for row in k_rows})
    ticks = ",".join(str(k) for k in k_values)

    ucb_cost = theory_plot(construction_stats, s_t, ["UCB appendix", "UCB direct"]) + construction_cost_plot(
        construction_stats, baseline_cost_stats, "UCB", "ucbk", False
    )
    ts_cost = theory_plot(construction_stats, s_t, ["TS appendix", "TS direct"]) + construction_cost_plot(
        construction_stats, baseline_cost_stats, "TS", "tsk", False
    )
    ucb_direct = "UCB direct"
    ucb_appendix = "UCB appendix"
    ts_direct = "TS direct"
    ts_appendix = "TS appendix"
    ucb_ratio = (
        shaded("ucbappratio", "myorange", "square*",
               construction_coords(construction_stats, ucb_appendix, "target_total_ratio", "mean"),
               construction_coords(construction_stats, ucb_appendix, "target_total_ratio", "upper"),
               construction_coords(construction_stats, ucb_appendix, "target_total_ratio", "lower"),
               "Threshold $N_K/T$", add_legend=False)
        + shaded("ucbdirectratio", "myblue", "x",
                 construction_coords(construction_stats, ucb_direct, "target_total_ratio", "mean"),
                 construction_coords(construction_stats, ucb_direct, "target_total_ratio", "upper"),
                 construction_coords(construction_stats, ucb_direct, "target_total_ratio", "lower"),
                 "Direct $N_K/T$", add_legend=False)
        + shaded("ucbapponlineratio", "myorange", "o",
                 construction_coords(construction_stats, ucb_appendix, "target_online_ratio", "mean"),
                 construction_coords(construction_stats, ucb_appendix, "target_online_ratio", "upper"),
                 construction_coords(construction_stats, ucb_appendix, "target_online_ratio", "lower"),
                 "Threshold $N_K^{\\on}/H$", style="solid", add_legend=False)
        + shaded("ucbonlineratio", "myblue", "triangle*",
                 construction_coords(construction_stats, ucb_direct, "target_online_ratio", "mean"),
                 construction_coords(construction_stats, ucb_direct, "target_online_ratio", "upper"),
                 construction_coords(construction_stats, ucb_direct, "target_online_ratio", "lower"),
                 "Direct $N_K^{\\on}/H$", style="solid", add_legend=False)
        + simple_plot(baseline_ratio_stats, "junr", "black!65", "square*", "Jun2018 clipped UCB", add_legend=False)
        + simple_plot(baseline_ratio_stats, "xur", "mypurple", "square*", "Xu2021 observation-free UCB", add_legend=False)
        + simple_plot(baseline_post_ratio_stats, "xupostucr", "mypurple", "x", "Xu2021 observation-free UCB", style="solid", add_legend=False)
    )
    ts_ratio = (
        shaded("tsappratio", "myorange", "square*",
               construction_coords(construction_stats, ts_appendix, "target_total_ratio", "mean"),
               construction_coords(construction_stats, ts_appendix, "target_total_ratio", "upper"),
               construction_coords(construction_stats, ts_appendix, "target_total_ratio", "lower"),
               "Threshold $N_K/T$", add_legend=False)
        + shaded("tsdirectratio", "myblue", "x",
                 construction_coords(construction_stats, ts_direct, "target_total_ratio", "mean"),
                 construction_coords(construction_stats, ts_direct, "target_total_ratio", "upper"),
                 construction_coords(construction_stats, ts_direct, "target_total_ratio", "lower"),
                 "Direct $N_K/T$", add_legend=False)
        + shaded("tsapponlineratio", "myorange", "o",
                 construction_coords(construction_stats, ts_appendix, "target_online_ratio", "mean"),
                 construction_coords(construction_stats, ts_appendix, "target_online_ratio", "upper"),
                 construction_coords(construction_stats, ts_appendix, "target_online_ratio", "lower"),
                 "Threshold $N_K^{\\on}/H$", style="solid", add_legend=False)
        + shaded("tsonlineratio", "myblue", "triangle*",
                 construction_coords(construction_stats, ts_direct, "target_online_ratio", "mean"),
                 construction_coords(construction_stats, ts_direct, "target_online_ratio", "upper"),
                 construction_coords(construction_stats, ts_direct, "target_online_ratio", "lower"),
                 "Direct $N_K^{\\on}/H$", style="solid", add_legend=False)
        + simple_plot(baseline_ratio_stats, "zuor", "black!65", "square*", "Zuo2024 clipped TS", add_legend=False)
        + simple_plot(baseline_ratio_stats, "xutsr", "mypurple", "square*", "Xu2021 observation-free TS", add_legend=False)
        + simple_plot(baseline_post_ratio_stats, "xuposttsr", "mypurple", "x", "Xu2021 observation-free TS", style="solid", add_legend=False)
    )

    content = rf"""% Auto-generated by exp/generate_k_sweep_tikz.py.
% Source files: {K_RESULTS_FILE}; {BASELINE_RESULTS_FILE}
\begin{{figure*}}[t]
\centering
\subfloat[UCB cost versus $K$]{{%
\begin{{minipage}}[t]{{0.49\linewidth}}
\vspace{{0pt}}
\centering
{cost_axis_block(ucb_cost, "Cost", ticks)}
\end{{minipage}}}}
\hfill
\subfloat[Thompson Sampling cost versus $K$]{{%
\begin{{minipage}}[t]{{0.49\linewidth}}
\vspace{{0pt}}
\centering
{cost_axis_block(ts_cost, "Cost", ticks)}
\end{{minipage}}}}
\par\vspace{{0.25em}}
{shared_cost_legend()}
\par\vspace{{0.75em}}
\subfloat[UCB target selection ratio]{{%
\begin{{minipage}}[t]{{0.49\linewidth}}
\vspace{{0pt}}
\centering
{axis_block_no_legend(ucb_ratio, "Target ratio", "1.05", ticks)}
\end{{minipage}}}}
\hfill
\subfloat[Thompson Sampling target selection ratio]{{%
\begin{{minipage}}[t]{{0.49\linewidth}}
\vspace{{0pt}}
\centering
{axis_block_no_legend(ts_ratio, "Target ratio", "1.05", ticks)}
\end{{minipage}}}}
\par\vspace{{0.15em}}
{shared_ratio_legend()}
\caption{{{mlrunner.DATASET_LABEL} $K$-sweep with $T=200{{,}}000$ over ten repeats. For each $K$, the $K-1$ highest-count movies are non-target arms and the smallest-positive-mean movie is the target. Panels (a)--(b) report total, target-arm, and average per-non-target cost for both offline constructions and the bounded online heuristics. Panels (c)--(d) distinguish total target-arm exposure $N_K/T$ from post-deployment success $N_K^{{\on}}/H$.}}
\label{{fig:k_sweep_cost_decomposition}}
\end{{figure*}}
"""
    def panel(start, end):
        i = content.index(start)
        j = content.index(end, i)
        return content[i:j].rstrip()

    ucb_cost_panel = panel(r"\subfloat[UCB cost versus $K$]", "\\hfill\n\\subfloat[Thompson Sampling cost versus $K$]")
    ts_cost_panel = panel(r"\subfloat[Thompson Sampling cost versus $K$]", r"\par\vspace{0.25em}")
    ucb_ratio_panel = panel(r"\subfloat[UCB target selection ratio]", "\\hfill\n\\subfloat[Thompson Sampling target selection ratio]")
    ts_ratio_panel = panel(r"\subfloat[Thompson Sampling target selection ratio]", r"\par\vspace{0.15em}")

    ucb_content = rf"""% Auto-generated by exp/generate_k_sweep_tikz.py.
% Source files: {K_RESULTS_FILE}; {BASELINE_RESULTS_FILE}
\begin{{figure*}}[t]
\centering
{ucb_cost_panel}
\hfill
{ucb_ratio_panel}
\par\vspace{{0.15em}}
{combined_legend()}
\caption{{{mlrunner.DATASET_LABEL} UCB $K$-sweep with $T=200{{,}}000$ over ten repeats. Panel (a) reports total, target-arm, and average per-non-target cost for both offline constructions and the bounded heuristic. Panel (b) distinguishes $N_K/T$ from $N_K^{{\on}}/H$.}}
\label{{fig:k_sweep_cost_decomposition}}
\end{{figure*}}
"""
    ts_content = rf"""% Auto-generated by exp/generate_k_sweep_tikz.py.
% Source files: {K_RESULTS_FILE}; {BASELINE_RESULTS_FILE}
\begin{{figure*}}[t]
\centering
{ts_cost_panel}
\hfill
{ts_ratio_panel}
\par\vspace{{0.15em}}
{combined_legend()}
\caption{{{mlrunner.DATASET_LABEL} Thompson Sampling $K$-sweep with $T=200{{,}}000$ over ten repeats. Panel (a) reports total, target-arm, and average per-non-target cost for both offline constructions and the bounded heuristic. Panel (b) distinguishes $N_K/T$ from $N_K^{{\on}}/H$.}}
\label{{fig:k_sweep_cost_decomposition_ts}}
\end{{figure*}}
"""
    FIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    FIG_FILE.write_text(ucb_content, encoding="utf-8")
    TS_FIG_FILE.write_text(ts_content, encoding="utf-8")
    fragment_dir = ROOT_DIR / "fig" / "fragments"
    required = ["fixed_ucb_cost.tex", "fixed_ucb_ratio.tex", "fixed_ts_cost.tex", "fixed_ts_ratio.tex"]
    missing = [name for name in required if not (fragment_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing fixed-T fragments {missing}; run generate_fixed_T_tikz.py first")
    fixed = {name: (fragment_dir / name).read_text(encoding="utf-8") for name in required}

    def two_panel_figure(left, right, legend, caption, label):
        return rf"""% Auto-generated by exp/generate_k_sweep_tikz.py.
\begin{{figure*}}[t]
\centering
{left}
\hfill
{right}
\par\vspace{{0.15em}}
{legend}
\caption{{{caption}}}
\label{{{label}}}
\end{{figure*}}
"""

    UCB_COST_FIG_FILE.write_text(two_panel_figure(
        fixed["fixed_ucb_cost.tex"], ucb_cost_panel, metric_legend("cost"),
        f"UCB attack costs versus horizon $T$ and number of arms $K$ on {mlrunner.DATASET_LABEL}.",
        "fig:ucb_cost_experiments"), encoding="utf-8")
    UCB_RATIO_FIG_FILE.write_text(two_panel_figure(
        fixed["fixed_ucb_ratio.tex"], ucb_ratio_panel, metric_legend("ratio"),
        f"UCB target-arm ratios versus horizon $T$ and number of arms $K$ on {mlrunner.DATASET_LABEL}.",
        "fig:ucb_ratio_experiments"), encoding="utf-8")
    TS_COST_FIG_FILE.write_text(two_panel_figure(
        fixed["fixed_ts_cost.tex"], ts_cost_panel, metric_legend("cost"),
        f"Thompson Sampling attack costs versus horizon $T$ and number of arms $K$ on {mlrunner.DATASET_LABEL}.",
        "fig:ts_cost_experiments"), encoding="utf-8")
    TS_RATIO_FIG_FILE.write_text(two_panel_figure(
        fixed["fixed_ts_ratio.tex"], ts_ratio_panel, metric_legend("ratio"),
        f"Thompson Sampling target-arm ratios versus horizon $T$ and number of arms $K$ on {mlrunner.DATASET_LABEL}.",
        "fig:ts_ratio_experiments"), encoding="utf-8")
    print(f"wrote {FIG_FILE}")
    print(f"wrote {TS_FIG_FILE}")
    print(f"wrote {UCB_COST_FIG_FILE}")
    print(f"wrote {UCB_RATIO_FIG_FILE}")
    print(f"wrote {TS_COST_FIG_FILE}")
    print(f"wrote {TS_RATIO_FIG_FILE}")


def Main():
    write_figure(read_csv(K_RESULTS_FILE), read_csv(BASELINE_RESULTS_FILE))


if __name__ == "__main__":
    Main()
