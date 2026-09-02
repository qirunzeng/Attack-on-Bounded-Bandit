import csv
import math
from collections import defaultdict
from pathlib import Path

import mlrunner

PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR
RESULTS_FILE = mlrunner.RESULTS_DIR / "ml_fixed_T_results.csv"
BASELINE_RESULTS_FILE = mlrunner.RESULTS_DIR / "ml_fixed_T_baseline_results.csv"
FIG_FILE = ROOT_DIR / "fig" / "fixed_T_sweep.tex"
TS_FIG_FILE = ROOT_DIR / "fig" / "fixed_T_sweep_ts.tex"
FRAGMENT_DIR = ROOT_DIR / "fig" / "fragments"


def read_rows():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Missing {RESULTS_FILE}. Run mlrunner.py first.")
    with RESULTS_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "search_log_json" not in (reader.fieldnames or []):
            raise RuntimeError(f"Stale result schema in {RESULTS_FILE}; rerun mlrunner.py.")
        return list(reader)


def read_baseline_rows():
    if not BASELINE_RESULTS_FILE.exists():
        return []
    with BASELINE_RESULTS_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "H_base" not in (reader.fieldnames or []):
            raise RuntimeError(f"Stale result schema in {BASELINE_RESULTS_FILE}; rerun fixed_T_baseline_runner.py.")
        return list(reader)


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    if n <= 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, math.sqrt(var)


def aggregate(rows):
    cost_grouped = defaultdict(list)
    success_ratio_grouped = defaultdict(list)
    total_target_share_grouped = defaultdict(list)
    s_t_by_T = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        T = int(row["T"])
        algorithm = row["algorithm"]
        cost_grouped[(algorithm, T)].append(float(row["Cost_n"]))
        target_on = float(row.get("N_on_10") or 0.0)
        target_off = float(row.get("n_10") or 0.0)
        total_target_share_grouped[(algorithm, T)].append((mlrunner.N0_i + target_on + target_off) / T)
        if row["target_online_ratio"] != "" and "clean" not in algorithm:
            success_ratio_grouped[(algorithm, T)].append(float(row["target_online_ratio"]))
        s_t_by_T[T] = float(row["S_T"])

    cost_stats = {}
    for key, values in cost_grouped.items():
        cost_stats[key] = mean_std(values)

    success_ratio_stats = {}
    for key, values in success_ratio_grouped.items():
        success_ratio_stats[key] = mean_std(values)

    total_target_share_stats = {}
    for key, values in total_target_share_grouped.items():
        total_target_share_stats[key] = mean_std(values)
    return cost_stats, success_ratio_stats, total_target_share_stats, s_t_by_T


def aggregate_baseline(rows):
    cost_grouped = defaultdict(list)
    target_share_grouped = defaultdict(list)
    for row in rows:
        if row.get("status") == "infeasible":
            continue
        T = int(row["T"])
        algorithm = row["algorithm"]
        cost_grouped[(algorithm, T)].append(float(row["native_cost"]))
        if row.get("target_ratio") not in {None, ""}:
            target_share_grouped[(algorithm, T)].append(float(row["target_ratio"]))
    return (
        {key: mean_std(values) for key, values in cost_grouped.items()},
        {key: mean_std(values) for key, values in target_share_grouped.items()},
    )


def coord_series(stats, algorithm, which):
    points = []
    for alg, T in stats:
        if alg != algorithm:
            continue
        mean, std = stats[(alg, T)]
        if which == "mean":
            y = mean
        elif which == "upper":
            y = mean + std
        elif which == "lower":
            y = max(0.0, mean - std)
        else:
            raise ValueError(which)
        points.append((T, y))
    points.sort()
    return " ".join(f"({x},{y:.8g})" for x, y in points)


def theory_bound(stats, s_t_by_T, algorithms):
    c = 0.0
    for algorithm in algorithms:
        for alg, T in stats:
            if alg != algorithm:
                continue
            mean, std = stats[(alg, T)]
            c = max(c, (mean + std) / s_t_by_T[T])
    return " ".join(f"({T},{c * s_t_by_T[T]:.8g})" for T in sorted(s_t_by_T))


def xticks(rows):
    values = sorted({int(row["T"]) for row in rows})
    ticks = ",".join(str(v) for v in values)
    labels = []
    for value in values:
        if value >= 1_000_000:
            labels.append(f"{value / 1_000_000:g}M")
        else:
            labels.append(f"{value // 1000}k")
    return ticks, ",".join(labels)


def shaded_plot(name, color, mark, mean_coords, upper_coords, lower_coords, legend, style="solid", add_legend=False):
    legend_line = rf"\addlegendentry{{{legend}}}" if add_legend else ""
    return rf"""
\addplot[name path={name}upper, draw=none, forget plot] coordinates {{{upper_coords}}};
\addplot[name path={name}lower, draw=none, forget plot] coordinates {{{lower_coords}}};
\addplot[{color}, fill opacity=0.10, draw=none, forget plot]
    fill between[of={name}upper and {name}lower];
\addplot[
    color={color},
    opacity=0.78,
    thick,
    {style},
    mark={mark},
    mark options={{fill=white, draw={color}, fill opacity=0.55, draw opacity=0.78}}
] coordinates {{{mean_coords}}};
{legend_line}
"""


def shared_cost_legend():
    return r"""\begin{tikzpicture}
\begin{axis}[
    hide axis,
    xmin=0,
    xmax=1,
    ymin=0,
    ymax=1,
    width=0.62\linewidth,
    height=2.0cm,
    legend columns=5,
    legend style={
        draw=black!40,
        fill=white,
        font=\fontsize{5}{6}\selectfont,
        column sep=0.35em,
        row sep=0pt,
        inner xsep=2pt,
        inner ysep=1pt,
        at={(0.5,0.5)},
        anchor=center,
    },
]
\addlegendimage{black!65, thick, mark=o, mark options={fill=white, draw=black!65}}
\addlegendentry{Heuristic}
\addlegendimage{myorange, thick, mark=o, mark options={fill=white, draw=myorange}}
\addlegendentry{Threshold}
\addlegendimage{myblue, thick, mark=x, mark options={draw=myblue}}
\addlegendentry{Direct}
\addlegendimage{black!55, dashed, thick}
\addlegendentry{Scaled $S_T$}
\addlegendimage{mypurple, thick, mark=diamond*, mark options={fill=white, draw=mypurple}}
\addlegendentry{Xu et al.}
\end{axis}
\end{tikzpicture}"""


def shared_ratio_legend():
    return r"""\begin{tikzpicture}
\begin{axis}[
    hide axis,
    xmin=0,
    xmax=1,
    ymin=0,
    ymax=1,
    width=0.86\linewidth,
    height=2.0cm,
    legend columns=7,
    legend style={
        draw=black!40,
        fill=white,
        font=\fontsize{5}{6}\selectfont,
        column sep=0.35em,
        row sep=0pt,
        inner xsep=2pt,
        inner ysep=1pt,
        at={(0.5,0.5)},
        anchor=center,
    },
]
\addlegendimage{black!45, thick, mark=x, mark options={draw=black!45}}
\addlegendentry{Clean $N_K/T$}
\addlegendimage{black!65, thick, mark=square*, mark options={fill=white, draw=black!65}}
\addlegendentry{Heuristic $N_K/T$}
\addlegendimage{myorange, thick, mark=square*, mark options={fill=white, draw=myorange}}
\addlegendentry{Threshold $N_K/T$}
\addlegendimage{myblue, thick, mark=x, mark options={draw=myblue}}
\addlegendentry{Direct $N_K/T$}
\addlegendimage{myorange, thick, mark=o, mark options={fill=white, draw=myorange}}
\addlegendentry{Threshold $N_K^{\on}/H$}
\addlegendimage{myblue, thick, mark=triangle*, mark options={fill=white, draw=myblue}}
\addlegendentry{Direct $N_K^{\on}/H$}
\addlegendimage{mypurple, thick, mark=square*, mark options={fill=white, draw=mypurple}}
\addlegendentry{Xu et al. $N_K/T$}
\end{axis}
\end{tikzpicture}"""


def combined_legend():
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[
    hide axis,
    xmin=0, xmax=1, ymin=0, ymax=1,
    width=0.96\linewidth,
    height=2.0cm,
    legend columns=4,
    legend style={{
        draw=black!40,
        fill=white,
        font=\fontsize{{5}}{{6}}\selectfont,
        column sep=0.3em,
        row sep=0pt,
        inner xsep=2pt,
        inner ysep=1pt,
        at={{(0.5,0.5)}},
        anchor=center,
    }},
]
\addlegendimage{{black!65, thick, mark=o, mark options={{fill=white, draw=black!65}}}}
\addlegendentry{{Heuristic}}
\addlegendimage{{myorange, thick, mark=o, mark options={{fill=white, draw=myorange}}}}
\addlegendentry{{Threshold}}
\addlegendimage{{myblue, thick, mark=x, mark options={{draw=myblue}}}}
\addlegendentry{{Direct}}
\addlegendimage{{black!55, dashed, thick}}
\addlegendentry{{Scaled $S_T$ bound}}
\addlegendimage{{black!45, thick, mark=x, mark options={{draw=black!45}}}}
\addlegendentry{{Clean $N_K/T$}}
\addlegendimage{{myorange, thick, mark=square*, mark options={{fill=white, draw=myorange}}}}
\addlegendentry{{Threshold $N_K/T$}}
\addlegendimage{{myblue, thick, mark=x, mark options={{draw=myblue}}}}
\addlegendentry{{Direct $N_K/T$}}
\end{{axis}}
\end{{tikzpicture}}"""


def write_figure(rows):
    cost_stats, success_ratio_stats, total_target_share_stats, s_t_by_T = aggregate(rows)
    baseline_cost_stats, baseline_target_share_stats = aggregate_baseline(read_baseline_rows())
    tick_values, tick_labels = xticks(rows)
    max_T_tex = f"{max(int(row['T']) for row in rows):,}".replace(",", "{,}")

    ucb_direct = shaded_plot(
        "ucbdir",
        "myblue",
        "x",
        coord_series(cost_stats, "UCB direct", "mean"),
        coord_series(cost_stats, "UCB direct", "upper"),
        coord_series(cost_stats, "UCB direct", "lower"),
        "Direct",
    )
    ucb_appendix = shaded_plot(
        "ucbapp",
        "myorange",
        "o",
        coord_series(cost_stats, "UCB appendix", "mean"),
        coord_series(cost_stats, "UCB appendix", "upper"),
        coord_series(cost_stats, "UCB appendix", "lower"),
        "Threshold",
    )
    ts_direct = shaded_plot(
        "tsdir",
        "myblue",
        "x",
        coord_series(cost_stats, "TS direct", "mean"),
        coord_series(cost_stats, "TS direct", "upper"),
        coord_series(cost_stats, "TS direct", "lower"),
        "Direct",
    )
    ts_appendix = shaded_plot(
        "tsapp",
        "myorange",
        "o",
        coord_series(cost_stats, "TS appendix", "mean"),
        coord_series(cost_stats, "TS appendix", "upper"),
        coord_series(cost_stats, "TS appendix", "lower"),
        "Threshold",
    )
    ucb_success_appendix = shaded_plot(
        "ucbtapp",
        "myorange",
        "o",
        coord_series(success_ratio_stats, "UCB appendix", "mean"),
        coord_series(success_ratio_stats, "UCB appendix", "upper"),
        coord_series(success_ratio_stats, "UCB appendix", "lower"),
        "Threshold",
    )
    ucb_success_direct = shaded_plot(
        "ucbtdir",
        "myblue",
        "triangle*",
        coord_series(success_ratio_stats, "UCB direct", "mean"),
        coord_series(success_ratio_stats, "UCB direct", "upper"),
        coord_series(success_ratio_stats, "UCB direct", "lower"),
        "Direct",
    )
    ts_success_appendix = shaded_plot(
        "tstapp",
        "myorange",
        "o",
        coord_series(success_ratio_stats, "TS appendix", "mean"),
        coord_series(success_ratio_stats, "TS appendix", "upper"),
        coord_series(success_ratio_stats, "TS appendix", "lower"),
        "Threshold",
    )
    ts_success_direct = shaded_plot(
        "tstdir",
        "myblue",
        "triangle*",
        coord_series(success_ratio_stats, "TS direct", "mean"),
        coord_series(success_ratio_stats, "TS direct", "upper"),
        coord_series(success_ratio_stats, "TS direct", "lower"),
        "Direct",
    )
    ucb_share_clean = shaded_plot(
        "ucbshareclean",
        "black!45",
        "x",
        coord_series(total_target_share_stats, "UCB clean", "mean"),
        coord_series(total_target_share_stats, "UCB clean", "upper"),
        coord_series(total_target_share_stats, "UCB clean", "lower"),
        "Clean target share",
    )
    ucb_share_appendix = shaded_plot(
        "ucbshareapp",
        "myorange",
        "square*",
        coord_series(total_target_share_stats, "UCB appendix", "mean"),
        coord_series(total_target_share_stats, "UCB appendix", "upper"),
        coord_series(total_target_share_stats, "UCB appendix", "lower"),
        "Threshold target share",
    )
    ucb_share_direct = shaded_plot(
        "ucbsharedir",
        "myblue",
        "x",
        coord_series(total_target_share_stats, "UCB direct", "mean"),
        coord_series(total_target_share_stats, "UCB direct", "upper"),
        coord_series(total_target_share_stats, "UCB direct", "lower"),
        "Direct target share",
    )
    ts_share_clean = shaded_plot(
        "tsshareclean",
        "black!45",
        "x",
        coord_series(total_target_share_stats, "TS clean", "mean"),
        coord_series(total_target_share_stats, "TS clean", "upper"),
        coord_series(total_target_share_stats, "TS clean", "lower"),
        "Clean target share",
    )
    ts_share_appendix = shaded_plot(
        "tsshareapp",
        "myorange",
        "square*",
        coord_series(total_target_share_stats, "TS appendix", "mean"),
        coord_series(total_target_share_stats, "TS appendix", "upper"),
        coord_series(total_target_share_stats, "TS appendix", "lower"),
        "Threshold target share",
    )
    ts_share_direct = shaded_plot(
        "tssharedir",
        "myblue",
        "x",
        coord_series(total_target_share_stats, "TS direct", "mean"),
        coord_series(total_target_share_stats, "TS direct", "upper"),
        coord_series(total_target_share_stats, "TS direct", "lower"),
        "Direct target share",
    )
    ucb_cost_jun = shaded_plot(
        "ucbjun",
        "black!65",
        "o",
        coord_series(baseline_cost_stats, "Jun2018 clipped UCB", "mean"),
        coord_series(baseline_cost_stats, "Jun2018 clipped UCB", "upper"),
        coord_series(baseline_cost_stats, "Jun2018 clipped UCB", "lower"),
        "Jun2018 clipped UCB",
    )
    ts_cost_zuo = shaded_plot(
        "tszuo",
        "black!65",
        "o",
        coord_series(baseline_cost_stats, "Zuo2024 clipped TS", "mean"),
        coord_series(baseline_cost_stats, "Zuo2024 clipped TS", "upper"),
        coord_series(baseline_cost_stats, "Zuo2024 clipped TS", "lower"),
        "Zuo2024 clipped TS",
    )
    ucb_cost_xu = shaded_plot(
        "ucbxu", "mypurple", "diamond*",
        coord_series(baseline_cost_stats, "Xu2021 observation-free UCB", "mean"),
        coord_series(baseline_cost_stats, "Xu2021 observation-free UCB", "upper"),
        coord_series(baseline_cost_stats, "Xu2021 observation-free UCB", "lower"),
        r"\citet{xu2021observation}",
    )
    ts_cost_xu = shaded_plot(
        "tsxu", "mypurple", "diamond*",
        coord_series(baseline_cost_stats, "Xu2021 observation-free TS", "mean"),
        coord_series(baseline_cost_stats, "Xu2021 observation-free TS", "upper"),
        coord_series(baseline_cost_stats, "Xu2021 observation-free TS", "lower"),
        r"\citet{xu2021observation}",
    )
    ucb_share_xu = shaded_plot(
        "ucbrxu", "mypurple", "square*",
        coord_series(baseline_target_share_stats, "Xu2021 observation-free UCB", "mean"),
        coord_series(baseline_target_share_stats, "Xu2021 observation-free UCB", "upper"),
        coord_series(baseline_target_share_stats, "Xu2021 observation-free UCB", "lower"),
        r"\citet{xu2021observation}",
    )
    ts_share_xu = shaded_plot(
        "tsrxu", "mypurple", "square*",
        coord_series(baseline_target_share_stats, "Xu2021 observation-free TS", "mean"),
        coord_series(baseline_target_share_stats, "Xu2021 observation-free TS", "upper"),
        coord_series(baseline_target_share_stats, "Xu2021 observation-free TS", "lower"),
        r"\citet{xu2021observation}",
    )
    ucb_share_jun = shaded_plot(
        "ucbrjun",
        "black!65",
        "square*",
        coord_series(baseline_target_share_stats, "Jun2018 clipped UCB", "mean"),
        coord_series(baseline_target_share_stats, "Jun2018 clipped UCB", "upper"),
        coord_series(baseline_target_share_stats, "Jun2018 clipped UCB", "lower"),
        "Jun2018 clipped UCB",
    )
    ts_share_zuo = shaded_plot(
        "tsrzuo",
        "black!65",
        "square*",
        coord_series(baseline_target_share_stats, "Zuo2024 clipped TS", "mean"),
        coord_series(baseline_target_share_stats, "Zuo2024 clipped TS", "upper"),
        coord_series(baseline_target_share_stats, "Zuo2024 clipped TS", "lower"),
        "Zuo2024 clipped TS",
    )
    ucb_theory = theory_bound(cost_stats, s_t_by_T, ["UCB appendix", "UCB direct"])
    ts_theory = theory_bound(cost_stats, s_t_by_T, ["TS appendix", "TS direct"])

    content = rf"""% Auto-generated by generate_fixed_T_tikz.py.
% Source file: {RESULTS_FILE}
\begin{{figure*}}[t]
\centering
\subfloat[UCB attack cost versus $T$]{{%
\begin{{minipage}}[t]{{0.49\linewidth}}
\vspace{{0pt}}
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=\linewidth,
    height=3.8cm,
    xmode=log,
    log basis x=10,
    xlabel={{}},
    scaled x ticks=false,
    ylabel={{{{\small Attack cost $\mathcal{{C}}(\mathbf{{n}})$}}}},
    ymin=0,
    tick align=outside,
    tick pos=left,
    xtick={{{tick_values}}},
    xticklabels={{{tick_labels}}},
    xticklabel style={{font=\scriptsize, rotate=30, anchor=east}},
    yticklabel style={{font=\scriptsize}},
    axis line style={{black}},
        legend pos=north west,
        reverse legend,
    legend style={{
        fill opacity=0.5,
        draw opacity=1,
        text opacity=1,
        font=\fontsize{{5}}{{6}}\selectfont,
        row sep=0pt,
        inner xsep=2pt,
        inner ysep=1pt
    }},
]
{ucb_cost_jun}
{ucb_cost_xu}
{ucb_appendix}
{ucb_direct}
\addplot[
    color=black!55,
    dashed,
    thick,
    mark=none
] coordinates {{{ucb_theory}}};
\end{{axis}}
\end{{tikzpicture}}
\end{{minipage}}}}
\hfill
\subfloat[Thompson Sampling attack cost versus $T$]{{%
\begin{{minipage}}[t]{{0.49\linewidth}}
\vspace{{0pt}}
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=\linewidth,
    height=3.8cm,
    xmode=log,
    log basis x=10,
    xlabel={{}},
    scaled x ticks=false,
    ylabel={{{{\small Attack cost $\mathcal{{C}}(\mathbf{{n}})$}}}},
    ymin=0,
    tick align=outside,
    tick pos=left,
    xtick={{{tick_values}}},
    xticklabels={{{tick_labels}}},
    xticklabel style={{font=\scriptsize, rotate=30, anchor=east}},
    yticklabel style={{font=\scriptsize}},
    axis line style={{black}},
        legend pos=north west,
        reverse legend,
    legend style={{
        fill opacity=0.5,
        draw opacity=1,
        text opacity=1,
        font=\fontsize{{5}}{{6}}\selectfont,
        row sep=0pt,
        inner xsep=2pt,
        inner ysep=1pt
    }},
]
{ts_cost_zuo}
{ts_cost_xu}
{ts_appendix}
{ts_direct}
\addplot[
    color=black!55,
    dashed,
    thick,
    mark=none
] coordinates {{{ts_theory}}};
\end{{axis}}
\end{{tikzpicture}}
\end{{minipage}}}}
\par\vspace{{0.15em}}
{shared_cost_legend()}
\par\vspace{{0.45em}}
\subfloat[UCB target-arm ratio versus $T$]{{%
\begin{{minipage}}[t]{{0.49\linewidth}}
\vspace{{0pt}}
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=\linewidth,
    height=3.8cm,
    xmode=log,
    log basis x=10,
    xlabel={{}},
    scaled x ticks=false,
    ylabel={{{{\small Target-arm ratio}}}},
    ymin=0,
    ymax=1.05,
    tick align=outside,
    tick pos=left,
    xtick={{{tick_values}}},
    xticklabels={{{tick_labels}}},
    xticklabel style={{font=\scriptsize, rotate=30, anchor=east}},
    yticklabel style={{font=\scriptsize}},
    axis line style={{black}},
        legend pos=north west,
        reverse legend,
    legend style={{
        fill opacity=0.5,
        draw opacity=1,
        text opacity=1,
        font=\fontsize{{5}}{{6}}\selectfont,
        row sep=0pt,
        inner xsep=2pt,
        inner ysep=1pt
    }},
]
{ucb_share_clean}
{ucb_share_jun}
{ucb_share_xu}
{ucb_share_appendix}
{ucb_share_direct}
{ucb_success_appendix}
{ucb_success_direct}
\end{{axis}}
\end{{tikzpicture}}
\end{{minipage}}}}
\hfill
\subfloat[Thompson Sampling target-arm ratio versus $T$]{{%
\begin{{minipage}}[t]{{0.49\linewidth}}
\vspace{{0pt}}
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=\linewidth,
    height=3.8cm,
    xmode=log,
    log basis x=10,
    xlabel={{}},
    scaled x ticks=false,
    ylabel={{{{\small Target-arm ratio}}}},
    ymin=0,
    ymax=1.05,
    tick align=outside,
    tick pos=left,
    xtick={{{tick_values}}},
    xticklabels={{{tick_labels}}},
    xticklabel style={{font=\scriptsize, rotate=30, anchor=east}},
    yticklabel style={{font=\scriptsize}},
    axis line style={{black}},
        legend pos=north west,
        reverse legend,
    legend style={{
        fill opacity=0.5,
        draw opacity=1,
        text opacity=1,
        font=\fontsize{{5}}{{6}}\selectfont,
        row sep=0pt,
        inner xsep=2pt,
        inner ysep=1pt
    }},
]
{ts_share_clean}
{ts_share_zuo}
{ts_share_xu}
{ts_share_appendix}
{ts_share_direct}
{ts_success_appendix}
{ts_success_direct}
\end{{axis}}
\end{{tikzpicture}}
\end{{minipage}}}}
\par\vspace{{0.15em}}
{shared_ratio_legend()}
\caption{{{mlrunner.DATASET_LABEL} fixed-horizon experiments up to $T={max_T_tex}$ over ten repeats. Top panels show native attack costs for our fixed-$T$ offline constructions and bounded heuristics on the same horizon grid; solid curves show means and shaded bands show one sample standard deviation. The dashed curve is a scaled $S_T=T^{{2/3}}(K\log T)^{{1/3}}$ bound, with the scale chosen to upper-bound our displayed mean-plus-standard-deviation envelope in each cost panel. \citet{{xu2021observation}} observation-free costs are omitted from the plots because they are near the full horizon even at the only feasible displayed point and are reported in the text instead. Bottom panels separate two quantities: square markers show the total target-arm fraction $N_K/T$, where offline fake target samples are counted in $N_K$, while circle markers show our post-attack success ratio $N_K^{{\on}}/H$. The cost panels and ratio panels each use a shared centered legend.}}
\label{{fig:fixed_T_sweep}}
\end{{figure*}}
"""
    def panel(start, end):
        i = content.index(start)
        j = content.index(end, i)
        return content[i:j].rstrip()

    ucb_cost_panel = panel(r"\subfloat[UCB attack cost versus $T$]", "\\hfill\n\\subfloat[Thompson Sampling attack cost versus $T$]")
    ts_cost_panel = panel(r"\subfloat[Thompson Sampling attack cost versus $T$]", r"\par\vspace{0.15em}")
    ucb_ratio_panel = panel(r"\subfloat[UCB target-arm ratio versus $T$]", "\\hfill\n\\subfloat[Thompson Sampling target-arm ratio versus $T$]")
    ts_ratio_panel = panel(r"\subfloat[Thompson Sampling target-arm ratio versus $T$]", r"\par\vspace{0.15em}",)

    ucb_content = rf"""% Auto-generated by generate_fixed_T_tikz.py.
% Source file: {RESULTS_FILE}
\begin{{figure*}}[t]
\centering
{ucb_cost_panel}
\hfill
{ucb_ratio_panel}
\par\vspace{{0.15em}}
{combined_legend()}
\caption{{{mlrunner.DATASET_LABEL} fixed-horizon UCB experiments up to $T={max_T_tex}$ over ten repeats. Panel (a) reports native attack costs for our two offline constructions and the bounded heuristic of \citet{{jun2018adversarial}}; the dashed curve is a scaled $S_T=T^{{2/3}}(K\log T)^{{1/3}}$ bound chosen to upper-bound the displayed mean-plus-standard-deviation envelope of our costs. Panel (b) reports total target-arm exposure $N_K/T$ and, for our attacks, post-deployment success $N_K^{{\on}}/H$. \citet{{xu2021observation}} is omitted because its observation-free cost dominates the displayed scale.}}
\label{{fig:fixed_T_sweep}}
\end{{figure*}}
"""
    ts_content = rf"""% Auto-generated by generate_fixed_T_tikz.py.
% Source file: {RESULTS_FILE}
\begin{{figure*}}[t]
\centering
{ts_cost_panel}
\hfill
{ts_ratio_panel}
\par\vspace{{0.15em}}
{combined_legend()}
\caption{{{mlrunner.DATASET_LABEL} fixed-horizon Thompson Sampling experiments up to $T={max_T_tex}$ over ten repeats. Panel (a) reports native attack costs for our two offline constructions and the bounded heuristic of \citet{{zuo2024near}}; the dashed curve is a scaled $S_T=T^{{2/3}}(K\log T)^{{1/3}}$ bound chosen to upper-bound the displayed mean-plus-standard-deviation envelope of our costs. Panel (b) reports total target-arm exposure $N_K/T$ and, for our attacks, post-deployment success $N_K^{{\on}}/H$. \citet{{xu2021observation}} is omitted because its observation-free cost dominates the displayed scale.}}
\label{{fig:fixed_T_sweep_ts}}
\end{{figure*}}
"""
    FIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    FIG_FILE.write_text(ucb_content, encoding="utf-8")
    TS_FIG_FILE.write_text(ts_content, encoding="utf-8")
    FRAGMENT_DIR.mkdir(parents=True, exist_ok=True)
    (FRAGMENT_DIR / "fixed_ucb_cost.tex").write_text(ucb_cost_panel, encoding="utf-8")
    (FRAGMENT_DIR / "fixed_ucb_ratio.tex").write_text(ucb_ratio_panel, encoding="utf-8")
    (FRAGMENT_DIR / "fixed_ts_cost.tex").write_text(ts_cost_panel, encoding="utf-8")
    (FRAGMENT_DIR / "fixed_ts_ratio.tex").write_text(ts_ratio_panel, encoding="utf-8")
    print(f"wrote {FIG_FILE}")
    print(f"wrote {TS_FIG_FILE}")


def Main():
    write_figure(read_rows())


if __name__ == "__main__":
    Main()
