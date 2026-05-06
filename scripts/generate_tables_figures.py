from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "marginal_bt",
    "source_contextual_bt",
    "fixed_target_contextual_bt",
    "fixed_target_ci_certified_bt",
    "ss_cbt",
]

METHOD_LABELS = {
    "marginal_bt": "Marginal BT",
    "source_contextual_bt": "Source C-BT",
    "fixed_target_contextual_bt": "Fixed-Target C-BT",
    "fixed_target_ci_certified_bt": "Fixed-Target C-BT + CI Cert.",
    "ss_cbt": "DR-CBT",
}

METHOD_COLORS = {
    "marginal_bt": "#8c8c8c",
    "source_contextual_bt": "#5f85b3",
    "fixed_target_contextual_bt": "#e0a458",
    "fixed_target_ci_certified_bt": "#b56576",
    "ss_cbt": "#2a9d5b",
}

SCENARIO_LABELS = {
    "exact_bt_mild_shift": "Exact + Mild Shift",
    "misspecified_shift": "Misspecified Shift",
    "low_overlap_sparse": "Low Overlap + Sparse",
}

BENCHMARK_LABELS = {
    "mt_bench": "MT-Bench",
    "arena_language": "Arena-Language",
    "atp_surface": "ATP-Surface",
    "wta_surface": "WTA-Surface",
}


def _plot_metric(df: pd.DataFrame, metric: str, ylabel: str, output_prefix: Path) -> None:
    pivot = df.pivot(index="scenario", columns="method", values=metric).reindex(columns=METHOD_ORDER)
    ax = pivot.plot(kind="bar", figsize=(9, 4))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_prefix.with_suffix(".png"), dpi=200)
    plt.savefig(output_prefix.with_suffix(".pdf"))
    plt.close()


def _plot_main_synthetic_figure(pair_summary: pd.DataFrame, output_prefix: Path) -> None:
    ordered = pair_summary.copy()
    ordered["scenario"] = pd.Categorical(
        ordered["scenario"],
        categories=["exact_bt_mild_shift", "misspecified_shift", "low_overlap_sparse"],
        ordered=True,
    )
    ordered = ordered.sort_values(["scenario", "method"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    metrics = [
        ("coverage", "Pairwise Interval-Inclusion", (0.0, 0.95)),
        ("false_stable_rate", "False Stable-Order Rate", (0.0, 0.22)),
    ]
    x = np.arange(3)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHOD_ORDER))

    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        for offset, method in zip(offsets, METHOD_ORDER):
            sub = ordered[ordered["method"] == method]
            vals = [float(sub[sub["scenario"] == scenario][metric].iloc[0]) for scenario in ordered["scenario"].cat.categories]
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.6,
                label=METHOD_LABELS[method],
            )
            if method == "ss_cbt":
                for bar in bars:
                    bar.set_hatch("//")
            for xpos, val in zip(x + offset, vals):
                ax.text(xpos, val + 0.01 * (ylim[1] - ylim[0]), f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in ordered["scenario"].cat.categories], rotation=12, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_uncertainty_panel(
    ax: plt.Axes,
    focus: pd.DataFrame,
    mixture_order: list[str],
    mixture_labels: dict[str, str],
    xlabel: str,
    title: str,
) -> None:
    focus = focus.copy()
    focus["mixture"] = pd.Categorical(focus["mixture"], categories=mixture_order, ordered=True)
    focus = focus.sort_values("mixture")
    y = np.arange(len(focus))[::-1]

    for ypos, (_, row) in zip(y, focus.iterrows()):
        ax.plot(
            [row["robust_lower"], row["robust_upper"]],
            [ypos, ypos],
            color="#2a9d5b",
            linewidth=6,
            solid_capstyle="round",
        )
        ax.plot(row["fixed_contrast"], ypos, marker="o", markersize=7, color="#e76f51")

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([mixture_labels[m] for m in focus["mixture"]])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_real_uncertainty_focus(mt_pairwise: pd.DataFrame, arena_pairwise: pd.DataFrame, output_prefix: Path) -> None:
    mt_focus = mt_pairwise[
        (mt_pairwise["model_i"] == "claude-v1") & (mt_pairwise["model_j"] == "gpt-3.5-turbo")
    ].copy()
    arena_focus = arena_pairwise[
        (arena_pairwise["model_i"] == "gemini-2.5-flash")
        & (arena_pairwise["model_j"] == "chatgpt-4o-latest-20250326")
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.5), sharey=False)
    _plot_uncertainty_panel(
        axes[0],
        mt_focus,
        ["source_mix", "balanced", "coding_heavy", "writing_heavy"],
        {
            "source_mix": "Source Mix",
            "balanced": "Balanced",
            "coding_heavy": "Coding-Heavy",
            "writing_heavy": "Writing-Heavy",
        },
        "Contrast: Claude-v1 minus GPT-3.5-Turbo",
        "MT-Bench: A Local Prompt-Mix Flip",
    )
    _plot_uncertainty_panel(
        axes[1],
        arena_focus,
        ["source_mix", "balanced", "english_heavy", "non_english_heavy"],
        {
            "source_mix": "Source Mix",
            "balanced": "Balanced",
            "english_heavy": "English-Heavy",
            "non_english_heavy": "Non-English-Heavy",
        },
        "Contrast: Gemini-Flash minus GPT-4o",
        "Arena: A Language-Mix Flip",
    )
    fig.text(
        0.5,
        0.02,
        "Green interval: robust bound   Orange point: fixed-target contrast",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    plt.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_real_method_comparison(real_summary: pd.DataFrame, output_prefix: Path) -> None:
    ordered = real_summary.copy()
    ordered["topk_f1"] = np.where(
        ordered["topk_precision"] + ordered["topk_recall"] > 0,
        2 * ordered["topk_precision"] * ordered["topk_recall"] / (ordered["topk_precision"] + ordered["topk_recall"]),
        0.0,
    )
    ordered["benchmark"] = pd.Categorical(
        ordered["benchmark"],
        categories=["mt_bench", "arena_language", "atp_surface", "wta_surface"],
        ordered=True,
    )
    ordered["method"] = pd.Categorical(
        ordered["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    ordered = ordered.sort_values(["benchmark", "method"])

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.55), sharex=True)
    metrics = [
        # Start bar charts at zero so low-performing baselines are not visually clipped
        ("stable_pair_precision", "Stable-Pair Precision", (0.0, 1.02)),
        ("topk_f1", "Stable Top-3 F1", (0.0, 1.02)),
    ]
    x = np.arange(len(ordered["benchmark"].cat.categories))
    width = 0.14
    offsets = np.linspace(-2.0 * width, 2.0 * width, len(METHOD_ORDER))

    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        for offset, method in zip(offsets, METHOD_ORDER):
            sub = ordered[ordered["method"] == method]
            vals = [
                float(sub[sub["benchmark"] == benchmark][metric].iloc[0])
                for benchmark in ordered["benchmark"].cat.categories
            ]
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.6,
                label=METHOD_LABELS[method],
            )
            if method == "ss_cbt":
                for bar in bars:
                    bar.set_hatch("//")
            for xpos, val in zip(x + offset, vals):
                ax.text(
                    xpos,
                    val + 0.01 * (ylim[1] - ylim[0]),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels([BENCHMARK_LABELS[b] for b in ordered["benchmark"].cat.categories])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    plt.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _draw_box(ax, xy, width, height, label, facecolor):
    rect = Rectangle(xy, width, height, facecolor=facecolor, edgecolor="black", linewidth=1.0)
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, label, ha="center", va="center", fontsize=10)


def _draw_arrow(ax, start, end):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="black")
    ax.add_patch(arrow)


def _plot_method_overview(output_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 4.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def add_panel(xy, width, height, face, edge, title):
        panel = FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.024",
            facecolor=face,
            edgecolor="#d9dee7",
            linewidth=0.9,
            zorder=0,
        )
        ax.add_patch(panel)
        chip = FancyBboxPatch(
            (xy[0] + 0.024, xy[1] + height - 0.085),
            width - 0.048,
            0.072,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor="white",
            edgecolor=edge,
            linewidth=1.25,
            zorder=1,
        )
        ax.add_patch(chip)
        ax.text(
            xy[0] + 0.045,
            xy[1] + height - 0.049,
            title,
            ha="left",
            va="center",
            fontsize=11.2,
            fontweight="bold",
            color="#243447",
            zorder=2,
        )

    def add_box(xy, width, height, label, edge, fontsize=10.0, lw=1.3):
        patch = FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.014,rounding_size=0.018",
            facecolor="white",
            edgecolor=edge,
            linewidth=lw,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(
            xy[0] + width / 2,
            xy[1] + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#1b1f24",
            zorder=3,
        )

    def add_arrow(start, end, color="#324a5f", rad=0.0):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.7,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
                zorder=1,
            )
        )

    def add_shift_set_inset(xy, width, height):
        # Outer card
        patch = FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.014,rounding_size=0.018",
            facecolor="white",
            edgecolor="#304b61",
            linewidth=1.3,
            zorder=2,
        )
        ax.add_patch(patch)

        ax.text(
            xy[0] + width / 2,
            xy[1] + height - 0.035,
            r"Nominal law $\widehat q^0$ and shift set $\mathcal{U}(\widehat q^0)$",
            ha="center",
            va="center",
            fontsize=9.4,
            color="#1b1f24",
            zorder=3,
        )

        # Local simplex geometry inside the box.
        tri = np.array(
            [
                [xy[0] + 0.028, xy[1] + 0.045],
                [xy[0] + width - 0.028, xy[1] + 0.045],
                [xy[0] + width / 2, xy[1] + height - 0.07],
            ]
        )
        ax.add_patch(
            Polygon(
                tri,
                closed=True,
                facecolor="none",
                edgecolor="#708090",
                linewidth=1.0,
                zorder=2,
            )
        )

        poly = np.array(
            [
                [xy[0] + 0.072, xy[1] + 0.082],
                [xy[0] + 0.135, xy[1] + 0.082],
                [xy[0] + 0.165, xy[1] + 0.122],
                [xy[0] + 0.135, xy[1] + 0.162],
                [xy[0] + 0.072, xy[1] + 0.162],
                [xy[0] + 0.042, xy[1] + 0.122],
            ]
        )
        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor="#f1bf75",
                edgecolor="#c57b17",
                linewidth=1.2,
                alpha=0.58,
                zorder=3,
                joinstyle="round",
            )
        )

        sample_pts = np.array(
            [
                [xy[0] + 0.086, xy[1] + 0.102],
                [xy[0] + 0.112, xy[1] + 0.116],
                [xy[0] + 0.126, xy[1] + 0.142],
                [xy[0] + 0.098, xy[1] + 0.142],
            ]
        )
        ax.scatter(sample_pts[:, 0], sample_pts[:, 1], s=16, color="#c57b17", alpha=0.24, zorder=4)

        q0 = np.array([xy[0] + 0.102, xy[1] + 0.119])
        ax.scatter([q0[0]], [q0[1]], s=34, color="#1f3552", zorder=5)
        ax.text(q0[0] + 0.010, q0[1] - 0.008, r"$q^0$", fontsize=9.4, color="#1f3552", zorder=5)
        ax.text(xy[0] + 0.126, xy[1] + 0.136, r"$\mathcal{U}(q^0)$", fontsize=9.4, color="#8a5610", zorder=5)

        ax.text(tri[0, 0] - 0.01, tri[0, 1] - 0.02, r"$g_1$", fontsize=8.4, color="#4a5c6a", zorder=4)
        ax.text(tri[1, 0] + 0.004, tri[1, 1] - 0.02, r"$g_2$", fontsize=8.4, color="#4a5c6a", zorder=4)
        ax.text(tri[2, 0] - 0.008, tri[2, 1] + 0.01, r"$g_3$", fontsize=8.4, color="#4a5c6a", zorder=4)

    panels = [
        ((0.04, 0.14), 0.29, 0.66, "#eef4ff", "#4a78c2", "1  Contextual Data"),
        ((0.355, 0.14), 0.25, 0.66, "#fff6e8", "#d08a1f", "2  Deployment Shift Model"),
        ((0.63, 0.14), 0.33, 0.66, "#eef8f1", "#2e9b57", "3  Robust Outputs"),
    ]
    for panel in panels:
        add_panel(*panel)

    add_box(
        (0.075, 0.49),
        0.22,
        0.16,
        "Observed pairwise records\n$item_i$, $item_j$, outcome, context",
        "#304b61",
        fontsize=10.3,
    )
    add_box(
        (0.075, 0.26),
        0.22,
        0.17,
        "Reweighted pooled\ncontextual BT fit\n$\\widehat\\theta_{ig}=\\widehat\\mu_i+\\widehat\\delta_{ig}$",
        "#304b61",
        fontsize=10.3,
    )
    add_shift_set_inset((0.392, 0.41), 0.18, 0.22)

    add_box(
        (0.665, 0.49),
        0.27,
        0.12,
        "Pairwise LP bounds\n$\\widehat\\psi^-_{ij},\\widehat\\psi^+_{ij}$",
        "#2f7a52",
        fontsize=10.0,
    )
    add_box(
        (0.665, 0.31),
        0.27,
        0.11,
        "Certified stable top-$k$\nfrom pairwise certificates",
        "#2f7a52",
        fontsize=10.0,
    )
    add_box(
        (0.665, 0.16),
        0.27,
        0.10,
        "Rank-flip sensitivity\nand abstention diagnostics",
        "#2f7a52",
        fontsize=10.0,
    )

    add_arrow((0.185, 0.49), (0.185, 0.43))
    add_arrow((0.295, 0.345), (0.392, 0.52))
    add_arrow((0.572, 0.52), (0.665, 0.55))
    add_arrow((0.80, 0.49), (0.80, 0.42))
    add_arrow((0.80, 0.31), (0.80, 0.26))

    add_box(
        (0.385, 0.24),
        0.195,
        0.062,
        "Theory: identification + bridge",
        "#d9b54a",
        fontsize=8.7,
        lw=1.0,
    )
    add_box(
        (0.385, 0.155),
        0.195,
        0.062,
        "Computation: exact LP post-aggregation",
        "#6d8fd6",
        fontsize=8.7,
        lw=1.0,
    )

    ax.text(
        0.80,
        0.67,
        "Reporting principle: certify stable conclusions,\nabstain on deployment-sensitive comparisons",
        ha="center",
        va="center",
        fontsize=9.0,
        color="#2e7b55",
        zorder=3,
    )

    plt.tight_layout()
    plt.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summary tables and figures")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--real-dir", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_summary = pd.read_csv(input_dir / "summary_pair_metrics.csv")
    topk_summary = pd.read_csv(input_dir / "summary_topk_metrics.csv")

    comparison = pair_summary.merge(topk_summary, on=["scenario", "method"], how="inner")
    comparison.to_csv(output_dir / "comparison_table.csv", index=False)

    ablation = comparison[
        [
            "scenario",
            "method",
            "coverage",
            "false_stable_rate",
            "declaration_rate",
            "topk_recall",
            "avg_set_size",
        ]
    ].copy()
    ablation.to_csv(output_dir / "ablation_table.csv", index=False)

    _plot_metric(pair_summary, "coverage", "Pairwise interval-inclusion", output_dir / "coverage_by_scenario")
    _plot_metric(pair_summary, "false_stable_rate", "False stable-order rate", output_dir / "false_order_rate")
    _plot_metric(topk_summary, "topk_recall", "Top-k recall", output_dir / "topk_recall")
    _plot_main_synthetic_figure(pair_summary, output_dir / "main_synthetic_figure")
    _plot_method_overview(output_dir / "method_overview")

    if args.real_dir:
        real_dir = Path(args.real_dir)
        mt_pairwise_path = real_dir / "mt_bench" / "real_pairwise_bounds.csv"
        if not mt_pairwise_path.exists():
            mt_pairwise_path = real_dir / "real_pairwise_bounds.csv"
        arena_pairwise_path = real_dir / "arena_language" / "real_pairwise_bounds.csv"

        mt_pairwise = pd.read_csv(mt_pairwise_path)
        arena_pairwise = pd.read_csv(arena_pairwise_path)
        _plot_real_uncertainty_focus(mt_pairwise, arena_pairwise, output_dir / "real_uncertainty_focus")
        real_summary = pd.read_csv(real_dir / "real_method_comparison_summary.csv")
        _plot_real_method_comparison(real_summary, output_dir / "real_method_comparison")


if __name__ == "__main__":
    main()
