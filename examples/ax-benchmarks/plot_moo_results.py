#!/usr/bin/env python3
"""Generate plots from MOO benchmark results.

This script generates:
1. Hypervolume convergence plots: Shows how hypervolume improves over iterations
2. Pareto front scatter plots: Shows the final Pareto fronts from each method

Usage:
    # Generate plots from default results directory
    python plot_moo_results.py

    # Specify custom results directory
    python plot_moo_results.py --results-dir my_results/moo

    # Generate only hypervolume plots
    python plot_moo_results.py --plot-type hypervolume

    # Generate only Pareto front plots
    python plot_moo_results.py --plot-type pareto

Output:
    - {problem_name}_hypervolume.png: Hypervolume convergence plot
    - {problem_name}_pareto.png: Pareto front scatter plot
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Error: matplotlib is required for plotting.")
    print("Install with: pip install matplotlib")
    sys.exit(1)


# Color scheme for methods
METHOD_COLORS = {
    "ax": "#2563EB",      # Blue
    "sobol": "#DC2626",   # Red
}

METHOD_LABELS = {
    "ax": "Ax MOO",
    "sobol": "Sobol (baseline)",
}


def load_results(results_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all MOO benchmark results from directory.

    Args:
        results_dir: Directory containing results JSON files

    Returns:
        Dict mapping problem name to results dict
    """
    all_results = {}

    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    for filename in os.listdir(results_dir):
        if filename.startswith("results_") and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)

            problem_name = data["metadata"]["problem_name"]
            all_results[problem_name] = data

    if not all_results:
        print(f"Warning: No results files found in '{results_dir}'")

    return all_results


def plot_hypervolume_convergence(
    results: Dict[str, Any],
    output_path: str,
    title: Optional[str] = None,
):
    """Generate hypervolume convergence plot.

    Args:
        results: Results dict for a single problem
        output_path: Path to save the plot
        title: Optional custom title
    """
    metadata = results["metadata"]
    stats = results["stats"]
    problem_name = metadata["problem_name"]

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(stats.keys())
    num_trials = metadata["num_trials"]

    for method in methods:
        method_stats = stats[method]
        mean_curve = np.array(method_stats["mean_hv_curve"])
        sem_curve = np.array(method_stats["sem_hv_curve"])

        if len(mean_curve) == 0:
            continue

        iterations = np.arange(1, len(mean_curve) + 1)

        color = METHOD_COLORS.get(method, "#666666")
        label = METHOD_LABELS.get(method, method)

        # Plot mean line
        ax.plot(
            iterations,
            mean_curve,
            color=color,
            linewidth=2,
            label=label,
            linestyle="-" if method == "ax" else "--",
        )

        # Plot confidence band (+/- 1 SEM)
        ax.fill_between(
            iterations,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=color,
            alpha=0.2,
        )

    # Mark initial phase for Ax
    num_init = metadata.get("num_init", 0)
    if num_init > 0:
        ax.axvline(
            x=num_init,
            color="#666666",
            linestyle=":",
            linewidth=1,
            alpha=0.7,
        )
        ax.text(
            num_init + 0.5,
            ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            "Init phase",
            fontsize=9,
            color="#666666",
        )

    # Formatting
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Hypervolume", fontsize=12)

    if title is None:
        title = f"{problem_name}: Hypervolume Convergence"
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add metadata text
    num_replications = metadata.get("num_replications", "?")
    ax.text(
        0.02,
        0.98,
        f"n = {num_replications} replications\nShaded: +/- 1 SEM",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")


def plot_pareto_front(
    results: Dict[str, Any],
    output_path: str,
    title: Optional[str] = None,
    replication_idx: int = 0,
):
    """Generate Pareto front scatter plot.

    Args:
        results: Results dict for a single problem
        output_path: Path to save the plot
        title: Optional custom title
        replication_idx: Which replication to plot (-1 for best, 0 for first)
    """
    metadata = results["metadata"]
    raw_results = results["results"]
    problem_name = metadata["problem_name"]

    fig, ax = plt.subplots(figsize=(10, 8))

    methods = list(raw_results.keys())
    objective_names = metadata.get("objective_names", ["Objective 1", "Objective 2"])
    ref_point = metadata.get("ref_point", [None, None])

    for method in methods:
        replications = raw_results[method]

        if not replications:
            continue

        # Select replication to plot
        if replication_idx == -1:
            # Find best replication (highest final hypervolume)
            best_idx = 0
            best_hv = 0
            for i, rep in enumerate(replications):
                if rep["hypervolume_curve"]:
                    final_hv = rep["hypervolume_curve"][-1]
                    if final_hv > best_hv:
                        best_hv = final_hv
                        best_idx = i
            rep = replications[best_idx]
        else:
            rep_idx = min(replication_idx, len(replications) - 1)
            rep = replications[rep_idx]

        all_Y = np.array(rep.get("all_Y", []))
        all_feasible = np.array(rep.get("all_feasible", []))
        pareto_Y = np.array(rep.get("final_pareto_Y", []))

        if len(all_Y) == 0:
            continue

        color = METHOD_COLORS.get(method, "#666666")
        label = METHOD_LABELS.get(method, method)

        # Plot all evaluated points
        if len(all_Y) > 0:
            # Infeasible points
            if len(all_feasible) > 0:
                infeasible_mask = ~all_feasible
                if np.any(infeasible_mask):
                    ax.scatter(
                        all_Y[infeasible_mask, 0],
                        all_Y[infeasible_mask, 1],
                        c=color,
                        alpha=0.15,
                        s=20,
                        marker="x",
                    )

                # Feasible points (non-Pareto)
                feasible_Y = all_Y[all_feasible]
            else:
                feasible_Y = all_Y

            # Plot feasible non-Pareto points
            if len(feasible_Y) > 0 and len(pareto_Y) > 0:
                # Mark Pareto points differently
                pareto_set = set(tuple(p) for p in pareto_Y)
                non_pareto_mask = [
                    tuple(y) not in pareto_set for y in feasible_Y
                ]
                non_pareto_Y = feasible_Y[non_pareto_mask]
                if len(non_pareto_Y) > 0:
                    ax.scatter(
                        non_pareto_Y[:, 0],
                        non_pareto_Y[:, 1],
                        c=color,
                        alpha=0.3,
                        s=30,
                        marker="o",
                        edgecolors="none",
                    )

        # Plot Pareto front
        if len(pareto_Y) > 0:
            # Sort by first objective for line plot
            sorted_indices = np.argsort(pareto_Y[:, 0])
            sorted_pareto = pareto_Y[sorted_indices]

            ax.scatter(
                sorted_pareto[:, 0],
                sorted_pareto[:, 1],
                c=color,
                s=100,
                marker="*",
                edgecolors="black",
                linewidths=0.5,
                label=f"{label} Pareto",
                zorder=10,
            )

            # Connect Pareto points with step function
            ax.step(
                sorted_pareto[:, 0],
                sorted_pareto[:, 1],
                color=color,
                linewidth=1.5,
                where="post",
                alpha=0.7,
            )

    # Plot reference point
    if ref_point[0] is not None and ref_point[1] is not None:
        ax.scatter(
            ref_point[0],
            ref_point[1],
            c="black",
            s=150,
            marker="s",
            label="Reference point",
            zorder=5,
        )

    # Formatting
    ax.set_xlabel(objective_names[0] if len(objective_names) > 0 else "Objective 1", fontsize=12)
    ax.set_ylabel(objective_names[1] if len(objective_names) > 1 else "Objective 2", fontsize=12)

    if title is None:
        title = f"{problem_name}: Pareto Front"
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add legend for point types
    legend_elements = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                   markersize=12, markeredgecolor="black", label="Pareto optimal"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=8, alpha=0.3, label="Feasible (dominated)"),
        plt.Line2D([0], [0], marker="x", color="gray", markersize=8,
                   alpha=0.3, linestyle="None", label="Infeasible"),
    ]

    # Add second legend
    legend2 = ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=9,
        title="Point types",
        framealpha=0.9,
    )
    ax.add_artist(legend2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")


def plot_comparison_summary(
    all_results: Dict[str, Dict[str, Any]],
    output_path: str,
):
    """Generate summary bar chart comparing methods across problems.

    Args:
        all_results: Dict mapping problem name to results
        output_path: Path to save the plot
    """
    if not all_results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    problems = list(all_results.keys())
    methods = list(all_results[problems[0]]["stats"].keys())
    n_problems = len(problems)
    n_methods = len(methods)

    x = np.arange(n_problems)
    width = 0.35

    for i, method in enumerate(methods):
        means = []
        sems = []
        for problem in problems:
            stats = all_results[problem]["stats"][method]
            means.append(stats["final_hv_mean"])
            sems.append(stats["final_hv_sem"])

        color = METHOD_COLORS.get(method, f"C{i}")
        label = METHOD_LABELS.get(method, method)

        bars = ax.bar(
            x + i * width - width / 2,
            means,
            width,
            label=label,
            color=color,
            yerr=sems,
            capsize=3,
        )

    ax.set_xlabel("Problem", fontsize=12)
    ax.set_ylabel("Final Hypervolume", fontsize=12)
    ax.set_title("MOO Benchmark: Final Hypervolume Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(problems, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    """Generate MOO benchmark plots."""
    parser = argparse.ArgumentParser(
        description="Generate plots from MOO benchmark results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmark_results/moo",
        help="Directory containing results JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (defaults to results-dir)",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["all", "hypervolume", "pareto", "summary"],
        default="all",
        help="Type of plots to generate",
    )
    parser.add_argument(
        "--replication",
        type=int,
        default=0,
        help="Replication index for Pareto plot (0=first, -1=best)",
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = args.output_dir if args.output_dir else args.results_dir

    print("=" * 70)
    print("MOO BENCHMARK PLOTTING")
    print("=" * 70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Plot type: {args.plot_type}")
    print("=" * 70)

    # Load results
    all_results = load_results(args.results_dir)

    if not all_results:
        print("No results to plot.")
        return

    print(f"\nFound results for {len(all_results)} problem(s):")
    for name in all_results.keys():
        print(f"  - {name}")

    # Generate plots
    print("\nGenerating plots...")

    for problem_name, results in all_results.items():
        print(f"\n{problem_name}:")

        if args.plot_type in ["all", "hypervolume"]:
            output_path = os.path.join(output_dir, f"{problem_name}_hypervolume.png")
            plot_hypervolume_convergence(results, output_path)

        if args.plot_type in ["all", "pareto"]:
            output_path = os.path.join(output_dir, f"{problem_name}_pareto.png")
            plot_pareto_front(results, output_path, replication_idx=args.replication)

    # Generate summary plot if multiple problems
    if args.plot_type in ["all", "summary"] and len(all_results) >= 1:
        print("\nSummary:")
        output_path = os.path.join(output_dir, "moo_summary.png")
        plot_comparison_summary(all_results, output_path)

    print("\n" + "=" * 70)
    print("Plotting complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
