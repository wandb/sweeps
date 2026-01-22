#!/usr/bin/env python3
"""Simplified benchmarking using Ax's native benchmark framework.
 
This module provides a streamlined approach to benchmarking by:
1. Using Ax's native BenchmarkProblem and create_problem_from_botorch()
2. Wrapping W&B searchers as ExternalGenerationNode implementations
3. Supporting both direct calls and full W&B logger flow
 
Usage:
    # Run SOO benchmark (bayes vs ax)
    python ax_native_benchmarks.py --mode soo --trials 50 --replications 10
 
    # Run MOO benchmark
    python ax_native_benchmarks.py --mode moo --trials 50 --replications 10
 
    # Run integration test (vanilla ax vs W&B e2e)
    python ax_native_benchmarks.py --mode integration --trials 20 --replications 5
"""
 
from __future__ import annotations
 
import argparse
import os
import sys
import tempfile
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
 
import numpy as np
 
# Set W&B offline before importing
os.environ.setdefault("WANDB_MODE", "offline")
 
try:
    import torch
    from botorch.test_functions.multi_objective import C2DTLZ2, DTLZ2, WeldedBeam
    from botorch.test_functions.synthetic import Branin, Hartmann
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    from botorch.utils.multi_objective.pareto import is_non_dominated
except ImportError as e:
    print(f"Error: Required package missing: {e}")
    sys.exit(1)
 
# Add sweeps to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
 
from sweeps.run import RunState, SweepRun, next_runs  # noqa: E402
from sweeps.config import SweepConfig  # noqa: E402
from sweeps.bayes_search import bayes_search_next_runs  # noqa: E402
from sweeps.ax_search import ax_search_next_runs  # noqa: E402
 
warnings.filterwarnings("ignore")
 
 
# =============================================================================
# Problem Definitions (minimal wrappers - just what Ax doesn't provide)
# =============================================================================
 
@dataclass
class BenchmarkProblem:
    """Minimal benchmark problem definition."""
    name: str
    dim: int
    bounds: list[tuple[float, float]]
    evaluate: Callable[[dict[str, float]], dict[str, float]]
    optimal_value: float | None = None
    is_minimization: bool = True
    # MOO fields
    num_objectives: int = 1
    objective_names: list[str] = field(default_factory=lambda: ["value"])
    ref_point: list[float] | None = None
    num_constraints: int = 0
    constraint_names: list[str] = field(default_factory=list)
 
    def create_sweep_config(self, method: str) -> dict[str, Any]:
        """Generate sweep config for this problem."""
        parameters = {
            f"x{i}": {"min": float(lo), "max": float(hi)}
            for i, (lo, hi) in enumerate(self.bounds)
        }
 
        if self.num_objectives == 1:
            return {
                "method": method,
                "parameters": parameters,
                "metric": {
                    "name": self.objective_names[0],
                    "goal": "minimize" if self.is_minimization else "maximize",
                },
            }
        else:
            config = {
                "method": method,
                "parameters": parameters,
                "metrics": [
                    {"name": name, "goal": "minimize", "threshold": self.ref_point[i]}
                    for i, name in enumerate(self.objective_names)
                ],
            }
            if self.num_constraints > 0:
                config["metric_constraints"] = [f"{n} <= 0" for n in self.constraint_names]
            return config
 
 
def _make_botorch_evaluator(func, dim: int, obj_names: list[str], constraint_names: list[str] | None = None):
    """Create an evaluate function from a BoTorch test function."""
    def evaluate(params: dict[str, float]) -> dict[str, float]:
        X = torch.tensor([[params[f"x{i}"] for i in range(dim)]], dtype=torch.double)
        result = {}
 
        # Objectives
        Y = func(X)
        if Y.dim() == 1:
            result[obj_names[0]] = float(Y.item())
        else:
            for i, name in enumerate(obj_names):
                result[name] = float(Y[0, i].item())
 
        # Constraints
        if constraint_names and hasattr(func, "evaluate_slack"):
            slack = func.evaluate_slack(X)
            for i, name in enumerate(constraint_names):
                result[name] = float(-slack[0, i].item())  # Convert slack to g <= 0
 
        return result
    return evaluate
 
 
# Problem registry using factory functions
def _branin() -> BenchmarkProblem:
    func = Branin()
    return BenchmarkProblem(
        name="Branin_2D", dim=2,
        bounds=[(-5.0, 10.0), (0.0, 15.0)],
        evaluate=_make_botorch_evaluator(func, 2, ["value"]),
        optimal_value=0.397887,
    )
 
def _hartmann6() -> BenchmarkProblem:
    func = Hartmann(dim=6)
    return BenchmarkProblem(
        name="Hartmann6_6D", dim=6,
        bounds=[(0.0, 1.0)] * 6,
        evaluate=_make_botorch_evaluator(func, 6, ["value"]),
        optimal_value=-3.32237,
    )
 
def _dtlz2() -> BenchmarkProblem:
    func = DTLZ2(dim=6, num_objectives=2, negate=False)
    return BenchmarkProblem(
        name="DTLZ2_6D", dim=6, num_objectives=2,
        bounds=[(0.0, 1.0)] * 6,
        evaluate=_make_botorch_evaluator(func, 6, ["f1", "f2"]),
        objective_names=["f1", "f2"],
        ref_point=[1.1, 1.1],
    )
 
def _c2dtlz2() -> BenchmarkProblem:
    func = C2DTLZ2(dim=6, num_objectives=2, negate=False)
    return BenchmarkProblem(
        name="C2DTLZ2_6D", dim=6, num_objectives=2, num_constraints=1,
        bounds=[(0.0, 1.0)] * 6,
        evaluate=_make_botorch_evaluator(func, 6, ["f1", "f2"], ["c1"]),
        objective_names=["f1", "f2"],
        constraint_names=["c1"],
        ref_point=[1.1, 1.1],
    )
 
def _welded_beam() -> BenchmarkProblem:
    func = WeldedBeam(negate=False)
    bounds = [(0.125, 5.0), (0.1, 10.0), (0.1, 10.0), (0.125, 5.0)]
    return BenchmarkProblem(
        name="WeldedBeam_4D", dim=4, num_objectives=2, num_constraints=4,
        bounds=bounds,
        evaluate=_make_botorch_evaluator(func, 4, ["cost", "deflection"], ["g1", "g2", "g3", "g4"]),
        objective_names=["cost", "deflection"],
        constraint_names=["g1", "g2", "g3", "g4"],
        ref_point=[40.0, 0.015],
    )
 
PROBLEMS = {
    "branin": _branin, "hartmann6": _hartmann6,
    "dtlz2": _dtlz2, "c2dtlz2": _c2dtlz2, "welded_beam": _welded_beam,
}
 
 
# =============================================================================
# External Generation Nodes (wrap W&B searchers for Ax benchmark framework)
# =============================================================================
 
class BaseSearchNode:
    """Base class for search method wrappers."""
 
    def __init__(self, sweep_config: dict[str, Any], random_seed: int = 42):
        self.sweep_config = sweep_config
        self.random_seed = random_seed
        self.sweep_runs: list[SweepRun] = []
 
    @abstractmethod
    def get_next_candidate(self) -> dict[str, float]:
        """Get next parameter suggestion."""
        pass
 
    def record_observation(self, params: dict[str, float], metrics: dict[str, float]) -> None:
        """Record a completed trial."""
        self.sweep_runs.append(SweepRun(
            state=RunState.finished,
            config={k: {"value": v} for k, v in params.items()},
            summary_metrics=metrics,
        ))
 
 
class DirectBayesNode(BaseSearchNode):
    """Direct call to bayes_search_next_runs (no W&B logging)."""
 
    def get_next_candidate(self) -> dict[str, float]:
        suggestions = bayes_search_next_runs(self.sweep_runs, self.sweep_config, n=1)
        return {k: v["value"] for k, v in suggestions[0].config.items()}
 
 
class DirectAxNode(BaseSearchNode):
    """Direct call to ax_search_next_runs (no W&B logging)."""
 
    def get_next_candidate(self) -> dict[str, float]:
        suggestions = ax_search_next_runs(
            self.sweep_runs, self.sweep_config, n=1, random_seed=self.random_seed
        )
        return {k: v["value"] for k, v in suggestions[0].config.items()}
 
 
class WandBLoggerNode(BaseSearchNode):
    """Full W&B logger flow: next_runs → wandb.init → wandb.log → wandb.finish."""
 
    def __init__(self, sweep_config: dict[str, Any], random_seed: int = 42, wandb_dir: str | None = None):
        super().__init__(sweep_config, random_seed)
        self.wandb_dir = wandb_dir or tempfile.mkdtemp()
        self._pending_params: dict[str, float] | None = None
 
    def get_next_candidate(self) -> dict[str, float]:
        config = SweepConfig(self.sweep_config)
        suggestions = next_runs(config, self.sweep_runs, validate=False, n=1, random_seed=self.random_seed)
        self._pending_params = {k: v["value"] for k, v in suggestions[0].config.items()}
        return self._pending_params
 
    def record_observation(self, params: dict[str, float], metrics: dict[str, float]) -> None:
        """Record via full W&B flow."""
        import wandb
 
        run = wandb.init(
            project="ax-native-benchmark",
            config=params,
            dir=self.wandb_dir,
            reinit=True,
        )
        wandb.log(metrics)
        wandb.finish(quiet=True)
 
        # Also record as SweepRun for next iteration
        super().record_observation(params, metrics)
 
 
# =============================================================================
# Benchmark Runner
# =============================================================================
 
@dataclass
class ReplicationResult:
    """Result of a single benchmark replication."""
    seed: int
    best_values: list[float]  # SOO: best value at each trial
    hypervolume_curve: list[float]  # MOO: hypervolume at each trial
    final_value: float  # SOO: final best, MOO: final hypervolume
 
 
def compute_hypervolume(Y: np.ndarray, ref_point: list[float]) -> float:
    """Compute hypervolume for minimization objectives."""
    if len(Y) == 0:
        return 0.0
    neg_ref = torch.tensor([-r for r in ref_point], dtype=torch.double)
    hv = Hypervolume(ref_point=neg_ref)
    return float(hv.compute(torch.tensor(-Y, dtype=torch.double)))
 
 
def get_pareto_front(Y: np.ndarray, feasible: np.ndarray | None = None) -> np.ndarray:
    """Get Pareto front from objective values."""
    if len(Y) == 0:
        return np.array([])
    if feasible is not None:
        Y = Y[feasible]
        if len(Y) == 0:
            return np.array([])
    Y_tensor = torch.tensor(Y, dtype=torch.double)
    mask = is_non_dominated(-Y_tensor)  # Negate for minimization
    return Y_tensor[mask].numpy()
 
 
def run_replication(
    problem: BenchmarkProblem,
    node: BaseSearchNode,
    num_trials: int,
) -> ReplicationResult:
    """Run a single benchmark replication."""
    all_values = []
    all_Y = []
    all_feasible = []
    best_values = []
    hv_curve = []
 
    for _ in range(num_trials):
        params = node.get_next_candidate()
        result = problem.evaluate(params)
        node.record_observation(params, result)
 
        if problem.num_objectives == 1:
            # SOO tracking
            val = result[problem.objective_names[0]]
            all_values.append(val)
            best_so_far = min(all_values) if problem.is_minimization else max(all_values)
            best_values.append(best_so_far)
        else:
            # MOO tracking
            obj_vals = [result[name] for name in problem.objective_names]
            all_Y.append(obj_vals)
 
            if problem.num_constraints > 0:
                is_feasible = all(result[c] <= 0 for c in problem.constraint_names)
            else:
                is_feasible = True
            all_feasible.append(is_feasible)
 
            Y_array = np.array(all_Y)
            feasible_mask = np.array(all_feasible)
            if np.any(feasible_mask):
                pareto = get_pareto_front(Y_array, feasible_mask)
                hv = compute_hypervolume(pareto, problem.ref_point) if len(pareto) > 0 else 0.0
            else:
                hv = 0.0
            hv_curve.append(hv)
 
    if problem.num_objectives == 1:
        final = best_values[-1] if best_values else float("inf")
    else:
        final = hv_curve[-1] if hv_curve else 0.0
 
    return ReplicationResult(
        seed=node.random_seed,
        best_values=best_values,
        hypervolume_curve=hv_curve,
        final_value=final,
    )
 
 
def benchmark_method(
    problem: BenchmarkProblem,
    node_factory: Callable[[dict, int], BaseSearchNode],
    num_trials: int,
    num_replications: int,
    base_seed: int = 0,
    method: str = "ax",
) -> list[ReplicationResult]:
    """Run benchmark across multiple replications."""
    results = []
    config = problem.create_sweep_config(method)
 
    for i in range(num_replications):
        seed = base_seed + i
        np.random.seed(seed)
        torch.manual_seed(seed)
 
        print(f"    Rep {i+1}/{num_replications} (seed={seed})... ", end="", flush=True)

        node = node_factory(config, seed)
        result = run_replication(problem, node, num_trials)
        results.append(result)

        metric = "HV" if problem.num_objectives > 1 else "Best"
        print(f"{metric}={result.final_value:.6f}")
 
    return results
 
 
def compute_stats(results: list[ReplicationResult], is_moo: bool) -> dict[str, float]:
    """Compute aggregate statistics."""
    finals = [r.final_value for r in results]
    return {
        "mean": float(np.mean(finals)),
        "sem": float(np.std(finals) / np.sqrt(len(finals))),
        "median": float(np.median(finals)),
    }
 
 
# =============================================================================
# Main Benchmark Modes
# =============================================================================
 
def run_soo_benchmark(problems: list[str], num_trials: int, num_replications: int, seed: int):
    """Head-to-head SOO benchmark: bayes vs ax (direct)."""
    print("=" * 70)
    print("SOO BENCHMARK: bayes vs ax (direct)")
    print("=" * 70)
 
    for problem_name in problems:
        problem = PROBLEMS[problem_name]()
        if problem.num_objectives > 1:
            continue
 
        print(f"\n{problem.name} (optimal={problem.optimal_value})")
 
        for method_name, node_factory in [
            ("bayes", lambda c, s: DirectBayesNode(c, s)),
            ("ax", lambda c, s: DirectAxNode(c, s)),
        ]:
            print(f"  {method_name}:")
            results = benchmark_method(problem, node_factory, num_trials, num_replications, seed, method=method_name)
            stats = compute_stats(results, is_moo=False)
            print(f"    → {stats['mean']:.6f} ± {stats['sem']:.6f}")
 
 
def run_moo_benchmark(problems: list[str], num_trials: int, num_replications: int, seed: int):
    """MOO benchmark using ax (direct)."""
    print("=" * 70)
    print("MOO BENCHMARK: ax (direct)")
    print("=" * 70)
 
    for problem_name in problems:
        problem = PROBLEMS[problem_name]()
        if problem.num_objectives == 1:
            continue
 
        print(f"\n{problem.name} ({problem.num_objectives} obj, {problem.num_constraints} constraints)")
        print(f"  ax:")
 
        results = benchmark_method(
            problem,
            lambda c, s: DirectAxNode(c, s),
            num_trials, num_replications, seed,
            method="ax",
        )
        stats = compute_stats(results, is_moo=True)
        print(f"    → Final HV: {stats['mean']:.6f} ± {stats['sem']:.6f}")
 
 
def run_integration_test(problem_name: str, num_trials: int, num_replications: int, seed: int):
    """Integration test: vanilla ax (direct) vs W&B e2e logger flow."""
    print("=" * 70)
    print("INTEGRATION TEST: Direct ax vs W&B Logger Flow")
    print("=" * 70)
 
    problem = PROBLEMS[problem_name]()
    print(f"\nProblem: {problem.name}")
 
    with tempfile.TemporaryDirectory() as wandb_dir:
        all_match = True
 
        for i in range(num_replications):
            rep_seed = seed + i
            np.random.seed(rep_seed)
            torch.manual_seed(rep_seed)
 
            config = problem.create_sweep_config("ax")
 
            # Direct path
            direct_node = DirectAxNode(config, rep_seed)
            direct_result = run_replication(problem, direct_node, num_trials)
 
            # Reset seed for W&B path
            np.random.seed(rep_seed)
            torch.manual_seed(rep_seed)
 
            # W&B logger path
            wandb_node = WandBLoggerNode(config, rep_seed, wandb_dir)
            wandb_result = run_replication(problem, wandb_node, num_trials)
 
            diff = wandb_result.final_value - direct_result.final_value
            match = abs(diff) < 1e-9
            all_match = all_match and match
            status = "OK" if match else "MISMATCH"
 
            metric = "HV" if problem.num_objectives > 1 else "Best"
            print(f"  Rep {i+1}/{num_replications}: Direct={direct_result.final_value:.6f} "
                  f"W&B={wandb_result.final_value:.6f} Diff={diff:+.2e} [{status}]")
 
    print("=" * 70)
    if all_match:
        print("PASS: W&B logger flow matches direct ax")
    else:
        print("FAIL: Results differ")
    print("=" * 70)
 
    return all_match
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Ax-native benchmarking with W&B searcher integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["soo", "moo", "integration", "all"], default="all",
        help="Benchmark mode",
    )
    parser.add_argument(
        "--problems", nargs="+", default=["branin", "hartmann6", "dtlz2"],
        choices=list(PROBLEMS.keys()),
        help="Problems to benchmark",
    )
    parser.add_argument("--trials", type=int, default=30, help="Trials per replication")
    parser.add_argument("--replications", type=int, default=5, help="Number of replications")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    args = parser.parse_args()
 
    if args.mode in ("soo", "all"):
        run_soo_benchmark(args.problems, args.trials, args.replications, args.seed)
 
    if args.mode in ("moo", "all"):
        run_moo_benchmark(args.problems, args.trials, args.replications, args.seed)
 
    if args.mode in ("integration", "all"):
        # Test on first MOO problem
        moo_problems = [p for p in args.problems if PROBLEMS[p]().num_objectives > 1]
        if moo_problems:
            run_integration_test(moo_problems[0], args.trials, args.replications, args.seed)
 
 
if __name__ == "__main__":
    main()
