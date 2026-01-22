# Ax Benchmarks and Integration Tests

Tests and benchmarks validating the Ax integration with W&B Sweeps.

## Overview

This directory contains:
- **Benchmarks** comparing Ax against the existing sklearn-based Bayesian optimization
- **Integration tests** verifying the W&B wrapper produces correct results
- **Test problems** (synthetic functions) for reproducible evaluation

For a simple getting started example, see `examples/ax_getting_started.py`.

## Files

| File | Description |
|------|-------------|
| `moo_benchmark_synthetic.py` | Benchmarks Ax MOO on test functions (DTLZ2, C2DTLZ2, WeldedBeam), tracking hypervolume |
| `bayes_benchmark_synthetic.py` | Compares Ax vs sklearn-bayes on single-objective problems (Branin, Hartmann6, etc.) |
| `bayes_benchmark_mnist.py` | Compares Ax vs sklearn-bayes on real MNIST hyperparameter tuning |
| `ax_integration_test.py` | Validates W&B Ax wrapper matches direct Ax Client API exactly |
| `ax_wandb_pipeline_test.py` | Tests full W&B logging pipeline with Ax suggestions |
| `plot_moo_results.py` | Generates hypervolume plots from benchmark results |
| `benchmark_problems.py` | Defines test problems (Branin, DTLZ2, etc.) |
| `benchmark_utils.py` | Shared utilities (JSON serialization) |
| `configs/` | Sample YAML configs for constrained MOO problems |

## Quick Start

**Run MOO benchmark:**
```bash
python moo_benchmark_synthetic.py --problems dtlz2 --trials 32 --replications 1
```

**Compare Ax vs sklearn-bayes:**
```bash
python bayes_benchmark_synthetic.py --problems branin --trials 24 --replications 3
```

**Verify correctness (W&B wrapper = Direct Ax):**
```bash
python ax_integration_test.py --problem dtlz2 --trials 20 --replications 3
```

## Benchmark Methodology

- **Hypervolume**: Standard MOO metric for Pareto front quality
- **Multiple replications**: Different random seeds for statistical robustness
- **JSON output**: Per-replication curves, final Pareto fronts, aggregate statistics

Results are saved to `benchmark_results/` for analysis and plotting.
