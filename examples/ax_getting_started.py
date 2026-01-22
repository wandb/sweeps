#!/usr/bin/env python3
"""Getting started with Ax Bayesian optimization in W&B Sweeps.

This example shows how to use Ax for hyperparameter optimization with W&B logging.
Ax provides sophisticated Bayesian optimization that learns from previous trials
to suggest better hyperparameters.

Usage:
    # Run directly (uses wandb offline mode for demo)
    python ax_getting_started.py

    # Or use with a W&B sweep:
    # 1. Create sweep: wandb sweep sweep-ax.yaml
    # 2. Run agent: wandb agent <sweep_id>

The key differences from standard W&B sweeps:
    - method: ax (instead of bayes, grid, or random)
    - Ax learns from failed trials automatically
    - Supports log-scale, categorical, and integer parameters
    - Can handle multiple objectives (see sweep-ax-moo.yaml)
"""

import argparse
import math
import os
import sys

# For demo purposes, run in offline mode
if "WANDB_MODE" not in os.environ:
    os.environ["WANDB_MODE"] = "offline"

import wandb

# Add path to import sweeps
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sweeps.run import next_runs, RunState, SweepRun


def objective_function(learning_rate: float, num_layers: int, dropout: float) -> float:
    """A simple objective function to optimize.

    In practice, this would be your model training code that returns
    a validation metric (loss, accuracy, etc.)

    This mock function has an optimum around:
        learning_rate=0.01, num_layers=3, dropout=0.2
    """
    # Simulate a loss landscape with some noise
    lr_term = (math.log10(learning_rate) + 2) ** 2  # optimum at lr=0.01
    layer_term = (num_layers - 3) ** 2  # optimum at 3 layers
    dropout_term = (dropout - 0.2) ** 2  # optimum at 0.2 dropout

    loss = lr_term + 0.5 * layer_term + 2 * dropout_term
    return loss


def main():
    parser = argparse.ArgumentParser(
        description="Ax + W&B getting started example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trials", type=int, default=15, help="Number of trials to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Define the sweep configuration
    # This is the same format you'd put in a YAML file
    config = {
        "method": "ax",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {
                "min": 0.0001,
                "max": 0.1,
                "distribution": "log_uniform_values",
            },
            "num_layers": {"min": 1, "max": 5},
            "dropout": {"min": 0.0, "max": 0.5},
        },
    }

    print("=" * 60)
    print("Ax + W&B Sweeps: Getting Started")
    print("=" * 60)
    print(f"Running {args.trials} trials with Ax Bayesian optimization")
    print("=" * 60)

    # Track completed runs for Ax to learn from
    completed_runs = []

    for trial in range(args.trials):
        # Get next suggestion from Ax
        suggestions = next_runs(
            config, completed_runs, validate=False, n=1, random_seed=args.seed
        )

        if not suggestions:
            print("No more suggestions from Ax")
            break

        # Extract hyperparameters
        params = {k: v["value"] for k, v in suggestions[0].config.items()}

        # Initialize W&B run with suggested hyperparameters
        run = wandb.init(
            project="ax-getting-started",
            config=params,
            reinit=True,
        )

        # Evaluate the objective (your training code goes here)
        loss = objective_function(
            learning_rate=params["learning_rate"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
        )

        # Log results to W&B
        wandb.log({"loss": loss})

        print(
            f"Trial {trial + 1:2d}/{args.trials}: "
            f"lr={params['learning_rate']:.5f}, "
            f"layers={params['num_layers']}, "
            f"dropout={params['dropout']:.2f} "
            f"-> loss={loss:.4f}"
        )

        # Finish the W&B run
        wandb.finish(quiet=True)

        # Record completed run for Ax to learn from
        completed_runs.append(
            SweepRun(
                state=RunState.finished,
                config=suggestions[0].config,
                summary_metrics={"loss": loss},
            )
        )

    # Find best result
    best_run = min(completed_runs, key=lambda r: r.summary_metrics["loss"])
    best_params = {k: v["value"] for k, v in best_run.config.items()}

    print("\n" + "=" * 60)
    print("Best configuration found:")
    print(f"  learning_rate: {best_params['learning_rate']:.5f}")
    print(f"  num_layers: {best_params['num_layers']}")
    print(f"  dropout: {best_params['dropout']:.2f}")
    print(f"  loss: {best_run.summary_metrics['loss']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
