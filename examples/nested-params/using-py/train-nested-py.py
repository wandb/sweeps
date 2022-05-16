"""
    Example using nested configs for sweeps and agents that
    are defined and run from python. To run this example:

    > python train-nested-py.py
"""

import argparse
import time

import wandb

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--project", type=str, default=None, help="project")

# A user-specified nested config.
CONFIG = {
    "epochs": 5,
    "batch_size": 32,
    "optimizer": {
        "lr": 0.01,  # The "lr" is nested behind "optimizer"
    },
}

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {
        "name": "loss",
        "goal": "minimize",
    },
    "parameters": {
        "batch_size": {"values": [32, 64]},
        "optimizer": {
            "parameters": {
                "lr": {"values": [0.001, 0.0001]},
                "nadam": {
                    "parameters": {
                        "beta": {"values": [0.9, 0.95]},
                    },
                },
            }
        },
    },
}


def _train_function(config):
    # Do some fake taining
    for epoch in range(config["epochs"]):
        # You can access nested properties in the config!
        _batch_size = config["batch_size"]
        _lr = config["optimizer"]["lr"]
        _beta = wandb.config["optimizer"]["nadam"]["beta"]
        print(
            f"Fake training with batch size {_batch_size} and lr {_lr} and beta {_beta}"
        )
        # Fake loss has following relationships:
        # - goes down with each epoch
        # - larger batch size makes it go down faster
        # - larger learning rate makes it go down faster
        _fake_loss = 1 - (epoch / config["epochs"]) * _lr * _batch_size
        wandb.log({"loss": _fake_loss})
        time.sleep(0.3)


if __name__ == "__main__":
    args = parser.parse_args()
    print("Create sweep from python.")
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.project)

    def train_function():
        with wandb.init(project=args.project, config=CONFIG):
            _train_function(wandb.config)

    print("Create and run agent from python.")
    wandb.agent(sweep_id, train_function, count=3)
