"""
    Example using nested configs for sweeps and agents that
    are defined and run from python. To run this example:

    > python train-nested.py
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
        "lr": 0.01, # The "lr" is nested behind "optimizer"
    }
}

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {
        "name": "loss",
        "goal": "minimize",
    },
    "parameters"  :{
        "batch_size": {"values": [32, 64]},
        "optimizer": {
            "parameters": {
                "lr": {"values": [0.001, 0.0001]},
            },
        },
    }
}


def _train_function(config):
    # Do some fake taining
    for _ in range(config["epochs"]):
        # You can access nested properties in the config!
        _batch_size = config["batch_size"]
        _lr = config["optimizer"]["lr"]
        print(f"Fake training with batch size {_batch_size} and lr {_lr}")
        wandb.log({"loss_metric": 0.01})
        time.sleep(0.3)

if __name__ == "__main__":
    args = parser.parse_args()
    print('Create sweep from python.')
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.project)
    
    def train_function():
        with wandb.init(project=args.project, config=CONFIG):
            _train_function(wandb.config)

    print('Create and run agent from python.')
    wandb.agent(sweep_id, train_function, count=2)