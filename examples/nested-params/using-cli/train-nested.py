"""
    Example using nested configs for sweeps and agents that
    are defined via YAML and run from the CLI. To run this example:

    > wandb sweep sweep-nested.yaml

    Then use the sweep id to run the agent:

    > wandb agent <USERNAME/PROJECTNAME/SWEEPID>
"""
import argparse
import time

import wandb

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--project", type=str, default=None, help="project")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.01)


def train_function(config):
    # Do some fake taining
    for _ in range(config["epochs"]):
        # You can access nested properties in the config!
        _batch_size = config["batch_size"]
        _lr = config["optimizer"]["lr"]
        print(f"Fake training with batch size {_batch_size} and lr {_lr}")
        wandb.log({"loss": 0.01})
        time.sleep(0.3)

if __name__ == "__main__":
    args = parser.parse_args()
    wandb.init(project=args.project, config=args)
    train_function(wandb.config)
    wandb.finish()