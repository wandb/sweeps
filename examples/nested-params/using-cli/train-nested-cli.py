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
parser.add_argument("--optimizer", type=str, default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    wandb.init(project=args.project, config=args)

    # Do some fake taining
    for epoch in range(wandb.config["epochs"]):
        # You can access nested properties in the config!
        _batch_size = wandb.config["batch_size"]
        _lr = wandb.config["optimizer"]["lr"]
        _beta = wandb.config["optimizer"]["nadam"]["beta"]
        print(
            f"Fake training with batch size {_batch_size} and lr {_lr} and beta {_beta}"
        )
        # Fake loss has following relationships:
        # - goes down with each epoch
        # - larger batch size makes it go down faster
        # - larger learning rate makes it go down faster
        _fake_loss = 1 - (epoch / wandb.config["epochs"]) * _lr * _batch_size
        wandb.log({"loss": _fake_loss})
        time.sleep(0.3)

    wandb.finish()
