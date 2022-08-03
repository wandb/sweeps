#!/usr/bin/env python
import argparse
import random
import time

import wandb

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--project", type=str, default="sweep-examples", help="project")
parser.add_argument(
    "--epochs", type=int, default=2, help="total number of epochs to run."
)
parser.add_argument(
    "--variance", type=float, default=1, help="variance on mock metric starting value."
)
parser.add_argument(
    "--base", type=float, default=5, help="starting value of mock metric."
)
parser.add_argument(
    "--increment", type=float, default=5, help="increment magnitude for mock metric."
)
parser.add_argument(
    "--direction", type=float, default=1, help="direction of increment for mock metric."
)
parser.add_argument(
    "--sleep", type=float, default=1, help="time to wait between each epoch."
)
args = parser.parse_args()

run = wandb.init(project=args.project)

# The wandb config object holds the latest hyperparameter values
print(f"wandb.config: {wandb.config}")

# Mock model training
metric = args.base + random.random() * args.variance
for e, x in enumerate(range(args.epochs)):
    metric += args.increment * args.direction
    if metric < 0:
        metric = 0
    wandb.log({"loss_metric": metric})
    print("INFO: epoch %3d = loss %f" % (e, metric))
    time.sleep(args.sleep)
