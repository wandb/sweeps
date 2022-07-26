#!/usr/bin/env python
import argparse
import random
import time
import os

import wandb

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--use_jobs", action='store_true', help="Uses Jobs, an experimental feature of Launch")
parser.add_argument("--project", type=str, default=None, help="project")
parser.add_argument("--epochs", type=int, default=5, help="epochs")
parser.add_argument("--variance", type=float, default=1, help="param1")
parser.add_argument("--base", type=float, default=5, help="param2")
parser.add_argument("--increment", type=float, default=5, help="param2")
parser.add_argument("--direction", type=float, default=5, help="param2")
parser.add_argument("--sleep", type=float, default=1, help="param2")
args = parser.parse_args()


if args.use_jobs:
    settings = wandb.Settings()
    settings.update({"enable_job_creation": True})
    run = wandb.init(project=args.project, settings=settings)
    # This will trigger the creation of aget Job artifact.
    run.log_code()
else:
    run = wandb.init(project=args.project)
    # Force configuration update (should not be needed in the future)
    wandb.config.update({"dummy": 1})
    time.sleep(1)

# The wandb config object holds the latest hyperparameter values
print(f"wandb.config: {wandb.config}")

metric = args.base + random.random() * args.variance
for e, x in enumerate(range(args.epochs)):
    metric += args.increment * args.direction
    if metric < 0:
        metric = 0
    wandb.log({"loss_metric": metric})
    print("INFO:  %3d = %f" % (e, metric))
    time.sleep(args.sleep)

if args.use_jobs:
    wandb.finish()