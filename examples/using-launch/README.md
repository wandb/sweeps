# Sweeps on Launch 🚀 

_Using Launch to run a sweep._

Create the sweep as you would normally, but specify a `queue`. Here we specify the default queue

```
WANDB_BASE_URL=https://api.wandb.test wandb sweep sweep-nested.yaml --queue my_nested_sweep
```

Within the [Launch UI in your workspace](https://wandb.ai/wandb/launch-welcome/launch) you should now see a launch queue with a daimyo job on it.

Start the daimyo by pointing a launch agent at the queue:

```
WANDB_BASE_URL=https://api.wandb.test wandb launch-agent -q my_nested_sweep -j 2
```

Within the [Launch UI in your workspace](https://wandb.ai/wandb/launch-welcome/launch) you should now see sweeps jobs on the launch queue, these are being added there by the daimyo. The launch agent will now work through these jobs.