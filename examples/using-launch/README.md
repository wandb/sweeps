# Sweeps on Launch 🚀 

_Using Launch to run a sweep._

Create the sweep as you would normally, but specify a `queue`. Here we specify the default queue

```
wandb sweep sweep-nested.yaml --queue default
```

Within the [Launch UI in your workspace](https://wandb.ai/wandb/launch-welcome/launch) you should now see a launch queue with a daimyo job on it.

Start the daimyo by pointing a launch agent at the queue:

```
wandb launch-agent -q default -p nested-examples
```

Within the [Launch UI in your workspace](https://wandb.ai/wandb/launch-welcome/launch) you should now see sweeps jobs on the launch queue, these are being added there by the daimyo.

Start another launch agent to work through these jobs.

```
wandb launch-agent -q default -p nested-examples
```