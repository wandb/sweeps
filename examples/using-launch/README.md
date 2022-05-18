# Sweeps on Launch 🚀 

_Using Launch to run a sweep._

Create the sweep as you would normally, but specify a `queue`. Here we specify the default queue

```
wandb sweep sweep-nested.yaml --queue default
```

Within the [Launch UI in your workspace](https://wandb.ai/wandb/launch-welcome/launch) you should now see a launch queue with some work on it.

Start a launch agent to work through that queue.

```
wandb launch-agent -q default -p nested-examples
```

The launch agent will run the classic sweep agent as a local python process.