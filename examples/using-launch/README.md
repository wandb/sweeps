# Sweeps on Launch 🚀 

Using Launch to run a sweep:

### Through the CLI

Create the sweep as you would normally, but specify a `queue`. 

```
wandb sweep sweep-nested.yaml --queue mylaunchsweep
```

Within the Launch UI in your workspace you should now see a launch queue with agent work on it.



Start a launch agent to work through that queue.

```
wandb launch-agent -q mylaunchsweep
```