# Sweeps on Launch 🚀 

_Using Launch to run a sweep._

### Local Process

For this example, make sure you are in this folder:

```
cd sweeps/examples/launch-jobs
```

Launch requires a Job Artifact to run jobs. Create one by running the example:

```
python train-launch-jobs.py
```

Now that you have that Job Artifact, you can create a sweep that uses it. You will also need to specify a Queue and the Resource.

```
wandb sweep \
    --queue default \
    --resource local-process \
    --job="job-source-sweep-examples-examples_launch-jobs_train-launch-jobs.py:v0" \
    sweep-launch-jobs.yaml
```

Within the [Launch UI in your workspace](https://wandb.ai/wandb/launch-welcome/launch) you should now see a launch Queue with a Scheduler job on it. Start the Scheduler by pointing a launch agent at the queue:

```
wandb launch-agent \
    --queue default \
    --project sweeps-examples \
    -j -1
```

Within the [Launch UI in your workspace](https://wandb.ai/wandb/launch-welcome/launch) you should now see sweeps jobs on the launch queue, these are being added there by the Scheduler. The launch agent will now work through these jobs.

### Kubernetes

For this example, make sure you are in this folder:

```
cd sweeps/examples/launch-jobs
```

Launch requires a Job Artifact to run jobs. Create one by running the example:

```
WANDB_ENTITY="launch-test" \
python train-launch-jobs.py
```

Now that you have that Job Artifact, you can create a sweep that uses it. You will also need to specify a Queue and the Resource.

```
WANDB_ENTITY="launch-test" \
wandb sweep \
    --queue default \
    --resource kubernetes \
    --job="job-source-sweep-examples-examples_launch-jobs_train-launch-jobs.py:v0" \
    sweep-launch-jobs.yaml
```

Within the [Launch UI in your workspace](https://wandb.ai/wandb/launch-welcome/launch) you should now see a launch Queue with a Scheduler job on it. Start the Scheduler by pointing a launch agent at the queue:

```
wandb launch-agent \
    --queue default \
    --project sweeps-examples \
    -j -1
```

Within the [Launch UI in your workspace](https://wandb.ai/wandb/launch-welcome/launch) you should now see sweeps jobs on the launch queue, these are being added there by the Scheduler. The launch agent will now work through these jobs.