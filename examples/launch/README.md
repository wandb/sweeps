# Sweeps on Launch 🚀

There are currently three ways to run a Sweep that uses Launch Jobs:

- [Local Process](#local-process) - Run Jobs as local processes in your local machine
- [Local Container](#local-container) - Run Jobs as local (docker) containers in your local machine
- [Kubernetes](#kubernetes) - Run Jobs inside a K8s cluster previously configured through Launch.

These examples use the entity `launch-test` and project `wandb-launch-sweeps`. Replace with your wandb entity and project. These environment variables below are used for easy copy-pasting of the commands in the sections below.

```
export WANDB_ENTITY="launch-test"
export WANDB_PROJECT="wandb-launch-sweeps"
```

## Local Process

For this example, make sure you are in this folder:

```
cd sweeps/examples/launch
```

Launch requires a Job Artifact to run jobs. Create one by running the example:

```
python train-launch-jobs.py
```

Now that you have that Job Artifact, you can create a sweep that uses it:

```
wandb sweep \
    --launch_config launch-config-local-process.yaml \
    --entity="${WANDB_ENTITY}" \
    --project="${WANDB_PROJECT}" \
    sweep-launch-jobs.yaml
```

Within the [Launch UI in your workspace](https://wandb.ai/wandb/launch-welcome/launch) you should now see a launch Queue with a Scheduler job on it. Start the Scheduler by pointing a launch agent at the queue:

```
wandb launch-agent \
    --queues default \
    --entity="${WANDB_ENTITY}" \
    --project="${WANDB_PROJECT}" \
    -j -1
```

Within the [Launch UI in your workspace](https://wandb.ai/wandb/launch-welcome/launch) you should now see sweeps jobs on the launch queue, these are being added there by the Scheduler. The launch agent will now work through these jobs.

## Local Container

For this example, make sure you are in this folder:

```
cd sweeps/examples/launch-jobs
```

Launch requires a Job Artifact to run jobs. Create one by running the example:

```
python train-launch-jobs.py
```

Now that you have that Job Artifact, you can create a sweep that uses it:

```
wandb sweep \
    --launch_config launch-config-local-container.yaml \
    --entity="${WANDB_ENTITY}" \
    --project="${WANDB_PROJECT}" \
    sweep-launch-jobs.yaml
```

Within the [Launch UI in your workspace](https://wandb.ai/wandb/launch-welcome/launch) you should now see a launch Queue with a Scheduler job on it. Start the Scheduler by pointing a launch agent at the queue:

```
wandb launch-agent \
    --queues default \
    --entity="${WANDB_ENTITY}" \
    --project="${WANDB_PROJECT}" \
    -j -1
```

Within the [Launch UI in your workspace](https://wandb.ai/wandb/launch-welcome/launch) you should now see sweeps jobs on the launch queue, these are being added there by the Scheduler. The launch agent will now work through these jobs.

## Kubernetes

For this example, make sure you are in this folder:

```
cd sweeps/examples/launch-jobs
```

Launch requires a Job Artifact to run jobs. Create one by running the example:

```
WANDB_ENTITY="launch-test" \
python train-launch-jobs.py
```

Now that you have that Job Artifact, you can create a sweep that uses it.

```
WANDB_ENTITY="launch-test" \
wandb sweep \
    --launch_config launch-config-kube.yaml \
    sweep-launch-jobs.yaml
```

Within the [Launch UI in your workspace](https://wandb.ai/wandb/launch-welcome/launch) you should now see a launch Queue with a Scheduler job on it. Start the Scheduler by pointing a launch agent at the queue:

```
wandb launch-agent \
    --queues default \
    --project="wandb-launch-sweeps" \
    -j -1
```

Within the [Launch UI in your workspace](https://wandb.ai/wandb/launch-welcome/launch) you should now see sweeps jobs on the launch queue, these are being added there by the Scheduler. The launch agent will now work through these jobs.
