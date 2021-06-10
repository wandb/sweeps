# W&B Hyperparameter Sweeps Engine

This repo contains the routines that generate hyperparameter sweep suggestions in
the W&B backend and [client](https://github.com/wandb/client) local controller.

Issues are not enabled in this repository.
__Please [open issues related to sweeps in the wandb client library github issues page](https://github.com/wandb/client/issues/new/choose).__

### Installation
This repository is designed to be pinned as a submodule by repos that depend on it.
To install:

```shell script
cd repository_that_requires_sweeps
git submodule add https://github.com/wandb/sweeps.git
```

To install `sweeps` dependencies when you install your repository's dependencies, add a line to your repository's `requirements.txt` that points to `sweeps`'s requirements.txt:
```
-r sweeps/requirements.txt
```

### Examples

__Get next run in a sweep.__

Requires two arguments, `config`, the config that defines the sweep, and `runs`, the other runs in the sweep

`config`:
```python
{
    "metric": {"name": "loss", "goal": "minimize"},
    "method": "bayes",
    "parameters": {
        "v1": {"min": 1, "max": 10},
        "v2": {"min": 1.0, "max": 10.0},
    },
}
```

`runs`:
```python
[
    SweepRun(
        name="b",
        state=RunState.finished,
        history=[
            {"loss": 5.0},
        ],
        config={"v1": {"value": 7}, "v2": {"value": 6}},
        summary_metrics={"zloss": 1.2},
    ),
    SweepRun(
        name="b2",
        state=RunState.finished,
        config={"v1": {"value": 1}, "v2": {"value": 8}},
        summary_metrics={"loss": 52.0},
        history=[],
    )
]
```

Codepath:

```python
suggestion = next_run(config, runs)
```
`next_run`:
* validates that sweep config conforms to the jsonschema in `config/schema.json`, if not, it raises a `ValidationError`
* parses the config file and determines the method that it should use to find the next run (in this case `bayes_search_next_run`)
* calls `bayes_search_next_run(config, runs)` and returns the suggested `SweepRun`



__Return list of runs to stop in a sweep.__

Requires two arguments, `config`, the config that defines the sweep, and `runs`, the other runs in the sweep

`config`:
```python
{
    "method": "grid",
    "metric": {"name": "loss", "goal": "minimize"},
    "early_terminate": {
        "type": "hyperband",
        "max_iter": 5,
        "eta": 2,
        "s": 2,
    },
    "parameters": {"a": {"values": [1, 2, 3]}},
}
```

`runs`:
```python
[
    SweepRun(
        name="a",
        state=RunState.finished,  # This is already stopped
        history=[
            {"loss": 10},
            {"loss": 9},
        ],
    ),
    SweepRun(
        name="b",
        state=RunState.running,  # This should be stopped
        history=[
            {"loss": 10},
            {"loss": 10},
        ],
    ),
    SweepRun(
        name="c",
        state=RunState.running,  # This passes band 1 but not band 2
        history=[
            {"loss": 10},
            {"loss": 8},
            {"loss": 8},
        ],
    ),
    SweepRun(
        name="d",
        state=RunState.running,
        history=[
            {"loss": 10},
            {"loss": 7},
            {"loss": 7},
        ],
    ),
    SweepRun(
        name="e",
        state=RunState.finished,
        history=[
            {"loss": 10},
            {"loss": 6},
            {"loss": 6},
        ],
    ),
]
```

Codepath:

```python
to_stop = stop_runs(config, runs)
```
`stop_runs`:
* validates that sweep config conforms to the jsonschema in `config/schema.json`, if not, it raises a `ValidationError`
* parses the config file and determines the method that it should use to early terminate runs (in this case `hyperband_stop_runs`)
* calls `hyperband_stop_runs(config, runs)` and returns the `SweepRun`s to stop


### Testing
To run tests:

```
pip install -r requirements.dev.txt
pytest
```




### Contributing

Install the development requirements:

```
pip install -r requirements.dev.txt
```

Install the pre-commit hooks:
```
pre-commit install .
```

PRs must:

* Not degrade test coverage (automatically calculated via codecov)
* Use type-hints on all public functions
* Pass linting checks
* Be reviewed and approved by at least 1 maintainer
