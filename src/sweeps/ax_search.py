"""Bayesian optimization using Meta's ax-platform.

This module provides Bayesian optimization capabilities using the ax-platform
library, which offers more advanced optimization strategies compared to the
sklearn-based implementation in bayes_search.py.

Key Features:
- Leverages Ax's sophisticated trial status management (no manual imputation)
- Failed trials teach the model what parameter regions to avoid
- Running trials contribute as pending observations
- Advanced Bayesian optimization strategies from Meta Research

Supported parameter types:
- Continuous (min/max with uniform distribution)
- Integer (min/max with integer values)
- Categorical (values list)
- Log-scale (log_uniform_values distribution)

Not supported (use method='bayes' instead):
- Normal distributions (normal, q_normal, log_normal, q_log_normal)
- Beta distributions (beta, q_beta)
- Deprecated distributions (log_uniform, inv_log_uniform v1 variants)

Example usage:
    config = {
        'method': 'ax',
        'parameters': {
            'learning_rate': {'min': 0.001, 'max': 0.1, 'distribution': 'log_uniform_values'},
            'batch_size': {'min': 16, 'max': 128},
            'optimizer': {'values': ['adam', 'sgd', 'rmsprop']}
        },
        'metric': {'name': 'val_loss', 'goal': 'minimize'}
    }
    suggestions = ax_search_next_runs([], config, n=5)
"""

import logging
from contextlib import contextmanager
from typing import List, Union

# Sweeps imports
from .config.cfg import SweepConfig
from .params import HyperParameter, HyperParameterSet
from .run import RunState, SweepRun

# Ax imports (with clear error if not installed)
try:
    from ax.api.client import Client
    from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig
except ImportError as e:
    raise ImportError(
        "ax method requires ax-platform. " "Install with: pip install sweeps[ax]"
    ) from e

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_ax_logging():
    """Context manager to temporarily suppress Ax client's info-level logging.

    This is used to silence verbose output during bulk trial operations like
    attaching historical trials, which can produce excessive log messages.
    """
    # Get the Ax client logger
    ax_logger = logging.getLogger("ax.api.client")

    # Save current level and set to WARNING
    original_level = ax_logger.level
    ax_logger.setLevel(logging.WARNING)

    try:
        yield
    finally:
        # Restore original level
        ax_logger.setLevel(original_level)


# Map of unsupported parameter types to their display names
UNSUPPORTED_TYPES = {
    HyperParameter.NORMAL: "normal",
    HyperParameter.Q_NORMAL: "q_normal",
    HyperParameter.LOG_NORMAL: "log_normal",
    HyperParameter.Q_LOG_NORMAL: "q_log_normal",
    HyperParameter.BETA: "beta",
    HyperParameter.Q_BETA: "q_beta",
    HyperParameter.INV_LOG_UNIFORM_V1: "inv_log_uniform (deprecated)",
    HyperParameter.INV_LOG_UNIFORM_V2: "inv_log_uniform",
    HyperParameter.LOG_UNIFORM_V1: "log_uniform (deprecated)",
    HyperParameter.Q_LOG_UNIFORM_V1: "q_log_uniform (deprecated)",
}


def _convert_parameter_to_ax_config(
    param: HyperParameter,
) -> Union[RangeParameterConfig, ChoiceParameterConfig, None]:
    """
    Convert a sweeps HyperParameter to an Ax ParameterConfig.

    Args:
        param: HyperParameter to convert

    Returns:
        RangeParameterConfig for continuous/integer parameters,
        ChoiceParameterConfig for categorical parameters,
        or None for constants (which are not searchable)

    Raises:
        ValueError: If parameter type is not supported by ax method
    """
    # Skip constants - they are not part of the search space
    if param.type == HyperParameter.CONSTANT:
        return None

    # Check for unsupported types first
    if param.type in UNSUPPORTED_TYPES:
        raise ValueError(
            f"Parameter '{param.name}' uses distribution '{UNSUPPORTED_TYPES[param.type]}' "
            f"which is not supported by ax method. "
            f"Supported types: uniform, int_uniform, q_uniform, log_uniform_values, categorical. "
            f"Consider using method='bayes' for sklearn-based optimization with these distributions."
        )

    # Continuous parameters
    if param.type == HyperParameter.UNIFORM:
        return RangeParameterConfig(
            name=param.name,
            bounds=(param.config["min"], param.config["max"]),
            parameter_type="float",
        )

    # Integer parameters
    elif param.type == HyperParameter.INT_UNIFORM:
        return RangeParameterConfig(
            name=param.name,
            bounds=(param.config["min"], param.config["max"]),
            parameter_type="int",
        )

    # Quantized uniform (treat as float)
    elif param.type == HyperParameter.Q_UNIFORM:
        return RangeParameterConfig(
            name=param.name,
            bounds=(param.config["min"], param.config["max"]),
            parameter_type="float",
        )

    # Log-scale parameters
    elif param.type in [HyperParameter.LOG_UNIFORM_V2, HyperParameter.Q_LOG_UNIFORM_V2]:
        return RangeParameterConfig(
            name=param.name,
            bounds=(param.config["min"], param.config["max"]),
            parameter_type="float",
            scaling="log",
        )

    # Categorical parameters (both uniform and weighted)
    elif param.type in [HyperParameter.CATEGORICAL, HyperParameter.CATEGORICAL_PROB]:
        # Note: Ax treats all categorical parameters uniformly
        # If probabilities are specified in the config, they are ignored
        # Determine parameter type from first value
        values = param.config["values"]
        if len(values) == 0:
            raise ValueError(f"Parameter '{param.name}' has empty values list")

        # Infer parameter_type from the first value
        first_value = values[0]
        if isinstance(first_value, bool):
            param_type = "bool"
        elif isinstance(first_value, int):
            param_type = "int"
        elif isinstance(first_value, float):
            param_type = "float"
        elif isinstance(first_value, str):
            param_type = "str"
        else:
            param_type = "str"  # Default to string

        return ChoiceParameterConfig(
            name=param.name,
            values=values,
            parameter_type=param_type,
        )

    else:
        # Fallback for any other parameter type
        raise ValueError(
            f"Parameter '{param.name}' has unknown or unsupported type '{param.type}'. "
            f"Please use a supported distribution type for ax method."
        )


def _create_ax_client_from_config(
    config: Union[dict, SweepConfig],
    params: HyperParameterSet,
    random_seed: int = 42,
) -> Client:
    """
    Create and configure an Ax Client based on sweep config.

    Supports both single-objective optimization (using 'metric') and
    multi-objective optimization (using 'metrics').

    Args:
        config: Sweep configuration
        params: HyperParameterSet from config["parameters"]
        random_seed: Random seed for reproducibility

    Returns:
        Configured Client ready for trial generation

    Raises:
        ValueError: If no searchable parameters are found
    """
    # Convert all searchable parameters to Ax configs
    ax_parameters = []
    for param in params.searchable_params:
        ax_param = _convert_parameter_to_ax_config(param)
        if ax_param is not None:  # Skip constants
            ax_parameters.append(ax_param)

    if len(ax_parameters) == 0:
        raise ValueError(
            "No searchable parameters found for Ax optimization. "
            "At least one non-constant parameter is required."
        )

    logger.debug(f"Converted {len(ax_parameters)} parameters to Ax format")

    # Create client
    client = Client(random_seed=random_seed)

    # Configure experiment
    client.configure_experiment(
        parameters=ax_parameters,
        name=config.get("name", "sweep"),
        description=config.get("description"),
    )

    # Configure optimization based on single vs multi-objective
    is_moo = _is_moo_config(config)

    if is_moo:
        # Multi-objective optimization
        outcome_constraints = config.get("metric_constraints")
        has_thresholds = any(m.get("threshold") is not None for m in config["metrics"])

        if has_thresholds:
            # Build full optimization config with thresholds using official Ax API
            from ax.core.metric import Metric
            from ax.core.objective import MultiObjective, Objective
            from ax.core.optimization_config import MultiObjectiveOptimizationConfig
            from ax.core.outcome_constraint import ObjectiveThreshold
            from ax.core.types import ComparisonOp
            from ax.api.utils.instantiation.from_string import parse_outcome_constraint

            # Build objectives and thresholds
            objectives = []
            objective_thresholds = []

            for m in config["metrics"]:
                metric = Metric(name=m["name"])
                minimize = m["goal"] == "minimize"
                objectives.append(Objective(metric=metric, minimize=minimize))

                threshold = m.get("threshold")
                if threshold is not None:
                    objective_thresholds.append(
                        ObjectiveThreshold(
                            metric=metric,
                            bound=threshold,
                            relative=False,
                            op=ComparisonOp.LEQ if minimize else ComparisonOp.GEQ,
                        )
                    )

            # Parse outcome constraints from strings
            parsed_constraints = []
            if outcome_constraints:
                for constraint_str in outcome_constraints:
                    parsed_constraints.append(parse_outcome_constraint(constraint_str))

            # Create optimization config with thresholds
            opt_config = MultiObjectiveOptimizationConfig(
                objective=MultiObjective(objectives=objectives),
                objective_thresholds=objective_thresholds,
                outcome_constraints=parsed_constraints,
            )
            client.set_optimization_config(opt_config)
        else:
            # No thresholds - use simple string-based API
            objective_parts = []
            for m in config["metrics"]:
                if m["goal"] == "minimize":
                    objective_parts.append(f"-{m['name']}")
                else:
                    objective_parts.append(m["name"])
            objective = ", ".join(objective_parts)

            client.configure_optimization(
                objective=objective,
                outcome_constraints=outcome_constraints,
            )

        logger.debug(
            f"Configured Ax experiment with MOO and "
            f"{len(outcome_constraints or [])} constraints"
        )
    else:
        # Single-objective optimization
        metric_name = config["metric"]["name"]
        goal = config["metric"]["goal"]

        # Ax uses objective string format:
        # - "-metric_name" for minimization (negative to convert max to min)
        # - "metric_name" for maximization
        if goal == "minimize":
            objective = f"-{metric_name}"
        else:  # maximize
            objective = metric_name

        # Get outcome constraints if specified (also supported for single-objective)
        outcome_constraints = config.get("metric_constraints")

        client.configure_optimization(
            objective=objective,
            outcome_constraints=outcome_constraints,
        )

        logger.debug(f"Configured Ax experiment with objective: {objective}")

    # Configure generation strategy (default to "fast")
    # "fast" uses Bayesian optimization with good defaults
    client.configure_generation_strategy(
        method="fast",
        initialization_random_seed=random_seed,
        initialize_with_center=False,
    )

    return client


def _extract_parameters_from_run(run: SweepRun, params: HyperParameterSet) -> dict:
    """
    Extract parameter values from a run's config.

    Args:
        run: SweepRun with config
        params: HyperParameterSet for parameter mapping

    Returns:
        Dict mapping parameter names to values
        Example: {"learning_rate": 0.001, "batch_size": 32}

    Raises:
        ValueError: If required parameters are missing from run config
    """
    parameters = {}

    for param in params.searchable_params:
        value = params._get_val_from_config(run.config, param.name)

        if value is None:
            raise ValueError(
                f"Parameter '{param.name}' not found in run config. "
                f"Run may be from a different sweep configuration."
            )

        parameters[param.name] = value

    return parameters


def _extract_metric_from_run(run: SweepRun, metric_name: str) -> float:
    """
    Extract the final metric value from a completed run.

    Uses summary_metric to get the final (summary) metric value.

    Args:
        run: SweepRun with metric data
        metric_name: Name of the metric

    Returns:
        Metric value (float)

    Raises:
        ValueError: If metric cannot be extracted
    """
    try:
        return run.summary_metric(metric_name)
    except (KeyError, ValueError) as e:
        raise ValueError(f"Cannot extract metric '{metric_name}' from run: {e}")


def _extract_latest_metric_from_run(run: SweepRun, metric_name: str) -> float:
    """
    Extract the latest metric value from a running trial.

    Args:
        run: SweepRun with history
        metric_name: Name of the metric

    Returns:
        Latest metric value (float)

    Raises:
        ValueError: If no metric data available
    """
    history = run.metric_history(metric_name, filter_invalid=True)

    if len(history) == 0:
        raise ValueError(f"No metric data available for '{metric_name}' in run history")

    return history[-1]


def _extract_metrics_from_run(
    run: SweepRun, metric_names: List[str]
) -> dict:
    """
    Extract multiple metric values from a completed run.

    Uses summary_metric to get the final (summary) metric value for each metric.

    Args:
        run: SweepRun with metric data
        metric_names: List of metric names to extract

    Returns:
        Dict mapping metric name to value

    Raises:
        ValueError: If any metric cannot be extracted
    """
    metrics = {}
    missing = []

    for metric_name in metric_names:
        try:
            metrics[metric_name] = run.summary_metric(metric_name)
        except (KeyError, ValueError):
            missing.append(metric_name)

    if missing:
        raise ValueError(f"Cannot extract metrics {missing} from run")

    return metrics


def _extract_latest_metrics_from_run(
    run: SweepRun, metric_names: List[str]
) -> dict:
    """
    Extract the latest values for multiple metrics from a running trial.

    Args:
        run: SweepRun with history
        metric_names: List of metric names to extract

    Returns:
        Dict mapping metric name to latest value (only includes metrics with data)
    """
    metrics = {}

    for metric_name in metric_names:
        try:
            history = run.metric_history(metric_name, filter_invalid=True)
            if len(history) > 0:
                metrics[metric_name] = history[-1]
        except (KeyError, ValueError):
            pass  # Skip metrics without data

    return metrics


def _attach_historical_trials_to_client(
    client: Client,
    runs: List[SweepRun],
    params: HyperParameterSet,
    metric_names: List[str],
) -> dict:
    """
    Attach historical trial data from runs to the Ax Client using native Ax trial statuses.

    This function leverages Ax's sophisticated trial status management:
    - COMPLETED: Trials with valid metric data that Ax learns from
    - FAILED: Crashed/failed runs that teach Ax what regions to avoid
    - RUNNING: Currently executing trials treated as pending observations
    - ABANDONED: Trials that were stopped before completion

    No manual imputation is needed - Ax handles all trial states natively.

    Trial Status Mapping:
    - RunState.finished → attach_data + complete_trial (auto COMPLETED/FAILED)
    - RunState.failed/crashed/killed → mark_trial_failed()
    - RunState.running → attach_data if available, else left as RUNNING
    - RunState.pending/preempting/preempted → mark_trial_abandoned()

    Args:
        client: Configured Ax Client
        runs: Historical sweep runs
        params: HyperParameterSet for parameter extraction
        metric_names: List of metric names to extract (single for SOO, multiple for MOO)

    Returns:
        Dict with statistics: {
            "completed": int,
            "failed": int,
            "running": int,
            "abandoned": int,
            "skipped": int
        }
    """
    stats = {
        "completed": 0,
        "failed": 0,
        "running": 0,
        "abandoned": 0,
        "skipped": 0,
    }

    is_moo = len(metric_names) > 1

    for run in runs:
        # Extract parameters from run config
        try:
            parameters = _extract_parameters_from_run(run, params)
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping run with invalid parameters: {e}")
            stats["skipped"] += 1
            continue

        # Attach trial to Ax experiment to get trial_index
        trial_index = client.attach_trial(parameters=parameters)

        # Handle based on run state using Ax-native trial status
        if run.state == RunState.finished:
            # Finished run - try to extract metric(s) and complete
            try:
                if is_moo:
                    # Multi-objective: extract all metrics
                    metric_values = _extract_metrics_from_run(run, metric_names)
                else:
                    # Single-objective: extract single metric
                    metric_value = _extract_metric_from_run(run, metric_names[0])
                    metric_values = {metric_names[0]: metric_value}

                # Attach data and complete trial
                # complete_trial will automatically mark as COMPLETED or FAILED
                # based on whether metric data is present
                client.complete_trial(trial_index=trial_index, raw_data=metric_values)
                stats["completed"] += 1

            except ValueError as e:
                # No valid metric - mark as failed
                logger.warning(
                    f"Trial {trial_index} missing metric(s), marking as FAILED: {e}"
                )
                client.mark_trial_failed(
                    trial_index=trial_index, failed_reason=f"Missing metric: {e}"
                )
                stats["failed"] += 1

        elif run.state in [RunState.failed, RunState.crashed, RunState.killed]:
            # Failed/crashed run - mark as FAILED
            # Ax will learn to avoid similar parameter configurations
            client.mark_trial_failed(
                trial_index=trial_index, failed_reason=f"Run {run.state}"
            )
            stats["failed"] += 1

        elif run.state == RunState.running:
            # Running trial - try to attach partial data if available
            try:
                if is_moo:
                    # Multi-objective: get latest values for all available metrics
                    metric_values = _extract_latest_metrics_from_run(run, metric_names)
                    if metric_values:
                        client.attach_data(
                            trial_index=trial_index, raw_data=metric_values
                        )
                        logger.debug(
                            f"Attached partial data for running trial {trial_index}"
                        )
                    else:
                        logger.debug(
                            f"Trial {trial_index} is running but has no data yet"
                        )
                else:
                    # Single-objective: get latest metric value if available
                    metric_value = _extract_latest_metric_from_run(
                        run, metric_names[0]
                    )
                    client.attach_data(
                        trial_index=trial_index,
                        raw_data={metric_names[0]: metric_value},
                    )
                    logger.debug(
                        f"Attached partial data for running trial {trial_index}"
                    )
            except ValueError:
                # No data yet - leave trial as RUNNING
                # Ax will treat this as a pending observation
                logger.debug(f"Trial {trial_index} is running but has no data yet")

            stats["running"] += 1

        else:  # pending, preempting, preempted
            # These trials haven't started or were preempted - mark as abandoned
            # Ax will exclude them from optimization
            client.mark_trial_abandoned(trial_index=trial_index)
            stats["abandoned"] += 1

    logger.debug(
        f"Attached trials - Completed: {stats['completed']}, "
        f"Failed: {stats['failed']}, Running: {stats['running']}, "
        f"Abandoned: {stats['abandoned']}, Skipped: {stats['skipped']}"
    )

    return stats


def _ax_params_to_sweep_config(ax_params: dict, params: HyperParameterSet) -> dict:
    """
    Convert Ax parameterization dict to sweeps config format.

    Ax provides: {"param1": 0.5, "param2": 10, "param3": "choice_a"}
    Sweeps expects: {"param1": {"value": 0.5}, "param2": {"value": 10}, ...}

    Also handles:
    - Type conversions (ensure int params are ints)
    - Nested parameters (using NESTING_DELIMITER)
    - Constants (added from original params)

    Args:
        ax_params: Dict from Ax (parameter name -> value)
        params: Original HyperParameterSet for type info and constants

    Returns:
        Sweeps config dict format with "value" keys
    """
    # Set parameter values from Ax suggestion
    for param in params:
        if param.type == HyperParameter.CONSTANT:
            # Constants keep their original value
            continue
        elif param.name in ax_params:
            # Get value from Ax suggestion
            value = ax_params[param.name]

            # Type conversion for integer parameters
            if param.type == HyperParameter.INT_UNIFORM:
                value = int(value)

            # Set the value on the parameter object
            param.value = value
        else:
            logger.warning(
                f"Parameter '{param.name}' not in Ax suggestion, "
                f"using default or previous value"
            )

    # Convert to sweeps config format (with nested "value" keys)
    return params.to_config()


def _is_moo_config(config: dict) -> bool:
    """Check if config is for multi-objective optimization.

    Args:
        config: Sweep configuration dict

    Returns:
        True if config uses 'metrics' (MOO), False if uses 'metric' (single-objective)
    """
    return "metrics" in config


def _parse_constraint_metric_names(metric_constraints: List[str]) -> List[str]:
    """Extract metric names from constraint expressions.

    Ax only supports <= and >= operators for outcome constraints.

    Args:
        metric_constraints: List of constraint strings like "g1 <= 0", "accuracy >= 0.9"

    Returns:
        List of metric names extracted from constraints

    Raises:
        ValueError: If constraint uses unsupported operator (<, >, ==)

    Examples:
        >>> _parse_constraint_metric_names(["g1 <= 0", "g2 <= 0"])
        ['g1', 'g2']
        >>> _parse_constraint_metric_names(["loss <= 1.0", "accuracy >= 0.9"])
        ['loss', 'accuracy']
    """
    metric_names = []
    for constraint in metric_constraints:
        # Ax only supports <= and >= operators
        if "<=" in constraint:
            metric_name = constraint.split("<=")[0].strip()
            if metric_name:
                metric_names.append(metric_name)
        elif ">=" in constraint:
            metric_name = constraint.split(">=")[0].strip()
            if metric_name:
                metric_names.append(metric_name)
        else:
            raise ValueError(
                f"Invalid constraint format: '{constraint}'. "
                f"Ax only supports '<=' and '>=' operators. "
                f"Use format like 'metric_name <= value' or 'metric_name >= value'."
            )
    return metric_names


def _validate_config(config: dict) -> None:
    """Validate sweep config for ax method.

    Checks:
    - method == "ax"
    - Either "metric" (single-objective) OR "metrics" (multi-objective) exists
    - "parameters" section exists and is non-empty
    - early_terminate is not used with multi-objective optimization
    - metric_constraints format is valid

    Args:
        config: Sweep configuration dict

    Raises:
        ValueError: With descriptive message if validation fails
    """
    if "method" not in config:
        raise ValueError("Sweep config must contain 'method' section")

    if config["method"] != "ax":
        raise ValueError(
            f"Invalid sweep configuration for Ax search. "
            f"Expected method='ax', got method='{config['method']}'"
        )

    # Check mutual exclusion of metric and metrics
    has_metric = "metric" in config
    has_metrics = "metrics" in config

    if has_metric and has_metrics:
        raise ValueError(
            "Cannot specify both 'metric' (single-objective) and 'metrics' "
            "(multi-objective) in the same config."
        )

    if not has_metric and not has_metrics:
        raise ValueError(
            'Ax Bayesian search requires either "metric" (single-objective) '
            'or "metrics" (multi-objective) section in config'
        )

    # Validate single-objective metric
    if has_metric:
        if "name" not in config["metric"]:
            raise ValueError('Metric section must contain "name" field')

        if "goal" not in config["metric"]:
            raise ValueError(
                'Metric section must contain "goal" field (minimize or maximize)'
            )

        if config["metric"]["goal"] not in ["minimize", "maximize"]:
            raise ValueError(
                f"Metric goal must be 'minimize' or 'maximize', "
                f"got '{config['metric']['goal']}'"
            )

    # Validate multi-objective metrics
    if has_metrics:
        metrics = config["metrics"]
        if not isinstance(metrics, list):
            raise ValueError(f"metrics must be a list, got {type(metrics)}")

        if len(metrics) < 2:
            raise ValueError(
                "metrics must contain at least 2 objectives for multi-objective optimization"
            )

        for i, m in enumerate(metrics):
            if not isinstance(m, dict):
                raise ValueError(f"metrics[{i}] must be a dict")
            if "name" not in m:
                raise ValueError(f'metrics[{i}] must contain "name" field')
            if "goal" not in m:
                raise ValueError(f'metrics[{i}] must contain "goal" field')
            if m["goal"] not in ["minimize", "maximize"]:
                raise ValueError(
                    f"metrics[{i}]['goal'] must be 'minimize' or 'maximize', "
                    f"got '{m['goal']}'"
                )

        # Check for early_terminate + MOO conflict
        if "early_terminate" in config:
            raise ValueError(
                "early_terminate is not supported with multi-objective optimization. "
                "Please remove early_terminate or use single-objective optimization."
            )

    if "parameters" not in config:
        raise ValueError('Ax Bayesian search requires "parameters" section in config')

    if not isinstance(config["parameters"], dict) or len(config["parameters"]) == 0:
        raise ValueError("Parameters section must be a non-empty dict")


def ax_search_next_runs(
    runs: List[SweepRun],
    config: Union[dict, SweepConfig],
    validate: bool = False,
    n: int = 1,
    random_seed: int = 42,
    **kwargs,
) -> List[SweepRun]:
    """
    Suggest runs using ax-platform Bayesian optimization.

    This implementation fully leverages Ax's native capabilities:
    - Sophisticated trial status management (COMPLETED, FAILED, RUNNING, ABANDONED)
    - No manual imputation needed - Ax handles incomplete data natively
    - Failed trials teach the model what parameter regions to avoid
    - Running trials contribute as pending observations
    - Advanced Bayesian optimization strategies from Meta Research
    - Multi-objective optimization with Pareto frontier support

    Main workflow:
    1. Validate config and extract parameters
    2. Create Ax Client with experiment configuration
    3. Attach historical trial data with native Ax status handling
    4. Generate n new trials using Ax's optimization
    5. Convert Ax parameterizations to SweepRun format
    6. For MOO, include Pareto frontier in search_info
    7. Return list of SweepRun objects

    Args:
        runs: List of existing runs in the sweep
        config: Sweep configuration dict or SweepConfig
        validate: Whether to validate config against schema
        n: Number of new runs to generate
        random_seed: Random seed for reproducibility
        **kwargs: Additional arguments (reserved for future extensions)

    Returns:
        List of n SweepRun objects with suggested configurations.
        For multi-objective optimization, search_info includes 'pareto_frontier'.

    Raises:
        ValueError: For invalid config or unsupported parameter types
        ImportError: If ax-platform is not installed
        RuntimeError: If Ax optimization fails

    Example (single-objective):
        >>> config = {
        ...     'method': 'ax',
        ...     'parameters': {
        ...         'learning_rate': {'min': 0.001, 'max': 0.1, 'distribution': 'log_uniform_values'},
        ...         'batch_size': {'min': 16, 'max': 128}
        ...     },
        ...     'metric': {'name': 'loss', 'goal': 'minimize'}
        ... }
        >>> suggestions = ax_search_next_runs([], config, n=5)
        >>> len(suggestions)
        5

    Example (multi-objective):
        >>> config = {
        ...     'method': 'ax',
        ...     'parameters': {'learning_rate': {'min': 0.001, 'max': 0.1}},
        ...     'metrics': [
        ...         {'name': 'accuracy', 'goal': 'maximize'},
        ...         {'name': 'latency', 'goal': 'minimize'}
        ...     ]
        ... }
        >>> suggestions = ax_search_next_runs([], config, n=3)
        >>> # After some runs complete, search_info will contain pareto_frontier
    """
    # 1. Validation
    if validate:
        config = SweepConfig(config)

    _validate_config(config)

    # 2. Extract configuration
    params = HyperParameterSet.from_config(config["parameters"])
    is_moo = _is_moo_config(config)

    # Get metric names based on single vs multi-objective
    if is_moo:
        metric_names = [m["name"] for m in config["metrics"]]
    else:
        metric_names = [config["metric"]["name"]]

    # Also include constraint metric names if present
    # Ax expects all metrics referenced in constraints to be reported
    if "metric_constraints" in config and config["metric_constraints"]:
        constraint_metric_names = _parse_constraint_metric_names(
            config["metric_constraints"]
        )
        # Add constraint metrics that aren't already in metric_names
        for name in constraint_metric_names:
            if name not in metric_names:
                metric_names.append(name)

    if len(params.searchable_params) == 0:
        raise ValueError(
            "No searchable parameters found. "
            "At least one non-constant parameter is required for Ax."
        )

    mode_str = "multi-objective" if is_moo else "single-objective"
    logger.info(
        f"Starting Ax {mode_str} optimization for {len(params.searchable_params)} "
        f"searchable parameters, {len(runs)} historical runs"
    )

    # 3. Create Ax Client
    try:
        client = _create_ax_client_from_config(config, params, random_seed=random_seed)
    except Exception as e:
        raise RuntimeError(f"Failed to create Ax client: {e}") from e

    # 4. Attach historical data using Ax-native trial status
    if len(runs) > 0:
        # Suppress Ax's verbose info logging during bulk trial operations
        with _suppress_ax_logging():
            stats = _attach_historical_trials_to_client(
                client, runs, params, metric_names
            )
        logger.info(f"Trial attachment statistics: {stats}")
    else:
        logger.info("No historical runs - Ax will use initialization strategy")

    # 5. Generate new trials
    logger.info(f"Generating {n} new trial suggestions")

    try:
        next_parameterizations = client.get_next_trials(max_trials=n)
    except Exception as e:
        raise RuntimeError(
            f"Ax optimization failed: {e}. "
            "This may occur if the search space is fully explored, "
            "if there are no valid trials to learn from, or "
            "if the optimization cannot generate valid candidates. "
            "Consider checking your parameter ranges and metric data."
        ) from e

    # 6. Get Pareto frontier for MOO (if we have completed trials)
    pareto_frontier = None
    if is_moo and len(runs) > 0:
        try:
            # Get Pareto optimal points from the experiment
            # Returns list of tuples: (parameters, metrics, trial_index, arm_name)
            pareto_results = client.get_pareto_frontier(use_model_predictions=False)
            if pareto_results:
                pareto_frontier = []
                for params_dict, metrics_dict, trial_index, arm_name in pareto_results:
                    pareto_frontier.append({
                        "trial_index": trial_index,
                        "arm_name": arm_name,
                        "parameters": dict(params_dict),
                        "metrics": dict(metrics_dict),
                    })
                logger.debug(f"Found {len(pareto_frontier)} Pareto optimal points")
        except Exception as e:
            # Pareto frontier retrieval is optional, don't fail on errors
            logger.warning(f"Could not retrieve Pareto frontier: {e}")

    # 7. Convert to SweepRun format
    suggested_runs = []

    for trial_index, ax_params in next_parameterizations.items():
        # Convert Ax parameterization to sweeps config format
        sweep_config = _ax_params_to_sweep_config(ax_params, params)

        # Add search metadata
        search_info = {
            "method": "ax",
            "ax_trial_index": trial_index,
        }

        # Add MOO-specific info
        if is_moo:
            search_info["is_multi_objective"] = True
            search_info["objective_names"] = metric_names
            if pareto_frontier is not None:
                search_info["pareto_frontier"] = pareto_frontier

        suggested_runs.append(SweepRun(config=sweep_config, search_info=search_info))

    logger.info(f"Successfully generated {len(suggested_runs)} trial suggestions")

    return suggested_runs
