from copy import deepcopy
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats as scipy_stats
from sklearn import gaussian_process as sklearn_gaussian

from ._types import ArrayLike, floating, integer
from .config.cfg import SweepConfig
from .config.schema import fill_validate_metric
from .params import HyperParameter, HyperParameterSet
from .run import RunState, SweepRun, is_number, run_state_is_terminal

GAUSSIAN_PROCESS_NUGGET = 1e-7
STD_NUMERICAL_STABILITY_EPSILON = 1e-6


class ImputeStrategy(str, Enum):
    best = "best"
    worst = "worst"
    latest = "latest"


def bayes_baseline_validate_and_fill(config: Dict) -> Dict:
    config = deepcopy(config)

    if "metric" not in config:
        raise ValueError('Bayesian search requires "metric" section')

    if config["method"] != "bayes":
        raise ValueError("Invalid sweep configuration for bayes_search_next_run.")

    config = fill_validate_metric(config)

    return config


def fit_normalized_gaussian_process(
    X: ArrayLike, y: ArrayLike, nu: floating = 1.5
) -> Tuple[sklearn_gaussian.GaussianProcessRegressor, floating, floating]:
    gp = sklearn_gaussian.GaussianProcessRegressor(
        kernel=sklearn_gaussian.kernels.Matern(nu=nu),
        n_restarts_optimizer=2,
        alpha=GAUSSIAN_PROCESS_NUGGET,
        random_state=2,
    )

    y_stddev: ArrayLike
    if len(y) == 1:
        y = np.array(y)
        y_mean = y[0]
        y_stddev = 1.0
    else:
        y_mean = np.mean(y)
        y_stddev = np.std(y) + STD_NUMERICAL_STABILITY_EPSILON
    y_norm = (y - y_mean) / y_stddev
    gp.fit(X, y_norm)
    return gp, y_mean, y_stddev


def sigmoid(x: ArrayLike) -> ArrayLike:
    return np.exp(-np.logaddexp(0, -x))


def random_sample(X_bounds: ArrayLike, num_test_samples: integer) -> ArrayLike:
    num_hyperparameters = len(X_bounds)
    test_X = np.empty((int(num_test_samples), num_hyperparameters))
    for ii in range(num_test_samples):
        for jj in range(num_hyperparameters):
            if type(X_bounds[jj][0]) == int:
                assert type(X_bounds[jj][1]) == int
                test_X[ii, jj] = np.random.randint(X_bounds[jj][0], X_bounds[jj][1])
            else:
                test_X[ii, jj] = (
                    np.random.uniform() * (X_bounds[jj][1] - X_bounds[jj][0])
                    + X_bounds[jj][0]
                )
    return test_X


def train_gaussian_process(
    sample_X: ArrayLike,
    sample_y: ArrayLike,
    X_bounds: Optional[ArrayLike] = None,
    current_X: ArrayLike = None,
    nu: floating = 1.5,
    max_samples: integer = 100,
) -> Tuple[sklearn_gaussian.GaussianProcessRegressor, floating, floating]:
    """Trains a Gaussian Process function from sample_X, sample_y data.

    Handles the case where there are other training runs in flight (current_X)

    Arguments:
        sample_X: vector of already evaluated sets of hyperparameters
        sample_y: vector of already evaluated loss function values
        X_bounds: minimum and maximum values for every dimension of X
        current_X: hyperparameters currently being explored
        nu: input to the Matern function, higher numbers make it smoother 0.5, 1.5, 2.5 are good values
         see http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html

    Returns:
        gp: the gaussian process function
        y_mean: mean
        y_stddev: stddev

        To make a prediction with gp on real world data X, need to call:
        (gp.predict(X) * y_stddev) + y_mean
    """
    if current_X is not None:
        current_X = np.array(current_X)
        if len(current_X.shape) != 2:
            raise ValueError("Current X must be a 2 dimensional array")

        # we can't let the current samples be bigger than max samples
        # because we need to use some real samples to build the curve
        if current_X.shape[0] > max_samples - 5:
            print(
                "current_X is bigger than max samples - 5 so dropping some currently running parameters"
            )
            current_X = current_X[: (max_samples - 5), :]  # type: ignore
    if len(sample_y.shape) != 1:
        raise ValueError("Sample y must be a 1 dimensional array")

    if sample_X.shape[0] != sample_y.shape[0]:
        raise ValueError(
            "Sample X and sample y must be the same size {} {}".format(
                sample_X.shape[0], sample_y.shape[0]
            )
        )

    if X_bounds is not None and sample_X.shape[1] != len(X_bounds):
        raise ValueError(
            "Bounds must be the same length as Sample X's second dimension"
        )

    # gaussian process takes a long time to train, so if there's more than max_samples
    # we need to sample from it
    if sample_X.shape[0] > max_samples:
        sample_indices = np.random.randint(sample_X.shape[0], size=max_samples)
        X = sample_X[sample_indices]
        y = sample_y[sample_indices]
    else:
        X = sample_X
        y = sample_y
    gp, y_mean, y_stddev = fit_normalized_gaussian_process(X, y, nu=nu)
    if current_X is not None:
        # if we have some hyperparameters running, we pretend that they return
        # the prediction of the function we've fit
        X = np.append(X, current_X, axis=0)
        current_y_fantasy = (gp.predict(current_X) * y_stddev) + y_mean
        y = np.append(y, current_y_fantasy)
        gp, y_mean, y_stddev = fit_normalized_gaussian_process(X, y, nu=nu)
    return gp, y_mean, y_stddev


def filter_nans(sample_X: ArrayLike, sample_y: ArrayLike) -> ArrayLike:
    is_row_finite = ~(np.isnan(sample_X).any(axis=1) | np.isnan(sample_y))
    sample_X = sample_X[is_row_finite, :]
    sample_y = sample_y[is_row_finite]
    return sample_X, sample_y


def next_sample(
    *,
    sample_X: ArrayLike,
    sample_y: ArrayLike,
    X_bounds: Optional[ArrayLike] = None,
    current_X: Optional[ArrayLike] = None,
    nu: floating = 1.5,
    max_samples_for_gp: integer = 100,
    improvement: floating = 0.01,
    num_points_to_try: integer = 1000,
    opt_func: str = "expected_improvement",
    test_X: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, floating, floating, floating, floating, str]:
    """Calculates the best next sample to look at via bayesian optimization.

    Args:
        sample_X: ArrayLike, shape (N_runs, N_params)
            2d array of already evaluated sets of hyperparameters
        sample_y: ArrayLike, shape (N_runs,)
            1d array of already evaluated loss function values
        X_bounds: ArrayLike, optional, shape (N_params, 2), default None
            2d array minimum and maximum values for every dimension of X
        current_X: ArrayLike, optional, shape (N_runs_in_flight, N_params), default None
            hyperparameters currently being explored
        nu: floating, optional, default = 1.5
            input to the Matern function, higher numbers make it smoother. 0.5,
            1.5, 2.5 are good values  see

               http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html

        max_samples_for_gp: integer, optional, default 100
            maximum samples to consider (since algo is O(n^3)) for performance,
            but also adds some randomness. this number of samples will be chosen
            randomly from the sample_X and used to train the GP.
        improvement: floating, optional, default 0.1
            amount of improvement to optimize for -- higher means take more exploratory risks
        num_points_to_try: integer, optional, default 1000
            number of X values to try when looking for value with highest expected probability
            of improvement
        opt_func: one of {"expected_improvement", "prob_of_improvement"} - whether to optimize expected
                improvement of probability of improvement.  Expected improvement is generally better - may want
                to remove probability of improvement at some point.  (But I think prboability of improvement
                is a little easier to calculate)
        test_X: X values to test when looking for the best values to try

    Returns:
        suggested_X: optimal X value to try
        prob_of_improvement: probability of an improvement
        predicted_y: predicted value
        predicted_std: stddev of predicted value
        expected_improvement: expected improvement
        warnings: warnings encountered
    """
    # Sanity check the data
    sample_X = np.array(sample_X)
    sample_y = np.array(sample_y)
    if test_X is not None:
        test_X = np.array(test_X)
    if len(sample_X.shape) != 2:
        raise ValueError("Sample X must be a 2 dimensional array")

    if len(sample_y.shape) != 1:
        raise ValueError("Sample y must be a 1 dimensional array")

    if sample_X.shape[0] != sample_y.shape[0]:
        raise ValueError("Sample X and y must be same length")

    if test_X is not None:
        # if test_X is set, usually this is for simulation/testing
        if X_bounds is not None:
            raise ValueError("Can't set test_X and X_bounds")

    else:
        # normal case where we randomly sample our test_X
        if X_bounds is None:
            raise ValueError("Must pass in test_X or X_bounds")

    filtered_X, filtered_y = filter_nans(sample_X, sample_y)

    # we can't run this algothim with less than two sample points, so we'll
    # just return a random point
    if filtered_X.shape[0] < 2:
        if test_X is not None:
            # pick a random row from test_X
            row = np.random.choice(test_X.shape[0])
            X = test_X[row, :]
        else:
            X = random_sample(X_bounds, 1)[0]
        if filtered_X.shape[0] < 1:
            prediction = 0.0
        else:
            prediction = filtered_y[0]
        return (
            X,
            1.0,
            prediction,
            np.nan,
            np.nan,
            "",
        )

    # build the acquisition function
    gp, y_mean, y_stddev, = train_gaussian_process(
        filtered_X, filtered_y, X_bounds, current_X, nu, max_samples_for_gp
    )
    # Look for the minimum value of our fitted-target-function + (kappa * fitted-target-std_dev)
    if test_X is None:  # this is the usual case
        test_X = random_sample(X_bounds, num_points_to_try)
    y_pred, y_pred_cov = gp.predict(test_X, return_cov=True)

    # HACK: Covariance matrix uses cholenky decomposition, so
    # use the diagonal of the covariance matrix to get the std_dev
    y_pred_std = np.sqrt(np.diag(y_pred_cov))

    # best value of y we've seen so far.  i.e. y*
    min_unnorm_y = np.min(filtered_y)

    """
    if opt_func == "probability_of_improvement":
        min_norm_y = (min_unnorm_y - y_mean) / y_stddev - improvement
    else:
    """
    min_norm_y = (min_unnorm_y - y_mean) / y_stddev

    Z = -(y_pred - min_norm_y) / (y_pred_std + STD_NUMERICAL_STABILITY_EPSILON)
    prob_of_improve: np.ndarray = scipy_stats.norm.cdf(Z)
    e_i = -(y_pred - min_norm_y) * scipy_stats.norm.cdf(
        Z
    ) + y_pred_std * scipy_stats.norm.pdf(Z)

    """
    if opt_func == "probability_of_improvement":
        best_test_X_index = np.argmax(prob_of_improve)
    else:
    """

    best_test_X_index = np.argmax(e_i)

    # Make sure Kernel is not too close to boundaries
    # from https://github.com/scikit-learn/scikit-learn/blob/baf0ea25d6dd034403370fea552b21a6776bef18/sklearn/gaussian_process/kernels.py#L411
    list_close = np.isclose(gp.kernel_.bounds, np.atleast_2d(gp.kernel_.theta).T)
    warnings = ""
    idx = 0
    for hyp in gp.kernel_.hyperparameters:
        if hyp.fixed:
            continue
        for _ in range(hyp.n_elements):
            if list_close[idx, 0] or list_close[idx, 1]:
                warnings = "\n Some dimmensions of kernel are close to their bounds (bad fit), the next sample will be a random sample within parameter space"
                best_test_X_index = np.random.randint(0, test_X.shape[0] - 1)
                break
            idx += 1

    suggested_X = test_X[best_test_X_index]
    suggested_X_prob_of_improvement = prob_of_improve[best_test_X_index]
    suggested_X_predicted_y = y_pred[best_test_X_index] * y_stddev + y_mean
    suggested_X_predicted_std = y_pred_std[best_test_X_index] * y_stddev

    # recalculate expected improvement
    min_norm_y = (min_unnorm_y - y_mean) / y_stddev
    z_best = -(y_pred[best_test_X_index] - min_norm_y) / (
        y_pred_std[best_test_X_index] + STD_NUMERICAL_STABILITY_EPSILON
    )
    suggested_X_expected_improvement = -(
        y_pred[best_test_X_index] - min_norm_y
    ) * scipy_stats.norm.cdf(z_best) + y_pred_std[
        best_test_X_index
    ] * scipy_stats.norm.pdf(
        z_best
    )

    return (
        suggested_X,
        suggested_X_prob_of_improvement,
        suggested_X_predicted_y,
        suggested_X_predicted_std,
        suggested_X_expected_improvement,
        warnings,
    )


def impute(
    goal: str,
    metric_name: str,
    impute_strategy: ImputeStrategy,
    run: Optional[SweepRun] = None,
    runs: Optional[List[SweepRun]] = None,
) -> floating:
    """Impute the value of a run's metric using a specified strategy."""
    failed_val = 0.0
    worst_func = min if goal == "maximize" else max
    if impute_strategy == ImputeStrategy.best:
        if run is None:
            raise ValueError("impute_strategy == best requires a nonnull run")
        try:
            return run.metric_extremum(
                metric_name, kind="minimum" if goal == "minimize" else "maximum"
            )
        except ValueError:
            return failed_val
    elif impute_strategy == ImputeStrategy.worst:
        # we calc the max metric to put as the metric for failed runs
        # so that our bayesian search stays away from them
        worst_metric: floating = np.inf if goal == "maximize" else -np.inf
        if runs is None:
            raise ValueError("impute_strategy == worst requires nonnull list of runs")
        for run in runs:
            if run_state_is_terminal(run.state):
                try:
                    run_extremum = run.metric_extremum(
                        metric_name, kind="minimum" if goal == "maximize" else "maximum"
                    )
                except ValueError:
                    continue  # exclude run from worst_run calculation
                worst_metric = worst_func(worst_metric, run_extremum)
        if not np.isfinite(worst_metric):
            return failed_val
        return worst_metric
    elif impute_strategy == ImputeStrategy.latest:
        if run is None:
            raise ValueError("impute_strategy == latest requires a nonnull run")
        history = run.metric_history(metric_name, filter_invalid=True)
        if len(history) == 0:
            return failed_val
        return history[-1]
    else:
        raise ValueError(f"invalid impute strategy: {impute_strategy}")


def _construct_gp_data(
    runs: List[SweepRun], config: Union[dict, SweepConfig]
) -> Tuple[HyperParameterSet, ArrayLike, ArrayLike, ArrayLike, str]:
    goal = config["metric"]["goal"]
    metric_name = config["metric"]["name"]
    impute_strategy = ImputeStrategy(config["metric"]["impute"])
    params = HyperParameterSet.from_config(config["parameters"])

    if len(params.searchable_params) == 0:
        raise ValueError("Need at least one searchable parameter for bayes search.")

    sample_X: ArrayLike = []
    current_X: ArrayLike = []
    y: ArrayLike = []

    X_norms = params.normalize_runs_as_array(runs)
    worst_metric = impute(goal, metric_name, ImputeStrategy.worst, runs=runs)
    for run, X_norm in zip(runs, X_norms):
        if run.state == RunState.finished:
            try:
                metric = run.metric_extremum(
                    metric_name, kind="maximum" if goal == "maximize" else "minimum"
                )
            except ValueError:
                if impute_strategy != "worst":
                    metric = impute(
                        goal, metric_name, impute_strategy, run=run, runs=runs
                    )  # default
                else:
                    metric = worst_metric
            y.append(metric)
            sample_X.append(X_norm)
        elif run.state in [RunState.failed, RunState.crashed, RunState.killed]:
            if impute_strategy != "worst":
                metric = impute(goal, metric_name, impute_strategy, run=run, runs=runs)
            else:
                metric = worst_metric
            y.append(metric)
            sample_X.append(X_norm)
        elif run.state in [
            RunState.running,
            RunState.preempting,
            RunState.preempted,
            RunState.pending,
        ]:
            # run is in progress
            # we wont use the metric, but we should pass it into our optimizer to
            # account for the fact that it is running
            current_X.append(X_norm)
        else:
            raise ValueError("Run is in unknown state")

    if len(sample_X) == 0:
        sample_X = np.empty([0, 0])
    else:
        sample_X = np.asarray(sample_X)

    if len(current_X) > 0:
        current_X = np.array(current_X)

    # impute bad metric values from y
    y = np.asarray(y)
    if len(y) > 0:
        y[~np.isfinite(y)] = worst_metric

    warnings = ""
    if len(y) == 0 or y[~np.asarray(list(map(is_number, y)))].size == len(y):
        warnings += "\nSweep has no valid samples of the metric."

    # next_sample is a minimizer, so if we are trying to
    # maximize, we need to negate y
    y *= -1 if goal == "maximize" else 1

    return params, sample_X, current_X, y, warnings


def bayes_search_next_run(
    runs: List[SweepRun],
    config: Union[dict, SweepConfig],
    validate: bool = False,
    minimum_improvement: floating = 0.1,
) -> SweepRun:
    """Suggest runs using Bayesian optimization.

    >>> suggestion = bayes_search_next_run([], {
    ...    'method': 'bayes',
    ...    'parameters': {'a': {'min': 1., 'max': 2.}},
    ...    'metric': {'name': 'loss', 'goal': 'maximize'}
    ... })

    Args:
        runs: The runs in the sweep.
        config: The sweep's config.
        minimum_improvement: The minimium improvement to optimize for. Higher means take more exploratory risks.
        validate: Whether to validate `sweep_config` against the SweepConfig JSONschema.
           If true, will raise a Validation error if `sweep_config` does not conform to
           the schema. If false, will attempt to run the sweep with an unvalidated schema.

    Returns:
        The suggested run.
    """

    if validate:
        config = SweepConfig(config)

    config = bayes_baseline_validate_and_fill(config)

    params, sample_X, current_X, y, warnings_construct_gp_data = _construct_gp_data(
        runs, config
    )
    X_bounds = [[0.0, 1.0]] * len(params.searchable_params)

    (
        suggested_X,
        suggested_X_prob_of_improvement,
        suggested_X_predicted_y,
        suggested_X_predicted_std,
        suggested_X_expected_improvement,
        warnings_next_sample,
    ) = next_sample(
        sample_X=sample_X,
        sample_y=y,
        X_bounds=X_bounds,
        current_X=current_X if len(current_X) > 0 else None,
        improvement=minimum_improvement,
    )

    # convert the parameters from vector of [0,1] values
    # to the original ranges
    for param in params:
        if param.type == HyperParameter.CONSTANT:
            continue
        try_value = suggested_X[params.param_names_to_index[param.name]]
        param.value = param.ppf(try_value)

    ret_dict = params.to_config()
    info = {
        "success_probability": suggested_X_prob_of_improvement,
        "predicted_value": suggested_X_predicted_y,
        "predicted_value_std_dev": suggested_X_predicted_std,
        "expected_improvement": suggested_X_expected_improvement,
        "warnings": warnings_construct_gp_data + warnings_next_sample,
    }
    return SweepRun(config=ret_dict, search_info=info)


def bayes_search_next_runs(
    runs: List[SweepRun],
    config: Union[dict, SweepConfig],
    validate: bool = False,
    n: int = 1,
    minimum_improvement: floating = 0.1,
):
    ret: List[SweepRun] = []
    for _ in range(n):
        suggestion = bayes_search_next_run(
            runs + ret, config, validate, minimum_improvement
        )
        ret.append(suggestion)
    return ret
