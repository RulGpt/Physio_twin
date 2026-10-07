"""
PHYSIO-TWIN
-----------
Personal baseline estimation.

This module estimates an individual's normal historical pattern
from longitudinal observations.

It does not make a medical assessment.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


EPSILON = 1e-8


@dataclass(frozen=True)
class BaselineStatistics:
    """
    Descriptive statistics for one longitudinal variable.
    """

    count: int
    mean: float
    standard_deviation: float
    median: float
    minimum: float
    maximum: float


def calculate_baseline(
    values: Iterable[float],
) -> BaselineStatistics:
    """
    Calculate descriptive statistics for a numeric history.

    Parameters
    ----------
    values:
        Historical observations for one variable.

    Returns
    -------
    BaselineStatistics
        Mean, standard deviation, median, minimum and maximum.

    Raises
    ------
    ValueError
        If no observations are provided.
    """

    values_array = np.asarray(
        list(values),
        dtype=float,
    )

    if values_array.size == 0:
        raise ValueError(
            "At least one observation is required."
        )

    if not np.all(np.isfinite(values_array)):
        raise ValueError(
            "Baseline values must contain only finite numbers."
        )

    return BaselineStatistics(
        count=int(values_array.size),
        mean=float(np.mean(values_array)),
        standard_deviation=float(
            np.std(values_array, ddof=1)
            if values_array.size > 1
            else 0.0
        ),
        median=float(np.median(values_array)),
        minimum=float(np.min(values_array)),
        maximum=float(np.max(values_array)),
    )


def calculate_z_score(
    value: float,
    baseline: BaselineStatistics,
) -> float:
    """
    Calculate deviation of a value from a personal baseline.

    Formula:

        z = (x - mean) / (std + epsilon)

    If the baseline contains only one observation, or has zero
    variance, the result is defined as 0.0.

    This avoids producing an artificial infinite deviation when
    there is insufficient baseline variability information.
    """

    if not np.isfinite(value):
        raise ValueError(
            "Value must be finite."
        )

    if baseline.count < 2:
        return 0.0

    if baseline.standard_deviation < EPSILON:
        return 0.0

    return (
        (value - baseline.mean)
        / (
            baseline.standard_deviation
            + EPSILON
        )
    )