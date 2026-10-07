from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .experiment import FilteredDataset, mae_by_state, rmse_by_state, run_filter
from .mismatch import MismatchDataset
from .state_space import StateSpaceConfig


@dataclass(frozen=True)
class RobustnessResult:
    scenario: str
    rmse_fatigue: float
    rmse_recovery: float
    rmse_load: float
    mae_fatigue: float
    mae_recovery: float
    mae_load: float
    mean_normalized_innovation: float


def _normalized_innovation(
    config: StateSpaceConfig,
    dataset: MismatchDataset,
    filtered: FilteredDataset,
) -> float:
    """Summarize normalized residual size for observed channels.

    Large values indicate that observations are systematically less
    compatible with the assumed model/noise configuration.
    """
    values = []
    for i, y in enumerate(dataset.observations):
        if not np.all(np.isfinite(y)):
            continue
        residual = y - config.C @ filtered.predicted_states[i]
        # Innovation covariance from the predicted covariance is not stored,
        # so use a dimension-normalized residual magnitude as a transparent
        # diagnostic rather than pretending it is a full likelihood score.
        scale = np.sqrt(np.diag(config.R))
        values.append(np.mean(np.abs(residual) / scale))
    return float(np.mean(values))


def evaluate_mismatch(
    config: StateSpaceConfig,
    dataset: MismatchDataset,
    warmup: int = 30,
) -> RobustnessResult:
    filtered = run_filter(
        config,
        type(
            "DatasetAdapter",
            (),
            {
                "inputs": dataset.inputs,
                "observations": dataset.observations,
            },
        )(),
    )

    rmse = rmse_by_state(
        dataset.true_states[warmup:],
        filtered.states[warmup:],
    )
    mae = mae_by_state(
        dataset.true_states[warmup:],
        filtered.states[warmup:],
    )

    return RobustnessResult(
        scenario=dataset.scenario,
        rmse_fatigue=float(rmse[0]),
        rmse_recovery=float(rmse[1]),
        rmse_load=float(rmse[2]),
        mae_fatigue=float(mae[0]),
        mae_recovery=float(mae[1]),
        mae_load=float(mae[2]),
        mean_normalized_innovation=_normalized_innovation(
            config, dataset, filtered
        ),
    )


def evaluate_all_mismatches(
    config: StateSpaceConfig,
    datasets: dict[str, MismatchDataset],
    warmup: int = 30,
) -> list[RobustnessResult]:
    return [
        evaluate_mismatch(config, dataset, warmup=warmup)
        for dataset in datasets.values()
    ]
