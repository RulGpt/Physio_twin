import numpy as np

from model.m3_experiment import (
    evaluate_persistence_model,
    evaluate_population_model,
    fit_and_evaluate_personalized_model,
    make_reference_config,
    run_single_experiment,
)


def make_data(days=40):
    rng = np.random.default_rng(42)

    inputs = np.column_stack(
        [
            rng.uniform(0.0, 1.5, days),
            rng.uniform(0.0, 3.0, days),
            rng.uniform(0.0, 1.0, days),
        ]
    )

    observations = np.column_stack(
        [
            rng.normal(1.0, 0.2, days),
            rng.normal(0.8, 0.2, days),
        ]
    )

    return inputs, observations


def test_persistence_test_period():

    _, observations = make_data()

    train = observations[:30]
    test = observations[30:]

    result = evaluate_persistence_model(
        train_observations=train,
        test_observations=test,
    )

    assert result.n_predictions == 10

    assert np.isfinite(
        result.overall_rmse
    )


def test_population_model_prediction():

    config = make_reference_config()

    inputs, observations = make_data()

    result = evaluate_population_model(
        config=config,
        train_inputs=inputs[:30],
        train_observations=observations[:30],
        test_inputs=inputs[30:],
        test_observations=observations[30:],
    )

    assert result.n_predictions == 10

    assert np.isfinite(
        result.overall_mae
    )

    assert np.isfinite(
        result.overall_rmse
    )


def test_personalized_model_prediction():

    config = make_reference_config()

    inputs, observations = make_data()

    result = fit_and_evaluate_personalized_model(
        initial_config=config,
        train_inputs=inputs[:30],
        train_observations=observations[:30],
        test_inputs=inputs[30:],
        test_observations=observations[30:],
        maxiter=20,
    )

    assert result.n_predictions == 10

    assert np.isfinite(
        result.overall_mae
    )

    assert np.isfinite(
        result.overall_rmse
    )


def test_complete_m3_experiment():

    config = make_reference_config()

    rows = run_single_experiment(
        config=config,
        days=40,
        seed=42,
        train_ratio=0.75,
        maxiter=20,
    )

    assert len(rows) == 3

    models = {
        row["model"]
        for row in rows
    }

    assert models == {
        "persistence",
        "population",
        "personalized",
    }

    for row in rows:

        assert row["n_predictions"] == 10

        assert np.isfinite(
            row["overall_mae"]
        )

        assert np.isfinite(
            row["overall_rmse"]
        )