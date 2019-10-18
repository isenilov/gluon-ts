# Standard library imports
import numpy as np

# First-party imports
import pytest

from gluonts.dataset.artificial import constant_dataset
from gluonts.distribution import (
    MultivariateGaussianOutput,
    LowrankMultivariateGaussianOutput,
)
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.multivariate.datasets.dataset import multivariate_constant_dataset
from gluonts.trainer import Trainer


dataset = multivariate_constant_dataset()

estimator = DeepVAREstimator


@pytest.mark.timeout(20000)
@pytest.mark.parametrize(
    "distr_output, num_batches_per_epoch, Estimator, hybridize, use_copula",
    [
        # TODO not supported for now
        # (GaussianOutput(dim=target_dim), 10, estimator, True),
        # (GaussianOutput(dim=target_dim), 10, estimator, False),
        (
            LowrankMultivariateGaussianOutput(dim=dataset.target_dim, rank=2),
            10,
            estimator,
            True,
            True,
        ),
        (
            LowrankMultivariateGaussianOutput(dim=dataset.target_dim, rank=2),
            10,
            estimator,
            False,
            False,
        ),
        (
            LowrankMultivariateGaussianOutput(dim=dataset.target_dim, rank=2),
            10,
            estimator,
            True,
            False,
        ),
        # fails with nan for now
        # (MultivariateGaussianOutput(dim=target_dim), 10, estimator, False),
        # (MultivariateGaussianOutput(dim=target_dim), 10, estimator, True),
    ],
)
def test_deepvar(
    distr_output, num_batches_per_epoch, Estimator, hybridize, use_copula
):

    estimator = Estimator(
        num_cells=20,
        num_layers=1,
        pick_incomplete=True,
        target_dim=dataset.target_dim,
        prediction_length=dataset.prediction_length,
        # target_dim=target_dim,
        freq=dataset.freq,
        distr_output=distr_output,
        scaling=False,
        use_copula=use_copula,
        trainer=Trainer(
            epochs=1,
            batch_size=8,
            learning_rate=1e-10,
            num_batches_per_epoch=num_batches_per_epoch,
            hybridize=hybridize,
        ),
    )

    agg_metrics, _ = backtest_metrics(
        train_dataset=dataset.train_ds,
        test_dataset=dataset.test_ds,
        forecaster=estimator,
        num_eval_samples=10,
    )

    # todo relatively large value for now as datasets is slow without linear lag skip-connection
    assert agg_metrics['ND'] < 5.0
