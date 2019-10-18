import pytest
import numpy as np
import mxnet as mx

from gluonts.distribution import (
    LowrankMultivariateGaussianOutput,
    LowrankMultivariateGaussian,
)
from gluonts.distribution.lowrank_gp import LowrankGPOutput
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.model.gpvar import GPVAREstimator
from gluonts.multivariate.datasets.dataset import (
    make_dataset,
    MultivariateDatasetInfo,
)
from gluonts.trainer import Trainer
from gluonts.model.estimator import Estimator

target_dim = 3
rank = 2
prediction_length = 5
num_periods = 10000

np.random.seed(10)
mx.random.seed(10)


def make_artificial_distribution(dim=3, rank=2):
    np.random.seed(12)
    mu = np.arange(dim)
    D = np.eye(dim) * (np.arange(dim) / dim + 0.5)
    W = (np.arange(dim * rank) - dim * rank / 2).reshape((dim, rank))

    distr = LowrankMultivariateGaussian(
        mu=mx.nd.array([mu]),
        D=mx.nd.array([np.diag(D)]),
        W=mx.nd.array([W]),
        dim=dim,
        rank=rank,
    )
    return distr


def deepvar_estimator(
    use_copula: bool, ds: MultivariateDatasetInfo
) -> Estimator:
    return DeepVAREstimator(
        num_cells=10,
        num_layers=1,
        target_dim=ds.target_dim,
        prediction_length=ds.prediction_length,
        freq=ds.freq,
        distr_output=LowrankMultivariateGaussianOutput(
            dim=ds.target_dim, rank=rank
        ),
        scaling=False,
        use_copula=use_copula,
        trainer=Trainer(
            hybridize=True,
            epochs=epochs,
            learning_rate=0.05,
            num_batches_per_epoch=40,
        ),
    )


def gpvar_estimator(
    use_copula: bool, ds: MultivariateDatasetInfo
) -> Estimator:
    return GPVAREstimator(
        num_cells=10,
        num_layers=1,
        target_dim=ds.target_dim,
        prediction_length=ds.prediction_length,
        freq=ds.freq,
        scaling=False,
        distr_output=LowrankGPOutput(rank=rank),
        use_copula=use_copula,
        trainer=Trainer(
            hybridize=True,
            epochs=epochs,
            learning_rate=0.05,
            num_batches_per_epoch=40,
        ),
    )


distr = make_artificial_distribution(dim=target_dim, rank=rank)
values = distr.sample(num_samples=num_periods).squeeze().asnumpy().T
dataset = make_dataset(values, prediction_length)
mu_true = distr.mu.asnumpy()[0]
Sigma_true = distr.variance.asnumpy()[0]

epochs = 40

estimators = [
    deepvar_estimator(use_copula=False, ds=dataset),
    deepvar_estimator(use_copula=True, ds=dataset),
    gpvar_estimator(use_copula=False, ds=dataset),
    gpvar_estimator(use_copula=True, ds=dataset),
]


# TODO flaky test but not sure to fix it
@pytest.mark.timeout(200)
@pytest.mark.parametrize("estimator", estimators)
def test_covariance(estimator):
    predictor = estimator.train(dataset.train_ds)

    # (num_samples, target_dim, prediction_length)
    samples = list(predictor.predict(dataset.test_ds))[0].samples

    # (num_samples * prediction_length, target_dim)
    samples = samples.transpose((0, 2, 1)).reshape(-1, dataset.target_dim)

    Sigma_estimated = np.cov(samples.T)

    print(estimator)

    print(Sigma_estimated)

    print(Sigma_true)

    mae = np.abs(Sigma_estimated - Sigma_true) / np.abs(Sigma_true)

    print(f"\nmae: {mae} (mean {mae.mean()})")

    # tests whether the avg_mu is close to the expected values (-i)
    assert np.allclose(samples.mean(axis=0), mu_true, rtol=1.0, atol=1.0)

    # tests whether the variance is close to the expected true_covariance
    assert mae.mean() < 0.5
