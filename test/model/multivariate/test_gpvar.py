# First-party imports
import pytest
import mxnet as mx
from gluonts.distribution import LowrankMultivariateGaussian
from gluonts.distribution.lowrank_gp import LowrankGPOutput, GPArgProj
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.model.gpvar import GPVAREstimator
from gluonts.multivariate.datasets.dataset import multivariate_constant_dataset
from gluonts.trainer import Trainer

dataset = multivariate_constant_dataset()


def test_gp_output():
    # test that gp output gives expected shapes
    batch = 1
    hidden_size = 3
    dim = 4
    rank = 2

    states = mx.ndarray.ones(shape=(batch, dim, hidden_size))

    lowrank_gp_output = LowrankGPOutput(dim=dim, rank=rank)

    proj = lowrank_gp_output.get_args_proj()

    proj.initialize()

    distr_args = proj(states)

    mu, D, W = distr_args

    assert mu.shape == (batch, dim)
    assert D.shape == (batch, dim)
    assert W.shape == (batch, dim, rank)

    print(distr_args)


def test_gpvar_proj():
    # test that gp proj gives expected shapes
    batch = 1
    hidden_size = 3
    dim = 4
    rank = 2

    states = mx.ndarray.ones(shape=(batch, dim, hidden_size))

    gp_proj = GPArgProj(rank=rank)
    gp_proj.initialize()

    distr_args = gp_proj(states)

    mu, D, W = distr_args

    assert mu.shape == (batch, dim)
    assert D.shape == (batch, dim)
    assert W.shape == (batch, dim, rank)

    print(distr_args)

    distr = LowrankMultivariateGaussian(rank, *distr_args, dim)

    assert distr.mean.shape == (batch, dim)


@pytest.mark.parametrize("hybridize", [True, False])
@pytest.mark.parametrize("target_dim_sample", [None, 2])
@pytest.mark.parametrize("use_copula", [True, False])
def test_smoke(hybridize: bool, target_dim_sample: int, use_copula: bool):
    num_batches_per_epoch = 1
    estimator = GPVAREstimator(
        distr_output=LowrankGPOutput(rank=2),
        num_cells=1,
        num_layers=1,
        pick_incomplete=True,
        prediction_length=dataset.prediction_length,
        target_dim=dataset.target_dim,
        target_dim_sample=target_dim_sample,
        freq=dataset.freq,
        use_copula=use_copula,
        trainer=Trainer(
            epochs=2,
            batch_size=8,
            learning_rate=1e-4,
            num_batches_per_epoch=num_batches_per_epoch,
            hybridize=hybridize,
        ),
    )

    print(estimator)

    agg_metrics, _ = backtest_metrics(
        train_dataset=dataset.train_ds,
        test_dataset=dataset.test_ds,
        forecaster=estimator,
        num_eval_samples=10,
    )

    # todo relatively large value for now as datasets is slow without linear lag skip-connection
    assert agg_metrics['ND'] < 2.0
