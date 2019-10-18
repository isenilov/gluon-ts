# First-party imports
import pytest
import tempfile

from gluonts.model.estimator import TrainOutput
from gluonts.multivariate.datasets.dataset import multivariate_constant_dataset
from gluonts.multivariate.hyperparams import FastHyperparams
from gluonts.multivariate.multivariate_models import models_dict
from gluonts.multivariate.sagemaker_utils._entrypoint import (
    backtest_multivariate_metrics,
)

dataset = multivariate_constant_dataset()

models = models_dict.keys()


@pytest.mark.timeout(30)
@pytest.mark.parametrize("model_name", models_dict.keys())
def test_benchmark_models(model_name):

    estimator = models_dict[model_name](
        freq=dataset.freq,
        prediction_length=dataset.prediction_length,
        params=FastHyperparams(epochs=1, num_batches_per_epoch=1),
        target_dim=dataset.target_dim,
    )

    agg_metrics, _, train_output = backtest_multivariate_metrics(
        train_dataset=dataset.train_ds,
        test_dataset=dataset.test_ds,
        forecaster=estimator,
        num_eval_samples=10,
    )

    with tempfile.TemporaryDirectory() as tmp:
        train_output.serialize(tmp)
        deserialized_model = TrainOutput.deserialize(tmp)
        distr = deserialized_model.trained_net.distribution

        mu = distr.mean[0].asnumpy()
        Sigma = distr.sample_variance(num_samples=5).asnumpy()

        print(Sigma.mean())

    nd = 10.0
    # nd = 1.0

    assert agg_metrics['ND'] < nd
