import pytest

from spark.config.config_set import UniversalConfigSet
from spark.model.gaussian_model import GaussianModel
from spark.model.training_data import TrainingData


@pytest.mark.parametrize("training_data, config_set",
                         [(TrainingData(),
                           UniversalConfigSet(10, 1024 * 10))])
def test_gaussian_model(training_data, config_set):
    model = GaussianModel(training_data, config_set)
    with pytest.raises(Exception, match="No training data found"):
        assert model.train()
