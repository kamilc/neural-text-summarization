import unittest
import doctest

from tests.model_tests import TestModel
from tests.metrics_tests import TestMetrics
from tests.trainer_tests import TestTrainer
from tests.dataset_tests import TestDataset

doctest.testmod()
unittest.main(
    failfast=True,
    buffer=False
)
