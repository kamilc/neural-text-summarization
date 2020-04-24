import unittest
import doctest

from tests.model_tests import TestModel
from tests.decoder_tests import TestDecoder
from tests.metrics_tests import TestMetrics
from tests.trainer_tests import TestTrainer

doctest.testmod()
unittest.main(
    failfast=True,
    buffer=False
)
