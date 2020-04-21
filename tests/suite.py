import unittest
import doctest

from tests.trainer_tests import TestTrainer
from tests.model_tests import TestModel
from tests.decoder_tests import TestDecoder

doctest.testmod()
unittest.main(
    failfast=True,
    buffer=False
)
