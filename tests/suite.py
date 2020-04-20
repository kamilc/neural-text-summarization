import unittest
import doctest

from tests.trainer_tests import TestTrainer
from tests.model_tests import TestModel

doctest.testmod()
unittest.main(
    failfast=True,
    buffer=False
)
