import unittest
import doctest
from hypothesis import given, settings, note, assume, reproduce_failure
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst

import torch
from lib.metrics import Metrics

class TestMetrics(unittest.TestCase):
    def test_loss_average_works(self):
        cumulative_metrics = Metrics.empty("train")

        for i in range(0, 10):
            cumulative_metrics += Metrics("train", torch.tensor([i*1.0]).item())

        self.assertEqual(cumulative_metrics.loss, 4.5)
        self.assertEqual(cumulative_metrics.running_mean_loss(5), 7)
        self.assertEqual(cumulative_metrics.running_mean_loss(100), 4.5)

if __name__ == '__main__':
    doctest.testmod()
    unittest.main(
        failfast=True,
        buffer=False
    )
