import unittest
import doctest
from hypothesis import given, settings, note, assume, reproduce_failure
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst

import itertools
import torch
import spacy
import pandas as pd
import numpy as np
import random
from summarize.summarize_net import SummarizeNet
from summarize.trainer import Trainer

class TestModel(unittest.TestCase):
    @given(
        st.sampled_from([4, 8, 12]),
        st.sampled_from([100, 200]),
        st.sampled_from([32, 64, 128]),
        st.sampled_from([1, 2, 3]),
    )
    @settings(max_examples=10, deadline=1000)
    def test_summarize_net_returns_correct_shapes(self, batch_size, seq_len, hidden_size, num_layers):
        model = SummarizeNet(
            hidden_size=hidden_size,
            input_size=300,
            num_layers=num_layers,
        )

        embeddings = torch.rand((batch_size, seq_len, 300))
        modes = torch.rand((batch_size))

        pred_embeddings, pred_modes = model(embeddings, modes)

        self.assertEqual(pred_embeddings.shape[0], batch_size)
        self.assertEqual(pred_embeddings.shape[1], seq_len)
        self.assertEqual(pred_embeddings.shape[2], 300)
        self.assertEqual(len(pred_embeddings.shape), 3)

        self.assertEqual(pred_modes.shape[0], batch_size)
        self.assertEqual(pred_modes.shape[1], 1)
        self.assertEqual(len(pred_modes.shape), 2)

