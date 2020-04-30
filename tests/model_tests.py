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
from lib.nlp.vocabulary import Vocabulary
from tests.support import Support
from summarize.summarize_net import SummarizeNet
from summarize.trainer import Trainer

#support = Support()
#vocabulary = Vocabulary(support.nlp, support.articles)

class TestModel(unittest.TestCase):
    @given(
        st.sampled_from([4, 8, 12]),
        st.sampled_from([100, 200]),
        st.sampled_from([32, 64, 128]),
        st.sampled_from([1, 2, 3]),
        st.sampled_from([100, 200]),
        st.floats(min_value=0, max_value=1)
    )
    @settings(max_examples=10, deadline=10000)
    def test_summarize_net_returns_correct_shapes(self, batch_size, seq_len,
                                                  hidden_size, num_layers,
                                                  vocabulary_size, dropout_rate):
        model = SummarizeNet(
            device='cuda',
            hidden_size=hidden_size,
            input_size=300,
            num_heads=10,
            dim_feedforward_transformer=128,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            vocabulary_size=vocabulary_size
        )

        embeddings = torch.rand((batch_size, seq_len, 300)).cuda()
        modes = torch.rand((batch_size, 1)).cuda()

        decoded, state = model(
            embeddings,
            modes
        )

        self.assertEqual(decoded.shape[0], batch_size)
        self.assertEqual(decoded.shape[1], seq_len)
        self.assertEqual(decoded.shape[2], vocabulary_size)
        self.assertEqual(len(decoded.shape), 3)

        self.assertEqual(state.shape[0], batch_size)
        self.assertEqual(state.shape[1], hidden_size)
        self.assertEqual(len(state.shape), 2)

if __name__ == '__main__':
    doctest.testmod()
    unittest.main(
        failfast=True,
        buffer=False
    )
