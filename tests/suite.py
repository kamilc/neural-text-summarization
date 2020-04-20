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
from lib.nlp.vocabulary import Vocabulary

nlp = spacy.load(
    "en_core_web_lg",
    disable=["tagger", "ner", "textcat"]
)

articles = pd.read_parquet("data/articles-processed.parquet.gzip")

class TestPackage(unittest.TestCase):
    def test_trainer_batches_yields_proper_ixs(self):
        vocabulary = Vocabulary(nlp, [ articles["text"], articles["headline"] ])

        for mode in ['train', 'test', 'val']:
            trainer = Trainer(
                name='unit-test-run-1',
                vocabulary=vocabulary,
                dataframe=articles,
                optimizer_class_name='Adam',
                model_args={
                    'hidden_size': 128,
                    'input_size': 300,
                    'num_layers': 2,
                    'cutoffs': [1, 2],
                    'vocabulary_size': len(vocabulary)
                },
                optimizer_args={},
                batch_size=32,
                update_every=1,
                probability_of_mask_for_word=0.3,
                device=torch.device('cpu')
            )
            self.assertGreater(len(trainer.datasets[mode]), 0)
            ixs = [
                batch.ix
                for batch in itertools.islice(trainer.batches(mode), 10)
            ]
            self.assertEqual(list(ixs), list(range(1, 11)))

    @given(
        st.sampled_from([4, 8, 12]),
        st.sampled_from([100, 200]),
        st.sampled_from([32, 64, 128]),
        st.sampled_from([1, 2, 3]),
        st.sampled_from([100, 200])
    )
    @settings(max_examples=10, deadline=1000)
    def test_summarize_net_returns_correct_shapes(self, batch_size, seq_len, hidden_size, num_layers, vocabulary_size):
        model = SummarizeNet(
            hidden_size=hidden_size,
            input_size=300,
            num_layers=num_layers,
            vocabulary_size=vocabulary_size,
        )

        embeddings = torch.rand((batch_size, seq_len, 300))
        target = torch.tensor(
            np.stack(
                [
                    np.array(
                        random.choices(range(0, vocabulary_size), k=seq_len)
                    )
                    for _ in range(0, batch_size)
                ]
            )
        ).int()
        modes = torch.rand((batch_size))

        pred_logits, pred_modes = model(embeddings, modes)

        self.assertEqual(pred_logits.shape[0], batch_size)
        self.assertEqual(pred_logits.shape[1], seq_len)
        self.assertEqual(pred_logits.shape[2], vocabulary_size)
        self.assertEqual(len(pred_logits.shape), 3)

        self.assertEqual(pred_modes.shape[0], batch_size)
        self.assertEqual(pred_modes.shape[1], 1)
        self.assertEqual(len(pred_modes.shape), 2)


doctest.testmod()
unittest.main(
    failfast=True,
    exit=False
)
