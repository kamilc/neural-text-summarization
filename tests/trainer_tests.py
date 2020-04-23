import unittest
import doctest
from hypothesis import given, settings, note, assume, reproduce_failure
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst

import itertools
import torch
import spacy
import numpy as np
import random
from summarize.summarize_net import SummarizeNet
from summarize.trainer import Trainer

from tests.support import Support

support = Support()

class TestTrainer(unittest.TestCase):
    def test_trainer_batches_yields_proper_ixs(self):
        for mode in ['train', 'test', 'val']:
            trainer = Trainer(
                name='unit-test-run-1',
                nlp=support.nlp,
                dataframe=support.articles,
                optimizer_class_name='Adam',
                model_args={
                    'hidden_size': 128,
                    'input_size': 300,
                    'num_layers': 2,
                    'num_heads': 10,
                    'dropout_rate': 0.2,
                    'dim_feedforward_transformer': 128
                },
                optimizer_args={},
                batch_size=32,
                update_every=1,
                device=torch.device('cpu')
            )
            self.assertGreater(len(trainer.datasets[mode]), 0)
            ixs = [
                batch.ix
                for batch in itertools.islice(trainer.batches(mode), 10)
            ]
            self.assertEqual(list(ixs), list(range(1, 11)))

    def test_text_decoding_from_embeddings_work(self):
        trainer = Trainer(
            name='unit-test-run-1',
            nlp=support.nlp,
            dataframe=support.articles,
            optimizer_class_name='Adam',
            model_args={
                'hidden_size': 128,
                'input_size': 300,
                'num_layers': 2,
                'num_heads': 10,
                'dropout_rate': 0.2,
                'dim_feedforward_transformer': 128
            },
            optimizer_args={},
            batch_size=32,
            update_every=1,
            device=torch.device('cuda')
        )

        update_info = next(trainer.updates("train"))
        inferred_text = next(update_info.decoded_inferred_texts)

        self.assertNotEqual(inferred_text, "")
        self.assertIsNotNone(inferred_text)

if __name__ == '__main__':
    doctest.testmod()
    unittest.main(
        failfast=True,
        buffer=False
    )
