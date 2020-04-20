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

class TestTrainer(unittest.TestCase):
    def test_trainer_batches_include_onehot_encoded_word_ids(self):
        vocabulary = Vocabulary(nlp, [ articles["text"], articles["headline"] ])

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

        batch = next(trainer.batches('train'))

        self.assertEqual(batch.word_ids.shape[0], 32)
        self.assertEqual(batch.word_ids.shape[1], batch.word_embeddings.shape[1])
        self.assertEqual(batch.word_ids.shape[2], len(vocabulary))
        self.assertEqual(len(batch.word_ids), 3)

        self.assertIn(batch.word_ids[0, 0, 0], [0.0, 1.0])

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
