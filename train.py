#!/usr/bin/env python
# coding: utf-8

import os
import functools
import statistics
import itertools
import random
import math
from pathlib import Path
import pdb

import pandas as pd
import swifter
import numpy as np
import hickle as hkl
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import spacy
from cached_property import cached_property

if 'nlp' not in vars():
    nlp = spacy.load(
        "en_core_web_lg",
        disable=["tagger", "ner", "textcat"]
    )

if 'articles' not in vars():
    articles = pd.read_parquet("data/articles-processed.parquet.gzip")



class TextToParsedDoc(object):
    def __init__(self, nlp):
        self.nlp = nlp

    def __call__(self, sample):
        sample['doc'] = sample.swifter.progress_bar(False).apply(lambda row: self.nlp(row['text']), axis=1)
        return sample

class WordsToVectors(object):
    def __init__(self, nlp):
        self.nlp = nlp

    def document_embeddings(self, doc):
        word_embeddings = [
            [ l.vector ] if l.whitespace_ == '' else [ l.vector, np.zeros_like(l.vector) ] for l in doc
        ]

        return np.stack(
            [
                vector for vectors in word_embeddings for vector in vectors
            ]
        )

    def __call__(self, sample):

        sample['word_embeddings'] = sample.swifter.progress_bar(False).apply(
            lambda row: self.document_embeddings(row['doc']),
            axis=1
        )

        return sample

class AddNoiseToEmbeddings(object):
    def __init__(self, probability_of_mask_for_word):
        self.probability_of_mask_for_word = probability_of_mask_for_word
        self.rng = np.random.default_rng()

    def mask_vector(self, vector):
        """
        Masks words with zeros randomly
        """
        seq_len = vector.shape[0]
        vector_len = vector.shape[1]

        mask = np.repeat(
            self.rng.choice(
                [0, 1],
                seq_len,
                p=[
                    self.probability_of_mask_for_word,
                    (1 - self.probability_of_mask_for_word)
                ]
            ).reshape((seq_len, 1)),
            vector_len,
            axis=1
        )

        return vector * mask

    def __call__(self, sample):
        sample['noisy_word_embeddings'] = sample['word_embeddings'].apply(self.mask_vector)

        return sample

class MergeBatch(object):
    def __init__(self, device):
        self.device = device

    def stack_vectors(self, vectors):
        max_seq = max([vector.shape[0] for vector in vectors])

        return np.stack(
            [
                np.pad(vector, [(0, max_seq - vector.shape[0]), (0, 0)])
                for vector in vectors
            ]
        )

    def __call__(self, sample):
        del sample['doc']

        sample = sample.to_dict(orient="list")

        sample['word_embeddings'] = torch.from_numpy(
            self.stack_vectors(
                sample['word_embeddings']
            ).astype(np.float32, copy=False)
        ).to(self.device)

        if 'noisy_word_embeddings' in sample:
            sample['noisy_word_embeddings'] = torch.from_numpy(
                self.stack_vectors(
                    sample['noisy_word_embeddings']
                ).astype(np.float32, copy=False)
            ).to(self.device)

        sample['mode'] = torch.from_numpy(
            np.stack(
                sample['mode']
            ).astype(np.float32, copy=False)
        ).to(self.device)

        return sample

class SetAllToSummarizing(object):
    def __call__(self, sample):
        sample['mode'] = np.ones_like(sample['mode']).astype(np.float32, copy=False)

        return sample

class Vocabulary(object):
    def __init__(self, nlp, series):
        self.nlp = nlp

        if Path("vocabulary.pickle").exists():
            with open('vocabulary.pickle', 'rb') as handle:
                data = pickle.load(handle)

            self.words = data['words']
            self.index = data['index']
        else:
            text = ""
            words = []
            index = {}
            counts = {}

            for serie in series:
                for text in serie.fillna('').values.tolist():
                    text_counts = nlp(text).count_by(spacy.attrs.LOWER)

                    for ix in text_counts:
                        if ix in counts:
                            counts[ix] += text_counts[ix]
                        else:
                            counts[ix] = text_counts[ix]

            for ix, _ in sorted([(ix, counts[ix]) for ix in counts],key=lambda t: t[1],reverse=True):
                words.append(nlp.vocab[ix].text)
                index[ix] = len(words)

            self.words = words
            self.index = index

            with open('vocabulary.pickle', 'wb') as handle:
                pickle.dump({'words': self.words, 'index': self.index}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.words)

    def decode(self, probs):
        """
        probs: BxSxV tensor where:
          B = batch size
          S = sequence length
          V = vocabulary size
        """
        pass

class DataLoader(object):
    def __init__(self, dataset, batch_size=8):
        self.dataset = dataset
        self.batch_size = batch_size

    @property
    def epoch_size(self):
        return math.ceil(len(self.dataset) / self.batch_size) * self.batch_size

    def __iter__(self):
        ids = random.choices(range(0, len(self.dataset)), k=self.epoch_size)

        for start_ix in range(0, self.epoch_size, self.batch_size):
            yield self.dataset[ids[start_ix:(start_ix + self.batch_size)]]

class ArticlesBatch:
    def __init__(self, data, ix=0):
        self.data = data
        self.ix = ix


    def __getattr__(self, name):
        if name in self.data:
            return self.data[name]
        else:
            raise AttributeError(f"Attribute missing: {name}")

class Decoder(object):
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def decode_embeddings(self, word_embeddings):
        pass
#         data = word_embeddings.cpu().data.numpy()

#         return [
#             self.decode_embeddings_1d(data[ix, :, :])
#             for ix in range(0, data.shape[0])
#         ]

    def decode_embeddings_1d(self, word_embeddings):
        """
        Decodes a single document. Word embeddings given are of shape (N, D)
        where N is the number of lexemes and D the dimentionality of the embedding vector
        """

        pass

#         return "".join(
#             [
#                 token.text.lower() if not token.is_oov else " "
#                 for token in [
#                     self.nlp.vocab[ks[0]]
#                     for ks in self.nlp.vocab.vectors.most_similar(
#                         word_embeddings, n=1
#                     )[0]
#                 ]
#             ]
#         ).strip()


# In[485]:


class Metrics(object):
    def __init__(self, mode, loss=None):
        self.mode = mode
        self.losses = [loss.cpu().item()] if loss is not None else []

    @classmethod
    def empty(cls, mode):
        return cls(mode)

    @property
    def loss(self):
        if len(self.losses) == 0:
            return 0
        else:
            return statistics.mean(self.losses)

    @property
    def last_loss(self):
        return self.losses[len(self.losses) - 1]

    def running_mean_loss(self, n=100):
        cumsum = np.cumsum(np.insert(np.array(self.losses), 0, 0))
        return (cumsum[n:] - cumsum[:-n]) / float(n)

    def __add__(self, other):
        self.losses += other.losses

        return self


# In[486]:


class UpdateInfo(object):
    def __init__(self, decoder, batch, word_embeddings, loss_sum, mode):
        self.decoder = decoder
        self.batch = batch
        self.word_embeddings = word_embeddings
        self.loss_sum = loss_sum
        self.mode = mode

    @property
    def from_train(self):
        return self.mode == "train"

    @property
    def from_evaluate(self):
        return self.mode == "val"

    @cached_property
    def decoded_inferred_texts(self):
        return self.decoder.decode_embeddings(self.word_embeddings)

    @cached_property
    def metrics(self):
        return Metrics(self.mode, self.loss_sum)

    def __str__(self):
        return f"{self.mode} | {self.batch.ix}\t| Loss: {loss_sum}\t"

class BaseTrainer:
    def __init__(self, name, vocabulary, dataframe,
                 optimizer_class_name,
                 model_args, optimizer_args,
                 batch_size, update_every,
                 probability_of_mask_for_word,
                 device
                ):
        self.name = name

        self.device = device

        self.datasets = {
            "train": ArticlesDataset(
                dataframe,
                "train",
                transforms=[
                    TextToParsedDoc(vocabulary.nlp),
                    WordsToVectors(vocabulary.nlp),
                    AddNoiseToEmbeddings(probability_of_mask_for_word),
                    MergeBatch(device)
                ]
            ),
            "test":  ArticlesDataset(
                dataframe,
                "test",
                transforms=[
                    TextToParsedDoc(vocabulary.nlp),
                    WordsToVectors(vocabulary.nlp),
                    AddNoiseToEmbeddings(0),
                    SetAllToSummarizing(),
                    MergeBatch(device)
                ]
            ),
            "val":  ArticlesDataset(
                dataframe,
                "val",
                transforms=[
                    TextToParsedDoc(vocabulary.nlp),
                    WordsToVectors(vocabulary.nlp),
                    AddNoiseToEmbeddings(0),
                    MergeBatch(device)
                ]
            )
        }

        self.batch_size = batch_size
        self.update_every = update_every

        self.optimizer_class_name = optimizer_class_name

        self.model_args = model_args
        self.optimizer_args = optimizer_args

        self.current_batch_id = 0

        self.decoder = Decoder(vocabulary)

        if self.has_checkpoint:
            self.load_last_checkpoint()

    @cached_property
    def model(self):
        try:
            return SummarizeNet.load(f"{self.checkpoint_path}/model.pth").to(self.device)
        except FileNotFoundError:
            return SummarizeNet(**self.model_args).to(self.device)

    @cached_property
    def optimizer(self):
        class_ = getattr(torch.optim, self.optimizer_class_name)

        return class_(self.model.parameters(), **self.optimizer_args)

    @property
    def checkpoint_path(self):
        return f"checkpoints/{self.name}/batch-#{self.current_batch_id}"

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.model.save(f"{self.checkpoint_path}/model.pth")

        torch.save(
            {
                'current_batch_id': self.current_batch_id,
                'batch_size': self.batch_size,
                'update_every': self.update_every,
                'optimizer_class_name': self.optimizer_class_name,
                'optimizer_args': self.optimizer_args,
                'optimizer_state_dict': self.optimizer.state_dict()
            },
            f"{self.checkpoint_path}/trainer.pth"
        )

    @property
    def checkpoint_directories(self):
        return sorted(Path(".").glob(f"checkpoints/{self.name}/batch-*"), reverse=True)

    @property
    def has_checkpoint(self):
        return len(self.checkpoint_directories) > 0

    def load_last_checkpoint(self):
        path = self.checkpoint_directories[0]

        data = torch.load(f"{path}/trainer.pth")

        self.batch_size = data['batch_size']
        self.update_every = data['update_every']

        self.optimizer_class_name = data['optimizer_class_name']
        self.optimizer_args = data['optimizer_args']

        self.current_batch_id = data['current_batch_id']

        if 'model' in self.__dict__:
            del self.__dict__['model']

        if 'optimzer' in self.__dict__:
            del self.__dict__['optimizer']

        self.optimizer.load_state_dict(data['optimizer_state_dict'])

    def batches(self, mode):
        while True:
            loader = DataLoader(
                self.datasets[mode],
                batch_size=self.batch_size
            )

            for data in loader:
                self.current_batch_id += 1

                yield(
                    ArticlesBatch(
                        data,
                        ix=self.current_batch_id
                    )
                )

    def work_batch(self, batch):
        raise NotImplementedError

    def updates(self, mode="train", update_every=None):
        batches = self.batches(mode)
        loss_sum = 0

        if update_every is None:
            update_every = self.update_every

        for batch in batches:
            if mode == "train":
                self.model.train()
            else:
                self.model.eval()

            loss, word_embeddings = self.work_batch(batch)
            loss /= self.update_every

            if mode == "train":
                loss.backward()

            loss_sum += loss

            # we're doing the accumulated gradients trick to get the gradients variance
            # down while being able to use commodity GPU:
            if batch.ix % update_every == 0:
                if mode == "train":
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                yield(UpdateInfo(self.decoder, batch, word_embeddings, loss_sum, mode=mode))

                loss_sum = 0

    def train_and_evaluate_updates(self, evaluate_every=100):
        train_updates = self.updates(mode="train")
        evaluate_updates = self.updates(mode="val")

        for update_info in train_updates:
            yield(update_info)

            if update_info.batch.ix != 0 and update_info.batch.ix % evaluate_every == 0:
                yield(next(evaluate_updates))

    def test_updates(self):
        return self.updates(mode="test", update_every=1)


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)

    def compute_loss(self, word_embeddings, original_word_embeddings, discriminate_probs):
        embeddings_loss = F.cosine_embedding_loss(
          word_embeddings.reshape((-1, word_embeddings.shape[2])),
          original_word_embeddings.reshape((-1, original_word_embeddings.shape[2])),
          torch.ones(word_embeddings.shape[0] * word_embeddings.shape[1]).to(self.device)
        )

        discriminator_loss = F.binary_cross_entropy(
            discriminate_probs,
            torch.zeros_like(discriminate_probs).to(self.device)
        )

        return embeddings_loss + discriminator_loss


    def work_batch(self, batch):
        word_embeddings, discriminate_probs = self.model(
            batch.noisy_word_embeddings,
            batch.mode
        )

        # we're diverging from the article here by outputting the word embeddings
        # instead of the probabilities for each word in a vocabulary
        # our loss function is using the cosine embedding loss coupled with
        # the discriminator loss:
        return (
            self.compute_loss(word_embeddings, batch.word_embeddings, discriminate_probs),
            word_embeddings
        )

class InNotebookTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(InNotebookTrainer, self).__init__(*args, **kwargs)

        self.writer = SummaryWriter(comment=self.name)

    def train(self, evaluate_every=1000):
        test_updates = self.test_updates()

        cumulative_train_metrics = Metrics.empty(mode="train")
        cumulative_evaluate_metrics = Metrics.empty(mode="eval")

        for update_info in self.train_and_evaluate_updates(evaluate_every=evaluate_every):
            if update_info.from_train:
                cumulative_train_metrics += update_info.metrics

                print(f"{update_info.batch.ix}")

                self.writer.add_scalar(
                    'loss/train',
                    update_info.metrics.loss,
                    update_info.batch.ix
                )

            if update_info.from_evaluate:
                cumulative_evaluate_metrics += update_info.metrics

                self.writer.add_scalar(
                    'loss/eval',
                    update_info.metrics.loss,
                    update_info.batch.ix
                )

                print(f"Eval: {update_info.metrics.loss}")
                print(f"Saving checkpoint")
                self.save_checkpoint()

#             if update_info.batch.ix % 1000 == 0 and update_info.batch.ix != 0:
#                 test_update = next(test_updates)

#                 self.test_texts_stream.write(
#                     (
#                         update_info.batch.text,
#                         update_info.decoded_inferred_texts
#                     )
#                 )

    def test(self):
        cumulative_metrics = Metrics.empty(mode="test")

        for update_info in self.test_updates():
            cumulative_metrics += update_info.metrics

        print(cumulative_metrics)

RUN_TESTS = True

import unittest
from hypothesis import given, settings, note, assume, reproduce_failure
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst

class TestNotebook(unittest.TestCase):
    def test_trainer_batches_yields_proper_ixs(self):
        vocabulary = Vocabulary(nlp, [ articles["text"], articles["headline"] ])

        for mode in ['train', 'test', 'val']:
            trainer = Trainer(
                'unit-test-run-1',
                vocabulary,
                articles,
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
            ixs = [batch.ix for batch in itertools.islice(trainer.batches(mode), 10)]
            self.assertEqual(list(ixs), list(range(1, 11)))

    @given(
        st.sampled_from([4, 8, 12]),
        st.sampled_from([100, 200]),
        st.sampled_from([32, 64, 128]),
        st.sampled_from([1, 2, 3]),
        st.sampled_from([100, 200])
    )
    @settings(max_examples=10)
    def test_summarize_net_returns_correct_shapes(self, batch_size, seq_len, hidden_size, num_layers, vocabulary_size):
        model = SummarizeNet(
            hidden_size=hidden_size,
            input_size=300,
            num_layers=num_layers,
            vocabulary_size=vocabulary_size,
            cutoffs=[1, vocabulary_size - 1]
        )

        embeddings = torch.rand((batch_size, seq_len, 300))
        target = torch.rand((batch_size, seq_len)).int()
        modes = torch.rand((batch_size))

        pred_probs, pred_modes, loss = model(embeddings, target, modes)

        self.assertEqual(pred_probs.shape[0], batch_size)
        self.assertEqual(pred_probs.shape[1], seq_len)
        self.assertEqual(len(pred_probs.shape), 2)

        self.assertEqual(pred_modes.shape[0], batch_size)
        self.assertEqual(len(pred_modes.shape), 1)

        self.assertGreater(loss.item(), 0)

if __name__ == '__main__' and RUN_TESTS:
    import doctest

    doctest.testmod()
    unittest.main(
        argv=['first-arg-is-ignored'],
        failfast=True,
        exit=False
    )

if not RUN_TESTS:
    if 'trainer' in vars():
        print(f"About to delete old trainer")
        del trainer

    vocabulary = Vocabulary(nlp, [ articles["text"], articles["headline"] ])

    trainer = InNotebookTrainer(
        'test-run-1',
        vocabulary,
        articles,
        optimizer_class_name='Adam',
        model_args={
            'hidden_size': 128,
            'input_size': 300,
            'num_layers': 2,
            'vocabulary_size': len(vocabulary)
        },
        optimizer_args={},
        batch_size=32,
        update_every=1,
        probability_of_mask_for_word=0.2,
        device=torch.device('cuda')
    )

    trainer.train()

