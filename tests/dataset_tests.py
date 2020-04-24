import unittest
import doctest
from hypothesis import given, settings, note, assume, reproduce_failure
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst

import torch
import torch.nn.functional as F

from lib.batch import Batch
from lib.dataloading.dataloader import DataLoader

from summarize.articles_dataset import ArticlesDataset
from summarize.text_to_parsed_doc import TextToParsedDoc
from summarize.words_to_vectors import WordsToVectors
from summarize.set_all_to_summarizing import SetAllToSummarizing
from summarize.merge_batch import MergeBatch

from tests.support import Support

support = Support()

class TestDataset(unittest.TestCase):
    def test_modes_are_assigned_correctly(self):
        dataset = ArticlesDataset(
            support.articles,
            "train",
            transforms=[
                TextToParsedDoc(support.nlp),
                WordsToVectors(support.nlp),
                MergeBatch(torch.device("cuda"))
            ]
        )
        loader = DataLoader(
            dataset,
            batch_size=8
        )

        word2vec = WordsToVectors(support.nlp).document_embeddings

        current_batch_id = 0

        for data in loader:
            if current_batch_id > 10:
                break
            else:
                current_batch_id += 1

                batch = Batch(
                    data,
                    ix=current_batch_id
                )

                for i in range(0, len(batch.idx)):
                    ds = support.articles[
                        support.articles.normalized_title == batch.title[i]
                    ]

                    self.assertEqual(ds.shape[0], 1)

                    if batch.mode[i].item() == 1.0:
                        self.assertEqual(ds.iloc[0].headline, batch.text[i])
                    else:
                        self.assertEqual(ds.iloc[0].text, batch.text[i])

                    embeddings = torch.from_numpy(
                        word2vec(
                            support.nlp(batch.text[i])
                        )
                    )

                    diff = batch.word_embeddings[i, :, :].shape[0] - embeddings.shape[0]
                    embeddings = F.pad(embeddings, (0,0,0,diff))

                    self.assertTrue(
                        (embeddings.cpu() == batch.word_embeddings.cpu()[i, :, :]).all()
                    )

if __name__ == '__main__':
    doctest.testmod()
    unittest.main(
        failfast=True,
        buffer=False
    )

