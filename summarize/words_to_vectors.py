import numpy as np
import torch

class WordsToVectors(object):
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def document_embeddings(self, doc):
        return self.vocabulary.embed(
            [
                l.text.lower()
                for l in doc
            ]
        )

    def __call__(self, sample):
        sample['doc'] = sample.apply(
            lambda row: self.vocabulary.nlp(row['text']),
            axis=1
        )

        sample['word_embeddings'] = sample.apply(
            lambda row: self.document_embeddings(row['doc']),
            axis=1
        )

        sample['lengths'] = sample.apply(
            lambda row: len(row['doc']),
            axis=1
        )

        del sample['doc']

        return sample
