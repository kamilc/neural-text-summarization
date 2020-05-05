import numpy as np
import torch

class WordsToVectors(object):
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def document_embeddings(self, doc, mode):
        return self.vocabulary.embed(
            ['<start-full>' if mode == 0.0 else '<start-short>'] +
            [
                l.text.lower()
                for l in doc
            ] +
            ['<end>']
        )

    def __call__(self, sample):
        sample['doc'] = sample.apply(
            lambda row: self.vocabulary.nlp(row['text']),
            axis=1
        )

        sample['word_embeddings'] = sample.apply(
            lambda row: self.document_embeddings(row['doc'], row['mode']),
            axis=1
        )

        sample['lengths'] = sample.apply(
            lambda row: len(row['doc']) + 2,
            axis=1
        )

        del sample['doc']

        return sample
