import numpy as np
import torch

class WordsToVectors(object):
    def __init__(self, vocabulary, no_period_trick=False):
        self.vocabulary = vocabulary
        self.no_period_trick = no_period_trick

    def document_embeddings(self, doc, mode):
        return self.vocabulary.embed(
            ['<start-full>' if mode == 0.0 else '<start-short>'] +
            [
                l.text.lower()
                for l in doc
            ] +
            ['<end>']
        )

    def to_document(self, text):
        txt = text.replace('.', '') if self.no_period_trick else text

        return self.vocabulary.nlp(txt)

    def __call__(self, sample):
        sample['doc'] = sample.apply(
            lambda row: self.to_document(row['text']),
            axis=1
        )

        sample['clean_doc'] = sample.apply(
            lambda row: self.to_document(row['orig_text']),
            axis=1
        )

        sample['word_embeddings'] = sample.apply(
            lambda row: self.document_embeddings(row['doc'], row['mode']),
            axis=1
        )

        sample['clean_word_embeddings'] = sample.apply(
            lambda row: self.document_embeddings(row['clean_doc'], row['mode']),
            axis=1
        )

        sample['lengths'] = sample.apply(
            lambda row: len(row['doc']) + 2,
            axis=1
        )

        del sample['doc']

        return sample
