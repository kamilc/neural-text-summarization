import numpy as np

class WordsToVectors(object):
    def __init__(self, nlp):
        self.nlp = nlp

    def document_embeddings(self, doc):
        return np.stack(
            [
                lexeme.vector for lexeme in doc
            ]
        )

    def __call__(self, sample):
        sample['word_embeddings'] = sample.swifter.progress_bar(False).apply(
            lambda row: self.document_embeddings(row['doc']),
            axis=1
        )

        sample['lengths'] = sample.swifter.progress_bar(False).apply(
            lambda row: len(row['doc']),
            axis=1
        )

        return sample
