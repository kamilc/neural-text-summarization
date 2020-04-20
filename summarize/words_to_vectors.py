import numpy as np

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
