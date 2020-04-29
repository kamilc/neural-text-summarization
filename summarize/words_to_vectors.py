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
        sample['text_embeddings'] = sample.swifter.progress_bar(False).apply(
            lambda row: self.document_embeddings(row['text_doc']),
            axis=1
        )

        sample['headline_embeddings'] = sample.swifter.progress_bar(False).apply(
            lambda row: self.document_embeddings(row['headline_doc']),
            axis=1
        )

        return sample
