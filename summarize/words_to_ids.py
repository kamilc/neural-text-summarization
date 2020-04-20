import swifter
import torch
import numpy as np

class WordsToIds(object):
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def document_word_ids(self, doc):

        indices = np.array([[
            self.vocabulary.word_to_id(word.text.lower())
            for word in doc
        ]]).reshape(-1)

        indices = torch.from_numpy(indices)

        return torch.nn.functional.one_hot(indices, len(self.vocabulary)).to_sparse()

    def __call__(self, sample):
        sample['word_ids'] = sample.apply(
            lambda row: self.document_word_ids(row['doc']),
            axis=1
        )

        return sample
