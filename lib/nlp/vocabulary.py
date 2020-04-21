import pickle
import spacy
from pathlib import Path

class Vocabulary(object):
    def __init__(self, nlp, series):
        self.nlp = nlp

        self.row2key = {}
        for key, row in nlp.vocab.vectors.key2row.items():
            self.row2key[row] = key

    def __len__(self):
        return len(self.nlp.vocab)

    def decode(self, probs):
        """
        probs: BxSxV tensor where:
          B = batch size
          S = sequence length
          V = vocabulary size
        """
        pass
