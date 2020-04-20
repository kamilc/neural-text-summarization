import pickle
import spacy
from pathlib import Path

class Vocabulary(object):
    def __init__(self, nlp, series):
        self.nlp = nlp

        if Path("tmp/vocabulary.pickle").exists():
            with open('tmp/vocabulary.pickle', 'rb') as handle:
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

            with open('tmp/vocabulary.pickle', 'wb') as handle:
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
