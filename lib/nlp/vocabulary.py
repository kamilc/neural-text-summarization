import pickle
import spacy
import torch
import torch.nn.functional as F
from rich.progress import track
from pathlib import Path

class Vocabulary(object):
    def __init__(self, nlp, series, size=None):
        self.nlp = nlp
        self.size = size

        if Path("tmp/vocabulary.pickle").exists():
            with open('tmp/vocabulary.pickle', 'rb') as handle:
                data = pickle.load(handle)

            self._words = data['words']
            self._index = data['index']
            self._sorted_data = data['sorted_data']
        else:
            text = ""
            words = ['<start-full>', '<start-short>', '<end>', '❟']
            index = {
                nlp.vocab['<start-full>'].orth: 0,
                nlp.vocab['<start-short>'].orth: 1,
                nlp.vocab['<end>'].orth: 2,
                nlp.vocab['❟'].orth: 3,
            }
            counts = {}

            for serie in series:
                for text in track(serie.fillna('').values.tolist(), description="Counting vocabulary words"):
                    text_counts = nlp(text.strip().lower()).count_by(spacy.attrs.LOWER)

                    for ix in text_counts:
                        if ix in counts:
                            counts[ix] += text_counts[ix]
                        else:
                            counts[ix] = text_counts[ix]

            sorted_data = sorted(
                [(ix, counts[ix]) for ix in counts],
                key=lambda t: t[1],
                reverse=True
            )

            for ix, _ in track(sorted_data, description="Building vocabulary index"):
                index[ix] = len(words)
                words.append(nlp.vocab[ix].text)

            self._words = words
            self._index = index
            self._sorted_data = sorted_data

            with open('tmp/vocabulary.pickle', 'wb') as handle:
                pickle.dump(
                    {
                        'words': self._words,
                        'index': self._index,
                        'sorted_data': self._sorted_data
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL
                )

    @property
    def words(self):
        return self._words[0:len(self)]

    def orth_to_word_id(self, orth):
        if orth in self._index.keys():
            key = self._index[orth]

            if key < len(self):
                return key
            else:
                return 3
        else:
            return 3

    def encode(self, texts, modes):
        classes = [
            torch.Tensor(
                [ mode ] +
                [
                    self.orth_to_word_id(l.orth)
                    for l in self.nlp(text.strip().lower())
                ] +
                [ 2 ]
            )
            for text, mode in zip(texts, modes)
        ]

        max_seq = max([c.shape[0] for c in classes])

        return torch.stack(
            [
                F.pad(vector, (0, max_seq - vector.shape[0]))
                for vector in classes
            ]
        )

    def token_vector(self, token):
        if token == '<start-full>':
            return (torch.ones(300) * 0).cos()
        elif token == '<start-short>':
            return (torch.ones(300) * 1).cos()
        elif token == '<end>':
            return (torch.ones(300) * 2).cos()
        else:
            return torch.tensor(self.nlp.vocab[token.lower()].vector)

    def embed(self, tokens):
        """
        Turns a list of string tokens into a vector 1xSxD
        """

        return torch.stack(
            [
                self.token_vector(token)
                for token in tokens
            ]
        )

    def __len__(self):
        return self.size if self.size is not None else len(self._words)
