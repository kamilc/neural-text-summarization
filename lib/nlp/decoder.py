import torch
import torch.nn.functional as F
from pathlib import Path
import pickle

class Decoder(object):
    def __init__(self, nlp, device, reference_size=1):
        self.nlp = nlp
        self.device = device
        self.reference_size = reference_size

        if Path("tmp/decoder.pickle").exists():
            with open('tmp/decoder.pickle', 'rb') as handle:
                data = pickle.load(handle)

            self.row2key = data['row2key']
            self.vocab_distances = data['vocab_distances']
        else:
            # initialize the reverse lookup index:
            self.row2key = {}
            for key, row in nlp.vocab.vectors.key2row.items():
                self.row2key[row] = key

            # initialize the reference cosine distance vectors:
            print(f"pre-computing vocabulary reference distances")

            self.vocab_distances = self.compute_distances(
                torch.from_numpy(nlp.vocab.vectors.data)
            ).transpose(1, 0).cpu()

            print(f"done")

            with open('tmp/decoder.pickle', 'wb') as handle:
                pickle.dump(
                    {'row2key': self.row2key, 'vocab_distances': self.vocab_distances},
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL
                )

        self.vocab_distances = self.vocab_distances.to(device=self.device)
        self.vocab_distances = self.vocab_distances[:, 0:reference_size]

    def compute_distances(self, word_vectors):
        selector = torch.zeros_like(word_vectors)
        distances = []

        for ix in range(0, self.reference_size):
            selector[:,:] = 0
            selector[:,ix] = 1
            distances.append(F.cosine_similarity(word_vectors, selector))

        return torch.stack(distances)


    def decode_embeddings(self, batch_word_embeddings):
        space_key = self.nlp.vocab[' '].orth

        for doc_ix in range(0, batch_word_embeddings.shape[0]):
            word_embeddings = batch_word_embeddings[doc_ix, :, :]
            distances = self.compute_distances(word_embeddings).to(device=self.device)
            rows = []

            for ix in range(0, distances.shape[1]):
                diffs = (self.vocab_distances - distances[:, ix])**2
                rows.append(diffs.sum(dim=1).argmin().item())

            keys = [
                self.row2key[row] if row in self.row2key else space_key
                for row in rows
            ]

            words = [
                self.nlp.vocab[key].text.lower()
                for key in keys
            ]

            yield(''.join(words))
