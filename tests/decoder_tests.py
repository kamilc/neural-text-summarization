import unittest
import doctest
from hypothesis import given, settings, note, assume, reproduce_failure
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst

import torch
import spacy
from lib.nlp.decoder import Decoder
from summarize.words_to_vectors import WordsToVectors

from tests.support import Support

support = Support()

class TestDecoder(unittest.TestCase):
    def test_decoding_works(self):
        device = torch.device('cuda')
        words_to_vectors = WordsToVectors(support.nlp)
        decoder = Decoder(support.nlp, device)

        text = "this seems to work just fine"
        word_embeddings = words_to_vectors.document_embeddings(
            support.nlp(text)
        )

        inferred = decoder.decode_embeddings(
            torch.from_numpy(word_embeddings).unsqueeze(0).to(device=device)
        )

        self.assertEqual(list(inferred)[0], text)

if __name__ == '__main__':
    doctest.testmod()
    unittest.main(
        failfast=True,
        buffer=False
    )
