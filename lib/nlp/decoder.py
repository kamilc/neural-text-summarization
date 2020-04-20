
class Decoder(object):
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def decode_embeddings(self, word_embeddings):
        pass
#         data = word_embeddings.cpu().data.numpy()

#         return [
#             self.decode_embeddings_1d(data[ix, :, :])
#             for ix in range(0, data.shape[0])
#         ]

    def decode_embeddings_1d(self, word_embeddings):
        """
        Decodes a single document. Word embeddings given are of shape (N, D)
        where N is the number of lexemes and D the dimentionality of the embedding vector
        """

        pass

#         return "".join(
#             [
#                 token.text.lower() if not token.is_oov else " "
#                 for token in [
#                     self.nlp.vocab[ks[0]]
#                     for ks in self.nlp.vocab.vectors.most_similar(
#                         word_embeddings, n=1
#                     )[0]
#                 ]
#             ]
#         ).strip()
