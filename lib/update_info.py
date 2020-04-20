from lib.metrics import Metrics

class UpdateInfo(object):
    def __init__(self, decoder, batch, word_embeddings, loss_sum, mode):
        self.decoder = decoder
        self.batch = batch
        self.word_embeddings = word_embeddings
        self.loss_sum = loss_sum
        self.mode = mode

    @property
    def from_train(self):
        return self.mode == "train"

    @property
    def from_evaluate(self):
        return self.mode == "val"

    @cached_property
    def decoded_inferred_texts(self):
        return self.decoder.decode_embeddings(self.word_embeddings)

    @cached_property
    def metrics(self):
        return Metrics(self.mode, self.loss_sum)

    def __str__(self):
        return f"{self.mode} | {self.batch.ix}\t| Loss: {loss_sum}\t"
