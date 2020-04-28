from cached_property import cached_property
from lib.metrics import Metrics

class UpdateInfo(object):
    def __init__(self, vocabulary, batch, result, loss_sum, mode):
        self.vocabulary = vocabulary
        self.batch = batch
        self.result = result
        self.loss_sum = loss_sum
        self.mode = mode

    @property
    def from_train(self):
        return self.mode == "train"

    @property
    def from_evaluate(self):
        return self.mode == "val"

    @cached_property
    def decoded_word_ids(self):
        return self.result.argmax(dim=2).tolist()

    @cached_property
    def decoded_inferred_texts(self):
        return (
            ' '.join([ self.vocabulary.words[id] for id in ids])
            for ids in self.decoded_word_ids
        )

    @cached_property
    def metrics(self):
        return Metrics(self.mode, self.loss_sum)
