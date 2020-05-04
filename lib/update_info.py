from cached_property import cached_property
from lib.metrics import Metrics

class UpdateInfo(object):
    def __init__(self, model, vocabulary, batch, result, losses, mode):
        loss, discriminator_loss, model_loss, fooling_loss = losses

        self.model = model
        self.vocabulary = vocabulary
        self.batch = batch
        self.result = result
        self.loss = loss
        self.discriminator_loss = discriminator_loss
        self.model_loss = model_loss
        self.fooling_loss = fooling_loss
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
        return self.model.predict(
            self.vocabulary,
            self.batch.word_embeddings,
            self.batch.lengths
        )

    @cached_property
    def metrics(self):
        return Metrics(self.mode, self.loss, self.model_loss, self.fooling_loss)
