from cached_property import cached_property
from lib.metrics import Metrics
import torch

class UpdateInfo(object):
    def __init__(self, model, vocabulary, batch, result, losses, mode, no_period_trick=False):
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
        self.no_period_trick = no_period_trick

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
    def period_vector(self, device):
        return torch.tensor(
            self.vocabulary.nlp.vocab["."].vector
        ).to(device)

    @cached_property
    def decoded_inferred_texts(self):
        embeddings = self.batch.word_embeddings

        if self.no_period_trick:
            embeddings = embeddings.clone()

            period_vector = self.period_vector(
                embeddings.device
            )

            embeddings[ torch.eq(embeddings, period_vector).all(dim=2) ] = 0

        return self.model.predict(
            self.vocabulary,
            embeddings,
            self.batch.lengths
        )

    @cached_property
    def metrics(self):
        text = None
        predicted = None

        if self.mode != "train":
            text = self.batch.text
            predicted = self.decoded_inferred_texts

        return Metrics(
            self.mode,
            self.loss,
            self.model_loss,
            self.fooling_loss,
            text,
            predicted
        )
