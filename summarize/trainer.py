from cached_property import cached_property
import torch
import torch.nn.functional as F

from lib.base_trainer import BaseTrainer
from lib.nlp.decoder import Decoder

from summarize.summarize_net import SummarizeNet
from summarize.articles_dataset import ArticlesDataset
from summarize.text_to_parsed_doc import TextToParsedDoc
from summarize.words_to_vectors import WordsToVectors
from summarize.set_all_to_summarizing import SetAllToSummarizing
from summarize.merge_batch import MergeBatch

class Trainer(BaseTrainer):
    def __init__(self, nlp, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)

        self.nlp = nlp
        self.decoder = Decoder(nlp, self.device)

    @property
    def model_class(self):
        return SummarizeNet

    @cached_property
    def datasets(self):
        return {
            "train": ArticlesDataset(
                self.dataframe,
                "train",
                transforms=[
                    TextToParsedDoc(self.nlp),
                    WordsToVectors(self.nlp),
                    MergeBatch(self.device)
                ]
            ),
            "test":  ArticlesDataset(
                self.dataframe,
                "test",
                transforms=[
                    TextToParsedDoc(self.nlp),
                    WordsToVectors(self.nlp),
                    SetAllToSummarizing(),
                    MergeBatch(self.device)
                ]
            ),
            "val":  ArticlesDataset(
                self.dataframe,
                "val",
                transforms=[
                    TextToParsedDoc(self.nlp),
                    WordsToVectors(self.nlp),
                    MergeBatch(self.device)
                ]
            )
        }

    def compute_loss(self, word_embeddings, original_word_embeddings, discriminate_probs, original_modes):
        embeddings_loss = F.cosine_embedding_loss(
          word_embeddings.reshape((-1, word_embeddings.shape[2])),
          original_word_embeddings.reshape((-1, original_word_embeddings.shape[2])),
          torch.ones(word_embeddings.shape[0] * word_embeddings.shape[1]).to(self.device)
        )

        discriminator_loss = F.binary_cross_entropy(
            discriminate_probs,
            original_modes
        )

        return embeddings_loss # + discriminator_loss

    def work_batch(self, batch):
        word_embeddings, discriminate_probs = self.model(
            batch.word_embeddings.to(self.device),
            batch.word_embeddings_len.to(self.device),
            batch.mode.to(self.device)
        )

        return (
            self.compute_loss(word_embeddings, batch.word_embeddings, discriminate_probs, batch.mode),
            word_embeddings
        )
