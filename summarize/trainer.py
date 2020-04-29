from cached_property import cached_property
import torch
import torch.nn.functional as F

from lib.base_trainer import BaseTrainer

from summarize.summarize_net import SummarizeNet
from summarize.articles_dataset import ArticlesDataset
from summarize.text_to_parsed_doc import TextToParsedDoc
from summarize.words_to_vectors import WordsToVectors
from summarize.set_all_to_summarizing import SetAllToSummarizing
from summarize.merge_batch import MergeBatch

class Trainer(BaseTrainer):
    def __init__(self, vocabulary, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)

        self.vocabulary = vocabulary

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
                    TextToParsedDoc(self.vocabulary.nlp),
                    WordsToVectors(self.vocabulary.nlp),
                    MergeBatch(self.device)
                ]
            ),
            "test":  ArticlesDataset(
                self.dataframe,
                "test",
                transforms=[
                    TextToParsedDoc(self.vocabulary.nlp),
                    WordsToVectors(self.vocabulary.nlp),
                    MergeBatch(self.device)
                ]
            ),
            "val":  ArticlesDataset(
                self.dataframe,
                "val",
                transforms=[
                    TextToParsedDoc(self.vocabulary.nlp),
                    WordsToVectors(self.vocabulary.nlp),
                    MergeBatch(self.device)
                ]
            )
        }

    def compute_loss(self, text_logits, text_classes, text_state,
                     headline_logits, headline_classes, headline_state):

        headline_classes = F.pad(
            headline_classes,
            (0, headline_logits.shape[1] - headline_classes.shape[1])
        )

        headline_loss = F.cross_entropy(
            headline_logits.reshape(-1, headline_logits.shape[2]).to(self.device),
            headline_classes.long().reshape(-1).to(self.device)
        )

        text_loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.shape[2]).to(self.device),
            text_classes.long().reshape(-1).to(self.device)
        )

        gist_loss = F.mse_loss(text_state, headline_state)

        return headline_loss + text_loss + gist_loss

    def work_batch(self, batch):
        text_logits, text_state, headline_logits, headline_state = self.model(
            batch.text_embeddings.to(self.device),
            batch.headline_embeddings.to(self.device)
        )

        return (
            self.compute_loss(
                text_logits,
                self.vocabulary.encode(batch.text),
                text_state,
                headline_logits,
                self.vocabulary.encode(batch.headline),
                headline_state
            ),
            headline_logits
        )
