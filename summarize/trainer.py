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

    def compute_loss(self, logits, classes, logits_reconstruct, classes_reconstruct):
        classes = F.pad(
            classes,
            (0, classes_reconstruct.shape[1] - classes.shape[1], 0, 0)
        )

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[2]).to(self.device),
            classes.long().reshape(-1).to(self.device)
        )

        loss_reconstruct = F.cross_entropy(
            logits.reshape(-1, logits_reconstruct.shape[2]).to(self.device),
            classes_reconstruct.long().reshape(-1).to(self.device)
        )

        return loss + loss_reconstruct

    def work_batch(self, batch):
        logits, reproduce_logits = self.model(
            batch.word_embeddings.to(self.device)
        )

        return (
            self.compute_loss(
                logits,
                self.vocabulary.encode(batch.headline),
                reproduce_logits,
                self.vocabulary.encode(batch.text)
            ),
            logits
        )
