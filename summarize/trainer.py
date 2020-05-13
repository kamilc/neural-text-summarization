from cached_property import cached_property
import torch
import torch.nn.functional as F

from lib.base_trainer import BaseTrainer

from summarize.summarize_net import SummarizeNet
from summarize.discriminator_net import DiscriminatorNet
from summarize.articles_dataset import ArticlesDataset
from summarize.rocstories_dataset import RocstoriesDataset
from summarize.words_to_vectors import WordsToVectors
from summarize.merge_batch import MergeBatch

class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)

    @property
    def model_class(self):
        return SummarizeNet

    @property
    def discriminator_class(self):
        return DiscriminatorNet

    @cached_property
    def datasets(self):
        klass = ArticlesDataset if self.dataset_class_name == "ArticlesDataset" else RocstoriesDataset
        return {
            "train": klass(
                self.dataframe,
                "train",
                transforms=[
                    WordsToVectors(self.vocabulary),
                    MergeBatch(self.device)
                ]
            ),
            "test":  klass(
                self.dataframe,
                "test",
                transforms=[
                    WordsToVectors(self.vocabulary),
                    MergeBatch(self.device)
                ]
            ),
            "val":  klass(
                self.dataframe,
                "val",
                transforms=[
                    WordsToVectors(self.vocabulary),
                    MergeBatch(self.device)
                ]
            )
        }

    def compute_model_loss(self, logits, classes):
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[2]).to(self.device),
            classes.long().reshape(-1).to(self.device)
        )

    def work_model(self, batch):
        logits, state = self.model(
            batch.word_embeddings.to(self.device),
            batch.mode.to(self.device)
        )

        return (
            self.compute_model_loss(
                logits,
                self.vocabulary.encode(batch.text),
            ),
            logits,
            state
        )
