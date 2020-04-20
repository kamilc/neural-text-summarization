from cached_property import cached_property

from lib.base_trainer import BaseTrainer
from lib.nlp.decoder import Decoder

from summarize.summarize_net import SummarizeNet
from summarize.articles_dataset import ArticlesDataset
from summarize.text_to_parsed_doc import TextToParsedDoc
from summarize.words_to_vectors import WordsToVectors
from summarize.add_noise_to_embeddings import AddNoiseToEmbeddings
from summarize.set_all_to_summarizing import SetAllToSummarizing
from summarize.words_to_ids import WordsToIds
from summarize.merge_batch import MergeBatch

class Trainer(BaseTrainer):
    def __init__(self, vocabulary, probability_of_mask_for_word, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)

        self.vocabulary = vocabulary
        self.probability_of_mask_for_word = probability_of_mask_for_word
        self.decoder = Decoder(vocabulary)

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
                    WordsToIds(self.vocabulary),
                    AddNoiseToEmbeddings(self.probability_of_mask_for_word),
                    MergeBatch(self.device)
                ]
            ),
            "test":  ArticlesDataset(
                self.dataframe,
                "test",
                transforms=[
                    TextToParsedDoc(self.vocabulary.nlp),
                    WordsToVectors(self.vocabulary.nlp),
                    WordsToIds(self.vocabulary),
                    AddNoiseToEmbeddings(0),
                    SetAllToSummarizing(),
                    MergeBatch(self.device)
                ]
            ),
            "val":  ArticlesDataset(
                self.dataframe,
                "val",
                transforms=[
                    TextToParsedDoc(self.vocabulary.nlp),
                    WordsToVectors(self.vocabulary.nlp),
                    WordsToIds(self.vocabulary),
                    AddNoiseToEmbeddings(0),
                    MergeBatch(self.device)
                ]
            )
        }

    def compute_loss(self, word_embeddings, original_word_embeddings, discriminate_probs):
        embeddings_loss = F.cosine_embedding_loss(
          word_embeddings.reshape((-1, word_embeddings.shape[2])),
          original_word_embeddings.reshape((-1, original_word_embeddings.shape[2])),
          torch.ones(word_embeddings.shape[0] * word_embeddings.shape[1]).to(self.device)
        )

        discriminator_loss = F.binary_cross_entropy(
            discriminate_probs,
            torch.zeros_like(discriminate_probs).to(self.device)
        )

        return embeddings_loss + discriminator_loss


    def work_batch(self, batch):
        logits, discriminate_probs = self.model(
            batch.noisy_word_embeddings,
            batch.mode
        )

        return (
            self.compute_loss(logits, batch.word_embeddings, discriminate_probs),
            word_embeddings
        )
