from torch.utils.tensorboard import SummaryWriter
from cached_property import cached_property
from summarize.trainer import Trainer
from lib.metrics import Metrics
import torch

class TensorboardTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(TensorboardTrainer, self).__init__(*args, **kwargs)

        self.writer = SummaryWriter(log_dir=f"runs/{self.name}")

    def train(self, evaluate_every=100):
        test_updates = self.test_updates()

        cumulative_train_metrics = Metrics.empty(mode="train")
        cumulative_evaluate_metrics = Metrics.empty(mode="eval")

        for update_info in self.train_and_evaluate_updates(evaluate_every=evaluate_every):
            if update_info.from_train:
                cumulative_train_metrics += update_info.metrics

                print(f"{update_info.batch.ix} \t| {update_info.metrics.loss} \t= {update_info.model_loss} \t+ {update_info.fooling_loss} \t| {update_info.discriminator_loss}")

                if update_info.batch.ix % 200 == 0:
                    with torch.no_grad():
                        predicted = update_info.decoded_inferred_texts[0].replace('\n', ' ').strip('❟ ❟ ❟')
                        headline = update_info.batch.orig_headline[0].replace('\n', ' ').lower().strip()
                        text = update_info.batch.orig_text[0].replace('\n', ' ').lower().strip()
                        print(f"{update_info.batch.ix}\n\nTEXT:\n{text} \n\nHEADLINE:\n{headline} \n\nPREDICTED SUMMARY:\n{predicted}")

                if update_info.batch.ix % 10 == 0:
                    self.writer.add_scalar(
                        'loss/train',
                        cumulative_train_metrics.loss,
                        update_info.batch.ix
                    )

                    self.writer.add_scalar(
                        'model-loss/train',
                        cumulative_train_metrics.loss,
                        update_info.batch.ix
                    )

                    self.writer.add_scalar(
                        'fooling-loss/train',
                        cumulative_train_metrics.loss,
                        update_info.batch.ix
                    )

                    cumulative_train_metrics = Metrics.empty(mode="train")

            if update_info.from_evaluate:
                cumulative_evaluate_metrics += update_info.metrics

                if len(cumulative_evaluate_metrics) == 10:
                    with torch.no_grad():
                        predicted = update_info.decoded_inferred_texts[0].replace('\n', ' ').strip('❟ ❟ ❟')
                        headline = update_info.batch.orig_headline[0].replace('\n', ' ').lower().strip()
                        text = update_info.batch.orig_text[0].replace('\n', ' ').lower().strip()
                        print(f"{update_info.batch.ix}\n\nEVAL TEXT:\n{text} \n\nEVAL HEADLINE:\n{headline} \n\nEVAL PREDICTED SUMMARY:\n{predicted}")

                        self.writer.add_text(
                            'text/eval',
                            text,
                            int(update_info.batch.ix / evaluate_every)
                        )

                        self.writer.add_text(
                            'headline/eval',
                            headline,
                            int(update_info.batch.ix / evaluate_every)
                        )

                        self.writer.add_text(
                            'predicted/eval',
                            predicted,
                            int(update_info.batch.ix / evaluate_every)
                        )

                    self.writer.add_scalar(
                        'rouge-1/eval',
                        cumulative_evaluate_metrics.rouge_score,
                        int(update_info.batch.ix / evaluate_every)
                    )

                    cumulative_evaluate_metrics = Metrics.empty(mode="eval")

                print(f"Eval: {update_info.metrics.loss}")

            if update_info.batch.ix % 600 == 0 and update_info.batch.ix != 0:
                print(f"Saving checkpoint")
                self.save_checkpoint()

    def test(self):
        cumulative_metrics = Metrics.empty(mode="test")

        for update_info in self.test_updates():
            cumulative_metrics += update_info.metrics

        print(cumulative_metrics)
