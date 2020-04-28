from torch.utils.tensorboard import SummaryWriter
from cached_property import cached_property
from summarize.trainer import Trainer
from lib.metrics import Metrics

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

                print(f"{update_info.batch.ix} => {update_info.metrics.loss} ({update_info.batch.word_embeddings.shape})") # {[ len(t) for t in update_info.batch.text ]}")

                if update_info.batch.ix % 10 == 0:
                    self.writer.add_scalar(
                        'loss/train',
                        cumulative_train_metrics.running_mean_loss(),
                        update_info.batch.ix
                    )

                    cumulative_train_metrics = Metrics.empty(mode="train")

            if update_info.from_evaluate:
                cumulative_evaluate_metrics += update_info.metrics

                if len(cumulative_evaluate_metrics) == 10:
                    self.writer.add_scalar(
                        'loss/eval',
                        cumulative_evaluate_metrics.loss,
                        int(update_info.batch.ix / evaluate_every)
                    )
                    cumulative_evaluate_metrics = Metrics.empty(mode="eval")

                print(f"Eval: {update_info.metrics.loss}")

            if update_info.batch.ix % 100 == 0 and update_info.batch.ix != 0:
                print(f"Saving checkpoint")
                self.save_checkpoint()

            if update_info.batch.ix % 100 == 0 and update_info.batch.ix != 0:
                test_update = next(test_updates)

                text = next(update_info.decoded_inferred_texts)

                print(f"TEST at {update_info.batch.ix}\n\nORIGINAL:\n{update_info.batch.text[0].strip().lower()} \n\nPREDICTED SUMMARY:\n{text}")

                self.writer.add_text(
                    'test/text',
                    text,
                    update_info.batch.ix
                )

    def test(self):
        cumulative_metrics = Metrics.empty(mode="test")

        for update_info in self.test_updates():
            cumulative_metrics += update_info.metrics

        print(cumulative_metrics)
