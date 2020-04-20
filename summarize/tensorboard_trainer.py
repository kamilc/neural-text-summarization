from torch.utils.tensorboard import SummaryWriter
from cached_property import cached_property

class TensorboardTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SummarizeTrainer, self).__init__(*args, **kwargs)

        self.writer = SummaryWriter(comment=self.name)

    def train(self, evaluate_every=1000):
        test_updates = self.test_updates()

        cumulative_train_metrics = Metrics.empty(mode="train")
        cumulative_evaluate_metrics = Metrics.empty(mode="eval")

        for update_info in self.train_and_evaluate_updates(evaluate_every=evaluate_every):
            if update_info.from_train:
                cumulative_train_metrics += update_info.metrics

                print(f"{update_info.batch.ix}")

                self.writer.add_scalar(
                    'loss/train',
                    update_info.metrics.loss,
                    update_info.batch.ix
                )

            if update_info.from_evaluate:
                cumulative_evaluate_metrics += update_info.metrics

                self.writer.add_scalar(
                    'loss/eval',
                    update_info.metrics.loss,
                    update_info.batch.ix
                )

                print(f"Eval: {update_info.metrics.loss}")
                print(f"Saving checkpoint")
                self.save_checkpoint()

#             if update_info.batch.ix % 1000 == 0 and update_info.batch.ix != 0:
#                 test_update = next(test_updates)

#                 self.test_texts_stream.write(
#                     (
#                         update_info.batch.text,
#                         update_info.decoded_inferred_texts
#                     )
#                 )

    def test(self):
        cumulative_metrics = Metrics.empty(mode="test")

        for update_info in self.test_updates():
            cumulative_metrics += update_info.metrics

        print(cumulative_metrics)
