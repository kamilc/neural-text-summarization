from cached_property import cached_property
import torch
import os
from pathlib import Path

from lib.batch import Batch
from lib.update_info import UpdateInfo
from lib.dataloading.dataloader import DataLoader

class BaseTrainer:
    def __init__(self, name, dataframe,
                 optimizer_class_name,
                 model_args, optimizer_args,
                 batch_size, update_every,
                 device
                ):
        self.name = name

        self.dataframe = dataframe
        self.device = device
        self.batch_size = batch_size
        self.update_every = update_every

        self.optimizer_class_name = optimizer_class_name

        self.model_args = model_args
        self.optimizer_args = optimizer_args

        self.current_batch_id = 0

        if self.has_checkpoint:
            self.load_last_checkpoint()

    @property
    def model_class(self):
        pass

    @cached_property
    def model(self):
        try:
            return self.model_class.load(f"{self.checkpoint_path}/model.pth").to(self.device)
        except FileNotFoundError:
            return self.model_class(self.device, **self.model_args).to(self.device)

    @cached_property
    def optimizer(self):
        class_ = getattr(torch.optim, self.optimizer_class_name)

        return class_(self.model.parameters(), **self.optimizer_args)

    @property
    def checkpoint_path(self):
        return f"checkpoints/{self.name}/batch-#{self.current_batch_id}"

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.model.save(f"{self.checkpoint_path}/model.pth")

        torch.save(
            {
                'current_batch_id': self.current_batch_id,
                'batch_size': self.batch_size,
                'update_every': self.update_every,
                'optimizer_class_name': self.optimizer_class_name,
                'optimizer_args': self.optimizer_args,
                'optimizer_state_dict': self.optimizer.state_dict()
            },
            f"{self.checkpoint_path}/trainer.pth"
        )

    @property
    def checkpoint_directories(self):
        return sorted(Path(".").glob(f"checkpoints/{self.name}/batch-*"), reverse=True)

    @property
    def has_checkpoint(self):
        return len(self.checkpoint_directories) > 0

    def load_last_checkpoint(self):
        path = self.checkpoint_directories[0]

        data = torch.load(f"{path}/trainer.pth")

        self.batch_size = data['batch_size']
        self.update_every = data['update_every']

        self.optimizer_class_name = data['optimizer_class_name']
        self.optimizer_args = data['optimizer_args']

        self.current_batch_id = data['current_batch_id']

        if 'model' in self.__dict__:
            del self.__dict__['model']

        if 'optimzer' in self.__dict__:
            del self.__dict__['optimizer']

        self.optimizer.load_state_dict(data['optimizer_state_dict'])

    def batches(self, mode):
        while True:
            loader = DataLoader(
                self.datasets[mode],
                batch_size=self.batch_size
            )

            for data in loader:
                self.current_batch_id += 1

                yield(
                    Batch(
                        data,
                        ix=self.current_batch_id
                    )
                )

    def work_batch(self, batch):
        raise NotImplementedError

    def updates(self, mode="train", update_every=None):
        batches = self.batches(mode)
        loss_sum = 0

        if update_every is None:
            update_every = self.update_every

        for batch in batches:
            if mode == "train":
                self.model.train()
            else:
                self.model.eval()

            loss, word_embeddings = self.work_batch(batch)
            loss /= self.update_every

            if mode == "train":
                loss.backward()

            loss_sum += loss

            # we're doing the accumulated gradients trick to get the gradients variance
            # down while being able to use commodity GPU:
            if batch.ix % update_every == 0:
                if mode == "train":
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                yield(UpdateInfo(self.decoder, batch, word_embeddings, loss_sum, mode=mode))

                loss_sum = 0

    def train_and_evaluate_updates(self, evaluate_every=100):
        train_updates = self.updates(mode="train")
        evaluate_updates = self.updates(mode="val")

        for update_info in train_updates:
            yield(update_info)

            if update_info.batch.ix != 0 and update_info.batch.ix % evaluate_every == 0:
                for _ in range(0, 10):
                    yield(next(evaluate_updates))

    def test_updates(self):
        return self.updates(mode="test", update_every=1)
