from cached_property import cached_property
import torch
import torch.nn.functional as F
import os
import numpy as np
from pathlib import Path

from lib.batch import Batch
from lib.update_info import UpdateInfo
from lib.dataloading.dataloader import DataLoader

class BaseTrainer:
    def __init__(self, name, dataframe, dataset_class_name,
                 optimizer_class_name,
                 discriminator_optimizer_class_name,
                 model_args, optimizer_args,
                 discriminator_args, discriminator_optimizer_args,
                 vocabulary,
                 batch_size,
                 device,
                 no_period_trick = False
                ):
        self.name = name

        self.vocabulary = vocabulary
        self.dataframe = dataframe
        self.dataset_class_name = dataset_class_name
        self.device = device
        self.batch_size = batch_size

        self.no_period_trick = no_period_trick

        self.optimizer_class_name = optimizer_class_name
        self.discriminator_optimizer_class_name = discriminator_optimizer_class_name

        self.model_args = model_args
        self.model_args['vocabulary_size'] = len(vocabulary)
        self.optimizer_args = optimizer_args

        self.discriminator_args = discriminator_args
        self.discriminator_optimizer_args = discriminator_optimizer_args

        self.current_batch_id = 0

        if self.has_checkpoint:
            self.load_last_checkpoint()

    @property
    def model_class(self):
        pass

    @property
    def discriminator_class(self):
        pass

    @cached_property
    def model(self):
        try:
            return self.model_class.load(f"{self.checkpoint_path}/model.pth").to(self.device)
        except FileNotFoundError:
            return self.model_class(
                self.device,
                **self.model_args
            ).to(self.device)

    @cached_property
    def discriminator(self):
        try:
            return self.discriminator_class.load(f"{self.checkpoint_path}/discriminator.pth").to(self.device)
        except FileNotFoundError:
            return self.discriminator_class(self.device, **self.discriminator_args).to(self.device)

    @cached_property
    def optimizer(self):
        class_ = getattr(torch.optim, self.optimizer_class_name)

        print(f"{self.optimizer_args}")

        return class_(self.model.parameters(), **self.optimizer_args)

    @cached_property
    def discriminator_optimizer(self):
        class_ = getattr(torch.optim, self.discriminator_optimizer_class_name)

        return class_(self.discriminator.parameters(), **self.discriminator_optimizer_args)

    @property
    def checkpoint_path(self):
        return f"checkpoints/{self.name}/batch-#{self.current_batch_id}"

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.model.save(f"{self.checkpoint_path}/model.pth")
        self.discriminator.save(f"{self.checkpoint_path}/discriminator.pth")

        torch.save(
            {
                'current_batch_id': self.current_batch_id,
                'batch_size': self.batch_size,
                'optimizer_class_name': self.optimizer_class_name,
                'optimizer_args': self.optimizer_args,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'discriminator_optimizer_class_name': self.discriminator_optimizer_class_name,
                'discriminator_optimizer_args': self.discriminator_optimizer_args,
                'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict()
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

    def work_model(self, batch):
        raise NotImplementedError

    def work_discriminator(self, state, batch):
        raise NotImplementedError

    def updates(self, mode="train"):
        print(f"Batch size: {self.batch_size}")

        batches = self.batches(mode)

        for batch in batches:
            if mode == "train":
                self.model.train()
                self.discriminator.train()
            else:
                self.model.eval()
                self.discriminator.eval()

            self.optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()

            logits, state = self.model(
                batch.word_embeddings.to(self.device),
                batch.clean_word_embeddings.to(self.device),
                batch.lengths.to(self.device),
                batch.mode.to(self.device)
            )

            mode_probs_disc = self.discriminator(state.detach())
            mode_probs = self.discriminator(state)

            discriminator_loss = F.binary_cross_entropy(
                mode_probs_disc,
                batch.mode
            )

            discriminator_loss.backward(retain_graph=True)

            if mode == "train":
                self.discriminator_optimizer.step()

            text = batch.text.copy()

            if self.no_period_trick:
                text = [txt.replace('.', '') for txt in text]

            classes = self.vocabulary.encode(text, modes=batch.mode)
            classes = classes.roll(-1, dims=1)
            classes[:,classes.shape[1]-1] = 3

            model_loss = torch.tensor(0).cuda()

            if logits.shape[0:2] == classes.shape:
                model_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[2]).to(self.device),
                    classes.long().reshape(-1).to(self.device),
                    ignore_index=3
                )
            else:
                print(f"WARNING: Skipping model loss for inconsistency between logits and classes shapes")

            fooling_loss = F.binary_cross_entropy(
                mode_probs,
                torch.ones_like(batch.mode).to(self.device)
            )

            loss = model_loss + (0.1 * fooling_loss)

            loss.backward()
            if mode == "train":
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()

            yield(
                UpdateInfo(
                    self.model,
                    self.vocabulary,
                    batch,
                    logits,
                    [
                        loss.cpu().item(),
                        discriminator_loss.cpu().item(),
                        model_loss.cpu().item(),
                        fooling_loss.cpu().item()
                    ],
                    mode=mode,
                    no_period_trick=self.no_period_trick
                )
            )

    def train_and_evaluate_updates(self, evaluate_every=100):
        train_updates = self.updates(mode="train")
        evaluate_updates = self.updates(mode="val")

        for update_info in train_updates:
            yield(update_info)

            if update_info.batch.ix != 0 and update_info.batch.ix % evaluate_every == 0:
                for _ in range(0, 10):
                    yield(next(evaluate_updates))

    def test_updates(self):
        return self.updates(mode="test")
