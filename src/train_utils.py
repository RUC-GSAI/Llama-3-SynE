# encoding: utf-8
import torch
import datasets
import transformers
from typing import Dict, Union
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
from transformers import Trainer
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available


class NoShuffleSeq2SeqTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # dataloader_params["sampler"] = SequentialSampler(self.train_dataset) # Original
            dataloader_params["sampler"] = self._get_eval_sampler(self.train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["shuffle"] = False

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


def get_wsd_scheduler(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1, stable_ratio=1.0
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        num_stable_steps = stable_ratio * num_training_steps
        if current_step < num_stable_steps:
            return 1.0
        return max(
            0.1,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_stable_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class WSDTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_wsd_scheduler(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
            self._created_lr_scheduler = True
            print("Using WSD scheduler")
        return self.lr_scheduler

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        optimizer = self.optimizer
        self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=optimizer
        )
        print("Scheduler", self.lr_scheduler)


class WSDNoShuffleTrainer(NoShuffleSeq2SeqTrainer, WSDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
