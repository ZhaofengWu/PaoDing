import argparse

import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW


class BaseModel(pl.LightningModule):
    def __init__(self, hparam: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, mode: str) -> None:
        """To set up self.dataset_size"""
        if mode != "train":
            return

        self._train_dataloader = self.get_dataloader("train", self.hparams.batch_size, shuffle=True)
        self.dataset_size = len(self._train_dataloader.dataset)

    def get_dataloader(self, split: str, batch_size: int, shuffle=False) -> DataLoader:
        raise NotImplementedError("You must implement this for your task")

    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    def val_dataloader(self, shuffle=False) -> list[DataLoader]:
        dataloaders = [self.get_dataloader("dev", self.args.eval_batch_size, shuffle=shuffle)]
        if self.has_secondary_split:
            dataloaders.append(
                self.get_dataloader("dev2", self.args.eval_batch_size, shuffle=shuffle)
            )
        return dataloaders

    def test_dataloader(self, shuffle=False) -> list[DataLoader]:
        dataloaders = [self.get_dataloader("test", self.args.eval_batch_size, shuffle=shuffle)]
        if self.has_secondary_split:
            dataloaders.append(
                self.get_dataloader("test2", self.args.eval_batch_size, shuffle=shuffle)
            )
        return dataloaders

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict]]:
        "Prepare optimizer and schedule (linear warmup and decay)"

        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "layer_norm.weight"]
        assert self.named_parameters == self.model.named_parameters()
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        scheduler = self.get_lr_scheduler(optimizer)

        return [optimizer], [scheduler]

    def get_lr_scheduler(self, optimizer: Optimizer) -> dict:
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = (
            self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        )
        total_steps = (self.dataset_size / effective_batch_size) * self.hparams.epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=total_steps
        )
        return {"scheduler": scheduler, "interval": "step", "frequency": 1}

    def test_step(self, batch, batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    @staticmethod
    def add_model_specific_args(parser: argparse.ArgumentParser):
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--max_epochs", default=3, type=int)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
