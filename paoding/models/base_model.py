import argparse
from collections import defaultdict
from typing import Any

from allennlp.training.metrics import Metric
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW

from paoding.data.collator import collate_fn
from paoding.data.dataset import Dataset
from paoding.data.sortish_sampler import make_sortish_sampler


class BaseModel(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters()
        # pytorch-lightning calls this, but we call it ourselves here in case the __init__ of
        # children modules need dataset attributes, e.g., num_labels
        self.prepare_data()
        self.metrics = self.setup_metrics()

    def prepare_data(self):
        self.dataset = Dataset(self.hparams)

    def _get_dataloader(self, split: str, batch_size: int, shuffle=False) -> DataLoader:
        dataset_split = self.dataset[split]
        lens = [len(ids) for ids in dataset_split[self.dataset.sort_key]]
        sampler = make_sortish_sampler(
            lens, batch_size, distributed=self.hparams.gpus > 1, perturb=shuffle
        )
        dataloader = DataLoader(
            dataset_split,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=lambda batch: collate_fn(
                batch,
                self.dataset.tokenizer.pad_token_id,
                self.dataset.tokenizer.pad_token_type_id,
                self.dataset.tokenizer.padding_side,
                self.dataset.output_mode,
            ),
        )
        return dataloader

    def setup(self, stage: str = None):
        """To set up self.dataset_size"""
        if stage != "fit":
            return

        self._train_dataloader = self._get_dataloader(
            "train", self.hparams.batch_size, shuffle=True
        )
        self.dataset_size = len(self._train_dataloader.dataset)

    def setup_metrics(self) -> dict[str, Metric]:
        return {
            split: {name: Metric.by_name(name) for name in self.dataset.metric_names}
            for split in self.dataset.dev_splits + self.dataset.test_splits
        }

    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    def val_dataloader(self, shuffle=False) -> list[DataLoader]:
        return [
            self._get_dataloader(split, self.hparams.eval_batch_size, shuffle=shuffle)
            for split in self.dataset.dev_splits
        ]

    def test_dataloader(self, shuffle=False) -> list[DataLoader]:
        return [
            self._get_dataloader(split, self.hparams.eval_batch_size, shuffle=shuffle)
            for split in self.dataset.test_splits
        ]

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

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.dataset.output_mode == "classification":
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        elif self.dataset.output_mode == "regression":
            loss = F.mse_loss(logits.view(-1), labels.view(-1))
        else:
            raise KeyError(f"Output mode not supported: {self.dataset.output_mode}")
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        self.log("lr", self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1], prog_bar=True)
        return {"loss": self.compute_loss(self(batch)["logits"], batch["labels"])}

    def eval_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, mode: str, dataloader_idx=0
    ) -> dict[str, Any]:
        assert mode in {"dev", "test"}

        logits = self(batch)["logits"]
        labels = batch["labels"]
        assert logits.dim() == 2 and labels.dim() == 1

        if self.dataset.output_mode == "classification":
            preds = logits.argmax(dim=1)
        elif self.dataset.output_mode == "regression":
            preds = logits.squeeze(dim=1)
        else:
            raise KeyError(f"Output mode not supported: {self.dataset.output_mode}")

        splits = self.dataset.dev_splits if mode == "dev" else self.dataset.test_splits
        split = splits[dataloader_idx]
        for metric in self.metrics[split].values():
            metric(*metric.detach_tensors(preds, labels))

        return {"loss": self.compute_loss(logits, labels).detach().cpu()}

    def eval_epoch_end(self, outputs: list[dict[str, Any]], mode: str):
        assert isinstance(outputs, list)
        assert mode in {"dev", "test"}

        # We gather individual metrics from each dataloader and compute the average if there is
        # more than one
        more_than_1 = len(outputs) > 1
        if more_than_1:
            sums = defaultdict(int)
        for i in range(len(outputs)):
            split = (self.dataset.dev_splits if mode == "dev" else self.dataset.test_splits)[i]
            assert split != "avg"  # reserved keyword for below
            metrics = self.get_metrics(split, reset=True)
            for k, v in metrics.items():
                if more_than_1:
                    self.log(f"{k}_{split}", v)
                    sums[k] += v
                else:
                    self.log(k, v)
        if more_than_1:
            for k, v in sums.items():
                self.log(f"{k}_avg", v / len(outputs))

    def get_metrics(self, split: str, reset=False) -> dict[str, Any]:
        metrics = {name: metric.get_metric() for name, metric in self.metrics[split].items()}
        if reset:
            for metric in self.metrics[split].values():
                metric.reset()
        return metrics

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx=0
    ) -> dict[str, Any]:
        return self.eval_step(batch, batch_idx, "dev", dataloader_idx=dataloader_idx)

    def validation_epoch_end(self, outputs: list[dict[str, Any]]):
        return self.eval_epoch_end(outputs, "dev")

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx=0
    ) -> dict[str, Any]:
        return self.eval_step(batch, batch_idx, "test", dataloader_idx=dataloader_idx)

    def test_epoch_end(self, outputs: list[dict[str, Any]]):
        return self.eval_epoch_end(outputs, "test")

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--gradient_clip_val", default=1.0, type=float)
        parser.add_argument("--accumulate_grad_batches", default=1, type=int)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--epochs", default=3, type=int, dest="max_epochs")
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
