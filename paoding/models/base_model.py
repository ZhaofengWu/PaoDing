import argparse
from collections import defaultdict
from typing import Any, Optional, Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.apply_func import apply_to_collection
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
    def __init__(self, hparam: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters()
        # pytorch-lightning calls this, but we call it ourselves here in case the __init__ of
        # children modules need dataset attributes, e.g., num_labels
        self.prepare_data()

    @property
    def pad_token_id(self):
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def pad_token_type_id(self):
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def padding_side(self):
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

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
                self.pad_token_id,
                self.pad_token_type_id,
                self.padding_side,
                self.dataset.output_mode,
            ),
        )
        return dataloader

    def setup(self, stage: Optional[str] = None) -> None:
        """To set up self.dataset_size"""
        if stage != "fit":
            return

        self._train_dataloader = self._get_dataloader("train", self.hparams.batch_size, shuffle=True)
        self.dataset_size = len(self._train_dataloader.dataset)

    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    def val_dataloader(self, shuffle=False) -> list[DataLoader]:
        dataloaders = [self._get_dataloader("dev", self.args.eval_batch_size, shuffle=shuffle)]
        if self.has_secondary_split:
            dataloaders.append(
                self._get_dataloader("dev2", self.args.eval_batch_size, shuffle=shuffle)
            )
        return dataloaders

    def test_dataloader(self, shuffle=False) -> list[DataLoader]:
        dataloaders = [self._get_dataloader("test", self.args.eval_batch_size, shuffle=shuffle)]
        if self.has_secondary_split:
            dataloaders.append(
                self._get_dataloader("test2", self.args.eval_batch_size, shuffle=shuffle)
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

    def step(self, batch) -> dict[str, Any]:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def compute_loss(self, logits, labels):
        if self.dataset.output_mode == "classification":
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        elif self.dataset.output_mode == "regression":
            loss = F.mse_loss(logits.view(-1), labels.view(-1))
        else:
            raise KeyError(f"Output mode not supported: {self.dataset.output_mode}")
        return loss

    def training_step(self, batch, batch_idx) -> dict[str, Any]:
        self.log("lr", self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1], prog_bar=True)
        return {"loss": self.compute_loss(self._step(batch)["logits"], batch["labels"])}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        output = {
            "logits": self._step(batch)["logits"],
            "labels": batch["labels"],
        }
        return apply_to_collection(output, torch.Tensor, lambda t: t.detach.cpu())

    def compute_metrics(self, outputs: dict[str, Any]) -> dict[str, Union[int, float]]:
        # TODO: is this a list of dict? np.array or tensor?
        logits = np.concatenate([output["logits"] for output in outputs], axis=0)
        if self.dataset.output_mode == "classification":
            preds = np.argmax(logits, axis=1)
        elif self.dataset.output_mode == "regression":
            preds = np.squeeze(logits)
        else:
            raise KeyError(f"Output mode not supported: {self.dataset.output_mode}")

        # TODO: here too
        labels = np.concatenate([output["labels"] for output in outputs], axis=0)
        return self.dataset.compute_metrics(predictions=preds, references=labels)

    def validation_epoch_end(self, outputs: Union[dict[str, Any], list[dict[str, Any]]]):
        logs = {}
        if isinstance(outputs, list):
            assert len(outputs) > 1
            # Each dataloader has its own individual metrics suffixed with its index,
            # and we also aggregate (average) metrics across dataloaders
            sums = defaultdict(int)
            for i, one_split_outputs in enumerate(outputs):
                metrics = self.compute_metrics(one_split_outputs)
                for k, v in metrics.items():
                    logs[k + str(i + 1)] = v
                    sums[k] += v
            logs.update({k: v / len(outputs) for k, v in sums.items()})
        else:
            logs.update(self.compute_metrics(outputs))

        for k, v in logs.items():
            self.log(k, v)

    def test_step(self, batch, batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    @staticmethod
    def add_model_specific_args(parser: argparse.ArgumentParser):
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--gradient_clip_val", default=1.0, type=float)
        parser.add_argument("--accumulate_grad_batches", default=1, type=int)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--epochs", default=3, type=int, dest="max_epochs")
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
