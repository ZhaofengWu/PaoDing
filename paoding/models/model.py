import argparse
from collections import defaultdict
from typing import Any, Union

from allennlp.training.metrics import Metric
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW, PreTrainedTokenizerBase


class Model(pl.LightningModule):
    def __init__(self, hparams: Union[argparse.Namespace, dict]):
        super().__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.save_hyperparameters(hparams)
        # pytorch-lightning calls this, but we call it ourselves here in case the __init__ of
        # children modules need dataset attributes, e.g., num_labels
        self.prepare_data()
        self.metrics = self.setup_metrics()

    def setup_tokenizer(self) -> PreTrainedTokenizerBase:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def prepare_data(self):
        self.tokenizer = self.setup_tokenizer()
        self.dataset = self.hparams.dataset_class(self.hparams, self.tokenizer)

    def setup(self, stage: str = None):
        """To set up self.dataset_size"""
        if stage != "fit":
            return

        # TODO: maybe simply get len(dataset) so we don't have to create a dataloader?
        self._train_dataloader = self.dataset.dataloader(
            "train", self.hparams.batch_size, shuffle=True
        )
        self.dataset_size = len(self._train_dataloader.dataset)

    def setup_metrics(self) -> dict[str, dict[str, Metric]]:
        return {
            split: {name: Metric.by_name(name)() for name in self.dataset.metric_names}
            for split in (
                [self.dataset.train_split] + self.dataset.dev_splits + self.dataset.test_splits
            )
        }

    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    def val_dataloader(self, shuffle=False) -> list[DataLoader]:
        return [
            self.dataset.dataloader(split, self.hparams.eval_batch_size, shuffle=shuffle)
            for split in self.dataset.dev_splits
        ]

    def test_dataloader(self, shuffle=False) -> list[DataLoader]:
        return [
            self.dataset.dataloader(split, self.hparams.eval_batch_size, shuffle=shuffle)
            for split in self.dataset.test_splits
        ]

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict]]:
        "Prepare optimizer and schedule (linear warmup and decay)"

        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "layer_norm.weight"]
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
            lr=self.hparams.lr,
            eps=self.hparams.adam_epsilon,
        )
        scheduler = self.get_lr_scheduler(optimizer)

        return [optimizer], [scheduler]

    def get_lr_scheduler(self, optimizer: Optimizer) -> dict:
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        if self.hparams.lr_scheduler_total_steps is not None:
            total_steps = self.hparams.lr_scheduler_total_steps
        else:
            effective_batch_size = (
                self.hparams.batch_size * self.hparams.accumulate_grad_batches * num_devices
            )
            # Sometimes dataset_size could be smaller than the effective_batch_size
            total_steps = max(self.dataset_size / effective_batch_size, 1) * self.hparams.epochs

        if self.hparams.warmup_steps > 0 and self.hparams.warmup_ratio > 0:
            raise ValueError("--warmup_steps and --warmup_ratio are mutually exclusive.")
        warmup_steps = (
            int(total_steps * self.hparams.warmup_ratio)
            if self.hparams.warmup_ratio > 0
            else self.hparams.warmup_steps
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        return {"scheduler": scheduler, "interval": "step", "frequency": 1}

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int
    ):
        """See https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"""
        optimizer.zero_grad(set_to_none=True)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor = None, reduce=True
    ) -> torch.Tensor:
        if self.dataset.output_mode == "classification":
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        elif self.dataset.output_mode == "token_classification":
            assert mask.any(dim=-1).all()
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
            )
            loss = loss.view_as(labels) * mask
            if reduce:
                loss = loss.sum() / mask.sum()
        elif self.dataset.output_mode == "regression":
            loss = F.mse_loss(logits.view(-1), labels.view(-1))
        else:
            raise KeyError(f"Output mode not supported: {self.dataset.output_mode}")
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        loss = self.compute_loss(
            self(batch)["logits"], batch[self.dataset.label_key], batch.get("label_mask")
        )
        self.log("train_loss", loss)
        self.log("lr", self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1], prog_bar=True)
        return {"loss": loss}

    def get_predictions(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.dataset.output_mode in ("classification", "token_classification"):
            return logits.argmax(dim=-1)
        elif self.dataset.output_mode == "regression":
            return logits.squeeze(dim=-1)
        else:
            raise KeyError(f"Output mode not supported: {self.dataset.output_mode}")

    def eval_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        split: str,
        compute_loss=True,
    ) -> dict[str, Any]:
        logits = self(batch)["logits"]
        preds = self.get_predictions(logits, batch)
        labels = batch[self.dataset.label_key]

        for metric in self.metrics[split].values():
            metric(*metric.detach_tensors(preds, labels))

        loss_dict = (
            {"loss": self.compute_loss(logits, labels, batch.get("label_mask")).detach().cpu()}
            if compute_loss
            else {}
        )
        return loss_dict | {
            "preds": preds.detach().cpu().numpy(),
            "labels": labels.detach().cpu().numpy(),
        }

    def eval_epoch_end(self, splits: list[str], outputs: list[list[dict[str, Any]]], set_preds_labels=False):
        def safe_setattr(obj, k, v):
            assert not hasattr(obj, k)
            setattr(obj, k, v)

        num_splits = len(splits)
        # We gather individual metrics from each dataloader and compute the average if there is
        # more than one
        if num_splits > 1:
            sums = defaultdict(int)

        assert len(splits) == len(outputs)
        for split, split_outputs in zip(splits, outputs):
            metrics = self.get_metrics(split, reset=True)
            for k, v in metrics.items():
                if num_splits > 1:
                    assert f"{k}_{split}" not in metrics
                    self.log(f"{k}_{split}", v)
                    sums[k] += v
                else:
                    self.log(k, v)

            if set_preds_labels:
                assert num_splits == 1  # set_preds_labels iff testing iff only one split
                preds = np.concatenate([output["preds"] for output in split_outputs], axis=0)
                labels = np.concatenate([output["labels"] for output in split_outputs], axis=0)
                # Hack: see https://github.com/PyTorchLightning/pytorch-lightning/issues/12969
                safe_setattr(self, "_preds", preds)
                safe_setattr(self, "_labels", labels)
        if num_splits > 1:
            for k, v in sums.items():
                # It's important to keep the aggregate metric to be the original name, since it is
                # the sort key.
                self.log(k, v / num_splits)

    def get_metrics(self, split: str, reset=False) -> dict[str, Any]:
        metrics = {name: metric.get_metric() for name, metric in self.metrics[split].items()}
        if reset:
            for metric in self.metrics[split].values():
                metric.reset()
        return metrics

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx=0
    ) -> dict[str, Any]:
        return self.eval_step(batch, batch_idx, self.dataset.dev_splits[dataloader_idx])

    def validation_epoch_end(
        self, outputs: Union[list[list[dict[str, Any]]], list[dict[str, Any]]]
    ):
        # pytorch-lightning "conveniently" unwraps the list when there's only one dataloader,
        # so we need a check here.
        self.eval_epoch_end(
            self.dataset.dev_splits, [outputs] if isinstance(outputs[0], dict) else outputs
        )

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx=0
    ) -> dict[str, Any]:
        # self.current_eval_split is set in evaluate.py
        return self.eval_step(batch, batch_idx, self.current_test_split)

    def test_epoch_end(self, outputs: list[dict[str, Any]]):
        # pytorch-lightning "conveniently" unwraps the list when there's only one dataloader,
        # so the first element is a dict iff there's only one split
        assert isinstance(outputs[0], dict)
        # self.current_eval_split is set in evaluate.py
        self.eval_epoch_end([self.current_test_split], [outputs], set_preds_labels=True)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("--lr", default=2e-5, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--clip_norm", default=0.0, type=float)
        parser.add_argument("--accumulate_grad_batches", default=1, type=int)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--warmup_ratio", default=0.0, type=float)
        parser.add_argument("--lr_scheduler_total_steps", default=None, type=int)
        parser.add_argument("--epochs", default=3, type=int)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
