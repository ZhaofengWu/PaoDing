import argparse
from collections import defaultdict
from typing import Any

from lightning_utilities.core.apply_func import apply_to_collection
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
import torchmetrics
from torchmetrics import Metric
from transformers import get_linear_schedule_with_warmup
from transformers import PreTrainedTokenizerBase

from paoding.argument_parser import ArgumentParser
from paoding.data.tokenizer import Tokenizer


class Model(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace | dict):
        super().__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.save_hyperparameters(hparams)

        if self.hparams.eval_batch_size is None:
            self.hparams.eval_batch_size = self.hparams.batch_size
        # pytorch-lightning calls this, but we call it ourselves here in case the __init__ of
        # children modules need dataset attributes, e.g., num_labels
        self._data_is_prepared = False
        self.prepare_data()
        self.metrics = self.setup_metrics()

    def setup_tokenizer(self) -> PreTrainedTokenizerBase:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def prepare_data(self):
        if self._data_is_prepared:
            return
        self.tokenizer = self.setup_tokenizer()
        self.dataset = self.hparams.dataset_class(self.hparams, self.tokenizer)
        if isinstance(self.tokenizer, Tokenizer):
            self.tokenizer.prepare(self.dataset)
        self._data_is_prepared = True

    def setup(self, stage: str = None):
        """To set up self.dataset_size"""
        if stage != "fit":
            return

        # TODO: maybe simply get len(dataset) so we don't have to create a dataloader?
        self._train_dataloader = self.dataset.dataloader(
            self.dataset.train_split, self.hparams.batch_size, shuffle=True
        )
        self.dataset_size = len(self._train_dataloader.dataset)

    @property
    def metric_init_kwargs(self) -> dict[str, Any]:
        kwargs = {}
        if self.dataset.task == "classification":
            kwargs["num_classes"] = self.dataset.num_labels
        if self.dataset.task == "causal_lm":
            # TODO: this won't work for gpt2 now. See https://github.com/Lightning-AI/metrics/issues/54
            assert self.tokenizer.pad_token_id is not None
            kwargs["ignore_index"] = self.tokenizer.pad_token_id
        return kwargs

    def setup_metrics(self) -> dict[str, dict[str, Metric]]:
        metric_splits = (
            [self.dataset.train_split]
            + self.dataset.dev_splits
            + self.dataset.test_splits
            + ["aggregate"]
        )
        return {
            split: {
                name: getattr(torchmetrics, name)(**self.metric_init_kwargs)
                for name in self.dataset.metric_names
            }
            for split in metric_splits
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
        optimizer = torch.optim.AdamW(
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

    def on_fit_start(self):
        apply_to_collection(self.metrics, Metric, lambda metric: metric.to(self.device))

    def on_validation_start(self):
        apply_to_collection(self.metrics, Metric, lambda metric: metric.to(self.device))

    def on_test_start(self):
        apply_to_collection(self.metrics, Metric, lambda metric: metric.to(self.device))

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor = None, reduce=True
    ) -> torch.Tensor:
        match self.dataset.task:
            case "classification":
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            case "token_classification" | "causal_lm":
                assert mask.any(dim=-1).all()
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
                )
                loss = loss.reshape_as(labels) * mask
                if reduce:
                    loss = loss.sum() / mask.sum()
            case "regression":
                loss = F.mse_loss(logits.reshape(-1), labels.reshape(-1))
            case "token_regression" | "token_multi_regression":
                assert mask.any(dim=-1).all()
                loss = F.mse_loss(logits, labels, reduction="none")
                if self.dataset.task == "token_multi_regression":
                    mask = mask.unsqueeze(-1)
                loss = loss.reshape_as(labels) * mask
                if reduce:
                    loss = loss.sum() / mask.sum()
                    if self.dataset.task == "token_multi_regression":
                        loss /= logits.shape[-1]
            case _:
                raise KeyError(f"Output mode not supported: {self.dataset.task}")
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        output = self(batch)
        if "loss" in output:
            loss = output["loss"]
        else:
            loss = self.compute_loss(
                output["logits"],
                batch[self.dataset.label_key],
                batch.get(self.dataset.label_mask_key),
            )
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "lr", self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[-1], prog_bar=True
        )
        return {"loss": loss}

    def get_predictions(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        match self.dataset.task:
            case "classification" | "token_classification":
                return logits.argmax(dim=-1)
            case "regression" | "token_regression":
                return logits.squeeze(dim=-1)
            case "token_multi_regression":
                return logits
            case "causal_lm":
                # Perplexity is a weird metric as it needs the raw distribution... So this is
                # actually not a prediction, and we need to do something else for real decoding.
                # But then, the problem there is more complicated anyway since there's beam search
                # etc.
                return logits
            case _:
                raise KeyError(f"Output mode not supported: {self.dataset.task}")

    def eval_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, split: str, compute_loss=True
    ) -> dict[str, Any]:
        output = self(batch)
        preds = self.get_predictions(output["logits"], batch)
        labels = batch[self.dataset.label_key]

        for s in (split, "aggregate"):
            for metric in self.metrics[s].values():
                # torchmetrics doesn't take a mask which is stupid. See issue
                # https://github.com/Lightning-AI/metrics/issues/1282
                # So we have to flattent these tensors.
                flat_preds = preds
                flat_labels = labels
                if self.dataset.label_mask_key in batch:
                    label_mask = batch[self.dataset.label_mask_key]
                    flat_preds = preds[label_mask]
                    flat_labels = labels[label_mask]
                    if isinstance(metric, torchmetrics.Perplexity):
                        # For other metrics, we merge the batch dim and the seq dim, but perplexity
                        # requires them to be separate.
                        flat_preds = flat_preds.unsqueeze(0)
                        flat_labels = flat_labels.unsqueeze(0)
                metric(flat_preds.detach(), flat_labels.detach())

        return_dict = {
            "preds": preds.detach().cpu().numpy(),
            "labels": labels.detach().cpu().numpy(),
        }
        if compute_loss:
            loss = (
                output["loss"]
                if "loss" in output
                else self.compute_loss(
                    output["logits"], labels, batch.get(self.dataset.label_mask_key)
                )
            )
            return_dict["loss"] = loss.detach().cpu()
        return return_dict

    def eval_epoch_end(self, splits: list[str], outputs: list[list[dict[str, Any]]]):
        num_splits = len(splits)
        # We gather individual metrics from each dataloader and compute the average if there is
        # more than one
        if num_splits > 1:
            sums = defaultdict(int)

        for split, split_outputs in zip(splits, outputs, strict=True):
            metrics = self.get_metrics(split, reset=True)

            if "loss" in split_outputs[0]:
                loss = sum(o["loss"] for o in split_outputs) / len(split_outputs)
                metrics |= {"loss": loss}

            for k, v in metrics.items():
                if num_splits > 1:
                    assert f"{k}_{split}" not in metrics
                    self.log(f"{k}_{split}", v)
                    sums[k] += v
                else:
                    self.log(k, v)

        agg_metrics = self.get_metrics("aggregate", reset=True)
        if num_splits > 1:
            for k, v in sums.items():
                # It's important to keep the aggregate metric to be the original name, since it is
                # the sort key.
                self.log(k, v / num_splits)
            for k, v in agg_metrics.items():
                self.log(k + "_microaggregate", v)

    def get_metrics(self, split: str, reset=False) -> dict[str, Any]:
        metrics = {name: metric.compute() for name, metric in self.metrics[split].items()}
        if reset:
            for metric in self.metrics[split].values():
                metric.reset()
        return metrics

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx=0
    ) -> dict[str, Any]:
        return self.eval_step(batch, batch_idx, self.dataset.dev_splits[dataloader_idx])

    def validation_epoch_end(self, outputs: list[list[dict[str, Any]]] | list[dict[str, Any]]):
        # pytorch-lightning "conveniently" unwraps the list when there's only one dataloader,
        # so we need a check here.
        self.eval_epoch_end(
            self.dataset.dev_splits, [outputs] if isinstance(outputs[0], dict) else outputs
        )

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx=0
    ) -> dict[str, Any]:
        # self.eval_test_split is set in evaluate.py
        return self.eval_step(batch, batch_idx, self.eval_test_splits[dataloader_idx])

    def test_epoch_end(self, outputs: list[dict[str, Any]]):
        # pytorch-lightning "conveniently" unwraps the list when there's only one dataloader,
        # so we need a check here.
        self.eval_epoch_end(
            self.eval_test_splits, [outputs] if isinstance(outputs[0], dict) else outputs
        )

    @staticmethod
    def add_args(parser: ArgumentParser):
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
        parser.add_argument("--eval_batch_size", default=None, type=int)
