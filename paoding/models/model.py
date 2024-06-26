import argparse
from collections import defaultdict
from typing import Any, Optional

from lightning_utilities.core.apply_func import apply_to_collection
import lightning as pl
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
import torchmetrics
from torchmetrics import Metric
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import PreTrainedTokenizerBase

from paoding.argument_parser import ArgumentParser
from paoding.data.tokenizer import Tokenizer

torch.set_float32_matmul_precision("high")

SCHEDULER_MAP = {
    "cosine": get_cosine_schedule_with_warmup,
    "linear": get_linear_schedule_with_warmup,
}


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
        self._eval_outputs = {}
        self.scheduler_step_counter = 0

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

    def metric_init_kwargs(self, metric_name: str) -> dict[str, Any]:
        kwargs = {}
        if self.dataset.task == "classification":
            kwargs["num_classes"] = self.dataset.num_labels
        if metric_name == "MulticlassAccuracy":
            kwargs["average"] = "micro"
        return kwargs

    def setup_metric(self, metric_name: str) -> Metric:
        try:
            metric_cls = getattr(torchmetrics, metric_name)
        except AttributeError:
            metric_cls = getattr(torchmetrics.classification, metric_name)
        return metric_cls(**self.metric_init_kwargs(metric_name))

    def setup_metrics(self) -> dict[str, dict[str, Metric]]:
        metric_splits = (
            [self.dataset.train_split]
            + self.dataset.dev_splits
            + self.dataset.test_splits
            + ["aggregate"]
        )
        return {
            split: {name: self.setup_metric(name) for name in self.dataset.metric_names}
            for split in metric_splits
        }

    def train_dataloader(self) -> DataLoader:
        return self.dataset.dataloader(
            self.dataset.train_split, self.hparams.batch_size, shuffle=True
        )

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
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_epsilon,
        )
        scheduler = self.get_lr_scheduler(optimizer)

        return [optimizer], [scheduler]

    def get_lr_scheduler(self, optimizer: Optimizer) -> dict:
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        if self.hparams.lr_scheduler_total_steps is not None:
            total_steps = self.hparams.lr_scheduler_total_steps
        else:
            dataset_size = len(self.dataset[self.dataset.train_split])
            effective_batch_size = (
                self.hparams.batch_size * self.hparams.accumulate_grad_batches * num_devices
            )
            # Sometimes dataset_size could be smaller than the effective_batch_size
            total_steps = max(dataset_size / effective_batch_size, 1) * self.hparams.epochs

        if self.hparams.warmup_steps > 0 and self.hparams.warmup_ratio > 0:
            raise ValueError("--warmup_steps and --warmup_ratio are mutually exclusive.")
        warmup_steps = (
            int(total_steps * self.hparams.warmup_ratio)
            if self.hparams.warmup_ratio > 0
            else self.hparams.warmup_steps
        )
        scheduler = SCHEDULER_MAP[self.hparams.scheduler_type](
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        return {"scheduler": scheduler, "interval": "step", "frequency": 1}

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]):
        # From https://github.com/Lightning-AI/lightning/issues/5558#issuecomment-1774469254
        assert hasattr(scheduler.optimizer, "_step_count")

        if self.scheduler_step_counter < scheduler.optimizer._step_count:
            super().lr_scheduler_step(scheduler, metric)
            self.scheduler_step_counter += 1
            assert self.scheduler_step_counter == scheduler.optimizer._step_count

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer):
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
        self,
        logits: torch.Tensor,
        labels: torch.Tensor = None,
        mask: torch.Tensor = None,
        logits_rejected: torch.Tensor = None,
        reduce=True,
    ) -> torch.Tensor:
        match self.dataset.task:
            case "classification" | "causal_lm" | "seq2seq" | "masked_lm":
                if self.dataset.task in {"causal_lm", "seq2seq"}:
                    assert mask is not None and mask.any(dim=-1).all()
                if self.dataset.task == "masked_lm":
                    assert mask is not None
                if mask is not None:
                    labels = labels.masked_fill(~mask, -100)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="mean" if mask is None else "none",
                )
                if mask is not None:
                    loss = loss.reshape_as(labels) * mask
                    if reduce:
                        loss = loss.sum() / mask.sum()
            case "regression" | "multi_regression":
                loss = F.mse_loss(
                    logits.reshape(-1),
                    labels.reshape(-1),
                    reduction="mean" if mask is None else "none",
                )
                if mask is not None:
                    if self.dataset.task == "multi_regression":
                        mask = mask.unsqueeze(-1)
                    loss = loss.reshape_as(labels) * mask
                    if reduce:
                        loss = loss.sum() / mask.sum()
                        if self.dataset.task == "multi_regression":
                            loss /= logits.shape[-1]
            case "pairwise_rm":  # TODO: support for pairwise RM has not been tested
                loss = -F.logsigmoid(logits - logits_rejected)
                if reduce:
                    loss = loss.mean()
            case _:
                raise KeyError(f"Output mode not supported: {self.dataset.task}")
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        output = self(batch)
        if "loss" in output:
            loss = output["loss"]
        else:
            if self.dataset.task == "pairwise_rm":
                loss = self.compute_loss(
                    output["logits_chosen"], logits_rejected=output["logits_rejected"]
                )
            else:
                loss = self.compute_loss(
                    output["logits"],
                    batch[self.dataset.label_key],
                    batch.get(self.dataset.label_mask_key),
                )
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "lr",
            self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[-1],
            prog_bar=True,
            rank_zero_only=True,
        )
        return {"loss": loss}

    def get_predictions(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        match self.dataset.task:
            case "classification" | "masked_lm":
                return logits.argmax(dim=-1)
            case "regression":
                return logits.squeeze(dim=-1)
            case "multi_regression":
                return logits
            case "causal_lm" | "seq2seq":
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

        # torchmetrics doesn't take a mask which is stupid. See issue
        # https://github.com/Lightning-AI/metrics/issues/1282
        # So we have to flatten these tensors.
        flat_preds = preds
        flat_labels = labels
        if self.dataset.label_mask_key in batch:
            label_mask = batch[self.dataset.label_mask_key]
            flat_preds = preds[label_mask]
            flat_labels = labels[label_mask]
        for s in (split, "aggregate"):
            for metric in self.metrics[s].values():
                if getattr(metric, "supports_mask", False):
                    # An opportunity for custom metrics that want to take un-flattened preds/labels
                    label_mask = batch.get(self.dataset.label_mask_key)
                    metric(
                        preds.detach(),
                        labels.detach(),
                        label_mask.detach() if label_mask is not None else None,
                    )
                else:
                    # For other metrics, we merge the batch dim and the seq dim, but perplexity
                    # requires them to be separate.
                    is_perplexity = isinstance(metric, torchmetrics.Perplexity)
                    metric(
                        (flat_preds.unsqueeze(0) if is_perplexity else flat_preds).detach(),
                        (flat_labels.unsqueeze(0) if is_perplexity else flat_labels).detach(),
                    )

        return_dict = {}
        if compute_loss:
            loss = (
                output["loss"]
                if "loss" in output
                else self.compute_loss(
                    output["logits"], labels, batch.get(self.dataset.label_mask_key)
                )
            )
            return_dict["loss"] = loss.detach().cpu()
        if getattr(self, "analysis_enabled", False):
            return_dict["preds"] = preds.detach().cpu().numpy()
            return_dict["labels"] = labels.detach().cpu().numpy()

        if split not in self._eval_outputs:
            self._eval_outputs[split] = []
        self._eval_outputs[split].append(return_dict)

        return return_dict

    def on_eval_epoch_end(self, splits: list[str]):
        num_splits = len(splits)
        # We gather individual metrics from each dataloader and compute the average if there is
        # more than one
        if num_splits > 1:
            sums = defaultdict(int)

        for split in splits:
            split_outputs = self._eval_outputs[split]
            metrics = self.get_metrics(split, reset=True)

            if "loss" in split_outputs[0]:
                loss = sum(o["loss"] for o in split_outputs) / len(split_outputs)
                metrics |= {"loss": loss}

            for k, v in metrics.items():
                if num_splits > 1:
                    assert f"{k}_{split}" not in metrics
                    self.log(f"{k}_{split}", v, sync_dist=True)
                    sums[k] += v
                else:
                    self.log(k, v, sync_dist=True)
            self._eval_outputs[split].clear()

        agg_metrics = self.get_metrics("aggregate", reset=True)
        if num_splits > 1:
            for k, v in sums.items():
                # It's important to keep the aggregate metric to be the original name, since it is
                # the sort key.
                self.log(k, v / num_splits, sync_dist=True)
            for k, v in agg_metrics.items():
                self.log(k + "_microaggregate", v, sync_dist=True)

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

    def on_validation_epoch_end(self):
        self.on_eval_epoch_end(self.dataset.dev_splits)

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx=0
    ) -> dict[str, Any]:
        # self.eval_test_split is set in evaluate.py
        return self.eval_step(batch, batch_idx, self.eval_test_splits[dataloader_idx])

    def on_test_epoch_end(self):
        self.on_eval_epoch_end(self.eval_test_splits)

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--lr", default=2e-5, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--clip_norm", default=0.0, type=float)
        parser.add_argument("--accumulate_grad_batches", default=1, type=int)
        parser.add_argument("--adam_beta1", default=0.9, type=float)
        parser.add_argument("--adam_beta2", default=0.999, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument(
            "--scheduler_type", default="linear", choices=SCHEDULER_MAP.keys(), type=str
        )
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--warmup_ratio", default=0.0, type=float)
        parser.add_argument("--lr_scheduler_total_steps", default=None, type=int)
        parser.add_argument("--epochs", default=3, type=int)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=None, type=int)
