import argparse
from importlib import reload
import logging
import os
import json
from pathlib import Path
from typing import Type

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

reload(logging)

from paoding.data.dataset import Dataset
from paoding.models.model import Model


logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def __init__(self):
        self.best_epoch = None
        self.best_dev_metric = None
        self.best_dev_metrics = None

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module: Model):
        logger.info("")
        logger.info("***** Validation results *****")

        assert pl_module.dataset.metric_watch_mode in {"max", "min"}

        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}".format(key, str(metrics[key])))

            if key == pl_module.dataset.metric_to_watch and not trainer.sanity_checking:
                if (
                    self.best_dev_metric is None
                    or (
                        pl_module.dataset.metric_watch_mode == "max"
                        and metrics[key] > self.best_dev_metric
                    )
                    or (
                        pl_module.dataset.metric_watch_mode == "min"
                        and metrics[key] < self.best_dev_metric
                    )
                ):
                    self.best_epoch = trainer.current_epoch
                    self.best_dev_metric = metrics[key]
                    self.best_dev_metrics = {
                        k: v
                        for k, v in metrics.items()
                        if k not in {"log", "progress_bar", "loss", "val_loss", "lr", "epoch"}
                    }

        if not trainer.sanity_checking:
            logger.info(f"best_epoch = {self.best_epoch}")
            for key, value in sorted(self.best_dev_metrics.items()):
                logger.info(f"best_{key} = {value}")


def parse_meta_args(add_args_fn):
    """
    A meta parser that can be used to decide the model and dataset classes.
    Typical usage:

    ```
    meta_args, extra_args = parse_meta_args(add_meta_args)
    train(
        MODEL_MAP[meta_args.model_type],
        DATASET_MAP[meta_args.data_type],
        args=extra_args,
    )
    """
    parser = argparse.ArgumentParser()
    add_args_fn(parser)
    return parser.parse_known_args()


def add_generic_args(parser: argparse.ArgumentParser):
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)


def train(model_class: Type[Model], dataset_class: Type[Dataset], args: argparse.Namespace = None):
    parser = argparse.ArgumentParser()
    add_generic_args(parser)
    model_class.add_args(parser)
    dataset_class.add_args(parser)
    args = parser.parse_args(args=args)

    output_dir = args.output_dir
    if os.path.exists(output_dir):
        content = os.listdir(output_dir)
        # For DDP, when subprocesses are launched, there'll be a log.txt inside the folder already
        if len(content) > 0 and content != ["log.txt"]:
            raise ValueError(f"Output directory ({output_dir}) already exists and is not empty.")
    else:
        os.mkdir(output_dir)

    assert (
        getattr(args, "model_class", None) is None and getattr(args, "dataset_class", None) is None
    )
    args.model_class = model_class
    args.dataset_class = dataset_class
    json.dump(
        {k: str(v) if k in ("model_class", "dataset_class") else v for k, v in vars(args).items()},
        open(os.path.join(output_dir, "args.json"), "w"),
    )

    if args.gpus is None:
        args.gpus = (
            len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
            if "CUDA_VISIBLE_DEVICES" in os.environ
            else 0
        )

    pl.seed_everything(args.seed)

    # Set by pytorch-lightning
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    logging.basicConfig(
        level=logging.INFO if local_rank <= 0 else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "log.txt")),
            logging.StreamHandler(),
        ],
    )

    model = model_class(args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename=f"{{epoch}}_{{{model.dataset.metric_to_watch}:.4f}}",
        monitor=model.dataset.metric_to_watch,
        mode=model.dataset.metric_watch_mode,
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        gradient_clip_val=args.clip_norm,
        gpus=args.gpus,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.epochs,
        precision=16 if args.fp16 else 32,
        callbacks=[LoggingCallback(), checkpoint_callback],
        replace_sampler_ddp=False,
    )
    trainer.fit(model)

    if local_rank <= 0:
        os.symlink(
            checkpoint_callback.best_model_path,
            Path(checkpoint_callback.best_model_path).parent / "best.ckpt",
        )
