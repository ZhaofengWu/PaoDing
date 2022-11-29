import argparse
from importlib import reload
import logging
import os
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Type

# PyTorch-Lightning's interruption of sigterm when using slurm seems to cause issues with
# multiprocessing. See https://github.com/PyTorchLightning/pytorch-lightning/issues/5225
# and https://github.com/PyTorchLightning/pytorch-lightning/issues/5969. So disabling it.
os.environ.pop("SLURM_NTASKS", None)
os.environ.pop("SLURM_JOB_NAME", None)

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cloud_io import load as pl_load
import torchmetrics

reload(logging)

from paoding.argument_parser import ArgumentParser
from paoding.data.dataset import Dataset
from paoding.models.model import Model
from paoding.utils import get_logger


logger = get_logger(__name__)


class LoggingCallback(pl.Callback):
    def __init__(self):
        self.best_epoch = None
        self.best_dev_metric = None
        self.best_dev_metrics = None

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module: Model):
        if not trainer.sanity_checking:
            return

        logger.info("")
        logger.info("***** Validation results after sanity checking *****")
        self.log_metrics(trainer, pl_module)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: Model):
        logger.info("")
        logger.info(f"***** Validation results at epoch {trainer.current_epoch} *****")
        self.log_metrics(trainer, pl_module)

    def log_metrics(self, trainer: pl.Trainer, pl_module: Model):
        metrics = trainer.callback_metrics
        higher_is_better = getattr(torchmetrics, pl_module.dataset.metric_to_watch).higher_is_better
        assert higher_is_better is not None
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}".format(key, str(metrics[key])))

            if key == pl_module.dataset.metric_to_watch and not trainer.sanity_checking:
                if (
                    self.best_dev_metric is None
                    or (higher_is_better and metrics[key] > self.best_dev_metric)
                    or (not higher_is_better and metrics[key] < self.best_dev_metric)
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
    Example usage:

    ```
    def add_meta_args(parser: ArgumentParser):
        parser.add_argument("--model_type", type=str, required=True)
        parser.add_argument("--data_type", type=str, required=True)

    meta_args, extra_args = parse_meta_args(add_meta_args)
    train(
        MODEL_MAP[meta_args.model_type],
        DATASET_MAP[meta_args.data_type],
        args=extra_args,
    )
    ```
    """
    parser = ArgumentParser()
    add_args_fn(parser)
    return parser.parse_known_args()


def add_generic_args(parser: ArgumentParser):
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument(
        "--ckpt_path", default=None, type=str, help="If specified, load the weights from this ckpt."
    )
    parser.add_argument("--non_strict_load", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--no_wandb", action="store_true")  # for debugging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Turn on a few things that may slow down the run but makes debugging easier. E.g.,"
        " running in post-mortem mode and using no separate dataloader worker.",
    )


def wrapped_train(
    hparams: argparse.Namespace,
    argv: list[str],
    model_class: Type[Model],
    dataset_class: Type[Dataset],
    wandb_info: dict = None,
) -> tuple[str, Any]:
    output_dir = hparams.output_dir
    if os.path.exists(output_dir):
        content = os.listdir(output_dir)
        whitelist_files = {"log.txt", "lightning_logs", "hparams.json"}
        if len(content) > 0 and any(c not in whitelist_files for c in content):
            raise ValueError(f"Output directory ({output_dir}) already exists and is not empty.")
        for c in content:
            # TODO: check if this works for DDP -- log.txt is created by the master process and
            # may be assumed to exist by the launched processes?
            full_path = os.path.join(output_dir, c)
            if os.path.isfile(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
            else:
                # Technically there may be other possibilities, but we know the things on the
                # whitelist shouldn't be anything weird.
                assert False
    else:
        os.mkdir(output_dir)

    assert (
        getattr(hparams, "model_class", None) is None
        and getattr(hparams, "dataset_class", None) is None
    )
    hparams.model_class = model_class
    hparams.dataset_class = dataset_class
    json.dump(
        {
            k: str(v) if k in ("model_class", "dataset_class") else v
            for k, v in vars(hparams).items()
        },
        open(os.path.join(output_dir, "hparams.json"), "w"),
    )

    if hparams.gpus is None:
        hparams.gpus = (
            len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
            if "CUDA_VISIBLE_DEVICES" in os.environ
            else 0
        )

    pl.seed_everything(hparams.seed)

    # Set by pytorch-lightning
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    logging.basicConfig(
        level=logging.INFO if local_rank <= 0 else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "log.txt")),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger.info(f"Command: {sys.executable} {' '.join(argv)}")

    model = model_class(hparams)
    if hparams.ckpt_path is not None:
        ckpt = pl_load(hparams.ckpt_path)
        keys = model.load_state_dict(ckpt["state_dict"], strict=not hparams.non_strict_load)
        if hparams.non_strict_load:
            if keys.missing_keys:
                logger.warning(f"Missing keys in state dict: {keys.missing_keys}")
            if keys.unexpected_keys:
                logger.warning(f"Unexpected keys in state dict: {keys.unexpected_keys}")

    loging_callback = LoggingCallback()
    higher_is_better = getattr(torchmetrics, model.dataset.metric_to_watch).higher_is_better
    assert higher_is_better is not None
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename=f"{{epoch}}_{{{model.dataset.metric_to_watch}:.4f}}",
        monitor=model.dataset.metric_to_watch,
        mode="max" if higher_is_better else "min",
        save_top_k=1,
        save_last=True,
    )

    trainer_loggers = [TensorBoardLogger(hparams.output_dir)]
    if not hparams.no_wandb and wandb_info is not None:
        output_dir_basename = os.path.basename(os.path.normpath(hparams.output_dir))
        trainer_loggers.append(WandbLogger(name=output_dir_basename, **wandb_info))
    trainer = pl.Trainer(
        default_root_dir=hparams.output_dir,
        gradient_clip_val=hparams.clip_norm,
        gpus=hparams.gpus,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.epochs,
        precision=16 if hparams.fp16 else 32,
        logger=trainer_loggers,
        callbacks=[loging_callback, checkpoint_callback],
        replace_sampler_ddp=False,
    )
    trainer.fit(model)

    if not trainer.interrupted and local_rank <= 0:
        best_model_path = Path(checkpoint_callback.best_model_path)
        symlink_path = best_model_path.parent / "best.ckpt"
        # Use relative path so that the symlink still works after directory rename
        os.symlink(os.path.relpath(best_model_path, symlink_path.parent), symlink_path)
        print(f"Model saved at {symlink_path} -> {checkpoint_callback.best_model_path}")

    return model.dataset.metric_to_watch, loging_callback.best_dev_metric


def train(
    model_class: Type[Model],
    dataset_class: Type[Dataset],
    args: list = None,
    wandb_info: dict = None,
) -> tuple[str, Any]:
    argv = list(sys.argv)  # idk how argparser uses sys.argv, so making a backup to be safe

    parser = ArgumentParser()
    add_generic_args(parser)
    model_class.add_args(parser)
    dataset_class.add_args(parser)
    hparams = parser.parse_args(args=args)

    if hparams.debug:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        try:
            return wrapped_train(hparams, argv, model_class, dataset_class, wandb_info=wandb_info)
        except Exception as e:
            import pdb
            import traceback

            if not isinstance(e, (pdb.bdb.BdbQuit, KeyboardInterrupt)):
                print("\n" + ">" * 100 + "\n")
                traceback.print_exc()
                print()
                pdb.post_mortem()
    else:
        return wrapped_train(hparams, argv, model_class, dataset_class, wandb_info=wandb_info)
