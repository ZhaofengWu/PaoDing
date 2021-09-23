import argparse
import os
from typing import Type

import pytorch_lightning as pl

from paoding.models.model import Model


def add_eval_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--ckpt_path",
        default=None,
        type=str,
        required=True,
        help="The path to the checkpoint.",
    )
    parser.add_argument(
        "--override_data_dir",
        default=None,
        type=str,
        help="The checkpoint contains the path to the eval data, which may not exist if you are"
        " using a model trained on a different platform. Provided this flag ovreride the data"
        " directory.",
    )
    parser.add_argument(
        "--test_set",
        action="store_true",
        help="Evaluate on the dev set by default; use this flag to evaluate on the test set.",
    )
    parser.add_argument("--gpus", type=int, default=None)


def evaluate(model_class: Type[Model], strict_load=True):
    parser = argparse.ArgumentParser()
    add_eval_args(parser)
    hparams = parser.parse_args()

    if hparams.gpus is None:
        hparams.gpus = (
            len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
            if "CUDA_VISIBLE_DEVICES" in os.environ
            else 0
        )

    load_kwargs = {}
    if hparams.override_data_dir is not None:
        load_kwargs["data_dir"] = hparams.override_data_dir
    model = model_class.load_from_checkpoint(hparams.ckpt_path, strict=strict_load, **load_kwargs)
    model.freeze()

    trainer = pl.Trainer(gpus=hparams.gpus, default_root_dir=model.hparams.output_dir)
    trainer.test(
        model=model,
        dataloaders=model.test_dataloader() if hparams.test_set else model.val_dataloader(),
    )
