import argparse
from importlib import reload
import logging
import os
from typing import Type

# PyTorch-Lightning's interruption of sigterm when using slurm seems to cause issues with
# multiprocessing. See https://github.com/PyTorchLightning/pytorch-lightning/issues/5225
# and https://github.com/PyTorchLightning/pytorch-lightning/issues/5969. So disabling it.
os.environ.pop("SLURM_NTASKS", None)
os.environ.pop("SLURM_JOB_NAME", None)

import pytorch_lightning as pl

reload(logging)

from paoding.analysis import add_analysis_args, analyze
from paoding.models.model import Model
from paoding.utils import get_logger


logger = get_logger(__name__)


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
        help="The checkpoint contains the path to the eval data which overrides the data directory."
        " This is necessary, for example, when the original data path does not exist if you are"
        " using a model trained in a different environment.",
    )
    parser.add_argument(
        "--split",
        default=None,
        type=str,
        help="A single split on which to evaluate. Takes precedence over --splits.",
    )
    parser.add_argument(
        "--splits",
        default="dev",
        type=str,
        choices=["train", "dev", "test"],
        help="Evaluate on the train split or all dev or test splits.",
    )
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no_log_file", action="store_true")

    add_analysis_args(parser)

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
    if hparams.batch_size is not None:
        load_kwargs["eval_batch_size"] = hparams.batch_size
    model = model_class.load_from_checkpoint(hparams.ckpt_path, strict=strict_load, **load_kwargs)
    model.freeze()

    trainer = pl.Trainer(gpus=hparams.gpus, default_root_dir=model.hparams.output_dir)
    if hparams.splits == "train":
        splits = [model.dataset.train_split]
    else:
        splits = getattr(model.dataset, f"{hparams.splits}_splits")
    if hparams.split is not None:
        splits = [hparams.split]
    assert splits is not None

    # Set by pytorch-lightning
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    ckpt_dir = os.path.dirname(hparams.ckpt_path)
    log_file = os.path.join(ckpt_dir, f"eval_{'_'.join(splits)}.txt")
    if os.path.exists(log_file) and not hparams.no_log_file:
        raise ValueError(f"Log file ({log_file}) already exists.")
    handlers = [logging.StreamHandler()]
    if not hparams.no_log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO if local_rank <= 0 else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        handlers=handlers,
        force=True,
    )

    for split in splits:
        logger.info(f"Evaluating on {split=}")
        model.current_test_split = split
        dataloader = model.dataset.dataloader(split, model.hparams.eval_batch_size, shuffle=False)
        results = trainer.test(
            model=model,
            dataloaders=dataloader,
        )
        logger.info(str(results))
        analyze(hparams, model._labels, model._preds)
        # For safety:
        del model._labels
        del model._preds

    if not hparams.no_log_file:
        logger.info(f"Log saved to {log_file}")
