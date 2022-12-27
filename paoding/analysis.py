import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from paoding.argument_parser import ArgumentParser
from paoding.utils import get_logger


logger = get_logger(__name__)


ALL_ANALYSES = ["confusion_matrix", "ascii_confusion_matrix", "log_predictions"]


def add_analysis_args(parser: ArgumentParser):
    for analysis in ALL_ANALYSES:
        parser.add_argument(f"--{analysis}", action="store_true")


def analysis_enabled(hparams):
    assert all(hasattr(hparams, analysis) for analysis in ALL_ANALYSES)
    return any(getattr(hparams, analysis) for analysis in ALL_ANALYSES)


def analyze(hparams, labels, preds, dataloader, split):
    if hparams.confusion_matrix or hparams.ascii_confusion_matrix:
        assert not (hparams.confusion_matrix and hparams.ascii_confusion_matrix)
        plot_confusion_matrix(
            labels, preds, title=f"Confusion matrix {split}", ascii=hparams.ascii_confusion_matrix
        )
    if hparams.log_predictions:
        log_predictions(hparams, labels, preds, dataloader, split)


def plot_confusion_matrix(
    labels,
    preds,
    classes=None,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    ascii=False,
) -> np.ndarray:
    cm = confusion_matrix(labels, preds)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    if ascii:
        print(title)
        print(cm)
        return cm

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if classes is None:
        num_classes = max(max(labels), max(preds)) - min(min(labels), min(preds)) + 1
        classes = list(range(num_classes))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

    return cm


def log_predictions(hparams, labels, preds, dataloader, split):
    ckpt_dir = os.path.dirname(hparams.ckpt_path)
    pred_file = os.path.join(ckpt_dir, f"pred_{split}.txt")

    assert not os.path.exists(pred_file), f"Prediction file {pred_file} exists."

    with open(pred_file, "w") as f:
        # This zip relies on we not shuffling
        for p, l, e in zip(preds, labels, dataloader.dataset, strict=True):
            # Why not "correct"/"incorrect", you ask? Because the fact that "correct" is a substring
            # of "incorrect" makes searching harder
            f.write(
                "\t".join([str(e), f"pred{p}", f"label{l}", "right" if p == l else "wrong"]) + "\n"
            )

    logger.info(f"Predictions logged to {pred_file}")
