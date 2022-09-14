import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from paoding.argument_parser import ArgumentParser
from paoding.utils import get_logger


logger = get_logger(__name__)


def add_analysis_args(parser: ArgumentParser):
    parser.add_argument("--confusion_matrix", action="store_true")
    parser.add_argument("--log_predictions", action="store_true")


def analyze(hparams, labels, preds, dataloader, split):
    if hparams.confusion_matrix:
        plot_confusion_matrix(labels, preds)
    if hparams.log_predictions:
        log_predictions(hparams, labels, preds, dataloader, split)


def plot_confusion_matrix(
    labels, preds, classes=None, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
) -> np.ndarray:
    cm = confusion_matrix(labels, preds)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

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
        assert len(preds) == len(labels) == len(dataloader.dataset)
        # This zip relies on we not shuffling
        for p, l, e in zip(preds, labels, dataloader.dataset):
            # Why not "correct"/"incorrect", you ask? Because the fact that "correct" is a substring
            # of "incorrect" makes searching harder
            f.write(
                "\t".join([str(e), f"pred{p}", f"label{l}", "right" if p == l else "wrong"]) + "\n"
            )

    logger.info(f"Predictions logged to {pred_file}")
