import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def add_analysis_args(parser: argparse.ArgumentParser):
    parser.add_argument("--confusion_matrix", action="store_true")


def analyze(hparams, labels, preds):
    if hparams.confusion_matrix:
        plot_confusion_matrix(labels, preds)


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
