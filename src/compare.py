from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

from src.logger import logger


def plot_roc(name, labels, predictions, axis, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)

    axis.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    axis.set_xlabel("False positives [%]")
    axis.set_ylabel("True positives [%]")
    # axis.set_xlim([-0.5, 20])
    # axis.set_ylim([80, 100.5])
    axis.grid(True)
    axis.set_aspect("equal")
    return axis


def compare():
    fig, ax = plt.subplots()
    ax = plt.gca()
    for prediction_file in Path("models/").glob("*/test_predictions.csv"):
        df = pd.read_csv(prediction_file)
        ax = plot_roc(
            f"Model: {prediction_file.parent.name}", df["y_true"], df["y_pred"], axis=ax
        )
    plt.legend()
    fig.savefig("roc_curves.png")


if __name__ == "__main__":
    logger.info("I am comparing !")
    compare()
