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

    roc_file = "roc_curves.png"
    fig.savefig(roc_file)
    logger.info(f"ROC curves saved in {roc_file}")


if __name__ == "__main__":
    logger.info("Comparing the models...")
    compare()
