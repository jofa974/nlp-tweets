import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.extract import find
from sklearn.metrics import f1_score, precision_recall_curve

from src.dataset import Dataset
from src.logger import logger
from src.models import model_factory
from src.preprocessors import preprocessor_factory


def find_best_threshold(y_true, y_prob):
    """Find the best threshold for maximum F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = (2 * precisions * recalls) / (precisions + recalls)
    return thresholds[np.argmax(f1s)]


def test(model_class, preprocessor_class):

    preprocessor = preprocessor_factory.get_preprocessor(preprocessor_class)
    preprocessor.load()

    data_path = Path(f"data/prepared/{preprocessor_class}")
    ds = Dataset()
    ds.load_features(data_path, stage="test")
    ds.load_labels(data_path, stage="test")

    ds._features = preprocessor.apply(ds._features)

    model = model_factory.get_model(model_class, dataset=ds)
    model.load()

    predictions = model.predict()
    df = pd.DataFrame({"y_pred": predictions, "y_true": ds._labels})
    df.to_csv(f"models/{model_class}/test_predictions.csv", index=False)

    Y_test_pred = model.predict_class(threshold=0.5)

    metrics = {"f1_score": f1_score(y_true=ds._labels, y_pred=Y_test_pred)}

    with open(f"models/{model_class}/test_metrics.json", "w") as f:
        json.dump(metrics, f)

    best_thresh = find_best_threshold(y_true=ds._labels, y_prob=Y_test_pred)
    logger.info(f"Best threshold is {best_thresh}")


if __name__ == "__main__":
    logger.info("I am testing !")
    parser = argparse.ArgumentParser(description="Test model")
    parser.add_argument(
        "--model-class",
        type=str,
        default="SKLogisticRegression",
        help="A model class. Must be implemented in a model.py file.",
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="SKCountVectorizer",
        help="A preprocessor class. Must be a sub-class of Preprocessor",
    )

    args = parser.parse_args()
    test(args.model_class, args.preprocessor)
