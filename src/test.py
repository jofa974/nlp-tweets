import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from prepare import SKCountVectorizer, TFTokenizer


def test(model_name, preprocessor_name):

    out_path = Path(f"data/prepared/{preprocessor_name}")
    with open(out_path / "texts.json", "r") as f:
        X_test = json.load(f)["test"]

    with open(out_path / "labels.json", "r") as f:
        Y_test = json.load(f)["test"]

    # TODO: use factory
    constructors = {"SKCountVectorizer": SKCountVectorizer, "TFTokenizer": TFTokenizer}

    preprocessor = constructors[args.preprocessor]()
    preprocessor.load()
    X_test = preprocessor.apply_preprocessor(X_test)

    model = joblib.load(f"models/{model_name}.joblib")

    Y_test_pred = model.predict(X_test)
    metrics = {"f1_score": f1_score(y_true=Y_test, y_pred=Y_test_pred)}

    with open(f"models/{model_name}_test_metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    from logger import setup_applevel_logger

    log = setup_applevel_logger(file_name="app_debug.log")

    log.info("I am testing !")
    parser = argparse.ArgumentParser(description="Test model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="logistic_regression",
        help="A model name. Must be a class registered in src/models.py:factory",
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="SKCountVectorizer",
        help="A preprocessor's name. Must be a sub-class of Preprocessor",
    )

    args = parser.parse_args()
    test(args.model_name, args.preprocessor)
