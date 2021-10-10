import argparse
import joblib
import json

import numpy as np
from sklearn.metrics import f1_score

from pathlib import Path


def test(model_name):

    out_path = Path("data/prepared")
    with open(out_path / "texts.json", "r") as f:
        X_test = json.load(f)["test"]

    with open(out_path / "labels.json", "r") as f:
        Y_test = json.load(f)["test"]

    vectorizer = joblib.load(out_path / "vectorizer.joblib")
    X_test = vectorizer.transform(X_test).toarray()

    model = joblib.load(f"models/{model_name}.joblib")

    Y_test_pred = model.predict(X_test)
    metrics = {"f1_score": f1_score(y_true=Y_test, y_pred=Y_test_pred)}

    with open(f"models/{model_name}_test_metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    from logger import setup_applevel_logger

    log = setup_applevel_logger(file_name="app_debug.log")

    log.info("I am testing !")
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="LogisticRegression",
        help="A model name. Must be a class registered in src/models.py:factory",
    )

    args = parser.parse_args()
    test(args.model_name)
