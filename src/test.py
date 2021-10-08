import joblib
import json

import numpy as np
from sklearn.metrics import f1_score

from pathlib import Path


def test():

    out_path = Path("data/prepared")
    with open(out_path / "texts.json", "r") as f:
        X_test = json.load(f)["test"]

    with open(out_path / "labels.json", "r") as f:
        Y_test = json.load(f)["test"]

    vectorizer = joblib.load(out_path / "vectorizer.joblib")
    X_test = vectorizer.transform(X_test).toarray()

    model = joblib.load("models/simple_model.joblib")

    Y_test_pred = model.predict(X_test)
    metrics = {"f1_score": f1_score(y_true=Y_test, y_pred=Y_test_pred)}

    with open("models/test_metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    from logger import setup_applevel_logger

    log = setup_applevel_logger(file_name="app_debug.log")

    log.info("I am testing !")
    test()
