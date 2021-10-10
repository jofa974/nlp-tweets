import argparse
import json

import joblib
import numpy as np
from sklearn.metrics import f1_score
from pathlib import Path
from models import factory
from constants import PARAMS


def train(model_name):

    out_path = Path("data/prepared")
    with open(out_path / "texts.json", "r") as f:
        X_train = json.load(f)["train"]

    with open(out_path / "labels.json", "r") as f:
        Y_train = json.load(f)["train"]

    vectorizer = joblib.load(out_path / "vectorizer.joblib")
    X_train = vectorizer.transform(X_train).toarray()

    model = factory.create(model_name, **PARAMS["models"][model_name]["kwargs"])
    model.fit(X_train, Y_train)

    Y_train_pred = model.predict(X_train)
    metrics = {"f1_score": f1_score(y_true=Y_train, y_pred=Y_train_pred)}

    with open(f"models/{model_name}_train_metrics.json", "w") as f:
        json.dump(metrics, f)

    joblib.dump(model, f"models/{model_name}.joblib")


if __name__ == "__main__":
    from logger import setup_applevel_logger

    log = setup_applevel_logger(file_name="app_debug.log")

    log.info("I am training !")

    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="LogisticRegression",
        help="A model name. Must be a class registered in src/models.py:factory",
    )

    args = parser.parse_args()
    train(args.model_name)
