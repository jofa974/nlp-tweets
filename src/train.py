import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from constants import PARAMS
from models import factory
from prepare import SKCountVectorizer, TFTokenizer


def train(model_name, preprocessor_name):

    out_path = Path(f"data/prepared/{preprocessor_name}")
    with open(out_path / "texts.json", "r") as f:
        X_train = json.load(f)["train"]

    with open(out_path / "labels.json", "r") as f:
        Y_train = json.load(f)["train"]

    # TODO: use factory
    constructors = {"SKCountVectorizer": SKCountVectorizer, "TFTokenizer": TFTokenizer}

    preprocessor = constructors[args.preprocessor]()
    preprocessor.load()
    X_train = preprocessor.apply_preprocessor(X_train)

    kwargs = {
        "vocab_size": preprocessor.vocab_size,
        "input_shape": X_train.shape[1],
        "lr": PARAMS["lr"],
    }

    model = factory.create(model_name, **kwargs)
    model.make_model()
    model.fit(X_train, Y_train)

    Y_train_pred = model.predict(X_train)
    metrics = {"f1_score": f1_score(y_true=Y_train, y_pred=Y_train_pred)}

    with open(f"models/{model_name}_train_metrics.json", "w") as f:
        json.dump(metrics, f)

    model.save(model_name)


if __name__ == "__main__":
    from logger import setup_applevel_logger

    log = setup_applevel_logger(file_name="app_debug.log")

    log.info("I am training !")

    parser = argparse.ArgumentParser(description="Train model")
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
    train(args.model_name, args.preprocessor)
