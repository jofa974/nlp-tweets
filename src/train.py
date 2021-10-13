import argparse
import json
from pathlib import Path

from sklearn.metrics import f1_score

from constants import PARAMS
from models import SKLogisticRegression, TFConv1D
from prepare import SKCountVectorizer, TFTokenizer
from src.logger import logger


def train(model_class, preprocessor_class):

    out_path = Path(f"data/prepared/{preprocessor_class}")
    with open(out_path / "texts.json", "r") as f:
        X_train = json.load(f)["train"]

    with open(out_path / "labels.json", "r") as f:
        Y_train = json.load(f)["train"]

    preprocessor = globals()[preprocessor_class]()
    preprocessor.load()
    X_train = preprocessor.apply_preprocessor(X_train)

    kwargs = {
        "vocab_size": preprocessor.vocab_size,
        "input_shape": X_train.shape[1],
        "lr": PARAMS["lr"],
        "name": model_class,
    }

    model = globals()[model_class](**kwargs)
    model.make_model()
    model.fit(X_train, Y_train)

    Y_train_pred = model.predict(X_train)
    metrics = {"f1_score": f1_score(y_true=Y_train, y_pred=Y_train_pred)}

    with open(f"models/{model_class}/train_metrics.json", "w") as f:
        json.dump(metrics, f)

    model.save(model_class)


if __name__ == "__main__":
    logger.info("I am training !")

    parser = argparse.ArgumentParser(description="Train model")
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
        help="A preprocessor class. Must be a sub-class of Preprocessor.",
    )

    args = parser.parse_args()
    train(args.model_class, args.preprocessor)
