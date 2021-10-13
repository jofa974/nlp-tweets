import argparse
import json
from pathlib import Path

from sklearn.metrics import f1_score

from models import SKLogisticRegression, TFConv1D
from prepare import SKCountVectorizer, TFTokenizer
from src.logger import logger


def test(model_class, preprocessor_class):

    out_path = Path(f"data/prepared/{preprocessor_class}")
    with open(out_path / "texts.json", "r") as f:
        X_test = json.load(f)["test"]

    with open(out_path / "labels.json", "r") as f:
        Y_test = json.load(f)["test"]

    preprocessor = globals()[preprocessor_class]()
    preprocessor.load()

    model = globals()[model_class](
        train=True, preprocessor=preprocessor, features=X_test, labels=Y_test
    )
    model.load()

    Y_test_pred = model.predict()
    metrics = {"f1_score": f1_score(y_true=Y_test, y_pred=Y_test_pred)}

    with open(f"models/{model_class}/test_metrics.json", "w") as f:
        json.dump(metrics, f)


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
