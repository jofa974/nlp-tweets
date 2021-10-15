import argparse
import json
from pathlib import Path

import yaml
from sklearn.metrics import f1_score

from models import SKLogisticRegression, TFConv1D
from prepare import SKCountVectorizer, TFTokenizer
from src.dataset import Dataset
from src.logger import logger


def train(model_class, preprocessor_class):

    preprocessor = globals()[preprocessor_class]()
    preprocessor.load()

    data_path = Path(f"data/prepared/{preprocessor_class}")
    ds = Dataset()
    ds.load_features(data_path, stage="train")
    ds.load_labels(data_path, stage="train")

    ds._features = preprocessor.apply(ds._features)

    model = globals()[model_class](dataset=ds)
    model.make_model(vocab_size=preprocessor.vocab_size)
    model.fit()

    Y_train_pred = model.predict()
    metrics = {"f1_score": f1_score(y_true=ds._labels, y_pred=Y_train_pred)}

    with open(f"models/{model_class}/train_metrics.json", "w") as f:
        json.dump(metrics, f)

    model.save()


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
