import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from src.dataset import Dataset
from src.logger import logger
from src.models import model_factory
from src.preprocessors import preprocessor_factory


def train(model_class, preprocessor_class):

    preprocessor = preprocessor_factory.get_preprocessor(preprocessor_class)
    preprocessor.load()

    data_path = Path(f"data/prepared/{preprocessor_class}")
    ds = Dataset()
    ds.load_features(data_path, stage="train")
    ds.load_labels(data_path, stage="train")

    ds._features = preprocessor.apply(ds._features)

    model = model_factory.get_model_from_preproc(
        model_class, preprocessor, input_shape=ds.input_shape
    )

    model.summary()
    model.fit(ds, use_validation=True)

    Y_train_pred = model.predict_class(dataset=ds, threshold=0.5)
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
