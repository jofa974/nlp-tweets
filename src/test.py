import numpy as np
import argparse
import json
from pathlib import Path

from sklearn.metrics import f1_score

from src.dataset import Dataset
from src.logger import logger
from src.models import model_factory
from src.preprocessors import preprocessor_factory


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
    np.savetxt(f"models/{model_class}/test_predictions.csv", predictions)

    Y_test_pred = model.predict_class(threshold=0.5)

    metrics = {"f1_score": f1_score(y_true=ds._labels, y_pred=Y_test_pred)}

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
