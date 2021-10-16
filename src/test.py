import argparse
import json
from pathlib import Path

from sklearn.metrics import f1_score

from models import SKLogisticRegression, TFConv1D
from src import preprocessors
from src.dataset import Dataset
from src.logger import logger
from src.preprocessors import preprocessor_factory


def test(model_class, preprocessor_class):

    preprocessor = preprocessor_factory.get_preprocessor(args.preprocessor)
    preprocessor.load()

    data_path = Path(f"data/prepared/{preprocessor_class}")
    ds = Dataset()
    ds.load_features(data_path, stage="test")
    ds.load_labels(data_path, stage="test")

    ds._features = preprocessor.apply(ds._features)

    model = globals()[model_class](dataset=ds)
    model.load()

    Y_test_pred = model.predict()
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
