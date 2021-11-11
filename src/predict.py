import argparse
from pathlib import Path

import pandas as pd

from src.dataset import Dataset
from src.logger import logger
from src.models import model_factory
from src.preprocessors import preprocessor_factory


def predict(model_class, preprocessor_class):

    preprocessor = preprocessor_factory.get_preprocessor(preprocessor_class)
    preprocessor.load()

    # Test data
    ds_test = Dataset()
    ds_test.load_raw_to_df(raw_file="data/raw/test.csv")
    ds_test.prepare_features(preprocessor)
    ds_test._features = preprocessor.apply(ds_test._features)

    model = model_factory.get_model(model_class)
    model.load()

    predictions = model.predict_class(dataset=ds_test, threshold=0.5)
    df = pd.DataFrame({"id": ds_test._id, "target": predictions})
    df.to_csv(f"models/{model_class}/submission.csv", index=False)


if __name__ == "__main__":
    print("I am making prediction !")

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
    predict(args.model_class, args.preprocessor)
