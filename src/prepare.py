import argparse
from pathlib import Path

from src.dataset import Dataset
from src.logger import logger
from src.preprocessors import preprocessor_factory

if __name__ == "__main__":

    logger.info("I am preparing the data !")

    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="SKCountVectorizer",
        help="A preprocessor's name. Must be a sub-class of Preprocessor",
    )
    args = parser.parse_args()

    # Training data
    ds = Dataset()
    ds.load_raw_to_df(raw_file="data/raw/train.csv")

    preprocessor = preprocessor_factory.get_preprocessor(args.preprocessor)
    ds.prepare_features(preprocessor)
    ds.train_test_split(save_path=Path("data/prepared") / args.preprocessor)

    preprocessor.fit(ds._features)
    preprocessor.save()

    logger.info("Done.")
