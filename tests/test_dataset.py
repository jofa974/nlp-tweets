from pathlib import Path

from src.dataset import Dataset
from src.preprocessors import constructors

ds = Dataset()


class TestDataset:
    def test_load_raw_to_df(self):
        pass


def test_prepare_features():
    ds = Dataset()
    ds.load_raw_to_df(raw_file=Path(__file__).parent.resolve() / "train_sample.csv")

    preprocessor = constructors["SKCountVectorizer"]()

    ds.prepare_features(preprocessor)

    assert ds._features.iloc[-1] == "my car fast"
