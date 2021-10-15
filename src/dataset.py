import pandas as pd
from tqdm import tqdm


class Dataset:
    def __init__(self):
        self._features = None
        self._labels = None
        self._df = None
        self._preprocessor = None

    def load_raw_to_df(self, raw_file="data/raw/train.csv"):
        df = pd.read_csv(str(raw_file))
        self._features = df["text"]
        self._labels = df["target"]

    def prepare_features(self, preprocessor):
        tqdm.pandas()

        self._features = self._features.progress_apply(preprocessor.remove_url)
        self._features = self._features.progress_apply(preprocessor.lemmatize)
