import json

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.logger import logger


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

    def train_test_split(self, out_path):
        X_train, X_test, Y_train, Y_test = train_test_split(
            self._features,
            self._labels,
            test_size=0.2,
            random_state=42,
        )
        logger.info("Saving texts...")
        texts = {
            "train": X_train.tolist(),
            # "val": X_val.tolist(),
            "test": X_test.tolist(),
        }
        with open(out_path / "texts.json", "w") as f:
            json.dump(texts, f)

        logger.info("Saving labels...")
        labels = {
            "train": Y_train.tolist(),
            # "val": Y_val.tolist(),
            "test": Y_test.tolist(),
        }
        with open(out_path / "labels.json", "w") as f:
            json.dump(labels, f)

    def load_features(self, dir_path, stage="train"):
        # out_path = Path(f"data/prepared/{preprocessor_class}")
        with open(dir_path / "texts.json", "r") as f:
            self._features = json.load(f)[stage]

    def load_labels(self, dir_path, stage="train"):
        with open(dir_path / "labels.json", "r") as f:
            self._labels = json.load(f)[stage]

    @property
    def input_shape(self):
        return self._features.shape[1]

    def make_tf_batched_data(self, batch_size):
        buffer_size = 100000
        batched_data = tf.data.Dataset.from_tensor_slices(
            (self._features, self._labels)
        )
        batched_data = batched_data.shuffle(buffer_size)
        batched_data = batched_data.batch(batch_size)
        return batched_data
