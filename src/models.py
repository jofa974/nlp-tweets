from pathlib import Path

import joblib
import tensorflow as tf
import yaml
from sklearn.linear_model import LogisticRegression

from src.logger import logger


class CustomModel:
    def __init__(self, dataset=None):
        self.model = None
        self.dataset = dataset
        self.name = self.__class__.__name__
        self.get_params()

    def make_model(self, vocab_size=0):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def predict(self):
        return self.model.predict(self.dataset._features)

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def get_params(self):
        with open("params.yaml", "r") as f:
            self.params = yaml.safe_load(f)[self.name]


class SKLogisticRegression(CustomModel):
    def __init__(self, train=True, dataset=None):
        super(SKLogisticRegression, self).__init__(
            dataset=dataset,
        )

    def make_model(self, vocab_size=0):
        self.model = LogisticRegression(max_iter=100)

    def fit(self):
        self.model.fit(self.dataset._features, self.dataset._labels)

    def save(self):
        joblib.dump(self.model, f"models/{self.name}/model.joblib")

    def load(self):
        self.model = joblib.load(f"models/{self.name}/model.joblib")


class TFConv1D(CustomModel):
    def __init__(self, train=True, dataset=None):
        super(TFConv1D, self).__init__(
            dataset=dataset,
        )

    def make_model(self, vocab_size=0):
        input_shape = self.dataset.input_shape
        self.model = tf.keras.Sequential(
            [
                # Layer Input Word Embedding
                tf.keras.layers.Embedding(
                    vocab_size + 1,
                    output_dim=512,
                    input_shape=[
                        input_shape,
                    ],
                ),
                tf.keras.layers.Conv1D(128, 3, activation="relu"),
                # Flatten
                tf.keras.layers.Flatten(),
                # Layer Dense classique
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                # Output layer with number of output neurons equal to class number with softmax function
                tf.keras.layers.Dense(1, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.params["lr"]),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )

    # TODO: validation data via a dataset class
    def fit(self, validation_data=None):
        batched_data = self.dataset.make_tf_batched_data(self.params["batch_size"])
        self.model.fit(batched_data, epochs=self.params["epochs"])

    def save(self):
        self.model.save(f"models/{self.name}/model.h5")

    def load(self):
        self.model = tf.keras.models.load_model(f"models/{self.name}/model.h5")
