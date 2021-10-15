from pathlib import Path

import joblib
import tensorflow as tf
import yaml
from sklearn.linear_model import LogisticRegression

from src.logger import logger


class CustomModel:
    def __init__(self, train=True, preprocessor=None, features=[], labels=[]):
        self.model = None
        self.train = train
        self.preprocessor = preprocessor
        self.features = self.preprocessor.apply_preprocessor(features)
        self.labels = labels
        self.name = self.__class__.__name__
        self.get_params()
        self.make_dataset()

    def make_dataset(self):
        self.dataset = (self.features, self.labels)

    def make_model(self):
        self.model = LogisticRegression(max_iter=100)

    def fit(self):
        self.model.fit(*self.dataset)

    def predict(self):
        return self.model.predict(self.features)

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def get_params(self):
        with open("params.yaml", "r") as f:
            self.params = yaml.safe_load(f)[self.name]


class SKLogisticRegression(CustomModel):
    def __init__(self, train=True, preprocessor=None, features=[], labels=[]):
        super(SKLogisticRegression, self).__init__(
            train=train,
            preprocessor=preprocessor,
            features=features,
            labels=labels,
        )

    def make_model(self):
        self.model = LogisticRegression(max_iter=100)

    def save(self):
        joblib.dump(self.model, f"models/{self.name}/model.joblib")

    def load(self):
        self.model = joblib.load(f"models/{self.name}/model.joblib")


class TFConv1D(CustomModel):
    def __init__(self, train=True, preprocessor=None, features=[], labels=[]):
        super(TFConv1D, self).__init__(
            train=train,
            preprocessor=preprocessor,
            features=features,
            labels=labels,
        )

    def make_model(self):
        vocab_size = self.preprocessor.vocab_size
        self.model = tf.keras.Sequential(
            [
                # Layer Input Word Embedding
                tf.keras.layers.Embedding(
                    vocab_size + 1,
                    output_dim=512,
                    input_shape=[
                        self.input_shape,
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

    def make_dataset(self):
        self.input_shape = self.features.shape[1]

        buffer_size = 100000
        self.dataset = tf.data.Dataset.from_tensor_slices((self.features, self.labels))
        self.dataset = self.dataset.shuffle(buffer_size)
        self.dataset = self.dataset.batch(self.params["batch_size"])

    # TODO: validation data via a dataset class
    def fit(self, validation_data=None):
        self.model.fit(self.dataset, epochs=self.params["epochs"])

    def save(self):
        self.model.save(f"models/{self.name}/model.h5")

    def load(self):
        self.model = tf.keras.models.load_model(f"models/{self.name}/model.h5")
