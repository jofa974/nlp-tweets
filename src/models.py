from pathlib import Path

import joblib
import tensorflow as tf
import yaml
from sklearn.linear_model import LogisticRegression


class CustomModel:
    def __init__(self, dataset=None):
        self.model = None
        self.dataset = dataset
        self.name = self.__class__.__name__
        self.get_params()

    def make_model(self, vocab_size=0):
        raise NotImplementedError

    def fit(self, validation_data=None):
        raise NotImplementedError

    def predict(self):
        predictions = self.model.predict(self.dataset._features)
        threshold = 0.5
        return list(map((lambda x: 1 if x > threshold else 0), predictions))

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

    def fit(self, validation_data=None):
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
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    64, activation="relu", input_shape=(input_shape,)
                ),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.params["lr"]),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )

    def fit(self, validation_data=None):
        batched_data = self.dataset.make_tf_batched_data(self.params["batch_size"])
        if validation_data:
            batched_val = validation_data.make_tf_batched_data(
                self.params["batch_size"]
            )
        else:
            batched_val = None
        self.model.fit(
            batched_data, epochs=self.params["epochs"], validation_data=batched_val
        )

    def save(self):
        self.model.save(f"models/{self.name}/model.h5")

    def load(self):
        self.model = tf.keras.models.load_model(f"models/{self.name}/model.h5")
